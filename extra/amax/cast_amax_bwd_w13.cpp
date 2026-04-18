#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#ifndef N_ELEMS
#define N_ELEMS 234881024
#endif
#ifndef HIDDEN
#define HIDDEN 14336
#endif
#ifndef NUM_WG
#define NUM_WG 1024
#endif
#ifndef THREADS_PER_WG
#define THREADS_PER_WG 256
#endif

constexpr int VEC = 8;
constexpr float FP8_MAX = 448.0f;

static_assert(N_ELEMS % VEC == 0, "N_ELEMS must be divisible by VEC");
static_assert(HIDDEN % VEC == 0, "HIDDEN must be divisible by VEC");

extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void
fused_silu_mul_bwd_w13(
    __hip_bfloat16*       __restrict__ grad_xw13_out,    // bf16, 2*N_ELEMS (interleaved layout)
    const __hip_bfloat16* __restrict__ xw13,             // bf16, 2*N_ELEMS (interleaved)
    const __hip_bfloat16* __restrict__ grad_x2,          // bf16, N_ELEMS
    const __hip_bfloat16* __restrict__ amax_state)       // bf16 scalar
{
  const int tid = threadIdx.x;
  const int wg  = blockIdx.x;
  const int gid = wg * THREADS_PER_WG + tid;
  const int stride_elems = NUM_WG * THREADS_PER_WG * VEC;

  const float scale = FP8_MAX / (static_cast<float>(*amax_state) + 1e-8f);

  for (int base = gid * VEC; base < N_ELEMS; base += stride_elems) {
    const int outer = base / HIDDEN;
    const int inner = base % HIDDEN;
    const int xw1_off = outer * 2 * HIDDEN + inner;
    const int xw3_off = xw1_off + HIDDEN;

    float4 x1_raw = *reinterpret_cast<const float4*>(&xw13[xw1_off]);
    float4 x3_raw = *reinterpret_cast<const float4*>(&xw13[xw3_off]);
    float4 g_raw  = *reinterpret_cast<const float4*>(&grad_x2[base]);

    const __hip_bfloat16 *x1 = reinterpret_cast<const __hip_bfloat16*>(&x1_raw);
    const __hip_bfloat16 *x3 = reinterpret_cast<const __hip_bfloat16*>(&x3_raw);
    const __hip_bfloat16 *gv = reinterpret_cast<const __hip_bfloat16*>(&g_raw);

    __hip_bfloat16 out1[VEC], out3[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; i++) {
      const float f1 = static_cast<float>(x1[i]);
      const float f3 = static_cast<float>(x3[i]);
      const float fg = static_cast<float>(gv[i]);
      const float sig = 1.0f / (1.0f + __expf(-f1));
      const float silu = f1 * sig;
      const float silu_prime = sig + silu * (1.0f - sig);
      const float gs = fg * scale;
      out1[i] = static_cast<__hip_bfloat16>(gs * silu_prime * f3);
      out3[i] = static_cast<__hip_bfloat16>(gs * silu);
    }

    *reinterpret_cast<float4*>(&grad_xw13_out[xw1_off]) = *reinterpret_cast<float4*>(out1);
    *reinterpret_cast<float4*>(&grad_xw13_out[xw3_off]) = *reinterpret_cast<float4*>(out3);
  }
}
