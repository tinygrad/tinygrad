#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

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
#ifndef LAYER_SCALE
#define LAYER_SCALE 0
#endif

constexpr int VEC = 8;
constexpr float FP8_MAX = 448.0f;

static_assert(N_ELEMS % VEC == 0, "N_ELEMS must be divisible by VEC");
static_assert(HIDDEN % VEC == 0, "HIDDEN must be divisible by VEC");

// fused silu*mul backward, two outputs in a single HBM pass:
//   1) fp8  grad_xw13_fp8  — delayed-scale quantize using grad_amax_state (mailbox to matmul bwd)
//   2) fp32 grad_amax_next — scalar |grad_xw13| via global atomic max
// grad_amax_state is read for the fp8 scale. The store of new_grad_amax into grad_amax_state's
// buffer is built in Python as a separate effect and threaded into grad_a via .after(store).
extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void
fused_silu_mul_bwd_w13(
    __hip_fp8_storage_t*  __restrict__ grad_xw13_fp8_out,    // fp8,  2*N_ELEMS
    float*                __restrict__ grad_amax_next,       // fp32 scalar, initialized to 0 before launch
    const __hip_bfloat16* __restrict__ xw13,                 // bf16, 2*N_ELEMS
    const __hip_bfloat16* __restrict__ grad_x2,              // bf16, N_ELEMS
    const float*          __restrict__ amax_state,           // fp32 scalar (fwd x2 amax)
    const float*          __restrict__ grad_amax_state       // fp32 scalar or n_layers (delayed grad amax)
#if LAYER_SCALE
    , const int*        __restrict__ layer_num
#endif
)
{
  __shared__ float sdata[THREADS_PER_WG];

  const int tid = threadIdx.x;
  const int wg  = blockIdx.x;
  const int gid = wg * THREADS_PER_WG + tid;
  const int stride_elems = NUM_WG * THREADS_PER_WG * VEC;

#if LAYER_SCALE
  const int layer = layer_num[0];
#else
  const int layer = 0;
#endif
  float* grad_amax_next_layer = grad_amax_next + layer;
  const float scale = FP8_MAX / (static_cast<float>(amax_state[layer]) + 1e-8f);
  const float grad_amax = static_cast<float>(grad_amax_state[layer]);
  const float g_scale = FP8_MAX / (grad_amax + 1e-8f);
  float local_max = 0.0f;

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

    float g1v[VEC], g3v[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; i++) {
      const float f1 = static_cast<float>(x1[i]);
      const float f3 = static_cast<float>(x3[i]);
      const float fg = static_cast<float>(gv[i]);
      const float sig = 1.0f / (1.0f + __expf(-f1));
      const float silu = f1 * sig;
      const float silu_prime = sig + silu * (1.0f - sig);
      const float gs = fg * scale;
      const float g1 = gs * silu_prime * f3;
      const float g3 = gs * silu;
      local_max = fmaxf(local_max, fmaxf(fabsf(g1), fabsf(g3)));
      // ROCm 7.1's saturating float2 helper clamps an overflowing lane 1 with lane 0, so clamp each lane before packing.
      g1v[i] = fmaxf(-FP8_MAX, fminf(FP8_MAX, g1 * g_scale));
      g3v[i] = fmaxf(-FP8_MAX, fminf(FP8_MAX, g3 * g_scale));
    }

    union packed_fp8x8 { uint64_t u64; __hip_fp8x2_storage_t x2[4]; } fp8_1, fp8_3;
    #pragma unroll
    for (int i = 0; i < VEC; i += 2) {
      fp8_1.x2[i/2] = __hip_cvt_float2_to_fp8x2(make_float2(g1v[i], g1v[i+1]), __HIP_SATFINITE, __HIP_E4M3);
      fp8_3.x2[i/2] = __hip_cvt_float2_to_fp8x2(make_float2(g3v[i], g3v[i+1]), __HIP_SATFINITE, __HIP_E4M3);
    }
    *reinterpret_cast<uint64_t*>(&grad_xw13_fp8_out[xw1_off]) = fp8_1.u64;
    *reinterpret_cast<uint64_t*>(&grad_xw13_fp8_out[xw3_off]) = fp8_3.u64;
  }

  sdata[tid] = local_max;
  __syncthreads();
  for (int s = THREADS_PER_WG / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  if (tid == 0 && sdata[0] > *grad_amax_next_layer) {
    atomicMax(reinterpret_cast<int32_t*>(grad_amax_next_layer), __float_as_int(sdata[0]));
  }
}
