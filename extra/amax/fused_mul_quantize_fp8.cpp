#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#ifndef N_ELEMS
#define N_ELEMS 67108864
#endif
#ifndef HIDDEN
#define HIDDEN 4096
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
fused_mul_quantize_fp8(
    __hip_fp8_storage_t*  __restrict__ fp8_out,         // fp8, N_ELEMS
    __hip_bfloat16*       __restrict__ amax_buf,        // bf16, NUM_WG
    const __hip_bfloat16* __restrict__ x,               // bf16, N_ELEMS
    const __hip_bfloat16* __restrict__ weight,          // bf16, HIDDEN (per-hidden scale)
    const __hip_bfloat16* __restrict__ amax_state)      // bf16 scalar
{
  __shared__ float sdata[THREADS_PER_WG];

  const int tid = threadIdx.x;
  const int wg  = blockIdx.x;
  const int gid = wg * THREADS_PER_WG + tid;
  const int stride_elems = NUM_WG * THREADS_PER_WG * VEC;

  const float scale = FP8_MAX / (static_cast<float>(*amax_state) + 1e-8f);
  float local_max = 0.0f;

  for (int base = gid * VEC; base < N_ELEMS; base += stride_elems) {
    const int h = base % HIDDEN;   // 0..HIDDEN-VEC, 8-aligned (since base is 8-aligned and HIDDEN divides VEC)
    float4 x_raw = *reinterpret_cast<const float4*>(&x[base]);
    float4 w_raw = *reinterpret_cast<const float4*>(&weight[h]);

    const __hip_bfloat16 *xi = reinterpret_cast<const __hip_bfloat16*>(&x_raw);
    const __hip_bfloat16 *wi = reinterpret_cast<const __hip_bfloat16*>(&w_raw);

    __hip_fp8_storage_t out[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; i++) {
      const float val = static_cast<float>(xi[i]) * static_cast<float>(wi[i]);
      local_max = fmaxf(local_max, fabsf(val));
      const float scaled = fmaxf(-FP8_MAX, fminf(FP8_MAX, val * scale));
      out[i] = __hip_cvt_float_to_fp8(scaled, __HIP_SATFINITE, __HIP_E4M3);
    }

    *reinterpret_cast<uint64_t*>(&fp8_out[base]) = *reinterpret_cast<uint64_t*>(out);
  }

  // LDS tree-reduce per-WG amax
  sdata[tid] = local_max;
  __syncthreads();
  for (int s = THREADS_PER_WG / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }

  if (tid == 0) amax_buf[wg] = static_cast<__hip_bfloat16>(sdata[0]);
}
