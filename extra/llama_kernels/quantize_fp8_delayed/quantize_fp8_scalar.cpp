#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

// Pure one-pass bf16 -> fp8 quantize with delayed scalar scale. No amax computation.

#ifndef N_ELEMS
#define N_ELEMS 67108864
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

extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void
quantize_fp8_scalar(
    __hip_fp8_storage_t*  __restrict__ fp8_out,         // fp8, N_ELEMS
    const __hip_bfloat16* __restrict__ x,               // bf16, N_ELEMS
    const float*          __restrict__ amax_state)      // fp32 scalar (delayed)
{
  const int tid = threadIdx.x;
  const int wg  = blockIdx.x;
  const int gid = wg * THREADS_PER_WG + tid;
  const int stride_elems = NUM_WG * THREADS_PER_WG * VEC;

  const float scale = FP8_MAX / (static_cast<float>(*amax_state) + 1e-8f);

  for (int base = gid * VEC; base < N_ELEMS; base += stride_elems) {
    float4 x_raw = *reinterpret_cast<const float4*>(&x[base]);
    const __hip_bfloat16 *xi = reinterpret_cast<const __hip_bfloat16*>(&x_raw);

    __hip_fp8_storage_t out[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; i++) {
      const float v = static_cast<float>(xi[i]);
      const float scaled = fmaxf(-FP8_MAX, fminf(FP8_MAX, v * scale));
      out[i] = __hip_cvt_float_to_fp8(scaled, __HIP_SATFINITE, __HIP_E4M3);
    }
    *reinterpret_cast<uint64_t*>(&fp8_out[base]) = *reinterpret_cast<uint64_t*>(out);
  }
}
