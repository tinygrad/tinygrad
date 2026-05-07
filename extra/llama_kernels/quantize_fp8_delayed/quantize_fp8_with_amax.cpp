#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

// One-pass bf16 -> fp8 quantize using a scalar delayed amax state,
// AND simultaneously computes per-WG |x| max partials for the next step's amax state.
// Saves one full HBM pass over the grad tensor vs. doing quantize + separate abs().max().

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
quantize_fp8_with_amax(
    __hip_fp8_storage_t*  __restrict__ fp8_out,       // out: fp8, N_ELEMS
    float*                __restrict__ amax_partial,  // out: fp32, NUM_WG per-WG partials
    const __hip_bfloat16* __restrict__ x,             // in:  bf16, N_ELEMS
    const float*          __restrict__ amax_state)    // in:  fp32 scalar (delayed)
{
  __shared__ float sdata[THREADS_PER_WG];

  const int tid = threadIdx.x;
  const int wg  = blockIdx.x;
  const int gid = wg * THREADS_PER_WG + tid;
  const int stride_elems = NUM_WG * THREADS_PER_WG * VEC;

  const float scale = FP8_MAX / (static_cast<float>(*amax_state) + 1e-8f);
  float local_max = 0.0f;

  for (int base = gid * VEC; base < N_ELEMS; base += stride_elems) {
    float4 x_raw = *reinterpret_cast<const float4*>(&x[base]);
    const __hip_bfloat16 *xi = reinterpret_cast<const __hip_bfloat16*>(&x_raw);

    __hip_fp8_storage_t out[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; i++) {
      const float v = static_cast<float>(xi[i]);
      local_max = fmaxf(local_max, fabsf(v));
      const float scaled = fmaxf(-FP8_MAX, fminf(FP8_MAX, v * scale));
      out[i] = __hip_cvt_float_to_fp8(scaled, __HIP_SATFINITE, __HIP_E4M3);
    }
    *reinterpret_cast<uint64_t*>(&fp8_out[base]) = *reinterpret_cast<uint64_t*>(out);
  }

  sdata[tid] = local_max;
  __syncthreads();
  for (int s = THREADS_PER_WG / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  if (tid == 0) amax_partial[wg] = sdata[0];
}
