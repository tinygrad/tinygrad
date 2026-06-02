#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

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
#ifndef EPS_LITERAL
#define EPS_LITERAL 1e-5f
#endif

constexpr int VEC = 8;
static_assert(N_ELEMS % HIDDEN == 0, "N_ELEMS must be a multiple of HIDDEN");
static_assert(HIDDEN % (THREADS_PER_WG * VEC) == 0, "HIDDEN must be divisible by THREADS_PER_WG*VEC");

constexpr int ROWS = N_ELEMS / HIDDEN;
constexpr int ELEMS_PER_THREAD = HIDDEN / THREADS_PER_WG;
constexpr int VECS_PER_THREAD = ELEMS_PER_THREAD / VEC;

extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void rmsnorm_weighted_fwd(
    __hip_bfloat16*       __restrict__ out,
    __hip_bfloat16*       __restrict__ x_normed_out,
    float*                __restrict__ rrms_out,
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ weight) {
  __shared__ float sdata[THREADS_PER_WG];

  const int tid = threadIdx.x;
  const int wg = blockIdx.x;
  const float inv_hidden = 1.0f / static_cast<float>(HIDDEN);

  for (int row = wg; row < ROWS; row += NUM_WG) {
    const int row_off = row * HIDDEN;
    float regs[ELEMS_PER_THREAD];
    float sum_sq = 0.0f;

    #pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; v++) {
      const int h_base = tid * VEC + v * THREADS_PER_WG * VEC;
      float4 raw = *reinterpret_cast<const float4*>(&x[row_off + h_base]);
      const __hip_bfloat16 *xi = reinterpret_cast<const __hip_bfloat16*>(&raw);
      #pragma unroll
      for (int i = 0; i < VEC; i++) {
        const float f = static_cast<float>(xi[i]);
        regs[v * VEC + i] = f;
        sum_sq += f * f;
      }
    }

    sdata[tid] = sum_sq;
    __syncthreads();
    for (int s = THREADS_PER_WG / 2; s > 0; s >>= 1) {
      if (tid < s) sdata[tid] += sdata[tid + s];
      __syncthreads();
    }
    const float rrms = 1.0f / sqrtf(sdata[0] * inv_hidden + EPS_LITERAL);
    if (tid == 0) rrms_out[row] = rrms;

    #pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; v++) {
      const int h_base = tid * VEC + v * THREADS_PER_WG * VEC;
      float4 w_raw = *reinterpret_cast<const float4*>(&weight[h_base]);
      const __hip_bfloat16 *wi = reinterpret_cast<const __hip_bfloat16*>(&w_raw);
      __hip_bfloat16 yn[VEC];
      __hip_bfloat16 yo[VEC];
      #pragma unroll
      for (int i = 0; i < VEC; i++) {
        const float x_normed = regs[v * VEC + i] * rrms;
        yn[i] = static_cast<__hip_bfloat16>(x_normed);
        yo[i] = static_cast<__hip_bfloat16>(x_normed * static_cast<float>(wi[i]));
      }
      *reinterpret_cast<float4*>(&x_normed_out[row_off + h_base]) = *reinterpret_cast<float4*>(yn);
      *reinterpret_cast<float4*>(&out[row_off + h_base]) = *reinterpret_cast<float4*>(yo);
    }
    __syncthreads();
  }
}
