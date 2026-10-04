#include <hip/hip_runtime.h>
#include <math.h>
#ifndef GEMM_M
#define GEMM_M 16384
#endif
#ifndef GEMM_N
#define GEMM_N 128256
#endif
constexpr int THREADS = 256;
constexpr int TILES_N = GEMM_N / 256;
static_assert(GEMM_M % THREADS == 0 && GEMM_N % 256 == 0);
__device__ __forceinline__ float fast_exp(float x) {
#if MX_CE_EXACT_EXP
  return expf(x);
#else
  x = fmaxf(x, -80.0f);
  const int bits = (int)(x * 12102203.0f + 1065353216.0f);
  return __builtin_bit_cast(float, bits);
#endif
}

extern "C" __global__ __launch_bounds__(THREADS) void lmhead_ce_finalize_mxfp8(
    float *__restrict__ lse, float *__restrict__ loss_parts, const uint8_t *__restrict__ scratch) {
  const int row = blockIdx.x * THREADS + threadIdx.x;
  const float *pmax = reinterpret_cast<const float *>(scratch);
  const float *psum = pmax + (long long)GEMM_M * TILES_N;
  const float *ptarget = psum + (long long)GEMM_M * TILES_N;
  const long long base = (long long)row * TILES_N;
  float mx = -3.402823466e+38f;
  #pragma unroll 1
  for (int j = 0; j < TILES_N; j++) mx = fmaxf(mx, pmax[base + j]);
  float sm = 0.0f;
  #pragma unroll 1
  for (int j = 0; j < TILES_N; j++) sm += psum[base + j] * fast_exp(pmax[base + j] - mx);
  const float row_lse = mx + logf(sm);
  lse[row] = row_lse;
  __shared__ float losses[THREADS];
  losses[threadIdx.x] = (row_lse - ptarget[row]) * (1.0f / GEMM_M);
  __syncthreads();
  #pragma unroll
  for (int stride = THREADS / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride) losses[threadIdx.x] += losses[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) loss_parts[blockIdx.x] = losses[0];
}
