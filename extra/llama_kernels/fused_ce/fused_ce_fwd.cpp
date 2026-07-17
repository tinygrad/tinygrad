#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#ifndef ROWS
#define ROWS 16384
#endif
#ifndef VOCAB
#define VOCAB 128256
#endif
#ifndef THREADS
#define THREADS 1024
#endif

constexpr int WAVE_SIZE = 64;
constexpr int VEC = 8;
static_assert(THREADS % WAVE_SIZE == 0);
static_assert(THREADS <= 1024);
static_assert(VOCAB % VEC == 0);

__device__ __forceinline__ float wave_max(float x) {
  #pragma unroll
  for (int offset = WAVE_SIZE / 2; offset; offset >>= 1) x = fmaxf(x, __shfl_down(x, offset));
  return x;
}

__device__ __forceinline__ float wave_sum(float x) {
  #pragma unroll
  for (int offset = WAVE_SIZE / 2; offset; offset >>= 1) x += __shfl_down(x, offset);
  return x;
}

extern "C" __global__ __launch_bounds__(THREADS) void fused_ce_loss_fwd(
    float* __restrict__ loss_out,
    float* __restrict__ max_out,
    float* __restrict__ lse_out,
    const __hip_bfloat16* __restrict__ logits,
    const int* __restrict__ targets) {
  __shared__ float wave_values[THREADS / WAVE_SIZE];
  const int tid = threadIdx.x;
  const size_t row = blockIdx.x;
  const __hip_bfloat16* row_logits = logits + row * VOCAB;

  float local_max = -3.402823466e+38F;
  for (int v = tid * VEC; v < VOCAB; v += THREADS * VEC) {
    const float4 packed = *reinterpret_cast<const float4*>(row_logits + v);
    const __hip_bfloat16* values = reinterpret_cast<const __hip_bfloat16*>(&packed);
    #pragma unroll
    for (int i = 0; i < VEC; i++) local_max = fmaxf(local_max, static_cast<float>(values[i]));
  }
  local_max = wave_max(local_max);
  if ((tid & (WAVE_SIZE - 1)) == 0) wave_values[tid / WAVE_SIZE] = local_max;
  __syncthreads();
  local_max = tid < THREADS / WAVE_SIZE ? wave_values[tid] : -3.402823466e+38F;
  if (tid < WAVE_SIZE) local_max = wave_max(local_max);
  if (tid == 0) wave_values[0] = local_max;
  __syncthreads();
  const float row_max = wave_values[0];

  float local_sum = 0.0f;
  for (int v = tid * VEC; v < VOCAB; v += THREADS * VEC) {
    const float4 packed = *reinterpret_cast<const float4*>(row_logits + v);
    const __hip_bfloat16* values = reinterpret_cast<const __hip_bfloat16*>(&packed);
    #pragma unroll
    for (int i = 0; i < VEC; i++) local_sum += __builtin_amdgcn_exp2f((static_cast<float>(values[i]) - row_max) * 1.4426950408889634f);
  }
  local_sum = wave_sum(local_sum);
  if ((tid & (WAVE_SIZE - 1)) == 0) wave_values[tid / WAVE_SIZE] = local_sum;
  __syncthreads();
  local_sum = tid < THREADS / WAVE_SIZE ? wave_values[tid] : 0.0f;
  if (tid < WAVE_SIZE) local_sum = wave_sum(local_sum);

  if (tid == 0) {
    const float lse = logf(local_sum) + row_max;
    loss_out[row] = lse - static_cast<float>(row_logits[targets[row]]);
    max_out[row] = row_max;
    lse_out[row] = lse;
  }
}
