#include <hip/hip_runtime.h>

#ifndef TOKENS
#define TOKENS 16384
#endif
#ifndef EXPERTS
#define EXPERTS 32
#endif
#ifndef TOPK
#define TOPK 4
#endif
#ifndef THREADS
#define THREADS 256
#endif
#ifndef FUSED_BIAS_GRAD
#define FUSED_BIAS_GRAD 0
#endif

extern "C" __global__ __launch_bounds__(THREADS) void moe_router_topk_bwd(
    float *__restrict__ grad_logits, float *__restrict__ grad_bias_partials, const float *__restrict__ grad_weights,
    const float *__restrict__ weights, const int *__restrict__ indices) {
  #if FUSED_BIAS_GRAD
  constexpr int WARPS = THREADS / 64;
  __shared__ float partial[WARPS][EXPERTS];
  const int warp = threadIdx.x / 64;
  if (threadIdx.x < WARPS * EXPERTS) partial[threadIdx.x / EXPERTS][threadIdx.x % EXPERTS] = 0.0f;
  __syncthreads();
  #endif
  const int token = blockIdx.x * THREADS + threadIdx.x;
  if (token < TOKENS) {
    const long long in = (long long)token * TOPK;
    float dot = 0.0f;
    #pragma unroll
    for (int i = 0; i < TOPK; i++) dot += grad_weights[in + i] * weights[in + i];
    const long long out = (long long)token * EXPERTS;
    #pragma unroll
    for (int e = 0; e < EXPERTS; e++) grad_logits[out + e] = 0.0f;
    #pragma unroll
    for (int i = 0; i < TOPK; i++) {
      const int e = indices[in + i];
      const float dg = weights[in + i] * (grad_weights[in + i] - dot);
      grad_logits[out + e] = dg;
      #if FUSED_BIAS_GRAD
      atomicAdd(&partial[warp][e], dg);
      #endif
    }
  }
  #if FUSED_BIAS_GRAD
  __syncthreads();
  if (threadIdx.x < EXPERTS) {
    float sum = 0.0f;
    #pragma unroll
    for (int w = 0; w < WARPS; w++) sum += partial[w][threadIdx.x];
    grad_bias_partials[(long long)blockIdx.x * EXPERTS + threadIdx.x] = sum;
  }
  #endif
}
