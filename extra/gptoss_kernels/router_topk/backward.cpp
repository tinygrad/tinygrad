#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(256) void moe_router_topk_bwd(
    float *__restrict__ grad_logits, const float *__restrict__ grad_weights,
    const float *__restrict__ weights, const int *__restrict__ indices) {
  const int token = blockIdx.x * 256 + threadIdx.x;
  if (token >= TOKENS) return;
  const long long in = (long long)token * 4;
  float dot = 0.0f;
  #pragma unroll
  for (int i = 0; i < 4; i++) dot += grad_weights[in + i] * weights[in + i];
  const long long out = (long long)token * 32;
  #pragma unroll
  for (int e = 0; e < 32; e++) grad_logits[out + e] = 0.0f;
  #pragma unroll
  for (int i = 0; i < 4; i++) grad_logits[out + indices[in + i]] = weights[in + i] * (grad_weights[in + i] - dot);
}
