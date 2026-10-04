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

static_assert(TOPK <= EXPERTS && TOPK <= 8, "unsupported TOPK");

extern "C" __global__ __launch_bounds__(THREADS) void moe_router_topk(
    float *__restrict__ weights, int *__restrict__ indices, const float *__restrict__ logits) {
  const int token = blockIdx.x * THREADS + threadIdx.x;
  if (token >= TOKENS) return;
  float topv[TOPK];
  int topi[TOPK];
  #pragma unroll
  for (int i = 0; i < TOPK; i++) { topv[i] = -3.402823466e+38f; topi[i] = -1; }

  const long long base = (long long)token * EXPERTS;
  #pragma unroll
  for (int e = 0; e < EXPERTS; e++) {
    float v = logits[base + e];
    int pos = TOPK;
    #pragma unroll
    for (int i = 0; i < TOPK; i++) if (pos == TOPK && v > topv[i]) pos = i;
    if (pos < TOPK) {
      #pragma unroll
      for (int i = TOPK - 1; i > 0; i--) if (i > pos) { topv[i] = topv[i-1]; topi[i] = topi[i-1]; }
      topv[pos] = v;
      topi[pos] = e;
    }
  }

  float denom = 0.0f;
  float ex[TOPK];
  #pragma unroll
  for (int i = 0; i < TOPK; i++) { ex[i] = expf(topv[i] - topv[0]); denom += ex[i]; }
  const long long out = (long long)token * TOPK;
  #pragma unroll
  for (int i = 0; i < TOPK; i++) {
    weights[out + i] = ex[i] / denom;
    indices[out + i] = topi[i];
  }
}
