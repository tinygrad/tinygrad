#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// Fused forward sparse-CE with label smoothing.
// SINGLE-PASS online softmax + vectorized 8-wide bf16 loads for HBM coalescing.

#ifndef VOCAB
#define VOCAB 128256
#endif
#ifndef THREADS_PER_WG
#define THREADS_PER_WG 256
#endif
#ifndef LABEL_SMOOTHING
#define LABEL_SMOOTHING 0.1f
#endif

constexpr int VEC = 8;

extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void
fused_ce_loss_fwd(
    float*                __restrict__ loss_out,  // out: fp32, ROWS
    float*                __restrict__ max_out,   // out: fp32, ROWS
    float*                __restrict__ lse_out,   // out: fp32, ROWS
    const __hip_bfloat16* __restrict__ logits,    // in:  bf16, ROWS*VOCAB
    const int*            __restrict__ targets)   // in:  int32, ROWS
{
  __shared__ float sdata_m[THREADS_PER_WG];
  __shared__ float sdata_s[THREADS_PER_WG];
  __shared__ float sdata_sumx[THREADS_PER_WG];
  __shared__ float sdata_tgt[THREADS_PER_WG];

  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const int target = targets[row];
  const __hip_bfloat16* row_logits = logits + (size_t)row * VOCAB;

  float m = -INFINITY;
  float s = 0.0f;
  float sum_x = 0.0f;
  float target_logit = 0.0f;
  constexpr bool needs_sum_x = (LABEL_SMOOTHING != 0.0f);

  // Vectorized stride: each iter loads 8 bf16 = 16 bytes. Warp loads 32*16 = 512 bytes (4 cache lines).
  const int VOCAB_VEC = VOCAB & ~(VEC - 1);  // round down to multiple of VEC
  for (int i = tid * VEC; i < VOCAB_VEC; i += THREADS_PER_WG * VEC) {
    float4 raw = *reinterpret_cast<const float4*>(&row_logits[i]);
    const __hip_bfloat16* xi = reinterpret_cast<const __hip_bfloat16*>(&raw);
    #pragma unroll
    for (int k = 0; k < VEC; k++) {
      const float x = static_cast<float>(xi[k]);
      if constexpr (needs_sum_x) sum_x += x;
      if (i + k == target) target_logit = x;
      if (x > m) {
        s = s * __expf(m - x) + 1.0f;
        m = x;
      } else {
        s += __expf(x - m);
      }
    }
  }
  // tail (VOCAB not divisible by VEC):
  for (int i = VOCAB_VEC + tid; i < VOCAB; i += THREADS_PER_WG) {
    const float x = static_cast<float>(row_logits[i]);
    if constexpr (needs_sum_x) sum_x += x;
    if (i == target) target_logit = x;
    if (x > m) { s = s * __expf(m - x) + 1.0f; m = x; }
    else        { s += __expf(x - m); }
  }

  sdata_m[tid] = m;
  sdata_s[tid] = s;
  sdata_sumx[tid] = sum_x;
  sdata_tgt[tid] = target_logit;
  __syncthreads();

  for (int step = THREADS_PER_WG / 2; step > 0; step >>= 1) {
    if (tid < step) {
      const float m1 = sdata_m[tid];
      const float m2 = sdata_m[tid + step];
      const float s1 = sdata_s[tid];
      const float s2 = sdata_s[tid + step];
      const float m_new = fmaxf(m1, m2);
      const float s_new = s1 * __expf(m1 - m_new) + s2 * __expf(m2 - m_new);
      sdata_m[tid] = m_new;
      sdata_s[tid] = s_new;
      sdata_sumx[tid] += sdata_sumx[tid + step];
      sdata_tgt[tid]  += sdata_tgt[tid + step];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float row_max = sdata_m[0];
    const float row_sum_exp = sdata_s[0];
    const float row_sum_x = sdata_sumx[0];
    const float tgt = sdata_tgt[0];
    const float row_lse = logf(row_sum_exp) + row_max;
    const float mean_logits = row_sum_x / static_cast<float>(VOCAB);
    const float loss = row_lse - (1.0f - LABEL_SMOOTHING) * tgt - LABEL_SMOOTHING * mean_logits;
    loss_out[row] = loss;
    max_out[row]  = row_max;
    lse_out[row]  = row_lse;
  }
}
