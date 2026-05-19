#include <metal_stdlib>
using namespace metal;

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

#define VEC 8

kernel void
fused_ce_loss_fwd_metal(
    device float*        __restrict__ loss_out,  // out: fp32, ROWS
    device float*        __restrict__ max_out,   // out: fp32, ROWS
    device float*        __restrict__ lse_out,   // out: fp32, ROWS
    const device bfloat* __restrict__ logits,    // in:  bf16, ROWS*VOCAB
    const device int*    __restrict__ targets,   // in:  int32, ROWS
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
  threadgroup float sdata_m[THREADS_PER_WG];
  threadgroup float sdata_s[THREADS_PER_WG];
  threadgroup float sdata_sumx[THREADS_PER_WG];
  threadgroup float sdata_tgt[THREADS_PER_WG];

  const int tid = lid.x;
  const int row = gid.x;
  const int target = targets[row];
  const device bfloat* row_logits = logits + (size_t)row * VOCAB;

  float m = -3.4028234663852886e38f;
  float s = 0.0f;
  float sum_x = 0.0f;
  float target_logit = 0.0f;
  constexpr bool needs_sum_x = (LABEL_SMOOTHING != 0.0f);

  // Vectorized stride: each iter loads 8 bf16 = 16 bytes. Warp loads 32*16 = 512 bytes (4 cache lines).
  const int VOCAB_VEC = VOCAB & ~(VEC - 1);  // round down to multiple of VEC
  for (int i = tid * VEC; i < VOCAB_VEC; i += THREADS_PER_WG * VEC) {
    bfloat4 raw0 = *reinterpret_cast<const device bfloat4*>(&row_logits[i]);
    bfloat4 raw1 = *reinterpret_cast<const device bfloat4*>(&row_logits[i + 4]);
    float4 x0 = float4(raw0);
    float4 x1 = float4(raw1);
    #pragma unroll
    for (int k = 0; k < VEC; k++) {
      const float x = k < 4 ? x0[k] : x1[k - 4];
      if constexpr (needs_sum_x) sum_x += x;
      if (i + k == target) target_logit = x;
      if (x > m) {
        s = s * exp(m - x) + 1.0f;
        m = x;
      } else {
        s += exp(x - m);
      }
    }
  }
  // tail (VOCAB not divisible by VEC):
  for (int i = VOCAB_VEC + tid; i < VOCAB; i += THREADS_PER_WG) {
    const float x = static_cast<float>(row_logits[i]);
    if constexpr (needs_sum_x) sum_x += x;
    if (i == target) target_logit = x;
    if (x > m) { s = s * exp(m - x) + 1.0f; m = x; }
    else       { s += exp(x - m); }
  }

  sdata_m[tid] = m;
  sdata_s[tid] = s;
  sdata_sumx[tid] = sum_x;
  sdata_tgt[tid] = target_logit;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int step = THREADS_PER_WG / 2; step > 0; step >>= 1) {
    if (tid < step) {
      const float m1 = sdata_m[tid];
      const float m2 = sdata_m[tid + step];
      const float s1 = sdata_s[tid];
      const float s2 = sdata_s[tid + step];
      const float m_new = fmax(m1, m2);
      const float s_new = s1 * exp(m1 - m_new) + s2 * exp(m2 - m_new);
      sdata_m[tid] = m_new;
      sdata_s[tid] = s_new;
      sdata_sumx[tid] += sdata_sumx[tid + step];
      sdata_tgt[tid]  += sdata_tgt[tid + step];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) {
    const float row_max = sdata_m[0];
    const float row_sum_exp = sdata_s[0];
    const float row_sum_x = sdata_sumx[0];
    const float tgt = sdata_tgt[0];
    const float row_lse = log(row_sum_exp) + row_max;
    const float mean_logits = row_sum_x / static_cast<float>(VOCAB);
    const float loss = row_lse - (1.0f - LABEL_SMOOTHING) * tgt - LABEL_SMOOTHING * mean_logits;
    loss_out[row] = loss;
    max_out[row]  = row_max;
    lse_out[row]  = row_lse;
  }
}
