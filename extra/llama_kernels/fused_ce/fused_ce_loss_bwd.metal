#include <metal_stdlib>
using namespace metal;

// Vectorized CE bwd: 8-wide bf16 loads + stores.

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
fused_ce_loss_bwd_metal(
    device bfloat*       __restrict__ d_logits,
    const device bfloat* __restrict__ logits,
    const device float*  __restrict__ lse,
    const device int*    __restrict__ targets,
    const device float*  __restrict__ scale_in,
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
  const int tid = lid.x;
  const int row = gid.x;
  const int target = targets[row];
  const float lse_r = lse[row];
  const device bfloat* row_logits  = logits   + (size_t)row * VOCAB;
  device bfloat*       row_dlogits = d_logits + (size_t)row * VOCAB;
  const float inv_vocab = 1.0f / static_cast<float>(VOCAB);
  const float scale = *scale_in;
  const float ls_term = LABEL_SMOOTHING * inv_vocab;

  const int VOCAB_VEC = VOCAB & ~(VEC - 1);
  for (int i = tid * VEC; i < VOCAB_VEC; i += THREADS_PER_WG * VEC) {
    bfloat4 raw0 = *reinterpret_cast<const device bfloat4*>(&row_logits[i]);
    bfloat4 raw1 = *reinterpret_cast<const device bfloat4*>(&row_logits[i + 4]);
    float4 x0 = float4(raw0);
    float4 x1 = float4(raw1);
    bfloat4 out0;
    bfloat4 out1;
    #pragma unroll
    for (int k = 0; k < VEC; k++) {
      const float x = k < 4 ? x0[k] : x1[k - 4];
      float g = exp(x - lse_r);
      if (i + k == target) g -= (1.0f - LABEL_SMOOTHING);
      g -= ls_term;
      if (k < 4) out0[k] = bfloat(g * scale);
      else       out1[k - 4] = bfloat(g * scale);
    }
    *reinterpret_cast<device bfloat4*>(&row_dlogits[i]) = out0;
    *reinterpret_cast<device bfloat4*>(&row_dlogits[i + 4]) = out1;
  }
  for (int i = VOCAB_VEC + tid; i < VOCAB; i += THREADS_PER_WG) {
    const float x = static_cast<float>(row_logits[i]);
    float g = exp(x - lse_r);
    if (i == target) g -= (1.0f - LABEL_SMOOTHING);
    g -= ls_term;
    row_dlogits[i] = bfloat(g * scale);
  }
}
