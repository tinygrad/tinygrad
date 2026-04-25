#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

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

constexpr int VEC = 8;

extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void
fused_ce_loss_bwd(
    __hip_bfloat16*       __restrict__ d_logits,
    const __hip_bfloat16* __restrict__ logits,
    const float*          __restrict__ lse,
    const int*            __restrict__ targets,
    const float*          __restrict__ scale_in)
{
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const int target = targets[row];
  const float lse_r = lse[row];
  const __hip_bfloat16* row_logits  = logits   + (size_t)row * VOCAB;
  __hip_bfloat16*       row_dlogits = d_logits + (size_t)row * VOCAB;
  const float inv_vocab = 1.0f / static_cast<float>(VOCAB);
  const float scale = *scale_in;
  const float ls_term = LABEL_SMOOTHING * inv_vocab;

  const int VOCAB_VEC = VOCAB & ~(VEC - 1);
  for (int i = tid * VEC; i < VOCAB_VEC; i += THREADS_PER_WG * VEC) {
    float4 raw = *reinterpret_cast<const float4*>(&row_logits[i]);
    const __hip_bfloat16* xi = reinterpret_cast<const __hip_bfloat16*>(&raw);
    __hip_bfloat16 out[VEC];
    #pragma unroll
    for (int k = 0; k < VEC; k++) {
      const float x = static_cast<float>(xi[k]);
      float g = __expf(x - lse_r);
      if (i + k == target) g -= (1.0f - LABEL_SMOOTHING);
      g -= ls_term;
      out[k] = static_cast<__hip_bfloat16>(g * scale);
    }
    *reinterpret_cast<float4*>(&row_dlogits[i]) = *reinterpret_cast<float4*>(out);
  }
  for (int i = VOCAB_VEC + tid; i < VOCAB; i += THREADS_PER_WG) {
    const float x = static_cast<float>(row_logits[i]);
    float g = __expf(x - lse_r);
    if (i == target) g -= (1.0f - LABEL_SMOOTHING);
    g -= ls_term;
    row_dlogits[i] = static_cast<__hip_bfloat16>(g * scale);
  }
}
