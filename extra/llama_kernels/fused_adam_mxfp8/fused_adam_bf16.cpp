typedef __bf16 hip_bfloat16;
extern "C" __attribute__((device, const)) float __ocml_sqrt_f32(float);

#ifndef N_ELEMS
#define N_ELEMS 1024
#endif
#ifndef B1
#define B1 0.9f
#endif
#ifndef B2
#define B2 0.95f
#endif
#ifndef OMB1
#define OMB1 0.1f
#endif
#ifndef OMB2
#define OMB2 0.05f
#endif
#ifndef EPS
#define EPS 1.0e-5f
#endif
#ifndef GPTOSS_ADAM_BF16_RAW_CLIP
#define GPTOSS_ADAM_BF16_RAW_CLIP 0
#endif

#if GPTOSS_ADAM_BF16_RAW_CLIP
__attribute__((device, always_inline)) float adam_clip_bf16(float value, float scale) {
  unsigned int bits = __builtin_bit_cast(unsigned int, value * scale);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  float rounded = __builtin_bit_cast(float, bits & 0xffff0000u);
  asm volatile("" : "+v"(rounded));
  return rounded;
}
#endif

// GPT-OSS' BF16 embedding and LM-head parameters use BF16 optimizer state and a
// ZeRO-sharded FP32 master. The ordinary tinygrad graph walks the 700+ MB local
// shard three times: once each for m, v, and master, and recomputes m_new/v_new
// in the master pass. Do the three updates together. The input gradient already
// has the same BF16 clipping boundary as the lazy optimizer expression.
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(256, 256))) fused_adam_bf16_vocab(
    hip_bfloat16 *__restrict__ m, hip_bfloat16 *__restrict__ v,
    float *__restrict__ master, const hip_bfloat16 *__restrict__ grad,
#if GPTOSS_ADAM_BF16_RAW_CLIP
    const float *__restrict__ clip_scale,
#endif
    const float *__restrict__ lr,
    const float *__restrict__ b1_t, const float *__restrict__ b2_t) {
  constexpr int THREADS = 256;
  constexpr int ELEMS_PER_THREAD = 4;
  static_assert(N_ELEMS % (THREADS * ELEMS_PER_THREAD) == 0);
  const long long base = (long long)__builtin_amdgcn_workgroup_id_x() *
                         (THREADS * ELEMS_PER_THREAD) +
                         __builtin_amdgcn_workitem_id_x();

  // Hoist the scalar optimizer inputs and schedule the four independent square
  // roots together. This is the same phase scheduling that wins in FC1 Adam.
  const float inv_b1 = 1.0f / (1.0f - b1_t[0]);
  const float inv_b2 = 1.0f / (1.0f - b2_t[0]);
  const float lr_v = lr[0];
#if GPTOSS_ADAM_BF16_RAW_CLIP
  const float grad_clip = clip_scale[0];
#endif
  float m_new[ELEMS_PER_THREAD], v_new[ELEMS_PER_THREAD];

  #pragma unroll
  for (int j = 0; j < ELEMS_PER_THREAD; j++) {
    const long long idx = base + j * THREADS;
    float g = (float)grad[idx];
#if GPTOSS_ADAM_BF16_RAW_CLIP
    g = adam_clip_bf16(g, grad_clip);
#endif
    // tinygrad's reference codegen rounds OMB1*g, then uses it as the addend
    // of fma(B1, m, ...). Spell that contraction explicitly: swapping which
    // product is rounded first changes a handful of BF16 boundary values.
    const float m_grad = OMB1 * g;
    m_new[j] = __builtin_fmaf(B1, (float)m[idx], m_grad);
    // The reference evaluates (OMB2*g)*g as two rounded multiplies before
    // adding B2*v with an FMA. The empty VGPR constraint prevents contraction
    // of those two multiplies without introducing memory traffic.
    float v_grad = OMB2 * g;
    asm volatile("" : "+v"(v_grad));
    v_grad *= g;
    asm volatile("" : "+v"(v_grad));
    v_new[j] = __builtin_fmaf(B2, (float)v[idx], v_grad);
  }

  #pragma unroll
  for (int j = 0; j < ELEMS_PER_THREAD; j++) {
    const long long idx = base + j * THREADS;
    const float new_w = master[idx] -
      (lr_v * m_new[j] * inv_b1 * (1.0f / (__ocml_sqrt_f32(v_new[j] * inv_b2) + EPS)));
    m[idx] = (hip_bfloat16)m_new[j];
    v[idx] = (hip_bfloat16)v_new[j];
    master[idx] = new_w;
  }
}
