#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

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
#define EPS 1.0e-8f
#endif
#ifndef WEIGHT_DECAY
#define WEIGHT_DECAY 0.0f
#endif
#ifndef STATE_BF16
#define STATE_BF16 1
#endif
#ifndef EMIT_GROUPED_SI
#define EMIT_GROUPED_SI 0
#endif
#ifndef LAST_DIM
#define LAST_DIM 32
#endif
#ifndef EXPERT_ROWS
#define EXPERT_ROWS 1
#endif
#ifndef GPTOSS_ADAM_DOWN_LOGICAL
#define GPTOSS_ADAM_DOWN_LOGICAL 0
#endif
#ifndef GPTOSS_ADAM_REAL_ROWS
#define GPTOSS_ADAM_REAL_ROWS 2880
#endif
#ifndef GPTOSS_ADAM_REAL_COLS
#define GPTOSS_ADAM_REAL_COLS 2880
#endif
#ifndef GPTOSS_ADAM_FC1_PIPELINE
#define GPTOSS_ADAM_FC1_PIPELINE 0
#endif
#ifndef GPTOSS_ADAM_FC1_ROW_SKIP
#define GPTOSS_ADAM_FC1_ROW_SKIP 0
#endif
#ifndef GPTOSS_ADAM_FC1_SPLIT_MAIN
#define GPTOSS_ADAM_FC1_SPLIT_MAIN 0
#endif
#ifndef GPTOSS_ADAM_FC1_SPLIT_TAIL
#define GPTOSS_ADAM_FC1_SPLIT_TAIL 0
#endif
#if GPTOSS_ADAM_FC1_SPLIT_MAIN && GPTOSS_ADAM_FC1_SPLIT_TAIL
#error "FC1 Adam main and tail are separate launches"
#endif
#ifndef GPTOSS_ADAM_DOWN_PIPELINE
#define GPTOSS_ADAM_DOWN_PIPELINE 0
#endif
#ifndef GPTOSS_ADAM_RAW_CLIP
#define GPTOSS_ADAM_RAW_CLIP 0
#endif
#if STATE_BF16
using state_t = __hip_bfloat16;
#else
using state_t = float;
#endif

using short2v = short __attribute__((ext_vector_type(2)));

#ifndef GPTOSS_ADAM_COMPACT_Q
#define GPTOSS_ADAM_COMPACT_Q 0
#endif

// Only the transport output is compact. All optimizer-state accesses retain the original physical indexing.
#if GPTOSS_ADAM_COMPACT_Q
__device__ __forceinline__ void adam_store_q_pair(unsigned char *q, long long idx, unsigned char a, unsigned char b) {
  static_assert(LAST_DIM == 3072 && (EXPERT_ROWS == 5888 || EXPERT_ROWS == 3072));
  constexpr int live_rows = EXPERT_ROWS == 5888 ? 5760 : 2880;
  static_assert(N_ELEMS > 0 && N_ELEMS <= 0x7fffffff, "compact q addressing requires a bounded GPTOSS tensor");
  const unsigned int local_idx = static_cast<unsigned int>(idx);
  const unsigned int row = local_idx / LAST_DIM;
  const unsigned int col = local_idx % LAST_DIM, expert_row = row % EXPERT_ROWS;
  if (expert_row >= live_rows || col >= 2880) return;
  const unsigned int compact_idx = (row / EXPERT_ROWS * live_rows + expert_row) * 2880 + col;
  q[compact_idx] = a;
  q[compact_idx + 1] = b;
}
#endif

#if GPTOSS_ADAM_RAW_CLIP
// Keep the BF16 value boundary observable to the optimizer. A C++ float->BF16->float temporary is otherwise
// folded through under -ffast-math and changes the following Adam FMAs relative to the materialized clip kernel.
__device__ __forceinline__ float adam_clip_bf16(float value, float scale) {
  unsigned int bits = __builtin_bit_cast(unsigned int, value * scale);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  float rounded = __builtin_bit_cast(float, bits & 0xffff0000u);
#if !GPTOSS_ADAM_DOWN_LOGICAL
  asm volatile("" : "+v"(rounded));
#endif
  return rounded;
}
#define ADAM_M_NEW(old_m, g) __builtin_fmaf(B1, (float)(old_m), OMB1 * (g))
#if GPTOSS_ADAM_DOWN_LOGICAL
// With the down specialization's one-element-per-thread schedule, hipcc contracts the established materialized-
// clip v expression differently from the four-lane FC1 path. Keeping that expression here preserves the exact
// full-precision v_new consumed by the master update; the stored BF16 moment alone would hide the difference.
#define ADAM_V_NEW(old_v, g) (B2 * (float)(old_v) + OMB2 * (g) * (g))
#else
#define ADAM_V_NEW(old_v, g) __builtin_fmaf(B2, (float)(old_v), OMB2 * (g) * (g))
#endif
#else
#define ADAM_M_NEW(old_m, g) (B1 * (float)(old_m) + OMB1 * (g))
#define ADAM_V_NEW(old_v, g) (B2 * (float)(old_v) + OMB2 * (g) * (g))
#endif

#if EMIT_GROUPED_SI || GPTOSS_ADAM_DOWN_PIPELINE
#define ADAM_THREADS 64
#else
#define ADAM_THREADS 256
#endif

#if GPTOSS_ADAM_DOWN_PIPELINE
// These spellings reproduce the existing one-element down kernel's FP32 associations while allowing four
// independent update chains to be scheduled in one wave. In particular, the persisted BF16 v alone cannot
// expose a reassociation here: its full-precision value also feeds the master update before being rounded.
__device__ __forceinline__ float adam_down_v_exact(float old_v, float g) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
  const float g2 = g * g;
  return B2 * old_v + OMB2 * g2;
}

__device__ __forceinline__ float adam_down_update_exact(float m_new, float v_new, float b1t, float b2t) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
  const float denom = sqrtf(v_new / (1.0f - b2t)) + EPS;
  const float corrected = __builtin_fmaf(-b1t, denom, denom);
  return m_new / corrected;
}
#endif

#if GPTOSS_ADAM_DOWN_LOGICAL
// Exact GPT-OSS down-weight update for the live 2880x2880 rectangle inside each physical 3072x3072 expert.
// Kept separate from the generic body so the 11x11 full workgroup rectangle pays no per-element tail predicate.
__device__ __forceinline__ int adam_down_update_one(long long idx, unsigned int pair_lane,
    state_t *__restrict__ m, state_t *__restrict__ v, float *__restrict__ master,
    unsigned char *__restrict__ q, const __hip_bfloat16 *__restrict__ grad,
    const float grad_clip, const float *__restrict__ lr, const float *__restrict__ b1_t, const float *__restrict__ b2_t) {
  float g = (float)grad[idx];
#if GPTOSS_ADAM_RAW_CLIP
  // Match tinygrad's materialized `(grad * scale).cast(bfloat16)` boundary before any Adam arithmetic.
  g = adam_clip_bf16(g, grad_clip);
#endif
  const float m_new = ADAM_M_NEW(m[idx], g);
#if GPTOSS_ADAM_DOWN_PIPELINE
  const float v_new = adam_down_v_exact((float)v[idx], g);
  const float update = adam_down_update_exact(m_new, v_new, b1_t[0], b2_t[0]);
#else
  const float v_new = ADAM_V_NEW(v[idx], g);
  const float update = (m_new / (1.0f - b1_t[0])) / (sqrtf(v_new / (1.0f - b2_t[0])) + EPS);
#endif
  const float old_w = master[idx];
  const float new_w = old_w - lr[0] * (update + WEIGHT_DECAY * old_w);
  m[idx] = (state_t)m_new;
  v[idx] = (state_t)v_new;
  master[idx] = new_w;

  float amax = fabsf(new_w);
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) amax = fmaxf(amax, __shfl_down(amax, offset, 32));
  amax = __shfl(amax, 0, 32);
  int scale_e8 = (int)((__builtin_bit_cast(unsigned, fmaxf(amax, 1.0e-38f)) >> 23) & 0xffu);
  scale_e8 = max(0, min(254, scale_e8));

  const float other = __shfl_down(new_w, 1, 32);
  if ((pair_lane & 1) == 0) {
    const float scale_f = __builtin_bit_cast(float, (unsigned)scale_e8 << 23);
    short2v packed = {0, 0};
    packed = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(packed, new_w, other, scale_f, false);
    const unsigned char *bytes = reinterpret_cast<const unsigned char *>(&packed);
#if GPTOSS_ADAM_COMPACT_Q
    adam_store_q_pair(q, idx, bytes[0], bytes[1]);
#else
    q[idx] = bytes[0];
    q[idx + 1] = bytes[1];
#endif
  }
  return scale_e8;
}
#endif

#if GPTOSS_ADAM_FC1_SPLIT_MAIN && GPTOSS_ADAM_RAW_CLIP && GPTOSS_ADAM_COMPACT_Q && STATE_BF16 && EMIT_GROUPED_SI && \
    N_ELEMS == 4*5888*3072 && LAST_DIM == 3072 && EXPERT_ROWS == 5888
// Each lane owns four contiguous values; eight neighboring lanes retain one 32-element MX block.
using adam_u16x4 = unsigned short __attribute__((ext_vector_type(4)));
using adam_f32x4 = float __attribute__((ext_vector_type(4)));

extern "C" __global__ __launch_bounds__(64) void fused_adam_mxfp8(
    state_t *__restrict__ m, state_t *__restrict__ v, float *__restrict__ master,
    unsigned char *__restrict__ q, unsigned char *__restrict__ e8, unsigned int *__restrict__ grouped_si,
    const __hip_bfloat16 *__restrict__ grad, const float *__restrict__ clip_scale,
    const float *__restrict__ lr, const float *__restrict__ b1_t, const float *__restrict__ b2_t) {
  const unsigned tid = __builtin_amdgcn_workitem_id_x();
  const unsigned compact_row = __builtin_amdgcn_workgroup_id_y();
  const unsigned col_group = __builtin_amdgcn_workgroup_id_x();
  const unsigned expert = compact_row/5760, expert_row = compact_row%5760;
  const unsigned row = expert*5888+expert_row;
  const unsigned base = row*3072+col_group*256+tid*4;
  const adam_u16x4 gs = *reinterpret_cast<const adam_u16x4*>(grad+base);
  const adam_u16x4 ms = *reinterpret_cast<const adam_u16x4*>(m+base);
  const adam_u16x4 vs = *reinterpret_cast<const adam_u16x4*>(v+base);
  const adam_f32x4 ws = *reinterpret_cast<const adam_f32x4*>(master+base);
  float m_news[4], v_news[4], new_ws[4];
  const float grad_clip = clip_scale[0];
  #pragma unroll
  for (int j=0;j<4;j++) {
    // Expand BF16 explicitly: vector-element bitcasts to the HIP wrapper type miscompile on this toolchain.
    float g = __builtin_bit_cast(float,(unsigned)gs[j]<<16);
    g = adam_clip_bf16(g,grad_clip);
    m_news[j] = ADAM_M_NEW(__builtin_bit_cast(float,(unsigned)ms[j]<<16),g);
    v_news[j] = ADAM_V_NEW(__builtin_bit_cast(float,(unsigned)vs[j]<<16),g);
  }
  adam_u16x4 out_m, out_v;
  adam_f32x4 out_w;
  #pragma unroll
  for (int j=0;j<4;j++) {
    const float update = (m_news[j] / (1.0f - b1_t[0])) / (sqrtf(v_news[j] / (1.0f - b2_t[0])) + EPS);
    const float old_w = ws[j];
    new_ws[j] = old_w - lr[0] * (update + WEIGHT_DECAY * old_w);
    out_m[j] = __builtin_bit_cast(unsigned short,(state_t)m_news[j]);
    out_v[j] = __builtin_bit_cast(unsigned short,(state_t)v_news[j]);
    out_w[j] = new_ws[j];
  }
  *reinterpret_cast<adam_u16x4*>(m+base) = out_m;
  *reinterpret_cast<adam_u16x4*>(v+base) = out_v;
  *reinterpret_cast<adam_f32x4*>(master+base) = out_w;
  float amax = fmaxf(fmaxf(fabsf(new_ws[0]),fabsf(new_ws[1])),fmaxf(fabsf(new_ws[2]),fabsf(new_ws[3])));
  #pragma unroll
  for (int offset=4;offset>0;offset>>=1) amax = fmaxf(amax,__shfl_down(amax,offset,8));
  amax = __shfl(amax,0,8);
  int scale_e8 = (int)((__builtin_bit_cast(unsigned,fmaxf(amax,1.0e-38f))>>23)&0xffu);
  scale_e8 = max(0,min(254,scale_e8));
  const float scale_f = __builtin_bit_cast(float,(unsigned)scale_e8<<23);
  short2v p0={0,0},p1={0,0};
  p0 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p0,new_ws[0],new_ws[1],scale_f,false);
  p1 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p1,new_ws[2],new_ws[3],scale_f,false);
  const unsigned packed = (unsigned)(unsigned short)p0[0] | ((unsigned)(unsigned short)p1[0]<<16);
  *reinterpret_cast<unsigned*>(q+compact_row*2880+col_group*256+tid*4) = packed;
  if (tid==0) {
    const unsigned s0=scale_e8, s1=__builtin_amdgcn_readlane(scale_e8,8), s2=__builtin_amdgcn_readlane(scale_e8,16);
    const unsigned s3=__builtin_amdgcn_readlane(scale_e8,24), s4=__builtin_amdgcn_readlane(scale_e8,32);
    const unsigned s5=__builtin_amdgcn_readlane(scale_e8,40), s6=__builtin_amdgcn_readlane(scale_e8,48);
    const unsigned s7=__builtin_amdgcn_readlane(scale_e8,56);
    const unsigned low=s0|(s1<<8)|(s2<<16)|(s3<<24), high=s4|(s5<<8)|(s6<<16)|(s7<<24);
    reinterpret_cast<unsigned long long*>(e8)[row*12+col_group] = (unsigned long long)low|((unsigned long long)high<<32);
    const unsigned si_idx=(expert*24+col_group*2)*5888+expert_row;
    grouped_si[si_idx]=low;
    grouped_si[si_idx+5888]=high;
  }
}
#else
extern "C" __global__ __launch_bounds__(ADAM_THREADS) void fused_adam_mxfp8(
    state_t *__restrict__ m, state_t *__restrict__ v, float *__restrict__ master,
    unsigned char *__restrict__ q, unsigned char *__restrict__ e8
#if EMIT_GROUPED_SI
    , unsigned int *__restrict__ grouped_si
#endif
    ,
    const __hip_bfloat16 *__restrict__ grad
#if GPTOSS_ADAM_RAW_CLIP
    , const float *__restrict__ clip_scale
#endif
    , const float *__restrict__ lr, const float *__restrict__ b1_t, const float *__restrict__ b2_t) {
#if GPTOSS_ADAM_FC1_SPLIT_MAIN
  // The hot launch contains only the eleven complete 256-column blocks of each logical row. Compact rows omit
  // the physical 128-row expert padding; translate back to the persistent [E,5888,3072] ABI for every access.
  const unsigned int compact_row = __builtin_amdgcn_workgroup_id_y();
  const unsigned int split_expert = compact_row / 5760;
  const unsigned int split_expert_row = compact_row % 5760;
  const unsigned int wgid = (split_expert * 5888 + split_expert_row) * 12 + __builtin_amdgcn_workgroup_id_x();
#elif GPTOSS_ADAM_FC1_SPLIT_TAIL
  // One rare workgroup owns the final physical column block, or initializes all fresh quantized bytes for a
  // padded row. Keeping this out of the hot binary avoids burdening the eleven complete blocks with tail code.
  const unsigned int split_physical_row = __builtin_amdgcn_workgroup_id_x();
  const unsigned int wgid = split_physical_row * 12 + 11;
#else
  const unsigned int wgid = __builtin_amdgcn_workgroup_id_x();
#endif
  const unsigned int tid = __builtin_amdgcn_workitem_id_x();
#if GPTOSS_ADAM_RAW_CLIP
  const float grad_clip = clip_scale[0];
#else
  constexpr float grad_clip = 1.0f;
#endif
#if EMIT_GROUPED_SI
  static_assert(LAST_DIM % 256 == 0);
  static_assert(N_ELEMS % 256 == 0);
#if GPTOSS_ADAM_FC1_ROW_SKIP
  // GPT-OSS' FC1 state is physically [E,5888,3072], while rows 5760:5888 are invariant padding. Avoid the
  // expensive Adam/sqrt/quantize path there, but initialize every fresh q/e8/SI output byte exactly as before.
  static_assert(LAST_DIM == 3072 && EXPERT_ROWS == 5888);
  {
    const long long physical_row_pad = wgid / 12;
    const int expert_row_pad = physical_row_pad % 5888;
    if (expert_row_pad >= 5760) {
#if GPTOSS_ADAM_FC1_SPLIT_TAIL
      // The main launch omits padded rows entirely. This one tail workgroup initializes their complete fresh
      // q/e8/SI ABI while intentionally leaving ignored persistent m/v/master padding bit-for-bit untouched.
      const int expert_pad = physical_row_pad / 5888;
#if !GPTOSS_ADAM_COMPACT_Q
      const long long row_q = physical_row_pad * 3072;
#endif
      const long long row_e8 = physical_row_pad * 96;
#if !GPTOSS_ADAM_COMPACT_Q
      for (int col = tid; col < 3072; col += 64) q[row_q + col] = 0;
#endif
      for (int col = tid; col < 96; col += 64) e8[row_e8 + col] = 0;
      for (int col = tid; col < 24; col += 64)
        grouped_si[((long long)expert_pad * 24 + col) * 5888 + expert_row_pad] = 0;
#else
      #pragma unroll
      for (int j = 0; j < 4; j++) if ((tid & 1) == 0)
#if GPTOSS_ADAM_COMPACT_Q
        adam_store_q_pair(q, (long long)wgid * 256 + tid + j * 64, 0, 0);
#else
        reinterpret_cast<unsigned short *>(q + (long long)wgid * 256 + tid + j * 64)[0] = 0;
#endif
      if (tid == 0) {
        const int wg_col_pad = wgid % 12;
        const int expert_pad = physical_row_pad / 5888;
        const long long si_idx_pad = ((long long)expert_pad * 24 + wg_col_pad * 2) * 5888 + expert_row_pad;
        reinterpret_cast<unsigned long long *>(e8)[wgid] = 0;
        grouped_si[si_idx_pad] = 0;
        grouped_si[si_idx_pad + 5888] = 0;
      }
#endif
      return;
    }
  }
#endif
#if GPTOSS_ADAM_FC1_SPLIT_TAIL
  static_assert(LAST_DIM == 3072 && EXPERT_ROWS == 5888 && GPTOSS_ADAM_FC1_PIPELINE && GPTOSS_ADAM_FC1_ROW_SKIP);
  // The final physical column block has exactly 64 live values (two MXFP8 blocks). Retain the established BF16
  // clip boundary and FP32 Adam expressions for those values; zero the remaining q and scale bytes explicitly.
  {
    const long long idx = (long long)wgid * 256 + tid;
    float g = (float)grad[idx];
#if GPTOSS_ADAM_RAW_CLIP
    g = adam_clip_bf16(g, grad_clip);
#endif
    const float m_new = ADAM_M_NEW(m[idx], g);
    const float v_new = ADAM_V_NEW(v[idx], g);
    const float update = (m_new / (1.0f - b1_t[0])) / (sqrtf(v_new / (1.0f - b2_t[0])) + EPS);
    const float old_w = master[idx];
    const float new_w = old_w - lr[0] * (update + WEIGHT_DECAY * old_w);
    m[idx] = (state_t)m_new;
    v[idx] = (state_t)v_new;
    master[idx] = new_w;

    float amax = fabsf(new_w);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) amax = fmaxf(amax, __shfl_down(amax, offset, 32));
    amax = __shfl(amax, 0, 32);
    int scale_e8 = (int)((__builtin_bit_cast(unsigned, fmaxf(amax, 1.0e-38f)) >> 23) & 0xffu);
    scale_e8 = max(0, min(254, scale_e8));
    const float other = __shfl_down(new_w, 1, 32);
    if ((tid & 1) == 0) {
      const float scale_f = __builtin_bit_cast(float, (unsigned)scale_e8 << 23);
      short2v packed = {0, 0};
      packed = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(packed, new_w, other, scale_f, false);
      const unsigned char *bytes = reinterpret_cast<const unsigned char *>(&packed);
#if GPTOSS_ADAM_COMPACT_Q
      adam_store_q_pair(q, idx, bytes[0], bytes[1]);
#else
      q[idx] = bytes[0];
      q[idx + 1] = bytes[1];
#endif
      #pragma unroll
#if GPTOSS_ADAM_COMPACT_Q
      for (int j = 1; j < 4; j++) adam_store_q_pair(q, idx + j * 64, 0, 0);
#else
      for (int j = 1; j < 4; j++) reinterpret_cast<unsigned short *>(q + idx + j * 64)[0] = 0;
#endif
    }
    if (tid == 0) {
      const unsigned int s0 = scale_e8, s1 = __builtin_amdgcn_readlane(scale_e8, 32);
      reinterpret_cast<unsigned long long *>(e8)[wgid] = (unsigned long long)s0 | ((unsigned long long)s1 << 8);
      const long long physical_row = wgid / 12;
      const int expert = physical_row / 5888, expert_row = physical_row % 5888;
      const long long si_idx = ((long long)expert * 24 + 22) * 5888 + expert_row;
      grouped_si[si_idx] = s0 | (s1 << 8);
      grouped_si[si_idx + 5888] = 0;
    }
    return;
  }
#endif
  int scales[4];
#if GPTOSS_ADAM_FC1_PIPELINE
  float m_news[4], v_news[4], new_ws[4];
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    const long long idx = (long long)wgid * 256 + tid + j * 64;
    float g = (float)grad[idx];
#if GPTOSS_ADAM_RAW_CLIP
    g = adam_clip_bf16(g, grad_clip);
#endif
    m_news[j] = ADAM_M_NEW(m[idx], g);
    v_news[j] = ADAM_V_NEW(v[idx], g);
  }
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    const long long idx = (long long)wgid * 256 + tid + j * 64;
    const float update = (m_news[j] / (1.0f - b1_t[0])) / (sqrtf(v_news[j] / (1.0f - b2_t[0])) + EPS);
    const float old_w = master[idx];
    new_ws[j] = old_w - lr[0] * (update + WEIGHT_DECAY * old_w);
    m[idx] = (state_t)m_news[j];
    v[idx] = (state_t)v_news[j];
    master[idx] = new_ws[j];
  }
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    const long long idx = (long long)wgid * 256 + tid + j * 64;
    float amax = fabsf(new_ws[j]);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) amax = fmaxf(amax, __shfl_down(amax, offset, 32));
    amax = __shfl(amax, 0, 32);
    int scale_e8 = (int)((__builtin_bit_cast(unsigned, fmaxf(amax, 1.0e-38f)) >> 23) & 0xffu);
    scales[j] = max(0, min(254, scale_e8));

    const float other = __shfl_down(new_ws[j], 1, 32);
    if ((tid & 1) == 0) {
      const float scale_f = __builtin_bit_cast(float, (unsigned)scales[j] << 23);
      short2v packed = {0, 0};
      packed = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(packed, new_ws[j], other, scale_f, false);
      const unsigned char *bytes = reinterpret_cast<const unsigned char *>(&packed);
#if GPTOSS_ADAM_COMPACT_Q
      adam_store_q_pair(q, idx, bytes[0], bytes[1]);
#else
      q[idx] = bytes[0];
      q[idx + 1] = bytes[1];
#endif
    }
  }
#else
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    const long long idx = (long long)wgid * 256 + tid + j * 64;
    float g = (float)grad[idx];
#if GPTOSS_ADAM_RAW_CLIP
    g = adam_clip_bf16(g, grad_clip);
#endif
    const float m_new = ADAM_M_NEW(m[idx], g);
    const float v_new = ADAM_V_NEW(v[idx], g);
    const float update = (m_new / (1.0f - b1_t[0])) / (sqrtf(v_new / (1.0f - b2_t[0])) + EPS);
    const float old_w = master[idx];
    const float new_w = old_w - lr[0] * (update + WEIGHT_DECAY * old_w);
    m[idx] = (state_t)m_new;
    v[idx] = (state_t)v_new;
    master[idx] = new_w;

    float amax = fabsf(new_w);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) amax = fmaxf(amax, __shfl_down(amax, offset, 32));
    amax = __shfl(amax, 0, 32);
    int scale_e8 = (int)((__builtin_bit_cast(unsigned, fmaxf(amax, 1.0e-38f)) >> 23) & 0xffu);
    scales[j] = max(0, min(254, scale_e8));

    const float other = __shfl_down(new_w, 1, 32);
    if ((tid & 1) == 0) {
      const float scale_f = __builtin_bit_cast(float, (unsigned)scales[j] << 23);
      short2v packed = {0, 0};
      packed = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(packed, new_w, other, scale_f, false);
      const unsigned char *bytes = reinterpret_cast<const unsigned char *>(&packed);
#if GPTOSS_ADAM_COMPACT_Q
      adam_store_q_pair(q, idx, bytes[0], bytes[1]);
#else
      q[idx] = bytes[0];
      q[idx + 1] = bytes[1];
#endif
    }
  }
#endif

  // One physical wave owns all eight 32-value scales for this 256-element workgroup. Lane zero can therefore
  // emit the ordinary e8 row fragment and both packed SI words with full-width stores and no barrier.
  if (tid == 0) {
    const unsigned int s0 = scales[0], s1 = __builtin_amdgcn_readlane(scales[0], 32);
    const unsigned int s2 = scales[1], s3 = __builtin_amdgcn_readlane(scales[1], 32);
    const unsigned int s4 = scales[2], s5 = __builtin_amdgcn_readlane(scales[2], 32);
    const unsigned int s6 = scales[3], s7 = __builtin_amdgcn_readlane(scales[3], 32);
    const unsigned long long raw = (unsigned long long)s0 | ((unsigned long long)s1 << 8) |
      ((unsigned long long)s2 << 16) | ((unsigned long long)s3 << 24) | ((unsigned long long)s4 << 32) |
      ((unsigned long long)s5 << 40) | ((unsigned long long)s6 << 48) | ((unsigned long long)s7 << 56);
    reinterpret_cast<unsigned long long *>(e8)[wgid] = raw;

    const long long row = wgid / (LAST_DIM / 256);
    const int wg_col = wgid % (LAST_DIM / 256);
    const int expert = row / EXPERT_ROWS, expert_row = row % EXPERT_ROWS;
    const long long si_idx = ((long long)expert * (LAST_DIM / 128) + wg_col * 2) * EXPERT_ROWS + expert_row;
    grouped_si[si_idx] = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);
    grouped_si[si_idx + EXPERT_ROWS] = s4 | (s5 << 8) | (s6 << 16) | (s7 << 24);
  }
#else
  const long long idx = (long long)wgid * 256 + tid;
  if (idx >= N_ELEMS) return;

#if GPTOSS_ADAM_DOWN_LOGICAL
  static_assert(LAST_DIM == 3072 && EXPERT_ROWS == 3072 && GPTOSS_ADAM_REAL_ROWS == 2880 && GPTOSS_ADAM_REAL_COLS == 2880);
  const long long physical_row = wgid / (LAST_DIM / 256);
  const int wg_col = wgid % (LAST_DIM / 256);
  const int expert_row = physical_row % EXPERT_ROWS;
#if GPTOSS_ADAM_DOWN_PIPELINE
  // One wave still owns the baseline's 256 contiguous elements. The final live column tile has exactly 64 values,
  // so j=0 is complete and j=1..3 are physical padding; padded expert rows similarly only emit q/e8 zeroes.
  const long long base_idx = (long long)wgid * 256 + tid;
  if (expert_row >= GPTOSS_ADAM_REAL_ROWS) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      const long long jidx = base_idx + j * 64;
#if GPTOSS_ADAM_COMPACT_Q
      if ((tid & 1) == 0) adam_store_q_pair(q, jidx, 0, 0);
#else
      if ((tid & 1) == 0) reinterpret_cast<unsigned short *>(q + jidx)[0] = 0;
#endif
      if ((tid & 31) == 0) e8[jidx / 32] = 0;
    }
    return;
  }
  const int live_js = wg_col == GPTOSS_ADAM_REAL_COLS / 256 ? 1 : 4;
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    const long long jidx = base_idx + j * 64;
    if (j >= live_js) {
#if GPTOSS_ADAM_COMPACT_Q
      if ((tid & 1) == 0) adam_store_q_pair(q, jidx, 0, 0);
#else
      if ((tid & 1) == 0) reinterpret_cast<unsigned short *>(q + jidx)[0] = 0;
#endif
      if ((tid & 31) == 0) e8[jidx / 32] = 0;
      continue;
    }
    const int scale_e8 = adam_down_update_one(jidx, tid, m, v, master, q, grad, grad_clip, lr, b1_t, b2_t);
    if ((tid & 31) == 0) e8[jidx / 32] = (unsigned char)scale_e8;
  }
#else
  // The optimizer state and master padding are persistent invariant zeros, so leave them untouched. q/e8 are fresh
  // output buffers each step and must be explicitly initialized to the physical ABI's zero encoding.
  if (expert_row >= GPTOSS_ADAM_REAL_ROWS ||
      (wg_col == GPTOSS_ADAM_REAL_COLS / 256 && tid >= GPTOSS_ADAM_REAL_COLS % 256)) {
    if ((idx & 31) == 0) e8[idx / 32] = 0;
#if GPTOSS_ADAM_COMPACT_Q
    if ((idx & 1) == 0) adam_store_q_pair(q, idx, 0, 0);
#else
    if ((idx & 1) == 0) reinterpret_cast<unsigned short *>(q + idx)[0] = 0;
#endif
    return;
  }
  const int scale_e8 = adam_down_update_one(idx, idx, m, v, master, q, grad, grad_clip, lr, b1_t, b2_t);
  if ((idx & 31) == 0) e8[idx / 32] = (unsigned char)scale_e8;
#endif
#else
  float g = (float)grad[idx];
#if GPTOSS_ADAM_RAW_CLIP
  g = adam_clip_bf16(g, grad_clip);
#endif
  const float m_new = ADAM_M_NEW(m[idx], g);
  const float v_new = ADAM_V_NEW(v[idx], g);
  const float update = (m_new / (1.0f - b1_t[0])) / (sqrtf(v_new / (1.0f - b2_t[0])) + EPS);
  const float old_w = master[idx];
  const float new_w = old_w - lr[0] * (update + WEIGHT_DECAY * old_w);
  m[idx] = (state_t)m_new;
  v[idx] = (state_t)v_new;
  master[idx] = new_w;

  float amax = fabsf(new_w);
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) amax = fmaxf(amax, __shfl_down(amax, offset, 32));
  amax = __shfl(amax, 0, 32);
  int scale_e8 = (int)((__builtin_bit_cast(unsigned, fmaxf(amax, 1.0e-38f)) >> 23) & 0xffu);
  scale_e8 = max(0, min(254, scale_e8));
  if ((idx & 31) == 0) e8[idx / 32] = (unsigned char)scale_e8;

  const float other = __shfl_down(new_w, 1, 32);
  if ((idx & 1) == 0) {
    const float scale_f = __builtin_bit_cast(float, (unsigned)scale_e8 << 23);
    short2v packed = {0, 0};
    packed = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(packed, new_w, other, scale_f, false);
    const unsigned char *bytes = reinterpret_cast<const unsigned char *>(&packed);
#if GPTOSS_ADAM_COMPACT_Q
    adam_store_q_pair(q, idx, bytes[0], bytes[1]);
#else
    q[idx] = bytes[0];
    q[idx + 1] = bytes[1];
#endif
  }
#endif
#endif
}
#endif
