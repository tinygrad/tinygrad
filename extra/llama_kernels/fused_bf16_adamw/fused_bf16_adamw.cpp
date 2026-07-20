#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>

#ifndef ELEMS
#define ELEMS 65667072
#endif
#ifndef BETA1
#define BETA1 0.9f
#endif
#ifndef BETA2
#define BETA2 0.95f
#endif
#ifndef ONE_MINUS_BETA1
#define ONE_MINUS_BETA1 0.1f
#endif
#ifndef ONE_MINUS_BETA2
#define ONE_MINUS_BETA2 0.05f
#endif
#ifndef EPS
#define EPS 1e-5f
#endif
#ifndef WEIGHT_DECAY
#define WEIGHT_DECAY 0.1f
#endif

constexpr int NUM_WG = 512;
constexpr int THREADS_PER_WG = 256;

extern "C" __global__ __launch_bounds__(THREADS_PER_WG, 1) void
fused_bf16_adamw(float *master, __hip_bfloat16 *weight, __hip_bfloat16 *m, __hip_bfloat16 *v, const __hip_bfloat16 *grad,
                 const float *grad_scale, const float *lr, const float *b1_t, const float *b2_t) {
  const size_t pair_elems = ELEMS / 2;
  float2 *master2 = reinterpret_cast<float2*>(master);
  __hip_bfloat162 *weight2 = reinterpret_cast<__hip_bfloat162*>(weight);
  __hip_bfloat162 *m2 = reinterpret_cast<__hip_bfloat162*>(m);
  __hip_bfloat162 *v2 = reinterpret_cast<__hip_bfloat162*>(v);
  const __hip_bfloat162 *grad2 = reinterpret_cast<const __hip_bfloat162*>(grad);
  const float gs = grad_scale[0];
  const float lr0 = lr[0];
  const float m_correction = 1.0f / (1.0f - b1_t[0]);
  const float v_correction = 1.0f / (1.0f - b2_t[0]);
  const float decay = lr0 * WEIGHT_DECAY;

  for (size_t i = blockIdx.x * THREADS_PER_WG + threadIdx.x; i < pair_elems; i += NUM_WG * THREADS_PER_WG) {
    float2 g = __bfloat1622float2(grad2[i]);
    g = __bfloat1622float2(__float22bfloat162_rn(make_float2(g.x * gs, g.y * gs)));
    const float2 m_old = __bfloat1622float2(m2[i]);
    const float2 v_old = __bfloat1622float2(v2[i]);
    const float2 old_w = master2[i];
    const float2 m_new = make_float2(BETA1 * m_old.x + ONE_MINUS_BETA1 * g.x, BETA1 * m_old.y + ONE_MINUS_BETA1 * g.y);
    const float2 v_new = make_float2(BETA2 * v_old.x + ONE_MINUS_BETA2 * g.x * g.x, BETA2 * v_old.y + ONE_MINUS_BETA2 * g.y * g.y);
    const float2 update = make_float2(lr0 * (m_new.x * m_correction) / (sqrtf(v_new.x * v_correction) + EPS),
                                      lr0 * (m_new.y * m_correction) / (sqrtf(v_new.y * v_correction) + EPS));
    const float2 new_w = make_float2(old_w.x - update.x - decay * old_w.x, old_w.y - update.y - decay * old_w.y);
    master2[i] = new_w;
    weight2[i] = __float22bfloat162_rn(new_w);
    m2[i] = __float22bfloat162_rn(m_new);
    v2[i] = __float22bfloat162_rn(v_new);
  }
}
