#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>

#ifndef LAYERS
#define LAYERS 32
#endif
#ifndef LAYER_ELEMS
#define LAYER_ELEMS 16777216
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
constexpr float FP8_MAX = 448.0f;
constexpr float AMAX_MARGIN = 1.1f;

extern "C" __global__ __launch_bounds__(THREADS_PER_WG, 1) void
fused_fp8_adamw(float *master, __hip_fp8_storage_t *weight, float *next_inv, __hip_bfloat16 *m, __hip_bfloat16 *v,
                const __hip_bfloat16 *grad, const float *grad_scale, const float *inv_scale, const float *lr, const float *b1_t, const float *b2_t) {
  if (blockIdx.x >= LAYERS * NUM_WG) return;
  const int layer = blockIdx.x / NUM_WG;
  const int wg = blockIdx.x % NUM_WG;
  const size_t layer_start = (size_t)layer * LAYER_ELEMS;
  const size_t pair_start = layer_start / 2;
  const size_t pair_elems = LAYER_ELEMS / 2;
  float2 *master2 = reinterpret_cast<float2*>(master);
  unsigned short *weight2 = reinterpret_cast<unsigned short*>(weight);
  __hip_bfloat162 *m2 = reinterpret_cast<__hip_bfloat162*>(m);
  __hip_bfloat162 *v2 = reinterpret_cast<__hip_bfloat162*>(v);
  const __hip_bfloat162 *grad2 = reinterpret_cast<const __hip_bfloat162*>(grad);
  const float gs = grad_scale[0];
  const float inv = 1.0f / inv_scale[layer];
  const float lr0 = lr[0];
  const float m_correction = 1.0f / (1.0f - b1_t[0]);
  const float v_correction = 1.0f / (1.0f - b2_t[0]);
  const float decay = lr0 * WEIGHT_DECAY;
  float local_max = 0.0f;

  for (size_t i = wg * THREADS_PER_WG + threadIdx.x; i < pair_elems; i += NUM_WG * THREADS_PER_WG) {
    const size_t idx = pair_start + i;
    float2 g = __bfloat1622float2(grad2[idx]);
    g = __bfloat1622float2(__float22bfloat162_rn(make_float2(g.x * gs, g.y * gs)));
    const float2 m_old = __bfloat1622float2(m2[idx]);
    const float2 v_old = __bfloat1622float2(v2[idx]);
    const float2 old_w = master2[idx];
    const float2 m_new = make_float2(BETA1 * m_old.x + ONE_MINUS_BETA1 * g.x, BETA1 * m_old.y + ONE_MINUS_BETA1 * g.y);
    const float2 v_new = make_float2(BETA2 * v_old.x + ONE_MINUS_BETA2 * g.x * g.x, BETA2 * v_old.y + ONE_MINUS_BETA2 * g.y * g.y);
    const float2 update = make_float2(lr0 * (m_new.x * m_correction) / (sqrtf(v_new.x * v_correction) + EPS),
                                      lr0 * (m_new.y * m_correction) / (sqrtf(v_new.y * v_correction) + EPS));
    const float2 new_w = make_float2(old_w.x - update.x - decay * old_w.x, old_w.y - update.y - decay * old_w.y);
    const float2 scaled = make_float2(fminf(FP8_MAX, fmaxf(-FP8_MAX, new_w.x * inv)),
                                      fminf(FP8_MAX, fmaxf(-FP8_MAX, new_w.y * inv)));
    const __hip_fp8_storage_t q0 = __hip_cvt_float_to_fp8(scaled.x, __HIP_SATFINITE, __HIP_E4M3);
    const __hip_fp8_storage_t q1 = __hip_cvt_float_to_fp8(scaled.y, __HIP_SATFINITE, __HIP_E4M3);
    m2[idx] = __float22bfloat162_rn(m_new);
    v2[idx] = __float22bfloat162_rn(v_new);
    master2[idx] = new_w;
    weight2[idx] = (unsigned short)q0 | ((unsigned short)q1 << 8);
    local_max = fmaxf(local_max, fmaxf(fabsf(__half2float(__half(__hip_cvt_fp8_to_halfraw(q0, __HIP_E4M3)))),
                                      fabsf(__half2float(__half(__hip_cvt_fp8_to_halfraw(q1, __HIP_E4M3))))));
  }

  __shared__ float maxima[THREADS_PER_WG];
  maxima[threadIdx.x] = local_max;
  __syncthreads();
  for (int step = 128; step; step >>= 1) {
    if (threadIdx.x < step) maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + step]);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float candidate = (maxima[0] * inv_scale[layer] * AMAX_MARGIN + 1e-8f) / FP8_MAX;
    if (candidate > next_inv[layer])
      __hip_atomic_fetch_max((int*)&next_inv[layer], __float_as_int(candidate), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
}
