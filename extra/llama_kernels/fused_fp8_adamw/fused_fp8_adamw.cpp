#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include "kittens.cuh"

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

constexpr int NUM_WG = 1024;
constexpr int THREADS_PER_WG = 256;
constexpr float FP8_MAX = 448.0f;
constexpr float AMAX_MARGIN = 1.1f;

__device__ __forceinline__ float fp8_abs(__hip_fp8_storage_t q) {
  return fabsf(__half2float(__half(__hip_cvt_fp8_to_halfraw(q, __HIP_E4M3))));
}

extern "C" __global__ __launch_bounds__(THREADS_PER_WG, 1) void
fused_fp8_adamw(float *master, __hip_fp8_storage_t *weight, float *next_inv, __hip_bfloat16 *m, __hip_bfloat16 *v,
                const __hip_bfloat16 *grad, const float *grad_scale, const float *inv_scale, const float *lr, const float *b1_t, const float *b2_t) {
  if (blockIdx.x >= LAYERS * NUM_WG) return;
  const int layer = blockIdx.x / NUM_WG;
  const int wg = blockIdx.x % NUM_WG;
  const size_t quad_start = (size_t)layer * LAYER_ELEMS / 4;
  const size_t quad_elems = LAYER_ELEMS / 4;
  float4 *master4 = reinterpret_cast<float4*>(master);
  unsigned int *weight4 = reinterpret_cast<unsigned int*>(weight);
  __hip_bfloat162 *m2 = reinterpret_cast<__hip_bfloat162*>(m);
  __hip_bfloat162 *v2 = reinterpret_cast<__hip_bfloat162*>(v);
  const __hip_bfloat162 *grad2 = reinterpret_cast<const __hip_bfloat162*>(grad);
  const float gs = grad_scale[0];
  const float inv = 1.0f / inv_scale[layer];
  const float lr0 = lr[0];
  const float m_correction = 1.0f / (1.0f - b1_t[0]);
  const float v_correction = 1.0f / (1.0f - b2_t[0]);
  const float decay = lr0 * WEIGHT_DECAY;
  unsigned int local_max = 0;

  for (size_t i = wg * THREADS_PER_WG + threadIdx.x; i < quad_elems; i += NUM_WG * THREADS_PER_WG) {
    const size_t idx = quad_start + i;
    const size_t pair = idx * 2;
    const float4 old = kittens::load_global_vec4(master4 + idx);
    const float2 g01_raw = __bfloat1622float2(grad2[pair]);
    const float2 g23_raw = __bfloat1622float2(grad2[pair+1]);
    const float2 g01 = __bfloat1622float2(__float22bfloat162_rn(make_float2(g01_raw.x * gs, g01_raw.y * gs)));
    const float2 g23 = __bfloat1622float2(__float22bfloat162_rn(make_float2(g23_raw.x * gs, g23_raw.y * gs)));
    const float2 mo01 = __bfloat1622float2(m2[pair]), mo23 = __bfloat1622float2(m2[pair+1]);
    const float2 vo01 = __bfloat1622float2(v2[pair]), vo23 = __bfloat1622float2(v2[pair+1]);
    const float old_w[4] = {old.x, old.y, old.z, old.w};
    const float gv[4] = {g01.x, g01.y, g23.x, g23.y};
    const float mv[4] = {mo01.x, mo01.y, mo23.x, mo23.y};
    const float vv[4] = {vo01.x, vo01.y, vo23.x, vo23.y};
    float mn[4], vn[4], nw[4], scaled[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      mn[j] = BETA1 * mv[j] + ONE_MINUS_BETA1 * gv[j];
      vn[j] = BETA2 * vv[j] + ONE_MINUS_BETA2 * gv[j] * gv[j];
      const float update = lr0 * (mn[j] * m_correction) / (sqrtf(vn[j] * v_correction) + EPS);
      nw[j] = old_w[j] - update - decay * old_w[j];
      scaled[j] = fminf(FP8_MAX, fmaxf(-FP8_MAX, nw[j] * inv));
    }
    const unsigned short q01 = __hip_cvt_float2_to_fp8x2(make_float2(scaled[0], scaled[1]), __HIP_SATFINITE, __HIP_E4M3);
    const unsigned short q23 = __hip_cvt_float2_to_fp8x2(make_float2(scaled[2], scaled[3]), __HIP_SATFINITE, __HIP_E4M3);
    local_max = max(local_max, max(max(q01 & 0x7f, (q01 >> 8) & 0x7f), max(q23 & 0x7f, (q23 >> 8) & 0x7f)));
    m2[pair] = __float22bfloat162_rn(make_float2(mn[0], mn[1]));
    m2[pair+1] = __float22bfloat162_rn(make_float2(mn[2], mn[3]));
    v2[pair] = __float22bfloat162_rn(make_float2(vn[0], vn[1]));
    v2[pair+1] = __float22bfloat162_rn(make_float2(vn[2], vn[3]));
    master4[idx] = make_float4(nw[0], nw[1], nw[2], nw[3]);
    weight4[idx] = (unsigned int)q01 | ((unsigned int)q23 << 16);
  }

  #pragma unroll
  for (int step = kittens::WARP_THREADS / 2; step; step >>= 1)
    local_max = max(local_max, kittens::packed_shfl_down(kittens::MASK_ALL, local_max, step));
  __shared__ unsigned int maxima[THREADS_PER_WG / kittens::WARP_THREADS];
  const int lane = kittens::laneid(), warp = kittens::warpid();
  if (lane == 0) maxima[warp] = local_max;
  __syncthreads();
  if (warp == 0) {
    local_max = lane < THREADS_PER_WG / kittens::WARP_THREADS ? maxima[lane] : 0;
    #pragma unroll
    for (int step = kittens::WARP_THREADS / 2; step; step >>= 1)
      local_max = max(local_max, kittens::packed_shfl_down(kittens::MASK_ALL, local_max, step));
  }
  if (threadIdx.x == 0) {
    const float candidate = (fp8_abs((__hip_fp8_storage_t)local_max) * inv_scale[layer] * AMAX_MARGIN + 1e-8f) / FP8_MAX;
    if (candidate > next_inv[layer])
      __hip_atomic_fetch_max((int*)&next_inv[layer], __float_as_int(candidate), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
}
