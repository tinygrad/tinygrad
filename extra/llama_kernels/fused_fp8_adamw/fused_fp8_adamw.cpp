#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>

#ifndef LAYERS
#define LAYERS 32
#endif
#ifndef LAYER_ELEMS
#define LAYER_ELEMS 16777216
#endif
#ifndef WEIGHT_DECAY
#define WEIGHT_DECAY 0.1f
#endif

constexpr int NUM_WG = 256;
constexpr int THREADS_PER_WG = 256;
constexpr float FP8_MAX = 448.0f;
constexpr float AMAX_MARGIN = 1.1f;

extern "C" __global__ __launch_bounds__(THREADS_PER_WG, 1) void
fused_fp8_adamw(float *master, __hip_fp8_storage_t *weight, float *next_inv, const float *update,
                const float *inv_scale, const float *lr) {
  if (blockIdx.x >= LAYERS * NUM_WG) return;
  const int layer = blockIdx.x / NUM_WG;
  const int wg = blockIdx.x % NUM_WG;
  const int layer_start = layer * LAYER_ELEMS;
  float local_max = 0.0f;

  for (int i = wg * THREADS_PER_WG + threadIdx.x; i < LAYER_ELEMS; i += NUM_WG * THREADS_PER_WG) {
    const int idx = layer_start + i;
    const float old_w = master[idx];
    const float new_w = old_w - update[idx] - lr[0] * WEIGHT_DECAY * old_w;
    const float scaled = fminf(FP8_MAX, fmaxf(-FP8_MAX, new_w / inv_scale[layer]));
    const __hip_fp8_storage_t q = __hip_cvt_float_to_fp8(scaled, __HIP_SATFINITE, __HIP_E4M3);
    master[idx] = new_w;
    weight[idx] = q;
    local_max = fmaxf(local_max, fabsf(__half2float(__half(__hip_cvt_fp8_to_halfraw(q, __HIP_E4M3)))));
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
