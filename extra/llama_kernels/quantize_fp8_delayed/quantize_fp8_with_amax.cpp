#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#ifndef N_ELEMS
#define N_ELEMS 67108864
#endif
#ifndef NUM_WG
#define NUM_WG 1024
#endif
#ifndef THREADS_PER_WG
#define THREADS_PER_WG 256
#endif
#ifndef LAYER_SCALE
#define LAYER_SCALE 0
#endif

constexpr int VEC = 8;
constexpr int WAVE_SIZE = 64;
constexpr float FP8_MAX = 448.0f;

static_assert(N_ELEMS % (NUM_WG * THREADS_PER_WG * VEC) == 0, "unsupported quantize shape");
static_assert(THREADS_PER_WG % WAVE_SIZE == 0, "workgroup must contain whole waves");

__forceinline__ __device__ float atomicMaxOfNonNegative(float* addr, float value) {
  return __int_as_float(atomicMax(reinterpret_cast<int32_t*>(addr), __float_as_int(value)));
}

extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void
quantize_fp8_with_amax(
    __hip_fp8_storage_t*  __restrict__ fp8_out,
    float*                __restrict__ amax_out,
    const __hip_bfloat16* __restrict__ x,
    const float*          __restrict__ amax_state
#if LAYER_SCALE
    , const int*          __restrict__ layer_num
#endif
) {
  __shared__ float wave_max[THREADS_PER_WG / WAVE_SIZE];
  const int tid = threadIdx.x;
#if LAYER_SCALE
  const int layer = layer_num[0];
#else
  const int layer = 0;
#endif
  const float scale = FP8_MAX / (amax_state[layer] + 1e-8f);
  float local_max = 0.0f;

  for (size_t base = ((size_t)blockIdx.x * THREADS_PER_WG + tid) * VEC; base < N_ELEMS;
       base += (size_t)NUM_WG * THREADS_PER_WG * VEC) {
    const float4 raw = *reinterpret_cast<const float4*>(&x[base]);
    const __hip_bfloat16* vals = reinterpret_cast<const __hip_bfloat16*>(&raw);
    __hip_fp8_storage_t out[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; i++) {
      const float value = static_cast<float>(vals[i]);
      local_max = fmaxf(local_max, fabsf(value));
      out[i] = __hip_cvt_float_to_fp8(fmaxf(-FP8_MAX, fminf(FP8_MAX, value * scale)), __HIP_SATFINITE, __HIP_E4M3);
    }
    *reinterpret_cast<uint64_t*>(&fp8_out[base]) = *reinterpret_cast<uint64_t*>(out);
  }

  #pragma unroll
  for (int offset = WAVE_SIZE / 2; offset; offset >>= 1) local_max = fmaxf(local_max, __shfl_down(local_max, offset));
  if ((tid & (WAVE_SIZE - 1)) == 0) wave_max[tid / WAVE_SIZE] = local_max;
  __syncthreads();

  local_max = tid < THREADS_PER_WG / WAVE_SIZE ? wave_max[tid] : 0.0f;
  if (tid < WAVE_SIZE) {
    #pragma unroll
    for (int offset = WAVE_SIZE / 2; offset; offset >>= 1) local_max = fmaxf(local_max, __shfl_down(local_max, offset));
  }
  float* amax_out_layer = amax_out + layer;
  if (tid == 0 && local_max > *amax_out_layer) atomicMaxOfNonNegative(amax_out_layer, local_max);
}
