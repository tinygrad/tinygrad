#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "quantize_mxfp4_device.h"

#if !defined(KERNEL_NAME) || !defined(M_DIM) || !defined(N_DIM)
#error kernel dimensions and name must be defined
#endif

namespace {

constexpr int M = M_DIM;
constexpr int N = N_DIM;
constexpr int HIDDEN = N / 2;
constexpr int BLOCK = 32;
constexpr int TILE_M = 128;
constexpr int THREADS_PER_ROW = 8;
constexpr int VALUES_PER_THREAD = 4;
constexpr int SMEM_STRIDE = BLOCK + 2;
constexpr float LOG2E = 1.4426950408889634f;

static_assert(M % TILE_M == 0 && HIDDEN % BLOCK == 0);

__device__ __forceinline__ void swiglu_grads(const __hip_bfloat16* packed, const __hip_bfloat16* grad,
                                             int row, int col, float& dact, float& dgate) {
  const float act = static_cast<float>(packed[row * N + col]);
  const float gate = static_cast<float>(packed[row * N + HIDDEN + col]);
  const float upstream = static_cast<float>(grad[row * HIDDEN + col]);
  const float sigmoid = 1.0f / (1.0f + exp2f(-LOG2E * act));
  const float silu = act * sigmoid;
  dact = upstream * (sigmoid + silu * (1.0f - sigmoid)) * gate;
  dgate = upstream * silu;
}

__device__ __forceinline__ void store_quantized_tile(
    uint16_t* tile, __hip_bfloat16* grad_out,
    uint8_t* row_fp4, uint8_t* row_scale, uint8_t* col_fp4, uint8_t* col_scale,
    const __hip_bfloat16 (&values)[VALUES_PER_THREAD], int row, int col, int line, int lane) {
  #pragma unroll
  for (int j = 0; j < VALUES_PER_THREAD; j++) {
    grad_out[row * N + col + j] = values[j];
    tile[line * SMEM_STRIDE + lane * VALUES_PER_THREAD + j] = *reinterpret_cast<const uint16_t*>(&values[j]);
  }
  __syncthreads();

  const mxfp4::Quantized4 row_result = mxfp4::quantize(
    mxfp4::load_bf16x4(tile + line * SMEM_STRIDE + lane * VALUES_PER_THREAD), lane);
  mxfp4::store_fp4<false>(row_fp4, row, col / 2, N / 2, row_result.fp4);
  if (lane == 0) mxfp4::store_scale(row_scale, row, col / BLOCK, N / BLOCK, row_result.scale);

  const int row_lane = lane * VALUES_PER_THREAD;
  const int col_line = col - lane * VALUES_PER_THREAD + line;
  const mxfp4::Quantized4 col_result = mxfp4::quantize(make_float4(
    __uint_as_float(static_cast<uint32_t>(tile[(row_lane + 0) * SMEM_STRIDE + line]) << 16),
    __uint_as_float(static_cast<uint32_t>(tile[(row_lane + 1) * SMEM_STRIDE + line]) << 16),
    __uint_as_float(static_cast<uint32_t>(tile[(row_lane + 2) * SMEM_STRIDE + line]) << 16),
    __uint_as_float(static_cast<uint32_t>(tile[(row_lane + 3) * SMEM_STRIDE + line]) << 16)), lane);
  mxfp4::store_fp4<false>(col_fp4, col_line, (row - line + row_lane) / 2, M / 2, col_result.fp4);
  if (lane == 0) mxfp4::store_scale(col_scale, col_line, (row - line) / BLOCK, M / BLOCK, col_result.scale);
  __syncthreads();
}

} // namespace

extern "C" __global__ __launch_bounds__(256, 8)
void KERNEL_NAME(__hip_bfloat16* __restrict__ grad_out,
                 uint8_t* __restrict__ row_fp4, uint8_t* __restrict__ row_scale,
                 uint8_t* __restrict__ col_fp4, uint8_t* __restrict__ col_scale,
                 const __hip_bfloat16* __restrict__ packed, const __hip_bfloat16* __restrict__ grad) {
  __shared__ uint16_t tile[BLOCK * SMEM_STRIDE];
  const int line = threadIdx.x / THREADS_PER_ROW;
  const int lane = threadIdx.x % THREADS_PER_ROW;
  const int block_m = blockIdx.x * TILE_M;
  const int block_col = blockIdx.y * BLOCK;

  #pragma unroll
  for (int chunk_m = 0; chunk_m < TILE_M / BLOCK; chunk_m++) {
    const int row = block_m + chunk_m * BLOCK + line;
    const int col = block_col + lane * VALUES_PER_THREAD;
    __hip_bfloat16 dact[VALUES_PER_THREAD], dgate[VALUES_PER_THREAD];
    #pragma unroll
    for (int j = 0; j < VALUES_PER_THREAD; j++) {
      float dact_f, dgate_f;
      swiglu_grads(packed, grad, row, col + j, dact_f, dgate_f);
      dact[j] = __hip_bfloat16(dact_f);
      dgate[j] = __hip_bfloat16(dgate_f);
    }
    store_quantized_tile(tile, grad_out, row_fp4, row_scale, col_fp4, col_scale, dact, row, col, line, lane);
    store_quantized_tile(tile, grad_out, row_fp4, row_scale, col_fp4, col_scale, dgate, row, HIDDEN + col, line, lane);
  }
}
