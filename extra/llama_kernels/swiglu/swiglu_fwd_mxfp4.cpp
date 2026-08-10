#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "quantize_mxfp4_device.h"

#if !defined(KERNEL_NAME) || !defined(M_DIM) || !defined(N_DIM)
#error kernel dimensions and name must be defined
#endif

namespace {

constexpr int M = M_DIM;
constexpr int N = N_DIM;
constexpr int INPUT_N = 2 * N;
constexpr int BLOCK = 32;
constexpr int TILE_M = 128;
constexpr int TILE_N = 64;
constexpr int THREADS_PER_ROW = 8;
constexpr int VALUES_PER_THREAD = 4;
constexpr int SMEM_STRIDE = BLOCK + 2;
constexpr float LOG2E = 1.4426950408889634f;

static_assert(M % TILE_M == 0 && N % TILE_N == 0);

} // namespace

extern "C" __global__ __launch_bounds__(256, 8)
void KERNEL_NAME(__hip_bfloat16* __restrict__ out,
                 uint8_t* __restrict__ row_fp4, uint8_t* __restrict__ row_scale,
                 uint8_t* __restrict__ col_fp4, uint8_t* __restrict__ col_scale,
                 const __hip_bfloat16* __restrict__ packed) {
  __shared__ uint16_t tile[BLOCK * SMEM_STRIDE];
  const int line = threadIdx.x / THREADS_PER_ROW;
  const int lane = threadIdx.x % THREADS_PER_ROW;
  const int block_m = blockIdx.x * TILE_M;
  const int block_n = blockIdx.y * TILE_N;

  #pragma unroll
  for (int chunk_m = 0; chunk_m < TILE_M / BLOCK; chunk_m++) {
    #pragma unroll
    for (int chunk_n = 0; chunk_n < TILE_N / BLOCK; chunk_n++) {
      const int row = block_m + chunk_m * BLOCK + line;
      const int col = block_n + chunk_n * BLOCK + lane * VALUES_PER_THREAD;
      __hip_bfloat16 values[VALUES_PER_THREAD];
      #pragma unroll
      for (int j = 0; j < VALUES_PER_THREAD; j++) {
        const float act = static_cast<float>(packed[row * INPUT_N + col + j]);
        const float gate = static_cast<float>(packed[row * INPUT_N + N + col + j]);
        values[j] = __hip_bfloat16(act / (1.0f + exp2f(-LOG2E * act)) * gate);
        tile[line * SMEM_STRIDE + lane * VALUES_PER_THREAD + j] = *reinterpret_cast<const uint16_t*>(&values[j]);
      }
      __syncthreads();

      const auto row_result = mxfp4::quantize(mxfp4::load_bf16x4(tile + line * SMEM_STRIDE + lane * VALUES_PER_THREAD), lane);
      mxfp4::store_fp4<false>(row_fp4, row, col / 2, N / 2, row_result.fp4);
      if (lane == 0) mxfp4::store_scale(row_scale, row, col / BLOCK, N / BLOCK, row_result.scale);

      const int row_lane = lane * VALUES_PER_THREAD;
      const int col_line = block_n + chunk_n * BLOCK + line;
      const auto col_result = mxfp4::quantize(make_float4(
        __uint_as_float(static_cast<uint32_t>(tile[(row_lane + 0) * SMEM_STRIDE + line]) << 16),
        __uint_as_float(static_cast<uint32_t>(tile[(row_lane + 1) * SMEM_STRIDE + line]) << 16),
        __uint_as_float(static_cast<uint32_t>(tile[(row_lane + 2) * SMEM_STRIDE + line]) << 16),
        __uint_as_float(static_cast<uint32_t>(tile[(row_lane + 3) * SMEM_STRIDE + line]) << 16)), lane);
      mxfp4::store_fp4<true>(col_fp4, col_line, (block_m + chunk_m * BLOCK + row_lane) / 2, M / 2, col_result.fp4);
      if (lane == 0) mxfp4::store_scale(col_scale, col_line, (block_m + chunk_m * BLOCK) / BLOCK, M / BLOCK, col_result.scale);
      __syncthreads();
    }
  }
}
