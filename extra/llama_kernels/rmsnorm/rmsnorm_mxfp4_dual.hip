#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "quantize_mxfp4_device.h"
#include "quantize_mxfp4_8.h"

// Preserve the original 128-thread reduction tree. Reload after reduction rather than
// retaining a full normalized tile: this avoids register spills and allows two LDS tiles per CU.
constexpr int ROW_THREADS = 128, PARALLEL_ROWS = 4, TILE_ROWS = 32, TILE_COLS = 1024, VEC = 8;
constexpr int THREADS = ROW_THREADS * PARALLEL_ROWS;
constexpr int VECTOR_COLS = ROW_THREADS * VEC;
static_assert(HIDDEN == 4096 && ROWS % TILE_ROWS == 0);
// Column quantization assigns eight consecutive rows to each lane. Spread lane groups across LDS banks.
__device__ __forceinline__ uint16_t* tile_at(uint16_t (*tile)[TILE_COLS], int row, int col) {
  return &tile[row][col ^ ((row / 4) * 8)];
}
extern "C" __global__ __launch_bounds__(THREADS)
void KERNEL_NAME(__hip_bfloat16* __restrict__ out,
#if ADD_RESIDUAL
                __hip_bfloat16* __restrict__ h_out,
#endif
                float* __restrict__ rrms, const __hip_bfloat16* __restrict__ x,
#if ADD_RESIDUAL
                const __hip_bfloat16* __restrict__ residual,
#endif
                const __hip_bfloat16* __restrict__ weight,
                uint8_t* __restrict__ row_fp4, uint8_t* __restrict__ row_scale,
                uint8_t* __restrict__ col_fp4, uint8_t* __restrict__ col_scale) {
  __shared__ float sums[PARALLEL_ROWS][ROW_THREADS];
  __shared__ float scales[TILE_ROWS];
  __shared__ uint16_t tile[TILE_ROWS][TILE_COLS];
  const int tid = threadIdx.x, rt = tid % ROW_THREADS, pr = tid / ROW_THREADS;
  const int base_row = blockIdx.x * TILE_ROWS;
  #pragma unroll
  for (int r = 0; r < TILE_ROWS / PARALLEL_ROWS; r++) {
    const int row = base_row + pr + r * PARALLEL_ROWS;
    float sum = 0.0f;
    #pragma unroll
    for (int v = 0; v < HIDDEN / VECTOR_COLS; v++) {
      const int col = rt * VEC + v * VECTOR_COLS;
      const float4 raw = *reinterpret_cast<const float4*>(x + row * HIDDEN + col);
      const auto* p = reinterpret_cast<const __hip_bfloat16*>(&raw);
#if ADD_RESIDUAL
      const float4 res = *reinterpret_cast<const float4*>(residual + row * HIDDEN + col);
      const auto* rp = reinterpret_cast<const __hip_bfloat16*>(&res);
      __hip_bfloat16 hb[VEC];
#endif
      #pragma unroll
      for (int i = 0; i < VEC; i++) {
#if ADD_RESIDUAL
        hb[i] = __hip_bfloat16(static_cast<float>(p[i]) + static_cast<float>(rp[i]));
        const float q = static_cast<float>(hb[i]);
#else
        const float q = static_cast<float>(p[i]);
#endif
        sum = __fmaf_rn(q, q, sum);
      }
#if ADD_RESIDUAL
      *reinterpret_cast<float4*>(h_out + row * HIDDEN + col) = *reinterpret_cast<float4*>(hb);
#endif
    }
    sums[pr][rt] = sum;
    __syncthreads();
    // The original tree first pairs lanes 0..63 with 64..127. After that,
    // each wave can finish the same tree without workgroup-wide barriers.
    sum += sums[pr][rt ^ 64];
    #pragma unroll
    for (int s = 32; s; s >>= 1) sum += __shfl_xor(sum, s);
    const float rs = rsqrtf(sum / float(HIDDEN) + EPS);
    if (rt == 0) { rrms[row] = rs; scales[pr + r * PARALLEL_ROWS] = rs; }
    __syncthreads();
  }
  #pragma unroll 1
  for (int v = 0; v < HIDDEN / TILE_COLS; v++) {
    #pragma unroll
    for (int r = 0; r < TILE_ROWS / PARALLEL_ROWS; r++) {
      const int row = base_row + pr + r * PARALLEL_ROWS;
      const float rs = scales[pr + r * PARALLEL_ROWS];
      __hip_bfloat16 y[VEC];
#if ADD_RESIDUAL
      const uint4 raw = *reinterpret_cast<const uint4*>(h_out + row * HIDDEN + rt * VEC + v * VECTOR_COLS);
#else
      const uint4 raw = *reinterpret_cast<const uint4*>(x + row * HIDDEN + rt * VEC + v * VECTOR_COLS);
#endif
      const auto* xp = reinterpret_cast<const __hip_bfloat16*>(&raw);
      const float4 raw_w = *reinterpret_cast<const float4*>(weight + rt * VEC + v * VECTOR_COLS);
      const auto* wp = reinterpret_cast<const __hip_bfloat16*>(&raw_w);
      #pragma unroll
      for (int i = 0; i < VEC; i++) {
        #pragma clang fp reassociate(off) contract(off)
        y[i] = __hip_bfloat16((static_cast<float>(xp[i]) * static_cast<float>(wp[i])) * rs);
      }
      *reinterpret_cast<uint4*>(tile_at(tile,pr + r * PARALLEL_ROWS,rt * VEC)) = *reinterpret_cast<uint4*>(y);
    }
    __syncthreads();
    #pragma unroll 1
    for (int q = 0; q < TILE_ROWS * TILE_COLS / (THREADS * 8); q++) {
      const int index = tid + q * THREADS, col = index / 4, row = (index % 4) * 8;
      const int rrow = index / (TILE_COLS / 8), rcol = index % (TILE_COLS / 8) * 8;
      const float4 lo = make_float4(__uint_as_float(uint32_t(*tile_at(tile,row,col)) << 16),
        __uint_as_float(uint32_t(*tile_at(tile,row+1,col)) << 16), __uint_as_float(uint32_t(*tile_at(tile,row+2,col)) << 16),
        __uint_as_float(uint32_t(*tile_at(tile,row+3,col)) << 16));
      const float4 hi = make_float4(__uint_as_float(uint32_t(*tile_at(tile,row+4,col)) << 16),
        __uint_as_float(uint32_t(*tile_at(tile,row+5,col)) << 16), __uint_as_float(uint32_t(*tile_at(tile,row+6,col)) << 16),
        __uint_as_float(uint32_t(*tile_at(tile,row+7,col)) << 16));
      const auto rvalue = mxfp4::quantize8(mxfp4::load_bf16x4(tile_at(tile,rrow,rcol)),
        mxfp4::load_bf16x4(tile_at(tile,rrow,rcol+4)),tid & 3);
      const auto value = mxfp4::quantize8(lo,hi,tid & 3);
      mxfp4::store_fp4_8<false>(row_fp4, base_row + rrow, (v * TILE_COLS + rcol) / 2, HIDDEN / 2, rvalue.fp4);
      if ((tid & 3) == 0) mxfp4::store_scale(row_scale, base_row + rrow, (v * TILE_COLS + rcol) / 32, HIDDEN / 32, rvalue.scale);
      mxfp4::store_fp4_8<true>(col_fp4, v * TILE_COLS + col, (base_row + row) / 2, ROWS / 2, value.fp4);
      if ((tid & 3) == 0) mxfp4::store_scale(col_scale, v * TILE_COLS + col, base_row / 32, ROWS / 32, value.scale);
    }
    __syncthreads();
  }
}
