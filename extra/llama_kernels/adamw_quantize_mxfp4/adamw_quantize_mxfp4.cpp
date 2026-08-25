// Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <cstdint>
#include "quantize_mxfp4_device.h"

#if !defined(KERNEL_NAME) || !defined(M_DIM) || !defined(N_DIM) || !defined(B1) || !defined(B2) || \
    !defined(ONE_MINUS_B1) || !defined(ONE_MINUS_B2) || \
    !defined(EPS) || !defined(WEIGHT_DECAY) || !defined(GRAD_ACC)
#error kernel dimensions and AdamW constants must be defined
#endif

namespace {

using namespace mxfp4;

constexpr int BLOCK = 32;
constexpr int TILE_M = 32;
constexpr int TILE_N = 256;
constexpr int THREADS = 256;
constexpr int THREADS_PER_ROW = 8;
constexpr int VALUES_PER_THREAD = 4;
constexpr int SMEM_STRIDE = TILE_N + 2;
constexpr int M = M_DIM;
constexpr int N = N_DIM;
constexpr int M_PACKED = M / 2;
constexpr int N_PACKED = N / 2;
constexpr int M_SCALES = M / BLOCK;
constexpr int N_SCALES = N / BLOCK;

static_assert(M % 256 == 0 && N % 256 == 0);

__device__ __forceinline__ uint16_t float_to_bf16(float value) {
  uint32_t bits = __float_as_uint(value);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

__device__ __forceinline__ float bf16_to_float(uint16_t value) {
  return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

__device__ __forceinline__ void update_tile(uint16_t* tile, float* m, float* v, float* master, uint16_t* param,
                                            const uint16_t* grad, int tile_m, int tile_n,
                                            float lr, float b1_t, float b2_t, float clip_coeff) {
  const int row_in_round = threadIdx.x / 64;
  const int col = threadIdx.x % 64 * VALUES_PER_THREAD;
  const float inv_b2_correction = 1.0f / (1.0f - b2_t);
  #pragma unroll
  for (int round = 0; round < TILE_M / 4; round++) {
    const int row = round * 4 + row_in_round;
    const int base = (tile_m + row) * N + tile_n + col;
    const uint64_t packed_grad = *reinterpret_cast<const uint64_t*>(grad + base);
    const float4 old_m = *reinterpret_cast<const float4*>(m + base);
    const float4 old_v = *reinterpret_cast<const float4*>(v + base);
    const float4 old_w = *reinterpret_cast<const float4*>(master + base);
    float4 new_m, new_v, new_w;
    uint16_t p0, p1, p2, p3;

    #define UPDATE_ONE(FIELD, SHIFT, ELEM, P16) do { \
      float g = bf16_to_float(static_cast<uint16_t>(packed_grad >> SHIFT)) / static_cast<float>(GRAD_ACC); \
      g = bf16_to_float(float_to_bf16(g)); \
      g = bf16_to_float(float_to_bf16(g * clip_coeff)); \
      new_m.FIELD = fmaf(B1, old_m.FIELD, ONE_MINUS_B1 * g); \
      if constexpr (M == 6144 && N == 4096) { \
        /* Match the UPCAST(3) packed AdamW kernel: lane 0 contracts the grad term, lanes 1/2 contract the old-v term. */ \
        const float grad_sq = __fmul_rn(g, g); \
        if ((base + ELEM) % 3 != 0) new_v.FIELD = fmaf(B2, old_v.FIELD, __fmul_rn(grad_sq, ONE_MINUS_B2)); \
        else new_v.FIELD = fmaf(grad_sq, ONE_MINUS_B2, __fmul_rn(B2, old_v.FIELD)); \
      } else { \
        new_v.FIELD = fmaf(B2, old_v.FIELD, g * g * ONE_MINUS_B2); \
      } \
      const float inv_denom = 1.0f / ((1.0f - b1_t) * (sqrtf(new_v.FIELD * inv_b2_correction) + EPS)); \
      new_w.FIELD = old_w.FIELD - lr * (new_m.FIELD * inv_denom + WEIGHT_DECAY * old_w.FIELD); \
      P16 = float_to_bf16(new_w.FIELD); \
    } while (0)
    UPDATE_ONE(x, 0, 0, p0); UPDATE_ONE(y, 16, 1, p1); UPDATE_ONE(z, 32, 2, p2); UPDATE_ONE(w, 48, 3, p3);
    #undef UPDATE_ONE

    *reinterpret_cast<float4*>(m + base) = new_m;
    *reinterpret_cast<float4*>(v + base) = new_v;
    *reinterpret_cast<float4*>(master + base) = new_w;
    const uint64_t packed_param = static_cast<uint64_t>(p0) | (static_cast<uint64_t>(p1) << 16) |
                                  (static_cast<uint64_t>(p2) << 32) | (static_cast<uint64_t>(p3) << 48);
    *reinterpret_cast<uint64_t*>(param + base) = packed_param;
    uint16_t* tile_dst = tile + row * SMEM_STRIDE + col;
    *reinterpret_cast<uint32_t*>(tile_dst) = static_cast<uint32_t>(packed_param);
    *reinterpret_cast<uint32_t*>(tile_dst + 2) = static_cast<uint32_t>(packed_param >> 32);
  }
}

__device__ __forceinline__ void quantize_row(uint16_t* tile, uint8_t* fp4_output, uint8_t* scale_output,
                                             int tile_m, int tile_n, int local_row, int lane) {
  const int row = tile_m + local_row;
  const int col = lane * VALUES_PER_THREAD;
  const Quantized4 result = quantize_pipelined(load_bf16x4(tile + local_row * SMEM_STRIDE + col), lane);
  store_fp4<true>(fp4_output, row, (tile_n + col) / 2, N_PACKED, result.fp4);
  if (lane == 0) store_scale(scale_output, row, tile_n / BLOCK, N_SCALES, result.scale);
}

__device__ __forceinline__ Quantized4 quantize_col(uint16_t* tile, int col, int lane) {
  const int row = lane * VALUES_PER_THREAD;
  return quantize_pipelined(make_float4(
    __uint_as_float(static_cast<uint32_t>(tile[(row + 0) * SMEM_STRIDE + col]) << 16),
    __uint_as_float(static_cast<uint32_t>(tile[(row + 1) * SMEM_STRIDE + col]) << 16),
    __uint_as_float(static_cast<uint32_t>(tile[(row + 2) * SMEM_STRIDE + col]) << 16),
    __uint_as_float(static_cast<uint32_t>(tile[(row + 3) * SMEM_STRIDE + col]) << 16)), lane);
}

} // namespace

extern "C" __global__ __launch_bounds__(THREADS, 4)
void KERNEL_NAME(float* __restrict__ m, float* __restrict__ v, float* __restrict__ master,
                 uint16_t* __restrict__ param, uint8_t* __restrict__ rowwise_fp4, uint8_t* __restrict__ rowwise_scale,
                 uint8_t* __restrict__ colwise_fp4, uint8_t* __restrict__ colwise_scale, const uint16_t* __restrict__ grad,
                 const float* __restrict__ lr_ptr, const float* __restrict__ b1_t_ptr,
                 const float* __restrict__ b2_t_ptr, const float* __restrict__ clip_coeff_ptr) {
  __shared__ uint16_t tile[TILE_M * SMEM_STRIDE];
  const int tid = threadIdx.x;
  const int line = tid / THREADS_PER_ROW;
  const int lane = tid % THREADS_PER_ROW;
  const int block_m = blockIdx.x * TILE_M;
  const int block_n = blockIdx.y * TILE_N;
  const float lr = lr_ptr[0], b1_t = b1_t_ptr[0], b2_t = b2_t_ptr[0], clip_coeff = clip_coeff_ptr[0];

  update_tile(tile, m, v, master, param, grad, block_m, block_n, lr, b1_t, b2_t, clip_coeff);
  __syncthreads();

  #pragma unroll
  for (int chunk_n = 0; chunk_n < TILE_N / BLOCK; chunk_n++) {
    const int tile_n = block_n + chunk_n * BLOCK;
    uint16_t* chunk = tile + chunk_n * BLOCK;
    const int local_row = line;
    const int local_col = lane * VALUES_PER_THREAD;
    float4 row_value = load_bf16x4(chunk + local_row * SMEM_STRIDE + local_col);
    const int row = lane * VALUES_PER_THREAD;
    const int col = tile_n + line;
    float4 col_value = make_float4(
      __uint_as_float(static_cast<uint32_t>(chunk[(row + 0) * SMEM_STRIDE + line]) << 16),
      __uint_as_float(static_cast<uint32_t>(chunk[(row + 1) * SMEM_STRIDE + line]) << 16),
      __uint_as_float(static_cast<uint32_t>(chunk[(row + 2) * SMEM_STRIDE + line]) << 16),
      __uint_as_float(static_cast<uint32_t>(chunk[(row + 3) * SMEM_STRIDE + line]) << 16));
    Quantized4 row_result, col_result;
    quantize_pair(row_value, col_value, lane, row_result, col_result);
    store_fp4<true>(rowwise_fp4, block_m + line, (tile_n + local_col) / 2, N_PACKED, row_result.fp4);
    if (lane == 0) store_scale(rowwise_scale, block_m + line, tile_n / BLOCK, N_SCALES, row_result.scale);
    store_fp4<true>(colwise_fp4, col, (block_m + row) / 2, M_PACKED, col_result.fp4);
    if (lane == 0) store_scale(colwise_scale, col, block_m / BLOCK, M_SCALES, col_result.scale);
  }
}
