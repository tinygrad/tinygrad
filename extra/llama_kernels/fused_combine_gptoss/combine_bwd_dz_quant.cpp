#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

// GPT-OSS local shard: routed dZ, both MX block directions, and ordered bias partials.
// Every consumer uses a quantized companion or bias partial: no full BF16 intermediate.
constexpr int M = 73728, N = 3072, D = 2880, K = 4;
constexpr int TM = 32, TN = 128, STRIDE = TN + 1;
#ifndef COMBINE_COL_SI
#define COMBINE_COL_SI 0
#endif
typedef short short2v __attribute__((ext_vector_type(2)));

__device__ __forceinline__ unsigned cvt4(float a, float b, float c, float d, int e) {
  const float sf = __builtin_bit_cast(float, (unsigned)e << 23);
  short2v lo = {0, 0}, hi = {0, 0};
  lo = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(lo, a, b, sf, false);
  hi = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(hi, c, d, sf, false);
  return (__builtin_bit_cast(unsigned, lo) & 65535u) | (__builtin_bit_cast(unsigned, hi) << 16);
}

template<int CTRL> __device__ __forceinline__ float dpp_move(float x) {
  return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), CTRL, 0xf, 0xf, false));
}

extern "C" __global__ __launch_bounds__(TN) void combine_bwd_dz_quant(
    unsigned char* __restrict__ row_q, unsigned char* __restrict__ row_e8,
    unsigned char* __restrict__ col_q, unsigned char* __restrict__ col_e8,
    float* __restrict__ bias_partial, const __hip_bfloat16* __restrict__ d_out,
    const int* __restrict__ src_row, const float* __restrict__ weights) {
  const int tid = threadIdx.x, tile_m = blockIdx.x / (N / TN), tile_n = blockIdx.x % (N / TN);
  const int n = tile_n * TN + tid;
  __shared__ __hip_bfloat16 lds[TM * STRIDE];
  float amax = 0.0f, tile_sum = 0.0f;
  #pragma unroll
  for (int mm = 0; mm < TM; mm++) {
    const int m = tile_m * TM + mm, sr = src_row[m];
    __hip_bfloat16 bv = (__hip_bfloat16)0.0f;
    // Do not form input addresses for sentinel rows or padded columns.
    if (sr >= 0 && n < D) bv = (__hip_bfloat16)((float)d_out[(long long)(sr / K) * D + n] * weights[sr]);
    const float v = (float)bv;  // mandatory BF16 rounding before either quantization or the bias sum
    lds[mm * STRIDE + tid] = bv;
    amax = fmaxf(amax, fabsf(v));
    tile_sum += v;  // same sequential 32-row FP32 sum as transpose_quantize_mxfp8_bias
    float ra = fabsf(v);
    ra = fmaxf(ra, dpp_move<0xb1>(ra));
    ra = fmaxf(ra, dpp_move<0x4e>(ra));
    ra = fmaxf(ra, dpp_move<0x141>(ra));
    ra = fmaxf(ra, dpp_move<0x140>(ra));
    ra = fmaxf(ra, __shfl_xor(ra, 16, 32));
    const int rs = min(254, (int)((__builtin_bit_cast(unsigned, ra) >> 23) & 255));
    if ((tid & 31) == 0) row_e8[(long long)m * (N / 32) + n / 32] = rs;
    const float other = dpp_move<0xb1>(v);
    if ((tid & 1) == 0) {
      short2v pk = {0, 0};
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(pk, v, other, __builtin_bit_cast(float, (unsigned)rs << 23), false);
      *reinterpret_cast<unsigned short*>(&row_q[(long long)m * N + n]) = (unsigned short)__builtin_bit_cast(unsigned, pk);
    }
  }
  const int e8 = min(254, (int)((__builtin_bit_cast(unsigned, amax) >> 23) & 255));
#if COMBINE_COL_SI
  // Each 32-row tile owns one byte of the packed 128-row scale word; byte stores have disjoint ownership.
  col_e8[((long long)(tile_m / 4) * N + n) * 4 + tile_m % 4] = e8;
#else
  col_e8[(long long)n * (M / 32) + tile_m] = e8;
#endif
  bias_partial[(long long)tile_m * N + n] = tile_sum;
  __syncthreads();
  // Same coalesced transposed writes and four-value quantization as the separate TQ kernel.
  const int lane = tid & 63, wid = tid >> 6, mm4 = (lane & 7) * 4, nbase = wid * 64 + (lane >> 3);
  #pragma unroll
  for (int j = 0; j < 8; j++) {
    const int ln = nbase + j * 8, scale = __shfl(e8, (lane >> 3) + j * 8, 64);
    const unsigned packed = cvt4((float)lds[(mm4 + 0) * STRIDE + ln], (float)lds[(mm4 + 1) * STRIDE + ln],
                                 (float)lds[(mm4 + 2) * STRIDE + ln], (float)lds[(mm4 + 3) * STRIDE + ln], scale);
    *reinterpret_cast<unsigned*>(&col_q[(long long)(tile_n * TN + ln) * M + tile_m * TM + mm4]) = packed;
  }
}
