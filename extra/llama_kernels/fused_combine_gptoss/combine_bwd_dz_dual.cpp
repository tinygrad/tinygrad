#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#ifndef G_DIM
#define G_DIM 1
#endif
#ifndef M_DIM
#define M_DIM 9216
#endif
#ifndef T_DIM
#define T_DIM 2048
#endif
#ifndef D_DIM
#define D_DIM 2880
#endif
#ifndef P_DIM
#define P_DIM 3072
#endif
#ifndef K_DIM
#define K_DIM 4
#endif
#ifndef THREADS_PER_WG
#define THREADS_PER_WG 256
#endif
#ifndef NATIVE_MXFP8_CVT
#define NATIVE_MXFP8_CVT 1
#endif

typedef short short2v __attribute__((ext_vector_type(2)));

constexpr int BLK = 32;
constexpr int ROW_GROUP = 4 * BLK;        // four scale bytes make one packed uint32
constexpr int TILE_N = THREADS_PER_WG;
constexpr int TOTAL_M = G_DIM * M_DIM;
constexpr int N_TILES = P_DIM / TILE_N;
constexpr int SCALE_BLOCKS = P_DIM / BLK;
constexpr float FP8_MAX = 448.0f;

static_assert(THREADS_PER_WG == 256, "combine_bwd_dz_dual requires 256 threads");
static_assert(M_DIM % ROW_GROUP == 0, "per-device grouped rows must be a multiple of 128");
static_assert(P_DIM % TILE_N == 0, "physical width must be a multiple of 256");
static_assert(D_DIM <= P_DIM && D_DIM % BLK == 0, "real width must use whole MX blocks");

__device__ __forceinline__ int block_e8(float amax) {
#if NATIVE_MXFP8_CVT
  int e8 = (int)((__builtin_bit_cast(unsigned, amax) >> 23) & 0xFFu);
  return min(e8, 254);
#else
  int e8 = (int)floorf(log2f(fmaxf(amax, 1e-38f))) + 127;
  return max(0, min(254, e8));
#endif
}

__device__ __forceinline__ unsigned short cvt_pair(float x0, float x1, int e8) {
#if NATIVE_MXFP8_CVT
  const float scale_f = __builtin_bit_cast(float, (unsigned)e8 << 23);
  short2v packed = {0, 0};
  packed = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(packed, x0, x1, scale_f, false);
  const unsigned char *bytes = reinterpret_cast<const unsigned char *>(&packed);
  return (unsigned short)bytes[0] | ((unsigned short)bytes[1] << 8);
#else
  const float qscale = exp2f((float)(127 - e8));
  const unsigned char q0 = __hip_cvt_float_to_fp8(fmaxf(-FP8_MAX, fminf(FP8_MAX, x0 * qscale)), __HIP_SATFINITE, __HIP_E4M3);
  const unsigned char q1 = __hip_cvt_float_to_fp8(fmaxf(-FP8_MAX, fminf(FP8_MAX, x1 * qscale)), __HIP_SATFINITE, __HIP_E4M3);
  return (unsigned short)q0 | ((unsigned short)q1 << 8);
#endif
}

// One workgroup owns (128 grouped rows x 256 hidden columns). It computes each bf16 dZ value once, and emits:
//   * bf16 dZ (TOTAL_M,P_DIM),
//   * rowwise MXFP8 dZ + raw e8 for down dgrad, and
//   * columnwise MXFP8 dZ^T + packed e8 for down wgrad.
// The column product is byte-equal to transpose_quantize_mxfp8(dz_bf16): a thread owns exactly the same ordered
// 32-value column block, consumes the bf16-rounded products, and uses the same amax/e8/cvt code.
extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void
combine_bwd_dz_dual(__hip_bfloat16* __restrict__ dz_bf16,
                    __hip_fp8_storage_t* __restrict__ dz_row,
                    uint8_t* __restrict__ dz_row_e8,
                    __hip_fp8_storage_t* __restrict__ dz_col, // (P_DIM,TOTAL_M)
                    uint32_t* __restrict__ dz_col_si,         // (TOTAL_M/128,P_DIM)
                    const __hip_bfloat16* __restrict__ d_out,
                    const int* __restrict__ src_row,
                    const float* __restrict__ weights) {
  const int tid = threadIdx.x;
  const int tile_m128 = blockIdx.x / N_TILES;
  const int tile_n = blockIdx.x - tile_m128 * N_TILES;
  const int gm0 = tile_m128 * ROW_GROUP;
  const int n = tile_n * TILE_N + tid;
  const int lane = tid & (BLK - 1);

  // Loading routing metadata once per row avoids repeating it for all 256 columns.
  __shared__ int s_src[ROW_GROUP];
  __shared__ float s_weight[ROW_GROUP];
  if (tid < ROW_GROUP) {
    const int gm = gm0 + tid;
    const int src = src_row[gm];
    s_src[tid] = src;
    float weight = 0.0f;
    if (src >= 0) {
      const int g = gm / M_DIM;
      const int t = src / K_DIM;
      const int j = src - t * K_DIM;
      weight = weights[((long long)g * T_DIM + t) * K_DIM + j];
    }
    s_weight[tid] = weight;
  }
  __syncthreads();

  uint32_t packed_col_e8 = 0;
#pragma unroll
  for (int sub = 0; sub < 4; sub++) {
    float vals[BLK];
    float col_amax = 0.0f;
#pragma unroll
    for (int mm = 0; mm < BLK; mm++) {
      const int r = sub * BLK + mm;
      const int gm = gm0 + r;
      __hip_bfloat16 vb = (__hip_bfloat16)0.0f;
      if (n < D_DIM && s_src[r] >= 0) {
        const int g = gm / M_DIM;
        const int t = s_src[r] / K_DIM;
        const long long in_idx = ((long long)g * T_DIM + t) * D_DIM + n;
        // Required RNE boundary. Both stored bf16 and both quantizers consume this rounded value.
        vb = (__hip_bfloat16)((float)d_out[in_idx] * s_weight[r]);
      }
      dz_bf16[(long long)gm * P_DIM + n] = vb;
      const float v = (float)vb;
      vals[mm] = v;
      col_amax = fmaxf(col_amax, fabsf(v));

      // Rowwise MX block: each width-32 subgroup reduces one row/block.
      float row_amax = fabsf(v);
#pragma unroll
      for (int off = BLK / 2; off > 0; off >>= 1)
        row_amax = fmaxf(row_amax, __shfl_down(row_amax, off, BLK));
      row_amax = __shfl(row_amax, 0, BLK);
      const int row_e8 = block_e8(row_amax);
      if (lane == 0)
        dz_row_e8[(long long)gm * SCALE_BLOCKS + n / BLK] = (uint8_t)row_e8;
      const float other = __shfl_down(v, 1, BLK);
      if ((lane & 1) == 0)
        *reinterpret_cast<unsigned short*>(&dz_row[(long long)gm * P_DIM + n]) = cvt_pair(v, other, row_e8);
    }

    // Columnwise MX block: identical loop order and conversion to transpose_quantize_mxfp8.
    const int col_e8 = block_e8(col_amax);
    __hip_fp8_storage_t out[BLK];
#pragma unroll
    for (int mm = 0; mm < BLK; mm += 2) {
      const unsigned short pair = cvt_pair(vals[mm], vals[mm + 1], col_e8);
      out[mm] = (unsigned char)pair;
      out[mm + 1] = (unsigned char)(pair >> 8);
    }
    const long long obase = (long long)n * TOTAL_M + gm0 + sub * BLK;
    *reinterpret_cast<uint4*>(&dz_col[obase]) = *reinterpret_cast<uint4*>(&out[0]);
    *reinterpret_cast<uint4*>(&dz_col[obase + 16]) = *reinterpret_cast<uint4*>(&out[16]);
    packed_col_e8 |= (uint32_t)(uint8_t)col_e8 << (8 * sub);
  }
  dz_col_si[(long long)tile_m128 * P_DIM + n] = packed_col_e8;
}
