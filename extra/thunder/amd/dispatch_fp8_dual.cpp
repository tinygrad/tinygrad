#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_bf16.h>

#ifndef G_DIM
#define G_DIM 1
#endif
#ifndef M_DIM
#define M_DIM 73728
#endif
#ifndef T_DIM
#define T_DIM 16384
#endif
#ifndef D_DIM
#define D_DIM 3072
#endif
#ifndef TOPK
#define TOPK 4
#endif
#ifndef THREADS
#define THREADS 128
#endif
#ifndef NATIVE_MXFP8_CVT
#define NATIVE_MXFP8_CVT 1
#endif
#ifndef DISPATCH_DUAL_M32
#define DISPATCH_DUAL_M32 0
#endif
#ifndef DISPATCH_ROW_SI
#define DISPATCH_ROW_SI 0
#endif
#ifndef DISPATCH_APPEND_ROW_SI
#define DISPATCH_APPEND_ROW_SI 0
#endif
constexpr int BLK = 32;
constexpr int BLOCKS_PER_WG = DISPATCH_DUAL_M32 ? 1 : 4;
constexpr int ROW_TILE = BLOCKS_PER_WG * BLK;
constexpr int D_TILE = THREADS;
constexpr float FP8_MAX = 448.0f;

static_assert(M_DIM % ROW_TILE == 0, "M_DIM must be divisible by ROW_TILE");
static_assert(D_DIM % D_TILE == 0, "D_DIM must be divisible by THREADS");
static_assert(D_DIM % BLK == 0, "D_DIM must be block aligned");
#if DISPATCH_APPEND_ROW_SI
static_assert(DISPATCH_ROW_SI && DISPATCH_DUAL_M32 && THREADS == 128,
              "appended packed row scales are specialized for GPT-OSS M32/t128 dispatch");
#endif
#if NATIVE_MXFP8_CVT
typedef short short2v __attribute__((ext_vector_type(2)));
#endif

// Dispatch each pre-quantized token to its expert-grouped row and produce the columnwise MXFP8 copy used by FC1
// wgrad in the same pass. q_row/e8_row are the ordinary forward operand. q_col/si_col are exactly
// quantize_mxfp8(dequant(q_row,e8_row).T), with q_col physically (D,G*M) and si_col physically (G*M/128,D).
extern "C" __global__ __launch_bounds__(THREADS) void dispatch_fp8_dual(
    __hip_fp8_storage_t *__restrict__ q_row,
#if DISPATCH_ROW_SI
    uint32_t *__restrict__ row_si,
#else
    uint8_t *__restrict__ e8_row,
#endif
    __hip_fp8_storage_t *__restrict__ q_col,
    uint32_t *__restrict__ si_col,
    const __hip_fp8_storage_t *__restrict__ q_in,
    const uint8_t *__restrict__ e8_in,
    const int *__restrict__ src_row) {
  const int tid = threadIdx.x;
#if DISPATCH_APPEND_ROW_SI
  constexpr int DISPATCH_WGS = G_DIM * (M_DIM / ROW_TILE) * (D_DIM / D_TILE);
  if (blockIdx.x >= DISPATCH_WGS) {
    constexpr long long ROWS = (long long)G_DIM * M_DIM;
    constexpr int SCALE_GROUPS = D_DIM / 128;
    const long long idx = (long long)(blockIdx.x - DISPATCH_WGS) * THREADS + tid;
    if (idx < ROWS * SCALE_GROUPS) {
      const int scale_group = idx / ROWS;
      const int grouped_row = idx - (long long)scale_group * ROWS;
      const int group = grouped_row / M_DIM;
      const int src = src_row[grouped_row];
      uint32_t packed = 0;
      if (src >= 0) {
        const int token = src / TOPK;
        packed = *reinterpret_cast<const uint32_t *>(
          &e8_in[((long long)group * T_DIM + token) * (D_DIM / 32) + scale_group * 4]);
      }
      row_si[idx] = packed;
    }
    return;
  }
#endif
  int tile = blockIdx.x;
  const int tile_d = tile % (D_DIM / D_TILE);
  tile /= D_DIM / D_TILE;
  const int row_tile = tile % (M_DIM / ROW_TILE);
  const int group = tile / (M_DIM / ROW_TILE);
  const int d = tile_d * D_TILE + tid;
  const int m0 = row_tile * ROW_TILE;
  const int m_group = m0 / (4 * BLK);
  const int first_sub = (m0 / BLK) & 3;

  __shared__ int src_rows[ROW_TILE];
  if (tid < ROW_TILE) src_rows[tid] = src_row[(long long)group * M_DIM + m0 + tid];
  __syncthreads();

  uint32_t packed_e8 = 0;
  #pragma unroll
  for (int block = 0; block < BLOCKS_PER_WG; block++) {
    const int sub = first_sub + block;
    __hip_fp8_storage_t out[BLK];
    float vals[BLK];
    float amax = 0.0f;

    #pragma unroll
    for (int mm = 0; mm < BLK; mm++) {
      const int m = m0 + block * BLK + mm;
      const int src = src_rows[block * BLK + mm];
      const bool valid = src >= 0;
      const int token = valid ? src / TOPK : 0;
      const long long qi = ((long long)group * T_DIM + token) * D_DIM + d;
      const uint8_t qv = valid ? q_in[qi] : 0;

      // The scale address is half-wave uniform. Repeated lane loads coalesce/cache well and are cheaper here than
      // putting a shuffle dependency in every one of the 128 inner iterations.
      const unsigned src_scale = valid ? e8_in[((long long)group * T_DIM + token) * (D_DIM / BLK) + d / BLK] : 0;

      q_row[((long long)group * M_DIM + m) * D_DIM + d] = qv;
#if DISPATCH_ROW_SI
      // Packed row scales are emitted by appended row-SI-only workgroups (or the standalone fallback): this
      // D-parallel workgroup cannot coalesce the scale-major destination without serializing leader lanes.
#else
      if ((tid & 31) == 0)
        e8_row[((long long)group * M_DIM + m) * (D_DIM / BLK) + d / BLK] = (uint8_t)src_scale;
#endif

      const __hip_fp8_e4m3 fq = __builtin_bit_cast(__hip_fp8_e4m3, qv);
      const float dqscale = __builtin_bit_cast(float, src_scale ? src_scale << 23 : 0x00400000u);
      const float v = (float)(__hip_bfloat16)((float)fq * dqscale);
      vals[mm] = v;
      amax = fmaxf(amax, fabsf(v));
    }

#if NATIVE_MXFP8_CVT
    int e8 = (int)((__builtin_bit_cast(unsigned, amax) >> 23) & 0xFFu);
    e8 = max(0, min(254, e8));
    const float scale_f = __builtin_bit_cast(float, (unsigned)e8 << 23);
    #pragma unroll
    for (int mm = 0; mm < BLK; mm += 2) {
      short2v acc = {0, 0};
      acc = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(acc, vals[mm], vals[mm + 1], scale_f, false);
      const unsigned char *rb = reinterpret_cast<const unsigned char *>(&acc);
      out[mm] = rb[0];
      out[mm + 1] = rb[1];
    }
#else
    int e8 = (int)floorf(log2f(fmaxf(amax, 1e-38f))) + 127;
    e8 = max(0, min(254, e8));
    const float qscale = exp2f((float)(127 - e8));
    #pragma unroll
    for (int mm = 0; mm < BLK; mm++)
      out[mm] = __hip_cvt_float_to_fp8(fmaxf(-FP8_MAX, fminf(FP8_MAX, vals[mm] * qscale)), __HIP_SATFINITE, __HIP_E4M3);
#endif

    const long long col_base = (long long)d * (G_DIM * M_DIM) + (long long)group * M_DIM + m0 + block * BLK;
    *reinterpret_cast<uint4 *>(&q_col[col_base]) = *reinterpret_cast<uint4 *>(&out[0]);
    *reinterpret_cast<uint4 *>(&q_col[col_base + 16]) = *reinterpret_cast<uint4 *>(&out[16]);
    #if DISPATCH_DUAL_M32
    reinterpret_cast<uint8_t *>(si_col)[(((long long)group * (M_DIM / (4 * BLK)) + m_group) * D_DIM + d) * 4 + sub] = (uint8_t)e8;
    #else
    packed_e8 |= (uint32_t)(uint8_t)e8 << (8 * sub);
    #endif
  }
#if !DISPATCH_DUAL_M32
  si_col[((long long)group * (M_DIM / ROW_TILE) + m_group) * D_DIM + d] = packed_e8;
#endif
}
