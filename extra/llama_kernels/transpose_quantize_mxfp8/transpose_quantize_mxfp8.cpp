#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_bf16.h>

#ifndef M_DIM
#define M_DIM 8192
#endif
#ifndef N_DIM
#define N_DIM 14336
#endif
#ifndef SRC_N_DIM
#define SRC_N_DIM N_DIM
#endif
#ifndef THREADS_PER_WG
#define THREADS_PER_WG 256
#endif

// NATIVE_MXFP8_CVT: use the gfx950 hardware scaled-fp8 conversion (v_cvt_scalef32_pk_fp8_f32) + exponent-bit
// e8m0 extraction, instead of the software floor(log2f)+exp2f block-scale + scalar __hip_cvt_float_to_fp8.
// Byte-exact by construction (both round-to-nearest-even the SAME exact value v*2^(127-e8); block-normalization
// keeps scaled values in [1,2) so saturation never triggers). Default off; flip once proven.
#ifndef NATIVE_MXFP8_CVT
#define NATIVE_MXFP8_CVT 0
#endif
#ifndef TQ_BIAS_PARTIAL
#define TQ_BIAS_PARTIAL 0
#endif
#ifndef TQ_ROW_OUTPUT
#define TQ_ROW_OUTPUT 0
#endif
#ifndef TQ_DIRECT_FULL_LOAD
#define TQ_DIRECT_FULL_LOAD 0
#endif
#ifndef TQ_ROW_SI_BYTE_STORE
#define TQ_ROW_SI_BYTE_STORE 0
#endif
#ifndef TQ_ROW_E8_STORE
#define TQ_ROW_E8_STORE 1
#endif
#ifndef TQ_COL_SI_BYTE_STORE
#define TQ_COL_SI_BYTE_STORE 0
#endif
#ifndef GPTOSS_REQUANT_PACKED_INPUT
#define GPTOSS_REQUANT_PACKED_INPUT 0
#endif
#ifndef GPTOSS_REQUANT_SI_ONLY
#define GPTOSS_REQUANT_SI_ONLY 0
#endif
#ifndef GPTOSS_REQUANT_TRIM_DGRAD
#define GPTOSS_REQUANT_TRIM_DGRAD 0
#endif
#ifndef GPTOSS_REQUANT_DIRECT_BF16
#define GPTOSS_REQUANT_DIRECT_BF16 0
#endif
#ifndef GPTOSS_REQUANT_BFE_SCALE
#define GPTOSS_REQUANT_BFE_SCALE 0
#endif
#ifndef GPTOSS_REQUANT_PAIR
#define GPTOSS_REQUANT_PAIR 0
#endif
#if NATIVE_MXFP8_CVT
typedef short short2v __attribute__((ext_vector_type(2)));
#endif

constexpr int BLK = 32;
constexpr int TILE_M = BLK;                 // one mxfp8 block along M per tile
constexpr int TILE_N = THREADS_PER_WG;      // 256, one output column per thread
constexpr int LDS_STRIDE = TILE_N + 1;      // +1 pad: stride 257 ≡ 1 (mod 32) -> conflict-free column reads
constexpr int N_TILES_N = N_DIM / TILE_N;
constexpr float FP8_MAX = 448.0f;

static_assert(M_DIM % TILE_M == 0, "M_DIM must be a multiple of 32");
static_assert(N_DIM % TILE_N == 0, "N_DIM must be a multiple of 256");
static_assert(SRC_N_DIM <= N_DIM, "source columns must fit in padded output");

__device__ __forceinline__ uint32_t tq_cvt_four(float v0, float v1, float v2, float v3, int e8) {
#if NATIVE_MXFP8_CVT
  const float sf = __builtin_bit_cast(float, (unsigned)e8 << 23);
  short2v a = {0, 0}, b = {0, 0};
  a = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(a, v0, v1, sf, false);
  b = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(b, v2, v3, sf, false);
  const unsigned char *ra = reinterpret_cast<const unsigned char *>(&a);
  const unsigned char *rb = reinterpret_cast<const unsigned char *>(&b);
  return (uint32_t)ra[0] | ((uint32_t)ra[1] << 8) | ((uint32_t)rb[0] << 16) | ((uint32_t)rb[1] << 24);
#else
  const float qscale = exp2f((float)(127 - e8));
  const auto cvt = [qscale](float v) {
    return __hip_cvt_float_to_fp8(fmaxf(-FP8_MAX, fminf(FP8_MAX, v * qscale)), __HIP_SATFINITE, __HIP_E4M3);
  };
  return (uint32_t)cvt(v0) | ((uint32_t)cvt(v1) << 8) | ((uint32_t)cvt(v2) << 16) | ((uint32_t)cvt(v3) << 24);
#endif
}

#if TQ_ROW_OUTPUT
__device__ __forceinline__ uint16_t tq_cvt_two(float v0, float v1, int e8) {
#if NATIVE_MXFP8_CVT
  const float sf = __builtin_bit_cast(float, (unsigned)e8 << 23);
  short2v a = {0, 0};
  a = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(a, v0, v1, sf, false);
  const unsigned char *ra = reinterpret_cast<const unsigned char *>(&a);
  return (uint16_t)ra[0] | ((uint16_t)ra[1] << 8);
#else
  const float qscale = exp2f((float)(127 - e8));
  const auto cvt = [qscale](float v) {
    return __hip_cvt_float_to_fp8(fmaxf(-FP8_MAX, fminf(FP8_MAX, v * qscale)), __HIP_SATFINITE, __HIP_E4M3);
  };
  return (uint16_t)cvt(v0) | ((uint16_t)cvt(v1) << 8);
#endif
}
#endif

#ifndef TRANSPOSE_REQUANTIZE
extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void
transpose_quantize_mxfp8(__hip_fp8_storage_t* __restrict__ q,        // (N_DIM, M_DIM)
                         uint8_t* __restrict__ e8_out,               // (N_DIM, M_DIM/32)
#if TQ_BIAS_PARTIAL
                         float* __restrict__ bias_partial,           // (M_DIM/32, N_DIM), sum of each BF16 32-row tile
#endif
#if TQ_ROW_OUTPUT
                         __hip_fp8_storage_t* __restrict__ row_q,    // (M_DIM, N_DIM)
                         uint8_t* __restrict__ row_e8,               // (M_DIM, N_DIM/32)
                         uint32_t* __restrict__ row_si,              // (N_DIM/128, M_DIM)
#endif
                         const __hip_bfloat16* __restrict__ g)       // (M_DIM, N_DIM)
{
  const int tid    = threadIdx.x;
  const int tile_m = blockIdx.x / N_TILES_N;     // which 32-block along M
  const int tile_n = blockIdx.x % N_TILES_N;
  const int n = tile_n * TILE_N + tid;
  __shared__ __hip_bfloat16 lds[TILE_M * LDS_STRIDE];
  float amax = 0.0f, tile_sum = 0.0f;
  #pragma unroll
  for (int mm = 0; mm < TILE_M; mm++) {
#if TQ_DIRECT_FULL_LOAD && SRC_N_DIM == N_DIM
    // The launch grid contains exactly N_DIM/TILE_N complete tiles, hence n is in bounds. Spell this out because
    // LLVM cannot derive the runtime grid bound and otherwise emits the same n<SRC_N_DIM select for all 32 loads.
    const __hip_bfloat16 bv = g[(long long)(tile_m * TILE_M + mm) * SRC_N_DIM + n];
#else
    const __hip_bfloat16 bv = n < SRC_N_DIM ? g[(long long)(tile_m * TILE_M + mm) * SRC_N_DIM + n] : (__hip_bfloat16)0.0f;
#endif
    const float v = (float)bv;
    lds[mm * LDS_STRIDE + tid] = bv;
    amax = fmaxf(amax, fabsf(v));
    tile_sum += v;
  }
#if NATIVE_MXFP8_CVT
  // e8m0 = biased exponent field of amax; bit-exact vs floorf(log2f)+127 for normals, agrees after clamp for 0/subnormal
  int e8 = (int)((__builtin_bit_cast(unsigned, amax) >> 23) & 0xFFu);
  e8 = max(0, min(254, e8));
#else
  int e8 = (int)floorf(log2f(fmaxf(amax, 1e-38f))) + 127;
  e8 = max(0, min(254, e8));
#endif

#if TQ_COL_SI_BYTE_STORE
  // The dense MX GEMM reads only the packed scale layout. Four neighboring 32-row tiles own distinct bytes of
  // one packed word, so write those bytes in final layout and avoid the raw-e8 buffer plus mx_pack kernel.
  reinterpret_cast<uint8_t *>(e8_out)[(((long long)(tile_m / 4) * N_DIM + n) * 4) + tile_m % 4] = (uint8_t)e8;
#else
  e8_out[n * (M_DIM / BLK) + tile_m] = (uint8_t)e8;
#endif
#if TQ_BIAS_PARTIAL
  bias_partial[(long long)tile_m * N_DIM + n] = tile_sum;
#endif
  __syncthreads();
  // Each wave writes eight transposed rows at a time; eight adjacent lanes cover one complete 32-byte row.
  const int lane = tid & 63, wid = tid >> 6, mm4 = (lane & 7) * 4, nbase = wid * 64 + (lane >> 3);
  #pragma unroll
  for (int j = 0; j < 8; j++) {
    const int local_n = nbase + j * 8;
    const int src_lane = (lane >> 3) + j * 8;
    const int out_e8 = __shfl(e8, src_lane, 64);
    const uint32_t packed = tq_cvt_four((float)lds[(mm4 + 0) * LDS_STRIDE + local_n],
                                        (float)lds[(mm4 + 1) * LDS_STRIDE + local_n],
                                        (float)lds[(mm4 + 2) * LDS_STRIDE + local_n],
                                        (float)lds[(mm4 + 3) * LDS_STRIDE + local_n], out_e8);
    const long long obase = (long long)(tile_n * TILE_N + local_n) * M_DIM + tile_m * TILE_M + mm4;
    *reinterpret_cast<uint32_t *>(&q[obase]) = packed;
  }
#if TQ_ROW_OUTPUT
  // The same 32x256 LDS tile also contains eight rowwise MX blocks for each row. A wave owns one row at a time;
  // four 16-lane groups each process a 32-value block as packed pairs, so both q and packed scales are coalesced.
  static_assert(THREADS_PER_WG == 128 || THREADS_PER_WG == 256,
                "dual row output requires complete 128-column scale superblocks");
  const int lane16 = lane & 15, group16 = lane >> 4;
  #pragma unroll
  for (int mb = 0; mb < 256 / THREADS_PER_WG; mb++) {
   #pragma unroll
   for (int rr = 0; rr < 8; rr++) {
    const int local_m = mb * (THREADS_PER_WG / 8) + wid * 8 + rr;
    const int m = tile_m * TILE_M + local_m;
    #pragma unroll
    for (int half = 0; half < THREADS_PER_WG / 128; half++) {
      const int local_n0 = half * 128 + group16 * BLK + lane16 * 2;
      const float v0 = (float)lds[local_m * LDS_STRIDE + local_n0];
      const float v1 = (float)lds[local_m * LDS_STRIDE + local_n0 + 1];
      float row_amax = fmaxf(fabsf(v0), fabsf(v1));
      #pragma unroll
      for (int off = 8; off; off >>= 1) row_amax = fmaxf(row_amax, __shfl_down(row_amax, off, 16));
      int row_scale = (int)((__builtin_bit_cast(unsigned, row_amax) >> 23) & 0xFFu);
      row_scale = max(0, min(254, row_scale));
      row_scale = __shfl(row_scale, group16 * 16, 64);
      const long long row_base = (long long)m * N_DIM + tile_n * TILE_N + local_n0;
      *reinterpret_cast<uint16_t *>(&row_q[row_base]) = tq_cvt_two(v0, v1, row_scale);
#if TQ_ROW_E8_STORE
      if (lane16 == 0) row_e8[(long long)m * (N_DIM / BLK) + tile_n * (THREADS_PER_WG / BLK) + half * 4 + group16] = (uint8_t)row_scale;
#endif
#if TQ_ROW_SI_BYTE_STORE
      // Each 16-lane subgroup already has its scale. Let its leader write the corresponding byte of the packed
      // word directly, avoiding four full-wave shuffles whose result was consumed only by lane zero.
      if (lane16 == 0) reinterpret_cast<uint8_t *>(row_si)[
        (((long long)(tile_n * (THREADS_PER_WG / 128) + half) * M_DIM + m) * 4) + group16] = (uint8_t)row_scale;
#else
      const uint32_t e0 = (uint32_t)__shfl(row_scale, 0, 64);
      const uint32_t e1 = (uint32_t)__shfl(row_scale, 16, 64);
      const uint32_t e2 = (uint32_t)__shfl(row_scale, 32, 64);
      const uint32_t e3 = (uint32_t)__shfl(row_scale, 48, 64);
      if (lane == 0) row_si[(long long)(tile_n * (THREADS_PER_WG / 128) + half) * M_DIM + m] = e0 | (e1 << 8) | (e2 << 16) | (e3 << 24);
#endif
    }
   }
  }
#endif
}
#else

#ifndef E_DIM
#define E_DIM 32
#endif
#ifndef REQUANT_M32
#define REQUANT_M32 0
#endif

// The exact GPT-OSS FC1 dgrad contracts only the first 5760 columns of W^T and only forms the first 2880
// output rows.  Its physical operands remain (32,3072,5888), but q_out[:,2880:,:] and q_out[:,:,:5760:] plus
// their packed scales are never read.  Decode a compact launch grid over just that live rectangle; the final
// 128-thread N tile is half-full.  This is deliberately unavailable to the generic requantize entry point.
#if GPTOSS_REQUANT_TRIM_DGRAD
static_assert(REQUANT_M32 && GPTOSS_REQUANT_PACKED_INPUT && GPTOSS_REQUANT_SI_ONLY,
              "trimmed requant is only valid for the packed SI-only M32 path");
static_assert(E_DIM == 32 && M_DIM == 5888 && N_DIM == 3072 && THREADS_PER_WG == 128,
              "trimmed requant is specialized for the exact GPT-OSS FC1 weight");
constexpr int REQUANT_REAL_M = 5760;
constexpr int REQUANT_REAL_N = 2880;
constexpr int REQUANT_N_TILES = N_TILES_N;
constexpr int REQUANT_M32_TILES = REQUANT_REAL_M / BLK;
#else
constexpr int REQUANT_N_TILES = N_TILES_N;
constexpr int REQUANT_M32_TILES = M_DIM / BLK;
#endif

#if GPTOSS_REQUANT_DIRECT_BF16
static_assert(REQUANT_M32 && GPTOSS_REQUANT_PACKED_INPUT && GPTOSS_REQUANT_SI_ONLY && NATIVE_MXFP8_CVT &&
              E_DIM == 32 && N_DIM == 3072 && THREADS_PER_WG == 128 &&
              ((GPTOSS_REQUANT_TRIM_DGRAD && M_DIM == 5888) || (!GPTOSS_REQUANT_TRIM_DGRAD && M_DIM == 3072)),
              "direct scaled FP8-to-BF16 is restricted to exact GPT-OSS FC1/down dgrad operands");
#endif

constexpr int REQUANT_M = 4 * BLK;
constexpr int REQUANT_M_GROUPS = M_DIM / REQUANT_M;

static_assert(M_DIM % REQUANT_M == 0, "M_DIM must be a multiple of 128");


extern "C" __global__ __launch_bounds__(GPTOSS_REQUANT_PAIR ? 64 : THREADS_PER_WG) void
transpose_requantize_mxfp8(__hip_fp8_storage_t* __restrict__ q_out,        // (E_DIM, N_DIM, M_DIM)
                           uint8_t* __restrict__ e8_out,                   // (E_DIM, N_DIM, M_DIM/32)
                           uint32_t* __restrict__ si_out,                  // (E_DIM, M_DIM/128, N_DIM)
#if GPTOSS_REQUANT_PACKED_INPUT
                           const uint32_t* __restrict__ si_in,             // (E_DIM, N_DIM/128, M_DIM)
#endif
                           const __hip_fp8_storage_t* __restrict__ q_in,   // (E_DIM, M_DIM, N_DIM)
                           const uint8_t* __restrict__ e8_in)              // (E_DIM, M_DIM, N_DIM/32)
{
#if GPTOSS_REQUANT_PAIR
  static_assert(E_DIM == 32 && M_DIM == 3072 && N_DIM == 3072 && THREADS_PER_WG == 128 &&
                REQUANT_M32 && GPTOSS_REQUANT_PACKED_INPUT && GPTOSS_REQUANT_SI_ONLY &&
                GPTOSS_REQUANT_DIRECT_BF16 && GPTOSS_REQUANT_BFE_SCALE && !GPTOSS_REQUANT_TRIM_DGRAD,
                "paired requant is qualified only for the exact GPT-OSS down weight");
  constexpr unsigned M = M_DIM, N = N_DIM;
  const unsigned tile_n = blockIdx.x % 24, tile_m = blockIdx.x / 24 % 96, expert = blockIdx.x / (24 * 96);
  const unsigned col = tile_n * 128 + threadIdx.x * 2;
  float amax0 = 0.f, amax1 = 0.f;
  float vals[2][32];
  // Each lane owns two adjacent columns. Both lanes of the scaled FP8-to-BF16 conversion are used;
  // column maxima and output quantization remain separate, with the same BF16 rounding boundary.
  #pragma unroll
  for (unsigned r = 0; r < 32; r++) {
    const unsigned m = tile_m * 32 + r;
    const unsigned bits = *reinterpret_cast<const unsigned short*>(q_in + ((size_t)expert * M + m) * N + col);
    const unsigned sw = si_in[((size_t)expert * 24 + tile_n) * M + m];
    unsigned scale;
    const unsigned shift = (threadIdx.x / 16) * 8;
    asm("v_bfe_u32 %0, %1, %2, 8" : "=v"(scale) : "s"(sw), "v"(shift));
    const float sf = __builtin_bit_cast(float, max(scale << 23, 0x00400000u));
    const auto pair = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(bits, sf, false);
    vals[0][r] = (float)pair[0]; vals[1][r] = (float)pair[1];
    amax0 = fmaxf(amax0, fabsf((float)pair[0])); amax1 = fmaxf(amax1, fabsf((float)pair[1]));
  }
  const unsigned scales[2] = {min(254u, (__builtin_bit_cast(unsigned, amax0) >> 23) & 255u),
                              min(254u, (__builtin_bit_cast(unsigned, amax1) >> 23) & 255u)};
  #pragma unroll
  for (unsigned c = 0; c < 2; c++) {
    unsigned char packed[32];
    const float sf = __builtin_bit_cast(float, scales[c] << 23);
    #pragma unroll
    for (unsigned r = 0; r < 32; r += 2) {
      short2v acc = {0, 0};
      acc = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(acc, vals[c][r], vals[c][r + 1], sf, false);
      const unsigned char *bytes = reinterpret_cast<const unsigned char*>(&acc);
      packed[r] = bytes[0]; packed[r + 1] = bytes[1];
    }
    const size_t offset = ((size_t)expert * N + col + c) * M + tile_m * 32;
    *reinterpret_cast<uint4*>(q_out + offset) = *reinterpret_cast<uint4*>(packed);
    *reinterpret_cast<uint4*>(q_out + offset + 16) = *reinterpret_cast<uint4*>(packed + 16);
    reinterpret_cast<unsigned char*>(si_out)[(((size_t)expert * (M / 128) + tile_m / 4) * N + col + c) * 4 + tile_m % 4] = scales[c];
  }
#elif REQUANT_M32
#define GPTOSS_FC1_COALESCED_Q (GPTOSS_REQUANT_TRIM_DGRAD && GPTOSS_REQUANT_DIRECT_BF16 && GPTOSS_REQUANT_BFE_SCALE && NATIVE_MXFP8_CVT)
  // One 32-row MX block per workgroup. This keeps only 32 values live instead of four blocks (128 values),
  // increasing occupancy. Four workgroups write disjoint bytes of the packed uint32 scale word.
#if GPTOSS_REQUANT_PACKED_INPUT
  // The GPT-OSS schedules use exactly one 128-column scale super-block per workgroup. Their already-packed forward
  // scales supply one uniform uint32 word for every source row; only the selected byte varies across lane groups.
  // Spelling the address without tid lets LLVM issue a scalar/broadcast load instead of redundant raw-e8 loads.
  static_assert(THREADS_PER_WG == 128, "packed GPT-OSS M32 requant expects one 128-column scale block per workgroup");
#endif
  const int tid = threadIdx.x;
  int tile = blockIdx.x;
  const int tile_n = tile % REQUANT_N_TILES;
  tile /= REQUANT_N_TILES;
  const int tile_m = tile % REQUANT_M32_TILES;
  const int expert = tile / REQUANT_M32_TILES;
  const int m_group = tile_m / 4, sub = tile_m % 4;
  const long long n = tile_n * TILE_N + tid;
#if GPTOSS_REQUANT_TRIM_DGRAD
#if GPTOSS_FC1_COALESCED_Q
  // Every lane must reach the LDS exchange, including the half-full final column tile.
  if (tile_n * TILE_N >= REQUANT_REAL_N) return;
#else
  if (n >= REQUANT_REAL_N) return;
#endif
#endif
  float vals[TILE_M];
  float amax = 0.0f;
  #pragma unroll
  for (int mm = 0; mm < TILE_M; mm++) {
    const long long m = tile_m * TILE_M + mm;
    const long long src = ((long long)expert * M_DIM + m) * N_DIM + n;
#if GPTOSS_REQUANT_DIRECT_BF16
    const uint32_t fq_bits = (uint32_t)__builtin_bit_cast(uint8_t, q_in[src]);
#else
    const __hip_fp8_e4m3 fq = __builtin_bit_cast(__hip_fp8_e4m3, q_in[src]);
#endif
#if GPTOSS_REQUANT_PACKED_INPUT
    const uint32_t scale_word = si_in[((long long)expert * (N_DIM / 128) + tile_n) * M_DIM + m];
#if GPTOSS_REQUANT_DIRECT_BF16 && GPTOSS_REQUANT_BFE_SCALE
    // Extract the lane group's packed scale byte directly. The ordinary shift/mask expression lowers to three
    // VALU instructions after the scalar scale load; gfx950 v_bfe performs the identical unsigned extraction in one.
    unsigned src_scale;
    const unsigned scale_shift = 8 * (tid / BLK);
    asm("v_bfe_u32 %0, %1, %2, 8" : "=v"(src_scale) : "s"(scale_word), "v"(scale_shift));
#else
    const unsigned src_scale = (scale_word >> (8 * (tid / BLK))) & 0xffu;
#endif
#else
    const long long src_e8 = ((long long)expert * M_DIM + m) * (N_DIM / BLK) + n / BLK;
    const unsigned src_scale = e8_in[src_e8];
#endif
#if GPTOSS_REQUANT_DIRECT_BF16 && (GPTOSS_REQUANT_TRIM_DGRAD || GPTOSS_REQUANT_BFE_SCALE)
    // src_scale is a byte, so max(src_scale<<23, 1<<22) is exactly the original zero-special ternary.
    // Avoiding its per-row VCC compare/select chain also lowers the exact FC1 kernel from 42 to 40 VGPRs.
    const float dqscale = __builtin_bit_cast(float, max(src_scale << 23, 0x00400000u));
#else
    const float dqscale = __builtin_bit_cast(float, src_scale ? src_scale << 23 : 0x00400000u);
#endif
#if GPTOSS_REQUANT_DIRECT_BF16
    // gfx950 converts scaled FP8 directly to BF16. Taking the low lane retains the exact BF16 rounding boundary
    // of (__hip_bfloat16)((float)fq * dqscale), while avoiding separate FP8->F32, multiply, and BF16-convert ops.
    const auto bf_pair = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(fq_bits, dqscale, false);
    const float v = (float)bf_pair[0];
#else
    const float v = (float)(__hip_bfloat16)((float)fq * dqscale);
#endif
    vals[mm] = v;
    amax = fmaxf(amax, fabsf(v));
  }
#if NATIVE_MXFP8_CVT
  int e8 = (int)((__builtin_bit_cast(unsigned, amax) >> 23) & 0xFFu);
  e8 = max(0, min(254, e8));
  const float scale_f = __builtin_bit_cast(float, (unsigned)e8 << 23);
#else
  int e8 = (int)floorf(log2f(fmaxf(amax, 1e-38f))) + 127;
  e8 = max(0, min(254, e8));
  const float qscale = exp2f((float)(127 - e8));
#endif
  __hip_fp8_storage_t out[TILE_M];
#if NATIVE_MXFP8_CVT
  #pragma unroll
  for (int mm = 0; mm < TILE_M; mm += 2) {
    short2v acc = {0, 0};
    acc = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(acc, vals[mm], vals[mm + 1], scale_f, false);
    const unsigned char *rb = reinterpret_cast<const unsigned char *>(&acc);
    out[mm] = rb[0]; out[mm + 1] = rb[1];
  }
#else
  #pragma unroll
  for (int mm = 0; mm < TILE_M; mm++)
    out[mm] = __hip_cvt_float_to_fp8(fmaxf(-FP8_MAX, fminf(FP8_MAX, vals[mm] * qscale)), __HIP_SATFINITE, __HIP_E4M3);
#endif
#if GPTOSS_FC1_COALESCED_Q
  // Exchange the final bytes through padded LDS so adjacent lanes write adjacent 16-byte vectors.
  __shared__ uint32_t staged_q[128 * 9];
  #pragma unroll
  for (int w = 0; w < 8; w++) staged_q[tid * 9 + w] = reinterpret_cast<const uint32_t*>(out)[w];
  __syncthreads();
  const int lane = tid & 63, wave = tid >> 6;
  #pragma unroll
  for (int j = 0; j < 2; j++) {
    const int local_n = wave * 64 + j * 32 + lane / 2;
    const int word = (lane % 2) * 4;
    const int out_n = tile_n * TILE_N + local_n;
    if (out_n < REQUANT_REAL_N) {
      const long long dst = ((long long)expert * N_DIM + out_n) * M_DIM + tile_m * 32 + word * 4;
      *reinterpret_cast<uint4*>(q_out + dst) = make_uint4(staged_q[local_n * 9 + word + 0], staged_q[local_n * 9 + word + 1],
                                                       staged_q[local_n * 9 + word + 2], staged_q[local_n * 9 + word + 3]);
    }
  }
#else
  const long long obase = ((long long)expert * N_DIM + n) * M_DIM + (long long)tile_m * TILE_M;
  *reinterpret_cast<uint4*>(&q_out[obase]) = *reinterpret_cast<uint4*>(&out[0]);
  *reinterpret_cast<uint4*>(&q_out[obase + 16]) = *reinterpret_cast<uint4*>(&out[16]);
#endif
#if !GPTOSS_REQUANT_SI_ONLY
  e8_out[((long long)expert * N_DIM + n) * (M_DIM / BLK) + tile_m] = (uint8_t)e8;
#endif
#if GPTOSS_FC1_COALESCED_Q
  if (n < REQUANT_REAL_N)
#endif
    reinterpret_cast<uint8_t *>(si_out)[(((long long)expert * REQUANT_M_GROUPS + m_group) * N_DIM + n) * 4 + sub] = (uint8_t)e8;
#else
  const int tid = threadIdx.x;
  int tile = blockIdx.x;
  const int tile_n = tile % N_TILES_N;
  tile /= N_TILES_N;
  const int m_group = tile % REQUANT_M_GROUPS;
  const int expert = tile / REQUANT_M_GROUPS;
  const long long n = tile_n * TILE_N + tid;
  uint32_t packed_e8 = 0;

  #pragma unroll
  for (int sub = 0; sub < 4; sub++) {
    const int tile_m = 4 * m_group + sub;
    float vals[TILE_M];
    float amax = 0.0f;
    #pragma unroll
    for (int mm = 0; mm < TILE_M; mm++) {
      const long long m = tile_m * TILE_M + mm;
      const long long src = ((long long)expert * M_DIM + m) * N_DIM + n;
      const long long src_e8 = ((long long)expert * M_DIM + m) * (N_DIM / BLK) + n / BLK;
      const __hip_fp8_e4m3 fq = __builtin_bit_cast(__hip_fp8_e4m3, q_in[src]);
#if GPTOSS_REQUANT_PACKED_INPUT
      const uint32_t scale_word = si_in[((long long)expert * (N_DIM / 128) + n / 128) * M_DIM + m];
      const unsigned src_scale = (scale_word >> (8 * ((n / BLK) & 3))) & 0xffu;
#else
      const unsigned src_scale = e8_in[src_e8];
#endif
      const float dqscale = __builtin_bit_cast(float, src_scale ? src_scale << 23 : 0x00400000u);
      const float v = (float)(__hip_bfloat16)((float)fq * dqscale);
      vals[mm] = v;
      amax = fmaxf(amax, fabsf(v));
    }
#if NATIVE_MXFP8_CVT
    int e8 = (int)((__builtin_bit_cast(unsigned, amax) >> 23) & 0xFFu);
    e8 = max(0, min(254, e8));
#else
    int e8 = (int)floorf(log2f(fmaxf(amax, 1e-38f))) + 127;
    e8 = max(0, min(254, e8));
    float qscale = exp2f((float)(127 - e8));
#endif

    __hip_fp8_storage_t out[TILE_M];
#if NATIVE_MXFP8_CVT
    const float scale_f = __builtin_bit_cast(float, (unsigned)e8 << 23);
    #pragma unroll
    for (int mm = 0; mm < TILE_M; mm += 2) {
      short2v acc = {0, 0};
      acc = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(acc, vals[mm], vals[mm + 1], scale_f, false);
      const unsigned char *rb = reinterpret_cast<const unsigned char *>(&acc);
      out[mm] = rb[0];
      out[mm + 1] = rb[1];
    }
#else
    #pragma unroll
    for (int mm = 0; mm < TILE_M; mm++)
      out[mm] = __hip_cvt_float_to_fp8(fmaxf(-FP8_MAX, fminf(FP8_MAX, vals[mm] * qscale)), __HIP_SATFINITE, __HIP_E4M3);
#endif

    const long long obase = ((long long)expert * N_DIM + n) * M_DIM + (long long)tile_m * TILE_M;
    *reinterpret_cast<uint4*>(&q_out[obase]) = *reinterpret_cast<uint4*>(&out[0]);
    *reinterpret_cast<uint4*>(&q_out[obase + 16]) = *reinterpret_cast<uint4*>(&out[16]);
    e8_out[((long long)expert * N_DIM + n) * (M_DIM / BLK) + tile_m] = (uint8_t)e8;
    packed_e8 |= (uint32_t)(uint8_t)e8 << (8 * sub);
  }
  si_out[((long long)expert * REQUANT_M_GROUPS + m_group) * N_DIM + n] = packed_e8;
#endif
}

#endif
