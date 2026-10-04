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
#define P_DIM D_DIM
#endif
#ifndef K_DIM
#define K_DIM 4
#endif
#ifndef THREADS_PER_WG
#define THREADS_PER_WG 256
#endif
#ifndef NONTEMPORAL_BF16
#define NONTEMPORAL_BF16 0
#endif

// One workgroup owns one grouped output row. Its eight width-32 subgroups each
// process one MXFP8 block at a time, so both d_out reads and output writes are
// contiguous. The bf16 store and the MXFP8 amax both use the same rounded value.
typedef short short2v __attribute__((ext_vector_type(2)));

constexpr int BLOCK = 32;
constexpr int SUBGROUPS_PER_WG = THREADS_PER_WG / BLOCK;
constexpr int SCALE_BLOCKS = P_DIM / BLOCK;
constexpr int REAL_SCALE_BLOCKS = D_DIM / BLOCK;

static_assert(THREADS_PER_WG == 256, "combine_bwd_dz requires 256 threads");
static_assert(THREADS_PER_WG % 64 == 0, "workgroup must contain whole AMD wavefronts");
static_assert(D_DIM % BLOCK == 0 && P_DIM % BLOCK == 0 && D_DIM <= P_DIM, "dimensions must use whole MXFP8 blocks");
static_assert(G_DIM > 0 && M_DIM > 0 && T_DIM > 0 && K_DIM > 0, "dimensions must be positive");

__device__ __forceinline__ void store_bf16(__hip_bfloat16* ptr, __hip_bfloat16 value) {
#if NONTEMPORAL_BF16
  // dz_bf16 is a 453 MiB streaming output at the GPTOSS production shape and
  // cannot usefully remain in L2. Keep the smaller fp8/e8 companion outputs
  // cached for the immediately-following dgrad consumer.
  __builtin_nontemporal_store(__builtin_bit_cast(unsigned short, value), reinterpret_cast<unsigned short*>(ptr));
#else
  *ptr = value;
#endif
}

extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void
combine_bwd_dz(__hip_bfloat16* __restrict__ dz_bf16,
               __hip_fp8_storage_t* __restrict__ dz_fp8,
               uint8_t* __restrict__ dz_e8,
               const __hip_bfloat16* __restrict__ d_out,
               const int* __restrict__ src_row,
               const float* __restrict__ weights) {
  const int gm = blockIdx.x;
  const int g = gm / M_DIM;
  const int m = gm - g * M_DIM;
  if (g >= G_DIM) return;

  const int tid = threadIdx.x;
  const int subgroup = tid / BLOCK;
  const int lane = tid & (BLOCK - 1);
  const long long out_base = (long long)gm * P_DIM;
  const int src = src_row[gm];

  // The sentinel path returns before forming any d_out or weights address.
  if (src < 0) {
    for (int db = subgroup; db < SCALE_BLOCKS; db += SUBGROUPS_PER_WG) {
      const int d = db * BLOCK + lane;
      store_bf16(&dz_bf16[out_base + d], (__hip_bfloat16)0.0f);
      if ((lane & 1) == 0)
        *reinterpret_cast<unsigned short*>(&dz_fp8[out_base + d]) = 0;
      if (lane == 0) dz_e8[(long long)gm * SCALE_BLOCKS + db] = 0;
    }
    return;
  }

  const int t = src / K_DIM;
  const int j = src - t * K_DIM;
  const float weight = weights[((long long)g * T_DIM + t) * K_DIM + j];
  const long long in_base = ((long long)g * T_DIM + t) * D_DIM;

  for (int db = subgroup; db < SCALE_BLOCKS; db += SUBGROUPS_PER_WG) {
    if (db >= REAL_SCALE_BLOCKS) {
      const int d = db * BLOCK + lane;
      store_bf16(&dz_bf16[out_base + d], (__hip_bfloat16)0.0f);
      if ((lane & 1) == 0)
        *reinterpret_cast<unsigned short*>(&dz_fp8[out_base + d]) = 0;
      if (lane == 0) dz_e8[(long long)gm * SCALE_BLOCKS + db] = 0;
      continue;
    }
    const int d = db * BLOCK + lane;

    // This cast is the required RNE bf16 boundary: quantization must consume
    // the value that was stored, not the unrounded fp32 product.
    const __hip_bfloat16 vec_bf16 = (__hip_bfloat16)((float)d_out[in_base + d] * weight);
    store_bf16(&dz_bf16[out_base + d], vec_bf16);
    const float vec = (float)vec_bf16;

    float amax = fabsf(vec);
#pragma unroll
    for (int offset = BLOCK / 2; offset > 0; offset >>= 1)
      amax = fmaxf(amax, __shfl_down(amax, offset, BLOCK));
    amax = __shfl(amax, 0, BLOCK);

    // FAST_E8M0: the exponent field is floor(log2(amax))+127 for normal
    // values, zero for an all-zero/subnormal block, and is clamped below NaN.
    int e8 = (int)((__builtin_bit_cast(unsigned, amax) >> 23) & 0xFFu);
    e8 = min(e8, 254);
    if (lane == 0) dz_e8[(long long)gm * SCALE_BLOCKS + db] = (uint8_t)e8;

    // gfx950 v_cvt_scalef32_pk_fp8_f32. Pair adjacent lanes inside each
    // width-32 subgroup; even lanes write the two resulting contiguous bytes.
    const float other = __shfl_down(vec, 1, BLOCK);
    if ((lane & 1) == 0) {
      const float scale_f = __builtin_bit_cast(float, (unsigned)e8 << 23);
      short2v packed = {0, 0};
      packed = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(packed, vec, other, scale_f, false);
      const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&packed);
      const unsigned short out = (unsigned short)bytes[0] | ((unsigned short)bytes[1] << 8);
      *reinterpret_cast<unsigned short*>(&dz_fp8[out_base + d]) = out;
    }
  }
}
