#include "kittens.cuh"

using namespace kittens;

#ifndef ROUTER_M
#define ROUTER_M 16384
#endif
#ifndef ROUTER_K
#define ROUTER_K 2880
#endif
#ifndef ROUTER_E
#define ROUTER_E 32
#endif
#ifndef NUM_WARPS
#define NUM_WARPS 16
#endif
#ifndef NUM_ACCUMS
#define NUM_ACCUMS 1
#endif
#ifndef DUAL_EXPERT
#define DUAL_EXPERT 0
#endif
#ifndef PREFETCH
#define PREFETCH 0
#endif

constexpr int THREADS = NUM_WARPS * 64;
constexpr int TILE = 16;
constexpr int EXPERT_TILES = DUAL_EXPERT ? 2 : 1;
using float4_t = float __attribute__((ext_vector_type(4)));

static_assert(ROUTER_E == 32, "router wgrad is specialized for 32 experts");
static_assert(ROUTER_K % TILE == 0, "hidden size must be divisible by 16");
static_assert(ROUTER_M % (NUM_WARPS * 4) == 0, "waves split the token reduction evenly");
static_assert(NUM_ACCUMS > 0 && (NUM_ACCUMS & (NUM_ACCUMS - 1)) == 0, "accumulator count must be a power of two");

// Computes out[E,K] = x[M,K].T @ gradient[M,E], stored transposed in row-major [E,K].
// The gate gradient stays FP32: unlike the old padded BF16 GEMM path, neither input nor output is quantized here.
// One workgroup owns a 16-feature x 16-expert output tile. Its waves split the M reduction, then wave 0 combines
// their FP32 MFMA accumulators in LDS. NUM_ACCUMS can add independent chains when latency hiding is useful.
extern "C" __global__ __launch_bounds__(THREADS, 2) void moe_router_fp32_wgrad(
    bf16 *__restrict__ out, const bf16 *__restrict__ x, const float *__restrict__ gradient) {
  const int lane = laneid();
  const int warp = threadIdx.x / 64;
  const int tile = blockIdx.x;
#if DUAL_EXPERT
  const int feature_base = tile * TILE;
  const int expert_base = 0;
#else
  const int feature_base = (tile / (ROUTER_E / TILE)) * TILE;
  const int expert_base = (tile % (ROUTER_E / TILE)) * TILE;
#endif
  const int k_lane = lane / TILE;
  const int mn_lane = lane % TILE;

  float4_t accum[EXPERT_TILES][NUM_ACCUMS] = {};
  int ai = 0;
#if PREFETCH
  static_assert(NUM_ACCUMS == 1 && DUAL_EXPERT == 1 && NUM_WARPS == 16);
  static_assert((ROUTER_M / 4) % (NUM_WARPS * PREFETCH) == 0);
  #pragma unroll 1
  for (int mt = warp; mt < ROUTER_M / 4; mt += NUM_WARPS * PREFETCH) {
    float av[PREFETCH], bv0[PREFETCH], bv1[PREFETCH];
    #pragma unroll
    for (int p = 0; p < PREFETCH; p++) {
      const int token = (mt + p * NUM_WARPS) * 4 + k_lane;
      av[p] = (float)x[(long long)token * ROUTER_K + feature_base + mn_lane];
      bv0[p] = gradient[(long long)token * ROUTER_E + mn_lane];
      bv1[p] = gradient[(long long)token * ROUTER_E + TILE + mn_lane];
    }
    static_assert(PREFETCH == 8);
    // Keep each chunk's inputs live together; without this constraint clang serializes load/MFMA pairs.
    asm volatile("" :: "v"(av[0]), "v"(bv0[0]), "v"(bv1[0]), "v"(av[1]), "v"(bv0[1]), "v"(bv1[1]),
                       "v"(av[2]), "v"(bv0[2]), "v"(bv1[2]), "v"(av[3]), "v"(bv0[3]), "v"(bv1[3]),
                       "v"(av[4]), "v"(bv0[4]), "v"(bv1[4]), "v"(av[5]), "v"(bv0[5]), "v"(bv1[5]),
                       "v"(av[6]), "v"(bv0[6]), "v"(bv1[6]), "v"(av[7]), "v"(bv0[7]), "v"(bv1[7]) : "memory");
    #pragma unroll
    for (int p = 0; p < PREFETCH; p++) {
      accum[0][0] = __builtin_amdgcn_mfma_f32_16x16x4f32(av[p], bv0[p], accum[0][0], 0, 0, 0);
      accum[1][0] = __builtin_amdgcn_mfma_f32_16x16x4f32(av[p], bv1[p], accum[1][0], 0, 0, 0);
    }
  }
#else
  #pragma unroll 1
  for (int mt = warp; mt < ROUTER_M / 4; mt += NUM_WARPS) {
    const int token = mt * 4 + k_lane;
    const float av = (float)x[(long long)token * ROUTER_K + feature_base + mn_lane];
    #pragma unroll
    for (int et = 0; et < EXPERT_TILES; et++) {
      const float bv = gradient[(long long)token * ROUTER_E + expert_base + et * TILE + mn_lane];
      accum[et][ai] = __builtin_amdgcn_mfma_f32_16x16x4f32(av, bv, accum[et][ai], 0, 0, 0);
    }
    ai = (ai + 1) & (NUM_ACCUMS - 1);
  }
#endif
  #pragma unroll
  for (int et = 0; et < EXPERT_TILES; et++) {
    #pragma unroll
    for (int stride = 1; stride < NUM_ACCUMS; stride *= 2) {
      #pragma unroll
      for (int i = 0; i < NUM_ACCUMS; i += 2 * stride) accum[et][i] += accum[et][i + stride];
    }
  }

  __shared__ float partial[EXPERT_TILES][NUM_WARPS][64][4];
  #pragma unroll
  for (int et = 0; et < EXPERT_TILES; et++) {
    #pragma unroll
    for (int i = 0; i < 4; i++) partial[et][warp][lane][i] = accum[et][0][i];
  }
  __syncthreads();

  if (warp == 0) {
    #pragma unroll
    for (int et = 0; et < EXPERT_TILES; et++) {
      float4_t total = {};
      #pragma unroll
      for (int w = 0; w < NUM_WARPS; w++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) total[i] += partial[et][w][lane][i];
      }
      // The 16x16 MFMA C layout gives each lane four adjacent feature rows at one expert column.
      const int expert = expert_base + et * TILE + lane % 16;
      const int feature0 = feature_base + 4 * (lane / 16);
      const float2 lo = {total[0], total[1]}, hi = {total[2], total[3]};
      const bf16_2 blo = base_types::convertor<bf16_2, float2>::convert(lo);
      const bf16_2 bhi = base_types::convertor<bf16_2, float2>::convert(hi);
      reinterpret_cast<uint32_t *>(out)[((long long)expert * ROUTER_K + feature0) / 2] =
          *reinterpret_cast<const uint32_t *>(&blo);
      reinterpret_cast<uint32_t *>(out)[((long long)expert * ROUTER_K + feature0 + 2) / 2] =
          *reinterpret_cast<const uint32_t *>(&bhi);
    }
  }
}
