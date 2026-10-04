#include "kittens.cuh"

using namespace kittens;
namespace dh_owner {

#ifndef TOKENS
#define TOKENS 2048
#endif
#ifndef VOCAB
#define VOCAB 128256
#endif
#ifndef HIDDEN
#define HIDDEN 3072
#endif
#ifndef PERSIST_WG
#define PERSIST_WG 256
#endif

// One 512-thread workgroup owns a 256 x 256 dHidden tile. Its eight
// wavefronts form a 2 x 4 array; each wavefront owns four 64 x 32 FP32
// accumulator tiles. The full VOCAB contraction advances in 64-wide chunks.
constexpr int BLOCK_M = 256;
constexpr int BLOCK_H = 256;
constexpr int HALF_BLOCK_M = BLOCK_M / 2;
constexpr int HALF_BLOCK_H = BLOCK_H / 2;
constexpr int K_STEP = 64;
constexpr int WARPS_M = 2;
constexpr int WARPS_H = 4;
constexpr int WARP_TILE_M = 64;
constexpr int WARP_TILE_H = 32;
constexpr int NUM_WARPS = WARPS_M * WARPS_H;
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
constexpr int M_GROUP = 8;
constexpr int M_TILES = TOKENS / BLOCK_M;
constexpr int H_TILES = HIDDEN / BLOCK_H;
constexpr int OUTPUT_TILES = M_TILES * H_TILES;
constexpr int K_TILES = VOCAB / K_STEP;

using G = kittens::group<NUM_WARPS>;
using DLogitsGL = gl<bf16, 1, 1, TOKENS, VOCAB>;
using WeightGL = gl<bf16, 1, 1, VOCAB, HIDDEN>;
using DHiddenGL = gl<bf16, 1, 1, TOKENS, HIDDEN>;
using DLogitsST = st_bf<HALF_BLOCK_M, K_STEP, st_16x32_s>;
using WeightST = st_bf<K_STEP, HALF_BLOCK_H, st_32x16_s>;

// dHidden[TOKENS,HIDDEN] = dlogits[TOKENS,VOCAB] @
// weight[VOCAB,HIDDEN]. All arrays are physically row-major BF16. Each
// workgroup retains its complete FP32 contraction without partials or atomics.
__device__ __forceinline__ void lmhead_ce_dh_owner_body(
    int physical_wg, bf16 *__restrict__ dhidden_ptr,
    bf16 *__restrict__ dlogits_ptr, bf16 *__restrict__ weight_ptr,
    int *shared_base) {
  static_assert(TOKENS % BLOCK_M == 0);
  static_assert(HIDDEN % BLOCK_H == 0);
  static_assert(VOCAB % K_STEP == 0);
  static_assert(PERSIST_WG > 0);
  if (physical_wg >= PERSIST_WG) return;

  DLogitsGL dlogits{dlogits_ptr, nullptr, nullptr, nullptr, nullptr};
  WeightGL weight{weight_ptr, nullptr, nullptr, nullptr, nullptr};
  DHiddenGL dhidden{dhidden_ptr, nullptr, nullptr, nullptr, nullptr};

  shared_allocator al(shared_base);
  DLogitsST (&ds)[2][2] = al.allocate<DLogitsST, 2, 2>();
  WeightST (&ws)[2][2] = al.allocate<WeightST, 2, 2>();

  constexpr int DLOADS =
      DLogitsST::rows * DLogitsST::cols * sizeof(bf16) /
      (DLogitsST::underlying_subtile_bytes_per_thread * NUM_THREADS);
  constexpr int WLOADS =
      WeightST::rows * WeightST::cols * sizeof(bf16) /
      (WeightST::underlying_subtile_bytes_per_thread * NUM_THREADS);
  static_assert(
      DLOADS * DLogitsST::underlying_subtile_bytes_per_thread * NUM_THREADS ==
      DLogitsST::rows * DLogitsST::cols * sizeof(bf16));
  static_assert(
      WLOADS * WeightST::underlying_subtile_bytes_per_thread * NUM_THREADS ==
      WeightST::rows * WeightST::cols * sizeof(bf16));
  uint32_t d_offsets[DLOADS], w_offsets[WLOADS];
  G::prefill_swizzled_offsets(ds[0][0], dlogits, d_offsets);
  G::prefill_swizzled_offsets(ws[0][0], weight, w_offsets);

  const int wid = warpid();
  const int warp_m = wid / WARPS_H;
  const int warp_h = wid % WARPS_H;

  for (int owner = physical_wg; owner < OUTPUT_TILES;
       owner += PERSIST_WG) {
    // Visit eight adjacent token blocks per hidden block, matching the proven
    // 256x256 GEMM mapping and preserving its L2 reuse.
    const int tiles_per_group = M_GROUP * H_TILES;
    const int group = owner / tiles_per_group;
    const int first_m = group * M_GROUP;
    const int remaining_m = M_TILES - first_m;
    const int group_m = remaining_m < M_GROUP ? remaining_m : M_GROUP;
    const int in_group = owner % tiles_per_group;
    const int tile_m = first_m + in_group % group_m;
    const int tile_h = in_group / group_m;

    rt_fl<WARP_TILE_M, WARP_TILE_H, col_l, rt_16x16_s> accum[2][2];
    zero(accum[0][0]);
    zero(accum[0][1]);
    zero(accum[1][0]);
    zero(accum[1][1]);

    rt_bf<WARP_TILE_M, K_STEP, row_l, rt_16x32_s> dreg;
    rt_bf<K_STEP, WARP_TILE_H, col_l, rt_32x16_s> wreg_0;
    rt_bf<K_STEP, WARP_TILE_H, col_l, rt_32x16_s> wreg_1;

    int read_buffer = 0;
    kittens::load<2, false, DLogitsST, DLogitsGL, coord<DLogitsST>,
                  NUM_THREADS, coherency::non_temporal>(
        ds[0][0], dlogits, {0, 0, tile_m * 2, 0}, d_offsets);
    G::load(ws[0][0], weight, {0, 0, 0, tile_h * 2}, w_offsets);
    kittens::load<2, false, DLogitsST, DLogitsGL, coord<DLogitsST>,
                  NUM_THREADS, coherency::non_temporal>(
        ds[0][1], dlogits, {0, 0, tile_m * 2 + 1, 0}, d_offsets);
    G::load(ws[0][1], weight, {0, 0, 0, tile_h * 2 + 1}, w_offsets);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    // The alternate shared buffer is filled while MFMA consumes the current
    // one. The four FP32 register tiles remain live across every K chunk.
    for (int kt = 0; kt < K_TILES; ++kt) {
      const int write_buffer = read_buffer ^ 1;
      if (kt + 1 < K_TILES) {
        kittens::load<2, false, DLogitsST, DLogitsGL, coord<DLogitsST>,
                      NUM_THREADS, coherency::non_temporal>(
            ds[write_buffer][0], dlogits,
            {0, 0, tile_m * 2, kt + 1}, d_offsets);
        G::load(ws[write_buffer][0], weight,
                {0, 0, kt + 1, tile_h * 2}, w_offsets);
        kittens::load<2, false, DLogitsST, DLogitsGL, coord<DLogitsST>,
                      NUM_THREADS, coherency::non_temporal>(
            ds[write_buffer][1], dlogits,
            {0, 0, tile_m * 2 + 1, kt + 1}, d_offsets);
        G::load(ws[write_buffer][1], weight,
                {0, 0, kt + 1, tile_h * 2 + 1}, w_offsets);
      }

      load(dreg, subtile_inplace<WARP_TILE_M, K_STEP>(
                     ds[read_buffer][0], {warp_m, 0}));
      load(wreg_0, subtile_inplace<K_STEP, WARP_TILE_H>(
                       ws[read_buffer][0], {0, warp_h}));
      load(wreg_1, subtile_inplace<K_STEP, WARP_TILE_H>(
                       ws[read_buffer][1], {0, warp_h}));
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_setprio(1);
      mma_AB(accum[0][0], dreg, wreg_0, accum[0][0]);
      mma_AB(accum[0][1], dreg, wreg_1, accum[0][1]);
      __builtin_amdgcn_s_setprio(0);
      __builtin_amdgcn_sched_barrier(0);

      load(dreg, subtile_inplace<WARP_TILE_M, K_STEP>(
                     ds[read_buffer][1], {warp_m, 0}));
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_setprio(1);
      mma_AB(accum[1][0], dreg, wreg_0, accum[1][0]);
      mma_AB(accum[1][1], dreg, wreg_1, accum[1][1]);
      __builtin_amdgcn_s_setprio(0);
      __builtin_amdgcn_sched_barrier(0);

      if (kt + 1 < K_TILES) {
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        read_buffer = write_buffer;
      }
    }

    // Each element is converted to BF16 exactly once, after the complete
    // VOCAB contraction.
    store(dhidden, accum[0][0],
          {0, 0, (tile_m * 2) * WARPS_M + warp_m,
           tile_h * 2 * WARPS_H + warp_h});
    store(dhidden, accum[0][1],
          {0, 0, (tile_m * 2) * WARPS_M + warp_m,
           tile_h * 2 * WARPS_H + WARPS_H + warp_h});
    store(dhidden, accum[1][0],
          {0, 0, (tile_m * 2) * WARPS_M + WARPS_M + warp_m,
           tile_h * 2 * WARPS_H + warp_h});
    store(dhidden, accum[1][1],
          {0, 0, (tile_m * 2) * WARPS_M + WARPS_M + warp_m,
           tile_h * 2 * WARPS_H + WARPS_H + warp_h});
    __builtin_amdgcn_s_barrier();
  }
}
}  // namespace dh_owner

#ifndef DH_OWNER_DEVICE_ONLY
extern "C" __global__ __launch_bounds__(dh_owner::NUM_THREADS, 2) void lmhead_ce_dh_owner(
    bf16 *__restrict__ dhidden_ptr, bf16 *__restrict__ dlogits_ptr, bf16 *__restrict__ weight_ptr) {
  const int physical_wg = __builtin_amdgcn_workgroup_id_x();
  __shared__ alignment_dummy __shm[MAX_SHARED_MEMORY / sizeof(alignment_dummy)];
  dh_owner::lmhead_ce_dh_owner_body(physical_wg, dhidden_ptr, dlogits_ptr, weight_ptr, (int *)&__shm[0]);
}
#endif
