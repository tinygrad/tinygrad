#include "kittens.cuh"

using namespace kittens;

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
#define DW_OWNER_DEVICE_ONLY
#include "lmhead_ce_dw_owner.cpp"
#define DH_OWNER_DEVICE_ONLY
#include "lmhead_ce_dh_owner.cpp"

constexpr int BLOCK = 256;
constexpr int HALF_BLOCK = 128;
constexpr int K_STEP = 64;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int NUM_WARPS = WARPS_M * WARPS_N;
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
constexpr int REG_M = HALF_BLOCK / WARPS_M;
constexpr int REG_N = HALF_BLOCK / WARPS_N;

static_assert(PERSIST_WG > 0 && PERSIST_WG <= 256,
              "gfx950 persistent grids must have at most one workgroup per CU");
static_assert(TOKENS % BLOCK == 0 && VOCAB % BLOCK == 0 && HIDDEN % BLOCK == 0);
static_assert(TOKENS % K_STEP == 0 && VOCAB % K_STEP == 0 && HIDDEN % K_STEP == 0);

using G = kittens::group<NUM_WARPS>;
using RowST = st_bf<HALF_BLOCK, K_STEP, st_16x32_s>;
using HiddenGL = gl<bf16, 1, 1, TOKENS, HIDDEN>;
using WeightGL = gl<bf16, 1, 1, VOCAB, HIDDEN>;
using DLogitsGL = gl<bf16, 1, 1, TOKENS, VOCAB>;

// barrier_state is {arrival_count, epoch}. Initialize arrival_count to zero;
// epoch may have any value. Do not share it between concurrent launches.
//
// Residency invariant: launch exactly PERSIST_WG one-dimensional workgroups on
// MI350X/gfx950, with PERSIST_WG <= 256. The kernel's launch bound guarantees
// one block fits per CU. If the whole grid cannot co-reside, this software
// barrier can deadlock while resident blocks wait for unscheduled blocks.
template <int EXPECTED_WGS>
__device__ __forceinline__ void grid_barrier(int *__restrict__ state) {
  asm volatile("s_dcache_wb_vol");
  __threadfence_system();
  __builtin_amdgcn_s_barrier();
  if (threadIdx.x == 0) {
    const int epoch = __atomic_load_n(state + 1, __ATOMIC_ACQUIRE);
    const int prior = __atomic_fetch_add(state, 1, __ATOMIC_ACQ_REL);
    if (prior == EXPECTED_WGS - 1) {
      __atomic_store_n(state, 0, __ATOMIC_RELAXED);
      __threadfence_system();
      __atomic_fetch_add(state + 1, 1, __ATOMIC_RELEASE);
    } else {
      while (__atomic_load_n(state + 1, __ATOMIC_ACQUIRE) == epoch) { }
    }
  }
  __builtin_amdgcn_s_barrier();
  __threadfence_system();
  __builtin_amdgcn_s_barrier();
  asm volatile("s_dcache_inv_vol");
  asm volatile("s_waitcnt lgkmcnt(0)");
  __builtin_amdgcn_s_barrier();
}

// One 256x256 logits owner, using the gemm_bf16.cpp 2x4-wave decomposition.
// Complete FP32 accumulators are first rounded into a BF16 LDS tile; the CE
// derivative consumes those rounded logits and is itself rounded to BF16.
__device__ __forceinline__ void make_dlogits_tile(
    int tile_m, int tile_n, int *shared_base,
    const HiddenGL &Hidden, const WeightGL &Weight, const DLogitsGL &DLogits,
    const int *__restrict__ targets, const float *__restrict__ lse,
    const float *__restrict__ loss_scale) {
  shared_allocator al(shared_base);
  RowST (&As)[2] = al.allocate<RowST, 2>();
  RowST (&Bs)[2] = al.allocate<RowST, 2>();
  const int wid = warpid(), warp_m = wid / WARPS_N, warp_n = wid % WARPS_N;

  rt_bf<REG_M, K_STEP, row_l, rt_16x32_s> a;
  rt_bf<REG_N, K_STEP, row_l, rt_16x32_s> b0, b1;
  rt_fl<REG_M, REG_N, col_l, rt_16x16_s> c[2][2];
  zero(c[0][0]); zero(c[0][1]); zero(c[1][0]); zero(c[1][1]);

  for (int kk = 0; kk < HIDDEN / K_STEP; ++kk) {
    G::load(As[0], Hidden, {0, 0, tile_m * 2, kk});
    G::load(As[1], Hidden, {0, 0, tile_m * 2 + 1, kk});
    G::load(Bs[0], Weight, {0, 0, tile_n * 2, kk});
    G::load(Bs[1], Weight, {0, 0, tile_n * 2 + 1, kk});
    asm volatile("s_waitcnt vmcnt(0)");
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    load(b0, subtile_inplace<REG_N, K_STEP>(Bs[0], {warp_n, 0}));
    load(b1, subtile_inplace<REG_N, K_STEP>(Bs[1], {warp_n, 0}));
    load(a, subtile_inplace<REG_M, K_STEP>(As[0], {warp_m, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(c[0][0], a, b0, c[0][0]);
    mma_ABt(c[0][1], a, b1, c[0][1]);
    load(a, subtile_inplace<REG_M, K_STEP>(As[1], {warp_m, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    mma_ABt(c[1][0], a, b0, c[1][0]);
    mma_ABt(c[1][1], a, b1, c[1][1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
  }

  bf16 *logits = reinterpret_cast<bf16 *>(shared_base);
  const int lane = laneid(), row4 = 4 * (lane / 16), col16 = lane % 16;
#define STORE_LOGITS(ACC, RB, CB)                                                    \
  _Pragma("unroll") for (int i = 0; i < REG_M / 16; ++i) {                         \
    _Pragma("unroll") for (int j = 0; j < REG_N / 16; ++j) {                       \
      const int r = (RB) + warp_m * REG_M + i * 16 + row4;                          \
      const int q = (CB) + warp_n * REG_N + j * 16 + col16;                         \
      logits[(r + 0) * BLOCK + q] = base_types::convertor<bf16, float>::convert((ACC).tiles[i][j].data[0].x); \
      logits[(r + 1) * BLOCK + q] = base_types::convertor<bf16, float>::convert((ACC).tiles[i][j].data[0].y); \
      logits[(r + 2) * BLOCK + q] = base_types::convertor<bf16, float>::convert((ACC).tiles[i][j].data[1].x); \
      logits[(r + 3) * BLOCK + q] = base_types::convertor<bf16, float>::convert((ACC).tiles[i][j].data[1].y); \
    }                                                                                \
  }
  STORE_LOGITS(c[0][0], 0, 0);
  STORE_LOGITS(c[0][1], 0, HALF_BLOCK);
  STORE_LOGITS(c[1][0], HALF_BLOCK, 0);
  STORE_LOGITS(c[1][1], HALF_BLOCK, HALF_BLOCK);
#undef STORE_LOGITS
  __builtin_amdgcn_s_barrier();

  bf16 *dz = (bf16 *)&DLogits[{0, 0, 0, 0}];
  for (int i = threadIdx.x; i < BLOCK * BLOCK; i += NUM_THREADS) {
    const int lr = i / BLOCK, lc = i - lr * BLOCK;
    const int gr = tile_m * BLOCK + lr, gc = tile_n * BLOCK + lc;
    const float z = base_types::convertor<float, bf16>::convert(logits[i]);
    const float p = expf(fminf(z - lse[gr], 0.0f));
    const float v = (p - (targets[gr] == gc ? 1.0f : 0.0f)) * loss_scale[0];
    dz[(long long)gr * VOCAB + gc] = base_types::convertor<bf16, float>::convert(v);
  }
  asm volatile("s_waitcnt vmcnt(0)");
  __builtin_amdgcn_s_barrier();
}


// Outputs: BF16 dHidden, BF16 dWeight, BF16 dlogits, then two-int barrier state.
extern "C" __global__ __launch_bounds__(NUM_THREADS, 1)
void lmhead_ce_bwd_persistent(
    bf16 *__restrict__ d_hidden_ptr, bf16 *__restrict__ d_weight_ptr,
    bf16 *__restrict__ dlogits_ptr, int *__restrict__ barrier_state,
    bf16 *__restrict__ hidden_ptr, bf16 *__restrict__ weight_ptr,
    const int *__restrict__ targets, const float *__restrict__ lse,
    const float *__restrict__ loss_scale) {
  const int wg = __builtin_amdgcn_workgroup_id_x();
  HiddenGL Hidden{hidden_ptr, nullptr, nullptr, nullptr, nullptr};
  WeightGL Weight{weight_ptr, nullptr, nullptr, nullptr, nullptr};
  DLogitsGL DLogits{dlogits_ptr, nullptr, nullptr, nullptr, nullptr};
  __shared__ alignment_dummy __shm[MAX_SHARED_MEMORY / sizeof(alignment_dummy)];
  int *shared_base = (int *)&__shm[0];

  // Phase 1: 4008 default-shape tiles, grid-stride over the persistent grid.
  constexpr int LOGIT_NT = VOCAB / BLOCK;
  constexpr int LOGIT_TILES = (TOKENS / BLOCK) * LOGIT_NT;
  for (int owner = wg; owner < LOGIT_TILES; owner += PERSIST_WG)
    make_dlogits_tile(owner / LOGIT_NT, owner % LOGIT_NT, shared_base,
                      Hidden, Weight, DLogits, targets, lse, loss_scale);

  grid_barrier<PERSIST_WG>(barrier_state);

  // Phase 2a: unique dHidden output owners. No FP32 partitions are emitted.
  dh_owner::lmhead_ce_dh_owner_body(wg, d_hidden_ptr, dlogits_ptr, weight_ptr, shared_base);

  // Phase 2b: the proven double-buffered output-owner AtB kernel body.
  lmhead_ce_dw_owner_body(wg, d_weight_ptr, dlogits_ptr, hidden_ptr, shared_base);
}
