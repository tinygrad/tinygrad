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

constexpr int DW_BLOCK_V = 256;
constexpr int DW_BLOCK_H = 256;
constexpr int DW_K_STEP = 64;
constexpr int DW_WARPS_V = 2;
constexpr int DW_WARPS_H = 4;
constexpr int DW_NUM_WARPS = DW_WARPS_V * DW_WARPS_H;
constexpr int DW_NUM_THREADS = DW_NUM_WARPS * WARP_THREADS;
constexpr int DW_WAVE_V = DW_BLOCK_V / DW_WARPS_V;
constexpr int DW_WAVE_H = DW_BLOCK_H / DW_WARPS_H;
constexpr int DW_HALF_WAVE_V = DW_WAVE_V / 2;
constexpr int DW_HALF_WAVE_H = DW_WAVE_H / 2;
constexpr int DW_V_TILES = VOCAB / DW_BLOCK_V;
constexpr int DW_H_TILES = HIDDEN / DW_BLOCK_H;
constexpr int DW_OUTPUT_TILES = DW_V_TILES * DW_H_TILES;

using DWGroup = kittens::group<DW_NUM_WARPS>;
using DWDLogitsGL = gl<bf16, 1, 1, TOKENS, VOCAB>;

// dweight[VOCAB,HIDDEN] = dlogits[TOKENS,VOCAB].T @ hidden[TOKENS,HIDDEN].
// A 512-thread (eight-wave) workgroup owns one complete 256x256 VxH tile.
// The waves form a 2x4 grid; each wave retains four 64x32 FP32 fragments
// across the entire TOKENS contraction and converts them to BF16 only at store.
// Launch exactly PERSIST_WG workgroups. Each workgroup grid-strides over tiles.
__device__ __forceinline__ void lmhead_ce_dw_owner_body(
    int physical_wg, bf16 *__restrict__ dweight_ptr,
    bf16 *__restrict__ dlogits_ptr, bf16 *__restrict__ hidden_ptr,
    int *shared_base) {
    static_assert(TOKENS % DW_K_STEP == 0, "TOKENS must be a multiple of 64");
    static_assert(VOCAB % DW_BLOCK_V == 0, "VOCAB must be a multiple of 256");
    static_assert(HIDDEN % DW_BLOCK_H == 0, "HIDDEN must be a multiple of 256");
    static_assert(PERSIST_WG > 0, "PERSIST_WG must be positive");


    DWDLogitsGL dlogits{dlogits_ptr, nullptr, nullptr, nullptr, nullptr};
    gl<bf16, 1, 1, TOKENS, HIDDEN> hidden{
        hidden_ptr, nullptr, nullptr, nullptr, nullptr};
    gl<bf16, 1, 1, VOCAB, HIDDEN> dweight{
        dweight_ptr, nullptr, nullptr, nullptr, nullptr};

    shared_allocator al(shared_base);

    using DLogitsST = st_bf<DW_K_STEP, DW_BLOCK_V / 2, st_32x16_s>;
    using HiddenST = st_bf<DW_K_STEP, DW_BLOCK_H / 2, st_32x16_s>;
    DLogitsST (&dlogits_s)[2][2] = al.allocate<DLogitsST, 2, 2>();
    HiddenST (&hidden_s)[2][2] = al.allocate<HiddenST, 2, 2>();

    rt_bf<DW_K_STEP, DW_HALF_WAVE_V, col_l, rt_32x16_s> dlogits_r;
    rt_bf<DW_K_STEP, DW_HALF_WAVE_H, col_l, rt_32x16_s> hidden_r0;
    rt_bf<DW_K_STEP, DW_HALF_WAVE_H, col_l, rt_32x16_s> hidden_r1;
    rt_fl<DW_HALF_WAVE_V, DW_HALF_WAVE_H, col_l, rt_16x16_s> accum[2][2];

    const int warp_id = kittens::warpid();
    const int warp_v = warp_id / DW_WARPS_H;
    const int warp_h = warp_id % DW_WARPS_H;
    constexpr int reduction_tiles = TOKENS / DW_K_STEP;

    const bf16 *dlogits_base = (bf16 *)&dlogits[{0, 0, 0, 0}];
    const bf16 *hidden_base = (bf16 *)&hidden[{0, 0, 0, 0}];
    const int dlogits_row_stride = dlogits.template stride<2>() * sizeof(bf16);
    const int hidden_row_stride = hidden.template stride<2>() * sizeof(bf16);
    i32x4 dlogits_srsrc = make_srsrc(
        dlogits_base, TOKENS * dlogits_row_stride, dlogits_row_stride);
    i32x4 hidden_srsrc = make_srsrc(
        hidden_base, TOKENS * hidden_row_stride, hidden_row_stride);

    const int wid = warpid() % DW_NUM_WARPS;
    constexpr int elements_per_wave =
        (16 / sizeof(bf16)) * kittens::WARP_THREADS;
#define DW_LDS_ADDRESS(tile)                                                    \
    __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(                       \
        reinterpret_cast<uintptr_t>(&(tile).data[0]) +                          \
        wid * elements_per_wave * sizeof(bf16)))
    const uint32_t dl_lds_00 = DW_LDS_ADDRESS(dlogits_s[0][0]);
    const uint32_t dl_lds_01 = DW_LDS_ADDRESS(dlogits_s[0][1]);
    const uint32_t dl_lds_10 = DW_LDS_ADDRESS(dlogits_s[1][0]);
    const uint32_t dl_lds_11 = DW_LDS_ADDRESS(dlogits_s[1][1]);
    const uint32_t h_lds_00 = DW_LDS_ADDRESS(hidden_s[0][0]);
    const uint32_t h_lds_01 = DW_LDS_ADDRESS(hidden_s[0][1]);
    const uint32_t h_lds_10 = DW_LDS_ADDRESS(hidden_s[1][0]);
    const uint32_t h_lds_11 = DW_LDS_ADDRESS(hidden_s[1][1]);
#undef DW_LDS_ADDRESS

    using DType = typename DLogitsST::dtype;
    constexpr int bytes_per_thread = DLogitsST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_memcpy = bytes_per_thread * DW_NUM_THREADS;
    constexpr int memcpy_per_half_tile =
        (DW_K_STEP * (DW_BLOCK_V / 2) * sizeof(DType)) / bytes_per_memcpy;
    uint32_t dlogits_offsets[memcpy_per_half_tile];
    uint32_t hidden_offsets[memcpy_per_half_tile];
    DWGroup::prefill_swizzled_offsets(dlogits_s[0][0], dlogits, dlogits_offsets);
    DWGroup::prefill_swizzled_offsets(hidden_s[0][0], hidden, hidden_offsets);

    for (int output_tile = physical_wg; output_tile < DW_OUTPUT_TILES;
         output_tile += PERSIST_WG) {
        // Use the proven grouped ordering: eight neighboring vocab tiles sweep
        // all hidden tiles before advancing to the next vocab group.
        constexpr int V_GROUP = 8;
        constexpr int tiles_per_group = V_GROUP * DW_H_TILES;
        const int group = output_tile / tiles_per_group;
        const int first_v = group * V_GROUP;
        const int remaining_v = DW_V_TILES - first_v;
        const int group_v = remaining_v < V_GROUP ? remaining_v : V_GROUP;
        const int in_group = output_tile % tiles_per_group;
        const int tile_v = first_v + in_group % group_v;
        const int tile_h = in_group / group_v;

        zero(accum[0][0]);
        zero(accum[0][1]);
        zero(accum[1][0]);
        zero(accum[1][1]);

        // Prime ping buffer with the first 64-token reduction chunk.
        kittens::load<2, false, DLogitsST, DWDLogitsGL, coord<DLogitsST>, DW_NUM_THREADS, coherency::non_temporal>(
            dlogits_s[0][0], dlogits, {0, 0, 0, tile_v * 2},
            dlogits_offsets, dlogits_srsrc, dlogits_base, dl_lds_00);
        kittens::load<2, false, DLogitsST, DWDLogitsGL, coord<DLogitsST>, DW_NUM_THREADS, coherency::non_temporal>(
            dlogits_s[0][1], dlogits, {0, 0, 0, tile_v * 2 + 1},
            dlogits_offsets, dlogits_srsrc, dlogits_base, dl_lds_01);
        DWGroup::load(hidden_s[0][0], hidden, {0, 0, 0, tile_h * 2},
                      hidden_offsets, hidden_srsrc, hidden_base, h_lds_00);
        DWGroup::load(hidden_s[0][1], hidden, {0, 0, 0, tile_h * 2 + 1},
                      hidden_offsets, hidden_srsrc, hidden_base, h_lds_01);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        #pragma unroll
        for (int kt = 0; kt < reduction_tiles; ++kt) {
            const int read_buffer = kt & 1;
            const int write_buffer = read_buffer ^ 1;

            // Global-to-LDS prefetch for kt+1 overlaps the current LDS loads
            // and MFMA work. The current buffer is never overwritten.
            if (kt + 1 < reduction_tiles) {
                const uint32_t dl_lds_0 = write_buffer ? dl_lds_10 : dl_lds_00;
                const uint32_t dl_lds_1 = write_buffer ? dl_lds_11 : dl_lds_01;
                const uint32_t h_lds_0 = write_buffer ? h_lds_10 : h_lds_00;
                const uint32_t h_lds_1 = write_buffer ? h_lds_11 : h_lds_01;
                kittens::load<2, false, DLogitsST, DWDLogitsGL, coord<DLogitsST>, DW_NUM_THREADS, coherency::non_temporal>(
                    dlogits_s[write_buffer][0], dlogits, {0, 0, kt + 1, tile_v * 2},
                    dlogits_offsets, dlogits_srsrc, dlogits_base, dl_lds_0);
                kittens::load<2, false, DLogitsST, DWDLogitsGL, coord<DLogitsST>, DW_NUM_THREADS, coherency::non_temporal>(
                    dlogits_s[write_buffer][1], dlogits, {0, 0, kt + 1, tile_v * 2 + 1},
                    dlogits_offsets, dlogits_srsrc, dlogits_base, dl_lds_1);
                DWGroup::load(hidden_s[write_buffer][0], hidden,
                              {0, 0, kt + 1, tile_h * 2}, hidden_offsets,
                              hidden_srsrc, hidden_base, h_lds_0);
                DWGroup::load(hidden_s[write_buffer][1], hidden,
                              {0, 0, kt + 1, tile_h * 2 + 1}, hidden_offsets,
                              hidden_srsrc, hidden_base, h_lds_1);
            }

            auto hs0 = subtile_inplace<DW_K_STEP, DW_HALF_WAVE_H>(
                hidden_s[read_buffer][0], {0, warp_h});
            auto hs1 = subtile_inplace<DW_K_STEP, DW_HALF_WAVE_H>(
                hidden_s[read_buffer][1], {0, warp_h});
            auto ds0 = subtile_inplace<DW_K_STEP, DW_HALF_WAVE_V>(
                dlogits_s[read_buffer][0], {0, warp_v});
            load(hidden_r0, hs0);
            load(hidden_r1, hs1);
            load(dlogits_r, ds0);
            asm volatile("s_waitcnt lgkmcnt(0)");

            __builtin_amdgcn_s_setprio(1);
            mma_AtB(accum[0][0], dlogits_r, hidden_r0, accum[0][0]);
            mma_AtB(accum[0][1], dlogits_r, hidden_r1, accum[0][1]);
            __builtin_amdgcn_s_setprio(0);

            auto ds1 = subtile_inplace<DW_K_STEP, DW_HALF_WAVE_V>(
                dlogits_s[read_buffer][1], {0, warp_v});
            load(dlogits_r, ds1);
            asm volatile("s_waitcnt lgkmcnt(0)");

            __builtin_amdgcn_s_setprio(1);
            mma_AtB(accum[1][0], dlogits_r, hidden_r0, accum[1][0]);
            mma_AtB(accum[1][1], dlogits_r, hidden_r1, accum[1][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_sched_barrier(0);

            if (kt + 1 < reduction_tiles) {
                asm volatile("s_waitcnt vmcnt(0)");
            }
            // Also protects the final buffer before a persistent workgroup
            // reuses LDS for its next output tile.
            __builtin_amdgcn_s_barrier();
        }

        store(dweight, accum[0][0],
              {0, 0, (tile_v * 2) * DW_WARPS_V + warp_v,
               tile_h * 2 * DW_WARPS_H + warp_h});
        store(dweight, accum[0][1],
              {0, 0, (tile_v * 2) * DW_WARPS_V + warp_v,
               tile_h * 2 * DW_WARPS_H + DW_WARPS_H + warp_h});
        store(dweight, accum[1][0],
              {0, 0, (tile_v * 2) * DW_WARPS_V + DW_WARPS_V + warp_v,
               tile_h * 2 * DW_WARPS_H + warp_h});
        store(dweight, accum[1][1],
              {0, 0, (tile_v * 2) * DW_WARPS_V + DW_WARPS_V + warp_v,
               tile_h * 2 * DW_WARPS_H + DW_WARPS_H + warp_h});
    }
}

#ifndef DW_OWNER_DEVICE_ONLY
extern "C" __global__ __launch_bounds__(DW_NUM_THREADS, 1) void lmhead_ce_dw_owner(
    bf16 *__restrict__ dweight_ptr,
    bf16 *__restrict__ dlogits_ptr,
    bf16 *__restrict__ hidden_ptr) {
    const int physical_wg = __builtin_amdgcn_workgroup_id_x();
    __shared__ alignment_dummy __shm[MAX_SHARED_MEMORY / sizeof(alignment_dummy)];
    lmhead_ce_dw_owner_body(physical_wg, dweight_ptr, dlogits_ptr, hidden_ptr, (int *)&__shm[0]);
}
#endif
