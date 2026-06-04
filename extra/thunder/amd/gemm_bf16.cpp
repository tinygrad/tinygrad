#include "kittens.cuh"

using namespace kittens;

#ifndef GEMM_M
constexpr int GEMM_M = 8192;
#endif
#ifndef GEMM_N
constexpr int GEMM_N = 8192;
#endif
#ifndef GEMM_K
constexpr int GEMM_K = 8192;
#endif

constexpr int NUM_WARPS = GEMM_NUM_WARPS;
using G = kittens::group<NUM_WARPS>;

__global__ __launch_bounds__(GEMM_NUM_WARPS * WARP_THREADS, 2) void hk_bf16_gemm(bf16 *C_ptr, bf16 *A_ptr, bf16 *B_ptr) {
    constexpr int M = GEMM_M, N = GEMM_N, K = GEMM_K;

    kittens::gl<bf16, 1, 1, M, K> A{A_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<bf16, 1, 1, N, K> B{B_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<bf16, 1, 1, M, N> C{C_ptr, nullptr, nullptr, nullptr, nullptr};

    constexpr int WARPS_COL = GEMM_WARPS_COL;
    constexpr int WARPS_ROW = GEMM_WARPS_ROW;
    constexpr int BLOCK_SIZE_ROW = GEMM_BLOCK_M;
    constexpr int BLOCK_SIZE_COL = GEMM_BLOCK_N;
    constexpr int BLOCK_K = GEMM_BLOCK_K;
    constexpr int blocks_per_col = N / BLOCK_SIZE_COL;
    constexpr int k_iters = K / BLOCK_K;
    constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    constexpr int HALF_BLOCK_SIZE_ROW = BLOCK_SIZE_ROW / 2;
    constexpr int HALF_BLOCK_SIZE_COL = BLOCK_SIZE_COL / 2;
    constexpr int REG_BLOCK_M = BLOCK_SIZE_ROW / WARPS_ROW / 2;
    constexpr int REG_BLOCK_N = BLOCK_SIZE_COL / WARPS_COL / 2;

    using ST_A = st_bf<HALF_BLOCK_SIZE_ROW, BLOCK_K, st_32x32_s>;
    using ST_B = st_bf<HALF_BLOCK_SIZE_COL, BLOCK_K, st_32x32_s>;
    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    using RT_A = rt_bf<REG_BLOCK_M, BLOCK_K, row_l, rt_32x16_s>;
    using RT_B = rt_bf<REG_BLOCK_N, BLOCK_K, row_l, rt_32x16_s>;
    using RT_C = rt_fl<REG_BLOCK_M, REG_BLOCK_N, col_l, rt_32x32_s>;

    RT_A a;
    RT_B b0;
    RT_B b1;
    RT_C cA;
    RT_C cB;
    RT_C cC;
    RT_C cD;

    int global_block_id = blockIdx.x;
    int block_row = global_block_id / blocks_per_col;
    int block_col = global_block_id % blocks_per_col;
    int warp_m = warpid() / WARPS_COL;
    int warp_n = warpid() % WARPS_COL;
    int tic = 0, toc = 1;

    constexpr int bytes_per_thread = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile_A = HALF_BLOCK_SIZE_ROW * BLOCK_K * sizeof(bf16) / bytes_per_memcpy;
    constexpr int memcpy_per_tile_B = HALF_BLOCK_SIZE_COL * BLOCK_K * sizeof(bf16) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile_A];
    uint32_t swizzled_offsets_B[memcpy_per_tile_B];
    G::prefill_swizzled_offsets(As[tic][0], A, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[tic][0], B, swizzled_offsets_B);

    zero(cA);
    zero(cB);
    zero(cC);
    zero(cD);

    G::load(Bs[tic][0], B, {0, 0, block_col * 2, 0}, swizzled_offsets_B);
    G::load(As[tic][0], A, {0, 0, block_row * 2, 0}, swizzled_offsets_A);
    G::load(Bs[tic][1], B, {0, 0, block_col * 2 + 1, 0}, swizzled_offsets_B);
    G::load(As[tic][1], A, {0, 0, block_row * 2 + 1, 0}, swizzled_offsets_A);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    #pragma unroll 4
    for (int k = 0; k < k_iters; k++, tic ^= 1, toc ^= 1) {
        if (k + 1 < k_iters) {
            G::load(Bs[toc][0], B, {0, 0, block_col * 2, k + 1}, swizzled_offsets_B);
            G::load(As[toc][0], A, {0, 0, block_row * 2, k + 1}, swizzled_offsets_A);
            G::load(Bs[toc][1], B, {0, 0, block_col * 2 + 1, k + 1}, swizzled_offsets_B);
            G::load(As[toc][1], A, {0, 0, block_row * 2 + 1, k + 1}, swizzled_offsets_A);
        }

        auto bs_subtile0 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[tic][0], {warp_n, 0});
        load(b0, bs_subtile0);
        auto as_subtile0 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][0], {warp_m, 0});
        load(a, as_subtile0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cA, a, b0, cA);
        __builtin_amdgcn_s_setprio(0);

        auto bs_subtile1 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
        load(b1, bs_subtile1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cB, a, b1, cB);
        __builtin_amdgcn_s_setprio(0);

        auto as_subtile1 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][1], {warp_m, 0});
        load(a, as_subtile1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cC, a, b0, cC);
        __builtin_amdgcn_s_setprio(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cD, a, b1, cD);
        __builtin_amdgcn_s_setprio(0);
        if (k + 1 < k_iters) {
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();
        }
    }

    store(C, cA, {0, 0, block_row * WARPS_ROW * 2 + warp_m, block_col * WARPS_COL * 2 + warp_n});
    store(C, cB, {0, 0, block_row * WARPS_ROW * 2 + warp_m, block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
    store(C, cC, {0, 0, block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m, block_col * WARPS_COL * 2 + warp_n});
    store(C, cD, {0, 0, block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m, block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
}
