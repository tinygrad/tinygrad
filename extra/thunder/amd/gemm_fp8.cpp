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

// scale_mode: 0=no scale, 1=x only, 2=w only, 3=both
#ifndef SCALE_MODE
#define SCALE_MODE 3
#endif

constexpr int NUM_WARPS = 4;

using G = kittens::group<NUM_WARPS>;

__global__ __launch_bounds__(256, 1) void hk_fp8_gemm(bf16 *C_ptr, fp8e4m3 *A_ptr, fp8e4m3 *B_ptr
#if SCALE_MODE == 1
    , float *x_scale_ptr
#elif SCALE_MODE == 2
    , float *w_scale_ptr
#elif SCALE_MODE == 3
    , float *x_scale_ptr, float *w_scale_ptr
#endif
) {
    constexpr int M = GEMM_M, N = GEMM_N, K = GEMM_K;

    kittens::gl<fp8e4m3, 1, 1, M, K> A{A_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<fp8e4m3, 1, 1, N, K> B{B_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<bf16, 1, 1, M, N> C{C_ptr, nullptr, nullptr, nullptr, nullptr};

    constexpr int WARPS_COL = 2;
    constexpr int WARPS_ROW = 2;
    constexpr int BLOCK_SIZE_ROW = 128;
    constexpr int BLOCK_SIZE_COL = 128;
    constexpr int BLOCK_K = 128;
    constexpr int k_step = BLOCK_K;
    constexpr int blocks_col = N / BLOCK_SIZE_COL;
    constexpr int k_iters = K / BLOCK_K;

    using ST_A = st_fp8e4m3<BLOCK_SIZE_ROW / 2, BLOCK_K, st_16x128_s>;
    using ST_B = st_fp8e4m3<BLOCK_SIZE_COL / 2, BLOCK_K, st_16x128_s>;
    using RT_A = rt_fp8e4m3<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>;
    using RT_B = rt_fp8e4m3<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>;
    using RT_C = rt_fl<BLOCK_SIZE_ROW / 2 / WARPS_ROW, BLOCK_SIZE_COL / 2 / WARPS_COL, col_l, rt_16x16_s>;

    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    RT_C c[2][2];

    int global_block_id = blockIdx.x;
    const int block_row = global_block_id / blocks_col;
    const int block_col = global_block_id % blocks_col;

    int curr = 0, next = 1;
    int warp_m = warpid() / WARPS_COL;
    int warp_n = warpid() % WARPS_COL;

    {
    __builtin_amdgcn_sched_barrier(0);
    RT_A a[2];
    RT_B b[2];

    G::load(As[curr][0], A, {0, 0, block_row*WARPS_ROW, 0});
    G::load(Bs[curr][0], B, {0, 0, block_col*WARPS_COL, 0});
    G::load(Bs[curr][1], B, {0, 0, block_col*WARPS_COL+1, 0});
    G::load(As[curr][1], A, {0, 0, block_row*WARPS_ROW+1, 0});

    zero(c[0][0]);
    zero(c[0][1]);
    zero(c[1][0]);
    zero(c[1][1]);

    G::load(As[next][0], A, {0, 0, block_row*WARPS_ROW, 1});
    G::load(Bs[next][0], B, {0, 0, block_col*WARPS_COL, 1});
    G::load(Bs[next][1], B, {0, 0, block_col*WARPS_COL+1, 1});
    G::load(As[next][1], A, {0, 0, block_row*WARPS_ROW+1, 1});

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(14)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto a_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[curr][0], {warp_m, 0});
    load(a[0], a_subtile_0);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(12)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto b_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[curr][0], {warp_n, 0});
    load(b[0], b_subtile_0);

    #pragma unroll
    for (int k = 0; k < k_iters - 2; ++k, curr ^= 1, next ^= 1) {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(8)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[curr][1], {warp_n, 0});
        G::load(As[curr][0], A, {0, 0, block_row*WARPS_ROW, k + 2});
        load(b[1], bs_subtile_1);
        mma_ABt(c[0][0], a[0], b[0], c[0][0]);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[curr][1], {warp_m, 0});
        G::load(Bs[curr][0], B, {0, 0, block_col*WARPS_COL, k + 2});
        load(a[1], a_subtile_1);
        mma_ABt(c[0][1], a[0], b[1], c[0][1]);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(8)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_0_next = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[next][0], {warp_m, 0});
        G::load(Bs[curr][1], B, {0, 0, block_col*WARPS_COL+1, k + 2});
        load(a[0], a_subtile_0_next);
        mma_ABt(c[1][0], a[1], b[0], c[1][0]);

        auto b_subtile_0_next = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[next][0], {warp_n, 0});
        G::load(As[curr][1], A, {0, 0, block_row*WARPS_ROW+1, k + 2});
        load(b[0], b_subtile_0_next);
        mma_ABt(c[1][1], a[1], b[1], c[1][1]);
    }

    {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(8)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto b_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[curr][1], {warp_n, 0});
        load(b[1], b_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][0], a[0], b[0], c[0][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[curr][1], {warp_m, 0});
        load(a[1], a_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][1], a[0], b[1], c[0][1]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_0_next = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[next][0], {warp_m, 0});
        load(a[0], a_subtile_0_next);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][0], a[1], b[0], c[1][0]);
        __builtin_amdgcn_sched_barrier(0);

        auto b_subtile_0_next = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[next][0], {warp_n, 0});
        load(b[0], b_subtile_0_next);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][1], a[1], b[1], c[1][1]);
        __builtin_amdgcn_sched_barrier(0);

        curr ^= 1;
        next ^= 1;
    }

    {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto b_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[curr][1], {warp_n, 0});
        load(b[1], b_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][0], a[0], b[0], c[0][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[curr][1], {warp_m, 0});
        load(a[1], a_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][1], a[0], b[1], c[0][1]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][0], a[1], b[0], c[1][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][1], a[1], b[1], c[1][1]);
        __builtin_amdgcn_sched_barrier(0);
    }
    __builtin_amdgcn_sched_barrier(0);
    }

   // apply x_scale * w_scale before bf16 store to prevent overflow
#if SCALE_MODE == 1
    float scale = *x_scale_ptr;
    mul(c[0][0], c[0][0], scale);
    mul(c[0][1], c[0][1], scale);
    mul(c[1][0], c[1][0], scale);
    mul(c[1][1], c[1][1], scale);
#elif SCALE_MODE == 2
    float scale = *w_scale_ptr;
    mul(c[0][0], c[0][0], scale);
    mul(c[0][1], c[0][1], scale);
    mul(c[1][0], c[1][0], scale);
    mul(c[1][1], c[1][1], scale);
#elif SCALE_MODE == 3
    float scale = *x_scale_ptr * *w_scale_ptr;
    mul(c[0][0], c[0][0], scale);
    mul(c[0][1], c[0][1], scale);
    mul(c[1][0], c[1][0], scale);
    mul(c[1][1], c[1][1], scale);
#endif

    store(C, c[0][0], {0, 0, (block_row * WARPS_ROW) * 2 + warp_m, (block_col * WARPS_COL) * 2 + warp_n});
    store(C, c[0][1], {0, 0, (block_row * WARPS_ROW) * 2 + warp_m, (block_col * WARPS_COL + 1) * 2 + warp_n});
    store(C, c[1][0], {0, 0, (block_row * WARPS_ROW + 1) * 2 + warp_m, (block_col * WARPS_COL) * 2 + warp_n});
    store(C, c[1][1], {0, 0, (block_row * WARPS_ROW + 1) * 2 + warp_m, (block_col * WARPS_COL + 1) * 2 + warp_n});
}
