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

constexpr int BLOCK_SIZE = 256;
constexpr int HALF_BLOCK_SIZE = BLOCK_SIZE / 2;
constexpr int K_STEP = 64;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int REG_BLOCK_M = BLOCK_SIZE / WARPS_M;
constexpr int REG_BLOCK_N = BLOCK_SIZE / WARPS_N;
constexpr int HALF_REG_BLOCK_M = REG_BLOCK_M / 2;
constexpr int HALF_REG_BLOCK_N = REG_BLOCK_N / 2;
constexpr int NUM_WARPS = WARPS_M * WARPS_N;
constexpr int NUM_THREADS = WARP_THREADS * NUM_WARPS;

using G = kittens::group<NUM_WARPS>;

// Computes C = A.T @ B where A is GEMM_K x GEMM_M and B is GEMM_K x GEMM_N.
__global__ __launch_bounds__(NUM_THREADS, 2) void hk_bf16_atb_gemm(bf16 *C_ptr, bf16 *A_ptr, bf16 *B_ptr) {
    constexpr int M = GEMM_M, N = GEMM_N, K = GEMM_K;
    static_assert(M % BLOCK_SIZE == 0 && N % BLOCK_SIZE == 0 && K % K_STEP == 0);

    kittens::gl<bf16, 1, 1, K, M> A{A_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<bf16, 1, 1, K, N> B{B_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<bf16, 1, 1, M, N> C{C_ptr, nullptr, nullptr, nullptr, nullptr};

    __shared__ alignment_dummy __shm[MAX_SHARED_MEMORY / sizeof(alignment_dummy)];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_bf<K_STEP, HALF_BLOCK_SIZE, st_32x16_s>;
    using ST_B = st_bf<K_STEP, HALF_BLOCK_SIZE, st_32x16_s>;
    ST_A (&As)[2][2] = al.allocate<ST_A, 2, 2>();
    ST_B (&Bs)[2][2] = al.allocate<ST_B, 2, 2>();

    rt_bf<K_STEP, HALF_REG_BLOCK_M, col_l, rt_32x16_s> A_tile;
    rt_bf<K_STEP, HALF_REG_BLOCK_N, col_l, rt_32x16_s> B_tile_0;
    rt_bf<K_STEP, HALF_REG_BLOCK_N, col_l, rt_32x16_s> B_tile_1;
    rt_fl<HALF_REG_BLOCK_M, HALF_REG_BLOCK_N, col_l, rt_16x16_s> C_accum[2][2];
    zero(C_accum[0][0]);
    zero(C_accum[0][1]);
    zero(C_accum[1][0]);
    zero(C_accum[1][1]);

    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    const int WGM = 8;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, 64);

    const int num_pid_m = M / BLOCK_SIZE;
    const int num_pid_n = N / BLOCK_SIZE;
    const int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    int row = pid_m;
    int col = pid_n;

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;
    const int num_tiles = K / K_STEP;

    int tic = 0;
    int toc = 1;

    using T = typename ST_A::dtype;
    constexpr int bytes_per_thread = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * K_STEP * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile / 2];
    uint32_t swizzled_offsets_B[memcpy_per_tile / 2];
    G::prefill_swizzled_offsets(As[0][0], A, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0], B, swizzled_offsets_B);

    // Use tile-local buffer resources from the generic loader. The whole-matrix SRD fast path corrupts
    // the second 128 output rows once K reaches 8192, while this path has indistinguishable throughput.
    G::load(As[tic][0], A, {0, 0, 0, row * 2}, swizzled_offsets_A);
    G::load(Bs[tic][0], B, {0, 0, 0, col * 2}, swizzled_offsets_B);
    G::load(As[tic][1], A, {0, 0, 0, row * 2 + 1}, swizzled_offsets_A);
    G::load(Bs[tic][1], B, {0, 0, 0, col * 2 + 1}, swizzled_offsets_B);

    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    G::load(As[toc][0], A, {0, 0, 1, row * 2}, swizzled_offsets_A);
    G::load(Bs[toc][0], B, {0, 0, 1, col * 2}, swizzled_offsets_B);
    G::load(Bs[toc][1], B, {0, 0, 1, col * 2 + 1}, swizzled_offsets_B);

    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

    #pragma unroll
    for (int tile = 0; tile < num_tiles - 2; tile += 2) {
        auto st_subtile_b = subtile_inplace<K_STEP, HALF_REG_BLOCK_N>(Bs[0][0], {0, warp_col});
        load(B_tile_0, st_subtile_b);
        auto st_subtile_a = subtile_inplace<K_STEP, HALF_REG_BLOCK_M>(As[0][0], {0, warp_row});
        load(A_tile, st_subtile_a);
        G::load(As[1][1], A, {0, 0, tile + 1, row * 2 + 1}, swizzled_offsets_A);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        st_subtile_b = subtile_inplace<K_STEP, HALF_REG_BLOCK_N>(Bs[0][1], {0, warp_col});
        load(B_tile_1, st_subtile_b);
        G::load(Bs[0][0], B, {0, 0, tile + 2, col * 2}, swizzled_offsets_B);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        st_subtile_a = subtile_inplace<K_STEP, HALF_REG_BLOCK_M>(As[0][1], {0, warp_row});
        load(A_tile, st_subtile_a);
        G::load(As[0][0], A, {0, 0, tile + 2, row * 2}, swizzled_offsets_A);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        st_subtile_b = subtile_inplace<K_STEP, HALF_REG_BLOCK_N>(Bs[1][0], {0, warp_col});
        load(B_tile_0, st_subtile_b);
        G::load(Bs[0][1], B, {0, 0, tile + 2, col * 2 + 1}, swizzled_offsets_B);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        st_subtile_a = subtile_inplace<K_STEP, HALF_REG_BLOCK_M>(As[1][0], {0, warp_row});
        load(A_tile, st_subtile_a);
        G::load(As[0][1], A, {0, 0, tile + 2, row * 2 + 1}, swizzled_offsets_A);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        st_subtile_b = subtile_inplace<K_STEP, HALF_REG_BLOCK_N>(Bs[1][1], {0, warp_col});
        load(B_tile_1, st_subtile_b);
        G::load(Bs[1][0], B, {0, 0, tile + 3, col * 2}, swizzled_offsets_B);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        st_subtile_a = subtile_inplace<K_STEP, HALF_REG_BLOCK_M>(As[1][1], {0, warp_row});
        load(A_tile, st_subtile_a);
        G::load(As[1][0], A, {0, 0, tile + 3, row * 2}, swizzled_offsets_A);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        G::load(Bs[1][1], B, {0, 0, tile + 3, col * 2 + 1}, swizzled_offsets_B);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    {
        int tile = num_tiles - 2;
        auto st_subtile_b = subtile_inplace<K_STEP, HALF_REG_BLOCK_N>(Bs[tic][0], {0, warp_col});
        load(B_tile_0, st_subtile_b);
        auto st_subtile_a = subtile_inplace<K_STEP, HALF_REG_BLOCK_M>(As[tic][0], {0, warp_row});
        load(A_tile, st_subtile_a);
        G::load(As[toc][1], A, {0, 0, tile + 1, row * 2 + 1}, swizzled_offsets_A);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        st_subtile_b = subtile_inplace<K_STEP, HALF_REG_BLOCK_N>(Bs[tic][1], {0, warp_col});
        load(B_tile_1, st_subtile_b);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        st_subtile_a = subtile_inplace<K_STEP, HALF_REG_BLOCK_M>(As[tic][1], {0, warp_row});
        load(A_tile, st_subtile_a);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        mma_AtB(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        tic ^= 1, toc ^= 1;
    }

    {
        auto st_subtile_b = subtile_inplace<K_STEP, HALF_REG_BLOCK_N>(Bs[tic][0], {0, warp_col});
        load(B_tile_0, st_subtile_b);
        auto st_subtile_a = subtile_inplace<K_STEP, HALF_REG_BLOCK_M>(As[tic][0], {0, warp_row});
        load(A_tile, st_subtile_a);
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        st_subtile_b = subtile_inplace<K_STEP, HALF_REG_BLOCK_N>(Bs[tic][1], {0, warp_col});
        load(B_tile_1, st_subtile_b);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        st_subtile_a = subtile_inplace<K_STEP, HALF_REG_BLOCK_M>(As[tic][1], {0, warp_row});
        load(A_tile, st_subtile_a);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
        mma_AtB(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }

    store(C, C_accum[0][0], {0, 0, (row * 2) * WARPS_M + warp_row, col * 2 * WARPS_N + warp_col});
    store(C, C_accum[0][1], {0, 0, (row * 2) * WARPS_M + warp_row, col * 2 * WARPS_N + WARPS_N + warp_col});
    store(C, C_accum[1][0], {0, 0, (row * 2) * WARPS_M + WARPS_M + warp_row, col * 2 * WARPS_N + warp_col});
    store(C, C_accum[1][1], {0, 0, (row * 2) * WARPS_M + WARPS_M + warp_row, col * 2 * WARPS_N + WARPS_N + warp_col});
}
