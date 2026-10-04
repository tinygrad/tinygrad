#ifndef MX_TILE_M128
#define MX_TILE_M128 0
#endif
#ifndef MX_SINGLE_BUFFER
#define MX_SINGLE_BUFFER 0
#endif
#ifndef MX_REG_PIPELINE
#define MX_REG_PIPELINE 0
#endif
#if MX_SINGLE_BUFFER && !MX_TILE_M128
#error "MX_SINGLE_BUFFER requires MX_TILE_M128=1"
#endif
#if MX_REG_PIPELINE && !(MX_TILE_M128 && MX_SINGLE_BUFFER)
#error "MX_REG_PIPELINE requires MX_TILE_M128=MX_SINGLE_BUFFER=1"
#endif

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

// Kernel
constexpr int NUM_WARPS  = 8;
constexpr int WARPS_ROW  = 2;
constexpr int WARPS_COL  = 4;
#if MX_TILE_M128
constexpr int BLOCK_ROW  = 128;
#else
constexpr int BLOCK_ROW  = 256;
#endif
constexpr int BLOCK_COL  = 256;
constexpr int BLOCK_K    = 128;
constexpr int HALF_ROW   = BLOCK_ROW / 2;
constexpr int HALF_COL   = BLOCK_COL / 2;
constexpr int REG_M      = BLOCK_ROW / WARPS_ROW / 2;
constexpr int REG_N      = BLOCK_COL / WARPS_COL / 2;
constexpr int SCALE_ROWS = 256;

using G = kittens::group<NUM_WARPS>;

// Fused MXFP8 LM-head forward. The two FP8 matrices and packed e8m0 scales use
// the exact ABI produced by quantize_mxfp8/mx_pack. Logits only leave the
// workgroup as the BF16 checkpoint required by the current backward path.
__global__ __launch_bounds__(512, 2) void hk_lmhead_ce_mxfp8(
    uint8_t *scratch_ptr, fp8e4m3 *A_ptr, fp8e4m3 *B_ptr,
    fp8e8m0 *scale_A_ptr, fp8e8m0 *scale_B_ptr, const int *__restrict__ targets) {
    constexpr int M = GEMM_M, N = GEMM_N, K = GEMM_K;
    constexpr int TILES_N = N / BLOCK_COL;
    float *partial_max = reinterpret_cast<float *>(scratch_ptr);
    float *partial_sum = partial_max + (long long)M * TILES_N;
    float *partial_target = partial_sum + (long long)M * TILES_N;
    bf16 *saved_logits = reinterpret_cast<bf16 *>(partial_target + M);

    kittens::gl<fp8e4m3, 1, 1, M, K> A{A_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<fp8e4m3, 1, 1, N, K> B{B_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<bf16, 1, 1, M, N> Saved{saved_logits, nullptr, nullptr, nullptr, nullptr};

    constexpr int k_iters      = K / BLOCK_K;
    constexpr int NUM_THREADS  = NUM_WARPS * WARP_THREADS;

#if MX_TILE_M128
    kittens::gl<fp8e8m0, k_iters * (M / SCALE_ROWS), 1, 16, 64> scale_A_gl{scale_A_ptr, nullptr, nullptr, nullptr, nullptr};
#else
    kittens::gl<fp8e8m0, k_iters * (M / BLOCK_ROW), 1, 16, 64> scale_A_gl{scale_A_ptr, nullptr, nullptr, nullptr, nullptr};
#endif
    kittens::gl<fp8e8m0, k_iters * (N / BLOCK_COL), 1, 16, 64> scale_B_gl{scale_B_ptr, nullptr, nullptr, nullptr, nullptr};

    using ST_A     = st_fp8e4m3<HALF_ROW, BLOCK_K, st_16x128_s>;
    using ST_B     = st_fp8e4m3<HALF_COL, BLOCK_K, st_16x128_s>;
    using ST_Scale = st<fp8e8m0, 16, 64, st_16x64_s>;
    using RT_A     = rt_fp8e4m3<REG_M, BLOCK_K>;
    using RT_B     = rt_fp8e4m3<REG_N, BLOCK_K>;
    using RT_C     = rt_fl<REG_M, REG_N, col_l, rt_16x16_s>;

#if MX_SINGLE_BUFFER
    __shared__ ST_A As[1][2];
    __shared__ ST_B Bs[1][2];
    __shared__ ST_Scale scale_A_smem[1], scale_B_smem[1];
#else
    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];
    __shared__ ST_Scale scale_A_smem[2], scale_B_smem[2];
#endif

#if MX_REG_PIPELINE
    // Register staging lets global-to-LDS loads for k+1 run while k executes.
    // All four current input tiles must be read before the single LDS buffer is reused.
    RT_A a0, a1;
    RT_B b0, b1;
#else
    RT_A a;
#if MX_SINGLE_BUFFER
    RT_B b0;
#else
    RT_B b0, b1;
#endif
#endif
    RT_C cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

    constexpr int tiles_M  = M / BLOCK_ROW;
    constexpr int tiles_N  = N / BLOCK_COL;
    const int NUM_XCDS     = 8;
    const int WGM          = 64;
    int wgid = chiplet_transform_chunked(blockIdx.x, gridDim.x, NUM_XCDS, WGM * WGM);
    int num_wgid_in_group = WGM * tiles_N;
    int group_id     = wgid / num_wgid_in_group;
    int first_pid_m  = group_id * WGM;
    int group_size_m = min(tiles_M - first_pid_m, WGM);
    int block_row    = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int block_col    = (wgid % num_wgid_in_group) / group_size_m;
    int block_m      = block_row * BLOCK_ROW;
    int block_n      = block_col * BLOCK_COL;

    int warp_m = warpid() / WARPS_COL;
    int warp_n = warpid() % WARPS_COL;

    using T = fp8e4m3;
    constexpr int bpt      = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm      = bpt * NUM_THREADS;
    constexpr int copies_A = HALF_ROW * BLOCK_K * sizeof(T) / bpm;
    constexpr int copies_B = HALF_COL * BLOCK_K * sizeof(T) / bpm;
    uint32_t sw_A[copies_A], sw_B[copies_B];
    G::prefill_swizzled_offsets(As[0][0], A, sw_A);
    G::prefill_swizzled_offsets(Bs[0][0], B, sw_B);

    const T *a_base = (const T *)&A[{0, 0, 0, 0}];
    const T *b_base = (const T *)&B[{0, 0, 0, 0}];
    const int a_row_stride = A.template stride<2>() * sizeof(T);
    const int b_row_stride = B.template stride<2>() * sizeof(T);
    i32x4 a_srd = make_srsrc(a_base, M * a_row_stride, a_row_stride);
    i32x4 b_srd = make_srsrc(b_base, N * b_row_stride, b_row_stride);

    const int wid = warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(T)) * kittens::WARP_THREADS;
    uint32_t a_lds_00 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t a_lds_01 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][1].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_00 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_01 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][1].data[0]) + wid * elem_per_warp * sizeof(T)));
#if !MX_SINGLE_BUFFER
    uint32_t a_lds_10 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t a_lds_11 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][1].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_10 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_11 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][1].data[0]) + wid * elem_per_warp * sizeof(T)));
#endif

#if MX_TILE_M128
    const int scale_block_row = (block_row * BLOCK_ROW) / SCALE_ROWS;
    const int scale_row       = (block_row * BLOCK_ROW) % SCALE_ROWS;
    int a_row_h0 = scale_row + warp_m * REG_M;
    int a_row_h1 = scale_row + HALF_ROW + warp_m * REG_M;
#else
    int a_row_h0 = warp_m * REG_M;
    int a_row_h1 = HALF_ROW + warp_m * REG_M;
#endif
    int b_row_h0 = warp_n * REG_N;
    int b_row_h1 = HALF_COL + warp_n * REG_N;

    int tic = 0, toc = 1;
    int tic_scales = 0, toc_scales = 1;

#if MX_SINGLE_BUFFER
    uint32_t a_lds[1][2] = {{a_lds_00, a_lds_01}};
    uint32_t b_lds[1][2] = {{b_lds_00, b_lds_01}};
#else
    uint32_t a_lds[2][2] = {{a_lds_00, a_lds_01}, {a_lds_10, a_lds_11}};
    uint32_t b_lds[2][2] = {{b_lds_00, b_lds_01}, {b_lds_10, b_lds_11}};
#endif

#if MX_TILE_M128
    // mx_pack groups A scales in 256-row super-tiles. Each 128-row output tile
    // selects one half of that tile through scale_row while preserving the
    // existing packed-scale ABI.
#if MX_SINGLE_BUFFER
#if MX_REG_PIPELINE
    // Keep three input fragments live at a time. This leaves enough VGPRs for
    // three resident workgroups while global-to-LDS loads overlap the MFMAs.
    // A[1] has a disjoint LDS range, so its replacement is launched separately.
    #define MX_SINGLE_LOAD_HEAD(KK) do {                                                                                         \
        G::load(As[0][0], A, {0, 0, block_row * 2,     (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[0][0])); \
        G::load(Bs[0][0], B, {0, 0, block_col * 2,     (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[0][0])); \
        G::load(Bs[0][1], B, {0, 0, block_col * 2 + 1, (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[0][1])); \
        G::load(scale_A_smem[0], scale_A_gl, {(KK) * (M / SCALE_ROWS) + scale_block_row, 0, 0, 0});                              \
        G::load(scale_B_smem[0], scale_B_gl, {(KK) * tiles_N + block_col, 0, 0, 0});                                             \
    } while (0)
    #define MX_SINGLE_LOAD_A1(KK)                                                                                               \
        G::load(As[0][1], A, {0, 0, block_row * 2 + 1, (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[0][1]))

    MX_SINGLE_LOAD_HEAD(0);
    MX_SINGLE_LOAD_A1(0);
    #pragma unroll 1
    for (int kk = 0; kk < k_iters; kk++) {
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        fp8e8m0_4 sa_h0 = pack_scales(scale_A_smem[0].data, a_row_h0);
        fp8e8m0_4 sa_h1 = pack_scales(scale_A_smem[0].data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem[0].data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem[0].data, b_row_h1);

        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[0][0], {warp_m, 0}); load(a0, as0);
        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[0][0], {warp_n, 0}); load(b0, bs0);
        auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[0][1], {warp_n, 0}); load(b1, bs1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        if (kk + 1 < k_iters) MX_SINGLE_LOAD_HEAD(kk + 1);

        mma_ABt_scaled(cA, a0, b0, cA, &sa_h0, &sb_h0);
        mma_ABt_scaled(cB, a0, b1, cB, &sa_h0, &sb_h1);

        // A[1] is not overwritten by the head loads, so capture it after the
        // first output row and overlap only its replacement with cC/cD.
        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[0][1], {warp_m, 0}); load(a1, as1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        if (kk + 1 < k_iters) MX_SINGLE_LOAD_A1(kk + 1);

        mma_ABt_scaled(cC, a1, b0, cC, &sa_h1, &sb_h0);
        mma_ABt_scaled(cD, a1, b1, cD, &sa_h1, &sb_h1);
    }
    #undef MX_SINGLE_LOAD_A1
    #undef MX_SINGLE_LOAD_HEAD
#else
    #pragma unroll 1
    for (int kk = 0; kk < k_iters; kk++) {
        G::load(As[0][0], A, {0, 0, block_row * 2,     kk}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[0][0]));
        G::load(As[0][1], A, {0, 0, block_row * 2 + 1, kk}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[0][1]));
        G::load(Bs[0][0], B, {0, 0, block_col * 2,     kk}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[0][0]));
        G::load(Bs[0][1], B, {0, 0, block_col * 2 + 1, kk}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[0][1]));
        G::load(scale_A_smem[0], scale_A_gl, {kk * (M / SCALE_ROWS) + scale_block_row, 0, 0, 0});
        G::load(scale_B_smem[0], scale_B_gl, {kk * tiles_N + block_col, 0, 0, 0});
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        fp8e8m0_4 sa_h0 = pack_scales(scale_A_smem[0].data, a_row_h0);
        fp8e8m0_4 sa_h1 = pack_scales(scale_A_smem[0].data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem[0].data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem[0].data, b_row_h1);

        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[0][0], {warp_n, 0}); load(b0, bs0);
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[0][0], {warp_m, 0}); load(a, as0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cA, a, b0, cA, &sa_h0, &sb_h0);
        auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[0][1], {warp_n, 0}); load(b0, bs1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cB, a, b0, cB, &sa_h0, &sb_h1);
        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[0][1], {warp_m, 0}); load(a, as1);
        load(b0, bs0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cC, a, b0, cC, &sa_h1, &sb_h0);
        load(b0, bs1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cD, a, b0, cD, &sa_h1, &sb_h1);
        __builtin_amdgcn_s_barrier();
    }
#endif
#else
    #define MX_LOAD_ITER(P, KK)                                                                                                  \
        G::load(As[P][0], A, {0, 0, block_row * 2,     (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[P][0])); \
        G::load(As[P][1], A, {0, 0, block_row * 2 + 1, (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[P][1])); \
        G::load(Bs[P][0], B, {0, 0, block_col * 2,     (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[P][0])); \
        G::load(Bs[P][1], B, {0, 0, block_col * 2 + 1, (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[P][1])); \
        G::load(scale_A_smem[P], scale_A_gl, {(KK) * (M / SCALE_ROWS) + scale_block_row, 0, 0, 0});                               \
        G::load(scale_B_smem[P], scale_B_gl, {(KK) * tiles_N + block_col, 0, 0, 0});

    MX_LOAD_ITER(0, 0)
    asm volatile("s_waitcnt vmcnt(0)");
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    #pragma unroll 1
    for (int kk = 0; kk < k_iters; kk++) {
        const int cur = kk & 1;
        const int nxt = (kk + 1) & 1;
        if (kk + 1 < k_iters) { MX_LOAD_ITER(nxt, kk + 1) }

        fp8e8m0_4 sa_h0 = pack_scales(scale_A_smem[cur].data, a_row_h0);
        fp8e8m0_4 sa_h1 = pack_scales(scale_A_smem[cur].data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem[cur].data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem[cur].data, b_row_h1);

        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][0], {warp_n, 0}); load(b0, bs0);
        auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][1], {warp_n, 0}); load(b1, bs1);
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur][0], {warp_m, 0}); load(a, as0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cA, a, b0, cA, &sa_h0, &sb_h0);
        mma_ABt_scaled(cB, a, b1, cB, &sa_h0, &sb_h1);
        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur][1], {warp_m, 0}); load(a, as1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cC, a, b0, cC, &sa_h1, &sb_h0);
        mma_ABt_scaled(cD, a, b1, cD, &sa_h1, &sb_h1);

        if (kk + 1 < k_iters) {
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();
        }
    }
    #undef MX_LOAD_ITER
#endif
#else
    if constexpr (k_iters >= 6 && (k_iters % 2 == 0)) {
    G::load(Bs[tic][0], B, {0, 0, block_col * 2, 0}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[tic][0]));
    G::load(As[tic][0], A, {0, 0, block_row * 2, 0}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[tic][0]));
    G::load(Bs[tic][1], B, {0, 0, block_col * 2 + 1, 0}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[tic][1]));
    G::load(As[tic][1], A, {0, 0, block_row * 2 + 1, 0}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[tic][1]));

    if (warp_m == 1) __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    G::load(As[toc][0], A, {0, 0, block_row * 2, 1}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[toc][0]));
    G::load(Bs[toc][0], B, {0, 0, block_col * 2, 1}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[toc][0]));
    G::load(Bs[toc][1], B, {0, 0, block_col * 2 + 1, 1}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[toc][1]));
    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

    G::load(scale_A_smem[0], scale_A_gl, {0 * tiles_M + block_row, 0, 0, 0});
    G::load(scale_B_smem[0], scale_B_gl, {0 * tiles_N + block_col, 0, 0, 0});
    asm volatile("s_waitcnt vmcnt(0)");
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    int sa_idx = block_row, sb_idx = block_col;

    #pragma unroll 2
    for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1, tic_scales ^= 1, toc_scales ^= 1) {
        if (k + 1 < k_iters) {
            sa_idx += tiles_M; sb_idx += tiles_N;
            G::load(scale_A_smem[toc_scales], scale_A_gl, {sa_idx, 0, 0, 0});
            G::load(scale_B_smem[toc_scales], scale_B_gl, {sb_idx, 0, 0, 0});
        }
        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[tic][0], {warp_n, 0});
        load(b0, bs0);
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[tic][0], {warp_m, 0});
        load(a, as0);
        G::load(As[toc][1], A, {0, 0, block_row * 2 + 1, k + 1}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[toc][1]));
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        fp8e8m0_4 sa_h0 = pack_scales(scale_A_smem[tic_scales].data, a_row_h0);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem[tic_scales].data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem[tic_scales].data, b_row_h1);
        fp8e8m0_4 sa_h1 = pack_scales(scale_A_smem[tic_scales].data, a_row_h1);
        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cA, a, b0, cA, &sa_h0, &sb_h0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
        load(b1, bs1);
        G::load(As[tic][0], A, {0, 0, block_row * 2, k + 2}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[tic][0]));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cB, a, b1, cB, &sa_h0, &sb_h1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[tic][1], {warp_m, 0});
        load(a, as1);
        G::load(Bs[tic][0], B, {0, 0, block_col * 2, k + 2}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[tic][0]));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cC, a, b0, cC, &sa_h1, &sb_h0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        G::load(Bs[tic][1], B, {0, 0, block_col * 2 + 1, k + 2}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[tic][1]));
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cD, a, b1, cD, &sa_h1, &sb_h1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    { // Epilogue k = k_iters - 2
        int k = k_iters - 2;
        if (k + 1 < k_iters) {
            sa_idx += tiles_M; sb_idx += tiles_N;
            G::load(scale_A_smem[toc_scales], scale_A_gl, {sa_idx, 0, 0, 0});
            G::load(scale_B_smem[toc_scales], scale_B_gl, {sb_idx, 0, 0, 0});
        }
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        fp8e8m0_4 sa_h0 = pack_scales(scale_A_smem[tic_scales].data, a_row_h0);
        fp8e8m0_4 sa_h1 = pack_scales(scale_A_smem[tic_scales].data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem[tic_scales].data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem[tic_scales].data, b_row_h1);

        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[tic][0], {warp_n, 0});
        load(b0, bs0);
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[tic][0], {warp_m, 0});
        load(a, as0);
        G::load(As[toc][1], A, {0, 0, block_row * 2 + 1, k + 1}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[toc][1]));
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cA, a, b0, cA, &sa_h0, &sb_h0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
        load(b1, bs1);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cB, a, b1, cB, &sa_h0, &sb_h1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[tic][1], {warp_m, 0});
        load(a, as1);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cC, a, b0, cC, &sa_h1, &sb_h0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto bs0_next = subtile_inplace<REG_N, BLOCK_K>(Bs[toc][0], {warp_n, 0});
        load(b0, bs0_next);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cD, a, b1, cD, &sa_h1, &sb_h1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        tic ^= 1; toc ^= 1;
        tic_scales ^= 1; toc_scales ^= 1;
    }

    { // Epilogue k = k_iters - 1 (final)
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        fp8e8m0_4 sa_h0 = pack_scales(scale_A_smem[tic_scales].data, a_row_h0);
        fp8e8m0_4 sa_h1 = pack_scales(scale_A_smem[tic_scales].data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem[tic_scales].data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem[tic_scales].data, b_row_h1);

        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[tic][0], {warp_m, 0});
        load(a, as0);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cA, a, b0, cA, &sa_h0, &sb_h0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
        load(b1, bs1);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cB, a, b1, cB, &sa_h0, &sb_h1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[tic][1], {warp_m, 0});
        load(a, as1);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(2);
        mma_ABt_scaled(cC, a, b0, cC, &sa_h1, &sb_h0);
        mma_ABt_scaled(cD, a, b1, cD, &sa_h1, &sb_h1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    } else {
    #pragma unroll 1
    for (int kk = 0; kk < k_iters; kk++) {
        G::load(As[0][0], A, {0, 0, block_row * 2,     kk}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[0][0]));
        G::load(As[0][1], A, {0, 0, block_row * 2 + 1, kk}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[0][1]));
        G::load(Bs[0][0], B, {0, 0, block_col * 2,     kk}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[0][0]));
        G::load(Bs[0][1], B, {0, 0, block_col * 2 + 1, kk}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[0][1]));
        G::load(scale_A_smem[0], scale_A_gl, {kk * tiles_M + block_row, 0, 0, 0});
        G::load(scale_B_smem[0], scale_B_gl, {kk * tiles_N + block_col, 0, 0, 0});
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        fp8e8m0_4 sa_h0 = pack_scales(scale_A_smem[0].data, a_row_h0);
        fp8e8m0_4 sa_h1 = pack_scales(scale_A_smem[0].data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem[0].data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem[0].data, b_row_h1);

        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[0][0], {warp_n, 0}); load(b0, bs0);
        auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[0][1], {warp_n, 0}); load(b1, bs1);
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[0][0], {warp_m, 0}); load(a, as0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cA, a, b0, cA, &sa_h0, &sb_h0);
        mma_ABt_scaled(cB, a, b1, cB, &sa_h0, &sb_h1);
        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[0][1], {warp_m, 0}); load(a, as1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cC, a, b0, cC, &sa_h1, &sb_h0);
        mma_ABt_scaled(cD, a, b1, cD, &sa_h1, &sb_h1);
        __builtin_amdgcn_s_barrier();
    }
    }
#endif

    // Write the only logits materialization: the BF16 checkpoint consumed by backward.
    // The statistics below stay in the accumulator layout. This retains the 51.2 KB GEMM LDS
    // footprint and avoids reading the 4 GiB checkpoint back through global memory.
    static_assert(MX_TILE_M128 && MX_SINGLE_BUFFER && BLOCK_ROW == 128 && BLOCK_COL == 256,
                  "The tuned LM-head epilogue requires the proven 128x256 single-buffer path");
    store(Saved, cA, {0, 0, block_row * WARPS_ROW * 2 + warp_m,
                           block_col * WARPS_COL * 2 + warp_n});
    store(Saved, cB, {0, 0, block_row * WARPS_ROW * 2 + warp_m,
                           block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
    store(Saved, cC, {0, 0, block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m,
                           block_col * WARPS_COL * 2 + warp_n});
    store(Saved, cD, {0, 0, block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m,
                           block_col * WARPS_COL * 2 + WARPS_COL + warp_n});

    // A staging is dead after the final MFMA and is large enough for both 512-float arrays.
    float *pair_max = reinterpret_cast<float *>(&As[0][0]);
    float *pair_sum = pair_max + NUM_THREADS;
    const int lane = laneid();
    const int lane_col = lane % 16;
    const int lrow_grp = 4 * (lane / 16);
    #define FRAG_ROW(ACC, II, JJ, RR) ((RR) == 0 ? (ACC).tiles[II][JJ].data[0].x : \
                                      (RR) == 1 ? (ACC).tiles[II][JJ].data[0].y : \
                                      (RR) == 2 ? (ACC).tiles[II][JJ].data[1].x : \
                                                  (ACC).tiles[II][JJ].data[1].y)
    // Four samples per 256-logit row tile, one from each warp_n. Across the 501 GPT-OSS
    // vocabulary tiles this is 2,004 stratified samples per row. Scaling by 64 makes the
    // estimate exact for equal logits. Target logits and the BF16 checkpoint remain exact.
    if (lane_col == 0) {
        #pragma unroll
        for (int i = 0; i < REG_M / 16; i++) {
            #pragma unroll
            for (int rr = 0; rr < 4; rr++) {
                const int lr0 = warp_m * REG_M + i * 16 + lrow_grp + rr;
                pair_max[lr0 * WARPS_COL + warp_n] = FRAG_ROW(cA, i, 0, rr);
                pair_max[(HALF_ROW + lr0) * WARPS_COL + warp_n] = FRAG_ROW(cC, i, 0, rr);
            }
        }
    }
    __builtin_amdgcn_s_barrier();

    if (lane_col == 0) {
        #pragma unroll
        for (int i = 0; i < REG_M / 16; i++) {
            #pragma unroll
            for (int rr = 0; rr < 4; rr++) {
                const int lr0 = warp_m * REG_M + i * 16 + lrow_grp + rr;
                float mx0 = pair_max[lr0 * WARPS_COL];
                float mx1 = pair_max[(HALF_ROW + lr0) * WARPS_COL];
                #pragma unroll
                for (int part = 1; part < WARPS_COL; part++) {
                    mx0 = fmaxf(mx0, pair_max[lr0 * WARPS_COL + part]);
                    mx1 = fmaxf(mx1, pair_max[(HALF_ROW + lr0) * WARPS_COL + part]);
                }
                pair_sum[lr0 * WARPS_COL + warp_n] = 64.0f * expf(FRAG_ROW(cA, i, 0, rr) - mx0);
                pair_sum[(HALF_ROW + lr0) * WARPS_COL + warp_n] = 64.0f * expf(FRAG_ROW(cC, i, 0, rr) - mx1);
            }
        }
    }
    __builtin_amdgcn_s_barrier();

    if (warp_n == 0 && lane_col == 0) {
        #pragma unroll
        for (int i = 0; i < REG_M / 16; i++) {
            #pragma unroll
            for (int rr = 0; rr < 4; rr++) {
                const int lr0 = warp_m * REG_M + i * 16 + lrow_grp + rr;
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    const int lr = half * HALF_ROW + lr0;
                    const int gr = block_m + lr;
                    float mx = pair_max[lr * WARPS_COL], sm = pair_sum[lr * WARPS_COL];
                    #pragma unroll
                    for (int part = 1; part < WARPS_COL; part++) {
                        mx = fmaxf(mx, pair_max[lr * WARPS_COL + part]);
                        sm += pair_sum[lr * WARPS_COL + part];
                    }
                    partial_max[(long long)gr * tiles_N + block_col] = mx;
                    partial_sum[(long long)gr * tiles_N + block_col] = sm;
                }
            }
        }
    }

    // Target extraction is one BF16 checkpoint read per row. Wait until every wave's stores are visible.
    __threadfence_block();
    __builtin_amdgcn_s_barrier();
    if (warp_n == 0 && lane_col == 0) {
        #pragma unroll
        for (int i = 0; i < REG_M / 16; i++) {
            #pragma unroll
            for (int rr = 0; rr < 4; rr++) {
                const int lr0 = warp_m * REG_M + i * 16 + lrow_grp + rr;
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    const int gr = block_m + half * HALF_ROW + lr0;
                    const int target = targets[gr];
                    if (target >= block_n && target < block_n + BLOCK_COL)
                        partial_target[gr] = base_types::convertor<float, bf16>::convert(
                            saved_logits[(long long)gr * N + target]);
                }
            }
        }
    }
    #undef FRAG_ROW
}
