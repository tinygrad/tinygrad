#include "kittens.cuh"

using namespace kittens;

// Fused MoE DOWN gemm + per-expert bias + router-weighted COMBINE scatter. Identical MMA loop to the base grouped
// gemm (grouped_mxfp8_gemm.cpp); the epilogue, on the register accumulator tile, folds:
//   z = rne_bf16( trunc_bf16(gemm_accum) + w_down_bias[e] )     (byte-identical to the reference z, written to Z_ptr)
//   contrib = rne_bf16( z * row_weight[row] )                   (matches combine's bf16 product)
//   atomicAdd_fp32( Out[dest_token[row], col], contrib )        (weighted scatter into the token output row)
// Out is an fp32 accumulator (zero-init by the wrapper); a downstream cast lowers it to bf16. The scatter is
// order-dependent across the k experts a token routes to (k different workgroups), so Out is NOT byte-exact with the
// deterministic (sel*weights).sum(k) combine -- it is fp32-accumulated (~bf16-eps close, and more accurate). Z is
// still materialized (byte-exact) because the combine's router-weight gradient d_weights = <d_out, z> needs it.
// pad rows (dest_token < 0) do not scatter.

#ifndef GEMM_M
constexpr int GEMM_M = 8192;
#endif
#ifndef GEMM_N
constexpr int GEMM_N = 8192;
#endif
#ifndef GEMM_K
constexpr int GEMM_K = 8192;
#endif
#ifndef GEMM_E
constexpr int GEMM_E = 8;
#endif
#ifndef GEMM_NREAL
constexpr int GEMM_NREAL = GEMM_N;   // real (unpadded) output width; cols >= NREAL are padding (not written)
#endif
#ifndef NO_SCATTER
#define NO_SCATTER 0                 // debug: skip the atomic scatter (isolate the z-write from the scatter)
#endif
#ifndef NO_ZWRITE
#define NO_ZWRITE 0                  // debug: skip the Z store (isolate the scatter)
#endif

// Kernel
constexpr int NUM_WARPS  = 8;
constexpr int WARPS_ROW  = 2;
constexpr int WARPS_COL  = 4;
constexpr int BLOCK_ROW  = 256;
constexpr int BLOCK_COL  = 256;
constexpr int BLOCK_K    = 128;
constexpr int HALF_ROW   = BLOCK_ROW / 2;
constexpr int HALF_COL   = BLOCK_COL / 2;
constexpr int REG_M      = BLOCK_ROW / WARPS_ROW / 2;
constexpr int REG_N      = BLOCK_COL / WARPS_COL / 2;

using G = kittens::group<NUM_WARPS>;

// float(rne_bf16(s)): round-to-nearest-even to bf16 then back to float. Matches tinygrad's cast_float_to_bf16
// (the '+ bias' / '* weight' both produce bf16 in the reference). inf/nan preserved via the exponent-bits check.
__device__ __forceinline__ float rne_bf16_f(float s) {
    unsigned int x = __float_as_uint(s);
    unsigned int neg = 0u - x;                                  // tinygrad: (-x & 0x7f800000).ne(0)
    unsigned int r = ((neg & 0x7f800000u) != 0u) ? (x + ((x >> 16) & 1u) + 0x7fffu)
                                                 : (((x & 0xffffu) != 0u) ? (x | 0x10000u) : x);
    return __uint_as_float(r & 0xffff0000u);
}
__device__ __forceinline__ float sw_bf16_add_roundtrip(float a, float b) { return rne_bf16_f(__fadd_rn(a, b)); }

__global__ __launch_bounds__(512, 2) void grouped_mxfp8_gemm_down_combine_kernel(
    float *Out_ptr, bf16 *Z_ptr,
    const bf16 *__restrict__ DownBias_ptr,
    const int *__restrict__ DestToken_ptr,
    const bf16 *__restrict__ RowWeight_ptr,
    const int *__restrict__ dest_row_unused,
    const float *__restrict__ weights_unused,
    fp8e4m3 *A_ptr, fp8e4m3 *B_ptr, fp8e8m0 *scale_A_ptr, fp8e8m0 *scale_B_ptr,
    const uint8_t *__restrict__ a_e8_unused,
    const uint8_t *__restrict__ b_e8_unused,
    const int *__restrict__ expert_off) {
    constexpr int M = GEMM_M, N = GEMM_N, K = GEMM_K, E = GEMM_E, NREAL = GEMM_NREAL;

    kittens::gl<fp8e4m3, 1, 1, M, K>     A{A_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<fp8e4m3, 1, 1, E * N, K> B{B_ptr, nullptr, nullptr, nullptr, nullptr};  // all experts stacked on rows

    constexpr int k_iters      = K / BLOCK_K;
    constexpr int NUM_THREADS  = NUM_WARPS * WARP_THREADS;

    kittens::gl<fp8e8m0, k_iters * (M / BLOCK_ROW), 1, 16, 64>     scale_A_gl{scale_A_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<fp8e8m0, E * k_iters * (N / BLOCK_COL), 1, 16, 64> scale_B_gl{scale_B_ptr, nullptr, nullptr, nullptr, nullptr};

    using ST_A     = st_fp8e4m3<HALF_ROW, BLOCK_K, st_16x128_s>;
    using ST_B     = st_fp8e4m3<HALF_COL, BLOCK_K, st_16x128_s>;
    using ST_Scale = st<fp8e8m0, 16, 64, st_16x64_s>;
    using RT_A     = rt_fp8e4m3<REG_M, BLOCK_K>;
    using RT_B     = rt_fp8e4m3<REG_N, BLOCK_K>;
    using RT_C     = rt_fl<REG_M, REG_N, col_l, rt_16x16_s>;

    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];
    __shared__ ST_Scale scale_A_smem[2], scale_B_smem[2];

    RT_A a;
    RT_B b0, b1;
    RT_C cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

    constexpr int tiles_M  = M / BLOCK_ROW;
    constexpr int tiles_N  = N / BLOCK_COL;
    const int NUM_XCDS     = 8;
    const int WGM          = 8;
    int wgid = chiplet_transform_chunked(blockIdx.x, gridDim.x, NUM_XCDS, WGM * WGM);
    int num_wgid_in_group = WGM * tiles_N;
    int group_id     = wgid / num_wgid_in_group;
    int first_pid_m  = group_id * WGM;
    int group_size_m = min(tiles_M - first_pid_m, WGM);
    int block_row    = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int block_col    = (wgid % num_wgid_in_group) / group_size_m;
    int block_m      = block_row * BLOCK_ROW;
    int block_n      = block_col * BLOCK_COL;

    int e = 0;
    #pragma unroll
    for (int i = 1; i < E; i++) e += (expert_off[i] <= block_row * BLOCK_ROW);
    e = __builtin_amdgcn_readfirstlane(e);
    const int bcol_base = e * (N / HALF_COL);          // expert base in B row-tile (128-row) units
    const int sb_base   = e * (k_iters * tiles_N);     // expert base into scale_B batches

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
    i32x4 a_srd = make_srsrc(a_base, (uint32_t)((uint64_t)M * a_row_stride), a_row_stride);
    i32x4 b_srd = make_srsrc(b_base, (uint32_t)((uint64_t)E * N * b_row_stride), b_row_stride);

    const int wid = warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(T)) * kittens::WARP_THREADS;
    uint32_t a_lds_00 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t a_lds_01 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][1].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t a_lds_10 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t a_lds_11 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][1].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_00 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_01 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][1].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_10 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_11 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][1].data[0]) + wid * elem_per_warp * sizeof(T)));

    int a_row_h0 = warp_m * REG_M;
    int a_row_h1 = HALF_ROW + warp_m * REG_M;
    int b_row_h0 = warp_n * REG_N;
    int b_row_h1 = HALF_COL + warp_n * REG_N;


    uint32_t a_lds[2][2] = {{a_lds_00, a_lds_01}, {a_lds_10, a_lds_11}};
    uint32_t b_lds[2][2] = {{b_lds_00, b_lds_01}, {b_lds_10, b_lds_11}};

#if DOUBLE_BUFFER
    #define LOAD_ITER(P, KK)                                                                                               \
        G::load(As[P][0], A, {0, 0, block_row * 2,     (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[P][0]));   \
        G::load(As[P][1], A, {0, 0, block_row * 2 + 1, (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[P][1]));   \
        G::load(Bs[P][0], B, {0, 0, bcol_base + block_col * 2,     (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[P][0])); \
        G::load(Bs[P][1], B, {0, 0, bcol_base + block_col * 2 + 1, (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[P][1])); \
        G::load(scale_A_smem[P], scale_A_gl, {(KK) * tiles_M + block_row, 0, 0, 0});                                       \
        G::load(scale_B_smem[P], scale_B_gl, {sb_base + (KK) * tiles_N + block_col, 0, 0, 0});
    RT_A a0, a1;
    LOAD_ITER(0, 0);
    asm volatile("s_waitcnt vmcnt(0)");
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    #pragma unroll 1
    for (int kk = 0; kk < k_iters; kk++) {
        const int cur = kk & 1;
        const int nxt = (kk + 1) & 1;
        if (kk + 1 < k_iters) { LOAD_ITER(nxt, kk + 1); }

        fp8e8m0_4 sa_h0 = pack_scales(scale_A_smem[cur].data, a_row_h0);
        fp8e8m0_4 sa_h1 = pack_scales(scale_A_smem[cur].data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem[cur].data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem[cur].data, b_row_h1);

        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][0], {warp_n, 0}); load(b0, bs0);
        auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][1], {warp_n, 0}); load(b1, bs1);
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur][0], {warp_m, 0}); load(a0, as0);
        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur][1], {warp_m, 0}); load(a1, as1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cA, a0, b0, cA, &sa_h0, &sb_h0);
        mma_ABt_scaled(cB, a0, b1, cB, &sa_h0, &sb_h1);
        mma_ABt_scaled(cC, a1, b0, cC, &sa_h1, &sb_h0);
        mma_ABt_scaled(cD, a1, b1, cD, &sa_h1, &sb_h1);

        if (kk + 1 < k_iters) { asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier(); }
    }
    #undef LOAD_ITER
#else
    #pragma unroll 1
    for (int kk = 0; kk < k_iters; kk++) {
        G::load(As[0][0], A, {0, 0, block_row * 2,     kk}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[0][0]));
        G::load(As[0][1], A, {0, 0, block_row * 2 + 1, kk}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[0][1]));
        G::load(Bs[0][0], B, {0, 0, bcol_base + block_col * 2,     kk}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[0][0]));
        G::load(Bs[0][1], B, {0, 0, bcol_base + block_col * 2 + 1, kk}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[0][1]));
        G::load(scale_A_smem[0], scale_A_gl, {kk * tiles_M + block_row, 0, 0, 0});
        G::load(scale_B_smem[0], scale_B_gl, {sb_base + kk * tiles_N + block_col, 0, 0, 0});
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
#endif

    // ---- Fused bias + router-weighted COMBINE scatter epilogue ------------------------------------------
    // Per register accumulator element (grouped row = row, output col = gcol): fold the per-expert down bias, write
    // the byte-exact z (for the backward's d_weights), then atomic-scatter the router-weighted contribution into the
    // destination token row. Mirrors the swiglu kernel's REG_ACC (row,col) decomposition of the MFMA output tile.
    const int lane     = laneid();          // 0..63
    const int lrow_grp = 4 * (lane / 16);   // 0,4,8,12
    const int lcol     = lane % 16;         // 0..15
    // The DOUBLE_BUFFER K-loop's last iteration issues its MMAs with NO trailing drain (the `s_waitcnt vmcnt(0);
    // s_barrier` is guarded `if (kk+1 < k_iters)`), so outstanding loads / MFMA accumulators may not be settled when
    // this register epilogue reads them directly. The single-buffer loop drains + barriers every iter, so it is safe.
    // Drain here to match (mirrors the loop's own inter-iteration sync).
    asm volatile("s_waitcnt vmcnt(0)");
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    // process one REG_MxREG_N accumulator ACC at block-local col base COL_BASE (0/HALF_COL), row base ROW_BASE (0/HALF_ROW).
    #define DOWN_ACC(ACC, COL_BASE, ROW_BASE)                                                                 \
    {                                                                                                         \
        _Pragma("unroll") for (int i = 0; i < REG_M/16; i++) {                                                \
            _Pragma("unroll") for (int jj = 0; jj < REG_N/16; jj++) {                                         \
                const int gcol = block_n + (COL_BASE) + warp_n * REG_N + jj * 16 + lcol;                      \
                if (gcol < NREAL) {                                                                           \
                    float bias_f = base_types::convertor<float, bf16>::convert(DownBias_ptr[e * NREAL + gcol]); \
                    float guv[4] = {(ACC).tiles[i][jj].data[0].x, (ACC).tiles[i][jj].data[0].y,               \
                                    (ACC).tiles[i][jj].data[1].x, (ACC).tiles[i][jj].data[1].y};              \
                    _Pragma("unroll") for (int r = 0; r < 4; r++) {                                           \
                        const int row = block_m + (ROW_BASE) + warp_m * REG_M + i * 16 + lrow_grp + r;        \
                        /* z = rne_bf16( trunc_bf16(gemm_accum) + bias ), byte-exact vs the reference z */    \
                        float zg = base_types::convertor<float, bf16>::convert(base_types::convertor<bf16, float>::convert(guv[r])); \
                        float z  = sw_bf16_add_roundtrip(zg, bias_f);                                         \
                        if (!NO_ZWRITE) Z_ptr[row * NREAL + gcol] = base_types::convertor<bf16, float>::convert(z); \
                        const int dt = DestToken_ptr[row];                                                    \
                        if (!NO_SCATTER && dt >= 0) {                                                         \
                            float w = base_types::convertor<float, bf16>::convert(RowWeight_ptr[row]);        \
                            float contrib = rne_bf16_f(__fmul_rn(z, w));  /* matches combine's bf16 product */\
                            __hip_atomic_fetch_add(&Out_ptr[dt * NREAL + gcol], contrib,                      \
                                                   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);               \
                        }                                                                                     \
                    }                                                                                         \
                }                                                                                             \
            }                                                                                                 \
        }                                                                                                     \
    }

    DOWN_ACC(cA, 0,        0);
    DOWN_ACC(cB, HALF_COL, 0);
    DOWN_ACC(cC, 0,        HALF_ROW);
    DOWN_ACC(cD, HALF_COL, HALF_ROW);
    #undef DOWN_ACC
}
