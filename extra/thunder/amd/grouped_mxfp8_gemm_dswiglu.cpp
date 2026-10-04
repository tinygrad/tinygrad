#include "kittens.cuh"

using namespace kittens;

// tinygrad's AMD HIPRenderer lowers .exp2()/.log2() to __ocml_exp2_f32/__ocml_log2_f32 (accurate OCML) and
// compiles WITHOUT -ffast-math (strict IEEE). This kernel is built without -ffast-math too and calls OCML +
// correctly-rounded _rn intrinsics so the recompute+dSwiGLU+quantize epilogue is byte-identical to the
// reference (grouped gemm recompute -> bf16 add bias -> _custom_swiglu_bwd_fp8).
extern "C" __device__ float __ocml_exp2_f32(float);
extern "C" __device__ float __ocml_log2_f32(float);

#ifndef GEMM_M
constexpr int GEMM_M = 8192;
#endif
#ifndef GEMM_N
constexpr int GEMM_N = 8192;   // gate_up width == 2*inter (interleaved gate/linear)
#endif
#ifndef GEMM_K
constexpr int GEMM_K = 8192;
#endif
#ifndef GEMM_E
constexpr int GEMM_E = 8;
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

// gpt-oss clamped swiglu constants (must match extra/llama_kernels/fused_swiglu_quantize_gptoss)
constexpr float SW_LIMIT   = 7.0f;
constexpr float SW_FP8_MAX = 448.0f;
constexpr float SW_ALPHA   = 1.702f;                                     // == -1.7020000219345093f negated
// (ALPHA * LOG2E), folded in fp64 then narrowed to fp32 == tinygrad's rendered 2.4554669857025146f
constexpr float SW_ALPHA_LOG2E = (float)(1.702 * 1.4426950408889634);

using G = kittens::group<NUM_WARPS>;

// float -> bf16 (round-to-nearest-even, tinygrad's cast_float_to_bf16) -> float. Used to bf16-round d_h before
// quantizing, so dh_fp8 == quantize_mxfp8(dh_bf16) byte-for-byte (the reference quantizes the same bf16 value).
__device__ __forceinline__ float f_to_bf16_to_f(float s) {
    unsigned int x = __float_as_uint(s);
    unsigned int neg = 0u - x;                                  // tinygrad: (-x & 0x7f800000).ne(0)
    unsigned int r = ((neg & 0x7f800000u) != 0u) ? (x + ((x >> 16) & 1u) + 0x7fffu)
                                                 : (((x & 0xffffu) != 0u) ? (x | 0x10000u) : x);
    return __uint_as_float(r & 0xffff0000u);
}
// reproduce tinygrad's bf16 '+' for the bias fold: h = rne_bf16(fp32(gate_up) + fp32(bias)), read back to fp32.
__device__ __forceinline__ float sw_bf16_add_roundtrip(float a, float b) { return f_to_bf16_to_f(__fadd_rn(a, b)); }

// Fused grouped mxfp8 FC1 BACKWARD ("Kernel 3"): recomputes gate_up = xg @ w_gate_up^T (M, N) with the base
// grouped-gemm mma loop, folds the per-expert bias (h = gate_up + bias, bf16), applies the gpt-oss clamped-SwiGLU
// DERIVATIVE using the incoming grad (grad_aq, bf16) scaled by the forward y block-scales (Yfwd_e8), and emits the
// full-width d_h three ways: bf16 (for wgrad), fp8e4m3 + e8m0 block-scales along the 2*inter axis (for dgrad).
// gate_up never round-trips through HBM (recompute stays in registers/LDS). mma loop is IDENTICAL to base gemm.
__global__ __launch_bounds__(512, 2) void grouped_mxfp8_gemm_dswiglu_kernel(
    bf16 *Dhbf16_ptr, fp8e4m3 *Dhfp8_ptr, uint8_t *Dhe8_ptr,
    const bf16 *__restrict__ Bias_ptr,
    const bf16 *__restrict__ GradAq_ptr,
    const uint8_t *__restrict__ Yfwd_e8_ptr,
    fp8e4m3 *A_ptr, fp8e4m3 *B_ptr, fp8e8m0 *scale_A_ptr, fp8e8m0 *scale_B_ptr,
    const uint8_t *__restrict__ a_e8_unused,
    const uint8_t *__restrict__ b_e8_unused,
    const int *__restrict__ expert_off) {
    constexpr int M = GEMM_M, N = GEMM_N, K = GEMM_K, E = GEMM_E;
    constexpr int INTER           = N / 2;              // padded inter (grad_aq width; Yfwd_e8 width * 32)
    constexpr int INTER_PER_BLOCK = BLOCK_COL / 2;      // 128 inter columns per 256-wide gate_up block
    constexpr int DH_BLK_PER_BLOCK = BLOCK_COL / 32;    // 8 d_h 32-blocks per 256-wide block

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

    // ---- Recompute -> bias -> dSwiGLU -> mxfp8-quantize epilogue ----------------------------------------
    // Reuse As[2][2] LDS (dead after mma) as a plain row-major bf16[128][256] stripe of the recomputed gate_up
    // (TRUNCATED to bf16 via the same convertor the base gemm uses for C), exactly as the fwd kernel. Then each
    // thread owns one (row, 32-block of the 2*inter axis) == 16 inter indices (32 d_h columns): it re-reads the
    // 32 gate_up cols, folds bias (bf16 add), applies the SwiGLU derivative scaled by ga = grad_aq*qscale, bf16-
    // rounds d_h, takes the thread-local amax over its 32 d_h values, and writes bf16 + fp8 + one e8.
    bf16 *gu = reinterpret_cast<bf16 *>(&As[0][0]);
    const int lane      = laneid();          // 0..63
    const int lrow_grp  = 4 * (lane / 16);   // 0,4,8,12
    const int lcol      = lane % 16;         // 0..15
    const int tid       = threadIdx.x;       // 0..511

    #define STORE_ACC(ACC, COL_BASE)                                                                          \
        _Pragma("unroll") for (int i = 0; i < 4; i++) {                                                       \
            const int r0 = warp_m * 64 + i * 16 + lrow_grp;                                                   \
            _Pragma("unroll") for (int j = 0; j < 2; j++) {                                                   \
                const int col = (COL_BASE) + warp_n * 32 + j * 16 + lcol;                                     \
                gu[(r0 + 0) * BLOCK_COL + col] = base_types::convertor<bf16, float>::convert((ACC).tiles[i][j].data[0].x); \
                gu[(r0 + 1) * BLOCK_COL + col] = base_types::convertor<bf16, float>::convert((ACC).tiles[i][j].data[0].y); \
                gu[(r0 + 2) * BLOCK_COL + col] = base_types::convertor<bf16, float>::convert((ACC).tiles[i][j].data[1].x); \
                gu[(r0 + 3) * BLOCK_COL + col] = base_types::convertor<bf16, float>::convert((ACC).tiles[i][j].data[1].y); \
            }                                                                                                 \
        }

    // dSwiGLU + quantize for the 128-row stripe in `gu`. 128 rows * 8 d_h-blocks = 1024 tasks / 512 threads = 2.
    #define DSWIGLU_STRIPE(STRIPE)                                                                            \
        _Pragma("unroll") for (int pass = 0; pass < 2; pass++) {                                              \
            const int task       = pass * 512 + tid;                                                          \
            const int row        = task / DH_BLK_PER_BLOCK;                                                   \
            const int dblk       = task % DH_BLK_PER_BLOCK;                                                   \
            const int global_row = block_m + (STRIPE) * 128 + row;                                            \
            float wg[16], wl[16];                                                                             \
            float amax = 0.0f;                                                                                \
            _Pragma("unroll") for (int lane16 = 0; lane16 < 16; lane16++) {                                   \
                const int local_inter  = dblk * 16 + lane16;                                                  \
                const int global_inter = block_col * INTER_PER_BLOCK + local_inter;                           \
                /* recompute h = bf16(gate_up) + bias (bf16 add), matching the fwd/reference */               \
                float gate = base_types::convertor<float, bf16>::convert(gu[row * BLOCK_COL + 2 * local_inter]);      \
                float lin  = base_types::convertor<float, bf16>::convert(gu[row * BLOCK_COL + 2 * local_inter + 1]);  \
                const int gu_col = block_n + 2 * local_inter;                                                 \
                gate = sw_bf16_add_roundtrip(gate, base_types::convertor<float, bf16>::convert(Bias_ptr[e * N + gu_col]));     \
                lin  = sw_bf16_add_roundtrip(lin,  base_types::convertor<float, bf16>::convert(Bias_ptr[e * N + gu_col + 1])); \
                /* ga = grad_aq(bf16) * qscale, qscale = exp2(127 - Yfwd_e8[row, inter/32]) */                \
                float e8v = (float)Yfwd_e8_ptr[global_row * (INTER / 32) + global_inter / 32];                \
                /* Replicate _custom_swiglu_bwd_fp8's EXACT op order (verified from its generated HIP source) so the
                   bf16 d_h is byte-exact even in the denormal regime (xg very negative -> sig denormal). tinygrad
                   uses -xg (=nxg) throughout, a flat left-assoc product, and plain '/'; _rn adds prevent any FMA. */ \
                float qscale = __ocml_exp2_f32(__fsub_rn(127.0f, e8v));                                       \
                float ga = base_types::convertor<float, bf16>::convert(GradAq_ptr[global_row * INTER + global_inter]) * qscale; \
                float nxg = -gate; nxg = (nxg < -SW_LIMIT) ? -SW_LIMIT : nxg;   /* alu34 = -min(gate,7) */    \
                float sig = __fdiv_rn(1.0f, __fadd_rn(1.0f, __ocml_exp2_f32(nxg * SW_ALPHA_LOG2E)));  /* alu36 correctly-rounded */ \
                float a37 = (lin < -SW_LIMIT) ? -SW_LIMIT : lin;  float a38 = -a37;                           \
                float a40 = (a38 < -SW_LIMIT) ? -SW_LIMIT : a38;                                              \
                float xl_p1  = __fsub_rn(1.0f, a40);                            /* (1 - alu40) == xl + 1 */    \
                float glu_ok = (gate < SW_LIMIT) ? 1.0f : 0.0f;                 /* alu41 */                    \
                float ok_gt  = (-SW_LIMIT < lin) ? 1.0f : 0.0f;                 /* alu179 = (lin > -7) */      \
                float ok_lt  = (lin < SW_LIMIT) ? 1.0f : 0.0f;                  /* alu178 = (lin < 7) */       \
                float sprime_inner = fmaf(nxg * __fsub_rn(1.0f, sig), -SW_ALPHA, 1.0f);                       \
                float d_gate = ga * sig * sprime_inner * xl_p1 * glu_ok;        /* left-assoc, plain muls */   \
                float d_lin  = -(nxg * sig * ga * ok_gt * ok_lt);              /* left-assoc, plain muls */    \
                float wgg = f_to_bf16_to_f(d_gate);                                                           \
                float wlg = f_to_bf16_to_f(d_lin);                                                            \
                wg[lane16] = wgg; wl[lane16] = wlg;                                                           \
                amax = fmaxf(amax, fmaxf(fabsf(wgg), fabsf(wlg)));                                            \
                Dhbf16_ptr[global_row * N + gu_col]     = base_types::convertor<bf16, float>::convert(wgg);   \
                Dhbf16_ptr[global_row * N + gu_col + 1] = base_types::convertor<bf16, float>::convert(wlg);   \
            }                                                                                                 \
            float e8f = __fadd_rn(floorf(__ocml_log2_f32(fmaxf(amax, 1e-38f))), 127.0f);                      \
            e8f = fminf(fmaxf(e8f, 0.0f), 254.0f);                                                            \
            float qs = __ocml_exp2_f32(__fsub_rn(127.0f, e8f));                                               \
            _Pragma("unroll") for (int lane16 = 0; lane16 < 16; lane16++) {                                   \
                const int local_inter = dblk * 16 + lane16;                                                   \
                const int gu_col = block_n + 2 * local_inter;                                                 \
                float qg = fminf(fmaxf(__fmul_rn(wg[lane16], qs), -SW_FP8_MAX), SW_FP8_MAX);                  \
                float ql = fminf(fmaxf(__fmul_rn(wl[lane16], qs), -SW_FP8_MAX), SW_FP8_MAX);                  \
                Dhfp8_ptr[global_row * N + gu_col]     = base_types::convertor<fp8e4m3, float>::convert(qg);  \
                Dhfp8_ptr[global_row * N + gu_col + 1] = base_types::convertor<fp8e4m3, float>::convert(ql);  \
            }                                                                                                 \
            Dhe8_ptr[global_row * (N / 32) + block_col * DH_BLK_PER_BLOCK + dblk] = (uint8_t)e8f;             \
        }

    __builtin_amdgcn_s_barrier();          // As is free after the last mma iteration's readers finished
    STORE_ACC(cA, 0);
    STORE_ACC(cB, 128);
    __builtin_amdgcn_s_barrier();
    DSWIGLU_STRIPE(0);
    __builtin_amdgcn_s_barrier();
    STORE_ACC(cC, 0);
    STORE_ACC(cD, 128);
    __builtin_amdgcn_s_barrier();
    DSWIGLU_STRIPE(1);

    #undef STORE_ACC
    #undef DSWIGLU_STRIPE
}
