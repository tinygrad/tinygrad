#include "kittens.cuh"

using namespace kittens;

#ifndef GEMM_K
#define GEMM_K 16384
#endif
#ifndef GPTOSS_LMHEAD_WGRAD_PART
#define GPTOSS_LMHEAD_WGRAD_PART 0
#endif
#ifndef GPTOSS_LMHEAD_WGRAD_WGM
#define GPTOSS_LMHEAD_WGRAD_WGM 16
#endif
// GPTOSS_LMHEAD_WGRAD_PART:
//   0: full physical 128256x3072 output (split-path opt-out)
//   1: the first 11 full 256-column tiles
//   2: a dedicated final 256x128 container with only its first 64 columns live
static_assert(GPTOSS_LMHEAD_WGRAD_PART >= 0 && GPTOSS_LMHEAD_WGRAD_PART <= 2);
constexpr int M = 128256, N = 3072, K = GEMM_K, REAL_N = 2880;
static_assert(K == 16384);

constexpr int NUM_WARPS = 8;
#if GPTOSS_LMHEAD_WGRAD_PART == 2
constexpr int WARPS_ROW = 4, WARPS_COL = 2;
constexpr int BLOCK_ROW = 256, BLOCK_COL = 128;
#else
constexpr int WARPS_ROW = 2, WARPS_COL = 4;
constexpr int BLOCK_ROW = 128, BLOCK_COL = 256;
#endif
constexpr int BLOCK_K = 128, HALF_ROW = BLOCK_ROW / 2, HALF_COL = BLOCK_COL / 2;
constexpr int REG_M = BLOCK_ROW / WARPS_ROW / 2, REG_N = BLOCK_COL / WARPS_COL / 2;
constexpr int SCALE_ROWS = 256, NUM_THREADS = NUM_WARPS * WARP_THREADS;
using G = kittens::group<NUM_WARPS>;
#if defined(GPTOSS_LMHEAD_PACK32) && GPTOSS_LMHEAD_PACK32 && GPTOSS_LMHEAD_WGRAD_PART == 1 && GPTOSS_LMHEAD_WGRAD_WGM == 16
// Preserve the generic scale path for the unsplit and live-64-column tail kernels.
__device__ __forceinline__ fp8e8m0_4 pack_dense_scales_32(const fp8e8m0 *smem_scales, int row_offset) {
    const int lid = laneid(), r16 = lid & 15, k_sub = lid >> 4;
    const fp8e8m0_4 *s4 = reinterpret_cast<const fp8e8m0_4 *>(smem_scales);
    const fp8e8m0_4 w0 = s4[row_offset + r16];
    const fp8e8m0_4 w1 = s4[row_offset + 16 + r16];
    return __builtin_amdgcn_perm(w0, w1, 0x0C0C0000u | (k_sub << 8) | (4u + k_sub));
}
static_assert(REG_M == 32 && REG_N == 32);
#define pack_scales pack_dense_scales_32
#endif
typedef uint32_t uint4v __attribute__((ext_vector_type(4)));

extern "C" __global__ __launch_bounds__(512, 2) void mxfp8_gemm_kernel(
    bf16 *C_ptr, fp8e4m3 *A_ptr, fp8e4m3 *B_ptr, fp8e8m0 *scale_A_ptr, fp8e8m0 *scale_B_ptr,
    const uint8_t *__restrict__ a_e8_unused, const uint8_t *__restrict__ b_e8_unused) {
    constexpr int k_iters = K / BLOCK_K;
    constexpr int physical_tiles_n = N / 256;
    kittens::gl<fp8e4m3, 1, 1, M, K> A{A_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<fp8e4m3, 1, 1, N, K> B{B_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<bf16, 1, 1, M, N> C{C_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<fp8e8m0, k_iters * (M / SCALE_ROWS), 1, 16, 64> scale_A_gl{scale_A_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<fp8e8m0, k_iters * physical_tiles_n, 1, 16, 64> scale_B_gl{scale_B_ptr, nullptr, nullptr, nullptr, nullptr};

    using ST_A = st_fp8e4m3<HALF_ROW, BLOCK_K, st_16x128_s>;
    using ST_B = st_fp8e4m3<HALF_COL, BLOCK_K, st_16x128_s>;
    using ST_Scale = st<fp8e8m0, 16, 64, st_16x64_s>;
    using RT_A = rt_fp8e4m3<REG_M, BLOCK_K>;
    using RT_B = rt_fp8e4m3<REG_N, BLOCK_K>;
    using RT_C = rt_fl<REG_M, REG_N, col_l, rt_16x16_s>;
    __shared__ ST_A As[1][2];
    __shared__ ST_B Bs[1][2];
    __shared__ ST_Scale scale_A_smem[1], scale_B_smem[1];
    RT_A a;
    RT_B b0;
    RT_C cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

    constexpr int tiles_M = M / BLOCK_ROW;
#if GPTOSS_LMHEAD_WGRAD_PART == 2
    int block_row = chiplet_transform_chunked(blockIdx.x, gridDim.x, 8, 64);
    constexpr int block_col = 22, scale_block_col = 11;
#else
    constexpr int launch_tiles_n = GPTOSS_LMHEAD_WGRAD_PART == 1 ? 11 : physical_tiles_n;
    constexpr int WGM = GPTOSS_LMHEAD_WGRAD_WGM;
    int wgid = chiplet_transform_chunked(blockIdx.x, gridDim.x, 8, WGM * WGM);
    int num_wgid_in_group = WGM * launch_tiles_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(tiles_M - first_pid_m, WGM);
    int block_row = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int block_col = (wgid % num_wgid_in_group) / group_size_m;
    int scale_block_col = block_col;
#endif
    int block_m = block_row * BLOCK_ROW;
    int block_n = block_col * BLOCK_COL;
    int warp_m = warpid() / WARPS_COL, warp_n = warpid() % WARPS_COL;

    using T = fp8e4m3;
    constexpr int bpt = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * NUM_THREADS;
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
    constexpr int elem_per_warp = (16 / sizeof(T)) * WARP_THREADS;
    uint32_t a_lds_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][0].data[0]) + wid * elem_per_warp));
    uint32_t a_lds_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][1].data[0]) + wid * elem_per_warp));
    uint32_t b_lds_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][0].data[0]) + wid * elem_per_warp));
#if GPTOSS_LMHEAD_WGRAD_PART != 2
    uint32_t b_lds_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][1].data[0]) + wid * elem_per_warp));
#endif
    const int scale_block_row = (block_row * BLOCK_ROW) / SCALE_ROWS;
    const int scale_row = (block_row * BLOCK_ROW) % SCALE_ROWS;
    int a_row_h0 = scale_row + warp_m * REG_M;
    int a_row_h1 = scale_row + HALF_ROW + warp_m * REG_M;
    int b_row_h0 = warp_n * REG_N;
#if GPTOSS_LMHEAD_WGRAD_PART != 2
    int b_row_h1 = HALF_COL + warp_n * REG_N;
#endif

    #pragma unroll 1
    for (int kk = 0; kk < k_iters; kk++) {
        G::load(As[0][0], A, {0, 0, block_row * 2, kk}, sw_A, a_srd, a_base, a_lds_0);
        G::load(As[0][1], A, {0, 0, block_row * 2 + 1, kk}, sw_A, a_srd, a_base, a_lds_1);
        G::load(Bs[0][0], B, {0, 0, block_col * 2, kk}, sw_B, b_srd, b_base, b_lds_0);
#if GPTOSS_LMHEAD_WGRAD_PART != 2
        G::load(Bs[0][1], B, {0, 0, block_col * 2 + 1, kk}, sw_B, b_srd, b_base, b_lds_1);
#endif
        G::load(scale_A_smem[0], scale_A_gl, {kk * (M / SCALE_ROWS) + scale_block_row, 0, 0, 0});
        G::load(scale_B_smem[0], scale_B_gl, {kk * physical_tiles_n + scale_block_col, 0, 0, 0});
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        fp8e8m0_4 sa_h0 = pack_scales(scale_A_smem[0].data, a_row_h0);
        fp8e8m0_4 sa_h1 = pack_scales(scale_A_smem[0].data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem[0].data, b_row_h0);
        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[0][0], {warp_n, 0}); load(b0, bs0);
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[0][0], {warp_m, 0}); load(a, as0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cA, a, b0, cA, &sa_h0, &sb_h0);
#if GPTOSS_LMHEAD_WGRAD_PART != 2
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem[0].data, b_row_h1);
        auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[0][1], {warp_n, 0}); load(b0, bs1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cB, a, b0, cB, &sa_h0, &sb_h1);
#endif
        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[0][1], {warp_m, 0}); load(a, as1);
        load(b0, bs0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cC, a, b0, cC, &sa_h1, &sb_h0);
#if GPTOSS_LMHEAD_WGRAD_PART != 2
        load(b0, bs1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cD, a, b0, cD, &sa_h1, &sb_h1);
#endif
        __builtin_amdgcn_s_barrier();
    }
    store(C, cA, {0, 0, block_row * WARPS_ROW * 2 + warp_m, block_col * WARPS_COL * 2 + warp_n});
#if GPTOSS_LMHEAD_WGRAD_PART != 2
    store(C, cB, {0, 0, block_row * WARPS_ROW * 2 + warp_m, block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
#endif
    store(C, cC, {0, 0, block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m, block_col * WARPS_COL * 2 + warp_n});
#if GPTOSS_LMHEAD_WGRAD_PART != 2
    store(C, cD, {0, 0, block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m, block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
#else
    constexpr int PAD = N - REAL_N, PAD_VECS_PER_ROW = PAD / 8;
    #pragma unroll 1
    for (int i = threadIdx.x; i < BLOCK_ROW * PAD_VECS_PER_ROW; i += NUM_THREADS) {
        const int row = i / PAD_VECS_PER_ROW, vec = i - row * PAD_VECS_PER_ROW;
        bf16 *dst = C_ptr + (long long)(block_m + row) * N + REAL_N;
        reinterpret_cast<uint4v *>(dst)[vec] = uint4v{0u, 0u, 0u, 0u};
    }
#endif
}
