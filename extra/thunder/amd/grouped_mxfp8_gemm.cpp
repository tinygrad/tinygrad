#include "kittens.cuh"
#ifndef MOE_SKIP_EMPTY
#define MOE_SKIP_EMPTY 0
#endif
#ifndef MOE_SKIP_NO_ZERO
#define MOE_SKIP_NO_ZERO 0
#endif
#ifndef GPTOSS_DOWN_DGRAD_SKIP_EMPTY_VEC
#define GPTOSS_DOWN_DGRAD_SKIP_EMPTY_VEC 0
#endif
#ifndef FUSED_OUT_SCALE
#define FUSED_OUT_SCALE 0
#endif
#ifndef FUSED_OUT_SCALE_PACKED
#define FUSED_OUT_SCALE_PACKED 0
#endif
#ifndef DGRAD_STAGE_OUT_SCALE
#define DGRAD_STAGE_OUT_SCALE 0
#endif
#ifndef FUSED_DOWN_BIAS
#define FUSED_DOWN_BIAS 0
#endif
#ifndef GPTOSS_DOWN_FWD_TRUE_GRID
#define GPTOSS_DOWN_FWD_TRUE_GRID 0
#endif
#ifndef GPTOSS_DGRAD_TAIL
#define GPTOSS_DGRAD_TAIL 0
#endif
#ifndef GPTOSS_DGRAD_PACK_B32
#define GPTOSS_DGRAD_PACK_B32 0
#endif
#ifndef GPTOSS_DGRAD_SCALE_BROADCAST_LDS
#define GPTOSS_DGRAD_SCALE_BROADCAST_LDS 0
#endif
#ifndef GPTOSS_DGRAD_SPLIT_LGKM
#define GPTOSS_DGRAD_SPLIT_LGKM 0
#endif
#ifndef GPTOSS_DGRAD_BINARY_EXPERT
#define GPTOSS_DGRAD_BINARY_EXPERT 0
#endif
#ifndef GPTOSS_DGRAD_SCALAR_LDS
#define GPTOSS_DGRAD_SCALAR_LDS 0
#endif
#ifndef GPTOSS_DOWN_FWD_BINARY_EXPERT
#define GPTOSS_DOWN_FWD_BINARY_EXPERT 0
#endif
#ifndef GPTOSS_DOWN_FWD_SCALAR_LDS
#define GPTOSS_DOWN_FWD_SCALAR_LDS 0
#endif
#ifndef GPTOSS_DOWN_DGRAD_SCALAR_LDS
#define GPTOSS_DOWN_DGRAD_SCALAR_LDS 0
#endif
#ifndef GPTOSS_DOWN_LOGICAL_K
#define GPTOSS_DOWN_LOGICAL_K 0
#endif
#ifndef GPTOSS_DOWN_REAL_K
#define GPTOSS_DOWN_REAL_K 2880
#endif
#ifndef GPTOSS_DOWN_DGRAD_OUT_TAIL
#define GPTOSS_DOWN_DGRAD_OUT_TAIL 0
#endif
#ifndef GPTOSS_DOWN_OUT_TAIL
#define GPTOSS_DOWN_OUT_TAIL GPTOSS_DOWN_DGRAD_OUT_TAIL
#endif
#ifndef GPTOSS_DOWN_REAL_N
#define GPTOSS_DOWN_REAL_N 2880
#endif
#ifndef GPTOSS_DOWN_DGRAD_K64_TAIL
#define GPTOSS_DOWN_DGRAD_K64_TAIL 0
#endif


using namespace kittens;

typedef uint32_t uint4v __attribute__((ext_vector_type(4)));

#if GPTOSS_DOWN_DGRAD_K64_TAIL
// The final logical down-DGRAD contraction tile has only 64 live K values. Issue global-to-LDS loads only for
// that half; its unused LDS half is deliberately left untouched because gptoss_load_rt_k64_final never reads it.
template<ducks::st::all ST, ducks::gl::all GL>
__device__ __forceinline__ void gptoss_load_tile_k64_final(ST &dst, const GL &src, const coord<ST> &idx,
    const uint32_t *__restrict__ swizzled_offsets, i32x4 srd, const void *base_ptr, uint32_t lds_base) {
    using T = typename ST::dtype;
    static_assert(sizeof(T) == 1 && ST::cols == 128);
    constexpr int BYTES_PER_THREAD = 16, BYTES_PER_COPY = BYTES_PER_THREAD * 512;
    constexpr int COPIES = ST::rows * ST::cols / BYTES_PER_COPY;
    static_assert(COPIES == 2);
    coord<> unit_coord = idx.template unit_coord<2, 3>();
    T *gptr = (T *)&src[unit_coord];
    uint32_t soff = kittens::to_sgpr_u32(static_cast<uint32_t>(
        reinterpret_cast<const char *>(gptr) - reinterpret_cast<const char *>(base_ptr)));
    uint32_t lds_cur = lds_base;
    asm volatile("" : "+s"(lds_cur));
    #pragma unroll
    for (int i = 0; i < COPIES; i++) {
        if ((swizzled_offsets[i] & 127u) < 64u) {
            kittens::llvm_amdgcn_raw_buffer_load_lds(srd, (kittens::as3_uint32_ptr)(uintptr_t)lds_cur,
                                                     16, swizzled_offsets[i], soff, 0,
                                                     static_cast<int>(kittens::coherency::cache_all));
        }
        lds_cur += BYTES_PER_COPY;
    }
}

// CDNA4 exposes this block-scaled operation only as one 16x16x128 instruction. Read the live lower half from LDS
// and explicitly zero the upper operand registers so the live accumulation order remains byte-identical.
template<ducks::rt::row_layout RT, ducks::st::all ST>
__device__ __forceinline__ void gptoss_load_rt_k64_final(RT &dst, const ST &src) {
    static_assert(RT::cols == 128 && RT::width == 1 && RT::base_tile_stride == 16);
    static_assert(ST::underlying_subtile_rows == 16 && ST::underlying_subtile_cols == 128);
    const int lane = laneid(), row = lane % 16, col = 16 * (lane / 16);
    const uint32_t addr = reinterpret_cast<uintptr_t>(&src.data[0]) + src.swizzle({row, col});
    #pragma unroll
    for (int ii = 0; ii < ST::subtiles_per_col; ii++) {
        const int offset = ii * ST::underlying_subtile_bytes;
        asm volatile("ds_read_b128 %0, %1 offset:%2\n"
                     : "=v"(*reinterpret_cast<float4 *>(&dst.tiles[ii][0].data[0]))
                     : "v"(addr), "i"(offset) : "memory");
        *reinterpret_cast<float4 *>(&dst.tiles[ii][0].data[4]) = float4{0.0f, 0.0f, 0.0f, 0.0f};
    }
}
#endif

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
#ifndef GPTOSS_REAL_N
#define GPTOSS_REAL_N 2880
#endif
#ifndef GPTOSS_REAL_K
#define GPTOSS_REAL_K 5760
#endif
// FUSED_DOWN_BIAS: real (unpadded) output width; the per-expert bias is (E, GEMM_NREAL) and only columns
// < GEMM_NREAL are biased (the padding columns are sliced off downstream).
#ifndef GEMM_NREAL
#define GEMM_NREAL GEMM_N
#endif

// Kernel
constexpr int NUM_WARPS  = 8;
constexpr int WARPS_ROW  = 2;
constexpr int WARPS_COL  = 4;
// tile size overridable for the MFU retile (smaller tile -> fewer accumulator VGPR -> more waves/SIMD). default 256x256.
#ifndef BLOCK_ROW_D
#define BLOCK_ROW_D 256
#endif
#ifndef BLOCK_COL_D
#define BLOCK_COL_D 256
#endif
constexpr int BLOCK_ROW  = BLOCK_ROW_D;
constexpr int BLOCK_COL  = BLOCK_COL_D;
constexpr int BLOCK_K    = 128;
constexpr int HALF_ROW   = BLOCK_ROW / 2;
constexpr int HALF_COL   = BLOCK_COL / 2;
constexpr int REG_M      = BLOCK_ROW / WARPS_ROW / 2;
constexpr int REG_N      = BLOCK_COL / WARPS_COL / 2;

using G = kittens::group<NUM_WARPS>;

#if GPTOSS_DGRAD_PACK_B32
// RT_B has only two 16-row MFMA tiles, so opsel_b consumes scale bytes 0/1. The generic four-byte helper reads
// beyond the 256-row scale tile for b_row_h1=224; build only the two live bytes and avoid crossing LDS regions.
__device__ __forceinline__ fp8e8m0_4 pack_dgrad_b_scales_32(const fp8e8m0 *smem_scales, int row_offset) {
    const int lid = laneid(), r16 = lid & 15, k_sub = lid >> 4;
    const fp8e8m0_4 *s4 = reinterpret_cast<const fp8e8m0_4 *>(smem_scales);
    const fp8e8m0_4 w0 = s4[row_offset + r16];
    const fp8e8m0_4 w1 = s4[row_offset + 16 + r16];
    const fp8e8m0_4 sel = 0x0C0C0000u | (k_sub << 8) | (4u + k_sub);
    return __builtin_amdgcn_perm(w0, w1, sel);
}
#endif

__global__ __launch_bounds__(512, 2) void grouped_mxfp8_gemm_kernel(bf16 *C_ptr, fp8e4m3 *A_ptr, fp8e4m3 *B_ptr, fp8e8m0 *scale_A_ptr, fp8e8m0 *scale_B_ptr,
    const uint8_t *__restrict__ a_e8_unused,
    const uint8_t *__restrict__ b_e8_unused,
    const int *__restrict__ expert_off
#if FUSED_OUT_SCALE
    , const uint8_t *__restrict__ out_scale_e8   // (M, N/32): restore the original activation's block scale
#endif
#if FUSED_DOWN_BIAS
    , const bf16 *__restrict__ down_bias_ptr    // (E, GEMM_NREAL) per-expert per-column bias, added in fp32 pre-store
#endif
    ) {
    constexpr int M = GEMM_M, N = GEMM_N, K = GEMM_K, E = GEMM_E;

    kittens::gl<fp8e4m3, 1, 1, M, K>     A{A_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<fp8e4m3, 1, 1, E * N, K> B{B_ptr, nullptr, nullptr, nullptr, nullptr};  // all experts stacked on rows
    kittens::gl<bf16, 1, 1, M, N>        C{C_ptr, nullptr, nullptr, nullptr, nullptr};

    constexpr int full_k_iters = K / BLOCK_K;
#if GPTOSS_DGRAD_TAIL
    constexpr int k_iters      = GPTOSS_REAL_K / BLOCK_K;
    static_assert(E == 32 && M == 73728 && N == 3072 && K == 5888);
    static_assert(GPTOSS_REAL_N == 2880 && GPTOSS_REAL_K == 5760);
    static_assert(GPTOSS_REAL_N % BLOCK_COL == 64 && GPTOSS_REAL_K % BLOCK_K == 0);
#elif GPTOSS_DOWN_LOGICAL_K
    // GPT-OSS down/FC2's forward input and backward output-gradient contraction are logically 2880 wide but use
    // the physical 3072 ABI.
    // Execute the 23 tiles covering [0,2944); the omitted 24th tile [2944,3072) is identically zero.
    constexpr int k_iters      = (GPTOSS_DOWN_REAL_K + BLOCK_K - 1) / BLOCK_K;
    static_assert(E == 32 && M == 73728 && N == 3072 && K == 3072);
    static_assert(GPTOSS_DOWN_REAL_K == 2880 && k_iters == 23 && full_k_iters == 24);
#else
    constexpr int k_iters      = full_k_iters;
#endif
    constexpr int NUM_THREADS  = NUM_WARPS * WARP_THREADS;

    kittens::gl<fp8e8m0, full_k_iters * (M / BLOCK_ROW), 1, 16, 64>     scale_A_gl{scale_A_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<fp8e8m0, E * full_k_iters * (N / BLOCK_COL), 1, 16, 64> scale_B_gl{scale_B_ptr, nullptr, nullptr, nullptr, nullptr};

    using ST_A     = st_fp8e4m3<HALF_ROW, BLOCK_K, st_16x128_s>;
    using ST_B     = st_fp8e4m3<HALF_COL, BLOCK_K, st_16x128_s>;
    using ST_Scale = st<fp8e8m0, 16, 64, st_16x64_s>;
    using RT_A     = rt_fp8e4m3<REG_M, BLOCK_K>;
    using RT_B     = rt_fp8e4m3<REG_N, BLOCK_K>;
    using RT_C     = rt_fl<REG_M, REG_N, col_l, rt_16x16_s>;

    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];
    __shared__ ST_Scale scale_A_smem[2], scale_B_smem[2];
#if FUSED_OUT_SCALE_PACKED && DGRAD_STAGE_OUT_SCALE == 2
    // Mode 2 intentionally never aliases the matrix-pipeline LDS. This costs 2 KiB/workgroup but isolates the
    // diagnostic scale staging from every outstanding or compiler-reordered As/Bs access.
    __shared__ uint32_t packed_scale_smem_dedicated[512];
#endif

    RT_A a;
    RT_B b0, b1;
    RT_C cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

    constexpr int tiles_M  = M / BLOCK_ROW;
    constexpr int tiles_N  = N / BLOCK_COL;
    const int NUM_XCDS     = 8;
    #ifndef GROUPED_WGM
#define GROUPED_WGM 8
#endif
    const int WGM          = GROUPED_WGM;
#if GPTOSS_DOWN_FWD_TRUE_GRID
    // Exact callback launch count: do not read an unbound hidden grid argument for this qualified forward path.
    static_assert(FUSED_DOWN_BIAS && E == 32 && M == 73728 && N == 3072 && K == 3072 && GEMM_NREAL == 2880);
    static_assert(DOUBLE_BUFFER && GPTOSS_DGRAD_PACK_B32 && WGM == 8);
    int wgid = chiplet_transform_chunked(blockIdx.x, tiles_M * tiles_N, NUM_XCDS, WGM * WGM);
#else
    int wgid = chiplet_transform_chunked(blockIdx.x, gridDim.x, NUM_XCDS, WGM * WGM);
#endif
    int num_wgid_in_group = WGM * tiles_N;
    int group_id     = wgid / num_wgid_in_group;
    int first_pid_m  = group_id * WGM;
    int group_size_m = min(tiles_M - first_pid_m, WGM);
    int block_row    = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int block_col    = (wgid % num_wgid_in_group) / group_size_m;
    int block_m      = block_row * BLOCK_ROW;
    int block_n      = block_col * BLOCK_COL;
#if GPTOSS_DGRAD_TAIL || GPTOSS_DOWN_OUT_TAIL
    const bool quarter_n_tail = block_col == tiles_N - 1;
#else
    constexpr bool quarter_n_tail = false;
#endif
#if GPTOSS_DOWN_OUT_TAIL
    static_assert(E == 32 && M == 73728 && N == 3072 && K == 3072);
    static_assert(GPTOSS_DOWN_REAL_N == 2880 && GPTOSS_DOWN_REAL_N % BLOCK_COL == 64);
#endif
#if MOE_SKIP_EMPTY
    if (block_m >= __builtin_amdgcn_readfirstlane(expert_off[E])) {
#if !MOE_SKIP_NO_ZERO
        const int tid = threadIdx.x;
#if GPTOSS_DOWN_DGRAD_SKIP_EMPTY_VEC
        // Exact down-dInput still exposes the physical 3072-column buffer, so inactive routing rows must be zero.
        // Write eight adjacent bf16 zeros per thread instead of revisiting the tile 128 times with scalar stores.
        static_assert(GPTOSS_DOWN_LOGICAL_K && GPTOSS_DOWN_OUT_TAIL && GPTOSS_DOWN_DGRAD_K64_TAIL && !FUSED_DOWN_BIAS);
        static_assert(E == 32 && M == 73728 && N == 3072 && K == 3072 && BLOCK_COL % 8 == 0);
        constexpr int VECS_PER_ROW = BLOCK_COL / 8;
        #pragma unroll 1
        for (int i = tid; i < BLOCK_ROW * VECS_PER_ROW; i += NUM_THREADS) {
            const int row = i / VECS_PER_ROW, vec = i % VECS_PER_ROW;
            bf16 *dst = C_ptr + (long long)(block_m + row) * N + block_n;
            reinterpret_cast<uint4v *>(dst)[vec] = uint4v{0u, 0u, 0u, 0u};
        }
#else
        #pragma unroll 1
        for (int i = tid; i < BLOCK_ROW * BLOCK_COL; i += NUM_THREADS) {
            const int row = i / BLOCK_COL, col = i % BLOCK_COL;
            reinterpret_cast<uint16_t *>(C_ptr)[(block_m + row) * N + block_n + col] = 0;
        }
#endif
#endif
        return;
    }
#endif

    int e = 0;
#if GPTOSS_DGRAD_BINARY_EXPERT || GPTOSS_DOWN_FWD_BINARY_EXPERT
    static_assert(E == 32, "binary expert lookup is specialized for GPT-OSS's 32 experts");
    // Monotonic padded offsets make this an upper-bound search. All probes compile to uniform scalar loads.
    const int expert_row = block_row * BLOCK_ROW;
    e += (expert_off[e + 16] <= expert_row) ? 16 : 0;
    e += (expert_off[e +  8] <= expert_row) ?  8 : 0;
    e += (expert_off[e +  4] <= expert_row) ?  4 : 0;
    e += (expert_off[e +  2] <= expert_row) ?  2 : 0;
    e += (expert_off[e +  1] <= expert_row) ?  1 : 0;
#else
    #pragma unroll
    for (int i = 1; i < E; i++) e += (expert_off[i] <= block_row * BLOCK_ROW);
#endif
    e = __builtin_amdgcn_readfirstlane(e);
    const int bcol_base = e * (N / HALF_COL);          // expert base in B row-tile (128-row) units
    const int sb_base   = e * (full_k_iters * tiles_N); // expert base into the physical scale_B batches

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
    uint32_t b_lds_00 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_01 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][1].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t a_lds_10 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t a_lds_11 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][1].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_10 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_11 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][1].data[0]) + wid * elem_per_warp * sizeof(T)));

    int a_row_h0 = warp_m * REG_M;
    int a_row_h1 = HALF_ROW + warp_m * REG_M;
    int b_row_h0 = warp_n * REG_N;
    int b_row_h1 = HALF_COL + warp_n * REG_N;
#if GPTOSS_DGRAD_PACK_B32
    #define PACK_DGRAD_B_SCALES(PTR, OFF) pack_dgrad_b_scales_32((PTR), (OFF))
#else
    #define PACK_DGRAD_B_SCALES(PTR, OFF) pack_scales((PTR), (OFF))
#endif


    uint32_t a_lds[2][2] = {{a_lds_00, a_lds_01}, {a_lds_10, a_lds_11}};
    uint32_t b_lds[2][2] = {{b_lds_00, b_lds_01}, {b_lds_10, b_lds_11}};

#if DOUBLE_BUFFER
    // Software-pipelined double-buffered K-loop: prefetch iter kk+1's A/B/scales into the OTHER LDS buffer while the
    // matrix core consumes iter kk, so the exposed HBM load latency (which the single-buffered loop waits on every
    // iter before any MMA) is hidden under the MMAs. One barrier per iter -- it makes the buffer we're about to READ
    // next iter visible; the WAR on the buffer two iters back is covered by that same barrier plus the pre-MMA
    // lgkmcnt(0) that drains this iter's ds_reads. P2 folded in: two RT_A regs (a0,a1) keep BOTH A operand ds_reads
    // in flight so all 4 MMAs issue back-to-back under a single lgkmcnt(0) (kills the mid-loop WAR reload of 'a').
#if GPTOSS_DGRAD_SCALAR_LDS || GPTOSS_DOWN_FWD_SCALAR_LDS || GPTOSS_DOWN_DGRAD_SCALAR_LDS
    // Exact GPT-OSS kernels: select the ping-pong LDS plane from the eight scalar bases directly instead of
    // materializing runtime-indexed address arrays. The FC1 and down variants are independently gated.
#if GPTOSS_DGRAD_SCALAR_LDS
    static_assert(GPTOSS_DGRAD_TAIL && !FUSED_DOWN_BIAS && E == 32 && M == 73728 && N == 3072 && K == 5888);
#else
    static_assert(GPTOSS_DOWN_LOGICAL_K && GPTOSS_DOWN_OUT_TAIL && E == 32 && M == 73728 && N == 3072 && K == 3072);
    static_assert((GPTOSS_DOWN_FWD_SCALAR_LDS && FUSED_DOWN_BIAS) ||
                  (GPTOSS_DOWN_DGRAD_SCALAR_LDS && !FUSED_DOWN_BIAS));
#endif
    #define LOAD_ITER(P, KK)                                                                                               \
        G::load(As[P][0], A, {0, 0, block_row * 2,     (KK)}, sw_A, a_srd, a_base, (P) ? a_lds_10 : a_lds_00);            \
        G::load(As[P][1], A, {0, 0, block_row * 2 + 1, (KK)}, sw_A, a_srd, a_base, (P) ? a_lds_11 : a_lds_01);            \
        G::load(Bs[P][0], B, {0, 0, bcol_base + block_col * 2,     (KK)}, sw_B, b_srd, b_base, (P) ? b_lds_10 : b_lds_00); \
        if (!quarter_n_tail) G::load(Bs[P][1], B, {0, 0, bcol_base + block_col * 2 + 1, (KK)}, sw_B, b_srd, b_base, (P) ? b_lds_11 : b_lds_01); \
        G::load(scale_A_smem[P], scale_A_gl, {(KK) * tiles_M + block_row, 0, 0, 0});                                       \
        G::load(scale_B_smem[P], scale_B_gl, {sb_base + (KK) * tiles_N + block_col, 0, 0, 0});
#else
    #define LOAD_ITER(P, KK)                                                                                               \
        G::load(As[P][0], A, {0, 0, block_row * 2,     (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[P][0]));   \
        G::load(As[P][1], A, {0, 0, block_row * 2 + 1, (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[P][1]));   \
        G::load(Bs[P][0], B, {0, 0, bcol_base + block_col * 2,     (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[P][0])); \
        if (!quarter_n_tail) G::load(Bs[P][1], B, {0, 0, bcol_base + block_col * 2 + 1, (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[P][1])); \
        G::load(scale_A_smem[P], scale_A_gl, {(KK) * tiles_M + block_row, 0, 0, 0});                                       \
        G::load(scale_B_smem[P], scale_B_gl, {sb_base + (KK) * tiles_N + block_col, 0, 0, 0});
#endif
    RT_A a0, a1;
    LOAD_ITER(0, 0);
    asm volatile("s_waitcnt vmcnt(0)");
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
#if GPTOSS_DOWN_DGRAD_K64_TAIL
    static_assert(GPTOSS_DOWN_LOGICAL_K && GPTOSS_DOWN_REAL_K == 2880 && k_iters == 23 && full_k_iters == 24);
    static_assert(!FUSED_DOWN_BIAS && E == 32 && M == 73728 && N == 3072 && K == 3072);
    #pragma unroll 1
    for (int kk = 0; kk < k_iters - 2; kk++) {
        const int cur = kk & 1;
        const int nxt = (kk + 1) & 1;
        LOAD_ITER(nxt, kk + 1);

        if (!quarter_n_tail || warp_n < 2) {
            fp8e8m0_4 sa_h0 = pack_scales(scale_A_smem[cur].data, a_row_h0);
            fp8e8m0_4 sa_h1 = pack_scales(scale_A_smem[cur].data, a_row_h1);
            fp8e8m0_4 sb_h0 = PACK_DGRAD_B_SCALES(scale_B_smem[cur].data, b_row_h0);
            fp8e8m0_4 sb_h1 = PACK_DGRAD_B_SCALES(scale_B_smem[cur].data, b_row_h1);

            auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][0], {warp_n, 0}); load(b0, bs0);
            if (!quarter_n_tail) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][1], {warp_n, 0}); load(b1, bs1); }
            auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur][0], {warp_m, 0}); load(a0, as0);
            auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur][1], {warp_m, 0}); load(a1, as1);
#if GPTOSS_DGRAD_SPLIT_LGKM
            asm volatile("s_waitcnt lgkmcnt(8)");
#else
            asm volatile("s_waitcnt lgkmcnt(0)");
#endif
            mma_ABt_scaled(cA, a0, b0, cA, &sa_h0, &sb_h0);
            if (!quarter_n_tail) mma_ABt_scaled(cB, a0, b1, cB, &sa_h0, &sb_h1);
#if GPTOSS_DGRAD_SPLIT_LGKM
            asm volatile("s_waitcnt lgkmcnt(0)");
#endif
            mma_ABt_scaled(cC, a1, b0, cC, &sa_h1, &sb_h0);
            if (!quarter_n_tail) mma_ABt_scaled(cD, a1, b1, cD, &sa_h1, &sb_h1);
        }

        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }

    // The main loop left the penultimate tile in plane 1. Stage only the live half of the final tile into plane 0
    // while computing the penultimate tile, preserving the original overlap and accumulation order.
    {
        constexpr int kk = k_iters - 2;
        constexpr int cur = kk & 1;
        constexpr int nxt = (kk + 1) & 1;
        gptoss_load_tile_k64_final(As[nxt][0], A, {0, 0, block_row * 2, k_iters - 1}, sw_A, a_srd, a_base,
                                   __builtin_amdgcn_readfirstlane(a_lds[nxt][0]));
        gptoss_load_tile_k64_final(As[nxt][1], A, {0, 0, block_row * 2 + 1, k_iters - 1}, sw_A, a_srd, a_base,
                                   __builtin_amdgcn_readfirstlane(a_lds[nxt][1]));
        gptoss_load_tile_k64_final(Bs[nxt][0], B, {0, 0, bcol_base + block_col * 2, k_iters - 1}, sw_B, b_srd, b_base,
                                   __builtin_amdgcn_readfirstlane(b_lds[nxt][0]));
        if (!quarter_n_tail)
            gptoss_load_tile_k64_final(Bs[nxt][1], B, {0, 0, bcol_base + block_col * 2 + 1, k_iters - 1}, sw_B, b_srd, b_base,
                                       __builtin_amdgcn_readfirstlane(b_lds[nxt][1]));
        G::load(scale_A_smem[nxt], scale_A_gl, {(k_iters - 1) * tiles_M + block_row, 0, 0, 0});
        G::load(scale_B_smem[nxt], scale_B_gl, {sb_base + (k_iters - 1) * tiles_N + block_col, 0, 0, 0});

        if (!quarter_n_tail || warp_n < 2) {
            fp8e8m0_4 sa_h0 = pack_scales(scale_A_smem[cur].data, a_row_h0);
            fp8e8m0_4 sa_h1 = pack_scales(scale_A_smem[cur].data, a_row_h1);
            fp8e8m0_4 sb_h0 = PACK_DGRAD_B_SCALES(scale_B_smem[cur].data, b_row_h0);
            fp8e8m0_4 sb_h1 = PACK_DGRAD_B_SCALES(scale_B_smem[cur].data, b_row_h1);
            auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][0], {warp_n, 0}); load(b0, bs0);
            if (!quarter_n_tail) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][1], {warp_n, 0}); load(b1, bs1); }
            auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur][0], {warp_m, 0}); load(a0, as0);
            auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur][1], {warp_m, 0}); load(a1, as1);
            asm volatile("s_waitcnt lgkmcnt(0)");
            mma_ABt_scaled(cA, a0, b0, cA, &sa_h0, &sb_h0);
            if (!quarter_n_tail) mma_ABt_scaled(cB, a0, b1, cB, &sa_h0, &sb_h1);
            mma_ABt_scaled(cC, a1, b0, cC, &sa_h1, &sb_h0);
            if (!quarter_n_tail) mma_ABt_scaled(cD, a1, b1, cD, &sa_h1, &sb_h1);
        }
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }

    // Use the same x128 MFMA for the final tile, but skip its upper-half LDS reads and feed explicit register zeroes.
    {
        constexpr int kk = k_iters - 1;
        constexpr int cur = kk & 1;
        if (!quarter_n_tail || warp_n < 2) {
            fp8e8m0_4 sa_h0 = pack_scales(scale_A_smem[cur].data, a_row_h0);
            fp8e8m0_4 sa_h1 = pack_scales(scale_A_smem[cur].data, a_row_h1);
            fp8e8m0_4 sb_h0 = PACK_DGRAD_B_SCALES(scale_B_smem[cur].data, b_row_h0);
            fp8e8m0_4 sb_h1 = PACK_DGRAD_B_SCALES(scale_B_smem[cur].data, b_row_h1);
            auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][0], {warp_n, 0});
            gptoss_load_rt_k64_final(b0, bs0);
            if (!quarter_n_tail) {
                auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][1], {warp_n, 0});
                gptoss_load_rt_k64_final(b1, bs1);
            }
            auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur][0], {warp_m, 0});
            auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur][1], {warp_m, 0});
            gptoss_load_rt_k64_final(a0, as0);
            gptoss_load_rt_k64_final(a1, as1);
#if GPTOSS_DGRAD_SPLIT_LGKM
            asm volatile("s_waitcnt lgkmcnt(8)");
#else
            asm volatile("s_waitcnt lgkmcnt(0)");
#endif
            mma_ABt_scaled(cA, a0, b0, cA, &sa_h0, &sb_h0);
            if (!quarter_n_tail) mma_ABt_scaled(cB, a0, b1, cB, &sa_h0, &sb_h1);
#if GPTOSS_DGRAD_SPLIT_LGKM
            asm volatile("s_waitcnt lgkmcnt(0)");
#endif
            mma_ABt_scaled(cC, a1, b0, cC, &sa_h1, &sb_h0);
            if (!quarter_n_tail) mma_ABt_scaled(cD, a1, b1, cD, &sa_h1, &sb_h1);
        }
    }
#else
    #pragma unroll 1
    for (int kk = 0; kk < k_iters; kk++) {
        const int cur = kk & 1;
        const int nxt = (kk + 1) & 1;
        if (kk + 1 < k_iters) { LOAD_ITER(nxt, kk + 1); }

        if (!quarter_n_tail || warp_n < 2) {
            fp8e8m0_4 sa_h0 = pack_scales(scale_A_smem[cur].data, a_row_h0);
            fp8e8m0_4 sa_h1 = pack_scales(scale_A_smem[cur].data, a_row_h1);
            fp8e8m0_4 sb_h0 = PACK_DGRAD_B_SCALES(scale_B_smem[cur].data, b_row_h0);
            fp8e8m0_4 sb_h1 = PACK_DGRAD_B_SCALES(scale_B_smem[cur].data, b_row_h1);

            auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][0], {warp_n, 0}); load(b0, bs0);
            if (!quarter_n_tail) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][1], {warp_n, 0}); load(b1, bs1); }
            auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur][0], {warp_m, 0}); load(a0, as0);
            auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur][1], {warp_m, 0}); load(a1, as1);
#if GPTOSS_DGRAD_SPLIT_LGKM
            // RT_A<64,128> emits eight LDS reads. Drain a0 while leaving a1's eight reads in flight, overlap those
            // reads under a0's MFMAs, then formally drain a1 immediately before its first consumer.
            asm volatile("s_waitcnt lgkmcnt(8)");
#else
            asm volatile("s_waitcnt lgkmcnt(0)");
#endif
            mma_ABt_scaled(cA, a0, b0, cA, &sa_h0, &sb_h0);
            if (!quarter_n_tail) mma_ABt_scaled(cB, a0, b1, cB, &sa_h0, &sb_h1);
#if GPTOSS_DGRAD_SPLIT_LGKM
            asm volatile("s_waitcnt lgkmcnt(0)");
#endif
            mma_ABt_scaled(cC, a1, b0, cC, &sa_h1, &sb_h0);
            if (!quarter_n_tail) mma_ABt_scaled(cD, a1, b1, cD, &sa_h1, &sb_h1);
        }

        if (kk + 1 < k_iters) { asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier(); }
    }
#endif
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
        fp8e8m0_4 sb_h0 = PACK_DGRAD_B_SCALES(scale_B_smem[0].data, b_row_h0);
        fp8e8m0_4 sb_h1 = PACK_DGRAD_B_SCALES(scale_B_smem[0].data, b_row_h1);

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
    #undef PACK_DGRAD_B_SCALES

#if FUSED_OUT_SCALE
    // The old graph materialized bf16(C), then evaluated bf16(bf16(C) * bf16(2^(e8-127))). Reproduce both bf16
    // rounding boundaries in registers: this col-layout store's scalar convertor truncates the GEMM result, while
    // tinygrad's HIP bf16 cast after the multiply is RNE. Explicitly perform both and store the final bf16 bits
    // directly, avoiding an otherwise redundant bf16->float->bf16 round-trip through the generic tile store.
    // Every e8m0 scale is an exact float/bf16 power of two; e8=0 is the sole subnormal encoding.
    {
#if FUSED_OUT_SCALE_PACKED && (DGRAD_STAGE_OUT_SCALE == 1 || DGRAD_STAGE_OUT_SCALE == 2)
        // The packed layout makes all four warp_n values read a different byte from the same uint32 word. Stage
        // the two 256-row word planes once per workgroup, reusing the now-dead A LDS, instead of fetching each
        // word four times from global memory. Each of the 512 threads moves exactly one dword.
#if DGRAD_STAGE_OUT_SCALE == 1
        uint32_t *packed_scale_smem = reinterpret_cast<uint32_t *>(&As[0][0]);
#else
        uint32_t *packed_scale_smem = packed_scale_smem_dedicated;
#endif
        // Some tail warps skip all matrix work, so synchronize before they overwrite the last K tile.
        __builtin_amdgcn_s_barrier();
        const int scale_tid = threadIdx.x;
        const int scale_plane = scale_tid >> 8;
        const int scale_row = scale_tid & 255;
        if (!quarter_n_tail || scale_plane == 0) {
            packed_scale_smem[scale_tid] = reinterpret_cast<const uint32_t *>(out_scale_e8)[
                (long long)((block_n >> 7) + scale_plane) * M + block_m + scale_row];
        }
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
#endif
        static_assert(REG_N == 32 && (HALF_COL % 32) == 0);
        const int lane = laneid();
        const int lane16 = lane & 15;
        const int lrow_grp = 4 * (lane / 16);
        #define E8_TO_FLOAT(E8) __builtin_bit_cast(float, (uint32_t)((E8) ? ((uint32_t)(E8) << 23) : 0x00400000u))
#if FUSED_OUT_SCALE_PACKED
        // mx_pack layout: uint32 scale groups first, rows second. Each word holds four adjacent 32-column e8 bytes.
#if DGRAD_STAGE_OUT_SCALE == 1 || DGRAD_STAGE_OUT_SCALE == 2
        // The four rows consumed by a subgroup are 16-byte aligned in the staged LDS planes.
        #define LOAD_OUT_SCALES(S, ROW, SCOL) {                                                         \
            const uint4v words = *reinterpret_cast<const uint4v *>(&packed_scale_smem[                   \
                (((SCOL) >> 2) & 1) * BLOCK_ROW + ((ROW) - block_m)]);                                  \
            const int shift = 8 * ((SCOL) & 3);                                                         \
            (S) = ((words[0] >> shift) & 0xffu) | ((words[1] >> shift) & 0xffu) << 8 |                  \
                  ((words[2] >> shift) & 0xffu) << 16 | ((words[3] >> shift) & 0xffu) << 24;            \
        }
#else
        // Diagnostic fallback: read the same four-row vector directly from the packed global layout.
        #define LOAD_OUT_SCALES(S, ROW, SCOL) {                                                         \
            const uint4v words = *reinterpret_cast<const uint4v *>(&reinterpret_cast<const uint32_t *>( \
                out_scale_e8)[(long long)((SCOL) >> 2) * M + (ROW)]);                                  \
            const int shift = 8 * ((SCOL) & 3);                                                         \
            (S) = ((words[0] >> shift) & 0xffu) | ((words[1] >> shift) & 0xffu) << 8 |                  \
                  ((words[2] >> shift) & 0xffu) << 16 | ((words[3] >> shift) & 0xffu) << 24;            \
        }
#endif
#else
        #define LOAD_OUT_SCALES(S, ROW, SCOL) {                                                         \
            (S) = (uint32_t)out_scale_e8[(long long)((ROW) + 0) * (N / 32) + (SCOL)] |                  \
                  (uint32_t)out_scale_e8[(long long)((ROW) + 1) * (N / 32) + (SCOL)] << 8 |             \
                  (uint32_t)out_scale_e8[(long long)((ROW) + 2) * (N / 32) + (SCOL)] << 16 |            \
                  (uint32_t)out_scale_e8[(long long)((ROW) + 3) * (N / 32) + (SCOL)] << 24;             \
        }
#endif
        #define STORE_SCALED_PAIR(PAIR, S0, S1, ROW0, GCOL) {                                            \
            bf16 b0 = base_types::convertor<bf16, float>::convert((PAIR).x);                            \
            bf16 b1 = base_types::convertor<bf16, float>::convert((PAIR).y);                            \
            float2 product = {                                                                          \
                __fmul_rn(base_types::convertor<float, bf16>::convert(b0), E8_TO_FLOAT(S0)),             \
                __fmul_rn(base_types::convertor<float, bf16>::convert(b1), E8_TO_FLOAT(S1))};            \
            bf16_2 final_b = base_types::convertor<bf16_2, float2>::convert(product);                    \
            const uint32_t final_bits = *reinterpret_cast<uint32_t *>(&final_b);                         \
            C_ptr[(long long)(ROW0) * N + (GCOL)] = __builtin_bit_cast(bf16, (uint16_t)final_bits);      \
            C_ptr[(long long)((ROW0) + 1) * N + (GCOL)] =                                                \
                __builtin_bit_cast(bf16, (uint16_t)(final_bits >> 16));                                  \
        }
        // A 16-lane subgroup covers 16 adjacent columns, while scales cover 32 columns. Both jj tiles therefore
        // share one scale. Only the subgroup leader loads each row's byte; broadcast it to the other 15 lanes.
        #define APPLY_OUT_SCALE(ACC, ROW_BASE, COL_BASE) {                                               \
            _Pragma("unroll") for (int i = 0; i < REG_M/16; i++) {                                     \
                const int grow = block_m + (ROW_BASE) + warp_m * REG_M + i * 16 + lrow_grp;             \
                const int gscol = (block_n + (COL_BASE) + warp_n * REG_N) >> 5;                          \
                uint32_t scales = 0;                                                                    \
                /* All-lane mode uses LDS's same-address broadcast and removes the leader ds_bpermute. */ \
                if (GPTOSS_DGRAD_SCALE_BROADCAST_LDS || lane16 == 0) {                                  \
                    LOAD_OUT_SCALES(scales, grow, gscol);                                               \
                }                                                                                       \
                if (!GPTOSS_DGRAD_SCALE_BROADCAST_LDS) {                                                \
                    const int leader = lane - lane16;                                                   \
                    scales = __shfl(scales, leader);                                                    \
                }                                                                                       \
                const uint32_t s0 = scales & 0xffu, s1 = (scales >> 8) & 0xffu;                         \
                const uint32_t s2 = (scales >> 16) & 0xffu, s3 = scales >> 24;                          \
                _Pragma("unroll") for (int jj = 0; jj < REG_N/16; jj++) {                              \
                    const int gcol = block_n + (COL_BASE) + warp_n * REG_N + jj * 16 + lane16;          \
                    STORE_SCALED_PAIR((ACC).tiles[i][jj].data[0], s0, s1, grow, gcol);                  \
                    STORE_SCALED_PAIR((ACC).tiles[i][jj].data[1], s2, s3, grow + 2, gcol);              \
                }                                                                                       \
            }                                                                                           \
        }
        if (!quarter_n_tail || warp_n < 2) {
            APPLY_OUT_SCALE(cA, 0, 0);
            if (!quarter_n_tail) APPLY_OUT_SCALE(cB, 0, HALF_COL);
            APPLY_OUT_SCALE(cC, HALF_ROW, 0);
            if (!quarter_n_tail) APPLY_OUT_SCALE(cD, HALF_ROW, HALF_COL);
        }
#if GPTOSS_DGRAD_TAIL
        if (quarter_n_tail) {
            constexpr int PAD = N - GPTOSS_REAL_N;
            const int tid = threadIdx.x;
            #pragma unroll 1
            for (int i = tid; i < BLOCK_ROW * PAD; i += NUM_THREADS) {
                const int row = i / PAD, col = i % PAD;
                reinterpret_cast<uint16_t *>(C_ptr)[(long long)(block_m + row) * N + GPTOSS_REAL_N + col] = 0;
            }
        }
#endif
        #undef APPLY_OUT_SCALE
        #undef LOAD_OUT_SCALES
        #undef STORE_SCALED_PAIR
        #undef E8_TO_FLOAT
    }
#endif

#if FUSED_DOWN_BIAS
    // Fold the per-expert down-gemm bias into the epilogue: add down_bias[e, col] (fp32) to every accumulator element
    // in that column BEFORE the single bf16 store round (fewer roundings than a separate bf16 +bias elementwise kernel).
    // RT_C (col_l) holds 4 elements per (tile, column) that span 4 rows -> one bias load, broadcast to all 4. cA/cB/cC/cD
    // map to (COL_BASE, ROW_BASE) exactly as the swiglu epilogue (the MMA structure is byte-identical). Columns
    // >= GEMM_NREAL are padding (sliced off downstream) -> skip. Only the down-FWD call defines FUSED_DOWN_BIAS; the
    // shared dgrad/fc1 compiles leave this out and are byte-unchanged.
    {
        const int lane = laneid();
        const int lcol = lane % 16;                     // 0..15 column within a 16x16 base tile
#if GPTOSS_DOWN_OUT_TAIL
        // The exact GPT-OSS down shape has two useful facts that the generic bounds check obscures from clang:
        //   * every non-tail tile ends at column 2815, and the guarded tail path ends at column 2879;
        //   * cA/cC (and cB/cD) cover different row halves but the same columns.
        // Load each column bias once, then reuse it across all row fragments and both row-half accumulators. Besides
        // removing the now-proven bounds checks, this cuts the full-tile epilogue from 32 serialized bias loads per
        // wave to four (the 64-column tail from 16 to two) without changing any accumulator's single fp32 add.
        #define ADD_DOWN_BIAS_ROW_PAIR(ACC0, ACC1, COL_BASE)                                                   \
            _Pragma("unroll") for (int jj = 0; jj < REG_N/16; jj++) {                                       \
                const int gcol = block_n + (COL_BASE) + warp_n * REG_N + jj * 16 + lcol;                     \
                float bf = base_types::convertor<float, bf16>::convert(down_bias_ptr[e * GEMM_NREAL + gcol]); \
                _Pragma("unroll") for (int i = 0; i < REG_M/16; i++) {                                      \
                    (ACC0).tiles[i][jj].data[0].x += bf; (ACC0).tiles[i][jj].data[0].y += bf;                 \
                    (ACC0).tiles[i][jj].data[1].x += bf; (ACC0).tiles[i][jj].data[1].y += bf;                 \
                    (ACC1).tiles[i][jj].data[0].x += bf; (ACC1).tiles[i][jj].data[0].y += bf;                 \
                    (ACC1).tiles[i][jj].data[1].x += bf; (ACC1).tiles[i][jj].data[1].y += bf;                 \
                }                                                                                            \
            }
        if (!quarter_n_tail) {
            ADD_DOWN_BIAS_ROW_PAIR(cA, cC, 0);
            ADD_DOWN_BIAS_ROW_PAIR(cB, cD, HALF_COL);
        } else if (warp_n < 2) {
            ADD_DOWN_BIAS_ROW_PAIR(cA, cC, 0);
        }
        #undef ADD_DOWN_BIAS_ROW_PAIR
#else
        #define ADD_DOWN_BIAS(ACC, COL_BASE)                                                                    \
            _Pragma("unroll") for (int i = 0; i < REG_M/16; i++) {                                              \
                _Pragma("unroll") for (int jj = 0; jj < REG_N/16; jj++) {                                       \
                    const int gcol = block_n + (COL_BASE) + warp_n * REG_N + jj * 16 + lcol;                    \
                    if (gcol < GEMM_NREAL) {                                                                    \
                        float bf = base_types::convertor<float, bf16>::convert(down_bias_ptr[e * GEMM_NREAL + gcol]); \
                        (ACC).tiles[i][jj].data[0].x += bf; (ACC).tiles[i][jj].data[0].y += bf;                 \
                        (ACC).tiles[i][jj].data[1].x += bf; (ACC).tiles[i][jj].data[1].y += bf;                 \
                    }                                                                                           \
                }                                                                                               \
            }
        ADD_DOWN_BIAS(cA, 0);
        ADD_DOWN_BIAS(cB, HALF_COL);
        ADD_DOWN_BIAS(cC, 0);
        ADD_DOWN_BIAS(cD, HALF_COL);
        #undef ADD_DOWN_BIAS
#endif
    }
#endif
#if !FUSED_OUT_SCALE
#if GPTOSS_DOWN_OUT_TAIL
    if (quarter_n_tail) {
        // The logical output ends 64 columns into the last 256-column tile. Only warp_n 0/1 own live columns.
        if (warp_n < 2) {
            store(C, cA, {0, 0, block_row * WARPS_ROW * 2 + warp_m, block_col * WARPS_COL * 2 + warp_n});
            store(C, cC, {0, 0, block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m, block_col * WARPS_COL * 2 + warp_n});
        }
        // Preserve the physical 3072-column ABI: explicitly initialize columns 2880..3071 with contiguous
        // 16-byte stores, rather than converting/storing zero accumulator fragments for those columns.
        constexpr int PAD = N - GPTOSS_DOWN_REAL_N;
        constexpr int PAD_VECS_PER_ROW = PAD / 8;
        #pragma unroll 1
        for (int i = threadIdx.x; i < BLOCK_ROW * PAD_VECS_PER_ROW; i += NUM_THREADS) {
            const int row = i / PAD_VECS_PER_ROW;
            const int vec = i - row * PAD_VECS_PER_ROW;
            bf16 *dst = C_ptr + (long long)(block_m + row) * N + GPTOSS_DOWN_REAL_N;
            reinterpret_cast<uint4v *>(dst)[vec] = uint4v{0u, 0u, 0u, 0u};
        }
    } else
#endif
    {
        store(C, cA, {0, 0, block_row * WARPS_ROW * 2 + warp_m, block_col * WARPS_COL * 2 + warp_n});
        store(C, cB, {0, 0, block_row * WARPS_ROW * 2 + warp_m, block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
        store(C, cC, {0, 0, block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m, block_col * WARPS_COL * 2 + warp_n});
        store(C, cD, {0, 0, block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m, block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
    }
#endif
}
