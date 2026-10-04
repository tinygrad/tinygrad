#include "kittens.cuh"
using namespace kittens;

#ifndef WGRAD_M
constexpr int WGRAD_M = 8192;
#endif

#ifndef WGRAD_N
constexpr int WGRAD_N = 8192;
#endif
#ifndef WGRAD_K
constexpr int WGRAD_K = 8192;
#endif
#ifndef WGRAD_E
constexpr int WGRAD_E = 8;
#endif

#ifndef WGRAD_DBUF
#define WGRAD_DBUF 0
#endif
#ifndef FUSED_WGRAD_DBIAS
#define FUSED_WGRAD_DBIAS 0
#endif
#ifndef GPTOSS_WGRAD_TAIL
#define GPTOSS_WGRAD_TAIL 0
#endif
#ifndef GPTOSS_DOWN_WGRAD_TAIL
#define GPTOSS_DOWN_WGRAD_TAIL 0
#endif
#ifndef GPTOSS_DOWN_WGRAD_EDGE_LOAD64
#define GPTOSS_DOWN_WGRAD_EDGE_LOAD64 0
#endif
#ifndef GPTOSS_WGRAD_PAIR_LOOP
#define GPTOSS_WGRAD_PAIR_LOOP 0
#endif
#ifndef GPTOSS_WGRAD_PACK_B32
#define GPTOSS_WGRAD_PACK_B32 0
#endif
#ifndef GPTOSS_WGRAD_INTERIOR_FAST
#define GPTOSS_WGRAD_INTERIOR_FAST 0
#endif
#ifndef GPTOSS_WGRAD_A_FIRST
#define GPTOSS_WGRAD_A_FIRST 0
#endif
#ifndef GPTOSS_WGRAD_B1_OVERLAP
#define GPTOSS_WGRAD_B1_OVERLAP 0
#endif
#ifndef GPTOSS_WGRAD_SCALE_FIRST
#define GPTOSS_WGRAD_SCALE_FIRST 0
#endif
#ifndef GPTOSS_WGRAD_WGM
#define GPTOSS_WGRAD_WGM 1
#endif
#ifndef GPTOSS_WGRAD_XCD_MAP
#define GPTOSS_WGRAD_XCD_MAP 0
#endif
#ifndef GPTOSS_WGRAD_XCDS
#define GPTOSS_WGRAD_XCDS 2
#endif
#ifndef GPTOSS_WGRAD_XCD_CHUNK
#define GPTOSS_WGRAD_XCD_CHUNK 96
#endif
#ifndef GPTOSS_WGRAD_EXPERT_COUNTS
#define GPTOSS_WGRAD_EXPERT_COUNTS 0
#endif
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


#if GPTOSS_WGRAD_TAIL
constexpr int GPTOSS_REAL_N = 5760;
constexpr int GPTOSS_REAL_K = 2880;
static_assert(WGRAD_M == 73728 && WGRAD_N == 5888 && WGRAD_K == 3072 && WGRAD_E == 32);
static_assert(WGRAD_N - GPTOSS_REAL_N == HALF_ROW);
static_assert(GPTOSS_REAL_K % BLOCK_COL == 2 * REG_N);
#elif GPTOSS_DOWN_WGRAD_TAIL
constexpr int GPTOSS_REAL_N = 2880;
constexpr int GPTOSS_REAL_K = 2880;
static_assert(WGRAD_M == 73728 && WGRAD_N == 3072 && WGRAD_K == 3072 && WGRAD_E == 32);
static_assert(GPTOSS_REAL_N % BLOCK_ROW == REG_M);
static_assert(GPTOSS_REAL_K % BLOCK_COL == 2 * REG_N);
#endif

using G = kittens::group<NUM_WARPS>;

#if GPTOSS_DOWN_WGRAD_EDGE_LOAD64
// In the exact down-wgrad edge tiles only the first 64 rows of the surviving 128x128 A/B half-tile are consumed.
// A normal group load emits two 16-byte loads per thread; the first collective load covers rows [0,64), so the
// second can be omitted without changing the LDS addresses observed by the live warp_m/warp_n values.
template<ducks::st::all ST, ducks::gl::all GL>
__device__ __forceinline__ void gptoss_load_edge_rows64(ST &dst, const GL &src, const coord<ST> &idx,
    const uint32_t *__restrict__ swizzled_offsets, i32x4 srd, const void *base_ptr, uint32_t lds_base) {
    using T = typename ST::dtype;
    static_assert(sizeof(T) == 1 && ST::rows == 128 && ST::cols == 128);
    coord<> unit_coord = idx.template unit_coord<2, 3>();
    T *gptr = (T *)&src[unit_coord];
    uint32_t soff = kittens::to_sgpr_u32(static_cast<uint32_t>(
        reinterpret_cast<const char *>(gptr) - reinterpret_cast<const char *>(base_ptr)));
    uint32_t lds_cur = lds_base;
    asm volatile("" : "+s"(lds_cur));
    kittens::llvm_amdgcn_raw_buffer_load_lds(srd, (kittens::as3_uint32_ptr)(uintptr_t)lds_cur,
                                              16, swizzled_offsets[0], soff, 0,
                                              static_cast<int>(kittens::coherency::cache_all));
}
#endif

#if GPTOSS_WGRAD_PACK_B32
// RT_B has two 16-row MFMA tiles, so opsel_b only selects bytes 0/1. The generic four-byte helper reads past the
// scale tile for b_row_h1=224; build only the two live bytes and avoid crossing into the other DBUF plane.
__device__ __forceinline__ fp8e8m0_4 pack_scales_b32(const fp8e8m0 *smem_scales, int row_offset) {
    const int lid = laneid(), r16 = lid & 15, k_sub = lid >> 4;
    const fp8e8m0_4 *s4 = reinterpret_cast<const fp8e8m0_4 *>(smem_scales);
    const fp8e8m0_4 w0 = s4[row_offset + r16];
    const fp8e8m0_4 w1 = s4[row_offset + 16 + r16];
    const fp8e8m0_4 sel = 0x0C0C0000u | (k_sub << 8) | (4u + k_sub);
    return __builtin_amdgcn_perm(w0, w1, sel);
}
#endif


using gptoss_i8 = int __attribute__((ext_vector_type(8)));
using gptoss_f16 = float __attribute__((ext_vector_type(16)));
template<int KHALF, typename ST>
__device__ __forceinline__ gptoss_i8 gptoss_load32(const ST &tile, int row_base) {
    gptoss_i8 value;
    const int row = row_base + (laneid() & 31);
    const int col = KHALF*64 + (laneid() >> 5)*16;
    const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(tile.data));
    const uint32_t tile_offset = (row / 16)*2048;
    const uint32_t addr0 = base + tile_offset + ST::swizzle({row % 16, col});
    const uint32_t addr1 = base + tile_offset + ST::swizzle({row % 16, col+32});
    asm volatile("ds_read_b128 %0, %1" : "=v"(reinterpret_cast<float4 *>(&value)[0]) : "v"(addr0) : "memory");
    asm volatile("ds_read_b128 %0, %1" : "=v"(reinterpret_cast<float4 *>(&value)[1]) : "v"(addr1) : "memory");
    return value;
}
template<int KHALF>
__device__ __forceinline__ uint32_t gptoss_scale32(const fp8e8m0 *ptr, int row_base) {
    const auto *words = reinterpret_cast<const uint32_t *>(ptr);
    return (words[row_base+(laneid() & 31)] >> (8*(KHALF*2+(laneid() >> 5)))) & 255u;
}
template<typename C>
__device__ __forceinline__ void gptoss_mma32(C &dst, gptoss_i8 a, gptoss_i8 b, uint32_t sa, uint32_t sb) {
    auto &acc = *reinterpret_cast<gptoss_f16 *>(dst.data);
    acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, acc, 0, 0, 0, sa, 0, sb);
}

__global__ __launch_bounds__(512, 2) void grouped_mxfp8_wgrad_kernel(bf16 *C_ptr, fp8e4m3 *A_ptr, fp8e4m3 *B_ptr,
    fp8e8m0 *scale_A_ptr, fp8e8m0 *scale_B_ptr,
    const int *__restrict__ expert_off
#if FUSED_WGRAD_DBIAS
    , float *__restrict__ bias_grad, const bf16 *__restrict__ g_bf16
#endif
#if GPTOSS_WGRAD_EXPERT_COUNTS
    , const int *__restrict__ expert_counts
#endif
#if GPTOSS_WGRAD_SHARDS
    , bf16 *C1, bf16 *C2, bf16 *C3, bf16 *C4, bf16 *C5, bf16 *C6, bf16 *C7
#endif
    ) {
    constexpr int M = WGRAD_M, N = WGRAD_N, K = WGRAD_K, E = WGRAD_E;

    kittens::gl<fp8e4m3, 1, 1, N, M>     A{A_ptr, nullptr, nullptr, nullptr, nullptr};  // g^T
    kittens::gl<fp8e4m3, 1, 1, K, M>     B{B_ptr, nullptr, nullptr, nullptr, nullptr};  // x^T
#if !GPTOSS_WGRAD_SHARDS
    kittens::gl<bf16, 1, 1, E * N, K>    C{C_ptr, nullptr, nullptr, nullptr, nullptr};  // grad_w, experts stacked
#endif

    constexpr int m_blocks    = M / BLOCK_K;      // 128-wide blocks along the contraction (token) axis
    constexpr int tiles_N     = N / BLOCK_ROW;
    constexpr int tiles_K     = K / BLOCK_COL;
    constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    kittens::gl<fp8e8m0, m_blocks * tiles_N, 1, 16, 64> scale_A_gl{scale_A_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<fp8e8m0, m_blocks * tiles_K, 1, 16, 64> scale_B_gl{scale_B_ptr, nullptr, nullptr, nullptr, nullptr};

    using ST_A     = st_fp8e4m3<HALF_ROW, BLOCK_K, st_16x128_s>;
    using ST_B     = st_fp8e4m3<HALF_COL, BLOCK_K, st_16x128_s>;
    using ST_Scale = st<fp8e8m0, 16, 64, st_16x64_s>;
    using RT_A     = rt_fp8e4m3<REG_M, BLOCK_K>;
    using RT_B     = rt_fp8e4m3<REG_N, BLOCK_K>;
    using RT_C     = rt_fl<REG_M, REG_N, col_l, rt_32x32_s>;

#if WGRAD_DBUF
    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];
    __shared__ ST_Scale scale_A_smem[2], scale_B_smem[2];
#else
    __shared__ ST_A As[2];
    __shared__ ST_B Bs[2];
    __shared__ ST_Scale scale_A_smem, scale_B_smem;
#endif

    RT_A a;
    RT_B b0, b1;
    RT_C cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

#if GPTOSS_WGRAD_XCD_MAP
    const int wg        = chiplet_transform_chunked(blockIdx.x, E * tiles_N * tiles_K,
                                                     GPTOSS_WGRAD_XCDS, GPTOSS_WGRAD_XCD_CHUNK);
#else
    const int wg        = blockIdx.x;
#endif
    const int e         = wg / (tiles_N * tiles_K);
    const int rem       = wg % (tiles_N * tiles_K);
#if GPTOSS_WGRAD_WGM > 1
    // Group adjacent N tiles while advancing K outside the group. This trades some A reuse for B reuse without
    // interleaving experts, and handles the final partial N group exactly.
    constexpr int WGM = GPTOSS_WGRAD_WGM;
    const int first_row = (rem / (WGM * tiles_K)) * WGM;
    const int group_rows = first_row + WGM <= tiles_N ? WGM : tiles_N - first_row;
    const int group_rem = rem % (WGM * tiles_K);
    const int block_col = group_rem / group_rows;
    const int block_row = first_row + group_rem % group_rows;
#else
    const int block_row = rem / tiles_K;   // over grad_w rows (N)
    const int block_col = rem % tiles_K;   // over grad_w cols (K)
#endif
#if GPTOSS_WGRAD_TAIL
    const bool half_n_tail = block_row == tiles_N - 1;
    const bool quarter_k_tail = block_col == tiles_K - 1;
#elif GPTOSS_DOWN_WGRAD_TAIL
    // Both logical dimensions end 64 elements into their final 256x256 output tile. Treat the N edge as a
    // half-tile load (only As[][0]) and additionally execute only warp_m==0 from that surviving 128-row half.
    const bool quarter_n_tail = block_row == tiles_N - 1;
    const bool half_n_tail = quarter_n_tail;
    const bool quarter_k_tail = block_col == tiles_K - 1;
#else
    constexpr bool quarter_n_tail = false;
    constexpr bool half_n_tail = false;
    constexpr bool quarter_k_tail = false;
#endif
#if GPTOSS_WGRAD_TAIL
    constexpr bool quarter_n_tail = false;
#endif

    const int o0  = __builtin_amdgcn_readfirstlane(expert_off[e]);
    const int kk0 = o0 / BLOCK_K;
#if GPTOSS_WGRAD_EXPERT_COUNTS
    // GPT-OSS routing pads each expert segment to 256 rows. Contract only the 128-row tiles that contain a live
    // routed row; this can remove the wholly-zero second half of the final padded segment without changing the
    // order of any live MFMA. Both dH and the dispatched activation are exactly zero in those padding rows.
    const int nk  = (__builtin_amdgcn_readfirstlane(expert_counts[e]) + BLOCK_K - 1) / BLOCK_K;
#else
    const int nk  = (__builtin_amdgcn_readfirstlane(expert_off[e + 1]) - o0) / BLOCK_K;
#endif

    const int warp_m = warpid() / WARPS_COL;
    const int warp_n = warpid() % WARPS_COL;
    const bool live_n_warp = !quarter_n_tail || warp_m == 0;
    const bool live_k_warp = !quarter_k_tail || warp_n < 2;

    using T = fp8e4m3;
    constexpr int bpt      = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm      = bpt * NUM_THREADS;
    constexpr int copies_A = HALF_ROW * BLOCK_K * sizeof(T) / bpm;
    constexpr int copies_B = HALF_COL * BLOCK_K * sizeof(T) / bpm;
    uint32_t sw_A[copies_A], sw_B[copies_B];
#if WGRAD_DBUF
    G::prefill_swizzled_offsets(As[0][0], A, sw_A);
    G::prefill_swizzled_offsets(Bs[0][0], B, sw_B);
#else
    G::prefill_swizzled_offsets(As[0], A, sw_A);
    G::prefill_swizzled_offsets(Bs[0], B, sw_B);
#endif

    const T *a_base = (const T *)&A[{0, 0, 0, 0}];
    const T *b_base = (const T *)&B[{0, 0, 0, 0}];
    const int a_row_stride = A.template stride<2>() * sizeof(T);
    const int b_row_stride = B.template stride<2>() * sizeof(T);
    // make_srsrc leaves these linear when the byte stride exceeds the 14-bit cache-swizzle field.
    i32x4 a_srd = make_srsrc(a_base, (uint32_t)((uint64_t)N * a_row_stride), a_row_stride);
    i32x4 b_srd = make_srsrc(b_base, (uint32_t)((uint64_t)K * b_row_stride), b_row_stride);
    const int wid = warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(T)) * kittens::WARP_THREADS;
#if WGRAD_DBUF
    uint32_t a_lds_00 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t a_lds_01 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][1].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t a_lds_10 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t a_lds_11 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][1].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_00 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_01 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][1].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_10 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_11 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][1].data[0]) + wid * elem_per_warp * sizeof(T)));
#else
    uint32_t a_lds_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t a_lds_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1].data[0]) + wid * elem_per_warp * sizeof(T)));
#endif

    const int a_row_h0 = warp_m * REG_M;
    const int a_row_h1 = HALF_ROW + warp_m * REG_M;
    const int b_row_h0 = warp_n * REG_N;
    const int b_row_h1 = HALF_COL + warp_n * REG_N;
#if GPTOSS_WGRAD_PACK_B32
    #define PACK_B_SCALES(PTR, OFF) pack_scales_b32((PTR), (OFF))
#else
    #define PACK_B_SCALES(PTR, OFF) pack_scales((PTR), (OFF))
#endif
#if GPTOSS_WGRAD_A_FIRST
    #define ISSUE_WGRAD_OPERANDS(P)                                                                                     \
        if (live_n_warp) { auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[P][0], {warp_m, 0}); load(a, as0); }           \
        if (live_k_warp) { auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[P][0], {warp_n, 0}); load(b0, bs0); }          \
        if (!quarter_k_tail) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[P][1], {warp_n, 0}); load(b1, bs1); }
    #define ISSUE_WGRAD_OPERANDS_FULL(P)                                                                                \
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[P][0], {warp_m, 0}); load(a, as0);                                \
        { auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[P][0], {warp_n, 0}); load(b0, bs0); }                           \
        { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[P][1], {warp_n, 0}); load(b1, bs1); }
#else
    #define ISSUE_WGRAD_OPERANDS(P)                                                                                     \
        if (live_k_warp) { auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[P][0], {warp_n, 0}); load(b0, bs0); }          \
        if (!quarter_k_tail) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[P][1], {warp_n, 0}); load(b1, bs1); }      \
        if (live_n_warp) { auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[P][0], {warp_m, 0}); load(a, as0); }
    #define ISSUE_WGRAD_OPERANDS_FULL(P)                                                                                \
        { auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[P][0], {warp_n, 0}); load(b0, bs0); }                           \
        { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[P][1], {warp_n, 0}); load(b1, bs1); }                           \
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[P][0], {warp_m, 0}); load(a, as0);
#endif

#if WGRAD_DBUF
#if GPTOSS_DOWN_WGRAD_EDGE_LOAD64
    static_assert(GPTOSS_DOWN_WGRAD_TAIL);
    #define LOAD_WGRAD_A_EDGE(P, KK, LDS)                                                                                \
        if (quarter_n_tail) gptoss_load_edge_rows64(As[P][0], A, {0, 0, block_row * 2, (KK)}, sw_A, a_srd, a_base, (LDS)); \
        else G::load(As[P][0], A, {0, 0, block_row * 2, (KK)}, sw_A, a_srd, a_base, (LDS));
    #define LOAD_WGRAD_B_EDGE(P, KK, LDS)                                                                                \
        if (quarter_k_tail) gptoss_load_edge_rows64(Bs[P][0], B, {0, 0, block_col * 2, (KK)}, sw_B, b_srd, b_base, (LDS)); \
        else G::load(Bs[P][0], B, {0, 0, block_col * 2, (KK)}, sw_B, b_srd, b_base, (LDS));
#else
    #define LOAD_WGRAD_A_EDGE(P, KK, LDS) G::load(As[P][0], A, {0, 0, block_row * 2, (KK)}, sw_A, a_srd, a_base, (LDS));
    #define LOAD_WGRAD_B_EDGE(P, KK, LDS) G::load(Bs[P][0], B, {0, 0, block_col * 2, (KK)}, sw_B, b_srd, b_base, (LDS));
#endif
    #define LOAD_WGRAD_STAGE0(KK)                                                                                         \
        LOAD_WGRAD_A_EDGE(0, (KK), __builtin_amdgcn_readfirstlane(a_lds_00))                                              \
        if (!half_n_tail) G::load(As[0][1], A, {0, 0, block_row * 2 + 1, (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds_01)); \
        LOAD_WGRAD_B_EDGE(0, (KK), __builtin_amdgcn_readfirstlane(b_lds_00))                                              \
        if (!quarter_k_tail) G::load(Bs[0][1], B, {0, 0, block_col * 2 + 1, (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds_01)); \
        G::load(scale_A_smem[0], scale_A_gl, {(KK) * tiles_N + block_row, 0, 0, 0});                                      \
        G::load(scale_B_smem[0], scale_B_gl, {(KK) * tiles_K + block_col, 0, 0, 0});
    #define LOAD_WGRAD_STAGE1(KK)                                                                                         \
        LOAD_WGRAD_A_EDGE(1, (KK), __builtin_amdgcn_readfirstlane(a_lds_10))                                              \
        if (!half_n_tail) G::load(As[1][1], A, {0, 0, block_row * 2 + 1, (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds_11)); \
        LOAD_WGRAD_B_EDGE(1, (KK), __builtin_amdgcn_readfirstlane(b_lds_10))                                              \
        if (!quarter_k_tail) G::load(Bs[1][1], B, {0, 0, block_col * 2 + 1, (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds_11)); \
        G::load(scale_A_smem[1], scale_A_gl, {(KK) * tiles_N + block_row, 0, 0, 0});                                      \
        G::load(scale_B_smem[1], scale_B_gl, {(KK) * tiles_K + block_col, 0, 0, 0});

    if (nk > 0) {
        LOAD_WGRAD_STAGE0(kk0);
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }

#if GPTOSS_WGRAD_PAIR_LOOP
    // Literal ping/pong indices remove parity/address selection from the hot loop while retaining load(N+1)-during-
    // MMA(N) overlap. The guards preserve the original behavior for an odd final contraction tile.
    #define COMPUTE_WGRAD_STAGE(P) uint32_t pa00 = reinterpret_cast<const uint32_t *>(scale_A_smem[P].data)[0+warp_m*64+0+(laneid() & 31)]; \
        uint32_t pa01 = reinterpret_cast<const uint32_t *>(scale_A_smem[P].data)[0+warp_m*64+32+(laneid() & 31)]; \
        uint32_t pb0 = reinterpret_cast<const uint32_t *>(scale_B_smem[P].data)[0+warp_n*32+(laneid() & 31)]; \
        uint32_t pa10 = reinterpret_cast<const uint32_t *>(scale_A_smem[P].data)[128+warp_m*64+0+(laneid() & 31)]; \
        uint32_t pa11 = reinterpret_cast<const uint32_t *>(scale_A_smem[P].data)[128+warp_m*64+32+(laneid() & 31)]; \
        uint32_t pb1 = reinterpret_cast<const uint32_t *>(scale_B_smem[P].data)[128+warp_n*32+(laneid() & 31)]; \
        { \
        gptoss_i8 a00 = {}; \
        if (live_n_warp) a00 = gptoss_load32<0>(As[P][0], warp_m*64+0); \
        uint32_t sa00 = ((pa00 >> (8*(0+(laneid() >> 5)))) & 255u); \
        gptoss_i8 a01 = {}; \
        if (live_n_warp) a01 = gptoss_load32<0>(As[P][0], warp_m*64+32); \
        uint32_t sa01 = ((pa01 >> (8*(0+(laneid() >> 5)))) & 255u); \
        gptoss_i8 a10 = {}; \
        if (!half_n_tail) a10 = gptoss_load32<0>(As[P][1], warp_m*64+0); \
        uint32_t sa10 = ((pa10 >> (8*(0+(laneid() >> 5)))) & 255u); \
        gptoss_i8 a11 = {}; \
        if (!half_n_tail) a11 = gptoss_load32<0>(As[P][1], warp_m*64+32); \
        uint32_t sa11 = ((pa11 >> (8*(0+(laneid() >> 5)))) & 255u); \
        gptoss_i8 bb0 = {}; \
        if (live_k_warp) bb0 = gptoss_load32<0>(Bs[P][0], warp_n*32); \
        uint32_t sb0 = ((pb0 >> (8*(0+(laneid() >> 5)))) & 255u); \
        gptoss_i8 bb1 = {}; \
        if (!quarter_k_tail) bb1 = gptoss_load32<0>(Bs[P][1], warp_n*32); \
        uint32_t sb1 = ((pb1 >> (8*(0+(laneid() >> 5)))) & 255u); \
        asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(sa00), "+v"(sa01), "+v"(sa10), "+v"(sa11), "+v"(sb0), "+v"(sb1) : : "memory"); \
        if (live_n_warp && live_k_warp) gptoss_mma32(cA.tiles[0][0], a00, bb0, sa00, sb0); \
        if (live_n_warp && live_k_warp) gptoss_mma32(cA.tiles[1][0], a01, bb0, sa01, sb0); \
        if (live_n_warp && !quarter_k_tail) gptoss_mma32(cB.tiles[0][0], a00, bb1, sa00, sb1); \
        if (live_n_warp && !quarter_k_tail) gptoss_mma32(cB.tiles[1][0], a01, bb1, sa01, sb1); \
        if (!half_n_tail && live_k_warp) gptoss_mma32(cC.tiles[0][0], a10, bb0, sa10, sb0); \
        if (!half_n_tail && live_k_warp) gptoss_mma32(cC.tiles[1][0], a11, bb0, sa11, sb0); \
        if (!half_n_tail && !quarter_k_tail) gptoss_mma32(cD.tiles[0][0], a10, bb1, sa10, sb1); \
        if (!half_n_tail && !quarter_k_tail) gptoss_mma32(cD.tiles[1][0], a11, bb1, sa11, sb1); \
        } \
        { \
        gptoss_i8 a00 = {}; \
        if (live_n_warp) a00 = gptoss_load32<1>(As[P][0], warp_m*64+0); \
        uint32_t sa00 = ((pa00 >> (8*(2+(laneid() >> 5)))) & 255u); \
        gptoss_i8 a01 = {}; \
        if (live_n_warp) a01 = gptoss_load32<1>(As[P][0], warp_m*64+32); \
        uint32_t sa01 = ((pa01 >> (8*(2+(laneid() >> 5)))) & 255u); \
        gptoss_i8 a10 = {}; \
        if (!half_n_tail) a10 = gptoss_load32<1>(As[P][1], warp_m*64+0); \
        uint32_t sa10 = ((pa10 >> (8*(2+(laneid() >> 5)))) & 255u); \
        gptoss_i8 a11 = {}; \
        if (!half_n_tail) a11 = gptoss_load32<1>(As[P][1], warp_m*64+32); \
        uint32_t sa11 = ((pa11 >> (8*(2+(laneid() >> 5)))) & 255u); \
        gptoss_i8 bb0 = {}; \
        if (live_k_warp) bb0 = gptoss_load32<1>(Bs[P][0], warp_n*32); \
        uint32_t sb0 = ((pb0 >> (8*(2+(laneid() >> 5)))) & 255u); \
        gptoss_i8 bb1 = {}; \
        if (!quarter_k_tail) bb1 = gptoss_load32<1>(Bs[P][1], warp_n*32); \
        uint32_t sb1 = ((pb1 >> (8*(2+(laneid() >> 5)))) & 255u); \
        asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(sa00), "+v"(sa01), "+v"(sa10), "+v"(sa11), "+v"(sb0), "+v"(sb1) : : "memory"); \
        if (live_n_warp && live_k_warp) gptoss_mma32(cA.tiles[0][0], a00, bb0, sa00, sb0); \
        if (live_n_warp && live_k_warp) gptoss_mma32(cA.tiles[1][0], a01, bb0, sa01, sb0); \
        if (live_n_warp && !quarter_k_tail) gptoss_mma32(cB.tiles[0][0], a00, bb1, sa00, sb1); \
        if (live_n_warp && !quarter_k_tail) gptoss_mma32(cB.tiles[1][0], a01, bb1, sa01, sb1); \
        if (!half_n_tail && live_k_warp) gptoss_mma32(cC.tiles[0][0], a10, bb0, sa10, sb0); \
        if (!half_n_tail && live_k_warp) gptoss_mma32(cC.tiles[1][0], a11, bb0, sa11, sb0); \
        if (!half_n_tail && !quarter_k_tail) gptoss_mma32(cD.tiles[0][0], a10, bb1, sa10, sb1); \
        if (!half_n_tail && !quarter_k_tail) gptoss_mma32(cD.tiles[1][0], a11, bb1, sa11, sb1); \
        }
#if GPTOSS_WGRAD_INTERIOR_FAST
#if GPTOSS_WGRAD_B1_OVERLAP
    // ISSUE_WGRAD_OPERANDS_FULL ends with four B1 LDS reads. The preceding A/B0 reads are already complete at
    // lgkmcnt(4), so start their first MFMA while B1 finishes instead of stalling the wave on all eight reads.
    #define COMPUTE_WGRAD_H0()                                                                                          \
        asm volatile("s_waitcnt lgkmcnt(4)");                                                                           \
        mma_ABt_scaled(cA, a, b0, cA, &sa_h0, &sb_h0);                                                                 \
        asm volatile("s_waitcnt lgkmcnt(0)");
#else
    #define COMPUTE_WGRAD_H0()                                                                                          \
        asm volatile("s_waitcnt lgkmcnt(0)");                                                                           \
        mma_ABt_scaled(cA, a, b0, cA, &sa_h0, &sb_h0);
#endif
#if GPTOSS_WGRAD_SCALE_FIRST
    // The scale tiles are tiny, but when issued after all 64 KiB of A/B traffic their latency reaches the stage
    // barrier. Put both first so those fetches overlap the four large raw-buffer-to-LDS transfers.
    #define LOAD_WGRAD_STAGE0_FULL(KK)                                                                                   \
        G::load(scale_A_smem[0], scale_A_gl, {(KK) * tiles_N + block_row, 0, 0, 0});                                     \
        G::load(scale_B_smem[0], scale_B_gl, {(KK) * tiles_K + block_col, 0, 0, 0});                                     \
        G::load(As[0][0], A, {0, 0, block_row * 2,     (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds_00)); \
        G::load(As[0][1], A, {0, 0, block_row * 2 + 1, (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds_01)); \
        G::load(Bs[0][0], B, {0, 0, block_col * 2,     (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds_00)); \
        G::load(Bs[0][1], B, {0, 0, block_col * 2 + 1, (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds_01));
    #define LOAD_WGRAD_STAGE1_FULL(KK)                                                                                   \
        G::load(scale_A_smem[1], scale_A_gl, {(KK) * tiles_N + block_row, 0, 0, 0});                                     \
        G::load(scale_B_smem[1], scale_B_gl, {(KK) * tiles_K + block_col, 0, 0, 0});                                     \
        G::load(As[1][0], A, {0, 0, block_row * 2,     (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds_10)); \
        G::load(As[1][1], A, {0, 0, block_row * 2 + 1, (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds_11)); \
        G::load(Bs[1][0], B, {0, 0, block_col * 2,     (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds_10)); \
        G::load(Bs[1][1], B, {0, 0, block_col * 2 + 1, (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds_11));
#else
    #define LOAD_WGRAD_STAGE0_FULL(KK)                                                                                   \
        G::load(As[0][0], A, {0, 0, block_row * 2,     (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds_00)); \
        G::load(As[0][1], A, {0, 0, block_row * 2 + 1, (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds_01)); \
        G::load(Bs[0][0], B, {0, 0, block_col * 2,     (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds_00)); \
        G::load(Bs[0][1], B, {0, 0, block_col * 2 + 1, (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds_01)); \
        G::load(scale_A_smem[0], scale_A_gl, {(KK) * tiles_N + block_row, 0, 0, 0});                                     \
        G::load(scale_B_smem[0], scale_B_gl, {(KK) * tiles_K + block_col, 0, 0, 0});
    #define LOAD_WGRAD_STAGE1_FULL(KK)                                                                                   \
        G::load(As[1][0], A, {0, 0, block_row * 2,     (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds_10)); \
        G::load(As[1][1], A, {0, 0, block_row * 2 + 1, (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds_11)); \
        G::load(Bs[1][0], B, {0, 0, block_col * 2,     (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds_10)); \
        G::load(Bs[1][1], B, {0, 0, block_col * 2 + 1, (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds_11)); \
        G::load(scale_A_smem[1], scale_A_gl, {(KK) * tiles_N + block_row, 0, 0, 0});                                     \
        G::load(scale_B_smem[1], scale_B_gl, {(KK) * tiles_K + block_col, 0, 0, 0});
#endif
    #define COMPUTE_WGRAD_STAGE_FULL(P) uint32_t pa00 = reinterpret_cast<const uint32_t *>(scale_A_smem[P].data)[0+warp_m*64+0+(laneid() & 31)]; \
        uint32_t pa01 = reinterpret_cast<const uint32_t *>(scale_A_smem[P].data)[0+warp_m*64+32+(laneid() & 31)]; \
        uint32_t pb0 = reinterpret_cast<const uint32_t *>(scale_B_smem[P].data)[0+warp_n*32+(laneid() & 31)]; \
        uint32_t pa10 = reinterpret_cast<const uint32_t *>(scale_A_smem[P].data)[128+warp_m*64+0+(laneid() & 31)]; \
        uint32_t pa11 = reinterpret_cast<const uint32_t *>(scale_A_smem[P].data)[128+warp_m*64+32+(laneid() & 31)]; \
        uint32_t pb1 = reinterpret_cast<const uint32_t *>(scale_B_smem[P].data)[128+warp_n*32+(laneid() & 31)]; \
        { \
        gptoss_i8 a00 = {}; \
        a00 = gptoss_load32<0>(As[P][0], warp_m*64+0); \
        uint32_t sa00 = ((pa00 >> (8*(0+(laneid() >> 5)))) & 255u); \
        gptoss_i8 a01 = {}; \
        a01 = gptoss_load32<0>(As[P][0], warp_m*64+32); \
        uint32_t sa01 = ((pa01 >> (8*(0+(laneid() >> 5)))) & 255u); \
        gptoss_i8 a10 = {}; \
        a10 = gptoss_load32<0>(As[P][1], warp_m*64+0); \
        uint32_t sa10 = ((pa10 >> (8*(0+(laneid() >> 5)))) & 255u); \
        gptoss_i8 a11 = {}; \
        a11 = gptoss_load32<0>(As[P][1], warp_m*64+32); \
        uint32_t sa11 = ((pa11 >> (8*(0+(laneid() >> 5)))) & 255u); \
        gptoss_i8 bb0 = {}; \
        bb0 = gptoss_load32<0>(Bs[P][0], warp_n*32); \
        uint32_t sb0 = ((pb0 >> (8*(0+(laneid() >> 5)))) & 255u); \
        gptoss_i8 bb1 = {}; \
        bb1 = gptoss_load32<0>(Bs[P][1], warp_n*32); \
        uint32_t sb1 = ((pb1 >> (8*(0+(laneid() >> 5)))) & 255u); \
        asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(sa00), "+v"(sa01), "+v"(sa10), "+v"(sa11), "+v"(sb0), "+v"(sb1) : : "memory"); \
        gptoss_mma32(cA.tiles[0][0], a00, bb0, sa00, sb0); \
        gptoss_mma32(cA.tiles[1][0], a01, bb0, sa01, sb0); \
        gptoss_mma32(cB.tiles[0][0], a00, bb1, sa00, sb1); \
        gptoss_mma32(cB.tiles[1][0], a01, bb1, sa01, sb1); \
        gptoss_mma32(cC.tiles[0][0], a10, bb0, sa10, sb0); \
        gptoss_mma32(cC.tiles[1][0], a11, bb0, sa11, sb0); \
        gptoss_mma32(cD.tiles[0][0], a10, bb1, sa10, sb1); \
        gptoss_mma32(cD.tiles[1][0], a11, bb1, sa11, sb1); \
        } \
        { \
        gptoss_i8 a00 = {}; \
        a00 = gptoss_load32<1>(As[P][0], warp_m*64+0); \
        uint32_t sa00 = ((pa00 >> (8*(2+(laneid() >> 5)))) & 255u); \
        gptoss_i8 a01 = {}; \
        a01 = gptoss_load32<1>(As[P][0], warp_m*64+32); \
        uint32_t sa01 = ((pa01 >> (8*(2+(laneid() >> 5)))) & 255u); \
        gptoss_i8 a10 = {}; \
        a10 = gptoss_load32<1>(As[P][1], warp_m*64+0); \
        uint32_t sa10 = ((pa10 >> (8*(2+(laneid() >> 5)))) & 255u); \
        gptoss_i8 a11 = {}; \
        a11 = gptoss_load32<1>(As[P][1], warp_m*64+32); \
        uint32_t sa11 = ((pa11 >> (8*(2+(laneid() >> 5)))) & 255u); \
        gptoss_i8 bb0 = {}; \
        bb0 = gptoss_load32<1>(Bs[P][0], warp_n*32); \
        uint32_t sb0 = ((pb0 >> (8*(2+(laneid() >> 5)))) & 255u); \
        gptoss_i8 bb1 = {}; \
        bb1 = gptoss_load32<1>(Bs[P][1], warp_n*32); \
        uint32_t sb1 = ((pb1 >> (8*(2+(laneid() >> 5)))) & 255u); \
        asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(sa00), "+v"(sa01), "+v"(sa10), "+v"(sa11), "+v"(sb0), "+v"(sb1) : : "memory"); \
        gptoss_mma32(cA.tiles[0][0], a00, bb0, sa00, sb0); \
        gptoss_mma32(cA.tiles[1][0], a01, bb0, sa01, sb0); \
        gptoss_mma32(cB.tiles[0][0], a00, bb1, sa00, sb1); \
        gptoss_mma32(cB.tiles[1][0], a01, bb1, sa01, sb1); \
        gptoss_mma32(cC.tiles[0][0], a10, bb0, sa10, sb0); \
        gptoss_mma32(cC.tiles[1][0], a11, bb0, sa11, sb0); \
        gptoss_mma32(cD.tiles[0][0], a10, bb1, sa10, sb1); \
        gptoss_mma32(cD.tiles[1][0], a11, bb1, sa11, sb1); \
        }

    // One uniform dispatch branch; 22*11 of 23*12 GPT-OSS output tiles take this predicate-free body.
    if (!half_n_tail && !quarter_k_tail) {
        #pragma unroll 1
        for (int t = 0; t < nk; t += 2) {
            if (t + 1 < nk) { LOAD_WGRAD_STAGE1_FULL(kk0 + t + 1); }
            { COMPUTE_WGRAD_STAGE_FULL(0); }
            if (t + 1 < nk) {
                asm volatile("s_waitcnt vmcnt(0)");
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_barrier();
                if (t + 2 < nk) { LOAD_WGRAD_STAGE0_FULL(kk0 + t + 2); }
                { COMPUTE_WGRAD_STAGE_FULL(1); }
                if (t + 2 < nk) {
                    asm volatile("s_waitcnt vmcnt(0)");
                    asm volatile("s_waitcnt lgkmcnt(0)");
                    __builtin_amdgcn_s_barrier();
                }
            }
        }
    } else {
#endif
    #pragma unroll 1
    for (int t = 0; t < nk; t += 2) {
        if (t + 1 < nk) { LOAD_WGRAD_STAGE1(kk0 + t + 1); }
        { COMPUTE_WGRAD_STAGE(0); }
        if (t + 1 < nk) {
            asm volatile("s_waitcnt vmcnt(0)");
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            if (t + 2 < nk) { LOAD_WGRAD_STAGE0(kk0 + t + 2); }
            { COMPUTE_WGRAD_STAGE(1); }
            if (t + 2 < nk) {
                asm volatile("s_waitcnt vmcnt(0)");
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_barrier();
            }
        }
    }
#if GPTOSS_WGRAD_INTERIOR_FAST
    }
    #undef COMPUTE_WGRAD_H0
    #undef COMPUTE_WGRAD_STAGE_FULL
    #undef LOAD_WGRAD_STAGE0_FULL
    #undef LOAD_WGRAD_STAGE1_FULL
#endif
    #undef COMPUTE_WGRAD_STAGE
    #undef ISSUE_WGRAD_OPERANDS
    #undef ISSUE_WGRAD_OPERANDS_FULL
#else
    #pragma unroll 1
    for (int t = 0; t < nk; t++) {
        const int cur = t & 1;
        const int nxt = (t + 1) & 1;
        if (t + 1 < nk) {
            if (nxt) { LOAD_WGRAD_STAGE1(kk0 + t + 1); }
            else { LOAD_WGRAD_STAGE0(kk0 + t + 1); }
        }

        fp8e8m0_4 sa_h0 = pack_scales(scale_A_smem[cur].data, a_row_h0);
        fp8e8m0_4 sa_h1 = pack_scales(scale_A_smem[cur].data, a_row_h1);
        fp8e8m0_4 sb_h0 = PACK_B_SCALES(scale_B_smem[cur].data, b_row_h0);
        fp8e8m0_4 sb_h1 = PACK_B_SCALES(scale_B_smem[cur].data, b_row_h1);

        if (live_k_warp) { auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][0], {warp_n, 0}); load(b0, bs0); }
        if (!quarter_k_tail) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][1], {warp_n, 0}); load(b1, bs1); }
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur][0], {warp_m, 0}); load(a, as0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        if (live_k_warp) mma_ABt_scaled(cA, a, b0, cA, &sa_h0, &sb_h0);
        if (!quarter_k_tail) mma_ABt_scaled(cB, a, b1, cB, &sa_h0, &sb_h1);
        if (!half_n_tail) {
            auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur][1], {warp_m, 0}); load(a, as1);
            asm volatile("s_waitcnt lgkmcnt(0)");
            if (live_k_warp) mma_ABt_scaled(cC, a, b0, cC, &sa_h1, &sb_h0);
            if (!quarter_k_tail) mma_ABt_scaled(cD, a, b1, cD, &sa_h1, &sb_h1);
        }

        if (t + 1 < nk) {
            asm volatile("s_waitcnt vmcnt(0)");
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
        }
    }
#endif
    #undef LOAD_WGRAD_STAGE0
    #undef LOAD_WGRAD_STAGE1
    #undef LOAD_WGRAD_A_EDGE
    #undef LOAD_WGRAD_B_EDGE
#else
    #pragma unroll 1
    for (int t = 0; t < nk; t++) {
        const int kk = kk0 + t;
        G::load(As[0], A, {0, 0, block_row * 2,     kk}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds_0));
        if (!half_n_tail) G::load(As[1], A, {0, 0, block_row * 2 + 1, kk}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds_1));
        G::load(Bs[0], B, {0, 0, block_col * 2,     kk}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds_0));
        if (!quarter_k_tail) G::load(Bs[1], B, {0, 0, block_col * 2 + 1, kk}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds_1));
        G::load(scale_A_smem, scale_A_gl, {kk * tiles_N + block_row, 0, 0, 0});
        G::load(scale_B_smem, scale_B_gl, {kk * tiles_K + block_col, 0, 0, 0});
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        fp8e8m0_4 sa_h0 = pack_scales(scale_A_smem.data, a_row_h0);
        fp8e8m0_4 sa_h1 = pack_scales(scale_A_smem.data, a_row_h1);
        fp8e8m0_4 sb_h0 = PACK_B_SCALES(scale_B_smem.data, b_row_h0);
        fp8e8m0_4 sb_h1 = PACK_B_SCALES(scale_B_smem.data, b_row_h1);

        if (live_k_warp) { auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[0], {warp_n, 0}); load(b0, bs0); }
        if (!quarter_k_tail) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[1], {warp_n, 0}); load(b1, bs1); }
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[0], {warp_m, 0}); load(a, as0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        if (live_k_warp) mma_ABt_scaled(cA, a, b0, cA, &sa_h0, &sb_h0);
        if (!quarter_k_tail) mma_ABt_scaled(cB, a, b1, cB, &sa_h0, &sb_h1);
        if (!half_n_tail) {
            auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[1], {warp_m, 0}); load(a, as1);
            asm volatile("s_waitcnt lgkmcnt(0)");
            if (live_k_warp) mma_ABt_scaled(cC, a, b0, cC, &sa_h1, &sb_h0);
            if (!quarter_k_tail) mma_ABt_scaled(cD, a, b1, cD, &sa_h1, &sb_h1);
        }
        __builtin_amdgcn_s_barrier();
    }
#endif
    #undef PACK_B_SCALES
#if GPTOSS_WGRAD_SHARDS
    static_assert(E == 32 && !FUSED_WGRAD_DBIAS);
    bf16 *dst = C_ptr;
    switch (e / 4) {
      case 1: dst=C1; break; case 2: dst=C2; break; case 3: dst=C3; break;
      case 4: dst=C4; break; case 5: dst=C5; break; case 6: dst=C6; break; case 7: dst=C7; break;
    }
    kittens::gl<bf16, 1, 1, (E/8) * N, K> C{dst, nullptr, nullptr, nullptr, nullptr};
    const int crow_base = (e % 4) * (N / REG_M);
#else
    const int crow_base = e * (N / REG_M);   // grad_w rows are experts stacked; store coord is in REG_M units
#endif
    store(C, cA, {0, 0, crow_base + block_row * WARPS_ROW * 2 + warp_m,              block_col * WARPS_COL * 2 + warp_n});
    store(C, cB, {0, 0, crow_base + block_row * WARPS_ROW * 2 + warp_m,              block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
    store(C, cC, {0, 0, crow_base + block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m,  block_col * WARPS_COL * 2 + warp_n});
    store(C, cD, {0, 0, crow_base + block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m,  block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
#if FUSED_WGRAD_DBIAS
    if (block_col == 0 && threadIdx.x < BLOCK_ROW) {
        const int n = block_row * BLOCK_ROW + threadIdx.x;
        float sum = 0.0f;
#if GPTOSS_WGRAD_TAIL
        if (n < GPTOSS_REAL_N) {
#endif
        #pragma unroll 1
        for (int m = o0; m < o0 + nk * BLOCK_K; m++)
            sum += base_types::convertor<float, bf16>::convert(g_bf16[(long long)m * N + n]);
#if GPTOSS_WGRAD_TAIL
        }
#endif
        bias_grad[e * N + n] = sum;
    }
#endif
}
