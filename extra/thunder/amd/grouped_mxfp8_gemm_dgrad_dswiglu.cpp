#include "kittens.cuh"
#ifndef MOE_SKIP_EMPTY
#define MOE_SKIP_EMPTY 0
#endif
#ifndef DGRAD_PACKED_IO
#define DGRAD_PACKED_IO 0
#endif


using namespace kittens;

extern "C" __device__ float __ocml_exp2_f32(float);
extern "C" __device__ float __ocml_log2_f32(float);

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
#ifndef H_N
constexpr int H_N = GEMM_N * 2;
#endif
#ifndef REAL_INTER
constexpr int REAL_INTER = GEMM_N;
#endif
#ifndef FUSED_DGRAD_COLW
#define FUSED_DGRAD_COLW 0
#endif
#ifndef NATIVE_MXFP8_CVT
#define NATIVE_MXFP8_CVT 0
#endif
#ifndef DGRAD_NATIVE_ROW
#define DGRAD_NATIVE_ROW 0
#endif
#ifndef DGRAD_TILE_N128
#define DGRAD_TILE_N128 0
#endif
#ifndef DGRAD_REG_EPILOGUE
#define DGRAD_REG_EPILOGUE 0
#endif
#ifndef DGRAD_FAST_EXP
#define DGRAD_FAST_EXP 0
#endif
#if DGRAD_REG_EPILOGUE && (!DGRAD_PACKED_IO || !NATIVE_MXFP8_CVT || !DGRAD_NATIVE_ROW || FUSED_DGRAD_COLW || DGRAD_TILE_N128)
#error "DGRAD_REG_EPILOGUE is the packed/native/row-native/N256/colw-off production specialization"
#endif
#if NATIVE_MXFP8_CVT
typedef short short2v __attribute__((ext_vector_type(2)));
#endif

constexpr int NUM_WARPS = 8;
constexpr int WARPS_ROW = 2;
constexpr int WARPS_COL = 4;
constexpr int BLOCK_ROW = 256;
constexpr int BLOCK_COL = DGRAD_TILE_N128 ? 128 : 256;
constexpr int BLOCK_K = 128;
constexpr int HALF_ROW = BLOCK_ROW / 2;
constexpr int HALF_COL = BLOCK_COL / 2;
constexpr int REG_M = BLOCK_ROW / WARPS_ROW / 2;
constexpr int REG_N = BLOCK_COL / WARPS_COL / 2;
constexpr int SWIGLU_BLOCKS = BLOCK_COL * 2 / 32;
constexpr int COLW_LDS_STRIDE = BLOCK_COL * 2 + (DGRAD_TILE_N128 ? 0 : 1);

constexpr float SW_LIMIT = 7.0f;
constexpr float SW_FP8_MAX = 448.0f;
constexpr float SW_ALPHA = 1.702f;
constexpr float SW_ALPHA_LOG2E = (float)(1.702 * 1.4426950408889634);

using G = kittens::group<NUM_WARPS>;


// Max-reduce independently in each 16-lane DPP row. The first two steps reduce each quad;
// row rotations by 4 and 8 then union all four quads. DPP avoids LDS-backed __shfl waits.
__device__ __forceinline__ float dgrad_amax16(float x) {
    x = fmaxf(x, __shfl_xor(x, 1, 16));
    x = fmaxf(x, __shfl_xor(x, 2, 16));
    x = fmaxf(x, __shfl_xor(x, 4, 16));
    x = fmaxf(x, __shfl_xor(x, 8, 16));
    return x;
}

__device__ __forceinline__ float f_to_bf16_to_f(float s) {
    unsigned int x = __float_as_uint(s);
    unsigned int neg = 0u - x;
    unsigned int r = ((neg & 0x7f800000u) != 0u) ? (x + ((x >> 16) & 1u) + 0x7fffu)
                                                 : (((x & 0xffffu) != 0u) ? (x | 0x10000u) : x);
    return __uint_as_float(r & 0xffff0000u);
}

__device__ __forceinline__ void dswiglu_stripe(
    const bf16 *__restrict__ dy, const bf16 *__restrict__ H_ptr,
    bf16 *__restrict__ Dhbf16_ptr, fp8e4m3 *__restrict__ Dhfp8_ptr, uint8_t *__restrict__ Dhe8_ptr,
    fp8e4m3 *__restrict__ Dhcol_ptr, uint8_t *__restrict__ Dhcol_e8_ptr, bf16 *__restrict__ colw_smem,
    const int block_m, const int block_col, const int stripe) {
    constexpr int H_INTER = H_N / 2;
    const int tid = threadIdx.x;
    #pragma unroll
    for (int pass = 0; pass < SWIGLU_BLOCKS / 4; pass++) {
        const int task = pass * 512 + tid;
        const int row = task / SWIGLU_BLOCKS;
        const int row32 = tid / SWIGLU_BLOCKS;
        const int dblk = task % SWIGLU_BLOCKS;
        const int global_row = block_m + stripe * 128 + row;
        const int inter_base = block_col * BLOCK_COL + dblk * 16;
        if (inter_base < H_INTER) {
            float wg[16], wl[16];
            float amax = 0.0f;
            #pragma unroll
            for (int lane16 = 0; lane16 < 16; lane16++) {
                const int local_inter = dblk * 16 + lane16;
                const int global_inter = inter_base + lane16;
                const int hcol = 2 * global_inter;
                const int local_hcol = 2 * local_inter;
#if DGRAD_PACKED_IO
                const uint32_t hpair = reinterpret_cast<const uint32_t *>(H_ptr)[(long long)global_row * (H_N / 2) + global_inter];
                float gate = base_types::convertor<float, bf16>::convert(__builtin_bit_cast(bf16, (uint16_t)hpair));
                float lin = base_types::convertor<float, bf16>::convert(__builtin_bit_cast(bf16, (uint16_t)(hpair >> 16)));
#else
                float gate = base_types::convertor<float, bf16>::convert(H_ptr[global_row * H_N + hcol]);
                float lin = base_types::convertor<float, bf16>::convert(H_ptr[global_row * H_N + hcol + 1]);
#endif
                float ga = global_inter < REAL_INTER ?
                    base_types::convertor<float, bf16>::convert(dy[row * BLOCK_COL + local_inter]) : 0.0f;
                float nxg = -gate;
                nxg = (nxg < -SW_LIMIT) ? -SW_LIMIT : nxg;
                float sig = __fdiv_rn(1.0f, __fadd_rn(1.0f, __ocml_exp2_f32(nxg * SW_ALPHA_LOG2E)));
                float a37 = (lin < -SW_LIMIT) ? -SW_LIMIT : lin;
                float a38 = -a37;
                float a40 = (a38 < -SW_LIMIT) ? -SW_LIMIT : a38;
                float xl_p1 = __fsub_rn(1.0f, a40);
                float glu_ok = (gate < SW_LIMIT) ? 1.0f : 0.0f;
                float ok_gt = (-SW_LIMIT < lin) ? 1.0f : 0.0f;
                float ok_lt = (lin < SW_LIMIT) ? 1.0f : 0.0f;
                float sprime_inner = fmaf(nxg * __fsub_rn(1.0f, sig), -SW_ALPHA, 1.0f);
                float d_gate = ga * sig * sprime_inner * xl_p1 * glu_ok;
                float d_lin = -(nxg * sig * ga * ok_gt * ok_lt);
                float wgg = f_to_bf16_to_f(d_gate);
                float wlg = f_to_bf16_to_f(d_lin);
                bf16 bg = base_types::convertor<bf16, float>::convert(wgg);
                bf16 bl = base_types::convertor<bf16, float>::convert(wlg);
                wg[lane16] = wgg;
                wl[lane16] = wlg;
                amax = fmaxf(amax, fmaxf(fabsf(wgg), fabsf(wlg)));
#if DGRAD_PACKED_IO
                const uint32_t dhpair = (uint32_t)__builtin_bit_cast(uint16_t, bg) |
                                        ((uint32_t)__builtin_bit_cast(uint16_t, bl) << 16);
                reinterpret_cast<uint32_t *>(Dhbf16_ptr)[(long long)global_row * (H_N / 2) + global_inter] = dhpair;
#else
                Dhbf16_ptr[global_row * H_N + hcol] = bg;
                Dhbf16_ptr[global_row * H_N + hcol + 1] = bl;
#endif
#if FUSED_DGRAD_COLW
                colw_smem[row32 * COLW_LDS_STRIDE + local_hcol] = bg;
                colw_smem[row32 * COLW_LDS_STRIDE + local_hcol + 1] = bl;
#endif
            }
#if NATIVE_MXFP8_CVT && DGRAD_NATIVE_ROW
            // FAST_E8M0: the biased exponent field is floor(log2(amax))+127 for
            // normal values and zero for the all-zero/subnormal row block.
            int row_e8 = (int)((__builtin_bit_cast(unsigned, amax) >> 23) & 0xffu);
            row_e8 = min(row_e8, 254);
            const float row_scale = __builtin_bit_cast(float, (unsigned)row_e8 << 23);
            uint8_t *__restrict__ Dhfp8_bytes = reinterpret_cast<uint8_t *>(Dhfp8_ptr);
            #pragma unroll
            for (int lane16 = 0; lane16 < 16; lane16++) {
                const int global_inter = inter_base + lane16;
                const int hcol = 2 * global_inter;
                short2v packed = {0, 0};
                packed = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
                    packed, wg[lane16], wl[lane16], row_scale, false);
                const unsigned char *rb = reinterpret_cast<const unsigned char *>(&packed);
#if DGRAD_PACKED_IO
                reinterpret_cast<uint16_t *>(Dhfp8_ptr)[(long long)global_row * (H_N / 2) + global_inter] =
                    (uint16_t)rb[0] | ((uint16_t)rb[1] << 8);
#else
                Dhfp8_bytes[global_row * H_N + hcol] = rb[0];
                Dhfp8_bytes[global_row * H_N + hcol + 1] = rb[1];
#endif
            }
            Dhe8_ptr[global_row * (H_N / 32) + inter_base / 16] = (uint8_t)row_e8;
#else
            float e8f = __fadd_rn(floorf(__ocml_log2_f32(fmaxf(amax, 1e-38f))), 127.0f);
            e8f = fminf(fmaxf(e8f, 0.0f), 254.0f);
            float qs = __ocml_exp2_f32(__fsub_rn(127.0f, e8f));
            #pragma unroll
            for (int lane16 = 0; lane16 < 16; lane16++) {
                const int global_inter = inter_base + lane16;
                const int hcol = 2 * global_inter;
                float qg = fminf(fmaxf(__fmul_rn(wg[lane16], qs), -SW_FP8_MAX), SW_FP8_MAX);
                float ql = fminf(fmaxf(__fmul_rn(wl[lane16], qs), -SW_FP8_MAX), SW_FP8_MAX);
#if DGRAD_PACKED_IO
                const fp8e4m3 qg8 = base_types::convertor<fp8e4m3, float>::convert(qg);
                const fp8e4m3 ql8 = base_types::convertor<fp8e4m3, float>::convert(ql);
                reinterpret_cast<uint16_t *>(Dhfp8_ptr)[(long long)global_row * (H_N / 2) + global_inter] =
                    (uint16_t)__builtin_bit_cast(uint8_t, qg8) | ((uint16_t)__builtin_bit_cast(uint8_t, ql8) << 8);
#else
                Dhfp8_ptr[global_row * H_N + hcol] = base_types::convertor<fp8e4m3, float>::convert(qg);
                Dhfp8_ptr[global_row * H_N + hcol + 1] = base_types::convertor<fp8e4m3, float>::convert(ql);
#endif
            }
            Dhe8_ptr[global_row * (H_N / 32) + inter_base / 16] = (uint8_t)e8f;
#endif
        }
#if FUSED_DGRAD_COLW
        __syncthreads();
        constexpr int HCOLS_PER_TILE = BLOCK_COL * 2;
        constexpr int ROWS_PER_PASS = 512 / SWIGLU_BLOCKS;
        const int colw_hcol = tid % HCOLS_PER_TILE;
        const int colw_row = (tid / HCOLS_PER_TILE) * 32;
        const int global_hcol = block_col * HCOLS_PER_TILE + colw_hcol;
        if (global_hcol < H_N) {
            float col_amax = 0.0f;
            #pragma unroll
            for (int mm = 0; mm < 32; mm++) {
                float v = base_types::convertor<float, bf16>::convert(
                    colw_smem[(colw_row + mm) * COLW_LDS_STRIDE + colw_hcol]);
                col_amax = fmaxf(col_amax, fabsf(v));
            }
#if NATIVE_MXFP8_CVT
            int col_e8 = (int)((__builtin_bit_cast(unsigned, col_amax) >> 23) & 0xffu);
            col_e8 = max(0, min(254, col_e8));
            const float scale_f = __builtin_bit_cast(float, (unsigned)col_e8 << 23);
#else
            int col_e8 = (int)floorf(__ocml_log2_f32(fmaxf(col_amax, 1e-38f))) + 127;
            col_e8 = max(0, min(254, col_e8));
            const float col_qscale = __ocml_exp2_f32((float)(127 - col_e8));
#endif
            uint4 packed[2];
            fp8e4m3 *packed_fp8 = reinterpret_cast<fp8e4m3 *>(packed);
#if NATIVE_MXFP8_CVT
            #pragma unroll
            for (int mm = 0; mm < 32; mm += 2) {
                float v0 = base_types::convertor<float, bf16>::convert(
                    colw_smem[(colw_row + mm) * COLW_LDS_STRIDE + colw_hcol]);
                float v1 = base_types::convertor<float, bf16>::convert(
                    colw_smem[(colw_row + mm + 1) * COLW_LDS_STRIDE + colw_hcol]);
                short2v acc = {0, 0};
                acc = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(acc, v0, v1, scale_f, false);
                const unsigned char *rb = reinterpret_cast<const unsigned char *>(&acc);
                reinterpret_cast<unsigned char *>(packed)[mm] = rb[0];
                reinterpret_cast<unsigned char *>(packed)[mm + 1] = rb[1];
            }
#else
            #pragma unroll
            for (int mm = 0; mm < 32; mm++) {
                float v = base_types::convertor<float, bf16>::convert(
                    colw_smem[(colw_row + mm) * COLW_LDS_STRIDE + colw_hcol]);
                float qv = fminf(fmaxf(v * col_qscale, -SW_FP8_MAX), SW_FP8_MAX);
                packed_fp8[mm] = base_types::convertor<fp8e4m3, float>::convert(qv);
            }
#endif
            const long long row_base = block_m + stripe * 128 + pass * ROWS_PER_PASS + colw_row;
            const long long out_base = (long long)global_hcol * GEMM_M + row_base;
            uint8_t *out_bytes = reinterpret_cast<uint8_t *>(Dhcol_ptr);
            *reinterpret_cast<uint4 *>(out_bytes + out_base) = packed[0];
            *reinterpret_cast<uint4 *>(out_bytes + out_base + 16) = packed[1];
            Dhcol_e8_ptr[(long long)global_hcol * (GEMM_M / 32) + row_base / 32] = (uint8_t)col_e8;
        }
        __syncthreads();
#endif
    }
}


__global__ __launch_bounds__(512, 2) void grouped_mxfp8_gemm_dgrad_dswiglu_kernel(
    bf16 *Dhbf16_ptr, fp8e4m3 *Dhfp8_ptr, uint8_t *Dhe8_ptr,
#if FUSED_DGRAD_COLW
    fp8e4m3 *Dhcol_ptr, uint8_t *Dhcol_e8_ptr,
#endif
    const bf16 *__restrict__ H_ptr,
    fp8e4m3 *A_ptr, fp8e4m3 *B_ptr, fp8e8m0 *scale_A_ptr, fp8e8m0 *scale_B_ptr,
    const uint8_t *__restrict__ a_e8_unused,
    const uint8_t *__restrict__ b_e8_unused,
    const int *__restrict__ expert_off) {
 #if !FUSED_DGRAD_COLW
    fp8e4m3 *Dhcol_ptr = nullptr;
    uint8_t *Dhcol_e8_ptr = nullptr;
 #endif
    constexpr int M = GEMM_M, N = GEMM_N, K = GEMM_K, E = GEMM_E;
    constexpr int H_INTER = H_N / 2;

    kittens::gl<fp8e4m3, 1, 1, M, K> A{A_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<fp8e4m3, 1, 1, E * N, K> B{B_ptr, nullptr, nullptr, nullptr, nullptr};

    constexpr int k_iters = K / BLOCK_K;
    constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    constexpr int scale_tiles_N = N / 256;
    constexpr int scale_blocks_per_tile = 256 / BLOCK_COL;

    kittens::gl<fp8e8m0, k_iters * (M / BLOCK_ROW), 1, 16, 64> scale_A_gl{scale_A_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<fp8e8m0, E * k_iters * scale_tiles_N, 1, 16, 64> scale_B_gl{scale_B_ptr, nullptr, nullptr, nullptr, nullptr};

    using ST_A = st_fp8e4m3<HALF_ROW, BLOCK_K, st_16x128_s>;
    using ST_B = st_fp8e4m3<HALF_COL, BLOCK_K, st_16x128_s>;
    using ST_Scale = st<fp8e8m0, 16, 64, st_16x64_s>;
    using RT_A = rt_fp8e4m3<REG_M, BLOCK_K>;
    using RT_B = rt_fp8e4m3<REG_N, BLOCK_K>;
    using RT_C = rt_fl<REG_M, REG_N, col_l, rt_16x16_s>;

    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];
    __shared__ ST_Scale scale_A_smem[2], scale_B_smem[2];

    RT_A a;
    RT_B b0, b1;
    RT_C cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

    constexpr int tiles_M = M / BLOCK_ROW;
    constexpr int tiles_N = N / BLOCK_COL;
    const int NUM_XCDS = 8;
    #ifndef GROUPED_WGM
#define GROUPED_WGM 8
#endif
    const int WGM = GROUPED_WGM;
    int wgid = chiplet_transform_chunked(blockIdx.x, gridDim.x, NUM_XCDS, WGM * WGM);
    int num_wgid_in_group = WGM * tiles_N;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(tiles_M - first_pid_m, WGM);
    int block_row = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int block_col = (wgid % num_wgid_in_group) / group_size_m;
    int block_m = block_row * BLOCK_ROW;
#if MOE_SKIP_EMPTY
    if (block_m >= __builtin_amdgcn_readfirstlane(expert_off[E])) {
        const int tid = threadIdx.x;
        const int h_start = block_col * BLOCK_COL * 2;
        const int h_count = min(BLOCK_COL * 2, H_N - h_start);
        #pragma unroll 1
        for (int i = tid; i < BLOCK_ROW * h_count; i += NUM_THREADS) {
            const int row = i / h_count, col = i % h_count;
            const long long off = (long long)(block_m + row) * H_N + h_start + col;
            reinterpret_cast<uint16_t *>(Dhbf16_ptr)[off] = 0;
            reinterpret_cast<uint8_t *>(Dhfp8_ptr)[off] = 0;
        }
        #pragma unroll 1
        for (int i = tid; i < BLOCK_ROW * (h_count / 32); i += NUM_THREADS) {
            const int row = i / (h_count / 32), col = i % (h_count / 32);
            Dhe8_ptr[(long long)(block_m + row) * (H_N / 32) + h_start / 32 + col] = 0;
        }
#if FUSED_DGRAD_COLW
        #pragma unroll 1
        for (int i = tid; i < h_count * BLOCK_ROW; i += NUM_THREADS) {
            const int col = i / BLOCK_ROW, row = i % BLOCK_ROW;
            reinterpret_cast<uint8_t *>(Dhcol_ptr)[(long long)(h_start + col) * M + block_m + row] = 0;
        }
        #pragma unroll 1
        for (int i = tid; i < h_count * (BLOCK_ROW / 32); i += NUM_THREADS) {
            const int col = i / (BLOCK_ROW / 32), row_block = i % (BLOCK_ROW / 32);
            Dhcol_e8_ptr[(long long)(h_start + col) * (M / 32) + block_m / 32 + row_block] = 0;
        }
#endif
        return;
    }
#endif

    int e = 0;
    #pragma unroll
    for (int i = 1; i < E; i++) e += (expert_off[i] <= block_m);
    e = __builtin_amdgcn_readfirstlane(e);
    const int bcol_base = e * (N / HALF_COL);
    const int sb_base = e * (k_iters * scale_tiles_N);

    int warp_m = warpid() / WARPS_COL;
    int warp_n = warpid() % WARPS_COL;

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
    int b_row_h0 = (block_col % scale_blocks_per_tile) * BLOCK_COL + warp_n * REG_N;
    int b_row_h1 = (block_col % scale_blocks_per_tile) * BLOCK_COL + HALF_COL + warp_n * REG_N;

    uint32_t a_lds[2][2] = {{a_lds_00, a_lds_01}, {a_lds_10, a_lds_11}};
    uint32_t b_lds[2][2] = {{b_lds_00, b_lds_01}, {b_lds_10, b_lds_11}};

#if DOUBLE_BUFFER
    #define LOAD_ITER0(KK)                                                                                                      \
        G::load(As[0][0], A, {0, 0, block_row * 2, (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds_00));      \
        G::load(As[0][1], A, {0, 0, block_row * 2 + 1, (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds_01));  \
        G::load(Bs[0][0], B, {0, 0, bcol_base + block_col * 2, (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds_00));     \
        G::load(Bs[0][1], B, {0, 0, bcol_base + block_col * 2 + 1, (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds_01)); \
        G::load(scale_A_smem[0], scale_A_gl, {(KK) * tiles_M + block_row, 0, 0, 0});                                              \
        G::load(scale_B_smem[0], scale_B_gl, {sb_base + (KK) * scale_tiles_N + block_col / scale_blocks_per_tile, 0, 0, 0});
    #define LOAD_ITER1(KK)                                                                                                      \
        G::load(As[1][0], A, {0, 0, block_row * 2, (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds_10));      \
        G::load(As[1][1], A, {0, 0, block_row * 2 + 1, (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds_11));  \
        G::load(Bs[1][0], B, {0, 0, bcol_base + block_col * 2, (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds_10));     \
        G::load(Bs[1][1], B, {0, 0, bcol_base + block_col * 2 + 1, (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds_11)); \
        G::load(scale_A_smem[1], scale_A_gl, {(KK) * tiles_M + block_row, 0, 0, 0});                                              \
        G::load(scale_B_smem[1], scale_B_gl, {sb_base + (KK) * scale_tiles_N + block_col / scale_blocks_per_tile, 0, 0, 0});
    LOAD_ITER0(0);
    asm volatile("s_waitcnt vmcnt(0)");
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    #pragma unroll 1
    for (int kk = 0; kk < k_iters; kk++) {
        const int cur = kk & 1;
        const int nxt = (kk + 1) & 1;
        if (kk + 1 < k_iters) {
            if (nxt) { LOAD_ITER1(kk + 1); }
            else { LOAD_ITER0(kk + 1); }
        }

        fp8e8m0_4 sa_h0 = pack_scales(scale_A_smem[cur].data, a_row_h0);
        fp8e8m0_4 sa_h1 = pack_scales(scale_A_smem[cur].data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem[cur].data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem[cur].data, b_row_h1);

        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][0], {warp_n, 0}); load(b0, bs0);
        auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][1], {warp_n, 0}); load(b1, bs1);
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur][0], {warp_m, 0}); RT_A a0; load(a0, as0);
        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur][1], {warp_m, 0}); RT_A a1; load(a1, as1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cA, a0, b0, cA, &sa_h0, &sb_h0);
        mma_ABt_scaled(cB, a0, b1, cB, &sa_h0, &sb_h1);
        mma_ABt_scaled(cC, a1, b0, cC, &sa_h1, &sb_h0);
        mma_ABt_scaled(cD, a1, b1, cD, &sa_h1, &sb_h1);

        if (kk + 1 < k_iters) {
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();
        }
    }
    #undef LOAD_ITER0
    #undef LOAD_ITER1
#else
    #pragma unroll 1
    for (int kk = 0; kk < k_iters; kk++) {
        G::load(As[0][0], A, {0, 0, block_row * 2, kk}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[0][0]));
        G::load(As[0][1], A, {0, 0, block_row * 2 + 1, kk}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[0][1]));
        G::load(Bs[0][0], B, {0, 0, bcol_base + block_col * 2, kk}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[0][0]));
        G::load(Bs[0][1], B, {0, 0, bcol_base + block_col * 2 + 1, kk}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[0][1]));
        G::load(scale_A_smem[0], scale_A_gl, {kk * tiles_M + block_row, 0, 0, 0});
        G::load(scale_B_smem[0], scale_B_gl, {sb_base + kk * scale_tiles_N + block_col / scale_blocks_per_tile, 0, 0, 0});
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

#if !DGRAD_REG_EPILOGUE
    // The staged fallback aliases the just-consumed A/B LDS. The last DBUF iteration has no trailing barrier.
    asm volatile("s_waitcnt vmcnt(0)");
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
#endif

#if DGRAD_REG_EPILOGUE
    // Production packed/native row epilogue. Preserve the reference's bf16 GEMM staging in registers,
    // then keep MFMA ownership so every 16-lane group owns one 32-H-column MXFP8 block.
    static_assert((H_INTER % 16) == 0 && (REAL_INTER % 16) == 0);
    const int lane = laneid();
    const int lrow_grp = 4 * (lane / 16);
    const int lcol = lane % 16;

    #define DGRAD_DIRECT_ROW(ACC_VALUE, RR, RB) {                                                                  \
        const int global_row = block_m + (RB) + warp_m * REG_M + i * 16 + lrow_grp + (RR);             \
        const int global_inter = group_inter + lcol;                                                           \
        float ga = 0.0f;                                                                                       \
        if (group_is_real) {                                                                                   \
            bf16 ga_b = base_types::convertor<bf16, float>::convert((ACC_VALUE));                              \
            ga = base_types::convertor<float, bf16>::convert(ga_b);                                            \
        }                                                                                                      \
        const uint32_t hpair = reinterpret_cast<const uint32_t *>(H_ptr)[                                     \
            (long long)global_row * H_INTER + global_inter];                                                   \
        float gate = base_types::convertor<float, bf16>::convert(                                             \
            __builtin_bit_cast(bf16, (uint16_t)hpair));                                                        \
        float lin = base_types::convertor<float, bf16>::convert(                                              \
            __builtin_bit_cast(bf16, (uint16_t)(hpair >> 16)));                                                \
        float nxg = -gate;                                                                                     \
        nxg = (nxg < -SW_LIMIT) ? -SW_LIMIT : nxg;                                                             \
        float sig = __fdiv_rn(1.0f, __fadd_rn(1.0f, DGRAD_FAST_EXP ?                                          \
            __builtin_amdgcn_exp2f(nxg * SW_ALPHA_LOG2E) : __ocml_exp2_f32(nxg * SW_ALPHA_LOG2E)));            \
        float a37 = (lin < -SW_LIMIT) ? -SW_LIMIT : lin;                                                       \
        float a38 = -a37;                                                                                      \
        float a40 = (a38 < -SW_LIMIT) ? -SW_LIMIT : a38;                                                       \
        float xl_p1 = __fsub_rn(1.0f, a40);                                                                    \
        float glu_ok = (gate < SW_LIMIT) ? 1.0f : 0.0f;                                                       \
        float ok_gt = (-SW_LIMIT < lin) ? 1.0f : 0.0f;                                                        \
        float ok_lt = (lin < SW_LIMIT) ? 1.0f : 0.0f;                                                         \
        float sprime_inner = fmaf(nxg * __fsub_rn(1.0f, sig), -SW_ALPHA, 1.0f);                               \
        float d_gate = ga * sig * sprime_inner * xl_p1 * glu_ok;                                              \
        float d_lin = -(nxg * sig * ga * ok_gt * ok_lt);                                                      \
        float wgg = f_to_bf16_to_f(d_gate);                                                                    \
        float wlg = f_to_bf16_to_f(d_lin);                                                                     \
        bf16 bg = base_types::convertor<bf16, float>::convert(wgg);                                            \
        bf16 bl = base_types::convertor<bf16, float>::convert(wlg);                                            \
        const uint32_t dhpair = (uint32_t)__builtin_bit_cast(uint16_t, bg) |                                  \
                                ((uint32_t)__builtin_bit_cast(uint16_t, bl) << 16);                             \
        reinterpret_cast<uint32_t *>(Dhbf16_ptr)[                                                              \
            (long long)global_row * H_INTER + global_inter] = dhpair;                                          \
        float row_amax = fmaxf(fabsf(wgg), fabsf(wlg));                                                        \
        row_amax = dgrad_amax16(row_amax);                                                                     \
        int row_e8 = (int)((__builtin_bit_cast(unsigned, row_amax) >> 23) & 0xffu);                            \
        row_e8 = min(row_e8, 254);                                                                             \
        const float row_scale = __builtin_bit_cast(float, (unsigned)row_e8 << 23);                            \
        short2v packed = {0, 0};                                                                               \
        packed = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(packed, wgg, wlg, row_scale, false);                \
        const unsigned char *rb = reinterpret_cast<const unsigned char *>(&packed);                            \
        reinterpret_cast<uint16_t *>(Dhfp8_ptr)[                                                               \
            (long long)global_row * H_INTER + global_inter] = (uint16_t)rb[0] | ((uint16_t)rb[1] << 8);       \
        if (lcol == 0) Dhe8_ptr[(long long)global_row * (H_N / 32) + group_inter / 16] = (uint8_t)row_e8;     \
    }

    #define DGRAD_DIRECT_ACC(ACC, COL_BASE, ROW_BASE) {                                                        \
        _Pragma("unroll") for (int i = 0; i < REG_M / 16; i++) {                                            \
            _Pragma("unroll") for (int j = 0; j < REG_N / 16; j++) {                                        \
                const int group_inter = block_col * BLOCK_COL + (COL_BASE) + warp_n * REG_N + j * 16;        \
                if (group_inter < H_INTER) {                                                                   \
                    const bool group_is_real = group_inter < REAL_INTER;                                       \
                    DGRAD_DIRECT_ROW((ACC).tiles[i][j].data[0].x, 0, ROW_BASE);                                         \
                    DGRAD_DIRECT_ROW((ACC).tiles[i][j].data[0].y, 1, ROW_BASE);                                         \
                    DGRAD_DIRECT_ROW((ACC).tiles[i][j].data[1].x, 2, ROW_BASE);                                         \
                    DGRAD_DIRECT_ROW((ACC).tiles[i][j].data[1].y, 3, ROW_BASE);                                         \
                }                                                                                              \
            }                                                                                                  \
        }                                                                                                      \
    }

    DGRAD_DIRECT_ACC(cA, 0, 0);
    DGRAD_DIRECT_ACC(cB, HALF_COL, 0);
    DGRAD_DIRECT_ACC(cC, 0, HALF_ROW);
    DGRAD_DIRECT_ACC(cD, HALF_COL, HALF_ROW);
    #undef DGRAD_DIRECT_ACC
    #undef DGRAD_DIRECT_ROW
#else
    bf16 *__restrict__ dy = reinterpret_cast<bf16 *>(&As[0][0]);
#if FUSED_DGRAD_COLW
    bf16 *__restrict__ colw_smem = reinterpret_cast<bf16 *>(&Bs[0][0]);
#else
    bf16 *__restrict__ colw_smem = nullptr;
#endif
    const int lane = laneid();
    const int lrow_grp = 4 * (lane / 16);
    const int lcol = lane % 16;
    static_assert(SWIGLU_BLOCKS <= 16);

    #define STORE_ACC(ACC, COL_BASE)                                                                           \
    {                                                                                                         \
        _Pragma("unroll") for (int i = 0; i < REG_M / 16; i++) {                                             \
            _Pragma("unroll") for (int j = 0; j < REG_N / 16; j++) {                                         \
                const int local_col = (COL_BASE) + warp_n * REG_N + j * 16 + lcol;                            \
                if (local_col < BLOCK_COL) {                                                                  \
                    const int row = warp_m * REG_M + i * 16 + lrow_grp;                                      \
                    dy[(row + 0) * BLOCK_COL + local_col] = base_types::convertor<bf16, float>::convert(      \
                        (ACC).tiles[i][j].data[0].x);                                                         \
                    dy[(row + 1) * BLOCK_COL + local_col] = base_types::convertor<bf16, float>::convert(      \
                        (ACC).tiles[i][j].data[0].y);                                                         \
                    dy[(row + 2) * BLOCK_COL + local_col] = base_types::convertor<bf16, float>::convert(      \
                        (ACC).tiles[i][j].data[1].x);                                                         \
                    dy[(row + 3) * BLOCK_COL + local_col] = base_types::convertor<bf16, float>::convert(      \
                        (ACC).tiles[i][j].data[1].y);                                                         \
                }                                                                                             \
            }                                                                                                 \
        }                                                                                                     \
    }

    STORE_ACC(cA, 0);
    STORE_ACC(cB, HALF_COL);
    __builtin_amdgcn_s_barrier();
    dswiglu_stripe(dy, H_ptr, Dhbf16_ptr, Dhfp8_ptr, Dhe8_ptr, Dhcol_ptr, Dhcol_e8_ptr,
                   colw_smem, block_m, block_col, 0);
    __builtin_amdgcn_s_barrier();
    STORE_ACC(cC, 0);
    STORE_ACC(cD, HALF_COL);
    __builtin_amdgcn_s_barrier();
    dswiglu_stripe(dy, H_ptr, Dhbf16_ptr, Dhfp8_ptr, Dhe8_ptr, Dhcol_ptr, Dhcol_e8_ptr,
                   colw_smem, block_m, block_col, 1);
    #undef STORE_ACC
#endif
}
