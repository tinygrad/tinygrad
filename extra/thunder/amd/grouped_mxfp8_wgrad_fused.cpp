#include "kittens.cuh"
#include <hip/hip_fp8.h>

using namespace kittens;

// Fused wgrad: reads bf16 g (M,N) and xg (M,K) directly and does the transpose + mxfp8 block-quantize IN the
// operand-load stage (per-32-M-block amax in registers -> e8 -> fp8, written straight into the swizzled MMA staging
// tiles), instead of two separate transpose_quantize_mxfp8 HBM kernels feeding pre-transposed fp8. Byte-exact with
// the transpose_quantize->gemm path: identical amax/e8/quantize math (transpose_quantize_mxfp8.cpp) + identical
// scale-smem layout (SCALE_PACK in grouped_mxfp8_gemm_swiglu.cpp). MMA/accumulate/store are unchanged from
// grouped_mxfp8_wgrad.cpp.
//
// DOUBLE_BUFFER (default): software-pipelined K-loop mirroring grouped_mxfp8_gemm.cpp. The producer for iter kk+1
// (load bf16 -> amax-in-registers -> quantize -> ds_write fp8 into the OTHER LDS buffer) runs while the matrix core
// consumes iter kk from the current buffer. The LDS holds ONLY fp8 (no bf16 staging tile), so As[2][2]+Bs[2][2]
// (128KB) + scale smem = 132KB fits CDNA4's 160KB LDS at 2 waves/SIMD (245->256 VGPR, few spills). Byte-identical
// to the single-buffered path (WGRAD_DBUF=0).
//
// MEASURED (MI350X, throttled): pipelining the load fixed the ORIGINAL collapse (167 -> 321 TFLOPS, ~2x) and IS
// byte-exact, BUT this fused kernel is STILL a ~2.7-2.9x NET LOSS vs [2x transpose_quantize + baseline wgrad gemm]
// and cannot be fixed by more pipelining. Proof: WGRAD_DBUF=1 (321 TFLOPS) ~= WGRAD_DBUF=0 (309) -- the double
// buffer barely helps, so the PRODUCE, not the pipeline, is the wall. The in-load produce (transposed bf16 gather +
// per-32-block amax + mxfp8 quantize) costs ~4x the MMA it feeds, PER OUTPUT TILE, and it recurs on every operand
// reuse: wgrad reuses g across all K-tiles and x across all N-tiles, so fusing the quantize into the LOAD
// re-quantizes g tiles_K times and x tiles_N times. The separate path amortizes the quantize to exactly ONCE (a
// dedicated transpose_quantize_mxfp8 kernel, ~820us for fc1) and the gemm then re-reads only compact, cacheable fp8.
// You cannot hide 4 units of produce behind 1 unit of MMA. The correct place to fold this quantize is the EPILOGUE
// of the kernel that PRODUCES g and x (each written once), NOT the wgrad operand load. Kept gated (FUSED_WGRAD_QUANT
// default-off) as a documented dead-end for operand-load fusion.
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
#define WGRAD_DBUF 1
#endif

// NATIVE_MXFP8_CVT: gfx950 hardware scaled-fp8 conversion (v_cvt_scalef32_pk_fp8_f32) + exponent-bit e8m0 extract in
// the in-load quantize, replacing the software floor(log2f)+exp2f + scalar __hip_cvt_float_to_fp8. Byte-exact (see
// transpose_quantize_mxfp8.cpp). Default off; the standalone-quantize control decides whether native produce flips
// the fused-wgrad verdict from loss to win.
#ifndef NATIVE_MXFP8_CVT
#define NATIVE_MXFP8_CVT 0
#endif
#if NATIVE_MXFP8_CVT
typedef short short2v __attribute__((ext_vector_type(2)));
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
constexpr float FP8_MAX  = 448.0f;

using G = kittens::group<NUM_WARPS>;

using ST_A     = st_fp8e4m3<HALF_ROW, BLOCK_K, st_16x128_s>;
using ST_B     = st_fp8e4m3<HALF_COL, BLOCK_K, st_16x128_s>;

// Produce one block-quantized operand tile straight from bf16 HBM into the two swizzled fp8 staging half-tiles
// S0/S1 (128 feature-rows each) + flat e8 scale smem. `src` is g (stride N) or xg (stride K); `fbase` is the
// feature column base (nbase / kbase); `mrow0` is the contraction-row base (kk*BLOCK_K) into src's M axis.
// Each of the 1024 (feature-row, 32-M-block) tasks: load its 32 contraction values into registers (coalesced
// across the warp -- adjacent threads = adjacent feature columns), block-32 amax, e8, quantize, swizzled fp8 store.
template<typename ST>
__device__ __forceinline__ void produce(ST &S0, ST &S1, uint8_t *sc_bytes,
                                         const bf16 *__restrict__ src, long mrow0, int stride, int fbase) {
    const int tid = threadIdx.x;
    #pragma unroll 1
    for (int task = tid; task < BLOCK_ROW * 4; task += NUM_WARPS * WARP_THREADS) {
        const int row  = task & (BLOCK_ROW - 1);   // 0..255  (output feature = N/K)
        const int sm   = task >> 8;                // 0..3    (which 32-wide block along M)
        const int col0 = sm * 32;
        const bf16 *col = &src[(mrow0 + col0) * (long)stride + (fbase + row)];   // 32 M-values at stride `stride`
        bf16 vals[32];                             // keep bf16 bits (16 VGPR) not float (32) -> less register pressure
        float amax = 0.0f;
        #pragma unroll
        for (int mm = 0; mm < 32; mm++) { bf16 v = col[mm * (long)stride]; vals[mm] = v; amax = fmaxf(amax, fabsf((float)v)); }
#if NATIVE_MXFP8_CVT
        int e8 = (int)((__builtin_bit_cast(unsigned, amax) >> 23) & 0xFFu);  // e8m0 = biased exponent field of amax
        e8 = max(0, min(254, e8));
#else
        int e8 = (int)floorf(log2f(fmaxf(amax, 1e-38f))) + 127;
        e8 = max(0, min(254, e8));
        float qscale = exp2f((float)(127 - e8));
#endif
        ST &S = (row < HALF_ROW) ? S0 : S1;
        const int r = row & (HALF_ROW - 1);        // 0..127 within the half-tile
        __hip_fp8_storage_t *dst = reinterpret_cast<__hip_fp8_storage_t *>(&S.data[0]);
#if NATIVE_MXFP8_CVT
        // hardware scaled cvt: scale exponent field = e8 -> fp8(v*2^(127-e8)); 2 fp8/instr, swizzled per-byte store
        const float scale_f = __builtin_bit_cast(float, (unsigned)e8 << 23);
        #pragma unroll
        for (int mm = 0; mm < 32; mm += 2) {
            short2v acc = {0, 0};
            acc = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(acc, (float)vals[mm], (float)vals[mm + 1], scale_f, false);
            const unsigned char *rb = reinterpret_cast<const unsigned char *>(&acc);
            dst[ST::swizzle({r, col0 + mm})]     = rb[0];
            dst[ST::swizzle({r, col0 + mm + 1})] = rb[1];
        }
#else
        #pragma unroll
        for (int mm = 0; mm < 32; mm++) {
            __hip_fp8_storage_t q = __hip_cvt_float_to_fp8(fmaxf(-FP8_MAX, fminf(FP8_MAX, (float)vals[mm] * qscale)),
                                                           __HIP_SATFINITE, __HIP_E4M3);
            dst[ST::swizzle({r, col0 + mm})] = q;
        }
#endif
        sc_bytes[row * 4 + sm] = (uint8_t)e8;       // flat s4[row] = 4 e8 (matches SCALE_PACK / pack_scales)
    }
}

__global__ __launch_bounds__(512, 2) void grouped_mxfp8_wgrad_fused_kernel(bf16 *C_ptr,
    const bf16 *__restrict__ g_ptr,       // g   (M, N) bf16
    const bf16 *__restrict__ x_ptr,       // xg  (M, K) bf16
    const int *__restrict__ expert_off) {
    constexpr int M = WGRAD_M, N = WGRAD_N, K = WGRAD_K, E = WGRAD_E;

    kittens::gl<bf16, 1, 1, E * N, K>    C{C_ptr, nullptr, nullptr, nullptr, nullptr};  // grad_w, experts stacked

    constexpr int tiles_N     = N / BLOCK_ROW;
    constexpr int tiles_K     = K / BLOCK_COL;

    using ST_Scale = st<fp8e8m0, 16, 64, st_16x64_s>;
    using RT_A     = rt_fp8e4m3<REG_M, BLOCK_K>;
    using RT_B     = rt_fp8e4m3<REG_N, BLOCK_K>;
    using RT_C     = rt_fl<REG_M, REG_N, col_l, rt_16x16_s>;

    RT_C cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

    const int wg        = blockIdx.x;
    const int e         = wg / (tiles_N * tiles_K);
    const int rem       = wg % (tiles_N * tiles_K);
    const int block_row = rem / tiles_K;   // over grad_w rows (N)
    const int block_col = rem % tiles_K;   // over grad_w cols (K)
    const int nbase     = block_row * BLOCK_ROW;   // g column base (N)
    const int kbase     = block_col * BLOCK_COL;   // x column base (K)

    const int o0  = __builtin_amdgcn_readfirstlane(expert_off[e]);
    const int kk0 = o0 / BLOCK_K;
    const int nk  = (__builtin_amdgcn_readfirstlane(expert_off[e + 1]) - o0) / BLOCK_K;

    const int warp_m = warpid() / WARPS_COL;
    const int warp_n = warpid() % WARPS_COL;

    const int a_row_h0 = warp_m * REG_M;
    const int a_row_h1 = HALF_ROW + warp_m * REG_M;
    const int b_row_h0 = warp_n * REG_N;
    const int b_row_h1 = HALF_COL + warp_n * REG_N;

#if WGRAD_DBUF
    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];
    __shared__ ST_Scale scale_A_smem[2], scale_B_smem[2];

    RT_A a;
    RT_B b0, b1;

    // prologue: produce tile kk0 into buffer 0 (skip for empty experts -> the loop below stores zeros, and the
    // unconditional read at kk0*BLOCK_K could be OOB at the tail; nk is uniform across the workgroup so the guarded
    // barrier is safe)
    if (nk > 0) {
        produce<ST_A>(As[0][0], As[0][1], reinterpret_cast<uint8_t *>(&scale_A_smem[0].data[0]), g_ptr, (long)kk0 * BLOCK_K, N, nbase);
        produce<ST_B>(Bs[0][0], Bs[0][1], reinterpret_cast<uint8_t *>(&scale_B_smem[0].data[0]), x_ptr, (long)kk0 * BLOCK_K, K, kbase);
        __builtin_amdgcn_s_barrier();
    }

    #pragma unroll 1
    for (int t = 0; t < nk; t++) {
        const int cur = t & 1;
        const int nxt = (t + 1) & 1;
        if (t + 1 < nk) {
            const long mrow_n = (long)(kk0 + t + 1) * BLOCK_K;
            produce<ST_A>(As[nxt][0], As[nxt][1], reinterpret_cast<uint8_t *>(&scale_A_smem[nxt].data[0]), g_ptr, mrow_n, N, nbase);
            produce<ST_B>(Bs[nxt][0], Bs[nxt][1], reinterpret_cast<uint8_t *>(&scale_B_smem[nxt].data[0]), x_ptr, mrow_n, K, kbase);
        }

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

        if (t + 1 < nk) __builtin_amdgcn_s_barrier();
    }
#else
    __shared__ ST_A As[2];
    __shared__ ST_B Bs[2];
    __shared__ ST_Scale scale_A_smem, scale_B_smem;
    uint8_t *scA_bytes = reinterpret_cast<uint8_t *>(&scale_A_smem.data[0]);
    uint8_t *scB_bytes = reinterpret_cast<uint8_t *>(&scale_B_smem.data[0]);

    RT_A a;
    RT_B b0, b1;

    #pragma unroll 1
    for (int t = 0; t < nk; t++) {
        const int kk = kk0 + t;
        const long mrow0 = (long)kk * BLOCK_K;
        produce<ST_A>(As[0], As[1], scA_bytes, g_ptr, mrow0, N, nbase);
        produce<ST_B>(Bs[0], Bs[1], scB_bytes, x_ptr, mrow0, K, kbase);
        __builtin_amdgcn_s_barrier();

        fp8e8m0_4 sa_h0 = pack_scales(scale_A_smem.data, a_row_h0);
        fp8e8m0_4 sa_h1 = pack_scales(scale_A_smem.data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem.data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem.data, b_row_h1);

        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[0], {warp_n, 0}); load(b0, bs0);
        auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[1], {warp_n, 0}); load(b1, bs1);
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[0], {warp_m, 0}); load(a, as0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cA, a, b0, cA, &sa_h0, &sb_h0);
        mma_ABt_scaled(cB, a, b1, cB, &sa_h0, &sb_h1);
        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[1], {warp_m, 0}); load(a, as1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cC, a, b0, cC, &sa_h1, &sb_h0);
        mma_ABt_scaled(cD, a, b1, cD, &sa_h1, &sb_h1);
        __builtin_amdgcn_s_barrier();
    }
#endif

    const int crow_base = e * (N / REG_M);
    store(C, cA, {0, 0, crow_base + block_row * WARPS_ROW * 2 + warp_m,              block_col * WARPS_COL * 2 + warp_n});
    store(C, cB, {0, 0, crow_base + block_row * WARPS_ROW * 2 + warp_m,              block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
    store(C, cC, {0, 0, crow_base + block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m,  block_col * WARPS_COL * 2 + warp_n});
    store(C, cD, {0, 0, crow_base + block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m,  block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
}
