// FUSED_FC1_COLW (gated, default off): ALSO emit the COLUMNWISE (transposed) mxfp8 of the post-SwiGLU activation
// from K1's epilogue (NVIDIA's K1 "quantize-once in the producing kernel" pattern). See the block comment above the
// swiglu constants for the full description. NATIVE_MXFP8_CVT selects the hardware scaled-fp8 cvt + exponent-bit e8m0
// (must match how the reference transpose_quantize_mxfp8 is compiled). Defaults kept here so the conditional include
// (software path needs __hip_cvt_float_to_fp8) resolves before "kittens.cuh".
#ifndef FUSED_FC1_COLW
#define FUSED_FC1_COLW 0
#endif
#ifndef FC1_XCOL_ANCHOR
#define FC1_XCOL_ANCHOR 0
#endif
#ifndef NATIVE_MXFP8_CVT
#define NATIVE_MXFP8_CVT 0
#endif
#ifndef MOE_SKIP_EMPTY
#define MOE_SKIP_EMPTY 0
#endif
#ifndef MOE_SKIP_NO_ZERO
#define MOE_SKIP_NO_ZERO 0
#endif
// Exact GPT-OSS production specialization: global empty row blocks still initialize the FC2 physical column tail,
// but all normal-width H/Y/Ye8/Ycol/Ycol_e8 stores are dead. Keep this separate from generic MOE_SKIP_NO_ZERO.
#ifndef FC1_SKIP_EMPTY_MAIN_ZERO
#define FC1_SKIP_EMPTY_MAIN_ZERO 0
#endif
// REG_EPILOGUE experiment: split the four accumulator rows of each gate/linear lane pair by parity instead of
// redundantly evaluating identical SwiGLU values in both lanes. Default off retains the proven epilogue verbatim.
#ifndef FC1_SPLIT_ROWS
#define FC1_SPLIT_ROWS 0
#endif
// Native rowwise MXFP8 for the split-row epilogue: exponent-bit e8m0 plus one packed scaled-fp8 conversion for
// the two inter values owned by a lane/row. Opt-in and only valid with the gfx950 native conversion contract.
#ifndef FC1_NATIVE_ROW
#define FC1_NATIVE_ROW 0
#endif
#ifndef FC1_FAST_EXP
#define FC1_FAST_EXP 0
#endif
#ifndef FC1_FAST_MATH
#define FC1_FAST_MATH 0
#endif
#ifndef FC1_FAST_RCP
#define FC1_FAST_RCP 0
#endif
#ifndef FC1_FAST_UNROUNDED
#define FC1_FAST_UNROUNDED 0
#endif
#ifndef FC1_MED3_CLAMP
#define FC1_MED3_CLAMP 0
#endif
#ifndef FC1_BINARY_EXPERT
#define FC1_BINARY_EXPERT 0
#endif
// The SwiGLU gate/linear partner is always lane^1. On gfx950 the fixed ds_swizzle form avoids the address
// calculation emitted for the generic shuffle intrinsic; the wrapper enables this only for exact GPT-OSS FC1.
#ifndef FC1_DS_SWIZZLE_XOR1
#define FC1_DS_SWIZZLE_XOR1 0
#endif
#ifndef FC1_REAL_N
#define FC1_REAL_N GEMM_N
#endif
#ifndef FC1_HALF_N_TAIL
#define FC1_HALF_N_TAIL 0
#endif
#ifndef FC1_AMAX_PINGPONG
#define FC1_AMAX_PINGPONG 0
#endif
#ifndef FC1_WARP_SCALE32
#define FC1_WARP_SCALE32 0
#endif
#ifndef FC1_DUAL_COLW_LDS
#define FC1_DUAL_COLW_LDS 0
#endif
// Pair adjacent 32-token columnwise blocks per thread and emit their two
// e8m0 scales with one aligned store. This is valid only when both 128-row
// stripes remain resident in separate dead MMA LDS buffers.
#ifndef FC1_COLW_ADJ_SCALE2
#define FC1_COLW_ADJ_SCALE2 0
#endif
#ifndef FC1_COLW_PACKED
#define FC1_COLW_PACKED 0
#endif
// Stage the four rowwise e8m0 bytes in otherwise-unused columnwise LDS and
// commit them with one aligned dword store (or a short on the half-N tail).
#ifndef FC1_ROW_SCALE_LDS4
#define FC1_ROW_SCALE_LDS4 0
#endif
// Keep the eight wave-uniform ping-pong LDS addresses as scalar values. Indexing a local [2][2] array with the
// runtime buffer selector makes LLVM allocate 32 bytes/thread in LDS (16KB/workgroup) and reload it in the K loop.
#ifndef FC1_SCALAR_LDS_BASES
#define FC1_SCALAR_LDS_BASES 0
#endif
#ifndef FC1_FULL_LDS_TILES
#define FC1_FULL_LDS_TILES 0
#endif
#ifndef FC1_FULL_LDS_TAIL_HALF
#define FC1_FULL_LDS_TAIL_HALF 0
#endif
#ifndef FC1_FIXED_WGM_MAP
#define FC1_FIXED_WGM_MAP 0
#endif
#ifndef FC1_VECTOR_TAIL_ZERO
#define FC1_VECTOR_TAIL_ZERO 0
#endif
#ifndef FC1_SKIP_H_TAIL_ZERO
#define FC1_SKIP_H_TAIL_ZERO 0
#endif
#ifndef FC1_VECTOR_EMPTY_ZERO
#define FC1_VECTOR_EMPTY_ZERO 0
#endif
#ifndef FC1_ZERO_NT
#define FC1_ZERO_NT 0
#endif
#ifndef FC1_RESTRICT_READS
#define FC1_RESTRICT_READS 0
#endif
#ifndef FC1_PACK_A32
#define FC1_PACK_A32 0
#endif
#ifndef FC1_EP_JJ_OUTER
#define FC1_EP_JJ_OUTER 0
#endif
// Exact GPT-OSS routing exposes true (unpadded) expert counts. Dispatch initializes every padded activation row to
// q=0, so an expert's final 256-row block can omit its lower-half A read and cC/cD MFMAs when <=128 rows are live;
// the ordinary zero accumulators still flow through bias/SwiGLU and reproduce every padded output byte.
#ifndef FC1_SKIP_PAD_HALF_ROWS
#define FC1_SKIP_PAD_HALF_ROWS 0
#endif
// Exact GPT-OSS padding specialization. In a final expert tile, a lower 32-row wave band beyond expert_counts has
// exact +0 accumulators and therefore repeats the same bias/SwiGLU result on every row. Evaluate that scalar math
// once per lane/column and reuse it while preserving every H/Y/Ye8/Ycol/Ycol_e8 output byte.
#ifndef FC1_PAD_LOWER_REUSE
#define FC1_PAD_LOWER_REUSE 0
#endif
// Once every backward consumer contracts only the exact ceil(count/128) live prefix, a final expert tile whose
// lower 128 rows are wholly padding can leave that half of H/Y/Ye8/Ycol/Ycol_e8 untouched. This is deliberately
// separate from PAD_LOWER_REUSE: the latter preserves the full initialized-output ABI, while this specialization
// relies on the GPT-OSS count-aware producer/consumer contract and removes the dead epilogue work entirely.
#ifndef FC1_OMIT_PAD_LOWER_OUTPUTS
#define FC1_OMIT_PAD_LOWER_OUTPUTS 0
#endif
// Exact GPT-OSS extension of the padding contraction skip. For a partial
// expert-final row tile, omit the upper 32-row wave bands that begin beyond
// expert_counts. The common complete-upper-half path is a separate template
// instantiation, so it does not pay a predicate in each of the 23 K tiles.
#ifndef FC1_PAD_UPPER_SKIP
#define FC1_PAD_UPPER_SKIP 0
#endif
// GPT-OSS has 22 complete 256-column FC1 tiles and one 128-column tail.  A
// single uniform outer branch lets the complete-tile contraction discard all
// half-tail B1/cB/cD predicates while preserving the dedicated tail body.
#ifndef FC1_INTERIOR_FAST
#define FC1_INTERIOR_FAST 0
#endif
// The exact GPT-OSS interior-fast path has fixed FP8 byte strides and wave-uniform
// tile coordinates. Pass the already-computed byte SOFF directly to the raw
// global-to-LDS loader instead of reconstructing a GL coordinate/pointer
// difference for every A/B prefetch.
#ifndef FC1_PRECOMPUTED_SOFF
#define FC1_PRECOMPUTED_SOFF 0
#endif
// The exact GPT-OSS register epilogue reuses every bias value across four row
// warps. Stage one 256-column tile in dead scale LDS before entering it.
#ifndef FC1_BIAS_LDS
#define FC1_BIAS_LDS 0
#endif
#ifndef FC1_MAP_XCDS
#define FC1_MAP_XCDS 8
#endif
#ifndef FC1_MAP_CHUNK
#define FC1_MAP_CHUNK (GROUPED_WGM * GROUPED_WGM)
#endif
#if FC1_RESTRICT_READS & 1
#define FC1_A_RESTRICT __restrict__
#else
#define FC1_A_RESTRICT
#endif
#if FC1_RESTRICT_READS & 2
#define FC1_B_RESTRICT __restrict__
#else
#define FC1_B_RESTRICT
#endif
#if FC1_RESTRICT_READS & 4
#define FC1_SCALE_A_RESTRICT __restrict__
#else
#define FC1_SCALE_A_RESTRICT
#endif
#if FC1_RESTRICT_READS & 8
#define FC1_SCALE_B_RESTRICT __restrict__
#else
#define FC1_SCALE_B_RESTRICT
#endif
#if FC1_ZERO_NT
#define FC1_STORE_ZERO(PTR, VALUE) __builtin_nontemporal_store((VALUE), (PTR))
#else
#define FC1_STORE_ZERO(PTR, VALUE) (*(PTR) = (VALUE))
#endif
#if FC1_FULL_LDS_TILES && LOW_LDS
#error "FC1_FULL_LDS_TILES requires the double-buffered path"
#endif
#if FC1_FULL_LDS_TAIL_HALF && !FC1_FULL_LDS_TILES
#error "FC1_FULL_LDS_TAIL_HALF requires FC1_FULL_LDS_TILES"
#endif
#if FC1_HALF_N_TAIL && (!defined(DOUBLE_BUFFER) || LOW_LDS)
#error "FC1_HALF_N_TAIL is implemented for the production double-buffered path"
#endif
#if FC1_NATIVE_ROW && !NATIVE_MXFP8_CVT
#error "FC1_NATIVE_ROW requires NATIVE_MXFP8_CVT=1"
#endif
#if FC1_WARP_SCALE32 && (!FC1_SPLIT_ROWS || !FC1_NATIVE_ROW)
#error "FC1_WARP_SCALE32 requires the split-row native epilogue"
#endif
#if FC1_DUAL_COLW_LDS && (!FC1_WARP_SCALE32 || !FUSED_FC1_COLW || LOW_LDS)
#error "FC1_DUAL_COLW_LDS requires the double-buffered warp-scale32 fused-columnwise epilogue"
#endif
#if FC1_COLW_ADJ_SCALE2 && !FC1_DUAL_COLW_LDS
#error "FC1_COLW_ADJ_SCALE2 requires FC1_DUAL_COLW_LDS"
#endif
#if FC1_ROW_SCALE_LDS4 && !FC1_DUAL_COLW_LDS
#error "FC1_ROW_SCALE_LDS4 requires FC1_DUAL_COLW_LDS"
#endif
#if FC1_OMIT_PAD_LOWER_OUTPUTS && (!FC1_SKIP_PAD_HALF_ROWS || !FC1_PAD_LOWER_REUSE || !FC1_DUAL_COLW_LDS)
#error "FC1_OMIT_PAD_LOWER_OUTPUTS requires the exact count-aware dual-LDS padded-half specialization"
#endif
#if FC1_BIAS_LDS && !(REG_EPILOGUE || LOW_LDS)
#error "FC1_BIAS_LDS requires the register epilogue"
#endif
#if FUSED_FC1_COLW && !NATIVE_MXFP8_CVT
#include <hip/hip_fp8.h>
#endif

#include "kittens.cuh"

using namespace kittens;

#if (FUSED_FC1_COLW || FC1_NATIVE_ROW) && NATIVE_MXFP8_CVT
typedef short short2v __attribute__((ext_vector_type(2)));
#endif
typedef uint32_t uint4v __attribute__((ext_vector_type(4)));

#if FC1_PRECOMPUTED_SOFF
template <typename ST>
__device__ __forceinline__ void fc1_load_soff(ST &dst,
    const uint32_t *__restrict__ swizzled_offsets, i32x4 srd,
    uint32_t soff, uint32_t lds_base) {
    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * 512;
    constexpr int memcpy_per_tile = ST::rows * ST::cols * sizeof(typename ST::dtype) / bytes_per_memcpy;
    static_assert((ST::rows * ST::cols * sizeof(typename ST::dtype)) % bytes_per_memcpy == 0);
    (void)dst;
    asm volatile("" : "+s"(soff), "+s"(lds_base));
    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {
        int32_t lds_byte = lds_base;
        asm volatile("" : "+s"(lds_byte));
        asm volatile("s_mov_b32 m0, %0" :: "s"(lds_byte));
        llvm_amdgcn_raw_buffer_load_lds(srd, (as3_uint32_ptr)0, bytes_per_thread,
                                       swizzled_offsets[i], soff, 0,
                                       static_cast<int>(coherency::cache_all));
        lds_base += bytes_per_memcpy;
    }
}
#endif

#if FC1_DS_SWIZZLE_XOR1
#define FC1_XOR1(V) __hip_ds_swizzlef((V), 0x041f)
#else
#define FC1_XOR1(V) __shfl_xor((V), 1)
#endif

// tinygrad's AMD HIPRenderer lowers .exp2()/.log2() to __ocml_exp2_f32/__ocml_log2_f32 (accurate OCML, NOT the
// ~1ULP hardware v_exp/v_log) and compiles WITHOUT -ffast-math. This kernel is built with -ffast-math (for the
// gemm), so the SwiGLU epilogue calls these OCML symbols explicitly and uses correctly-rounded _rn intrinsics
// for every arithmetic op, which are immune to fast-math's rcp substitution / FMA-contraction / reassociation.
// That makes the fused SwiGLU byte-identical to the reference (gemm -> bf16 add -> fused_swiglu_quantize).
extern "C" __device__ float __ocml_exp2_f32(float);
extern "C" __device__ float __ocml_log2_f32(float);
__device__ __forceinline__ float sw_exp2(float x) {
#if FC1_FAST_EXP
    return __builtin_amdgcn_exp2f(x);
#else
    return __ocml_exp2_f32(x);
#endif
}

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
#ifndef FC1_OUT_INTER
#define FC1_OUT_INTER (GEMM_N / 2)
#endif
#ifndef FC1_BIAS_N
#define FC1_BIAS_N GEMM_N
#endif
#ifndef FC1_REAL_K
#define FC1_REAL_K GEMM_K
#endif
#ifndef FC1_PEEL_FINAL
#define FC1_PEEL_FINAL 0
#endif

// MFU tile experiments (all gated; default 256x256 is the MEASURED CDNA4 optimum for this kernel). BLOCK_COL stays
// 256 so INTER_PER_BLOCK=128 (the swiglu gate/linear pairing) is UNCHANGED, and the mx_pack scale_A super-tiles
// (256 rows each) are loaded per block: st_start/srow_a index the spanned super-tile(s), reusing the proven
// scale-load + swizzle (no SCALE_PACK / mx_pack rework, byte-exact).
//   TILE_M128 (128x256): REG_M 64->32, accumulator VGPR halved. Byte-exact, and with LOW_LDS reaches 3 waves/SIMD
//     (vs 2). But MEASURED SLOWER: 128+LOW_LDS 617 vs 256+DBUF 861 TFLOPS -- halving rows ~2x's the B-load/FLOP
//     (256xK B reused over 128 rows not 256), which dwarfs the occupancy gain (B-traffic -22%, lost DBUF -9%).
//   TILE_M512 (512x256): better B-reuse (B over 512 rows) BUT INFEASIBLE -- 512x256/8warps = 256 fp32 accumulators
//     PER LANE = the whole architectural VGPR budget; the AMDGPU backend crashes ("Cannot select mfma.scale") on
//     the register pressure. No warp layout helps (accumulators/lane = tile/warps is invariant). Dead end.

// Kernel
constexpr int NUM_WARPS  = 8;
#if FC1_WARP_SCALE32
constexpr int WARPS_ROW  = 4;
constexpr int WARPS_COL  = 2;
#else
constexpr int WARPS_ROW  = 2;
constexpr int WARPS_COL  = 4;
#endif
#if defined(TILE_M512)
constexpr int BLOCK_ROW  = 512;
#elif defined(TILE_M128)
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
constexpr int SCALE_ROWS = 256;                                       // mx_pack packs scale_A in 256-row super-tiles (BLOCK_ROW-independent)
constexpr int SCALE_SUP  = (BLOCK_ROW + SCALE_ROWS - 1) / SCALE_ROWS; // 256-row super-tiles a block SPANS: 1 (128/256), 2 (512)

// gpt-oss clamped swiglu constants (must match extra/llama_kernels/fused_swiglu_quantize_gptoss)
constexpr float SW_LIMIT   = 7.0f;
constexpr float SW_FP8_MAX = 448.0f;
// (ALPHA * -LOG2E), folded in fp64 then narrowed to fp32 (matches the reference constant fold)
constexpr float SW_NEG_ALPHA_LOG2E = (float)(1.702 * -1.4426950408889634);

// FUSED_FC1_COLW (gated, default off): ALSO emit the COLUMNWISE (transposed) mxfp8 of the post-SwiGLU activation from
// K1's epilogue -- NVIDIA's "quantize-once in the producing kernel" K1 pattern. The down/FC2 wgrad (dW2 = y^T . dz)
// needs y transposed+quantized (fp8 (INTER, M) + e8m0 blocks along the 32-TOKEN axis); currently a separate
// transpose_quantize_mxfp8 kernel re-reads y from HBM to produce it. Here y is already in registers post-SwiGLU, so
// we stage x_phys = dequant(rowwise y_fp8) into a small LDS tile (aliasing the dead Bs mma buffers), then read it
// COLUMNWISE (32 tokens per inter column) exactly like transpose_quantize_mxfp8. Byte-EXACT with
// transpose_quantize_mxfp8(x_phys): x_phys = float(fp8 rowwise q) * 2^(rowscale_e8-127) reproduces the backward's
// aq.cast(bf16)*_mx_block_scale(ae8).cast(bf16) (power-of-2 scale -> exact), the columnwise amax/e8/native-cvt logic
// is copied verbatim from transpose_quantize_mxfp8.cpp. The LDS tile reuses the mma-loop Bs buffers (dead in the
// epilogue) so PEAK LDS is unchanged (occupancy preserved). REG_EPILOGUE path only (the production epilogue).
// (FUSED_FC1_COLW / NATIVE_MXFP8_CVT defaults + short2v are declared at the top of the file, before the includes.)

using G = kittens::group<NUM_WARPS>;

#if FC1_PACK_A32
// The exact 4x2-warp GPT-OSS layout gives RT_A two 16-row tiles. MFMA consumes only scale bytes 0/1;
// omit the generic helper's two dead loads, which cross the scale tile for a_row_h1=224.
__device__ __forceinline__ fp8e8m0_4 pack_fc1_a_scales_32(const fp8e8m0 *smem_scales, int row_offset) {
    const int lid = laneid(), r16 = lid & 15, k_sub = lid >> 4;
    const fp8e8m0_4 *s4 = reinterpret_cast<const fp8e8m0_4 *>(smem_scales);
    const fp8e8m0_4 w0 = s4[row_offset + r16];
    const fp8e8m0_4 w1 = s4[row_offset + 16 + r16];
    const fp8e8m0_4 sel = 0x0C0C0000u | (k_sub << 8) | (4u + k_sub);
    return __builtin_amdgcn_perm(w0, w1, sel);
}
#define PACK_FC1_A_SCALES(PTR, OFF) pack_fc1_a_scales_32((PTR), (OFF))
#else
#define PACK_FC1_A_SCALES(PTR, OFF) pack_scales((PTR), (OFF))
#endif

// Reproduce tinygrad's bf16 '+' exactly: the reference computes h = (bf16 gate_up) + (bf16 bias) which upcasts
// both bf16 operands to fp32 (exact), adds in fp32, then materializes h back to bf16 via round-to-nearest-even
// (tinygrad's cast_float_to_bf16). fused_swiglu_quantize then reads that bf16 h back to fp32. This returns
// float(rne_bf16(a + b)) so the fused epilogue's SwiGLU input is bit-identical to the reference's.
__device__ __forceinline__ float sw_bf16_add_roundtrip(float a, float b) {
#if FC1_FAST_UNROUNDED
    return a + b;
#elif FC1_FAST_MATH
    bf16 h = base_types::convertor<bf16, float>::convert(a + b);
    return base_types::convertor<float, bf16>::convert(h);
#else
    float s = __fadd_rn(a, b);
    unsigned int x = __float_as_uint(s);
    unsigned int neg = 0u - x;                                  // tinygrad: (-x & 0x7f800000).ne(0)
    unsigned int r = ((neg & 0x7f800000u) != 0u) ? (x + ((x >> 16) & 1u) + 0x7fffu)
                                                 : (((x & 0xffffu) != 0u) ? (x | 0x10000u) : x);
    return __uint_as_float(r & 0xffff0000u);                    // (r >> 16) << 16, i.e. bf16 bits then back to fp32
#endif
}
__device__ __forceinline__ float sw_sigmoid(float g) {
#if FC1_FAST_RCP
    return __builtin_amdgcn_rcpf(1.0f + __builtin_amdgcn_exp2f(g * SW_NEG_ALPHA_LOG2E));
#elif FC1_FAST_MATH
    return 1.0f / (1.0f + __builtin_amdgcn_exp2f(g * SW_NEG_ALPHA_LOG2E));
#else
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, sw_exp2(__fmul_rn(g, SW_NEG_ALPHA_LOG2E))));
#endif
}
__device__ __forceinline__ float sw_act(float g, float sig, float l) {
#if FC1_FAST_MATH
    return (g * sig) * (l + 1.0f);
#else
    return __fmul_rn(__fmul_rn(g, sig), __fadd_rn(l, 1.0f));
#endif
}

#if FC1_COLW_PACKED
static __device__ __forceinline__ long long fc1_col_si(int cols,int col,int token_block) {
    return ((long long)(token_block/4)*cols+col)*4+token_block%4;
}
static __device__ __forceinline__ void fc1_col_si_zero8(uint8_t *ptr,int cols,int col,int token_block) {
    // Eight adjacent raw scale bytes become two separated packed words.
    FC1_STORE_ZERO(reinterpret_cast<uint32_t *>(ptr+fc1_col_si(cols,col,token_block)),uint32_t{0});
    FC1_STORE_ZERO(reinterpret_cast<uint32_t *>(ptr+fc1_col_si(cols,col,token_block+4)),uint32_t{0});
}
#define FC1_COLW_SCALE_INDEX(RAW, COL, BLOCK) fc1_col_si(OUT_INTER,COL,BLOCK)
#define FC1_COLW_SCALE_ZERO8(RAW, COL, BLOCK) fc1_col_si_zero8(Ycolw_e8_ptr,OUT_INTER,COL,BLOCK)
#else
// Preserve the original raw expressions, including their integer evaluation order.
#define FC1_COLW_SCALE_INDEX(RAW, COL, BLOCK) RAW
#define FC1_COLW_SCALE_ZERO8(RAW, COL, BLOCK) FC1_STORE_ZERO(reinterpret_cast<uint64_t *>(&Ycolw_e8_ptr[RAW]), uint64_t{0})
#endif

// Fused grouped mxfp8 FC1: computes gate_up = xg @ w_gate_up^T (M, N) exactly like the base grouped gemm, folds
// the per-expert bias (h = gate_up + bias, in bf16), then fuses the gpt-oss clamped SwiGLU + mxfp8-quantize into
// the epilogue via an on-chip LDS round-trip. Outputs the pre-swiglu h_bf16 (M, N) [saved for the backward, so the
// gemm is never re-run], y_fp8 (M, inter) fp8e4m3 and y_e8 (M, inter/32) e8m0. Bias is (E, N) bf16 (per-expert;
// every 256-row block is one expert); pad it with zeros over N-padding columns.
// min-blocks-per-CU launch-bounds hint; overridable (-DLB_MINBLK=N) to explore occupancy with the smaller tile.
#ifndef LB_MINBLK
#define LB_MINBLK 2
#endif
__global__ __launch_bounds__(512, LB_MINBLK)
void grouped_mxfp8_gemm_swiglu_kernel(bf16 *H_ptr, fp8e4m3 *Y_ptr, uint8_t *Ye8_ptr,
    const bf16 *__restrict__ Bias_ptr,
    fp8e4m3 *FC1_A_RESTRICT A_ptr, fp8e4m3 *FC1_B_RESTRICT B_ptr,
    fp8e8m0 *FC1_SCALE_A_RESTRICT scale_A_ptr, fp8e8m0 *FC1_SCALE_B_RESTRICT scale_B_ptr,
    const uint8_t *__restrict__ a_e8_unused,
    const uint8_t *__restrict__ b_e8_unused,
    const int *__restrict__ expert_off
#if FUSED_FC1_COLW
    , fp8e4m3 *Ycolw_ptr,        // (INTER, M) fp8e4m3 -- transposed/columnwise y (32-token blocks along M)
    uint8_t *Ycolw_e8_ptr        // raw (INTER, M/32) uint8, or packed (M/128, INTER) uint32
#endif
#if FC1_XCOL_ANCHOR
    , const fp8e4m3 *__restrict__ Xcol_anchor,
    const uint32_t *__restrict__ Xsi_anchor
#endif
#if FC1_SKIP_PAD_HALF_ROWS
    , const int *__restrict__ expert_counts
#endif
    ) {
    constexpr int M = GEMM_M, N = GEMM_N, K = GEMM_K, E = GEMM_E;
    constexpr int INTER           = N / 2;              // inter columns computed by the gate/up GEMM
    constexpr int OUT_INTER       = FC1_OUT_INTER;       // physical FC2 input width (may include a zero tail)
    constexpr int INTER_PER_BLOCK = BLOCK_COL / 2;      // 128 inter columns per 256-wide gate_up block
    constexpr int EBLK_PER_ROW    = OUT_INTER / 32;     // e8 blocks per physical output row
    static_assert(OUT_INTER >= INTER && OUT_INTER % 32 == 0);

    kittens::gl<fp8e4m3, 1, 1, M, K>     A{A_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<fp8e4m3, 1, 1, E * N, K> B{B_ptr, nullptr, nullptr, nullptr, nullptr};  // all experts stacked on rows

    // Exact GPT-OSS may carry the logical 2880-wide activation/weight in a 3072-wide zero-padded allocation.
    // Round the useful K extent up to an MFMA block, while retaining the physical scale strides below.
    constexpr int k_phys_iters = K / BLOCK_K;
    constexpr int k_iters      = (FC1_REAL_K + BLOCK_K - 1) / BLOCK_K;
    static_assert(FC1_REAL_K <= K && k_iters <= k_phys_iters);
    constexpr int NUM_THREADS  = NUM_WARPS * WARP_THREADS;

    kittens::gl<fp8e8m0, k_phys_iters * (M / SCALE_ROWS), 1, 16, 64>    scale_A_gl{scale_A_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<fp8e8m0, E * k_phys_iters * (N / BLOCK_COL), 1, 16, 64> scale_B_gl{scale_B_ptr, nullptr, nullptr, nullptr, nullptr};

    using ST_A     = st_fp8e4m3<HALF_ROW, BLOCK_K, st_16x128_s>;
    using ST_B     = st_fp8e4m3<HALF_COL, BLOCK_K, st_16x128_s>;
    using ST_A_Full = st_fp8e4m3<BLOCK_ROW, BLOCK_K, st_16x128_s>;
    using ST_B_Full = st_fp8e4m3<BLOCK_COL, BLOCK_K, st_16x128_s>;
    using ST_Scale = st<fp8e8m0, 16, 64, st_16x64_s>;
    using RT_A     = rt_fp8e4m3<REG_M, BLOCK_K>;
    using RT_B     = rt_fp8e4m3<REG_N, BLOCK_K>;
    using RT_C     = rt_fl<REG_M, REG_N, col_l, rt_16x16_s>;

#if FC1_FULL_LDS_TILES
    // The two row halves are physically contiguous and use the same 16x128 swizzle. Expose each complete A/B tile
    // as one LDS object so the global-to-LDS pipeline needs one address/SRD setup per operand instead of two.
    __shared__ ST_A_Full As[2];
    __shared__ ST_B_Full Bs[2];
    __shared__ ST_Scale scale_A_smem[2][SCALE_SUP], scale_B_smem[2];
#elif LOW_LDS
    // occupancy variant: single-buffered (index = row-half only, no double-buffer) -> 64KB tile LDS instead of 128KB,
    // so 2 workgroups co-reside per CU = 4 waves/SIMD (vs 2). Forces the register epilogue (the 128KB bf16 round-trip
    // scratch cannot fit). Trades DBUF's software pipeline (exposed load latency now hidden by the extra waves).
    __shared__ ST_A As[2];
    __shared__ ST_B Bs[2];
#else
    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];
    __shared__ ST_Scale scale_A_smem[2][SCALE_SUP], scale_B_smem[2];  // scale_A spans SCALE_SUP contiguous super-tiles per buffer
#endif
#if LOW_LDS
    __shared__ ST_Scale scale_A_smem[2][SCALE_SUP], scale_B_smem[2];
#endif

    RT_A a;
    RT_B b0, b1;
    RT_C cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

    constexpr int tiles_M  = M / BLOCK_ROW;
    constexpr int tiles_N  = N / BLOCK_COL;
    const int NUM_XCDS     = FC1_MAP_XCDS;
    #ifndef GROUPED_WGM
#define GROUPED_WGM 8
#endif
    const int WGM          = GROUPED_WGM;
    int wgid = chiplet_transform_chunked(blockIdx.x, gridDim.x, NUM_XCDS, FC1_MAP_CHUNK);
    int num_wgid_in_group = WGM * tiles_N;
    int group_id     = wgid / num_wgid_in_group;
    int first_pid_m  = group_id * WGM;
#if FC1_FIXED_WGM_MAP
    // GPT-OSS FC1 has 288 row tiles, exactly divisible by the selected WGM. Remove the unreachable partial-group path so
    // LLVM can replace its second dynamic div/mod pair with fixed-WGM operations.
    static_assert(tiles_M % GROUPED_WGM == 0, "fixed FC1 WGM mapping requires complete row groups");
    int rem          = wgid % num_wgid_in_group;
    int block_row    = first_pid_m + rem % WGM;
    int block_col    = rem / WGM;
#else
    int group_size_m = min(tiles_M - first_pid_m, WGM);
    int block_row    = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int block_col    = (wgid % num_wgid_in_group) / group_size_m;
#endif
    int block_m      = block_row * BLOCK_ROW;
    int block_n      = block_col * BLOCK_COL;
    constexpr int REAL_N = FC1_REAL_N;
    static_assert(REAL_N <= N && REAL_N % 2 == 0);
    static_assert(!FC1_HALF_N_TAIL || REAL_N % BLOCK_COL == HALF_COL);
    constexpr bool HAS_HALF_N_TAIL = FC1_HALF_N_TAIL && (REAL_N % BLOCK_COL == HALF_COL);
    const bool half_n_tail = HAS_HALF_N_TAIL && block_col == tiles_N - 1;
    // The gpt-oss FC1 math produces 2944 columns while FC2 consumes K=3072. Let the final N tile initialize that
    // local padding directly in the custom outputs, avoiding crop/pad compute kernels between FC1 and FC2. Rowwise
    // zero scales use e8=127, matching Tensor.pad(value=127); columnwise zero scales use 0, matching its old pad.
    #if FUSED_FC1_COLW && FC1_VECTOR_TAIL_ZERO
    #define ZERO_FC2_COLW_TAIL(PAD) {                                                                         \
        _Pragma("unroll 1") for (int i = threadIdx.x; i < (PAD) * (BLOCK_ROW / 16); i += NUM_THREADS) {      \
            const int col = i / (BLOCK_ROW / 16), vec = i % (BLOCK_ROW / 16);                                \
            FC1_STORE_ZERO(reinterpret_cast<uint4v *>(&Ycolw_ptr[(long long)(INTER + col) * M + block_m + vec * 16]), uint4v{}); \
        }                                                                                                     \
        _Pragma("unroll 1") for (int col = threadIdx.x; col < (PAD); col += NUM_THREADS)                     \
            FC1_COLW_SCALE_ZERO8((long long)(INTER + col) * (M / 32) + block_m / 32, INTER + col, block_m/32); \
    }
    #elif FUSED_FC1_COLW
    #define ZERO_FC2_COLW_TAIL(PAD) {                                                                         \
        _Pragma("unroll 1") for (int i = threadIdx.x; i < (PAD) * BLOCK_ROW; i += NUM_THREADS) {             \
            const int col = i / BLOCK_ROW, row = i % BLOCK_ROW;                                               \
            reinterpret_cast<uint8_t *>(Ycolw_ptr)[(long long)(INTER + col) * M + block_m + row] = 0;         \
        }                                                                                                     \
        _Pragma("unroll 1") for (int i = threadIdx.x; i < (PAD) * (BLOCK_ROW / 32); i += NUM_THREADS) {      \
            const int col = i / (BLOCK_ROW / 32), rb = i % (BLOCK_ROW / 32);                                  \
            Ycolw_e8_ptr[FC1_COLW_SCALE_INDEX((long long)(INTER + col) * (M / 32) + block_m / 32 + rb, INTER+col, block_m/32+rb)] = 0; \
        }                                                                                                     \
    }
    #else
    #define ZERO_FC2_COLW_TAIL(PAD) {}
    #endif
    #if FC1_VECTOR_TAIL_ZERO
    #define ZERO_FC2_TAIL() {                                                                                  \
        if constexpr (OUT_INTER > INTER) if (block_col == N / BLOCK_COL - 1) {                                \
            constexpr int PAD = OUT_INTER - INTER;                                                            \
            static_assert(PAD % 16 == 0 && PAD % 32 == 0);                                                    \
            _Pragma("unroll 1") for (int i = threadIdx.x; i < BLOCK_ROW * (PAD / 16); i += NUM_THREADS) {    \
                const int row = i / (PAD / 16), vec = i % (PAD / 16);                                         \
                FC1_STORE_ZERO(reinterpret_cast<uint4v *>(&Y_ptr[(block_m + row) * OUT_INTER + INTER + vec * 16]), uint4v{}); \
            }                                                                                                 \
            _Pragma("unroll 1") for (int row = threadIdx.x; row < BLOCK_ROW; row += NUM_THREADS)             \
                FC1_STORE_ZERO(reinterpret_cast<uint32_t *>(&Ye8_ptr[(block_m + row) * EBLK_PER_ROW + INTER / 32]), uint32_t{0x7f7f7f7fu}); \
            ZERO_FC2_COLW_TAIL(PAD);                                                                          \
        }                                                                                                     \
    }
    #else
    #define ZERO_FC2_TAIL() {                                                                                  \
        if constexpr (OUT_INTER > INTER) if (block_col == N / BLOCK_COL - 1) {                                \
            constexpr int PAD = OUT_INTER - INTER;                                                            \
            _Pragma("unroll 1") for (int i = threadIdx.x; i < BLOCK_ROW * PAD; i += NUM_THREADS) {           \
                const int row = i / PAD, col = i % PAD;                                                       \
                reinterpret_cast<uint8_t *>(Y_ptr)[(block_m + row) * OUT_INTER + INTER + col] = 0;           \
            }                                                                                                 \
            _Pragma("unroll 1") for (int i = threadIdx.x; i < BLOCK_ROW * (PAD / 32); i += NUM_THREADS) {    \
                const int row = i / (PAD / 32), col = i % (PAD / 32);                                         \
                Ye8_ptr[(block_m + row) * EBLK_PER_ROW + INTER / 32 + col] = 127;                             \
            }                                                                                                 \
            ZERO_FC2_COLW_TAIL(PAD);                                                                          \
        }                                                                                                     \
    }
    #endif
#if MOE_SKIP_EMPTY
    if (block_m >= __builtin_amdgcn_readfirstlane(expert_off[E])) {
#if FC1_SKIP_EMPTY_MAIN_ZERO
        ZERO_FC2_TAIL();
        return;
#else
#if !MOE_SKIP_NO_ZERO
        const int tid = threadIdx.x;
#if FC1_VECTOR_EMPTY_ZERO
        #pragma unroll 1
        for (int i = tid; i < BLOCK_ROW * (BLOCK_COL / 8); i += NUM_THREADS) {
            const int row = i / (BLOCK_COL / 8), vec = i % (BLOCK_COL / 8);
            FC1_STORE_ZERO(reinterpret_cast<uint4v *>(&H_ptr[(block_m + row) * N + block_n + vec * 8]), uint4v{});
        }
        #pragma unroll 1
        for (int i = tid; i < BLOCK_ROW * (INTER_PER_BLOCK / 16); i += NUM_THREADS) {
            const int row = i / (INTER_PER_BLOCK / 16), vec = i % (INTER_PER_BLOCK / 16);
            FC1_STORE_ZERO(reinterpret_cast<uint4v *>(&Y_ptr[(block_m + row) * OUT_INTER + block_col * INTER_PER_BLOCK + vec * 16]), uint4v{});
        }
        #pragma unroll 1
        for (int row = tid; row < BLOCK_ROW; row += NUM_THREADS)
            FC1_STORE_ZERO(reinterpret_cast<uint32_t *>(&Ye8_ptr[(block_m + row) * EBLK_PER_ROW + block_col * (INTER_PER_BLOCK / 32)]), uint32_t{0});
#if FUSED_FC1_COLW
        #pragma unroll 1
        for (int i = tid; i < INTER_PER_BLOCK * (BLOCK_ROW / 16); i += NUM_THREADS) {
            const int col = i / (BLOCK_ROW / 16), vec = i % (BLOCK_ROW / 16);
            FC1_STORE_ZERO(reinterpret_cast<uint4v *>(&Ycolw_ptr[(long long)(block_col * INTER_PER_BLOCK + col) * M + block_m + vec * 16]), uint4v{});
        }
        #pragma unroll 1
        for (int col = tid; col < INTER_PER_BLOCK; col += NUM_THREADS)
            FC1_COLW_SCALE_ZERO8((long long)(block_col * INTER_PER_BLOCK + col) * (M / 32) + block_m / 32,
                                block_col * INTER_PER_BLOCK + col, block_m/32);
#endif
#else
        #pragma unroll 1
        for (int i = tid; i < BLOCK_ROW * BLOCK_COL; i += NUM_THREADS) {
            const int row = i / BLOCK_COL, col = i % BLOCK_COL;
            reinterpret_cast<uint16_t *>(H_ptr)[(block_m + row) * N + block_n + col] = 0;
        }
        #pragma unroll 1
        for (int i = tid; i < BLOCK_ROW * INTER_PER_BLOCK; i += NUM_THREADS) {
            const int row = i / INTER_PER_BLOCK, col = i % INTER_PER_BLOCK;
            reinterpret_cast<uint8_t *>(Y_ptr)[(block_m + row) * OUT_INTER + block_col * INTER_PER_BLOCK + col] = 0;
        }
        #pragma unroll 1
        for (int i = tid; i < BLOCK_ROW * (INTER_PER_BLOCK / 32); i += NUM_THREADS) {
            const int row = i / (INTER_PER_BLOCK / 32), col = i % (INTER_PER_BLOCK / 32);
            Ye8_ptr[(block_m + row) * EBLK_PER_ROW + block_col * (INTER_PER_BLOCK / 32) + col] = 0;
        }
#if FUSED_FC1_COLW
        #pragma unroll 1
        for (int i = tid; i < INTER_PER_BLOCK * BLOCK_ROW; i += NUM_THREADS) {
            const int col = i / BLOCK_ROW, row = i % BLOCK_ROW;
            reinterpret_cast<uint8_t *>(Ycolw_ptr)[(block_col * INTER_PER_BLOCK + col) * M + block_m + row] = 0;
        }
        #pragma unroll 1
        for (int i = tid; i < INTER_PER_BLOCK * (BLOCK_ROW / 32); i += NUM_THREADS) {
            const int col = i / (BLOCK_ROW / 32), row_block = i % (BLOCK_ROW / 32);
            Ycolw_e8_ptr[FC1_COLW_SCALE_INDEX((block_col * INTER_PER_BLOCK + col) * (M / 32) + block_m / 32 + row_block,
                                            block_col*INTER_PER_BLOCK+col, block_m/32+row_block)] = 0;
        }
#endif
#endif
#endif
        ZERO_FC2_TAIL();
        return;
#endif
    }
#endif

    int e = 0;
#if FC1_BINARY_EXPERT
    static_assert(E == 32, "binary expert lookup is specialized for GPT-OSS's 32 experts");
    // expert_off is monotonic and every workgroup belongs to exactly one expert. Five uniform upper-bound probes
    // replace the 31 offset loads/comparisons in the generic linear scan.
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
#if FC1_SKIP_PAD_HALF_ROWS
    static_assert(E == 32 && BLOCK_ROW == 256 && HALF_ROW == 128,
                  "padded-half-row skip is specialized for GPT-OSS FC1");
    const int active_count = __builtin_amdgcn_readfirstlane(expert_counts[e]);
    const int expert_start = __builtin_amdgcn_readfirstlane(expert_off[e]);
  #if FC1_OMIT_PAD_LOWER_OUTPUTS
    const bool omit_lower_outputs = block_m + HALF_ROW >= expert_start + active_count;
  #else
    constexpr bool omit_lower_outputs = false;
  #endif
  #if FC1_PAD_LOWER_REUSE
    #if FC1_PAD_UPPER_SKIP
    const bool skip_upper_mma = !half_n_tail &&
                                block_m + (warpid() / WARPS_COL) * REG_M >= expert_start + active_count;
    #else
    constexpr bool skip_upper_mma = false;
    #endif
    const bool skip_lower_mma = block_m + HALF_ROW + (warpid() / WARPS_COL) * REG_M >= expert_start + active_count;
  #else
    constexpr bool skip_upper_mma = false;
    const bool skip_lower_mma = block_m + HALF_ROW >= expert_start + active_count;
  #endif
#else
    constexpr bool skip_upper_mma = false;
    constexpr bool skip_lower_mma = false;
    constexpr bool omit_lower_outputs = false;
#endif
    const int bcol_base = e * (N / HALF_COL);          // expert base in B row-tile (128-row) units
    const int sb_base   = e * (k_phys_iters * tiles_N); // expert base into physically 3072-wide scale_B batches

    int warp_m = warpid() / WARPS_COL;
    int warp_n = warpid() % WARPS_COL;

    using T = fp8e4m3;
    constexpr int bpt      = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm      = bpt * NUM_THREADS;
#if FC1_FULL_LDS_TILES
    constexpr int copies_A = BLOCK_ROW * BLOCK_K * sizeof(T) / bpm;
    constexpr int copies_B = BLOCK_COL * BLOCK_K * sizeof(T) / bpm;
#else
    constexpr int copies_A = HALF_ROW * BLOCK_K * sizeof(T) / bpm;
    constexpr int copies_B = HALF_COL * BLOCK_K * sizeof(T) / bpm;
#endif
    uint32_t sw_A[copies_A], sw_B[copies_B];
#if FC1_FULL_LDS_TILES
    G::prefill_swizzled_offsets(As[0], A, sw_A);
    G::prefill_swizzled_offsets(Bs[0], B, sw_B);
#elif LOW_LDS
    G::prefill_swizzled_offsets(As[0], A, sw_A);
    G::prefill_swizzled_offsets(Bs[0], B, sw_B);
#else
    G::prefill_swizzled_offsets(As[0][0], A, sw_A);
    G::prefill_swizzled_offsets(Bs[0][0], B, sw_B);
#endif

    const T *a_base = (const T *)&A[{0, 0, 0, 0}];
    const T *b_base = (const T *)&B[{0, 0, 0, 0}];
    const int a_row_stride = A.template stride<2>() * sizeof(T);
    const int b_row_stride = B.template stride<2>() * sizeof(T);
    i32x4 a_srd = make_srsrc(a_base, (uint32_t)((uint64_t)M * a_row_stride), a_row_stride);
    i32x4 b_srd = make_srsrc(b_base, (uint32_t)((uint64_t)E * N * b_row_stride), b_row_stride);
#if FC1_PRECOMPUTED_SOFF
    static_assert(NUM_THREADS == 512 && sizeof(T) == 1,
                  "direct FC1 byte offsets require the exact eight-wave FP8 specialization");
    uint32_t a_soff_base = __builtin_amdgcn_readfirstlane((uint32_t)(block_m * K));
    uint32_t b_soff_base = __builtin_amdgcn_readfirstlane((uint32_t)((e * N + block_n) * K));
    asm volatile("" : "+s"(a_soff_base), "+s"(b_soff_base));
    #define FC1_LOAD_A(P, KK) \
        fc1_load_soff(As[P], sw_A, a_srd, a_soff_base + (uint32_t)(KK) * BLOCK_K, (P) ? a_lds_1 : a_lds_0)
    #define FC1_LOAD_B(P, KK) \
        fc1_load_soff(Bs[P], sw_B, b_srd, b_soff_base + (uint32_t)(KK) * BLOCK_K, (P) ? b_lds_1 : b_lds_0)
    #define FC1_LOAD_B_TAIL(P, KK, TILE) \
        fc1_load_soff(TILE, sw_B, b_srd, b_soff_base + (uint32_t)(KK) * BLOCK_K, (P) ? b_lds_1 : b_lds_0)
#else
    #define FC1_LOAD_A(P, KK) \
        G::load(As[P], A, {0, 0, block_row, (KK)}, sw_A, a_srd, a_base, (P) ? a_lds_1 : a_lds_0)
    #define FC1_LOAD_B(P, KK) \
        G::load(Bs[P], B, {0, 0, bfull_base + block_col, (KK)}, sw_B, b_srd, b_base, (P) ? b_lds_1 : b_lds_0)
    #define FC1_LOAD_B_TAIL(P, KK, TILE) \
        G::load(TILE, B, {0, 0, bcol_base + block_col * 2, (KK)}, sw_B, b_srd, b_base, (P) ? b_lds_1 : b_lds_0)
#endif

    const int wid = warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(T)) * kittens::WARP_THREADS;
    // scale_A super-tiles are SCALE_ROWS rows each; this block starts at super-tile st_start and its row 0 sits at
    // flat offset srow_a within the loaded (SCALE_SUP-tile contiguous) scale region. 256: st=block_row,srow=0.
    // 128: st=block_row/2, srow=(block_row&1)*128 (half of one super-tile). 512: st=block_row*2, srow=0 (spans 2).
    const int st_start = (block_row * BLOCK_ROW) / SCALE_ROWS;
#if SCALE_PACK
    const int srow_a = 0;                                    // SCALE_PACK writes exactly BLOCK_ROW rows at s4A[0..)
#else
    const int srow_a = (block_row * BLOCK_ROW) % SCALE_ROWS;
#endif
    int a_row_h0 = srow_a + warp_m * REG_M;
    int a_row_h1 = srow_a + HALF_ROW + warp_m * REG_M;
    int b_row_h0 = warp_n * REG_N;
    int b_row_h1 = HALF_COL + warp_n * REG_N;

#if FC1_FULL_LDS_TILES
    uint32_t a_lds_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t a_lds_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1].data[0]) + wid * elem_per_warp * sizeof(T)));
#elif LOW_LDS
    uint32_t a_lds_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t a_lds_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t a_lds[2] = {a_lds_0, a_lds_1};
    uint32_t b_lds[2] = {b_lds_0, b_lds_1};
#else
    uint32_t a_lds_00 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t a_lds_01 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0][1].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t a_lds_10 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t a_lds_11 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1][1].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_00 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_01 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0][1].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_10 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_11 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1][1].data[0]) + wid * elem_per_warp * sizeof(T)));
#if !FC1_SCALAR_LDS_BASES
    uint32_t a_lds[2][2] = {{a_lds_00, a_lds_01}, {a_lds_10, a_lds_11}};
    uint32_t b_lds[2][2] = {{b_lds_00, b_lds_01}, {b_lds_10, b_lds_11}};
#endif
#endif

#if SCALE_PACK
    // Read the raw e8 scales (row-major, ALREADY kernel inputs a_e8_unused/b_e8_unused) straight into the scale LDS
    // tiles, eliminating the separate mx_pack/mx_pack_3d transpose copy kernels. s4[row] = uint32(e8[base+row][kk*4:+4])
    // is byte-identical to the pre-packed (scale-major) x_si/w_si tile that G::load produced, and matches pack_scales'
    // linear s4[row_offset + j*16 + r16] read. (K/32 % 4 == 0 => each uint32 read is 4B-aligned.)
    #define LOAD_SCALES(BUF, KK) {                                                                                     \
        constexpr int KU32 = K / 128;                    /* uint32 (= 4 e8-blocks) per row */                         \
        uint32_t *s4A = reinterpret_cast<uint32_t *>(&scale_A_smem[BUF][0].data[0]);                                  \
        uint32_t *s4B = reinterpret_cast<uint32_t *>(&scale_B_smem[BUF].data[0]);                                     \
        const uint32_t *xe_a = reinterpret_cast<const uint32_t *>(a_e8_unused);                                       \
        const uint32_t *xe_b = reinterpret_cast<const uint32_t *>(b_e8_unused);                                       \
        for (int rr = threadIdx.x; rr < BLOCK_ROW; rr += NUM_THREADS)                                                 \
            s4A[rr] = xe_a[(block_row * BLOCK_ROW + rr) * KU32 + (KK)];                                               \
        for (int cc = threadIdx.x; cc < BLOCK_COL; cc += NUM_THREADS)                                                 \
            s4B[cc] = xe_b[(e * N + block_col * BLOCK_COL + cc) * KU32 + (KK)];                                       \
    }
#else
    #define LOAD_SCALES(BUF, KK)                                                                                       \
        _Pragma("unroll") for (int su = 0; su < SCALE_SUP; su++)                                                       \
            G::load(scale_A_smem[BUF][su], scale_A_gl, {(KK) * (M / SCALE_ROWS) + st_start + su, 0, 0, 0});           \
        G::load(scale_B_smem[BUF], scale_B_gl, {sb_base + (KK) * tiles_N + block_col, 0, 0, 0});
#endif

#if LOW_LDS
    // single-buffered K-loop (index = row-half); relies on the extra occupancy (4 waves/SIMD) to hide load latency.
    #pragma unroll 1
    for (int kk = 0; kk < k_iters; kk++) {
        G::load(As[0], A, {0, 0, block_row * 2,     kk}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[0]));
        G::load(As[1], A, {0, 0, block_row * 2 + 1, kk}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[1]));
        G::load(Bs[0], B, {0, 0, bcol_base + block_col * 2,     kk}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[0]));
        G::load(Bs[1], B, {0, 0, bcol_base + block_col * 2 + 1, kk}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[1]));
        LOAD_SCALES(0, kk);
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        fp8e8m0_4 sa_h0 = PACK_FC1_A_SCALES(scale_A_smem[0][0].data, a_row_h0);
        fp8e8m0_4 sa_h1 = PACK_FC1_A_SCALES(scale_A_smem[0][0].data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem[0].data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem[0].data, b_row_h1);

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
#elif DOUBLE_BUFFER
    // Software-pipelined double-buffered K-loop: prefetch iter kk+1's A/B/scales into the OTHER LDS buffer while the
    // matrix core consumes iter kk, so the exposed HBM load latency (which the single-buffered loop waits on every
    // iter before any MMA) is hidden under the MMAs. One barrier per iter -- it makes the buffer we're about to READ
    // next iter visible; the WAR on the buffer two iters back is covered by that same barrier plus the pre-MMA
    // lgkmcnt(0) that drains this iter's ds_reads. P2 folded in: two RT_A regs (a0,a1) keep BOTH A operand ds_reads
    // in flight so all 4 MMAs issue back-to-back under a single lgkmcnt(0) (kills the mid-loop WAR reload of 'a').
    // Byte-identical to the single-buffered path: same operands, same MMA order and accumulation.
#if FC1_INTERIOR_FAST
    static_assert(FC1_FULL_LDS_TILES && FC1_FULL_LDS_TAIL_HALF && FC1_PEEL_FINAL,
                  "interior-fast loop requires the exact GPT-OSS tiled/peeled configuration");
#if FC1_PAD_UPPER_SKIP
    static_assert(FC1_SKIP_PAD_HALF_ROWS && FC1_PAD_LOWER_REUSE && FC1_WARP_SCALE32,
                  "upper-band skip requires exact GPT-OSS counts and the 32-row warp padding path");
#endif
    if (!half_n_tail) {
#if FC1_PAD_UPPER_SKIP
    auto fc1_interior_contraction = [&]<bool PAD_UPPER_SKIP>() __attribute__((always_inline)) {
    #define FC1_LOAD_UPPER_A(CUR)                                                                                       \
        if (!PAD_UPPER_SKIP || !skip_upper_mma) {                                                                       \
            auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[CUR], {warp_m, 0}); load(a0, as0); }
    #define FC1_MMA_UPPER()                                                                                             \
        if (!PAD_UPPER_SKIP || !skip_upper_mma) {                                                                       \
            mma_ABt_scaled(cA, a0, b0, cA, &sa_h0, &sb_h0);                                                            \
            mma_ABt_scaled(cB, a0, b1, cB, &sa_h0, &sb_h1); }
#else
    #define FC1_LOAD_UPPER_A(CUR)                                                                                       \
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[CUR], {warp_m, 0}); load(a0, as0)
    #define FC1_MMA_UPPER()                                                                                             \
        mma_ABt_scaled(cA, a0, b0, cA, &sa_h0, &sb_h0);                                                                \
        mma_ABt_scaled(cB, a0, b1, cB, &sa_h0, &sb_h1)
#endif
#if FC1_FULL_LDS_TILES
    const int bfull_base = e * (N / BLOCK_COL);
#if FC1_FULL_LDS_TAIL_HALF
    #define LOAD_ITER(P, KK) {                                                                                   \
        FC1_LOAD_A(P, KK);                                                                                      \
        if (false) {                                                                                       \
            auto btail = subtile_inplace<HALF_COL, BLOCK_K>(Bs[P], {0, 0});                                     \
            FC1_LOAD_B_TAIL(P, KK, btail);                                                                       \
        } else FC1_LOAD_B(P, KK);                                                                                \
        LOAD_SCALES(P, (KK)) }
#else
    #define LOAD_ITER(P, KK)                                                                                     \
        FC1_LOAD_A(P, KK);                                                                                      \
        FC1_LOAD_B(P, KK);                                                                                     \
        LOAD_SCALES(P, (KK))
#endif
#elif FC1_SCALAR_LDS_BASES
    #define LOAD_ITER(P, KK)                                                                                               \
        G::load(As[P][0], A, {0, 0, block_row * 2,     (KK)}, sw_A, a_srd, a_base, (P) ? a_lds_10 : a_lds_00);   \
        G::load(As[P][1], A, {0, 0, block_row * 2 + 1, (KK)}, sw_A, a_srd, a_base, (P) ? a_lds_11 : a_lds_01);   \
        G::load(Bs[P][0], B, {0, 0, bcol_base + block_col * 2,     (KK)}, sw_B, b_srd, b_base, (P) ? b_lds_10 : b_lds_00); \
        if (!false) G::load(Bs[P][1], B, {0, 0, bcol_base + block_col * 2 + 1, (KK)}, sw_B, b_srd, b_base, (P) ? b_lds_11 : b_lds_01); \
        LOAD_SCALES(P, (KK))
#else
    #define LOAD_ITER(P, KK)                                                                                               \
        G::load(As[P][0], A, {0, 0, block_row * 2,     (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[P][0]));   \
        G::load(As[P][1], A, {0, 0, block_row * 2 + 1, (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[P][1]));   \
        G::load(Bs[P][0], B, {0, 0, bcol_base + block_col * 2,     (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[P][0])); \
        if (!false) G::load(Bs[P][1], B, {0, 0, bcol_base + block_col * 2 + 1, (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[P][1])); \
        LOAD_SCALES(P, (KK))
#endif
    RT_A a0, a1;
    LOAD_ITER(0, 0);
    asm volatile("s_waitcnt vmcnt(0)");
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
#if FC1_PEEL_FINAL
    // Exact GPT-OSS has 23 contraction tiles. Peel the final tile so the two
    // has-next predicates disappear from all 22 steady-state iterations.
    static_assert(k_iters == 23, "peeled FC1 loop requires GPT-OSS real K");
    #pragma unroll 1
    for (int kk = 0; kk < k_iters - 1; kk++) {
        const int cur = kk & 1;
        const int nxt = (kk + 1) & 1;
        LOAD_ITER(nxt, kk + 1);

        fp8e8m0_4 sa_h0 = PACK_FC1_A_SCALES(scale_A_smem[cur][0].data, a_row_h0);
        fp8e8m0_4 sa_h1 = PACK_FC1_A_SCALES(scale_A_smem[cur][0].data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem[cur].data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem[cur].data, b_row_h1);

#if FC1_FULL_LDS_TILES
        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur], {warp_n, 0}); load(b0, bs0);
        if (!false) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur], {WARPS_COL + warp_n, 0}); load(b1, bs1); }
        FC1_LOAD_UPPER_A(cur);
        if (!skip_lower_mma) { auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur], {WARPS_ROW + warp_m, 0}); load(a1, as1); }
#else
        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][0], {warp_n, 0}); load(b0, bs0);
        if (!false) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][1], {warp_n, 0}); load(b1, bs1); }
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur][0], {warp_m, 0}); load(a0, as0);
        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur][1], {warp_m, 0}); load(a1, as1);
#endif
        asm volatile("s_waitcnt lgkmcnt(0)");
        FC1_MMA_UPPER();
        if (!skip_lower_mma) {
            mma_ABt_scaled(cC, a1, b0, cC, &sa_h1, &sb_h0);
            if (!false) mma_ABt_scaled(cD, a1, b1, cD, &sa_h1, &sb_h1);
        }

        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
    {
        constexpr int cur = 0;
        fp8e8m0_4 sa_h0 = PACK_FC1_A_SCALES(scale_A_smem[cur][0].data, a_row_h0);
        fp8e8m0_4 sa_h1 = PACK_FC1_A_SCALES(scale_A_smem[cur][0].data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem[cur].data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem[cur].data, b_row_h1);

#if FC1_FULL_LDS_TILES
        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur], {warp_n, 0}); load(b0, bs0);
        if (!false) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur], {WARPS_COL + warp_n, 0}); load(b1, bs1); }
        FC1_LOAD_UPPER_A(cur);
        if (!skip_lower_mma) { auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur], {WARPS_ROW + warp_m, 0}); load(a1, as1); }
#else
        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][0], {warp_n, 0}); load(b0, bs0);
        if (!false) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][1], {warp_n, 0}); load(b1, bs1); }
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur][0], {warp_m, 0}); load(a0, as0);
        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur][1], {warp_m, 0}); load(a1, as1);
#endif
        asm volatile("s_waitcnt lgkmcnt(0)");
        FC1_MMA_UPPER();
        if (!skip_lower_mma) {
            mma_ABt_scaled(cC, a1, b0, cC, &sa_h1, &sb_h0);
            if (!false) mma_ABt_scaled(cD, a1, b1, cD, &sa_h1, &sb_h1);
        }
    }
#else
    #pragma unroll 1
    for (int kk = 0; kk < k_iters; kk++) {
        const int cur = kk & 1;
        const int nxt = (kk + 1) & 1;
        if (kk + 1 < k_iters) { LOAD_ITER(nxt, kk + 1); }

        fp8e8m0_4 sa_h0 = PACK_FC1_A_SCALES(scale_A_smem[cur][0].data, a_row_h0);
        fp8e8m0_4 sa_h1 = PACK_FC1_A_SCALES(scale_A_smem[cur][0].data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem[cur].data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem[cur].data, b_row_h1);

#if FC1_FULL_LDS_TILES
        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur], {warp_n, 0}); load(b0, bs0);
        if (!false) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur], {WARPS_COL + warp_n, 0}); load(b1, bs1); }
        FC1_LOAD_UPPER_A(cur);
        if (!skip_lower_mma) { auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur], {WARPS_ROW + warp_m, 0}); load(a1, as1); }
#else
        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][0], {warp_n, 0}); load(b0, bs0);
        if (!false) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][1], {warp_n, 0}); load(b1, bs1); }
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur][0], {warp_m, 0}); load(a0, as0);
        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur][1], {warp_m, 0}); load(a1, as1);
#endif
        asm volatile("s_waitcnt lgkmcnt(0)");
        FC1_MMA_UPPER();
        if (!skip_lower_mma) {
            mma_ABt_scaled(cC, a1, b0, cC, &sa_h1, &sb_h0);
            if (!false) mma_ABt_scaled(cD, a1, b1, cD, &sa_h1, &sb_h1);
        }

        if (kk + 1 < k_iters) { asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier(); }
    }
#endif
    #undef LOAD_ITER
#undef FC1_LOAD_UPPER_A
#undef FC1_MMA_UPPER
#if FC1_PAD_UPPER_SKIP
    };
    if (block_m + HALF_ROW <= expert_start + active_count)
        fc1_interior_contraction.template operator()<false>();
    else
        fc1_interior_contraction.template operator()<true>();
#endif
    } else {
#if FC1_FULL_LDS_TILES
    const int bfull_base = e * (N / BLOCK_COL);
#if FC1_FULL_LDS_TAIL_HALF
    #define LOAD_ITER(P, KK) {                                                                                   \
        FC1_LOAD_A(P, KK);                                                                                      \
        if (true) {                                                                                       \
            auto btail = subtile_inplace<HALF_COL, BLOCK_K>(Bs[P], {0, 0});                                     \
            FC1_LOAD_B_TAIL(P, KK, btail);                                                                       \
        } else FC1_LOAD_B(P, KK);                                                                                \
        LOAD_SCALES(P, (KK)) }
#else
    #define LOAD_ITER(P, KK)                                                                                     \
        FC1_LOAD_A(P, KK);                                                                                      \
        FC1_LOAD_B(P, KK);                                                                                     \
        LOAD_SCALES(P, (KK))
#endif
#elif FC1_SCALAR_LDS_BASES
    #define LOAD_ITER(P, KK)                                                                                               \
        G::load(As[P][0], A, {0, 0, block_row * 2,     (KK)}, sw_A, a_srd, a_base, (P) ? a_lds_10 : a_lds_00);   \
        G::load(As[P][1], A, {0, 0, block_row * 2 + 1, (KK)}, sw_A, a_srd, a_base, (P) ? a_lds_11 : a_lds_01);   \
        G::load(Bs[P][0], B, {0, 0, bcol_base + block_col * 2,     (KK)}, sw_B, b_srd, b_base, (P) ? b_lds_10 : b_lds_00); \
        if (!true) G::load(Bs[P][1], B, {0, 0, bcol_base + block_col * 2 + 1, (KK)}, sw_B, b_srd, b_base, (P) ? b_lds_11 : b_lds_01); \
        LOAD_SCALES(P, (KK))
#else
    #define LOAD_ITER(P, KK)                                                                                               \
        G::load(As[P][0], A, {0, 0, block_row * 2,     (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[P][0]));   \
        G::load(As[P][1], A, {0, 0, block_row * 2 + 1, (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[P][1]));   \
        G::load(Bs[P][0], B, {0, 0, bcol_base + block_col * 2,     (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[P][0])); \
        if (!true) G::load(Bs[P][1], B, {0, 0, bcol_base + block_col * 2 + 1, (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[P][1])); \
        LOAD_SCALES(P, (KK))
#endif
    RT_A a0, a1;
    LOAD_ITER(0, 0);
    asm volatile("s_waitcnt vmcnt(0)");
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
#if FC1_PEEL_FINAL
    // Exact GPT-OSS has 23 contraction tiles. Peel the final tile so the two
    // has-next predicates disappear from all 22 steady-state iterations.
    static_assert(k_iters == 23, "peeled FC1 loop requires GPT-OSS real K");
    #pragma unroll 1
    for (int kk = 0; kk < k_iters - 1; kk++) {
        const int cur = kk & 1;
        const int nxt = (kk + 1) & 1;
        LOAD_ITER(nxt, kk + 1);

        fp8e8m0_4 sa_h0 = PACK_FC1_A_SCALES(scale_A_smem[cur][0].data, a_row_h0);
        fp8e8m0_4 sa_h1 = PACK_FC1_A_SCALES(scale_A_smem[cur][0].data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem[cur].data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem[cur].data, b_row_h1);

#if FC1_FULL_LDS_TILES
        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur], {warp_n, 0}); load(b0, bs0);
        if (!true) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur], {WARPS_COL + warp_n, 0}); load(b1, bs1); }
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur], {warp_m, 0}); load(a0, as0);
        if (!skip_lower_mma) { auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur], {WARPS_ROW + warp_m, 0}); load(a1, as1); }
#else
        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][0], {warp_n, 0}); load(b0, bs0);
        if (!true) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][1], {warp_n, 0}); load(b1, bs1); }
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur][0], {warp_m, 0}); load(a0, as0);
        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur][1], {warp_m, 0}); load(a1, as1);
#endif
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cA, a0, b0, cA, &sa_h0, &sb_h0);
        if (!true) mma_ABt_scaled(cB, a0, b1, cB, &sa_h0, &sb_h1);
        if (!skip_lower_mma) {
            mma_ABt_scaled(cC, a1, b0, cC, &sa_h1, &sb_h0);
            if (!true) mma_ABt_scaled(cD, a1, b1, cD, &sa_h1, &sb_h1);
        }

        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
    {
        constexpr int cur = 0;
        fp8e8m0_4 sa_h0 = PACK_FC1_A_SCALES(scale_A_smem[cur][0].data, a_row_h0);
        fp8e8m0_4 sa_h1 = PACK_FC1_A_SCALES(scale_A_smem[cur][0].data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem[cur].data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem[cur].data, b_row_h1);

#if FC1_FULL_LDS_TILES
        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur], {warp_n, 0}); load(b0, bs0);
        if (!true) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur], {WARPS_COL + warp_n, 0}); load(b1, bs1); }
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur], {warp_m, 0}); load(a0, as0);
        if (!skip_lower_mma) { auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur], {WARPS_ROW + warp_m, 0}); load(a1, as1); }
#else
        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][0], {warp_n, 0}); load(b0, bs0);
        if (!true) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][1], {warp_n, 0}); load(b1, bs1); }
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur][0], {warp_m, 0}); load(a0, as0);
        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur][1], {warp_m, 0}); load(a1, as1);
#endif
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cA, a0, b0, cA, &sa_h0, &sb_h0);
        if (!true) mma_ABt_scaled(cB, a0, b1, cB, &sa_h0, &sb_h1);
        if (!skip_lower_mma) {
            mma_ABt_scaled(cC, a1, b0, cC, &sa_h1, &sb_h0);
            if (!true) mma_ABt_scaled(cD, a1, b1, cD, &sa_h1, &sb_h1);
        }
    }
#else
    #pragma unroll 1
    for (int kk = 0; kk < k_iters; kk++) {
        const int cur = kk & 1;
        const int nxt = (kk + 1) & 1;
        if (kk + 1 < k_iters) { LOAD_ITER(nxt, kk + 1); }

        fp8e8m0_4 sa_h0 = PACK_FC1_A_SCALES(scale_A_smem[cur][0].data, a_row_h0);
        fp8e8m0_4 sa_h1 = PACK_FC1_A_SCALES(scale_A_smem[cur][0].data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem[cur].data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem[cur].data, b_row_h1);

#if FC1_FULL_LDS_TILES
        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur], {warp_n, 0}); load(b0, bs0);
        if (!true) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur], {WARPS_COL + warp_n, 0}); load(b1, bs1); }
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur], {warp_m, 0}); load(a0, as0);
        if (!skip_lower_mma) { auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur], {WARPS_ROW + warp_m, 0}); load(a1, as1); }
#else
        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][0], {warp_n, 0}); load(b0, bs0);
        if (!true) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][1], {warp_n, 0}); load(b1, bs1); }
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur][0], {warp_m, 0}); load(a0, as0);
        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur][1], {warp_m, 0}); load(a1, as1);
#endif
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cA, a0, b0, cA, &sa_h0, &sb_h0);
        if (!true) mma_ABt_scaled(cB, a0, b1, cB, &sa_h0, &sb_h1);
        if (!skip_lower_mma) {
            mma_ABt_scaled(cC, a1, b0, cC, &sa_h1, &sb_h0);
            if (!true) mma_ABt_scaled(cD, a1, b1, cD, &sa_h1, &sb_h1);
        }

        if (kk + 1 < k_iters) { asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier(); }
    }
#endif
    #undef LOAD_ITER
    }
#else
#if FC1_FULL_LDS_TILES
    const int bfull_base = e * (N / BLOCK_COL);
#if FC1_FULL_LDS_TAIL_HALF
    #define LOAD_ITER(P, KK) {                                                                                   \
        FC1_LOAD_A(P, KK);                                                                                      \
        if (half_n_tail) {                                                                                       \
            auto btail = subtile_inplace<HALF_COL, BLOCK_K>(Bs[P], {0, 0});                                     \
            FC1_LOAD_B_TAIL(P, KK, btail);                                                                       \
        } else FC1_LOAD_B(P, KK);                                                                                \
        LOAD_SCALES(P, (KK)) }
#else
    #define LOAD_ITER(P, KK)                                                                                     \
        FC1_LOAD_A(P, KK);                                                                                      \
        FC1_LOAD_B(P, KK);                                                                                     \
        LOAD_SCALES(P, (KK))
#endif
#elif FC1_SCALAR_LDS_BASES
    #define LOAD_ITER(P, KK)                                                                                               \
        G::load(As[P][0], A, {0, 0, block_row * 2,     (KK)}, sw_A, a_srd, a_base, (P) ? a_lds_10 : a_lds_00);   \
        G::load(As[P][1], A, {0, 0, block_row * 2 + 1, (KK)}, sw_A, a_srd, a_base, (P) ? a_lds_11 : a_lds_01);   \
        G::load(Bs[P][0], B, {0, 0, bcol_base + block_col * 2,     (KK)}, sw_B, b_srd, b_base, (P) ? b_lds_10 : b_lds_00); \
        if (!half_n_tail) G::load(Bs[P][1], B, {0, 0, bcol_base + block_col * 2 + 1, (KK)}, sw_B, b_srd, b_base, (P) ? b_lds_11 : b_lds_01); \
        LOAD_SCALES(P, (KK))
#else
    #define LOAD_ITER(P, KK)                                                                                               \
        G::load(As[P][0], A, {0, 0, block_row * 2,     (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[P][0]));   \
        G::load(As[P][1], A, {0, 0, block_row * 2 + 1, (KK)}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[P][1]));   \
        G::load(Bs[P][0], B, {0, 0, bcol_base + block_col * 2,     (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[P][0])); \
        if (!half_n_tail) G::load(Bs[P][1], B, {0, 0, bcol_base + block_col * 2 + 1, (KK)}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[P][1])); \
        LOAD_SCALES(P, (KK))
#endif
    RT_A a0, a1;
    LOAD_ITER(0, 0);
    asm volatile("s_waitcnt vmcnt(0)");
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
#if FC1_PEEL_FINAL
    // Exact GPT-OSS has 23 contraction tiles. Peel the final tile so the two
    // has-next predicates disappear from all 22 steady-state iterations.
    static_assert(k_iters == 23, "peeled FC1 loop requires GPT-OSS real K");
    #pragma unroll 1
    for (int kk = 0; kk < k_iters - 1; kk++) {
        const int cur = kk & 1;
        const int nxt = (kk + 1) & 1;
        LOAD_ITER(nxt, kk + 1);

        fp8e8m0_4 sa_h0 = PACK_FC1_A_SCALES(scale_A_smem[cur][0].data, a_row_h0);
        fp8e8m0_4 sa_h1 = PACK_FC1_A_SCALES(scale_A_smem[cur][0].data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem[cur].data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem[cur].data, b_row_h1);

#if FC1_FULL_LDS_TILES
        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur], {warp_n, 0}); load(b0, bs0);
        if (!half_n_tail) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur], {WARPS_COL + warp_n, 0}); load(b1, bs1); }
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur], {warp_m, 0}); load(a0, as0);
        if (!skip_lower_mma) { auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur], {WARPS_ROW + warp_m, 0}); load(a1, as1); }
#else
        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][0], {warp_n, 0}); load(b0, bs0);
        if (!half_n_tail) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][1], {warp_n, 0}); load(b1, bs1); }
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur][0], {warp_m, 0}); load(a0, as0);
        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur][1], {warp_m, 0}); load(a1, as1);
#endif
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cA, a0, b0, cA, &sa_h0, &sb_h0);
        if (!half_n_tail) mma_ABt_scaled(cB, a0, b1, cB, &sa_h0, &sb_h1);
        if (!skip_lower_mma) {
            mma_ABt_scaled(cC, a1, b0, cC, &sa_h1, &sb_h0);
            if (!half_n_tail) mma_ABt_scaled(cD, a1, b1, cD, &sa_h1, &sb_h1);
        }

        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
    {
        constexpr int cur = 0;
        fp8e8m0_4 sa_h0 = PACK_FC1_A_SCALES(scale_A_smem[cur][0].data, a_row_h0);
        fp8e8m0_4 sa_h1 = PACK_FC1_A_SCALES(scale_A_smem[cur][0].data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem[cur].data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem[cur].data, b_row_h1);

#if FC1_FULL_LDS_TILES
        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur], {warp_n, 0}); load(b0, bs0);
        if (!half_n_tail) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur], {WARPS_COL + warp_n, 0}); load(b1, bs1); }
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur], {warp_m, 0}); load(a0, as0);
        if (!skip_lower_mma) { auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur], {WARPS_ROW + warp_m, 0}); load(a1, as1); }
#else
        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][0], {warp_n, 0}); load(b0, bs0);
        if (!half_n_tail) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][1], {warp_n, 0}); load(b1, bs1); }
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur][0], {warp_m, 0}); load(a0, as0);
        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur][1], {warp_m, 0}); load(a1, as1);
#endif
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cA, a0, b0, cA, &sa_h0, &sb_h0);
        if (!half_n_tail) mma_ABt_scaled(cB, a0, b1, cB, &sa_h0, &sb_h1);
        if (!skip_lower_mma) {
            mma_ABt_scaled(cC, a1, b0, cC, &sa_h1, &sb_h0);
            if (!half_n_tail) mma_ABt_scaled(cD, a1, b1, cD, &sa_h1, &sb_h1);
        }
    }
#else
    #pragma unroll 1
    for (int kk = 0; kk < k_iters; kk++) {
        const int cur = kk & 1;
        const int nxt = (kk + 1) & 1;
        if (kk + 1 < k_iters) { LOAD_ITER(nxt, kk + 1); }

        fp8e8m0_4 sa_h0 = PACK_FC1_A_SCALES(scale_A_smem[cur][0].data, a_row_h0);
        fp8e8m0_4 sa_h1 = PACK_FC1_A_SCALES(scale_A_smem[cur][0].data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem[cur].data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem[cur].data, b_row_h1);

#if FC1_FULL_LDS_TILES
        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur], {warp_n, 0}); load(b0, bs0);
        if (!half_n_tail) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur], {WARPS_COL + warp_n, 0}); load(b1, bs1); }
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur], {warp_m, 0}); load(a0, as0);
        if (!skip_lower_mma) { auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur], {WARPS_ROW + warp_m, 0}); load(a1, as1); }
#else
        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][0], {warp_n, 0}); load(b0, bs0);
        if (!half_n_tail) { auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[cur][1], {warp_n, 0}); load(b1, bs1); }
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[cur][0], {warp_m, 0}); load(a0, as0);
        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[cur][1], {warp_m, 0}); load(a1, as1);
#endif
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cA, a0, b0, cA, &sa_h0, &sb_h0);
        if (!half_n_tail) mma_ABt_scaled(cB, a0, b1, cB, &sa_h0, &sb_h1);
        if (!skip_lower_mma) {
            mma_ABt_scaled(cC, a1, b0, cC, &sa_h1, &sb_h0);
            if (!half_n_tail) mma_ABt_scaled(cD, a1, b1, cD, &sa_h1, &sb_h1);
        }

        if (kk + 1 < k_iters) { asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_s_barrier(); }
    }
#endif
    #undef LOAD_ITER
#endif
#else
    #pragma unroll 1
    for (int kk = 0; kk < k_iters; kk++) {
#if FC1_SCALAR_LDS_BASES
        G::load(As[0][0], A, {0, 0, block_row * 2,     kk}, sw_A, a_srd, a_base, a_lds_00);
        G::load(As[0][1], A, {0, 0, block_row * 2 + 1, kk}, sw_A, a_srd, a_base, a_lds_01);
        G::load(Bs[0][0], B, {0, 0, bcol_base + block_col * 2,     kk}, sw_B, b_srd, b_base, b_lds_00);
        G::load(Bs[0][1], B, {0, 0, bcol_base + block_col * 2 + 1, kk}, sw_B, b_srd, b_base, b_lds_01);
#else
        G::load(As[0][0], A, {0, 0, block_row * 2,     kk}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[0][0]));
        G::load(As[0][1], A, {0, 0, block_row * 2 + 1, kk}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds[0][1]));
        G::load(Bs[0][0], B, {0, 0, bcol_base + block_col * 2,     kk}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[0][0]));
        G::load(Bs[0][1], B, {0, 0, bcol_base + block_col * 2 + 1, kk}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds[0][1]));
#endif
        LOAD_SCALES(0, kk);
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        fp8e8m0_4 sa_h0 = PACK_FC1_A_SCALES(scale_A_smem[0][0].data, a_row_h0);
        fp8e8m0_4 sa_h1 = PACK_FC1_A_SCALES(scale_A_smem[0][0].data, a_row_h1);
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

    #undef FC1_LOAD_B_TAIL
    #undef FC1_LOAD_B
    #undef FC1_LOAD_A

    // ---- Fused SwiGLU + mxfp8-quantize epilogue --------------------------------------------------------
    // Reuse the As[2][2] LDS (dead after the mma loop) as a plain row-major bf16[128][256] stripe scratch:
    // 128*256*2 == 65536 == sizeof(ST_A[2][2]). Store the fp32 gate_up accumulators as bf16 (TRUNCATED, via
    // the same convertor the base gemm uses to write C) so the LDS holds the exact bytes the reference would
    // read back from HBM. Then each thread owns one (row, 32-inter-block): it reads the 64 contiguous gate_up
    // columns, computes the SwiGLU, its own 32-wide block amax, the e8 scale, and writes 32 fp8 + 1 e8.
    const int lane      = laneid();          // 0..63
    const int lrow_grp  = 4 * (lane / 16);   // 0,4,8,12  (row group within a 16x16 base tile)
    const int lcol      = lane % 16;         // 0..15     (column within a 16x16 base tile)
    const int tid       = threadIdx.x;       // 0..511

#if REG_EPILOGUE || LOW_LDS
    // ---- Register-based epilogue: SwiGLU in registers (shfl for the gate/linear partner), block-32 amax via
    // intra-warp __shfl_xor + a 2KB LDS exchange of the per-(row,warp) partial amax across the warp_n^1 pair.
    // NO full 256x256 gate_up LDS round-trip. Byte-identical to the LDS path (same bf16-trunc/bias/OCML/_rn math;
    // max is order-independent). Reuse As (dead after the mma) as the small float amax scratch [warp_n(4)][row(128)].
#if FC1_FULL_LDS_TILES
    float *amax_lds = reinterpret_cast<float *>(&As[0]);
#elif LOW_LDS
    float *amax_lds = reinterpret_cast<float *>(&As[0]);
#else
    float *amax_lds = reinterpret_cast<float *>(&As[0][0]);
#endif
#if FC1_BIAS_LDS
    bf16 *bias_lds = reinterpret_cast<bf16 *>(&scale_A_smem[0][0]);
    #define FC1_BIAS_AT(GCOL) bias_lds[(GCOL) - block_n]
#else
    #define FC1_BIAS_AT(GCOL) Bias_ptr[e * FC1_BIAS_N + (GCOL)]
#endif
#if FC1_AMAX_PINGPONG
    // Alternate two tiny scratch regions. The next REG_ACC's barrier guarantees every thread finished reading the
    // previous region before it is reused two calls later, so a second barrier after each read is unnecessary.
    #define AMAX_LDS_INDEX(BUF, WARP, ROW) ((BUF) * WARPS_COL * HALF_ROW + (WARP) * HALF_ROW + (ROW))
    #define AMAX_REUSE_BARRIER()
#else
    #define AMAX_LDS_INDEX(BUF, WARP, ROW) ((WARP) * HALF_ROW + (ROW))
    #define AMAX_REUSE_BARRIER() __builtin_amdgcn_s_barrier();
#endif
    const int partner_n = warp_n ^ 1;
#if FUSED_FC1_COLW
    // Columnwise (transposed) mxfp8 staging tile: reuse the dead mma Bs buffers as a padded bf16
    // [HALF_ROW rows][INTER_PER_BLOCK cols] stripe of x_phys = dequant(rowwise y). +1 col pad
    // (stride INTER_PER_BLOCK+1 == 1 mod 32) makes the columnwise (token-axis) reads conflict-free, exactly like
    // transpose_quantize_mxfp8's LDS_STRIDE. 128*129*2 = 33024 B < 65536 B (Bs), so PEAK LDS is unchanged.
    constexpr int COLW_STRIDE = INTER_PER_BLOCK + 1;
#if FC1_FULL_LDS_TILES
    bf16 *ycolw_lds = reinterpret_cast<bf16 *>(&Bs[0]);
#elif LOW_LDS
    bf16 *ycolw_lds = reinterpret_cast<bf16 *>(&Bs[0]);
#else
    bf16 *ycolw_lds = reinterpret_cast<bf16 *>(&Bs[0][0]);
#endif
#if FC1_DUAL_COLW_LDS
    // The warp-scale32 path does not use As for cross-warp amax. Give each row stripe its own dead MMA buffer so
    // all accumulator epilogues can finish before either columnwise read, shortening live ranges and barriers.
#if FC1_FULL_LDS_TILES
    bf16 *ycolw_lds_alt = reinterpret_cast<bf16 *>(&As[0]);
#elif LOW_LDS
    bf16 *ycolw_lds_alt = reinterpret_cast<bf16 *>(&As[0]);
#else
    bf16 *ycolw_lds_alt = reinterpret_cast<bf16 *>(&As[0][0]);
#endif
    #define WRITE_Y_NATIVE_WIDE(BYTE, ROW, GI, BLR, WBI, DQSCALE, RB) {                                     \
        fp8e4m3 fq = __builtin_bit_cast(fp8e4m3, (unsigned char)(BYTE));                                      \
        Y_ptr[(ROW) * OUT_INTER + (GI)] = fq;                                                                  \
        float xphys = base_types::convertor<float, fp8e4m3>::convert(fq) * (DQSCALE);                         \
        bf16 *stripe_lds = ((RB) == 0) ? ycolw_lds : ycolw_lds_alt;                                           \
        stripe_lds[(BLR) * COLW_STRIDE + (WBI)] = base_types::convertor<bf16, float>::convert(xphys); }
    #define READ_YCOLW(STRIPE, INDEX) (((STRIPE) == 0 ? ycolw_lds : ycolw_lds_alt)[INDEX])
#else
    #define WRITE_Y_NATIVE_WIDE(BYTE, ROW, GI, BLR, WBI, DQSCALE, RB)                                        \
        WRITE_Y_NATIVE(BYTE, ROW, GI, BLR, WBI, DQSCALE)
    #define READ_YCOLW(STRIPE, INDEX) (ycolw_lds[INDEX])
#endif
    // dqscale = 2^(e8f-127) (exact power of 2); x_phys = float(fp8 rowwise q) * dqscale reproduces the backward's
    // aq.cast(bf16) * _mx_block_scale(ae8).cast(bf16) bit-for-bit. Staged as bf16 (the reference x_phys dtype).
    #define COLW_DQ(E8F) const float dqscale = __ocml_exp2_f32(__fsub_rn((E8F), 127.0f));
    #define WRITE_Y(Q, ROW, GI, BLR, WBI) {                                                                    \
        fp8e4m3 fq = base_types::convertor<fp8e4m3, float>::convert(Q);                                        \
        Y_ptr[(ROW) * OUT_INTER + (GI)] = fq;                                                                  \
        float xphys = base_types::convertor<float, fp8e4m3>::convert(fq) * dqscale;                            \
        ycolw_lds[(BLR) * COLW_STRIDE + (WBI)] = base_types::convertor<bf16, float>::convert(xphys); }
    #define WRITE_Y_NATIVE(BYTE, ROW, GI, BLR, WBI, DQSCALE) {                                                \
        fp8e4m3 fq = __builtin_bit_cast(fp8e4m3, (unsigned char)(BYTE));                                      \
        Y_ptr[(ROW) * OUT_INTER + (GI)] = fq;                                                                  \
        float xphys = base_types::convertor<float, fp8e4m3>::convert(fq) * (DQSCALE);                         \
        ycolw_lds[(BLR) * COLW_STRIDE + (WBI)] = base_types::convertor<bf16, float>::convert(xphys); }
#else
    #define COLW_DQ(E8F)
    #define WRITE_Y(Q, ROW, GI, BLR, WBI) Y_ptr[(ROW) * OUT_INTER + (GI)] = base_types::convertor<fp8e4m3, float>::convert(Q);
    #define WRITE_Y_NATIVE(BYTE, ROW, GI, BLR, WBI, DQSCALE)                                                  \
        Y_ptr[(ROW) * OUT_INTER + (GI)] = __builtin_bit_cast(fp8e4m3, (unsigned char)(BYTE));
    #define WRITE_Y_NATIVE_WIDE(BYTE, ROW, GI, BLR, WBI, DQSCALE, RB)                                        \
        WRITE_Y_NATIVE(BYTE, ROW, GI, BLR, WBI, DQSCALE)
#endif
    // process one REG_MxREG_N accumulator ACC at block-local col base COL_BASE (0/HALF_COL), row base ROW_BASE (0/HALF_ROW).
#if FC1_SPLIT_ROWS
    // Both parity lanes in a gate/linear pair are useful: even evaluates accumulator rows 0/1, odd evaluates 2/3.
    // XOR 2/4/8 stays within parity, so each selected row still reduces the same 16 inter values before the
    // partner-warp LDS exchange. lcol 0/1 are the unique scale writers for their respective two-row ownership.
#if FC1_WARP_SCALE32
    // With WARPS_ROW=4/WARPS_COL=2, each warp owns 64 gate_up columns == one complete 32-inter MXFP8 scale block.
    // Reduce its four per-lane inter values within the warp and quantize directly: no cross-warp LDS exchange or
    // amax barrier is required. The output tile and eight-warp MFMA work are unchanged from the 2x4 layout.
#if FC1_MED3_CLAMP
    #define FC1_CLAMP_L(LIN) __builtin_amdgcn_fmed3f((LIN), -SW_LIMIT, SW_LIMIT)
#else
    #define FC1_CLAMP_L(LIN) fminf(fmaxf((LIN), -SW_LIMIT), SW_LIMIT)
#endif
    #define FAST_E8M0(AMAX) min(254, (int)((__builtin_bit_cast(unsigned, (AMAX)) >> 23) & 0xffu))
#if FC1_ROW_SCALE_LDS4
    #define WRITE_ROW_SCALE(E8, ROW, BLR, CB, RB) {                                                   \
        if (lcol < 2) {                                                                               \
            bf16 *rsbuf = ((RB) == 0) ? ycolw_lds : ycolw_lds_alt;                                   \
            reinterpret_cast<unsigned char *>(rsbuf)[HALF_ROW * COLW_STRIDE * sizeof(bf16) +         \
                (BLR) * 4 + ((CB) / HALF_COL) * 2 + warp_n] = (uint8_t)(E8); } }
#else
    #define WRITE_ROW_SCALE(E8, ROW, BLR, CB, RB) {                                                   \
        if (lcol < 2)                                                                                 \
            Ye8_ptr[(ROW) * EBLK_PER_ROW + block_col * (INTER_PER_BLOCK / 32) +                       \
                    ((CB) / HALF_COL) * 2 + warp_n] = (uint8_t)(E8); }
#endif
    #define WIDE_QUANT_ROW(I, RR, BLR, BMAX, CB, RB) {                                                        \
        const int e8i = FAST_E8M0(BMAX);                                                                       \
        const float row_scale = __builtin_bit_cast(float, (unsigned)e8i << 23);                              \
        short2v packed[2] = {{0, 0}, {0, 0}};                                                                 \
        packed[0] = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(                                                \
            packed[0], yv[0][I][RR], yv[1][I][RR], row_scale, false);                                        \
        packed[1] = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(                                                \
            packed[1], yv[2][I][RR], yv[3][I][RR], row_scale, false);                                        \
        const int row = block_m + (RB) + (BLR);                                                               \
        _Pragma("unroll") for (int jj = 0; jj < 4; jj++) {                                                   \
            const unsigned char *row_bytes = reinterpret_cast<const unsigned char *>(&packed[jj >> 1]);       \
            const int wbi = (((CB) + warp_n * REG_N + jj * 16 + lcol) >> 1);                                 \
            const int gi = block_col * INTER_PER_BLOCK + wbi;                                                 \
            WRITE_Y_NATIVE_WIDE(row_bytes[jj & 1], row, gi, (BLR), wbi, row_scale, RB);                       \
        }                                                                                                     \
        WRITE_ROW_SCALE(e8i, row, BLR, CB, RB)                                                                \
    }
#if FC1_EP_JJ_OUTER
    // gcol/bias are invariant across the two accumulator row tiles. Traverse columns outside rows so the
    // epilogue issues one bias load per output column and shortens the temporary H live range.
    #define REG_ACC(ACC, COL_BASE, ROW_BASE, AMAX_BUF)                                                        \
    {                                                                                                         \
        float yv[4][REG_M/16][2];                                                                             \
        float wpart[REG_M/16][2];                                                                             \
        const int rbase = (lane & 1) * 2;                                                                     \
        _Pragma("unroll") for (int jj = 0; jj < 4; jj++) {                                                   \
            const int gcol = block_n + (COL_BASE) + warp_n * REG_N + jj * 16 + lcol;                         \
            const float bias_f = base_types::convertor<float, bf16>::convert(FC1_BIAS_AT(gcol));               \
            _Pragma("unroll") for (int i = 0; i < REG_M/16; i++) {                                           \
                float guv[4] = {(ACC).tiles[i][jj].data[0].x, (ACC).tiles[i][jj].data[0].y,                   \
                                (ACC).tiles[i][jj].data[1].x, (ACC).tiles[i][jj].data[1].y};                  \
                float hf[4];                                                                                  \
                _Pragma("unroll") for (int r = 0; r < 4; r++) {                                              \
                    float gu_f = base_types::convertor<float, bf16>::convert(base_types::convertor<bf16, float>::convert(guv[r])); \
                    float hh = sw_bf16_add_roundtrip(gu_f, bias_f);                                           \
                    hf[r] = hh;                                                                               \
                    const int row = block_m + (ROW_BASE) + warp_m * REG_M + i * 16 + lrow_grp + r;            \
                    H_ptr[row * N + gcol] = base_types::convertor<bf16, float>::convert(hh);                  \
                }                                                                                             \
                _Pragma("unroll") for (int rr = 0; rr < 2; rr++) {                                           \
                    float own_lo = hf[rr], own_hi = hf[rr + 2];                                               \
                    float part_lo = FC1_XOR1(own_lo), part_hi = FC1_XOR1(own_hi);                            \
                    float own = ((lane & 1) == 0) ? own_lo : own_hi;                                         \
                    float part = ((lane & 1) == 0) ? part_lo : part_hi;                                      \
                    float gate = ((lane & 1) == 0) ? own : part;                                              \
                    float lin  = ((lane & 1) == 0) ? part : own;                                              \
                    float g = fminf(gate, SW_LIMIT);                                                          \
                    float l = FC1_CLAMP_L(lin);                                                               \
                    yv[jj][i][rr] = sw_act(g, sw_sigmoid(g), l);                                             \
                }                                                                                             \
            }                                                                                                 \
        }                                                                                                     \
        _Pragma("unroll") for (int i = 0; i < REG_M/16; i++) {                                               \
            _Pragma("unroll") for (int rr = 0; rr < 2; rr++) {                                               \
                float p = fmaxf(fmaxf(fabsf(yv[0][i][rr]), fabsf(yv[1][i][rr])),                             \
                                fmaxf(fabsf(yv[2][i][rr]), fabsf(yv[3][i][rr])));                             \
                p = fmaxf(p, __shfl_xor(p, 2)); p = fmaxf(p, __shfl_xor(p, 4)); p = fmaxf(p, __shfl_xor(p, 8)); \
                wpart[i][rr] = p;                                                                             \
            }                                                                                                 \
        }                                                                                                     \
        _Pragma("unroll") for (int i = 0; i < REG_M/16; i++) _Pragma("unroll") for (int rr = 0; rr < 2; rr++) { \
            const int r = rbase + rr;                                                                         \
            const int blr = warp_m * REG_M + i * 16 + lrow_grp + r;                                           \
            WIDE_QUANT_ROW(i, rr, blr, wpart[i][rr], COL_BASE, ROW_BASE)                                     \
        }                                                                                                     \
    }
#else
    #define REG_ACC(ACC, COL_BASE, ROW_BASE, AMAX_BUF)                                                        \
    {                                                                                                         \
        float yv[4][REG_M/16][2];                                                                             \
        float wpart[REG_M/16][2];                                                                             \
        const int rbase = (lane & 1) * 2;                                                                     \
        _Pragma("unroll") for (int i = 0; i < REG_M/16; i++) {                                               \
            float hf[4][4];                                                                                   \
            _Pragma("unroll") for (int jj = 0; jj < 4; jj++) {                                               \
                float guv[4] = {(ACC).tiles[i][jj].data[0].x, (ACC).tiles[i][jj].data[0].y,                   \
                                (ACC).tiles[i][jj].data[1].x, (ACC).tiles[i][jj].data[1].y};                  \
                const int gcol = block_n + (COL_BASE) + warp_n * REG_N + jj * 16 + lcol;                      \
                float bias_f = base_types::convertor<float, bf16>::convert(FC1_BIAS_AT(gcol));                 \
                _Pragma("unroll") for (int r = 0; r < 4; r++) {                                               \
                    float gu_f = base_types::convertor<float, bf16>::convert(base_types::convertor<bf16, float>::convert(guv[r])); \
                    float hh = sw_bf16_add_roundtrip(gu_f, bias_f);                                           \
                    hf[jj][r] = hh;                                                                           \
                    const int row = block_m + (ROW_BASE) + warp_m * REG_M + i * 16 + lrow_grp + r;            \
                    H_ptr[row * N + gcol] = base_types::convertor<bf16, float>::convert(hh);                  \
                }                                                                                             \
            }                                                                                                 \
            _Pragma("unroll") for (int jj = 0; jj < 4; jj++) {                                               \
                _Pragma("unroll") for (int rr = 0; rr < 2; rr++) {                                            \
                    float own_lo = hf[jj][rr], own_hi = hf[jj][rr + 2];                                      \
                    float part_lo = FC1_XOR1(own_lo), part_hi = FC1_XOR1(own_hi);                            \
                    float own = ((lane & 1) == 0) ? own_lo : own_hi;                                         \
                    float part = ((lane & 1) == 0) ? part_lo : part_hi;                                      \
                    float gate = ((lane & 1) == 0) ? own : part;                                              \
                    float lin  = ((lane & 1) == 0) ? part : own;                                              \
                    float g = fminf(gate, SW_LIMIT);                                                          \
                    float l = FC1_CLAMP_L(lin);                                                               \
                    yv[jj][i][rr] = sw_act(g, sw_sigmoid(g), l);                                             \
                }                                                                                             \
            }                                                                                                 \
            _Pragma("unroll") for (int rr = 0; rr < 2; rr++) {                                               \
                float p = fmaxf(fmaxf(fabsf(yv[0][i][rr]), fabsf(yv[1][i][rr])),                             \
                                fmaxf(fabsf(yv[2][i][rr]), fabsf(yv[3][i][rr])));                             \
                p = fmaxf(p, __shfl_xor(p, 2)); p = fmaxf(p, __shfl_xor(p, 4)); p = fmaxf(p, __shfl_xor(p, 8)); \
                wpart[i][rr] = p;                                                                             \
            }                                                                                                 \
        }                                                                                                     \
        _Pragma("unroll") for (int i = 0; i < REG_M/16; i++) _Pragma("unroll") for (int rr = 0; rr < 2; rr++) { \
            const int r = rbase + rr;                                                                         \
            const int blr = warp_m * REG_M + i * 16 + lrow_grp + r;                                           \
            WIDE_QUANT_ROW(i, rr, blr, wpart[i][rr], COL_BASE, ROW_BASE)                                     \
        }                                                                                                     \
    }
#endif
#else
#if FC1_NATIVE_ROW
    #define FAST_E8M0(AMAX) min(254, (int)((__builtin_bit_cast(unsigned, (AMAX)) >> 23) & 0xffu))
    #define SPLIT_QUANT_ROW(I, RR, BLR, BMAX, CB, RB) {                                                 \
        const int e8i = FAST_E8M0(BMAX);                                                               \
        const float row_scale = __builtin_bit_cast(float, (unsigned)e8i << 23);                        \
        short2v packed = {0, 0};                                                                       \
        packed = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(                                             \
            packed, yv[0][I][RR], yv[1][I][RR], row_scale, false);                                     \
        const unsigned char *row_bytes = reinterpret_cast<const unsigned char *>(&packed);              \
        const int row = block_m + (RB) + (BLR);                                                         \
        _Pragma("unroll") for (int jj = 0; jj < 2; jj++) {                                              \
            const int wbi = (((CB) + warp_n * REG_N + jj * 16 + lcol) >> 1);                           \
            const int gi = block_col * INTER_PER_BLOCK + wbi;                                           \
            WRITE_Y_NATIVE(row_bytes[jj], row, gi, (BLR), wbi, row_scale);                              \
        }                                                                                               \
        if (lcol < 2 && (warp_n & 1) == 0)                                                             \
            Ye8_ptr[row * EBLK_PER_ROW + block_col * (INTER_PER_BLOCK / 32) + (((CB) / HALF_COL) * 2 + (warp_n >> 1))] = (uint8_t)e8i; \
    }
#else
    #define SPLIT_QUANT_ROW(I, RR, BLR, BMAX, CB, RB) {                                                 \
        float e8f = __fadd_rn(floorf(__ocml_log2_f32(fmaxf((BMAX), 1e-38f))), 127.0f);                 \
        e8f = fminf(fmaxf(e8f, 0.0f), 254.0f);                                                         \
        float qscale = __ocml_exp2_f32(__fsub_rn(127.0f, e8f));                                        \
        COLW_DQ(e8f)                                                                                    \
        const int row = block_m + (RB) + (BLR);                                                         \
        _Pragma("unroll") for (int jj = 0; jj < 2; jj++) {                                              \
            const int wbi = (((CB) + warp_n * REG_N + jj * 16 + lcol) >> 1);                           \
            const int gi = block_col * INTER_PER_BLOCK + wbi;                                           \
            float q = fminf(fmaxf(__fmul_rn(yv[jj][I][RR], qscale), -SW_FP8_MAX), SW_FP8_MAX);        \
            WRITE_Y(q, row, gi, (BLR), wbi);                                                            \
        }                                                                                               \
        if (lcol < 2 && (warp_n & 1) == 0)                                                             \
            Ye8_ptr[row * EBLK_PER_ROW + block_col * (INTER_PER_BLOCK / 32) + (((CB) / HALF_COL) * 2 + (warp_n >> 1))] = (uint8_t)e8f; \
    }
#endif
    #define REG_ACC(ACC, COL_BASE, ROW_BASE, AMAX_BUF)                                                        \
    {                                                                                                         \
        float yv[2][REG_M/16][2];   /* [jj][i][owned row] */                                                 \
        float wpart[REG_M/16][2];   /* this parity's two selected rows */                                    \
        const int rbase = (lane & 1) * 2;                                                                     \
        _Pragma("unroll") for (int i = 0; i < REG_M/16; i++) {                                                \
            float hf[2][4];                                                                                   \
            _Pragma("unroll") for (int jj = 0; jj < 2; jj++) {                                                \
                float guv[4] = {(ACC).tiles[i][jj].data[0].x, (ACC).tiles[i][jj].data[0].y,                   \
                                (ACC).tiles[i][jj].data[1].x, (ACC).tiles[i][jj].data[1].y};                  \
                const int gcol = block_n + (COL_BASE) + warp_n * REG_N + jj * 16 + lcol;                      \
                float bias_f = base_types::convertor<float, bf16>::convert(FC1_BIAS_AT(gcol));                 \
                _Pragma("unroll") for (int r = 0; r < 4; r++) {                                               \
                    float gu_f = base_types::convertor<float, bf16>::convert(base_types::convertor<bf16, float>::convert(guv[r])); \
                    float hh = sw_bf16_add_roundtrip(gu_f, bias_f);                                           \
                    hf[jj][r] = hh;                                                                           \
                    const int row = block_m + (ROW_BASE) + warp_m * REG_M + i * 16 + lrow_grp + r;            \
                    H_ptr[row * N + gcol] = base_types::convertor<bf16, float>::convert(hh);                  \
                }                                                                                             \
            }                                                                                                 \
            _Pragma("unroll") for (int jj = 0; jj < 2; jj++) {                                                \
                _Pragma("unroll") for (int rr = 0; rr < 2; rr++) {                                            \
                    float own_lo = hf[jj][rr];                                                               \
                    float own_hi = hf[jj][rr + 2];                                                           \
                    float part_lo = FC1_XOR1(own_lo);                                                        \
                    float part_hi = FC1_XOR1(own_hi);                                                        \
                    float own = ((lane & 1) == 0) ? own_lo : own_hi;                                         \
                    float part = ((lane & 1) == 0) ? part_lo : part_hi;                                      \
                    float gate = ((lane & 1) == 0) ? own : part;                                              \
                    float lin  = ((lane & 1) == 0) ? part : own;                                              \
                    float g = fminf(gate, SW_LIMIT);                                                          \
                    float l = fminf(fmaxf(lin, -SW_LIMIT), SW_LIMIT);                                         \
                    float sig = sw_sigmoid(g); \
                    yv[jj][i][rr] = sw_act(g, sig, l);                         \
                }                                                                                             \
            }                                                                                                 \
            _Pragma("unroll") for (int rr = 0; rr < 2; rr++) {                                                \
                float p = fmaxf(fabsf(yv[0][i][rr]), fabsf(yv[1][i][rr]));                                    \
                p = fmaxf(p, __shfl_xor(p, 2)); p = fmaxf(p, __shfl_xor(p, 4)); p = fmaxf(p, __shfl_xor(p, 8)); \
                wpart[i][rr] = p;                                                                             \
            }                                                                                                 \
        }                                                                                                     \
        if (lcol < 2) {                                                                                       \
            _Pragma("unroll") for (int i = 0; i < REG_M/16; i++) _Pragma("unroll") for (int rr = 0; rr < 2; rr++) { \
                const int r = rbase + rr;                                                                     \
                amax_lds[AMAX_LDS_INDEX(AMAX_BUF, warp_n, warp_m * REG_M + i * 16 + lrow_grp + r)] = wpart[i][rr]; \
            }                                                                                                 \
        }                                                                                                     \
        __builtin_amdgcn_s_barrier();                                                                         \
        _Pragma("unroll") for (int i = 0; i < REG_M/16; i++) _Pragma("unroll") for (int rr = 0; rr < 2; rr++) { \
            const int r = rbase + rr;                                                                         \
            const int blr = warp_m * REG_M + i * 16 + lrow_grp + r;                                           \
            float bmax = fmaxf(wpart[i][rr], amax_lds[AMAX_LDS_INDEX(AMAX_BUF, partner_n, blr)]);            \
            SPLIT_QUANT_ROW(i, rr, blr, bmax, COL_BASE, ROW_BASE)                                             \
        }                                                                                                     \
        AMAX_REUSE_BARRIER()                                                                                   \
    }
#endif
#else
    #define REG_ACC(ACC, COL_BASE, ROW_BASE, AMAX_BUF)                                                        \
    {                                                                                                         \
        float yv[2][REG_M/16][4];   /* [jj][i][r] swiglu output (valid on even/gate lanes) */                 \
        float wpart[REG_M/16][4];   /* [i][r] this warp's 16-inter partial amax */                            \
        _Pragma("unroll") for (int i = 0; i < REG_M/16; i++) {                                                \
            float hf[2][4];                                                                                   \
            _Pragma("unroll") for (int jj = 0; jj < 2; jj++) {                                                \
                float guv[4] = {(ACC).tiles[i][jj].data[0].x, (ACC).tiles[i][jj].data[0].y,                   \
                                (ACC).tiles[i][jj].data[1].x, (ACC).tiles[i][jj].data[1].y};                  \
                const int gcol = block_n + (COL_BASE) + warp_n * REG_N + jj * 16 + lcol;                      \
                float bias_f = base_types::convertor<float, bf16>::convert(FC1_BIAS_AT(gcol));                 \
                _Pragma("unroll") for (int r = 0; r < 4; r++) {                                               \
                    float gu_f = base_types::convertor<float, bf16>::convert(base_types::convertor<bf16, float>::convert(guv[r])); \
                    float hh = sw_bf16_add_roundtrip(gu_f, bias_f);            /* h = rne_bf16(gate_up + bias) */ \
                    hf[jj][r] = hh;                                                                           \
                    const int row = block_m + (ROW_BASE) + warp_m * REG_M + i * 16 + lrow_grp + r;            \
                    H_ptr[row * N + gcol] = base_types::convertor<bf16, float>::convert(hh);                  \
                }                                                                                             \
            }                                                                                                 \
            /* swiglu: pull the linear partner (laneid^1); even & odd of a pair compute the same y for inter j */ \
            _Pragma("unroll") for (int jj = 0; jj < 2; jj++) {                                                \
                _Pragma("unroll") for (int r = 0; r < 4; r++) {                                               \
                    float own = hf[jj][r];                                                                    \
                    float part = FC1_XOR1(own);                                                               \
                    float gate = ((lane & 1) == 0) ? own : part;                                              \
                    float lin  = ((lane & 1) == 0) ? part : own;                                              \
                    float g = fminf(gate, SW_LIMIT);                                                          \
                    float l = fminf(fmaxf(lin, -SW_LIMIT), SW_LIMIT);                                         \
                    float sig = sw_sigmoid(g); \
                    yv[jj][i][r] = sw_act(g, sig, l);                          \
                }                                                                                             \
            }                                                                                                 \
            /* per-row partial = max over this lane's 2 inters, then reduce the 8 even lanes (shfl_xor 2,4,8) */ \
            _Pragma("unroll") for (int r = 0; r < 4; r++) {                                                   \
                float p = fmaxf(fabsf(yv[0][i][r]), fabsf(yv[1][i][r]));                                      \
                p = fmaxf(p, __shfl_xor(p, 2)); p = fmaxf(p, __shfl_xor(p, 4)); p = fmaxf(p, __shfl_xor(p, 8)); \
                wpart[i][r] = p;                                                                              \
            }                                                                                                 \
        }                                                                                                     \
        if (lcol == 0) {                                                                                      \
            _Pragma("unroll") for (int i = 0; i < REG_M/16; i++) _Pragma("unroll") for (int r = 0; r < 4; r++)\
                amax_lds[AMAX_LDS_INDEX(AMAX_BUF, warp_n, warp_m * REG_M + i * 16 + lrow_grp + r)] = wpart[i][r]; \
        }                                                                                                     \
        __builtin_amdgcn_s_barrier();                                                                         \
        _Pragma("unroll") for (int i = 0; i < REG_M/16; i++) _Pragma("unroll") for (int r = 0; r < 4; r++) {  \
            const int blr = warp_m * REG_M + i * 16 + lrow_grp + r;                                           \
            float bmax = fmaxf(wpart[i][r], amax_lds[AMAX_LDS_INDEX(AMAX_BUF, partner_n, blr)]); /* block-32 amax (2 warps) */\
            float e8f = __fadd_rn(floorf(__ocml_log2_f32(fmaxf(bmax, 1e-38f))), 127.0f);                      \
            e8f = fminf(fmaxf(e8f, 0.0f), 254.0f);                                                            \
            float qscale = __ocml_exp2_f32(__fsub_rn(127.0f, e8f));                                           \
            COLW_DQ(e8f)                                                                                       \
            const int row = block_m + (ROW_BASE) + blr;                                                       \
            if ((lane & 1) == 0) {                                                                            \
                _Pragma("unroll") for (int jj = 0; jj < 2; jj++) {                                            \
                    const int wbi = (((COL_BASE) + warp_n * REG_N + jj * 16 + lcol) >> 1);                     \
                    const int gi = block_col * INTER_PER_BLOCK + wbi;                                          \
                    float q = fminf(fmaxf(__fmul_rn(yv[jj][i][r], qscale), -SW_FP8_MAX), SW_FP8_MAX);         \
                    WRITE_Y(q, row, gi, blr, wbi);                                                             \
                }                                                                                             \
                if (lcol == 0 && (warp_n & 1) == 0)                                                           \
                    Ye8_ptr[row * EBLK_PER_ROW + block_col * (INTER_PER_BLOCK / 32) + (((COL_BASE) / HALF_COL) * 2 + (warp_n >> 1))] = (uint8_t)e8f; \
            }                                                                                                 \
        }                                                                                                     \
        AMAX_REUSE_BARRIER()           /* reuse amax_lds for the next accumulator */                          \
    }
#endif

#if FC1_BIAS_LDS
    // Scale operands are already in registers and their LDS is dead once all
    // MFMAs have issued. The contraction's existing final barrier also
    // publishes this cooperative bias copy, avoiding another synchronization.
    if (tid < (half_n_tail ? HALF_COL / 8 : BLOCK_COL / 8)) {
        const uint4v *bias_src = reinterpret_cast<const uint4v *>(&Bias_ptr[e * FC1_BIAS_N + block_n]);
        reinterpret_cast<uint4v *>(bias_lds)[tid] = bias_src[tid];
    }
    asm volatile("s_waitcnt vmcnt(0)");
    asm volatile("s_waitcnt lgkmcnt(0)");
#endif
    __builtin_amdgcn_s_barrier();       // As free after the last mma iteration; optional bias tile ready
#if FUSED_FC1_COLW
    // ---- Columnwise (transposed) mxfp8 emit ---------------------------------------------------------------
    // After the two REG_ACCs of a HALF_ROW stripe have staged x_phys into ycolw_lds (row-major [row][inter]), each
    // of the 512 threads owns one (within-block inter column, 32-token block) and reads its column of 32 tokens
    // from LDS (the transpose), computing the block-32 amax over TOKENS + e8m0 + native fp8 cvt exactly like
    // transpose_quantize_mxfp8. Output y_colw is (INTER, M): 32 contiguous fp8 along M per (inter, token-block).
#if NATIVE_MXFP8_CVT
    // e8m0 = biased-exponent field of amax (bit-exact vs floorf(log2)+127 for normals); hardware scaled cvt.
    #define COLW_QUANT(VALS, OUT, E8VAR)                                                                       \
        int E8VAR = (int)((__builtin_bit_cast(unsigned, amax) >> 23) & 0xFFu); E8VAR = max(0, min(254, E8VAR)); \
        const float colw_scale = __builtin_bit_cast(float, (unsigned)(E8VAR) << 23);                          \
        _Pragma("unroll") for (int t = 0; t < 32; t += 2) {                                                    \
            short2v acc = {0, 0};                                                                             \
            acc = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(acc, (VALS)[t], (VALS)[t + 1], colw_scale, false); \
            const unsigned char *rb = reinterpret_cast<const unsigned char *>(&acc);                          \
            (OUT)[t] = rb[0]; (OUT)[t + 1] = rb[1]; }
#else
    #define COLW_QUANT(VALS, OUT, E8VAR)                                                                       \
        int E8VAR = (int)floorf(log2f(fmaxf(amax, 1e-38f))) + 127; E8VAR = max(0, min(254, E8VAR));           \
        const float colw_qs = exp2f((float)(127 - (E8VAR)));                                                  \
        _Pragma("unroll") for (int t = 0; t < 32; t++)                                                        \
            (OUT)[t] = __hip_cvt_float_to_fp8(fmaxf(-SW_FP8_MAX, fminf(SW_FP8_MAX, (VALS)[t] * colw_qs)), __HIP_SATFINITE, __HIP_E4M3);
#endif
#if FC1_COLW_ADJ_SCALE2
    #define FC1_COLW_SCALE_WRITE(STRIPE, GINTER, GROW0, E8) pair_e8[STRIPE] = (uint8_t)(E8)
#else
    #define FC1_COLW_SCALE_WRITE(STRIPE, GINTER, GROW0, E8) \
        Ycolw_e8_ptr[FC1_COLW_SCALE_INDEX((long long)(GINTER) * (M / 32) + (GROW0) / 32, (GINTER), (GROW0)/32)] = (uint8_t)(E8)
#endif
    #define COLW_STRIPE(STRIPE) {                                                                             \
        const int cwcol  = tid & (INTER_PER_BLOCK - 1);          /* within-block inter col (0..127) */         \
        const int pair   = (tid / INTER_PER_BLOCK) & 3;                                                        \
        const int block32 = FC1_COLW_ADJ_SCALE2 ? pair * 2 + (STRIPE) : (STRIPE) * 4 + pair;                  \
        const int pstripe = block32 / 4;                                                                       \
        const int tblk   = block32 & 3;                                                                         \
        const int ginter = block_col * INTER_PER_BLOCK + cwcol;                                                \
        const int grow0  = block_m + block32 * 32;                                                              \
        if ((!half_n_tail || cwcol < (REAL_N % BLOCK_COL) / 2) &&                                             \
            (!omit_lower_outputs || block32 < HALF_ROW / 32)) {                                                \
        float vals[32]; float amax = 0.0f;                                                                    \
        _Pragma("unroll") for (int t = 0; t < 32; t++) {                                                      \
            float v = base_types::convertor<float, bf16>::convert(READ_YCOLW(pstripe, (tblk * 32 + t) * COLW_STRIDE + cwcol)); \
            vals[t] = v; amax = fmaxf(amax, fabsf(v)); }                                                       \
        unsigned char out[32]; COLW_QUANT(vals, out, e8i);                                                   \
        long long obase = (long long)ginter * M + grow0;                                                      \
        *reinterpret_cast<uint4 *>(&Ycolw_ptr[obase])      = *reinterpret_cast<uint4 *>(&out[0]);             \
        *reinterpret_cast<uint4 *>(&Ycolw_ptr[obase + 16]) = *reinterpret_cast<uint4 *>(&out[16]);            \
        FC1_COLW_SCALE_WRITE(STRIPE, ginter, grow0, e8i); } }
#if FC1_DUAL_COLW_LDS
    REG_ACC(cA, 0, 0, 0);
    if (!half_n_tail) REG_ACC(cB, HALF_COL, 0, 1);
#if FC1_PAD_LOWER_REUSE
    // A fully padded lower wave band has zero accumulators. Its exact post-bias result is identical on all four
    // rows owned by a split-row lane, so reuse the SwiGLU value, row scale, and native conversion across them.
    #define REG_ACC_PAD(COL_BASE, ROW_BASE)                                                                 \
    {                                                                                                      \
        float yv[4];                                                                                        \
        const int rbase = (lane & 1) * 2;                                                                  \
        _Pragma("unroll") for (int jj = 0; jj < 4; jj++) {                                                \
            const int gcol = block_n + (COL_BASE) + warp_n * REG_N + jj * 16 + lcol;                      \
            const float bias_f = base_types::convertor<float, bf16>::convert(                              \
                FC1_BIAS_AT(gcol));                                                                        \
            const float hh = sw_bf16_add_roundtrip(0.0f, bias_f);                                          \
            const bf16 hh_b = base_types::convertor<bf16, float>::convert(hh);                             \
            _Pragma("unroll") for (int i = 0; i < REG_M/16; i++) _Pragma("unroll") for (int r = 0; r < 4; r++) { \
                const int row = block_m + (ROW_BASE) + warp_m * REG_M + i * 16 + lrow_grp + r;             \
                H_ptr[row * N + gcol] = hh_b;                                                               \
            }                                                                                              \
            const float part = FC1_XOR1(hh);                                                               \
            const float gate = ((lane & 1) == 0) ? hh : part;                                              \
            const float lin  = ((lane & 1) == 0) ? part : hh;                                              \
            const float g = fminf(gate, SW_LIMIT);                                                         \
            const float l = FC1_CLAMP_L(lin);                                                              \
            yv[jj] = sw_act(g, sw_sigmoid(g), l);                                                          \
        }                                                                                                  \
        float p = fmaxf(fmaxf(fabsf(yv[0]), fabsf(yv[1])), fmaxf(fabsf(yv[2]), fabsf(yv[3])));           \
        p = fmaxf(p, __shfl_xor(p, 2)); p = fmaxf(p, __shfl_xor(p, 4)); p = fmaxf(p, __shfl_xor(p, 8)); \
        const int e8i = FAST_E8M0(p);                                                                       \
        const float row_scale = __builtin_bit_cast(float, (unsigned)e8i << 23);                           \
        short2v packed[2] = {{0, 0}, {0, 0}};                                                              \
        packed[0] = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(packed[0], yv[0], yv[1], row_scale, false); \
        packed[1] = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(packed[1], yv[2], yv[3], row_scale, false); \
        _Pragma("unroll") for (int i = 0; i < REG_M/16; i++) _Pragma("unroll") for (int rr = 0; rr < 2; rr++) { \
            const int r = rbase + rr;                                                                      \
            const int blr = warp_m * REG_M + i * 16 + lrow_grp + r;                                       \
            const int row = block_m + (ROW_BASE) + blr;                                                    \
            _Pragma("unroll") for (int jj = 0; jj < 4; jj++) {                                           \
                const unsigned char *row_bytes = reinterpret_cast<const unsigned char *>(&packed[jj >> 1]); \
                const int wbi = (((COL_BASE) + warp_n * REG_N + jj * 16 + lcol) >> 1);                     \
                const int gi = block_col * INTER_PER_BLOCK + wbi;                                          \
                WRITE_Y_NATIVE_WIDE(row_bytes[jj & 1], row, gi, blr, wbi, row_scale, ROW_BASE);           \
            }                                                                                              \
            WRITE_ROW_SCALE(e8i, row, blr, COL_BASE, ROW_BASE)                                           \
        }                                                                                                  \
    }
    if (skip_lower_mma) {
#if FC1_OMIT_PAD_LOWER_OUTPUTS
        if (!omit_lower_outputs) {
#endif
        REG_ACC_PAD(0, HALF_ROW);
        if (!half_n_tail) REG_ACC_PAD(HALF_COL, HALF_ROW);
#if FC1_OMIT_PAD_LOWER_OUTPUTS
        }
#endif
    } else {
        REG_ACC(cC, 0, HALF_ROW, 0);
        if (!half_n_tail) REG_ACC(cD, HALF_COL, HALF_ROW, 1);
    }
    #undef REG_ACC_PAD
#else
    REG_ACC(cC, 0, HALF_ROW, 0);
    if (!half_n_tail) REG_ACC(cD, HALF_COL, HALF_ROW, 1);
#endif
    __builtin_amdgcn_s_barrier();       // both stripe buffers fully staged
#if FC1_ROW_SCALE_LDS4
    // One thread per output row gathers its four scale bytes after the existing publishing barrier. The physical
    // half-N tail owns only two live scale bytes, so preserve the two untouched padding bytes with a short store.
    if (tid < BLOCK_ROW && (!omit_lower_outputs || tid < HALF_ROW)) {
        const int stripe = tid / HALF_ROW, blr = tid & (HALF_ROW - 1);
        bf16 *rsbuf = stripe == 0 ? ycolw_lds : ycolw_lds_alt;
        const unsigned char *rs = reinterpret_cast<const unsigned char *>(rsbuf) +
                                  HALF_ROW * COLW_STRIDE * sizeof(bf16) + blr * 4;
        const long long ro = (long long)(block_m + tid) * EBLK_PER_ROW +
                             block_col * (INTER_PER_BLOCK / 32);
        if (half_n_tail)
            *reinterpret_cast<unsigned short *>(&Ye8_ptr[ro]) = *reinterpret_cast<const unsigned short *>(rs);
        else
            *reinterpret_cast<unsigned *>(&Ye8_ptr[ro]) = *reinterpret_cast<const unsigned *>(rs);
    }
#endif
#if FC1_COLW_ADJ_SCALE2
    unsigned char pair_e8[2];
#endif
    COLW_STRIPE(0);
    COLW_STRIPE(1);
#if FC1_COLW_ADJ_SCALE2
    {
        const int cwcol = tid & (INTER_PER_BLOCK - 1);
        const int pair = (tid / INTER_PER_BLOCK) & 3;
        if ((!half_n_tail || cwcol < (REAL_N % BLOCK_COL) / 2) &&
            (!omit_lower_outputs || pair < (HALF_ROW / 32) / 2)) {
            const int ginter = block_col * INTER_PER_BLOCK + cwcol;
            const long long sbase = FC1_COLW_SCALE_INDEX((long long)ginter * (M / 32) + block_m / 32 + pair * 2,
                                                        ginter, block_m/32+pair*2);
            *reinterpret_cast<unsigned short *>(&Ycolw_e8_ptr[sbase]) =
                (unsigned short)pair_e8[0] | ((unsigned short)pair_e8[1] << 8);
        }
    }
#endif
#else
    REG_ACC(cA, 0, 0, 0);
    if (!half_n_tail) REG_ACC(cB, HALF_COL, 0, 1);
    __builtin_amdgcn_s_barrier();       // stripe-0 x_phys fully staged in ycolw_lds
    COLW_STRIPE(0);
    __builtin_amdgcn_s_barrier();       // stripe-0 columnwise reads done before cC/cD overwrite ycolw_lds
    REG_ACC(cC, 0, HALF_ROW, 0);
    if (!half_n_tail) REG_ACC(cD, HALF_COL, HALF_ROW, 1);
    __builtin_amdgcn_s_barrier();       // stripe-1 x_phys fully staged
    COLW_STRIPE(1);
#endif
    #undef COLW_STRIPE
    #undef COLW_QUANT
    #undef FC1_COLW_SCALE_WRITE
#else
    REG_ACC(cA, 0, 0, 0);
    if (!half_n_tail) REG_ACC(cB, HALF_COL, 0, 1);
    REG_ACC(cC, 0, HALF_ROW, half_n_tail ? 1 : 0);
    if (!half_n_tail) REG_ACC(cD, HALF_COL, HALF_ROW, 1);
#endif
    #undef SPLIT_QUANT_ROW
    #undef WIDE_QUANT_ROW
    #undef FC1_CLAMP_L
    #undef FAST_E8M0
    #undef WRITE_ROW_SCALE
    #undef WRITE_Y
    #undef WRITE_Y_NATIVE
    #undef WRITE_Y_NATIVE_WIDE
    #undef READ_YCOLW
    #undef COLW_DQ
    #undef AMAX_REUSE_BARRIER
    #undef AMAX_LDS_INDEX
    #undef FC1_BIAS_AT
#else
#if FC1_FULL_LDS_TILES
    bf16 *gu = reinterpret_cast<bf16 *>(&As[0]);
#else
    bf16 *gu = reinterpret_cast<bf16 *>(&As[0][0]);
#endif

    // store one REG_MxREG_N accumulator tile into the stripe buffer at block-local column base COL_BASE (0/HALF_COL)
    #define STORE_ACC(ACC, COL_BASE)                                                                          \
        _Pragma("unroll") for (int i = 0; i < REG_M/16; i++) {                                                \
            const int r0 = warp_m * REG_M + i * 16 + lrow_grp;                                                \
            _Pragma("unroll") for (int j = 0; j < 2; j++) {                                                   \
                const int col = (COL_BASE) + warp_n * REG_N + j * 16 + lcol;                                  \
                gu[(r0 + 0) * BLOCK_COL + col] = base_types::convertor<bf16, float>::convert((ACC).tiles[i][j].data[0].x); \
                gu[(r0 + 1) * BLOCK_COL + col] = base_types::convertor<bf16, float>::convert((ACC).tiles[i][j].data[0].y); \
                gu[(r0 + 2) * BLOCK_COL + col] = base_types::convertor<bf16, float>::convert((ACC).tiles[i][j].data[1].x); \
                gu[(r0 + 3) * BLOCK_COL + col] = base_types::convertor<bf16, float>::convert((ACC).tiles[i][j].data[1].y); \
            }                                                                                                 \
        }

    // quantize the HALF_ROW-row stripe currently in `gu`. `stripe` selects which row-half of the block.
    // (BLOCK_ROW/HALF_ROW=2 threads-per-row-group cover HALF_ROW rows; rows >= HALF_ROW are idle under TILE_M128.)
    #define QUANTIZE_STRIPE(STRIPE)                                                                           \
        if ((tid / 4) < HALF_ROW) {                                                                          \
            const int row        = tid / 4;                                                                  \
            const int iblk       = tid % 4;                                                                  \
            const int global_row = block_m + (STRIPE) * HALF_ROW + row;                                      \
            float ys[32];                                                                                    \
            float amax = 0.0f;                                                                               \
            _Pragma("unroll") for (int t = 0; t < 32; t++) {                                                 \
                const int li   = iblk * 32 + t;                                                              \
                float gate = base_types::convertor<float, bf16>::convert(gu[row * BLOCK_COL + 2 * li]);      \
                float lin  = base_types::convertor<float, bf16>::convert(gu[row * BLOCK_COL + 2 * li + 1]);  \
                /* fold per-expert bias BEFORE swiglu, matching reference bf16 add: h = rne_bf16(gate_up + bias) */ \
                const int gu_col = block_n + 2 * li;                                                         \
                gate = sw_bf16_add_roundtrip(gate, base_types::convertor<float, bf16>::convert(Bias_ptr[e * FC1_BIAS_N + gu_col]));     \
                lin  = sw_bf16_add_roundtrip(lin,  base_types::convertor<float, bf16>::convert(Bias_ptr[e * FC1_BIAS_N + gu_col + 1])); \
                /* materialize h = gate_up + bias (bf16, full width) so the backward reads it instead of recomputing */ \
                H_ptr[global_row * N + gu_col]     = base_types::convertor<bf16, float>::convert(gate);      \
                H_ptr[global_row * N + gu_col + 1] = base_types::convertor<bf16, float>::convert(lin);       \
                float g = fminf(gate, SW_LIMIT);                                                             \
                float l = fminf(fmaxf(lin, -SW_LIMIT), SW_LIMIT);                                            \
                /* sig = 1/(1+exp2(g*c)); act = g*sig*(l+1). _rn intrinsics + OCML match tinygrad's strict IEEE */ \
                float sig = sw_sigmoid(g); \
                float y = sw_act(g, sig, l);                                  \
                ys[t] = y;                                                                                   \
                amax = fmaxf(amax, fabsf(y));                                                                \
            }                                                                                                \
            float e8f = __fadd_rn(floorf(__ocml_log2_f32(fmaxf(amax, 1e-38f))), 127.0f);                     \
            e8f = fminf(fmaxf(e8f, 0.0f), 254.0f);                                                           \
            float qscale = __ocml_exp2_f32(__fsub_rn(127.0f, e8f));                                          \
            _Pragma("unroll") for (int t = 0; t < 32; t++) {                                                 \
                const int li = iblk * 32 + t;                                                                \
                const int global_inter = block_col * INTER_PER_BLOCK + li;                                   \
                float q = fminf(fmaxf(__fmul_rn(ys[t], qscale), -SW_FP8_MAX), SW_FP8_MAX);                   \
                Y_ptr[global_row * OUT_INTER + global_inter] = base_types::convertor<fp8e4m3, float>::convert(q);\
            }                                                                                                \
            Ye8_ptr[global_row * EBLK_PER_ROW + block_col * (INTER_PER_BLOCK / 32) + iblk] = (uint8_t)e8f;   \
        }

    __builtin_amdgcn_s_barrier();          // As is free after the last mma iteration's readers finished
    // stripe 0: row-half 0 -> cA (cols 0..HALF_COL-1) + cB (cols HALF_COL..BLOCK_COL-1)
    STORE_ACC(cA, 0);
    STORE_ACC(cB, HALF_COL);
    __builtin_amdgcn_s_barrier();
    QUANTIZE_STRIPE(0);
    __builtin_amdgcn_s_barrier();
    // stripe 1: row-half 1 -> cC (cols 0..HALF_COL-1) + cD (cols HALF_COL..BLOCK_COL-1)
    STORE_ACC(cC, 0);
    STORE_ACC(cD, HALF_COL);
    __builtin_amdgcn_s_barrier();
    QUANTIZE_STRIPE(1);

    #undef STORE_ACC
    #undef QUANTIZE_STRIPE
#endif
#if FC1_HALF_N_TAIL
    if (half_n_tail) {
        constexpr int REAL_INTER = REAL_N / 2;
        constexpr int COMPUTED_PAD = INTER - REAL_INTER;
#if FC1_VECTOR_TAIL_ZERO
        static_assert(HALF_COL % 8 == 0 && COMPUTED_PAD % 16 == 0 && COMPUTED_PAD % 32 == 0);
#if !FC1_SKIP_H_TAIL_ZERO
        #pragma unroll 1
        for (int i = tid; i < BLOCK_ROW * (HALF_COL / 8); i += NUM_THREADS) {
            const int row = i / (HALF_COL / 8), vec = i % (HALF_COL / 8);
            FC1_STORE_ZERO(reinterpret_cast<uint4v *>(&H_ptr[(long long)(block_m + row) * N + block_n + HALF_COL + vec * 8]), uint4v{});
        }
#endif
        #pragma unroll 1
        for (int i = tid; i < BLOCK_ROW * (COMPUTED_PAD / 16); i += NUM_THREADS) {
            const int row = i / (COMPUTED_PAD / 16), vec = i % (COMPUTED_PAD / 16);
            FC1_STORE_ZERO(reinterpret_cast<uint4v *>(&Y_ptr[(long long)(block_m + row) * OUT_INTER + REAL_INTER + vec * 16]), uint4v{});
        }
        #pragma unroll 1
        for (int row = tid; row < BLOCK_ROW; row += NUM_THREADS)
            FC1_STORE_ZERO(reinterpret_cast<uint16_t *>(&Ye8_ptr[(long long)(block_m + row) * EBLK_PER_ROW + REAL_INTER / 32]), uint16_t{0});
#if FUSED_FC1_COLW
        #pragma unroll 1
        for (int i = tid; i < COMPUTED_PAD * (BLOCK_ROW / 16); i += NUM_THREADS) {
            const int col = i / (BLOCK_ROW / 16), vec = i % (BLOCK_ROW / 16);
            FC1_STORE_ZERO(reinterpret_cast<uint4v *>(&Ycolw_ptr[(long long)(REAL_INTER + col) * M + block_m + vec * 16]), uint4v{});
        }
        #pragma unroll 1
        for (int col = tid; col < COMPUTED_PAD; col += NUM_THREADS)
            FC1_COLW_SCALE_ZERO8((long long)(REAL_INTER + col) * (M / 32) + block_m / 32, REAL_INTER + col, block_m/32);
#endif
#else
#if !FC1_SKIP_H_TAIL_ZERO
        #pragma unroll 1
        for (int i = tid; i < BLOCK_ROW * HALF_COL; i += NUM_THREADS) {
            const int row = i / HALF_COL, col = i % HALF_COL;
            reinterpret_cast<uint16_t *>(H_ptr)[(long long)(block_m + row) * N + block_n + HALF_COL + col] = 0;
        }
#endif
        #pragma unroll 1
        for (int i = tid; i < BLOCK_ROW * COMPUTED_PAD; i += NUM_THREADS) {
            const int row = i / COMPUTED_PAD, col = i % COMPUTED_PAD;
            reinterpret_cast<uint8_t *>(Y_ptr)[(long long)(block_m + row) * OUT_INTER + REAL_INTER + col] = 0;
        }
        #pragma unroll 1
        for (int i = tid; i < BLOCK_ROW * (COMPUTED_PAD / 32); i += NUM_THREADS) {
            const int row = i / (COMPUTED_PAD / 32), col = i % (COMPUTED_PAD / 32);
            Ye8_ptr[(long long)(block_m + row) * EBLK_PER_ROW + REAL_INTER / 32 + col] = 0;
        }
#if FUSED_FC1_COLW
        #pragma unroll 1
        for (int i = tid; i < COMPUTED_PAD * BLOCK_ROW; i += NUM_THREADS) {
            const int col = i / BLOCK_ROW, row = i % BLOCK_ROW;
            reinterpret_cast<uint8_t *>(Ycolw_ptr)[(long long)(REAL_INTER + col) * M + block_m + row] = 0;
        }
        #pragma unroll 1
        for (int i = tid; i < COMPUTED_PAD * (BLOCK_ROW / 32); i += NUM_THREADS) {
            const int col = i / (BLOCK_ROW / 32), rb = i % (BLOCK_ROW / 32);
            Ycolw_e8_ptr[FC1_COLW_SCALE_INDEX((long long)(REAL_INTER + col) * (M / 32) + block_m / 32 + rb,
                                            REAL_INTER+col, block_m/32+rb)] = 0;
        }
#endif
#endif
    }
#endif
    ZERO_FC2_TAIL();
    #undef ZERO_FC2_TAIL
    #undef ZERO_FC2_COLW_TAIL
}
#undef FC1_STORE_ZERO
#undef FC1_A_RESTRICT
#undef FC1_B_RESTRICT
#undef FC1_SCALE_A_RESTRICT
#undef FC1_SCALE_B_RESTRICT
#undef FC1_XOR1
