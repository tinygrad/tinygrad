# Custom hand-rolled Metal kernels for the LLM decode path.
#
# Baseline hot path (from VIZ): per block per token, tinygrad emits
#   ffn_norm reduce (r_32_32n1)  -- 1 kernel
#   ffn_gate matvec              -- 1 kernel
#   ffn_up matvec                -- 1 kernel
#   silu*up elementwise          -- 1 kernel
#   ffn_down matvec + residual   -- 1 kernel
# With Qwen3.5-0.8B's 24 blocks that's 120 launches @ ~10us each = ~1.2 ms of pure
# dispatch overhead per token. We collapse these into 2 launches per block:
#   1. fused_gate_up_q8: RMSNorm + gate + up + silu*mul -> z[hidden]
#   2. fused_down_q8:    down @ z + residual            -> out[dim]
#
# One kernel would be nicer but keeping hidden[3584] in threadgroup memory would
# either force 1 threadgroup per token (undersubscribes the GPU) or redundant
# recomputation. Splitting lets each kernel use a grid large enough to saturate
# the GPU while still fusing the expensive RMSNorm+gate+up+silu chain.
#
# Enable with CUSTOM_MLP=1.

from __future__ import annotations
import functools
from tinygrad import Device, Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo


def _find_raw_q8_blocks(weight: Tensor) -> Tensor | None:
    """Walk the weight tensor's uop chain looking for the CONTIGUOUS uchar node
    that holds raw Q8_0 blocks (2-byte fp16 scale + 32 int8 qs per 34-byte block)
    or the raw F32 bytes for norm weights.
    """
    seen: set[int] = set()
    stack = [weight.uop]
    while stack:
        u = stack.pop()
        if id(u) in seen: continue
        seen.add(id(u))
        if u.op is Ops.CONTIGUOUS and u.dtype.scalar() == dtypes.uchar:
            return Tensor(u)
        stack.extend(u.src)
    return None


# ---- Pre-dequantized fp16 weight cache (Kimi v12 "fp16 weight buffers") --------------
# Convert Q8_0 blocks -> fp16 weights once at first use, cache forever.
# Trades ~2x memory (1.06 B/weight -> 2 B/weight) for simpler kernels that do
# native half loads instead of per-byte int8 unpack. Must be called OUTSIDE a
# JIT-captured function (we realize the tensors once up-front).
_FP16_WEIGHT_CACHE: dict[int, Tensor] = {}

def _q8_to_fp16(weight: Tensor) -> Tensor:
    """Return a pre-realized fp16 Tensor equivalent to `weight`. Caches across calls."""
    key = id(weight.uop.base if hasattr(weight.uop, 'base') else weight.uop)
    if key in _FP16_WEIGHT_CACHE:
        return _FP16_WEIGHT_CACHE[key]
    # weight.dtype is already half; .contiguous() materializes the dequant chain.
    fp16 = weight.contiguous().realize()
    _FP16_WEIGHT_CACHE[key] = fp16
    return fp16


def _prepare_ffn_fp16_weights(norm_w: Tensor, gate_w: Tensor, up_w: Tensor, down_w: Tensor):
    """Eagerly realize fp16 versions of FFN weights. Call once per block during
    model setup, outside any JIT context."""
    _q8_to_fp16(gate_w)
    _q8_to_fp16(up_w)
    _q8_to_fp16(down_w)


# ---------------- Kernel 1: fused RMSNorm + gate + up + silu*mul ----------------
# Inputs:  data0 = z      (float, shape (hidden,))        -- output: silu(gate@xn)*up@xn
#          data1 = h      (float, shape (dim,))           -- input residual stream
#          data2 = norm_w (float, shape (dim,))           -- ffn_norm.weight as raw F32 bytes
#          data3 = gate   (uchar, shape (hidden*dim/32, 34))  -- raw Q8_0 gate blocks
#          data4 = up     (uchar, shape (hidden*dim/32, 34))  -- raw Q8_0 up blocks
#
# Grid strategy: one threadgroup produces ROWS_PER_GROUP output rows of z.
# 32 threads cooperate on each row's dim=1024 reduction. Each thread sweeps
# dim/32 = 32 reduce iterations, one Q8 block per iter. Final reduce is a
# simd_sum within the warp (no threadgroup memory needed since 32 threads = 1 SIMD group).
#
# Hardcoded for Qwen3.5-0.8B: dim=1024, hidden=3584. hidden=3584=2^6*7*8 factors
# nicely as 112 groups * 32 rows, or 224 * 16, etc. I pick 112 groups * 32 rows
# each thread produces 32 rows -> each thread computes 32 partial dot products.
# Actually simpler: 32 groups of 112 rows with 1 row per thread... no, 32 threads
# must cooperate on the reduction. Use 112 groups * 32 rows/group, each group's
# 32 threads each own one output row and do the full 1024-dim reduce via
# simd_sum across the group's 32 threads. 112 * 32 = 3584 = hidden. clean.
_GATE_UP_METAL_SRC = r"""
#include <metal_stdlib>
using namespace metal;

constant constexpr uint DIM = 1024u;
constant constexpr uint HIDDEN = 3584u;
constant constexpr float RMS_EPS = 1e-6f;
constant constexpr uint WARPS = 4u;
constant constexpr uint SIMD = 32u;
constant constexpr uint THREADS = WARPS * SIMD;                 // 128
constant constexpr uint ROWS_PER_WARP = 2u;
constant constexpr uint ROWS_PER_GROUP = WARPS * ROWS_PER_WARP; // 8
constant constexpr uint BLOCKS_PER_ROW = DIM / 32u;             // 32

// Grid: HIDDEN/ROWS_PER_GROUP = 3584/8 = 448 threadgroups * 128 threads.
// Each WARP independently computes ROWS_PER_WARP=2 output rows using all 32 lanes
// on the 1024-wide reduction. No cross-warp reduction needed -- pure simd_sum.
kernel void fused_gate_up_q8(
    device float* data0,                      // z[HIDDEN]
    device const float* data1,                // h[DIM]
    device const float* data2,                // norm_w[DIM]
    device const uchar* data3,                // gate Q8 blocks
    device const uchar* data4,                // up Q8 blocks
    uint gid [[threadgroup_position_in_grid]],
    uint lidx0 [[thread_index_in_simdgroup]],
    uint lidx1 [[simdgroup_index_in_threadgroup]]) {
  threadgroup float x_norm[DIM];
  threadgroup float rms_scratch[WARPS];

  uint tid = lidx1 * SIMD + lidx0;

  // ---- RMSNorm ----
  float sq = 0.0f;
  #pragma unroll
  for (uint i = 0; i < DIM; i += THREADS) {
    float v = data1[i + tid];
    sq += v * v;
  }
  float warp_sq = simd_sum(sq);
  if (lidx0 == 0) rms_scratch[lidx1] = warp_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total_sq = rms_scratch[0] + rms_scratch[1] + rms_scratch[2] + rms_scratch[3];
  float inv_rms = rsqrt(total_sq / float(DIM) + RMS_EPS);

  #pragma unroll
  for (uint i = 0; i < DIM; i += THREADS) {
    x_norm[i + tid] = data1[i + tid] * inv_rms * data2[i + tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- Gate/Up matvec: each warp handles ROWS_PER_WARP output rows ----
  // Within each warp, 32 lanes own 1 Q8 block each (32 blocks * 32 cols = 1024).
  uint cb = lidx0;
  float xchunk[32];
  uint xbase = cb * 32u;
  #pragma unroll
  for (uint j = 0; j < 32u; j++) xchunk[j] = x_norm[xbase + j];

  uint row0 = gid * ROWS_PER_GROUP + lidx1 * ROWS_PER_WARP;
  #pragma unroll
  for (uint r = 0; r < ROWS_PER_WARP; r++) {
    uint blk_idx = (row0 + r) * BLOCKS_PER_ROW + cb;
    device const uchar* gblk = data3 + blk_idx * 34u;
    device const uchar* ublk = data4 + blk_idx * 34u;
    float gscale = float(*((device const half*)gblk));
    float uscale = float(*((device const half*)ublk));
    float gacc = 0.0f, uacc = 0.0f;
    #pragma unroll
    for (uint j = 0; j < 32u; j++) {
      float x = xchunk[j];
      gacc += float(as_type<int8_t>(gblk[2u + j])) * x;
      uacc += float(as_type<int8_t>(ublk[2u + j])) * x;
    }
    float g_full = simd_sum(gscale * gacc);
    float u_full = simd_sum(uscale * uacc);
    if (lidx0 == 0) {
      float silu_g = g_full / (1.0f + exp(-g_full));
      data0[row0 + r] = silu_g * u_full;
    }
  }
}
"""

# ---------------- Kernel 2: fused down + residual ----------------
# Inputs:  data0 = out    (float, shape (dim,))           -- output: h + down@z
#          data1 = h      (float, shape (dim,))           -- input residual
#          data2 = z      (float, shape (hidden,))        -- from kernel 1
#          data3 = down   (uchar, shape (dim*hidden/32, 34)) -- raw Q8_0 down blocks
#
# Grid: DIM/THREADS = 32 groups * 32 threads/group = 1024 threads, one per output row.
# Each thread owns one row and does the full hidden=3584 reduce across 112 Q8 blocks.
_DOWN_RESIDUAL_METAL_SRC = r"""
#include <metal_stdlib>
using namespace metal;

constant constexpr uint DIM = 1024u;
constant constexpr uint HIDDEN = 3584u;
constant constexpr uint WARPS = 4u;
constant constexpr uint SIMD = 32u;
constant constexpr uint THREADS = WARPS * SIMD;          // 128
constant constexpr uint BLOCKS_PER_ROW = HIDDEN / 32u;   // 112
constant constexpr uint ROWS_PER_GROUP = 8u;             // emit 8 output rows per TG

kernel void fused_down_q8(
    device float* data0,                      // out[DIM]
    device const float* data1,                // h[DIM] (residual)
    device const float* data2,                // z[HIDDEN]
    device const uchar* data3,                // down Q8 blocks
    uint gid [[threadgroup_position_in_grid]],
    uint lidx0 [[thread_index_in_simdgroup]],
    uint lidx1 [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red[WARPS * ROWS_PER_GROUP];

  uint tid = lidx1 * SIMD + lidx0;
  uint cb = tid;  // col-block index 0..127 (valid if < 112)
  bool valid = cb < BLOCKS_PER_ROW;

  float zchunk[32];
  if (valid) {
    uint base = cb * 32u;
    #pragma unroll
    for (uint j = 0; j < 32u; j++) zchunk[j] = data2[base + j];
  } else {
    #pragma unroll
    for (uint j = 0; j < 32u; j++) zchunk[j] = 0.0f;
  }

  float partial[ROWS_PER_GROUP];
  uint row0 = gid * ROWS_PER_GROUP;
  if (valid) {
    #pragma unroll
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
      uint blk_idx = (row0 + r) * BLOCKS_PER_ROW + cb;
      device const uchar* blk = data3 + blk_idx * 34u;
      float scale = float(*((device const half*)blk));
      float acc = 0.0f;
      #pragma unroll
      for (uint j = 0; j < 32u; j++) {
        acc += float(as_type<int8_t>(blk[2u + j])) * zchunk[j];
      }
      partial[r] = scale * acc;
    }
  } else {
    #pragma unroll
    for (uint r = 0; r < ROWS_PER_GROUP; r++) partial[r] = 0.0f;
  }

  #pragma unroll
  for (uint r = 0; r < ROWS_PER_GROUP; r++) {
    float warp_partial = simd_sum(partial[r]);
    if (lidx0 == 0) red[lidx1 * ROWS_PER_GROUP + r] = warp_partial;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lidx1 == 0 && lidx0 < ROWS_PER_GROUP) {
    float full = red[0 * ROWS_PER_GROUP + lidx0] + red[1 * ROWS_PER_GROUP + lidx0]
               + red[2 * ROWS_PER_GROUP + lidx0] + red[3 * ROWS_PER_GROUP + lidx0];
    data0[row0 + lidx0] = data1[row0 + lidx0] + full;
  }
}
"""


@functools.cache
def _compiled_gate_up() -> bytes:
    from tinygrad.runtime.ops_metal import MetalCompiler
    return MetalCompiler().compile(_GATE_UP_METAL_SRC)


@functools.cache
def _compiled_down() -> bytes:
    from tinygrad.runtime.ops_metal import MetalCompiler
    return MetalCompiler().compile(_DOWN_RESIDUAL_METAL_SRC)


def _gate_up_kernel(z: UOp, h: UOp, norm_w: UOp, gate_w: UOp, up_w: UOp) -> UOp:
    lib = _compiled_gate_up()
    assert z.numel() == 3584, f"fused_gate_up_q8 expects hidden=3584, got {z.numel()}"
    # Grid: HIDDEN/ROWS_PER_GROUP = 3584/8 = 448 threadgroups * 128 threads (4 warps).
    sink = UOp.sink(
        UOp.special(448, "gidx0"),
        UOp.special(32, "lidx0"),
        UOp.special(4, "lidx1"),
        z, h, norm_w, gate_w, up_w,
        arg=KernelInfo(name="fused_gate_up_q8"),
    )
    return UOp(
        Ops.PROGRAM,
        src=(
            sink,
            UOp(Ops.DEVICE, arg=Device.DEFAULT),
            UOp(Ops.LINEAR, src=(*sink.src, sink)),
            UOp(Ops.SOURCE, arg=_GATE_UP_METAL_SRC),
            UOp(Ops.BINARY, arg=lib),
        ),
    )


def _down_kernel(out: UOp, h: UOp, z: UOp, down_w: UOp) -> UOp:
    lib = _compiled_down()
    assert out.numel() == 1024, f"fused_down_q8 expects dim=1024, got {out.numel()}"
    # Grid: DIM/ROWS_PER_GROUP = 1024/8 = 128 threadgroups * 128 threads (4 warps).
    sink = UOp.sink(
        UOp.special(128, "gidx0"),
        UOp.special(32, "lidx0"),
        UOp.special(4, "lidx1"),
        out, h, z, down_w,
        arg=KernelInfo(name="fused_down_q8"),
    )
    return UOp(
        Ops.PROGRAM,
        src=(
            sink,
            UOp(Ops.DEVICE, arg=Device.DEFAULT),
            UOp(Ops.LINEAR, src=(*sink.src, sink)),
            UOp(Ops.SOURCE, arg=_DOWN_RESIDUAL_METAL_SRC),
            UOp(Ops.BINARY, arg=lib),
        ),
    )


def fused_ffn_with_residual(h: Tensor, norm_w: Tensor,
                            gate_w: Tensor, up_w: Tensor, down_w: Tensor) -> Tensor:
    """Replacement for `h + ffn_down(silu(ffn_gate(ffn_norm(h))) * ffn_up(ffn_norm(h)))`.

    Uses raw Q8_0 weight bytes (cheaper than fp16 on bandwidth-bound M-series).
    """
    assert h.numel() == 1024, f"fused_ffn only supports dim=1024, got {h.numel()}"

    norm_raw = _find_raw_q8_blocks(norm_w)
    gate_raw = _find_raw_q8_blocks(gate_w)
    up_raw = _find_raw_q8_blocks(up_w)
    down_raw = _find_raw_q8_blocks(down_w)
    if any(t is None for t in (norm_raw, gate_raw, up_raw, down_raw)):
        raise NotImplementedError("fused_ffn_with_residual: all weights must trace to a raw uchar buffer")

    z = Tensor.empty(3584, dtype=dtypes.float, device=h.device)
    z, h_after, *_ = Tensor.custom_kernel(z, h, norm_raw, gate_raw, up_raw, fxn=_gate_up_kernel)

    out_empty = Tensor.empty(h.shape, dtype=dtypes.float, device=h.device)
    out, *_ = Tensor.custom_kernel(out_empty, h_after, z, down_raw, fxn=_down_kernel)

    return out
