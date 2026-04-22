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

import functools
from tinygrad import Device, Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates

# Shape constants (Qwen3.5-0.8B dense FFN).
_DIM, _HIDDEN = 1024, 3584
# Q8_0: 34 bytes per 32 weights (2B fp16 scale + 32B int8 qs).
_Q8_BYTES_PER_WEIGHT_NUMERATOR = 34
_Q8_BYTES_PER_WEIGHT_DENOMINATOR = 32


def _q8_bytes(numel: int) -> int:
  """Byte count for a Q8_0-encoded weight tensor with `numel` scalar weights."""
  assert numel % _Q8_BYTES_PER_WEIGHT_DENOMINATOR == 0
  return (numel // _Q8_BYTES_PER_WEIGHT_DENOMINATOR) * _Q8_BYTES_PER_WEIGHT_NUMERATOR


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
constant constexpr uint ROWS_PER_GROUP = 8u;

// 128 threads, each owns 1 Q8 col-block (tid < 112 valid; 16 idle). Each thread
// accumulates 8 partial dot products (one per output row). Within each warp,
// simd_sum reduces 32 lanes -> 1 partial per row; cross-warp via tg-mem.
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
  uint cb = tid;
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
  return MetalCompiler().compile_cached(_GATE_UP_METAL_SRC)


@functools.cache
def _compiled_down() -> bytes:
  from tinygrad.runtime.ops_metal import MetalCompiler
  return MetalCompiler().compile_cached(_DOWN_RESIDUAL_METAL_SRC)


def _gate_up_kernel(z: UOp, h: UOp, norm_w: UOp, gate_w: UOp, up_w: UOp) -> UOp:
  lib = _compiled_gate_up()
  assert z.numel() == 3584, f"fused_gate_up_q8 expects hidden=3584, got {z.numel()}"
  # Estimates (Qwen3.5-0.8B dense FFN):
  #   ops: gate matvec (2*H*D FMAs) + up matvec (2*H*D) + RMSNorm (~5*D) + silu*mul (~2*H)
  #        dominated by matvecs -> 4*H*D = 14.68M ops
  #   mem: h (D*4B) + norm_w (D*4B) + gate Q8 (_q8_bytes(H*D)) + up Q8 + z out (H*4B)
  ops = 4 * _HIDDEN * _DIM + 5 * _DIM + 2 * _HIDDEN
  mem = (_DIM + _DIM + _HIDDEN) * 4 + 2 * _q8_bytes(_HIDDEN * _DIM)
  # Grid: HIDDEN/ROWS_PER_GROUP = 3584/8 = 448 threadgroups * 128 threads (4 warps).
  sink = UOp.sink(
    UOp.special(448, "gidx0"),
    UOp.special(32, "lidx0"),
    UOp.special(4, "lidx1"),
    z, h, norm_w, gate_w, up_w,
    arg=KernelInfo(name="fused_gate_up_q8", estimates=Estimates(ops=ops, mem=mem)),
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
  # Estimates:
  #   ops: down matvec (2*D*H) + residual add (D) = 2*D*H + D = 7.34M
  #   mem: h (D*4B) + z (H*4B) + down Q8 (_q8_bytes(D*H)) + out (D*4B)
  ops = 2 * _DIM * _HIDDEN + _DIM
  mem = (_DIM + _HIDDEN + _DIM) * 4 + _q8_bytes(_DIM * _HIDDEN)
  # Grid: DIM/ROWS_PER_GROUP = 1024/8 = 128 threadgroups * 128 threads (4 warps).
  sink = UOp.sink(
    UOp.special(128, "gidx0"),
    UOp.special(32, "lidx0"),
    UOp.special(4, "lidx1"),
    out, h, z, down_w,
    arg=KernelInfo(name="fused_down_q8", estimates=Estimates(ops=ops, mem=mem)),
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
  if norm_raw is None or gate_raw is None or up_raw is None or down_raw is None:
    raise NotImplementedError("fused_ffn_with_residual: all weights must trace to a raw uchar buffer")

  z = Tensor.empty(3584, dtype=dtypes.float, device=h.device)
  z, h_after, *_ = Tensor.custom_kernel(z, h, norm_raw, gate_raw, up_raw, fxn=_gate_up_kernel)

  out_empty = Tensor.empty(h.shape, dtype=dtypes.float, device=h.device)
  out, *_ = Tensor.custom_kernel(out_empty, h_after, z, down_raw, fxn=_down_kernel)

  return out


# ---------------- Attention QKV fusion ----------------
# Qwen3.5-0.8B TransformerBlock:
#   q = attn_q(attn_norm(x))  # shape (4096,) = n_heads(8) * 2 * head_dim(256), the 2 is (q, gate)
#   k = attn_k(attn_norm(x))  # shape (512,)  = n_kv_heads(2) * head_dim(256)
#   v = attn_v(attn_norm(x))  # shape (512,)  = n_kv_heads(2) * head_dim(256)
# Baseline emits 3 matvec kernels (r_4096_32_32 @ 288us, r_512_32_32 x2 @ 73us) + norm reduce.
# This fuses all three matvecs + RMSNorm into one dispatch emitting a single (5120,) qkv buffer.
#
# Shapes (Qwen3.5-0.8B):
_ATTN_DIM = 1024
_ATTN_Q_OUT = 4096   # n_heads * 2 * head_dim (includes output gate interleave)
_ATTN_KV_OUT = 512   # n_kv_heads * head_dim
_ATTN_QKV_OUT = _ATTN_Q_OUT + 2 * _ATTN_KV_OUT  # 5120


_ATTN_QKV_METAL_SRC = r"""
#include <metal_stdlib>
using namespace metal;

constant constexpr uint DIM = 1024u;
constant constexpr uint Q_OUT = 4096u;
constant constexpr uint KV_OUT = 512u;
constant constexpr uint QKV_OUT = 5120u;
constant constexpr float RMS_EPS = 1e-6f;
constant constexpr uint WARPS = 4u;
constant constexpr uint SIMD = 32u;
constant constexpr uint THREADS = WARPS * SIMD;                 // 128
constant constexpr uint ROWS_PER_WARP = 2u;
constant constexpr uint ROWS_PER_GROUP = WARPS * ROWS_PER_WARP; // 8
constant constexpr uint BLOCKS_PER_ROW = DIM / 32u;             // 32

// Grid: QKV_OUT/ROWS_PER_GROUP = 5120/8 = 640 threadgroups * 128 threads.
// Writes to THREE separate output buffers (q/k/v) so downstream reshape+slice
// doesn't trigger a copy. Logical output layout:
//   rows [0    .. 4096) -> data0 (q_out)     weights data4 (attn_q Q8)
//   rows [4096 .. 4608) -> data1 (k_out)     weights data5 (attn_k Q8)
//   rows [4608 .. 5120) -> data2 (v_out)     weights data6 (attn_v Q8)
kernel void fused_attn_qkv_q8(
    device float* data0,                      // q_out[4096]
    device float* data1,                      // k_out[512]
    device float* data2,                      // v_out[512]
    device const float* data3,                // x[DIM] (pre-norm input)
    device const float* data4_norm,           // attn_norm.weight[DIM]   -- renamed to avoid data4 reuse
    device const uchar* data5,                // attn_q Q8 weights
    device const uchar* data6,                // attn_k Q8 weights
    device const uchar* data7,                // attn_v Q8 weights
    uint gid [[threadgroup_position_in_grid]],
    uint lidx0 [[thread_index_in_simdgroup]],
    uint lidx1 [[simdgroup_index_in_threadgroup]]) {
  threadgroup float x_norm[DIM];
  threadgroup float rms_scratch[WARPS];

  uint tid = lidx1 * SIMD + lidx0;

  // ---- RMSNorm (shared across q/k/v) ----
  float sq = 0.0f;
  #pragma unroll
  for (uint i = 0; i < DIM; i += THREADS) {
    float v = data3[i + tid];
    sq += v * v;
  }
  float warp_sq = simd_sum(sq);
  if (lidx0 == 0) rms_scratch[lidx1] = warp_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total_sq = rms_scratch[0] + rms_scratch[1] + rms_scratch[2] + rms_scratch[3];
  float inv_rms = rsqrt(total_sq / float(DIM) + RMS_EPS);

  #pragma unroll
  for (uint i = 0; i < DIM; i += THREADS) {
    x_norm[i + tid] = data3[i + tid] * inv_rms * data4_norm[i + tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- matvec phase ----
  uint cb = lidx0;
  float xchunk[32];
  uint xbase = cb * 32u;
  #pragma unroll
  for (uint j = 0; j < 32u; j++) xchunk[j] = x_norm[xbase + j];

  uint row0 = gid * ROWS_PER_GROUP + lidx1 * ROWS_PER_WARP;
  #pragma unroll
  for (uint r = 0; r < ROWS_PER_WARP; r++) {
    uint row = row0 + r;

    // Pick weights + destination based on which segment this row falls in.
    device const uchar* wblk;
    device float* dst;
    uint dst_idx;
    if (row < Q_OUT) {
      wblk = data5 + (row * BLOCKS_PER_ROW + cb) * 34u;
      dst = data0; dst_idx = row;
    } else if (row < Q_OUT + KV_OUT) {
      uint kr = row - Q_OUT;
      wblk = data6 + (kr * BLOCKS_PER_ROW + cb) * 34u;
      dst = data1; dst_idx = kr;
    } else {
      uint vr = row - Q_OUT - KV_OUT;
      wblk = data7 + (vr * BLOCKS_PER_ROW + cb) * 34u;
      dst = data2; dst_idx = vr;
    }
    float scale = float(*((device const half*)wblk));
    float acc = 0.0f;
    #pragma unroll
    for (uint j = 0; j < 32u; j++) {
      acc += float(as_type<int8_t>(wblk[2u + j])) * xchunk[j];
    }
    float full = simd_sum(scale * acc);
    if (lidx0 == 0) dst[dst_idx] = full;
  }
}
"""


@functools.cache
def _compiled_attn_qkv() -> bytes:
  from tinygrad.runtime.ops_metal import MetalCompiler
  return MetalCompiler().compile_cached(_ATTN_QKV_METAL_SRC)


def _attn_qkv_kernel(q_out: UOp, k_out: UOp, v_out: UOp, x: UOp, norm_w: UOp, q_w: UOp, k_w: UOp, v_w: UOp) -> UOp:
  lib = _compiled_attn_qkv()
  assert q_out.numel() == _ATTN_Q_OUT, f"q_out must be {_ATTN_Q_OUT}"
  assert k_out.numel() == _ATTN_KV_OUT and v_out.numel() == _ATTN_KV_OUT
  ops = 2 * _ATTN_QKV_OUT * _ATTN_DIM + 5 * _ATTN_DIM
  mem = (_ATTN_DIM + _ATTN_DIM + _ATTN_QKV_OUT) * 4 + _q8_bytes(_ATTN_Q_OUT * _ATTN_DIM) + 2 * _q8_bytes(_ATTN_KV_OUT * _ATTN_DIM)
  # Grid: QKV_OUT/ROWS_PER_GROUP = 5120/8 = 640 threadgroups * 128 threads.
  sink = UOp.sink(
    UOp.special(640, "gidx0"),
    UOp.special(32, "lidx0"),
    UOp.special(4, "lidx1"),
    q_out, k_out, v_out, x, norm_w, q_w, k_w, v_w,
    arg=KernelInfo(name="fused_attn_qkv_q8", estimates=Estimates(ops=ops, mem=mem)),
  )
  return UOp(
    Ops.PROGRAM,
    src=(
      sink,
      UOp(Ops.DEVICE, arg=Device.DEFAULT),
      UOp(Ops.LINEAR, src=(*sink.src, sink)),
      UOp(Ops.SOURCE, arg=_ATTN_QKV_METAL_SRC),
      UOp(Ops.BINARY, arg=lib),
    ),
  )


def fused_attn_qkv(x: Tensor, norm_w: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor,
                   q_shape: tuple[int,...], kv_shape: tuple[int,...]) -> tuple[Tensor, Tensor, Tensor]:
  """Replacement for `attn_q(attn_norm(x)), attn_k(attn_norm(x)), attn_v(attn_norm(x))`.

    Fuses the RMSNorm + 3 matvecs into one kernel with THREE separate output
    buffers (q, k, v) allocated with the caller's desired shape so downstream
    reshape() calls fold away without materializing.
    """
  assert x.numel() == _ATTN_DIM, f"fused_attn_qkv: x must have {_ATTN_DIM} elements, got {x.numel()}"

  norm_raw = _find_raw_q8_blocks(norm_w)
  q_raw = _find_raw_q8_blocks(q_w)
  k_raw = _find_raw_q8_blocks(k_w)
  v_raw = _find_raw_q8_blocks(v_w)
  if norm_raw is None or q_raw is None or k_raw is None or v_raw is None:
    raise NotImplementedError("fused_attn_qkv: all weights must trace to a raw uchar buffer")

  q = Tensor.empty(q_shape,  dtype=dtypes.float, device=x.device)
  k = Tensor.empty(kv_shape, dtype=dtypes.float, device=x.device)
  v = Tensor.empty(kv_shape, dtype=dtypes.float, device=x.device)
  q, k, v, *_ = Tensor.custom_kernel(q, k, v, x, norm_raw, q_raw, k_raw, v_raw, fxn=_attn_qkv_kernel)
  return q, k, v
