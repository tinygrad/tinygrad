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

    gguf.py:57 builds every Q8_0 weight as
        blocks[:,:2].bitcast(half).cast(float) * blocks[:,2:].bitcast(int8)
    where `blocks` is the one `contiguous()` uchar tensor. F32 weights (like the
    norm weights) likewise trace back to a `contiguous()` uchar buffer holding
    their little-endian float bytes. Finding that node lets us hand the raw bytes
    to a custom kernel without triggering rematerialization.
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
constant constexpr uint THREADS = 32u;
constant constexpr float RMS_EPS = 1e-6f;
constant constexpr uint ROWS_PER_GROUP = 7u;
constant constexpr uint GATE_BLOCKS_PER_ROW = DIM / 32u;  // 32

// Use threadgroup reduction via shared memory (Apple GPUs are fast on this).
// Grid: HIDDEN/ROWS_PER_GROUP = 512 threadgroups * 32 threads.
kernel void fused_gate_up_q8(
    device float* data0,                      // z[HIDDEN]
    device const float* data1,                // h[DIM]
    device const float* data2,                // norm_w[DIM]
    device const uchar* data3,                // gate Q8 blocks
    device const uchar* data4,                // up Q8 blocks
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float x_norm[DIM];
  threadgroup float redg[THREADS * ROWS_PER_GROUP];  // 7*32 = 224 (baseline uses this exact size)
  threadgroup float redu[THREADS * ROWS_PER_GROUP];

  // ---- RMSNorm ----
  float sq = 0.0f;
  #pragma unroll
  for (uint i = 0; i < DIM; i += THREADS) {
    float v = data1[i + tid];
    sq += v * v;
  }
  float inv_rms = rsqrt(simd_sum(sq) / float(DIM) + RMS_EPS);

  #pragma unroll
  for (uint i = 0; i < DIM; i += THREADS) {
    x_norm[i + tid] = data1[i + tid] * inv_rms * data2[i + tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- Gate/Up matvec ----
  // Each thread processes column-block `tid` of its threadgroup's 7 output rows.
  // Preload 32 x_norm floats for this thread's column-block (reused across all 7 rows).
  float xchunk[32];
  uint x_base = tid * 32u;
  #pragma unroll
  for (uint j = 0; j < 32u; j++) xchunk[j] = x_norm[x_base + j];

  float gp[ROWS_PER_GROUP];
  float up[ROWS_PER_GROUP];
  uint row0 = gid * ROWS_PER_GROUP;
  #pragma unroll
  for (uint r = 0; r < ROWS_PER_GROUP; r++) {
    uint blk_idx = (row0 + r) * GATE_BLOCKS_PER_ROW + tid;
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
    gp[r] = gscale * gacc;
    up[r] = uscale * uacc;
  }

  // Cross-thread reduction via threadgroup memory (baseline pattern).
  // Each thread writes its 7 partials; lane 0..6 each reduce one row.
  #pragma unroll
  for (uint r = 0; r < ROWS_PER_GROUP; r++) {
    redg[tid * ROWS_PER_GROUP + r] = gp[r];
    redu[tid * ROWS_PER_GROUP + r] = up[r];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < ROWS_PER_GROUP) {
    float g_full = 0.0f, u_full = 0.0f;
    #pragma unroll
    for (uint i = 0; i < THREADS; i++) {
      g_full += redg[i * ROWS_PER_GROUP + tid];
      u_full += redu[i * ROWS_PER_GROUP + tid];
    }
    float silu_g = g_full / (1.0f + exp(-g_full));
    data0[row0 + tid] = silu_g * u_full;
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
constant constexpr uint THREADS = 32u;
constant constexpr uint DOWN_BLOCKS_PER_ROW = HIDDEN / 32u;  // 112
constant constexpr uint ROWS_PER_GROUP = 8u;
constant constexpr uint BLOCKS_PER_THREAD = 4u;              // 32 * 4 = 128 >= 112; last 16 masked

// Grid: DIM/ROWS_PER_GROUP = 128 threadgroups * 32 threads.
kernel void fused_down_q8(
    device float* data0,                      // out[DIM]
    device const float* data1,                // h[DIM]
    device const float* data2,                // z[HIDDEN]
    device const uchar* data3,                // down Q8 blocks
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float red[THREADS * ROWS_PER_GROUP];  // 8 * 32 = 256

  // Preload this thread's z chunk (128 floats = 4 Q8 blocks worth).
  uint blk_start = tid * BLOCKS_PER_THREAD;
  float zchunk[BLOCKS_PER_THREAD * 32u];  // 128
  #pragma unroll
  for (uint b = 0; b < BLOCKS_PER_THREAD; b++) {
    uint cb = blk_start + b;
    uint base = cb * 32u;
    #pragma unroll
    for (uint j = 0; j < 32u; j++) {
      uint c = base + j;
      zchunk[b*32u + j] = (c < HIDDEN) ? data2[c] : 0.0f;
    }
  }

  // Accumulate 8 partial dot products in register.
  float partial[ROWS_PER_GROUP];
  uint row0 = gid * ROWS_PER_GROUP;
  #pragma unroll
  for (uint r = 0; r < ROWS_PER_GROUP; r++) {
    float p = 0.0f;
    #pragma unroll
    for (uint b = 0; b < BLOCKS_PER_THREAD; b++) {
      uint cb = blk_start + b;
      if (cb >= DOWN_BLOCKS_PER_ROW) break;
      uint blk_idx = (row0 + r) * DOWN_BLOCKS_PER_ROW + cb;
      device const uchar* blk = data3 + blk_idx * 34u;
      float scale = float(*((device const half*)blk));
      float acc = 0.0f;
      #pragma unroll
      for (uint j = 0; j < 32u; j++) {
        acc += float(as_type<int8_t>(blk[2u + j])) * zchunk[b*32u + j];
      }
      p += scale * acc;
    }
    partial[r] = p;
  }

  // Threadgroup-memory reduction, 8 rows per group, 32-thread lane-sum per row.
  #pragma unroll
  for (uint r = 0; r < ROWS_PER_GROUP; r++) {
    red[tid * ROWS_PER_GROUP + r] = partial[r];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < ROWS_PER_GROUP) {
    float full = 0.0f;
    #pragma unroll
    for (uint i = 0; i < THREADS; i++) {
      full += red[i * ROWS_PER_GROUP + tid];
    }
    data0[row0 + tid] = data1[row0 + tid] + full;
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
    # Grid: HIDDEN/ROWS_PER_GROUP = 3584/7 = 512 threadgroups, 32 threads each.
    sink = UOp.sink(
        UOp.special(512, "gidx0"),
        UOp.special(32, "lidx0"),
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
    # Grid: DIM/ROWS_PER_GROUP = 1024/8 = 128 threadgroups, 32 threads each.
    sink = UOp.sink(
        UOp.special(128, "gidx0"),
        UOp.special(32, "lidx0"),
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

    Emits two custom kernels:
      1. fused_gate_up_q8: z = silu(gate @ rmsnorm(h, norm_w)) * (up @ rmsnorm(h, norm_w))
      2. fused_down_q8:    out = h + down @ z
    """
    assert h.numel() == 1024, f"fused_ffn only supports dim=1024, got {h.numel()}"

    norm_raw = _find_raw_q8_blocks(norm_w)
    gate_raw = _find_raw_q8_blocks(gate_w)
    up_raw = _find_raw_q8_blocks(up_w)
    down_raw = _find_raw_q8_blocks(down_w)
    if any(t is None for t in (norm_raw, gate_raw, up_raw, down_raw)):
        raise NotImplementedError("fused_ffn_with_residual: all weights must trace to a raw uchar buffer")

    # Force h to be materialized ONCE (attention's output), then thread it through
    # both custom_kernels via AFTER to avoid a second contiguous copy.
    # The first custom_kernel call contiguous-copies h internally, but that copy
    # IS the materialization we want -- we then reuse h_after for kernel 2.
    z = Tensor.empty(3584, dtype=dtypes.float, device=h.device)
    z, h_after, *_ = Tensor.custom_kernel(z, h, norm_raw, gate_raw, up_raw, fxn=_gate_up_kernel)

    out_empty = Tensor.empty(h.shape, dtype=dtypes.float, device=h.device)
    out, *_ = Tensor.custom_kernel(out_empty, h_after, z, down_raw, fxn=_down_kernel)

    return out
