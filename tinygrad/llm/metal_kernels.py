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
constant constexpr uint GATE_BLOCKS_PER_ROW = DIM / 32u;      // 32 blocks per row
constant constexpr uint N_GROUPS = HIDDEN / THREADS;          // 112

// Dequant-and-dot one Q8_0 block (34 bytes) against 32 floats of x.
static inline float q8_block_dot(device const uchar* blk, thread const float* xchunk) {
    half scale = as_type<half>((ushort)(uint(blk[0]) | (uint(blk[1]) << 8)));
    float acc = 0.0f;
    #pragma unroll
    for (uint j = 0; j < 32u; j++) {
        int8_t qs = as_type<int8_t>(blk[2u + j]);
        acc += float(qs) * xchunk[j];
    }
    return float(scale) * acc;
}

kernel void fused_gate_up_q8(
    device float* data0,            // z[HIDDEN]
    device const float* data1,      // h[DIM]
    device const float* data2,      // norm_w[DIM]
    device const uchar* data3,      // gate Q8 blocks
    device const uchar* data4,      // up Q8 blocks
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
  // Threadgroup-shared x_norm (reused across all THREADS output rows in this group).
  threadgroup float x_norm[DIM];

  // ---- Phase 1: RMSNorm (computed once per threadgroup) ----
  // Each thread accumulates partial sum(h^2) over a strided slice.
  float sq = 0.0f;
  for (uint i = tid; i < DIM; i += THREADS) {
    float v = data1[i];
    sq += v * v;
  }
  // Reduce across the 32 threads (= 1 SIMD group). simd_sum is a warp reduction.
  float total_sq = simd_sum(sq);
  float inv_rms = rsqrt(total_sq / float(DIM) + RMS_EPS);

  // Write x_norm = h * inv_rms * norm_w (parallel, strided).
  for (uint i = tid; i < DIM; i += THREADS) {
    x_norm[i] = data1[i] * inv_rms * data2[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- Phase 2: gate/up matvec + silu*mul ----
  // This threadgroup owns rows [gid*THREADS .. gid*THREADS+31] of the output.
  // Each thread owns exactly one output row r = gid*THREADS + tid.
  uint row = gid * THREADS + tid;
  float g = 0.0f, u = 0.0f;
  for (uint b = 0; b < GATE_BLOCKS_PER_ROW; b++) {
    uint blk_idx = row * GATE_BLOCKS_PER_ROW + b;
    // Load 32 floats of x_norm corresponding to this block's input cols.
    float xchunk[32];
    #pragma unroll
    for (uint j = 0; j < 32u; j++) xchunk[j] = x_norm[b * 32u + j];
    g += q8_block_dot(data3 + blk_idx * 34u, xchunk);
    u += q8_block_dot(data4 + blk_idx * 34u, xchunk);
  }
  float silu_g = g / (1.0f + exp(-g));
  data0[row] = silu_g * u;
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
constant constexpr uint DOWN_BLOCKS_PER_ROW = HIDDEN / 32u;  // 112 blocks per row

static inline float q8_block_dot(device const uchar* blk, thread const float* xchunk) {
    half scale = as_type<half>((ushort)(uint(blk[0]) | (uint(blk[1]) << 8)));
    float acc = 0.0f;
    #pragma unroll
    for (uint j = 0; j < 32u; j++) {
        int8_t qs = as_type<int8_t>(blk[2u + j]);
        acc += float(qs) * xchunk[j];
    }
    return float(scale) * acc;
}

kernel void fused_down_q8(
    device float* data0,            // out[DIM]
    device const float* data1,      // h[DIM]  (residual)
    device const float* data2,      // z[HIDDEN]
    device const uchar* data3,      // down Q8 blocks
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  uint row = gid * THREADS + tid;  // row index in [0, DIM)
  float y = 0.0f;
  for (uint b = 0; b < DOWN_BLOCKS_PER_ROW; b++) {
    uint blk_idx = row * DOWN_BLOCKS_PER_ROW + b;
    float xchunk[32];
    #pragma unroll
    for (uint j = 0; j < 32u; j++) xchunk[j] = data2[b * 32u + j];
    y += q8_block_dot(data3 + blk_idx * 34u, xchunk);
  }
  data0[row] = data1[row] + y;
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
    # Grid: HIDDEN/THREADS = 112 threadgroups, 32 threads each.
    sink = UOp.sink(
        UOp.special(112, "gidx0"),
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
    # Grid: DIM/THREADS = 32 threadgroups, 32 threads each.
    sink = UOp.sink(
        UOp.special(32, "gidx0"),
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
    # Flatten (B, T, D) -> (D,) since benchmark runs B=T=1.
    h_flat = h.reshape(-1)
    assert h_flat.numel() == 1024, f"fused_ffn only supports dim=1024, got {h_flat.numel()}"

    norm_raw = _find_raw_q8_blocks(norm_w)
    gate_raw = _find_raw_q8_blocks(gate_w)
    up_raw = _find_raw_q8_blocks(up_w)
    down_raw = _find_raw_q8_blocks(down_w)
    if any(t is None for t in (norm_raw, gate_raw, up_raw, down_raw)):
        raise NotImplementedError("fused_ffn_with_residual: all weights must trace to a raw uchar buffer")

    z = Tensor.empty(3584, dtype=dtypes.float, device=h.device)
    # Multi-output: indices returned are [out, *inputs_after_kernel]. We keep the
    # returned `h_after` so the second kernel reuses the already-realized buffer
    # instead of forcing another `.contiguous()` copy (which costs ~13us x 24 blocks).
    z, h_after, *_ = Tensor.custom_kernel(z, h_flat, norm_raw, gate_raw, up_raw, fxn=_gate_up_kernel)

    out_flat = Tensor.empty(1024, dtype=dtypes.float, device=h.device)
    out_flat, *_ = Tensor.custom_kernel(out_flat, h_after, z, down_raw, fxn=_down_kernel)

    return out_flat.reshape(h.shape)
