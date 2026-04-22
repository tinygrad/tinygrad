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
constant constexpr float RMS_EPS = 1e-6f;
constant constexpr uint WARPS = 4u;                             // 4 warps/threadgroup
constant constexpr uint SIMD = 32u;                             // Apple SIMD width
constant constexpr uint THREADS = WARPS * SIMD;                 // 128 threads/TG
constant constexpr uint ROWS_PER_GROUP = 8u;                    // emit 8 output rows per TG (matches baseline acc0[8])
constant constexpr uint BLOCKS_PER_ROW = DIM / 32u;             // 32 (Q8 blocks per dim)
constant constexpr uint BLOCKS_PER_WARP = BLOCKS_PER_ROW / WARPS; // 8 (cols per warp)

// Grid: HIDDEN/ROWS_PER_GROUP = 3584/8 = 448 threadgroups * 128 threads.
// Each TG emits 8 output rows. The 1024-wide reduction splits as
//   4 warps (lidx1) each owning a stripe of 8 Q8 blocks (256 cols),
//   32 SIMD lanes (lidx0) within each warp each do dot products across all 8 rows.
// Final reduce: threadgroup memory combines 4 warps' partials.
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
  // Cross-warp reduction buffer: 4 warps x 8 rows x 2 (gate+up).
  threadgroup float redg[WARPS * ROWS_PER_GROUP];
  threadgroup float redu[WARPS * ROWS_PER_GROUP];

  uint tid = lidx1 * SIMD + lidx0;

  // ---- RMSNorm ----
  // Each thread handles DIM/THREADS = 8 floats.
  float sq = 0.0f;
  #pragma unroll
  for (uint i = 0; i < DIM; i += THREADS) {
    float v = data1[i + tid];
    sq += v * v;
  }
  // simd_sum reduces within warp; then we need cross-warp reduction.
  float warp_sq = simd_sum(sq);
  if (lidx0 == 0) redg[lidx1] = warp_sq;  // reuse redg for this, 4 floats
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total_sq = redg[0] + redg[1] + redg[2] + redg[3];
  float inv_rms = rsqrt(total_sq / float(DIM) + RMS_EPS);

  // Write x_norm = h * inv_rms * norm_w.
  #pragma unroll
  for (uint i = 0; i < DIM; i += THREADS) {
    x_norm[i + tid] = data1[i + tid] * inv_rms * data2[i + tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- Gate/Up matvec ----
  // Thread (lidx0, lidx1) owns column-block `lidx1 * BLOCKS_PER_WARP + something`.
  // Actually: 32 lanes x 8 blocks per warp = 256 cols per warp. Each lane does 1 block across 8 iters.
  // Simpler: the warp covers 8 blocks (256 cols), 32 lanes split by row within the block.
  // Let's pick: each thread handles 1 Q8 block at index (lidx1 * SIMD + lidx0) / some mapping.
  // Actually the cleanest: thread owns BLOCK (lidx1*8 + inner_block) for some inner_block from 0..7.
  // With 32 lanes and 8 blocks/warp, 4 lanes per block => 4-way split within a block.
  //
  // Simpler/better decomposition: lidx1 picks one of 4 stripes of 8 blocks each; lidx0 picks
  // one of 32 "rows" for the 8 output rows... no wait, output rows = 8 per TG.
  //
  // Let me just do: each thread owns ONE Q8 block (col-block lidx1*8 + lidx0/4 = 4-16..32 range).
  // Hmm, 32 lanes across 8 blocks/warp means 4 lanes per block. 4 lanes do partial dot on 8-column slice each.
  //
  // Rather than get fancy: each thread handles column-block (lidx1*SIMD + lidx0) / ??? of 32 total blocks.
  // 128 threads, 32 total col-blocks -> 4 threads per col-block -> each thread does 8 cols of a block.
  // Let: col_block = tid / 4, col_within = tid % 4  (8 cols per thread).
  //
  // For ROWS_PER_GROUP=8, each thread computes 8 partial dots (one per output row).
  uint cb = tid / 4u;             // 0..31 (col block index within the 1024-wide input)
  uint cw = (tid & 3u) * 8u;      // 0, 8, 16, or 24 (col offset within the 32-col block)

  // Preload 8 floats of x_norm for this thread's sub-block.
  float xchunk[8];
  #pragma unroll
  for (uint j = 0; j < 8u; j++) xchunk[j] = x_norm[cb * 32u + cw + j];

  // Compute 8 partial dots, one per output row.
  float gp[ROWS_PER_GROUP];
  float up[ROWS_PER_GROUP];
  uint row0 = gid * ROWS_PER_GROUP;
  #pragma unroll
  for (uint r = 0; r < ROWS_PER_GROUP; r++) {
    uint blk_idx = (row0 + r) * BLOCKS_PER_ROW + cb;
    device const uchar* gblk = data3 + blk_idx * 34u;
    device const uchar* ublk = data4 + blk_idx * 34u;
    float gscale = float(*((device const half*)gblk));
    float uscale = float(*((device const half*)ublk));
    float gacc = 0.0f, uacc = 0.0f;
    #pragma unroll
    for (uint j = 0; j < 8u; j++) {
      float x = xchunk[j];
      gacc += float(as_type<int8_t>(gblk[2u + cw + j])) * x;
      uacc += float(as_type<int8_t>(ublk[2u + cw + j])) * x;
    }
    gp[r] = gscale * gacc;
    up[r] = uscale * uacc;
  }

  // Reduce: 128 threads -> 8 output rows. Each row sums 128 partials.
  // Use threadgroup memory: each thread writes its 8 partials, then lanes 0..7 reduce 128-way each.
  // redg/redu need 128 * 8 = 1024 each -> too big for small threadgroup mem. Let's do it differently:
  // simd_sum across each warp (reduces 32 partials -> 4 warp-level partials), then cross-warp sum via TG mem.
  #pragma unroll
  for (uint r = 0; r < ROWS_PER_GROUP; r++) {
    float gw = simd_sum(gp[r]);
    float uw = simd_sum(up[r]);
    if (lidx0 == 0) {
      redg[lidx1 * ROWS_PER_GROUP + r] = gw;
      redu[lidx1 * ROWS_PER_GROUP + r] = uw;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Final cross-warp reduce: 8 lanes (one per row) sum 4 warp partials.
  if (lidx1 == 0 && lidx0 < ROWS_PER_GROUP) {
    float g_full = redg[0 * ROWS_PER_GROUP + lidx0] + redg[1 * ROWS_PER_GROUP + lidx0]
                 + redg[2 * ROWS_PER_GROUP + lidx0] + redg[3 * ROWS_PER_GROUP + lidx0];
    float u_full = redu[0 * ROWS_PER_GROUP + lidx0] + redu[1 * ROWS_PER_GROUP + lidx0]
                 + redu[2 * ROWS_PER_GROUP + lidx0] + redu[3 * ROWS_PER_GROUP + lidx0];
    float silu_g = g_full / (1.0f + exp(-g_full));
    data0[row0 + lidx0] = silu_g * u_full;
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
// 128 threads cooperate on the 3584-wide reduction. 112 Q8 blocks / 128 threads
// doesn't divide evenly; instead: each WARP handles 28 blocks (28*4=112), and
// each of 32 lanes in a warp covers 28/32 * 32 cols = interleaved...
// Simpler: 4 warps, each warp owns 28 blocks. Within a warp, 32 lanes cooperate
// on those 28 blocks. 28 blocks = 896 cols, split as 32 lanes x 28 cols each.
// That's awkward. Let me do: 4 warps x 32 blocks = 128 blocks total, mask last 16 off.
// Each thread (lane in warp) handles BLOCKS_PER_THREAD = 1 block (32 cols).
// 4 warps * 32 lanes = 128 threads, each with 1 block -> covers 128 blocks (16 OOB).
constant constexpr uint BLOCKS_PER_THREAD = 1u;

kernel void fused_down_q8(
    device float* data0,                      // out[DIM]
    device const float* data1,                // h[DIM] (residual)
    device const float* data2,                // z[HIDDEN]
    device const uchar* data3,                // down Q8 blocks
    uint gid [[threadgroup_position_in_grid]],
    uint lidx0 [[thread_index_in_simdgroup]],
    uint lidx1 [[simdgroup_index_in_threadgroup]]) {
  // Cross-warp reduction buffer: 4 warps * 8 rows = 32.
  threadgroup float red[WARPS * ROWS_PER_GROUP];

  uint tid = lidx1 * SIMD + lidx0;
  uint cb = tid;  // block-column index (0..127, valid if < 112)
  bool valid = cb < BLOCKS_PER_ROW;

  // Preload 32 floats of z for this thread's block.
  float zchunk[32];
  if (valid) {
    uint base = cb * 32u;
    #pragma unroll
    for (uint j = 0; j < 32u; j++) zchunk[j] = data2[base + j];
  } else {
    #pragma unroll
    for (uint j = 0; j < 32u; j++) zchunk[j] = 0.0f;
  }

  // Compute 8 partial dots (one per output row).
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

  // Warp-level reduce (32 partials -> 1 per warp per row), then cross-warp via TG mem.
  #pragma unroll
  for (uint r = 0; r < ROWS_PER_GROUP; r++) {
    float warp_partial = simd_sum(partial[r]);
    if (lidx0 == 0) red[lidx1 * ROWS_PER_GROUP + r] = warp_partial;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Final cross-warp reduce: 8 lanes (one per row) in warp 0 sum 4 warp partials.
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
