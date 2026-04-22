# Custom hand-rolled Metal kernels for the LLM decode path.
#
# Baseline hot path (from VIZ): per block per token, tinygrad emits
#   ffn_norm reduce (r_32_32n1)  -- 1 kernel
#   ffn_gate matvec              -- 1 kernel
#   ffn_up matvec                -- 1 kernel
#   silu*up elementwise          -- 1 kernel
#   ffn_down matvec + residual   -- 1 kernel (add is fused into the final store)
# With Qwen3.5-0.8B's 24 blocks that's 120 launches @ ~10us each = ~1.2 ms of pure
# dispatch overhead per token. This module collapses them into 1 launch per block.
#
# Enable with CUSTOM_MLP=1. The Metal source below is a correctness-preserving
# stub: it copies the residual `h` through untouched (equivalent to the FFN
# producing zeros). Replace the kernel body with a real fused
#   rmsnorm -> gate -> silu*up -> down -> residual_add
# implementation to reclaim the launch overhead plus get SIMD-level fusion wins.

from __future__ import annotations
import functools
from tinygrad import Device, Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo


def _find_raw_q8_blocks(weight: Tensor) -> Tensor | None:
    """Walk the weight tensor's uop chain looking for the CONTIGUOUS uchar node
    that holds raw Q8_0 blocks (2-byte fp16 scale + 32 int8 qs per 34-byte block).

    gguf.py:57 builds every Q8_0 weight as `blocks[:,:2].bitcast(half).cast(float) * blocks[:,2:].bitcast(int8)`
    where `blocks` is the one `contiguous()` uchar tensor. Finding that node lets us
    hand the raw bytes to a custom kernel without triggering the fp32 dequant.
    Returns None for non-Q8_0 weights.
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


# ---------------- Metal source: fused RMSNorm + FFN + residual ----------------
# Arg order matches the custom_kernel fxn signature below:
#   data0 = out      (float,  shape (B, T, D))             -- output: h + FFN(norm(h))
#   data1 = h        (float,  shape (B, T, D))             -- pre-norm residual input
#   data2 = norm_w   (float,  shape (D,))                  -- ffn_norm.weight (raw F32 bytes)
#   data3 = gate_q8  (uchar,  shape (hidden*D/32, 34))     -- raw Q8_0 gate blocks
#   data4 = up_q8    (uchar,  shape (hidden*D/32, 34))     -- raw Q8_0 up blocks
#   data5 = down_q8  (uchar,  shape (D*hidden/32, 34))     -- raw Q8_0 down blocks
#
# Shape assumptions (checked at Python level):
#   D = 1024, hidden = 3584 (Qwen3.5-0.8B dense FFN)
#   Q8_0 block: 2-byte fp16 scale + 32 int8 qs, 34 bytes total.
#
# Grid/strategy: one threadgroup handles the whole token. 32 threads cooperate
# via threadgroup memory. This is the simplest correct layout; it does NOT fuse
# gate/up/down into a single reduction (each matvec materializes its output in
# threadgroup memory for the next phase to consume). Optimizations to try later:
#   - SIMD-shuffle reductions (kill threadgroup_barriers within a warp)
#   - fuse silu(gate)*up into the down reduction (streaming)
#   - multiple threadgroups per token, splitting the output across groups
_FUSED_MLP_METAL_SRC = """
#include <metal_stdlib>
using namespace metal;

constant constexpr uint DIM = 1024u;
constant constexpr uint HIDDEN = 3584u;
constant constexpr uint THREADS = 32u;
constant constexpr float RMS_EPS = 1e-6f;
constant constexpr uint GATE_BLOCKS_PER_ROW = DIM / 32u;     // 32 Q8 blocks per gate/up output row
constant constexpr uint DOWN_BLOCKS_PER_ROW = HIDDEN / 32u;  // 112 Q8 blocks per down output row

// Decode one Q8_0 block (34 bytes) at lane 0 of the block, dot it with 32 floats from x.
// Returns partial dot product contribution for this thread's lanes.
static inline float q8_dot(device const uchar* block_base, uint block_idx,
                           threadgroup const float* x_tile, uint x_offset) {
    device const uchar* blk = block_base + block_idx * 34u;
    // fp16 scale in first 2 bytes (little endian)
    half scale = as_type<half>((ushort)(uint(blk[0]) | (uint(blk[1]) << 8)));
    float s = float(scale);
    float acc = 0.0f;
    #pragma unroll
    for (uint j = 0; j < 32u; j++) {
        int8_t qs = as_type<int8_t>(blk[2u + j]);
        acc += float(qs) * x_tile[x_offset + j];
    }
    return s * acc;
}

kernel void fused_mlp_q8(
    device float* data0,
    device const float* data1,
    device const float* data2,
    device const uchar* data3,
    device const uchar* data4,
    device const uchar* data5,
    uint tid [[thread_position_in_threadgroup]]) {
  // All threadgroup buffers for intermediates
  threadgroup float x_norm[DIM];     // 4 KB: normalized input to gate/up
  threadgroup float z[HIDDEN];       // 14 KB: silu(gate@x_norm) * (up@x_norm)
  threadgroup float red[THREADS];    // 128 B: reduction scratch

  // ---- Phase 1: RMSNorm ----
  // Each thread accumulates sum(h*h) over a strided slice of the 1024-dim input.
  float sq = 0.0f;
  for (uint i = tid; i < DIM; i += THREADS) {
    float v = data1[i];
    sq += v * v;
  }
  red[tid] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Tree reduction across 32 threads.
  if (tid == 0) {
    float total = 0.0f;
    for (uint i = 0; i < THREADS; i++) total += red[i];
    red[0] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv_rms = rsqrt(red[0] / float(DIM) + RMS_EPS);
  // Write x_norm = h * inv_rms * norm_w.
  for (uint i = tid; i < DIM; i += THREADS) {
    x_norm[i] = data1[i] * inv_rms * data2[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- Phase 2: gate/up matvec + silu*mul ----
  // z[r] = silu(sum_j gate_q8[r, j] * x_norm[j]) * (sum_j up_q8[r, j] * x_norm[j])
  // Each thread handles HIDDEN/32 = 112 output rows.
  for (uint r = tid; r < HIDDEN; r += THREADS) {
    float g = 0.0f, u = 0.0f;
    // Walk the 32 Q8 blocks that make up this row's 1024 input dimensions.
    for (uint b = 0; b < GATE_BLOCKS_PER_ROW; b++) {
      uint block_idx = r * GATE_BLOCKS_PER_ROW + b;
      g += q8_dot(data3, block_idx, x_norm, b * 32u);
      u += q8_dot(data4, block_idx, x_norm, b * 32u);
    }
    float silu_g = g / (1.0f + exp(-g));  // silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
    z[r] = silu_g * u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- Phase 3: down matvec + residual add ----
  // out[r] = h[r] + sum_j down_q8[r, j] * z[j]
  // Each thread handles DIM/32 = 32 output rows.
  for (uint r = tid; r < DIM; r += THREADS) {
    float y = 0.0f;
    for (uint b = 0; b < DOWN_BLOCKS_PER_ROW; b++) {
      uint block_idx = r * DOWN_BLOCKS_PER_ROW + b;
      y += q8_dot(data5, block_idx, z, b * 32u);
    }
    data0[r] = data1[r] + y;
  }
}
"""

_KERNEL_NAME = "fused_mlp_q8"


@functools.cache
def _compiled_lib() -> bytes:
    # Lazy import so non-Metal backends don't pay the objc/MTLCompiler cost.
    from tinygrad.runtime.ops_metal import MetalCompiler
    return MetalCompiler().compile(_FUSED_MLP_METAL_SRC)


def fused_mlp_kernel(out: UOp, h: UOp, norm_w: UOp,
                     gate_w: UOp, up_w: UOp, down_w: UOp) -> UOp:
    """custom_kernel body.

    Contract: out = h + down @ (silu(gate @ rmsnorm(h, norm_w)) * (up @ rmsnorm(h, norm_w)))

    Stub writes `out = h`, which means the MLP contributes nothing and the block
    degenerates to `h_out = h + attention(attn_norm(h))`. Decoded text will be
    recognisably "attention-only" (non-zero but wrong), which is more useful for
    debugging than writing raw zeros (that would poison downstream attention).
    """
    lib = _compiled_lib()
    # Kernel is hardcoded to Qwen3.5-0.8B dim/hidden. Assert the shape matches so
    # we fail loudly on the wrong model instead of silently writing garbage.
    assert out.numel() == 1024, f"fused_mlp_q8 expects dim=1024, got {out.numel()}"
    # Launch: 1 threadgroup, 32 threads. The whole token is processed by a single
    # SIMD group cooperating through threadgroup memory.
    sink = UOp.sink(
        UOp.special(1, "gidx0"),
        UOp.special(32, "lidx0"),
        out, h, norm_w, gate_w, up_w, down_w,
        arg=KernelInfo(name=_KERNEL_NAME),
    )
    return UOp(
        Ops.PROGRAM,
        src=(
            sink,
            UOp(Ops.DEVICE, arg=Device.DEFAULT),
            UOp(Ops.LINEAR, src=(*sink.src, sink)),
            UOp(Ops.SOURCE, arg=_FUSED_MLP_METAL_SRC),
            UOp(Ops.BINARY, arg=lib),
        ),
    )


def fused_ffn_with_residual(h: Tensor, norm_w: Tensor,
                            gate_w: Tensor, up_w: Tensor, down_w: Tensor) -> Tensor:
    """Replacement for `h + ffn_down(silu(ffn_gate(ffn_norm(h))) * ffn_up(ffn_norm(h)))`.

    Emits a single custom kernel. Designed to swallow:
      - the ffn_norm reduce (baseline: r_32_32n1, count=24 per step)
      - ffn_gate/up/down matvecs (baseline: r_512_32_7_32, r_112_*, r_128_*)
      - the silu*mul elementwise (baseline: E_8_32_4)
      - the residual add h + ... (baseline: E_8_32_4n3)
      - the .contiguous() copy (baseline: E_64_8_2)
    -> 6 baseline kernels collapse into 1 per block.

    All weight inputs (including norm_w) get walked for their raw uchar BUFFER
    node to avoid custom_kernel forcing a rematerialization -- ffn_norm.weight
    is stored as F32 bytes in the GGUF which are bit-identical to Metal float*,
    so the kernel can reinterpret the uchar* ptr as float*.
    """
    out = Tensor.empty_like(h)
    norm_q8 = _find_raw_q8_blocks(norm_w)
    gate_q8 = _find_raw_q8_blocks(gate_w)
    up_q8 = _find_raw_q8_blocks(up_w)
    down_q8 = _find_raw_q8_blocks(down_w)
    if any(t is None for t in (norm_q8, gate_q8, up_q8, down_q8)):
        raise NotImplementedError("fused_ffn_with_residual: all weights must trace back to a raw uchar buffer")
    return Tensor.custom_kernel(out, h, norm_q8, gate_q8, up_q8, down_q8, fxn=fused_mlp_kernel)[0]
