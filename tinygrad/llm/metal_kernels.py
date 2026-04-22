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


# ---------------- Metal source (stub: out = h, i.e. FFN contributes 0) ----------------
# Arg order matches the custom_kernel fxn signature below:
#   data0 = out      (float,  shape (B, T, D))             -- output: h + FFN(norm(h))
#   data1 = h        (float,  shape (B, T, D))             -- pre-norm residual input
#   data2 = norm_w   (float,  shape (D,))                  -- ffn_norm.weight (fp16 view, small)
#   data3 = gate_q8  (uchar,  shape (hidden*D/32, 34))     -- raw Q8_0 gate blocks
#   data4 = up_q8    (uchar,  shape (hidden*D/32, 34))     -- raw Q8_0 up blocks
#   data5 = down_q8  (uchar,  shape (D*hidden/32, 34))     -- raw Q8_0 down blocks
# Stub copies h -> out (makes the FFN act as identity so the attention path keeps
# running; useful for debugging model structure while the real kernel is WIP).
_FUSED_MLP_METAL_SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void fused_mlp_q8(
    device float* data0,
    device const float* data1,
    device const float* data2,
    device const uchar* data3,
    device const uchar* data4,
    device const uchar* data5,
    constant uint& n_out [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  // STUB: out = h (FFN contributes zero). Replace with:
  //   1. RMSNorm:    x = h * rsqrt(mean(h*h) + eps) * norm_w
  //   2. gate_mv:    g = dequant_q8(data3) @ x            // (hidden,)
  //   3. up_mv:      u = dequant_q8(data4) @ x            // (hidden,)
  //   4. silu*mul:   z = silu(g) * u                      // (hidden,)
  //   5. down_mv:    y = dequant_q8(data5) @ z            // (dim,)
  //   6. residual:   out = h + y                          // (dim,)
  // Fuse 1+2+3 by re-reading h once per block-tile and holding x in threadgroup mem.
  // Fuse 4+5+6 by streaming z through the reduce in ffn_down and adding h at store time.
  uint i = gid.x * 256u + lid.x;
  if (i < n_out) data0[i] = data1[i];
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
    n_out = out.numel()
    # Launch grid: one thread per output element, 256 per group, round up.
    # TODO(real kernel): one threadgroup per output row, SIMD-shuffle reduce across the hidden axis.
    local_size = 256
    global_size = (n_out + local_size - 1) // local_size
    sink = UOp.sink(
        UOp.special(global_size, "gidx0"),
        UOp.special(local_size, "lidx0"),
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

    Weights are passed as their raw Q8_0 block tensors (see _find_raw_q8_blocks).
    """
    out = Tensor.empty_like(h)
    gate_q8 = _find_raw_q8_blocks(gate_w)
    up_q8 = _find_raw_q8_blocks(up_w)
    down_q8 = _find_raw_q8_blocks(down_w)
    if gate_q8 is None or up_q8 is None or down_q8 is None:
        # non-Q8_0 model -- bail to the slow path in the caller
        raise NotImplementedError("fused_ffn_with_residual: weights must be Q8_0")
    return Tensor.custom_kernel(out, h, norm_w, gate_q8, up_q8, down_q8, fxn=fused_mlp_kernel)[0]
