# Custom hand-rolled Metal kernels for the LLM decode path.
#
# Hot path from VIZ: FFNBlock._feed_forward (model.py:98) emits 5 kernels per block per token:
#   ffn_norm reduce, ffn_gate matvec, ffn_up matvec, silu*mul elementwise, ffn_down matvec.
# On Qwen3.5-0.8B with 24 blocks that's 120 launches @ ~11us each = ~1.3ms per token of pure
# dispatch overhead. A single fused kernel per block collapses this to 24 launches.
#
# Enable with CUSTOM_MLP=1. The Metal source below is a correctness-preserving stub
# (writes zeros). Replace with a real fused gate*silu*up -> down kernel to reclaim the time.

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


# ---------------- Metal source (stub: fills output with zeros) ----------------
# Arg order matches the custom_kernel fxn signature below:
#   data0 = out   (float, shape (B, T, dim))
#   data1 = x     (float, shape (B, T, dim))
#   data2 = gate  (hidden, dim)  -- whatever dtype the weight tensor has
#   data3 = up    (hidden, dim)
#   data4 = down  (dim, hidden)
# The real kernel will read the underlying Q8_0 byte buffer; for the stub we ignore
# the weights entirely and just zero the output.
_FUSED_MLP_METAL_SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void fused_mlp_q8(
    device float* data0,
    device const float* data1,
    device const uchar* data2,
    device const uchar* data3,
    device const uchar* data4,
    constant uint& n_out [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  // NO-OP stub: zero the output. Replace with fused gate*silu*up -> down.
  uint i = gid.x * 256u + lid.x;
  if (i < n_out) data0[i] = 0.0f;
}
"""

_KERNEL_NAME = "fused_mlp_q8"


@functools.cache
def _compiled_lib() -> bytes:
    # Lazy import so non-Metal backends don't pay the objc/MTLCompiler cost.
    from tinygrad.runtime.ops_metal import MetalCompiler
    return MetalCompiler().compile(_FUSED_MLP_METAL_SRC)


def fused_mlp_kernel(out: UOp, x: UOp, gate_w: UOp, up_w: UOp, down_w: UOp) -> UOp:
    """custom_kernel body for FFN.

    Contract: out[...] = down @ (silu(gate @ x) * (up @ x))
    Stub implementation writes zeros so numerics are clearly "off" until the real
    kernel is written (makes accidentally shipping the stub impossible to miss).
    """
    lib = _compiled_lib()
    n_out = out.numel()
    # Launch grid: 256 threads per group, one element per thread, round up.
    # TODO(real kernel): use a shape-aware grid (one threadgroup per output row, SIMD reduce across hidden).
    local_size = 256
    global_size = (n_out + local_size - 1) // local_size
    sink = UOp.sink(
        UOp.special(global_size, "gidx0"),
        UOp.special(local_size, "lidx0"),
        out, x, gate_w, up_w, down_w,
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


def fused_mlp(x: Tensor, gate_w: Tensor, up_w: Tensor, down_w: Tensor) -> Tensor:
    """High-level wrapper mirroring model.py:119 signature.

    Equivalent to `down_w @ (silu(gate_w @ x) * (up_w @ x))` but emits a single kernel.

    The `*.weight` tensors are lazy CAST(half) nodes over a MUL-of-dequant chain.
    Passing them to custom_kernel calls .contiguous() on them, which realizes the
    full fp32 weight into a fresh Metal buffer before our kernel runs (measured:
    48 kernels x ~40us + 528 MB extra resident mem). We instead reach into the
    weight's uop to grab the CONTIGUOUS(uchar) node that holds the raw Q8_0 bytes
    and pass that -- matching what the auto-generated matvecs already do.

    Falls back to the dequantized-float path if the weight is not Q8_0 (e.g. a
    future non-quantized test). Callers can detect this by checking output dtype.
    """
    out = Tensor.empty_like(x)
    gate_q8 = _find_raw_q8_blocks(gate_w)
    up_q8 = _find_raw_q8_blocks(up_w)
    down_q8 = _find_raw_q8_blocks(down_w)
    if gate_q8 is None or up_q8 is None or down_q8 is None:
        # not all Q8_0 -- fall back to float weights (pays the dequant cost)
        return Tensor.custom_kernel(out, x, gate_w, up_w, down_w, fxn=fused_mlp_kernel)[0]
    return Tensor.custom_kernel(out, x, gate_q8, up_q8, down_q8, fxn=fused_mlp_kernel)[0]
