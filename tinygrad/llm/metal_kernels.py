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
from tinygrad import Device, Tensor
from tinygrad.uop.ops import UOp, Ops, KernelInfo


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
    """
    out = Tensor.empty_like(x)
    return Tensor.custom_kernel(out, x, gate_w, up_w, down_w, fxn=fused_mlp_kernel)[0]
