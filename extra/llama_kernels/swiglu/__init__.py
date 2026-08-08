import functools, math
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, KernelInfo
from tinygrad.renderer import Estimates
from extra.llama_kernels import alloc_like

LOG2_E = 1.4426950408889634

def _shape(x:UOp) -> tuple[int, int, int]:
  rows, two_hidden = math.prod(x.shape[:-1]), x.shape[-1]
  assert two_hidden % 2 == 0
  return rows, two_hidden // 2, rows * two_hidden // 2

@functools.cache
def _swiglu_forward(dst:UOp, packed:UOp) -> UOp:
  rows, hidden, count = _shape(packed)
  dst, packed = dst.reshape(count), packed.reshape(rows, 2 * hidden)
  idx = UOp.range(count, 0)
  row, col = idx // hidden, idx % hidden
  act, gate = packed[row, col].cast(dtypes.float), packed[row, col + hidden].cast(dtypes.float)
  sig = (1.0 + (act * -LOG2_E).exp2()).reciprocal()
  return dst[idx].store((act * sig * gate).cast(dst.dtype)).end(idx).sink(
    arg=KernelInfo(f"swiglu_fwd_{count}", estimates=Estimates(ops=5 * count, mem=6 * count)))

@functools.cache
def _swiglu_backward(dst:UOp, packed:UOp, grad:UOp) -> UOp:
  rows, hidden, count = _shape(packed)
  dst, packed, grad = dst.reshape(rows, 2 * hidden), packed.reshape(rows, 2 * hidden), grad.reshape(count)
  idx = UOp.range(count, 0)
  row, col = idx // hidden, idx % hidden
  act, gate, upstream = (packed[row, col].cast(dtypes.float), packed[row, col + hidden].cast(dtypes.float), grad[idx].cast(dtypes.float))
  sig = (1.0 + (act * -LOG2_E).exp2()).reciprocal()
  silu = act * sig
  write_act = dst[row, col].store((upstream * sig * (1.0 + act * (1.0 - sig)) * gate).cast(dst.dtype))
  write_gate = dst.after(write_act)[row, col + hidden].store((upstream * silu).cast(dst.dtype))
  return write_gate.end(idx).sink(arg=KernelInfo(f"swiglu_bwd_{count}", estimates=Estimates(ops=10 * count, mem=10 * count)))

def _swiglu_gradient(gradient:UOp, kernel:UOp) -> tuple[None, UOp]:
  _, packed = kernel.src[1:]
  axis = packed.axis if isinstance(packed.device, tuple) else None
  dst = alloc_like(packed.shape, packed.dtype, packed.device, axis)
  out = Tensor.custom_kernel(dst, Tensor(packed, device=packed.device), Tensor(gradient, device=packed.device), fxn=_swiglu_backward)[0]
  return None, out.uop

def swiglu(packed:Tensor) -> Tensor:
  assert packed.dtype == dtypes.bfloat16 and packed.ndim >= 2, f"expected BF16 packed activations, got {packed.dtype} {packed.shape}"
  assert packed.shape[-1] % 2 == 0, f"packed SwiGLU dimension must be even, got {packed.shape[-1]}"
  axis = packed.uop.axis if isinstance(packed.device, tuple) else None
  dst = alloc_like((*packed.shape[:-1], packed.shape[-1] // 2), packed.dtype, packed.device, axis)
  return Tensor.custom_kernel(dst, packed, fxn=_swiglu_forward, grad_fxn=_swiglu_gradient)[0]
