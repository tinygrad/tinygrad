import functools, math, pathlib
from tinygrad import Tensor, dtypes, function
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from extra.llama_kernels import alloc_like, compile_hip

@functools.cache
def _residual_kernel(out:UOp, x:UOp, proj:UOp, bias:UOp, moe:UOp) -> UOp:
  rows = math.prod(x.shape[:-1])
  assert out.shape == x.shape == moe.shape and x.shape[-1] == 2880
  assert proj.shape == (rows, 3072) and bias.shape == (2880,)
  assert all(t.dtype == dtypes.bfloat16 for t in (out, x, proj, bias, moe))
  sink = UOp.sink(out.base, x.base, proj.base, bias.base, moe.base,
                  UOp.special(256, "lidx0"), UOp.special(rows, "gidx0"), arg=KernelInfo("gptoss_residual_join_vec"))
  src = (pathlib.Path(__file__).parent/"residual.cpp").read_text()
  lib = compile_hip(src, [f"-DROWS={rows}", "-DREAL_D=2880", "-DPAD_D=3072", "-fno-fast-math"])
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def _residual_backward(gradient:UOp, call:UOp) -> tuple:
  return gradient, None, None, None, gradient

@function(grad_fxn=_residual_backward)
def residual_join(h:Tensor, x:Tensor, proj:Tensor, bias:Tensor, moe:Tensor) -> Tensor:
  out = alloc_like(x.shape, x.dtype, x.device, x.uop.axis)
  return Tensor.custom_kernel(out, x, proj, bias, moe, fxn=_residual_kernel)[0]
