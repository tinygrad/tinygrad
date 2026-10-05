import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from extra.llama_kernels import alloc_like, compile_hip

@functools.cache
def _combine_forward(out:UOp, z:UOp, dest_row:UOp, weights:UOp) -> UOp:
  groups, tokens, hidden = out.shape
  rows = z.shape[1]
  assert hidden == 2880 and z.shape == (groups, rows, hidden)
  assert dest_row.shape == (groups, tokens*4) and weights.shape == (groups, tokens, 4)
  assert out.dtype == z.dtype == dtypes.bfloat16 and dest_row.dtype == dtypes.int32 and weights.dtype == dtypes.float32
  sink = UOp.sink(out.base, z.base, dest_row.base, weights.base, UOp.special(256, "lidx0"),
                  UOp.special(groups*tokens, "gidx0"), arg=KernelInfo("gptoss_combine_forward"))
  src = (pathlib.Path(__file__).parent/"forward.cpp").read_text()
  lib = compile_hip(src, [f"-DTOKENS={tokens}", f"-DROWS={rows}", "-fno-fast-math"])
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def _combine_backward(gradient:UOp, kernel:UOp) -> tuple:
  from extra.gemm.moe_routing import grouped_gather_rows
  z, dest_row, weights = (Tensor(u) for u in kernel.src[2:5])
  groups, tokens, k = weights.shape
  sel = grouped_gather_rows(z, dest_row, groups).reshape(groups, tokens, k, z.shape[-1])
  reference = (sel * weights.unsqueeze(-1).cast(sel.dtype)).sum(2)
  dz, dw = reference.gradient(z, weights, gradient=Tensor(gradient))
  return None, dz.uop, None, dw.uop

def fused_combine(z:Tensor, dest_row:Tensor, weights:Tensor) -> Tensor:
  out = alloc_like((*weights.shape[:2], z.shape[-1]), z.dtype, z.device, z.uop.axis)
  return Tensor.custom_kernel(out, z, dest_row, weights, fxn=_combine_forward, grad_fxn=_combine_backward)[0]
