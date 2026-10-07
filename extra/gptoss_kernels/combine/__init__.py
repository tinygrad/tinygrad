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

@functools.cache
def _combine_weights_backward(out:UOp, z:UOp, gradient:UOp, dest_row:UOp) -> UOp:
  groups, tokens, k = out.shape
  rows = z.shape[1]
  assert k == 4 and z.shape == (groups, rows, 2880) and gradient.shape == (groups, tokens, 2880)
  assert dest_row.shape == (groups, tokens*4) and dest_row.dtype == dtypes.int32
  assert out.dtype == dtypes.float32 and z.dtype == gradient.dtype == dtypes.bfloat16
  sink = UOp.sink(out.base, z.base, gradient.base, dest_row.base, UOp.special(256, "lidx0"),
                  UOp.special(groups*tokens, "gidx0"), arg=KernelInfo("gptoss_combine_weights_backward"))
  src = (pathlib.Path(__file__).parent/"weights_backward.cpp").read_text()
  lib = compile_hip(src, [f"-DTOKENS={tokens}", f"-DROWS={rows}", "-fno-fast-math"])
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

@functools.cache
def _inverse_rows(out:UOp, dest_row:UOp) -> UOp:
  groups, entries = dest_row.shape
  g, i = UOp.range(groups, 0), UOp.range(entries, 1)
  row = dest_row.index(g, i).load().cast(dtypes.weakint)
  return out.index(g, row).store(i.cast(dtypes.int32)).end(g, i).sink(arg=KernelInfo("gptoss_combine_inverse", opts_to_apply=()))

@functools.cache
def _combine_input_backward(out:UOp, gradient:UOp, src_row:UOp, weights:UOp) -> UOp:
  groups, rows, hidden = out.shape
  tokens = gradient.shape[1]
  assert hidden == 2880 and gradient.shape == (groups, tokens, hidden)
  assert src_row.shape == (groups, rows) and weights.shape == (groups, tokens, 4)
  assert out.dtype == gradient.dtype == dtypes.bfloat16 and src_row.dtype == dtypes.int32 and weights.dtype == dtypes.float32
  sink = UOp.sink(out.base, gradient.base, src_row.base, weights.base, UOp.special(256, "lidx0"),
                  UOp.special(groups*rows, "gidx0"), arg=KernelInfo("gptoss_combine_input_backward"))
  src = (pathlib.Path(__file__).parent/"input_backward.cpp").read_text()
  lib = compile_hip(src, [f"-DTOKENS={tokens}", f"-DROWS={rows}", "-fno-fast-math"])
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def _combine_backward(gradient:UOp, kernel:UOp) -> tuple:
  z, dest_row, weights = (Tensor(u) for u in kernel.src[2:5])
  src_row = z[:, :, 0].full_like(-1, dtype=dtypes.int32)
  src_row = Tensor.custom_kernel(src_row, dest_row, fxn=_inverse_rows)[0]
  dz = alloc_like(z.shape, z.dtype, z.device, z.uop.axis)
  dz = Tensor.custom_kernel(dz, Tensor(gradient), src_row, weights, fxn=_combine_input_backward)[0]
  dw = alloc_like(weights.shape, weights.dtype, weights.device, weights.uop.axis)
  dw = Tensor.custom_kernel(dw, z, Tensor(gradient), dest_row, fxn=_combine_weights_backward)[0]
  return None, dz.uop, None, dw.uop

def fused_combine(z:Tensor, dest_row:Tensor, weights:Tensor) -> Tensor:
  out = alloc_like((*weights.shape[:2], z.shape[-1]), z.dtype, z.device, z.uop.axis)
  return Tensor.custom_kernel(out, z, dest_row, weights, fxn=_combine_forward, grad_fxn=_combine_backward)[0]
