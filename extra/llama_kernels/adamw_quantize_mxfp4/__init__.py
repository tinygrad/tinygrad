import functools, math, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
from tinygrad.schedule.allreduce import _allreduce_view
from extra.llama_kernels.quantize_mxfp4 import alloc_mxfp4_outputs

def _physical_view(x:Tensor) -> Tensor:
  if (view:=x.uop.contiguous_view()) is None or x.uop.numel() == x.uop.base.numel(): return x
  view_buf, offset = view
  view_size = x.uop.numel() * x.dtype.itemsize // view_buf.dtype.itemsize
  physical = _allreduce_view(view_buf, offset, offset+view_size)
  if physical.dtype != x.dtype: physical = physical.bitcast(x.dtype)
  if x.uop.base.op is Ops.AFTER: physical = physical.after(x.uop.base)
  return Tensor(physical.reshape(x.shape))

@functools.cache
def _custom_adamw_quantize_mxfp4(m:UOp, v:UOp, master:UOp, param:UOp, row_fp4:UOp, row_scale:UOp,
                                  col_fp4:UOp, col_scale:UOp, grad:UOp, lr:UOp, b1_t:UOp, b2_t:UOp, clip_coeff:UOp,
                                  *, b1:float, b2:float, eps:float, wd:float, grad_acc:int) -> UOp:
  M, N = math.prod(param.shape[:-1]), param.shape[-1]
  assert M % 256 == 0 and N % 256 == 0
  assert m.shape == v.shape == master.shape == param.shape == grad.shape
  name = f"adamw_quantize_mxfp4_dual_{M}_{N}"
  bufs = (m, v, master, param, row_fp4, row_scale, col_fp4, col_scale, grad, lr, b1_t, b2_t, clip_coeff)
  outputs = bufs[:8]
  mem = M*N*(2 + 4*6 + 2 + 1) + M*N//16
  sink = UOp.sink(*(x.base for x in bufs),
                  *(UOp(Ops.CUSTOM, src=(x.base.index(0),), arg=("", dtypes.void)) for x in outputs),
                  UOp.special(256, "lidx0"), UOp.special(M//32, "gidx0"), UOp.special(N//256, "gidx1"),
                  arg=KernelInfo(name, estimates=Estimates(ops=40*M*N, mem=mem)))
  src = (pathlib.Path(__file__).parent/"adamw_quantize_mxfp4.cpp").read_text()
  defines = [f"-I{pathlib.Path(__file__).parent.parent/'quantize_mxfp4'}",
             f"-DKERNEL_NAME={name}", f"-DM_DIM={M}", f"-DN_DIM={N}",
             f"-DB1={b1}f", f"-DB2={b2}f", f"-DONE_MINUS_B1={1.0-b1}f", f"-DONE_MINUS_B2={1.0-b2}f",
             f"-DEPS={eps}f", f"-DWEIGHT_DECAY={wd}f", f"-DGRAD_ACC={grad_acc}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=HIPCCCompiler("gfx950", ["-std=c++20", *defines]).compile_cached(src))))

def adamw_quantize_mxfp4(param:Tensor, grad:Tensor, m:Tensor, v:Tensor, master:Tensor, lr:Tensor, b1_t:Tensor, b2_t:Tensor,
                         clip_coeff:Tensor, *, b1:float, b2:float, eps:float, wd:float, grad_acc:int,
                         out:tuple[Tensor, Tensor, Tensor, Tensor]|None=None):
  assert param.dtype == grad.dtype == dtypes.bfloat16
  assert m.dtype == v.dtype == master.dtype == dtypes.float32
  assert param.shape == grad.shape == m.shape == v.shape == master.shape
  param, grad, m, v, master = (_physical_view(x) for x in (param, grad, m, v, master))
  outputs = alloc_mxfp4_outputs(param) if out is None else out
  fxn = functools.partial(_custom_adamw_quantize_mxfp4, b1=b1, b2=b2, eps=eps, wd=wd, grad_acc=grad_acc)
  ret = Tensor.custom_kernel(m, v, master, param, *outputs, grad, lr, b1_t, b2_t, clip_coeff, fxn=fxn)
  return ret[:8]
