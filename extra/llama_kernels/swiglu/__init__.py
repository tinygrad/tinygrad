import functools, math, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.helpers import ceildiv
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.llama_kernels import alloc_like, compile_hip

@functools.cache
def _custom_swiglu(out:UOp, x_w13:UOp) -> UOp:
  hidden, n_elems = x_w13.shape[-1]//2, math.prod(x_w13.shape[:-1]) * x_w13.shape[-1]//2
  num_wg = min(ceildiv(ceildiv(n_elems, 16), 512), 65535)
  threads, workgroups = UOp.special(512, "lidx0"), UOp.special(num_wg, "gidx0")
  sink = UOp.sink(out.base, x_w13.base, threads, workgroups,
                  arg=KernelInfo(f"swiglu_fwd_{n_elems}", estimates=Estimates(ops=5*n_elems, mem=6*n_elems)))
  src = (pathlib.Path(__file__).parent/"swiglu.hip").read_text()
  lib = compile_hip(src, [f"-DN_ELEMS={n_elems}", f"-DHIDDEN={hidden}", f"-DNUM_WG={num_wg}", "-DTHREADS_PER_WG=512"])
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

@functools.cache
def _custom_swiglu_bwd(grad_out:UOp, x_w13:UOp, grad_act:UOp) -> UOp:
  hidden, n_elems = x_w13.shape[-1]//2, math.prod(x_w13.shape[:-1]) * x_w13.shape[-1]//2
  num_wg = min(ceildiv(ceildiv(n_elems, 16), 512), 65535)
  threads, workgroups = UOp.special(512, "lidx0"), UOp.special(num_wg, "gidx0")
  sink = UOp.sink(grad_out.base, x_w13.base, grad_act.base, threads, workgroups,
                  arg=KernelInfo(f"swiglu_bwd_{n_elems}", estimates=Estimates(ops=10*n_elems, mem=10*n_elems)))
  src = (pathlib.Path(__file__).parent/"swiglu.hip").read_text()
  lib = compile_hip(src, [f"-DN_ELEMS={n_elems}", f"-DHIDDEN={hidden}", f"-DNUM_WG={num_wg}", "-DTHREADS_PER_WG=512", "-DSWIGLU_BACKWARD"])
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def _swiglu_bwd(gradient:UOp, kernel:UOp):
  _, x_w13 = kernel.src[1:]
  axis = x_w13.axis if isinstance(x_w13.device, tuple) else None
  grad_out = alloc_like(x_w13.shape, dtypes.bfloat16, x_w13.device, axis)
  grad_out, *_ = Tensor.custom_kernel(grad_out, Tensor(x_w13, device=x_w13.device), Tensor(gradient, device=x_w13.device),
                                      fxn=_custom_swiglu_bwd)
  return (None, grad_out.uop)

def swiglu(x_w13:Tensor) -> Tensor:
  assert x_w13.dtype == dtypes.bfloat16 and x_w13.ndim >= 2 and x_w13.shape[-1] % 32 == 0
  *prefix, two_k = x_w13.shape
  axis = x_w13.uop.axis if isinstance(x_w13.device, tuple) else None
  out = alloc_like((*prefix, two_k//2), dtypes.bfloat16, x_w13.device, axis)
  return Tensor.custom_kernel(out, x_w13, fxn=_custom_swiglu, grad_fxn=_swiglu_bwd)[0]
