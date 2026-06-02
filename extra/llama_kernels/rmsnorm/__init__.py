from __future__ import annotations
import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.llama_kernels import NUM_WG, THREADS_PER_WG, alloc_like, dname_of, compile_hip

def _src_weighted() -> str: return (pathlib.Path(__file__).parent/"rmsnorm_weighted.cpp").read_text()

def rmsnorm_fwd(x_in:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  x = x_in.float()
  rrms = (x.square().mean(-1, keepdim=True) + eps).rsqrt()
  return (x * rrms).cast(x_in.dtype), rrms

@functools.cache
def _rmsnorm_fwd_fxn(x_in_p, eps, device):
  return rmsnorm_fwd(Tensor(x_in_p, device=device), eps)

def _rmsnorm_bwd(grad:UOp, call:UOp) -> tuple:
  x_normed = Tensor(call.gettuple(0)).float()
  do_float = Tensor(grad).float()
  d_x = Tensor(call.gettuple(1)) * (do_float - x_normed * (do_float * x_normed).mean(-1, keepdim=True))
  return (d_x.cast(call.src[1].dtype).uop,)

def rmsnorm(x_in:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  fxn = _rmsnorm_fwd_fxn(x_in.as_param(0).uop, eps, x_in.device)
  call = UOp.maketuple(fxn[0].uop, fxn[1].uop).call(x_in.uop, grad_fxn=_rmsnorm_bwd)
  return Tensor(call.gettuple(0)), Tensor(call.gettuple(1))

@functools.cache
def _rmsnorm_weighted_fwd(out:UOp, x_normed_out:UOp, rrms_out:UOp, x:UOp, weight:UOp, dname:str, eps_val:float) -> UOp:
  MBS, SEQ, HIDDEN = x.shape
  n_elems = MBS * SEQ * HIDDEN
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(NUM_WG, "gidx0")
  mem = n_elems * 2 + n_elems * 2 + n_elems * 2 + HIDDEN * 2 + MBS * SEQ * 4
  sink = UOp.sink(out.base, x_normed_out.base, rrms_out.base, x.base, weight.base, threads, workgroups,
                  arg=KernelInfo(f"rmsnorm_weighted_fwd_{n_elems}_h{HIDDEN}_eps{eps_val:.0e}",
                                 estimates=Estimates(ops=6*n_elems, mem=mem)))
  defines = [f"-DN_ELEMS={n_elems}", f"-DHIDDEN={HIDDEN}", f"-DNUM_WG={NUM_WG}",
             f"-DTHREADS_PER_WG={THREADS_PER_WG}", f"-DEPS_LITERAL={eps_val}f"]
  src = _src_weighted()
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

def _rmsnorm_weighted_bwd(gradient:UOp, kernel:UOp) -> tuple:
  # fwd inputs: (out, x_normed_out, rrms_out, x, weight)
  _, x_normed_u, rrms_u, x_u, weight_u = kernel.src[1:]
  grad = Tensor(gradient).float()
  x_normed = Tensor(x_normed_u.after(kernel)).float()
  rrms = Tensor(rrms_u.after(kernel)).reshape(*x_normed.shape[:-1], 1)
  weight = Tensor(weight_u).float()
  grad_normed = grad * weight
  grad_x = rrms * (grad_normed - x_normed * (grad_normed * x_normed).mean(-1, keepdim=True))
  grad_w = (grad * x_normed).sum(axis=tuple(range(grad.ndim-1))).cast(weight_u.dtype)
  return (None, None, None, grad_x.cast(x_u.dtype).uop, grad_w.uop)

def rmsnorm_weighted(x:Tensor, weight:Tensor, eps:float) -> Tensor:
  assert x.dtype == dtypes.bfloat16 and weight.dtype == dtypes.bfloat16
  assert x.shape[-1] == weight.shape[-1], f"HIDDEN mismatch: x={x.shape}, weight={weight.shape}"
  MBS, SEQ, HIDDEN = x.shape
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  if isinstance(x.device, tuple): assert axis in (None, 0, 1), f"unsupported sharding axis={axis}"
  out = alloc_like((MBS, SEQ, HIDDEN), dtypes.bfloat16, x.device, axis)
  x_normed_out = alloc_like((MBS, SEQ, HIDDEN), dtypes.bfloat16, x.device, axis)
  rrms_out = alloc_like((MBS, SEQ), dtypes.float32, x.device, axis)
  fxn = functools.partial(_rmsnorm_weighted_fwd, dname=dname_of(x.device), eps_val=eps)
  out, *_ = Tensor.custom_kernel(out, x_normed_out, rrms_out, x, weight, fxn=fxn, grad_fxn=_rmsnorm_weighted_bwd)
  return out
