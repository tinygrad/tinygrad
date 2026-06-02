from __future__ import annotations
import functools
from tinygrad import Tensor, dtypes
from tinygrad.helpers import prod
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from extra.llama_kernels import alloc_like

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
def _rmsnorm_weighted_fwd(out:UOp, x:UOp, weight:UOp, eps_val:float) -> UOp:
  MBS, SEQ, HIDDEN = x.shape
  ROWS, n_elems = MBS * SEQ, prod(x.shape)
  assert n_elems % HIDDEN == 0

  mem = n_elems * 2 + n_elems * 2 + HIDDEN * 2
  out, x = out.reshape(n_elems), x.reshape(n_elems)

  row = UOp.range(ROWS, 0)
  h_reduce = UOp.range(HIDDEN, 1, AxisType.REDUCE)
  x_reduce = x[row * HIDDEN + h_reduce].cast(dtypes.float)
  rrms = ((x_reduce * x_reduce).reduce(h_reduce, arg=Ops.ADD) * (1.0 / HIDDEN) + eps_val).sqrt().reciprocal()
  h = UOp.range(HIDDEN, 2)
  idx = row * HIDDEN + h
  y = x[idx].cast(dtypes.float) * rrms * weight[h].cast(dtypes.float)
  store = out[idx].store(y.cast(out.dtype.base))

  return store.end(h, row).sink(arg=KernelInfo(f"rmsnorm_weighted_fwd_{n_elems}_h{HIDDEN}_eps{eps_val:.0e}"))

def _rmsnorm_weighted_bwd(gradient:UOp, kernel:UOp, eps_val:float) -> tuple:
  # fwd inputs: (out, x, weight). Recompute rrms/x_normed to avoid a full forward save.
  _, x_u, weight_u = kernel.src[1:]
  grad = Tensor(gradient).float()
  x = Tensor(x_u).float()
  rrms = (x.square().mean(-1, keepdim=True) + eps_val).rsqrt()
  x_normed = x * rrms
  weight = Tensor(weight_u).float()
  grad_normed = grad * weight
  grad_x = rrms * (grad_normed - x_normed * (grad_normed * x_normed).mean(-1, keepdim=True))
  grad_w = (grad * x_normed).sum(axis=tuple(range(grad.ndim-1))).cast(weight_u.dtype)
  return (None, grad_x.cast(x_u.dtype).uop, grad_w.uop)

def rmsnorm_weighted(x:Tensor, weight:Tensor, eps:float) -> Tensor:
  assert x.dtype == dtypes.bfloat16 and weight.dtype == dtypes.bfloat16
  assert x.shape[-1] == weight.shape[-1], f"HIDDEN mismatch: x={x.shape}, weight={weight.shape}"
  MBS, SEQ, HIDDEN = x.shape
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  if isinstance(x.device, tuple): assert axis in (None, 0, 1), f"unsupported sharding axis={axis}"
  out = alloc_like((MBS, SEQ, HIDDEN), dtypes.bfloat16, x.device, axis)
  fxn = functools.partial(_rmsnorm_weighted_fwd, eps_val=eps)
  out, *_ = Tensor.custom_kernel(out, x, weight, fxn=fxn, grad_fxn=functools.partial(_rmsnorm_weighted_bwd, eps_val=eps))
  return out
