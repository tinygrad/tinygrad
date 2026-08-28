from __future__ import annotations
import functools
from tinygrad import Tensor
from tinygrad.uop.ops import UOp

def rmsnorm_fwd(x_in:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  x = x_in.float()
  rrms = (x.square().mean(-1, keepdim=True) + eps).rsqrt()
  return (x * rrms).cast(x_in.dtype), rrms

@functools.cache
def _rmsnorm_fwd_fxn(x_in_p, eps, device):
  return rmsnorm_fwd(Tensor(x_in_p, device=device), eps)

def _rmsnorm_bwd(grad:UOp, call:UOp) -> tuple:
  x_normed, rrms = map(Tensor, call.returned_outputs)
  do_float = Tensor(grad).float()
  d_x = rrms * (do_float - x_normed.float() * (do_float * x_normed.float()).mean(-1, keepdim=True))
  return (d_x.cast(call.src[1].dtype).uop,)

def rmsnorm(x_in:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  fxn = _rmsnorm_fwd_fxn(x_in.as_param(0).uop, eps, x_in.device)
  call = UOp.call_outputs([fxn[0].uop, fxn[1].uop], x_in.uop, grad_fxn=_rmsnorm_bwd)
  return map(Tensor, call.returned_outputs)
