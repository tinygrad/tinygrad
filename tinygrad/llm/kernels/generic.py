import functools
from typing import cast
from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo

@functools.cache
def _gated_delta_prefill_kernel(core:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, kq:UOp) -> UOp:
  batch, heads, tokens, value_dim = cast(tuple[int, int, int, int], core.shape)
  key_dim, alpha_dim = cast(int, q.shape[-1]), cast(int, alpha.shape[-1]) if len(alpha.shape) == 4 else 1
  core, v = (x.reshape(batch*heads, tokens, value_dim) for x in (core, v))
  q, k = (x.reshape(batch*heads, tokens, key_dim) for x in (q, k))
  beta, kq = (x.reshape(batch*heads, tokens) for x in (beta, kq))
  alpha, state = alpha.reshape(batch*heads, tokens, alpha_dim), state.reshape(batch*heads, value_dim, key_dim)
  bh, row, cols = UOp.range(batch*heads, 0, AxisType.GLOBAL), UOp.range(value_dim, 2), tuple(range(key_dim))
  current = UOp.placeholder((key_dim,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  current = current.after(UOp.group(*(current[col].store(state[bh, row, col].float()) for col in cols)))
  token = UOp.range(tokens, 1, AxisType.REDUCE)
  previous = tuple(current.after(token)[col].load() for col in cols)
  keys, queries = (tuple(x[bh, token, col].load() for col in cols) for x in (k, q))
  av, bv = alpha[bh, token, row if alpha_dim > 1 else 0].load(), beta[bh, token].load()
  state_k = sum((x*y for x,y in zip(previous, keys)), UOp.const(0, dtypes.float32))
  state_q = sum((x*y for x,y in zip(previous, queries)), UOp.const(0, dtypes.float32))
  delta = (v[bh, token, row].load() - state_k*av) * bv
  step = UOp.group(core[bh, token, row].store(state_q*av + delta*kq[bh, token]),
                   *(current[col].store(x*av + delta*y) for col,x,y in zip(cols, previous, keys))).end(token)
  stores = (state[bh, row, col].store(current.after(step)[col].load().cast(state.dtype)) for col in cols)
  return UOp.group(*stores).end(row, bh).sink(arg=KernelInfo(name="gated_delta_prefill", opts_to_apply=()))

def gated_delta_prefill(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor) -> Tensor:
  batch, heads, tokens, key_dim = q.shape
  value_dim = v.shape[-1]
  assert q.shape == k.shape and v.shape[:3] == q.shape[:3] and beta.shape == (batch, heads, tokens)
  assert alpha.shape in ((batch, heads, tokens), (batch, heads, tokens, value_dim))
  assert state.shape == (batch, heads, value_dim, key_dim)
  core, kq = Tensor.empty_like(v), (q*k).sum(-1).contiguous()
  return Tensor.custom_kernel(core, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(), state, kq,
                              fxn=_gated_delta_prefill_kernel)[0]
