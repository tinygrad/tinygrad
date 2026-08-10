import functools
from typing import cast
from tinygrad import Tensor, UOp, Device, Context, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo

def amd_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool:
  """The hand-written wave32 kernel is intentionally limited to RDNA3/gfx11."""
  if device is None: return False
  device = device[0] if isinstance(device, tuple) else device
  with Context(ALLOW_DEVICE_USAGE=1):
    return (target:=getattr(Device[device], "target", None)) is not None and target[0] == 11

@functools.cache
def _gated_delta_prefill_kernel(core:UOp, next_state:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, kq:UOp) -> UOp:
  batch, heads, tokens, value_dim = cast(tuple[int, int, int, int], core.shape)
  key_dim, alpha_dim = cast(int, q.shape[-1]), cast(int, alpha.shape[-1]) if len(alpha.shape) == 4 else 1
  core, v = (x.reshape(batch*heads, tokens, value_dim) for x in (core, v))
  q, k = (x.reshape(batch*heads, tokens, key_dim) for x in (q, k))
  beta, kq = (x.reshape(batch*heads, tokens) for x in (beta, kq))
  alpha = alpha.reshape(batch*heads, tokens, alpha_dim)
  state, next_state = (x.reshape(batch*heads, value_dim, key_dim) for x in (state, next_state))
  bh, row, cols = UOp.range(batch*heads, 0, AxisType.GLOBAL), UOp.range(value_dim, 2), tuple(range(key_dim))
  current = UOp.placeholder((key_dim,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  current = current.after(UOp.group(*(current[col].store(state[bh, row, col].float()) for col in cols)))
  token = UOp.range(tokens, 1, AxisType.REDUCE)
  previous = tuple(current.after(token)[col].load() for col in cols)
  keys, queries = (tuple(x[bh, token, col].load() for col in cols) for x in (k, q))
  av = tuple(alpha[bh, token, col if alpha_dim > 1 else 0].load() for col in cols)
  bv = beta[bh, token].load()
  state_k = sum((x*a*y for x,a,y in zip(previous, av, keys)), UOp.const(0, dtypes.float32))
  state_q = sum((x*a*y for x,a,y in zip(previous, av, queries)), UOp.const(0, dtypes.float32))
  delta = (v[bh, token, row].load() - state_k) * bv
  step = UOp.group(core[bh, token, row].store(state_q + delta*kq[bh, token]),
                   *(current[col].store(x*a + delta*y) for col,x,a,y in zip(cols, previous, av, keys))).end(token)
  stores = (next_state[bh, row, col].store(current.after(step)[col].load().cast(next_state.dtype)) for col in cols)
  return UOp.group(*stores).end(row, bh).sink(arg=KernelInfo(name="gated_delta_prefill", opts_to_apply=()))

def gated_delta_prefill(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor) -> tuple[Tensor, Tensor]:
  batch, heads, tokens, key_dim = q.shape
  value_dim = v.shape[-1]
  assert q.shape == k.shape and v.shape[:3] == q.shape[:3] and beta.shape == (batch, heads, tokens)
  assert alpha.shape in ((batch, heads, tokens), (batch, heads, tokens, key_dim))
  assert state.shape == (batch, heads, value_dim, key_dim)
  kernel = _gated_delta_prefill_kernel
  if amd_custom_kernels_supported(q.device) and key_dim % 32 == 0 and value_dim % 4 == 0:
    from tinygrad.llm.kernels.amd import _gated_delta_prefill_kernel as kernel
  core, next_state, kq = Tensor.empty_like(v), Tensor.empty_like(state), (q*k).sum(-1).contiguous()
  result = Tensor.custom_kernel(core, next_state, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(), state, kq,
                                fxn=kernel)
  return result[0], result[1]
