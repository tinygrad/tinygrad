import functools
from typing import cast
from tinygrad import Tensor, UOp, nn, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import prod
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, resolve

class Linear(nn.Linear):
  ggml_type:int|None = None
  def __init__(self, in_features:int, out_features:int, bias=True):
    super().__init__(in_features, out_features, bias)
    self.in_features, self.out_features = in_features, out_features
    self._raw_offset_uop:UOp|None = None
  def set_quantized(self, decoded:Tensor) -> Tensor|None:
    packed_sizes = {decoded.numel() // 256 * type_size:typ for typ,type_size in ((13, 176), (14, 210), (23, 136))}
    raw = next((u for u in decoded.uop.toposort() if u.op is Ops.SHRINK and u.dtype == dtypes.uint8 and prod(u.shape) in packed_sizes), None)
    if raw is None: return None
    self.weight, self.ggml_type = Tensor(raw).flatten(), packed_sizes[prod(raw.shape)]
    raw_offset = self.weight.uop.contiguous_view_offset()
    assert raw_offset is not None and raw_offset % 4 == 0 and self.weight.uop.buf_uop.dtype == dtypes.uint8
    if self.ggml_type == 23 and str(self.weight.device).startswith("AMD"):
      from tinygrad.llm.kernels.amd import iq4_half_lut
      iq4_half_lut(str(self.weight.device))
    return Tensor([raw_offset // 4], dtype=dtypes.uint64, device=self.weight.device)
  def __call__(self, x:Tensor) -> Tensor:
    if self.ggml_type in (13, 14, 23) and str(self.weight.device).startswith("AMD"):
      from tinygrad.llm.kernels.amd import q8_linear
      return q8_linear(self, x)
    return super().__call__(x)

def cached_attention(q:Tensor, stacked_kv:Tensor, cache_kv:Tensor, cache_scale:Tensor|None, start_pos:int|UOp) -> Tensor:
  if cache_kv.dtype == dtypes.int8:
    from tinygrad.llm.kernels.amd import quantized_attention
    assert cache_scale is not None
    return quantized_attention(q, stacked_kv, cache_kv, cache_scale, start_pos)
  T = q.shape[2]
  assigned_kv = Tensor(cache_kv.uop.after(cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(stacked_kv.uop)))
  k, v = assigned_kv[0, :, :, 0:start_pos+T, :], assigned_kv[1, :, :, 0:start_pos+T, :]
  mask = Tensor.full((1, 1, T, k.shape[-2]), float("-inf"), dtype=q.dtype, device=q.device, buffer=False).triu(start_pos+1) \
    if resolve(T != 1) else None
  return q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)

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
  kernel = _gated_delta_prefill_kernel
  if str(q.device).startswith("AMD") and key_dim % 32 == 0 and value_dim % 4 == 0:
    from tinygrad.llm.kernels.amd import _gated_delta_prefill_kernel as kernel
  core, kq = Tensor.empty_like(v), (q*k).sum(-1).contiguous()
  return Tensor.custom_kernel(core, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(), state, kq,
                              fxn=kernel)[0]
