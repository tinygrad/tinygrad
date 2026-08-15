import functools
from typing import cast
from tinygrad import Tensor, UOp, nn, dtypes, Device, Context
from tinygrad.device import Buffer
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import prod
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

def amd_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool:
  # the custom kernels are tuned for RDNA3 (gfx11): the WMMA register layouts don't match gfx12 (RDNA4)
  # or CDNA (MFMA-only, wave64), and the dp4a builtins and 32-lane wave ops aren't portable either.
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] != "AMD": return False
  # Device[...] trips ALLOW_DEVICE_USAGE=0 in function contexts, the device is always open here anyway
  with Context(ALLOW_DEVICE_USAGE=1):
    return (t:=getattr(Device[device], "target", None)) is not None and t[0] == 11

class Linear(nn.Linear):
  ggml_type:int|None = None
  def __init__(self, in_features:int, out_features:int, bias=True):
    super().__init__(in_features, out_features, bias)
    self.in_features, self.out_features = in_features, out_features
    self.use_custom_quant = True
  def set_quantized(self, decoded:Tensor):
    packed_sizes = {decoded.numel() // 256 * type_size:typ for typ,type_size in ((13, 176), (14, 210), (23, 136))}
    raw = next((u for u in decoded.uop.toposort() if u.op is Ops.SHRINK and u.dtype == dtypes.uint8 and prod(u.shape) in packed_sizes), None)
    if raw is None: return
    raw_offset = raw.contiguous_view_offset()
    assert raw_offset is not None and raw_offset % 4 == 0 and raw.buf_uop.dtype == dtypes.uint8
    self.ggml_type = packed_sizes[prod(raw.shape)]
    # Q5_K and IQ4_XS kernels consume words. Store a typed buffer view directly: a lazy BITCAST is decomposed into
    # byte-combining ALU before custom-kernel scheduling and would copy the entire packed weight on every JIT graph.
    packed_dtype = dtypes.uint8 if self.ggml_type == 14 else dtypes.uint32
    self.weight = Tensor(UOp.from_buffer(cast(Buffer, raw.buf_uop.buffer).view(raw.max_numel() * raw.dtype.itemsize // packed_dtype.itemsize,
                                                                              packed_dtype, raw_offset)))
  def __call__(self, x:Tensor) -> Tensor:
    static = isinstance(x.numel(), int)
    supported = self.use_custom_quant and amd_custom_kernels_supported(self.weight.device)
    if self.ggml_type is None and not static: supported = self.use_custom_quant = False
    if self.ggml_type is None and supported: self.set_quantized(self.weight)
    if self.ggml_type in (13, 14, 23) and supported:
      from tinygrad.llm.kernels.amd import q8_linear
      return q8_linear(self, x)
    return super().__call__(x)

@functools.cache
def _gated_delta_prefill_kernel(core:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, kq:UOp, start_pos:UOp|None=None) -> UOp:
  batch, heads, tokens, value_dim = cast(tuple[int, int, int, int], core.shape)
  key_dim, alpha_dim = cast(int, q.shape[-1]), cast(int, alpha.shape[-1]) if len(alpha.shape) == 4 else 1
  core, v = (x.reshape(batch*heads, tokens, value_dim) for x in (core, v))
  q, k = (x.reshape(batch*heads, tokens, key_dim) for x in (q, k))
  beta, kq = (x.reshape(batch*heads, tokens) for x in (beta, kq))
  alpha, state = alpha.reshape(batch*heads, tokens, alpha_dim), state.reshape(batch*heads, value_dim, key_dim)
  bh, row, cols = UOp.range(batch*heads, 0, AxisType.GLOBAL), UOp.range(value_dim, 2), tuple(range(key_dim))
  current = UOp.placeholder((key_dim,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  initial = None if start_pos is None else start_pos.eq(0)
  current = current.after(UOp.group(*(current[col].store(state[bh, row, col].float() if initial is None else
    initial.where(0, state[bh, row, col].float())) for col in cols)))
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

def gated_delta_prefill(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor, start_pos:Tensor|None=None) -> Tensor:
  batch, heads, tokens, key_dim = q.shape
  value_dim = v.shape[-1]
  assert q.shape == k.shape and v.shape[:3] == q.shape[:3] and beta.shape == (batch, heads, tokens)
  assert alpha.shape in ((batch, heads, tokens), (batch, heads, tokens, value_dim))
  assert state.shape == (batch, heads, value_dim, key_dim)
  kernel = _gated_delta_prefill_kernel
  if amd_custom_kernels_supported(q.device) and key_dim % 32 == 0 and value_dim % 4 == 0:
    from tinygrad.llm.kernels.amd import _gated_delta_prefill_kernel as kernel
  core, kq = Tensor.empty_like(v), (q*k).sum(-1).contiguous()
  srcs = (core, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(), state, kq)
  if start_pos is None: return Tensor.custom_kernel(*srcs, fxn=kernel)[0]
  contig = tuple(x.uop if x.uop.op is Ops.AFTER else x.uop.contiguous() for x in srcs)
  params = tuple(UOp.placeholder_like(x, slot=i) for i,x in enumerate(contig))
  assert start_pos.uop.is_bound_var
  call = kernel(*params, start_pos.uop.src[0]).call(*contig, start_pos.uop)
  return Tensor(contig[0].after(call))
