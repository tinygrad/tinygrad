import functools
from typing import cast
from tinygrad import Tensor, UOp, Device, Context, dtypes
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo

def amd_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool:
  """The hand-written wave32 kernel is intentionally limited to RDNA3/gfx11."""
  if device is None: return False
  device = device[0] if isinstance(device, tuple) else device
  with Context(ALLOW_DEVICE_USAGE=1):
    return (target:=getattr(Device[device], "target", None)) is not None and target[0] == 11

def amd_wave64_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool:
  """CDNA4 wave64 kernels used by the MI350X K3 path."""
  if device is None: return False
  device = device[0] if isinstance(device, tuple) else device
  if device.startswith("NULL:HIP:gfx950"): return True  # compile-only CDNA4 coverage without MI350X hardware
  with Context(ALLOW_DEVICE_USAGE=1):
    return (target:=getattr(Device[device], "target", None)) is not None and target[:2] == (9, 5)

def amd_packed_mxfp4_supported(device:str|tuple[str, ...]|None) -> bool:
  return amd_custom_kernels_supported(device) or amd_wave64_custom_kernels_supported(device)

def amd_exact_bf16_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool:
  """The LDS-reduced exact BF16 pair kernels are portable across RDNA3 and CDNA4."""
  return amd_custom_kernels_supported(device) or amd_wave64_custom_kernels_supported(device)

def amd_int32_item(x:Tensor, host:memoryview) -> int:
  """Copy a realized replicated AMD scalar without constructing a new scheduler graph."""
  if x.numel() != 1 or x.dtype != dtypes.int32 or host.nbytes != 4: raise ValueError("expected one int32 and a four-byte host view")
  buf = x.uop.buffer
  if isinstance(buf, MultiBuffer): buf = buf.bufs[0]
  if not isinstance(buf, Buffer) or not buf.device.startswith("AMD"): raise ValueError("expected a realized AMD buffer")
  buf.allocator._copyout(host, buf._buf)
  return int.from_bytes(host, byteorder="little", signed=True)

def mxfp4_expert_linear(sel:Tensor, x:Tensor, weight:Tensor, scale:Tensor, partial:bool=False) -> Tensor:
  """Run a TP routed projection without materializing selected BF16 weights."""
  from tinygrad.llm.kernels.amd import _mxfp4_expert_linear_kernel, _mxfp4_expert_linear_wave64_kernel
  batch, tokens, topk = sel.shape
  out_features = weight.shape[1]
  weight_axis = weight.uop.axis
  if isinstance(weight.device, tuple):
    devices = weight.device
    # Gate/up shard their output dimension. Down shards its reduction dimension;
    # represent each GPU's partial as a size-one device axis, then all-reduce it.
    axis = 3 if weight_axis == 1 else 4
    shard_shape: tuple[int|UOp, ...]
    if weight_axis == 1:
      if out_features % len(devices): raise ValueError(f"expert output {out_features} is not divisible by {len(devices)} devices")
      shard_shape = (batch, tokens, topk, out_features//len(devices))
    elif weight_axis == 2:
      shard_shape = (batch, tokens, topk, out_features, 1)
    else: raise ValueError(f"unsupported expert TP axis {weight_axis}")
    partial_dtype = dtypes.float32 if weight_axis == 2 else dtypes.bfloat16
    parts = [Tensor.empty(*shard_shape, dtype=partial_dtype, device=device).uop for device in devices]
    out = Tensor(parts[0].mstack(*parts[1:]).unshard(axis))
  else:
    out = Tensor.empty(batch, tokens, topk, out_features, dtype=dtypes.bfloat16, device=weight.device)
  kernel = _mxfp4_expert_linear_wave64_kernel if amd_wave64_custom_kernels_supported(weight.device) else _mxfp4_expert_linear_kernel
  out = Tensor.custom_kernel(out, sel.contiguous(), x.contiguous(), weight, scale, fxn=kernel)[0]
  return out if weight_axis == 2 and partial else out.sum(4).cast(dtypes.bfloat16) if weight_axis == 2 else out

def bf16_partial_linear(x:Tensor, weight:Tensor) -> Tensor:
  """Return output-shaped FP32 TP partials with a final device axis, without all-reduce."""
  from tinygrad.llm.kernels.amd import _bf16_partial_linear_kernel
  if not isinstance(weight.device, tuple) or weight.uop.axis != 1: raise ValueError("partial linear expects input-sharded TP weight")
  batch, tokens, _ = x.shape
  devices, out_features = weight.device, weight.shape[0]
  shard_shape = (batch, tokens, out_features, 1)
  parts = [Tensor.empty(*shard_shape, dtype=dtypes.float32, device=device).uop for device in devices]
  out = Tensor(parts[0].mstack(*parts[1:]).unshard(3))
  return Tensor.custom_kernel(out, x.contiguous(), weight, fxn=_bf16_partial_linear_kernel)[0]

def bf16_matvec(x:Tensor, weight:Tensor) -> Tensor:
  from tinygrad.llm.kernels.amd import _bf16_matvec_kernel
  batch, tokens, _ = x.shape
  out_features = weight.shape[0]
  if isinstance(weight.device, tuple):
    devices = weight.device
    if weight.uop.axis == 0:
      shard_shape = (batch, tokens, out_features//len(devices))
      parts = [Tensor.empty(*shard_shape, dtype=dtypes.bfloat16, device=device).uop for device in devices]
      out = Tensor(parts[0].mstack(*parts[1:]).unshard(2))
    elif weight.uop.axis is None: out = Tensor.empty(batch, tokens, out_features, dtype=dtypes.bfloat16, device=weight.device)
    else: raise ValueError("bf16_matvec expects output-sharded or replicated TP weight")
  else: out = Tensor.empty(batch, tokens, out_features, dtype=dtypes.bfloat16, device=weight.device)
  return Tensor.custom_kernel(out, x.contiguous(), weight, fxn=_bf16_matvec_kernel)[0]

def mxfp8_quantize_dequantize(x:Tensor) -> Tensor:
  """gfx11 software MXFP8 round trip without a multi-kernel reduction graph."""
  from tinygrad.llm.kernels.amd import _mxfp8_qdq_kernel
  out = Tensor.empty_like(x, dtype=dtypes.bfloat16)
  return Tensor.custom_kernel(out, x.contiguous(), fxn=_mxfp8_qdq_kernel)[0]

def kda_qkv_linear(x:Tensor, qw:Tensor, kw:Tensor, vw:Tensor) -> tuple[Tensor, Tensor, Tensor]:
  """Fuse equal-sized output-sharded KDA Q/K/V decode projections."""
  from tinygrad.llm.kernels.amd import _kda_qkv_kernel
  batch, tokens, _ = x.shape
  out_features = qw.shape[0]
  if not (qw.shape == kw.shape == vw.shape): raise ValueError("fused KDA Q/K/V weights must have equal shapes")
  if isinstance(qw.device, tuple):
    devices = qw.device
    if qw.uop.axis != 0 or out_features % len(devices): raise ValueError("fused KDA Q/K/V expects output-sharded weights")
    shard_shape = (batch, tokens, out_features//len(devices))
    def make_out() -> Tensor:
      parts = [Tensor.empty(*shard_shape, dtype=dtypes.bfloat16, device=device).uop for device in devices]
      return Tensor(parts[0].mstack(*parts[1:]).unshard(2))
    outs: tuple[Tensor, Tensor, Tensor] = (make_out(), make_out(), make_out())
  else:
    outs = (Tensor.empty(batch, tokens, out_features, dtype=dtypes.bfloat16, device=qw.device),
            Tensor.empty(batch, tokens, out_features, dtype=dtypes.bfloat16, device=qw.device),
            Tensor.empty(batch, tokens, out_features, dtype=dtypes.bfloat16, device=qw.device))
  ret = Tensor.custom_kernel(*outs, x.contiguous(), qw, kw, vw, fxn=_kda_qkv_kernel)
  return ret[0], ret[1], ret[2]

def dual_bf16_matvec(x:Tensor, aw:Tensor, bw:Tensor, fast:bool=False) -> tuple[Tensor, Tensor]:
  """Fuse two equal-shaped BF16 decode projections that consume the same input."""
  from tinygrad.llm.kernels.amd import _dual_bf16_matvec_fast_kernel, _dual_bf16_matvec_kernel
  batch, tokens, _ = x.shape
  out_features = aw.shape[0]
  if aw.shape != bw.shape: raise ValueError("fused BF16 weights must have equal shapes")
  if isinstance(aw.device, tuple) and aw.uop.axis == 0:
    devices = aw.device
    if out_features % len(devices): raise ValueError("fused BF16 output is not divisible by the device count")
    shard_shape = (batch, tokens, out_features//len(devices))
    def make_out() -> Tensor:
      parts = [Tensor.empty(*shard_shape, dtype=dtypes.bfloat16, device=device).uop for device in devices]
      return Tensor(parts[0].mstack(*parts[1:]).unshard(2))
    outs = (make_out(), make_out())
  else:
    outs = (Tensor.empty(batch, tokens, out_features, dtype=dtypes.bfloat16, device=aw.device),
            Tensor.empty(batch, tokens, out_features, dtype=dtypes.bfloat16, device=aw.device))
  ret = Tensor.custom_kernel(*outs, x.contiguous(), aw, bw, fxn=_dual_bf16_matvec_fast_kernel if fast else _dual_bf16_matvec_kernel)
  return ret[0], ret[1]

def dual_input_bf16_matvec(ax:Tensor, bx:Tensor, aw:Tensor, bw:Tensor) -> tuple[Tensor, Tensor]:
  """Fuse equal-shaped BF16 projections with separate inputs and identical TP layouts."""
  from tinygrad.llm.kernels.amd import _dual_input_bf16_matvec_kernel
  if ax.shape != bx.shape or aw.shape != bw.shape: raise ValueError("fused BF16 inputs and weights must have equal shapes")
  batch, tokens, _ = ax.shape
  out_features = aw.shape[0]
  if isinstance(aw.device, tuple) and aw.uop.axis == 0:
    devices = aw.device
    if out_features % len(devices): raise ValueError("fused BF16 output is not divisible by the device count")
    shard_shape = (batch, tokens, out_features//len(devices))
    def make_out() -> Tensor:
      parts = [Tensor.empty(*shard_shape, dtype=dtypes.bfloat16, device=device).uop for device in devices]
      return Tensor(parts[0].mstack(*parts[1:]).unshard(2))
    outs = (make_out(), make_out())
  else:
    outs = (Tensor.empty(batch, tokens, out_features, dtype=dtypes.bfloat16, device=aw.device),
            Tensor.empty(batch, tokens, out_features, dtype=dtypes.bfloat16, device=aw.device))
  ret = Tensor.custom_kernel(*outs, ax.contiguous(), bx.contiguous(), aw, bw, fxn=_dual_input_bf16_matvec_kernel)
  return ret[0], ret[1]

def kda_fgb_linear(x:Tensor, gw:Tensor, fw:Tensor, bw:Tensor) -> tuple[Tensor, Tensor, Tensor]:
  """Fuse replicated KDA g/f low-rank projections with its output-sharded beta projection."""
  from tinygrad.llm.kernels.amd import _kda_fgb_kernel
  if gw.shape != fw.shape or not isinstance(gw.device, tuple) or gw.uop.axis is not None or \
     bw.device != gw.device or bw.uop.axis != 0: raise ValueError("unsupported KDA f/g/beta TP layout")
  batch, tokens, _ = x.shape
  devices, rank, beta_features = gw.device, gw.shape[0], bw.shape[0]
  gout = Tensor.empty(batch, tokens, rank, dtype=dtypes.bfloat16, device=devices)
  fout = Tensor.empty(batch, tokens, rank, dtype=dtypes.bfloat16, device=devices)
  beta_shape = (batch, tokens, beta_features//len(devices))
  parts = [Tensor.empty(*beta_shape, dtype=dtypes.bfloat16, device=device).uop for device in devices]
  bout = Tensor(parts[0].mstack(*parts[1:]).unshard(2))
  ret = Tensor.custom_kernel(gout, fout, bout, x.contiguous(), gw, fw, bw, fxn=_kda_fgb_kernel)
  return ret[0], ret[1], ret[2]

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
