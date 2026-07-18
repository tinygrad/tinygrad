from __future__ import annotations
import functools, itertools, pathlib
from dataclasses import dataclass, replace
from tinygrad import Device, Tensor, nn, UOp, TinyJit, getenv, function, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.llm.gguf import get_ggml_quantization, ggml_data_to_tensor, gguf_load, _GGML_QUANT
from tinygrad.uop.ops import resolve, Ops, KernelInfo, AxisType

def _q8_kernel(quant:UOp, scale:UOp, x:UOp, in_features:int) -> UOp:
  x = x.flatten()
  token, group = UOp.range(quant.shape[0], 0), UOp.range(in_features // 32, 1)
  lane = UOp.range(32, 2, axis_type=AxisType.REDUCE)
  amax = UOp.placeholder((1,), dtypes.float32, 0, addrspace=AddrSpace.REG)
  amax = amax.after(token, group)[0].set(0.0)
  amax = amax[0].set(amax.after(lane)[0].maximum(x[token * in_features + group * 32 + lane].cast(dtypes.float32).abs()), end=lane)
  d = (amax[0] / 127).maximum(1e-8)
  stores = [scale[token, group].store(d)]
  for word_idx in range(8):
    word = UOp.const(dtypes.uint32, 0)
    for byte_idx in range(4):
      value = (x[token * in_features + group * 32 + word_idx * 4 + byte_idx].cast(dtypes.float32) / d).round().maximum(-127).minimum(127)
      byte = value.cast(dtypes.int8).bitcast(dtypes.uint8).cast(dtypes.uint32)
      word = word | (byte << (8 * byte_idx))
    stores.append(quant[token, group, word_idx].store(word))
  return UOp.group(*stores).end(token, group).sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))

def _q8_quantize(x:Tensor, tokens:int, in_features:int) -> tuple[Tensor, Tensor]:
  quant = Tensor.empty(tokens, in_features // 32, 8, dtype=dtypes.uint32, device=x.device)
  scale = Tensor.empty(tokens, in_features // 32, dtype=dtypes.float32, device=x.device)
  return tuple(Tensor.custom_kernel(quant, scale, x,
    fxn=lambda quant,scale,x:_q8_kernel(quant, scale, x, in_features))[:2])  # type: ignore[return-value]

def _amd_dp4a(a:UOp, b:UOp, c:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, dtypes.int32, (a.cast(dtypes.int32), b.cast(dtypes.int32), c),
             arg="__builtin_amdgcn_sudot4(true, {}, true, {}, {}, false)")

def _amd_wave_sum(value:UOp, lane:UOp, lane_count:int) -> UOp:
  assert lane_count in (8, 16, 32)
  for offset in (16, 8, 4, 2, 1)[{32:0, 16:1, 8:2}[lane_count]:]:
    value = value + UOp(Ops.CUSTOM, dtypes.float32, (((lane ^ offset) * 4).cast(dtypes.int32), value),
      arg="__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute({0}, __builtin_bit_cast(int, {1})))")
  return value

def _q8_linear_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int, raw_offset:int=0) -> UOp:
  token_tile = 4 if out.shape[0] % 4 == 0 else 1
  token_block, output = UOp.range(out.shape[0] // token_tile, 0), UOp.range(out_features, 1)
  tokens = tuple(token_block * token_tile + i for i in range(token_tile))
  group_count, lane_count = in_features // 32, min(32, in_features // 32)
  lane = UOp.range(lane_count, 2, axis_type=AxisType.LOCAL)

  def group_dot(group:UOp) -> list[UOp]:
    block = output * group_count + group
    base, odd = raw_offset + block * 8 + block // 2, (block & 1).ne(0)
    dots = [UOp.const(dtypes.int32, 0)] * token_tile
    for word_idx in range(8):
      # Q8_0 blocks are 34 bytes, so their two-byte scale makes alternate blocks word-aligned. Read aligned u32s
      # directly; the other blocks need only two adjacent words instead of four individual byte loads.
      word = odd.where(raw[base + 1 + word_idx], (raw[base + word_idx] >> 16) | (raw[base + 1 + word_idx] << 16))
      dots = [_amd_dp4a(word, xq[token, group, word_idx], dot) for token,dot in zip(tokens, dots)]
    dbits = odd.where(raw[base] >> 16, raw[base] & 0xffff).cast(dtypes.uint16)
    return [dot.cast(dtypes.float32) * xd[token, group] * dbits.bitcast(dtypes.float16).float() for token,dot in zip(tokens, dots)]

  values = [UOp.const(dtypes.float32, 0)] * token_tile
  for offset in range(0, group_count, lane_count):
    dots = group_dot((lane + offset).valid(lane + offset < group_count))
    values = [value + dot for value,dot in zip(values, dots)]
  totals = [_amd_wave_sum(value, lane, lane_count) for value in values]
  stores = [out[token.valid(lane.eq(0)), output].store(total.cast(out.dtype)) for token,total in zip(tokens, totals)]
  return UOp.group(*stores).end(token_block, output, lane).sink(
    arg=KernelInfo(name="linear_q8", opts_to_apply=()))

class Linear(nn.Linear):
  def __init__(self, in_features:int, out_features:int, bias=True):
    super().__init__(in_features, out_features, bias)
    self.in_features, self.out_features = in_features, out_features
    self.ggml_type:int|None = None
  def set_quantized(self, packed:Tensor, ggml_type:int):
    self.weight, self.ggml_type = packed.flatten(), ggml_type
  def prepare(self, x:Tensor) -> tuple[Tensor, Tensor]|None:
    return _q8_quantize(x, int(x.numel()) // self.in_features, self.in_features) \
      if self.ggml_type == 8 and str(self.weight.device).startswith("AMD") else None
  def __call__(self, x:Tensor, prepared:tuple[Tensor, Tensor]|None=None) -> Tensor:
    if self.ggml_type == 8 and str(self.weight.device).startswith("AMD"):
      tokens = int(x.numel()) // self.in_features
      xq, xd = prepared if prepared is not None else _q8_quantize(x, tokens, self.in_features)
      out = Tensor.empty(tokens, self.out_features, dtype=dtypes.float32, device=x.device)
      raw, raw_offset = self.weight.uop, 0
      while raw.op in (Ops.BITCAST, Ops.RESHAPE): raw = raw.src[0]
      while raw.op is Ops.SHRINK:
        raw_offset += raw.src[1].arg * raw.dtype.itemsize
        raw = raw.src[0]
      assert raw_offset % 4 == 0 and raw.dtype == dtypes.uint8
      srcs = (out.uop, raw, xq.uop, xd.uop)
      params = [UOp.placeholder_like(src, slot=i) for i,src in enumerate(srcs)]
      params[1] = params[1].replace(dtype=dtypes.uint32, src=(params[1].src[0] * raw.dtype.itemsize // 4,),
                                    arg=replace(params[1].arg, dtype=dtypes.uint32))
      kernel = _q8_linear_kernel(params[0], params[1], params[2], params[3], self.out_features, self.in_features, raw_offset // 4).call(*srcs)
      out = Tensor(srcs[0].after(kernel)).reshape(*x.shape[:-1], self.out_features)
      return out if self.bias is None else out + self.bias
    return super().__call__(x)

def _packed_expert_kernel(out:UOp, raw:UOp, sel:UOp, xq:UOp, xd:UOp, lut:UOp,
                          out_features:int, in_features:int, ggml_type:int, routes_per_input:int) -> UOp:
  route, output = UOp.range(out.shape[0], 0), UOp.range(out_features, 1)
  group_count, lane_count = in_features // 32, min(32, in_features // 32)
  lane = UOp.range(lane_count, 2, axis_type=AxisType.LOCAL)
  expert, xidx = sel[route].cast(dtypes.index), route // routes_per_input
  type_size = _GGML_QUANT[ggml_type][1]
  expert_size = out_features * in_features // 256 * type_size

  def group_dot(group:UOp) -> UOp:
    block, subgroup = group // 8, group % 8
    base = expert * expert_size + output * (in_features // 256 * type_size) + block * type_size
    dot = UOp.const(dtypes.int32, 0)
    if ggml_type == 21:  # IQ3_S
      for word_idx in range(8):
        qi = raw[base + 2 + subgroup * 8 + word_idx].cast(dtypes.uint16) + \
          (((raw[base + 66 + subgroup] >> word_idx) & 1).cast(dtypes.uint16) << 8)
        word, signs = UOp.const(dtypes.uint32, 0), raw[base + 74 + subgroup * 4 + word_idx // 2]
        for byte_idx in range(4):
          sign = ((signs >> (word_idx % 2 * 4 + byte_idx)) & 1).ne(0).where(-1, 1).cast(dtypes.int8)
          byte = (lut[qi.cast(dtypes.index) * 4 + byte_idx] * sign).cast(dtypes.int8).bitcast(dtypes.uint8).cast(dtypes.uint32)
          word = word | (byte << (8 * byte_idx))
        dot = _amd_dp4a(word, xq[xidx, group, word_idx], dot)
      scale_shift = (4 * (subgroup % 2)).cast(dtypes.uint8)
      scale = 1 + 2 * ((raw[base + 106 + subgroup // 2] >> scale_shift) & 15).cast(dtypes.float32)
    else:  # IQ4_XS
      for word_idx in range(8):
        word = UOp.const(dtypes.uint32, 0)
        for byte_idx in range(4):
          qbyte = raw[base + 8 + subgroup * 16 + (word_idx % 4) * 4 + byte_idx]
          q = (qbyte >> (4 * (word_idx // 4))) & 15
          byte = lut[q.cast(dtypes.index)].cast(dtypes.int8).bitcast(dtypes.uint8).cast(dtypes.uint32)
          word = word | (byte << (8 * byte_idx))
        dot = _amd_dp4a(word, xq[xidx, group, word_idx], dot)
      low = (raw[base + 4 + subgroup // 2] >> (4 * (subgroup % 2)).cast(dtypes.uint8)) & 15
      high_word = raw[base + 2].cast(dtypes.uint16) | (raw[base + 3].cast(dtypes.uint16) << 8)
      scale = ((low.cast(dtypes.uint16) | (((high_word >> (2 * subgroup).cast(dtypes.uint16)) & 3) << 4)).cast(dtypes.uint8).
               bitcast(dtypes.int8)-32).float()
    dbits = raw[base].cast(dtypes.uint16) | (raw[base + 1].cast(dtypes.uint16) << 8)
    return dot.cast(dtypes.float32) * xd[xidx, group] * dbits.bitcast(dtypes.float16).float() * scale

  value = sum((group_dot((lane + offset).valid(lane + offset < group_count)) for offset in range(0, group_count, lane_count)),
              UOp.const(dtypes.float32, 0))
  total = _amd_wave_sum(value, lane, lane_count)
  return out[route.valid(lane.eq(0)), output].store(total.cast(out.dtype)).end(route, output, lane).sink(
    arg=KernelInfo(name=f"expert_q8_{ggml_type}", opts_to_apply=()))

@functools.cache
def _expert_lut(device:str, ggml_type:int) -> Tensor:
  from tinygrad.runtime.autogen import ggml_common
  values = [((word >> (8 * i)) & 0xff) for word in ggml_common.iq3s_grid for i in range(4)] if ggml_type == 21 else ggml_common.kvalues_iq4nl
  return Tensor(values, dtype=dtypes.int8, device=device).contiguous().realize()

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).clone(device)

class ExpertWeights:
  """Like nn.Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.num_experts, self.in_features, self.out_features = num_experts, in_features, out_features
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
    self.ggml_type:int|None = None
  def set_quantized(self, weight:Tensor, packed:Tensor, ggml_type:int):
    assert weight.shape == (self.num_experts, self.out_features, self.in_features)
    self.weight, self.ggml_type = packed.flatten(), ggml_type
  def prepare(self, x:Tensor) -> tuple[Tensor, Tensor]:
    return _q8_quantize(x, int(x.numel()) // self.in_features, self.in_features)
  def __call__(self, sel:Tensor, x:Tensor, prepared:tuple[Tensor, Tensor]|None=None) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    ggml_type = self.ggml_type
    if ggml_type in (21, 23) and str(self.weight.device).startswith("AMD"):
      input_count = int(x.numel()) // self.in_features
      routes_per_input = int(sel.numel()) // input_count
      xq, xd = prepared if prepared is not None else self.prepare(x)
      flat_sel = sel if len(sel.shape) == 1 else sel.flatten().clone()
      out = Tensor.empty(int(sel.numel()), self.out_features, dtype=dtypes.float32, device=x.device)
      out = Tensor.custom_kernel(out, self.weight, flat_sel, xq, xd, _expert_lut(str(x.device), ggml_type),
        fxn=lambda out,raw,sel,xq,xd,lut:_packed_expert_kernel(out, raw, sel, xq, xd, lut, self.out_features,
                                                              self.in_features, ggml_type, routes_per_input))[0]
      return out if len(sel.shape) == 1 else out.reshape(*sel.shape, self.out_features)
    if self.ggml_type is None: weight = self.weight[sel]
    else:
      packed = self.weight.reshape(self.num_experts, -1)[sel].flatten()
      weight = ggml_data_to_tensor(packed, int(sel.numel()) * self.out_features * self.in_features,
                                    self.ggml_type, contiguous=False).reshape(*sel.shape, self.out_features, self.in_features)
      if getenv("HALF", 1): weight = weight.cast('float16')
    return (x.unsqueeze(-2) @ weight.transpose(-1, -2)).contiguous().squeeze(-2)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
  sel = x.const_like(0).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

def _inverse_unit_lower_kernel(out:UOp, x:UOp, n:int) -> UOp:
  outer_count = 1
  for dim in out.shape[:-2]:
    assert isinstance(dim, int)
    outer_count *= dim
  outer, lane = UOp.range(outer_count, 0), UOp.range(n, 1, axis_type=AxisType.LOCAL)
  raw = UOp.placeholder((n*n,), x.dtype, 0, addrspace=AddrSpace.LOCAL)
  solved = UOp.placeholder((n*n,), x.dtype, 1, addrspace=AddrSpace.LOCAL)
  ready = UOp.group(*(raw[row*n+lane].store(x.flatten()[outer*n*n+row*n+lane]) for row in range(n))).barrier()
  for row in range(n):
    base, previous = raw.after(ready), solved.after(ready)
    value = base[row*n+lane] + sum((base[row*n+i] * previous[i*n+lane] for i in range(row)), UOp.const(x.dtype, 0))
    ready = solved.after(ready)[row*n+lane].store((lane < row).where(value, UOp.const(x.dtype, 0))).barrier()
  result = solved.after(ready)
  stores = [out.flatten()[outer*n*n+row*n+lane].store(lane.eq(row).where(UOp.const(x.dtype, 1), result[row*n+lane]))
            for row in range(n)]
  return UOp.group(*stores).end(outer, lane).sink(arg=KernelInfo(name="inverse_unit_lower", opts_to_apply=()))

def inverse_unit_lower(x:Tensor) -> Tensor:
  """Reference-ordered inverse of I-x for a strictly lower-triangular x."""
  n = x.shape[-1]
  assert isinstance(n, int)
  if n == 64 and str(x.device).startswith("AMD"):
    out = Tensor.empty(*x.shape, dtype=x.dtype, device=x.device)
    return Tensor.custom_kernel(out, x, fxn=lambda out,x:_inverse_unit_lower_kernel(out, x, n))[0]
  rows = [x[..., 0, :].const_like(0)]
  for i in range(1, n):
    prefix = x[..., i, :i]
    previous = Tensor.stack(*rows, dim=-2)[..., :, :i]
    rows.append((prefix + (prefix.unsqueeze(-1) * previous).sum(-2)).pad((0, n-i)))
  return Tensor.stack(*rows, dim=-2) + Tensor.eye(n, dtype=x.dtype)

def l2norm(x:Tensor) -> Tensor: return x * (x.square().sum(-1, keepdim=True) + 1e-6).rsqrt()

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int

@dataclass(frozen=True)
class TransformerConfig:
  num_blocks: int
  dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  norm_eps: float
  vocab_size: int
  head_dim: int
  rope_theta: float
  rope_dim: int
  v_head_dim: int
  max_context: int = 0
  qk_norm: int = 0
  num_experts: int = 0
  num_experts_per_tok: int = 0
  norm_topk_prob: bool = False
  q_lora_rank: int = 0
  kv_lora_rank: int = 0
  shared_expert_dim: int = 0
  full_attention_interval: int = 0
  attn_output_gate: bool = False
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  qkv_bias: bool = False
  expert_bias: bool = False

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config
    self.pending_state:tuple[Tensor, Tensor]|None = None

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = Linear(config.dim, config.num_experts, bias=False)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      self.ffn_gate_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_up_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, config.dim)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate    = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = Linear(config.hidden_dim, config.dim, bias=False)

  def _feed_forward(self, x:Tensor) -> Tensor:
    if hasattr(self, 'ffn_gate_exps'):
      h = x.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      prepared = self.ffn_gate_exps.prepare(h) if self.ffn_gate_exps.ggml_type in (21, 23) and str(h.device).startswith("AMD") else None
      logits = self.ffn_gate_inp(x, prepared)
      if hasattr(self, 'exp_probs_b'):
        probs = logits.sigmoid()
        _, sel = pairwise_topk(probs + self.exp_probs_b["bias"], self.config.num_experts_per_tok)
        probs = probs.gather(-1, sel)
        if self.config.norm_topk_prob: probs = probs / probs.sum(axis=-1, keepdim=True)
      else:
        vals, sel = pairwise_topk(logits, self.config.num_experts_per_tok)
        probs = vals.softmax(-1) if self.config.norm_topk_prob else logits.softmax(-1).gather(-1, sel)
      probs = probs * self.config.routed_scaling_factor
      if prepared is not None:
        flat_sel = sel.flatten().clone()
        gate, up = self.ffn_gate_exps(flat_sel, h, prepared), self.ffn_up_exps(flat_sel, h, prepared)
        x_down = self.ffn_down_exps(flat_sel, (gate.silu() * up).contiguous()).reshape(*sel.shape, self.config.dim)
      else: x_down = self.ffn_down_exps(sel, (self.ffn_gate_exps(sel, h).silu() * self.ffn_up_exps(sel, h)).contiguous())
      out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      if hasattr(self, 'ffn_gate_shexp'):
        shexp = self.ffn_down_shexp(self.ffn_gate_shexp(x, prepared).silu().contiguous() * self.ffn_up_shexp(x, prepared))
        if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out
    # TODO: remove the need for this contiguous
    prepared = self.ffn_gate.prepare(x)
    return self.ffn_down(self.ffn_gate(x, prepared).silu().contiguous() * self.ffn_up(x, prepared))

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  # return writes that reset this block's state after a cache mismatch
  def _state_reset_ops(self) -> list[Tensor]: return []
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp, use_flash:bool=False, kv_len:int|UOp|None=None,
                 valid_len:int|UOp|None=None) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp, use_flash:bool=False, kv_len:int|UOp|None=None, valid_len:int|UOp|None=None):
    self._init_state(x)
    if hasattr(self, 'ssm_a'):
      self.pending_state = None
      @function(precompile=True, allow_implicit=True)
      def _run_stateful(x:Tensor, start_pos:int|UOp, valid_len:int|UOp|None):
        h = x + self._attention(self.attn_norm(x), start_pos, use_flash, kv_len, valid_len)
        out = (h + self._feed_forward(self.ffn_norm(h))).contiguous()
        assert self.pending_state is not None
        return out, *self.pending_state
      out, conv_state, recurrent_state = _run_stateful(x, start_pos, valid_len)
      stores = (getattr(self, "conv_state").uop.store(conv_state.uop), getattr(self, "recurrent_state").uop.store(recurrent_state.uop))
      state = getattr(self, "recurrent_state").uop.after(*stores)
      return Tensor(out.uop.after(state))
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    def _run(x:Tensor, start_pos:int|UOp):
      h =     x + self._attention(self.attn_norm(x), start_pos, use_flash, kv_len)
      return (h + self._feed_forward(self.ffn_norm(h))).contiguous()
    return function(precompile=True, allow_implicit=True)(_run)(x, start_pos)

class TransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    assert config.v_head_dim == config.head_dim, "TransformerBlock requires v_head_dim == head_dim"

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = config.head_dim * config.n_heads * (2 if config.attn_output_gate else 1)
    kv_proj_out      = config.head_dim * config.n_kv_heads
    self.attn_q      = Linear(config.dim, q_proj_out,  bias=config.qkv_bias)
    self.attn_k      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_v      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_output = Linear(config.head_dim * config.n_heads, config.dim, bias=False)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)

  def _attention(self, x:Tensor, start_pos:int|UOp, use_flash:bool=False, kv_len:int|UOp|None=None,
                 valid_len:int|UOp|None=None) -> Tensor:
    prepared = self.attn_q.prepare(x)
    q, k, v = self.attn_q(x, prepared), self.attn_k(x, prepared), self.attn_v(x, prepared)
    if self.config.qk_norm and self.config.qk_norm != self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape
    if self.config.attn_output_gate:
      qg = q.reshape(B, T, self.config.n_heads, 2, self.config.head_dim)
      q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, self.config.n_heads * self.config.head_dim)
    q = q.reshape(B, T, self.config.n_heads,    self.config.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.config.qk_norm == self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    q = apply_rope(q[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1)
    k = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1)

    # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
    assigned_kv = Tensor(self.cache_kv.uop.after(
      self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).cast(self.cache_kv.dtype).uop)))
    cache_len = start_pos + T if kv_len is None else kv_len
    k = assigned_kv[0, :, :, 0:cache_len, :]
    v = assigned_kv[1, :, :, 0:cache_len, :]

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    flash_decode = resolve(T == 1) and kv_len is not None and str(x.device).startswith("AMD") and self.config.head_dim == 256
    if flash_decode:
      decode_len = self.config.max_context
      decode_pos = (start_pos.unbind()[0] if isinstance(start_pos, UOp) else start_pos) + 1
      decode_mask = (Tensor.arange(decode_len) < Tensor(decode_pos)) \
        .where(0.0, float("-inf")).reshape(1, 1, 1, decode_len)
      attn = q.scaled_dot_product_attention(assigned_kv[0, :, :, :decode_len], assigned_kv[1, :, :, :decode_len],
                                            attn_mask=decode_mask, enable_gqa=True)
    elif use_flash:
      from extra.gemm.amd_flash_attention import amd_flash_attention_causal_cached
      valid_kv_len = ((start_pos.unbind()[0] + 1) if isinstance(start_pos, UOp) else start_pos + 1) if flash_decode else \
                     (start_pos.unbind()[0] if isinstance(start_pos, UOp) else start_pos) + T
      q_flat = q.half().reshape(B*self.config.n_heads, T, self.config.head_dim)
      out = Tensor.empty(B*self.config.n_heads, q_flat.shape[1], self.config.head_dim, dtype="float32", device=x.device)
      attn = Tensor.custom_kernel(out, q_flat, assigned_kv,
        fxn=functools.partial(amd_flash_attention_causal_cached, valid_kv_len=valid_kv_len))[0] \
        .reshape(B, self.config.n_heads, q_flat.shape[1], -1)
    else:
      mask:Tensor|None
      if kv_len is not None:
        mask = None if resolve(T == 1) and self.config.ssm is not None else \
          Tensor.full((1, 1, 1, kv_len), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1)
      else:
        mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
          if resolve(T != 1) else None
      attn = q.float().scaled_dot_product_attention(k.float(), v.float(), attn_mask=mask, enable_gqa=True)  # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # TODO: how is the dtype of this determined?
      # Decode uses fixed-size KV buckets. Unwritten entries must be zero: masking happens after QK, so values left
      # uninitialized by Tensor.empty can inject NaNs before the mask is applied.
      self.cache_kv = Tensor.zeros(2, x.shape[0], self.config.n_kv_heads, self.config.max_context+192, self.config.head_dim,
                                   dtype="float16", device=x.device).contiguous()
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context+192, self.config.rope_theta, device=x.device)

class MLATransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    if config.q_lora_rank > 0:
      self.attn_q_a = Linear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.attn_kv_a_mqa = Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp, use_flash:bool=False, kv_len:int|UOp|None=None,
                 valid_len:int|UOp|None=None) -> Tensor:
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_proj = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x))) if self.config.q_lora_rank > 0 else self.attn_q(x)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(apply_rope(q_rope, self.freqs_cis[start_pos:start_pos+T]), dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = apply_rope(
      kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2),
      self.freqs_cis[start_pos:start_pos+T])

    k_store = c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1)
    k = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.uop)))[:, :, 0:start_pos+T, :]
    v = k[..., :self.config.kv_lora_rank]

    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    attn = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    if mask is not None: attn = attn + mask
    attn = attn.softmax(-1)
    attn = ((attn @ v) @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return self.attn_output(attn)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context+192, self.config.kv_lora_rank + self.config.rope_dim, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context+192, self.config.rope_theta, device=x.device)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    self.attn_qkv, self.attn_gate = Linear(config.dim, self.conv_channels, bias=False), Linear(config.dim, ssm.inner_size, bias=False)
    self.ssm_alpha, self.ssm_beta = Linear(config.dim, self.num_v_heads, bias=False), Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), Linear(ssm.inner_size, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp, use_flash:bool=False, kv_len:int|UOp|None=None,
                 valid_len:int|UOp|None=None) -> Tensor:
    B, T, _ = x.shape
    conv_state, initial_state = self.conv_state, self.recurrent_state

    if T == 1:
      x = x.half()
      prepared = self.attn_gate.prepare(x)
      out_gate = self.attn_gate(x, prepared).reshape(B, 1, self.num_v_heads, self.head_v_dim)
      beta = self.ssm_beta(x, prepared).sigmoid().reshape(B, self.num_v_heads, 1, 1)
      alpha = ((self.ssm_alpha(x, prepared).float() + self.ssm_dt["bias"]).softplus() * self.ssm_a).reshape(B, self.num_v_heads, 1, 1).exp()
      conv_window = conv_state.cat(self.attn_qkv(x, prepared), dim=1)
      conv_out = (conv_window * self.ssm_conv1d["weight"].T.unsqueeze(0)).sum(1).silu()
      q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
      q = l2norm(q.reshape(B, self.num_k_heads, self.head_k_dim)).repeat(1, self.num_v_heads//self.num_k_heads, 1)
      k = l2norm(k.reshape(B, self.num_k_heads, self.head_k_dim)).repeat(1, self.num_v_heads//self.num_k_heads, 1)
      v = v.reshape(B, self.num_v_heads, self.head_v_dim)
      q, k, v = q.mul(self.head_k_dim**-0.5).unsqueeze(-1), k.unsqueeze(-1), v.unsqueeze(-1)
      recurrent_state = initial_state * alpha
      recurrent_state = recurrent_state + ((v - recurrent_state@k) * beta)@k.transpose(-1, -2)
      self.pending_state = (conv_window[:, 1:, :].cast(self.conv_state.dtype).contiguous(),
                            recurrent_state.cast(self.recurrent_state.dtype).contiguous())
      core_attn_out = self.ssm_norm((recurrent_state@q).squeeze(-1).reshape(B, 1, self.num_v_heads, self.head_v_dim))
      return self.ssm_out((core_attn_out * out_gate.silu()).reshape(B, 1, -1).cast(x.dtype))

    # Batched projections and causal depthwise convolution.
    x = x.half()
    prepared = self.attn_gate.prepare(x)
    out_gate = self.attn_gate(x, prepared).reshape(B, T, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x, prepared).sigmoid().reshape(B, T, self.num_v_heads)
    log_alpha = ((self.ssm_alpha(x, prepared).float() + self.ssm_dt["bias"]).softplus() * self.ssm_a).reshape(B, T, self.num_v_heads)
    if valid_len is not None:
      active = (Tensor.arange(T) < Tensor(valid_len)).reshape(1, T, 1)
      beta, log_alpha = beta * active, log_alpha * active
    conv_window = conv_state.cat(self.attn_qkv(x, prepared), dim=1)
    conv_out = functools.reduce(lambda a,b: a+b,
      (conv_window[:, i:i+T] * self.ssm_conv1d["weight"][:, i] for i in range(self.ssm_conv_kernel))).silu()
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    q = l2norm(q.reshape(B, T, self.num_k_heads, self.head_k_dim)).repeat(1, 1, self.num_v_heads//self.num_k_heads, 1)
    k = l2norm(k.reshape(B, T, self.num_k_heads, self.head_k_dim)).repeat(1, 1, self.num_v_heads//self.num_k_heads, 1)
    v = v.reshape(B, T, self.num_v_heads, self.head_v_dim)

    # Chunked gated delta rule. The strictly-lower update is the triangular solve from the reference implementation.
    q, k, v, beta, log_alpha = [z.transpose(1, 2).float() for z in (q, k, v, beta, log_alpha)]
    q = q * self.head_k_dim**-0.5
    state = initial_state.transpose(-1, -2).float()
    core_chunks = []
    for start in range(0, T, 64):
      qc, kc, vc, bc, gc = q[:,:,start:start+64], k[:,:,start:start+64], v[:,:,start:start+64], \
                            beta[:,:,start:start+64], log_alpha[:,:,start:start+64]
      chunk_len = qc.shape[2]
      g = (gc @ Tensor.ones(chunk_len, chunk_len, dtype=gc.dtype).tril().T).contiguous()
      decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).exp().tril().contiguous()
      base = (-(kc * bc.unsqueeze(-1) @ kc.transpose(-1, -2) * decay).tril(-1)).contiguous()
      attn = inverse_unit_lower(base)
      value = attn @ (vc * bc.unsqueeze(-1))
      k_cumdecay = attn @ (kc * bc.unsqueeze(-1) * g.exp().unsqueeze(-1))
      value = value - k_cumdecay @ state
      core_chunks.append((qc * g.exp().unsqueeze(-1)) @ state + (qc @ kc.transpose(-1, -2) * decay) @ value)
      state = state * g[..., -1, None, None].exp() + \
              (kc * (g[..., -1, None] - g).exp().unsqueeze(-1)).transpose(-1, -2) @ value
    core_attn_out = functools.reduce(lambda a,b: a.cat(b, dim=2), core_chunks)
    core_attn_out = self.ssm_norm(core_attn_out.transpose(1, 2))
    out = self.ssm_out((core_attn_out * out_gate.silu()).reshape(B, T, -1).cast(x.dtype)).contiguous()

    state_pos = T if valid_len is None else valid_len
    self.pending_state = (conv_window[:, state_pos:state_pos+self.ssm_conv_kernel-1, :].cast(self.conv_state.dtype).contiguous(),
                          state.transpose(-1, -2).cast(self.recurrent_state.dtype).contiguous())
    return out

  # recurrent state can't be partially reused after divergence, force a full rebuild
  def _state_reset_ops(self):
    return [self.conv_state.assign(self.conv_state.const_like(0)),
            self.recurrent_state.assign(self.recurrent_state.const_like(0))] if hasattr(self, "conv_state") else []
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return 0 if prefix_len != cached_len else prefix_len

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_v_dim, device=x.device).clone()

class Transformer:
  def __init__(self, config:TransformerConfig):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
    self.blk:list[FFNBlock] = [GatedDeltaNetBlock(config, config.ssm) if config.ssm and (i+1) % config.full_attention_interval != 0 else
                               block_cls(dense_config if i < config.leading_dense_blocks else config) for i in range(config.num_blocks)]
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.parameter_count = 0
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.flash_prefill_jit = TinyJit(functools.partial(self.forward, use_flash=True))
    self.sample_prefill_jit = TinyJit(functools.partial(self.forward, sample=True))
    self.rollout_jits:dict[int, TinyJit] = {}
    self.sample_rollout_jits:dict[int, TinyJit] = {}

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, use_flash:bool=False, kv_len:int|UOp|None=None,
              valid_len:int|UOp|None=None, sample:bool=False) -> Tensor:
    x = self.token_embd(tokens).float()                   # (B, T, D)
    for block in self.blk:
      x = block(x, start_pos, use_flash, kv_len, valid_len)
    last = x[:, -1:] if valid_len is None else x[:, valid_len-1:valid_len]
    logits = self.output(self.output_norm(last))[:, -1, :]
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    if not sample: return logits.argmax(-1, keepdim=True)
    return (logits / temperature - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def forward_recurrent_decode(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, valid_len:int|UOp|None=None,
                               sample:bool=False) -> Tensor:
    return self.forward(tokens, start_pos, temperature, kv_len=start_pos+1, valid_len=valid_len, sample=sample)

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, use_flash:bool=False,
               valid_len:int|UOp|None=None, sample:bool=False) -> Tensor:
    jit_kwargs = {"valid_len":valid_len}
    if resolve(tokens.shape[1] == 1):
      pos = start_pos.unbind()[1] if isinstance(start_pos, UOp) else start_pos
      if self.has_recurrent_block:
        key = 0
      else:
        min_bucket = max(1, getenv("DECODE_BUCKET", 256))
        kv_len = key = min(self.max_context, max(min_bucket, 1 << pos.bit_length()))
      rollout_jits = self.sample_rollout_jits if sample else self.rollout_jits
      if key not in rollout_jits:
        rollout_jits[key] = TinyJit(functools.partial(self.forward_recurrent_decode, sample=sample) if self.has_recurrent_block else
                                    functools.partial(self.forward, kv_len=kv_len, sample=sample))
      jit = rollout_jits[key]
    else:
      jit = self.sample_prefill_jit if sample else self.flash_prefill_jit if use_flash else self.prefill_jit
    ret = jit(tokens.contiguous(), start_pos, temperature, **jit_kwargs)
    return ret[0] if isinstance(ret, tuple) else ret

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0))) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    kv, state_dict = gguf_load(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    ssm = None
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
    if arch in ('qwen35', 'qwen35moe', 'glm4moe'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
      if ('attn_q.weight' in name or 'attn_q_b.weight' in name) and (arch == 'llama' or kv_lora_rank):
        w = state_dict[name].reshape(n_heads, state_dict[name].shape[0]//n_heads, -1)
        prefix = head_dim-rope_dim
        state_dict[name] = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(-1, w.shape[-1])
      elif arch == 'llama' and 'attn_k.weight' in name:
        w = state_dict[name].reshape(n_kv_heads, state_dict[name].shape[0]//n_kv_heads, -1)
        state_dict[name] = w.rearrange("n (h two) d -> n (two h) d", two=2).reshape(-1, w.shape[-1])
      elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
        state_dict[name] = state_dict[name][:kv_lora_rank].cat(state_dict[name][kv_lora_rank:].rearrange("(h two) d -> (two h) d", two=2), dim=0)
    config = TransformerConfig(
      num_blocks=kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0), dim=kv[f'{arch}.embedding_length'],
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0)),
      n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim,
      rope_theta=kv[f'{arch}.rope.freq_base'],
      rope_dim=rope_dim,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
      qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe')),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe'), ssm=ssm,
      full_attention_interval=kv.get(f'{arch}.full_attention_interval', 0),
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict)
    model = Transformer(config)
    model.parameter_count = sum(int(weight.numel()) for weight in state_dict.values())
    packed_weights:set[str] = set()
    def resolve_owner(path:list[str]):
      obj = model
      for part in path: obj = obj[int(part)] if isinstance(obj, list) else getattr(obj, part)
      return obj
    for name, weight in state_dict.items():
      parts = name.split('.')
      quantization = get_ggml_quantization(weight)
      if quantization is not None and quantization[1] == 8 and parts[-1] == "weight" and isinstance(owner:=resolve_owner(parts[:-1]), Linear):
        owner.set_quantized(*quantization)
        state_dict[name], packed_weights = owner.weight, packed_weights | {name}
      elif len(parts) == 4 and parts[0] == "blk" and parts[2].endswith("_exps") and parts[3] == "weight" and quantization is not None:
        expert_weights = getattr(model.blk[int(parts[1])], parts[2])
        expert_weights.set_quantized(weight, *quantization)
        state_dict[name], packed_weights = expert_weights.weight, packed_weights | {name}

    state_dict = {k:v if k in packed_weights else v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    expert_types = {getattr(block, name).ggml_type for block in model.blk if hasattr(block, "ffn_gate_exps")
                    for name in ("ffn_gate_exps", "ffn_down_exps")}
    for ggml_type in expert_types:
      if ggml_type in (21, 23) and str(model.token_embd.weight.device).startswith("AMD"): _expert_lut(str(model.token_embd.weight.device), ggml_type)
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    return model, kv

  def get_start_pos(self, tokens:list[int]) -> int:
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def warmup(self, chunk_size:int=256):
    direct_capture = not self.has_recurrent_block and all(isinstance(block, TransformerBlock) for block in self.blk)
    if direct_capture:
      device = str(self.token_embd.weight.device)
      direct_capture = device.startswith("AMD") and Device[device].renderer.target.arch.startswith("gfx11")

    # Capture both prefill JITs. Different first tokens prevent the second pass from reusing the first pass's KV cache.
    recurrent_chunk = min(chunk_size, 192)
    warm_len = min(recurrent_chunk * 3 + 1 if self.has_recurrent_block else chunk_size * 2, self.max_context - 1)
    if warm_len > 0:
      if direct_capture:
        x = Tensor.zeros(1, 1, self.blk[0].config.dim)
        for block in self.blk: block._init_state(x)
        Tensor.realize(*[state for block in self.blk for state in (getattr(block, "cache_kv"), getattr(block, "freqs_cis"))])
        self.prefill_jit.cnt = self.flash_prefill_jit.cnt = 1
        next(self.generate([0] * warm_len, chunk_size=chunk_size))
      elif self.has_recurrent_block:
        warm = self.generate([0] * warm_len, chunk_size=chunk_size)
        next(warm)
        next(warm)
        next(warm)
      else:
        for salt in range(2): next(self.generate([salt] + [0] * (warm_len - 1), chunk_size=chunk_size))

    # Rollout uses fixed power-of-two KV shapes. Capture every shape up front so requests never pay a JIT transition.
    if not self.has_recurrent_block:
      min_bucket = max(1, getenv("DECODE_BUCKET", 256))
      bucket_positions:dict[int, int] = {}
      for pos in [0] + [1 << i for i in range(self.max_context.bit_length())]:
        bucket = min(self.max_context, max(min_bucket, 1 << pos.bit_length()))
        bucket_positions.setdefault(bucket, pos)
      v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
      token, temperature = Tensor([[0]], dtype="int32"), Tensor([0.0])
      for bucket, pos in sorted(bucket_positions.items()):
        if direct_capture:
          self.rollout_jits[bucket] = TinyJit(functools.partial(self.forward, kv_len=bucket))
          self.rollout_jits[bucket].cnt = 1
        for _ in range(1 if direct_capture else 2):
          result = self(token, v_start_pos.bind(pos), temperature)
          assert isinstance(result, Tensor)
          result.realize()

    if resets := [r for block in self.blk for r in block._state_reset_ops()]: Tensor.realize(*resets)
    self._cached_tokens = []

  def generate(self, tokens:list[int], chunk_size:int=256, temperature:float=0.0):
    if self.has_recurrent_block: chunk_size = min(chunk_size, 192)
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    temp = Tensor([temperature])
    # assign all input tokens once, then slice from start_pos for the model call
    t = Tensor(tokens + [0] * (self.max_context + chunk_size - len(tokens)), dtype="int32").reshape(1, self.max_context + chunk_size)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    if start_pos < len(self._cached_tokens) and (resets := [r for b in self.blk for r in b._state_reset_ops()]): Tensor.realize(*resets)
    out, prompt_len = None, len(tokens)
    while len(tokens) < self.max_context:
      remaining = len(tokens) - start_pos
      can_flash = bool(getenv("AMD_FLASH_ATTENTION", 1)) and start_pos > 0 and remaining >= chunk_size and chunk_size % 64 == 0
      if can_flash:
        device = str(self.token_embd.weight.device)
        can_flash = device.startswith("AMD") and Device[device].renderer.target.arch.startswith("gfx11")
      use_flash = can_flash and start_pos % 64 == 0
      sp = v_start_pos.bind(start_pos)
      # The flash kernel requires its cached prefix to start on a 64-token tile. Cache reuse can resume at any
      # token, so process one short generic chunk to reach the next tile boundary before entering flash prefill.
      actual_nt = min(chunk_size, remaining)
      nt = chunk_size if use_flash or self.has_recurrent_block and start_pos < prompt_len else 1 if self.has_recurrent_block else \
           v_toks.bind(min(64 - start_pos % 64, remaining) if can_flash else actual_nt)
      inp = t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out
      valid_len = v_toks.bind(actual_nt) if self.has_recurrent_block and nt == chunk_size else None
      result = self(inp, sp, temp, use_flash=True, valid_len=valid_len, sample=temperature > 0) if use_flash else \
               self(inp, sp, temp, valid_len=valid_len, sample=temperature > 0)
      out = result.realize()
      start_pos += actual_nt if self.has_recurrent_block else nt if isinstance(nt, int) else nt.val
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]
