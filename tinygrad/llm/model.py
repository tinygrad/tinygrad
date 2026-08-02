from __future__ import annotations
import functools, itertools, pathlib
from dataclasses import dataclass, replace
from typing import Any
from tinygrad import Device, Tensor, nn, UOp, TinyJit, getenv, function, dtypes, Context
from tinygrad.device import Buffer, BufferSpec
from tinygrad.llm.kernels import amd as llm_amd
from tinygrad.llm.gguf import get_ggml_quantization, gguf_load
from tinygrad.uop.ops import resolve, Ops

class PackedWeight:
  weight:Tensor
  def _init_packed(self):
    self.ggml_type:int|None = None
    self._raw_uop:UOp|None = None
    self._raw_offset_uop:UOp|None = None
  def set_quantized(self, packed:Tensor, ggml_type:int):
    self.weight, self.ggml_type = packed.flatten(), ggml_type
    self._raw_uop = self._raw_offset_uop = None
    # IQ4 prefill uses a device LUT. Build it with the weights, before this linear can be captured by TinyJit.
    if ggml_type == 23 and str(packed.device).startswith("AMD"): llm_amd.iq4_half_lut(str(packed.device))
  def _packed_offset(self) -> Tensor:
    raw, raw_offset = self.weight.uop, 0
    while raw.op in (Ops.BITCAST, Ops.RESHAPE): raw = raw.src[0]
    while raw.op is Ops.SHRINK:
      raw_offset += raw.src[1].arg * raw.dtype.itemsize
      raw = raw.src[0]
    assert raw_offset % 4 == 0 and raw.dtype == dtypes.uint8
    self._raw_uop = raw
    return Tensor([raw_offset // 4], dtype=dtypes.uint64, device=self.weight.device)
  def _prepare_packed(self):
    self._raw_offset_uop = self._packed_offset().realize().uop

class Linear(nn.Linear, PackedWeight):
  def __init__(self, in_features:int, out_features:int, bias=True):
    # GGUF loading replaces every LLM weight. Lazy zeros avoid constructing hundreds of random-init graphs first,
    # while keeping directly-created test models deterministic and valid.
    self.weight = Tensor.zeros(out_features, in_features)
    self.bias = Tensor.zeros(out_features) if bias else None
    self.in_features, self.out_features = in_features, out_features
    self._init_packed()
  def prepare(self, x:Tensor, with_sum:bool=False) -> tuple[Tensor, ...]|None:
    if (with_sum or self.ggml_type == 13) and self.ggml_type in (13, 14, 23) and \
       str(self.weight.device).startswith("AMD"):
      return llm_amd.q8_quantize_sum(x, int(x.numel()) // self.in_features, self.in_features)
    return llm_amd.q8_quantize(x, int(x.numel()) // self.in_features, self.in_features) \
      if self.ggml_type in (14, 23) and str(self.weight.device).startswith("AMD") else None
  def __call__(self, x:Tensor, prepared:tuple[Tensor, ...]|None=None) -> Tensor:
    if self.ggml_type in (13, 14, 23) and str(self.weight.device).startswith("AMD"):
      return llm_amd.q8_linear(self, x, prepared)
    return super().__call__(x)

class Embedding(nn.Embedding, PackedWeight):
  def __init__(self, vocab_size:int, embed_size:int):
    self.weight = Tensor.zeros(vocab_size, embed_size)
    self.vocab_size, self.embed_size = vocab_size, embed_size
    self._init_packed()
  def __call__(self, idx:Tensor) -> Tensor:
    if self.ggml_type == 12 and str(self.weight.device).startswith("AMD"): return llm_amd.q4_embedding(self, idx)
    return super().__call__(idx)

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2).to(device)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).to(device).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  table = freqs.cos().cat(freqs.sin(), dim=-1)
  if device is not None and str(device).startswith("AMD") and end > 8192:
    size = table.numel()
    assert isinstance(size, int)
    storage = Buffer(str(device), size, table.dtype, options=BufferSpec(host=True))
    return Tensor(UOp.from_buffer(storage).reshape(table.shape)).assign(table).realize()
  return table.clone(device)

class ExpertWeights:
  """Like nn.Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)

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

def l2norm(x:Tensor) -> Tensor: return x * (x.square().sum(-1, keepdim=True) + 1e-6).rsqrt()

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int
  kda: bool = False

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
  ssm_layers: tuple[bool, ...] = ()
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
    self.attn_norm, self.ffn_norm = nn.RMSNorm(config.dim, config.norm_eps), nn.RMSNorm(config.dim, config.norm_eps)
    if config.num_experts > 0:
      self.ffn_gate_inp = nn.Linear(config.dim, config.num_experts, bias=False)
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      self.ffn_gate_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_up_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, config.dim)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = nn.Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = nn.Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = nn.Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate, self.ffn_up = Linear(config.dim, config.hidden_dim, bias=False), Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down = Linear(config.hidden_dim, config.dim, bias=False)

  def _feed_forward(self, x:Tensor) -> Tensor:
    if hasattr(self, 'ffn_gate_exps'):
      h, logits = x.unsqueeze(2), self.ffn_gate_inp(x)
      if hasattr(self, 'exp_probs_b'):
        probs = logits.sigmoid()
        _, sel = pairwise_topk(probs + self.exp_probs_b["bias"], self.config.num_experts_per_tok)
        probs = probs.gather(-1, sel)
        if self.config.norm_topk_prob: probs = probs / probs.sum(axis=-1, keepdim=True)
      else:
        vals, sel = pairwise_topk(logits, self.config.num_experts_per_tok)
        probs = vals.softmax(-1) if self.config.norm_topk_prob else logits.softmax(-1).gather(-1, sel)
      probs = probs * self.config.routed_scaling_factor
      x_down = self.ffn_down_exps(sel, (self.ffn_gate_exps(sel, h).silu() * self.ffn_up_exps(sel, h)).contiguous())
      out = (x_down * probs.unsqueeze(-1)).sum(axis=2)
      if hasattr(self, 'ffn_gate_shexp'):
        shexp = self.ffn_down_shexp(self.ffn_gate_shexp(x).silu().contiguous() * self.ffn_up_shexp(x))
        if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out
    prepared = self.ffn_gate.prepare(x)
    gate, up = self.ffn_gate(x, prepared), self.ffn_up(x, prepared)
    return self.ffn_down(gate.silu() * up)

  def _normalized_feed_forward(self, x:Tensor) -> Tensor:
    return self._feed_forward(self.ffn_norm(x))

  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  def _state_reset_ops(self) -> list[Tensor]: return []
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp, use_flash:bool=False, kv_len:int|UOp|None=None,
                 valid_len:int|UOp|None=None, input_norm:nn.RMSNorm|None=None) -> Tensor: raise NotImplementedError

  def __call__(self, x:Tensor, start_pos:int|UOp, use_flash:bool=False, kv_len:int|UOp|None=None, valid_len:int|UOp|None=None):
    self._init_state(x)
    if hasattr(self, 'ssm_a'):
      self.pending_state, self.pending_recurrent_inplace = None, False
      @function(precompile=True, allow_implicit=True)
      def _run_stateful(x:Tensor, start_pos:int|UOp, valid_len:int|UOp|None):
        h = x + self._attention(self.attn_norm(x), start_pos, use_flash, kv_len, valid_len)
        out = (h + self._normalized_feed_forward(h)).contiguous()
        assert self.pending_state is not None
        return (out, self.pending_state[0]) if self.pending_recurrent_inplace else (out, *self.pending_state)
      stateful_out = _run_stateful(x, start_pos, valid_len)
      out, conv_state = stateful_out[:2]
      recurrent_state = getattr(self, "recurrent_state")
      stores = [getattr(self, "conv_state").uop.store(conv_state.uop)]
      if not self.pending_recurrent_inplace: stores.append(recurrent_state.uop.store(stateful_out[2].uop))
      return Tensor(out.uop.after(recurrent_state.uop.after(*stores)))
    def _run(x:Tensor, start_pos:int|UOp):
      h = x + self._attention(self.attn_norm(x), start_pos, use_flash, kv_len)
      return (h + self._normalized_feed_forward(h)).contiguous()
    return function(precompile=True, allow_implicit=True)(_run)(x, start_pos)

class TransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    self.kv_cache_host = False
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
                 valid_len:int|UOp|None=None, input_norm:nn.RMSNorm|None=None) -> Tensor:
    prepared:tuple[Tensor, ...]|None
    prepared = self.attn_q.prepare(x, any(layer.ggml_type in (12, 13) for layer in (self.attn_q, self.attn_k, self.attn_v)))
    q = self.attn_q(x, prepared)
    if prepared is not None and self.attn_k.ggml_type == self.attn_v.ggml_type == 8:
      k, v = self.attn_k(x, prepared), self.attn_v(x, prepared)
    else: k, v = self.attn_k(x, prepared), self.attn_v(x, prepared)
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
    stacked_kv = Tensor.stack(k, v)
    if self.cache_kv.dtype == dtypes.int8:
      scale = (stacked_kv.float().abs().max(axis=-1, keepdim=True) / 127).maximum(1e-8).half()
      packed_kv = (stacked_kv.float() / scale).round().clip(-127, 127).cast(dtypes.int8)
      stores = (self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(packed_kv.uop),
                self.cache_kv_scale[:, :, :, start_pos:start_pos+T].uop.store(scale.squeeze(-1).uop))
      assigned_kv, assigned_scale = Tensor(self.cache_kv.uop.after(*stores)), Tensor(self.cache_kv_scale.uop.after(*stores))
    else:
      assigned_kv = Tensor(self.cache_kv.uop.after(
        self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(stacked_kv.cast(self.cache_kv.dtype).uop)))
      assigned_scale = None
    cache_len = start_pos + T if kv_len is None else kv_len
    k, v = assigned_kv[0, :, :, 0:cache_len, :], assigned_kv[1, :, :, 0:cache_len, :]
    if assigned_scale is not None:
      k, v = k.float() * assigned_scale[0, :, :, 0:cache_len, None], v.float() * assigned_scale[1, :, :, 0:cache_len, None]

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_causal = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    flash_decode = resolve(T == 1) and kv_len is not None and str(x.device).startswith("AMD") and self.config.head_dim == 256
    if flash_decode:
      decode_len = kv_len if isinstance(kv_len, int) else self.config.max_context
      decode_pos = (start_pos.unbind()[0] if isinstance(start_pos, UOp) else start_pos) + 1
      attn = llm_amd.amd_flash_attention_decode(q.half(), assigned_kv, decode_pos, decode_len, assigned_scale)
    elif use_flash:
      start = start_pos.unbind()[0] if isinstance(start_pos, UOp) else start_pos
      valid = valid_len.unbind()[0] if isinstance(valid_len, UOp) else valid_len
      valid_kv_len, key_limit = start + T, start + valid if valid is not None else None
      attn = llm_amd.flash_attention_causal_cached(q.half(), assigned_kv, valid_kv_len, key_limit, assigned_scale)
    else:
      mask:Tensor|None
      if kv_len is not None:
        mask = None if resolve(T == 1) and self.config.ssm is not None else \
          Tensor.full((1, 1, 1, kv_len), float("-inf"), dtype=x.dtype, device=x.device, buffer=False).triu(start_pos+1)
      else:
        mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device, buffer=False).triu(start_pos+1) \
          if resolve(T != 1) else None
      attn = q.float().scaled_dot_product_attention(k.float(), v.float(), attn_mask=mask, enable_gqa=True)  # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # TODO: how is the dtype of this determined?
      # Decode uses fixed-size KV buckets. Unwritten entries must be zero: masking happens after QK, so values left
      # uninitialized by Tensor.empty can inject NaNs before the mask is applied.
      cache_dtype = dtypes.int8 if self.config.max_context > 8192 and str(x.device).startswith("AMD") else dtypes.float16
      cache_shape = (2, x.shape[0], self.config.n_kv_heads, self.config.max_context+192, self.config.head_dim)
      if self.kv_cache_host and str(x.device).startswith("AMD"):
        cache_size = 1
        for dim in cache_shape:
          assert isinstance(dim, int)
          cache_size *= dim
        storage = Buffer(str(x.device), cache_size, cache_dtype, options=BufferSpec(host=True))
        self.cache_kv = Tensor(UOp.from_buffer(storage).reshape(cache_shape))
        self.cache_kv.assign(self.cache_kv.const_like(0)).realize()
      else: self.cache_kv = Tensor.zeros(*cache_shape, dtype=cache_dtype, device=x.device).contiguous()
      if cache_dtype == dtypes.int8:
        self.cache_kv_scale = Tensor.zeros(2, x.shape[0], self.config.n_kv_heads, self.config.max_context+192,
                                           dtype=dtypes.float16, device=x.device).contiguous()
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context+192, self.config.rope_theta, device=x.device)

class MLATransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    if config.q_lora_rank > 0:
      self.attn_q_a = nn.Linear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = nn.Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.attn_kv_a_mqa = nn.Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = nn.Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp, use_flash:bool=False, kv_len:int|UOp|None=None,
                 valid_len:int|UOp|None=None, input_norm:nn.RMSNorm|None=None) -> Tensor:
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_proj = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x))) if self.config.q_lora_rank > 0 else self.attn_q(x)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    if not self.config.ssm or not self.config.ssm.kda: q_rope = apply_rope(q_rope, self.freqs_cis[start_pos:start_pos+T])
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(q_rope, dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2)
    if not self.config.ssm or not self.config.ssm.kda: k_rope = apply_rope(k_rope, self.freqs_cis[start_pos:start_pos+T])

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
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    self.attn_qkv = Linear(config.dim, self.conv_channels, bias=False)
    if ssm.kda:
      self.ssm_g_a, self.ssm_g_b = Linear(config.dim, self.head_v_dim, bias=False), Linear(self.head_v_dim, ssm.inner_size, bias=False)
      self.ssm_f_a, self.ssm_f_b = Linear(config.dim, self.head_k_dim, bias=False), Linear(self.head_k_dim, ssm.inner_size, bias=False)
    else:
      self.attn_gate = Linear(config.dim, ssm.inner_size, bias=False)
      self.ssm_alpha = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_beta = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_beta_alpha_weight:Tensor|None = None
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(ssm.inner_size if ssm.kda else self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads, 1) if ssm.kda else Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), Linear(ssm.inner_size, config.dim, bias=False)

  def _finish_state(self, conv_state:Tensor, recurrent_state:Tensor) -> Tensor:
    self.pending_state = (conv_state.contiguous(), recurrent_state.contiguous())
    if str(recurrent_state.device).startswith("AMD"): return recurrent_state
    stores = (self.conv_state.uop.store(self.pending_state[0].uop), self.recurrent_state.uop.store(self.pending_state[1].uop))
    self.pending_recurrent_inplace = True
    return Tensor(self.recurrent_state.uop.after(*stores))

  def _attention(self, x:Tensor, start_pos:int|UOp, use_flash:bool=False, kv_len:int|UOp|None=None,
                 valid_len:int|UOp|None=None, input_norm:nn.RMSNorm|None=None) -> Tensor:
    B, T, _ = x.shape
    conv_state, initial_state = self.conv_state, self.recurrent_state
    if hasattr(self, "ssm_g_a"):
      assert T == 1
      x = x.half()
      out_gate, qkv = self.ssm_g_b(self.ssm_g_a(x)), self.attn_qkv(x)
      beta = self.ssm_beta(x).sigmoid().reshape(B, self.num_v_heads, 1, 1)
      alpha = self.ssm_f_b(self.ssm_f_a(x))
      alpha = ((alpha.float() + self.ssm_dt["bias"]).softplus().reshape(B, self.num_v_heads, -1) *
               self.ssm_a.reshape(1, self.num_v_heads, -1)).exp().unsqueeze(-2)
      conv_window = conv_state.cat(qkv, dim=1)
      conv_out = (conv_window * self.ssm_conv1d["weight"].T.unsqueeze(0)).sum(1).silu()
      q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
      q = l2norm(q.reshape(B, self.num_k_heads, self.head_k_dim)).repeat(1, self.num_v_heads//self.num_k_heads, 1)
      k = l2norm(k.reshape(B, self.num_k_heads, self.head_k_dim)).repeat(1, self.num_v_heads//self.num_k_heads, 1)
      v = v.reshape(B, self.num_v_heads, self.head_v_dim)
      q, k, v = q.mul(self.head_k_dim**-0.5).unsqueeze(-1), k.unsqueeze(-1), v.unsqueeze(-1)
      recurrent_state = initial_state * alpha
      recurrent_state = recurrent_state + ((v - recurrent_state @ k) * beta) @ k.transpose(-1, -2)
      recurrent_state = self._finish_state(conv_window[:, 1:, :].cast(self.conv_state.dtype),
                                           recurrent_state.cast(self.recurrent_state.dtype))
      core = self.ssm_norm((recurrent_state @ q).squeeze(-1).reshape(B, 1, self.num_v_heads, self.head_v_dim))
      gate = out_gate.reshape(B, 1, self.num_v_heads, self.head_v_dim).sigmoid()
      return self.ssm_out((core * gate).reshape(B, 1, -1).cast(x.dtype))
    if T == 1:
      if input_norm is None: x = x.half()
      prepared = self.attn_gate.prepare(x, self.attn_qkv.ggml_type in (12, 13))
      out_gate, qkv = self.attn_gate(x, prepared), self.attn_qkv(x, prepared)
      if self.ssm_beta_alpha_weight is not None:
        beta_alpha = x @ self.ssm_beta_alpha_weight.T
        beta, alpha = beta_alpha.reshape(B, 1, -1).split(self.num_v_heads, dim=-1)
      else: beta, alpha = self.ssm_beta(x, prepared), self.ssm_alpha(x, prepared)
      out_gate = out_gate.reshape(B, 1, self.num_v_heads, self.head_v_dim)
      beta, alpha = beta.sigmoid(), ((alpha.float() + self.ssm_dt["bias"]).softplus() * self.ssm_a).exp()
      conv_window = conv_state.cat(qkv, dim=1)
      conv_out = (conv_window * self.ssm_conv1d["weight"].T.unsqueeze(0)).sum(1).silu()
      q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
      q = q.reshape(B, self.num_k_heads, self.head_k_dim)
      k = k.reshape(B, self.num_k_heads, self.head_k_dim)
      q, k = (l2norm(q), l2norm(k)) if str(x.device).startswith("AMD") else (q.normalize(dim=-1), k.normalize(dim=-1))
      q, k = q.repeat(1, self.num_v_heads//self.num_k_heads, 1), k.repeat(1, self.num_v_heads//self.num_k_heads, 1)
      v, q = v.reshape(B, self.num_v_heads, self.head_v_dim), q * self.head_k_dim**-0.5
      qv, kv = q.unsqueeze(-1), k.unsqueeze(-1)
      alpha4, beta4 = alpha.reshape(B, self.num_v_heads, 1, 1), beta.reshape(B, self.num_v_heads, 1, 1)
      if str(x.device).startswith("AMD"):
        state_k, state_q = (initial_state @ kv.cat(qv, dim=-1)).split(1, dim=-1)
        delta = (v.unsqueeze(-1) - state_k * alpha4) * beta4
        recurrent_state = initial_state * alpha4 + delta @ kv.transpose(-1, -2)
        core = (state_q * alpha4 + delta * (kv.transpose(-1, -2) @ qv)).squeeze(-1)
      else:
        recurrent_state = initial_state * alpha4
        recurrent_state = recurrent_state + ((v.unsqueeze(-1) - recurrent_state @ kv) * beta4) @ kv.transpose(-1, -2)
        core = (recurrent_state @ qv).squeeze(-1)
      recurrent_state = self._finish_state(conv_window[:, 1:, :].cast(self.conv_state.dtype),
                                           recurrent_state.cast(self.recurrent_state.dtype))
      if not str(x.device).startswith("AMD"): core = (recurrent_state @ qv).squeeze(-1)
      core = self.ssm_norm(core.reshape(B, 1, self.num_v_heads, self.head_v_dim))
      return self.ssm_out((core * out_gate.silu()).reshape(B, 1, -1).cast(x.dtype))

    assert str(x.device).startswith("AMD"), "batched GatedDeltaNet prefill currently requires AMD"
    x = x.half()
    prepared = self.attn_gate.prepare(x, self.attn_qkv.ggml_type in (12, 13))
    out_gate, qkv = self.attn_gate(x, prepared), self.attn_qkv(x, prepared)
    if self.ssm_beta_alpha_weight is not None: beta, alpha = (x @ self.ssm_beta_alpha_weight.T).split(self.num_v_heads, dim=-1)
    else: beta, alpha = self.ssm_beta(x, prepared), self.ssm_alpha(x, prepared)
    out_gate = out_gate.reshape(B, T, self.num_v_heads, self.head_v_dim)
    beta = beta.sigmoid().reshape(B, T, self.num_v_heads)
    log_alpha = ((alpha.float() + self.ssm_dt["bias"]).softplus() * self.ssm_a).reshape(B, T, self.num_v_heads)
    if valid_len is not None:
      active = (Tensor.arange(T).to(x.device) < Tensor(valid_len, device=x.device)).reshape(1, T, 1)
      beta, log_alpha = beta * active, log_alpha * active
    conv_window = conv_state.cat(qkv, dim=1)
    conv_out = functools.reduce(lambda a,b: a+b,
      (conv_window[:, i:i+T] * self.ssm_conv1d["weight"][:, i] for i in range(self.ssm_conv_kernel))).silu()
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    q = l2norm(q.reshape(B, T, self.num_k_heads, self.head_k_dim)).repeat(1, 1, self.num_v_heads//self.num_k_heads, 1)
    k = l2norm(k.reshape(B, T, self.num_k_heads, self.head_k_dim)).repeat(1, 1, self.num_v_heads//self.num_k_heads, 1)
    v = v.reshape(B, T, self.num_v_heads, self.head_v_dim)
    q, k, v, beta, log_alpha = [z.transpose(1, 2).float() for z in (q, k, v, beta, log_alpha)]
    core, recurrent_state = llm_amd.gated_delta_prefill(q * self.head_k_dim**-0.5, k, v, beta, log_alpha.exp(), initial_state)
    core = self.ssm_norm(core.transpose(1, 2))
    out = self.ssm_out((core * out_gate.silu()).reshape(B, T, -1).cast(x.dtype)).contiguous()
    state_pos = T if valid_len is None else valid_len
    self.pending_state = (conv_window[:, state_pos:state_pos+self.ssm_conv_kernel-1, :].cast(self.conv_state.dtype).contiguous(),
                          recurrent_state)
    return out

  def _state_reset_ops(self):
    return [self.conv_state.assign(self.conv_state.const_like(0)),
            self.recurrent_state.assign(self.recurrent_state.const_like(0))] if hasattr(self, "conv_state") else []
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return 0 if prefix_len != cached_len else prefix_len

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      state_dtype = dtypes.float16 if str(x.device).startswith("AMD") else x.dtype
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_v_dim,
                                          dtype=state_dtype, device=x.device).clone()

class Transformer:
  def __init__(self, config:TransformerConfig):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
    self.blk:list[FFNBlock] = [GatedDeltaNetBlock(config, config.ssm) if config.ssm and config.ssm_layers[i] else
                               block_cls(dense_config if i < config.leading_dense_blocks else config) for i in range(config.num_blocks)]
    if config.max_context > 8192:
      # A full Q8 cache for 262k Qwen leaves no graph workspace on a 24 GB card. Keep two fixed layers host-mapped;
      # this avoids runtime cache growth while leaving the other attention layers resident in VRAM.
      for block in [block for block in self.blk if isinstance(block, TransformerBlock)][-2:]: block.kv_cache_host = True
    self.token_embd  = Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    self._state_checkpoints: list[Tensor] = []
    self._state_checkpoint_pos = 0
    self._save_state_jit:Any = None
    self._restore_state_jit:Any = None
    self._warming_up = False
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.flash_prefill_jit = TinyJit(functools.partial(self.forward, use_flash=True))
    self.sample_prefill_jit = TinyJit(functools.partial(self.forward, sample=True))
    self.recurrent_prefill_jits:dict[tuple[int, bool, bool], Any] = {}
    self.rollout_jits:dict[int, Any] = {}
    self.sample_rollout_jits:dict[int, Any] = {}

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, use_flash:bool=False, kv_len:int|UOp|None=None,
              valid_len:int|UOp|None=None, sample:bool=False) -> Tensor:
    x = self.token_embd(tokens).float()                   # (B, T, D)
    for block in self.blk: x = block(x, start_pos, use_flash, kv_len, valid_len)
    last = x[:, tokens.shape[1]-1:tokens.shape[1]] if valid_len is None else x[:, valid_len-1:valid_len]
    normalized = self.output_norm(last)
    logits = self.output(normalized)[:, -1, :]
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    if not sample: return logits.argmax(-1, keepdim=True)
    return (logits / temperature - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def forward_recurrent_decode(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, decode_len:int,
                               valid_len:int|UOp|None=None, sample:bool=False) -> Tensor:
    return tokens.assign(self.forward(tokens, start_pos, temperature, kv_len=decode_len, valid_len=valid_len, sample=sample))

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, use_flash:bool=False,
               valid_len:int|UOp|None=None, sample:bool|None=None) -> Tensor:
    if sample is None: sample = getattr(self, "_sample", False)
    jit_kwargs = {"valid_len":valid_len}
    if resolve(tokens.shape[1] == 1):
      if self.has_recurrent_block:
        key = self.max_context
      else:
        pos = start_pos.unbind()[1] if isinstance(start_pos, UOp) else start_pos
        min_bucket = max(1, getenv("DECODE_BUCKET", 256))
        kv_len = key = min(self.max_context, max(min_bucket, 1 << pos.bit_length()))
      rollout_jits = self.sample_rollout_jits if sample else self.rollout_jits
      if key not in rollout_jits:
        rollout_jits[key] = TinyJit(functools.partial(self.forward_recurrent_decode, decode_len=key, sample=sample) if self.has_recurrent_block else
                                    functools.partial(self.forward, kv_len=kv_len, sample=sample))
      jit = rollout_jits[key]
    elif self.has_recurrent_block:
      prefill_key = (int(tokens.shape[1]), use_flash, sample)
      if prefill_key not in self.recurrent_prefill_jits:
        self.recurrent_prefill_jits[prefill_key] = TinyJit(functools.partial(self.forward, use_flash=use_flash, sample=sample))
      jit = self.recurrent_prefill_jits[prefill_key]
    else:
      jit = self.sample_prefill_jit if sample else self.flash_prefill_jit if use_flash else self.prefill_jit
    ret = jit(tokens.contiguous(), start_pos, temperature, **jit_kwargs)
    return ret[0] if isinstance(ret, tuple) else ret

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0))) -> tuple[Transformer, dict]:
    kv, state_dict = gguf_load(gguf)

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    ssm = None
    ssm_layers:tuple[bool, ...] = ()
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
      ssm_layers = tuple((i+1) % kv[f'{arch}.full_attention_interval'] != 0 for i in range(kv[f'{arch}.block_count']))
    elif arch == 'kimi-linear':
      ssm_layers = tuple(x == 0 for x in n_kv_heads)
      n_kv_heads = max(n_kv_heads)
      ssm = SSMConfig(kv[f'{arch}.ssm.conv_kernel'], kv[f'{arch}.kda.head_dim'], n_heads, n_heads,
                      n_heads*kv[f'{arch}.kda.head_dim'], kda=True)
      for i, is_ssm in enumerate(ssm_layers):
        if not is_ssm: continue
        state_dict[f"blk.{i}.attn_qkv.weight"] = state_dict.pop(f"blk.{i}.attn_q.weight").cat(
          state_dict.pop(f"blk.{i}.attn_k.weight"), state_dict.pop(f"blk.{i}.attn_v.weight"), dim=0).contiguous()
        state_dict[f"blk.{i}.ssm_conv1d.weight"] = state_dict.pop(f"blk.{i}.ssm_conv1d_q.weight").cat(
          state_dict.pop(f"blk.{i}.ssm_conv1d_k.weight"), state_dict.pop(f"blk.{i}.ssm_conv1d_v.weight"), dim=0).squeeze(1).contiguous()
        state_dict[f"blk.{i}.ssm_out.weight"] = state_dict.pop(f"blk.{i}.attn_output.weight")
    if arch in ('qwen35', 'qwen35moe', 'glm4moe'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
      if arch == 'kimi-linear': continue
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
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe', 'kimi-linear')),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe'), ssm=ssm,
      ssm_layers=ssm_layers,
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict)
    model = Transformer(config)
    load_device = next(iter(state_dict.values())).device
    for param in nn.state.get_parameters(model): param.replace(param.to(load_device))
    packed_weights:set[str] = set()
    packed_layers:list[PackedWeight] = []
    def resolve_owner(path:list[str]):
      obj = model
      for part in path: obj = obj[int(part)] if isinstance(obj, list) else getattr(obj, part)
      return obj
    for name, weight in state_dict.items():
      parts = name.split('.')
      quantization = get_ggml_quantization(weight)
      owner = resolve_owner(parts[:-1]) if parts[-1] == "weight" else None
      packed = quantization is not None and str(load_device).startswith("AMD") and \
        (isinstance(owner, Linear) and quantization[1] in (13, 14, 23) or
        isinstance(owner, Embedding) and quantization[1] == 12 and str(load_device).startswith("AMD"))
      if packed:
        assert quantization is not None and isinstance(owner, PackedWeight)
        owner.set_quantized(*quantization)
        packed_layers.append(owner)
        state_dict[name], packed_weights = owner.weight, packed_weights | {name}

    state_dict = {k:v if k in packed_weights else v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    recurrent_weights = []
    for block in model.blk:
      if isinstance(block, GatedDeltaNetBlock) and hasattr(block, "ssm_alpha"):
        if block.ssm_beta.ggml_type is None and block.ssm_alpha.ggml_type is None:
          block.ssm_beta_alpha_weight = block.ssm_beta.weight.cat(block.ssm_alpha.weight).contiguous()
          recurrent_weights.append(block.ssm_beta_alpha_weight)
    if recurrent_weights: Tensor.realize(*recurrent_weights)
    # Custom kernels need the shared GGUF buffer and byte offset before function tracing disables device access.
    packed_offsets = [layer._packed_offset() for layer in packed_layers]
    if packed_offsets: Tensor.realize(*packed_offsets)
    for layer,offset in zip(packed_layers, packed_offsets): layer._raw_offset_uop = offset.uop
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    return model, kv

  def get_start_pos(self, tokens:list[int]) -> int:
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    # Recurrent state has no token dimension to slice. Roll back to its latest aligned checkpoint so resumed flash
    # prefill uses the same global chunk boundaries as a full prompt.
    if self.has_recurrent_block:
      return self._state_checkpoint_pos if prefix_len >= self._state_checkpoint_pos else 0
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def _recurrent_states(self) -> list[Tensor]:
    return [getattr(block, name) for block in self.blk for name in ("conv_state", "recurrent_state") if hasattr(block, name)]

  def _init_state_checkpoints(self):
    if not self._state_checkpoints:
      states = self._recurrent_states()
      if not states: return
      self._state_checkpoints = [Tensor.zeros_like(state).contiguous().realize() for state in states]
      def copy_jit(pairs:list[tuple[Tensor, Tensor]]) -> Any:
        def copy_states() -> Tensor:
          copies = [dest.assign(src) for dest,src in pairs]
          Tensor.realize(*copies)
          return copies[-1]
        jit = TinyJit(copy_states)
        jit()
        jit()
        return jit
      self._save_state_jit = copy_jit(list(zip(self._state_checkpoints, states)))
      self._restore_state_jit = copy_jit(list(zip(states, self._state_checkpoints)))

  def _save_state_checkpoint(self, pos:int):
    self._init_state_checkpoints()
    if self._save_state_jit is None: return
    self._save_state_jit()
    self._state_checkpoint_pos = pos

  def _restore_state_checkpoint(self):
    assert self._restore_state_jit is not None
    self._restore_state_jit()

  @Context(PARALLEL_COMPILE=getenv("PARALLEL_COMPILE", 12))
  def warmup(self, chunk_size:int=256):
    device = self.token_embd.weight.device
    direct_capture = not self.has_recurrent_block and all(isinstance(block, TransformerBlock) for block in self.blk)
    if direct_capture:
      device = str(self.token_embd.weight.device)
      direct_capture = device.startswith("AMD") and Device[device].renderer.target.arch.startswith("gfx11")

    # Recurrent prefill has one fixed padded shape with symbolic valid length.
    recurrent_chunk = min(chunk_size, 256)
    warm_len = min(recurrent_chunk if self.has_recurrent_block else chunk_size * 2, self.max_context - 1)
    if warm_len > 0:
      if direct_capture:
        x = Tensor.zeros(1, 1, self.blk[0].config.dim, device=device)
        for block in self.blk: block._init_state(x)
        Tensor.realize(*[state for block in self.blk for state in (getattr(block, "cache_kv"), getattr(block, "freqs_cis"))])
        self.flash_prefill_jit.cnt = 1
        next(self.generate([0] * warm_len, chunk_size=chunk_size))
      elif self.has_recurrent_block:
        # State creation must happen outside JIT capture: capturing _init_state makes the first real request
        # reuse initialization buffers instead of the persistent recurrent/KV state.
        x = Tensor.zeros(1, 1, self.blk[0].config.dim, device=device)
        for block in self.blk: block._init_state(x)
        states = [getattr(block, name) for block in self.blk
                  for name in ("cache_kv", "cache_kv_scale", "freqs_cis", "conv_state", "recurrent_state")
                  if hasattr(block, name)]
        Tensor.realize(*states)
        self._init_state_checkpoints()
        self.prefill_jit.cnt = self.flash_prefill_jit.cnt = 1
        self.recurrent_prefill_jits[(warm_len, False, False)] = self.prefill_jit
        self.recurrent_prefill_jits[(warm_len, True, False)] = self.flash_prefill_jit
        decode_len = self.max_context
        self.rollout_jits[decode_len] = TinyJit(
          functools.partial(self.forward_recurrent_decode, decode_len=decode_len, sample=False))
        self.rollout_jits[decode_len].cnt = 1
        self._warming_up = True
        warm = self.generate([0] * warm_len, chunk_size=chunk_size)
        prefill_batch = getenv("PREFILL_JIT_BATCH_SIZE", 128)
        with Context(JIT_BATCH_SIZE=prefill_batch): next(warm)
        with Context(JIT_BATCH_SIZE=0): next(warm)
        self._warming_up = False
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
      token, temperature = Tensor([[0]], dtype="int32", device=device), Tensor([0.0], device=device)
      for bucket, pos in sorted(bucket_positions.items()):
        if direct_capture:
          self.rollout_jits[bucket] = TinyJit(functools.partial(self.forward, kv_len=bucket))
          self.rollout_jits[bucket].cnt = 1
        for _ in range(1 if direct_capture else 2):
          result = self(token, v_start_pos.bind(pos), temperature)
          assert isinstance(result, Tensor)
          result.realize()

    if self._state_checkpoints:
      # Recurrent warmup starts from the zero checkpoint. Restore it directly instead of scheduling state clears and
      # then copying the same zeros back into the checkpoint.
      self._restore_state_checkpoint()
      self._state_checkpoint_pos = 0
    elif resets := [r for block in self.blk for r in block._state_reset_ops()]: Tensor.realize(*resets)
    self._cached_tokens = []

  def generate(self, tokens:list[int], chunk_size:int|None=None, temperature:float=0.0):
    start_pos = self.get_start_pos(tokens)
    self._sample = temperature > 0
    chunk_size = min(chunk_size or 256, 256) if self.has_recurrent_block else chunk_size or 32
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    device = self.token_embd.weight.device
    temp = Tensor([temperature], device=device)
    # Dense attention needs a symbolic slice into one input buffer. Recurrent prefill instead creates fixed-size
    # chunk tensors below; allocating its input at max_context makes short requests spend most of their time converting zeros.
    t = None if self.has_recurrent_block else \
      Tensor(tokens + [0] * (self.max_context + chunk_size - len(tokens)), dtype="int32", device=device).reshape(1, self.max_context + chunk_size)
    # start_pos describes what's currently valid in the caches
    if start_pos < len(self._cached_tokens):
      if self.has_recurrent_block and self._state_checkpoints and start_pos == self._state_checkpoint_pos:
        self._restore_state_checkpoint()
      elif resets := [r for b in self.blk for r in b._state_reset_ops()]:
        Tensor.realize(*resets)
        if self._state_checkpoints: self._save_state_checkpoint(0)
    out, prompt_len = None, len(tokens)
    while len(tokens) < self.max_context:
      remaining = len(tokens) - start_pos
      recurrent_prefill = self.has_recurrent_block and start_pos < prompt_len
      can_flash = bool(getenv("AMD_FLASH_ATTENTION", 1)) and chunk_size % 64 == 0 and \
                  (recurrent_prefill if self.has_recurrent_block else start_pos > 0)
      if can_flash:
        device = str(self.token_embd.weight.device)
        can_flash = device.startswith("AMD") and Device[device].renderer.target.arch.startswith("gfx11")
      use_flash = can_flash and (self.has_recurrent_block or start_pos % 64 == 0)
      sp = v_start_pos.bind(start_pos)
      # Dense attention aligns cache reuse to a 64-token flash tile. Recurrent prefill always uses its fixed,
      # padded chunk shape, and key_limit excludes the padding from attention.
      actual_nt = min(chunk_size, remaining)
      nt = chunk_size if use_flash or self.has_recurrent_block and start_pos < prompt_len else 1 if self.has_recurrent_block else \
           v_toks.bind(min(64 - start_pos % 64, remaining) if can_flash else actual_nt)
      if self.has_recurrent_block and (start_pos < prompt_len or out is None):
        assert isinstance(nt, int)
        inp = Tensor(tokens[start_pos:start_pos+actual_nt] + [0] * (nt-actual_nt), dtype="int32", device=device).reshape(1, nt)
      elif start_pos < prompt_len or out is None:
        assert t is not None
        inp = t[:, sp:sp+nt]
      else: inp = out
      valid_len = v_toks.bind(actual_nt) if recurrent_prefill or use_flash and actual_nt < chunk_size else None
      # Save once immediately before a short final chunk. This is the nearest globally aligned state that can be
      # reused without changing flash-attention's numerical tile layout.
      if not self._warming_up and recurrent_prefill and remaining < chunk_size and start_pos % chunk_size == 0:
        self._save_state_checkpoint(start_pos)
      if use_flash: result = self(inp, sp, temp, use_flash=True, valid_len=valid_len)
      elif valid_len is not None: result = self(inp, sp, temp, valid_len=valid_len)
      else: result = self(inp, sp, temp)
      out = result.realize()
      start_pos += actual_nt
      # Generated tool calls are reconstructed by clients and are not guaranteed token-identical on the next request.
      # Keep the reusable checkpoint at the stable prompt boundary instead of overwriting it inside generated output.
      if not self._warming_up and self.has_recurrent_block and start_pos == prompt_len and start_pos % chunk_size == 0:
        self._save_state_checkpoint(start_pos)
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]
