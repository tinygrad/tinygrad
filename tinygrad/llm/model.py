from __future__ import annotations
import functools, itertools, pathlib
from dataclasses import dataclass, replace
from typing import cast
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes, Context
from tinygrad.llm.kernels import amd as llm_amd
from tinygrad.llm.gguf import gguf_load
from tinygrad.helpers import prod
from tinygrad.uop.ops import resolve, Ops

def unwrap_var(x:int|UOp|None): return x.unbind()[0] if isinstance(x, UOp) else x

class Linear(nn.Linear):
  ggml_type:int|None = None
  def set_quantized(self, decoded:Tensor) -> Tensor|None:
    packed_sizes = {decoded.numel() // 256 * type_size:typ for typ,type_size in ((13, 176), (14, 210), (23, 136))}
    raw = next((u for u in decoded.uop.toposort() if u.op is Ops.SHRINK and u.dtype == dtypes.uint8 and prod(u.shape) in packed_sizes), None)
    if raw is None: return None
    self.weight, self.ggml_type = Tensor(raw).flatten(), packed_sizes[prod(raw.shape)]
    raw_offset = self.weight.uop.contiguous_view_offset()
    assert raw_offset is not None and raw_offset % 4 == 0 and self.weight.uop.buf_uop.dtype == dtypes.uint8
    if self.ggml_type == 23 and str(self.weight.device).startswith("AMD"): llm_amd.iq4_half_lut(str(self.weight.device))
    return Tensor([raw_offset // 4], dtype=dtypes.uint64, device=self.weight.device)
  def __init__(self, in_features:int, out_features:int, bias=True):
    super().__init__(in_features, out_features, bias)
    self.in_features, self.out_features = in_features, out_features
    self._raw_offset_uop:UOp|None = None
  def __call__(self, x:Tensor) -> Tensor:
    return llm_amd.q8_linear(self, x) if self.ggml_type in (13, 14, 23) and str(self.weight.device).startswith("AMD") else \
      super().__call__(x)

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).clone(device)

class ExpertWeights:
  """Like Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
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
      logits = self.ffn_gate_inp(x)
      if hasattr(self, 'exp_probs_b'):
        probs = logits.sigmoid()
        _, sel = pairwise_topk(probs + self.exp_probs_b["bias"], self.config.num_experts_per_tok)
        probs = probs.gather(-1, sel)
        if self.config.norm_topk_prob: probs = probs / probs.sum(axis=-1, keepdim=True)
      else:
        vals, sel = pairwise_topk(logits, self.config.num_experts_per_tok)
        probs = vals.softmax(-1) if self.config.norm_topk_prob else logits.softmax(-1).gather(-1, sel)
      probs = probs * self.config.routed_scaling_factor
      x_down = self.ffn_down_exps(sel, (self.ffn_gate_exps(sel, h).silu() * self.ffn_up_exps(sel, h)).contiguous())  # (B, T, k, D)
      out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      if hasattr(self, 'ffn_gate_shexp'):
        shexp = self.ffn_down_shexp(self.ffn_gate_shexp(x).silu().contiguous() * self.ffn_up_shexp(x))
        if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out
    return self.ffn_down(self.ffn_gate(x).silu() * self.ffn_up(x))

  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp, valid_len:int|UOp|None=None) -> Tensor:
    raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp, valid_len:int|UOp|None=None):
    self._init_state(x)
    if hasattr(self, 'attn_gate'):
      @function(precompile=True, allow_implicit=True)
      def _run_stateful(x:Tensor, start_pos:int|UOp, valid_len:int|UOp|None):
        attn, conv_state = cast(tuple[Tensor, Tensor], self._attention(self.attn_norm(x), start_pos, valid_len))
        h = x + attn
        return (h + self._feed_forward(self.ffn_norm(h))).contiguous(), conv_state
      out, conv_state = _run_stateful(x, start_pos, valid_len)
      state = getattr(self, "conv_state")
      return Tensor(out.uop.after(state.uop.after(state.uop.store(conv_state.uop))))
    def _run(x:Tensor, start_pos:int|UOp):
      h = x + self._attention(self.attn_norm(x), start_pos)
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

  def _attention(self, x:Tensor, start_pos:int|UOp, valid_len:int|UOp|None=None):
    q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
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
      if resolve(T == 1):
        attn = llm_amd.amd_flash_attention_decode(q.half(), assigned_kv, cast(int|UOp, unwrap_var(start_pos))+1,
                                                  assigned_scale, self.config.max_context)
      else:
        start, valid = cast(int|UOp, unwrap_var(start_pos)), unwrap_var(valid_len)
        attn = llm_amd.flash_attention_causal_cached(q.half(), assigned_kv, start+T, assigned_scale,
                                                     start+valid if valid is not None else None)
    else:
      assigned_kv = Tensor(self.cache_kv.uop.after(
        self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(stacked_kv.cast(self.cache_kv.dtype).uop)))
      k, v = assigned_kv[0, :, :, 0:start_pos+T, :], assigned_kv[1, :, :, 0:start_pos+T, :]
      # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
      mask = Tensor.full((1, 1, T, k.shape[-2]), float("-inf"), dtype=x.dtype, device=x.device, buffer=False).triu(start_pos+1) \
        if resolve(T != 1) else None
      attn = q.float().scaled_dot_product_attention(k.float(), v.float(), attn_mask=mask, enable_gqa=True)  # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # Zero padding prevents masked, unwritten cache entries from injecting NaNs before the mask.
      cache_dtype = dtypes.int8 if self.config.ssm is not None and self.config.max_context > 8192 and \
        str(x.device).startswith("AMD") else dtypes.float16
      cache_shape = (2, x.shape[0], self.config.n_kv_heads, self.config.max_context+192, self.config.head_dim)
      self.cache_kv = Tensor.zeros(*cache_shape, dtype=cache_dtype, device=x.device).contiguous()
      if cache_dtype == dtypes.int8: self.cache_kv_scale = Tensor.zeros(*cache_shape[:-1], dtype=dtypes.float16, device=x.device).contiguous()
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

  def _attention(self, x:Tensor, start_pos:int|UOp, valid_len:int|UOp|None=None) -> Tensor:
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
      self.ssm_beta = Linear(config.dim, self.num_v_heads, bias=False)
    else:
      self.attn_gate = Linear(config.dim, ssm.inner_size, bias=False)
      self.ssm_beta_alpha = Linear(config.dim, 2*self.num_v_heads, bias=False)
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(ssm.inner_size if ssm.kda else self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads, 1) if ssm.kda else Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), Linear(ssm.inner_size, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp, valid_len:int|UOp|None=None):
    B, T, _ = x.shape
    is_qwen = hasattr(self, "attn_gate")
    assert is_qwen or T == 1, "Kimi GatedDeltaNetBlock currently only supports T=1"
    x = x.half()
    out_gate = self.attn_gate(x) if is_qwen else self.ssm_g_b(self.ssm_g_a(x))
    if is_qwen:
      beta, alpha = self.ssm_beta_alpha(x).split(self.num_v_heads, dim=-1)
    else: beta, alpha = self.ssm_beta(x), self.ssm_f_b(self.ssm_f_a(x))
    conv_window = self.conv_state.cat(self.attn_qkv(x), dim=1)
    conv_out = functools.reduce(lambda a,b: a+b,
      (conv_window[:, i:i+T] * self.ssm_conv1d["weight"][:, i] for i in range(self.ssm_conv_kernel))).silu()
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    q = q.reshape(B, T, self.num_k_heads, self.head_k_dim).normalize(dim=-1, eps=1e-6 if is_qwen else 1e-12).repeat(
      1, 1, self.num_v_heads//self.num_k_heads, 1)
    k = k.reshape(B, T, self.num_k_heads, self.head_k_dim).normalize(dim=-1, eps=1e-6 if is_qwen else 1e-12).repeat(
      1, 1, self.num_v_heads//self.num_k_heads, 1)
    v = v.reshape(B, T, self.num_v_heads, self.head_v_dim)
    initial_state = self.recurrent_state
    if not is_qwen:
      out_gate = out_gate.reshape(B, 1, self.num_v_heads, self.head_v_dim).sigmoid()
      beta = beta.sigmoid().reshape(B, self.num_v_heads, 1, 1)
      alpha = ((alpha.float() + self.ssm_dt["bias"]).softplus().reshape(B, self.num_v_heads, -1) *
               self.ssm_a.reshape(1, self.num_v_heads, -1)).exp().unsqueeze(-2)
      q, k, v = q[:, 0].mul(self.head_k_dim**-0.5).unsqueeze(-1), k[:, 0].unsqueeze(-1), v[:, 0].unsqueeze(-1)
      recurrent_state = initial_state*alpha + ((v-initial_state*alpha@k)*beta)@k.transpose(-1, -2)
      stores = (self.recurrent_state.uop.store(recurrent_state.cast(self.recurrent_state.dtype).uop),
                self.conv_state.uop.store(conv_window[:, 1:, :].cast(self.conv_state.dtype).uop))
      state = Tensor(self.recurrent_state.uop.after(*stores))
      core = self.ssm_norm((state@q).squeeze(-1).reshape(B, 1, self.num_v_heads, self.head_v_dim))
      return self.ssm_out((core*out_gate).reshape(B, 1, -1).cast(x.dtype))
    out_gate = out_gate.reshape(B, T, self.num_v_heads, self.head_v_dim)
    beta, log_alpha = beta.sigmoid().reshape(B, T, self.num_v_heads), \
      ((alpha.float() + self.ssm_dt["bias"]).softplus() * self.ssm_a).reshape(B, T, self.num_v_heads)
    if valid_len is not None:
      active = (Tensor.arange(T).to(x.device) < Tensor(valid_len, device=x.device)).reshape(1, T, 1)
      beta, log_alpha = beta * active, log_alpha * active
    q, k, v, beta, log_alpha = [z.transpose(1, 2).float() for z in (q, k, v, beta, log_alpha)]
    core = llm_amd.gated_delta_prefill(q * self.head_k_dim**-0.5, k, v, beta, log_alpha.exp(), initial_state)
    out = self.ssm_out((self.ssm_norm(core.transpose(1, 2)) * out_gate.silu()).reshape(B, T, -1).cast(x.dtype)).contiguous()
    state_pos = T if valid_len is None else valid_len
    conv_state = conv_window[:, state_pos:state_pos+self.ssm_conv_kernel-1].cast(self.conv_state.dtype).contiguous()
    return out, conv_state

  def _state_reset_ops(self): return [self.conv_state.assign(self.conv_state.const_like(0)),
    self.recurrent_state.assign(self.recurrent_state.const_like(0))] if hasattr(self, "conv_state") else []
  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_v_dim, device=x.device).clone()

class Transformer:
  def __init__(self, config:TransformerConfig):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
    self.blk:list[FFNBlock] = [GatedDeltaNetBlock(dense_config if i < config.leading_dense_blocks else config, config.ssm)
                               if config.ssm and config.ssm_layers[i] else
                               block_cls(dense_config if i < config.leading_dense_blocks else config) for i in range(config.num_blocks)]
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)
    self.recurrent_rollout_jit = TinyJit(self.forward_recurrent_decode)

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, valid_len:int|UOp|None=None) -> Tensor:
    x = self.token_embd(tokens).float()                   # (B, T, D)
    for block in self.blk: x = block(x, start_pos, valid_len)
    last = x[:, tokens.shape[1]-1:tokens.shape[1]] if valid_len is None else x[:, valid_len-1:valid_len]
    logits = self.output(self.output_norm(last))[:, -1, :]
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    if self.has_recurrent_block: return logits.argmax(-1, keepdim=True)
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def forward_recurrent_decode(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, valid_len:int|UOp|None=None) -> Tensor:
    return tokens.assign(self.forward(tokens, start_pos, temperature, valid_len))

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, valid_len:int|UOp|None=None) -> Tensor:
    if not self.has_recurrent_block:
      return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens.contiguous(), start_pos, temperature)
    jit = self.recurrent_rollout_jit if resolve(tokens.shape[1] == 1) else self.prefill_jit
    return jit(tokens.contiguous(), start_pos, temperature, valid_len=valid_len)

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
    ssm_layers: tuple[bool, ...] = ()
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
      ssm_layers = tuple((i+1) % kv[f'{arch}.full_attention_interval'] != 0 for i in range(kv[f'{arch}.block_count']))
      for i,is_ssm in enumerate(ssm_layers):
        if is_ssm: state_dict[f"blk.{i}.ssm_beta_alpha.weight"] = state_dict.pop(f"blk.{i}.ssm_beta.weight").cat(
          state_dict.pop(f"blk.{i}.ssm_alpha.weight"), dim=0).contiguous()
    elif arch == 'kimi-linear':
      ssm_layers = tuple(x == 0 for x in n_kv_heads)
      n_kv_heads = max(n_kv_heads)
      ssm = SSMConfig(kv[f'{arch}.ssm.conv_kernel'], kv[f'{arch}.kda.head_dim'], n_heads, n_heads, n_heads*kv[f'{arch}.kda.head_dim'], kda=True)
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
    if getenv("HALF", 1): state_dict = {name:weight.cast('float16') for name,weight in state_dict.items()}
    packed_layers:list[tuple[Linear, Tensor]] = []
    layers = cast(dict[str, Linear], nn.state.get_state_dict(model, tensor_type=Linear))
    for name, owner in layers.items():
      key, weight = f"{name}.weight", state_dict[f"{name}.weight"]
      if str(weight.device).startswith("AMD") and (offset:=owner.set_quantized(weight)) is not None:
        packed_layers.append((owner, offset))
        state_dict[key] = owner.weight
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    if packed_layers: Tensor.realize(*(offset for _,offset in packed_layers))
    for layer,offset in packed_layers: layer._raw_offset_uop = offset.uop
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    return model, kv

  def get_start_pos(self, tokens:list[int]) -> int:
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return 0 if self.has_recurrent_block and prefix_len != len(self._cached_tokens) else prefix_len

  def warmup(self, chunk_size:int=256):
    assert self.has_recurrent_block
    x = Tensor.zeros(1, 1, self.blk[0].config.dim, device=self.token_embd.weight.device)
    for block in self.blk: block._init_state(x)
    states = [getattr(block, name) for block in self.blk
              for name in ("cache_kv", "cache_kv_scale", "freqs_cis", "conv_state", "recurrent_state") if hasattr(block, name)]
    Tensor.realize(*states)
    self.prefill_jit.cnt = self.recurrent_rollout_jit.cnt = 1
    warm = self.generate([0] * min(chunk_size, 256, self.max_context-1), chunk_size=chunk_size)
    with Context(JIT_BATCH_SIZE=getenv("PREFILL_JIT_BATCH_SIZE", 512)): next(warm)
    with Context(JIT_BATCH_SIZE=0): next(warm)

    if resets := [r for block in self.blk if hasattr(block, "_state_reset_ops") for r in block._state_reset_ops()]: Tensor.realize(*resets)
    self._cached_tokens = []

  def generate(self, tokens:list[int], chunk_size:int|None=None, temperature:float=0.0):
    start_pos = self.get_start_pos(tokens)
    decode_resume = self.has_recurrent_block and bool(self._cached_tokens) and start_pos > 0
    chunk_size = min(chunk_size or 256, 256) if self.has_recurrent_block else chunk_size or 32
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    device = self.token_embd.weight.device
    temp = Tensor([temperature], device=device)
    t = Tensor(tokens + [0] * (self.max_context + chunk_size - len(tokens)), dtype="int32", device=device).reshape(1, self.max_context + chunk_size)
    if start_pos < len(self._cached_tokens) and \
       (resets := [r for b in self.blk if hasattr(b, "_state_reset_ops") for r in b._state_reset_ops()]): Tensor.realize(*resets)
    out, prompt_len = None, len(tokens)
    while len(tokens) < self.max_context:
      recurrent_prefill = self.has_recurrent_block and start_pos < prompt_len and not decode_resume
      sp = v_start_pos.bind(start_pos)
      actual_nt = min(1 if decode_resume and start_pos < prompt_len else chunk_size, len(tokens)-start_pos)
      nt = chunk_size if recurrent_prefill else 1 if self.has_recurrent_block else v_toks.bind(actual_nt)
      if decode_resume and start_pos < prompt_len and out is not None:
        inp = out.assign(Tensor([[tokens[start_pos]]], dtype="int32", device=device)).realize()
      elif start_pos < prompt_len or out is None:
        inp = t[:, sp:sp+nt]
      else: inp = out
      valid_len = v_toks.bind(actual_nt) if recurrent_prefill else None
      out = self(inp, sp, temp, valid_len=valid_len).realize()
      start_pos += actual_nt
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]
