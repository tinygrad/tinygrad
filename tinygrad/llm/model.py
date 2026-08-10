from __future__ import annotations
import array, functools, itertools, pathlib
from dataclasses import dataclass, replace
from typing import Callable, cast
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes
from tinygrad.device import MultiBuffer
from tinygrad.nn import Linear
from tinygrad.llm.gguf import gguf_load
from tinygrad.llm.quant import dequantize_mxfp4, quantize_dequantize_mxfp8
from tinygrad.llm.kernels import amd_custom_kernels_supported, amd_exact_bf16_custom_kernels_supported, amd_int32_item, \
  amd_packed_mxfp4_supported, amd_wave64_custom_kernels_supported, bf16_matvec, bf16_mfma_splitk, bf16_partial_linear, \
  dual_bf16_matvec, dual_input_bf16_matvec, \
  gated_delta_prefill, kda_fgb_linear, kda_qkv_linear, mxfp4_expert_linear, mxfp8_quantize_dequantize
from tinygrad.uop.ops import resolve

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

class MXFP4ExpertWeights:
  """Routed-expert weights stored as packed OCP MXFP4 with one E8M0 scale per 32 values."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    if in_features % 32: raise ValueError(f"MXFP4 expert input size must be divisible by 32, got {in_features}")
    self.in_features, self.out_features = in_features, out_features
    self.weight = Tensor.zeros(num_experts, out_features, in_features//2, dtype=dtypes.uint8)
    self.weight_scale = Tensor.full((num_experts, out_features, in_features//32), 127, dtype=dtypes.uint8)
  def __call__(self, sel:Tensor, x:Tensor, quantized:bool=False, partial:bool=False) -> Tensor:
    # Only selected weights are expanded, so packed storage remains resident during generation.
    if isinstance(self.weight.device, tuple) and not isinstance(sel.device, tuple): sel = sel.shard(self.weight.device, axis=None)
    if not quantized:
      x = mxfp8_quantize_dequantize(x.cast(dtypes.bfloat16)) if amd_custom_kernels_supported(x.device) else \
        quantize_dequantize_mxfp8(x.cast(dtypes.bfloat16))
    # gfx11 has no native FP4 instructions, but decoding nibbles inside the dot product still avoids
    # the much larger selected-expert BF16 temporary. Gate/up weights are output-sharded in TP.
    if amd_packed_mxfp4_supported(self.weight.device):
      return mxfp4_expert_linear(sel, x, self.weight, self.weight_scale, partial=partial)
    weight = dequantize_mxfp4(self.weight[sel], self.weight_scale[sel], dtype=dtypes.bfloat16)
    return (x.unsqueeze(-2) @ weight.transpose(-1, -2)).contiguous().squeeze(-2)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def l2norm(x:Tensor, eps:float=1e-6) -> Tensor:
  """FLA-compatible L2 normalization: FP32 reduction and epsilon inside the square root."""
  dtype, x = x.dtype, x.float()
  return (x * (x.square().sum(axis=-1, keepdim=True, dtype=dtypes.float32) + eps).rsqrt()).cast(dtype)

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
  sel = x.const_like(0).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

def iterative_topk(x:Tensor, k:int) -> tuple[Tensor, Tensor]:
  """O(k*N) top-k for very wide MoE routers, with stable first-index tie breaking."""
  work, values, indices = x, [], []
  for _ in range(k):
    sel = work.argmax(-1, keepdim=True)
    values.append(x.gather(-1, sel))
    indices.append(sel)
    work = work.scatter(-1, sel, x.dtype.min)
  return values[0].cat(*values[1:], dim=-1), indices[0].cat(*indices[1:], dim=-1)

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int
  kda: bool = False
  channel_decay: bool = False

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
  expert_mxfp4: bool = False
  bf16_activations: bool = False
  kda_split_qkv: bool = False
  # Kimi K3 extensions. Defaults preserve all existing model behavior.
  activation_situ_beta: float = 0.0
  activation_situ_linear_beta: float = 0.0
  routed_expert_dim: int = 0
  latent_moe_norm: bool = False
  route_weights_uncorrected: bool = False
  attn_res_block_size: int = 0
  kda_full_rank_gate: bool = False
  kda_gate_lower_bound: float = 0.0
  recurrent_prefill_chunked: bool = False
  recurrent_prefill_chunk_size: int = 0

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
      expert_cls = MXFP4ExpertWeights if config.expert_mxfp4 else ExpertWeights
      expert_dim = config.routed_expert_dim or config.dim
      self.ffn_gate_exps = expert_cls(config.num_experts, expert_dim, config.hidden_dim)
      self.ffn_up_exps = expert_cls(config.num_experts, expert_dim, config.hidden_dim)
      self.ffn_down_exps = expert_cls(config.num_experts, config.hidden_dim, expert_dim)
      if config.routed_expert_dim:
        self.ffn_routed_down = Linear(config.dim, expert_dim, bias=False)
        self.ffn_routed_up = Linear(expert_dim, config.dim, bias=False)
        if config.latent_moe_norm: self.ffn_routed_norm = nn.RMSNorm(expert_dim, config.norm_eps)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate    = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = Linear(config.hidden_dim, config.dim, bias=False)

    if config.attn_res_block_size:
      self.attn_res_norm, self.mlp_res_norm = nn.RMSNorm(config.dim, config.norm_eps), nn.RMSNorm(config.dim, config.norm_eps)
      self.attn_res_proj, self.mlp_res_proj = Linear(config.dim, 1, bias=False), Linear(config.dim, 1, bias=False)

  def _activation(self, gate:Tensor, up:Tensor) -> Tensor:
    if not self.config.activation_situ_beta: return gate.silu() * up
    gate32, up32, beta = gate.float(), up.float(), self.config.activation_situ_beta
    gate32 = beta * (gate32 / beta).tanh() * gate32.sigmoid()
    if (linear_beta := self.config.activation_situ_linear_beta): up32 = linear_beta * (up32 / linear_beta).tanh()
    return (gate32 * up32).cast(gate.dtype)

  def _feed_forward(self, x:Tensor) -> Tensor:
    if hasattr(self, 'ffn_gate_exps'):
      h = x.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      # Kimi computes router logits in FP32 even though the residual stream and weights are BF16.
      logits = x.float().linear(self.ffn_gate_inp.weight.float().transpose()) if self.config.bf16_activations else self.ffn_gate_inp(x)
      if hasattr(self, 'exp_probs_b'):
        scores = logits.sigmoid()
        adjusted_scores = scores + self.exp_probs_b["bias"]
        topk = iterative_topk if self.config.num_experts >= 512 else pairwise_topk
        _, sel = topk(adjusted_scores, self.config.num_experts_per_tok)
        probs = (scores if self.config.route_weights_uncorrected else adjusted_scores).gather(-1, sel)
        # Kimi-Linear-48B's older reference weights corrected scores. K3 selects with the correction
        # but gathers the uncorrected sigmoid scores, so keep this an explicit compatibility switch.
        if self.config.norm_topk_prob: probs = probs / (probs.sum(axis=-1, keepdim=True) + 1e-20)
      else:
        vals, sel = pairwise_topk(logits, self.config.num_experts_per_tok)
        probs = vals.softmax(-1) if self.config.norm_topk_prob else logits.softmax(-1).gather(-1, sel)
      probs = probs * self.config.routed_scaling_factor
      if hasattr(self, 'ffn_routed_down'): h = self.ffn_routed_down(x).unsqueeze(2)
      if isinstance(self.ffn_gate_exps, MXFP4ExpertWeights) and amd_packed_mxfp4_supported(h.device):
        hq = mxfp8_quantize_dequantize(h.cast(dtypes.bfloat16)) if amd_custom_kernels_supported(h.device) else \
          quantize_dequantize_mxfp8(h.cast(dtypes.bfloat16))
        gate = self.ffn_gate_exps(sel, hq, quantized=True)
        up = cast(MXFP4ExpertWeights, self.ffn_up_exps)(sel, hq, quantized=True)
      else: gate, up = self.ffn_gate_exps(sel, h), self.ffn_up_exps(sel, h)
      routed_activation = self._activation(gate, up).contiguous()
      combine_down = resolve(x.shape[1] == 1) and isinstance(self.ffn_down_exps, MXFP4ExpertWeights) and \
        hasattr(self, 'ffn_gate_shexp') and not hasattr(self, 'ffn_routed_up') and amd_custom_kernels_supported(x.device)
      x_down = cast(MXFP4ExpertWeights, self.ffn_down_exps)(sel, routed_activation, partial=True) if combine_down else \
        self.ffn_down_exps(sel, routed_activation)
      out = (x_down * probs.unsqueeze(-1).unsqueeze(-1)).sum(axis=2) if combine_down else \
        (x_down * probs.unsqueeze(-1)).sum(axis=2).cast(x_down.dtype)  # (B, T, D[, devices])
      combine_final = resolve(x.shape[1] == 1) and hasattr(self, 'ffn_routed_up') and hasattr(self, 'ffn_gate_shexp') and \
        not hasattr(self, 'ffn_gate_inp_shexp') and isinstance(self.ffn_routed_up.weight.device, tuple) and \
        isinstance(self.ffn_down_shexp.weight.device, tuple) and self.ffn_routed_up.weight.uop.axis == self.ffn_down_shexp.weight.uop.axis == 1 and \
        self.ffn_routed_up.weight.shape[1] % (32*len(self.ffn_routed_up.weight.device)) == 0 and \
        self.ffn_down_shexp.weight.shape[1] % (32*len(self.ffn_down_shexp.weight.device)) == 0 and \
        amd_exact_bf16_custom_kernels_supported(x.device)
      if hasattr(self, 'ffn_routed_up'):
        if hasattr(self, 'ffn_routed_norm'): out = self.ffn_routed_norm(out)
        out = bf16_partial_linear(out, self.ffn_routed_up.weight) if combine_final else self.ffn_routed_up(out)
      if hasattr(self, 'ffn_gate_shexp'):
        if resolve(x.shape[1] == 1) and amd_exact_bf16_custom_kernels_supported(x.device) and \
           self.ffn_gate_shexp.weight.shape == self.ffn_up_shexp.weight.shape:
          shared_gate, shared_up = dual_bf16_matvec(x, self.ffn_gate_shexp.weight, self.ffn_up_shexp.weight,
                                                    fast=amd_custom_kernels_supported(x.device))
        else: shared_gate, shared_up = self.ffn_gate_shexp(x).contiguous(), self.ffn_up_shexp(x).contiguous()
        shared_activation = self._activation(shared_gate, shared_up).contiguous()
        if combine_down:
          out = (out + bf16_partial_linear(shared_activation, self.ffn_down_shexp.weight)).sum(3).cast(dtypes.bfloat16)
        elif combine_final:
          out = (out + bf16_partial_linear(shared_activation, self.ffn_down_shexp.weight)).sum(3).cast(dtypes.bfloat16)
        else:
          shexp = self.ffn_down_shexp(shared_activation)
          if hasattr(self, 'ffn_gate_inp_shexp'):
            shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
          out = out + shexp
      return out
    # TODO: remove the need for this contiguous
    if resolve(x.shape[1] == 1) and amd_exact_bf16_custom_kernels_supported(x.device) and \
       self.ffn_gate.weight.shape == self.ffn_up.weight.shape:
      dense_gate, dense_up = dual_bf16_matvec(x, self.ffn_gate.weight, self.ffn_up.weight, fast=amd_custom_kernels_supported(x.device))
    else: dense_gate, dense_up = self.ffn_gate(x).contiguous(), self.ffn_up(x).contiguous()
    dense_activation = self._activation(dense_gate, dense_up).contiguous()
    if resolve(x.shape[1] == 1) and isinstance(self.ffn_down.weight.device, tuple) and self.ffn_down.weight.uop.axis == 1 and \
       self.ffn_down.weight.shape[1] % (32*len(self.ffn_down.weight.device)) == 0 and amd_exact_bf16_custom_kernels_supported(x.device):
      return bf16_partial_linear(dense_activation, self.ffn_down.weight).sum(3).cast(dtypes.bfloat16)
    return self.ffn_down(dense_activation)

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  # return writes that reset this block's state after a cache mismatch
  def _state_reset_ops(self) -> list[Tensor]: return []
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp):
    self._init_state(x)
    # Kimi's heterogeneous TP shards are captured by the outer TinyJit; per-block precompilation
    # cannot represent their differently shaped local buffers as one implicit parameter bundle.
    if self.config.bf16_activations:
      h = x + self._attention(self.attn_norm(x), start_pos)
      return (h + self._feed_forward(self.ffn_norm(h))).contiguous()
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp):
      h =     x + self._attention(self.attn_norm(x), start_pos)
      return (h + self._feed_forward(self.ffn_norm(h))).contiguous()
    return _run(x, start_pos)

  @staticmethod
  def _apply_attn_res(prefix_sum:Tensor, block_residual:Tensor, proj:Linear, norm:nn.RMSNorm) -> Tensor:
    # Both inputs are flattened over B*T. Scoring is intentionally FP32, matching K3 eager inference.
    v = block_residual.cat(prefix_sum.unsqueeze(1), dim=1)
    vf = v.float()
    k = vf * (vf.square().mean(axis=-1, keepdim=True) + norm.eps).rsqrt()
    assert norm.weight is not None
    scores = (k * (norm.weight.float() * proj.weight.squeeze(0).float())).sum(axis=-1)
    return (scores.softmax(-1).unsqueeze(1) @ vf).squeeze(1).cast(v.dtype)

  def attn_residual(self, x:Tensor, start_pos:int|UOp, block_residual:Tensor, layer_idx:int) -> tuple[Tensor, Tensor]:
    self._init_state(x)
    shape, prefix_sum = x.shape, x
    prefix:Tensor|None = prefix_sum
    if block_residual.shape[1]: x = self._apply_attn_res(x.reshape(-1, shape[-1]), block_residual,
                                                        self.attn_res_proj, self.attn_res_norm).reshape(shape)
    if layer_idx % self.config.attn_res_block_size == 0:
      block_residual = block_residual.cat(prefix_sum.reshape(-1, shape[-1]).unsqueeze(1), dim=1)
      prefix = None
    attn = self._attention(self.attn_norm(x), start_pos)
    prefix = attn if prefix is None else prefix + attn
    x = self._apply_attn_res(prefix.reshape(-1, shape[-1]), block_residual,
                             self.mlp_res_proj, self.mlp_res_norm).reshape(shape)
    mlp = self._feed_forward(self.ffn_norm(x))
    return (prefix + mlp).contiguous(), block_residual

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

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
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
    assigned_kv = Tensor(self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).uop)))
    k = assigned_kv[0, :, :, 0:start_pos+T, :]
    v = assigned_kv[1, :, :, 0:start_pos+T, :]

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    # Build the static T×T causal corner on-device, then prepend the unmasked cached prefix.
    # A broadcast const with symbolic width otherwise defaults to CPU in multi-device graphs.
    mask = Tensor.full((1, 1, T, T), float("-inf"), dtype=x.dtype, device=x.device).triu(1).pad(((0, 0),)*3+((start_pos, 0),)) \
      if resolve(T != 1) else None
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # TODO: how is the dtype of this determined?
      self.cache_kv = Tensor.empty(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim,
                                   device=x.device, dtype=x.dtype)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

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
    if config.attn_output_gate: self.attn_gate = Linear(config.dim, config.n_heads * config.v_head_dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    mfma_decode = resolve(T == 1) and x.shape[-1] % 256 == 0 and amd_wave64_custom_kernels_supported(x.device)
    q_a_mfma = self.config.q_lora_rank > 0 and mfma_decode and self.attn_q_a.weight.shape[0] % 16 == 0
    q_a = bf16_mfma_splitk(x, self.attn_q_a.weight) if q_a_mfma else \
      self.attn_q_a(x) if self.config.q_lora_rank > 0 else None
    q_proj = self.attn_q_b(self.attn_q_a_norm(q_a)) if q_a is not None else self.attn_q(x)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    if not self.config.ssm or not self.config.ssm.kda: q_rope = apply_rope(q_rope, self.freqs_cis[start_pos:start_pos+T])
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(q_rope, dim=-1)

    kv_a_mfma = mfma_decode and self.attn_kv_a_mqa.weight.shape[0] % 16 == 0
    kv_a = bf16_mfma_splitk(x, self.attn_kv_a_mqa.weight) if kv_a_mfma else self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2)
    if not self.config.ssm or not self.config.ssm.kda: k_rope = apply_rope(k_rope, self.freqs_cis[start_pos:start_pos+T])

    k_store = c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1)
    k = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.uop)))[:, :, 0:start_pos+T, :]
    v = k[..., :self.config.kv_lora_rank]

    mask = Tensor.full((1, 1, T, T), float("-inf"), dtype=x.dtype, device=x.device).triu(1).pad(((0, 0),)*3+((start_pos, 0),)) \
      if resolve(T != 1) else None
    attn = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    if mask is not None: attn = attn + mask
    # Match eager Kimi MLA: normalize attention scores in FP32, then return to the query dtype.
    attn = attn.softmax(-1, dtype=dtypes.float32).cast(q.dtype)
    attn = ((attn @ v) @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    if hasattr(self, "attn_gate"): attn = attn * self.attn_gate(x).sigmoid()
    if resolve(T == 1) and isinstance(self.attn_output.weight.device, tuple) and \
       self.attn_output.weight.shape[1] % (32*len(self.attn_output.weight.device)) == 0 and amd_exact_bf16_custom_kernels_supported(attn.device):
      return bf16_partial_linear(attn, self.attn_output.weight).sum(3).cast(dtypes.bfloat16)
    return self.attn_output(attn)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim,
                                  device=x.device, dtype=x.dtype)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    if ssm.kda and config.kda_split_qkv:
      self.attn_q, self.attn_k = Linear(config.dim, self.q_dim, bias=False), Linear(config.dim, self.q_dim, bias=False)
      self.attn_v = Linear(config.dim, ssm.inner_size, bias=False)
      self.ssm_q_conv1d = {"weight": Tensor.zeros(self.q_dim, self.ssm_conv_kernel)}
      self.ssm_k_conv1d = {"weight": Tensor.zeros(self.q_dim, self.ssm_conv_kernel)}
      self.ssm_v_conv1d = {"weight": Tensor.zeros(ssm.inner_size, self.ssm_conv_kernel)}
    else:
      self.attn_qkv = Linear(config.dim, self.conv_channels, bias=False)
      self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    if ssm.kda:
      if config.kda_full_rank_gate: self.ssm_g_full = Linear(config.dim, ssm.inner_size, bias=False)
      else: self.ssm_g_a, self.ssm_g_b = Linear(config.dim, self.head_v_dim, bias=False), Linear(self.head_v_dim, ssm.inner_size, bias=False)
      self.ssm_f_a, self.ssm_f_b = Linear(config.dim, self.head_k_dim, bias=False), Linear(self.head_k_dim, ssm.inner_size, bias=False)
    else:
      self.attn_gate = Linear(config.dim, ssm.inner_size, bias=False)
      self.ssm_alpha = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_beta = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_dt = {"bias": Tensor.zeros(ssm.inner_size if ssm.kda else self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.head_v_dim if ssm.channel_decay else self.num_v_heads, 1) if ssm.kda else Tensor.zeros(self.num_v_heads)
    self.kda_channel_decay = ssm.channel_decay
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), Linear(ssm.inner_size, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape

    # input processing
    # Kimi-Linear is a BF16 model. Qwen 3.5 GGDN checkpoints historically use FP16 here.
    x = x.cast(dtypes.bfloat16) if self.config.ssm and self.config.ssm.kda else x.half()
    fused_fg = hasattr(self, "ssm_g_a") and resolve(T == 1) and amd_custom_kernels_supported(x.device) and \
      self.ssm_g_a.weight.shape == self.ssm_f_a.weight.shape
    mfma_decode = resolve(T == 1) and x.shape[-1] % 256 == 0 and amd_wave64_custom_kernels_supported(x.device)
    if hasattr(self, "ssm_g_full"): out_gate = self.ssm_g_full(x)
    elif hasattr(self, "ssm_g_a"):
      if fused_fg:
        gate_a, alpha_a, beta_logits = kda_fgb_linear(x, self.ssm_g_a.weight, self.ssm_f_a.weight, self.ssm_beta.weight)
        out_gate, alpha_logits = dual_input_bf16_matvec(gate_a, alpha_a, self.ssm_g_b.weight, self.ssm_f_b.weight)
      else: out_gate = self.ssm_g_b(self.ssm_g_a(x))
    else: out_gate = self.attn_gate(x)
    if not fused_fg: beta_logits = self.ssm_beta(x)
    if not fused_fg:
      if hasattr(self, "ssm_f_a"):
        f_a = bf16_mfma_splitk(x, self.ssm_f_a.weight) if mfma_decode and self.ssm_f_a.weight.shape[0] % 16 == 0 else self.ssm_f_a(x)
        alpha_logits = self.ssm_f_b(f_a)
      else: alpha_logits = self.ssm_alpha(x)

    # Causal depthwise Q/K/V convolution. All tokens are projected together, then the recurrent
    # update is fused into one kernel so prefill doesn't build a Python-unrolled graph.
    split_qkv = hasattr(self, "attn_q")
    if split_qkv:
      if resolve(T == 1) and amd_packed_mxfp4_supported(x.device) and \
         self.attn_q.weight.shape == self.attn_k.weight.shape == self.attn_v.weight.shape:
        projected_q, projected_k, projected_v = kda_qkv_linear(x, self.attn_q.weight, self.attn_k.weight, self.attn_v.weight)
      else: projected_q, projected_k, projected_v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
      # Snapshot mutable caches before constructing the recurrence. Otherwise the final store can
      # overwrite their buffers before earlier outputs in a multi-token lazy graph consume them.
      conv_state_q, conv_state_k, conv_state_v = self.conv_state_q.clone(), self.conv_state_k.clone(), self.conv_state_v.clone()
    else: projected, conv_state = self.attn_qkv(x), self.conv_state
    def causal_conv(projected:Tensor, state:Tensor, weight:Tensor) -> tuple[Tensor, Tensor]:
      window = state.cat(projected, dim=1)
      out = functools.reduce(lambda a,b: a+b, (window[:, i:i+T] * weight[:, i] for i in range(self.ssm_conv_kernel))).silu()
      return out, window[:, T:T+self.ssm_conv_kernel-1]
    if split_qkv:
      q, conv_state_q = causal_conv(projected_q, conv_state_q, self.ssm_q_conv1d["weight"])
      k, conv_state_k = causal_conv(projected_k, conv_state_k, self.ssm_k_conv1d["weight"])
      v, conv_state_v = causal_conv(projected_v, conv_state_v, self.ssm_v_conv1d["weight"])
    else:
      conv_out, conv_state = causal_conv(projected, conv_state, self.ssm_conv1d["weight"])
      q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)

    q, k = q.reshape(B, T, self.num_k_heads, self.head_k_dim), k.reshape(B, T, self.num_k_heads, self.head_k_dim)
    q, k = (l2norm(q), l2norm(k)) if self.config.ssm and self.config.ssm.kda else (q.normalize(dim=-1), k.normalize(dim=-1))
    q = q.repeat(1, 1, self.num_v_heads//self.num_k_heads, 1).transpose(1, 2).float() * self.head_k_dim**-0.5
    k = k.repeat(1, 1, self.num_v_heads//self.num_k_heads, 1).transpose(1, 2).float()
    v = v.reshape(B, T, self.num_v_heads, self.head_v_dim).transpose(1, 2).float()
    beta = (beta_logits.float() if self.config.ssm and self.config.ssm.kda else beta_logits).sigmoid().transpose(1, 2)
    gate_logits = (alpha_logits.float() + self.ssm_dt["bias"]).reshape(B, T, self.num_v_heads, -1)
    a_shape = (1, 1, 1, self.head_v_dim) if self.kda_channel_decay else (1, 1, self.num_v_heads, 1)
    if self.config.kda_gate_lower_bound:
      log_alpha = self.config.kda_gate_lower_bound * ((-self.ssm_a).reshape(a_shape) * gate_logits).sigmoid()
    else: log_alpha = gate_logits.softplus() * self.ssm_a.reshape(a_shape)
    alpha = log_alpha.squeeze(-1).transpose(1, 2).exp() if log_alpha.shape[-1] == 1 else log_alpha.permute(0, 2, 1, 3).exp()
    if T == 1:
      # Keep decode on the small elementwise graph. The fused prefill kernel writes a temporary
      # recurrent matrix, which is worthwhile for multiple tokens but needlessly copies state at T=1.
      decay = alpha if len(alpha.shape) == 4 else alpha.unsqueeze(-1)
      recurrent_state = self.recurrent_state * decay
      k1, q1 = k[:, :, 0].unsqueeze(-1), q[:, :, 0].unsqueeze(-1)
      recurrent_state = recurrent_state + ((v[:, :, 0].unsqueeze(-1) - recurrent_state@k1) * beta[:, :, 0].reshape(B, self.num_v_heads, 1, 1)) @ \
                        k1.transpose(-1, -2)
      core = (recurrent_state @ q1).squeeze(-1).unsqueeze(2)
    else: core, recurrent_state = gated_delta_prefill(q, k, v, beta, alpha, self.recurrent_state)
    core = core.transpose(1, 2)

    # Store each cache with its own AFTER. Multi-device lowering handles one sharded STORE per
    # AFTER; grouping these effects under one cache silently drops stores on the other shards.
    state_updates:list[Tensor]
    if split_qkv:
      state_updates = [self.conv_state_q.assign(conv_state_q.cast(self.conv_state_q.dtype)),
                       self.conv_state_k.assign(conv_state_k.cast(self.conv_state_k.dtype)),
                       self.conv_state_v.assign(conv_state_v.cast(self.conv_state_v.dtype))]
    else: state_updates = [self.conv_state.assign(conv_state.cast(self.conv_state.dtype))]
    state_updates.append(self.recurrent_state.assign(recurrent_state.cast(self.recurrent_state.dtype)))
    core_attn_out = self.ssm_norm(core.cast(x.dtype) if self.config.ssm and self.config.ssm.kda else core)
    gate = out_gate.reshape(B, T, self.num_v_heads, self.head_v_dim)
    gate = gate.float().sigmoid().cast(core_attn_out.dtype) if hasattr(self, "ssm_g_a") else gate.silu()
    out = (core_attn_out * gate).reshape(B, T, -1)
    out = out.cast(x.dtype)
    ret = bf16_partial_linear(out, self.ssm_out.weight).sum(3).cast(dtypes.bfloat16) if resolve(T == 1) and \
      isinstance(self.ssm_out.weight.device, tuple) and self.ssm_out.weight.shape[1] % (32*len(self.ssm_out.weight.device)) == 0 and \
      amd_exact_bf16_custom_kernels_supported(out.device) else self.ssm_out(out)
    return ret.realize(*state_updates)

  # Recurrent state can be reused only when the new prompt exactly extends all currently valid state.
  def _state_tensors(self) -> tuple[Tensor, ...]:
    if hasattr(self, "conv_state_q"):
      return self.conv_state_q, self.conv_state_k, self.conv_state_v, self.recurrent_state
    return (self.conv_state, self.recurrent_state) if hasattr(self, "conv_state") else ()
  def _state_reset_ops(self): return [s.assign(s.const_like(0)) for s in self._state_tensors()]
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len if prefix_len == cached_len else 0

  def _init_state(self, x):
    if not hasattr(self, "conv_state") and not hasattr(self, "conv_state_q"):
      if hasattr(self, "attn_q"):
        device = x.device[0] if isinstance(x.device, tuple) else x.device
        self.conv_state_q = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.q_dim, device=device, dtype=x.dtype).clone()
        self.conv_state_k = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.q_dim, device=device, dtype=x.dtype).clone()
        self.conv_state_v = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.num_v_heads*self.head_v_dim, device=device, dtype=x.dtype).clone()
        self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_k_dim, device=device).clone()
        if isinstance(x.device, tuple):
          for state in (self.conv_state_q, self.conv_state_k, self.conv_state_v): state.shard_(x.device, axis=2)
          self.recurrent_state.shard_(x.device, axis=1)
      else:
        self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device, dtype=x.dtype).clone()
        self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_k_dim, device=x.device).clone()

class Transformer:
  def __init__(self, config:TransformerConfig):
    self.config = config
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
    self.blk:list[FFNBlock] = [GatedDeltaNetBlock(dense_config if i < config.leading_dense_blocks else config, config.ssm)
                               if config.ssm and config.ssm_layers[i] else
                               block_cls(dense_config if i < config.leading_dense_blocks else config) for i in range(config.num_blocks)]
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = Linear(config.dim, config.vocab_size, bias=False)
    if config.attn_res_block_size:
      self.output_attn_res_norm = nn.RMSNorm(config.dim, config.norm_eps)
      self.output_attn_res_proj = Linear(config.dim, 1, bias=False)
    self.max_context = config.max_context
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    self._state_zero_hosts:dict[int, memoryview] = {}
    self._token_buffer:Tensor|None = None
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)
    self.greedy_prefill_jit = TinyJit(self.forward)
    self.greedy_rollout_jit = TinyJit(self.forward)
    self.recurrent_prefill_jits:dict[int, Callable[..., Tensor]] = {}
    self.recurrent_greedy_prefill_jits:dict[int, Callable[..., Tensor]] = {}
    self.reset_jit = TinyJit(self._reset_state)

  def _reset_state(self) -> None:
    if resets := [r for b in self.blk for r in b._state_reset_ops()]: Tensor.realize(*resets)

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor|None) -> Tensor:
    if len(tokens.shape) == 1: tokens = tokens.reshape(1, -1)
    x = self.token_embd(tokens).cast(dtypes.bfloat16) if self.config.bf16_activations else self.token_embd(tokens).float()
    block_residual = Tensor.zeros(x.shape[0]*x.shape[1], 0, x.shape[2], device=x.device, dtype=x.dtype) \
      if self.config.attn_res_block_size else None
    for i, block in enumerate(self.blk):
      if block_residual is not None: x, block_residual = block.attn_residual(x, start_pos, block_residual, i)
      else: x = block(x, start_pos)
      # Tensor indexing lowers selected experts through a fused one-hot reduction. Keeping all 26
      # of those high-level graphs alive until the final output is scheduled exhausts host memory.
      # A realization boundary lowers one block at a time; TinyJit still captures and memory-plans
      # the resulting schedules for rollout replay.
      if self.config.expert_mxfp4: x.realize()
    if block_residual is not None:
      x = FFNBlock._apply_attn_res(x.reshape(-1, x.shape[-1]), block_residual,
                                   self.output_attn_res_proj, self.output_attn_res_norm).reshape(x.shape)
    final_x = self.output_norm(x)
    if temperature is None and resolve(tokens.numel() == 1) and amd_exact_bf16_custom_kernels_supported(x.device):
      return bf16_matvec(final_x, self.output.weight).argmax(-1, keepdim=True)
    logits = self.output(final_x)[:, -1, :]
    if temperature is None: return logits.argmax(-1, keepdim=True)
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor|None) -> Tensor:
    token_count = tokens.numel()
    if self.has_recurrent_block and resolve(token_count != 1):
      assert isinstance(token_count, int)
      cache = self.recurrent_greedy_prefill_jits if temperature is None else self.recurrent_prefill_jits
      jit = cache.setdefault(token_count, TinyJit(self.forward))
      return jit(tokens.flatten().contiguous(), start_pos, temperature)
    if temperature is None:
      return (self.greedy_prefill_jit if resolve(token_count != 1) else self.greedy_rollout_jit)(tokens.flatten().contiguous(), start_pos, None)
    return (self.prefill_jit if resolve(token_count != 1) else self.rollout_jit)(tokens.flatten().contiguous(), start_pos, temperature)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0))) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    kv, state_dict = gguf_load(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

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
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    return model, kv

  def warmup(self):
    # Capture the only two shapes used by recurrent serving: a full prefill chunk and one-token rollout.
    # Two chunks exercise both the initial and nonzero-position prefill paths before the server opens.
    recurrent_chunk = self.config.recurrent_prefill_chunk_size or 32
    prompt = [0] * max(1, min(recurrent_chunk*2, self.max_context-2)) if self.has_recurrent_block else [0]
    # Recurrent serving also executes one replay so graph creation/lowering cannot leak into request latency.
    for _ in range(3 if self.has_recurrent_block else 2): list(zip(range(2), self.generate(prompt)))

  def _reset_amd_state(self) -> None:
    """Zero realized recurrent buffers without changing their UOp identities or invalidating captured graphs."""
    for state in (s for block in self.blk if isinstance(block, GatedDeltaNetBlock) for s in block._state_tensors()):
      realized = state.uop.buf_uop.buffer
      buffers = realized.bufs if isinstance(realized, MultiBuffer) else [realized]
      for buf in buffers:
        zero = self._state_zero_hosts.setdefault(buf.nbytes, memoryview(bytearray(buf.nbytes)))
        buf.ensure_allocated().allocator._copyin(buf._buf, zero)

  def get_start_pos(self, tokens:list[int]) -> int:
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def generate(self, tokens:list[int], chunk_size:int=32, temperature:float=0.0):
    chunked_recurrent = self.has_recurrent_block and self.config.recurrent_prefill_chunked
    if chunked_recurrent and self.config.recurrent_prefill_chunk_size:
      chunk_size = min(chunk_size, self.config.recurrent_prefill_chunk_size)
    if self.has_recurrent_block and not chunked_recurrent: chunk_size = 1
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    model_device = self.token_embd.weight.device
    temp = None if temperature == 0.0 else Tensor([temperature], device=model_device)
    # Keep the replicated AMD token buffer identity stable across HTTP requests so captured graphs
    # see the same input topology. Updating this small int32 buffer is cheaper than rebuilding JITs.
    if isinstance(model_device, tuple) and all(d.startswith("AMD") for d in model_device):
      if self._token_buffer is None:
        self._token_buffer = Tensor.empty(1, self.max_context, dtype=dtypes.int32, device=model_device).realize()
      token_storage = self._token_buffer.uop.buf_uop.buffer
      token_buffers = token_storage.bufs if isinstance(token_storage, MultiBuffer) else [token_storage]
      input_host = memoryview(array.array('i', tokens + [0] * (self.max_context-len(tokens)))).cast('B')
      for buf in token_buffers: buf.ensure_allocated().allocator._copyin(buf._buf, input_host)
      t = self._token_buffer
    else: t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32", device=model_device).reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    if start_pos < len(self._cached_tokens) and self.has_recurrent_block:
      if isinstance(model_device, tuple) and all(d.startswith("AMD") for d in model_device): self._reset_amd_state()
      else: self.reset_jit()
    out, prompt_len = None, len(tokens)
    token_host = memoryview(bytearray(4)) if isinstance(model_device, tuple) and all(d.startswith("AMD") for d in model_device) else None
    while len(tokens) < self.max_context:
      remaining = len(tokens) - start_pos
      # Full recurrent chunks use the high-throughput prefill graph. Process the tail through the
      # rollout graph so every request uses only the two shapes captured during server warmup.
      n_toks = chunk_size if chunked_recurrent and remaining >= chunk_size else 1 if chunked_recurrent else min(chunk_size, remaining)
      # Recurrent blocks execute an explicit recurrence over T. Give them a static chunk length so
      # Python constructs the recurrence once per encountered size; decode remains the T=1 JIT.
      if chunked_recurrent:
        # Token count is static for the recurrent kernel, but cache position must remain a runtime
        # variable so repeated chunks do not replay MLA stores at the capture position.
        sp = v_start_pos.bind(start_pos)
        model_input = t[:, sp:sp+n_toks] if start_pos < prompt_len or out is None else out
      else:
        sp = v_start_pos.bind(start_pos)
        nt = v_toks.bind(n_toks)
        model_input = t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out
      out = self(model_input, sp, temp).realize()
      start_pos += n_toks
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(amd_int32_item(out, token_host) if token_host is not None else int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]
