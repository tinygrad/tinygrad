from __future__ import annotations
import math
from tinygrad import Tensor, nn, UOp
from tinygrad.dtype import dtypes
from tinygrad.nn.rope import apply_rope_interleaved, precompute_rope_freqs_cis
from tinygrad.nn.moe import ExpertWeights

def _topk_pairwise(scores: Tensor, k: int) -> tuple[Tensor, Tensor]:
  """
  O(n^2) pairwise comparison topk. WAY faster for small n (e.g. 64 experts)
  TODO: should Tensor.topk use this for small n?
  """
  n = scores.shape[-1]
  s_col = scores.unsqueeze(-1)  # (..., n, 1)
  s_row = scores.unsqueeze(-2)  # (..., 1, n)
  gt = (s_row > s_col)
  eq = (s_row == s_col)
  j_idx = Tensor.arange(n).reshape(1, 1, n)
  i_idx = Tensor.arange(n).reshape(1, n, 1)
  ranks = (gt | (eq & (j_idx < i_idx))).sum(-1)  # (..., n), 0=largest
  target = Tensor.arange(k).reshape(1, k)
  match = (ranks.unsqueeze(-1) == target.unsqueeze(-2)).float()  # (..., n, k)
  i_range = Tensor.arange(n, dtype=dtypes.float).reshape(1, n, 1)
  indices = (match * i_range).sum(-2).cast(dtypes.int)
  values = scores.gather(-1, indices)
  return values, indices

class PerHeadWeights:
  def __init__(self, n_heads:int, dim1:int, dim2:int):
    self.weight = Tensor.zeros(n_heads, dim1, dim2)

class MLATransformerBlock:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, norm_eps:float, max_context:int,
               q_lora_rank:int, kv_lora_rank:int, qk_nope_head_dim:int, qk_rope_head_dim:int, v_head_dim:int,
               num_experts:int=0, num_experts_per_tok:int=0, n_shared_experts:int=0, moe_hidden_dim:int=0,
               expert_gating_func:int=0, expert_weights_norm:bool=False, expert_weights_scale:float=1.0, mscale:float=1.0,
               rope_theta:float=10000.0, yarn_scaling_factor:float=1.0, yarn_params=None):
    self.n_heads = n_heads
    self.qk_nope_head_dim = qk_nope_head_dim
    self.qk_rope_head_dim = qk_rope_head_dim
    self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    self.kv_lora_rank = kv_lora_rank
    self.q_lora_rank = q_lora_rank
    self.max_context = max_context
    self.mscale = mscale
    self.rope_theta = rope_theta
    self.yarn_scaling_factor = yarn_scaling_factor
    self.yarn_params = yarn_params
    self._attn_scale = mscale * mscale / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)
    if q_lora_rank > 0:
      self.attn_q_a = nn.Linear(dim, q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(q_lora_rank, norm_eps)
      self.attn_q_b = nn.Linear(q_lora_rank, n_heads * self.q_head_dim, bias=False)
    else:
      self.attn_q = nn.Linear(dim, n_heads * self.q_head_dim, bias=False)
    self.attn_kv_a_mqa = nn.Linear(dim, kv_lora_rank + qk_rope_head_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(kv_lora_rank, norm_eps)
    self.attn_k_b = PerHeadWeights(n_heads, kv_lora_rank, qk_nope_head_dim)
    self.attn_v_b = PerHeadWeights(n_heads, v_head_dim, kv_lora_rank)
    self.attn_output = nn.Linear(n_heads * v_head_dim, dim, bias=False)
    self.attn_norm = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm = nn.RMSNorm(dim, norm_eps)
    if num_experts > 0:
      self.num_experts_per_tok = num_experts_per_tok
      self.expert_gating_func = expert_gating_func
      self.expert_weights_norm = expert_weights_norm
      self.expert_weights_scale = expert_weights_scale
      self.ffn_gate_inp = nn.Linear(dim, num_experts, bias=False)
      class ExpProbsBias:
        def __init__(self): self.bias = Tensor.zeros(num_experts)
      self.exp_probs_b = ExpProbsBias()
      self.moe_hidden_dim = moe_hidden_dim
      self.ffn_gate_exps = ExpertWeights(num_experts, dim, moe_hidden_dim)
      self.ffn_up_exps = ExpertWeights(num_experts, dim, moe_hidden_dim)
      self.ffn_down_exps = ExpertWeights(num_experts, moe_hidden_dim, dim)
      if n_shared_experts > 0:
        shexp_hidden = n_shared_experts * moe_hidden_dim
        self.ffn_gate_shexp = nn.Linear(dim, shexp_hidden, bias=False)
        self.ffn_up_shexp = nn.Linear(dim, shexp_hidden, bias=False)
        self.ffn_down_shexp = nn.Linear(shexp_hidden, dim, bias=False)
    else:
      self.ffn_gate = nn.Linear(dim, hidden_dim, bias=False)
      self.ffn_up = nn.Linear(dim, hidden_dim, bias=False)
      self.ffn_down = nn.Linear(hidden_dim, dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    x_norm = self.attn_norm(x)
    B, T, _ = x.shape
    if self.q_lora_rank > 0: q = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x_norm)))
    else: q = self.attn_q(x_norm)
    kv_out = self.attn_kv_a_mqa(x_norm)
    q = q.reshape(B, T, self.n_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    compressed_kv, k_pe = kv_out.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.reshape(B, T, 1, self.qk_rope_head_dim).transpose(1, 2)
    freqs_all = precompute_rope_freqs_cis(self.qk_rope_head_dim, self.max_context, self.rope_theta, self.yarn_params, self.yarn_scaling_factor)
    freqs_cis = freqs_all[start_pos:start_pos+T]
    if freqs_cis.device != x.device: freqs_cis = freqs_cis.to(x.device)
    if freqs_cis.dtype != dtypes.float16: freqs_cis = freqs_cis.cast(dtypes.float16)
    q_pe = apply_rope_interleaved(q_pe, freqs_cis, use_float32=False)
    k_pe = apply_rope_interleaved(k_pe, freqs_cis, use_float32=False)
    q_nope = q_nope @ self.attn_k_b.weight.transpose(-1, -2)
    q = q_nope.cat(q_pe, dim=-1)
    kv_normed = self.attn_kv_a_norm(compressed_kv).unsqueeze(1)
    k_new = kv_normed.cat(k_pe, dim=-1)
    cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
    if not hasattr(self, "cache_k") or start_pos == 0:
      self.cache_k = Tensor.empty((B, 1, self.max_context, cache_dim), dtype=kv_normed.dtype, device=kv_normed.device).contiguous().realize()
    self.cache_k[:, :, start_pos:start_pos+T, :].assign(k_new).realize()
    k = self.cache_k[:, :, 0:start_pos+T, :]
    # Attention scores
    qk = q.matmul(k.transpose(-2, -1)) * self._attn_scale
    if T > 1:
      qk = qk + Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(int(start_pos)+1)
      attn_weights = qk.softmax(-1)
    else:
      e = qk.float().exp()
      attn_weights = (e / e.sum(-1, keepdim=True)).half()
    # Absorbed V: (attn @ kv_normed_cache) @ v_b^T
    attn = (attn_weights.matmul(k[:, :, :, :self.kv_lora_rank]) @ self.attn_v_b.weight.transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return x + self.attn_output(attn)

  def _feed_forward(self, h: Tensor) -> Tensor:
    h_norm = self.ffn_norm(h)
    if hasattr(self, 'ffn_gate_up_exps'):
      gate_weight = self.ffn_gate_inp_f32 if hasattr(self, 'ffn_gate_inp_f32') else self.ffn_gate_inp.weight.float()
      router_logits = h_norm.float() @ gate_weight.T
      if self.expert_gating_func == 2: gate_scores = router_logits.sigmoid()
      elif self.expert_gating_func == 3: gate_scores = router_logits
      else: gate_scores = router_logits.softmax(-1)
      selection_scores = gate_scores + self.exp_probs_b.bias if hasattr(self, 'exp_probs_b') else gate_scores
      _, sel = _topk_pairwise(selection_scores, self.num_experts_per_tok)
      probs = gate_scores.gather(-1, sel)
      if self.expert_gating_func == 3: probs = probs.softmax(-1)
      elif self.expert_weights_norm: probs = probs / probs.sum(axis=-1, keepdim=True).maximum(6.103515625e-5)
      gate_up = self.ffn_gate_up_exps(sel, h_norm)
      gate, up = gate_up.split([self.moe_hidden_dim, self.moe_hidden_dim], dim=-1)
      gated = (gate.silu() * up)
      weighted_gated = (gated * probs.unsqueeze(-1).cast(gated.dtype)).contiguous()
      expert_out = self.ffn_down_exps(sel, weighted_gated)
      if self.num_experts_per_tok == 4:
        moe = expert_out[:, :, 0] + expert_out[:, :, 1] + expert_out[:, :, 2] + expert_out[:, :, 3]
      else:
        moe = expert_out.sum(axis=2)
      out = moe * self.expert_weights_scale
      if hasattr(self, 'ffn_gate_up_shexp'):
        shexp_gate_up = self.ffn_gate_up_shexp(h_norm)
        shexp_dim = shexp_gate_up.shape[-1] // 2
        out = out + self.ffn_down_shexp(shexp_gate_up[..., :shexp_dim].silu() * shexp_gate_up[..., shexp_dim:])
      elif hasattr(self, 'ffn_gate_shexp'):
        out = out.contiguous() + self.ffn_down_shexp(self.ffn_gate_shexp(h_norm).silu() * self.ffn_up_shexp(h_norm))
      return h + out.cast(h.dtype)
    gated = self.ffn_gate(h_norm).silu() * self.ffn_up(h_norm)
    return h + self.ffn_down(gated)

  def __call__(self, x: Tensor, start_pos: int|UOp):
    return self._feed_forward(self._attention(x, start_pos)).contiguous()

def load_mla_params_from_gguf(kv: dict, arch: str) -> dict:
  """Extract MLA architecture params from GGUF metadata. Returns dict of MLA params."""
  def ak(s, d=0): return kv.get(f'{arch}.{s}', d)
  qk_rope_head_dim = ak('rope.dimension_count')
  key_length = ak('attention.key_length_mla', ak('attention.key_length'))
  return {
    'q_lora_rank': ak('attention.q_lora_rank'),
    'kv_lora_rank': ak('attention.kv_lora_rank'),
    'qk_rope_head_dim': qk_rope_head_dim,
    'qk_nope_head_dim': key_length - qk_rope_head_dim if key_length > 0 else 0,
    'v_head_dim': ak('attention.value_length_mla', ak('attention.value_length')),
    'n_shared_experts': ak('expert_shared_count'),
    'moe_hidden_dim': ak('expert_feed_forward_length'),
    'leading_dense_blocks': ak('leading_dense_block_count'),
    'expert_gating_func': ak('expert_gating_func') or (2 if arch == 'glm4' else 1),
    'expert_weights_norm': ak('expert_weights_norm', False),
    'expert_weights_scale': ak('expert_weights_scale', 1.0),
  }

def split_mla_kv_weights(state_dict: dict, quantized_tensors: dict|None, num_blocks: int, n_heads: int, mla: dict) -> None:
  """Split combined attn_kv_b into separate attn_k_b and attn_v_b for MLA models."""
  from tinygrad import getenv
  qk_nope, v_dim, kv_rank = mla['qk_nope_head_dim'], mla['v_head_dim'], mla['kv_lora_rank']
  def split_kvb(kv_b: Tensor) -> tuple[Tensor, Tensor]:
    k_b, v_b = kv_b.reshape(n_heads, qk_nope + v_dim, kv_rank).split([qk_nope, v_dim], dim=1)
    return k_b.transpose(1, 2), v_b  # k_b: (n_heads, kv_lora_rank, qk_nope), v_b: (n_heads, v_head, kv_lora_rank)
  for i in range(num_blocks):
    key = f'blk.{i}.attn_kv_b.weight'
    if key in state_dict:
      k_b, v_b = split_kvb(state_dict.pop(key))
    elif quantized_tensors and key in quantized_tensors:
      blocks, shape, ggml_type = quantized_tensors.pop(key)[:3]
      kv_b = nn.state.GGML_QUANT_INFO[ggml_type][2](blocks).reshape(*shape)
      k_b, v_b = split_kvb(kv_b.cast('float16') if getenv("HALF", 1) else kv_b)
    else: continue
    state_dict[f'blk.{i}.attn_k_b.weight'] = k_b
    state_dict[f'blk.{i}.attn_v_b.weight'] = v_b
