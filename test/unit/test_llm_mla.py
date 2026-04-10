import unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.apps.llm import Transformer, TransformerConfig, apply_rope_interleaved

class TestMLA(unittest.TestCase):
  def _make_config(self, **kwargs):
    return TransformerConfig(**{
      "num_blocks": 1, "dim": 64, "hidden_dim": 128, "n_heads": 4, "n_kv_heads": 1,
      "norm_eps": 1e-5, "vocab_size": 100, "head_dim": 0, "rope_theta": 10000.0, "rope_dim": 0,
      "max_context": 32, "kv_lora_rank": 16, "qk_nope_head_dim": 8, "qk_rope_head_dim": 8, "v_head_dim": 8,
    } | kwargs)

  def test_mla_attention_matches_naive(self):
    config = self._make_config(max_context=16)
    from tinygrad.apps.llm import MLATransformerBlock, precompute_freqs_cis

    block = MLATransformerBlock(config)
    c = config
    B, T = 1, 4

    x = Tensor.randn(B, T, c.dim)
    x_norm = block.attn_norm(x)

    # --- Our absorbed implementation ---
    q = block.attn_q(x_norm).reshape(B, T, c.n_heads, c.qk_nope_head_dim + c.qk_rope_head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :c.qk_nope_head_dim], q[..., c.qk_nope_head_dim:]
    freqs = precompute_freqs_cis(c.qk_rope_head_dim, 16, c.rope_theta)
    q_rope = apply_rope_interleaved(q_rope, freqs[0:T])

    kv_a = block.attn_kv_a_mqa(x_norm)
    c_kv = block.attn_kv_a_norm(kv_a[..., :c.kv_lora_rank])
    k_rope = kv_a[..., c.kv_lora_rank:].reshape(B, T, 1, c.qk_rope_head_dim).transpose(1, 2)
    k_rope = apply_rope_interleaved(k_rope, freqs[0:T])

    # --- Naive (non-absorbed): expand K and V, do standard attention ---
    k_nope_naive = c_kv.unsqueeze(1) @ block.attn_k_b_weight  # (B, H, T, nope)
    k_naive = k_nope_naive.cat(k_rope.expand(-1, c.n_heads, -1, -1), dim=-1)  # (B, H, T, nope+rope)
    v_naive = c_kv.unsqueeze(1) @ block.attn_v_b_weight.transpose(-1, -2)  # (B, H, T, v_dim)

    q_naive = q_nope.cat(q_rope, dim=-1)
    scale = 1.0 / (c.qk_nope_head_dim + c.qk_rope_head_dim) ** 0.5
    scores_naive = (q_naive @ k_naive.transpose(-1, -2)) * scale
    # causal mask
    mask = Tensor.full((1, 1, T, T), float("-inf")).triu(1)
    attn_naive = (scores_naive + mask).softmax(-1) @ v_naive  # (B, H, T, v_dim)
    out_naive = block.attn_output(attn_naive.transpose(1, 2).reshape(B, T, -1))

    # --- Absorbed: q_nope @ wk_b^T, then dot with compressed kv ---
    q_nope_abs = q_nope @ block.attn_k_b_weight.transpose(-1, -2)  # (B, H, T, lora)
    q_abs = q_nope_abs.cat(q_rope, dim=-1)  # (B, H, T, lora+rope)
    k_abs = c_kv.reshape(B, 1, T, c.kv_lora_rank).cat(k_rope.reshape(B, 1, T, c.qk_rope_head_dim), dim=-1)
    scores_abs = (q_abs @ k_abs.transpose(-1, -2)) * scale
    attn_abs = (scores_abs + mask).softmax(-1)
    # attn @ v_compressed @ wv_b
    v_compressed = c_kv.reshape(B, 1, T, c.kv_lora_rank)
    attn_abs_out = (attn_abs @ v_compressed) @ block.attn_v_b_weight.transpose(-1, -2)
    out_abs = block.attn_output(attn_abs_out.transpose(1, 2).reshape(B, T, -1))

    # Compare
    naive_np = out_naive.realize().numpy()
    abs_np = out_abs.realize().numpy()
    np.testing.assert_allclose(naive_np, abs_np, atol=1e-4, rtol=1e-4,
      err_msg="Absorbed MLA should match naive MLA")

  def test_shared_expert_gate_optional(self):
    from tinygrad import nn
    model = Transformer(self._make_config(num_experts=4, num_experts_per_tok=2, shared_expert_dim=32, shared_expert_gate=False))
    self.assertNotIn('blk.0.ffn_gate_inp_shexp.weight', nn.state.get_state_dict(model))
    out = [t for _, t in zip(range(5), model.generate([1, 2, 3, 4]))]
    self.assertEqual(len(out), 5)
