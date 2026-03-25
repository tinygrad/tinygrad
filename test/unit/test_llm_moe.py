import unittest
import numpy as np
from dataclasses import replace
from tinygrad import Tensor
from tinygrad.apps.llm import Transformer, TransformerBlock, TransformerConfig


def _moe_config(dim=8, hidden=16, n_heads=2, num_experts=4, num_experts_per_tok=2):
  return TransformerConfig(num_blocks=1, dim=dim, hidden_dim=hidden, n_heads=n_heads, n_kv_heads=n_heads,
                           norm_eps=1e-5, vocab_size=100, head_dim=dim//n_heads, rope_theta=10000, max_context=16,
                           num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)


class TestMoEFeedForward(unittest.TestCase):
  def _setup_experts(self, block: TransformerBlock, dim=8, hidden=16, num_experts=4):
    # expert i produces a predictable magnitude via gate_exps scaling (i+1)
    block.ffn_gate_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) * (i + 1) for i in range(num_experts)])
    block.ffn_up_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) for _ in range(num_experts)])
    block.ffn_down_exps.weight = Tensor.stack(*[Tensor.eye(dim, hidden) for _ in range(num_experts)])
    block.ffn_norm.weight = Tensor.ones(dim)  # make norm a no-op

  def test_moe_feed_forward_softmax_path(self):
    dim, hidden, n_heads = 8, 16, 2
    num_experts, k = 4, 2
    block = TransformerBlock(_moe_config(dim, hidden, n_heads, num_experts, k))
    self._setup_experts(block, dim, hidden, num_experts)

    # router prefers experts 0 and 2
    block.ffn_gate_inp.weight = Tensor([[1, 0, 1, 0]] * dim).T
    h = Tensor.ones(1, 1, dim)
    out = block._feed_forward(h)

    expected = 1 + (Tensor([1.0]).silu().item() + Tensor([3.0]).silu().item()) / 2
    np.testing.assert_allclose(out.numpy()[0, 0, 0], expected, rtol=1e-2)

  def test_moe_feed_forward_norm_topk_prob(self):
    dim, hidden, n_heads = 8, 16, 2
    num_experts, k = 4, 2
    block = TransformerBlock(replace(_moe_config(dim, hidden, n_heads, num_experts, k), norm_topk_prob=True))
    self._setup_experts(block, dim, hidden, num_experts)

    # equal top-2 experts, normalization should keep mean behavior stable
    block.ffn_gate_inp.weight = Tensor([[0.1, 0, 0.1, 0]] * dim).T
    h = Tensor.ones(1, 1, dim)
    out = block._feed_forward(h)

    expected = 1 + (Tensor([1.0]).silu().item() + Tensor([3.0]).silu().item()) / 2
    np.testing.assert_allclose(out.numpy()[0, 0, 0], expected, rtol=1e-2)

  def test_deepseek_sigmoid_routing_bias_affects_selection(self):
    dim, hidden, n_heads = 8, 16, 2
    num_experts, k = 4, 2
    cfg = replace(_moe_config(dim, hidden, n_heads, num_experts, k),
                  kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4, routed_scaling_factor=1.0)
    block = TransformerBlock(cfg)
    self._setup_experts(block, dim, hidden, num_experts)

    # zero logits => sigmoid probs all 0.5, selection is controlled only by exp_probs_b_bias
    block.ffn_gate_inp.weight = Tensor.zeros(*block.ffn_gate_inp.weight.shape)
    block.exp_probs_b_bias = Tensor([-100.0, -100.0, 100.0, 100.0])

    h = Tensor.ones(1, 1, dim)
    out = block._feed_forward(h)

    # expects experts 2 and 3 chosen, but weighted by unbiased probs (0.5 each)
    expected = 1 + 0.5 * Tensor([3.0]).silu().item() + 0.5 * Tensor([4.0]).silu().item()
    np.testing.assert_allclose(out.numpy()[0, 0, 0], expected, rtol=1e-2)

  def test_routed_scaling_factor_scales_output(self):
    dim, hidden, n_heads = 8, 16, 2
    num_experts, k = 4, 2
    base = replace(_moe_config(dim, hidden, n_heads, num_experts, k),
                   kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4, routed_scaling_factor=1.0)
    scaled = replace(base, routed_scaling_factor=0.5)

    b1, b2 = TransformerBlock(base), TransformerBlock(scaled)
    self._setup_experts(b1, dim, hidden, num_experts)
    self._setup_experts(b2, dim, hidden, num_experts)

    # identical routing behavior
    for b in (b1, b2):
      b.ffn_gate_inp.weight = Tensor.zeros(*b.ffn_gate_inp.weight.shape)
      b.exp_probs_b_bias = Tensor([-100.0, -100.0, 100.0, 100.0])

    h = Tensor.ones(1, 1, dim)
    out1 = b1._feed_forward(h).numpy()[0, 0, 0]
    out2 = b2._feed_forward(h).numpy()[0, 0, 0]

    # residual (1.0) is unchanged; routed contribution should scale by ~0.5
    contrib1, contrib2 = out1 - 1.0, out2 - 1.0
    np.testing.assert_allclose(contrib2 / contrib1, 0.5, rtol=1e-2)


class TestMoEModelConfig(unittest.TestCase):
  def test_dense_leading_blocks_disable_experts(self):
    cfg = TransformerConfig(num_blocks=3, dim=64, hidden_dim=128, n_heads=4, n_kv_heads=4,
      norm_eps=1e-5, vocab_size=100, head_dim=16, rope_theta=10000.0, max_context=32,
      num_experts=8, num_experts_per_tok=2, leading_dense_blocks=1, dense_hidden_dim=96)
    model = Transformer(cfg)
    self.assertFalse(hasattr(model.blk[0], 'ffn_gate_exps'))
    self.assertTrue(hasattr(model.blk[1], 'ffn_gate_exps'))


if __name__ == '__main__':
  unittest.main()
