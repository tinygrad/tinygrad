import unittest
import numpy as np
from tinygrad import Tensor

class TestExpertWeights(unittest.TestCase):
  def test_expert_weights_shape(self):
    from tinygrad.apps.llm import ExpertWeights
    num_experts, in_features, out_features = 8, 64, 128
    ew = ExpertWeights(num_experts, in_features, out_features)
    self.assertEqual(ew.weight.shape, (num_experts, out_features, in_features))

  def test_expert_weights_call_single_token(self):
    from tinygrad.apps.llm import ExpertWeights
    num_experts, in_features, out_features = 4, 8, 16
    T, k = 1, 2  # 1 token, 2 experts
    ew = ExpertWeights(num_experts, in_features, out_features)
    ew.weight = Tensor.ones(num_experts, out_features, in_features)

    x = Tensor.ones(T, 1, in_features)  # (T, 1, D)
    sel = Tensor([[0, 1]])  # (T, k)
    out = ew(sel, x)
    self.assertEqual(out.shape, (T, k, out_features))
    # ones @ ones.T = in_features for each output
    np.testing.assert_allclose(out.numpy(), np.full((T, k, out_features), in_features))

  def test_expert_weights_batch_tokens(self):
    from tinygrad.apps.llm import ExpertWeights
    num_experts, in_features, out_features = 8, 4, 6
    T, k = 3, 2  # 3 tokens, 2 experts per token
    ew = ExpertWeights(num_experts, in_features, out_features)
    ew.weight = Tensor.ones(num_experts, out_features, in_features)

    x = Tensor.ones(T, 1, in_features)  # (T, 1, D)
    sel = Tensor([[0, 1], [2, 3], [4, 5]])  # (T, k)
    out = ew(sel, x)
    self.assertEqual(out.shape, (T, k, out_features))
    np.testing.assert_allclose(out.numpy(), np.full((T, k, out_features), in_features))


class TestMoEFeedForward(unittest.TestCase):
  def test_moe_feed_forward_shapes(self):
    from tinygrad.apps.llm import TransformerBlock
    dim, hidden_dim, n_heads, n_kv_heads = 64, 128, 4, 2
    num_experts, num_experts_per_tok = 8, 2

    block = TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps=1e-5, head_dim=dim//n_heads,
                             rope_theta=10000, max_context=32, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)

    # Check MoE weights exist
    self.assertTrue(hasattr(block, 'ffn_gate_exps'))
    self.assertTrue(hasattr(block, 'ffn_up_exps'))
    self.assertTrue(hasattr(block, 'ffn_down_exps'))
    self.assertTrue(hasattr(block, 'ffn_gate_inp'))

    # Check shapes
    self.assertEqual(block.ffn_gate_exps.weight.shape, (num_experts, hidden_dim, dim))
    self.assertEqual(block.ffn_up_exps.weight.shape, (num_experts, hidden_dim, dim))
    self.assertEqual(block.ffn_down_exps.weight.shape, (num_experts, dim, hidden_dim))

  def test_moe_vs_dense_block_creation(self):
    from tinygrad.apps.llm import TransformerBlock
    dim, hidden_dim, n_heads, n_kv_heads = 64, 128, 4, 2

    # Dense block (no MoE)
    dense_block = TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps=1e-5, head_dim=dim//n_heads,
                                   rope_theta=10000, max_context=32, num_experts=0, num_experts_per_tok=0)
    self.assertFalse(hasattr(dense_block, 'ffn_gate_exps'))
    self.assertTrue(hasattr(dense_block, 'ffn_gate'))

    # MoE block
    moe_block = TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps=1e-5, head_dim=dim//n_heads,
                                 rope_theta=10000, max_context=32, num_experts=8, num_experts_per_tok=2)
    self.assertTrue(hasattr(moe_block, 'ffn_gate_exps'))
    self.assertFalse(hasattr(moe_block, 'ffn_gate'))

  def test_moe_feed_forward_single_token(self):
    from tinygrad.apps.llm import TransformerBlock
    dim, hidden_dim, n_heads, n_kv_heads = 32, 64, 2, 2
    num_experts, num_experts_per_tok = 4, 2

    block = TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps=1e-5, head_dim=dim//n_heads,
                             rope_theta=10000, max_context=16, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)

    # Initialize with small random weights for numerical stability
    block.ffn_gate_inp.weight = Tensor.randn(num_experts, dim) * 0.1
    block.ffn_gate_exps.weight = Tensor.randn(num_experts, hidden_dim, dim) * 0.1
    block.ffn_up_exps.weight = Tensor.randn(num_experts, hidden_dim, dim) * 0.1
    block.ffn_down_exps.weight = Tensor.randn(num_experts, dim, hidden_dim) * 0.1

    h = Tensor.randn(1, 1, dim)  # single token
    out = block._feed_forward(h)
    self.assertEqual(out.shape, (1, 1, dim))

  def test_moe_feed_forward_multiple_tokens(self):
    from tinygrad.apps.llm import TransformerBlock
    dim, hidden_dim, n_heads, n_kv_heads = 32, 64, 2, 2
    num_experts, num_experts_per_tok = 4, 2
    T = 5  # multiple tokens (prefill)

    block = TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps=1e-5, head_dim=dim//n_heads,
                             rope_theta=10000, max_context=16, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)

    # Initialize with small random weights
    block.ffn_gate_inp.weight = Tensor.randn(num_experts, dim) * 0.1
    block.ffn_gate_exps.weight = Tensor.randn(num_experts, hidden_dim, dim) * 0.1
    block.ffn_up_exps.weight = Tensor.randn(num_experts, hidden_dim, dim) * 0.1
    block.ffn_down_exps.weight = Tensor.randn(num_experts, dim, hidden_dim) * 0.1

    h = Tensor.randn(1, T, dim)  # multiple tokens
    out = block._feed_forward(h)
    self.assertEqual(out.shape, (1, T, dim))


if __name__ == '__main__':
  unittest.main()
