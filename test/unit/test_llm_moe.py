import unittest
from tinygrad.apps.llm import Transformer, TransformerConfig

class TestLLMMoE(unittest.TestCase):
  def test_dense_leading_blocks_disable_experts(self):
    cfg = TransformerConfig(num_blocks=3, dim=64, hidden_dim=128, n_heads=4, n_kv_heads=4,
      norm_eps=1e-5, vocab_size=100, head_dim=16, rope_theta=10000.0, max_context=32,
      num_experts=8, num_experts_per_tok=2, leading_dense_blocks=1, dense_hidden_dim=96)
    model = Transformer(cfg)
    self.assertFalse(hasattr(model.blk[0], 'ffn_gate_exps'))
    self.assertTrue(hasattr(model.blk[1], 'ffn_gate_exps'))

  def test_moe_forward_is_finite(self):
    cfg = TransformerConfig(num_blocks=1, dim=64, hidden_dim=128, n_heads=4, n_kv_heads=4,
      norm_eps=1e-5, vocab_size=100, head_dim=16, rope_theta=10000.0, max_context=16,
      num_experts=8, num_experts_per_tok=2, routed_scaling_factor=1.0)
    model = Transformer(cfg)
    toks = [1,2,3,4]
    out = [t for _, t in zip(range(3), model.generate(list(toks)))]
    self.assertEqual(len(out), 3)

if __name__ == '__main__':
  unittest.main()
