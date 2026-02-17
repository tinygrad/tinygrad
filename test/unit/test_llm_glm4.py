import unittest
from tinygrad import Tensor, Device, UOp
from tinygrad.apps.llm import Transformer

class TestGLM4(unittest.TestCase):
  def test_glm4_architecture_config(self):
    # GLM-4.7 / DeepSeek-V2 style parameters- because quantized version use this architecture
    params = {
      "num_blocks": 2,
      "dim": 64,
      "hidden_dim": 128,
      "n_heads": 4,
      "n_kv_heads": 4,
      "norm_eps": 1e-5,
      "vocab_size": 1000,
      "head_dim": 32,
      "rope_theta": 1000000.0,
      "max_context": 128,
      "num_experts": 8,
      "num_experts_per_tok": 2,
      "q_lora_rank": 32,
      "kv_lora_rank": 32,
      "qk_nope_head_dim": 16,
      "qk_rope_head_dim": 16,
      "v_head_dim": 16,
      "num_shared_experts": 1,
      "shared_hidden_dim": 32,
      "routed_scaling_factor": 1.8,
      "norm_topk_prob": True,
      "expert_hidden_dim": 16,
      "leading_dense_block_count": 1
    }

    model = Transformer(**params)
    
    # 1. Verify leading dense blocks (Block 0)
    self.assertFalse(hasattr(model.blk[0], 'ffn_gate_exps'), "Block 0 should be DENSE")
    self.assertTrue(hasattr(model.blk[0], 'ffn_gate'), "Block 0 should have dense ffn_gate")
    
    # 2. Verify MoE blocks (Block 1+)
    self.assertTrue(hasattr(model.blk[1], 'ffn_gate_exps'), "Block 1 should be MoE")
    self.assertEqual(model.blk[1].num_experts_per_tok, 2)
    self.assertTrue(hasattr(model.blk[1], 'ffn_gate_shared'), "Block 1 should have shared experts")

    # 3. Verify MLA params
    self.assertEqual(model.blk[0].q_lora_rank, 32)
    self.assertEqual(model.blk[0].kv_lora_rank, 32)
    self.assertEqual(model.blk[0].qk_nope_head_dim, 16)
    self.assertEqual(model.blk[0].qk_rope_head_dim, 16)

    # 4. Functional test (Prefill)
    tokens = Tensor([[1, 2, 3, 4]], dtype="int32")
    out = model(tokens, 0)
    out.realize()
    self.assertEqual(out.shape, (1, 1))

    # 5. Functional test (Generation with symbolic start_pos)
    v_start_pos = UOp.variable("start_pos", 1, 127)
    tokens_gen = Tensor([[5]], dtype="int32")
    out_gen = model(tokens_gen, v_start_pos.bind(4))
    out_gen.realize()
    self.assertEqual(out_gen.shape, (1, 1))

if __name__ == '__main__':
  unittest.main()
