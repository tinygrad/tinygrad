import unittest
from tinygrad import dtypes, nn
from tinygrad.llm.kimi import KIMI_LOGICAL_BYTES, KIMI_SSM_LAYERS, KIMI_TENSOR_COUNT, _shard_kimi, _validate_kimi_state, kimi_config
from tinygrad.llm.model import Transformer

class TestKimiLinear(unittest.TestCase):
  def test_architecture_config(self):
    config = kimi_config(4096)
    self.assertEqual((config.num_blocks, config.dim, config.n_heads, config.vocab_size), (27, 2304, 32, 163840))
    self.assertEqual(tuple(i for i, is_kda in enumerate(KIMI_SSM_LAYERS) if not is_kda), (3, 7, 11, 15, 19, 23, 26))
    self.assertEqual((config.num_experts, config.num_experts_per_tok, config.shared_expert_dim), (256, 8, 1024))
    self.assertTrue(config.expert_mxfp4 and config.bf16_activations and config.kda_split_qkv)
    self.assertFalse(config.shared_expert_gate)

  def test_tp4_schema_and_axes(self):
    model = Transformer(kimi_config(32))
    state = nn.state.get_state_dict(model)
    self.assertEqual(len(state), KIMI_TENSOR_COUNT)
    self.assertNotIn("blk.1.ffn_gate_inp_shexp.weight", state)
    self.assertEqual(state["blk.1.ffn_gate_exps.weight"].dtype, dtypes.uint8)
    self.assertEqual(state["blk.1.ffn_gate_exps.weight_scale"].dtype, dtypes.uint8)

    _shard_kimi(model, ("NULL:0", "NULL:1", "NULL:2", "NULL:3"))
    state = nn.state.get_state_dict(model)
    for name, axis in (("token_embd.weight", 0), ("blk.1.ffn_gate_exps.weight", 1),
                       ("blk.1.ffn_down_exps.weight_scale", 2), ("blk.3.attn_k_b.weight", 0)):
      self.assertEqual(state[name].uop.axis, axis, name)
    self.assertIsNone(state["blk.1.attn_norm.weight"].uop.axis)

  def test_converted_schema_validation(self):
    model = Transformer(kimi_config(1))
    state = {name:value if value.dtype is dtypes.uint8 else value.cast(dtypes.bfloat16)
             for name,value in nn.state.get_state_dict(model).items()}
    _validate_kimi_state(model, state)
    self.assertEqual(sum(value.nbytes() for value in state.values()), KIMI_LOGICAL_BYTES)

if __name__ == "__main__": unittest.main()
