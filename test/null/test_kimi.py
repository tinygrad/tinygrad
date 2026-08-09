import unittest
from tinygrad import Tensor, dtypes, nn
from tinygrad.llm.kimi import _shard_kimi
from tinygrad.llm.model import SSMConfig, Transformer, TransformerConfig

class TestKimiTP4(unittest.TestCase):
  def test_prefill_and_decode_graph(self):
    devices = ("NULL:0", "NULL:1", "NULL:2", "NULL:3")
    config = TransformerConfig(num_blocks=2, dim=32, hidden_dim=128, n_heads=4, n_kv_heads=1, norm_eps=1e-5,
      vocab_size=64, head_dim=12, rope_theta=10000, rope_dim=4, v_head_dim=8, max_context=4, kv_lora_rank=16,
      num_experts=8, num_experts_per_tok=2, norm_topk_prob=True, shared_expert_dim=32, ssm_layers=(True, False),
      ssm=SSMConfig(4, 8, 4, 4, 32, True), shared_expert_gate=False, leading_dense_blocks=1, dense_hidden_dim=64,
      routed_scaling_factor=2.446, expert_bias=True, expert_mxfp4=True, bf16_activations=True, kda_split_qkv=True)
    model = Transformer(config)
    for name, value in nn.state.get_state_dict(model).items():
      fill = 127 if name.endswith("weight_scale") else 0
      value.replace(Tensor.full(value.shape, fill, dtype=value.dtype if value.dtype is dtypes.uint8 else dtypes.bfloat16, device="NULL"))
    _shard_kimi(model, devices)

    temperature = Tensor([0.0], device=devices)
    prefill = model(Tensor([[1, 2]], dtype=dtypes.int32, device=devices), 0, temperature).realize()
    model(Tensor([[1, 2]], dtype=dtypes.int32, device=devices), 0, temperature).realize()  # replay prefill JIT
    decode = model(Tensor([[3]], dtype=dtypes.int32, device=devices), 2, temperature).realize()
    model(Tensor([[4]], dtype=dtypes.int32, device=devices), 3, temperature).realize()  # replay decode JIT
    self.assertEqual(prefill.shape, (1, 1))
    self.assertEqual(decode.shape, (1, 1))
    self.assertEqual(model.blk[0].recurrent_state.uop.axis, 1)
    self.assertEqual(model.blk[1].cache_k.dtype, dtypes.bfloat16)

if __name__ == "__main__": unittest.main()
