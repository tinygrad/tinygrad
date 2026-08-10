import unittest
from dataclasses import replace
import numpy as np
from tinygrad import Tensor, dtypes, nn
from tinygrad.llm.kimi_k3 import KIMI_K3_FULL_ATTN_LAYERS, KIMI_K3_SSM_LAYERS, KIMI_K3_TEXT_SIZE, KIMI_K3_TP8_BYTES_PER_GPU, \
  _layer_sources, _shard_kimi_k3, _validate_config, kimi_k3_config, kimi_k3_smoke_config
from tinygrad.llm.model import FFNBlock, Transformer

def small_k3_config(max_context:int=4): return replace(kimi_k3_smoke_config(max_context), num_experts=8)

class TestKimiK3(unittest.TestCase):
  def test_official_config(self):
    c = kimi_k3_config(1_048_576)
    self.assertEqual((c.num_blocks, c.dim, c.n_heads, c.num_experts, c.num_experts_per_tok), (93, 7168, 96, 896, 16))
    self.assertEqual((sum(KIMI_K3_SSM_LAYERS), len(KIMI_K3_FULL_ATTN_LAYERS)), (69, 24))
    self.assertEqual(KIMI_K3_FULL_ATTN_LAYERS, (*range(3, 93, 4), 92))
    self.assertEqual((c.routed_expert_dim, c.hidden_dim, c.shared_expert_dim), (3584, 3072, 6144))
    self.assertTrue(c.route_weights_uncorrected and c.kda_full_rank_gate and c.attn_output_gate)
    self.assertEqual((c.activation_situ_beta, c.activation_situ_linear_beta, c.kda_gate_lower_bound), (4.0, 25.0, -5.0))

  def test_config_rejects_wrong_checkpoint(self):
    with self.assertRaisesRegex(ValueError, "not the supported official"):
      _validate_config({"model_type":"kimi_linear", "hidden_size":2304})

  def test_official_mapping_covers_model(self):
    model = Transformer(kimi_k3_config(1))
    state = nn.state.get_state_dict(model)
    targets = {"token_embd.weight", "output_norm.weight", "output.weight", "output_attn_res_norm.weight", "output_attn_res_proj.weight"}
    for i,is_kda in enumerate(KIMI_K3_SSM_LAYERS):
      for target in _layer_sources(i, is_kda).values(): targets.update(target.split("|"))
      if i:
        for name in ("ffn_gate_exps.weight", "ffn_gate_exps.weight_scale", "ffn_up_exps.weight", "ffn_up_exps.weight_scale",
                     "ffn_down_exps.weight", "ffn_down_exps.weight_scale"): targets.add(f"blk.{i}.{name}")
    self.assertEqual(targets, set(state))
    self.assertEqual(state["blk.1.ffn_gate_exps.weight"].shape, (896, 3072, 1792))
    self.assertEqual(state["blk.1.ffn_gate_exps.weight_scale"].shape, (896, 3072, 112))
    _shard_kimi_k3(model, tuple(f"NULL:{i}" for i in range(8)))
    total, per_gpu = 0, 0
    for name,value in state.items():
      dtype = dtypes.uint8 if name.endswith(("weight_scale", "_exps.weight")) else \
        dtypes.float32 if name.endswith(("ssm_a", "ssm_dt.bias")) else dtypes.bfloat16
      size = value.numel() * dtype.itemsize
      total += size
      per_gpu += size if value.uop.axis is None else size//8
    self.assertEqual((total, per_gpu), (KIMI_K3_TEXT_SIZE, KIMI_K3_TP8_BYTES_PER_GPU))

  def test_situ_matches_reference(self):
    block = FFNBlock(small_k3_config())
    gate, up = Tensor([[-8., -1., 0., 3.]]), Tensor([[-30., -2., 5., 40.]])
    got = block._activation(gate, up).numpy()
    g, u = gate.numpy().astype(np.float32), up.numpy().astype(np.float32)
    expected = (4*np.tanh(g/4)/(1+np.exp(-g))) * (25*np.tanh(u/25))
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

  def test_attention_residual_matches_reference(self):
    block = FFNBlock(small_k3_config())
    block.attn_res_norm.weight.assign([1.0+i/16 for i in range(32)])
    block.attn_res_proj.weight.assign([[(-1.0)**i/8 for i in range(32)]])
    prefix, residual = Tensor.arange(64).reshape(2, 32).float()/16, Tensor.arange(128).reshape(2, 2, 32).float()/32
    got = block._apply_attn_res(prefix, residual, block.attn_res_proj, block.attn_res_norm).numpy()
    v = np.concatenate((residual.numpy(), prefix.numpy()[:, None]), axis=1).astype(np.float32)
    k = v / np.sqrt(np.mean(v*v, axis=-1, keepdims=True) + 1e-5)
    scores = np.sum(k * block.attn_res_norm.weight.numpy() * block.attn_res_proj.weight.numpy()[0], axis=-1)
    probs = np.exp(scores-scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    expected = np.matmul(probs[:, None], v).squeeze(1)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

  def test_tp8_schema(self):
    model = Transformer(small_k3_config())
    _shard_kimi_k3(model, tuple(f"NULL:{i}" for i in range(8)))
    state = nn.state.get_state_dict(model)
    for name,axis in (("token_embd.weight",0), ("blk.1.ffn_gate_exps.weight",1), ("blk.1.ffn_down_exps.weight_scale",2),
                      ("blk.1.ffn_routed_down.weight",1), ("blk.0.ssm_g_full.weight",0), ("blk.1.attn_q_b.weight",0)):
      self.assertEqual(state[name].uop.axis, axis, name)
    self.assertIsNone(state["blk.1.attn_res_norm.weight"].uop.axis)
    self.assertIsNone(state["blk.1.ffn_routed_norm.weight"].uop.axis)

  def test_chunked_recurrent_generate(self):
    model = Transformer(small_k3_config(max_context=8))
    for name,value in nn.state.get_state_dict(model).items():
      fill = 127 if name.endswith("weight_scale") else 0
      value.replace(Tensor.full(value.shape, fill, dtype=value.dtype if value.dtype is dtypes.uint8 else dtypes.bfloat16, device="PYTHON"))
    for _ in range(3): self.assertIsInstance(next(model.generate([1, 2, 3, 4], chunk_size=2)), int)
    self.assertEqual(model._cached_tokens[:4], [1, 2, 3, 4])

if __name__ == "__main__": unittest.main()
