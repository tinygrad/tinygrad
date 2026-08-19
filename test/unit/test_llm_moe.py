import unittest
import numpy as np
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad import dtypes, GlobalCounters, Tensor, UOp
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.model import _attach_quantized_experts, ExpertGating, ExpertWeights, TransformerBlock, TransformerConfig

def _moe_config(dim=8, hidden=16, n_heads=2, num_experts=4, num_experts_per_tok=2):
  return TransformerConfig(
    num_blocks=1, dim=dim, hidden_dim=hidden, n_heads=n_heads, n_kv_heads=n_heads,
    norm_eps=1e-5, vocab_size=100, head_dim=dim//n_heads, rope_theta=10000,
    rope_dim=dim//n_heads, v_head_dim=dim//n_heads, max_context=16,
    num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)

def _q8_expert_layer(num_experts=4, dim=32):
  raw = np.empty((num_experts, dim, 34), dtype=np.uint8)
  raw[:, :, :2] = np.frombuffer(np.float16(1).tobytes(), dtype=np.uint8)
  for expert in range(num_experts): raw[expert, :, 2:] = expert + 1
  packed = Tensor(raw.flatten()).realize()
  layer = ExpertWeights(num_experts, dim, dim)
  layer.set_quantized(ggml_data_to_tensor(packed, num_experts * dim * dim, 8).reshape(num_experts, dim, dim), packed, 8)
  return layer

class TestMoEFeedForward(unittest.TestCase):
  def test_moe_feed_forward(self):
    dim, hidden, n_heads = 8, 16, 2
    num_experts, k = 4, 2

    block = TransformerBlock(_moe_config(dim, hidden, n_heads, num_experts, k))

    # set up weights: gate scales by (expert_id+1), up/down are identity-ish, router picks experts 0,2
    block.ffn_gate_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) * (i + 1) for i in range(num_experts)])
    block.ffn_up_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) for _ in range(num_experts)])
    block.ffn_down_exps.weight = Tensor.stack(*[Tensor.eye(dim, hidden) for _ in range(num_experts)])
    block.ffn_gate_inp.weight = Tensor([[1, 0, 1, 0]] * dim).T  # router strongly prefers experts 0 and 2
    block.ffn_norm.weight = Tensor.ones(dim)  # identity norm

    # input of ones -> after norm still ~ones -> experts 0,2 selected -> weighted sum of silu outputs
    h = Tensor.ones(1, 1, dim)
    out = block._feed_forward(block.ffn_norm(h))

    # expected moe_output ≈ avg(silu(1), silu(3))
    expected = (Tensor([1.0]).silu().item() + Tensor([3.0]).silu().item()) / 2
    np.testing.assert_allclose(out.numpy()[0, 0, 0], expected, rtol=1e-2)

  def test_quantized_expert_selection(self):
    dim = 32
    out = _q8_expert_layer()(Tensor([[[1, 3]]]), Tensor.ones(1, 1, 1, dim)).numpy()
    expected = np.array([2, 4], dtype=np.float32).reshape(1, 1, 2, 1).repeat(dim, axis=-1) * dim
    np.testing.assert_allclose(out, expected)

  def test_quantized_expert_selection_symbolic_prefill(self):
    dim, layer = 32, _q8_expert_layer()
    sel_np = np.array([[[1, 3], [0, 2]]], dtype=np.int32)
    toks = UOp.variable("toks", 1, 4).bind(2)
    with patch("tinygrad.llm.model.ggml_data_to_tensor", wraps=ggml_data_to_tensor) as dequant:
      out = layer(Tensor(sel_np)[:, :toks], Tensor.ones(1, 2, 1, dim)[:, :toks])[:, :2].numpy()
    self.assertEqual(dequant.call_args.args[1], layer.num_experts * dim * dim)
    expected = (sel_np + 1)[..., None].repeat(dim, axis=-1) * dim
    np.testing.assert_allclose(out, expected)

  def test_attach_quantized_experts_skips_unsupported_blocks(self):
    expert = ExpertWeights(2, 32, 32)
    blocks = [SimpleNamespace(ffn_gate_exps=expert)]
    weight, packed = Tensor.zeros(2, 32, 32), Tensor.zeros(2 * 32 * 34, dtype=dtypes.uint8)
    state_dict = {"blk.0.ffn_gate_exps.weight":weight, "blk.1.ffn_gate_exps.weight":weight, "blk.0.fake_exps.weight":weight}
    with patch("tinygrad.llm.model.get_ggml_quantization", return_value=(packed, 8)):
      packed_weights = _attach_quantized_experts(blocks, state_dict)
    self.assertEqual(packed_weights, {"blk.0.ffn_gate_exps.weight"})
    self.assertIs(state_dict["blk.0.ffn_gate_exps.weight"], expert.weight)
    self.assertIs(state_dict["blk.1.ffn_gate_exps.weight"], weight)
    self.assertIs(state_dict["blk.0.fake_exps.weight"], weight)

  def test_iq3_xxs_expert_selection_ops(self):
    num_experts, dim = 64, 256
    packed = Tensor.zeros(num_experts * dim * dim // 256 * 98, dtype=dtypes.uint8).contiguous().realize()
    layer = ExpertWeights(num_experts, dim, dim)
    layer.set_quantized(ggml_data_to_tensor(packed, num_experts * dim * dim, 18).reshape(num_experts, dim, dim), packed, 18)
    sel, x = Tensor([[[0, 63]]]).realize(), Tensor.ones(1, 1, 1, dim).realize()

    GlobalCounters.reset()
    out = layer(sel, x).realize()
    self.assertLess(GlobalCounters.global_ops, 5_000_000)
    np.testing.assert_equal(out.numpy(), 0)

  def test_moe_feed_forward_batched(self):
    dim, hidden, n_heads = 8, 16, 2
    num_experts, k = 4, 2

    block = TransformerBlock(_moe_config(dim, hidden, n_heads, num_experts, k))

    # same setup as BS=1 test
    block.ffn_gate_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) * (i + 1) for i in range(num_experts)])
    block.ffn_up_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) for _ in range(num_experts)])
    block.ffn_down_exps.weight = Tensor.stack(*[Tensor.eye(dim, hidden) for _ in range(num_experts)])
    block.ffn_gate_inp.weight = Tensor([[1, 0, 1, 0]] * dim).T
    block.ffn_norm.weight = Tensor.ones(dim)

    # test with BS=2, T=3
    h = Tensor.ones(2, 3, dim)
    out = block._feed_forward(block.ffn_norm(h))

    # all outputs should match the BS=1 expected value
    expected = (Tensor([1.0]).silu().item() + Tensor([3.0]).silu().item()) / 2
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-2)

  def test_moe_feed_forward_norm_topk_prob(self):
    dim, hidden, n_heads = 8, 16, 2
    num_experts, k = 4, 2

    block = TransformerBlock(replace(_moe_config(dim, hidden, n_heads, num_experts, k), norm_topk_prob=True))

    block.ffn_gate_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) * (i + 1) for i in range(num_experts)])
    block.ffn_up_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) for _ in range(num_experts)])
    block.ffn_down_exps.weight = Tensor.stack(*[Tensor.eye(dim, hidden) for _ in range(num_experts)])
    block.ffn_gate_inp.weight = Tensor([[0.1, 0, 0.1, 0]] * dim).T  # equal top-2 experts, but only ~69% mass before renorm
    block.ffn_norm.weight = Tensor.ones(dim)

    h = Tensor.ones(1, 1, dim)
    out = block._feed_forward(block.ffn_norm(h))

    expected = (Tensor([1.0]).silu().item() + Tensor([3.0]).silu().item()) / 2
    np.testing.assert_allclose(out.numpy()[0, 0, 0], expected, rtol=1e-2)

  def test_moe_feed_forward_shared_expert(self):
    dim, hidden, n_heads = 8, 16, 2
    num_experts, k = 4, 2

    block = TransformerBlock(replace(_moe_config(dim, hidden, n_heads, num_experts, k), shared_expert_dim=dim))

    block.ffn_gate_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) * (i + 1) for i in range(num_experts)])
    block.ffn_up_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) for _ in range(num_experts)])
    block.ffn_down_exps.weight = Tensor.stack(*[Tensor.eye(dim, hidden) for _ in range(num_experts)])
    block.ffn_gate_inp.weight = Tensor([[1, 0, 1, 0]] * dim).T
    block.ffn_gate_shexp.weight = Tensor.eye(dim) * 2
    block.ffn_up_shexp.weight = Tensor.eye(dim)
    block.ffn_down_shexp.weight = Tensor.eye(dim)
    block.ffn_gate_inp_shexp["weight"] = Tensor.zeros(dim)
    block.ffn_norm.weight = Tensor.ones(dim)

    h = Tensor.ones(1, 1, dim)
    out = block._feed_forward(block.ffn_norm(h))

    moe_expected = (Tensor([1.0]).silu().item() + Tensor([3.0]).silu().item()) / 2
    shared_expected = Tensor([2.0]).silu().item() * 0.5
    expected = moe_expected + shared_expected
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-2)

  def test_moe_feed_forward_gating_funcs(self):
    dim, hidden, n_heads = 8, 16, 2
    num_experts, k = 4, 2
    logits = np.array([4.0, 3.0, 0.0, -1.0], dtype=np.float32)
    def softmax(x):
      probs = np.exp(x - x.max())
      return probs / probs.sum()
    for gating_func in ExpertGating:
      for norm_topk_prob in (False, True):
        block = TransformerBlock(replace(_moe_config(dim, hidden, n_heads, num_experts, k),
                                         expert_gating_func=gating_func, norm_topk_prob=norm_topk_prob))
        block.ffn_gate_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) for _ in range(num_experts)])
        block.ffn_up_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) * (i + 1) for i in range(num_experts)])
        block.ffn_down_exps.weight = Tensor.stack(*[Tensor.eye(dim, hidden) for _ in range(num_experts)])
        block.ffn_gate_inp.weight = Tensor((logits / dim)[None, :].repeat(dim, 0).T)
        out = block._feed_forward(Tensor.ones(1, 1, dim)).numpy()[0, 0, 0]

        if gating_func == ExpertGating.SOFTMAX: selection_scores = softmax(logits)
        elif gating_func == ExpertGating.SIGMOID: selection_scores = 1 / (1 + np.exp(-logits))
        elif gating_func == ExpertGating.SOFTMAX_WEIGHT: selection_scores = logits
        else: selection_scores = np.sqrt(np.logaddexp(0, logits))
        sel = np.argsort(selection_scores)[-k:]
        weights = softmax(logits[sel]) if gating_func == ExpertGating.SOFTMAX_WEIGHT else selection_scores[sel]
        if norm_topk_prob: weights /= weights.sum()
        expected = (weights * (sel + 1)).sum() / (1 + np.exp(-1))
        np.testing.assert_allclose(out, expected, rtol=1e-3)

if __name__ == '__main__':
  unittest.main()
