import math
import unittest

from tinygrad import Context, Device, Tensor, nn
from tinygrad.nn.state import get_state_dict

from examples.mlperf.models.llama2_lora import (
  LORA_TARGETS, LoRALinear, ToyLlama2LoRA, adapter_parameters, adapter_state_dict, backward_adapters, shifted_causal_loss,
)


IS_NULL = Device.DEFAULT.startswith("NULL")


@unittest.skipIf(IS_NULL, "NULL has no values to compare")
class TestLoRALinear(unittest.TestCase):
  def setUp(self): Tensor.manual_seed(7)

  def test_zero_b_matches_frozen_base(self):
    layer = LoRALinear(3, 4, rank=2, alpha=4, dropout=0.0, bias=True)
    x = Tensor([[1.0, -2.0, 0.5], [0.25, 1.0, -1.0]])
    self.assertTrue(layer(x).allclose(layer.base(x), atol=0.0, rtol=0.0).item())

  def test_rank_alpha_scale(self):
    layer = LoRALinear(3, 2, rank=2, alpha=4, dropout=0.0, bias=False)
    layer.weight.assign(Tensor.zeros_like(layer.weight))
    layer.lora_A.assign(Tensor.ones_like(layer.lora_A))
    layer.lora_B.assign(Tensor.ones_like(layer.lora_B))
    got = layer(Tensor([[1.0, 2.0, 3.0]]))
    self.assertTrue(got.allclose(Tensor([[24.0, 24.0]]), atol=1e-6, rtol=0.0).item())

  def test_dropout_eval_and_seeded_training(self):
    layer = LoRALinear(8, 3, rank=3, alpha=3, dropout=0.5, bias=False)
    layer.lora_B.assign(Tensor.ones_like(layer.lora_B))
    x = Tensor.ones(2, 8)
    with Context(TRAINING=0): eval_a, eval_b = layer(x).realize(), layer(x).realize()
    self.assertTrue(eval_a.allclose(eval_b, atol=0.0, rtol=0.0).item())
    Tensor.manual_seed(123)
    with Context(TRAINING=1): train_a = layer(x).realize()
    Tensor.manual_seed(123)
    with Context(TRAINING=1): train_b = layer(x).realize()
    self.assertTrue(train_a.allclose(train_b, atol=0.0, rtol=0.0).item())
    self.assertFalse(train_a.allclose(eval_a, atol=0.0, rtol=0.0).item())

  def test_dropout_one_rejected(self):
    with self.assertRaises(ValueError): LoRALinear(3, 2, dropout=1.0)


@unittest.skipIf(IS_NULL, "NULL has no values to compare")
class TestTrainingSemantics(unittest.TestCase):
  def setUp(self): Tensor.manual_seed(11)

  def test_shifted_loss_ignores_minus_100(self):
    logits = Tensor([[[0.0, 2.0, -1.0], [8.0, -4.0, 1.0], [1.0, 0.0, 3.0], [0.0, 0.0, 0.0]]])
    labels = Tensor([[0, 1, -100, 2]])
    got = shifted_causal_loss(logits, labels)
    expected = Tensor.stack(logits[0, 0], logits[0, 2]).sparse_categorical_crossentropy(Tensor([1, 2]))
    self.assertTrue(got.allclose(expected, atol=1e-6, rtol=1e-6).item())

  def test_shifted_loss_custom_ignore_index(self):
    logits = Tensor([[[0.0, 2.0, -1.0], [8.0, -4.0, 1.0], [1.0, 0.0, 3.0], [0.0, 0.0, 0.0]]])
    labels = Tensor([[0, 1, 7, 2]])
    got = shifted_causal_loss(logits, labels, ignore_index=7)
    expected = Tensor.stack(logits[0, 0], logits[0, 2]).sparse_categorical_crossentropy(Tensor([1, 2]))
    self.assertTrue(got.allclose(expected, atol=1e-6, rtol=1e-6).item())

  def test_all_ignored_loss_and_adapter_gradients_are_finite_zero(self):
    model = ToyLlama2LoRA(vocab_size=9, dim=8, hidden_dim=12, n_heads=2, rank=2, alpha=4, dropout=0.0)
    loss = model.loss(Tensor([[1, 2, 3]]), Tensor([[4, 7, 7]]), ignore_index=7)
    backward_adapters(loss, model)
    self.assertTrue(math.isfinite(loss.item()))
    self.assertEqual(loss.item(), 0.0)
    for param in adapter_parameters(model):
      self.assertIsNotNone(param.grad)
      self.assertTrue(math.isfinite(param.grad.abs().max().item()))
      self.assertEqual(param.grad.abs().max().item(), 0.0)

  def test_only_adapters_receive_gradients_and_update(self):
    layer = LoRALinear(3, 2, rank=2, alpha=2, dropout=0.0, bias=True)
    base_before = layer.weight.detach().clone().realize()
    adapter_before = layer.lora_B.detach().clone().realize()
    optimizer = nn.optim.SGD(adapter_parameters(layer), lr=0.1)
    with Context(TRAINING=1):
      optimizer.zero_grad()
      backward_adapters(layer(Tensor([[1.0, -2.0, 3.0]])).sum(), layer)
      self.assertIsNone(layer.weight.grad)
      self.assertIsNone(layer.bias.grad)
      self.assertTrue(all(param.grad is not None for param in adapter_parameters(layer)))
      optimizer.step()
    self.assertTrue(layer.weight.allclose(base_before, atol=0.0, rtol=0.0).item())
    self.assertFalse(layer.lora_B.allclose(adapter_before, atol=0.0, rtol=0.0).item())

  def test_backward_adapters_accumulates(self):
    layer = LoRALinear(3, 2, rank=2, alpha=2, dropout=0.0)
    loss = layer(Tensor([[1.0, -2.0, 3.0]])).sum()
    backward_adapters(loss, layer)
    first = [param.grad.detach().clone().realize() for param in adapter_parameters(layer)]
    backward_adapters(loss, layer)
    for param, initial_grad in zip(adapter_parameters(layer), first):
      self.assertTrue(param.grad.allclose(initial_grad * 2, atol=1e-6, rtol=1e-6).item())

  def test_tiny_forward_loss_update(self):
    model = ToyLlama2LoRA(vocab_size=13, dim=8, hidden_dim=16, n_heads=2, n_layers=1, rank=2, alpha=4, dropout=0.0)
    tokens, labels = Tensor([[1, 2, 3, 4]]), Tensor([[-100, 2, 3, 4]])
    optimizer = nn.optim.SGD(adapter_parameters(model), lr=0.05)
    lora_b_before = [param.detach().clone().realize() for name, param in adapter_state_dict(model).items() if name.endswith("lora_B")]
    with Context(TRAINING=1):
      optimizer.zero_grad()
      logits = model(tokens)
      loss = shifted_causal_loss(logits, labels)
      backward_adapters(loss, model)
      self.assertTrue(math.isfinite(loss.item()))
      self.assertTrue(all(math.isfinite(param.grad.abs().max().item()) for param in adapter_parameters(model)))
      optimizer.step()
    self.assertEqual(logits.shape, (1, 4, 13))
    self.assertEqual(loss.shape, ())
    lora_b_after = [param for name, param in adapter_state_dict(model).items() if name.endswith("lora_B")]
    self.assertTrue(any(not after.allclose(before, atol=0.0, rtol=0.0).item() for before, after in zip(lora_b_before, lora_b_after)))

  def test_adapters_affect_logits(self):
    model = ToyLlama2LoRA(vocab_size=9, dim=8, hidden_dim=12, n_heads=2, rank=2, alpha=4, dropout=0.0)
    tokens = Tensor([[1, 2, 3]])
    with Context(TRAINING=0): baseline = model(tokens).realize()
    model.layers[0].o_proj.lora_B.assign(Tensor.ones_like(model.layers[0].o_proj.lora_B))
    with Context(TRAINING=0): adapted = model(tokens).realize()
    self.assertFalse(adapted.allclose(baseline, atol=1e-7, rtol=0.0).item())

  def test_future_tokens_do_not_affect_prior_logits(self):
    model = ToyLlama2LoRA(vocab_size=9, dim=8, hidden_dim=12, n_heads=2, rank=2, alpha=4, dropout=0.0)
    with Context(TRAINING=0):
      first = model(Tensor([[1, 2, 3, 4]])).realize()
      changed_future = model(Tensor([[1, 2, 8, 7]])).realize()
    self.assertTrue(first[:, :2].allclose(changed_future[:, :2], atol=1e-6, rtol=1e-6).item())


class TestModelContract(unittest.TestCase):
  def test_parameter_contract(self):
    layer = LoRALinear(3, 2, rank=2, bias=True)
    self.assertFalse(layer.weight.is_param)
    self.assertFalse(layer.bias.is_param)
    self.assertTrue(layer.lora_A.is_param)
    self.assertTrue(layer.lora_B.is_param)

  def test_head_divisibility_validation(self):
    with self.assertRaises(ValueError): ToyLlama2LoRA(vocab_size=9, dim=7, hidden_dim=12, n_heads=2)

  def test_target_whitelist(self):
    model = ToyLlama2LoRA(vocab_size=9, dim=8, hidden_dim=12, n_heads=2, n_layers=2, rank=2, alpha=4)
    names = tuple(adapter_state_dict(model))
    self.assertEqual(LORA_TARGETS, ("qkv_proj", "o_proj"))
    self.assertTrue(all(name.split(".")[-2] in LORA_TARGETS for name in names))
    self.assertFalse(any("gate_proj" in name or "up_proj" in name or "down_proj" in name for name in names))

  def test_stable_adapter_state_names(self):
    model = ToyLlama2LoRA(vocab_size=9, dim=8, hidden_dim=12, n_heads=2, n_layers=2, rank=2, alpha=4)
    self.assertEqual(tuple(adapter_state_dict(model)), (
      "layers.0.qkv_proj.lora_A", "layers.0.qkv_proj.lora_B",
      "layers.0.o_proj.lora_A", "layers.0.o_proj.lora_B",
      "layers.1.qkv_proj.lora_A", "layers.1.qkv_proj.lora_B",
      "layers.1.o_proj.lora_A", "layers.1.o_proj.lora_B",
    ))
    state = get_state_dict(model)
    self.assertTrue(all(name in state for name in adapter_state_dict(model)))

  def test_null_portable_graph(self):
    model = ToyLlama2LoRA(vocab_size=9, dim=8, hidden_dim=12, n_heads=2, n_layers=1, rank=2, alpha=4)
    logits = model(Tensor([[1, 2, 3]]))
    loss = shifted_causal_loss(logits, Tensor([[-100, 2, 3]]))
    self.assertEqual(logits.shape, (1, 3, 9))
    self.assertEqual(loss.shape, ())


if __name__ == "__main__": unittest.main()
