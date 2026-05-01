import os
os.environ["WQKV"] = "1"
import tempfile, unittest
import numpy as np
from tinygrad import Tensor, nn, dtypes
from tinygrad.nn.state import get_parameters, safe_save
from tinygrad.device import is_dtype_supported, Device
from extra.models.llama import Transformer
from examples.mlperf.models.flat_llama import FlatTransformer, apply_grad
from examples.mlperf.optim import GradAccClipAdamW

def copy_weights(flat:FlatTransformer, ref:Transformer):
  n_layers = flat.n_layers
  Tensor.realize(*nn.state.get_state_dict(ref).values())
  flat.wqkv.assign(Tensor(np.stack([ref.layers[i].attention.wqkv.weight.numpy() for i in range(n_layers)])))
  flat.wo.assign(Tensor(np.stack([ref.layers[i].attention.wo.weight.numpy() for i in range(n_layers)])))
  flat.w1.assign(Tensor(np.stack([ref.layers[i].feed_forward.w1.weight.numpy() for i in range(n_layers)])))
  flat.w2.assign(Tensor(np.stack([ref.layers[i].feed_forward.w2.weight.numpy() for i in range(n_layers)])))
  flat.w3.assign(Tensor(np.stack([ref.layers[i].feed_forward.w3.weight.numpy() for i in range(n_layers)])))
  flat.attention_norm.assign(Tensor(np.stack([ref.layers[i].attention_norm.weight.numpy() for i in range(n_layers)])))
  flat.ffn_norm.assign(Tensor(np.stack([ref.layers[i].ffn_norm.weight.numpy() for i in range(n_layers)])))
  flat.norm.weight.assign(Tensor(ref.norm.weight.numpy()))
  flat.tok_embeddings.weight.assign(Tensor(ref.tok_embeddings.weight.numpy()).cast(flat.tok_embeddings.weight.dtype))
  flat.output.assign(Tensor(ref.output.weight.numpy()[None]).cast(flat.output.dtype))

def split_attention_state(ref:Transformer):
  state = nn.state.get_state_dict(ref)
  split_state = {
    "tok_embeddings.weight": state["tok_embeddings.weight"],
    "norm.weight": state["norm.weight"],
    "output.weight": state["output.weight"],
  }
  for i, layer in enumerate(ref.layers):
    attn = layer.attention
    q_dim = attn.n_heads * attn.head_dim
    kv_dim = attn.n_kv_heads * attn.head_dim
    wq, wk, wv = state[f"layers.{i}.attention.wqkv.weight"].split([q_dim, kv_dim, kv_dim], dim=0)
    split_state[f"layers.{i}.attention.wq.weight"] = wq
    split_state[f"layers.{i}.attention.wk.weight"] = wk
    split_state[f"layers.{i}.attention.wv.weight"] = wv
    split_state[f"layers.{i}.attention.wo.weight"] = state[f"layers.{i}.attention.wo.weight"]
    split_state[f"layers.{i}.feed_forward.w1.weight"] = state[f"layers.{i}.feed_forward.w1.weight"]
    split_state[f"layers.{i}.feed_forward.w2.weight"] = state[f"layers.{i}.feed_forward.w2.weight"]
    split_state[f"layers.{i}.feed_forward.w3.weight"] = state[f"layers.{i}.feed_forward.w3.weight"]
    split_state[f"layers.{i}.attention_norm.weight"] = state[f"layers.{i}.attention_norm.weight"]
    split_state[f"layers.{i}.ffn_norm.weight"] = state[f"layers.{i}.ffn_norm.weight"]
  return split_state

class TestFlatLlama(unittest.TestCase):
  def test_forward_match(self):
    Tensor.manual_seed(42)
    params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)
    ref = Transformer(**params, disable_kv_cache=True)
    flat = FlatTransformer(**params)
    copy_weights(flat, ref)
    Tensor.realize(*nn.state.get_state_dict(flat).values())

    tokens = Tensor([[1, 50, 100, 999, 2]])
    ref_logits = ref(tokens, 0, temperature=float("nan")).realize()
    flat_logits = flat(tokens).realize()
    self.assertEqual(ref_logits.shape, flat_logits.shape)
    diff = (ref_logits - flat_logits).abs().max().item()
    self.assertLess(diff, 1e-2, f"forward mismatch: max abs diff {diff}")

  def test_backward_match(self):
    Tensor.manual_seed(42)
    params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)
    ref = Transformer(**params, disable_kv_cache=True)
    flat = FlatTransformer(**params)
    copy_weights(flat, ref)

    for p in get_parameters(ref): p.requires_grad_(True)
    for p in get_parameters(flat): p.requires_grad_(True)
    Tensor.realize(*nn.state.get_state_dict(flat).values())

    tokens = Tensor([[1, 50, 100, 999, 2, 10]])

    ref_loss = ref(tokens[:, :-1], 0, temperature=float("nan")).sparse_categorical_crossentropy(tokens[:, 1:])
    ref_loss.backward()
    ref_grads = {k: v.grad.numpy() for k, v in nn.state.get_state_dict(ref).items() if v.grad is not None}

    flat_loss = flat(tokens[:, :-1]).sparse_categorical_crossentropy(tokens[:, 1:])
    flat_loss.backward()
    flat_grads = {k: v.grad.numpy() for k, v in nn.state.get_state_dict(flat).items() if v.grad is not None}

    # check loss matches
    self.assertAlmostEqual(ref_loss.item(), flat_loss.item(), places=3)

    # check output weight grad matches
    diff = abs(ref_grads["output.weight"] - flat_grads["output"][0]).max()
    self.assertLess(diff, 5e-3, f"output.weight grad mismatch: max abs diff {diff}")

    # check per-layer weight grads match
    for i in range(params["n_layers"]):
      for flat_key, ref_key in [
        ("wqkv", f"layers.{i}.attention.wqkv.weight"),
        ("wo", f"layers.{i}.attention.wo.weight"),
        ("w1", f"layers.{i}.feed_forward.w1.weight"),
        ("w2", f"layers.{i}.feed_forward.w2.weight"),
        ("w3", f"layers.{i}.feed_forward.w3.weight"),
      ]:
        diff = abs(ref_grads[ref_key] - flat_grads[flat_key][i]).max()
        self.assertLess(diff, 5e-3, f"layer {i} {flat_key} grad mismatch: max abs diff {diff}")

  def test_lora_zero_init_match(self):
    Tensor.manual_seed(42)
    params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)
    ref = Transformer(**params, disable_kv_cache=True)
    flat = FlatTransformer(**params)
    flat_lora = FlatTransformer(**params, lora_rank=8, lora_alpha=16, lora_dropout=0.0)
    copy_weights(flat, ref)
    copy_weights(flat_lora, ref)
    Tensor.realize(*nn.state.get_state_dict(flat).values(), *nn.state.get_state_dict(flat_lora).values())

    tokens = Tensor([[1, 50, 100, 999, 2]])
    flat_logits = flat(tokens).realize()
    flat_lora_logits = flat_lora(tokens).realize()
    self.assertEqual(flat_logits.shape, flat_lora_logits.shape)
    diff = (flat_logits - flat_lora_logits).abs().max().item()
    self.assertLess(diff, 1e-5, f"lora zero-init mismatch: max abs diff {diff}")

  def test_lora_adapter_only_grads(self):
    Tensor.manual_seed(42)
    params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)
    ref = Transformer(**params, disable_kv_cache=True)
    flat = FlatTransformer(**params, lora_rank=8, lora_alpha=16, lora_dropout=0.0)
    copy_weights(flat, ref)

    state = nn.state.get_state_dict(flat)
    for name, tensor in state.items(): tensor.requires_grad_("lora" in name)
    Tensor.realize(*state.values())

    tokens = Tensor([[1, 50, 100, 999, 2, 10]])
    with Tensor.train():
      loss = flat(tokens[:, :-1]).sparse_categorical_crossentropy(tokens[:, 1:])
      loss.backward()

    grad_names = {k for k, v in state.items() if v.grad is not None}
    self.assertSetEqual(grad_names, {"wqkv_lora_a", "wqkv_lora_b", "wo_lora_a", "wo_lora_b"})
    self.assertGreater(state["wqkv_lora_b"].grad.abs().max().item(), 0.0)
    self.assertGreater(state["wo_lora_b"].grad.abs().max().item(), 0.0)

  def test_lora_adapter_state_dict(self):
    params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)
    flat = FlatTransformer(**params, lora_rank=8, lora_alpha=16, lora_dropout=0.0)
    adapter_state = flat.adapter_state_dict()
    self.assertSetEqual(set(adapter_state), {"wqkv_lora_a", "wqkv_lora_b", "wo_lora_a", "wo_lora_b"})
    self.assertListEqual(flat.adapter_parameters(), [
      adapter_state["wqkv_lora_a"], adapter_state["wqkv_lora_b"], adapter_state["wo_lora_a"], adapter_state["wo_lora_b"],
    ])

  def test_adapter_state_dict_empty_without_lora(self):
    params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)
    flat = FlatTransformer(**params)
    self.assertEqual(flat.adapter_state_dict(), {})
    self.assertEqual(flat.adapter_parameters(), [])

  @unittest.skipUnless(Device.DEFAULT in {"CPU", "NULL"}, "multi-device graph test")
  def test_mp_lora_apply_grad_multi_layer(self):
    Tensor.manual_seed(42)
    devices = (f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1")
    params = dict(dim=32, hidden_dim=64, n_heads=8, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=64, rope_theta=10000, max_context=8)
    flat = FlatTransformer(**params, lora_rank=2, lora_alpha=4, lora_dropout=0.0, base_quantize="int8")
    flat.shard(devices, mp=True)

    optim = GradAccClipAdamW(flat.adapter_parameters(), lr=0.001, grad_acc=1, clip_norm=0.3, fused=False)
    grads = []
    for p in optim.params:
      p.grad = p.zeros_like().contiguous().realize()
      grads.append(p.grad)
    self.assertEqual([p.uop.axis for p in optim.params], [g.uop.axis for g in grads])
    tokens = Tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=dtypes.int32).shard(devices)
    with Tensor.train():
      loss = flat(tokens[:, :-1]).sparse_categorical_crossentropy(tokens[:, 1:])
      for param, grad_buf, new_grad in zip(optim.params, grads, loss.gradient(*optim.params)):
        apply_grad(grad_buf, new_grad.uop)
        self.assertEqual(param.uop.axis, grad_buf.uop.axis)
      Tensor.realize(*grads)
      optim.fstep(grads)

  def test_load_from_state_dict(self):
    Tensor.manual_seed(42)
    params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)
    ref = Transformer(**params, disable_kv_cache=True)
    flat = FlatTransformer(**params)
    flat.load_from_state_dict(split_attention_state(ref))
    Tensor.realize(*nn.state.get_state_dict(flat).values())

    tokens = Tensor([[1, 50, 100, 999, 2]])
    ref_logits = ref(tokens, 0, temperature=float("nan")).realize()
    flat_logits = flat(tokens).realize()
    np.testing.assert_allclose(flat_logits.numpy(), ref_logits.numpy(), atol=1e-2, rtol=1e-2)

  def test_load_from_pretrained_safetensors(self):
    Tensor.manual_seed(42)
    params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)
    ref = Transformer(**params, disable_kv_cache=True)
    flat = FlatTransformer(**params)

    with tempfile.TemporaryDirectory() as tmpdir:
      fn = os.path.join(tmpdir, "model.safetensors")
      safe_save(split_attention_state(ref), fn)
      flat.load_from_pretrained(fn)

    tokens = Tensor([[1, 50, 100, 999, 2]])
    ref_logits = ref(tokens, 0, temperature=float("nan")).realize()
    flat_logits = flat(tokens).realize()
    np.testing.assert_allclose(flat_logits.numpy(), ref_logits.numpy(), atol=1e-2, rtol=1e-2)

  def test_load_from_pretrained_safetensors_int8_base(self):
    Tensor.manual_seed(42)
    params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)
    ref = Transformer(**params, disable_kv_cache=True)
    flat = FlatTransformer(**params, base_quantize="int8")

    with tempfile.TemporaryDirectory() as tmpdir:
      fn = os.path.join(tmpdir, "model.safetensors")
      safe_save(split_attention_state(ref), fn)
      flat.load_from_pretrained(fn)

    self.assertEqual(flat.wqkv.dtype, dtypes.int8)
    self.assertEqual(flat.wo.dtype, dtypes.int8)
    self.assertEqual(flat.w1.dtype, dtypes.int8)
    self.assertEqual(flat.w2.dtype, dtypes.int8)
    self.assertEqual(flat.w3.dtype, dtypes.int8)

    tokens = Tensor([[1, 50, 100, 999, 2]])
    ref_logits = ref(tokens, 0, temperature=float("nan")).realize()
    flat_logits = flat(tokens).realize()
    np.testing.assert_allclose(flat_logits.numpy(), ref_logits.numpy(), atol=3e-1, rtol=2e-1)

  @unittest.skipUnless(Device.DEFAULT == "CPU", "multi-device CPU test")
  def test_load_from_pretrained_mp_int8_base_pads_vocab(self):
    Tensor.manual_seed(42)
    ref_params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1000, rope_theta=10000, max_context=64)
    flat_params = ref_params | {"vocab_size": 1024}
    devices = (f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1")
    ref = Transformer(**ref_params, disable_kv_cache=True)
    flat = FlatTransformer(**flat_params, base_quantize="int8")
    flat.shard(devices, mp=True)

    with tempfile.TemporaryDirectory() as tmpdir:
      fn = os.path.join(tmpdir, "model.safetensors")
      safe_save(split_attention_state(ref), fn)
      flat.load_from_pretrained(fn)

    tokens = Tensor([[1, 50, 100, 999, 2]], device=devices[0])
    ref_logits = ref(tokens.to(devices[0]), 0, temperature=float("nan")).numpy()
    flat_logits = flat(tokens.shard(devices)).numpy()
    self.assertEqual(flat_logits.shape[-1], 1024)
    np.testing.assert_allclose(flat_logits[..., :1000], ref_logits, atol=3e-1, rtol=2e-1)

  @unittest.skipUnless(Device.DEFAULT == "CPU", "multi-device CPU test")
  def test_forward_match_mp(self):
    Tensor.manual_seed(42)
    params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)
    from tinygrad import Device
    devices = (f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1")
    ref = Transformer(**params)
    flat = FlatTransformer(**params)
    copy_weights(flat, ref)
    Tensor.realize(*nn.state.get_state_dict(flat).values())
    flat.shard(devices, mp=True)

    tokens = Tensor([[1, 50, 100, 999, 2]], device=devices[0])
    ref_logits = ref(tokens.to(devices[0]), 0, temperature=float("nan")).numpy()
    flat_logits = flat(tokens.shard(devices)).numpy()
    self.assertEqual(ref_logits.shape, flat_logits.shape)
    np.testing.assert_allclose(flat_logits, ref_logits, atol=1e-2, rtol=1e-2)

  @unittest.skipUnless(Device.DEFAULT == "CPU", "multi-device CPU test")
  def test_forward_match_dp(self):
    Tensor.manual_seed(42)
    params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)
    from tinygrad import Device
    devices = (f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1")
    ref = Transformer(**params)
    flat = FlatTransformer(**params)
    copy_weights(flat, ref)
    Tensor.realize(*nn.state.get_state_dict(flat).values())
    flat.shard(devices)

    tokens = Tensor([[1, 50, 100, 999, 2], [2, 100, 50, 1, 999]], device=devices[0])
    ref_logits = ref(tokens.to(devices[0]), 0, temperature=float("nan")).numpy()
    flat_logits = flat(tokens.shard(devices, axis=0)).numpy()
    self.assertEqual(ref_logits.shape, flat_logits.shape)
    np.testing.assert_allclose(flat_logits, ref_logits, atol=1e-2, rtol=1e-2)

  @unittest.skipUnless(is_dtype_supported(dtypes.fp8e4m3), "fp8 not supported on this device")
  def test_forward_fp8(self):
    import examples.mlperf.models.flat_llama as flat_llama_mod
    old_fp8 = flat_llama_mod.FP8
    try:
      flat_llama_mod.FP8 = 1
      Tensor.manual_seed(42)
      params = dict(dim=128, hidden_dim=256, n_heads=4, n_kv_heads=2, n_layers=2, norm_eps=1e-5, vocab_size=1024, rope_theta=10000, max_context=64)
      ref = Transformer(**params)
      flat = FlatTransformer(**params)
      copy_weights(flat, ref)
      Tensor.realize(*nn.state.get_state_dict(flat).values())

      tokens = Tensor([[1, 50, 100, 999, 2]])
      ref_logits = ref(tokens, 0, temperature=float("nan")).numpy()
      flat_logits = flat(tokens).numpy()
      self.assertEqual(ref_logits.shape, flat_logits.shape)
      # FP8 has lower precision, allow larger tolerance
      np.testing.assert_allclose(flat_logits, ref_logits, atol=1.0, rtol=0.1)
    finally:
      flat_llama_mod.FP8 = old_fp8

if __name__ == "__main__":
  unittest.main()
