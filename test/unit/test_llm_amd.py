import unittest
import numpy as np
from tinygrad import Tensor, UOp, dtypes, nn
from tinygrad.llm.kernels import Linear
from tinygrad.llm.kernels.amd import q8_quantize, quantized_attention
from tinygrad.llm.gguf import ggml_data_to_tensor

class TestQ8Quantize(unittest.TestCase):
  def test_values_and_scales(self):
    if not str(Tensor.empty(1).device).startswith("AMD"): self.skipTest("AMD required")
    x = np.linspace(-3.1, 2.7, 64, dtype=np.float32).reshape(2, 32)
    quant, scale = q8_quantize(Tensor(x), 2, 32)
    scale_np = np.maximum(np.max(np.abs(x), axis=-1, keepdims=True) / 127, 1e-8)
    expected = np.clip(np.rint(x / scale_np), -127, 127).astype(np.int8)
    np.testing.assert_array_equal(quant.bitcast(dtypes.int8).reshape(2, 32).numpy(), expected)
    np.testing.assert_allclose(scale.numpy(), scale_np, rtol=1e-6)

  def test_q6_linear_compiles(self):
    if not str(Tensor.empty(1).device).startswith("AMD"): self.skipTest("AMD required")
    rng = np.random.default_rng(42)
    packed = rng.integers(0, 256, 210, dtype=np.uint8)
    packed[-2:] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, 256, 14).reshape(1, 256)
    linear = Linear(256, 1, bias=False)
    nn.state.load_state_dict(linear, {"weight":decoded}, verbose=False, realize=False)
    self.assertTrue(np.isfinite(linear(Tensor.randn(1, 256)).realize().item()))
    self.assertEqual(linear.weight.uop.buf_uop.buffer.offset, 4)

  def test_q6_linear_multiple_tokens(self):
    if not str(Tensor.empty(1).device).startswith("AMD"): self.skipTest("AMD required")
    rng = np.random.default_rng(42)
    in_features, blocks = 2048, 16*2048//256
    packed = rng.integers(0, 256, blocks*210, dtype=np.uint8)
    for i in range(blocks): packed[i*210+208:i*210+210] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, 16*in_features, 14).reshape(16, in_features)
    weight = decoded.numpy()
    linear = Linear(in_features, 16, bias=False)
    nn.state.load_state_dict(linear, {"weight":decoded}, verbose=False, realize=False)
    x = rng.normal(size=(3, in_features)).astype(np.float32)
    scale = np.maximum(np.abs(x).reshape(3, in_features//32, 32).max(-1, keepdims=True) / 127, 1e-8)
    xq = np.clip(np.rint(x.reshape(3, in_features//32, 32) / scale), -127, 127) * scale
    np.testing.assert_allclose(linear(Tensor(x)).numpy(), xq.reshape(3, in_features) @ weight.T, rtol=2e-3, atol=2e-2)
    self.assertEqual(linear.ggml_type, 14)

    generic = Linear(in_features, 16, bias=False)
    nn.state.load_state_dict(generic, {"weight":decoded}, verbose=False, realize=False)
    generic(Tensor.randn(4, in_features)[:UOp.variable("tokens", 1, 4).bind(2)])
    self.assertFalse(generic.use_custom_quant)
    self.assertIsNone(generic.ggml_type)

  def test_attention_uses_physical_cache_length(self):
    if not str(Tensor.empty(1).device).startswith("AMD"): self.skipTest("AMD required")
    q, k, v = Tensor.zeros(1, 2, 1, 32), Tensor.randn(1, 1, 1, 32), Tensor.randn(1, 1, 1, 32)
    cache = Tensor.empty(2, 1, 1, 256, 32, dtype=dtypes.int8).contiguous()
    scale = Tensor.empty(2, 1, 1, 256, dtype=dtypes.float16).contiguous()
    out = quantized_attention(q, Tensor.stack(k, v), cache, scale, 0).realize()
    np.testing.assert_allclose(out.numpy(), v.expand(1, 2, 1, 32).numpy(), rtol=2e-2, atol=2e-2)

  def test_prefill_attention_unaligned_start(self):
    if not str(Tensor.empty(1).device).startswith("AMD"): self.skipTest("AMD required")
    rng = np.random.default_rng(42)
    start_pos = 1718
    q = Tensor.zeros(1, 8, 32, 128)
    old_kv = rng.normal(size=(2, 1, 1, start_pos, 128)).astype(np.float32)
    new_kv = rng.normal(size=(2, 1, 1, 32, 128)).astype(np.float32)
    cache = Tensor.empty(2, 1, 1, 2048, 128, dtype=dtypes.int8).contiguous()
    scale = Tensor.zeros(2, 1, 1, 2048, dtype=dtypes.float16).contiguous()
    old_scale = np.maximum(np.max(np.abs(old_kv), axis=-1, keepdims=True) / 127, 1e-8).astype(np.float16)
    Tensor.realize(cache[:, :, :, :start_pos].assign(Tensor(np.rint(old_kv / old_scale).astype(np.int8))),
                   scale[:, :, :, :start_pos].assign(Tensor(old_scale.squeeze(-1))))
    out = quantized_attention(q, Tensor(new_kv), cache, scale, UOp.variable("start_pos", 0, 2047).bind(start_pos)).realize()
    values = cache[1, 0, 0, :start_pos+32].numpy().astype(np.float32) * \
      scale[1, 0, 0, :start_pos+32].numpy().astype(np.float32)[:, None]
    expected = np.stack([values[:start_pos+i+1].mean(0) for i in range(32)])[None, None].repeat(8, axis=1)
    np.testing.assert_allclose(out.numpy(), expected, rtol=2e-3, atol=2e-3)

if __name__ == "__main__": unittest.main()
