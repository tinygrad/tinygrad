import unittest
import numpy as np
from tinygrad import Tensor, dtypes, nn
from tinygrad.llm.kernels import Linear, cached_attention
from tinygrad.llm.kernels.amd import q8_quantize
from tinygrad.llm.gguf import ggml_data_to_tensor

class TestQ8Quantize(unittest.TestCase):
  def test_values_and_scales(self):
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
    self.assertIsNotNone(linear._raw_offset)

  def test_attention_uses_physical_cache_length(self):
    if not str(Tensor.empty(1).device).startswith("AMD"): self.skipTest("AMD required")
    q, k, v = Tensor.zeros(1, 2, 1, 32), Tensor.randn(1, 1, 1, 32), Tensor.randn(1, 1, 1, 32)
    cache = Tensor.empty(2, 1, 1, 256, 32, dtype=dtypes.int8).contiguous()
    scale = Tensor.empty(2, 1, 1, 256, dtype=dtypes.float16).contiguous()
    out = cached_attention(q, Tensor.stack(k, v), cache, scale, 0).realize()
    np.testing.assert_allclose(out.numpy(), v.expand(1, 2, 1, 32).numpy(), rtol=2e-2, atol=2e-2)

if __name__ == "__main__": unittest.main()
