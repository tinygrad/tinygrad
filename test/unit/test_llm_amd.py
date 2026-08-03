import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.amd import q8_quantize
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.model import Linear

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
    offset = linear.set_quantized(decoded)
    assert offset is not None
    linear._raw_offset_uop = offset.realize().uop
    self.assertTrue(np.isfinite(linear(Tensor.randn(1, 256)).realize().item()))

if __name__ == "__main__": unittest.main()
