import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.amd import q8_quantize

class TestQ8Quantize(unittest.TestCase):
  def test_values_and_scales(self):
    x = np.linspace(-3.1, 2.7, 64, dtype=np.float32).reshape(2, 32)
    quant, scale = q8_quantize(Tensor(x), 2, 32)
    scale_np = np.maximum(np.max(np.abs(x), axis=-1, keepdims=True) / 127, 1e-8)
    expected = np.clip(np.rint(x / scale_np), -127, 127).astype(np.int8)
    np.testing.assert_array_equal(quant.bitcast(dtypes.int8).reshape(2, 32).numpy(), expected)
    np.testing.assert_allclose(scale.numpy(), scale_np, rtol=1e-6)

if __name__ == "__main__": unittest.main()
