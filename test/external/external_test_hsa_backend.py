import os
import unittest

# import subprocess
from tinygrad.tensor import Tensor, dtypes
import numpy as np


class TestHSABackend(unittest.TestCase):
  def test_float16_to_bfloat16_conversion(self):
    original_tensor = Tensor([1.0, 2.0, 3.0], dtype=dtypes.float16)
    converted_tensor = original_tensor.cast(dtypes.bfloat16)
    self.assertEqual(converted_tensor.dtype, dtypes.bfloat16)

    back_to_float32 = converted_tensor.cast(dtypes.float32)
    original_to_float32 = original_tensor.cast(dtypes.float32)
    np.testing.assert_allclose(back_to_float32.numpy(), original_to_float32.numpy(), rtol=1e-2, atol=1e-3)

  def test_float16_to_bfloat16_edge_cases(self):
    edge_cases = Tensor([0.0, -0.0, float('inf'), float('-inf'), float('nan')], dtype=dtypes.float16)
    converted = edge_cases.cast(dtypes.bfloat16).cast(dtypes.float32)
    np.testing.assert_equal(converted.numpy(), edge_cases.cast(dtypes.float32).numpy())

  def test_float16_to_bfloat16_range_precision(self):
    large_value = Tensor([65504.0], dtype=dtypes.float16)  # Max representable in float16
    small_value = Tensor([6.1035e-5], dtype=dtypes.float16)  # Smallest positive normal float16

    large_converted = large_value.cast(dtypes.bfloat16).cast(dtypes.float32)
    small_converted = small_value.cast(dtypes.bfloat16).cast(dtypes.float32)
    print(large_converted.numpy(), large_value.cast(dtypes.float32).numpy())
    print(small_converted.numpy(), small_value.cast(dtypes.float32).numpy())

    np.testing.assert_allclose(large_converted.numpy(), large_value.cast(dtypes.float32).numpy(), rtol=1e-2, atol=1e-3)
    np.testing.assert_equal(small_converted.numpy(), small_value.cast(dtypes.float32).numpy())

  def test_float16_to_bfloat16_randomized(self):
    np.random.seed(42)  # For reproducibility
    random_values = Tensor(np.random.uniform(-65504, 65504, 1000), dtype=dtypes.float16)
    converted = random_values.cast(dtypes.bfloat16).cast(dtypes.float32)
    np.testing.assert_allclose(converted.numpy(), random_values.cast(dtypes.float32).numpy(), rtol=1e-2, atol=1e-3)


if __name__ == "__main__":
  unittest.main()
