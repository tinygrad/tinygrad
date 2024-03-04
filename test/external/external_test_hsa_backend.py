import os
import unittest

# import subprocess
from tinygrad.tensor import Tensor, dtypes
import numpy as np


class TestHSABackend(unittest.TestCase):
  # def test_llama_with_hsa(self):
  #   result = subprocess.run(
  #     [
  #       "python3",
  #       "examples/llama.py",
  #       "--temperature=0",
  #       "--count=50",
  #       '--prompt="Hello."',
  #       "--timing",
  #       "--shard",
  #       "2",
  #       "--size",
  #       "70B",
  #       "--gen",
  #       "2",
  #     ],
  #     capture_output=True,
  #     text=True,
  #   )
  #   print("STDOUT:", result.stdout)  # TODO remove after debugging
  #   print("STDERR:", result.stderr)  # TODO remove after debugging
  #   self.assertEqual(result.returncode, 0)

  #   # Optional: Check output for expected values
  #   # self.assertIn("expected output", result.stdout)

  def test_float16_to_bfloat16_conversion(self):
    # os.environ["HSA"] = "1"

    original_tensor = Tensor([1.0, 2.0, 3.0], dtype=dtypes.float16)
    converted_tensor = original_tensor.cast(dtypes.bfloat16)
    self.assertEqual(converted_tensor.dtype, dtypes.bfloat16)

    back_to_float32 = converted_tensor.cast(dtypes.float32)
    original_to_float32 = original_tensor.cast(dtypes.float32)

    np.testing.assert_allclose(back_to_float32.numpy(), original_to_float32.numpy(), rtol=1e-3, atol=1e-6)


if __name__ == "__main__":
  unittest.main()
