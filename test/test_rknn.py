import os
import unittest
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.helpers import getenv
from tinygrad.runtime.ops_rknn import RKNNRuntime


class TestRKNN(unittest.TestCase):
  def test_rknn_select_and_add(self):
    out = (Tensor.ones(16, device="RKNN") + 2).realize().numpy()
    np.testing.assert_allclose(out, np.full((16,), 3.0, dtype=np.float32), atol=0, rtol=0)

  def test_rknn_matmul_fallback_on_non_rk3588(self):
    dev = Device["RKNN"]
    pre_attempts = dev.rknn.stats["matmul_attempts"]
    x = Tensor.rand(8, 8, device="RKNN")
    y = Tensor.rand(8, 8, device="RKNN")
    out = (x @ y).realize().numpy()
    self.assertEqual(out.shape, (8, 8))
    self.assertEqual(dev.rknn.stats["matmul_attempts"], pre_attempts)

  def test_rknn_require_lib_raises(self):
    old = {k: os.environ.get(k) for k in ("RKNN_OFFLOAD", "RKNN_REQUIRE_LIB", "RKNN_LIB")}
    try:
      os.environ["RKNN_OFFLOAD"] = "1"
      os.environ["RKNN_REQUIRE_LIB"] = "1"
      os.environ["RKNN_LIB"] = "/definitely/missing/librknnrt.so"
      getenv.cache_clear()
      with self.assertRaises(RuntimeError):
        RKNNRuntime()
    finally:
      for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
      getenv.cache_clear()


if __name__ == "__main__":
  unittest.main()
