import unittest
import numpy as np
import torch
from tinygrad import Tensor, Context, dtypes
from tinygrad.device import Device, is_dtype_supported

class TestNumericalAccuracy(unittest.TestCase):
  # TODO: blocker for true onnx fp16
  @unittest.skipIf(not is_dtype_supported(dtypes.half) or Device.DEFAULT == "CLANG", "non-clang backends fail")
  @unittest.expectedFailure
  def test_conv2d_fp16_default(self):
    x,w,b = np.random.randn(1, 12, 128, 128), np.random.randn(32, 12, 3, 3), np.random.randn(32)
    tiny_x, tiny_w, tiny_b = (Tensor(t, dtype=dtypes.half) for t in (x,w,b))
    torch_x, torch_w, torch_b = (torch.tensor(t, dtype=torch.half) for t in (x,w,b))
    tinygrad_out = Tensor.conv2d(tiny_x, tiny_w, tiny_b)
    torch_out = torch.nn.functional.conv2d(torch_x, torch_w, torch_b)
    np.testing.assert_allclose(tinygrad_out.numpy(), torch_out.numpy(), rtol=5e-3, atol=5e-3)

  # TODO: blocker for true onnx fp16
  @unittest.skipIf(not is_dtype_supported(dtypes.half) or Device.DEFAULT == "CLANG", "non-clang backends fail")
  @unittest.expectedFailure
  def test_conv2d_fp16_noopt(self):
    with Context(NOOPT=1):
      self.test_conv2d_fp16_default()

  @unittest.skipIf(not is_dtype_supported(dtypes.half))
  def test_conv2d_fp16_default_vs_noopt(self):
    x,w,b = Tensor.randn(1, 12, 128, 128, dtype=dtypes.half), Tensor.randn(32, 12, 3, 3, dtype=dtypes.half), Tensor.randn(32, dtype=dtypes.half)
    default = Tensor.conv2d(x,w,b).numpy()
    with Context(NOOPT=1):
      noopt = Tensor.conv2d(x,w,b).numpy()
    np.testing.assert_allclose(default, noopt, rtol=2e-3, atol=2e-3)

if __name__ == "__main__":
  unittest.main()
