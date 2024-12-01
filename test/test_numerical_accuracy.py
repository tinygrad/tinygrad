import unittest
import numpy as np
import torch
from tinygrad import Tensor, Context, dtypes
from tinygrad.device import Device, is_dtype_supported

class TestNumericalAccuracy(unittest.TestCase):
  # TODO: blocker for true onnx fp16
  @unittest.skipUnless(is_dtype_supported(dtypes.half) and Device.DEFAULT != "CLANG", "non-clang backends fail")
  @unittest.expectedFailure
  def test_conv2d_fp16_default(self):
    x,w,b = np.random.randn(1,12,128,128), np.random.randn(32,12,3,3), np.random.randn(32)
    tiny_x, tiny_w, tiny_b = (Tensor(t, dtype=dtypes.half) for t in (x,w,b))
    torch_x, torch_w, torch_b = (torch.tensor(t, dtype=torch.half) for t in (x,w,b))
    tinygrad_out = Tensor.conv2d(tiny_x, tiny_w, tiny_b)
    torch_out = torch.nn.functional.conv2d(torch_x, torch_w, torch_b)
    np.testing.assert_allclose(tinygrad_out.numpy(), torch_out.numpy(), rtol=5e-3, atol=5e-3)

  @unittest.skipUnless(is_dtype_supported(dtypes.half) and Device.DEFAULT != "CLANG", "non-clang backends fail")
  @unittest.expectedFailure
  def test_conv2d_fp16_noopt(self):
    with Context(NOOPT=1):
      self.test_conv2d_fp16_default()

  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_conv2d_fp16_default_vs_noopt(self):
    x,w,b = Tensor.randn(1,12,128,128, dtype=dtypes.half), Tensor.randn(32,12,3,3, dtype=dtypes.half), Tensor.randn(32, dtype=dtypes.half)
    default = Tensor.conv2d(x,w,b).numpy()
    with Context(NOOPT=1):
      noopt = Tensor.conv2d(x,w,b).numpy()
    np.testing.assert_allclose(default, noopt, rtol=2e-3, atol=2e-3)

  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_conv2d_fp16_vs_fp32(self):
    x,w,b = Tensor.randn(1,12,128,128), Tensor.randn(32,12,3,3), Tensor.randn(32)
    fp32_out = Tensor.conv2d(x.float(), w.float(), b.float())
    fp16_out = Tensor.conv2d(x.half(), w.half(), b.half())
    np.testing.assert_allclose(fp32_out.numpy(), fp16_out.numpy(), rtol=2e-2, atol=2e-2)

if __name__ == "__main__":
  np.random.seed(0)
  unittest.main()
