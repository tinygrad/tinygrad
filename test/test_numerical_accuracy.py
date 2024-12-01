import unittest
import numpy as np
import torch
from tinygrad import Tensor, Context, dtypes
from tinygrad.device import Device, is_dtype_supported

class TestNumericalAccuracy(unittest.TestCase):
  # TODO: blocker for true onnx fp16
  @unittest.skipIf(not is_dtype_supported(dtypes.half, Device.DEFAULT) or Device.DEFAULT == "CLANG", "non-cpu backends fail")
  @unittest.expectedFailure
  def test_conv2d_fp16(self):
    x,w,b = np.random.randn(1, 12, 128, 128), np.random.randn(32, 12, 3, 3), np.random.randn(32)
    tx,tw,tb = (Tensor(t, dtype=dtypes.half) for t in (x,w,b))
    tx_,tw_,tb_ = (torch.tensor(t, dtype=torch.half) for t in (x,w,b))
    tinygrad_out = Tensor.conv2d(tx,tw,tb)
    torch_out = torch.nn.functional.conv2d(tx_,tw_,tb_)
    np.testing.assert_allclose(tinygrad_out.numpy(), torch_out.numpy(), rtol=5e-3, atol=5e-3)

  # TODO: blocker for true onnx fp16
  @unittest.skipIf(not is_dtype_supported(dtypes.half, Device.DEFAULT) or Device.DEFAULT == "CLANG", "non-cpu backends fail")
  @unittest.expectedFailure
  def test_conv2d_fp16_noopt(self):
    with Context(NOOPT=1):
      self.test_conv2d_fp16()

if __name__ == "__main__":
  unittest.main()
