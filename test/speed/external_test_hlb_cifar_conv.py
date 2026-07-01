import unittest
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import Context


class TestHLBCifarConv(unittest.TestCase):
  @unittest.skipUnless(dtypes.half in Device[Device.DEFAULT].renderer.supported_dtypes(), "need half support")
  def test_hlb_cifar_backward_conv_8x8(self):
    Tensor.manual_seed(0)
    dtypes.default_float = dtypes.half
    x = Tensor.randn(512, 256, 8, 8).realize()
    w = Tensor.randn(512, 256, 3, 3).realize()
    with Context(TRAINING=1):
      y = x.conv2d(w, padding=1).sum()
      y.backward()
      Tensor.realize(x.grad, w.grad)

  @unittest.skipUnless(dtypes.half in Device[Device.DEFAULT].renderer.supported_dtypes(), "need half support")
  def test_hlb_cifar_backward_conv_7x7(self):
    Tensor.manual_seed(0)
    dtypes.default_float = dtypes.half
    x = Tensor.randn(512, 256, 7, 7).realize()
    w = Tensor.randn(512, 256, 3, 3).realize()
    with Context(TRAINING=1):
      y = x.conv2d(w, padding=1).sum()
      y.backward()
      Tensor.realize(x.grad, w.grad)


if __name__ == "__main__":
  unittest.main()
