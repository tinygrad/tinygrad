import unittest
import numpy as np
from tinygrad import Device, dtypes, Tensor
from tinygrad.helpers import ImageDType

@unittest.skipIf(Device.DEFAULT != "GPU", "only images on GPU")
class TestImageDType(unittest.TestCase):
  def test_image_and_back(self):
    data = Tensor.randn(9*27*4).realize()
    tst = data.numpy()
    it = data.cast(dtypes.imagef((9,27,4))).realize()
    assert isinstance(it.lazydata.realized.dtype, ImageDType)
    np.testing.assert_equal(tst, it.numpy())

  def test_image_and_back_wrong_shape(self):
    data = Tensor.randn(9*27*4).realize()
    tst = data.numpy()
    it = data.cast(dtypes.imagef((9,12,4))).realize()
    assert not isinstance(it.lazydata.realized.dtype, ImageDType)
    np.testing.assert_equal(tst, it.numpy())

  def test_shrink_load_float(self):
    it = Tensor.randn(4).cast(dtypes.imagef((1,1,4))).realize()
    imgv = it.numpy()
    np.testing.assert_equal(imgv[0:2], it[0:2].numpy())

  def test_mul_stays_image(self):
    it = Tensor.randn(4).cast(dtypes.imagef((1,1,4))).realize()
    out = (it*2).realize()
    assert isinstance(out.lazydata.realized.dtype, ImageDType)

  def test_shrink_max(self):
    it = Tensor.randn(8).cast(dtypes.imagef((1,2,4))).realize()
    imgv = it.numpy()
    np.testing.assert_equal(np.maximum(imgv[0:3], 0), it[0:3].relu().numpy())

  def test_shrink_to_float(self):
    it = Tensor.randn(4, 4).cast(dtypes.imagef((1,4,4))).realize()
    imgv = it.numpy()
    np.testing.assert_equal(np.maximum(imgv[:, 0], 0), it[:, 0].relu().realize())

if __name__ == '__main__':
  unittest.main()
