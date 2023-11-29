import unittest
import numpy as np
from tinygrad import Device, dtypes, Tensor
from tinygrad.helpers import ImageDType

@unittest.skipIf(Device.DEFAULT != "GPU", "only images on GPU")
class TestImageDType(unittest.TestCase):
  def test_shrink_load_float(self):
    it = Tensor.randn(4).cast(dtypes.imagef((1,1,4))).realize()
    imgv = it.numpy()
    np.testing.assert_equal(imgv[0:2], it[0:2].numpy())

  def test_mul_stays_image(self):
    it = Tensor.randn(4).cast(dtypes.imagef((1,1,4))).realize()
    out = (it*2).realize()
    assert isinstance(out.lazydata.realized.dtype, ImageDType)

if __name__ == '__main__':
  unittest.main()
