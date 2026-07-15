import unittest
from tinygrad.helpers import Timing
from tinygrad import Tensor, Device
import numpy as np

class TestDevCopySpeeds(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.sz = 768
    cls.dev = Device["AMD"]
    if not cls.dev.is_usb(): raise unittest.SkipTest("only test this on USB devices")

  def testCopyCPUtoDefault(self):
    for _ in range(10):
      t = Tensor.ones(self.sz, self.sz, device="CPU").contiguous().realize()
      with Timing(f"copyin of {t.nbytes()/1e6:.2f} MB:  ", on_exit=lambda ns: f" @ {t.nbytes()/ns * 1e3:.2f} MB/s"): # noqa: F821
        t.to(Device.DEFAULT).realize()
        Device[Device.DEFAULT].synchronize()
      del t

  def testCopyDefaulttoCPU(self):
    t = Tensor.ones(self.sz, self.sz).contiguous().realize()
    for _ in range(10):
      with Timing(f"copyout of {t.nbytes()/1e6:.2f} MB:  ", on_exit=lambda ns: f" @ {t.nbytes()/ns * 1e3:.2f} MB/s"):
        t.to('CPU').realize()

  def testValidateCopies(self):
    t = Tensor.randn(self.sz, self.sz, device="CPU").contiguous().realize()
    x = t.to(Device.DEFAULT).realize()
    Device[Device.DEFAULT].synchronize()

    y = x.to('CPU').realize()

    np.testing.assert_equal(t.numpy(), y.numpy())
    del x, y, t

if __name__ == "__main__":
  unittest.main()
