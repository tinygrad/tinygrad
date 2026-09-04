import unittest
from tinygrad.helpers import Timing, getenv
from tinygrad import Tensor, Device
import numpy as np

class TestDevCopySpeeds(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.sz = getenv("SIZE", 2000000)
    cls.dev = Device["AMD"]
    if not cls.dev.is_usb(): raise unittest.SkipTest("only test this on USB devices")

  def testCopyCPUtoDefault(self):
    for _ in range(10):
      t = Tensor.ones(self.sz, device="CPU", dtype='uchar').contiguous().realize()
      with Timing(f"copyin of {t.nbytes()/1e6:.2f} MB:  ", on_exit=lambda ns: f" @ {t.nbytes()/ns * 1e3:.2f} MB/s"): # noqa: F821
        t.to(Device.DEFAULT).realize()
        Device[Device.DEFAULT].synchronize()
      del t

  def testCopyDefaulttoCPU(self):
    t = Tensor.ones(self.sz, dtype='uchar').contiguous().realize()
    for _ in range(10):
      with Timing(f"copyout of {t.nbytes()/1e6:.2f} MB:  ", on_exit=lambda ns: f" @ {t.nbytes()/ns * 1e3:.2f} MB/s"):
        t.to('CPU').realize()

  def testValidateCopies(self):
    t = Tensor.randn(self.sz, device="CPU", dtype='uchar').contiguous().realize()
    x = t.to(Device.DEFAULT).realize()
    Device[Device.DEFAULT].synchronize()

    y = x.to('CPU').realize()

    np.testing.assert_equal(t.numpy(), y.numpy())
    del x, y, t

  def testCopyinRingWrap(self):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, 1 << 20, dtype=np.uint8)
    np.testing.assert_array_equal(a, Tensor(a, device="AMD").numpy())
    ring = self.dev.sdma_queue(0)
    for _ in range(3):
      # Leave enough NOP padding to delay SDMA until both SRAM windows have received USB data.
      target = ring.ring.nbytes - 0xe700
      padding = target - ring.put_value % ring.ring.nbytes - 16  # four-dword timeline fence
      self.assertGreaterEqual(padding, 0)
      q = self.dev.hw_copy_queue_t()
      q.q(*([0] * (padding // 4)))
      q.signal(self.dev.timeline_signal, self.dev.next_timeline()).submit(self.dev)
      self.dev.synchronize()
      a = rng.integers(0, 256, 256 << 20, dtype=np.uint8)
      np.testing.assert_array_equal(a, Tensor(a, device="AMD").numpy())

if __name__ == "__main__":
  unittest.main()
