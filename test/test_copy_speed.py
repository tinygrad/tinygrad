import unittest
from tinygrad import Tensor
from tinygrad.ops import Device
from tinygrad.helpers import Timing, CI

N = 4096 if CI else 16384
class TestCopySpeed(unittest.TestCase):
  @classmethod
  def setUpClass(cls): Device[Device.DEFAULT].synchronize()

  def testCopySHMtoDefault(self):
    t = Tensor.empty(N, N, device="disk:/dev/shm/test_X").realize()
    #t = Tensor.empty(N, N, device="disk:shm:test_X").realize()
    for _ in range(3):
      with Timing("sync:  ", on_exit=lambda ns: f" @ {t.nbytes()/ns:.2f} GB/s"):
        with Timing("queue: "):
          t.to(Device.DEFAULT).realize()
        Device[Device.DEFAULT].synchronize()

  def testCopyCPUtoDefault(self):
    t = Tensor.rand(N, N, device="cpu").realize()
    print(f"buffer: {t.nbytes()*1e-9:.2f} GB")
    for _ in range(3):
      with Timing("sync:  ", on_exit=lambda ns: f" @ {t.nbytes()/ns:.2f} GB/s"):
        with Timing("queue: "):
          t.to(Device.DEFAULT).realize()
        Device[Device.DEFAULT].synchronize()

  def testCopyCPUtoDefaultFresh(self):
    print("fresh copy")
    for _ in range(3):
      t = Tensor.rand(N, N, device="cpu").realize()
      with Timing("sync:  ", on_exit=lambda ns: f" @ {t.nbytes()/ns:.2f} GB/s"):
        with Timing("queue: "):
          t.to(Device.DEFAULT).realize()
        Device[Device.DEFAULT].synchronize()
      del t

  def testCopyDefaulttoCPU(self):
    t = Tensor.rand(N, N).realize()
    print(f"buffer: {t.nbytes()*1e-9:.2f} GB")
    for _ in range(3):
      with Timing("sync:  ", on_exit=lambda ns: f" @ {t.nbytes()/ns:.2f} GB/s"):
        t.to('cpu').realize()

  @unittest.skipIf(CI, "CI doesn't have 6 GPUs")
  def testCopyCPUto6GPUs(self):
    t = Tensor.rand(N, N, device="cpu").realize()
    print(f"buffer: {t.nbytes()*1e-9:.2f} GB")
    for _ in range(3):
      with Timing("sync:  ", on_exit=lambda ns: f" @ {t.nbytes()/ns:.2f} GB/s ({t.nbytes()*6/ns:.2f} GB/s total)"):
        with Timing("queue: "):
          for g in range(6):
            t.to(f"gpu:{g}").realize()
        Device[f"gpu"].synchronize()

if __name__ == '__main__':
  unittest.main()