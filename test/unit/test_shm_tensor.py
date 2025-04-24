import unittest, sys
import multiprocessing.shared_memory as shared_memory
from tinygrad.tensor import Tensor, Device
import numpy as np

def validate_memlock_limit(t: Tensor):
  if sys.platform != "win32":
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
    if rlimit < t.nbytes(): raise unittest.SkipTest(f"This test requires RLIMIT_MEMLOCK of at least {t.nbytes()//1024} KiB.")

class TestRawShmBuffer(unittest.TestCase):
  def test_e2e(self):
    t = Tensor.randn(2, 2, 2).realize()
    validate_memlock_limit(t)

    # copy to shm
    shm_name = (s := shared_memory.SharedMemory(create=True, size=t.nbytes())).name
    s.close()
    t_shm = t.to(f"disk:shm:{shm_name}").realize()

    # copy from shm
    t2 = t_shm.to(Device.DEFAULT).realize()

    assert np.allclose(t.numpy(), t2.numpy())
    s.unlink()

  def test_e2e_big(self):
    t = Tensor.randn(2048, 2048, 8).realize()
    validate_memlock_limit(t)

    # copy to shm
    shm_name = (s := shared_memory.SharedMemory(create=True, size=t.nbytes())).name
    s.close()
    t_shm = t.to(f"disk:shm:{shm_name}").realize()

    # copy from shm
    t2 = t_shm.to(Device.DEFAULT).realize()

    assert np.allclose(t.numpy(), t2.numpy())
    s.unlink()

if __name__ == "__main__":
  unittest.main()
