import os
os.environ["PROFILE"] = "1"
os.environ["PMC"] = "1"

import unittest
import functools, contextlib
import numpy as np
from tinygrad import Tensor, Context, Device
from tinygrad.uop.ops import UOp, KernelInfo, AxisType
from tinygrad.runtime.ops_amd import ProfilePMCEvent
from extra.sqtt.roc import print_pmc

def copy_kernel(B, A, stride=1):
  n_threads = 32
  assert A.size >= n_threads, f"{A.size} is too small, min size {n_threads}"
  g = UOp.range(A.size//n_threads, 0, AxisType.GLOBAL)
  l = UOp.range(n_threads, 1, AxisType.LOCAL)
  i = g * n_threads + l
  index = (i * stride) % A.size
  return B[index].store(A[index]).sink(arg=KernelInfo(name=f"copy_{A.size}_stride_{stride}", opts_to_apply=()))

dev = Device[Device.DEFAULT]

@contextlib.contextmanager
def save_pmc():
  # clear the old traces
  dev.profile_events.clear()
  pmc:list[ProfilePMCEvent] = []
  yield pmc
  for e in dev.profile_events:
    if isinstance(e, ProfilePMCEvent): pmc.append(e)

@unittest.skipIf(dev.device != "AMD", "tests PMC counters on AMD")
class TestPMC(unittest.TestCase):
  @Context(IGNORE_OOB=0)
  def test_copy(self, stride:int=1):
    N = 1 << 25 # ~134MB
    a = Tensor(np.arange(N, dtype=np.uint32)+1).realize()
    b = Tensor(np.zeros(N, dtype=np.uint32)).realize()
    b = Tensor.custom_kernel(b, a, fxn=functools.partial(copy_kernel, stride=stride))[0]
    with save_pmc() as pmc:
      b.realize()
    print_pmc(pmc[0])
    np.testing.assert_equal(a.numpy(), b.numpy())

  def test_copy_uncoalesced(self): return self.test_copy(stride=17)

if __name__ == "__main__":
  unittest.main()
