import unittest

from tinygrad.codegen.linearizer import Linearizer
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.features.search import time_linearizer, bufs_from_lin
from tinygrad.device import Compiled, Device, Buffer
from tinygrad.ops import LoadOps
from tinygrad.tensor import Tensor

class TestTimeLinearizer(unittest.TestCase):
  def setUp(self) -> None:
    if not isinstance(Device[Device.DEFAULT], Compiled): raise unittest.SkipTest("only test for compiled backends")

  def test_reasonable_time(self):
    si = [si for si in Tensor([1,2,3,4]).add(1).lazydata.schedule() if si.ast.op not in LoadOps][0]
    rawbufs = [Buffer(Device.DEFAULT, si.out.st.size(), si.out.dtype)] + [Buffer(Device.DEFAULT, x.st.size(), x.dtype) for x in si.inputs]
    tm = time_linearizer(Linearizer(si.ast), rawbufs, allow_test_size=False, cnt=10)
    assert tm > 0 and tm != float('inf')

  def test_bufs_from_lin_device(self):
    si = [si for si in Tensor([1,2,3,4]).add(1).lazydata.schedule() if si.ast.op not in LoadOps][0]
    opts = LinearizerOptions(device=Device.DEFAULT)
    bufs = bufs_from_lin(Linearizer(si.ast), opts.device)
    assert all(b.device == opts.device for b in bufs)

if __name__ == '__main__':
  unittest.main()
