import unittest

from tinygrad.codegen.linearizer import Linearizer
from tinygrad.features.search import time_linearizer
from tinygrad.ops import Compiled, Device, LoadOps
from tinygrad.tensor import Tensor

class TestTimeLinearizer(unittest.TestCase):
  def setUp(self) -> None:
    if not isinstance(Device[Device.DEFAULT], Compiled): raise unittest.SkipTest("only test for compiled backends")

  def test_reasonable_time(self):
    si = [si for si in Tensor([1,2,3,4]).add(1).lazydata.schedule() if si.ast.op not in LoadOps][0]
    rawbufs = [Device[Device.DEFAULT].buffer(si.out.st.size(), si.out.dtype)] + [Device[Device.DEFAULT].buffer(x.st.size(), x.dtype) for x in si.inputs]
    tm = time_linearizer(Linearizer(si.ast), rawbufs, allow_test_size=False, cnt=10)
    assert tm > 0 and tm != float('inf')

if __name__ == '__main__':
  unittest.main()
