import unittest

from tinygrad.codegen.linearizer import Linearizer
from tinygrad.realize import create_schedule
from tinygrad.features.search import time_linearizer
from tinygrad.device import Compiled, Device, Buffer
from tinygrad.ops import LoadOps
from tinygrad.tensor import Tensor

class TestTimeLinearizer(unittest.TestCase):
  def setUp(self) -> None:
    if not isinstance(Device[Device.DEFAULT], Compiled): raise unittest.SkipTest("only test for compiled backends")

  def test_reasonable_time(self):
    si = [si for si in create_schedule([Tensor([1,2,3,4]).add(1).lazydata]) if si.ast.op not in LoadOps][0]
    rawbufs = [Buffer(Device.DEFAULT, si.out.st.real_size(), si.out.dtype)] + [Buffer(Device.DEFAULT, x.st.real_size(), x.dtype) for x in si.inputs]
    tm = time_linearizer(Linearizer(si.ast), rawbufs, allow_test_size=False, cnt=10)
    assert tm > 0 and tm != float('inf')

if __name__ == '__main__':
  unittest.main()
