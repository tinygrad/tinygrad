import unittest
import time
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.device import InterpretedASTRunner
from tinygrad.lazy import create_schedule
from tinygrad.realize import run_schedule, lower_schedule_item

class TestFusionOp(unittest.TestCase):
  def test_contiguous_add(self):
    def test(contig=False):
      bt = Tensor(np.arange(16), dtype=dtypes.float32).reshape(4,4)
      x = bt.permute(1,0)
      if contig: x = x.contiguous()
      return (x.permute(1,0) + bt).data()
    assert test() == test(True)

  def test_expand_fuse(self):
    bt = Tensor(np.ones((10, 1)), dtype=dtypes.float32)
    out = (bt*2).expand(10,10).sum(1)
    sched = create_schedule([out.lazydata], None)
    run_schedule(sched)
    outd = out.data().tolist()
    assert all(x == 20.0 for x in outd)

  # TODO: fix this test to be fast and remove O(2^n) behavior
  def test_recursive_add(self):
    st = time.perf_counter()
    a = Tensor([1,2,3,4])
    for _ in range(20): a = a + a
    sched = create_schedule([a.lazydata], None)
    ji = lower_schedule_item(sched[-1])
    et = time.perf_counter()
    self.assertLess(et-st, 10.0)
    assert isinstance(ji, InterpretedASTRunner) or len(ji.prg) < 5000

if __name__ == '__main__':
  unittest.main(verbosity=2)
