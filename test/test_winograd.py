import unittest
from tinygrad.helpers import Timing
from tinygrad.tensor import Tensor
from tinygrad.ops import LoadOps
from tinygrad.codegen.linearizer import Linearizer

class TestWinograd(unittest.TestCase):
  def setUp(self):
    self.old = Tensor.wino
    Tensor.wino = 1
  def tearDown(self): Tensor.wino = self.old

  def test_speed(self):
    x = Tensor.empty(1,4,9,9)
    w = Tensor.empty(4,4,3,3)

    with Timing("running conv: "):
      out = Tensor.conv2d(x, w)

    with Timing("scheduling: "):
      sched = out.lazydata.schedule()

    for i,s in enumerate(sched):
      if s[0].op in LoadOps: continue
      with Timing(f"linearize {i}: "):
        l = Linearizer(s[0])
        l.hand_coded_optimizations()
        l.linearize()

if __name__ == '__main__':
  unittest.main(verbosity=2)