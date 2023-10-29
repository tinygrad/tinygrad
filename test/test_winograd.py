import unittest
from tinygrad.helpers import Timing, CI
from tinygrad.tensor import Tensor
from tinygrad.ops import LoadOps
from tinygrad.codegen.linearizer import Linearizer
from test.test_net_speed import start_profile, stop_profile

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
      if s.ast.op in LoadOps: continue
      ops = s.ast.get_lazyops()
      with Timing(f"linearize {i} with {len(ops):4d} ops: "):
        l = Linearizer(s.ast)
        l.hand_coded_optimizations()
        l.linearize()

  def test_profile(self):
    x,w = Tensor.rand(1,4,9,9).realize(), Tensor.rand(4,4,3,3).realize()
    if not CI: pr = start_profile()
    out = Tensor.conv2d(x,w).realize()
    if not CI: stop_profile(pr, sort='time')
    out.numpy()

if __name__ == '__main__':
  unittest.main(verbosity=2)