import unittest
from tinygrad import Tensor, Device
from tinygrad.engine.realize import get_program
from tinygrad.codegen.opt import Opt, OptOps
from test.external.process_replay.process_replay import replay_get_program

N = 16
class TestProcessReplay(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.ast = (Tensor.empty(N, N) @ Tensor.empty(N, N)).schedule()[-1].ast
    cls.renderer = Device[Device.DEFAULT].renderer

  def test_replay_no_opts(self):
    # opts=None means use default heuristic path
    p = get_program(self.ast, self.renderer)
    good, compare, _ = replay_get_program(p, self.ast, self.renderer)
    self.assertEqual(good, compare)

  def test_replay_empty_opts(self):
    # opts=[] means explicitly apply zero opts (unoptimized)
    p = get_program(self.ast, self.renderer, opts=[])
    good, compare, _ = replay_get_program(p, self.ast, self.renderer, opts=[])
    self.assertEqual(good, compare)

  def test_replay_with_opt(self):
    # opts=[Opt(...)] means apply a specific opt
    opts = [Opt(OptOps.UPCAST, 0, 4)]
    p = get_program(self.ast, self.renderer, opts=opts)
    good, compare, _ = replay_get_program(p, self.ast, self.renderer, opts=opts)
    self.assertEqual(good, compare)

if __name__ == '__main__':
  unittest.main(verbosity=2)
