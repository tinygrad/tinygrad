import unittest

from tinygrad import UOp, dtypes
from tinygrad.codegen.late.linearizer import linearize
from tinygrad.uop.ops import Ops

class TestLinearizerOrder(unittest.TestCase):
  def test_incomparable_args(self):
    # Equal-priority UOps can carry heterogeneous args. Their ordering must not rely on Python comparing those args.
    sink = UOp.sink(UOp(Ops.NOOP, dtypes.void, arg=None), UOp(Ops.NOOP, dtypes.void, arg=(1,)))
    self.assertEqual(set(linearize(sink)), set(sink.toposort()))

if __name__ == "__main__":
  unittest.main()
