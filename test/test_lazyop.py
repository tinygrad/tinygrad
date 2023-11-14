import unittest
from tinygrad.tensor import Tensor

# stuff needed to unpack a kernel
from tinygrad.ops import LazyOp, TernaryOps, BinaryOps, UnaryOps, ReduceOps, BufferOps, MemBuffer, ConstBuffer
from tinygrad.lazy import LazyBuffer
from tinygrad.helpers import dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.shape.symbolic import Variable
import numpy as np
import time
inf, nan = float('inf'), float('nan')

class TestLazyOp(unittest.TestCase):
  def test_lazyop_str(self):
    t = Tensor.rand(10) + Tensor.rand(10)
    s = t.lazydata.schedule()
    ast = s[-1].ast
    ast_remade = eval(str(ast))
    self.assertEqual(ast, ast_remade)

  def test_selfreferential_speed(self):
    x = LazyBuffer.fromCPU(np.array([1]))
    st = time.perf_counter_ns()
    for i in range(30):
      p = x
      for n in range(i):
        p = p.e(BinaryOps.ADD, p)
      if time.perf_counter_ns()-st > 1e6 or i > 13:
        assert i > 13

if __name__ == '__main__':
  unittest.main()
