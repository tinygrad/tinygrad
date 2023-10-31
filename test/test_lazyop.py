import unittest
from tinygrad.tensor import Tensor

# stuff needed to unpack a kernel
from tinygrad.ops import LazyOp, TernaryOps, BinaryOps, UnaryOps, ReduceOps, BufferOps, MemBuffer, ConstBuffer
from tinygrad.helpers import dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.shape.symbolic import Variable
inf, nan = float('inf'), float('nan')

class TestLazyOp(unittest.TestCase):
  def test_lazyop_str(self):
    t = Tensor.rand(10) + Tensor.rand(10)
    s = t.lazydata.schedule()
    ast = s[-1].ast
    ast_remade = eval(str(ast))
    self.assertEqual(ast, ast_remade)

if __name__ == '__main__':
  unittest.main()
