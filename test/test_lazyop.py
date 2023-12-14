from typing import Any
import unittest
from hypothesis import assume, given, strategies as st
from tinygrad.tensor import Tensor

# stuff needed to unpack a kernel
# ruff: noqa: F401
from tinygrad.ops import LazyOp, TernaryOps, BinaryOps, UnaryOps, ReduceOps, BufferOps, MemBuffer, ConstBuffer, get_lazyop_info
from tinygrad.lazy import LazyBuffer
from tinygrad.helpers import DTYPES_DICT, dtypes
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
    st = time.monotonic()
    for i in range(25):
      p = LazyBuffer.fromCPU(np.array([1]))
      for _ in range(i): p = p.e(BinaryOps.ADD, p)
      # sanity check if caching works this should be way faster
      assert time.monotonic() -st < 0.5, f"{i}"

class TestMULACC(unittest.TestCase):
  floats = [dtype for dtype in DTYPES_DICT.values() if dtypes.is_float(dtype)]
  @staticmethod
  def _get_reduce(out): return [op for op in out.lazydata.schedule()[-1].ast.get_lazyops() if op.op == ReduceOps.SUM][0]
  @staticmethod
  def _get_cast(out): return [op for op in out.lazydata.schedule()[-1].ast.get_lazyops() if op.op == UnaryOps.CAST][0]

  def test_basic_mulacc_fusion(self):
    x = Tensor.rand(1024,1024)
    w = Tensor.rand(1024,1024)
    out = (x*w).sum(-1)
    reduceop = self._get_reduce(out)
    assert reduceop.src[0].op == TernaryOps.MULACC and len(reduceop.src[0].src) == 2

  def test_mulacc_fusion_mid_ast(self):
    x = Tensor.rand(1024,1024)
    w = Tensor.rand(1024,1024)
    out = ((x.add(4).mul(2))*w).sum(-1).add(4).mul(2)
    reduceop = self._get_reduce(out)
    assert reduceop.src[0].op == TernaryOps.MULACC and len(reduceop.src[0].src) == 2

  @given(st.sampled_from(floats), st.sampled_from(floats))
  def test_mulacc_fusion_midcast(self, d1, d2):
    assume(d1 != d2)
    x = Tensor.rand(1024,1024, dtype=d1)
    w = Tensor.rand(1024,1024, dtype=d1)
    out = (x*w).cast(d2).sum(-1)
    reduceop = self._get_reduce(out)
    castop = self._get_cast(out)
    assert reduceop.src[0].op == TernaryOps.MULACC
    assert castop.src[0].op == ReduceOps.SUM
    assert castop.arg[0] == d2

if __name__ == '__main__':
  unittest.main()
