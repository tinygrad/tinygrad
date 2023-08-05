import unittest, math
import numpy as np
from tinygrad.helpers import dtypes
from tinygrad.tensor import Device
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, ASTRunner
from tinygrad.codegen.linearizer import UOps, Token, ConstOp, MemOp
from tinygrad.shape.symbolic import Variable

def _test_single_value(tc, tt, vals, op):
  uops = [
    [UOps.DEFINE_GLOBAL, None, [], ('data0', tc.dtype)],
    *[[UOps.LOAD, ta, [], ConstOp(a, Variable.ands([]))] for ta,a in zip(tt, vals)],
    [UOps.ALU, tc, tt, op],
    [UOps.STORE, None, [tc], MemOp('data0', Variable.num(0), False, tc.dtype, Variable.ands([]))]
  ]
  src, global_size, local_size = Device[Device.DEFAULT].renderer("test", uops)
  buf = Device[Device.DEFAULT].buffer(1, tc.dtype)
  #print(src)
  prg = ASTRunner("test", src, global_size, local_size).build(Device[Device.DEFAULT].runtime)
  prg([buf])
  return buf.toCPU()[0]

def _test_unaryops(bop, a, dt=dtypes.float32):
  return _test_single_value(Token('c', dt), [Token('a', dt)], [a], bop)
def _test_binaryops(bop, a, b, dt=dtypes.float32):
  return _test_single_value(Token('c', dt), [Token('a', dt), Token('b', dt)], [a,b], bop)
def _test_ternaryops(bop, a, b, c, dt=dtypes.float32):
  return _test_single_value(Token('d', dt), [Token('a', dt), Token('b', dt), Token('c', dt)], [a,b,c], bop)

class TestUOps(unittest.TestCase):
  def _test_uop_fxn(self, bop, fxn):
    for a in [-2.0, 2.0]:
      v1, v2 = _test_unaryops(bop,a), fxn(a)
      if not (math.isnan(v1) and math.isnan(v2)): self.assertAlmostEqual(v1, v2)
  def test_exp2(self): self._test_uop_fxn(UnaryOps.EXP2, lambda a: np.exp2(a))
  def test_log2(self): self._test_uop_fxn(UnaryOps.LOG2, lambda a: math.log2(a) if a > 0 else float('nan'))
  def test_sin(self): self._test_uop_fxn(UnaryOps.SIN, lambda a: math.sin(a))
  def test_sqrt(self): self._test_uop_fxn(UnaryOps.SQRT, lambda a: math.sqrt(a) if a >= 0 else float('nan'))
  def test_recip(self): self._test_uop_fxn(UnaryOps.RECIP, lambda a: 1.0/a)

  def _test_bop_fxn(self, bop, fxn):
    for a in [-2.0, 2.0]:
      for b in [-3.0, 3.0]:
        self.assertAlmostEqual(_test_binaryops(bop,a,b), fxn(a,b))
  def test_add(self): self._test_bop_fxn(BinaryOps.ADD, lambda a,b: a+b)
  def test_sub(self): self._test_bop_fxn(BinaryOps.SUB, lambda a,b: a-b)
  def test_mul(self): self._test_bop_fxn(BinaryOps.MUL, lambda a,b: a*b)
  def test_div(self): self._test_bop_fxn(BinaryOps.DIV, lambda a,b: a/b)
  def test_cmpeq(self): self._test_bop_fxn(BinaryOps.CMPEQ, lambda a,b: float(a==b))
  def test_max(self): self._test_bop_fxn(BinaryOps.MAX, lambda a,b: max(a,b))

  def _test_top_fxn(self, bop, fxn):
    for a in [-2.0, 0, 1, 2.0]:
      for b in [-3.0, 3.0]:
        for c in [-4.0, 4.0]:
          self.assertAlmostEqual(_test_ternaryops(bop,a,b,c), fxn(a,b,c))
  def test_mulacc(self): self._test_top_fxn(TernaryOps.MULACC, lambda a,b,c: (a*b)+c)
  def test_where(self): self._test_top_fxn(TernaryOps.WHERE, lambda a,b,c: b if a!=0 else c)

if __name__ == '__main__':
  unittest.main(verbosity=2)

