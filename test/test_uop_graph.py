import unittest
from tinygrad import dtypes, Variable
from tinygrad.dtype import PtrDType
from tinygrad.ops import BinaryOps, TernaryOps, UnaryOps
from tinygrad.codegen.uops import UOpGraph, UOps, UOp

class TestUOpGraph(unittest.TestCase):
  def test_add_constant_fold(self):
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    out = UOp(UOps.ALU, dtypes.float, (c1, c2), BinaryOps.ADD)
    g = UOpGraph([out])
    self.assertEqual(len(g.uops), 1)
    out = g.uops[-1]
    self.assertEqual(out.op, UOps.CONST)
    self.assertEqual(out.arg, 3.0)

  def test_where_same_fold(self):
    v = UOp(UOps.DEFINE_VAR, dtypes.int, arg=Variable('tmp', 0, 1))
    c0 = UOp(UOps.CONST, dtypes.int, arg=0)
    vc = UOp(UOps.ALU, dtypes.bool, (v, c0), BinaryOps.CMPNE)
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    out = UOp(UOps.ALU, dtypes.float, (vc, c1, c1), TernaryOps.WHERE)
    g = UOpGraph([out])
    self.assertEqual(len(g.uops), 1)
    out = g.uops[-1]
    self.assertEqual(out.op, UOps.CONST)
    self.assertEqual(out.arg, 1.0)

  def test_where_const_fold(self):
    bf = UOp(UOps.CONST, dtypes.bool, arg=False)
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    out = UOp(UOps.ALU, dtypes.float, (bf, c1, c2), TernaryOps.WHERE)
    g = UOpGraph([out])
    self.assertEqual(len(g.uops), 1)
    out = g.uops[-1]
    self.assertEqual(out.op, UOps.CONST)
    self.assertEqual(out.arg, 2.0)

  def test_const_cast(self):
    bf = UOp(UOps.CONST, dtypes.bool, arg=False)
    out = UOp(UOps.CAST, dtypes.int, (bf,))
    g = UOpGraph([out])
    self.assertEqual(len(g.uops), 1)
    out = g.uops[-1]
    self.assertEqual(out.op, UOps.CONST)
    self.assertEqual(out.arg, 0)

  def test_cast_vectorized_fold(self):
    d0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=(0, True))
    idx = UOp(UOps.CONST, dtypes.int, arg=0)
    ld = UOp(UOps.LOAD, dtypes.float.vec(2), (d0, idx))
    cast = UOp(UOps.CAST, dtypes.float.vec(2), (ld,))
    x = UOp(UOps.GEP, dtypes.float, (cast, ), arg=0)
    alu = UOp(UOps.ALU, dtypes.float, (x, ), UnaryOps.SQRT)
    out = UOp(UOps.STORE, dtypes.float, (d0, idx, alu))
    g = UOpGraph([out])
    self.assertEqual(len([x for x in g.uops if x.op is UOps.CAST]), 0)

  def test_depth_2_const_fold(self):
    v = UOp(UOps.DEFINE_VAR, dtypes.int, arg=Variable('tmp', 0, 1))
    c2 = UOp(UOps.CONST, dtypes.int, arg=2)
    c4 = UOp(UOps.CONST, dtypes.int, arg=4)
    vc = UOp(UOps.ALU, dtypes.int, (v, c2), BinaryOps.ADD)
    out = UOp(UOps.ALU, dtypes.int, (vc, c4), BinaryOps.ADD)
    g = UOpGraph([out])
    self.assertEqual(len(g.uops), 3)
    out = g.uops[-1]
    self.assertEqual(out.op, UOps.ALU)
    self.assertEqual(out.arg, BinaryOps.ADD)
    self.assertEqual(out.vin[1].op, UOps.CONST)
    self.assertEqual(out.vin[1].arg, 6)

if __name__ == '__main__':
  unittest.main(verbosity=2)
