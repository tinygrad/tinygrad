import unittest
from tinygrad import dtypes, Variable
from tinygrad.ops import BinaryOps, TernaryOps
from tinygrad.codegen.uops import UOpGraph, UOps

class TestUOpGraph(unittest.TestCase):
  def test_add_constant_fold(self):
    g = UOpGraph()
    c1 = g.add(UOps.CONST, dtypes.float, arg=1.0)
    c2 = g.add(UOps.CONST, dtypes.float, arg=2.0)
    out = g.add(UOps.ALU, dtypes.float, (c1, c2), BinaryOps.ADD)
    g.remove_childless({out})
    self.assertEqual(len(g.uops), 1)
    self.assertEqual(out.uop, UOps.CONST)
    self.assertEqual(out.arg, 3.0)

  def test_where_same_fold(self):
    g = UOpGraph()
    v = g.add(UOps.DEFINE_VAR, dtypes.int, arg=Variable('tmp', 0, 1))
    c0 = g.add(UOps.CONST, dtypes.int, arg=0)
    vc = g.add(UOps.ALU, dtypes.bool, (v, c0), BinaryOps.CMPEQ)
    c1 = g.add(UOps.CONST, dtypes.float, arg=1.0)
    out = g.add(UOps.ALU, dtypes.float, (vc, c1, c1), TernaryOps.WHERE)
    g.remove_childless({out})
    self.assertEqual(len(g.uops), 1)
    self.assertEqual(out.uop, UOps.CONST)
    self.assertEqual(out.arg, 1.0)

  def test_where_const_fold(self):
    g = UOpGraph()
    bf = g.add(UOps.CONST, dtypes.bool, arg=False)
    c1 = g.add(UOps.CONST, dtypes.float, arg=1.0)
    c2 = g.add(UOps.CONST, dtypes.float, arg=2.0)
    out = g.add(UOps.ALU, dtypes.float, (bf, c1, c2), TernaryOps.WHERE)
    g.remove_childless({out})
    self.assertEqual(len(g.uops), 1)
    self.assertEqual(out.uop, UOps.CONST)
    self.assertEqual(out.arg, 2.0)

  def test_const_cast(self):
    g = UOpGraph()
    bf = g.add(UOps.CONST, dtypes.bool, arg=False)
    out = g.add(UOps.CAST, dtypes.int, (bf,))
    g.remove_childless({out})
    self.assertEqual(len(g.uops), 1)
    self.assertEqual(out.uop, UOps.CONST)
    self.assertEqual(out.arg, 0)

  def test_insert_before(self):
    g = UOpGraph()
    g.add(UOps.CONST, dtypes.int, arg=0)
    three = g.add(UOps.CONST, dtypes.int, arg=3)
    g.add(UOps.CONST, dtypes.int, arg=1, insert_before=three)
    g.add(UOps.CONST, dtypes.int, arg=2, insert_before=three)
    g.add(UOps.CONST, dtypes.int, arg=4)
    for i,uop in enumerate(g.uops): self.assertEqual(i, uop.arg)

if __name__ == '__main__':
  unittest.main(verbosity=2)
