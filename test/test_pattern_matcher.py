import unittest
from tinygrad.dtype import dtypes
from tinygrad.ops import BinaryOps, UnaryOps
from tinygrad.codegen.uops import UOpGraph, UOps, PatternMatcher, UOp, UPat

class TestPatternMatcher(unittest.TestCase):
  def assert_equiv_uops(self, uop1:UOp, uop2:UOp):
    # NOTE: direct UOps __eq__ is comparing object reference, use this function to compare two uops
    self.assertEqual(uop1.uop, uop2.uop)
    self.assertEqual(uop1.dtype, uop2.dtype)
    self.assertEqual(uop1.arg, uop2.arg)

  def test_simple_match(self):
    matcher = PatternMatcher([(UPat(**{"name": "x", "uop": UOps.CONST, "dtype": dtypes.float}), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.int, arg=1)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_dtype_set(self):
    matcher = PatternMatcher([(UPat(**{"name": "x", "uop": UOps.CONST, "dtype": set([dtypes.float32, dtypes.float64, dtypes.bool])}), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float64, arg=1.0)
    c3 = UOp(UOps.CONST, dtypes.float16, arg=1.0)
    c4 = UOp(UOps.CONST, dtypes.int, arg=1)
    c5 = UOp(UOps.CONST, dtypes.bool, arg=True)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), c2)
    self.assertEqual(matcher.rewrite(c3), None)
    self.assertEqual(matcher.rewrite(c4), None)
    self.assertEqual(matcher.rewrite(c5), c5)

  def test_vin_one(self):
    matcher = PatternMatcher([(UPat(**{"name": "x", "uop": UOps.ALU, "vin":(UPat(**{"uop": UOps.CONST}), UPat(**{"uop": UOps.CONST}))}), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    c3 = UOp(UOps.ALU, dtypes.float, (c1,c2), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_vin_permutations(self):
    matcher = PatternMatcher([(UPat(**{"name": "x", "uop": UOps.ALU, "vin":[UPat(**{"uop": UOps.CONST}), UPat(**{"uop": UOps.ALU})]}), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    c3 = UOp(UOps.ALU, dtypes.float, (c1,c2), BinaryOps.ADD)
    c4 = UOp(UOps.ALU, dtypes.float, (c3,c2), BinaryOps.ADD)
    c5 = UOp(UOps.ALU, dtypes.float, (c2,c3), BinaryOps.ADD)
    c6 = UOp(UOps.ALU, dtypes.float, (c3,c4), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c3), None)
    self.assertEqual(matcher.rewrite(c4), c4)
    self.assertEqual(matcher.rewrite(c5), c5)
    self.assertEqual(matcher.rewrite(c6), None)

  def test_vin_repeat(self):
    matcher = PatternMatcher([(UPat(**{"name": "x", "uop": UOps.ALU, "vin": UPat(**{"uop": UOps.CONST})}), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    c3 = UOp(UOps.ALU, dtypes.float, (c1,c2), BinaryOps.ADD)
    c4 = UOp(UOps.ALU, dtypes.float, (c2,c3), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c4), None)

  def test_constant_arg(self):
    matcher = PatternMatcher([(UPat(**{"name": "x", "uop": UOps.CONST, "arg": 42}), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.int, arg=42)
    c2 = UOp(UOps.CONST, dtypes.int, arg=43)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_zero_add(self):
    matcher = PatternMatcher([(UPat(**{"uop": UOps.ALU, "arg": BinaryOps.ADD, "vin": [UPat(**{"name": "x"}), UPat(**{"uop": UOps.CONST, "arg": 0})]}), lambda x: x)])
    c0 = UOp(UOps.CONST, dtypes.float, arg=0.0)
    x = UOp(UOps.LOAD, dtypes.float, arg='x')
    a1 = UOp(UOps.ALU, dtypes.float, (c0,x), BinaryOps.ADD)
    a2 = UOp(UOps.ALU, dtypes.float, (x,c0), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(a1), x)
    self.assertEqual(matcher.rewrite(a2), x)

  def test_sub_zero(self):
    matcher = PatternMatcher([(UPat(**{"uop": UOps.ALU, "arg": BinaryOps.SUB, "vin": (UPat(**{"name": "x"}), UPat(**{"uop": UOps.CONST, "arg": 0}))}), lambda x: x)])
    c0 = UOp(UOps.CONST, dtypes.float, arg=0.0)
    x = UOp(UOps.LOAD, dtypes.float, arg='x')
    a1 = UOp(UOps.ALU, dtypes.float, (c0,x), BinaryOps.SUB)
    a2 = UOp(UOps.ALU, dtypes.float, (x,c0), BinaryOps.SUB)
    self.assertEqual(matcher.rewrite(a1), None)
    self.assertEqual(matcher.rewrite(a2), x)

  def test_zero_mul(self):
    matcher = PatternMatcher([(UPat(**{"uop": UOps.ALU, "arg": BinaryOps.MUL, "vin": [UPat(**{}), UPat(**{"name": "c", "uop": UOps.CONST, "arg": 0})]}), lambda c: c)])
    c0 = UOp(UOps.CONST, dtypes.float, arg=0.0)
    x = UOp(UOps.LOAD, dtypes.float, arg='x')
    a1 = UOp(UOps.ALU, dtypes.float, (c0,x), BinaryOps.MUL)
    a2 = UOp(UOps.ALU, dtypes.float, (x,c0), BinaryOps.MUL)
    self.assertEqual(matcher.rewrite(a1), c0)
    self.assertEqual(matcher.rewrite(a2), c0)

  def test_self_sub(self):
    matcher = PatternMatcher([(UPat(**{"uop": UOps.ALU, "arg": BinaryOps.SUB, "vin": (UPat(**{"name": "x"}), UPat(**{"name": "x"}))}), lambda x: UOp.const(x.dtype, 0))])   # x-x -> 0
    c0_int = UOp(UOps.CONST, dtypes.int, arg=0)
    c0_float = UOp(UOps.CONST, dtypes.float, arg=0.0)
    c1 = UOp(UOps.CONST, dtypes.int, arg=10)
    c2 = UOp(UOps.CONST, dtypes.float, arg=55.55)
    a1 = UOp(UOps.ALU, dtypes.int, (c1,c1), BinaryOps.SUB)
    a2 = UOp(UOps.ALU, dtypes.float, (c2,c2), BinaryOps.SUB)
    self.assert_equiv_uops(matcher.rewrite(a1), c0_int)
    self.assert_equiv_uops(matcher.rewrite(a2), c0_float)

  def test_fold_neg_mul(self):
    matcher = PatternMatcher([(UPat(**{"uop": UOps.ALU, "arg": BinaryOps.MUL, "vin": [UPat(**{"name": "x"}), UPat(**{"uop": UOps.CONST, "arg": -1})]}), lambda x: -x)])
    c1 = UOp(UOps.LOAD, dtypes.float, arg='xx')
    c2 = UOp(UOps.CONST, dtypes.float, arg=-1)
    a1 = UOp(UOps.ALU, dtypes.float, (c1,c2), BinaryOps.MUL)
    a2 = UOp(UOps.ALU, dtypes.float, (c1), UnaryOps.NEG)
    self.assert_equiv_uops(matcher.rewrite(a1), a2)

  def test_nested_pattern(self):
    matcher = PatternMatcher([
        (UPat(**{"name": "x", "uop": UOps.ALU, "vin": [
            UPat(**{"uop": UOps.ALU, "vin": [UPat(**{"uop": UOps.CONST}), UPat(**{"uop": UOps.CONST})]}),
            UPat(**{"uop": UOps.CONST})]
        }), lambda x: x)
    ])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    inner_op = UOp(UOps.ALU, dtypes.float, (c1, c2), BinaryOps.ADD)
    outer_op = UOp(UOps.ALU, dtypes.float, (inner_op, c2), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(outer_op), outer_op)
    self.assertEqual(matcher.rewrite(inner_op), None)

  @unittest.skip("no longer supported")
  def test_rewrite_graph_folds(self):
    uops = UOpGraph()
    uops.add(UOps.CONST, dtypes.float, arg=2.0, simplify=False)
    matcher = PatternMatcher([(UPat(**{"name": "x", "uop": UOps.CONST, "dtype": dtypes.float}),
                                lambda x: UOp(UOps.CAST, dtypes.int, (UOp(UOps.ALU, x.dtype, (x, x), BinaryOps.ADD),)))])
    matcher.rewrite_graph(uops)
    # TODO: fix this. it's 2 now
    # self.assertEqual(len(uops.uops), 1)
    self.assertEqual(len(uops.uops), 2)
    self.assert_equiv_uops(UOp(UOps.CONST, dtypes.int, arg=4), uops.uops[-1])

  @unittest.skip("no longer supported")
  def test_rewrite_graph_adds(self):
    uops = UOpGraph()
    uops.add(UOps.CONST, dtypes.int, arg=2, simplify=False)
    matcher = PatternMatcher([(UPat(**{"name": "x", "uop": UOps.CONST, "dtype": dtypes.int}),
                               lambda x: UOp(UOps.STORE, x.dtype, (UOp(UOps.DEFINE_GLOBAL, x.dtype, tuple(), None), x)))])
    matcher.rewrite_graph(uops)
    uops.remove_childless(set(x for x in uops if x.uop in {UOps.STORE}))

    self.assertEqual(len(uops.uops), 3)

    e1 = UOp(UOps.CONST, dtypes.int, arg=2)
    e2 = UOp(UOps.DEFINE_GLOBAL, dtypes.int, tuple())
    e3 = UOp(UOps.STORE, dtypes.int, (e2,e1))

    self.assert_equiv_uops(e1, uops.uops[0])
    self.assert_equiv_uops(e2, uops.uops[1])
    self.assert_equiv_uops(e3, uops.uops[2])

if __name__ == '__main__':
  unittest.main(verbosity=2)
