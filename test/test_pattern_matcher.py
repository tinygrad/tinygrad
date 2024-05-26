import unittest
from tinygrad.dtype import dtypes
from tinygrad.ops import BinaryOps
from tinygrad.codegen.uops import UOpGraph, UOps, PatternMatcher, UOp

class TestPatternMatcher(unittest.TestCase):
  def assert_equiv_uops(self, uop1:UOp, uop2:UOp):
    # NOTE: direct UOps __eq__ is comparing object reference, use this function to compare two uops
    self.assertEqual(uop1.uop, uop2.uop)
    self.assertEqual(uop1.dtype, uop2.dtype)
    self.assertEqual(uop1.arg, uop2.arg)

  def test_simple_match(self):
    matcher = PatternMatcher([({"__name__": "x", "uop": UOps.CONST, "dtype": dtypes.float}, lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.int, arg=1)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_uop(self):
    matcher = PatternMatcher([({"__name__": "x", "uop": UOps.CONST}, lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.ALU, dtypes.float, (c1, c1), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_uop_set(self):
    matcher = PatternMatcher([({"__name__": "x", "uop": {UOps.CONST, UOps.DEFINE_ACC}}, lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.ALU, dtypes.float, (c1, c1), BinaryOps.ADD)
    c3 = UOp(UOps.DEFINE_ACC, dtypes.int, arg=(1,))
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)
    self.assertEqual(matcher.rewrite(c3), c3)

  def test_arg(self):
    matcher = PatternMatcher([
      ({"__name__": "x", "uop": UOps.CONST, "arg": 0}, lambda x: x),
      ({"__name__": "x", "uop": UOps.CONST, "arg": False}, lambda x: x),
      ({"__name__": "x", "uop": UOps.ALU, "arg": BinaryOps.MAX}, lambda x: x),
    ])
    c1 = UOp(UOps.CONST, dtypes.float, arg=0.)
    c2 = UOp(UOps.CONST, dtypes.bool, arg=False)
    c3 = UOp(UOps.ALU, dtypes.float, (c1, c1), arg=BinaryOps.MAX)
    c4 = UOp(UOps.ALU, dtypes.float, (c1, c1), arg=BinaryOps.MUL)
    c5 = UOp(UOps.CONST, dtypes.int, arg=-1)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), c2)
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c4), None)
    self.assertEqual(matcher.rewrite(c5), None)

  def test_dup_name(self):
    matcher = PatternMatcher([({"uop": UOps.PHI, "vin": ({"uop": UOps.DEFINE_ACC, "__name__": "acc"}, {"__name__": "acc"})},
      lambda acc: UOp.const(acc.dtype, acc.arg[0]))])
    acc = UOp(UOps.DEFINE_ACC, dtype=dtypes.int, arg=(1,))
    c1 = UOp(UOps.PHI, vin=(acc, acc))
    c2 = UOp(UOps.PHI, vin=(acc, UOp(UOps.CONST, arg=(1,))))
    expected_rewrite = UOp.const(acc.dtype, acc.arg[0])
    self.assert_equiv_uops(matcher.rewrite(c1), expected_rewrite)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_dtype(self):
    matcher = PatternMatcher([({"__name__": "x", "uop": UOps.CONST, "dtype": dtypes.float32}, lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float64, arg=1.0)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_dtype_set(self):
    matcher = PatternMatcher([({"__name__": "x", "uop": UOps.CONST, "dtype": set([dtypes.float32, dtypes.float64])}, lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float64, arg=1.0)
    c3 = UOp(UOps.CONST, dtypes.float16, arg=1.0)
    c4 = UOp(UOps.CONST, dtypes.int, arg=1)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), c2)
    self.assertEqual(matcher.rewrite(c3), None)
    self.assertEqual(matcher.rewrite(c4), None)

  def test_vin_one(self):
    matcher = PatternMatcher([({"__name__": "x", "uop": UOps.ALU, "vin":({"uop": UOps.CONST}, {"uop": UOps.ALU})}, lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.ALU, dtypes.float, (c1,c1), arg=BinaryOps.ADD)
    c3 = UOp(UOps.ALU, dtypes.float, (c1,c2), BinaryOps.ADD)
    c4 = UOp(UOps.ALU, dtypes.float, (c2,c1), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c2), None)
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c4), None)

  def test_vin_permutations(self):
    matcher = PatternMatcher([({"__name__": "x", "uop": UOps.ALU, "vin":[{"uop": UOps.CONST}, {"uop": UOps.ALU}]}, lambda x: x)])
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
    matcher = PatternMatcher([({"__name__": "x", "uop": UOps.ALU, "vin":{"uop": UOps.CONST}}, lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    c3 = UOp(UOps.ALU, dtypes.float, (c1,c2), BinaryOps.ADD)
    c4 = UOp(UOps.ALU, dtypes.float, (c2,c3), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c4), None)

  def test_allow_len(self):
    matcher = PatternMatcher([({"__name__": "x", "uop": UOps.LOAD, "vin": ({"uop": UOps.CONST},), "__allow_len__": {2,4}}, lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    c3 = UOp(UOps.CONST, dtypes.float, arg=3.0)
    c4 = UOp(UOps.LOAD, dtypes.float, (c1,))
    c5 = UOp(UOps.LOAD, dtypes.float, (c1,c2))
    c6 = UOp(UOps.LOAD, dtypes.float, (c1,c2,c3))
    c7 = UOp(UOps.LOAD, dtypes.float, (c1,c2,c3,c3))
    self.assertEqual(matcher.rewrite(c4), c4)
    self.assertEqual(matcher.rewrite(c5), c5)
    self.assertEqual(matcher.rewrite(c6), None)
    self.assertEqual(matcher.rewrite(c7), c7)

  @unittest.skip("no longer supported")
  def test_rewrite_graph_folds(self):
    uops = UOpGraph()
    uops.add(UOps.CONST, dtypes.float, arg=2.0, simplify=False)
    matcher = PatternMatcher([({"__name__": "x", "uop": UOps.CONST, "dtype": dtypes.float},
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
    matcher = PatternMatcher([({"__name__": "x", "uop": UOps.CONST, "dtype": dtypes.int},
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
