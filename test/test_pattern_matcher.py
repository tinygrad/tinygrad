import unittest
from tinygrad.dtype import dtypes
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
from tinygrad.codegen.uops import UOpGraph, UOps, PatternMatcher, UOp, UPat, loop_collapse

class TestPatternMatcher(unittest.TestCase):
  def assert_equiv_uops(self, uop1: UOp, uop2: UOp):
    # NOTE: direct UOps __eq__ is comparing object reference, use this function to compare two uops
    self.assertEqual(uop1.uop, uop2.uop)
    self.assertEqual(uop1.dtype, uop2.dtype)
    self.assertEqual(uop1.arg, uop2.arg)

  def test_simple_match(self):
    matcher = PatternMatcher([(UPat(name="x", uop=UOps.CONST, dtype=dtypes.float), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.int, arg=1)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_dtype_set(self):
    matcher = PatternMatcher([(UPat(name="x", uop=UOps.CONST, dtype=set([dtypes.float32, dtypes.float64, dtypes.bool])), lambda x: x)])
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
    matcher = PatternMatcher([(UPat(name="x", uop=UOps.ALU, vin=(UPat(uop=UOps.CONST), UPat(uop=UOps.CONST))), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    c3 = UOp(UOps.ALU, dtypes.float, (c1, c2), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_vin_permutations(self):
    matcher = PatternMatcher([(UPat(name="x", uop=UOps.ALU, vin=[UPat(uop=UOps.CONST), UPat(uop=UOps.ALU)]), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    c3 = UOp(UOps.ALU, dtypes.float, (c1, c2), BinaryOps.ADD)
    c4 = UOp(UOps.ALU, dtypes.float, (c3, c2), BinaryOps.ADD)
    c5 = UOp(UOps.ALU, dtypes.float, (c2, c3), BinaryOps.ADD)
    c6 = UOp(UOps.ALU, dtypes.float, (c3, c4), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c3), None)
    self.assertEqual(matcher.rewrite(c4), c4)
    self.assertEqual(matcher.rewrite(c5), c5)
    self.assertEqual(matcher.rewrite(c6), None)

  def test_vin_repeat(self):
    matcher = PatternMatcher([(UPat(name="x", uop=UOps.ALU, vin=UPat(uop=UOps.CONST)), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    c3 = UOp(UOps.ALU, dtypes.float, (c1, c2), BinaryOps.ADD)
    c4 = UOp(UOps.ALU, dtypes.float, (c2, c3), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c4), None)

  def test_constant_arg(self):
    matcher = PatternMatcher([(UPat(name="x", uop=UOps.CONST, arg=42), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.int, arg=42)
    c2 = UOp(UOps.CONST, dtypes.int, arg=43)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_zero_add(self):
    matcher = PatternMatcher([(UPat(uop=UOps.ALU, arg=BinaryOps.ADD, vin=[UPat(name="x"), UPat(uop=UOps.CONST, arg=0)]), lambda x: x)])
    c0 = UOp(UOps.CONST, dtypes.float, arg=0.0)
    x = UOp(UOps.LOAD, dtypes.float, arg='x')
    a1 = UOp(UOps.ALU, dtypes.float, (c0, x), BinaryOps.ADD)
    a2 = UOp(UOps.ALU, dtypes.float, (x, c0), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(a1), x)
    self.assertEqual(matcher.rewrite(a2), x)

  def test_sub_zero(self):
    matcher = PatternMatcher([(UPat(uop=UOps.ALU, arg=BinaryOps.SUB, vin=(UPat(name="x"), UPat(uop=UOps.CONST, arg=0))), lambda x: x)])
    c0 = UOp(UOps.CONST, dtypes.float, arg=0.0)
    x = UOp(UOps.LOAD, dtypes.float, arg='x')
    a1 = UOp(UOps.ALU, dtypes.float, (c0, x), BinaryOps.SUB)
    a2 = UOp(UOps.ALU, dtypes.float, (x, c0), BinaryOps.SUB)
    self.assertEqual(matcher.rewrite(a1), None)
    self.assertEqual(matcher.rewrite(a2), x)

  def test_zero_mul(self):
    matcher = PatternMatcher([(UPat(uop=UOps.ALU, arg=BinaryOps.MUL, vin=[UPat(), UPat(name="c", uop=UOps.CONST, arg=0)]), lambda c: c)])
    c0 = UOp(UOps.CONST, dtypes.float, arg=0.0)
    x = UOp(UOps.LOAD, dtypes.float, arg='x')
    a1 = UOp(UOps.ALU, dtypes.float, (c0, x), BinaryOps.MUL)
    a2 = UOp(UOps.ALU, dtypes.float, (x, c0), BinaryOps.MUL)
    self.assertEqual(matcher.rewrite(a1), c0)
    self.assertEqual(matcher.rewrite(a2), c0)

  def test_self_sub(self):
    matcher = PatternMatcher([(UPat(uop=UOps.ALU, arg=BinaryOps.SUB, vin=(UPat(name="x"), UPat(name="x"))), lambda x: UOp.const(x.dtype, 0))])   # x-x -> 0
    c0_int = UOp(UOps.CONST, dtypes.int, arg=0)
    c0_float = UOp(UOps.CONST, dtypes.float, arg=0.0)
    c1 = UOp(UOps.CONST, dtypes.int, arg=10)
    c2 = UOp(UOps.CONST, dtypes.float, arg=55.55)
    a1 = UOp(UOps.ALU, dtypes.int, (c1, c1), BinaryOps.SUB)
    a2 = UOp(UOps.ALU, dtypes.float, (c2, c2), BinaryOps.SUB)
    self.assert_equiv_uops(matcher.rewrite(a1), c0_int)
    self.assert_equiv_uops(matcher.rewrite(a2), c0_float)

  def test_fold_neg_mul(self):
    matcher = PatternMatcher([(UPat(uop=UOps.ALU, arg=BinaryOps.MUL, vin=[UPat(name="x"), UPat(uop=UOps.CONST, arg=-1)]), lambda x: -x)])
    c1 = UOp(UOps.LOAD, dtypes.float, arg='xx')
    c2 = UOp(UOps.CONST, dtypes.float, arg=-1)
    a1 = UOp(UOps.ALU, dtypes.float, (c1, c2), BinaryOps.MUL)
    a2 = UOp(UOps.ALU, dtypes.float, (c1), UnaryOps.NEG)
    self.assert_equiv_uops(matcher.rewrite(a1), a2)

  def test_nested_pattern(self):
    matcher = PatternMatcher([
        (UPat(name="x", uop=UOps.ALU, vin=[
            UPat(uop=UOps.ALU, vin=[UPat(uop=UOps.CONST), UPat(uop=UOps.CONST)]),
            UPat(uop=UOps.CONST)
        ]), lambda x: x)
    ])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    inner_op = UOp(UOps.ALU, dtypes.float, (c1, c2), BinaryOps.ADD)
    outer_op = UOp(UOps.ALU, dtypes.float, (inner_op, c2), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(outer_op), outer_op)
    self.assertEqual(matcher.rewrite(inner_op), None)

  def test_unmul(self):
    matcher = PatternMatcher([
        (UPat(uop=UOps.ALU, arg=BinaryOps.MUL, vin=[UPat(uop=UOps.CONST, name="c1"),
                                                    UPat(uop=UOps.UNMUL, vin=[UPat(uop=UOps.CONST, name="c2"), UPat(name="v")])]),
          lambda c1, c2, v: v if c1.arg == c2.arg else None),
        (UPat(uop=UOps.UNMUL, vin=(UPat(uop=UOps.CONST, name="zero", arg=0), UPat())), lambda zero: zero)
    ])
    c0 = UOp(UOps.CONST, dtypes.float, arg=0.0)
    c1 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    v = UOp(UOps.LOAD, dtypes.float, arg='x')
    unmul = UOp(UOps.UNMUL, dtypes.float, (c2, v))
    mul = UOp(UOps.ALU, dtypes.float, (c1, unmul), BinaryOps.MUL)
    self.assertEqual(matcher.rewrite(mul), v)
    self.assertEqual(matcher.rewrite(UOp(UOps.UNMUL, dtypes.float, (c0, v))), c0)

  def test_max_special(self):
    matcher = PatternMatcher([(UPat(uop=UOps.ALU, arg=BinaryOps.MAX, vin=[UPat(name="c", uop=UOps.CONST), UPat(name="s", uop=UOps.SPECIAL)]),
                                lambda c, s: c if (s.arg[2] - 1) <= c.arg else None)])
    c = UOp(UOps.CONST, dtypes.int, arg=5)
    s = UOp(UOps.SPECIAL, dtypes.int, arg=(0, 0, 4))
    max_op = UOp(UOps.ALU, dtypes.int, (c, s), BinaryOps.MAX)
    self.assertEqual(matcher.rewrite(max_op), c)

  def test_cast_noop(self):
    matcher = PatternMatcher([(UPat(name="root", uop=UOps.CAST), lambda root: root.vin[0] if root.dtype is root.vin[0].dtype else None)])
    c = UOp(UOps.CONST, dtypes.float, arg=5.0)
    cast_op = UOp(UOps.CAST, dtypes.float, (c,))
    self.assertEqual(matcher.rewrite(cast_op), c)

  def test_phi_noop(self):
    matcher = PatternMatcher([
      (UPat(uop=UOps.PHI, vin=(UPat(uop=UOps.DEFINE_ACC, name="acc"), UPat(name="acc"))), lambda acc: UOp.const(acc.dtype, acc.arg[0])),
      (UPat(uop=UOps.PHI, vin=(UPat(uop=UOps.DEFINE_ACC, vin=()), UPat(name="x"))), lambda x: x),
      (UPat(uop=UOps.PHI, vin=(UPat(uop=UOps.CONST), UPat(name="x"))), lambda x: x),
    ])
    acc = UOp(UOps.DEFINE_ACC, dtypes.int, (UOp(UOps.RANGE, dtypes.int, arg=(0, 10)),))
    x = UOp(UOps.CONST, dtypes.int, arg=5)
    phi_op = UOp(UOps.PHI, dtypes.int, (acc, x))
    self.assertEqual(matcher.rewrite(phi_op), x)
    phi_op = UOp(UOps.PHI, dtypes.int, (x, x))
    self.assertEqual(matcher.rewrite(phi_op), x)

  def test_arange_loop_folding(self):
    matcher = PatternMatcher([
      (UPat(uop=UOps.ALU, arg=TernaryOps.WHERE, vin=(
          UPat(uop=UOps.ALU, arg=BinaryOps.CMPLT, vin=(
              UPat(uop=UOps.ALU, arg=BinaryOps.ADD, vin=[
                  UPat(name="idx"), UPat(uop=UOps.ALU, arg=BinaryOps.MUL,
                                          vin=[UPat(name="mval", uop=UOps.CONST), UPat(uop=UOps.RANGE, vin=(UPat(name="loop_start"), UPat(name="loop_end")))])]),
              UPat(name="compval", uop=UOps.CONST))), UPat(name="multconst", uop=UOps.CONST), UPat(uop=UOps.CONST, arg=0))), loop_collapse)
    ])

    idx = UOp(UOps.LOAD, dtypes.int, arg='idx')
    mval = UOp(UOps.CONST, dtypes.int, arg=2)
    loop_start = UOp(UOps.CONST, dtypes.int, arg=0)
    loop_end = UOp(UOps.CONST, dtypes.int, arg=10)
    range_op = UOp(UOps.RANGE, dtypes.int, arg=(loop_start, loop_end))
    add_op = UOp(UOps.ALU, dtypes.int, (idx, UOp(UOps.ALU, dtypes.int, (mval, range_op), BinaryOps.MUL)), BinaryOps.ADD)
    compval = UOp(UOps.CONST, dtypes.int, arg=20)
    cmplt_op = UOp(UOps.ALU, dtypes.bool, (add_op, compval), BinaryOps.CMPLT)
    multconst = UOp(UOps.CONST, dtypes.int, arg=5)
    where_op = UOp(UOps.ALU, dtypes.int, (cmplt_op, multconst, UOp(UOps.CONST, dtypes.int, arg=0)), TernaryOps.WHERE)
    self.assertEqual(matcher.rewrite(where_op), loop_collapse(idx, mval, loop_start, loop_end, compval, multconst))

if __name__ == '__main__':
  unittest.main(verbosity=2)