import unittest, itertools
from tinygrad.dtype import dtypes
from tinygrad.ops import UOps, UOp, BinaryOps, TernaryOps, ReduceOps, UnaryOps # noqa: F401
from tinygrad.ops import PatternMatcher, UPat

class TestPatternMatcher(unittest.TestCase):
  def test_simple_match(self):
    matcher = PatternMatcher([(UPat(UOps.CONST, name="x", dtype=dtypes.float), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.int, arg=1)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_match_sz_0(self):
    match_cnt = 0
    def fxn(x):
      nonlocal match_cnt
      match_cnt += 1
      assert len(x.src) == 0
      return UOp(UOps.CONST, src=(UOp(UOps.CONST),))
    matcher = PatternMatcher([(UPat(UOps.CONST, src=(), name="x"), fxn)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    # second rewrite shouldn't match anything
    c1 = matcher.rewrite(c1)
    c1 = matcher.rewrite(c1)
    self.assertEqual(match_cnt, 1)

  def test_uop(self):
    matcher = PatternMatcher([(UPat(UOps.CONST, name="x"), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.ALU, dtypes.float, (c1, c1), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_uop_set(self):
    matcher = PatternMatcher([(UPat({UOps.CONST, UOps.CAST}, name="x"), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.bool, arg=False)
    c2 = UOp(UOps.CAST, dtypes.int, (c1,))
    c3 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c4 = UOp(UOps.ALU, dtypes.float, (c3, c3), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), c2)
    self.assertEqual(matcher.rewrite(c4), None)

  def test_arg(self):
    matcher = PatternMatcher([
      (UPat(UOps.CONST, arg=0, name="x"), lambda x: x),
      (UPat(UOps.CONST, arg=False, name="x"), lambda x: x),
      (UPat(UOps.ALU, arg=BinaryOps.MAX, name="x"), lambda x: x),
    ])
    c1 = UOp(UOps.CONST, dtypes.float, arg=0.0)
    c2 = UOp(UOps.CONST, dtypes.bool, arg=False)
    c3 = UOp(UOps.ALU, dtypes.float, (c1, c1), arg=BinaryOps.MAX)
    c4 = UOp(UOps.ALU, dtypes.float, (c1, c1), arg=BinaryOps.MUL)
    c5 = UOp(UOps.CONST, dtypes.int, arg=-1)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), c2)
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c4), None)
    self.assertEqual(matcher.rewrite(c5), None)

  def test_filter_arg(self):
    matcher = PatternMatcher([
      (UPat(UOps.ALU, arg=BinaryOps.MUL, src=[UPat(UOps.CONST, name="c"), UPat(UOps.CONST, arg=2)], name="x"),
       lambda x,c: x if c.arg in {1, -1} else None)
    ])
    y1 = UOp(UOps.CONST, dtypes.int, arg=1)
    y2 = UOp(UOps.CONST, dtypes.int, arg=2)
    y3 = UOp(UOps.CONST, dtypes.int, arg=-1)
    c1 = UOp(UOps.ALU, dtypes.int, (y1, y2), BinaryOps.MUL)
    c2 = UOp(UOps.ALU, dtypes.int, (y2, y2), BinaryOps.MUL)
    c3 = UOp(UOps.ALU, dtypes.int, (y3, y2), BinaryOps.MUL)
    c4 = UOp(UOps.ALU, dtypes.int, (y2, y1), BinaryOps.MUL)
    c5 = UOp(UOps.ALU, dtypes.int, (y2, y3), BinaryOps.MUL)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c4), c4)
    self.assertEqual(matcher.rewrite(c5), c5)

  def test_dup_name(self):
    matcher = PatternMatcher([(UPat(UOps.ALU, name="x", src=(UPat(UOps.CONST, name="y"), UPat(UOps.CONST, name="y"))), lambda x, y: x)])
    y1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    y2 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c1 = UOp(UOps.ALU, dtypes.float, (y1, y1), BinaryOps.ADD)
    c2 = UOp(UOps.ALU, dtypes.float, (y1, y2), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_dtype(self):
    matcher = PatternMatcher([(UPat(UOps.CONST, name="x", dtype=dtypes.float32), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float64, arg=1.0)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), None)

  def test_dtype_set(self):
    matcher = PatternMatcher([(UPat(UOps.CONST, name="x", dtype={dtypes.float32, dtypes.float64}), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float64, arg=1.0)
    c3 = UOp(UOps.CONST, dtypes.float16, arg=1.0)
    c4 = UOp(UOps.CONST, dtypes.int, arg=1)
    self.assertEqual(matcher.rewrite(c1), c1)
    self.assertEqual(matcher.rewrite(c2), c2)
    self.assertEqual(matcher.rewrite(c3), None)
    self.assertEqual(matcher.rewrite(c4), None)

  def test_src_one(self):
    matcher = PatternMatcher([(UPat(UOps.ALU, name="x", src=(UPat(UOps.CONST), UPat(UOps.CONST))), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    c3 = UOp(UOps.ALU, dtypes.float, (c1,c2), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c2), None)
    matcher = PatternMatcher([(UPat(UOps.ALU, name="x", src=(UPat(UOps.CONST), UPat(UOps.ALU))), lambda x: x)])
    c4 = UOp(UOps.ALU, dtypes.float, (c1,c3), BinaryOps.ADD)
    c5 = UOp(UOps.ALU, dtypes.float, (c3,c1), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c3), None)
    self.assertEqual(matcher.rewrite(c4), c4)
    self.assertEqual(matcher.rewrite(c5), None)

  def test_src_permutations(self):
    matcher = PatternMatcher([(UPat(UOps.ALU, name="x", src=[UPat(UOps.CONST), UPat(UOps.ALU)]), lambda x: x)])
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

  def test_src_repeat(self):
    matcher = PatternMatcher([(UPat(UOps.ALU, name="x", src=UPat(UOps.CONST)), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    c3 = UOp(UOps.ALU, dtypes.float, (c1,c2), BinaryOps.ADD)
    c4 = UOp(UOps.ALU, dtypes.float, (c2,c3), BinaryOps.ADD)
    self.assertEqual(matcher.rewrite(c3), c3)
    self.assertEqual(matcher.rewrite(c4), None)

  def test_allow_len(self):
    matcher = PatternMatcher([(UPat(UOps.ALU, name="x", src=(UPat(UOps.CONST),), allow_any_len=True, arg=TernaryOps.MULACC), lambda x: x)])
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    c3 = UOp(UOps.CONST, dtypes.float, arg=3.0)
    c4 = UOp(UOps.ALU, dtypes.float, (c1,), UnaryOps.EXP2)
    c5 = UOp(UOps.ALU, dtypes.float, (c1,c2), BinaryOps.ADD)
    c6 = UOp(UOps.ALU, dtypes.float, (c1,c2,c3), TernaryOps.MULACC)
    self.assertEqual(matcher.rewrite(c4), None)
    self.assertEqual(matcher.rewrite(c5), None)
    self.assertEqual(matcher.rewrite(c6), c6)

  def test_deep_src_permutations(self):
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    u1 = (c1 + c2) + c1
    u2 = (c2 + c1) + c1
    matcher = PatternMatcher([
      (UPat(UOps.ALU, src=[UPat(UOps.ALU, src=[UPat(name='a'), UPat(name='b')]), UPat(name='b')]), lambda a,b: b)
    ])
    self.assertIsNotNone(matcher.rewrite(u1))
    self.assertIsNotNone(matcher.rewrite(u2))

  def _assert_eq_upat(self, a:UPat, b:UPat):
    assert (sorted(map(str,a.op)) if a.op else [] == (sorted(map(str,b.op)) if b.op else []))
    assert (sorted(a.dtype) if a.dtype else [] == (sorted(b.dtype) if b.dtype else []))
    assert (a.name, type(a.src)) == (b.name, type(b.src))
    def simple_src(u:UPat):
      if u.src is None: return []
      if isinstance(u.src, itertools.repeat): return next(u.src[0])
      return u.src[0]
    for a,b in zip(simple_src(a), simple_src(b)): self._assert_eq_upat(a, b)

if __name__ == '__main__':
  unittest.main(verbosity=2)
