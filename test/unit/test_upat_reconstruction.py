import unittest
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher

class TestAlgebraic(unittest.TestCase):
  """Test algebraic pattern matching using UPat on both LHS and RHS."""

  def test_plus_0(self):
    """Test x + 0 -> x simplification."""
    pm = PatternMatcher([
      (UPat.var("x") + 0, UPat.var("x"))
    ])
    expr = UOp.const(dtypes.int, 4) + 0
    result = pm.rewrite(expr)
    self.assertEqual(result, UOp.const(dtypes.int, 4))

  def test_div_mul(self):
    """Test (x * x2) / x2 -> x simplification."""
    pm = PatternMatcher([
      ((UPat.var("x") * UPat.var("x2")) / UPat.var("x2"), UPat.var("x"))
    ])
    a, b = UOp.const(dtypes.int, 3), UOp.const(dtypes.int, 4)
    expr = (a * b) / b
    result = pm.rewrite(expr)
    self.assertEqual(result, a)

  def test_mul_is_and(self):
    """Test x * y -> x & y for bool dtype."""
    pm = PatternMatcher([
      (UPat.var('x', dtype=dtypes.bool) * UPat.var('y', dtype=dtypes.bool),
       UPat.var('x') & UPat.var('y'))
    ])
    x, y = UOp.const(dtypes.bool, True), UOp.const(dtypes.bool, True)
    expr = x * y
    result = pm.rewrite(expr)
    self.assertEqual(result, x & y)

  def test_div_neg_1(self):
    """Test x // -1 -> x * -1 rewrite."""
    pm = PatternMatcher([
      (UPat.var("x") // -1, UPat.var("x") * -1)
    ])
    expr = UOp.const(dtypes.float, 4.0) // -1
    result = pm.rewrite(expr)
    self.assertEqual(result, UOp.const(dtypes.float, 4.0) * -1)

if __name__ == '__main__':
  unittest.main(verbosity=2)
