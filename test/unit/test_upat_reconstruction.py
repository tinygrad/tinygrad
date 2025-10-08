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

class TestUPatReconstruction(unittest.TestCase):
  """Test UPat reconstruction infrastructure."""

  def test_variable_reference(self):
    """Test simple variable reference reconstruction."""
    pm = PatternMatcher([
      (UPat.var("x") + 0, UPat.var("x"))
    ])
    expr = UOp.const(dtypes.int, 5) + 0
    result = pm.rewrite(expr)
    self.assertEqual(result, UOp.const(dtypes.int, 5))

  def test_constant_reconstruction(self):
    """Test constant reconstruction with dtype inference."""
    pm = PatternMatcher([
      (UPat.var("x") + UPat.var("y"), UPat.var("x") + 1)
    ])
    a, b = UOp.const(dtypes.int, 2), UOp.const(dtypes.int, 3)
    expr = a + b
    result = pm.rewrite(expr)
    expected = a + 1
    self.assertEqual(result, expected)

  def test_nested_operations(self):
    """Test nested operation reconstruction."""
    pm = PatternMatcher([
      (UPat.var("x") + (UPat.var("y") * UPat.var("z")),
       (UPat.var("x") * UPat.var("z")) + UPat.var("y"))
    ])
    a, b, c = UOp.const(dtypes.int, 1), UOp.const(dtypes.int, 2), UOp.const(dtypes.int, 3)
    expr = a + (b * c)
    result = pm.rewrite(expr)
    expected = (a * c) + b
    self.assertEqual(result, expected)

  def test_multiple_variables(self):
    """Test reconstruction with multiple variables."""
    pm = PatternMatcher([
      (UPat.var("a") * UPat.var("b") * UPat.var("c"),
       UPat.var("c") * UPat.var("b") * UPat.var("a"))
    ])
    a, b, c = UOp.const(dtypes.int, 2), UOp.const(dtypes.int, 3), UOp.const(dtypes.int, 4)
    expr = a * b * c
    result = pm.rewrite(expr)
    expected = c * b * a
    self.assertEqual(result, expected)

  def test_float_dtype(self):
    """Test dtype preservation for float operations."""
    pm = PatternMatcher([
      (UPat.var("x", dtype=dtypes.float) + 0.0, UPat.var("x"))
    ])
    expr = UOp.const(dtypes.float, 3.14) + 0.0
    result = pm.rewrite(expr)
    self.assertEqual(result, UOp.const(dtypes.float, 3.14))
    self.assertEqual(result.dtype, dtypes.float)

  def test_bool_dtype(self):
    """Test bool dtype operations."""
    pm = PatternMatcher([
      (UPat.var("x", dtype=dtypes.bool) * UPat.var("y", dtype=dtypes.bool),
       UPat.var("x") & UPat.var("y"))
    ])
    x, y = UOp.const(dtypes.bool, True), UOp.const(dtypes.bool, False)
    expr = x * y
    result = pm.rewrite(expr)
    self.assertEqual(result, x & y)

  def test_constant_with_explicit_dtype(self):
    """Test constant reconstruction with explicit dtype in UPat."""
    pm = PatternMatcher([
      (UPat(Ops.CONST, name="x"), UPat.const(dtypes.float, 1.0))
    ])
    expr = UOp.const(dtypes.int, 5)
    result = pm.rewrite(expr)
    self.assertEqual(result.dtype, dtypes.float)
    self.assertEqual(result.arg, 1.0)

  def test_no_match_returns_none(self):
    """Test that non-matching patterns return None."""
    pm = PatternMatcher([
      (UPat(Ops.ADD, src=(UPat.var("x"), UPat.const(dtypes.int, 0))), UPat.var("x"))
    ])
    expr = UOp.const(dtypes.int, 5) + 1  # doesn't match (not adding 0)
    result = pm.rewrite(expr)
    self.assertIsNone(result)

  def test_commutative_match(self):
    """Test that commutative operations match in any order."""
    pm = PatternMatcher([
      (UPat.var("x") + UPat.const(dtypes.int, 0), UPat.var("x"))
    ])
    # Test 0 + x (should match due to commutativity)
    expr = UOp.const(dtypes.int, 0) + UOp.const(dtypes.int, 5)
    result = pm.rewrite(expr)
    self.assertEqual(result, UOp.const(dtypes.int, 5))

  def test_chained_rewrites(self):
    """Test multiple pattern rewrites in sequence."""
    pm = PatternMatcher([
      (UPat.var("x") + 0, UPat.var("x")),
      (UPat.var("x") * 1, UPat.var("x")),
    ])
    x = UOp.const(dtypes.int, 5)
    # Test first pattern
    expr1 = x + 0
    result1 = pm.rewrite(expr1)
    self.assertEqual(result1, x)
    # Test second pattern
    expr2 = x * 1
    result2 = pm.rewrite(expr2)
    self.assertEqual(result2, x)

  def test_pattern_with_arg(self):
    """Test pattern matching with specific arg values."""
    pm = PatternMatcher([
      (UPat.var("x") * UPat.const(dtypes.int, 2), UPat.var("x") + UPat.var("x"))
    ])
    expr = UOp.const(dtypes.int, 5) * 2
    result = pm.rewrite(expr)
    expected = UOp.const(dtypes.int, 5) + UOp.const(dtypes.int, 5)
    self.assertEqual(result, expected)

class TestEdgeCases(unittest.TestCase):
  """Test edge cases and error conditions."""

  def test_empty_store(self):
    """Test constant reconstruction with empty variable store."""
    pm = PatternMatcher([
      (UPat(Ops.CONST, arg=5), UPat.const(None, 10))  # dtype=None to test default
    ])
    expr = UOp.const(dtypes.int, 5)
    result = pm.rewrite(expr)
    # Should use dtypes.int as default
    self.assertEqual(result.arg, 10)

  def test_dtype_inference_from_store(self):
    """Test that dtype is correctly inferred from matched variables."""
    pm = PatternMatcher([
      (UPat.var("x", dtype=dtypes.float) + UPat.var("y", dtype=dtypes.float),
       UPat.var("x") + 1)  # 1 should get float dtype from store
    ])
    x, y = UOp.const(dtypes.float, 2.0), UOp.const(dtypes.float, 3.0)
    expr = x + y
    result = pm.rewrite(expr)
    # The constant 1 should be converted to float
    self.assertEqual(result, x + 1)
    self.assertEqual(result.src[1].dtype, dtypes.float)

  def test_backward_compatibility_lambda(self):
    """Test that lambda RHS still works (backward compatibility)."""
    pm = PatternMatcher([
      (UPat.var("x") + 0, lambda x: x)
    ])
    expr = UOp.const(dtypes.int, 5) + 0
    result = pm.rewrite(expr)
    self.assertEqual(result, UOp.const(dtypes.int, 5))

  def test_backward_compatibility_tuple(self):
    """Test that tuple RHS (pickled functions) still works."""
    # Create a simple lambda and pickle it
    import types
    fxn = lambda x: x
    # Simulate the tuple format used in __reduce__
    from tinygrad.uop.ops import deconstruct_function
    tuple_form = deconstruct_function(fxn)

    pm = PatternMatcher([
      (UPat.var("x") + 0, tuple_form)
    ])
    expr = UOp.const(dtypes.int, 5) + 0
    result = pm.rewrite(expr)
    self.assertEqual(result, UOp.const(dtypes.int, 5))

  def test_mixed_rhs_types(self):
    """Test PatternMatcher with mixed RHS types (UPat, lambda, tuple)."""
    fxn = lambda x: x
    from tinygrad.uop.ops import deconstruct_function
    tuple_form = deconstruct_function(fxn)

    pm = PatternMatcher([
      (UPat.var("x") + 0, UPat.var("x")),          # UPat RHS
      (UPat.var("x") * 1, lambda x: x),             # Lambda RHS
      (UPat.var("x") + UPat.var("x"), tuple_form),  # Tuple RHS
    ])

    expr1 = UOp.const(dtypes.int, 5) + 0
    result1 = pm.rewrite(expr1)
    self.assertEqual(result1, UOp.const(dtypes.int, 5))

    expr2 = UOp.const(dtypes.int, 5) * 1
    result2 = pm.rewrite(expr2)
    self.assertEqual(result2, UOp.const(dtypes.int, 5))

    expr3 = UOp.const(dtypes.int, 5) + UOp.const(dtypes.int, 5)
    result3 = pm.rewrite(expr3)
    self.assertEqual(result3, UOp.const(dtypes.int, 5))

class TestIntegration(unittest.TestCase):
  """Integration tests for full algebraic rewrite scenarios."""

  def test_algebraic_simplification_chain(self):
    """Test a chain of algebraic simplifications."""
    pm = PatternMatcher([
      (UPat.var("x") + 0, UPat.var("x")),
      (UPat.var("x") * 1, UPat.var("x")),
    ])

    # Test x + 0
    x = UOp.const(dtypes.int, 5)
    expr = x + 0
    result = pm.rewrite(expr)
    self.assertEqual(result, x)

    # Test x * 1
    expr2 = x * 1
    result2 = pm.rewrite(expr2)
    self.assertEqual(result2, x)

  def test_associativity_rewrite(self):
    """Test associativity rewrite pattern."""
    pm = PatternMatcher([
      ((UPat.var("x") + UPat.var("y")) + UPat.var("z"),
       UPat.var("x") + (UPat.var("y") + UPat.var("z")))
    ])
    a, b, c = UOp.const(dtypes.int, 1), UOp.const(dtypes.int, 2), UOp.const(dtypes.int, 3)
    expr = (a + b) + c
    result = pm.rewrite(expr)
    expected = a + (b + c)
    self.assertEqual(result, expected)

  def test_distributivity(self):
    """Test distributive law: x * (y + z) -> (x * y) + (x * z)."""
    pm = PatternMatcher([
      (UPat.var("x") * (UPat.var("y") + UPat.var("z")),
       (UPat.var("x") * UPat.var("y")) + (UPat.var("x") * UPat.var("z")))
    ])
    a, b, c = UOp.const(dtypes.int, 2), UOp.const(dtypes.int, 3), UOp.const(dtypes.int, 4)
    expr = a * (b + c)
    result = pm.rewrite(expr)
    expected = (a * b) + (a * c)
    self.assertEqual(result, expected)

if __name__ == '__main__':
  unittest.main(verbosity=2)
