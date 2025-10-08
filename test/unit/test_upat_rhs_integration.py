"""
Integration tests for UPat RHS (right-hand side) support in PatternMatcher.

This test suite validates:
1. Backward compatibility with existing lambda and tuple patterns
2. Mixed RHS types (UPat + lambda + tuple) in the same PatternMatcher
3. Dtype preservation and inference through rewrite chains
4. Integration with real subsystems (symbolic_simple, etc.)
5. No regressions in existing pattern matching functionality

The UPat RHS feature allows algebraic patterns like:
  (UPat.var("x") + 0, UPat.var("x"))  # Instead of lambda x: x

This is critical infrastructure used in 34+ modules across tinygrad for:
- Algebraic simplification (codegen/simplify.py)
- Symbolic mathematics (uop/symbolic.py)
- Schedule optimization (schedule/rangeify.py)
- Code generation (codegen/__init__.py)
"""

import unittest
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat
from tinygrad.uop.symbolic import symbolic_simple


class TestBackwardCompatibility(unittest.TestCase):
  """Test that existing lambda and tuple patterns still work correctly."""

  def test_lambda_patterns_still_work(self):
    """Existing lambda patterns must continue to work unchanged."""
    # Simple lambda pattern: x + 0 -> x
    pm = PatternMatcher([(UPat.var("x") + 0, lambda x: x)])

    # Test with int
    expr_int = UOp.const(dtypes.int, 5) + 0
    result_int = pm.rewrite(expr_int)
    self.assertEqual(result_int.arg, 5)
    self.assertEqual(result_int.dtype, dtypes.int)

    # Test with float
    expr_float = UOp.const(dtypes.float, 3.14) + 0
    result_float = pm.rewrite(expr_float)
    self.assertEqual(result_float.arg, 3.14)
    self.assertEqual(result_float.dtype, dtypes.float)

  def test_lambda_with_multiple_vars(self):
    """Lambda patterns with multiple variables should work."""
    # Pattern: (x * y) / y -> x
    pm = PatternMatcher([
      ((UPat.var("x") * UPat.var("y")) / UPat.var("y"), lambda x, y: x)
    ])

    x_val = UOp.const(dtypes.float, 4.0)
    y_val = UOp.const(dtypes.float, 2.0)
    expr = (x_val * y_val) / y_val
    result = pm.rewrite(expr)

    # Should return x (4.0)
    self.assertIsNotNone(result)
    self.assertEqual(result.arg, 4.0)

  def test_lambda_with_const_like(self):
    """Lambda patterns that create new constants should work."""
    # Pattern: x // x -> 1 (using const_like)
    pm = PatternMatcher([
      (UPat.var("x") // UPat.var("x"), lambda x: x.const_like(1))
    ])

    expr = UOp.const(dtypes.int, 5) // UOp.const(dtypes.int, 5)
    result = pm.rewrite(expr)

    self.assertIsNotNone(result)
    self.assertEqual(result.arg, 1)
    self.assertEqual(result.dtype, dtypes.int)

  def test_lambda_with_operations(self):
    """Lambda patterns that create operations should work."""
    # Pattern: x // -1 -> x * -1
    pm = PatternMatcher([
      (UPat.var("x") // -1, lambda x: x * -1)
    ])

    expr = UOp.const(dtypes.int, 4) // -1
    result = pm.rewrite(expr)

    self.assertIsNotNone(result)
    self.assertEqual(result.op, Ops.MUL)
    self.assertEqual(result.src[0].arg, 4)
    # The constant -1 should be in src
    self.assertTrue(any(s.arg == -1 for s in result.src if s.op == Ops.CONST))


class TestMixedRHSTypes(unittest.TestCase):
  """Test that UPat, lambda, and tuple RHS types can coexist in the same PatternMatcher."""

  def test_upat_and_lambda_in_same_matcher(self):
    """UPat and lambda patterns should work together in the same PatternMatcher."""
    pm = PatternMatcher([
      # UPat RHS: x + 0 -> x
      (UPat.var("x") + 0, UPat.var("x")),
      # Lambda RHS: x * 1 -> x
      (UPat.var("x") * 1, lambda x: x),
      # UPat RHS with operation: x * 0 -> 0
      (UPat.var("x") * 0, UPat.const(dtypes.int, 0)),
    ])

    # Test UPat RHS (x + 0)
    expr1 = UOp.const(dtypes.int, 5) + 0
    result1 = pm.rewrite(expr1)
    self.assertIsNotNone(result1)
    self.assertEqual(result1.arg, 5)

    # Test lambda RHS (x * 1)
    expr2 = UOp.const(dtypes.int, 5) * 1
    result2 = pm.rewrite(expr2)
    self.assertIsNotNone(result2)
    self.assertEqual(result2.arg, 5)

    # Test UPat RHS with constant (x * 0)
    expr3 = UOp.const(dtypes.int, 5) * 0
    result3 = pm.rewrite(expr3)
    self.assertIsNotNone(result3)
    self.assertEqual(result3.arg, 0)

  def test_complex_mixed_patterns(self):
    """Complex patterns with mixed RHS types should work."""
    pm = PatternMatcher([
      # Lambda: Boolean identity
      (UPat.var("x", dtype=dtypes.bool) & UPat.var("x"), lambda x: x),
      # UPat: Boolean to bitwise for non-bool types
      (UPat.var("x") * UPat.var("y"), UPat.var("x") & UPat.var("y")),
      # Lambda: Division by 1
      (UPat.var("x") / 1, lambda x: x),
    ])

    # Test lambda pattern
    bool_val = UOp.const(dtypes.bool, True)
    expr_bool = bool_val & bool_val
    result_bool = pm.rewrite(expr_bool)
    self.assertIsNotNone(result_bool)

    # Test UPat pattern
    x = UOp.const(dtypes.int, 3)
    y = UOp.const(dtypes.int, 5)
    expr_mul = x * y
    result_mul = pm.rewrite(expr_mul)
    self.assertIsNotNone(result_mul)
    self.assertEqual(result_mul.op, Ops.AND)

    # Test lambda pattern
    expr_div = UOp.const(dtypes.float, 3.14) / 1
    result_div = pm.rewrite(expr_div)
    self.assertIsNotNone(result_div)


class TestDtypePreservation(unittest.TestCase):
  """Test that dtypes are correctly preserved and inferred through rewrites."""

  def test_dtype_inference_from_variables(self):
    """Constants in RHS should infer dtype from matched variables."""
    # Pattern: x + 0 -> x
    pm = PatternMatcher([
      (UPat.var("x") + 0, UPat.var("x"))
    ])

    # Test with different dtypes
    for dtype in [dtypes.int, dtypes.float, dtypes.int32, dtypes.float64]:
      expr = UOp.const(dtype, 42) + 0
      result = pm.rewrite(expr)
      self.assertIsNotNone(result, f"Failed for dtype {dtype}")
      self.assertEqual(result.dtype, dtype, f"Dtype mismatch for {dtype}")

  def test_dtype_preservation_in_operations(self):
    """Operations in RHS should preserve dtype from operands."""
    # Pattern: x // -1 -> x * -1
    pm = PatternMatcher([
      (UPat.var("x") // -1, UPat.var("x") * -1)
    ])

    for dtype in [dtypes.int, dtypes.float, dtypes.int32]:
      expr = UOp.const(dtype, 10) // -1
      result = pm.rewrite(expr)
      self.assertIsNotNone(result, f"Failed for dtype {dtype}")
      self.assertEqual(result.dtype, dtype, f"Dtype mismatch for {dtype}")
      # Check that the constant -1 also has the correct dtype
      const_src = [s for s in result.src if s.op == Ops.CONST and s.arg == -1]
      if const_src:
        self.assertEqual(const_src[0].dtype, dtype, f"Constant dtype mismatch for {dtype}")

  def test_dtype_in_boolean_operations(self):
    """Boolean operations should preserve bool dtype."""
    # Pattern: bool(x) * bool(y) -> bool(x) & bool(y)
    pm = PatternMatcher([
      (UPat.var("x", dtype=dtypes.bool) * UPat.var("y", dtype=dtypes.bool),
       UPat.var("x") & UPat.var("y"))
    ])

    x = UOp.const(dtypes.bool, True)
    y = UOp.const(dtypes.bool, False)
    expr = x * y
    result = pm.rewrite(expr)

    self.assertIsNotNone(result)
    self.assertEqual(result.dtype, dtypes.bool)
    self.assertEqual(result.op, Ops.AND)

  def test_dtype_chain_preservation(self):
    """Dtype should be preserved through multiple rewrites."""
    pm = PatternMatcher([
      (UPat.var("x") + 0, UPat.var("x")),
      (UPat.var("x") * 1, UPat.var("x")),
    ])

    # Start with float, apply multiple rewrites
    expr = UOp.const(dtypes.float, 3.14)
    expr = expr + 0  # Should rewrite to original
    result1 = pm.rewrite(expr)
    self.assertEqual(result1.dtype, dtypes.float)

    expr2 = result1 * 1  # Should rewrite to original again
    result2 = pm.rewrite(expr2)
    self.assertEqual(result2.dtype, dtypes.float)


class TestSymbolicIntegration(unittest.TestCase):
  """Test integration with real tinygrad subsystems like symbolic_simple."""

  def test_symbolic_simple_patterns_work(self):
    """Verify that symbolic_simple PatternMatcher still works correctly."""
    # Create simple expressions that symbolic_simple should simplify

    # Test: x + 0 -> x
    x = UOp.variable("x", 0, 10)
    expr1 = x + 0
    result1 = symbolic_simple.rewrite(expr1)
    # symbolic_simple should simplify this
    self.assertIsNotNone(result1)

    # Test: x * 1 -> x
    expr2 = x * 1
    result2 = symbolic_simple.rewrite(expr2)
    self.assertIsNotNone(result2)

    # Test: x ^ 0 -> x (for integers)
    x_int = UOp.variable("x_int", 0, 10, dtypes.int)
    expr3 = x_int ^ 0
    result3 = symbolic_simple.rewrite(expr3)
    self.assertIsNotNone(result3)

  def test_symbolic_with_constants(self):
    """Test symbolic patterns with constant folding."""
    # Create expressions with constants
    c1 = UOp.const(dtypes.int, 5)
    c2 = UOp.const(dtypes.int, 0)

    # Test: 5 + 0 should simplify
    expr = c1 + c2
    result = symbolic_simple.rewrite(expr)
    self.assertIsNotNone(result)

  def test_symbolic_division_patterns(self):
    """Test symbolic division patterns."""
    x = UOp.variable("x", 1, 100)

    # Test: x // 1 -> x
    expr1 = x // 1
    result1 = symbolic_simple.rewrite(expr1)
    self.assertIsNotNone(result1)

    # Test: x // -1 -> -x
    expr2 = x // -1
    result2 = symbolic_simple.rewrite(expr2)
    self.assertIsNotNone(result2)


class TestReconstructionEdgeCases(unittest.TestCase):
  """Test edge cases in UPat to UOp reconstruction."""

  def test_nested_operations(self):
    """Test deeply nested UPat operations."""
    # Pattern: (x + y) * 2 -> (x * 2) + (y * 2) [distributive property]
    pm = PatternMatcher([
      ((UPat.var("x") + UPat.var("y")) * 2,
       (UPat.var("x") * 2) + (UPat.var("y") * 2))
    ])

    x = UOp.const(dtypes.int, 3)
    y = UOp.const(dtypes.int, 5)
    expr = (x + y) * 2
    result = pm.rewrite(expr)

    self.assertIsNotNone(result)
    self.assertEqual(result.op, Ops.ADD)
    # Both sources should be multiplications
    self.assertTrue(all(s.op == Ops.MUL for s in result.src))

  def test_multiple_variable_references(self):
    """Test patterns with multiple references to the same variable."""
    # Pattern: x + x -> x * 2
    pm = PatternMatcher([
      (UPat.var("x") + UPat.var("x"), UPat.var("x") * 2)
    ])

    x = UOp.const(dtypes.int, 7)
    expr = x + x
    result = pm.rewrite(expr)

    self.assertIsNotNone(result)
    self.assertEqual(result.op, Ops.MUL)
    self.assertEqual(result.src[0].arg, 7)

  def test_commutative_operations(self):
    """Test that commutative operations are handled correctly."""
    # Pattern: 0 + x -> x (commutative, so x + 0 should also work)
    pm = PatternMatcher([
      (UPat.var("x") + 0, UPat.var("x"))
    ])

    x = UOp.const(dtypes.int, 42)

    # Test both orderings
    expr1 = x + 0
    expr2 = 0 + x

    result1 = pm.rewrite(expr1)
    result2 = pm.rewrite(expr2)

    self.assertIsNotNone(result1)
    self.assertIsNotNone(result2)
    self.assertEqual(result1.arg, 42)
    self.assertEqual(result2.arg, 42)

  def test_constant_only_rhs(self):
    """Test RHS that is only a constant (no variables)."""
    # Pattern: x * 0 -> 0
    pm = PatternMatcher([
      (UPat.var("x") * 0, UPat.const(dtypes.int, 0))
    ])

    expr = UOp.const(dtypes.int, 999) * 0
    result = pm.rewrite(expr)

    self.assertIsNotNone(result)
    self.assertEqual(result.arg, 0)
    self.assertEqual(result.op, Ops.CONST)


class TestCompilationModes(unittest.TestCase):
  """Test that UPat RHS works in both compiled and interpreted modes."""

  def test_interpreted_mode(self):
    """Test UPat RHS in interpreted mode (UPAT_COMPILE=0)."""
    # Force interpreted mode by creating a fresh PatternMatcher with compiled=False
    pm = PatternMatcher([
      (UPat.var("x") + 0, UPat.var("x")),
      (UPat.var("x") * 1, UPat.var("x")),
    ], compiled=False)

    expr1 = UOp.const(dtypes.int, 5) + 0
    result1 = pm.rewrite(expr1)
    self.assertIsNotNone(result1)
    self.assertEqual(result1.arg, 5)

    expr2 = UOp.const(dtypes.int, 5) * 1
    result2 = pm.rewrite(expr2)
    self.assertIsNotNone(result2)
    self.assertEqual(result2.arg, 5)

  def test_compiled_mode(self):
    """Test UPat RHS in compiled mode (UPAT_COMPILE=1)."""
    # Force compiled mode
    pm = PatternMatcher([
      (UPat.var("x") + 0, UPat.var("x")),
      (UPat.var("x") * 1, UPat.var("x")),
    ], compiled=True)

    expr1 = UOp.const(dtypes.int, 5) + 0
    result1 = pm.rewrite(expr1)
    self.assertIsNotNone(result1)
    self.assertEqual(result1.arg, 5)

    expr2 = UOp.const(dtypes.int, 5) * 1
    result2 = pm.rewrite(expr2)
    self.assertIsNotNone(result2)
    self.assertEqual(result2.arg, 5)


class TestRealWorldPatterns(unittest.TestCase):
  """Test real-world pattern matching scenarios from tinygrad subsystems."""

  def test_algebraic_simplification(self):
    """Test common algebraic simplification patterns."""
    pm = PatternMatcher([
      # Additive identity
      (UPat.var("x") + 0, UPat.var("x")),
      # Multiplicative identity
      (UPat.var("x") * 1, UPat.var("x")),
      # Absorbing element
      (UPat.var("x") * 0, UPat.const(dtypes.int, 0)),
      # Division identity
      (UPat.var("x") // 1, UPat.var("x")),
      # Self division
      (UPat.var("x") // UPat.var("x"), UPat.const(dtypes.int, 1)),
    ])

    x = UOp.const(dtypes.int, 42)

    # Test each pattern
    tests = [
      (x + 0, 42, "additive identity"),
      (x * 1, 42, "multiplicative identity"),
      (x * 0, 0, "absorbing element"),
      (x // 1, 42, "division identity"),
      (x // x, 1, "self division"),
    ]

    for expr, expected_arg, name in tests:
      result = pm.rewrite(expr)
      self.assertIsNotNone(result, f"Failed to rewrite {name}")
      self.assertEqual(result.arg, expected_arg, f"Wrong result for {name}")

  def test_boolean_algebra(self):
    """Test boolean algebra patterns."""
    pm = PatternMatcher([
      # Idempotent
      (UPat.var("x", dtype=dtypes.bool) & UPat.var("x"), UPat.var("x")),
      (UPat.var("x", dtype=dtypes.bool) | UPat.var("x"), UPat.var("x")),
      # Identity
      (UPat.var("x", dtype=dtypes.bool) & UPat.const(dtypes.bool, True), UPat.var("x")),
      (UPat.var("x", dtype=dtypes.bool) | UPat.const(dtypes.bool, False), UPat.var("x")),
    ])

    x = UOp.const(dtypes.bool, True)
    true_const = UOp.const(dtypes.bool, True)
    false_const = UOp.const(dtypes.bool, False)

    # Test idempotent
    expr1 = x & x
    result1 = pm.rewrite(expr1)
    self.assertIsNotNone(result1)

    expr2 = x | x
    result2 = pm.rewrite(expr2)
    self.assertIsNotNone(result2)

    # Test identity
    expr3 = x & true_const
    result3 = pm.rewrite(expr3)
    self.assertIsNotNone(result3)

    expr4 = x | false_const
    result4 = pm.rewrite(expr4)
    self.assertIsNotNone(result4)

  def test_division_optimization(self):
    """Test division optimization patterns."""
    pm = PatternMatcher([
      # Division by -1 -> multiplication by -1
      (UPat.var("x") // -1, UPat.var("x") * -1),
      # Division cancellation: (x * y) / y -> x
      ((UPat.var("x") * UPat.var("y")) / UPat.var("y"), UPat.var("x")),
    ])

    x = UOp.const(dtypes.int, 10)
    y = UOp.const(dtypes.int, 5)

    # Test division by -1
    expr1 = x // -1
    result1 = pm.rewrite(expr1)
    self.assertIsNotNone(result1)
    self.assertEqual(result1.op, Ops.MUL)

    # Test division cancellation
    expr2 = (x * y) / y
    result2 = pm.rewrite(expr2)
    self.assertIsNotNone(result2)
    self.assertEqual(result2.arg, 10)


if __name__ == '__main__':
  unittest.main(verbosity=2)
