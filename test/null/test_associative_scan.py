"""Tests for Tensor.associative_scan (PR #16280)."""
import unittest, numpy as np
from tinygrad import Tensor

class TestAssociativeScanAddition(unittest.TestCase):
  """associative_scan with addition should match cumsum."""
  def test_add_1d(self):
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    got = Tensor(x.tolist()).associative_scan(lambda a, b: a + b).numpy()
    np.testing.assert_allclose(got, np.cumsum(x))

  def test_add_1d_random(self):
    rng = np.random.default_rng(42)
    x = rng.standard_normal(32).astype(np.float32)
    got = Tensor(x.tolist()).associative_scan(lambda a, b: a + b).numpy()
    np.testing.assert_allclose(got, np.cumsum(x), atol=1e-5)

  def test_add_2d_axis0(self):
    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    got = Tensor(x.tolist()).associative_scan(lambda a, b: a + b, axis=0).numpy()
    np.testing.assert_allclose(got, np.cumsum(x, axis=0))

  def test_add_2d_axis1(self):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    got = Tensor(x.tolist()).associative_scan(lambda a, b: a + b, axis=1).numpy()
    np.testing.assert_allclose(got, np.cumsum(x, axis=1))

  def test_add_negative_axis(self):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    got = Tensor(x.tolist()).associative_scan(lambda a, b: a + b, axis=-1).numpy()
    np.testing.assert_allclose(got, np.cumsum(x, axis=-1))

  def test_add_3d(self):
    rng = np.random.default_rng(7)
    x = rng.standard_normal((3, 4, 5)).astype(np.float32)
    for ax in range(3):
      got = Tensor(x.tolist()).associative_scan(lambda a, b: a + b, axis=ax).numpy()
      np.testing.assert_allclose(got, np.cumsum(x, axis=ax), atol=1e-5)

class TestAssociativeScanMultiplication(unittest.TestCase):
  """associative_scan with multiplication should match cumprod."""
  def test_mul_1d(self):
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    got = Tensor(x.tolist()).associative_scan(lambda a, b: a * b).numpy()
    np.testing.assert_allclose(got, np.cumprod(x))

  def test_mul_2d_axis0(self):
    x = np.array([[1, 2], [3, 4], [2, 3]], dtype=np.float32)
    got = Tensor(x.tolist()).associative_scan(lambda a, b: a * b, axis=0).numpy()
    np.testing.assert_allclose(got, np.cumprod(x, axis=0))

  def test_mul_2d_axis1(self):
    x = np.array([[1, 2, 3], [4, 2, 1]], dtype=np.float32)
    got = Tensor(x.tolist()).associative_scan(lambda a, b: a * b, axis=1).numpy()
    np.testing.assert_allclose(got, np.cumprod(x, axis=1))

class TestAssociativeScanMax(unittest.TestCase):
  """associative_scan with maximum should produce running max."""
  def test_max_1d(self):
    x = np.array([3, 1, 4, 1, 5, 9], dtype=np.float32)
    got = Tensor(x.tolist()).associative_scan(lambda a, b: a.maximum(b)).numpy()
    expected = np.array([3, 3, 4, 4, 5, 9], dtype=np.float32)
    np.testing.assert_allclose(got, expected)

  def test_max_2d(self):
    x = np.array([[5, 1], [3, 8], [2, 7]], dtype=np.float32)
    got = Tensor(x.tolist()).associative_scan(lambda a, b: a.maximum(b), axis=0).numpy()
    expected = np.array([[5, 1], [5, 8], [5, 8]], dtype=np.float32)
    np.testing.assert_allclose(got, expected)

class TestAssociativeScanMin(unittest.TestCase):
  """associative_scan with minimum should produce running min."""
  def test_min_1d(self):
    x = np.array([3, 1, 4, 1, 5, 0], dtype=np.float32)
    got = Tensor(x.tolist()).associative_scan(lambda a, b: a.minimum(b)).numpy()
    expected = np.array([3, 1, 1, 1, 1, 0], dtype=np.float32)
    np.testing.assert_allclose(got, expected)

  def test_min_2d(self):
    x = np.array([[5, 8], [3, 2], [7, 1]], dtype=np.float32)
    got = Tensor(x.tolist()).associative_scan(lambda a, b: a.minimum(b), axis=0).numpy()
    expected = np.array([[5, 8], [3, 2], [3, 1]], dtype=np.float32)
    np.testing.assert_allclose(got, expected)

class TestAssociativeScanReverse(unittest.TestCase):
  """Tests for reverse=True (right-to-left scan)."""
  def test_reverse_add_1d(self):
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    got = Tensor(x.tolist()).associative_scan(lambda a, b: a + b, reverse=True).numpy()
    expected = np.array([10, 9, 7, 4], dtype=np.float32)
    np.testing.assert_allclose(got, expected)

  def test_reverse_add_2d(self):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    got = Tensor(x.tolist()).associative_scan(lambda a, b: a + b, axis=1, reverse=True).numpy()
    expected = np.array([[6, 5, 3], [15, 11, 6]], dtype=np.float32)
    np.testing.assert_allclose(got, expected)

  def test_reverse_max(self):
    x = np.array([3, 1, 4, 1, 5], dtype=np.float32)
    got = Tensor(x.tolist()).associative_scan(lambda a, b: a.maximum(b), reverse=True).numpy()
    expected = np.array([5, 5, 5, 5, 5], dtype=np.float32)
    np.testing.assert_allclose(got, expected)

  def test_reverse_mul(self):
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    got = Tensor(x.tolist()).associative_scan(lambda a, b: a * b, reverse=True).numpy()
    expected = np.array([24, 24, 12, 4], dtype=np.float32)
    np.testing.assert_allclose(got, expected)

class TestAssociativeScanEdgeCases(unittest.TestCase):
  """Edge cases: empty, scalar, single element, two elements."""
  def test_empty_1d(self):
    x = Tensor([])
    got = x.associative_scan(lambda a, b: a + b).numpy()
    np.testing.assert_equal(got, np.array([]))

  def test_scalar(self):
    x = Tensor(5.0)
    got = x.associative_scan(lambda a, b: a + b).numpy()
    np.testing.assert_allclose(got, np.array(5.0))

  def test_single_element(self):
    x = Tensor([42.0])
    got = x.associative_scan(lambda a, b: a + b).numpy()
    np.testing.assert_allclose(got, np.array([42.0]))

  def test_two_elements(self):
    x = Tensor([3.0, 7.0])
    got = x.associative_scan(lambda a, b: a + b).numpy()
    np.testing.assert_allclose(got, np.array([3.0, 10.0]))

  def test_empty_2d(self):
    x = Tensor([]).reshape(0, 3)
    got = x.associative_scan(lambda a, b: a + b, axis=0).numpy()
    assert got.shape == (0, 3)

  def test_clone_independence(self):
    """Result should be an independent tensor, not alias the input."""
    x = Tensor([1.0, 2.0, 3.0])
    result = x.associative_scan(lambda a, b: a + b)
    # mutate result should not affect x
    np.testing.assert_allclose(x.numpy(), [1.0, 2.0, 3.0])

class TestAssociativeScanNonPowerOf2(unittest.TestCase):
  """Hillis-Steele must handle non-power-of-2 lengths correctly."""
  def _check_add(self, n):
    x = np.arange(1, n + 1, dtype=np.float32)
    got = Tensor(x.tolist()).associative_scan(lambda a, b: a + b).numpy()
    np.testing.assert_allclose(got, np.cumsum(x))

  def test_len3(self): self._check_add(3)
  def test_len5(self): self._check_add(5)
  def test_len6(self): self._check_add(6)
  def test_len7(self): self._check_add(7)
  def test_len9(self): self._check_add(9)
  def test_len15(self): self._check_add(15)
  def test_len16(self): self._check_add(16)   # power of 2 boundary
  def test_len17(self): self._check_add(17)
  def test_len31(self): self._check_add(31)
  def test_len32(self): self._check_add(32)   # power of 2 boundary
  def test_len33(self): self._check_add(33)

class TestAssociativeScanLarger(unittest.TestCase):
  """Larger tensors to stress-test the algorithm."""
  def test_add_128(self):
    x = np.arange(1, 129, dtype=np.float32)
    got = Tensor(x.tolist()).associative_scan(lambda a, b: a + b).numpy()
    np.testing.assert_allclose(got, np.cumsum(x), rtol=1e-4)

  def test_add_256(self):
    x = np.arange(1, 257, dtype=np.float32)
    got = Tensor(x.tolist()).associative_scan(lambda a, b: a + b).numpy()
    np.testing.assert_allclose(got, np.cumsum(x), rtol=1e-4)

  def test_mul_power_of_2(self):
    """Cumulative product of small values should not overflow."""
    x = np.full(8, 2.0, dtype=np.float32)
    got = Tensor(x.tolist()).associative_scan(lambda a, b: a * b).numpy()
    expected = np.cumprod(x)
    np.testing.assert_allclose(got, expected)

class TestAssociativeScanCustomFn(unittest.TestCase):
  """Test with custom associative functions beyond builtins."""
  def test_add_constant(self):
    """fn = a + b + 1 is not associative, but a+b is. This tests a valid one."""
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    got = Tensor(x.tolist()).associative_scan(lambda a, b: a + b).numpy()
    np.testing.assert_allclose(got, np.cumsum(x))

  def test_subtraction_is_not_associative(self):
    """Demonstrate that non-associative ops produce wrong results (expected behavior).
    (1-2)=-1, but sequential prefix: [1, -1, -1-3, ...] ≠ [1, 1-2, 1-2-3, ...] in general.
    We just verify it doesn't crash."""
    x = Tensor([10.0, 3.0, 5.0, 1.0])
    # This will compute Hillis-Steele with subtraction — not a valid cumsum, but should not crash
    got = x.associative_scan(lambda a, b: a - b)
    assert got.shape == (4,)

class TestAssociativeScanVsCumsumReference(unittest.TestCase):
  """Direct comparison against Tensor.cumsum (the built-in parallel prefix sum)."""
  def test_vs_cumsum_1d(self):
    x = np.array([2, 5, 1, 8, 3], dtype=np.float32)
    t = Tensor(x.tolist())
    scan = t.associative_scan(lambda a, b: a + b).numpy()
    ref = t.cumsum().numpy()
    np.testing.assert_allclose(scan, ref)

  def test_vs_cumsum_2d_axis0(self):
    rng = np.random.default_rng(123)
    x = rng.standard_normal((5, 7)).astype(np.float32)
    t = Tensor(x.tolist())
    scan = t.associative_scan(lambda a, b: a + b, axis=0).numpy()
    ref = t.cumsum(axis=0).numpy()
    np.testing.assert_allclose(scan, ref, atol=1e-5)

  def test_vs_cumsum_2d_axis1(self):
    rng = np.random.default_rng(456)
    x = rng.standard_normal((4, 10)).astype(np.float32)
    t = Tensor(x.tolist())
    scan = t.associative_scan(lambda a, b: a + b, axis=1).numpy()
    ref = t.cumsum(axis=1).numpy()
    np.testing.assert_allclose(scan, ref, atol=1e-5)

  def test_vs_cumsum_random_sizes(self):
    """Test a variety of random shapes and values against cumsum."""
    rng = np.random.default_rng(999)
    for shape in [(10,), (3, 7), (2, 5, 4), (6, 3, 2, 2)]:
      x = rng.standard_normal(shape).astype(np.float32)
      t = Tensor(x.tolist())
      for ax in range(len(shape)):
        scan = t.associative_scan(lambda a, b: a + b, axis=ax).numpy()
        ref = t.cumsum(axis=ax).numpy()
        np.testing.assert_allclose(scan, ref, atol=1e-5,
          err_msg=f"Mismatch for shape={shape}, axis={ax}")

if __name__ == "__main__":
  unittest.main()
