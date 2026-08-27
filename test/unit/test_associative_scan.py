import unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.scan import associative_scan


class TestAssociativeScan(unittest.TestCase):
  def test_add_1d(self):
    x = Tensor([1, 2, 3, 4, 5])
    out = associative_scan(lambda a, b: a+b, x)
    np.testing.assert_equal(out.numpy(), np.array([1, 3, 6, 10, 15]))

  def test_mul_1d(self):
    x = Tensor([1, 2, 3, 4, 5])
    out = associative_scan(lambda a, b: a*b, x)
    np.testing.assert_equal(out.numpy(), np.array([1, 2, 6, 24, 120]))

  def test_axis(self):
    x = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    out = associative_scan(lambda a, b: a+b, x, axis=1)
    np.testing.assert_equal(out.numpy(), np.array([[1, 3, 6, 10], [5, 11, 18, 26]]))

  def test_tuple_affine_recurrence(self):
    # Compose affine maps x -> a*x+b. Composition is associative but not commutative.
    a = Tensor([2., 3., 4., 5.])
    b = Tensor([1., 1., 1., 1.])
    def compose(left, right):
      a0, b0 = left
      a1, b1 = right
      return a1*a0, a1*b0+b1

    out_a, out_b = associative_scan(compose, (a, b))
    np.testing.assert_allclose(out_a.numpy(), np.array([2., 6., 24., 120.]))
    np.testing.assert_allclose(out_b.numpy(), np.array([1., 4., 17., 86.]))

  def test_negative_axis(self):
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    out = associative_scan(lambda a, b: a+b, x, axis=-1)
    np.testing.assert_equal(out.numpy(), np.array([[1, 3, 6], [4, 9, 15]]))

  def test_empty(self):
    x = Tensor.empty(0)
    out = associative_scan(lambda a, b: a+b, x)
    self.assertEqual(out.shape, (0,))


if __name__ == "__main__": unittest.main()
