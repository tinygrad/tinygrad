import unittest
import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.scan import associative_scan


class TestAssociativeScan(unittest.TestCase):
  def test_add_non_power_of_two(self):
    x = Tensor.arange(7)
    out = associative_scan(lambda a, b: a + b, x)
    np.testing.assert_equal(out.numpy(), np.arange(7).cumsum())

  def test_mul(self):
    x = Tensor([1, 2, 3, 4], dtype=dtypes.int32)
    out = associative_scan(lambda a, b: a * b, x)
    np.testing.assert_equal(out.numpy(), np.array([1, 2, 6, 24], dtype=np.int32))

  def test_axis(self):
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    out = associative_scan(lambda a, b: a + b, x, axis=1)
    np.testing.assert_equal(out.numpy(), np.array([[1, 3, 6], [4, 9, 15]]))

  def test_reverse(self):
    x = Tensor([1, 2, 3, 4])
    out = associative_scan(lambda a, b: a + b, x, reverse=True)
    np.testing.assert_equal(out.numpy(), np.array([10, 9, 7, 4]))

  def test_maximum(self):
    x = Tensor([3, 1, 4, 2, 5])
    out = associative_scan(lambda a, b: a.maximum(b), x)
    np.testing.assert_equal(out.numpy(), np.array([3, 3, 4, 4, 5]))

  def test_reverse_preserves_non_commutative_order(self):
    arr = np.array([[[1., 1.], [0., 1.]], [[2., 0.], [1., 1.]], [[1., 0.], [3., 1.]]], dtype=np.float32)
    out = associative_scan(lambda a, b: a.matmul(b), Tensor(arr), reverse=True).numpy()
    expected = np.empty_like(arr)
    expected[-1] = arr[-1]
    for i in range(len(arr)-2, -1, -1): expected[i] = arr[i] @ expected[i+1]
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__": unittest.main()
