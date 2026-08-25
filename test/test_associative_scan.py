import unittest
import numpy as np

from tinygrad import Tensor, associative_scan


class TestAssociativeScan(unittest.TestCase):
  def test_add_1d(self):
    x = Tensor([1, 2, 3, 4, 5])
    np.testing.assert_equal(associative_scan(lambda a,b: a+b, x).numpy(), [1, 3, 6, 10, 15])

  def test_add_axis_1(self):
    x = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    expected = np.cumsum(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), axis=1)
    np.testing.assert_equal(associative_scan(lambda a,b: a+b, x, axis=1).numpy(), expected)

  def test_max(self):
    x = Tensor([3, 1, 4, 2, 8, 5])
    expected = np.maximum.accumulate(np.array([3, 1, 4, 2, 8, 5]))
    np.testing.assert_equal(associative_scan(lambda a,b: a.maximum(b), x).numpy(), expected)

  def test_mul(self):
    x = Tensor([1, 2, 3, 4])
    np.testing.assert_equal(associative_scan(lambda a,b: a*b, x).numpy(), [1, 2, 6, 24])

  def test_reverse(self):
    x = Tensor([1, 2, 3, 4])
    np.testing.assert_equal(associative_scan(lambda a,b: a+b, x, reverse=True).numpy(), [10, 9, 7, 4])

  def test_negative_axis(self):
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    expected = np.cumsum(np.array([[1, 2, 3], [4, 5, 6]]), axis=-1)
    np.testing.assert_equal(associative_scan(lambda a,b: a+b, x, axis=-1).numpy(), expected)

  def test_noncommutative_matrix_product(self):
    values = np.array([
      [[1, 2], [0, 1]],
      [[2, 0], [1, 3]],
      [[1, 1], [2, 1]],
    ], dtype=np.float32)

    forward = np.empty_like(values)
    forward[0] = values[0]
    for i in range(1, len(values)): forward[i] = forward[i-1] @ values[i]
    np.testing.assert_allclose(associative_scan(lambda a,b: a@b, Tensor(values)).numpy(), forward)

    reverse = np.empty_like(values)
    reverse[-1] = values[-1]
    for i in range(len(values)-2, -1, -1): reverse[i] = values[i] @ reverse[i+1]
    np.testing.assert_allclose(associative_scan(lambda a,b: a@b, Tensor(values), reverse=True).numpy(), reverse)


if __name__ == "__main__": unittest.main()
