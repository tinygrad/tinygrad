import unittest
import numpy as np

from tinygrad import Tensor
from tinygrad.scan import associative_scan


class TestAssociativeScan(unittest.TestCase):
  def test_add_non_power_of_two(self):
    x = Tensor.arange(7)
    out = associative_scan(lambda a, b: a + b, x)
    np.testing.assert_equal(out.numpy(), np.arange(7).cumsum())

  def test_mul(self):
    x = Tensor([1, 2, 3, 4], dtype="int32")
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


if __name__ == "__main__": unittest.main()
