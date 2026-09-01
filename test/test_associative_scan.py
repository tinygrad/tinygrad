
import unittest
import numpy as np

from tinygrad import Tensor, associative_scan

class TestAssociativeScan(unittest.TestCase):
  def test_add_axis(self):
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    self.assertEqual(associative_scan(lambda a,b: a+b, x, axis=1).tolist(), [[1, 3, 6], [4, 9, 15]])

  def test_matrix_order_forward_and_reverse(self):
    values = np.array([[[1, 1], [0, 1]], [[1, 0], [1, 1]], [[2, 0], [0, 1]]], dtype=np.float32)
    forward = np.empty_like(values)
    forward[0] = values[0]
    for i in range(1, len(values)): forward[i] = forward[i-1] @ values[i]
    reverse = np.empty_like(values)
    reverse[-1] = values[-1]
    for i in range(len(values)-2, -1, -1): reverse[i] = values[i] @ reverse[i+1]
    np.testing.assert_allclose(associative_scan(lambda a,b: a@b, Tensor(values)).numpy(), forward)
    np.testing.assert_allclose(associative_scan(lambda a,b: a@b, Tensor(values), reverse=True).numpy(), reverse)

  def test_tuple_affine_recurrence(self):
    # Compose h -> a*h+b transforms, matching the tuple state used by parallel SSM-style recurrences.
    a, b = Tensor([2, 3, 4]), Tensor([1, 1, 1])
    def compose(x, y):
      a1, b1 = x
      a2, b2 = y
      return a2*a1, a2*b1+b2
    out_a, out_b = associative_scan(compose, (a, b))
    self.assertEqual(out_a.tolist(), [2, 6, 24])
    self.assertEqual(out_b.tolist(), [1, 4, 17])

  def test_negative_axis_and_empty(self):
    x = Tensor([[1, 2], [3, 4]])
    self.assertEqual(associative_scan(lambda a,b: a+b, x, axis=-1).tolist(), [[1, 3], [3, 7]])
    self.assertEqual(associative_scan(lambda a,b: a+b, Tensor([])).shape, (0,))

if __name__ == "__main__": unittest.main()
