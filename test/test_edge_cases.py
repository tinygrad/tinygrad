import unittest
import numpy as np
import torch
from tinygrad import Tensor

class TestEdgeCases(unittest.TestCase):
  @unittest.expectedFailure
  def test_split_negative_size(self):
    a = Tensor.arange(10)
    with self.assertRaises(ValueError):
      a.split([-1, 11])

  @unittest.expectedFailure
  def test_circular_pad_negative(self):
    x = Tensor.arange(9).reshape(1, 1, 3, 3)
    out = x.pad((0, -1, 1, 2), mode="circular")
    torch_out = torch.nn.functional.pad(torch.arange(9).reshape(1, 1, 3, 3), (0, -1, 1, 2), mode="circular")
    np.testing.assert_allclose(out.numpy(), torch_out.numpy())

  @unittest.expectedFailure
  def test_sort_empty(self):
    t = Tensor([])
    values, indices = t.sort()
    np.testing.assert_equal(values.numpy(), np.array([]))
    np.testing.assert_equal(indices.numpy(), np.array([], dtype=np.int32))

if __name__ == "__main__":
  unittest.main()
