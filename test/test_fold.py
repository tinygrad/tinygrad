import numpy as np
import unittest
from tinygrad import Tensor

class TestScan(unittest.TestCase):
  def test_reduce_add(self):
    a = Tensor.randn(10, 10).realize()
    a_red = a.sum(axis=1)
    np.testing.assert_allclose(a_red.numpy(), a.numpy().sum(axis=1), atol=1e-6)

  def test_fold_add(self):
    a = Tensor.randn(10, 10).realize()
    init = Tensor.zeros(10, 1).contiguous()
    a_red = (a+init).fold(init).reshape(10)
    np.testing.assert_allclose(a_red.numpy(), a.numpy().sum(axis=1), atol=1e-6)

if __name__ == '__main__':
  unittest.main()
