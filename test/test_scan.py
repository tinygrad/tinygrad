import numpy as np
import unittest
from tinygrad import Tensor

class TestScan(unittest.TestCase):
  def test_scan_add(self):
    a = Tensor.randn(10, 10).realize()
    init = Tensor.zeros(10, 1)
    a_red = (a+init).scan(init)
    np.testing.assert_allclose(a_red.numpy(), a.numpy().sum(axis=1))

if __name__ == '__main__':
  unittest.main()
