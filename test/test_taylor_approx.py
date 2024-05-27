import unittest
import numpy as np
from tinygrad import  Tensor

class TestSymbolicOps(unittest.TestCase):
  def test_exp_about_zero(self):
    _a = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    a = Tensor(_a)
    a = a.exp()
    np.testing.assert_allclose(np.exp(_a), a.numpy())

if __name__ == '__main__':
  unittest.main()