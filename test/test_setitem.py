import unittest
from tinygrad import Tensor

import numpy as np

class TestAssign(unittest.TestCase):
  def test_set_value(self):
    n = np.random.random((10, 10, 10)).astype(np.float32)
    t = Tensor(n)
    t[2:5,6:9,:] = 3
    n[2:5,6:9,:] = 3
    np.testing.assert_allclose(t.numpy(), n)