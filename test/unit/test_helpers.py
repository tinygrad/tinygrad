import unittest
import numpy as np
from tinygrad.helpers import polyN
from tinygrad.tensor import Tensor

class TestPolyN(unittest.TestCase):
  def test_tensor(self):
    np.testing.assert_allclose(polyN(Tensor([1.0, 2.0, 3.0, 4.0]), [1.0, -2.0, 1.0]).numpy(), [0.0, 1.0, 4.0, 9.0])

if __name__ == '__main__':
  unittest.main()
