import numpy as np
import unittest
from tinygrad import Tensor

class TestOpMixin(unittest.TestCase):
  def test_uop_cumsum(self):
    np.testing.assert_equal(Tensor(Tensor([1, 2, 3, 4, 5]).uop.cumsum()).numpy(), [1, 3, 6, 10, 15])
    # large case exercises the _split_cumalu path
    np.testing.assert_equal(Tensor(Tensor.ones(600).uop.cumsum()).numpy(), np.arange(1, 601).astype(np.float32))

if __name__ == "__main__":
  unittest.main()
