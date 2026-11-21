import unittest, numpy as np
from tinygrad import Tensor

class TestQuantizedMath(unittest.TestCase):
  def test_int_sum(self):
    # a float sum of products of ints sums in int32: exact past the 24 bits of a float
    x, w = np.random.randint(0, 256, (8, 4096)), np.random.randint(0, 256, 4096)
    out = ((Tensor(x.astype(np.uint8)).int()-3).float() * (Tensor(w.astype(np.uint8)).int()-7).float()).sum(1)
    np.testing.assert_equal(out.numpy(), ((x-3)*(w-7)).sum(1).astype(np.float32))

if __name__ == '__main__':
  unittest.main()
