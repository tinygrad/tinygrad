import unittest
import numpy as np
from tinygrad import Tensor

class TestMaskedReductions(unittest.TestCase):
  def test_masked_sums(self):
    r = Tensor.arange(8).reshape(8, 1).expand(8, 8)
    for name, mask in (
      ('affine', lambda r,i: ((r+i+2) == 5) & (r % 2 == 0)),
      ('interval', lambda r,i: r+i >= 13),
      ('negative_numerator', lambda r,i: (r-i) % 8 < 1),
      ('multiple_hits', lambda r,i: (r+i) % 7 < 1),
      ('no_hits', lambda r,i: (r+i+1) % 32 < 1),
      ('conditional_hit', lambda r,i: (r+i) % 32 < 1),
      ('negative_modulus', lambda r,i: (r+i) % -7 < 1),
      ('nonlinear', lambda r,i: r == r*r),
    ):
      with self.subTest(mask=name):
        actual = mask(r, r.T).where(r*r+1, 0).sum(axis=0).numpy()
        expected = [sum(k*k+1 for k in range(8) if mask(k, j)) for j in range(8)]
        np.testing.assert_array_equal(actual, expected)

if __name__ == '__main__': unittest.main()
