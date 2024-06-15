import unittest, operator, math
import numpy as np
import torch
from tinygrad import Tensor

def _test_binary(op, rawa, rawb):
  # ground truth
  py = op(rawa, rawb)
  for a in (rawa, [rawa]):
    for b in (rawb, [rawb]):
      np.testing.assert_allclose(op(np.array(a), np.array(b)).item(), py)
      np.testing.assert_allclose(op(torch.tensor(a), torch.tensor(b)).item(), py)
      np.testing.assert_allclose(op(Tensor(a), Tensor(b)).item(), py)

class TestMul(unittest.TestCase):
  def test_mul_zero(self):
    _test_binary(operator.mul, 0, math.inf)

if __name__ == '__main__':
  unittest.main()