import unittest
import numpy as np
from tinygrad.helpers import polyN

class TestPolyN(unittest.TestCase):
  def test_float(self):
    np.testing.assert_allclose(polyN(1.0, [1.0, -2.0, 1.0]), 0.0)
    np.testing.assert_allclose(polyN(2.0, [1.0, -2.0, 1.0]), 1.0)
    np.testing.assert_allclose(polyN(3.0, [1.0, -2.0, 1.0]), 4.0)
    np.testing.assert_allclose(polyN(4.0, [1.0, -2.0, 1.0]), 9.0)

  def test_tensor(self):
    from tinygrad.tensor import Tensor
    np.testing.assert_allclose(polyN(Tensor([1.0, 2.0, 3.0, 4.0]), [1.0, -2.0, 1.0]).numpy(), [0.0, 1.0, 4.0, 9.0])

  def test_uop(self):
    from tinygrad.dtype import dtypes
    from tinygrad.uop.ops import UOp
    from test.helpers import eval_uop
    np.testing.assert_allclose(eval_uop(polyN(UOp.const(dtypes.float, 1.0), [1.0, -2.0, 1.0])), 0.0)
    np.testing.assert_allclose(eval_uop(polyN(UOp.const(dtypes.float, 2.0), [1.0, -2.0, 1.0])), 1.0)
    np.testing.assert_allclose(eval_uop(polyN(UOp.const(dtypes.float, 3.0), [1.0, -2.0, 1.0])), 4.0)
    np.testing.assert_allclose(eval_uop(polyN(UOp.const(dtypes.float, 4.0), [1.0, -2.0, 1.0])), 9.0)

if __name__ == '__main__':
  unittest.main()
