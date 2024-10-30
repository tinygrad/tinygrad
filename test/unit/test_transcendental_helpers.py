import unittest, math
import numpy as np
from tinygrad import dtypes
from tinygrad.ops import UOp
from tinygrad.codegen.transcendental import payne_hanek_reduction, cody_waite_reduction, frexp, rintk
from test.helpers import eval_uop

class TestTranscendentalFunctions(unittest.TestCase):
  def test_payne_hanek_reduction(self):
    r, q = (eval_uop(u) for u in payne_hanek_reduction(UOp.const(dtypes.float64, 12 * math.pi + 0.1)))
    # TODO: should r be in [0, pi/2) per doc?
    np.testing.assert_allclose(r, 0.1 - math.pi / 2)
    np.testing.assert_equal(q, 1)

  def test_cody_waite_reduction(self):
    r, q = (eval_uop(u) for u in cody_waite_reduction(UOp.const(dtypes.float64, 12 * math.pi + 0.1)))
    np.testing.assert_allclose(r, 0.1)
    # TODO: should q be in [0, 1, 2, 3]?
    np.testing.assert_equal(q, 12)

  def test_frexp(self):
    mantissa, exponent = (eval_uop(u) for u in frexp(UOp.const(dtypes.float64, 0.0)))
    np.testing.assert_equal(mantissa, 0.0)
    np.testing.assert_equal(exponent, 0)

    mantissa, exponent = (eval_uop(u) for u in frexp(UOp.const(dtypes.float64, 1.0)))
    np.testing.assert_equal(mantissa, 0.5)
    np.testing.assert_equal(exponent, 1)

    mantissa, exponent = (eval_uop(u) for u in frexp(UOp.const(dtypes.float64, -1.0)))
    np.testing.assert_equal(mantissa, 0.5)
    np.testing.assert_equal(exponent, 1)

    mantissa, exponent = (eval_uop(u) for u in frexp(UOp.const(dtypes.float64, 2.0)))
    np.testing.assert_equal(mantissa, 0.5)
    np.testing.assert_equal(exponent, 2)

    mantissa, exponent = (eval_uop(u) for u in frexp(UOp.const(dtypes.float64, 5.0)))
    np.testing.assert_equal(mantissa, 0.625)
    np.testing.assert_equal(exponent, 3)

  def test_rintk(self):
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, 0.0))), 0)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, 5.0))), 5)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, 5.5))), 6)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, 5.999))), 6)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, -5.0))), -5)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, -5.5))), -6)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, -5.999))), -6)

if __name__ == '__main__':
  unittest.main()
