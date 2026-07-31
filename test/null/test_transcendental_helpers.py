import unittest, math
import numpy as np
from tinygrad import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.codegen.decomp.transcendental import payne_hanek_reduction, cody_waite_reduction, frexp, rintk, pow2if
from test.helpers import eval_uop

class TestTranscendentalFunctions(unittest.TestCase):
  def test_payne_hanek_reduction(self):
    # TODO: Test constant input when constant folding is fixed (or maybe test both variants)
    # Load input value from a buffer to prevent constant folding
    input_buf = UOp.param(1, dtypes.double, (1,))
    loaded_value = input_buf.index(UOp.const(0, dtypes.int)).load()
    def eval_payne_hanek_reduction(v:float) -> tuple[float, int]:
      return tuple(eval_uop(u, [(dtypes.float64, [v])]) for u in payne_hanek_reduction(loaded_value))

    r, q = eval_payne_hanek_reduction(12 * math.pi + 0.1)
    np.testing.assert_allclose(r, 0.1 - math.pi / 2)
    np.testing.assert_equal(q, 1)

    r, q = eval_payne_hanek_reduction(12 * math.pi)
    np.testing.assert_allclose(r, 0.0, atol=1e-8)
    np.testing.assert_equal(q, 4)

    r, q = eval_payne_hanek_reduction(12 * math.pi - 0.1)
    np.testing.assert_allclose(r, -0.1)
    np.testing.assert_equal(q, 4)

  def test_cody_waite_reduction(self):
    r, q = (eval_uop(u) for u in cody_waite_reduction(UOp.const(12 * math.pi + 0.1, dtypes.float64)))
    np.testing.assert_allclose(r, 0.1)
    np.testing.assert_equal(q, 12)

  def test_frexp(self):
    for x in (1, -1):
      mantissa, exponent = (eval_uop(u) for u in frexp(UOp.const(x, dtypes.float64)))
      np.testing.assert_equal(mantissa, 0.5)
      np.testing.assert_equal(exponent, 1)

    for x in (2, -2):
      mantissa, exponent = (eval_uop(u) for u in frexp(UOp.const(2.0, dtypes.float64)))
      np.testing.assert_equal(mantissa, 0.5)
      np.testing.assert_equal(exponent, 2)

    mantissa, exponent = (eval_uop(u) for u in frexp(UOp.const(5.0, dtypes.float64)))
    np.testing.assert_equal(mantissa, 0.625)
    np.testing.assert_equal(exponent, 3)

    mantissa, exponent = (eval_uop(u) for u in frexp(UOp.const(1000.0, dtypes.float64)))
    np.testing.assert_allclose(mantissa, 0.9765625)
    np.testing.assert_equal(exponent, 10)

  def test_rintk(self):
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(0.0, dtypes.float))), 0)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(5.0, dtypes.float))), 5)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(5.5, dtypes.float))), 6)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(5.999, dtypes.float))), 6)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(-5.0, dtypes.float))), -5)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(-5.5, dtypes.float))), -6)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(-5.999, dtypes.float))), -6)

  def test_pow2if(self):
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(0, dtypes.int), dtypes.float)), 1.0)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(1, dtypes.int), dtypes.float)), 2.0)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(2, dtypes.int), dtypes.float)), 4.0)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(10, dtypes.int), dtypes.float)), 1024.0)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(63, dtypes.int), dtypes.float)), 2**63)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(-1, dtypes.int), dtypes.float)), 0.5)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(-2, dtypes.int), dtypes.float)), 0.25)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(-10, dtypes.int), dtypes.float)), 2**-10)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(-63, dtypes.int), dtypes.float)), 2**-63)

if __name__ == '__main__':
  unittest.main()
