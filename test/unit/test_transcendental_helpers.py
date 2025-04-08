import unittest, math
from typing import Optional
import numpy as np
from tinygrad import dtypes
from tinygrad.dtype import DType
from tinygrad.ops import UOp, Ops
from tinygrad.codegen.transcendental import TRANSCENDENTAL_SUPPORTED_DTYPES, payne_hanek_reduction, cody_waite_reduction, frexp, rintk, pow2if, xpow, xexp2, xlog2, trig_poly
from test.helpers import eval_uop

class TestTranscendentalFunctions(unittest.TestCase):
  def test_payne_hanek_reduction(self):
    # TODO: Test constant input when constant folding is fixed (or maybe test both variants)
    # Load input value from a buffer to prevent constant folding
    input_buf = UOp(Ops.DEFINE_GLOBAL, dtypes.double.ptr(), arg=1, src=())
    loaded_value = UOp.load(input_buf.index(UOp.const(dtypes.int, 0)), dtype=dtypes.double)
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
    r, q = (eval_uop(u) for u in cody_waite_reduction(UOp.const(dtypes.float64, 12 * math.pi + 0.1)))
    np.testing.assert_allclose(r, 0.1)
    np.testing.assert_equal(q, 12)

  def test_frexp(self):
    for x in (1, -1):
      mantissa, exponent = (eval_uop(u) for u in frexp(UOp.const(dtypes.float64, x)))
      np.testing.assert_equal(mantissa, 0.5)
      np.testing.assert_equal(exponent, 1)

    for x in (2, -2):
      mantissa, exponent = (eval_uop(u) for u in frexp(UOp.const(dtypes.float64, 2.0)))
      np.testing.assert_equal(mantissa, 0.5)
      np.testing.assert_equal(exponent, 2)

    mantissa, exponent = (eval_uop(u) for u in frexp(UOp.const(dtypes.float64, 5.0)))
    np.testing.assert_equal(mantissa, 0.625)
    np.testing.assert_equal(exponent, 3)

    mantissa, exponent = (eval_uop(u) for u in frexp(UOp.const(dtypes.float64, 1000.0)))
    np.testing.assert_allclose(mantissa, 0.9765625)
    np.testing.assert_equal(exponent, 10)

  def test_rintk(self):
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, 0.0))), 0)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, 5.0))), 5)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, 5.5))), 6)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, 5.999))), 6)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, -5.0))), -5)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, -5.5))), -6)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, -5.999))), -6)

  def test_pow2if(self):
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, 0), dtypes.float)), 1.0)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, 1), dtypes.float)), 2.0)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, 2), dtypes.float)), 4.0)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, 10), dtypes.float)), 1024.0)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, 63), dtypes.float)), 2**63)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, -1), dtypes.float)), 0.5)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, -2), dtypes.float)), 0.25)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, -10), dtypes.float)), 2**-10)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, -63), dtypes.float)), 2**-63)

# class TestVectorizedTranscendetalFunctions(unittest.TestCase):

#   def _test_vectorization_preserved(self, fxn, in_vec, vcount):
#     out_vec = fxn(in_vec)
#     self.uops_equal(out_vec, cmp_vcount=vcount)

#   def test_vectorization_preserved(self):
#     # given a vectorized input, check that the fxn output is vectorized with the same vcount
#     for dtype_scalar in TRANSCENDENTAL_SUPPORTED_DTYPES:
#       for val in [-2,1.3,194]:
#         for vcount in [1,4,19]:
#           if dtype_scalar == dtypes.float16:
#             continue
#           in_vec = UOp.const(dtype_scalar.vec(vcount), val)
#           self._test_vectorization_preserved(payne_hanek_reduction, in_vec, vcount)
#           self._test_vectorization_preserved(lambda x: xpow(x, x), in_vec, vcount)
#           self._test_vectorization_preserved(xexp2, in_vec, vcount)
#           self._test_vectorization_preserved(cody_waite_reduction, in_vec, vcount)

def uops_equal(u1:UOp|tuple, u2:Optional[UOp|tuple]=None, cmp_op:bool=False, cmp_scalar_dtype:bool|DType=False, cmp_vcount:bool|int=False):
  # instead of comparing u1 to u2, compare u1 to expected op, scalar_dtype, or vcount
  if u2 == None:
    return uops_equal(u1, u1, cmp_op=cmp_op, cmp_scalar_dtype=cmp_scalar_dtype, cmp_vcount=cmp_vcount)
  # compare u1 to u2
  if isinstance(u1, UOp) and isinstance(u2, UOp):
    if cmp_op: assert u1.op == u2.op if isinstance(cmp_op, bool) else u1.op == u2.op == cmp_op, f'ops must match:\n{u1=}\n{u2=}\n{u1.op=}\n{u2.op=}\n{cmp_op=}'
    if cmp_scalar_dtype: assert u1.dtype.scalar() == u2.dtype.scalar() if isinstance(cmp_scalar_dtype, bool) else u1.dtype.scalar() == u2.dtype.scalar() == cmp_scalar_dtype, f'dtype must match:\n{u1=}\n{u2=}\n{u1.dtype.scalar()=}\n{u2.dtype.scalar()=}\n{cmp_scalar_dtype=}'
    if cmp_vcount: assert u1.dtype.vcount == u2.dtype.vcount if isinstance(cmp_vcount, bool) else u1.dtype.vcount == u2.dtype.vcount == cmp_vcount, f'vcount must match:\n{u1=}\n{u2=}\n{u1.dtype.vcount=}\n{u2.dtype.vcount=}\n{cmp_vcount=}'
  # recursive call
  for x1, x2 in zip((u1 if isinstance(u1, tuple) else u1.src), (u2 if isinstance(u2, tuple) else u2.src)):
    uops_equal(x1, x2, cmp_op=cmp_op, cmp_scalar_dtype=cmp_scalar_dtype, cmp_vcount=cmp_vcount)

class TestTranscendetalVectorizationPreserved(unittest.TestCase):

  def _test_vectorization_preserved(self, fxn, scalar_dtypes=TRANSCENDENTAL_SUPPORTED_DTYPES, vals=[-2,1.3,194], vcounts=[1,4,19]):
    # given a vectorized input, check that the fxn output is vectorized with the same vcount
    for scalar_dtype in scalar_dtypes:
      for val in vals:
        for vcount in vcounts:
          in_vec = UOp.const(scalar_dtype.vec(vcount), val)
          out_vec = fxn(in_vec)
          uops_equal(out_vec, cmp_vcount=vcount)

  def test_xpow(self): return self._test_vectorization_preserved(lambda x: xpow(x, x))
  def test_xexp2(self): return self._test_vectorization_preserved(xexp2)
  def test_payne_hanek_reduction(self): return self._test_vectorization_preserved(payne_hanek_reduction)
  def test_cody_waite_reduction(self): return self._test_vectorization_preserved(cody_waite_reduction)


class TestTranscendetalScalarVectorInputs(unittest.TestCase):

  def _test_scalar_vec_equality(self, fxn, scalar_dtypes=TRANSCENDENTAL_SUPPORTED_DTYPES, vals=[-2,1.3,194], vcounts=[1,4,19]):
    # given a scalar and vectorized input, check that the fxn outputs have the same ops, scalar_dtypes, and evaluate to the same value
    # the vectorization stuff (vcount, __eq__) can differ
    for scalar_dtype in scalar_dtypes:
      for val in vals:
        for vcount in vcounts:
          in_scalar, in_vec = UOp.const(scalar_dtype, val), UOp.const(scalar_dtype.vec(vcount), val)
          out_scalar, out_vec = fxn(in_scalar), fxn(in_vec)
          uops_equal(out_scalar, out_vec, cmp_op=True, cmp_scalar_dtype=True)

  def test_xpow(self): return self._test_scalar_vec_equality(lambda x: xpow(x, x))
  def test_xexp2(self): return self._test_scalar_vec_equality(xexp2)
  def test_xlog2(self): return self._test_scalar_vec_equality(xlog2)
  def test_payne_hanek_reduction(self): return self._test_scalar_vec_equality(payne_hanek_reduction)
  def test_cody_waite_reduction(self): return self._test_scalar_vec_equality(cody_waite_reduction)
  def test_trig_poly(self): return self._test_scalar_vec_equality(lambda x: trig_poly(x, [0.1], [0.2]))

if __name__ == '__main__':
  unittest.main()
