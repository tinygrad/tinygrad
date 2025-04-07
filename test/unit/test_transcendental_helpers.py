import unittest, math
from typing import Optional
import numpy as np
from tinygrad import dtypes
from tinygrad.dtype import DType
from tinygrad.ops import UOp, Ops
from tinygrad.codegen.transcendental import TRANSCENDENTAL_SUPPORTED_DTYPES, payne_hanek_reduction, cody_waite_reduction, frexp, rintk, pow2if, xpow
from tinygrad.codegen.transcendental import _ifand, shl, shr, ilogb2k, sin_poly, xsin, xexp2, xlog2, ldexp3k, sin_poly_large, sin_poly_small, ldexp2k, trig_poly, _lazy_map_numbers
from test.helpers import eval_uop
from tinygrad.helpers import iterable
from icecream import ic

def uops_equal(u1:UOp|tuple, u2:Optional[UOp|tuple]=None, cmp_op:bool=False, cmp_scalar_dtype:bool|DType=False, cmp_vcount:bool|int=False, cmp_eval:bool=True, cmp_eq:bool=False):
  # compare u1 to expected op, scalar_dtype, or vcount
  if u2 == None:
    return uops_equal(u1, u1, cmp_op=cmp_op, cmp_scalar_dtype=cmp_scalar_dtype, cmp_vcount=cmp_vcount, cmp_eval=cmp_eval, cmp_eq=cmp_eq)
  # compare u1, u2
  if isinstance(u1, UOp) and isinstance(u2, UOp):
    if cmp_op: assert u1.op == u2.op == cmp_op, f'ops must match:\n{u1.op=}\n{u2.op=}\n{cmp_op=}'
    if cmp_scalar_dtype: assert u1.dtype.scalar() == u2.dtype.scalar() == cmp_scalar_dtype, f'dtype must match:\n{u1.dtype.scalar()=}\n{u2.dtype.scalar()=}\n{cmp_scalar_dtype=}'
    if cmp_vcount: assert u1.dtype.vcount == u2.dtype.vcount == cmp_vcount, f'vcount must match:\n{u1.dtype.vcount=}\n{u2.dtype.vcount=}\n{cmp_vcount=}'
    if cmp_eval: assert eval_uop(u1) == eval_uop(u2), f'eval must match:\n{eval_uop(u1)=}\n{eval_uop(u2)=}'
    if cmp_eq: assert u1 == u2, f'equality must match:\n{u1=}\n{u2=}'
  # recursive call
  for x1, x2 in zip((u1 if isinstance(u1, tuple) else u1.src), (u2 if isinstance(u2, tuple) else u2.src)):
    uops_equal(x1, x2, cmp_op=cmp_op, cmp_scalar_dtype=cmp_scalar_dtype, cmp_vcount=cmp_vcount, cmp_eval=cmp_eval, cmp_eq=cmp_eq)

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

class TestVectorizedTranscendetalFunctions(unittest.TestCase):
  def _check_all_uops_vectorized(self, u:tuple|UOp, vcount:int):
    # check all UOps in u are vectorized with vcount
    if isinstance(u, UOp): assert u.dtype.vcount == vcount, f'expected {vcount=} but got {u.dtype.vcount=} for UOp {u=}'
    [self._check_all_uops_vectorized(x, vcount) for x in (u if isinstance(u, tuple) else u.src)]

  def _get_inputs(self, mode:str='floats') -> tuple[UOp, DType]:
    for val in [-2,1.3,194]:
      for vcount in [1,2,4,19]:
        for _dtype in TRANSCENDENTAL_SUPPORTED_DTYPES if mode == 'floats' else (dtypes.int64, dtypes.int32, dtypes.int16):
          dtype: DType = _dtype.vec(vcount)
          d = UOp.const(dtype, val)
          yield d, dtype

  def test_preserves_vectorization(self):
    # verify that when given a vectorized (or scalar) input, the function returns a vectorized (or scalar) output
    for d, dtype in self._get_inputs():
      uops_equal(payne_hanek_reduction(d), cmp_eval=False, cmp_vcount=dtype.vcount)
      uops_equal(cody_waite_reduction(d), cmp_eval=False, cmp_vcount=dtype.vcount)
      uops_equal(xpow(d, d), cmp_eval=False, cmp_vcount=dtype.vcount)
      uops_equal(xexp2(d), cmp_eval=False, cmp_vcount=dtype.vcount)

  def test_match(self):

    def check_scalar_vec_equality(fxn, dtype, *args, vec_size=2, val=0.1, **kwargs):
      in_scalar, in_vec = UOp.const(dtype, val), UOp.const(dtype.vec(vec_size), val)
      out_scalar, out_vec = fxn(in_scalar, *args, **kwargs), fxn(in_vec, *args, **kwargs)
      # fxn outputs must match exactly except for vectorization stuff (vcount, __eq__)
      uops_equal(out_scalar, out_vec, cmp_op=True, cmp_scalar_dtype=True, cmp_eval=True)

    # test functions on vec and scalar versions of the same input
    check_scalar_vec_equality(trig_poly, dtypes.float32, [0.1], [0.2])
    check_scalar_vec_equality(trig_poly, dtypes.float64, [0.1], [0.2])
    check_scalar_vec_equality(cody_waite_reduction, dtypes.float32)
    check_scalar_vec_equality(cody_waite_reduction, dtypes.float64)

    # coeff32, coeff64 = [0.1], [0.2]
    # in_64_scalar, in_64_vec = UOp.const(dtypes.float64, 0.1), UOp.const(dtypes.float64.vec(2), 0.1)
    # assert eval_uop(trig_poly(in_64_scalar, coeff32, coeff64)) == eval_uop(trig_poly(in_64_vec, coeff32, coeff64))

    # assert eval_uop(cody_waite_reduction)



if __name__ == '__main__':
  unittest.main()
