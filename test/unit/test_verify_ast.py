from __future__ import annotations
import unittest

from tinygrad import Tensor
from tinygrad.codegen.kernel import Kernel
from tinygrad.helpers import DEBUG
from tinygrad.ops import UOp, UOps, ReduceOps, print_uops
from tinygrad.codegen.kernel import verify_ast
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad import dtypes
from tinygrad.shape.view import View

class InvalidASTException(Exception): pass
def helper_test_verify_ast(*stores:UOp) -> Kernel:
  sink = UOp(UOps.SINK, dtypes.void, stores)
  if DEBUG >= 3:
    for op in stores: print(op)
  try: verify_ast(sink)
  except AssertionError as e: raise InvalidASTException(e.args)
  k = Kernel(sink)
  k.linearize()
  if DEBUG >= 6: print_uops(k.uops)
  if DEBUG >= 4: print(k.to_program().src)
  return k

class TestVerifyAST(unittest.TestCase):
  def test_tiny_add(self):
    dtype = dtypes.int
    buf_0 = UOp(UOps.DEFINE_GLOBAL, dtype.ptr(), (), 0)
    buf_1 = UOp(UOps.DEFINE_GLOBAL, dtype.ptr(), (), 1)
    buf_2 = UOp(UOps.DEFINE_GLOBAL, dtype.ptr(), (), 2)
    a = UOp(UOps.LOAD, dtype, (buf_1, ShapeTracker.from_shape((32, 1)).to_uop()))
    b = UOp(UOps.LOAD, dtype, (buf_2, ShapeTracker.from_shape((32, 1)).to_uop()))
    store = UOp(UOps.STORE, dtypes.void, (buf_0, ShapeTracker.from_shape((32, 1)).to_uop(), a+b))
    helper_test_verify_ast(store)

  def test_exactly_one_full_shape(self):
    dtype = dtypes.int
    bufs = [UOp(UOps.DEFINE_GLOBAL, dtype.ptr(), (), i) for i in range(6)]
    a = UOp(UOps.LOAD, dtype, (bufs[2], ShapeTracker.from_shape((32, 1)).to_uop()))
    b = UOp(UOps.LOAD, dtype, (bufs[3], ShapeTracker.from_shape((32, 1)).to_uop()))
    st0 = UOp.store(bufs[0], ShapeTracker.from_shape((32, 1)).to_uop(), a+b)
    a = UOp(UOps.LOAD, dtype, (bufs[4], ShapeTracker.from_shape((32, 32)).to_uop()))
    b = UOp(UOps.LOAD, dtype, (bufs[5], ShapeTracker.from_shape((32, 32)).to_uop()))
    st1 = UOp.store(bufs[1], ShapeTracker.from_shape((32, 32)).to_uop(), a+b)
    with self.assertRaises(InvalidASTException): helper_test_verify_ast(st0, st1)

  def test_no_implicit_broadcasting(self):
    bufs = [UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), (), i) for i in range(2)]
    a = UOp(UOps.LOAD, dtypes.float, (bufs[1], ShapeTracker.from_shape((4, 32)).to_uop()))
    b = a + UOp(UOps.REDUCE_AXIS, dtypes.float, (a,), (ReduceOps.MAX, (1,)))
    st = UOp(UOps.STORE, dtypes.void, (bufs[0], ShapeTracker.from_shape((4, 32)).to_uop(), b))
    with self.assertRaises(InvalidASTException): helper_test_verify_ast(st)

  def test_shrink_ok(self):
    bufs = [UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), (), i) for i in range(2)]
    a = UOp(UOps.LOAD, dtypes.float, (bufs[1], ShapeTracker((View((32, 32), strides=(32, 1), offset=0, mask=None, contiguous=True),)).to_uop()))
    b = UOp(UOps.LOAD, dtypes.float, (bufs[1], ShapeTracker((View((32, 32), strides=(0, 1), offset=0, mask=None, contiguous=False),)).to_uop()))
    st = UOp.store(bufs[0], ShapeTracker.from_shape((32, 32)).to_uop(), a+b)
    helper_test_verify_ast(st)

  def test_reduce_store(self):
    bufs = [UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), (), i) for i in range(2)]
    a = UOp(UOps.LOAD, dtypes.float, (bufs[1], ShapeTracker.from_shape((32, 1)).to_uop()))
    r = UOp(UOps.REDUCE_AXIS, dtypes.float, (a,), (ReduceOps.SUM, (0,)))
    st = UOp.store(bufs[0], ShapeTracker.from_shape((32, 1)).to_uop(), r)
    with self.assertRaisesRegex(InvalidASTException, "implicit expand"): helper_test_verify_ast(st)

  def test_reduce_add_store(self):
    bufs = [UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), (), i) for i in range(2)]
    a = UOp(UOps.LOAD, dtypes.float, (bufs[1], ShapeTracker.from_shape((32, 1)).to_uop()))
    r = UOp(UOps.REDUCE_AXIS, dtypes.float, (a,), (ReduceOps.SUM, (0,)))
    st = UOp.store(bufs[0], ShapeTracker.from_shape((32, 1)).to_uop(), r+a)
    with self.assertRaisesRegex(InvalidASTException, "implicit expand"): helper_test_verify_ast(st)

  def test_buffer_uops_st(self):
    a = Tensor.randn(4, 4)+2
    uop_sts = verify_ast(a.schedule()[-1].ast)
    store_st = [st for u,st in uop_sts.items() if u.op is UOps.STORE][0]
    self.assertEqual(store_st, ShapeTracker.from_shape((4, 4)))
    const_st = [st for u,st in uop_sts.items() if u.op is UOps.VALID][0]
    self.assertEqual(const_st, ShapeTracker.from_shape((1, 1)).expand((4, 4)))

if __name__ == '__main__':
  unittest.main()
