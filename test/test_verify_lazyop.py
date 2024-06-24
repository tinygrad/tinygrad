from __future__ import annotations
import unittest
from tinygrad.codegen.linearizer import Linearizer
#from tinygrad.codegen.lowerer import Lowerer
from tinygrad.engine.graph import print_tree
from tinygrad.helpers import DEBUG
from tinygrad.ops import BinaryOps, BufferOps, MemBuffer, LazyOp, ReduceOps, verify_lazyop
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad import dtypes
from tinygrad.shape.view import View

class LazyOp(LazyOp):
  def __add__(self, other:LazyOp): return LazyOp(BinaryOps.ADD, (self, other))

class InvalidLazyOpException(Exception): pass
def lower(*ast:LazyOp):
  if DEBUG >= 3:
    for op in ast: print_tree(op)
  try: verify_lazyop(*ast)
  except AssertionError: raise InvalidLazyOpException()
  k = Linearizer(*ast)
  k.linearize()
  if DEBUG >= 6: k.uops.print()
  if DEBUG >= 4: print(k.to_program().src)
  return k

class TestVerifyLazyOp(unittest.TestCase):
  def test_tiny_add(self):
    dtype = dtypes.int
    st = ShapeTracker.from_shape((32, 1))
    a = LazyOp(BufferOps.LOAD, arg=MemBuffer(1, dtype, st))
    b = LazyOp(BufferOps.LOAD, arg=MemBuffer(2, dtype, st))
    out = LazyOp(BufferOps.STORE, (a+b, ), arg=MemBuffer(0, dtype, st))
    lower(out)

  def test_exactly_one_full_shape(self):
    a = LazyOp(BufferOps.LOAD, arg=MemBuffer(1, dtypes.int, ShapeTracker.from_shape((32, 1))))
    b = LazyOp(BufferOps.LOAD, arg=MemBuffer(2, dtypes.int, ShapeTracker.from_shape((32, 1))))
    out0 = LazyOp(BufferOps.STORE, (a+b, ), MemBuffer(0, dtypes.int, ShapeTracker.from_shape((32, 1))))
    c = LazyOp(BufferOps.LOAD, arg=MemBuffer(3, dtypes.int, ShapeTracker.from_shape((32, 32))))
    d = LazyOp(BufferOps.LOAD, arg=MemBuffer(4, dtypes.int, ShapeTracker.from_shape((32, 32))))
    out1 = LazyOp(BufferOps.STORE, (c+d, ), MemBuffer(0, dtypes.int, ShapeTracker.from_shape((32, 32))))
    with self.assertRaises(InvalidLazyOpException): lower(out0, out1)

  def test_no_implicit_broadcasting(self):
    t = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.float, ShapeTracker.from_shape((4, 32))))
    b  = t + LazyOp(ReduceOps.MAX, (t, ), (1, ))
    out = LazyOp(BufferOps.STORE, (b, ), MemBuffer(0, dtypes.float, ShapeTracker.from_shape((4, 32))))
    with self.assertRaises(InvalidLazyOpException): lower(out)

  def test_shrink_ok(self):
    a = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.float, ShapeTracker((View((32, 32), strides=(32, 1), offset=0, mask=None, contiguous=True),))))
    b = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.float, ShapeTracker((View((32, 32), strides=(0, 1), offset=0, mask=None, contiguous=False),))))
    out = LazyOp(BufferOps.STORE, (a+b, ), MemBuffer(0, dtypes.float, ShapeTracker.from_shape((32, 32))))
    lower(out)

if __name__ == '__main__':
  unittest.main()
