from __future__ import annotations
import unittest
from tinygrad.codegen.linearizer import Linearizer
#from tinygrad.codegen.lowerer import Lowerer
from tinygrad.engine.graph import print_tree
from tinygrad.helpers import DEBUG
from tinygrad.ops import BufferOps, MemBuffer, LazyOp, verify_lazyop
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad import dtypes

def lower(*ast:LazyOp):
  if DEBUG >= 3:
    for op in ast: print_tree(op)
  verify_lazyop(*ast)
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

  def test_one_full_shape(self):
    a = LazyOp(BufferOps.LOAD, arg=MemBuffer(1, dtypes.int, ShapeTracker.from_shape((32, 1))))
    b = LazyOp(BufferOps.LOAD, arg=MemBuffer(2, dtypes.int, ShapeTracker.from_shape((32, 1))))
    out0 = LazyOp(BufferOps.STORE, (a+b, ), MemBuffer(0, dtypes.int, ShapeTracker.from_shape((32, 1))))
    c = LazyOp(BufferOps.LOAD, arg=MemBuffer(3, dtypes.int, ShapeTracker.from_shape((32, 32))))
    d = LazyOp(BufferOps.LOAD, arg=MemBuffer(4, dtypes.int, ShapeTracker.from_shape((32, 32))))
    out1 = LazyOp(BufferOps.STORE, (c+d, ), MemBuffer(0, dtypes.int, ShapeTracker.from_shape((32, 32))))
    with self.assertRaises(AssertionError): lower(out0, out1)

  def test_no_implicit_broadcasting(self):
    pass

  def test_no_implicit_movementops(self):
    pass

  def test_shrink_ok(self):
    pass

  def test_expand_not_ok(self):
    pass

  def test_unsafe_pad_not_ok(self):
    pass

if __name__ == '__main__':
  unittest.main()
