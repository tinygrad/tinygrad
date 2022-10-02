from cbuffer cimport CBuffer
cimport numpy as np
import numpy as np
from typing import Tuple
from tinygrad.helpers import prod
from tinygrad.ops import UnaryOps, BinaryOps, MovementOps, ReduceOps
from tinygrad.shapetracker import ShapeTracker

cdef class RawCPUBuffer:
  cdef CBuffer *buf
  st: ShapeTracker

  def __init__(self, shape, RawCPUBuffer parent=None):
    # TODO: copied from ops gpu, generic this?
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))
    if parent is not None: self.buf = parent.buf
    else: self.buf = new CBuffer(prod(shape))

  @property
  def shape(self): return self.st.shape

  def contiguous_op(RawCPUBuffer x) -> RawCPUBuffer:
    return x if x.st.contiguous else x.unary_op(UnaryOps.NOOP)

  @staticmethod
  def fromCPU(np.ndarray x):
    ret = RawCPUBuffer([x.shape[i] for i in range(x.ndim)])
    ret.buf.copyin(x.data)
    return ret

  def toCPU(RawCPUBuffer self):
    x: RawCPUBuffer
    print("toCPU", self.buf.size, self.st)
    x = self.contiguous_op()
    buf = memoryview(<float[:prod(x.shape)]> x.buf.buf)
    return np.frombuffer(buf, dtype=np.float32).reshape(x.shape)

  # 1 free generic op same as GPU (superclass with shapetracker?)

  def movement_op(RawCPUBuffer x, op, arg): return type(x)(ShapeTracker(x.st).movement_op(op, arg), x)

  # 3 actual ops

  REQUIRES_SIMPLE_REDUCE = True
  def reduce_op(RawCPUBuffer x, op:ReduceOps, new_shape:Tuple[int, ...]): 
    return x

  def unary_op(RawCPUBuffer x, op):
    print(op, x.st)
    return x

  # TODO: shape/strides for x and y combined
  def binary_op(RawCPUBuffer x, op, RawCPUBuffer y):
    print(op, x.st, y.st)
    ret = RawCPUBuffer(x.shape)
    ret.buf = new CBuffer(prod(x.shape))
    if op == BinaryOps.ADD: ret.buf.add(x.buf, y.buf)
    elif op == BinaryOps.MUL: ret.buf.mul(x.buf, y.buf)
    else: raise NotImplementedError()
    # TODO: write binary op in c++
    return ret

  # can all be combined into _processing_op
