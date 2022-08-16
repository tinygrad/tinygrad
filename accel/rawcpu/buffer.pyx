from cbuffer cimport CBuffer
cimport numpy as np
import numpy as np
from tinygrad.helpers import prod
from tinygrad.ops import BinaryOps, MovementOps
from tinygrad.shapetracker import ShapeTracker

cdef class RawCPUBuffer:
  cdef CBuffer *buf
  st: ShapeTracker

  def __init__(self, shape):
    # TODO: copied from ops gpu, generic this?
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))

  @property
  def shape(self): return self.st.shape

  @staticmethod
  def fromCPU(np.ndarray x):
    ret = RawCPUBuffer([x.shape[i] for i in range(x.ndim)])
    ret.buf = new CBuffer(x.size, x.data)
    return ret

  def toCPU(RawCPUBuffer self):
    print("toCPU", self.buf.size, self.st)
    buf = memoryview(<float[:prod(self.shape)]> self.buf.buf)
    return np.frombuffer(buf, dtype=np.float32).reshape(self.shape)

  def movement_op(RawCPUBuffer x, op, arg):
    ret = RawCPUBuffer(ShapeTracker(x.st).movement_op(op, arg))
    ret.buf = x.buf
    return ret

  def binary_op(RawCPUBuffer x, op, RawCPUBuffer y):
    print(op, x.st, y.st)
    ret = RawCPUBuffer(x.shape)
    ret.buf = new CBuffer(prod(x.shape))
    if op == BinaryOps.ADD: ret.buf.add(x.buf, y.buf)
    elif op == BinaryOps.MUL: ret.buf.mul(x.buf, y.buf)
    else: raise NotImplementedError()
    # TODO: write binary op in c++
    return ret

