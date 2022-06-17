from tinygrad.shapetracker import ShapeTracker
import numpy as np

# TODO: these aren't really ops
from tinygrad.ops import log_op

class LazyBuffer:
  def __init__(self, shape, hostbuf=None):
    self.shape, self.hostbuf = shape, hostbuf
    self.dtype = np.float32

  def op(self, op, inp, arg=None):
    log_op(op, self, inp)
    return self

  @staticmethod
  def fromCPU(x):
    return LazyBuffer(x.shape, x)

  def toCPU(self):
    # this realizes the tensor 
    return np.zeros(self.shape, self.dtype)

class Ops:
  def unary_op(ctx, op, x): return LazyBuffer(x.shape).op(op, [x])
  def binary_op(ctx, op, x, y): return LazyBuffer(x.shape).op(op, [x,y])
  def reduce_op(ctx, op, x, new_shape): return LazyBuffer(new_shape).op(op, [x], new_shape)
  def movement_op(ctx, op, x, arg): return LazyBuffer(ShapeTracker(*x.shape).movement_op(op, arg).shape).op(op, [x], arg)
  def processing_op(ctx, op, x, w, C): return LazyBuffer((C.bs, C.cout, C.oy, C.ox)).op(op, [x,w], C)
