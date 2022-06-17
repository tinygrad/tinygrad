from tinygrad.shapetracker import ShapeTracker
import numpy as np

# TODO: these aren't really ops
from accel.opencl.ops_opencl import preprocessing_op, postprocessing_op

class LazyBuffer:
  def __init__(self, shape, hostbuf=None):
    self.shape, self.hostbuf = shape, hostbuf
    self.dtype = np.float32

  @staticmethod
  def fromCPU(x):
    return LazyBuffer(x.shape, x)

  def toCPU(self):
    # this realizes the tensor 
    pass

class Ops:
  buffer = LazyBuffer

  def unary_op(ctx, op, x):
    ret = LazyBuffer(x.shape)
    return ret

  def binary_op(ctx, op, x, y):
    ret = LazyBuffer(x.shape)
    return ret

  def reduce_op(ctx, op, x, new_shape):
    ret = LazyBuffer(new_shape)
    return ret

  def movement_op(ctx, op, x, arg):
    ret = LazyBuffer(ShapeTracker(*x.shape).movement_op(op, arg).shape)
    return ret

  def processing_op(ctx,op,x,w,out_shape,C):
    ret = LazyBuffer(out_shape)
    return ret
