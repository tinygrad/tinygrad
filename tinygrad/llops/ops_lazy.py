from matplotlib.pyplot import isinteractive
from tinygrad.shapetracker import ShapeTracker
import functools
import numpy as np
import sys
sys.setrecursionlimit(10000)

# TODO: these aren't really ops
from tinygrad.ops import BinaryOps, MovementOps, UnaryOps, log_op

class LazyBuffer:
  def __init__(self, shape, op=None, src=[], arg=None, hostbuf=None):
    self.shapetracker, self.hostbuf = ShapeTracker(*shape), hostbuf
    self.dtype = np.float32
    self.op, self.src, self.arg = op, src, arg
    self.did_realize = False

  @property
  def shape(self): return self.shapetracker.shape

  def realize(self):
    if self.did_realize or self.op is None: return self
    self.did_realize = True
    srcs = [s.realize() for s in self.src]
    # TODO: do real op here
    log_op(self.op, self, srcs)
    return self

  @staticmethod
  def fromCPU(x):
    return LazyBuffer(x.shape, hostbuf=x)

  def toCPU(self):
    self.realize()
    # this realizes the tensor 
    return np.zeros(self.shape, self.dtype)

  def unary_op(self, op):
    return LazyBuffer(self.shape, op, [self])
  
  def binary_op(self, op, y):
    return LazyBuffer(self.shape, op, [self, y])

  def reduce_op(self, op, new_shape):
    return LazyBuffer(new_shape, op, [self], new_shape)

  @functools.lru_cache()
  def movement_op(self, op, arg):
    # if we got a movement op, push it above any UnaryOp/BinaryOp
    if isinstance(self.op, BinaryOps) or isinstance(self.op, UnaryOps):
      src = [x.movement_op(op, arg) for x in self.src]
      return LazyBuffer(src[0].shape, self.op, src)
    
    st = ShapeTracker(*self.shape).movement_op(op, arg)
    if len(st.views) == 1: return self   # return self for trivial movement ops
    return LazyBuffer(st.shape, op, [self], arg)

  def processing_op(self, op, x, w, C):
    return LazyBuffer(C.out_shape, op, [x,w], C)

# universal dispatcher?
class Ops:
  def unary_op(ctx, op, x): return x.unary_op(op)
  def binary_op(ctx, op, x, y): return x.binary_op(op, y)
  def reduce_op(ctx, op, x, new_shape): return x.reduce_op(op, new_shape)
  def movement_op(ctx, op, x, arg): return x.movement_op(op, arg)
  def processing_op(ctx, op, x, w, C): return x.processing_op(op, x, w, C)
