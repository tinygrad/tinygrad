from matplotlib.pyplot import isinteractive
from tinygrad.shapetracker import ShapeTracker
import numpy as np
from copy import deepcopy

# TODO: these aren't really ops
from tinygrad.ops import BinaryOps, MovementOps, UnaryOps, log_op

class LazyBuffer:
  def __init__(self, shape, op=None, src=[], arg=None, hostbuf=None):
    self.shape, self.hostbuf = shape, hostbuf
    self.dtype = np.float32

    # if we got a movement op, push it above any UnaryOp/BinaryOp
    # this is done on construction
    if isinstance(op, MovementOps):
      if isinstance(src[0].op, BinaryOps) or isinstance(src[0].op, UnaryOps):
        mop = op
        op = src[0].op
        src = [LazyBuffer(self.shape, mop, [x], arg) for x in src[0].src]
        arg = None

    self.op, self.src, self.arg = op, src, arg
    self.did_realize = False

  def realize(self):
    if self.did_realize or self.op is None: return
    self.did_realize = True
    for s in self.src:
      s.realize()
    log_op(self.op, self, self.src)

  @staticmethod
  def fromCPU(x):
    return LazyBuffer(x.shape, hostbuf=x)

  def toCPU(self):
    self.realize()
    # this realizes the tensor 
    return np.zeros(self.shape, self.dtype)

class Ops:
  def unary_op(ctx, op, x): return LazyBuffer(x.shape, op, [x])
  def binary_op(ctx, op, x, y): return LazyBuffer(x.shape, op, [x,y])
  def reduce_op(ctx, op, x, new_shape): return LazyBuffer(new_shape).op(op, [x], new_shape)
  def movement_op(ctx, op, x, arg): return LazyBuffer(ShapeTracker(*x.shape).movement_op(op, arg).shape, op, [x], arg)
  #def processing_op(ctx, op, x, w, C): return LazyBuffer((C.bs, C.cout, C.oy, C.ox), op, [x,w], C)
  def processing_op(ctx, op, x, w, C): return LazyBuffer((C.bs*C.oy, C.ox*C.cout//4, 4), op, [x,w], C)
