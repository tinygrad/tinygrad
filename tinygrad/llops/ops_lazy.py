from matplotlib.pyplot import isinteractive
from tinygrad.shapetracker import ShapeTracker
import functools
import numpy as np
import sys
sys.setrecursionlimit(10000)

# TODO: these aren't really ops
from tinygrad.ops import BinaryOps, MovementOps, UnaryOps, log_op

def tp(a,b): return tuple(list(a) + list(b))

class LazyBuffer:
  def __init__(self, shape, op=[None], src=[], arg=[], hostbuf=None):
    self.shape, self.hostbuf = tuple(shape), hostbuf
    self.dtype = np.float32
    self.op, self.src, self.arg = op, src, arg
    self.did_realize = False

  def realize(self):
    if self.did_realize or self.op[0] is None: return self
    self.did_realize = True
    srcs = [s.realize() for s in self.src]
    # TODO: do real op here
    log_op(self.op, self, srcs)
    return self

  def __repr__(self):
    return f"<<< LB ops:{self.op} src:{self.src} >>>"

  @staticmethod
  def fromCPU(x):
    return LazyBuffer(x.shape, hostbuf=x)

  def toCPU(self):
    self.realize()
    # this realizes the tensor 
    return np.zeros(self.shape, self.dtype)

  def elementwise_op(self, ops, src=[]):
    # TODO: if there's one (and only one) conv op in src, merge it in
    merge_ops, merge_src = [], []
    retain_src = []
    for x in src:
      if isinstance(x.op[0], BinaryOps) or isinstance(x.op[0], UnaryOps):
        merge_ops += x.op
        merge_src += x.src
      else:
        retain_src.append(x)
    if len(merge_src) > 0:
      return merge_src[0].elementwise_op(merge_ops + ops, merge_src + retain_src)
    return LazyBuffer(self.shape, ops, src)

  @functools.lru_cache()
  def movement_op(self, ops, args):
    st = ShapeTracker(*self.shape)
    for o,a in zip(ops, args):
      st = st.movement_op(o, a)
    if len(st.views) == 1: return self   # return self for trivial movement ops

    # this is wrong, but try it. it should confirm that the strides are correct also
    if st.shape == self.shape: return self
    # TODO: FIX THIS

    # if self is a UnaryOp/BinaryOp, push any movement op above them
    if isinstance(self.op[0], BinaryOps) or isinstance(self.op[0], UnaryOps):
      src = [x.movement_op(ops, args) for x in self.src]
      return src[0].elementwise_op(self.op, src)

    # if self is a movement op, merge this one in
    if isinstance(self.op[0], MovementOps):
      return self.src[0].movement_op(tp(self.op, ops), tp(self.arg, args))

    # otherwise just create the movement op
    return LazyBuffer(st.shape, ops, [self], args)

  def reduce_op(self, op, new_shape): return LazyBuffer(new_shape, [op], [self], new_shape)
  def processing_op(self, op, w, C): return LazyBuffer(C.out_shape, [op], [self,w], C)

# universal dispatcher?
class Ops:
  def unary_op(ctx, op, x): return x.elementwise_op([op], [x])
  def binary_op(ctx, op, x, y): return x.elementwise_op([op], [x,y])
  def movement_op(ctx, op, x, arg): return x.movement_op((op,), (tuple(arg),))
  # blocker ops
  def reduce_op(ctx, op, x, new_shape): return x.reduce_op(op, new_shape)
  def processing_op(ctx, op, x, w, C): return x.processing_op(op, w, C)
