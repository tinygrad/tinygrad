from matplotlib.pyplot import isinteractive
from tinygrad.shapetracker import ShapeTracker
import functools
import numpy as np
import sys
sys.setrecursionlimit(10000)

# TODO: these aren't really ops
from tinygrad.ops import BinaryOps, MovementOps, ProcessingOps, UnaryOps, log_op

def tp(a,b,c=[]): return tuple(list(a) + list(b) + list(c))

# movement ops can be moved above elementwise ops 
SHUFFLE_MOVEMENT_OPS = True

# sequential movement ops can be flattened into 0 or 1 movement ops 
MERGE_MOVEMENT_OPS = True

# "sequential" elementwise ops can be merged into 1 big elementwise op
MERGE_ELEMENTWISE_OPS = True

# after the conv is done, it can run elementwise ops on its output
MERGE_ELEMENTWISE_INTO_CONV_OUTPUT = True

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

  @functools.lru_cache()
  def elementwise_op(self, ops, src):
    if MERGE_ELEMENTWISE_INTO_CONV_OUTPUT:
      convs = list(set([x for x in src if isinstance(x.op[0], ProcessingOps)]))
      if len(convs) == 1:
        conv = convs[0]
        nsrc = [x for x in tp(conv.src, src) if x != conv]
        return LazyBuffer(conv.shape, tp(conv.op, ops), tuple(nsrc), conv.arg)
      if len(convs) == 2:
        # see if they are the same conv
        if convs[0].arg == convs[1].arg and convs[0].src[0:2] == convs[1].src[0:2]:
          nsrc = [x for x in tp(convs[0].src, convs[1].src[2:], src) if x not in convs]
          nops = tp(convs[0].op, convs[1].op[1:], ops)
          return LazyBuffer(convs[0].shape, nops, tuple(nsrc), convs[0].arg)
        else:
          # would need to handle this to make accumulate work
          pass

    if MERGE_ELEMENTWISE_OPS:
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
        return merge_src[0].elementwise_op(tp(merge_ops, ops), tp(merge_src, retain_src))

    return LazyBuffer(self.shape, ops, src)

  #@functools.lru_cache()
  def movement_op(self, ops, args):
    st = ShapeTracker(*self.shape)
    for o,a in zip(ops, args):
      st = st.movement_op(o, a)
    if len(st.views) == 1: return self   # return self for trivial movement ops

    # this is wrong, but try it. it should confirm that the strides are correct also
    if st.shape == self.shape: return self
    # TODO: FIX THIS

    if SHUFFLE_MOVEMENT_OPS:
      # if self is a UnaryOp/BinaryOp, push any movement op above them
      if isinstance(self.op[0], BinaryOps) or isinstance(self.op[0], UnaryOps):
        src = [x.movement_op(ops, args) for x in self.src]
        return src[0].elementwise_op(self.op, tuple(src))

    if MERGE_MOVEMENT_OPS:
      # if self is a movement op, merge this one in
      if isinstance(self.op[0], MovementOps):
        return self.src[0].movement_op(tp(self.op, ops), tp(self.arg, args))

    # otherwise just create the movement op
    return LazyBuffer(st.shape, ops, [self], args)

  def reduce_op(self, op, new_shape): return LazyBuffer(new_shape, [op], [self], new_shape)
  def processing_op(self, ops, args, C): return LazyBuffer(C.out_shape, ops, args, C)

# universal dispatcher?
class Ops:
  def unary_op(ctx, op, x): return x.elementwise_op((op,), (x,))
  def binary_op(ctx, op, x, y): return x.elementwise_op((op,), (x,y))
  def movement_op(ctx, op, x, arg): return x.movement_op((op,), (tuple(arg),))
  # blocker ops
  def reduce_op(ctx, op, x, new_shape): return x.reduce_op(op, new_shape)
  def processing_op(ctx, op, x, w, C): return x.processing_op([op], [x,w], C)
