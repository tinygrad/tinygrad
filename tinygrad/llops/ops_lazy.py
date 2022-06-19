from __future__ import annotations
from tinygrad.shapetracker import ShapeTracker
import functools
import numpy as np
import sys
sys.setrecursionlimit(10000)

from typing import Union, NamedTuple, List, Any, Tuple
from tinygrad.ops import ReduceOps, BinaryOps, MovementOps, ProcessingOps, LoadOps
Op = Union[BinaryOps, ReduceOps, MovementOps, ProcessingOps, LoadOps]

MERGE_MOVEMENT_OPS = True
SHUFFLE_MOVEMENT_OPS = True
REMOVE_MOVEMENT_NOPS = False
MERGE_ELEMENTWISE_OPS = True
MERGE_ELEMENTWISE_INTO_CONV_OUTPUT = True

class LazyOp(NamedTuple):
  op: Op
  src: List[Union[LazyOp, LazyBuffer]]
  arg: Any = None

@functools.lru_cache(maxsize=None)
def get_lazybuffers_for_buffer(x:LazyBuffer):
  return get_lazybuffers(x.op)

def get_lazybuffers(op:LazyOp):
  ret = []
  for x in op.src:
    if isinstance(x, LazyOp):
      ret += get_lazybuffers(x)
    elif isinstance(x, LazyBuffer):
      ret.append(x)
    else: raise Exception("wtf")
  return ret

@functools.lru_cache(maxsize=None)
def find_conv_buf(x: LazyBuffer):
  return find_conv(x.op)

def find_conv(x:LazyOp):
  if isinstance(x, LazyBuffer):
    return None
  if isinstance(x.op, ProcessingOps):
    return x
  for s in x.src:
    tst = find_conv(s)
    if tst is not None:
      return tst
  return None

@functools.lru_cache(maxsize=None)
def depends(me:LazyBuffer, needle:LazyBuffer) -> bool:
  bufs = get_lazybuffers_for_buffer(me)
  if needle in bufs:
    return True
  ret = False
  for b in bufs:
    if depends(b, needle):
      ret = True
      break
  return ret

class LazyBuffer:
  def __init__(self, shape:tuple, optype:Op, op:LazyOp):
    assert isinstance(op, LazyOp)
    assert isinstance(op.src, list)
    assert isinstance(shape, tuple)

    self.shape = shape
    self.optype = optype
    self.dtype = np.float32
    self.op = op
    self.realized = None

@functools.lru_cache(maxsize=None)
def elementwise_op(op, srcs:Tuple[LazyBuffer]) -> LazyBuffer:
  Buffer = srcs[0].__class__
  out_shape = srcs[0].shape

  if MERGE_ELEMENTWISE_INTO_CONV_OUTPUT:
    cnt = sum([x.optype == ProcessingOps for x in srcs])
    if cnt == 1:
      srcs = [x.op if x.optype == ProcessingOps else x for x in srcs]
      return Buffer(out_shape, ProcessingOps, LazyOp(op, srcs))
    elif cnt == 2:
      # have to confirm they are the same conv
      c1, c2 = [find_conv_buf(x) for x in srcs]
      if c1.op == c1.op and c1.arg == c2.arg and tuple(c1.src) == tuple(c2.src):
        srcs = [x.op if x.optype == ProcessingOps else x for x in srcs]
        return Buffer(out_shape, ProcessingOps, LazyOp(op, srcs))
      else:
        if depends(srcs[0], srcs[1]):
          srcs = [srcs[0].op, srcs[1]]
        elif depends(srcs[1], srcs[0]):
          srcs = [srcs[0], srcs[1].op]
        else:
          # all three are okay
          #return Buffer(out_shape, BinaryOps, LazyOp(op, list(srcs)))
          srcs = [srcs[0].op, srcs[1]]
          #srcs = [srcs[0], srcs[1].op]
        return Buffer(out_shape, ProcessingOps, LazyOp(op, srcs))

  if MERGE_ELEMENTWISE_OPS:
    # remove the buffers from any BinaryOps that feed into this
    srcs = [x.op if x.optype == BinaryOps else x for x in srcs]

  return Buffer(out_shape, BinaryOps, LazyOp(op, list(srcs)))

# caching is safe here, the same op and arg applied to the same buffer is the same
@functools.lru_cache(maxsize=None)
def movement_op(op:MovementOps, x:LazyBuffer, arg) -> LazyBuffer:
  Buffer = x.__class__

  st = ShapeTracker(*x.shape)
  # TODO: Refactor shapetracker to return a new shapetracker
  st = st.movement_op(op, arg)
  if len(st.views) == 1: return x    # this is a no-op

  if REMOVE_MOVEMENT_NOPS and x.optype == MovementOps:
    # if this MovementOp is a no op, remove it
    # TODO: Use shapetracker in lazybuffer to make this simple
    root = x.op
    op_arg = [(op, arg)]
    while isinstance(root, LazyOp):
      op_arg.append((root.op, root.arg))
      root = root.src[0]
    assert isinstance(root, LazyBuffer)
    rst = ShapeTracker(*root.shape)
    for o,a in op_arg[::-1]:
      rst = rst.movement_op(o, a)
    # TODO: this check is wrong, we used the shapetracker for a reason
    if rst.shape == root.shape:
      return root

  if SHUFFLE_MOVEMENT_OPS and x.optype == BinaryOps:
    # if this MovementOp is being applied to a BinaryOp, apply the MovementOp to all the BinaryOp inputs instead
    def replace_w_movement_op(y:Union[LazyOp, LazyBuffer]) -> LazyBuffer:
      if isinstance(y, LazyBuffer): return movement_op(op, y, arg)
      elif isinstance(y, LazyOp): return elementwise_op(y.op, tuple([replace_w_movement_op(z) for z in y.src]))
    return replace_w_movement_op(x.op)

  if MERGE_MOVEMENT_OPS and x.optype == MovementOps:
    # if a MovementOp is applied to a MovementOp, merge them and use one buffer
    x = x.op

  return Buffer(st.shape, MovementOps, LazyOp(op, [x], arg))

def reduce_op(op, x, new_shape):
  Buffer = x.__class__
  return Buffer(new_shape, ReduceOps, LazyOp(op, [x], new_shape))

def processing_op(op, x, w, C):
  Buffer = x.__class__
  return Buffer(C.out_shape, ProcessingOps, LazyOp(op, [x, w], C))

# universal dispatcher?
class Ops:
  def unary_op(ctx, op, x):
    return elementwise_op(op, (x,))
  def binary_op(ctx, op, x, y):
    return elementwise_op(op, (x,y))
  def movement_op(ctx, op, x, arg):
    return movement_op(op, x, tuple(arg))
  def reduce_op(ctx, op, x, new_shape):
    return reduce_op(op, x, new_shape)
  def processing_op(ctx, op, x, w, C):
    return processing_op(op, x, w, C)
