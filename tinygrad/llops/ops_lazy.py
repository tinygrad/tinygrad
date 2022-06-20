from __future__ import annotations
from typing import Union, NamedTuple, List, Any, Tuple
from tinygrad.helpers import prod
from tinygrad.shapetracker import ShapeTracker
import functools

from tinygrad.ops import ReduceOps, BinaryOps, MovementOps, ProcessingOps, LoadOps, log_op
Op = Union[BinaryOps, ReduceOps, MovementOps, ProcessingOps, LoadOps]

MERGE_MOVEMENT_OPS = True
SHUFFLE_MOVEMENT_OPS = True
REMOVE_MOVEMENT_NOPS = True
MERGE_ELEMENTWISE_OPS = False

class LazyOp(NamedTuple):
  op: Op
  src: List[Union[LazyOp, LazyBuffer]]
  arg: Any = None

def get_root(x:LazyOp) -> LazyBuffer: return x if isinstance(x, LazyBuffer) else get_root(x.src[0])

class LazyBuffer:
  def __init__(self, shape:tuple, optype:Op, op:LazyOp):
    self.st = ShapeTracker(shape)
    self.optype, self.op = optype, op
    self.realized = None

  @property
  def shape(self): return self.st.shape

  def realize(self:LazyBuffer):
    if self.realized is None:
      self.realized, real_srcs = _realize(self)
      def get_lazyops(op:LazyOp) -> List[Op]:
        ret = [op.op]
        # TODO: reduce?
        for x in [get_lazyops(x) for x in op.src if isinstance(x, LazyOp)]:
          ret += x
        return ret
      # in lazy mode, we don't log until we realize
      log_op(self.optype, get_lazyops(self.op), self.realized, real_srcs)
    return self.realized

  @staticmethod
  def fromCPU(x):
    ret = LazyBuffer(x.shape, LoadOps, LazyOp(LoadOps.FROMCPU, [], x))
    ret.realize()   # early realize on these
    return ret

  def toCPU(self):
    return self.realize().toCPU()

# this function determines the backing buffer
import tinygrad.llops.ops_gpu as gops
def _realize(self:LazyBuffer) -> Tuple[gops.GPUBuffer, List[gops.GPUBuffer]]:
  if self.optype == LoadOps:
    #print("load", self, self.shape, self.op.arg if prod(self.shape) == 1 else "<data>")
    return gops.GPUBuffer(self.shape, self.op.arg), []
  elif self.optype == ReduceOps:
    real_src = self.op.srcs[0].realize()
    return gops.reduce_op(self.op.op, real_src, self.op.arg), [real_src]
  elif self.optype == MovementOps:
    real_src = get_root(self.op).realize()
    return gops.GPUBuffer(self.st, real_src), [real_src]
  elif self.optype == BinaryOps:
    srcs = [x.realize() for x in self.op.src]
    return gops.elementwise_op(list(zip(["A", "B"], srcs)), gops.code_for_op[self.op.op]), srcs
  elif self.optype == ProcessingOps:
    srcs = [x.realize() for x in self.op.src]
    return gops.processing_op(self.op.op, srcs[0], srcs[1], self.op.arg), srcs

def elementwise_op(op, srcs:Tuple[LazyBuffer]) -> LazyBuffer:
  out_shape = srcs[0].shape

  if MERGE_ELEMENTWISE_OPS:
    # remove the buffers from any BinaryOps that feed into this
    srcs = [x.op if x.optype == BinaryOps else x for x in srcs]

  return LazyBuffer(out_shape, BinaryOps, LazyOp(op, list(srcs)))

def unary_op(op, x):
  return elementwise_op(op, (x,))

def binary_op(op, x, y):
  return elementwise_op(op, (x,y))

@functools.lru_cache(maxsize=None)
def movement_op(op:MovementOps, x:LazyBuffer, arg):
  if SHUFFLE_MOVEMENT_OPS and x.optype == BinaryOps:
    # if this MovementOp is being applied to a BinaryOp, apply the MovementOp to all the BinaryOp inputs instead
    def replace_w_movement_op(y:Union[LazyOp, LazyBuffer]) -> LazyBuffer:
      if isinstance(y, LazyBuffer): return movement_op(op, y, arg)
      elif isinstance(y, LazyOp): return elementwise_op(y.op, tuple([replace_w_movement_op(z) for z in y.src]))
    return replace_w_movement_op(x.op)

  # if a MovementOp is applied to a MovementOp, merge them and use one buffer
  ret = LazyBuffer(x.st, MovementOps, LazyOp(op, [x.op if MERGE_MOVEMENT_OPS and x.optype == MovementOps else x], arg))
  ret.st.movement_op(op, arg)

  if REMOVE_MOVEMENT_NOPS and x.optype == MovementOps:
    root = get_root(x.op)
    if ret.st.contiguous and ret.st.shape == root.shape:
      return root

  return ret

def reduce_op(op, x, new_shape):
  return LazyBuffer(new_shape, ReduceOps, LazyOp(op, [x], new_shape))

def processing_op(op, x, w, C):
  return LazyBuffer(C.out_shape, ProcessingOps, LazyOp(op, [x, w], C))
