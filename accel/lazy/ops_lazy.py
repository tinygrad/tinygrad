from __future__ import annotations
import os
from typing import Union, NamedTuple, List, Any, Tuple, Dict
from tinygrad.shapetracker import ShapeTracker
import functools, operator
from tinygrad.helpers import prod, ConvArgs
import sys
sys.setrecursionlimit(10000)

from tinygrad.ops import ReduceOps, BinaryOps, MovementOps, ProcessingOps, log_op, DEBUG, GRAPH
from enum import Enum
LoadOps = Enum("LoadOps", ["FROMCPU", "CONTIGUOUS"])
Op = Union[BinaryOps, ReduceOps, MovementOps, ProcessingOps, LoadOps]

# -O1
CACHE_LAZYBUFFERS = True    # this leaks tons of memory. TODO: only cache unresolved LazyBuffers
MERGE_MOVEMENT_OPS = True
MERGE_UNARY_OPS = True
REMOVE_MOVEMENT_NOPS = True

# -O2
SHUFFLE_MOVEMENT_OPS = True
MERGE_ELEMENTWISE_OPS = True
FOLD_CONSTANTS_INTO_KERNELS = True   # should depend on the JIT if it's a float or a number

# -O3
SHUFFLE_SLICE_OPS = False  # NOTE: 0/0 is NaN if you slice, so this can change the output
MERGE_ELEMENTWISE_INTO_CONV_OUTPUT = False  # TODO: should this be done at resolve time?

class LazyOp(NamedTuple):
  op: Op
  src: Tuple[Union[LazyOp, LazyBuffer]]
  arg: Any = None

def get_lazyops(op:LazyOp) -> List[LazyOp]: return functools.reduce(operator.add, [get_lazyops(x) for x in op.src if isinstance(x, LazyOp)], [op])
def get_lazybuffers(op:LazyOp) -> List[LazyBuffer]: return functools.reduce(operator.add, [get_lazybuffers(x) if isinstance(x, LazyOp) else [x] for x in op.src], [])
def find_conv(op:LazyOp) -> LazyOp: return [x for x in get_lazyops(op) if isinstance(x.op, ProcessingOps)][0]

# TODO: i'm sure this is a real algorithm
def cmp(buf1:LazyBuffer, buf2:LazyBuffer):
  explore1, explore2 = [buf1], [buf2]
  expanded1, expanded2 = set(), set()
  while len(explore1) and len(explore2):
    if buf2 in explore1: return -1
    if buf1 in explore2: return 1
    x1 = explore1.pop(0)
    x2 = explore2.pop(0)
    if x1 in expanded2 or x2 in expanded1: return 0
    if x1 not in expanded1 and x1.realized is None:
      explore1 += get_lazybuffers(x1.op)
      expanded1.add(x1)
    if x2 not in expanded2 and x2.realized is None:
      explore2 += get_lazybuffers(x2.op)
      expanded2.add(x2)
  return 0

# TODO: confirm cmp is the same thing as this
#@functools.lru_cache(maxsize=None)
#def depends(haystack:LazyBuffer, needle:LazyBuffer):
#  gen = get_lazybuffers(haystack.op)
#  if needle in gen: return True
#  return any(depends(x, needle) for x in gen if x.realized is None)

class LazyBuffer:
  def __init__(self, shape:Union[ShapeTracker, Tuple[int]], optype:Op, op:LazyOp):
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(shape)
    self.shape = self.st.shape
    self.optype, self.op = optype, op
    self.realized = None

  def __repr__(self): return f"<LB {self.shape} {self.optype}>"

  def realize(self:LazyBuffer):
    if self.realized is None:
      self.realized, real_srcs = _realize(self)
      # TODO: get if logging in a better way
      if DEBUG or GRAPH:
        # in lazy mode, we don't log until we realize
        log_op(self.optype, [x.op for x in get_lazyops(self.op)], self.realized, real_srcs)
      del self.op
    assert self.realized.shape == self.shape
    return self.realized

  @staticmethod
  def fromCPU(x):
    ret = LazyBuffer(x.shape, LoadOps, LazyOp(LoadOps.FROMCPU, tuple(), x))
    #ret.realize()
    return ret

  def toCPU(self):
    return self.realize().toCPU()

  def unary_op(x, op): return elementwise_op(op, (x,))
  def binary_op(x, op, y:LazyBuffer): return elementwise_op(op, (x,y))

  @functools.lru_cache(maxsize=None if CACHE_LAZYBUFFERS else 0)
  def contiguous_op(x:LazyBuffer) -> LazyBuffer: return x if x.st.contiguous else LazyBuffer(x.shape, LoadOps, LazyOp(LoadOps.CONTIGUOUS, (x,)))

  @functools.lru_cache(maxsize=None if CACHE_LAZYBUFFERS else 0)
  def movement_op(x:LazyBuffer, op:MovementOps, arg) -> LazyBuffer:
    # TODO: SHUFFLE_SLICE_OPS is okay if it's a shrink
    if SHUFFLE_MOVEMENT_OPS and x.optype == BinaryOps and x.realized is None and (SHUFFLE_SLICE_OPS or op != MovementOps.SLICE):
      # if this MovementOp is being applied to a BinaryOp, apply the MovementOp to all the BinaryOp inputs instead
      def replace_with_movement_op(y:Union[LazyOp, LazyBuffer]) -> LazyBuffer:
        if isinstance(y, LazyBuffer): return y.movement_op(op, arg)
        return elementwise_op(y.op, tuple(replace_with_movement_op(z) for z in y.src))
      return replace_with_movement_op(x.op)

    # if a MovementOp is applied to a MovementOp, merge them and use one buffer
    ret = LazyBuffer(ShapeTracker(x.st).movement_op(op, arg), MovementOps, LazyOp(op, (x.op if MERGE_MOVEMENT_OPS and x.optype == MovementOps and x.realized is None else x,), arg))

    if REMOVE_MOVEMENT_NOPS and x.realized is None and ret.st.contiguous:
      root = get_lazybuffers(ret.op)[0]
      if ret.st.shape == root.shape:
        return root

    return ret

  @functools.lru_cache(maxsize=None if CACHE_LAZYBUFFERS else 0)
  def reduce_op(x:LazyBuffer, op:ReduceOps, new_shape:Tuple[int]):
    return LazyBuffer(new_shape, ReduceOps, LazyOp(op, (x,), new_shape))

  @functools.lru_cache(maxsize=None if CACHE_LAZYBUFFERS else 0)
  def processing_op(x:LazyBuffer, op:ProcessingOps, w:LazyBuffer, C:ConvArgs):
    return LazyBuffer(C.out_shape, ProcessingOps, LazyOp(op, (x.contiguous_op(), w.contiguous_op()), C))

def ast_op(op: Op, srcs_code: List[str]) -> str:
  code = gops.code_for_op[op]
  if len(srcs_code) >= 1: code = code.replace("A", srcs_code[0])
  if len(srcs_code) >= 2: code = code.replace("B", srcs_code[1])
  return code

def ast(x: Union[LazyBuffer, LazyOp], lazy_srcs: Dict[LazyBuffer, str]) -> str:
  if isinstance(x, LazyBuffer): return lazy_srcs[x]
  # if it's not a LazyBuffer, it's an op
  if x.op == ProcessingOps.CONV: return "acc"
  return ast_op(x.op, [ast(src, lazy_srcs) for src in x.src])

# this is needed to reduce convs from 186 -> 174
@functools.lru_cache(maxsize=None if CACHE_LAZYBUFFERS else 0)
def elementwise_op(op, srcs:Tuple[LazyBuffer]) -> LazyBuffer:
  out_shape = srcs[0].shape

  if MERGE_ELEMENTWISE_INTO_CONV_OUTPUT:
    psrcs = [x for x in srcs if x.optype == ProcessingOps and x.realized is None]
    if len(psrcs) > 0:
      if len(psrcs) == 1:
        srcs = [x.op if x in psrcs else x for x in srcs]
      elif len(psrcs) == 2:
        c1, c2 = [find_conv(x.op) for x in psrcs]  # this returns a list of LazyOps
        if c1 == c2:   # NOTE: this compare relies on working caching for processing_ops
          # they are the same conv, merge them
          srcs = [srcs[0].op, srcs[1].op]
        else:
          # they are not the same conv, choose one
          order = cmp(srcs[0], srcs[1])
          if order == -1: #if depends(srcs[0], srcs[1]):
            srcs = [srcs[0].op, srcs[1]]
          elif order == 1: #elif depends(srcs[1], srcs[0]):
            srcs = [srcs[0], srcs[1].op]
          else:
            # all three are okay
            #return LazyBuffer(out_shape, BinaryOps, LazyOp(op, list(srcs)))
            srcs = [srcs[0].op, srcs[1]]
            #srcs = [srcs[0], srcs[1].op]
      return LazyBuffer(out_shape, ProcessingOps, LazyOp(op, tuple(srcs)))

  if (MERGE_UNARY_OPS and len(srcs) == 1) or MERGE_ELEMENTWISE_OPS:
    # remove the buffers from any BinaryOps that feed into this
    srcs = tuple(x.op if x.optype == BinaryOps and x.realized is None else x for x in srcs)

  return LazyBuffer(out_shape, BinaryOps, LazyOp(op, srcs))

# these functions determines the backing buffer
if int(os.getenv("LAZY_OPENCL", 0)) == 1:
  import tinygrad.llops.ops_opencl as gops
else:
  import tinygrad.llops.ops_gpu as gops

def _realize_binary_op(self:LazyBuffer) -> Tuple[gops.GPUBuffer, List[gops.GPUBuffer]]:
  # optional
  if self.optype == ProcessingOps:
    conv = find_conv(self.op)
    conv_x, conv_w = conv.src[0], conv.src[1]
    seen = {conv_x:conv_x, conv_w:conv_w}
    real_srcs = [("input", conv_x.realize()), ("weight", conv_w.realize())]
    assert real_srcs[0][1].st.contiguous and real_srcs[1][1].st.contiguous
    arg = conv.arg
  else:
    seen = {}
    real_srcs : List[Tuple[str, gops.GPUBuffer]] = []
    arg = None
  lazy_srcs : List[LazyBuffer] = [seen.setdefault(x,x) for x in get_lazybuffers(self.op) if x not in seen]
  real_dict : Dict[LazyBuffer, str] = {}
  for s in lazy_srcs:
    if s.optype in [LoadOps, MovementOps] and s.realized is None:
      if s.optype == MovementOps: root = get_lazybuffers(s.op)[0]
      else: root = s
      # NOTE: if this is used, there can be 0 sources for a kernel
      if FOLD_CONSTANTS_INTO_KERNELS and root.realized is None and root.optype == LoadOps and root.op.op == LoadOps.FROMCPU and root.shape == (1,):
        if not s.st.needs_valid():
          real_dict[s] = f"({root.op.arg[0]}f)"
        else:
          # TODO: this is a terrible hack, and it's very unclear if it's always right
          # can't we just replace the getter function?
          inline_valid = s.st.expr().replace("valid=valid && ", "").replace(";idx=0", "").replace("//", "/").replace("idx", "gid")
          if ';' not in inline_valid:
            real_dict[s] = f"(({inline_valid}) * {str(root.op.arg[0])}f)"
          else:
            print(f"couldn't fold {str(root.op.arg[0])} with expr {s.st.expr()}")
    if s not in real_dict:  # nicer way to write this?
      real_dict[s] = f"arg_{len(real_srcs)}"
      real_srcs.append((f"arg_{len(real_srcs)}", s.realize()))
  code = ast(self.op, real_dict)
  return gops.GPUBuffer(self.shape)._processing_op(real_srcs, code, arg), [x[1] for x in real_srcs]

def _realize(self:LazyBuffer) -> Tuple[gops.GPUBuffer, List[gops.GPUBuffer]]:
  if self.optype == LoadOps and self.op.op == LoadOps.FROMCPU:
    #print("load", self, self.shape, self.op.arg if prod(self.shape) == 1 else "<data>")
    if self.shape == (1,): print("NOTE: resolving unary Tensor", self)
    return gops.GPUBuffer.fromCPU(self.op.arg), []
  elif self.optype == LoadOps and self.op.op == LoadOps.CONTIGUOUS:
    real_src = self.op.src[0].realize()
    return real_src.contiguous_op(), [real_src]
  elif self.optype == ReduceOps:
    real_src = self.op.src[0].realize()
    return real_src.reduce_op(self.op.op, self.op.arg), [real_src]
  elif self.optype == MovementOps:
    real_src = get_lazybuffers(self.op)[0].realize()
    return gops.GPUBuffer(self.st, real_src), [real_src]
  elif self.optype in [BinaryOps, ProcessingOps]:
    return _realize_binary_op(self)
