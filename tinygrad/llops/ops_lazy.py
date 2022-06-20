from __future__ import annotations
from typing import Union, NamedTuple, List, Any, Tuple, Dict
from tinygrad.helpers import prod
from tinygrad.shapetracker import ShapeTracker
import functools, operator

from tinygrad.ops import ReduceOps, BinaryOps, MovementOps, ProcessingOps, LoadOps, log_op, DEBUG, GRAPH
Op = Union[BinaryOps, ReduceOps, MovementOps, ProcessingOps, LoadOps]

MERGE_MOVEMENT_OPS = True
SHUFFLE_MOVEMENT_OPS = True
REMOVE_MOVEMENT_NOPS = True
MERGE_ELEMENTWISE_OPS = True
MERGE_ELEMENTWISE_INTO_CONV_OUTPUT = True

class LazyOp(NamedTuple):
  op: Op
  src: Tuple[Union[LazyOp, LazyBuffer]]
  arg: Any = None

def get_root(x:LazyOp) -> LazyBuffer: return x if isinstance(x, LazyBuffer) else get_root(x.src[0])
def get_lazyops(op:LazyOp) -> List[Op]: return functools.reduce(operator.add, [get_lazyops(x) for x in op.src if isinstance(x, LazyOp)], [op.op])
def get_lazybuffers(op:LazyOp) -> List[LazyBuffer]: return functools.reduce(operator.add, [get_lazybuffers(x) if isinstance(x, LazyOp) else [x] for x in op.src], [])

# TODO: this is very slow
def depends(me:LazyBuffer, needle:LazyBuffer) -> bool:
  @functools.lru_cache(maxsize=None)
  def _depends(me:LazyBuffer, needle:LazyBuffer) -> bool:
    if me.realized is not None: return False
    bufs = get_lazybuffers(me.op)
    if needle in bufs:
      return True
    ret = False
    for b in bufs:
      if _depends(b, needle):
        ret = True
        break
    return ret
  return _depends(me, needle)

def find_conv(x:Union[LazyOp,LazyBuffer]):
  if isinstance(x, LazyBuffer):
    return None
  if isinstance(x.op, ProcessingOps):
    return x
  for s in x.src:
    tst = find_conv(s)
    if tst is not None:
      return tst
  return None

class LazyBuffer:
  def __init__(self, shape:Union[ShapeTracker, Tuple[int]], optype:Op, op:LazyOp):
    self.st = ShapeTracker(shape)
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
        log_op(self.optype, get_lazyops(self.op), self.realized, real_srcs)
      del self.op
    return self.realized

  @staticmethod
  def fromCPU(x):
    ret = LazyBuffer(x.shape, LoadOps, LazyOp(LoadOps.FROMCPU, tuple(), x))
    #ret.realize()
    return ret

  def toCPU(self):
    return self.realize().toCPU()

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

# these functions determines the backing buffer
import tinygrad.llops.ops_gpu as gops

def _realize_binary_op(self:LazyBuffer, has_conv:bool=False) -> Tuple[gops.GPUBuffer, List[gops.GPUBuffer]]:
  # optional
  if has_conv:
    conv = find_conv(self.op)
    conv_x, conv_w = conv.src[0], conv.src[1]
    seen = {conv_x:conv_x, conv_w:conv_w}
    conv_x_real, conv_w_real = conv_x.realize(), conv_w.realize()
    # TODO: this contiguous shouldn't be here, it should be inserted at build time if needed
    real_srcs = [("input", gops.contiguous(conv_x_real)), ("weight", gops.contiguous(conv_w_real))]
    arg = conv.arg
  else:
    seen = {}
    real_srcs : List[Tuple[str, gops.GPUBuffer]] = []
    arg = None
  lazy_srcs : List[LazyBuffer] = [seen.setdefault(x,x) for x in get_lazybuffers(self.op) if x not in seen]
  real_dict : Dict[LazyBuffer, str] = {}
  for s in lazy_srcs:
    if s.optype == MovementOps and s.realized is None:
      root = get_root(s.op)
      if root.realized is None and root.optype == LoadOps and root.shape == (1,):
        if not s.st.needs_valid():
          real_dict[s] = str(root.op.arg[0]) + "f"
        else:
          # TODO: this is a terrible hack, and it's very unclear if it's always right
          inline_valid = s.st.expr().replace("valid=valid && ", "").replace(";idx=0", "").replace("//", "/").replace("idx", "gid")
          if ';' not in inline_valid:
            real_dict[s] = f"(({inline_valid}) * {str(root.op.arg[0])}f)"
    if s not in real_dict:  # nicer way to write this?
      real_dict[s] = f"arg_{len(real_srcs)}"
      real_srcs.append((f"arg_{len(real_srcs)}", s.realize()))
  code = ast(self.op, real_dict)
  return gops._processing_op(real_srcs, code, arg), [x[1] for x in real_srcs]

def _realize(self:LazyBuffer) -> Tuple[gops.GPUBuffer, List[gops.GPUBuffer]]:
  if self.optype == LoadOps:
    #print("load", self, self.shape, self.op.arg if prod(self.shape) == 1 else "<data>")
    return gops.GPUBuffer.fromCPU(self.op.arg), []
  elif self.optype == ReduceOps:
    real_src = self.op.src[0].realize()
    return gops.reduce_op(self.op.op, real_src, self.op.arg), [real_src]
  elif self.optype == MovementOps:
    real_src = get_root(self.op).realize()
    return gops.GPUBuffer(self.st, real_src), [real_src]
  elif self.optype == BinaryOps:
    return _realize_binary_op(self)
  elif self.optype == ProcessingOps:
    return _realize_binary_op(self, has_conv=True)
    #real_srcs = [x.realize() for x in self.op.src]
    #return gops.processing_op(self.op.op, real_srcs[0], real_srcs[1], self.op.arg), real_srcs

def elementwise_op(op, srcs:Tuple[LazyBuffer]) -> LazyBuffer:
  out_shape = srcs[0].shape

  if MERGE_ELEMENTWISE_INTO_CONV_OUTPUT:
    cnt = sum([x.optype == ProcessingOps for x in srcs])
    if cnt == 1:
      srcs = [x.op if x.optype == ProcessingOps else x for x in srcs]
      return LazyBuffer(out_shape, ProcessingOps, LazyOp(op, srcs))
    elif cnt == 2:
      # have to confirm they are the same conv
      c1, c2 = [find_conv(x.op) for x in srcs]
      if c1.op == c1.op and c1.arg == c2.arg and tuple(c1.src) == tuple(c2.src):
        srcs = [x.op if x.optype == ProcessingOps else x for x in srcs]
        return LazyBuffer(out_shape, ProcessingOps, LazyOp(op, srcs))
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
        return LazyBuffer(out_shape, ProcessingOps, LazyOp(op, srcs))

  if MERGE_ELEMENTWISE_OPS:
    # remove the buffers from any BinaryOps that feed into this
    srcs = tuple([x.op if x.optype == BinaryOps else x for x in srcs])

  return LazyBuffer(out_shape, BinaryOps, LazyOp(op, srcs))

def unary_op(op, x): return elementwise_op(op, (x,))
def binary_op(op, x, y): return elementwise_op(op, (x,y))

@functools.lru_cache(maxsize=None)
def movement_op(op:MovementOps, x:LazyBuffer, arg):
  if SHUFFLE_MOVEMENT_OPS and x.optype == BinaryOps:
    # if this MovementOp is being applied to a BinaryOp, apply the MovementOp to all the BinaryOp inputs instead
    def replace_w_movement_op(y:Union[LazyOp, LazyBuffer]) -> LazyBuffer:
      if isinstance(y, LazyBuffer): return movement_op(op, y, arg)
      elif isinstance(y, LazyOp): return elementwise_op(y.op, tuple([replace_w_movement_op(z) for z in y.src]))
    return replace_w_movement_op(x.op)

  # if a MovementOp is applied to a MovementOp, merge them and use one buffer
  ret = LazyBuffer(x.st, MovementOps, LazyOp(op, (x.op if MERGE_MOVEMENT_OPS and x.optype == MovementOps else x,), arg))
  ret.shape = ret.st.movement_op(op, arg).shape   # update the shape after we modify the ShapeTracker

  if REMOVE_MOVEMENT_NOPS and x.optype == MovementOps:
    root = get_root(x.op)
    if ret.st.contiguous and ret.st.shape == root.shape:
      return root

  return ret

def reduce_op(op, x, new_shape):
  return LazyBuffer(new_shape, ReduceOps, LazyOp(op, (x,), new_shape))

def processing_op(op, x, w, C):
  return LazyBuffer(C.out_shape, ProcessingOps, LazyOp(op, (x, w), C))
