from __future__ import annotations
import numpy as np
from typing import Union, Optional, Any, Tuple, List, Set
from tinygrad.helpers import prod, dtypes, DType, merge_dicts, DEBUG
from tinygrad.ops import LoadOps, UnaryOps, BinaryOps, TernaryOps, ReduceOps, BufferOps
from tinygrad.ops import Op, LazyOp, ConstBuffer, MemBuffer, ScheduleItem, vars_from_ast
from tinygrad.shape.symbolic import sint
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.device import Buffer
from tinygrad.graph import print_tree, log_lazybuffer
from weakref import ref, WeakSet, WeakValueDictionary

lazycache: WeakValueDictionary = WeakValueDictionary()
def create_lazybuffer(device:str, st:ShapeTracker, dtype:DType,
                      op:Optional[Op]=None, arg:Any=None, srcs:Tuple[LazyBuffer, ...]=(),
                      base:Optional[LazyBuffer]=None):
  if 0 in st.shape: st, op, arg, srcs = ShapeTracker.from_shape(st.shape), LoadOps.CONST, 0, ()

  wop = (device, st, dtype, op, arg, tuple(ref(x) for x in srcs), ref(base) if base else None)
  if wop in lazycache: return lazycache[wop]

  ret = LazyBuffer(device, st, dtype, op, arg, srcs, base=base)
  # TODO: remove LoadOps.CONST here while keeping a pretty graph and working fusions
  if op not in {LoadOps.EMPTY, LoadOps.CUSTOM, LoadOps.CONST}: lazycache[wop] = ret
  return ret

class LazyBuffer:
  def __init__(self, device:str, st:ShapeTracker, dtype:DType,
               op:Optional[Op]=None, arg:Any=None, srcs:Tuple[LazyBuffer, ...]=(),
               base:Optional[LazyBuffer]=None):
    self.device, self.st, self.dtype = device, st, dtype
    self.shape = self.st.shape
    assert base is None or base.base == base
    self._base = base
    self.children: WeakSet[LazyBuffer] = WeakSet()
    for x in srcs: x.base.children.add(self.base)
    self.op, self.arg, self.srcs = op, arg, srcs  # this is a LazyOp, except the src is LazyBuffers and not LazyOps
    self._realized: Optional[Buffer] = None
    self.output_buffer: Optional[Buffer] = None

  def __repr__(self) -> str:
    return f"<LB {self.device} {self.shape} contig:{self.st.contiguous} {self.op} {self.realized}>"

  @property
  def base(self) -> LazyBuffer: return self._base if self._base is not None else self

  @property
  def realized(self): return self.base._realized

  @staticmethod
  def new(device, shape:Tuple[int, ...], dtype:DType, op, arg):
    return create_lazybuffer(device, ShapeTracker.from_shape(shape), dtype.scalar(), op, arg)

  def const(self, val:Union[float, int]) -> LazyBuffer:
    return LazyBuffer.new(self.device, (), self.dtype, LoadOps.CONST, val).reshape((1,)*len(self.shape)).expand(self.shape)

  # NOTE: this no longer always breaks the graph
  def contiguous(self):
    return self if self.st.contiguous and self.st.size() == self.base.st.size() and not self.is_unrealized_const() else self.e(LoadOps.CONTIGUOUS)

  def cast(self, dtype:DType, bitcast:bool=False):
    return create_lazybuffer(self.device, ShapeTracker.from_shape(self.shape), dtype, UnaryOps.CAST, (dtype, bitcast), (self,))

  def is_unrealized_const(self): return not self.realized and self.base.op == LoadOps.CONST
  def is_unrealized_contiguous_const(self): return not self.realized and self.op == LoadOps.CONST

  def schedule(self, seen=None): return create_schedule(self, seen)

  @staticmethod
  def fromCPU(x: np.ndarray) -> LazyBuffer:
    ret = LazyBuffer("CPU", ShapeTracker.from_shape(x.shape), dtypes.from_np(x.dtype), op=LoadOps.EMPTY)
    ret._realized = Buffer("CPU", prod(x.shape), dtypes.from_np(x.dtype), x.flatten())
    return ret

  def copy_to_device(self, device:str) -> LazyBuffer:
    # COPY there and back = no COPY at all
    if not self.realized and self.op == LoadOps.COPY and self.srcs[0].device == device: return self.srcs[0]

    # TODO: const doesn't have to be copied (issues with disk tensor)
    #if self.is_unrealized_const(): return self.const(self.base.arg)._view(self.st)
    out = self.contiguous()
    return create_lazybuffer(device, out.st, out.dtype, LoadOps.COPY, srcs=(out,))

  def e(self:LazyBuffer, op:Union[LoadOps, UnaryOps, BinaryOps, TernaryOps], *srcs:LazyBuffer, arg:Optional[Any]=None) -> LazyBuffer:
    srcs = (self,)+srcs
    return create_lazybuffer(self.device, ShapeTracker.from_shape(self.shape), max(x.dtype for x in srcs), op, arg, srcs)

  def r(self:LazyBuffer, op:ReduceOps, new_shape:Tuple[sint, ...]) -> LazyBuffer:
    if self.shape == tuple(new_shape): return self
    return create_lazybuffer(self.device, ShapeTracker.from_shape(new_shape), self.dtype, op, new_shape, (self,))

  def _view(self:LazyBuffer, new_st:ShapeTracker) -> LazyBuffer:
    if new_st.contiguous and self.base.shape == new_st.shape: return self.base
    return create_lazybuffer(self.device, new_st, self.dtype, base=self.base)

  # movement ops
  def reshape(self:LazyBuffer, arg:Tuple[sint, ...]) -> LazyBuffer: return self._view(self.st.reshape(arg))
  def pad(self:LazyBuffer, arg:Tuple[Tuple[int, int], ...]) -> LazyBuffer: return self._view(self.st.pad(arg))
  def expand(self:LazyBuffer, arg:Tuple[sint, ...]) -> LazyBuffer: return self._view(self.st.expand(arg))
  def permute(self:LazyBuffer, arg:Tuple[int, ...]) -> LazyBuffer: return self._view(self.st.permute(arg))
  def shrink(self:LazyBuffer, arg:Tuple[Tuple[sint, sint], ...]) -> LazyBuffer: return self._view(self.st.shrink(arg))
  def stride(self:LazyBuffer, arg:Tuple[int, ...]) -> LazyBuffer: return self._view(self.st.stride(arg))

# *** schedule creation ***

def _recursive_lazyop(buf:LazyBuffer, inputs:List[LazyBuffer], st:ShapeTracker, seen_children:Set[LazyBuffer]):
  if buf != buf.base:
    st = buf.st+st
    buf = buf.base
  # all buffers here are base now
  assert buf.op is not None

  # consts are always fused and generated
  if buf.op == LoadOps.CONST:
    return LazyOp(BufferOps.CONST, (), ConstBuffer(float(buf.arg), buf.dtype, st.simplify().unbind()))

  # if we aren't fusing it, it's a load and we add it to the inputs
  if buf not in seen_children:
    if buf not in inputs: inputs.append(buf)
    return LazyOp(BufferOps.LOAD, (), MemBuffer(inputs.index(buf)+1, buf.dtype, st.simplify().unbind()))

  # if it's a reduce, we have to change the shapetracker
  if buf.op in ReduceOps:
    st = ShapeTracker.from_shape(buf.srcs[0].shape)

  # otherwise we fuse it like normal
  return LazyOp(buf.op, tuple(_recursive_lazyop(x, inputs, st, seen_children) for x in buf.srcs), buf.arg)

# this function determines what fuses
def _get_lazyop(out:LazyBuffer, inputs:List[LazyBuffer], st:ShapeTracker) -> LazyOp:
  potential_inputs = [(out,st)]
  merged_reduce: Optional[LazyBuffer] = None
  output_st = st
  seen_children = set()

  # first we do a (non-recursive) pass to get the inputs and fused reduce
  while len(potential_inputs):
    old, potential_inputs = potential_inputs, []
    for pi,st in old:
      log_lazybuffer(pi)
      if pi.realized: continue  # if it's realized we just use it

      # maybe merge an elementwise op, as long as it doesn't expand and all the children have been seen
      if isinstance(pi.base.op, (UnaryOps, BinaryOps, TernaryOps)) and prod(pi.base.st.shape) >= prod(pi.st.shape):
        allowed = True
        if pi != out:
          for x in pi.base.children:
            if x not in seen_children: allowed = False
        if allowed:
          new_st = pi.st+st if pi.base != pi else st
          potential_inputs += [(x,new_st) for x in pi.base.srcs]
          seen_children.add(pi.base)

      # maybe merge a reduce, if it's contiguous and it's the one we are merging
      elif pi.base.op in ReduceOps and (merged_reduce is None or merged_reduce == pi.base):
        new_st = pi.st+st if pi.base != pi else st
        if new_st.contiguous and new_st.size() == pi.base.st.size():
          merged_reduce = pi.base
          output_st = pi.base.st
          potential_inputs.append((pi.base.srcs[0], pi.base.srcs[0].st))
          seen_children.add(pi.base)

  # then we do a recursive pass to generate the LazyOp
  op = _recursive_lazyop(out, inputs, output_st, seen_children)
  return LazyOp(BufferOps.STORE, (op, ), MemBuffer(0, out.dtype, output_st.simplify().unbind()))

def _create_schedule(out:LazyBuffer, seen:Set[LazyBuffer]) -> List[ScheduleItem]:
  if out in seen or out.realized or out.is_unrealized_const(): return []
  seen.add(out)
  log_lazybuffer(out)
  if out.base is not out: return _create_schedule(out.base, seen)
  assert out.base == out and out.op is not None

  inputs: List[LazyBuffer] = []
  if out.op == LoadOps.COPY:
    log_lazybuffer(out.srcs[0])
    op, inputs = LazyOp(LoadOps.COPY, (), out.srcs[0].base), [out.srcs[0].base]
  elif out.op == LoadOps.CUSTOM:
    op, inputs = LazyOp(LoadOps.CUSTOM, (), out.arg), list(out.srcs)
  elif out.op == LoadOps.EMPTY:
    op = LazyOp(LoadOps.EMPTY)
  else:
    base = out.srcs[0] if out.op == LoadOps.CONTIGUOUS else out
    log_lazybuffer(base)
    op = _get_lazyop(base, inputs, ShapeTracker.from_shape(out.shape))

  ret: List[ScheduleItem] = []
  for x in inputs:
    assert x.base == x, f"all inputs must be base, {x} isn't"
    ret += _create_schedule(x, seen)

  # check if we can reuse the output buffer
  # if it's aliased, don't use it
  if out.output_buffer is not None:
    for i,a in enumerate(inputs):
      # TODO: if this is contiguous it's fine
      if a.realized == out.output_buffer:
        if any(not x.arg.st.contiguous for x in op.get_lazyops() if x.op == BufferOps.LOAD and x.arg.idx == i+1):
          out.output_buffer = None
          break

  if DEBUG >= 5: print_tree(op)

  var_vals = merge_dicts([out.st.var_vals] + [buf.st.var_vals for buf in inputs])
  return ret + [ScheduleItem(op, out, tuple(inputs), {k:var_vals[k] for k in vars_from_ast(op)})]

def create_schedule(out:LazyBuffer, seen:Optional[Set[LazyBuffer]]=None) -> List[ScheduleItem]:
  if seen is None: seen = set()
  log_lazybuffer(out, scheduled=True)
  return _create_schedule(out, seen)
