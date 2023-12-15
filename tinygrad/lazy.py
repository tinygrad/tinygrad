from __future__ import annotations
import numpy as np
from typing import Union, Optional, Any, Tuple, List, Set, Dict
from tinygrad.helpers import prod, dtypes, DType, merge_dicts, dedup, flatten
from tinygrad.ops import LoadOps, UnaryOps, BinaryOps, TernaryOps, ReduceOps, BufferOps
from tinygrad.ops import Op, LazyOp, ConstBuffer, MemBuffer, ScheduleItem, vars_from_ast
from tinygrad.shape.symbolic import sint
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.device import Buffer
from tinygrad.graph import log_lazybuffer
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
  # TODO: might be possible to remove LoadOps.COPY
  if op not in {LoadOps.EMPTY, LoadOps.CUSTOM, LoadOps.CONST, LoadOps.COPY}: lazycache[wop] = ret
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

  def schedule(self, seen=None): return create_schedule([self], seen)

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

def _recursive_lazyop(buf:LazyBuffer, inputs:List[LazyBuffer], st:ShapeTracker, realizes:Set[LazyBuffer], first=True):
  if buf != buf.base:
    st = buf.st+st
    buf = buf.base
  # all buffers here are base now
  assert buf.op is not None

  # consts are always fused and generated
  if buf.op == LoadOps.CONST:
    return LazyOp(BufferOps.CONST, (), ConstBuffer(float(buf.arg), buf.dtype, st.simplify().unbind()))

  # if we aren't fusing it, it's a load and we add it to the inputs
  if buf.realized or (buf in realizes and (not first or buf.op in LoadOps)):
    if buf not in inputs: inputs.append(buf)
    return LazyOp(BufferOps.LOAD, (), MemBuffer(inputs.index(buf)+1, buf.dtype, st.simplify().unbind()))

  # if it's a reduce, we have to change the shapetracker
  if buf.op in ReduceOps:
    assert st.contiguous, "ReduceOps late fusion must be contiguous"
    st = ShapeTracker.from_shape(buf.srcs[0].shape)

  # otherwise we fuse it like normal
  return LazyOp(buf.op, tuple(_recursive_lazyop(x, inputs, st, realizes, False) for x in buf.srcs), buf.arg)

def _get_bufs_for_chunk(out:LazyBuffer, realized:Set[LazyBuffer], first=True) -> Set[LazyBuffer]:
  if out.realized or (out in realized and not first): return set()
  return set([out]).union(*[_get_bufs_for_chunk(x.base, realized, False) for x in out.srcs])

def create_schedule(outs:List[LazyBuffer], seen:Optional[Set[LazyBuffer]]=None) -> List[ScheduleItem]:
  if seen is None: seen = set()
  for out in outs: log_lazybuffer(out, scheduled=True)

  # realize all places where the buffer is expanded
  realized: Set[LazyBuffer] = set([x.base for x in outs if not x.realized])
  allbufs: Dict[LazyBuffer, None] = {}
  def recurse_lb(buf:LazyBuffer):
    if buf in allbufs or buf.realized: return
    allbufs[buf] = None
    log_lazybuffer(buf)
    if buf.base != buf:
      if prod(buf.base.st.shape) < prod(buf.st.shape):
        realized.add(buf.base)
      return recurse_lb(buf.base)
    if buf.op in LoadOps: realized.add(buf.base)
    for x in buf.srcs: recurse_lb(x)
  for out in outs: recurse_lb(out.base)

  # find all reduces, and pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce
  reduce_for_op: Dict[LazyBuffer, LazyBuffer] = {}
  for r in allbufs.keys():
    if r != r.base or r.op not in ReduceOps or r in realized: continue

    # follow the reduce down
    child_set: Dict[LazyBuffer, ShapeTracker] = {r: r.st}
    realized_children: Dict[LazyBuffer, ShapeTracker] = {}
    forced_realized = False
    while not forced_realized and len(child_set):
      next_child_set = {}
      for tr,st in child_set.items():
        if tr in realized:
          realized_children[tr] = st
          # can only have one output buffer
          # can only reduce contiguous
          # max one reduceop per kernel
          if len(realized_children) > 1 or not st.contiguous or (tr in reduce_for_op and reduce_for_op[tr] != r):
            forced_realized = True
            break
          continue
        for tr_next in tr.children:
          if not tr_next.realized:
            # max one reduceop per kernel
            if tr_next.op in ReduceOps:
              forced_realized = True
              break
            next_child_set[tr_next] = st + [s for s in tr_next.srcs if s.base == tr][0].st
      child_set = next_child_set
    if forced_realized:
      # TODO: can chase this down to contiguous children
      realized.add(r)
    else:
      assert len(realized_children) == 1
      reduce_for_op[next(iter(realized_children.keys()))] = r

  # schedule
  def recursive_schedule(out:LazyBuffer) -> List[ScheduleItem]:
    if out in seen or out.realized or out.op == LoadOps.CONST: return []
    assert out.base == out
    seen.add(out)

    inputs: List[LazyBuffer] = []
    if out.op == LoadOps.COPY:
      op, inputs = LazyOp(LoadOps.COPY, (), out.srcs[0].base), [out.srcs[0].base]
    elif out.op == LoadOps.CUSTOM:
      op, inputs = LazyOp(LoadOps.CUSTOM, (), out.arg), list(out.srcs)
    elif out.op == LoadOps.EMPTY:
      op = LazyOp(LoadOps.EMPTY)
    else:
      base = out.srcs[0] if out.op == LoadOps.CONTIGUOUS else out
      output_st = ShapeTracker.from_shape(reduce_for_op[out].shape if out in reduce_for_op else base.shape)
      op = _recursive_lazyop(base, inputs, output_st, realized)
      op = LazyOp(BufferOps.STORE, (op, ), MemBuffer(0, base.dtype, output_st.simplify().unbind()))

    var_vals = merge_dicts([out.st.var_vals] + [buf.st.var_vals for buf in inputs])
    return flatten(recursive_schedule(x.base) for x in inputs) + [ScheduleItem(op, out, tuple(inputs), {k:var_vals[k] for k in vars_from_ast(op)})]
  return flatten(recursive_schedule(x.base) for x in outs)