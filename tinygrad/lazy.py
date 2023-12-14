from __future__ import annotations
import numpy as np
from typing import Union, Optional, Any, Tuple
from tinygrad.helpers import prod, dtypes, DType
from tinygrad.ops import LoadOps, UnaryOps, BinaryOps, TernaryOps, ReduceOps, Op
from tinygrad.shape.symbolic import sint
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.device import Buffer
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
