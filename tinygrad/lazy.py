from __future__ import annotations
import numpy as np
from typing import Union, Optional, Any, Tuple, List
from tinygrad.helpers import prod, dtypes, DType
from tinygrad.ops import LoadOps, UnaryOps, BinaryOps, TernaryOps, ReduceOps, Op
from tinygrad.shape.symbolic import sint
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.device import Buffer

class LazyBuffer:
  def __init__(self, device:str, st:ShapeTracker, dtype:DType, op:Optional[Op]=None, arg:Any=None, srcs:Tuple[LazyBuffer, ...]=(), realized:Optional[Buffer]=None, base:Optional[LazyBuffer]=None):
    self.device, self.st, self.dtype = device, st, dtype
    self.shape = self.st.shape
    assert base is None or base.base == base
    self._base = base
    self.op, self.arg, self.srcs = op, arg, srcs  # this is a LazyOp, except the src is LazyBuffers and not LazyOps
    self._realized = realized
    self.output_buffer: Optional[Buffer] = None

  def __repr__(self) -> str:
    return f"<LB {self.device} {self.shape} contig:{self.st.contiguous} {self.op if hasattr(self, 'op') else self.realized}>"

  @property
  def base(self) -> LazyBuffer: return self._base if self._base is not None else self

  @property
  def realized(self): return self.base._realized

  @staticmethod
  def new(device, shape:Tuple[int], dtype:DType, op, arg): return LazyBuffer(device, ShapeTracker.from_shape(shape), dtype.scalar(), op, arg)

  def const(self, val:Union[float, int]) -> LazyBuffer: return LazyBuffer.new(self.device, self.shape, self.dtype, LoadOps.CONST, val)

  # NOTE: this no longer breaks the graph
  def contiguous(self): return self if self.st.contiguous else self.e(LoadOps.CONTIGUOUS)

  def is_unrealized_const(self): return not self.realized and self.base.op == LoadOps.CONST
  def is_unrealized_contiguous_const(self): return not self.realized and self.op == LoadOps.CONST

  @staticmethod
  def fromCPU(x: np.ndarray) -> LazyBuffer:
    return LazyBuffer("CPU", ShapeTracker.from_shape(x.shape), dtypes.from_np(x.dtype), realized=Buffer("CPU", prod(x.shape), dtypes.from_np(x.dtype), x.flatten()))

  def copy_to_device(self, device:str) -> LazyBuffer:
    out = self.contiguous()
    return LazyBuffer(device, out.st, out.dtype, LoadOps.COPY, srcs=(out,))    # TODO: rename to LoadOps.COPY

  def e(self:LazyBuffer, op:Union[LoadOps, UnaryOps, BinaryOps, TernaryOps], *srcs:LazyBuffer, arg:Optional[Any]=None) -> LazyBuffer:
    srcs = (self,)+srcs
    return LazyBuffer(self.device, ShapeTracker.from_shape(self.shape), max(x.dtype for x in srcs), op, arg, srcs)

  def r(self:LazyBuffer, op:ReduceOps, new_shape:Tuple[sint, ...]) -> LazyBuffer:
    return LazyBuffer(self.device, ShapeTracker.from_shape(new_shape), self.dtype, op, new_shape, (self,))

  def reshape(self:LazyBuffer, arg:Tuple[sint, ...]) -> LazyBuffer: return LazyBuffer(self.device, self.st.reshape(arg), self.dtype, base=self.base)
  def pad(self:LazyBuffer, arg:Tuple[Tuple[int, int], ...]) -> LazyBuffer: return LazyBuffer(self.device, self.st.pad(arg), self.dtype, base=self.base)
  def expand(self: LazyBuffer, arg:Tuple[sint, ...]) -> LazyBuffer: return LazyBuffer(self.device, self.st.expand(arg), self.dtype, base=self.base)
  def permute(self: LazyBuffer, arg:Tuple[int, ...]) -> LazyBuffer: return LazyBuffer(self.device, self.st.permute(arg), self.dtype, base=self.base)
  def shrink(self:LazyBuffer, arg:Tuple[Tuple[sint, sint], ...]) -> LazyBuffer: return LazyBuffer(self.device, self.st.shrink(arg), self.dtype, base=self.base)
  def stride(self:LazyBuffer, arg:Tuple[int, ...]) -> LazyBuffer: return LazyBuffer(self.device, self.st.stride(arg), self.dtype, base=self.base)
