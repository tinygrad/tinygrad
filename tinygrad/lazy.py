from __future__ import annotations
import numpy as np
from typing import Union, Optional, Any, Tuple, List
from tinygrad.helpers import prod, dtypes, DType
from tinygrad.ops import LoadOps, UnaryOps, BinaryOps, TernaryOps, ReduceOps, Op
from tinygrad.shape.symbolic import sint
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.device import Buffer

class LazySrc:
  op: Optional[Op] = None
  src: Tuple[LazyBuffer, ...] = ()
  arg: Any = None

class LazyBuffer:
  def __init__(self, device:Tuple[str], st:Tuple[ShapeTracker], dtype:DType, op:Optional[Op]=None, arg:Any=None, src:Tuple[LazyBuffer, ...]=(), realized:Optional[List[Buffer]]=None, base:Optional[LazyBuffer]=None):
    self.device, self.st, self.dtype = device, st, dtype
    self.shape = self.st[0].shape
    self._base = base
    self.op, self.arg, self.src = op, arg, src
    self.realized = realized

  @property
  def base(self) -> LazyBuffer: return self._base if self._base is not None else self

  def is_unrealized_const(self): return not self.realized and self.base.op == LoadOps.CONST
  def is_unrealized_contiguous_const(self): return not self.realized and self.op == LoadOps.CONST

  @staticmethod
  def fromCPU(x: np.ndarray) -> LazyBuffer:
    return LazyBuffer(["CPU"], (ShapeTracker.from_shape(x.shape),), dtypes.from_np(x.dtype), realized=[Buffer("CPU", prod(x.shape), dtypes.from_np(x.dtype), x.flatten())])

  def copy_to_device(self, device:Tuple[str]) -> LazyBuffer:
    out = self.e(LoadOps.CONTIGUOUS)
    return LazyBuffer(device, out.st, out.dtype, LoadOps.COPY, src=(out,))    # TODO: rename to LoadOps.COPY

  def e(self:LazyBuffer, op:Union[LoadOps, UnaryOps, BinaryOps, TernaryOps], *srcs:LazyBuffer, arg:Optional[Any]=None) -> LazyBuffer:
    srcs = (self,)+srcs
    return LazyBuffer(self.device, tuple(ShapeTracker.from_shape(self.shape) for _ in self.st), max(x.dtype for x in srcs), op, arg, srcs)

  def r(self:LazyBuffer, op:ReduceOps, new_shape:Tuple[sint, ...]) -> LazyBuffer:
    return LazyBuffer(self.device, tuple(ShapeTracker.from_shape(new_shape) for _ in self.st), self.dtype, op, new_shape, (self,))

  def reshape(self:LazyBuffer, arg:Tuple[sint, ...]) -> LazyBuffer: return LazyBuffer(self.device, tuple(x.reshape(arg) for x in self.st), self.dtype, base=self)
  def pad(self:LazyBuffer, arg:Tuple[Tuple[int, int], ...]) -> LazyBuffer: return LazyBuffer(self.device, tuple(x.pad(arg) for x in self.st), self.dtype, base=self)
  def expand(self: LazyBuffer, arg:Tuple[sint, ...]) -> LazyBuffer: return LazyBuffer(self.device, tuple(x.expand(arg) for x in self.st), self.dtype, base=self)
  def permute(self: LazyBuffer, arg:Tuple[int, ...]) -> LazyBuffer: return LazyBuffer(self.device, tuple(x.permute(arg) for x in self.st), self.dtype, base=self)
  def shrink(self:LazyBuffer, arg:Tuple[Tuple[sint, sint], ...]) -> LazyBuffer: return LazyBuffer(self.device, tuple(x.shrink(arg) for x in self.st), self.dtype, base=self)
  def stride(self:LazyBuffer, arg:Tuple[int, ...]) -> LazyBuffer: return LazyBuffer(self.device, tuple(x.stride(arg) for x in self.st), self.dtype, base=self)
