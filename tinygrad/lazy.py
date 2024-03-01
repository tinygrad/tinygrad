from __future__ import annotations
import math
from typing import Union, Optional, Any, Tuple, List, Dict, cast
from tinygrad.dtype import cast_scalar, dtypes, DType, Scalar
from tinygrad.helpers import prod, getenv, all_int, all_same
from tinygrad.ops import LoadOps, UnaryOps, BinaryOps, TernaryOps, ReduceOps, Op
from tinygrad.shape.symbolic import sint
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.device import Buffer
from weakref import ref, ReferenceType

lazycache: Dict[Any, ReferenceType[LazyBuffer]] = {}
def create_lazybuffer(device:str, st:ShapeTracker, dtype:DType, op:Optional[Op]=None, arg:Any=None, srcs:Tuple[LazyBuffer, ...]=(),
                      base:Optional[LazyBuffer]=None, enable_cache=bool(getenv("LAZYCACHE", 1))):
  if st.size == 0 and op not in {LoadOps.SYNC, LoadOps.WAIT}: op, arg, srcs, base = LoadOps.CONST, 0, (), None

  cache_key = (device, st, dtype, op, arg, tuple(ref(x) for x in srcs)) if base is None else (st, ref(base))
  if (rret := lazycache.get(cache_key, None)): return cast(LazyBuffer, rret())  # NOTE: this should always be a live reference

  return LazyBuffer(device, st, dtype, op, arg, srcs, base=base, cache_key=cache_key if enable_cache else None)

class LazyBuffer:
  def __init__(self, device:str, st:ShapeTracker, dtype:DType,
               op:Optional[Op]=None, arg:Any=None, srcs:Tuple[LazyBuffer, ...]=(),
               base:Optional[LazyBuffer]=None, cache_key=None):
    self.device, self.st, self.dtype, self.shape, self.size, self.cache_key = device, st, dtype, st.shape, st.size, cache_key
    self._base: Optional[LazyBuffer] = None
    if base is None:
      # properties on base
      self.op, self.arg, self.srcs = op, arg, srcs  # this is a LazyOp, except the src is LazyBuffers and not LazyOps
      self.realized: Optional[Buffer] = None
      self.output_buffer: Optional[Buffer] = None
      self.contiguous_child: Optional[Tuple[ReferenceType[LazyBuffer], ShapeTracker]] = None
      self.forced_realize = False
    else:
      # properties on view
      assert base.base == base, "base must be a base itself"
      self._base = base
    if cache_key is not None: lazycache[cache_key] = ref(self)

  def __del__(self): lazycache.pop(self.cache_key, None)

  def __repr__(self) -> str:
    return f"<LB {self.device} {self.shape} contig:{self.st.contiguous} {self.st if self.base != self else (self.op, self.realized)}>"

  # NOTE: this has to be a function to prevent self reference
  @property
  def base(self) -> LazyBuffer: return self._base if self._base is not None else self

  @staticmethod
  def loadop(op, shape:Tuple[sint,...], dtype:DType, device:str, arg=None, src:Optional[LazyBuffer]=None, enable_cache=False) -> LazyBuffer:
    return create_lazybuffer(device, ShapeTracker.from_shape(shape), dtype, op, arg, (src,) if src is not None else (), enable_cache=enable_cache)

  def const(self, val:Scalar, shape:Optional[Tuple[sint,...]]=None) -> LazyBuffer:
    shape = self.shape if shape is None else shape
    return LazyBuffer.loadop(LoadOps.CONST, tuple(), self.dtype, self.device, arg=cast_scalar(val, self.dtype)).reshape((1,)*len(shape)).expand(shape)

  def contiguous(self):
    if not self.st.contiguous or self.size != self.base.size or self.is_unrealized_const():
      ret = self.e(LoadOps.CONTIGUOUS)
      if (sti := self.st.invert(self.base.shape)) is not None: self.base.contiguous_child = ref(ret), sti
      return ret
    self.base.forced_realize = True
    return self

  def cast(self, dtype:DType, bitcast:bool=False):
    if self.dtype == dtype: return self
    # TODO: applying this makes gpt2 slower
    if getenv("CAST_BEFORE_VIEW", 1) and dtype.itemsize <= self.dtype.itemsize and self != self.base:
      return self.base.cast(dtype, bitcast)._view(self.st)
    return create_lazybuffer(self.device, ShapeTracker.from_shape(self.shape), dtype, UnaryOps.CAST, (dtype, bitcast), (self,))

  def is_unrealized_const(self): return not self.base.realized and self.base.op is LoadOps.CONST
  def is_unrealized_contiguous_const(self): return self.base == self and not self.base.realized and self.op is LoadOps.CONST

  def _copy(self, device:str) -> LazyBuffer:
    sync_size = 1 if self.device.startswith("HIP") else 0
    if self.device.startswith("EXT") or self.device.startswith("DISK"):
      # DISK/EXT don't sync
      return create_lazybuffer(device, ShapeTracker.from_shape(self.shape), self.dtype, LoadOps.COPY, None, (self,), enable_cache=False)
    sync = LazyBuffer.loadop(LoadOps.SYNC, (sync_size,), dtypes.uint32, self.device, src=self, enable_cache=True)
    wait = LazyBuffer.loadop(LoadOps.WAIT, (0,), dtypes.uint32, device, src=sync, enable_cache=True)
    return create_lazybuffer(device, ShapeTracker.from_shape(self.shape), self.dtype, LoadOps.COPY, None, (self, wait), enable_cache=False)

  def copy_to_device(self, device:str) -> LazyBuffer:
    # no COPY
    if self.device == device: return self

    # double COPY = one COPY
    if self.st.contiguous and self.size == self.base.size and not self.base.realized and self.base.op is LoadOps.COPY:
      return self.base.srcs[0].copy_to_device(device).reshape(self.st.shape)

    # const doesn't have to be copied (issues with disk tensor)
    if self.is_unrealized_const():
      return LazyBuffer.loadop(LoadOps.CONST, tuple(), self.dtype, device, arg=self.base.arg)._view(self.st)

    # if it's a shrink, do the shrink before the copy with CONTIGUOUS
    if prod(self.st.shape) < prod(self.base.st.shape): return self.contiguous()._copy(device)

    # copy the base and apply the shapetracker on the new device
    return self.base._copy(device)._view(self.st)

  def e(self, op:Union[LoadOps, UnaryOps, BinaryOps, TernaryOps], *in_srcs:LazyBuffer, arg:Optional[Any]=None) -> LazyBuffer:
    srcs: List[LazyBuffer] = []
    for s in (self,)+in_srcs:
      if s == s.base and s.base.contiguous_child and (root:=s.base.contiguous_child[0]()) is not None:
        srcs.append(root._view(s.base.contiguous_child[1]))
      else:
        srcs.append(s)
    assert all_same(dts:=[x.dtype.scalar() for x in (srcs[1:] if op is TernaryOps.WHERE else srcs)]), f"all dtypes must match {dts} on {op}"
    assert all_same([x.shape for x in srcs]), f"all shapes must be the same {[x.shape for x in srcs]}"
    if op is TernaryOps.WHERE: assert srcs[0].dtype == dtypes.bool, "TernaryOps.WHERE must have the first arg be bool"
    if op is UnaryOps.NEG: assert srcs[0].dtype != dtypes.bool, "UnaryOps.NEG does not accept dtype bool"
    out_dtype = dtypes.bool if op in (BinaryOps.CMPLT, BinaryOps.CMPEQ) else srcs[-1].dtype
    return create_lazybuffer(self.device, ShapeTracker.from_shape(self.shape), out_dtype, op, arg, tuple(srcs))

  # *** reduce ops ***

  def _reduce_op(self, op:ReduceOps, axis:Tuple[int, ...]) -> LazyBuffer:
    assert all(0 <= x < len(self.shape) for x in axis), f"axis args {axis} out of range for shape {self.shape}"
    axis = tuple(x for x in axis if self.shape[x] != 1)
    if len(axis) == 0: return self
    new_shape = tuple(1 if i in axis else s for i,s in enumerate(self.shape))
    return create_lazybuffer(self.device, ShapeTracker.from_shape(new_shape), self.dtype, op, axis, (self,))

  def r(self, op:ReduceOps, axis:Tuple[int, ...]) -> LazyBuffer:
    new_shape = tuple(1 if i in axis else s for i,s in enumerate(self.shape))
    # TODO: this logic should move to the scheduler
    if self.size == 0 and 0 not in new_shape: return self.const({ReduceOps.SUM: 0.0, ReduceOps.MAX: -math.inf}[op], new_shape)
    # TODO: can we split symbolic shape if the reduce axis is not symbolic?
    if not all_int(self.shape) or (0 in self.shape) or prod(self.shape) // prod(new_shape) < getenv("REDUCEOP_SPLIT_THRESHOLD", 32768):
      return self._reduce_op(op, axis)
    heuristic, divisor, dim_to_split = max(((divisor := math.gcd(256, s))/(st or math.inf), divisor, i) for i,(s,st) in \
                                           enumerate(zip(self.shape, self.st.real_strides())) if i in axis and (st is None or isinstance(st, int)))
    if divisor < 16 or heuristic < 0.1: return self._reduce_op(op, axis)
    # choose largest divisor (>=16) to split on, penalize large strides
    def splitted_shape(dim_aft_div):
      return self.shape[:dim_to_split] + (self.shape[dim_to_split]//divisor,) + dim_aft_div + self.shape[dim_to_split+1:]
    return self.reshape(splitted_shape((divisor,)))._reduce_op(op, (dim_to_split+1,)).reshape(splitted_shape(()))._reduce_op(op, axis)

  # *** movement ops ***

  def _view(self, new_st:ShapeTracker) -> LazyBuffer:
    if self.st.size == 0 or (new_st.views[-1].mask is not None and all((x[1]-x[0]) == 0 for x in new_st.views[-1].mask)):
      return self.const(0, new_st.shape)
    if new_st.contiguous and self.base.shape == new_st.shape: return self.base
    return create_lazybuffer(self.device, new_st, self.dtype, base=self.base)

  def reshape(self, arg:Tuple[sint, ...]): return self._view(self.st.reshape(arg))
  def pad(self, arg:Tuple[Tuple[sint, sint], ...]): return self._view(self.st.pad(arg))
  def expand(self, arg:Tuple[sint, ...]): return self._view(self.st.expand(arg))
  def permute(self, arg:Tuple[int, ...]): return self._view(self.st.permute(arg))
  def shrink(self, arg:Tuple[Tuple[sint, sint], ...]): return self._view(self.st.shrink(arg))
  def stride(self, arg:Tuple[int, ...]): return self._view(self.st.stride(arg))
