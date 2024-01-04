from __future__ import annotations
from typing import Optional, Union, Any, Tuple, List
import functools
from tinygrad.helpers import all_same, dedup
from tinygrad.dtype import DType
from tinygrad.ops import BinaryOps, LoadOps, UnaryOps, TernaryOps, ReduceOps
from tinygrad.lazy import LazyBuffer, create_schedule
from tinygrad.shape.shapetracker import ShapeTracker, sint

def all_reduce(lbs):
  # TODO: replace this with ring reduce
  return [functools.reduce(lambda x,y: x.e(BinaryOps.ADD, y), [x.copy_to_device(lb.device) for x in lbs]) for lb in lbs]

def to_sharded(lbs:List[LazyBuffer], axis:int) -> List[LazyBuffer]:
  assert lbs[0].shape[axis] % len(lbs) == 0, f"{lbs[0].shape=} {axis=} {len(lbs)=}"
  sz = lbs[0].shape[axis] // len(lbs)
  return [lb.shrink(tuple((0,s) if a != axis else (sz*i,sz*(i+1)) for a,s in enumerate(lb.shape))) for i,lb in enumerate(lbs)]

class MultiLazyBuffer:
  def __init__(self, lbs:List[LazyBuffer], axis:Optional[int]):
    assert all(isinstance(x, LazyBuffer) for x in lbs) and len(lbs) >= 2, "all lbs must be LazyBuffers, and we need at least two of them"
    assert all_same([(x.shape, x.dtype, x.st) for x in lbs]), "all multilazybuffer needs same shape, dtype, and st"
    self.lbs, self.axis, self.dtype, self.device = lbs, axis, lbs[0].dtype, tuple(x.device for x in lbs)
    self.shape = tuple(s*len(self.lbs) if a == self.axis else s for a,s in enumerate(lbs[0].shape))

  def __repr__(self):
    return f"<MLB{chr(10)}{chr(10).join([f'{x.device} {x.st}' for x in self.lbs])}>"

  @staticmethod
  def from_sharded(lb:LazyBuffer, devices:Tuple[str, ...], axis:Optional[int]=None):
    lbs = [lb.contiguous() if lb.base != lb else lb] * len(devices)
    return MultiLazyBuffer([lb.copy_to_device(d).contiguous() for lb,d in zip(to_sharded(lbs, axis) if axis is not None else lbs, devices)], axis)

  def copy_to_device(self, device:str) -> LazyBuffer:
    if self.axis is None: return self.lbs[0].copy_to_device(device)
    sz = self.lbs[0].shape[self.axis]
    llbs = []
    for i,lb in enumerate([lb.copy_to_device(device) for lb in self.lbs]):
      pad_arg = tuple((0,0) if a != self.axis else (sz*i,(s*len(self.lbs))-sz*(i+1)) for a,s in enumerate(lb.shape))
      llbs.append(lb.pad(pad_arg))
    return functools.reduce(lambda x,y: x.e(BinaryOps.ADD, y), llbs)

  # TODO: fix this
  def is_unrealized_contiguous_const(self): return False

  # passthroughs
  def schedule(self, seen=None): return create_schedule(self.lbs, seen)
  def cast(self, dtype:DType, bitcast:bool=False): return MultiLazyBuffer([x.cast(dtype, bitcast) for x in self.lbs], self.axis)
  def const(self, val:Union[float, int]) -> MultiLazyBuffer: return MultiLazyBuffer([x.const(val) for x in self.lbs], self.axis)
  def contiguous(self): return MultiLazyBuffer([x.contiguous() for x in self.lbs], self.axis)

  # elementwise is simple
  def e(self, op:Union[LoadOps, UnaryOps, BinaryOps, TernaryOps], *in_srcs:MultiLazyBuffer, arg:Optional[Any]=None) -> MultiLazyBuffer:
    msrcs = (self,)+in_srcs
    assert all(isinstance(x, MultiLazyBuffer) for x in msrcs), f"all buffers must be MultiLazyBuffer {msrcs}"
    assert all_same([x.device for x in msrcs]), f"all buffers must have the same device {[x.device for x in msrcs]}"

    # NOTE: they all have to share an axis, we always choose [-1]
    axis = axes[-1] if len(axes := dedup([x.axis for x in msrcs if x.axis is not None])) else None
    srcs = []
    for mlb in msrcs:
      if mlb.axis == axis: srcs.append(mlb.lbs)
      elif mlb.axis is None and axis is not None: srcs.append(to_sharded(mlb.lbs, axis))
      else: srcs.append(to_sharded([mlb.copy_to_device(lb.device) for lb in mlb.lbs], axis))
    return MultiLazyBuffer([lsrcs[0].e(op, *lsrcs[1:], arg=arg) for lsrcs in zip(*srcs)], axis)

  def _shape_to_single_shard(self, shape): return tuple(s//len(self.lbs) if a == self.axis else s for a,s in enumerate(shape))

  def r(self, op:ReduceOps, new_shape:Tuple[sint, ...]) -> MultiLazyBuffer:
    if self.axis is not None and new_shape[self.axis] == 1:
      # all-reduce on sharded axes
      return MultiLazyBuffer(all_reduce([x.r(op, new_shape) for x in self.lbs]), None)
    # reduce on non sharded axes, piecewise is fine. if axis is None this is also correct
    return MultiLazyBuffer([x.r(op, self._shape_to_single_shard(new_shape)) for x in self.lbs], self.axis)

  # *** movement ops ***

  def reshape(self, arg:Tuple[sint, ...]):
    if self.axis is None: return MultiLazyBuffer([x.reshape(arg) for x in self.lbs], None)
    # TODO: this can be wrong
    st = ShapeTracker.from_shape(self.shape)
    rs = st.real_strides()[self.axis]
    new_axis = st.reshape(arg).real_strides().index(rs)
    narg = tuple(s//len(self.lbs) if a == new_axis else s for a,s in enumerate(arg))
    return MultiLazyBuffer([x.reshape(narg) for x in self.lbs], new_axis)

  def pad(self, arg:Tuple[Tuple[sint, sint], ...]):
    assert self.axis is None or arg[self.axis] == (0,0), "padding not supported on sharded axis"
    return MultiLazyBuffer([x.pad(arg) for x in self.lbs], self.axis)
  def expand(self, arg:Tuple[sint, ...]):
    # NOTE: this assert isn't needed, sharded axis can have dim 1
    assert self.axis is None or arg[self.axis] == self.lbs[0].shape[self.axis] * len(self.lbs), "expand not supported on sharded axis"
    return MultiLazyBuffer([x.expand(self._shape_to_single_shard(arg)) for x in self.lbs], self.axis)
  def permute(self, arg:Tuple[int, ...]):
    # all permutes supported!
    return MultiLazyBuffer([x.permute(arg) for x in self.lbs], arg.index(self.axis) if self.axis is not None else None)
  def shrink(self, arg:Tuple[Tuple[sint, sint], ...]):
    assert self.axis is None or arg[self.axis] == (0, self.lbs[0].shape[self.axis] * len(self.lbs)), "shrinking not supported on sharded axis"
    narg = tuple((s1//len(self.lbs), s2//len(self.lbs)) if a == self.axis else (s1,s2) for a,(s1,s2) in enumerate(arg))
    return MultiLazyBuffer([x.shrink(narg) for x in self.lbs], self.axis)
  def stride(self, arg:Tuple[int, ...]):
    assert self.axis is None or arg[self.axis] == 1, "flipping not supported on sharded axis"
    return MultiLazyBuffer([x.stride(arg) for x in self.lbs], self.axis)
