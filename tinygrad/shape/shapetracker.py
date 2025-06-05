# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
from dataclasses import dataclass
import functools
from typing import Optional, Callable, Any
from tinygrad.helpers import merge_dicts, getenv, prod
from tinygrad.shape.view import View, strides_for_shape, unravel
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops, graph_rewrite, Variable, sint, sint_to_uop, Context, PatternMatcher, UPat, GroupOp
from tinygrad.codegen.symbolic import split_uop, symbolic_flat, uop_given_valid, simplify_valid

# If a node overflow, its srcs need to be checked to see if this overflow is the result of an ALU operation,
# or that the node simply inherits the dtype from srcs. Upcast is either `Ops.CAST`+`replace` or just `replace`.
def handle_upcast(u: UOp) -> UOp|None:
  dtype = dtypes.int64.vec(u.dtype.count) if u.dtype.count > 1 else dtypes.int64
  # check for overflow, upcast this to int64
  if u.vmax > dtypes.max(dtypes.int) or u.vmin < dtypes.min(dtypes.int):
    return u.replace(dtype=dtype, src=tuple([x.cast(dtype) for x in u.src]))
  # if any inputs are int64 and this *doesn't* overflow, cast back to int
  if any(x.dtype == dtypes.int64 for x in u.src):
    return u.replace(dtype=dtype, src=tuple([x.cast(dtype) for x in u.src])).cast(u.dtype)
  return None
pm_upcast = PatternMatcher([(UPat(GroupOp.ALU, dtype=dtypes.int, name="u"), handle_upcast),])

def apply_mop(st: Any|ShapeTracker, mop_arg: tuple[Ops, tuple]) -> ShapeTracker:
  mop, arg = mop_arg
  if mop == Ops.RESHAPE:
    # shapetracker doesn't allow flattening with -1 but required for Ops.RESHAPE
    if arg == (-1,): return st.reshape((prod(st.shape),))
    return st.reshape(arg)
  if mop == Ops.PERMUTE: return st.permute(arg)
  if mop == Ops.EXPAND:
    if len(arg) != len(st.shape): st = st.reshape((1,*st.shape))
    return st.expand(arg)
  if mop == Ops.PAD: return st.pad(arg)
  if mop == Ops.SHRINK: return st.shrink(arg)
  if mop == Ops.FLIP: return st.flip(arg)
  raise ValueError("invalid mop")

@functools.cache
def st_to_movement_ops(st) -> list[tuple[Ops, Any]]:
  if 0 in st.shape: return []
  ops:list[tuple[Ops, tuple]] = []
  for i, v in enumerate(st.views):
    shape = tuple(y-x for x,y in v.mask) if v.mask else v.shape
    offset = v.offset + sum(st*(s-1) for s,st in zip(shape, v.strides) if st<0)
    pos = offset + (sum(x*st for (x,_),st in zip(v.mask, v.strides)) if v.mask else 0)
    dims = [s for s,st in zip(shape, v.strides) if st]
    strides: list[int] = [abs(st) if isinstance(st,int) else st for st in v.strides if st]
    buffer = sum((s-1)*st for s,st in zip(dims,strides)) + 1
    if i: buffer = prod(st.views[i-1].shape) - pos if dims else 1
    order, pairs = map(list, zip(*sorted(enumerate(zip(dims, strides)), key=lambda p: (p[1][1], -p[1][0]), reverse=True))) if dims else ([], [])
    ops.extend([(Ops.RESHAPE, (-1,)), (Ops.SHRINK, ((pos, pos+buffer),))])
    if strides:
      if (pairs[0][0]*pairs[0][1]) - buffer > 0:
        ops.append((Ops.PAD, ((0, (pairs[0][0] * pairs[0][1]) - buffer),)))
      for j, shape_stride in enumerate(pairs):
        if j<len(pairs)-1 and shape_stride[1] < pairs[j+1][0]*pairs[j+1][1]:
          remaining_buffer = pairs[j-1][1] if j>0 else buffer
          ops.append((Ops.EXPAND, (shape_stride[0], *(s[0] for s in pairs[:j]), remaining_buffer)))
          ops.append((Ops.PERMUTE, (*range(1,j+1), 0, j+1)))
          ops.append((Ops.RESHAPE, (*(s[0] for s in pairs[:j]), shape_stride[0]*remaining_buffer)))
          ops.append((Ops.PAD, (*((0,0) for _ in range(j)), (0, shape_stride[0]*shape_stride[1]))))
          ops.append((Ops.RESHAPE, (*(s[0] for s in pairs[:j+1]), remaining_buffer+shape_stride[1])))
          pairs[j] = (pairs[j][0], remaining_buffer+shape_stride[1])
        else:
          ops.append((Ops.SHRINK, (*((0, s[0]) for s in pairs[:j]), (0, shape_stride[0]*shape_stride[1]))))
          ops.append((Ops.RESHAPE, (*[s[0] for s in pairs[:j+1]], shape_stride[1])))
      ops.extend([(Ops.SHRINK, (*[(0, s[0]) for s in pairs], (0,1))), (Ops.RESHAPE, tuple(s[0] for s in pairs))])
      if order != list(range(len(order))): ops.append((Ops.PERMUTE, tuple(order.index(i) for i in range(len(strides)))))
    ops.append((Ops.RESHAPE, tuple(s if st else 1 for s,st in zip(shape, v.strides))))
    if any(i<0 for i in v.strides): ops.append((Ops.FLIP, tuple(-1 if st<0 else 1 for st in v.strides)))
    # then, we apply pre expand pads
    if v.mask is not None:
      pre_expand_pads = tuple((x,s-y) if st != 0 else (0,0) for (x,y),s,st in zip(v.mask, v.shape, v.strides))
      post_expand_pads = tuple((x,s-y) if st == 0 else (0,0) for (x,y),s,st in zip(v.mask, v.shape, v.strides))
      if any(x != (0,0) for x in pre_expand_pads):
        ops.append((Ops.PAD, pre_expand_pads))
        shape = tuple(x+s[0]+s[1] for x,s in zip(shape, pre_expand_pads))
    if any(s != 1 and st == 0 for s,st in zip(shape, v.strides)): ops.append((Ops.EXPAND, shape))
    if v.mask is not None and any(x != (0,0) for x in post_expand_pads): ops.append((Ops.PAD, post_expand_pads))
  return ops

@functools.cache
def views_to_indexed_uops(views: tuple[View, ...], _idxs:Optional[tuple[UOp, ...]]=None) -> tuple[UOp, UOp]:
  idx, valid = views[-1].to_indexed_uops(_idxs)
  for view in reversed(views[0:-1]):
    view = view.minify()
    idx, valid = view.to_indexed_uops([sint_to_uop(i) for i in unravel(view.shape, idx)], valid)
  # symbolic
  idx, valid = graph_rewrite(UOp.sink(idx, valid), symbolic_flat, name="indexing sym @ 1").src
  # simplify
  if (newvalid:=simplify_valid(valid)) is not None: valid = newvalid
  if (newidx:=uop_given_valid(valid, idx)) is not None: idx = newidx
  # symbolic again, upcast if needed
  return graph_rewrite(UOp.sink(idx, valid), symbolic_flat+pm_upcast, name="indexing sym @ 2").src

@functools.cache
def views_to_real_strides(views: tuple[View, ...], ignore_valid=False) -> tuple[Optional[sint], ...]:
  # NOTE: if a stride is not always valid, it will be None
  if len(views) == 1 and views[-1].mask is None: return views[-1].strides
  ret: list[Optional[sint]] = [None] * len(views[-1].shape)
  idx, valid = views_to_indexed_uops(views)
  for c in split_uop(idx, Ops.ADD):
    if c.op is Ops.RANGE: ret[c.arg] = 1
    if c.op is Ops.MUL and c.src[0].op is Ops.RANGE and c.src[1].op is Ops.CONST: ret[c.src[0].arg] = c.src[1].arg
    if c.op is Ops.MUL and c.src[1].op is Ops.RANGE and c.src[0].op is Ops.CONST: ret[c.src[1].arg] = c.src[0].arg
  used_ranges = [x.arg for x in idx.toposort() if x.op is Ops.RANGE]
  ret = [x if i in used_ranges else 0 for i,x in enumerate(ret)]
  if not ignore_valid:
    for masked_axis in [x.arg for x in valid.toposort() if x.op is Ops.RANGE]: ret[masked_axis] = None
  return tuple(ret)

@dataclass(frozen=True, order=True)
class ShapeTracker:
  views: tuple[View, ...]

  def __add__(self, st:ShapeTracker) -> ShapeTracker:
    ret = self
    for v in st.views: ret = ShapeTracker(ret.views + (v,)).simplify() # one view at a time = better simplification
    return ret

  def invert(self, out_shape:tuple[sint, ...]) -> Optional[ShapeTracker]:
    inverted_views:list[View] = []
    for v,s in zip(self.views[::-1], [x.shape for x in self.views[::-1][1:]]+[out_shape]):
      if (inverted:= v.invert(s)) is None: return None
      inverted_views.append(inverted)
    return ShapeTracker(tuple(inverted_views)).reshape(out_shape)

  @staticmethod
  def from_shape(shape:tuple[sint, ...], strides:tuple[sint, ...]|None=None) -> ShapeTracker: return ShapeTracker((View.create(shape, strides),))

  @property
  def contiguous(self) -> bool: return len(self.views) == 1 and self.views[0].contiguous

  @property
  def consecutive(self) -> bool: return len(self.views) == 1 and (v:=self.views[0]).mask is None and v.strides == strides_for_shape(v.shape)

  @property
  def shape(self) -> tuple[sint, ...]: return self.views[-1].shape

  @property
  def size(self) -> int: return self.views[-1].size()

  def reduce(self, axis:tuple[int, ...]) -> tuple[sint, ...]: return tuple(1 if i in axis else s for i,s in enumerate(self.shape))

  def to_uop(self) -> UOp: return UOp(Ops.VIEW, dtypes.void, (), self)
  def to_indexed_uops(self, _idxs:Optional[list[UOp]|tuple[UOp, ...]]=None) -> tuple[UOp, UOp]:
    return views_to_indexed_uops(self.views, tuple(_idxs) if _idxs is not None else None)

  def to_movement_ops(self) -> list[tuple[Ops, Any]]: return st_to_movement_ops(self)
  # upper bound on buffer size required to fit this shapetracker
  def real_size(self) -> int:
    if 0 in self.shape: return 0
    view = (v.shrink(v.mask) if (v:=self.views[0]).mask else v)
    idx, _ = views_to_indexed_uops((view,))
    assert idx.vmax < 1e12, f"real_size broken for {self}"
    return int(idx.vmax + 1)

  def vars(self) -> set[Variable]: return set().union(*[v.vars() for v in self.views])

  @property
  def var_vals(self) -> dict[Variable, int]: return merge_dicts([dict([v.unbind()]) for v in self.vars()])

  def unbind(self) -> tuple[ShapeTracker, dict[Variable, int]]:
    unbound_views, var_vals = zip(*[v.unbind() for v in self.views])
    if all(len(x) == 0 for x in var_vals): return self, {}
    return ShapeTracker(tuple(unbound_views)), merge_dicts(var_vals)
  def substitute(self, dvars:dict[UOp, UOp]): return ShapeTracker(tuple(x.substitute(dvars) for x in self.views))

  def real_strides(self, ignore_valid=False) -> tuple[Optional[sint], ...]:
    with Context(TRACK_MATCH_STATS=0): return views_to_real_strides(self.views, ignore_valid)
  def unit_stride_axes(self, ignore_valid=False) -> list[int]: return [i for i,st in enumerate(self.real_strides(ignore_valid)) if st == 1]

  def axis_is_masked(self, axis:int) -> bool:
    with Context(TRACK_MATCH_STATS=0):
      _, valid = self.to_indexed_uops()
      return axis in [x.arg for x in graph_rewrite(valid, symbolic_flat).toposort() if x.op is Ops.RANGE]

  def simplify(self) -> ShapeTracker:
    if len(self.views) >= 2 and (new_view := self.views[-2] + self.views[-1]) is not None:
      return ShapeTracker(self.views[:-2] + (new_view,)).simplify()
    return self

  # *** under this line are the movement ops ***

  def pad(self, arg: tuple[tuple[sint, sint], ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].pad(arg), ))
  def shrink(self, arg: tuple[tuple[sint, sint], ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].shrink(arg), ))
  def expand(self, new_shape: tuple[sint, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].expand(new_shape), ))
  def permute(self, axis: tuple[int, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].permute(axis), ))
  def flip(self, mul: tuple[int, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].flip(mul), ))

  def reshape(self, new_shape: tuple[sint, ...]) -> ShapeTracker:
    if getenv("MERGE_VIEW", 1) and (new_view := self.views[-1].reshape(new_shape)) is not None: return ShapeTracker(self.views[0:-1] + (new_view,))
    return ShapeTracker(self.views + (View.create(new_shape), ))

  def mop(self, op, arg): return mops[op](self, arg)

mops: dict[Ops, Callable] = {Ops.RESHAPE: ShapeTracker.reshape, Ops.PERMUTE: ShapeTracker.permute, Ops.EXPAND: ShapeTracker.expand,
                             Ops.SHRINK: ShapeTracker.shrink, Ops.FLIP: ShapeTracker.flip, Ops.PAD: ShapeTracker.pad}
