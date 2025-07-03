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
def views_to_movement_ops(views: tuple["View", ...]) -> list[tuple[Ops, Any]]:
  # Note: Don't early return for shapes with 0 - let the algorithm handle it
  # This maintains compatibility with previous behavior for torch backend
  ops: list[tuple[Ops, Any]] = []

  for i, view in enumerate(views):
    # resolve the eff shape for this view
    shape = tuple(b - a for a, b in view.mask) if view.mask else view.shape

    # compute the starting position within the underlying buffer
    offset = view.offset + sum(strd * (dim - 1) for dim, strd in zip(shape, view.strides) if strd < 0)
    pos = offset + (sum(beg * strd for (beg, _), strd in zip(view.mask, view.strides)) if view.mask else 0)

    # collect (dim, stride) pairs for non-zero strides
    strides = [(dim, abs(strd) if isinstance(strd, int) else strd) for dim, strd in zip(shape, view.strides) if strd]
    buffer = sum((d - 1) * s for d, s in strides) + 1 if strides else 1
    if i: buffer = (prod(views[i - 1].shape) - pos) if strides else 1

    # initial reshape + shrink to isolate the relevant window
    ops.extend([(Ops.RESHAPE, (-1,)),
                (Ops.SHRINK, ((pos, pos + buffer),))])

    if strides:
      order, pairs = zip(*sorted(enumerate(strides), key=lambda p: (p[1][1], -p[1][0]), reverse=True))
      order, pairs = list(order), list(pairs)
      extra = pairs[0][0] * pairs[0][1] - buffer
      if extra > 0: ops.append((Ops.PAD, ((0, extra),)))

      for j, (dim, stride) in enumerate(pairs):
        if j < len(pairs) - 1 and stride < pairs[j + 1][0] * pairs[j + 1][1]:
          remaining = pairs[j - 1][1] if j else buffer
          ops.extend([(Ops.EXPAND, (dim, *(p[0] for p in pairs[:j]), remaining)),
                      (Ops.PERMUTE, (*range(1, j + 1), 0, j + 1)),
                      (Ops.RESHAPE, (*(p[0] for p in pairs[:j]), dim * remaining)),
                      (Ops.PAD, (*((0, 0) for _ in range(j)), (0, dim * stride))),
                      (Ops.RESHAPE, (*(p[0] for p in pairs[:j + 1]), remaining + stride)),])
          pairs[j] = (dim, remaining + stride)
        else: ops.extend([(Ops.SHRINK, (*((0, p[0]) for p in pairs[:j]), (0, dim * stride))),
                          (Ops.RESHAPE, (*[p[0] for p in pairs[:j + 1]], stride)),])

      ops.extend([(Ops.SHRINK, (*[(0, p[0]) for p in pairs], (0, 1))),
                  (Ops.RESHAPE, tuple(p[0] for p in pairs)),])

      # restore original axis order if needed
      if order != list(range(len(order))): ops.append((Ops.PERMUTE, tuple(order.index(k) for k in range(len(order)))))

    # final reshape to the intended shape
    ops.append((Ops.RESHAPE, tuple(dim if strd else 1 for dim, strd in zip(shape, view.strides))))
    # handle negative strides via flip
    if any(strd < 0 for strd in view.strides): ops.append((Ops.FLIP, tuple(-1 if strd < 0 else 1 for strd in view.strides)))

    # mask-related padding
    if view.mask is not None:
      pre_pad = tuple((beg, dim - end) if strd != 0 else (0, 0) for (beg, end), dim, strd in zip(view.mask, view.shape, view.strides))
      post_pad = tuple((beg, dim - end) if strd == 0 else (0, 0) for (beg, end), dim, strd in zip(view.mask, view.shape, view.strides))
      if any(p != (0, 0) for p in pre_pad):
        ops.append((Ops.PAD, pre_pad))
        shape = tuple(dim + l + r for dim, (l, r) in zip(shape, pre_pad))
    else: post_pad = ()

    # expand axes and pad
    if any(dim != 1 and strd == 0 for dim, strd in zip(shape, view.strides)): ops.append((Ops.EXPAND, shape))
    if view.mask is not None and any(p != (0, 0) for p in post_pad): ops.append((Ops.PAD, post_pad))
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
  def to_movement_ops(self) -> list[tuple[Ops, Any]]: return views_to_movement_ops(self.views)
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
