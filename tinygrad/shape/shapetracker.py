# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
from dataclasses import dataclass
import functools
from typing import Callable
from tinygrad.helpers import merge_dicts, getenv
from tinygrad.shape.view import View, unravel
from tinygrad.uop.symbolic import sym
from tinygrad.uop.ops import UOp, Ops, graph_rewrite, Variable, sint, sint_to_uop, Context

@functools.cache
def views_to_valid_uop(views: tuple[View, ...], _idxs:tuple[UOp, ...]|None=None) -> UOp:
  idx = views[-1].to_valid_uop(_idxs)
  for view in reversed(views[0:-1]):
    idx = view.to_valid_uop([sint_to_uop(i) for i in unravel(view.shape, idx)])
  with Context(TRACK_MATCH_STATS=0):
    return graph_rewrite(idx, sym, name="indexing sym @ 1")

@functools.cache
def views_to_is_expanded(views: tuple[View, ...]) -> tuple[bool, ...]:
  # NOTE: return if each dim is expanded
  if len(views) == 1 and views[-1].mask is None: return tuple([bool(st==0) for st in views[-1].strides])
  idx = views_to_valid_uop(views).get_idx()
  used_ranges = [x.arg[0] for x in idx.toposort() if x.op is Ops.RANGE]
  return tuple([i not in used_ranges for i in range(len(views[-1].shape))])

@dataclass(frozen=True, order=True)
class ShapeTracker:
  views: tuple[View, ...]

  def __add__(self, st:ShapeTracker) -> ShapeTracker:
    ret = self
    for v in st.views: ret = ShapeTracker(ret.views + (v,)).simplify() # one view at a time = better simplification
    return ret

  @staticmethod
  def from_shape(shape:tuple[sint, ...], strides:tuple[sint, ...]|None=None) -> ShapeTracker: return ShapeTracker((View.create(shape, strides),))

  @property
  def contiguous(self) -> bool: return len(self.views) == 1 and self.views[0].contiguous

  @property
  def shape(self) -> tuple[sint, ...]: return self.views[-1].shape

  @property
  def size(self) -> int: return self.views[-1].size()

  def vars(self) -> set[Variable]: return set().union(*[v.vars() for v in self.views])

  @property
  def var_vals(self) -> dict[str, int]: return merge_dicts([{(vu:=v.unbind())[0].expr:vu[1]} for v in self.vars()])

  def unbind(self) -> tuple[ShapeTracker, dict[Variable, int]]:
    unbound_views, var_vals = zip(*[v.unbind() for v in self.views])
    if all(len(x) == 0 for x in var_vals): return self, {}
    return ShapeTracker(tuple(unbound_views)), merge_dicts(var_vals)

  def is_expanded(self) -> tuple[bool, ...]:
    with Context(TRACK_MATCH_STATS=0): return views_to_is_expanded(self.views)

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
    if getenv("MERGE_VIEW", 0) and (new_view := self.views[-1].reshape(new_shape)) is not None: return ShapeTracker(self.views[0:-1] + (new_view,))
    return ShapeTracker(self.views + (View.create(new_shape), ))

  def mop(self, op, arg): return mops[op](self, arg)

mops: dict[Ops, Callable] = {Ops.RESHAPE: ShapeTracker.reshape, Ops.PERMUTE: ShapeTracker.permute, Ops.EXPAND: ShapeTracker.expand,
                             Ops.SHRINK: ShapeTracker.shrink, Ops.FLIP: ShapeTracker.flip, Ops.PAD: ShapeTracker.pad}
