# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
import functools
from typing import Tuple, Union, List, Optional, cast
from tinygrad.helpers import prod, DEBUG
from tinygrad.shape.symbolic import Variable, MulNode, NumNode, Node, SumNode, sint
from tinygrad.shape.view import View

@functools.lru_cache(maxsize=None)
def merge_views(vm2:View, vm1:View) -> Optional[View]:
  if vm2.mask: return None  # this isn't supported yet
  mst = ShapeTracker(vm1.shape, [vm2, vm1])
  strides = mst.real_strides()
  if None in strides: return None
  return View.create(vm1.shape, cast(Tuple[sint, ...], strides), mst.real_offset(), vm1.mask)

@functools.lru_cache(maxsize=None)
def idxs_to_idx(shape:Tuple[int, ...], idxs) -> Node:
  assert len(idxs) == len(shape), "need an idx for all dimensions"
  acc = 1
  ret = []
  for tidx,d in reversed(list(zip(idxs, shape))):
    ret.append(tidx * acc)
    acc *= d
  return Variable.sum(ret)

class ShapeTracker:
  __slots__ = "views"
  def __init__(self, shape:Union[ShapeTracker, Tuple[sint, ...]], views:Optional[List[View]]=None):
    self.views: List[View] = views if views is not None else [*shape.views] if isinstance(shape, ShapeTracker) else [View.create(shape)]
  def __repr__(self): return f"ShapeTracker(shape={self.views[-1].shape}, views={self.views})"
  def copy(self) -> ShapeTracker: return ShapeTracker(self.views[-1].shape, [*self.views])

  @property
  def contiguous(self) -> bool: return len(self.views) == 1 and self.views[0].contiguous

  @property
  def shape(self) -> Tuple[sint, ...]: return self.views[-1].shape

  @property
  def key(self) -> Tuple[View, ...]: return tuple(self.views)

  # this is the real size (ish)
  def size(self): return self.views[-1].size()

  # these are multiview strides, value is None if it's not a simple strided dimension
  # TODO: this can be shared code between simplify and merge_views
  def real_offset(self) -> int:
    real_offset, mask = self.expr_node(Variable('zero', 0, 0))
    assert real_offset.__class__ is NumNode, f"how is the offset not a number? {real_offset} {mask}"
    return real_offset.b

  # NOTE: if a stride is not always valid, it will be None
  def real_strides(self, ignore_valid=False) -> Tuple[Optional[sint], ...]:
    if len(self.views) == 1 and self.views[-1].mask is None: return self.views[-1].strides
    idxs = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(self.shape)]
    idx, valid = self.expr_idxs(idxs)
    ret: List[Optional[sint]] = [None] * len(self.views[-1].shape)
    for this_dim in (idx.nodes if isinstance(idx, SumNode) else [idx]):
      if isinstance(this_dim, MulNode) and isinstance(this_dim.a, Variable) and this_dim.a in idxs:
        ret[idxs.index(this_dim.a)] = this_dim.b
      elif isinstance(this_dim, Variable):
        ret[idxs.index(this_dim)] = 1
    idx_vars, valid_vars = idx.vars(), valid.vars()
    for i,tidx in enumerate(idxs):
      if tidx in valid_vars and not ignore_valid: ret[i] = None
      elif tidx not in idx_vars: ret[i] = 0
    return tuple(ret)
  def unit_stride_axes(self, ignore_valid=False) -> List[int]: return [i for i,st in enumerate(self.real_strides(ignore_valid)) if st == 1]

  def _expr_idx(self, idx, valid) -> Tuple[Node, Node]:
    for v in reversed(self.views[0:-1]):
      if valid.max == 0: return Variable.num(-1), valid
      valid = v.expr_node_mask(idx, valid)
      idx = v.expr_node(idx)
    return idx, valid

  def simplify(self):
    if len(self.views) >= 2:
      new_view = merge_views(self.views[-2], self.views[-1])
      if new_view:
        if DEBUG >= 4: print(f"st simplify : {self.views[-2]} + {self.views[-1]} = {new_view}")
        self.views = self.views[:-2] + [new_view]
        self.simplify()

  def expr_idxs(self, idxs=None):
    if idxs is None: idxs = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(self.shape)]
    idx = self.views[-1].expr_idxs(tuple(idxs))
    valid = self.views[-1].expr_node_mask(idxs_to_idx(self.views[-1].shape, tuple(idxs)))
    return self._expr_idx(idx, valid)

  def expr_node(self, idx='idx'):
    if idx.__class__ is str: idx = Variable(idx, 0, prod(self.shape)-1)
    return self._expr_idx(self.views[-1].expr_node(idx), self.views[-1].expr_node_mask(idx))

  def axis_is_masked(self, axis) -> bool:
    _, valid = self.expr_idxs()
    return f'idx{axis}' in [v.expr for v in valid.vars()]

  # *** under this line are the movement ops ***

  def pad(self, arg: Tuple[Tuple[int, int], ...]):
    self.views[-1] = self.views[-1].pad(arg)
    return self

  def shrink(self, arg: Tuple[Tuple[int, int], ...]):
    self.views[-1] = self.views[-1].shrink(arg)
    return self

  def expand(self, new_shape: Tuple[sint, ...]):
    self.views[-1] = self.views[-1].expand(new_shape)
    return self

  def permute(self, axis: Tuple[int, ...]):
    self.views[-1] = self.views[-1].permute(axis)
    return self

  def stride(self, mul: Tuple[int, ...]):
    self.views[-1] = self.views[-1].stride(mul)
    return self

  def reshape(self, new_shape: Tuple[sint, ...]):
    new_view = self.views[-1].reshape(new_shape)
    if new_view is None:
      extra_view = View.create(new_shape)
      # last chance to merge. TODO: move into View
      if (merged_view := merge_views(self.views[-1], extra_view)) is not None:
        self.views[-1] = merged_view
      else:
        self.views.append(extra_view)
    else:
      self.views[-1] = new_view
    return self

# returns the axes to create new_shape if new_shape can be created by combining axis from old_shape
# TODO: if we remove movementops from lazy.py we can delete this
def get_contraction(old_shape:Tuple[sint, ...], new_shape:Tuple[sint, ...]) -> Optional[List[List[int]]]:
  # Pre-allocate all groups.
  axis_groups: List[List[int]] = [[] for _ in range(len(new_shape))]
  # Index for new_shape and axis_groups.
  i: int = 0
  old_shape_i: int = 0
  while old_shape_i < len(old_shape):
    # 1s exist in new_shape only will lead to empty axes group creations.
    if new_shape[i] == 1 and old_shape[old_shape_i] != 1:
      if i < len(new_shape) - 1: i += 1
    else:
      axis_groups[i].append(old_shape_i)
      axis_group_size = prod([old_shape[x] for x in axis_groups[i]])
      # Move to next axes group if total size of all dimensions match.
      if axis_group_size == new_shape[i]:
        if i < len(new_shape) - 1: i += 1
      elif axis_group_size > new_shape[i]: return None
      old_shape_i += 1
  return axis_groups
