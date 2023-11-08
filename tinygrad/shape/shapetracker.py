# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
import functools, operator
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, cast
from tinygrad.ops import MovementOps
from tinygrad.helpers import prod, DEBUG, dedup
from tinygrad.shape.symbolic import Variable, MulNode, NumNode, Node, SumNode, sint
from tinygrad.shape.view import View

@functools.lru_cache(maxsize=None)
def to_shape_strides(shape:Tuple[int, ...], strides:Tuple[int, ...]) -> Tuple[Tuple[int, int], ...]:
  assert len(shape) == len(strides)
  ret = [(shape[0], strides[0])] if shape else []
  for i in range(1, len(shape)):
    if ret[-1][1] == shape[i]*strides[i] or ret[-1][0] == 1:
      ret[-1] = (ret[-1][0] * shape[i], strides[i])
    elif shape[i] == 1:
      continue
    else:
      ret.append((shape[i], strides[i]))
  return tuple(ret)

def expr_node_mask(view:View, idx, valid=None) -> Node:
  expr = [valid] if valid is not None else []
  if view.mask is not None:
    acc = 1
    for ns,(x,y) in reversed(list(zip(view.shape, view.mask))):
      if x != 0 or y != ns:
        base = ((idx//acc) % ns)
        expr += [base >= x, base < y]
      acc *= ns
  return Variable.ands(expr)

# generate an expression if you have a single idx variable
def expr_node(view:View, idx=None) -> Node:
  if idx is None: idx = Variable('idx', 0, prod(view.shape)-1)
  ret: List[Node] = [Variable.num(view.offset) if isinstance(view.offset, int) else view.offset] if view.offset else []
  acc = 1
  for d,s in reversed(to_shape_strides(view.shape, view.strides)):
    ret.append(((idx//acc)%d)*s)
    acc *= d
  return Variable.sum(ret)

# generate an expression if you have a variable or expression for each index
def expr_idxs(view:View, idxs) -> Node:
  assert len(idxs) == len(view.shape), f"need an idx for all dimensions {idxs} vs {view.shape}"
  return Variable.sum([Variable.num(view.offset) if isinstance(view.offset, int) else view.offset] + [idx*st for idx,sh,st in zip(idxs, view.shape, view.strides) if sh != 1 and st != 0])

@functools.lru_cache(maxsize=None)
def merge_views(vm2:View, vm1:View) -> Optional[View]:
  if vm2.mask: return None  # this isn't supported yet
  mst = ShapeTracker((vm2, vm1))
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

@dataclass(frozen=True)
class ShapeTracker:
  views: Tuple[View, ...]
  def __post_init__(self): assert isinstance(self.views, tuple) and all(isinstance(v, View) for v in self.views), "ShapeTracker must be created with a tuple of Views"

  @staticmethod
  def from_shape(shape:Tuple[sint, ...]): return ShapeTracker((View.create(shape),))

  @property
  def contiguous(self) -> bool: return len(self.views) == 1 and self.views[0].contiguous

  @property
  def shape(self) -> Tuple[sint, ...]: return self.views[-1].shape

  # this is the real size (ish)
  def size(self): return self.views[-1].size()

  def vars(self) -> List[Variable]: return dedup(functools.reduce(operator.add, [v.vars() for v in self.views], []))

  @property
  def var_vals(self) -> Dict[Variable, int]:
    ret:Dict[Variable, int] = {}
    for v in self.vars():
      var, val = v.unbind()
      assert var not in ret or ret[var] == val, f"{var} has conflicted values {val} and {ret[var]}"
      ret[var] = val
    return ret

  def unbind(self) -> ShapeTracker: return ShapeTracker(tuple(v.unbind() for v in self.views))

  def to_movement_ops(self) -> List[Tuple[MovementOps, Tuple]]:
    to_apply:List[Tuple[MovementOps, Tuple]] = []
    for v in self.views:
      real_shape = tuple(y-x for x,y in v.mask) if v.mask else v.shape
      offset = v.offset + sum(v.strides[i] * (real_shape[i]-1) for i in range(len(v.strides)) if v.strides[i]<0)
      real_offset = offset + (sum(x*st for (x,_),st in zip(v.mask, v.strides)) if v.mask else 0)
      real_real_shape = [s for s,st in zip(real_shape, v.strides) if st]
      strides = [abs(st) for st in v.strides if st and not isinstance(st, Node)]
      buffer_size = sum((s-1)*st for s, st in zip(real_real_shape,strides)) + 1
      def sort_by_strides(shape, strides): return sorted(zip(shape, strides), key=lambda k: k[1], reverse=True), sorted(range(len(strides)), key=lambda k: strides[k], reverse=True)
      ordered_shape_strides, order = sort_by_strides(real_real_shape, strides)
      to_apply.append((MovementOps.RESHAPE, (-1,)))
      to_apply.append((MovementOps.SHRINK, ((real_offset,real_offset+buffer_size),)))
      if strides:
        if (ordered_shape_strides[0][0] * ordered_shape_strides[0][1]) - buffer_size > 0:
          to_apply.append((MovementOps.PAD, ((0, (ordered_shape_strides[0][0] * ordered_shape_strides[0][1]) - buffer_size),)))
        for i, shape_stride in enumerate(ordered_shape_strides):
          total_size = shape_stride[0] * shape_stride[1]
          if i<len(ordered_shape_strides)-1 and shape_stride[1] < ordered_shape_strides[i+1][0]*ordered_shape_strides[i+1][1]:
            remaining_buffer = ordered_shape_strides[i-1][1] if i>0 else buffer_size
            to_apply.append((MovementOps.EXPAND, (shape_stride[0], *(s[0] for s in ordered_shape_strides[:i]), remaining_buffer)))
            to_apply.append((MovementOps.PERMUTE, (*range(1,i+1), 0, i+1)))
            to_apply.append((MovementOps.RESHAPE, (*(s[0] for s in ordered_shape_strides[:i]), shape_stride[0]*remaining_buffer)))
            to_apply.append((MovementOps.PAD, (*((0,0) for _ in range(i)), (0, shape_stride[0]*shape_stride[1]))))
            to_apply.append((MovementOps.RESHAPE, (*(s[0] for s in ordered_shape_strides[:i+1]), remaining_buffer+shape_stride[1])))
          else:
            to_apply.append((MovementOps.SHRINK, (*((0, s[0]) for s in ordered_shape_strides[:i]), (0,total_size))))
            to_apply.append((MovementOps.RESHAPE, (*[s[0] for s in ordered_shape_strides[:i+1]], shape_stride[1])))
        to_apply.append((MovementOps.SHRINK, (*[(0, s[0]) for s in ordered_shape_strides], (0,1))))
        to_apply.append((MovementOps.RESHAPE, tuple(s[0] for s in ordered_shape_strides)))
        if order != list(range(len(order))):
          to_apply.append((MovementOps.PERMUTE, tuple(order.index(i) for i in range(len(strides)))))
      to_apply.append((MovementOps.RESHAPE, tuple(s if st else 1 for s,st in zip(real_shape, v.strides))))
      # flip
      if any(i < 0 for i in v.strides): to_apply.append((MovementOps.STRIDE, tuple(-1 if st<0 else 1 for st in v.strides)))
      # then, we apply pre expand pads
      if v.mask is not None:
        pre_expand_pads = tuple((x,s-y) if st != 0 else (0,0) for (x,y),s,st in zip(v.mask, v.shape, v.strides))
        post_expand_pads = tuple((x,s-y) if st == 0 else (0,0) for (x,y),s,st in zip(v.mask, v.shape, v.strides))
        if any(x != (0,0) for x in pre_expand_pads):
          to_apply.append((MovementOps.PAD, pre_expand_pads))
          real_shape = tuple(x+s[0]+s[1] for x,s in zip(real_shape, pre_expand_pads))
      # then, we do any expands
      if any(s != 1 and st == 0 for s,st in zip(real_shape, v.strides)): to_apply.append((MovementOps.EXPAND, real_shape))
      # lastly, we apply post expand pads
      if v.mask is not None and any(x != (0,0) for x in post_expand_pads): to_apply.append((MovementOps.PAD, post_expand_pads))
    return to_apply

  # these are multiview strides, value is None if it's not a simple strided dimension
  # TODO: this can be shared code between simplify and merge_views
  def real_offset(self) -> sint:
    real_offset, _ = self.expr_node(Variable('zero', 0, 0))
    return real_offset.b if isinstance(real_offset, NumNode) else real_offset

  # NOTE: if a stride is not always valid, it will be None
  def real_strides(self, ignore_valid=False) -> Tuple[Optional[sint], ...]:
    if len(self.views) == 1 and self.views[-1].mask is None: return self.views[-1].strides
    idxs = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(self.shape)]
    idx, valid = self.expr_idxs(idxs)
    ret: List[Optional[sint]] = [None] * len(self.views[-1].shape)
    for this_dim in (idx.nodes if isinstance(idx, SumNode) else [idx]):
      if isinstance(this_dim, MulNode) and isinstance(this_dim.a, Variable) and this_dim.a in idxs:
        ret[idxs.index(this_dim.a)] = this_dim.b
      elif isinstance(this_dim, Variable) and this_dim in idxs:
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
      valid = expr_node_mask(v, idx, valid)
      idx = expr_node(v, idx)
    return idx, valid

  def simplify(self) -> ShapeTracker:
    if len(self.views) >= 2:
      new_view = merge_views(self.views[-2], self.views[-1])
      if new_view:
        if DEBUG >= 4: print(f"st simplify : {self.views[-2]} + {self.views[-1]} = {new_view}")
        return ShapeTracker(self.views[:-2] + (new_view,)).simplify()
    return self

  def expr_idxs(self, idxs=None):
    if idxs is None: idxs = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(self.shape)]
    idx = expr_idxs(self.views[-1], tuple(idxs))
    valid = expr_node_mask(self.views[-1], idxs_to_idx(self.views[-1].shape, tuple(idxs)))
    return self._expr_idx(idx, valid)

  def expr_node(self, idx='idx'):
    if idx.__class__ is str: idx = Variable(idx, 0, prod(self.shape)-1)
    return self._expr_idx(expr_node(self.views[-1], idx), expr_node_mask(self.views[-1], idx))

  def axis_is_masked(self, axis) -> bool:
    _, valid = self.expr_idxs()
    return f'idx{axis}' in [v.expr for v in valid.vars()]

  # *** under this line are the movement ops ***

  def pad(self, arg: Tuple[Tuple[int, int], ...]) -> ShapeTracker:
    return ShapeTracker(self.views[0:-1] + (self.views[-1].pad(arg), ))

  def shrink(self, arg: Tuple[Tuple[sint, sint], ...]) -> ShapeTracker:
    return ShapeTracker(self.views[0:-1] + (self.views[-1].shrink(arg), ))

  def expand(self, new_shape: Tuple[sint, ...]) -> ShapeTracker:
    return ShapeTracker(self.views[0:-1] + (self.views[-1].expand(new_shape), ))

  def permute(self, axis: Tuple[int, ...]) -> ShapeTracker:
    return ShapeTracker(self.views[0:-1] + (self.views[-1].permute(axis), ))

  def stride(self, mul: Tuple[int, ...]) -> ShapeTracker:
    return ShapeTracker(self.views[0:-1] + (self.views[-1].stride(mul), ))

  def reshape(self, new_shape: Tuple[sint, ...]) -> ShapeTracker:
    new_view = self.views[-1].reshape(new_shape)
    if new_view is None:
      extra_view = View.create(new_shape)
      # last chance to merge. TODO: move into View
      if (merged_view := merge_views(self.views[-1], extra_view)) is not None:
        return ShapeTracker(self.views[0:-1] + (merged_view,))
      return ShapeTracker(self.views + (extra_view, ))
    return ShapeTracker(self.views[0:-1] + (new_view,))

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
