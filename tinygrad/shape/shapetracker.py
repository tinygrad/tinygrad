# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
import functools
from typing import Tuple, Union, List, Optional, NamedTuple
from tinygrad.helpers import prod, DEBUG
from tinygrad.shape.symbolic import Variable, MulNode, NumNode, Node, SumNode, is_sym_int

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

@functools.lru_cache(maxsize=None)
def is_contiguous(shape:Tuple[int, ...], strides:Tuple[int, ...]) -> bool: return all(s1 == s2 or s == 1 for s,s1,s2 in zip(shape, strides, strides_for_shape(shape)))

@functools.lru_cache(maxsize=None)
def filter_strides(shape:Tuple[int, ...], strides:Tuple[int, ...]) -> Tuple[int, ...]:
  return tuple(stride if shp != 1 else 0 for stride, shp in zip(strides, shape))

class View(NamedTuple):
  shape:Tuple[int, ...]
  strides:Tuple[int, ...]
  offset:int = 0
  mask:Optional[Tuple[Tuple[int, int]]] = None

  @staticmethod
  @functools.lru_cache(None)
  def create(shape, strides=None, offset=0, mask=None): return View(shape, filter_strides(shape, strides) if strides else strides_for_shape(shape), offset, mask)

  def __repr__(self): return f"View(shape={self.shape}, strides={self.strides}, offset={self.offset}, mask={self.mask})"

  @property
  def contiguous(self): return self.offset == 0 and is_contiguous(self.shape, self.strides) and self.mask is None

  @property
  def shape_strides(self): return to_shape_strides(self.shape, self.strides)
  
  def expr_node_mask(self, idx, valid=None) -> Node: return expr_node_mask(self.shape, self.mask, idx, valid)
  def expr_node(self, idx=None) -> Node: return expr_node_view(self.shape, self.shape_strides, self.offset, idx)
  def expr_idxs(self, idxs) -> Node: return expr_idxs_view(self.shape, self.strides, self.offset, idxs)

@functools.lru_cache(maxsize=None)
def expr_node_mask(shape, mask, idx, valid) -> Node:
  expr = [valid] if valid is not None else []
  if mask is not None:
    acc = 1
    for ns,(x,y) in reversed(list(zip(shape, mask))):
      if x != 0 or y != ns:
        base = ((idx//acc) % ns)
        expr += [base >= x, base < y]
      acc *= ns
  return Variable.ands(expr)

# generate an expression if you have a single idx variable
@functools.lru_cache(maxsize=None)
def expr_node_view(shape, shape_strides, offset, idx) -> Node:
  if idx is None: idx = Variable('idx', 0, prod(shape)-1)
  ret: List[Node] = [Variable.num(offset) if isinstance(offset, int) else offset] if offset else []
  acc = 1
  for d,s in reversed(shape_strides):
    ret.append(((idx//acc)%d)*s)
    acc *= d
  return Variable.sum(ret)

# generate an expression if you have a variable or expression for each index
@functools.lru_cache(maxsize=None)
def expr_idxs_view(shape, strides, offset, idxs) -> Node:
  assert len(idxs) == len(shape), f"need an idx for all dimensions {idxs} vs {shape}"
  return Variable.sum([Variable.num(offset) if isinstance(offset, int) else offset] + [idx*st for idx,sh,st in zip(idxs, shape, strides) if sh != 1 and st != 0])

@functools.lru_cache(maxsize=None)
def size(view: View): return prod([s for s,st in zip(view.shape, view.strides) if st != 0])

@functools.lru_cache(maxsize=None)
def idxs_to_idx(shape:Tuple[int, ...], idxs) -> Node:
  assert len(idxs) == len(shape), "need an idx for all dimensions"
  acc = 1
  ret = []
  for tidx,d in reversed(list(zip(idxs, shape))):
    ret.append(tidx * acc)
    acc *= d
  return Variable.sum(ret)

@functools.lru_cache(maxsize=None)
def strides_for_shape(shape:Tuple[int, ...]) -> Tuple[int, ...]:
  strides = [1] if shape else []
  for d in shape[::-1][:-1]: strides = [d*strides[0]] + strides
  return filter_strides(shape, tuple(strides))

def merge_views(vm2:View, vm1:View) -> Optional[View]:
  if vm2.mask: return None  # this isn't supported yet
  strides = real_strides((vm2,vm1))
  if None in strides: return None
  return View.create(vm1.shape, strides, real_offset((vm2, vm1)), vm1.mask)

@functools.lru_cache(maxsize=None)
def get_pad_args(shape:Tuple[int,...], arg:Tuple[Tuple[int, int], ...]):
  return tuple([(-b,s+e) for s,(b,e) in zip(shape, arg)]), tuple([(b,s+b) for s,(b,_) in zip(shape, arg)])

@functools.lru_cache(maxsize=None)
def get_unsafe_resize_offset(strides, arg):
  return sum([s * x[0] for s, x in zip(strides,arg)])

class ShapeTracker:
  __slots__ = "views"
  def __init__(self, shape:Union[ShapeTracker, Tuple[Union[Node,int], ...]], views:Optional[Union[Tuple[View,...],List[View]]]=None):
    self.views: Tuple[View, ...] = tuple(views) if views is not None else tuple([*shape.views]) if isinstance(shape, ShapeTracker) else (View.create(shape),)
  def __repr__(self): return f"ShapeTracker(shape={self.views[-1].shape}, views={self.views})"
  def copy(self) -> ShapeTracker: return ShapeTracker(self.views[-1].shape, tuple([*self.views]))

  def reshape(self, new_shape): 
    if self.views[-1].shape == new_shape: return self
    self.views = self.views[:-1] + reshape(self.views[-1], new_shape)
    return self
  def expand(self, new_shape):
    self.views = self.views[:-1] + (expand(self.views[-1], new_shape),)
    return self
  def shrink(self, arg):
    self.views = self.views[:-1] + (shrink(self.views[-1], arg),)
    return self
  def stride(self, mul):
    self.views = self.views[:-1] + (stride(self.views[-1], mul),)
    return self
  def permute(self, axis):
    self.views = self.views[:-1] + (permute(self.views[-1], axis),)
    return self
  def pad(self, arg):
    self.views = self.views[:-1] + (pad(self.views[-1], arg),)
    return self
  
  def size(self): return size(self.views[-1])  # this is the real size (ish)
  def simplify(self): self.views = simplify(self.views)
  def unit_stride_axes(self, ignore_valid=False) -> List[int]: return unit_stride_axes(self.views, ignore_valid)
  def expr_idxs(self, idxs=None): return expr_idxs(self.views, None if idxs is None else tuple(idxs))
  def real_strides(self, ignore_valid=False) -> Tuple[Optional[Union[Node, int]], ...]: return real_strides(self.views, ignore_valid)
  def axis_is_masked(self, axis) -> bool: return axis_is_masked(self.views, axis)
  def expr_node(self, idx='idx'): return expr_node(self.views, idx)

  @property
  def contiguous(self) -> bool: return len(self.views) == 1 and self.views[0].contiguous

  @property
  def shape(self) -> Tuple[int, ...]: return self.views[-1].shape # NOTE: real type is Tuple[Union[Node, int], ...] but mypy complains about prod(shape)

  @property
  def key(self) -> Tuple[View, ...]: return self.views

# these are multiview strides, value is None if it's not a simple strided dimension
# TODO: this can be shared code between simplify and merge_views
@functools.lru_cache(maxsize=None)
def real_offset(views) -> int:
  real_offset, mask = expr_node(views, Variable('zero', 0, 0))
  assert real_offset.__class__ is NumNode, f"how is the offset not a number? {real_offset} {mask}"
  return real_offset.b

# NOTE: if a stride is not always valid, it will be None
@functools.lru_cache(maxsize=None)
def real_strides(views: Tuple[View], ignore_valid=False) -> Tuple[Optional[Union[Node, int]], ...]:
  if len(views) == 1 and views[-1].mask is None: return views[-1].strides
  idxs = tuple([Variable(f"idx{i}", 0, s-1) for i,s in enumerate(views[-1].shape)])
  idx, valid = expr_idxs(views, idxs)
  ret: List[Optional[Union[Node, int]]] = [None] * len(views[-1].shape)
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

@functools.lru_cache(maxsize=None)
def unit_stride_axes(views: Tuple[View], ignore_valid=False) -> List[int]: return [i for i,st in enumerate(real_strides(views, ignore_valid)) if st == 1]

@functools.lru_cache(maxsize=None)
def _expr_idx(views: Tuple[View], idx, valid) -> Tuple[Node, Node]:
  for v in reversed(views[:-1]):
    if valid.max == 0: return Variable.num(-1), valid
    valid = v.expr_node_mask(idx, valid)
    idx = v.expr_node(idx)
  return idx, valid

@functools.lru_cache(maxsize=None)
def simplify(views: Tuple[View]):
  if len(views) >= 2:
    new_view = merge_views(views[-2], views[-1])
    if new_view:
      if DEBUG >= 4: print(f"st simplify : {views[-2]} + {views[-1]} = {new_view}")
      views = views[:-2] + (new_view,)
      if len(views) == 1: return views
      views = simplify(views)
  return views

@functools.lru_cache(maxsize=None)
def expr_idxs(views: Tuple[View], idxs=None):
  if idxs is None: idxs = tuple([Variable(f"idx{i}", 0, s-1) for i,s in enumerate(views[-1].shape)])
  idx = views[-1].expr_idxs(idxs)
  valid = views[-1].expr_node_mask(idxs_to_idx(views[-1].shape, idxs))
  return _expr_idx(views, idx, valid)

@functools.lru_cache(maxsize=None)
def expr_node(views: Tuple[View], idx='idx'):
  if idx.__class__ is str: idx = Variable(idx, 0, prod(views[-1].shape)-1)
  return _expr_idx(views, views[-1].expr_node(idx), views[-1].expr_node_mask(idx))

@functools.lru_cache(maxsize=None)
def axis_is_masked(views: Tuple[View], axis) -> bool: 
  _, valid = expr_idxs(views)
  return f'idx{axis}' in [v.expr for v in valid.vars()]
  # *** under this line are the movement ops ***

@functools.lru_cache(maxsize=None)
def __unsafe_resize(view: View, arg: Tuple[Tuple[int, int], ...], mask=None) -> View:
  offset = get_unsafe_resize_offset(view.strides, arg)
  if view.mask:
    # move the old mask
    nmask = tuple([(max(mx-ax, 0), min(my-ax, ay-ax)) for (mx,my),(ax,ay) in zip(view.mask, arg)])
    # merge the masks if we have two
    mask = tuple([(max(mx1, mx2), min(my1, my2)) for (mx1, my1), (mx2, my2) in zip(nmask, mask)]) if mask is not None else nmask
  return View.create(tuple([y-x for x,y in arg]), view.strides, view.offset+offset, mask)

@functools.lru_cache(maxsize=None)
def pad(view: View, arg: Tuple[Tuple[int, int], ...]) -> View:
  assert all((b>=0 and e>=0) for b,e in arg) and len(arg) == len(view.shape)
  if any(b or e for b, e in arg):
    zvarg, mask = get_pad_args(view.shape, arg)
    view = __unsafe_resize(view, zvarg, mask=mask)
  return view

@functools.lru_cache(maxsize=None)
def shrink(view: View, arg: Tuple[Tuple[int, int], ...]) -> View:
  assert all((b>=0 and e<=s) for s,(b,e) in zip(view.shape,arg)) and len(arg) == len(view.shape)
  return __unsafe_resize(view, arg)

@functools.lru_cache(maxsize=None)
def expand(view: View, new_shape: Tuple[Union[Node,int], ...]) -> View:
  assert len(new_shape) == len(view.shape)
  assert all(is_sym_int(x) and (s == x or (s == 1 and st == 0)) for s,x,st in zip(view.shape, new_shape, view.strides)), f"can't expand {view.shape} into {new_shape}"
  # NOTE: can the mask ever be (0,0)?
  mask = tuple([(((0,0) if m != (0,1) else (0,ns)) if s != ns else m) for m,s,ns in zip(view.mask, view.shape, new_shape)]) if view.mask else None
  return View.create(new_shape, view.strides, view.offset, mask)

@functools.lru_cache(maxsize=None)
def reshape(view: View, new_shape: Tuple[Union[Node,int], ...]):
  if view.shape == new_shape: return (view,)
  assert all(is_sym_int(x) and x > 0 for x in new_shape), f"shape must be symbolic ints and can't contain 0 or negative numbers {new_shape}"
  # only check size for int shapes. we don't check symbolic here as long as the reshape itself can be done
  assert not (all(isinstance(s, int) for s in view.shape) and all(isinstance(s, int) for s in new_shape)) or prod(view.shape) == prod(new_shape), f"can't reshape {view.shape} -> {new_shape}" # type: ignore  # mypy cannot resolve, all ints here
  shape, mask, strides, offset = view.shape, view.mask, view.strides, view.offset
  # check if this is adding or removing 1s (only)
  # NOTE: this is optional, but removes most calls to (expensive!) merge_views (with mask, not optional)
  if [x for x in shape if x != 1] == [x for x in new_shape if x != 1]:
    new_strides: List[int] = [y for x,y in zip(shape, strides) if x != 1]
    new_strides_tuple: Tuple[int, ...] = tuple([0 if x == 1 else new_strides.pop(0) for x in new_shape])
    new_mask_tuple = None
    if mask:
      for x,y in zip(shape, mask):
        if x == 1 and y != (0, 1):
          new_mask_tuple = ((0,0),) * len(new_shape)
          break
      else:
        new_mask: List[Tuple[int, int]] = [y for x,y in zip(shape, mask) if x != 1]
        new_mask_tuple = tuple([(0,1) if x == 1 else new_mask.pop(0) for x in new_shape])
    return (View.create(new_shape, new_strides_tuple, offset, new_mask_tuple),)
  new_view = View.create(new_shape)
  if view.contiguous: return (new_view,) # NOTE: if it's contiguous it can't have an offset
  if (merged_view:=merge_views(view, new_view)) is not None: return (merged_view,)
  if DEBUG >= 5: print(f"WARNING: creating new view with reshape {view} -> {new_shape}")
  return (view, new_view)


@functools.lru_cache(maxsize=None)
def permute(view: View, axis: Tuple[int, ...]) -> View:
  assert all(isinstance(x, int) and x >= 0 and x < len(view.shape) for x in axis), f"invalid permute {axis} for {view.shape}"
  assert len(set(axis)) == len(axis) and len(axis) == len(view.shape), f"can't permute {view.shape} with {axis}"
  return View.create(tuple([view.shape[a] for a in axis]), tuple([view.strides[a] for a in axis]), view.offset, tuple([view.mask[a] for a in axis]) if view.mask is not None else None)

# except for the negative case, you can build this from the others. invertible in the negative case
@functools.lru_cache(maxsize=None)
def stride(view: View, mul: Tuple[int, ...]):
  assert all(isinstance(x, int) and x != 0 for x in mul), f"invalid stride {mul} for {view.shape}"
  strides = tuple([z*m for z,m in zip(view.strides, mul)])
  new_shape = tuple([(s+(abs(m)-1))//abs(m) for s,m in zip(view.shape, mul)])
  offset = sum([(s-1)*z for s,z,m in zip(view.shape, view.strides, mul) if m < 0])
  mask = tuple([(((mx if m > 0 else s-my)+(abs(m)-1))//abs(m), ((my if m > 0 else s-mx)+(abs(m)-1))//abs(m)) for (mx,my),s,m in zip(view.mask, view.shape, mul)]) if view.mask is not None else None
  return View.create(new_shape, strides, view.offset + offset, mask)
  
# returns the axes to create new_shape if new_shape can be created by combining axis from old_shape
def get_contraction(old_shape:Tuple[int, ...], new_shape:Tuple[int, ...]) -> Optional[List[List[int]]]:
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
