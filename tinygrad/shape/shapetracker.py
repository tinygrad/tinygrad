# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
import functools
from typing import Dict, Tuple, Union, List, Optional, cast, NamedTuple
from tinygrad.helpers import prod, DEBUG, partition
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

class ViewInternal(NamedTuple):
  shape:Tuple[int, ...]
  strides:Tuple[int, ...]
  offset:int
  mask:Optional[Tuple[Tuple[int, int]]]
  contiguous:bool
  shape_strides:Tuple[Tuple[int, int], ...]

@functools.lru_cache(maxsize=None)
class View(ViewInternal):
  def __new__(cls, shape, strides=None, offset=0, mask=None):
    strides_from_shape = strides_for_shape(shape)
    strides = strides_from_shape if not strides else filter_strides(shape, strides)
    contiguous = offset == 0 and is_contiguous(shape, strides) and mask is None
    return super().__new__(cls, shape, strides, offset, mask, contiguous, to_shape_strides(shape, strides))
  def __init__(self, shape, strides=None, offset=0, mask=None, contiguous=False, shape_strides=()): super().__init__()

  def expr_node_mask(self, idx, valid=None) -> Node:
    expr = [valid] if valid is not None else []
    if self.mask is not None:
      acc = 1
      for ns,(x,y) in reversed(list(zip(self.shape, self.mask))):
        base = ((idx//acc) % ns)
        expr += [base >= x, base < y]
        acc *= ns
    return Variable.ands(expr)

  # generate an expression if you have a single idx variable
  def expr_node(self, idx=None) -> Node:
    if idx is None: idx = Variable('idx', 0, prod(self.shape)-1)
    ret: List[Node] = [Variable.num(self.offset) if isinstance(self.offset, int) else self.offset] if self.offset else []
    acc = 1
    for d,s in reversed(self.shape_strides):
      ret.append(((idx//acc)%d)*s)
      acc *= d
    return Variable.sum(ret)

  # generate an expression if you have a variable or expression for each index
  def expr_idxs(self, idxs) -> Node:
    assert len(idxs) == len(self.shape), f"need an idx for all dimensions {idxs} vs {self.shape}"
    return Variable.sum([Variable.num(self.offset) if isinstance(self.offset, int) else self.offset] + [idx*st for idx,sh,st in zip(idxs, self.shape, self.strides) if sh != 1 and st != 0])

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

@functools.lru_cache(maxsize=None)
def merge_views(vm2:View, vm1:View) -> Optional[View]:
  if vm2.mask: return None  # this isn't supported yet
  mst = ShapeTracker(vm1.shape, [vm2, vm1])
  strides = mst.real_strides()
  if None in strides: return None
  return View(vm1.shape, strides, mst.real_offset(), vm1.mask)

@functools.lru_cache(maxsize=None)
def _reshape(view: View, new_shape:Tuple[int, ...]) -> Tuple[View, bool]:
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
    return View(new_shape, new_strides_tuple, offset, new_mask_tuple), False

  new_view = View(new_shape)
  if view.contiguous: return new_view, False # NOTE: if it's contiguous it can't have an offset
  if (merged_view := merge_views(view, new_view)) is not None: return merged_view, False
  if DEBUG >= 4: print(f"WARNING: creating new view with reshape {view} -> {new_shape}")
  return new_view, True

@functools.lru_cache(maxsize=None)
def get_pad_args(shape:Tuple[int,...], arg:Tuple[Tuple[int, int], ...]):
  return tuple([(-b,s+e) for s,(b,e) in zip(shape, arg)]), tuple([(b,s+b) for s,(b,_) in zip(shape, arg)])

@functools.lru_cache(maxsize=None)
def get_unsafe_resize_offset(strides, arg):
  return sum([s * x[0] for s, x in zip(strides,arg)])

class ShapeTracker:
  __slots__ = "views", "var_vals"
  def __init__(self, shape:Union[ShapeTracker, Tuple[Union[Node,int], ...]], views:Optional[List[View]]=None):
    self.views: List[View] = views if views is not None else ([*cast(ShapeTracker, shape).views] if shape.__class__ is ShapeTracker else [View(shape)])
    self.var_vals: Dict[Variable, int] = shape.var_vals if isinstance(shape, ShapeTracker) else {}
  def __repr__(self): return f"ShapeTracker(shape={self.views[-1].shape}, views={self.views}, var_vals={self.var_vals})"
  def copy(self) -> ShapeTracker: return ShapeTracker(self.views[-1].shape, [*self.views])

  @property
  def contiguous(self) -> bool: return len(self.views) == 1 and self.views[0].contiguous

  @property
  def shape(self) -> Tuple[int, ...]: return self.views[-1].shape # NOTE: real type is Tuple[Union[Node, int], ...] but mypy complains about prod(shape)

  @property
  def key(self) -> Tuple[View, ...]: return tuple(self.views)

  # this is the real size (ish)
  def size(self): return prod([s for s,st in zip(self.views[-1].shape, self.views[-1].strides) if st != 0])

  # these are multiview strides, value is None if it's not a simple strided dimension
  # TODO: this can be shared code between simplify and merge_views
  def real_offset(self) -> int:
    real_offset, mask = self.expr_node(Variable('zero', 0, 0))
    assert real_offset.__class__ is NumNode, f"how is the offset not a number? {real_offset} {mask}"
    return real_offset.b

  # NOTE: if a stride is not always valid, it will be None
  def real_strides(self, ignore_valid=False) -> Tuple[Optional[Union[Node, int]], ...]:
    if len(self.views) == 1 and self.views[-1].mask is None: return self.views[-1].strides
    idxs = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(self.shape)]
    idx, valid = self.expr_idxs(idxs)
    ret: List[Optional[Union[Node, int]]] = [None] * len(self.views[-1].shape)
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

  def _expr_idx(self, idx, valid):
    for v in reversed(self.views[0:-1]):
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

  def needs_valid(self) -> bool:
    return any(v.mask is not None for v in self.views)

  # *** under this line are the movement ops ***

  def __unsafe_resize(self, arg: Tuple[Tuple[int, int], ...], mask=None):
    offset = get_unsafe_resize_offset(self.views[-1].strides, arg)
    if self.views[-1].mask:
      # move the old mask
      nmask = tuple([(max(mx-ax, 0), min(my-ax, ay-ax)) for (mx,my),(ax,ay) in zip(self.views[-1].mask, arg)])
      # merge the masks if we have two
      mask = tuple([(max(mx1, mx2), min(my1, my2)) for (mx1, my1), (mx2, my2) in zip(nmask, mask)]) if mask is not None else nmask
    self.views[-1] = View(tuple([y-x for x,y in arg]), self.views[-1].strides, self.views[-1].offset+offset, mask)

  def pad(self, arg: Tuple[Tuple[int, int], ...]):
    assert all((b>=0 and e>=0) for b,e in arg) and len(arg) == len(self.shape)
    if any(b or e for b, e in arg):
      zvarg, mask = get_pad_args(self.shape, arg)
      self.__unsafe_resize(zvarg, mask=mask)
    return self

  def shrink(self, arg: Tuple[Tuple[int, int], ...]):
    assert all((b>=0 and e<=s) for s,(b,e) in zip(self.shape,arg)) and len(arg) == len(self.shape)
    self.__unsafe_resize(arg)
    return self

  def expand(self, new_shape: Tuple[Union[Node,int], ...]) -> ShapeTracker:
    assert len(new_shape) == len(self.views[-1].shape)
    assert all(is_sym_int(x) and (s == x or (s == 1 and st == 0)) for s,x,st in zip(self.shape, new_shape, self.views[-1].strides)), f"can't expand {self.shape} into {new_shape}"
    # NOTE: can the mask ever be (0,0)?
    mask = tuple([(((0,0) if m != (0,1) else (0,ns)) if s != ns else m) for m,s,ns in zip(self.views[-1].mask, self.shape, new_shape)]) if self.views[-1].mask else None
    self.views[-1] = View(new_shape, self.views[-1].strides, self.views[-1].offset, mask)
    return self

  def reshape(self, new_shape: Tuple[Union[Node,int], ...]):
    if self.views[-1].shape == new_shape: return self
    new_ints, new_nodes = partition(new_shape, lambda s: isinstance(s, int))
    if new_nodes and all(isinstance(s, int) for s in self.shape):
      # reshape from all int shape into shape with a variable, update the variable value
      assert len(new_nodes) == 1 and isinstance(new_nodes[0], Variable), "only support adding one Variable to the int shape"
      new_var, new_val = new_nodes[0], prod(self.shape) // prod(new_ints)
      if new_var not in self.var_vals:
        assert new_var.min <= new_val <= new_var.max, f"variable value {new_val} out of range [{new_var.min}, {new_var.max}]"
        self.var_vals[new_var] = new_val
      else: assert self.var_vals[new_var] == new_val, f"value conflicts, was {self.var_vals[new_var]}, set to {new_val}"
    elif not new_nodes: self.var_vals = {}
    assert all(is_sym_int(x) and x > 0 for x in new_shape), f"shape must be symbolic ints and can't contain 0 or negative numbers {new_shape}"
    # only check size for int shapes. we don't check symbolic here as long as the reshape itself can be done
    if all(isinstance(s, int) for s in self.shape) and all(isinstance(s, int) for s in new_shape):
      assert prod(self.shape) == prod(new_shape), f"can't reshape {self.shape} -> {new_shape}" # type: ignore  # mypy cannot resolve, all ints here
    new_view, extra = _reshape(self.views[-1], new_shape)
    if extra: self.views.append(new_view)
    else: self.views[-1] = new_view
    return self

  def permute(self, axis: Tuple[int, ...]):
    assert all(isinstance(x, int) and x >= 0 and x < len(self.shape) for x in axis), f"invalid permute {axis} for {self.shape}"
    assert len(set(axis)) == len(axis) and len(axis) == len(self.shape), f"can't permute {self.shape} with {axis}"
    self.views[-1] = View(tuple([self.views[-1].shape[a] for a in axis]), tuple([self.views[-1].strides[a] for a in axis]), self.views[-1].offset, tuple([self.views[-1].mask[a] for a in axis]) if self.views[-1].mask is not None else None)
    return self

  # except for the negative case, you can build this from the others. invertible in the negative case
  def stride(self, mul: Tuple[int, ...]):
    assert all(isinstance(x, int) and x != 0 for x in mul), f"invalid stride {mul} for {self.shape}"
    strides = tuple([z*m for z,m in zip(self.views[-1].strides, mul)])
    new_shape = tuple([(s+(abs(m)-1))//abs(m) for s,m in zip(self.views[-1].shape, mul)])
    offset = sum([(s-1)*z for s,z,m in zip(self.views[-1].shape, self.views[-1].strides, mul) if m < 0])
    mask = tuple([(((mx if m > 0 else s-my)+(abs(m)-1))//abs(m), ((my if m > 0 else s-mx)+(abs(m)-1))//abs(m)) for (mx,my),s,m in zip(self.views[-1].mask, self.views[-1].shape, mul)]) if self.views[-1].mask is not None else None
    self.views[-1] = View(new_shape, strides, self.views[-1].offset + offset, mask)
    return self

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
