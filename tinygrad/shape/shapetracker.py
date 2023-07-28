# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
from enum import Enum, auto
import functools
from typing import Dict, Tuple, Union, List, Optional, Callable, NamedTuple
from tinygrad.helpers import prod, DEBUG
from tinygrad.shape.symbolic import Variable, MulNode, NumNode, Node, SumNode, is_sym_int

# these ops live here
class MovementOps(Enum): RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); STRIDE = auto() # noqa: E702

@functools.lru_cache(maxsize=None)
def to_shape_strides(shape:Tuple[int, ...], strides:Tuple[int, ...]) -> Tuple[Tuple[int, int], ...]:
  assert len(shape) == len(strides)
  ret = [(shape[0], strides[0])] if len(shape) > 0 else []
  for i in range(1, len(shape)):
    if (strides[i] != 0 and ret[-1][1] == shape[i]*strides[i]) or ret[-1][0] == 1 or (strides[i] == 0 and ret[-1][1] == 0):
      ret[-1] = (ret[-1][0] * shape[i], strides[i])
    else:
      ret.append((shape[i], strides[i]))
  return tuple(ret)

@functools.lru_cache(maxsize=None)
def shape_strides_contiguous(shape:Tuple[int, ...], strides:Tuple[int, ...]) -> bool: return all(s1 == s2 or s == 1 for s,s1,s2 in zip(shape, strides, strides_for_shape(shape)))

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
    contiguous = offset == 0 and shape_strides_contiguous(shape, strides) and mask is None
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
    if idx is None: idx = Variable('idx', 0, prod(self.shape))
    ret: List[Node] = [Variable.num(self.offset)] if self.offset else []
    acc = 1
    for d,s in reversed(self.shape_strides):
      ret.append(((idx//acc)%d)*s)
      acc *= d
    return Variable.sum(ret)

  # generate an expression if you have a variable or expression for each index
  def expr_idxs(self, idxs) -> Node:
    assert len(idxs) == len(self.shape), f"need an idx for all dimensions {idxs} vs {self.shape}"
    return Variable.sum([Variable.num(self.offset)] + [idx*st for idx,sh,st in zip(idxs, self.shape, self.strides) if sh != 1 and st != 0])


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
  return tuple([st if s != 1 else 0 for st, s in zip(strides, shape)])


@functools.lru_cache(maxsize=None)
def merge_views(vm2:View, vm1:View) -> Optional[View]:
  if vm2.mask: return None  # this isn't supported yet
  strides = ShapeTracker.real_strides((vm2, vm1))
  if None in strides: return None
  return View(vm1.shape, strides, ShapeTracker.real_offset((vm2, vm1)), vm1.mask)

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

  new_view = View(new_shape, strides_for_shape(new_shape))
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

class ShapeTracker(NamedTuple):
  views: Tuple[View]

  @property
  def contiguous(self) -> bool: return self.views[-1].contiguous and len(self.views) == 1
  @property
  def shape(self) -> Tuple[int, ...]: return self.views[-1].shape

  @staticmethod
  @functools.lru_cache(None)
  def from_shape(shape: Tuple[int]) -> ShapeTracker: return ShapeTracker((View(shape),))

  # this is the real size (ish)
  def size(self): return prod([s for s,st in zip(self.views[-1].shape, self.views[-1].strides) if st != 0])

  # these are multiview strides, value is None if it's not a simple strided dimension
  # TODO: this can be shared code between simplify and merge_views
  @staticmethod
  def real_offset(views: Tuple[View, ...]) -> int:
    real_offset, mask = ShapeTracker.expr_node(views, Variable('zero', 0, 0))
    assert real_offset.__class__ is NumNode, f"how is the offset not a number? {real_offset} {mask}"
    return real_offset.b

  # NOTE: if a stride is not always valid, it will be None
  @staticmethod
  @functools.lru_cache(None)
  def real_strides(views: Tuple[View], ignore_valid=False) -> Tuple[Optional[int], ...]:
    if len(views) == 1 and views[-1].mask is None: return views[-1].strides
    idxs = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(views[-1].shape)]
    idx, valid = ShapeTracker.expr_idxs(views, tuple(idxs))
    ret: List[Optional[int]] = [None] * len(views[-1].shape)
    for this_dim in (idx.nodes if isinstance(idx, SumNode) else [idx]):
      if isinstance(this_dim, MulNode) and isinstance(this_dim.a, Variable) and isinstance(this_dim.b, int):
        ret[idxs.index(this_dim.a)] = this_dim.b
      elif isinstance(this_dim, Variable):
        ret[idxs.index(this_dim)] = 1
    idx_vars, valid_vars = idx.vars(), valid.vars()
    for i,tidx in enumerate(idxs):
      if tidx in valid_vars and not ignore_valid: ret[i] = None
      elif tidx not in idx_vars: ret[i] = 0
    return tuple(ret)


  @staticmethod
  @functools.lru_cache(None)
  def _expr_idx(views: Tuple[View], idx, valid):
    for v in reversed(views[:-1]):    # type: ignore
      valid = v.expr_node_mask(idx, valid)
      idx = v.expr_node(idx)
    return idx, valid

  @staticmethod
  def simplify(views: Tuple[View]) -> Tuple[View]:
    if len(views) >= 2:
      new_view = merge_views(views[-2], views[-1])  # type: ignore
      if new_view:
        if DEBUG >= 4: print(f"st simplify : {views[-2]} + {views[-1]} = {new_view}")  # type: ignore
        return ShapeTracker.simplify((*views[:-2], new_view))
    return views

  @staticmethod
  @functools.lru_cache(None)
  def expr_idxs(views: Tuple[View], idxs=None):
    if idxs is None: idxs = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(views[-1].shape)]
    return ShapeTracker._expr_idx(views, views[-1].expr_idxs(idxs), views[-1].expr_node_mask(idxs_to_idx(views[-1].shape, idxs)))
  
  @staticmethod
  def expr_node(views: Tuple[View,...], idx='idx'):
    if idx.__class__ is str: idx = Variable(idx, 0, prod(views[-1].shape)-1)
    return ShapeTracker._expr_idx(views, views[-1].expr_node(idx), views[-1].expr_node_mask(idx))

  @staticmethod
  def needs_valid(views: Tuple[View]) -> bool: return any(v.mask is not None for v in views)

  # *** under this line are the movement ops ***

  @staticmethod
  def __unsafe_resize(view: View, arg: Tuple[Tuple[int, int], ...], mask=None) -> View:
    offset = get_unsafe_resize_offset(view.strides, arg)
    if view.mask:
      # move the old mask
      nmask = tuple([(max(mx-ax, 0), min(my-ax, ay-ax)) for (mx,my),(ax,ay) in zip(view.mask, arg)])
      # merge the masks if we have two
      mask = tuple([(max(mx1, mx2), min(my1, my2)) for (mx1, my1), (mx2, my2) in zip(nmask, mask)]) if mask is not None else nmask
    return View(tuple([y-x for x,y in arg]), view.strides, view.offset+offset, mask)

  @staticmethod
  def pad(view: View, arg: Tuple[Tuple[int, int], ...]) -> View:
    assert all((b>=0 and e>=0) for b,e in arg) and len(arg) == len(view.shape)
    if any(b or e for b, e in arg):
      zvarg, mask = get_pad_args(view.shape, arg)
    return ShapeTracker.__unsafe_resize(view, zvarg, mask=mask)

  @staticmethod
  def shrink(view: View, arg: Tuple[Tuple[int, int], ...]) -> View:
    assert all((b>=0 and e<=s) for s,(b,e) in zip(view.shape,arg)) and len(arg) == len(view.shape)
    return ShapeTracker.__unsafe_resize(view, arg)

  @staticmethod
  @functools.lru_cache(None)
  def expand(view: View, new_shape:Tuple[int, ...]) -> View:
    assert len(new_shape) == len(view.shape)
    assert all(isinstance(x, int) and (s == x or (s == 1 and st == 0)) for s,x,st in zip(view.shape, new_shape, view.strides)), f"can't expand {view.shape} into {new_shape}"
    # NOTE: can the mask ever be (0,0)?
    mask = tuple([(((0,0) if m != (0,1) else (0,ns)) if s != ns else m) for m,s,ns in zip(view.mask, view.shape, new_shape)]) if view.mask else None
    return View(new_shape, view.strides, view.offset, mask)

  @staticmethod
  @functools.lru_cache(None)
  def reshape(views: Tuple[View], new_shape: Tuple[int, ...]):
    if views[-1].shape == new_shape: return views
    assert all(is_sym_int(x) and x > 0 for x in new_shape), f"shape must be symbolic ints and can't contain 0 or negative numbers {new_shape}"
    # only check size for int shapes. we don't check symbolic here as long as the reshape itself can be done
    if all(isinstance(s, int) for s in views[-1].shape) and all(isinstance(s, int) for s in new_shape):
      assert prod(views[-1].shape) == prod(new_shape), f"can't reshape {views[-1].shape} -> {new_shape}"
    new_view, extra = _reshape(views[-1], new_shape)
    return (views if extra else views[:-1]) + (new_view, )

  @staticmethod
  def permute(view: View, axis: Tuple[int, ...]) -> View:
    assert all(isinstance(x, int) and x >= 0 and x < len(view.shape) for x in axis), f"invalid permute {axis} for {view.shape}"
    assert len(set(axis)) == len(axis) == len(view.shape), f"can't permute {view.shape} with {axis}"
    return View(tuple([view.shape[a] for a in axis]), tuple([view.strides[a] for a in axis]), view.offset, tuple([view.mask[a] for a in axis]) if view.mask is not None else None)

  # except for the negative case, you can build this from the others. invertible in the negative case
  @staticmethod
  def stride(view: View, mul: Tuple[int, ...]) -> View:
    assert all(isinstance(x, int) and x != 0 for x in mul), f"invalid stride {mul} for {view.shape}"
    strides = tuple([z*m for z,m in zip(view.strides, mul)])
    new_shape = tuple([(s+(abs(m)-1))//abs(m) for s,m in zip(view.shape, mul)])
    offset = sum([(s-1)*z for s,z,m in zip(view.shape, view.strides, mul) if m < 0])
    mask = tuple([(((mx if m > 0 else s-my)+(abs(m)-1))//abs(m), ((my if m > 0 else s-mx)+(abs(m)-1))//abs(m)) for (mx,my),s,m in zip(view.mask, view.shape, mul)]) if view.mask is not None else None
    return View(new_shape, strides, view.offset + offset, mask)

  # *** entry point for external ***

  @staticmethod
  @functools.lru_cache(None)
  def movement_op(views: Tuple[View], op: MovementOps, arg:Union[Tuple[int, ...], Tuple[Tuple[int, int], ...]]) -> ShapeTracker:
    assert isinstance(arg, tuple) and (len(arg) == len(views[-1].shape) or op == MovementOps.RESHAPE), f"arg {arg} for {op} doesn't match dim of shape {views[-1].shape}"
    return op_func[op](views if op is MovementOps.RESHAPE else views[-1], arg)
 

op_func: Dict[MovementOps, Callable] = {MovementOps.RESHAPE: ShapeTracker.reshape, MovementOps.EXPAND: ShapeTracker.expand, MovementOps.PAD: ShapeTracker.pad,
                                         MovementOps.SHRINK: ShapeTracker.shrink, MovementOps.PERMUTE: ShapeTracker.permute, MovementOps.STRIDE: ShapeTracker.stride}

@functools.lru_cache(maxsize=None)
def unit_stride_axes(views, ignore_valid=False) -> List[int]: return [i for i,st in enumerate(ShapeTracker.real_strides(views, ignore_valid)) if st == 1]

# returns the axes to create new_shape if new_shape can be created by combining axis from old_shape
@functools.lru_cache(maxsize=None)
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
      if new_shape[i] % old_shape[old_shape_i] != 0 or prod([old_shape[x] for x in axis_groups[i]]) * old_shape[old_shape_i] > new_shape[i]:
        return None
      axis_groups[i].append(old_shape_i)
      # Move to next axes group if total size of all dimensions match.
      if prod([old_shape[x] for x in axis_groups[i]]) == new_shape[i]:
        if i < len(new_shape) - 1: i += 1
      old_shape_i += 1
  return axis_groups
