# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
from enum import Enum, auto
import functools
from typing import Dict, Tuple, Union, List, Optional, Callable, cast, NamedTuple
from tinygrad.helpers import prod, DEBUG
from tinygrad.shape.symbolic import Variable, MulNode, NumNode, Node

# these ops live here
class MovementOps(Enum): RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); STRIDE = auto() # noqa: E702

@functools.lru_cache(maxsize=None)
def to_shape_strides(shape:Tuple[int, ...], strides:Tuple[int, ...]) -> List[Tuple[int, int]]:
  assert len(shape) == len(strides)
  ret = [(shape[0], strides[0])] if len(shape) > 0 else []
  for i in range(1, len(shape)):
    if (strides[i] != 0 and ret[-1][1] == shape[i]*strides[i]) or ret[-1][0] == 1 or (strides[i] == 0 and ret[-1][1] == 0):
      ret[-1] = (ret[-1][0] * shape[i], strides[i])
    else:
      ret.append((shape[i], strides[i]))
  return tuple(ret)

@functools.lru_cache(maxsize=None)
def is_contiguous(shape:Tuple[int, ...], strides:Tuple[int, ...]) -> bool:   
  for s,s1,s2 in zip(shape, strides, strides_for_shape(shape)):
    if not s == 1 or s1 == s2: return False
  return True

@functools.lru_cache(maxsize=None)
def filter_strides(shape:Tuple[int, ...], strides:Tuple[int, ...]) -> Tuple[int, ...]:
  new_strides = []
  for stride, shp in zip(strides, shape):
    if shp != 1: new_strides.append(stride)
    else: new_strides.append(0)
  return tuple(new_strides)


class View(NamedTuple):
  shape:Tuple[int, ...]
  strides:Tuple[int, ...]
  shape_strides:Tuple[int, ...]
  contiguous: bool
  offset:int=0
  mask:Optional[Tuple[Tuple[int, int]]]=None

  @staticmethod
  @functools.lru_cache(maxsize=None)
  def create(shape, strides=None, offset=0, mask=None):
    strides = strides_for_shape(shape) if strides is None else strides
    strides = filter_strides(shape, strides)
    shape_strides = to_shape_strides(shape, strides)
    contiguous = offset == 0 and mask is None and is_contiguous(shape, strides) 
    return View(shape, strides, shape_strides, contiguous, offset, mask)
                
  @staticmethod
  def expr_node_mask(shape, mask, idx, valid=None) -> Node:
    expr = [valid] if valid is not None else []
    if mask is not None:
      acc = 1
      for ns,(x,y) in reversed(list(zip(shape, mask))):
        base = ((idx//acc) % ns)
        expr += [base >= x, base < y]
        acc *= ns
    return Node.num(1) if not expr  else expr[0] if len(expr) == 1 else Variable.ands(expr)

  @staticmethod
  @functools.lru_cache(maxsize=None)
  def idxs_to_idx(shape, idxs):
    assert len(idxs) == len(shape), "need an idx for all dimensions"
    acc = 1
    ret = []
    for tidx,d in reversed(list(zip(idxs, shape))):
      ret.append(tidx * acc)
      acc *= d
    return Node.num(0) if not ret else ret[0] if len(ret) == 1 else Variable.sum(ret)


  # generate an expression if you have a single idx variable
  @staticmethod
  @functools.lru_cache(maxsize=None)
  def expr_node(shape, strides, offset, idx=None) -> Node:
    if idx is None: idx = Variable('idx', 0, prod(shape))
    ret = [Node.num(offset)] if offset else []
    acc = 1
    for d,s in reversed(to_shape_strides(shape, strides)):
      if not (s==0 or acc > idx.max or (idx.__class__ in (MulNode, NumNode) and (idx.b/acc)%d == 0)):
        ret.append(((idx//acc)%d)*s)
      acc *= d
    return Node.sum(ret)

  # generate an expression if you have a variable or expression for each index
  @staticmethod
  @functools.lru_cache(maxsize=None)
  def expr_idxs(shape, strides, offset, idxs):
    assert len(idxs) == len(shape), f"need an idx for all dimensions {idxs} vs {shape}"
    nodes = [idx*st for idx,sh,st in zip(idxs, shape, strides) if sh != 1 and st != 0]
    if not offset: return Node.num(0) if not nodes else nodes[0] if len(nodes) == 1 else Node.sum(nodes)
    return Variable.sum([Variable.num(offset)] + nodes)
  
@functools.lru_cache(maxsize=None)
def strides_for_shape(shape:Tuple[int, ...]) -> Tuple[int, ...]:
  strides = [1] if shape else []
  for d in shape[::-1][:-1]: strides = [d*strides[0]] + strides
  return tuple([st if s != 1 else 0 for st, s in zip(strides, shape)])

def view_from_shape(shape:Tuple[int, ...]) -> View:
  assert all([isinstance(x, int) for x in shape])
  return View.create(tuple(shape))

@functools.lru_cache(maxsize=None)
def merge_views(vm2:View, vm1:View) -> Optional[View]:
  if vm2.mask: return None  # this isn't supported yet
  strides = ShapeTracker.real_strides((vm2, vm1), vm1.shape)
  if None in strides: return None
  return View.create(vm1.shape, strides, ShapeTracker.real_offset((vm2, vm1)), vm1.mask)

@functools.lru_cache(maxsize=None)
def _reshape(view: View, new_shape: Tuple[int, ...]) -> Tuple[View, bool]:
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
          new_mask_tuple = tuple([(0,0) for _ in new_shape])
          break
      else:
        new_mask: List[Tuple[int, int]] = [y for x,y in zip(shape, mask) if x != 1]
        new_mask_tuple = tuple([(0,1) if x == 1 else new_mask.pop(0) for x in new_shape])
    return View.create(new_shape, new_strides_tuple, offset, new_mask_tuple), False

  new_view = View.create(new_shape, strides_for_shape(new_shape))
  if view.contiguous: return new_view, False # NOTE: if it's contiguous it can't have an offset
  else:
    if (merged_view := merge_views(view, new_view)) is not None: return merged_view, False
    else:
      if DEBUG >= 4: print(f"WARNING: creating new view with reshape {view} -> {new_shape}")
      return new_view, True

@functools.lru_cache(maxsize=None)
def get_pad_args(shape, arg: Tuple[Tuple[int, int], ...]):
  return tuple([(-b,s+e) for s,(b,e) in zip(shape, arg)]), tuple([(b,s+b) for s,(b,_) in zip(shape, arg)])

@functools.lru_cache(maxsize=None)
def get_unsafe_resize_offset(strides, arg):
  return sum([s * x[0] for s, x in zip(strides,arg)])

class ShapeTracker:
  __slots__ = "views"
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], views:Optional[List[View]]=None):
    self.views: List[View] = views if views is not None else ([*cast(ShapeTracker, shape).views] if shape.__class__ is ShapeTracker else [view_from_shape(shape)])
  def __repr__(self): return f"ShapeTracker(shape={self.views[-1].shape}, views={self.views})"
  def copy(self) -> ShapeTracker: return ShapeTracker(self.views[-1].shape, [*self.views])

  @property
  def contiguous(self) -> bool: return len(self.views) == 1 and self.views[0].contiguous

  @property
  def shape(self) -> Tuple[int, ...]: return self.views[-1].shape

  @property
  def key(self) -> Tuple[int, ...]: return tuple(map(View.key, self.views))

  # this is the real size (ish)
  def size(self): return prod([s for s,st in zip(self.views[-1].shape, self.views[-1].strides) if st != 0])

  # these are multiview strides, value is None if it's not a simple strided dimension
  # TODO: this can be shared code between simplify and merge_views

  @functools.lru_cache(None)
  @staticmethod
  def real_offset(views) -> int:
    real_offset, mask = ShapeTracker.expr_node(views, Variable('zero', 0, 0))
    assert real_offset.__class__ is NumNode, f"how is the offset not a number? {real_offset} {mask}"
    return real_offset.b

  @staticmethod
  @functools.lru_cache(None)
  def real_strides(views, shape) -> Tuple[Optional[int], ...]:
    if len(views) == 1: return views[-1].strides
    ret: List[Optional[int]] = []
    acc, real_offset = 1, ShapeTracker.real_offset(views)
    for s in reversed(shape):
      if s == 1:  # fast path, all shape 1 have stride 0
        ret.append(0)
        continue
      var = Variable('idx', 0, s-1)
      this_dim, _ = ShapeTracker.expr_node(views, var*acc)
      this_dim -= real_offset
      acc *= s
      # TODO: sometimes a mod here is okay if you are say, reading a float4, since you only care %4
      # if test.__class__ is ModNode and test.b%4 == 0: return check_no_mul(test.a, var)   # removing a mod is okay
      if this_dim.__class__ is MulNode and cast(MulNode, this_dim).a.__class__ is Variable: ret.append(this_dim.b)
      elif this_dim.__class__ is NumNode and this_dim.b == 0: ret.append(0)
      elif this_dim.__class__ is Variable: ret.append(1)
      else: ret.append(None)
    return tuple(ret[::-1])
  
  def unit_stride_axes(self) -> List[int]: return [i for i,st in enumerate(ShapeTracker.real_strides(tuple(self.views), self.shape)) if st == 1]

  @staticmethod
  def expr_idx(front_views, idx, valid):
    for v in reversed(front_views):
      valid = View.expr_node_mask(v.shape, v.mask, idx, valid)
      idx = View.expr_node(v.shape, v.strides, v.offset, idx)
    return idx, valid

  @staticmethod
  def simplify(views):
    if len(views) >= 2:
      new_view = merge_views(views[-2], views[-1])
      if new_view:
        if DEBUG >= 4: print(f"st simplify : {views[-2]} + {views[-1]} = {new_view}")
        return ShapeTracker.simplify(tuple(list(views)[:-2] + [new_view]))   
    return list(views)

  @staticmethod
  def expr_idxs(views, idxs=None):
    v = views[-1]
    if idxs is None: idxs = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(v.shape)]
    idx = View.expr_idxs(v.shape, v.strides, v.offset, tuple(idxs))
    valid = View.expr_node_mask(v.shape, v.mask, View.idxs_to_idx(v.shape, tuple(idxs)))
    return ShapeTracker.expr_idx(views[0:-1], idx, valid)
  
  @staticmethod
  def expr_node(views, idx='idx'):
    v = views[-1]
    if idx.__class__ is str: idx = Variable(idx, 0, prod(v.shape)-1)
    return ShapeTracker.expr_idx(views[0:-1], View.expr_node(v.shape, v.strides, v.offset, idx), View.expr_node_mask(v.shape, v.mask, idx))
  
  def needs_valid(self) -> bool:
    return any([v.mask is not None for v in self.views])

  # *** under this line are the movement ops ***

  def __unsafe_resize(self, arg: Tuple[Tuple[int, int], ...], mask=None):
    offset = get_unsafe_resize_offset(self.views[-1].strides, arg)
    if self.views[-1].mask:
      # move the old mask
      nmask = tuple([(max(mx-ax, 0), min(my-ax, ay-ax)) for (mx,my),(ax,ay) in zip(self.views[-1].mask, arg)])
      # merge the masks if we have two
      mask = tuple([(max(mx1, mx2), min(my1, my2)) for (mx1, my1), (mx2, my2) in zip(nmask, mask)]) if mask is not None else nmask
    self.views[-1] = View.create(tuple([y-x for x,y in arg]), self.views[-1].strides, self.views[-1].offset+offset, mask)

  def pad(self, arg: Tuple[Tuple[int, int], ...]):
    assert all((b>=0 and e>=0) for b,e in arg) and len(arg) == len(self.views[-1].shape)
    if any([b or e for b, e in arg]):
      zvarg, mask = get_pad_args(self.views[-1].shape, arg)
      self.__unsafe_resize(zvarg, mask=mask)
    return self

  def shrink(self, arg: Tuple[Tuple[int, int], ...]):
    assert all((b>=0 and e<=s) for s,(b,e) in zip(self.views[-1].shape,arg)) and len(arg) == len(self.views[-1].shape)
    self.__unsafe_resize(arg)
    return self

  def expand(self, new_shape: Tuple[int, ...]) -> ShapeTracker:
    assert len(new_shape) == len(self.views[-1].shape)
    assert all(isinstance(x, int) and (s == x or (s == 1 and st == 0)) for s,x,st in zip(self.views[-1].shape, new_shape, self.views[-1].strides)), f"can't expand {self.views[-1].shape} into {new_shape}"
    # NOTE: can the mask ever be (0,0)?
    mask = tuple([(((0,0) if m != (0,1) else (0,ns)) if s != ns else m) for m,s,ns in zip(self.views[-1].mask, self.views[-1].shape, new_shape)]) if self.views[-1].mask else None
    self.views[-1] = View.create(new_shape, self.views[-1].strides, self.views[-1].offset, mask)
    return self

  def reshape(self, new_shape: Tuple[int, ...]):
    if self.views[-1].shape == new_shape: return self
    assert all([isinstance(x, int) and x > 0 for x in new_shape]), f"shape must be ints and can't contain 0 or negative numbers {new_shape}"
    assert prod(self.views[-1].shape) == prod(new_shape), f"can't reshape {self.views[-1].shape} -> {new_shape}"
    new_view, extra = _reshape(self.views[-1], new_shape)
    if extra: self.views.append(new_view)
    else: self.views[-1] = new_view
    return self

  def permute(self, axis: Tuple[int, ...]):
    assert all(isinstance(x, int) and x >= 0 and x < len(self.views[-1].shape) for x in axis), f"invalid permute {axis} for {self.views[-1].shape}"
    assert len(set(axis)) == len(axis) and len(axis) == len(self.views[-1].shape), f"can't permute {self.views[-1].shape} with {axis}"
    self.views[-1] = View.create(tuple([self.views[-1].shape[a] for a in axis]), tuple([self.views[-1].strides[a] for a in axis]), self.views[-1].offset, tuple([self.views[-1].mask[a] for a in axis]) if self.views[-1].mask is not None else None)
    return self

  # except for the negative case, you can build this from the others. invertible in the negative case
  def stride(self, mul: Tuple[int, ...]):
    assert all(isinstance(x, int) and x != 0 for x in mul), f"invalid stride {mul} for {self.views[-1].shape}"
    strides = tuple([z*m for z,m in zip(self.views[-1].strides, mul)])
    new_shape = tuple([(s+(abs(m)-1))//abs(m) for s,m in zip(self.views[-1].shape, mul)])
    offset = sum([(s-1)*z for s,z,m in zip(self.views[-1].shape, self.views[-1].strides, mul) if m < 0])
    mask = tuple([(((mx if m > 0 else s-my)+(abs(m)-1))//abs(m), ((my if m > 0 else s-mx)+(abs(m)-1))//abs(m)) for (mx,my),s,m in zip(self.views[-1].mask, self.views[-1].shape, mul)]) if self.views[-1].mask is not None else None
    self.views[-1] = View.create(new_shape, strides, self.views[-1].offset + offset, mask)
    return self

  # *** entry point for external ***

  def movement_op(self, op: MovementOps, arg:Union[Tuple[int, ...], Tuple[Tuple[int, int], ...]]) -> ShapeTracker:
    assert isinstance(arg, tuple) and (len(arg) == len(self.views[-1].shape) or op == MovementOps.RESHAPE), f"arg {arg} for {op} doesn't match dim of shape {self.views[-1].shape}"
    dispatch[op](self, arg)
    return self

dispatch: Dict[MovementOps, Callable] = {MovementOps.RESHAPE: ShapeTracker.reshape, MovementOps.EXPAND: ShapeTracker.expand, MovementOps.PAD: ShapeTracker.pad,
                                         MovementOps.SHRINK: ShapeTracker.shrink, MovementOps.PERMUTE: ShapeTracker.permute, MovementOps.STRIDE: ShapeTracker.stride}

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
      if new_shape[i] % old_shape[old_shape_i] != 0 or prod([old_shape[x] for x in axis_groups[i]]) * old_shape[old_shape_i] > new_shape[i]:
        return None
      axis_groups[i].append(old_shape_i)
      # Move to next axes group if total size of all dimensions match.
      if prod([old_shape[x] for x in axis_groups[i]]) == new_shape[i]:
        if i < len(new_shape) - 1: i += 1
      old_shape_i += 1
  return axis_groups
