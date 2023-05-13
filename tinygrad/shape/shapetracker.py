# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
import functools
from enum import Enum, auto
from typing import Tuple, Union, List, Optional, Dict, Callable
from tinygrad.helpers import prod, DEBUG
from tinygrad.shape.symbolic import Variable, MulNode, NumNode, Node, SumNode, ModNode

# these ops live here
class MovementOps(Enum): RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); STRIDE = auto() # noqa: E702

def check_no_mul(test, var):
  if test == var: return True
  if isinstance(test, SumNode): return any(check_no_mul(x, var) for x in test.nodes) # in a sum is okay
  if isinstance(test, ModNode) and test.b%4 == 0: return check_no_mul(test.a, var)   # removing a mod is okay
  return False

@functools.lru_cache(maxsize=None)
def to_shape_strides(shape:Tuple[int, ...], strides:Tuple[int, ...]) -> List[Tuple[int, int]]:
  assert len(shape) == len(strides)
  ret = [(shape[0], strides[0])] if len(shape) > 0 else []
  for i in range(1, len(shape)):
    if (strides[i] != 0 and ret[-1][1] == shape[i]*strides[i]) or ret[-1][0] == 1 or (strides[i] == 0 and ret[-1][1] == 0):
      ret[-1] = (ret[-1][0] * shape[i], strides[i])
    else:
      ret.append((shape[i], strides[i]))
  return ret

@functools.lru_cache(maxsize=None)
def is_contiguous(shape:Tuple[int, ...], strides:Tuple[int, ...]) -> bool: return all(s1 == s2 or s == 1 for s,s1,s2 in zip(shape, strides, strides_for_shape(shape)))

class View:
  def __init__(self, shape:Tuple[int, ...], strides:Tuple[int, ...], offset:int=0, mask:Optional[Tuple[Tuple[int, int], ...]]=None):
    self.shape, self.strides, self.offset = shape, tuple(stride if shp != 1 else 0 for stride,shp in zip(strides, shape)), offset
    self.mask = mask
    self.shape_strides = to_shape_strides(self.shape, self.strides)
    self.contiguous: bool = self.offset == 0 and is_contiguous(self.shape, self.strides) and mask is None

  def __repr__(self): return f"View({self.shape}, {self.strides}, {self.offset}, {self.mask})"

  def expr_node_mask(self, idx, valid=None) -> Node:
    expr = [valid] if valid is not None else []
    if self.mask is not None:
      acc = 1
      for ns,(x,y) in list(zip(self.shape, self.mask))[::-1]:
        base = ((idx//acc) % ns)
        expr += [base >= x, base < y]
        acc *= ns
    return Variable.ands(expr)

  def idxs_to_idx(self, idxs):
    assert len(idxs) == len(self.shape), "need an idx for all dimensions"
    acc = 1
    ret = []
    for tidx,d in list(zip(idxs, self.shape))[::-1]:
      ret.append(tidx * acc)
      acc *= d
    return Variable.sum(ret)

  # generate an expression if you have a single idx variable
  def expr_node(self, idx=None) -> Node:
    if idx is None: idx = Variable('idx', 0, prod(self.shape))
    ret = [Variable.num(self.offset)]
    acc = 1
    for d,s in self.shape_strides[::-1]:
      ret.append(((idx//acc)%d)*s)
      acc *= d
    return Variable.sum(ret)

  # generate an expression if you have a variable or expression for each index
  def expr_idxs(self, idxs):
    assert len(idxs) == len(self.shape), f"need an idx for all dimensions {idxs} vs {self.shape}"
    return Variable.sum([Variable.num(self.offset)] + [idx*st for idx,sh,st in zip(idxs, self.shape, self.strides) if sh != 1 and st != 0])

@functools.lru_cache(maxsize=None)
def strides_for_shape(shape:Tuple[int, ...]) -> Tuple[int, ...]:
  strides = [1]
  for d in shape[::-1][:-1]: strides = [d*strides[0]] + strides
  return tuple(st if s != 1 else 0 for st, s in zip(strides, shape))

@functools.lru_cache(maxsize=None)
def view_from_shape(shape:Tuple[int, ...]) -> View:
  assert all(isinstance(x, int) for x in shape) and len(shape) != 0
  return View(tuple(shape), strides_for_shape(shape))

@functools.lru_cache(maxsize=None)
def merge_views(vm2:View, vm1:View) -> Optional[View]:
  if vm2.mask: return None  # this isn't supported yet
  new_strides, new_offset = [], vm2.expr_node(Variable.num(vm1.offset))
  assert isinstance(new_offset, NumNode), "new_offset wasn't a number?!?"
  for s,st in zip(vm1.shape, vm1.strides):
    this_dim = View(vm2.shape, vm2.strides).expr_node(Variable('idx', 0, s-1)*st)
    if s == 1:
      new_strides.append(0)   # all shape 1 can have stride 0
    elif isinstance(this_dim, NumNode) and this_dim.b == 0:
      new_strides.append(0)
    elif isinstance(this_dim, Variable):
      new_strides.append(1)
    elif isinstance(this_dim, MulNode) and isinstance(this_dim.a, Variable):
      new_strides.append(this_dim.b)
    else:
      if DEBUG >= 4: print("can't simplify", s, this_dim.render())
      break
  return View(vm1.shape, tuple(new_strides), new_offset.b, vm1.mask) if len(new_strides) == len(vm1.strides) else None

class ShapeTracker:
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], views:Optional[List[View]]=None):
    self.views: List[View] = views if views is not None else (shape.views[:] if isinstance(shape, ShapeTracker) else [view_from_shape(shape)])
  def __repr__(self): return f"ShapeTracker(shape={self.shape}, views={self.views})"
  def copy(self) -> ShapeTracker: return ShapeTracker(self.shape, self.views[:])

  @property
  def contiguous(self) -> bool: return len(self.views) == 1 and self.views[-1].contiguous

  @property
  def shape(self) -> Tuple[int, ...]: return self.views[-1].shape

  # this is the real size (ish)
  def size(self): return prod([s for s,st in zip(self.shape, self.views[-1].strides) if st != 0])

  def unit_stride_axes(self) -> List[int]:
    ret, acc = [], 1
    for j,s in list(enumerate(self.shape))[::-1]:
      if s == 1: continue
      var = Variable('idx', 0, s-1)
      this_dim = self.expr_node(var*acc)
      acc *= s
      if check_no_mul(this_dim[0], var): ret.append(j)
    return ret

  def _expr_idx(self, idx, valid):
    for v in self.views[0:-1][::-1]:
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
    idx = self.views[-1].expr_idxs(idxs)
    valid = self.views[-1].expr_node_mask(self.views[-1].idxs_to_idx(idxs))
    return self._expr_idx(idx, valid)

  def expr_node(self, idx='idx'):
    if isinstance(idx, str): idx = Variable(idx, 0, prod(self.shape)-1)
    return self._expr_idx(self.views[-1].expr_node(idx), self.views[-1].expr_node_mask(idx))

  def needs_valid(self) -> bool:
    return any(v.mask is not None for v in self.views)

  # *** under this line are the movement ops ***

  def __unsafe_resize(self, arg: Tuple[Tuple[int, int], ...], mask=None):
    offset = sum([self.views[-1].strides[i]*x for i,(x,_) in enumerate(arg)])
    if self.views[-1].mask:
      # move the old mask
      nmask = tuple((max(mx-ax, 0), min(my-ax, ay-ax)) for (mx,my),(ax,ay) in zip(self.views[-1].mask, arg))
      # merge the masks if we have two
      mask = tuple((max(mx1, mx2), min(my1, my2)) for (mx1, my1), (mx2, my2) in zip(nmask, mask)) if mask is not None else nmask
    self.views[-1] = View(tuple(y-x for x,y in arg), self.views[-1].strides, self.views[-1].offset+offset, mask)

  def pad(self, arg: Tuple[Tuple[int, int], ...]):
    assert all((b>=0 and e>=0) for b,e in arg) and len(arg) == len(self.shape)
    if all(b==0 and e==0 for b,e in arg): return self
    zvarg = tuple((-b,s+e) for s,(b,e) in zip(self.shape, arg))
    self.__unsafe_resize(zvarg, mask=tuple((b,s+b) for s,(b,_) in zip(self.shape, arg)))

  def shrink(self, arg: Tuple[Tuple[int, int], ...]):
    assert all((b>=0 and e<=s) for s,(b,e) in zip(self.shape,arg)) and len(arg) == len(self.shape)
    self.__unsafe_resize(arg)

  def expand(self, new_shape: Tuple[int, ...]):
    assert all(isinstance(x, int) and (s == x or (s == 1 and st == 0)) for s,x,st in zip(self.shape, new_shape, self.views[-1].strides)), f"can't expand {self.shape} into {new_shape}"
    # NOTE: can the mask ever be (0,0)?
    mask = tuple((((0,0) if m != (0,1) else (0,ns)) if s != ns else m) for m,s,ns in zip(self.views[-1].mask, self.shape, new_shape)) if self.views[-1].mask else None
    self.views[-1] = View(new_shape, self.views[-1].strides, self.views[-1].offset, mask)

  def reshape(self, new_shape: Tuple[int, ...]):
    if self.shape == new_shape: return self
    assert all(isinstance(x, int) and x > 0 for x in new_shape), f"shape must be ints and can't contain 0 or negative numbers {new_shape}"
    assert prod(self.shape) == prod(new_shape), f"can't reshape {self.shape} -> {new_shape}"

    # check if this is adding or removing 1s (only)
    # NOTE: this is optional, but removes most calls to (expensive!) merge_views (with mask, not optional)
    if tuple(x for x in self.shape if x != 1) == tuple(x for x in new_shape if x != 1):
      old_strides = [y for x,y in zip(self.shape, self.views[-1].strides) if x != 1]
      new_strides_tuple = tuple(0 if x == 1 else old_strides.pop(0) for x in new_shape)
      new_mask_tuple = None
      if self.views[-1].mask:
        if any(y!=(0,1) for x,y in zip(self.shape, self.views[-1].mask) if x == 1):
          # mask it all out!
          new_mask_tuple = tuple((0,0) for _ in new_shape)
        else:
          old_mask = [y for x,y in zip(self.shape, self.views[-1].mask) if x != 1]
          new_mask_tuple = tuple((0,1) if x == 1 else old_mask.pop(0) for x in new_shape)
      self.views[-1] = View(new_shape, new_strides_tuple, self.views[-1].offset, new_mask_tuple)
      return self

    view = View(new_shape, strides_for_shape(new_shape))
    if self.contiguous: self.views[-1] = view   # NOTE: if it's contiguous it can't have an offset
    else:
      if (merged_view := merge_views(self.views[-1], view)) is not None: self.views[-1] = merged_view
      else:
        if DEBUG >= 4: print(f"WARNING: creating new view with reshape {self} -> {new_shape}")
        self.views.append(view)

  def permute(self, axis: Tuple[int, ...]):
    assert all(isinstance(x, int) and x >= 0 and x < len(self.shape) for x in axis), f"invalid permute {axis} for {self.shape}"
    assert len(set(axis)) == len(axis) and len(axis) == len(self.shape), f"can't permute {self.shape} with {axis}"
    self.views[-1] = View(tuple(self.shape[a] for a in axis), tuple(self.views[-1].strides[a] for a in axis), self.views[-1].offset, tuple(self.views[-1].mask[a] for a in axis) if self.views[-1].mask is not None else None)

  # except for the negative case, you can build this from the others. invertible in the negative case
  def stride(self, mul: Tuple[int, ...]):
    assert all(isinstance(x, int) and x != 0 for x in mul), f"invalid stride {mul} for {self.shape}"
    strides = tuple(z*m for z,m in zip(self.views[-1].strides, mul))
    new_shape = tuple((s+(abs(m)-1))//abs(m) for s,m in zip(self.shape, mul))
    offset = sum([(s-1)*z for s,z,m in zip(self.shape, self.views[-1].strides, mul) if m < 0])
    mask = tuple((((mx if m > 0 else s-my)+(abs(m)-1))//abs(m), ((my if m > 0 else s-mx)+(abs(m)-1))//abs(m)) for (mx,my),s,m in zip(self.views[-1].mask, self.shape, mul)) if self.views[-1].mask is not None else None
    self.views[-1] = View(new_shape, strides, self.views[-1].offset + offset, mask)

  # *** entry point for external ***

  def movement_op(self, op, arg:Union[Tuple[int, ...], Tuple[Tuple[int, int], ...]]) -> ShapeTracker:
    assert isinstance(arg, tuple) and (len(arg) == len(self.shape) or op == MovementOps.RESHAPE), f"arg {arg} for {op} doesn't match dim of shape {self.shape}"
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
