# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
import functools, itertools, operator
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Set, cast, Union, Iterable
from tinygrad.ops import MovementOps

from tinygrad.helpers import prod, DEBUG, merge_dicts
from tinygrad.shape.symbolic import Variable, MulNode, Node, SumNode, NumNode, sint
from tinygrad.shape.view import View, _merge_dims

def expr_node_mask(view:View, idx:Node, valid:Optional[Node]=None) -> Node:
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
def expr_node(view:View, idx:Optional[Node]=None) -> Node:
  if idx is None: idx = Variable('idx', 0, prod(view.shape)-1)
  ret: List[Node] = [NumNode(view.offset) if isinstance(view.offset, int) else view.offset] if view.offset else []
  acc = 1
  for d,s,_ in reversed(_merge_dims(view.shape, view.strides)):
    ret.append(((idx//acc)%d)*s)
    acc *= d
  return Variable.sum(ret)

# generate an expression if you have a variable or expression for each index
def expr_idxs(view:View, idxs:Tuple[Node, ...]) -> Node:
  assert len(idxs) == len(view.shape), f"need an idx for all dimensions {idxs} vs {view.shape}"
  return Variable.sum([NumNode(view.offset) if isinstance(view.offset, int) else view.offset] + [idx*st for idx,sh,st in zip(idxs, view.shape, view.strides) if sh != 1 and st != 0])

@functools.lru_cache(maxsize=None)
def merge_views(vm2:View, vm1:View) -> Optional[View]:
  if vm2.mask or vm1.offset != 0: return None  # this isn't supported yet
  if None in (strides := ShapeTracker((vm2, vm1)).real_strides()): return None
  return View.create(vm1.shape, cast(Tuple[sint, ...], strides), vm2.offset, vm1.mask)

@functools.lru_cache(maxsize=None)
def idxs_to_idx(shape:Tuple[int, ...], idxs:Tuple[Node, ...]) -> Node:
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

  def size(self) -> int:
    if 0 in self.shape: return 0
    ret = self.expr_idxs()[0].max
    while not isinstance(ret, int): ret = ret.max    # TODO: this is a while loop?!? it should be more clear what max does
    assert isinstance(ret, int), f"ret must be integer, {ret=} isn't"
    return ret+1

  def vars(self) -> Set[Variable]: return set.union(*[v.vars() for v in self.views], set())

  @property
  def var_vals(self) -> Dict[Variable, int]: return merge_dicts([dict([v.unbind()]) for v in self.vars()])

  def unbind(self) -> ShapeTracker: return ShapeTracker(tuple(v.unbind() for v in self.views))

  def to_movement_ops(self) -> List[Tuple[MovementOps, Tuple]]:
    to_apply:List[Tuple[MovementOps, Tuple]] = []
    for v in self.views:
      real_shape = tuple(y-x for x,y in v.mask) if v.mask else v.shape
      real_offset = 0 if 0 in real_shape else (v.offset + (sum(x*st for (x,_),st in zip(v.mask, v.strides)) if v.mask else 0))
      # first, we apply the offset
      # then, we make it the correct shape
      # then, we apply permutations
      to_apply.append((MovementOps.AS_STRIDED, (tuple([s if st != 0 else 1 for s,st in zip(real_shape, v.strides)]), v.strides, real_offset)))
      # then, we apply pre expand pads
      if v.mask is not None:
        pre_expand_pads = tuple((x,s-y) if st != 0 else (0,0) for (x,y),s,st in zip(v.mask, v.shape, v.strides))
        post_expand_pads = tuple((x,s-y) if st == 0 else (0,0) for (x,y),s,st in zip(v.mask, v.shape, v.strides))
        if any(x != (0,0) for x in pre_expand_pads):
          to_apply.append((MovementOps.PAD, pre_expand_pads))
          real_shape = tuple(x+s[0]+s[1] for x,s in zip(real_shape, pre_expand_pads))
      # then, we do any expands
      # NOTE: this is a good idea even without masks, since torch doesn't support negative strides and has to make a copy
      if any(s != 1 and st == 0 for s,st in zip(real_shape, v.strides)): to_apply.append((MovementOps.EXPAND, real_shape))
      # lastly, we apply post expand pads
      if v.mask is not None and any(x != (0,0) for x in post_expand_pads): to_apply.append((MovementOps.PAD, post_expand_pads))
    return to_apply

  # NOTE: if a stride is not always valid, it will be None
  def real_strides(self, ignore_valid=False) -> Tuple[Optional[sint], ...]:
    if len(self.views) == 1 and self.views[-1].mask is None: return self.views[-1].strides
    idxs: List[Node] = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(self.shape)]
    idx, valid = self.expr_idxs(idxs)
    ret: List[Optional[sint]] = [None] * len(self.views[-1].shape)
    for this_dim in (idx.nodes if isinstance(idx, SumNode) else [idx]):
      idx_maybe, stride_maybe = (this_dim.a, this_dim.b) if isinstance(this_dim, MulNode) else (this_dim, 1)
      try: ret[idxs.index(idx_maybe)] = stride_maybe
      except ValueError: pass
    idx_vars, valid_vars = idx.vars(), valid.vars()
    for i,tidx in enumerate(idxs):
      if tidx in valid_vars and not ignore_valid: ret[i] = None
      elif tidx not in idx_vars: ret[i] = 0
    return tuple(ret)

  def unit_stride_axes(self, ignore_valid=False) -> List[int]: return [i for i,st in enumerate(self.real_strides(ignore_valid)) if st == 1]

  def _expr_idx(self, idx:Node, valid:Node) -> Tuple[Node, Node]:
    for v in reversed(self.views[0:-1]):
      if valid.max == 0: return NumNode(-1), valid
      valid = expr_node_mask(v, idx, valid)
      idx = expr_node(v, idx)
    return idx, valid

  def simplify(self) -> ShapeTracker:
    if len(self.views) >= 2:
      if (new_view := merge_views(self.views[-2], self.views[-1])) is not None:
        if DEBUG >= 4: print(f"st simplify : {self.views[-2]} + {self.views[-1]} = {new_view}")
        return ShapeTracker(self.views[:-2] + (new_view,)).simplify()
    return self

  def expr_idxs(self, idxs:Optional[Iterable[Node]]=None):
    if idxs is None: idxs = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(self.shape)]
    idx = expr_idxs(self.views[-1], tuple(idxs))
    valid = expr_node_mask(self.views[-1], idxs_to_idx(self.views[-1].shape, tuple(idxs)))
    return self._expr_idx(idx, valid)

  def expr_node(self, idx:Union[Node,str]='idx'):
    if isinstance(idx, str): idx = Variable(idx, 0, prod(self.shape)-1)
    return self._expr_idx(expr_node(self.views[-1], idx), expr_node_mask(self.views[-1], idx))

  def axis_is_masked(self, axis:int) -> bool:
    _, valid = self.expr_idxs()
    return f'idx{axis}' in [v.expr for v in valid.vars()]

  # *** under this line are the movement ops ***

  def pad(self, arg: Tuple[Tuple[int, int], ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].pad(arg), ))
  def shrink(self, arg: Tuple[Tuple[sint, sint], ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].shrink(arg), ))
  def expand(self, new_shape: Tuple[sint, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].expand(new_shape), ))
  def permute(self, axis: Tuple[int, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].permute(axis), ))
  def stride(self, mul: Tuple[int, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].stride(mul), ))

  def reshape(self, new_shape: Tuple[sint, ...]) -> ShapeTracker:
    if (new_view := self.views[-1].reshape(new_shape)) is not None: return ShapeTracker(self.views[0:-1] + (new_view,))
    return ShapeTracker(self.views + (View.create(new_shape), ))

# returns the axes to create new_shape if new_shape can be created by combining axis from old_shape
# TODO: if we remove movementops from lazy.py we can delete this
def get_contraction(old_shape:Tuple[sint, ...], new_shape:Tuple[sint, ...]) -> Optional[List[List[int]]]:
  acc_old, acc_new = list(itertools.accumulate(old_shape, operator.mul)), list(itertools.accumulate(new_shape, operator.mul))
  try: split = [acc_old.index(acc)+1 if acc != 1 else 0 for acc in acc_new]
  except ValueError: return None
  return [list(range(st,ed)) for st,ed in zip([0]+split[:-1], split[:-1]+[len(old_shape)])]
