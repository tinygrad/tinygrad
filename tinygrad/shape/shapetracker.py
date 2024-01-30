# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
import functools, math
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Set, cast, Iterable, Union
from tinygrad.helpers import merge_dicts, getenv
from tinygrad.shape.symbolic import Variable, MulNode, Node, SumNode, NumNode, sint
from tinygrad.shape.view import View, strides_for_shape

def un1d(shape:Tuple[sint, ...], offs:sint) -> List[sint]:
  strides = strides_for_shape(shape)
  result = []
  for stride in strides:
    here = offs // stride if stride else 0
    result.append(here)
    offs -= here * stride
  return result

@functools.lru_cache(maxsize=None)
def merge_views(vm2:View, vm1:View) -> Optional[View]:
  if vm1.contiguous and vm1.shape == vm2.shape: return vm2
  if vm2.contiguous: return vm1
  if not vm2.mask and vm1.offset == 0 and None not in (rstrides := ShapeTracker((vm2, vm1)).real_strides()):
    return View.create(vm1.shape, cast(Tuple[sint, ...], rstrides), vm2.offset, vm1.mask)
  if vm1.mask:
    for b,e in vm1.mask:
      if not (b < e): return View.create(vm1.shape, (0,) * len(vm1.shape), 0, ((0,0),) * len(vm1.shape))
    return (merged := merge_views(vm2, vm1.shrink(vm1.mask))) and merged.pad(tuple((b,s-e) for (b,e),s in zip(vm1.mask, vm1.shape)))

  # Project vm1's offset and strides on to vm2.
  origin = un1d(vm2.shape, vm1.offset)
  terms: List[List[Tuple[int, sint]]] = [[] for _ in origin]
  strides: List[sint] = [0] * len(vm1.shape)
  for d1, st in enumerate(vm1.strides):
    if st == 0: continue
    for d2, (o, s1) in enumerate(zip(origin, un1d(vm2.shape, vm1.offset + st))):
      if (s1 := s1 - o) == 0: continue
      terms[d2].append((d1, s1))
      strides[d1] += s1 * vm2.strides[d2]

  # Merge dimensions in vm2 if required.
  # NB: Merging too many dimensions can make it difficult to project vm2's mask, hence only combining when required.
  idxs: List[Node] = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(vm1.shape)]
  merged_size, merged_term = 1, NumNode(0)
  extents: List[Tuple[sint, Node]] = []
  for term, s, o in zip(reversed(terms), reversed(vm2.shape), reversed(origin)):
    merged_term += Variable.sum([idxs[d1] * (s1 * merged_size) for d1, s1 in term]) + o * merged_size
    merged_size *= s
    if not (merged_term >= merged_size) and not (merged_term < 0):
      extents.append((merged_size, merged_term))
      merged_size, merged_term = 1, NumNode(0)
  if merged_term: return None
  if (vm2_shape := tuple(s for s,_ in reversed(extents))) != vm2.shape:
    return (reshaped_vm2 := vm2.reshape(vm2_shape)) and merge_views(reshaped_vm2, vm1)

  if vm2.mask:
    # Try to project vm2's mask on to vm1.
    newb, newe, bad = [0] * len(vm1.shape), list(vm1.shape), False
    for d2, ((b, e), o, (_, t)) in enumerate(zip(vm2.mask, origin, reversed(extents))):
      if not (t.min < b or t.max >= e): continue
      if not isinstance(o, int) or not isinstance(b, int) or not isinstance(e, int):
        bad = True
        continue
      term = terms[d2]
      if len(term) != 1:
        if not term and newe: newe[0] = 0
        else: bad = True
        continue
      d1, s1 = term[0]
      if not isinstance(s1, int) or not isinstance(newe[d1], int):
        bad = True
        continue
      newb[d1] = max(newb[d1], math.ceil((b - o if s1 > 0 else e - o - 1) / s1))
      newe[d1] = min(newe[d1], (b - o if s1 < 0 else e - o - 1) // s1 + 1)

    # If any of vm1 was masked off, try again with that mask in place.
    for b, e, s in zip(newb, newe, vm1.shape):
      if b != 0 or e != s:
        return merge_views(vm2, View.create(vm1.shape, vm1.strides, vm1.offset, tuple(zip(newb, newe))))
    # Otherwise if vm2's mask was violated, then cannot merge.
    if bad: return None

  return View.create(vm1.shape, tuple(strides), sum(o * s for o, s in zip(origin, vm2.strides)) + vm2.offset)

def _expr_view(view:View, idxs:List[Node], valid:Optional[Node]=None) -> Tuple[Node, Node]:
  assert len(idxs) == len(view.shape), f"need an idx for all dimensions {idxs} vs {view.shape}"
  iexpr: List[Node] = [NumNode(view.offset) if isinstance(view.offset, int) else view.offset]
  vexpr: List[Node] = [valid] if valid is not None else []
  for idx,sh,st,m in zip(idxs, view.shape, view.strides, view.mask if view.mask is not None else [None]*len(view.shape)):
    if sh != 1 and st != 0: iexpr.append(idx*st)
    if m is not None: vexpr += [idx >= m[0], idx < m[1]]
  return Node.sum(iexpr), Node.ands(vexpr)

@dataclass(frozen=True)
class ShapeTracker:
  views: Tuple[View, ...]

  def __add__(self, st:ShapeTracker) -> ShapeTracker:
    ret = self
    for v in st.views: ret = ShapeTracker(ret.views + (v,)).simplify() # one view at a time = better simplification
    return ret

  def invert(self, out_shape:Tuple[sint, ...]) -> Optional[ShapeTracker]:
    ret = tuple(v.invert(s) for v,s in zip(self.views[::-1], [x.shape for x in self.views[::-1][1:]]+[out_shape]))
    return ShapeTracker(cast(Tuple[View, ...], ret)).reshape(out_shape) if all(x is not None for x in ret) else None

  @staticmethod
  def from_shape(shape:Tuple[sint, ...]): return ShapeTracker((View.create(shape),))

  @property
  def contiguous(self) -> bool: return len(self.views) == 1 and self.views[0].contiguous

  @property
  def shape(self) -> Tuple[sint, ...]: return self.views[-1].shape

  @property
  def size(self) -> int: return self.views[-1].size()

  def real_size(self) -> int:
    if 0 in self.shape: return 0
    ret = cast(Union[int, Node], self.expr_idxs()[0].max)   # TODO: this is due to typing issues in symbolic!
    while not isinstance(ret, int): ret = ret.max    # TODO: this is a while loop?!? it should be more clear what max does
    assert isinstance(ret, int), f"ret must be integer, {ret=} isn't"
    return ret+1

  def vars(self) -> Set[Variable]: return set.union(*[v.vars() for v in self.views], set())

  @property
  def var_vals(self) -> Dict[Variable, int]: return merge_dicts([dict([v.unbind()]) for v in self.vars()])

  def unbind(self) -> Tuple[ShapeTracker, Dict[Variable, int]]:
    unbound_views, var_vals = zip(*[v.unbind() for v in self.views])
    return ShapeTracker(tuple(unbound_views)), merge_dicts(var_vals)

  # NOTE: if a stride is not always valid, it will be None
  def real_strides(self, ignore_valid=False) -> Tuple[Optional[sint], ...]:
    if len(self.views) == 1 and self.views[-1].mask is None: return self.views[-1].strides
    idxs: List[Node] = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(self.shape)]
    idx, valid = self.expr_idxs(idxs)
    ret: List[Optional[sint]] = [None] * len(self.views[-1].shape)
    bad_idx_vars: Set[Variable] = set()
    for this_dim in (idx.nodes if isinstance(idx, SumNode) else [idx]):
      idx_maybe, stride_maybe = (this_dim.a, this_dim.b) if isinstance(this_dim, MulNode) else (this_dim, 1)
      try: ret[idxs.index(idx_maybe)] = stride_maybe
      except ValueError: bad_idx_vars = bad_idx_vars.union(idx_maybe.vars())
    idx_vars, valid_vars = idx.vars(), valid.vars()
    for i,tidx in enumerate(idxs):
      if tidx in bad_idx_vars or (tidx in valid_vars and not ignore_valid): ret[i] = None
      elif tidx not in idx_vars: ret[i] = 0
    return tuple(ret)

  def unit_stride_axes(self, ignore_valid=False) -> List[int]: return [i for i,st in enumerate(self.real_strides(ignore_valid)) if st == 1]

  def expr_idxs(self, idxs:Optional[Iterable[Node]]=None) -> Tuple[Node, Node]:
    idxs = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(self.shape)] if idxs is None else list(idxs)
    idx, valid = _expr_view(self.views[-1], idxs)
    for view in reversed(self.views[0:-1]):
      if valid.max == 0: return NumNode(-1), valid
      view = view.minify()
      acc, idxs = 1, []
      for d in reversed(view.shape):
        idxs.append((idx//acc)%d)
        acc *= d
      idx, valid = _expr_view(view, idxs[::-1], valid)
    return idx, valid

  def axis_is_masked(self, axis:int) -> bool:
    _, valid = self.expr_idxs()
    return f'idx{axis}' in [v.expr for v in valid.vars()]

  def simplify(self) -> ShapeTracker:
    if len(self.views) >= 2 and (new_view := merge_views(self.views[-2], self.views[-1])) is not None:
      return ShapeTracker(self.views[:-2] + (new_view,)).simplify()
    return self

  # *** under this line are the movement ops ***

  def pad(self, arg: Tuple[Tuple[sint, sint], ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].pad(arg), ))
  def shrink(self, arg: Tuple[Tuple[sint, sint], ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].shrink(arg), ))
  def expand(self, new_shape: Tuple[sint, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].expand(new_shape), ))
  def permute(self, axis: Tuple[int, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].permute(axis), ))
  def stride(self, mul: Tuple[int, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].stride(mul), ))

  def reshape(self, new_shape: Tuple[sint, ...]) -> ShapeTracker:
    if getenv("MERGE_VIEW", 1) and (new_view := self.views[-1].reshape(new_shape)) is not None: return ShapeTracker(self.views[0:-1] + (new_view,))
    return ShapeTracker(self.views + (View.create(new_shape), ))
