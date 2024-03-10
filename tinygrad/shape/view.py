from __future__ import annotations
import functools, operator, itertools
import math
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional, Dict, Set, cast
from tinygrad.helpers import prod, all_int, argsort
from tinygrad.shape.symbolic import MulNode, Node, NumNode, SumNode, Variable, sint

@functools.lru_cache(maxsize=None)
def canonicalize_strides(shape:Tuple[sint, ...], strides:Tuple[sint, ...]) -> Tuple[sint, ...]:
  return tuple(0 if s == 1 else st for s, st in zip(shape, strides))

@functools.lru_cache(maxsize=None)
def strides_for_shape(shape:Tuple[sint, ...]) -> Tuple[sint, ...]:
  if not shape: return ()
  strides = tuple(itertools.accumulate(reversed(shape[1:]), operator.mul, initial=1))
  return canonicalize_strides(shape, strides[::-1])

@functools.lru_cache(maxsize=None)
def _merge_dims(shape:Tuple[int, ...], strides:Tuple[int, ...], mask:Optional[Tuple[Tuple[int, int], ...]]=None) -> Tuple[Tuple[int, int, int], ...]:
  # merge contiguous subparts or zero strided dims. ret = List[(merged_dims, stride, merged dims w/o zero stride), ...]
  if not shape: return tuple()
  assert len(shape) == len(strides)
  ret = [(shape[0], strides[0], shape[0] if strides[0] else 0)]
  # wrt merging zero strided dimensions
  merging = strides[0] == 0 and (mask[0][1] - mask[0][0] == 1 if mask else shape[0] == 1)
  for i, (sh, st) in enumerate(zip(shape[1:], strides[1:]), start=1):
    if sh == 1: continue
    if merging or ret[-1][1] == sh * st: # mergeable
      ret[-1] = (ret[-1][0] * sh, st, (sh if merging else ret[-1][2] * sh) if st else 0)
    else: ret.append((sh, st, sh if st else 0)) # begin new
    # merging ends with either non-zero strided dim or zero strided dim with mask range > 1
    merging = st == 0 and (mask[i][1] - mask[i][0] == 1 if mask else sh == 1)
  return tuple(ret)

@functools.lru_cache(maxsize=None)
def _reshape_mask(view: View, new_shape:Tuple[sint, ...]) -> Tuple[Optional[Tuple[Tuple[sint, sint], ...]], bool]:
  if view.mask is None: return view.mask, False
  if any(not isinstance(m[0], int) or not isinstance(m[1], int) for m in view.mask): return view.mask, True
  new_mask: List[Tuple[int, int]] = []

  r_masks, r_shape, r_new_shape = reversed(view.mask), reversed(view.shape), reversed(new_shape)
  curr_stride, old_dim, new_dim, mask = 1, next(r_shape, 1), next(r_new_shape, 1), next(r_masks, (0,1))
  if mask[1] - mask[0] < 1: return ((0, 0),) * len(new_shape), False # invalid mask

  while len(new_mask) < len(new_shape):
    (l, r), next_stride = mask, new_dim * curr_stride

    if old_dim >= next_stride: # need to split mask.
      if old_dim == next_stride: # simply copy the mask and get next batch for merging
        new_mask.append((l // curr_stride, (r - 1) // curr_stride + 1))
        curr_stride, old_dim, new_dim, mask = 1, next(r_shape, 1), next(r_new_shape, 1), next(r_masks, (0,1))
        if mask[1] - mask[0] < 1: return ((0, 0),) * len(new_shape), False # invalid mask

      else: # mask can only be splitted if reshape doesn't cut across the mask.
        if ((l % next_stride != 0 or r % next_stride != 0) and l // next_stride != (r - 1) // next_stride): return view.mask, True
        new_mask.append((l % next_stride // curr_stride, (r - 1) % next_stride // curr_stride + 1))
        curr_stride, new_dim = next_stride,  next(r_new_shape, 1) # need to get mask for next dimension

    else:
      next_mask = next(r_masks, (0, 1))
      # combine if the mask can unfold continuously
      if mask != (0, old_dim) and next_mask[1] - next_mask[0] != 1: return view.mask, True
      mask, old_dim = (next_mask[0] * old_dim + l, (next_mask[1] - 1) * old_dim + r), old_dim * next(r_shape, 1)

  for mask in r_masks: # if the old shape has leading 1s, need to make sure their mask is (0,1)
    if mask != (0, 1): return ((0, 0),) * len(new_shape), False # invalid mask

  return tuple(reversed(new_mask)), False

@dataclass(frozen=True)
class View:
  shape:Tuple[sint, ...]
  strides:Tuple[sint, ...]
  offset:sint
  mask:Optional[Tuple[Tuple[sint, sint], ...]]
  contiguous:bool

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def size(self) -> int:
    # NOTE: Variable and the Node derived from it in symbolic shapes can only have int as max.
    ret = prod([x.max if isinstance(x, Node) else x for x in self.shape])
    assert isinstance(ret, int), f"{ret=} is not int"
    return ret

  def expr_view(self, idxs: List[Node], valid: Optional[Node] = None) -> Tuple[Node, Node]:
    assert len(idxs) == len(self.shape), f"need an idx for all dimensions {idxs} vs {self.shape}"
    iexpr: List[Node] = [NumNode(self.offset) if isinstance(self.offset, int) else self.offset]
    vexpr: List[Node] = [valid] if valid is not None else []
    for idx, sh, st, m in zip(idxs, self.shape, self.strides, self.mask if self.mask is not None else [None]*len(self.shape)):
      if sh != 1 and st != 0: iexpr.append(idx*st)
      if m is not None: vexpr += [idx >= m[0], idx < m[1]]
    return Node.sum(iexpr), Node.ands(vexpr)

  @staticmethod
  @functools.lru_cache(maxsize=None)
  def expr_idxs(views, idxs:Optional[Iterable[Node]]=None) -> Tuple[Node, Node]:
    shape = views[-1].shape
    idxs = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(shape)] if idxs is None else list(idxs)
    idx, valid = views[-1].expr_view(idxs)
    for view in reversed(views[0:-1]):
      if valid.max == 0: return NumNode(-1), valid
      view = view.minify()
      acc, idxs = 1, []
      for d in reversed(view.shape):
        idxs.append((idx//acc)%d)
        acc *= d
      idx, valid = view.expr_view(idxs[::-1], valid)
    return idx, valid

  @staticmethod
  @functools.lru_cache(maxsize=None)
  def calculate_real_strides(view1: View, view2: View, ignore_valid=False) -> Tuple[Optional[sint], ...]:
    shape = view2.shape
    views = (view1, view2) if view1 is not None else (view2,)
    idxs: Tuple[Variable, ...] = tuple(Variable(f"idx{i}", 0, s-1) for i,s in enumerate(shape))
    idx, valid = View.expr_idxs(views, idxs)
    ret: List[Optional[sint]] = [None] * len(shape)
    bad_idx_vars: Set[Variable] = set()
    for this_dim in (idx.nodes if isinstance(idx, SumNode) else [idx]):
      idx_maybe, stride_maybe = (this_dim.a, this_dim.b) if isinstance(this_dim, MulNode) else (this_dim, 1)
      idx_maybe_index = next((i for i, idx in enumerate(idxs) if idx == idx_maybe), None)
      try:
        if idx_maybe_index is None: raise TypeError
        else: ret[idx_maybe_index] = cast(sint, stride_maybe)
      except TypeError:
        bad_idx_vars = bad_idx_vars.union(idx_maybe.vars())
    idx_vars, valid_vars = idx.vars(), valid.vars()
    for i,tidx in enumerate(idxs):
      if tidx in bad_idx_vars or (tidx in valid_vars and not ignore_valid): ret[i] = None
      elif tidx not in idx_vars: ret[i] = 0
    return tuple(ret)

  @staticmethod
  @functools.lru_cache(maxsize=None)
  def un1d(shape:Tuple[sint, ...], offs:sint) -> List[sint]:
    strides = strides_for_shape(shape)
    result = []
    for stride in strides:
      here = offs // stride if stride else 0
      result.append(here)
      offs -= here * stride
    return result

  @functools.lru_cache(maxsize=None)
  #vm2 is old, self is old
  def __add__(self, other:View) -> Optional[View]:
    if self.contiguous: return other
    if other.contiguous and other.shape == self.shape: return self
    if other.contiguous and other.size() == self.size() and (ret := self.reshape(other.shape)) is not None: return ret
    if not self.mask and other.offset == 0 and None not in (rstrides := View.calculate_real_strides(self, other)):
      return View.create(other.shape, cast(Tuple[sint, ...], rstrides), self.offset, other.mask)
    if other.mask:
      for b,e in other.mask:
        if not (b < e): return View.create(other.shape, (0,) * len(other.shape), 0, ((0,0),) * len(other.shape))
      return (merged := (self + (other.shrink(other.mask)))) and merged.pad(tuple((b,s-e) for (b,e),s in zip(other.mask, other.shape)))

    # Project other's offset and strides on to self.
    origin = View.un1d(self.shape, other.offset)
    terms: List[List[Tuple[int, sint]]] = [[] for _ in origin]
    strides: List[sint] = [0] * len(other.shape)
    for d1, st in enumerate(other.strides):
      if st == 0: continue
      for d2, (o, s1) in enumerate(zip(origin, View.un1d(self.shape, other.offset + st))):
        if (s1 := s1 - o) == 0: continue
        terms[d2].append((d1, s1))
        strides[d1] += s1 * self.strides[d2]

    # Merge dimensions in self if required.
    # NB: Merging too many dimensions can make it difficult to project self's mask, hence only combining when required.
    idxs: List[Node] = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(other.shape)]
    merged_size, merged_term = 1, NumNode(0)
    extents: List[Tuple[sint, Node]] = []
    for term, s, o in zip(reversed(terms), reversed(self.shape), reversed(origin)):
      merged_term += Variable.sum([idxs[d1] * (s1 * merged_size) for d1, s1 in term]) + o * merged_size
      merged_size *= s
      if not (merged_term >= merged_size) and not (merged_term < 0):
        extents.append((merged_size, merged_term))
        merged_size, merged_term = 1, NumNode(0)
    if merged_term: return None
    if (self_shape := tuple(s for s,_ in reversed(extents))) != self.shape:
      return (reshaped_self := self.reshape(self_shape)) and (other + reshaped_self)

    if self.mask:
      # Try to project self's mask on to other.
      newb, newe, bad = [0] * len(other.shape), list(other.shape), False
      for d2, ((b, e), o, (_, t)) in enumerate(zip(self.mask, origin, reversed(extents))):
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

      # If any of other was masked off, try again with that mask in place.
      for b, e, s in zip(newb, newe, other.shape):
        if b != 0 or e != s:
          return (View.create(other.shape, other.strides, other.offset, tuple(zip(newb, newe))) + self)
      # selfwise if self's mask was violated, then cannot merge.
      if bad: return None

    return View.create(other.shape, tuple(strides), sum(o * s for o, s in zip(origin, self.strides)) + self.offset)

  @staticmethod
  @functools.lru_cache(maxsize=None)
  def create(shape:Tuple[sint, ...], strides:Optional[Tuple[sint, ...]]=None, offset:sint=0, mask:Optional[Tuple[Tuple[sint, sint], ...]]=None):
    strides = canonicalize_strides(shape, strides) if strides else strides_for_shape(shape)
    # canonicalize empty mask
    if mask is not None and all(m == (0,s) for m,s in zip(mask, shape)): mask = None
    contiguous = offset == 0 and mask is None and strides == strides_for_shape(shape)
    # if any dimension has size >1, but is masked such that only one index in the dimension is unmasked
    # then its stride can also be set to 0, albeit with a corresponding adjustment required to the offset
    # TODO: assert comparison with LtNode to avoid mis-using symbolic
    if mask and any(elim := [not (b+1 < e) for b,e in mask]):
      if any(not (b < e) for b,e in mask):
        strides, offset, mask = (0,) * len(shape), 0, ((0,0),) * len(shape)
      offset += sum((strides[i] * mask[i][0]) if e else 0 for i, e in enumerate(elim))
      strides = tuple(0 if e else st for st,e in zip(strides, elim))
    return View(shape, strides, offset, mask, contiguous)

  @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none
  def vars(self) -> Set[Variable]:
    flatten_mask = tuple(x for m in self.mask for x in m) if self.mask is not None else tuple()
    return functools.reduce(operator.or_, [x.vars() for x in self.shape+self.strides+(self.offset,)+flatten_mask if isinstance(x, Node)], set())

  @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none
  def unbind(self) -> Tuple[View, Dict[Variable, int]]:
    var_unboundvar_val = [(v, v.unbind()) for v in self.vars() if v.val is not None]
    unbound_vars = {v:uv for v,(uv,_) in var_unboundvar_val}
    new_shape = tuple([s if isinstance(s, int) else s.substitute(unbound_vars) for s in self.shape])
    new_strides = tuple([s if isinstance(s, int) else s.substitute(unbound_vars) for s in self.strides])
    new_offset = self.offset if isinstance(self.offset, int) else self.offset.substitute(unbound_vars)
    new_mask = tuple((a if isinstance(a, int) else a.substitute(unbound_vars),
                      b if isinstance(b, int) else b.substitute(unbound_vars)) for (a, b) in self.mask) if self.mask is not None else None
    return View.create(new_shape, new_strides, new_offset, new_mask), dict(x[1] for x in var_unboundvar_val)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def invert(self, out_shape:Tuple[sint, ...]) -> Optional[View]:
    ret = View.create(self.shape)
    if self.mask: ret = ret.shrink(self.mask)
    ret = ret.stride(tuple(-1 if x < 0 else 1 for x in self.strides)).permute(argsort(tuple(-x if x > 0 else x for x in self.strides)))
    return ret if prod(ret.shape) == prod(out_shape) else None   # don't support shrink, expand, or stride != (-1, 1)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def minify(self):
    min_shape = tuple(x[0] for x in _merge_dims(self.shape, self.strides, self.mask))
    return nv if (nv := self.reshape(min_shape)) else self

  def __unsafe_resize(self, arg: Tuple[Tuple[sint, sint], ...], mask=None) -> View:
    offset = sum([s * x[0] for s, x in zip(self.strides,arg)])
    if self.mask:
      # move the old mask
      nmask = tuple([(max(0, min(mx-ax,ay-ax)), max(0, min(my-ax,ay-ax))) for (mx,my),(ax,ay) in zip(self.mask, arg)])
      # merge the masks if we have two
      mask = tuple([(max(mx1, mx2), min(my1, my2)) for (mx1, my1), (mx2, my2) in zip(nmask, mask)]) if mask is not None else nmask
    shape = [y-x for x,y in arg]
    if mask is not None and all(m[0] == 0 and m[1] == s for m,s in zip(mask, shape)): mask = None
    return View.create(tuple(s.b if isinstance(s, NumNode) else s for s in shape), self.strides, self.offset+offset, mask)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def pad(self, arg: Tuple[Tuple[int, int], ...]) -> View:
    assert all((b>=0 and e>=0) for b,e in arg) and len(arg) == len(self.shape), f"{self.shape=}, {arg=}"
    if any(b or e for b, e in arg):
      zvarg = tuple([(-b,s+e) for s,(b,e) in zip(self.shape, arg)])
      mask = tuple([(b,s+b) for s,(b,_) in zip(self.shape, arg)])
      return self.__unsafe_resize(zvarg, mask=mask)
    return self

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def shrink(self, arg: Tuple[Tuple[sint, sint], ...]) -> View:
    assert all((0<=b<=e<=s) for s,(b,e) in zip(self.shape,arg)) and len(arg) == len(self.shape), f"invalid shrink {arg} for {self.shape}"
    return self.__unsafe_resize(arg)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def expand(self, new_shape: Tuple[sint, ...]) -> View:
    if len(new_shape) != len(self.shape): raise ValueError(f"expand arg {new_shape=} must have same number of dimensions as shape {self.shape=}")
    if 0 in self.shape:
      assert all((s == x == 0) or (s > 0 and (x % s) == 0) for s,x in zip(self.shape, new_shape)), f"can't expand {self.shape} into {new_shape}"
      return View.create(new_shape)
    assert all((s == x or (s == 1 and st == 0)) for s,x,st in zip(self.shape, new_shape, self.strides)), f"can't expand {self.shape} into {new_shape}"
    # NOTE: can the mask ever be (0,0)?
    mask = tuple([(((0,0) if m != (0,1) else (0,ns)) if s != ns else m) for m,s,ns in zip(self.mask, self.shape, new_shape)]) if self.mask else None
    return View.create(new_shape, self.strides, self.offset, mask)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def permute(self, axis: Tuple[int, ...]) -> View:
    assert all(isinstance(x, int) and x >= 0 and x < len(self.shape) for x in axis), f"invalid permute {axis} for {self.shape}"
    assert len(set(axis)) == len(axis) and len(axis) == len(self.shape), f"can't permute {self.shape} with {axis}"
    return View.create(tuple(self.shape[a] for a in axis), tuple(self.strides[a] for a in axis), self.offset,
                       tuple(self.mask[a] for a in axis) if self.mask is not None else None)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def stride(self, mul: Tuple[int, ...]) -> View:
    # except for the negative case, you can build this from the others. invertible in the negative case
    assert all(isinstance(x, int) and x != 0 for x in mul), f"invalid stride {mul} for {self.shape}"
    strides = tuple([z*m for z,m in zip(self.strides, mul)])
    new_shape = tuple([(s+(abs(m)-1))//abs(m) for s,m in zip(self.shape, mul)])
    offset = sum([(s-1)*z for s,z,m in zip(self.shape, self.strides, mul) if m < 0])
    mask = tuple([(((mx if m > 0 else s-my)+(abs(m)-1))//abs(m), ((my if m > 0 else s-mx)+(abs(m)-1))//abs(m)) \
                  for (mx,my),s,m in zip(self.mask, self.shape, mul)]) if self.mask is not None else None
    return View.create(new_shape, strides, self.offset + offset, mask)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def reshape(self, new_shape: Tuple[sint, ...]) -> Optional[View]:
    if self.shape == new_shape: return self

    assert all(x >= 0 for x in new_shape), f"shape can't contain negative numbers {new_shape}"
    if 0 in self.shape:
      assert 0 in new_shape, f"cannot reshape 0 size to {new_shape}"
      return View.create(new_shape)
    # check for the same size
    if all_int(self.shape):
      assert all(isinstance(s, (int, Variable)) for s in new_shape), f"{self.shape=} -> {new_shape=} contains non (int, Variable) dim"
      if prod(self.shape) != prod([s if isinstance(s, int) else cast(Variable,s).val for s in new_shape]):
        raise ValueError(f"size mismatched, can't reshape {self.shape=} -> {new_shape=}")

    if new_shape == () and self.mask and any(mx==my for (mx,my) in self.mask): return None

    # after the asserts, it's okay to check contiguous
    if self.contiguous: return View.create(new_shape)

    strides, r_new_shape = [], reversed(new_shape)
    for merged_dim, new_stride, real_dim in reversed(_merge_dims(self.shape, self.strides, self.mask)):
      acc = 1
      # TODO: this <= and != is for symbolic!?
      while acc <= merged_dim and acc != merged_dim and (new_dim := next(r_new_shape, None)):
        strides.append(new_stride)
        if new_dim != 1: new_stride *= (new_dim if (acc :=  acc * new_dim) < real_dim else 0)
      if acc != merged_dim: break
    else:
      strides += [0,] * (len(new_shape) - len(strides))
      new_mask, extra = _reshape_mask(self, new_shape)
      if not extra:
        new_strides = canonicalize_strides(tuple(e-b for b,e in new_mask) if new_mask else new_shape, tuple(reversed(strides)))
        extra_offset = (sum(m[0] * s for m,s in zip(self.mask, self.strides)) if self.mask else 0) - \
                       (sum(m[0] * s for m,s in zip(new_mask, new_strides)) if new_mask else 0)
        return View.create(new_shape, new_strides, self.offset + extra_offset, new_mask)

    return None
