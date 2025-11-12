# ruff: noqa: E501, A001, A002, A006
# A001 Variable `input` is shadowing a Python builtin
# A002 Function argument `input` is shadowing a Python builtin
# A006 Lambda argument `input` is shadowing a Python builtin
from __future__ import annotations
import functools, itertools, operator
from typing import Callable, Sequence, cast, List, Tuple
from dataclasses import dataclass
from tinygrad.helpers import merge_dicts, getenv, prod, all_int, flatten
from tinygrad.uop.symbolic import sym
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops, graph_rewrite, Variable, sint, sint_to_uop, Context, resolve, smax, smin, ssimplify
from enum import Enum, auto

@functools.cache
def canonicalize_strides(shape:tuple[sint, ...], strides:tuple[sint, ...]) -> tuple[sint, ...]:
  return tuple(0 if s == 1 else st for s, st in zip(shape, strides))

@functools.cache
def strides_for_shape(shape:tuple[sint, ...]) -> tuple[sint, ...]:
  if not shape: return ()
  import itertools, operator
  strides = tuple(itertools.accumulate(reversed(shape[1:]), operator.mul, initial=1))[::-1]
  return canonicalize_strides(shape, strides)

@functools.cache
def merge_dims(shape:tuple[int, ...], strides:tuple[int, ...], mask:tuple[tuple[int, int], ...]|None=None) -> tuple[tuple[int, int, int], ...]:
  # merge contiguous sub-parts or zero strided dims
  # any stride 0, masked from dim=1, or contiguous part is merged into next dim.
  # stride != 0 to stride == 0 starts a new merging block
  # ret = tuple[(merged_size, stride, merged size w/o zero stride), ...]
  if not shape: return ()
  assert len(shape) == len(strides) and (mask is None or len(shape) == len(mask))
  ret = [(shape[0], strides[0], shape[0] if strides[0] != 0 else 0)]
  # merge this dim to next dim if size is 1
  merging = (mask[0][1] - mask[0][0] == 1) if mask is not None else shape[0] == 1
  for i, (s, st) in enumerate(zip(shape[1:], strides[1:]), start=1):
    # always merge 1
    if s == 1: continue
    last_s, last_st, last_pre_expand_s = ret[-1]
    # merge last dim with this dim if merging or strides matched
    if merging or last_st == s * st: ret[-1] = (last_s * s, st, (s if merging else last_pre_expand_s * s))
    else: ret.append((s, st, s))
    # merge this dim to next dim if size is 1
    merging = (mask[i][1] - mask[i][0] == 1) if mask is not None else s == 1
  return tuple(ret)

@functools.cache
def _reshape_mask(_mask:tuple[tuple[sint, sint], ...]|None, old_shape:tuple[sint, ...], new_shape:tuple[sint, ...]) \
  -> tuple[tuple[sint, sint], ...]|None:
  """Returns the new mask if reshape is possible, and None if not possible."""
  if _mask is None: return tuple((0, s) for s in new_shape)
  if not all_int(flatten(_mask)): return None

  new_mask: list[tuple[int, int]] = []
  # _mask is all int here
  r_masks, r_shape, r_new_shape = reversed(cast(tuple[tuple[int, int], ...], _mask)), reversed(old_shape), reversed(new_shape)
  curr_stride, old_dim, new_dim, mask = 1, next(r_shape, 1), next(r_new_shape, 1), next(r_masks, (0,1))

  while len(new_mask) < len(new_shape):
    (l, r), next_stride = mask, ssimplify(new_dim * curr_stride)

    # need to split mask
    if old_dim == next_stride: # simply copy the mask and get next batch for merging
      new_mask.append((l // curr_stride, (r - 1) // curr_stride + 1))
      curr_stride, old_dim, new_dim, mask = 1, next(r_shape, 1), next(r_new_shape, 1), next(r_masks, (0,1))
    elif old_dim > next_stride: # mask can only be splitted if reshape doesn't cut across the mask.
      if old_dim % next_stride != 0: return None
      if (l % next_stride != 0 or r % next_stride != 0) and l // next_stride != (r - 1) // next_stride: return None
      new_mask.append((l % next_stride // curr_stride, (r - 1) % next_stride // curr_stride + 1))
      curr_stride, new_dim = next_stride,  next(r_new_shape, 1) # need to get mask for next dimension
    else:
      next_mask = next(r_masks, (0, 1))
      # combine if the mask can unfold continuously
      if mask != (0, old_dim) and l != r and next_mask[1] - next_mask[0] != 1: return None
      mask, old_dim = (next_mask[0] * old_dim + l, (next_mask[1] - 1) * old_dim + r), ssimplify(old_dim * next(r_shape, 1))

  return tuple(reversed(new_mask))

def unravel(shape:tuple[sint, ...], offset:sint) -> list[sint]:
  # find the position of offset on each dimension based on shape
  # similar to unravel_index in numpy/torch
  acc, idxs = 1, []
  for d in reversed(shape):
    idxs.append((offset//acc)%d)
    acc *= d
  return idxs[::-1]

@dataclass(frozen=True)
class View:
  shape:tuple[sint, ...]
  strides:tuple[sint, ...]
  offset:sint
  mask:tuple[tuple[sint, sint], ...]|None
  contiguous:bool

  def to_valid_uop(self, idxs:Sequence[UOp]|None=None) -> UOp:
    """valid.where(idx, INVALID)"""
    if idxs is None: idxs = [UOp.range(s, i) for i,s in enumerate(self.shape)]
    iexpr = sint_to_uop(self.offset)
    where = UOp.const(dtypes.bool, True)
    for idx,sh,st,m in zip(idxs, self.shape, self.strides, self.mask if self.mask is not None else itertools.repeat(None)):
      iexpr = iexpr + idx*sint_to_uop(st)
      if m is not None:
        if resolve(m[0] != 0): where &= (idx >= sint_to_uop(m[0]))
        if resolve(m[1] != sh): where &= (idx < sint_to_uop(m[1]))
    return where.where(iexpr, UOp.invalid())

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def size(self) -> int:
    ret = prod([x.vmax if isinstance(x, UOp) else x for x in self.shape])
    assert isinstance(ret, int), f"{ret=} is not int"
    return ret

  @staticmethod
  @functools.cache
  def create(shape:tuple[sint, ...], strides:tuple[sint, ...]|None=None, offset:sint=0, mask:tuple[tuple[sint, sint], ...]|None=None):
    # TODO: resolve shouldn't be needed here
    if not all(resolve(s >= 0) for s in shape): raise ValueError(f"Trying to create View with negative dimension: {shape=}")
    strides = canonicalize_strides(shape, strides) if strides else strides_for_shape(shape)
    # canonicalize 0 in shape
    if 0 in shape: return View(shape, (0,) * len(shape), offset=0, mask=None, contiguous=True)
    # canonicalize no-op mask
    if mask is not None and all(m == (0,s) for m,s in zip(mask, shape)): mask = None
    # if any dimension has size >1, but is masked such that only one index in the dimension is unmasked
    # then its stride can also be set to 0, albeit with a corresponding adjustment required to the offset
    if mask and any(elim := [not resolve(b+1 < e) for b,e in mask]):
      if any(not resolve(b < e) for b,e in mask):
        strides, offset, mask = (0,) * len(shape), 0, ((0,0),) * len(shape)
      offset += sum((strides[i] * mask[i][0]) if e else 0 for i, e in enumerate(elim))
      strides = tuple(0 if e else st for st,e in zip(strides, elim))
    # simplify as we go
    if isinstance(offset, UOp): offset = cast(sint, offset.ssimplify())
    shape = tuple(cast(sint, x.ssimplify()) if isinstance(x, UOp) else x for x in shape)
    # TODO: enabling stride simplification breaks symbolic jit
    """
    strides = tuple(x.ssimplify() if isinstance(x, UOp) else x for x in strides)
    if mask: mask = tuple((s.ssimplify() if isinstance(s, UOp) else s, e.ssimplify() if isinstance(e, UOp) else e) for s,e in mask)
    """
    contiguous = offset == 0 and mask is None and strides == strides_for_shape(shape)
    return View(shape, strides, offset, mask, contiguous)

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def vars(self) -> set[Variable]:
    flatten_mask = tuple(x for m in self.mask for x in m) if self.mask is not None else tuple()
    return functools.reduce(operator.or_, [x.vars() for x in self.shape+self.strides+(self.offset,)+flatten_mask if isinstance(x, UOp)], set())

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def unbind(self) -> tuple[View, dict[Variable, int]]:
    var_unboundvar_val = [(v, v.unbind()) for v in self.vars() if v.op is Ops.BIND]
    unbound_vars = {v:uv for v,(uv,_) in var_unboundvar_val}
    return self.substitute(unbound_vars), dict(x[1] for x in var_unboundvar_val)

  def substitute(self, dvars:dict[UOp, UOp]):
    def _substitute(x:sint): return x if isinstance(x, int) else x.substitute(dvars)
    new_shape = tuple(map(_substitute, self.shape))
    new_strides = tuple(map(_substitute, self.strides))
    new_offset = _substitute(self.offset)
    new_mask = tuple((_substitute(x[0]), _substitute(x[1])) for x in self.mask) if self.mask is not None else None
    return View.create(new_shape, new_strides, new_offset, new_mask)

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def __add__(self, vm1:View) -> View|None:
    vm2 = self
    if vm2.contiguous or vm1.size() == 0: return vm1
    if vm1.contiguous and vm1.shape == vm2.shape: return vm2
    if vm1.contiguous and vm1.size() == vm2.size() and (ret := vm2.reshape(vm1.shape)) is not None: return ret
    if vm1.mask:
      if (new_vm1 := vm1.shrink(vm1.mask)) == vm1 or (merged := vm2 + new_vm1) is None: return None
      return merged.pad(tuple((b,s-e) for (b,e),s in zip(vm1.mask, vm1.shape)))
    if not all_int(vm1.shape):
      # if all strides are 0 and vm2 is unmasked, return vm1
      if all(x == 0 for x in vm2.strides+vm1.strides) and vm2.mask is None: return vm1
      return None

    # Project vm1's offset and strides on to vm2.
    origin = [ssimplify(o) for o in unravel(vm2.shape, vm1.offset)]
    terms: list[list[tuple[int, sint]]] = [[] for _ in vm2.shape]
    strides: list[sint] = [0] * len(vm1.shape)
    for d1, st in enumerate(vm1.strides):
      if st == 0: continue
      for d2, (o, s1) in enumerate(zip(origin, unravel(vm2.shape, vm1.offset + st))):
        if not resolve((s1 := s1 - o)!=0): continue  # if s1 can possibly be 0
        terms[d2].append((d1, s1))
        strides[d1] += ssimplify(s1 * vm2.strides[d2])
    return None

  def __unsafe_resize(self, arg: tuple[tuple[sint, sint], ...], mask=None) -> View:
    offset = sum([s * x[0] for s, x in zip(self.strides,arg)])
    if self.mask:
      # move the old mask
      nmask = tuple([(smax(0, smin(mx-ax,ay-ax)), smax(0, smin(my-ax,ay-ax))) for (mx,my),(ax,ay) in zip(self.mask, arg)])
      # merge the masks if we have two
      mask = tuple([(smax(mx1, mx2), smin(my1, my2)) for (mx1, my1), (mx2, my2) in zip(nmask, mask)]) if mask is not None else nmask
    return View.create(tuple([y-x for x,y in arg]), self.strides, self.offset+offset, mask)

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def pad(self, arg: tuple[tuple[sint, sint], ...]) -> View:
    assert len(arg) == len(self.shape), f"invalid pad {arg} for {self.shape}"
    # NOTE: not checking for symbolic arg
    for b,e in arg: assert not all_int([b,e]) or b>=0 and e>=0, f"invalid pad {arg} for {self.shape}"
    if any(resolve(b!=0) or resolve(e!=0) for b, e in arg):
      zvarg = tuple([(-b,s+e) for s,(b,e) in zip(self.shape, arg)])
      mask = tuple([(b,s+b) for s,(b,_) in zip(self.shape, arg)])
      return self.__unsafe_resize(zvarg, mask=mask)
    return self

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def shrink(self, arg: tuple[tuple[sint, sint], ...]) -> View:
    assert len(arg) == len(self.shape), f"invalid shrink {arg} for {self.shape}"
    # NOTE: not checking for symbolic arg
    for s,(b,e) in zip(self.shape,arg): assert not all_int([s,b,e]) or (0<=b<=e<=s), f"invalid shrink {arg} for {self.shape}"
    return self.__unsafe_resize(arg)
  @functools.cache  # pylint: disable=method-cache-max-size-none
  def slice(self, dim:int, start:int, end:int, step:int) -> View:
    assert 0 <= dim < len(self.shape), f"invalid dim {dim} for shape {self.shape}"
    assert step > 0, "slice step must be positive"
    size = self.shape[dim]
    start = max(0, min(size, start))
    end = max(0, min(size, end))
    if end < start: end = start
    new_extent = 0 if end <= start else ssimplify((end-start + step-1)//step)
    new_shape = list(self.shape)
    new_shape[dim] = new_extent
    new_strides = list(self.strides)
    new_strides[dim] = new_strides[dim] * step
    new_offset = self.offset + start * self.strides[dim]
    if self.mask is None: base_mask = tuple((0,s) for s in self.shape)
    else: base_mask = self.mask
    new_mask = list(base_mask)
    new_mask[dim] = (0, new_extent)
    return View.create(tuple(new_shape), tuple(new_strides), new_offset, tuple(new_mask))

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def expand(self, new_shape: tuple[sint, ...]) -> View:
    if len(new_shape) != len(self.shape): raise ValueError(f"expand arg {new_shape=} must have same number of dimensions as shape {self.shape=}")
    # NOTE: does not check multiple of symbolic shape
    assert all(resolve(s == ns) or s == 1 for s,ns in zip(self.shape, new_shape)), f"can't expand {self.shape} into {new_shape}"
    if 0 in self.shape: return View.create(new_shape)
    # TODO: resolve may not be needed, but it's hard because vars need to be canonicalized
    mask = tuple([(((0,0) if m != (0,1) else (0,ns)) if resolve(s != ns) and resolve(s == 1, False) else m) \
                  for m,s,ns in zip(self.mask, self.shape, new_shape)]) if self.mask else None
    return View.create(new_shape, self.strides, self.offset, mask)

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def permute(self, axis: tuple[int, ...]) -> View:
    assert sorted(axis) == list(range(len(self.shape))), f"invalid permutation {axis} of len {len(self.shape)}"
    return View.create(tuple(self.shape[a] for a in axis), tuple(self.strides[a] for a in axis), self.offset,
                       tuple(self.mask[a] for a in axis) if self.mask is not None else None)

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def flip(self, arg: tuple[bool, ...]) -> View:
    offset = sum((s-1)*z for s,z,f in zip(self.shape, self.strides, arg) if f)
    mask = tuple((s-my,s-mx) if f else (mx,my) for (mx,my),s,f in zip(self.mask, self.shape, arg)) if self.mask is not None else None
    return View.create(self.shape, tuple(-z if f else z for z,f in zip(self.strides, arg)), self.offset+offset, mask)

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def reshape(self, new_shape: tuple[sint, ...]) -> View|None:
    if self.shape == new_shape: return self

    if not all(x >= 0 for x in new_shape): raise ValueError(f"shape can't contain negative numbers {new_shape}")
    # check for the same size
    if resolve(prod(self.shape) != prod(new_shape), True): raise ValueError(f"size mismatched, can't reshape {self.shape=} -> {new_shape=}")

    if 0 in self.shape: return View.create(new_shape)
    if new_shape == () and self.mask and any(mx==my for (mx,my) in self.mask): return None

    # after the asserts, it's okay to check contiguous
    if self.contiguous: return View.create(new_shape)

    r_strides, r_new_shape = [], reversed(new_shape)
    for merged_size, new_stride, real_size in reversed(merge_dims(self.shape, self.strides, self.mask)):
      acc = 1
      # TODO: third resolve shouldn't be needed
      while resolve(acc <= merged_size) and resolve(acc != merged_size) and resolve((new_dim := next(r_new_shape, 0)) > 0):
        r_strides.append(new_stride * acc)
        acc = acc * new_dim
        if not resolve(acc < real_size): new_stride = 0
      if resolve(acc != merged_size): return None
    new_strides = (0,) * (len(new_shape) - len(r_strides)) + tuple(r_strides[::-1])

    if (new_mask:=_reshape_mask(self.mask, self.shape, new_shape)) is not None:
      extra_offset = (sum(m[0] * s for m,s in zip(self.mask, self.strides)) if self.mask else 0) - \
                     (sum(m[0] * s for m,s in zip(new_mask, new_strides)))
      return View.create(new_shape, new_strides, self.offset + extra_offset, new_mask)

    return None



@functools.cache
def get_buffer_size(shape, strides, offset, mask):
  real_shape = tuple(y-x for x,y in mask) if mask else shape
  offset = offset + sum(stp * (s-1) for s,stp in zip(real_shape, strides) if stp < 0)
  real_offset = offset + (sum(x*stp for (x,_),stp in zip(mask, strides)) if mask else 0)
  real_real_shape = [s for s,stp in zip(real_shape, strides) if stp]
  strides = [abs(stp) if isinstance(stp, int) else stp for stp in strides if stp]
  return real_offset + sum((s-1)*stp for s, stp in zip(real_real_shape, strides)) + 1

class MovementOps(Enum): RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); STRIDE = auto()  # noqa: E702

def _apply_st_mop(st: ShapeTracker, mop_arg: Tuple[MovementOps, Tuple]) -> ShapeTracker:
  mop, arg = mop_arg
  if mop is MovementOps.RESHAPE:
    if arg == (-1,): return st.reshape((prod(st.shape),))
    return st.reshape(arg)
  if mop is MovementOps.PERMUTE: return st.permute(arg)
  if mop is MovementOps.EXPAND:
    if len(arg) != len(st.shape): st = st.reshape((1, *st.shape))
    return st.expand(arg)
  if mop is MovementOps.PAD: return st.pad(arg)
  if mop is MovementOps.SHRINK: return st.shrink(arg)
  if mop is MovementOps.STRIDE:
    assert all(x in (-1, 1) for x in arg)
    return st.flip(tuple(i for i,x in enumerate(arg) if x == -1))
  raise ValueError(f"invalid movement op {mop}")

def _make_scratch_st(st: ShapeTracker) -> ShapeTracker:
  first = st.views[0]
  return ShapeTracker.from_shape(( get_buffer_size(first.shape, first.strides, first.offset, first.mask), ))

def to_movement_ops(st: ShapeTracker) -> List[Tuple[MovementOps, Tuple]]:
  to_apply: List[Tuple[MovementOps, Tuple]] = []
  for i, v in enumerate(st.views):
    real_shape = tuple(y-x for x,y in v.mask) if v.mask else v.shape
    offset = (v.offset or 0) + sum(stp*(s-1) for s,stp in zip(real_shape, v.strides) if stp < 0)
    real_offset = offset + (sum(x*stp for (x,_),stp in zip(v.mask, v.strides)) if v.mask else 0)
    real_real_shape = [s for s,stp in zip(real_shape, v.strides) if stp]
    strides: List[int] = [abs(stp) if isinstance(stp, int) else stp for stp in v.strides if stp]
    buffer_size = sum((s-1)*stp for s,stp in zip(real_real_shape, strides)) + 1
    if i: buffer_size = prod(st.views[i-1].shape) - real_offset if real_shape else 1
    def sort_by_strides(shape, strides):
      paired = sorted(zip(shape, strides), key=lambda k: (k[1], -k[0]), reverse=True)
      order = sorted(range(len(strides)), key=lambda k: (strides[k], -real_real_shape[k]), reverse=True)
      return paired, order
    ordered_shape_strides, order = sort_by_strides(real_real_shape, strides)
    to_apply.extend([(MovementOps.RESHAPE, (-1,)), (MovementOps.SHRINK, ((real_offset, real_offset+buffer_size),))])
    if strides:
      if (ordered_shape_strides[0][0]*ordered_shape_strides[0][1]) - buffer_size > 0:
        to_apply.append((MovementOps.PAD, ((0, (ordered_shape_strides[0][0] * ordered_shape_strides[0][1]) - buffer_size),)))
      for j, shape_stride in enumerate(ordered_shape_strides):
        if j < len(ordered_shape_strides)-1 and shape_stride[1] < ordered_shape_strides[j+1][0]*ordered_shape_strides[j+1][1]:
          remaining_buffer = ordered_shape_strides[j-1][1] if j > 0 else buffer_size
          to_apply.append((MovementOps.EXPAND, (shape_stride[0], *(s[0] for s in ordered_shape_strides[:j]), remaining_buffer)))
          to_apply.append((MovementOps.PERMUTE, (*range(1, j+1), 0, j+1)))
          to_apply.append((MovementOps.RESHAPE, (*(s[0] for s in ordered_shape_strides[:j]), shape_stride[0]*remaining_buffer)))
          to_apply.append((MovementOps.PAD, (*((0,0) for _ in range(j)), (0, shape_stride[0]*shape_stride[1]))))
          to_apply.append((MovementOps.RESHAPE, (*(s[0] for s in ordered_shape_strides[:j+1]), remaining_buffer+shape_stride[1])))
          ordered_shape_strides[j] = (ordered_shape_strides[j][0], remaining_buffer+shape_stride[1])
        else:
          to_apply.append((MovementOps.SHRINK, (*((0, s[0]) for s in ordered_shape_strides[:j]), (0, shape_stride[0]*shape_stride[1]))))
          to_apply.append((MovementOps.RESHAPE, (*[s[0] for s in ordered_shape_strides[:j+1]], shape_stride[1])))
      to_apply.extend([(MovementOps.SHRINK, (*[(0, s[0]) for s in ordered_shape_strides], (0,1))), (MovementOps.RESHAPE, tuple(s[0] for s in ordered_shape_strides))])
      if order != list(range(len(order))): to_apply.append((MovementOps.PERMUTE, tuple(order.index(i) for i in range(len(strides)))))
    to_apply.append((MovementOps.RESHAPE, tuple(s if stp else 1 for s,stp in zip(real_shape, v.strides))))
    if any(stp < 0 for stp in v.strides): to_apply.append((MovementOps.STRIDE, tuple(-1 if stp < 0 else 1 for stp in v.strides)))
    if v.mask is not None:
      pre_expand = tuple((x,s-y) if stp != 0 else (0,0) for (x,y),s,stp in zip(v.mask, v.shape, v.strides))
      post_expand = tuple((x,s-y) if stp == 0 else (0,0) for (x,y),s,stp in zip(v.mask, v.shape, v.strides))
      if any(x != (0,0) for x in pre_expand):
        to_apply.append((MovementOps.PAD, pre_expand))
        real_shape = tuple(x+s[0]+s[1] for x,s in zip(real_shape, pre_expand))
    if any(s != 1 and stp == 0 for s,stp in zip(real_shape, v.strides)): to_apply.append((MovementOps.EXPAND, real_shape))
    if v.mask is not None:
      post_expand = tuple((x,s-y) if stp == 0 else (0,0) for (x,y),s,stp in zip(v.mask, v.shape, v.strides))
      if any(x != (0,0) for x in post_expand): to_apply.append((MovementOps.PAD, post_expand))

  scratch_st = _make_scratch_st(st)
  ret: List[Tuple[MovementOps, Tuple]] = []
  seen: dict[ShapeTracker, List[Tuple[MovementOps, Tuple]]] = {}
  for mop_arg in to_apply:
    scratch_st = _apply_st_mop(scratch_st, mop_arg)
    if scratch_st in seen:
      ret = seen[scratch_st][:]
    else:
      if ret and ret[-1][0] is MovementOps.RESHAPE and mop_arg[0] is MovementOps.RESHAPE:
        ret[-1] = mop_arg
      else:
        if mop_arg == (MovementOps.RESHAPE, -1): mop_arg = (MovementOps.RESHAPE, (prod(st.shape),))
        ret.append(mop_arg)
      seen[scratch_st] = ret[:]
  return ret

@functools.cache
def views_to_valid_uop(views: tuple[View, ...], _idxs:tuple[UOp, ...]|None=None) -> UOp:
  import itertools
  idx = views[-1].to_valid_uop(_idxs)
  for view in reversed(views[0:-1]):
    idx = view.to_valid_uop([sint_to_uop(i) for i in unravel(view.shape, idx)])
  with Context(TRACK_MATCH_STATS=0):
    return graph_rewrite(idx, sym, name="indexing sym @ 1")

@functools.cache
def views_to_is_expanded(views: tuple[View, ...]) -> tuple[bool, ...]:
  # NOTE: return if each dim is expanded
  if len(views) == 1 and views[-1].mask is None: return tuple([bool(st==0) for st in views[-1].strides])
  idx = views_to_valid_uop(views).get_idx()
  used_ranges = [x.arg[0] for x in idx.toposort() if x.op is Ops.RANGE]
  return tuple([i not in used_ranges for i in range(len(views[-1].shape))])

@dataclass(frozen=True, order=True)
class ShapeTracker:
  views: tuple[View, ...]

  def __add__(self, st:ShapeTracker) -> ShapeTracker:
    ret = self
    for v in st.views: ret = ShapeTracker(ret.views + (v,)).simplify() # one view at a time = better simplification
    return ret

  @staticmethod
  def from_shape(shape:tuple[sint, ...], strides:tuple[sint, ...]|None=None) -> ShapeTracker: return ShapeTracker((View.create(shape, strides),))

  @property
  def contiguous(self) -> bool: return len(self.views) == 1 and self.views[0].contiguous

  @property
  def shape(self) -> tuple[sint, ...]: return self.views[-1].shape

  @property
  def size(self) -> int: return self.views[-1].size()

  def vars(self) -> set[Variable]: return set().union(*[v.vars() for v in self.views])

  @property
  def var_vals(self) -> dict[str, int]: return merge_dicts([{(vu:=v.unbind())[0].expr:vu[1]} for v in self.vars()])

  def unbind(self) -> tuple[ShapeTracker, dict[Variable, int]]:
    unbound_views, var_vals = zip(*[v.unbind() for v in self.views])
    if all(len(x) == 0 for x in var_vals): return self, {}
    return ShapeTracker(tuple(unbound_views)), merge_dicts(var_vals)

  def simplify(self) -> ShapeTracker:
    if len(self.views) >= 2 and (new_view := self.views[-2] + self.views[-1]) is not None:
      return ShapeTracker(self.views[:-2] + (new_view,)).simplify()
    return self

  def _with_last_view(self, fn: Callable[[View], View]) -> ShapeTracker:
    return ShapeTracker(self.views[:-1] + (fn(self.views[-1]), ))

  # *** under this line are the movement ops ***

  def pad(self, arg: tuple[tuple[sint, sint], ...]) -> ShapeTracker: return self._with_last_view(lambda v: v.pad(arg))
  def shrink(self, arg: tuple[tuple[sint, sint], ...]) -> ShapeTracker: return self._with_last_view(lambda v: v.shrink(arg))
  def expand(self, new_shape: tuple[sint, ...]) -> ShapeTracker: return self._with_last_view(lambda v: v.expand(new_shape))
  def permute(self, axis: tuple[int, ...]) -> ShapeTracker: return self._with_last_view(lambda v: v.permute(axis))
  def flip(self, mul: tuple[int, ...]) -> ShapeTracker: return self._with_last_view(lambda v: v.flip(mul))
  def slice(self, dim:int, start:int, end:int, step:int) -> ShapeTracker: return self._with_last_view(lambda v: v.slice(dim, start, end, step))

  def reshape(self, new_shape: tuple[sint, ...]) -> ShapeTracker:
    if getenv("MERGE_VIEW", 1) and (new_view := self.views[-1].reshape(new_shape)) is not None: return ShapeTracker(self.views[0:-1] + (new_view,))
    return ShapeTracker(self.views + (View.create(new_shape), ))
