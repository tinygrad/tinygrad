# ruff: noqa: E501, A001, A002, A006
# A001 Variable `input` is shadowing a Python builtin
# A002 Function argument `input` is shadowing a Python builtin
# A006 Lambda argument `input` is shadowing a Python builtin
from __future__ import annotations
import functools, weakref
from typing import Callable, Sequence, cast
from dataclasses import dataclass
from tinygrad.helpers import merge_dicts, getenv, prod, all_int, flatten
from tinygrad.uop.symbolic import sym
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops, graph_rewrite, Variable, sint, sint_to_uop, Context, resolve, smax, smin, ssimplify

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

from tinygrad import Tensor, dtypes, Device
from tinygrad.uop.ops import Ops
from tinygrad.helpers import getenv, prod
from enum import Enum, auto
import os
TORCH_DEBUG = getenv("TORCH_DEBUG")
import torch, pathlib, math, operator, functools, inspect
from typing import List, Tuple
torch.autograd.grad_mode.set_multithreading_enabled(False)
from tinygrad.dtype import _from_torch_dtype, _to_torch_dtype
_ext_dir = os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(pathlib.Path(__file__).resolve().parents[2] / ".torch_extensions"))
pathlib.Path(_ext_dir).mkdir(parents=True, exist_ok=True)

# https://pytorch.org/docs/stable/torch.compiler_ir.html

def _from_torch_device(device: torch.device): return f"{Device.DEFAULT}:{device.index or 0}"
def _to_torch_device(device: str): return torch.device("tiny", int(device.partition(":")[2] or 0))

import torch.utils.cpp_extension
mod = torch.utils.cpp_extension.load(name="custom_device_extension", sources=[str(pathlib.Path(__file__).parent / "wrapped_tensor.cpp")])
def wrap(x:Tensor) -> torch.Tensor:
  _ensure_view_tracking(x)
  return mod.wrap(x, _to_torch_dtype(x.dtype), _to_torch_device(x.device).index)
def unwrap(x:torch.Tensor) -> Tensor:
  assert isinstance(x, torch.Tensor), f"x isn't {type(x)}"
  return mod.unwrap(x)
class TinyBackend:
  def is_initialized(self): return True
  def is_available(self): return True
  def current_device(self): return 0
  def _is_in_bad_fork(self): return False
  def manual_seed_all(self, seed: int): Tensor.manual_seed(seed)
  def device_count(self): return getenv("GPUS", 1) # TODO: device count in tiny?
torch.utils.rename_privateuse1_backend("tiny")
torch._register_device_module("tiny", TinyBackend())
torch.utils.generate_methods_for_privateuse1_backend()
aten = torch.ops.aten

# track view relationships for in place operations
def is_view(tensor: Tensor): return hasattr(tensor, "_view_base")
def canonical_base(view: Tensor):
  base = getattr(view, "_view_base", view)
  return _ensure_view_tracking(base)
def derived_views(base: Tensor): return [t for tref in getattr(base, "_views", set()) if (t:=tref()) is not None]
def _ensure_view_tracking(tensor: Tensor) -> Tensor:
  if not hasattr(tensor, "_view_ops"): tensor._view_ops = tuple()
  if not hasattr(tensor, "_view_st"): tensor._view_st = ShapeTracker.from_shape(tuple(tensor.shape))
  elif tensor._view_st.shape != tuple(tensor.shape):
    tensor._view_ops = tuple()
    tensor._view_st = ShapeTracker.from_shape(tuple(tensor.shape))
  return tensor
def _get_view_ops(tensor: Tensor) -> tuple:
  return _ensure_view_tracking(tensor)._view_ops
def _get_view_st(tensor: Tensor) -> ShapeTracker:
  return _ensure_view_tracking(tensor)._view_st

def _apply_ops(target, ops: tuple, handlers: dict[str, Callable]):
  ret = target
  for op, args in ops:
    handler = handlers.get(op)
    if handler is None: raise ValueError(f"unknown view op {op}")
    ret = handler(ret, args)
  return ret

def _expand_st(st: ShapeTracker, shape: tuple[int, ...]) -> ShapeTracker:
  cur_shape = st.shape
  if len(shape) != len(cur_shape):
    added = (1,) * (len(shape) - len(cur_shape))
    st = st.reshape(added + cur_shape)
  return st.expand(shape)

def _flip_axes(obj, flips):
  axes = tuple(i for i, f in enumerate(flips) if f)
  return obj.flip(axes) if axes else obj

def _slice_view(st: ShapeTracker, args):
  dim, start, end, step = args
  return st.slice(dim, start, end, step)

def _slice_tensor(tensor: Tensor, dim=0, start=None, end=None, step=1):
  slices = [slice(None)] * tensor.ndim
  slices[dim] = slice(start, end, step)
  return tensor[tuple(slices)]

_ST_VIEW_HANDLERS = {
  "reshape": lambda st, shape: st.reshape(shape),
  "expand": _expand_st,
  "permute": lambda st, axis: st.permute(axis),
  "pad": lambda st, arg: st.pad(arg),
  "shrink": lambda st, arg: st.shrink(arg),
  "flip": _flip_axes,
  "slice": _slice_view,
  "bitcast": lambda st, _: st,
  "detach": lambda st, _: st,
}

_TENSOR_VIEW_HANDLERS = {
  "reshape": lambda t, shape: t.reshape(shape),
  "expand": lambda t, shape: t.expand(shape),
  "permute": lambda t, axis: t.permute(axis),
  "pad": lambda t, arg: t.pad(arg),
  "shrink": lambda t, arg: t.shrink(arg),
  "flip": _flip_axes,
  "slice": lambda t, args: _slice_tensor(t, *args),
  "bitcast": lambda t, dtype: t.bitcast(dtype),
  "detach": lambda t, _: t.detach(),
}

def _apply_view_st(st: ShapeTracker, ops: tuple) -> ShapeTracker:
  return _apply_ops(st, ops, _ST_VIEW_HANDLERS)

def _apply_view_ops(tensor: Tensor, ops: tuple) -> Tensor:
  return _apply_ops(tensor, ops, _TENSOR_VIEW_HANDLERS)
def wrap_view_op(_name, fn, recorder):
  def _wrap(*args,**kwargs):
    args = [unwrap(x) if isinstance(x, torch.Tensor) else x for x in args]
    kwargs = {k:unwrap(v) if isinstance(v, torch.Tensor) else v for k,v in kwargs.items()}
    parent = args[0]
    ret = fn(*args,**kwargs)
    ret._view_base = base = canonical_base(parent)
    if not hasattr(base, "_views"): base._views = set()
    base._views.add(weakref.ref(ret))
    parent_ops = _get_view_ops(parent)
    parent_st = _get_view_st(parent)
    new_ops = tuple(recorder(parent, args, kwargs, ret))
    ret._view_ops = parent_ops + new_ops
    ret._view_st = _apply_view_st(parent_st, new_ops)
    return wrap(ret)
  return _wrap
def _record_simple(name, value_fn):
  return lambda parent, args, kwargs, ret: ((name, value_fn(parent, args, kwargs, ret)),)

_record_reshape = _record_simple("reshape", lambda _parent, _args, _kwargs, ret: tuple(ret.shape))
_record_expand = _record_simple("expand", lambda _parent, _args, _kwargs, ret: tuple(ret.shape))
_record_bitcast = _record_simple("bitcast", lambda _parent, _args, _kwargs, ret: ret.dtype)
_record_detach = _record_simple("detach", lambda *_: None)

def _record_permute_from_order(order): return (("permute", tuple(order)),)
def _record_transpose_dims(parent, dim0, dim1):
  ndim = parent.ndim
  dim0 %= ndim
  dim1 %= ndim
  order = list(range(ndim))
  order[dim0], order[dim1] = order[dim1], order[dim0]
  return _record_permute_from_order(order)
def _record_select(parent, args, kwargs, ret):
  _, dim, idx = args
  ndim = parent.ndim
  dim = dim % ndim
  dim_size = parent.shape[dim]
  if idx < 0: idx += dim_size
  idx = max(0, min(dim_size-1, idx))
  parent_shape = tuple(parent.shape)
  shrink_arg = tuple((idx, idx+1) if i == dim else (0, parent_shape[i]) for i in range(ndim))
  return (("shrink", shrink_arg), ("reshape", tuple(ret.shape)))
def _record_slice(parent, args, kwargs, ret):
  _, *rest = args
  dim = kwargs.get("dim", rest[0] if len(rest) > 0 else 0)
  start = kwargs.get("start", rest[1] if len(rest) > 1 else None)
  end = kwargs.get("end", rest[2] if len(rest) > 2 else None)
  step = kwargs.get("step", rest[3] if len(rest) > 3 else 1)
  dim = dim % parent.ndim
  size = parent.shape[dim]
  step = kwargs.get("step", step)
  step = 1 if step is None else step
  if step <= 0: raise NotImplementedError("slice with non-positive step is not supported")
  start = kwargs.get("start", start)
  end = kwargs.get("end", end)
  start = 0 if start is None else start
  end = size if end is None else end
  if start < 0: start += size
  if end < 0: end += size
  start = max(0, min(size, start))
  end = max(0, min(size, end))
  if step > 1 and end < start: end = start
  return (("slice", (dim, start, end, step)),)

view_ops = {
  "aten.view": (Tensor.reshape, _record_reshape),
  "aten._unsafe_view": (Tensor.reshape, _record_reshape),  # when are views unsafe, and do we care?
  "aten.view.dtype": (lambda self,dtype: self.bitcast(_from_torch_dtype(dtype)), _record_bitcast),
  "aten.expand": (Tensor.expand, _record_expand),
  "aten.t": (Tensor.transpose, lambda parent, args, kwargs, ret: _record_transpose_dims(parent, -2, -1)),
  "aten.transpose.int": (Tensor.transpose, lambda parent, args, kwargs, ret: _record_transpose_dims(parent, args[1], args[2])),
  "aten.squeeze.dim": (Tensor.squeeze, _record_reshape),
  "aten.unsqueeze": (Tensor.unsqueeze, _record_reshape),
  "aten.detach": (Tensor.detach, _record_detach),
  "aten.select.int": (lambda self, dim, idx: self[(slice(None),) * (dim%self.ndim) + (idx,)], _record_select),
  "aten.slice.Tensor": (_slice_tensor, _record_slice),
}

for k,(fn, recorder) in view_ops.items(): torch.library.impl(k.replace("aten.", "aten::"), "privateuseone")(wrap_view_op(k, fn, recorder))

torch.library.impl("aten::alias", "privateuseone")(wrap_view_op("aten::alias", lambda self: self, lambda *a, **k: ()))

# in place operations with views
def realize_with_views(self: Tensor, views: Tensor):
  if not self.uop.is_contiguous(): self.replace(self.contiguous())
  self.replace(self.clone().realize())
  for v in views:
    ops = getattr(v, "_view_ops", None)
    if not ops: continue
    ret = _apply_view_ops(self, tuple(ops))
    v.replace(ret)
def maybe_realize_storage(self: Tensor) -> bool:
  if realize:=is_view(self): realize_with_views((base:=canonical_base(self)), derived_views(base))
  return realize
def inplace_fn(outvars: str|list[str]):
  if type(outvars) is str: outvars = [outvars]
  def decorator(fn):
    sig = inspect.signature(fn)
    def wrapper(*args, **kwargs):
      bound = sig.bind(*args, **kwargs)
      outs = [kwargs.get(v, bound.arguments.get(v)) for v in outvars]
      outs = [unwrap(o) if isinstance(o, torch.Tensor) else o for o in outs]
      realize = any(maybe_realize_storage(o) for o in outs)
      ret = fn(*args, **kwargs)
      if realize: Tensor.realize(*(o for o in outs))
      return ret
    return wrapper
  return decorator

# *** bad functions on CPU ***

@torch.library.impl("aten::_index_put_impl_", "privateuseone")
@inplace_fn("self")
def _index_put_impl_(self, indices, values, accumulate=False, unsafe=False):
  # TODO: move to tinygrad
  ret = aten._index_put_impl_(self.cpu(), [x.cpu() if isinstance(x, torch.Tensor) else None for x in indices], values.cpu(), accumulate, unsafe).to(self.device)
  return wrap(unwrap(self).assign(unwrap(ret)))

@torch.library.impl("aten::index_put", "privateuseone")
def index_put(self, indices, values, accumulate=False):
  return aten.index_put(self.cpu(), [z.cpu() if isinstance(z, torch.Tensor) else None for z in indices], values.clone().cpu(), accumulate).tiny()

@torch.library.impl("aten::isin.Tensor_Tensor_out", "privateuseone")
def isin_tensor_tensor_out(x, y, *, assume_unique=False, invert=False, out=None): return out.copy_(aten.isin(x.cpu(), y.cpu(), assume_unique=assume_unique, invert=invert).tiny())

@torch.library.impl("aten::randperm.generator_out", "privateuseone")
def randperm_generator(n, generator=None, out=None):
  return out.copy_(wrap(Tensor.randperm(n, generator=generator, device=unwrap(out).device)))

@torch.library.impl("aten::cummax", "privateuseone")
def cummax(self, dim):
  # TODO: support cummax with indices to match torch
  cummax, indices = aten.cummax(self.cpu(), dim)
  return (cummax.tiny(), indices.tiny())

@torch.library.impl("aten::nonzero", "privateuseone")
# TODO: move to tinygrad
def nonzero(self): return aten.nonzero(self.cpu()).tiny()

@torch.library.impl("aten::_linalg_eigh", "privateuseone")
# TODO: move to tinygrad
def _linalg_eigh(self, UPLO: str = 'U'):
  w, v = torch.linalg.eigh(self.cpu(), UPLO=UPLO)
  return w.tiny(), v.tiny()

@torch.library.impl("aten::_linalg_det", "privateuseone")
# TODO: move to tinygrad
def _linalg_det(self: torch.Tensor):
  result = aten._linalg_det(self.cpu())
  return result[0].tiny(), result[1].tiny(), result[2].tiny()

def upsample_backward(grad_out, output_size, input_size, *args, f=None): return f(grad_out.cpu(), output_size, input_size, *args).tiny()

for i in [
  "upsample_linear1d_backward", "upsample_nearest1d_backward", "_upsample_nearest_exact1d_backward",
  "upsample_nearest2d_backward", "_upsample_nearest_exact2d_backward",
  "upsample_nearest3d_backward", "_upsample_nearest_exact3d_backward",
  "upsample_trilinear3d_backward", "upsample_bilinear2d_backward"
]:
  torch.library.impl(f"aten::{i}", "privateuseone")(functools.partial(upsample_backward, f=getattr(aten, i)))

# *** end bad functions on CPU ***

@torch.library.impl("aten::index.Tensor", "privateuseone")
def index_tensor(x, y):
  return wrap(unwrap(x)[[unwrap(_y.to(x.device)) if _y is not None else slice(None) for _y in y]])

@torch.library.impl("aten::zero_", "privateuseone")
@inplace_fn("x")
def zero_(x):
  if TORCH_DEBUG: print(f"zero_ {x.shape}")
  tt = unwrap(x)
  tt.assign(tt.zeros_like())

@torch.library.impl("aten::fill_.Scalar", "privateuseone")
@inplace_fn("x")
def fill_scalar(x, y):
  if TORCH_DEBUG: print(f"fill_.Scalar {x.shape} {y}")
  tt = unwrap(x)
  tt.assign(tt.full_like(y))

@torch.library.impl("aten::_local_scalar_dense", "privateuseone")
def _local_scalar_dense(tensor): return unwrap(tensor).item()


def _as_strided_impl(tensor:Tensor, size, stride, storage_offset):
  base = canonical_base(tensor)
  try:
    ops, st = _strided_view_ops(tuple(base.shape), tuple(size), tuple(stride), storage_offset)
    ret = _apply_view_ops(base, ops)
    ret._view_base = base
    if not hasattr(base, "_views"): base._views = set()
    base._views.add(weakref.ref(ret))
    ret._view_ops = ops
    ret._view_st = st
    return ret
  except Exception:
    flat = base.contiguous().reshape((-1,))
    total = int(prod(size))
    return flat[storage_offset:storage_offset+total].reshape(tuple(size))

@torch.library.impl("aten::as_strided", "privateuseone")
def as_strided(tensor:torch.Tensor, size, stride, storage_offset=None):
  storage_offset = storage_offset or tensor.storage_offset()
  return wrap(_as_strided_impl(unwrap(tensor), size, stride, storage_offset))

@torch.library.impl("aten::_reshape_alias", "privateuseone")
def _reshape_alias(tensor:torch.Tensor, size, stride):
  return wrap(_as_strided_impl(unwrap(tensor), size, stride, tensor.storage_offset()))

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
  return ShapeTracker.from_shape(( _get_buffer_size(first.shape, first.strides, first.offset, first.mask), ))

def _to_movement_ops(st: ShapeTracker) -> List[Tuple[MovementOps, Tuple]]:
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

def _get_buffer_size(shape, strides, offset, mask):
  real_shape = tuple(y-x for x,y in mask) if mask else shape
  offset = offset + sum(stp * (s-1) for s,stp in zip(real_shape, strides) if stp < 0)
  real_offset = offset + (sum(x*stp for (x,_),stp in zip(mask, strides)) if mask else 0)
  real_real_shape = [s for s,stp in zip(real_shape, strides) if stp]
  strides = [abs(stp) if isinstance(stp, int) else stp for stp in strides if stp]
  return real_offset + sum((s-1)*stp for s, stp in zip(real_real_shape, strides)) + 1

def _movement_to_view_op(mop: MovementOps, arg: Tuple) -> Tuple[str, Tuple]:
  if mop is MovementOps.RESHAPE: return ("reshape", tuple(arg))
  if mop is MovementOps.PERMUTE: return ("permute", tuple(arg))
  if mop is MovementOps.EXPAND: return ("expand", tuple(arg))
  if mop is MovementOps.PAD: return ("pad", tuple(arg))
  if mop is MovementOps.SHRINK: return ("shrink", tuple(arg))
  if mop is MovementOps.STRIDE: return ("flip", tuple(x == -1 for x in arg))
  raise ValueError(f"unsupported movement op {mop}")

def _strided_view_ops(base_shape: tuple, size: tuple, stride: tuple, storage_offset: int) -> tuple[tuple, ShapeTracker]:
  st = ShapeTracker.from_shape(base_shape)
  st = ShapeTracker(st.views + (View.create(size, stride, storage_offset),))
  mops = _to_movement_ops(st)
  if mops and mops[0] == (MovementOps.RESHAPE, base_shape): mops = mops[1:]
  return tuple(_movement_to_view_op(mop, arg) for mop,arg in mops), st

@torch.library.impl("aten::empty_strided", "privateuseone")
def empty_strided(size, stride, dtype, layout=None, device=None, pin_memory=False):
  if TORCH_DEBUG: print(f"empty_strided {size=} {stride=} {dtype=} {layout=} {device=} {pin_memory=}")
  ret = Tensor.empty(*size, dtype=_from_torch_dtype(dtype), device=_from_torch_device(device)).contiguous()
  # TODO: should return with requested strides
  return wrap(ret)

@torch.library.impl("aten::empty.memory_format", "privateuseone")
def empty_memory_format(size, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
  if TORCH_DEBUG: print(f"empty.memory_format {size=} {dtype=} {layout=} {device=} {pin_memory=} {memory_format=}")
  ret = Tensor.empty(*size, dtype=_from_torch_dtype(dtype or torch.get_default_dtype()), device=_from_torch_device(device)).contiguous()
  return wrap(ret)

@torch.library.impl("aten::max_pool2d_with_indices", "privateuseone")
def max_pool2d_with_indices(self:torch.Tensor, kernel_size:tuple[int, ...], stride=None, padding=0, dilation=1, ceil_mode=False):
  # TODO: supprt stride [] in tinygrad?
  if stride is not None and len(stride) == 0: stride = None
  ret, idx = unwrap(self).max_pool2d(kernel_size, stride, dilation, padding, ceil_mode, return_indices=True)
  return (wrap(ret), wrap(idx.cast(dtypes.int64)))

@torch.library.impl("aten::max_pool2d_with_indices_backward", "privateuseone")
def max_pool2d_with_indices_backward(grad_out:torch.Tensor, self:torch.Tensor, kernel_size:tuple[int, ...], stride=None, padding=0, dilation=1, ceil_mode=False, indices=None):
  return wrap(Tensor.max_unpool2d(unwrap(grad_out), unwrap(indices), output_size=unwrap(self).shape))

@torch.library.impl("aten::max_unpool2d", "privateuseone")
def max_unpool2d(self:torch.Tensor, indices:torch.Tensor, output_size):
  return wrap(unwrap(self).max_unpool2d(unwrap(indices), output_size=output_size))

@torch.library.impl("aten::arange", "privateuseone")
def arange(end, dtype=None, device=None, pin_memory=None):
  has_float = isinstance(end, float)
  return wrap(Tensor.arange(0, end, dtype=_from_torch_dtype(dtype or (torch.get_default_dtype() if has_float else torch.int64))))

@torch.library.impl("aten::arange.start", "privateuseone")
def arange_start(start, end, dtype=None, device=None, pin_memory=None):
  has_float = any(isinstance(x, float) for x in (start, end))
  return wrap(Tensor.arange(start, end, dtype=_from_torch_dtype(dtype or (torch.get_default_dtype() if has_float else torch.int64))))

@torch.library.impl("aten::arange.start_step", "privateuseone")
def arange_start_step(start, end, step, dtype=None, device=None, pin_memory=None):
  has_float = any(isinstance(x, float) for x in (start, end, step))
  return wrap(Tensor.arange(start, end, step, dtype=_from_torch_dtype(dtype or (torch.get_default_dtype() if has_float else torch.int64))))

@torch.library.impl("aten::convolution_overrideable", "privateuseone")
def convolution_overrideable(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
  if TORCH_DEBUG >= 1:
    print(f"convolution {input.shape=} {weight.shape=} {stride=} {padding=} {dilation=} {transposed=} {output_padding=} {groups=}")
  input, weight, bias = unwrap(input), unwrap(weight), unwrap(bias) if bias is not None else None
  # TODO: fix test_biased_conv2d fails without realize()
  if not transposed: return wrap(input.conv2d(weight, bias, groups=groups, stride=stride, dilation=dilation, padding=padding).realize())
  return wrap(input.conv_transpose2d(weight, bias, groups=groups, stride=stride, dilation=dilation, padding=padding, output_padding=output_padding).realize())

@torch.library.impl("aten::convolution_backward_overrideable", "privateuseone")
def convolution_backward_overrideable(grad_out, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask):
  if TORCH_DEBUG >= 1:
    print(f"convolution_backward {input.shape=} {weight.shape=} {stride=} {padding=} {dilation=} {transposed=} {output_padding=} {groups=}")
  grad_out, input, weight, bias = unwrap(grad_out), unwrap(input), unwrap(weight), Tensor.zeros(weight.shape[0], device=_from_torch_device(weight.device))
  if not transposed: out = Tensor.conv2d(input, weight, bias, groups=groups, stride=stride, dilation=dilation, padding=padding)
  else:
    bias = Tensor.zeros(weight.shape[1] * groups)
    out = Tensor.conv_transpose2d(input, weight, bias, groups=groups, stride=stride, dilation=dilation, padding=padding, output_padding=output_padding)
  grads = out.gradient(*[t for t,m in zip([input, weight, bias], output_mask) if m], gradient=grad_out)
  return tuple([wrap(grads.pop(0)) if m else None for m in output_mask])

@torch.library.impl("aten::slice_backward", "privateuseone")
def slice_backward(grad_out, input_sizes, dim, start, end, step):
  grad_input = Tensor.zeros(input_sizes).contiguous()
  slices = [slice(None)] * len(input_sizes)
  slices[dim] = slice(start, end, step)
  grad_input[slices] = unwrap(grad_out)
  return wrap(grad_input)

@torch.library.impl("aten::select_backward", "privateuseone")
def select_backward(grad_out, input_sizes, dim, index):
  grad_input = Tensor.zeros(input_sizes).contiguous()
  slices = [slice(None)] * len(input_sizes)
  slices[dim] = index
  grad_input[slices] = unwrap(grad_out)
  return wrap(grad_input)

def avg_pool(self, kernel_size, stride=[], padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
  return wrap(unwrap(self).avg_pool2d(kernel_size, stride if stride != [] else None, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad))

def avg_pool_backward(grad_out, self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
  self, grad_out = unwrap(self), unwrap(grad_out)
  out = Tensor.avg_pool2d(self, kernel_size, stride if stride != [] else None, dilation=1, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)
  return wrap(out.gradient(self, gradient=grad_out)[0])

for dim in [2, 3]:
  torch.library.impl(f"aten::avg_pool{dim}d", "privateuseone")(avg_pool)
  torch.library.impl(f"aten::avg_pool{dim}d_backward", "privateuseone")(avg_pool_backward)

def pad_forward(self, padding, mode=None): return wrap(Tensor.pad(unwrap(self), padding, mode=mode))

def pad_backward(grad_out, self, padding, mode):
  self, grad_out = unwrap(self), unwrap(grad_out)
  out = Tensor.pad(self, padding, mode=mode)
  return wrap(out.gradient(self, gradient=grad_out)[0])

for dim in [1, 2, 3]:
  for pad_type, mode in [("replication", "replicate"), ("reflection", "reflect")]:
    torch.library.impl(f"aten::{pad_type}_pad{dim}d", "privateuseone")(functools.partial(pad_forward, mode=mode))
    torch.library.impl(f"aten::{pad_type}_pad{dim}d_backward", "privateuseone")(functools.partial(pad_backward, mode=mode))

def upsample(self, size, align_corners=False, mode=None): return wrap(Tensor.interpolate(unwrap(self), size, mode=mode, align_corners=align_corners))
for i,pre in enumerate(["", "bi", "tri"]):
  torch.library.impl(f"aten::upsample_{pre}linear{i+1}d", "privateuseone")(functools.partial(upsample, mode="linear"))
  torch.library.impl(f"aten::upsample_nearest{i+1}d", "privateuseone")(functools.partial(upsample, mode="nearest"))
  torch.library.impl(f"aten::_upsample_nearest_exact{i+1}d", "privateuseone")(functools.partial(upsample, mode="nearest-exact"))

@torch.library.impl("aten::scatter_add.out", "privateuseone")
@inplace_fn("out")
def scatter_add(self, dim, index, src, out):
  self, index, src, out = unwrap(self), unwrap(index), unwrap(src), unwrap(out)
  if self.shape == (): return wrap(out.assign(src))
  return wrap(out.assign(Tensor.scatter_reduce(self, dim, index, src, reduce='sum')))

@torch.library.impl("aten::_copy_from", "privateuseone")
def _copy_from(src: torch.Tensor, dest, non_blocking=False):
  realize = dest.is_tiny and maybe_realize_storage(unwrap(dest))
  cast_dtype = _from_torch_dtype(dest.dtype)
  if src.is_tiny and dest.is_tiny:
    to_device = _from_torch_device(dest.device)
    src,dest = unwrap(src),unwrap(dest)
    # TODO we need to properly match dest shape and strides, not blindly assign
    if dest.uop.is_contiguous() or dest.uop.is_realized: src = src.contiguous() # this only solves some cases
    dest.assign(src.cast(cast_dtype).to(to_device))
    if realize: Tensor.realize(dest)
  elif src.is_tiny and dest.is_cpu:
    # TODO: is there a better way?
    dest.resize_(src.numel()).resize_(src.shape)
    src_arr = torch.from_numpy(unwrap(src).cast(cast_dtype).numpy()).clone()
    if dest.is_contiguous():
      dest.copy_(src_arr)
    else:
      dest.numpy()[...] = src_arr.cpu().numpy()
  elif src.is_cpu and dest.is_tiny:
    to_device = _from_torch_device(dest.device)
    # TODO we need to properly match dest shape and strides, not blindly assign
    unwrap(dest).assign(Tensor(src.numpy()).cast(cast_dtype).to(to_device))
    if realize: Tensor.realize(unwrap(dest))
  else:
    raise NotImplementedError(f"can't copy from {src.device} -> {dest.device}")

@torch.library.impl("aten::cat.out", "privateuseone")
@inplace_fn("out")
def cat_out(tensors, dim=0, out=None):
  unwrap(out).assign(Tensor.cat(*[unwrap(x) for x in tensors], dim=dim))

@torch.library.impl("aten::topk.values", "privateuseone")
@inplace_fn(["values", "indices"])
def topk_values(input, k, dim=None, largest=True, sorted=True, values=None, indices=None):
  out_values, out_indices = unwrap(input).topk(k, dim if dim is not None else -1, largest, sorted)
  unwrap(values).assign(out_values)
  unwrap(indices).assign(out_indices.cast(dtypes.int64))
  return wrap(out_values), wrap(out_indices)

@torch.library.impl("aten::sort.values_stable", "privateuseone")
@inplace_fn(["values", "indices"])
def sort_values(input, dim=-1, descending=False, stable=True, values=None, indices=None):
  out_values, out_indices = unwrap(input).sort(dim, descending)
  unwrap(values).assign(out_values)
  unwrap(indices).assign(out_indices.cast(dtypes.int64))
  return wrap(out_values), wrap(out_indices)

@torch.library.impl("aten::_linalg_svd", "privateuseone")
def _linalg_svd(self, full_matrices=False):
  U, S, Vh = unwrap(self).svd(full_matrices)
  return wrap(U), wrap(S), wrap(Vh)

# register some decompositions
from torch._decomp import get_decompositions
decomps = [
  aten.native_batch_norm, aten.native_batch_norm_backward,
  aten.native_layer_norm_backward,
  aten.linalg_cross,
  aten.addmm,
  aten.addcmul,
  aten.addcdiv,
  aten._log_softmax_backward_data,
  aten.threshold_backward,
  aten.softplus_backward,
  aten.elu,  # elu has a scale + input_scale param
  aten.elu_backward,
  aten.softplus,
  aten.logaddexp,
  aten.threshold,
  aten.nll_loss_forward,
  aten.nll_loss_backward,
  aten.nll_loss2d_backward,
  # AttributeError: 'int' object has no attribute '_broadcasted'
  aten.sigmoid_backward,
  aten.tanh_backward,
  aten.sinc,
  aten._prelu_kernel,
  aten.softshrink,
  aten.hardshrink,
  aten.log_sigmoid_forward,
  aten.log_sigmoid_backward,
  aten.isneginf,
  aten.isposinf,
  aten.nan_to_num,
  aten.logit,
  aten.rsub,
  aten.index_select,
  aten.native_dropout, aten.native_dropout_backward,
  aten._softmax_backward_data, aten.embedding_dense_backward,
  aten.linalg_vector_norm,
  aten.binary_cross_entropy, aten.binary_cross_entropy_backward,
  aten.upsample_nearest2d.out,
  # activations
  aten.hardswish, aten.hardswish_backward,
  aten.hardtanh, aten.hardtanh_backward,
  aten.gelu, aten.gelu_backward,
  aten.logical_and,
  aten.randint,
  aten.eye,
  aten.hardsigmoid_backward,
  aten.leaky_relu_backward,
  aten.nll_loss2d_forward,
  aten.unfold_backward,
  # NOTE: many of these don't work or cause infinite loops
  #aten.var_mean,
  #aten.var,
  #aten.rsqrt,
  #aten.max_pool2d_with_indices,
  # NOTE: these are prims
  #aten.digamma,
  #aten.erfinv,
  #aten.lgamma,
  # this needs copy_strided
  #aten.lerp,
  aten.norm,
]
for k,v in get_decompositions(decomps).items():
  key = str(k._schema).split("(")[0]
  if TORCH_DEBUG >= 2: print("register decomp for", k)
  torch.library.impl(key, "privateuseone")(v)

# NOTE: we should only implement the "out" form, it should be 0 overhead
# TODO: due to issue with empty / is_realized, it is slow to use assign so we use replace
# the goal is to make as much as we can this
simple_tensor_methods = [
  # unary (ish)
  "log", "log2", "sqrt", "rsqrt", "sign", "silu", "hardsigmoid", "exp", "exp2", "neg", "reciprocal", "bitwise_not",
  "sigmoid", "clamp", "mish", "erf", "leaky_relu",
  # trig
  "acos", "acosh", "cos", "cosh", "asin", "asinh", "sin", "sinh", "atan", "atanh", "tan", "tanh",
  # rounding
  "ceil", "round", "floor", "trunc",
  # binary
  "mul", "div", "maximum", "minimum", "copysign",
  # modify
  "tril", "triu",
  # reduce
  "all", "any", "argmax", "argmin", "cumsum", "cumprod",
  # complex
  "avg_pool2d", "linspace"]

tiny_backend_out = {**{f"aten.{x}.out":getattr(Tensor,x) for x in simple_tensor_methods}, **{
  "aten.add.out": lambda input,other,alpha=1: input+alpha*other,
  "aten.sub.out": lambda input,other,alpha=1: input-alpha*other, # NOTE: this is also needed to handle reverse
  "aten.div.out_mode": Tensor.div,
  "aten.mul.out": operator.mul,
  "aten.bmm.out": operator.matmul,
  # NOTE: because these methods have a name with "Tensor" in them, they can't go in simple tensor methods
  "aten.remainder.Tensor_out": Tensor.mod,
  "aten.pow.Tensor_Tensor_out": Tensor.pow,
  "aten.pow.Tensor_Scalar_out": Tensor.pow,
  "aten.pow.Scalar_out": lambda input,exponent: input**exponent,
  "aten.bitwise_and.Tensor_out": Tensor.bitwise_and,
  "aten.bitwise_or.Tensor_out": Tensor.bitwise_or,
  "aten.bitwise_xor.Tensor_out": Tensor.bitwise_xor,
  "aten.eq.Tensor_out": Tensor.eq, "aten.eq.Scalar_out": Tensor.eq,
  "aten.ne.Tensor_out": Tensor.ne, "aten.ne.Scalar_out": Tensor.ne,
  "aten.ge.Tensor_out": Tensor.__ge__, "aten.ge.Scalar_out": Tensor.__ge__,
  "aten.gt.Tensor_out": Tensor.__gt__, "aten.gt.Scalar_out": Tensor.__gt__,
  "aten.lt.Tensor_out": Tensor.__lt__, "aten.lt.Scalar_out": Tensor.__lt__,
  "aten.le.Tensor_out": Tensor.__le__, "aten.le.Scalar_out": Tensor.__le__,
  "aten.clamp_max.Tensor_out": lambda input,max_: input.clamp(max_=max_),
  "aten.clamp_min.Tensor_out": lambda input,min_: input.clamp(min_=min_),
  "aten.fmod.Tensor_out": lambda input,other: input-input.div(other, rounding_mode="trunc")*other,
  # TODO: this might result in overflow issues
  "aten.round.decimals_out": lambda self,decimals: (self*10**decimals).round()/10**decimals,
  # TODO: support this in tinygrad
  "aten.bitwise_left_shift.Tensor_out": lambda x,y: x*(2**y),
  "aten.bitwise_right_shift.Tensor_out": lambda x,y: x//(2**y),
  # not in tinygrad. are there decomps for these?
  "aten.log10.out": lambda self: self.log2() * (math.log(2) / math.log(10)),
  "aten.log1p.out": lambda self: (self+1).log(),
  "aten.expm1.out": lambda self: self.exp() - 1,
  "aten.fmax.out": lambda input,other: Tensor.where(input.isnan() & ~other.isnan(), other, Tensor.where(~input.isnan() & other.isnan(), input, Tensor.maximum(input, other))),
  "aten.fmin.out": lambda input,other: Tensor.where(input.isnan() & ~other.isnan(), other, Tensor.where(~input.isnan() & other.isnan(), input, Tensor.minimum(input, other))),
  "aten.amax.out": lambda self,dim=None: self.max(axis=dim),
  "aten.amin.out": lambda self,dim=None: self.min(axis=dim),
  # TODO: this gets the shape wrong
  #"aten.arange.start_out": Tensor.arange,
  "aten.lerp.Scalar_out": Tensor.lerp,
  "aten.scatter.value_out": Tensor.scatter,
  "aten.where.self_out": Tensor.where,
  "aten.prod.int_out": Tensor.prod,
  "aten.scatter.src_out": Tensor.scatter,
  # NOTE: axis=[] in torch means all, change tinygrad?
  "aten.sum.IntList_out": lambda self,axis,keepdim=False,dtype=None:
    self.sum(axis if axis is None or len(axis) else None, keepdim,
                         dtype = _from_torch_dtype(dtype) if dtype is not None else None),
}}

# we add the "out" here
def wrap_out(f):
  @inplace_fn("out")
  def _wrap_out(*args, **kwargs):
    out = kwargs.pop('out')
    assigned = f(*args, **kwargs)
    if getenv("ALLOW_DTYPE_MISMATCH", 1): assigned = assigned.cast(out.dtype)
    assert out.shape == assigned.shape, f"shape mismatch: {assigned.shape} -> {out.shape}"
    assert out.device == assigned.device, f"device mismatch: {assigned.device} -> {out.device}"
    assert out.dtype == assigned.dtype, f"dtype mismatch: {assigned.dtype} -> {out.dtype}"
    if out.uop.is_realized: assigned = assigned.contiguous() # TODO: how does this map to torch's semantics
    return out.assign(assigned)
  return _wrap_out

tiny_backend = {**{k:wrap_out(v) for k,v in tiny_backend_out.items()}, **{
  "aten.remainder.Scalar_Tensor": lambda x,y: x%y,
  "aten.floor_divide": lambda x,y: x//y,
  "aten.floor_divide_.Tensor": inplace_fn("x")(lambda x,y: x.assign(x//y)),
  # TODO: use tinygrad methods, but they require x to be unsigned
  "aten.__lshift__.Scalar": lambda x,y: x*(2**y),
  "aten.__ilshift__.Scalar": inplace_fn("x")(lambda x,y: x.assign(x*(2**y))),
  "aten.__rshift__.Scalar": lambda x,y: x//(2**y),
  "aten.__irshift__.Scalar": inplace_fn("x")(lambda x,y: x.assign(x//(2**y))),
  # relu doesn't have an out form?
  "aten.relu": Tensor.relu,
  "aten.relu_": inplace_fn("x")(lambda x: x.assign(x.relu())),
  "aten.mean": Tensor.mean,
  "aten.mean.dim": Tensor.mean,
  "aten.min": Tensor.min,
  "aten.max": Tensor.max,
  "aten.mm": Tensor.matmul,
  "aten.mv": Tensor.matmul,
  "aten.dot": Tensor.dot,
  "aten.prod": Tensor.prod,
  "aten.isnan": Tensor.isnan,
  "aten.std.correction": Tensor.std,
  "aten.std_mean.correction": Tensor.std_mean,
  "aten.var.correction": Tensor.var,
  "aten.var_mean.correction": Tensor.var_mean,
  "aten.scatter.value": Tensor.scatter,
  "aten.scatter.value_reduce": Tensor.scatter,
  "aten.gather": lambda self, dim, index: self.gather(dim, index.cast(dtypes.int)),
  "aten.where.self": Tensor.where, # NOTE: this is needed as well as the out type
  "aten.repeat": Tensor.repeat,
  "aten._softmax": lambda self,dim,half_to_float: self.softmax(dim),
  "aten._log_softmax": lambda self,dim,half_to_float: self.log_softmax(dim),
  "aten.random_": inplace_fn("self")(lambda self:
    self.assign(Tensor.randint(*self.shape, low=dtypes.min(self.dtype), high=dtypes.max(self.dtype), device=self.device, dtype=self.dtype))),
  "aten.random_.from": inplace_fn("self")(lambda self, from_, to:
    self.assign(Tensor.randint(*self.shape, low=from_, high=to, device=self.device, dtype=self.dtype))),
  "aten.uniform_": inplace_fn("self")(lambda self, low=0, high=1: self.assign(Tensor.uniform(*self.shape, low=low, high=high, dtype=self.dtype))),
  "aten.normal_": inplace_fn("self")(lambda self, mean=0, std=1: self.assign(Tensor.normal(*self.shape, mean=mean, std=std, dtype=self.dtype))),
  # these don't work in out form, they have size 0
  "aten.abs": Tensor.abs,
  "aten.logical_not": Tensor.logical_not,
  "aten.logical_or_": inplace_fn("x")(lambda x, y: x.assign(x | y)),
  "aten.multinomial": Tensor.multinomial,
  "aten.masked_fill_.Scalar": inplace_fn("self")(lambda self, mask, value: self.assign(self.masked_fill(mask, value))),
  "aten.masked_fill_.Tensor": inplace_fn("self")(lambda self, mask, value: self.assign(self.masked_fill(mask, value))),
  "aten.masked_fill.Scalar": Tensor.masked_fill,
  "aten.masked_fill.Tensor": Tensor.masked_fill,
  "aten.masked_select": Tensor.masked_select,
  "aten.all": Tensor.all,
  "aten.sgn": Tensor.sign,
  "aten.acos": Tensor.acos,
  "aten.any": Tensor.any,
  "aten.bitwise_not": Tensor.bitwise_not,
  "aten.argmax": Tensor.argmax,
  "aten.argmin": Tensor.argmin,
  "aten.asinh": Tensor.asinh,
  "aten.mul": Tensor.mul,
  "aten.atanh": Tensor.atanh,
  "aten.fill_.Tensor": Tensor.full, # TODO: looks wrong
  "aten.flip": Tensor.flip,
  "aten.scatter_reduce.two": Tensor.scatter_reduce,
  "aten.squeeze_.dim": lambda self, dim: self.replace(self.squeeze(dim), allow_shape_mismatch=True), # TODO: inplace view op, here?
  "aten.add.Tensor": lambda input,other,alpha=1: input+alpha*other,
  "aten.linspace": lambda start, stop, steps, dtype=None, **kwargs:
    Tensor.linspace(start, stop, steps, **({"dtype": _from_torch_dtype(dtype)} if dtype is not None else {})),
  "aten.topk": Tensor.topk,
  "aten.constant_pad_nd": lambda self, padding, value=0.0: self.pad(padding, mode="constant", value=value),
  "aten.cumsum": lambda self, dim: (self.contiguous() if prod(self.shape) > 512 else self).cumsum(dim),
  "aten.logsumexp": lambda self, axis, keepdim=False: self.logsumexp(axis[0], keepdim=keepdim),
  "aten.roll": Tensor.roll,
  "aten.logcumsumexp": Tensor.logcumsumexp,
  "aten.lerp.Tensor": Tensor.lerp,
  "aten.ones_like": lambda self, dtype=None, device=None, **kwargs:
    self.ones_like(**{k: v for k, v in {"dtype": _from_torch_dtype(dtype) if dtype else None,
                                        "device": _from_torch_device(device) if device else None}.items() if v is not None}),
  "aten.max.dim": lambda self, dim, keepdim=False: (self.max(dim, keepdim), self.argmax(dim, keepdim).cast(dtype=dtypes.int64)),
  "aten.unfold": Tensor.unfold,
}}

def wrap_fxn(k,f):
  def nf(*args, **kwargs):
    if TORCH_DEBUG:
      print(k, len(args), [x.shape if isinstance(x, torch.Tensor) else x for x in args],
                          {k:v.shape if isinstance(v, torch.Tensor) else v for k,v in kwargs.items()})
    args = [unwrap(x) if isinstance(x, torch.Tensor) else x for x in args]
    kwargs = {k:unwrap(v) if isinstance(v, torch.Tensor) else v for k,v in kwargs.items()}
    out = f(*args, **kwargs)
    if isinstance(out, Tensor): return wrap(out)
    elif isinstance(out, tuple): return tuple(wrap(x) for x in out)
    else: raise RuntimeError(f"unknown output type {type(out)}")
  return nf

for k,v in tiny_backend.items(): torch.library.impl(k.replace("aten.", "aten::"), "privateuseone")(wrap_fxn(k,v))

@torch.library.impl("aten::equal", "privateuseone")
def equal(x: torch.Tensor, y: torch.Tensor): return (x==y).all().item()

if TORCH_DEBUG:
  from torch.utils._python_dispatch import TorchDispatchMode
  class DispatchLog(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
      #print(f"Dispatch Log: {func}(*{args}, **{kwargs})")
      print(f"Dispatch Log: {func}")
      return func(*args, **(kwargs or {}))
  (_dispatch_log:=DispatchLog()).__enter__() # NOTE: must be kept alive

# NOTE: patch torch optimizer step to avoid continously growing the computation graph
import weakref
_torch_modules_with_buffers: weakref.WeakSet[torch.nn.Module] = weakref.WeakSet()
def register_torch_buffer(mod, _name, _buffer): _torch_modules_with_buffers.add(mod)
def get_real_tinygrad_buffers():
  res = set()
  for mod in _torch_modules_with_buffers:
    for _,b in mod.named_buffers(recurse=False):
      if b is not None and b.is_tiny:
        res.add(unwrap(b))
  return res
torch.nn.modules.module.register_module_buffer_registration_hook(register_torch_buffer)

from torch.nn.modules import Module
def param_hook(_grad):
  if _grad is not None and _grad.is_tiny: Tensor.realize(unwrap(_grad))
def module_hook(module:Module, _name, _submodule):
  for param in _submodule.parameters(recurse=False):
    if param.requires_grad: param.register_hook(param_hook)
torch.nn.modules.module.register_module_module_registration_hook(module_hook)

def realize_optimizer_step(optimizer: torch.optim.Optimizer, *args, **kwargs):
  tinygrad_tensors = []
  for param_group in optimizer.param_groups:
    for param in param_group["params"]:
      if param is None: continue
      tinygrad_tensors.append(param.data)
  for state_dict in optimizer.state.values():
    for _, value in state_dict.items():
      if torch.is_tensor(value): tinygrad_tensors.append(value)
  real_tinygrad_tensors = [unwrap(x) for x in tinygrad_tensors if x.is_tiny]
  real_tinygrad_tensors += get_real_tinygrad_buffers()
  if len(real_tinygrad_tensors): Tensor.realize(*real_tinygrad_tensors)

_optimizer_init = torch.optim.Optimizer.__init__
def _optimizer_patched_init(self, *args, **kwargs):
  _optimizer_init(self, *args, **kwargs)
  self.register_step_post_hook(realize_optimizer_step)
torch.optim.Optimizer.__init__ = _optimizer_patched_init
