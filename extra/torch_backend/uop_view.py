from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple
from tinygrad import Tensor
from tinygrad.uop.ops import GroupOp, Ops, UOp, sint
from tinygrad.helpers import canonicalize_strides, strides_for_shape

MovementChain = Tuple[Tuple[Ops, Any], ...]

_WRAPPER_OPS = {Ops.DETACH, Ops.BITCAST, Ops.CONTIGUOUS, Ops.CONTIGUOUS_BACKWARD}


def _normalize_uop(value: UOp|Tensor) -> UOp:
  return value.uop if isinstance(value, Tensor) else value


def collect_movement_chain(uop: UOp|Tensor) -> tuple[MovementChain, UOp]:
  """
  Walk a tensor's UOp tree, collecting the movement ops applied on top of the base buffer.
  Returns the movement ops ordered from base -> view and the ultimate base UOp.
  PAD is excluded since it creates new storage and is not a pure view operation.
  """
  chain: list[Tuple[Ops, Any]] = []
  cur = _normalize_uop(uop)
  while True:
    if cur.op in _WRAPPER_OPS:
      if not cur.src: break
      cur = cur.src[0]
      continue
    if cur.op is Ops.MULTI:  # MULTI is a movement-like wrapper
      if not cur.src: break
      cur = cur.src[0]
      continue
    # PAD creates new storage and breaks the view chain
    if cur.op is Ops.PAD: break
    if cur.op in GroupOp.Movement:
      chain.append((cur.op, cur.marg))
      if not cur.src: break
      cur = cur.src[0]
      continue
    break
  chain.reverse()
  return tuple(chain), cur


@dataclass(frozen=True)
class ViewSpec:
  shape: tuple[sint, ...]
  strides: tuple[sint, ...]
  offset: sint

  @property
  def contiguous(self) -> bool:
    return self.offset == 0 and self.strides == canonicalize_strides(self.shape, strides_for_shape(self.shape))


def view_spec_from_uop(uop: UOp|Tensor) -> ViewSpec:
  chain, base = collect_movement_chain(uop)
  shape = base.shape
  strides = strides_for_shape(shape)
  offset: sint = 0

  for op, arg in chain:
    match op:
      case Ops.RESHAPE:
        shape = tuple(arg)
        strides = strides_for_shape(shape)
      case Ops.EXPAND:
        new_shape = tuple(arg)
        new_strides: list[sint] = []
        old_idx = 0
        for ns in new_shape:
          if old_idx < len(shape) and ns == shape[old_idx]:
            new_strides.append(strides[old_idx])
            old_idx += 1
          else:
            new_strides.append(0)
        shape = new_shape
        strides = tuple(new_strides)
      case Ops.SHRINK:
        dims: list[sint] = []
        dim_strides: list[sint] = []
        for i, (start, end) in enumerate(arg):
          stride = strides[i]
          offset += start * stride
          dims.append(end - start)
          dim_strides.append(stride)
        shape = tuple(dims)
        strides = tuple(dim_strides)
      case Ops.PERMUTE:
        order = tuple(arg)
        shape = tuple(shape[i] for i in order)
        strides = tuple(strides[i] for i in order)
      case Ops.FLIP:
        flip_flags = tuple(arg)
        offs, new_strides = offset, []
        for dim, flag in enumerate(flip_flags):
          stride = strides[dim]
          if flag:
            offs += (shape[dim] - 1) * stride
            new_strides.append(-stride)
          else:
            new_strides.append(stride)
        offset = offs
        strides = tuple(new_strides)
      case _:
        raise NotImplementedError(f"view_spec_from_uop does not support {op}")

  return ViewSpec(shape, canonicalize_strides(shape, strides), offset)


import weakref
from typing import Any, Callable
from tinygrad import Tensor
from tinygrad.helpers import prod

def is_view(tensor: Tensor) -> bool: return hasattr(tensor, "_view_base")

def canonical_base(view: Tensor) -> Tensor:
  seen = set()
  while hasattr(view, "_view_base"):
    if id(view) in seen: break
    seen.add(id(view))
    view = view._view_base
  return view

def derived_views(base: Tensor) -> list[Tensor]:
  return [t for tref in getattr(base, "_views", set()) if (t:=tref()) is not None]

def register_view(ret_tiny:Tensor, parent:Tensor, op_info:tuple[str, Any]|None=None):
  ret_tiny._view_base = parent
  if op_info is not None: ret_tiny._view_op = op_info
  parent_perm = getattr(parent, "_view_perm", tuple(range(parent.ndim)))
  if not hasattr(ret_tiny, "_view_perm"):
    ret_tiny._view_perm = parent_perm
  base = canonical_base(parent)
  if not hasattr(base, "_views"): base._views = set()
  base._views.add(weakref.ref(ret_tiny))

def _identity_perm(ndim:int) -> tuple[int,...]: return tuple(range(ndim))

def _permute_to_view(t:Tensor, perm:tuple[int,...]|None):
  if perm is None or perm == _identity_perm(len(perm)): return t
  return t.permute(tuple(perm))

def _permute_to_base(t:Tensor, perm:tuple[int,...]|None):
  if perm is None or perm == _identity_perm(len(perm)): return t
  inv = tuple(perm.index(i) for i in range(len(perm)))
  return t.permute(inv)

def update_view_region(tt:Tensor, updater:Callable[[Tensor], Tensor]) -> bool:
  if not hasattr(tt, '_view_base'): return False
  base = canonical_base(tt)
  if not base.uop.is_contiguous(): base.replace(base.contiguous())
  
  # Collect the movement chain to understand how to rebuild and update the view
  chain, _ = collect_movement_chain(tt)
  
  # DEBUG
  import os
  if os.getenv('TORCH_DEBUG'):
    print(f'update_view_region: tt.shape={tt.shape}, base.shape={base.shape}, chain={chain}')
  
  # Rebuild the view from the base
  view_region = base
  for op, arg in chain:
    if os.getenv('TORCH_DEBUG'):
      print(f'  Applying {op} with arg={arg}, current shape={view_region.shape}')
    if op == Ops.RESHAPE: view_region = view_region.reshape(arg)
    elif op == Ops.EXPAND: view_region = view_region.expand(arg)
    elif op == Ops.SHRINK: view_region = view_region.shrink(arg)
    elif op == Ops.PERMUTE: view_region = view_region.permute(arg)
    elif op == Ops.FLIP:
      axes = tuple(i for i, f in enumerate(arg) if f)
      view_region = view_region.flip(axes) if axes else view_region
  
  # Apply the update
  new_region = updater(view_region)
  
  # Now update the base tensor by applying inverse operations
  # Handle cases where the chain is all SHRINKs (common for slicing)
  if all(op == Ops.SHRINK for op, _ in chain):
    # Combine all shrinks into a single slice operation
    # Start with full slices for all dimensions
    slices = [slice(None)] * len(base.shape)
    cur_shape = list(base.shape)
    
    for op, arg in chain:
      # Apply each shrink to compute the final slices
      new_slices = []
      for dim, (start, end) in enumerate(arg):
        if dim < len(slices):
          old_slice = slices[dim]
          if isinstance(old_slice, slice):
            # Compose with existing slice
            old_start = old_slice.start or 0
            new_slices.append(slice(old_start + start, old_start + end))
          else:
            new_slices.append(slice(start, end))
        else:
          new_slices.append(slice(start, end))
      slices = new_slices
      cur_shape = [end - start for start, end in arg]
    
    # Now apply the update
    updated_base = base.clone()
    updated_base[tuple(slices)] = new_region
    base.replace(updated_base)
    tt.replace(new_region)
    return True
  elif len(chain) == 1 and chain[0][0] == Ops.SHRINK:
    # Single shrink case - use slicing
    slices = [slice(start, end) for start, end in chain[0][1]]
    updated_base = base.clone()
    updated_base[tuple(slices)] = new_region
    base.replace(updated_base)
    tt.replace(new_region)
    return True
  elif not chain:
    # No operations - direct replacement
    base.replace(new_region)
    tt.replace(new_region)
    return True
  else:
    # Complex chain - fallback to not handling view updates
    return False

def assign_view_value(tt:Tensor, value:Tensor) -> bool:
  return update_view_region(tt, lambda _region: value)

def _aligned_other(tt:Tensor, other:Tensor|int|float) -> Tensor|int|float:
  if not isinstance(other, Tensor): return other
  if hasattr(other, "_view_base") and canonical_base(other) is canonical_base(tt):
    base = canonical_base(other)
    if hasattr(tt, "_view_indices"): region = base[tt._view_indices]
    else: region = base.reshape(tt.shape)
    perm = getattr(tt, "_view_perm", _identity_perm(region.ndim))
    return _permute_to_view(region, perm)
  return other
