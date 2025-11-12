from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any, Callable
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import GroupOp, Ops, UOp, sint
from tinygrad.helpers import canonicalize_strides, strides_for_shape, prod
from tinygrad.dtype import _from_torch_dtype

MovementChain = tuple[tuple[Ops, Any], ...]

_WRAPPER_OPS = {Ops.DETACH, Ops.BITCAST, Ops.CONTIGUOUS, Ops.CONTIGUOUS_BACKWARD}

# exported symbols to backend
# from .uop_view import maybe_realize_storage, _as_strided_impl, register_view, view_spec_from_uop, update_view_region, view_ops

@dataclass(frozen=True)
class ViewSpec:
  shape: tuple[sint, ...]
  strides: tuple[sint, ...]
  offset: sint


def collect_movement_chain(uop: UOp|Tensor) -> tuple[MovementChain, UOp]:
  """
  Walk a tensor's UOp tree, collecting the movement ops applied on top of the base buffer.
  Returns the movement ops ordered from base -> view and the ultimate base UOp.
  PAD is excluded since it creates new storage and is not a pure view operation.
  """
  chain: list[tuple[Ops, Any]] = []
  cur = uop.uop if isinstance(uop, Tensor) else uop
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



def _slice_tensor(tensor: Tensor, dim=0, start=None, end=None, step=1):
  slices = [slice(None)] * tensor.ndim
  slices[dim] = slice(start, end, step)
  return tensor[tuple(slices)]


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
  "aten.permute": (Tensor.permute, lambda parent, args, kwargs, ret: _record_permute_from_order(args[1:])),
  "aten.t": (Tensor.transpose, lambda parent, args, kwargs, ret: _record_transpose_dims(parent, -2, -1)),
  "aten.transpose.int": (Tensor.transpose, lambda parent, args, kwargs, ret: _record_transpose_dims(parent, args[1], args[2])),
  "aten.squeeze.dim": (Tensor.squeeze, _record_reshape),
  "aten.unsqueeze": (Tensor.unsqueeze, _record_reshape),
  "aten.detach": (Tensor.detach, _record_detach),
  "aten.select.int": (lambda self, dim, idx: self[(slice(None),) * (dim%self.ndim) + (idx,)], _record_select),
  "aten.slice.Tensor": (_slice_tensor, _record_slice),
}



# in place operations with views
def rebuild_view_from_chain(base: Tensor, view: Tensor, chain: MovementChain):
  """Rebuild a view tensor from its base using a pre-collected movement chain."""
  # Build the view by applying the movement operations to the base
  ret = base
  for op, arg in chain:
    if op == Ops.RESHAPE: ret = ret.reshape(arg)
    elif op == Ops.EXPAND: ret = ret.expand(arg)
    elif op == Ops.SHRINK: ret = ret.shrink(arg)
    elif op == Ops.PERMUTE: ret = ret.permute(arg)
    elif op == Ops.FLIP:
      axes = tuple(i for i, f in enumerate(arg) if f)
      ret = ret.flip(axes) if axes else ret
    else: raise NotImplementedError(f"rebuild_view_from_chain does not support {op}")
  view.replace(ret)


def _as_strided_impl(tensor:Tensor, size, stride, storage_offset):
  """Fallback implementation of as_strided using gather indices."""
  base = canonical_base(tensor)
  flat = base.contiguous().reshape((-1,))
  
  # Build gather indices for the strided view
  indices = []
  for idx in range(int(prod(size))):
    offset = storage_offset
    remaining = idx
    for dim_size, dim_stride in zip(reversed(size), reversed(stride)):
      dim_idx = remaining % dim_size
      offset += dim_idx * dim_stride
      remaining //= dim_size
    indices.append(offset)
  
  # Gather and reshape
  idx_tensor = Tensor(indices, dtype=dtypes.int32, device=base.device)
  ret = flat[idx_tensor].reshape(tuple(size))
  register_view(ret, base, ("as_strided", (size, stride, storage_offset)))
  return ret


def realize_with_views(self: Tensor, views: list[Tensor]):
  # Collect all movement chains BEFORE realizing, since realization will change the UOp tree
  view_chains: list[tuple[Tensor, MovementChain]] = []
  as_strided_views: list[tuple[Tensor, tuple]] = []
  
  for v in views:
    if is_view(v):
      # Check if this is an as_strided view (stored in _view_op)
      if hasattr(v, '_view_op') and v._view_op[0] == 'as_strided':
        as_strided_views.append((v, v._view_op[1]))  # (view, (size, stride, offset))
      else:
        chain, _ = collect_movement_chain(v)
        view_chains.append((v, chain))
  
  # Now realize the base
  if not self.uop.is_contiguous(): self.replace(self.contiguous())
  self.replace(self.clone().realize())
  
  # Rebuild each view using its pre-collected chain
  for v, chain in view_chains:
    rebuild_view_from_chain(self, v, chain)
  
  # Rebuild as_strided views
  for v, (size, stride, offset) in as_strided_views:
    rebuilt = _as_strided_impl(self, size, stride, offset)
    v.replace(rebuilt)


def maybe_realize_storage(self: Tensor) -> bool:
  if realize:=is_view(self): realize_with_views((base:=canonical_base(self)), derived_views(base))
  return realize