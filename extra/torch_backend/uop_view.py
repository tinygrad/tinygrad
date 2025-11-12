from __future__ import annotations
import weakref
from typing import Any, Callable
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import GroupOp, Ops, UOp, sint
from tinygrad.helpers import canonicalize_strides, strides_for_shape, prod
from tinygrad.dtype import _from_torch_dtype

def _movement_chain(uop: UOp|Tensor) -> list[tuple[Ops, Any]]:
  # Walk up the UOp graph collecting movement operations
  parents, cur = [], (uop.uop if isinstance(uop, Tensor) else uop)
  while cur.op in {Ops.DETACH, Ops.BITCAST, Ops.CONTIGUOUS, Ops.CONTIGUOUS_BACKWARD, Ops.MULTI} or cur.op in GroupOp.Movement:
    if cur.op in GroupOp.Movement and cur.op is not Ops.PAD: parents.append((cur.op, cur.marg))
    if not cur.src: break
    cur = cur.src[0]
  return list(reversed(parents))

def _compute_strides(uop: UOp|Tensor) -> tuple[tuple[sint, ...], sint]:
  chain = _movement_chain(uop)
  cur = uop.uop if isinstance(uop, Tensor) else uop
  while cur.op in {Ops.DETACH, Ops.BITCAST, Ops.CONTIGUOUS, Ops.CONTIGUOUS_BACKWARD, Ops.MULTI} or cur.op in GroupOp.Movement:
    if cur.op in GroupOp.Movement and cur.op is not Ops.PAD: break
    if not cur.src: break
    cur = cur.src[0]
  shape, strides, offset = cur.shape, strides_for_shape(cur.shape), 0
  for op, arg in chain:
    if op is Ops.RESHAPE: shape, strides = tuple(arg), strides_for_shape(arg)
    elif op is Ops.EXPAND:
      new_strides, old_idx = [], 0
      for ns in arg:
        new_strides.append(strides[old_idx] if old_idx < len(shape) and ns == shape[old_idx] else 0)
        if old_idx < len(shape) and ns == shape[old_idx]: old_idx += 1
      shape, strides = tuple(arg), tuple(new_strides)
    elif op is Ops.SHRINK:
      offset += sum(start * strides[i] for i, (start, _) in enumerate(arg) if i < len(strides))
      shape = tuple(end - start for start, end in arg)
      strides = tuple(strides[i] for i in range(len(shape)) if i < len(strides))
    elif op is Ops.PERMUTE: shape, strides = tuple(shape[i] for i in arg if i < len(shape)), tuple(strides[i] for i in arg if i < len(strides))
    elif op is Ops.FLIP:
      offset += sum((shape[i] - 1) * strides[i] for i, f in enumerate(arg) if f)
      strides = tuple(-strides[i] if arg[i] else strides[i] for i in range(len(arg)))
  return canonicalize_strides(shape, strides), offset

def _canonical_base(view: Tensor) -> Tensor:
  seen = set()
  while hasattr(view, "_view_base") and id(view) not in seen: seen.add(id(view)); view = view._view_base
  return view

def _apply_chain(base: Tensor, chain: list[tuple[Ops, Any]]) -> Tensor:
  ret = base
  for op, arg in chain:
    if op == Ops.RESHAPE: ret = ret.reshape(arg)
    elif op == Ops.EXPAND: ret = ret.expand(arg)
    elif op == Ops.SHRINK: ret = ret.shrink(arg)
    elif op == Ops.PERMUTE: ret = ret.permute(arg)
    elif op == Ops.FLIP: ret = ret.flip(tuple(i for i, f in enumerate(arg) if f))
  return ret

def register_view(ret:Tensor, parent:Tensor):
  ret._view_base = parent
  if not hasattr(ret, "_view_perm"): ret._view_perm = getattr(parent, "_view_perm", tuple(range(parent.ndim)))
  base = _canonical_base(parent)
  if not hasattr(base, "_views"): base._views = set()
  base._views.add(weakref.ref(ret))

def update_shrink_region(tt:Tensor, updater:Callable[[Tensor], Tensor]) -> bool:
  if not hasattr(tt, '_view_base'): return False
  base = _canonical_base(tt)
  chain = _movement_chain(tt)
  # Only handle SHRINK chains
  if not (all(op == Ops.SHRINK for op, _ in chain) or not chain): return False
  if not base.uop.is_contiguous(): base.replace(base.contiguous())
  # Apply shrinks to get view region
  view_region = _apply_chain(base, chain)
  new_region = updater(view_region)
  # Update base with the new region
  slices = [slice(None)] * len(base.shape)
  for op, arg in chain:
    for dim, (start, end) in enumerate(arg):
      old_start = slices[dim].start or 0
      slices[dim] = slice(old_start + start, old_start + end)
  updated_base = base.clone()
  updated_base[tuple(slices)] = new_region
  base.replace(updated_base)
  tt.replace(new_region)
  return True

def _as_strided_impl(tensor:Tensor, size, stride, storage_offset):
  base = _canonical_base(tensor)
  flat = base.contiguous().reshape((-1,))
  indices = []
  for idx in range(int(prod(size))):
    offset, remaining = storage_offset, idx
    for dim_size, dim_stride in zip(reversed(size), reversed(stride)):
      offset += (remaining % dim_size) * dim_stride
      remaining //= dim_size
    indices.append(offset)
  ret = flat[Tensor(indices, dtype=dtypes.int32, device=base.device)].reshape(tuple(size))
  register_view(ret, base)
  ret._as_strided_params = (size, stride, storage_offset)
  return ret

def maybe_realize_storage(self: Tensor) -> bool:
  if not hasattr(self, "_view_base"): return False
  base = _canonical_base(self)
  views = [t for tref in getattr(base, "_views", set()) if (t:=tref()) is not None]
  if not base.uop.is_contiguous(): base.replace(base.contiguous())
  base.replace(base.clone().realize())
  # Rebuild views
  for v in views:
     v.replace(_as_strided_impl(base, *v._as_strided_params) if hasattr(v, '_as_strided_params') else _apply_chain(base, _movement_chain(v)))
  return True

# View operation definitions

def _slice_tensor(tensor: Tensor, dim=0, start=None, end=None, step=1):
  slices = [slice(None)] * tensor.ndim
  slices[dim] = slice(start, end, step)
  return tensor[tuple(slices)]

# Map torch ops to tinygrad functions - UOp graph already records the operations
view_ops = {
  "aten.view": Tensor.reshape,
  "aten._unsafe_view": Tensor.reshape,
  "aten.view.dtype": lambda self, dtype: self.bitcast(_from_torch_dtype(dtype)),
  "aten.expand": Tensor.expand,
  "aten.permute": Tensor.permute,
  "aten.t": Tensor.transpose,
  "aten.transpose.int": Tensor.transpose,
  "aten.squeeze.dim": Tensor.squeeze,
  "aten.unsqueeze": Tensor.unsqueeze,
  "aten.detach": Tensor.detach,
  "aten.select.int": lambda self, dim, idx: self[(slice(None),) * (dim % self.ndim) + (idx,)],
  "aten.slice.Tensor": _slice_tensor,
}