from __future__ import annotations
import weakref
from typing import Any, Callable, NamedTuple
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import GroupOp, Ops, UOp, sint
from tinygrad.helpers import canonicalize_strides, strides_for_shape, prod
from tinygrad.dtype import _from_torch_dtype

class _ViewSpec(NamedTuple):
  strides: tuple[sint, ...]
  offset: sint

def _movement_chain(uop: UOp|Tensor) -> list[tuple[Ops, Any]]:
  chain, cur = [], (uop.uop if isinstance(uop, Tensor) else uop)
  while cur.op in {Ops.DETACH, Ops.BITCAST, Ops.CONTIGUOUS, Ops.CONTIGUOUS_BACKWARD, Ops.MULTI} or cur.op in GroupOp.Movement:
    if cur.op in GroupOp.Movement and cur.op is not Ops.PAD: chain.append((cur.op, cur.marg))
    if not cur.src: break
    cur = cur.src[0]
  chain.reverse()
  return chain

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
    elif op is Ops.PERMUTE: shape, strides = tuple(shape[i] for i in arg), tuple(strides[i] for i in arg)
    elif op is Ops.FLIP:
      offset += sum((shape[i] - 1) * strides[i] for i, f in enumerate(arg) if f)
      strides = tuple(-strides[i] if arg[i] else strides[i] for i in range(len(arg)))
  return canonicalize_strides(shape, strides), offset

def _canonical_base(view: Tensor) -> Tensor:
  seen = set()
  while hasattr(view, "_view_base") and id(view) not in seen: seen.add(id(view)); view = view._view_base
  return view

def register_view(ret:Tensor, parent:Tensor, op_info:tuple[str, Any]|None=None):
  ret._view_base = parent
  if op_info is not None: ret._view_op = op_info
  if not hasattr(ret, "_view_perm"): ret._view_perm = getattr(parent, "_view_perm", tuple(range(parent.ndim)))
  base = _canonical_base(parent)
  if not hasattr(base, "_views"): base._views = set()
  base._views.add(weakref.ref(ret))

def update_view_region(tt:Tensor, updater:Callable[[Tensor], Tensor]) -> bool:
  if not hasattr(tt, '_view_base'): return False
  base = _canonical_base(tt)
  if not base.uop.is_contiguous(): base.replace(base.contiguous())
  chain = _movement_chain(tt)
  # Rebuild view from base
  view_region = base
  for op, arg in chain:
    if op == Ops.RESHAPE: view_region = view_region.reshape(arg)
    elif op == Ops.EXPAND: view_region = view_region.expand(arg)
    elif op == Ops.SHRINK: view_region = view_region.shrink(arg)
    elif op == Ops.PERMUTE: view_region = view_region.permute(arg)
    elif op == Ops.FLIP: view_region = view_region.flip(tuple(i for i, f in enumerate(arg) if f))
  new_region = updater(view_region)
  # Update base for simple shrink chains
  if all(op == Ops.SHRINK for op, _ in chain) or not chain:
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
  return False

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
  register_view(ret, base, ("as_strided", (size, stride, storage_offset)))
  return ret

def maybe_realize_storage(self: Tensor) -> bool:
  if not hasattr(self, "_view_base"): return False
  base = _canonical_base(self)
  views = [t for tref in getattr(base, "_views", set()) if (t:=tref()) is not None]
  # Collect chains before realizing
  view_chains = []
  for v in views:
    if hasattr(v, '_view_op') and v._view_op[0] == 'as_strided':
      view_chains.append((v, v._view_op[1], True))
    else:
      view_chains.append((v, _movement_chain(v), False))
  # Realize base
  if not base.uop.is_contiguous(): base.replace(base.contiguous())
  base.replace(base.clone().realize())
  # Rebuild views
  for v, chain_or_args, is_as_strided in view_chains:
    if is_as_strided:
      size, stride, offset = chain_or_args
      v.replace(_as_strided_impl(base, size, stride, offset))
    else:
      ret = base
      for op, arg in chain_or_args:
        if op == Ops.RESHAPE: ret = ret.reshape(arg)
        elif op == Ops.EXPAND: ret = ret.expand(arg)
        elif op == Ops.SHRINK: ret = ret.shrink(arg)
        elif op == Ops.PERMUTE: ret = ret.permute(arg)
        elif op == Ops.FLIP: ret = ret.flip(tuple(i for i, f in enumerate(arg) if f))
      v.replace(ret)
  return True

# View operation definitions
def _slice_tensor(tensor: Tensor, dim=0, start=None, end=None, step=1):
  slices = [slice(None)] * tensor.ndim
  slices[dim] = slice(start, end, step)
  return tensor[tuple(slices)]

def _record_simple(name, value_fn): return lambda p, a, k, r: ((name, value_fn(p, a, k, r)),)
def _record_permute_from_order(order): return (("permute", tuple(order)),)
def _record_transpose_dims(parent, dim0, dim1):
  order = list(range(parent.ndim))
  order[dim0 % parent.ndim], order[dim1 % parent.ndim] = order[dim1 % parent.ndim], order[dim0 % parent.ndim]
  return _record_permute_from_order(order)
def _record_select(parent, args, kwargs, ret):
  _, dim, idx = args
  dim, dim_size = dim % parent.ndim, parent.shape[dim % parent.ndim]
  idx = (idx + dim_size if idx < 0 else idx)
  idx = max(0, min(dim_size - 1, idx))
  shrink_arg = tuple((idx, idx+1) if i == dim else (0, parent.shape[i]) for i in range(parent.ndim))
  return (("shrink", shrink_arg), ("reshape", tuple(ret.shape)))
def _record_slice(parent, args, kwargs, ret):
  _, *rest = args
  dim = kwargs.get("dim", rest[0] if rest else 0) % parent.ndim
  size = parent.shape[dim]
  start = kwargs.get("start", rest[1] if len(rest) > 1 else None)
  end = kwargs.get("end", rest[2] if len(rest) > 2 else None)
  step = kwargs.get("step", rest[3] if len(rest) > 3 else 1) or 1
  if step <= 0: raise NotImplementedError("slice with non-positive step")
  start = max(0, min(size, (start or 0) + (size if start and start < 0 else 0)))
  end = max(0, min(size, (end or size) + (size if end and end < 0 else 0)))
  if step > 1 and end < start: end = start
  return (("slice", (dim, start, end, step)),)

view_ops = {
  "aten.view": (Tensor.reshape, _record_simple("reshape", lambda p, a, k, r: tuple(r.shape))),
  "aten._unsafe_view": (Tensor.reshape, _record_simple("reshape", lambda p, a, k, r: tuple(r.shape))),
  "aten.view.dtype": (lambda self, dtype: self.bitcast(_from_torch_dtype(dtype)), _record_simple("bitcast", lambda p, a, k, r: r.dtype)),
  "aten.expand": (Tensor.expand, _record_simple("expand", lambda p, a, k, r: tuple(r.shape))),
  "aten.permute": (Tensor.permute, lambda p, a, k, r: _record_permute_from_order(a[1:])),
  "aten.t": (Tensor.transpose, lambda p, a, k, r: _record_transpose_dims(p, -2, -1)),
  "aten.transpose.int": (Tensor.transpose, lambda p, a, k, r: _record_transpose_dims(p, a[1], a[2])),
  "aten.squeeze.dim": (Tensor.squeeze, _record_simple("reshape", lambda p, a, k, r: tuple(r.shape))),
  "aten.unsqueeze": (Tensor.unsqueeze, _record_simple("reshape", lambda p, a, k, r: tuple(r.shape))),
  "aten.detach": (Tensor.detach, _record_simple("detach", lambda *_: None)),
  "aten.select.int": (lambda self, dim, idx: self[(slice(None),) * (dim % self.ndim) + (idx,)], _record_select),
  "aten.slice.Tensor": (_slice_tensor, _record_slice),
}