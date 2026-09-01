from __future__ import annotations
from typing import Callable

from tinygrad.tensor import Tensor

ScanTree = Tensor | tuple["ScanTree", ...] | list["ScanTree"]

def _tree_map(fn:Callable[[Tensor], Tensor], x:ScanTree) -> ScanTree:
  if isinstance(x, Tensor): return fn(x)
  if isinstance(x, tuple): return tuple(_tree_map(fn, y) for y in x)
  return [_tree_map(fn, y) for y in x]

def _tree_zip(fn:Callable[[Tensor, Tensor], Tensor], a:ScanTree, b:ScanTree) -> ScanTree:
  if isinstance(a, Tensor) and isinstance(b, Tensor): return fn(a, b)
  if type(a) is not type(b) or len(a) != len(b): raise ValueError("associative_scan tree mismatch")
  assert isinstance(a, (tuple, list)) and isinstance(b, (tuple, list))
  out = [_tree_zip(fn, x, y) for x,y in zip(a, b)]
  return tuple(out) if isinstance(a, tuple) else out

def associative_scan(fn:Callable[[ScanTree, ScanTree], ScanTree], elems:ScanTree, axis:int=0, reverse:bool=False) -> ScanTree:
  """Inclusive parallel scan of a Tensor tree using associative binary function `fn`."""
  leaves:list[Tensor] = []
  def collect(x:Tensor) -> Tensor:
    leaves.append(x)
    return x
  _tree_map(collect, elems)
  if not leaves: raise ValueError("associative_scan requires at least one Tensor")
  ndim = leaves[0].ndim
  if ndim == 0: return elems
  axis = axis if axis >= 0 else axis + ndim
  if not 0 <= axis < ndim: raise IndexError(f"axis {axis} out of range for ndim {ndim}")
  if any(x.ndim != ndim or x.shape[axis] != leaves[0].shape[axis] for x in leaves): raise ValueError("scan leaves must share scan dimension")
  if not isinstance(n:=leaves[0].shape[axis], int): raise ValueError("associative_scan requires a concrete scan dimension")
  if n <= 1: return elems

  res = _tree_map(lambda x: x.flip(axis) if reverse else x, elems)
  for i in range((n-1).bit_length()):
    offset = 1 << i
    shifted = _tree_map(lambda x: x.pad(tuple((offset, 0) if d == axis else None for d in range(x.ndim))).shrink(
      tuple((0, n) if d == axis else (0, x.shape[d]) for d in range(x.ndim))), res)
    combined = fn(res, shifted) if reverse else fn(shifted, res)
    mask_shape = [1] * ndim
    mask_shape[axis] = n
    mask = Tensor.arange(n).reshape(mask_shape) >= offset
    res = _tree_zip(lambda old, new: mask.where(new, old), res, combined)
  return _tree_map(lambda x: x.flip(axis) if reverse else x, res)
