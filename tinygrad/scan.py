from __future__ import annotations
from typing import Callable, TypeAlias
from tinygrad.tensor import Tensor

ScanTree: TypeAlias = Tensor | tuple["ScanTree", ...] | list["ScanTree"]


def _map_tree(fn:Callable[[Tensor], Tensor], tree:ScanTree) -> ScanTree:
  if isinstance(tree, Tensor): return fn(tree)
  if isinstance(tree, tuple): return tuple(_map_tree(fn, x) for x in tree)
  if isinstance(tree, list): return [_map_tree(fn, x) for x in tree]
  raise TypeError(f"associative_scan only supports Tensor leaves, got {type(tree).__name__}")


def _zip_tree(fn:Callable[[Tensor, Tensor], Tensor], left:ScanTree, right:ScanTree) -> ScanTree:
  if isinstance(left, Tensor) and isinstance(right, Tensor): return fn(left, right)
  if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
    return tuple(_zip_tree(fn, a, b) for a, b in zip(left, right))
  if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
    return [_zip_tree(fn, a, b) for a, b in zip(left, right)]
  raise TypeError("associative_scan combine function must preserve the input tree structure")


def _leaves(tree:ScanTree) -> list[Tensor]:
  if isinstance(tree, Tensor): return [tree]
  if isinstance(tree, (tuple, list)):
    return [leaf for child in tree for leaf in _leaves(child)]
  raise TypeError(f"associative_scan only supports Tensor leaves, got {type(tree).__name__}")


def _resolve_axis(t:Tensor, axis:int) -> int:
  if not isinstance(axis, int): raise TypeError(f"axis must be int, got {type(axis).__name__}")
  if axis < 0: axis += t.ndim
  if axis < 0 or axis >= t.ndim: raise IndexError(f"axis {axis} is out of bounds for tensor of dimension {t.ndim}")
  return axis


def _slice_tensor(t:Tensor, axis:int, start:int|None, end:int|None) -> Tensor:
  idx = [slice(None)] * t.ndim
  idx[_resolve_axis(t, axis)] = slice(start, end)
  return t[tuple(idx)]


def associative_scan(fn:Callable[[ScanTree, ScanTree], ScanTree], elems:ScanTree, axis:int=0) -> ScanTree:
  """
  Computes an inclusive parallel prefix scan using an associative binary function.

  `fn(left, right)` must be associative and preserve the structure and shapes of
  its inputs. `elems` may be a Tensor or a tuple/list tree of Tensor leaves. All
  leaves must have the same length along `axis`.

  The implementation uses a Hillis-Steele style doubling scan. It has O(log N)
  dependency depth, which makes recurrent computations such as affine state-space
  updates parallelizable while preserving operand order for non-commutative
  associative functions.
  """
  leaves = _leaves(elems)
  if not leaves: raise ValueError("associative_scan requires at least one Tensor leaf")

  axes = [_resolve_axis(t, axis) for t in leaves]
  sizes = [t.shape[a] for t, a in zip(leaves, axes)]
  if any(not isinstance(n, int) for n in sizes): raise ValueError("associative_scan currently requires a static scan dimension")
  if any(n != sizes[0] for n in sizes[1:]): raise ValueError("all Tensor leaves must have the same scan dimension length")
  n = int(sizes[0])
  if n <= 1: return elems

  out = elems
  offset = 1
  while offset < n:
    left = _map_tree(lambda t: _slice_tensor(t, axis, 0, n-offset), out)
    right = _map_tree(lambda t: _slice_tensor(t, axis, offset, n), out)
    combined = fn(left, right)

    combined_leaves = _leaves(combined)
    out_leaves = _leaves(out)
    if len(combined_leaves) != len(out_leaves): raise ValueError("combine function changed the number of Tensor leaves")
    for result, original in zip(combined_leaves, out_leaves):
      a = _resolve_axis(result, axis)
      oa = _resolve_axis(original, axis)
      if result.ndim != original.ndim or result.shape[:a] + result.shape[a+1:] != original.shape[:oa] + original.shape[oa+1:]:
        raise ValueError("combine function must preserve non-scan dimensions")
      if result.shape[a] != n-offset: raise ValueError("combine function must preserve the scan dimension")

    prefix = _map_tree(lambda t: _slice_tensor(t, axis, 0, offset), out)
    out = _zip_tree(lambda a, b: a.cat(b, dim=_resolve_axis(a, axis)), prefix, combined)
    offset <<= 1

  return out
