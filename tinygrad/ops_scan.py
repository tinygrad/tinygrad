from __future__ import annotations
from typing import Callable

from tinygrad.tensor import Tensor


def _slice_axis(x:Tensor, axis:int, start:int|None=None, stop:int|None=None, step:int|None=None) -> Tensor:
  idx = [slice(None)] * x.ndim
  idx[axis] = slice(start, stop, step)
  return x[tuple(idx)]


def associative_scan(fn:Callable[[Tensor, Tensor], Tensor], elems:Tensor, axis:int=0, reverse:bool=False) -> Tensor:
  """Inclusive parallel associative scan over ``elems``.

  ``fn`` must be associative and operate on tensors with matching shapes. The
  implementation uses a divide-and-conquer recursion with O(log N) dependency
  depth, similar to JAX ``lax.associative_scan``.

  Args:
    fn: associative binary function, e.g. ``lambda a, b: a + b``.
    elems: tensor supporting slicing, ``cat``, ``stack`` and ``flip``.
    axis: axis to scan over.
    reverse: scan from right to left while preserving operand order.
  """
  if not callable(fn): raise TypeError("fn must be callable")
  if elems.ndim == 0: raise ValueError("associative_scan requires at least one dimension")
  axis = axis if axis >= 0 else axis + elems.ndim
  if axis < 0 or axis >= elems.ndim: raise IndexError(f"axis {axis} out of range for ndim {elems.ndim}")
  n = elems.shape[axis]
  if not isinstance(n, int): raise ValueError("associative_scan currently requires a concrete scan dimension")
  if n == 0: return elems

  x = elems.flip((axis,)) if reverse else elems
  combine:Callable[[Tensor, Tensor], Tensor] = (lambda a,b: fn(b,a)) if reverse else fn

  def scan(cur:Tensor) -> Tensor:
    length = cur.shape[axis]
    if not isinstance(length, int): raise ValueError("associative_scan currently requires a concrete scan dimension")
    if length < 2: return cur

    left = _slice_axis(cur, axis, 0, length-1, 2)
    right = _slice_axis(cur, axis, 1, length, 2)
    odd_prefix = scan(combine(left, right))

    first = _slice_axis(cur, axis, 0, 1)
    if length > 2:
      even_src = _slice_axis(cur, axis, 2, length, 2)
      even_src_len = (length-1) // 2
      prior = _slice_axis(odd_prefix, axis, 0, even_src_len)
      even_tail = combine(prior, even_src)
      even_prefix = first.cat(even_tail, dim=axis)
    else:
      even_prefix = first

    pairs = length // 2
    ev = _slice_axis(even_prefix, axis, 0, pairs)
    od = _slice_axis(odd_prefix, axis, 0, pairs)
    interleaved = ev.stack(od, dim=axis+1).flatten(axis, axis+1)
    if length % 2:
      interleaved = interleaved.cat(_slice_axis(even_prefix, axis, pairs, pairs+1), dim=axis)
    return interleaved

  out = scan(x)
  return out.flip((axis,)) if reverse else out
