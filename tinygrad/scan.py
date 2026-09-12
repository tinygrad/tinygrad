from collections.abc import Callable

from tinygrad.tensor import Tensor


def associative_scan(fn:Callable[[Tensor, Tensor], Tensor], elems:Tensor, axis:int=0, reverse:bool=False) -> Tensor:
  """Inclusive parallel associative scan over ``axis``.

  ``fn`` must be associative and operate elementwise on equally-shaped Tensor slices.
  The implementation uses recursive doubling, requiring O(log n) combine stages rather
  than a sequential Python loop over the scan dimension.
  """
  axis = elems._resolve_dim(axis)
  if elems.shape[axis] == 0: return elems

  x = elems.flip(axis) if reverse else elems
  x = x.transpose(axis, -1)
  offset = 1
  while offset < x.shape[-1]:
    x = x[..., :offset].cat(fn(x[..., :-offset], x[..., offset:]), dim=-1)
    offset <<= 1
  x = x.transpose(axis, -1)
  return x.flip(axis) if reverse else x
