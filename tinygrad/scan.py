from tinygrad.tensor import Tensor


def associative_scan(fn, elems, axis=0, reverse=False):
  axis = elems._resolve_dim(axis)
  if elems.shape[axis] == 0:
    return elems
  x = elems.flip(axis) if reverse else elems
  n = x.shape[axis]
  offset = 1
  while offset < n:
    p = tuple((offset, 0) if d == axis else (0, 0) for d in range(x.ndim))
    s = tuple((0, sz) if d == axis else (0, sz) for d, sz in enumerate(x.shape))
    shifted = x.pad(p).shrink(s)
    if reverse:
      combined = fn(x, shifted)
    else:
      combined = fn(shifted, x)
    idx_shape = (1,) * axis + (n,) + (1,) * (x.ndim - axis - 1)
    mask = (Tensor.arange(n).reshape(idx_shape) >= offset).expand(x.shape)
    x = mask.where(combined, x)
    offset <<= 1
  return x.flip(axis) if reverse else x
