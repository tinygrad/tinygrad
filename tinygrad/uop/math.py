from tinygrad.helpers import ceildiv, flatten
from tinygrad.uop.ops import UOp

def pool_uop(x: UOp, kernel: tuple[int, int], stride: tuple[int, int]) -> UOp:
  """
  Sliding-window tiling using pure UOp movement ops.
  x: (..., H, W)
  returns: (..., t_y, t_x, k_y, k_x)
  """
  k_, s_, d_ = kernel, stride, (1, 1)  # dilation=1 always for Winograd
  noop = [None] * (x.ndim - 2)
  i_ = tuple(x.shape[-2:])

  o_ = [ceildiv(i - d * (k - 1), s) for i, d, k, s in zip(i_, d_, k_, s_)]
  # input size scaling factor
  from tinygrad.uop.ops import smax
  f_ = [smax(1, ceildiv(o * s - d, i)) for o, s, i, d in zip(o_, s_, i_, d_)]
  # repeats so we don't need padding
  x = x.repeat([1] * len(noop) + [ceildiv(k * (i * f + d), i) for k, i, d, f in zip(k_, i_, d_, f_)])
  # handle dilation
  x = x.shrink_to(noop + [k * (i * f + d) for k, i, d, f in zip(k_, i_, d_, f_)])
  x = x.reshape(noop + flatten((k, (i * f + d)) for k, i, d, f in zip(k_, i_, d_, f_)))
  # handle stride
  x = x.shrink_to(noop + flatten((k, o * s) for k, o, s in zip(k_, o_, s_))).reshape(noop + flatten((k, o, s) for k, o, s in zip(k_, o_, s_)))
  x = x.shrink_to(noop + flatten((k, o, 1) for k, o in zip(k_, o_))).reshape(noop + flatten((k, o) for k, o in zip(k_, o_)))
  # permute to move reduce to the end
  return x.permute(*range(len(noop)), *[len(noop) + i * 2 + 1 for i in range(len(i_))], *[len(noop) + i * 2 for i in range(len(i_))])
