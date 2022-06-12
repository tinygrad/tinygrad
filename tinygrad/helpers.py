import numpy as np
from collections import namedtuple

def prod(x): return int(np.prod(x))

def reduce_shape(shape, axis):
  return [1 if i in axis else shape[i] for i in range(len(shape))]

def get_conv_args(x_shape, w_shape, stride, groups):
  # TODO: https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout
  conv_args = namedtuple('conv_args',
    ['H', 'W', 'groups', 'rcout', 'cin', 'oy', 'ox', 'iy', 'ix', 'ys', 'xs', 'bs', 'cout'])
  cout,cin,H,W = w_shape
  ys,xs = (stride, stride) if isinstance(stride, int) else stride
  bs,cin_,iy,ix = x_shape
  oy,ox = (iy-(H-ys))//ys, (ix-(W-xs))//xs
  if cin*groups != cin_: raise Exception(f"Input Tensor shape {x_shape} does not match the shape of the weights {w_shape}. ({cin*groups} vs. {cin_})")
  assert cout % groups == 0
  rcout = cout//groups
  return conv_args(H, W, groups, rcout, cin, oy, ox, iy, ix, ys, xs, bs, cout)
