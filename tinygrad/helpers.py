import numpy as np
from collections import namedtuple

def prod(x): return int(np.prod(x))

def reduce_shape(shape, axis):
  return [1 if i in axis else shape[i] for i in range(len(shape))]

def get_conv_args(x_shape, w_shape, stride=1, groups=1, padding=0, dilation=1):
  # TODO: https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout
  conv_args = namedtuple('conv_args',
    ['H', 'W', 'groups', 'rcout', 'cin', 'oy', 'ox', 'iy', 'ix', 'ys', 'xs', 'bs', 'cout', 'py', 'px', 'dy','dx', 'out_shape'])
  cout,cin,H,W = w_shape
  ys,xs = (stride, stride) if isinstance(stride, int) else stride
  py,px = (padding, padding) if isinstance(padding, int) else padding
  dy,dx = (dilation, dilation) if isinstance(dilation, int) else dilation
  bs,cin_,iy,ix = x_shape
  # TODO: should be easy to support asymmetric padding by changing output size
  # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html describes these sizes well
  oy = (iy + 2*py - dy * (H-1) - 1)//ys + 1
  ox = (ix + 2*px - dx * (W-1) - 1)//xs + 1
  if cin*groups != cin_: raise Exception(f"Input Tensor shape {x_shape} does not match the shape of the weights {w_shape}. ({cin*groups} vs. {cin_})")
  assert cout % groups == 0
  return conv_args(H, W, groups, cout//groups, cin, oy, ox, iy, ix, ys, xs, bs, cout, py, px, dy, dx, (bs, cout, oy, ox))
