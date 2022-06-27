from collections import namedtuple
import math

def prod(x): return math.prod(x)
def reduce_shape(shape, axis):
  return [1 if i in axis else shape[i] for i in range(len(shape))]

ConvArgs = namedtuple('ConvArgs', ['H', 'W', 'groups', 'rcout', 'cin', 'oy', 'ox', 'iy', 'ix', 'ys', 'xs', 'bs', 'cout', 'py', 'py_', 'px', 'px_', 'dy', 'dx', 'out_shape'])
def get_conv_args(x_shape, w_shape, stride=1, groups=1, padding=0, dilation=1):
  # TODO: https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout
  cout,cin,H,W = w_shape
  ys,xs = (stride, stride) if isinstance(stride, int) else stride
  if not isinstance(padding, int) and len(padding) == 4: px,px_,py,py_ = padding
  else: py,px = (padding, padding) if isinstance(padding, int) else padding; py_, px_ = py, px
  dy,dx = (dilation, dilation) if isinstance(dilation, int) else dilation
  bs,cin_,iy,ix = x_shape
  # TODO: should be easy to support asymmetric padding by changing output size
  # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html describes these sizes well
  oy = (iy + py + py_ - dy * (H-1) - 1)//ys + 1
  ox = (ix + px + px_ - dx * (W-1) - 1)//xs + 1
  if cin*groups != cin_: raise Exception(f"Input Tensor shape {x_shape} does not match the shape of the weights {w_shape}. ({cin*groups} vs. {cin_})")
  assert cout % groups == 0
  return ConvArgs(H, W, groups, cout//groups, cin, oy, ox, iy, ix, ys, xs, bs, cout, py, py_, px, px_, dy, dx, (bs, cout, oy, ox))
