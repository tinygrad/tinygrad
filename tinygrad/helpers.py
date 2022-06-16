import numpy as np
from collections import namedtuple

def prod(x): return int(np.prod(x))

def reduce_shape(shape, axis):
  return [1 if i in axis else shape[i] for i in range(len(shape))]

<<<<<<< HEAD
def binary_broadcast(x_shape, y_shape, extra=False):
  n_dims = max(len(x_shape), len(y_shape))
  shape_x, shape_y = np.ones(n_dims, dtype=np.int32), np.ones(n_dims, dtype=np.int32)
  shape_x[:len(x_shape)] = np.array(x_shape, dtype=np.int32)
  shape_y[:len(y_shape)] = np.array(y_shape, dtype=np.int32)
  if not np.all((shape_x == 1) | (shape_y == 1) | (shape_x == shape_y)):
    raise Exception(f"binary op unbroadcastable shape mismatch: {x_shape} vs {y_shape}")
  shape_ret = tuple([int(x) for x in np.maximum(shape_x, shape_y)])

  if extra:
    dimlist, complist = [], [] # note: len(dimlist) may be less than n_dims
    def push(dim, comp):
      if len(complist) > 0 and complist[-1] == comp:
        dimlist[-1] *= dim
      elif comp != (False, False):
        dimlist.append(dim); complist.append(comp)
    for i in range(n_dims): # group together any adjacent dimensions that we can to simplify broadcasting
      push(np.int32(max(shape_x[i], shape_y[i])), (shape_x[i] > 1, shape_y[i] > 1))

  return (shape_ret, dimlist, complist) if extra else shape_ret

def get_conv_args(x_shape, w_shape, stride=0, groups=1, padding=0):
  # TODO: https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout
  conv_args = namedtuple('conv_args',
    ['H', 'W', 'groups', 'rcout', 'cin', 'oy', 'ox', 'iy', 'ix', 'ys', 'xs', 'bs', 'cout', 'py', 'px'])
  cout,cin,H,W = w_shape
  ys,xs = (stride, stride) if isinstance(stride, int) else stride
  py,px = (padding, padding) if isinstance(padding, int) else padding
  bs,cin_,iy,ix = x_shape
  oy,ox = (iy+py*2-(H-ys))//ys, (ix+px*2-(W-xs))//xs
  if cin*groups != cin_: raise Exception(f"Input Tensor shape {x_shape} does not match the shape of the weights {w_shape}. ({cin*groups} vs. {cin_})")
  assert cout % groups == 0
  rcout = cout//groups
  return conv_args(H, W, groups, rcout, cin, oy, ox, iy, ix, ys, xs, bs, cout, py, px)

# Buffers should extend this
class ShapeTracker:
  def __init__(self, shape):
    self.shape = shape
    self.strides = [1]
    for d in self.shape[::-1][:-1]:
      self.strides = [d*self.strides[0]] + self.strides

  @property
  def shape(self):
    return tuple(self.shape)

  def reshape(self, new_shape):
    pass
  
  def permute(self, axis):
    pass
  
  def slice(self, arg):
    pass



=======
def get_conv_args(x_shape, w_shape, stride=1, groups=1, padding=0, dilation=1):
  # TODO: https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout
  conv_args = namedtuple('conv_args',
    ['H', 'W', 'groups', 'rcout', 'cin', 'oy', 'ox', 'iy', 'ix', 'ys', 'xs', 'bs', 'cout', 'py', 'px', 'dy','dx'])
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
  return conv_args(H, W, groups, cout//groups, cin, oy, ox, iy, ix, ys, xs, bs, cout, py, px, dy, dx)
>>>>>>> origin/master
