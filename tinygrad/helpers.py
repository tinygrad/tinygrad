import numpy as np
from collections import namedtuple

def prod(x): return int(np.prod(x))

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

def get_conv_args(x_shape, w_shape, stride, groups):
  conv_args = namedtuple('conv_args',
    ['H', 'W', 'groups', 'rcout', 'cin', 'oy', 'ox', 'iy', 'ix', 'ys', 'xs', 'bs'])
  cout,cin,H,W = w_shape
  ys,xs = (stride, stride) if isinstance(stride, int) else stride
  bs,cin_,iy,ix = x_shape
  oy,ox = (iy-(H-ys))//ys, (ix-(W-xs))//xs
  if cin*groups != cin_: raise Exception(f"Input Tensor shape {x_shape} does not match the shape of the weights {w_shape}. ({cin*groups} vs. {cin_})")
  assert cout % groups == 0
  rcout = cout//groups
  return conv_args(H, W, groups, rcout, cin, oy, ox, iy, ix, ys, xs, bs)

from enum import Enum
UnaryOps = Enum("UnaryOps", ["RELU", "EXP", "LOG", "NEG", "SIGN"])
BinaryOps = Enum("BinaryOps", ["ADD", "SUB", "MUL", "DIV", "POW", "A", "CMPEQ"])
ReduceOps = Enum("ReduceOps", ["SUM", "MAX"])
MovementOps = Enum("MovementOps", ["RESHAPE", "PERMUTE", "SLICE"])
ProcessingOps = Enum("ProcessingOps", ["CONV", "CONVT", "CONVDW"])
