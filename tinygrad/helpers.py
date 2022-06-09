import numpy as np

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

from enum import Enum
class UnaryOps(Enum):
  RELU = 0
  EXP = 1
  LOG = 2
  NEG = 3
  SIGN = 4

class BinaryOps(Enum):
  ADD = 0
  SUB = 1
  MUL = 2
  DIV = 3
  POW = 4
  A = 5
  CMPEQ = 6

class ReduceOps(Enum):
  SUM = 0
  MAX = 1