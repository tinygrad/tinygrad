import numpy as np
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps

class CPUBuffer(np.ndarray):
  def __new__(cls, shape, dtype=np.float32): return np.zeros(shape, dtype=dtype).view(CPUBuffer)
  def relu(x): return np.maximum(x, 0)
  def exp(x): return np.exp(x)
  def log(x): return np.log(x)
  def sign(x): return np.sign(x)
  def flip(x, axis): return np.flip(x, axis)
  def amax(x, *args, **kwargs): return np.amax(x, *args, **kwargs)
  def permute(x, order): return x.transpose(order)
  def custompad(x, padding): return np.pad(x, padding).view(CPUBuffer) if any(x > 0 or y > 0 for x,y in padding) else x
  def expand(x, new_shape): return np.broadcast_to(x, new_shape).view(CPUBuffer)

  @staticmethod
  def fromCPU(x): return x.view(CPUBuffer)
  def toCPU(x): return x

  def unary_op(x, op):
    if op == UnaryOps.NOOP: return x[:]
    elif op == UnaryOps.NEG: return -x
    elif op == UnaryOps.RELU: return x.relu()
    elif op == UnaryOps.EXP: return x.exp()
    elif op == UnaryOps.LOG: return x.log()
    elif op == UnaryOps.SIGN: return x.sign()
    else: raise Exception(f"{op} isn't supported")

  def binary_op(x, op, y):
    if op == BinaryOps.ADD: return x+y
    elif op == BinaryOps.SUB: return x-y
    elif op == BinaryOps.MUL: return x*y
    elif op == BinaryOps.DIV: return x/y
    elif op == BinaryOps.POW: return x**y
    elif op == BinaryOps.CMPEQ: return 1.0*(x==y)
    else: raise Exception(f"{op} isn't supported")

  def reduce_op(x, op, new_shape):
    if x.shape == new_shape:   # this is just a copy, regardless of the reduce op
      return x[:]
    else:
      if new_shape == (1,):      # full reduce
        axis = tuple(range(len(x.shape)))
      else:
        assert len(x.shape) == len(new_shape)
        axis = tuple([i for i,(a,b) in enumerate(zip(x.shape, new_shape)) if a != b])
      if op == ReduceOps.SUM: return x.sum(axis, keepdims=True)
      elif op == ReduceOps.MAX: return x.amax(axis, keepdims=True)
      else: raise Exception(f"{op} isn't supported")

  def movement_op(x, op, arg=None):
    if op == MovementOps.RESHAPE: return x.reshape(arg)
    elif op == MovementOps.PERMUTE: return x.permute(arg)
    elif op == MovementOps.FLIP: return x.flip(arg)
    elif op == MovementOps.SLICE:
      padding = [(max(0, -p[0]), max(0, p[1]-x.shape[i])) for i,p in enumerate(arg)]
      return x.custompad(padding)[tuple(slice(p[0] + padding[i][0], p[1] + padding[i][0], None) for i,p in enumerate(arg))]
    elif op == MovementOps.EXPAND: return x.expand(arg)
    else: raise Exception(f"{op} isn't supported")

  def processing_op(x,op,w,C):
    assert op == ProcessingOps.CONV, f"{op} isn't supported"
    x = x.movement_op(MovementOps.SLICE, ((0, x.shape[0]), (0, x.shape[1]), (-C.py, x.shape[2]+C.py_), (-C.px, x.shape[3]+C.px_)))
    gx = x.reshape(C.bs,C.groups,C.cin,x.shape[2],x.shape[3])
    tx = np.lib.stride_tricks.as_strided(gx,
      shape=(C.bs, C.groups, C.cin, C.oy, C.ox, C.H, C.W),
      strides=(*gx.strides[0:3], gx.strides[3]*C.sy, gx.strides[4]*C.sx, gx.strides[3]*C.dy, gx.strides[4]*C.dx),
      writeable=False,
    )
    tw = w.reshape(C.groups, C.rcout, C.cin, C.H, C.W)
    tmp = np.empty((C.bs,C.groups,C.oy,C.ox,C.rcout),dtype=x.dtype)
    for g in range(C.groups):
      #ijYXyx,kjyx -> iYXk ->ikYX
      tmp[:,g] = np.tensordot(tx[:,g], tw[g], ((1,4,5),(1,2,3)))
    return np.moveaxis(tmp,4,2).reshape(C.bs, C.groups*C.rcout, C.oy, C.ox).view(CPUBuffer)
