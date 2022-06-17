import numpy as np
from tinygrad.helpers import get_conv_args
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
  def custompad(x, padding): return np.pad(x, padding).view(CPUBuffer)
  def expand(x, new_shape): return np.broadcast_to(x, new_shape).view(CPUBuffer)

  @staticmethod
  def fromCPU(x): return x
  def toCPU(x): return x

def unary_op(ctx, op, x):
  if op == UnaryOps.RELU: return x.relu()
  elif op == UnaryOps.EXP: return x.exp()
  elif op == UnaryOps.LOG: return x.log()
  elif op == UnaryOps.NEG: return -x
  elif op == UnaryOps.SIGN: return x.sign()
  else: raise Exception(f"{op} isn't supported")

def binary_op(ctx, op, x, y):
  if op == BinaryOps.ADD: return x+y
  elif op == BinaryOps.SUB: return x-y
  elif op == BinaryOps.MUL: return x*y
  elif op == BinaryOps.DIV: return y/x
  elif op == BinaryOps.POW: return x**y
  elif op == BinaryOps.CMPEQ: return 1.0*(x==y)
  else: raise Exception(f"{op} isn't supported")

def reduce_op(ctx, op, inp, new_shape):
  if inp.shape == new_shape:   # this is just a copy, regardless of the reduce op
    return inp[:]
  else:
    if new_shape == (1,):      # full reduce
      axis = tuple(range(len(inp.shape)))
    else:
      assert len(inp.shape) == len(new_shape)
      axis = tuple([i for i,(a,b) in enumerate(zip(inp.shape, new_shape)) if a != b])
    if op == ReduceOps.SUM: return inp.sum(axis, keepdims=True)
    elif op == ReduceOps.MAX: return inp.amax(axis, keepdims=True)
    else: raise Exception(f"{op} isn't supported")

def movement_op(ctx, op, x, arg=None):
  if op == MovementOps.RESHAPE: return x.reshape(arg)
  elif op == MovementOps.PERMUTE: return x.permute(arg)
  elif op == MovementOps.FLIP: return x.flip(arg)
  elif op == MovementOps.SLICE:
    padding = [(max(0, -p[0]), max(0, p[1]-x.shape[i])) for i,p in enumerate(arg)]
    x = x.custompad(padding)
    slicee = [(p[0] + padding[i][0], p[1] + padding[i][0]) for i,p in enumerate(arg)]
    return x[tuple([slice(x[0], x[1], None) for x in slicee])]
  elif op == MovementOps.EXPAND: return x.expand(arg)
  else: raise Exception(f"{op} isn't supported")

def processing_op(ctx,op,x,w,C):
  assert op == ProcessingOps.CONV, f"{op} isn't supported"
  if C.px > 0 or C.py > 0: x = np.pad(x, [(0,0), (0,0), (C.py, C.py), (C.px, C.px)])
  gx = x.reshape(C.bs,C.groups,C.cin,x.shape[2],x.shape[3])
  tx = np.lib.stride_tricks.as_strided(gx,
    shape=(C.bs, C.groups, C.cin, C.oy, C.ox, C.H, C.W),
    strides=(*gx.strides[0:3], gx.strides[3]*C.ys, gx.strides[4]*C.xs, gx.strides[3]*C.dy, gx.strides[4]*C.dx),
    writeable=False,
  )
  tw = w.reshape(C.groups, C.rcout, C.cin, C.H, C.W)
  tmp = np.zeros((C.bs,C.groups,C.oy,C.ox,C.rcout),dtype=x.dtype)
  for g in range(C.groups):
    #ijYXyx,kjyx -> iYXk ->ikYX
    tmp[:,g] += np.tensordot(tx[:,g], tw[g], ((1,4,5),(1,2,3)))
  return np.moveaxis(tmp,4,2).reshape(C.bs, C.groups*C.rcout, C.oy, C.ox).view(CPUBuffer)
