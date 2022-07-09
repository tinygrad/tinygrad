import operator
import numpy as np
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps

class CPUBuffer(np.ndarray):
  fxn_for_op = {
    UnaryOps.NOOP: lambda x: x[:], UnaryOps.NEG: lambda x: -x, UnaryOps.RELU: lambda x: x.relu(),
    UnaryOps.EXP: lambda x: x.exp(), UnaryOps.LOG: lambda x: x.log(), UnaryOps.SIGN: lambda x: x.sign(),
    BinaryOps.ADD: operator.add, BinaryOps.SUB: operator.sub, BinaryOps.MUL: operator.mul,
    BinaryOps.DIV: operator.truediv, BinaryOps.POW: operator.pow, BinaryOps.CMPEQ: lambda x,y: 1.0*(x==y)
  }

  def relu(x): return np.maximum(x, 0)
  def exp(x): return np.exp(x)
  def log(x): return np.log(x)
  def sign(x): return np.sign(x)
  def flip(x, axis): return np.flip(x, axis)
  def amax(x, *args, **kwargs): return np.amax(x, *args, **kwargs)
  def permute(x, order): return x.transpose(order)
  def custompad(x, padding): return np.pad(x, padding).view(CPUBuffer) if any(x != 0 or y != 0 for x,y in padding) else x
  def expand(x, new_shape): return np.broadcast_to(x, new_shape).view(CPUBuffer)

  @staticmethod
  def fromCPU(x): return x.view(CPUBuffer)
  def toCPU(x): return x

  def unary_op(x, op): return CPUBuffer.fxn_for_op[op](x)
  def binary_op(x, op, y): return CPUBuffer.fxn_for_op[op](x, y)

  def reduce_op(x, op, new_shape):
    assert len(x.shape) == len(new_shape)
    axis = tuple([i for i,(a,b) in enumerate(zip(x.shape, new_shape)) if a != b])
    if x.shape == new_shape: return x[:]   # this is just a copy, regardless of the reduce op
    elif op == ReduceOps.SUM: return x.sum(axis, keepdims=True)
    elif op == ReduceOps.MAX: return x.amax(axis, keepdims=True)

  def movement_op(x, op, arg=None):
    if op == MovementOps.RESHAPE: return x.reshape(arg)
    elif op == MovementOps.PERMUTE: return x.permute(arg)
    elif op == MovementOps.FLIP: return x.flip(arg)
    elif op == MovementOps.SLICE:
      padding = [(max(0, -p[0]), max(0, p[1]-x.shape[i])) for i,p in enumerate(arg)]
      return x.custompad(padding)[tuple(slice(p[0] + padding[i][0], p[1] + padding[i][0], None) for i,p in enumerate(arg))]
    elif op == MovementOps.EXPAND: return x.expand(arg)
    elif op == MovementOps.STRIDED: return np.lib.stride_tricks.as_strided(x, shape=[x[0] for x in arg], strides=[x[1]*4 for x in arg]).view(CPUBuffer)

  def processing_op(x,op,w,C):
    assert op == ProcessingOps.CONV, f"{op} isn't supported"
    x = x.movement_op(MovementOps.SLICE, ((0, x.shape[0]), (0, x.shape[1]), (-C.py, x.shape[2]+C.py_), (-C.px, x.shape[3]+C.px_)))
    gx = x.ravel().reshape(C.bs,C.groups,C.cin,x.shape[2],x.shape[3])
    tx = np.lib.stride_tricks.as_strided(gx,
      shape=(C.bs, C.groups, C.cin, C.H, C.W, C.oy, C.ox),
      strides=(*gx.strides[0:3], gx.strides[3]*C.dy, gx.strides[4]*C.dx, gx.strides[3]*C.sy, gx.strides[4]*C.sx))
    tw = w.reshape(C.groups, C.rcout, C.cin, C.H, C.W)

    # too bad this doesn't mix with stride_tricks, it can be very slow
    #out = np.einsum("nGCHWhw, GkCHW -> nGkhw", tx, tw)

    # 3 lines is faster than 1
    tmp = np.empty((C.groups,C.rcout,C.bs,C.oy,C.ox), dtype=x.dtype)
    for g in range(C.groups): tmp[g] = np.tensordot(tw[g], tx[:,g], ((1,2,3),(1,2,3)))
    out = np.einsum("Gknhw -> nGkhw", tmp)

    return out.reshape(C.bs, C.groups*C.rcout, C.oy, C.ox).view(CPUBuffer)
