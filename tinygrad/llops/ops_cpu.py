import operator
import numpy as np
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps

class CPUBuffer(np.ndarray):
  fxn_for_op = {
    UnaryOps.NOOP: lambda x: x[:], UnaryOps.NEG: lambda x: -x, UnaryOps.RELU: lambda x: x.relu(),
    UnaryOps.EXP: lambda x: x.exp(), UnaryOps.LOG: lambda x: x.log(), UnaryOps.SIGN: lambda x: x.sign(), UnaryOps.RECIPROCAL: lambda x: 1.0/x,
    BinaryOps.ADD: operator.add, BinaryOps.SUB: operator.sub, BinaryOps.MUL: operator.mul,
    BinaryOps.DIV: operator.truediv, BinaryOps.POW: operator.pow, BinaryOps.CMPEQ: lambda x,y: (x==y).float(),
    ReduceOps.SUM: lambda x, axis: x.sum(axis, keepdims=True), ReduceOps.MAX: lambda x, axis: x.amax(axis, keepdims=True)
  }

  def relu(x): return np.maximum(x, 0)
  def exp(x): return np.exp(x)
  def log(x): return np.log(x)
  def sign(x): return np.sign(x)
  def float(x): return x.astype(np.float32)
  def flip(x, axis): return np.flip(x, axis)
  def amax(x, *args, **kwargs): return np.amax(x, *args, **kwargs)
  def permute(x, order): return x.transpose(order)
  def pad(x, padding): return np.pad(x, padding).view(CPUBuffer)
  def expand(x, new_shape): return np.broadcast_to(x, new_shape).view(CPUBuffer)
  def as_strided(x, size, stride): return np.lib.stride_tricks.as_strided(x, shape=size, strides=[y*x.dtype.itemsize for y in stride]).view(CPUBuffer)
  def contiguous(x): return x.ravel().reshape(x.shape)

  @staticmethod
  def fromCPU(x): return x.view(CPUBuffer)
  def toCPU(x): return x

  def unary_op(x, op): return CPUBuffer.fxn_for_op[op](x)
  def binary_op(x, op, y): return CPUBuffer.fxn_for_op[op](x, y)

  def reduce_op(x, op, new_shape):
    assert len(x.shape) == len(new_shape)
    axis = tuple([i for i,(a,b) in enumerate(zip(x.shape, new_shape)) if a != b])
    return CPUBuffer.fxn_for_op[op](x, axis) if len(axis) > 0 else x[:]

  def movement_op(x, op, arg=None):
    if op == MovementOps.SHRINK:
      return x[tuple(slice(p[0], p[1], None) for p in arg)]
    elif op == MovementOps.STRIDED:
      return x.contiguous().as_strided([x[0] for x in arg], [x[1] for x in arg])
    else:
      return getattr(x, op.name.lower())(arg)

  PREPAD = True
  def processing_op(x,op,w,C):
    assert op == ProcessingOps.CONV, f"{op} isn't supported"
    tx = x.movement_op(MovementOps.STRIDED, (
      (C.bs, C.groups*C.cin*x.shape[2]*x.shape[3]), (C.groups, C.cin*x.shape[2]*x.shape[3]),
      (C.oy, C.sy*x.shape[3]), (C.ox, C.sx), (C.cin, x.shape[2]*x.shape[3]), (C.H, C.dy*x.shape[3]), (C.W, C.dx)))
    tw = w.reshape(C.groups, C.rcout, C.cin, C.H, C.W)
    out = np.einsum("nGhwCHW, GkCHW -> nGkhw", tx.contiguous(), tw.contiguous())
    return out.reshape(C.bs, C.groups*C.rcout, C.oy, C.ox).view(CPUBuffer)
