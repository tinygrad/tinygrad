from __future__ import annotations
import operator
import numpy as np
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps, GenericExecAST
from tinygrad.helpers import shape_to_axis

class CPUBuffer(np.ndarray, GenericExecAST):
  fxn_for_op = {
    UnaryOps.NOOP: lambda x: x[:], UnaryOps.NEG: lambda x: -x, UnaryOps.RELU: lambda x: x.relu(),
    UnaryOps.EXP: lambda x: x.exp(), UnaryOps.LOG: lambda x: x.log(), UnaryOps.SIGN: lambda x: x.sign(), UnaryOps.RECIPROCAL: lambda x: 1.0/x,
    BinaryOps.ADD: operator.add, BinaryOps.SUB: operator.sub, BinaryOps.MUL: operator.mul,
    BinaryOps.DIV: operator.truediv, BinaryOps.POW: operator.pow, BinaryOps.CMPEQ: lambda x,y: (x==y).float(),
    ReduceOps.SUM: lambda x, new_shape: x.sum(shape_to_axis(x.shape, new_shape), keepdims=True) if tuple(x.shape) != tuple(new_shape) else x[:],
    ReduceOps.MAX: lambda x, new_shape: x.amax(shape_to_axis(x.shape, new_shape), keepdims=True) if tuple(x.shape) != tuple(new_shape) else x[:],
    MovementOps.SHRINK: lambda x, arg: x[tuple(slice(p[0], p[1], None) for p in arg)]
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
  def strided(x, arg): return np.lib.stride_tricks.as_strided(x.ravel().reshape(x.shape), shape=[y[0] for y in arg], strides=[y[1]*x.dtype.itemsize for y in arg]).view(CPUBuffer)

  @staticmethod
  def fromCPU(x): return x.view(CPUBuffer)
  def toCPU(x): return x

  def contiguous(x): return x.ravel().reshape(x.shape)
  def unary_op(x, op): return CPUBuffer.fxn_for_op[op](x)
  def binary_op(x, op, y): return CPUBuffer.fxn_for_op[op](x, y)
  def reduce_op(x, op, new_shape): return CPUBuffer.fxn_for_op[op](x, new_shape)
  def movement_op(x, op, arg=None): return CPUBuffer.fxn_for_op[op](x, arg) if op in CPUBuffer.fxn_for_op else getattr(x, op.name.lower())(arg)

  def processing_op(x,op,w,C):
    assert op == ProcessingOps.CONV, f"{op} isn't supported"
    tx = x.movement_op(MovementOps.STRIDED, (
      (C.bs, C.groups*C.cin*x.shape[2]*x.shape[3]), (C.groups, C.cin*x.shape[2]*x.shape[3]),
      (C.oy, C.sy*x.shape[3]), (C.ox, C.sx), (C.cin, x.shape[2]*x.shape[3]), (C.H, C.dy*x.shape[3]), (C.W, C.dx)))
    tw = w.reshape(C.groups, C.rcout, C.cin, C.H, C.W)
    out = np.einsum("nGhwCHW, GkCHW -> nGkhw", tx.ravel().reshape(tx.shape), tw.ravel().reshape(tw.shape))
    return out.reshape(C.bs, C.groups*C.rcout, C.oy, C.ox).view(CPUBuffer)