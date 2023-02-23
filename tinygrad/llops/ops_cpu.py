import numpy as np
import operator
from typing import ClassVar, Callable, Dict
from tinygrad.ops import UnaryOps, BinaryOps, MovementOps, ReduceOps, ProcessingOps, GenericExecAST, Op
from tinygrad.helpers import shape_to_axis, einsum_mulacc

base_fxn_for_op : Dict[Op, Callable] = {
  UnaryOps.NOOP: lambda x: x[:], UnaryOps.NEG: lambda x: -x, UnaryOps.GT0: lambda x: operator.gt(x, 0.0), UnaryOps.RECIPROCAL: lambda x: 1.0/x,
  BinaryOps.ADD: operator.add, BinaryOps.SUB: operator.sub, BinaryOps.MUL: operator.mul, BinaryOps.DIV: operator.truediv, BinaryOps.POW: operator.pow,
  ReduceOps.SUM: lambda x, new_shape: x.sum(shape_to_axis(x.shape, new_shape), keepdims=True) if tuple(x.shape) != tuple(new_shape) else x[:],
  ReduceOps.MAX: lambda x, new_shape: (x.amax if hasattr(x, 'amax') else x.max)(shape_to_axis(x.shape, new_shape), keepdims=True) if tuple(x.shape) != tuple(new_shape) else x[:],
  MovementOps.SHRINK: lambda x, arg: x[tuple(slice(p[0], p[1], None) for p in arg)],
}

def numpy_strided(x,arg): return np.lib.stride_tricks.as_strided(x.ravel().reshape(x.shape), shape=[y[0] for y in arg], strides=[y[1]*x.dtype.itemsize for y in arg])
def numpy_conv(x,w,C):
  assert C.px == 0 and C.px_ == 0 and C.py == 0 and C.py_ == 0, "padding in conv is not supported"
  tx = numpy_strided(x, (
    (C.bs, C.groups*C.cin*x.shape[2]*x.shape[3]), (C.groups, C.cin*x.shape[2]*x.shape[3]),
    (C.oy, C.sy*x.shape[3]), (C.ox, C.sx), (C.cin, x.shape[2]*x.shape[3]), (C.H, C.dy*x.shape[3]), (C.W, C.dx)))
  tw = w.reshape(C.groups, C.rcout, C.cin, C.H, C.W)
  out = np.einsum("nGhwCHW, GkCHW -> nGkhw", tx.ravel().reshape(tx.shape), tw.ravel().reshape(tw.shape))
  return out.reshape(C.bs, C.groups*C.rcout, C.oy, C.ox)

def mulacc(a, b, new_shape):
  subscripts, a_slices, b_slices = einsum_mulacc(a.shape, a.strides, b.shape, b.strides, new_shape)
  return np.einsum(subscripts, a[a_slices].copy(), b[b_slices].copy()).reshape(new_shape)

numpy_fxn_for_op : Dict[Op, Callable] = {**base_fxn_for_op, **{
  UnaryOps.RELU: lambda x: np.maximum(x, 0), UnaryOps.EXP: lambda x: np.exp(x), UnaryOps.LOG: lambda x: np.log(x), BinaryOps.CMPEQ: lambda x,y: (x==y).astype(np.float32),
  MovementOps.FLIP: lambda x, axis: np.flip(x, axis), MovementOps.PERMUTE: lambda x, order: x.transpose(order),
  MovementOps.PAD: lambda x, padding: np.pad(x, padding), MovementOps.EXPAND: lambda x, new_shape: np.broadcast_to(x, new_shape),
  MovementOps.STRIDED: numpy_strided, ProcessingOps.CONV: numpy_conv,
  ProcessingOps.MULACC: mulacc
}}

class CPUBuffer(GenericExecAST):
  fxn_for_op : ClassVar = numpy_fxn_for_op

  @staticmethod
  def fromCPU(x): return CPUBuffer(x)
  def toCPU(x): return x.buf
