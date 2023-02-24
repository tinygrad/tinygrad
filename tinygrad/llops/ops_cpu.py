import numpy as np
import operator
from typing import ClassVar, Callable, Dict
from tinygrad.ops import UnaryOps, BinaryOps, MovementOps, ReduceOps, ProcessingOps, GenericExecAST, Op
from tinygrad.helpers import shape_to_axis

base_fxn_for_op : Dict[Op, Callable] = {
  UnaryOps.NOOP: lambda x: x[:], UnaryOps.NEG: lambda x: -x, UnaryOps.RECIPROCAL: lambda x: 1.0/x,
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

def einsum_mulacc(einsum, get_strides):
  def einscripts(x): return ''.join(["abcdefghijklmnopqrstuvwxyz"[i] for i in x])
  def axes_slice(strides): return [i for i in range(len(strides)) if strides[i] != 0], tuple(slice(None) if strides[i] != 0 else 0 for i in range(len(strides)))
  def mulacc(a, b, new_shape):
    (a_axes, a_slices), (b_axes, b_slices) = axes_slice(get_strides(a)), axes_slice(get_strides(b))
    out = [i for i in range(len(new_shape)) if a.shape[i] == new_shape[i] and (i in a_axes or i in b_axes)]
    return einsum(f"{einscripts(a_axes)}, {einscripts(b_axes)} -> {einscripts(out)}", a[a_slices], b[b_slices]).reshape(new_shape)
  return mulacc

numpy_fxn_for_op : Dict[Op, Callable] = {**base_fxn_for_op, **{
  UnaryOps.RELU: lambda x: np.maximum(x, 0), UnaryOps.EXP: lambda x: np.exp(x), UnaryOps.LOG: lambda x: np.log(x),
  UnaryOps.GT0: lambda x: (x > 0.0).astype(np.float32), BinaryOps.CMPEQ: lambda x,y: (x==y).astype(np.float32),
  MovementOps.FLIP: lambda x, axis: np.flip(x, axis), MovementOps.PERMUTE: lambda x, order: x.transpose(order),
  MovementOps.PAD: lambda x, padding: np.pad(x, padding), MovementOps.EXPAND: lambda x, new_shape: np.broadcast_to(x, new_shape),
  MovementOps.STRIDED: numpy_strided, ProcessingOps.CONV: numpy_conv,
  ProcessingOps.MULACC: einsum_mulacc(lambda s,a,b: np.einsum(s, a.copy(), b.copy()), lambda x: x.strides)
}}

class CPUBuffer(GenericExecAST):
  fxn_for_op : ClassVar = numpy_fxn_for_op

  @staticmethod
  def fromCPU(x): return CPUBuffer(x)
  def toCPU(x): return x.buf
