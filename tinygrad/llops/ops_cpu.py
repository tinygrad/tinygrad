import numpy as np
import operator
from typing import ClassVar, Callable, Dict
from tinygrad.ops import UnaryOps, BinaryOps, MovementOps, ReduceOps, FusedOps, GenericExecAST, Op
from tinygrad.helpers import shape_to_axis

base_fxn_for_op : Dict[Op, Callable] = {
  UnaryOps.NOOP: lambda x: x[:], UnaryOps.NEG: lambda x: -x, UnaryOps.NOT: lambda x: (1.0 - x),
  BinaryOps.ADD: operator.add, BinaryOps.SUB: operator.sub, BinaryOps.MUL: operator.mul, BinaryOps.DIV: operator.truediv, BinaryOps.POW: operator.pow,
  ReduceOps.SUM: lambda x, new_shape: x.sum(shape_to_axis(x.shape, new_shape), keepdims=True) if tuple(x.shape) != tuple(new_shape) else x[:],
  ReduceOps.MAX: lambda x, new_shape: (x.amax if hasattr(x, 'amax') else x.max)(shape_to_axis(x.shape, new_shape), keepdims=True) if tuple(x.shape) != tuple(new_shape) else x[:],
  MovementOps.SHRINK: lambda x, arg: x[tuple(slice(p[0], p[1], None) for p in arg)],
}

def einsum_mulacc(einsum, get_strides, expand):
  def einscripts(x): return ''.join(["abcdefghijklmnopqrstuvwxyz"[i] for i in x])
  def axes_slice(strides): return [i for i in range(len(strides)) if strides[i] != 0], tuple(slice(None) if strides[i] != 0 else 0 for i in range(len(strides)))
  def mulacc(a, b, new_shape):
    (a_axes, a_slices), (b_axes, b_slices) = axes_slice(get_strides(a)), axes_slice(get_strides(b))
    out = [i for i in range(len(new_shape)) if a.shape[i] == new_shape[i] and (i in a_axes or i in b_axes)]
    ret = einsum(f"{einscripts(a_axes)}, {einscripts(b_axes)} -> {einscripts(out)}", a[a_slices], b[b_slices])
    return expand(ret.reshape([(1 if i not in a_axes and i not in b_axes else s) for i,s in enumerate(new_shape)]), new_shape)
  return mulacc

numpy_fxn_for_op : Dict[Op, Callable] = {**base_fxn_for_op, **{
  UnaryOps.EXP: lambda x: np.exp(x), UnaryOps.LOG: lambda x: np.log(x),
  BinaryOps.MAX: np.maximum, BinaryOps.CMPEQ: lambda x,y: (x==y).astype(np.float32),
  MovementOps.FLIP: lambda x, axis: np.flip(x, axis), MovementOps.PERMUTE: lambda x, order: x.transpose(order),
  MovementOps.PAD: lambda x, padding: np.pad(x, padding), MovementOps.EXPAND: lambda x, new_shape: np.broadcast_to(x, new_shape),
  FusedOps.MULACC: einsum_mulacc(lambda s,a,b: np.einsum(s, a.copy(), b.copy()), lambda x: x.strides, np.broadcast_to)
}}

class CPUBuffer(GenericExecAST):
  fxn_for_op : ClassVar = numpy_fxn_for_op

  @staticmethod
  def fromCPU(x): return CPUBuffer(x)
  def toCPU(x): return x.buf
