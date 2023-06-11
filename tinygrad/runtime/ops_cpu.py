import numpy as np
import operator
from typing import Callable, Dict, Tuple, Optional
from tinygrad.helpers import dtypes, DType
from tinygrad.ops import UnaryOps, BinaryOps, MovementOps, ReduceOps, FusedOps, Op, Interpreted
from tinygrad.runtime.lib import RawBuffer

def shape_to_axis(old_shape:Tuple[int, ...], new_shape:Tuple[int, ...]) -> Tuple[int, ...]:
  assert len(old_shape) == len(new_shape), "reduce shapes must have same dimensions"
  return tuple(i for i,(a,b) in enumerate(zip(old_shape, new_shape)) if a != b)

base_fxn_for_op: Dict[Op, Callable] = {
  BinaryOps.ADD: operator.add, BinaryOps.SUB: operator.sub, BinaryOps.MUL: operator.mul, BinaryOps.DIV: operator.truediv, BinaryOps.POW: operator.pow,
  ReduceOps.SUM: lambda x, new_shape: x.sum(shape_to_axis(x.shape, new_shape), keepdims=True) if tuple(x.shape) != tuple(new_shape) else x[:],
  ReduceOps.MAX: lambda x, new_shape: (x.amax if hasattr(x, 'amax') else x.max)(shape_to_axis(x.shape, new_shape), keepdims=True) if tuple(x.shape) != tuple(new_shape) else x[:],
  MovementOps.RESHAPE: lambda x, arg: x.reshape(arg), MovementOps.SHRINK: lambda x, arg: x[tuple(slice(p[0], p[1], None) for p in arg)],
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

numpy_fxn_for_op: Dict[Op, Callable] = {**base_fxn_for_op, **{
  UnaryOps.NOOP: lambda x: np.require(x, requirements='C'), UnaryOps.EXP: np.exp, UnaryOps.LOG: np.log, UnaryOps.CAST: lambda x,y: x.astype(y.np), UnaryOps.SIN: np.sin,
  BinaryOps.MAX: np.maximum, BinaryOps.CMPEQ: lambda x,y: (x==y).astype(np.float32),
  MovementOps.PERMUTE: lambda x, order: x.transpose(order), MovementOps.PAD: np.pad, MovementOps.EXPAND: np.broadcast_to,
  MovementOps.STRIDE: lambda x, arg: x[tuple(slice(None, None, i) for i in arg)],
  FusedOps.MULACC: einsum_mulacc(lambda s,a,b: np.einsum(s, a.copy(), b.copy()), lambda x: x.strides, np.broadcast_to),
}}

class RawNumpyBuffer(RawBuffer):
  def __init__(self, size:int, dtype:DType, buf:Optional[np.ndarray]=None): super().__init__(size, dtype, buf if buf is not None else np.empty([size], dtype.np))
  @classmethod
  def fromCPU(cls, x): return cls(x.size, dtypes.from_np(x.dtype), x)
  def toCPU(self): return self._buf
CPUBuffer = Interpreted(RawNumpyBuffer, numpy_fxn_for_op, from_underlying=RawNumpyBuffer.fromCPU)
