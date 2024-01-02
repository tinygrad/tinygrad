import numpy as np
from typing import Callable, Dict, Tuple
from tinygrad.helpers import flat_mv
from tinygrad.ops import BufferOps, UnaryOps, BinaryOps, TernaryOps,  ReduceOps, MovementOps, Op
from tinygrad.device import Interpreted, Allocator

def reduce_axis(in_shape:Tuple[int, ...], out_shape:Tuple[int, ...]) -> Tuple[int, ...]:
  assert len(in_shape) == len(out_shape), "reduce shapes must have same dimensions"
  return tuple(i for i,(a,b) in enumerate(zip(in_shape, out_shape)) if a != b)

def einsum_mulacc(einsum, get_strides, expand):
  def einscripts(x): return ''.join(["abcdefghijklmnopqrstuvwxyz"[i] for i in x])
  def axes_slice(strides): return tuple(i for i,s in enumerate(strides) if s != 0), tuple(slice(None) if s != 0 else 0 for s in strides)
  def mulacc(a, b, new_shape):
    (a_axes, a_slices), (b_axes, b_slices) = axes_slice(get_strides(a)), axes_slice(get_strides(b))
    out = [i for i in range(len(new_shape)) if a.shape[i] == new_shape[i] and (i in a_axes or i in b_axes)]
    ret = einsum(f"{einscripts(a_axes)}, {einscripts(b_axes)} -> {einscripts(out)}", a[a_slices], b[b_slices])
    return expand(ret.reshape(tuple(1 if i not in a_axes and i not in b_axes else s for i,s in enumerate(new_shape))), new_shape)
  return mulacc

def as_strided(x, arg):
  shape, stride, offset = arg
  return np.ndarray(shape, x.dtype, buffer=np.require(x, requirements='C'), offset=offset*x.dtype.itemsize,
                    strides=tuple(y*x.dtype.itemsize for y in stride))

numpy_fxn_for_op: Dict[Op, Callable] = {
  BufferOps.CONST: lambda val, dtype: np.array(val, dtype=dtype.np),
  UnaryOps.EXP2: np.exp2, UnaryOps.LOG2: np.log2, UnaryOps.SIN: np.sin, UnaryOps.SQRT: np.sqrt,
  UnaryOps.CAST: lambda x,y: x.view(y[0].np) if y[1] else x.astype(y[0].np, copy=False),
  UnaryOps.NEG: lambda x: np.logical_not(x) if x.dtype == np.bool_ else np.negative(x),
  BinaryOps.MAX: np.maximum, BinaryOps.CMPLT: np.less, BinaryOps.CMPEQ: np.equal, BinaryOps.ADD: np.add, BinaryOps.SUB: np.subtract,
  BinaryOps.MUL: np.multiply, BinaryOps.DIV: lambda x, y: np.divide(x, y).astype(x.dtype, copy=False), BinaryOps.XOR: np.bitwise_xor,
  ReduceOps.SUM: lambda x, new_shape: x.sum(reduce_axis(x.shape, new_shape), dtype=x.dtype, keepdims=True) if x.shape != new_shape else x,
  ReduceOps.MAX: lambda x, new_shape: x.max(reduce_axis(x.shape, new_shape), keepdims=True) if x.shape != new_shape else x,
  TernaryOps.MULACC: einsum_mulacc(lambda s,a,b: np.einsum(s, a.copy(), b.copy(), optimize=True), lambda x: x.strides, np.broadcast_to),
  TernaryOps.WHERE: np.where, MovementOps.AS_STRIDED: as_strided, MovementOps.EXPAND: np.broadcast_to, MovementOps.PAD: np.pad
}

class NumpyAllocator(Allocator):
  def _alloc(self, size:int): return np.empty(size, dtype=np.uint8)
  def as_buffer(self, src:np.ndarray) -> memoryview: return flat_mv(np.require(src, requirements='C').data)
  def copyin(self, dest:np.ndarray, src:memoryview): np.copyto(dest, np.frombuffer(src, dest.dtype).reshape(dest.shape))
  def copyout(self, dest:memoryview, src:np.ndarray): np.copyto(np.frombuffer(dest, src.dtype).reshape(src.shape), src)

CPUDevice = Interpreted(NumpyAllocator(), numpy_fxn_for_op)
