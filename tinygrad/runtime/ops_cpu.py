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
  def get_input_axes(t, sum_axes): return tuple(i for i,stride in enumerate(get_strides(t)) if stride != 0 or i in sum_axes)
  def get_sliced_input(t, axes): return t[tuple(slice(None) if i in axes else 0 for i in range(len(get_strides(t))))]
  def mulacc(a, b, out_shape):
    sum_axes = tuple(i for i,s in enumerate(out_shape) if s == 1)
    a_axes, b_axes = get_input_axes(a, sum_axes), get_input_axes(b, sum_axes)
    a_input, b_input = get_sliced_input(a, a_axes), get_sliced_input(b, b_axes)
    out_axes = [i for i in range(len(out_shape)) if (i in a_axes or i in b_axes) and i not in sum_axes]
    ret = einsum(f"{einscripts(a_axes)}, {einscripts(b_axes)} -> {einscripts(out_axes)}", a_input, b_input)
    return expand(ret.reshape(tuple(1 if (i not in a_axes and i not in b_axes) else s for i,s in enumerate(out_shape))), out_shape)
  return mulacc

def as_strided(x, arg):
  shape, stride, offset = arg
  return np.ndarray(shape, x.dtype, buffer=np.require(x, requirements='C'), offset=offset*x.dtype.itemsize,
                    strides=tuple(y*x.dtype.itemsize for y in stride))

numpy_fxn_for_op: Dict[Op, Callable] = {
  BufferOps.CONST: lambda val, dtype: np.array(val, dtype=dtype.np),
  UnaryOps.EXP2: np.exp2, UnaryOps.LOG2: np.log2, UnaryOps.SIN: np.sin, UnaryOps.SQRT: np.sqrt, UnaryOps.NEG: np.negative,
  UnaryOps.CAST: lambda x,y: x.view(y[0].np) if y[1] else x.astype(y[0].np, copy=False),
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

class CPUDevice(Interpreted):
  def __init__(self, device:str): super().__init__(device, NumpyAllocator(), numpy_fxn_for_op)
