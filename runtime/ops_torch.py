import torch
import numpy as np
from typing import Dict, Callable
from tinygrad.ops import BufferOps, UnaryOps, BinaryOps, TernaryOps, ReduceOps, MovementOps, Op
from tinygrad.device import Interpreted, Allocator
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv, flatten
from tinygrad.runtime.ops_cpu import einsum_mulacc, reduce_axis

device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if getenv("MPS", 0) else "cpu"))
type_map = {torch.bool: dtypes.bool,
            torch.int8: dtypes.int8, torch.uint8: dtypes.uint8, torch.int16: dtypes.int16, torch.int32: dtypes.int32, torch.int64: dtypes.int64,
            torch.float16: dtypes.float16, torch.bfloat16: dtypes.bfloat16, torch.float32: dtypes.float32, torch.float64: dtypes.float64}
inverse_type_map = {v: k for k,v in type_map.items()}
# TODO: should unsupported types fail instead of implicit conversion?
inverse_type_map.update({dtypes.uint16: torch.int16, dtypes.uint32: torch.int32, dtypes.uint64: torch.int64})
def np_type_cvt(t): return {np.uint32: np.int32, np.uint64: np.int64}.get(t, t)

def as_strided(x, arg):
  shape, stride, offset = arg
  x = x.contiguous()
  offset += x.storage_offset()    # NOTE: contiguous can still have a storage_offset, so we adjust for it
  if any(i < 0 for i in stride):
    return torch.as_strided(x, shape, tuple(abs(i) for i in stride),
      offset + sum((s-1)*a if a < 0 else 0 for (s,a) in zip(shape, stride))).flip([i for i,a in enumerate(stride) if a < 0])
  return torch.as_strided(x, shape, stride, offset)

torch_fxn_for_op: Dict[Op, Callable] = {
  # TODO: torch.tensor should work here. it doesn't due to "overflow" in uint8
  #BufferOps.CONST: lambda val, dtype: torch.tensor(val, device=device, dtype=inverse_type_map[dtype]),
  BufferOps.CONST: lambda val, dtype: torch.from_numpy(np.array(val, dtype=np_type_cvt(dtype.np))).to(device),
  UnaryOps.EXP2: torch.exp2, UnaryOps.LOG2: torch.log2, UnaryOps.SIN: torch.sin, UnaryOps.SQRT: torch.sqrt,
  UnaryOps.CAST: lambda x,y: (x.view if y[1] else x.type)(inverse_type_map[y[0]]),
  UnaryOps.NEG: lambda x: torch.logical_not(x) if x.dtype is torch.bool else torch.neg(x),
  BinaryOps.ADD: torch.add, BinaryOps.SUB: torch.sub, BinaryOps.MUL: torch.mul, BinaryOps.DIV: lambda x,y: torch.div(x, y).type(x.dtype),
  BinaryOps.XOR: torch.bitwise_xor, BinaryOps.MAX: torch.maximum, BinaryOps.CMPLT: torch.lt, BinaryOps.CMPEQ: torch.eq,
  ReduceOps.SUM: lambda x, new_shape: x.sum(reduce_axis(x.shape, new_shape), dtype=x.dtype, keepdims=True) if x.shape != new_shape else x,
  ReduceOps.MAX: lambda x, new_shape: x.amax(reduce_axis(x.shape, new_shape), keepdims=True) if x.shape != new_shape else x,
  TernaryOps.MULACC: einsum_mulacc(lambda s,a,b: torch.einsum(s, a.float(), b.float()).type(a.dtype), lambda x: x.stride(), lambda x,s: x.expand(s)),
  TernaryOps.WHERE: torch.where, MovementOps.AS_STRIDED: as_strided, MovementOps.EXPAND: lambda x, arg: x.expand(arg),
  MovementOps.PAD: lambda x, padding: torch.nn.functional.pad(x, flatten(padding[::-1])),
}

class TorchAllocator(Allocator):
  def _alloc(self, size:int): return torch.empty([size], device=device, dtype=torch.uint8)
  def copyin(self, dest:torch.Tensor, src:memoryview): dest.copy_(torch.frombuffer(src, dtype=dest.dtype))
  def copyout(self, dest:memoryview, src:torch.Tensor): torch.frombuffer(dest, dtype=src.dtype).copy_(src.flatten())

TorchDevice = Interpreted(TorchAllocator(), torch_fxn_for_op)
