import torch
from typing import Dict, Callable, Optional
from tinygrad.ops import UnaryOps, BinaryOps, MovementOps, FusedOps, Op, Interpreted
from tinygrad.helpers import getenv, dtypes, prod, DType
from tinygrad.runtime.ops_cpu import base_fxn_for_op, einsum_mulacc
from tinygrad.runtime.lib import RawBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if getenv("MPS", 0) else "cpu"))
type_map = {torch.float16: dtypes.float16, torch.float32: dtypes.float32, torch.int8: dtypes.int8, torch.int32: dtypes.int32, torch.int64: dtypes.int64, torch.uint8: dtypes.uint8, torch.bool: dtypes.bool}
inverse_type_map = {v:k for k,v in type_map.items()}

torch_fxn_for_op: Dict[Op, Callable] = {**base_fxn_for_op, **{
  UnaryOps.NOOP: lambda x: x.contiguous(), UnaryOps.EXP: lambda x: x.exp(), UnaryOps.LOG: lambda x: x.log(), UnaryOps.CAST: lambda x,y: x.type(next(k for k,v in type_map.items() if v==y)), UnaryOps.SIN: torch.sin,
  BinaryOps.MAX: torch.maximum, BinaryOps.CMPEQ: lambda x,y: (x==y).float(),
  MovementOps.PAD: lambda x, padding: torch.nn.functional.pad(x, [item for sublist in padding[::-1] for item in sublist]),
  FusedOps.MULACC: einsum_mulacc(lambda s,a,b: torch.einsum(s, a.float(), b.float()).type(torch.promote_types(a.dtype, b.dtype)), lambda x: x.stride(), lambda x,s: x.expand(s)),
  MovementOps.STRIDE: lambda x, arg: x[tuple(slice(None, None, abs(i)) for i in arg)].flip([i for i,a in enumerate(arg) if a < 0]),
  MovementOps.EXPAND: lambda x, arg: x.expand(arg), MovementOps.PERMUTE: lambda x, arg: x.permute(arg)
}}

class RawTorchBuffer(RawBuffer):
  def __init__(self, size:int, dtype:DType, buf:Optional[torch.Tensor]=None): super().__init__(size, dtype, buf if buf is not None else torch.empty([size], dtype=inverse_type_map[dtype]))
  @classmethod
  def fromCPU(cls, x):
    buf = torch.from_numpy(x).requires_grad_(False).to(device)
    return cls(prod(x.shape), type_map[buf.dtype], buf)
  def toCPU(self): return self._buf.cpu().numpy()
TorchBuffer = Interpreted(RawTorchBuffer, torch_fxn_for_op, from_underlying=lambda x: RawTorchBuffer(prod(x.shape), type_map[x.dtype], x))
