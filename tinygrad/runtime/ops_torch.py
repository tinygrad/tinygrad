import torch
from typing import Dict, Callable
from tinygrad.ops import UnaryOps, BinaryOps, MovementOps, FusedOps, Op, Interpreted
from tinygrad.helpers import getenv, dtypes, prod
from tinygrad.runtime.ops_cpu import base_fxn_for_op, einsum_mulacc
from tinygrad.runtime.lib import RawBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if getenv("MPS", 0) else "cpu"))
type_map = {torch.float16: dtypes.float16, torch.float32: dtypes.float32}

torch_fxn_for_op: Dict[Op, Callable] = {**base_fxn_for_op, **{
  UnaryOps.NOOP: lambda x: x.contiguous(), UnaryOps.EXP: lambda x: x.exp(), UnaryOps.LOG: lambda x: x.log(), UnaryOps.CAST: lambda x,y: x.type(next(k for k,v in type_map.items() if v==y)),
  BinaryOps.MAX: torch.maximum, BinaryOps.CMPEQ: lambda x,y: (x==y).float(),
  MovementOps.PAD: lambda x, padding: torch.nn.functional.pad(x, [item for sublist in padding[::-1] for item in sublist]),
  FusedOps.MULACC: einsum_mulacc(lambda s,a,b: torch.einsum(s, a.float(), b.float()).type(torch.promote_types(a.dtype, b.dtype)), lambda x: x.stride(), lambda x,s: x.expand(s)),
  MovementOps.STRIDE: lambda x, arg: x[tuple(slice(None, None, abs(i)) for i in arg)].flip([i for i,a in enumerate(arg) if a < 0]),
  MovementOps.EXPAND: lambda x, arg: x.expand(arg), MovementOps.PERMUTE: lambda x, arg: x.permute(arg)
}}

class RawTorchBuffer(RawBuffer):
  def __init__(self, buf:torch.Tensor): super().__init__(prod(buf.shape), type_map[buf.dtype], buf)
  @classmethod
  def fromCPU(cls, x): return cls(torch.from_numpy(x).requires_grad_(False).to(device))
  def toCPU(self): return self._buf.cpu().numpy()
TorchBuffer = Interpreted(RawTorchBuffer, torch_fxn_for_op)
