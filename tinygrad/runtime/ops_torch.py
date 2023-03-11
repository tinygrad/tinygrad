import torch
from typing import ClassVar, Dict, Callable
from tinygrad.ops import UnaryOps, BinaryOps, MovementOps, FusedOps, InterpretedBuffer, Op
from tinygrad.helpers import getenv, dtypes
from tinygrad.runtime.ops_cpu import base_fxn_for_op, einsum_mulacc

torch_fxn_for_op: Dict[Op, Callable] = {**base_fxn_for_op, **{
  UnaryOps.NOOP: lambda x: x.contiguous(), UnaryOps.EXP: lambda x: x.exp(), UnaryOps.LOG: lambda x: x.log(),
  BinaryOps.MAX: torch.maximum, BinaryOps.CMPEQ: lambda x,y: (x==y).float(),
  MovementOps.PAD: lambda x, padding: torch.nn.functional.pad(x, [item for sublist in padding[::-1] for item in sublist]),
  FusedOps.MULACC: einsum_mulacc(lambda s,a,b: torch.einsum(s, a.float(), b.float()).type(a.dtype), lambda x: x.stride(), lambda x,s: x.expand(s)),
  MovementOps.STRIDE: lambda x, arg: x.__getitem__(tuple(slice(None, None, abs(i)) for i in arg)).flip([i for i,a in enumerate(arg) if a < 0])
}}

device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if getenv("MPS", 0) else "cpu"))
class TorchBuffer(InterpretedBuffer):
  fxn_for_op: ClassVar = torch_fxn_for_op
  to_tinygrad_dtype = staticmethod(lambda lbuf: {torch.float16: dtypes.float16, torch.float32: dtypes.float32}[lbuf.dtype])

  @staticmethod
  def fromCPU(x): return TorchBuffer(torch.from_numpy(x).requires_grad_(False).to(device))
  def toCPU(self): return self._buf.cpu().numpy()
