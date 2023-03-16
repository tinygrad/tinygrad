import torch
from typing import Dict, Callable
from tinygrad.ops import UnaryOps, BinaryOps, MovementOps, FusedOps, Op, Interpreted
from tinygrad.helpers import getenv, dtypes, prod
from tinygrad.runtime.ops_cpu import base_fxn_for_op, einsum_mulacc
from tinygrad.runtime.lib import RawBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if getenv("MPS", 0) else "cpu"))

torch_fxn_for_op: Dict[Op, Callable] = {**base_fxn_for_op, **{
  UnaryOps.NOOP: lambda x: x.contiguous(), UnaryOps.EXP: lambda x: x.exp(), UnaryOps.LOG: lambda x: x.log(),
  BinaryOps.MAX: torch.maximum, BinaryOps.CMPEQ: lambda x,y: (x==y).float(),
  MovementOps.PAD: lambda x, padding: torch.nn.functional.pad(x, [item for sublist in padding[::-1] for item in sublist]),
  FusedOps.MULACC: einsum_mulacc(lambda s,a,b: torch.einsum(s, a.float(), b.float()).type(a.dtype), lambda x: x.stride(), lambda x,s: x.expand(s)),
  MovementOps.STRIDE: lambda x, arg: x[tuple(slice(None, None, abs(i)) for i in arg)].flip([i for i,a in enumerate(arg) if a < 0]),
  MovementOps.EXPAND: lambda x, arg: x.expand(arg), MovementOps.PERMUTE: lambda x, arg: x.permute(arg)
}}

class RawTorchBuffer(RawBuffer):
  def __init__(self, buf:torch.Tensor):
    super().__init__(prod(buf.shape), {torch.float16: dtypes.float16, torch.float32: dtypes.float32}[buf.dtype])
    self._buf = buf
  @classmethod
  def fromCPU(cls, x): return cls(torch.from_numpy(x).requires_grad_(False).to(device))
  def toCPU(self): return self._buf.cpu().numpy()
TorchBuffer = Interpreted(RawTorchBuffer, torch_fxn_for_op)

"""
class TorchBuffer(InterpretedBuffer):
  fxn_for_op: ClassVar = torch_fxn_for_op
  def to_tinygrad_dtype(self): return {torch.float16: dtypes.float16, torch.float32: dtypes.float32}[self._buf.dtype]
  def toCPU(self): return self._buf.cpu().numpy()
"""