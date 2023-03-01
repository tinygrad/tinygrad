import torch
from typing import ClassVar, Final, Dict, Callable
from tinygrad.ops import UnaryOps, BinaryOps, MovementOps, ProcessingOps, FusedOps, GenericExecAST, Op
from tinygrad.helpers import getenv
from tinygrad.llops.ops_cpu import base_fxn_for_op, einsum_mulacc

torch_fxn_for_op : Dict[Op, Callable] = {**base_fxn_for_op, **{
  UnaryOps.EXP: lambda x: x.exp(), UnaryOps.LOG: lambda x: x.log(),
  BinaryOps.MAX: torch.maximum, BinaryOps.CMPEQ: lambda x,y: (x==y).float(),
  MovementOps.PAD: lambda x, padding: torch.nn.functional.pad(x, [item for sublist in padding[::-1] for item in sublist]),
  MovementOps.STRIDED: lambda x, arg: x.contiguous().as_strided([y[0] for y in arg], [y[1] for y in arg]),
  ProcessingOps.CONV: lambda x,w,C: C.px == C.px_ and C.py == C.py_ and torch.conv2d(x, w, stride=(C.sy, C.sx), groups=C.groups, dilation=(C.dy, C.dx), padding=(C.py, C.px)),
  FusedOps.MULACC: einsum_mulacc(torch.einsum, lambda x: x.stride(), lambda x,s: x.expand(s))
}}

device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if getenv("MPS", 0) else "cpu"))
class TorchBuffer(GenericExecAST):
  fxn_for_op : ClassVar = torch_fxn_for_op
  SUPPORTS_SIMPLE_PADDING : Final = True

  @staticmethod
  def fromCPU(data): return TorchBuffer(torch.from_numpy(data).requires_grad_(False).to(device))
  def toCPU(x): return x.buf.cpu().numpy()
