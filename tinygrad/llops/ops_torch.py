import torch
from typing import Final
from tinygrad.ops import UnaryOps, BinaryOps, MovementOps, ProcessingOps, GenericBufExecAST, base_fxn_for_op
from tinygrad.helpers import getenv

specialized_fxn_for_op = (lambda d: d.update(base_fxn_for_op) or d)({
  UnaryOps.RELU: lambda x: x.relu(), UnaryOps.EXP: lambda x: x.exp(), UnaryOps.LOG: lambda x: x.log(), BinaryOps.CMPEQ: lambda x,y: (x==y).float(),
  MovementOps.PAD: lambda x, padding: torch.nn.functional.pad(x, [item for sublist in padding[::-1] for item in sublist]),
  MovementOps.STRIDED: lambda x, arg: x.contiguous().as_strided([y[0] for y in arg], [y[1] for y in arg])
})

device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if getenv("MPS", 0) else "cpu"))
class TorchBuffer(GenericBufExecAST):
  fxn_for_op : Final = specialized_fxn_for_op
  def __init__(self, lbuf:torch.Tensor): self.buf, self.shape = lbuf, tuple(lbuf.shape)

  @staticmethod
  def fromCPU(data): return TorchBuffer(torch.from_numpy(data).requires_grad_(False).to(device))
  def toCPU(x): return x.buf.cpu().numpy()

  SUPPORTS_SIMPLE_PADDING = True
  def processing_op(x,op,w,C):
    assert op == ProcessingOps.CONV, f"{op} isn't supported"
    assert C.px == C.px_ and C.py == C.py_, "asymmetric padding in conv is not supported"
    return TorchBuffer(torch.conv2d(x.buf, w.buf, stride=(C.sy, C.sx), groups=C.groups, dilation=(C.dy, C.dx), padding=(C.py, C.px)))
