import torch
import operator
from typing import Tuple
from tinygrad.llops.ops_cpu import CPUBuffer  # type: ignore
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps, GenericExecAST
from tinygrad.helpers import getenv, shape_to_axis

device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if getenv("MPS", 0) else "cpu"))
class TorchBuffer(GenericExecAST):
  fxn_for_op = {
    UnaryOps.NOOP: lambda x: x[:].contiguous(), UnaryOps.NEG: lambda x: -x, UnaryOps.RELU: lambda x: x.relu(),
    UnaryOps.EXP: lambda x: x.exp(), UnaryOps.LOG: lambda x: x.log(), UnaryOps.GT0: lambda x: operator.gt(x, 0.0), UnaryOps.RECIPROCAL: lambda x: 1.0/x,
    BinaryOps.ADD: operator.add, BinaryOps.SUB: operator.sub, BinaryOps.MUL: operator.mul,
    BinaryOps.DIV: operator.truediv, BinaryOps.POW: operator.pow, BinaryOps.CMPEQ: lambda x,y: (x==y).float(),
    ReduceOps.SUM: lambda x, new_shape: x.sum(shape_to_axis(x.shape, new_shape), keepdims=True) if tuple(x.shape) != tuple(new_shape) else x[:],
    ReduceOps.MAX: lambda x, new_shape: x.amax(shape_to_axis(x.shape, new_shape), keepdims=True) if tuple(x.shape) != tuple(new_shape) else x[:],
    MovementOps.SHRINK: lambda x, arg: x[tuple(slice(p[0], p[1], None) for p in arg)],
    MovementOps.PAD: lambda x, padding: torch.nn.functional.pad(x, [item for sublist in padding[::-1] for item in sublist]),
    MovementOps.STRIDED: lambda x, arg: x.contiguous().as_strided([y[0] for y in arg], [y[1] for y in arg])
  }

  def __init__(self, lbuf:torch.Tensor): self.buf = lbuf
  @property
  def shape(self) -> Tuple[int, ...]: return self.buf.shape

  @staticmethod
  def fromCPU(data): return TorchBuffer(torch.from_numpy(data).requires_grad_(False).to(device))
  def toCPU(x): return x.buf.cpu().numpy()

  contiguous, unary_op, binary_op, reduce_op, movement_op = CPUBuffer.contiguous, CPUBuffer.unary_op, CPUBuffer.binary_op, CPUBuffer.reduce_op, CPUBuffer.movement_op

  SUPPORTS_SIMPLE_PADDING = True
  def processing_op(x,op,w,C):
    assert op == ProcessingOps.CONV, f"{op} isn't supported"
    assert C.px == C.px_ and C.py == C.py_, "asymmetric padding in conv is not supported"
    return TorchBuffer(torch.conv2d(x.buf, w.buf, stride=(C.sy, C.sx), groups=C.groups, dilation=(C.dy, C.dx), padding=(C.py, C.px)))
