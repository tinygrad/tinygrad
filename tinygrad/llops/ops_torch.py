import torch
from tinygrad.llops.ops_cpu import CPUBuffer  # type: ignore
from tinygrad.ops import ProcessingOps

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class TorchBuffer(torch.Tensor):
  def pad(x, padding): return torch.nn.functional.pad(x, [item for sublist in padding[::-1] for item in sublist])

  @staticmethod
  def fromCPU(data): return TorchBuffer(torch.from_numpy(data).requires_grad_(False)).to(device)
  def toCPU(x): return x.cpu().numpy()

  unary_op, binary_op, reduce_op, movement_op = CPUBuffer.unary_op, CPUBuffer.binary_op, CPUBuffer.reduce_op, CPUBuffer.movement_op

  def processing_op(x,op,w,C):
    assert op == ProcessingOps.CONV, f"{op} isn't supported"
    return torch.conv2d(x, w, stride=(C.sy, C.sx), groups=C.groups, dilation=(C.dy, C.dx))
