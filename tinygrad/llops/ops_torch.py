import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class TorchBuffer(torch.Tensor):
  def __new__(cls, shape):
    if isinstance(shape, torch.Tensor):
      return super().__new__(cls, shape)
    else:
      return TorchBuffer(torch.zeros(shape)).to(device)
  custompad = lambda x,padding: torch.nn.functional.pad(x, [item for sublist in padding[::-1] for item in sublist])
  @staticmethod
  def fromCPU(data):
    return TorchBuffer(torch.from_numpy(data).requires_grad_(False)).to(device)
  def toCPU(x):
    return x.cpu().numpy()

# ************* unary+binary+reduce+movement ops *************

from tinygrad.llops.ops_cpu import unary_op, binary_op, reduce_op, movement_op

# ************* processing ops *************

from tinygrad.ops import ProcessingOps

def processing_op(op,x,w,C):
  assert op == ProcessingOps.CONV, f"{op} isn't supported"
  return torch.conv2d(x, w, stride=(C.ys, C.xs), groups=C.groups, dilation=(C.dy, C.dx), padding=(C.py, C.px))
