import torch
import numpy as np

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
  def getdtype(self):
    return np.float32

# ************* unary+binary+reduce+movement ops *************

from tinygrad.llops.ops_cpu import unary_op, binary_op, reduce_op, movement_op

# ************* processing ops *************

from tinygrad.ops import ProcessingOps

def processing_op(op,x,w,ret,C):
  stride, groups, dilation, padding = (C.ys, C.xs), C.groups, (C.dy, C.dx), (C.py, C.px)
  # stride is the same as doing the full conv and slicing with stride at the end
  # dilation is the same as conving with a weight matrix with 0s added
  if op == ProcessingOps.CONV:
    ret[:] = torch.conv2d(x, w, stride=stride, groups=groups, dilation=dilation, padding=padding)
  elif op == ProcessingOps.CONVT:
    if stride == (1,1):
      # strided needs weird "unstride": https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
      # it's 0 insertion between the inputs
      w = w.reshape(C.groups, C.rcout, C.cin, C.H, C.W).flip(3, 4).transpose(2, 1).reshape(C.groups*C.cin, C.rcout, C.H, C.W)
      ret[:] = torch.conv2d(x, w, dilation=dilation, padding=((C.H-1)*C.dy-C.py,(C.W-1)*C.dx-C.px), groups=groups)
    else:
      output_padding = [ret.shape[d+2] - ((x.shape[d+2] - padding[d]*2 - 1) * stride[d] + 1 + dilation[d] * (w.shape[d+2] - 1)) for d in range(2)]
      ret[:] = torch.conv_transpose2d(x, w, padding=padding, stride=stride, groups=groups, output_padding=output_padding, dilation=dilation)
