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

def convdw(x,grad_output,dw,C):
  # NOTE: torch.nn.grad.conv2d_weight is wrong for groups in pytorch, wonder who it affects 
  # https://github.com/pytorch/pytorch/issues/51430
  grad_output = grad_output.reshape(C.bs * C.groups, C.rcout, C.oy, C.ox).repeat(1, C.cin, 1, 1)
  grad_output = grad_output.reshape(C.bs * C.groups * C.cin * C.rcout, 1, C.oy, C.ox)
  x = x.reshape(1, C.bs * C.groups * C.cin, C.iy, C.ix)
  # NOTE: this conv2d always has batch size 1.
  grad_weight = torch.conv2d(x, grad_output, dilation=(C.ys, C.xs), groups=C.bs*C.groups*C.cin)
  grad_weight = grad_weight.reshape(C.bs, C.groups, C.cin, C.rcout, *grad_weight.shape[2:]).transpose(3, 2).sum(dim=0)
  dw[:] = grad_weight.reshape(C.groups*C.rcout, C.cin, *grad_weight.shape[3:])[:, :, :dw.shape[2], :dw.shape[3]]

def processing_op(op,x,w,ret,C):
  stride, groups = (C.ys, C.xs), C.groups
  if op == ProcessingOps.CONV:
    ret[:] = torch.conv2d(x, w, stride=stride, groups=groups)
  elif op == ProcessingOps.CONVT:
    if stride == 1 or stride == (1,1):
      # strided needs weird "undilation": https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
      w = w.reshape(C.groups, C.rcout, C.cin, C.H, C.W).flip(3, 4).transpose(2, 1).reshape(C.groups*C.cin, C.rcout, C.H, C.W)
      ret[:] = torch.conv2d(x, w, padding=(C.H-1,C.W-1), groups=groups)
    else:
      output_padding = [ret.shape[d+2] - ((x.shape[d+2] - 1) * stride[d] + 1 + (w.shape[d+2] - 1)) for d in range(2)]
      ret[:] = torch.conv_transpose2d(x, w, stride=stride, groups=groups, output_padding=output_padding)
  elif op == ProcessingOps.CONVDW:
    convdw(x,w,ret,C)
