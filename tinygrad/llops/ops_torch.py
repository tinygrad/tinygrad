import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class TorchBuffer(torch.Tensor):
  def __new__(cls, shape):
    if isinstance(shape, torch.Tensor):
      return super().__new__(cls, shape)
    else:
      return TorchBuffer(torch.zeros(shape))
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

from tinygrad.helpers import get_conv_args, ProcessingOps

def conv(x,w,ret,stride,groups):
  ret[:] = torch.conv2d(x, w, stride=stride, groups=groups)
  return ret

def convdw(x,grad_output,dw,stride,groups):
  # NOTE: torch.nn.grad.conv2d_weight is wrong for groups in pytorch, wonder who it affects 
  # https://github.com/pytorch/pytorch/issues/51430
  C = get_conv_args(x.shape, dw.shape, stride, groups)
  grad_output = grad_output.reshape(C.bs, C.groups, C.rcout, C.oy, C.ox).repeat(1, 1, C.cin, 1, 1)
  grad_output = grad_output.reshape(C.bs * C.groups * C.rcout * C.cin, 1, C.oy, C.ox)
  x = x.reshape(1, C.bs * C.groups * C.cin, C.iy, C.ix)
  #print(input.shape, grad_output.shape)
  grad_weight = torch.conv2d(x, grad_output, dilation=stride, groups=C.bs*C.groups*C.cin)
  grad_weight = grad_weight.reshape(C.bs, grad_weight.shape[1] // C.bs, *grad_weight.shape[2:]).sum(dim=0)
  grad_weight = grad_weight.view(C.groups, C.cin, C.rcout, *grad_weight.shape[1:]).transpose(2, 1)
  # narrow removes excess for strided
  dw[:] = grad_weight.contiguous().view(C.groups*C.rcout, C.cin, *grad_weight.shape[3:]).narrow(
            2, 0, dw.shape[2]).narrow(3, 0, dw.shape[3])
  return dw

def convdx(grad_output,w,dx,stride,groups):
  dx[:] = torch.nn.grad.conv2d_input(dx.shape, w, grad_output, stride=stride, groups=groups)
  # correct for non strided
  # strided needs weird padding: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
  #C = get_conv_args(dx.shape, w.shape, stride, groups)
  #w = w.reshape(C.groups, C.rcout, C.cin, C.H, C.W).flip(3, 4).transpose(2, 1).reshape(C.groups*C.cin, C.rcout, C.H, C.W)
  #ret = torch.conv2d(grad_output, w, padding=(C.H-1,C.W-1), groups=groups)
  return dx

def processing_op(op,a,b,ret,stride,groups):
  if op == ProcessingOps.CONV: conv(a,b,ret,stride,groups)
  elif op == ProcessingOps.CONVT: convdx(a,b,ret,stride,groups)
  elif op == ProcessingOps.CONVDW: convdw(a,b,ret,stride,groups)
  return ret
