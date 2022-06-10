import torch
import numpy as np
from tinygrad.helpers import get_conv_args

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

from tinygrad.llops.ops_cpu import unary_op, binary_op, reduce_op, reshape, perm_axis, inner_slice, matmul

# ************* processing ops *************

def conv(x,w,ret,stride,groups):
  ret[:] = torch.nn.functional.conv2d(x, w, stride=stride, groups=groups)
  return ret

def convdw(input,grad_output,dw,stride,groups):
  # NOTE: torch.nn.grad.conv2d_weight is wrong for groups in pytorch, wonder who it affects 
  # https://github.com/pytorch/pytorch/issues/51430
  C = get_conv_args(input.shape, dw.shape, stride, groups)
  grad_output = grad_output.reshape(C.bs, C.groups, C.rcout, C.oy, C.ox).repeat(1, 1, C.cin, 1, 1)
  grad_output = grad_output.reshape(C.bs * C.groups * C.rcout * C.cin, 1, C.oy, C.ox)
  input = input.reshape(1, C.bs * C.groups * C.cin, C.iy, C.ix)
  grad_weight = torch.nn.functional.conv2d(input, grad_output, dilation=stride, groups=C.bs*C.groups*C.cin)
  grad_weight = grad_weight.reshape(C.bs, grad_weight.shape[1] // C.bs, *grad_weight.shape[2:]).sum(dim=0)
  grad_weight = grad_weight.view(C.groups, C.cin, C.rcout, *grad_weight.shape[1:]).transpose(2, 1)
  # narrow removes excess for strided
  dw[:] = grad_weight.contiguous().view(C.groups*C.rcout, C.cin, *grad_weight.shape[3:]).narrow(
            2, 0, dw.shape[2]).narrow(3, 0, dw.shape[3])
  return dw

def convdx(w,grad_output,dx,stride,groups):
  dx[:] = torch.nn.grad.conv2d_input(dx.shape, w, grad_output, stride=stride, groups=groups)
  return dx
