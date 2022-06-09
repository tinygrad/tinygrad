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

from tinygrad.llops.ops_cpu import unary_op, binary_op, reduce_op, reshape, perm_axis, inner_slice, matmul

# ************* processing ops *************

def conv(x,w,ret,C):
  ret[:] = torch.nn.functional.conv2d(x, w, stride=(C.ys,C.xs), groups=C.groups)
  return ret

def convdw(x,grad_output,dw,C):
  dw[:] = torch.nn.grad.conv2d_weight(x, dw.shape, grad_output, stride=(C.ys,C.xs), groups=C.groups)
  return dw

def convdx(w,grad_output,dx,C):
  dx[:] = torch.nn.grad.conv2d_input(dx.shape, w, grad_output, stride=(C.ys,C.xs), groups=C.groups)
  return dx
