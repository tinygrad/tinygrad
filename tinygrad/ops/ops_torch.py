import torch
import numpy as np
from ..tensor import Function

class TorchBuffer(torch.Tensor):
  def custompad(x, padding):
    return torch.nn.functional.pad(x, [item for sublist in padding[::-1] for item in sublist])
  @staticmethod
  def fromCPU(data):
    return TorchBuffer(torch.from_numpy(data).requires_grad_(False))
  def toCPU(x):
    return x.numpy()
  def getdtype(self):
    return np.float32

# ************* unary+binary+reduce+movement ops *************

from tinygrad.ops.ops_cpu import ReLU, Log, Exp, Add, Sub, Mul, Pow, Sum, Max, Reshape, Transpose, Slice

# ************* processing ops *************

from tinygrad.ops.ops_cpu import Matmul

class Conv2D(Function):
  def forward(ctx, x, w, stride=1, groups=1):
    ctx.save_for_backward(x, w, stride, groups)
    return torch.nn.functional.conv2d(x, w, stride=stride, groups=groups)

  def backward(ctx, grad_output):
    x, w, stride, groups = ctx.saved_tensors
    grad_input = torch.nn.grad.conv2d_input(x.shape, w, grad_output, stride=stride, groups=groups)
    grad_weight = torch.nn.grad.conv2d_weight(x, w.shape, grad_output, stride=stride, groups=groups)
    return grad_input, grad_weight
