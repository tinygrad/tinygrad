import torch
import numpy as np
from .tensor import Function

# ************* unary+binary+reduce ops *************

from tinygrad.ops_cpu import ReLU, Log, Exp, Add, Sub, Mul, Pow, Sum, Max

# ************* movement ops *************

from tinygrad.ops_cpu import Reshape, Transpose

def inner_slice(x, arg):
  padding = [(max(0, -p[0]), max(0, p[1]-x.shape[i])) for i,p in enumerate(arg)]
  x = torch.nn.functional.pad(x, [item for sublist in padding[::-1] for item in sublist])
  slicee = [(p[0] + padding[i][0], p[1] + padding[i][0]) for i,p in enumerate(arg)]
  return x[tuple([slice(x[0], x[1], None) for x in slicee])]

class Slice(Function):
  def forward(ctx, x, arg=None):
    ctx.save_for_backward(x.shape)
    return inner_slice(x, arg)

  def backward(ctx, grad_output):
    shape, = ctx.saved_tensors
    narg = [(0-p[0], grad_output.shape[i]+(shape[i]-p[1])) for i,p in enumerate(ctx.arg)]
    return inner_slice(grad_output, narg)

# ************* processing ops *************

from tinygrad.ops_cpu import Matmul

class Conv2D(Function):
  def forward(ctx, x, w, stride=1, groups=1):
    ctx.save_for_backward(x, w, stride, groups)
    return torch.nn.functional.conv2d(x, w, stride=stride, groups=groups)

  def backward(ctx, grad_output):
    x, w, stride, groups = ctx.saved_tensors
    grad_input = torch.nn.grad.conv2d_input(x.shape, w, grad_output, stride=stride, groups=groups)
    grad_weight = torch.nn.grad.conv2d_weight(x, w.shape, grad_output, stride=stride, groups=groups)
    return grad_input, grad_weight
