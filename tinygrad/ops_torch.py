import torch
import numpy as np
from .tensor import Function

# ************* unary ops *************

class ReLU(Function):
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return input.relu()

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return grad_output * (input >= 0)

class Log(Function):
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return input.log()

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return grad_output / input

class Exp(Function):
  def forward(ctx, input):
    ret = input.exp()
    ctx.save_for_backward(ret)
    return ret

  def backward(ctx, grad_output):
    ret, = ctx.saved_tensors
    return grad_output * ret

# ************* binary ops *************

from tinygrad.ops_cpu import Add, Sub, Mul, unbroadcast

class Pow(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x ** y

  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    return unbroadcast(y * (x**(y-1.0)) * grad_output, x.shape), \
           unbroadcast((x**y) * torch.log(x) * grad_output, y.shape)

# ************* reduce ops *************

class Sum(Function):
  def forward(ctx, input, axis=None):
    ctx.save_for_backward(input, axis)
    return input.sum(axis) if axis != None else input.sum().reshape((1,))

  def backward(ctx, grad_output):
    input, axis = ctx.saved_tensors
    if isinstance(axis, int): axis = [axis]
    shape = [1 if axis is None or i in axis else input.shape[i] for i in range(len(input.shape))]
    return grad_output.reshape(shape) + torch.zeros_like(input)

class Max(Function):
  def forward(ctx, inp, axis=None):
    if isinstance(axis, int): axis = [axis]
    ret = torch.amax(inp, axis=None if axis is None else tuple(axis), keepdims=True)
    ctx.save_for_backward(inp, axis, ret)
    if axis is not None:
      ret = ret.reshape([inp.shape[i] for i in range(len(inp.shape)) if i not in axis])
    return ret

  def backward(ctx, grad_output):
    input, axis, ret = ctx.saved_tensors
    shape = [1 if axis is None or i in axis else input.shape[i] for i in range(len(input.shape))]
    ret2 = (input==ret.reshape(shape))
    div = ret2.sum(axis=tuple(axis), keepdims=True) if axis is not None else ret2.sum()
    return ret2*grad_output.reshape(shape)/div

# ************* movement ops *************

from tinygrad.ops_cpu import Reshape

class Transpose(Function):
  def forward(ctx, x, order):
    ctx.save_for_backward(order)
    return x.permute(order)

  def backward(ctx, x):
    return x.permute(tuple(np.argsort(ctx.order)))

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
