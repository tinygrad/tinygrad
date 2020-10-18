# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from functools import partialmethod
import numpy as np

# **** start with three base classes ****

class Context:
  def __init__(self):
    self.saved_tensors = []

  def save_for_backward(self, *x):
    self.saved_tensors.extend(x)

class Tensor:
  def __init__(self, data, _children=()):
    self.data = data
    self.grad = np.zeros_like(data)

    # internal variables used for autograd graph construction
    self._prev = set(_children)

class Function:
  def apply(self, arg, *x):
    ctx = Context()
    x = [self]+list(x)
    ret = Tensor(arg.forward(ctx, *[t.data for t in x]))
    return ret

def register(name, fxn):
  setattr(Tensor, name, partialmethod(fxn.apply, fxn))

# **** implement a few functions ****
    
"""
class ReLU(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return np.maximum(input, 0)

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    grad_input = grad_output.clone()
    grad_input[input < 0] = 0
    return grad_input
setattr(Tensor, 'relu', partialmethod(run, ReLU))
"""

class Dot(Function):
  @staticmethod
  def forward(ctx, input, weight):
    ctx.save_for_backward(input, weight)
    return input.dot(weight)

  @staticmethod
  def backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    grad_input = grad_output.dot(weight.T)
    grad_weight = grad_output.dot(input)
    return grad_input, grad_weight
register('dot', Dot)

# may be wrong
class Sum(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return input.sum()

  @staticmethod
  def backward(ctx, grad_output):
    input = ctx.saved_tensors
    return grad_output * input
register('sum', Sum)

