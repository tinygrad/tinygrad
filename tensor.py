# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from functools import partialmethod
import numpy as np

# **** start with three base classes ****

class Context:
  def __init__(self, arg, *tensors):
    self.arg = arg
    self.parents = tensors
    self.saved_tensors = []

  def save_for_backward(self, *x):
    self.saved_tensors.extend(x)

class Tensor:
  def __init__(self, data, _children=()):
    self.data = data
    self.grad = None

    # internal variables used for autograd graph construction
    self._ctx = None

  def __str__(self):
    return "Tensor of shape %r with grad %r" % (self.data.shape, self.grad)

  def backward(self, allow_fill=True):
    #print("running backward on", self)
    if self._ctx is None:
      return

    if self.grad is None and allow_fill:
      # fill in the first grad with one
      assert self.data.size == 1
      self.grad = np.ones_like(self.data)

    assert(self.grad is not None)

    grads = self._ctx.arg.backward(self._ctx, self.grad)
    for t,g in zip(self._ctx.parents, grads):
      t.grad = g
      t.backward(False)

class Function:
  def apply(self, arg, *x):
    ctx = Context(arg, self, *x)
    ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
    ret._ctx = ctx
    return ret

def register(name, fxn):
  setattr(Tensor, name, partialmethod(fxn.apply, fxn))

# **** implement a few functions ****
    
class ReLU(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return np.maximum(input, 0)

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    grad_input = grad_output.copy()
    grad_input[input < 0] = 0
    return grad_input,
register('relu', ReLU)

class Dot(Function):
  @staticmethod
  def forward(ctx, input, weight):
    ctx.save_for_backward(input, weight)
    return input.dot(weight)

  @staticmethod
  def backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    grad_input = grad_output.dot(weight.T)
    grad_weight = grad_output.T.dot(input).T
    return grad_input, grad_weight
register('dot', Dot)

class Sum(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return input.sum()

  @staticmethod
  def backward(ctx, grad_output):
    input = ctx.saved_tensors
    return grad_output * np.ones_like(input)
register('sum', Sum)

