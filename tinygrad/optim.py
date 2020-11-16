# sorted in order of increasing complexity

import numpy as np
from tinygrad.tensor import Tensor

class Optimizer:
  def __init__(self, params):
    self.params = params

  def zero_grad(self):
    for param in self.params:
      param.grad = None

class SGD(Optimizer):
  def __init__(self, params, lr=0.001):
    super(SGD, self).__init__(params)
    self.lr = Tensor([lr], gpu=params[0].gpu)

  def step(self):
    for t in self.params:
      t -= t.grad * self.lr

class RMSprop(Optimizer):
  def __init__(self, params, lr=0.001, decay=0.9, eps=1e-8):
    super(RMSprop, self).__init__(params)
    self.lr = Tensor([lr], gpu=params[0].gpu)
    self.decay = Tensor([decay], gpu=params[0].gpu)
    self.eps = Tensor([eps], gpu=params[0].gpu)

    self.v = [Tensor(np.zeros(t.shape, dtype=np.float32), gpu=params[0].gpu) for t in self.params]

    self.one = Tensor([1], gpu=self.params[0].gpu)
    self.two = Tensor([2], gpu=self.params[0].gpu)

  def step(self):
    for i, t in enumerate(self.params):
      self.v[i] = self.decay * self.v[i] + (self.one - self.decay) * t.grad.pow(self.two)
      t -= self.lr.div(self.v[i].sqrt() + self.eps) * t.grad

class Adam(Optimizer):
  def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    super(Adam, self).__init__(params)
    self.lr = Tensor([lr], gpu=params[0].gpu)
    self.b1 = Tensor([b1], gpu=params[0].gpu)
    self.b2 = Tensor([b2], gpu=params[0].gpu)
    self.eps = Tensor([eps], gpu=params[0].gpu)
    self.t = Tensor([0], gpu=params[0].gpu)

    self.m = [Tensor(np.zeros(t.shape, dtype=np.float32), gpu=params[0].gpu) for t in self.params]
    self.v = [Tensor(np.zeros(t.shape, dtype=np.float32), gpu=params[0].gpu) for t in self.params]

    self.one = Tensor([1], gpu=self.params[0].gpu)
    self.two = Tensor([2], gpu=self.params[0].gpu)

  def step(self):
    self.t = self.t + self.one
    a = self.lr * (self.one - self.b2.pow(self.t)).sqrt().div(self.one - self.b1.pow(self.t))
    for i,t in enumerate(self.params):
      self.m[i] = self.b1 * self.m[i] + (self.one - self.b1) * t.grad
      self.v[i] = self.b2 * self.v[i] + (self.one - self.b2) * t.grad.pow(self.two)
      t -= a * self.m[i].div(self.v[i].sqrt() + self.eps)
