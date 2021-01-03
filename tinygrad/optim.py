# sorted in order of increasing complexity
from tinygrad.tensor import Tensor

class Optimizer:
  def __init__(self, params):
    self.params = [x for x in params if x.requires_grad]

  def zero_grad(self):
    for param in self.params:
      param.grad = None

class SGD(Optimizer):
  def __init__(self, params, lr=0.001):
    super().__init__(params)
    self.lr = lr

  def step(self):
    for t in self.params:
      t -= t.grad * self.lr

class RMSprop(Optimizer):
  def __init__(self, params, lr=0.001, decay=0.9, eps=1e-8):
    super().__init__(params)
    self.lr, self.decay, self.eps = lr, decay, eps

    self.v = [Tensor.zeros(*t.shape, device=params[0].device, requires_grad=False) for t in self.params]

  def step(self):
    for i, t in enumerate(self.params):
      self.v[i] = self.decay * self.v[i] + (1.0 - self.decay) * t.grad * t.grad
      t -= (t.grad * self.lr).div(self.v[i].sqrt() + self.eps)

class Adam(Optimizer):
  def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    super().__init__(params)
    self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0

    self.m = [Tensor.zeros(*t.shape, device=params[0].device, requires_grad=False) for t in self.params]
    self.v = [Tensor.zeros(*t.shape, device=params[0].device, requires_grad=False) for t in self.params]

  def step(self):
    self.t = self.t + 1
    a = self.lr * ((1.0 - self.b2**self.t)**0.5) / (1.0 - self.b1**self.t)
    for i, t in enumerate(self.params):
      self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * t.grad
      self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * t.grad * t.grad
      t -= a * self.m[i].div(self.v[i].sqrt() + self.eps)
