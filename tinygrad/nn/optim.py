# sorted in order of increasing complexity
from typing import List
from tinygrad.helpers import dedup, flatten, getenv
from tinygrad.tensor import Tensor

class Optimizer:
  def __init__(self, params: List[Tensor], lr: float):
    # if it's None, but being put into an optimizer, set it to True
    for x in params:
      if x.requires_grad is None: x.requires_grad = True

    self.params: List[Tensor] = dedup([x for x in params if x.requires_grad])
    assert len(self.params) != 0, "optimizer must have at least one param"
    self.device = self.params[0].device
    self.buffers: List[Tensor] = dedup([x for x in params if not x.requires_grad])   # buffers are still realized
    self.lr = lr if getenv("CONST_LR") else Tensor([lr], requires_grad=False, device=self.device).contiguous()

  def zero_grad(self):
    for param in self.params: param.grad = None

  def step(self): Tensor.corealize(self.schedule_step())
  def schedule_step(self) -> List[Tensor]: return self._step()+self.params+self.buffers
  def _step(self) -> List[Tensor]: raise NotImplementedError

class OptimizerGroup(Optimizer):
  def __init__(self, *optimizers: Optimizer): # pylint: disable=super-init-not-called
    self.optimizers = optimizers
    self.params, self.buffers = flatten([o.params for o in self.optimizers]), flatten([o.buffers for o in self.optimizers])
  def __getitem__(self, i): return self.optimizers[i]
  def zero_grad(self): [o.zero_grad() for o in self.optimizers]
  def _step(self) -> List[Tensor]: return [x for o in self.optimizers for x in o._step()]

# LARS is essentially just trust ratio to SGD so if we just set the trust coeff 0.0 its just standard SGD.
def SGD(params: List[Tensor], lr=0.001, momentum=0.0, weight_decay=0.0, nesterov=False, classic=False):
  return LARS(params, lr, momentum, weight_decay, nesterov, classic, tcoef=0.0)

class LARS(Optimizer):
  def __init__(self, params:List[Tensor], lr=0.001, momentum=0.9, weight_decay=1e-4, nesterov=False, classic=True, tcoef=0.001):
    super().__init__(params, lr)
    self.momentum, self.wd, self.nesterov, self.classic, self.tcoef = momentum, weight_decay, nesterov, classic, tcoef
    self.b = [Tensor.zeros(*t.shape, dtype=t.dtype, device=t.device, requires_grad=False) for t in self.params] if self.momentum else []

  def _step(self) -> List[Tensor]:
    for i, t in enumerate(self.params):
      assert t.grad is not None
      # contiguous is needed since the grads can allegedly form a "diamond"
      # TODO: fix this in lazy.py
      g = t.grad.contiguous()
      if self.tcoef != 0:
        r1 = t.detach().square().sum().sqrt()
        r2 = g.square().sum().sqrt()
        r = (r1 > 0).where((r2 > 0).where(self.tcoef * r1 / (r2 + self.wd * r1), 1.0), 1.0)
      else: r = 1.0
      g = g + self.wd * t.detach()
      # classic momentum does post learning rate update
      if self.classic: g = g * r * self.lr
      if self.momentum:
        self.b[i].assign(self.momentum * self.b[i] + g)  # NOTE: self.b[i] is zero on the first run, no if required
        g = (g + self.momentum * self.b[i]) if self.nesterov else self.b[i]
      # popular momentum does pre learning rate update
      if not self.classic: g = g * r * self.lr
      t.assign(t.detach() - g)
    return self.b

# LAMB is essentially just the trust ratio part of LARS applied to Adam/W so if we just set the trust ratio to 1.0 its just Adam/W.
def AdamW(params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, wd=0.01): return LAMB(params, lr, b1, b2, eps, wd, adam=True)
def Adam(params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8): return LAMB(params, lr, b1, b2, eps, 0.0, adam=True)

class LAMB(Optimizer):
  def __init__(self, params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-6, wd=0.0, adam=False):
    super().__init__(params, lr)
    self.eps, self.wd, self.adam = eps, wd, adam
    self.b1, self.b2, self.t = (Tensor([x], device=self.device, requires_grad=False).realize() for x in [b1, b2, 0])
    self.m = [Tensor.zeros(*t.shape, dtype=t.dtype, device=t.device, requires_grad=False).contiguous() for t in self.params]
    self.v = [Tensor.zeros(*t.shape, dtype=t.dtype, device=t.device, requires_grad=False).contiguous() for t in self.params]

  def _step(self) -> List[Tensor]:
    self.t.assign(self.t + 1)
    for i, t in enumerate(self.params):
      assert t.grad is not None
      self.m[i].assign(self.b1 * self.m[i] + (1.0 - self.b1) * t.grad)
      self.v[i].assign(self.b2 * self.v[i] + (1.0 - self.b2) * (t.grad * t.grad))
      m_hat = self.m[i] / (1.0 - self.b1 ** self.t)
      v_hat = self.v[i] / (1.0 - self.b2 ** self.t)
      up = (m_hat / (v_hat.sqrt() + self.eps)) + self.wd * t.detach()
      if not self.adam:
        r1 = t.detach().square().sum().sqrt()
        r2 = up.square().sum().sqrt()
        r = Tensor.where(r1 > 0, Tensor.where(r2 > 0, r1 / r2, 1.0), 1.0)
      else:
        r = 1.0
      t.assign(t.detach() - self.lr * r * up)
    return [self.t] + self.m + self.v
