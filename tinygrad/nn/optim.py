# sorted in order of increasing complexity
from typing import List, Union, Dict
from tinygrad.helpers import dedup
from tinygrad.tensor import Tensor

class Optimizer:
  def __init__(self, params: List[Union[Dict, Tensor]],  lr: float):
    self.lr = Tensor([lr], requires_grad=False)

    self.param_groups: List[Dict] = []
    param_groups = list(params)
    if not param_groups: param_groups = [{'params': []}]
    elif not isinstance(param_groups[0], dict): param_groups = [{'params': param_groups}]
    for param_group in param_groups: self.add_param_group(param_group)

  def zero_grad(self):
    for param_group in self.param_groups:
      for param in param_group['params']: param.grad = None

  def add_param_group(self, param_group):
    # if it's None, but being put into an optimizer, set it to True
    for x in param_group['params']:
      if x.requires_grad is None: x.requires_grad = True

    param_group['params'] = dedup([x for x in param_group['params'] if x.requires_grad]) if param_group.get('params') else []
    param_group['buffers'] = dedup([x for x in param_group['params'] if not x.requires_grad]) if param_group.get('params') else []
    param_group['lr'] = self.lr
    self.param_groups.append(param_group)

  def realize(self, params):
    # TODO: corealize
    # NOTE: in extra is too late for most of the params due to issues with assign
    for p in params: p.realize()

class SGD(Optimizer):
  def __init__(self, params: List[Union[Dict, Tensor]], lr=0.001, momentum=0, weight_decay=0.0, nesterov=False):
    super().__init__(params, lr)
    self.momentum, self.wd, self.nesterov = momentum, weight_decay, nesterov
    for param_group in self.param_groups:
      param_group['b'] = [Tensor.zeros(*t.shape, device=t.device, requires_grad=False) for t in param_group['params']] if param_group.get('momentum', self.momentum) else []

  # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
  def step(self) -> None:
    for param_group in self.param_groups:
      for i, t in enumerate(param_group['params']):
        assert t.grad is not None
        g = t.grad.realize() + param_group.get('wd', self.wd) * t.detach()
        if param_group.get('momentum', self.momentum):
          param_group['b'][i].assign(self.momentum * param_group['b'][i] + g).realize()  # NOTE: param_group['b'][i] is zero on the first run, no if required
          g = (g + param_group.get('momentum', self.momentum) * param_group['b'][i]) if param_group.get('nesterov', self.nesterov) else param_group['b'][i]
        t.assign(t.detach() - g * param_group.get('lr', self.lr))
      self.realize(param_group['params'] + param_group['buffers'] + param_group['b'])

class LAMB(Optimizer):
  def __init__(self, params: List[Union[Dict, Tensor]], lr=0.001, b1=0.9, b2=0.999, eps=1e-6, wd=0.0, adam=False):
    super().__init__(params, lr)
    self.b1, self.b2, self.eps, self.wd, self.adam, self.t = b1, b2, eps, wd, adam, Tensor([0], requires_grad=False).realize()
    for param_group in self.param_groups:
      param_group['m'] = [Tensor.zeros(*t.shape, device=t.device, requires_grad=False) for t in param_group['params']]
      param_group['v'] = [Tensor.zeros(*t.shape, device=t.device, requires_grad=False) for t in param_group['params']]

  def step(self) -> None:
    self.t.assign(self.t + 1).realize()
    for param_group in self.param_groups:
      for i, t in enumerate(param_group['params']):
        assert t.grad is not None
        g = t.grad.realize()
        param_group['m'][i].assign(param_group.get('b1', self.b1) * param_group['m'][i] + (1.0 - param_group.get('b1', self.b1)) * g).realize()
        param_group['v'][i].assign(param_group.get('b2', self.b2) * param_group['v'][i] + (1.0 - param_group.get('b2', self.b2)) * (g * g)).realize()
        m_hat = param_group['m'][i] / (1.0 - param_group.get('b1', self.b1)**self.t)
        v_hat = param_group['v'][i] / (1.0 - param_group.get('b2', self.b2)**self.t)
        up = (m_hat / (v_hat.sqrt() + param_group.get('eps', self.eps))) + param_group.get('wd', self.wd) * t.detach()
        if not self.adam:
          r1 = t.detach().square().sum().sqrt()
          r2 = up.square().sum().sqrt()
          r = Tensor.where(r1 > 0, Tensor.where(r2 > 0, r1 / r2, 1.0), 1.0)
        else:
          r = 1.0
        t.assign(t.detach() - param_group.get('lr', self.lr) * r * up)
      self.realize(param_group['params'] + param_group['buffers'] + [self.t] + param_group['m'] + param_group['v'])

# LAMB is essentially just the trust ratio part of LARS applied to Adam/W so if we just set the trust ratio to 1.0 its just Adam/W.
def AdamW(params: List[Union[Dict, Tensor]], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, wd=0.01): return LAMB(params, lr, b1, b2, eps, wd, adam=True)
def Adam(params: List[Union[Dict, Tensor]], lr=0.001, b1=0.9, b2=0.999, eps=1e-8): return LAMB(params, lr, b1, b2, eps, 0.0, adam=True)