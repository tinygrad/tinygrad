from typing import List, Set

from tinygrad import Tensor
from tinygrad.nn.optim import Optimizer

# https://github.com/mlcommons/training/blob/master/image_classification/tensorflow2/lars_optimizer.py
class LARS(Optimizer):
  def __init__(self, params: List[Tensor], lr, momentum=0.9, weight_decay=1e-4, eta=0.001, eps=0.0, skip_list=None, nesterov=False):
    super().__init__(params, lr)
    assert momentum >= 0.0 and weight_decay >= 0.0
    self.momentum, self.weight_decay, self.eta, self.eps, self.nesterov = momentum, weight_decay, eta, eps, nesterov
    self.b = [Tensor.zeros(*t.shape, device=t.device, requires_grad=False) for t in self.params]
    self.skip_list = set(skip_list or [])

  def step(self):
    for i, t in enumerate(self.params):
      assert t.grad is not None
      g = t.grad.contiguous()
      w = t.detach()

      if t not in self.skip_list:
        g_norm = (g * g).sum().sqrt()
        w_norm = (w * w).sum().sqrt()
        trust_ratio = ((w_norm > 0) * (g_norm > 0)).where(
            self.eta * w_norm / (g_norm + self.weight_decay * w_norm + self.eps),
          1.0)

        scaled_lr = self.lr * trust_ratio
        g = g + self.weight_decay * t.detach()
      else:
        scaled_lr = self.lr

      g = g * scaled_lr
      if self.momentum:
        self.b[i].assign(self.momentum * self.b[i] + g)
        g = (g + self.momentum * self.b[i]) if self.nesterov else self.b[i]
      t.assign(t.detach() - g)
    self.realize(self.b)
