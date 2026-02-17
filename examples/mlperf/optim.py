from tinygrad.tensor import Tensor
from tinygrad.nn.optim import LAMB

class GradAccClipAdamW(LAMB):
  def __init__(self, params:list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-6, weight_decay=0.0, grad_acc=1, clip_norm=1.0):
    super().__init__(params, lr, b1, b2, eps, weight_decay, adam=True, fused=True)
    self.grad_acc, self.clip_norm = grad_acc, clip_norm

  def _step(self, params:list[Tensor], grads:list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    grads[0] = grads[0] / self.grad_acc
    total_norm = grads[0].float().square().sum().sqrt()
    grads[0] = (grads[0] * (self.clip_norm / (total_norm + 1e-6)).clamp(max_=1.0)).cast(grads[0].dtype)
    return super()._step(params, grads)
