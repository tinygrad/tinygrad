from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.nn.optim import Optimizer
from tinygrad.helpers import FUSE_OPTIM

class GradAccClipAdamW(Optimizer):
  def __init__(self, params:list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-6, weight_decay=0.0, grad_acc=1, clip_norm=1.0, device=None, fused=FUSE_OPTIM):
    super().__init__(params, lr, device, fused)
    self.b1, self.b2, self.eps, self.wd = b1, b2, eps, weight_decay
    self.b1_t, self.b2_t = (Tensor.ones((1,), dtype=dtypes.float32, device=self.device, requires_grad=False).contiguous() for _ in [b1, b2])
    self.m = self._new_optim_param()
    self.v = self._new_optim_param()
    self.grad_acc, self.clip_norm = grad_acc, clip_norm

  def _step(self, params:list[Tensor], grads:list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    for i in range(len(grads)):
      if grads[i].device != self.m[i].device: grads[i] = grads[i].to(self.m[i].device)

    if self.fused:
      grads[0] = grads[0] / self.grad_acc
      total_norm = grads[0].float().square().sum().sqrt()
      grads[0] = (grads[0] * (self.clip_norm / (total_norm + 1e-6)).clamp(max_=1.0)).cast(grads[0].dtype)
    else:
      total_norm = Tensor.zeros((), dtype=dtypes.float32, device=self.device)
      for g in grads:
        total_norm += g.float().square().sum()
      total_norm = total_norm.sqrt()
      for i in range(len(grads)):
        grads[i] = grads[i] / self.grad_acc
        grads[i] = (grads[i] * (self.clip_norm / (total_norm + 1e-6)).clamp(max_=1.0)).cast(grads[i].dtype)

    ret = []
    self.b1_t *= self.b1
    self.b2_t *= self.b2
    for i, (t, g) in enumerate(zip(params, grads)):
      self.m[i].assign((self.b1 * self.m[i] + (1.0 - self.b1) * g).cast(self.m[i].dtype))
      self.v[i].assign((self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)).cast(self.v[i].dtype))
      m_hat = self.m[i] / (1.0 - self.b1_t)
      v_hat = self.v[i] / (1.0 - self.b2_t)
      up = m_hat / (v_hat.sqrt() + self.eps)
      ret.append((self.lr * up).cast(t.dtype))
    return ret, [self.b1_t, self.b2_t] + self.m + self.v

  def _apply_update(self, t:Tensor, up:Tensor) -> Tensor:
    up = up.shard_like(t) + self.lr.to(t.device) * self.wd * t.detach()
    return t.detach() - up.cast(t.dtype)
