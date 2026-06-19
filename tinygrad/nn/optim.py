# sorted in order of increasing complexity
import itertools, math
from tinygrad.helpers import dedup, flatten, getenv, unwrap, FUSE_OPTIM
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes, least_upper_dtype, to_dtype
from tinygrad.uop.ops import sint

class Optimizer:
  """
  Base class for all optimizers.
  """
  def __init__(self, params: list[Tensor], lr: float, device=None, fused=FUSE_OPTIM):
    if lr < 0: raise ValueError(f"Invalid learning rate: {lr}")
    self.params: list[Tensor] = dedup([x for x in params if x.is_param])
    assert len(self.params) != 0, "optimizer must have at least one param"
    self.buffers: list[Tensor] = dedup([x for x in params if not x.is_param])   # buffers are still realized
    self.device = device or self.params[0].device
    self.param_dtype = to_dtype(getenv("OPTIM_DTYPE", "float32"))
    self.fused = fused
    # store lr in at least float32 precision
    self.lr = Tensor(lr if getenv("CONST_LR") else [lr], device=self.device,
                     dtype=least_upper_dtype(dtypes.default_float, dtypes.float32))
    if self.fused: self.pos_params = list(itertools.accumulate(self.params, lambda x,y: x+y.numel(), initial=0))

  def _new_optim_param(self) -> list[Tensor]:
    if self.fused: return [Tensor.zeros(self.pos_params[-1], dtype=self.param_dtype, device=self.device)]
    if isinstance(self.device, tuple): return [Tensor.zeros_like(t, dtype=self.param_dtype) for t in self.params]
    else: return [Tensor.zeros(t.shape, dtype=self.param_dtype, device=self.device) for t in self.params]

  def zero_grad(self):
    """
    Zeroes the gradients of all the parameters.
    """
    for param in self.params: param.grad = None

  def step(self):
    """
    Performs a single optimization step.
    """
    Tensor.realize(*self.schedule_step())

  def schedule_step(self) -> list[Tensor]:
    """
    Returns the tensors that need to be realized to perform a single optimization step.
    """
    if not Tensor.training: raise RuntimeError(
            f"""Tensor.training={Tensor.training}, Tensor.training must be enabled to use the optimizer.
                - help: Consider setting Tensor.training=True before calling Optimizer.step().""")
    if self.fused:
      # optimizer fusion just concatenates all the buffers, runs the _step, then splits them back up
      # NOTE: contiguous is for speed
      out, extra = self._step([Tensor.cat(*[t.flatten() for t in self.params], dim=0)],
                              [Tensor.cat(*[unwrap(t.grad).contiguous().flatten() for t in self.params], dim=0)])
      updates = [out[0][self.pos_params[i]:self.pos_params[i+1]].reshape(tt.shape) for i, tt in enumerate(self.params)]
    else:
      updates, extra = self._step(self.params, [unwrap(t.grad) for t in self.params])
    for i, tt in enumerate(self.params): tt.assign(self._apply_update(tt, updates[i]))
    return extra+self.params+self.buffers

  def _step(self, params:list[Tensor], grads:list[Tensor]) -> tuple[list[Tensor], list[Tensor]]: raise NotImplementedError
  def _apply_update(self, t:Tensor, up:Tensor) -> Tensor: return t.detach() - up.to(t.device)

class OptimizerGroup(Optimizer):
  """
  Combines multiple optimizers into one.
  """
  def __init__(self, *optimizers: Optimizer): # pylint: disable=super-init-not-called
    self.optimizers = optimizers
    self.params, self.buffers = flatten([o.params for o in self.optimizers]), flatten([o.buffers for o in self.optimizers])
  def __getitem__(self, i): return self.optimizers[i]
  def zero_grad(self): [o.zero_grad() for o in self.optimizers]
  def schedule_step(self) -> list[Tensor]: return [x for o in self.optimizers for x in o.schedule_step()]

# LARS is essentially just trust ratio to SGD so if we just set the trust coeff 0.0 it's just standard SGD.
def SGD(params: list[Tensor], lr=0.001, momentum=0.0, weight_decay=0.0, nesterov=False, classic=False, device=None, fused=FUSE_OPTIM):
  """
  Stochastic Gradient Descent (SGD) optimizer with optional momentum and weight decay.

  `classic` is a boolean flag that determines whether to use the popular momentum update rule or the classic momentum update rule.
  """
  return LARS(params, lr, momentum, weight_decay, 0, None, nesterov, classic=classic, pre_wd=True, tcoef=0.0, device=device, fused=fused)

# Muon applies the newton schulz algorithm on gradient. also can include momentum, nesterov, and weight decay
def Muon(params: list[Tensor], lr=0.001, momentum=0.95, weight_decay=0.1, ns_steps=5, ns_coefficients=(3.4445, -4.775, 2.0315),
         nesterov=True, device=None, fused=FUSE_OPTIM):
  """
  SGD with newton-schulz iteration and post momentum weight decay.

  - Described: https://kellerjordan.github.io/posts/muon/
  - Paper: https://arxiv.org/pdf/2502.16982
  """
  assert not fused, "FUSE_OPTIM not allowed for Muon optimizer"
  return LARS(params, lr, momentum, weight_decay, ns_steps, ns_coefficients, nesterov,
              classic=False, pre_wd=False, tcoef=0.0, device=device, fused=fused)

class LARS(Optimizer):
  """
  Layer-wise Adaptive Rate Scaling (LARS) optimizer with optional momentum and weight decay.

  - Paper: https://arxiv.org/abs/1708.03888v3
  """
  def __init__(self, params:list[Tensor], lr=0.001, momentum=0.9, weight_decay=1e-4, ns_steps=0, ns_coefficients=None,
               nesterov=False, classic=True, pre_wd=True, tcoef=0.001, device=None, fused=FUSE_OPTIM):
    if momentum < 0: raise ValueError(f"Invalid momentum value: {momentum}")
    super().__init__(params, lr, device, fused)
    self.momentum, self.wd, self.ns_steps, self.ns_coefficients  = momentum, weight_decay, ns_steps, ns_coefficients
    self.nesterov, self.classic, self.pre_wd, self.tcoef = nesterov, classic, pre_wd, tcoef
    self.b = self._new_optim_param() if self.momentum else []

  def _step(self, params:list[Tensor], grads:list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    ret = []
    for i, (t, g) in enumerate(zip(params, grads)):
      if self.tcoef != 0:
        r1 = t.detach().square().sum().sqrt()
        r2 = g.square().sum().sqrt()
        r:Tensor|float = (r1 > 0).where((r2 > 0).where(self.tcoef * r1 / (r2 + self.wd * r1), 1.0), 1.0)
      else: r = 1.0
      if self.pre_wd and self.wd > 0: g = g + self.wd * t.detach()
      # classic momentum does post learning rate update
      if self.classic: g = g * r * self.lr
      if self.momentum:
        self.b[i].assign(self.momentum * self.b[i] + g)  # NOTE: self.b[i] is zero on the first run, no if required
        g = (g + self.momentum * self.b[i]) if self.nesterov else self.b[i]
      if self.ns_coefficients: g = g.reshape(g.shape[0], -1).newton_schulz(self.ns_steps, self.ns_coefficients).reshape(g.shape)
      # muon does post momentum weight decay
      if not self.pre_wd and self.wd > 0: t = t.detach() * (1.0 - self.wd * self.lr)
      # popular momentum does pre learning rate update
      if not self.classic: g = g * r * self.lr
      ret.append(g.cast(t.dtype))
    return ret, self.b

# LAMB is essentially just the trust ratio part of LARS applied to Adam/W so if we just set the trust ratio to 1.0 it's just Adam/W.
def AdamW(params: list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01, device=None, fused=FUSE_OPTIM):
  """
  AdamW optimizer with optional weight decay.

  - Paper: https://arxiv.org/abs/1711.05101v3
  """
  return LAMB(params, lr, b1, b2, eps, weight_decay, adam=True, device=device, fused=fused)
def Adam(params: list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, device=None, fused=FUSE_OPTIM):
  """
  Adam optimizer.

  - Paper: https://arxiv.org/abs/1412.6980
  """
  return LAMB(params, lr, b1, b2, eps, 0.0, adam=True, device=device, fused=fused)

class LAMB(Optimizer):
  """
  LAMB optimizer with optional weight decay.

  - Paper: https://arxiv.org/abs/1904.00962
  """
  def __init__(self, params: list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-6, weight_decay=0.0, adam=False, device=None, fused=FUSE_OPTIM):
    if weight_decay < 0: raise ValueError(f"Invalid weight_decay value: {weight_decay}")
    super().__init__(params, lr, device, fused)
    self.b1, self.b2, self.eps, self.wd, self.adam = b1, b2, eps, weight_decay, adam
    self.b1_t, self.b2_t = (Tensor.ones((1,), dtype=dtypes.float32, device=self.device).is_param_(False) for _ in [b1, b2])
    self.m = self._new_optim_param()
    self.v = self._new_optim_param()

  def _step(self, params:list[Tensor], grads:list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    ret = []
    self.b1_t *= self.b1
    self.b2_t *= self.b2
    for i, (t, g) in enumerate(zip(params, grads)):
      if g.device != self.m[i].device: g = g.contiguous().to(self.m[i].device)
      self.m[i].assign((self.b1 * self.m[i] + (1.0 - self.b1) * g).cast(self.m[i].dtype))
      self.v[i].assign((self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)).cast(self.v[i].dtype))
      m_hat = self.m[i] / (1.0 - self.b1_t)
      v_hat = self.v[i] / (1.0 - self.b2_t)
      up = (m_hat / (v_hat.sqrt() + self.eps)).shard_like(t) + self.wd * t.detach()
      if not self.adam:
        r1 = t.detach().square().sum().sqrt()
        r2 = up.square().sum().sqrt()
        r: Tensor|float = Tensor.where(r1 > 0, Tensor.where(r2 > 0, r1 / r2, 1.0), 1.0)
      else:
        r = 1.0
      ret.append((self.lr * r * up).cast(t.dtype))
    return ret, [self.b1_t, self.b2_t] + self.m + self.v

class APOLLO(Optimizer):
  """
  APOLLO optimizer with low-rank gradient scaling.

  APOLLO is applied to 2D parameter tensors. Other tensors use the same AdamW update without projection.

  - Paper: https://arxiv.org/abs/2412.05270
  """
  def __init__(self, params:list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-6, weight_decay=0.0, rank=256,
               proj="random", scale_type="channel", scale=1.0, update_proj_gap=200, proj_type="std", scale_front=False,
               disable_nl=False, correct_bias=True, device=None, fused=FUSE_OPTIM):
    assert not fused, "FUSE_OPTIM not allowed for APOLLO optimizer"
    if not 0 <= b1 < 1: raise ValueError(f"Invalid b1 value: {b1}")
    if not 0 <= b2 < 1: raise ValueError(f"Invalid b2 value: {b2}")
    if eps < 0: raise ValueError(f"Invalid eps value: {eps}")
    if weight_decay < 0: raise ValueError(f"Invalid weight_decay value: {weight_decay}")
    if rank < 1: raise ValueError(f"Invalid rank value: {rank}")
    if scale < 0: raise ValueError(f"Invalid scale value: {scale}")
    if update_proj_gap < 1: raise ValueError(f"Invalid update_proj_gap value: {update_proj_gap}")
    if proj != "random": raise NotImplementedError("tinygrad APOLLO supports proj='random'")
    if scale_type not in {"channel", "tensor"}: raise ValueError(f"Invalid scale_type value: {scale_type}")
    if proj_type not in {"std", "reverse_std", "left", "right"}: raise ValueError(f"Invalid proj_type value: {proj_type}")
    super().__init__(params, lr, device, fused)
    if isinstance(self.device, tuple): raise AssertionError("APOLLO optimizer state must be on a single device")
    self.b1, self.b2, self.eps, self.wd, self.rank = b1, b2, eps, weight_decay, rank
    self.scale_type, self.scale, self.update_proj_gap = scale_type, scale, update_proj_gap
    self.proj_type, self.scale_front, self.disable_nl, self.correct_bias = proj_type, scale_front, disable_nl, correct_bias
    self.apollo = [t.ndim == 2 for t in self.params]
    self.step_t = Tensor.zeros((1,), dtype=dtypes.int32, device=self.device).is_param_(False)
    self.b1_t, self.b2_t = (Tensor.ones((1,), dtype=dtypes.float32, device=self.device).is_param_(False) for _ in [b1, b2])
    self.m = [Tensor.zeros(self._state_shape(t), dtype=self.param_dtype, device=self.device) for t in self.params]
    self.v = [Tensor.zeros(self._state_shape(t), dtype=self.param_dtype, device=self.device) for t in self.params]
    self.projs = [Tensor.zeros(self._proj_shape(t.shape), dtype=self.param_dtype, device=self.device) if a else
                  Tensor.zeros((0,), dtype=self.param_dtype, device=self.device) for t,a in zip(self.params, self.apollo)]
    self.scaled_grad = [Tensor.zeros((1,), dtype=self.param_dtype, device=self.device) for _ in self.params]

  def _right(self, shape:tuple[sint, ...]) -> bool:
    if self.proj_type == "right": return True
    if self.proj_type == "left": return False
    return (shape[0] >= shape[1]) == (self.proj_type == "std")

  def _rank(self, shape:tuple[sint, ...]) -> int: return min(self.rank, shape[0], shape[1])
  def _low_shape(self, shape:tuple[sint, ...]) -> tuple[sint, sint]:
    return (shape[0], self._rank(shape)) if self._right(shape) else (self._rank(shape), shape[1])
  def _proj_shape(self, shape:tuple[sint, ...]) -> tuple[sint, sint]:
    return (self._rank(shape), shape[1]) if self._right(shape) else (shape[0], self._rank(shape))
  def _state_shape(self, t:Tensor) -> tuple[sint, ...]: return self._low_shape(t.shape) if t.ndim == 2 else t.shape

  def _new_proj(self, g:Tensor, right:bool) -> Tensor:
    rank = self._rank(g.shape)
    return Tensor.randn(*((rank, g.shape[1]) if right else (g.shape[0], rank)), dtype=self.param_dtype, device=self.device) / math.sqrt(rank)

  def _project(self, g:Tensor, i:int, refresh:Tensor) -> tuple[Tensor, bool]:
    right = self._right(g.shape)
    proj = self.projs[i].assign(refresh.where(self._new_proj(g, right), self.projs[i]))
    return (g @ proj.transpose() if right else proj.transpose() @ g), right

  def _adam_update(self, g:Tensor, i:int) -> Tensor:
    self.m[i].assign((self.b1 * self.m[i] + (1.0 - self.b1) * g).cast(self.m[i].dtype))
    self.v[i].assign((self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)).cast(self.v[i].dtype))
    return self.m[i] / (self.v[i].sqrt() + self.eps)

  def _step(self, params:list[Tensor], grads:list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    ret = []
    self.step_t.assign(self.step_t + 1)
    first, refresh = self.step_t.eq(1), ((self.step_t - 1) % self.update_proj_gap).eq(0)
    self.b1_t *= self.b1
    self.b2_t *= self.b2
    step_size = self.lr * ((1.0 - self.b2_t).sqrt() / (1.0 - self.b1_t) if self.correct_bias else 1.0)
    for i, (t, g) in enumerate(zip(params, grads)):
      if g.device != self.m[i].device: g = g.contiguous().to(self.m[i].device)
      g = g.cast(self.param_dtype)
      if self.apollo[i]:
        low, right = self._project(g, i, refresh)
        norm_grad = self._adam_update(low, i)
        if self.scale_type == "channel":
          axis = 1 if right else 0
          factor = norm_grad.square().sum(axis).sqrt() / (low.square().sum(axis).sqrt() + 1e-8)
          if axis == 1: factor = factor.unsqueeze(1)
        else:
          factor = norm_grad.square().sum().sqrt() / (low.square().sum().sqrt() + 1e-8)
        norm_grad = g * factor
        if self.scale_front: norm_grad = norm_grad * math.sqrt(self.scale)
        if not self.disable_nl:
          norm = norm_grad.square().sum().sqrt().reshape(1)
          limiter = (norm / (self.scaled_grad[i] + 1e-8)).maximum(1.01) / 1.01
          norm_grad, norm = first.where(norm_grad, norm_grad / limiter), first.where(norm, norm / limiter)
          self.scaled_grad[i].assign(norm.cast(self.scaled_grad[i].dtype))
        if not self.scale_front: norm_grad = norm_grad * math.sqrt(self.scale)
      else:
        norm_grad = self._adam_update(g, i)
      up = step_size * norm_grad
      if self.wd > 0: up = up + self.lr * self.wd * (t.detach().to(up.device) - up)
      ret.append(up.cast(t.dtype))
    return ret, [self.step_t, self.b1_t, self.b2_t] + self.m + self.v + self.projs + self.scaled_grad
