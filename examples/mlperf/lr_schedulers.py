import math
from tinygrad import dtypes, Tensor
from tinygrad.nn.optim import Optimizer

from extra.lr_scheduler import LR_Scheduler
from typing import Callable

# https://github.com/mlcommons/training/blob/e237206991d10449d9675d95606459a3cb6c21ad/image_classification/tensorflow2/lars_util.py
class PolynomialDecayWithWarmup(LR_Scheduler):
  def __init__(self, optimizer: Optimizer, initial_lr, end_lr, train_steps, warmup, power=2):
    super().__init__(optimizer)
    self.epoch_counter = self.epoch_counter.cast(dtypes.float32)
    assert train_steps > 0 and warmup > 0
    self.warmup = min(warmup, train_steps)
    self.initial_lr, self.end_lr, self.epochs, self.power = initial_lr, end_lr, train_steps, power

    # set lr for first warmup step
    self.optimizer.lr.assign(self.get_lr()).realize()

  def get_lr(self):
    # LR is 0 on the first step, matching the reference.
    warmup_lr = (self.epoch_counter * (1.0 / self.warmup)) * self.initial_lr
    x = (1 - (self.epoch_counter - self.warmup) / (self.epochs - self.warmup + 1))
    return (self.epoch_counter <= self.warmup).where(warmup_lr, (self.initial_lr - self.end_lr) * x ** self.power + self.end_lr).cast(self.optimizer.lr.dtype)

class CosineAnnealingLRWithWarmup(LR_Scheduler):
  def __init__(self, optimizer:Optimizer, base_lr, end_lr, warmup_steps:int, decay_steps:int):
    assert warmup_steps > 0 and decay_steps > 0
    super().__init__(optimizer)
    self.base_lr = base_lr
    self.end_lr = end_lr
    self.warmup_steps = warmup_steps
    self.decay_steps = decay_steps
    # set lr for first warmup step
    self.optimizer.lr.assign(self.get_lr()).realize()

  def get_lr(self):
    warmup_lr = ((self.epoch_counter+1) / self.warmup_steps) * self.base_lr
    decay_lr = self.end_lr + 0.5 * (self.base_lr-self.end_lr) * (1 + (((self.epoch_counter+1-self.warmup_steps)/self.decay_steps) * math.pi).cos())
    return (self.epoch_counter < self.warmup_steps).where(warmup_lr, decay_lr).cast(self.optimizer.lr.dtype)

try: import numpy as np
except: pass
# Reference: https://github.com/mlcommons/training/blob/64b14a9abc74e08779a175abca7d291f8c957632/stable_diffusion/ldm/lr_scheduler.py, Lines 36-97
# TODO: refactor this code for better integration with tinygrad's LR_Scheduler
# TODO: use Tensors for everything instead of numpy/python
class LambdaWarmUpCosineScheduler2:
  """
  supports repeated iterations, configurable via lists
  note: use with a base_lr of 1.0.
  """
  def __init__(self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
    assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
    self.lr_warm_up_steps = warm_up_steps
    self.f_start = f_start
    self.f_min = f_min
    self.f_max = f_max
    self.cycle_lengths = cycle_lengths
    self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
    self.last_f = 0.
    self.verbosity_interval = verbosity_interval

  def find_in_interval(self, n):
    interval = 0
    for cl in self.cum_cycles[1:]:
      if n <= cl:
        return interval
      interval += 1

  def schedule(self, n, **kwargs):
    cycle = self.find_in_interval(n)
    n = n - self.cum_cycles[cycle]
    if self.verbosity_interval > 0:
      if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                                                    f"current cycle {cycle}")
    if n < self.lr_warm_up_steps[cycle]:
      f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
      self.last_f = f
      return f
    else:
      t = (n - self.lr_warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
      t = min(t, 1.0)
      f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (1 + np.cos(t * np.pi))
      self.last_f = f
      return f

  def __call__(self, n, **kwargs):
      return self.schedule(n, **kwargs)

class LambdaLinearScheduler(LambdaWarmUpCosineScheduler2):

  def schedule(self, n, **kwargs):
    cycle = self.find_in_interval(n)
    n = n - self.cum_cycles[cycle]
    if self.verbosity_interval > 0:
      if n % self.verbosity_interval == 0:
        print(f"current step: {n}, recent lr-multiplier: {self.last_f}, current cycle {cycle}")

    if n < self.lr_warm_up_steps[cycle]:
      f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
      self.last_f = f
      return f
    else:
      f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (self.cycle_lengths[cycle] - n) / (self.cycle_lengths[cycle])
      self.last_f = f
      return f

# based on torch.optim.lr_scheduler.LambdaLR
class LambdaLR(LR_Scheduler):
  def __init__(self, optimizer:Optimizer, base_lr:float, lr_lambda:Callable):
    self.optimizer, self.base_lr, self.lr_lambda = optimizer, base_lr, lr_lambda
    self.epoch_counter = Tensor([0], requires_grad=False, device=self.optimizer.device)

  def get_lr(self):
    # LR_Scheduler.schedule_step increments self.epoch_counter by 1 before calling get_lr,
    #  but we need to calc. our first lr with self.epoch_counter=0
    lr = self.base_lr * self.lr_lambda(self.epoch_counter.item() - 1)
    return Tensor([lr])