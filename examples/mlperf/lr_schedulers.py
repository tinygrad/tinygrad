from tinygrad import Tensor
from tinygrad.nn.optim import Optimizer

from extra.lr_scheduler import LR_Scheduler

class PolynomialLR(LR_Scheduler):
  def __init__(self, optimizer: Optimizer, end_lr, epochs, warmup=0, power=2):
    super().__init__(optimizer)
    warmup = min(warmup, epochs)
    self.start_lr = self.optimizer.lr.numpy().item() if isinstance(self.optimizer.lr, Tensor) else self.optimizer.lr
    self.end_lr, self.epochs, self.power, self.warmup = end_lr, epochs, power, warmup

  def get_lr(self):
    warmup_lr = ((self.epoch_counter + 1) * (1.0 / (self.warmup + 1))) * self.start_lr
    x = (1 - (self.epoch_counter - self.warmup) / (self.epochs - self.warmup))
    return (self.epoch_counter < self.warmup).where(warmup_lr, (self.start_lr - self.end_lr) * x ** self.power + self.end_lr)
