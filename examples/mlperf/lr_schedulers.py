from tinygrad import Tensor
from tinygrad.nn.optim import Optimizer

from extra.lr_scheduler import LR_Scheduler

# https://github.com/mlcommons/training/blob/e237206991d10449d9675d95606459a3cb6c21ad/image_classification/tensorflow2/lars_util.py
class PolynomialDecayWithWarmup(LR_Scheduler):
  def __init__(self, optimizer: Optimizer, train_steps, initial_learning_rate, end_learning_rate, warmup_steps, power=2):
    super().__init__(optimizer)
    self.warmup = min(warmup_steps, train_steps)
    self.epochs, self.start_lr, self.end_lr, self.power = train_steps, initial_learning_rate, end_learning_rate, power

    # set lr for first warmup step
    self.optimizer.lr.assign(lr.cast(self.optimizer.lr.dtype) if isinstance(lr := self.get_lr(), Tensor) else lr).realize()

  def get_lr(self):
    warmup_lr = (self.epoch_counter * (1.0 / self.warmup)) * self.start_lr
    x = (1 - (self.epoch_counter - self.warmup) / (self.epochs - self.warmup + 1))
    return (self.epoch_counter <= self.warmup).where(warmup_lr, (self.start_lr - self.end_lr) * x ** self.power + self.end_lr)
