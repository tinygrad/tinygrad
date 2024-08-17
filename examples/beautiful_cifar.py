from typing import List, Callable
import functools
from tinygrad import Tensor, nn

# from https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
batchsize = 1024
bias_scaler = 64
hyp = {
  'opt': {
    'bias_lr':        1.525 * bias_scaler/512, # TODO: Is there maybe a better way to express the bias and batchnorm scaling? :'))))
    'non_bias_lr':    1.525 / 512,
    'bias_decay':     6.687e-4 * batchsize/bias_scaler,
    'non_bias_decay': 6.687e-4 * batchsize,
    'scaling_factor': 1./9,
    'percent_start': .23,
    'loss_scale_scaler': 1./32, # * Regularizer inside the loss summing (range: ~1/512 - 16+). FP8 should help with this somewhat too, whenever it comes out. :)
  },
  'net': {
    'whitening': {
      'kernel_size': 2,
      'num_examples': 50000,
    },
    'batch_norm_momentum': .4, # * Don't forget momentum is 1 - momentum here (due to a quirk in the original paper... >:( )
    'cutmix_size': 3,
    'cutmix_epochs': 6,
    'pad_amount': 2,
    'base_depth': 64 ## This should be a factor of 8 in some way to stay tensor core friendly
  },
  'misc': {
    'ema': {
      'epochs': 10, # Slight bug in that this counts only full epochs and then additionally runs the EMA for any fractional epochs at the end too
      'decay_base': .95,
      'decay_pow': 3.,
      'every_n_steps': 5,
    },
    'train_epochs': 12.1,
    'device': 'cuda',
    'data_location': 'data.pt',
  }
}

scaler = 2. ## You can play with this on your own if you want, for the first beta I wanted to keep things simple (for now) and leave it out of the hyperparams dict
depths = {
  'init':   round(scaler**-1*hyp['net']['base_depth']), # 32  w/ scaler at base value
  'block1': round(scaler** 0*hyp['net']['base_depth']), # 64  w/ scaler at base value
  'block2': round(scaler** 2*hyp['net']['base_depth']), # 256 w/ scaler at base value
  'block3': round(scaler** 3*hyp['net']['base_depth']), # 512 w/ scaler at base value
  'num_classes': 10
}
whiten_conv_depth = 3*hyp['net']['whitening']['kernel_size']**2

class ConvGroup:
  def __init__(self, channels_in, channels_out):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, bias=False),
      Tensor.max_pool2d,
      nn.BatchNorm(channels_out),
      Tensor.gelu,
      nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm(channels_out),
      Tensor.gelu]
  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)

class SpeedyConvNet:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      nn.Conv2d(3, whiten_conv_depth, kernel_size=hyp['net']['whitening']['kernel_size'], padding=0, bias=False),
      Tensor.gelu,
      ConvGroup(2*whiten_conv_depth, depths['block1']),
      ConvGroup(depths['block1'], depths['block2']),
      ConvGroup(depths['block2'], depths['block3']),
      functools.partial(Tensor.max, axis=(2,3)),
      nn.Linear(depths['block3'], depths['num_classes'], bias=False),
      lambda x: x / hyp['opt']['scaling_factor']]
  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)


if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = nn.datasets.cifar()
  cifar10_std, cifar10_mean = X_train.std_mean(axis=(0, 2, 3))
  X_train = (X_train - cifar10_mean.view(1, -1, 1, 1)) / cifar10_std.view(1, -1, 1, 1)
  X_test = (X_test - cifar10_mean.view(1, -1, 1, 1)) / cifar10_std.view(1, -1, 1, 1)
  Y_train = Y_train.one_hot(10)
  Y_test = Y_test.one_hot(10)
  print(Y_train.shape)

