import math

from tinygrad import Tensor, nn, dtypes
from tinygrad.helpers import prod, argfix
from tinygrad.features.multi import MultiLazyBuffer

# rejection sampling truncated randn
def rand_truncn(*shape, dtype=None, truncstds=2, **kwargs) -> Tensor:
  CNT=8
  x = Tensor.randn(*(*shape, CNT), dtype=dtype, **kwargs)
  ctr = Tensor.arange(CNT).reshape((1,) * len(x.shape[:-1]) + (CNT,)).expand(x.shape)
  take = (x.abs() <= truncstds).where(ctr, CNT).min(axis=-1, keepdim=True)  # set to 0 if no good samples
  return (ctr == take).where(x, 0).sum(axis=-1)

# https://github.com/keras-team/keras/blob/v2.15.0/keras/initializers/initializers.py#L1026-L1065
def he_normal(*shape, a: float = 0.00, **kwargs) -> Tensor:
  std = math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(argfix(*shape)[1:])) / 0.87962566103423978
  return std * rand_truncn(*shape, **kwargs)

class Conv2dHeNormal(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    self.in_channels, self.out_channels = in_channels, out_channels  # for testing
    self.weight = he_normal(out_channels, in_channels//groups, *self.kernel_size, a=0.0, dtype=dtypes.float32)
    if bias: self.bias = self.bias.cast(dtypes.float32)
  def __call__(self, x: Tensor):
    return x.conv2d(self.weight.cast(dtypes.default_float), self.bias.cast(dtypes.default_float) if self.bias is not None else None,
                    padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)

class Linear(nn.Linear):
  def __init__(self, in_features, out_features, bias=True):
    super().__init__(in_features, out_features, bias=bias)
    self.weight = Tensor.normal((out_features, in_features), mean=0.0, std=0.01, dtype=dtypes.float32)
    if bias: self.bias = Tensor.zeros(out_features, dtype=dtypes.float32)
  def __call__(self, x:Tensor):
    return x.linear(self.weight.cast(dtypes.default_float).transpose(), self.bias.cast(dtypes.default_float) if self.bias is not None else None)

class UnsyncedBatchNorm:
  def __init__(self, sz:int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1, num_devices=1):
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum
    self.num_devices = num_devices

    if affine: self.weight, self.bias = Tensor.ones(sz, dtype=dtypes.float32), Tensor.zeros(sz, dtype=dtypes.float32)
    else: self.weight, self.bias = None, None

    self.running_mean = Tensor.zeros(num_devices, sz, dtype=dtypes.float32, requires_grad=False)
    self.running_var = Tensor.ones(num_devices, sz, dtype=dtypes.float32, requires_grad=False)
    self.num_batches_tracked = Tensor.zeros(1, dtype=dtypes.int, requires_grad=False)

  def __call__(self, x:Tensor):
    if isinstance(x.lazydata, MultiLazyBuffer): assert x.lazydata.axis is None or x.lazydata.axis == 0 and len(x.lazydata.lbs) == self.num_devices

    xr = x.reshape(self.num_devices, -1, *x.shape[1:]).cast(dtypes.float32)
    batch_mean, batch_invstd = self.calc_stats(xr)
    ret = xr.batchnorm(
      self.weight.reshape(1, -1).expand((self.num_devices, -1)),
      self.bias.reshape(1, -1).expand((self.num_devices, -1)),
      batch_mean, batch_invstd, axis=(0, 2))
    return ret.reshape(x.shape).cast(x.dtype)

  def calc_stats(self, x:Tensor):
    if Tensor.training:
      # This requires two full memory accesses to x
      # https://github.com/pytorch/pytorch/blob/c618dc13d2aa23625cb0d7ada694137532a4fa33/aten/src/ATen/native/cuda/Normalization.cuh
      # There's "online" algorithms that fix this, like https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
      batch_mean = x.mean(axis=(1,3,4))
      y = (x - batch_mean.reshape(shape=[batch_mean.shape[0], 1, batch_mean.shape[1], 1, 1]))
      batch_var = (y*y).mean(axis=(1,3,4))
      batch_invstd = batch_var.add(self.eps).pow(-0.5)

      # NOTE: wow, this is done all throughout training in most PyTorch models
      if self.track_running_stats:
        self.running_mean.assign((1-self.momentum) * self.running_mean + self.momentum * batch_mean.detach().cast(self.running_mean.dtype))
        batch_var_adjust = prod(y.shape[1:]) / (prod(y.shape[1:]) - y.shape[2])
        self.running_var.assign((1-self.momentum) * self.running_var + self.momentum * batch_var_adjust * batch_var.detach().cast(self.running_var.dtype))
        self.num_batches_tracked += 1
    else:
      batch_mean = self.running_mean
      # NOTE: this can be precomputed for static inference. we expand it here so it fuses
      batch_invstd = self.running_var.reshape(self.running_var.shape[0], 1, self.running_var.shape[1], 1, 1).expand(x.shape).add(self.eps).rsqrt()
    return batch_mean, batch_invstd
