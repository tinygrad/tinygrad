from tinygrad.tensor import Tensor
import numpy as np

class BatchNorm2D:
  def __init__(self, sz, eps=1e-5, track_running_stats=False, training=False, momentum=0.1):
    self.eps, self.track_running_stats, self.training, self.momentum = eps, track_running_stats, training, momentum

    self.weight, self.bias = Tensor.ones(sz), Tensor.zeros(sz)

    self.running_mean, self.running_var = Tensor.zeros(sz, requires_grad=False), Tensor.ones(sz, requires_grad=False)
    self.num_batches_tracked = Tensor.zeros(1, requires_grad=False)

  def __call__(self, x):
    if self.track_running_stats or self.training:
      batch_mean = x.mean(axis=(0,2,3))
      y = (x - batch_mean.reshape(shape=[1, -1, 1, 1]))
      batch_var = (y*y).mean(axis=(0,2,3))

    if self.track_running_stats:
      self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
      self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
      if self.num_batches_tracked is None: self.num_batches_tracked = Tensor.zeros(1, requires_grad=False)
      self.num_batches_tracked += 1

    if self.training:
      return self.normalize(x, batch_mean, batch_var)

    return self.normalize(x, self.running_mean, self.running_var)

  def normalize(self, x, mean, var):
    x = (x - mean.reshape(shape=[1, -1, 1, 1])) * self.weight.reshape(shape=[1, -1, 1, 1])
    return x.div(var.add(self.eps).reshape(shape=[1, -1, 1, 1])**0.5) + self.bias.reshape(shape=[1, -1, 1, 1])

class Linear:
  def __init__(self, in_dim, out_dim, bias=True):
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.use_bias = bias
    self.weight = Tensor.uniform(in_dim, out_dim)
    if self.use_bias:
      self.bias = Tensor.zeros(out_dim)

  def __call__(self, x):
    B, *dims, D = x.shape
    x = x.reshape(shape=(B * np.prod(dims).astype(np.int32), D))
    x = x.dot(self.weight)
    if self.use_bias:
      x = x.add(self.bias.reshape(shape=[1, -1]))
    x = x.reshape(shape=(B, *dims, -1))
    return x

class Dropout:
  def __init__(self, p=0.5):
    self.p = p

  def __call__(self, x):
    return x.dropout(p=self.p)

class Identity:
  def __call__(self, x):
    return x

class Conv2d:
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    self.out_channels = out_channels
    self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else (kernel_size[0], kernel_size[1])
    self.stride = (stride, stride) if isinstance(stride, int) else (stride[0], stride[1])
    self.padding = (padding, ) * 4 if isinstance(padding, int) else (padding[0], padding[0], padding[1], padding[1])
    self.use_bias = bias
    self.weight = Tensor.uniform(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
    if self.use_bias:
      self.bias = Tensor.uniform(out_channels)

  def __call__(self, x):
    if self.padding[0] > 0:
      x = x.pad2d(padding=self.padding)
    x = x.conv2d(self.weight, stride=self.stride)
    if self.use_bias:
      x = x.add(self.bias.reshape(shape=(1, -1, 1, 1)))
    return x

class Sequential:
  def __init__(self, *layers):
    self.layers = layers

  def __call__(self, x):
    for l in self.layers:
      x = l(x)
    return x
