from tinygrad.tensor import Tensor

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
      self.num_batches_tracked += 1

    if self.training:
      return self.normalize(x, batch_mean, batch_var)

    return self.normalize(x, self.running_mean, self.running_var)

  def normalize(self, x, mean, var):
    x = (x - mean.reshape(shape=[1, -1, 1, 1])) * self.weight.reshape(shape=[1, -1, 1, 1])
    return x.div(var.add(self.eps).reshape(shape=[1, -1, 1, 1])**0.5) + self.bias.reshape(shape=[1, -1, 1, 1])

