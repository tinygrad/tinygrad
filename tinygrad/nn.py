from tinygrad.tensor import Tensor

class MaxPool2d:
  def __init__(self, kernel_size, stride):
    if type(kernel_size) == int:
      self.kernel_size = (kernel_size, kernel_size)
    else: self.kernel_size = kernel_size
    self.stride = stride if (stride is not None) else kernel_size

  def __call__(self, input):
    # TODO: Implement strided max_pool2d, and maxpool2d for 3d inputs
    return x.max_pool2d(kernel_size=self.kernel_size)


class DetectionLayer:
  def __init__(self, anchors):
    self.anchors = anchors
  
  def __call__(self, input):
    # TODO: Implement detection layer
    return input

class EmptyLayer:
  def __init__(self):
    pass
  
  def __call__(self, input):
    return input

class Upsample:
  def __init__(self, scale_factor = 2, mode = "nearest"):
    self.scale_factor, self.mode = scale_factor, mode
  
  def upsampleNearest(self, input):
    # TODO: Implement actual interpolation function
    # inspired: https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/nn/functional/upsampling.h
    # Honestly, idk if this implementation is even correct, but it works for this model. Need to implement interp function
    return input.cpu().data.repeat(self.scale_factor, axis=len(input.shape)-2).repeat(self.scale_factor, axis=len(input.shape)-1)
    # return input.cpu().data.repeat(self.scale_factor, axis=1).repeat(self.scale_factor, axis=1)


  def __call__(self, input):
    return Tensor(self.upsampleNearest(input))
    #input.cpu().data = self.upsampleNearest(input)
    #return input
    # return self.upsampleNearest(input)

class LeakyReLU:
  def __init__(self, neg_slope):
    self.neg_slope = neg_slope

  def __call__(self, input):
    return input.leakyrelu(self.neg_slope)
  


class Conv2d:
  def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, groups = 1, bias = True):
    self.in_channels, self.out_channels, self.stride, self.padding, self.groups, self.bias = in_channels, out_channels, stride, padding, groups, bias # Wow this is terrible

    if type(kernel_size) == int:
      self.kernel_size = (kernel_size, kernel_size)
    else: self.kernel_size = kernel_size

    assert out_channels % groups == 0 and in_channels % groups == 0

    self.weight = Tensor.uniform(out_channels, in_channels // groups, *self.kernel_size)
    if self.bias:
      self.bias = Tensor.uniform(1, out_channels, 1, 1)
    else:
      self.bias = None
  
  def __repr__(self):
    return f"<Conv2d Layer with in_channels {self.in_channels!r}, out_channels {self.out_channels!r}, weights with shape {self.weight.shape!r}>"
  
  def __call__(self, x):
    if self.padding != 0:
      if self.bias is not None:
        x = x.pad2d(padding=[self.padding] * 4).conv2d(self.weight, stride=self.stride, groups=self.groups).add(self.bias)
      else:
        x = x.pad2d(padding=[self.padding] * 4).conv2d(self.weight, stride=self.stride, groups=self.groups)
    else:
      if self.bias is not None:
        x = x.conv2d(self.weight, stride=self.stride, groups=self.groups).add(self.bias)
      else:
        x = x.conv2d(self.weight, stride=self.stride, groups=self.groups)
    
    return x


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

