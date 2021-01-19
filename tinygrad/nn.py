from tinygrad.tensor import Tensor

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
    # TODO: Implement bilinear upsampling with pure numpy
    return input.cpu().data.repeat(self.scale_factor, axis=1)
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
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0, groups = 1, bias = False):
    self.in_channels, self.out_channels, self.stride, self.padding, self.groups, self.bias = in_channels, out_channels, stride, padding, groups, bias # Wow this is terrible

    if type(kernel_size) == int:
      self.kernel_size = (kernel_size, kernel_size)
    else: self.kernel_size = kernel_size

    assert out_channels % groups == 0 and in_channels % groups == 0

    self.weights = Tensor.uniform(out_channels, in_channels // groups, *self.kernel_size)
    if self.bias:
      self.biases = Tensor.uniform(1, out_channels, 1, 1)
    else:
      self.biases = None
  
  def __repr__(self):
    return f"<Conv2d Layer with in_channels {self.in_channels!r}, out_channels {self.out_channels!r}, weights with shape {self.weights.shape!r}>"

  
  def __call__(self, x):
    """
    x = input.reshape(shape=(-1, self.inp, self.w, self.h))
    for cweight, cbias in zip(self.cweights, self.cbiases):
      x = x.pad2d(padding=[1,1,1,1]).conv2d(cweight).add(cbias).relu()
    x = self._bn(x)
    x = self._seb(x)
    """
    # Figure out this line... wtf
    # x = x.reshape(shape=(-1, self.in_channels, self.out_channels))
    #x = input
    """
    for weights, biases in zip(self.weights, self.biases):
      if padding != 0:
        # TODO: Check padding line
        # Mauybe should be padding=[self.padding, self.padding * 2]
        x = x.pad2d(self.padding).conv2d(weights, self.stride, self.groups).add(biases)
      else:
        x = x.conv2d(weights, self.stride, self.groups).add(biases)
    """
    if self.padding != 0:
      # TODO: Check padding line
      x = x.pad2d(padding=[self.padding] * 4).conv2d(self.weights, stride=self.stride, groups=self.groups).add(self.biases)
    else:
      x = x.conv2d(self.weights, stride=self.stride, groups=self.groups).add(self.biases)
    
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

