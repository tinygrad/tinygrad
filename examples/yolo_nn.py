from tinygrad.tensor import Tensor

# PyTorch style layers for tinygrad. These layers are here because of tinygrads
# line limit.

class MaxPool2d:
  def __init__(self, kernel_size, stride):
    if type(kernel_size) == int:
      self.kernel_size = (kernel_size, kernel_size)
    else: self.kernel_size = kernel_size
    self.stride = stride if (stride is not None) else kernel_size
  
  def __repr__(self):
    return f"MaxPool2d(kernel_size={self.kernel_size!r}, stride={self.stride!r})"
  
  def __call__(self, input):
    # TODO: Implement strided max_pool2d, and maxpool2d for 3d inputs
    return x.max_pool2d(kernel_size=self.kernel_size)


class DetectionLayer:
  def __init__(self, anchors):
    self.anchors = anchors
  
  def __call__(self, input):
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
    return input.cpu().data.repeat(self.scale_factor, axis=len(input.shape)-2).repeat(self.scale_factor, axis=len(input.shape)-1)

  def __repr__(self):
    return f"Upsample(scale_factor={self.scale_factor!r}, mode={self.mode!r})"

  def __call__(self, input):
    return Tensor(self.upsampleNearest(input))

class LeakyReLU:
  def __init__(self, neg_slope):
    self.neg_slope = neg_slope
  
  def __repr__(self):
    return f"LeakyReLU({self.neg_slope!r})"

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
    return f"Conv2d({self.in_channels!r}, {self.out_channels!r}, kernel_size={self.kernel_size!r} stride={self.stride!r}"
  
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

def strided_pool2d(x, kernel_size=(2,2), stride=2, pooling='max'):
  import numpy as np
  from numpy.lib.stride_tricks import as_strided

  output_shape = ((x.shape[2] - kernel_size[0])//stride + 1, (x.shape[3] - kernel_size[1])//stride + 1)
  output_array = np.ndarray(shape=(x.shape[0], x.shape[1], output_shape[0], output_shape[1]))

  for i in range(x.shape[1]): # iterate channels (RGB)
    input_data = x[0][i]
    output_shape = ((input_data.shape[0] - kernel_size[0])//stride + 1, (input_data.shape[1] - kernel_size[1])//stride + 1)
    strided = as_strided(input_data, shape = output_shape + kernel_size, strides = (stride * input_data.data.strides[0], stride * input_data.data.strides[1]) + input_data.data.strides)
    strided = strided.reshape(-1, *kernel_size)
    if pooling == 'max':
      output_array[0][i] = strided.max(axis=(1,2)).reshape(output_shape)
    elif pooling == 'avg':
      output_array[0][i] = strided.mean(axis=(1,2)).reshape(output_shape)
    else:
      raise Exception("strided_pool2d() only supports 'max' and 'avg' pooling options")
  return output_array
