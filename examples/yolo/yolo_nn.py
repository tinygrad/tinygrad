from tinygrad.tensor import Tensor

# PyTorch style layers for tinygrad. These layers are here because of tinygrads
# line limit.

class MaxPool2d:
  def __init__(self, kernel_size, stride):
    if isinstance(kernel_size, int): self.kernel_size = (kernel_size, kernel_size)
    else: self.kernel_size = kernel_size
    self.stride = stride if (stride is not None) else kernel_size
  
  def __repr__(self):
    return f"MaxPool2d(kernel_size={self.kernel_size!r}, stride={self.stride!r})"
  
  def __call__(self, input):
    # TODO: Implement strided max_pool2d, and maxpool2d for 3d inputs
    return input.max_pool2d(kernel_size=self.kernel_size)


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

