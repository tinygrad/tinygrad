# TODO: implement BatchNorm2d and Swish
# aka batch_norm, pad, swish, dropout
# https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth
# a rough copy of
# https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py

class BatchNorm2D:
  def __init__(self, sz):
    self.weight = Tensor.zeros(sz)
    self.bias = Tensor.zeros(sz)
    # TODO: need running_mean and running_var

  def __call__(self, x):
    # this work at inference?
    return x * self.weight + self.bias

class MBConvBlock:
  def __init__(self, d0, d1, d2, d3):
    self._expand_conv = Tensor.zeros(d1, d0, 1, 1)
    self._bn0 = BatchNorm2D(d1)
    self._depthwise_conv = Tensor.zeros(d1, 1, 3, 3)
    self._bn1 = BatchNorm2D(d1)
    self._se_reduce = Tensor.zeros(d2, d1, 1, 1)
    self._se_reduce_bias = Tensor.zeros(d2)
    self._se_expand = Tensor.zeros(d1, d2, 1, 1)
    self._se_expand_bias = Tensor.zeros(d1)
    self._project_conv = Tensor.zeros(d3, d2, 1, 1)
    self._bn2 = BatchNorm2D(d3)

  def __call__(self, x):
    x = self._bn0(x.conv2d(self._expand_conv))
    x = self._bn1(x.conv2d(self._depthwise_conv))  # TODO: repeat on axis 1
    x = x.conv2d(self._se_reduce) + self._se_reduce_bias
    x = x.conv2d(self._se_expand) + self._se_expand_bias
    x = self._bn2(x.conv2d(self._project_conv))
    return x.swish()

class EfficientNet:
  def __init__(self):
    self._conv_stem = Tensor.zeros(32, 3, 3, 3)
    self._bn0 = BatchNorm2D(32)
    self._blocks = []
    # TODO: create blocks

    self._conv_head = Tensor.zeros(1280, 320, 1, 1)
    self._bn1 = BatchNorm2D(1280)
    self._fc = Tensor.zeros(1280, 1000)

  def forward(x):
    x = self._bn0(x.pad(0,1,0,1).conv2d(self._conv_stem, stride=2))
    for b in self._blocks:
      x = b(x)
    x = self._bn1(x.conv2d(self._conv_head))
    x = x.avg_pool2d() # wrong
    x = x.dropout(0.2)
    return x.dot(self_fc).swish()

