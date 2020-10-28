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
  def __init__(self, input_filters, expand_ratio, se_ratio, output_filters):
    oup = expand_ratio * input_filters
    if expand_ratio != 1:
      self._expand_conv = Tensor.zeros(oup, input_filters, 1, 1)
      self._bn0 = BatchNorm2D(oup)
    self._depthwise_conv = Tensor.zeros(oup, 1, 3, 3)
    self._bn1 = BatchNorm2D(oup)

    num_squeezed_channels = max(1, int(input_filters * se_ratio))
    self._se_reduce = Tensor.zeros(num_squeezed_channels, oup, 1, 1)
    self._se_reduce_bias = Tensor.zeros(num_squeezed_channels)
    self._se_expand = Tensor.zeros(oup, num_squeezed_channels, 1, 1)
    self._se_expand_bias = Tensor.zeros(oup)

    self._project_conv = Tensor.zeros(output_filters, oup, 1, 1)
    self._bn2 = BatchNorm2D(output_filters)

  def __call__(self, x):
    x = self._bn0(x.conv2d(self._expand_conv)).swish()
    x = self._bn1(x.conv2d(self._depthwise_conv)).swish()  # TODO: repeat on axis 1

    # has_se
    x_squeezed = x.avg_pool2d()
    x_squeezed = (x_squeezed.conv2d(self._se_reduce) + self._se_reduce_bias).swish()
    x_squeezed = x_squeezed.conv2d(self._se_expand) + self._se_expand_bias
    x = x * x_squeezed.sigmoid()

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
    x = x.avg_pool2d() # wrong?
    x = x.dropout(0.2)
    return x.dot(self_fc).swish()

