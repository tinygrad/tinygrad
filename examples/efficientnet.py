# TODO: implement BatchNorm2d and Swish
# aka batch_norm, pad, swish, dropout
# https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth
# a rough copy of
# https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
from tinygrad.tensor import Tensor

def swish(x):
  return x.mul(x.sigmoid())

class BatchNorm2D:
  def __init__(self, sz):
    self.weight = Tensor.zeros(sz)
    self.bias = Tensor.zeros(sz)
    # TODO: need running_mean and running_var

  def __call__(self, x):
    # this work at inference?
    x = x.mul(self.weight.reshape(shape=[1, -1, 1, 1]))
    x = x.add(self.bias.reshape(shape=[1, -1, 1, 1]))
    return x

class MBConvBlock:
  def __init__(self, kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio):
    oup = expand_ratio * input_filters
    if expand_ratio != 1:
      self._expand_conv = Tensor.zeros(oup, input_filters, 1, 1)
      self._bn0 = BatchNorm2D(oup)
    else:
      self._expand_conv = None

    self.pad = (kernel_size-1)//2
    self.strides = strides

    self._depthwise_conv = Tensor.zeros(oup, 1, kernel_size, kernel_size)
    self._bn1 = BatchNorm2D(oup)

    num_squeezed_channels = max(1, int(input_filters * se_ratio))
    self._se_reduce = Tensor.zeros(num_squeezed_channels, oup, 1, 1)
    self._se_reduce_bias = Tensor.zeros(num_squeezed_channels)
    self._se_expand = Tensor.zeros(oup, num_squeezed_channels, 1, 1)
    self._se_expand_bias = Tensor.zeros(oup)

    self._project_conv = Tensor.zeros(output_filters, oup, 1, 1)
    self._bn2 = BatchNorm2D(output_filters)

  def __call__(self, x):
    if self._expand_conv:
      x = swish(self._bn0(x.conv2d(self._expand_conv)))
    x = x.pad2d(padding=(self.pad, self.pad, self.pad, self.pad))
    x = x.conv2d(self._depthwise_conv, stride=self.strides, groups=self._depthwise_conv.shape[0])
    x = swish(self._bn1(x))

    # has_se
    x_squeezed = x.avg_pool2d(kernel_size=x.shape[2:4])
    x_squeezed = swish(x_squeezed.conv2d(self._se_reduce).add(self._se_reduce_bias.reshape(shape=[1, -1, 1, 1])))
    x_squeezed = x_squeezed.conv2d(self._se_expand).add(self._se_expand_bias.reshape(shape=[1, -1, 1, 1]))
    x = x.mul(x_squeezed.sigmoid())

    x = self._bn2(x.conv2d(self._project_conv))
    return swish(x)

class EfficientNet:
  def __init__(self):
    self._conv_stem = Tensor.zeros(32, 3, 3, 3)
    self._bn0 = BatchNorm2D(32)
    blocks_args = [
      [1, 3, (1,1), 1, 32, 16, 0.25],
      [2, 3, (2,2), 6, 16, 24, 0.25],
      [2, 5, (2,2), 6, 24, 40, 0.25],
      [3, 3, (2,2), 6, 40, 80, 0.25],
      [3, 5, (1,1), 6, 80, 112, 0.25],
      [4, 5, (1,1), 6, 112, 192, 0.25],
      [1, 3, (1,1), 6, 192, 320, 0.25],
    ]
    self._blocks = []
    # num_repeats, kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio
    for b in blocks_args:
      args = b[1:]
      for n in range(b[0]):
        self._blocks.append(MBConvBlock(*args))
        args[3] = args[4]
        args[1] = (1,1)
    self._conv_head = Tensor.zeros(1280, 320, 1, 1)
    self._bn1 = BatchNorm2D(1280)
    self._fc = Tensor.zeros(1280, 1000)

  def forward(self, x):
    x = x.pad2d(padding=(0,1,0,1))
    x = self._bn0(x.conv2d(self._conv_stem, stride=2))
    for b in self._blocks:
      x = b(x)
    x = self._bn1(x.conv2d(self._conv_head))
    x = x.avg_pool2d(kernel_size=x.shape[2:4]).reshape(shape=(-1, 1280))
    #x = x.dropout(0.2)
    return swish(x.dot(self._fc))

if __name__ == "__main__":
  model = EfficientNet()
  out = model.forward(Tensor.zeros(1, 3, 224, 224))
  print(out)

