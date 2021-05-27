import math
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import BatchNorm2D
from extra.utils import fetch, fake_torch_load

USE_TORCH = False

class MBConvBlock:
  def __init__(self, kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio, has_se):
    oup = expand_ratio * input_filters
    if expand_ratio != 1:
      self._expand_conv = Tensor.uniform(oup, input_filters, 1, 1)
      self._bn0 = BatchNorm2D(oup)
    else:
      self._expand_conv = None

    self.strides = strides
    if strides == (2,2):
      self.pad = [(kernel_size-1)//2-1, (kernel_size-1)//2]*2
    else:
      self.pad = [(kernel_size-1)//2]*4

    self._depthwise_conv = Tensor.uniform(oup, 1, kernel_size, kernel_size)
    self._bn1 = BatchNorm2D(oup)

    self.has_se = has_se
    if self.has_se:
      num_squeezed_channels = max(1, int(input_filters * se_ratio))
      self._se_reduce = Tensor.uniform(num_squeezed_channels, oup, 1, 1)
      self._se_reduce_bias = Tensor.zeros(num_squeezed_channels)
      self._se_expand = Tensor.uniform(oup, num_squeezed_channels, 1, 1)
      self._se_expand_bias = Tensor.zeros(oup)

    self._project_conv = Tensor.uniform(output_filters, oup, 1, 1)
    self._bn2 = BatchNorm2D(output_filters)

  def __call__(self, inputs):
    x = inputs
    if self._expand_conv:
      x = self._bn0(x.conv2d(self._expand_conv)).swish()
    x = x.pad2d(padding=self.pad)
    x = x.conv2d(self._depthwise_conv, stride=self.strides, groups=self._depthwise_conv.shape[0])
    x = self._bn1(x).swish()

    # has_se
    if self.has_se:
      x_squeezed = x.avg_pool2d(kernel_size=x.shape[2:4])
      x_squeezed = x_squeezed.conv2d(self._se_reduce).add(self._se_reduce_bias.reshape(shape=[1, -1, 1, 1])).swish()
      x_squeezed = x_squeezed.conv2d(self._se_expand).add(self._se_expand_bias.reshape(shape=[1, -1, 1, 1]))
      x = x.mul(x_squeezed.sigmoid())

    x = self._bn2(x.conv2d(self._project_conv))
    if x.shape == inputs.shape:
      x = x.add(inputs)
    return x

class EfficientNet:
  def __init__(self, number=0, classes=1000, has_se=True):
    self.number = number
    global_params = [
      # width, depth
      (1.0, 1.0), # b0
      (1.0, 1.1), # b1
      (1.1, 1.2), # b2
      (1.2, 1.4), # b3
      (1.4, 1.8), # b4
      (1.6, 2.2), # b5
      (1.8, 2.6), # b6
      (2.0, 3.1), # b7
      (2.2, 3.6), # b8
      (4.3, 5.3), # l2
    ][number]

    def round_filters(filters):
      multiplier = global_params[0]
      divisor = 8
      filters *= multiplier
      new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
      if new_filters < 0.9 * filters: # prevent rounding by more than 10%
        new_filters += divisor
      return int(new_filters)

    def round_repeats(repeats):
      return int(math.ceil(global_params[1] * repeats))

    out_channels = round_filters(32)
    self._conv_stem = Tensor.uniform(out_channels, 3, 3, 3)
    self._bn0 = BatchNorm2D(out_channels)
    blocks_args = [
      [1, 3, (1,1), 1, 32, 16, 0.25],
      [2, 3, (2,2), 6, 16, 24, 0.25],
      [2, 5, (2,2), 6, 24, 40, 0.25],
      [3, 3, (2,2), 6, 40, 80, 0.25],
      [3, 5, (1,1), 6, 80, 112, 0.25],
      [4, 5, (2,2), 6, 112, 192, 0.25],
      [1, 3, (1,1), 6, 192, 320, 0.25],
    ]
    self._blocks = []
    # num_repeats, kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio
    for b in blocks_args:
      args = b[1:]
      args[3] = round_filters(args[3])
      args[4] = round_filters(args[4])
      for n in range(round_repeats(b[0])):
        self._blocks.append(MBConvBlock(*args, has_se=has_se))
        args[3] = args[4]
        args[1] = (1,1)

    in_channels = round_filters(320)
    out_channels = round_filters(1280)
    self._conv_head = Tensor.uniform(out_channels, in_channels, 1, 1)
    self._bn1 = BatchNorm2D(out_channels)
    self._fc = Tensor.uniform(out_channels, classes)
    self._fc_bias = Tensor.zeros(classes)

  def forward(self, x):
    x = x.pad2d(padding=(0,1,0,1))
    x = self._bn0(x.conv2d(self._conv_stem, stride=2)).swish()
    #print(x.shape, x.data[:, 0, 0, 0])
    for block in self._blocks:
      x = block(x)
    x = self._bn1(x.conv2d(self._conv_head)).swish()
    x = x.avg_pool2d(kernel_size=x.shape[2:4])
    x = x.reshape(shape=(-1, x.shape[1]))
    #x = x.dropout(0.2)
    return x.dot(self._fc).add(self._fc_bias.reshape(shape=[1,-1]))

  def load_weights_from_torch(self):
    # load b0
    # https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py#L551
    if self.number == 0:
      b0 = fetch("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth")
    elif self.number == 2:
      b0 = fetch("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth")
    elif self.number == 4:
      b0 = fetch("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth")
    elif self.number == 7:
      b0 = fetch("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth")
    else:
      raise Exception("no pretrained weights")

    if USE_TORCH:
      import io
      import torch
      b0 = torch.load(io.BytesIO(b0))
    else:
      b0 = fake_torch_load(b0)

    for k,v in b0.items():
      if '_blocks.' in k:
        k = "%s[%s].%s" % tuple(k.split(".", 2))
      mk = "self."+k
      #print(k, v.shape)
      try:
        mv = eval(mk)
      except AttributeError:
        try:
          mv = eval(mk.replace(".weight", ""))
        except AttributeError:
          mv = eval(mk.replace(".bias", "_bias"))
      vnp = v.numpy().astype(np.float32) if USE_TORCH else v.astype(np.float32)
      vnp = vnp if k != '_fc.weight' else vnp.T
      vnp = vnp if vnp.shape != () else np.array([vnp])

      if mv.shape == vnp.shape:
        mv.assign(Tensor(vnp))
      else:
        print("MISMATCH SHAPE IN %s, %r %r" % (k, mv.shape, vnp.shape))
