# load weights from
# https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth
# a rough copy of
# https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
import os
GPU = os.getenv("GPU", None) is not None
import sys
import math
import io
import time
import numpy as np
np.set_printoptions(suppress=True)

from tinygrad.tensor import Tensor
from tinygrad.utils import fetch
from tinygrad.nn import BatchNorm2D

class MBConvBlock:
  def __init__(self, kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio):
    oup = expand_ratio * input_filters
    if expand_ratio != 1:
      self._expand_conv = Tensor.zeros(oup, input_filters, 1, 1)
      self._bn0 = BatchNorm2D(oup)
    else:
      self._expand_conv = None

    self.strides = strides
    if strides == (2,2):
      self.pad = [(kernel_size-1)//2-1, (kernel_size-1)//2]*2
    else:
      self.pad = [(kernel_size-1)//2]*4

    self._depthwise_conv = Tensor.zeros(oup, 1, kernel_size, kernel_size)
    self._bn1 = BatchNorm2D(oup)

    num_squeezed_channels = max(1, int(input_filters * se_ratio))
    self._se_reduce = Tensor.zeros(num_squeezed_channels, oup, 1, 1)
    self._se_reduce_bias = Tensor.zeros(num_squeezed_channels)
    self._se_expand = Tensor.zeros(oup, num_squeezed_channels, 1, 1)
    self._se_expand_bias = Tensor.zeros(oup)

    self._project_conv = Tensor.zeros(output_filters, oup, 1, 1)
    self._bn2 = BatchNorm2D(output_filters)

  def __call__(self, inputs):
    x = inputs
    if self._expand_conv:
      x = self._bn0(x.conv2d(self._expand_conv)).swish()
    x = x.pad2d(padding=self.pad)
    x = x.conv2d(self._depthwise_conv, stride=self.strides, groups=self._depthwise_conv.shape[0])
    x = self._bn1(x).swish()

    # has_se
    x_squeezed = x.avg_pool2d(kernel_size=x.shape[2:4])
    x_squeezed = x_squeezed.conv2d(self._se_reduce).add(self._se_reduce_bias.reshape(shape=[1, -1, 1, 1])).swish()
    x_squeezed = x_squeezed.conv2d(self._se_expand).add(self._se_expand_bias.reshape(shape=[1, -1, 1, 1]))
    x = x.mul(x_squeezed.sigmoid())

    x = self._bn2(x.conv2d(self._project_conv))
    if x.shape == inputs.shape:
      x = x.add(inputs)
    return x

class EfficientNet:
  def __init__(self, number=0):
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
    self._conv_stem = Tensor.zeros(out_channels, 3, 3, 3)
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
        self._blocks.append(MBConvBlock(*args))
        args[3] = args[4]
        args[1] = (1,1)

    in_channels = round_filters(320)
    out_channels = round_filters(1280)
    self._conv_head = Tensor.zeros(out_channels, in_channels, 1, 1)
    self._bn1 = BatchNorm2D(out_channels)
    self._fc = Tensor.zeros(out_channels, 1000)
    self._fc_bias = Tensor.zeros(1000)

  def forward(self, x):
    x = x.pad2d(padding=(0,1,0,1))
    x = self._bn0(x.conv2d(self._conv_stem, stride=2)).swish()
    for block in self._blocks:
      #print(x.shape)
      x = block(x)
    x = self._bn1(x.conv2d(self._conv_head)).swish()
    x = x.avg_pool2d(kernel_size=x.shape[2:4])
    x = x.reshape(shape=(-1, x.shape[1]))
    #x = x.dropout(0.2)
    return x.dot(self._fc).add(self._fc_bias.reshape(shape=[1,-1]))

  def load_weights_from_torch(self):
    # load b0
    import torch
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
    b0 = torch.load(io.BytesIO(b0))

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
      vnp = v.numpy().astype(np.float32)
      mv.data[:] = vnp if k != '_fc.weight' else vnp.T
      if GPU:
        mv.cuda_()

def infer(model, img):
  # preprocess image
  aspect_ratio = img.size[0] / img.size[1]
  img = img.resize((int(224*max(aspect_ratio,1.0)), int(224*max(1.0/aspect_ratio,1.0))))

  img = np.array(img)
  y0,x0=(np.asarray(img.shape)[:2]-224)//2
  retimg = img = img[y0:y0+224, x0:x0+224]

  # if you want to look at the image
  """
  import matplotlib.pyplot as plt
  plt.imshow(img)
  plt.show()
  """

  # low level preprocess
  img = np.moveaxis(img, [2,0,1], [0,1,2])
  img = img.astype(np.float32).reshape(1,3,224,224)
  img /= 255.0
  img -= np.array([0.485, 0.456, 0.406]).reshape((1,-1,1,1))
  img /= np.array([0.229, 0.224, 0.225]).reshape((1,-1,1,1))

  # run the net
  if GPU:
    out = model.forward(Tensor(img).cuda()).cpu()
  else:
    out = model.forward(Tensor(img))

  # if you want to look at the outputs
  """
  import matplotlib.pyplot as plt
  plt.plot(out.data[0])
  plt.show()
  """
  return out, retimg

if __name__ == "__main__":
  # instantiate my net
  model = EfficientNet(int(os.getenv("NUM", "0")))
  model.load_weights_from_torch()

  # category labels
  import ast
  lbls = fetch("https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt")
  lbls = ast.literal_eval(lbls.decode('utf-8'))

  # load image and preprocess
  from PIL import Image
  url = sys.argv[1]
  if url == 'webcam':
    import cv2
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while 1:
      _ = cap.grab() # discard one frame to circumvent capture buffering
      ret, frame = cap.read()
      img = Image.fromarray(frame[:, :, [2,1,0]])
      out, retimg = infer(model, img)
      print(np.argmax(out.data), np.max(out.data), lbls[np.argmax(out.data)])
      SCALE = 3
      simg = cv2.resize(retimg, (224*SCALE, 224*SCALE))
      retimg = cv2.cvtColor(simg, cv2.COLOR_RGB2BGR)
      cv2.imshow('capture', retimg)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
    cv2.destroyAllWindows()
  else:
    img = Image.open(io.BytesIO(fetch(url)))
    st = time.time()
    out, _ = infer(model, img)
    print(np.argmax(out.data), np.max(out.data), lbls[np.argmax(out.data)])
    print("did inference in %.2f s" % (time.time()-st))
  #print("NOT", np.argmin(out.data), np.min(out.data), lbls[np.argmin(out.data)])

