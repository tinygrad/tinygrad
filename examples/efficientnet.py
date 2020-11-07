# load weights from
# https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth
# a rough copy of
# https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
import os
GPU = os.getenv("GPU", None) is not None
import sys
import io
import time
import numpy as np
np.set_printoptions(suppress=True)

from tinygrad.tensor import Tensor
from tinygrad.utils import fetch

# BatchNorm2D and swish
from tinygrad.nn import *

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

  def __call__(self, inputs):
    x = inputs
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
    if x.shape == inputs.shape:
      x = x.add(inputs)
    return x

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
      [4, 5, (2,2), 6, 112, 192, 0.25],
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
    self._fc_bias = Tensor.zeros(1000)

  def forward(self, x):
    x = x.pad2d(padding=(0,1,0,1))
    x = swish(self._bn0(x.conv2d(self._conv_stem, stride=2)))
    for block in self._blocks:
      #print(x.shape)
      x = block(x)
    x = swish(self._bn1(x.conv2d(self._conv_head)))
    x = x.avg_pool2d(kernel_size=x.shape[2:4])
    x = x.reshape(shape=(-1, 1280))
    #x = x.dropout(0.2)
    return x.dot(self._fc).add(self._fc_bias.reshape(shape=[1,-1]))

  def load_weights_from_torch(self):
    # load b0
    import torch
    b0 = fetch("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth")
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
  model = EfficientNet()
  model.load_weights_from_torch()

  # category labels
  import ast
  lbls = fetch("https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt")
  lbls = ast.literal_eval(lbls.decode('utf-8'))

  # load image and preprocess
  from PIL import Image
  url = sys.argv[1]
  if url == 'webcam':
    import pygame
    pygame.init()
    SCALE = 3
    screen = pygame.display.set_mode((224*SCALE, 224*SCALE))
    pygame.display.set_caption("capture")

    import cv2
    cap = cv2.VideoCapture(0)
    while 1:
      ret, frame = cap.read()
      frame = Image.fromarray(frame[:, :, [2,1,0]])
      out, retimg = infer(model, frame)
      simg = cv2.resize(retimg, (224*SCALE, 224*SCALE))
      pygame.surfarray.blit_array(screen, np.array(simg).swapaxes(0,1))
      pygame.display.update()
      for e in pygame.event.get():
        pass
      print(np.argmax(out.data), np.max(out.data), lbls[np.argmax(out.data)])
  else:
    img = Image.open(io.BytesIO(fetch(url)))
    st = time.time()
    out, _ = infer(model, img)
    print(np.argmax(out.data), np.max(out.data), lbls[np.argmax(out.data)])
    print("did inference in %.2f s" % (time.time()-st))
  #print("NOT", np.argmin(out.data), np.min(out.data), lbls[np.argmin(out.data)])

