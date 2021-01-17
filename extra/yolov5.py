#!/usr/bin/env python
import os
import sys
import numpy as np
from tinygrad.tensor import Tensor, GPU
from tinygrad.nn import BatchNorm2D
from extra.utils import get_parameters
GPU = os.getenv("GPU", None) is not None
QUICK = os.getenv("QUICK", None) is not None
DEBUG = os.getenv("DEBUG", None) is not None

class SqueezeExciteBlock2D:
  def __init__(self, filters):
    self.filters = filters
    self.weight1 = Tensor.uniform(self.filters, self.filters//32)
    self.bias1 = Tensor.uniform(1,self.filters//32)
    self.weight2 = Tensor.uniform(self.filters//32, self.filters)
    self.bias2 = Tensor.uniform(1, self.filters)

  def __call__(self, input):
    se = input.avg_pool2d(kernel_size=(input.shape[2], input.shape[3])) #GlobalAveragePool2D
    se = se.reshape(shape=(-1, self.filters))
    se = se.dot(self.weight1) + self.bias1
    se = se.relu()
    se = se.dot(self.weight2) + self.bias2
    se = se.sigmoid().reshape(shape=(-1,self.filters,1,1)) #for broadcasting
    se = input.mul(se)
    return se

class ConvBlock:
  def __init__(self, h, w, inp, filters=128, conv=3):
    self.h, self.w = h, w
    self.inp = inp
    #init weights
    self.cweights = [Tensor.uniform(filters, inp if i==0 else filters, conv, conv) for i in range(3)]
    self.cbiases = [Tensor.uniform(1, filters, 1, 1) for i in range(3)]
    #init layers
    self._bn = BatchNorm2D(128, training=True)
    self._seb = SqueezeExciteBlock2D(filters)

  def __call__(self, input):
    x = input.reshape(shape=(-1, self.inp, self.w, self.h))
    for cweight, cbias in zip(self.cweights, self.cbiases):
      x = x.pad2d(padding=[1,1,1,1]).conv2d(cweight).add(cbias).relu()
    x = self._bn(x)
    x = self._seb(x)
    return x

class BigConvNet:
  def __init__(self):
    self.conv = [ConvBlock(28,28,1), ConvBlock(28,28,128), ConvBlock(14,14,128)]
    self.weight1 = Tensor.uniform(128,10)
    self.weight2 = Tensor.uniform(128,10)

  def parameters(self):
    if DEBUG: #keeping this for a moment
      pars = [par for par in get_parameters(self) if par.requires_grad]
      no_pars = 0
      for par in pars:
        print(par.shape)
        no_pars += np.prod(par.shape)
      print('no of parameters', no_pars)
      return pars
    else:
      return get_parameters(self)

  def save(self, filename):
    with open(filename+'.npy', 'wb') as f:
      for par in get_parameters(self):
        #if par.requires_grad:
        np.save(f, par.cpu().data)

  def load(self, filename):
    with open(filename+'.npy', 'rb') as f:
      for par in get_parameters(self):
        #if par.requires_grad:
        try:
          par.cpu().data[:] = np.load(f)
          if GPU:
            par.gpu()
        except:
          print('Could not load parameter')

  def forward(self, x):
    x = self.conv[0](x)
    x = self.conv[1](x)
    x = x.avg_pool2d(kernel_size=(2,2))
    x = self.conv[2](x)
    x1 = x.avg_pool2d(kernel_size=(14,14)).reshape(shape=(-1,128)) #global
    x2 = x.max_pool2d(kernel_size=(14,14)).reshape(shape=(-1,128)) #global
    xo = x1.dot(self.weight1) + x2.dot(self.weight2)
    return xo.logsoftmax()

def parse_model(d, ch):


class yolo:
	def __init__(self, num_classes, model_size=(0.33, 0.5), match_thresh=4, giou_ratio=1, img_sizes=(320, 416), score_thresh=0.1, nms_thresh=0.6, detections=100):
		# Original
		anchors1 = [
      [[10, 13], [16, 30], [33, 23]],
      [[30, 61], [62, 45], [59, 119]],
      [[116, 90], [156, 198], [373, 326]]
    ]
    # [320, 416]
    anchors = [
      [[6.1, 8.1], [20.6, 12.6], [11.2, 23.7]],
      [[36.2, 26.8], [25.9, 57.2], [57.8, 47.9]],
      [[122.1, 78.3], [73.7, 143.8], [236.1, 213.1]],
    ]
		# Backbone
		self.backbone = [
			[-1, 1, Focus, [64, 3]], # 0-P1/2
			[-1, 1, Conv, [128, 3, 2]], # 1-P2/4
			[-1, 3, C3, [128]],
			[-1, 1, Conv, [256, 3, 2]], # 3-P3/8
			[-1, 9, C3, [256]],
			[-1, 1, Conv, [512, 3, 2]], # 5-P4/16
			[-1, 9, C3, [512]],
			[-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
			[-1, 1, SPP, [1024, [5, 9, 13]]],
			[-1, 3, C3, [1024, False]], # 9
		]

		# Header
		self.header = [
			[-1, 1, Conv, [512, 1, 1]],
			[-1, 1, nn.Upsample, [None, 2, 'nearest']],
			[[-1, 6], 1, Concat, [1]],  # cat backbone P4
			[-1, 3, C3, [512, False]],  # 13

			[-1, 1, Conv, [256, 1, 1]],
			[-1, 1, nn.Upsample, [None, 2, 'nearest']],
			[[-1, 4], 1, Concat, [1]],  # cat backbone P3
			[-1, 3, C3, [256, False]],  # 17 (P3/8-small)

			[-1, 1, Conv, [256, 3, 2]],
			[[-1, 14], 1, Concat, [1]],  # cat head P4
			[-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

			[-1, 1, Conv, [512, 3, 2]],
			[[-1, 10], 1, Concat, [1]],  # cat head P5
			[-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

			[[17, 20, 23], 1, Detect, [nc, anchors]
		]
	
	def parse_model():
		layers, save, c2 = [], [], None
		for i, (f, n, m, args) in enumerate()

def main():
	model_sizes = {"small": (0.33, 0.5), "medium": (0.67, 0.75), "large": (1, 1), "extreme": (1.33, 1.25)}
  num_classes = len(d_test.dataset.classes)
  model = yolo(num_classes, model_sizes[args.model_size], **args.kwargs)
  model.head.eval_with_loss = args.eval_with_loss
  
	### LOAD MODEL
  checkpoint = torch.load(args.ckpt_path, map_location=device)
  if "ema" in checkpoint:
      model.load_state_dict(checkpoint["ema"][0])
      print(checkpoint["eval_info"])
  else:
      model.load_state_dict(checkpoint)
	###
  
	model.fuse()
  print("evaluating...")
  B = time.time()
  eval_output, iter_eval = yolo.evaluate(model, d_test, device, args, evaluation=args.evaluation)
  B = time.time() - B
  print(eval_output)
  print("\ntotal time of this evaluation: {:.2f} s, speed: {:.2f} FPS".format(B, args.batch_size / iter_eval))

if __name__ == "__main__":
  main()
