#!/usr/bin/env python
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'test'))

import numpy as np
from tinygrad.tensor import Tensor, GPU
from tinygrad.nn import BatchNorm2D
from tinygrad.utils import get_parameters
from test_mnist import fetch_mnist, train, evaluate
import tinygrad.optim as optim
from tqdm import trange

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
    self.h = h
    self.w = w
    self.filters = filters
    self.conv = conv
    self.inp = inp
    self.c1 = Tensor.uniform(filters, inp, conv, conv)
    self.c2 = Tensor.uniform(filters, filters, conv, conv)
    self.c3 = Tensor.uniform(filters, filters,conv,conv)

    self._bn = BatchNorm2D(128)
    self._seb = SqueezeExciteBlock2D(filters)
  
  def __call__(self, input):
    x = input.reshape(shape=(-1, self.inp, self.w, self.h)) # hacks
    x = x.conv2d(self.c1).relu()
    x = x.conv2d(self.c2).relu()
    x = x.conv2d(self.c3).relu()
    x = self._bn(x)
    x = self._seb(x)
    return x

class BigConvNet:
  def __init__(self):
    self.conv = [ConvBlock(28,28,1), ConvBlock(22,22,128), ConvBlock(8,8,128)]
    self.weight1 = Tensor.uniform(128,10)
    self.weight2 = Tensor.uniform(128,10)

  def parameters(self):
    if DEBUG := True: #keeping this for a moment
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
    with open('file'+'.npy', 'wb') as f:
      for par in get_parameters(self) if par.requires_grad:
        np.save(f, par.cpu().data)

  def load(self, filename):
    with open('file'+'.npy', 'wb') as f:
      for par in get_parameters(self) if par.requires_grad:
        #todo
        #par = Tensor(np.load(f))
  
  def forward(self, x):
    x = self.conv[0](x)
    x = self.conv[1](x)
    x = x.avg_pool2d(kernel_size=(2,2))
    x = self.conv[2](x)
    x1 = x.avg_pool2d(kernel_size=(2,2)).reshape(shape=(-1,128)) #global
    x2 = x.max_pool2d(kernel_size=(2,2)).reshape(shape=(-1,128)) #global
    xo = x1.dot(self.weight1) + x2.dot(self.weight2)
    return xo.logsoftmax()

if __name__ == "__main__":
  lrs = [1e-3, 1e-4, 1e-5]
  steps = [1, 1, 1] #[4000, 1000, 1000]
  lmbd = 0.00025
  lossfn = lambda out,y: out.mul(y).mean() + lmbd*(model.weight1.abs() + model.weight2.abs()).sum()
  X_train, Y_train, X_test, Y_test = fetch_mnist()
  np.random.seed(1337)
  
  model = BigConvNet()
  for lr, st in zip(lrs, steps):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(model, optimizer, steps=st, lossfn=lossfn)
  evaluate(model)
