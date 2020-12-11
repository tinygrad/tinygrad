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
      x = x.conv2d(cweight).add(cbias).relu()
    x = self._bn(x)
    x = self._seb(x)
    return x

class BigConvNet:
  def __init__(self):
    self.conv = [ConvBlock(28,28,1), ConvBlock(22,22,128), ConvBlock(8,8,128)]
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
            par.cuda()
        except:
          print('Could not load parameter')

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
  steps = [1, 1, 1] if QUICK else [4000, 1000, 1000]

  #testing
  lrs, steps = [1e-5, 1e-6], [100, 100]
  lmbd = 0.00025
  lossfn = lambda out,y: out.mul(y).mean() + lmbd*(model.weight1.abs() + model.weight2.abs()).sum()
  X_train, Y_train, X_test, Y_test = fetch_mnist()
  np.random.seed(1337)
  
  model = BigConvNet()
 
  if sys.argv[1] is not None:
    try:
      model.load(sys.argv[1])
      print('Loaded weights "'+sys.argv[1]+'", evaluating...')
      evaluate(model, X_test, Y_test)
    except:
      print('could not load weights "'+sys.argv[1]+'".')
 
  if GPU:
    params = get_parameters(model)
    [x.cuda_() for x in params]

  for lr, st in zip(lrs, steps):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(model, X_train, Y_train optimizer, steps=st, lossfn=lossfn, gpu=GPU)
    model.save('checkpoint')
  model.load('checkpoint')
  evaluate(model, X_test, Y_test)
