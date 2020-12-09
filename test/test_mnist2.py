#!/usr/bin/env python
import os
import unittest
import numpy as np
from tinygrad.tensor import Tensor, GPU
from tinygrad.nn import BatchNorm2D
from tinygrad.utils import fetch, get_parameters
import tinygrad.optim as optim
from tqdm import trange

# mnist loader
def fetch_mnist():
  import gzip
  parse = lambda dat: np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()
  X_train = parse(fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"))[0x10:].reshape((-1, 28, 28))
  Y_train = parse(fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"))[8:]
  X_test = parse(fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"))[0x10:].reshape((-1, 28, 28))
  Y_test = parse(fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"))[8:]
  return X_train, Y_train, X_test, Y_test

# load the mnist dataset
X_train, Y_train, X_test, Y_test = fetch_mnist()

# create a model
class TinyBobNet:
  def __init__(self):
    self.l1 = Tensor.uniform(784, 128)
    self.l2 = Tensor.uniform(128, 10)

  def parameters(self):
    return get_parameters(self)

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

# create a model with a conv layer
class TinyConvNet:
  def __init__(self):
    # https://keras.io/examples/vision/mnist_convnet/
    conv = 3
    #inter_chan, out_chan = 32, 64
    inter_chan, out_chan = 8, 16   # for speed
    self.c1 = Tensor.uniform(inter_chan,1,conv,conv)
    self.c2 = Tensor.uniform(out_chan,inter_chan,conv,conv)
    self.l1 = Tensor.uniform(out_chan*5*5, 10)

  def parameters(self):
    return get_parameters(self)

  def forward(self, x):
    x = x.reshape(shape=(-1, 1, 28, 28)) # hacks
    x = x.conv2d(self.c1).relu().max_pool2d()
    x = x.conv2d(self.c2).relu().max_pool2d()
    x = x.reshape(shape=[x.shape[0], -1])
    return x.dot(self.l1).logsoftmax()
  
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
    self.conv1 = ConvBlock(28,28,1)
    self.conv2 = ConvBlock(22,22,128)
    self.conv3 = ConvBlock(8,8,128)
    self.weight1 = Tensor.uniform(128,10)
    self.weight2 = Tensor.uniform(128,10)
    self.bias = Tensor.uniform(1,10)


  def parameters(self):
    pars = [par for par in get_parameters(self) if par.requires_grad]
    no_pars = 0
    for par in pars:
      print(par.shape)
      no_pars += np.prod(par.shape)
    print('no of parameters', no_pars)
    return pars

  
  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.avg_pool2d(kernel_size=(2,2))
    x = self.conv3(x)
    x1 = x.avg_pool2d(kernel_size=(2,2)).reshape(shape=(-1,128)) #global
    x2 = x.max_pool2d(kernel_size=(2,2)).reshape(shape=(-1,128)) #global
    xo = x1.dot(self.weight1) + x2.dot(self.weight2) + self.bias
    return xo.logsoftmax()

def train(model, optim, steps, BS=128, gpu=False):
  losses, accuracies = [], []
  lmbd = Tensor([0.00025],requires_grad=False)
  for i in (t := trange(steps, disable=os.getenv('CI') is not None)):
    samp = np.random.randint(0, X_train.shape[0], size=(BS))

    x = Tensor(X_train[samp].reshape((-1, 28*28)).astype(np.float32), gpu=gpu)
    Y = Y_train[samp]
    y = np.zeros((len(samp),10), np.float32)
    # correct loss for NLL, torch NLL loss returns one per row
    y[range(y.shape[0]),Y] = -10.0
    y = Tensor(y, gpu=gpu)

    # network
    out = model.forward(x)

    # NLL loss function
    #l2 = (model.weight1.mul(model.weight1)+(model.weight2.mul(model.weight2))).sum()
    loss = out.mul(y).mean() # +l2.mul(lmbd) #+0.00025 model.weight1 model.weight2
    optim.zero_grad()
    loss.backward()
    optim.step()

    cat = np.argmax(out.cpu().data, axis=1)
    accuracy = (cat == Y).mean()

    # printing
    loss = loss.cpu().data
    losses.append(loss)
    accuracies.append(accuracy)
    t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

def evaluate(model, gpu=False, BS=128):
  def numpy_eval():
    Y_test_preds_out = np.zeros((len(Y_test),10))
    for i in trange(len(Y_test)//BS, disable=os.getenv('CI') is not None):
      Y_test_preds_out[i*BS:(i+1)*BS] = model.forward(Tensor(X_test[i*BS:(i+1)*BS].reshape((-1, 28*28)).astype(np.float32), gpu=gpu)).cpu().data
    Y_test_preds = np.argmax(Y_test_preds_out, axis=1)
    return (Y_test == Y_test_preds).mean()

  accuracy = numpy_eval()
  print("test set accuracy is %f" % accuracy)
  #assert accuracy > 0.95

if __name__ == "__main__":
  np.random.seed(1337)
  model = BigConvNet()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  train(model, optimizer, steps=4000)
  optimizer = optim.Adam(model.parameters(), lr=0.0001)
  train(model, optimizer, steps=1000)
  optimizer = optim.Adam(model.parameters(), lr=0.00001)
  train(model, optimizer, steps=1000)
  evaluate(model)
