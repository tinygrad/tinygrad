#!/usr/bin/env python
import os
import unittest
import numpy as np
from tinygrad.tensor import Tensor
import tinygrad.optim as optim
from extra.training import train, evaluate
from extra.utils import get_parameters
from datasets import fetch_mnist

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

class TestMNIST(unittest.TestCase):

  def test_conv(self):
    np.random.seed(1337)
    model = TinyConvNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, X_train, Y_train, optimizer, steps=200)
    assert evaluate(model, X_test, Y_test) > 0.95

  def test_sgd(self):
    np.random.seed(1337)
    model = TinyBobNet()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    train(model, X_train, Y_train, optimizer, steps=1000)
    assert evaluate(model, X_test, Y_test) > 0.95

  def test_rmsprop(self):
    np.random.seed(1337)
    model = TinyBobNet()
    optimizer = optim.RMSprop(model.parameters(), lr=0.0002)
    train(model,  X_train, Y_train, optimizer, steps=1000)
    assert evaluate(model, X_test, Y_test) > 0.95

if __name__ == '__main__':
  unittest.main()
