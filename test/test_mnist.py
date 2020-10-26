#!/usr/bin/env python
import os
import unittest
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.utils import layer_init_uniform, fetch_mnist
import tinygrad.optim as optim
from tqdm import trange

# load the mnist dataset
X_train, Y_train, X_test, Y_test = fetch_mnist()

# create a model
class TinyBobNet:
  def __init__(self):
    self.l1 = Tensor(layer_init_uniform(784, 128))
    self.l2 = Tensor(layer_init_uniform(128, 10))

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

# create a model with a conv layer
class TinyConvNet:
  def __init__(self):
    # https://keras.io/examples/vision/mnist_convnet/
    conv = 3
    #inter_chan, out_chan = 32, 64
    inter_chan, out_chan = 8, 16   # for speed
    self.c1 = Tensor(layer_init_uniform(inter_chan,1,conv,conv))
    self.c2 = Tensor(layer_init_uniform(out_chan,inter_chan,conv,conv))
    self.l1 = Tensor(layer_init_uniform(out_chan*5*5, 10))

  def forward(self, x):
    x.data = x.data.reshape((-1, 1, 28, 28)) # hacks
    x = x.conv2d(self.c1).relu().max_pool2d()
    x = x.conv2d(self.c2).relu().max_pool2d()
    x = x.reshape(Tensor(np.array((x.shape[0], -1))))
    return x.dot(self.l1).logsoftmax()

def train(model, optim, steps, BS=128):
  losses, accuracies = [], []
  for i in (t := trange(steps, disable=os.getenv('CI') is not None)):
    samp = np.random.randint(0, X_train.shape[0], size=(BS))
    
    x = Tensor(X_train[samp].reshape((-1, 28*28)).astype(np.float32))
    Y = Y_train[samp]
    y = np.zeros((len(samp),10), np.float32)
    # correct loss for NLL, torch NLL loss returns one per row
    y[range(y.shape[0]),Y] = -10.0
    y = Tensor(y)
    
    # network
    out = model.forward(x)

    # NLL loss function
    loss = out.mul(y).mean()
    loss.backward()
    optim.step()
    
    cat = np.argmax(out.data, axis=1)
    accuracy = (cat == Y).mean()
    
    # printing
    loss = loss.data
    losses.append(loss)
    accuracies.append(accuracy)
    t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

def evaluate(model):
  def numpy_eval():
    Y_test_preds_out = model.forward(Tensor(X_test.reshape((-1, 28*28)).astype(np.float32)))
    Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
    return (Y_test == Y_test_preds).mean()

  accuracy = numpy_eval()
  print("test set accuracy is %f" % accuracy)
  assert accuracy > 0.95

class TestMNIST(unittest.TestCase):
  def test_conv(self):
    np.random.seed(1337)
    model = TinyConvNet()
    optimizer = optim.Adam([model.c1, model.c2, model.l1], lr=0.001)
    train(model, optimizer, steps=200)
    evaluate(model)
    
  def test_sgd(self):
    np.random.seed(1337)
    model = TinyBobNet()
    optimizer = optim.SGD([model.l1, model.l2], lr=0.001)
    train(model, optimizer, steps=1000)
    evaluate(model)
    
  def test_rmsprop(self):
    np.random.seed(1337)
    model = TinyBobNet()
    optimizer = optim.RMSprop([model.l1, model.l2], lr=0.0002)
    train(model, optimizer, steps=1000)
    evaluate(model)

if __name__ == '__main__':
  unittest.main()
