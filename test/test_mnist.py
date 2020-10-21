#!/usr/bin/env python
import os
import unittest
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.utils import layer_init_uniform, fetch_mnist
import tinygrad.optim as tinygrad_optim
from tqdm import trange
np.random.seed(1337)

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
# perfect if you like slow speeds and very little accuracy gains
class TinyConvNet:
  def __init__(self):
    self.chans = 4
    self.c1 = Tensor(layer_init_uniform(self.chans,1,3,3))
    self.l1 = Tensor(layer_init_uniform(26*26*self.chans, 128))
    self.l2 = Tensor(layer_init_uniform(128, 10))

  def forward(self, x):
    x.data = x.data.reshape((-1, 1, 28, 28)) # hacks
    x = x.conv2d(self.c1).reshape(Tensor(np.array((-1, 26*26*self.chans)))).relu()
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()


class TestMNIST(unittest.TestCase):

  def test_mnist(self):
    if os.getenv("CONV") == "1":
      model = TinyConvNet()
      optim = tinygrad_optim.Adam([model.c1, model.l1, model.l2], lr=0.001)
      steps = 400
    else:
      model = TinyBobNet()
      optim = tinygrad_optim.SGD([model.l1, model.l2], lr=0.001)
      steps = 1000

    BS = 128
    losses, accuracies = [], []
    for i in (t := trange(steps)):
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

    # evaluate
    def numpy_eval():
      Y_test_preds_out = model.forward(Tensor(X_test.reshape((-1, 28*28)).astype(np.float32)))
      Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
      return (Y_test == Y_test_preds).mean()

    accuracy = numpy_eval()
    print("test set accuracy is %f" % accuracy)
    assert accuracy > 0.95


if __name__ == '__main__':
  unittest.main()
