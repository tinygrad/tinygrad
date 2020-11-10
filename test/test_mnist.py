#!/usr/bin/env python
import os
import unittest
import numpy as np
from tinygrad.tensor import Tensor, GPU
from tinygrad.utils import layer_init_uniform, fetch
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
    self.l1 = Tensor(layer_init_uniform(784, 128))
    self.l2 = Tensor(layer_init_uniform(128, 10))

  def parameters(self):
    return [self.l1, self.l2]

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

  def parameters(self):
    return [self.l1, self.c1, self.c2]

  def forward(self, x):
    x = x.reshape(shape=(-1, 1, 28, 28)) # hacks
    x = x.conv2d(self.c1).relu().max_pool2d()
    x = x.conv2d(self.c2).relu().max_pool2d()
    x = x.reshape(shape=[x.shape[0], -1])
    return x.dot(self.l1).logsoftmax()

def train(model, optim, steps, BS=128, gpu=False):
  losses, accuracies = [], []
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
    loss = out.mul(y).mean()
    loss.backward()
    optim.step()
    
    cat = np.argmax(out.cpu().data, axis=1)
    accuracy = (cat == Y).mean()
    
    # printing
    loss = loss.cpu().data
    losses.append(loss)
    accuracies.append(accuracy)
    t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

def evaluate(model, gpu=False):
  def numpy_eval():
    Y_test_preds_out = model.forward(Tensor(X_test.reshape((-1, 28*28)).astype(np.float32), gpu=gpu)).cpu()
    Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
    return (Y_test == Y_test_preds).mean()

  accuracy = numpy_eval()
  print("test set accuracy is %f" % accuracy)
  assert accuracy > 0.95

class TestMNIST(unittest.TestCase):
  @unittest.skipUnless(GPU, "Requires GPU")
  @unittest.expectedFailure
  def test_conv_gpu(self):
    np.random.seed(1337)
    model = TinyConvNet()
    [x.cuda_() for x in model.parameters()]
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    train(model, optimizer, steps=1000, gpu=True)
    evaluate(model, gpu=True)

  def test_conv(self):
    np.random.seed(1337)
    model = TinyConvNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, optimizer, steps=200)
    evaluate(model)

  @unittest.skipUnless(GPU, "Requires GPU")
  def test_sgd_gpu(self):
    np.random.seed(1337)
    model = TinyBobNet()
    [x.cuda_() for x in model.parameters()]
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    train(model, optimizer, steps=1000, gpu=True)
    evaluate(model, gpu=True)
    
  def test_sgd(self):
    np.random.seed(1337)
    model = TinyBobNet()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    train(model, optimizer, steps=1000)
    evaluate(model)
    
  def test_rmsprop(self):
    np.random.seed(1337)
    model = TinyBobNet()
    optimizer = optim.RMSprop(model.parameters(), lr=0.0002)
    train(model, optimizer, steps=1000)
    evaluate(model)

if __name__ == '__main__':
  unittest.main()
