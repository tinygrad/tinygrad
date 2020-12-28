import os
import time
import numpy as np
from extra.efficientnet import EfficientNet
from tinygrad.tensor import Tensor
from extra.utils import get_parameters, fetch
from tqdm import trange
import tinygrad.optim as optim
import io
import tarfile
import pickle

class TinyConvNet:
  def __init__(self, classes=10):
    conv = 3
    inter_chan, out_chan = 8, 16   # for speed
    self.c1 = Tensor.uniform(inter_chan,3,conv,conv)
    self.c2 = Tensor.uniform(out_chan,inter_chan,conv,conv)
    self.l1 = Tensor.uniform(out_chan*6*6, classes)

  def forward(self, x):
    x = x.conv2d(self.c1).relu().max_pool2d()
    x = x.conv2d(self.c2).relu().max_pool2d()
    x = x.reshape(shape=[x.shape[0], -1])
    return x.dot(self.l1).logsoftmax()

def load_cifar():
  tt = tarfile.open(fileobj=io.BytesIO(fetch('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')), mode='r:gz')
  db = pickle.load(tt.extractfile('cifar-10-batches-py/data_batch_1'), encoding="bytes")
  X = db[b'data'].reshape((-1, 3, 32, 32))
  Y = np.array(db[b'labels'])
  return X, Y

if __name__ == "__main__":
  X_train, Y_train = load_cifar()
  classes = 10

  Tensor.default_gpu = os.getenv("GPU") is not None
  TINY = os.getenv("TINY") is not None
  TRANSFER = os.getenv("TRANSFER") is not None
  if TINY:
    model = TinyConvNet(classes)
  elif TRANSFER:
    model = EfficientNet(int(os.getenv("NUM", "0")), classes, has_se=True)
    model.load_weights_from_torch()
  else:
    model = EfficientNet(int(os.getenv("NUM", "0")), classes, has_se=False)

  parameters = get_parameters(model)
  print("parameters", len(parameters))
  optimizer = optim.Adam(parameters, lr=0.001)

  #BS, steps = 16, 32
  BS, steps = 64 if TINY else 16, 2048

  for i in (t := trange(steps)):
    samp = np.random.randint(0, X_train.shape[0], size=(BS))

    img = X_train[samp].astype(np.float32)

    st = time.time()
    out = model.forward(Tensor(img))
    fp_time = (time.time()-st)*1000.0

    Y = Y_train[samp]
    y = np.zeros((BS,classes), np.float32)
    y[range(y.shape[0]),Y] = -classes
    y = Tensor(y)
    loss = out.logsoftmax().mul(y).mean()

    optimizer.zero_grad()

    st = time.time()
    loss.backward()
    bp_time = (time.time()-st)*1000.0

    st = time.time()
    optimizer.step()
    opt_time = (time.time()-st)*1000.0

    #print(out.cpu().data)

    st = time.time()
    loss = loss.cpu().data
    cat = np.argmax(out.cpu().data, axis=1)
    accuracy = (cat == Y).mean()
    finish_time = (time.time()-st)*1000.0

    # printing
    t.set_description("loss %.2f accuracy %.2f -- %.2f + %.2f + %.2f + %.2f = %.2f" %
      (loss, accuracy,
      fp_time, bp_time, opt_time, finish_time,
      fp_time + bp_time + opt_time + finish_time))

    del out, y, loss
