import os
import traceback
import time
import numpy as np
from models.efficientnet import EfficientNet
from tinygrad.tensor import Tensor
from extra.utils import get_parameters
from tqdm import trange
from tinygrad.nn import BatchNorm2D
import tinygrad.nn.optim as optim
from datasets import fetch_cifar

class TinyConvNet:
  def __init__(self, classes=10):
    conv = 3
    inter_chan, out_chan = 8, 16   # for speed
    self.c1 = Tensor.uniform(inter_chan,3,conv,conv)
    #self.bn1 = BatchNorm2D(inter_chan)
    self.c2 = Tensor.uniform(out_chan,inter_chan,conv,conv)
    #self.bn2 = BatchNorm2D(out_chan)
    self.l1 = Tensor.uniform(out_chan*6*6, classes)

  def forward(self, x):
    x = x.conv2d(self.c1).relu().max_pool2d()
    #x = self.bn1(x)
    x = x.conv2d(self.c2).relu().max_pool2d()
    #x = self.bn2(x)
    x = x.reshape(shape=[x.shape[0], -1])
    return x.dot(self.l1)

if __name__ == "__main__":
  IMAGENET = os.getenv("IMAGENET") is not None
  classes = 1000 if IMAGENET else 10

  TINY = os.getenv("TINY") is not None
  TRANSFER = os.getenv("TRANSFER") is not None
  if TINY:
    model = TinyConvNet(classes)
  elif TRANSFER:
    model = EfficientNet(int(os.getenv("NUM", "0")), classes, has_se=True)
    model.load_from_pretrained()
  else:
    model = EfficientNet(int(os.getenv("NUM", "0")), classes, has_se=False)

  parameters = get_parameters(model)
  print("parameter count", len(parameters))
  optimizer = optim.Adam(parameters, lr=0.001)

  BS, steps = int(os.getenv("BS", "64" if TINY else "16")), int(os.getenv("STEPS", "2048"))
  print("training with batch size %d for %d steps" % (BS, steps))

  if IMAGENET:
    from datasets.imagenet import fetch_batch
    from multiprocessing import Process, Queue
    def loader(q):
      while 1:
        try:
          q.put(fetch_batch(BS))
        except Exception:
          traceback.print_exc()
    q = Queue(16)
    for i in range(2):
      p = Process(target=loader, args=(q,))
      p.daemon = True
      p.start()
  else:
    X_train, Y_train = fetch_cifar()

  Tensor.training = True
  for i in (t := trange(steps)):
    if IMAGENET:
      X, Y = q.get(True)
    else:
      samp = np.random.randint(0, X_train.shape[0], size=(BS))
      X, Y = X_train[samp], Y_train[samp]

    st = time.time()
    out = model.forward(Tensor(X.astype(np.float32), requires_grad=False))
    fp_time = (time.time()-st)*1000.0

    y = np.zeros((BS,classes), np.float32)
    y[range(y.shape[0]),Y] = -classes
    y = Tensor(y, requires_grad=False)
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
