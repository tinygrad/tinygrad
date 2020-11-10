import os
import time
import numpy as np
from efficientnet import EfficientNet
from tinygrad.tensor import Tensor

if __name__ == "__main__":
  Tensor.default_gpu = os.getenv("GPU") is not None
  model = EfficientNet()

  BS = 4

  img = np.zeros((BS,3,224,224), dtype=np.float32)

  st = time.time()
  out = model.forward(Tensor(img))
  et = time.time()
  print("forward %.2f s" % (et-st))

  Y = [0]*BS

  y = np.zeros((BS,1000), np.float32)
  y[range(y.shape[0]),Y] = -1000.0
  y = Tensor(y)
  loss = out.logsoftmax().mul(y).mean()

  st = time.time()
  loss.backward()
  et = time.time()
  print("backward %.2f s" % (et-st))

