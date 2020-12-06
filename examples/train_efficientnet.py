import os
import time
import numpy as np
from extra.efficientnet import EfficientNet
from tinygrad.tensor import Tensor
from tinygrad.utils import get_parameters

if __name__ == "__main__":
  Tensor.default_gpu = os.getenv("GPU") is not None
  model = EfficientNet(int(os.getenv("NUM", "0")))
  parameters = get_parameters(model)
  print(len(parameters))

  BS = 16
  img = np.zeros((BS,3,224,224), dtype=np.float32)

  for i in range(32):
    print("running batch %d, %d tensors allocated" % (i, Tensor.allocated))

    st = time.time()
    out = model.forward(Tensor(img))
    et = time.time()
    print("forward %.2f s" % (et-st))

    Y = [0]*BS

    y = np.zeros((BS,1000), np.float32)
    y[range(y.shape[0]),Y] = -1000.0
    y = Tensor(y)
    loss = out.logsoftmax().mul(y).mean()

    # zero grad
    for p in parameters:
      p.grad = None

    st = time.time()
    loss.backward()
    et = time.time()
    print("backward %.2f s" % (et-st))

    del out, y, loss

