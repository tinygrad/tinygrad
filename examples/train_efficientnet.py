import time
import numpy as np
from efficientnet import EfficientNet
from tinygrad.tensor import Tensor

if __name__ == "__main__":
  Tensor.default_gpu = True
  model = EfficientNet()

  BS = 4

  img = np.zeros((BS,3,224,224), dtype=np.float32)

  st = time.time()
  out = model.forward(Tensor(img))
  et = time.time()
  print("%.2f s" % (et-st))

