#!/usr/bin/env python3
import os
import numpy as np
import random

from tinygrad.tensor import Device
from extra.utils import get_parameters
from extra.training import train, evaluate
from extra.transformer import Transformer
from tinygrad.optim import Adam

# dataset idea from https://github.com/karpathy/minGPT/blob/master/play_math.ipynb
def make_dataset():
  ds = []
  for i in range(100):
    for j in range(100):
      s = i+j
      ds.append([i//10, i%10, j//10, j%10, s//100, (s//10)%10, s%10])
  random.shuffle(ds)
  ds = np.array(ds)
  ds_X = ds[:, 0:6]
  ds_Y = np.copy(ds[:, 1:])
  ds_X_train, ds_X_test = ds_X[0:8000], ds_X[8000:]
  ds_Y_train, ds_Y_test = ds_Y[0:8000], ds_Y[8000:]

  return ds_X_train, ds_Y_train, ds_X_test, ds_Y_test

from tinygrad.optim import Adam
if __name__ == "__main__":
  model = Transformer(10, 6, 2, 128, 4)

  X_train, Y_train, X_test, Y_test = make_dataset()
  optim = Adam(get_parameters(model), lr=0.001)

  for i in range(5):
    train(model, X_train, Y_train, optim, 500, BS=32, device=Device.GPU if os.getenv("GPU") else Device.CPU)
    evaluate(model, X_test, Y_test, num_classes=10)


