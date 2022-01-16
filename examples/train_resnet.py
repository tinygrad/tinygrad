#!/usr/bin/env python3
import os
import numpy as np
import random
from PIL import Image

from tinygrad.tensor import Device
from extra.utils import get_parameters
from extra.training import train, evaluate
from models.resnet import ResNet18, ResNet34, ResNet50
from tinygrad.optim import Adam
from test.test_mnist import fetch_mnist

from tinygrad.optim import Adam

class ComposeTransforms:
  def __init__(self, trans):
    self.trans = trans

  def __call__(self, x):
    for t in self.trans:
      x = t(x)
    return x

if __name__ == "__main__":
  model = ResNet18(num_classes=10, pretrained=True)

  X_train, Y_train, X_test, Y_test = fetch_mnist()
  X_train = X_train.reshape(-1, 28, 28).astype(np.uint8)
  X_test = X_test.reshape(-1, 28, 28).astype(np.uint8)
  lr = 5e-5
  transform = ComposeTransforms([
    lambda x: [Image.fromarray(xx, mode='L').resize((64, 64)) for xx in x],
    lambda x: np.stack([np.asarray(xx) for xx in x], 0),
    lambda x: x / 255.0,
    lambda x: np.tile(np.expand_dims(x, 1), (1, 3, 1, 1)).astype(np.float32),
  ])
  for i in range(10):
    optim = Adam(get_parameters(model), lr=lr)
    train(model, X_train, Y_train, optim, 50, BS=32, transform=transform)
    acc, Y_test_preds = evaluate(model, X_test, Y_test, num_classes=10, return_predict=True, transform=transform)
    lr /= 1.2
    print(f'reducing lr to {lr:.4f}')
