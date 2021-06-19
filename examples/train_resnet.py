#!/usr/bin/env python3
import os
import numpy as np
import random
from PIL import Image

from tinygrad.tensor import Device
from extra.utils import get_parameters
from extra.training import train, evaluate
from extra.resnet import ResNet18, ResNet34, ResNet50
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
  model = ResNet18(num_classes=10, pretrained=False)

  X_train, Y_train, X_test, Y_test = fetch_mnist()
  X_train, Y_train, X_test, Y_test = fetch_mnist()
  lr = 0.003
  transform = ComposeTransforms([
    lambda x: x.reshape(-1, 28, 28), 
    lambda x: [Image.fromarray(xx, mode='L').resize((224, 224)) for xx in x],
    lambda x: np.expand_dims(np.stack([np.asarray(xx) for xx in x], 0), 1),
    lambda x: np.tile(x, (1, 3, 1, 1)),
    lambda x: (x / 255.0).astype(np.float32),
  ])
  for i in range(10):
    optim = Adam(get_parameters(model), lr=lr)
    train(model, X_train, Y_train, optim, 50, BS=64, transform=transform)
    acc, Y_test_preds = evaluate(model, X_test, Y_test, num_classes=10, return_predict=True, transform=transform)
    lr /= 1.2
    print(f'reducing lr to {lr:.4f}')
    if acc > 0.998:
      wrong=0
      for k in range(len(Y_test_preds)):
        if (Y_test_preds[k] != Y_test[k]).any():
          wrong+=1
          a,b,c,x = X_test[k,:2], X_test[k,2:4], Y_test[k,-3:], Y_test_preds[k,-3:]
          print(f'{a[0]}{a[1]} + {b[0]}{b[1]} = {x[0]}{x[1]}{x[2]} (correct: {c[0]}{c[1]}{c[2]})')
      print(f'Wrong predictions: {wrong}, acc = {acc:.4f}')
