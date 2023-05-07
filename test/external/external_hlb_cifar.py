#!/usr/bin/env python3
from examples.hlb_cifar10 import SpeedyResNet, fetch_batch
from examples.hlb_cifar10_torch import SpeedyResNet as SpeedyResNetTorch
from datasets import fetch_cifar
from test.models.test_end2end import compare_tiny_torch

if __name__ == "__main__":
  X_test, Y_test = fetch_cifar(train=False)
  X, Y = fetch_batch(X_test, Y_test, 32)
  print(X.shape, Y.shape)
  model = SpeedyResNet()
  model_torch = SpeedyResNetTorch()
  compare_tiny_torch(model, model_torch, X, Y)
