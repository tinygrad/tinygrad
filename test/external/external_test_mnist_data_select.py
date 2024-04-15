#!/bin/bash
from tinygrad import Tensor
from tinygrad.features.datasets import mnist

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = mnist()
  samples = Tensor.randint(512, high=X_train.shape[0])
  select = X_train[samples].realize()
