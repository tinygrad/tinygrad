#!/bin/bash
from tinygrad import Tensor
from extra.datasets import fetch_mnist

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = fetch_mnist(tensors=True)
  # samples = Tensor.randint(512, high=X_train.shape[0])
  samples = Tensor.arange(10)
  select = X_train[samples].realize();
  # print(select.shape)
  print(select[:10, 0, 10, 10].numpy())
