#!/usr/bin/env python
import unittest
import numpy as np
from datasets import *

class TestDatasets(unittest.TestCase):
  def test_mnist(self):
    ds = MNIST(root='./data', train=True, download=True)
    self.__test_dataset(ds, sample_shape=(1, 28, 28))
    self.__test_dataloader(ds, sample_shape=(1, 28, 28))

  def test_cifar10(self):
    ds = CIFAR10(root='./data', train=True, download=True)
    self.__test_dataset(ds, sample_shape=(3, 32, 32))
    self.__test_dataloader(ds, sample_shape=(3, 32, 32))

  def test_cifar100(self):
    ds = CIFAR100(root='./data', train=True, download=True)
    self.__test_dataset(ds, sample_shape=(3, 32, 32))
    self.__test_dataloader(ds, sample_shape=(3, 32, 32))

  def __test_dataset(self, dataset, sample_shape):
    x, y = next(iter(dataset))
    dims = x.shape
    assert x.min() >= 0 
    assert x.max() <= 1.0
    assert list(dims) == list(sample_shape)
    assert x.dtype == np.float32
    assert y.dtype == np.int32

  def __test_dataloader(self, dataset, sample_shape):
    batch_size = 4
    dl = dataset.dataloader(batch_size=batch_size) # or DataLoader(dataset, ...) like torch
    x, y = next(iter(dl))
    B, *dims = x.shape
    assert B == batch_size
    assert list(dims) == list(sample_shape)
    assert x.dtype == np.float32
    assert y.dtype == np.int32

if __name__ == '__main__':
  unittest.main()
