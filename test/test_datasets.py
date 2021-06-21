#!/usr/bin/env python
import unittest
import numpy as np
from datasets import *

class TestDatasets(unittest.TestCase):
  def test_mnist(self):
    ds = MNIST(root='./data', train=True, download=True)
    x, y = next(iter(ds))
    C, H, W = x.shape
    assert x.min() >= 0 
    assert x.max() <= 1.0
    assert C == 1 
    assert H == 28 
    assert W == 28
    assert x.dtype == np.float32
    assert y.dtype == np.int32
    self.__test_dataloader(ds, sample_shape=(C, H, W))

  def test_cifar10(self):
    ds = CIFAR10(root='./data', train=True, download=True)
    x, y = next(iter(ds))
    C, H, W = x.shape
    assert x.min() >= 0 
    assert x.max() <= 1.0
    assert C == 3
    assert H == 32
    assert W == 32
    assert x.dtype == np.float32
    assert y.dtype == np.int32
    self.__test_dataloader(ds, sample_shape=(C, H, W))

  def test_cifar100(self):
    ds = CIFAR100(root='./data', train=True, download=True)
    x, y = next(iter(ds))
    C, H, W = x.shape
    assert x.min() >= 0 
    assert x.max() <= 1.0
    assert C == 3
    assert H == 32
    assert W == 32
    assert x.dtype == np.float32
    assert y.dtype == np.int32
    self.__test_dataloader(ds, sample_shape=(C, H, W))

  def __test_dataloader(self, dataset, sample_shape):
    batch_size = 4
    dl = dataset.dataloader(batch_size=batch_size) # or DataLoader(dataset, ...) like torch
    x, y = next(iter(dl))
    B, *dims = x.shape
    assert B == batch_size
    assert list(sample_shape) == list(dims)
    assert x.dtype == np.float32
    assert y.dtype == np.int32


if __name__ == '__main__':
  unittest.main()
