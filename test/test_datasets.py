#!/usr/bin/env python
import unittest
import numpy as np
from datasets import *

class TestDatasets(unittest.TestCase):
  def test_datasets(self):
    for ds_cls in [MNIST, CIFAR10, CIFAR100]:
      print(f'Testing {ds_cls}')
      ds = ds_cls(root='./data', train=True, download=True)
      dl = ds.dataloader(batch_size=4) # or DataLoader(dataset, ...) like torch
      self.__test_dataiterator(ds)
      self.__test_dataiterator(dl, batch_size=dl.batch_size)

  def __test_dataiterator(self, di, batch_size=None):
    x, y = next(iter(di))
    if batch_size is None:
      dims, sample_shape = x.shape, di.sample_shape
    else:
      B, *dims = x.shape
      sample_shape = di.dataset.sample_shape
      assert B == batch_size
    assert x.min() >= 0 
    assert x.max() <= 1.0
    assert list(dims) == list(sample_shape)
    assert x.dtype == np.float32
    assert y.dtype == np.int32

if __name__ == '__main__':
  unittest.main()
