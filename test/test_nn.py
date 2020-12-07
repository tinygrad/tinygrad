#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.nn import *
import torch

class TestNN(unittest.TestCase):
  def test_batchnorm2d(self):
    sz = 4

    for training in [False, True]:
      # create in tinygrad
      bn = BatchNorm2D(sz, eps=1e-5, training=training)
      bn.weight = Tensor.randn(sz)
      bn.bias = Tensor.randn(sz)
      bn.running_mean = Tensor.randn(sz)
      bn.running_var = Tensor.randn(sz)
      bn.running_var.data[bn.running_var.data < 0] = 0
      #debug
      bn.running_mean = Tensor.zeros(sz)
      bn.running_var = Tensor.zeros(sz)

      # create in torch
      #with torch.no_grad():
      if True:
        tbn = torch.nn.BatchNorm2d(sz,momentum=None,eps=1e-5).eval()
        tbn.training=training
        tbn.track_running_stats=False
        tbn.weight[:] = torch.tensor(bn.weight.data) #, requires_grad=True)
        tbn.bias[:] = torch.tensor(bn.bias.data) #,requires_grad=True)
        tbn.running_mean[:] = torch.tensor(bn.running_mean.data)
        tbn.running_var[:] = torch.tensor(bn.running_var.data)

      # trial
      inn = Tensor.randn(2, sz, 3, 3)

      # in tinygrad
      outt = bn(inn)

      # in torch
      toutt = tbn(torch.tensor(inn.data))

      # close
      np.testing.assert_allclose(outt.data, toutt.detach().numpy(), rtol=1e-5)


if __name__ == '__main__':
  unittest.main()
