#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.tensor import GPU, ANE, Device
from tinygrad.nn import *
from extra.utils import get_parameters
import torch

class TestNN(unittest.TestCase):
  device = Device.CPU

  def test_batchnorm2d(self, training=False):
    sz = 4

    # create in tinygrad
    bn = BatchNorm2D(sz, eps=1e-5, training=training, track_running_stats=training)
    bn.weight = Tensor.randn(sz)
    bn.bias = Tensor.randn(sz)
    bn.running_mean = Tensor.randn(sz)
    bn.running_var = Tensor.randn(sz)
    bn.running_var.data[bn.running_var.data < 0] = 0

    # create in torch
    with torch.no_grad():
      tbn = torch.nn.BatchNorm2d(sz).eval()
      tbn.training = training
      tbn.weight[:] = torch.tensor(bn.weight.data)
      tbn.bias[:] = torch.tensor(bn.bias.data)
      tbn.running_mean[:] = torch.tensor(bn.running_mean.data)
      tbn.running_var[:] = torch.tensor(bn.running_var.data)

    np.testing.assert_allclose(bn.running_mean.data, tbn.running_mean.detach().numpy(), rtol=1e-5)
    np.testing.assert_allclose(bn.running_var.data, tbn.running_var.detach().numpy(), rtol=1e-5)

    # trial
    inn = Tensor.randn(2, sz, 3, 3, device=self.device)

    # in tinygrad
    outt = bn(inn)

    # in torch
    toutt = tbn(torch.tensor(inn.cpu().data))

    # close
    np.testing.assert_allclose(outt.data, toutt.detach().numpy(), rtol=5e-5)

    np.testing.assert_allclose(bn.running_mean.data, tbn.running_mean.detach().numpy(), rtol=1e-5)

    # TODO: this is failing
    #np.testing.assert_allclose(bn.running_var.data, tbn.running_var.detach().numpy(), rtol=1e-5)

  def test_batchnorm2d_training(self):
    self.test_batchnorm2d(True)

@unittest.skipUnless(GPU, "Requires GPU")
class TestNNGPU(TestNN):
  device = Device.GPU

  @unittest.skip("Tests not added")
  def test_batchnorm2d(self): pass

  @unittest.skip("Tests not added")
  def test_batchnorm2d_training(self): pass


@unittest.skipUnless(ANE, "Requires ANE")
class TestNNANE(TestNN):
  device=Device.ANE

  @unittest.skip("Tests not added")
  def test_batchnorm2d(self): pass

  @unittest.skip("Tests not added")
  def test_batchnorm2d_training(self): pass


if __name__ == '__main__':
  unittest.main()
