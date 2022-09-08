#!/usr/bin/env python
from cmath import exp
import unittest
import numpy as np
from tinygrad.tensor import Tensor, Device
from tinygrad.nn import *
from extra.utils import get_parameters
import torch

@unittest.skipUnless(Device.DEFAULT == Device.CPU, "Not Implemented")
class TestNN(unittest.TestCase):

  def test_max_pool2d(self):
    for size, kernel_size, stride in [
        # kernel size == stride
        ((1, 1, 8, 16), (3, 3), (3, 3)),
        ((1, 1, 8, 8), (3, 3), (3, 3)),
        # kernel size < stride
        ((1, 1, 8, 16), (3, 3), (1, 1)),
        ((1, 1, 8, 8), (3, 3), (1, 1)),
        # kernel size < stride
        ((1, 1, 8, 16), (3, 3), (5, 5)),
        ((1, 1, 8, 8), (3, 3), (5, 5)),
        # rectange kernel
        ((1, 1, 8, 16), (3, 2), (3, 2)),
        ((1, 1, 8, 8), (3, 2), (3, 2)),
      ]:
      array = np.random.uniform(size=size)
      expected = torch.nn.functional.max_pool2d(torch.from_numpy(array), kernel_size=kernel_size, stride=stride).numpy()
      actual = Tensor(array).max_pool2d(kernel_size=kernel_size, stride=stride).numpy()
      print(expected.dtype)
      print(actual.dtype)
      np.testing.assert_allclose(actual, expected)

  def test_batchnorm2d(self, training=False):
    sz = 4

    # create in tinygrad
    Tensor.training = training
    bn = BatchNorm2D(sz, eps=1e-5, track_running_stats=training)
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
    inn = Tensor.randn(2, sz, 3, 3)

    # in tinygrad
    outt = bn(inn)

    # in torch
    toutt = tbn(torch.tensor(inn.cpu().data))

    # close
    np.testing.assert_allclose(outt.data, toutt.detach().numpy(), rtol=5e-5)

    np.testing.assert_allclose(bn.running_mean.data, tbn.running_mean.detach().numpy(), rtol=1e-5)

    # TODO: this is failing
    # np.testing.assert_allclose(bn.running_var.data, tbn.running_var.detach().numpy(), rtol=1e-5)

  def test_batchnorm2d_training(self):
    self.test_batchnorm2d(True)

  def test_linear(self):
    def _test_linear(x):

      # create in tinygrad
      model = Linear(in_dim, out_dim)
      z = model(x)

      # create in torch
      with torch.no_grad():
        torch_layer = torch.nn.Linear(in_dim, out_dim).eval()
        torch_layer.weight[:] = torch.tensor(model.weight.numpy(), dtype=torch.float32)
        torch_layer.bias[:] = torch.tensor(model.bias.numpy(), dtype=torch.float32)
        torch_x = torch.tensor(x.cpu().data, dtype=torch.float32)
        torch_z = torch_layer(torch_x)

      # test
      np.testing.assert_allclose(z.data, torch_z.detach().numpy(), atol=5e-4, rtol=1e-5)

    BS, T, in_dim, out_dim = 4, 2, 8, 16
    _test_linear(Tensor.randn(BS, in_dim))
    _test_linear(Tensor.randn(BS, T, in_dim)) # test with more dims

  def test_conv2d(self):
    BS, C1, H, W = 4, 16, 224, 224
    C2, K, S, P = 64, 7, 2, 1
    
    # create in tinygrad
    layer = Conv2d(C1, C2, kernel_size=K, stride=S, padding=P)

    # create in torch
    with torch.no_grad():
      torch_layer = torch.nn.Conv2d(C1, C2, kernel_size=K, stride=S, padding=P).eval()
      torch_layer.weight[:] = torch.tensor(layer.weight.data, dtype=torch.float32)
      torch_layer.bias[:] = torch.tensor(layer.bias.data, dtype=torch.float32)

    # test
    x = Tensor.uniform(BS, C1, H, W)
    z = layer(x)
    torch_x = torch.tensor(x.cpu().data)
    torch_z = torch_layer(torch_x)
    np.testing.assert_allclose(z.data, torch_z.detach().numpy(), atol=5e-4, rtol=1e-5)

if __name__ == '__main__':
  unittest.main()
