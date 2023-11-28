#!/usr/bin/env python
import unittest
from tinygrad import Tensor, Device
from tinygrad.helpers import dtypes
from examples.beautiful_mnist import Model as MNIST
from examples.hlb_cifar10 import SpeedyResNet

@unittest.skipIf(Device.DEFAULT != "HIP", reason="testing HIP->rdna3 compilation needs HIP=1")
class TestHIPCompilationRDNA(unittest.TestCase):
  def test_compile_hip_mnist(self):
    model = MNIST()

    input = Tensor.rand(512,1,28,28)
    output = model(input)
    output.numpy()

  def test_compile_hip_speedyresnet(self):
    W = Tensor.rand(12,3,2,2)
    model = SpeedyResNet(W)

    input = Tensor.rand(512, 3, 32, 32)
    output = model(input)
    output.numpy()

  @unittest.expectedFailure
  def test_compile_hip_speedyresnet_hf(self):
    Tensor.default_type = dtypes.float16

    W = Tensor.rand(12,3,2,2)
    model = SpeedyResNet(W)

    input = Tensor.rand(512, 3, 32, 32)
    output = model(input)
    output.numpy()

if __name__ == "__main__":
  unittest.main()