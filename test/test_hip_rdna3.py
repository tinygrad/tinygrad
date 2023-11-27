#!/usr/bin/env python
import unittest
from tinygrad import Tensor
from tinygrad.helpers import dtypes, getenv
from examples.beautiful_mnist import Model as MNIST
from examples.hlb_cifar10 import SpeedyResNet


class TestHIPCompilationRDNA(unittest.TestCase):
  def test_compile_hip_mnist(self):
    if not getenv("HIP"): raise unittest.SkipTest("testing HIP->rdna3 compilation needs HIP=1")
    model = MNIST()

    input = Tensor.rand(512,1,28,28)
    output = model(input)
    output.numpy()

  def test_compile_hip_speedyresnet(self):
    if not getenv("HIP"): raise unittest.SkipTest("testing HIP->rdna3 compilation needs HIP=1")
    W = Tensor.rand(12,3,2,2)
    model = SpeedyResNet(W)

    input = Tensor.rand(512, 3, 32, 32)
    output = model(input)
    output.numpy()

  @unittest.expectedFailure
  def test_compile_hip_speedyresnet_hf(self):
    if not getenv("HIP"): raise unittest.SkipTest("testing HIP->rdna3 compilation needs HIP=1")
    Tensor.default_type = dtypes.float16

    W = Tensor.rand(12,3,2,2)
    model = SpeedyResNet(W)

    input = Tensor.rand(512, 3, 32, 32)
    output = model(input)
    output.numpy()

if __name__ == "__main__":
  unittest.main()