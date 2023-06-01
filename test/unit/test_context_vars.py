import pathlib
import unittest
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import Context
import tinygrad.mlops as mlops
from tinygrad.nn import BatchNorm2d

class TestContextVars(unittest.TestCase):
  def get_tensor_with_grad(self):
    a = Tensor.ones(1, requires_grad=True)
    a.relu().backward()
    return a

  @Context(no_grad=True)
  def test_no_grad(self):
    assert not self.get_tensor_with_grad().grad
    with Context(no_grad=False):
      assert self.get_tensor_with_grad().grad
    assert not self.get_tensor_with_grad().grad
        
  @Context(training=False)
  def test_training_dropout(self):
    x = Tensor.ones(1)
    assert x == x.dropout()
    with Context(training=True):
      x = Tensor.ones(1)
      assert x != x.dropout()
    x = Tensor.ones(1)
    assert x == x.dropout() 

  @Context(training=False)
  def test_training_batchnorm2d(self):
    bn = BatchNorm2d(1)
    bn(Tensor.ones(1, 1, 2, 2))
    assert not bn.running_mean.numpy().any()

    with Context(training=True):
      bn = BatchNorm2d(1)
      bn(Tensor.ones(1, 1, 2, 2))
      assert bn.running_mean.numpy().any()

    bn = BatchNorm2d(1)
    bn(Tensor.ones(1, 1, 2, 2))
    assert not bn.running_mean.numpy().any()

if __name__ == "__main__":
  unittest.main()
