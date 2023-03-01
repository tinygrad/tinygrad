import math
import unittest
import numpy as np
from scipy import stats
from tinygrad.tensor import Tensor

def helper_test_normal(func, shape=(20, 23), alpha=0.05):
  x = func(*shape).cpu().numpy().flatten()
  _, p = stats.normaltest(x)
  return p >= alpha

def helper_compare_distribution(tinygrad_func, numpy_func, shape=(20, 23), alpha=0.05):
  Tensor.manual_seed(1337)
  np.random.seed(1337)
  x = tinygrad_func(*shape).cpu().numpy().flatten()
  y = numpy_func(shape).flatten()
  _, p = stats.kstest(x, y)
  return p >= alpha

class TestRandomness(unittest.TestCase):

  def test_rand(self):
    self.assertFalse(helper_test_normal(Tensor.rand))

  def test_randn(self):
    self.assertTrue(helper_test_normal(Tensor.randn))

  def test_uniform(self):
    self.assertFalse(helper_test_normal(Tensor.uniform))
    self.assertTrue(helper_compare_distribution(Tensor.uniform, lambda x: np.random.rand(*x) * 2 - 1))

  def test_scaled_uniform(self):
    self.assertFalse(helper_test_normal(Tensor.scaled_uniform))
    self.assertTrue(helper_compare_distribution(Tensor.scaled_uniform, lambda x: (np.random.rand(*x) * 2 - 1) / math.sqrt(math.prod(x))))

  def test_glorot_uniform(self):
    self.assertFalse(helper_test_normal(Tensor.glorot_uniform))
    self.assertTrue(helper_compare_distribution(Tensor.glorot_uniform, lambda x: (np.random.rand(*x) * 2 - 1) * math.sqrt(6 / (x[0] + math.prod(x[1:])))))


if __name__ == "__main__":
  unittest.main()
