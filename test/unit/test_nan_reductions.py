import unittest

import numpy as np

from tinygrad import Tensor


class TestNaNReductions(unittest.TestCase):
  def test_max_propagates_nan(self):
    out = Tensor([1.0, float("nan"), 3.0]).max().numpy()
    self.assertTrue(np.isnan(out))

  def test_min_propagates_nan(self):
    out = Tensor([1.0, float("nan"), 3.0]).min().numpy()
    self.assertTrue(np.isnan(out))

  def test_max_axis_propagates_nan(self):
    out = Tensor([[float("nan"), 2.0], [1.0, 3.0]]).max(axis=0).numpy()
    np.testing.assert_equal(np.isnan(out), np.array([True, False]))
    np.testing.assert_allclose(out[1:], np.array([3.0]))

  def test_elementwise_maximum_propagates_nan(self):
    out = Tensor([1.0, float("nan")]).maximum(Tensor([float("nan"), 2.0])).numpy()
    self.assertTrue(np.isnan(out[0]))
    self.assertTrue(np.isnan(out[1]))

