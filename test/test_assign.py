#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.tensor import Tensor

class TestAssign(unittest.TestCase):
  def test_simple_assignment(self):
    a = Tensor.arange(16).reshape(4,4)
    b = Tensor.arange(16).reshape(4,4)
    a.realize()
    b.realize()
    ba1 = a.lazydata.realized
    bb1 = b.lazydata.realized
    a += b
    a.realize()
    ba2 = a.lazydata.realized
    assert ba1 == ba2 and ba1 != bb1
    np.testing.assert_allclose(a.numpy(), (np.arange(16)*2).reshape((4,4)))

  def test_permuted_assignment(self):
    a = Tensor.arange(16).reshape(4,4)
    b = Tensor.arange(16).reshape(4,4)
    a.realize()
    b.realize()
    ba1 = a.lazydata.realized
    bb1 = b.lazydata.realized
    a = a.permute(1,0)
    a += b
    a.realize()
    ba2 = a.lazydata.realized
    assert ba1 != ba2 and ba1 != bb1
    np.testing.assert_allclose(a.numpy(), np.arange(16).reshape((4,4)) + np.arange(16).reshape((4,4)).transpose(1,0))

  def test_post_permuted_assignment(self):
    a = Tensor.arange(16).reshape(4,4)
    b = Tensor.arange(16).reshape(4,4)
    a.realize()
    b.realize()
    ba1 = a.lazydata.realized
    bb1 = b.lazydata.realized
    a.assign(a.permute(1,0) + b)   # this should not work!
    a.realize()
    ba2 = a.lazydata.realized
    # NOTE: don't test that it's assigned
    assert ba1 == ba2 and ba1 != bb1
    np.testing.assert_allclose(a.numpy(), np.arange(16).reshape((4,4)) + np.arange(16).reshape((4,4)).transpose(1,0))

  # TODO: is there a way to sneak in a permute such that it returns the wrong answer?

if __name__ == "__main__":
  unittest.main()
