#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.lazy import LAZY
from tinygrad.ops import GlobalCounters
from tinygrad.graph import nm

N = 200  # has to be bigger than the cache to fail

class TestAssign(unittest.TestCase):
  def test_simple_assignment(self):
    a = Tensor.arange(N*N).reshape(N,N)
    b = Tensor.arange(N*N).reshape(N,N)
    a.realize()
    b.realize()
    ba1 = a.lazydata.realized
    bb1 = b.lazydata.realized
    a += b
    a.realize()
    ba2 = a.lazydata.realized
    if LAZY: assert ba1 == ba2 and ba1 != bb1
    np.testing.assert_allclose(a.numpy(), (np.arange(N*N)*2).reshape((N,N)))

  def test_permuted_assignment(self):
    a = Tensor.arange(N*N).reshape(N,N)
    b = Tensor.arange(N*N).reshape(N,N)
    a.realize()
    b.realize()
    ba1 = a.lazydata.realized
    bb1 = b.lazydata.realized
    a = a.permute(1,0)
    a += b
    a.realize()
    ba2 = a.lazydata.realized
    assert ba1 != ba2 and ba1 != bb1
    np.testing.assert_allclose(a.numpy(), np.arange(N*N).reshape((N,N)) + np.arange(N*N).reshape((N,N)).transpose(1,0))

  def test_post_permuted_assignment(self):
    a = Tensor.arange(N*N).reshape(N,N)
    b = Tensor.arange(N*N).reshape(N,N)
    a.realize()
    b.realize()
    #GlobalCounters.cache = []
    ba1 = a.lazydata.realized
    bb1 = b.lazydata.realized
    a.assign(a.permute(1,0) + b)   # this should not work!
    a.realize()
    ba2 = a.lazydata.realized
    # NOTE: don't test that it's assigned
    #assert ba1 == ba2 and ba1 != bb1

    """
    if len(GlobalCounters.cache):
      runner, args = GlobalCounters.cache[0]
      b0, b1, b2 = args
      print(nm(b0), id(b0.cl))
      print(nm(b1), id(b1.cl))
      print(nm(b2), id(b2.cl))
    """

    np.testing.assert_allclose(a.numpy(), np.arange(N*N).reshape((N,N)) + np.arange(N*N).reshape((N,N)).transpose(1,0))

  # TODO: is there a way to sneak in a permute such that it returns the wrong answer?

if __name__ == "__main__":
  unittest.main()
