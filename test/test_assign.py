#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.ops import Device
from tinygrad.helpers import dtypes

N = 200  # has to be bigger than the cache to fail

class TestAssign(unittest.TestCase):
  def test_simple_assignment(self):
    a = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    b = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    a.realize()
    b.realize()
    ba1 = a.lazydata.realized
    bb1 = b.lazydata.realized
    a += b
    a.realize()
    ba2 = a.lazydata.realized
    assert ba1 == ba2 and ba1 != bb1
    np.testing.assert_allclose(a.numpy(), (np.arange(N*N)*2).reshape((N,N)))

  @unittest.skipIf(Device.DEFAULT == "CPU" or Device.DEFAULT == "TORCH", "questionable tests")
  def test_permuted_assignment(self):
    a = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    b = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
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
    a = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    b = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
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
    np.testing.assert_allclose(a.numpy(), np.arange(N*N).reshape((N,N)) + np.arange(N*N).reshape((N,N)).transpose(1,0))

  # TODO: is there a way to sneak in a permute such that it returns the wrong answer?

  def test_cast_assignment(self):
    a = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    a.realize()
    oba1 = a.lazydata.output_buffer
    a.assign(a.cast(dtypes.int32).realize())
    a.realize()
    oba2 = a.lazydata.output_buffer
    assert oba1 is None and oba2 is None
    np.testing.assert_allclose(a.numpy(), np.arange(N*N,dtype=np.int32).reshape((N,N)))

if __name__ == "__main__":
  unittest.main()
