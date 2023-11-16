#!/usr/bin/env python
import unittest
import numpy as np
import torch
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

class TestSetitemAssign(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_whole_slice(self):
    n1 = np.random.random((3, 3)).astype(np.float32)
    n2 = np.random.random((3, 3)).astype(np.float32)

    s1 = s2 = Tensor(n1).realize()
    t1 = t2 = torch.tensor(n1)
    u1 = u2 = np.array(n1)

    s2[:] = Tensor(n2)
    t2[:] = torch.tensor(n2)
    u2[:] = np.array(n2)

    assert s1.realize().lazydata.realized is s2.realize().lazydata.realized
    assert t1 is t2
    assert u1 is u2
    np.testing.assert_allclose(u2, t2.numpy())
    np.testing.assert_allclose(u2, s2.numpy())

    s1 = s2 = Tensor(n1).realize()
    s2[:, :] = Tensor(n2)

    assert s1.realize().lazydata.realized is s2.realize().lazydata.realized
    np.testing.assert_allclose(u2, s2.numpy())

    s1 = s2 = Tensor(n1).realize()
    s2[0:3, :] = Tensor(n2)

    assert s1.realize().lazydata.realized is s2.realize().lazydata.realized
    # anything assignment through getitem is not real
    # np.testing.assert_allclose(u2, s2.numpy())


  def test_slice_dim0(self):
    pass
  #   n1 = np.random.random((3, 3))
  #   n2 = np.random.random((1, 3))

  #   s1 = s2 = Tensor(n1)
  #   t1 = t2 = torch.tensor(n1)
  #   u1 = u2 = np.array(n1)

  #   s2[:1] = Tensor(n2)
  #   t2[:1] = torch.tensor(n2)
  #   u2[:1] = np.array(n2)

  #   assert s1 is s2
  #   assert t1 is t2
  #   assert u1 is u2

  #   np.testing.assert_allclose(u2, t2.numpy())
  #   # fails silently now, current behavior is no-op
  #   # np.testing.assert_allclose(u2, s2.numpy())


if __name__ == "__main__":
  unittest.main()
