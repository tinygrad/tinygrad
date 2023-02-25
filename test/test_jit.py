#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.tensor import Tensor, Device
from tinygrad.jit import TinyJit

@unittest.skipUnless(Device.DEFAULT == "GPU", "JIT is only for GPU")
class TestJit(unittest.TestCase):
  def test_simple_jit(self):
    @TinyJit
    def add(a, b): return (a+b).realize()
    for _ in range(3):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      c = add(a, b)
      np.testing.assert_equal(c.numpy(), a.numpy()+b.numpy())

  def test_kwargs_jit(self):
    @TinyJit
    def add_kwargs(first, second): return (first+second).realize()
    for _ in range(3):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      c = add_kwargs(first=a, second=b)
      np.testing.assert_equal(c.numpy(), a.numpy()+b.numpy())

  def test_array_jit(self):
    @TinyJit
    def add_array(a, arr): return (a+arr[0]).realize()
    for i in range(3):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      a.realize(), b.realize()
      c = add_array(a, [b])
      if i == 2:
        # should fail once jitted since jit can't handle arrays
        np.testing.assert_equal(np.any(np.not_equal(c.numpy(),a.numpy()+b.numpy())), True)
      else:
        np.testing.assert_equal(c.numpy(), a.numpy()+b.numpy())

if __name__ == '__main__':
  unittest.main()