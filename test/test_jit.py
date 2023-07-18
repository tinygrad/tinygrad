#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.tensor import Tensor, Device
from tinygrad.jit import TinyJit, JIT_SUPPORTED_DEVICE

# NOTE: METAL fails, might be platform and optimization options dependent.
@unittest.skipUnless(Device.DEFAULT in JIT_SUPPORTED_DEVICE and Device.DEFAULT not in ["METAL", "WEBGPU"], f"no JIT on {Device.DEFAULT}")
class TestJit(unittest.TestCase):
  def test_simple_jit(self):
    @TinyJit
    def add(a, b): return (a+b).realize()
    for _ in range(5):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      c = add(a, b)
      np.testing.assert_equal(c.numpy(), a.numpy()+b.numpy())

  def test_jit_shape_mismatch(self):
    @TinyJit
    def add(a, b): return (a+b).realize()
    for _ in range(3):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      c = add(a, b)
    bad = Tensor.randn(20, 20)
    with self.assertRaises(AssertionError):
      add(a, bad)

  def test_jit_duplicate_fail(self):
    # the jit doesn't support duplicate arguments
    @TinyJit
    def add(a, b): return (a+b).realize()
    a = Tensor.randn(10, 10)
    with self.assertRaises(AssertionError):
      add(a, a)

  def test_kwargs_jit(self):
    @TinyJit
    def add_kwargs(first, second): return (first+second).realize()
    for _ in range(5):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      c = add_kwargs(first=a, second=b)
      np.testing.assert_equal(c.numpy(), a.numpy()+b.numpy())

  def test_array_jit(self):
    @TinyJit
    def add_array(a, arr): return (a+arr[0]).realize()
    for i in range(5):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      a.realize(), b.realize()
      c = add_array(a, [b])
      if i >= 2:
        # should fail once jitted since jit can't handle arrays
        np.testing.assert_equal(np.any(np.not_equal(c.numpy(),a.numpy()+b.numpy())), True)
      else:
        np.testing.assert_equal(c.numpy(), a.numpy()+b.numpy())

  def test_method_jit(self):
    class Fun:
      def __init__(self):
        self.a = Tensor.randn(10, 10)
      @TinyJit
      def __call__(self, b:Tensor) -> Tensor:
        return (self.a+b).realize()
    fun = Fun()
    for _ in range(5):
      b = Tensor.randn(10, 10)
      c = fun(b)
      np.testing.assert_equal(c.numpy(), fun.a.numpy()+b.numpy())

  def test_jit_handles_constant_input(self):
    @TinyJit
    def add(a, b): return (a+b).realize()
    for _ in range(3):
      a = Tensor.randn(10, 10)
      b = Tensor([1])
      c = add(a, b)
      np.testing.assert_equal(c.numpy(), a.numpy()+1)

if __name__ == '__main__':
  unittest.main()