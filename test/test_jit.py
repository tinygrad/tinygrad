#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.tensor import Tensor, Device
from tinygrad.jit import TinyJit, JIT_SUPPORTED_DEVICE
import pytest

pytestmark = pytest.mark.webgpu

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
      np.testing.assert_allclose(c.numpy(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
    assert len(add.jit_cache) == 1

  def test_jit_multiple_outputs(self):
    @TinyJit
    def f(a, b): return (a+b).realize(), (a-b).realize(), (a*b).realize()
    for _ in range(5):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      c, d, e = f(a, b)
      np.testing.assert_allclose(c.numpy(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
      np.testing.assert_allclose(d.numpy(), a.numpy()-b.numpy(), atol=1e-4, rtol=1e-5)
      np.testing.assert_allclose(e.numpy(), a.numpy()*b.numpy(), atol=1e-4, rtol=1e-5)
    assert len(f.jit_cache) == 3

  def test_nothing_jitted(self):
    @TinyJit
    def add(a, b): return a+b
    with self.assertRaises(AssertionError):
      for _ in range(5):
        a = Tensor.randn(10, 10)
        b = Tensor.randn(10, 10)
        c = add(a, b)

  def test_jit_shape_mismatch(self):
    @TinyJit
    def add(a, b): return (a+b).realize()
    for _ in range(5):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      c = add(a, b)
    bad = Tensor.randn(20, 20)
    with self.assertRaises(AssertionError):
      add(a, bad)

  def test_jit_shape_views_mismatch(self):
    @TinyJit
    def add(a): return (a+1).realize()
    with self.assertRaises(AssertionError):
      for i in range(1,5):
        # a has an offset that the kernel doesn't know about
        a = Tensor.randn(10, 10).realize()[:, i:i+2]
        add(a)

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
      np.testing.assert_allclose(c.numpy(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
    assert len(add_kwargs.jit_cache) == 1

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
        np.testing.assert_allclose(np.any(np.not_equal(c.numpy(),a.numpy()+b.numpy())), True, atol=1e-4, rtol=1e-5)
      else:
        np.testing.assert_allclose(c.numpy(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
    assert len(add_array.jit_cache) == 1

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
      np.testing.assert_allclose(c.numpy(), fun.a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
    assert len(fun.__call__.func.__self__.jit_cache) == 1

  def test_jit_size1_input(self):
    @TinyJit
    def f(a, b): return (a+b).realize()
    a = Tensor([1, 2, 3])
    for i in range(5):
      np.testing.assert_allclose(f(a, Tensor([i])).numpy(), (a+i).numpy(), atol=1e-4, rtol=1e-5)
    assert len(f.jit_cache) == 1

  def test_jit_output_non_tensor_fail(self):
    @TinyJit
    def f(a, b, i): return (a+b).realize(), i
    output1, output2 = [], []
    expect1, expect2 = [], []
    for i in range(5):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      o1, o2 = f(a, b, i)
      output1.append(o1.numpy().copy())
      output2.append(o2)
      expect1.append(a.numpy().copy()+b.numpy().copy())
      expect2.append(i)
    np.testing.assert_allclose(output1, expect1, atol=1e-4, rtol=1e-5)
    # the jit only works with Tensor outputs
    assert output2 != expect2
    assert len(f.jit_cache) == 1

  @unittest.skip("random isn't working in JIT")
  def test_jit_random_regen(self):
    def f(a, b):
      rn = Tensor.randn(*a.shape)
      return ((a+b)*rn).realize()
    a = Tensor.randn(10, 10)
    b = Tensor.randn(10, 10)

    Tensor._seed = 1234
    jf = TinyJit(f)
    res = set()
    for _ in range(5):
      o1 = jf(a, b)
      res.add(o1.numpy()[0][0])
    assert len(res) == 5, "All values should be different, rand works in jit."

    Tensor._seed = 1234
    jf2 = TinyJit(f)
    res2 = set()
    for _ in range(5):
      o1 = jf2(a, b)
      res2.add(o1.numpy()[0][0])
    assert len(res2) == 5, "All values should be different, rand works in jit."
    assert res == res2, "Jit rand is not reproducible with the same seed"

    Tensor._seed = 3421
    jf3 = TinyJit(f)
    res3 = set()
    for _ in range(5):
      o1 = jf3(a, b)
      res3.add(o1.numpy()[0][0])
    assert len(res3) == 5, "All values should be different, rand works in jit."
    assert res3 != res2, "Jit rand is diff with diff seeds"

  def test_jit_realization_and_sampling(self):
    w = Tensor.eye(5)

    @TinyJit
    def foo (x): return w.dot(x).realize()

    arg  = [
        Tensor([1,2,3,4,5]),
        Tensor([1,3,3,4,6]),
        Tensor([1,2,5,4,7]),
        Tensor([0,2,3,1,0]),
    ]

    Y = [foo(e).numpy() for e in arg]

    foo(Tensor([7,7,7,7,7]))
    want = [[1., 2., 3., 4., 5.],
            [1., 3., 3., 4., 6.],
            [1., 2., 5., 4., 7.],
            [0., 2., 3., 1., 0.]]
    np.testing.assert_allclose(want, Y)

if __name__ == '__main__':
  unittest.main()