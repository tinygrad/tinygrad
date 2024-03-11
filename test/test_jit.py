#!/usr/bin/env python
import unittest, functools
import numpy as np

from test.helpers import assert_jit_cache_len
from tinygrad.tensor import Tensor
from tinygrad.features.jit import TinyJit
from tinygrad.device import Device
from tinygrad.helpers import CI

def _simple_test(add, extract=lambda x: x):
  for _ in range(5):
    a = Tensor.randn(10, 10)
    b = Tensor.randn(10, 10)
    c = add(a, b)
    np.testing.assert_allclose(extract(c).numpy(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
  assert_jit_cache_len(add, 1)

class TestJit(unittest.TestCase):
  def test_simple_jit(self):
    @TinyJit
    def add(a, b): return (a+b).realize()
    _simple_test(add)

  def test_simple_jit_norealize(self):
    @TinyJit
    def add(a, b): return (a+b)
    _simple_test(add)

  def test_simple_jit_norealize_list(self):
    @TinyJit
    def add(a, b): return [a+b]
    _simple_test(add, extract=lambda x: x[0])

  def test_simple_jit_norealize_dict(self):
    @TinyJit
    def add(a, b): return {"billy": a+b}
    _simple_test(add, extract=lambda x: x["billy"])

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
    assert_jit_cache_len(f, 3)

  def test_nothing_jitted(self):
    @TinyJit
    def add(a, b): return None
    with self.assertRaises(AssertionError):
      for _ in range(5):
        a = Tensor.randn(10, 10)
        b = Tensor.randn(10, 10)
        add(a, b)

  def test_jit_shape_mismatch(self):
    @TinyJit
    def add(a, b): return (a+b).realize()
    for _ in range(5):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      add(a, b)
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
    assert_jit_cache_len(add_kwargs, 1)

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
    assert_jit_cache_len(add_array, 1)

  def test_jit_copyin(self):
    @TinyJit
    def f(a):
      return a + Tensor([1,2,3])
    for _ in range(5):
      b = Tensor.randn(3)
      c = f(b)
      np.testing.assert_allclose(c.numpy(), b.numpy()+[1,2,3], atol=1e-4, rtol=1e-5)

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
    assert_jit_cache_len(fun.__call__.func.__self__, 1)

  def test_jit_size1_input(self):
    @TinyJit
    def f(a, b): return (a+b).realize()
    a = Tensor([1, 2, 3])
    for i in range(5):
      np.testing.assert_allclose(f(a, Tensor([i])).numpy(), (a+i).numpy(), atol=1e-4, rtol=1e-5)
    assert_jit_cache_len(f, 1)

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
    assert_jit_cache_len(f, 1)

  def test_jit_random_regen(self):
    def f(a, b):
      rn = Tensor.randn(*a.shape)
      return ((a+b)*rn).realize()
    a = Tensor.randn(10, 10).realize()  # realize these before resetting the random seed
    b = Tensor.randn(10, 10).realize()

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

  def test_jitted_read_assign(self):
    class Cache:
      def __init__(self):
        self.good_cache = Tensor.zeros(1)
        self.bad_cache = Tensor.zeros(1)
        self.good_jitted = TinyJit(self.good)
        self.bad_jitted = TinyJit(self.bad)

      def good(self, y, cache_v=None):
        if cache_v is not None:
          self.good_cache.assign(cache_v+1-1).realize()
        return (self.good_cache + y).realize()  # need + y to provide inputs to JIT

      def bad(self, y, cache_v=None):
        if cache_v is not None:
          self.bad_cache.assign(cache_v).realize()
        return (self.bad_cache + y).realize()

    cache = Cache()
    np.testing.assert_equal([0], cache.good_cache.numpy())
    np.testing.assert_equal([0], cache.bad_cache.numpy())

    zero = Tensor([0])
    one = Tensor([1])
    two = Tensor([2])

    # save [1] in the caches
    cache.good(zero, one)
    cache.bad(zero, one)
    np.testing.assert_equal([1], cache.good_cache.numpy())
    np.testing.assert_equal([1], cache.bad_cache.numpy())

    for i in range(5):
      x = Tensor([i]) # NOTE: if this doesn't change, it just hits the lazybuffer cache
      cache.good_jitted(x)
      cache.bad_jitted(x)

    # verify the jitted calls read 1 from the cache
    np.testing.assert_equal([1], cache.good_jitted(zero).numpy())
    np.testing.assert_equal([1], cache.bad_jitted(zero).numpy())

    # save [2] in the caches
    cache.good(zero, two)
    cache.bad(zero, two)
    np.testing.assert_equal([2], cache.good_cache.numpy())
    np.testing.assert_equal([2], cache.bad_cache.numpy())

    # verify the jitted calls read 2 from the cache
    np.testing.assert_equal([2], cache.good_jitted(zero).numpy())
    # but the bad_jitted doesn't!
    np.testing.assert_equal([1], cache.bad_jitted(zero).numpy())

    assert_jit_cache_len(cache.good_jitted, 1)
    assert_jit_cache_len(cache.bad_jitted, 1)

  def test_jit_buffer_behavior(self):
    @TinyJit
    def foo(x) -> Tensor: return x.sum().realize()

    result_1 = foo(Tensor([1] * 2))
    result_2 = foo(Tensor([2] * 2))
    result_3 = foo(Tensor([3] * 2))

    # expect the buffer to share underlying buffer
    np.testing.assert_allclose(result_1.numpy(), [2], atol=1e-4, rtol=1e-5)
    np.testing.assert_allclose(result_2.numpy(), [6], atol=1e-4, rtol=1e-5)
    np.testing.assert_allclose(result_3.numpy(), [6], atol=1e-4, rtol=1e-5)

  @unittest.skipIf(CI and Device.DEFAULT=="METAL", "no ICB in CI, creation of graph fails")
  def test_jit_batch_split(self):
    if Device[Device.DEFAULT].graph is None: raise unittest.SkipTest("only test graphs")

    # Create long jit with 83 kernels.
    def f(a, b, c, d, e):
      for _ in range(80):
        a = (a+b).realize()
      y = (a*c).realize()
      z = (y*d).realize()
      w = (z*e)
      return w.realize()

    a = Tensor.randn(10, 10).realize()
    b = Tensor.randn(10, 10).realize()
    c = Tensor.randn(10, 10).realize()
    d = Tensor.randn(10, 10).realize()
    e = Tensor.randn(10, 10).realize()

    jf = TinyJit(f)
    prev = None
    for _ in range(5):
      o = jf(a, b, c, d, e).numpy()
      if prev is not None: np.testing.assert_allclose(o, prev, atol=1e-4, rtol=1e-5)
      prev = o

    graph_t = Device[Device.DEFAULT].graph.func if isinstance(Device[Device.DEFAULT].graph, functools.partial) else Device[Device.DEFAULT].graph
    # Checking that 2 graphs are inited.
    assert isinstance(jf.jit_cache[0].prg, graph_t)
    assert isinstance(jf.jit_cache[1].prg, graph_t)

  def test_jit_const_inputs(self):
    @TinyJit
    def f(x,y): return (x+y).realize()
    for _ in range(5):
      np.testing.assert_equal(f(Tensor.ones(3), Tensor.zeros(3)).numpy(), np.ones(3))

    @TinyJit
    def g(x,y,z): return (x+y+z).realize()
    for i in range(5):
      np.testing.assert_equal(g(Tensor([i]*3), Tensor.ones(3), Tensor.zeros(3)).numpy(), np.array([i+1]*3))


if __name__ == '__main__':
  unittest.main()
