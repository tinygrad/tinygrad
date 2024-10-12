#!/usr/bin/env python
import unittest, functools
import numpy as np

from hypothesis import given, settings, strategies as strat
from test.helpers import assert_jit_cache_len
from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.device import Device
from tinygrad.helpers import CI, Context
from tinygrad.dtype import dtypes
from extra.models.unet import ResBlock

def _simple_test(add, extract=lambda x: x, N=10):
  for _ in range(5):
    a = Tensor.randn(N, N)
    b = Tensor.randn(N, N)
    c = add(a, b)
    np.testing.assert_allclose(extract(c).numpy(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
  assert_jit_cache_len(add, 1)

class TestJit(unittest.TestCase):

  @settings(deadline=2e4)
  @unittest.skipUnless(Device.DEFAULT in ["LLVM", "CLANG"], f"no support on {Device.DEFAULT}")
  @given(strat.sampled_from([Tensor.exp2, Tensor.log2, Tensor.sin]))
  def test_approx_jit_timeout(self, op):
    with Context(TRANSCENDENTAL=2):
      model = [ResBlock(16, 24, 16) for _ in range(4)]
      @TinyJit
      def fw_approx(t, t2):
        for l in model: t = l(t, t2)
        return op(t).realize()
      fw_approx(Tensor.empty(4, 16, 8, 8), Tensor.empty(1, 24))

  def test_simple_jit(self):
    @TinyJit
    def add(a, b): return (a+b).realize()
    _simple_test(add)

  def test_simple_jit_reset(self):
    @TinyJit
    def add(a, b): return (a+b).realize()
    _simple_test(add)
    add.reset()
    _simple_test(add, N=20)

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

  def test_jit_zero_does_not_jit(self):
    @TinyJit
    def add(a, b): return (a+b).realize()
    with Context(JIT=0):
      for i in range(5):
        a = Tensor([i])
        b = Tensor([i])
        c = add(a, b)
        np.testing.assert_allclose(c.numpy(), 2*i)
      assert_jit_cache_len(add, 0)

  def test_jit_not_capturing(self):
    @TinyJit
    def add(a, b):
      Tensor.zeros(4, 4).contiguous().realize()  # no-op kernel is captured
      return (a+b).realize()
    for i in range(5):
      a = Tensor([i])
      b = Tensor([i])
      c = add(a, b)
      np.testing.assert_allclose(c.numpy(), 2*i)
    assert_jit_cache_len(add, 2)

    @TinyJit
    def add2(a, b):
      with Context(CAPTURING=0):  # not captured
        Tensor.zeros(4, 4).contiguous().realize()
      return (a+b).realize()
    for i in range(5):
      a = Tensor([i])
      b = Tensor([i])
      c = add2(a, b)
      np.testing.assert_allclose(c.numpy(), 2*i)
    assert_jit_cache_len(add2, 1)

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

  def test_reorder_kwargs_jit(self):
    @TinyJit
    def add_kwargs(first, second): return (first/second).realize()
    for _ in range(2):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      c = add_kwargs(second=b, first=a)
      np.testing.assert_allclose(c.numpy(), a.numpy()/b.numpy(), atol=1e-4, rtol=1e-5)
    for _ in range(2):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      c = add_kwargs(first=a, second=b)
      np.testing.assert_allclose(c.numpy(), a.numpy()/b.numpy(), atol=1e-4, rtol=1e-5)
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

    Tensor.manual_seed(1234)
    jf = TinyJit(f)
    res = set()
    for _ in range(5):
      o1 = jf(a, b)
      res.add(o1.numpy()[0][0])
    assert len(res) == 5, "All values should be different, rand works in jit."

    Tensor.manual_seed(1234)
    jf2 = TinyJit(f)
    res2 = set()
    for _ in range(5):
      o1 = jf2(a, b)
      res2.add(o1.numpy()[0][0])
    assert len(res2) == 5, "All values should be different, rand works in jit."
    assert res == res2, "Jit rand is not reproducible with the same seed"

    Tensor.manual_seed(3421)
    jf3 = TinyJit(f)
    res3 = set()
    for _ in range(5):
      o1 = jf3(a, b)
      res3.add(o1.numpy()[0][0])
    assert len(res3) == 5, "All values should be different, rand works in jit."
    assert res3 != res2, "Jit rand is diff with diff seeds"

  def test_jit_multiple_random_regen(self):
    def f(a, b):
      rn = Tensor.randn(*a.shape)
      rn = rn * a
      rn2 = Tensor.randn(*a.shape)
      rn2 = rn2 * b
      rn = rn + rn2
      rn2 = rn2 + Tensor.randn(*a.shape)
      return ((a+b)*rn).realize(), ((a+b)*rn2).realize()
    a = Tensor.randn(10, 10).realize()  # realize these before resetting the random seed
    b = Tensor.randn(10, 10).realize()

    Tensor.manual_seed(1234)
    jf = TinyJit(f)
    res = set()
    for _ in range(5):
      o1, o2 = jf(a, b)
      res.add(o1.numpy()[0][0])
      res.add(o2.numpy()[0][0])
    assert len(res) == 10, "All values should be different, rand works in jit."

    Tensor.manual_seed(1234)
    jf2 = TinyJit(f)
    res2 = set()
    for _ in range(5):
      o1, o2 = jf2(a, b)
      res2.add(o1.numpy()[0][0])
      res2.add(o2.numpy()[0][0])
    assert len(res2) == 10, "All values should be different, rand works in jit."
    assert res == res2, "Jit rand is not reproducible with the same seed"

    Tensor.manual_seed(3421)
    jf3 = TinyJit(f)
    res3 = set()
    for _ in range(5):
      o1, o2 = jf3(a, b)
      res3.add(o1.numpy()[0][0])
      res3.add(o2.numpy()[0][0])
    assert len(res3) == 10, "All values should be different, rand works in jit."
    assert res3 != res2, "Jit rand is diff with diff seeds"

  def test_jit_random_after_unrealized_random(self):
    @TinyJit
    def f(): return Tensor.rand()
    Tensor.manual_seed(1234)
    Tensor.rand()
    res = [f().numpy() for _ in range(3)]
    assert res[1] != res[2]

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
    def g(x,y,z): return (x+y+z).realize()
    for i in range(5):
      np.testing.assert_equal(g(Tensor([i]*3), Tensor.ones(3), Tensor.zeros(3)).numpy(), np.array([i+1]*3))

  @unittest.skipIf(CI and Device.DEFAULT in {"GPU", "CUDA", "METAL", "NV", "AMD"}, "no GPU CI")
  def test_jitted_transfers(self):
    d0, d1 = f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1"

    def f(a, b):
      x = a.to(d1)
      y = b.to(d1)
      return x.realize(), y.realize()

    jf = TinyJit(f)
    for _ in range(5):
      a = Tensor.randn(10, 10, device=d0).realize()
      b = Tensor.randn(10, 10, device=d0).realize()
      xc, yc = jf(a, b)
      np.testing.assert_allclose(a.numpy(), xc.numpy(), atol=1e-4, rtol=1e-5)
      np.testing.assert_allclose(b.numpy(), yc.numpy(), atol=1e-4, rtol=1e-5)

  @unittest.skipIf(CI and Device.DEFAULT in {"GPU", "CUDA", "METAL"}, "no GPU/CUDA/METAL in CI, fine to run on AMD/NV")
  def test_jitted_view(self):
    d0, d1 = f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1"

    def f(a):
      x1 = a.sum(axis=(1,))
      x = (x1 + 5).bitcast(dtypes.int32)
      y = x.to(d1)
      return y.realize()

    jf = TinyJit(f)
    for _ in range(5):
      a = Tensor.randn(10, 1000, device=d0).realize()
      xc = jf(a)
      np.testing.assert_allclose((a.numpy().sum(axis=(1,)) + 5).view(np.int32), xc.numpy(), atol=1e-4, rtol=1e-5)

@unittest.skip("Pending multioutput implementation #3607")
class TestMultioutputJit(unittest.TestCase):
  def _test(self, f):
    for _ in range(5):
      a, b = Tensor.randn(10, 10), Tensor.randn(10, 10)
      out0, out1, out2 = f(a, b)
      np.testing.assert_allclose(out0.numpy(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
      np.testing.assert_allclose(out1.numpy(), a.numpy()-b.numpy(), atol=1e-4, rtol=1e-5)
      np.testing.assert_allclose(out2.numpy(), a.numpy()*b.numpy(), atol=1e-4, rtol=1e-5)

  def test_jit_multioutput_realize(self):
    @TinyJit
    def fxn(a, b): return (a+b).realize(), (a-b).realize(), (a*b).realize()
    self._test(fxn)
    assert_jit_cache_len(fxn, 3)

  def test_jit_multioutput_norealize(self):
    @TinyJit
    def fxn(a, b): return a+b, a-b, a*b
    self._test(fxn)
    assert_jit_cache_len(fxn, 1)

  def test_jit_multioutput_mix(self):
    @TinyJit
    def fxn(a, b): return a+b, a-b, (a*b).realize()
    self._test(fxn)
    assert_jit_cache_len(fxn, 2)

class TestJitInsideJit(unittest.TestCase):
  def test_jit_jit_error(self):
    @TinyJit
    def f(t): return t + 1

    @TinyJit
    def g(t): return f(t) * 3

    # NOTE: first does not raise
    g(Tensor([1])).realize()
    with self.assertRaisesRegex(RuntimeError, "having TinyJit inside another TinyJit is not supported"):
      g(Tensor([1])).realize()

if __name__ == '__main__':
  unittest.main()
