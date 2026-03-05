import numpy as np
import unittest
from tinygrad import Tensor
from tinygrad.jit import jit

class TestJit2(unittest.TestCase):
  def test_simple(self):
    @jit
    def f(a:Tensor, b:Tensor) -> Tensor: return a+b
    a, b = Tensor([1,2,3]), Tensor([4,5,6])
    np.testing.assert_equal(f(a,b).numpy(), [5,7,9])
    np.testing.assert_equal(f(a,b).numpy(), [5,7,9])
    np.testing.assert_equal(f(a,b).numpy(), [5,7,9])

  def test_captures_after_two(self):
    @jit
    def f(a:Tensor) -> Tensor: return (a+1).contiguous()
    a = Tensor([1,2,3]).contiguous()
    assert f._captured is None
    f(a)
    assert f._captured is None
    f(a)
    assert f._captured is not None
    # third call uses replay
    np.testing.assert_equal(f(a).numpy(), [2,3,4])

  def test_no_capture_on_different(self):
    cnt = [0]
    @jit
    def f(a:Tensor) -> Tensor:
      cnt[0] += 1
      return (a + cnt[0]).contiguous()
    a = Tensor([1,2,3]).contiguous()
    # each call builds a different graph (different constant)
    np.testing.assert_equal(f(a).numpy(), [2,3,4])
    np.testing.assert_equal(f(a).numpy(), [3,4,5])
    np.testing.assert_equal(f(a).numpy(), [4,5,6])
    assert f._captured is None

  def test_recursive(self):
    @jit
    def inner(a:Tensor) -> Tensor: return (a+1).contiguous()
    @jit
    def outer(a:Tensor) -> Tensor: return inner(inner(a))
    a = Tensor([1,2,3]).contiguous()
    np.testing.assert_equal(outer(a).numpy(), [3,4,5])
    np.testing.assert_equal(outer(a).numpy(), [3,4,5])
    np.testing.assert_equal(outer(a).numpy(), [3,4,5])

  def test_nested_inner_captured_first(self):
    @jit
    def inner(a:Tensor) -> Tensor: return (a+1).contiguous()
    @jit
    def outer(a:Tensor) -> Tensor: return inner(inner(a))
    a = Tensor([1,2,3]).contiguous()
    # capture inner first
    inner(a); inner(a)
    assert inner._captured is not None
    # now outer calls captured inner
    np.testing.assert_equal(outer(a).numpy(), [3,4,5])
    np.testing.assert_equal(outer(a).numpy(), [3,4,5])
    np.testing.assert_equal(outer(a).numpy(), [3,4,5])

  def test_nested_both_captured(self):
    @jit
    def inner(a:Tensor) -> Tensor: return (a+1).contiguous()
    @jit
    def outer(a:Tensor) -> Tensor: return inner(inner(a))
    a = Tensor([1,2,3]).contiguous()
    # capture inner
    inner(a); inner(a)
    assert inner._captured is not None
    # capture outer (inner is already captured, outer sees it as @function since ALLOW_DEVICE_USAGE=0)
    outer(a); outer(a)
    assert outer._captured is not None
    # replay both
    np.testing.assert_equal(outer(a).numpy(), [3,4,5])
    np.testing.assert_equal(outer(a).numpy(), [3,4,5])

  def test_implicit(self):
    w = Tensor([10,20,30])
    @jit
    def f(a:Tensor) -> Tensor: return a+w
    a = Tensor([1,2,3])
    np.testing.assert_equal(f(a).numpy(), [11,22,33])
    np.testing.assert_equal(f(a).numpy(), [11,22,33])
    np.testing.assert_equal(f(a).numpy(), [11,22,33])

  def test_kwargs(self):
    @jit
    def f(a:Tensor, b:Tensor) -> Tensor: return a+b
    a, b = Tensor([1,2,3]), Tensor([4,5,6])
    np.testing.assert_equal(f(a=a, b=b).numpy(), [5,7,9])
    np.testing.assert_equal(f(a=a, b=b).numpy(), [5,7,9])
    np.testing.assert_equal(f(a=a, b=b).numpy(), [5,7,9])

  def test_method(self):
    class Model:
      def __init__(self): self.w = Tensor([10,20,30])
      @jit
      def forward(self, x:Tensor) -> Tensor: return x + self.w
    m = Model()
    np.testing.assert_equal(m.forward(Tensor([1,2,3])).numpy(), [11,22,33])
    np.testing.assert_equal(m.forward(Tensor([1,2,3])).numpy(), [11,22,33])
    np.testing.assert_equal(m.forward(Tensor([1,2,3])).numpy(), [11,22,33])

  def test_reset(self):
    @jit
    def f(a:Tensor) -> Tensor: return (a+1).contiguous()
    a = Tensor([1,2,3]).contiguous()
    f(a); f(a)
    assert f._captured is not None
    f.reset()
    assert f._captured is None
    assert f._cnt == 0

  def test_dtype_mismatch_after_capture(self):
    @jit
    def f(a:Tensor) -> Tensor: return (a+1).contiguous()
    from tinygrad.dtype import dtypes
    a_int = Tensor([1,2,3]).contiguous()
    a_float = Tensor([1.0,2.0,3.0]).contiguous()
    f(a_int); f(a_int)
    assert f._captured is not None
    from tinygrad.engine.jit import JitError
    with self.assertRaises(JitError): f(a_float)

  def test_memory_planned(self):
    @jit
    def f(x:Tensor) -> Tensor: return (x + 1).contiguous()
    a = Tensor.ones(1024).contiguous()
    # run through function path (builds CALL, memory planned in schedule)
    f(a); f(a)
    assert f._captured is not None
    # third call: replay
    out = f(a)
    np.testing.assert_equal(out.numpy(), np.full(1024, 2.0))

  def test_implicit_assign(self):
    a = Tensor([1,2,3])
    a += 1
    c = Tensor([2,2,2]).contiguous()
    @jit
    def f(b:Tensor) -> Tensor: return a+b+c
    b = Tensor([10,20,30])
    np.testing.assert_equal(f(b).numpy(), [14,25,36])
    np.testing.assert_equal(f(b).numpy(), [14,25,36])
    np.testing.assert_equal(f(b).numpy(), [14,25,36])

  def test_assign_input(self):
    @jit
    def f(a:Tensor, b:Tensor) -> Tensor:
      a.assign(b+1)
      return a
    a = Tensor([1,2,3]).realize()
    b = Tensor([10,20,30]).realize()
    np.testing.assert_equal(f(a,b).numpy(), [11,21,31])
    np.testing.assert_equal(f(a,b).numpy(), [11,21,31])

  def test_multiple_calls_different_values(self):
    @jit
    def f(a:Tensor, b:Tensor) -> Tensor: return a+b
    # same shapes, different values — should still capture
    np.testing.assert_equal(f(Tensor([1,2,3]), Tensor([4,5,6])).numpy(), [5,7,9])
    np.testing.assert_equal(f(Tensor([10,20,30]), Tensor([1,1,1])).numpy(), [11,21,31])
    assert f._captured is not None
    np.testing.assert_equal(f(Tensor([100,200,300]), Tensor([1,2,3])).numpy(), [101,202,303])

  def test_nested_functions_memory_planned(self):
    @jit
    def f1(x:Tensor) -> Tensor: return (x + 1).contiguous()
    @jit
    def f2(x:Tensor) -> Tensor: return (x + 2).contiguous()
    @jit
    def f3(x:Tensor) -> Tensor: return (x + 3).contiguous()
    @jit
    def f4(x:Tensor) -> Tensor: return (x + 4).contiguous()
    @jit
    def pipeline(x:Tensor) -> Tensor: return f4(f3(f2(f1(x))))
    a = Tensor.ones(1024).contiguous()
    pipeline(a); pipeline(a)
    assert pipeline._captured is not None
    # all internal buffers (between f1-f2, f2-f3, f3-f4) should share one arena
    internal_bufs = [b for ei in pipeline.jit_cache for b in ei.bufs if b is not None and b._base is not None]
    self.assertGreater(len(internal_bufs), 0)
    bases = set(b._base for b in internal_bufs)
    self.assertEqual(len(bases), 1)
    # replay still produces correct results
    np.testing.assert_equal(pipeline(a).numpy(), np.full(1024, 11.0))

if __name__ == "__main__":
  unittest.main()
