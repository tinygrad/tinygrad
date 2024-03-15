#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad import dtypes, TinyJit, GlobalCounters, Variable

N = 200  # has to be bigger than the cache to fail

class TestAssign(unittest.TestCase):
  def test_simple_assignment(self):
    a = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    b = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    a.realize()
    b.realize()
    ba1 = a.lazydata.base.realized
    bb1 = b.lazydata.base.realized
    a += b
    a.realize()
    ba2 = a.lazydata.base.realized
    assert ba1 == ba2 and ba1 != bb1
    np.testing.assert_allclose(a.numpy(), (np.arange(N*N)*2).reshape((N,N)))

  def test_assign_zeros_good(self):
    a = Tensor.zeros(10,10).contiguous()
    a.assign(Tensor.ones(10,10))
    b = Tensor.zeros(10,10).contiguous()
    a.realize()
    np.testing.assert_allclose(b.numpy(), 0)

  def test_assign_zeros(self):
    a = Tensor.zeros(10,10).contiguous()
    b = Tensor.zeros(10,10).contiguous()
    #with self.assertRaises(RuntimeError):
    a.assign(Tensor.ones(10,10))
    a.realize()
    np.testing.assert_allclose(b.numpy(), 0)

  def test_assign_add(self):
    def f(x):
      x += 1
      x.realize()
    x = Tensor([0])
    f(x)
    assert x.item() == 1

  def test_assign_add_twice(self):
    # NOTE: this has two kernels
    def f(x):
      x += 1
      x += 1
      x.realize()
    x = Tensor([0])
    f(x)
    assert x.item() == 2

  def test_assign_add_double(self):
    def f(x):
      x += 1
      x.realize()
    x = Tensor([0])
    f(x)
    assert (out:=x.item()) == 1, f"expected 1, got {out}"
    x = Tensor([0])
    f(x)
    assert (out:=x.item()) == 1, f"expected 1, got {out}"

  def test_assign_add_jit(self):
    @TinyJit
    def f(x):
      x += 1
      x.realize()
    x = Tensor([0])
    for _ in range(5): f(x)
    assert x.item() == 5

  def test_assign_add_jit_other(self):
    @TinyJit
    def f(x):
      x += 1
      x.realize()
    x = Tensor([0])
    for _ in range(5): f(x)
    y = Tensor([0])
    for _ in range(4): f(y)
    assert y.item() == 4

  def test_assign_kv_cache(self):
    bsz, max_context = 2, 8

    class Attn:
      @TinyJit
      def __call__(self, xk:Tensor, start_pos:Variable):
        seqlen = xk.shape[1]
        if not hasattr(self, "cache_k"):
          self.cache_k = Tensor.zeros(bsz, max_context, 1, 1).contiguous()
        keys = self.cache_k.shrink((None, (0, start_pos), None, None)).cat(xk, dim=1).contiguous() if start_pos > 0 else xk
        self.cache_k.assign(keys.pad((None,(0,max_context-start_pos-seqlen),None,None)).contiguous()).realize()

    attn = Attn()
    xk = Tensor.ones(bsz, 3, 1, 1).contiguous()
    attn(xk, 0)
    for i in range(3,6):
      # copied from LLaMA
      start_pos = Variable("start_pos", 1, max_context).bind(i)
      xk = Tensor.ones(bsz, 1, 1, 1).contiguous()
      attn(xk, start_pos)

    out = attn.cache_k.flatten().numpy()
    np.testing.assert_allclose(out, [1.,1.,1.,1.,1.,1.,0.,0.,1.,1.,1.,1.,1.,1.,0.,0.])

  def test_assign_contiguous(self):
    b = Tensor.rand(4,4).realize()
    a = (Tensor.rand(4,4).realize() + 1)
    kc = GlobalCounters.kernel_count
    b.assign(a.contiguous()).realize()
    assert GlobalCounters.kernel_count - kc == 2

  def test_assign_contiguous_permute(self):
    b = Tensor.rand(4,4).realize()
    a = (Tensor.rand(4,4).realize() + 1).permute((1,0))
    kc = GlobalCounters.kernel_count
    b.assign(a.contiguous()).realize()
    assert GlobalCounters.kernel_count - kc == 2

  def test_permuted_assignment(self):
    a = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    b = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    a.realize()
    b.realize()
    ba1 = a.lazydata.base.realized
    bb1 = b.lazydata.base.realized
    with self.assertRaises(RuntimeError):
      a = a.permute(1,0)
      a += b
      a.realize()
      ba2 = a.lazydata.base.realized
      assert ba1 != ba2 and ba1 != bb1
      np.testing.assert_allclose(a.numpy(), np.arange(N*N).reshape((N,N)) + np.arange(N*N).reshape((N,N)).transpose(1,0))

  def test_post_permuted_assignment(self):
    a = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    b = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    a.realize()
    b.realize()
    #GlobalCounters.cache = []
    ba1 = a.lazydata.base.realized # noqa: F841
    bb1 = b.lazydata.base.realized # noqa: F841
    with self.assertRaises(RuntimeError):
      a.assign(a.permute(1,0) + b)   # this should not work!
      a.realize()
      ba2 = a.lazydata.base.realized # noqa: F841
      # NOTE: don't test that it's assigned
      #assert ba1 == ba2 and ba1 != bb1
      np.testing.assert_allclose(a.numpy(), np.arange(N*N).reshape((N,N)) + np.arange(N*N).reshape((N,N)).transpose(1,0))

  # TODO: is there a way to sneak in a permute such that it returns the wrong answer?

  @unittest.skip("don't use output buffer, and mismatch dtype no longer supported")
  def test_cast_assignment(self):
    a = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    a.realize()
    oba1 = a.lazydata.base.output_buffer
    a.assign(a.cast(dtypes.int32).realize())
    a.realize()
    oba2 = a.lazydata.base.output_buffer
    assert oba1 is None and oba2 is None
    np.testing.assert_allclose(a.numpy(), np.arange(N*N,dtype=np.int32).reshape((N,N)))

if __name__ == "__main__":
  unittest.main()
