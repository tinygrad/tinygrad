#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad import dtypes, Tensor, TinyJit, GlobalCounters, Variable
from tinygrad.device import is_dtype_supported
from tinygrad.helpers import temp, CI, CPU_LVP

N = 200  # has to be bigger than the cache to fail

class TestAssign(unittest.TestCase):
  def test_simple_assignment(self):
    a = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    b = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    a.realize()
    b.realize()
    ba1 = a.uop.base.realized
    bb1 = b.uop.base.realized
    a += b
    a.realize()
    ba2 = a.uop.base.realized
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
    out = x.item()
    assert out == 1, f"expected 1, got {out}"
    x = Tensor([0])
    f(x)
    out = x.item()
    assert out == 1, f"expected 1, got {out}"

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
    assert x.item() == 5

    y = Tensor([0])
    for _ in range(4): f(y)
    assert y.item() == 4

  def test_assign_other_jit(self):
    @TinyJit
    def f(x, a):
      x.assign(a)
      x.realize()
    x = Tensor([0])
    for i in range(1, 6):
      f(x, x.full_like(i).contiguous())  # const would be implicitly folded without contiguous
      assert x.item() == i

  def test_assign_add_other_jit(self):
    @TinyJit
    def f(x, a):
      x += a
      x.realize()
    x = Tensor([0])
    a = 0
    for i in range(1, 6):
      a += i
      f(x, x.full_like(i).contiguous())
      assert x.item() == a

  def test_assign_changes(self):
    a = Tensor.ones(4).contiguous().realize()
    old_a = a
    a.assign(Tensor.full((4,), 2.).contiguous())
    # NOTE: old_a is now 2, and this would match the behavior of pytorch
    new = a + old_a
    np.testing.assert_allclose(new.numpy(), 4)

  def test_assign_changes_alt(self, realize=False):
    a = Tensor(1).contiguous()
    if realize: a.realize()
    b = a.contiguous()    # b returns a new Tensor
    b.assign(2)
    b.realize()
    self.assertNotEqual(a.item(), b.item())
  # on a realized Tensor contiguous child changes the source
  @unittest.expectedFailure
  def test_assign_changes_realized_alt(self): return self.test_assign_changes_alt(realize=True)

  @unittest.skip("assign to contiguous shouldn't change the base buffer")
  def test_assign_changes_buffer_alt(self):
    a, b = [Tensor(Tensor(0).contiguous().realize().uop.as_buf()) for _ in range(2)]
    Tensor.realize(a.contiguous().assign(1), b.contiguous().assign(2))
    self.assertEqual((a + b).item(), 3)

  def test_assign_diamond_cycle(self):
    # NOTE: should *not* raise AssertionError from numpy
    with self.assertRaisesRegex(RuntimeError, "cycle"):
      a = Tensor.ones(4).contiguous().realize()
      times_a = a*3
      a.assign(Tensor.full((4,), 2.).contiguous())
      new = a + (times_a-1)
      np.testing.assert_allclose(new.numpy(), 4)

  def test_assign_diamond_contiguous_cycle(self):
    with self.assertRaisesRegex(RuntimeError, "cycle"):
      a = Tensor.ones(4).contiguous().realize()
      times_a = a*3
      a.assign(Tensor.full((4,), 2.))
      new = a.contiguous() + times_a-1
      np.testing.assert_allclose(new.numpy(), 4)

  def test_assign_diamond_possible(self):
    a = Tensor.ones(4).contiguous().realize()
    times_a = a*3
    a.assign(Tensor.full((4,), 2.))
    new = a + (times_a-1).contiguous()
    np.testing.assert_allclose(new.numpy(), 4)

  def test_assign_diamond_possible_contiguous(self):
    a = Tensor.ones(4).contiguous().realize()
    times_a = a*3
    a.assign(Tensor.full((4,), 2.).contiguous())
    new = a + (times_a-1).contiguous()
    np.testing.assert_allclose(new.numpy(), 4)

  def test_assign_diamond_both_contiguous(self):
    a = Tensor.ones(4).contiguous().realize()
    times_a = a*3
    a.assign(Tensor.full((4,), 2.))
    new = a.contiguous() + (times_a-1).contiguous()
    np.testing.assert_allclose(new.numpy(), 4)

  def test_assign_diamond_alt(self):
    a = Tensor.ones(4).contiguous().realize()
    a.assign(Tensor.full((4,), 2.).contiguous())
    times_a = a*3
    new = a + times_a
    np.testing.assert_allclose(new.numpy(), 8)

  @unittest.skipIf(CI and CPU_LVP, "flaky in CI")
  def test_double_assign(self):
    a = Tensor.ones(4).contiguous().realize()
    a += 1
    a += 1
    np.testing.assert_allclose(a.numpy(), 3)

  def test_crossover_assign(self):
    a = Tensor.full((4,), 2).contiguous().realize()
    b = Tensor.full((4,), 3).contiguous().realize()
    a += b
    b += a
    Tensor.realize(a,b)
    np.testing.assert_allclose(a.numpy(), 5)
    np.testing.assert_allclose(b.numpy(), 8)

  def test_assign_double_diamond(self):
    a = Tensor.full((4,), 2).contiguous().realize()
    b = Tensor.full((4,), 3).contiguous().realize()
    a_prev = a*4
    b_prev = b+3
    b += a_prev.contiguous()
    a += b_prev.contiguous()
    Tensor.realize(a, b)
    np.testing.assert_equal(b.numpy(), 11)
    np.testing.assert_equal(a.numpy(), 8)

  def test_assign_double_diamond_reduce(self):
    a0 = Tensor.full((16, 16), 10).contiguous().realize()
    a1 = Tensor.full((16, 16), 20).contiguous().realize()
    b0 = Tensor.full((16, ), 1).contiguous().realize()
    b1 = Tensor.full((16, ), 2).contiguous().realize()

    r0 = (a0 - b1.contiguous()).sum(1)
    r1 = (a1 - b0.contiguous()).sum(1)
    b0.assign(r0 * b0)
    b1.assign(r1 * b1)
    Tensor.realize(b0, b1)
    np.testing.assert_equal(b0.numpy(), 128)
    np.testing.assert_equal(b1.numpy(), 608)

  @unittest.skip("TODO: bring this assert back")
  def test_crossunder_assign(self):
    # NOTE: should *not* raise AssertionError from numpy
    with self.assertRaisesRegex(RuntimeError, "cycle"):
      a = Tensor.full((4,), 2).contiguous().realize()
      b = Tensor.full((4,), 3).contiguous().realize()
      c = a+9
      a += b
      b += c
      Tensor.realize(a,b)
      np.testing.assert_allclose(a.numpy(), 2+3)
      np.testing.assert_allclose(b.numpy(), 3+2+9)

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
    ba1 = a.uop.base.realized
    bb1 = b.uop.base.realized
    a = a.permute(1,0)
    a += b
    a.realize()
    ba2 = a.uop.base.realized
    np.testing.assert_allclose(a.numpy(), np.arange(N*N).reshape((N,N)) + np.arange(N*N).reshape((N,N)).transpose(1,0))
    # permute and base are the same buffer
    assert ba1 == ba2 and ba1 != bb1

  def test_post_permuted_assignment(self):
    a = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    b = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    a.realize()
    b.realize()
    #GlobalCounters.cache = []
    ba1 = a.uop.base.realized # noqa: F841
    bb1 = b.uop.base.realized # noqa: F841
    a.assign(a.permute(1,0) + b)   # this should not work!
    a.realize()
    ba2 = a.uop.base.realized # noqa: F841
    # NOTE: don't test that it's assigned
    #assert ba1 == ba2 and ba1 != bb1
    np.testing.assert_allclose(a.numpy(), np.arange(N*N).reshape((N,N)) + np.arange(N*N).reshape((N,N)).transpose(1,0))

  def test_post_permuted_assignment_alt(self):
    a = Tensor.arange(N*N).reshape(N,N).contiguous().realize()
    b = Tensor.arange(N*N).reshape(N,N).contiguous().realize()
    new_a = (a.T+b).numpy()
    a.assign(a.T+b)
    np.testing.assert_allclose(a.numpy(), new_a)

  def test_post_reshape_assignment_fine(self):
    a = Tensor.arange(N*N).reshape(N, N).contiguous().realize()
    b = Tensor.arange(N*N).reshape(N, N).contiguous().realize()
    rhs = a.reshape(-1).reshape(N, N)
    new_a = (rhs+b).numpy()
    a.assign(rhs+b)  # self-assign with reshape view is fine
    np.testing.assert_allclose(a.numpy(), new_a)

  @unittest.skip("multi output not supported anymore")
  def test_simple_assignment_multioutput(self):
    a = Tensor.randn(32, 32).realize()
    b = Tensor.full((32, ), 1.).contiguous().realize()
    c = Tensor.full((32, ), 2.).contiguous().realize()
    d = Tensor.full((32, ), 3.).contiguous().realize()

    r = a.sum(axis=1)
    b.assign(r + b)
    c.assign(r + c)
    d.assign(r + d)

    kc = GlobalCounters.kernel_count
    Tensor.realize(b, c, d)
    assert GlobalCounters.kernel_count - kc == 1
    np.testing.assert_allclose(b.numpy(), a.sum(1).numpy()+1)
    np.testing.assert_allclose(c.numpy(), a.sum(1).numpy()+2)
    np.testing.assert_allclose(d.numpy(), a.sum(1).numpy()+3)

  # NOTE: if the assign target is read/write in a single kernel, it should be contiguous

  def test_permuted_assignment_correct(self):
    a = Tensor.arange(4 * 4).reshape(4, 4).contiguous().realize()
    b = Tensor.arange(4 * 4).reshape(4, 4).contiguous().realize()
    a = a.permute(1, 0)
    new_val = a + b
    a.assign(new_val)
    np.testing.assert_equal(a.numpy(), np.arange(4 * 4).reshape(4, 4).transpose(1, 0) + np.arange(4 * 4).reshape(4, 4))

  def test_permuted_reduceop_child_dual_use(self):
    a = Tensor.randn(32, 32, 32).realize()
    b = Tensor.full((32, 32), 1.).contiguous().realize()
    r = a.sum(axis=1)
    b.assign(r + b.permute(1, 0))
    b.realize()
    np.testing.assert_allclose(b.numpy(), a.numpy().sum(axis=1)+np.ones((32, 32)).transpose(1, 0), atol=1e-6, rtol=1e-3)

  @unittest.skip("multi output not supported anymore")
  def test_permuted_reduceop_multioutput_dual_use(self):
    a = Tensor.randn(32, 32, 32).realize()
    b = Tensor.full((32, 32), 1.).contiguous().realize()
    c = Tensor.full((32, 32), 2.).contiguous().realize()

    with self.assertRaisesRegex(RuntimeError, "contiguous"):
      r = a.sum(axis=1)
      b_perm = b.permute(1, 0)
      b.assign(r + b)
      c.assign(r + b_perm)
      Tensor.realize(b, c)

  @unittest.skip("multi output not supported anymore")
  def test_permuted_reduceop_multioutput_dual_use_possible(self):
    a = Tensor.randn(32, 32, 32, dtype=dtypes.int).realize()
    b = Tensor.arange(32 * 32).reshape(32, 32).realize()
    c = Tensor.arange(32 * 32).reshape(32, 32).realize()

    kc = GlobalCounters.kernel_count
    r = a.sum(axis=1)
    b_perm = b.permute(1, 0)
    b.assign(r + b)
    c.assign(r + b_perm.contiguous())
    Tensor.realize(b, c)
    assert GlobalCounters.kernel_count - kc == 2
    np.testing.assert_equal(b.numpy(), a.numpy().sum(1) + np.arange(32 * 32).reshape(32, 32))
    np.testing.assert_equal(c.numpy(), a.numpy().sum(1) + np.arange(32 * 32).reshape(32, 32).transpose(1, 0))

  def test_permuted_assignment_masked_view_possible(self):
    a = Tensor.ones(4, 4).contiguous().realize()
    b = a.shrink((None, (0, 2))).pad((None, (0, 2)), value=2)
    a.assign(a + b)
    kc = GlobalCounters.kernel_count
    a.realize()
    assert GlobalCounters.kernel_count - kc == 1
    np.testing.assert_equal(a.numpy(), np.ones((4, 4))+np.pad(np.ones((4, 4))[:, 0:2], ((0, 0), (0, 2)), constant_values=2))

  def test_permuted_assignment_masked_view_not_contiguous(self):
    a = Tensor.ones(4, 4).contiguous().realize()
    b = a.shrink((None, (0, 2))).pad((None, (0, 2)), value=2).permute(1, 0)
    a.assign(a + b)
    a.realize()
    self.assertListEqual(a.tolist(), [[2.,2.,2.,2.],[2.,2.,2.,2.],[3.,3.,3.,3.], [3.,3.,3.,3.]])

  # TODO: is there a way to sneak in a permute such that it returns the wrong answer?

  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_setitem_half(self):
    a = Tensor.full((8,), 1.0, dtype=dtypes.half).contiguous().realize()
    b = Tensor.full((4,), 2.0, dtype=dtypes.half).contiguous().realize()
    assign = a[:4].assign(b)
    assign.realize()
    np.testing.assert_allclose(a.numpy(), [2., 2., 2., 2., 1., 1., 1., 1.])

  @unittest.skip("don't use output buffer, and mismatch dtype no longer supported")
  def test_cast_assignment(self):
    a = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    a.realize()
    oba1 = a.uop.base.output_buffer
    a.assign(a.cast(dtypes.int32).realize())
    a.realize()
    oba2 = a.uop.base.output_buffer
    assert oba1 is None and oba2 is None
    np.testing.assert_allclose(a.numpy(), np.arange(N*N,dtype=np.int32).reshape((N,N)))

  def test_disk_assignment(self):
    a = Tensor.empty(5, device=f"disk:{temp('disk_assignment')}").assign(Tensor.ones(5)).numpy()
    np.testing.assert_equal(a, np.ones(5))

  def test_assign_slice_then_read(self):
    """Assign to slice then read from buffer - read should see the assigned values.
    This is the KV cache pattern from llm.py.
    """
    v_pos = Variable("pos", 0, 3).bind(0)
    cache = Tensor.zeros(4, 4).contiguous().realize()
    cache[v_pos:v_pos+1, :].assign(Tensor.ones(1, 4))
    self.assertEqual(cache.sum().item(), 4.0)

class TestAssignOrdering(unittest.TestCase):
  """Tests for complex assign orderings that could differ between lazy and eager execution.

  The key principle: tinygrad's lazy execution with RAW/WAR dependency tracking should
  produce the same results as eager (immediate) execution for valid programs.

  These tests exercise edge cases where incorrect dependency tracking could cause:
  - Stale reads (reading before write completes)
  - Lost writes (write ordering reversed)
  - Race conditions (concurrent access to same buffer)
  """

  def test_overlapping_slice_assigns(self):
    """Overlapping slice assigns - later write should win for overlapping elements."""
    buf = Tensor.zeros(8).contiguous().realize()
    buf[0:4].assign(Tensor.ones(4))
    buf[2:6].assign(Tensor.ones(4) * 2)
    np.testing.assert_equal(buf.numpy(), [1,1,2,2,2,2,0,0])

  def test_overlapping_slice_assigns_reverse(self):
    """Overlapping slice assigns in reverse order."""
    buf = Tensor.zeros(8).contiguous().realize()
    buf[2:6].assign(Tensor.ones(4) * 2)
    buf[0:4].assign(Tensor.ones(4))
    np.testing.assert_equal(buf.numpy(), [1,1,1,1,2,2,0,0])

  def test_read_between_writes(self):
    """Read should see first write before second write happens."""
    buf = Tensor.zeros(4).contiguous().realize()
    buf.assign(Tensor.ones(4))
    r1 = buf.sum().realize()  # should see ones = 4
    buf.assign(Tensor.ones(4) * 2)
    r2 = buf.sum().realize()  # should see twos = 8
    self.assertEqual(r1.item(), 4)
    self.assertEqual(r2.item(), 8)

  def test_write_read_write_chain(self):
    """Write, read, write chain - middle read must complete before second write."""
    buf = Tensor.zeros(4).contiguous().realize()
    buf.assign(Tensor.ones(4) * 3)
    mid_sum = buf.sum()  # lazy read, should be 12
    buf.assign(Tensor.ones(4) * 5)
    final_sum = buf.sum()  # lazy read, should be 20
    # Realize in "wrong" order - final first
    self.assertEqual(final_sum.realize().item(), 20)
    self.assertEqual(mid_sum.realize().item(), 12)

  def test_slice_read_then_full_write(self):
    """Read from slice, then overwrite full buffer - WAR dependency works for full buffer assigns."""
    buf = Tensor([1.,2.,3.,4.]).contiguous().realize()
    partial = buf[0:2].sum()  # lazy read
    buf.assign(Tensor.ones(4) * 10)  # overwrite everything
    full = buf.sum()
    # WAR dependency correctly tracked - partial sees original data
    self.assertEqual(partial.realize().item(), 3)  # 1+2
    self.assertEqual(full.realize().item(), 40)

  def test_slice_write_then_full_read(self):
    """Write to slice, then read full buffer - orphan assign now triggered by .numpy()."""
    buf = Tensor.zeros(4).contiguous().realize()
    buf[1:3].assign(Tensor([5, 6]))
    np.testing.assert_equal(buf.numpy(), [0, 5, 6, 0])

  def test_chained_slice_copies(self):
    """Copy from one slice to another within same buffer - orphan assign now triggered by .numpy()."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    buf[4:8].assign(buf[0:4].contiguous())
    np.testing.assert_equal(buf.numpy(), [1, 2, 3, 4, 1, 2, 3, 4])

  def test_swap_slices(self):
    """Swap two non-overlapping slices - lazy assign behaves same as eager (values captured before writes)."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    left = buf[0:4].contiguous()  # lazy, but value captured when assign writes to same buffer
    right = buf[4:8].contiguous()  # lazy, but value captured when assign writes to same buffer
    buf[0:4].assign(right).realize()
    buf[4:8].assign(left).realize()
    np.testing.assert_equal(buf.numpy(), [5, 6, 7, 8, 1, 2, 3, 4])  # proper swap

  def test_swap_slices_overlapped(self):
    """Swap two overlapping slices - lazy assign behaves same as eager (values captured before writes)."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    left = buf[0:5].contiguous()   # indices 0-4, lazy but captured before writes
    right = buf[3:8].contiguous()  # indices 3-7, overlaps with left at indices 3-4
    buf[0:5].assign(right).realize()  # buf becomes [4, 5, 6, 7, 8, 6, 7, 8]
    buf[3:8].assign(left).realize()   # left was captured as [1, 2, 3, 4, 5], buf becomes [4, 5, 6, 1, 2, 3, 4, 5]
    np.testing.assert_equal(buf.numpy(), [4, 5, 6, 1, 2, 3, 4, 5])

  def test_reduction_after_partial_assign(self):
    """Reduction over buffer after partial assign - must see the assigned values."""
    buf = Tensor.zeros(4, 4).contiguous().realize()
    buf[0:2, :].assign(Tensor.ones(2, 4))  # top half = 1
    total = buf.sum()
    self.assertEqual(total.item(), 8)  # 2*4 ones

  def test_multiple_reductions_different_views(self):
    """Multiple reductions over different views of same buffer after assign."""
    buf = Tensor.zeros(4, 4).contiguous().realize()
    buf.assign(Tensor.arange(16).reshape(4, 4).float())
    row_sums = buf.sum(axis=1)  # [6, 22, 38, 54]
    col_sums = buf.sum(axis=0)  # [24, 28, 32, 36]
    total = buf.sum()  # 120
    # All should see the assigned values
    np.testing.assert_equal(row_sums.numpy(), [6, 22, 38, 54])
    np.testing.assert_equal(col_sums.numpy(), [24, 28, 32, 36])
    self.assertEqual(total.item(), 120)

  def test_assign_from_self_transformed(self):
    """Assign to buffer from transformed view of itself."""
    buf = Tensor([1, 2, 3, 4]).contiguous().realize()
    # Read and transform, then write back (requires reading before writing)
    buf.assign((buf * 2).contiguous())
    np.testing.assert_equal(buf.numpy(), [2, 4, 6, 8])

  def test_two_buffers_cross_assign(self):
    """Two buffers each reading from the other before writing."""
    a = Tensor([1, 2, 3, 4]).contiguous().realize()
    b = Tensor([10, 20, 30, 40]).contiguous().realize()
    # Both read from each other's original values
    a_new = (a + b).contiguous()
    b_new = (a * b).contiguous()
    a.assign(a_new)
    b.assign(b_new)
    Tensor.realize(a, b)
    np.testing.assert_equal(a.numpy(), [11, 22, 33, 44])
    np.testing.assert_equal(b.numpy(), [10, 40, 90, 160])

  def test_three_buffer_chain(self):
    """Chain: A depends on B, B depends on C - ordering matters."""
    a = Tensor.zeros(4).contiguous().realize()
    b = Tensor([1, 2, 3, 4]).contiguous().realize()
    c = Tensor([10, 10, 10, 10]).contiguous().realize()
    # b reads from c, a reads from b
    b.assign((b + c).contiguous())  # b = [11, 12, 13, 14]
    a.assign((a + b).contiguous())  # a should see new b = [11, 12, 13, 14]
    Tensor.realize(a, b)
    np.testing.assert_equal(b.numpy(), [11, 12, 13, 14])
    np.testing.assert_equal(a.numpy(), [11, 12, 13, 14])

  def test_interleaved_assign_read_patterns(self):
    """Complex interleaved pattern: write A, read A into B, write B, read B."""
    a = Tensor.zeros(4).contiguous().realize()
    b = Tensor.zeros(4).contiguous().realize()

    a.assign(Tensor([1, 2, 3, 4]))
    b.assign(a.contiguous())       # b should get [1,2,3,4]
    a.assign(Tensor([5, 6, 7, 8]))
    result = b.sum()               # should be 10, not 26

    self.assertEqual(result.item(), 10)
    np.testing.assert_equal(a.numpy(), [5, 6, 7, 8])
    np.testing.assert_equal(b.numpy(), [1, 2, 3, 4])

  def test_variable_slice_ordering(self):
    """Variable-indexed slices - tests symbolic dependency tracking."""
    v_i = Variable("i", 0, 3)

    buf = Tensor.zeros(4, 4).contiguous().realize()
    buf[v_i.bind(0):v_i.bind(0)+1, :].assign(Tensor.ones(1, 4))
    row0_sum = buf[0:1, :].sum()
    self.assertEqual(row0_sum.item(), 4)

    buf[v_i.bind(1):v_i.bind(1)+1, :].assign(Tensor.ones(1, 4) * 2)
    row1_sum = buf[1:2, :].sum()
    self.assertEqual(row1_sum.item(), 8)

  def test_multiple_slice_assigns_then_read(self):
    """Multiple non-overlapping slice assigns then read - RAW dependencies must ensure all writes complete before read."""
    buf = Tensor.zeros(4).contiguous().realize()
    buf[0:1].assign(Tensor.ones(1))
    buf[1:2].assign(Tensor.full((1,), 2.0))
    buf[2:3].assign(Tensor.full((1,), 3.0))
    # Sum should see all three writes: 1 + 2 + 3 + 0 = 6
    self.assertEqual(buf.sum().realize().item(), 6)

  def test_swap_slices_single_realize(self):
    """Swap slices with batched realize - both assigns in one schedule."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    left = buf[0:4].contiguous()
    right = buf[4:8].contiguous()
    a1 = buf[0:4].assign(right)
    a2 = buf[4:8].assign(left)
    Tensor.realize(a1, a2)
    np.testing.assert_equal(buf.numpy(), [5, 6, 7, 8, 1, 2, 3, 4])

  def test_swap_slices_2d(self):
    """Swap 2D slices - tests multi-dimensional view handling."""
    buf = Tensor.arange(16).reshape(4, 4).contiguous().realize()
    # Swap top-left 2x2 with bottom-right 2x2
    top_left = buf[0:2, 0:2].contiguous()      # [[0,1], [4,5]]
    bottom_right = buf[2:4, 2:4].contiguous()  # [[10,11], [14,15]]
    buf[0:2, 0:2].assign(bottom_right).realize()
    buf[2:4, 2:4].assign(top_left).realize()
    expected = np.array([[10, 11, 2, 3], [14, 15, 6, 7], [8, 9, 0, 1], [12, 13, 4, 5]])
    np.testing.assert_equal(buf.numpy(), expected)

  def test_rotate_three_slices(self):
    """Three-way rotation: A → B, B → C, C → A."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).contiguous().realize()
    a = buf[0:3].contiguous()  # [1, 2, 3]
    b = buf[3:6].contiguous()  # [4, 5, 6]
    c = buf[6:9].contiguous()  # [7, 8, 9]
    # Rotate: A gets C's value, B gets A's value, C gets B's value
    buf[0:3].assign(c).realize()  # A = [7, 8, 9]
    buf[3:6].assign(a).realize()  # B = [1, 2, 3] (captured before first assign)
    buf[6:9].assign(b).realize()  # C = [4, 5, 6] (captured before first assign)
    np.testing.assert_equal(buf.numpy(), [7, 8, 9, 1, 2, 3, 4, 5, 6])

  def test_contiguous_chain(self):
    """Multiple contiguous calls in chain."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    # Create contiguous views and use them in assignment
    view2 = buf[4:8].contiguous()
    buf[0:4].assign(view2).realize()
    np.testing.assert_equal(buf.numpy(), [5, 6, 7, 8, 5, 6, 7, 8])

  def test_cross_buffer_slice_assign(self):
    """Assign slice from one buffer to another buffer's slice."""
    buf1 = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    buf2 = Tensor([10, 20, 30, 40, 50, 60, 70, 80]).contiguous().realize()
    buf1[0:4].assign(buf2[4:8].contiguous())
    np.testing.assert_equal(buf1.numpy(), [50, 60, 70, 80, 5, 6, 7, 8])

  def test_multiple_lazy_contiguous_batch_realize(self):
    """Create multiple lazy contiguous views, then batch realize."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    v1 = buf[0:2].contiguous()
    v2 = buf[2:4].contiguous()
    v3 = buf[4:6].contiguous()
    v4 = buf[6:8].contiguous()
    # Realize all at once - should all see original values
    Tensor.realize(v1, v2, v3, v4)
    np.testing.assert_equal(v1.numpy(), [1, 2])
    np.testing.assert_equal(v2.numpy(), [3, 4])
    np.testing.assert_equal(v3.numpy(), [5, 6])
    np.testing.assert_equal(v4.numpy(), [7, 8])

  def test_write_after_write_same_region(self):
    """Multiple writes to same region - last write wins."""
    buf = Tensor.zeros(8).contiguous().realize()
    buf[0:4].assign(Tensor.ones(4))
    buf[0:4].assign(Tensor.ones(4) * 2)
    buf[0:4].assign(Tensor.ones(4) * 3)
    np.testing.assert_equal(buf.numpy(), [3, 3, 3, 3, 0, 0, 0, 0])

  def test_shrink_then_assign(self):
    """Assign to shrunk view."""
    buf = Tensor.zeros(4, 4).contiguous().realize()
    shrunk = buf[1:3, 1:3]  # 2x2 in the middle
    shrunk.assign(Tensor.ones(2, 2) * 5)
    expected = np.array([[0, 0, 0, 0], [0, 5, 5, 0], [0, 5, 5, 0], [0, 0, 0, 0]], dtype=np.float32)
    np.testing.assert_equal(buf.numpy(), expected)

  def test_assign_then_read_different_view(self):
    """Write to one view, read from a different (overlapping) view."""
    buf = Tensor.zeros(8).contiguous().realize()
    buf[0:4].assign(Tensor([1, 2, 3, 4]))
    # Read from overlapping but different view
    result = buf[2:6].sum()
    self.assertEqual(result.item(), 3 + 4 + 0 + 0)  # [3, 4, 0, 0]

  def test_copy_within_buffer_forward(self):
    """Copy forward within same buffer (src before dst)."""
    buf = Tensor([1, 2, 3, 4, 0, 0, 0, 0]).contiguous().realize()
    buf[4:8].assign(buf[0:4].contiguous())
    np.testing.assert_equal(buf.numpy(), [1, 2, 3, 4, 1, 2, 3, 4])

  def test_copy_within_buffer_backward(self):
    """Copy backward within same buffer (src after dst)."""
    buf = Tensor([0, 0, 0, 0, 5, 6, 7, 8]).contiguous().realize()
    buf[0:4].assign(buf[4:8].contiguous())
    np.testing.assert_equal(buf.numpy(), [5, 6, 7, 8, 5, 6, 7, 8])

  def test_copy_within_buffer_overlapping(self):
    """Copy overlapping region within same buffer."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    # Copy [2,3,4,5] to positions [0,1,2,3]
    buf[0:4].assign(buf[1:5].contiguous())
    np.testing.assert_equal(buf.numpy(), [2, 3, 4, 5, 5, 6, 7, 8])

  # === Tricky edge cases ===

  def test_swap_with_intermediate_computation(self):
    """Swap slices where values go through computation before assign."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    # left reads from right, but with arithmetic
    left_val = (buf[4:8].contiguous() * 2)  # [10, 12, 14, 16]
    right_val = (buf[0:4].contiguous() + 10)  # [11, 12, 13, 14]
    buf[0:4].assign(left_val)
    buf[4:8].assign(right_val)
    Tensor.realize(buf)
    np.testing.assert_equal(buf.numpy(), [10, 12, 14, 16, 11, 12, 13, 14])

  def test_diamond_read_pattern(self):
    """Two different computations read same source before full buffer assign."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    # Direct reads without intermediate contiguous
    sum_val = buf[0:4].sum()  # 10
    doubled = buf[0:4] * 2  # [2, 4, 6, 8]
    # Full buffer assign (WAR dependency tracked via rangeify)
    buf.assign(Tensor([100, 100, 100, 100, 100, 100, 100, 100]))
    # Both should see original values due to WAR tracking
    np.testing.assert_equal(sum_val.numpy(), 10)
    np.testing.assert_equal(doubled.numpy(), [2, 4, 6, 8])

  def test_triple_swap_all_at_once(self):
    """Rotate three regions A->B->C->A in single realize."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).contiguous().realize()
    a = buf[0:3].contiguous()  # [1, 2, 3]
    b = buf[3:6].contiguous()  # [4, 5, 6]
    c = buf[6:9].contiguous()  # [7, 8, 9]
    buf[0:3].assign(c)  # A gets C
    buf[3:6].assign(a)  # B gets A
    buf[6:9].assign(b)  # C gets B
    Tensor.realize(buf)
    np.testing.assert_equal(buf.numpy(), [7, 8, 9, 1, 2, 3, 4, 5, 6])

  def test_strided_slice_swap(self):
    """Swap using strided slices."""
    buf = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).contiguous().realize()
    row0 = buf[0].contiguous()  # [1, 2, 3, 4]
    row1 = buf[1].contiguous()  # [5, 6, 7, 8]
    buf[0].assign(row1)
    buf[1].assign(row0)
    Tensor.realize(buf)
    np.testing.assert_equal(buf.numpy(), [[5, 6, 7, 8], [1, 2, 3, 4]])

  def test_assign_derived_from_self_twice(self):
    """Assign value derived from same buffer through two paths."""
    buf = Tensor([1, 2, 3, 4]).contiguous().realize()
    # Two different views of same buffer
    view1 = buf[0:2].contiguous()  # [1, 2]
    view2 = buf[2:4].contiguous()  # [3, 4]
    combined = view1 + view2  # [4, 6] - elementwise add
    buf[0:2].assign(combined)
    np.testing.assert_equal(buf.numpy(), [4, 6, 3, 4])

  def test_clone_before_multiple_writes(self):
    """Clone captures original values before multiple slice writes."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    full_clone = buf.clone()
    buf[0:2].assign(Tensor([10, 20]))
    buf[2:4].assign(Tensor([30, 40]))
    buf[4:6].assign(Tensor([50, 60]))
    np.testing.assert_equal(full_clone.numpy(), [1, 2, 3, 4, 5, 6, 7, 8])

  def test_alternating_assign_realize(self):
    """Alternating assigns and realizes on same buffer."""
    buf = Tensor([1, 2, 3, 4]).contiguous().realize()
    buf[0:2].assign(Tensor([10, 20])).realize()
    np.testing.assert_equal(buf.numpy(), [10, 20, 3, 4])
    buf[2:4].assign(Tensor([30, 40])).realize()
    np.testing.assert_equal(buf.numpy(), [10, 20, 30, 40])
    buf[0:2].assign(Tensor([100, 200])).realize()
    np.testing.assert_equal(buf.numpy(), [100, 200, 30, 40])

  def test_assign_to_reshaped_view(self):
    """Assign to a view that's been reshaped."""
    buf = Tensor([[1, 2], [3, 4], [5, 6], [7, 8]]).contiguous().realize()
    # Reshape to 1D and assign
    flat = buf.reshape(8)
    flat[0:4].assign(Tensor([10, 20, 30, 40]))
    np.testing.assert_equal(buf.numpy(), [[10, 20], [30, 40], [5, 6], [7, 8]])

  def test_contiguous_of_contiguous_swap(self):
    """Swap using contiguous of contiguous views."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    # Double contiguous - shouldn't affect behavior
    left = buf[0:4].contiguous().contiguous()
    right = buf[4:8].contiguous().contiguous()
    buf[0:4].assign(right)
    buf[4:8].assign(left)
    Tensor.realize(buf)
    np.testing.assert_equal(buf.numpy(), [5, 6, 7, 8, 1, 2, 3, 4])

  def test_partial_overlap_left_to_right(self):
    """Assign where src and dst partially overlap (shift right)."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    # src [0:5] = [1,2,3,4,5], dst [3:8] - overlap at [3:5]
    src = buf[0:5].contiguous()
    buf[3:8].assign(src)
    np.testing.assert_equal(buf.numpy(), [1, 2, 3, 1, 2, 3, 4, 5])

  def test_partial_overlap_right_to_left(self):
    """Assign where src and dst partially overlap (shift left)."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    # src [3:8] = [4,5,6,7,8], dst [0:5] - overlap at [3:5]
    src = buf[3:8].contiguous()
    buf[0:5].assign(src)
    np.testing.assert_equal(buf.numpy(), [4, 5, 6, 7, 8, 6, 7, 8])

  def test_four_way_rotation(self):
    """Rotate four regions A->B->C->D->A."""
    buf = Tensor(list(range(16))).contiguous().realize()
    a = buf[0:4].contiguous()   # [0,1,2,3]
    b = buf[4:8].contiguous()   # [4,5,6,7]
    c = buf[8:12].contiguous()  # [8,9,10,11]
    d = buf[12:16].contiguous() # [12,13,14,15]
    buf[0:4].assign(d)   # A gets D
    buf[4:8].assign(a)   # B gets A
    buf[8:12].assign(b)  # C gets B
    buf[12:16].assign(c) # D gets C
    Tensor.realize(buf)
    np.testing.assert_equal(buf.numpy(), [12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

  def test_clone_before_partial_write(self):
    """Clone captures original values before partial write."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    full_clone = buf.clone()
    buf[0:4].assign(Tensor([10, 20, 30, 40])).realize()
    np.testing.assert_equal(full_clone.numpy(), [1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_equal(buf.numpy(), [10, 20, 30, 40, 5, 6, 7, 8])

  def test_assign_reduction_result_to_slice(self):
    """Assign a reduction result to a slice."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    sum_val = buf[0:4].sum()  # 10
    # Assign sum as single-element slice would fail, so broadcast
    result = Tensor([sum_val.numpy().item()] * 4)
    buf[4:8].assign(result)
    np.testing.assert_equal(buf.numpy(), [1, 2, 3, 4, 10, 10, 10, 10])

  def test_nested_buffer_operations(self):
    """Complex nested operations on buffer before assign."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    view = buf[0:4].contiguous()
    transformed = (view * 2) + 1
    buf[0:4].assign(Tensor([100, 100, 100, 100]))
    np.testing.assert_equal(transformed.numpy(), [3, 5, 7, 9])

  # === Full buffer assign and slice assign interaction ===

  def test_full_assign_then_slice_assign(self):
    """Full buffer assign then slice assign."""
    buf = Tensor([1, 2, 3, 4]).contiguous().realize()
    buf.assign(buf + 10)  # [11, 12, 13, 14]
    buf[0:2].assign(Tensor([100, 200]))  # [100, 200, 13, 14]
    np.testing.assert_equal(buf.numpy(), [100, 200, 13, 14])

  def test_slice_assign_then_full_assign(self):
    """Slice assign then full buffer assign."""
    buf = Tensor([1, 2, 3, 4]).contiguous().realize()
    buf[0:2].assign(Tensor([10, 20]))  # [10, 20, 3, 4]
    buf.assign(buf + 100)  # [110, 120, 103, 104]
    np.testing.assert_equal(buf.numpy(), [110, 120, 103, 104])

  # === Inplace operations (+=, -=, *=, etc.) and assign interaction ===

  def test_inplace_add_basic(self):
    """Basic inplace add should modify the buffer."""
    buf = Tensor([1, 2, 3, 4]).contiguous().realize()
    buf += 10
    np.testing.assert_equal(buf.numpy(), [11, 12, 13, 14])

  def test_inplace_add_then_slice_assign(self):
    """Inplace add followed by slice assign."""
    buf = Tensor([1, 2, 3, 4]).contiguous().realize()
    buf += 10  # [11, 12, 13, 14]
    buf[0:2].assign(Tensor([100, 200]))  # [100, 200, 13, 14]
    np.testing.assert_equal(buf.numpy(), [100, 200, 13, 14])

  def test_slice_assign_then_inplace_add(self):
    """Slice assign followed by inplace add."""
    buf = Tensor([1, 2, 3, 4]).contiguous().realize()
    buf[0:2].assign(Tensor([10, 20]))  # [10, 20, 3, 4]
    buf += 100  # [110, 120, 103, 104]
    np.testing.assert_equal(buf.numpy(), [110, 120, 103, 104])

  def test_read_before_inplace_add(self):
    """Read should capture value before inplace add."""
    buf = Tensor([1, 2, 3, 4]).contiguous().realize()
    sum_before = buf.sum()  # should be 10
    buf += 10
    self.assertEqual(sum_before.numpy().item(), 10)
    self.assertEqual(buf.sum().numpy().item(), 50)

  def test_inplace_add_twice(self):
    """Multiple inplace adds should accumulate."""
    buf = Tensor([1, 2, 3, 4]).contiguous().realize()
    buf += 10
    buf += 100
    np.testing.assert_equal(buf.numpy(), [111, 112, 113, 114])

  def test_inplace_mul_then_slice_assign(self):
    """Inplace multiply followed by slice assign."""
    buf = Tensor([1, 2, 3, 4]).contiguous().realize()
    buf *= 2  # [2, 4, 6, 8]
    buf[0:2].assign(Tensor([100, 200]))  # [100, 200, 6, 8]
    np.testing.assert_equal(buf.numpy(), [100, 200, 6, 8])

  def test_slice_inplace_add(self):
    """Inplace add on a slice view."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    buf[0:4] += 10  # [11, 12, 13, 14, 5, 6, 7, 8]
    np.testing.assert_equal(buf.numpy(), [11, 12, 13, 14, 5, 6, 7, 8])

  def test_slice_inplace_add_then_read(self):
    """Inplace add on slice, then read full buffer."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    buf[0:4] += 10
    # [11, 12, 13, 14, 5, 6, 7, 8] -> sum = 11+12+13+14+5+6+7+8 = 76
    self.assertEqual(buf.sum().numpy().item(), 76)

  def test_read_slice_before_inplace_add_on_other_slice(self):
    """Read from one slice before inplace add on different slice."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    left_sum = buf[0:4].sum()  # should be 10
    buf[4:8] += 100  # only modify right half
    self.assertEqual(left_sum.numpy().item(), 10)
    np.testing.assert_equal(buf.numpy(), [1, 2, 3, 4, 105, 106, 107, 108])

  def test_inplace_add_and_assign_interleaved(self):
    """Interleaved inplace ops and assigns."""
    buf = Tensor([1, 2, 3, 4]).contiguous().realize()
    buf += 10  # [11, 12, 13, 14]
    buf[0:2].assign(Tensor([0, 0]))  # [0, 0, 13, 14]
    buf += 1  # [1, 1, 14, 15]
    np.testing.assert_equal(buf.numpy(), [1, 1, 14, 15])

  def test_clone_before_inplace_add(self):
    """Clone should capture value before inplace add."""
    buf = Tensor([1, 2, 3, 4]).contiguous().realize()
    clone = buf.clone()
    buf += 100
    np.testing.assert_equal(clone.numpy(), [1, 2, 3, 4])
    np.testing.assert_equal(buf.numpy(), [101, 102, 103, 104])

  def test_inplace_sub_and_div(self):
    """Test -= and /= operations."""
    buf = Tensor([10, 20, 30, 40]).float().contiguous().realize()
    buf -= 5  # [5, 15, 25, 35]
    buf /= 5  # [1, 3, 5, 7]
    np.testing.assert_equal(buf.numpy(), [1, 3, 5, 7])

  def test_multiple_tensors_inplace_independence(self):
    """Inplace ops on one tensor shouldn't affect another."""
    buf1 = Tensor([1, 2, 3, 4]).contiguous().realize()
    buf2 = Tensor([10, 20, 30, 40]).contiguous().realize()
    buf1 += 100
    buf2 *= 2
    np.testing.assert_equal(buf1.numpy(), [101, 102, 103, 104])
    np.testing.assert_equal(buf2.numpy(), [20, 40, 60, 80])

  def test_slice_inplace_then_slice_assign(self):
    """Slice inplace add followed by slice assign - no realize needed between slice ops."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    buf[0:4] += 10  # [11, 12, 13, 14, 5, 6, 7, 8]
    buf[4:8].assign(Tensor([100, 200, 300, 400]))  # [11, 12, 13, 14, 100, 200, 300, 400]
    np.testing.assert_equal(buf.numpy(), [11, 12, 13, 14, 100, 200, 300, 400])

  def test_multiple_slice_inplace_ops(self):
    """Multiple slice inplace ops without realize."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    buf[0:2] += 10  # [11, 12, 3, 4, 5, 6, 7, 8]
    buf[2:4] *= 2   # [11, 12, 6, 8, 5, 6, 7, 8]
    buf[4:6] -= 1   # [11, 12, 6, 8, 4, 5, 7, 8]
    np.testing.assert_equal(buf.numpy(), [11, 12, 6, 8, 4, 5, 7, 8])

if __name__ == "__main__":
  unittest.main()
