import unittest
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.engine.memory import memory_plan_rewrite

# 1024 float32 = 4096 bytes = 0x1000 = min_block_size, so no rounding overhead
def buf(size=1024, dtype=dtypes.float): return UOp.new_buffer("NULL", size, dtype)
def kernel(*bufs): return UOp(Ops.SINK, arg=KernelInfo()).call(*bufs)
def linear(*calls): return UOp(Ops.LINEAR, src=calls)

class TestMemoryPlanRewrite(unittest.TestCase):
  def test_empty(self):
    l = linear()
    new_l, rmap = memory_plan_rewrite(l)
    self.assertIs(new_l, l)
    self.assertEqual(len(rmap), 0)

  def test_all_external(self):
    b0, b1 = buf(), buf()
    l = linear(kernel(b0, b1))
    new_l, rmap = memory_plan_rewrite(l, external_bufs={b0, b1})
    self.assertIs(new_l, l)
    self.assertEqual(len(rmap), 0)

  def test_internal_becomes_buffer_view(self):
    b_ext, b_int = buf(), buf()
    l = linear(kernel(b_int, b_ext))
    _, rmap = memory_plan_rewrite(l, external_bufs={b_ext})
    self.assertNotIn(b_ext, rmap)
    self.assertIn(b_int, rmap)
    self.assertEqual(rmap[b_int].op, Ops.BUFFER_VIEW)

  def test_all_internal(self):
    b0, b1 = buf(), buf()
    l = linear(kernel(b0, b1))
    _, rmap = memory_plan_rewrite(l)
    self.assertEqual(rmap[b0].op, Ops.BUFFER_VIEW)
    self.assertEqual(rmap[b1].op, Ops.BUFFER_VIEW)

  def test_shared_arena(self):
    b0, b1 = buf(), buf()
    l = linear(kernel(b0, b1))
    _, rmap = memory_plan_rewrite(l)
    self.assertIs(rmap[b0].src[0], rmap[b1].src[0])

  def test_overlapping_arena_size(self):
    b0, b1 = buf(), buf()
    # both alive in same kernel -> arena must fit both
    l = linear(kernel(b0, b1))
    _, rmap = memory_plan_rewrite(l)
    arena = rmap[b0].src[0]
    self.assertGreaterEqual(arena.arg * arena.dtype.itemsize, 2 * 0x1000)

  def test_non_overlapping_reuse(self):
    b0, b1, b2 = buf(), buf(), buf()
    # each only alive in its own kernel -> all reuse same space
    l = linear(kernel(b0), kernel(b1), kernel(b2))
    _, rmap = memory_plan_rewrite(l)
    arena = rmap[b0].src[0]
    self.assertEqual(arena.arg * arena.dtype.itemsize, 0x1000)

  def test_non_overlapping_offset_zero(self):
    b0, b1 = buf(), buf()
    l = linear(kernel(b0), kernel(b1))
    _, rmap = memory_plan_rewrite(l)
    self.assertEqual(rmap[b0].arg[1], 0)
    self.assertEqual(rmap[b1].arg[1], 0)

  def test_overlapping_different_offsets(self):
    b0, b1 = buf(), buf()
    l = linear(kernel(b0, b1))
    _, rmap = memory_plan_rewrite(l)
    self.assertNotEqual(rmap[b0].arg[1], rmap[b1].arg[1])

  def test_partial_overlap(self):
    b0, b1, b2 = buf(), buf(), buf()
    b_ext = buf()
    # b0 alive in k0,k1 | b1 alive in k1,k2 | b2 alive in k2,k3
    # b0 and b1 overlap, b0 and b2 don't -> arena < 3 bufs
    l = linear(kernel(b0, b_ext), kernel(b1, b0), kernel(b2, b1), kernel(b_ext, b2))
    _, rmap = memory_plan_rewrite(l, external_bufs={b_ext})
    arena = rmap[b0].src[0]
    self.assertLessEqual(arena.arg * arena.dtype.itemsize, 2 * 0x1000)

  def test_buffer_view_size_preserved(self):
    b = buf(size=2048)
    l = linear(kernel(b))
    _, rmap = memory_plan_rewrite(l)
    self.assertEqual(rmap[b].arg[0], 2048)

  def test_linear_rewritten(self):
    b = buf()
    l = linear(kernel(b))
    new_l, rmap = memory_plan_rewrite(l)
    self.assertIsNot(new_l, l)
    # the BUFFER in the new linear should be replaced with BUFFER_VIEW
    call = new_l.src[0]
    self.assertEqual(call.src[1].op, Ops.BUFFER_VIEW)

class TestMemoryPlanRewriteMulti(unittest.TestCase):
  def test_tuple_device_skipped(self):
    """Buffers with tuple device (multi-device) must not enter the memory planner."""
    b_multi = UOp.new_buffer(("NULL", "NULL:1"), 1024, dtypes.float)
    b_ext = buf()
    l = linear(kernel(b_multi, b_ext))
    _, rmap = memory_plan_rewrite(l, external_bufs={b_ext})
    self.assertNotIn(b_multi, rmap)

  def test_mstack_extends_lifetime(self):
    """Buffers referenced via MSTACK in a later step must not be freed early."""
    b_dev0 = UOp.new_buffer("NULL", 1024, dtypes.float)
    b_dev1 = UOp.new_buffer("NULL:1", 1024, dtypes.float)
    b_other = UOp.new_buffer("NULL", 1024, dtypes.float)
    b_out = UOp.new_buffer(("NULL", "NULL:1"), 2048, dtypes.float)
    mstk = UOp(Ops.MSTACK, dtypes.float, (b_dev0, b_dev1))
    # step 0: produce b_dev0, step 1: produce b_dev1,
    # step 2: produce b_other (b_dev0 still alive!), step 3: consume via MSTACK
    l = linear(kernel(b_dev0), kernel(b_dev1), kernel(b_other), kernel(b_out, mstk))
    _, rmap = memory_plan_rewrite(l)
    # b_dev0 and b_other overlap in lifetime -> must not alias
    self.assertNotEqual(rmap[b_dev0].arg[1], rmap[b_other].arg[1])

  def test_mselect_extends_lifetime(self):
    """Buffers referenced via MSELECT(MSTACK(...)) in a later step must not be freed early."""
    b_a = UOp.new_buffer("NULL", 1024, dtypes.float)
    b_b = UOp.new_buffer("NULL:1", 1024, dtypes.float)
    b_c = UOp.new_buffer("NULL", 1024, dtypes.float)
    mstk = UOp(Ops.MSTACK, dtypes.float, (b_a, b_b))
    msel = mstk.mselect(0)
    # step 0: produce b_a, step 1: produce b_b,
    # step 2: produce b_c (b_a still alive!), step 3: consume b_a via mselect(mstack)
    l = linear(kernel(b_a), kernel(b_b), kernel(b_c), kernel(b_c, msel))
    _, rmap = memory_plan_rewrite(l)
    self.assertNotEqual(rmap[b_a].arg[1], rmap[b_c].arg[1])

  def test_mstack_buffers_become_buffer_view(self):
    """Per-device buffers consumed via MSTACK should still be planned (become BUFFER_VIEW)."""
    b_dev0 = UOp.new_buffer("NULL", 1024, dtypes.float)
    b_dev1 = UOp.new_buffer("NULL:1", 1024, dtypes.float)
    b_out = UOp.new_buffer(("NULL", "NULL:1"), 2048, dtypes.float)
    mstk = UOp(Ops.MSTACK, dtypes.float, (b_dev0, b_dev1))
    l = linear(kernel(b_dev0), kernel(b_dev1), kernel(b_out, mstk))
    _, rmap = memory_plan_rewrite(l)
    self.assertIn(b_dev0, rmap)
    self.assertEqual(rmap[b_dev0].op, Ops.BUFFER_VIEW)
    self.assertIn(b_dev1, rmap)
    self.assertEqual(rmap[b_dev1].op, Ops.BUFFER_VIEW)

  def test_mstack_non_overlapping_reuse(self):
    """Per-device buffers with non-overlapping lifetimes should reuse space."""
    b0 = UOp.new_buffer("NULL", 1024, dtypes.float)
    b1 = UOp.new_buffer("NULL", 1024, dtypes.float)
    b_out0 = UOp.new_buffer(("NULL", "NULL:1"), 2048, dtypes.float)
    b_out1 = UOp.new_buffer(("NULL", "NULL:1"), 2048, dtypes.float)
    b_x = UOp.new_buffer("NULL:1", 1024, dtypes.float)
    mstk0 = UOp(Ops.MSTACK, dtypes.float, (b0, b_x))
    mstk1 = UOp(Ops.MSTACK, dtypes.float, (b1, b_x))
    # b0 alive [0,1], b1 alive [2,3] -> non-overlapping, can reuse
    l = linear(kernel(b0), kernel(b_out0, mstk0), kernel(b1), kernel(b_out1, mstk1))
    _, rmap = memory_plan_rewrite(l)
    self.assertEqual(rmap[b0].arg[1], rmap[b1].arg[1])

class TestFunctionMemoryPlan(unittest.TestCase):
  def test_function_chain_shares_buffers(self):
    from tinygrad import Tensor, function
    @function
    def f(x:Tensor) -> Tensor: return (x + 1).contiguous()
    a = Tensor.ones(1024).contiguous()
    schedule = f(f(f(a))).schedule()
    internal_bufs = [b for ei in schedule for b in ei.bufs if b is not None and b._base is not None]
    self.assertGreater(len(internal_bufs), 0)
    bases = set(b._base for b in internal_bufs)
    self.assertEqual(len(bases), 1)

  def test_nested_function_calls(self):
    from tinygrad import Tensor, function
    @function
    def f(x:Tensor) -> Tensor: return (x + 1).contiguous()
    @function
    def g(x:Tensor) -> Tensor: return f(f(x))
    a = Tensor.ones(1024).contiguous()
    schedule = g(a).schedule()
    internal_bufs = [b for ei in schedule for b in ei.bufs if b is not None and b._base is not None]
    self.assertGreater(len(internal_bufs), 0)
    bases = set(b._base for b in internal_bufs)
    self.assertEqual(len(bases), 1)

if __name__ == "__main__":
  unittest.main()
