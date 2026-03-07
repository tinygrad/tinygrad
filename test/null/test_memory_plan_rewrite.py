import unittest
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.engine.memory import memory_plan_rewrite

# 1024 float32 = 4096 bytes, 32-byte aligned so no rounding overhead
def buf(size=1024, dtype=dtypes.float): return UOp.new_buffer("NULL", size, dtype)
def kernel(*bufs): return UOp(Ops.SINK, arg=KernelInfo()).call(*bufs)
def linear(*calls): return UOp(Ops.LINEAR, src=calls)

def get_buffer_views(l:UOp) -> list[UOp]:
  """Collect all BUFFER_VIEW UOps from a rewritten linear."""
  return [src for si in l.src for src in si.src[1:] if src.op is Ops.BUFFER_VIEW]

class TestMemoryPlanRewrite(unittest.TestCase):
  def test_empty(self):
    l = linear()
    self.assertIs(memory_plan_rewrite(l), l)

  def test_all_external(self):
    b0, b1 = buf(), buf()
    l = linear(kernel(b0, b1))
    self.assertIs(memory_plan_rewrite(l, external_bufs={b0, b1}), l)

  def test_internal_becomes_buffer_view(self):
    b_ext, b_int = buf(), buf()
    l = linear(kernel(b_int, b_ext))
    new_l = memory_plan_rewrite(l, external_bufs={b_ext})
    views = get_buffer_views(new_l)
    self.assertEqual(len(views), 1)

  def test_all_internal(self):
    b0, b1 = buf(), buf()
    l = linear(kernel(b0, b1))
    new_l = memory_plan_rewrite(l)
    views = get_buffer_views(new_l)
    self.assertEqual(len(views), 2)

  def test_shared_arena(self):
    b0, b1 = buf(), buf()
    l = linear(kernel(b0, b1))
    new_l = memory_plan_rewrite(l)
    views = get_buffer_views(new_l)
    self.assertIs(views[0].src[0], views[1].src[0])

  def test_overlapping_arena_size(self):
    b0, b1 = buf(), buf()
    l = linear(kernel(b0, b1))
    new_l = memory_plan_rewrite(l)
    views = get_buffer_views(new_l)
    arena = views[0].src[0]
    self.assertGreaterEqual(arena.arg * arena.dtype.itemsize, 2 * 4096)

  def test_non_overlapping_reuse(self):
    b0, b1, b2 = buf(), buf(), buf()
    l = linear(kernel(b0), kernel(b1), kernel(b2))
    new_l = memory_plan_rewrite(l)
    views = get_buffer_views(new_l)
    arena = views[0].src[0]
    self.assertEqual(arena.arg * arena.dtype.itemsize, 4096)

  def test_non_overlapping_offset_zero(self):
    b0, b1 = buf(), buf()
    l = linear(kernel(b0), kernel(b1))
    new_l = memory_plan_rewrite(l)
    views = get_buffer_views(new_l)
    self.assertEqual(views[0].arg[1], 0)
    self.assertEqual(views[1].arg[1], 0)

  def test_overlapping_different_offsets(self):
    b0, b1 = buf(), buf()
    l = linear(kernel(b0, b1))
    new_l = memory_plan_rewrite(l)
    views = get_buffer_views(new_l)
    self.assertNotEqual(views[0].arg[1], views[1].arg[1])

  def test_partial_overlap(self):
    b0, b1, b2 = buf(), buf(), buf()
    b_ext = buf()
    l = linear(kernel(b0, b_ext), kernel(b1, b0), kernel(b2, b1), kernel(b_ext, b2))
    new_l = memory_plan_rewrite(l, external_bufs={b_ext})
    views = get_buffer_views(new_l)
    arena = views[0].src[0]
    self.assertLessEqual(arena.arg * arena.dtype.itemsize, 2 * 4096)

  def test_buffer_view_size_preserved(self):
    b = buf(size=2048)
    l = linear(kernel(b))
    new_l = memory_plan_rewrite(l)
    views = get_buffer_views(new_l)
    self.assertEqual(views[0].arg[0], 2048)

  def test_linear_rewritten(self):
    b = buf()
    l = linear(kernel(b))
    new_l = memory_plan_rewrite(l)
    self.assertIsNot(new_l, l)
    self.assertEqual(new_l.src[0].src[1].op, Ops.BUFFER_VIEW)

class TestMemoryPlanRewriteMulti(unittest.TestCase):
  def test_tuple_device_skipped(self):
    """Buffers with tuple device (multi-device) must not enter the memory planner."""
    b_multi = UOp.new_buffer(("NULL", "NULL:1"), 1024, dtypes.float)
    b_ext = buf()
    l = linear(kernel(b_multi, b_ext))
    new_l = memory_plan_rewrite(l, external_bufs={b_ext})
    # b_multi should remain a BUFFER, not become BUFFER_VIEW
    self.assertEqual(new_l.src[0].src[1].op, Ops.BUFFER)

  def test_mstack_extends_lifetime(self):
    """Buffers referenced via MSTACK in a later step must not be freed early."""
    b_dev0 = UOp.new_buffer("NULL", 1024, dtypes.float)
    b_dev1 = UOp.new_buffer("NULL:1", 1024, dtypes.float)
    b_other = UOp.new_buffer("NULL", 1024, dtypes.float)
    b_out = UOp.new_buffer(("NULL", "NULL:1"), 2048, dtypes.float)
    mstk = UOp(Ops.MSTACK, dtypes.float, (b_dev0, b_dev1))
    l = linear(kernel(b_dev0), kernel(b_dev1), kernel(b_other), kernel(b_out, mstk))
    new_l = memory_plan_rewrite(l)
    # b_dev0 and b_other are both on NULL, find their views
    null_views = [src for si in new_l.src for src in si.src[1:] if src.op is Ops.BUFFER_VIEW and src.src[0].device == "NULL"]
    offsets = set(v.arg[1] for v in null_views)
    self.assertGreater(len(offsets), 1)  # must have different offsets (not aliased)

  def test_mselect_extends_lifetime(self):
    """Buffers referenced via MSELECT(MSTACK(...)) in a later step must not be freed early."""
    b_a = UOp.new_buffer("NULL", 1024, dtypes.float)
    b_b = UOp.new_buffer("NULL:1", 1024, dtypes.float)
    b_c = UOp.new_buffer("NULL", 1024, dtypes.float)
    mstk = UOp(Ops.MSTACK, dtypes.float, (b_a, b_b))
    msel = mstk.mselect(0)
    l = linear(kernel(b_a), kernel(b_b), kernel(b_c), kernel(b_c, msel))
    new_l = memory_plan_rewrite(l)
    null_views = [src for si in new_l.src for src in si.src[1:] if src.op is Ops.BUFFER_VIEW and src.src[0].device == "NULL"]
    offsets = set(v.arg[1] for v in null_views)
    self.assertGreater(len(offsets), 1)

  def test_mstack_buffers_become_buffer_view(self):
    """Per-device buffers consumed via MSTACK should still be planned (become BUFFER_VIEW)."""
    b_dev0 = UOp.new_buffer("NULL", 1024, dtypes.float)
    b_dev1 = UOp.new_buffer("NULL:1", 1024, dtypes.float)
    b_out = UOp.new_buffer(("NULL", "NULL:1"), 2048, dtypes.float)
    mstk = UOp(Ops.MSTACK, dtypes.float, (b_dev0, b_dev1))
    l = linear(kernel(b_dev0), kernel(b_dev1), kernel(b_out, mstk))
    new_l = memory_plan_rewrite(l)
    views = get_buffer_views(new_l)
    self.assertGreaterEqual(len(views), 2)

  def test_mstack_non_overlapping_reuse(self):
    """Per-device buffers with non-overlapping lifetimes should reuse space."""
    b0 = UOp.new_buffer("NULL", 1024, dtypes.float)
    b1 = UOp.new_buffer("NULL", 1024, dtypes.float)
    b_out0 = UOp.new_buffer(("NULL", "NULL:1"), 2048, dtypes.float)
    b_out1 = UOp.new_buffer(("NULL", "NULL:1"), 2048, dtypes.float)
    b_x = UOp.new_buffer("NULL:1", 1024, dtypes.float)
    mstk0 = UOp(Ops.MSTACK, dtypes.float, (b0, b_x))
    mstk1 = UOp(Ops.MSTACK, dtypes.float, (b1, b_x))
    l = linear(kernel(b0), kernel(b_out0, mstk0), kernel(b1), kernel(b_out1, mstk1))
    new_l = memory_plan_rewrite(l)
    # b0 and b1 don't overlap -> should get same offset
    null_views = [src for si in new_l.src for src in si.src[1:] if src.op is Ops.BUFFER_VIEW and src.src[0].device == "NULL"]
    self.assertTrue(all(v.arg[1] == 0 for v in null_views))

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
