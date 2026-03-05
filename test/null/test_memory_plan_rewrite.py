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

if __name__ == "__main__":
  unittest.main()
