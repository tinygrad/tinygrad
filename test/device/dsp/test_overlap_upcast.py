import unittest, numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import AxisType
from tinygrad.codegen.opt import Opt, OptOps

def applied_opts(t:Tensor) -> list[tuple]:
  from tinygrad.codegen import to_program
  linear, _ = t.linear_with_vars()
  calls = [u for u in linear.src if u.src[0].op.name == "SINK"]
  return [to_program(c.src[0], Device[Device.DEFAULT].renderer).src[0].arg.applied_opts for c in calls]

@unittest.skipUnless(Device[Device.DEFAULT].renderer.vector_lanes, "an upcast overlaps on a renderer with vectors")
class TestOverlapUpcast(unittest.TestCase):
  def test_overlap(self):
    a, b = np.random.randint(-100, 100, 100).astype(np.int8), np.random.randint(-100, 100, 100).astype(np.int8)
    np.testing.assert_equal((Tensor(a) + Tensor(b)).numpy(), a+b)
    self.assertIn((Opt(OptOps.SPLIT, 0, (64, AxisType.UPCAST)),), applied_opts(Tensor(a) + Tensor(b)))

  def test_no_overlap_reading_output(self):
    t = Tensor(np.arange(100, dtype=np.int8)).realize()
    t += 1
    np.testing.assert_equal(t.numpy(), np.arange(100, dtype=np.int8)+1)
    t.assign(t+1)
    self.assertNotIn((Opt(OptOps.SPLIT, 0, (64, AxisType.UPCAST)),), applied_opts(t))

  def test_no_overlap_reduce(self):
    x = np.random.randint(-3, 3, (4, 100)).astype(np.int32)
    out = Tensor(x, dtype=dtypes.int32).sum(1)
    np.testing.assert_equal(out.numpy(), x.sum(1))

  def test_4d_view_writes(self):
    a = np.arange(120, dtype=np.int8).reshape(2, 3, 4, 5)
    b = np.full((1, 3, 4, 5), 42, dtype=np.int8)
    t = Tensor(a).contiguous().realize()
    t[1:2] = Tensor(b)
    ref = a.copy()
    ref[1:2] = b
    np.testing.assert_equal(t.numpy(), ref)
    t[:, 1:2] += 3
    ref[:, 1:2] += 3
    np.testing.assert_equal(t.numpy(), ref)
    p = np.full((3, 4, 5, 2), -7, dtype=np.int8)
    t.permute(1, 2, 3, 0).assign(Tensor(p)).realize()
    np.testing.assert_equal(t.numpy(), p.transpose(3, 0, 1, 2))

  def test_bool_buffers(self):
    a, b = np.random.randint(-50, 50, 256, dtype=np.int8), np.random.randint(-50, 50, 256, dtype=np.int8)
    m = (Tensor(a) > Tensor(b)).realize()
    np.testing.assert_equal(m.numpy(), a > b)
    m1 = Tensor((a[:100] > 0)).realize()
    m2 = Tensor((b[:100] > 0)).realize()
    np.testing.assert_equal((m1 & m2).numpy(), (a[:100] > 0) & (b[:100] > 0))

class TestDSPAllocator(unittest.TestCase):
  def test_arena_tlsf_sizing(self):
    from tinygrad.runtime.ops_dsp import DSPAllocator, DSPBuffer
    from tinygrad.device import BufferSpec
    class FakeShare: fd = 99
    class FakeDSP:
      util_handle = 1
      def __init__(self): self.n = 0
      def util(self, op, *vals, data=b"", fd=None):
        self.n += 1
        return (0x10000000 * self.n, 0)
    alloc = DSPAllocator(FakeDSP())
    alloc._ion = lambda sz: DSPBuffer(0x20000000, sz, FakeShare())
    for sz in (0, 16, 1024, 31 << 20, 32 << 20, 64 << 20):
      s = alloc._alloc(sz, BufferSpec())
      self.assertEqual(s.buf.size, sz)
      alloc._free(s, BufferSpec())

if __name__ == '__main__':
  unittest.main()
