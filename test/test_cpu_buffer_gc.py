#!/usr/bin/env python
import gc, unittest, weakref
from tinygrad.tensor import Tensor
from tinygrad.device import Buffer
from tinygrad.helpers import GlobalCounters
from tinygrad import Device

# ---------- helpers ----------------------------------------------------------------
def bufs_allocated() -> int:
  gc.collect()
  return sum(isinstance(o, Buffer) and o.is_allocated() for o in gc.get_objects())

CPU_ONLY = Device.DEFAULT == "CPU"

def skip_if_not_cpu(fn):
  return unittest.skipUnless(CPU_ONLY, "GC tests run only on CPU backend")(fn)

# ---------- tests ------------------------------------------------------------------
class TestGCCpu(unittest.TestCase):
  @skip_if_not_cpu
  def test_single_buffer(self):
    Tensor.manual_seed(0)
    _ = Tensor.randn(1)
    base = bufs_allocated()
    t = Tensor.randn(512, 512)
    ref = weakref.ref(t._buffer)
    del t
    gc.collect()
    self.assertIsNone(ref())
    self.assertLessEqual(bufs_allocated() - base, 0)
  @skip_if_not_cpu
  def test_churn(self):
    loops = 300
    base_mem = GlobalCounters.mem_used
    for _ in range(loops):
      Tensor.randn(128, 128)
    gc.collect()
    self.assertLess(GlobalCounters.mem_used - base_mem, 64 * 1024)
  @skip_if_not_cpu
  def test_manual_buffer_gc(self):
    import numpy as np
    Tensor.manual_seed(0)
    _ = Tensor.randn(1)
    baseline = bufs_allocated()

    dtype = Tensor([0.0]).dtype
    buf   = Buffer("CPU", 1024, dtype)
    buf.allocate()

    src = np.arange(1024, dtype=np.float32)
    buf.copyin(memoryview(src))
    dst = np.empty_like(src)
    buf.copyout(memoryview(dst))
    np.testing.assert_array_equal(src, dst)

    ref = weakref.ref(buf)
    del buf
    for _ in range(2): gc.collect()

    self.assertIsNone(ref())
    self.assertEqual(bufs_allocated() - baseline, 0)
  @skip_if_not_cpu
  def test_view_buffer_release(self):
    t = Tensor.randn(256, 256).contiguous()
    base = t._buffer
    view = Buffer(t.device, t.size(), t.dtype, offset=0)
    vref, bref = weakref.ref(view), weakref.ref(base)
    del view
    gc.collect()
    self.assertIsNone(vref())
    self.assertIsNotNone(bref())
    
  @skip_if_not_cpu
  def test_subbuffer_chain_gc(self):
    Tensor.manual_seed(0); _ = Tensor.randn(1)
    baseline = bufs_allocated()
    base = Buffer("CPU", 4096, Tensor([0.0]).dtype).allocate()
    v1   = Buffer("CPU", 1024, base.dtype, base=base, offset=0)
    v2   = Buffer("CPU", 1024, base.dtype, base=base, offset=2048)
    r_base, r1, r2 = map(weakref.ref, (base, v1, v2))
    del v1
    gc.collect()
    self.assertIsNone(r1())
    self.assertIsNotNone(r_base())
    del v2
    gc.collect()
    self.assertIsNone(r2())
    self.assertIsNotNone(r_base())
    del base
    gc.collect()
    gc.collect()
    self.assertIsNone(r_base())
    self.assertEqual(bufs_allocated() - baseline, 0)

  @skip_if_not_cpu
  def test_thread_buffer_gc(self):
    import threading

    Tensor.manual_seed(0); _ = Tensor.randn(1)
    baseline = bufs_allocated()

    def worker():
      for _ in range(100):
        Buffer("CPU", 2048, Tensor([0.0]).dtype).allocate()

    th = threading.Thread(target=worker)
    th.start(); th.join()
    gc.collect(); gc.collect()

    self.assertEqual(bufs_allocated() - baseline, 0)

  @skip_if_not_cpu
  def test_refcount_helpers(self):
    buf = Buffer("CPU", 1024, Tensor([0.0]).dtype).allocate()
    self.assertEqual(buf.uop_refcount, 0)
    buf.ref(+2)
    self.assertEqual(buf.uop_refcount, 2)
    buf.ref(-2)
    self.assertEqual(buf.uop_refcount, 0)
    r = weakref.ref(buf)
    del buf
    gc.collect()
    gc.collect()
    self.assertIsNone(r())

  @skip_if_not_cpu
  def test_zero_length_buffer_gc(self):
    baseline = bufs_allocated()
    zb = Buffer("CPU", 1, Tensor([0.0]).dtype).allocate()
    ref = weakref.ref(zb)
    del zb
    gc.collect()
    gc.collect()
    self.assertIsNone(ref())
    self.assertEqual(bufs_allocated() - baseline, 0)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
  unittest.main()