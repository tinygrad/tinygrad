#!/usr/bin/env python
import gc, unittest, weakref
from tinygrad.tensor import Tensor
from tinygrad.device import Buffer
from tinygrad.helpers import GlobalCounters

# ---------- helpers ----------------------------------------------------------------
def bufs_allocated() -> int:
    gc.collect()
    return sum(isinstance(o, Buffer) and o.is_allocated() for o in gc.get_objects())

# ---------- tests ------------------------------------------------------------------
class TestGCCpu(unittest.TestCase):

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

    def test_churn(self):
      loops = 300
      base_mem = GlobalCounters.mem_used
      for _ in range(loops):
        Tensor.randn(128, 128)
      gc.collect()
      self.assertLess(GlobalCounters.mem_used - base_mem, 64 * 1024)

    def test_manual_buffer_gc(self):
      import numpy as np
      Tensor.manual_seed(0); _ = Tensor.randn(1)
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
                        

    def test_view_buffer_release(self):
      t = Tensor.randn(256, 256).contiguous()
      base = t._buffer
      view = Buffer(t.device, t.size(), t.dtype, offset=0)
      vref, bref = weakref.ref(view), weakref.ref(base)
      del view; gc.collect()
      self.assertIsNone(vref())
      self.assertIsNotNone(bref())

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()