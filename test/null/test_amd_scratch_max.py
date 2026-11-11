import unittest
from types import SimpleNamespace
from unittest import mock
import tinygrad.runtime.ops_amd as ops_amd
from tinygrad.runtime.autogen import hsa

def _dev(target=(11, 0, 0), cu_cnt=96, se_cnt=6, slots=32):
  d = SimpleNamespace(target=target, cu_cnt=cu_cnt, se_cnt=se_cnt, xccs=1, iface=SimpleNamespace(props={'max_slots_scratch_cu': slots}))
  d.scratch_waves = lambda p: ops_amd.AMDDevice.scratch_waves(d, p)
  return d

def _tmpring(d, psize):
  t = ops_amd.AMDDevice.tmpring_size(d, psize)
  r = hsa.union_COMPUTE_TMPRING_SIZE_GFX11_bitfields.from_buffer_copy(t.to_bytes(4, 'little'))
  return r.WAVES, r.WAVESIZE

def _nbytes(d, psize):
  # (scratch_buffer's size for a queue past the first: no state but the queue's buffer)
  d.queue_scratch, d.device = {}, "AMD"
  with mock.patch.object(ops_amd, "Buffer", lambda dev, nbytes, **kw: SimpleNamespace(nbytes=nbytes)):
    return ops_amd.AMDDevice.scratch_buffer(d, psize, queue=1).nbytes

class TestAMDScratchMax(unittest.TestCase):
  def test_off_unchanged(self):
    d = _dev()
    with mock.patch.object(ops_amd, "AMD_SCRATCH_MAX", 0):
      waves, wavesize = _tmpring(d, 1260)
      self.assertEqual(waves, 96 * 32 // 6)
      self.assertEqual(_nbytes(d, 1260), 1260 * 64 * 32 * 96)
      self.assertGreaterEqual(_nbytes(d, 1260), waves * 6 * wavesize * 256)

  def test_capped_fits(self):
    d = _dev()
    with mock.patch.object(ops_amd, "AMD_SCRATCH_MAX", 64):
      big = _nbytes(d, 1260)
      self.assertLessEqual(big, 64 << 20)
      # every kernel's waves x wave bytes fit in the scratch made for the largest
      for p in [0, 128, 256, 576, 1000, 1260]:
        waves, wavesize = _tmpring(d, p)
        self.assertGreaterEqual(waves, 1)
        self.assertLessEqual(waves * 6 * wavesize * 256, big, p)
      # a small kernel keeps all its waves
      self.assertEqual(_tmpring(d, 128)[0], 96 * 32 // 6)

  def test_cap_below_one_wave(self):
    d = _dev()
    with mock.patch.object(ops_amd, "AMD_SCRATCH_MAX", 1):
      big = _nbytes(d, 100000)
      waves, wavesize = _tmpring(d, 100000)
      self.assertEqual(waves, 1)
      self.assertGreaterEqual(big, waves * 6 * wavesize * 256)

if __name__ == "__main__":
  unittest.main()
