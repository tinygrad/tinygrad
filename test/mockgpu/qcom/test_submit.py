import ctypes, errno, mmap, os, unittest
from unittest.mock import patch
from tinygrad.runtime.autogen import kgsl, libc
from test.mockgpu.qcom.qcomdriver import QCOMDriver, _ioctl_request
from test.mockgpu.qcom.test_compute import packet

class TestKGSLSubmit(unittest.TestCase):
  def setUp(self):
    self.driver = QCOMDriver()
    self.fd = self.driver.open('/dev/kgsl-3d0', 0, 0, self.driver.tracked_files[0]).fd
    self.ctx = self.call(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, kgsl.struct_kgsl_drawctxt_create()).drawctxt_id
    flags = kgsl.KGSL_MEMFLAGS_USE_CPU_MAP | (12 << kgsl.KGSL_MEMALIGN_SHIFT)
    self.obj = self.call(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, kgsl.struct_kgsl_gpuobj_alloc(size=4096, flags=flags))
    self.addr = self.driver.mmap(self.fd, 4096, 3, mmap.MAP_SHARED, self.obj.id*4096)

  def tearDown(self):
    self.assertEqual(libc.munmap(self.addr, 4096), 0)
    for fd in list(self.driver.fds): self.driver.close(fd)

  def call(self, op, arg):
    self.assertEqual(self.driver.ioctl(self.fd, _ioctl_request(op), ctypes.addressof(arg)), 0)
    return arg

  def submit(self, raw):
    ctypes.memmove(self.addr, raw, len(raw))
    ib = kgsl.struct_kgsl_command_object(gpuaddr=self.addr, size=len(raw), flags=kgsl.KGSL_CMDLIST_IB)
    cmd = kgsl.struct_kgsl_gpu_command(context_id=self.ctx, cmdlist=ctypes.addressof(ib), cmdsize=32, numcmds=1, timestamp=99)
    return self.call(kgsl.IOCTL_KGSL_GPU_COMMAND, cmd).timestamp

  def event(self, value): return packet(0x46, 4, (self.addr+256) & 0xffffffff, (self.addr+256) >> 32, value)

  def test_visibility_and_timestamps(self):
    for expected in (1, 2):
      self.assertEqual(self.submit(self.event(expected)), expected)
      self.assertEqual(ctypes.c_uint32.from_address(self.addr+256).value, expected)
      for kind in (1, 2, 3):
        ts = self.call(kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID,
                       kgsl.struct_kgsl_cmdstream_readtimestamp_ctxtid(context_id=self.ctx, type=kind))
        self.assertEqual(ts.timestamp, expected)
      self.call(kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID,
                kgsl.struct_kgsl_device_waittimestamp_ctxtid(context_id=self.ctx, timestamp=expected, timeout=0))

  def test_failure_does_not_publish(self):
    self.submit(self.event(7))
    with self.assertRaisesRegex(OSError, 'unsupported packet'): self.submit(self.event(9) + packet(0x10))
    self.assertEqual(ctypes.c_uint32.from_address(self.addr+256).value, 7)
    self.assertEqual(self.driver.timestamps[self.ctx], 1)
    self.assertEqual(self.driver.gpus[self.ctx].flushes, 1)

  def test_short_write_rolls_back(self):
    self.submit(self.event(7))
    pwrite, count = os.pwrite, 0
    def short_write(fd, data, off):
      nonlocal count
      count += 1
      return pwrite(fd, data[:300] if count == 1 else data, off)
    with patch('os.pwrite', side_effect=short_write):
      with self.assertRaisesRegex(OSError, 'short KGSL write'): self.submit(self.event(9))
    self.assertEqual(count, 2)
    self.assertEqual(ctypes.c_uint32.from_address(self.addr+256).value, 7)
    self.assertEqual(self.driver.timestamps[self.ctx], 1)

  def test_bad_timestamp_and_ownership(self):
    req = kgsl.struct_kgsl_device_waittimestamp_ctxtid(context_id=self.ctx, timestamp=1, timeout=1)
    with self.assertRaisesRegex(OSError, 'not completed') as error: self.call(kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID, req)
    self.assertEqual(error.exception.errno, errno.ETIMEDOUT)
    ts = kgsl.struct_kgsl_cmdstream_readtimestamp_ctxtid(context_id=self.ctx, type=0, timestamp=99)
    with self.assertRaisesRegex(OSError, 'timestamp type 0'): self.call(kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID, ts)
    self.assertEqual(ts.timestamp, 99)
    other = self.driver.open('/dev/kgsl-3d0', 0, 0, self.driver.tracked_files[0]).fd
    for op, arg in ((kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID, ts), (kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID, req)):
      with self.assertRaisesRegex(OSError, 'not owned'):
        self.driver.ioctl(other, _ioctl_request(op), ctypes.addressof(arg))
    self.call(kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY, kgsl.struct_kgsl_drawctxt_destroy(drawctxt_id=self.ctx))
    self.assertEqual((self.driver.gpus, self.driver.timestamps), ({}, {}))
    with self.assertRaisesRegex(OSError, 'not owned'): self.call(kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID, req)

  def test_write_failure_restores_all_attempted_objects(self):
    regions = {0x1000: {'backing_fd': 1}, 0x2000: {'backing_fd': 2}}
    before = {0x1000: bytearray(b'old1'), 0x2000: bytearray(b'old2')}
    after = {0x1000: bytearray(b'new1'), 0x2000: bytearray(b'new2')}
    for rollback_error in (False, True):
      with self.subTest(rollback_error=rollback_error):
        responses = [4, OSError(errno.EIO, 'write failed'), OSError(errno.EIO, 'restore failed') if rollback_error else 4, 4]
        with patch('os.pwrite', side_effect=responses) as write:
          with self.assertRaisesRegex(OSError, 'KGSL rollback failed at 0x1000' if rollback_error else 'write failed'):
            self.driver.publish(regions, before, after)
          self.assertEqual(write.call_count, 4)
          self.assertEqual(write.call_args_list[-2].args, (1, before[0x1000], 0))
          self.assertEqual(write.call_args_list[-1].args, (2, before[0x2000], 0))
          self.assertEqual(self.driver.last_submission_stats['restored_bytes'], 4 if rollback_error else 8)

  def test_timestamp_overflow_and_unmapped_ib(self):
    self.driver.timestamps[self.ctx] = 0xffffffff
    with self.assertRaisesRegex(OSError, 'timestamp wrap'): self.submit(packet(0x26))
    self.driver.timestamps[self.ctx] = 0
    self.call(kgsl.IOCTL_KGSL_GPUOBJ_FREE, kgsl.struct_kgsl_gpuobj_free(id=self.obj.id))
    with self.assertRaisesRegex(OSError, 'KGSL command.*backing buffers'): self.submit(packet(0x26))
    self.assertEqual(self.driver.timestamps[self.ctx], 0)

if __name__ == '__main__': unittest.main()
