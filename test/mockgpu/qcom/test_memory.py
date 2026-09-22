import ctypes, errno, mmap, os, unittest
from unittest.mock import patch
from tinygrad.runtime.autogen import kgsl, libc
from test.mockgpu.qcom.qcomdriver import QCOMDriver, _ioctl_request

class TestKGSLMemory(unittest.TestCase):
  def setUp(self):
    self.driver = QCOMDriver()
    self.fd = self.driver.open('/dev/kgsl-3d0', 0, 0, self.driver.tracked_files[0]).fd
    self.maps: list[tuple[int, int]] = []

  def tearDown(self):
    for addr, size in self.maps: self.assertEqual(libc.munmap(addr, size), 0)
    for fd in list(self.driver.fds): self.driver.close(fd)

  def call(self, op, arg):
    self.driver.ioctl(self.fd, _ioctl_request(op), ctypes.addressof(arg))
    return arg

  def alloc(self, size=4096, flags=kgsl.KGSL_MEMFLAGS_USE_CPU_MAP | (12 << kgsl.KGSL_MEMALIGN_SHIFT)):
    return self.call(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, kgsl.struct_kgsl_gpuobj_alloc(size=size, flags=flags))

  def map(self, obj):
    addr = self.driver.mmap(self.fd, obj.size, 3, mmap.MAP_SHARED, obj.id * 4096)
    self.maps.append((addr, obj.size))
    return addr

  def test_snapshot_and_owner(self):
    obj = self.alloc()
    self.assertEqual(self.driver.buffers(self.fd), {})
    addr = self.map(obj)
    ctypes.c_uint32.from_address(addr).value = 0x12345678
    snapshot = self.driver.buffers(self.fd)
    self.assertEqual(snapshot[addr][:4], bytes.fromhex('78563412'))
    snapshot[addr][:4] = bytes(4)
    self.assertEqual(ctypes.c_uint32.from_address(addr).value, 0x12345678)
    other = self.driver.open('/dev/kgsl-3d0', 0, 0, self.driver.tracked_files[0]).fd
    self.assertEqual(self.driver.buffers(other), {})

  def test_free_and_close_revoke_gpu_access(self):
    obj = self.alloc()
    addr = self.map(obj)
    self.call(kgsl.IOCTL_KGSL_GPUOBJ_FREE, kgsl.struct_kgsl_gpuobj_free(id=obj.id))
    self.assertEqual(self.driver.buffers(self.fd), {})
    self.assertEqual(self.driver.maps, {})
    ctypes.c_uint32.from_address(addr).value = 42
    self.assertEqual(ctypes.c_uint32.from_address(addr).value, 42)
    self.map(self.alloc())
    self.driver.close(self.fd)
    self.assertEqual((self.driver.objects, self.driver.maps), ({}, {}))
    with self.assertRaisesRegex(OSError, 'invalid KGSL fd'): self.driver.buffers(self.fd)

  def test_unmap_and_address_reuse(self):
    obj = self.alloc()
    addr = self.map(obj)
    self.assertEqual(libc.munmap(addr, obj.size), 0)
    self.maps.remove((addr, obj.size))
    self.assertEqual(self.driver.buffers(self.fd), {})
    replacement = libc.mmap(addr, obj.size, 3, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | libc.MAP_FIXED_NOREPLACE, -1, 0)
    self.assertEqual(replacement, addr)
    self.maps.append((addr, obj.size))
    self.assertEqual(self.driver.buffers(self.fd), {})

  def test_partial_unmap(self):
    obj = self.alloc(8192)
    addr = self.map(obj)
    self.assertEqual(libc.munmap(addr+4096, 4096), 0)
    self.maps.remove((addr, 8192))
    self.maps.append((addr, 4096))
    self.assertEqual(self.driver.buffers(self.fd), {})

  def test_alias_and_non_cpu_map(self):
    self.map(self.alloc(flags=0))
    self.assertEqual(self.driver.buffers(self.fd), {})
    obj = self.alloc()
    first, second = self.map(obj), self.map(obj)
    with self.assertRaisesRegex(OSError, 'multiple GPU mappings'): self.driver.buffers(self.fd)
    self.assertEqual(libc.munmap(second, obj.size), 0)
    self.maps.remove((second, obj.size))
    self.assertEqual(set(self.driver.buffers(self.fd)), {first})

  def test_failed_map_and_short_read(self):
    obj = self.alloc()
    with patch('test.mockgpu.qcom.qcomdriver.libc.mmap', return_value=ctypes.c_void_p(-1).value):
      with patch('test.mockgpu.qcom.qcomdriver.ctypes.get_errno', return_value=errno.ENOMEM):
        with self.assertRaises(OSError): self.map(obj)
    self.assertEqual(self.driver.maps, {})
    self.map(obj)
    open_fds = set(os.listdir('/proc/self/fd'))
    with patch('os.pread', return_value=b''):
      with self.assertRaisesRegex(OSError, 'short KGSL backing read'): self.driver.buffers(self.fd)
    self.assertEqual(set(os.listdir('/proc/self/fd')), open_fds)

if __name__ == '__main__': unittest.main()
