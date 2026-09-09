import gc, unittest, weakref
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes
from tinygrad.helpers import Context

class TestBuffer(unittest.TestCase):
  def test_host_view(self):
    b = Buffer("CPU", 4, dtypes.uint32)
    v = b.view(2, dtypes.uint16, 4)
    host = v.host
    host.view(fmt='H')[0] = 0x1234
    self.assertEqual(b.host.view(fmt='H')[2], 0x1234)
    self.assertEqual(v._buf.va_addr, b._buf.va_addr + 4)
    self.assertIs(v.host, host)
    self.assertIs(v.meta, b.meta)

  def test_memoryview_keeps_allocation_alive(self):
    for device in ("CPU", "PYTHON", "NPY"):
      with self.subTest(device=device), Context(LRU=0):
        b = Buffer(device, 8, dtypes.uint8).ensure_allocated()
        b.host[:] = b"abcdefgh"
        v = b.view(4, dtypes.uint8, 2).ensure_allocated()
        mv = v.as_memoryview(force_zero_copy=True)[1:]
        b_ref, v_ref = weakref.ref(b), weakref.ref(v)
        del b, v
        gc.collect()
        self.assertIsNotNone(b_ref())
        self.assertIsNotNone(v_ref())
        self.assertEqual(bytes(mv), b"def")
        del mv
        gc.collect()
        self.assertIsNone(v_ref())
        self.assertIsNone(b_ref())

  def test_mapping(self):
    b = Buffer("CPU", 8, dtypes.uint8, initial_value=b"abcdefgh")
    self.assertIs(b.get_storage("PYTHON")[0][1], b.get_buf("PYTHON"))
    v = b.view(4, dtypes.uint8, 2)
    mapped = v.get_storage("PYTHON")
    self.assertEqual(bytes(mapped[0][0]), b"cdef")
    self.assertIs(mapped[1], v.host)
    self.assertIsNone(mapped[0][1])
    self.assertIs(v.get_storage("PYTHON")[0], mapped[0])

  def test_view_reallocation(self):
    b = Buffer("CPU", 8, dtypes.uint8)
    v = b.view(4, dtypes.uint8, 2)
    old = v.get_storage("PYTHON")[0]
    b.deallocate()
    b.allocate()
    self.assertFalse(v.is_allocated())
    v.host[:] = b"test"
    self.assertIsNot(v.get_storage("PYTHON")[0], old)
    self.assertEqual(bytes(v.get_buf("PYTHON")), b"test")

  def test_cache_owned_storage_only(self):
    for opaque in (None, memoryview(bytearray(8))):
      with self.subTest(imported=opaque is not None), Context(LRU=1):
        b = Buffer("PYTHON", 8, dtypes.uint8, opaque=opaque)
        buf = b._buf
        b.deallocate()
        self.assertEqual(b._buf is buf, opaque is None)

if __name__ == "__main__": unittest.main()
