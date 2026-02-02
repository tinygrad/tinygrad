import unittest
from tinygrad import dtypes, Device
from tinygrad.device import is_dtype_supported
from tinygrad.helpers import Context
from tinygrad.runtime.ops_null import NullDevice

@unittest.skipUnless(Device.DEFAULT=="NULL", "Don't run when testing non-NULL backends")
class TestNULLSupportsDTypes(unittest.TestCase):
  def test_null_supports_ints_floats_bool(self):
    dts = dtypes.ints + dtypes.floats + (dtypes.bool,)
    not_supported = [dt for dt in dts if not is_dtype_supported(dt, "NULL")]
    self.assertFalse(not_supported, msg=f"expected these dtypes to be supported by NULL: {not_supported}")

class TestNULLVRAM(unittest.TestCase):
  @Context(NULL_VRAM_SIZE=1 << 20) # 1 MB VRAM limit
  def test_oom(self):
    allocator = NullDevice("NULL").allocator
    buf1 = allocator.alloc(512 << 10)  # 512 KB
    self.assertIsNotNone(buf1)
    with self.assertRaises(MemoryError):
      allocator.alloc(1 << 20)  # 1 MB

  @Context(NULL_VRAM_SIZE=1 << 20) # 1 MB VRAM limit
  def test_allow_realloc(self):
    allocator = NullDevice("NULL").allocator
    buf1 = allocator.alloc(900 << 10)  # 900 KB
    self.assertIsNotNone(buf1)
    allocator.free(buf1, 900 << 10)
    buf2 = allocator.alloc(900 << 10)  # 900 KB
    self.assertIsNotNone(buf2)

  # by default null device has infinite memory
  @Context(NULL_VRAM_SIZE=-1)
  def test_unlimited_default(self):
    allocator = NullDevice("NULL").allocator
    buf = allocator.alloc(1 << 30)  # 1 GB
    self.assertIsNone(buf) # null alloc returns nothing

if __name__ == "__main__":
  unittest.main()
