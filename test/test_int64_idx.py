import unittest
from tinygrad import Tensor, dtypes, Context, Device

class TestInt64Indexing(unittest.TestCase):
  @unittest.skipIf(Device.DEFAULT in ("METAL", "NV"), "memory error in METAL, NV takes too long")
  def test_int64_indexing(self):
    t = Tensor.ones((1 << 31) + 4, dtype=dtypes.int8).contiguous()
    assert t[-1].item() == 1

  @unittest.skipUnless(Device.DEFAULT in ("CLANG"), "CLANG for int64 max range")
  def test_int64_indexing_noopt(self):
    with Context(NOOPT=1):
      t = Tensor.ones((1 << 31) + 4, dtype=dtypes.int8).contiguous()
      assert t[-1].item() == 1

if __name__ == '__main__':
  unittest.main()