import unittest, array, time
from tinygrad.helpers import mv_address
from tinygrad.runtime.support.hcq import MMIOInterface

class TestHCQIface(unittest.TestCase):
  def setUp(self):
    self.size = 4 << 10
    self.buffer = bytearray(self.size)
    self.mv = memoryview(self.buffer).cast('I')
    self.mmio = MMIOInterface(mv_address(self.mv), self.size, fmt='I')

  def test_getitem_setitem(self):
    self.mmio[1] = 0xdeadbeef
    self.assertEqual(self.mmio[1], 0xdeadbeef)
    values = array.array('I', [10, 20, 30, 40])
    self.mmio[2:6] = values
    read_slice = self.mmio[2:6]
    # self.assertIsInstance(read_slice, array.array)
    self.assertEqual(read_slice, values.tolist())
    self.assertEqual(self.mv[2:6].tolist(), values.tolist())

  def test_view(self):
    full = self.mmio.view()
    self.assertEqual(len(full), len(self.mmio))
    self.mmio[0] = 0x12345678
    self.assertEqual(full[0], 0x12345678)

    # offset-only view
    self.mmio[1] = 0xdeadbeef
    off = self.mmio.view(offset=4)
    self.assertEqual(off[0], 0xdeadbeef)

    # offset + size view: write into sub-view and confirm underlying buffer
    values = array.array('I', [11, 22, 33])
    sub = self.mmio.view(offset=8, size=12)
    sub[:] = values
    self.assertEqual(sub[:], values.tolist())
    self.assertEqual(self.mv[2:5].tolist(), values.tolist())

  def test_speed(self):
    start = time.perf_counter()
    for i in range(10000):
      self.mmio[3:100] = array.array('I', [i] * 97)
      _ = self.mmio[3:100]
    end = time.perf_counter()

    mvstart = time.perf_counter()
    for i in range(10000):
      self.mv[3:100] = array.array('I', [i] * 97)
      _ = self.mv[3:100].tolist()
    mvend = time.perf_counter()
    print(f"speed: hcq {end - start:.6f}s vs plain mv {mvend - mvstart:.6f}s")

if __name__ == "__main__":
  unittest.main()
