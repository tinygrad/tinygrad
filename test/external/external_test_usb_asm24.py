import unittest, time
from tinygrad.runtime.support.usb import ASM24Controller

class TestASMController(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.ctrl = ASM24Controller()

  def test_write_and_read(self):
    base = 0xF000
    data = b"hello!"
    self.ctrl.write(base, data)
    out = self.ctrl.read(base, len(data))
    self.assertEqual(out, data)

  def test_scsi_write_and_read_from_f000(self):
    payload = bytes([0x5B]) * 4096
    self.ctrl.scsi_write(payload, lba=0)
    back = self.ctrl.read(0xF000, len(payload))
    self.assertEqual(back, payload)

  def test_scsi_write_speed_4k(self):
    payload = bytes([0x5A]) * 4096
    start = time.perf_counter()
    self.ctrl.scsi_write(payload, lba=0)
    dur_ms = (time.perf_counter() - start) * 1000
    print(f"scsi_write 4K took {dur_ms:.3f} ms")

  def test_read_speed_4k(self):
    payload = bytes([0xA5]) * 4096
    self.ctrl.write(0xF000, payload)
    start = time.perf_counter()
    out = self.ctrl.read(0xF000, 4096)
    dur_ms = (time.perf_counter() - start) * 1000
    print(f"read 4K took {dur_ms:.3f} ms")
    self.assertEqual(out, payload)

if __name__ == "__main__":
  unittest.main()