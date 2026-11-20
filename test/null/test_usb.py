import ctypes, unittest
from unittest.mock import patch
from tinygrad.helpers import mv_address
from tinygrad.runtime.autogen import libusb
from tinygrad.runtime.support.usb import USB3, CustomASM24Controller, alloc_cbuffer, STREAM

class TestUSBTransfers(unittest.TestCase):
  def test_bulk_write_memoryview_and_readonly_fallback(self):
    usb = USB3.__new__(USB3)
    usb.handle = None
    usb._bulk_buf, usb._bulk_mv = alloc_cbuffer(32)
    usb._transferred = ctypes.c_int(0)
    calls = []
    def transfer(handle, endpoint, data, length, transferred, timeout):
      calls.append((ctypes.addressof(data), length))
      transferred.value = length
      return 0
    with patch.object(libusb, 'libusb_bulk_transfer', transfer):
      payload = bytearray(b'abcd')
      usb.bulk_write(payload)
      self.assertEqual(calls[-1], (mv_address(memoryview(payload)), 4))
      usb.bulk_write(b'xy')
      self.assertEqual(calls[-1], (ctypes.addressof(usb._bulk_buf), 2))

  def test_pcie_reads_are_bounded_and_ordered(self):
    requests, reads = [], []
    class FakeUSB:
      def bulk_read(self, size, timeout=1000):
        reads.append(size)
        return bytes([len(reads)]) * size
    controller = CustomASM24Controller.__new__(CustomASM24Controller)
    controller.usb = FakeUSB()
    controller._f0_out = lambda fmt, byte_en, address, count, mode: requests.append((fmt,byte_en,address,count,mode))

    result = controller.pcie_mem_read(0x1000,4 * STREAM + 4)

    self.assertEqual(requests,[(0x20,0x0f,0x1000,STREAM,2),(0x20,0x0f,0x1000 + 4 * STREAM,1,2)])
    self.assertEqual(reads,[4 * STREAM,4])
    self.assertEqual(len(result),4 * STREAM + 4)
    self.assertEqual(result[:4],b'\x01' * 4)
    self.assertEqual(result[-4:],b'\x02' * 4)

if __name__ == '__main__': unittest.main()
