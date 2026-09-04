import struct, unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
from tinygrad.runtime.autogen.am import sdma_6_0_0 as sdma
from tinygrad.runtime.ops_amd import AMDAllocator
from tinygrad.runtime.support.hcq import HCQBuffer
from tinygrad.runtime.support.usb import alloc_cbuffer

class CopyinUSB:
  def __init__(self, delayed_bulk=False, delayed_drain=False, delayed_read=False):
    self.usb = self
    self.delayed_bulk, self.delayed_drain = delayed_bulk, delayed_drain
    self.delayed_read = delayed_read
    self.fence, self.seq, self.slot, self.reads = 0, 0, 0, 0
    self.pending, self.occupied, self.landed = {}, set(), []
    self.pending_reads = {}
    self.read_tag = -2

  def control_write_async(self, request, value=0, index=0):
    assert request == 0xF2
    assert not self.pending, "F2 reconfigured with a bulk transfer still pending"
    assert not self.pending_reads, "F2 armed before its drain fence read completed"
    self.slot = index & 0xFF
    assert self.slot not in self.occupied, "F2 recycled a window before SDMA drained it"
    return -1

  def bulk_write_async(self, buf):
    tag = self.seq
    self.seq += 1
    self.pending[tag] = (self.slot, buf, bytes(buf))
    if not self.delayed_bulk: self.bulk_wait(tag)
    return tag

  def bulk_wait(self, tag):
    if tag in self.pending_reads:
      buf, value, length = self.pending_reads.pop(tag)
      buf[:] = self.read(value, length)
      return
    if tag not in self.pending: return
    slot, buf, expected = self.pending.pop(tag)
    assert bytes(buf) == expected, "staging buffer modified while USB still owns it"
    assert struct.unpack_from('<I', buf, len(buf)-4)[0] == 0x51000000 | tag
    self.occupied.add(slot)
    self.landed.append((tag, slot))
    if not self.delayed_drain: self.drain()

  def drain(self):
    if self.landed:
      seq, slot = self.landed.pop(0)
      self.occupied.remove(slot)
      self.fence = seq + 1

  def read(self, addr, size):
    assert (addr, size) == (0xA800, 8)
    self.reads += 1
    if self.reads % 3 == 0: self.drain()  # GPU progress is independent of USB completion.
    return self.fence.to_bytes(8, 'little')

  def control_read_async(self, request, length, value=0):
    assert request == 0xE4
    if self.delayed_read:
      tag = self.read_tag
      self.read_tag -= 1
      buf = bytearray(length)
      self.pending_reads[tag] = (buf, value, length)
      return tag, memoryview(buf)
    return -1, memoryview(self.read(value, length))

class TestAMDUSBCopyin(unittest.TestCase):
  def check_copyin(self, usb):
    q = MagicMock()
    for method in ('wait', 'copy', 'write', 'signal'): getattr(q, method).return_value = q
    allocator = AMDAllocator.__new__(AMDAllocator)
    allocator.dev = SimpleNamespace(is_usb=lambda: True, iface=SimpleNamespace(pci_dev=SimpleNamespace(usb=usb), sys_buf=HCQBuffer(0, 0x1000)),
      timeline_signal=None, timeline_value=1, next_timeline=lambda: 1, sdma=sdma, hw_copy_queue_t=lambda: q)
    allocator._usb_seq = 0
    allocator._usb_stage = [alloc_cbuffer(0x40000) for _ in range(2)]
    allocator._usb_wins = (HCQBuffer(0, 0x40000), HCQBuffer(0x40000, 0x40000))
    # Include an odd number of chunks and successive calls starting in either window.
    for size in (3 * (0x40000 - 4), 0x40000, 1):
      allocator._copyin(HCQBuffer(0x100000, size), memoryview(bytearray(size)))
      self.assertEqual(usb.fence, allocator._usb_seq)
      self.assertFalse(usb.pending)
      self.assertFalse(usb.pending_reads)
      self.assertFalse(usb.occupied)

  def test_delayed_drain(self): self.check_copyin(CopyinUSB(delayed_drain=True))
  def test_delayed_bulk(self): self.check_copyin(CopyinUSB(delayed_bulk=True))
  def test_delayed_bulk_and_drain(self): self.check_copyin(CopyinUSB(delayed_bulk=True, delayed_drain=True))
  def test_delayed_fence_read(self): self.check_copyin(CopyinUSB(delayed_bulk=True, delayed_drain=True, delayed_read=True))

if __name__ == '__main__': unittest.main()
