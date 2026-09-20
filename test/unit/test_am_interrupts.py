import contextlib, ctypes, functools, io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tinygrad.runtime.autogen.am import am
from tinygrad.runtime.support.am.amdev import AMRegister
from tinygrad.runtime.support.am.ip import AM_IH
from tinygrad.runtime.support.amd import import_asic_regs
from tinygrad.runtime.support.memory import MMIOInterface
from tinygrad.runtime.support.usb import USBMMIOInterface


class RegisterDevice:
  def __init__(self, version):
    self.values = {}
    self.registers = import_asic_regs('osssys', version, cls=functools.partial(AMRegister, adev=self, bases={0: (0,)}))
    self.regIH_RB_RPTR = self.reg('regIH_RB_RPTR')
    self.soc = SimpleNamespace(ih_srcs_names={1: {2: 'CP_EOP_INTR', 3: 'UNEXPECTED'}}, ih_clients={1: 'TEST'})
    self.ip_ver, self.devfmt, self.is_err_state = {am.GC_HWIP: (11, 0, 0)}, 'test', False
    self.is_vf = True  # RAS handling is separate from the interrupt-ring tests.

  def reg(self, name): return self.registers[name]
  def rreg(self, address, **kwargs): return self.values.get(address, 0)
  def wreg(self, address, value, **kwargs): self.values[address] = value


class RingTransport:
  def __init__(self, entries):
    self.storage = (ctypes.c_uint32 * ((256 << 10) // 4))()
    for offset, source in entries:
      self.storage[offset // 4] = 1 | (source << 8)
    self.pending = {offset // 4 + word for offset, _ in entries for word in range(8)}
    self.bulk_buffer = bytearray(256 << 10)
    self.reads = []

  def pcie_mem_read(self, offset, size):
    if not size or any(index not in self.pending for index in range(offset // 4, (offset + size) // 4)):
      raise AssertionError(f'read outside pending interrupt entries: bytes {offset}:{offset + size}')
    self.reads.append((offset, size))
    self.bulk_buffer[:size] = bytes(self.storage)[offset:offset + size]
    # The actual USB transport returns a view of one reusable transfer buffer.
    return memoryview(self.bulk_buffer)[:size]


class TestAMInterruptPointers(unittest.TestCase):
  def make_ih(self, version, rptr, wptr, entries=(), overflow=False, usb=True):
    device = RegisterDevice(version)
    # The raw hardware pointers are byte offsets. AMRegister must encode/decode
    # their generated OFFSET fields when accessing the dword-indexed ring.
    device.regIH_RB_RPTR.write(rptr)
    device.reg('regIH_RB_WPTR').write(wptr | int(overflow))
    ih = AM_IH(device)
    ih.ring_size, ih.rings = 256 << 10, [(0, 0, '', 0)]
    self.transport = RingTransport(entries)
    ih.ring_view = USBMMIOInterface(self.transport, 0, ih.ring_size, 'I') if usb else \
      MMIOInterface(ctypes.addressof(self.transport.storage), ih.ring_size, 'I')
    return ih

  def test_empty_ring_at_nonzero_offset(self):
    for version in ((4, 4, 2), (6, 0, 0), (6, 1, 0), (7, 0, 0)):
      with self.subTest(version=version):
        ih = self.make_ih(version, 32, 32)
        ih.interrupt_handler()
        self.assertEqual(self.transport.reads, [])
        self.assertEqual(ih.adev.regIH_RB_RPTR.read(), 32)

  def test_pending_entries_are_read_once(self):
    for usb in (False, True):
      with self.subTest(usb=usb):
        ih = self.make_ih((6, 0, 0), 32, 96, entries=((32, 2), (64, 2)), usb=usb)
        ih.interrupt_handler()
        self.assertEqual(self.transport.reads, [(32, 64)] if usb else [])
        self.assertEqual(ih.adev.regIH_RB_RPTR.read(), 96)
        ih.interrupt_handler()
        self.assertEqual(self.transport.reads, [(32, 64)] if usb else [])

  def test_pending_entries_wrap_at_end_of_ring(self):
    end = 256 << 10
    for usb in (False, True):
      with self.subTest(usb=usb):
        ih = self.make_ih((6, 0, 0), end - 32, 32, entries=((end - 32, 3), (0, 2)), usb=usb)
        with contextlib.redirect_stdout(output := io.StringIO()): ih.interrupt_handler()
        self.assertEqual(self.transport.reads, [(end - 32, 32), (0, 32)] if usb else [])
        self.assertEqual(ih.adev.regIH_RB_RPTR.read(), 32)
        self.assertTrue(ih.adev.is_err_state)
        self.assertEqual(output.getvalue().count('src=UNEXPECTED(3)'), 1)

  def test_large_completion_backlog_uses_one_transfer(self):
    ih = self.make_ih((6, 0, 0), 0, 6215 * 32, entries=tuple((offset * 32, 2) for offset in range(6215)))
    ih.interrupt_handler()
    self.assertEqual(self.transport.reads, [(0, 6215 * 32)])
    self.assertEqual(ih.adev.regIH_RB_RPTR.read(), 6215 * 32)
    self.assertFalse(ih.adev.is_err_state)

  def test_wrap_to_zero_does_not_read_an_empty_second_range(self):
    end = 256 << 10
    ih = self.make_ih((6, 0, 0), end - 32, 0, entries=((end - 32, 2),))
    ih.interrupt_handler()
    self.assertEqual(self.transport.reads, [(end - 32, 32)])
    self.assertEqual(ih.adev.regIH_RB_RPTR.read(), 0)

  def test_fault_arriving_during_snapshot_is_left_pending(self):
    ih = self.make_ih((6, 0, 0), 0, 32, entries=((0, 2), (32, 3)))
    read = self.transport.pcie_mem_read

    def read_with_new_interrupt(offset, size):
      data = read(offset, size)
      ih.adev.reg('regIH_RB_WPTR').write(64)
      return data

    with patch.object(self.transport, 'pcie_mem_read', side_effect=read_with_new_interrupt):
      ih.interrupt_handler()
    self.assertEqual(ih.adev.regIH_RB_RPTR.read(), 32)
    self.assertFalse(ih.adev.is_err_state)
    with contextlib.redirect_stdout(output := io.StringIO()): ih.interrupt_handler()
    self.assertEqual(self.transport.reads, [(0, 32), (32, 32)])
    self.assertEqual(ih.adev.regIH_RB_RPTR.read(), 64)
    self.assertTrue(ih.adev.is_err_state)
    self.assertEqual(output.getvalue().count('src=UNEXPECTED(3)'), 1)

  def test_drain_tracks_hardware_write_pointer(self):
    ih = self.make_ih((6, 0, 0), 0, 64)
    for pointer in (64, 96, 0, 32):
      with self.subTest(pointer=pointer):
        ih.adev.reg('regIH_RB_WPTR').write(pointer)
        ih.drain()
        self.assertEqual(ih.adev.regIH_RB_RPTR.read(), pointer)
        ih.interrupt_handler()
    self.assertEqual(self.transport.reads, [])

  def test_drain_clears_overflow_without_changing_pointer_units(self):
    ih = self.make_ih((6, 0, 0), 0, 96, overflow=True)
    ih.drain()
    self.assertEqual(ih.adev.regIH_RB_RPTR.read(), 96)
    self.assertEqual(ih.adev.reg('regIH_RB_WPTR').read(), 96)
    self.assertEqual(ih.adev.reg('regIH_RB_CNTL').read_bitfields()['wptr_overflow_clear'], 0)


if __name__ == '__main__':
  unittest.main()
