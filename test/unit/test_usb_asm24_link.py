import struct, sys, time
import pytest

class FakeUSB:
  """minimal stand-in for USB3: LTSSM reaches L0 after ready_after_reads register reads, once the rail is powered"""
  def __init__(self, ready_after_reads:int, rail_mv:int):
    self.ready_after_reads, self.rail_mv, self.reads, self.powered = ready_after_reads, rail_mv, 0, False
  def control_write(self, request:int, value:int=0, index:int=0, data:bytes=b'', timeout:int=1000):
    if request == 0xF3: self.powered = bool(value)
  def control_read(self, request:int, length:int, value:int=0, index:int=0, timeout:int=1000) -> memoryview:
    if request == 0xC0: return memoryview(struct.pack('<Hhb', self.rail_mv if self.powered else 0, 0, 0))
    assert (request, value) == (0xE4, 0xB450)
    self.reads += 1
    return memoryview(bytes([0x78 if self.powered and self.reads > self.ready_after_reads else 0x00]))

@pytest.mark.skipif(sys.platform != "linux", reason="usb backend is linux only")
def test_link_that_trains_after_power_on_is_accepted():
  from tinygrad.runtime.support.usb import CustomASM24Controller
  CustomASM24Controller(FakeUSB(ready_after_reads=3, rail_mv=13000))

@pytest.mark.skipif(sys.platform != "linux", reason="usb backend is linux only")
def test_unpowered_rail_fails_without_waiting():
  from tinygrad.runtime.support.usb import CustomASM24Controller
  st = time.monotonic()
  with pytest.raises(RuntimeError): CustomASM24Controller(FakeUSB(ready_after_reads=1<<30, rail_mv=0))
  assert time.monotonic() - st < 1.0
