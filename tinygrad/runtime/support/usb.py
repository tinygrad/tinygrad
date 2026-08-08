from __future__ import annotations
import ctypes, struct, time, functools, itertools
from tinygrad.runtime.autogen import libusb
from tinygrad.helpers import DEBUG, DEV, to_mv, round_up, ceildiv
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support import c

def alloc_cbuffer(sz:int) -> tuple[ctypes.Array, memoryview]: return (buf:=(ctypes.c_ubyte * sz)()), to_mv(ctypes.addressof(buf), sz)
def checked(fn, msg=None):
  @functools.wraps(fn)
  def wrapper(*args):
    if (rc:=fn(*args)) < 0: raise RuntimeError(f"{msg or fn.__name__}: {ctypes.string_at(libusb.libusb_strerror(rc)).decode()}")
    return rc
  return wrapper

class USB3:
  @staticmethod
  @functools.cache
  def ctx():
    ctx = c.init_c_var(ctypes.POINTER(libusb.struct_libusb_context), checked(libusb.libusb_init))
    if DEBUG >= 6: checked(libusb.libusb_set_option)(ctx, libusb.LIBUSB_OPTION_LOG_LEVEL, 4)
    return ctx

  @classmethod
  @functools.cache
  def list_devices(cls, vendor:int, dev:int) -> list[tuple[c.POINTER[libusb.struct_libusb_device], str]]:
    ret = []
    for i in range(checked(libusb.libusb_get_device_list)(cls.ctx(), devs:=ctypes.POINTER(ctypes.POINTER(libusb.struct_libusb_device))())):
      desc = c.init_c_var(libusb.struct_libusb_device_descriptor, lambda x: checked(libusb.libusb_get_device_descriptor)(devs[i], x))
      if (desc.idVendor, desc.idProduct) == (vendor, dev):
        ret.append((libusb.libusb_ref_device(devs[i]), f"usb:{libusb.libusb_get_bus_number(devs[i])}-{libusb.libusb_get_device_address(devs[i])}"))
    libusb.libusb_free_device_list(devs, 1)
    return ret

  def __init__(self, dev:c.POINTER[libusb.struct_libusb_device], *args, **kwargs):
    self._tags, self._transferred = itertools.count(1), ctypes.c_int(0)
    self._bulk_buf, self._bulk_mv = alloc_cbuffer(4 << 20)
    self._ctrl_buf, self._ctrl_mv = alloc_cbuffer(0x1000)

    self.handle = c.init_c_var(c.POINTER[libusb.struct_libusb_device_handle], lambda x: checked(libusb.libusb_open)(dev, x))

    # Read product string descriptor
    _buf = (ctypes.c_ubyte * 256)()
    _desc = libusb.struct_libusb_device_descriptor()
    checked(libusb.libusb_get_device_descriptor)(libusb.libusb_get_device(self.handle), ctypes.byref(_desc))
    _ret = checked(libusb.libusb_get_string_descriptor_ascii)(self.handle, _desc.iProduct, _buf, 256)
    self.product = bytes(_buf[:_ret]).decode("ascii", errors="replace")
    assert self.product.startswith("custom") or self.product.startswith("AS2462")

    # Detach kernel driver if needed
    if checked(libusb.libusb_kernel_driver_active)(self.handle, 0):
      checked(libusb.libusb_detach_kernel_driver)(self.handle, 0)
      checked(libusb.libusb_reset_device)(self.handle)

    # Set configuration and claim interface
    checked(libusb.libusb_set_configuration)(self.handle, 1)
    checked(libusb.libusb_claim_interface)(self.handle, 0)
    checked(libusb.libusb_set_interface_alt_setting)(self.handle, 0, 0)

  def control_write(self, request:int, value:int=0, index:int=0, data:bytes=b'', timeout:int=1000):
    assert len(data) <= len(self._ctrl_mv)
    self._ctrl_mv[:len(data)] = data
    assert checked(libusb.libusb_control_transfer)(self.handle, 0x40, request, value, index, self._ctrl_buf, len(data), timeout) == len(data)

  def control_read(self, request:int, length:int, value:int=0, index:int=0, timeout:int=1000) -> memoryview:
    assert length <= len(self._ctrl_mv)
    assert checked(libusb.libusb_control_transfer)(self.handle, 0xC0, request, value, index, self._ctrl_buf, length, timeout) == length
    return self._ctrl_mv[:length]

  def bulk_write(self, payload:bytes|memoryview, timeout:int=1000):
    if len(payload) > len(self._bulk_mv): self._bulk_buf, self._bulk_mv = alloc_cbuffer(len(payload))
    self._bulk_mv[:len(payload)] = payload
    checked(libusb.libusb_bulk_transfer, "bulk OUT 0x02 failed") \
      (self.handle, 0x02, self._bulk_buf, len(payload), self._transferred, timeout)
    assert self._transferred.value == len(payload), f"bulk OUT short write: {self._transferred.value}/{len(payload)} bytes"

  def bulk_read(self, length:int, timeout:int=1000) -> memoryview:
    if length > len(self._bulk_mv): self._bulk_buf, self._bulk_mv = alloc_cbuffer(length)
    checked(libusb.libusb_bulk_transfer, "bulk IN 0x81 failed")(self.handle, 0x81, self._bulk_buf, length, self._transferred, timeout)
    return self._bulk_mv[:self._transferred.value]

  # NOTE: keep it for flash.py
  def send_batch(self, cdbs:list[bytes], odata:list[bytes|None]|None=None):
    for cdb, data in zip(cdbs, odata or [None] * len(cdbs)):
      self.bulk_write(struct.pack("<IIIBBB16s", 0x43425355, tag:=next(self._tags), len(data) if data is not None else 0, 0, 0, len(cdb), cdb))
      if data is not None: self.bulk_write(data)
      sig, rtag, _, status = struct.unpack("<IIIB", self.bulk_read(13, timeout=2000))
      assert (sig, rtag, status) == (0x53425355, tag, 0)

class CustomASM24Controller:
  PCIE_BULK_CHUNK_SIZE = 1 << 20
  GSP_RING_PAGE, GSP_RING_PAGES, GSP_STREAM_BATCH_PAGES = 44, 84, 28
  GSP_STREAM_FIRST_WRITE, GSP_STREAM_PERIOD = 0.003, 0.0014

  def __init__(self, usb:USB3):
    self.usb = usb

    # Custom firmware now boots with PCIe off. Power it on before probing the link.
    ltssm = self.read(0xB450, 1)[0]
    if ltssm != 0x78: self.set_pcie_power(True)
    ltssm = self.read(0xB450, 1)[0]
    if ltssm != 0x78: raise RuntimeError(f"PCIe link not up (LTSSM=0x{ltssm:02X}), custom firmware not ready")

  def set_pcie_power(self, enabled:bool, timeout:int=10000): self.usb.control_write(0xF3, value=int(enabled), timeout=timeout)

  def _f0_out(self, fmt_type:int, byte_en:int, address:int, value:int, mode:int=0):
    self.usb.control_write(0xF0, fmt_type | (byte_en << 8), mode & 0x03, struct.pack('<III', address & 0xFFFFFFFF, address >> 32, value), 5000)

  def _f0_in(self) -> tuple[int, int, int]:
    data = self.usb.control_read(0xF0, 8, timeout=5000)
    return struct.unpack_from('<I', data)[0], (data[4] >> 5) & 0x7, data[7]

  def pcie_request(self, fmt_type:int, address:int, value:int|None=None, size:int=4, cnt:int=10):
    assert size > 0 and size <= 4, f"Invalid size {size}"
    if DEBUG >= 5: print("pcie_request", hex(fmt_type), hex(address), value, size)

    offset = address & 0x3
    byte_en = ((1 << size) - 1) << offset
    self._f0_out(fmt_type, byte_en, address & ~0x3, (value << (8 * offset)) if value is not None else 0)

    # Fast path: memory writes and messages don't return completions.
    if ((fmt_type & 0b11011111) == 0b01000000) or ((fmt_type & 0b10111000) == 0b00110000): return

    # Read TLPs and config writes: read completion via 0xF0 IN. Retry on error/timeout.
    data, cpl_status, ret_status = self._f0_in()
    if ret_status != 0:
      time.sleep(0.001)  # TODO: this sleep is very picky
      if cnt > 0: return self.pcie_request(fmt_type, address, value, size, cnt=cnt-1)
      raise RuntimeError(f"TLP error after retries: ret_status={ret_status}, address={address:#x}")

    if cpl_status:
      status_map = {0b001: f"Unsupported Request: {address:#x}", 0b100: "Completer Abort", 0b010: "Config Retry"}
      raise RuntimeError(f"TLP completion status: {status_map.get(cpl_status, f'Reserved (0b{cpl_status:03b})')}")

    if value is None: return (data >> (8 * offset)) & ((1 << (8 * size)) - 1)

  def pcie_cfg_req(self, byte_addr:int, bus:int=1, dev:int=0, fn:int=0, value:int|None=None, size:int=4):
    assert byte_addr >> 12 == 0 and bus >> 8 == 0 and dev >> 5 == 0 and fn >> 3 == 0
    fmt_type = (0x44 if value is not None else 0x4) | int(bus > 0)
    address = (bus << 24) | (dev << 19) | (fn << 16) | (byte_addr & 0xfff)
    return self.pcie_request(fmt_type, address, value, size)

  def pcie_mem_write(self, address:int, data:bytes):
    """Streaming PCIe memory write via 0xF0 mode 1 + bulk OUT. Data is little-endian dwords on the wire."""
    if not data: return
    assert len(data) % 4 == 0, f"pcie_mem_write requires 4-byte aligned size, got {len(data)}"
    if len(data) > self.PCIE_BULK_CHUNK_SIZE:
      for off in range(0, len(data), self.PCIE_BULK_CHUNK_SIZE):
        self.pcie_mem_write(address + off, data[off:off+self.PCIE_BULK_CHUNK_SIZE])
      return
    fmt_type = 0x60 if address >> 32 else 0x40
    self._f0_out(fmt_type, 0x0F, address, len(data) // 4, mode=1)
    self.usb.bulk_write(data)

  def pcie_mem_read(self, address:int, nbytes:int) -> memoryview:
    """Streaming PCIe memory read via 0xF0 mode 2 + bulk IN. Returns little-endian bytes."""
    assert nbytes % 4 == 0, f"pcie_mem_read requires 4-byte aligned size, got {nbytes}"
    assert address >= 0 and (address >> 32 or address + nbytes <= (1 << 32)), "PCIe transfer crosses the 32-bit address boundary"
    if nbytes > self.PCIE_BULK_CHUNK_SIZE:
      return memoryview(b''.join(bytes(self.pcie_mem_read(address + off, min(self.PCIE_BULK_CHUNK_SIZE, nbytes - off)))
                                 for off in range(0, nbytes, self.PCIE_BULK_CHUNK_SIZE)))
    fmt_type = 0x20 if address >> 32 else 0x00
    if nbytes == 4: return memoryview(struct.pack("<I", self.pcie_request(fmt_type, address)))
    self._f0_out(fmt_type, 0x0F, address, nbytes // 4, mode=2)
    return self.usb.bulk_read(nbytes, timeout=30000)

  def read(self, base_addr:int, length:int) -> bytes:
    """Read from chip XDATA via vendor control IN (bRequest=0xE4). wValue=addr, wLength=size."""
    result = b''
    for off in range(0, length, 0xFF):
      chunk = min(0xFF, length - off)
      result += self.usb.control_read(0xE4, chunk, value=base_addr + off)
    return result

  def write(self, base_addr:int, data:bytes):
    """Write to chip XDATA via vendor control OUT (bRequest=0xE5). wValue=addr, wIndex=val."""
    for off, val in enumerate(data): self.usb.control_write(0xE5, value=base_addr + off, index=val)

  def scsi_write_arm(self, size:int, start_slot:int=0):
    """Arm one bulk OUT transfer to an SRAM slot range."""
    padded_size = round_up(size, 512)
    sectors, num_slots = padded_size // 512, ceildiv(padded_size, 0x4000)
    assert 0 < sectors < 0x8000, f"invalid F2 sector count {sectors:#x}"
    assert 0 <= start_slot < 32 and start_slot + num_slots <= 32, f"SRAM slot range {start_slot}:{start_slot+num_slots} is out of bounds"
    windex = (start_slot & 0xFF) | ((num_slots & 0xFF) << 8)
    self.usb.control_write(0xF2, value=sectors, index=windex)

  def scsi_write(self, buf:bytes|memoryview, start_slot:int=0):
    """Write to SRAM via 0xF2 vendor command + bulk OUT."""
    padded_size = round_up(len(buf), 512)
    buf_padded = buf if len(buf) == padded_size else bytes(buf) + bytes(padded_size - len(buf))
    self.scsi_write_arm(len(buf_padded), start_slot)
    self.usb.bulk_write(buf_padded)

  def sram_stream_start(self, count:int, size:int, start_slot:int):
    """Arm equal whole-slot SRAM writes; firmware rotates the destination after each bulk completion."""
    assert 0 < count < 1 << 16 and size % 0x4000 == 0
    num_slots = size // 0x4000
    assert 0 < num_slots < 0x100 and 0 <= start_slot < 32 and start_slot + num_slots <= 32
    self.usb.control_write(0xF5, value=count, index=start_slot | (num_slots << 8))

  @classmethod
  def gsp_stream_chunks(cls, image:bytes|memoryview):
    ring_size, batch_size = cls.GSP_RING_PAGES * 0x1000, cls.GSP_STREAM_BATCH_PAGES * 0x1000
    assert cls.GSP_RING_PAGE % 4 == 0 and cls.GSP_RING_PAGES % cls.GSP_STREAM_BATCH_PAGES == 0
    for off in range(ring_size, len(image), batch_size):
      chunk = bytes(image[off:off+batch_size])
      yield (cls.GSP_STREAM_FIRST_WRITE + (off-ring_size) / ring_size * cls.GSP_STREAM_PERIOD,
             cls.GSP_RING_PAGE // 4 + (off % ring_size) // 0x4000, chunk.ljust(batch_size, b'\x00'))

  def stream_gsp_image(self, image:bytes|memoryview, launched_at:float):
    """Keep the SEC2-visible SRAM ring populated while it verifies the GSP image."""
    ring_size, batch_size = self.GSP_RING_PAGES * 0x1000, self.GSP_STREAM_BATCH_PAGES * 0x1000
    chunk_count = ceildiv(max(0, len(image) - ring_size), batch_size)
    for i, (delay, start_slot, payload) in enumerate(self.gsp_stream_chunks(image)):
      deadline = launched_at + delay
      while time.perf_counter() < deadline: pass
      if i == 0: self.sram_stream_start(chunk_count, len(payload), start_slot)
      self.usb.bulk_write(payload)

  def scsi_read_arm(self, size:int, start_slot:int=0):
    padded_size, num_slots = round_up(size, 512), ceildiv(size, 0x4000)
    assert 0 < padded_size // 512 < 0x8000, f"invalid F2 sector count {padded_size // 512:#x}"
    assert 0 <= start_slot < 32 and start_slot + num_slots <= 32, \
      f"SRAM slot range {start_slot}:{start_slot+num_slots} is out of bounds"
    windex = (start_slot & 0xFF) | ((num_slots & 0xFF) << 8)
    self.usb.control_write(0xF2, value=(ceildiv(size, 512) & 0x7FFF) | 0x8000, index=windex)

  def sram_read(self, offset:int, size:int) -> bytes:
    assert 0 <= offset <= 0x80000 and 0 <= size <= 0x80000 - offset, f"SRAM read {offset:#x}+{size:#x} is out of bounds"
    if size == 0: return b''
    slot, slot_offset = divmod(offset, 0x4000)
    assert slot_offset + size <= 0x1000, "F6 reads must fit within the first 4 KiB page of one SRAM slot"
    first_sector, last_sector = slot_offset // 512, ceildiv(slot_offset + size, 512)
    self.usb.control_write(0xF6, value=slot, index=first_sector | ((last_sector - first_sector) << 8))
    expected_size = (last_sector - first_sector) * 512
    data = bytes(self.usb.bulk_read(expected_size))
    if len(data) != expected_size: raise RuntimeError(f"SRAM slot {slot:#x} short read: {len(data)}/{expected_size} bytes")
    start = slot_offset - first_sector * 512
    return data[start:start+size]

  def scsi_read(self, size:int) -> memoryview: return self.usb.bulk_read(round_up(size, 512), timeout=10000)[:size]

class USBMMIOInterface(MMIOInterface):
  def __init__(self, usb, addr, size, fmt, pcimem=True, sram_start_slot:int|None=None): # pylint: disable=super-init-not-called
    self.usb, self.addr, self.nbytes, self.fmt, self.el_sz = usb, addr, size, fmt, struct.calcsize(fmt)
    self.pcimem, self.sram_start_slot = pcimem, sram_start_slot

  def _off_from_index(self, index):
    if isinstance(index, slice):
      start, stop, step = index.indices(len(self))
      if step != 1: raise IndexError("USB MMIO slices require a unit stride")
      return (start * self.el_sz, (stop - start) * self.el_sz)
    if index < 0: index += len(self)
    if not 0 <= index < len(self): raise IndexError(index)
    return (index * self.el_sz, self.el_sz)

  def __getitem__(self, index):
    off, sz = self._off_from_index(index)
    if self.pcimem:
      if sz == 0: data = memoryview(b"")
      else:
        start, end = self.addr + off, self.addr + off + sz
        aligned_start, aligned_end = start & ~0x3, round_up(end, 4)
        data = self.usb.pcie_mem_read(aligned_start, aligned_end - aligned_start)[start-aligned_start:end-aligned_start]
    else: data = self.usb.scsi_read(sz) if self.sram_start_slot is not None else self.usb.read(self.addr + off, sz)
    if isinstance(index, slice): return data if self.fmt == 'B' else memoryview(data).cast(self.fmt).tolist()
    return int.from_bytes(data, "little")

  def __setitem__(self, index, data):
    off, sz = self._off_from_index(index)
    data = struct.pack(self.fmt, data) if isinstance(data, int) else bytes(data)
    assert len(data) == sz, f"USB MMIO write size mismatch: {len(data)} != {sz}"
    if not self.pcimem:
      self.usb.scsi_write(data, start_slot=self.sram_start_slot) if self.sram_start_slot is not None else self.usb.write(self.addr + off, data)
    elif data:
      start, end = self.addr + off, self.addr + off + len(data)
      aligned_start, aligned_end = start & ~0x3, round_up(end, 4)
      if start == aligned_start and end == aligned_end: aligned = data
      else:
        aligned = bytearray(self.usb.pcie_mem_read(aligned_start, aligned_end - aligned_start))
        aligned[start-aligned_start:end-aligned_start] = data
      self.usb.pcie_mem_write(aligned_start, aligned)

  def view(self, offset:int=0, size:int|None=None, fmt=None):
    return USBMMIOInterface(self.usb, self.addr+offset, self.nbytes-offset if size is None else size, fmt=fmt or self.fmt, pcimem=self.pcimem,
                            sram_start_slot=self.sram_start_slot)

class ASM24GSPQueueInterface(MMIOInterface):
  PAGE_SIZE, SLOT_SIZE, SRAM_SIZE, SRAM_PADDR = 0x1000, 0x4000, 0x80000, 0x200000
  PTE_PADDR = SRAM_PADDR + PAGE_SIZE
  STATUS_PADDRS = (SRAM_PADDR, SRAM_PADDR + SLOT_SIZE, SRAM_PADDR + 2 * SLOT_SIZE,
                   SRAM_PADDR + 3 * SLOT_SIZE, SRAM_PADDR + 4 * SLOT_SIZE)
  TRANSFER_START_SLOT, TRANSFER_SLOT_COUNT = 5, 25
  TRANSFER_PADDR, TRANSFER_SIZE = SRAM_PADDR + TRANSFER_START_SLOT * SLOT_SIZE, TRANSFER_SLOT_COUNT * SLOT_SIZE
  COMMAND_PADDRS = (0x27F000, 0x27B000, 0x27C000, 0x27D000, 0x27E000)
  PAGE_PADDRS = (PTE_PADDR, *COMMAND_PADDRS, *STATUS_PADDRS)

  def __init__(self, usb:CustomASM24Controller, size:int=0xB000, fmt='B', offset:int=0, root:ASM24GSPQueueInterface|None=None,
               mirror:bytes|None=None):
    self.usb, self.offset, self.nbytes, self.fmt, self.el_sz = usb, offset, size, fmt, struct.calcsize(fmt)
    if root is None:
      assert size == len(self.PAGE_PADDRS) * self.PAGE_SIZE, f"invalid NVIDIA GSP queue allocation size {size:#x}"
      if mirror is not None and len(mirror) != self.SRAM_SIZE: raise ValueError(f"invalid SRAM mirror size {len(mirror):#x}")
      self._root, self._mirror = self, bytearray(mirror or bytes(self.SRAM_SIZE))
    else: self._root = root

  def paddrs(self) -> list[int]: return list(self.PAGE_PADDRS)

  def __len__(self): return self.nbytes // self.el_sz

  def _off_from_index(self, index):
    if isinstance(index, slice):
      assert index.step in (None, 1), "strided queue slices are not supported"
      start, stop = index.start or 0, index.stop if index.stop is not None else len(self)
      return start * self.el_sz, (stop - start) * self.el_sz
    return index * self.el_sz, self.el_sz

  def _page_offset(self, logical_page:int) -> int:
    paddr = self.PAGE_PADDRS[logical_page]
    if not self.SRAM_PADDR <= paddr < self.SRAM_PADDR + self.SRAM_SIZE:
      raise ValueError(f"GSP queue page {paddr:#x} is outside ASM2464 SRAM")
    return paddr - self.SRAM_PADDR

  def _pieces(self, offset:int, size:int):
    end = offset + size
    while offset < end:
      page, page_off = divmod(offset, self.PAGE_SIZE)
      chunk = min(end - offset, self.PAGE_SIZE - page_off)
      yield self._page_offset(page) + page_off, chunk, page >= 6
      offset += chunk

  def __getitem__(self, index):
    off, size = self._off_from_index(index)
    assert 0 <= off <= self.nbytes and off + size <= self.nbytes
    absolute, out = self.offset + off, bytearray()
    for mapped, chunk, live in self._pieces(absolute, size):
      out += self.usb.sram_read(mapped, chunk) if live else self._root._mirror[mapped:mapped+chunk]
    if isinstance(index, slice): return bytes(out) if self.fmt == 'B' else memoryview(out).cast(self.fmt).tolist()
    return int.from_bytes(out, "little")

  def __setitem__(self, index, data):
    off, size = self._off_from_index(index)
    assert 0 <= off <= self.nbytes and off + size <= self.nbytes
    raw = struct.pack(self.fmt, data) if isinstance(data, int) else bytes(data)
    assert len(raw) == size, f"queue write size mismatch: {len(raw)} != {size}"

    dirty_slots:set[int] = set()
    pos = 0
    for mapped, chunk, _ in self._pieces(self.offset + off, size):
      self._root._mirror[mapped:mapped+chunk] = raw[pos:pos+chunk]
      dirty_slots.update(range(mapped // self.SLOT_SIZE, ceildiv(mapped + chunk, self.SLOT_SIZE)))
      pos += chunk

    slots = sorted(dirty_slots)
    while slots:
      start = end = slots.pop(0)
      while slots and slots[0] == end + 1: end = slots.pop(0)
      self.usb.scsi_write(bytes(self._root._mirror[start*self.SLOT_SIZE:(end+1)*self.SLOT_SIZE]), start_slot=start)

  def view(self, offset:int=0, size:int|None=None, fmt=None):
    assert 0 <= offset <= self.nbytes and (size is None or offset + size <= self.nbytes)
    return ASM24GSPQueueInterface(self.usb, self.nbytes-offset if size is None else size, fmt or self.fmt, self.offset+offset, self._root)

if DEV.interface.startswith("MOCK"): from test.mockgpu.usb import MockUSB3 as USB3  # type: ignore  # noqa: F811
