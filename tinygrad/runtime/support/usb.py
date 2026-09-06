import ctypes, struct, time, functools, itertools
from tinygrad.runtime.autogen import libusb, libc
from tinygrad.helpers import DEBUG, DEV, to_mv, from_mv, round_up, ceildiv, to_tuple
from tinygrad.dtype import dtypes, DType, AddrSpace
from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher
from tinygrad.device import Buffer, BufferSpec
from tinygrad.runtime.support.hcq2 import HCQ_RUNTIME_DEV, EncodeCtx, ccall, rt_addr, unwrap_view, all_devices_in
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
    # async bulk OUT state: tag -> (pooled transfer, keepalive payload mv); transfer errors latch into _async_err
    self._async_seq, self._async_err = itertools.count(1), 0
    self._async_pending: dict = {}
    self._async_pool: list = []
    self._async_cb = libusb.libusb_transfer_cb_fn(self._on_bulk_done)

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

  def bulk_write(self, payload:bytes, timeout:int=1000):
    if len(payload) > len(self._bulk_mv): self._bulk_buf, self._bulk_mv = alloc_cbuffer(len(payload))
    self._bulk_mv[:len(payload)] = payload
    checked(libusb.libusb_bulk_transfer, "bulk OUT 0x02 failed") \
      (self.handle, 0x02, self._bulk_buf, len(payload), self._transferred, timeout)
    assert self._transferred.value == len(payload), f"bulk OUT short write: {self._transferred.value}/{len(payload)} bytes"

  def _on_bulk_done(self, xfer):  # runs in libusb event handling; latch errors (exceptions here are unraisable)
    exp = xfer.contents.length - 8 if xfer.contents.type == libusb.LIBUSB_TRANSFER_TYPE_CONTROL else xfer.contents.length
    if xfer.contents.status != 0 or xfer.contents.actual_length != exp: self._async_err = xfer.contents.status or -1
    self._async_pool.append(self._async_pending.pop(int(xfer.contents.user_data or 0))[0])

  def _submit_async(self, endpoint:int, xtype:int, payload:bytes|bytearray|memoryview, timeout:int) -> int:  # payload kept alive till bulk_wait
    tr = self._async_pool.pop() if self._async_pool else libusb.libusb_alloc_transfer(0)
    tr.contents.dev_handle, tr.contents.endpoint, tr.contents.type = self.handle, endpoint, xtype
    tr.contents.timeout, tr.contents.length = timeout, len(payload)
    tr.contents.buffer = ctypes.cast(from_mv(memoryview(payload), ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte))
    tr.contents.callback, tr.contents.user_data = self._async_cb, (tag := next(self._async_seq))
    self._async_pending[tag] = (tr, payload)
    checked(libusb.libusb_submit_transfer, "async submit failed")(tr)
    return tag

  def bulk_write_async(self, payload:memoryview, timeout:int=10000) -> int:
    """Queue a bulk OUT transfer without blocking; payload is kept alive until bulk_wait(tag)."""
    return self._submit_async(0x02, libusb.LIBUSB_TRANSFER_TYPE_BULK, payload, timeout)

  def control_write_async(self, request:int, value:int=0, index:int=0, data:bytes=b"", timeout:int=1000) -> int:
    """Queue a vendor control OUT without blocking; completes via bulk_wait(tag) like bulk_write_async."""
    setup = bytearray(struct.pack('<BBHHH', 0x40, request, value, index, len(data)) + data)
    return self._submit_async(0, libusb.LIBUSB_TRANSFER_TYPE_CONTROL, setup, timeout)

  def control_read_async(self, request:int, length:int, value:int=0, index:int=0, timeout:int=1000) -> tuple[int, memoryview]:
    """Queue a vendor control IN without blocking; the data lands in the returned buffer by bulk_wait(tag)."""
    buf = bytearray(struct.pack('<BBHHH', 0xC0, request, value, index, length)) + bytearray(length)
    return self._submit_async(0, libusb.LIBUSB_TRANSFER_TYPE_CONTROL, buf, timeout), memoryview(buf)[8:]

  def bulk_wait(self, tag:int):
    """Block until the tagged transfer completes; raises if any async transfer failed. LIBUSB_ERROR_INTERRUPTED is retried."""
    while tag in self._async_pending:
      if (rc:=libusb.libusb_handle_events(None)) < 0 and rc != libusb.LIBUSB_ERROR_INTERRUPTED:
        raise RuntimeError(f"libusb_handle_events: {ctypes.string_at(libusb.libusb_strerror(rc)).decode()}")
    if self._async_err: raise RuntimeError(f"async bulk OUT failed: status={self._async_err}")

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
    self._f0_out(0x60, 0x0F, address, len(data) // 4, mode=1)
    self.usb.bulk_write(data)

  def pcie_mem_read(self, address:int, nbytes:int) -> memoryview:
    """Streaming PCIe memory read via 0xF0 mode 2 + bulk IN. Returns little-endian bytes."""
    assert nbytes % 4 == 0, f"pcie_mem_read requires 4-byte aligned size, got {nbytes}"
    self._f0_out(0x20, 0x0F, address, nbytes // 4, mode=2)
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

  def scsi_write(self, buf:bytes, slot_start:int=0):
    """Write to SRAM via 0xF2 vendor command + bulk OUT."""
    buf_padded = buf + b'\x00' * (round_up(len(buf), 512) - len(buf))
    self.usb.control_write(0xF2, value=len(buf_padded) // 512, index=(slot_start & 0xFF) | (ceildiv(len(buf_padded), 0x4000) << 8))
    self.usb.bulk_write(buf_padded)

  def scsi_read_arm(self, size:int):
    windex = (ceildiv(size, 0x4000) & 0xFF) << 8
    self.usb.control_write(0xF2, value=(ceildiv(size, 512) & 0x7FFF) | 0x8000, index=windex)

  def scsi_read(self, size:int) -> memoryview: return self.usb.bulk_read(round_up(size, 512), timeout=10000)[:size]

class USBMMIOInterface(MMIOInterface):
  def __init__(self, usb, addr, size, fmt, pcimem=True): # pylint: disable=super-init-not-called
    self.usb, self.addr, self.nbytes, self.fmt, self.el_sz, self.pcimem = usb, addr, size, fmt, struct.calcsize(fmt), pcimem

  def _off_from_index(self, index):
    if isinstance(index, slice): return ((index.start or 0) * self.el_sz, ((index.stop or len(self))-(index.start or 0)) * self.el_sz)
    return (index * self.el_sz, self.el_sz)

  def __getitem__(self, index):
    off, sz = self._off_from_index(index)
    if self.pcimem:
      assert sz % 4 == 0 and off % 4 == 0, f"pcie_mem_read requires 4-byte aligned access, got off={off}, sz={sz}"
      data = self.usb.pcie_mem_read(self.addr + off, sz)
    else: data = self.usb.scsi_read(sz) if self.addr == 0xf000 else self.usb.read(self.addr + off, sz)
    return data if isinstance(index, slice) else int.from_bytes(data, "little")

  def __setitem__(self, index, data):
    off, _ = self._off_from_index(index)
    data = struct.pack(self.fmt, data) if isinstance(data, int) else bytes(data)
    if not self.pcimem: self.usb.scsi_write(data) if self.addr == 0xf000 else self.usb.write(self.addr + off, data)
    else:
      # writes are whole dwords
      assert len(data) % 4 == 0 and off % 4 == 0, f"pcie_mem_write requires 4-byte aligned access, got off={off}, sz={len(data)}"
      self.usb.pcie_mem_write(self.addr+off, data)

  def view(self, offset:int=0, size:int|None=None, fmt=None):
    return USBMMIOInterface(self.usb, self.addr+offset, self.nbytes-offset if size is None else size, fmt=fmt or self.fmt, pcimem=self.pcimem)

# *****************
# hcq2: the host program drives the board over libusb. an access to device memory becomes a transfer on the link, the buffer holding the
# libusb handle. every transfer is h.after(call), so the link threads through the program in order: h is always the link after all before it.
# a copy between the host and vram goes through the controller's sram in chunks: the queue moves each chunk, the host streams it

# *****************
# 0. helpers

HALF, CHUNK = 0x40000, 0x40000 - 512 # the sram has two halves, a chunk fills one: its payload, then the block holding its sentinel
PENDING = 0xff # a transfer status libusb never sets: the program marks a transfer before submitting it, reaps it once the status changed
USB_HOST = ("usb_link", "usb_stage", "usb_xfer", "usb_zeros", "put_value", "cmdbuf_copy") # the device's placeholders that live in host memory

def usb_buf(dev, tag:str, n:int=1, dt:DType=dtypes.uint8) -> UOp: return UOp.placeholder((n,), dt, 0, device=to_tuple(dev)[0], tag=f"usb_{tag}")
def usb_link(dev) -> UOp: return usb_buf(dev, "link", 3, dtypes.uint64) # [the libusb handle, its context, the chunks copied so far]
def usb_xfer(x:UOp, field:str) -> UOp: # a field of a libusb transfer struct
  f = getattr(libusb.struct_libusb_transfer, field)
  return x[f.offset:f.offset + f.size].bitcast({4: dtypes.int32, 8: dtypes.uint64}[f.size]).index(0)

def usb_reg(dt:DType, *vals:UOp|int) -> UOp: # an array on the program's stack holding vals: what a transfer reads or writes. a wide val takes two
  words = [w for v in vals for w in ((v, v >> 32) if isinstance(v, UOp) and v.dtype.itemsize > dt.itemsize else (v,))]
  r = UOp.placeholder((max(2, len(words)),), dt, addrspace=AddrSpace.REG)
  return r.after(*[r.index(i).store(w.cast(dt) if isinstance(w, UOp) else UOp.const(w, dt)) for i, w in enumerate(words)])

def _addr(b:UOp, idx:UOp, dt:DType) -> UOp: return rt_addr(b) + (idx * dt.itemsize).cast(dtypes.uint64) # of an element of a view
def _host(b:UOp) -> bool: return b.device is None or all_devices_in(b.device, frozenset({HCQ_RUNTIME_DEV.value.split(":")[0]})) # or the stack

# *****************
# 1. transfers

def usb_ctrl(h:UOp, rtype:int, req:int, val:UOp|int, idx:UOp|int, data:UOp, n:UOp|int, timeout:int=1000) -> UOp:
  return h.after(ccall(libusb.libusb_control_transfer, h.index(0).load(), rtype, req, val, idx, data, n, timeout))

def usb_bulk(h:UOp, ep:int, data:UOp, n:UOp|int, timeout:int=10000) -> UOp: # NULL actual_length
  return h.after(ccall(libusb.libusb_bulk_transfer, h.index(0).load(), ep, data, n, UOp.const(0, dtypes.uint64), timeout))

def usb_poke(h:UOp, addr:UOp, val:UOp) -> UOp: # 0xF0 mode 0: a dword in one control transfer
  return usb_ctrl(h, 0x40, 0xF0, 0x60 | 0x0F00, 0, usb_reg(dtypes.uint32, addr, val.bitcast(dtypes.uint32)).index(0), 12, 5000)

def usb_stream(h:UOp, addr:UOp, data:UOp, n:UOp|int, write:bool) -> UOp: # 0xF0 mode 1/2: the header, then the payload on the bulk endpoint
  hdr = usb_reg(dtypes.uint32, addr, n // 4)
  h = usb_ctrl(h, 0x40, 0xF0, (0x60 if write else 0x20) | 0x0F00, 1 if write else 2, hdr.index(0), 12, 5000)
  return usb_bulk(h, 0x02 if write else 0x81, data, n)

# *****************
# 2. prep: the queue side of a copy, chunk by chunk. g numbers the chunks of the queue, the host side counts them the same way

def usb_wire(size:UOp|int) -> UOp|int: return (size + 4 + 511) // 512 * 512 # a chunk on the wire: the payload, then its sentinel in the last dword
def usb_sentinel(g:UOp) -> UOp: return ((g & 0xFFFFFF) | 0x51000000).cast(dtypes.uint32)
def _staged(call:UOp) -> bool: return call.op is Ops.CALL and call.src[0].op is Ops.COPY and _host(call.src[1]) != _host(call.src[2])

def usb_chunks(lin:UOp) -> UOp: # per chunk: wait for the host, move the chunk between the sram and vram, release the sram
  dev, seq = lin.arg[0][0], itertools.count()
  def ins(name:str, dst:UOp, val:UOp) -> UOp: return UOp(Ops.INS, arg=(name, dtypes.void), src=(dst, val))
  def chunks(call:UOp) -> list[UOp]:
    cin = _host(call.src[2]) # into vram from the host, or out of it
    (vram, win), sram, ops = ((call.src[1], CHUNK) if cin else (call.src[2], 2 * HALF)), usb_buf(dev, "sram", 2 * HALF), []
    for k, off in enumerate(range(0, vram.nbytes(), win)): # off and nb are bytes: the two ends of a copy can differ in dtype
      g, nb, wo = usb_link(dev).index(2).load() + next(seq), min(win, vram.nbytes() - off), (k & 1) * HALF if cin else 0
      w, v = sram[wo:wo + nb], vram[off // vram.dtype.itemsize:(off + nb) // vram.dtype.itemsize]
      if cin: ops.append(ins("wait_eq", sram[(o:=wo + usb_wire(nb) - 4):o + 4].bitcast(dtypes.uint32), usb_sentinel(g))) # the host streamed it
      else: ops.append(ins("wait", usb_buf(dev, "go", 1, dtypes.uint32), g + 1)) # the host armed a read of the window
      ops.append(call.replace(src=(call.src[0], v, w) if cin else (call.src[0], w, v)))
      ops.append(ins("store", usb_buf(dev, "fence", 1, dtypes.uint32), g + 1)) # the chunk is done with the sram
      if not cin: ops.append(ins("store", usb_buf(dev, "cq", 0x1000)[12:16].bitcast(dtypes.uint32), UOp.const(0, dtypes.uint32))) # send it
    return ops
  return lin.replace(src=tuple(u for call in lin.src for u in (chunks(call) if _staged(call) else [call])))

# *****************
# 3. batch: the host side of the copies, once every queue is running

def usb_batch(s:UOp) -> UOp|None: # the copies of the batch, in queue order, ride along a host block after the last submit
  calls = [call for submit in s.src for call in submit.without_after.src[0].src if _staged(call)]
  return s.replace(src=(*s.src, UOp.custom_function("usb_host", *calls).after(*s.src[-1:]))) if calls else None
pm_usb_batch = PatternMatcher([(UPat(Ops.SINK, name="s"), usb_batch)])

def usb_reap(h:UOp, xfer:UOp) -> UOp: # poll the event loop until the async transfer is done
  loop = UOp.range(UOp(Ops.NOOP), next(UOp.unique_num), dtype=dtypes.void, src=(h,)) # a loop on the link, unique by it
  events = ccall(libusb.libusb_handle_events_timeout, h.after(loop).index(1).load(), usb_reg(dtypes.uint64, 0, 0).index(0))
  status = usb_xfer(xfer.after(events), "status").load()
  return h.after(status.end(loop, status.eq(PENDING)))

def usb_drained(h:UOp, need:UOp) -> UOp: # wait until the queue is done with the chunks before need: it may lag one chunk, the other half
  loop, slot = UOp.range(UOp(Ops.NOOP), next(UOp.unique_num), dtype=dtypes.void, src=(h,)), usb_reg(dtypes.uint32)
  h = usb_ctrl(h.after(loop), 0xC0, 0xE4, rt_addr(usb_buf(h.device, "fence", 1, dtypes.uint32)), 0, slot.index(0), 1) # 0xE4 reads controller memory
  fence = slot.after(h).index(0).load().cast(dtypes.uint64) # one byte can't tear, the count is compared mod 256
  return h.after(fence.end(loop, ((need - fence) & 0xff) > 1))

def usb_chunk(h:UOp, src:UOp, k:UOp, half:int) -> UOp: # stream chunk k of src into a half of the sram: an async bulk after its arm
  xfer, stage = usb_buf(h.device, f"xfer{half}", 64), usb_buf(h.device, "stage", 2 * HALF)
  g, size = h.index(2).load() + k.cast(dtypes.uint64), (UOp.const(src.nbytes(), dtypes.int) - k * CHUNK).minimum(CHUNK)
  wire = usb_wire(size)
  h = usb_reap(h, xfer) # the transfer that used this half before
  h = h.after(ccall(libc.memcpy, stage.after(h).index(half * HALF), rt_addr(src) + (k * CHUNK).cast(dtypes.uint64), size.cast(dtypes.uint64)))
  h = h.after(stage.after(h).bitcast(dtypes.uint32).index((half * HALF + wire - 4) // 4).store(usb_sentinel(g)))
  h = usb_drained(h, g) # the queue is done with the chunk that used this half before
  h = usb_ctrl(h, 0x40, 0xF2, wire // 512, half * 16 | ((wire + 0x3fff) // 0x4000 << 8), UOp.const(0, dtypes.uint64), 0) # arm the write
  xfer = xfer.after(h)
  xfer = xfer.after(usb_xfer(xfer, "status").store(PENDING), usb_xfer(xfer, "length").store(wire),
                    usb_xfer(xfer, "buffer").store(rt_addr(stage) + half * HALF))
  return h.after(ccall(libusb.libusb_submit_transfer, xfer.index(0)))

def usb_copyin(h:UOp, src:UOp, k0:int) -> tuple[UOp, int]: # the chunks alternate the halves, two transfers in flight
  n, first = ceildiv(src.nbytes(), CHUNK), UOp.const(k0, dtypes.int)
  h = usb_drained(h, h.index(2).load() + (k0 + 1)) # the sram is free
  # a loop over pairs of chunks: the halves are static. a single pair is unrolled: the linearizer places a one trip loop with the code around it
  pairs, rest = (n // 2, n % 2) if n // 2 > 1 else (0, n)
  if pairs:
    j = UOp.range(pairs, next(UOp.unique_num), dtype=dtypes.int)
    h = h.after(usb_chunk(usb_chunk(h.after(j), src, first + j * 2, 0), src, first + j * 2 + 1, 1).end(j))
  for k in range(n - rest, n): h = usb_chunk(h, src, first + k, k & 1)
  for half in range(2): h = usb_reap(h, usb_buf(h.device, f"xfer{half}", 64)) # nothing in flight after a copy: a read may come next
  return h, k0 + n

def usb_copyout(h:UOp, dst:UOp, k0:int) -> tuple[UOp, int]: # per window: arm a read of the sram, release the queue to fill it, pull it
  n, stage, seq = ceildiv(dst.nbytes(), 2 * HALF), usb_buf(h.device, "stage", 2 * HALF), h.index(2).load()
  h = usb_drained(h, seq + (k0 + 1)) # the sram is free and nothing else writes the controller's memory
  k = UOp.range(n, next(UOp.unique_num), dtype=dtypes.int)
  size = (UOp.const(dst.nbytes(), dtypes.int) - k * (2 * HALF)).minimum(2 * HALF)
  wire = (size + 511) // 512 * 512
  hk = usb_ctrl(h.after(k), 0x40, 0xF2, (wire // 512) | 0x8000, (wire + 0x3fff) // 0x4000 << 8, UOp.const(0, dtypes.uint64), 0) # arm the read
  hk = usb_poke(hk, rt_addr(usb_buf(h.device, "go", 1, dtypes.uint32)), (seq + (k + k0 + 1).cast(dtypes.uint64)).cast(dtypes.uint32))
  hk = usb_bulk(hk, 0x81, stage.index(0), wire)
  hk = hk.after(ccall(libc.memcpy, rt_addr(dst) + (k * (2 * HALF)).cast(dtypes.uint64), stage.after(hk).index(0), size.cast(dtypes.uint64)))
  return h.after(hk.end(k)), k0 + n

def usb_host(ctx:EncodeCtx, f:UOp) -> UOp: # the copies in order, k0 counts their chunks like the queue does
  h, k0 = usb_link(ctx.devs[0]), 0
  for call in f.src: h, k0 = usb_copyin(h, call.src[2], k0) if _host(call.src[2]) else usb_copyout(h, call.src[1], k0)
  return h.index(2).store(h.index(2).load() + k0)
pm_usb_encode = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg="usb_host", name="f"), usb_host)])

# *****************
# 4. lower: the host's accesses to device memory. a load streams the value into a register, a store pokes it, a loop of stores streams the source

def _remote(b:UOp) -> bool: return (p:=unwrap_view(b)[0]).op is Ops.PARAM and not _host(p) and not str(p.tag).startswith(USB_HOST)
def _deps(b:UOp) -> tuple[UOp, ...]: # what a buffer view is after
  return (b.src[1:] if b.op is Ops.AFTER else ()) + (_deps(b.src[0]) if b.op in (Ops.BITCAST, Ops.SHRINK, Ops.AFTER) else ())

def _affine(idx:UOp, r:UOp) -> UOp|None: # idx = base + r: the base, None if idx doesn't walk r with unit stride
  if idx is r: return UOp.const(0, r.dtype)
  if idx.op is not Ops.ADD or r not in idx.src: return None
  base = idx.src[1] if idx.src[0] is r else idx.src[0]
  return base if r not in base.ranges else None

def usb_load(b:UOp, idx:UOp, ld:UOp) -> UOp:
  slot = usb_reg(ld.dtype)
  h = usb_stream(usb_link(b.device).after(*_deps(b)), _addr(b, idx, ld.dtype), slot.index(0), ld.dtype.itemsize, False)
  return slot.after(h).index(0).load()

def usb_store(b:UOp, idx:UOp, v:UOp) -> UOp:
  if idx.op is Ops.STACK: # a patch: word by word, each after the one before
    h = usb_store(b, idx.src[0], v.src[0])
    for i, w in zip(idx.src[1:], v.src[1:]): h = usb_store(b.after(h), i, w)
    return h
  h, addr = usb_link(b.device).after(*_deps(b)), _addr(b, idx, v.dtype)
  if v.dtype.itemsize == 4: return usb_poke(h, addr, v)
  return usb_poke(usb_poke(h, addr, v.cast(dtypes.uint32)), addr + 4, (v >> 32).cast(dtypes.uint32))

def usb_copy(dst:UOp, di:UOp, v:UOp, r:UOp) -> UOp|None: # a loop of unit stride stores from host memory: one stream
  if not _remote(dst): return None
  if v.op is Ops.LOAD and not _remote(sb:=v.src[0].src[0]): s0, deps = _affine(v.src[0].src[1], r), _deps(sb) # from host memory
  elif v.vmin == v.vmax == 0: sb, s0, deps = usb_buf(dst.device, "zeros", r.src[0].vmax * v.dtype.itemsize), UOp.const(0, dtypes.int), ()
  else: return usb_store(dst, di, v).end(r)
  if s0 is None or (d0:=_affine(di, r)) is None: return usb_store(dst, di, v).end(r)
  h, cnt = usb_link(dst.device).after(*_deps(dst), *deps, *r.src[1:]), r.src[0]
  # the firmware can't stream zero bytes: an empty loop streams one element into a scratch word instead
  addr = (cnt > 0).where(_addr(dst, d0, v.dtype), rt_addr(usb_buf(dst.device, "scratch", 1, dtypes.uint32)))
  return usb_stream(h, addr, sb.index(s0.minimum(sb.max_numel() - 1)), (cnt * v.dtype.itemsize).maximum(v.dtype.itemsize), True)

pm_usb_lower = PatternMatcher([ # a store inside a loop is left to the loop's END
  (UPat.var("dst").index(UPat.var("di")).store(UPat.var("v")).end(UPat(Ops.RANGE, name="r")), usb_copy),
  (UPat.var("b").index(UPat.var("idx")).store(UPat.var("v")), lambda b, idx, v: None if idx.ranges or not _remote(b) else usb_store(b, idx, v)),
  (UPat.var("b").index(UPat.var("idx")).load(name="ld"), lambda b, idx, ld: usb_load(b, idx, ld) if _remote(b) else None),
])

# *****************
# 5. bufferize: the device's usb state, bound once for the life of the device

def _init(b:Buffer, data:bytes) -> Buffer: # the buffer, allocated and holding data
  b.ensure_allocated()._buf.cpu_view().view(fmt='B')[:len(data)] = data
  return b
def _region(dev, b) -> Buffer: # a window of the controller's memory: not ours to free
  return Buffer(dev.device, b.size, dtypes.uint8, options=BufferSpec(external_ptr=b.va_addr, nolru=True)).allocate(opaque=b)

@functools.cache
def _link(dev) -> Buffer: # [the libusb handle, its context, the chunks copied so far]
  handles = [ctypes.addressof(x.contents) for x in (dev.iface.pci_dev.usb.usb.handle, USB3.ctx())]
  return _init(Buffer("CPU", 3, dtypes.uint64), struct.pack('QQQ', *handles, 0))
@functools.cache
def _xfer(dev, tag:str) -> Buffer: # an async bulk out: the program sets its buffer and length, then reaps its status
  t = libusb.libusb_alloc_transfer(0).contents
  t.dev_handle, t.endpoint, t.type, t.timeout = dev.iface.pci_dev.usb.usb.handle, 0x02, libusb.LIBUSB_TRANSFER_TYPE_BULK, 10000
  return Buffer("CPU", ctypes.sizeof(t), dtypes.uint8, options=BufferSpec(external_ptr=ctypes.addressof(t), nolru=True), preallocate=True)
@functools.cache
def _cpu(dev, tag:str, n:int) -> Buffer: return Buffer("CPU", n, dtypes.uint8, options=BufferSpec(nolru=True), preallocate=True)
@functools.cache
def _word(dev, tag:str) -> Buffer: # a dword in vram
  return _init(Buffer(dev.device, 1, dtypes.uint32, options=BufferSpec(uncached=True, cpu_access=True, nolru=True)), bytes(4))
@functools.cache
def _fence(dev) -> Buffer: return _init(_region(dev, dev.iface.sys_buf).view(1, dtypes.uint32, 0x800), bytes(4))
@functools.cache
def _sram(dev) -> Buffer: return _init(_region(dev, dev.iface.sram), bytes(2 * HALF)) # no stale sentinel from an earlier process
@functools.cache
def _cq(dev) -> Buffer: return _region(dev, dev.iface.cq_buf)

pm_usb_bufferize = PatternMatcher([
  (UPat(Ops.PARAM, tag="usb_link"), lambda ctx: _link(ctx)),
  (UPat(Ops.PARAM, tag={"usb_xfer0", "usb_xfer1"}, name="b"), lambda ctx, b: _xfer(ctx, b.tag)),
  (UPat(Ops.PARAM, tag={"usb_stage", "usb_zeros"}, name="b"), lambda ctx, b: _cpu(ctx, b.tag, b.max_numel())),
  (UPat(Ops.PARAM, tag={"usb_go", "usb_scratch"}, name="b"), lambda ctx, b: _word(ctx, b.tag)), # words in vram the host pokes
  (UPat(Ops.PARAM, tag="usb_fence"), lambda ctx: _fence(ctx)),
  (UPat(Ops.PARAM, tag="usb_sram"), lambda ctx: _sram(ctx)),
  (UPat(Ops.PARAM, tag="usb_cq"), lambda ctx: _cq(ctx)),
  (UPat(Ops.PARAM, name="b"), lambda b: Buffer("CPU", b.max_numel(), b.dtype, preallocate=True) if str(b.tag).startswith("cmdbuf_copy") else None),
]) # the sdma cmdbuf streams into the ring from the host

if DEV.interface.startswith("MOCK"): from test.mockgpu.usb import MockUSB3 as USB3  # type: ignore  # noqa: F811
