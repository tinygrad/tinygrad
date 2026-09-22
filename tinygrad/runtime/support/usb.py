from typing import cast
import ctypes, struct, time, functools, itertools
from tinygrad.runtime.autogen import libusb, libc
from tinygrad.helpers import DEBUG, DEV, to_mv, round_up, ceildiv, to_tuple, flatten
from tinygrad.dtype import dtypes, DType, AddrSpace
from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher
from tinygrad.device import Buffer, BufferSpec
from tinygrad.runtime.support.hcq2 import HCQ_RUNTIME_DEV, HCQ_DEVS, ccall, cfield, patch, unwrap_view, all_devices_in
from tinygrad.runtime.support.memory import MMIOInterface
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

  def bulk_write(self, payload:bytes, timeout:int=1000):
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
    for off in range(0, len(data), USB_MAX_STREAM):
      chunk = data[off:off+USB_MAX_STREAM]
      self._f0_out(0x60, 0x0F, address + off, len(chunk) // 4, mode=1)
      self.usb.bulk_write(chunk)

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
    else: data = self.usb.read(self.addr + off, sz)
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
# UOps implementation

# sram layout: two halves, each with a reserved sentinel block
HALF, CHUNK, SLOT = 0x40000, 0x40000 - 512, 0x4000

USB_MAX_STREAM = 1 << 20
# host memory: link, staging, one stream's worth of zeros
USB_HOST_SIZE = 32 + 2 * HALF + USB_MAX_STREAM
def usb_host(dev) -> UOp: return UOp.placeholder((USB_HOST_SIZE,), dtypes.uint8, 0, device=to_tuple(dev)[0], tag="usb_host")
def usb_link(dev) -> UOp: return usb_host(dev)[:24].bitcast(dtypes.uint64) # [handle, context, previous batch chunks]
def usb_stage(dev) -> UOp: return usb_host(dev)[32:32 + 2 * HALF] # host buffers for the sram halves

def usb_xfer(dev, half:int) -> UOp: # one bulk OUT transfer per half
  return UOp.placeholder((ctypes.sizeof(libusb.struct_libusb_transfer),), dtypes.uint8, 0, device=to_tuple(dev)[0], tag=f"usb_xfer{half}")

# vram words
def usb_vram(dev) -> UOp: return UOp.placeholder((2,), dtypes.uint32, 0, device=to_tuple(dev)[0], tag="usb_vram")
def usb_go(dev) -> UOp: return usb_vram(dev)[:1] # chunk id + 1. the host has armed its read
def usb_scratch(dev) -> UOp: return usb_vram(dev)[1:] # dummy target for empty copies

# bridge memory: sys, cq, sram
def usb_asm24(dev) -> UOp: return UOp.placeholder((0x85000,), dtypes.uint8, 0, device=to_tuple(dev)[0], tag="usb_asm24")
def usb_fence(dev) -> UOp: return usb_asm24(dev)[0x800:0x804].bitcast(dtypes.uint32) # completed gpu chunks
def usb_cq(dev) -> UOp: return usb_asm24(dev)[0x100c:0x1010].bitcast(dtypes.uint32) # completion queue (gpu reset to zero)
def usb_sram(dev) -> UOp: return usb_asm24(dev)[0x5000:0x5000 + 2 * HALF]

def usb_word(v:UOp|int, dt:DType) -> UOp: return v.cast(dt) if isinstance(v, UOp) else UOp.const(v, dt)
def usb_stack(dt:DType, *vals:UOp|int) -> UOp: # stack array for transfer data
  r = UOp.placeholder((max(1, len(vals)),), dt, addrspace=AddrSpace.REG)
  return r.after(*[r.index(i).store(usb_word(v, dt)) for i, v in enumerate(vals)])

def usb_ctrl(h:UOp, rtype:int, req:int, val:UOp|int, idx:UOp|int, data:UOp, n:UOp|int, timeout:int=1000) -> UOp:
  return ccall(libusb.libusb_control_transfer, h.index(0).load(), rtype, req, val, idx, data, n, timeout)

def usb_bulk(h:UOp, ep:int, data:UOp, n:UOp|int, timeout:int=10000) -> UOp: # NULL actual_length
  return ccall(libusb.libusb_bulk_transfer, h.index(0).load(), ep, data, n, UOp.const(0, dtypes.uint64), timeout)

def usb_poke(h:UOp, addr:UOp, val:UOp) -> UOp: # 0xF0 mode 0: write a dword
  return usb_ctrl(h, 0x40, 0xF0, 0x60 | 0x0F00, 0, usb_stack(dtypes.uint64, addr, val.bitcast(dtypes.uint32).cast(dtypes.uint64)).index(0), 12, 5000)

def usb_stream(h:UOp, addr:UOp, data:UOp, n:UOp|int, write:bool) -> UOp: # 0xF0 mode 1/2: header, then bulk data
  hdr = usb_ctrl(h, 0x40, 0xF0, (0x60 if write else 0x20) | 0x0F00, 1 if write else 2, usb_stack(dtypes.uint64, addr, n // 4).index(0), 12, 5000)
  return usb_bulk(h.after(hdr), 0x02 if write else 0x81, data, n)

# *****************
# staging rewrites

def is_host(b:UOp) -> bool: return b.device is None or not all_devices_in(b.device, HCQ_DEVS - {"CPU"}) # stack or host memory
def usb_wire(size:UOp|int) -> UOp|int: return (size + 512 + SLOT - 1) // SLOT * SLOT # payload and sentinel block, slot aligned
def usb_sentinel(g:UOp) -> UOp: return ((g & 0xFFFFFF) | 0x51000000).cast(dtypes.uint32)
def is_staged(call:UOp) -> bool: return call.op is Ops.CALL and call.body.op is Ops.STORE and is_host(call.src[1]) != is_host(call.src[2])
def usb_window(call:UOp) -> tuple[UOp, int]: # host view and the bytes a chunk moves
  return (call.src[2], CHUNK) if is_host(call.src[2]) else (call.src[1], 2 * CHUNK)
def usb_split(nbytes:int, win:int) -> list[tuple[UOp|int, int]]: # (chunk, bytes): the full chunks are one range, then the tail
  full, tail = divmod(nbytes, win)
  r = UOp.range(full, next(UOp.unique_num), dtype=dtypes.int) if full > 1 else 0 # no one-trip loops: the linearizer misplaces them
  return ([(r, win)] if full else []) + ([(full, tail)] if tail else [])

def usb_ins(name:str, *src:UOp|int) -> UOp:
  return UOp(Ops.INS, arg=(name, dtypes.void), src=tuple(s if isinstance(s, UOp) else UOp.const(s, dtypes.uint32) for s in src))

def usb_copy_slicer(call:UOp, first:int, run:int) -> list[UOp]: # first: chunk id of the copy and of its run
  dst, src = call.src[1:]
  vram, (host, win), ops = (dst if is_host(src) else src).bitcast(dtypes.uint8), usb_window(call), list[UOp]()
  dev, sram = vram.device, usb_sram(vram.device).getaddr(vram.device)

  for r, nb in usb_split(host.nbytes(), win):
    n, va = (i:=usb_word(r, dtypes.uint64)) + first, vram.getaddr(dev) + i * win
    if is_host(src): # copyin: wait for data, copy, release the half
      end = sram + (((n - run) & 1) + 1) * HALF
      ins = [usb_ins("wait_eq", end - 4, usb_sentinel(n)), usb_ins("copy", va, end - usb_wire(nb), nb), usb_ins("store", end - 4, 0)]
    else: # copyout: wait for the read, fill sram, send
      ins = [usb_ins("wait", usb_go(dev), n + 1), usb_ins("store", usb_go(dev), 0)]
      ins += [usb_ins("copy", sram + wo, va + po, pb) for wo, po, pb in ((0, 0, min(nb, CHUNK)), (HALF, CHUNK, nb - CHUNK)) if pb > 0]
      ins += [usb_ins("store", usb_cq(dev), 0)]
    ins += [usb_ins("store", usb_fence(dev), n + 1)]
    ops += [UOp(Ops.LINEAR, src=tuple(ins)).end(r)] if isinstance(r, UOp) else ins # the full chunks are one ranged block
  return ops

def usb_copy_rewriter(s:UOp) -> UOp|None:
  lins = [submit.without_after.src[0] for submit in s.src]
  if not (copies:=[call for lin in lins for call in lin.src if is_staged(call)]): return None

  # group copies
  runs, nums, n = [], list[tuple[int, int]](), 0
  for cin, grp in itertools.groupby(copies, key=lambda call: is_host(call.src[2])):
    hosts, k = list[tuple[UOp, int]](), 0 # (host view, first chunk in the run) per copy
    for call in grp:
      host, win = usb_window(call)
      nums, hosts, k = nums + [(n + k, n)], hosts + [(host, k)], k + ceildiv(host.nbytes(), win)
    runs.append((cin, n, usb_table(hosts, k, win, lins[0].arg[0][0]), k))
    n += k

  # rewrite gpu
  chunks = (usb_copy_slicer(call, *num) for call, num in zip(copies, nums))
  s = s.substitute({lin: lin.replace(src=tuple(flatten(next(chunks) if is_staged(c) else [c] for c in lin.src))) for lin in lins})

  # host side
  # TODO: maybe as cf and then unwrap?
  h = usb_link(lins[0].arg[0][0]).after(s.src[-1])
  drained = h.after(usb_drained(h, h.index(2).load() + 1))
  h = h.after(usb_ctrl(drained, 0x40, 0xE5, usb_fence(h.device).getaddr("CPU"), 0, UOp.const(0, dtypes.uint64), 0))
  for cin, run, table, k in runs: h = (usb_copyin if cin else usb_copyout)(h, table, k, run)
  return s.replace(src=(*s.src, h.index(2).store(UOp.const(n, dtypes.uint64))))
pm_usb_batch = PatternMatcher([(UPat(Ops.SINK, name="s"), usb_copy_rewriter)])

# *****************
# host functions

def usb_table(hosts:list[tuple[UOp, int]], n:int, win:int, dev) -> UOp: # [host address, bytes] per chunk
  table = UOp.placeholder((2 * n,), dtypes.uint64, device=HCQ_RUNTIME_DEV.value, tag="usb_table")
  rows:list[tuple[int|UOp, UOp]] = []
  for host, k in hosts:
    base, boff = unwrap_view(host)
    addr = base.bitcast(dtypes.uint8)[boff:boff + host.nbytes()].getaddr(to_tuple(dev)[0])
    for r, nb in usb_split(host.nbytes(), win):
      rows += [(16 * (k + r), addr + usb_word(r, dtypes.uint64) * win), (16 * (k + r) + 8, UOp.const(nb, dtypes.uint64))]
  return patch(table, rows)

def usb_reap(h:UOp, xfer:UOp) -> UOp: # poll while pending (0xff); idle transfers return
  loop = UOp.range(UOp(Ops.NOOP), next(UOp.unique_num), dtype=dtypes.void, src=(h,))
  events = ccall(libusb.libusb_handle_events_timeout, h.after(loop).index(1).load(), usb_stack(dtypes.uint64, 0, 0).index(0)) # zero timeout
  status = cfield(xfer.after(events), libusb.struct_libusb_transfer, "status").load()
  return events.backedge(loop, status.eq(0xff))

def usb_drained(h:UOp, need:UOp) -> UOp: # wait for fence == need - 1 or need, mod 256. one byte read avoids tearing
  loop, slot = UOp.range(UOp(Ops.NOOP), next(UOp.unique_num), dtype=dtypes.void, src=(h,)), usb_stack(dtypes.uint32)
  read = usb_ctrl(h.after(loop), 0xC0, 0xE4, usb_fence(h.device).getaddr("CPU"), 0, slot.index(0), 1)
  fence = slot.after(read).index(0).load()
  return read.backedge(loop, ((need - fence.cast(dtypes.uint64)) & 0xff) > 1)

def usb_chunk(h:UOp, table:UOp, i:UOp, half:int, run:int) -> UOp: # send chunk i, numbered run + i
  addr, size = table.index(2 * i).load(), table.index(2 * i + 1).load().cast(dtypes.int)
  n, wire, end = (i + run).cast(dtypes.uint64), cast(UOp, usb_wire(size)), (half + 1) * HALF
  xfer, stage = usb_xfer(h.device, half), usb_stage(h.device)

  # reuse the host buffer after its transfer completes
  h = h.after(usb_reap(h, xfer))
  h = h.after(ccall(libc.memcpy, stage.after(h).index(end - wire), addr, size.cast(dtypes.uint64)))
  h = h.after(stage.after(h).bitcast(dtypes.uint32).index(end // 4 - 1).store(usb_sentinel(n)))

  # reuse sram after the GPU copy completes
  h = h.after(usb_drained(h, n))
  h = h.after(usb_ctrl(h, 0x40, 0xF2, wire // 512, ((end - wire) // SLOT) | (wire // SLOT << 8), UOp.const(0, dtypes.uint64), 0))
  field = functools.partial(cfield, xfer:=xfer.after(h), libusb.struct_libusb_transfer)
  xfer = xfer.after(field("status").store(0xff), field("length").store(wire.cast(dtypes.uint)),
                    field("buffer").store(stage.getaddr("CPU") + (end - wire).cast(dtypes.uint64)))
  return ccall(libusb.libusb_submit_transfer, xfer.index(0))

def usb_copyin(h:UOp, table:UOp, n:int, run:int) -> UOp: # pipeline writes through two halves
  h = h.after(usb_drained(h, UOp.const(run + 1, dtypes.uint64))) # both halves must be free

  # unroll one pair: the linearizer misplaces one-trip loops
  if (pairs:=n // 2 if n // 2 > 1 else 0):
    j = UOp.range(pairs, next(UOp.unique_num), dtype=dtypes.int)
    hj = h.after(j, usb_chunk(h.after(j), table, j * 2, 0, run))
    h = h.after(usb_chunk(hj, table, j * 2 + 1, 1, run).end(j))
  for i in range(pairs * 2, n): h = h.after(usb_chunk(h, table, UOp.const(i, dtypes.int), i & 1, run))
  return h

def usb_copyout(h:UOp, table:UOp, n:int, run:int) -> UOp: # read back through both halves
  stage = usb_stage(h.device)
  h = h.after(usb_drained(h, UOp.const(run + 1, dtypes.uint64))) # wait before arming the read

  i = UOp.range(n, next(UOp.unique_num), dtype=dtypes.int)
  addr, size = table.index(2 * i).load(), table.index(2 * i + 1).load().cast(dtypes.int)
  first, second = size.minimum(CHUNK), (size - CHUNK).maximum(0) # payload bytes per half
  wire = (size + (second > 0).where(UOp.const(512, dtypes.int), UOp.const(0, dtypes.int)) + 511) // 512 * 512

  # arm the read, allow the GPU copy, receive the data
  hi = h.after(i, usb_ctrl(h.after(i), 0x40, 0xF2, (wire // 512) | 0x8000, (wire + 0x3fff) // 0x4000 << 8, UOp.const(0, dtypes.uint64), 0))
  hi = hi.after(usb_poke(hi, usb_go(h.device).getaddr("CPU"), (i + run + 1).cast(dtypes.uint32)))
  hi = hi.after(usb_bulk(hi, 0x81, stage.index(0), wire))
  hi = hi.after(ccall(libc.memcpy, addr, stage.after(hi).index(0), first.cast(dtypes.uint64)))
  hi = hi.after(ccall(libc.memcpy, addr + CHUNK, stage.after(hi).index(HALF), second.cast(dtypes.uint64)))
  return h.after(hi.end(i))

# lower device memory accesses to USB transfers
def is_remote(b:UOp) -> bool:
  return (p:=unwrap_view(b)[0]).op is Ops.PARAM and not is_host(p) and not str(p.tag).startswith(("usb_host", "usb_xfer", "put_value", "cmdbuf_copy"))
def usb_addr(b:UOp, idx:UOp, dt:DType) -> UOp: return b.getaddr("CPU") + (idx * dt.itemsize).cast(dtypes.uint64) # byte address of b[idx]
def usb_deps(b:UOp) -> tuple[UOp, ...]: # dependencies through views
  return (b.src[1:] if b.op is Ops.AFTER else ()) + (usb_deps(b.src[0]) if b.op in (Ops.BITCAST, Ops.SHRINK, Ops.AFTER) else ())
def usb_affine(idx:UOp, r:UOp) -> UOp|None: # base of base + r, independent of r
  if idx is r: return UOp.const(0, r.dtype)
  if idx.op is not Ops.ADD or r not in idx.src: return None
  base = idx.src[1] if idx.src[0] is r else idx.src[0]
  return base if r not in base.ranges else None

def usb_load(b:UOp, idx:UOp, ld:UOp) -> UOp:
  slot = usb_stack(ld.dtype)
  read = usb_stream(usb_link(b.device).after(*usb_deps(b)), usb_addr(b, idx, ld.dtype), slot.index(0), ld.dtype.itemsize, False)
  return slot.after(read).index(0).load()

def usb_store(b:UOp, idx:UOp, v:UOp) -> UOp:
  # write each patch word in order
  if idx.op is Ops.STACK:
    h = usb_store(b, idx.src[0], v.src[0])
    for i, w in zip(idx.src[1:], v.src[1:]): h = usb_store(b.after(h), i, w)
    return h

  # each control transfer writes 32 bits
  h, addr = usb_link(b.device).after(*usb_deps(b)), usb_addr(b, idx, v.dtype)
  loop, value = None, v
  if v.dtype.itemsize == 8 and str(unwrap_view(b)[0].tag).startswith("kernargs"):
    cache = UOp.placeholder((1,), v.dtype, device=HCQ_RUNTIME_DEV.value, volatile=True, tag="usb_arg_cache")
    cache = cache.after(cache.store(UOp(Ops.BINARY, arg=bytes(v.dtype.itemsize)).bitcast(v.dtype)))
    loop = UOp.range(cache.index(0).load().ne(v).cast(dtypes.int), next(UOp.unique_num), dtype=dtypes.int,
                     src=(h, v.cast(dtypes.uint32).cast(dtypes.uint64), (v >> 32).cast(dtypes.uint32).cast(dtypes.uint64)))
    h = h.after(loop)
  ret = usb_poke(h, addr, v) if v.dtype.itemsize == 4 else \
    usb_poke(h.after(usb_poke(h, addr, v.cast(dtypes.uint32))), addr + 4, (v >> 32).cast(dtypes.uint32))
  return cache.after(ret.end(loop)).index(0).store(value) if loop is not None else ret

def usb_copy(dst:UOp, di:UOp, v:UOp, r:UOp) -> UOp|None: # contiguous copy/fill
  if not is_remote(dst): return None

  # source buffer or zeros
  if v.op is Ops.LOAD and not is_remote(sb:=v.src[0].src[0]): s0, deps = usb_affine(v.src[0].src[1], r), usb_deps(sb)
  elif v.vmin == v.vmax == 0: sb, s0, deps = usb_host(dst.device)[32 + 2 * HALF:], UOp.const(0, dtypes.int), ()
  else: return usb_store(dst, di, v).end(r)
  if s0 is None or (d0:=usb_affine(di, r)) is None: return usb_store(dst, di, v).end(r)

  # empty loops write to scratch: zero-byte streams hang
  h, cnt = usb_link(dst.device).after(*usb_deps(dst), *deps, *r.src[1:]), r.src[0]
  addr = (cnt > 0).where(usb_addr(dst, d0, v.dtype), usb_scratch(dst.device).getaddr("CPU"))
  size = (cnt * v.dtype.itemsize).maximum(v.dtype.itemsize)
  loops = (UOp.range(ceildiv(size, USB_MAX_STREAM), next(UOp.unique_num), dtype=dtypes.int, src=(h,)),) if size.vmax > USB_MAX_STREAM else ()
  off = loops[0] * USB_MAX_STREAM if loops else UOp.const(0, dtypes.int)
  if v.op is Ops.LOAD: s0 += off // v.dtype.itemsize # fills reuse the zero buffer
  return usb_stream(h.after(*loops), addr + off.cast(dtypes.uint64), sb.index(s0.minimum(sb.max_numel() - 1)),
                    (size - off).minimum(USB_MAX_STREAM), True).end(*loops)

pm_usb_lower = PatternMatcher([
  (UPat.var("dst").index(UPat.var("di")).store(UPat.var("v")).end(UPat(Ops.RANGE, name="r")), usb_copy),
  (UPat.var("b").index(UPat.var("idx")).store(UPat.var("v")), lambda b, idx, v: None if idx.ranges or not is_remote(b) else usb_store(b, idx, v)),
  (UPat.var("b").index(UPat.var("idx")).load(name="ld"), lambda b, idx, ld: usb_load(b, idx, ld) if is_remote(b) else None),
])

# *****************
# bufferize

@functools.cache
def _host_block(dev) -> Buffer: # link, staging, zeros
  b = Buffer("CPU", USB_HOST_SIZE, dtypes.uint8, options=BufferSpec(nolru=True), preallocate=True)
  b.host.view(fmt='B')[:16] = struct.pack('QQ', *[ctypes.addressof(x.contents) for x in (dev.iface.pci_dev.usb.usb.handle, USB3.ctx())])
  return b
@functools.cache
def _xfer(dev, tag:str) -> Buffer: # fixed fields; status, length, buffer change per chunk
  t = libusb.libusb_alloc_transfer(0).contents
  t.dev_handle, t.endpoint, t.type, t.timeout = dev.iface.pci_dev.usb.usb.handle, 0x02, libusb.LIBUSB_TRANSFER_TYPE_BULK, 10000
  return Buffer("CPU", ctypes.sizeof(t), dtypes.uint8, options=BufferSpec(external_ptr=ctypes.addressof(t), nolru=True), preallocate=True)
@functools.cache
def _words(dev) -> Buffer: # zero the read signal and scratch
  b = Buffer(dev.device, 2, dtypes.uint32, options=BufferSpec(uncached=True, cpu_access=True, nolru=True), preallocate=True)
  b.host.view(fmt='B')[:8] = bytes(8)
  return b
pm_usb_bufferize = PatternMatcher([
  (UPat(Ops.PARAM, tag="usb_host"), lambda ctx: _host_block(ctx)),
  (UPat(Ops.PARAM, tag={"usb_xfer0", "usb_xfer1"}, name="b"), lambda ctx, b: _xfer(ctx, b.tag)),
  (UPat(Ops.PARAM, tag="usb_vram"), lambda ctx: _words(ctx)),
  (UPat(Ops.PARAM, tag="usb_asm24"), lambda ctx: ctx.iface.ctrl),
  (UPat(Ops.PARAM, name="b"), lambda b: Buffer("CPU", b.max_numel(), b.dtype, preallocate=True) if str(b.tag).startswith("cmdbuf_copy") else None),
])

if DEV.interface.startswith("MOCK"): from test.mockgpu.usb import MockUSB3 as USB3  # type: ignore  # noqa: F811
