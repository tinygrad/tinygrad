import ctypes, struct, dataclasses, array, itertools, time
from typing import Sequence
from tinygrad.runtime.autogen import libusb
from tinygrad.helpers import DEBUG, to_mv, round_up, OSX, getenv
from tinygrad.runtime.support.hcq import MMIOInterface

class USB3:
  def __init__(self, vendor:int, dev:int, ep_data_in:int, ep_stat_in:int, ep_data_out:int, ep_cmd_out:int, max_streams:int=31, use_bot=False):
    self.vendor, self.dev = vendor, dev
    self.ep_data_in, self.ep_stat_in, self.ep_data_out, self.ep_cmd_out = ep_data_in, ep_stat_in, ep_data_out, ep_cmd_out
    self.max_streams, self.use_bot = max_streams, use_bot
    self.ctx = ctypes.POINTER(libusb.struct_libusb_context)()

    if libusb.libusb_init(ctypes.byref(self.ctx)): raise RuntimeError("libusb_init failed")
    if DEBUG >= 6: libusb.libusb_set_option(self.ctx, libusb.LIBUSB_OPTION_LOG_LEVEL, 4)

    self.handle = libusb.libusb_open_device_with_vid_pid(self.ctx, self.vendor, self.dev)
    if not self.handle: raise RuntimeError(f"device {self.vendor:04x}:{self.dev:04x} not found. sudo required?")

    # Detach kernel driver if needed
    if libusb.libusb_kernel_driver_active(self.handle, 0):
      libusb.libusb_detach_kernel_driver(self.handle, 0)
      libusb.libusb_reset_device(self.handle)

    # Set configuration and claim interface
    if libusb.libusb_set_configuration(self.handle, 1): raise RuntimeError("set_configuration failed")
    if libusb.libusb_claim_interface(self.handle, 0): raise RuntimeError("claim_interface failed. sudo required?")

    if use_bot:
      self._tag = 0
    else:
      if libusb.libusb_set_interface_alt_setting(self.handle, 0, 1): raise RuntimeError("alt_setting failed")

      # Clear any stalled endpoints
      all_eps = (self.ep_data_out, self.ep_data_in, self.ep_stat_in, self.ep_cmd_out)
      for ep in all_eps: libusb.libusb_clear_halt(self.handle, ep)

      # Allocate streams (falls back to no-stream UAS on USB 2.0)
      stream_eps = (ctypes.c_uint8 * 3)(self.ep_data_out, self.ep_data_in, self.ep_stat_in)
      rc = libusb.libusb_alloc_streams(self.handle, self.max_streams * len(stream_eps), stream_eps, len(stream_eps))
      self.use_streams = rc >= 0
      if not self.use_streams: self.max_streams = 1
      self._uas_tag = 0

      # Base cmd
      cmd_template = bytes([0x01, 0x00, 0x00, 0x01, *([0] * 12), 0xE4, 0x24, 0x00, 0xB2, 0x1A, 0x00, 0x00, 0x00, *([0] * 8)])

      # Init pools
      self.tr = {ep: [libusb.libusb_alloc_transfer(0) for _ in range(self.max_streams)] for ep in all_eps}

      self.buf_cmd = [(ctypes.c_uint8 * len(cmd_template))(*cmd_template) for _ in range(self.max_streams)]
      self.buf_stat = [(ctypes.c_uint8 * 64)() for _ in range(self.max_streams)]
      self.buf_data_in = [(ctypes.c_uint8 * 0x1000)() for _ in range(self.max_streams)]
      self.buf_data_out = [(ctypes.c_uint8 * 0x80000)() for _ in range(self.max_streams)]
      self.buf_data_out_mvs = [to_mv(ctypes.addressof(self.buf_data_out[i]), 0x80000) for i in range(self.max_streams)]

      for slot in range(self.max_streams): struct.pack_into(">B", self.buf_cmd[slot], 3, slot + 1)

  def _prep_transfer(self, tr, ep, stream_id, buf, length):
    tr.contents.dev_handle, tr.contents.endpoint, tr.contents.length, tr.contents.buffer = self.handle, ep, length, buf
    tr.contents.status, tr.contents.flags, tr.contents.timeout, tr.contents.num_iso_packets = 0xff, 0, 1000, 0
    tr.contents.type = (libusb.LIBUSB_TRANSFER_TYPE_BULK_STREAM if stream_id is not None else libusb.LIBUSB_TRANSFER_TYPE_BULK)
    if stream_id is not None: libusb.libusb_transfer_set_stream_id(tr, stream_id)
    return tr

  def _submit_and_wait(self, cmds):
    for tr in cmds: libusb.libusb_submit_transfer(tr)

    running = len(cmds)
    while running:
      libusb.libusb_handle_events(self.ctx)
      running = len(cmds)
      for tr in cmds:
        if tr.contents.status == libusb.LIBUSB_TRANSFER_COMPLETED: running -= 1
        elif tr.contents.status != 0xFF: raise RuntimeError(f"EP 0x{tr.contents.endpoint:02X} error: {tr.contents.status}")

  def _bulk_out(self, ep: int, payload: bytes, timeout: int = 1000):
    transferred = ctypes.c_int(0)
    rc = libusb.libusb_bulk_transfer(
      self.handle,
      ep,
      (ctypes.c_ubyte * len(payload))(*payload),
      len(payload),
      ctypes.byref(transferred),
      timeout,
    )
    assert rc == 0, f"bulk OUT 0x{ep:02X} failed: {rc}"
    assert transferred.value == len(payload), f"bulk OUT short write on 0x{ep:02X}: {transferred.value}/{len(payload)} bytes"

  def _bulk_in(self, ep: int, length: int, timeout: int = 1000) -> bytes:
    buf, transferred = (ctypes.c_ubyte * length)(), ctypes.c_int(0)
    rc = libusb.libusb_bulk_transfer(
      self.handle,
      ep,
      buf,
      length,
      ctypes.byref(transferred),
      timeout,
    )
    assert rc == 0, f"bulk IN 0x{ep:02X} failed: {rc}"
    return bytes(buf[:transferred.value])

  def send_batch(self, cdbs:list[bytes], idata:list[int]|None=None, odata:list[bytes|None]|None=None) -> list[bytes|None]:
    idata, odata = idata or [0] * len(cdbs), odata or [None] * len(cdbs)
    results:list[bytes|None] = []
    tr_window, op_window = [], []

    for idx, (cdb, rlen, send_data) in enumerate(zip(cdbs, idata, odata)):
      if self.use_bot:
        dir_in = rlen > 0
        data_len = rlen if dir_in else (len(send_data) if send_data is not None else 0)
        assert not (rlen > 0 and send_data is not None), "BOT mode only supports either read or write per command"

        # CBW
        self._tag += 1
        flags = 0x80 if dir_in else 0x00
        cbw = struct.pack("<IIIBBB", 0x43425355, self._tag, data_len, flags, 0, len(cdb)) + cdb + b"\x00" * (16 - len(cdb))
        self._bulk_out(self.ep_data_out, cbw)

        # DAT
        if dir_in:
          results.append(self._bulk_in(self.ep_data_in, rlen))
        else:
          if send_data is not None:
            self._bulk_out(self.ep_data_out, send_data)
          results.append(None)

        # CSW
        sig, rtag, residue, status = struct.unpack("<IIIB", self._bulk_in(self.ep_data_in, 13, timeout=2000))
        assert sig == 0x53425355, f"Bad CSW signature 0x{sig:08X}, expected 0x53425355"
        assert rtag == self._tag, f"CSW tag mismatch: got {rtag}, expected {self._tag}"
        if status != 0 and cdb[0] != 0x8A: assert status == 0, f"SCSI command failed, CSW status=0x{status:02X}, residue={residue}"
      elif not self.use_streams:
        # UAS without streams: serialize cmd -> data -> status, unique tag per command
        slot = 0
        self.buf_cmd[slot][16:16+len(cdb)] = list(cdb)
        self._uas_tag = (self._uas_tag % 255) + 1  # UAS tag must be unique and non-zero
        self.buf_cmd[slot][3] = self._uas_tag

        # 1. Send command IU
        self._bulk_out(self.ep_cmd_out, bytes(self.buf_cmd[slot]))

        # 2. Data phase + status
        if rlen:
          if rlen > len(self.buf_data_in[slot]): self.buf_data_in[slot] = (ctypes.c_uint8 * round_up(rlen, 0x1000))()
          results.append(self._bulk_in(self.ep_data_in, rlen))
          _stat = self._bulk_in(self.ep_stat_in, 64)
        elif send_data is not None:
          for _retry in range(10):
            _rtt = self._bulk_in(self.ep_stat_in, 64)  # Ready-to-Transfer IU or early completion
            if _rtt[0] == 0x07: break  # RTT: device ready for data
            # Device sent Sense/Response instead of RTT, re-send command
            self._uas_tag = (self._uas_tag % 255) + 1
            self.buf_cmd[slot][3] = self._uas_tag
            self._bulk_out(self.ep_cmd_out, bytes(self.buf_cmd[slot]))
          else: raise RuntimeError("UAS: failed to get Ready-to-Transfer after 10 retries")
          self._bulk_out(self.ep_data_out, send_data)
          _stat = self._bulk_in(self.ep_stat_in, 64)
          results.append(None)
        else:
          # No data phase - just read status
          _stat = self._bulk_in(self.ep_stat_in, 64)
          results.append(None)
      else:
        # allocate slot and stream. stream is 1-based
        slot, stream = idx % self.max_streams, (idx % self.max_streams) + 1

        # build cmd packet
        self.buf_cmd[slot][16:16+len(cdb)] = list(cdb)

        # cmd + stat transfers
        tr_window.append(self._prep_transfer(self.tr[self.ep_cmd_out][slot], self.ep_cmd_out, None, self.buf_cmd[slot], len(self.buf_cmd[slot])))
        tr_window.append(self._prep_transfer(self.tr[self.ep_stat_in][slot], self.ep_stat_in, stream, self.buf_stat[slot], 64))

        if rlen:
          if rlen > len(self.buf_data_in[slot]): self.buf_data_in[slot] = (ctypes.c_uint8 * round_up(rlen, 0x1000))()
          tr_window.append(self._prep_transfer(self.tr[self.ep_data_in][slot], self.ep_data_in, stream, self.buf_data_in[slot], rlen))

        if send_data is not None:
          if len(send_data) > len(self.buf_data_out[slot]):
            self.buf_data_out[slot] = (ctypes.c_uint8 * len(send_data))()
            self.buf_data_out_mvs[slot] = to_mv(ctypes.addressof(self.buf_data_out[slot]), len(send_data))

          self.buf_data_out_mvs[slot][:len(send_data)] = bytes(send_data)
          tr_window.append(self._prep_transfer(self.tr[self.ep_data_out][slot], self.ep_data_out, stream, self.buf_data_out[slot], len(send_data)))

        op_window.append((idx, slot, rlen))
        if (idx + 1 == len(cdbs)) or len(op_window) >= self.max_streams:
          self._submit_and_wait(tr_window)
          for idx, slot, rlen in op_window: results.append(bytes(self.buf_data_in[slot][:rlen]) if rlen else None)
          tr_window = []

    return results

@dataclasses.dataclass(frozen=True)
class WriteOp: addr:int; data:bytes; ignore_cache:bool=True # noqa: E702

@dataclasses.dataclass(frozen=True)
class ReadOp: addr:int; size:int # noqa: E702

@dataclasses.dataclass(frozen=True)
class ScsiWriteOp: data:bytes; lba:int=0 # noqa: E702

class ASM24Controller:
  def __init__(self):
    self.usb = USB3(0xADD1, 0x0001, 0x81, 0x83, 0x02, 0x04, use_bot=bool(getenv("USE_BOT", 0)))
    self._cache: dict[int, int|None] = {}
    self._pci_cacheable: list[tuple[int, int]] = []
    self._pci_cache: dict[int, int|None] = {}

    # Init controller.
    self.exec_ops([WriteOp(0x54b, b' '), WriteOp(0x54e, b'\x04'), WriteOp(0x5a8, b'\x02'), WriteOp(0x5f8, b'\x04'),
      WriteOp(0x7ec, b'\x01\x00\x00\x00'), WriteOp(0xc422, b'\x02'), WriteOp(0x0, b'\x33')])

  def exec_ops(self, ops:Sequence[WriteOp|ReadOp|ScsiWriteOp]):
    cdbs:list[bytes] = []
    idata:list[int] = []
    odata:list[bytes|None] = []

    def _add_req(cdb:bytes, i:int, o:bytes|None):
      nonlocal cdbs, idata, odata
      cdbs, idata, odata = cdbs + [cdb], idata + [i], odata + [o]

    for op in ops:
      if isinstance(op, WriteOp):
        for off, value in enumerate(op.data):
          addr = ((op.addr + off) & 0x1FFFF) | 0x500000
          if not op.ignore_cache and self._cache.get(addr) == value: continue
          _add_req(struct.pack('>BBBHB', 0xE5, value, addr >> 16, addr & 0xFFFF, 0), 0, None)
          self._cache[addr] = value
      elif isinstance(op, ReadOp):
        assert op.size <= 0xff
        addr = (op.addr & 0x1FFFF) | 0x500000
        _add_req(struct.pack('>BBBHB', 0xE4, op.size, addr >> 16, addr & 0xFFFF, 0), op.size, None)
        for i in range(op.size): self._cache[addr + i] = None
      elif isinstance(op, ScsiWriteOp):
        sectors = round_up(len(op.data), 512) // 512
        _add_req(struct.pack('>BBQIBB', 0x8A, 0, op.lba, sectors, 0, 0), 0, op.data+b'\x00'*((sectors*512)-len(op.data)))

    return self.usb.send_batch(cdbs, idata, odata)

  def write(self, base_addr:int, data:bytes, ignore_cache:bool=True): return self.exec_ops([WriteOp(base_addr, data, ignore_cache)])

  def scsi_write(self, buf:bytes, lba:int=0):
    #chunk = 0x2000 if not self.usb.use_streams else 0x10000
    chunk = 512
    if len(buf) > 0x4000: buf += b'\x00' * (round_up(len(buf), chunk) - len(buf))

    for i in range(0, len(buf), chunk):
      self.exec_ops([WriteOp(0x7ef, b'\x00', ignore_cache=True)]) # re-arm SCSI write path
      self.exec_ops([ScsiWriteOp(buf[i:i+chunk], lba), WriteOp(0x171, b'\xff\xff\xff', ignore_cache=True)])
      self.exec_ops([WriteOp(0xce6e, b'\x00\x00', ignore_cache=True)])

    if len(buf) > 0x4000:
      for i in range(4): self.exec_ops([WriteOp(0xce40 + i, b'\x00', ignore_cache=True)])

  def read(self, base_addr:int, length:int, stride:int=0xff) -> bytes:
    parts = self.exec_ops([ReadOp(base_addr + off, min(stride, length - off)) for off in range(0, length, stride)])
    return b''.join(p or b'' for p in parts)[:length]

  def _is_pci_cacheable(self, addr:int) -> bool: return any(x <= addr <= x + sz for x, sz in self._pci_cacheable)
  def pcie_prep_request(self, fmt_type:int, address:int, value:int|None=None, size:int=4) -> list[WriteOp]:
    if fmt_type == 0x60 and size == 4 and self._is_pci_cacheable(address) and self._pci_cache.get(address) == value: return []

    assert fmt_type >> 8 == 0 and size > 0 and size <= 4, f"Invalid fmt_type {fmt_type} or size {size}"
    if DEBUG >= 5: print("pcie_request", hex(fmt_type), hex(address), value, size)

    masked_address, offset = address & 0xFFFFFFFC, address & 0x3
    assert size + offset <= 4 and (value is None or value >> (8 * size) == 0)
    self._pci_cache[address] = value if size == 4 and fmt_type == 0x60 else None

    return ([WriteOp(0xB220, struct.pack('>I', value << (8 * offset)), ignore_cache=False)] if value is not None else []) + \
      [WriteOp(0xB218, struct.pack('>I', masked_address), ignore_cache=False), WriteOp(0xB21c, struct.pack('>I', address>>32), ignore_cache=False),
       WriteOp(0xB217, bytes([((1 << size) - 1) << offset]), ignore_cache=False), WriteOp(0xB210, bytes([fmt_type]), ignore_cache=False),
       WriteOp(0xB254, b"\x0f", ignore_cache=True), WriteOp(0xB296, b"\x04", ignore_cache=True)]

  def pcie_request(self, fmt_type, address, value=None, size=4, cnt=10):
    self.exec_ops(self.pcie_prep_request(fmt_type, address, value, size))

    # Fast path for write requests
    if ((fmt_type & 0b11011111) == 0b01000000) or ((fmt_type & 0b10111000) == 0b00110000): return

    while (stat:=self.read(0xB296, 1)[0]) & 2 == 0:
      if stat & 1:
        self.write(0xB296, bytes([0x01]))
        if cnt > 0: return self.pcie_request(fmt_type, address, value, size, cnt=cnt-1)
    assert stat == 2, f"stat read 2 was {stat}"

    # Retrieve completion data from Link Status (0xB22A, 0xB22B)
    b284 = self.read(0xB284, 1)[0]
    completion = struct.unpack('>H', self.read(0xB22A, 2))

    # Validate completion status based on PCIe request typ
    # Completion TLPs for configuration requests always have a byte count of 4.
    assert completion[0] & 0xfff == (4 if (fmt_type & 0xbe == 0x04) else size)

    # Extract completion status field
    status = (completion[0] >> 13) & 0x7

    # Handle completion errors or inconsistencies
    if status or ((fmt_type & 0xbe == 0x04) and (((value is None) and (not (b284 & 0x01))) or ((value is not None) and (b284 & 0x01)))):
      status_map = {0b001: f"Unsupported Request: invalid address/function (target might not be reachable): {address:#x}",
                    0b100: "Completer Abort: abort due to internal error", 0b010: "Configuration Request Retry Status: configuration space busy"}
      raise RuntimeError(f"TLP status: {status_map.get(status, 'Reserved (0b{:03b})'.format(status))}")

    if value is None: return (struct.unpack('>I', self.read(0xB220, 4))[0] >> (8 * (address & 0x3))) & ((1 << (8 * size)) - 1)

  def pcie_cfg_req(self, byte_addr, bus=1, dev=0, fn=0, value=None, size=4):
    assert byte_addr >> 12 == 0 and bus >> 8 == 0 and dev >> 5 == 0 and fn >> 3 == 0, f"Invalid byte_addr {byte_addr}, bus {bus}, dev {dev}, fn {fn}"

    fmt_type = (0x44 if value is not None else 0x4) | int(bus > 0)
    address = (bus << 24) | (dev << 19) | (fn << 16) | (byte_addr & 0xfff)
    return self.pcie_request(fmt_type, address, value, size)

  def pcie_mem_req(self, address, value=None, size=4): return self.pcie_request(0x60 if value is not None else 0x20, address, value, size)

  def pcie_mem_write(self, address, values, size):
    ops = [self.pcie_prep_request(0x60, address + i * size, value, size) for i, value in enumerate(values)]

    # Send in batches of 4 for OSX and 16 for Linux (benchmarked values)
    for i in range(0, len(ops), bs:=(4 if OSX else 16)): self.exec_ops(list(itertools.chain.from_iterable(ops[i:i+bs])))

class USBMMIOInterface(MMIOInterface):
  def __init__(self, usb, addr, size, fmt, pcimem=True): # pylint: disable=super-init-not-called
    self.usb, self.addr, self.nbytes, self.fmt, self.pcimem, self.el_sz = usb, addr, size, fmt, pcimem, struct.calcsize(fmt)

  def __getitem__(self, index): return self._access_items(index)
  def __setitem__(self, index, val): self._access_items(index, val)

  def _access_items(self, index, val=None):
    if isinstance(index, slice): return self._acc((index.start or 0) * self.el_sz, ((index.stop or len(self))-(index.start or 0)) * self.el_sz, val)
    return self._acc_one(index * self.el_sz, self.el_sz, val) if self.pcimem else self._acc(index * self.el_sz, self.el_sz, val)

  def view(self, offset:int=0, size:int|None=None, fmt=None):
    return USBMMIOInterface(self.usb, self.addr+offset, size or (self.nbytes - offset), fmt=fmt or self.fmt, pcimem=self.pcimem)

  def _acc_size(self, sz): return next(x for x in [('I', 4), ('H', 2), ('B', 1)] if sz % x[1] == 0)

  def _acc_one(self, off, sz, val=None):
    upper = 0 if sz < 8 else self.usb.pcie_mem_req(self.addr + off + 4, val if val is None else (val >> 32), 4)
    lower = self.usb.pcie_mem_req(self.addr + off, val if val is None else val & 0xffffffff, min(sz, 4))
    if val is None: return lower | (upper << 32)

  def _acc(self, off, sz, data=None):
    if data is None: # read op
      if not self.pcimem:
        return int.from_bytes(self.usb.read(self.addr + off, sz), "little") if sz == self.el_sz else self.usb.read(self.addr + off, sz)

      acc, acc_size = self._acc_size(sz)
      return bytes(array.array(acc, [self._acc_one(off + i * acc_size, acc_size) for i in range(sz // acc_size)]))

    # write op
    data = struct.pack(self.fmt, data) if isinstance(data, int) else bytes(data)

    if not self.pcimem:
      # Fast path for writing into buffer 0xf000
      use_cache = 0xa800 <= self.addr <= 0xb000
      return self.usb.scsi_write(bytes(data)) if self.addr == 0xf000 else self.usb.write(self.addr + off, bytes(data), ignore_cache=not use_cache)

    _, acc_sz = self._acc_size(len(data) * struct.calcsize(self.fmt))
    self.usb.pcie_mem_write(self.addr+off, [int.from_bytes(data[i:i+acc_sz], "little") for i in range(0, len(data), acc_sz)], acc_sz)

if getenv("MOCKGPU"): from test.mockgpu.usb import MockUSB3 as USB3  # type: ignore  # noqa: F811

# =============================================================================
# USB2 controller — uses 0xF0 vendor command with streaming bulk for PCIe access
# =============================================================================

MWR64, MRD64, CFGRD0, CFGRD1, CFGWR0, CFGWR1 = 0x60, 0x20, 0x04, 0x05, 0x44, 0x45
EP_OUT, EP_IN = 0x02, 0x81
READ_CHUNK = 16  # dwords per bulk IN chunk

class USB2Controller:
  def __init__(self):
    self.ctx = ctypes.POINTER(libusb.struct_libusb_context)()
    if libusb.libusb_init(ctypes.byref(self.ctx)): raise RuntimeError("libusb_init failed")
    self.handle = libusb.libusb_open_device_with_vid_pid(self.ctx, 0xADD1, 0x0001)
    if not self.handle: raise RuntimeError("USB2 device ADD1:0001 not found")
    if libusb.libusb_kernel_driver_active(self.handle, 0): libusb.libusb_detach_kernel_driver(self.handle, 0)
    if libusb.libusb_set_configuration(self.handle, 1): raise RuntimeError("set_configuration failed")
    if libusb.libusb_claim_interface(self.handle, 0): raise RuntimeError("claim_interface failed")
    self._pci_cacheable: list[tuple[int, int]] = []
    self._pci_cache: dict[int, int|None] = {}

  # -- low-level xdata access (0xE4/0xE5) --
  def xdata_read(self, addr, size=1):
    buf = (ctypes.c_ubyte * size)()
    ret = libusb.libusb_control_transfer(self.handle, 0xC0, 0xE4, addr, 0, buf, size, 1000)
    assert ret >= 0, f"E4 read 0x{addr:04X} failed: {ret}"
    return bytes(buf[:ret])

  def xdata_write(self, addr, val):
    ret = libusb.libusb_control_transfer(self.handle, 0x40, 0xE5, addr, val, None, 0, 1000)
    assert ret >= 0, f"E5 write 0x{addr:04X}=0x{val:02X} failed: {ret}"

  def read(self, base_addr, length):
    return self.xdata_read(base_addr, length)

  def write(self, base_addr, data, ignore_cache=True):
    for i, b in enumerate(data): self.xdata_write(base_addr + i, b)

  # -- 0xF0 single TLP --
  def _f0_out(self, fmt_type, be, mode, count, addr_lo, addr_hi, value_be):
    wval = fmt_type | (be << 8)
    widx = (mode & 0x03) | ((count & 0x3F) << 2)
    payload = struct.pack('<II', addr_lo, addr_hi) + struct.pack('>I', value_be)
    buf = (ctypes.c_ubyte * 12)(*payload)
    ret = libusb.libusb_control_transfer(self.handle, 0x40, 0xF0, wval, widx, buf, 12, 5000)
    assert ret >= 0, f"F0 OUT failed: {ret}"

  def _f0_in(self):
    buf = (ctypes.c_ubyte * 8)()
    ret = libusb.libusb_control_transfer(self.handle, 0xC0, 0xF0, 0, 0, buf, 8, 5000)
    assert ret >= 0, f"F0 IN failed: {ret}"
    return bytes(buf)

  def _is_pci_cacheable(self, addr): return any(x <= addr <= x + sz for x, sz in self._pci_cacheable)

  def pcie_request(self, fmt_type, address, value=None, size=4, retries=10):
    if fmt_type == 0x60 and size == 4 and self._is_pci_cacheable(address) and self._pci_cache.get(address) == value: return None
    if DEBUG >= 5: print("usb2 pcie_request", hex(fmt_type), hex(address), value, size)
    self._pci_cache[address] = value if size == 4 and fmt_type == 0x60 else None

    masked, offset = address & 0xFFFFFFFC, address & 0x3
    be = ((1 << size) - 1) << offset
    shifted = ((value << (8 * offset)) & 0xFFFFFFFF) if value is not None else 0
    self._f0_out(fmt_type, be, 0, 0, masked & 0xFFFFFFFF, address >> 32, shifted)
    is_write = ((fmt_type & 0xDF) == 0x40) or ((fmt_type & 0xB8) == 0x30)
    if is_write: return None
    result = self._f0_in()
    fw_status = result[7]
    if fw_status == 0x01:
      if retries > 0: return self.pcie_request(fmt_type, address, value, size, retries - 1)
      raise RuntimeError(f"Unsupported Request at 0x{address:08X}")
    if fw_status == 0xFF: raise TimeoutError(f"PCIe completion timeout at 0x{address:08X}")
    raw = struct.unpack('>I', result[0:4])[0]
    return (raw >> (8 * offset)) & ((1 << (8 * size)) - 1)

  def pcie_cfg_req(self, byte_addr, bus=1, dev=0, fn=0, value=None, size=4):
    fmt = (CFGWR1 if value is not None else CFGRD1) if bus > 0 else (CFGWR0 if value is not None else CFGRD0)
    address = (bus << 24) | (dev << 19) | (fn << 16) | (byte_addr & 0xFFF)
    return self.pcie_request(fmt, address, value, size)

  def pcie_mem_req(self, address, value=None, size=4):
    return self.pcie_request(MWR64 if value is not None else MRD64, address, value, size)

  # -- streaming bulk (mode 1=write, mode 2=read) --
  def _dma_setup(self, addr, mode, count=0):
    fmt = {0: 0, 1: MWR64, 2: MRD64}[mode]
    self._f0_out(fmt, 0x0F, mode, count, addr & 0xFFFFFFFF, addr >> 32, 0)

  def stream_write(self, addr, data):
    """Streaming write to PCIe address via bulk OUT."""
    import time
    self._dma_setup(addr, 1)
    buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    transferred = ctypes.c_int()
    ret = libusb.libusb_bulk_transfer(self.handle, EP_OUT, buf, len(data), ctypes.byref(transferred), 30000)
    assert ret == 0, f"bulk write failed: {ret}"
    time.sleep(0.001)

  def stream_read(self, addr, nbytes, chunk=READ_CHUNK):
    """Streaming read from PCIe address via bulk IN."""
    self._dma_setup(addr, 2, chunk)
    chunk_bytes = chunk * 4
    resp = (ctypes.c_ubyte * chunk_bytes)()
    result = bytearray()
    transferred = ctypes.c_int()
    while len(result) < nbytes:
      ret = libusb.libusb_bulk_transfer(self.handle, EP_IN, resp, chunk_bytes, ctypes.byref(transferred), 5000)
      assert ret == 0, f"bulk read failed: {ret}"
      result.extend(bytes(resp[:transferred.value]))
    return bytes(result[:nbytes])

class USB2MMIOInterface(MMIOInterface):
  def __init__(self, usb:USB2Controller, addr:int, size:int, fmt='B', pcimem=True):
    self.usb, self.addr, self.nbytes, self.fmt, self.pcimem = usb, addr, size, fmt, pcimem
    self.el_sz = struct.calcsize(fmt)

  def __getitem__(self, index): return self._access(index)
  def __setitem__(self, index, val): self._access(index, val)

  def view(self, offset=0, size=None, fmt=None):
    return USB2MMIOInterface(self.usb, self.addr + offset, size or (self.nbytes - offset), fmt=fmt or self.fmt, pcimem=self.pcimem)

  def _access(self, index, val=None):
    if isinstance(index, slice):
      start, stop = (index.start or 0) * self.el_sz, (index.stop or len(self)) * self.el_sz
      return self._acc_range(start, stop - start, val)
    return self._acc_one(index * self.el_sz, self.el_sz, val) if self.pcimem else self._acc_range(index * self.el_sz, self.el_sz, val)

  def _acc_one(self, off, sz, val=None):
    """Single dword access via single TLP."""
    upper = 0 if sz < 8 else self.usb.pcie_mem_req(self.addr + off + 4, val if val is None else (val >> 32), 4)
    lower = self.usb.pcie_mem_req(self.addr + off, val if val is None else val & 0xffffffff, min(sz, 4))
    if val is None: return lower | (upper << 32)

  def _acc_range(self, off, sz, data=None):
    """Range access — uses streaming bulk for large PCIe transfers, single TLP for small/non-PCIe."""
    if data is None:  # read
      if not self.pcimem:
        raw = self.usb.xdata_read(self.addr + off, sz)
        return int.from_bytes(raw, "little") if sz == self.el_sz else raw
      if sz >= 64:  # streaming read for large transfers
        raw = self.usb.stream_read(self.addr + off, sz)
        # Convert from big-endian dwords to host byte order
        arr = array.array('I')
        arr.frombytes(raw[:len(raw) - len(raw) % 4])
        arr.byteswap()
        return bytes(arr)
      # small read: use single TLPs
      acc_sz = 4 if sz % 4 == 0 else (2 if sz % 2 == 0 else 1)
      return bytes(array.array('I' if acc_sz == 4 else ('H' if acc_sz == 2 else 'B'),
                               [self._acc_one(off + i * acc_sz, acc_sz) for i in range(sz // acc_sz)]))
    # write
    data = struct.pack(self.fmt, data) if isinstance(data, int) else bytes(data)
    if not self.pcimem:
      for i, b in enumerate(data): self.usb.xdata_write(self.addr + off + i, b)
      return
    if len(data) >= 64:  # streaming write for large transfers
      # Convert from host byte order to big-endian dwords
      arr = array.array('I')
      arr.frombytes(data + b'\x00' * ((-len(data)) % 4))
      arr.byteswap()
      self.usb.stream_write(self.addr + off, arr.tobytes()[:len(data)])
      return
    # small write: use single TLPs
    acc_sz = 4 if len(data) % 4 == 0 else (2 if len(data) % 2 == 0 else 1)
    for i in range(0, len(data), acc_sz):
      self.usb.pcie_mem_req(self.addr + off + i, int.from_bytes(data[i:i+acc_sz], "little"), acc_sz)
