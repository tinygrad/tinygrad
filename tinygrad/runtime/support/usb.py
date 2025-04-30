import ctypes, struct, time, os, dataclasses
from tinygrad.runtime.autogen import libusb
from tinygrad.helpers import DEBUG
from tinygrad.runtime.support.hcq import MMIOInterface

class USB3:
  def __init__(self, vendor:int, dev:int, ep_data_in:int, ep_stat_in:int, ep_data_out:int, ep_cmd_out:int, max_streams:int=24, max_read_len:int=0x1000):
    self.vendor, self.dev = vendor, dev
    self.ep_data_in, self.ep_stat_in, self.ep_data_out, self.ep_cmd_out = ep_data_in, ep_stat_in, ep_data_out, ep_cmd_out
    self.max_streams, self.max_read_len = max_streams, max_read_len
    self.ctx = ctypes.POINTER(libusb.struct_libusb_context)()

    if libusb.libusb_init(ctypes.byref(self.ctx)): raise RuntimeError("libusb_init failed")
    if DEBUG >= 6: libusb.libusb_set_option(self.ctx, libusb.LIBUSB_OPTION_LOG_LEVEL, 4)

    self.handle = libusb.libusb_open_device_with_vid_pid(self.ctx, 0x2D01, 0x3666)
    if not self.handle: raise RuntimeError("device not found")

    # Detach kernel driver if needed
    if libusb.libusb_kernel_driver_active(self.handle, 0):
      libusb.libusb_detach_kernel_driver(self.handle, 0)
      libusb.libusb_reset_device(self.handle)

    # Set configuration and claim interface
    if libusb.libusb_set_configuration(self.handle, 1): raise RuntimeError("set_configuration failed")
    if libusb.libusb_claim_interface(self.handle, 0): raise RuntimeError("claim_interface failed")
    if libusb.libusb_set_interface_alt_setting(self.handle, 0, 1): raise RuntimeError("alt_setting failed")

    # Clear any stalled endpoints
    all_eps = (self.ep_data_out, self.ep_data_in, self.ep_stat_in, self.ep_cmd_out)
    for ep in all_eps: libusb.libusb_clear_halt(self.handle, ep)

    # Allocate streams
    stream_eps = (ctypes.c_uint8 * 3)(self.ep_data_out, self.ep_data_in, self.ep_stat_in)
    if (rc:=libusb.libusb_alloc_streams(self.handle, self.max_streams * len(stream_eps), stream_eps, len(stream_eps))) < 0:
      raise RuntimeError(f"alloc_streams failed: {rc}")

    # Base cmd
    cmd_template = bytes([0x01, 0x00, 0x00, 0x01, *([0] * 12), 0xE4, 0x24, 0x00, 0xB2, 0x1A, 0x00, 0x00, 0x00, *([0] * 8)])

    # Init pools
    self.tr = {ep: [libusb.libusb_alloc_transfer(0) for _ in range(self.max_streams)] for ep in all_eps}

    self.buf_cmd = [(ctypes.c_uint8 * len(cmd_template))(*cmd_template) for _ in range(self.max_streams)]
    self.buf_stat = [(ctypes.c_uint8 * 64)() for _ in range(self.max_streams)]
    self.buf_data_in = [(ctypes.c_uint8 * 0x1000)() for _ in range(self.max_streams)]
    self.buf_data_out = [(ctypes.c_uint8 * 0x1000)() for _ in range(self.max_streams)]

  def _prep_transfer(self, tr, ep, stream_id, buf, length):
    tr.contents.dev_handle, tr.contents.endpoint, tr.contents.length, tr.contents.buffer = self.handle, ep, length, buf
    tr.contents.status, tr.contents.flags, tr.contents.timeout = 0xff, 0, 1000
    tr.contents.type = (libusb.LIBUSB_TRANSFER_TYPE_BULK_STREAM if stream_id is not None else libusb.LIBUSB_TRANSFER_TYPE_BULK)
    tr.contents.num_iso_packets = 0
    if stream_id is not None: libusb.libusb_transfer_set_stream_id(tr, stream_id)

  def _submit_and_wait(self, cmds):
    for tr in cmds: libusb.libusb_submit_transfer(tr)

    while True:
      libusb.libusb_handle_events(self.ctx)
      ready = True
      for tr in cmds:
        if tr.contents.status == libusb.LIBUSB_TRANSFER_COMPLETED: continue
        if tr.contents.status != 0xFF: raise RuntimeError(f"EP 0x{tr.contents.endpoint:02X} error: {tr.contents.status}")
        ready = False
      if ready: return

  def send_batch(self, cdbs:list[bytes], idata:list[int]|None=None, odata:list[bytes|None]|None=None) -> list[bytes|None]:
    if idata is None: idata = [0] * len(cdbs)
    if odata is None: odata = [None] * len(cdbs)
    if len(cdbs) != len(idata): raise ValueError("cdbs and idata length mismatch")

    results: list[bytes|None] = [None] * len(cdbs)
    window: list[tuple[int, int, int]] = []  # (idx, slot, rlen)
    pending = []

    def _flush():
      nonlocal pending, window
      if not pending: return
      self._submit_and_wait(pending)
      for idx, slot, rlen in window:
        if rlen: results[idx] = bytes(self.buf_data_in[slot][:rlen])
        pending, window = [], []

    for idx, (cdb, rlen, send_data) in enumerate(zip(cdbs, idata, odata)):
      slot = idx % self.max_streams
      stream = slot + 1  # firmware convention

      # build cmd packet
      struct.pack_into(">BH", self.buf_cmd[slot], 3, stream, len(cdb))
      self.buf_cmd[slot][16:16+len(cdb)] = cdb

      # cmd + stat transfers
      self._prep_transfer(self.tr[self.ep_cmd_out][slot], self.ep_cmd_out, None, self.buf_cmd[slot], len(self.buf_cmd[slot]))
      self._prep_transfer(self.tr[self.ep_stat_in][slot], self.ep_stat_in, stream, self.buf_stat[slot], 64)
      pending += [self.tr[self.ep_stat_in][slot], self.tr[self.ep_cmd_out][slot]]

      if rlen:
        if rlen > self.max_read_len: raise ValueError("read length > max_read_len per CDB")
        self._prep_transfer(self.tr[self.ep_data_in][slot], self.ep_data_in, stream, self.buf_data_in[slot], rlen)
        pending.append(self.tr[self.ep_data_in][slot])
      
      if send_data is not None:
        self.buf_data_out[slot][:len(send_data)] = send_data
        self._prep_transfer(self.tr[self.ep_data_out][slot], self.ep_data_out, stream, self.buf_data_out[slot], 4096)
        pending.append(self.tr[self.ep_data_out][slot])

      window.append((idx, slot, rlen))
      if (idx + 1 == len(cdbs)) or len(window) >= self.max_streams: _flush()

    return results

@dataclasses.dataclass(frozen=True)
class WriteOp: addr:int; data:bytes; ignore_cache:bool=True # noqa: E702

@dataclasses.dataclass(frozen=True)
class ReadOp: addr:int; size:int; # noqa: E702

@dataclasses.dataclass(frozen=True)
class ScsiWriteOp: data:bytes; lba:int=0; # noqa: E702

class ASMController:
  def __init__(self):
    self.usb = USB3(0x2D01, 0x3666, 0x81, 0x83, 0x02, 0x04)
    self._cache: dict[int, int] = {}

    # Init controller.
    self.exec_ops([WriteOp(0x54b, b' ', ignore_cache=True), WriteOp(0x5a8, b'\x02', ignore_cache=True), WriteOp(0x5f8, b'\x04', ignore_cache=True),
      WriteOp(0x7ec, b'\x01\x00\x00\x00', ignore_cache=True), WriteOp(0xc422, b'\x02', ignore_cache=True), WriteOp(0x0, b'\x33', ignore_cache=True)])

  def ops_to_cmd(self, ops:list[WriteOp|ReadOp], _add_req:callable):
    for op in ops:
      if isinstance(op, WriteOp):
        for off, value in enumerate(op.data):
          addr = ((op.addr + off) & 0x1FFFF) | 0x500000
          if not op.ignore_cache and self._cache.get(addr) == value: continue
          _add_req(struct.pack('>BBBHB', 0xE5, value, addr >> 16, addr & 0xFFFF, 0), None, None)
          self._cache[addr] = value
      elif isinstance(op, ReadOp):
        assert op.size <= 0xff
        addr = (op.addr & 0x1FFFF) | 0x500000
        _add_req(struct.pack('>BBBHB', 0xE4, op.size, addr >> 16, addr & 0xFFFF, 0), op.size, None)
        for i in range(op.size): self._cache[addr + i] = None
      elif isinstance(op, ScsiWriteOp): _add_req(struct.pack('>BBQIBB', 0x8A, 0, op.lba, 4096//512, 0, 0), None, op.data)

  def exec_ops(self, ops:list[WriteOp|ReadOp]):
    cdbs, idata, odata = [], [], []
    def _add_req(cdb, i, o):
      nonlocal cdbs, idata, odata
      cdbs, idata, odata = cdbs + [cdb], idata + [i], odata + [o]

    self.ops_to_cmd(ops, _add_req)
    return self.usb.send_batch(cdbs, idata, odata)

  def write(self, base_addr:int, data:bytes, ignore_cache:bool=True): return self.exec_ops([WriteOp(base_addr, data, ignore_cache)])

  def scsi_write(self, buf:bytes, lba:int=0):
    self.exec_ops([ScsiWriteOp(buf, lba), WriteOp(0x171, b'\xff\xff\xff', ignore_cache=True), WriteOp(0xce6e, b'\x00\x00', ignore_cache=True)])

  def read(self, base_addr:int, length:int, stride:int=0xff) -> bytes:
    parts = self.exec_ops([ReadOp(base_addr + off, min(stride, length - off)) for off in range(0, length, stride)])
    return b''.join(p or b'' for p in parts)[:length]

  def pcie_request(self, fmt_type, address, value=None, size=4, cnt=10):
    assert fmt_type >> 8 == 0 and size > 0 and size <= 4, f"Invalid fmt_type {fmt_type} or size {size}"
    if DEBUG >= 3: print("pcie_request", hex(fmt_type), hex(address), value, size, cnt)

    masked_address, offset = address & 0xFFFFFFFC, address & 0x3
    assert size + offset <= 4

    ops: List[Op] = []
    if value is not None:
      assert value >> (8 * size) == 0
      ops.append(WriteOp(0xB220, struct.pack('>I', value << (8 * offset)), ignore_cache=False))

    ops += [WriteOp(0xB218, struct.pack('>I', masked_address), ignore_cache=False),
      WriteOp(0xB217, bytes([((1 << size) - 1) << offset]), ignore_cache=False),
      WriteOp(0xB210, bytes([fmt_type]), ignore_cache=False),
      WriteOp(0xB254, b"\x0f", ignore_cache=True), WriteOp(0xB296, b"\x04", ignore_cache=True)]
    self.exec_ops(ops)

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
      status_map = {0b000: "Successful Completion (SC)", 0b001: "Unsupported Request (UR)",
                    0b010: "Configuration Request Retry Status (CRS)", 0b100: "Completer Abort (CA)"}
      raise RuntimeError("Completion status: {}, 0xB284 bit 0: {}".format(status_map.get(status, "Reserved (0b{:03b})".format(status)), b284 & 0x01))

    if value is None: return (struct.unpack('>I', self.read(0xB220, 4))[0] >> (8 * offset)) & ((1 << (8 * size)) - 1)

  def pcie_cfg_req(self, byte_addr, bus=1, dev=0, fn=0, value=None, size=4):
    assert byte_addr >> 12 == 0 and bus >> 8 == 0 and dev >> 5 == 0 and fn >> 3 == 0, f"Invalid byte_addr {byte_addr}, bus {bus}, dev {dev}, fn {fn}"

    fmt_type = (0x44 if value is not None else 0x4) | int(bus > 0)
    address = (bus << 24) | (dev << 19) | (fn << 16) | (byte_addr & 0xfff)
    return self.pcie_request(fmt_type, address, value, size)

  def pcie_mem_req(self, address, value=None, size=4): return self.pcie_request(0x40 if value is not None else 0x0, address, value, size)

class USBMMIOInterface(MMIOInterface):
  def __init__(self, usb, addr, size, fmt, pci_spc=True):
    self.usb, self.addr, self.nbytes, self.fmt, self.pci_spc = usb, addr, size, fmt, pci_spc
    self.el_sz = struct.calcsize(self.fmt)
  def __getitem__(self, index):
    if isinstance(index, slice): return self._read((index.start or 0) * self.el_sz, ((index.stop or len(self)) - (index.start or 0)) * self.el_sz)
    if isinstance(index, int): return self._acc_one(index * self.el_sz, self.el_sz) if self.pci_spc else self._read(index * self.el_sz, self.el_sz)[0]
  def __setitem__(self, index, val):
    if isinstance(index, slice): return self._write((index.start or 0) * self.el_sz, ((index.stop or len(self)) - (index.start or 0)) * self.el_sz, val)
    if isinstance(index, int): self._acc_one(index * self.el_sz, self.el_sz, val) if self.pci_spc else self._write(index * self.el_sz, self.el_sz, val)

  def view(self, offset:int=0, size:int|None=None, fmt=None) -> MMIOInterface:
    return USBMMIOInterface(self.usb, self.addr+offset, size or (self.nbytes - offset), fmt=fmt or self.fmt, pci_spc=self.pci_spc)

  def _acc_size(self, sz): return next(x for x in [('I', 4), ('H', 2), ('B', 1)] if sz % x[1] == 0)
  def _acc_one(self, off, sz, val=None):
    upper = 0 if sz < 8 else self.usb.pcie_mem_req(self.addr + off + 4, val if val is None else (val >> 32), 4)
    lower = self.usb.pcie_mem_req(self.addr + off, val if val is None else val & 0xffffffff, min(sz, 4)) 
    if val is None: return lower | (upper << 32)

  def _convert(self, raw:list[int], from_t, to_t) -> list[int]:
    # normalize bytes → list of ints
    vals = list(raw) if isinstance(raw, (bytes, bytearray)) else raw
    if from_t == to_t:
        return vals

    # pack all from_t-values into one byte blob
    packed = struct.pack('<' + from_t * len(vals), *vals)

    # figure out how many to_t values fit in there
    to_size = struct.calcsize(to_t)
    count   = len(packed) // to_size

    # unpack as count×to_t
    return list(struct.unpack(f'<{count}{to_t}', packed))

  def _read(self, offset, size):
    if not self.pci_spc: return self.usb.read(self.addr + offset, size)

    acc, acc_size = self._acc_size(size)
    return self._convert([self._acc_one(offset + i * acc_size, acc_size) for i in range(size // acc_size)], acc, 'B')

  def _write(self, offset, _, data):
    if not self.pci_spc:
      if isinstance(data, int): data = struct.pack(self.fmt, data)
      if self.addr == 0xf000: return self.usb.scsi_write(bytes(data))
      return self.usb.write(self.addr + offset, bytes(data))

    acc, acc_size = self._acc_size(len(data) * struct.calcsize(self.fmt))
    per_slice = acc_size // self.el_sz
    assert per_slice > 0

    for i in range(len(data) // per_slice):
      self._acc_one(offset + i * acc_size, acc_size, self._convert(data[i * per_slice:(i + 1) * per_slice], self.fmt, acc)[0])
