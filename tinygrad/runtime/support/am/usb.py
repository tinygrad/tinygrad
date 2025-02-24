import ctypes, struct, time, os
from tinygrad.runtime.autogen import libc, libusb

class USBConnector:
  def __init__(self, name):
    self.usb_ctx = ctypes.POINTER(libusb.struct_libusb_context)()
    ret = libusb.libusb_init(ctypes.byref(self.usb_ctx))
    if ret != 0: raise Exception(f"Failed to init libusb: {ret}")

    # Open device
    self.handle = libusb.libusb_open_device_with_vid_pid(self.usb_ctx, 0x174c, 0x2464)
    if not self.handle: raise Exception("Failed to open device")

    # Detach kernel driver if needed
    if libusb.libusb_kernel_driver_active(self.handle, 0) == 1:
      ret = libusb.libusb_detach_kernel_driver(self.handle, 0)
      if ret != 0: raise Exception(f"Failed to detach kernel driver: {ret}")

    libusb.libusb_reset_device(self.handle)

    # Claim interface
    ret = libusb.libusb_claim_interface(self.handle, 0)
    if ret != 0: raise Exception(f"Failed to claim interface: {ret}")

    # # Set alternate setting to 1 (this is crucial!)
    ret = libusb.libusb_set_interface_alt_setting(self.handle, 0, 1)
    if ret != 0: raise Exception(f"Failed to set alternate setting: {ret}")

    usb_cmd = [
      0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xe4, 0x24, 0x00, 0xb2, 0x1a, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    ]
    self.read_cmd = (ctypes.c_uint8 * len(usb_cmd))(*bytes(usb_cmd))
    self.read_status = (ctypes.c_uint8 * 64)()
    self.read_data = (ctypes.c_uint8 * 112)()

    print("USB device initialized successfully")
    self._detect_version()

  def _detect_version(self):
    self.is_24 = False
    try: self.read(0x07f0, 6)
    except Exception: self.is_24 = True

  def _send(self, cdb, ret_len=0):
    def __send():
      ret = libusb.libusb_bulk_transfer(self.handle, 0x04, self.read_cmd, len(self.read_cmd), None, 1)
      if ret: return None

      if ret_len > 0:
        ret = libusb.libusb_bulk_transfer(self.handle, 0x81, self.read_data, ret_len, None, 1)
        if ret: return None

      ret = libusb.libusb_bulk_transfer(self.handle, 0x83, self.read_status, 64, None, 1)
      if ret: return None
      return self.read_data

    self.read_cmd[16:16+len(cdb)] = cdb
    for j in range(1000):
      read_data = __send()
      if read_data: return read_data
    raise RuntimeError("USB transfer failed")

  def read(self, start_addr, read_len, stride=255):
    data = bytearray(read_len)

    for i in range(0, read_len, stride):
      remaining = read_len - i
      buf_len = min(stride, remaining)
      current_addr = (start_addr + i)
      if self.is_24:
        assert current_addr >> 17 == 0
        current_addr &= 0x01ffff
        current_addr |= 0x500000
        cdb = struct.pack('>BBBHB', 0xe4, buf_len, current_addr >> 16, current_addr & 0xffff, 0x00)
      else: cdb = struct.pack('>BBBHB', 0xe4, buf_len, 0x00, current_addr, 0x00)
      data[i:i+buf_len] = self._send(cdb, buf_len)
    return bytes(data[:read_len])

  def write(self, start_addr, data):
    for offset, value in enumerate(data):
      current_addr = start_addr + offset
      if self.is_24:
        assert current_addr >> 17 == 0
        current_addr &= 0x01ffff
        current_addr |= 0x500000
        cdb = struct.pack('>BBBHB', 0xe5, value, current_addr >> 16, current_addr & 0xffff, 0x00)
      else: cdb = struct.pack('>BBBHB', 0xe5, value, 0x00, current_addr, 0x00)
      self._send(cdb)

  def pcie_write_request_fw(self, address, value):
    cdb = struct.pack('>BII', 0x03, address, value)
    self._send(cdb)

  def pcie_request(self, fmt_type, address, value=None, size=4, cnt=10):
    assert fmt_type >> 8 == 0
    assert size > 0 and size <= 4

    masked_address = address & 0xfffffffc
    offset = address & 0x00000003

    assert size + offset <= 4

    byte_enable = ((1 << size) - 1) << offset

    if value is not None:
      assert value >> (8 * size) == 0, f"{value}"
      shifted_value = value << (8 * offset)
      self.write(0xB220, struct.pack('>I', value << (8 * offset)))

    self.write(0xB210, struct.pack('>III', 0x00000001 | (fmt_type << 24), byte_enable, masked_address))
 
    if self.is_24:
      self.write(0xB216, bytes([0x20]))
      self.write(0xB296, bytes([0x04]))
    else:
      # Clear timeout bit.
      self.write(0xB296, bytes([0x01]))

    self.write(0xB254, bytes([0x0f]))

    # while self.read(0xB296, 1)[0] & 4 == 0: continue

    self.write(0xB296, bytes([0x04]))

    prep_st = time.perf_counter_ns()

    if ((fmt_type & 0b11011111) == 0b01000000) or ((fmt_type & 0b10111000) == 0b00110000): return

    while self.read(0xB296, 1)[0] & 2 == 0:
      if self.read(0xB296, 1)[0] & 1:
        self.write(0xB296, bytes([0x01]))
        print("pci redo")
        if cnt > 0: self.pcie_request(fmt_type, address, value, size, cnt=cnt-1)

    self.write(0xB296, bytes([0x02]))

    b284 = self.read(0xB284, 1)[0]
    b284_bit_0 = b284 & 0x01

    completion = struct.unpack('>I', self.read(0xB228, 4))

    if (fmt_type & 0xbe == 0x04):
      # Completion TLPs for configuration requests always have a byte count of 4.
      assert completion[0] & 0xfff == 4
    else:
      assert completion[0] & 0xfff == size

    status_map = {
      0b000: "Successful Completion (SC)",
      0b001: "Unsupported Request (UR)",
      0b010: "Configuration Request Retry Status (CRS)",
      0b100: "Completer Abort (CA)",
    }

    status = (completion[0] >> 13) & 0x7
    if status or ((fmt_type & 0xbe == 0x04) and (((value is None) and (not b284_bit_0)) or ((value is not None) and b284_bit_0))):
      raise Exception("Completion status: {}, 0xB284 bit 0: {}".format(
        status_map.get(status, "Reserved (0b{:03b})".format(status)), b284_bit_0))

    if value is None:
      full_value = struct.unpack('>I', self.read(0xB220, 4))[0]
      shifted_value = full_value >> (8 * offset)
      masked_value = shifted_value & ((1 << (8 * size)) - 1)
      return masked_value

  def pcie_cfg_req(self, byte_addr, bus=1, dev=0, fn=0, value=None, size=4):
    assert byte_addr >> 12 == 0

    assert bus >> 8 == 0
    assert dev >> 5 == 0
    assert fn >> 3 == 0

    cfgreq_type = int(bus > 0)
    assert cfgreq_type >> 1 == 0

    fmt_type = 0x04
    if value is not None: fmt_type = 0x44

    fmt_type |= cfgreq_type
    address = (bus << 24) | (dev << 19) | (fn << 16) | (byte_addr & 0xfff)

    return self.pcie_request(fmt_type, address, value, size)

  def pcie_mem_req(self, address, value=None, size=4):
    fmt_type = 0x00
    if value is not None: fmt_type = 0x40

    return self.pcie_request(fmt_type, address, value, size)

class SCSIConnector(USBConnector):
  def __init__(self, name):
    self._file = os.fdopen(os.open(name, os.O_RDWR | os.O_NONBLOCK))
    self._detect_version()

  def _send(self, cdb, ret_len=0):
    import sgio
    buf = bytearray(ret_len) if ret_len > 0 else None
    ret = sgio.execute(self._file, cdb, None, buf)
    assert ret == 0
    return buf
