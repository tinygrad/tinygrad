import ctypes, struct, time, os
from tinygrad.runtime.autogen import libc, libusb
from tinygrad.helpers import DEBUG
from hexdump import hexdump

class USBConnector:
  def __init__(self, name):
    self.usb_ctx = ctypes.POINTER(libusb.struct_libusb_context)()
    ret = libusb.libusb_init(ctypes.byref(self.usb_ctx))
    if ret != 0: raise Exception(f"Failed to init libusb: {ret}")

    # Open device
    self.handle = libusb.libusb_open_device_with_vid_pid(self.usb_ctx, 0x174c, 0x2463)
    if not self.handle: raise Exception("Failed to open device")

    # Detach kernel driver if needed
    if libusb.libusb_kernel_driver_active(self.handle, 0) == 1:
      ret = libusb.libusb_detach_kernel_driver(self.handle, 0)
      print("detach kernel driver")
      if ret != 0: raise Exception(f"Failed to detach kernel driver: {ret}")
      libusb.libusb_reset_device(self.handle)

    # Claim interface (gives -3 if we reset)
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

    # required to be set, but not a trigger
    self.write(0xB213, bytes([0x01]))
    self.write(0xB214, bytes([0, 0]))
    self.write(0xB216, bytes([0x20])) # Enable PCIe interface features (bus master, etc.)

  def _detect_version(self):
    try: self.read(0x07f0, 6)
    except Exception: pass

  def _send(self, cdb, ret_len=0):
    def __send():
      actual_length = ctypes.c_int(0)
      for i in range(3):
        #assert len(self.read_cmd) == 31
        ret = libusb.libusb_bulk_transfer(self.handle, 0x04, self.read_cmd, len(self.read_cmd), ctypes.byref(actual_length), 1)
        assert actual_length.value == len(self.read_cmd)
        if ret:
          print(i, "0x4", ret, len(self.read_cmd))
          return None

        if ret_len > 0:
          # wait for data ready
          ret = libusb.libusb_bulk_transfer(self.handle, 0x83, self.read_status, 64, ctypes.byref(actual_length), 1)
          if ret != 0 or self.read_status[0] != 6:
            print(f"WEIRD READ of {actual_length} with ret {ret}")
            hexdump(bytes(self.read_status))
            print("while sending")
            hexdump(bytes(self.read_cmd))
          assert ret == 0 and self.read_status[0] == 6  # 6 means ready to read

          # get data
          ret = libusb.libusb_bulk_transfer(self.handle, 0x81, self.read_data, ret_len, ctypes.byref(actual_length), 1)
          if ret:
            print(i, "0x81", ret, ret_len)
            continue
          assert actual_length.value == ret_len, f"only sent {actual_length.value}, wanted {ret_len}"

        # get status
        ret = libusb.libusb_bulk_transfer(self.handle, 0x83, self.read_status, 64, ctypes.byref(actual_length), 1)
        assert ret == 0 and self.read_status[0] == 3, f"got ret {ret} with length {actual_length}"

        # return a copy of the bytes
        return bytes(self.read_data[:])
      return None

    self.read_cmd[4:6] = len(cdb).to_bytes(2, 'big')
    self.read_cmd[16:16+len(cdb)] = cdb
    for j in range(1):
      #print(j)
      read_data = __send()
      if read_data: return read_data
    raise RuntimeError("USB transfer failed")

  def read(self, start_addr, read_len, stride=255):
    if DEBUG >= 2: print("read", hex(start_addr))
    data = bytearray(read_len)

    for i in range(0, read_len, stride):
      remaining = read_len - i
      buf_len = min(stride, remaining)
      current_addr = (start_addr + i)
      assert current_addr >> 17 == 0
      current_addr &= 0x01ffff
      current_addr |= 0x500000
      cdb = struct.pack('>BBBHB', 0xe4, buf_len, current_addr >> 16, current_addr & 0xffff, 0x00)
      data[i:i+buf_len] = self._send(cdb, buf_len)
    return bytes(data[:read_len])

  def write(self, start_addr, data):
    if DEBUG >= 2: print("write", hex(start_addr))
    for offset, value in enumerate(data):
      current_addr = start_addr + offset
      assert current_addr >> 17 == 0
      current_addr &= 0x01ffff
      current_addr |= 0x500000
      cdb = struct.pack('>BBBHB', 0xe5, value, current_addr >> 16, current_addr & 0xffff, 0x00)
      self._send(cdb)

  def pcie_write_request_fw(self, address, value):
    cdb = struct.pack('>BII', 0x03, address, value)
    self._send(cdb)

  def pcie_request(self, fmt_type, address, value=None, size=4, cnt=10):
    assert fmt_type >> 8 == 0
    assert size > 0 and size <= 4
    if DEBUG >= 1: print("pcie_request", hex(fmt_type), hex(address), value, size, cnt)

    # TODO: why is this needed?
    #time.sleep(0.005)

    # TODO: why is this needed? (the write doesn't matter, just that it's using USB)
    #self.write(0xB210, bytes([0]))
    #self.write(0xB210, bytes([0]))
    #self.write(0xB210, bytes([0]))
    #self.write(0xB210, bytes([0]))

    #print(self.read(0xB296, 1)[0])

    masked_address = address & 0xfffffffc
    offset = address & 0x00000003

    assert size + offset <= 4

    byte_enable = ((1 << size) - 1) << offset

    if value is not None:
      assert value >> (8 * size) == 0, f"{value}"
      shifted_value = value << (8 * offset)
      # Store write data in PCIE_CONFIG_DATA register (0xB220)
      self.write(0xB220, struct.pack('>I', value << (8 * offset)))

    # setup address + length
    self.write(0xB218, struct.pack('>I', masked_address))
    assert byte_enable < 0x100
    self.write(0xB217, bytes([byte_enable]))

    # Configure PCIe request by writing to PCIE_REQUEST_CONTROL (0xB210)
    self.write(0xB210, bytes([fmt_type]))

    # Clear PCIe completion timeout status in PCIE_STATUS_REGISTER (0xB296)
    self.write(0xB296, bytes([0x04]))

    # Clear any existing PCIe errors before proceeding (PCIE_ERROR_CLEAR: 0xB254)
    # this appears to be the trigger
    self.write(0xB254, bytes([0x0f]))

    # Wait for PCIe transaction to complete (PCIE_STATUS_REGISTER: 0xB296, bit 2)
    while (stat:=self.read(0xB296, 1)[0]) & 4 == 0:
      print("stat early poll", stat)
      continue
    assert stat == 6, f"stat was {stat}"
    #print("stat out", stat)

    # Acknowledge completion of PCIe request (PCIE_STATUS_REGISTER: 0xB295)
    self.write(0xB296, bytes([0x04]))

    if ((fmt_type & 0b11011111) == 0b01000000) or ((fmt_type & 0b10111000) == 0b00110000):
      assert False, "not supported"

    while (stat:=self.read(0xB296, 1)[0]) & 2 == 0:
      print("stat poll", stat)
      if stat & 1:
        self.write(0xB296, bytes([0x01]))
        print("pci redo")
        if cnt > 0: self.pcie_request(fmt_type, address, value, size, cnt=cnt-1)
    assert stat == 2, f"stat read 2 was {stat}"

    # Acknowledge PCIe completion (PCIE_STATUS_REGISTER: 0xB296)
    self.write(0xB296, bytes([0x02]))

    b284 = self.read(0xB284, 1)[0]
    b284_bit_0 = b284 & 0x01

    # Retrieve completion data from Link Status (0xB22A, 0xB22B)
    completion = struct.unpack('>H', self.read(0xB22A, 2))

    # Validate completion status based on PCIe request typ
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

    # Extract completion status field
    status = (completion[0] >> 13) & 0x7

    # Handle completion errors or inconsistencies
    if status or ((fmt_type & 0xbe == 0x04) and (((value is None) and (not b284_bit_0)) or ((value is not None) and b284_bit_0))):
      raise Exception("Completion status: {}, 0xB284 bit 0: {}".format(
        status_map.get(status, "Reserved (0b{:03b})".format(status)), b284_bit_0))

    if value is None:
      # Read from PCIE_CONFIG_DATA (0xB220)
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
