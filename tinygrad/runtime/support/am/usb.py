import ctypes, struct, time
from tinygrad.runtime.autogen import libc, libusb

class Asm236x:
  def __init__(self, name):
    print("Opening", name)
    self.usb_ctx = ctypes.POINTER(libusb.struct_libusb_context)()
    ret = libusb.libusb_init(ctypes.byref(self.usb_ctx))
    if ret != 0:
      raise Exception(f"Failed to init libusb: {ret}")

    # Set debug level
    # libusb.libusb_set_option(self.usb_ctx, libusb.LIBUSB_OPTION_LOG_LEVEL, libusb.LIBUSB_LOG_LEVEL_DEBUG)

    # Open device
    self.handle = libusb.libusb_open_device_with_vid_pid(self.usb_ctx, 0x174c, 0x2362)
    print(self.handle)
    if not self.handle:
      raise Exception("Failed to open device")

    # ret = libusb.libusb_detach_kernel_driver(self.handle, 0)
    # ret = libusb.libusb_detach_kernel_driver(self.handle, 1)
    # ret = libusb.libusb_detach_kernel_driver(self.handle, 2)
    # ret = libusb.libusb_detach_kernel_driver(self.handle, 3)
    # ret = libusb.libusb_detach_kernel_driver(self.handle, 4)
    # libusb.libusb_attach_kernel_driver(self.handle, 0)
    # exit(0)

    # Detach kernel driver if needed
    if libusb.libusb_kernel_driver_active(self.handle, 0) == 1:
      ret = libusb.libusb_detach_kernel_driver(self.handle, 0)
      if ret != 0:
        raise Exception(f"Failed to detach kernel driver: {ret}")

    # libusb.libusb_reset_device(self.handle)

    # Set configuration
    # ret = libusb.libusb_set_configuration(self.handle, 1)
    # if ret != 0:
    #   raise Exception(f"Failed to set configuration: {ret}")

    # # Claim interface
    ret = libusb.libusb_claim_interface(self.handle, 0)
    if ret != 0:
      raise Exception(f"Failed to claim interface: {ret}")

    # # Set alternate setting to 1 (this is crucial!)
    ret = libusb.libusb_set_interface_alt_setting(self.handle, 0, 1)
    if ret != 0:
      raise Exception(f"Failed to set alternate setting: {ret}")

    # Clear halts on endpoints
    # libusb.libusb_clear_halt(self.handle, 0x02)  # Command OUT endpoint
    # libusb.libusb_clear_halt(self.handle, 0x04)  # Command OUT endpoint
    # libusb.libusb_clear_halt(self.handle, 0x81)  # Status IN endpoint
    # libusb.libusb_clear_halt(self.handle, 0x83)  # Status IN endpoint

    usb_cmd = [
      0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xe4, 0x24, 0x00, 0xb2, 0x1a, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    ]
    self.read_cmd = (ctypes.c_uint8 * len(usb_cmd))(*bytes(usb_cmd))
    self.read_status = (ctypes.c_uint8 * 64)()
    self.read_data = (ctypes.c_uint8 * 112)()

    usb_cmd = [
      0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xe5, 0x24, 0x00, 0xb2, 0x1a, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    ]
    self.write_cmd = (ctypes.c_uint8 * len(usb_cmd))(*bytes(usb_cmd))
    self.write_cmds = [(ctypes.c_uint8 * len(usb_cmd))(*bytes(usb_cmd)) for i in range(100)]
    self.write_statuses = [(ctypes.c_uint8 * 64)() for i in range(100)]

    print("USB device initialized successfully")

    # self.setup_seq()

  def setup_transfer(self, transfer, endpoint, data, length):
    transfer.contents.dev_handle = self.handle
    transfer.contents.status = 0xff
    transfer.contents.flags = 0
    transfer.contents.endpoint = endpoint
    transfer.contents.type = libusb.LIBUSB_TRANSFER_TYPE_BULK
    transfer.contents.timeout = 1
    transfer.contents.length = length
    # transfer.contents.callback = None
    transfer.contents.user_data = None
    transfer.contents.buffer = data
    transfer.contents.num_iso_packets = 0

  def _send_ops_and_wait(self, *cmds):
    for x in cmds: libusb.libusb_submit_transfer(x)

    while True:
      libusb.libusb_handle_events(self.usb_ctx)

      all_complete = True
      for transfer in cmds:
        if transfer.contents.status == libusb.LIBUSB_TRANSFER_COMPLETED:
          continue
        elif transfer.contents.status != libusb.LIBUSB_TRANSFER_COMPLETED:
          if transfer.contents.status != 0xff: return False
          all_complete = False
      if all_complete: return True

  def write(self, start_addr, bulk_commands):
    # print('w', hex(start_addr), bulk_commands)

    # usb_cmd = [
    #   0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    #   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    #   0xe5, 0x24, 0x00, 0xb2, 0x1a, 0x00, 0x00, 0x00,
    #   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    # ]

    def __send():
      ret = libusb.libusb_bulk_transfer(self.handle, 0x04, self.write_cmd, len(self.write_cmd), None, 1)
      if ret: return None

      # TODO: check status here, to make it reliable?

      ret = libusb.libusb_bulk_transfer(self.handle, 0x83, self.read_status, 64, None, 1)
      if ret: return None

      return self.read_status

    # do it async
    # prep requests
    # req = []

    # for offset, cmd in enumerate(bulk_commands):
    #   write_cmd = self.write_cmds.pop()
    #   write_cmd[17] = cmd
    #   write_cmd[19] = (start_addr + offset) >> 8
    #   write_cmd[20] = (start_addr + offset) & 0xff

    #   status = self.write_statuses.pop()

    #   cmd_transfer = libusb.libusb_alloc_transfer(0)
    #   self.setup_transfer(cmd_transfer, 0x04, write_cmd, len(write_cmd))

    #   status_transfer = libusb.libusb_alloc_transfer(0)
    #   self.setup_transfer(status_transfer, 0x81, status, 64)

    #   req.append((cmd_transfer, status_transfer, write_cmd, status))

    # rest = len(req)
    # done = set()
    # failed = list(range(len(req)))
    # # while True:
    # for i in failed:
    #   # libusb.libusb_submit_transfer(req[i][0])
    #   libusb.libusb_bulk_transfer(self.handle, 4, req[i][2], len(req[i][2]), None, 1)
    #   ret = libusb.libusb_bulk_transfer(self.handle, 0x83, self.read_status, 64, None, 1)

        # libusb.libusb_submit_transfer(req[i][1])

      # for i in failed: libusb.libusb_submit_transfer(req[i][1])

      # failed.clear()
      # while True:
      #   libusb.libusb_handle_events(self.usb_ctx)
      #   for i,(cmd_transfer, status_transfer, write_cmd, status) in enumerate(req):
      #     if cmd_transfer.contents.status == 0xff: continue
      #     if cmd_transfer.contents.status != libusb.LIBUSB_TRANSFER_COMPLETED: failed.append(i)
      #     elif i not in done:
      #       rest -= 1
      #       done.add(i)
      #     print(cmd_transfer.contents.status)
      #   if rest == 0: break
      #   if len(failed) > 0: break
      # if rest == 0: break

    # for i in failed:
    #   ret = libusb.libusb_bulk_transfer(self.handle, 0x83, self.read_status, 64, None, 1)

    # rest = len(req)
    # done = set()
    # failed = list(range(len(req)))
    # while True:
    #   for i in failed: libusb.libusb_submit_transfer(req[i][1])
    #   failed.clear()
    #   while True:
    #     libusb.libusb_handle_events(self.usb_ctx)
    #     for i,(cmd_transfer, status_transfer, write_cmd, status) in enumerate(req):
    #       if status_transfer.contents.status == 0xff: continue
    #       if status_transfer.contents.status != libusb.LIBUSB_TRANSFER_COMPLETED: failed.append(i)
    #       elif i not in done:
    #         rest -= 1
    #         done.add(i)
    #       print(status_transfer.contents.status)
    #     if rest == 0: break
    #     if len(failed) > 0: break
    #   if rest == 0: break

    # for offset, cmd in enumerate(bulk_commands):
    #   write_cmd = self.write_cmds.pop()
    #   write_cmd[17] = cmd
    #   write_cmd[19] = (start_addr + offset) >> 8
    #   write_cmd[20] = (start_addr + offset) & 0xff

    #   status = self.write_statuses.pop()

    #   cmd_transfer = libusb.libusb_alloc_transfer(0)
    #   self.setup_transfer(cmd_transfer, 0x04, self.read_cmd, len(self.read_cmd))

    #   status_transfer = libusb.libusb_alloc_transfer(0)
    #   self.setup_transfer(status_transfer, 0x81, self.read_data, buf_len)

    # self.write_cmd = self.write_cmds.pop()
    for offset, cmd in enumerate(bulk_commands):
      self.write_cmd[17] = cmd
      self.write_cmd[19] = (start_addr + offset) >> 8
      self.write_cmd[20] = (start_addr + offset) & 0xff

      for j in range(1000):
        read_data = __send()
        if read_data: break

      # data = (ctypes.c_uint8 * len(usb_cmd))(*bytes(usb_cmd))
      # ret = libusb.libusb_bulk_transfer(self.handle, 4, data, len(usb_cmd), ctypes.byref(transferred:=ctypes.c_int()), 1)

      # data = (ctypes.c_uint8 * len(usb_cmd))(*bytes(usb_cmd))
      # ret = libusb.libusb_bulk_transfer(self.handle, 0x04, data, len(usb_cmd), ctypes.byref(transferred:=ctypes.c_int()), 1000)
      # ret = libusb.libusb_bulk_transfer(self.handle, 0x04, data, len(usb_cmd), ctypes.byref(transferred:=ctypes.c_int()), 1000)
      # assert ret == 0

      # status_buffer = (ctypes.c_uint8 * 32)()
      # ret = libusb.libusb_bulk_transfer(self.handle, 0x81, status_buffer, 32, None, 1000)
      # assert ret == 0

      # read_data = (ctypes.c_uint8 * 112)()
      # ret = libusb.libusb_bulk_transfer(self.handle, 0x83, read_data, 112, None, 1000)
      # assert ret == 0
      # assert transferred.value == len(usb_cmd)

  def read(self, start_addr, read_len, stride=255):
    # print('r', hex(start_addr), hex(read_len))
    data = bytearray(read_len)

    # usb_cmd = [
    #   0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    #   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    #   0xe4, 0x24, 0x00, 0xb2, 0x1a, 0x00, 0x00, 0x00,
    #   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    # ]
    # data = (ctypes.c_uint8 * len(usb_cmd))(*bytes(usb_cmd))

    def __send():
      # data = (ctypes.c_uint8 * len(usb_cmd))(*bytes(usb_cmd))
      ret = libusb.libusb_bulk_transfer(self.handle, 0x04, self.read_cmd, len(self.read_cmd), None, 1)
      if ret: return None
      # assert ret == 0

      # read_data = (ctypes.c_uint8 * buf_len)()
      ret = libusb.libusb_bulk_transfer(self.handle, 0x81, self.read_data, buf_len, None, 1)
      if ret: return None
      # assert ret == 0

      # status_buffer = (ctypes.c_uint8 * 64)()
      ret = libusb.libusb_bulk_transfer(self.handle, 0x83, self.read_status, 64, None, 1)
      if ret: return None

      # try as async
      # cmd_transfer = libusb.libusb_alloc_transfer(0)
      # self.setup_transfer(cmd_transfer, 0x04, self.read_cmd, len(self.read_cmd))

      # status_transfer = libusb.libusb_alloc_transfer(0)
      # self.setup_transfer(status_transfer, 0x81, self.read_data, buf_len)

      # data_transfer = libusb.libusb_alloc_transfer(0)
      # self.setup_transfer(data_transfer, 0x83, self.read_status, 64)

      # self._send_ops_and_wait(cmd_transfer)
      # if not self._send_ops_and_wait(status_transfer): return None
      # if not self._send_ops_and_wait(data_transfer): return None
      return self.read_data

    for i in range(0, read_len, stride):
      remaining = read_len - i
      buf_len = min(stride, remaining)
      # print(hex(buf_len))
      self.read_cmd[17] = buf_len
      self.read_cmd[19] = (start_addr + i) >> 8
      self.read_cmd[20] = (start_addr + i) & 0xff

      for j in range(1000):
        read_data = __send()
        # print(read_data)
        if read_data: break

      data[i:i+buf_len] = read_data[:buf_len]
      # print('read', buf_len, read_data[:buf_len])

    return bytes(data)

  def pcie_request(self, fmt_type, address, value=None, size=4, cnt=10):
    st = time.perf_counter_ns()

    # for i in range(100000):
    #   if i % 2 == 0:
    #     self.write(0xB210, bytes([0x01]))
    #     print(self.read(0xB210, 1))
    #     assert self.read(0xB210, 1) == bytes([0x01])
    #   else:
    #     self.write(0xB210, bytes([0x08]))
    #     print(self.read(0xB210, 1))
    #     assert self.read(0xB210, 1) == bytes([0x08])
      # print(self.read(0xB224, 12))
      # print(self.read(0xB210, 1))
      # for i in range(2000000):
      #   print(self.read(0xB210, 1))

      # validate
      # print(self.read(0xB210, 1))
      # print(self.read(0xB211, 1))
      # print(self.read(0xB212, 1))
      # print(self.read(0xB213, 1))
    # exit(0)

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

    en = time.perf_counter_ns()
    print(f"Prep waiting {(prep_st-st)*1e-6:.2}ms, wt {(en-prep_st)*1e-6:.2}ms")

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
    if value is not None:
        fmt_type = 0x44

    fmt_type |= cfgreq_type
    address = (bus << 24) | (dev << 19) | (fn << 16) | (byte_addr & 0xfff)

    return self.pcie_request(fmt_type, address, value, size)

  def pcie_mem_req(self, address, value=None, size=4):
    fmt_type = 0x00
    if value is not None:
        fmt_type = 0x40

    return self.pcie_request(fmt_type, address, value, size)
