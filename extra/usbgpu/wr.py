import array, time, ctypes, struct
from hexdump import hexdump
from tinygrad.runtime.support.am.usb import USBConnector
from tinygrad.runtime.autogen import pci

usb = USBConnector("")

def flash_read(start_addr, read_len, stride=128):
  data = bytearray(read_len)

  for i in range(0, read_len, stride):
    remaining = read_len - i
    buf_len = min(stride, remaining)

    flash_addr = start_addr + i
    flash_addr_lo = flash_addr & 0xff
    flash_addr_md = (flash_addr >> 8) & 0xff
    flash_addr_hi = (flash_addr >> 16) & 0xff

    # Set FLASH_CON_MODE to read, with normal I/O config.
    usb.write(0xC8AD, bytes([0x00]))

    # Set FLASH_CON_BUF_OFFSET to zero.
    usb.write(0xC8AE, struct.pack('>H', 0x0000))

    # Set FLASH_CON_ADDR_LEN_MAYBE to 3.
    usb.write(0xC8AC, bytes([0x03]))

    # Set the flash address.
    usb.write(0xC8A1, bytes([flash_addr_lo]))
    usb.write(0xC8A2, bytes([flash_addr_md]))
    usb.write(0xC8AB, bytes([flash_addr_hi]))

    # Set FLASH_CON_DATA_LEN to the number of bytes to read.
    usb.write(0xC8A3, struct.pack('>H', buf_len))

    # Set FLASH_CON_CSR bit 0 to start the read.
    usb.write(0xC8A9, bytes([0x01]))

    # Wait for read to finish.
    while usb.read(0xC8A9, 1)[0] & 1:
      continue

    buf = usb.read(0x7000, buf_len)

    data[i:i+buf_len] = buf

  return bytes(data)


cfg = flash_read(0x0, 0x80)
print(cfg)

wrbuf = (ctypes.c_uint8 * 0x80)()
for i in range(0x80): wrbuf[i] = cfg[i]

usb.wrcfg(wrbuf)
exit(0)

usb.write(0x64b, bytes([4]))
wrbuf = (ctypes.c_uint8 * 512)()
for i in range(512): wrbuf[i] = i
usb.scsi_write(wrbuf)

print(usb.read(0x2, 1))
print(usb.read(0xc426, 2))
print(usb.read(0x64f, 1))
print(usb.read(0x64b, 1))
usb.write(0x64b, bytes([3]))
print(usb.read(0xf000, 16))
# exit(0)

# print(usb.read(0x5a6, 1))
# print(usb.read(0x9000, 1))
# print(usb.read(0xf000, 16))
# print(usb.read(0xa000, 0x1000))
# print(usb.read(0xb000, 0x200))
# usb.write(0x5a6, bytes([1]))
# usb.write(0x203, bytes([1]))
# usb.write(0x20d, bytes([1]))
# usb.write(0x20e, bytes([1]))
# usb.write(0x7e5, bytes([0]))
print(usb.read(0x5a6, 1))

for i in range(32):
  print(usb.read(0xa000 + i * 16, 16))
  # print(usb.read(0xa200 + i * 16, 16))
# exit(0)

for i in range(32):
  usb.write(0xa200 + i * 16, struct.pack("IIHHHH", i, 0, 0, 0, i, 0))

# usb.scsi_read(512)
wrbuf = (ctypes.c_uint8 * 512)()
for i in range(512): wrbuf[i] = i
usb.write(0x7e5, bytes([0]))
print("br")
usb.scsi_write(wrbuf)
print("br")
print(usb.read(0x0, 16))
print(usb.read(0x0653, 2))
print(usb.read(0xf000, 16))
print("dump regs:")
print(usb.read(0x7e5, 1))
print(usb.read(0x203, 1))
print(usb.read(0x20d, 1))
print(usb.read(0x20e, 1))

exit(0)