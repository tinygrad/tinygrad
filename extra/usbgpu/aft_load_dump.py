import array, time, ctypes, struct
from hexdump import hexdump
from tinygrad.runtime.support.am.usb import USBConnector
from tinygrad.runtime.autogen import pci

usb = USBConnector("")

regs1, regs2 = {}, {}


# regs1[901a] = 13 != regs2[901a] = 1
# regs1[910c] = 170 != regs2[910c] = 168
# regs1[9111] = 0 != regs2[9111] = 1
# regs1[9113] = 0 != regs2[9113] = 8

# regs1[c450] = 4 != regs2[c450] = 0
# regs1[c478] = 1 != regs2[c478] = 3
# regs1[c479] = 0 != regs2[c479] = 2
# regs1[c4ef] = 0 != regs2[c4ef] = 1

# 0 vs 1
# regs1[c820] = 198 != regs2[c820] = 84
# regs1[c821] = 77 != regs2[c821] = 183
# regs1[c822] = 127 != regs2[c822] = 46
# regs1[c823] = 39 != regs2[c823] = 29
# 1 vs 2
# regs2[c820] = 84 != regs3[920] = 3
# regs2[c821] = 183 != regs3[921] = 240
# regs2[c822] = 46 != regs3[922] = 146
# regs2[c823] = 29 != regs3[923] = 8

def read_flash(addr, buf_len):
  remaining = buf_len

  flash_addr = addr
  flash_addr_lo = flash_addr & 0xff
  flash_addr_md = (flash_addr >> 8) & 0xff
  flash_addr_hi = (flash_addr >> 16) & 0xff

  # Set FLASH_CON_MODE to read, with normal I/O config.
  usb.write(0xC8AD, bytes([0x00]))
  usb.write(0xC8AA, bytes([0x03]))
  # print(usb.read(0xC8AA, 0x03))

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

  return usb.read(0x7000, buf_len)

def write_flash(addr, byte):
  flash_addr = addr
  flash_addr_lo = flash_addr & 0xff
  flash_addr_md = (flash_addr >> 8) & 0xff
  flash_addr_hi = (flash_addr >> 16) & 0xff

  usb.write(0x7000, bytes([byte]))

  # Set FLASH_CON_MODE to read, with normal I/O config.
  usb.write(0xC8AD, bytes([0x1f]))
  usb.write(0xC8AA, bytes([0x3]))

  # Set FLASH_CON_BUF_OFFSET to zero.
  usb.write(0xC8AE, struct.pack('>H', 0x0000))

  # Set FLASH_CON_ADDR_LEN_MAYBE to 3.
  usb.write(0xC8AC, bytes([0x03]))

  # Set the flash address.
  usb.write(0xC8A1, bytes([flash_addr_lo]))
  usb.write(0xC8A2, bytes([flash_addr_md]))
  usb.write(0xC8AB, bytes([flash_addr_hi]))

  usb.write(0x7000, bytes([byte]))

  # Set FLASH_CON_DATA_LEN to the number of bytes to read.
  usb.write(0xC8A3, struct.pack('>H', 1))

  # Set FLASH_CON_CSR bit 0 to start the read.
  usb.write(0xC8A9, bytes([0x1]))

  # Wait for read to finish.
  while usb.read(0xC8A9, 1)[0] & 1:
    continue
  print("hm",usb.read(0x7000, 1))

for i in range(0x0, 0x20000, 0x200):
  z = read_flash(i, 0x200)
  for j in range(0x200-8):
    if (z[j] == 0xe1 and z[j+3] == 0xe2 and z[j+6] == 0xe3 and z[j+9] == 0xe4):
      hexdump(z)
      print(hex(i+j))

# write_flash(0x3a1e, 0x6F)
z = read_flash(0x0, 128)
hexdump(z)

xxx = (ctypes.c_uint8 * 128)()
for i in range(128): xxx[i] = z[i]
xxx[5] = 0x31
usb.wrcfg(xxx)

# z = read_flash(0x0, 128)
# hexdump(z)

print(usb.read(0x8000, 128))
print(usb.read(0x6000, 128))
print(usb.read(0x9E00, 128))
print(usb.read(0xA000, 128))
print(usb.read(0xB000, 128))
print(usb.read(0xB800, 128))
print(usb.read(0xD000, 128))
print(usb.read(0x7000, 128))
print(usb.read(0xf000, 128))
print(usb.read(0xf800, 128))
exit(0)

xxx = (ctypes.c_uint8 * 256)()
for i in range(256): xxx[i] = 0x59
usb.post_write_request(xxx)

usb.write(0x06f9, bytes([0x30]))
print(usb.read(0x06f9, 1))

z = usb.read(0xf000, 0x100)
z = usb.read(0xf000, 0x100)
hexdump(z)
exit(0)

usb.write(0x911a, bytes([0x1]))
usb.write(0x910c, bytes([168]))

usb.write(0x9111, bytes([0x1]))
usb.write(0x9113, bytes([0x8]))

usb.write(0xc478, bytes([0x5]))
usb.write(0xc479, bytes([0x3]))

usb.write(0x9111, bytes([0x1]))
print(usb.read(0x9111, 1))
print(usb.read(0x9111, 1))
print(usb.read(0x9113, 1))

time.sleep(1)

print(hex(xxx[0]))
exit(0)

off = 0xc800

z = usb.read(off, 0x100)
for j in range(0x0, 0x100, 1): regs1[off + j] = z[j]

z = usb.read(off, 0x100)
for j in range(0x0, 0x100, 1): regs2[off + j] = z[j]

print("0 vs 1")
for i in range(0x0, 0x100, 1):
  if regs1[off + i] != regs2[off + i]:
    print(f"regs1[{i + off:x}] = {regs1[off + i]} != regs2[{i + off:x}] = {regs2[off + i]}")

regs3 = {}
z = usb.read(off, 0x100)
for j in range(0x0, 0x100, 1): regs3[off + j] = z[j]

print("1 vs 2")
for i in range(0x0, 0x100, 1):
  if regs3[off + i] != regs2[off + i]:
    print(f"regs2[{i + off:x}] = {regs2[off + i]} != regs3[{i + off:x}] = {regs3[off + i]}")