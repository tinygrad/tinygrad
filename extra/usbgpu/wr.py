import array, time, ctypes, struct
from hexdump import hexdump
from tinygrad.runtime.support.am.usb import USBConnector
from tinygrad.runtime.autogen import pci

usb = USBConnector("")


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