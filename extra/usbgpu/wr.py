import array, time, ctypes
from hexdump import hexdump
from tinygrad.runtime.support.am.usb import USBConnector
from tinygrad.runtime.autogen import pci

usb = USBConnector("")

print(usb.read(0x5a6, 1))
print(usb.read(0x9000, 1))
usb.write(0x5a6, bytes([1]))
usb.write(0x203, bytes([1]))
usb.write(0x20d, bytes([1]))
usb.write(0x20e, bytes([1]))
usb.write(0x7e5, bytes([0]))
print(usb.read(0x5a6, 1))

wrbuf = (ctypes.c_uint8 * 512)()
for i in range(512): wrbuf[i] = i
usb.scsi_write(wrbuf)

print("dump regs:")
print(usb.read(0x7e5, 1))
print(usb.read(0x203, 1))
print(usb.read(0x20d, 1))
print(usb.read(0x20e, 1))

exit(0)