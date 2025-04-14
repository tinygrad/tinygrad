import array, time, ctypes, struct
from hexdump import hexdump
from tinygrad.runtime.support.am.usb import USBConnector
from tinygrad.runtime.autogen import pci

usb = USBConnector("")

x = usb.read(0x0, 0x10000)
import pickle
pickle.dump(x, open("jl.bin", "wb"))

# xxx = (ctypes.c_uint8 * 512)()
# for i in range(512): xxx[i] = 0x5c
# usb.scsi_write(0xcccc, xxx)

# hexdump(usb.scsi_read(0xcccc, 1))
# # exit(0)

# print("dump")
# hexdump(usb.read(0xa000, 0x100))
# print("")
# hexdump(usb.read(0xb000, 0x100))
# #print(usb.scsi_read(0x10))

exit(0)
