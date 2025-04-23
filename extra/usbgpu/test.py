import array, time, ctypes, struct, random
from hexdump import hexdump
from tinygrad.runtime.support.am.usb import USBConnector
from tinygrad.runtime.autogen import pci
from tinygrad.helpers import Timing
from tinygrad import Device

usb = USBConnector("")

COUNTERS = 2

for i in range(COUNTERS): print(i, int(usb.read(0x3000 + i, 1)[0]))

import pickle
x = pickle.load(open("zpro.bin", "rb"))
if usb.read(0x0, 1) != b'\x33':
  # print(x[:0x1000])
  usb.write(0x0, x[:0x1000])
  usb.write(0x0, bytes([0x33]))
  usb.write(0xc422, b'\x02')
  # print(usb.read(0x398, 4))
# usb.write(0xc420, b'\x00')
# usb.write(0xc421, b'\x00')

# usb.write(0xc000, x[0xc000:0xe800])

# x = usb.read(0x0, 0x10000)
# pickle.dump(x, open("zpro.bin", "wb"))
# exit(0)

# print(usb.read(0xf000, 0x10))

xxx = (ctypes.c_uint8 * 4096)()
dfg = random.randint(0, 255)
for i in range(len(xxx)): xxx[i] = dfg

print("reset counters")
for i in range(COUNTERS): usb.write(0x3000 + i, bytes([0x0]))

# print(usb.read(0xce6e, 2), x[0xce6e:0xce70])
# usb.scsi_write(0xeaeb, xxx) # enters 3 times
# print(usb.read(0x3, 1))
# usb.write(0xce6e, x[0xce6e:0xce70])
# print(usb.read(0xce6e, 2), x[0xce6e:0xce70])
print(dfg, usb.read(0xf000, 0x10))
# usb.write(0x1, x[1:0x1000])
# usb.write(0xce6e, x[0xce6e:0xce70])
# usb.write(0xf000, bytes([0x01, 0x02, 0x03, 0x04]))
print(usb.write(0x3, bytes([1])))

with Timing():
  usb.scsi_write(0xeaeb, xxx)
  usb.write(0xce6e, x[0xce6e:0xce70])
  # usb.scsi_write(0xeaeb, xxx)
  # usb.write(0xce6e, x[0xce6e:0xce70])

with Timing():
  usb.read(0xf000, 0x1000)

print(usb.read(0x3, 1))

# for i in range(COUNTERS): print(i, int(usb.read(0x3000 + i, 1)[0]))

exit(0)
