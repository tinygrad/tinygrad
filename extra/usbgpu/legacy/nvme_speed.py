import array, time, ctypes, struct, random
from hexdump import hexdump
from tinygrad.runtime.support.usb import ASM24Controller, WriteOp, ScsiWriteOp
from tinygrad.runtime.autogen import pci
from tinygrad.helpers import Timing
from tinygrad import Device

from extra.usbgpu.patch import traps

usb = ASM24Controller()

import pickle
# pickle.dump(usb.read(0x0, 0xf000), open("rstate_2.bin", "wb"))
# exit(0)
rstate = pickle.load(open("rstate_2.bin", "rb"))
# usb.write(0x0, rstate[:0x2000])
# usb.write(0xc500, rstate[0xc500:0xc580])
# usb.write(0xc600, rstate[0xc600:0xc680])
# print("done")
# usb.write(0x0, rstate[:0xf000])

def dump_stats():
  for i,(x,x,name) in enumerate(traps):
    print(f"{i}: {name}: {usb.read(0x3000 + i, 1)}")

def reset_stats():
  for i,(x,x,name) in enumerate(traps):
    usb.write(0x3000 + i, b'\x00')

def real_scsi_write():
  self.exec_ops([ScsiWriteOp(buf, lba)])

# dump_stats()

# usb.read(0xf000, 0x1000)
# usb.write(0xce00, b'\x00\n\x01\x00\x00\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00 \x00\x00\x00\x00\x00\x00\x03 \x10\x00\x00\x00\x00\x00P\xce\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x03\x00\x04\x00\x00\x00\x00@\x04UP\x05U\x00\x00\x00\x8f\x00\x00\x00\x05\x00\x00\x00\x02\x00\x00\x00\x00\x7f\x10\x03\x0f\x00\xff\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x8f $\x00\x00\x01\x02\x00 \x00\x00:8`0\x00\x00\x00\x00\x00\x00')
# usb.write(0xafa, b'\x10\x00')

for i in range(4):
  # usb.read(0x1, 0x1000)
  sz = 4096
  xxx = (ctypes.c_uint8 * sz)()
  dfg = random.randint(0, 255)
  for i in range(len(xxx)): xxx[i] = dfg
  # print(dfg, usb.read(0xf000, 0x10))
  st = time.perf_counter_ns()
  usb.scsi_write(bytes(xxx), lba=0x1000 + i)

  # usb.read(0x10, 0xf0)
  # exit(0)

  usb.write(0x3800, b'\x00\x01')

  reset_stats()
  # usb.scsi_write(bytes(xxx), lba=0x1000 + i)
  print(hex(dfg), usb.scsi_read(sz, lba=0x1000 + i)[:100])
  # usb.read(0xb800, 0x10)
  usb.write(0x3800, b'\x00\x00')
  dump_stats()

  print(usb.read(0xb800, 0x10))

  # print("ok")
  exit(0)

  # for i in range(0x0000, 0xf000, 0x80):
  #   usb.scsi_read(sz, lba=0x1000 + i)
  #   st1 = usb.read(i, 0x80)
  #   st2 = usb.read(i, 0x80)
  #   for j in range(0x80):
  #     if st1[j] != st2[j]:
  #       print("diff", hex(i + j), hex(st1[j]), hex(st2[j]))

  # print(usb.read(0xaf9, 1))
  # usb.write(0xce00, b'\x00\n\x01\x00\x00\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00 \x00\x00\x00\x00\x00\x00\x03 \x10\x00\x00\x00\x00\x00P\xce\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x03\x00\x04\x00\x00\x00\x00@\x04UP\x05U\x00\x00\x00\x8f\x00\x00\x00\x05\x00\x00\x00\x02\x00\x00\x00\x00\x7f\x10\x03\x0f\x00\xff\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x8f $\x00\x00\x01\x02\x00 \x00\x00:8`0\x00\x00\x00\x00\x00\x00')
  # print(usb.read(0xce00, 0x80))
  # print(usb.read(0xce00, 0x80))
  # usb.scsi_read(sz, lba=0x1000 + i)[:0x400]
  # x = usb.read(0xce00, 0x80)
  # print(usb.scsi_read(sz, lba=0x1000 + i)[:0x400])
  # print(usb.scsi_read(sz, lba=0x1000 + i)[:0x400])
  # print(usb.read(0xf000, 0x80))
  # usb.write(0x0, rstate[:0x3000])
  print("done", hex(dfg))
  en = time.perf_counter_ns()
  print("mb/s is ", (0x1000) / (en - st) * 1e9 / 1024 / 1024)

exit(0)
