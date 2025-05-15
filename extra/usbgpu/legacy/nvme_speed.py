import array, time, ctypes, struct, random
from hexdump import hexdump
from tinygrad.runtime.support.usb import ASM24Controller, WriteOp, ScsiWriteOp
from tinygrad.runtime.autogen import pci
from tinygrad.helpers import Timing
from tinygrad import Device

usb = ASM24Controller()

import pickle
# pickle.dump(usb.read(0x0, 0xf000), open("rstate_2.bin", "wb"))
rstate = pickle.load(open("rstate_2.bin", "rb"))
# usb.write(0x0, rstate[:0x3000])
# print("done")
# usb.write(0x0, rstate[:0xf000])

def real_scsi_write():
  self.exec_ops([ScsiWriteOp(buf, lba)])

# usb.read(0xf000, 0x1000)
# usb.write(0xce00, b'\x00\n\x01\x00\x00\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00 \x00\x00\x00\x00\x00\x00\x03 \x10\x00\x00\x00\x00\x00P\xce\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x03\x00\x04\x00\x00\x00\x00@\x04UP\x05U\x00\x00\x00\x8f\x00\x00\x00\x05\x00\x00\x00\x02\x00\x00\x00\x00\x7f\x10\x03\x0f\x00\xff\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x8f $\x00\x00\x01\x02\x00 \x00\x00:8`0\x00\x00\x00\x00\x00\x00')
# usb.write(0xafa, b'\x10\x00')

for i in range(4):
  # usb.read(0x1, 0x1000)
  sz = 4096 * 16
  xxx = (ctypes.c_uint8 * sz)()
  dfg = random.randint(0, 255)
  for i in range(len(xxx)): xxx[i] = dfg
  # print(dfg, usb.read(0xf000, 0x10))
  st = time.perf_counter_ns()
  usb.scsi_write(bytes(xxx), lba=0x1000 + i)

  # map_x = {0x9: 0x1,0x2d:0x0,0x51:0x0,0xc2:0x1, 0xa80: 0x20, 0xaf5: 0x0, 0x9111: 0x0, 0x9113: 0x0, 0xc478: 0x8, 0xc479: 0x7, 0xc47a: 0x1, 0xc487: 0x1, 0xc489: 0x1, 0xc4f7: 0x1, 0xce02: 0x1,
  #   0xce88: 0x0, 0xce8d: 0x0, 0xcf02: 0x1, 0xcf88: 0x0, 0xcf8d: 0x0, 0xd010: 0x88, 0xd011: 0x0, 0xd012: 0x0, 0xd013: 0x0}
  # for k, v in map_x.items():
  #   usb.write(k, bytes([v]))

  print(usb.scsi_read(sz, lba=0x1000 + i)[:0x100])

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
