import array, time, ctypes, struct, threading
from hexdump import hexdump
from tinygrad.runtime.support.am.usb import USBConnector
from tinygrad.runtime.autogen import pci
from tinygrad.runtime.autogen import libc, libusb

usb = USBConnector("")

usb.write(0x54b, b'\x20')
usb.write(0x5a8, b'\x02')
usb.write(0x5f8, b'\x04')
usb.write(0x7ef, bytes([0])) # checked in FUN_CODE_3f4a
usb.write(0xc422, b'\x02') 
usb.write(0x648, bytes([1])) # c

print(usb.read(0xb200 + 0x51, 1))
print(usb.read(0xb200 + 0x52, 1))
exit(0)

# usb.write(0xb200 + 0x51, bytes([1]))
# usb.write(0xb200 + 0x52, bytes([1]))
# print(usb.read(0xb200 + 0x51, 1))
# print(usb.read(0xb200 + 0x52, 1))

xxx = (ctypes.c_uint8 * 4096)()
for i in range(4096): xxx[i] = 0x35
# a = usb.read(0xb000, 0x100)
# hexdump(a)

import pickle
borig = pickle.load(open("t1.bin", "rb"))

# hexdump(usb.read(0xf000, 0x100))
# b = usb.read(0xb000, 0x1000)
# for i in range(0, len(b)):
#     if borig[i] != b[i]: print("diff", hex(i), hex(borig[i]), hex(b[i]))

def print_cfg(bus, dev):
  cfg = []
  for i in range(0, 256, 4):
    #print("cfg", i)
    cfg.append(usb.pcie_cfg_req(i, bus=bus, dev=dev, fn=0, value=None, size=4))

  print("bus={}, dev={}".format(bus, dev))
  dmp = bytearray(array.array('I', cfg))
  hexdump(dmp)
  return dmp

def rescan_bus(bus, gpu_bus):
  print("set PCI_SUBORDINATE_BUS bus={} to {}".format(bus, gpu_bus))
  usb.pcie_cfg_req(pci.PCI_SUBORDINATE_BUS, bus=bus, dev=0, fn=0, value=gpu_bus, size=1)
  usb.pcie_cfg_req(pci.PCI_SECONDARY_BUS, bus=bus, dev=0, fn=0, value=bus+1, size=1)
  usb.pcie_cfg_req(pci.PCI_PRIMARY_BUS, bus=bus, dev=0, fn=0, value=max(0, bus), size=1)

  print("rescan bus={}".format(bus))
  usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=pci.PCI_BRIDGE_CTL_BUS_RESET, size=1)
  time.sleep(0.1)
  usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=pci.PCI_BRIDGE_CTL_PARITY|pci.PCI_BRIDGE_CTL_SERR, size=1)

  usb.pcie_cfg_req(pci.PCI_MEMORY_BASE, bus=bus, dev=0, fn=0, value=0xD0, size=2)
  usb.pcie_cfg_req(pci.PCI_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0xD0, size=2)
  usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_BASE, bus=bus, dev=0, fn=0, value=0x0000, size=2)
  usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0x0000, size=2)
  usb.pcie_cfg_req(pci.PCI_COMMAND, bus=bus, dev=0, fn=0, value=pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER, size=1)

  usb.pcie_cfg_req(pci.PCI_MIN_GNT, bus=bus, dev=0, fn=0, value=0x3, size=1)

  if bus == 1:
    print("gmm")
    usb.pcie_cfg_req(0x93, bus=bus, dev=0, fn=0, value=0x70, size=1)
    usb.pcie_cfg_req(0x9a, bus=bus, dev=0, fn=0, value=0x48, size=1)
    usb.pcie_cfg_req(0xb2, bus=bus, dev=0, fn=0, value=0x1f, size=1)

#exit(0)

# rescan_bus(0, gpu_bus=2)
# print_cfg(0, 0)

#try:
#except: print("bus=1, dev=0 failed")

# rescan_bus(1, gpu_bus=2)
# print_cfg(1, 0)

# sleep after we rescan the bus
time.sleep(0.1)

#try
#except: print("bus=2, dev=0 failed")
#exit(0)

def setup_bus(bus, gpu_bus):
  print("setup bus={}".format(bus))
  usb.pcie_cfg_req(pci.PCI_SUBORDINATE_BUS, bus=bus, dev=0, fn=0, value=gpu_bus, size=1)
  usb.pcie_cfg_req(pci.PCI_SECONDARY_BUS, bus=bus, dev=0, fn=0, value=bus+1, size=1)
  usb.pcie_cfg_req(pci.PCI_PRIMARY_BUS, bus=bus, dev=0, fn=0, value=max(0, bus), size=1)

  usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=pci.PCI_BRIDGE_CTL_BUS_RESET, size=1)
  usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=pci.PCI_BRIDGE_CTL_PARITY|pci.PCI_BRIDGE_CTL_SERR, size=1)
  usb.pcie_cfg_req(pci.PCI_COMMAND, bus=bus, dev=0, fn=0, value=pci.PCI_COMMAND_IO | pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER | pci.PCI_COMMAND_INTX_DISABLE, size=2)

  # usb.pcie_cfg_req(pci.PCI_IO_BASE, bus=bus, dev=0, fn=0, value=0x8, size=2)
  # usb.pcie_cfg_req(pci.PCI_IO_LIMIT, bus=bus, dev=0, fn=0, value=0x1000, size=2)
  
  usb.pcie_cfg_req(pci.PCI_MEMORY_BASE, bus=bus, dev=0, fn=0, value=0xD0, size=2)
  usb.pcie_cfg_req(pci.PCI_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0xD0, size=2)

  usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_BASE, bus=bus, dev=0, fn=0, value=0x0000, size=2)
  usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0x0000, size=2)

def setup_dev(bus, gpu_bus=0):
  # usb.pcie_cfg_req(pci.PCI_SUBORDINATE_BUS, bus=bus, dev=0, fn=0, value=gpu_bus, size=1)
  # usb.pcie_cfg_req(pci.PCI_SECONDARY_BUS, bus=bus, dev=0, fn=0, value=bus+1, size=1)
  # usb.pcie_cfg_req(pci.PCI_PRIMARY_BUS, bus=bus, dev=0, fn=0, value=max(0, bus), size=1)

  usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0, bus=bus, dev=0, fn=0, value=0x00d00000, size=4)

  usb.pcie_cfg_req(pci.PCI_COMMAND, bus=bus, dev=0, fn=0, value=pci.PCI_COMMAND_IO | pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER | pci.PCI_COMMAND_INTX_DISABLE, size=2)

# setup_dev(2, gpu_bus=2)
print_cfg(2, 0)

# print_cfg(3, 0)
# try:
# except: print("bus=3, dev=0 failed")

# setup_bus(3, gpu_bus=4)
# dmp = print_cfg(4, 0)
# print(dmp[0:4])
# assert dmp[0:4] == b"\x02\x10\x80\x74", "GPU NOT FOUND!"

# Send write in paralle and cancel it 1 sec later
# def write_thread():
#     usb.scsi_write(0xeaeb, xxx)
# thread = threading.Thread(target=write_thread)
# thread.start()
# time.sleep(1)
from tinygrad.helpers import getenv
is_fst = not getenv("SEC", 0)

if is_fst:
  usb.write(0xb200 + 0x51, bytes([0x1]))
  usb.write(0xb200 + 0x52, bytes([0x1]))
  usb.write(0xb808, bytes([0x1]))
  usb.write(0xb80a, bytes([0x1]))
  usb.write(0xb80d, bytes([0x1]))
  usb.write(0xb80e, bytes([0x1]))
  # usb.write(0xB296, bytes([0x04]))
  # print(usb.read(0xb296, 1))
  before = usb.read(0xa000, 0x1000)
  before2 = usb.read(0xb200, 0x1000)
  pickle.dump(before, open("b1.bin", "wb"))
  pickle.dump(before2, open("b2.bin", "wb"))
  usb.scsi_write(0xeaeb, xxx)

usb.write(0xb200 + 0x52, bytes([0x1]))
before = pickle.load(open("b1.bin", "rb"))
before2 = pickle.load(open("b2.bin", "rb"))
after = usb.read(0xa000, 0x1000)
after2 = usb.read(0xb200, 0x1000)

# print(after2[0x18:0x20])


# diff in before and after
for i in range(0, len(before)):
  if before[i] != after[i]: print("diff", hex(0xa000 + i), hex(before[i]), hex(after[i]))
print()
for i in range(0, len(before2)):
  if before2[i] != after2[i]: print("diff", hex(0xb200 + i), hex(before2[i]), hex(after2[i]))

# for x in range(0, 8192, 4):
#   print(hex(x), usb.pcie_mem_req(0xd00000 + x, None, 4))

print(usb.pcie_mem_req(0xd00000 + 0x1008, None, 4))
print(usb.pcie_mem_req(0xd00000 + 0x100c, None, 4))

# usb.pcie_mem_req(0xd00000 + 0x1008, value=0x1, size=4)
# usb.pcie_mem_req(0xd00000 + 0x100c, value=0x1, size=4)
# print(usb.read(0xb236, 1))
# print(usb.read(0xb23e, 1))
# print(usb.read(0xb242, 1))
# print(usb.read(0xb246, 1))
# print(usb.read(0xb24a, 1))
# print(usb.read(0xb24e, 1))
# b'\x01'
# b'\x01'
# b'\x02'
# b'\x03'
# b'\x02'
# b'\x03'

print(usb.read(0xb200 + 0x51, 1))
print(usb.read(0xb200 + 0x52, 1))
exit(0)

# usb.write(0xb000, borig[:0x200])
print(usb.read(0xf000, 0x10))
x = input()

# print(x)
# usb.write(0x548, b'\x01\x02\x01 ')
# usb.write(0x5a8, b'\x02\x01\x01\x01')
# usb.write(0x5f8, b'\x04\x01\x01\x02')

# print("ok?")

# usb.reset()
# borig = usb.read(0xb000, 0x200)
# usb.write(0xb000, borig[:0x200])
usb.write(0x548, b'\x01\x02\x01 ')
usb.write(0x5a8, b'\x02\x01\x01\x01')
usb.write(0x5f8, b'\x04\x01\x01\x02')
# usb.write(0xf000, bytes([0x0]))
hexdump(usb.read(0xf000, 0x10))

# libusb.libusb_clear_halt(usb.handle, 0x81)
# libusb.libusb_clear_halt(usb.handle, 0x83)
# libusb.libusb_clear_halt(usb.handle, 0x04)
# libusb.libusb_clear_halt(usb.handle, 0x02)

print("stop", "hm")
exit(0)

for i in range(64):
    print(i)
    usb.write(0xb000, borig)
    # a1 = usb.read(0xa000, 0x1000)
    # a = usb.read(0xb000, 0x1000)
    # usb.scsi_write(0xeaeb, xxx)
    b1 = usb.read(0xa000, 0x1000)
    b = usb.read(0xb000, 0x1000)
    for i in range(0, len(b)):
        if a1[i] != b1[i]: print("diff", hex(i), hex(a1[i]), hex(b1[i]))
    print()
    for i in range(0, len(b)):
        if a[i] != b[i]: print("diff", hex(i), hex(a[i]), hex(b[i]))

# import pickle
# pickle.dump(b, open("t1.bin", "wb"))

# usb.scsi_write(0xeaeb, xxx)

# usb.scsi_write(0xeaeb, xxx)

print("ok")
