import array, time
from hexdump import hexdump
from tinygrad.runtime.support.am.usb import USBConnector
from tinygrad.runtime.autogen import pci

usb = USBConnector("")

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
  usb.pcie_cfg_req(pci.PCI_PRIMARY_BUS, bus=bus, dev=0, fn=0, value=max(0, bus-1), size=1)

  print("rescan bus={}".format(bus))
  usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=pci.PCI_BRIDGE_CTL_BUS_RESET, size=1)
  time.sleep(0.1)
  usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=pci.PCI_BRIDGE_CTL_PARITY|pci.PCI_BRIDGE_CTL_SERR, size=1)

  usb.pcie_cfg_req(pci.PCI_MEMORY_BASE, bus=bus, dev=0, fn=0, value=0x1000, size=2)
  usb.pcie_cfg_req(pci.PCI_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0x2000, size=2)
  usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_BASE, bus=bus, dev=0, fn=0, value=0x2000, size=2)
  usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0xffff, size=2)

print_cfg(0, 0)
#exit(0)

rescan_bus(0, gpu_bus=4)

print_cfg(1, 0)
#try:
#except: print("bus=1, dev=0 failed")

rescan_bus(1, gpu_bus=4)

# sleep after we rescan the bus
time.sleep(0.1)

print_cfg(2, 0)
#try
#except: print("bus=2, dev=0 failed")
#exit(0)

def setup_bus(bus, gpu_bus):
  print("setup bus={}".format(bus))
  usb.pcie_cfg_req(pci.PCI_SUBORDINATE_BUS, bus=bus, dev=0, fn=0, value=gpu_bus, size=1)
  usb.pcie_cfg_req(pci.PCI_SECONDARY_BUS, bus=bus, dev=0, fn=0, value=bus+1, size=1)
  usb.pcie_cfg_req(pci.PCI_PRIMARY_BUS, bus=bus, dev=0, fn=0, value=max(0, bus-1), size=1)

  usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=pci.PCI_BRIDGE_CTL_BUS_RESET, size=1)
  usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=pci.PCI_BRIDGE_CTL_PARITY|pci.PCI_BRIDGE_CTL_SERR, size=1)
  usb.pcie_cfg_req(pci.PCI_COMMAND, bus=bus, dev=0, fn=0, value=pci.PCI_COMMAND_IO | pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER, size=1)

  usb.pcie_cfg_req(pci.PCI_MEMORY_BASE, bus=bus, dev=0, fn=0, value=0x1000, size=2)
  usb.pcie_cfg_req(pci.PCI_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0x2000, size=2)

  usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_BASE, bus=bus, dev=0, fn=0, value=0x2000, size=2)
  usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0xffff, size=2)

setup_bus(2, gpu_bus=4)
print_cfg(3, 0)
#try:
#except: print("bus=3, dev=0 failed")

setup_bus(3, gpu_bus=4)
dmp = print_cfg(4, 0)
print(dmp[0:4])
assert dmp[0:4] == b"\x02\x10\x80\x74", "GPU NOT FOUND!"
#try:
#except: print("bus=4, dev=0 failed")

time.sleep(0.1)

# usb.write(0xf000, bytes([0x31, 0x32, 0x44, 0x66]))

# xxx = (ctypes.c_uint8 * 512)()
# for i in range(512): xxx[i] = 0xAA
# usb.post_write_request(xxx)

# print(usb.read(0xc426, 2))
# usb.write(0xc401, usb.read(0xc401, 1))
# usb.write(0xc428, bytes([(usb.read(0xc428, 1)[0] & 0xfe) | 0]))
# usb.write(0xc426, bytes([0x1, 0x1]))
# usb.write(0xc413, bytes([(usb.read(0xc413, 1)[0] & 0xc0) | 0]))
# usb.write(0xc420, bytes([(usb.read(0xc420, 1)[0] & 0xc0) | 0]))
# usb.write(0xc421, bytes([(usb.read(0xc421, 1)[0] & 0xc0) | 0]))
# usb.write(0xc414, bytes([(usb.read(0xc414, 1)[0] & 0xc0) | 0]))
# usb.write(0xc412, bytes([(usb.read(0xc412, 1)[0] & 0xc0) | 0]))
# usb.write(0xc415, bytes([(usb.read(0xc415, 1)[0] & 0xc0) | 0]))
# usb.write(0xc429, bytes([(usb.read(0xc429, 1)[0] & 0xc0) | 0]))
# print(usb.read(0xc426, 2))

# for i in range(0x9000, 0x9400, 1):
#   bt = usb.read(i, 1)
#   usb.write(i, bt)

# for i in range(0xc000, 0xc800, 1):
#   if 0xc4e9 <= i <= 0xc800: continue
#   bt = usb.read(i, 1)
#   usb.write(i, bt)

# usb.write(0xf020, bytes([0x31, 0x32, 0x44, 0x66]))
# from hexdump import hexdump
def read_all_regs(rd, sz):
  regs = {}
  for i in range(0x9000, 0x9400, 0x40):
    usb.read(rd, sz)
    z = usb.read(i, 0x40)
    for j in range(0x0, 0x40, 1): regs[i + j] = z[j]
  for i in range(0xc000, 0xc800, 0x40):
    usb.read(rd, sz)
    z = usb.read(i, 0x40)
    for j in range(0x0, 0x40, 1): regs[i + j] = z[j]
  return regs
x1 = read_all_regs(0xf123, 0x46)
x2 = read_all_regs(0xf456, 0x57)
x3 = read_all_regs(0xf459, 0x58)
x4 = read_all_regs(0xf45a, 0x59)
x5 = read_all_regs(0xf45b, 0x5a)

# check all x1, x2, x3 and find where diff
for i in range(0x9000, 0x9400, 1):
  if len({x1[i], x2[i], x3[i], x4[i], x5[i]}) != 1:
    print(hex(i), hex(x1[i]), hex(x2[i]), hex(x3[i]), hex(x4[i]), hex(x5[i]))

# for i in range(0xc000, 0xc800, 1):
#   if x1[i] != x2[i] or x1[i] != x3[i] or x2[i] != x3[i]:
#     print(hex(i), hex(x1[i]), hex(x2[i]), hex(x3[i]))

# hang and read shit
xxx = (ctypes.c_uint8 * 512)()
for i in range(512): xxx[i] = 0x59
usb.post_read_request(xxx)
for i in range(0x9000, 0x9400, 1):
  usb.write(i, bytes([x5[i]]))
for i in range(0x9000, 0x9400, 1):
  usb.write(i, bytes([x5[i]]))
for i in range(0x9000, 0x9400, 1):
  usb.write(i, bytes([x5[i]]))
for i in range(0x9000, 0x9400, 1):
  usb.write(i, bytes([x5[i]]))
# for i in range(0xc000, 0xc400, 1):
#   usb.write(i, bytes([x3[i]]))
time.sleep(1)
print(hex(xxx[0]))

# hexdump(usb.read(0x8000, 0x80))
# hexdump(usb.read(0xd800, 0x80))
# hexdump(usb.read(0xd800, 0x80))


# print(hex(vram_bar.read(0x0, 4)))
# vram_bar.write(0x0, 0xdeddddd, 4)
# lst8000 = []
# print(usb.read(0x8000, 4)) # copy to shit
for i in range(0x0, 0x10000, 0x80):
  z = usb.read(i, 0x80)
  # print(i, z, len(z))
  for j in range(0x80-1):
    if (z[j] == 0xAA): print(i)
    # if z[j] == 0x80 and z[j + 1] == 0x0:
    #   lst8000.append(i + j)
    #   print("hmm 8000", hex(i+j))
    # if z[j] == 0xd0 and z[j + 1] == 0x0: print("hmm d000", hex(i+j))
    # if z[j] == 0xf0 and z[j + 1] == 0x0: print("hmm f000", hex(i+j))

exit(0)

for z in lst8000:
  for x in lst8000: usb.write(x, bytes([0x0, 0x0]))
  print("before read", hex(z), usb.read(z, 2))
  print("after read", hex(z), usb.read(z, 2))

for i in range(0x9000, 0x9400, 0x40):
  z = usb.read(i, 0x40)
  # print(i, z, len(z))
  for j in range(0x0, 0x40, 2):
    print(hex(i + j), hex(z[j] + (z[j+1] << 8)), bin(z[j] + (z[j+1] << 8)), z[j] + (z[j+1] << 8))
    # if (i+j) % 2 != 0: continue
    # if z[j] == 0x80 and z[j + 1] == 0x0:
    #   lst8000.append(i + j)
    #   print("hmm 8000", hex(i+j))
    # if z[j] == 0xd0 and z[j + 1] == 0x0: print("hmm d000", hex(i+j))
    # if z[j] == 0xf0 and z[j + 1] == 0x0: print("hmm f000", hex(i+j))

# 0x9008 -- size

exit(0)

# for i in range(0x9000, 0x9300, 1): usb.write(i, bytes([0x0]))
# for i in range(0x9000, 0x)

# print(self.bars)
# vram_bar.write(0x1000, 0xdeddddd, 4)
# print(vram_bar.read(0x1000, 1), vram_bar.read(0x1001, 1), vram_bar.read(0x1002, 1), vram_bar.read(0x1003, 1))

# exit(0)

# i = 0
# while True:
#   addr = [0, 0x1000, 0x5000][i % 3]
#   vram_bar.write(addr, [0xdeadbeef, 0x12345678][i % 2], 4)
#   assert vram_bar.read(addr, 4) == [0xdeadbeef, 0x12345678][i % 2]
#   i += 1
#   if (i % 1000) == 0: print(i)
