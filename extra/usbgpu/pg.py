import array, time, ctypes, struct
from hexdump import hexdump
from tinygrad.runtime.support.am.usb import USBConnector
from tinygrad.runtime.autogen import pci

usb = USBConnector("")

# print(usb.pcie_cfg_req(pci.PCI_SUBORDINATE_BUS, bus=2, dev=0, fn=0, value=None, size=1))

import pickle
pck_x = pickle.load(open("jl.bin", "rb"))

# print(pck_x[0xc420:0xc424])
# usb.write(0xc659, bytes([1]))

# wcdb = struct.pack('>BBQIBB',
#     0x8A,             # WRITE(16) opcode
#     0,                # flags
#     0,              # 64-bit LBA
#     512,  # number of blocks
#     0,                # group number
#     0                 # control
# )

# from tinygrad.runtime.autogen import libc, libusb

# # submit write
# usb.setup_transfer(usb.stat_transfers[1], 0x83, 0x2, usb.read_statuses[1], 64)
# usb.read_cmds[1][3] = 2
# usb.read_cmds[1][4:6] = len(wcdb).to_bytes(2, 'big')
# usb.read_cmds[1][16:16+len(wcdb)] = wcdb
# usb.setup_transfer(usb.cmd_transfers[1], 0x04, None, usb.read_cmds[1], len(usb.read_cmds[1]))
# libusb.libusb_submit_transfer(usb.cmd_transfers[1])
# print("ok")

# # submit read
# print(usb.read(0x9000, 0x10))
# exit(0)

# usb.write(0x7ef, pck_x[0x7ef:0x7f0])

# usb.write(0x548, b'\x00\x02\x00\x31')
# usb.write(0x5a8, b'\x02\x00\x00\x00')
# usb.write(0x5f8, b'\x04\x01\x01\x02')

# usb.write(0xc659, bytes([0]))
# exit(0)


usb.write(0x54b, b'\x20')
usb.write(0x5a8, b'\x02')
usb.write(0x5f8, b'\x04')
usb.write(0x7ef, bytes([0]))
usb.write(0xc422, b'\x02')
usb.write(0x648, bytes([1])) # c

print(pck_x[0xc420:0xc424])
# time.sleep(1)

# print(usb.pcie_cfg_req(pci.PCI_SUBORDINATE_BUS, bus=0, dev=0, fn=0, value=None, size=1))

# def rescan_bus(bus, gpu_bus):
#   print("set PCI_SUBORDINATE_BUS bus={} to {}".format(bus, gpu_bus))
#   usb.pcie_cfg_req(pci.PCI_SUBORDINATE_BUS, bus=bus, dev=0, fn=0, value=gpu_bus, size=1)
#   usb.pcie_cfg_req(pci.PCI_SECONDARY_BUS, bus=bus, dev=0, fn=0, value=bus+1, size=1)
#   usb.pcie_cfg_req(pci.PCI_PRIMARY_BUS, bus=bus, dev=0, fn=0, value=max(0, bus-1), size=1)

#   print("rescan bus={}".format(bus))
#   usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=pci.PCI_BRIDGE_CTL_BUS_RESET, size=1)
#   time.sleep(0.1)
#   usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=pci.PCI_BRIDGE_CTL_PARITY|pci.PCI_BRIDGE_CTL_SERR, size=1)

#   usb.pcie_cfg_req(pci.PCI_MEMORY_BASE, bus=bus, dev=0, fn=0, value=0x1000, size=2)
#   usb.pcie_cfg_req(pci.PCI_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0x2000, size=2)
#   usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_BASE, bus=bus, dev=0, fn=0, value=0x2000, size=2)
#   usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0xffff, size=2)

# rescan_bus(0, gpu_bus=4)
# rescan_bus(1, gpu_bus=4)
# time.sleep(0.5)
# print(usb.pcie_cfg_req(pci.PCI_SUBORDINATE_BUS, bus=2, dev=0, fn=0, value=None, size=1))
# exit(0)

# usb.write(0x648, bytes([0xff]))
# usb.write(0x649, bytes([0xff]))
# usb.write(0x9000, pck_x[0x9000:0x9400])
# usb.write(0xc000, pck_x[0xc000:0xc800])

# usb.write(0x548, b'\x01\x02\x01 ')
# usb.write(0x5a8, b'\x02\x01\x01\x01')
# usb.write(0x5f8, b'\x04\x01\x01\x02')

# usb.write(0x548, b'\x01\x02\x01 ')
# usb.write(0x5a8, b'\x02\x01\x01\x01')
# usb.write(0x5f8, b'\x04\x01\x01\x02')
# print(usb.read(0x648, 1))
# usb.write(0x648, bytes([1]))
# print(usb.read(0x648, 1))
xxx = (ctypes.c_uint8 * 4096)()
for i in range(4096): xxx[i] = 0x39

a = usb.read(0x0, 0xf000)
usb.scsi_write(0xeaeb, xxx)
b = usb.read(0x0, 0xf000)
hexdump(b)

import pickle
pickle.dump(b, open("jpro.bin", "wb"))

# sets = {9: 33, 45: 1, 81: 1, 84: 10, 90: 0, 125: 1, 160: 0, 300: 86, 335: 0, 404: 0, 534: 0, 536: 0, 538: 160, 539: 0, 1136: 10, 1138: 0, 1139: 8, 1140: 0, 1141: 1, 1142: 0, 1338: 0, 1353: 2, 1354: 1, 1447: 0, 1449: 1, 1450: 1, 1451: 1, 1460: 16, 1465: 0, 1466: 0, 1467: 0, 1469: 0, 1471: 0, 1473: 0, 1475: 0, 1476: 0, 1477: 0, 1478: 0, 1493: 0, 1497: 0, 1499: 0, 1500: 0, 1501: 0, 1503: 0, 1505: 0, 1507: 0, 1510: 0, 1511: 0, 1512: 0, 1529: 1, 1530: 1, 1531: 2, 1608: 0, 2026: 1, 2684: 0, 2688: 32, 2689: 216, 2690: 0, 2702: 4, 2723: 0, 2728: 15, 2729: 15, 2731: 15, 2805: 1, 2864: 2, 36954: 8, 36956: 1, 37013: 31, 37426: 4, 37978: 8, 37980: 1, 38037: 31, 38450: 4, 39002: 8, 39004: 1, 39061: 31, 39474: 4, 40026: 8, 40028: 1, 40085: 31, 45610: 0, 45611: 0, 45614: 0, 45649: 1, 45652: 3, 45866: 0, 45867: 0, 45870: 0, 45905: 1, 45908: 3, 46132: 15, 46134: 238, 46136: 0, 46160: 0, 46165: 16, 46209: 0, 46388: 15, 46390: 238, 46392: 0, 46416: 0, 46421: 16, 46465: 0, 46644: 15, 46646: 238, 46648: 0, 46672: 0, 46677: 16, 46721: 0, 46900: 15, 46902: 238, 46904: 0, 46928: 0, 46933: 16, 46977: 0, 50194: 3, 50195: 1, 50196: 128, 50197: 1, 50209: 1, 50215: 8, 50256: 0, 50296: 17, 50297: 16, 50298: 0, 50311: 1, 50313: 1, 50352: 57, 50382: 16, 50413: 1, 50423: 1, 50720: 5, 50768: 151, 50772: 0, 50777: 0, 50976: 5, 51024: 151, 51028: 0, 51033: 0, 51224: 248, 51225: 190, 51226: 239, 51227: 6, 51228: 159, 51229: 8, 51230: 130, 51231: 224, 51232: 224, 51233: 60, 51234: 15, 51235: 126, 51480: 248, 51481: 190, 51482: 239, 51483: 6, 51484: 159, 51485: 8, 51486: 130, 51487: 224, 51488: 54, 51489: 178, 51490: 248, 51491: 140, 52240: 18, 52242: 0, 52243: 199, 52266: 12, 52529: 1, 52533: 188, 52738: 1, 52760: 3, 52761: 32, 52762: 16, 52778: 233, 52796: 30, 52813: 234, 52817: 8, 52821: 1, 52826: 233, 52843: 234, 52845: 33, 52871: 16, 52872: 0, 52874: 7, 52877: 0, 52881: 1, 52882: 212, 52936: 234, 52937: 235, 52941: 8, 52994: 0, 53016: 3, 53017: 32, 53018: 16, 53034: 233, 53052: 30, 53069: 234, 53073: 8, 53077: 1, 53082: 233, 53099: 234, 53101: 33, 53127: 16, 53128: 1, 53130: 7, 53133: 1, 53137: 1, 53138: 212, 53192: 234, 53193: 235, 53197: 8, 53267: 206, 53268: 49, 53272: 234, 53273: 235, 53277: 8, 53299: 207, 53300: 48, 53304: 234, 53305: 235, 53309: 8, 54291: 210, 54292: 45, 54296: 234, 54297: 235, 54301: 8, 54323: 211, 54324: 44, 54328: 234, 54329: 235, 54333: 8, 56832: 4, 56835: 1, 56839: 10, 59237: 0, 59349: 16, 59350: 0, 59351: 0}
# for i in range(0x0, 0xf000, 1):
#   if 0xa000 <= i <= 0xafff: continue
#   if 0xb000 <= i <= 0xb1ff: continue
#   if a[i] != b[i] and (i // 0x1000) != 0x8 and (i // 0x1000) != 0x7:
#     print(hex(i), hex(a[i]), hex(b[i]))
#     sets[i] = b[i]
# print(sets)


# trans = usb.post_write_request(xxx)
# xxx[0] = 0x31
# for s in sets:
#   print(hex(s), hex(sets[s]))
# for s in sets: usb.write(s, bytes([sets[s]]))
# for s in sets:
#   if (a:=sets[s]) != (b:=usb.read(s, 1)[0]): print("fail", hex(s), hex(a), hex(b))
# for s in sets:
#   usb.write(s, bytes([sets[s]]))
# for s in sets:
#   # print("set", hex(s), hex(sets[s]))
#   usb.write(s, bytes([sets[s]]))
# for s in sets:
#   # print("set", hex(s), hex(sets[s]))
#   usb.write(s, bytes([sets[s]]))
# for s in sets:
#   # print("set", hex(s), hex(sets[s]))
#   usb.write(s, bytes([sets[s]]))

print(usb.read(0xf000, 0x10))
print(usb.read(0xc659, 1))
print(usb.write(0xc659, bytes([1])))
# print(usb.read(0xc659, 1))
exit(0)

# usb.write(0x0, pck_x)
# print("dune cfg")
# exit(0)

# # preans
# usb.write(0xb296, bytes([4]))
# print(usb.read(0xf000, 0x1))
# print(usb.read(0x8000, 0x1))
# print(usb.read(0xd800, 0x1))

print("ok")
xxx = (ctypes.c_uint8 * 4096)()
for i in range(4096): xxx[i] = 0x39
print("ok2")
# usb.fake_stream()
usb.scsi_write(0xeaeb, xxx)
# print(usb.read(0x648, 1))
# print(usb.read(0x649, 1))
print(usb.read(0x8000, 0x100))
print(usb.read(0xf000, 0x100))
print(usb.read(0xd800, 0x100))
for i in range(1000):
  xxx[i] = i
  usb.scsi_write(0xeaeb, xxx)
  # print(usb.read(0xf000, 0x1000))

# usb.write(0x648, bytes([0xff]))
# usb.write(0x649, bytes([0xff]))
# usb.write(0xc758, pck_x[0xc758:0xc75a]) # this bit restore power to gpu.
# usb.write(0xc600, pck_x[0xc600:0xc700])
# usb.write(0xc655, bytes([1]))
print(usb.read(0x648, 1))
print(usb.read(0x6e6, 1))
print(usb.read(0x9fa, 1))
# usb.write(0x0, pck_x[:0xf000])
# time.sleep()
# from hexdump import hexdump
# hexdump(usb.read(0xc600, 0x100))
# hexdump(pck_x[0xc600:0xc700])

# time.sleep(10)
# usb.scsi_write(0xeaeb, xxx)
# usb.write(0xc659, bytes([1])) # this bit is power related
# print(usb.read(0xf000, 0x1000))

print(usb.pcie_cfg_req(pci.PCI_SUBORDINATE_BUS, bus=0, dev=0, fn=0, value=None, size=1))

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

rescan_bus(0, gpu_bus=4)
rescan_bus(1, gpu_bus=4)
time.sleep(0.5)
print(usb.pcie_cfg_req(pci.PCI_SUBORDINATE_BUS, bus=2, dev=0, fn=0, value=None, size=1))

exit(0)

usb.write(0xc400, pck_x[0xc400:0xc800])
print(usb.pcie_cfg_req(pci.PCI_SUBORDINATE_BUS, bus=2, dev=0, fn=0, value=None, size=1))


# print(usb.read(0xc600, 0x100))
# print(pck_x[0xc600:0xc700])
# input()

# usb.write(0xc659, bytes([1])) # this bit is power related

exit(0)
# def cp_from_orig(st, en): usb.write(st, bytes([pck_x[st:en]]))

# def ff():
#   # print(bytes(pck_x[0x8000:0x8004]))
#   print(bytes(pck_x[0x548:0x54c]))
#   print(bytes(pck_x[0x5a8:0x5ac]))
#   print(bytes(pck_x[0x5f8:0x5fc]))

#   # usb.write(0x8000, bytes(pck_x[0x8000:0x8004]))
#   usb.write(0x548, bytes(pck_x[0x548:0x54c]))
#   usb.write(0x5a8, bytes(pck_x[0x5a8:0x5ac]))
#   usb.write(0x5f8, bytes(pck_x[0x5f8:0x5fc]))
# ff()
print(usb.read(0xf000, 0x1000))
print("ok3")
xxx2 = (ctypes.c_uint8 * 4096)()
for i in range(4096): xxx2[i] = 0x59
usb.scsi_write(0xeaeb, xxx2)
# ff()
print(usb.read(0xf000, 0x1000))

st = time.perf_counter_ns()
usb.scsi_write(0xeaeb, xxx2)
usb.scsi_write(0xeaeb, xxx)
en = time.perf_counter_ns()
print("time", (en - st) / 1e6, "mb/s", 4096 / ((en - st) / 1e6))

# for i in range(100):
#   xxx2[0] = i
#   usb.scsi_write(0xeaeb, xxx2)

print(usb.read(0xf000, 0x1000))

exit(0)

for i, z in enumerate(x):
  print("restore", hex(i), hex(z))
  usb.scsi_write(0xeaeb, xxx, wait=False)
  usb.write(i, bytes([z]))
  print("bef shit")
  usb.scsi_write(0xeaeb, xxx2, wait=False)
  print("bef wait")
  print(usb.read(0xf000, 0x1))
  # self._send(cdb, in_data=buf)

# usb.scsi_read(0xeaea, 1)

print("dump")
hexdump(usb.read(0xa000, 0x1000))
print("")
hexdump(usb.read(0xb000, 0x200))
print("")
hexdump(usb.read(0xf000, 0x1000))
exit(0)

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

# print_cfg(2, 0)
#try
#except: print("bus=2, dev=0 failed")
exit(0)

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

# setup_bus(2, gpu_bus=4)
# print_cfg(3, 0)
#try:
#except: print("bus=3, dev=0 failed")

# setup_bus(3, gpu_bus=4)
# dmp = print_cfg(4, 0)
# print(dmp[0:4])
# assert dmp[0:4] == b"\x02\x10\x80\x74", "GPU NOT FOUND!"
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
usb.write(0xc400, pck_x[0xc400:0xc600])
print(usb.pcie_cfg_req(pci.PCI_SUBORDINATE_BUS, bus=4, dev=0, fn=0, value=None, size=1))
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
