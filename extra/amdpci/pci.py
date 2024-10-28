import os, ctypes
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2
from tinygrad.helpers import to_mv, mv_address

def check(x): assert x == 0

check(libpciaccess.pci_system_init())

pci_iter = libpciaccess.pci_id_match_iterator_create(None)
print(pci_iter)

pcidev = None
while True:
  pcidev = libpciaccess.pci_device_next(pci_iter)
  if not pcidev: break
  dev_fmt = "{:04x}:{:02x}:{:02x}.{:d}".format(pcidev.contents.domain_16, pcidev.contents.bus, pcidev.contents.dev, pcidev.contents.func)
  print(dev_fmt)
  
  if pcidev.contents.vendor_id == 0x1002 and pcidev.contents.device_id == 0x744c:
    dev_fmt = "{:04x}:{:02x}:{:02x}.{:d}".format(pcidev.contents.domain_16, pcidev.contents.bus, pcidev.contents.dev, pcidev.contents.func)
    if dev_fmt == "0000:03:00.0": continue # skip it, use for kernel hacking.
    break
  
  
  # print(pcidev, hex(pcidev.contents.vendor_id), hex(pcidev.contents.device_id), pcidev.contents.domain_16, pcidev.contents.bus, pcidev.contents.dev, pcidev.contents.func)

assert pcidev is not None
pcidev = pcidev.contents
print(pcidev)

libpciaccess.pci_device_probe(ctypes.byref(pcidev))

# pcifd = os.open("/sys/bus/pci/devices/{:04x}:{:02x}:{:02x}.{:d}/resource".format(pcidev.domain_16, pcidev.bus, pcidev.dev, pcidev.func))
# assert pcifd > 0

pci_resources = []

pci_region = 0
pcifil = "/sys/bus/pci/devices/{:04x}:{:02x}:{:02x}.{:d}/resource".format(pcidev.domain_16, pcidev.bus, pcidev.dev, pcidev.func)
with open(pcifil, "r") as res:
  for line in res:
    # print(line)
    lowaddr, highaddr, flags = map(lambda x: int(x, 16), line.split()[:3])
    pci_resources.append((lowaddr, highaddr, flags))
    print(line, hex(lowaddr), hex(highaddr), hex(flags))
    size = highaddr - lowaddr + 1
    if 256 * 1024 <= size <= 4096 * 1024 and not (flags & (1 | 4 | 8)):
      break
    pci_region += 1

assert pci_region < 6
print(pci_region)

assert pci_region == 5, "5 for 7900xtx"

libpciaccess.pci_device_enable(ctypes.byref(pcidev))
print("Dev enabled")

# pci_region = 2
# print(pcidev.regions[pci_region].base_addr)
# for i in range(6):
#   print(pcidev.regions[i].base_addr, pcidev.regions[i].size)

pci_region_addr = pcidev.regions[pci_region].base_addr
pci_region_size = pcidev.regions[pci_region].size
print(pci_region_addr, pci_region_size)
x = libpciaccess.pci_device_map_range(ctypes.byref(pcidev), pci_region_addr, pci_region_size, libpciaccess.PCI_DEV_MAP_FLAG_WRITABLE, ctypes.byref(pcimem:=ctypes.c_void_p()))
print(x)
print(pcimem)


pci_mmio = to_mv(pcimem, pci_region_size).cast('I')

from hexdump import hexdump
# hexdump(pci_mmio)

# reg_offset = {}
# reg_offset[]

# doorbell bar mapping
doorbell_bar_region_addr = pcidev.regions[2].base_addr
doorbell_bar_region_size = pcidev.regions[2].size
x = libpciaccess.pci_device_map_range(ctypes.byref(pcidev), doorbell_bar_region_addr, doorbell_bar_region_size, libpciaccess.PCI_DEV_MAP_FLAG_WRITABLE, ctypes.byref(doorbell_bar_mem:=ctypes.c_void_p()))
doorbell_bar = to_mv(doorbell_bar_mem, doorbell_bar_region_size).cast('I')

IORESOURCE_UNSET = 0x20000000
assert pci_resources[2][2] & IORESOURCE_UNSET == 0, "flags is strange"

AMDGPU_NAVI10_DOORBELL64_VPE = 0x190
AMDGPU_NAVI10_DOORBELL_MAX_ASSIGNMENT	= AMDGPU_NAVI10_DOORBELL64_VPE
num_kernel_doorbells = min(AMDGPU_NAVI10_DOORBELL_MAX_ASSIGNMENT, doorbell_bar_region_size // 4)
assert num_kernel_doorbells > 0

# Resize bar
aper_base = pcidev.regions[0].base_addr
aper_size = pcidev.regions[0].size
# print(aper_base, hex(aper_size), hex(24 << 30), hex(aper_size) > hex(24 << 30))
x = libpciaccess.pci_device_map_range(ctypes.byref(pcidev), aper_base, aper_size, libpciaccess.PCI_DEV_MAP_FLAG_WRITABLE, ctypes.byref(vram_bar_mem:=ctypes.c_void_p()))

# print('vram_bar_mem', vram_bar_mem)
raw_vram = to_mv(vram_bar_mem, 24 << 30)

# TODO: place gart, need it??
gart_size = 512 << 20


# while

def amdgpu_rreg(reg):
  return pci_mmio[reg]

def amdgpu_wreg(reg, val):
  pci_mmio[reg] = val

def replay_rlc():
  # while amdgpu_rreg(0x16274) != 1: pass
  # val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x80000002, hex(val)
  # val = amdgpu_rreg(0x16274) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x80000) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000091
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x80001) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000866
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x80000) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xc) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x900ff) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000002
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x90000) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000091
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x90001) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000866
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x90000) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xc) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x866
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x200ff) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x4
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x20000) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x60
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x20001) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1c8
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x20002) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x304
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x20003) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x4e1
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x20000) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xc) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x4e1
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x300ff) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x8
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x30000) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x259
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x30001) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x3e8
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x30002) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x16274) # amdgpu_cgs_read_register:47:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x4b0
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x30003) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x640
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x30004) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x7d0
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x30005) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x899
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x30006) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x8cb
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x30007) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x8fd
  while amdgpu_rreg(0x16274) != 1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x30000) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xc) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x8fd
  val = amdgpu_rreg(0x125) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x125, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x125) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x125, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x125) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x125, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x125) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x125, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x52) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x39bc) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1000

  amdgpu_wreg(0x3696, 0x40) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x3697) != 0x40: pass
  #   print(amdgpu_rreg(0x3697))
  amdgpu_wreg(0x3696, 0x80) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x3697) != 0x80: pass
  #   print(amdgpu_rreg(0x3697))
  amdgpu_wreg(0x3696, 0xc0) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x3697) != 0xc0: pass
  #   print(amdgpu_rreg(0x3697))
  amdgpu_wreg(0x3696, 0x100) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x3697) != 0x100: pass
  #   print(amdgpu_rreg(0x3697))
  amdgpu_wreg(0x3696, 0x140) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x3697) != 0x140: pass
  #   print(amdgpu_rreg(0x3697))
  val = amdgpu_rreg(0x39bc) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1000
  amdgpu_wreg(0xcc, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0xce, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0xf8, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0xf9, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0xfa, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0xfb, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x52ef, 0x7) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x39e5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1002
  amdgpu_wreg(0x546e, 0x103d1110) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x546d, 0x21c7a) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x559f) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1
  amdgpu_wreg(0x559f, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5464) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1040000
  amdgpu_wreg(0x5464, 0x1340000) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x546e) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x103d1110
  amdgpu_wreg(0x546e, 0x103d1010) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5572) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  amdgpu_wreg(0x548a, 0x103d1110) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x5489, 0x21c7a) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x599f) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1
  amdgpu_wreg(0x599f, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5480) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1040000
  amdgpu_wreg(0x5480, 0x1440000) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x548a) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x103d1110
  amdgpu_wreg(0x548a, 0x103d1010) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5972) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  amdgpu_wreg(0x5452, 0x103d1110) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x5451, 0x21c7a) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x569f) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1
  amdgpu_wreg(0x569f, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5448) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1040000
  amdgpu_wreg(0x5448, 0x1240000) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5452) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x103d1110
  amdgpu_wreg(0x5452, 0x103d1010) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5672) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  amdgpu_wreg(0x541a, 0x103d1110) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x5419, 0x21c7a) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x589f) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1
  amdgpu_wreg(0x589f, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5410) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1040000
  amdgpu_wreg(0x5410, 0x1040000) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x541a) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x103d1110
  amdgpu_wreg(0x541a, 0x103d1010) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5872) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  val = amdgpu_rreg(0x3555) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x3555, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3540) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1
  amdgpu_wreg(0x3540, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3542) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1
  amdgpu_wreg(0x3542, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3544) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1
  amdgpu_wreg(0x3544, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3546) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1
  amdgpu_wreg(0x3546, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3549) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1
  amdgpu_wreg(0x3549, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x354b) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1
  amdgpu_wreg(0x354b, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x354d) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1
  amdgpu_wreg(0x354d, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x354f) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1
  amdgpu_wreg(0x354f, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x3555, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x52) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x5001) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = amdgpu_rreg(0x5081) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = amdgpu_rreg(0x5101) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = amdgpu_rreg(0x5181) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = amdgpu_rreg(0x397b) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x3
  amdgpu_wreg(0x397b, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x397c) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x3
  amdgpu_wreg(0x397c, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x397d) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x3
  amdgpu_wreg(0x397d, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x397e) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x3
  amdgpu_wreg(0x397e, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x9002) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf
  amdgpu_wreg(0x9000, 0xf) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9001, 0xf) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9002, 0xf) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9005, 0xf) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x93d8) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x400f
  amdgpu_wreg(0x93d8, 0x10f) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x9017) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0xf
  amdgpu_wreg(0x9015, 0xf) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9016, 0xf) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9017, 0xf) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x901a, 0xf) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x93dc) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x400f
  amdgpu_wreg(0x93dc, 0x10f) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x902c) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0xf
  amdgpu_wreg(0x902a, 0xf) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x902b, 0xf) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x902c, 0xf) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x902f, 0xf) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x93e0) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x400f
  amdgpu_wreg(0x93e0, 0x10f) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x9041) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0xf
  amdgpu_wreg(0x903f, 0xf) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9040, 0xf) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9041, 0xf) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9044, 0xf) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x93e4) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x400f
  amdgpu_wreg(0x93e4, 0x10f) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5001) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = amdgpu_rreg(0x5001) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = amdgpu_rreg(0x3ab3) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  val = amdgpu_rreg(0x3ab3) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  val = amdgpu_rreg(0x3ab3) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  amdgpu_wreg(0x3ab3, 0xf000b) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3ab6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x3ab6, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3adc) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x40
  amdgpu_wreg(0x3adc, 0x40) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3ab4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x500000
  amdgpu_wreg(0x3ab4, 0x500000) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x4185) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x70000000
  amdgpu_wreg(0x4185, 0x70000000) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x3555, 0x1) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x3540, 0x100) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3541) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xc0000000
  while amdgpu_rreg(0x3541) != 0x80000000: pass # Added wait here
  val = amdgpu_rreg(0x3541) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000000, hex(val)
  amdgpu_wreg(0x3555, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5001) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  # while amdgpu_rreg(0x5001) != 0x0: pass # Added wait here
  # val = amdgpu_rreg(0x501b) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x501b, 0x2000000) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x4f8a) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x4f8a, 0x1000) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5081) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = amdgpu_rreg(0x5081) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = amdgpu_rreg(0x3b8f) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  val = amdgpu_rreg(0x3b8f) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  val = amdgpu_rreg(0x3b8f) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  amdgpu_wreg(0x3b8f, 0xf000b) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3b92) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x3b92, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3bb8) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x40
  amdgpu_wreg(0x3bb8, 0x40) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3b90) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x500000
  amdgpu_wreg(0x3b90, 0x500000) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x42f0) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x70000000
  amdgpu_wreg(0x42f0, 0x70000000) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x3555, 0x1) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x3542, 0x100) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3543) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xc0000000
  while amdgpu_rreg(0x3543) != 0x80000000: pass
  val = amdgpu_rreg(0x3543) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000000
  amdgpu_wreg(0x3555, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5081) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = amdgpu_rreg(0x509b) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x509b, 0x2000000) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x4f9a) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x4f9a, 0x1000) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5101) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = amdgpu_rreg(0x5101) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = amdgpu_rreg(0x3c6b) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  val = amdgpu_rreg(0x3c6b) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  val = amdgpu_rreg(0x3c6b) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  amdgpu_wreg(0x3c6b, 0xf000b) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3c6e) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x3c6e, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3c94) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x40
  amdgpu_wreg(0x3c94, 0x40) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3c6c) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x500000
  amdgpu_wreg(0x3c6c, 0x500000) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x445b) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x70000000
  amdgpu_wreg(0x445b, 0x70000000) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x3555, 0x1) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x3544, 0x100) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3545) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xc0000000
  while amdgpu_rreg(0x3545) != 0x80000000: pass
  val = amdgpu_rreg(0x3545) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000000
  amdgpu_wreg(0x3555, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5101) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = amdgpu_rreg(0x511b) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x511b, 0x2000000) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x4faa) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x4faa, 0x1000) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5181) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = amdgpu_rreg(0x5181) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = amdgpu_rreg(0x3d47) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  val = amdgpu_rreg(0x3d47) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  val = amdgpu_rreg(0x3d47) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  amdgpu_wreg(0x3d47, 0xf000b) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3d4a) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x3d4a, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3d70) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x40
  amdgpu_wreg(0x3d70, 0x40) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3d48) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x500000
  amdgpu_wreg(0x3d48, 0x500000) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x45c6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x70000000
  amdgpu_wreg(0x45c6, 0x70000000) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x3555, 0x1) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x3546, 0x100) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3547) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xc0000000
  while amdgpu_rreg(0x3547) != 0x80000000: pass
  val = amdgpu_rreg(0x3547) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000000
  amdgpu_wreg(0x3555, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5181) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = amdgpu_rreg(0x519b) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x519b, 0x2000000) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x4fba) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x4fba, 0x1000) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5001) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = amdgpu_rreg(0x5081) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = amdgpu_rreg(0x5101) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = amdgpu_rreg(0x5181) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = amdgpu_rreg(0x64c0) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x64d1) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x64cf) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x2000
  val = amdgpu_rreg(0x64d1) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x64cf) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x2000
  val = amdgpu_rreg(0x64d0) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x64d0) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x64d5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x4f24) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x3555) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x3555, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3549) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x3549, 0x100) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x354a) != 0x80000000: pass
  val = amdgpu_rreg(0x354a) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000000, hex(val)
  amdgpu_wreg(0x3555, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x651c) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x652d) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x652b) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x2000
  val = amdgpu_rreg(0x652d) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x652b) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x2000
  val = amdgpu_rreg(0x652c) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x652c) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x6531) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x4f25) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x3555) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x3555, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x354b) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x354b, 0x100) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x354c) != 0x80000000: pass
  val = amdgpu_rreg(0x354c) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000000
  amdgpu_wreg(0x3555, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x6578) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x6589) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x6587) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x2000
  val = amdgpu_rreg(0x6589) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x6587) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x2000
  val = amdgpu_rreg(0x6588) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x6588) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x658d) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x4f26) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x3555) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x3555, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x354d) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x354d, 0x100) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x354e) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xc0000000
  while amdgpu_rreg(0x354e) != 0x80000000: pass
  val = amdgpu_rreg(0x354e) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000000
  amdgpu_wreg(0x3555, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x65d4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x65e5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x65e3) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x2000
  val = amdgpu_rreg(0x65e5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x65e3) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x2000
  val = amdgpu_rreg(0x65e4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x65e4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x65e9) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x4f27) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x3555) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x3555, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x354f) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x354f, 0x100) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3550) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0xc0000000
  while amdgpu_rreg(0x3550) != 0x80000000: pass
  val = amdgpu_rreg(0x3550) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000000
  amdgpu_wreg(0x3555, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x39bc) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1000
  amdgpu_wreg(0x39bc, 0x1000) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x124) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x5b193e3e, hex(val) 
  val = amdgpu_rreg(0x124) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x5b193e3e
  val = amdgpu_rreg(0x124) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x5b193e3e
  val = amdgpu_rreg(0x16274) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1
  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xe) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 0x1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x1) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0x11) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 0x1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0xb0091) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0x9) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 0x1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x23962
  while amdgpu_rreg(0x16274) != 0x1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0x15) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 0x1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x20060) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0x9) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 0x1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x17ae7
  while amdgpu_rreg(0x16274) != 0x1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0x15) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 0x1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x3f
  while amdgpu_rreg(0x16274) != 0x1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0xc0091) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0x9) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 0x1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x23962
  while amdgpu_rreg(0x16274) != 0x1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0x15) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 0x1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x3f
  amdgpu_wreg(0x3696, 0x180) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5572) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  val = amdgpu_rreg(0x5972) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  val = amdgpu_rreg(0x5672) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  val = amdgpu_rreg(0x5872) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  amdgpu_wreg(0x3846, 0x54) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3847) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x3846, 0x54) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x3847, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x38cb) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x20070
  amdgpu_wreg(0x38cb, 0x20070) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x38cd) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xc0000009
  amdgpu_wreg(0x38cd, 0xc0000009) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x3846, 0x54) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x3847, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9e97, 0x3) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9e97, 0x1) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9e98, 0x3) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9e98, 0x1) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9e81, 0x1) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9e8f, 0x100) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9e84, 0x20402) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x9e7d) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x9e7d, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x9e7c) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x9e7c, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x9e7b) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x9e7b, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x9e94) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x9e94, 0x3e80000) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9e8e, 0x81010000) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x3696, 0x1c0) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x3697) != 0x1c0: pass

  amdgpu_wreg(0x9ed8, 0x3) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9ed8, 0x1) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9ed9, 0x3) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9ed9, 0x1) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9ec2, 0x1) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9ed0, 0x100) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9ec5, 0x20402) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x9ebe) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x9ebe, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x9ebd) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x9ebd, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x9ebc) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x9ebc, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x9ed5) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x9ed5, 0x3e80000) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9ecf, 0x81010000) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x3696, 0x200) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x3697) != 0x200: pass

  amdgpu_wreg(0x9f19, 0x3) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9f19, 0x1) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9f1a, 0x3) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9f1a, 0x1) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9f03, 0x1) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9f11, 0x100) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9f06, 0x20402) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x9eff) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x9eff, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x9efe) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x9efe, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x9efd) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x9efd, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x9f16) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x9f16, 0x3e80000) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9f10, 0x81010000) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x3696, 0x240) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x3697) != 0x240: pass

  amdgpu_wreg(0x9f5a, 0x3) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9f5a, 0x1) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9f5b, 0x3) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9f5b, 0x1) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9f44, 0x1) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9f52, 0x100) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9f47, 0x20402) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x9f40) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x9f40, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x9f3f) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x9f3f, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x9f3e) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x9f3e, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x9f57) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x9f57, 0x3e80000) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x9f51, 0x81010000) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x3696, 0x280) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x3697) != 0x280: pass

  amdgpu_wreg(0x539e, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x134, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x13c, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x39f4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x200
  amdgpu_wreg(0x39f4, 0x200) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x52) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x39be) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x39c7, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x39d0, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x39d9, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x39c6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x39cf, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x39d8, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x39e1, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x39c5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x39ce, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x39d7, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x39e0, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x39c0) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x39c9, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x39d2, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x39db, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x39c1) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x39ca, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x39d3, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x39dc, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x39c2) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x39cb, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x39d4, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x39dd, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x39bf) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x39c8, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x39d1, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x39da, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x39c3) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x39cc, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x39d5, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x39de, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x39c4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x39cd, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x39d6, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x39df, 0x0) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 0x1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x7fff) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0x5) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 0x1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x6ac000) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0x6) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 0x1: pass

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x2) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0x8) # amdgpu_cgs_write_register:54:(offset)
  # while amdgpu_rreg(0x16274) != 0x1: pass 
  # TODO: ftAs

  amdgpu_wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x16273, 0x204e1) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x1628a, 0xa) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x16274) != 0x1: pass

  val = amdgpu_rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x4e1
  val = amdgpu_rreg(0x39bc) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1000
  amdgpu_wreg(0x39bc, 0x1000) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x397b) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  val = amdgpu_rreg(0x397c) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  val = amdgpu_rreg(0x397d) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  val = amdgpu_rreg(0x397e) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  val = amdgpu_rreg(0x397a) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x40202
  amdgpu_wreg(0x3984, 0x400100) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x3985) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x147faa15
  amdgpu_wreg(0x3985, 0x147faa15) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x3696, 0x2c0) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x3697) != 0x2c0: pass

  val = amdgpu_rreg(0x263e) # amdgpu_dm_plane_add_gfx11_modifiers:664:(adev->reg_offset[GC_HWIP][0][0] + 0x13de)
  assert val == 0x545, hex(val)
  val = amdgpu_rreg(0x263e) # amdgpu_dm_plane_add_gfx11_modifiers:664:(adev->reg_offset[GC_HWIP][0][0] + 0x13de)
  assert val == 0x545
  val = amdgpu_rreg(0x263e) # amdgpu_dm_plane_add_gfx11_modifiers:664:(adev->reg_offset[GC_HWIP][0][0] + 0x13de)
  assert val == 0x545
  val = amdgpu_rreg(0x263e) # amdgpu_dm_plane_add_gfx11_modifiers:664:(adev->reg_offset[GC_HWIP][0][0] + 0x13de)
  assert val == 0x545
  val = amdgpu_rreg(0x263e) # amdgpu_dm_plane_add_gfx11_modifiers:664:(adev->reg_offset[GC_HWIP][0][0] + 0x13de)
  assert val == 0x545
  val = amdgpu_rreg(0x263e) # amdgpu_dm_plane_add_gfx11_modifiers:664:(adev->reg_offset[GC_HWIP][0][0] + 0x13de)
  assert val == 0x545
  amdgpu_wreg(0x3696, 0x300) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x3697) != 0x300: pass

  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  val = amdgpu_rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  amdgpu_wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53ec) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  amdgpu_wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x5db5, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  amdgpu_wreg(0x5db6, 0x22220202) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  val = amdgpu_rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  amdgpu_wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53ec) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  amdgpu_wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x5db5, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  amdgpu_wreg(0x5db6, 0x22220202) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x3696, 0x340) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x3697) != 0x340: pass

  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  val = amdgpu_rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  amdgpu_wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53f4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  amdgpu_wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x5db5, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  amdgpu_wreg(0x5db6, 0x22220202) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  val = amdgpu_rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  amdgpu_wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53f4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  amdgpu_wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x5db5, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  amdgpu_wreg(0x5db6, 0x22220202) # amdgpu_cgs_write_register:54:(offset)
  amdgpu_wreg(0x3696, 0x380) # amdgpu_cgs_write_register:54:(offset)
  while amdgpu_rreg(0x3697) != 0x380: pass

  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  val = amdgpu_rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  amdgpu_wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53e4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  amdgpu_wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x5db5, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  amdgpu_wreg(0x5db6, 0x22220202) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  val = amdgpu_rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  amdgpu_wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53e4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  amdgpu_wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x5db5, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  amdgpu_wreg(0x5db6, 0x22220202) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  val = amdgpu_rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  amdgpu_wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53d4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  amdgpu_wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x5db5, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  amdgpu_wreg(0x5db6, 0x22220202) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  val = amdgpu_rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  amdgpu_wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53d4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  amdgpu_wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x5db5, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  amdgpu_wreg(0x5db6, 0x22220202) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53ec) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x53ed) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x53ed, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53ed) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x53ed, 0x100) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53ed) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  amdgpu_wreg(0x53ed, 0x10100) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53ed) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x10100
  amdgpu_wreg(0x53ed, 0x110100) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53ed) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x10100
  amdgpu_wreg(0x53ed, 0x1010100) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53f4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x53f5) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x53f5, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53f5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x53f5, 0x100) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53f5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  amdgpu_wreg(0x53f5, 0x10100) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53f5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x10100
  amdgpu_wreg(0x53f5, 0x110100) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53f5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x10100
  amdgpu_wreg(0x53f5, 0x1010100) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53e4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x53e5) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x53e5, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53e5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x53e5, 0x100) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53e5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  amdgpu_wreg(0x53e5, 0x10100) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53e5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x10100
  amdgpu_wreg(0x53e5, 0x110100) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53e5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x10100
  amdgpu_wreg(0x53e5, 0x1010100) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53d4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = amdgpu_rreg(0x53d5) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  amdgpu_wreg(0x53d5, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53d5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  amdgpu_wreg(0x53d5, 0x100) # amdgpu_cgs_write_register:54:(offset)
  val = amdgpu_rreg(0x53d5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  amdgpu_wreg(0x53d5, 0x10100) # amdgpu_cgs_write_register:54:(offset)

# amdgpu_wreg(0x5db4, 0x44440440)
# amdgpu_rreg(0x53e8)
# amdgpu_wreg(0x53e8, 0x0)
# print(hex(amdgpu_rreg(0x5db4)))
# amdgpu_wreg(0x5db4, 0x44440440)
# amdgpu_rreg(0x5db5)
# amdgpu_wreg(0x5db5, 0x0)
# print(hex(amdgpu_rreg(0x5db6)))
# # TODO: they do the same several times, need this?
# print(hex(amdgpu_rreg(0x358a)))
# print(hex(amdgpu_rreg(0x36c0)))
# amdgpu_wreg(0x36c0, 0x1)
# amdgpu_wreg(0x36b6, 0x0)
# print(hex(amdgpu_rreg(0x3802)))

# # Golden regs init?
# amdgpu_wreg(0x3802, 0x100)
# amdgpu_wreg(0x3802, 0x100)
# amdgpu_wreg(0x3697, 0x0)
# amdgpu_wreg(0x3696, 0x0)
# amdgpu_wreg(0x369f, 0x0)
# amdgpu_wreg(0x369e, 0x0)
# amdgpu_wreg(0x369b, 0x0)
# amdgpu_wreg(0x369a, 0x0)
# amdgpu_wreg(0x36a3, 0x0)
# amdgpu_wreg(0x36b8, 0x0)
# amdgpu_wreg(0x367b, 0xfe8e5400)
# amdgpu_wreg(0x367c, 0x85)
# amdgpu_wreg(0x3668, 0x63000000) # dmub bases setup
# amdgpu_wreg(0x3670, 0x8300e200)
# amdgpu_wreg(0x367d, 0xfe8f3600)
# amdgpu_wreg(0x367e, 0x85)
# amdgpu_wreg(0x3669, 0x64000000)
# amdgpu_wreg(0x3671, 0x84004000)
# amdgpu_wreg(0x367f, 0xfe8f7600)
# amdgpu_wreg(0x3680, 0x85)
# amdgpu_wreg(0x366a, 0x65000000)
# amdgpu_wreg(0x3672, 0x85010040)
# amdgpu_wreg(0x3658, 0xfe8f7600)
# amdgpu_wreg(0x3659, 0x85)
# amdgpu_wreg(0x3662, 0x8001003f)
# amdgpu_wreg(0x3681, 0xfe907700)
# amdgpu_wreg(0x3682, 0x85)
# amdgpu_wreg(0x366b, 0x66000000)
# amdgpu_wreg(0x3673, 0x8600b880)
# amdgpu_wreg(0x3698, 0xa0000010)
# amdgpu_wreg(0x3699, 0x10030)
# amdgpu_wreg(0x3694, 0x64000000)
# amdgpu_wreg(0x3695, 0x2000)
# amdgpu_wreg(0x369c, 0x64002000)
# amdgpu_wreg(0x369d, 0x2000)
# amdgpu_wreg(0x36b1, 0x0)
# assert amdgpu_rreg(0x36b1) == 0x0
# amdgpu_wreg(0x36b1, 0x0)
# assert amdgpu_rreg(0x3802) == 0x100
# amdgpu_wreg(0x3802, 0x0)

replay_rlc()

# card setup gfx_v11_0_hw_init.
regCP_STAT = 0x21a0 # adev->reg_offset[GC_HWIP][0][regCP_STAT_BASE_IDX] + 0x0f40
regRLC_RLCS_BOOTLOAD_STATUS = 0xee82 # adev->reg_offset[GC_HWIP][0][regRLC_RLCS_BOOTLOAD_STATUS_BASE_IDX] + 0x4e82
RLC_RLCS_BOOTLOAD_STATUS__BOOTLOAD_COMPLETE__SHIFT = 0x1f
RLC_RLCS_BOOTLOAD_STATUS__BOOTLOAD_COMPLETE_MASK = 0x80000000
def gfx_v11_0_wait_for_rlc_autoload_complete():
  while True:
    cp_status = amdgpu_rreg(regCP_STAT)
    # TODO: some exceptions here for other gpus
    if True:
      bootload_status = amdgpu_rreg(regRLC_RLCS_BOOTLOAD_STATUS)

    print("cp_status", hex(cp_status), "bootload_status", hex(bootload_status))
    if cp_status == 0 and ((bootload_status & RLC_RLCS_BOOTLOAD_STATUS__BOOTLOAD_COMPLETE_MASK) >> RLC_RLCS_BOOTLOAD_STATUS__BOOTLOAD_COMPLETE__SHIFT) == 1:
      print("rlc_autoload_complete")
      break
  
    # We have adev->firmware.load_type == AMDGPU_FW_LOAD_PSP, so skipping all the code down there

regGB_ADDR_CONFIG = 0x263e # adev->reg_offset[GC_HWIP][0][regGB_ADDR_CONFIG_BASE_IDX] + 0x13de
def get_gb_addr_config():
  gb_addr_config = amdgpu_rreg(regGB_ADDR_CONFIG)
  if gb_addr_config == 0: raise RuntimeError("error in get_gb_addr_config: gb_addr_config is 0")
  print("gb_addr_config", hex(gb_addr_config))

gfx_v11_0_wait_for_rlc_autoload_complete()
get_gb_addr_config()

# try load fw
def load_fw_into_vram(fname, address, name=None):
  with open(fname, 'rb') as file:
    file = bytearray(file.read())

  if name == "mes":
    x = amdgpu_2.struct_mes_firmware_header_v1_0.from_buffer(file[:ctypes.sizeof(amdgpu_2.struct_mes_firmware_header_v1_0)])
    # print(x.mes_ucode_offset_bytes)
    # print(x.mes_ucode_size_bytes)
  else:
    # print(fname)
    x = amdgpu_2.struct_gfx_firmware_header_v2_0.from_buffer(file[:ctypes.sizeof(amdgpu_2.struct_gfx_firmware_header_v2_0)])
    # print(x.ucode_offset_bytes)
    # print(x.ucode_size_bytes)
    # print(hex(x.ucode_start_addr_hi))
    # print(hex(x.ucode_start_addr_lo))

  for i in range(len(file)): raw_vram[i + address] = file[i]
  return x

raw_vram.cast('I')[0x0] = 0xdeadbeef

pfp_addr = 0x10000000
pfp_hdr = load_fw_into_vram('/lib/firmware/amdgpu/gc_11_0_0_pfp.bin', pfp_addr)

me_addr = 0x20000000
me_hdr = load_fw_into_vram('/lib/firmware/amdgpu/gc_11_0_0_me.bin', me_addr)

mec_addr = 0x30000000
mec_hdr = load_fw_into_vram('/lib/firmware/amdgpu/gc_11_0_0_mec.bin', mec_addr)

mes_sched_pipe_addr = 0x40000000
mes_2_hdr = load_fw_into_vram('/lib/firmware/amdgpu/gc_11_0_0_mes_2.bin', mes_sched_pipe_addr, "mes")

mes_kiq_addr = 0x50000000
mes1_hdr = load_fw_into_vram('/lib/firmware/amdgpu/gc_11_0_0_mes1.bin', mes_kiq_addr, "mes")

regGRBM_GFX_CNTL = 0xa900 # (adev->reg_offset[GC_HWIP][0][1] + 0x0900)
def soc21_grbm_select(me, pipe, queue, vmid):
  GRBM_GFX_CNTL__PIPEID__SHIFT=0x0
  GRBM_GFX_CNTL__MEID__SHIFT=0x2
  GRBM_GFX_CNTL__VMID__SHIFT=0x4
  GRBM_GFX_CNTL__QUEUEID__SHIFT=0x8

  grbm_gfx_cntl = (me << GRBM_GFX_CNTL__MEID__SHIFT) | (pipe << GRBM_GFX_CNTL__PIPEID__SHIFT) | (vmid << GRBM_GFX_CNTL__VMID__SHIFT) | (queue << GRBM_GFX_CNTL__QUEUEID__SHIFT)
  amdgpu_wreg(regGRBM_GFX_CNTL, grbm_gfx_cntl)

regCP_ME_CNTL = 0xa803 # adev->reg_offset[GC_HWIP][0][1] + 0x0803
regCP_MEC_RS64_CNTL = 0xc904 # adev->reg_offset[GC_HWIP][0][1] + 0x2904
regCP_PFP_PRGRM_CNTR_START = 0x30a4 # adev->reg_offset[GC_HWIP][0][0] + 0x1e44
regCP_PFP_PRGRM_CNTR_START_HI = 0x30b9 # adev->reg_offset[GC_HWIP][0][0] + 0x1e59
regCP_ME_PRGRM_CNTR_START = 0x30a5 # adev->reg_offset[GC_HWIP][0][0] + 0x1e45
regCP_ME_PRGRM_CNTR_START_HI = 0x30d9 # adev->reg_offset[GC_HWIP][0][0] + 0x1e79
regCP_MEC_RS64_PRGRM_CNTR_START = 0xc900 # adev->reg_offset[GC_HWIP][0][1] + 0x2900
regCP_MEC_RS64_PRGRM_CNTR_START_HI = 0xc938 # adev->reg_offset[GC_HWIP][0][1] + 0x2938
def gfx_v11_0_config_gfx_rs64():
  for pipe in range(2):
    soc21_grbm_select(0, pipe, 0, 0)
    amdgpu_wreg(regCP_PFP_PRGRM_CNTR_START, ((pfp_hdr.ucode_start_addr_hi << 30) | (pfp_hdr.ucode_start_addr_lo >> 2)) & 0xffffffff)
    amdgpu_wreg(regCP_PFP_PRGRM_CNTR_START_HI, pfp_hdr.ucode_start_addr_hi >> 2)
  soc21_grbm_select(0, 0, 0, 0)

  tmp = amdgpu_rreg(regCP_ME_CNTL)
  # assert tmp == 0x15300000, hex(tmp)
  amdgpu_wreg(regCP_ME_CNTL, 0x153c0000) # PFP_PIPE0_RESET | PFP_PIPE1_RESET
  amdgpu_wreg(regCP_ME_CNTL, 0x15300000) # cleared PFP_PIPE1_RESET | PFP_PIPE1_RESET

  for pipe in range(2):
    soc21_grbm_select(0, pipe, 0, 0)
    amdgpu_wreg(regCP_ME_PRGRM_CNTR_START, ((me_hdr.ucode_start_addr_hi << 30) | (me_hdr.ucode_start_addr_lo >> 2)) & 0xffffffff)
    amdgpu_wreg(regCP_ME_PRGRM_CNTR_START_HI, me_hdr.ucode_start_addr_hi >> 2)
  soc21_grbm_select(0, 0, 0, 0)

  tmp = amdgpu_rreg(regCP_ME_CNTL)
  amdgpu_wreg(regCP_ME_CNTL, 0x15300000) # ME_PIPE0_RESET | ME_PIPE1_RESET
  amdgpu_wreg(regCP_ME_CNTL, 0x15000000) # cleared ME_PIPE0_RESET | ME_PIPE1_RESET

  for pipe in range(4):
    soc21_grbm_select(1, pipe, 0, 0)
    amdgpu_wreg(regCP_MEC_RS64_PRGRM_CNTR_START, ((mec_hdr.ucode_start_addr_hi << 30) | (mec_hdr.ucode_start_addr_lo >> 2)) & 0xffffffff)
    amdgpu_wreg(regCP_MEC_RS64_PRGRM_CNTR_START_HI, mec_hdr.ucode_start_addr_hi >> 2)
  soc21_grbm_select(0, 0, 0, 0)

  tmp = amdgpu_rreg(regCP_ME_CNTL)
  amdgpu_wreg(regCP_ME_CNTL, 0x400f0000) # MEC_PIPE0..4_RESET
  amdgpu_wreg(regCP_ME_CNTL, 0x40000000) # cleared MEC_PIPE0..4_RESET

  print("done gfx_v11_0_config_gfx_rs64")

gart_base = 0x7fff00000000
gart_size = 0x20000000
pt_base = 0x5feb00001 # end of VRAM, 1 is for VALID?

regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32 = 0x2953
regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32 = 0x2954
def gfxhub_v3_0_setup_vm_pt_regs(pt_base, vmid):
  assert vmid == 0, "not suppoer any other"
  amdgpu_wreg(regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32, pt_base & 0xffffffff)
  amdgpu_wreg(regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32, (pt_base >> 32) & 0xffffffff)

regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32 = 0x2973
regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32 = 0x2974
regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32 = 0x2993
regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32 = 0x2994
def gfxhub_v3_0_init_gart_aperture_regs():
  gfxhub_v3_0_setup_vm_pt_regs(pt_base, 0)
  amdgpu_wreg(regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32, gart_base & 0xffffffff)
  amdgpu_wreg(regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32, (gart_base >> 32) & 0xffffffff)
  amdgpu_wreg(regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32, (gart_base + gart_size) & 0xffffffff)
  amdgpu_wreg(regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32, ((gart_base + gart_size) >> 32) & 0xffffffff)

agp_start = (0xffffff << 24)
agp_end = (0x0 << 24)
fb_start = (0x200000 << 18)
fb_end = (0x217fbf << 18)

regGCMC_VM_AGP_BASE = 0x28dc
regGCMC_VM_AGP_BOT = 0x28db
regGCMC_VM_AGP_TOP = 0x28da
regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR = 0x28dd
regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR = 0x28de
regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB = 0x2808
regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB = 0x2809
regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32 = 0x282b
regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32 = 0x282c
regGCVM_L2_PROTECTION_FAULT_CNTL2 = 0x2825
def gfxhub_v3_0_init_system_aperture_regs():
  amdgpu_wreg(regGCMC_VM_AGP_BASE, 0)
  amdgpu_wreg(regGCMC_VM_AGP_BOT, agp_start >> 24)
  amdgpu_wreg(regGCMC_VM_AGP_TOP, agp_end >> 24)
  amdgpu_wreg(regGCMC_VM_AGP_BOT, min(fb_start, agp_start) >> 18)
  amdgpu_wreg(regGCMC_VM_AGP_TOP, min(fb_end, agp_end) >> 18)
  amdgpu_wreg(regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB, 0x5feaff) # gfxhub_v3_0_init_system_aperture_regs:178:((adev->reg_offset[GC_HWIP][0][0] + 0x15a8))
  amdgpu_wreg(regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB, 0x0) # gfxhub_v3_0_init_system_aperture_regs:180:((adev->reg_offset[GC_HWIP][0][0] + 0x15a9))
  amdgpu_wreg(regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32, 0x3008e) # gfxhub_v3_0_init_system_aperture_regs:184:((adev->reg_offset[GC_HWIP][0][0] + 0x15cb))
  amdgpu_wreg(regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32, 0x0) # gfxhub_v3_0_init_system_aperture_regs:186:((adev->reg_offset[GC_HWIP][0][0] + 0x15cc))
  val = amdgpu_rreg(regGCVM_L2_PROTECTION_FAULT_CNTL2) # gfxhub_v3_0_init_system_aperture_regs:189:(adev->reg_offset[GC_HWIP][0][0] + 0x15c5)
  amdgpu_wreg(regGCVM_L2_PROTECTION_FAULT_CNTL2, 0x60000) # gfxhub_v3_0_init_system_aperture_regs:189:(adev->reg_offset[GC_HWIP][0][0] + 0x15c5)


regGCMC_VM_MX_L1_TLB_CNTL = 0x28df
def gfxhub_v3_0_init_tlb_regs():
  # tmp = REG_SET_FIELD(tmp, GCMC_VM_MX_L1_TLB_CNTL, ENABLE_L1_TLB, 1);
	# tmp = REG_SET_FIELD(tmp, GCMC_VM_MX_L1_TLB_CNTL, SYSTEM_ACCESS_MODE, 3);
	# tmp = REG_SET_FIELD(tmp, GCMC_VM_MX_L1_TLB_CNTL,
	# 		    ENABLE_ADVANCED_DRIVER_MODEL, 1);
	# tmp = REG_SET_FIELD(tmp, GCMC_VM_MX_L1_TLB_CNTL,
	# 		    SYSTEM_APERTURE_UNMAPPED_ACCESS, 0);
	# tmp = REG_SET_FIELD(tmp, GCMC_VM_MX_L1_TLB_CNTL, ECO_BITS, 0);
	# tmp = REG_SET_FIELD(tmp, GCMC_VM_MX_L1_TLB_CNTL,
	# 		    MTYPE, MTYPE_UC); /* UC, uncached */
  amdgpu_wreg(regGCMC_VM_MX_L1_TLB_CNTL, 0x1859)

def gfxhub_v3_0_init_cache_regs():
  # TODO: enable cache
  pass

regGCVM_CONTEXT0_CNTL = 0x28e8
def gfxhub_v3_0_enable_system_domain():
  # ENABLE_CONTEXT=1, PAGE_TABLE_DEPTH=0, RETRY_PERMISSION_OR_INVALID_PAGE_FAULT=0
  amdgpu_wreg(regGCVM_CONTEXT0_CNTL, 0x1fffe01)

regGCVM_CONTEXT1_CNTL = 0x28e9
regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_LO32 = 0x2975
regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_HI32 = 0x2976
regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_LO32 = 0x2995
regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_HI32 = 0x2996
def gfxhub_v3_0_setup_vmid_config():
  ctx_distance = 1
  ctx_addr_distance = 2

  for vmid in range(15):
    amdgpu_wreg(regGCVM_CONTEXT1_CNTL + ctx_distance * vmid, 0x1fffe07) # gfxhub_v3_0_setup_vmid_config:332:((adev->reg_offset[GC_HWIP][0][0] + 0x1689) + i * hub->ctx_distance)
    amdgpu_wreg(regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_LO32 + ctx_addr_distance * vmid, 0x0) # gfxhub_v3_0_setup_vmid_config:334:((adev->reg_offset[GC_HWIP][0][0] + 0x1715) + i * hub->ctx_addr_distance)
    amdgpu_wreg(regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_HI32 + ctx_addr_distance * vmid, 0x0) # gfxhub_v3_0_setup_vmid_config:336:((adev->reg_offset[GC_HWIP][0][0] + 0x1716) + i * hub->ctx_addr_distance)
    amdgpu_wreg(regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_LO32 + ctx_addr_distance * vmid, 0xffffffff) # gfxhub_v3_0_setup_vmid_config:338:((adev->reg_offset[GC_HWIP][0][0] + 0x1735) + i * hub->ctx_addr_distance)
    amdgpu_wreg(regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_HI32 + ctx_addr_distance * vmid, 0xf) # gfxhub_v3_0_setup_vmid_config:341:((adev->reg_offset[GC_HWIP][0][0] + 0x1736) + i * hub->ctx_addr_distance)

regGCVM_INVALIDATE_ENG0_ADDR_RANGE_LO32 = 0x292f
regGCVM_INVALIDATE_ENG0_ADDR_RANGE_HI32 = 0x2930
def gfxhub_v3_0_program_invalidation():
  eng_addr_distance = 2

  for i in range(18):
    amdgpu_wreg(regGCVM_INVALIDATE_ENG0_ADDR_RANGE_LO32 + eng_addr_distance * i, 0xffffffff)
    amdgpu_wreg(regGCVM_INVALIDATE_ENG0_ADDR_RANGE_HI32 + eng_addr_distance * i, 0x1f)

def hdp_v6_0_flush_hdp(): amdgpu_wreg(0x1fc00, 0x0)
def gmc_v11_0_flush_gpu_tlb():
  amdgpu_wreg(0x291c, 0xf80001)
  while amdgpu_rreg(0x292e) != 1: pass

def gfxhub_v3_0_set_fault_enable_default():
  val = amdgpu_rreg(0x307f) # gfxhub_v3_0_set_fault_enable_default:425:(adev->reg_offset[GC_HWIP][0][0] + 0x1e1f)
  amdgpu_wreg(0x307f, 0x408000) # gfxhub_v3_0_set_fault_enable_default:427:((adev->reg_offset[GC_HWIP][0][0] + 0x1e1f))
  val = amdgpu_rreg(0x2824) # gfxhub_v3_0_set_fault_enable_default:435:(adev->reg_offset[GC_HWIP][0][0] + 0x15c4)
  # assert val == 0x3ffffffc
  amdgpu_wreg(0x2824, 0x3ffffffc) # gfxhub_v3_0_set_fault_enable_default:465:((adev->reg_offset[GC_HWIP][0][0] + 0x15c4))
  amdgpu_wreg(0x1fc00, 0x0) # hdp_v6_0_flush_hdp:38:((adev->rmmio_remap.reg_offset + KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL) >> 2)
  gmc_v11_0_flush_gpu_tlb()

def gfxhub_v3_0_gart_enable():
  gfxhub_v3_0_init_gart_aperture_regs()
  gfxhub_v3_0_init_system_aperture_regs()
  gfxhub_v3_0_init_tlb_regs()
  gfxhub_v3_0_init_cache_regs()

  gfxhub_v3_0_enable_system_domain()
  # gfxhub_v3_0_disable_identity_aperture()
  gfxhub_v3_0_setup_vmid_config()
  gfxhub_v3_0_program_invalidation()
  gfxhub_v3_0_set_fault_enable_default()
  print("done gfxhub_v3_0_gart_enable")


def gfx_v11_0_gfxhub_enable():
  print("start gfx_v11_0_gfxhub_enable")
  gfxhub_v3_0_gart_enable()
  hdp_v6_0_flush_hdp()

  gmc_v11_0_flush_gpu_tlb()
  hdp_v6_0_flush_hdp()

def gfx_v11_0_init_golden_registers():
  # regTCP_CNTL, 0x20000000
  val = amdgpu_rreg(0xb9a2)
  amdgpu_wreg(0xb9a2, val | 0x20000000)

regGRBM_CNTL = 0x2000
def gfx_v11_0_constants_init():
  # this is a TODO
  val = amdgpu_rreg(0x2000) # gfx_v11_0_constants_init:1926:(adev->reg_offset[GC_HWIP][0][0] + 0x0da0)
  assert val == 0xff
  amdgpu_wreg(0x2000, 0xff) # gfx_v11_0_constants_init:1926:(adev->reg_offset[GC_HWIP][0][0] + 0x0da0)
  val = amdgpu_rreg(0x2249) # gfx_v11_0_get_sa_active_bitmap:1780:(adev->reg_offset[GC_HWIP][0][0] + 0x0fe9)
  assert val == 0xf00001
  val = amdgpu_rreg(0xfb92) # gfx_v11_0_get_sa_active_bitmap:1784:(adev->reg_offset[GC_HWIP][0][1] + 0x5b92)
  assert val == 0xf00000
  val = amdgpu_rreg(0x263d) # gfx_v11_0_get_rb_active_bitmap:1799:(adev->reg_offset[GC_HWIP][0][0] + 0x13dd)
  assert val == 0x1
  val = amdgpu_rreg(0xfb94) # gfx_v11_0_get_rb_active_bitmap:1803:(adev->reg_offset[GC_HWIP][0][1] + 0x5b94)
  assert val == 0x0
  val = amdgpu_rreg(0x2249) # gfx_v11_0_get_sa_active_bitmap:1780:(adev->reg_offset[GC_HWIP][0][0] + 0x0fe9)
  assert val == 0xf00001
  val = amdgpu_rreg(0xfb92) # gfx_v11_0_get_sa_active_bitmap:1784:(adev->reg_offset[GC_HWIP][0][1] + 0x5b92)
  assert val == 0xf00000
  amdgpu_wreg(0xc200, 0x40000000) # gfx_v11_0_select_se_sh:1773:((adev->reg_offset[GC_HWIP][0][1] + 0x2200))
  val = amdgpu_rreg(0x226f) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6755:(adev->reg_offset[GC_HWIP][0][0] + 0x100f)
  assert val == 0xfff00001
  val = amdgpu_rreg(0xfb90) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6756:(adev->reg_offset[GC_HWIP][0][1] + 0x5b90)
  # assert val == 0x0, hex(val)
  val = amdgpu_rreg(0x2249) # gfx_v11_0_get_sa_active_bitmap:1780:(adev->reg_offset[GC_HWIP][0][0] + 0x0fe9)
  assert val == 0xf00001
  val = amdgpu_rreg(0xfb92) # gfx_v11_0_get_sa_active_bitmap:1784:(adev->reg_offset[GC_HWIP][0][1] + 0x5b92)
  assert val == 0xf00000
  amdgpu_wreg(0xc200, 0x40000100) # gfx_v11_0_select_se_sh:1773:((adev->reg_offset[GC_HWIP][0][1] + 0x2200))
  val = amdgpu_rreg(0x226f) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6755:(adev->reg_offset[GC_HWIP][0][0] + 0x100f)
  assert val == 0xfff00001
  val = amdgpu_rreg(0xfb90) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6756:(adev->reg_offset[GC_HWIP][0][1] + 0x5b90)
  # assert val == 0x0
  val = amdgpu_rreg(0x2249) # gfx_v11_0_get_sa_active_bitmap:1780:(adev->reg_offset[GC_HWIP][0][0] + 0x0fe9)
  assert val == 0xf00001
  val = amdgpu_rreg(0xfb92) # gfx_v11_0_get_sa_active_bitmap:1784:(adev->reg_offset[GC_HWIP][0][1] + 0x5b92)
  assert val == 0xf00000
  amdgpu_wreg(0xc200, 0x40010000) # gfx_v11_0_select_se_sh:1773:((adev->reg_offset[GC_HWIP][0][1] + 0x2200))
  val = amdgpu_rreg(0x226f) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6755:(adev->reg_offset[GC_HWIP][0][0] + 0x100f)
  assert val == 0xfff00001
  val = amdgpu_rreg(0xfb90) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6756:(adev->reg_offset[GC_HWIP][0][1] + 0x5b90)
  # assert val == 0x0
  val = amdgpu_rreg(0x2249) # gfx_v11_0_get_sa_active_bitmap:1780:(adev->reg_offset[GC_HWIP][0][0] + 0x0fe9)
  assert val == 0xf00001
  val = amdgpu_rreg(0xfb92) # gfx_v11_0_get_sa_active_bitmap:1784:(adev->reg_offset[GC_HWIP][0][1] + 0x5b92)
  assert val == 0xf00000
  amdgpu_wreg(0xc200, 0x40010100) # gfx_v11_0_select_se_sh:1773:((adev->reg_offset[GC_HWIP][0][1] + 0x2200))
  val = amdgpu_rreg(0x226f) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6755:(adev->reg_offset[GC_HWIP][0][0] + 0x100f)
  assert val == 0xfff00001
  val = amdgpu_rreg(0xfb90) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6756:(adev->reg_offset[GC_HWIP][0][1] + 0x5b90)
  # assert val == 0x0
  val = amdgpu_rreg(0x2249) # gfx_v11_0_get_sa_active_bitmap:1780:(adev->reg_offset[GC_HWIP][0][0] + 0x0fe9)
  assert val == 0xf00001
  val = amdgpu_rreg(0xfb92) # gfx_v11_0_get_sa_active_bitmap:1784:(adev->reg_offset[GC_HWIP][0][1] + 0x5b92)
  assert val == 0xf00000
  amdgpu_wreg(0xc200, 0x40020000) # gfx_v11_0_select_se_sh:1773:((adev->reg_offset[GC_HWIP][0][1] + 0x2200))
  val = amdgpu_rreg(0x226f) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6755:(adev->reg_offset[GC_HWIP][0][0] + 0x100f)
  assert val == 0xfff00001
  val = amdgpu_rreg(0xfb90) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6756:(adev->reg_offset[GC_HWIP][0][1] + 0x5b90)
  # assert val == 0x0
  val = amdgpu_rreg(0x2249) # gfx_v11_0_get_sa_active_bitmap:1780:(adev->reg_offset[GC_HWIP][0][0] + 0x0fe9)
  assert val == 0xf00001
  val = amdgpu_rreg(0xfb92) # gfx_v11_0_get_sa_active_bitmap:1784:(adev->reg_offset[GC_HWIP][0][1] + 0x5b92)
  assert val == 0xf00000
  amdgpu_wreg(0xc200, 0x40020100) # gfx_v11_0_select_se_sh:1773:((adev->reg_offset[GC_HWIP][0][1] + 0x2200))
  val = amdgpu_rreg(0x226f) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6755:(adev->reg_offset[GC_HWIP][0][0] + 0x100f)
  assert val == 0xfff00001
  val = amdgpu_rreg(0xfb90) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6756:(adev->reg_offset[GC_HWIP][0][1] + 0x5b90)
  # assert val == 0x0
  val = amdgpu_rreg(0x2249) # gfx_v11_0_get_sa_active_bitmap:1780:(adev->reg_offset[GC_HWIP][0][0] + 0x0fe9)
  assert val == 0xf00001
  val = amdgpu_rreg(0xfb92) # gfx_v11_0_get_sa_active_bitmap:1784:(adev->reg_offset[GC_HWIP][0][1] + 0x5b92)
  assert val == 0xf00000
  amdgpu_wreg(0xc200, 0x40030000) # gfx_v11_0_select_se_sh:1773:((adev->reg_offset[GC_HWIP][0][1] + 0x2200))
  val = amdgpu_rreg(0x226f) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6755:(adev->reg_offset[GC_HWIP][0][0] + 0x100f)
  assert val == 0xfff00001
  val = amdgpu_rreg(0xfb90) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6756:(adev->reg_offset[GC_HWIP][0][1] + 0x5b90)
  # assert val == 0x0
  val = amdgpu_rreg(0x2249) # gfx_v11_0_get_sa_active_bitmap:1780:(adev->reg_offset[GC_HWIP][0][0] + 0x0fe9)
  assert val == 0xf00001
  val = amdgpu_rreg(0xfb92) # gfx_v11_0_get_sa_active_bitmap:1784:(adev->reg_offset[GC_HWIP][0][1] + 0x5b92)
  assert val == 0xf00000
  amdgpu_wreg(0xc200, 0x40030100) # gfx_v11_0_select_se_sh:1773:((adev->reg_offset[GC_HWIP][0][1] + 0x2200))
  val = amdgpu_rreg(0x226f) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6755:(adev->reg_offset[GC_HWIP][0][0] + 0x100f)
  assert val == 0xfff00001
  val = amdgpu_rreg(0xfb90) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6756:(adev->reg_offset[GC_HWIP][0][1] + 0x5b90)
  # assert val == 0x0
  val = amdgpu_rreg(0x2249) # gfx_v11_0_get_sa_active_bitmap:1780:(adev->reg_offset[GC_HWIP][0][0] + 0x0fe9)
  assert val == 0xf00001
  val = amdgpu_rreg(0xfb92) # gfx_v11_0_get_sa_active_bitmap:1784:(adev->reg_offset[GC_HWIP][0][1] + 0x5b92)
  assert val == 0xf00000
  amdgpu_wreg(0xc200, 0x40040000) # gfx_v11_0_select_se_sh:1773:((adev->reg_offset[GC_HWIP][0][1] + 0x2200))
  val = amdgpu_rreg(0x226f) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6755:(adev->reg_offset[GC_HWIP][0][0] + 0x100f)
  assert val == 0xfff00001
  val = amdgpu_rreg(0xfb90) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6756:(adev->reg_offset[GC_HWIP][0][1] + 0x5b90)
  # assert val == 0x0
  val = amdgpu_rreg(0x2249) # gfx_v11_0_get_sa_active_bitmap:1780:(adev->reg_offset[GC_HWIP][0][0] + 0x0fe9)
  assert val == 0xf00001
  val = amdgpu_rreg(0xfb92) # gfx_v11_0_get_sa_active_bitmap:1784:(adev->reg_offset[GC_HWIP][0][1] + 0x5b92)
  assert val == 0xf00000
  amdgpu_wreg(0xc200, 0x40040100) # gfx_v11_0_select_se_sh:1773:((adev->reg_offset[GC_HWIP][0][1] + 0x2200))
  val = amdgpu_rreg(0x226f) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6755:(adev->reg_offset[GC_HWIP][0][0] + 0x100f)
  assert val == 0xfff00001
  val = amdgpu_rreg(0xfb90) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6756:(adev->reg_offset[GC_HWIP][0][1] + 0x5b90)
  # assert val == 0x0
  val = amdgpu_rreg(0x2249) # gfx_v11_0_get_sa_active_bitmap:1780:(adev->reg_offset[GC_HWIP][0][0] + 0x0fe9)
  assert val == 0xf00001
  val = amdgpu_rreg(0xfb92) # gfx_v11_0_get_sa_active_bitmap:1784:(adev->reg_offset[GC_HWIP][0][1] + 0x5b92)
  assert val == 0xf00000
  amdgpu_wreg(0xc200, 0x40050000) # gfx_v11_0_select_se_sh:1773:((adev->reg_offset[GC_HWIP][0][1] + 0x2200))
  val = amdgpu_rreg(0x226f) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6755:(adev->reg_offset[GC_HWIP][0][0] + 0x100f)
  assert val == 0xfff00001
  val = amdgpu_rreg(0xfb90) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6756:(adev->reg_offset[GC_HWIP][0][1] + 0x5b90)
  # assert val == 0x0
  val = amdgpu_rreg(0x2249) # gfx_v11_0_get_sa_active_bitmap:1780:(adev->reg_offset[GC_HWIP][0][0] + 0x0fe9)
  assert val == 0xf00001
  val = amdgpu_rreg(0xfb92) # gfx_v11_0_get_sa_active_bitmap:1784:(adev->reg_offset[GC_HWIP][0][1] + 0x5b92)
  assert val == 0xf00000
  amdgpu_wreg(0xc200, 0x40050100) # gfx_v11_0_select_se_sh:1773:((adev->reg_offset[GC_HWIP][0][1] + 0x2200))
  val = amdgpu_rreg(0x226f) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6755:(adev->reg_offset[GC_HWIP][0][0] + 0x100f)
  assert val == 0xfff00001
  val = amdgpu_rreg(0xfb90) # gfx_v11_0_get_wgp_active_bitmap_per_sh:6756:(adev->reg_offset[GC_HWIP][0][1] + 0x5b90)
  # assert val == 0x0
  amdgpu_wreg(0xc200, 0xe0000000) # gfx_v11_0_select_se_sh:1773:((adev->reg_offset[GC_HWIP][0][1] + 0x2200))
  val = amdgpu_rreg(0xf006) # gfx_v11_0_get_tcc_info:1912:(adev->reg_offset[GC_HWIP][0][1] + 0x5006)
  assert val == 0x1
  val = amdgpu_rreg(0xfb96) # gfx_v11_0_get_tcc_info:1913:(adev->reg_offset[GC_HWIP][0][1] + 0x5b96)
  assert val == 0x0
  val = amdgpu_rreg(0x2545) # gfx_v11_0_constants_init:1934:(adev->reg_offset[GC_HWIP][0][0] + 0x12e5)
  assert val == 0x40000
  amdgpu_wreg(0xa900, 0x0) # soc21_grbm_select:247:((adev->reg_offset[GC_HWIP][0][1] + 0x0900))
  amdgpu_wreg(0xa9e4, 0xc00c) # gfx_v11_0_constants_init:1944:((adev->reg_offset[GC_HWIP][0][1] + 0x09e4))
  amdgpu_wreg(0xa900, 0x10) # soc21_grbm_select:247:((adev->reg_offset[GC_HWIP][0][1] + 0x0900))
  amdgpu_wreg(0xa9e4, 0xc00c) # gfx_v11_0_constants_init:1944:((adev->reg_offset[GC_HWIP][0][1] + 0x09e4))
  amdgpu_wreg(0xa9e3, 0x20001000) # gfx_v11_0_constants_init:1950:((adev->reg_offset[GC_HWIP][0][1] + 0x09e3))
  amdgpu_wreg(0xa900, 0x20) # soc21_grbm_select:247:((adev->reg_offset[GC_HWIP][0][1] + 0x0900))
  amdgpu_wreg(0xa9e4, 0xc00c) # gfx_v11_0_constants_init:1944:((adev->reg_offset[GC_HWIP][0][1] + 0x09e4))
  amdgpu_wreg(0xa9e3, 0x20001000) # gfx_v11_0_constants_init:1950:((adev->reg_offset[GC_HWIP][0][1] + 0x09e3))
  amdgpu_wreg(0xa900, 0x30) # soc21_grbm_select:247:((adev->reg_offset[GC_HWIP][0][1] + 0x0900))
  amdgpu_wreg(0xa9e4, 0xc00c) # gfx_v11_0_constants_init:1944:((adev->reg_offset[GC_HWIP][0][1] + 0x09e4))
  amdgpu_wreg(0xa9e3, 0x20001000) # gfx_v11_0_constants_init:1950:((adev->reg_offset[GC_HWIP][0][1] + 0x09e3))
  amdgpu_wreg(0xa900, 0x40) # soc21_grbm_select:247:((adev->reg_offset[GC_HWIP][0][1] + 0x0900))
  amdgpu_wreg(0xa9e4, 0xc00c) # gfx_v11_0_constants_init:1944:((adev->reg_offset[GC_HWIP][0][1] + 0x09e4))
  amdgpu_wreg(0xa9e3, 0x20001000) # gfx_v11_0_constants_init:1950:((adev->reg_offset[GC_HWIP][0][1] + 0x09e3))
  amdgpu_wreg(0xa900, 0x50) # soc21_grbm_select:247:((adev->reg_offset[GC_HWIP][0][1] + 0x0900))
  amdgpu_wreg(0xa9e4, 0xc00c) # gfx_v11_0_constants_init:1944:((adev->reg_offset[GC_HWIP][0][1] + 0x09e4))
  amdgpu_wreg(0xa9e3, 0x20001000) # gfx_v11_0_constants_init:1950:((adev->reg_offset[GC_HWIP][0][1] + 0x09e3))
  amdgpu_wreg(0xa900, 0x60) # soc21_grbm_select:247:((adev->reg_offset[GC_HWIP][0][1] + 0x0900))
  amdgpu_wreg(0xa9e4, 0xc00c) # gfx_v11_0_constants_init:1944:((adev->reg_offset[GC_HWIP][0][1] + 0x09e4))
  amdgpu_wreg(0xa9e3, 0x20001000) # gfx_v11_0_constants_init:1950:((adev->reg_offset[GC_HWIP][0][1] + 0x09e3))
  amdgpu_wreg(0xa900, 0x70) # soc21_grbm_select:247:((adev->reg_offset[GC_HWIP][0][1] + 0x0900))
  amdgpu_wreg(0xa9e4, 0xc00c) # gfx_v11_0_constants_init:1944:((adev->reg_offset[GC_HWIP][0][1] + 0x09e4))
  amdgpu_wreg(0xa9e3, 0x20001000) # gfx_v11_0_constants_init:1950:((adev->reg_offset[GC_HWIP][0][1] + 0x09e3))
  amdgpu_wreg(0xa900, 0x0) # soc21_grbm_select:247:((adev->reg_offset[GC_HWIP][0][1] + 0x0900))
  amdgpu_wreg(0xa900, 0x80) # soc21_grbm_select:247:((adev->reg_offset[GC_HWIP][0][1] + 0x0900))
  amdgpu_wreg(0xa9e4, 0xc00c) # gfx_v11_0_init_compute_vmid:1865:((adev->reg_offset[GC_HWIP][0][1] + 0x09e4))
  amdgpu_wreg(0xa9e3, 0x10002) # gfx_v11_0_init_compute_vmid:1866:((adev->reg_offset[GC_HWIP][0][1] + 0x09e3))
  val = amdgpu_rreg(0x31d2) # gfx_v11_0_init_compute_vmid:1869:(adev->reg_offset[GC_HWIP][0][0] + 0x1f72)
  # assert val == 0x0
  amdgpu_wreg(0x31d2, 0x8) # gfx_v11_0_init_compute_vmid:1871:((adev->reg_offset[GC_HWIP][0][0] + 0x1f72))
  amdgpu_wreg(0xa900, 0x90) # soc21_grbm_select:247:((adev->reg_offset[GC_HWIP][0][1] + 0x0900))
  amdgpu_wreg(0xa9e4, 0xc00c) # gfx_v11_0_init_compute_vmid:1865:((adev->reg_offset[GC_HWIP][0][1] + 0x09e4))
  amdgpu_wreg(0xa9e3, 0x10002) # gfx_v11_0_init_compute_vmid:1866:((adev->reg_offset[GC_HWIP][0][1] + 0x09e3))
  val = amdgpu_rreg(0x31d2) # gfx_v11_0_init_compute_vmid:1869:(adev->reg_offset[GC_HWIP][0][0] + 0x1f72)
  # assert val == 0x0
  amdgpu_wreg(0x31d2, 0x8) # gfx_v11_0_init_compute_vmid:1871:((adev->reg_offset[GC_HWIP][0][0] + 0x1f72))
  amdgpu_wreg(0xa900, 0xa0) # soc21_grbm_select:247:((adev->reg_offset[GC_HWIP][0][1] + 0x0900))
  amdgpu_wreg(0xa9e4, 0xc00c) # gfx_v11_0_init_compute_vmid:1865:((adev->reg_offset[GC_HWIP][0][1] + 0x09e4))
  amdgpu_wreg(0xa9e3, 0x10002) # gfx_v11_0_init_compute_vmid:1866:((adev->reg_offset[GC_HWIP][0][1] + 0x09e3))
  val = amdgpu_rreg(0x31d2) # gfx_v11_0_init_compute_vmid:1869:(adev->reg_offset[GC_HWIP][0][0] + 0x1f72)
  # assert val == 0x0
  amdgpu_wreg(0x31d2, 0x8) # gfx_v11_0_init_compute_vmid:1871:((adev->reg_offset[GC_HWIP][0][0] + 0x1f72))
  amdgpu_wreg(0xa900, 0xb0) # soc21_grbm_select:247:((adev->reg_offset[GC_HWIP][0][1] + 0x0900))
  amdgpu_wreg(0xa9e4, 0xc00c) # gfx_v11_0_init_compute_vmid:1865:((adev->reg_offset[GC_HWIP][0][1] + 0x09e4))
  amdgpu_wreg(0xa9e3, 0x10002) # gfx_v11_0_init_compute_vmid:1866:((adev->reg_offset[GC_HWIP][0][1] + 0x09e3))
  val = amdgpu_rreg(0x31d2) # gfx_v11_0_init_compute_vmid:1869:(adev->reg_offset[GC_HWIP][0][0] + 0x1f72)
  # assert val == 0x0
  amdgpu_wreg(0x31d2, 0x8) # gfx_v11_0_init_compute_vmid:1871:((adev->reg_offset[GC_HWIP][0][0] + 0x1f72))
  amdgpu_wreg(0xa900, 0xc0) # soc21_grbm_select:247:((adev->reg_offset[GC_HWIP][0][1] + 0x0900))
  amdgpu_wreg(0xa9e4, 0xc00c) # gfx_v11_0_init_compute_vmid:1865:((adev->reg_offset[GC_HWIP][0][1] + 0x09e4))
  amdgpu_wreg(0xa9e3, 0x10002) # gfx_v11_0_init_compute_vmid:1866:((adev->reg_offset[GC_HWIP][0][1] + 0x09e3))
  val = amdgpu_rreg(0x31d2) # gfx_v11_0_init_compute_vmid:1869:(adev->reg_offset[GC_HWIP][0][0] + 0x1f72)
  # assert val == 0x0
  amdgpu_wreg(0x31d2, 0x8) # gfx_v11_0_init_compute_vmid:1871:((adev->reg_offset[GC_HWIP][0][0] + 0x1f72))
  amdgpu_wreg(0xa900, 0xd0) # soc21_grbm_select:247:((adev->reg_offset[GC_HWIP][0][1] + 0x0900))
  amdgpu_wreg(0xa9e4, 0xc00c) # gfx_v11_0_init_compute_vmid:1865:((adev->reg_offset[GC_HWIP][0][1] + 0x09e4))
  amdgpu_wreg(0xa9e3, 0x10002) # gfx_v11_0_init_compute_vmid:1866:((adev->reg_offset[GC_HWIP][0][1] + 0x09e3))
  val = amdgpu_rreg(0x31d2) # gfx_v11_0_init_compute_vmid:1869:(adev->reg_offset[GC_HWIP][0][0] + 0x1f72)
  # assert val == 0x0
  amdgpu_wreg(0x31d2, 0x8) # gfx_v11_0_init_compute_vmid:1871:((adev->reg_offset[GC_HWIP][0][0] + 0x1f72))
  amdgpu_wreg(0xa900, 0xe0) # soc21_grbm_select:247:((adev->reg_offset[GC_HWIP][0][1] + 0x0900))
  amdgpu_wreg(0xa9e4, 0xc00c) # gfx_v11_0_init_compute_vmid:1865:((adev->reg_offset[GC_HWIP][0][1] + 0x09e4))
  amdgpu_wreg(0xa9e3, 0x10002) # gfx_v11_0_init_compute_vmid:1866:((adev->reg_offset[GC_HWIP][0][1] + 0x09e3))
  val = amdgpu_rreg(0x31d2) # gfx_v11_0_init_compute_vmid:1869:(adev->reg_offset[GC_HWIP][0][0] + 0x1f72)
  # assert val == 0x0
  amdgpu_wreg(0x31d2, 0x8) # gfx_v11_0_init_compute_vmid:1871:((adev->reg_offset[GC_HWIP][0][0] + 0x1f72))
  amdgpu_wreg(0xa900, 0xf0) # soc21_grbm_select:247:((adev->reg_offset[GC_HWIP][0][1] + 0x0900))
  amdgpu_wreg(0xa9e4, 0xc00c) # gfx_v11_0_init_compute_vmid:1865:((adev->reg_offset[GC_HWIP][0][1] + 0x09e4))
  amdgpu_wreg(0xa9e3, 0x10002) # gfx_v11_0_init_compute_vmid:1866:((adev->reg_offset[GC_HWIP][0][1] + 0x09e3))
  val = amdgpu_rreg(0x31d2) # gfx_v11_0_init_compute_vmid:1869:(adev->reg_offset[GC_HWIP][0][0] + 0x1f72)
  # assert val == 0x0
  amdgpu_wreg(0x31d2, 0x8) # gfx_v11_0_init_compute_vmid:1871:((adev->reg_offset[GC_HWIP][0][0] + 0x1f72))
  amdgpu_wreg(0xa900, 0x0) # soc21_grbm_select:247:((adev->reg_offset[GC_HWIP][0][1] + 0x0900))
  amdgpu_wreg(0x3310, 0x0) # gfx_v11_0_init_compute_vmid:1879:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * i)
  amdgpu_wreg(0x3311, 0x0) # gfx_v11_0_init_compute_vmid:1880:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * i)
  amdgpu_wreg(0x3328, 0x0) # gfx_v11_0_init_compute_vmid:1881:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + i)
  amdgpu_wreg(0x3338, 0x0) # gfx_v11_0_init_compute_vmid:1882:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + i)
  amdgpu_wreg(0x3312, 0x0) # gfx_v11_0_init_compute_vmid:1879:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * i)
  amdgpu_wreg(0x3313, 0x0) # gfx_v11_0_init_compute_vmid:1880:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * i)
  amdgpu_wreg(0x3329, 0x0) # gfx_v11_0_init_compute_vmid:1881:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + i)
  amdgpu_wreg(0x3339, 0x0) # gfx_v11_0_init_compute_vmid:1882:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + i)
  amdgpu_wreg(0x3314, 0x0) # gfx_v11_0_init_compute_vmid:1879:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * i)
  amdgpu_wreg(0x3315, 0x0) # gfx_v11_0_init_compute_vmid:1880:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * i)
  amdgpu_wreg(0x332a, 0x0) # gfx_v11_0_init_compute_vmid:1881:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + i)
  amdgpu_wreg(0x333a, 0x0) # gfx_v11_0_init_compute_vmid:1882:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + i)
  amdgpu_wreg(0x3316, 0x0) # gfx_v11_0_init_compute_vmid:1879:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * i)
  amdgpu_wreg(0x3317, 0x0) # gfx_v11_0_init_compute_vmid:1880:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * i)
  amdgpu_wreg(0x332b, 0x0) # gfx_v11_0_init_compute_vmid:1881:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + i)
  amdgpu_wreg(0x333b, 0x0) # gfx_v11_0_init_compute_vmid:1882:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + i)
  amdgpu_wreg(0x3318, 0x0) # gfx_v11_0_init_compute_vmid:1879:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * i)
  amdgpu_wreg(0x3319, 0x0) # gfx_v11_0_init_compute_vmid:1880:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * i)
  amdgpu_wreg(0x332c, 0x0) # gfx_v11_0_init_compute_vmid:1881:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + i)
  amdgpu_wreg(0x333c, 0x0) # gfx_v11_0_init_compute_vmid:1882:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + i)
  amdgpu_wreg(0x331a, 0x0) # gfx_v11_0_init_compute_vmid:1879:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * i)
  amdgpu_wreg(0x331b, 0x0) # gfx_v11_0_init_compute_vmid:1880:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * i)
  amdgpu_wreg(0x332d, 0x0) # gfx_v11_0_init_compute_vmid:1881:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + i)
  amdgpu_wreg(0x333d, 0x0) # gfx_v11_0_init_compute_vmid:1882:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + i)
  amdgpu_wreg(0x331c, 0x0) # gfx_v11_0_init_compute_vmid:1879:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * i)
  amdgpu_wreg(0x331d, 0x0) # gfx_v11_0_init_compute_vmid:1880:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * i)
  amdgpu_wreg(0x332e, 0x0) # gfx_v11_0_init_compute_vmid:1881:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + i)
  amdgpu_wreg(0x333e, 0x0) # gfx_v11_0_init_compute_vmid:1882:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + i)
  amdgpu_wreg(0x331e, 0x0) # gfx_v11_0_init_compute_vmid:1879:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * i)
  amdgpu_wreg(0x331f, 0x0) # gfx_v11_0_init_compute_vmid:1880:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * i)
  amdgpu_wreg(0x332f, 0x0) # gfx_v11_0_init_compute_vmid:1881:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + i)
  amdgpu_wreg(0x333f, 0x0) # gfx_v11_0_init_compute_vmid:1882:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + i)
  amdgpu_wreg(0x3302, 0x0) # gfx_v11_0_init_gds_vmid:1897:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * vmid)
  amdgpu_wreg(0x3303, 0x0) # gfx_v11_0_init_gds_vmid:1898:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * vmid)
  amdgpu_wreg(0x3321, 0x0) # gfx_v11_0_init_gds_vmid:1899:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + vmid)
  amdgpu_wreg(0x3331, 0x0) # gfx_v11_0_init_gds_vmid:1900:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + vmid)
  amdgpu_wreg(0x3304, 0x0) # gfx_v11_0_init_gds_vmid:1897:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * vmid)
  amdgpu_wreg(0x3305, 0x0) # gfx_v11_0_init_gds_vmid:1898:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * vmid)
  amdgpu_wreg(0x3322, 0x0) # gfx_v11_0_init_gds_vmid:1899:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + vmid)
  amdgpu_wreg(0x3332, 0x0) # gfx_v11_0_init_gds_vmid:1900:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + vmid)
  amdgpu_wreg(0x3306, 0x0) # gfx_v11_0_init_gds_vmid:1897:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * vmid)
  amdgpu_wreg(0x3307, 0x0) # gfx_v11_0_init_gds_vmid:1898:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * vmid)
  amdgpu_wreg(0x3323, 0x0) # gfx_v11_0_init_gds_vmid:1899:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + vmid)
  amdgpu_wreg(0x3333, 0x0) # gfx_v11_0_init_gds_vmid:1900:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + vmid)
  amdgpu_wreg(0x3308, 0x0) # gfx_v11_0_init_gds_vmid:1897:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * vmid)
  amdgpu_wreg(0x3309, 0x0) # gfx_v11_0_init_gds_vmid:1898:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * vmid)
  amdgpu_wreg(0x3324, 0x0) # gfx_v11_0_init_gds_vmid:1899:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + vmid)
  amdgpu_wreg(0x3334, 0x0) # gfx_v11_0_init_gds_vmid:1900:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + vmid)
  amdgpu_wreg(0x330a, 0x0) # gfx_v11_0_init_gds_vmid:1897:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * vmid)
  amdgpu_wreg(0x330b, 0x0) # gfx_v11_0_init_gds_vmid:1898:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * vmid)
  amdgpu_wreg(0x3325, 0x0) # gfx_v11_0_init_gds_vmid:1899:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + vmid)
  amdgpu_wreg(0x3335, 0x0) # gfx_v11_0_init_gds_vmid:1900:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + vmid)
  amdgpu_wreg(0x330c, 0x0) # gfx_v11_0_init_gds_vmid:1897:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * vmid)
  amdgpu_wreg(0x330d, 0x0) # gfx_v11_0_init_gds_vmid:1898:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * vmid)
  amdgpu_wreg(0x3326, 0x0) # gfx_v11_0_init_gds_vmid:1899:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + vmid)
  amdgpu_wreg(0x3336, 0x0) # gfx_v11_0_init_gds_vmid:1900:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + vmid)
  amdgpu_wreg(0x330e, 0x0) # gfx_v11_0_init_gds_vmid:1897:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * vmid)
  amdgpu_wreg(0x330f, 0x0) # gfx_v11_0_init_gds_vmid:1898:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * vmid)
  amdgpu_wreg(0x3327, 0x0) # gfx_v11_0_init_gds_vmid:1899:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + vmid)
  amdgpu_wreg(0x3337, 0x0) # gfx_v11_0_init_gds_vmid:1900:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + vmid)
  amdgpu_wreg(0x3310, 0x0) # gfx_v11_0_init_gds_vmid:1897:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * vmid)
  amdgpu_wreg(0x3311, 0x0) # gfx_v11_0_init_gds_vmid:1898:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * vmid)
  amdgpu_wreg(0x3328, 0x0) # gfx_v11_0_init_gds_vmid:1899:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + vmid)
  amdgpu_wreg(0x3338, 0x0) # gfx_v11_0_init_gds_vmid:1900:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + vmid)
  amdgpu_wreg(0x3312, 0x0) # gfx_v11_0_init_gds_vmid:1897:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * vmid)
  amdgpu_wreg(0x3313, 0x0) # gfx_v11_0_init_gds_vmid:1898:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * vmid)
  amdgpu_wreg(0x3329, 0x0) # gfx_v11_0_init_gds_vmid:1899:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + vmid)
  amdgpu_wreg(0x3339, 0x0) # gfx_v11_0_init_gds_vmid:1900:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + vmid)
  amdgpu_wreg(0x3314, 0x0) # gfx_v11_0_init_gds_vmid:1897:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * vmid)
  amdgpu_wreg(0x3315, 0x0) # gfx_v11_0_init_gds_vmid:1898:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * vmid)
  amdgpu_wreg(0x332a, 0x0) # gfx_v11_0_init_gds_vmid:1899:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + vmid)
  amdgpu_wreg(0x333a, 0x0) # gfx_v11_0_init_gds_vmid:1900:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + vmid)
  amdgpu_wreg(0x3316, 0x0) # gfx_v11_0_init_gds_vmid:1897:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * vmid)
  amdgpu_wreg(0x3317, 0x0) # gfx_v11_0_init_gds_vmid:1898:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * vmid)
  amdgpu_wreg(0x332b, 0x0) # gfx_v11_0_init_gds_vmid:1899:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + vmid)
  amdgpu_wreg(0x333b, 0x0) # gfx_v11_0_init_gds_vmid:1900:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + vmid)
  amdgpu_wreg(0x3318, 0x0) # gfx_v11_0_init_gds_vmid:1897:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * vmid)
  amdgpu_wreg(0x3319, 0x0) # gfx_v11_0_init_gds_vmid:1898:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * vmid)
  amdgpu_wreg(0x332c, 0x0) # gfx_v11_0_init_gds_vmid:1899:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + vmid)
  amdgpu_wreg(0x333c, 0x0) # gfx_v11_0_init_gds_vmid:1900:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + vmid)
  amdgpu_wreg(0x331a, 0x0) # gfx_v11_0_init_gds_vmid:1897:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * vmid)
  amdgpu_wreg(0x331b, 0x0) # gfx_v11_0_init_gds_vmid:1898:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * vmid)
  amdgpu_wreg(0x332d, 0x0) # gfx_v11_0_init_gds_vmid:1899:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + vmid)
  amdgpu_wreg(0x333d, 0x0) # gfx_v11_0_init_gds_vmid:1900:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + vmid)
  amdgpu_wreg(0x331c, 0x0) # gfx_v11_0_init_gds_vmid:1897:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * vmid)
  amdgpu_wreg(0x331d, 0x0) # gfx_v11_0_init_gds_vmid:1898:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * vmid)
  amdgpu_wreg(0x332e, 0x0) # gfx_v11_0_init_gds_vmid:1899:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + vmid)
  amdgpu_wreg(0x333e, 0x0) # gfx_v11_0_init_gds_vmid:1900:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + vmid)
  amdgpu_wreg(0x331e, 0x0) # gfx_v11_0_init_gds_vmid:1897:((adev->reg_offset[GC_HWIP][0][0] + 0x20a0) + 2 * vmid)
  amdgpu_wreg(0x331f, 0x0) # gfx_v11_0_init_gds_vmid:1898:((adev->reg_offset[GC_HWIP][0][0] + 0x20a1) + 2 * vmid)
  amdgpu_wreg(0x332f, 0x0) # gfx_v11_0_init_gds_vmid:1899:((adev->reg_offset[GC_HWIP][0][0] + 0x20c0) + vmid)
  amdgpu_wreg(0x333f, 0x0) # gfx_v11_0_init_gds_vmid:1900:((adev->reg_offset[GC_HWIP][0][0] + 0x20d0) + vmid)

# def gfx_v11_0_constants_init():
#   gfx_v11_0_setup_rb()

gfx_v11_0_config_gfx_rs64()
gfx_v11_0_gfxhub_enable()
gfx_v11_0_init_golden_registers()
gfx_v11_0_constants_init()

def nbio_v4_3_gc_doorbell_init():
  amdgpu_wreg(0x507a40, 0x30000007)
  amdgpu_wreg(0x507a43, 0x3000000d)

csb_addr = 0x85feafc000
csb_size = 0x3c0
def gfx_v11_0_init_csb():
  amdgpu_wreg(0xa988, csb_addr >> 32)
  amdgpu_wreg(0xa987, csb_addr & 0xffffffff)
  amdgpu_wreg(0xa989, csb_size)

def gfx_v11_0_rlc_enable_srm():
  amdgpu_wreg(0xec80, 0x3)

def gfx_v11_0_enable_gui_idle_interrupt():
  amdgpu_wreg(0x306a, 0x0)

def gfx_v11_0_cp_set_doorbell_range():
  amdgpu_wreg(0x305a, 0x458) # ring0
  amdgpu_wreg(0x305b, 0x7f8) 
  amdgpu_wreg(0x305c, 0x0) # kiq
  amdgpu_wreg(0x305d, 0x450)

def gfx_v11_0_cp_compute_enable():
  amdgpu_wreg(0xc904, 0x3c000000)

def gfx_v11_0_cp_gfx_enable():
  amdgpu_wreg(0xa803, 0x1000000)
  for i in range(1000000):
    val = amdgpu_rreg(0x21a0)
    print(hex(val))
    if val == 0: return
  raise Exception('gfx_v11_0_cp_gfx_enable timeout')

mes_uc_start_addr = [
  mes_2_hdr.mes_uc_start_addr_lo + (mes_2_hdr.mes_uc_start_addr_hi << 32),
  mes1_hdr.mes_uc_start_addr_lo + (mes1_hdr.mes_uc_start_addr_hi << 32)
]

mes_data_start_addr = [
  mes_2_hdr.mes_data_start_addr_lo + (mes_2_hdr.mes_data_start_addr_hi << 32),
  mes1_hdr.mes_data_start_addr_lo + (mes1_hdr.mes_data_start_addr_hi << 32)
]

def mes_v11_0_enable():
  # print([hex(x >> 2) for x in mes_uc_start_addr])
  amdgpu_wreg(0xc807, 0x40030000)  # regCP_MES_CNTL
  for pipe in range(2):
    soc21_grbm_select(3, pipe, 0, 0)
    amdgpu_wreg(0xc800, (mes_uc_start_addr[pipe] >> 2) & 0xffffffff) # mes_v11_0_enable:685:((adev->reg_offset[GC_HWIP][0][1] + 0x2800))
    amdgpu_wreg(0xc89d, (mes_uc_start_addr[pipe] >> 34)) # mes_v11_0_enable:687:((adev->reg_offset[GC_HWIP][0][1] + 0x289d))
  soc21_grbm_select(0, 0, 0, 0)
  amdgpu_wreg(0xc807, 0xc000000)

def mes_v11_0_kiq_setting():
  # TODO: pass ring to setup...
  amdgpu_wreg(0xa98a, 0x3068)
  amdgpu_wreg(0xa98a, 0x30e8)

class Amdring:
  def __init__(self):
    self.pipe = 0
    self.mqd_mv = memoryview(bytearray(ctypes.sizeof(amdgpu_2.struct_v11_compute_mqd)))
    self.mqd = amdgpu_2.struct_v11_compute_mqd.from_address(mv_address(self.mqd_mv))
    
    self.rptr_gpu_addr = 0xc0000000
    self.wptr_gpu_addr = 0xc0001000
    self.ring_gpu_addr = 0xc0004000
    self.mqd_gpu_addr = 0xd0000000
    self.hqd_gpu_addr = 0xd0100000
    self.eop_gpu_addr = 0xe0000000
    self.doorbell_index = 0

    self.next_ptr = 0x0

  def flush(self):
    for i, x in enumerate(self.mqd_mv): raw_vram[self.mqd_gpu_addr + i] = x

  def write(self, value):
    raw_vram.cast('I')[self.ring_gpu_addr // 4 + self.next_ptr] = value
    raw_vram.cast('I')[self.wptr_gpu_addr // 4] = self.next_ptr

def amdgpu_mm_wdoorbell64(index, value):
  doorbell_bar[index] = value

def mes_v11_0_mqd_init(ring):
  ring.mqd.header = 0xC0310800 # some magic?
  ring.mqd.compute_pipelinestat_enable = 0x00000001
  ring.mqd.compute_static_thread_mgmt_se0 = 0xffffffff
  ring.mqd.compute_static_thread_mgmt_se1 = 0xffffffff
  ring.mqd.compute_static_thread_mgmt_se2 = 0xffffffff
  ring.mqd.compute_static_thread_mgmt_se3 = 0xffffffff
  ring.mqd.compute_misc_reserved = 0x00000007
  
  eop_base_addr = ring.eop_gpu_addr >> 8

  ring.mqd.cp_hqd_eop_base_addr_lo = eop_base_addr & 0xffffffff
  ring.mqd.cp_hqd_eop_base_addr_hi = (eop_base_addr >> 32) & 0xffffffff
  ring.mqd.cp_hqd_eop_control = 0x8

  ring.mqd.cp_hqd_pq_rptr = 0
  ring.mqd.cp_hqd_pq_wptr_lo = 0
  ring.mqd.cp_hqd_pq_wptr_hi = 0

  ring.mqd.cp_mqd_base_addr_lo = ring.mqd_gpu_addr & 0xfffffffc
  ring.mqd.cp_mqd_base_addr_hi = (ring.mqd_gpu_addr >> 32) & 0xffffffff
  ring.mqd.cp_mqd_control = 0x40000060 # 

  ring.mqd.cp_hqd_pq_base_lo = ring.hqd_gpu_addr & 0xffffffff
  ring.mqd.cp_hqd_pq_base_hi = (ring.hqd_gpu_addr >> 32) & 0xffffffff

  ring.mqd.cp_hqd_pq_rptr_report_addr_lo = ring.rptr_gpu_addr & 0xfffffffc
  ring.mqd.cp_hqd_pq_rptr_report_addr_hi = (ring.rptr_gpu_addr >> 32) & 0xffffffff

  ring.mqd.cp_hqd_pq_wptr_report_addr_lo = ring.wptr_gpu_addr & 0xfffffffc
  ring.mqd.cp_hqd_pq_wptr_report_addr_hi = (ring.wptr_gpu_addr >> 32) & 0xffffffff
  ring.mqd.cp_hqd_pq_control = 0xd8308011

  ring.mqd.cp_hqd_pq_doorbell_control = 0x40000060
  ring.mqd.cp_hqd_vmid = 0
  ring.mqd.cp_hqd_active = 1

  ring.mqd.cp_hqd_persistent_state = 0xbe05501
  ring.mqd.cp_hqd_ib_control = 0x300000
  ring.mqd.cp_hqd_iq_timer = 0x0
  ring.mqd.cp_hqd_quantum = 0x0

def mes_v11_0_queue_init_register(ring):
  ring.flush()

  soc21_grbm_select(3, ring.pipe, 0, 0)
  
  amdgpu_wreg(0x320c, 0x0) # regCP_HQD_VMID
  amdgpu_wreg(0x3218, 0x0) # regCP_HQD_PQ_DOORBELL_CONTROL

  amdgpu_wreg(0x3209, ring.mqd.cp_mqd_base_addr_lo) # regCP_MQD_BASE_ADDR
  amdgpu_wreg(0x320a, ring.mqd.cp_mqd_base_addr_hi) # regCP_MQD_BASE_ADDR_HI

  amdgpu_wreg(0x322b, 0x0) # regCP_MQD_CONTROL

  amdgpu_wreg(0x3211, ring.mqd.cp_hqd_pq_base_lo) # regCP_HQD_PQ_BASE
  amdgpu_wreg(0x3212, ring.mqd.cp_hqd_pq_base_hi) # regCP_HQD_PQ_BASE_HI

  amdgpu_wreg(0x3214, ring.mqd.cp_hqd_pq_rptr_report_addr_lo) # regCP_HQD_PQ_RPTR_REPORT_ADDR
  amdgpu_wreg(0x3215, ring.mqd.cp_hqd_pq_rptr_report_addr_hi) # regCP_HQD_PQ_RPTR_REPORT_ADDR_HI

  assert ring.mqd.cp_hqd_pq_control == 0xd8308011
  amdgpu_wreg(0x321a, ring.mqd.cp_hqd_pq_control) # regCP_HQD_PQ_CONTROL

  amdgpu_wreg(0x3216, ring.mqd.cp_hqd_pq_wptr_poll_addr_lo) # regCP_HQD_PQ_WPTR_POLL_ADDR
  amdgpu_wreg(0x3217, ring.mqd.cp_hqd_pq_wptr_poll_addr_hi) # regCP_HQD_PQ_WPTR_POLL_ADDR_HI

  amdgpu_wreg(0x3218, ring.mqd.cp_hqd_pq_doorbell_control) # regCP_HQD_PQ_DOORBELL_CONTROL

  amdgpu_wreg(0x320d, ring.mqd.cp_hqd_persistent_state) # regCP_HQD_PERSISTENT_STATE

  amdgpu_wreg(0x320b, ring.mqd.cp_hqd_active) # regCP_HQD_ACTIVE

  soc21_grbm_select(0, 0, 0, 0)

the_ring = Amdring()
def mes_v11_0_queue_init(ring):
  mes_v11_0_mqd_init(ring)
  mes_v11_0_queue_init_register(ring)

def mes_v11_0_kiq_hw_init():
  mes_v11_0_enable()
  mes_v11_0_kiq_setting()
  mes_v11_0_queue_init(the_ring)

# nbio_v4_3_gc_doorbell_init()
gfx_v11_0_init_csb()
gfx_v11_0_rlc_enable_srm()
gfx_v11_0_enable_gui_idle_interrupt()
gfx_v11_0_cp_set_doorbell_range()
gfx_v11_0_cp_compute_enable()
gfx_v11_0_cp_gfx_enable()
mes_v11_0_kiq_hw_init()

from tinygrad.runtime.autogen import kfd, hsa, amd_gpu, libc
def gfx_v11_0_ring_test_ring():
  amdgpu_wreg(0x1fc00, 0x0) # hdp flush
  amdgpu_wreg(0xc040, 0xcafedead)

  the_ring.write(amd_gpu.PACKET3(amd_gpu.PACKET3_WRITE_DATA, 3))
  the_ring.write((1 << 16)) # kiq cmd?
  the_ring.write(0xc040)
  the_ring.write(0)
  the_ring.write(0xdeadc0de)

  amdgpu_mm_wdoorbell64(the_ring.doorbell_index, the_ring.next_ptr)

  while True:
    if raw_vram[the_ring.eop_gpu_addr] == 0xdeadc0de:
      break

gfx_v11_0_ring_test_ring()
print(raw_vram[0x0])

  # tmp = RREG32_SOC15(GC, 0, regCP_ME_CNTL);
	# tmp = REG_SET_FIELD(tmp, CP_ME_CNTL, PFP_PIPE0_RESET, 1);
	# tmp = REG_SET_FIELD(tmp, CP_ME_CNTL, PFP_PIPE1_RESET, 1);
	# WREG32_SOC15(GC, 0, regCP_ME_CNTL, tmp);

  


# amdgpu_wreg(0x5db4, 0x44440440)
# amdgpu_wreg(0x5db4, 0x44440440)
# amdgpu_wreg(0x5db4, 0x44440440)
# amdgpu_wreg(0x5db4, 0x44440440)
# amdgpu_wreg(0x5db4, 0x44440440)
# amdgpu_wreg(0x5db4, 0x44440440)
# amdgpu_wreg(0x5db4, 0x44440440)



# amdgpu_wreg(0x36b1, 0x0)
# assert amdgpu_rreg(0x0)
# amdgpu_wreg(0x36b1, 0x0)
# print(amdgpu_rreg(0x3802))
# while True:
#   print(amdgpu_rreg(0x36c0))


# GC_BASE__INST0_SEG0 = 0x00001260 // 4
# GC_BASE__INST0_SEG1 = 0x0000A000 // 4

# offset_is_hard = 0x0
# regTCP_CNTL = 0x19a2
# # print(hex(pci_mmio[0x2e680//4 + regTCP_CNTL]))
# regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32 = 0x16f4
# regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32 = 0x16f3


# pci_mmio[GC_BASE__INST0_SEG1 + 0x2040] = 0xCAFEDEAD
# print('d', hex(pci_mmio[GC_BASE__INST0_SEG1 + 0x2040]))

# # for i in range(16):
# #   print(hex(pci_mmio[0x168e + i]))

# def read_mmio_reg():
#   pass

# def write_mmio_reg():
#   pass

# # print(hex(pcidev.vendor_id))
