import os, ctypes
from tinygrad.runtime.autogen import libpciaccess
from tinygrad.helpers import to_mv

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


# while

def amdgpu_rreg(reg):
  return pci_mmio[reg]

def amdgpu_wreg(reg, val):
  pci_mmio[reg] = val

amdgpu_wreg(0x5db4, 0x44440440)
amdgpu_rreg(0x53e8)
amdgpu_wreg(0x53e8, 0x0)
print(hex(amdgpu_rreg(0x5db4)))
amdgpu_wreg(0x5db4, 0x44440440)
amdgpu_rreg(0x5db5)
amdgpu_wreg(0x5db5, 0x0)
print(hex(amdgpu_rreg(0x5db6)))
# TODO: they do the same several times, need this?
print(hex(amdgpu_rreg(0x358a)))
print(hex(amdgpu_rreg(0x36c0)))
amdgpu_wreg(0x36c0, 0x1)
amdgpu_wreg(0x36b6, 0x0)
print(hex(amdgpu_rreg(0x3802)))

# Golden regs init?
amdgpu_wreg(0x3802, 0x100)
amdgpu_wreg(0x3802, 0x100)
amdgpu_wreg(0x3697, 0x0)
amdgpu_wreg(0x3696, 0x0)
amdgpu_wreg(0x369f, 0x0)
amdgpu_wreg(0x369e, 0x0)
amdgpu_wreg(0x369b, 0x0)
amdgpu_wreg(0x369a, 0x0)
amdgpu_wreg(0x36a3, 0x0)
amdgpu_wreg(0x36b8, 0x0)
amdgpu_wreg(0x367b, 0xfe8e5400)
amdgpu_wreg(0x367c, 0x85)
amdgpu_wreg(0x3668, 0x63000000) # dmub bases setup
amdgpu_wreg(0x3670, 0x8300e200)
amdgpu_wreg(0x367d, 0xfe8f3600)
amdgpu_wreg(0x367e, 0x85)
amdgpu_wreg(0x3669, 0x64000000)
amdgpu_wreg(0x3671, 0x84004000)
amdgpu_wreg(0x367f, 0xfe8f7600)
amdgpu_wreg(0x3680, 0x85)
amdgpu_wreg(0x366a, 0x65000000)
amdgpu_wreg(0x3672, 0x85010040)
amdgpu_wreg(0x3658, 0xfe8f7600)
amdgpu_wreg(0x3659, 0x85)
amdgpu_wreg(0x3662, 0x8001003f)
amdgpu_wreg(0x3681, 0xfe907700)
amdgpu_wreg(0x3682, 0x85)
amdgpu_wreg(0x366b, 0x66000000)
amdgpu_wreg(0x3673, 0x8600b880)
amdgpu_wreg(0x3698, 0xa0000010)
amdgpu_wreg(0x3699, 0x10030)
amdgpu_wreg(0x3694, 0x64000000)
amdgpu_wreg(0x3695, 0x2000)
amdgpu_wreg(0x369c, 0x64002000)
amdgpu_wreg(0x369d, 0x2000)
amdgpu_wreg(0x36b1, 0x0)
assert amdgpu_rreg(0x36b1) == 0x0
amdgpu_wreg(0x36b1, 0x0)
assert amdgpu_rreg(0x3802) == 0x100
amdgpu_wreg(0x3802, 0x0)

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
