import os, mmap
from tinygrad.runtime.autogen import libc
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface

dev = None
for pcibus in FileIOInterface("/sys/bus/pci/devices").listdir():
  vendor = int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/vendor").read(), 16)
  device = int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/device").read(), 16)
  if vendor == 0x10de and device == 0x2b85: dev = pcibus

pcibus = dev

if FileIOInterface.exists(f"/sys/bus/pci/devices/{pcibus}/driver"):
  FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/driver/unbind", os.O_WRONLY).write(pcibus)

cfg_fd = FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/config", os.O_RDWR | os.O_SYNC | os.O_CLOEXEC)
bar_fds = {b: FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/resource{b}", os.O_RDWR | os.O_SYNC | os.O_CLOEXEC) for b in [0, 1, 3]}

bar_info = FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/resource", os.O_RDONLY).read().splitlines()
bar_info = {j:(int(start,16), int(end,16), int(flgs,16)) for j,(start,end,flgs) in enumerate(l.split() for l in bar_info)}
print(bar_info)

def _map_pci_range(bar, off=0, addr=0, size=None, fmt='B'):
  fd, sz = bar_fds[bar], size or (bar_info[bar][1] - bar_info[bar][0] + 1)
  libc.madvise(loc:=fd.mmap(addr, sz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if addr else 0), off), sz, libc.MADV_DONTFORK)
  assert loc != 0xffffffffffffffff, f"Failed to mmap {size} bytes at {hex(addr)}"
  return MMIOInterface(loc, sz, fmt=fmt)

regs = _map_pci_range(0, fmt='I')
fb = _map_pci_range(1)

fwpath = "/lib/firmware/nvidia/570.133.20/gsp_ga10x.bin"
fwbytes = FileIOInterface(fwpath, os.O_RDONLY).read(binary=True)
assert len(fwbytes) == 63534832

def wreg(addr, value): regs[addr // 4] = value
def rreg(addr): return regs[addr // 4]

pmc_boot_1 = rreg(0x00000004)
pmc_boot_0 = rreg(0x00000000)
pmc_boot_42 = rreg(0x00000A00)

print(hex(pmc_boot_42))

# prapare for bootstrap
# NV_PGSP_FALCON_ENGINE = 0x1103c0
# print(hex(rreg(NV_PGSP_FALCON_ENGINE)))
# exit(0)

# wreg(NV_PGSP_FALCON_ENGINE, rreg(NV_PGSP_FALCON_ENGINE) & ~0x1)
# while ((rreg(NV_PGSP_FALCON_ENGINE) >> 8) & 0b11) != 0b10:
#   print(hex(rreg(NV_PGSP_FALCON_ENGINE)))

# print("reset done")
# wreg(NV_PGSP_FALCON_ENGINE, rreg(NV_PGSP_FALCON_ENGINE) | 0x1)

# kfsp path

