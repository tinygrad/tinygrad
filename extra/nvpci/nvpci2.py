import os, mmap, re, array, gzip, struct, ctypes, time, subprocess
from tinygrad.helpers import fetch, to_mv, round_up, getenv
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.autogen import libc, pci
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface
from tinygrad.runtime.support.nv.nvdev import NVDev
from hexdump import hexdump

ADA = 1

os.system(cmd:=f"sudo sh -c 'echo 0 > /proc/sys/vm/compact_unevictable_allowed'")
os.system(cmd:=f"sudo sh -c 'echo 8 > /proc/sys/vm/nr_hugepages'")

dev = None
for pcibus in FileIOInterface("/sys/bus/pci/devices").listdir():
  vendor = int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/vendor").read(), 16)
  device = int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/device").read(), 16)
  if vendor == 0x10de and device == (0x2684 if ADA else 0x2b85): dev = pcibus

pcibus = dev

if FileIOInterface.exists(f"/sys/bus/pci/devices/{pcibus}/driver"):
  FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/driver/unbind", os.O_WRONLY).write(pcibus)

# FileIOInterface("/sys/module/vfio/parameters/enable_unsafe_noiommu_mode", os.O_RDWR).write("1")
# FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/driver_override", os.O_WRONLY).write("vfio-pci")
# FileIOInterface("/sys/bus/pci/drivers_probe", os.O_WRONLY).write(pcibus)

FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/enable", os.O_RDWR).write("1")

cfg_fd = FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/config", os.O_RDWR | os.O_SYNC | os.O_CLOEXEC)
bar_fds = {b: FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/resource{b}", os.O_RDWR | os.O_SYNC | os.O_CLOEXEC) for b in [0, 1, 3]}

bar_info = FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/resource", os.O_RDONLY).read().splitlines()
bar_info = {j:(int(start,16), int(end,16), int(flgs,16)) for j,(start,end,flgs) in enumerate(l.split() for l in bar_info)}

cfg_fd = FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/config", os.O_RDWR | os.O_SYNC | os.O_CLOEXEC)
pci_cmd = int.from_bytes(cfg_fd.read(2, binary=True, offset=pci.PCI_COMMAND), byteorder='little') | pci.PCI_COMMAND_MASTER
cfg_fd.write(pci_cmd.to_bytes(2, byteorder='little'), binary=True, offset=pci.PCI_COMMAND)
print('pci cfg', hex(pci_cmd))

def _map_pci_range(bar, off=0, addr=0, size=None, fmt='B'):
  fd, sz = bar_fds[bar], size or (bar_info[bar][1] - bar_info[bar][0] + 1)
  libc.madvise(loc:=fd.mmap(addr, sz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if addr else 0), off), sz, libc.MADV_DONTFORK)
  assert loc != 0xffffffffffffffff, f"Failed to mmap {size} bytes at {hex(addr)}"
  return MMIOInterface(loc, sz, fmt=fmt)

regs = _map_pci_range(0, fmt='I')
fb = _map_pci_range(1)

nvdev = NVDev(pcibus, regs, fb, None)
