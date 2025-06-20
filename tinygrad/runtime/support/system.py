import os, mmap, array
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface
from tinygrad.helpers import getenv, round_up, DEBUG, to_mv
from tinygrad.runtime.autogen import libc

MAP_LOCKED = 0x2000
class _System:
  def __init__(self): self.pagemap = None

  def enable_hugepages(self, cnt): os.system(f"sudo sh -c 'echo {cnt} > /proc/sys/vm/nr_hugepages'")
  def alloc_sysmem(self, size, contiguous=False, data:bytes=None) -> tuple[int, list[int]]:
    if self.pagemap is None:
      # Disable migration of locked pages
      if FileIOInterface(reloc_sysfs:="/proc/sys/vm/compact_unevictable_allowed", os.O_RDONLY).read()[0] != "0":
        os.system(cmd:=f"sudo sh -c 'echo 0 > {reloc_sysfs}'")
        assert FileIOInterface(reloc_sysfs, os.O_RDONLY).read()[0] == "0", f"Failed to disable migration of locked pages. Please run {cmd} manually."

      self.pagemap = FileIOInterface("/proc/self/pagemap", os.O_RDONLY)
    
    size = round_up(size, mmap.PAGESIZE)

    assert not contiguous or size <= (2 << 20), "Contiguous allocation is only supported for sizes <= 2 MiB"

    flags = libc.MAP_HUGETLB if contiguous and size > 0x1000 else 0
    va = FileIOInterface.anon_mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | MAP_LOCKED | flags, 0)

    # Read pagemap to get the physical address of each page. The pages are locked.
    self.pagemap.seek(va // mmap.PAGESIZE * 8)
    if data is not None: to_mv(va, len(data))[:] = data

    return va, [(x & ((1<<55) - 1)) * mmap.PAGESIZE for x in array.array('Q', self.pagemap.read(size//mmap.PAGESIZE*8, binary=True))]

  def pci_scan_bus(self, target_vendor, target_devices):
    result = []
    for pcibus in FileIOInterface("/sys/bus/pci/devices").listdir():
      vendor = int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/vendor").read(), 16)
      device = int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/device").read(), 16)
      if vendor == target_vendor and device in target_devices: result.append(pcibus)
    return result

System = _System()

class PCIDevice:
  def __init__(self, pcibus, bars, resize_bars=None):
    self.pcibus = pcibus

    if FileIOInterface.exists(f"/sys/bus/pci/devices/{self.pcibus}/driver"):
      FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/driver/unbind", os.O_WRONLY).write(self.pcibus)

    for i in resize_bars or []:
      supported_sizes = int(FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/resource{i}_resize", os.O_RDONLY).read(), 16)
      try: FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/resource{i}_resize", os.O_RDWR).write(str(supported_sizes.bit_length() - 1))
      except OSError as e: raise RuntimeError(f"Cannot resize BAR {i}: {e}. Ensure the resizable BAR option is enabled on your system.") from e

    self.cfg_fd = FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/config", os.O_RDWR | os.O_SYNC | os.O_CLOEXEC)
    self.bar_fds = {b: FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/resource{b}", os.O_RDWR | os.O_SYNC | os.O_CLOEXEC) for b in bars}

    bar_info = FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/resource", os.O_RDONLY).read().splitlines()
    self.bar_info = {j:(int(start,16), int(end,16), int(flgs,16)) for j,(start,end,flgs) in enumerate(l.split() for l in bar_info)}

    # TODO: VFIO
    FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/enable", os.O_RDWR).write("1")

  def read_config(self, offset, size): return int.from_bytes(self.cfg_fd.read(size, binary=True, offset=offset), byteorder='little')
  def write_config(self, offset, value, size): self.cfg_fd.write(value.to_bytes(size, byteorder='little'), binary=True, offset=offset)
  def map_bar(self, bar, off=0, addr=0, size=None, fmt='B') -> MMIOInterface:
    fd, sz = self.bar_fds[bar], size or (self.bar_info[bar][1] - self.bar_info[bar][0] + 1)
    libc.madvise(loc:=fd.mmap(addr, sz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if addr else 0), off), sz, libc.MADV_DONTFORK)
    assert loc != 0xffffffffffffffff, f"Failed to mmap {size} bytes at {hex(addr)}"
    return MMIOInterface(loc, sz, fmt=fmt)
