import ctypes, ctypes.util, mmap, functools, os
from test.mockgpu.driver import VirtDriver, VirtFileDesc, TextFileDesc, DirFileDesc, VirtFile
from test.mockgpu.am.amgpu import MockAMGPU, VRAM_SIZE

DOORBELL_SIZE = 0x2000
BAR5_SIZE = 0x10000000  # 256MB MMIO
PCIBUS = "mock:am:0"

libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p

# Resource file: 7 lines (BAR 0-5 + ROM), format: "start end flags"
_resource_lines = [
  f"0x0000000000000000 0x{VRAM_SIZE-1:016x} 0x0000000000000000",      # BAR 0: VRAM
  "0x0000000000000000 0x0000000000000000 0x0000000000000000",          # BAR 1: empty
  f"0x0000000000000000 0x{DOORBELL_SIZE-1:016x} 0x0000000000000000",   # BAR 2: doorbell
  "0x0000000000000000 0x0000000000000000 0x0000000000000000",          # BAR 3: empty
  "0x0000000000000000 0x0000000000000000 0x0000000000000000",          # BAR 4: empty
  f"0x0000000000000000 0x{BAR5_SIZE-1:016x} 0x0000000000000000",       # BAR 5: MMIO
  "0x0000000000000000 0x0000000000000000 0x0000000000000000",          # ROM
]

class PagemapFileDesc(VirtFileDesc):
  """Fake /proc/self/pagemap — returns pagemap entries with fake PFNs."""
  def __init__(self, fd, gpu):
    super().__init__(fd)
    self.gpu = gpu
  def seek(self, offset): self.off = offset
  def read_contents(self, size=None):
    entries = bytearray()
    for i in range((size or 8) // 8):
      vaddr = ((self.off // 8) + i) * 0x1000
      paddr = self.gpu._next_sysmem_paddr
      self.gpu._next_sysmem_paddr += 0x1000
      self.gpu._sysmem_map[paddr] = vaddr
      entries += ((1 << 63) | (paddr // 0x1000)).to_bytes(8, 'little')
    self.off += len(entries)
    return bytes(entries)

class PCIBarFileDesc(VirtFileDesc):
  """PCI BAR resource file — mmap returns GPU memory."""
  def __init__(self, fd, memfd, driver=None):
    super().__init__(fd)
    self.memfd, self.driver = memfd, driver
  def mmap(self, start, sz, prot, flags, fd, off):
    addr = libc.mmap(start, sz, prot, flags, self.memfd, off)
    if self.driver is not None:  # doorbell: track new mmap for write callbacks
      self.driver.track_address(addr, addr + sz, lambda mv, idx: None, lambda mv, idx: self.driver._emulate_execute())
    return addr

class PCIMMIOBarFileDesc(VirtFileDesc):
  """PCI BAR 5 resource file — mmap returns pre-allocated tracked MMIO memory."""
  def __init__(self, fd, bar5_addr):
    super().__init__(fd)
    self.bar5_addr = bar5_addr
  def mmap(self, start, sz, prot, flags, fd, off): return self.bar5_addr + off

class PCIConfigFileDesc(VirtFileDesc):
  """PCI config space — binary read/write."""
  def __init__(self, fd):
    super().__init__(fd)
    self.data = bytearray(256)
  def read_contents(self, size=None): return bytes(self.data[self.off:self.off + (size or len(self.data) - self.off)])
  def write_contents(self, content): self.data[self.off:self.off + len(content)] = content
  def seek(self, offset): self.off = offset

class PCIEnableFileDesc(VirtFileDesc):
  """PCI enable file — writable text."""
  def __init__(self, fd):
    super().__init__(fd)
  def read_contents(self, size=None): return "1\n"
  def write_contents(self, content): pass  # no-op

class AMDriver(VirtDriver):
  """Mock driver for AM (PCI baremetal) path using sysfs emulation."""
  def __init__(self):
    super().__init__()
    self.gpus:dict[int, MockAMGPU] = {}
    self._executing = False
    self.gpu = MockAMGPU(0)
    self.gpus[0] = self.gpu
    self.next_fd = 1 << 30

    # Allocate BAR 5 (MMIO) backed by real memory, tracked for register emulation
    self._bar5_addr = libc.mmap(0, BAR5_SIZE, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | mmap.MAP_ANONYMOUS, -1, 0)
    mmio = self.gpu.mmio
    self.track_address(self._bar5_addr, self._bar5_addr + BAR5_SIZE,
      lambda mv, idx: _bar5_sync_read(mv, idx, mmio), lambda mv, idx: _bar5_sync_write(mv, idx, mmio))

    # Register sysfs/proc files
    p = f"/sys/bus/pci/devices/{PCIBUS}"
    self.tracked_files += [
      VirtFile("/proc/sys/vm/compact_unevictable_allowed", functools.partial(TextFileDesc, text="0\n")),
      VirtFile("/proc/self/pagemap", functools.partial(PagemapFileDesc, gpu=self.gpu)),
      # pci_scan_bus
      VirtFile("/sys/bus/pci/devices", functools.partial(DirFileDesc, child_names=[PCIBUS])),
      VirtFile(f"{p}/vendor", functools.partial(TextFileDesc, text="0x1002\n")),
      VirtFile(f"{p}/device", functools.partial(TextFileDesc, text="0x74a1\n")),
      # PCIDevice.__init__ sysfs files
      VirtFile(f"{p}/enable", PCIEnableFileDesc),
      VirtFile(f"{p}/config", PCIConfigFileDesc),
      VirtFile(f"{p}/resource", functools.partial(TextFileDesc, text="\n".join(_resource_lines) + "\n")),
      VirtFile(f"{p}/resource0", functools.partial(PCIBarFileDesc, memfd=self.gpu.vram_fd)),
      VirtFile(f"{p}/resource2", functools.partial(PCIBarFileDesc, memfd=self.gpu.doorbell_fd, driver=self)),
      VirtFile(f"{p}/resource5", functools.partial(PCIMMIOBarFileDesc, bar5_addr=self._bar5_addr)),
    ]

  def _alloc_fd(self):
    fd = self.next_fd
    self.next_fd += 1
    return fd

  def open(self, name, flags, mode, virtfile): return virtfile.fdcls(self._alloc_fd())

  def _emulate_execute(self):
    if self._executing: return
    self._executing = True
    try:
      any_progress = True
      while any_progress:
        any_progress = False
        for gpu in self.gpus.values():
          for q in gpu.queues:
            if q.executing: any_progress |= q.execute() > 0
    finally:
      self._executing = False

def _bar5_sync_read(mv, idx, mmio):
  if isinstance(idx, slice):
    for i in range(idx.start or 0, idx.stop or len(mv), idx.step or 1): mv[i] = mmio[i]
  else: mv[idx] = mmio[idx]

def _bar5_sync_write(mv, idx, mmio):
  if isinstance(idx, slice):
    for i in range(idx.start or 0, idx.stop or len(mv), idx.step or 1): mmio[i] = mv[i]
  else: mmio[idx] = mv[idx]
