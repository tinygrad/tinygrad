import pathlib, re, ctypes, mmap, collections, functools, copy
import tinygrad.runtime.autogen.kfd as kfd
from tinygrad.helpers import from_mv, mv_address
from test.mockgpu.driver import PCIDesc, PCIRegion, VirtDriver, VirtFileDesc, TextFileDesc, DirFileDesc, VirtFile
from test.mockgpu.amd.amdgpu import AMDGPU, gpu_props

libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p

# class KFDFileDesc(VirtFileDesc):
#   def __init__(self, fd, driver):
#     super().__init__(fd)
#     self.driver = driver

#   def ioctl(self, fd, request, argp): return self.driver.kfd_ioctl(request, argp)
#   def mmap(self, start, sz, prot, flags, fd, offset): return offset


class AMDevice:
  def __init__(self, driver, vram_size):
    self.driver = driver
    self.vram_size = vram_size
    self.vram = memoryview(bytearray(vram_size))
    self.mmio = memoryview(bytearray(1 << 20))
    self.doorbells = memoryview(bytearray(2 << 20))
    self.reg_state = {}

    self._emu_boot()

  def _emu_boot(self):
    mmRCC_CONFIG_MEMSIZE = 0xde3
    self.reg_state[mmRCC_CONFIG_MEMSIZE] = self.vram_size >> 20

  def mmio_write(self, addr, value):
    print("mmio write", addr, value)
  def mmio_read(self, addr, value):
    print("mmio read", addr, value)

class BarDesc(VirtFileDesc):
  def __init__(self, amdevice, bar, dev, sz, rcb, wcb, fd):
    super().__init__(fd)
    self.amdevice, self.bar, self.dev, self.sz, self.rcb, self.wcb = amdevice, bar, dev, sz, rcb, wcb

  def mmap(self, start, sz, prot, flags, fd, offset):
    start = mv_address({0: self.amdevice.vram, 2: self.amdevice.doorbells, 5: self.amdevice.mmio}[self.bar])
    if self.rcb is not None: self.amdevice.driver.track_address(start, start + self.sz, self.rcb, self.wcb)
    return start

class AMDriver(VirtDriver):
  def __init__(self, gpus=6):
    super().__init__()

    regions = {0: PCIRegion(4 << 30), 2: PCIRegion(1 << 20), 5: PCIRegion(2 << 20)}
    self.pci_devs = [PCIDesc(self, vendor=0x1002, device=0x744c, domain=0, bus=i, slot=0, func=0, regions=regions) for i in range(gpus)]
    self.am_decs = [AMDevice(self, 4 << 30) for _ in range(gpus)]
    for a,d in zip(self.am_decs, self.pci_devs): self._prepare_pci(a, d)

    self.next_fd = (1 << 30)

  def probe(self, dev): pass
  def enable(self, dev): pass
  def cfg_read(self, dev, cmd): raise NotImplementedError()
  def cfg_write(self, dev, cmd, value): raise NotImplementedError()

  def _prepare_pci(self, am_dev, pci_dev):
    # self.doorbells[gpu_id] = memoryview(bytearray(0x2000))
    # self.gpus[gpu_id] = AMDGPU(gpu_id)

    pcibus = f"{pci_dev.domain:04x}:{pci_dev.bus:02x}:{pci_dev.slot:02x}.{pci_dev.func:d}"

    self.tracked_files += [
      VirtFile(f'/sys/bus/pci/devices/{pcibus}/resource0', functools.partial(BarDesc, am_dev, 0, pci_dev, (4 << 30), None, None)),
      VirtFile(f'/sys/bus/pci/devices/{pcibus}/resource2', functools.partial(BarDesc, am_dev, 2, pci_dev, (1 << 20),
        functools.partial(self._on_doorbell_read_pci, pci_dev), functools.partial(self._on_doorbell_write_pci, pci_dev))),
      VirtFile(f'/sys/bus/pci/devices/{pcibus}/resource5', functools.partial(BarDesc, am_dev, 5, pci_dev, (2 << 20),
        functools.partial(self._on_mmio_read_pci, pci_dev), functools.partial(self._on_mmio_write_pci, pci_dev))),
    ]

  def _alloc_fd(self):
    my_fd = self.next_fd
    self.next_fd = self.next_fd + 1
    return my_fd

  def _on_doorbell_read_pci(self, pci_dev, mv, index):
    print("doorbell read", pci_dev, mv, index)
  def _on_doorbell_write_pci(self, pci_dev, mv, index):
    print("doorbell write", pci_dev, mv, index)
  def _on_mmio_read_pci(self, pci_dev, mv, index):
    print("mmio read", pci_dev, mv, index)
  def _on_mmio_write_pci(self, pci_dev, mv, index):
    print("mmio write", pci_dev, mv, index)

  def open(self, name, flags, mode, virtfile): return virtfile.fdcls(self._alloc_fd())
