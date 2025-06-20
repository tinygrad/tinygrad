import os, mmap, array, functools, ctypes, select, contextlib
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface
from tinygrad.helpers import round_up, to_mv, getenv, OSX
from tinygrad.runtime.autogen import libc, vfio

MAP_FIXED, MAP_LOCKED, MAP_POPULATE, MAP_NORESERVE = 0x10, 0 if OSX else 0x2000, getattr(mmap, "MAP_POPULATE", 0 if OSX else 0x008000), 0x400

class _System:
  def alloc_sysmem(self, size:int, vaddr:int=0, contiguous:bool=False, data:bytes|None=None) -> tuple[int, list[int]]:
    assert not contiguous or size <= (2 << 20), "Contiguous allocation is only supported for sizes up to 2MB"
    flags = (libc.MAP_HUGETLB if contiguous and (size:=round_up(size, mmap.PAGESIZE)) > 0x1000 else 0) | (MAP_FIXED if vaddr else 0)
    va = FileIOInterface.anon_mmap(vaddr, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|mmap.MAP_ANONYMOUS|MAP_POPULATE|MAP_LOCKED|flags, 0)

    if data is not None: to_mv(va, len(data))[:] = data

    # Read pagemap to get the physical address of each page. The pages are locked.
    self.pagemap().seek(va // mmap.PAGESIZE * 8)
    return va, [(x & ((1<<55) - 1)) * mmap.PAGESIZE for x in array.array('Q', self.pagemap().read(size//mmap.PAGESIZE*8, binary=True))]

  def pci_scan_bus(self, target_vendor:int, target_devices:list[int]) -> list[str]:
    result = []
    for pcibus in FileIOInterface("/sys/bus/pci/devices").listdir():
      vendor = int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/vendor").read(), 16)
      device = int(FileIOInterface(f"/sys/bus/pci/devices/{pcibus}/device").read(), 16)
      if vendor == target_vendor and device in target_devices: result.append(pcibus)
    return sorted(result)

  @functools.cache
  def pagemap(self) -> FileIOInterface:
    if FileIOInterface(reloc_sysfs:="/proc/sys/vm/compact_unevictable_allowed", os.O_RDONLY).read()[0] != "0":
      os.system(cmd:=f"sudo sh -c 'echo 0 > {reloc_sysfs}'")
      assert FileIOInterface(reloc_sysfs, os.O_RDONLY).read()[0] == "0", f"Failed to disable migration of locked pages. Please run {cmd} manually."
    return FileIOInterface("/proc/self/pagemap", os.O_RDONLY)

  @functools.cache
  def vfio(self) -> FileIOInterface|None:
    try:
      if not FileIOInterface.exists("/sys/module/vfio"): os.system("sudo modprobe vfio-pci disable_idle_d3=1")

      FileIOInterface("/sys/module/vfio/parameters/enable_unsafe_noiommu_mode", os.O_RDWR).write("1")
      vfio_fd = FileIOInterface("/dev/vfio/vfio", os.O_RDWR)
      vfio.VFIO_CHECK_EXTENSION(vfio_fd, vfio.VFIO_NOIOMMU_IOMMU)

      return vfio_fd
    except OSError: return None

System = _System()

class PCIDevice:
  def __init__(self, pcibus:str, bars:list[int], resize_bars:list[int]|None=None):
    self.pcibus, self.irq_poller = pcibus, None

    if FileIOInterface.exists(f"/sys/bus/pci/devices/{self.pcibus}/driver"):
      FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/driver/unbind", os.O_WRONLY).write(self.pcibus)

    for i in resize_bars or []:
      supported_sizes = int(FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/resource{i}_resize", os.O_RDONLY).read(), 16)
      try: FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/resource{i}_resize", os.O_RDWR).write(str(supported_sizes.bit_length() - 1))
      except OSError as e: raise RuntimeError(f"Cannot resize BAR {i}: {e}. Ensure the resizable BAR option is enabled on your system.") from e

    if getenv("VFIO", 0) and (vfio_fd:=System.vfio()) is not None:
      FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/driver_override", os.O_WRONLY).write("vfio-pci")
      FileIOInterface("/sys/bus/pci/drivers_probe", os.O_WRONLY).write(self.pcibus)
      iommu_group = FileIOInterface.readlink(f"/sys/bus/pci/devices/{self.pcibus}/iommu_group").split('/')[-1]

      self.vfio_group = FileIOInterface(f"/dev/vfio/noiommu-{iommu_group}", os.O_RDWR)
      vfio.VFIO_GROUP_SET_CONTAINER(self.vfio_group, ctypes.c_int(vfio_fd.fd))

      with contextlib.suppress(OSError): vfio.VFIO_SET_IOMMU(vfio_fd, vfio.VFIO_NOIOMMU_IOMMU) # set iommu works only once for the fd.
      self.vfio_dev = FileIOInterface(fd=vfio.VFIO_GROUP_GET_DEVICE_FD(self.vfio_group, ctypes.create_string_buffer(self.pcibus.encode())))

      self.irq_fd = FileIOInterface.eventfd(0, 0)
      self.irq_poller = select.poll()
      self.irq_poller.register(self.irq_fd.fd, select.POLLIN)

      irqs = vfio.struct_vfio_irq_set(index=vfio.VFIO_PCI_MSI_IRQ_INDEX, flags=vfio.VFIO_IRQ_SET_DATA_EVENTFD|vfio.VFIO_IRQ_SET_ACTION_TRIGGER,
        argsz=ctypes.sizeof(vfio.struct_vfio_irq_set), count=1, data=(ctypes.c_int * 1)(self.irq_fd.fd))
      vfio.VFIO_DEVICE_SET_IRQS(self.vfio_dev, irqs)
    else: FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/enable", os.O_RDWR).write("1")

    self.cfg_fd = FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/config", os.O_RDWR | os.O_SYNC | os.O_CLOEXEC)
    self.bar_fds = {b: FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/resource{b}", os.O_RDWR | os.O_SYNC | os.O_CLOEXEC) for b in bars}

    bar_info = FileIOInterface(f"/sys/bus/pci/devices/{self.pcibus}/resource", os.O_RDONLY).read().splitlines()
    self.bar_info = {j:(int(start,16), int(end,16), int(flgs,16)) for j,(start,end,flgs) in enumerate(l.split() for l in bar_info)}

  def read_config(self, offset:int, size:int): return int.from_bytes(self.cfg_fd.read(size, binary=True, offset=offset), byteorder='little')
  def write_config(self, offset:int, value:int, size:int): self.cfg_fd.write(value.to_bytes(size, byteorder='little'), binary=True, offset=offset)
  def map_bar(self, bar:int, off:int=0, addr:int=0, size:int|None=None, fmt='B') -> MMIOInterface:
    fd, sz = self.bar_fds[bar], size or (self.bar_info[bar][1] - self.bar_info[bar][0] + 1)
    libc.madvise(loc:=fd.mmap(addr, sz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if addr else 0), off), sz, libc.MADV_DONTFORK)
    return MMIOInterface(loc, sz, fmt=fmt)
