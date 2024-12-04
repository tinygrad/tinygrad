import os, ctypes, mmap, struct, subprocess, select, fcntl
from tinygrad.helpers import getenv, mv_address, to_mv
from tinygrad.runtime.autogen import libpciaccess, vfio, libc

AM_DEBUG = getenv('AM_DEBUG', 0)

def scan_pci_devs():
  libpciaccess.pci_system_init()
  pci_iter = libpciaccess.pci_id_match_iterator_create(None)

  gpus = []
  while pcidev:=libpciaccess.pci_device_next(pci_iter):
    if pcidev.contents.vendor_id == 0x1002 and pcidev.contents.device_id == 0x744c: gpus.append(pcidev.contents)

  visible_devices = [int(x) for x in (getenv('VISIBLE_DEVICES', getenv('HIP_VISIBLE_DEVICES', ''))).split(',') if x.strip()]
  return [gpus[x] for x in visible_devices] if visible_devices else gpus

def read_pagemap(va):
  with open("/proc/self/pagemap", "rb") as pagemap:
    pagemap.seek(va // mmap.PAGESIZE * 8)

    entry = pagemap.read(8)
    if len(entry) != 8: return None

    entry_value = struct.unpack("Q", entry)[0]
    present = (entry_value >> 63) & 1
    swapped = (entry_value >> 62) & 1
    page_frame_number = entry_value & ((1 << 55) - 1)

    return None if not present or swapped else page_frame_number * mmap.PAGESIZE

class SystemMemory:
  def __init__(self, vaddr:int, size:int): self.vaddr, self.size = vaddr, size
  def phys_pages(self): return [read_pagemap(self.vaddr + i * mmap.PAGESIZE) for i in range(self.size // mmap.PAGESIZE)]

class HAL:
  def __init__(self, has_dma, has_iommu): self.has_dma, self.has_iommu = has_dma, has_iommu
  def open_device(self): raise NotImplementedError

class PCIHAL(HAL):
  def __init__(self, has_dma=True, has_iommu=False): 
    self.devs = scan_pci_devs()
    super().__init__(has_dma=has_dma, has_iommu=has_iommu)

  def open_device(self, dev_idx:int):
    pcidev = self.devs[dev_idx]
    if AM_DEBUG >= 1: print(f"Opening device {pcidev.domain_16:04x}:{pcidev.bus:02x}:{pcidev.dev:02x}.{pcidev.func:d}")
    libpciaccess.pci_device_probe(ctypes.byref(pcidev))
    libpciaccess.pci_device_enable(ctypes.byref(pcidev))
    return pcidev

  def vram_pci_addr(self, pcidev): return pcidev.regions[0].base_addr

  def map_pci_range(self, pcidev, bar, cast='I'):
    ret = libpciaccess.pci_device_map_range(ctypes.byref(pcidev), pcidev.regions[bar].base_addr, size:=pcidev.regions[bar].size,
      libpciaccess.PCI_DEV_MAP_FLAG_WRITABLE, ctypes.byref(pcimem:=ctypes.c_void_p()))
    return pcimem.value, to_mv(pcimem.value, size).cast(cast)

  def pci_set_master(self, pcidev):
    # TODO: parse from linux/include/uapi/linux/pci_regs.h
    libpciaccess.pci_device_cfg_read_u16(pcidev, ctypes.byref(val:=ctypes.c_uint16()), 0x4)
    libpciaccess.pci_device_cfg_write_u16(pcidev, val.value | 0x4, 0x4)

  def alloc_pinned_memory(self, sz:int) -> SystemMemory:
    va = libc.mmap(0, sz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | mmap.MAP_ANONYMOUS, -1, 0)
    libc.mlock(va, sz)
    return SystemMemory(va, sz)

class VFIOHAL(PCIHAL):
  vfio_fd:int = -1
  
  def __init__(self):
    if VFIOHAL.vfio_fd == -1:
      subprocess.run(['modprobe', 'vfio_pci'], capture_output=True, text=True, check=True)
      with open("/sys/module/vfio/parameters/enable_unsafe_noiommu_mode", 'w') as f: f.write("1")
      VFIOHAL.vfio_fd = os.open("/dev/vfio/vfio", os.O_RDWR)
      self.irq_poller = select.poll()
    assert vfio.VFIO_CHECK_EXTENSION(VFIOHAL.vfio_fd, vfio.VFIO_NOIOMMU_IOMMU), "VFIO does not support IOMMU"

    super().__init__(has_dma=True)

  def open_device(self, dev_idx:int):
    pcidev = self.devs[dev_idx]
    self.pcibus = f"{pcidev.domain_16:04x}:{pcidev.bus:02x}:{pcidev.dev:02x}.{pcidev.func:d}"
    if AM_DEBUG >= 1: print(f"Opening device (vfio) {self.pcibus}")

    if os.path.exists(f"/sys/bus/pci/devices/{self.pcibus}/driver"):
      with open(f"/sys/bus/pci/devices/{self.pcibus}/driver/unbind", 'w') as f: f.write(self.pcibus)
    with open(f"/sys/bus/pci/devices/{self.pcibus}/resource0_resize", 'w') as f: f.write("15")
    with open(f"/sys/bus/pci/devices/{self.pcibus}/driver_override", 'w') as f: f.write("vfio-pci")
    with open(f"/sys/bus/pci/drivers_probe", 'w') as f: f.write(self.pcibus)

    iommu_group = os.readlink(f"/sys/bus/pci/devices/{self.pcibus}/iommu_group").split('/')[-1]

    self.vfio_group = os.open(f"/dev/vfio/noiommu-{iommu_group}", os.O_RDWR)
    vfio.VFIO_GROUP_SET_CONTAINER(self.vfio_group, ctypes.c_int(VFIOHAL.vfio_fd))

    vfio.VFIO_SET_IOMMU(VFIOHAL.vfio_fd, vfio.VFIO_NOIOMMU_IOMMU)
    self.vfio_dev = vfio.VFIO_GROUP_GET_DEVICE_FD(self.vfio_group, (ctypes.c_char * (len(self.pcibus) + 1))(*bytearray(self.pcibus.encode() + b'\0')))

    self.irq_fd = os.eventfd(0, 0)
    self.irq_poller.register(self.irq_fd, select.POLLIN)

    irqs = vfio.struct_vfio_irq_set(index=vfio.VFIO_PCI_MSI_IRQ_INDEX, flags=vfio.VFIO_IRQ_SET_DATA_EVENTFD|vfio.VFIO_IRQ_SET_ACTION_TRIGGER,
      argsz=ctypes.sizeof(vfio.struct_vfio_irq_set), count=1, data=(ctypes.c_int * 1)(self.irq_fd))
    vfio.VFIO_DEVICE_SET_IRQS(self.vfio_dev, irqs)

    return pcidev

  def map_pci_range(self, pcidev, bar, cast='I'):
    vfio.VFIO_DEVICE_GET_REGION_INFO(self.vfio_dev, reg:=vfio.struct_vfio_region_info(argsz=ctypes.sizeof(vfio.struct_vfio_region_info), index=bar))
    addr = libc.mmap(0, reg.size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, self.vfio_dev, reg.offset)
    return addr, to_mv(addr, reg.size).cast(cast)
