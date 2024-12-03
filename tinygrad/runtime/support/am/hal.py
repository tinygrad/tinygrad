import os, ctypes, mmap, struct
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

# class VFIOHAL(PCIHAL):
#   vfio_fd = -1
  
#   def __init__(self): 
#     if VFIOHAL.vfio_fd == -1: VFIOHAL.vfio_fd = os.open("/dev/vfio/vfio", os.O_RDWR)
#     assert vfio.VFIO_CHECK_EXTENSION(VFIOHAL.vfio_fd, vfio.VFIO_NOIOMMU_IOMMU), "VFIO does not support IOMMU"
#     # assert vfio.VFIO_SET_IOMMU(VFIOHAL.vfio_fd, vfio.VFIO_NOIOMMU_IOMMU) == 0, "Failed to set IOMMU"

#     super().__init__(has_dma=True)

#   def open_device(self, dev_idx:int):
#     # readlink /sys/bus/pci/devices/0000:06:0d.0/iommu_group
#     self.pcidev = self.devs[dev_idx]
  
#     self.pcifmt = "{:04x}:{:02x}:{:02x}.{:d}".format(self.pcidev.domain_16, self.pcidev.bus, self.pcidev.dev, self.pcidev.func)
#     iommu_group = os.readlink(f"/sys/bus/pci/devices/{self.pcifmt}/iommu_group").split('/')[-1]
#     print(iommu_group)

#     self.vfio_group = os.open(f"/dev/vfio/noiommu-{iommu_group}", os.O_RDWR)
#     vfio.VFIO_GROUP_SET_CONTAINER(self.vfio_group, ctypes.c_int(VFIOHAL.vfio_fd))

#     va = self.alloc_pinned_memory(sz=4096)
#     print(hex(va))

#     print(read_pagemap(va))

#     # print(self.pcifmt)
#     # xxx = memoryview(bytearray(self.pcifmt.encode()))
#     # xxxx = (ctypes.c_char * len(self.pcifmt))(*bytearray(self.pcifmt.encode()))
#     # print(xxxx)
#     # self.vfio_dev = vfio.VFIO_GROUP_GET_DEVICE_FD(self.vfio_group, xxxx)
#     # print(self.vfio_dev)
#     # device = ioctl(group, VFIO_GROUP_GET_DEVICE_FD, "0000:06:0d.0");

#     # ioctl(container, VFIO_IOMMU_MAP_DMA, )

#     # dma_map.vaddr = mmap(0, 1024 * 1024, PROT_READ | PROT_WRITE,
#     #                  MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
#     # dma_map.size = 1024 * 1024;
#     # dma_map.iova = 0; /* 1MB starting at 0x0 from device view */
#     # dma_map.flags = VFIO_DMA_MAP_FLAG_READ | VFIO_DMA_MAP_FLAG_WRITE;

#     # ioctl(container, VFIO_IOMMU_MAP_DMA, &dma_map);

#     # vfio.VFIO_DEVICE_GET_INFO(self.vfio_dev, vfio_info:=vfio.struct_vfio_device_info())
#     # print(vfio_info)

    


