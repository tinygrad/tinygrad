import mmap, ctypes, struct
from tinygrad.runtime.autogen import libpciaccess

def pci_scan_bus(vendor_id, device_id):
  libpciaccess.pci_system_init()
  pci_iter = libpciaccess.pci_id_match_iterator_create(None)

  devs = []
  while pcidev:=libpciaccess.pci_device_next(pci_iter):
    if pcidev.contents.vendor_id == vendor_id and pcidev.contents.device_id == device_id: devs.append(pcidev.contents)

  return devs

def pci_set_master(pcidev):
  # TODO: parse from linux/include/uapi/linux/pci_regs.h
  libpciaccess.pci_device_cfg_read_u16(pcidev, ctypes.byref(val:=ctypes.c_uint16()), 0x4)
  libpciaccess.pci_device_cfg_write_u16(pcidev, val.value | 0x4, 0x4)

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
