import mmap
from tinygrad.runtime.autogen import libpciaccess

def scan_pci_devs(vendor_id, device_id):
  libpciaccess.pci_system_init()
  pci_iter = libpciaccess.pci_id_match_iterator_create(None)

  dev = []
  while pcidev:=libpciaccess.pci_device_next(pci_iter):
    if pcidev.contents.vendor_id == vendor_id and pcidev.contents.device_id == device_id: dev.append(pcidev.contents)

  return dev

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
