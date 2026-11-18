from __future__ import annotations

MT_PAGE_SIZE = 0x1000
MT_PTE_READ_ONLY = 0x3c00000000000001
MT_PTE_READ_WRITE = 0x3c0000000000000d

class MTMMU4:
  """Rogue MMU v4 page tables used by the MTT S80 (BVNC 35.4.1632.23)."""
  def __init__(self, root_pa:int, size:int=0x400000):
    self.root_pa, self.root, self.next = root_pa, bytearray(size), 0x1000
    self.pd:dict[int,int] = {}
    self.pt:dict[tuple[int,int],int] = {}
    self.pages:dict[int,int] = {}

  def _alloc(self) -> int:
    if self.next + MT_PAGE_SIZE > len(self.root): raise MemoryError("MT MMU root allocation exhausted")
    pa = self.next
    self.next += MT_PAGE_SIZE
    return self.root_pa + pa

  def map(self, va:int, pa:int, flags:int=MT_PTE_READ_ONLY) -> None:
    if va % MT_PAGE_SIZE or pa % MT_PAGE_SIZE: raise ValueError("MT MMU mappings must be 4 KiB aligned")
    if flags & 1 != 1: raise ValueError("MT MMU mapping must be present")
    self.pages[va] = pa
    pci, pdi, pti = (va >> 30) & 0x3ff, (va >> 21) & 0x1ff, (va >> 12) & 0x1ff
    if pci not in self.pd:
      self.pd[pci] = self._alloc()
      self.root[pci*4:pci*4+4] = (((self.pd[pci] >> 12) << 4) | 1).to_bytes(4, 'little')
    if (pci, pdi) not in self.pt:
      self.pt[pci,pdi] = self._alloc()
      off = self.pd[pci] - self.root_pa + pdi * 8
      self.root[off:off+8] = (self.pt[pci,pdi] | 1).to_bytes(8, 'little')
    off = self.pt[pci,pdi] - self.root_pa + pti * 8
    self.root[off:off+8] = ((pa & 0xfffffffff000) | flags).to_bytes(8, 'little')

  def map_range(self, va:int, pa:int, size:int, flags:int=MT_PTE_READ_ONLY) -> None:
    if size % MT_PAGE_SIZE: raise ValueError("MT MMU range size must be 4 KiB aligned")
    for off in range(0, size, MT_PAGE_SIZE): self.map(va + off, pa + off, flags)
