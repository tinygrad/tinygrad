from __future__ import annotations
import ctypes
from tinygrad.runtime.autogen.am import am
from tinygrad.helpers import to_mv, round_up, getenv

class PhysicalMemory:
  def __init__(self, adev, paddr, size): self.adev, self.paddr, self.size = adev, paddr, size
  def mc_addr(self): return self.adev.gmc.mc_base + self.paddr
  def cpu_addr(self): return self.adev.vram_cpu_addr + self.paddr
  def cpu_view(self): return to_mv(self.adev.vram_cpu_addr + self.paddr, self.size)

class VirtualMapping(PhysicalMemory):
  def __init__(self, adev, ptable_vaddr, paddr, size):
    self.vaddr, self.ptable_vaddr = ptable_vaddr + adev.gmc.vm_base, ptable_vaddr
    super().__init__(adev, paddr, size)

# TODO: Complete + tests
class PhysicalAllocator:
  def __init__(self, adev, vram_size):
    self.adev = adev
    self.vram_size = vram_size
    self.nxt = 0

    parts = vram_size // (cnt:=1)
    self.next_paddr = [parts * i for i in range(cnt)]

  def alloc(self, size:int, align=0x1000) -> PhysicalMemory:
    addr = round_up(self.next_paddr[self.nxt], align)
    self.next_paddr[self.nxt] = addr + size
    assert self.next_paddr[self.nxt] <= self.vram_size
    self.nxt += 1
    self.nxt %= len(self.next_paddr)
    return PhysicalMemory(self.adev, addr, size)

  def free(self, mem:PhysicalMemory): pass

class AMPageTableEntry:
  def __init__(self, pm, lv): self.pm, self.view, self.lv = pm, pm.cpu_view().cast('Q'), lv

  def set_table(self, entry_id, pte:PageTableEntry, valid=True):
    self.view[entry_id] = (pte.pm.paddr & 0x0000FFFFFFFFF000) | (am.AMDGPU_PTE_VALID if valid else 0)

  def set_page(self, entry_id, paddr, uncached=False, frag=0, valid=True):
    f = (am.AMDGPU_PTE_VALID if valid else 0) | am.AMDGPU_PTE_WRITEABLE | am.AMDGPU_PTE_READABLE | am.AMDGPU_PTE_EXECUTABLE \
      | am.AMDGPU_PTE_FRAG(frag) | (am.AMDGPU_PDE_PTE if self.lv != am.AMDGPU_VM_PTB else 0) | (am.AMDGPU_PTE_MTYPE_NV10(0, am.MTYPE_UC) if uncached else 0)
    self.view[entry_id] = (paddr & 0x0000FFFFFFFFF000) | f

  def get_entry(self, entry_id): return self.view[entry_id]

class MM:
  def __init__(self, adev, vram_size:int):
    self.adev, self.vram_size = adev, vram_size
    self.phys_allocator = PhysicalAllocator(adev, vram_size)
    self.next_vaddr = 0
    self.root_page_table = AMPageTableEntry(self.palloc(0x1000, zero=True), lv=am.AMDGPU_VM_PDB1)

  def page_table_walker(self, page_table, vaddr, size, offset=0, free_pt=False) -> Generator[Tuple[int, int, int, int], None, None]:
    pte_covers = 1 << ((9 * (3-page_table.lv)) + 12)

    def _move_cursor(sz):
      nonlocal vaddr, offset, size
      vaddr += sz
      offset += sz
      size -= sz

    def _level_down(va, sz):
      entry = page_table.get_entry(pte_idx:=(va // pte_covers) % 512)
      if entry & am.AMDGPU_PTE_VALID:
        assert entry & am.AMDGPU_PDE_PTE == 0, "Must be table"
        child_page_table = AMPageTableEntry(PhysicalMemory(self.adev, entry & 0x0000FFFFFFFFF000, 0x1000), lv=page_table.lv+1)
      else:
        child_page_table = AMPageTableEntry(self.palloc(0x1000, zero=True), lv=page_table.lv+1)
        page_table.set_table(pte_idx, child_page_table)
      yield from self.page_table_walker(child_page_table, va, sz, offset=offset)

      if free_pt and all(child_page_table.get_entry(i) & am.AMDGPU_PTE_VALID == 0 for i in range(512)):
        self.pfree(child_page_table.pm)
        page_table.set_page(pte_idx, valid=False)

    # First pte is not full covered
    if vaddr % pte_covers != 0:
      yield from _level_down(vaddr, min(pte_covers - (vaddr % pte_covers), size))
      _move_cursor(min(pte_covers - (vaddr % pte_covers), size))

    n_ptes = size // pte_covers
    if n_ptes > 0: yield (vaddr, offset, (vaddr // pte_covers) % 512, n_ptes, pte_covers, page_table)
    _move_cursor(n_ptes * pte_covers)

    # Last pte is not full covered
    if size > 0: yield from _level_down(vaddr, size)

  def map_range(self, vaddr, paddr, size, uncached=False) -> VirtualMapping:
    for va, off, pte_st_idx, n_ptes, pte_covers, page_table in self.page_table_walker(self.root_page_table, vaddr, size):
      # To optimize TLB entries count, need to map pages as contigous entries. Determine size of each chunks.
      while n_ptes > 0:
        frags_cnt = min((va.bit_length() - 1 if va != 0 else 31), (n_ptes * pte_covers).bit_length() - 1) - 12
        assert frags_cnt >= 0

        update_ptes = (1 << (frags_cnt + 12)) // pte_covers
        for pte_idx in range(update_ptes):
          assert page_table.get_entry(pte_st_idx + pte_idx) & am.AMDGPU_PTE_VALID == 0, "Entry already set"
          page_table.set_page(pte_st_idx + pte_idx, paddr=paddr + off, uncached=uncached, frag=frags_cnt, valid=True)
          # print(f"Mapping page: {hex(vaddr + off)} -> {hex(paddr + off)} (0x{size:x}), nptes={update_ptes} incr=0x{pte_covers:x} {uncached=} {frags_cnt=}")
          off += pte_covers

        pte_st_idx += update_ptes
        n_ptes -= update_ptes

    self.adev.gmc.flush_tlb(ip="GC", vmid=0)
    self.adev.gmc.flush_tlb(ip="MM", vmid=0)
    return VirtualMapping(self.adev, vaddr, paddr, size)

  def unmap_range(self, vm:VirtualMapping):
    for va, off, pte_st_idx, n_ptes, pte_covers, page_table in self.page_table_walker(self.root_page_table, vm.va, vm.size, free_pt=True):
      for pte_idx in range(update_ptes):
        assert page_table.get_entry(pte_st_idx + pte_idx) & am.AMDGPU_PTE_VALID == am.AMDGPU_PTE_VALID, "Entry must be set"
        page_table.set_page(pte_st_idx + pte_idx, valid=False)

  def valloc(self, size:int, align=0x1000, uncached=False) -> VirtualMapping:
    size = round_up(size, 0x1000)

    for i in range(31):
      if (1 << i) <= size: align = (1 << i)

    addr = round_up(self.next_vaddr, align)
    self.next_vaddr = addr + size
    assert self.next_vaddr <= self.adev.gmc.vm_end
    return self.map_range(addr, self.palloc(size).paddr, size, uncached=uncached)

  def vfree(self, vm:VirtualMapping):
    self.unmap_range(vm)
    self.pfree(vm.pm)

  def palloc(self, size, align=0x1000, zero=False) -> PhysicalMemory:
    pm = self.phys_allocator.alloc(size, align)
    if zero: ctypes.memset(pm.cpu_addr(), 0, pm.size)
    return pm

  def pfree(self, pm:PhysicalMemory): self.phys_allocator.free(pm)
