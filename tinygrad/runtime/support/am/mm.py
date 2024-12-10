from __future__ import annotations
import ctypes, collections
from typing import List, Optional, Dict, Tuple, cast, Protocol, Type, Union, TypeVar, Generic, Any
# from tinygrad.runtime.autogen.am import am
from tinygrad.helpers import to_mv, round_up, getenv, mv_address

# AM_DEBUG = getenv("AM_DEBUG", 0)

# PhysicalMemoryBlockType = TypeVar('PhysicalMemoryBlockType', bound='PhysicalMemoryBlock')

class TLSFAllocator:
  def __init__(self, size:int, base:int=0, block_size:int=16, lv2_cnt:int=16):
    self.size, self.base, self.block_size, self.l2_cnt = size, base, block_size, lv2_cnt.bit_length()
    self.storage = [collections.defaultdict(list) for _ in range(size.bit_length() + 1)]
    self.blocks = {0: (size, None, None, True)}  # size, next, prev (off size), is_free
    self._insert_block(0, size)

  def lv1(self, size): return size.bit_length()
  def lv2(self, size): return (size - (1 << (size.bit_length() - 1)))  // (1 << (size.bit_length() - self.l2_cnt))

  def _insert_block(self, start:int, size:int, prev:Optional[int]=None):
    if prev is None: prev = self.blocks[start][2]
    # print("insert", start, size, start + size, prev)
    self.storage[self.lv1(size)][self.lv2(size)].append(start)
    self.blocks[start] = (size, start + size, prev, True)
    return self

  def _remove_block(self, start:int, size:int, prev:Optional[int]=None):
    if prev is None: prev = self.blocks[start][2]
    # print("remove", start, size, start + size, prev)
    self.storage[self.lv1(size)][self.lv2(size)].remove(start)
    self.blocks[start] = (size, start + size, prev, False)
    return self

  def _verify_blocks(self):
    for start, (size, nxt, prev, is_free) in self.blocks.items():
      assert self.blocks.get(nxt) is None or self.blocks[nxt][2] == start, f"next block must point to current {nxt=}, {start=}"
      assert self.blocks.get(prev) is None or self.blocks[prev][1] == start, f"prev block must point to current, {prev=}, {start=}"
    return self

  def _split_block(self, start:int, size:int, new_size:int):
    nxt = self.blocks[start][1]
    assert self.blocks[start][3], "block must be free"
    self._remove_block(start, size)._insert_block(start, new_size)._insert_block(start + new_size, size - new_size, prev=start)
    if self.blocks.get(nxt) is not None: self.blocks[nxt] = (self.blocks[nxt][0], self.blocks[nxt][1], start + new_size, self.blocks[nxt][3])
    return self._verify_blocks()

  def _merge_right(self, start:int):
    size, nxt, prev, is_free = self.blocks[start]
    assert is_free, "block must be free"

    while is_free and self.blocks.get(nxt) is not None:
      if (blk:=self.blocks[nxt])[3] is False: break
      self._remove_block(start, size)._remove_block(nxt, blk[0])._insert_block(start, size:=size + blk[0])
      assert self.blocks[start][1] == blk[1]
      _, nxt, _, _ = self.blocks.pop(nxt)

    if self.blocks.get(nxt) is not None: self.blocks[nxt] = (self.blocks[nxt][0], self.blocks[nxt][1], start, self.blocks[nxt][3])
    self._verify_blocks()

  def _merge_block(self, start:int):
    while (x:=self.blocks[start][2]) is not None and self.blocks[x][3] is True: start = x
    self._merge_right(start)

  def alloc(self, o_size:int, align:int=1) -> int:
    size = max(self.block_size, o_size + align - 1)
    size = round_up(size, (1 << size.bit_length() - self.l2_cnt))

    for l1 in range(self.lv1(size), len(self.storage)):
      for l2 in range(self.lv2(size) if l1 == size.bit_length() else 0, (1 << self.l2_cnt)):
        if len(self.storage[l1][l2]) > 0:
          assert (nsize:=self.blocks[self.storage[l1][l2][0]][0]) >= size, "block must be larger"

          start = self.storage[l1][l2][0]
          if (new_start:=round_up(start, align)) != start:
            self._split_block(start, nsize, new_start - start)
            start, nsize = new_start, self.blocks[new_start][0]

          if nsize > size: self._split_block(start, nsize, size)
          self._remove_block(start, size)._verify_blocks()
          return start + self.base
    raise MemoryError("OOM")

  def free(self, start:int):
    self._insert_block(start - self.base, self.blocks[start - self.base][0])._merge_block(start - self.base)

# class PhysicalMemoryBlock:
#   def __init__(self, paddr, size): self.paddr, self.size = paddr, size

# class GPUPhysicalMemoryBlock(PhysicalMemoryBlock):
#   def __init__(self, adev, paddr, size):
#     super().__init__(paddr, size)
#     self.adev = adev

#   def mc_addr(self): return self.adev.gmc.mc_base + self.paddr
#   def cpu_addr(self): return mv_address(self.adev.vram) + self.paddr
#   def cpu_view(self): return to_mv(self.cpu_addr(), self.size)

# class ScatterList(Generic[PhysicalMemoryBlockType]):
#   def __init__(self, blocks:List[PhysicalMemoryBlockType]): self.blocks = blocks
#   def __iter__(self): return iter(self.blocks)

# class VirtualMapping(GPUPhysicalMemoryBlock):
#   def __init__(self, adev, ptable_vaddr, paddr, size, uncached=False, system=False, snooped=False):
#     self.vaddr, self.ptable_vaddr = ptable_vaddr + adev.gmc.vm_base, ptable_vaddr
#     self.uncached, self.system, self.snooped = uncached, system, snooped
#     super().__init__(adev, paddr, size)

# # TODO: Complete + tests
# class PhysicalAllocator:
#   def __init__(self, adev, vram_size):
#     self.adev = adev
#     self.vram_size = vram_size
#     self.nxt = 0

#     parts = vram_size // (cnt:=1)
#     self.next_paddr = [parts * i for i in range(cnt)]

#   def alloc(self, size:int, align=0x1000) -> GPUPhysicalMemoryBlock:
#     addr = round_up(self.next_paddr[self.nxt], align)
#     self.next_paddr[self.nxt] = addr + size
#     assert self.next_paddr[self.nxt] <= self.vram_size
#     self.nxt += 1
#     self.nxt %= len(self.next_paddr)
#     return GPUPhysicalMemoryBlock(self.adev, addr, size)

#   def free(self, mem:GPUPhysicalMemoryBlock): pass

# class AMPageTableEntry:
#   def __init__(self, pm, lv): self.pm, self.view, self.lv = pm, pm.cpu_view().cast('Q'), lv

#   def set_table(self, entry_id, pte:PageTableEntry, valid=True):
#     self.view[entry_id] = (pte.pm.paddr & 0x0000FFFFFFFFF000) | (am.AMDGPU_PTE_VALID if valid else 0)

#   def set_page(self, entry_id, paddr, uncached=False, system=False, snooped=False, frag=0, valid=True):
#     f = (am.AMDGPU_PTE_VALID if valid else 0) | am.AMDGPU_PTE_WRITEABLE | am.AMDGPU_PTE_READABLE | am.AMDGPU_PTE_EXECUTABLE \
#       | am.AMDGPU_PTE_FRAG(frag) | (am.AMDGPU_PDE_PTE if self.lv != am.AMDGPU_VM_PTB else 0) \
#       | ((am.AMDGPU_PTE_SYSTEM) if system else 0) | ((am.AMDGPU_PTE_SNOOPED) if snooped else 0) \
#       | (am.AMDGPU_PTE_MTYPE_NV10(0, am.MTYPE_UC) if uncached else 0)
#     self.view[entry_id] = (paddr & 0x0000FFFFFFFFF000) | f

#   def get_entry(self, entry_id): return self.view[entry_id]

# class AMMemoryManager:
#   va_allocator = TLSFAllocator(512 * (1 << 30)) # global, all addresses mapped

#   def __init__(self, adev, vram_size:int):
#     self.adev, self.vram_size = adev, vram_size
#     self.pa_allocator = TLSFAllocator(adev, vram_size) # per device
#     self.root_page_table = AMPageTableEntry(self.palloc(0x1000, zero=True), lv=am.AMDGPU_VM_PDB1)

#   def page_table_walker(self, page_table, vaddr, size, offset=0, free_pt=False) -> Generator[Tuple[int, int, int, int], None, None]:
#     pte_covers = 1 << ((9 * (3-page_table.lv)) + 12)

#     def _move_cursor(sz):
#       nonlocal vaddr, offset, size
#       vaddr, offset, size = vaddr + sz, offset + sz, size - sz

#     def _level_down(va, sz):
#       entry = page_table.get_entry(pte_idx:=(va // pte_covers) % 512)
#       if entry & am.AMDGPU_PTE_VALID:
#         assert entry & am.AMDGPU_PDE_PTE == 0, "Must be table"
#         child_page_table = AMPageTableEntry(GPUPhysicalMemoryBlock(self.adev, entry & 0x0000FFFFFFFFF000, 0x1000), lv=page_table.lv+1)
#       else:
#         child_page_table = AMPageTableEntry(self.palloc(0x1000, zero=True), lv=page_table.lv+1)
#         page_table.set_table(pte_idx, child_page_table)
#       yield from self.page_table_walker(child_page_table, va, sz, offset=offset)

#       if free_pt and all(child_page_table.get_entry(i) & am.AMDGPU_PTE_VALID == 0 for i in range(512)):
#         self.pfree(child_page_table.pm)
#         page_table.set_page(pte_idx, valid=False)

#     # First pte is not full covered
#     if vaddr % pte_covers != 0:
#       yield from _level_down(vaddr, min(pte_covers - (vaddr % pte_covers), size))
#       _move_cursor(min(pte_covers - (vaddr % pte_covers), size))

#     n_ptes = size // pte_covers
#     if n_ptes > 0: yield (vaddr, offset, (vaddr // pte_covers) % 512, n_ptes, pte_covers, page_table)
#     _move_cursor(n_ptes * pte_covers)

#     # Last pte is not full covered
#     if size > 0: yield from _level_down(vaddr, size)

#   def map_range(self, vaddr, size, paddr=None, uncached=False, system=False, snooped=False):
#     if AM_DEBUG >= 3: print(f"Mapping {vaddr=:#x} -> {paddr=:#x} ({size=:#x})")
#     for va, off, pte_st_idx, n_ptes, pte_covers, page_table in self.page_table_walker(self.root_page_table, vaddr, size):
#       # To optimize TLB entries count, need to map pages as contigous entries. Determine size of each chunks.
#       while n_ptes > 0:
#         frags_cnt = min((va.bit_length() - 1 if va != 0 else 31), (n_ptes * pte_covers).bit_length() - 1) - 12
#         assert frags_cnt >= 0

#         update_ptes = (1 << (frags_cnt + 12)) // pte_covers
#         if paddr is None: paddr, off = self.phys_allocator.alloc(update_ptes * pte_covers).paddr, 0

#         for pte_idx in range(update_ptes):
#           assert page_table.get_entry(pte_st_idx + pte_idx) & am.AMDGPU_PTE_VALID == 0, "Entry already set"
#           page_table.set_page(pte_st_idx + pte_idx, paddr=paddr + off, uncached=uncached, system=system, snooped=snooped, frag=frags_cnt, valid=True)
#           off += pte_covers

#         if AM_DEBUG >= 3: print(f"\tnptes={update_ptes:#x} incr={pte_covers:#x} upd_flags={page_table.get_entry(pte_st_idx):#x} frags={frags_cnt:#x}")

#         pte_st_idx += update_ptes
#         n_ptes -= update_ptes

#     self.adev.gmc.flush_tlb(ip="GC", vmid=0)
#     self.adev.gmc.flush_tlb(ip="MM", vmid=0)
#     return VirtualMapping(self.adev, vaddr, paddr, size, uncached=uncached, system=system, snooped=snooped)

#   def unmap_range(self, vaddr:int, size:int):
#     for va, off, pte_st_idx, n_ptes, pte_covers, page_table in self.page_table_walker(self.root_page_table, vaddr, size, free_pt=True):
#       for pte_idx in range(n_ptes):
#         assert page_table.get_entry(pte_st_idx + pte_idx) & am.AMDGPU_PTE_VALID == am.AMDGPU_PTE_VALID, "Entry must be set"
#         page_table.set_page(pte_st_idx + pte_idx, valid=False)

#   def map_from(self, vaddr:int, size:int, from_adev):
#     for va, off, pte_st_idx, n_ptes, pte_covers, page_table in self.page_table_walker(from_adev.mm.root_page_table, vaddr, size):
#       while n_ptes > 0:
#         entry = from_adev.mm.root_page_table.get_entry(pte_st_idx)
#         frags_cnt = (entry & am.AMDGPU_PTE_FRAG_MASK) >> am.AMDGPU_PTE_FRAG_SHIFT
#         update_ptes = (1 << (frags_cnt + 12)) // pte_covers

#         assert entry & am.AMDGPU_PTE_VALID, "Entry must be valid"
#         uncached = entry & am.AMDGPU_PTE_MTYPE_MASK == am.AMDGPU_PTE_MTYPE_NV10(0, am.MTYPE_UC)
#         system = entry & am.AMDGPU_PTE_SYSTEM == am.AMDGPU_PTE_SYSTEM
#         snooped = entry & am.AMDGPU_PTE_SNOOPED == am.AMDGPU_PTE_SNOOPED
#         paddr = (entry & 0x0000FFFFFFFFF000) if system else (entry & 0x0000FFFFFFFFF000) + from_adev.pci_pmem_base
#         self.map_range(va + off, paddr, update_ptes * pte_covers, uncached=uncached, system=True, snooped=snooped)

#         pte_st_idx += update_ptes
#         n_ptes -= update_ptes
#         off += pte_covers * update_ptes

#   def alloc_vaddr(self, size:int, align=0x1000) -> int:
#     size = round_up(size, 0x1000)
#     align = (1 << size.bit_length())

#     # for i in range(31):
#     #   if (1 << i) <= size: align = (1 << i)

#     addr = round_up(MM.next_vaddr, align)
#     MM.next_vaddr = addr + size

#     assert MM.next_vaddr <= self.adev.gmc.vm_end
#     return addr

#   def valloc(self, size:int, align=0x1000, uncached=False) -> VirtualMapping:
#     size = round_up(size, 0x1000)
#     return self.map_range(self.alloc_vaddr(size, align), self.palloc(size, zero=True).paddr, size, uncached=uncached)

#   def palloc(self, size, align=0x1000, zero=False) -> GPUPhysicalMemoryBlock:
#     pm = self.phys_allocator.alloc(size, align)
#     if zero: ctypes.memset(pm.cpu_addr(), 0, pm.size)
#     return pm

#   def pfree(self, pm:GPUPhysicalMemoryBlock): self.phys_allocator.free(pm)

# # class VirtualMemoryRange:
# #   def __init__(self, adev, ptable_vaddr:int, size:int, phys_sg:Optional[ScatterList]=None, contigous=False,
# #                uncached=False, system=False, snooped=False):
# #     self.adev, self.vaddr, self.ptable_vaddr, self.size = adev, ptable_vaddr + adev.gmc.vm_base, ptable_vaddr, size

# #     if contigous: phys_sg = ScatterList(self.adev.mm.palloc(self.size))

# #     # Allocate on map
# #     if phys_sg is None: self.adev.mm.map_range(self.vaddr, self.size, uncached=uncached, system=system, snooped=snooped)
# #     else:
# #       assert self.size == sum(pm.size for pm in phys_sg), f"Size mismatch: {self.size} != {sum(pm.size for pm in phys_sg)}"

# #       vaddr = self.vaddr
# #       for pm in phys_sg:
# #         paddr = pm.paddr if isinstance(pm, GPUPhysicalMemoryBlock) and self.adev != pm.adev else paddr + pm.adev.pci_pmem_base
# #         self.adev.mm.map_range(vaddr, paddr, pm.size, uncached=uncached, system=system, snooped=snooped)
# #         vaddr += pm.size
# #       self.phys_sg = phys_sg

# #   def __del__(self): self.adev.mm.unmap_range(self.vaddr, self.size)
