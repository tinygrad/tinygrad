from __future__ import annotations
import ctypes
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0, amdgpu_mmhub_3_0_0
from tinygrad.helpers import to_mv, mv_address, round_up

class PhysicalMemory:
  def __init__(self, adev, paddr, size): self.adev, self.paddr, self.size = adev, paddr, size
  def mc_addr(self): return self.adev.gmc.mc_base + self.paddr
  def cpu_addr(self): return self.adev.vram_cpu_addr + self.paddr
  def cpu_view(self): return to_mv(self.adev.vram_cpu_addr + self.paddr, self.size)

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

class PTE:
  def __init__(self, adev, pmem, lv): self.adev, self.pmem, self.view, self.lv, self.ptes = adev, pmem, pmem.cpu_view().cast('Q'), lv, {}
  def set_table(self, entry_id, pte:PTE): self.view[entry_id] = (pte.pmem.paddr & 0x0000FFFFFFFFF000) | amdgpu_2.AMDGPU_PTE_VALID
  def set_page(self, entry_id, paddr, uncached=False, frag=0):
    flags = amdgpu_2.AMDGPU_PTE_VALID | amdgpu_2.AMDGPU_PTE_WRITEABLE | amdgpu_2.AMDGPU_PTE_READABLE | amdgpu_2.AMDGPU_PTE_EXECUTABLE | amdgpu_2.AMDGPU_PTE_FRAG(frag)
    if self.lv > 0: flags |= amdgpu_2.AMDGPU_PDE_PTE
    if uncached: flags |= amdgpu_2.AMDGPU_PTE_MTYPE_NV10(0, 3) # 3 = MTYPE_UC
    else: flags |= amdgpu_2.AMDGPU_PTE_MTYPE_NV10(0, 0)
    # print("flags", hex(flags))
    self.view[entry_id] = (paddr & 0x0000FFFFFFFFF000) | flags
  def get_entry(self, entry_id): return self.view[entry_id]

class VirtualMapping(PhysicalMemory):
  def __init__(self, adev, ptable_vaddr, paddr, size):
    self.vaddr, ptable_vaddr = ptable_vaddr + adev.gmc.vm_base, ptable_vaddr
    super().__init__(adev, paddr, size)

class MM:
  def __init__(self, adev, vram_size):
    self.adev = adev
    self.phys_allocator = PhysicalAllocator(adev, vram_size)
    self.next_vaddr = 0
    self.root_pt = PTE(self.adev, self.palloc(0x1000, zero=True), lv=2) # 3rd level (2,1,0), 1gb pages

  def map_range(self, pde, vaddr, paddr, size, uncached=False) -> VirtualMapping:
    # print(f"map_range: {hex(vaddr)} -> {hex(paddr)} ({size}), level={pde.lv}")
    # pde.lv # 2, 1, 0
    assert size != 0 and paddr & 0xFFF == 0 and vaddr & 0xFFF == 0 and size % 0x1000 == 0, "must be page aligned"
    pte_covers = 1 << ((9 * pde.lv) + 12)
    lvaddr = vaddr % (1 << ((9 * (pde.lv + 1)) + 12))
    # entry_start_index = (lvaddr // pte_covers)

    entry_idx, cur_vaddr, cur_paddr, cur_size = lvaddr // pte_covers, vaddr, paddr, size
    while cur_size > 0:
      i_pte_covers = min(cur_size, pte_covers - cur_vaddr % pte_covers)
      assert i_pte_covers > 0

      if cur_vaddr % pte_covers == 0 and cur_size >= pte_covers:
        # max_alignment, ma_off = pte_covers, 0
        # for i in range(31):
        #   if (cur_vaddr % (1 << i)) == 0 and (1 << i) <= cur_size: max_alignment, ma_off = cur_vaddr, i
        # i_pte_covers = max_alignment

        # full cover, set as huge page
        assert pde.get_entry(entry_idx) & amdgpu_2.AMDGPU_PTE_VALID == 0, "entry already set"
        pde.set_page(entry_idx, cur_paddr, uncached=uncached, frag=pde.lv*9)
        print(hex(pde.lv*9))
      else:
        # set up table and recurse
        entry = pde.get_entry(entry_idx)
        if (entry & amdgpu_2.AMDGPU_PTE_VALID) == 0:
          pte = PTE(self.adev, self.palloc(0x1000, zero=True), lv=pde.lv - 1)
          pde.set_table(entry_idx, pte)
          entry = pde.get_entry(entry_idx)

        assert (entry & amdgpu_2.AMDGPU_PDE_PTE) == 0, "must be table"
        pte_addr = PhysicalMemory(self.adev, entry & 0x0000FFFFFFFFF000, 0x1000)

        self.map_range(PTE(self.adev, pte_addr, lv=pde.lv - 1), cur_vaddr, cur_paddr, i_pte_covers, uncached=uncached)

      cur_vaddr, cur_paddr, cur_size = cur_vaddr + i_pte_covers, cur_paddr + i_pte_covers, cur_size - i_pte_covers
      entry_idx += 1

    if pde == self.root_pt:
      self.adev.gmc.flush_tlb_gfxhub(0, 0, 0)
      self.adev.gmc.flush_tlb_mmhub(0, 0, 0)
    return VirtualMapping(self.adev, vaddr, paddr, size)

  def unmap_range(self, virtual_mapping:VirtualMapping): pass # TODO

  def valloc(self, size:int, align=0x1000, uncached=False) -> VirtualMapping:
    # print("valloc", size, uncached)

    size = round_up(size, 0x1000)
    addr = round_up(self.next_vaddr, max(align, 2 << 20))
    self.next_vaddr = addr + size
    assert self.next_vaddr <= self.adev.gmc.vm_end
    return self.map_range(self.root_pt, addr, self.palloc(size, align).paddr, size, uncached=uncached)
  def vfree(self, vm:VirtualMapping): pass

  def palloc(self, size, align=0x1000, zero=False) -> PhysicalMemory:
    pm = self.phys_allocator.alloc(size, align)
    if zero: ctypes.memset(pm.cpu_addr(), 0, pm.size)
    return pm
  def pfree(self, pm:PhysicalMemory): pass
