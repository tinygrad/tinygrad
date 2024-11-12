from __future__ import annotations
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
    self.next_paddr = 0

  def alloc(self, size:int, align=0x1000) -> PhysicalMemory:
    addr = round_up(self.next_paddr, align)
    self.next_paddr = addr + size
    assert self.next_paddr <= self.vram_size
    return PhysicalMemory(self.adev, addr, size)

  def free(self, mem:PhysicalMemory): pass

class PTE:
  def __init__(self, adev, pmem, lv): self.adev, self.pmem, self.view, self.lv, self.ptes = adev, pmem, pmem.cpu_view().cast('Q'), lv, {}
  def set_table(self, entry_id, pte:PTE):
    self.view[entry_id] = (pte.pmem.paddr & 0x0000FFFFFFFFF000) | amdgpu_2.AMDGPU_PTE_VALID | amdgpu_2.AMDGPU_PTE_FRAG(9)
  def set_page(self, entry_id, paddr, uncached=False):
    flags = amdgpu_2.AMDGPU_PTE_VALID | amdgpu_2.AMDGPU_PTE_WRITEABLE | amdgpu_2.AMDGPU_PTE_READABLE | amdgpu_2.AMDGPU_PTE_EXECUTABLE
    flags |= amdgpu_2.AMDGPU_PDE_PTE
    if uncached: flags |= amdgpu_2.AMDGPU_PTE_SNOOPED | amdgpu_2.AMDGPU_PTE_MTYPE_NV10(0, 3) # 3 = MTYPE_UC
    self.view[entry_id] = (paddr & 0x0000FFFFFFFFF000) | flags
  def get_entry(self, entry_id): return self.view[entry_id]

class VirtualMapping(PhysicalMemory):
  def __init__(self, adev, vaddr, paddr, size):
    self.vaddr = vaddr
    super().__init__(adev, paddr, size)

class MM:
  def __init__(self, adev, vram_size):
    self.adev = adev
    self.phys_allocator = PhysicalAllocator(adev, vram_size)
    self.next_vaddr = 0
    self.root_pt = PTE(self.adev, self.phys_allocator.alloc(0x1000), lv=2) # 3rd level (2,1,0), 1gb pages

  def map_range(self, pde, vaddr, paddr, size, uncached=False) -> VirtualMapping:
    # pde.lv # 2, 1, 0
    assert paddr & 0xFFF == 0 and vaddr & 0xFFF == 0 and size % 0x1000 == 0, "must be page aligned"
    pte_covers = 1 << ((9 * pde.lv) + 12)
    entry_start_index = (vaddr >> ((9 * pde.lv) + 12)) % 512
    entry_end_index = ((vaddr + size - 1) >> ((9 * pde.lv) + 12)) % 512

    cur_vaddr, cur_paddr, cur_size = vaddr, paddr, size
    for i in range(entry_start_index, entry_end_index + 1):
      if cur_size >= pte_covers:
        # full cover, set as huge page
        assert pde.get_entry(i) & amdgpu_2.AMDGPU_PTE_VALID == 0, "entry already set"
        pde.set_page(i, cur_paddr, uncached=uncached)
      else:
        # set up table and recurse
        entry = pde.get_entry(i)
        if entry & amdgpu_2.AMDGPU_PTE_VALID == 0:
          pte = PTE(self.adev, self.phys_allocator.alloc(0x1000), pde.lv - 1)
          pde.set_table(i, pte)
          entry = pde.get_entry(i)

        assert entry & amdgpu_2.AMDGPU_PDE_PTE == 0, "must be table"
        pte_addr = PhysicalMemory(self.adev, entry & 0x0000FFFFFFFFF000, 0x1000)
        self.map_range(PTE(self.adev, pte_addr, pde.lv - 1), cur_vaddr, cur_paddr, min(cur_size, pte_covers), uncached=uncached)

      cur_vaddr, cur_paddr, cur_size = cur_vaddr + pte_covers, cur_paddr + pte_covers, cur_size - pte_covers

    if pde == self.root_pt:
      self.adev.gmc.flush_tlb(0, 0, 0)
      self.adev.gmc.mmhub_flush_tlb(0, 0, 0)
    return VirtualMapping(self.adev, vaddr, paddr, size)
  def unmap_range(self, virtual_mapping:VirtualMapping): pass # TODO

  def valloc(self, size:int, align=0x1000) -> VirtualMapping:
    addr = round_up(self.next_vaddr, align)
    self.next_vaddr = addr + size
    assert self.next_vaddr <= self.adev.gmc.vm_end
    return self.adev.map_range(self.root_pt, addr, self.palloc(size, align).paddr, size)
  def vfree(self, vm:VirtualMapping): pass

  def palloc(self, size, align=0x1000) -> PhysicalMemory: return self.phys_allocator.alloc(size, align)
  def pfree(self, pm:PhysicalMemory): pass
