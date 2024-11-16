from __future__ import annotations
import ctypes
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0, amdgpu_mmhub_3_0_0
from tinygrad.helpers import to_mv, mv_address, round_up, getenv

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
  def __init__(self, adev, pm, lv): self.adev, self.pm, self.view, self.lv = adev, pm, pm.cpu_view().cast('Q'), lv
  def set_table(self, entry_id, pte:PTE): self.view[entry_id] = (pte.pm.paddr & 0x0000FFFFFFFFF000) | amdgpu_2.AMDGPU_PTE_VALID
  def set_page(self, entry_id, paddr, uncached=False, frag=0):
    flags = amdgpu_2.AMDGPU_PTE_VALID | amdgpu_2.AMDGPU_PTE_WRITEABLE | amdgpu_2.AMDGPU_PTE_READABLE | amdgpu_2.AMDGPU_PTE_EXECUTABLE | amdgpu_2.AMDGPU_PTE_FRAG(frag)
    if self.lv > 0: flags |= amdgpu_2.AMDGPU_PDE_PTE
    if uncached: flags |= amdgpu_2.AMDGPU_PTE_MTYPE_NV10(0, 3) # 3 = MTYPE_UC
    else: flags |= amdgpu_2.AMDGPU_PTE_MTYPE_NV10(0, 0)
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
    if getenv("TRACE_MM"): print(f"map_range: pde:0x{pde.pm.paddr:X} {hex(vaddr)} -> {hex(paddr)} ({size}), level={pde.lv}")
    # pde.lv # 2, 1, 0
    assert size != 0 and paddr & 0xFFF == 0 and vaddr & 0xFFF == 0 and size % 0x1000 == 0, "must be page aligned"
    pte_covers = 1 << ((9 * pde.lv) + 12)
    lvaddr = vaddr % (1 << ((9 * (pde.lv + 1)) + 12))
    entry_idx = (lvaddr // pte_covers)
    assert (512 - entry_idx) * pte_covers >= size, "must fit"

    cur_vaddr, cur_paddr, cur_size = vaddr, paddr, size
    while cur_size > 0:
      i_pte_covers = min(cur_size, pte_covers - cur_vaddr % pte_covers)
      assert i_pte_covers > 0

      if cur_vaddr % pte_covers == 0 and cur_size >= pte_covers:
        max_alignment, frags = i_pte_covers, 0
        for i in range(31):
          if (cur_vaddr % (1 << i)) == 0 and (1 << i) <= cur_size: max_alignment, frags = (1 << i), i - 12

        assert frags >= 0 and max_alignment % i_pte_covers == 0, "Must be aligned"

        # full cover, set as huge page
        for j in range(max_alignment // i_pte_covers):
          # print(pde.get_entry(entry_idx + j))
          assert pde.get_entry(entry_idx + j) & amdgpu_2.AMDGPU_PTE_VALID == 0, f"Entry already set pde:0x{pde.pm.paddr:X} {entry_idx + j} {hex(cur_vaddr+j*pte_covers)}"
          pde.set_page(entry_idx + j, cur_paddr + j * pte_covers, uncached=uncached, frag=frags)
          if getenv("TRACE_MM"):
            add = j * pte_covers
            print(f"\tMapping page: pde:0x{pde.pm.paddr:X} {entry_idx + j}: {hex(cur_vaddr+add)} -> {hex(cur_paddr+add)}, cons={max_alignment} ptes={max_alignment // i_pte_covers} {uncached=} {frags=}")

        entry_idx += (max_alignment // i_pte_covers) - 1 # TODO: looks bad
        i_pte_covers = max_alignment
      else:
        # set up table and recurse
        entry = pde.get_entry(entry_idx)
        if (entry & amdgpu_2.AMDGPU_PTE_VALID) == 0:
          pte = PTE(self.adev, self.palloc(0x1000, zero=True), lv=pde.lv - 1)
          # for j in range(512): assert pte.get_entry(j) == 0, "Must be zero"
          pde.set_table(entry_idx, pte)
          entry = pde.get_entry(entry_idx)
          # print("go alloc")

        assert (entry & amdgpu_2.AMDGPU_PDE_PTE) == 0, f"Must be table pde:{pde.pm.paddr:X} {entry_idx + j} {hex(cur_vaddr+j*pte_covers)}"
        # print("go pass", hex(entry & 0x0000FFFFFFFFF000))
        pte_addr = PhysicalMemory(self.adev, entry & 0x0000FFFFFFFFF000, 0x1000)

        self.map_range(PTE(self.adev, pte_addr, lv=pde.lv - 1), cur_vaddr, cur_paddr, i_pte_covers, uncached=uncached)

      cur_vaddr, cur_paddr, cur_size = cur_vaddr + i_pte_covers, cur_paddr + i_pte_covers, cur_size - i_pte_covers
      entry_idx += 1

    if pde == self.root_pt:
      if self.adev.gmc.gfx_enabled: self.adev.gmc.flush_tlb_gfxhub(0, 0, 0)
      if self.adev.gmc.mmhub_enabled: self.adev.gmc.flush_tlb_mmhub(0, 0, 0)
    return VirtualMapping(self.adev, vaddr, paddr, size)

  def unmap_range(self, virtual_mapping:VirtualMapping): pass # TODO

  def valloc(self, size:int, align=0x1000, uncached=False) -> VirtualMapping:
    # print("valloc", size, uncached)

    size = round_up(size, 0x1000)
    addr = round_up(self.next_vaddr, max(align, size))
    self.next_vaddr = addr + size
    assert self.next_vaddr <= self.adev.gmc.vm_end
    return self.map_range(self.root_pt, addr, self.palloc(size, align).paddr, size, uncached=uncached)
  def vfree(self, vm:VirtualMapping): pass

  def palloc(self, size, align=0x1000, zero=False) -> PhysicalMemory:
    pm = self.phys_allocator.alloc(size, align)
    if zero: ctypes.memset(pm.cpu_addr(), 0, pm.size)
    return pm
  def pfree(self, pm:PhysicalMemory): pass
