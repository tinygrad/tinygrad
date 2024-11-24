from __future__ import annotations
import ctypes
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_gc_11_0_0, amdgpu_mmhub_3_0_0
from tinygrad.runtime.autogen.am import am
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
    self.view[entry_id] = (paddr & 0x0000FFFFFFFFF000) | flags
  def get_entry(self, entry_id): return self.view[entry_id]

class VirtualMapping(PhysicalMemory):
  def __init__(self, adev, ptable_vaddr, paddr, size):
    self.vaddr, self.ptable_vaddr = ptable_vaddr + adev.gmc.vm_base, ptable_vaddr
    super().__init__(adev, paddr, size)
  def offset(self, byts): return VirtualMapping(self.adev, self.ptable_vaddr + byts, self.paddr + byts, self.size - byts)

class MM:
  def __init__(self, adev, vram_size:int):
    self.adev, self.vram_size = adev, vram_size
    self.phys_allocator = PhysicalAllocator(adev, vram_size)
    self.next_vaddr = 0
    self.root_pt = PTE(self.adev, self.palloc(0x1000, zero=True), lv=2) # 3rd level (2,1,0), 1gb pages

  def page_table_walker(self, page_table, vaddr, size, offset=0) -> Generator[Tuple[int, int, int, int], None, None]:
    pte_covers = 1 << ((9 * page_table.lv) + 12)

    def _move_cursor(sz):
      nonlocal vaddr, offset, size
      vaddr += sz
      offset += sz
      size -= sz

    def _level_down(va, sz):
      entry = page_table.get_entry(entry_idx:=(va // pte_covers) % 512)
      if entry & am.AMDGPU_PTE_VALID:
        assert (entry & amdgpu_2.AMDGPU_PDE_PTE) == 0, "Must be table"
        child_page_table = PTE(self.adev, PhysicalMemory(self.adev, entry & 0x0000FFFFFFFFF000, 0x1000), lv=page_table.lv - 1)
      else:
        child_page_table = PTE(self.adev, self.palloc(0x1000, zero=True), lv=page_table.lv - 1)
        page_table.set_table(entry_idx, child_page_table)
      yield from self.page_table_walker(child_page_table, va, sz, offset=offset)
      # TODO: Free the child_page_table after flags changed there.

    # First pte is not full covered
    if vaddr % pte_covers != 0:
      yield from _level_down(vaddr, min(pte_covers - (vaddr % pte_covers), size))
      _move_cursor(min(pte_covers - (vaddr % pte_covers), size))

    n_ptes = size // pte_covers
    if n_ptes > 0: yield (vaddr, offset, (vaddr // pte_covers) % 512, n_ptes, pte_covers, page_table)
    _move_cursor(n_ptes * pte_covers)

    # Last pte is not full covered
    if size > 0: yield from _level_down(vaddr, size)

  def map_range_2(self, vaddr, paddr, size, uncached=False) -> VirtualMapping:
    for va, off, pte_st_idx, n_ptes, pte_covers, page_table in self.page_table_walker(self.root_pt, vaddr, size):
      # To optimize TLB entries count, need to map pages as contigous entries. Determine size of each chunks.
      while n_ptes > 0:
        frags_cnt = min((va.bit_length() - 1 if va != 0 else 31), (n_ptes * pte_covers).bit_length() - 1) - 12
        assert frags_cnt >= 0

        update_ptes = (1 << (frags_cnt + 12)) // pte_covers
        for pte_idx in range(update_ptes):
          assert page_table.get_entry(pte_st_idx + pte_idx) & amdgpu_2.AMDGPU_PTE_VALID == 0, "Entry already set"
          page_table.set_page(pte_st_idx + pte_idx, paddr + off, uncached=uncached, frag=frags_cnt)
          # print(f"Mapping page: {hex(vaddr + off)} -> {hex(paddr + off)} (0x{size:x}), nptes={update_ptes} incr=0x{pte_covers:x} {uncached=} {frags_cnt=}")
          off += pte_covers

        pte_st_idx += update_ptes
        n_ptes -= update_ptes

    self.adev.gmc.flush_tlb(ip="GC", vmid=0)
    self.adev.gmc.flush_tlb(ip="MM", vmid=0)
    return VirtualMapping(self.adev, vaddr, paddr, size)

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
        print(f"\tnptes=0x{max_alignment // i_pte_covers:x} incr=0x{pte_covers:x} upd_flags=0x0 frags=0x{frags:x}")

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
      self.adev.gmc.flush_tlb(ip="GC", vmid=0)
      self.adev.gmc.flush_tlb(ip="MM", vmid=0)
    return VirtualMapping(self.adev, vaddr, paddr, size)

  def unmap_range(self, virtual_mapping:VirtualMapping): pass # TODO

  def valloc(self, size:int, align=0x1000, uncached=False) -> VirtualMapping:
    # print("valloc", size, uncached)

    size = round_up(size, 0x1000)

    # align = max(align, size)
    for i in range(31):
      if (1 << i) <= size: align = (1 << i)
    # if size % (2 << 20) == 0: align = max(align, size)
    # elif size >= 256 << 10: align = 2 << 20
    print("valloc", size, align)

    addr = round_up(self.next_vaddr, align)

    self.next_vaddr = addr + size
    assert self.next_vaddr <= self.adev.gmc.vm_end
    return self.map_range_2(addr, self.palloc(size).paddr, size, uncached=uncached)
  def vfree(self, vm:VirtualMapping): pass

  def palloc(self, size, align=0x1000, zero=False) -> PhysicalMemory:
    pm = self.phys_allocator.alloc(size, align)
    if zero: ctypes.memset(pm.cpu_addr(), 0, pm.size)
    return pm
  def pfree(self, pm:PhysicalMemory): pass
