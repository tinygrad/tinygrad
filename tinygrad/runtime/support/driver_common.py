from __future__ import annotations
import ctypes, collections, time, dataclasses, functools, fcntl, os, hashlib, re, gzip, struct
from tinygrad.helpers import mv_address, getenv, round_up, DEBUG, temp, fetch, getbits, to_mv
from tinygrad.runtime.autogen.nv import nv
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.allocator import TLSFAllocator
from tinygrad.runtime.support.system import System

@dataclasses.dataclass(frozen=True)
class VirtMapping: va_addr:int; size:int; paddrs:list[tuple[int, int]]; uncached:bool=False; system:bool=False; snooped:bool=False # noqa: E702

class PageTableTraverseContext:
  def __init__(self, dev, pt, vaddr, create_pts=False, free_pts=False, boot=False):
    self.dev, self.vaddr, self.create_pts, self.free_pts, self.boot = dev, vaddr, create_pts, free_pts, boot
    self.pt_stack:list[tuple[PageTableEntry, int, int]] = [(pt, self._pt_pte_idx(pt, vaddr), self._pt_pte_size(pt))]

  def _pt_pte_cnt(self, lv): return self.dev.mm.pte_cnt[lv]
  def _pt_pte_size(self, pt): return self.dev.mm.pte_covers[pt.lv]
  def _pt_pte_idx(self, pt, va): return (va // self._pt_pte_size(pt)) % self._pt_pte_cnt(pt.lv)

  def level_down(self):
    pt, pte_idx, _ = self.pt_stack[-1]

    if not pt.valid(pte_idx):
      assert self.create_pts, "Not allowed to create new page table"
      pt.set_entry(pte_idx, self.dev.mm.palloc(0x1000, zero=True, boot=True), table=True, valid=True)

    assert not pt.is_pte(pte_idx), f"Must be table pt={pt.paddr:#x}, {pt.lv=} {pte_idx=} {pt.read_fields(pte_idx)}"
    child_page_table = self.dev.mm.pt_t(self.dev, pt.address(pte_idx), lv=pt.lv+1)

    self.pt_stack.append((child_page_table, self._pt_pte_idx(child_page_table, self.vaddr), self._pt_pte_size(child_page_table)))
    return self.pt_stack[-1]

  def level_up(self):
    while self.pt_stack[-1][1] == self._pt_pte_cnt(len(self.pt_stack) - 1):
      _, pt_cnt, _ = self.pt_stack.pop()
      if pt_cnt == self._pt_pte_cnt(len(self.pt_stack)):
        self.pt_stack[-1] = (self.pt_stack[-1][0], self.pt_stack[-1][1] + 1, self.pt_stack[-1][2])

  def next(self, size:int, off=0):
    while size > 0:
      pt, pte_idx, pte_covers = self.pt_stack[-1]
      if self.create_pts:
        while pt.lv < self.dev.mm.first_page_lv or pte_covers > size or self.vaddr & (pte_covers-1) != 0: pt, pte_idx, pte_covers = self.level_down()
      else:
        while not pt.is_pte(pte_idx): pt, pte_idx, pte_covers = self.level_down()

      entries = min(size // pte_covers, self._pt_pte_cnt(len(self.pt_stack) - 1) - pte_idx)
      assert entries > 0, "Invalid entries"
      yield off, pt, pte_idx, entries, pte_covers

      size, off, self.vaddr = size - entries * pte_covers, off + entries * pte_covers, self.vaddr + entries * pte_covers
      self.pt_stack[-1] = (pt, pte_idx + entries, pte_covers)
      self.level_up()

class MemoryManager:
  def __init__(self, dev, vram_size:int, boot_partition_size:int, pt_t, pte_cnt:list[int], pte_covers:list[int], first_page_lv:int):
    self.dev, self.vram_size, self.pt_t, self.pte_cnt, self.pte_covers, self.first_page_lv = dev, vram_size, pt_t, pte_cnt, pte_covers, first_page_lv
    self.boot_allocator = TLSFAllocator(boot_partition_size, base=0) # per device
    self.pa_allocator = TLSFAllocator(vram_size - (boot_partition_size), base=self.boot_allocator.base) # per device
    self.root_page_table = self.pt_t(self.dev, self.palloc(0x1000, zero=not self.dev.smi_dev, boot=True), lv=0)

  def page_tables(self, vaddr:int):
    ctx = PageTableTraverseContext(self.dev, self.root_page_table, vaddr)
    for _ in ctx.next(1 << 30): return [pt for pt, _, _ in ctx.pt_stack]

  def map_range(self, vaddr:int, size:int, paddrs:list[tuple[int, int]], uncached=False, system=False, snooped=False, boot=False, nomap=False) -> VirtMapping:
    assert size == sum(p[1] for p in paddrs), f"Size mismatch {size=} {sum(p[1] for p in paddrs)=}"

    ctx = PageTableTraverseContext(self.dev, self.root_page_table, vaddr, create_pts=True, boot=boot)
    for paddr, psize in paddrs:
      for off, pt, pte_idx, pte_cnt, pte_covers in ctx.next(psize):
        for pte_off in range(pte_cnt):
          assert not pt.valid(pte_idx + pte_off), f"PTE already mapped: {pt.entries[pte_idx + pte_off]:#x} {pt.valid(pte_idx + pte_off)}"
          pt.set_entry(pte_idx + pte_off, paddr + off + pte_off * pte_covers, uncached=uncached, system=system, snooped=snooped,
                       frag=0x0, valid=True)

    self.dev.on_range_mapped()
    return VirtMapping(vaddr, size, paddrs, uncached=uncached, system=system, snooped=snooped)

  def unmap_range(self, vaddr:int, size:int):
    ctx = PageTableTraverseContext(self.dev, self.root_page_table, vaddr, free_pts=True)
    for off, pt, pte_idx, pte_cnt, pte_covers in ctx.next(size):
      for pte_id in range(pte_idx, pte_idx + pte_cnt):
        assert pt.is_valid(pte_id), f"PTE not mapped: {pt.entries[pte_id]:#x}"
        pt.set_entry(pte_id, paddr=0x0, valid=False)

  @classmethod
  def alloc_vaddr(cls, size:int, align=0x1000) -> int: return cls.va_allocator.alloc(size, max((1 << (size.bit_length() - 1)), align))

  def valloc(self, size:int, align=0x1000, uncached=False, contiguous=False, nomap=False) -> NVMapping:
    # Alloc physical memory and map it to the virtual address
    va = self.alloc_vaddr(size:=round_up(size, 0x1000), align)

    paddrs = [(self.palloc(size, zero=True), size)]
    return self.map_range(va, size, paddrs, uncached=uncached, nomap=nomap)

  def vfree(self, vm:AMMapping):
    self.unmap_range(vm.va_addr, vm.size)
    self.va_allocator.free(vm.va_addr)
    for paddr, _ in vm.paddrs: self.pa_allocator.free(paddr)

  def palloc(self, size:int, align:int=0x1000, zero=True, boot=False) -> int:
    # assert self.dev.is_booting == boot, "During booting, only boot memory can be allocated"
    paddr = (self.boot_allocator if boot else self.pa_allocator).alloc(round_up(size, 0x1000), align)
    # paddr = self.pa_allocator.alloc(round_up(size, 0x1000), align)
    if zero: self.dev.vram[paddr:paddr+size] = bytes(size)
    return paddr

  def pfree(self, paddr:int): self.pa_allocator.free(paddr)
