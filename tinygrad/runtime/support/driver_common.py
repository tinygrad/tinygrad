# from __future__ import annotations
# import ctypes, collections, time, dataclasses, functools, fcntl, os, hashlib, re, gzip, struct
# from tinygrad.helpers import mv_address, getenv, round_up, DEBUG, temp, fetch, getbits, to_mv
# from tinygrad.runtime.autogen.nv import nv
# from tinygrad.runtime.support.hcq import MMIOInterface
# from tinygrad.runtime.support.memory import TLSFAllocator
# from tinygrad.runtime.support.system import System

# @dataclasses.dataclass(frozen=True)
# class VirtMapping: va_addr:int; size:int; paddrs:list[tuple[int, int]]; uncached:bool=False; system:bool=False; snooped:bool=False # noqa: E702

# class PageTableTraverseContext:
#   def __init__(self, dev, pt, vaddr, create_pts=False, free_pts=False, boot=False):
#     self.dev, self.vaddr, self.create_pts, self.free_pts, self.boot = dev, vaddr - dev.mm.va_base, create_pts, free_pts, boot
#     self.pt_stack:list[tuple[Any, int, int]] = [(pt, self._pt_pte_idx(pt, self.vaddr), self._pt_pte_size(pt))]

#   def _pt_pte_cnt(self, lv): return self.dev.mm.pte_cnt[lv]
#   def _pt_pte_size(self, pt): return self.dev.mm.pte_covers[pt.lv]
#   def _pt_pte_idx(self, pt, va): return (va // self._pt_pte_size(pt)) % self._pt_pte_cnt(pt.lv)

#   def level_down(self):
#     pt, pte_idx, _ = self.pt_stack[-1]

#     if not pt.valid(pte_idx):
#       assert self.create_pts, "Not allowed to create new page table"
#       pt.set_entry(pte_idx, self.dev.mm.palloc(0x1000, zero=True, boot=self.boot), table=True, valid=True)

#     assert not pt.is_pte(pte_idx), f"Must be table pt={pt.paddr:#x}, {pt.lv=} {pte_idx=} {pt.read_fields(pte_idx)}"
#     child_page_table = self.dev.mm.pt_t(self.dev, pt.address(pte_idx), lv=pt.lv+1)

#     self.pt_stack.append((child_page_table, self._pt_pte_idx(child_page_table, self.vaddr), self._pt_pte_size(child_page_table)))
#     return self.pt_stack[-1]

#   def _try_free_pt(self) -> bool:
#     pt, _, _ = self.pt_stack[-1]
#     if self.free_pts and pt != self.dev.mm.root_page_table and all(not pt.valid(i) for i in range(self._pt_pte_cnt(self.pt_stack[-1][0].lv))):
#       self.dev.mm.pfree(pt.paddr)
#       parent_pt, parent_pte_idx, _ = self.pt_stack[-2]
#       parent_pt.set_entry(parent_pte_idx, 0x0, valid=False)
#       return True
#     return False

#   def level_up(self):
#     while self._try_free_pt() or self.pt_stack[-1][1] == self._pt_pte_cnt(self.pt_stack[-1][0].lv):
#       pt, pt_cnt, _ = self.pt_stack.pop()
#       if pt_cnt == self._pt_pte_cnt(pt.lv): self.pt_stack[-1] = (self.pt_stack[-1][0], self.pt_stack[-1][1] + 1, self.pt_stack[-1][2])

#   def next(self, size:int, off=0):
#     while size > 0:
#       pt, pte_idx, pte_covers = self.pt_stack[-1]
#       if self.create_pts:
#         while pt.lv < self.dev.mm.first_page_lv or pte_covers > size or self.vaddr & (pte_covers-1) != 0: pt, pte_idx, pte_covers = self.level_down()
#       else:
#         while not pt.is_pte(pte_idx): pt, pte_idx, pte_covers = self.level_down()

#       assert pte_covers == 0x1000
#       entries = min(size // pte_covers, self._pt_pte_cnt(pt.lv) - pte_idx)
#       assert entries > 0, f"Invalid entries {size=:#x}, {pte_covers=:#x}"
#       yield off, pt, pte_idx, entries, pte_covers

#       size, off, self.vaddr = size - entries * pte_covers, off + entries * pte_covers, self.vaddr + entries * pte_covers
#       self.pt_stack[-1] = (pt, pte_idx + entries, pte_covers)
#       self.level_up()

# class MemoryManager:
#   va_allocator: ClassVar[TLSFAllocator|None] = None

#   def __init__(self, dev, vram_size:int, boot_size:int, pt_t, pte_cnt:list[int], pte_covers:list[int], first_lv:int, first_page_lv:int, va_base:int):
#     self.dev, self.vram_size, self.va_base = dev, vram_size, va_base
#     self.pt_t, self.pte_cnt, self.pte_covers, self.first_page_lv = pt_t, pte_cnt, pte_covers, first_page_lv

#     self.boot_allocator = TLSFAllocator(boot_size, base=0) # per device
#     self.pa_allocator = TLSFAllocator(vram_size - (boot_size), base=self.boot_allocator.size) # per device
#     self.root_page_table = pt_t(self.dev, self.palloc(0x1000, zero=not self.dev.smi_dev, boot=True), lv=first_lv)

#   def _frag_size(self, va, sz, must_cover=True):
#     """
#     Calculate the tlb fragment size for a given virtual address and size.
#     If must_cover is True, the fragment size must cover the size, otherwise the biggest fragment size that fits the size is returned.
#     Fragment 0 is 4KB, 1 is 8KB and so on.
#     """
#     va_pwr2_div, sz_pwr2_div, sz_pwr2_max = va & -(va) if va > 0 else (1 << 63), sz & -(sz), (1 << (sz.bit_length() - 1))
#     return (min(va_pwr2_div, sz_pwr2_div) if must_cover else min(va_pwr2_div, sz_pwr2_max)).bit_length() - 1 - 12

#   def page_tables(self, vaddr:int):
#     ctx = PageTableTraverseContext(self.dev, self.root_page_table, vaddr)
#     for _ in ctx.next(1 << 30): return [pt for pt, _, _ in ctx.pt_stack]

#   def map_range(self, vaddr:int, size:int, paddrs:list[tuple[int, int]], uncached=False, system=False, snooped=False, boot=False) -> VirtMapping:
#     if getenv("MM_DEBUG", 0): print(f"mm {self.dev.devfmt}: mapping {vaddr=:#x} ({size=:#x})")

#     assert size == sum(p[1] for p in paddrs), f"Size mismatch {size=} {sum(p[1] for p in paddrs)=}"

#     ctx = PageTableTraverseContext(self.dev, self.root_page_table, vaddr, create_pts=True, boot=boot)
#     for paddr, psize in paddrs:
#       for off, pt, pte_idx, pte_cnt, pte_covers in ctx.next(psize):
#         for pte_off in range(pte_cnt):
#           assert not pt.valid(pte_idx + pte_off), f"PTE already mapped: {pt.entry(pte_idx + pte_off):#x}"
#           pt.set_entry(pte_idx + pte_off, paddr + off + pte_off * pte_covers, uncached=uncached, system=system, snooped=snooped,
#                        frag=self._frag_size(ctx.vaddr+off, pte_cnt * pte_covers), valid=True)

#     self.on_range_mapped()
#     return VirtMapping(vaddr, size, paddrs, uncached=uncached, system=system, snooped=snooped)

#   def unmap_range(self, vaddr:int, size:int):
#     if getenv("MM_DEBUG", 0): print(f"mm {self.dev.devfmt}: unmapping {vaddr=:#x} ({size=:#x})")

#     ctx = PageTableTraverseContext(self.dev, self.root_page_table, vaddr, free_pts=True)
#     for off, pt, pte_idx, pte_cnt, pte_covers in ctx.next(size):
#       for pte_id in range(pte_idx, pte_idx + pte_cnt):
#         assert pt.valid(pte_id), f"PTE not mapped: {pt.entry(pte_id):#x}"
#         pt.set_entry(pte_id, paddr=0x0, valid=False)

#   def on_range_mapped(self): pass

#   @classmethod
#   def alloc_vaddr(cls, size:int, align=0x1000) -> int:
#     assert cls.va_allocator is not None, "must be set it"
#     return cls.va_allocator.alloc(size, max((1 << (size.bit_length() - 1)), align))

#   def valloc(self, size:int, align=0x1000, uncached=False, contiguous=False) -> VirtMapping:
#     # Alloc physical memory and map it to the virtual address
#     va = self.alloc_vaddr(size:=round_up(size, 0x1000), align)

#     if contiguous: paddrs = [(self.palloc(size, zero=True), size)]
#     else:
#       # Traverse the PT to find the largest contiguous sizes we need to allocate. Try to allocate the longest segment to reduce TLB pressure.
#       paddrs = []
#       ctx = PageTableTraverseContext(self.dev, self.root_page_table, va, create_pts=True)
#       for off, _, _, seg_cnt, seg_size in ctx.next(size):
#         rem_len = seg_cnt * seg_size
#         while rem_len > 0:
#           # Try to allocate as long segment (power of 2) as possible
#           cont_seg_sz, paddr = 1 << (self._frag_size(ctx.vaddr+off, rem_len) + 12), None
#           while cont_seg_sz >= 0x1000:
#             try: paddr = self.palloc(cont_seg_sz, zero=False)
#             except MemoryError: cont_seg_sz //= 2
#             else: break

#           if paddr is not None: paddrs += [(paddr, cont_seg_sz)]
#           else:
#             for paddr, _ in paddrs: self.pa_allocator.free(paddr)
#             raise MemoryError(f"Failed to allocate a contiguous page. (allocation size={size:#x})")
#           rem_len, off = rem_len - cont_seg_sz, off + cont_seg_sz

#     return self.map_range(va, size, paddrs, uncached=uncached)

#   def vfree(self, vm:VirtMapping):
#     assert self.va_allocator is not None, "must be set it"
#     self.unmap_range(vm.va_addr, vm.size)
#     self.va_allocator.free(vm.va_addr)
#     for paddr, _ in vm.paddrs: self.pa_allocator.free(paddr)

#   def palloc(self, size:int, align:int=0x1000, zero=True, boot=False) -> int:
#     # assert self.dev.is_booting == boot, "During booting, only boot memory can be allocated"
#     paddr = (self.boot_allocator if boot else self.pa_allocator).alloc(round_up(size, 0x1000), align)
#     if zero: self.dev.vram[paddr:paddr+size] = bytes(size)
#     return paddr

#   def pfree(self, paddr:int): self.pa_allocator.free(paddr)
