from __future__ import annotations
import ctypes, collections, time, dataclasses, functools, fcntl, os, hashlib, re
from tinygrad.helpers import mv_address, getenv, round_up, DEBUG, temp, fetch, getbits
from tinygrad.runtime.autogen.nv import nv
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.allocator import TLSFAllocator
from tinygrad.runtime.support.nv.ip import NV_FLCN, NV_GSP
from tinygrad.runtime.support.nvd import alloc_sysmem
from hexdump import hexdump

NV_DEBUG = getenv("NV_DEBUG", 0)

class NVReg:
  def __init__(self, nvdev, base, addr, fields=None): self.nvdev, self.addr, self.fields = nvdev, base + addr, fields

  def add_field(self, name:str, start:int, end:int): self.fields[name] = (start, end)

  def read(self): return self.nvdev.rreg(self.addr)
  def read_bitfields(self) -> dict[str, int]: return self.decode(self.read())

  def write(self, _ini_val:int=0, **kwargs): self.nvdev.wreg(self.addr, _ini_val | self.encode(**kwargs))

  def update(self, **kwargs): self.write(self.read() & ~self.mask(*kwargs.keys()), **kwargs)

  def mask(self, *names):
    return functools.reduce(int.__or__, ((((1 << (self.fields[nm][1]-self.fields[nm][0] + 1)) - 1) << self.fields[nm][0]) for nm in names), 0)

  def encode(self, **kwargs) -> int: return functools.reduce(int.__or__, (value << self.fields[name][0] for name,value in kwargs.items()), 0)
  def decode(self, val: int) -> dict: return {name:getbits(val, start, end) for name,(start,end) in self.fields.items()}

# TODO: prob can optimize this
class NVRegSet:
  def __init__(self, nvdev, base, fn, fields=None): self.nvdev, self.base, self.fn, self.fields = nvdev, base, fn, fields or {}
  def add_field(self, name:str, start:int, end:int): self.fields[name] = (start, end)
  def __getitem__(self, idx:int): return NVReg(self.nvdev, self.base, self.fn(idx), fields=self.fields)

class NVRegBased:
  def __init__(self, nvdev, cls, offset, fields=None): self.nvdev, self.cls, self.offset, self.fields = nvdev, cls, offset, fields or {}
  def add_field(self, name:str, start:int, end:int): self.fields[name] = (start, end)
  def with_base(self, base:int): return self.cls(self.nvdev, base, self.offset, self.fields)

@dataclasses.dataclass(frozen=True)
class NVMapping: va_addr:int; size:int; paddrs:list[tuple[int, int]]; uncached:bool=False; system:bool=False; snooped:bool=False # noqa: E702

class NVPageTableEntry:
  def __init__(self, nvdev, paddr, lv): self.nvdev, self.paddr, self.lv, self.entries = nvdev, paddr, lv, nvdev.vram.view(paddr, 0x1000, fmt='Q')

  def set_entry(self, entry_id:int, paddr:int, table=False, uncached=False, system=False, snooped=False, frag=0, valid=True):
    uncached = True

    if not table:
      # x = self.nvdev.NV_MMU_VER2_PDE.encode(is_pte=True, address_sys=paddr >> 12, aperture=1, vol=uncached)

      aper = 2 if system else 0
      x = self.nvdev.NV_MMU_VER2_PTE.encode(valid=True, address_sys=paddr >> 12, aperture=aper, vol=uncached, kind=0)
      
      if self.lv !=3: self.entries[entry_id] = x
      else: self.entries[2*entry_id] = x

    elif self.lv == 3:
      x = self.nvdev.NV_MMU_VER2_DUAL_PDE.encode(is_pte=False, address_small_sys=paddr >> 12, aperture_small=1 if valid else 0, vol_small=uncached)
      self.entries[2*entry_id] = x & 0xffffffffffffffff
      self.entries[2*entry_id+1] = x >> 32
      assert entry_id < 256
    else:
      x = self.nvdev.NV_MMU_VER2_PDE.encode(is_pte=False, address_sys=paddr >> 12, aperture=1 if valid else 0, vol=uncached)
      self.entries[entry_id] = x
    
    print(entry_id, hex(paddr), table, uncached, system, snooped, frag, valid, hex(x))

  def entry(self, entry_id:int) -> int:
    return (self.entries[2*entry_id+1]<<32) | self.entries[2*entry_id] if self.lv == 3 else self.entries[entry_id]

  def read_fields(self, entry_id:int) -> dict:
    if self.lv == 3: return self.nvdev.NV_MMU_VER2_DUAL_PDE.decode(self.entry(entry_id))
    elif self.lv == 4: return self.nvdev.NV_MMU_VER2_PTE.decode(self.entry(entry_id))
    else: return self.nvdev.NV_MMU_VER2_PDE.decode(self.entry(entry_id))

  def is_pte(self, entry_id) -> bool: return self.read_fields(entry_id)['is_pte'] if self.lv <= 3 else True
  def valid(self, entry_id):
    if self.is_pte(entry_id): return self.read_fields(entry_id)['valid']
    elif self.lv == 3: return self.read_fields(entry_id)['aperture_small'] != 0
    return self.read_fields(entry_id)['aperture'] != 0

  def address(self, entry_id:int) -> int:
    if self.lv == 3: return self.read_fields(entry_id)['address_small_sys'] << 12
    return self.read_fields(entry_id)['address_sys'] << 12

class NVPageTableTraverseContext:
  def __init__(self, nvdev, pt, vaddr, create_pts=False, free_pts=False, boot=False):
    self.nvdev, self.vaddr, self.create_pts, self.free_pts, self.boot = nvdev, vaddr - nvdev.mm.va_allocator.base, create_pts, free_pts, boot
    self.pt_stack:list[tuple[NVPageTableEntry, int, int]] = [(pt, self._pt_pte_idx(pt, vaddr), self._pt_pte_size(pt))]

  def _pt_pte_cnt(self, lv): return [4, 512, 512, 256, 512][lv]
  def _pt_pte_size(self, pt): return [0x800000000000, 0x4000000000, 0x20000000, 0x200000, 0x1000][pt.lv]
  def _pt_pte_idx(self, pt, va): return (va // self._pt_pte_size(pt)) % self._pt_pte_cnt(pt.lv)

  def level_down(self):
    pt, pte_idx, _ = self.pt_stack[-1]

    if not pt.valid(pte_idx):
      assert self.create_pts, "Not allowed to create new page table"
      pt.set_entry(pte_idx, self.nvdev.mm.palloc(0x1000, zero=True, boot=self.boot), table=True, valid=True)

    assert not pt.is_pte(pte_idx), f"Must be table pt={pt.paddr:#x}, {pt.lv=} {pte_idx=} {pt.read_fields(pte_idx)}"
    print('level_down', hex(pt.address(pte_idx)))
    child_page_table = NVPageTableEntry(self.nvdev, pt.address(pte_idx), lv=pt.lv+1)

    self.pt_stack.append((child_page_table, self._pt_pte_idx(child_page_table, self.vaddr), self._pt_pte_size(child_page_table)))
    return self.pt_stack[-1]

  def level_up(self):
    while self.pt_stack[-1][1] == self._pt_pte_cnt(len(self.pt_stack) - 1):
      print("level_up")
      _, pt_cnt, _ = self.pt_stack.pop()
      if pt_cnt == self._pt_pte_cnt(len(self.pt_stack)):
        self.pt_stack[-1] = (self.pt_stack[-1][0], self.pt_stack[-1][1] + 1, self.pt_stack[-1][2])

  def next(self, size:int, off=0):
    while size > 0:
      pt, pte_idx, pte_covers = self.pt_stack[-1]
      if self.create_pts:
        while pte_covers > size: pt, pte_idx, pte_covers = self.level_down()
        # while pte_covers != 0x1000: pt, pte_idx, pte_covers = self.level_down() # test deep tables
      # else:
      #   while pt.lv!=am.AMDGPU_VM_PTB and not self.nvdev.gmc.is_pte_huge_page(pt.entries[pte_idx]): pt, pte_idx, pte_covers = self.level_down()

      entries = min(size // pte_covers, self._pt_pte_cnt(len(self.pt_stack) - 1) - pte_idx)
      assert entries > 0, "Invalid entries"
      yield off, pt, pte_idx, entries, pte_covers

      size, off, self.vaddr = size - entries * pte_covers, off + entries * pte_covers, self.vaddr + entries * pte_covers
      self.pt_stack[-1] = (pt, pte_idx + entries, pte_covers)
      self.level_up()

class NVMemoryManager:
  va_allocator = TLSFAllocator((1 << 49), base=0x0) # global for all devices.

  def __init__(self, nvdev:NVDev, vram_size:int):
    self.nvdev, self.vram_size = nvdev, vram_size
    # self.boot_allocator = TLSFAllocator(64 << 20, base=128<<20) # per device
    self.pa_allocator = TLSFAllocator(vram_size - (64 << 20), base=0x0) # per device
    self.root_page_table = NVPageTableEntry(self.nvdev, self.palloc(0x1000, zero=not self.nvdev.smi_dev, boot=True), lv=0)

  def map_range(self, vaddr:int, size:int, paddrs:list[tuple[int, int]], uncached=False, system=False, snooped=False, boot=False) -> NVMapping:
    if NV_DEBUG >= 2: print(f"nv {self.nvdev.devfmt}: mapping {vaddr=:#x} ({size=:#x})")

    assert size == sum(p[1] for p in paddrs), f"Size mismatch {size=} {sum(p[1] for p in paddrs)=}"

    ctx = NVPageTableTraverseContext(self.nvdev, self.root_page_table, vaddr, create_pts=True, boot=boot)
    for paddr, psize in paddrs:
      for off, pt, pte_idx, pte_cnt, pte_covers in ctx.next(psize):
        for pte_off in range(pte_cnt):
          assert not pt.valid(pte_idx + pte_off), f"PTE already mapped: {pt.entries[pte_idx + pte_off]:#x}"
          pt.set_entry(pte_idx + pte_off, paddr + off + pte_off * pte_covers, uncached=uncached, system=system, snooped=snooped,
                       frag=0x0, valid=True)

    # Invalidate TLB after mappings.
    return NVMapping(vaddr, size, paddrs, uncached=uncached, system=system, snooped=snooped)

  def unmap_range(self, vaddr:int, size:int):
    if NV_DEBUG >= 2: print(f"nv {self.nvdev.devfmt}: unmapping {vaddr=:#x} ({size=:#x})")

    ctx = NVPageTableTraverseContext(self.nvdev, self.root_page_table, vaddr, free_pts=True)
    for off, pt, pte_idx, pte_cnt, pte_covers in ctx.next(size):
      for pte_id in range(pte_idx, pte_idx + pte_cnt):
        assert pt.entries[pte_id] & am.AMDGPU_PTE_VALID == am.AMDGPU_PTE_VALID, f"PTE not mapped: {pt.entries[pte_id]:#x}"
        pt.set_entry(pte_id, paddr=0x0, valid=False)

  @staticmethod
  def alloc_vaddr(size:int, align=0x1000) -> int: return NVMemoryManager.va_allocator.alloc(size, max((1 << (size.bit_length() - 1)), align))

  def valloc(self, size:int, align=0x1000, uncached=False, contigous=False) -> NVMapping:
    # Alloc physical memory and map it to the virtual address
    va = self.alloc_vaddr(size:=round_up(size, 0x1000), align)

    paddrs = [(self.palloc(size, zero=True), size)]
    return self.map_range(va, size, paddrs, uncached=uncached)

  def vfree(self, vm:AMMapping):
    self.unmap_range(vm.va_addr, vm.size)
    self.va_allocator.free(vm.va_addr)
    for paddr, _ in vm.paddrs: self.pa_allocator.free(paddr)

  def palloc(self, size:int, align:int=0x1000, zero=True, boot=False) -> int:
    # assert self.nvdev.is_booting == boot, "During booting, only boot memory can be allocated"
    # paddr = (self.boot_allocator if boot else self.pa_allocator).alloc(round_up(size, 0x1000), align)
    paddr = self.pa_allocator.alloc(round_up(size, 0x1000), align)
    if zero: self.nvdev.vram[paddr:paddr+size] = bytes(size)
    return paddr

  def pfree(self, paddr:int): self.pa_allocator.free(paddr)

class NVDev:
  def __init__(self, devfmt, mmio:MMIOInterface, vram:MMIOInterface, rom:bytes):
    self.devfmt, self.mmio, self.vram, self.rom = devfmt, mmio, vram, rom
    self.included_files, self.reg_names, self.reg_offsets = set(), set(), {}

    self.smi_dev = False
    self.is_booting = False
    self._early_init()

    self.mm = NVMemoryManager(self, self.vram_size)
    self.flcn = NV_FLCN(self)
    self.gsp = NV_GSP(self)

    for ip in [self.flcn, self.gsp]: ip.init_sw()
    for ip in [self.flcn, self.gsp]: ip.init_hw()

  def wreg(self, addr, value):
    self.mmio[addr // 4] = value
    if NV_DEBUG >= 4: print(f"wreg: {hex(addr)} = {hex(value)}")
  def rreg(self, addr):
    return self.mmio[addr // 4]
    if NV_DEBUG >= 5: print(f"wreg: {hex(addr)} = {hex(value)}")

  def _early_init(self):
    for name in ['NV_MMU_VER2_PTE', 'NV_MMU_VER2_PDE', 'NV_MMU_VER2_DUAL_PDE']: self.__dict__[name] = NVReg(self, 0, 0, fields={})
    self.include("kernel-open/nvidia-uvm/hwref/turing/tu102/dev_mmu.h")
    self.include("src/common/inc/swref/published/turing/tu102/dev_vm.h")
    self.include("src/common/inc/swref/published/ampere/ga102/dev_gc6_island.h")
    self.include("src/common/inc/swref/published/ampere/ga102/dev_gc6_island_addendum.h")

    self.vram_size = self.NV_PGC6_AON_SECURE_SCRATCH_GROUP_42.read() << 20

  def _alloc_boot_struct(self, typ):
    va, paddrs = alloc_sysmem(ctypes.sizeof(typ), contigous=True)
    return typ.from_address(va), paddrs[0]

  def _download(self, file) -> str:
    url = f"https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/ed4be649623435ebb04f5e93f859bf46d977daa4/{file}"
    return fetch(url, subdir="defines").read_text()

  def include(self, file) -> str:
    if file in self.included_files: return
    self.included_files.add(file)

    txt = self._download(file)

    PARAM = re.compile(r'#define\s+(\w+)\s*\(\s*(\w+)\s*\)\s*(.+)')
    CONST = re.compile(r'#define\s+(\w+)\s+([0-9A-Fa-fx]+)')
    BITFLD = re.compile(r'#define\s+(\w+)\s+([0-9\+\-\*\(\)]+):([0-9\+\-\*\(\)]+)')

    regs_off = {'NV_PFALCON_FALCON': (None, 0x0), 'NV_PGSP_FALCON': 0x0, 'NV_PSEC_FALCON': 0x0, 'NV_PRISCV_RISCV': (None, 0x1000), 'NV_PGC6_AON': 0x0,
      'NV_PGC6_BSI': 0x0, 'NV_PFALCON_FBIF': (None, 0x600), 'NV_PFALCON2_FALCON': (None, 0x1000), 'NV_PBUS': 0x0, 'NV_PFB': 0x0,
      'NV_VIRTUAL_FUNCTION':0x00B80000}

    for raw in txt.splitlines():
      if raw.startswith("#define "):
        if (m := BITFLD.match(raw)):
          name, hi, lo = m.groups()
          for r in self.reg_names:
            if name.startswith(r+"_"):
              self.__dict__[r].add_field(name[len(r)+1:].lower(), eval(lo), eval(hi))
              break
          if name.startswith("NV_MMU_VER2_"):
            for r in ['NV_MMU_VER2_PTE', 'NV_MMU_VER2_PDE', 'NV_MMU_VER2_DUAL_PDE']:
              if name.startswith(r+"_"): self.__dict__[r].add_field(name[len(r)+1:].lower(), eval(lo), eval(hi))
          else: self.reg_offsets[name] = (eval(lo), eval(hi))
        elif (m := PARAM.match(raw)):
          name, param, expr = m.groups()
          expr = expr.strip().rstrip('\\').split('/*')[0].rstrip()
          reg_pref = next((prefix for prefix in regs_off.keys() if name.startswith(prefix)), None)

          if reg_pref is not None:
            fields = {}
            for k, v in self.reg_offsets.items():
              if k.startswith(name+'_'): fields[k[len(name)+1:]] = v

            if regs_off[reg_pref].__class__ is tuple:
              self.__dict__[name] = NVRegBased(self, NVRegSet, eval(f"lambda {param}: {expr} + {regs_off[reg_pref][1]}"), fields=fields)
            else:
              self.__dict__[name] = NVRegSet(self, regs_off[reg_pref], eval(f"lambda {param}: {expr}"), fields=fields)
            self.reg_names.add(name)
          else:
            assert self.__dict__.get(name) is None, f"Duplicate definition for {name} in {file}"
            self.__dict__[name] = eval(f"lambda {param}: {expr}")
        elif (m := CONST.match(raw)):
          name, value = m.groups()
          reg_pref = next((prefix for prefix in regs_off.keys() if name.startswith(prefix)), None)
          not_already_reg = not any(name.startswith(r+"_") for r in self.reg_names)

          if reg_pref is not None and not_already_reg:
            fields = {}
            for k, v in self.reg_offsets.items():
              if k.startswith(name+'_'): fields[k[len(name)+1:]] = v

            if regs_off[reg_pref].__class__ is tuple:
              self.__dict__[name] = NVRegBased(self, NVReg, int(value, 0) + regs_off[reg_pref][1], fields=fields)
            else:
              self.__dict__[name] = NVReg(self, regs_off[reg_pref], int(value, 0), fields=fields)
            self.reg_names.add(name)
          else:
            assert self.__dict__.get(name) in [None, int(value, 0)], f"Duplicate definition for {name} in {file}"
            self.__dict__[name] = int(value, 0)
