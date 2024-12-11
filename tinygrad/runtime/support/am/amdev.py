from __future__ import annotations
import os, ctypes, collections, time
from typing import Tuple, Dict, Set, Optional, Generator, List
from tinygrad.helpers import to_mv, mv_address, getenv, round_up
from tinygrad.runtime.autogen.am import am, mp_11_0, mp_13_0_0, nbio_4_3_0, mmhub_3_0_0, gc_11_0_0, osssys_6_0_0
from tinygrad.runtime.support.am.mm import TLSFAllocator
from tinygrad.runtime.support.am.firmware import Firmware
from tinygrad.runtime.support.am.ip import AM_SOC21, AM_GMC, AM_IH, AM_PSP, AM_SMU, AM_GFX, AM_SDMA
from tinygrad.runtime.support.hcq import BumpAllocator

AM_DEBUG = getenv("AM_DEBUG", 0)

class AMRegister:
  def __init__(self, adev, reg_off:int, reg_fields:Dict[str, Tuple[int, int]]):
    self.adev, self.reg_off, self.reg_fields = adev, reg_off, reg_fields

  def _parse_kwargs(self, **kwargs):
    mask, values = 0xffffffff, 0
    for k, v in kwargs.items():
      if k not in self.reg_fields: raise ValueError(f"Unknown register field: {k}. {self.reg_fields.keys()}")
      m, s = self.reg_fields[k]
      if v & (m>>s) != v: raise ValueError(f"Value {v} for {k} is out of range {m=} {s=}")
      mask &= ~m
      values |= v << s
    return mask, values

  def build(self, **kwargs) -> int: return self._parse_kwargs(**kwargs)[1]

  def update(self, **kwargs): self.write(value=self.read(), **kwargs)

  def write(self, value=0, **kwargs):
    mask, values = self._parse_kwargs(**kwargs)
    self.adev.wreg(self.reg_off, (value & mask) | values)

  def read(self, **kwargs): return self.adev.rreg(self.reg_off) & self._parse_kwargs(**kwargs)[0]

class AMPhysicalMemoryBlock:
  def __init__(self, adev:AMDev, paddr:int, size:int): self.adev, self.paddr, self.size = adev, paddr, size
  def mc_addr(self): return self.adev.gmc.mc_base + self.paddr
  def cpu_addr(self): return mv_address(self.adev.vram) + self.paddr
  def cpu_view(self): return to_mv(self.cpu_addr(), self.size)

class AMVirtualMapping:
  def __init__(self, va_addr:int, size:int, cpu_addr:Optional[int], paddr:Optional[int]):
    self.va_addr, self.size, self.cpu_addr, self.paddr = va_addr, size, cpu_addr, paddr

class AMPageTableEntry:
  def __init__(self, pm, lv): self.pm, self.view, self.lv = pm, pm.cpu_view().cast('Q'), lv

  def set_table(self, entry_id, pte:AMPageTableEntry, valid=True):
    # if self.pm.paddr == 0x1700000 and entry_id == 0: input("hmm tbl")
    self.view[entry_id] = (pte.pm.paddr & 0x0000FFFFFFFFF000) | (am.AMDGPU_PTE_VALID if valid else 0)

  def set_page(self, entry_id, paddr, uncached=False, system=False, snooped=False, frag=0, valid=True):
    # if self.pm.paddr == 0x1700000 and entry_id == 0: input("hmm pg")
    f = (am.AMDGPU_PTE_VALID if valid else 0) | am.AMDGPU_PTE_WRITEABLE | am.AMDGPU_PTE_READABLE | am.AMDGPU_PTE_EXECUTABLE \
      | am.AMDGPU_PTE_FRAG(frag) | (am.AMDGPU_PDE_PTE if self.lv != am.AMDGPU_VM_PTB else 0) \
      | ((am.AMDGPU_PTE_SYSTEM) if system else 0) | ((am.AMDGPU_PTE_SNOOPED) if snooped else 0) \
      | (am.AMDGPU_PTE_MTYPE_NV10(0, am.MTYPE_UC) if uncached else 0)
    self.view[entry_id] = (paddr & 0x0000FFFFFFFFF000) | f

  def get_entry(self, entry_id): return self.view[entry_id]

class AMMemoryManager:
  va_allocator = TLSFAllocator(512 * (1 << 30), base=0x7F0000000000) # global for all devices.

  def __init__(self, adev, vram_size:int):
    self.adev, self.vram_size = adev, vram_size
    self.pa_allocator = TLSFAllocator(vram_size) # per device
    self.root_page_table = AMPageTableEntry(self.palloc(0x1000, zero=True), lv=am.AMDGPU_VM_PDB1)

  def page_table_walker(self, page_table, vaddr, size, offset=0, free_pt=False, creat_pt=True) -> Generator[Tuple[int, int, int, int], None, None]:
    pte_covers = 1 << ((9 * (3-page_table.lv)) + 12)
    assert size // pte_covers < 512, "Size must be less than 512 ptes"

    # print("Walker", page_table, hex(vaddr), hex(size))

    def _move_cursor(sz):
      nonlocal vaddr, offset, size
      vaddr, offset, size = vaddr + sz, offset + sz, size - sz

    def _level_down(va, sz):
      entry = page_table.get_entry(pte_idx:=(va // pte_covers) % 512)
      if entry & am.AMDGPU_PTE_VALID:
        assert entry & am.AMDGPU_PDE_PTE == 0, f"Must be table pt={page_table.pm.paddr:#x}, {pte_idx=} {entry=:#x}"
        child_page_table = AMPageTableEntry(AMPhysicalMemoryBlock(page_table.pm.adev, entry & 0x0000FFFFFFFFF000, 0x1000), lv=page_table.lv+1)
      else:
        assert creat_pt, "Not allowed to create new page table"
        child_page_table = AMPageTableEntry(self.palloc(0x1000, zero=True), lv=page_table.lv+1)
        page_table.set_table(pte_idx, child_page_table)
      yield from self.page_table_walker(child_page_table, va, sz, offset=offset, free_pt=free_pt, creat_pt=creat_pt)

      if free_pt and all(child_page_table.get_entry(i) & am.AMDGPU_PTE_VALID == 0 for i in range(512)):
        self.pfree(child_page_table.pm)
        page_table.set_page(pte_idx, paddr=0, valid=False)

    # First pte is not full covered
    if vaddr % pte_covers != 0:
      yield from _level_down(vaddr, min(pte_covers - (vaddr % pte_covers), size))
      _move_cursor(min(pte_covers - (vaddr % pte_covers), size))

    n_ptes = size // pte_covers
    if n_ptes > 0: yield (vaddr, offset, (vaddr // pte_covers) % 512, n_ptes, pte_covers, page_table)
    _move_cursor(n_ptes * pte_covers)

    # Last pte is not full covered
    if size > 0: yield from _level_down(vaddr, size)

  def frags_walker(self, page_table, vaddr, size, from_entry=False, free_pt=False, creat_pt=True):
    # To optimize TLB entries count, need to map pages as contigous entries. Determine size of each chunks.
    for va, off, pte_st_idx, n_ptes, pte_covers, pt in self.page_table_walker(page_table, vaddr, size, free_pt=free_pt, creat_pt=creat_pt):
      inner_off = 0
      while n_ptes > 0:
        if from_entry: frags_cnt = (pt.get_entry(pte_st_idx) >> 7) & 0x1f
        else: frags_cnt = min((va.bit_length() - 1 if va != 0 else 31), (n_ptes * pte_covers).bit_length() - 1) - 12

        update_ptes = (1 << (frags_cnt + 12)) // pte_covers
        assert update_ptes > 0, f"Invalid update_ptes {update_ptes} {frags_cnt} {pte_covers}"

        yield va + inner_off, off + inner_off, pte_st_idx, update_ptes, pte_covers, pt, frags_cnt

        pte_st_idx, n_ptes, inner_off = pte_st_idx + update_ptes, n_ptes - update_ptes, inner_off + pte_covers * update_ptes

  def map_range(self, vaddr, size, paddr=None, uncached=False, system=False, snooped=False):
    if AM_DEBUG >= 2: print(f"Mapping {vaddr=:#x} -> {paddr} ({size=:#x})")

    vaddr = vaddr - AMMemoryManager.va_allocator.base
    for va, off, pte_st_idx, n_ptes, pte_covers, page_table, frags_cnt in self.frags_walker(self.root_page_table, vaddr, size):
      if paddr is None: lpaddr, off = self.pa_allocator.alloc(n_ptes * pte_covers), 0
      else: lpaddr = paddr

      # print("here with", pte_st_idx, n_ptes)
      for pte_idx in range(n_ptes):
        assert page_table.get_entry(pte_st_idx + pte_idx) & am.AMDGPU_PTE_VALID == 0, "Entry already set"
        page_table.set_page(pte_st_idx + pte_idx, paddr=lpaddr + off, uncached=uncached, system=system, snooped=snooped, frag=frags_cnt, valid=True)
        off += pte_covers

      if AM_DEBUG >= 3: print(f"\tnptes={n_ptes:#x} incr={pte_covers:#x} upd_flags={page_table.get_entry(pte_st_idx):#x} frags={frags_cnt:#x}")

    self.adev.gmc.flush_tlb(ip="GC", vmid=0)
    self.adev.gmc.flush_tlb(ip="MM", vmid=0)

  def unmap_range(self, vaddr:int, size:int, free_paddrs=True):
    if AM_DEBUG >= 2: print(f"Unmapping {vaddr=:#x} ({size=:#x})")
    # print(f"Unmapping {vaddr=:#x} ({size=:#x})")

    vaddr = vaddr - AMMemoryManager.va_allocator.base
    for va, off, pte_st_idx, n_ptes, pte_covers, page_table, _ in self.frags_walker(self.root_page_table, vaddr, size, from_entry=True, free_pt=True):
      entry = page_table.get_entry(pte_st_idx)
      if not(entry & am.AMDGPU_PTE_SYSTEM) and free_paddrs: self.pa_allocator.free(entry & 0x0000FFFFFFFFF000)

      for pte_idx in range(n_ptes):
        assert page_table.get_entry(pte_st_idx + pte_idx) & am.AMDGPU_PTE_VALID == am.AMDGPU_PTE_VALID, "Entry must be set"
        page_table.set_page(pte_st_idx + pte_idx, paddr=0x0, valid=False)

  def map_from(self, vaddr:int, size:int, from_adev):
    if AM_DEBUG >= 2: print(f"Mapping from {vaddr=:#x} {size=:#x} from {from_adev.pcidev}")

    vaddr = vaddr - AMMemoryManager.va_allocator.base
    for va, off, pte_st_idx, n_ptes, pte_covers, page_table, _ in self.frags_walker(from_adev.mm.root_page_table, vaddr, size, from_entry=True, creat_pt=False):
      # print("hm", hex(va + AMMemoryManager.va_allocator.base), hex(off), hex(n_ptes * pte_covers))
      entry = page_table.get_entry(pte_st_idx)
      uncached = entry & am.AMDGPU_PTE_MTYPE_NV10(0, am.MTYPE_UC) == am.AMDGPU_PTE_MTYPE_NV10(0, am.MTYPE_UC)
      snooped = entry & am.AMDGPU_PTE_SNOOPED == am.AMDGPU_PTE_SNOOPED
      paddr = (entry & 0x0000FFFFFFFFF000) if entry & am.AMDGPU_PTE_SYSTEM else (entry & 0x0000FFFFFFFFF000) + from_adev.pcidev.regions[0].base_addr
      self.map_range(va + AMMemoryManager.va_allocator.base, n_ptes * pte_covers, paddr=paddr, uncached=uncached, system=True, snooped=snooped)

    # print(f"Mapping done from {from_adev.pcidev} {vaddr=:#x} {size=:#x}")

  @staticmethod
  def alloc_vaddr(size:int, align=0x1000) -> int: return AMMemoryManager.va_allocator.alloc(size, max((1 << (size.bit_length() - 1)), align))

  def valloc(self, size:int, align=0x1000, uncached=False, contigous=False) -> AMVirtualMapping:
    pm = self.palloc(round_up(size, 0x1000), zero=True) if contigous else None
    self.map_range(va:=self.alloc_vaddr(size, align), size, paddr=pm.paddr if pm else None, uncached=uncached)
    return AMVirtualMapping(va, size, pm.cpu_addr() if pm is not None else None, pm.paddr if pm is not None else None)

  def vfree(self, vm:AMVirtualMapping):
    self.unmap_range(vm.va_addr, vm.size, free_paddrs=(vm.paddr is None))
    self.va_allocator.free(vm.va_addr)
    if vm.paddr is not None: self.pa_allocator.free(vm.paddr)

  def palloc(self, size, align=0x1000, zero=True) -> AMPhysicalMemoryBlock:
    pm = AMPhysicalMemoryBlock(self.adev, self.pa_allocator.alloc(round_up(size, 0x1000), align), size)
    if zero: ctypes.memset(pm.cpu_addr(), 0, pm.size)
    return pm

  def pfree(self, pm:AMPhysicalMemoryBlock): self.pa_allocator.free(pm.paddr)

class AMDev:
  def __init__(self, pcidev, vram_bar:memoryview, doorbell_bar:memoryview, mmio_bar:memoryview):
    self.pcidev = pcidev
    self.vram, self.doorbell64, self.mmio = vram_bar, doorbell_bar, mmio_bar

    self._run_discovery()
    self._build_regs()

    # Memory manager & firmware
    self.mm = AMMemoryManager(self, self.vram_size)
    self.fw = Firmware(self)

    # Initialize IP blocks
    self.soc21 = AM_SOC21(self)
    self.gmc = AM_GMC(self)
    self.ih = AM_IH(self)
    self.psp = AM_PSP(self)
    self.smu = AM_SMU(self)
    self.gfx = AM_GFX(self)
    self.sdma = AM_SDMA(self)

    if self.psp.is_sos_alive():
      if AM_DEBUG >= 2: print("sOS is alive, issue mode1 reset...")
      self.smu.mode1_reset()
      time.sleep(0.5)

    self.soc21.init()
    self.gmc.init()
    self.ih.init()
    self.psp.init()
    self.smu.init()
    self.gfx.init()
    self.sdma.init()

  def ip_base(self, ip:str, inst:int, seg:int) -> int: return self.regs_offset[am.__dict__.get(f"{ip}_HWIP")][inst][seg]
 
  def reg(self, reg:str) -> AMRegister: return self.__dict__[reg]
  
  def rreg(self, reg:int) -> int:
    val = self.indirect_rreg(reg * 4) if reg > len(self.mmio) else self.mmio[reg]
    if AM_DEBUG >= 4 and getattr(self, '_prev_rreg', None) != (reg, val): print(f"Reading register {reg:#x} with value {val:#x}")
    self._prev_rreg = (reg, val)
    return val

  def wreg(self, reg:int, val:int):
    if AM_DEBUG >= 4: print(f"Writing register {reg:#x} with value {val:#x}")
    if reg > len(self.mmio): self.indirect_wreg(reg * 4, val)
    else: self.mmio[reg] = val

  def wreg_pair(self, reg_base:str, lo_suffix:str, hi_suffix:str, val:int):
    self.reg(f"{reg_base}{lo_suffix}").write(val & 0xffffffff)
    self.reg(f"{reg_base}{hi_suffix}").write(val >> 32)

  def indirect_rreg(self, reg:int) -> int:
    self.regBIF_BX_PF0_RSMU_INDEX.write(reg)
    assert self.regBIF_BX_PF0_RSMU_INDEX.read() == reg
    return self.regBIF_BX_PF0_RSMU_DATA.read()

  def indirect_wreg(self, reg:int, val:int):
    self.regBIF_BX_PF0_RSMU_INDEX.write(reg)
    assert self.regBIF_BX_PF0_RSMU_INDEX.read() == reg
    self.regBIF_BX_PF0_RSMU_DATA.write(val)
    assert self.regBIF_BX_PF0_RSMU_DATA.read() == val

  def wait_reg(self, reg:AMRegister, value:int, mask=0xffffffff) -> int:
    for _ in range(10000):
      if ((rval:=reg.read()) & mask) == value: return rval
      time.sleep(0.001)
    raise RuntimeError(f'wait_reg timeout reg=0x{reg.reg_off:X} mask=0x{mask:X} value=0x{value:X} last_val=0x{rval}')

  def _run_discovery(self):
    # NOTE: Fixed register to query memory size without known ip bases to find the discovery table.
    #       The table is located at the end of VRAM - 64KB and is 10KB in size.
    mmRCC_CONFIG_MEMSIZE = 0xde3
    self.vram_size = self.rreg(mmRCC_CONFIG_MEMSIZE) << 20
    self.discovery_pm = AMPhysicalMemoryBlock(self, self.vram_size - (64 << 10), 10 << 10)

    bhdr = am.struct_binary_header.from_address(self.discovery_pm.cpu_addr())
    ihdr = am.struct_ip_discovery_header.from_address(ctypes.addressof(bhdr) + bhdr.table_list[am.IP_DISCOVERY].offset)
    assert ihdr.signature == am.DISCOVERY_TABLE_SIGNATURE and not ihdr.base_addr_64_bit

    # Mapping of HW IP to Discovery HW IP
    hw_id_map = {am.__dict__[x]: int(y) for x,y in am.hw_id_map}
    self.regs_offset = collections.defaultdict(dict)

    for num_die in range(ihdr.num_dies):
      dhdr = am.struct_die_header.from_address(ctypes.addressof(bhdr) + ihdr.die_info[num_die].die_offset)

      ip_offset = ctypes.addressof(bhdr) + ctypes.sizeof(dhdr) + ihdr.die_info[num_die].die_offset
      for num_ip in range(dhdr.num_ips):
        ip = am.struct_ip_v4.from_address(ip_offset)
        ba = (ctypes.c_uint32 * ip.num_base_address).from_address(ip_offset + 8)
        for hw_ip in range(1, am.MAX_HWIP):
          if hw_ip in hw_id_map and hw_id_map[hw_ip] == ip.hw_id: self.regs_offset[hw_ip][ip.instance_number] = [x for x in ba]

        ip_offset += 8 + (8 if ihdr.base_addr_64_bit else 4) * ip.num_base_address

  def _build_regs(self):
    mods = [("MP0", mp_13_0_0), ("MP1", mp_11_0, "mmMP1"), ("NBIO", nbio_4_3_0), ("MMHUB", mmhub_3_0_0), ("GC", gc_11_0_0), ("OSSSYS", osssys_6_0_0)]
    for info in mods:
      base, module, rpref = info if len(info) == 3 else (*info, "reg")
      reg_names: Set[str] = set(k[len(rpref):] for k in module.__dict__.keys() if k.startswith(rpref) and not k.endswith("_BASE_IDX"))
      reg_fields: Dict[str, List[int, int]] = collections.defaultdict(dict)
      for k, val in module.__dict__.items():
        if k.endswith("_MASK") and ((rname:=k.split("__")[0]) in reg_names):
          reg_fields[rname][k[2+len(rname):-5].lower()] = (val, module.__dict__.get(f"{k[:-5]}__SHIFT", val.bit_length() - 1))

      for k, regval in module.__dict__.items():
        if k.startswith(rpref) and not k.endswith("_BASE_IDX") and (base_idx:=getattr(module, f"{k}_BASE_IDX", None)) is not None:
          setattr(self, k, AMRegister(self, self.ip_base(base, 0, base_idx) + regval, reg_fields.get(k[len(rpref):], {})))
