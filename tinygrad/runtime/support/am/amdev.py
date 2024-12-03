from __future__ import annotations
import os, ctypes, collections, time
from typing import Tuple, Dict, Set, Optional
from tinygrad.helpers import to_mv, mv_address, getenv
from tinygrad.runtime.autogen.am import am, mp_11_0, mp_13_0_0, nbio_4_3_0, mmhub_3_0_0, gc_11_0_0, osssys_6_0_0
from tinygrad.runtime.support.am.mm import MM, GPUPhysicalMemoryBlock
from tinygrad.runtime.support.am.firmware import Firmware
from tinygrad.runtime.support.am.ip import AM_SOC21, AM_GMC, AM_IH, AM_PSP, AM_SMU, AM_GFX, AM_SDMA
from tinygrad.runtime.support.am.hal import HAL, PCIHAL, read_pagemap

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

class AMDev:
  hal:Optional[HAL] = None

  def __init__(self, dev_idx:int):
    if AMDev.hal is None: AMDev.hal = PCIHAL()
    self.hal_dev = AMDev.hal.open_device(dev_idx)

    self.vram_cpu_addr, self.vram = AMDev.hal.map_pci_range(self.hal_dev, bar=0, cast='B')
    self.doorbell_cpu_addr, self.doorbell64 = AMDev.hal.map_pci_range(self.hal_dev, bar=2, cast='Q')
    self.mmio_cpu_addr, self.mmio = AMDev.hal.map_pci_range(self.hal_dev, bar=5, cast='I')

    # print(read_pagemap(self.vram_cpu_addr))

    self._run_discovery()
    self._build_regs()

    # Memory manager & firmware
    self.mm = MM(self, self.vram_size)
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
    self.regRLC_SPM_MC_CNTL.write(0xf)
    self.ih.init()
    self.psp.init()
    self.smu.init()
    self.gfx.init()
    self.sdma.init()

    # print(read_pagemap(self.vram_cpu_addr))
    # exit(0)

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
    self.discovery_pm = GPUPhysicalMemoryBlock(self, self.vram_size - (64 << 10), 10 << 10)

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
