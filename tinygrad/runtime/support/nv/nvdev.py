from __future__ import annotations
import ctypes, collections, time, dataclasses, functools, fcntl, os, hashlib, re
from tinygrad.helpers import mv_address, getenv, round_up, DEBUG, temp, fetch, getbits
from tinygrad.runtime.autogen.nv import nv
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.allocator import TLSFAllocator
from tinygrad.runtime.support.nv.ip import NV_FLCN, NV_GSP
from tinygrad.runtime.support.nvd import alloc_sysmem

NV_DEBUG = getenv("NV_DEBUG", 0)

class NVReg:
  def __init__(self, nvdev, base, addr, fields=None): self.nvdev, self.addr, self.fields = nvdev, base + addr, fields

  def add_field(self, name:str, start:int, end:int): self.fields[name] = (start, end)

  def read(self): return self.nvdev.rreg(self.addr)
  def read_bitfields(self) -> dict[str, int]: return self.decode(self.read())

  def write(self, _ini_val:int=0, **kwargs): self.nvdev.wreg(self.addr, _ini_val | self.encode(**kwargs))

  def update(self, **kwargs): self.write(self.read() & ~self.mask(**kwargs), **kwargs)

  def mask(self, **kwargs):
    return functools.reduce(int.__or__, ((((1 << (self.fields[nm][1]-self.fields[nm][0] + 1)) - 1) << self.fields[nm][0]) for nm in kwargs.keys()), 0)

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

class NVDev:
  def __init__(self, devfmt, mmio:MMIOInterface, vram:MMIOInterface, rom:bytes):
    self.devfmt, self.mmio, self.vram, self.rom = devfmt, mmio, vram, rom
    self.included_files, self.reg_names, self.reg_offsets = set(), set(), {}

    self._early_init()

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
    self.include("src/common/inc/swref/published/ampere/ga102/dev_gc6_island.h")
    self.include("src/common/inc/swref/published/ampere/ga102/dev_gc6_island_addendum.h")

    self.vram_size = self.NV_PGC6_AON_SECURE_SCRATCH_GROUP_42.read() << 20

  def _alloc_boot_struct(self, typ):
    va, paddrs = alloc_sysmem(ctypes.sizeof(typ), contigous=True)
    return typ.from_address(va), paddrs[0]

  def _download(self, file) -> str:
    url = f"https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/e8113f665d936d9f30a6d508f3bacd1e148539be/{file}"
    return fetch(url, subdir="defines").read_text()

  def include(self, file) -> str:
    if file in self.included_files: return
    self.included_files.add(file)

    txt = self._download(file)

    PARAM = re.compile(r'#define\s+(\w+)\s*\(\s*(\w+)\s*\)\s*(.+)')
    CONST = re.compile(r'#define\s+(\w+)\s+([0-9A-Fa-fx]+)')
    BITFLD = re.compile(r'#define\s+(\w+)\s+(\d+):(\d+)')

    regs_off = {'NV_PFALCON_FALCON': (None, 0x0), 'NV_PGSP_FALCON': 0x0, 'NV_PSEC_FALCON': 0x0, 'NV_PRISCV_RISCV': (None, 0x1000), 'NV_PGC6_AON': 0x0,
      'NV_PFALCON_FBIF': (None, 0x600), 'NV_PFALCON2_FALCON': (None, 0x1000), 'NV_PBUS': 0x0, 'NV_PFB': 0x0}

    for raw in txt.splitlines():
      if raw.startswith("#define "):
        if (m := BITFLD.match(raw)):
          name, hi, lo = m.groups()
          for r in self.reg_names:
            if name.startswith(r+"_"):
              self.__dict__[r].add_field(name[len(r)+1:].lower(), int(lo), int(hi))
              break
          self.reg_offsets[name] = (int(lo), int(hi))
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
