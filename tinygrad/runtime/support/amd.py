import functools, re, tinygrad.runtime.autogen.am
from dataclasses import dataclass
from tinygrad.helpers import getbits, fetch

ROCM_URL = "https://raw.githubusercontent.com/ROCm/rocm-systems/cccc350dc620e61ae2554978b62ab3532dc10bd9/projects"

@dataclass
class AMDReg:
  name:str; offset:int; segment:int; fields:dict[str, tuple[int, int]]; bases:dict[int, tuple[int, ...]] # noqa: E702
  def __post_init__(self): self.addr:dict[int, int] = { inst: bases[self.segment] + self.offset for inst, bases in self.bases.items() }

  def encode(self, **kwargs) -> int: return functools.reduce(int.__or__, (value << self.fields[name][0] for name,value in kwargs.items()), 0)
  def decode(self, val: int) -> dict: return {name:getbits(val, start, end) for name,(start,end) in self.fields.items()}

  def fields_mask(self, *names) -> int:
    return functools.reduce(int.__or__, ((((1 << (self.fields[nm][1]-self.fields[nm][0]+1)) - 1) << self.fields[nm][0]) for nm in names), 0)

@dataclass
class AMDIP:
  name:str; version:tuple[int, ...]; bases:dict[int, tuple[int, ...]] # noqa: E702

  @functools.cached_property
  def regs(self): return import_asic_regs(self.name, self.version, cls=functools.partial(AMDReg, bases=self.bases))

  def __getattr__(self, name:str):
    if name in self.regs: return self.regs[name]
    if (name10:=name.replace('reg', 'mm')) in self.regs: return self.regs[name10]
    raise AttributeError(f"{self.name.upper()} has no register {name}")

# load the greatest module with matching major version that's less than or equal to the target version
# this is not universally correct, see below for an example, but appears reliable for recent gpus
# https://github.com/torvalds/linux/blob/9207d47f966be9f4d52e7e0119ac2b7a7e366f3e/drivers/gpu/drm/amd/amdgpu/amdgpu_discovery.c#L3163
def import_module(name:str, target:tuple[int, ...], submod=""):
  mod = getattr(tinygrad.runtime.autogen.am, submod) if submod else tinygrad.runtime.autogen.am
  if (children:=[c for c in mod.__all__ if c.startswith(name) and (v:=tuple(map(int, c.split('_')[1:])))[0] == target[0] and v <= target]):
    return getattr(mod, children[-1])
  raise ImportError(f"Failed to import {submod+'.' if submod else ''}{name} {'.'.join(map(str, target))}")

def header_download(file, url) -> str: return fetch(f"{url}/{file}", subdir="defines").read_text()

def import_soc(ip): return getattr(tinygrad.runtime.autogen.am, f"soc_{ip[0]}")

def import_pmc(ip) -> dict[str, tuple[str, int]]:
  res:dict[str, tuple[str, int]] = {}

  # NOTE: precise arch for mi300+, generic for others, since rocm headers lack some archs
  arch = f"gfx{ip[0]}{ip[1]:x}{ip[2]:x}" if ip[0] == 9 else f"gfx{ip[0]}"

  for sec in header_download("rocprofiler-compute/src/rocprof_compute_soc/profile_configs/counter_defs.yaml", ROCM_URL).split('- name: ')[1:]:
    for arch_spec in sec.split('- architectures:')[1:]:
      if arch in arch_spec and (block:=re.search(r'block:\s*([A-Za-z0-9_]+)', arch_spec)) and (ev:=re.search(r'event:\s*(\d+)', arch_spec)):
        res[sec.splitlines()[0].strip()] = (block.group(1), int(ev.group(1)))

  return res

def import_asic_regs(prefix:str, version:tuple[int, ...], cls=AMDReg) -> dict[str, AMDReg]:
  return {reg:cls(name=reg, offset=off, segment=seg, fields=fields) for reg,(off,seg,fields) in import_module(prefix, version, submod="regs").items()}
