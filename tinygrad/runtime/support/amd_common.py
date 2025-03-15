import functools
from collections import defaultdict
from dataclasses import dataclass
from math import log2
from tinygrad.helpers import getbits

@dataclass(frozen=True)
class IPReg:
  name: str
  offset: int
  segment: int
  fields: dict[str, tuple[int, int]]
  def encode_bitfields(self, **kwargs) -> int:
    return functools.reduce(int.__or__, (value << self.fields[name][0] for name,value in kwargs.items()))
  def decode_bitfields(self, val: int) -> dict:
    return {name:getbits(val, start, end) for name,(start,end) in self.fields.items()}

def collect_registers(module, cls=IPReg) -> dict[str, IPReg]:
  def _strip_name(name): return name[next(i for i,c in enumerate(name) if c.isupper()):]
  offsets: dict[str, int] = {}
  bases: dict[str, int] = {}
  fields: defaultdict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
  for field_name,field_value in module.__dict__.items():
    if (field_name.startswith('reg') or field_name.startswith('mm')) and field_name.endswith('_BASE_IDX'):
      bases[field_name[:-len('_BASE_IDX')]] = field_value
    elif field_name.startswith('reg') or field_name.startswith('mm'):
      offsets[field_name] = field_value
    elif '__' in field_name and field_name.endswith('_MASK'):
      reg_name, reg_field_name = field_name[:-len('_MASK')].split('__')
      fields[reg_name][reg_field_name.lower()] = (int(log2(field_value & -field_value)), int(log2(field_value)))
  # NOTE: Some registers like regGFX_IMU_FUSESTRAP in gc_11_0_0 are missing base idx, just skip them
  return {reg:cls(name=reg, offset=off, segment=bases[reg], fields=fields[_strip_name(reg)]) for reg,off in offsets.items() if reg in bases}
