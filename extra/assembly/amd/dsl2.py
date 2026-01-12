# dsl2.py - clean DSL for AMD assembly
from __future__ import annotations
from enum import IntEnum

# ══════════════════════════════════════════════════════════════
# Registers
# ══════════════════════════════════════════════════════════════

class Reg:
  __slots__ = ('idx', 'count')
  def __init__(self, idx: int, count: int = 1): self.idx, self.count = idx, count
  def __repr__(self): return f"{self._prefix}[{self.idx}]" if self.count == 1 else f"{self._prefix}[{self.idx}:{self.idx+self.count-1}]"

class SGPR(Reg): _prefix = 's'
class VGPR(Reg): _prefix = 'v'
class TTMP(Reg): _prefix = 'ttmp'

class RegFactory:
  def __init__(self, cls: type[Reg], enc_offset: int = 0): self.cls, self.enc_offset = cls, enc_offset
  def __getitem__(self, key):
    if isinstance(key, slice): return self.cls(key.start, key.stop - key.start)
    return self.cls(key)

s, v, ttmp = RegFactory(SGPR, 0), RegFactory(VGPR, 256), RegFactory(TTMP, 108)

# ══════════════════════════════════════════════════════════════
# BitField
# ══════════════════════════════════════════════════════════════

class BitField:
  required_size = None
  def __init__(self, hi: int, lo: int):
    self.hi, self.lo = hi, lo
    if self.required_size is not None and (hi - lo + 1) != self.required_size:
      raise RuntimeError(f"wrong size field: expected {self.required_size}, got {hi - lo + 1}")
  def mask(self) -> int: return (1 << (self.hi - self.lo + 1)) - 1
  def __eq__(self, val): return (self, val)

# ══════════════════════════════════════════════════════════════
# OpField with auto IntEnum
# ══════════════════════════════════════════════════════════════

class OpFieldMeta(type):
  def __new__(mcs, name, bases, ns):
    members = {k: v for k, v in ns.items() if isinstance(v, int) and not k.startswith('_')}
    cls = super().__new__(mcs, name, bases, {k: v for k, v in ns.items() if k not in members})
    if members: cls._enum = IntEnum(name, members); [setattr(cls, k, cls._enum[k]) for k in members]
    return cls

class OpField(BitField, metaclass=OpFieldMeta):
  _enum = None

# ══════════════════════════════════════════════════════════════
# Typed fields
# ══════════════════════════════════════════════════════════════

class VGPRField(BitField):
  required_size = 8

class SrcField(BitField):
  required_size = 9

# ══════════════════════════════════════════════════════════════
# Inst base class
# ══════════════════════════════════════════════════════════════

class Inst:
  _encoding: tuple[BitField, int] | None = None
  _fields: list[tuple[str, BitField]]
  _size: int = 4

  def __init_subclass__(cls):
    cls._encoding = None
    cls._fields = []
    for name, val in cls.__dict__.items():
      if isinstance(val, tuple) and len(val) == 2 and isinstance(val[0], BitField):
        cls._encoding = val
      elif isinstance(val, BitField):
        cls._fields.append((name, val))
    cls._fields.sort(key=lambda x: -x[1].hi)
    if cls._fields: cls._size = (max(f.hi for _, f in cls._fields) + 8) // 8

  def __init__(self, op, *args):
    self._raw = 0
    if self._encoding:
      bf, val = self._encoding
      self._raw |= (val & bf.mask()) << bf.lo
    # Set fields: op first, then remaining fields from args
    self._set('op', op)
    non_op = [(n, f) for n, f in self._fields if n != 'op']
    for (name, _), val in zip(non_op, args): self._set(name, val)

  def _set(self, name: str, val):
    field = next(f for n, f in self._fields if n == name)
    if isinstance(field, VGPRField):
      raw = val.idx if isinstance(val, VGPR) else val
    elif isinstance(field, SrcField):
      if isinstance(val, VGPR): raw = 256 + val.idx
      elif isinstance(val, SGPR): raw = val.idx
      else: raw = val
    else:
      raw = val.value if hasattr(val, 'value') else val
    self._raw = (self._raw & ~(field.mask() << field.lo)) | ((raw & field.mask()) << field.lo)

  def _get(self, name: str):
    field = next(f for n, f in self._fields if n == name)
    raw = (self._raw >> field.lo) & field.mask()
    if isinstance(field, VGPRField): return VGPR(raw)
    if isinstance(field, SrcField):
      if raw >= 256: return VGPR(raw - 256)
      if raw <= 105: return SGPR(raw)
      return raw  # TODO: handle special regs, constants
    if isinstance(field, OpField) and field._enum: return field._enum(raw)
    return raw

  def to_bytes(self) -> bytes: return self._raw.to_bytes(self._size, 'little')

  @classmethod
  def from_bytes(cls, data: bytes):
    inst = object.__new__(cls)
    inst._raw = int.from_bytes(data[:cls._size], 'little')
    return inst

  def __repr__(self):
    op = self._get('op')
    name = op.name.lower() if hasattr(op, 'name') else f"op{op}"
    args = [repr(self._get(n)) for n, _ in self._fields if n != 'op']
    return f"{name}({', '.join(args)})"

# ══════════════════════════════════════════════════════════════
# VOP1
# ══════════════════════════════════════════════════════════════

class VOP1Op(OpField):
  V_NOP_E32 = 0
  V_MOV_B32_E32 = 1

class VOP1(Inst):
  encoding = BitField(31, 25) == 0b0111111
  op = VOP1Op(16, 9)
  vdst = VGPRField(24, 17)
  src0 = SrcField(8, 0)
