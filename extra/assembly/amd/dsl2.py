# dsl2.py - clean DSL for AMD assembly
from __future__ import annotations
from enum import IntEnum

# ══════════════════════════════════════════════════════════════
# Registers - unified src encoding space (0-511)
# ══════════════════════════════════════════════════════════════

class Reg:
  _NAMES = {106: "VCC_LO", 107: "VCC_HI", 124: "NULL", 125: "M0", 126: "EXEC_LO", 127: "EXEC_HI",
            240: "0.5", 241: "-0.5", 242: "1.0", 243: "-1.0", 244: "2.0", 245: "-2.0", 246: "4.0", 247: "-4.0",
            248: "INV_2PI", 250: "DPP16", 253: "SCC", 255: "LIT"}
  _PAIRS = {106: "VCC", 126: "EXEC"}

  def __init__(self, offset: int = 0, sz: int = 512):
    self.offset, self.sz = offset, sz
  def __getitem__(self, key):
    if isinstance(key, slice):
      start, stop = key.start or 0, key.stop or (self.sz - 1)
      if start < 0 or stop >= self.sz: raise RuntimeError(f"slice [{start}:{stop}] out of bounds for size {self.sz}")
      return Reg(self.offset + start, stop - start + 1)  # inclusive
    if key < 0 or key >= self.sz: raise RuntimeError(f"index {key} out of bounds for size {self.sz}")
    return Reg(self.offset + key, 1)
  def __repr__(self):
    o, sz = self.offset, self.sz
    if 256 <= o < 512:
      idx = o - 256
      return f"v[{idx}]" if sz == 1 else f"v[{idx}:{idx + sz - 1}]"
    if o < 106: return f"s[{o}]" if sz == 1 else f"s[{o}:{o + sz - 1}]"
    if sz == 2 and o in self._PAIRS: return self._PAIRS[o]
    if sz == 1 and o in self._NAMES: return self._NAMES[o]
    if 108 <= o < 124:
      idx = o - 108
      return f"ttmp[{idx}]" if sz == 1 else f"ttmp[{idx}:{idx + sz - 1}]"
    if sz == 1 and 128 <= o <= 192: return str(o - 128)  # integers 0-64
    if sz == 1 and 193 <= o <= 208: return str(-(o - 192))  # integers -1 to -16
    raise RuntimeError(f"unknown register: offset={o}, sz={sz}")

# Full src encoding space
src = Reg(0, 512)

# Slices for each region (inclusive end)
s = src[0:105]           # SGPR0-105
VCC_LO = src[106]
VCC_HI = src[107]
VCC = src[106:107]
ttmp = src[108:123]      # TTMP0-15
NULL = src[124]
M0 = src[125]
EXEC_LO = src[126]
EXEC_HI = src[127]
EXEC = src[126:127]
# 128: 0, 129-192: integers 1-64, 193-208: integers -1 to -16
# 240-248: float constants (0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 1/(2*PI))
DPP16 = src[250]
SCC = src[253]
# 255: literal constant
v = src[256:511]         # VGPR0-255

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

  # Float encoding map
  _FLOAT_ENC = {0.5: 240, -0.5: 241, 1.0: 242, -1.0: 243, 2.0: 244, -2.0: 245, 4.0: 246, -4.0: 247}

  def _set(self, name: str, val):
    field = next(f for n, f in self._fields if n == name)
    if isinstance(field, VGPRField):
      raw = (val.offset - 256) if isinstance(val, Reg) else val
    elif isinstance(field, SrcField):
      if isinstance(val, Reg): raw = val.offset
      elif isinstance(val, float): raw = self._FLOAT_ENC.get(val, 255)  # 255 = literal
      elif isinstance(val, int) and 0 <= val <= 64: raw = 128 + val
      elif isinstance(val, int) and -16 <= val < 0: raw = 192 - val
      else: raw = val
    else:
      raw = val.value if hasattr(val, 'value') else val
    self._raw = (self._raw & ~(field.mask() << field.lo)) | ((raw & field.mask()) << field.lo)

  def _get(self, name: str):
    field = next(f for n, f in self._fields if n == name)
    raw = (self._raw >> field.lo) & field.mask()
    if isinstance(field, VGPRField): return src[256 + raw]
    if isinstance(field, SrcField): return src[raw]
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
