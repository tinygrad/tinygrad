# dsl2.py - clean DSL for AMD assembly
from __future__ import annotations
from enum import Enum

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
  def __init__(self, hi: int, lo: int, default: int | None = None):
    self.hi, self.lo, self.default = hi, lo, default
  def mask(self) -> int: return (1 << (self.hi - self.lo + 1)) - 1
  def encode(self, val) -> int:
    assert isinstance(val, int), f"BitField.encode expects int, got {type(val).__name__}"
    return val
  def decode(self, val): return val
  def set(self, raw: int, val) -> int:
    if val is None:
      if self.default is None: raise RuntimeError("no value provided and no default set")
      val = self.default
    encoded = self.encode(val)
    if encoded < 0 or encoded > self.mask(): raise RuntimeError(f"value {encoded} doesn't fit in {self.hi - self.lo + 1} bits")
    return (raw & ~(self.mask() << self.lo)) | (encoded << self.lo)
  def get(self, raw: int): return self.decode((raw >> self.lo) & self.mask())

class FixedBitField(BitField):
  def set(self, raw: int, val=None) -> int:
    assert val is None, f"FixedBitField does not accept values, got {val}"
    return super().set(raw, self.default)

class EnumBitField(BitField):
  def __init__(self, hi: int, lo: int, enum_cls):
    super().__init__(hi, lo)
    self._enum = enum_cls
  def encode(self, val) -> int:
    if not isinstance(val, self._enum): raise RuntimeError(f"expected {self._enum.__name__}, got {type(val).__name__}")
    return val.value
  def decode(self, raw): return self._enum(raw)

class SignedBitField(BitField):
  """Signed immediate field - encode validates range, decode performs sign extension."""
  def encode(self, val):
    bits = self.hi - self.lo + 1
    min_val, max_val = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if not (min_val <= val <= max_val): raise RuntimeError(f"signed value {val} out of range [{min_val}, {max_val}] for {bits}-bit field")
    return val & ((1 << bits) - 1)  # two's complement
  def decode(self, raw):
    bits = self.hi - self.lo + 1
    return raw - (1 << bits) if raw >= (1 << (bits - 1)) else raw

# ══════════════════════════════════════════════════════════════
# Typed fields
# ══════════════════════════════════════════════════════════════

import struct
def _f32(f: float) -> int: return struct.unpack('I', struct.pack('f', f))[0]

class SrcField(BitField):
  _valid_range = (0, 511)  # inclusive
  _FLOAT_ENC = {0.5: 240, -0.5: 241, 1.0: 242, -1.0: 243, 2.0: 244, -2.0: 245, 4.0: 246, -4.0: 247}

  def __init__(self, hi: int, lo: int, default=None):
    super().__init__(hi, lo, default)
    expected_size = self._valid_range[1] - self._valid_range[0] + 1
    actual_size = 1 << (hi - lo + 1)
    if actual_size != expected_size:
      raise RuntimeError(f"{self.__class__.__name__}: field size {hi - lo + 1} bits ({actual_size}) doesn't match range {self._valid_range} ({expected_size})")

  def encode(self, val) -> int:
    """Encode value. Returns 255 (literal marker) for out-of-range values."""
    if isinstance(val, Reg): offset = val.offset
    elif isinstance(val, float): offset = self._FLOAT_ENC.get(val, 255)
    elif isinstance(val, int) and 0 <= val <= 64: offset = 128 + val
    elif isinstance(val, int) and -16 <= val < 0: offset = 192 - val
    elif isinstance(val, int): offset = 255  # literal
    else: raise RuntimeError(f"invalid src value {val}")
    if not (self._valid_range[0] <= offset <= self._valid_range[1]):
      raise RuntimeError(f"{self.__class__.__name__}: {val} (offset {offset}) out of range {self._valid_range}")
    return offset - self._valid_range[0]

  def decode(self, raw): return src[raw + self._valid_range[0]]

class VGPRField(SrcField): _valid_range = (256, 511)
class SGPRField(SrcField): _valid_range = (0, 127)
class SSrcField(SrcField): _valid_range = (0, 255)

class SBaseField(BitField):
  """SMEM sbase field: encoded = sgpr_index // 2. Must be even-aligned SGPR."""
  def encode(self, val):
    if not isinstance(val, Reg): raise RuntimeError(f"SBaseField requires Reg, got {type(val).__name__}")
    if not (0 <= val.offset < 128): raise RuntimeError(f"SBaseField requires SGPR, got offset {val.offset}")
    if val.offset & 1: raise RuntimeError(f"SBaseField requires even SGPR index, got s[{val.offset}]")
    return val.offset >> 1
  def decode(self, raw): return src[raw << 1]

class SRsrcField(BitField):
  """MIMG/MUBUF srsrc/ssamp field: encoded = sgpr_index // 4. Must be 4-aligned SGPR."""
  def encode(self, val):
    if not isinstance(val, Reg): raise RuntimeError(f"SRsrcField requires Reg, got {type(val).__name__}")
    if not (0 <= val.offset < 128): raise RuntimeError(f"SRsrcField requires SGPR, got offset {val.offset}")
    if val.offset & 3: raise RuntimeError(f"SRsrcField requires 4-aligned SGPR index, got s[{val.offset}]")
    return val.offset >> 2
  def decode(self, raw): return src[raw << 2]

class VDSTYField(BitField):
  """VOPD vdsty: encoded = vgpr_idx >> 1. Only even VGPRs allowed (vdstx determines LSB)."""
  def encode(self, val):
    if not isinstance(val, Reg): raise RuntimeError(f"VDSTYField requires Reg, got {type(val).__name__}")
    if not (256 <= val.offset < 512): raise RuntimeError(f"VDSTYField requires VGPR, got offset {val.offset}")
    idx = val.offset - 256
    if idx & 1: raise RuntimeError(f"VDSTYField requires even VGPR index, got v[{idx}]")
    return idx >> 1
  def decode(self, raw): return raw  # raw value, actual vdsty = (raw << 1) | ((vdstx & 1) ^ 1)

# ══════════════════════════════════════════════════════════════
# Inst base class
# ══════════════════════════════════════════════════════════════

class Inst:
  _fields: list[tuple[str, BitField]]
  _size: int

  def __init_subclass__(cls):
    cls._fields = [(name, val) for name, val in cls.__dict__.items() if isinstance(val, BitField)]
    cls._size = (max(f.hi for _, f in cls._fields) + 8) // 8

  def __init__(self, *args, **kwargs):
    self._raw = 0
    self._literal: int | None = None
    args_iter = iter(args)
    for name, field in self._fields:
      if isinstance(field, FixedBitField): val = None
      elif name in kwargs: val = kwargs[name]
      else: val = next(args_iter, None)
      self._raw = field.set(self._raw, val)
      # Capture literal for SrcFields that encoded to 255
      if isinstance(field, SrcField) and val is not None and field.encode(val) + field._valid_range[0] == 255 and self._literal is None:
        self._literal = _f32(val) if isinstance(val, float) else val & 0xFFFFFFFF

  def __getattribute__(self, name: str):
    if name.startswith('_') or name in ('size', 'to_bytes', 'from_bytes', 'op_name'):
      return object.__getattribute__(self, name)
    fields = object.__getattribute__(self, '_fields')
    field = next((f for n, f in fields if n == name), None)
    if field is None: return object.__getattribute__(self, name)
    return field.get(object.__getattribute__(self, '_raw'))

  @property
  def op_name(self) -> str: return self.op.name
  def size(self) -> int: return self._size + (4 if self._literal is not None else 0)

  def to_bytes(self) -> bytes:
    result = self._raw.to_bytes(self._size, 'little')
    if self._literal is not None:
      result += (self._literal & 0xFFFFFFFF).to_bytes(4, 'little')
    return result

  @classmethod
  def from_bytes(cls, data: bytes):
    inst = object.__new__(cls)
    inst._raw = int.from_bytes(data[:cls._size], 'little')
    inst._literal = None
    for name, field in cls._fields:
      if isinstance(field, SrcField) and field.get(inst._raw).offset == 255:
        assert len(data) >= cls._size + 4, f"literal marker found but data too short: {len(data)} < {cls._size + 4}"
        inst._literal = int.from_bytes(data[cls._size:cls._size + 4], 'little')
        break
    return inst

  def __repr__(self):
    args = [n for n, f in self._fields if n != 'op' and not isinstance(f, FixedBitField)]
    return f"{self.op.name.lower()}({', '.join(repr(getattr(self, n)) for n in args)})"

# ══════════════════════════════════════════════════════════════
# VOP1
# ══════════════════════════════════════════════════════════════

class VOP1Op(Enum):
  V_NOP_E32 = 0
  V_MOV_B32_E32 = 1

class VOP2Op(Enum):
  V_CNDMASK_B32_E32 = 1

class VOP1(Inst):
  encoding = FixedBitField(31, 25, 0b0111111)
  op = EnumBitField(16, 9, VOP1Op)
  vdst = VGPRField(24, 17)
  src0 = SrcField(8, 0)
