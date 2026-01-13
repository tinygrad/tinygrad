# dsl.py - clean DSL for AMD assembly

# ══════════════════════════════════════════════════════════════
# Registers - unified src encoding space (0-511)
# ══════════════════════════════════════════════════════════════

def _reg_size(t: str | None) -> int: return {'b64': 2, 'f64': 2, 'u64': 2, 'i64': 2, 'b128': 4}.get(t, 1)

class Reg:
  _NAMES = {106: "VCC_LO", 107: "VCC_HI", 124: "NULL", 125: "M0", 126: "EXEC_LO", 127: "EXEC_HI",
            240: "0.5", 241: "-0.5", 242: "1.0", 243: "-1.0", 244: "2.0", 245: "-2.0", 246: "4.0", 247: "-4.0",
            248: "INV_2PI", 250: "DPP16", 253: "SCC", 255: "LIT"}
  _PAIRS = {106: "VCC", 126: "EXEC"}

  def __init__(self, offset: int = 0, sz: int = 512, *, neg: bool = False, abs_: bool = False, hi: bool = False):
    self.offset, self.sz = offset, sz
    self.neg, self.abs_, self.hi = neg, abs_, hi

  # TODO: remove these legacy aliases
  @property
  def count(self): return self.sz
  @property
  def idx(self): return self.offset

  def __hash__(self): return hash((self.offset, self.sz, self.neg, self.abs_, self.hi))
  def __getitem__(self, key):
    if isinstance(key, slice):
      start, stop = key.start or 0, key.stop or (self.sz - 1)
      if start < 0 or stop >= self.sz: raise RuntimeError(f"slice [{start}:{stop}] out of bounds for size {self.sz}")
      return Reg(self.offset + start, stop - start + 1)
    if key < 0 or key >= self.sz: raise RuntimeError(f"index {key} out of bounds for size {self.sz}")
    return Reg(self.offset + key, 1)
  def __eq__(self, other):
    if isinstance(other, Reg):
      return (self.offset == other.offset and self.sz == other.sz and
              self.neg == other.neg and self.abs_ == other.abs_ and self.hi == other.hi)
    return NotImplemented
  def __add__(self, other):
    if isinstance(other, int): return Reg(self.offset + other, self.sz)
    return NotImplemented
  def __neg__(self) -> 'Reg': return Reg(self.offset, self.sz, neg=not self.neg, abs_=self.abs_, hi=self.hi)
  def __abs__(self) -> 'Reg': return Reg(self.offset, self.sz, neg=self.neg, abs_=True, hi=self.hi)
  @property
  def h(self) -> 'Reg': return Reg(self.offset, self.sz, neg=self.neg, abs_=self.abs_, hi=True)
  @property
  def l(self) -> 'Reg': return Reg(self.offset, self.sz, neg=self.neg, abs_=self.abs_, hi=False)
  def __repr__(self):
    o, sz = self.offset, self.sz
    if 256 <= o < 512:
      idx = o - 256
      base = f"v[{idx}]" if sz == 1 else f"v[{idx}:{idx + sz - 1}]"
    elif o < 106: base = f"s[{o}]" if sz == 1 else f"s[{o}:{o + sz - 1}]"
    elif sz == 2 and o in self._PAIRS: base = self._PAIRS[o]
    elif sz == 1 and o in self._NAMES: base = self._NAMES[o]
    elif 108 <= o < 124:
      idx = o - 108
      base = f"ttmp[{idx}]" if sz == 1 else f"ttmp[{idx}:{idx + sz - 1}]"
    elif sz == 1 and 128 <= o <= 192: base = str(o - 128)
    elif sz == 1 and 193 <= o <= 208: base = str(-(o - 192))
    else: raise RuntimeError(f"unknown register: offset={o}, sz={sz}")
    if self.hi: base += ".h"
    if self.abs_: base = f"abs({base})"
    if self.neg: base = f"-{base}"
    return base

# Full src encoding space
src = Reg(0, 512)

# Slices for each region (inclusive end)
s = src[0:105]           # SGPR0-105
VCC_LO = src[106]
VCC_HI = src[107]
VCC = src[106:107]
ttmp = src[108:123]      # TTMP0-15
NULL = OFF = src[124]
M0 = src[125]
EXEC_LO = src[126]
EXEC_HI = src[127]
EXEC = src[126:127]
# 128: 0, 129-192: integers 1-64, 193-208: integers -1 to -16
# 240-248: float constants (0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 1/(2*PI))
INV_2PI = src[248]
DPP16 = src[250]
SCC = src[253]
# 255: literal constant
v = src[256:511]         # VGPR0-255

# ══════════════════════════════════════════════════════════════
# BitField
# ══════════════════════════════════════════════════════════════

class _Bits:
  """Helper for defining bit fields with slice syntax: bits[hi:lo] or bits[n]."""
  def __getitem__(self, key) -> 'BitField': return BitField(key.start, key.stop) if isinstance(key, slice) else BitField(key, key)
bits = _Bits()

class BitField:
  def __init__(self, hi: int, lo: int, default: int = 0):
    self.hi, self.lo, self.default, self.name = hi, lo, default, None
  def __set_name__(self, owner, name): self.name = name
  def __eq__(self, other) -> 'FixedBitField':
    if isinstance(other, int): return FixedBitField(self.hi, self.lo, other)
    return NotImplemented
  def enum(self, enum_cls) -> 'EnumBitField': return EnumBitField(self.hi, self.lo, enum_cls)
  def mask(self) -> int: return (1 << (self.hi - self.lo + 1)) - 1
  def encode(self, val) -> int:
    assert isinstance(val, int), f"BitField.encode expects int, got {type(val).__name__}"
    return val
  def decode(self, val): return val
  def set(self, raw: int, val) -> int:
    if val is None: val = self.default
    encoded = self.encode(val)
    if encoded < 0 or encoded > self.mask(): raise RuntimeError(f"field '{self.name}': value {encoded} doesn't fit in {self.hi - self.lo + 1} bits")
    return (raw & ~(self.mask() << self.lo)) | (encoded << self.lo)
  def __get__(self, obj, objtype=None):
    if obj is None: return self
    return self.decode((obj._raw >> self.lo) & self.mask())

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

# ══════════════════════════════════════════════════════════════
# Typed fields
# ══════════════════════════════════════════════════════════════

import struct
def _f32(f: float) -> int: return struct.unpack('I', struct.pack('f', f))[0]

class SrcField(BitField):
  _valid_range = (0, 511)  # inclusive
  _FLOAT_ENC = {0.5: 240, -0.5: 241, 1.0: 242, -1.0: 243, 2.0: 244, -2.0: 245, 4.0: 246, -4.0: 247}

  def __init__(self, hi: int, lo: int, default=s[0]):
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
    else: raise TypeError(f"invalid src value {val}")
    if not (self._valid_range[0] <= offset <= self._valid_range[1]):
      raise TypeError(f"{self.__class__.__name__}: {val} (offset {offset}) out of range {self._valid_range}")
    return offset - self._valid_range[0]

  def decode(self, raw): return src[raw + self._valid_range[0]]

class VGPRField(SrcField):
  _valid_range = (256, 511)
  def __init__(self, hi: int, lo: int, default=v[0]): super().__init__(hi, lo, default)
  def encode(self, val) -> int:
    if not isinstance(val, Reg): raise TypeError(f"VGPRField requires Reg, got {type(val).__name__}")
    # For 8-bit vdst fields in VOP1/VOP2 16-bit ops, bit 7 is opsel for dest half
    encoded = super().encode(val)
    if val.hi and (self.hi - self.lo + 1) == 8:
      if encoded >= 128:
        raise ValueError(f"VGPRField: v[{encoded}].h not encodable in 8-bit field (v[0:127] only for .h)")
      encoded |= 0x80
    return encoded
class SGPRField(SrcField): _valid_range = (0, 127)
class SSrcField(SrcField): _valid_range = (0, 255)

class AlignedSGPRField(BitField):
  """SGPR field with alignment requirement. Encoded as sgpr_index // alignment."""
  _align: int = 2
  def encode(self, val):
    if isinstance(val, int) and val == 0: return 0  # default: encode as s[0]
    if not isinstance(val, Reg): raise TypeError(f"{self.__class__.__name__} requires Reg, got {type(val).__name__}")
    if not (0 <= val.offset < 128): raise ValueError(f"{self.__class__.__name__} requires SGPR, got offset {val.offset}")
    if val.offset & (self._align - 1): raise ValueError(f"{self.__class__.__name__} requires {self._align}-aligned SGPR, got s[{val.offset}]")
    return val.offset >> (self._align.bit_length() - 1)
  def decode(self, raw): return src[raw << (self._align.bit_length() - 1)]

class SBaseField(AlignedSGPRField): _align = 2
class SRsrcField(AlignedSGPRField): _align = 4

class VDSTYField(BitField):
  """VOPD vdsty: encoded = vgpr_idx >> 1. Actual vgpr = (encoded << 1) | ((vdstx & 1) ^ 1)."""
  def encode(self, val):
    if not isinstance(val, Reg): raise TypeError(f"VDSTYField requires Reg, got {type(val).__name__}")
    if not (256 <= val.offset < 512): raise ValueError(f"VDSTYField requires VGPR, got offset {val.offset}")
    return (val.offset - 256) >> 1
  def decode(self, raw): return raw  # raw value, actual vdsty = (raw << 1) | ((vdstx & 1) ^ 1)

# ══════════════════════════════════════════════════════════════
# Operand info from XML
# ══════════════════════════════════════════════════════════════

from extra.assembly.amd.autogen.rdna3.operands import OPERANDS as OPERANDS_RDNA3
from extra.assembly.amd.autogen.rdna4.operands import OPERANDS as OPERANDS_RDNA4
from extra.assembly.amd.autogen.cdna.operands import OPERANDS as OPERANDS_CDNA
OPERANDS = {**OPERANDS_CDNA, **OPERANDS_RDNA3, **OPERANDS_RDNA4}

# ══════════════════════════════════════════════════════════════
# Inst base class
# ══════════════════════════════════════════════════════════════

class Inst:
  _fields: list[tuple[str, BitField]]
  _base_size: int

  def __init_subclass__(cls):
    # Collect fields from all parent classes, then override with this class's fields
    inherited = {}
    for base in reversed(cls.__mro__[1:]):
      if hasattr(base, '_fields'):
        inherited.update({name: field for name, field in base._fields})
    inherited.update({name: val for name, val in cls.__dict__.items() if isinstance(val, BitField)})
    cls._fields = list(inherited.items())
    cls._base_size = (max(f.hi for _, f in cls._fields) + 8) // 8

  def __init__(self, *args, **kwargs):
    self._raw = 0
    self._literal: int | None = kwargs.pop('literal', None)
    # Map positional args to field names (skip FixedBitFields)
    args_iter = iter(args)
    vals = {}
    for name, field in self._fields:
      if isinstance(field, FixedBitField): vals[name] = None
      elif name in kwargs: vals[name] = kwargs[name]
      else: vals[name] = next(args_iter, None)
    remaining = list(args_iter)
    assert not remaining, f"too many positional args: {remaining}"
    # Extract modifiers from Reg objects and merge into neg/abs/opsel
    neg_bits, abs_bits, opsel_bits = 0, 0, 0
    for name, bit in [('src0', 0), ('src1', 1), ('src2', 2)]:
      if name in vals and isinstance(vals[name], Reg):
        reg = vals[name]
        if reg.neg: neg_bits |= (1 << bit)
        if reg.abs_: abs_bits |= (1 << bit)
        if reg.hi: opsel_bits |= (1 << bit)
    if 'vdst' in vals and isinstance(vals['vdst'], Reg) and vals['vdst'].hi:
      opsel_bits |= (1 << 3)
    if neg_bits: vals['neg'] = (vals.get('neg') or 0) | neg_bits
    if abs_bits: vals['abs'] = (vals.get('abs') or 0) | abs_bits
    if opsel_bits: vals['opsel'] = (vals.get('opsel') or 0) | opsel_bits
    # Set all field values
    for name, field in self._fields:
      val = vals[name]
      self._raw = field.set(self._raw, val)
      # Capture literal for SrcFields that encoded to 255
      if isinstance(field, SrcField) and val is not None and field.encode(val) + field._valid_range[0] == 255 and self._literal is None:
        self._literal = _f32(val) if isinstance(val, float) else val & 0xFFFFFFFF
    # Validate register sizes against operand info (skip special registers like NULL, VCC, EXEC)
    for name, expected in self._get_field_sizes(vals).items():
      if (val := vals.get(name)) is None: continue
      if isinstance(val, Reg) and val.sz != expected and not (106 <= val.offset <= 127 or val.offset == 253):
        raise TypeError(f"{name} expects {expected} register(s), got {val.sz}")

  @property
  def op_name(self) -> str: return self.op.name
  @property
  def operands(self) -> dict: return OPERANDS.get(self.op, {}) if hasattr(self, 'op') else {}
  def _is_cdna(self) -> bool: return 'cdna' in type(self).__module__
  def _get_field_sizes(self, vals: dict) -> dict[str, int]:
    """Map field names to expected register sizes based on operand info."""
    sizes = {k: (v[1] + 31) // 32 for k, v in self.operands.items()}
    if not hasattr(self, 'op'): return sizes
    name = self.op_name.lower()
    # RDNA (WAVE32): condition masks and carry flags are 32-bit; CDNA (WAVE64) uses 64-bit
    if not self._is_cdna():
      if 'cndmask' in name and 'src2' in sizes: sizes['src2'] = 1
      if '_co_ci_' in name:
        if 'src2' in sizes: sizes['src2'] = 1
        if 'sdst' in sizes: sizes['sdst'] = 1
    # GLOBAL/FLAT: addr is 32-bit if saddr is valid SGPR, 64-bit if saddr is NULL
    # Check vals for saddr since some ops have the field but not in operand info
    if 'addr' in sizes and ('saddr' in sizes or 'saddr' in vals):
      saddr_val = vals.get('saddr')
      if isinstance(saddr_val, Reg): saddr_val = saddr_val.offset
      is_null_saddr = saddr_val in (None, 124, 125)  # 124=NULL, 125=M0
      sizes['addr'] = 2 if is_null_saddr else 1
      # saddr is 2 SGPRs when not NULL, otherwise skip validation (NULL is special single reg)
      if is_null_saddr: sizes.pop('saddr', None)
    # MUBUF/MTBUF: vaddr is variable (0-2 regs depending on idxen/offen), vdata depends on format
    if 'vaddr' in sizes: sizes.pop('vaddr')
    if 'vdata' in sizes: sizes.pop('vdata')
    # VOPC/VOP3 vdst for compares is wave-size dependent
    if 'vdst' in sizes and 'cmp' in name: sizes.pop('vdst')
    return sizes
  def _field_bits(self, name: str) -> int:
    """Get size in bits for a field from operand info."""
    return self.operands.get(name, (None, 0, None))[1]
  def is_src_64(self, n: int) -> bool:
    for name in (['src0', 'vsrc0', 'ssrc0'] if n == 0 else ['src1', 'vsrc1', 'ssrc1'] if n == 1 else ['src2']):
      if name in self.operands: return self.operands[name][1] == 64
    return False
  def is_src_16(self, n: int) -> bool:
    for name in (['src0', 'vsrc0', 'ssrc0'] if n == 0 else ['src1', 'vsrc1', 'ssrc1'] if n == 1 else ['src2']):
      if name in self.operands: return self.operands[name][1] == 16
    return False
  def is_dst_16(self) -> bool:
    for name in ['vdst', 'sdst', 'sdata']:
      if name in self.operands: return self.operands[name][1] == 16
    return False
  def dst_regs(self) -> int:
    for name in ['vdst', 'sdst', 'sdata']:
      if name in self.operands: return max(1, self.operands[name][1] // 32)
    return 1
  def data_regs(self) -> int:
    """Get data register count for memory ops (stores use 'data' field, loads use 'vdst')."""
    for name in ['data', 'vdata', 'data0']:
      if name in self.operands: return max(1, self.operands[name][1] // 32)
    return self.dst_regs()  # fallback to vdst for loads
  def src_regs(self, n: int) -> int:
    for name in (['src0', 'vsrc0', 'ssrc0'] if n == 0 else ['src1', 'vsrc1', 'ssrc1'] if n == 1 else ['src2']):
      if name in self._field_sizes: return self._field_sizes[name]
    return 1
  def num_srcs(self) -> int:
    """Get number of source operands from operand info."""
    ops = self.operands
    if 'src2' in ops: return 3
    if 'src1' in ops or 'vsrc1' in ops or 'ssrc1' in ops: return 2
    if 'src0' in ops or 'vsrc0' in ops or 'ssrc0' in ops: return 1
    return 0
  @classmethod
  def _size(cls) -> int: return cls._base_size
  def size(self) -> int: return self._base_size + (4 if self._literal is not None else 0)
  def disasm(self) -> str:
    from extra.assembly.amd.asm import disasm
    return disasm(self)

  def to_bytes(self) -> bytes:
    result = self._raw.to_bytes(self._base_size, 'little')
    if self._literal is not None:
      result += (self._literal & 0xFFFFFFFF).to_bytes(4, 'little')
    return result

  def has_literal(self) -> bool:
    """Check if instruction has a 32-bit literal constant."""
    for name, field in self._fields:
      if isinstance(field, SrcField) and getattr(self, name).offset == 255:
        return True
    # Check op, opx, opy for instructions that always have literals
    for attr in ('op', 'opx', 'opy'):
      if hasattr(self, attr) and any(x in getattr(self, attr).name for x in ('FMAMK', 'FMAAK', 'MADMK', 'MADAK', 'SETREG_IMM32')):
        return True
    return False

  @classmethod
  def from_bytes(cls, data: bytes):
    inst = object.__new__(cls)
    inst._raw = int.from_bytes(data[:cls._base_size], 'little')
    inst._literal = int.from_bytes(data[cls._base_size:cls._base_size + 4], 'little') if inst.has_literal() else None
    return inst

  def __eq__(self, other): return type(self) is type(other) and self._raw == other._raw and self._literal == other._literal
  def __hash__(self): return hash((type(self), self._raw, self._literal))
  @property
  def _field_sizes(self) -> dict[str, int]:
    """Get field sizes for repr - uses current field values."""
    vals = {name: getattr(self, name) for name, _ in self._fields}
    return self._get_field_sizes(vals)

  def __repr__(self):
    # collect (repr, is_default) pairs, strip trailing defaults so repr roundtrips with eval
    name, sizes = self.op.name.lower() if hasattr(self, 'op') else type(self).__name__, self._field_sizes
    def fmt(n, val):
      # resize regular registers to match type, but skip special registers (NULL, VCC, EXEC, etc)
      if isinstance(val, Reg) and (sz := sizes.get(n, 1)) > 1 and not (106 <= val.offset <= 127 or val.offset == 253):
        return repr(Reg(val.offset, sz, neg=val.neg, abs_=val.abs_, hi=val.hi))
      return repr(val)
    parts = [(fmt(n, v := getattr(self, n)), v == f.default) for n, f in self._fields if n != 'op' and not isinstance(f, FixedBitField)]
    while parts and parts[-1][1]: parts.pop()
    return f"{name}({', '.join(p[0] for p in parts)})"
