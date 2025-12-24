# library for RDNA3 assembly DSL
from __future__ import annotations
import functools

# *** bit field helpers ***

class BitField:
  def __init__(self, hi: int, lo: int):
    self.hi, self.lo = hi, lo
  def __eq__(self, val: int) -> tuple[BitField, int]: return (self, val)  # type: ignore # bits[31:30] == 0b10
  def __repr__(self): return f"bits[{self.hi}:{self.lo}]"

class _FieldFactory:
  def __getitem__(self, key) -> BitField:
    if isinstance(key, slice): return BitField(key.start, key.stop)
    return BitField(key, key)
bits = _FieldFactory()

# *** register types ***

class Reg:
  """Register operand with index and optional count for pairs/quads."""
  def __init__(self, idx: int, count: int = 1):
    self.idx, self.count = idx, count
  def __repr__(self):
    name = self.__class__.__name__.lower()
    return f"{name[0]}[{self.idx}]" if self.count == 1 else f"{name[0]}[{self.idx}:{self.idx + self.count}]"

class SGPR(Reg):
  """Scalar GPR - 106 available (0-105), encoded as 0-105."""
  def __class_getitem__(cls, key) -> SGPR:
    if isinstance(key, slice): return cls(key.start, key.stop - key.start)
    return cls(key)

class VGPR(Reg):
  """Vector GPR - 256 available (0-255), encoded as 256-511."""
  def __class_getitem__(cls, key) -> VGPR:
    if isinstance(key, slice): return cls(key.start, key.stop - key.start)
    return cls(key)

class TTMP(Reg):
  """Trap temporary registers - 16 available (0-15), encoded as 108-123."""
  def __class_getitem__(cls, key) -> TTMP:
    if isinstance(key, slice): return cls(key.start, key.stop - key.start)
    return cls(key)

# assembly-style aliases
s = SGPR
v = VGPR

# *** immediates ***

class Imm: pass
class SImm: pass

# *** instruction base classes ***

def _encode_field(value: int, hi: int, lo: int) -> int:
  """Place value into bit range [hi:lo]."""
  mask = (1 << (hi - lo + 1)) - 1
  return (value & mask) << lo

def _decode_field(word: int, hi: int, lo: int) -> int:
  """Extract bit range [hi:lo] from word."""
  mask = (1 << (hi - lo + 1)) - 1
  return (word >> lo) & mask

class Inst:
  """Base instruction class with encode/decode support."""
  _fields: dict[str, BitField]
  _encoding: tuple[BitField, int] | None = None

  def __init__(self, *args, **kwargs):
    self._values = {}
    field_names = [n for n in self._fields if n != 'encoding']
    for i, val in enumerate(args):
      if i < len(field_names): self._values[field_names[i]] = val
    self._values.update(kwargs)

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    cls._fields = {}
    for name, val in list(cls.__dict__.items()):
      if isinstance(val, BitField): cls._fields[name] = val
      elif isinstance(val, tuple) and len(val) == 2 and isinstance(val[0], BitField):
        cls._fields[name] = val[0]
        if name == 'encoding': cls._encoding = val

  def to_int(self) -> int:
    """Encode instruction to integer."""
    word = 0
    if self._encoding:
      bf, val = self._encoding
      word |= _encode_field(val, bf.hi, bf.lo)
    # fields that use source operand encoding (only when explicitly provided)
    src_fields = {'src0', 'src1', 'src2', 'ssrc0', 'ssrc1', 'sbase', 'soffset', 'saddr'}
    for name, bf in self._fields.items():
      if name == 'encoding': continue
      if name not in self._values:
        continue  # use implicit 0 for missing fields
      val = self._values[name]
      if isinstance(val, Reg) or (name in src_fields and isinstance(val, int)):
        val = encode_src(val)
      elif hasattr(val, 'value'): val = val.value
      word |= _encode_field(val, bf.hi, bf.lo)
    return word

  def _get_literal(self) -> int | None:
    """Get literal constant if any source needs it."""
    src_fields = {'src0', 'src1', 'src2', 'ssrc0', 'ssrc1'}
    for name in src_fields:
      if name in self._values:
        val = self._values[name]
        if isinstance(val, int) and not (0 <= val <= 64 or -16 <= val <= -1):
          return val
    return None

  def to_bytes(self) -> bytes:
    word = self.to_int()
    result = word.to_bytes(self._size(), 'little')
    lit = self._get_literal()
    if lit is not None:
      result += (lit & 0xffffffff).to_bytes(4, 'little')
    return result

  @classmethod
  def _size(cls) -> int:
    return 4 if issubclass(cls, Inst32) else 8

  @classmethod
  def from_int(cls, word: int):
    inst = object.__new__(cls)
    inst._values = {}
    for name, bf in cls._fields.items():
      if name == 'encoding': continue
      inst._values[name] = _decode_field(word, bf.hi, bf.lo)
    return inst

  @classmethod
  def from_bytes(cls, data: bytes):
    word = int.from_bytes(data[:cls._size()], 'little')
    return cls.from_int(word)

  def __repr__(self):
    parts = [f"{k}={v}" for k, v in self._values.items()]
    return f"{self.__class__.__name__}({', '.join(parts)})"

class Inst32(Inst):
  pass

class Inst64(Inst):
  def to_bytes(self) -> bytes:
    return self.to_int().to_bytes(8, 'little')

  @classmethod
  def from_bytes(cls, data: bytes):
    return cls.from_int(int.from_bytes(data[:8], 'little'))

# *** source operand encoding ***

def encode_src(val) -> int:
  """Encode a source operand to its 8/9-bit value."""
  if isinstance(val, SGPR): return val.idx
  if isinstance(val, VGPR): return 256 + val.idx
  if isinstance(val, TTMP): return 108 + val.idx
  if hasattr(val, 'value'): return val.value
  if isinstance(val, int):
    if 0 <= val <= 64: return 128 + val
    if -16 <= val <= -1: return 192 + (-val)
    return 255  # literal constant marker
  raise ValueError(f"cannot encode source: {val}")

def decode_src(val: int) -> str:
  """Decode a source encoding to string."""
  if val <= 105: return f"s{val}"
  if val == 106: return "vcc_lo"
  if val == 107: return "vcc_hi"
  if 108 <= val <= 123: return f"ttmp{val - 108}"
  if val == 124: return "null"
  if val == 125: return "m0"
  if val == 126: return "exec_lo"
  if val == 127: return "exec_hi"
  if 128 <= val <= 192: return str(val - 128)
  if 193 <= val <= 208: return str(-(val - 192))
  if val == 240: return "0.5"
  if val == 241: return "-0.5"
  if val == 242: return "1.0"
  if val == 243: return "-1.0"
  if val == 255: return "lit"
  if 256 <= val <= 511: return f"v{val - 256}"
  return f"?{val}"
