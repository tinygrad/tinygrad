# library for RDNA3 assembly DSL
from __future__ import annotations
import re
from enum import IntEnum

# *** bit field DSL: bits[31:30] == 0b10 ***
class BitField:
  def __init__(self, hi: int, lo: int, name: str | None = None): self.hi, self.lo, self.name = hi, lo, name
  def __set_name__(self, owner, name): self.name = name
  def __eq__(self, val: int) -> tuple[BitField, int]: return (self, val)  # type: ignore
  def mask(self) -> int: return (1 << (self.hi - self.lo + 1)) - 1
  def __get__(self, obj, objtype=None):
    if obj is None: return self
    val = unwrap(obj._values.get(self.name, 0))
    ann = getattr(type(obj), '__annotations__', {}).get(self.name)
    if ann and isinstance(ann, type) and issubclass(ann, IntEnum):
      try: return ann(val)
      except ValueError: pass
    return val
class _Bits:
  def __getitem__(self, key) -> BitField: return BitField(key.start, key.stop) if isinstance(key, slice) else BitField(key, key)
bits = _Bits()

# *** register types ***
class Reg:
  def __init__(self, idx: int, count: int = 1): self.idx, self.count = idx, count
  def __repr__(self): return f"{self.__class__.__name__.lower()[0]}[{self.idx}]" if self.count == 1 else f"{self.__class__.__name__.lower()[0]}[{self.idx}:{self.idx + self.count}]"
  @classmethod
  def __class_getitem__(cls, key): return cls(key.start, key.stop - key.start) if isinstance(key, slice) else cls(key)
class SGPR(Reg): pass
class VGPR(Reg): pass
class TTMP(Reg): pass
s, v = SGPR, VGPR

# *** field type markers ***
class SSrc: pass
class Src: pass
class Imm: pass
class SImm: pass
class RawImm:
  def __init__(self, val: int): self.val = val

def unwrap(val) -> int:
  return val.val if isinstance(val, RawImm) else val.value if hasattr(val, 'value') else val.idx if hasattr(val, 'idx') else val

# *** encoding ***
FLOAT_ENC = {0.5: 240, -0.5: 241, 1.0: 242, -1.0: 243, 2.0: 244, -2.0: 245, 4.0: 246, -4.0: 247}
SRC_FIELDS = {'src0', 'src1', 'src2', 'ssrc0', 'ssrc1', 'soffset'}
RAW_FIELDS = {'vdata', 'vdst', 'vaddr', 'addr', 'data', 'data0', 'data1', 'sdst', 'sdata'}

def encode_src(val) -> int:
  if isinstance(val, SGPR): return val.idx
  if isinstance(val, VGPR): return 256 + val.idx
  if isinstance(val, TTMP): return 108 + val.idx
  if hasattr(val, 'value'): return val.value
  if isinstance(val, float): return FLOAT_ENC.get(val, 255)
  return 128 + val if isinstance(val, int) and 0 <= val <= 64 else 192 + (-val) if isinstance(val, int) and -16 <= val <= -1 else 255

SPECIAL_DEC = {106: "vcc_lo", 107: "vcc_hi", 124: "null", 125: "m0", 126: "exec_lo", 127: "exec_hi", **{v: str(k) for k, v in FLOAT_ENC.items()}}
def decode_src(val: int) -> str:
  if val <= 105: return f"s{val}"
  if val in SPECIAL_DEC: return SPECIAL_DEC[val]
  if 108 <= val <= 123: return f"ttmp{val - 108}"
  if 128 <= val <= 192: return str(val - 128)
  if 193 <= val <= 208: return str(-(val - 192))
  if 256 <= val <= 511: return f"v{val - 256}"
  return "lit" if val == 255 else f"?{val}"

# *** instruction base class ***
class Inst:
  _fields: dict[str, BitField]
  _encoding: tuple[BitField, int] | None = None

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    cls._fields = {n: v[0] if isinstance(v, tuple) else v for n, v in cls.__dict__.items() if isinstance(v, BitField) or (isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], BitField))}
    if 'encoding' in cls._fields and isinstance(cls.__dict__.get('encoding'), tuple): cls._encoding = cls.__dict__['encoding']

  def __init__(self, *args, literal: int | None = None, **kwargs):
    self._values, self._literal = dict(zip([n for n in self._fields if n != 'encoding'], args)), literal
    self._values.update(kwargs)

  def _encode_field(self, name: str, val) -> int:
    if isinstance(val, RawImm): return val.val
    if name in {'srsrc', 'ssamp'}: return val.idx // 4 if isinstance(val, Reg) else val
    if name == 'sbase': return val.idx // 2 if isinstance(val, Reg) else val
    if name in RAW_FIELDS: return (108 + val.idx if isinstance(val, TTMP) else val.idx) if isinstance(val, Reg) else val
    if isinstance(val, Reg) or name in SRC_FIELDS: return encode_src(val)
    return val.value if hasattr(val, 'value') else val

  def to_int(self) -> int:
    word = (self._encoding[1] & self._encoding[0].mask()) << self._encoding[0].lo if self._encoding else 0
    for n, bf in self._fields.items():
      if n != 'encoding' and n in self._values: word |= (self._encode_field(n, self._values[n]) & bf.mask()) << bf.lo
    return word

  def _get_literal(self) -> int | None:
    from enum import IntEnum
    for n in SRC_FIELDS:
      if n in self._values and not isinstance(v := self._values[n], RawImm) and isinstance(v, int) and not isinstance(v, IntEnum) and not (0 <= v <= 64 or -16 <= v <= -1): return v
    return None

  def to_bytes(self) -> bytes:
    result = self.to_int().to_bytes(self._size(), 'little')
    return result + (lit & 0xffffffff).to_bytes(4, 'little') if (lit := self._get_literal() or getattr(self, '_literal', None)) else result

  @classmethod
  def _size(cls) -> int: return 4 if issubclass(cls, Inst32) else 8

  def size(self) -> int:
    """Size in bytes including literal if present."""
    return self._size() + (4 if self._literal is not None else 0)

  @classmethod
  def from_int(cls, word: int):
    inst = object.__new__(cls)
    inst._values = {n: RawImm(v) if n in SRC_FIELDS else v for n, bf in cls._fields.items() if n != 'encoding' for v in [(word >> bf.lo) & bf.mask()]}
    inst._literal = None
    return inst

  @classmethod
  def from_bytes(cls, data: bytes):
    inst = cls.from_int(int.from_bytes(data[:cls._size()], 'little'))
    op_val = inst._values.get('op', 0)
    has_literal = cls.__name__ == 'VOP2' and op_val in (44, 45, 55, 56)  # VOP2 FMAMK/FMAAK
    for n in SRC_FIELDS:
      if n in inst._values and isinstance(inst._values[n], RawImm) and inst._values[n].val == 255: has_literal = True
    if has_literal and len(data) >= cls._size() + 4: inst._literal = int.from_bytes(data[cls._size():cls._size()+4], 'little')
    return inst

  def __repr__(self): return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self._values.items())})"

  def disasm(self) -> str:
    op_val = unwrap(self._values.get('op', 0))
    try:
      from extra.assembly.rdna3 import autogen
      op_name = getattr(autogen, f"{self.__class__.__name__}Op")(op_val).name.lower() if hasattr(autogen, f"{self.__class__.__name__}Op") else f"op_{op_val}"
    except (ValueError, KeyError): op_name = f"op_{op_val}"
    def fmt(n, v):
      v = unwrap(v)
      if n in SRC_FIELDS: return f"0x{self._literal:x}" if v == 255 and getattr(self, '_literal', None) else decode_src(v) if v != 255 else "0xff"
      if n in ('sdst', 'vdst'): return f"{'s' if n == 'sdst' else 'v'}{v}"
      return f"v{v}" if n == 'vsrc1' else f"0x{v:x}" if n == 'simm16' else str(v)
    ops = [fmt(n, self._values.get(n, 0)) for n in self._fields if n not in ('encoding', 'op')]
    if self.__class__.__name__ == 'VOP2' and getattr(self, '_literal', None) and op_val in (44, 45, 55, 56):
      lit_str = f"0x{self._literal:x}"
      if op_val in (44, 55): ops.insert(2, lit_str)
      else: ops.append(lit_str)
    return f"{op_name} {', '.join(ops)}" if ops else op_name

class Inst32(Inst): pass
class Inst64(Inst):
  def to_bytes(self) -> bytes: return self.to_int().to_bytes(8, 'little')
  @classmethod
  def from_bytes(cls, data: bytes): return cls.from_int(int.from_bytes(data[:8], 'little'))

# *** waitcnt ***
def waitcnt(vmcnt: int = 0x7f, expcnt: int = 0x7, lgkmcnt: int = 0x3f) -> int:
  return (vmcnt & 0xf) | ((expcnt & 0x7) << 4) | (((vmcnt >> 4) & 0x7) << 7) | ((lgkmcnt & 0x3f) << 10)
def decode_waitcnt(val: int) -> tuple[int, int, int]:
  return (val & 0xf) | (((val >> 7) & 0x7) << 4), (val >> 4) & 0x7, (val >> 10) & 0x3f

# *** assembler ***
SPECIAL_REGS = {'vcc_lo': RawImm(106), 'vcc_hi': RawImm(107), 'null': RawImm(124), 'off': RawImm(124), 'm0': RawImm(125), 'exec_lo': RawImm(126), 'exec_hi': RawImm(127), 'scc': RawImm(253)}
FLOAT_CONSTS = {'0.5': 0.5, '-0.5': -0.5, '1.0': 1.0, '-1.0': -1.0, '2.0': 2.0, '-2.0': -2.0, '4.0': 4.0, '-4.0': -4.0}

def parse_operand(op: str) -> tuple:
  op = op.strip().lower()
  neg = op.startswith('-') and not op[1:2].isdigit(); op = op[1:] if neg else op
  abs_ = op.startswith('|') and op.endswith('|') or op.startswith('abs(') and op.endswith(')')
  op = op[1:-1] if op.startswith('|') else op[4:-1] if op.startswith('abs(') else op
  if op in FLOAT_CONSTS: return (FLOAT_CONSTS[op], neg, abs_)
  if re.match(r'^-?\d+$', op): return (int(op), neg, abs_)
  if m := re.match(r'^-?0x([0-9a-f]+)$', op):
    v = -int(m.group(1), 16) if op.startswith('-') else int(m.group(1), 16)
    return (RawImm(v) if 0 <= v <= 255 else v, neg, abs_)
  if op in SPECIAL_REGS: return (SPECIAL_REGS[op], neg, abs_)
  REG_MAP = {'s': SGPR, 'v': VGPR, 't': TTMP, 'ttmp': TTMP}
  if m := re.match(r'^([svt](?:tmp)?)\[(\d+):(\d+)\]$', op): return (REG_MAP[m.group(1)][int(m.group(2)):int(m.group(3))+1], neg, abs_)
  if m := re.match(r'^([svt](?:tmp)?)(\d+)$', op): return (REG_MAP[m.group(1)][int(m.group(2))], neg, abs_)
  raise ValueError(f"cannot parse operand: {op}")

def asm(text: str) -> Inst:
  from extra.assembly.rdna3 import autogen
  text = text.strip()
  clamp = 'clamp' in text.lower()
  if clamp: text = re.sub(r'\s+clamp\s*$', '', text, flags=re.I)
  parts = text.replace(',', ' ').split()
  if not parts: raise ValueError("empty instruction")
  mnemonic, op_str = parts[0].lower(), text[len(parts[0]):].strip()
  operands, current, depth, in_pipe = [], "", 0, False
  for ch in op_str:
    if ch == '[': depth += 1
    elif ch == ']': depth -= 1
    elif ch == '|': in_pipe = not in_pipe
    if ch == ',' and depth == 0 and not in_pipe: operands.append(current.strip()); current = ""
    else: current += ch
  if current.strip(): operands.append(current.strip())
  parsed = [parse_operand(op) for op in operands]
  values = [p[0] for p in parsed]
  neg_bits = sum((1 << (i-1)) for i, p in enumerate(parsed) if i > 0 and p[1])
  abs_bits = sum((1 << (i-1)) for i, p in enumerate(parsed) if i > 0 and p[2])
  lit = None
  if mnemonic in ('v_fmaak_f32', 'v_fmaak_f16') and len(values) == 4: lit, values = unwrap(values[3]), values[:3]
  elif mnemonic in ('v_fmamk_f32', 'v_fmamk_f16') and len(values) == 4: lit, values = unwrap(values[2]), [values[0], values[1], values[3]]
  for suffix in (['_e32', ''] if not (neg_bits or abs_bits or clamp) else ['', '_e32']):
    if hasattr(autogen, name := mnemonic.replace('.', '_') + suffix):
      inst = getattr(autogen, name)(*values, literal=lit)
      if neg_bits and 'neg' in inst._fields: inst._values['neg'] = neg_bits
      if abs_bits and 'abs' in inst._fields: inst._values['abs'] = abs_bits
      if clamp and 'clmp' in inst._fields: inst._values['clmp'] = 1
      return inst
  raise ValueError(f"unknown instruction: {mnemonic}")
