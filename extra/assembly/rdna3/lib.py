# library for RDNA3 assembly DSL
from __future__ import annotations
import re

# *** bit field DSL: bits[31:30] == 0b10 ***
class BitField:
  def __init__(self, hi: int, lo: int): self.hi, self.lo = hi, lo
  def __eq__(self, val: int) -> tuple[BitField, int]: return (self, val)  # type: ignore
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

# *** encoding ***
FLOAT_ENC = {0.5: 240, -0.5: 241, 1.0: 242, -1.0: 243, 2.0: 244, -2.0: 245, 4.0: 246, -4.0: 247}
SRC_FIELDS = {'src0', 'src1', 'src2', 'ssrc0', 'ssrc1', 'soffset'}

def encode_src(val) -> int:
  if isinstance(val, SGPR): return val.idx
  if isinstance(val, VGPR): return 256 + val.idx
  if isinstance(val, TTMP): return 108 + val.idx
  if hasattr(val, 'value'): return val.value
  if isinstance(val, float): return FLOAT_ENC.get(val, 255)
  if isinstance(val, int): return 128 + val if 0 <= val <= 64 else 192 + (-val) if -16 <= val <= -1 else 255
  raise ValueError(f"cannot encode source: {val}")

def decode_src(val: int) -> str:
  if val <= 105: return f"s{val}"
  if val in (106, 107): return ["vcc_lo", "vcc_hi"][val - 106]
  if 108 <= val <= 123: return f"ttmp{val - 108}"
  if 124 <= val <= 127: return ["null", "m0", "exec_lo", "exec_hi"][val - 124]
  if 128 <= val <= 192: return str(val - 128)
  if 193 <= val <= 208: return str(-(val - 192))
  if val in (FLOAT_DEC := {v: str(k) for k, v in FLOAT_ENC.items()}): return FLOAT_DEC[val]
  if 256 <= val <= 511: return f"v{val - 256}"
  return "lit" if val == 255 else f"?{val}"

# *** instruction base class ***
class Inst:
  _fields: dict[str, BitField]
  _encoding: tuple[BitField, int] | None = None

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    cls._fields = {}
    for name, val in list(cls.__dict__.items()):
      if isinstance(val, BitField): cls._fields[name] = val
      elif isinstance(val, tuple) and len(val) == 2 and isinstance(val[0], BitField):
        cls._fields[name] = val[0]
        if name == 'encoding': cls._encoding = val

  def __init__(self, *args, literal: int | None = None, **kwargs):
    self._values = dict(zip([n for n in self._fields if n != 'encoding'], args))
    self._values.update(kwargs)
    self._literal = literal

  def _encode_field(self, name: str, val) -> int:
    if isinstance(val, RawImm): return val.val
    if name in {'srsrc', 'ssamp'}: return val.idx // 4 if isinstance(val, Reg) else val
    if name == 'sbase': return val.idx // 2 if isinstance(val, Reg) else val
    if name in {'vdata', 'vaddr', 'addr', 'data', 'data0', 'data1'}: return val.idx if isinstance(val, Reg) else val
    # vdst can hold SGPR/TTMP for some instructions (e.g. v_readfirstlane_b32)
    if name == 'vdst': return (108 + val.idx if isinstance(val, TTMP) else val.idx) if isinstance(val, Reg) else val
    if name in {'sdst', 'sdata'}: return (108 + val.idx if isinstance(val, TTMP) else val.idx) if isinstance(val, Reg) else val
    if isinstance(val, Reg) or (name in SRC_FIELDS and isinstance(val, (int, float))): return encode_src(val)
    return val.value if hasattr(val, 'value') else val

  def to_int(self) -> int:
    word = 0
    if self._encoding:
      bf, val = self._encoding
      word |= (val & ((1 << (bf.hi - bf.lo + 1)) - 1)) << bf.lo
    for name, bf in self._fields.items():
      if name != 'encoding' and name in self._values:
        word |= (self._encode_field(name, self._values[name]) & ((1 << (bf.hi - bf.lo + 1)) - 1)) << bf.lo
    return word

  def _get_literal(self) -> int | None:
    from enum import IntEnum
    for name in SRC_FIELDS:
      if name in self._values and not isinstance(v := self._values[name], RawImm) and isinstance(v, int) and not isinstance(v, IntEnum) and not (0 <= v <= 64 or -16 <= v <= -1): return v
    return None

  def to_bytes(self) -> bytes:
    result = self.to_int().to_bytes(self._size(), 'little')
    lit = self._get_literal() or (self._literal if hasattr(self, '_literal') else None)
    return result + (lit & 0xffffffff).to_bytes(4, 'little') if lit is not None else result

  @classmethod
  def _size(cls) -> int: return 4 if issubclass(cls, Inst32) else 8
  @classmethod
  def from_int(cls, word: int, literal: int | None = None):
    inst = object.__new__(cls)
    inst._values, inst._literal = {}, literal
    for n, bf in cls._fields.items():
      if n == 'encoding': continue
      v = (word >> bf.lo) & ((1 << (bf.hi - bf.lo + 1)) - 1)
      inst._values[n] = RawImm(v) if n in SRC_FIELDS else v
    return inst
  @classmethod
  def from_bytes(cls, data: bytes):
    inst = cls.from_int(int.from_bytes(data[:cls._size()], 'little'))
    # check for literal (255 in any src field) and read the following dword
    for n in SRC_FIELDS:
      if n in inst._values and isinstance(inst._values[n], RawImm) and inst._values[n].val == 255 and len(data) >= cls._size() + 4:
        inst._literal = int.from_bytes(data[cls._size():cls._size()+4], 'little')
        break
    return inst
  def __repr__(self): return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self._values.items())})"

  def disasm(self) -> str:
    op_val = self._values.get('op', 0)
    op_val = op_val.val if isinstance(op_val, RawImm) else op_val
    try:
      from extra.assembly.rdna3 import autogen
      op_name = getattr(autogen, f"{self.__class__.__name__}Op")(op_val).name.lower() if hasattr(autogen, f"{self.__class__.__name__}Op") else f"op_{op_val}"
    except (ValueError, KeyError): op_name = f"op_{op_val}"
    operands = []
    for name in self._fields:
      if name not in ('encoding', 'op'):
        val = self._values.get(name, 0)
        val = val.val if isinstance(val, RawImm) else val
        # use literal value if this field has 255 (literal marker)
        if name in SRC_FIELDS and val == 255 and hasattr(self, '_literal') and self._literal is not None:
          operands.append(f"0x{self._literal:x}")
        else:
          operands.append(decode_src(val) if name in SRC_FIELDS else f"{'s' if name == 'sdst' else 'v'}{val}" if name in ('sdst', 'vdst') else f"v{val}" if name == 'vsrc1' else f"0x{val:x}" if name == 'simm16' else str(val))
    return f"{op_name} {', '.join(operands)}" if operands else op_name

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
REG_MAP = {'s': SGPR, 'v': VGPR, 't': TTMP, 'ttmp': TTMP}

def parse_operand(op: str) -> tuple:
  op = op.strip().lower()
  neg = op.startswith('-'); op = op[1:] if neg else op
  abs_ = (op.startswith('|') and op.endswith('|')) or (op.startswith('abs(') and op.endswith(')'))
  op = op[1:-1] if op.startswith('|') else op[4:-1] if op.startswith('abs(') else op
  if op in SPECIAL_REGS: return (SPECIAL_REGS[op], neg, abs_)
  if op in FLOAT_CONSTS: return (FLOAT_CONSTS[op], neg, abs_)
  if m := re.match(r'^([svt](?:tmp)?)\[(\d+):(\d+)\]$', op): return (REG_MAP[m.group(1)][int(m.group(2)):int(m.group(3))+1], neg, abs_)
  if m := re.match(r'^([svt](?:tmp)?)(\d+)$', op): return (REG_MAP[m.group(1)][int(m.group(2))], neg, abs_)
  if m := re.match(r'^0x([0-9a-f]+)$', op): return (int(m.group(1), 16), neg, abs_)
  if m := re.match(r'^-?\d+$', op): return (int(op), neg, abs_)
  raise ValueError(f"cannot parse operand: {op}")

def asm(text: str) -> Inst:
  from extra.assembly.rdna3 import autogen
  text = text.strip()
  clamp = 'clamp' in text.lower()
  if clamp: text = re.sub(r'\s+clamp\s*$', '', text, flags=re.I)
  parts = text.replace(',', ' ').split()
  if not parts: raise ValueError("empty instruction")
  mnemonic, op_str = parts[0].lower(), text[len(parts[0]):].strip()
  # split operands respecting brackets and pipes
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
  for suffix in ['', '_e32']:
    if hasattr(autogen, name := mnemonic.replace('.', '_') + suffix):
      inst = getattr(autogen, name)(*values)
      if neg_bits and 'neg' in inst._fields: inst._values['neg'] = neg_bits
      if abs_bits and 'abs' in inst._fields: inst._values['abs'] = abs_bits
      if clamp and 'clmp' in inst._fields: inst._values['clmp'] = 1
      return inst
  raise ValueError(f"unknown instruction: {mnemonic}")
