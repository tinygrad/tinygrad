# library for RDNA3 assembly DSL
from __future__ import annotations

# *** bit field DSL: bits[31:30] == 0b10 ***
class BitField:
  def __init__(self, hi: int, lo: int): self.hi, self.lo = hi, lo
  def __eq__(self, val: int) -> tuple[BitField, int]: return (self, val)  # type: ignore
class _Bits:
  def __getitem__(self, key) -> BitField: return BitField(key.start, key.stop) if isinstance(key, slice) else BitField(key, key)
bits = _Bits()

# *** register types: s[0], v[1:3], SGPR[4] ***
class Reg:
  def __init__(self, idx: int, count: int = 1): self.idx, self.count = idx, count
  def __repr__(self):
    n = self.__class__.__name__.lower()[0]
    return f"{n}[{self.idx}]" if self.count == 1 else f"{n}[{self.idx}:{self.idx + self.count}]"
  @classmethod
  def __class_getitem__(cls, key): return cls(key.start, key.stop - key.start) if isinstance(key, slice) else cls(key)

class SGPR(Reg): pass  # 0-105
class VGPR(Reg): pass  # encoded as 256-511
class TTMP(Reg): pass  # encoded as 108-123
s, v = SGPR, VGPR

# *** field type markers (for type annotations) ***
class SSrc: pass  # scalar source with full encoding
class Src: pass   # 9-bit source including VGPRs
class Imm: pass   # unsigned immediate
class SImm: pass  # signed immediate
class RawImm:     # bypass inline constant encoding
  def __init__(self, val: int): self.val = val

# *** source operand encoding ***
FLOAT_ENC = {0.5: 240, -0.5: 241, 1.0: 242, -1.0: 243, 2.0: 244, -2.0: 245, 4.0: 246, -4.0: 247}
FLOAT_DEC = {v: str(k) for k, v in FLOAT_ENC.items()}
SRC_FIELDS = {'src0', 'src1', 'src2', 'ssrc0', 'ssrc1'}

def encode_src(val) -> int:
  if isinstance(val, SGPR): return val.idx
  if isinstance(val, VGPR): return 256 + val.idx
  if isinstance(val, TTMP): return 108 + val.idx
  if hasattr(val, 'value'): return val.value
  if isinstance(val, float): return FLOAT_ENC.get(val, 255)
  if isinstance(val, int):
    if 0 <= val <= 64: return 128 + val
    if -16 <= val <= -1: return 192 + (-val)
    return 255  # literal marker
  raise ValueError(f"cannot encode source: {val}")

def decode_src(val: int) -> str:
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
  if val in FLOAT_DEC: return FLOAT_DEC[val]
  if val == 255: return "lit"
  if 256 <= val <= 511: return f"v{val - 256}"
  return f"?{val}"

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

  def __init__(self, *args, **kwargs):
    self._values = {}
    field_names = [n for n in self._fields if n != 'encoding']
    for i, val in enumerate(args):
      if i < len(field_names): self._values[field_names[i]] = val
    self._values.update(kwargs)

  def to_int(self) -> int:
    word = 0
    if self._encoding:
      bf, val = self._encoding
      word |= (val & ((1 << (bf.hi - bf.lo + 1)) - 1)) << bf.lo
    for name, bf in self._fields.items():
      if name == 'encoding' or name not in self._values: continue
      val = self._values[name]
      if isinstance(val, RawImm): val = val.val
      elif isinstance(val, Reg) or (name in SRC_FIELDS and isinstance(val, (int, float))): val = encode_src(val)
      elif hasattr(val, 'value'): val = val.value
      word |= (val & ((1 << (bf.hi - bf.lo + 1)) - 1)) << bf.lo
    return word

  def _get_literal(self) -> int | None:
    from enum import IntEnum
    for name in SRC_FIELDS:
      if name in self._values:
        val = self._values[name]
        if isinstance(val, RawImm): continue
        if isinstance(val, int) and not isinstance(val, IntEnum) and not (0 <= val <= 64 or -16 <= val <= -1):
          return val
    return None

  def to_bytes(self) -> bytes:
    result = self.to_int().to_bytes(self._size(), 'little')
    if (lit := self._get_literal()) is not None:
      result += (lit & 0xffffffff).to_bytes(4, 'little')
    return result

  @classmethod
  def _size(cls) -> int: return 4 if issubclass(cls, Inst32) else 8

  @classmethod
  def from_int(cls, word: int):
    inst = object.__new__(cls)
    inst._values = {}
    for name, bf in cls._fields.items():
      if name == 'encoding': continue
      val = (word >> bf.lo) & ((1 << (bf.hi - bf.lo + 1)) - 1)
      inst._values[name] = RawImm(val) if name in SRC_FIELDS else val
    return inst

  @classmethod
  def from_bytes(cls, data: bytes): return cls.from_int(int.from_bytes(data[:cls._size()], 'little'))

  def __repr__(self): return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self._values.items())})"

  def disasm(self) -> str:
    op_val = self._values.get('op', 0)
    if isinstance(op_val, RawImm): op_val = op_val.val
    op_name = f"op_{op_val}"
    try:
      from extra.assembly.rdna3 import autogen as enums
      if hasattr(enums, op_enum := f"{self.__class__.__name__}Op"):
        op_name = getattr(enums, op_enum)(op_val).name.lower()
    except (ValueError, KeyError): pass
    operands = []
    for name in self._fields:
      if name in ('encoding', 'op'): continue
      val = self._values.get(name, 0)
      if isinstance(val, RawImm): val = val.val
      if name in SRC_FIELDS: operands.append(decode_src(val))
      elif name in ('sdst', 'vdst'): operands.append(f"{'s' if name == 'sdst' else 'v'}{val}")
      elif name == 'vsrc1': operands.append(f"v{val}")
      elif name == 'simm16': operands.append(f"0x{val:x}")
      else: operands.append(str(val))
    return f"{op_name} {', '.join(operands)}" if operands else op_name

class Inst32(Inst): pass
class Inst64(Inst):
  def to_bytes(self) -> bytes: return self.to_int().to_bytes(8, 'little')
  @classmethod
  def from_bytes(cls, data: bytes): return cls.from_int(int.from_bytes(data[:8], 'little'))

# *** GFX11 s_waitcnt encoding: [3:0]=vmcnt[3:0], [6:4]=expcnt, [9:7]=vmcnt[6:4], [15:10]=lgkmcnt ***
def waitcnt(vmcnt: int = 0x7f, expcnt: int = 0x7, lgkmcnt: int = 0x3f) -> int:
  return (vmcnt & 0xf) | ((expcnt & 0x7) << 4) | (((vmcnt >> 4) & 0x7) << 7) | ((lgkmcnt & 0x3f) << 10)

def decode_waitcnt(val: int) -> tuple[int, int, int]:
  return (val & 0xf) | (((val >> 7) & 0x7) << 4), (val >> 4) & 0x7, (val >> 10) & 0x3f

# *** assembler: parse "v_add_f32_e32 v0, v1, v2" to instruction object ***
import re
SPECIAL_REGS = {'vcc_lo': RawImm(106), 'vcc_hi': RawImm(107), 'null': RawImm(124), 'off': RawImm(124), 'm0': RawImm(125), 'exec_lo': RawImm(126), 'exec_hi': RawImm(127), 'scc': RawImm(253)}
FLOAT_CONSTS = {'0.5': 0.5, '-0.5': -0.5, '1.0': 1.0, '-1.0': -1.0, '2.0': 2.0, '-2.0': -2.0, '4.0': 4.0, '-4.0': -4.0}

def parse_operand(op: str):
  """Parse an operand string to register/immediate value."""
  op = op.strip().lower()
  if op in SPECIAL_REGS: return SPECIAL_REGS[op]
  if op in FLOAT_CONSTS: return FLOAT_CONSTS[op]
  if m := re.match(r'^s\[(\d+):(\d+)\]$', op): return SGPR[int(m.group(1)):int(m.group(2))+1]
  if m := re.match(r'^s(\d+)$', op): return SGPR[int(m.group(1))]
  if m := re.match(r'^v\[(\d+):(\d+)\]$', op): return VGPR[int(m.group(1)):int(m.group(2))+1]
  if m := re.match(r'^v(\d+)$', op): return VGPR[int(m.group(1))]
  if m := re.match(r'^ttmp\[(\d+):(\d+)\]$', op): return TTMP[int(m.group(1)):int(m.group(2))+1]
  if m := re.match(r'^ttmp(\d+)$', op): return TTMP[int(m.group(1))]
  if m := re.match(r'^0x([0-9a-f]+)$', op): return int(m.group(1), 16)
  if m := re.match(r'^-?\d+$', op): return int(op)
  raise ValueError(f"cannot parse operand: {op}")

def asm(text: str) -> Inst:
  """Assemble a single instruction from text. Returns instruction object."""
  from extra.assembly.rdna3 import autogen
  text = text.strip()
  # split mnemonic and operands
  parts = text.replace(',', ' ').split()
  if not parts: raise ValueError("empty instruction")
  mnemonic, operands = parts[0].lower(), []
  if len(parts) > 1:
    # re-parse to handle brackets in register ranges
    op_str = text[len(parts[0]):].strip()
    current, depth = "", 0
    for ch in op_str:
      if ch == '[': depth += 1
      elif ch == ']': depth -= 1
      if ch == ',' and depth == 0:
        if current.strip(): operands.append(parse_operand(current))
        current = ""
      else: current += ch
    if current.strip(): operands.append(parse_operand(current))
  # find the instruction helper (e.g., v_add_f32_e32, s_mov_b32)
  for suffix in ['', '_e32']:
    name = mnemonic.replace('.', '_') + suffix
    if hasattr(autogen, name):
      helper = getattr(autogen, name)
      return helper(*operands)
  raise ValueError(f"unknown instruction: {mnemonic}")
