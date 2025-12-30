# library for RDNA3 assembly DSL
# mypy: ignore-errors
from __future__ import annotations
from enum import IntEnum
from typing import overload, Annotated, TypeVar, Generic

# Bit field DSL
class BitField:
  def __init__(self, hi: int, lo: int, name: str | None = None): self.hi, self.lo, self.name = hi, lo, name
  def __set_name__(self, owner, name): self.name, self._owner = name, owner
  def __eq__(self, val: int) -> tuple[BitField, int]: return (self, val)  # type: ignore
  def mask(self) -> int: return (1 << (self.hi - self.lo + 1)) - 1
  @property
  def marker(self) -> type | None:
    # Get marker from Annotated type hint if present
    import typing
    if hasattr(self, '_owner') and self.name:
      hints = typing.get_type_hints(self._owner, include_extras=True)
      if self.name in hints:
        hint = hints[self.name]
        if typing.get_origin(hint) is Annotated:
          args = typing.get_args(hint)
          return args[1] if len(args) > 1 else None
    return None
  @overload
  def __get__(self, obj: None, objtype: type) -> BitField: ...
  @overload
  def __get__(self, obj: object, objtype: type | None = None) -> int: ...
  def __get__(self, obj, objtype=None):
    if obj is None: return self
    val = unwrap(obj._values.get(self.name, 0))
    # Convert to IntEnum if marker is an IntEnum subclass
    if self.marker and isinstance(self.marker, type) and issubclass(self.marker, IntEnum):
      try: return self.marker(val)
      except ValueError: pass
    return val

class _Bits:
  def __getitem__(self, key) -> BitField: return BitField(key.start, key.stop) if isinstance(key, slice) else BitField(key, key)
bits = _Bits()

# Source operand with modifiers - base class for anything that can be a src with neg/abs
class SrcMod:
  __slots__ = ('val', 'neg', 'abs_')
  def __init__(self, val: int, neg: bool = False, abs_: bool = False): self.val, self.neg, self.abs_ = val, neg, abs_
  def __repr__(self): return f"{'-' if self.neg else ''}{'|' if self.abs_ else ''}{self.val}{'|' if self.abs_ else ''}"
  def __neg__(self): return SrcMod(self.val, not self.neg, self.abs_)
  def __abs__(self): return SrcMod(self.val, self.neg, True)

# Register types
class Reg(SrcMod):
  __slots__ = ('idx', 'count', 'hi')
  def __init__(self, idx: int, count: int = 1, hi: bool = False, neg: bool = False, abs_: bool = False):
    self.idx, self.count, self.hi = idx, count, hi
    super().__init__(idx, neg, abs_)
  def __repr__(self): return f"{self.__class__.__name__.lower()[0]}[{self.idx}]" if self.count == 1 else f"{self.__class__.__name__.lower()[0]}[{self.idx}:{self.idx + self.count}]"
  def __neg__(self): return self.__class__(self.idx, self.count, self.hi, not self.neg, self.abs_)
  def __abs__(self): return self.__class__(self.idx, self.count, self.hi, self.neg, True)
  @property
  def l(self): return self.__class__(self.idx, self.count, False, self.neg, self.abs_)
  @property
  def h(self): return self.__class__(self.idx, self.count, True, self.neg, self.abs_)

T = TypeVar('T', bound=Reg)
class _RegFactory(Generic[T]):
  def __init__(self, cls: type[T], name: str): self._cls, self._name = cls, name
  @overload
  def __getitem__(self, key: int) -> Reg: ...
  @overload
  def __getitem__(self, key: slice) -> Reg: ...
  def __getitem__(self, key: int | slice) -> Reg:
    return self._cls(key.start, key.stop - key.start + 1) if isinstance(key, slice) else self._cls(key)
  def __repr__(self): return f"<{self._name} factory>"

class SGPR(Reg): pass
class VGPR(Reg): pass
class TTMP(Reg): pass
s: _RegFactory[SGPR] = _RegFactory(SGPR, "SGPR")
v: _RegFactory[VGPR] = _RegFactory(VGPR, "VGPR")
ttmp: _RegFactory[TTMP] = _RegFactory(TTMP, "TTMP")

# Special registers as SrcMod objects (support -VCC_LO, abs(EXEC_LO), etc.)
VCC_LO, VCC_HI, VCC = SrcMod(106), SrcMod(107), SrcMod(106)
EXEC_LO, EXEC_HI, EXEC = SrcMod(126), SrcMod(127), SrcMod(126)
SCC, M0, NULL, OFF = SrcMod(253), SrcMod(125), SrcMod(124), SrcMod(124)

# Field type markers (runtime classes for validation)
class _SSrc: pass
class _Src: pass
class _Imm: pass
class _SImm: pass
class _VDSTYEnc: pass  # VOPD vdsty: encoded = actual >> 1, actual = (encoded << 1) | ((vdstx & 1) ^ 1)
class _SGPRField: pass
class _VGPRField: pass

# Type aliases for annotations - tells mypy it's a BitField while preserving marker info
SSrc = Annotated[BitField, _SSrc]
Src = Annotated[BitField, _Src]
Imm = Annotated[BitField, _Imm]
SImm = Annotated[BitField, _SImm]
VDSTYEnc = Annotated[BitField, _VDSTYEnc]
SGPRField = Annotated[BitField, _SGPRField]
VGPRField = Annotated[BitField, _VGPRField]
class RawImm:
  def __init__(self, val: int): self.val = val
  def __repr__(self): return f"RawImm({self.val})"
  def __eq__(self, other): return isinstance(other, RawImm) and self.val == other.val

def unwrap(val) -> int:
  if isinstance(val, RawImm): return val.val
  if isinstance(val, SrcMod) and not isinstance(val, Reg): return val.val  # Special registers like VCC_LO, NULL
  if hasattr(val, 'value'): return val.value  # IntEnum
  if hasattr(val, 'idx'): return val.idx  # Reg
  return val

# Encoding helpers
FLOAT_ENC = {0.5: 240, -0.5: 241, 1.0: 242, -1.0: 243, 2.0: 244, -2.0: 245, 4.0: 246, -4.0: 247}
SRC_FIELDS = {'src0', 'src1', 'src2', 'ssrc0', 'ssrc1', 'soffset', 'srcx0', 'srcy0'}
RAW_FIELDS = {'vdata', 'vdst', 'vaddr', 'addr', 'data', 'data0', 'data1', 'sdst', 'sdata', 'vsrc1'}

def _encode_reg(val: Reg) -> int:
  if isinstance(val, TTMP): return 108 + val.idx
  return val.idx  # hi bit is handled via opsel, not in register encoding

def encode_src(val) -> int:
  if isinstance(val, VGPR): return 256 + _encode_reg(val)
  if isinstance(val, Reg): return _encode_reg(val)
  if isinstance(val, SrcMod) and not isinstance(val, Reg):
    # SrcMod wraps either special registers (VCC_LO=106, EXEC_LO=126, etc.) or literals
    # Special register values are in valid encoding ranges - return as-is
    # Literals (large integers) need 255 marker
    v = val.val
    # Valid source encoding ranges: 0-127 (SGPRs/special), 128-192 (inline const), 193-208 (neg inline), 240-247 (float), 251-253 (special)
    if 0 <= v <= 127 or 240 <= v <= 255: return v  # SGPRs, special regs, float constants
    if 128 <= v <= 192: return v  # Inline positive constants (0-64)
    if 193 <= v <= 208: return v  # Inline negative constants (-1 to -16)
    return 255  # Literal marker - value stored separately
  if hasattr(val, 'value'): return val.value  # IntEnum
  if isinstance(val, float): return 128 if val == 0.0 else FLOAT_ENC.get(val, 255)
  return 128 + val if isinstance(val, int) and 0 <= val <= 64 else 192 + (-val) if isinstance(val, int) and -16 <= val <= -1 else 255

# Instruction base class
class Inst:
  _fields: dict[str, BitField]
  _encoding: tuple[BitField, int] | None = None
  _defaults: dict[str, int] = {}
  _values: dict[str, int | RawImm]
  _words: int  # size in 32-bit words, set by decode_program
  _literal: int | None

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    cls._fields = {n: v[0] if isinstance(v, tuple) else v for n, v in cls.__dict__.items() if isinstance(v, BitField) or (isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], BitField))}
    if 'encoding' in cls._fields and isinstance(cls.__dict__.get('encoding'), tuple): cls._encoding = cls.__dict__['encoding']

  def __init__(self, *args, literal: int | None = None, **kwargs):
    self._values, self._literal = dict(self._defaults), literal
    # Map positional args to field names
    field_names = [n for n in self._fields if n != 'encoding']
    orig_args = dict(zip(field_names, args))
    orig_args.update(kwargs)
    self._values.update(orig_args)
    # Validate register counts for SMEM instructions (before encoding)
    if self.__class__.__name__ == 'SMEM':
      op_val = orig_args.get(field_names[0]) if args else orig_args.get('op')
      if op_val is not None:
        if hasattr(op_val, 'value'): op_val = op_val.value
        expected_cnt = {0:1, 1:2, 2:4, 3:8, 4:16, 8:1, 9:2, 10:4, 11:8, 12:16}.get(op_val)
        sdata_val = orig_args.get('sdata')
        if expected_cnt is not None and isinstance(sdata_val, Reg) and sdata_val.count != expected_cnt:
          raise ValueError(f"SMEM op {op_val} expects {expected_cnt} registers, got {sdata_val.count}")
    # Validate register counts for SOP1 instructions (b32 = 1 reg, b64 = 2 regs)
    if self.__class__.__name__ == 'SOP1':
      op_val = orig_args.get(field_names[0]) if args else orig_args.get('op')
      if op_val is not None and hasattr(op_val, 'name'):
        expected = 2 if op_val.name.endswith('_B64') else 1
        sdst_val, ssrc0_val = orig_args.get('sdst'), orig_args.get('ssrc0')
        if isinstance(sdst_val, Reg) and sdst_val.count != expected:
          raise ValueError(f"SOP1 {op_val.name} expects {expected} destination register(s), got {sdst_val.count}")
        if isinstance(ssrc0_val, Reg) and ssrc0_val.count != expected:
          raise ValueError(f"SOP1 {op_val.name} expects {expected} source register(s), got {ssrc0_val.count}")
    # Type check and encode values
    for name, val in list(self._values.items()):
      if name == 'encoding': continue
      # For RawImm, only process RAW_FIELDS to unwrap to int
      if isinstance(val, RawImm):
        if name in RAW_FIELDS: self._values[name] = val.val
        continue
      field = self._fields.get(name)
      marker = field.marker if field else None
      # Type validation
      if marker is _SGPRField:
        if isinstance(val, VGPR): raise TypeError(f"field '{name}' requires SGPR, got VGPR")
        if not isinstance(val, (SGPR, TTMP, SrcMod, int, RawImm)): raise TypeError(f"field '{name}' requires SGPR, got {type(val).__name__}")
      if marker is _VGPRField:
        if not isinstance(val, VGPR): raise TypeError(f"field '{name}' requires VGPR, got {type(val).__name__}")
      if marker is _SSrc and isinstance(val, VGPR): raise TypeError(f"field '{name}' requires scalar source, got VGPR")
      # Encode source fields as RawImm for consistent disassembly
      if name in SRC_FIELDS:
        encoded = encode_src(val)
        # For VOP1/VOP2/VOPC (no opsel field), encode hi bit in src value
        if isinstance(val, Reg) and val.hi and 'opsel' not in self._fields:
          encoded |= 0x80
        self._values[name] = RawImm(encoded)
        # Handle neg/abs/opsel modifiers for VOP3 instructions
        if isinstance(val, SrcMod):
          if val.neg and 'neg' in self._fields:
            neg_bit = {'src0': 1, 'src1': 2, 'src2': 4}.get(name, 0)
            cur_neg = self._values.get('neg', 0)
            self._values['neg'] = (cur_neg.val if isinstance(cur_neg, RawImm) else cur_neg) | neg_bit
          if val.abs_ and 'abs' in self._fields:
            abs_bit = {'src0': 1, 'src1': 2, 'src2': 4}.get(name, 0)
            cur_abs = self._values.get('abs', 0)
            self._values['abs'] = (cur_abs.val if isinstance(cur_abs, RawImm) else cur_abs) | abs_bit
        # Handle hi (opsel) for 16-bit ops - only for formats with opsel field
        if isinstance(val, Reg) and val.hi and 'opsel' in self._fields:
          opsel_bit = {'src0': 1, 'src1': 2, 'src2': 4}.get(name, 0)
          cur_opsel = self._values.get('opsel', 0)
          self._values['opsel'] = (cur_opsel.val if isinstance(cur_opsel, RawImm) else cur_opsel) | opsel_bit
        # Track literal value if needed (encoded as 255)
        # For 64-bit ops, store literal in high 32 bits (to match from_bytes decoding and to_bytes encoding)
        if encoded == 255 and self._literal is None:
          if isinstance(val, SrcMod) and not isinstance(val, Reg):
            # SrcMod wrapping a literal value
            self._literal = (val.val << 32) if self._is_64bit_op() else val.val
          elif isinstance(val, int) and not isinstance(val, IntEnum):
            self._literal = (val << 32) if self._is_64bit_op() else val
          elif isinstance(val, float):
            import struct
            lit32 = struct.unpack('<I', struct.pack('<f', val))[0]
            self._literal = (lit32 << 32) if self._is_64bit_op() else lit32
      # Encode raw register fields for consistent repr
      elif name in RAW_FIELDS:
        if isinstance(val, Reg):
          encoded = _encode_reg(val)
          # For VOP1/VOP2/VOPC (no opsel field), encode hi bit in register value
          if val.hi and 'opsel' not in self._fields:
            encoded |= 0x80
          self._values[name] = encoded
          # Handle vdst hi (opsel bit 3) for 16-bit ops - only for formats with opsel field
          if name == 'vdst' and val.hi and 'opsel' in self._fields:
            cur_opsel = self._values.get('opsel', 0)
            self._values['opsel'] = (cur_opsel.val if isinstance(cur_opsel, RawImm) else cur_opsel) | 8
        elif hasattr(val, 'value'): self._values[name] = val.value  # IntEnum like SrcEnum.NULL
      # Encode sbase (divided by 2) and srsrc/ssamp (divided by 4)
      elif name == 'sbase':
        if isinstance(val, Reg): self._values[name] = val.idx // 2
        elif isinstance(val, SrcMod): self._values[name] = val.val // 2  # Special regs like VCC_LO
      elif name in {'srsrc', 'ssamp'} and isinstance(val, Reg):
        self._values[name] = val.idx // 4
      # VOPD vdsty: encode as actual >> 1 (constraint: vdsty parity must be opposite of vdstx)
      elif marker is _VDSTYEnc and isinstance(val, VGPR):
        self._values[name] = val.idx >> 1

  def _encode_field(self, name: str, val) -> int:
    if isinstance(val, RawImm): return val.val
    if isinstance(val, SrcMod) and not isinstance(val, Reg): return val.val  # Special regs like VCC_LO
    if name in {'srsrc', 'ssamp'}: return val.idx // 4 if isinstance(val, Reg) else val
    if name == 'sbase': return val.idx // 2 if isinstance(val, Reg) else val.val // 2 if isinstance(val, SrcMod) else val
    if name in RAW_FIELDS: return _encode_reg(val) if isinstance(val, Reg) else val
    if isinstance(val, Reg) or name in SRC_FIELDS: return encode_src(val)
    return val.value if hasattr(val, 'value') else val

  def to_int(self) -> int:
    word = (self._encoding[1] & self._encoding[0].mask()) << self._encoding[0].lo if self._encoding else 0
    for n, bf in self._fields.items():
      if n != 'encoding' and n in self._values: word |= (self._encode_field(n, self._values[n]) & bf.mask()) << bf.lo
    return word

  def _get_literal(self) -> int | None:
    for n in SRC_FIELDS:
      if n in self._values and not isinstance(v := self._values[n], RawImm) and isinstance(v, int) and not isinstance(v, IntEnum) and not (0 <= v <= 64 or -16 <= v <= -1): return v
    return None

  def _is_64bit_op(self) -> bool:
    """Check if this instruction uses 64-bit operands (and thus 64-bit literals).
    Exception: V_LDEXP_F64 has 32-bit integer src1, so its literal is 32-bit."""
    op = self._values.get('op')
    if op is None: return False
    # op may be an enum (from __init__) or an int (from from_int)
    op_name = op.name if hasattr(op, 'name') else None
    if op_name is None and self.__class__.__name__ == 'VOP3':
      from extra.assembly.amd.autogen.rdna3 import VOP3Op
      try: op_name = VOP3Op(op).name
      except ValueError: pass
    if op_name is None and self.__class__.__name__ == 'VOPC':
      from extra.assembly.amd.autogen.rdna3 import VOPCOp
      try: op_name = VOPCOp(op).name
      except ValueError: pass
    if op_name is None: return False
    # V_LDEXP_F64 has 32-bit integer exponent in src1, so literal is 32-bit
    if op_name == 'V_LDEXP_F64': return False
    return op_name.endswith(('_F64', '_B64', '_I64', '_U64'))

  def to_bytes(self) -> bytes:
    result = self.to_int().to_bytes(self._size(), 'little')
    lit = self._get_literal() or getattr(self, '_literal', None)
    if lit is None: return result
    # For 64-bit ops, literal is stored in high 32 bits internally, but encoded as 4 bytes
    lit32 = (lit >> 32) if self._is_64bit_op() else lit
    return result + (lit32 & 0xffffffff).to_bytes(4, 'little')

  @classmethod
  def _size(cls) -> int: return 4 if issubclass(cls, Inst32) else 8
  def size(self) -> int:
    # Literal is always 4 bytes in the binary (for 64-bit ops, it's in high 32 bits)
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
    has_literal = cls.__name__ == 'VOP2' and op_val in (44, 45, 55, 56)
    has_literal = has_literal or (cls.__name__ == 'SOP2' and op_val in (69, 70))
    # VOPD fmaak/fmamk always have a literal (opx/opy value 1 or 2)
    opx, opy = inst._values.get('opx', 0), inst._values.get('opy', 0)
    has_literal = has_literal or (cls.__name__ == 'VOPD' and (opx in (1, 2) or opy in (1, 2)))
    for n in SRC_FIELDS:
      if n in inst._values and isinstance(inst._values[n], RawImm) and inst._values[n].val == 255: has_literal = True
    if has_literal:
      # For 64-bit ops, the literal is 32 bits placed in the HIGH 32 bits of the 64-bit value
      # (low 32 bits are zero). This is how AMD hardware interprets 32-bit literals for 64-bit ops.
      if len(data) >= cls._size() + 4:
        lit32 = int.from_bytes(data[cls._size():cls._size()+4], 'little')
        inst._literal = (lit32 << 32) if inst._is_64bit_op() else lit32
    return inst

  def __repr__(self):
    # Use _fields order and exclude fields that are 0/default (for consistent repr after roundtrip)
    def is_zero(v): return (isinstance(v, int) and v == 0) or (isinstance(v, VGPR) and v.idx == 0 and v.count == 1)
    items = [(k, self._values[k]) for k in self._fields if k in self._values and k != 'encoding'
             and not (is_zero(self._values[k]) and k not in {'op'})]
    lit = f", literal={hex(self._literal)}" if self._literal is not None else ""
    return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in items)}{lit})"

  def __eq__(self, other):
    if not isinstance(other, Inst): return NotImplemented
    return self.__class__ == other.__class__ and self._values == other._values and self._literal == other._literal

  def __hash__(self): return hash((self.__class__.__name__, tuple(sorted((k, repr(v)) for k, v in self._values.items())), self._literal))

  def disasm(self) -> str:
    from extra.assembly.amd.asm import disasm
    return disasm(self)

class Inst32(Inst): pass
class Inst64(Inst): pass

# ═══════════════════════════════════════════════════════════════════════════════
# CODE GENERATION: generates autogen/__init__.py by parsing AMD ISA PDFs
# Supports both RDNA3.5 and CDNA4 instruction set PDFs - auto-detects format
# ═══════════════════════════════════════════════════════════════════════════════

PDF_URLS = {
  "rdna3": "https://docs.amd.com/api/khub/documents/UVVZM22UN7tMUeiW_4ShTQ/content",  # RDNA3.5
  "rdna4": "https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content",
  "cdna": ["https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf",
           "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf"],
}
FIELD_TYPES = {'SSRC0': 'SSrc', 'SSRC1': 'SSrc', 'SOFFSET': 'SSrc', 'SADDR': 'SSrc', 'SRC0': 'Src', 'SRC1': 'Src', 'SRC2': 'Src',
  'SDST': 'SGPRField', 'SBASE': 'SGPRField', 'SDATA': 'SGPRField', 'SRSRC': 'SGPRField', 'VDST': 'VGPRField', 'VSRC1': 'VGPRField', 'VDATA': 'VGPRField',
  'VADDR': 'VGPRField', 'ADDR': 'VGPRField', 'DATA': 'VGPRField', 'DATA0': 'VGPRField', 'DATA1': 'VGPRField', 'SIMM16': 'SImm', 'OFFSET': 'Imm',
  'OPX': 'VOPDOp', 'OPY': 'VOPDOp', 'SRCX0': 'Src', 'SRCY0': 'Src', 'VSRCX1': 'VGPRField', 'VSRCY1': 'VGPRField', 'VDSTX': 'VGPRField', 'VDSTY': 'VDSTYEnc'}
FIELD_ORDER = {
  'SOP2': ['op', 'sdst', 'ssrc0', 'ssrc1'], 'SOP1': ['op', 'sdst', 'ssrc0'], 'SOPC': ['op', 'ssrc0', 'ssrc1'],
  'SOPK': ['op', 'sdst', 'simm16'], 'SOPP': ['op', 'simm16'], 'VOP1': ['op', 'vdst', 'src0'], 'VOPC': ['op', 'src0', 'vsrc1'],
  'VOP2': ['op', 'vdst', 'src0', 'vsrc1'], 'VOP3SD': ['op', 'vdst', 'sdst', 'src0', 'src1', 'src2', 'clmp'],
  'SMEM': ['op', 'sdata', 'sbase', 'soffset', 'offset', 'glc', 'dlc'], 'DS': ['op', 'vdst', 'addr', 'data0', 'data1'],
  'VOP3': ['op', 'vdst', 'src0', 'src1', 'src2', 'omod', 'neg', 'abs', 'clmp', 'opsel'],
  'VOP3P': ['op', 'vdst', 'src0', 'src1', 'src2', 'neg', 'neg_hi', 'opsel', 'opsel_hi', 'clmp'],
  'FLAT': ['op', 'vdst', 'addr', 'data', 'saddr', 'offset', 'seg', 'dlc', 'glc', 'slc'],
  'MUBUF': ['op', 'vdata', 'vaddr', 'srsrc', 'soffset', 'offset', 'offen', 'idxen', 'glc', 'dlc', 'slc', 'tfe'],
  'MTBUF': ['op', 'vdata', 'vaddr', 'srsrc', 'soffset', 'offset', 'format', 'offen', 'idxen', 'glc', 'dlc', 'slc', 'tfe'],
  'MIMG': ['op', 'vdata', 'vaddr', 'srsrc', 'ssamp', 'dmask', 'dim', 'unrm', 'dlc', 'glc', 'slc'],
  'EXP': ['en', 'target', 'vsrc0', 'vsrc1', 'vsrc2', 'vsrc3', 'done', 'row'],
  'VINTERP': ['op', 'vdst', 'src0', 'src1', 'src2', 'waitexp', 'clmp', 'opsel', 'neg'],
  'VOPD': ['opx', 'opy', 'vdstx', 'vdsty', 'srcx0', 'vsrcx1', 'srcy0', 'vsrcy1'],
  'LDSDIR': ['op', 'vdst', 'attr', 'attr_chan', 'wait_va']}
SRC_EXTRAS = {233: 'DPP8', 234: 'DPP8FI', 250: 'DPP16', 251: 'VCCZ', 252: 'EXECZ', 254: 'LDS_DIRECT'}
FLOAT_MAP = {'0.5': 'POS_HALF', '-0.5': 'NEG_HALF', '1.0': 'POS_ONE', '-1.0': 'NEG_ONE', '2.0': 'POS_TWO', '-2.0': 'NEG_TWO',
  '4.0': 'POS_FOUR', '-4.0': 'NEG_FOUR', '1/(2*PI)': 'INV_2PI', '0': 'ZERO'}

def _parse_bits(s: str) -> tuple[int, int] | None:
  import re
  return (int(m.group(1)), int(m.group(2) or m.group(1))) if (m := re.match(r'\[(\d+)(?::(\d+))?\]', s)) else None

def _parse_fields_table(table: list, fmt: str, enums: set[str]) -> list[tuple]:
  import re
  fields = []
  for row in table[1:]:
    if not row or not row[0]: continue
    name, bits_str = row[0].split('\n')[0].strip(), (row[1] or '').split('\n')[0].strip()
    if not (bits := _parse_bits(bits_str)): continue
    enc_val, hi, lo = None, bits[0], bits[1]
    if name == 'ENCODING' and row[2]:
      # Handle both RDNA3 ('bXX) and CDNA4 (Must be: XX) encoding formats
      if m := re.search(r"(?:'b|Must be:\s*)([01_]+)", row[2]):
        enc_bits = m.group(1).replace('_', '')
        enc_val = int(enc_bits, 2)
        declared_width, actual_width = hi - lo + 1, len(enc_bits)
        if actual_width > declared_width: lo = hi - actual_width + 1
    ftype = f"{fmt}Op" if name == 'OP' and f"{fmt}Op" in enums else FIELD_TYPES.get(name.upper())
    fields.append((name, hi, lo, enc_val, ftype))
  return fields

def _parse_single_pdf(url: str) -> dict:
  """Parse a single PDF and return raw data (formats, enums, src_enum, doc_name, is_cdna)."""
  import re, pdfplumber
  from tinygrad.helpers import fetch

  pdf = pdfplumber.open(fetch(url))

  # Auto-detect document type from first page
  first_page_text = pdf.pages[0].extract_text() or ''
  is_cdna4 = 'CDNA4' in first_page_text or 'CDNA 4' in first_page_text
  is_cdna3 = 'CDNA3' in first_page_text or 'CDNA 3' in first_page_text or 'MI300' in first_page_text
  is_cdna = is_cdna3 or is_cdna4
  is_rdna4 = 'RDNA4' in first_page_text or 'RDNA 4' in first_page_text
  is_rdna35 = 'RDNA3.5' in first_page_text or 'RDNA 3.5' in first_page_text  # Check 3.5 before 3
  is_rdna3 = not is_rdna35 and ('RDNA3' in first_page_text or 'RDNA 3' in first_page_text)
  doc_name = "CDNA4" if is_cdna4 else "CDNA3" if is_cdna3 else "RDNA4" if is_rdna4 else "RDNA3.5" if is_rdna35 else "RDNA3" if is_rdna3 else "Unknown"

  # Find the "Microcode Formats" section - search for SOP2 format definition
  microcode_start = None
  total_pages = len(pdf.pages)
  # Search from likely locations (formats are typically 20-95% through the document - RDNA3 has them at ~25%)
  for i in range(int(total_pages * 0.2), total_pages):
    text = pdf.pages[i].extract_text() or ''
    # Look for "X.Y.Z. SOP2" section header or "Chapter X. Microcode Formats"
    if re.search(r'\d+\.\d+\.\d+\.\s+SOP2\b', text) or re.search(r'Chapter \d+\.\s+Microcode Formats', text):
      microcode_start = i
      break
  if microcode_start is None: microcode_start = int(total_pages * 0.9)

  pages = pdf.pages[microcode_start:microcode_start + 50]
  page_texts = [p.extract_text() or '' for p in pages]
  page_tables = [[t.extract() for t in p.find_tables()] for p in pages]
  full_text = '\n'.join(page_texts)

  # parse SSRC encoding from first page with VCC_LO
  src_enum = dict(SRC_EXTRAS)
  for text in page_texts[:10]:
    if 'SSRC0' in text and 'VCC_LO' in text:
      for m in re.finditer(r'^(\d+)\s+(\S+)', text, re.M):
        val, name = int(m.group(1)), m.group(2).rstrip('.:')
        if name in FLOAT_MAP: src_enum[val] = FLOAT_MAP[name]
        elif re.match(r'^[A-Z][A-Z0-9_]*$', name): src_enum[val] = name
      break

  # parse opcode tables
  enums: dict[str, dict[int, str]] = {}
  for m in re.finditer(r'Table \d+\. (\w+) Opcodes(.*?)(?=Table \d+\.|\n\d+\.\d+\.\d+\.\s+\w+\s*\nDescription|$)', full_text, re.S):
    if ops := {int(x.group(1)): x.group(2) for x in re.finditer(r'(\d+)\s+([A-Z][A-Z0-9_]+)', m.group(2))}:
      enums[m.group(1) + "Op"] = ops
  if vopd_m := re.search(r'Table \d+\. VOPD Y-Opcodes\n(.*?)(?=Table \d+\.|15\.\d)', full_text, re.S):
    if ops := {int(x.group(1)): x.group(2) for x in re.finditer(r'(\d+)\s+(V_DUAL_\w+)', vopd_m.group(1))}:
      enums["VOPDOp"] = ops
  enum_names = set(enums.keys())

  def is_fields_table(t) -> bool: return t and len(t) > 1 and t[0] and 'Field' in str(t[0][0] or '')
  def has_encoding(fields) -> bool: return any(f[0] == 'ENCODING' for f in fields)
  def has_header_before_fields(text) -> bool:
    return (pos := text.find('Field Name')) != -1 and bool(re.search(r'\d+\.\d+\.\d+\.\s+\w+\s*\n', text[:pos]))

  # find format headers with their page indices
  format_headers = []
  for i, text in enumerate(page_texts):
    for m in re.finditer(r'\d+\.\d+\.\d+\.\s+(\w+)\s*\n?Description', text): format_headers.append((m.group(1), i, m.start()))
    for m in re.finditer(r'\d+\.\d+\.\d+\.\s+(\w+)\s*\n', text):
      fmt_name = m.group(1)
      if is_cdna and fmt_name.isupper() and len(fmt_name) >= 2:
        format_headers.append((fmt_name, i, m.start()))
      elif m.start() > len(text) - 200 and 'Description' not in text[m.end():] and i + 1 < len(page_texts):
        next_text = page_texts[i + 1].lstrip()
        if next_text.startswith('Description') or (next_text.startswith('"RDNA') and 'Description' in next_text[:200]):
          format_headers.append((fmt_name, i, m.start()))

  # parse instruction formats
  formats: dict[str, list] = {}
  for fmt_name, page_idx, header_pos in format_headers:
    if fmt_name in formats: continue
    text, tables = page_texts[page_idx], page_tables[page_idx]
    field_pos = text.find('Field Name', header_pos)

    fields = None
    for offset in range(3):
      if page_idx + offset >= len(pages): break
      if offset > 0 and has_header_before_fields(page_texts[page_idx + offset]): break
      for t in page_tables[page_idx + offset] if offset > 0 or field_pos > header_pos else []:
        if is_fields_table(t) and (f := _parse_fields_table(t, fmt_name, enum_names)) and has_encoding(f):
          fields = f
          break
      if fields: break

    if not fields and field_pos > header_pos:
      for t in tables:
        if is_fields_table(t) and (f := _parse_fields_table(t, fmt_name, enum_names)):
          fields = f
          break

    if not fields: continue
    field_names = {f[0] for f in fields}

    for pg_offset in range(1, 3):
      if page_idx + pg_offset >= len(pages) or has_header_before_fields(page_texts[page_idx + pg_offset]): break
      for t in page_tables[page_idx + pg_offset]:
        if is_fields_table(t) and (extra := _parse_fields_table(t, fmt_name, enum_names)) and not has_encoding(extra):
          for ef in extra:
            if ef[0] not in field_names:
              fields.append(ef)
              field_names.add(ef[0])
          break
    formats[fmt_name] = fields

  # fix known PDF errors
  if 'SMEM' in formats:
    formats['SMEM'] = [(n, 13 if n == 'DLC' else 14 if n == 'GLC' else h, 13 if n == 'DLC' else 14 if n == 'GLC' else l, e, t)
                       for n, h, l, e, t in formats['SMEM']]

  return {"formats": formats, "enums": enums, "src_enum": src_enum, "doc_name": doc_name, "is_cdna": is_cdna}

def _merge_results(results: list[dict]) -> dict:
  """Merge multiple PDF parse results into a superset. Asserts if any conflicts."""
  merged = {"formats": {}, "enums": {}, "src_enum": dict(SRC_EXTRAS), "doc_names": [], "is_cdna": False}
  for r in results:
    merged["doc_names"].append(r["doc_name"])
    merged["is_cdna"] = merged["is_cdna"] or r["is_cdna"]
    # Merge src_enum (union, assert no conflicts)
    for val, name in r["src_enum"].items():
      if val in merged["src_enum"]:
        assert merged["src_enum"][val] == name, f"SrcEnum conflict: {val} = {merged['src_enum'][val]} vs {name}"
      else:
        merged["src_enum"][val] = name
    # Merge enums (union of ops per enum, assert no conflicts)
    for enum_name, ops in r["enums"].items():
      if enum_name not in merged["enums"]: merged["enums"][enum_name] = {}
      for val, name in ops.items():
        if val in merged["enums"][enum_name]:
          assert merged["enums"][enum_name][val] == name, f"{enum_name} conflict: {val} = {merged['enums'][enum_name][val]} vs {name}"
        else:
          merged["enums"][enum_name][val] = name
    # Merge formats (union of fields, assert no bit position conflicts for same field name)
    for fmt_name, fields in r["formats"].items():
      if fmt_name not in merged["formats"]:
        merged["formats"][fmt_name] = list(fields)
      else:
        existing = {f[0]: (f[1], f[2]) for f in merged["formats"][fmt_name]}  # name -> (hi, lo)
        for f in fields:
          name, hi, lo = f[0], f[1], f[2]
          if name in existing:
            assert existing[name] == (hi, lo), f"Format {fmt_name} field {name} conflict: bits {existing[name]} vs ({hi}, {lo})"
          else:
            merged["formats"][fmt_name].append(f)
  return merged

def generate(output_path: str | None = None, arch: str = "rdna3") -> dict:
  """Generate instruction definitions from AMD ISA PDF(s). Returns dict with formats for testing."""
  urls = PDF_URLS[arch]
  if isinstance(urls, str): urls = [urls]

  # Parse all PDFs and merge
  results = [_parse_single_pdf(url) for url in urls]
  if len(results) == 1:
    merged = results[0]
    doc_name = merged["doc_name"]
  else:
    merged = _merge_results(results)
    doc_name = "+".join(merged["doc_names"])

  formats, enums, src_enum = merged["formats"], merged["enums"], merged["src_enum"]

  # generate output
  def enum_lines(name, items):
    return [f"class {name}(IntEnum):"] + [f"  {n} = {v}" for v, n in sorted(items.items())] + [""]
  def field_key(f): return order.index(f[0].lower()) if f[0].lower() in order else 1000
  lines = [f"# autogenerated from AMD {doc_name} ISA PDF by dsl.py - do not edit", "from enum import IntEnum",
           "from typing import Annotated",
           "from extra.assembly.amd.dsl import bits, BitField, Inst32, Inst64, SGPR, VGPR, TTMP as TTMP, s as s, v as v, ttmp as ttmp, SSrc, Src, SImm, Imm, VDSTYEnc, SGPRField, VGPRField",
           "import functools", ""]
  lines += enum_lines("SrcEnum", src_enum) + sum([enum_lines(n, ops) for n, ops in sorted(enums.items())], [])
  # Format-specific field defaults (verified against LLVM test vectors)
  format_defaults = {'VOP3P': {'opsel_hi': 3, 'opsel_hi2': 1}}
  lines.append("# instruction formats")
  for fmt_name, fields in sorted(formats.items()):
    base = "Inst64" if max(f[1] for f in fields) > 31 or fmt_name == 'VOP3SD' else "Inst32"
    order = FIELD_ORDER.get(fmt_name, [])
    lines.append(f"class {fmt_name}({base}):")
    if enc := next((f for f in fields if f[0] == 'ENCODING'), None):
      enc_str = f"bits[{enc[1]}:{enc[2]}] == 0b{enc[3]:b}" if enc[1] != enc[2] else f"bits[{enc[1]}] == {enc[3]}"
      lines.append(f"  encoding = {enc_str}")
    if defaults := format_defaults.get(fmt_name):
      lines.append(f"  _defaults = {defaults}")
    for name, hi, lo, _, ftype in sorted([f for f in fields if f[0] != 'ENCODING'], key=field_key):
      if ftype and ftype.endswith('Op'):
        ann = f":Annotated[BitField, {ftype}]"
      else:
        ann = f":{ftype}" if ftype else ""
      lines.append(f"  {name.lower()}{ann} = bits[{hi}]" if hi == lo else f"  {name.lower()}{ann} = bits[{hi}:{lo}]")
    lines.append("")
  lines.append("# instruction helpers")
  for cls_name, ops in sorted(enums.items()):
    fmt = cls_name[:-2]
    for op_val, name in sorted(ops.items()):
      seg = {"GLOBAL": ", seg=2", "SCRATCH": ", seg=2"}.get(fmt, "")
      tgt = {"GLOBAL": "FLAT, GLOBALOp", "SCRATCH": "FLAT, SCRATCHOp"}.get(fmt, f"{fmt}, {cls_name}")
      if fmt in formats or fmt in ("GLOBAL", "SCRATCH"):
        if fmt in ("VOP1", "VOP2", "VOPC"):
          suffix = "_e32"
        elif fmt == "VOP3" and op_val < 512:
          suffix = "_e64"
        else:
          suffix = ""
        if name in ('V_FMAMK_F32', 'V_FMAMK_F16'):
          lines.append(f"def {name.lower()}{suffix}(vdst, src0, K, vsrc1): return {fmt}({cls_name}.{name}, vdst, src0, vsrc1, literal=K)")
        elif name in ('V_FMAAK_F32', 'V_FMAAK_F16'):
          lines.append(f"def {name.lower()}{suffix}(vdst, src0, vsrc1, K): return {fmt}({cls_name}.{name}, vdst, src0, vsrc1, literal=K)")
        else:
          lines.append(f"{name.lower()}{suffix} = functools.partial({tgt}.{name}{seg})")
  skip_exports = {'DPP8', 'DPP16'}
  src_names = {name for _, name in src_enum.items()}
  lines += [""] + [f"{name} = SrcEnum.{name}" for _, name in sorted(src_enum.items()) if name not in skip_exports]
  if "NULL" in src_names: lines.append("OFF = NULL\n")

  if output_path is not None:
    import pathlib
    pathlib.Path(output_path).write_text('\n'.join(lines))
  return {"formats": formats, "enums": enums, "src_enum": src_enum}

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Generate instruction definitions from AMD ISA PDF")
  parser.add_argument("--arch", choices=list(PDF_URLS.keys()) + ["all"], default="rdna3", help="Target architecture (default: rdna3)")
  args = parser.parse_args()
  if args.arch == "all":
    for arch in PDF_URLS.keys():
      result = generate(f"extra/assembly/amd/autogen/{arch}/__init__.py", arch=arch)
      print(f"{arch}: generated SrcEnum ({len(result['src_enum'])}) + {len(result['enums'])} opcode enums + {len(result['formats'])} format classes")
  else:
    result = generate(f"extra/assembly/amd/autogen/{args.arch}/__init__.py", arch=args.arch)
    print(f"generated SrcEnum ({len(result['src_enum'])}) + {len(result['enums'])} opcode enums + {len(result['formats'])} format classes")
