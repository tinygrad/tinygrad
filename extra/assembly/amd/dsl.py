# library for RDNA3 assembly DSL
# mypy: ignore-errors
from __future__ import annotations
import struct, math, re
from enum import IntEnum
from functools import cache, cached_property
from typing import overload, Annotated, TypeVar, Generic
from extra.assembly.amd.autogen.rdna3.enum import (VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp, VOPDOp, SOP1Op, SOP2Op,
  SOPCOp, SOPKOp, SOPPOp, SMEMOp, DSOp, FLATOp, MUBUFOp, MTBUFOp, MIMGOp, VINTERPOp)

# Common masks and bit conversion functions
MASK32, MASK64, MASK128 = 0xffffffff, 0xffffffffffffffff, (1 << 128) - 1
_struct_f, _struct_I = struct.Struct("<f"), struct.Struct("<I")
_struct_e, _struct_H = struct.Struct("<e"), struct.Struct("<H")
_struct_d, _struct_Q = struct.Struct("<d"), struct.Struct("<Q")
def _f32(i): return _struct_f.unpack(_struct_I.pack(i & MASK32))[0]
def _i32(f):
  if isinstance(f, int): f = float(f)
  if math.isnan(f): return 0xffc00000 if math.copysign(1.0, f) < 0 else 0x7fc00000
  if math.isinf(f): return 0x7f800000 if f > 0 else 0xff800000
  try: return _struct_I.unpack(_struct_f.pack(f))[0]
  except (OverflowError, struct.error): return 0x7f800000 if f > 0 else 0xff800000
def _sext(v, b): return v - (1 << b) if v & (1 << (b - 1)) else v
def _f16(i): return _struct_e.unpack(_struct_H.pack(i & 0xffff))[0]
def _i16(f):
  if math.isnan(f): return 0x7e00
  if math.isinf(f): return 0x7c00 if f > 0 else 0xfc00
  try: return _struct_H.unpack(_struct_e.pack(f))[0]
  except (OverflowError, struct.error): return 0x7c00 if f > 0 else 0xfc00
def _f64(i): return _struct_d.unpack(_struct_Q.pack(i & MASK64))[0]
def _i64(f):
  if math.isnan(f): return 0x7ff8000000000000
  if math.isinf(f): return 0x7ff0000000000000 if f > 0 else 0xfff0000000000000
  try: return _struct_Q.unpack(_struct_d.pack(f))[0]
  except (OverflowError, struct.error): return 0x7ff0000000000000 if f > 0 else 0xfff0000000000000

# Instruction spec - register counts and dtypes derived from instruction names
_REGS = {'B32': 1, 'B64': 2, 'B96': 3, 'B128': 4, 'B256': 8, 'B512': 16,
         'F32': 1, 'I32': 1, 'U32': 1, 'F64': 2, 'I64': 2, 'U64': 2,
         'F16': 1, 'I16': 1, 'U16': 1, 'B16': 1, 'I8': 1, 'U8': 1, 'B8': 1}
_CVT_RE = re.compile(r'CVT_([FIUB]\d+)_([FIUB]\d+)$')
_MAD_MUL_RE = re.compile(r'(?:MAD|MUL)_([IU]\d+)_([IU]\d+)$')
_PACK_RE = re.compile(r'PACK_([FIUB]\d+)_([FIUB]\d+)$')
_DST_SRC_RE = re.compile(r'_([FIUB]\d+)_([FIUB]\d+)$')
_SINGLE_RE = re.compile(r'_([FIUB](?:32|64|16|8|96|128|256|512))$')
@cache
def _suffix(name: str) -> tuple[str | None, str | None]:
  name = name.upper()
  if m := _CVT_RE.search(name): return m.group(1), m.group(2)
  if m := _MAD_MUL_RE.search(name): return m.group(1), m.group(2)
  if m := _PACK_RE.search(name): return m.group(1), m.group(2)
  if m := _DST_SRC_RE.search(name): return m.group(1), m.group(2)
  if m := _SINGLE_RE.search(name): return m.group(1), m.group(1)
  return None, None
_SPECIAL_REGS = {
  'V_LSHLREV_B64': (2, 1, 2, 1), 'V_LSHRREV_B64': (2, 1, 2, 1), 'V_ASHRREV_I64': (2, 1, 2, 1),
  'S_LSHL_B64': (2, 2, 1, 1), 'S_LSHR_B64': (2, 2, 1, 1), 'S_ASHR_I64': (2, 2, 1, 1),
  'S_BFE_U64': (2, 2, 1, 1), 'S_BFE_I64': (2, 2, 1, 1), 'S_BFM_B64': (2, 1, 1, 1),
  'S_BITSET0_B64': (2, 1, 1, 1), 'S_BITSET1_B64': (2, 1, 1, 1),
  'S_BITCMP0_B64': (1, 2, 1, 1), 'S_BITCMP1_B64': (1, 2, 1, 1),
  'V_LDEXP_F64': (2, 2, 1, 1), 'V_TRIG_PREOP_F64': (2, 2, 1, 1),
  'V_CMP_CLASS_F64': (1, 2, 1, 1), 'V_CMPX_CLASS_F64': (1, 2, 1, 1),
  'V_CMP_CLASS_F32': (1, 1, 1, 1), 'V_CMPX_CLASS_F32': (1, 1, 1, 1),
  'V_CMP_CLASS_F16': (1, 1, 1, 1), 'V_CMPX_CLASS_F16': (1, 1, 1, 1),
  'V_MAD_U64_U32': (2, 1, 1, 2), 'V_MAD_I64_I32': (2, 1, 1, 2),
  'V_QSAD_PK_U16_U8': (2, 2, 1, 2), 'V_MQSAD_PK_U16_U8': (2, 2, 1, 2), 'V_MQSAD_U32_U8': (4, 2, 1, 4),
}
_SPECIAL_DTYPE = {
  'V_LSHLREV_B64': ('B64', 'U32', 'B64', None), 'V_LSHRREV_B64': ('B64', 'U32', 'B64', None), 'V_ASHRREV_I64': ('I64', 'U32', 'I64', None),
  'S_LSHL_B64': ('B64', 'B64', 'U32', None), 'S_LSHR_B64': ('B64', 'B64', 'U32', None), 'S_ASHR_I64': ('I64', 'I64', 'U32', None),
  'S_BFE_U64': ('U64', 'U64', 'U32', None), 'S_BFE_I64': ('I64', 'I64', 'U32', None),
  'S_BFM_B64': ('B64', 'U32', 'U32', None), 'S_BITSET0_B64': ('B64', 'U32', None, None), 'S_BITSET1_B64': ('B64', 'U32', None, None),
  'S_BITCMP0_B64': ('SCC', 'B64', 'U32', None), 'S_BITCMP1_B64': ('SCC', 'B64', 'U32', None),
  'V_LDEXP_F64': ('F64', 'F64', 'I32', None), 'V_TRIG_PREOP_F64': ('F64', 'F64', 'U32', None),
  'V_CMP_CLASS_F64': ('VCC', 'F64', 'U32', None), 'V_CMPX_CLASS_F64': ('EXEC', 'F64', 'U32', None),
  'V_CMP_CLASS_F32': ('VCC', 'F32', 'U32', None), 'V_CMPX_CLASS_F32': ('EXEC', 'F32', 'U32', None),
  'V_CMP_CLASS_F16': ('VCC', 'F16', 'U32', None), 'V_CMPX_CLASS_F16': ('EXEC', 'F16', 'U32', None),
  'V_MAD_U64_U32': ('U64', 'U32', 'U32', 'U64'), 'V_MAD_I64_I32': ('I64', 'I32', 'I32', 'I64'),
  'V_QSAD_PK_U16_U8': ('B64', 'B64', 'B64', 'B64'), 'V_MQSAD_PK_U16_U8': ('B64', 'B64', 'B64', 'B64'),
  'V_MQSAD_U32_U8': ('B128', 'B64', 'B64', 'B128'),
}
@cache
def spec_regs(name: str) -> tuple[int, int, int, int]:
  uname = name.upper()
  if uname in _SPECIAL_REGS: return _SPECIAL_REGS[uname]
  if 'SAD' in uname and 'U8' in uname and 'QSAD' not in uname and 'MQSAD' not in uname: return 1, 1, 1, 1
  dst_suf, src_suf = _suffix(name)
  return _REGS.get(dst_suf, 1), _REGS.get(src_suf, 1), _REGS.get(src_suf, 1), _REGS.get(src_suf, 1)
@cache
def spec_dtype(name: str) -> tuple[str | None, str | None, str | None, str | None]:
  uname = name.upper()
  if uname in _SPECIAL_DTYPE: return _SPECIAL_DTYPE[uname]
  if 'SAD' in uname and ('U8' in uname or 'U16' in uname) and 'QSAD' not in uname and 'MQSAD' not in uname: return 'U32', 'U32', 'U32', 'U32'
  if '_CMP_' in uname or '_CMPX_' in uname:
    dst_suf, src_suf = _suffix(name)
    return 'EXEC' if '_CMPX_' in uname else 'VCC', src_suf, src_suf, None
  dst_suf, src_suf = _suffix(name)
  return dst_suf, src_suf, src_suf, src_suf
_F16_RE = re.compile(r'_[FIUB]16(?:_|$)')
_F64_RE = re.compile(r'_[FIUB]64(?:_|$)')
@cache
def spec_is_16bit(name: str) -> bool:
  uname = name.upper()
  if 'SAD' in uname or 'PACK' in uname or '_PK_' in uname or 'SAT_PK' in uname or 'DOT2' in uname: return False
  if '_F32' in uname or '_I32' in uname or '_U32' in uname or '_B32' in uname: return False
  return bool(_F16_RE.search(uname))
@cache
def spec_is_64bit(name: str) -> bool: return bool(_F64_RE.search(name.upper()))
_3SRC = {'FMA', 'MAD', 'MIN3', 'MAX3', 'MED3', 'DIV_FIX', 'DIV_FMAS', 'DIV_SCALE', 'SAD', 'LERP', 'ALIGN', 'CUBE', 'BFE', 'BFI',
         'PERM_B32', 'PERMLANE', 'CNDMASK', 'XOR3', 'OR3', 'ADD3', 'LSHL_OR', 'AND_OR', 'LSHL_ADD', 'ADD_LSHL', 'XAD', 'MAXMIN',
         'MINMAX', 'DOT2', 'DOT4', 'DOT8', 'WMMA', 'CVT_PK_U8', 'MULLIT', 'CO_CI'}
_2SRC = {'FMAC'}  # FMAC uses dst as implicit accumulator, so only 2 explicit sources
def spec_num_srcs(name: str) -> int:
  name = name.upper()
  if any(k in name for k in _2SRC): return 2
  return 3 if any(k in name for k in _3SRC) else 2
def is_dtype_16(dt: str | None) -> bool: return dt is not None and '16' in dt
def is_dtype_64(dt: str | None) -> bool: return dt is not None and '64' in dt

# Bit field DSL
class BitField:
  def __init__(self, hi: int, lo: int, name: str | None = None): self.hi, self.lo, self.name, self._marker = hi, lo, name, None
  def __set_name__(self, owner, name):
    import typing
    self.name, self._owner = name, owner
    # Cache marker at class definition time
    hints = typing.get_type_hints(owner, include_extras=True)
    if name in hints:
      hint = hints[name]
      if typing.get_origin(hint) is Annotated:
        args = typing.get_args(hint)
        self._marker = args[1] if len(args) > 1 else None
  def __eq__(self, val: int) -> tuple[BitField, int]: return (self, val)  # type: ignore
  def mask(self) -> int: return (1 << (self.hi - self.lo + 1)) - 1
  @property
  def marker(self) -> type | None: return self._marker
  @overload
  def __get__(self, obj: None, objtype: type) -> BitField: ...
  @overload
  def __get__(self, obj: object, objtype: type | None = None) -> int: ...
  def __get__(self, obj, objtype=None):
    if obj is None: return self
    val = unwrap(obj._values.get(self.name, 0))
    # Convert to IntEnum if marker is an IntEnum subclass
    if self.marker and isinstance(self.marker, type) and issubclass(self.marker, IntEnum):
      # VOP3 with VOPC opcodes (0-255) -> VOPCOp, VOP3SD opcodes -> VOP3SDOp
      if self.marker is VOP3Op:
        if val < 256: return VOPCOp(val)
        if val in Inst._VOP3SD_OPS: return VOP3SDOp(val)
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

# Encoding/decoding constants
FLOAT_ENC = {0.5: 240, -0.5: 241, 1.0: 242, -1.0: 243, 2.0: 244, -2.0: 245, 4.0: 246, -4.0: 247}
FLOAT_DEC = {v: str(k) for k, v in FLOAT_ENC.items()}
SPECIAL_GPRS = {106: "vcc_lo", 107: "vcc_hi", 124: "null", 125: "m0", 126: "exec_lo", 127: "exec_hi", 253: "scc"}
SPECIAL_PAIRS = {106: "vcc", 126: "exec"}
SRC_FIELDS = {'src0', 'src1', 'src2', 'ssrc0', 'ssrc1', 'soffset', 'srcx0', 'srcy0'}
RAW_FIELDS = {'vdata', 'vdst', 'vaddr', 'addr', 'data', 'data0', 'data1', 'sdst', 'sdata', 'vsrc1'}

def _encode_reg(val: Reg) -> int: return (108 if isinstance(val, TTMP) else 0) + val.idx

def _is_inline_const(v: int) -> bool: return 0 <= v <= 127 or 128 <= v <= 208 or 240 <= v <= 255

def encode_src(val) -> int:
  if isinstance(val, VGPR): return 256 + _encode_reg(val)
  if isinstance(val, Reg): return _encode_reg(val)
  if isinstance(val, SrcMod) and not isinstance(val, Reg): return val.val if _is_inline_const(val.val) else 255
  if hasattr(val, 'value'): return val.value  # IntEnum
  if isinstance(val, float): return 128 if val == 0.0 else FLOAT_ENC.get(val, 255)
  if isinstance(val, int): return 128 + val if 0 <= val <= 64 else 192 - val if -16 <= val <= -1 else 255
  return 255

def decode_src(val: int) -> str:
  if val <= 105: return f"s{val}"
  if val in SPECIAL_GPRS: return SPECIAL_GPRS[val]
  if val in FLOAT_DEC: return FLOAT_DEC[val]
  if 108 <= val <= 123: return f"ttmp{val - 108}"
  if 128 <= val <= 192: return str(val - 128)
  if 193 <= val <= 208: return str(-(val - 192))
  if 256 <= val <= 511: return f"v{val - 256}"
  return "lit" if val == 255 else f"?{val}"

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

  def _or_field(self, name: str, bit: int):
    cur = self._values.get(name, 0)
    self._values[name] = (cur.val if isinstance(cur, RawImm) else cur) | bit

  def _encode_src(self, name: str, val):
    """Encode a source field, handling modifiers and literals."""
    encoded = encode_src(val)
    has_opsel = 'opsel' in self._fields
    if isinstance(val, Reg) and val.hi and not has_opsel: encoded |= 0x80  # hi bit in src for VOP1/2/C
    self._values[name] = RawImm(encoded)
    # Handle neg/abs/opsel modifiers
    if isinstance(val, SrcMod):
      mod_bit = {'src0': 1, 'src1': 2, 'src2': 4}.get(name, 0)
      if val.neg and 'neg' in self._fields: self._or_field('neg', mod_bit)
      if val.abs_ and 'abs' in self._fields: self._or_field('abs', mod_bit)
    if isinstance(val, Reg) and val.hi and has_opsel:
      self._or_field('opsel', {'src0': 1, 'src1': 2, 'src2': 4}.get(name, 0))
    # Track literal value if needed
    if encoded == 255 and self._literal is None:
      import struct
      # Check if THIS source uses 64-bit encoding (not just src0)
      src_idx = {'src0': 0, 'src1': 1, 'src2': 2, 'ssrc0': 0, 'ssrc1': 1}.get(name, 0)
      src_regs = self.src_regs(src_idx)
      is_64 = src_regs == 2
      if isinstance(val, SrcMod) and not isinstance(val, Reg): lit32 = val.val & MASK32
      elif isinstance(val, int) and not isinstance(val, IntEnum): lit32 = val & MASK32
      elif isinstance(val, float): lit32 = (_i64(val) >> 32) if is_64 else _i32(val)  # f64: high 32 bits of f64 repr
      else: return
      self._literal = (lit32 << 32) if is_64 else lit32

  def _encode_raw(self, name: str, val):
    """Encode a raw register field (vdst, vdata, etc.)."""
    if isinstance(val, Reg):
      encoded = _encode_reg(val)
      if val.hi and 'opsel' not in self._fields: encoded |= 0x80
      self._values[name] = encoded
      if name == 'vdst' and val.hi and 'opsel' in self._fields: self._or_field('opsel', 8)
    elif hasattr(val, 'value'): self._values[name] = val.value

  def _validate(self, orig_args: dict):
    """Format-specific validation. Override in subclass or check by class name."""
    cls_name, op = self.__class__.__name__, orig_args.get('op')
    if hasattr(op, 'value'): op = op.value
    # SMEM: register count must match opcode
    if cls_name == 'SMEM' and op is not None:
      expected = {0:1, 1:2, 2:4, 3:8, 4:16, 8:1, 9:2, 10:4, 11:8, 12:16}.get(op)
      sdata = orig_args.get('sdata')
      if expected and isinstance(sdata, Reg) and sdata.count != expected:
        raise ValueError(f"SMEM op {op} expects {expected} registers, got {sdata.count}")
    # SOP1: b32=1 reg, b64=2 regs
    if cls_name == 'SOP1' and hasattr(orig_args.get('op'), 'name'):
      expected = 2 if orig_args['op'].name.endswith('_B64') else 1
      for fld in ('sdst', 'ssrc0'):
        if isinstance(orig_args.get(fld), Reg) and orig_args[fld].count != expected:
          raise ValueError(f"SOP1 {orig_args['op'].name} expects {expected} register(s) for {fld}, got {orig_args[fld].count}")

  def __init__(self, *args, literal: int | None = None, **kwargs):
    self._values, self._literal = dict(self._defaults), None
    field_names = [n for n in self._fields if n != 'encoding']
    orig_args = dict(zip(field_names, args)) | kwargs
    self._values.update(orig_args)
    self._validate(orig_args)
    # Pre-shift literal for 64-bit sources (literal param is always raw 32-bit value from user)
    if literal is not None:
      # Find which source uses the literal (255) and check its register count
      for n, idx in [('src0', 0), ('src1', 1), ('src2', 2), ('ssrc0', 0), ('ssrc1', 1)]:
        v = orig_args.get(n)
        if (isinstance(v, RawImm) and v.val == 255) or (isinstance(v, int) and v == 255):
          self._literal = (literal << 32) if self.src_regs(idx) == 2 else literal
          break
      else:
        self._literal = literal  # fallback if no literal source found
    cls_name = self.__class__.__name__

    # Format-specific setup
    if cls_name == 'FLAT' and 'sve' in self._fields:
      seg = self._values.get('seg', 0)
      if (seg.val if isinstance(seg, RawImm) else seg) == 1 and isinstance(orig_args.get('addr'), VGPR): self._values['sve'] = 1
    if cls_name == 'VOP3P':
      op = orig_args.get('op')
      if hasattr(op, 'value'): op = op.value
      if op in (32, 33, 34) and 'opsel_hi' not in orig_args: self._values['opsel_hi'] = self._values['opsel_hi2'] = 0

    # Encode all fields
    for name, val in list(self._values.items()):
      if name == 'encoding': continue
      if isinstance(val, RawImm):
        if name in RAW_FIELDS: self._values[name] = val.val
        continue
      field = self._fields.get(name)
      marker = field.marker if field else None
      # Type validation
      if marker is _SGPRField and isinstance(val, VGPR): raise TypeError(f"field '{name}' requires SGPR, got VGPR")
      if marker is _VGPRField and not isinstance(val, VGPR): raise TypeError(f"field '{name}' requires VGPR, got {type(val).__name__}")
      if marker is _SSrc and isinstance(val, VGPR): raise TypeError(f"field '{name}' requires scalar source, got VGPR")
      # Encode by field type
      if name in SRC_FIELDS: self._encode_src(name, val)
      elif name in RAW_FIELDS: self._encode_raw(name, val)
      elif name == 'sbase': self._values[name] = (val.idx if isinstance(val, Reg) else val.val if isinstance(val, SrcMod) else val * 2) // 2
      elif name in {'srsrc', 'ssamp'} and isinstance(val, Reg): self._values[name] = val.idx // 4
      elif marker is _VDSTYEnc and isinstance(val, VGPR): self._values[name] = val.idx >> 1

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
    """Check if this instruction uses 64-bit operands (and thus 64-bit literals)."""
    op = self._values.get('op')
    if op is None: return False
    op_name = op.name if hasattr(op, 'name') else None
    # Look up op name from int if needed (happens in from_bytes path)
    if op_name is None and self.__class__.__name__ == 'VOP3':
      try: op_name = VOP3Op(op).name
      except ValueError: pass
    if op_name is None and self.__class__.__name__ == 'VOPC':
      try: op_name = VOPCOp(op).name
      except ValueError: pass
    if op_name is None: return False
    # V_LDEXP_F64 has 32-bit integer src1, so literal is 32-bit
    return op_name != 'V_LDEXP_F64' and op_name.endswith(('_F64', '_B64', '_I64', '_U64'))

  def to_bytes(self) -> bytes:
    result = self.to_int().to_bytes(self._size(), 'little')
    lit = self._get_literal() or getattr(self, '_literal', None)
    if lit is None: return result
    # For 64-bit sources, literal is stored in high 32 bits internally, but encoded as 4 bytes
    # Find which source uses the literal (255) and check its register count
    lit_src_is_64 = False
    for n, idx in [('src0', 0), ('src1', 1), ('src2', 2), ('ssrc0', 0), ('ssrc1', 1)]:
      if n not in self._values: continue
      v = self._values[n]
      if (isinstance(v, RawImm) and v.val == 255) or (isinstance(v, int) and v == 255):
        lit_src_is_64 = self.is_src_64(idx)
        break
    lit32 = (lit >> 32) if lit_src_is_64 else lit
    return result + (lit32 & MASK32).to_bytes(4, 'little')

  @classmethod
  def _size(cls) -> int: return 4 if issubclass(cls, Inst32) else 12 if issubclass(cls, Inst96) else 8
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
      # Check which source uses the literal and whether THAT source is 64-bit
      if len(data) >= cls._size() + 4:
        lit32 = int.from_bytes(data[cls._size():cls._size()+4], 'little')
        # Find which source has literal (255) and check its register count
        lit_src_is_64 = False
        for n, idx in [('src0', 0), ('src1', 1), ('src2', 2)]:
          if n in inst._values and isinstance(inst._values[n], RawImm) and inst._values[n].val == 255:
            lit_src_is_64 = inst.src_regs(idx) == 2
            break
        inst._literal = (lit32 << 32) if lit_src_is_64 else lit32
    return inst

  def __repr__(self):
    # Use _fields order and exclude fields that are 0/default (for consistent repr after roundtrip)
    def is_zero(v): return (isinstance(v, int) and v == 0) or (isinstance(v, VGPR) and v.idx == 0 and v.count == 1)
    items = [(k, self._values[k]) for k in self._fields if k in self._values and k != 'encoding'
             and not (is_zero(self._values[k]) and k not in {'op'})]
    lit = f", literal={hex(self._literal)}" if self._literal is not None else ""
    return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in items)}{lit})"

  def __getattr__(self, name: str):
    if name.startswith('_'): raise AttributeError(name)
    return unwrap(self._values.get(name, 0))

  def lit(self, v: int, neg: bool = False) -> str:
    s = f"0x{self._literal:x}" if v == 255 and self._literal else decode_src(v)
    return f"-{s}" if neg else s

  def __eq__(self, other):
    if not isinstance(other, Inst): return NotImplemented
    return self.__class__ == other.__class__ and self._values == other._values and self._literal == other._literal

  def __hash__(self): return hash((self.__class__.__name__, tuple(sorted((k, repr(v)) for k, v in self._values.items())), self._literal))

  def disasm(self) -> str:
    from extra.assembly.amd.asm import disasm
    return disasm(self)

  _enum_map = {'VOP1': VOP1Op, 'VOP2': VOP2Op, 'VOP3': VOP3Op, 'VOP3SD': VOP3SDOp, 'VOP3P': VOP3POp, 'VOPC': VOPCOp,
               'SOP1': SOP1Op, 'SOP2': SOP2Op, 'SOPC': SOPCOp, 'SOPK': SOPKOp, 'SOPP': SOPPOp,
               'SMEM': SMEMOp, 'DS': DSOp, 'FLAT': FLATOp, 'MUBUF': MUBUFOp, 'MTBUF': MTBUFOp, 'MIMG': MIMGOp,
               'VOPD': VOPDOp, 'VINTERP': VINTERPOp}
  _VOP3SD_OPS = {288, 289, 290, 764, 765, 766, 767, 768, 769, 770}

  @property
  def op(self):
    """Return the op as an enum (e.g., VOP1Op.V_MOV_B32). VOP3 returns VOPCOp/VOP3SDOp for those op ranges."""
    val = self._values.get('op')
    if val is None: return None
    if hasattr(val, 'name'): return val  # already an enum
    cls_name = self.__class__.__name__
    assert cls_name in self._enum_map, f"no enum map for {cls_name}"
    return self._enum_map[cls_name](val)

  @cached_property
  def op_name(self) -> str:
    op = self.op
    return op.name if hasattr(op, 'name') else ''

  @cached_property
  def _spec_regs(self) -> tuple[int, int, int, int]: return spec_regs(self.op_name)
  @cached_property
  def _spec_dtype(self) -> tuple[str | None, str | None, str | None, str | None]: return spec_dtype(self.op_name)
  def dst_regs(self) -> int: return self._spec_regs[0]
  def src_regs(self, n: int) -> int: return self._spec_regs[n + 1]
  def num_srcs(self) -> int: return spec_num_srcs(self.op_name)
  def dst_dtype(self) -> str | None: return self._spec_dtype[0]
  def src_dtype(self, n: int) -> str | None: return self._spec_dtype[n + 1]
  def is_src_16(self, n: int) -> bool: return self._spec_regs[n + 1] == 1 and is_dtype_16(self._spec_dtype[n + 1])
  def is_src_64(self, n: int) -> bool: return self._spec_regs[n + 1] == 2
  def is_16bit(self) -> bool: return spec_is_16bit(self.op_name)
  def is_64bit(self) -> bool: return spec_is_64bit(self.op_name)
  def is_dst_16(self) -> bool: return self._spec_regs[0] == 1 and is_dtype_16(self._spec_dtype[0])

class Inst32(Inst): pass
class Inst64(Inst): pass
class Inst96(Inst): pass
