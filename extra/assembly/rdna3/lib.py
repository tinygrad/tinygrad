# library for RDNA3 assembly DSL
from __future__ import annotations
import re
from enum import IntEnum

# Bit field DSL
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

# Register types
class Reg:
  def __init__(self, idx: int, count: int = 1, hi: bool = False): self.idx, self.count, self.hi = idx, count, hi
  def __repr__(self): return f"{self.__class__.__name__.lower()[0]}[{self.idx}]" if self.count == 1 else f"{self.__class__.__name__.lower()[0]}[{self.idx}:{self.idx + self.count}]"
  @classmethod
  def __class_getitem__(cls, key): return cls(key.start, key.stop - key.start) if isinstance(key, slice) else cls(key)
class SGPR(Reg): pass
class VGPR(Reg): pass
class TTMP(Reg): pass
s, v = SGPR, VGPR

# Field type markers
class SSrc: pass
class Src: pass
class Imm: pass
class SImm: pass
class RawImm:
  def __init__(self, val: int): self.val = val

def unwrap(val) -> int:
  return val.val if isinstance(val, RawImm) else val.value if hasattr(val, 'value') else val.idx if hasattr(val, 'idx') else val

# Encoding helpers
FLOAT_ENC = {0.5: 240, -0.5: 241, 1.0: 242, -1.0: 243, 2.0: 244, -2.0: 245, 4.0: 246, -4.0: 247}
SRC_FIELDS = {'src0', 'src1', 'src2', 'ssrc0', 'ssrc1', 'soffset'}
RAW_FIELDS = {'vdata', 'vdst', 'vaddr', 'addr', 'data', 'data0', 'data1', 'sdst', 'sdata'}

def encode_src(val) -> int:
  if isinstance(val, SGPR): return val.idx | (0x80 if val.hi else 0)
  if isinstance(val, VGPR): return 256 + val.idx + (0x80 if val.hi else 0)  # .h sets bit 7 of VGPR encoding
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

# Instruction base class
class Inst:
  _fields: dict[str, BitField]
  _encoding: tuple[BitField, int] | None = None
  _defaults: dict[str, int] = {}  # field defaults

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    cls._fields = {n: v[0] if isinstance(v, tuple) else v for n, v in cls.__dict__.items() if isinstance(v, BitField) or (isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], BitField))}
    if 'encoding' in cls._fields and isinstance(cls.__dict__.get('encoding'), tuple): cls._encoding = cls.__dict__['encoding']

  def __init__(self, *args, literal: int | None = None, **kwargs):
    self._values, self._literal = dict(self._defaults), literal  # start with defaults
    self._values.update(zip([n for n in self._fields if n != 'encoding'], args))
    self._values.update(kwargs)

  def _encode_field(self, name: str, val) -> int:
    if isinstance(val, RawImm): return val.val
    if name in {'srsrc', 'ssamp'}: return val.idx // 4 if isinstance(val, Reg) else val
    if name == 'sbase': return val.idx // 2 if isinstance(val, Reg) else val
    if name in RAW_FIELDS:
      if isinstance(val, TTMP): return 108 + val.idx
      if isinstance(val, Reg): return val.idx | (0x80 if val.hi else 0)  # .h sets bit 7 for vdst
      return val
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

  def to_bytes(self) -> bytes:
    result = self.to_int().to_bytes(self._size(), 'little')
    return result + (lit & 0xffffffff).to_bytes(4, 'little') if (lit := self._get_literal() or getattr(self, '_literal', None)) else result

  @classmethod
  def _size(cls) -> int: return 4 if issubclass(cls, Inst32) else 8
  def size(self) -> int: return self._size() + (4 if self._literal is not None else 0)

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
    has_literal = has_literal or (cls.__name__ == 'SOP2' and op_val in (69, 70))  # S_FMAAK_F32, S_FMAMK_F32
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
    cls_name = self.__class__.__name__
    def fmt_src(v): return f"0x{self._literal:x}" if v == 255 and getattr(self, '_literal', None) else decode_src(v)
    def sreg(base, cnt): return f"s{base}" if cnt == 1 else f"s[{base}:{base+cnt-1}]"
    def vreg(base, cnt=1): return f"v{base}" if cnt == 1 else f"v[{base}:{base+cnt-1}]"
    # VOP1/VOP2/VOPC
    if cls_name == 'VOP1':
      return f"{op_name}_e32 v{unwrap(self._values['vdst'])}, {fmt_src(unwrap(self._values['src0']))}"
    if cls_name == 'VOP2':
      vdst, src0, vsrc1 = unwrap(self._values['vdst']), fmt_src(unwrap(self._values['src0'])), unwrap(self._values['vsrc1'])
      suffix = "" if op_name == "v_dot2acc_f32_f16" else "_e32"
      return f"{op_name}{suffix} v{vdst}, {src0}, v{vsrc1}" + (", vcc_lo" if op_name == "v_cndmask_b32" else "")
    if cls_name == 'VOPC':
      return f"{op_name}_e32 vcc_lo, {fmt_src(unwrap(self._values['src0']))}, v{unwrap(self._values['vsrc1'])}"
    # SOPP: handle s_waitcnt, s_delay_alu, s_endpgm specially
    if cls_name == 'SOPP':
      simm16 = unwrap(self._values.get('simm16', 0))
      if op_name == 's_endpgm': return 's_endpgm'
      if op_name == 's_barrier': return 's_barrier'
      if op_name == 's_waitcnt':
        vmcnt, expcnt, lgkmcnt = decode_waitcnt(simm16)
        parts = []
        if vmcnt != 0x3f: parts.append(f"vmcnt({vmcnt})")
        if expcnt != 0x7: parts.append(f"expcnt({expcnt})")  # RDNA3: expcnt is 4 bits, max 7
        if lgkmcnt != 0x3f: parts.append(f"lgkmcnt({lgkmcnt})")
        return f"s_waitcnt {' '.join(parts)}" if parts else "s_waitcnt 0"
      if op_name == 's_delay_alu':
        dep_names = ['VALU_DEP_1','VALU_DEP_2','VALU_DEP_3','VALU_DEP_4','TRANS32_DEP_1','TRANS32_DEP_2','TRANS32_DEP_3','FMA_ACCUM_CYCLE_1',
                     'SALU_CYCLE_1','SALU_CYCLE_2','SALU_CYCLE_3']
        skip_names = ['SAME','NEXT','SKIP_1','SKIP_2','SKIP_3','SKIP_4']
        id0, skip, id1 = simm16 & 0xf, (simm16 >> 4) & 0x7, (simm16 >> 7) & 0xf
        def dep_name(v): return dep_names[v-1] if 0 < v <= len(dep_names) else str(v)
        parts = [f"instid0({dep_name(id0)})"] if id0 else []
        if skip: parts.append(f"instskip({skip_names[skip]})"); parts.append(f"instid1({dep_name(id1)})" if id1 else "")
        return f"s_delay_alu {' | '.join(p for p in parts if p)}" if parts else "s_delay_alu 0"
      # Branch instructions use decimal offsets
      if op_name.startswith('s_cbranch') or op_name.startswith('s_branch'):
        return f"{op_name} {simm16}"
      return f"{op_name} 0x{simm16:x}" if simm16 else op_name
    # SMEM: s_load_bXX sdst, sbase, offset
    if cls_name == 'SMEM':
      sdata, sbase, soffset, offset = unwrap(self._values['sdata']), unwrap(self._values['sbase']), unwrap(self._values['soffset']), unwrap(self._values['offset'])
      width = {0:1, 1:2, 2:4, 3:8, 4:16, 8:1, 9:2, 10:4, 11:8, 12:16}.get(op_val, 1)
      off_str = f"0x{offset:x}" if offset else "null" if soffset == 124 else decode_src(soffset)
      return f"{op_name} {sreg(sdata, width)}, {sreg(sbase, 2)}, {off_str}"
    # FLAT: flat_*/global_*/scratch_* load/store
    if cls_name == 'FLAT':
      vdst, addr, data, saddr, offset, seg = [unwrap(self._values.get(f, 0)) for f in ['vdst', 'addr', 'data', 'saddr', 'offset', 'seg']]
      prefix = {0: 'flat', 1: 'scratch', 2: 'global'}.get(seg, 'flat')
      op_suffix = op_name.split('_', 1)[1] if '_' in op_name else op_name  # load_b32, store_b32, etc
      instr = f"{prefix}_{op_suffix}"
      is_store = 'store' in op_name
      width = {'b32':1, 'b64':2, 'b96':3, 'b128':4, 'u8':1, 'i8':1, 'u16':1, 'i16':1}.get(op_name.split('_')[-1], 1)
      # Address mode depends on saddr: 0x7F = no saddr (use 64-bit vaddr), else saddr is SGPR pair
      if saddr == 0x7F:
        addr_str, saddr_str = vreg(addr, 2), ""
      else:
        addr_str = vreg(addr)
        saddr_str = f", {sreg(saddr, 2)}" if saddr < 106 else f", off" if saddr == 124 else f", {decode_src(saddr)}"
      off_str = f" offset:{offset}" if offset else ""
      if is_store: return f"{instr} {addr_str}, {vreg(data, width)}{saddr_str}{off_str}"
      return f"{instr} {vreg(vdst, width)}, {addr_str}{saddr_str}{off_str}"
    # Generic disassembly for other formats
    def fmt(n, v):
      v = unwrap(v)
      if n in SRC_FIELDS: return fmt_src(v) if v != 255 else "0xff"
      if n in ('sdst', 'vdst'): return f"{'s' if n == 'sdst' else 'v'}{v}"
      return f"v{v}" if n == 'vsrc1' else f"0x{v:x}" if n == 'simm16' else str(v)
    ops = [fmt(n, self._values.get(n, 0)) for n in self._fields if n not in ('encoding', 'op')]
    return f"{op_name} {', '.join(ops)}" if ops else op_name

class Inst32(Inst): pass
class Inst64(Inst):
  def to_bytes(self) -> bytes:
    result = self.to_int().to_bytes(8, 'little')
    return result + (lit & 0xffffffff).to_bytes(4, 'little') if (lit := self._get_literal() or getattr(self, '_literal', None)) else result
  @classmethod
  def from_bytes(cls, data: bytes): return cls.from_int(int.from_bytes(data[:8], 'little'))

# Waitcnt helpers (RDNA3 format: bits 15:10=vmcnt, bits 9:4=lgkmcnt, bits 3:0=expcnt)
def waitcnt(vmcnt: int = 0x3f, expcnt: int = 0x7, lgkmcnt: int = 0x3f) -> int:
  return (expcnt & 0x7) | ((lgkmcnt & 0x3f) << 4) | ((vmcnt & 0x3f) << 10)
def decode_waitcnt(val: int) -> tuple[int, int, int]:
  return (val >> 10) & 0x3f, val & 0xf, (val >> 4) & 0x3f  # vmcnt, expcnt, lgkmcnt

# Assembler
SPECIAL_REGS = {'vcc_lo': RawImm(106), 'vcc_hi': RawImm(107), 'null': RawImm(124), 'off': RawImm(124), 'm0': RawImm(125), 'exec_lo': RawImm(126), 'exec_hi': RawImm(127), 'scc': RawImm(253)}
FLOAT_CONSTS = {'0.5': 0.5, '-0.5': -0.5, '1.0': 1.0, '-1.0': -1.0, '2.0': 2.0, '-2.0': -2.0, '4.0': 4.0, '-4.0': -4.0}
REG_MAP = {'s': SGPR, 'v': VGPR, 't': TTMP, 'ttmp': TTMP}

def parse_operand(op: str) -> tuple:
  op = op.strip().lower()
  neg = op.startswith('-') and not op[1:2].isdigit(); op = op[1:] if neg else op
  abs_ = op.startswith('|') and op.endswith('|') or op.startswith('abs(') and op.endswith(')')
  op = op[1:-1] if op.startswith('|') else op[4:-1] if op.startswith('abs(') else op
  # Handle .l/.h suffix (16-bit register halves)
  hi_half = op.endswith('.h')
  op = re.sub(r'\.[lh]$', '', op)
  if op in FLOAT_CONSTS: return (FLOAT_CONSTS[op], neg, abs_, hi_half)
  if re.match(r'^-?\d+$', op): return (int(op), neg, abs_, hi_half)
  if m := re.match(r'^-?0x([0-9a-f]+)$', op):
    v = -int(m.group(1), 16) if op.startswith('-') else int(m.group(1), 16)
    return (v, neg, abs_, hi_half)
  if op in SPECIAL_REGS: return (SPECIAL_REGS[op], neg, abs_, hi_half)
  if m := re.match(r'^([svt](?:tmp)?)\[(\d+):(\d+)\]$', op): return (REG_MAP[m.group(1)][int(m.group(2)):int(m.group(3))+1], neg, abs_, hi_half)
  if m := re.match(r'^([svt](?:tmp)?)(\d+)$', op):
    reg_cls = REG_MAP[m.group(1)]
    return (reg_cls(int(m.group(2)), 1, hi_half), neg, abs_, hi_half)
  raise ValueError(f"cannot parse operand: {op}")

SMEM_OPS = {'s_load_b32', 's_load_b64', 's_load_b128', 's_load_b256', 's_load_b512',
            's_buffer_load_b32', 's_buffer_load_b64', 's_buffer_load_b128', 's_buffer_load_b256', 's_buffer_load_b512'}
SOP1_SRC_ONLY = {'s_setpc_b64', 's_rfe_b64'}  # instructions with ssrc0 only, no sdst
SOP1_MSG_IMM = {'s_sendmsg_rtn_b32', 's_sendmsg_rtn_b64'}  # instructions with raw immediate in ssrc0
SOPK_IMM_ONLY = {'s_version'}  # instructions with simm16 only, no sdst
SOPK_IMM_FIRST = {'s_setreg_b32'}  # instructions where simm16 comes before sdst
SOPK_UNSUPPORTED = {'s_setreg_imm32_b32'}  # special 64-bit SOPK format

def asm(text: str) -> Inst:
  from extra.assembly.rdna3 import autogen
  text = text.strip()
  clamp = 'clamp' in text.lower()
  if clamp: text = re.sub(r'\s+clamp\s*$', '', text, flags=re.I)
  # Parse modifiers like wait_exp:N
  modifiers = {}
  if m := re.search(r'\s+wait_exp:(\d+)', text, re.I): modifiers['waitexp'] = int(m.group(1)); text = text[:m.start()] + text[m.end():]
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
  # Compute opsel bits for VOP3: bit0=src0.h, bit1=src1.h, bit2=src2.h, bit3=vdst.h
  opsel_bits = (8 if len(parsed) > 0 and parsed[0][3] else 0) | sum((1 << i) for i, p in enumerate(parsed[1:4]) if p[3])
  lit = None
  if mnemonic in ('v_fmaak_f32', 'v_fmaak_f16') and len(values) == 4: lit, values = unwrap(values[3]), values[:3]
  elif mnemonic in ('v_fmamk_f32', 'v_fmamk_f16') and len(values) == 4: lit, values = unwrap(values[2]), [values[0], values[1], values[3]]
  # VCC-using VOP2 instructions: skip implicit VCC operands (format: vdst, vcc_dst, src0, src1, vcc_src)
  vcc_ops = {'v_add_co_ci_u32', 'v_sub_co_ci_u32', 'v_subrev_co_ci_u32', 'v_add_co_u32', 'v_sub_co_u32', 'v_subrev_co_u32'}
  if mnemonic.replace('_e32', '') in vcc_ops and len(values) >= 5: values = [values[0], values[2], values[3]]
  # VOPC: skip implicit VCC destination operand (format: vcc_dst, src0, src1)
  if mnemonic.startswith('v_cmp') and len(values) >= 3 and operands[0].strip().lower() in ('vcc_lo', 'vcc_hi', 'vcc'):
    values = values[1:]  # skip vcc destination
  # VOP3SD (v_div_scale_*): has vdst, sdst, then 3 sources - neg/abs apply to sources (operands 2,3,4)
  vop3sd_ops = {'v_div_scale_f32', 'v_div_scale_f64'}
  if mnemonic in vop3sd_ops and len(parsed) >= 5:
    neg_bits = sum((1 << i) for i, p in enumerate(parsed[2:5]) if p[1])
    abs_bits = sum((1 << i) for i, p in enumerate(parsed[2:5]) if p[2])
  # Unsupported instructions
  if mnemonic in SOPK_UNSUPPORTED: raise ValueError(f"unsupported instruction: {mnemonic}")
  # SOP1 source-only instructions (no destination)
  elif mnemonic in SOP1_SRC_ONLY:
    return getattr(autogen, mnemonic)(ssrc0=values[0])
  # SOP1 instructions with raw immediate message ID
  elif mnemonic in SOP1_MSG_IMM:
    return getattr(autogen, mnemonic)(sdst=values[0], ssrc0=RawImm(unwrap(values[1])))
  # SOPK immediate-only instructions (no destination)
  elif mnemonic in SOPK_IMM_ONLY:
    return getattr(autogen, mnemonic)(simm16=values[0])
  # SOPK instructions with simm16 before sdst
  elif mnemonic in SOPK_IMM_FIRST:
    return getattr(autogen, mnemonic)(simm16=values[0], sdst=values[1])
  # SMEM: when third operand is immediate, use it as offset with soffset=NULL
  elif mnemonic in SMEM_OPS and len(operands) >= 3 and re.match(r'^-?[0-9]|^-?0x', operands[2].strip().lower()):
    return getattr(autogen, mnemonic)(sdata=values[0], sbase=values[1], offset=values[2], soffset=RawImm(124))
  # MUBUF: when vaddr is 'off', use 0 instead of NULL
  elif mnemonic.startswith('buffer_') and len(operands) >= 2 and operands[1].strip().lower() == 'off':
    return getattr(autogen, mnemonic)(vdata=values[0], vaddr=0, srsrc=values[2], soffset=RawImm(unwrap(values[3])) if len(values) > 3 else RawImm(0))
  for suffix in (['_e32', ''] if not (neg_bits or abs_bits or clamp) else ['', '_e32']):
    if hasattr(autogen, name := mnemonic.replace('.', '_') + suffix):
      use_opsel = 'opsel' in getattr(autogen, name).func._fields
      # For VOP3+, clear hi flags from registers (opsel handles hi half selection)
      vals = [type(v)(v.idx, v.count, False) if isinstance(v, Reg) and v.hi and use_opsel else v for v in values]
      inst = getattr(autogen, name)(*vals, literal=lit, **modifiers)
      if neg_bits and 'neg' in inst._fields: inst._values['neg'] = neg_bits
      if opsel_bits and use_opsel: inst._values['opsel'] = opsel_bits
      if abs_bits and 'abs' in inst._fields: inst._values['abs'] = abs_bits
      if clamp and 'clmp' in inst._fields: inst._values['clmp'] = 1
      return inst
  raise ValueError(f"unknown instruction: {mnemonic}")
