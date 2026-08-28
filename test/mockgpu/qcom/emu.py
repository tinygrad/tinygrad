from __future__ import annotations
import ctypes, math, re, struct, tempfile
from dataclasses import dataclass, field
from tinygrad.runtime.autogen import mesa, libc
from tinygrad.runtime.support import c
from functools import lru_cache
from tinygrad.helpers import data64, DEBUG

WAVE, MASK32, MASK16 = 64, 0xFFFFFFFF, 0xFFFF
TYPES = {0: 'f16', 1: 'f32', 2: 'u16', 3: 'u32', 4: 's16', 5: 's32', 6: 'u8', 7: 's8'}
CONDS = {0: 'lt', 1: 'le', 2: 'gt', 3: 'ge', 4: 'eq', 5: 'ne'}
TYPE_BYTES = {'f16': 2, 'f32': 4, 'u16': 2, 'u32': 4, 's16': 2, 's32': 4, 'u8': 1, 's8': 1, 'b16': 2, 'b32': 4}
# cat2 float immediates (FLUT); index 3 is 2.0 as in cmps.f.lt ..., (2.0)
FLUT = (0.0, 0.5, 1.0, 2.0, math.e, math.pi, 1.0 / math.pi, 1.0 / math.log2(math.e),
        math.log2(math.e), 1.0 / math.log2(10.0), math.log2(10.0), 4.0)

def _f32(u:int) -> float: return struct.unpack('<f', struct.pack('<I', u & MASK32))[0]
def _u32(f:float) -> int:
  try: return struct.unpack('<I', struct.pack('<f', float(f)))[0]
  except (OverflowError, ValueError): return 0x7f800000 if f > 0 else 0xff800000
def _f16(u:int) -> float: return struct.unpack('<e', struct.pack('<H', u & MASK16))[0]
def _u16f(f:float) -> int:
  try: return struct.unpack('<H', struct.pack('<e', float(f)))[0]
  except (OverflowError, ValueError): return 0x7c00 if f > 0 else 0xfc00
def _i32(u:int) -> int: return ctypes.c_int32(u & MASK32).value
def _i16(u:int) -> int: return ctypes.c_int16(u & MASK16).value
def _sext(u:int, bits:int) -> int:
  u &= (1 << bits) - 1
  return u - (1 << bits) if u & (1 << (bits - 1)) else u

FIELD_CB = c.CFUNCTYPE[None, [ctypes.c_void_p, c.POINTER[ctypes.c_char], c.POINTER[mesa.struct_isa_decode_value]]]
PRE_CB = c.CFUNCTYPE[None, [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]]
NO_MATCH_CB = c.CFUNCTYPE[None, [c.POINTER[mesa.struct__IO_FILE], c.POINTER[ctypes.c_uint32], ctypes.c_uint64]]

@dataclass
class Operand:
  slot: str
  kind: str = 'gpr'  # gpr, const, imm, rel_gpr, rel_const
  packed: int = 0
  half: bool = False
  absneg: int = 0
  src_r: bool = False
  imm: int = 0
  rel_off: int = 0

@dataclass
class Inst:
  name: str
  line: str
  dst: Operand|None = None
  srcs: list[Operand] = field(default_factory=list)
  repeat: int = 0
  sat: bool = False
  jp: bool = False
  pred: int|None = None
  extra: dict = field(default_factory=dict)

def _kw_from_fields(fields:list[tuple[str,str|None,int]]) -> dict:
  kw: dict = {}
  for name, s, num in fields:
    if name == 'NAME': kw['name'] = s
    elif name == 'REPEAT': kw['repeat'] = num
    elif name == 'SAT': kw['sat'] = bool(num)
    elif name == 'JP': kw['jp'] = bool(num)
    else: kw[name.lower()] = s if s is not None else num
  if 'name' not in kw and 'src_type' in kw:
    st, dt = TYPES.get(int(kw['src_type']), f"t{kw['src_type']}"), TYPES.get(int(kw.get('dst_type', 1)), 'u32')
    kw['name'] = f"{'mov' if st == dt else 'cov'}.{st}{dt}"
  if isinstance(kw.get('name'), str) and 'cond' in kw and '.' in kw['name'] and kw['name'].count('.') == 1:
    kw['name'] = f"{kw['name']}.{CONDS.get(int(kw['cond']), kw['cond'])}"
  kw.setdefault('name', 'unknown')
  return kw

def _fix_gpr_packed(fields:list[tuple[str,str|None,int]], inst:Inst):
  ops: list[Operand] = []
  cur: Operand|None = None
  gpr: int|None = None
  const: int|None = None
  swiz: int|None = None
  rel = False
  dst_half = False
  pending_half = False
  attach_half = True  # cat3 SRC2 HALF is emitted before SRC2 (after SRC2_NEG)
  def flush_ids():
    nonlocal gpr, const, swiz, rel
    if ops and (gpr is not None or const is not None or rel):
      op = ops[-1]
      if rel:
        op.kind = 'rel_const' if const is not None else 'rel_gpr'
        op.rel_off = const if const is not None else (gpr or 0)
        op.packed = swiz or 0
      elif gpr is not None: op.kind, op.packed = 'gpr', (gpr << 2) | (swiz or 0)
      elif const is not None: op.kind, op.packed = 'const', (const << 2) | (swiz or 0)
    gpr = const = swiz = None
    rel = False
  def start(slot, enc=0):
    nonlocal cur, pending_half, attach_half
    flush_ids()
    cur = Operand(slot=slot, packed=enc, half=(dst_half if slot == 'DST' else False) or pending_half)
    pending_half, attach_half = False, True
    ops.append(cur)
  for name, s, num in fields:
    if name == 'DST_HALF':
      dst_half = bool(num)
      for o in ops:
        if o.slot == 'DST': o.half = dst_half
    elif name in {'SRC1_R', 'SRC2_R', 'SRC3_R', 'SRC1_NEG', 'SRC2_NEG', 'SRC3_NEG'}:
      attach_half = False  # following HALF belongs to the upcoming src, not cur
      if name in {'SRC1_R', 'SRC2_R', 'SRC3_R'} and num:
        srcs = [o for o in ops if o.slot != 'DST']
        idx = {'SRC1_R': 0, 'SRC2_R': 1, 'SRC3_R': 2}[name]
        if idx < len(srcs): srcs[idx].src_r = True
    elif name in {'DST', 'SRC1', 'SRC2', 'SRC3'}: start(name, num)
    elif name == 'SRC':
      if cur is None or cur.slot == 'DST': start('SRC', num)
    elif name == 'GPR': gpr = num
    elif name == 'CONST': const = num
    elif name == 'SWIZ': swiz = num
    elif name in {'RELATIVE', 'RELATIV'}: rel = bool(num)
    elif name in {'OFFSET', 'OFF', 'ARRAY_OFFSET'} and cur is not None: cur.rel_off = _i32(num) if num > 0x7fff else num
    elif name == 'HALF':
      pending_half = bool(num)
      if cur is not None and attach_half:
        cur.half, pending_half = pending_half, False
    elif cur is not None and name == 'ABSNEG': cur.absneg = num
    elif cur is not None and name == 'SRC_R' and num: cur.src_r = True
    elif cur is not None and name == 'IMMED': cur.kind, cur.imm, cur.packed = 'imm', num, num
  flush_ids()
  inst.dst = next((o for o in ops if o.slot == 'DST'), None)
  inst.srcs = [o for o in ops if o.slot != 'DST']
  # cat3 emits SRC2_R / SRC2_NEG *before* SRC2, so apply flags in a second pass
  by_slot = {o.slot: o for o in ops}
  for name, _, num in fields:
    if not num: continue
    if name in {'SRC1_R', 'SRC2_R', 'SRC3_R'} and name[:4] in by_slot: by_slot[name[:4]].src_r = True
    elif name in {'SRC1_NEG', 'SRC2_NEG', 'SRC3_NEG'} and name[:4] in by_slot: by_slot[name[:4]].absneg = 1
  # p0.x..p0.w encode as dest 0xf8..0xfb; a0.x is 0xf4 (regid 61)
  if inst.dst is not None and 248 <= inst.dst.packed <= 251: inst.dst.kind = 'pred'
  if inst.dst is not None and 244 <= inst.dst.packed <= 247: inst.dst.kind = 'a0'
  _patch_rel_from_line(inst)
  _patch_imm_from_line(inst)
  _patch_half_imm_from_line(inst)

_REL_RE = re.compile(r'^([rc])<a0\.[xyzw](?:\s*([+-])\s*(\d+))?>\.([xyzw])$')
_SWIZ = {'x': 0, 'y': 1, 'z': 2, 'w': 3}
_NEG_IMM_RE = re.compile(r',\s*(-\d+)(?:\s|,|$)')

def _patch_imm_from_line(inst:Inst):
  # Cat2 IMMED is 11-bit signed. Mesa's field callback reports the unsigned
  # encoding (e.g. 0x7e9 for -23), so match a negative decimal from the listing.
  if (m := _NEG_IMM_RE.search(inst.line)) is None: return
  want = int(m.group(1))
  enc = want & 0x7ff
  for op in (inst.dst, *inst.srcs):
    if op is not None and op.kind == 'imm' and (op.imm & 0x7ff) == enc:
      op.imm = want

def _patch_half_imm_from_line(inst:Inst):
  # Mesa prints half FLUT as h(0.0); the HALF bit is on the other src, not the imm.
  body = inst.line.split(']', 1)[-1].strip() if inst.line else ''
  while body.startswith('(') and ')' in body: body = body[body.index(')') + 1:].lstrip()
  toks = body.split()
  if len(toks) < 2: return
  parts = [p.strip() for p in ' '.join(toks[1:]).split(',') if p.strip()]
  ops: list[Operand] = []
  if inst.dst is not None: ops.append(inst.dst)
  ops.extend(inst.srcs)
  for op, part in zip(ops, parts):
    if 'h(' in part: op.half = True

def _patch_rel_from_line(inst:Inst):
  body = inst.line.split(']', 1)[-1].strip() if inst.line else ''
  while body.startswith('(') and ')' in body: body = body[body.index(')') + 1:].lstrip()
  toks = body.split()
  if len(toks) < 2: return
  parts = [p.strip() for p in ' '.join(toks[1:]).split(',') if p.strip()]
  ops: list[Operand] = []
  if inst.dst is not None: ops.append(inst.dst)
  elif parts and _REL_RE.match(parts[0]):
    inst.dst = Operand(slot='DST', kind='rel_gpr')
    ops.append(inst.dst)
  ops.extend(inst.srcs)
  for op, part in zip(ops, parts):
    m = _REL_RE.match(part)
    if not m: continue
    file, sign, off, sw = m.group(1), m.group(2), m.group(3), m.group(4)
    op.kind = 'rel_const' if file == 'c' else 'rel_gpr'
    op.rel_off = int((sign or '') + (off or '0')) if off else 0
    op.packed = _SWIZ[sw]

def _name_from_disasm(text:str) -> str:
  s = text
  if ':' in s and s.split(':', 1)[0].lstrip().startswith('l') and s.split(':', 1)[0].strip()[1:].isdigit():
    s = s.split(':', 1)[1]
  while s.startswith('(') and ')' in s: s = s[s.index(')') + 1:].lstrip()
  tok = s.split()[0] if s.split() else 'unknown'
  return tok.rstrip(':')

@lru_cache(maxsize=256)
def decode_shader(image:bytes) -> list[Inst]:
  raw = (ctypes.c_uint32 * (max(2, (len(image) + 3) // 4)))()
  ctypes.memmove(ctypes.addressof(raw), image, len(image))
  collected: list[tuple[list, str]] = []
  cur_fields: list[tuple[str,str|None,int]] = []
  @PRE_CB
  def pre(_data, n, instr):
    nonlocal cur_fields
    fst, snd = data64(ctypes.cast(instr, ctypes.POINTER(ctypes.c_uint64)).contents.value)
    cur_fields = []
    collected.append((cur_fields, f"{n:04} [{fst:08x}_{snd:08x}]"))
  @FIELD_CB
  def field(_cb, name, val):
    raw_name = ctypes.cast(name, ctypes.c_char_p).value
    raw_str = ctypes.cast(val.contents.str, ctypes.c_char_p).value if val.contents.str else None
    nm = (raw_name or b'').decode()
    s = raw_str.decode() if raw_str is not None else None
    cur_fields.append((nm, s, val.contents.num))
  unmatched: list[str] = []
  @NO_MATCH_CB
  def no_match(_out, words, count):
    vals = [words[i] for i in range(min(int(count), 4))]
    unmatched.append(' '.join(f"{v:08x}" for v in vals))
  with tempfile.TemporaryFile('w+') as tf:
    mesa_fp = ctypes.cast(fp:=libc.fdopen(tf.fileno(), b"w"), ctypes.POINTER(mesa.struct__IO_FILE))
    # branch_labels extra lines misalign field_cb entries with the listing (phantom unknown insts at lN:)
    opts = mesa.struct_isa_decode_options(gpu_id=630, show_errors=True, max_errors=0, branch_labels=False,
                                          field_cb=field, pre_instr_cb=pre, no_match_cb=no_match)
    mesa.ir3_isa_disasm(raw, len(image), mesa_fp, opts)
    libc.fflush(fp)
    tf.seek(0)
    text = tf.read()
  if unmatched: raise NotImplementedError(f"undecodable IR3 {unmatched[0]}")
  disasm_lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not (ln.strip().endswith(':') and ln.strip()[0] == 'l')]
  insts: list[Inst] = []
  for i, (fields, hdr) in enumerate(collected):
    if not fields: continue
    line = f"{hdr} {disasm_lines[i]}" if i < len(disasm_lines) else hdr
    kw = _kw_from_fields(fields)
    if kw['name'] == 'unknown' and i < len(disasm_lines): kw['name'] = _name_from_disasm(disasm_lines[i])
    # cat1 swz/gat/sct encode DST0/SRC0 instead of DST/SRC, so they otherwise look like typeless movs
    if 'dst0' in kw and i < len(disasm_lines):
      parsed = _name_from_disasm(disasm_lines[i])
      if parsed.startswith(('swz', 'gat', 'sct')): kw['name'] = parsed
    inst = Inst(name=kw['name'], line=line, repeat=int(kw.get('repeat') or 0), sat=bool(kw.get('sat')), jp=bool(kw.get('jp')),
                extra={k: v for k, v in kw.items() if k not in {'name', 'repeat', 'sat', 'jp', 'sy', 'ss', 'ul', 'ei', 'eq', 'nop', 'zero'}})
    _fix_gpr_packed(fields, inst)
    typ = inst.extra.get('type')
    if isinstance(typ, int) and typ in {0, 2, 4, 6, 7}:
      if inst.name.startswith('ld') and inst.dst is not None: inst.dst.half = True
      if inst.name.startswith('st') and inst.srcs: inst.srcs[-1].half = True
    insts.append(inst)
  if DEBUG >= 6:
    for ins in insts: print(ins.line)
  return insts

class Wave:
  def __init__(self, nlanes:int):
    self.nlanes, self.exec = nlanes, (1 << nlanes) - 1
    self.rf = (ctypes.c_uint32 * (256 * WAVE))()
    self.hr = (ctypes.c_uint16 * (256 * WAVE))()
    self.pred = [0] * WAVE
    self.a0 = [0] * WAVE
    self.pc = 0
    self.lane_pc = [0] * nlanes
    self.join: list[int] = []
    self.branch_taken: int|None = None
    self.branch_target = 0
  def _idx(self, packed:int, lane:int) -> int:
    if not 0 <= packed < 256: raise RuntimeError(f"IR3 GPR packed {packed} out of range at {_last_line}")
    return packed * WAVE + lane
  def read_gpr(self, packed:int, lane:int, half:bool) -> int:
    if half: return self.hr[self._idx(packed, lane)]
    return self.rf[self._idx(packed, lane)]
  def write_gpr(self, packed:int, lane:int, half:bool, val:int):
    if half: self.hr[self._idx(packed, lane)] = val & MASK16
    else: self.rf[self._idx(packed, lane)] = val & MASK32

class CSState:
  def __init__(self, const:list[int], lm:bytearray, pvt_base:int, pvt_stride:int, tex_base:int, uav_base:int,
               samp_base:int, border:int, mapped_size):
    self.const, self.lm, self.pvt_base, self.pvt_stride = const, lm, pvt_base, pvt_stride
    self.tex_base, self.uav_base, self.samp_base, self.border, self.mapped_size = tex_base, uav_base, samp_base, border, mapped_size

def _apply_absneg(val:int, absneg:int, flt:bool, half:bool) -> int:
  if not absneg: return val
  if flt:
    sign = 0x8000 if half else 0x80000000
    if absneg == 1: return val ^ sign          # neg
    if absneg == 2: return val & ~sign         # abs
    return val | sign                          # absneg
  bits = 16 if half else 32
  mask = (1 << bits) - 1
  iv = _sext(val, bits)
  if absneg == 1: iv = -iv
  elif absneg == 2: iv = abs(iv)
  else: iv = -abs(iv)
  return iv & mask

def _rel_packed(wave:Wave, op:Operand, lane:int, rpt:int) -> int:
  return ((wave.a0[lane] + op.rel_off) << 2) + (op.packed & 3) + (rpt if op.src_r else 0)

def _read_src(wave:Wave, st:CSState, op:Operand, lane:int, rpt:int, flt:bool=False) -> int:
  packed = op.packed + (rpt if op.src_r and op.kind != 'imm' else 0)
  if op.kind == 'imm':
    if flt and 0 <= op.imm < len(FLUT):
      val = _u16f(FLUT[op.imm]) if op.half else _u32(FLUT[op.imm])
      return _apply_absneg(val, op.absneg, True, op.half)
    return op.imm
  if op.kind == 'pred': return (wave.pred[lane] >> ((op.packed - 248) & 3)) & 1
  if op.kind == 'a0': return wave.a0[lane] & MASK32
  if op.kind in {'const', 'rel_const'}:
    if op.kind == 'rel_const': packed = _rel_packed(wave, op, lane, rpt)
    val = st.const[packed] if 0 <= packed < len(st.const) else 0
    if op.half:
      lo, hi = val & MASK16, val >> 16
      # Mesa often writes an f32 into a half-const slot (low 16 may be nonzero, e.g. 0x3faaa000 = 1.333)
      if flt and hi: val = _u16f(_f32(val))
      else: val = lo
    return _apply_absneg(val, op.absneg, flt, op.half)
  if op.kind == 'rel_gpr': packed = _rel_packed(wave, op, lane, rpt)
  val = wave.read_gpr(packed, lane, op.half)
  return _apply_absneg(val, op.absneg, flt, op.half)

def _write_dst(wave:Wave, op:Operand, lane:int, rpt:int, val:int, sat:bool, flt:bool, half:bool):
  if op.kind == 'pred':
    bit = (op.packed - 248) & 3
    wave.pred[lane] = (wave.pred[lane] & ~(1 << bit)) | ((val & 1) << bit)
    return
  h = half or op.half
  if op.kind == 'a0':
    wave.a0[lane] = _i16(val) if h else _i32(val)
    return
  packed = _rel_packed(wave, op, lane, rpt) if op.kind == 'rel_gpr' else op.packed + rpt
  if sat:
    if flt:
      f = _f16(val) if h else _f32(val)
      f = 0.0 if f < 0 else 1.0 if f > 1 else f
      val = _u16f(f) if h else _u32(f)
    else:
      bits = 16 if h else 32
      mx = (1 << (bits - 1)) - 1
      iv = _sext(val, bits)
      val = max(-mx - 1, min(mx, iv)) & ((1 << bits) - 1)
  wave.write_gpr(packed, lane, h, val)

def _is_float_op(name:str) -> bool:
  op, _, rest = name.partition('.')
  # cat4 SFUs are always float (names are `log2` / `hlog2`, not `log2.f`)
  if op.startswith('h') and op[1:] in {'rcp', 'rsq', 'log2', 'exp2', 'sin', 'cos', 'sqrt'}: return True
  if op in {'rcp', 'rsq', 'sqrt', 'sin', 'cos', 'log2', 'exp2'}: return True
  if op in {'add', 'sub', 'mul', 'mad', 'min', 'max', 'absneg', 'sel', 'cmps', 'cmpv', 'neg',
            'floor', 'ceil', 'trunc', 'rndne', 'rndaz'}:
    return rest.startswith('f')
  return False

def _src(wave, st, inst:Inst, i:int, lane:int, rpt:int, default=0, flt:bool=False) -> int:
  return _read_src(wave, st, inst.srcs[i], lane, rpt, flt) if i < len(inst.srcs) else default

def _src_half(inst:Inst, i:int, half_dst:bool) -> bool:
  return inst.srcs[i].half if i < len(inst.srcs) else False

def _fv(inst:Inst, srcs:list[int], i:int, half_dst:bool) -> float:
  v = srcs[i] if i < len(srcs) else 0
  return _f16(v) if _src_half(inst, i, half_dst) else _f32(v)

def _fenc(r, half:bool) -> int:
  if isinstance(r, float): return _u16f(r) if half else _u32(r)
  return int(r) & (MASK16 if half else MASK32)

def _cmp(a:int, b:int, cond:str, typ:str) -> int:
  if typ in {'f', 'f32', 'f16'}:
    av, bv = (_f16(a), _f16(b)) if typ == 'f16' else (_f32(a), _f32(b))
  elif typ in {'s', 's32', 's16'}:
    bits = 16 if '16' in typ else 32
    av, bv = _sext(a, bits), _sext(b, bits)
  else:
    bits = 16 if '16' in typ else 32
    av, bv = a & ((1 << bits) - 1), b & ((1 << bits) - 1)
  if cond == 'lt': r = av < bv
  elif cond == 'le': r = av <= bv
  elif cond == 'gt': r = av > bv
  elif cond == 'ge': r = av >= bv
  elif cond == 'eq': r = av == bv
  elif cond == 'ne': r = av != bv
  else: raise NotImplementedError(f"cmps cond {cond}")
  return 1 if r else 0

def _convert(val:int, st:str, dt:str) -> int:
  def to_py(v, t):
    if t.startswith('f16'): return _f16(v)
    if t.startswith('f'): return _f32(v)
    if t.startswith('s16'): return _i16(v)
    if t.startswith('s8'): return ctypes.c_int8(v & 0xff).value
    if t.startswith('s'): return _i32(v)
    if t.startswith('u8'): return v & 0xff
    if t.startswith('u16') or t.startswith('b16'): return v & MASK16
    return v & MASK32
  # IR3 cov from a narrow integer to a wider signed dest sign-extends (cov.u8s32, cov.u8s16, …)
  src_bits = 8 if st in {'u8', 's8'} else (16 if '16' in st else 32)
  dst_signed_bits = 16 if dt.startswith('s16') else (32 if dt.startswith('s') else 0)
  if st[0] in 'us' and dst_signed_bits and src_bits < dst_signed_bits: x = _sext(val, src_bits)
  else: x = to_py(val, st)
  if dt.startswith('f16'): return _u16f(float(x))
  if dt.startswith('f'): return _u32(float(x))
  if dt.startswith('s16') or dt.startswith('u16') or dt.startswith('b16'): return int(x) & MASK16
  if dt.startswith('s8') or dt.startswith('u8'): return int(x) & 0xff
  return int(x) & MASK32

def _mem_width(inst:Inst) -> int:
  # TYPE is the memory type; TYPE_HALF / HALF only marks the GPR file (stg.u8 hrN is still 1 byte)
  typ = inst.extra.get('type')
  if isinstance(typ, int): return {0: 2, 1: 4, 2: 2, 3: 4, 4: 2, 5: 4, 6: 1, 7: 1}.get(typ, 4)
  name = inst.name
  for tag, w in (('.u32', 4), ('.s32', 4), ('.f32', 4), ('.u16', 2), ('.s16', 2), ('.f16', 2), ('.u8', 1), ('.s8', 1), ('.b32', 4), ('.b16', 2)):
    if tag in name: return w
  if inst.extra.get('type_half'): return 2
  return 4

def _addr64(wave:Wave, st:CSState, op:Operand, lane:int, rpt:int, off:int=0) -> int:
  lo = _read_src(wave, st, op, lane, rpt) & MASK32
  hi_op = Operand(slot=op.slot, kind=op.kind, packed=op.packed + 1, half=False, src_r=op.src_r)
  hi = _read_src(wave, st, hi_op, lane, rpt) & MASK32
  return ((hi << 32) | lo) + off

def _check_host_addr(addr:int, n:int, mapped_size, what:str):
  # reject NULL-page and non-canonical 48-bit pointers before ctypes can SIGABRT
  if addr < 0x1000 or addr >= (1 << 48) or (addr + n) > (1 << 48):
    raise RuntimeError(f"IR3 {what} bad pointer {addr:#x} size {n}")
  if mapped_size is not None:
    avail = mapped_size(addr)
    if n > avail: raise RuntimeError(f"IR3 {what} OOB {addr:#x} size {n} avail {avail}")

def _load_bytes(addr:int, n:int, mapped_size) -> bytes:
  _check_host_addr(addr, n, mapped_size, 'load')
  return ctypes.string_at(addr, n)

def _store_bytes(addr:int, data:bytes, mapped_size):
  _check_host_addr(addr, len(data), mapped_size, 'store')
  ctypes.memmove(addr, data, len(data))

def _tex_field(val:int, name:str, fallback:int) -> int:
  mask, shift = getattr(mesa, f"{name}__MASK", None), getattr(mesa, f"{name}__SHIFT", None)
  return ((val & mask) >> shift) if mask is not None and shift is not None else fallback

def _image_load(st:CSState, desc_base:int, idx:int, x:int, y:int, half:bool) -> int:
  p = (ctypes.c_uint32 * 16).from_address(desc_base + idx * 0x40)
  fmt0, c1, c2 = p[0], p[1], p[2]
  width = _tex_field(c1, 'A6XX_TEX_CONST_1_WIDTH', c1 & 0x7fff)
  height = _tex_field(c1, 'A6XX_TEX_CONST_1_HEIGHT', (c1 >> 15) & 0x7fff)
  pitch = _tex_field(c2, 'A6XX_TEX_CONST_2_PITCH', (c2 >> 12) & 0x3fffff)
  addr = p[4] | (p[5] << 32)
  fmt = _tex_field(fmt0, 'A6XX_TEX_CONST_0_FMT', fmt0 & 0xff)
  bpp = 8
  if fmt == getattr(mesa, 'FMT6_16_16_16_16_FLOAT', -1): bpp = 8
  if fmt == getattr(mesa, 'FMT6_32_32_32_32_FLOAT', -2): bpp = 16
  if x < 0 or y < 0 or x >= width or y >= height: return 0
  off = y * (pitch or (width * bpp)) + x * bpp
  data = _load_bytes(addr + off, 2 if half else 4, st.mapped_size)
  return int.from_bytes(data[:4], 'little')

def _exec_alu(name:str, inst:Inst, srcs:list[int], half:bool) -> int:
  parts = name.split('.')
  op = parts[0]
  typ = parts[1] if len(parts) > 1 else 'u'
  cond = parts[2] if len(parts) > 2 else ''
  if op.startswith('h') and op[1:] in {'rcp', 'rsq', 'log2', 'exp2', 'sin', 'cos', 'sqrt'}:
    op, half = op[1:], True
  a = srcs[0] if srcs else 0
  b = srcs[1] if len(srcs) > 1 else 0
  c = srcs[2] if len(srcs) > 2 else 0
  # dest-half with a 32-bit src is still a 32-bit ALU (shr.b hr, r, 16 must not mask the shift to 4 bits)
  src_half = bool(inst.srcs) and all(s.half or s.kind == 'imm' for s in inst.srcs)
  bits = 16 if half and src_half else 32
  mask = (1 << bits) - 1
  sm = bits - 1
  fdst = half or typ == 'f16'
  if op in {'add', 'sub', 'mul', 'min', 'max'} and typ in {'f', 'f32', 'f16'}:
    fn = {'add': lambda x, y: x + y, 'sub': lambda x, y: x - y, 'mul': lambda x, y: x * y,
          'min': lambda x, y: min(x, y), 'max': lambda x, y: max(x, y)}[op]
    return _fenc(fn(_fv(inst, srcs, 0, fdst), _fv(inst, srcs, 1, fdst)), fdst)
  if op == 'add': return (a + b) & mask
  if op == 'sub': return (a - b) & mask
  if op == 'mul' and typ in {'u24'}: return ((a & 0xffffff) * (b & 0xffffff)) & mask
  if op == 'mul' and typ in {'s24'}: return (_sext(a, 24) * _sext(b, 24)) & mask
  if op == 'mul': return (a * b) & mask
  if op == 'mull': return ((a & MASK16) * (b & MASK16)) & MASK32
  if op == 'mad' and typ in {'f', 'f32', 'f16'}:
    return _fenc(_fv(inst, srcs, 0, fdst) * _fv(inst, srcs, 1, fdst) + _fv(inst, srcs, 2, fdst), fdst)
  if op == 'mad': return (a * b + c) & mask
  if op == 'madsh':
    # 32-bit mul expansion: mull.u dst, a, b; madsh.m16 dst, b, a, dst → dst += (a>>16)*b << 16
    if 'm16' in name: return (c + ((_i16(a) * _i16(b >> 16)) << 16)) & MASK32
    return (a + _i16(b) * _i16(c)) & MASK32
  if op == 'min' and typ.startswith('s'): return min(_sext(a, bits), _sext(b, bits)) & mask
  if op == 'min': return min(a & mask, b & mask)
  if op == 'max' and typ.startswith('s'): return max(_sext(a, bits), _sext(b, bits)) & mask
  if op == 'max': return max(a & mask, b & mask)
  if op == 'and': return (a & b) & mask
  if op == 'andg': return ((a & b) | c) & mask
  if op == 'or': return (a | b) & mask
  if op == 'xor': return (a ^ b) & mask
  if op == 'not': return (~a) & mask
  if op == 'shl': return (a << (b & sm)) & mask
  if op == 'shr': return ((a & mask) >> (b & sm)) & mask
  if op == 'ashr': return (_sext(a, bits) >> (b & sm)) & mask
  if op == 'shrg': return ((b >> (a & sm)) | c) & mask
  if op == 'shrm': return ((b >> (a & sm)) & c) & mask
  if op == 'shlg': return ((b << (a & sm)) | c) & mask
  if op == 'shlm': return ((b << (a & sm)) & c) & mask
  if op == 'cmps' or op == 'cmpv':
    # dest may be half even when srcs are f32 (bool kernel); width follows half srcs
    # e.g. cmps.f.lt hr0.x, h(0.0), (r)hr0.x — CMPLT(0, t) for f16 (imm HALF is on the GPR src)
    cmp_typ = typ
    if inst.srcs and any(s.half for s in inst.srcs):
      if typ in {'f', 'f32'}: cmp_typ = 'f16'
      elif typ in {'s', 's32'}: cmp_typ = 's16'
      elif typ in {'u', 'u32'}: cmp_typ = 'u16'
      elif typ in {'b', 'b32'}: cmp_typ = 'b16'
    r = _cmp(a, b, cond or 'eq', cmp_typ)
    return r if op == 'cmps' else (mask if r else 0)
  if op == 'sel':
    # sel dst, src0, src1, src2  →  src1 ? src0 : src2  (Mesa bcsel true, cond, false)
    return a if (b & mask) else c
  if op in {'sign', 'signp'}:
    x = _f16(a) if half else _f32(a)
    sf = 0.0 if x == 0 else (1.0 if x > 0 else -1.0)
    return _u16f(sf) if half else _u32(sf)
  if op == 'absneg':
    return a  # ABSNEG modifier already applied in _read_src
  if op == 'clz':
    v = a & mask
    return bits if v == 0 else bits - v.bit_length()
  if op == 'sad': return (abs(_sext(a, bits) - _sext(b, bits)) + c) & mask
  if op == 'rcp':
    x = _f16(a) if half else _f32(a)
    if x == 0: return _fenc(math.copysign(math.inf, x), half)
    return _fenc(1.0 / x, half)
  if op == 'rsq':
    x = _f16(a) if half else _f32(a)
    if x == 0: return _fenc(math.copysign(math.inf, x), half)
    rf: float = 1.0 / math.sqrt(x) if x > 0 else float('nan')
    return _fenc(rf, half)
  if op == 'sqrt':
    x = _f16(a) if half else _f32(a)
    rf = math.sqrt(x) if x >= 0 else float('nan')
    return _fenc(rf, half)
  if op == 'log2':
    x = _f16(a) if half else _f32(a)
    rf = math.log2(x) if x > 0 else float('-inf') if x == 0 else float('nan')
    return _fenc(rf, half)
  if op == 'exp2':
    x = _f16(a) if half else _f32(a)
    try: rf = math.pow(2.0, x)
    except (OverflowError, ValueError): rf = math.inf if x > 0 else 0.0
    return _fenc(rf, half)
  if op in {'sin', 'cos'}:
    x = _f16(a) if half else _f32(a)
    try: rf = math.sin(x) if op == 'sin' else math.cos(x)
    except ValueError: rf = float('nan')
    return _fenc(rf, half)
  if op in {'floor', 'ceil', 'trunc', 'rndne', 'rndaz'}:
    x = _fv(inst, srcs, 0, fdst)
    if math.isnan(x) or math.isinf(x): fr: float = x
    elif op == 'floor': fr = float(math.floor(x))
    elif op == 'ceil': fr = float(math.ceil(x))
    elif op == 'trunc': fr = float(math.trunc(x))
    elif op == 'rndaz': fr = math.copysign(math.floor(abs(x) + 0.5), x)
    else: fr = float(round(x))
    return _u16f(fr) if half else _u32(fr)
  if op == 'bfrev':
    bits, v, out = (16 if half else 32), a, 0
    for i in range(bits):
      if v & (1 << i): out |= 1 << (bits - 1 - i)
    return out
  if op == 'cbits': return (a & mask).bit_count()
  if op in {'mov', 'cov'} or name.startswith('mov.') or name.startswith('cov.'):
    # types encoded in name mov.f32f32 / cov.u16s32
    rest = name.split('.', 1)[1] if '.' in name else ''
    if len(rest) >= 4:
      styp, dtyp = rest[:3], rest[3:]
      if rest[0] in 'fsu' and rest[1:].split('s')[0]:
        # u16s32, f32f32, s32s16, u8u32 etc
        for sl in (3, 2):
          if rest[:sl] in TYPES.values() or rest[:sl] in {'f32', 'f16', 'u32', 'u16', 's32', 's16', 'u8', 's8', 'b32', 'b16'}:
            styp, dtyp = rest[:sl], rest[sl:]
            break
      return _convert(a, styp, dtyp)
    return a & mask
  raise NotImplementedError(name)

_last_line = ''
_last_insts: list[Inst] = []

def _pred_bit(wave:Wave, lane:int, extra:dict, which:int) -> int:
  bit = int(extra.get(f'comp{which}', 0) or 0) & 3
  v = (wave.pred[lane] >> bit) & 1
  if extra.get(f'inv{which}'): v ^= 1
  return v

def execute_inst(inst:Inst, wave:Wave, st:CSState, rpt:int):
  global _last_line
  _last_line = inst.line
  name, extra = inst.name, inst.extra
  half = bool(inst.dst.half) if inst.dst else False
  wave.branch_taken = None
  # (jp) reconverges getone/predt before this instruction runs (once, not per repeat)
  if rpt == 0 and inst.jp and wave.join: wave.exec = wave.join.pop() | wave.exec
  if name in {'nop', 'nop.s'}: return
  if name == 'end':
    wave.pc = -1
    return
  if name.startswith(('swz', 'gat', 'sct')):
    pairs = [(int(extra[f'dst{i}']), int(extra[f'src{i}'])) for i in range(4) if f'dst{i}' in extra and f'src{i}' in extra]
    # gat/sct often encode only DST0; remaining dests are consecutive packed ids
    if 'dst0' in extra:
      d0 = int(extra['dst0'])
      src_ids = [int(extra[f'src{i}']) for i in range(4) if f'src{i}' in extra]
      if src_ids and (not pairs or len(src_ids) > len(pairs)):
        pairs = [(d0 + i, s) for i, s in enumerate(src_ids)]
    if not pairs: raise NotImplementedError(inst.line)
    styp, dtyp = TYPES.get(int(extra.get('src_type', 3)), 'u32'), TYPES.get(int(extra.get('dst_type', 3)), 'u32')
    h = bool(extra.get('dst_half'))
    for lane in range(wave.nlanes):
      if not ((wave.exec >> lane) & 1): continue
      vals = []
      for _, s in pairs:
        v = wave.read_gpr(s, lane, h)
        vals.append(_convert(v, styp, dtyp) if styp != dtyp else v)
      for (d, _), v in zip(pairs, vals): wave.write_gpr(d, lane, h, v)
    return
  if name == 'mova' or name.startswith('mova'):
    for lane in range(wave.nlanes):
      if not ((wave.exec >> lane) & 1): continue
      src = _src(wave, st, inst, 0, lane, rpt)
      hsrc = bool(inst.srcs and inst.srcs[0].half)
      wave.a0[lane] = _i16(src) if hsrc else _i32(src)
    return
  if name in {'predt', 'predf', 'prede'}:
    pmask = sum((1 << i) for i in range(wave.nlanes) if wave.pred[i] & 1)
    if name == 'prede':
      if wave.join: wave.exec = wave.join.pop()
    elif name == 'predt':
      wave.join.append(wave.exec)
      wave.exec &= pmask
    else:
      # predf is the else of predt: use the saved exec, not the then-mask
      orig = wave.join[-1] if wave.join else wave.exec
      if not wave.join: wave.join.append(wave.exec)
      wave.exec = orig & ~pmask
    return
  if name in {'getone', 'getlast'}:
    wave.join.append(wave.exec)
    bits = wave.exec
    pick = (1 << (bits.bit_length() - 1) if bits else 0) if name == 'getlast' else (bits & -bits)
    wave.exec = pick & ((1 << wave.nlanes) - 1)
    return
  if name in {'br', 'jump', 'call'} or name.startswith('br'):
    off = extra.get('immed', inst.srcs[0].imm if inst.srcs and inst.srcs[0].kind == 'imm' else extra.get('off', 0))
    off = int(off)
    if off & 0x80000000: off = _i32(off)
    elif off > 0x7fff: off = _sext(off, 16)
    target = wave.pc + off  # cat0 IMMED is relative to the branch instruction, not PC+1
    if name == 'jump' or name == 'call':
      wave.pc, wave.branch_taken, wave.branch_target = target, wave.exec, target
      return
    taken = 0
    for lane in range(wave.nlanes):
      if not ((wave.exec >> lane) & 1): continue
      a = _pred_bit(wave, lane, extra, 1)
      if name == 'brao': cond = a | _pred_bit(wave, lane, extra, 2)
      elif name == 'braa': cond = a & _pred_bit(wave, lane, extra, 2)
      else: cond = a
      if cond: taken |= 1 << lane
    wave.branch_taken, wave.branch_target = taken, target
    # Do not mask exec here: divergent lanes keep their own PCs in run_wave.
    if taken: wave.pc = target
    return

  if name.startswith(('ldg', 'ldl', 'ldp', 'ldib', 'ldgb', 'ldc')):
    width, size = _mem_width(inst), int(extra.get('size') or 1)
    off = int(extra.get('off') or 0)
    for lane in range(wave.nlanes):
      if not ((wave.exec >> lane) & 1) or inst.dst is None: continue
      if name.startswith('ldl'):
        addr = (_src(wave, st, inst, 0, lane, rpt) + off) & 0x7fff
        data = bytes(st.lm[addr:addr + width * size])
      elif name.startswith('ldp'):
        addr = st.pvt_base + lane * st.pvt_stride + (_src(wave, st, inst, 0, lane, rpt) + off)
        data = _load_bytes(addr, width * size, st.mapped_size)
      else:
        addr = _addr64(wave, st, inst.srcs[0], lane, rpt, off)
        data = _load_bytes(addr, width * size, st.mapped_size)
      for k in range(size):
        chunk = data[k * width:(k + 1) * width].ljust(width, b'\x00')
        val = int.from_bytes(chunk, 'little')
        # cat6 16-bit types write the half GPR file even when DST_HALF is not in the listing fields
        dst = Operand(slot='DST', kind='gpr', packed=inst.dst.packed + k, half=width <= 2)
        _write_dst(wave, dst, lane, rpt, val, False, False, dst.half)
    return
  if name.startswith(('stg', 'stl', 'stp', 'stib', 'stgb')):
    width, size = _mem_width(inst), int(extra.get('size') or 1)
    off = int(extra.get('off') or 0)
    # stg: srcs[0]=addr, srcs[-1]=data.  stl/stp: dst=addr, srcs[0]=data
    local = name.startswith('stl') or name.startswith('stp')
    if local:
      addr_op, data_op = inst.dst, (inst.srcs[0] if inst.srcs else None)
    else:
      data_op = inst.srcs[-1] if inst.srcs else None
      addr_op = inst.srcs[0] if inst.srcs else None
    for lane in range(wave.nlanes):
      if not ((wave.exec >> lane) & 1) or data_op is None or addr_op is None: continue
      payload = b''
      for k in range(size):
        dop = Operand(slot=data_op.slot, kind=data_op.kind, packed=data_op.packed + k, half=data_op.half or width <= 2, src_r=data_op.src_r)
        payload += (_read_src(wave, st, dop, lane, rpt) & ((1 << (8 * width)) - 1)).to_bytes(width, 'little')
      if name.startswith('stl'):
        addr = (_read_src(wave, st, addr_op, lane, rpt) + off) & 0xffff
        if addr + len(payload) > len(st.lm):
          raise RuntimeError(f"IR3 stl OOB {addr:#x} size {len(payload)} lm {len(st.lm)}")
        st.lm[addr:addr + len(payload)] = payload
      elif name.startswith('stp'):
        addr = st.pvt_base + lane * st.pvt_stride + _read_src(wave, st, addr_op, lane, rpt) + off
        _store_bytes(addr, payload, st.mapped_size)
      else:
        addr = _addr64(wave, st, addr_op, lane, rpt, off)
        _store_bytes(addr, payload, st.mapped_size)
    return
  if name.startswith(('isam', 'sam')):
    tex = int(extra.get('tex') or extra.get('samp') or extra.get('s#') or 0)
    for lane in range(wave.nlanes):
      if not ((wave.exec >> lane) & 1) or inst.dst is None: continue
      x = _src(wave, st, inst, 0, lane, rpt)
      y = _src(wave, st, inst, 1, lane, rpt) if len(inst.srcs) > 1 else 0
      # coords may be in consecutive src of one collect — use src0 packed and packed+1
      if len(inst.srcs) == 1:
        y = wave.read_gpr(inst.srcs[0].packed + 1, lane, inst.srcs[0].half)
      val = _image_load(st, st.uav_base if 'ibo' in name else st.tex_base, tex, _i32(x), _i32(y), half)
      _write_dst(wave, inst.dst, lane, rpt, val, inst.sat, True, half)
    return
  if name in {'bar', 'fence', 'l2of', 'getsp'} or name.startswith('bar'): return
  if name.startswith('atomic') or name.startswith('atomg'):
    raise NotImplementedError(inst.line)

  flt = _is_float_op(name)
  for lane in range(wave.nlanes):
    if not ((wave.exec >> lane) & 1): continue
    srcs = [_src(wave, st, inst, i, lane, rpt, flt=flt) for i in range(len(inst.srcs))]
    try: val = _exec_alu(name, inst, srcs, half)
    except NotImplementedError: raise NotImplementedError(inst.line)
    if inst.dst is not None:
      _write_dst(wave, inst.dst, lane, rpt, val, inst.sat, flt, half)
    elif name.startswith('cmps'): wave.pred[lane] = val & 1

def _is_bar(inst:Inst) -> bool: return inst.name == 'bar' or inst.name.startswith('bar.')
def _is_branch(inst:Inst) -> bool: return inst.name in {'br', 'jump', 'call'} or inst.name.startswith('br')
def _is_pred_cf(inst:Inst) -> bool: return inst.name in {'predt', 'predf', 'prede', 'getone', 'getlast'}

def _lane_live(wave:Wave, insts:list[Inst], i:int) -> bool:
  pc = wave.lane_pc[i]
  return 0 <= pc < len(insts) and insts[pc].name != 'end'

def _min_live_pc(wave:Wave, insts:list[Inst]) -> int|None:
  pcs = [wave.lane_pc[i] for i in range(wave.nlanes) if _lane_live(wave, insts, i)]
  return min(pcs) if pcs else None

def _wave_at_bar(wave:Wave, insts:list[Inst]) -> bool:
  pcs = [wave.lane_pc[i] for i in range(wave.nlanes) if _lane_live(wave, insts, i)]
  return bool(pcs) and all(_is_bar(insts[p]) for p in pcs)

def _step_wave(insts:list[Inst], wave:Wave, st:CSState) -> str:
  """Execute the lowest-PC bundle of lanes. Returns the instruction line."""
  pc = _min_live_pc(wave, insts)
  if pc is None: return ''
  inst = insts[pc]
  at_mask = sum((1 << i) for i in range(wave.nlanes) if wave.lane_pc[i] == pc)
  saved = wave.exec
  wave.exec, wave.pc = saved & at_mask, pc
  if inst.name in {'nop', 'nop.s'} and not inst.jp:
    for i in range(wave.nlanes):
      if wave.lane_pc[i] == pc: wave.lane_pc[i] = pc + 1
    wave.exec, wave.pc = saved, pc + 1
    return inst.line
  for rpt in range(inst.repeat + 1): execute_inst(inst, wave, st, rpt)
  at_lanes = [i for i in range(wave.nlanes) if wave.lane_pc[i] == pc]
  if _is_branch(inst) and wave.branch_taken is not None:
    taken, target = wave.branch_taken, wave.branch_target
    active = saved & at_mask
    nt = active & ~taken
    if taken and not nt:
      for i in at_lanes: wave.lane_pc[i] = target
    elif nt and not taken:
      for i in at_lanes: wave.lane_pc[i] = pc + 1
    else:
      for i in at_lanes:
        wave.lane_pc[i] = target if (taken >> i) & 1 else pc + 1
  else:
    dest = -1 if inst.name == 'end' else (wave.pc if wave.pc != pc else pc + 1)
    for i in at_lanes: wave.lane_pc[i] = dest
  if not _is_pred_cf(inst): wave.exec = saved
  live = _min_live_pc(wave, insts)
  wave.pc = pc if live is None else live
  return inst.line

def run_wave(insts:list[Inst], wave:Wave, st:CSState):
  wave.pc, wave.lane_pc = 0, [0] * wave.nlanes
  n = 0
  while _min_live_pc(wave, insts) is not None:
    line = _step_wave(insts, wave, st)
    n += 1
    if n > 8_000_000: raise RuntimeError(f"IR3 infinite loop at {line}")

def run_workgroup(insts:list[Inst], waves:list[Wave], st:CSState):
  for w in waves:
    w.pc, w.lane_pc = 0, [0] * w.nlanes
  n = 0
  while True:
    active = [w for w in waves if _min_live_pc(w, insts) is not None]
    if not active: return
    at_bar = [w for w in active if _wave_at_bar(w, insts)]
    if at_bar and len(at_bar) == len(active):
      for w in at_bar:
        bpc = _min_live_pc(w, insts)
        if bpc is not None:
          for i in range(w.nlanes):
            if w.lane_pc[i] == bpc: w.lane_pc[i] = bpc + 1
          w.pc = bpc + 1
      continue
    runnable = [w for w in active if not _wave_at_bar(w, insts)]
    if not runnable:
      for w in at_bar:
        bpc = _min_live_pc(w, insts)
        if bpc is not None:
          for i in range(w.nlanes):
            if w.lane_pc[i] == bpc: w.lane_pc[i] = bpc + 1
      continue
    line = ''
    for w in runnable:
      mpc = _min_live_pc(w, insts)
      if mpc is not None and _is_bar(insts[mpc]): continue  # wait for the rest of this wave
      line = _step_wave(insts, w, st)
    n += 1
    if n > 8_000_000: raise RuntimeError(f"IR3 infinite loop at {line or 'bar'}")

def _u64_reg(regs:dict[int,int], lo:int) -> int: return (regs.get(lo, 0) & MASK32) | ((regs.get(lo + 1, 0) & MASK32) << 32)

def _field(val:int, name:str) -> int:
  return (val & getattr(mesa, f"{name}__MASK")) >> getattr(mesa, f"{name}__SHIFT")

def run_cs(gpu, groups:tuple[int,int,int]):
  regs = gpu.regs
  base = _u64_reg(regs, mesa.REG_A6XX_SP_CS_BASE)
  prg_off = regs.get(mesa.REG_A6XX_SP_CS_PROGRAM_COUNTER_OFFSET, 0)
  instr_size = regs.get(mesa.REG_A6XX_SP_CS_INSTR_SIZE, 1) * 128
  image = ctypes.string_at(base + prg_off, max(8, instr_size - prg_off))
  global _last_insts
  insts = decode_shader(image)
  _last_insts = insts
  nd = regs.get(mesa.REG_A6XX_SP_CS_NDRANGE_0, 0)
  lx = _field(nd, 'A6XX_SP_CS_NDRANGE_0_LOCALSIZEX') + 1
  ly = _field(nd, 'A6XX_SP_CS_NDRANGE_0_LOCALSIZEY') + 1
  lz = _field(nd, 'A6XX_SP_CS_NDRANGE_0_LOCALSIZEZ') + 1
  cfg = regs.get(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0, 0xfcfcfcfc)
  wgid = _field(cfg, 'A6XX_SP_CS_CONST_CONFIG_0_WGIDCONSTID')
  wgsz = _field(cfg, 'A6XX_SP_CS_CONST_CONFIG_0_WGSIZECONSTID')
  lid = _field(cfg, 'A6XX_SP_CS_CONST_CONFIG_0_LOCALIDREGID')
  wge = regs.get(mesa.REG_A6XX_SP_CS_WGE_CNTL, 0xfc)
  lin = _field(wge, 'A6XX_SP_CS_WGE_CNTL_LINEARLOCALIDREGID')
  const_addr = gpu.constants_addr
  if const_addr is None: raise RuntimeError("A630 kernel launch has no constant-buffer address")
  const = list((ctypes.c_uint32 * 1024).from_address(const_addr))
  shmem = (_field(regs.get(mesa.REG_A6XX_SP_CS_CNTL_1, 1), 'A6XX_SP_CS_CNTL_1_SHARED_SIZE') or 1) * 1024
  lm = bytearray(max(shmem, 32 * 1024))
  pvt_base = _u64_reg(regs, mesa.REG_A6XX_SP_CS_PVT_MEM_BASE)
  pvt_param = regs.get(mesa.REG_A6XX_SP_CS_PVT_MEM_PARAM, 0)
  pvt_stride = max(512, (_field(pvt_param, 'A6XX_SP_CS_PVT_MEM_PARAM_MEMSIZEPERITEM') or 1) * 512)
  tex = _u64_reg(regs, mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE)
  uav = _u64_reg(regs, mesa.REG_A6XX_SP_CS_UAV_BASE)
  samp = _u64_reg(regs, mesa.REG_A6XX_SP_CS_SAMPLER_BASE)
  border = _u64_reg(regs, mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE)
  st = CSState(const, lm, pvt_base, pvt_stride, tex, uav, samp, border, gpu._mapped_size)
  gx, gy, gz = groups
  nthreads = lx * ly * lz
  for z in range(gz):
    for y in range(gy):
      for x in range(gx):
        st.lm = bytearray(max(shmem, 32 * 1024))
        if wgid != 0xfc:
          for i, v in enumerate((x, y, z)):
            if wgid + i < len(st.const): st.const[wgid + i] = v
        if wgsz != 0xfc:
          for i, v in enumerate((lx, ly, lz)):
            if wgsz + i < len(st.const): st.const[wgsz + i] = v
        waves: list[Wave] = []
        for wv in range((nthreads + WAVE - 1) // WAVE):
          nlanes = min(WAVE, nthreads - wv * WAVE)
          wave = Wave(nlanes)
          for lane in range(nlanes):
            t = wv * WAVE + lane
            lx_, ly_, lz_ = t % lx, (t // lx) % ly, t // (lx * ly)
            if lid != 0xfc:
              wave.write_gpr(lid, lane, False, lx_)
              wave.write_gpr(lid + 1, lane, False, ly_)
              wave.write_gpr(lid + 2, lane, False, lz_)
            if lin != 0xfc: wave.write_gpr(lin, lane, False, t)
            # WGIDCONSTID is a packed id; a630 injects the workgroup id into that GPR (r48.x in the fill kernel)
            if wgid != 0xfc:
              wave.write_gpr(wgid, lane, False, x)
              wave.write_gpr(wgid + 1, lane, False, y)
              wave.write_gpr(wgid + 2, lane, False, z)
            if wgsz != 0xfc:
              wave.write_gpr(wgsz, lane, False, lx)
              wave.write_gpr(wgsz + 1, lane, False, ly)
              wave.write_gpr(wgsz + 2, lane, False, lz)
          waves.append(wave)
        run_workgroup(insts, waves, st)
