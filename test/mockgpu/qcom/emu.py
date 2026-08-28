from __future__ import annotations
import ctypes, functools, math, os, struct, tempfile
from dataclasses import dataclass
from typing import Any
from tinygrad.runtime.autogen import libc, mesa

def _field(fields, name):
  try: return next(value for field, value in fields if field == name)
  except StopIteration as exc:
    raise ValueError(f'missing IR3 field {name}') from exc

def _field_or(fields, name, default):
  return next((value for field, value in fields if field == name), default)

def _reject_unsupported_modifiers(fields, name):
  unsupported = {'UL', 'EI'}
  if bad := next((field for field, value in fields if field in unsupported and value), None):
    raise NotImplementedError(f'unsupported IR3 modifier {bad}')

def _reg(value, half=False):
  constant = bool(value & 0x1000)
  value &= 0xfff
  if not constant and value // 4 == 61: kind = 'a'
  elif not constant and value // 4 == 62: kind = 'p'
  else: kind = ('hc' if constant else 'hr') if half else ('c' if constant else 'r')
  return kind, value // 4, value % 4

def _src(value, half=False):
  mod = (value >> 14) & 3
  if value & 0x3800 == 0x2800:
    flut = (0.0, 0.5, 1.0, 2.0, math.e, math.pi, 1 / math.pi, 1 / math.log2(math.e), math.log2(math.e), 1 / math.log2(10), math.log2(10), 4.0)
    index = value & 0x3ff
    if index >= len(flut): raise ValueError(f'invalid IR3 float immediate {index}')
    return _float_bits(flut[index], half), mod
  if value & 0x2000:
    immediate = value & 0x7ff
    return immediate - 0x800 if immediate & 0x400 else immediate, mod
  return _reg(value, half), mod

def _cat1_src(fields, half=False):
  if any(field == 'IMMED' for field, _ in fields): return _field(fields, 'IMMED')
  if any(field == 'CONST' for field, _ in fields): return _reg(0x1000 | _field(fields, 'CONST'), half)
  if any(field == 'OFFSET' for field, _ in fields): return ('rel', _field(fields, 'OFFSET'), int(any(field == 'CONST' for field, _ in fields)))
  return _reg(_field(fields, 'SRC'), half)

@functools.cache
def _local_ids(local_size, register, order=(0, 1, 2)):
  x_size, y_size, z_size = local_size
  lanes = [(x, y, z) for z in range(z_size) for y in range(y_size) for x in range(x_size)]
  return {('r', register, component): [lane[axis] for lane in lanes] for component, axis in enumerate(order)}

def _group_ids(workgroup_id, lane_count, register):
  return {('r', register, component): [workgroup_id[component]] * lane_count for component in range(3)}

class _Const:
  def __init__(self, words):
    self.words, self._rows = tuple(words), {}

  def row(self, kind, index, lanes):
    by_lanes = self._rows.setdefault((kind, index), {})
    if (row := by_lanes.get(lanes)) is None:
      word = self.words[index]
      row = by_lanes[lanes] = [word if kind == 'c' else word & 0xffff] * lanes
    return row

class _Regs(dict):
  def __init__(self, lanes, initial=None, constants=None):
    super().__init__(initial if initial is not None else {})
    self.lanes, self.constants = lanes, constants

  def __missing__(self, key):
    kind, number, component = key
    if self.constants is not None and kind in ('c', 'hc'):
      index = number * 4 + component
      if 0 <= index < len(self.constants.words):
        self[key] = row = self.constants.row(kind, index, self.lanes)
        return row
    raise KeyError(key)

  def get(self, key, default=None):
    try: return self[key]
    except KeyError: return default

def _lane_count(regs):
  if isinstance(regs, _Regs): return regs.lanes
  return len(next(iter(regs.values()))) if regs else 0

def _next_reg(reg):
  kind, number, component = reg
  return kind, number + (component == 3), (component + 1) % 4

def _reg_offset(reg, offset):
  kind, number, component = reg
  return kind, number + (component + offset) // 4, (component + offset) % 4

def _values(regs, src, lanes):
  if isinstance(src, int): return [src] * lanes
  values = regs.get(src)
  return [0] * lanes if values is None else values

def _float_values(regs, src, lanes, half):
  values = _values(regs, src, lanes)
  # Half ALU converts full-float constants; integer ops use the raw bits.
  if half and isinstance(src, tuple) and src[0] == 'hc' and (full := regs.get(('c', src[1], src[2]))) is not None:
    return [_float_bits(_float(value), True) if value >> 16 else value & 0xffff for value in full]
  return values
def _write(regs, dst, values, mask):
  if all(mask):
    regs[dst] = values
    return
  previous = regs.get(dst)
  regs[dst] = [value if active else old for value, old, active in
               zip(values, [0] * len(values) if previous is None else previous, mask, strict=True)]
def _s32(value): return ctypes.c_int32(value).value
def _signed(value, half=False): return ctypes.c_int16(value).value if half else _s32(value)
def _compare(lhs, rhs, condition):
  return (lhs < rhs, lhs <= rhs, lhs > rhs, lhs >= rhs, lhs == rhs, lhs != rhs)[condition]

def _float(value, half=False):
  mask, size, fmt = (0xffff, 2, '<e') if half else (0xffffffff, 4, '<f')
  return struct.unpack(fmt, (value & mask).to_bytes(size, 'little'))[0]
def _float_bits(value, half=False):
  try: return int.from_bytes(struct.pack('<e' if half else '<f', value), 'little')
  except OverflowError: return 0x7c00 if half and value > 0 else 0xfc00 if half else 0x7f800000 if value > 0 else 0xff800000

def _fma(a, b, c): return a * b + c

def _mod_float(value, modifier, half):
  value = _float(value, half)
  return -abs(value) if modifier == 3 else abs(value) if modifier == 2 else -value if modifier == 1 else value

def _mod_int(value, modifier, half=False):
  value = _signed(value, half)
  return -abs(value) if modifier == 3 else abs(value) if modifier == 2 else -value if modifier == 1 else value

def _typed_value(value, typ):
  if typ == mesa.TYPE_F16: return _float(value, True)
  if typ == mesa.TYPE_F32: return _float(value)
  if typ == mesa.TYPE_U16: return value & 0xffff
  if typ == mesa.TYPE_S16: return ctypes.c_int16(value).value
  if typ == mesa.TYPE_S32: return _s32(value)
  if typ in (mesa.TYPE_U8, mesa.TYPE_U8_32): return value & 0xff
  return value & 0xffffffff

def _convert(value, src_type, dst_type, rounding):
  if src_type == dst_type:
    return value & (0xffff if dst_type in (mesa.TYPE_F16, mesa.TYPE_U16, mesa.TYPE_S16, mesa.TYPE_U8) else 0xffffffff)
  # IR3 encodes signed 8-bit widening as U8 to S16/S32.
  value = ctypes.c_int8(value).value if src_type == mesa.TYPE_U8 and dst_type in (mesa.TYPE_S16, mesa.TYPE_S32) else \
    _typed_value(value, src_type)
  if dst_type == mesa.TYPE_F16: return _float_bits(float(value), True)
  if dst_type == mesa.TYPE_F32: return _float_bits(float(value))
  if isinstance(value, float):
    if not math.isfinite(value): value = 0
    elif rounding == 1: value = round(value)
    elif rounding == 2: value = math.ceil(value)
    elif rounding == 3: value = math.floor(value)
    else: value = math.trunc(value)
  width = 16 if dst_type in (mesa.TYPE_U16, mesa.TYPE_S16) else 8 if dst_type in (mesa.TYPE_U8, mesa.TYPE_U8_32) else 32
  return int(value) & ((1 << width) - 1)

def _itemsize(typ): return 2 if typ in (mesa.TYPE_F16, mesa.TYPE_U16, mesa.TYPE_S16) else 1 if typ in (mesa.TYPE_U8, mesa.TYPE_U8_32) else 4

@functools.cache
def _alu_runner(name, condition, modifiers, source_half, dest_half):
  float_op = name.endswith('.f') or name in {'rcp', 'rsq', 'log2', 'exp2', 'sin', 'cos', 'sqrt', 'hrsq', 'hlog2', 'hexp2'}
  def run(values):
    v = tuple(_mod_float(x, m, source_half) if float_op else _mod_int(x, m, source_half) if m else x
              for x, m in zip(values, modifiers, strict=True))
    if name in {'add.f', 'mul.f', 'min.f', 'max.f'}:
      if name == 'add.f': out = v[0] + v[1]
      elif name == 'mul.f': out = v[0] * v[1]
      elif math.isnan(v[0]): out = v[1]
      elif math.isnan(v[1]): out = v[0]
      else: out = min(v) if name == 'min.f' else max(v)
      return _float_bits(out, dest_half)
    if name in {'cmps.f', 'cmpv.f'}: return int(_compare(v[0], v[1], condition))
    if name == 'sign.f': return _float_bits(-1.0 if v[0] < 0 else 1.0 if v[0] > 0 else v[0], dest_half)
    if name == 'absneg.f': return _float_bits(v[0], dest_half)
    if name in {'floor.f', 'ceil.f', 'rndne.f', 'rndaz.f', 'trunc.f'}:
      if not math.isfinite(v[0]): return _float_bits(v[0], dest_half)
      if name == 'floor.f': rounded = math.floor(v[0])
      elif name == 'ceil.f': rounded = math.ceil(v[0])
      elif name == 'rndne.f': rounded = round(v[0])
      elif name == 'rndaz.f': rounded = math.copysign(math.ceil(abs(v[0])), v[0])
      else: rounded = math.trunc(v[0])
      return _float_bits(float(rounded), dest_half)
    if name in {'rcp', 'rsq', 'log2', 'exp2', 'sin', 'cos', 'sqrt', 'hrsq', 'hlog2', 'hexp2'}:
      try:
        if name == 'rcp': out = 1.0 / v[0]
        elif name in {'rsq', 'hrsq'}: out = 1.0 / math.sqrt(v[0])
        elif name in {'log2', 'hlog2'}: out = math.log2(v[0])
        elif name in {'exp2', 'hexp2'}: out = 2.0 ** v[0]
        elif name == 'sin': out = math.sin(v[0])
        elif name == 'cos': out = math.cos(v[0])
        else: out = math.sqrt(v[0])
      except (OverflowError, ValueError, ZeroDivisionError):
        if name in {'log2', 'hlog2'}: out = -math.inf if v[0] == 0 else math.nan
        elif name == 'rcp': out = math.copysign(math.inf, v[0])
        elif name in {'rsq', 'hrsq'}: out = math.inf if v[0] == 0 else math.nan
        elif name in {'exp2', 'hexp2'}: out = math.inf if v[0] > 0 else 0.0
        else: out = math.nan
      return _float_bits(out, dest_half)
    a, b = (v + (0,))[:2]
    if name in {'add.u', 'add.s'}: out = a + b
    elif name in {'sub.u', 'sub.s'}: out = a - b
    elif name in {'cmps.u', 'cmpv.u'}: return int(_compare(a & 0xffffffff, b & 0xffffffff, condition))
    elif name in {'cmps.s', 'cmpv.s'}: return int(_compare(_signed(a, source_half), _signed(b, source_half), condition))
    elif name == 'min.u': out = min(a & 0xffffffff, b & 0xffffffff)
    elif name == 'max.u': out = max(a & 0xffffffff, b & 0xffffffff)
    elif name == 'min.s': out = min(_signed(a, source_half), _signed(b, source_half))
    elif name == 'max.s': out = max(_signed(a, source_half), _signed(b, source_half))
    elif name == 'absneg.s': out = a
    elif name == 'and.b': out = a & b
    elif name == 'or.b': out = a | b
    elif name == 'xor.b': out = a ^ b
    elif name == 'not.b': out = ~a
    elif name == 'mul.u24':
      bits = 16 if source_half else 24
      out = (a & ((1 << bits) - 1)) * (b & ((1 << bits) - 1))
    elif name == 'mul.s24':
      bits, sign = (16, 1 << 15) if source_half else (24, 1 << 23)
      a, b = a & ((1 << bits) - 1), b & ((1 << bits) - 1)
      out = (a - (1 << bits) if a & sign else a) * (b - (1 << bits) if b & sign else b)
    elif name == 'mull.u': out = (a & 0xffff) * (b & 0xffff)
    elif name == 'bfrev.b': out = int(f'{a & 0xffffffff:032b}'[::-1], 2)
    elif name == 'clz.s': out = 32 if a == 0 else 31 - int(math.log2(abs(_s32(a))))
    elif name == 'clz.b': out = 0xffffffff if not a & 0xffffffff else 32 - (a & 0xffffffff).bit_length()
    elif name == 'cbits.b': out = (a & 0xffffffff).bit_count()
    elif name == 'shl.b': out = a << (b & 31)
    elif name == 'shr.b': out = (a & 0xffffffff) >> (b & 31)
    elif name == 'ashr.b': out = _signed(a, source_half) >> (b & 31)
    elif name == 'getbit.b': out = (a >> (b & 31)) & 1
    else: raise NotImplementedError(f'unsupported IR3 ALU {name}')
    return out & (0xffff if dest_half else 0xffffffff)
  return run

def _memory_offset(fields):
  offset = _field(fields, 'OFF')
  return offset - 0x2000 if offset & 0x1000 else offset

@dataclass(frozen=True)
class _Inst:
  name: str
  dst: Any
  srcs: tuple[Any, ...]
  sy: bool
  nop: int
  repeat: int = 0
  repeat_srcs: tuple[bool, ...] = ()
  src_mods: tuple[int, ...] = ()
  condition: int = 0
  types: tuple[int, int] = (mesa.TYPE_U32, mesa.TYPE_U32)
  sat: bool = False
  rounding: int = 0
  branch_offset: int = 0
  invert: bool = False
  source_half: bool = False
  inverts: tuple[bool, ...] = ()
def _decode_fields(fields):
  name = next((value for field, value in fields if field == 'NAME'), None)
  _reject_unsupported_modifiers(fields, name)
  if name is None and any(field == 'DST0' for field, _ in fields):
    src_type, dst_type = _field(fields, 'SRC_TYPE'), _field(fields, 'DST_TYPE')
    src_half, dst_half = src_type in (mesa.TYPE_F16, mesa.TYPE_U16, mesa.TYPE_S16, mesa.TYPE_U8), \
      dst_type in (mesa.TYPE_F16, mesa.TYPE_U16, mesa.TYPE_S16, mesa.TYPE_U8)
    if any(field == 'SRC2' for field, _ in fields):
      name, dsts, srcs = 'gat', tuple(_reg_offset(_reg(_field(fields, 'DST0'), dst_half), i) for i in range(4)), \
        tuple(_reg(_field(fields, f'SRC{i}'), src_half) for i in range(4))
    elif any(field == 'DST2' for field, _ in fields):
      name, dsts, srcs = 'sct', tuple(_reg(_field(fields, f'DST{i}'), dst_half) for i in range(4)), \
        tuple(_reg_offset(_reg(_field(fields, 'SRC0'), src_half), i) for i in range(4))
    else:
      name, dsts, srcs = 'swz', tuple(_reg(_field(fields, f'DST{i}'), dst_half) for i in range(2)), \
        tuple(_reg(_field(fields, f'SRC{i}'), src_half) for i in range(2))
    return _Inst(name, dsts, srcs, bool(_field(fields, 'SY')), 0, types=(src_type, dst_type), rounding=_field_or(fields, 'ROUND', 0))
  if name is None:
    if any(field == 'INVOCATION' for field, _ in fields) or \
       (_field_or(fields, 'RAW_BITS', 0) & (1 << 31) and not any(field == 'IMMED' for field, _ in fields)):
      raise NotImplementedError('unsupported IR3 movs broadcast')
    types = (_field(fields, 'SRC_TYPE'), _field(fields, 'DST_TYPE'))
    src_half, dst_half = bool(_field_or(fields, 'HALF', 0)), bool(_field_or(fields, 'DST_HALF', 0))
    relative_dst = any(field == 'OFFSET' for field, _ in fields) and sum(field == 'DST' for field, _ in fields) == 1
    if relative_dst:
      if any(field in {'IMMED', 'CONST'} for field, _ in fields): raise NotImplementedError('unsupported relative IR3 mov source')
      dst, src = (('relhr' if dst_half else 'relr'), _field(fields, 'OFFSET'), 0), _reg(_field(fields, 'SRC'), src_half)
    else: dst, src = _reg(_field(fields, 'DST'), dst_half), _cat1_src(fields, src_half)
    return _Inst('mov', dst, (src,), bool(_field(fields, 'SY')), 0, _field_or(fields, 'REPEAT', 0),
      (bool(_field_or(fields, 'SRC_R', 0)),), types=types, rounding=_field_or(fields, 'ROUND', 0),
      source_half=src_half)
  if name == 'nop':
    return _Inst(name, None, (), bool(_field(fields, 'SY')), 0)
  if name == 'end':
    return _Inst(name, None, (), bool(_field(fields, 'SY')), 0)
  if name in {'jump', 'br', 'bany', 'ball'}:
    srcs = () if name == 'jump' else ((_reg(248 + _field(fields, 'COMP1')),))
    return _Inst(name, None, srcs, bool(_field(fields, 'SY')), 0, branch_offset=ctypes.c_int32(_field(fields, 'IMMED')).value,
                          invert=bool(_field_or(fields, 'INV1', 0)))
  if name in {'brao', 'braa'}:
    srcs = tuple(_reg(248 + _field(fields, f'COMP{i}')) for i in (1, 2))
    return _Inst(name, None, srcs, bool(_field(fields, 'SY')), 0,
      branch_offset=ctypes.c_int32(_field(fields, 'IMMED')).value,
      inverts=tuple(bool(_field_or(fields, f'INV{i}', 0)) for i in (1, 2)))
  if name in {'predt', 'predf', 'prede'}:
    return _Inst(name, None, (), bool(_field(fields, 'SY')), 0)
  if name in {'bar', 'fence'}:
    return _Inst(name, None, (), bool(_field(fields, 'SY')), 0)
  if name in ('ashr.b', 'shl.b'):
    srcs = tuple(_src(_field(fields, src), bool(_field_or(fields, 'HALF', 0))) for src in ('SRC1', 'SRC2'))
    return _Inst(name, _reg(_field(fields, 'DST'), bool(_field_or(fields, 'DST_HALF', 0))), tuple(x[0] for x in srcs),
      bool(_field(fields, 'SY')), _field_or(fields, 'NOP', 0), _field_or(fields, 'REPEAT', 0),
      tuple(bool(value) for field, value in fields if field == 'SRC_R')[:2], tuple(x[1] for x in srcs),
      source_half=bool(_field_or(fields, 'HALF', 0)))
  if name == 'shrg':
    half = bool(_field_or(fields, 'HALF', 0))
    def alt_src(value): return value & 0xfff if value & 0x1000 else _src(value, half)[0]
    srcs = (alt_src(_field(fields, 'SRC1')), _src(_field(fields, 'SRC2'), half)[0], alt_src(_field(fields, 'SRC3')))
    return _Inst(name, _reg(_field(fields, 'DST'), bool(_field_or(fields, 'DST_HALF', 0))), srcs,
      bool(_field(fields, 'SY')), _field_or(fields, 'NOP', 0), _field_or(fields, 'REPEAT', 0),
      tuple(bool(_field_or(fields, f'SRC{i}_R', 0)) for i in range(1, 4)), source_half=half)
  if name == 'add.u':
    srcs = tuple(_src(_field(fields, src), bool(_field_or(fields, 'HALF', 0))) for src in ('SRC1', 'SRC2'))
    return _Inst(name, _reg(_field(fields, 'DST'), bool(_field_or(fields, 'DST_HALF', 0))), tuple(x[0] for x in srcs),
      bool(_field(fields, 'SY')), _field_or(fields, 'NOP', 0), _field_or(fields, 'REPEAT', 0),
      tuple(bool(value) for field, value in fields if field == 'SRC_R')[:2], tuple(x[1] for x in srcs),
      sat=bool(_field_or(fields, 'SAT', 0)), source_half=bool(_field_or(fields, 'HALF', 0)))
  if name == 'cmps.u':
    srcs_mods = tuple(_src(_field(fields, src), bool(_field_or(fields, 'HALF', 0))) for src in ('SRC1', 'SRC2'))
    return _Inst(name, _reg(_field(fields, 'DST'), bool(_field(fields, 'DST_HALF'))), tuple(x[0] for x in srcs_mods),
      bool(_field(fields, 'SY')), 0, _field_or(fields, 'REPEAT', 0), tuple(bool(value) for field, value in fields if field == 'SRC_R'),
      tuple(x[1] for x in srcs_mods), _field(fields, 'COND'), source_half=bool(_field_or(fields, 'HALF', 0)))
  if name == 'cmps.s':
    srcs_mods = tuple(_src(_field(fields, src), bool(_field_or(fields, 'HALF', 0))) for src in ('SRC1', 'SRC2'))
    return _Inst(name, _reg(_field(fields, 'DST'), bool(_field(fields, 'DST_HALF'))), tuple(x[0] for x in srcs_mods),
      bool(_field(fields, 'SY')), 0, _field_or(fields, 'REPEAT', 0), tuple(bool(value) for field, value in fields if field == 'SRC_R'),
      tuple(x[1] for x in srcs_mods), _field(fields, 'COND'), source_half=bool(_field_or(fields, 'HALF', 0)))
  if name == 'absneg.s':
    src, mod = _src(_field(fields, 'SRC1'), bool(_field_or(fields, 'HALF', 0)))
    return _Inst(name, _reg(_field(fields, 'DST'), bool(_field(fields, 'DST_HALF'))), (src,), bool(_field(fields, 'SY')), 0,
                          _field_or(fields, 'REPEAT', 0), (bool(_field_or(fields, 'SRC_R', 0)),), (mod,),
                          source_half=bool(_field_or(fields, 'HALF', 0)))
  if name == 'sel.b32':
    srcs = tuple(_src(_field(fields, src), bool(_field_or(fields, 'HALF', 0))) for src in ('SRC1', 'SRC2', 'SRC3'))
    return _Inst(name, _reg(_field(fields, 'DST')), tuple(x[0] for x in srcs), bool(_field(fields, 'SY')), 0,
      _field_or(fields, 'REPEAT', 0), tuple(bool(_field_or(fields, f'SRC{i}_R', 0)) for i in range(1, 4)), tuple(x[1] for x in srcs))
  if name == 'ldg':
    typ = _field(fields, 'TYPE')
    return _Inst('ldg', _reg(_field(fields, 'DST'), bool(_field_or(fields, 'TYPE_HALF', 0))),
      (_reg(_field(fields, 'SRC1')), _memory_offset(fields), _field(fields, 'SIZE')),
      bool(_field(fields, 'SY')), 0, types=(typ, typ))
  if name == 'ldg.a':
    typ = _field(fields, 'TYPE')
    return _Inst(name, _reg(_field(fields, 'DST'), bool(_field_or(fields, 'TYPE_HALF', 0))),
      (_reg(_field(fields, 'SRC1')), _reg(_field(fields, 'SRC2')), _field(fields, 'FULL_SHIFT'), _field(fields, 'SIZE')),
      bool(_field(fields, 'SY')), 0, types=(typ, typ))
  if name == 'stg':
    typ = _field(fields, 'TYPE')
    return _Inst('stg', None, (_reg(_field(fields, 'SRC1')), _reg(_field(fields, 'SRC3'), bool(_field_or(fields, 'TYPE_HALF', 0))),
      _memory_offset(fields), _field(fields, 'SIZE')), bool(_field(fields, 'SY')), 0, types=(typ, typ))
  if name == 'stg.a':
    typ = _field(fields, 'TYPE')
    return _Inst(name, None, (_reg(_field(fields, 'SRC1')), _reg(_field(fields, 'SRC2')),
      _field(fields, 'FULL_SHIFT'), _reg(_field(fields, 'SRC3'), bool(_field_or(fields, 'TYPE_HALF', 0))), _field(fields, 'SIZE')),
      bool(_field(fields, 'SY')), 0, types=(typ, typ))
  if name == 'stib.b':
    if not _field_or(fields, 'TYPED', 0) or _field_or(fields, 'MODE', 0):
      raise NotImplementedError('unsupported IR3 stib mode')
    typ, components, dimensions = _field(fields, 'TYPE'), _field(fields, 'TYPE_SIZE'), _field(fields, 'D')
    return _Inst('stib', None, (_reg(_field(fields, 'SRC2')), _reg(_field(fields, 'SRC1'),
      bool(_field_or(fields, 'TYPE_HALF', 0))), _field(fields, 'OFFSET'), _field(fields, 'SSBO'), dimensions, components),
      bool(_field(fields, 'SY')), 0, types=(typ, typ))
  if name == 'isam':
    if _field_or(fields, '3D', 0) or _field_or(fields, 'A', 0) or _field_or(fields, 'O', 0) or _field_or(fields, 'P', 0) or \
       _field_or(fields, 'SV', 0): raise NotImplementedError('unsupported IR3 isam modifier')
    typ, one_dimensional = _field(fields, 'TYPE'), not bool(_field_or(fields, '1D', 0))
    return _Inst(name, _reg(_field(fields, 'DST'), bool(_field_or(fields, 'DST_HALF', 0))),
      (_reg(_field(fields, 'SRC')), _field(fields, 'SAMP'), _field(fields, 'TEX'), 1 if one_dimensional else 2,
       _field(fields, 'WRMASK')), bool(_field(fields, 'SY')), 0, types=(typ, typ))
  if name in {'ldl', 'ldp'}:
    typ = _field(fields, 'TYPE')
    return _Inst(name, _reg(_field(fields, 'DST'), typ in (mesa.TYPE_F16, mesa.TYPE_U16, mesa.TYPE_S16, mesa.TYPE_U8)),
      (_reg(_field(fields, 'SRC')), _memory_offset(fields), _field(fields, 'SIZE')), bool(_field(fields, 'SY')), 0, types=(typ, typ))
  if name in {'stl', 'stp'}:
    typ = _field(fields, 'TYPE')
    return _Inst(name, None, (_reg(_field(fields, 'DST')), _reg(_field(fields, 'SRC'),
      typ in (mesa.TYPE_F16, mesa.TYPE_U16, mesa.TYPE_S16, mesa.TYPE_U8)), _memory_offset(fields), _field(fields, 'SIZE')),
      bool(_field(fields, 'SY')), 0, types=(typ, typ))
  if name.startswith('atomic.g.'):
    if _field_or(fields, 'TYPED', 0): raise NotImplementedError(f'unsupported typed IR3 atomic {name}')
    if _field(fields, 'TYPE_SIZE') != 1: raise NotImplementedError(f'unsupported vector IR3 atomic {name}')
    typ = _field(fields, 'TYPE')
    srcs = tuple(_reg(_field(fields, src)) for src in ('SRC1', 'SRC2', 'SRC3') if any(field == src for field, _ in fields))
    atomic_dst = _reg(_field(fields, 'DST')) if _field_or(fields, 'D', 0) else None
    return _Inst(name, atomic_dst, srcs, bool(_field(fields, 'SY')), 0, types=(typ, typ))
  if name == 'rcp':
    src, mod = _src(_field(fields, 'SRC'), bool(_field_or(fields, 'HALF', 0)))
    return _Inst(name, _reg(_field(fields, 'DST'), bool(_field_or(fields, 'DST_HALF', 0))), (src,),
      bool(_field(fields, 'SY')), 0, _field_or(fields, 'REPEAT', 0), (bool(_field_or(fields, 'SRC_R', 0)),), (mod,),
      source_half=bool(_field_or(fields, 'HALF', 0)))
  cat2_one = {'sign.f', 'absneg.f', 'floor.f', 'ceil.f', 'rndne.f', 'rndaz.f', 'trunc.f', 'absneg.s', 'not.b',
              'bfrev.b', 'clz.s', 'clz.b', 'setrm', 'cbits.b'}
  cat2_two = {'add.f', 'min.f', 'max.f', 'mul.f', 'cmps.f', 'cmpv.f', 'add.u', 'add.s', 'sub.u', 'sub.s', 'cmps.u', 'cmps.s',
              'min.u', 'min.s', 'max.u', 'max.s', 'and.b', 'or.b', 'xor.b', 'cmpv.u', 'cmpv.s', 'mul.u24', 'mul.s24', 'mull.u',
              'shl.b', 'shr.b', 'ashr.b', 'mgen.b', 'getbit.b', 'shb', 'msad'}
  if name in cat2_one | cat2_two:
    keys = ('SRC1',) if name in cat2_one else ('SRC1', 'SRC2')
    srcs = tuple(_src(_field(fields, src), bool(_field_or(fields, 'HALF', 0))) for src in keys)
    repeats = tuple(bool(value) for field, value in fields if field == 'SRC_R')[:len(keys)]
    condition = _field_or(fields, 'COND', 0)
    return _Inst(name, _reg(_field(fields, 'DST'), bool(_field_or(fields, 'DST_HALF', 0))), tuple(x[0] for x in srcs),
      bool(_field(fields, 'SY')), _field_or(fields, 'NOP', 0), _field_or(fields, 'REPEAT', 0), repeats,
      tuple(x[1] for x in srcs), condition, sat=bool(_field_or(fields, 'SAT', 0)), source_half=bool(_field_or(fields, 'HALF', 0)))
  if name in {'rsq', 'log2', 'exp2', 'sin', 'cos', 'sqrt', 'hrsq', 'hlog2', 'hexp2'}:
    src, mod = _src(_field(fields, 'SRC'), bool(_field_or(fields, 'HALF', 0)))
    return _Inst(name, _reg(_field(fields, 'DST'), bool(_field_or(fields, 'DST_HALF', 0))), (src,), bool(_field(fields, 'SY')), 0,
      _field_or(fields, 'REPEAT', 0), (bool(_field_or(fields, 'SRC_R', 0)),), (mod,), source_half=bool(_field_or(fields, 'HALF', 0)))
  cat3 = {'mad.u16', 'madsh.u16', 'mad.s16', 'madsh.m16', 'mad.u24', 'mad.s24', 'mad.f16', 'mad.f32',
          'sel.b16', 'sel.b32', 'sel.s16', 'sel.s32', 'sel.f16', 'sel.f32', 'sad.s16', 'sad.s32'}
  if name in cat3:
    srcs = tuple(_src(_field(fields, src), bool(_field_or(fields, 'HALF', 0))) for src in ('SRC1', 'SRC2', 'SRC3'))
    mods = tuple(1 if _field_or(fields, f'SRC{i}_NEG', 0) else x[1] for i, x in enumerate(srcs, 1))
    return _Inst(name, _reg(_field(fields, 'DST'), bool(_field_or(fields, 'DST_HALF', 0))), tuple(x[0] for x in srcs),
      bool(_field(fields, 'SY')), _field_or(fields, 'NOP', 0), _field_or(fields, 'REPEAT', 0),
      tuple(bool(_field_or(fields, f'SRC{i}_R', 0)) for i in range(1, 4)), mods, sat=bool(_field_or(fields, 'SAT', 0)))
  if name in {'shrm', 'shlm', 'shrg', 'shlg', 'andg'}:
    half = bool(_field_or(fields, 'HALF', 0))
    def alt_src(value): return value & 0xfff if value & 0x1000 else _src(value, half)[0]
    srcs = (alt_src(_field(fields, 'SRC1')), _src(_field(fields, 'SRC2'), half)[0], alt_src(_field(fields, 'SRC3')))
    return _Inst(name, _reg(_field(fields, 'DST'), bool(_field_or(fields, 'DST_HALF', 0))), srcs,
      bool(_field(fields, 'SY')), _field_or(fields, 'NOP', 0), _field_or(fields, 'REPEAT', 0),
      tuple(bool(_field_or(fields, f'SRC{i}_R', 0)) for i in range(1, 4)), source_half=half)
  if name not in ('add.f', 'mul.f'):
    raise NotImplementedError(f'unsupported IR3 instruction {name}')
  return _Inst(name, _reg(_field(fields, 'DST')),
                        tuple(_reg(_field(fields, src)) for src in ('SRC1', 'SRC2')),
                        bool(_field(fields, 'SY')), _field_or(fields, 'NOP', 0))

@functools.cache
def _decode(code:bytes, gpu_id:int=630) -> tuple[_Inst, ...]:
  if len(code) % 8:
    raise ValueError('IR3 code size must be a multiple of 8 bytes')
  raw: list[_Inst] = []
  current: list[tuple[str, str | int]] = []
  errors: list[Exception] = []

  @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p)
  def pre(_data, _number, _instruction):
    current.clear()

  @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_char),
                    ctypes.POINTER(mesa.struct_isa_decode_value))
  def field(_data, name, value):
    value = value.contents
    current.append((ctypes.string_at(name).decode(),
                    ctypes.string_at(value.str).decode() if value.str else value.num))

  @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p)
  def post(_data, _number, _instruction):
    bits = ctypes.cast(_instruction, ctypes.POINTER(ctypes.c_uint64)).contents.value
    try: raw.append(_decode_fields([*current, ('RAW_BITS', bits)]))
    except Exception as exc:
      errors.append(ValueError(f'IR3 decode failed at PC {_number} ({bits:#018x}): {exc}; fields={current.copy()}'))

  with tempfile.TemporaryFile('w+') as tf:
    fp = libc.fdopen(os.dup(tf.fileno()), b'w')
    try:
      opts = mesa.struct_isa_decode_options(gpu_id, True, 0, False, field_cb=field,
                                            pre_instr_cb=pre, post_instr_cb=post)
      out = ctypes.cast(fp, ctypes.POINTER(mesa.struct__IO_FILE))
      mesa.ir3_isa_disasm(code, len(code), out, opts)
      libc.fflush(fp)
    finally: libc.fclose(fp)
  if errors: raise errors[0]
  if len(raw) != len(code) // 8: raise ValueError('invalid IR3 instruction encoding')
  return tuple(raw)

_ALU_OPS = frozenset({'add.f', 'mul.f', 'min.f', 'max.f', 'cmps.f', 'cmpv.f', 'sign.f', 'absneg.f', 'floor.f', 'ceil.f',
  'rndne.f', 'rndaz.f', 'trunc.f', 'add.u', 'add.s', 'sub.u', 'sub.s', 'cmps.u', 'cmps.s', 'cmpv.u', 'cmpv.s', 'min.u', 'min.s',
  'max.u', 'max.s', 'absneg.s', 'and.b', 'or.b', 'xor.b', 'not.b', 'mul.u24', 'mul.s24', 'mull.u', 'bfrev.b', 'clz.s', 'clz.b',
  'cbits.b', 'shl.b', 'shr.b', 'ashr.b', 'getbit.b', 'rcp', 'rsq', 'log2', 'exp2', 'sin', 'cos', 'sqrt', 'hrsq', 'hlog2',
  'hexp2'})
_FLOAT_OPS = frozenset({'rcp', 'rsq', 'log2', 'exp2', 'sin', 'cos', 'sqrt', 'hrsq', 'hlog2', 'hexp2'}) | \
  frozenset(name for name in _ALU_OPS if name.endswith('.f'))

def _source_offset(src, offset):
  if isinstance(src, int): return src
  if src[0] in {'rel', 'relr', 'relhr'}: return (src[0], src[1] + offset, src[2])
  return _reg_offset(src, offset)

def _source_values(regs, src, lanes, half=False):
  if not isinstance(src, tuple) or src[0] != 'rel': return _values(regs, src, lanes)
  addresses = _values(regs, ('a', 61, 0), lanes)
  kind = ('hc' if src[2] else 'hr') if half else ('c' if src[2] else 'r')
  values = []
  for lane, address in enumerate(addresses):
    component = address + src[1]
    row = regs.get((kind, component // 4, component % 4))
    values.append(row[lane] if row is not None else 0)
  return values

def _write_destination(regs, dst, values, mask):
  if dst[0] not in {'relr', 'relhr'}: return _write(regs, dst, values, mask)
  addresses = _values(regs, ('a', 61, 0), len(values))
  kind = 'hr' if dst[0] == 'relhr' else 'r'
  for lane, (address, value, active) in enumerate(zip(addresses, values, mask, strict=True)):
    if not active: continue
    component = address + dst[1]
    target = (kind, component // 4, component % 4)
    previous = regs.get(target)
    if previous is None: regs[target] = previous = [0] * len(values)
    previous[lane] = value

def _check_access(check_range, address, size, name, pc):
  if check_range is None: raise RuntimeError(f'IR3 {name} requires a mapped-memory validator at PC {pc}')
  check_range(address, size)

def _private_lanes(private:bytearray|list[bytearray]|None, lanes:int, name:str) -> list[bytearray]:
  if isinstance(private, bytearray):
    if lanes == 1: return [private]
    raise RuntimeError(f'IR3 {name} requires one private backing per lane, got one backing for {lanes} lanes')
  if not isinstance(private, list) or len(private) != lanes or any(not isinstance(memory, bytearray) for memory in private):
    raise RuntimeError(f'IR3 {name} requires {lanes} bytearray private backings')
  return private

def _validate_targets(targets, mask, unit, check_range, name, pc):
  active = [target for target, active in zip(targets, mask, strict=True) if active]
  if active:
    low, high = min(active), max(active)
    try:
      check_range(low, high - low + unit)
      return
    except Exception: pass
  for lane, target in enumerate(targets):
    if mask[lane]: _check_access(check_range, target, unit, name, pc)

def _read_targets(targets, mask, itemsize):
  char = {1: 'B', 2: 'H', 4: 'I'}[itemsize]
  out: list[int] = []
  lane = 0
  while lane < len(targets):
    if not mask[lane]:
      out.append(0)
      lane += 1
      continue
    run = lane + 1
    while run < len(targets) and mask[run] and targets[run] == targets[run - 1] + itemsize: run += 1
    if run > lane + 1: out.extend(struct.unpack(f'<{run - lane}{char}', ctypes.string_at(targets[lane], (run - lane) * itemsize)))
    else: out.append(int.from_bytes(ctypes.string_at(targets[lane], itemsize), 'little'))
    lane = run
  return out

def _write_targets(targets, mask, values, itemsize):
  unit_mask = (1 << (itemsize * 8)) - 1
  lane = 0
  while lane < len(targets):
    if not mask[lane]:
      lane += 1
      continue
    run = lane + 1
    while run < len(targets) and mask[run] and targets[run] == targets[run - 1] + itemsize: run += 1
    if run > lane + 1 and itemsize == 4:
      ctypes.memmove(targets[lane], b''.join((values[index] & 0xffffffff).to_bytes(4, 'little') for index in range(lane, run)), (run - lane) * 4)
    else:
      for index in range(lane, run):
        ctypes.memmove(targets[index], (values[index] & unit_mask).to_bytes(itemsize, 'little'), itemsize)
    lane = run

def _exec(code:bytes, regs:dict[tuple[str, int, int], list[int]], gpu_id:int=630, check_range=None, start_pc=0,
                shared:bytearray|list[bytearray]|None=None, private:bytearray|list[bytearray]|None=None,
                stop_at_barrier=False, textures=(), ibos=(), resume_state:dict[str, Any]|None=None):
  program, pc = _decode(code, gpu_id), start_pc
  step_limit = max(100000, len(program) * 65536)
  lanes = _lane_count(regs)
  branch_frames: list[dict[str, Any]]
  if resume_state:
    steps = resume_state['steps']
    predication = resume_state['predication']
    exec_mask = resume_state['exec_mask']
    branch_frames = resume_state['branch_frames']
  else:
    steps, predication = 0, None
    exec_mask = [True] * lanes
    branch_frames = []
  while pc < len(program):
    if branch_frames:
      frame = branch_frames[-1]
      if frame['reconv'] is not None and pc == frame['reconv'][0]:
        exec_mask = frame['reconv'][1]
        branch_frames.pop()
      elif frame['alternate'] is not None and pc == frame['alternate'][0]:
        exec_mask = [a or b for a, b in zip(exec_mask, frame['alternate'][1], strict=True)]
        branch_frames.pop()
    inst_pc, inst = pc, program[pc]
    pc += 1
    steps += 1
    if steps > step_limit: raise RuntimeError(f'IR3 execution did not terminate at PC {inst_pc}')
    if inst.name == 'nop': continue
    if inst.name == 'end': break
    if inst.name == 'bar':
      if stop_at_barrier:
        if resume_state is not None:
          resume_state.clear()
          resume_state.update(steps=steps, predication=predication, exec_mask=exec_mask, branch_frames=branch_frames)
        return pc
      continue
    if inst.name == 'fence': continue
    if inst.name == 'jump':
      target = inst_pc + inst.branch_offset
      if branch_frames and branch_frames[-1]['alternate'] is not None:
        frame = branch_frames[-1]
        alternate_pc, alternate_mask = frame['alternate']
        frame['alternate'] = None
        frame['reconv'] = (target, [a or b for a, b in zip(exec_mask, alternate_mask, strict=True)])
        pc, exec_mask = alternate_pc, alternate_mask
      else: pc = target
      continue
    if inst.name == 'br':
      cond = [not bool(x) if inst.invert else bool(x) for x in _values(regs, inst.srcs[0], lanes)]
      taken = [active and value for active, value in zip(exec_mask, cond, strict=True)]
      fallthrough = [active and not value for active, value in zip(exec_mask, cond, strict=True)]
      if any(taken) and any(fallthrough):
        branch_frames.append({'alternate':(inst_pc + inst.branch_offset, taken), 'reconv':None})
        exec_mask = fallthrough
      elif any(taken): pc = inst_pc + inst.branch_offset
      continue
    if inst.name in {'bany', 'ball'}:
      cond = [not bool(x) if inst.invert else bool(x) for x in _values(regs, inst.srcs[0], lanes)]
      active_cond = [value for value, active in zip(cond, exec_mask, strict=True) if active]
      if (any(active_cond) if inst.name == 'bany' else all(active_cond)): pc = inst_pc + inst.branch_offset
      continue
    if inst.name in {'brao', 'braa'}:
      predicates = [[not bool(x) if invert else bool(x) for x in _values(regs, src, lanes)]
                    for src, invert in zip(inst.srcs, inst.inverts, strict=True)]
      lane_cond = [a or b if inst.name == 'brao' else a and b for a, b in zip(*predicates, strict=True)]
      taken = [active and value for active, value in zip(exec_mask, lane_cond, strict=True)]
      fallthrough = [active and not value for active, value in zip(exec_mask, lane_cond, strict=True)]
      if any(taken) and any(fallthrough):
        branch_frames.append({'alternate':(inst_pc + inst.branch_offset, taken), 'reconv':None})
        exec_mask = fallthrough
      elif any(taken): pc = inst_pc + inst.branch_offset
      continue
    if inst.name in {'predt', 'predf'}:
      predicate = _values(regs, ('p', 62, 0), lanes)
      predication = [bool(value) == (inst.name == 'predt') for value in predicate]
      continue
    if inst.name == 'prede':
      predication = None
      continue
    write_mask = exec_mask if predication is None else [active and pred for active, pred in zip(exec_mask, predication, strict=True)]
    if inst.name in {'ashr.b', 'shl.b'}:
      repeated = inst.repeat_srcs + (False,) * (2 - len(inst.repeat_srcs))
      runner = _alu_runner(inst.name, 0, inst.src_mods + (0,) * (2 - len(inst.src_mods)), inst.source_half, inst.dst[0].startswith('h'))
      for component in range(inst.repeat + 1):
        srcs = tuple(_source_offset(src, component) if repeat else src
                     for src, repeat in zip(inst.srcs, repeated, strict=True))
        values = [_values(regs, src, lanes) for src in srcs]
        _write(regs, _reg_offset(inst.dst, component), [runner(tuple(x)) for x in zip(*values, strict=True)], write_mask)
      continue
    if inst.name == 'shrg':
      repeated = inst.repeat_srcs + (False,) * (3 - len(inst.repeat_srcs))
      source_mask = 0xffff if inst.source_half else 0xffffffff
      dest_mask = 0xffff if inst.dst[0].startswith('h') else 0xffffffff
      for component in range(inst.repeat + 1):
        srcs = tuple(_source_offset(src, component) if repeat else src
                     for src, repeat in zip(inst.srcs, repeated, strict=True))
        shifts, values, others = (_source_values(regs, src, lanes, inst.source_half) for src in srcs)
        result = [(((value & source_mask) >> (shift & 31)) | other) & dest_mask
                  for shift, value, other in zip(shifts, values, others, strict=True)]
        _write(regs, _reg_offset(inst.dst, component), result, write_mask)
      continue
    if inst.name == 'add.u':
      mask = 0xffff if inst.dst[0].startswith('h') else 0xffffffff
      for component in range(inst.repeat + 1):
        srcs = tuple(_source_offset(src, component) if repeated else
                     src + component if repeated else src
                     for src, repeated in zip(inst.srcs, inst.repeat_srcs, strict=True))
        lhs = _values(regs, srcs[0], lanes)
        rhs = _values(regs, srcs[1], lanes)
        _write(regs, _reg_offset(inst.dst, component), [(x + y) & mask for x, y in zip(lhs, rhs, strict=True)], write_mask)
      continue
    if inst.name in ('cmps.u', 'cmps.s'):
      unsigned_mask = 0xffff if inst.source_half else 0xffffffff
      for component in range(inst.repeat + 1):
        srcs = tuple(_source_offset(src, component) if repeated else src
                     for src, repeated in zip(inst.srcs, inst.repeat_srcs, strict=True))
        lhs, rhs = (_values(regs, src, lanes) for src in srcs)
        signed = inst.name == 'cmps.s'
        _write(regs, _reg_offset(inst.dst, component), [int(_compare(_signed(x, inst.source_half) if signed else x & unsigned_mask,
          _signed(y, inst.source_half) if signed else y & unsigned_mask, inst.condition)) for x, y in zip(lhs, rhs, strict=True)], write_mask)
      continue
    if inst.name == 'absneg.s':
      mask = 0xffff if inst.dst[0].startswith('h') else 0xffffffff
      for component in range(inst.repeat + 1):
        src = _source_offset(inst.srcs[0], component) if inst.repeat_srcs[0] else inst.srcs[0]
        mod = inst.src_mods[0]
        vals = _values(regs, src, lanes)
        result = [(-abs(_signed(x, inst.source_half)) if mod == 3 else abs(_signed(x, inst.source_half)) if mod == 2
                   else -_signed(x, inst.source_half) if mod == 1 else _signed(x, inst.source_half)) & mask
                  for x in vals]
        _write(regs, _reg_offset(inst.dst, component), result, write_mask)
      continue
    if inst.name == 'sel.b32':
      for component in range(inst.repeat + 1):
        srcs = tuple(_source_offset(src, component) if repeated else src
                     for src, repeated in zip(inst.srcs, inst.repeat_srcs, strict=True))
        yes, cond, no = (_values(regs, src, lanes) for src in srcs)
        _write(regs, _reg_offset(inst.dst, component), [x if c else y for x, c, y in zip(yes, cond, no, strict=True)], write_mask)
      continue
    if inst.name in {'shrm', 'shlm', 'shlg', 'andg'}:
      repeated = inst.repeat_srcs + (False,) * (3 - len(inst.repeat_srcs))
      mask = 0xffff if inst.dst[0].startswith('h') else 0xffffffff
      for component in range(inst.repeat + 1):
        srcs = tuple(_source_offset(src, component) if repeat else src
                     for src, repeat in zip(inst.srcs, repeated, strict=True))
        first, second, third = (_values(regs, src, lanes) for src in srcs)
        result = []
        for a, b, c in zip(first, second, third, strict=True):
          if inst.name == 'shrm': value = ((b & 0xffffffff) >> (a & 31)) & c
          elif inst.name == 'shlm': value = (b << (a & 31)) & c
          elif inst.name == 'shlg': value = (b << (a & 31)) | c
          else: value = (b & a) | c
          result.append(value & mask)
        _write(regs, _reg_offset(inst.dst, component), result, write_mask)
      continue
    if inst.name.startswith(('mad.', 'madsh.', 'sel.', 'sad.')):
      repeated = inst.repeat_srcs + (False,) * (3 - len(inst.repeat_srcs))
      half = inst.name.endswith('16')
      for component in range(inst.repeat + 1):
        srcs = tuple(_source_offset(src, component) if repeat else src
                     for src, repeat in zip(inst.srcs, repeated, strict=True))
        values = [_float_values(regs, src, lanes, half) if inst.name.startswith('mad.f') else _values(regs, src, lanes) for src in srcs]
        result = []
        for raw in zip(*values, strict=True):
          if inst.name.startswith('mad.f'):
            val = _fma(_mod_float(raw[0], inst.src_mods[0], half), _mod_float(raw[1], inst.src_mods[1], half),
                       _mod_float(raw[2], inst.src_mods[2], half))
            result.append(_float_bits(min(1.0, max(0.0, val)) if inst.sat else val, half))
          elif inst.name.startswith('sel.'):
            result.append(raw[0] if raw[1] else raw[2])
          elif inst.name.startswith('sad.'):
            result.append((abs(_s32(raw[0]) - _s32(raw[1])) + raw[2]) & 0xffffffff)
          elif inst.name == 'madsh.m16':
            result.append(((((raw[0] & 0xffff) * ((raw[1] >> 16) & 0xffff)) << 16) + raw[2]) & 0xffffffff)
          else:
            signed = '.s' in inst.name
            bits = 16 if '16' in inst.name else 24
            mask, sign = (1 << bits) - 1, 1 << (bits - 1)
            a, b = raw[0] & mask, raw[1] & mask
            if signed: a, b = a - (1 << bits) if a & sign else a, b - (1 << bits) if b & sign else b
            result.append((a * b + raw[2]) & 0xffffffff)
        _write(regs, _reg_offset(inst.dst, component), result, write_mask)
      continue
    if inst.name in {'swz', 'gat', 'sct'}:
      source_values = [_values(regs, src, lanes).copy() for src in inst.srcs]
      for dst, values in zip(inst.dst, source_values, strict=True):
        converted = [_convert(value, inst.types[0], inst.types[1], inst.rounding) for value in values]
        _write(regs, dst, converted, write_mask)
      continue
    if inst.name == 'mov':
      for component in range(inst.repeat + 1):
        src = _source_offset(inst.srcs[0], component) if inst.repeat_srcs[0] else inst.srcs[0]
        converted = [_convert(x, inst.types[0], inst.types[1], inst.rounding)
                     for x in _source_values(regs, src, lanes, inst.source_half)]
        _write_destination(regs, _source_offset(inst.dst, component), converted, write_mask)
      continue
    if inst.name in {'ldl', 'ldp', 'stl', 'stp'}:
      itemsize = _itemsize(inst.types[0])
      if inst.name.endswith('l'):
        if shared is None: raise RuntimeError(f'IR3 {inst.name} has no backing memory')
        if isinstance(shared, list):
          if len(shared) != lanes or any(not isinstance(memory, bytearray) for memory in shared):
            raise RuntimeError(f'IR3 {inst.name} requires {lanes} shared-memory lane mappings')
          memories = shared
        else: memories = [shared] * lanes
      else: memories = _private_lanes(private, lanes, inst.name)
      if inst.name.startswith('ld'):
        address_reg, offset, size = inst.srcs
        addresses = _values(regs, address_reg, lanes)
        for component in range(size):
          out = []
          for lane, address in enumerate(addresses):
            pos = address + offset + component * itemsize
            if not write_mask[lane]:
              out.append(0)
              continue
            lane_memory = memories[lane]
            if pos < 0 or pos + itemsize > len(lane_memory): raise RuntimeError(f'IR3 {inst.name} out of bounds at {pos:#x}')
            out.append(int.from_bytes(lane_memory[pos:pos + itemsize], 'little'))
          _write(regs, _reg_offset(inst.dst, component), out, write_mask)
      else:
        address_reg, value_reg, offset, size = inst.srcs
        addresses = _values(regs, address_reg, lanes)
        for component in range(size):
          values = _values(regs, _reg_offset(value_reg, component), lanes)
          for lane, (address, value) in enumerate(zip(addresses, values, strict=True)):
            if not write_mask[lane]: continue
            pos = address + offset + component * itemsize
            lane_memory = memories[lane]
            if pos < 0 or pos + itemsize > len(lane_memory): raise RuntimeError(f'IR3 {inst.name} out of bounds at {pos:#x}')
            lane_memory[pos:pos + itemsize] = (value & ((1 << (itemsize * 8)) - 1)).to_bytes(itemsize, 'little')
      continue
    if inst.name == 'stib':
      coord_reg, value_reg, offset, resource_index, dimensions, components = inst.srcs
      if dimensions != 2 or not 1 <= components <= 4: raise NotImplementedError('unsupported IR3 image store shape')
      if resource_index >= len(ibos): raise RuntimeError(f'IR3 image store references missing IBO {resource_index}')
      image = ibos[resource_index]
      xs, ys = _values(regs, coord_reg, lanes), _values(regs, _next_reg(coord_reg), lanes)
      for lane, (x, y) in enumerate(zip(xs, ys, strict=True)):
        x, y = _s32(x), _s32(y)
        if not write_mask[lane] or not (0 <= x < image['width'] and 0 <= y < image['height']): continue
        address = image['address'] + y * image['pitch'] + x * components * image['itemsize'] + offset
        _check_access(check_range, address, components * image['itemsize'], inst.name, inst_pc)
        for component in range(components):
          value = _values(regs, _reg_offset(value_reg, component), lanes)[lane]
          if image['encoded_itemsize'] == 2:
            numeric = _float(value, inst.types[0] == mesa.TYPE_F16)
            value = _float_bits(_float(_float_bits(numeric, True), True))
          elif inst.types[0] == mesa.TYPE_F16: value = _float_bits(_float(value, True))
          ctypes.memmove(address + component * image['itemsize'], value.to_bytes(image['itemsize'], 'little'), image['itemsize'])
      continue
    if inst.name == 'isam':
      coord_reg, _sampler_index, texture_index, dimensions, write_components = inst.srcs
      if texture_index >= len(textures): raise RuntimeError(f'IR3 image sample references missing texture {texture_index}')
      image = textures[texture_index]
      coordinates = [_values(regs, _reg_offset(coord_reg, component), lanes) for component in range(dimensions)]
      selected = [component for component in range(4) if write_components & (1 << component)]
      outputs: list[list[int]] = [[] for _ in selected]
      for lane, raw_coords in enumerate(zip(*coordinates, strict=True)):
        coords = tuple(_s32(value) for value in raw_coords)
        if dimensions == 1:
          pixel = coords[0]
          x, y = pixel % image['width'], pixel // image['width']
        else: x, y = coords
        valid = write_mask[lane] and 0 <= x < image['width'] and 0 <= y < image['height']
        address = image['address'] + y * image['pitch'] + x * 4 * image['itemsize'] if valid else 0
        if valid: _check_access(check_range, address, 4 * image['itemsize'], inst.name, inst_pc)
        for output, component in zip(outputs, selected, strict=True):
          if not valid: output.append(0)
          else:
            value = int.from_bytes(ctypes.string_at(address + component * image['itemsize'], image['itemsize']), 'little')
            if inst.types[1] == mesa.TYPE_F16: value = _float_bits(_float(value), True)
            output.append(value)
      for offset, output in enumerate(outputs): _write(regs, _reg_offset(inst.dst, offset), output, write_mask)
      continue
    if inst.name in _ALU_OPS:
      repeated = inst.repeat_srcs + (False,) * (len(inst.srcs) - len(inst.repeat_srcs))
      modifiers = inst.src_mods + (0,) * (len(inst.srcs) - len(inst.src_mods))
      half = inst.dst[0].startswith('h')
      runner = _alu_runner(inst.name, inst.condition, modifiers, inst.source_half, half)
      for component in range(inst.repeat + 1):
        srcs = tuple(_source_offset(src, component) if repeat else src
                     for src, repeat in zip(inst.srcs, repeated, strict=True))
        float_op = inst.name in _FLOAT_OPS
        values = [_float_values(regs, src, lanes, inst.source_half) if float_op else _values(regs, src, lanes) for src in srcs]
        result = [runner(tuple(x)) for x in zip(*values, strict=True)]
        if inst.sat: result = [_float_bits(min(1.0, max(0.0, _float(x, half))), half) for x in result]
        _write(regs, _reg_offset(inst.dst, component), result, write_mask)
      continue
    if inst.name == 'cov.u16s32':
      _write(regs, inst.dst, [value & 0xffff for value in _values(regs, inst.srcs[0], lanes)], write_mask)
      continue
    if inst.name == 'mov.u32u32':
      src = inst.srcs[0]
      _write(regs, inst.dst, _values(regs, src, lanes).copy(), write_mask)
      continue
    if inst.name.startswith('atomic.g.'):
      if len(inst.srcs) not in (2, 3): raise NotImplementedError(f'unsupported IR3 atomic operands for {inst.name}')
      itemsize = _itemsize(inst.types[0])
      if itemsize != 4: raise NotImplementedError(f'unsupported {itemsize * 8}-bit IR3 atomic {inst.name}')
      address_reg, value_reg, *compare_reg = inst.srcs
      addresses = [low | high << 32 for low, high in
                   zip(_values(regs, address_reg, lanes), _values(regs, _next_reg(address_reg), lanes), strict=True)]
      values = _values(regs, value_reg, lanes)
      compares = _values(regs, compare_reg[0], lanes) if compare_reg else [0] * lanes
      old_values = [0] * lanes
      operation = inst.name.removeprefix('atomic.g.')
      for lane, (address, value, compare) in enumerate(zip(addresses, values, compares, strict=True)):
        if not write_mask[lane]: continue
        _check_access(check_range, address, itemsize, inst.name, inst_pc)
        old = ctypes.c_uint32.from_address(address).value
        if operation == 'add': new = old + value
        elif operation == 'sub': new = old - value
        elif operation in {'xchg', 'exchange'}: new = value
        elif operation in {'cmpxchg', 'cas'}: new = value if old == compare else old
        elif operation == 'and': new = old & value
        elif operation == 'or': new = old | value
        elif operation == 'xor': new = old ^ value
        elif operation == 'min': new = min(_s32(old), _s32(value))
        elif operation == 'max': new = max(_s32(old), _s32(value))
        elif operation == 'umin': new = min(old, value)
        elif operation == 'umax': new = max(old, value)
        else: raise NotImplementedError(f'unsupported IR3 global atomic {operation}')
        ctypes.c_uint32.from_address(address).value = new & 0xffffffff
        old_values[lane] = old
      if inst.dst is not None: _write(regs, inst.dst, old_values, write_mask)
      continue
    if inst.name == 'ldg':
      address_reg, offset, size = inst.srcs
      if not 1 <= size <= 4: raise NotImplementedError(f'unsupported ldg size {size}')
      if check_range is None and any(write_mask): _check_access(check_range, 0, 0, inst.name, inst_pc)
      itemsize = _itemsize(inst.types[0])
      addresses = [low | high << 32 for low, high in
                   zip(_values(regs, address_reg, lanes), _values(regs, _next_reg(address_reg), lanes), strict=True)]
      span = size * itemsize
      valid_lanes = write_mask.copy()
      if predication is None and any(write_mask):
        active = [addresses[lane] + offset for lane in range(lanes) if write_mask[lane]]
        try:
          check_range(min(active), max(active) - min(active) + span)
          checked = True
        except Exception: checked = False
        if not checked:
          for lane, address in enumerate(addresses):
            if not write_mask[lane]: continue
            try: _check_access(check_range, address + offset, span, inst.name, inst_pc)
            except Exception as exc:
              raise RuntimeError(f'IR3 {inst.name} memory fault at PC {inst_pc}, lane {lane}, address={address:#x}') from exc
      else:
        for lane, address in enumerate(addresses):
          if not write_mask[lane]: continue
          try: _check_access(check_range, address + offset, span, inst.name, inst_pc)
          except Exception as exc:
            if predication is not None:
              valid_lanes[lane] = False
              continue
            raise RuntimeError(f'IR3 {inst.name} memory fault at PC {inst_pc}, lane {lane}, address={address:#x}') from exc
      fmt = '<%d%s' % (size, {1: 'B', 2: 'H', 4: 'I'}[itemsize])
      loaded_components: list[list[int]] = [[] for _ in range(size)]
      lane = 0
      while lane < lanes:
        address = addresses[lane] + offset
        if not valid_lanes[lane]:
          for component in range(size): loaded_components[component].append(0)
          lane += 1
          continue
        run = lane + 1
        if span >= 4:
          while run < lanes and valid_lanes[run] and addresses[run] + offset == addresses[run - 1] + offset + span: run += 1
        if run > lane + 1:
          data = struct.unpack(f'<{(run - lane) * size}{fmt[2]}', ctypes.string_at(address, (run - lane) * span))
          for index in range(run - lane):
            for component in range(size): loaded_components[component].append(data[index * size + component])
        else:
          data = struct.unpack(fmt, ctypes.string_at(address, span))
          for component in range(size): loaded_components[component].append(data[component])
        lane = run
      for component in range(size): _write(regs, _reg_offset(inst.dst, component), loaded_components[component], write_mask)
      continue
    if inst.name == 'ldg.a':
      address_reg, index_reg, shift, size = inst.srcs
      if not 1 <= size <= 4: raise NotImplementedError(f'unsupported ldg.a size {size}')
      itemsize = _itemsize(inst.types[0])
      addresses = [low | high << 32 for low, high in
                   zip(_values(regs, address_reg, lanes), _values(regs, _next_reg(address_reg), lanes), strict=True)]
      indices = _values(regs, index_reg, lanes)
      targets = [address + (_s32(index) << shift) for address, index in zip(addresses, indices, strict=True)]
      _validate_targets(targets, write_mask, itemsize, check_range, inst.name, inst_pc)
      for component in range(size):
        output = _read_targets([target + component * itemsize for target in targets], write_mask, itemsize)
        _write(regs, _reg_offset(inst.dst, component), output, write_mask)
      continue
    if inst.name == 'stg':
      address_reg, value_reg, offset, size = inst.srcs
      if not 1 <= size <= 4: raise NotImplementedError(f'unsupported stg size {size}')
      if check_range is None and any(write_mask): _check_access(check_range, 0, 0, inst.name, inst_pc)
      itemsize = _itemsize(inst.types[0])
      addresses = [low | high << 32 for low, high in
                   zip(_values(regs, address_reg, lanes), _values(regs, _next_reg(address_reg), lanes), strict=True)]
      span = size * itemsize
      if any(write_mask):
        active = [addresses[lane] + offset for lane in range(lanes) if write_mask[lane]]
        try:
          check_range(min(active), max(active) - min(active) + span)
          checked = True
        except Exception: checked = False
        if not checked:
          for lane, address in enumerate(addresses):
            if not write_mask[lane]: continue
            try: _check_access(check_range, address + offset, span, inst.name, inst_pc)
            except Exception as exc:
              raise RuntimeError(f'IR3 {inst.name} memory fault at PC {inst_pc}, lane {lane}, address={address:#x}, '
                f'c0={[_values(regs, ("c", 0, i), lanes)[lane] for i in range(4)]}, '
                f'r0={[_values(regs, ("r", 0, i), lanes)[lane] for i in range(4)]}') from exc
      columns = [_values(regs, _reg_offset(value_reg, component), lanes) for component in range(size)]
      unit_mask = (1 << (itemsize * 8)) - 1
      lane = 0
      while lane < lanes:
        if not write_mask[lane]:
          lane += 1
          continue
        address = addresses[lane] + offset
        run = lane + 1
        while run < lanes and write_mask[run] and addresses[run] + offset == addresses[run - 1] + offset + span: run += 1
        if run > lane + 1 and itemsize == 4:
          payload = b''.join((column[index] & unit_mask).to_bytes(4, 'little') for index in range(lane, run) for column in columns)
          ctypes.memmove(address, payload, (run - lane) * span)
        else:
          for index in range(lane, run):
            ctypes.memmove(addresses[index] + offset,
              b''.join((column[index] & unit_mask).to_bytes(itemsize, 'little') for column in columns), span)
        lane = run
      continue
    if inst.name == 'stg.a':
      address_reg, index_reg, shift, value_reg, size = inst.srcs
      if not 1 <= size <= 4: raise NotImplementedError(f'unsupported stg.a size {size}')
      itemsize = _itemsize(inst.types[0])
      addresses = [low | high << 32 for low, high in
                   zip(_values(regs, address_reg, lanes), _values(regs, _next_reg(address_reg), lanes), strict=True)]
      indices = _values(regs, index_reg, lanes)
      targets = [address + (_s32(index) << shift) for address, index in zip(addresses, indices, strict=True)]
      _validate_targets(targets, write_mask, itemsize, check_range, inst.name, inst_pc)
      for component in range(size):
        _write_targets([target + component * itemsize for target in targets], write_mask,
                       _values(regs, _reg_offset(value_reg, component), lanes), itemsize)
      continue
    raise NotImplementedError(f'unsupported IR3 execution {inst.name}')
  if resume_state is not None: resume_state.clear()
  return None


def _batch_size(program, lane_count):
  if not 1 <= lane_count <= 64: return 1
  if any(inst.name in {'bany', 'ball', 'brao', 'braa'} or inst.name.startswith('atomic.') for inst in program): return 1
  return 256 // lane_count


def _dispatch(code, grid_size, local_size, local_id_register, check_range=None, workgroup_id_register=0xfc,
              textures=(), ibos=(), constant_words=()):
  lane_count = local_size[0] * local_size[1] * local_size[2]
  program = _decode(code)
  uses_private = any(inst.name in {'ldp', 'stp'} for inst in program)
  uses_shared = any(inst.name in {'ldl', 'stl'} for inst in program)
  all_local_ids = _local_ids(local_size, local_id_register)
  constants = _Const(constant_words)

  def make_regs(coord, wave_start, wave_lanes):
    regs = _Regs(wave_lanes, constants=constants)
    regs.update({key: values[wave_start:wave_start + wave_lanes] for key, values in all_local_ids.items()})
    if workgroup_id_register != 0xfc: regs.update(_group_ids(coord, wave_lanes, workgroup_id_register))
    return regs

  batch_groups = _batch_size(program, lane_count)
  if batch_groups > 1:
    coords = [(x, y, z) for z in range(grid_size[2]) for y in range(grid_size[1]) for x in range(grid_size[0])]
    batch_last_regs = _Regs(lane_count, constants=constants)
    for batch_start in range(0, len(coords), batch_groups):
      batch = coords[batch_start:batch_start + batch_groups]
      regs = _Regs(len(batch) * lane_count, constants=constants)
      for key, values in all_local_ids.items(): regs[key] = values[:lane_count] * len(batch)
      if workgroup_id_register != 0xfc:
        for component in range(3):
          regs[('r', workgroup_id_register, component)] = [coord[component] for coord in batch for _ in range(lane_count)]
      shared_lanes: list[bytearray] | None = [] if uses_shared else None
      private_lanes: list[bytearray] | None = [] if uses_private else None
      if shared_lanes is not None or private_lanes is not None:
        for _ in batch:
          if shared_lanes is not None:
            group_shared = bytearray(0x10000)
            shared_lanes.extend([group_shared] * lane_count)
          if private_lanes is not None: private_lanes.extend(bytearray(0x10000) for _ in range(lane_count))
      pc = 0
      resume_state: dict[str, Any] = {}
      while (next_pc := _exec(code, regs, check_range=check_range, start_pc=pc, shared=shared_lanes, private=private_lanes,
                              stop_at_barrier=True, textures=textures, ibos=ibos, resume_state=resume_state)) is not None: pc = next_pc
      offset = (len(batch) - 1) * lane_count
      batch_last_regs = _Regs(lane_count,
        {key: values[offset:offset + lane_count] for key, values in regs.items()}, constants)
    return batch_last_regs

  last_regs = {}
  for z in range(grid_size[2]):
    for y in range(grid_size[1]):
      for x in range(grid_size[0]):
        waves, privates = [], []
        resume_states: list[dict[str, Any]] = []
        for wave_start in range(0, lane_count, 64):
          wave_lanes = min(64, lane_count - wave_start)
          waves.append(make_regs((x, y, z), wave_start, wave_lanes))
          privates.append([bytearray(0x10000) for _ in range(wave_lanes)] if uses_private else None)
          resume_states.append({})
        shared, pcs, done = (bytearray(0x10000) if uses_shared else None), [0] * len(waves), [False] * len(waves)
        while not all(done):
          reached_barrier: list[int] = []
          for index, regs in enumerate(waves):
            if done[index]: continue
            next_pc = _exec(code, regs, check_range=check_range, start_pc=pcs[index], shared=shared, private=privates[index],
                            stop_at_barrier=True, textures=textures, ibos=ibos, resume_state=resume_states[index])
            if next_pc is None: done[index] = True
            else: pcs[index], reached_barrier = next_pc, [*reached_barrier, index]
          if reached_barrier and any(done): raise RuntimeError('IR3 barrier reached by only part of a workgroup')
        last_regs = waves[-1]
  return last_regs

class A630Emu:
  def __init__(self, check_range=None): self.check_range = check_range

  def execute_cs(self, shader:bytes, const:bytes, grid_size:tuple[int,int,int], local_size:tuple[int,int,int],
                 shader_iova:int=0, local_id_register:int=0, workgroup_id_register:int=0xfc) -> None:
    del shader_iova
    if not shader: return
    _dispatch(shader, grid_size, local_size, local_id_register, check_range=self.check_range, workgroup_id_register=workgroup_id_register,
              constant_words=tuple(int(x) for x in memoryview(const).cast('I')))
