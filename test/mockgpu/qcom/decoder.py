import ctypes, functools, os, tempfile
from dataclasses import dataclass
from typing import Any
from tinygrad.runtime.autogen import libc, mesa
from test.mockgpu.qcom.registers import _cat1_src, _field, _field_or, _reg, _reg_offset, _reject_unsupported_modifiers, _src

def _memory_offset(fields):
  offset = _field(fields, 'OFF')
  return offset - 0x2000 if offset & 0x1000 else offset

@dataclass(frozen=True)
class IR3Instruction:
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
    return IR3Instruction(name, dsts, srcs, bool(_field(fields, 'SY')), 0, types=(src_type, dst_type), rounding=_field_or(fields, 'ROUND', 0))
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
    return IR3Instruction('mov', dst, (src,), bool(_field(fields, 'SY')), 0, _field_or(fields, 'REPEAT', 0),
      (bool(_field_or(fields, 'SRC_R', 0)),), types=types, rounding=_field_or(fields, 'ROUND', 0),
      source_half=src_half)
  if name == 'nop':
    return IR3Instruction(name, None, (), bool(_field(fields, 'SY')), 0)
  if name == 'end':
    return IR3Instruction(name, None, (), bool(_field(fields, 'SY')), 0)
  if name in {'jump', 'br', 'bany', 'ball'}:
    srcs = () if name == 'jump' else ((_reg(248 + _field(fields, 'COMP1')),))
    return IR3Instruction(name, None, srcs, bool(_field(fields, 'SY')), 0, branch_offset=ctypes.c_int32(_field(fields, 'IMMED')).value,
                          invert=bool(_field_or(fields, 'INV1', 0)))
  if name in {'brao', 'braa'}:
    srcs = tuple(_reg(248 + _field(fields, f'COMP{i}')) for i in (1, 2))
    return IR3Instruction(name, None, srcs, bool(_field(fields, 'SY')), 0,
      branch_offset=ctypes.c_int32(_field(fields, 'IMMED')).value,
      inverts=tuple(bool(_field_or(fields, f'INV{i}', 0)) for i in (1, 2)))
  if name in {'predt', 'predf', 'prede'}:
    return IR3Instruction(name, None, (), bool(_field(fields, 'SY')), 0)
  if name in {'bar', 'fence'}:
    return IR3Instruction(name, None, (), bool(_field(fields, 'SY')), 0)
  if name in ('ashr.b', 'shl.b'):
    srcs = tuple(_src(_field(fields, src), bool(_field_or(fields, 'HALF', 0))) for src in ('SRC1', 'SRC2'))
    return IR3Instruction(name, _reg(_field(fields, 'DST'), bool(_field_or(fields, 'DST_HALF', 0))), tuple(x[0] for x in srcs),
      bool(_field(fields, 'SY')), _field_or(fields, 'NOP', 0), _field_or(fields, 'REPEAT', 0),
      tuple(bool(value) for field, value in fields if field == 'SRC_R')[:2], tuple(x[1] for x in srcs),
      source_half=bool(_field_or(fields, 'HALF', 0)))
  if name == 'shrg':
    half = bool(_field_or(fields, 'HALF', 0))
    def alt_src(value): return value & 0xfff if value & 0x1000 else _src(value, half)[0]
    srcs = (alt_src(_field(fields, 'SRC1')), _src(_field(fields, 'SRC2'), half)[0], alt_src(_field(fields, 'SRC3')))
    return IR3Instruction(name, _reg(_field(fields, 'DST'), bool(_field_or(fields, 'DST_HALF', 0))), srcs,
      bool(_field(fields, 'SY')), _field_or(fields, 'NOP', 0), _field_or(fields, 'REPEAT', 0),
      tuple(bool(_field_or(fields, f'SRC{i}_R', 0)) for i in range(1, 4)), source_half=half)
  if name == 'add.u':
    srcs = tuple(_src(_field(fields, src), bool(_field_or(fields, 'HALF', 0))) for src in ('SRC1', 'SRC2'))
    return IR3Instruction(name, _reg(_field(fields, 'DST'), bool(_field_or(fields, 'DST_HALF', 0))), tuple(x[0] for x in srcs),
      bool(_field(fields, 'SY')), _field_or(fields, 'NOP', 0), _field_or(fields, 'REPEAT', 0),
      tuple(bool(value) for field, value in fields if field == 'SRC_R')[:2], tuple(x[1] for x in srcs),
      sat=bool(_field_or(fields, 'SAT', 0)), source_half=bool(_field_or(fields, 'HALF', 0)))
  if name == 'cmps.u':
    srcs_mods = tuple(_src(_field(fields, src), bool(_field_or(fields, 'HALF', 0))) for src in ('SRC1', 'SRC2'))
    return IR3Instruction(name, _reg(_field(fields, 'DST'), bool(_field(fields, 'DST_HALF'))), tuple(x[0] for x in srcs_mods),
      bool(_field(fields, 'SY')), 0, _field_or(fields, 'REPEAT', 0), tuple(bool(value) for field, value in fields if field == 'SRC_R'),
      tuple(x[1] for x in srcs_mods), _field(fields, 'COND'), source_half=bool(_field_or(fields, 'HALF', 0)))
  if name == 'cmps.s':
    srcs_mods = tuple(_src(_field(fields, src), bool(_field_or(fields, 'HALF', 0))) for src in ('SRC1', 'SRC2'))
    return IR3Instruction(name, _reg(_field(fields, 'DST'), bool(_field(fields, 'DST_HALF'))), tuple(x[0] for x in srcs_mods),
      bool(_field(fields, 'SY')), 0, _field_or(fields, 'REPEAT', 0), tuple(bool(value) for field, value in fields if field == 'SRC_R'),
      tuple(x[1] for x in srcs_mods), _field(fields, 'COND'), source_half=bool(_field_or(fields, 'HALF', 0)))
  if name == 'absneg.s':
    src, mod = _src(_field(fields, 'SRC1'), bool(_field_or(fields, 'HALF', 0)))
    return IR3Instruction(name, _reg(_field(fields, 'DST'), bool(_field(fields, 'DST_HALF'))), (src,), bool(_field(fields, 'SY')), 0,
                          _field_or(fields, 'REPEAT', 0), (bool(_field_or(fields, 'SRC_R', 0)),), (mod,),
                          source_half=bool(_field_or(fields, 'HALF', 0)))
  if name == 'sel.b32':
    srcs = tuple(_src(_field(fields, src), bool(_field_or(fields, 'HALF', 0))) for src in ('SRC1', 'SRC2', 'SRC3'))
    return IR3Instruction(name, _reg(_field(fields, 'DST')), tuple(x[0] for x in srcs), bool(_field(fields, 'SY')), 0,
      _field_or(fields, 'REPEAT', 0), tuple(bool(_field_or(fields, f'SRC{i}_R', 0)) for i in range(1, 4)), tuple(x[1] for x in srcs))
  if name == 'ldg':
    typ = _field(fields, 'TYPE')
    return IR3Instruction('ldg', _reg(_field(fields, 'DST'), bool(_field_or(fields, 'TYPE_HALF', 0))),
      (_reg(_field(fields, 'SRC1')), _memory_offset(fields), _field(fields, 'SIZE')),
      bool(_field(fields, 'SY')), 0, types=(typ, typ))
  if name == 'ldg.a':
    typ = _field(fields, 'TYPE')
    return IR3Instruction(name, _reg(_field(fields, 'DST'), bool(_field_or(fields, 'TYPE_HALF', 0))),
      (_reg(_field(fields, 'SRC1')), _reg(_field(fields, 'SRC2')), _field(fields, 'FULL_SHIFT'), _field(fields, 'SIZE')),
      bool(_field(fields, 'SY')), 0, types=(typ, typ))
  if name == 'stg':
    typ = _field(fields, 'TYPE')
    return IR3Instruction('stg', None, (_reg(_field(fields, 'SRC1')), _reg(_field(fields, 'SRC3'), bool(_field_or(fields, 'TYPE_HALF', 0))),
      _memory_offset(fields), _field(fields, 'SIZE')), bool(_field(fields, 'SY')), 0, types=(typ, typ))
  if name == 'stg.a':
    typ = _field(fields, 'TYPE')
    return IR3Instruction(name, None, (_reg(_field(fields, 'SRC1')), _reg(_field(fields, 'SRC2')),
      _field(fields, 'FULL_SHIFT'), _reg(_field(fields, 'SRC3'), bool(_field_or(fields, 'TYPE_HALF', 0))), _field(fields, 'SIZE')),
      bool(_field(fields, 'SY')), 0, types=(typ, typ))
  if name == 'stib.b':
    if not _field_or(fields, 'TYPED', 0) or _field_or(fields, 'MODE', 0):
      raise NotImplementedError('unsupported IR3 stib mode')
    typ, components, dimensions = _field(fields, 'TYPE'), _field(fields, 'TYPE_SIZE'), _field(fields, 'D')
    return IR3Instruction('stib', None, (_reg(_field(fields, 'SRC2')), _reg(_field(fields, 'SRC1'),
      bool(_field_or(fields, 'TYPE_HALF', 0))), _field(fields, 'OFFSET'), _field(fields, 'SSBO'), dimensions, components),
      bool(_field(fields, 'SY')), 0, types=(typ, typ))
  if name == 'isam':
    if _field_or(fields, '3D', 0) or _field_or(fields, 'A', 0) or _field_or(fields, 'O', 0) or _field_or(fields, 'P', 0) or \
       _field_or(fields, 'SV', 0): raise NotImplementedError('unsupported IR3 isam modifier')
    typ, one_dimensional = _field(fields, 'TYPE'), not bool(_field_or(fields, '1D', 0))
    return IR3Instruction(name, _reg(_field(fields, 'DST'), bool(_field_or(fields, 'DST_HALF', 0))),
      (_reg(_field(fields, 'SRC')), _field(fields, 'SAMP'), _field(fields, 'TEX'), 1 if one_dimensional else 2,
       _field(fields, 'WRMASK')), bool(_field(fields, 'SY')), 0, types=(typ, typ))
  if name in {'ldl', 'ldp'}:
    typ = _field(fields, 'TYPE')
    return IR3Instruction(name, _reg(_field(fields, 'DST'), typ in (mesa.TYPE_F16, mesa.TYPE_U16, mesa.TYPE_S16, mesa.TYPE_U8)),
      (_reg(_field(fields, 'SRC')), _memory_offset(fields), _field(fields, 'SIZE')), bool(_field(fields, 'SY')), 0, types=(typ, typ))
  if name in {'stl', 'stp'}:
    typ = _field(fields, 'TYPE')
    return IR3Instruction(name, None, (_reg(_field(fields, 'DST')), _reg(_field(fields, 'SRC'),
      typ in (mesa.TYPE_F16, mesa.TYPE_U16, mesa.TYPE_S16, mesa.TYPE_U8)), _memory_offset(fields), _field(fields, 'SIZE')),
      bool(_field(fields, 'SY')), 0, types=(typ, typ))
  if name.startswith('atomic.g.'):
    if _field_or(fields, 'TYPED', 0): raise NotImplementedError(f'unsupported typed IR3 atomic {name}')
    if _field(fields, 'TYPE_SIZE') != 1: raise NotImplementedError(f'unsupported vector IR3 atomic {name}')
    typ = _field(fields, 'TYPE')
    srcs = tuple(_reg(_field(fields, src)) for src in ('SRC1', 'SRC2', 'SRC3') if any(field == src for field, _ in fields))
    atomic_dst = _reg(_field(fields, 'DST')) if _field_or(fields, 'D', 0) else None
    return IR3Instruction(name, atomic_dst, srcs, bool(_field(fields, 'SY')), 0, types=(typ, typ))
  if name == 'rcp':
    src, mod = _src(_field(fields, 'SRC'), bool(_field_or(fields, 'HALF', 0)))
    return IR3Instruction(name, _reg(_field(fields, 'DST'), bool(_field_or(fields, 'DST_HALF', 0))), (src,),
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
    return IR3Instruction(name, _reg(_field(fields, 'DST'), bool(_field_or(fields, 'DST_HALF', 0))), tuple(x[0] for x in srcs),
      bool(_field(fields, 'SY')), _field_or(fields, 'NOP', 0), _field_or(fields, 'REPEAT', 0), repeats,
      tuple(x[1] for x in srcs), condition, sat=bool(_field_or(fields, 'SAT', 0)), source_half=bool(_field_or(fields, 'HALF', 0)))
  if name in {'rsq', 'log2', 'exp2', 'sin', 'cos', 'sqrt', 'hrsq', 'hlog2', 'hexp2'}:
    src, mod = _src(_field(fields, 'SRC'), bool(_field_or(fields, 'HALF', 0)))
    return IR3Instruction(name, _reg(_field(fields, 'DST'), bool(_field_or(fields, 'DST_HALF', 0))), (src,), bool(_field(fields, 'SY')), 0,
      _field_or(fields, 'REPEAT', 0), (bool(_field_or(fields, 'SRC_R', 0)),), (mod,), source_half=bool(_field_or(fields, 'HALF', 0)))
  cat3 = {'mad.u16', 'madsh.u16', 'mad.s16', 'madsh.m16', 'mad.u24', 'mad.s24', 'mad.f16', 'mad.f32',
          'sel.b16', 'sel.b32', 'sel.s16', 'sel.s32', 'sel.f16', 'sel.f32', 'sad.s16', 'sad.s32'}
  if name in cat3:
    srcs = tuple(_src(_field(fields, src), bool(_field_or(fields, 'HALF', 0))) for src in ('SRC1', 'SRC2', 'SRC3'))
    mods = tuple(1 if _field_or(fields, f'SRC{i}_NEG', 0) else x[1] for i, x in enumerate(srcs, 1))
    return IR3Instruction(name, _reg(_field(fields, 'DST'), bool(_field_or(fields, 'DST_HALF', 0))), tuple(x[0] for x in srcs),
      bool(_field(fields, 'SY')), _field_or(fields, 'NOP', 0), _field_or(fields, 'REPEAT', 0),
      tuple(bool(_field_or(fields, f'SRC{i}_R', 0)) for i in range(1, 4)), mods, sat=bool(_field_or(fields, 'SAT', 0)))
  if name in {'shrm', 'shlm', 'shrg', 'shlg', 'andg'}:
    half = bool(_field_or(fields, 'HALF', 0))
    def alt_src(value): return value & 0xfff if value & 0x1000 else _src(value, half)[0]
    srcs = (alt_src(_field(fields, 'SRC1')), _src(_field(fields, 'SRC2'), half)[0], alt_src(_field(fields, 'SRC3')))
    return IR3Instruction(name, _reg(_field(fields, 'DST'), bool(_field_or(fields, 'DST_HALF', 0))), srcs,
      bool(_field(fields, 'SY')), _field_or(fields, 'NOP', 0), _field_or(fields, 'REPEAT', 0),
      tuple(bool(_field_or(fields, f'SRC{i}_R', 0)) for i in range(1, 4)), source_half=half)
  if name not in ('add.f', 'mul.f'):
    raise NotImplementedError(f'unsupported IR3 instruction {name}')
  return IR3Instruction(name, _reg(_field(fields, 'DST')),
                        tuple(_reg(_field(fields, src)) for src in ('SRC1', 'SRC2')),
                        bool(_field(fields, 'SY')), _field_or(fields, 'NOP', 0))

@functools.cache
def decode_ir3(code:bytes, gpu_id:int=630) -> tuple[IR3Instruction, ...]:
  if len(code) % 8:
    raise ValueError('IR3 code size must be a multiple of 8 bytes')
  raw: list[IR3Instruction] = []
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
