import ctypes, tempfile
from dataclasses import dataclass
from typing import Any
from tinygrad.runtime.autogen import libc, mesa

@dataclass(frozen=True)
class IR3Instruction:
  name: str
  dst: Any
  srcs: tuple[Any, ...]
  sy: bool
  nop: int
  repeat: int = 0
  repeat_srcs: tuple[bool, ...] = ()

def _field(fields, name):
  try: return next(value for field, value in fields if field == name)
  except StopIteration as exc:
    raise ValueError(f'missing IR3 field {name}') from exc

def _field_or(fields, name, default):
  return next((value for field, value in fields if field == name), default)

def _reject_unsupported_modifiers(fields, name):
  unsupported = {'SS', 'JP', 'SAT', 'UL', 'EI', 'ABSNEG', 'SRC1_R', 'SRC2_R', 'SRC3_R',
                 'SRC1_NEG', 'SRC2_NEG', 'SRC3_NEG', 'TYPE_HALF', 'ROUND', 'EQ'}
  if bad := next((field for field, value in fields if field in unsupported and value), None):
    raise NotImplementedError(f'unsupported IR3 modifier {bad}')
  if name != 'add.u' and any(field == 'SRC_R' and value for field, value in fields):
    raise NotImplementedError('unsupported IR3 modifier SRC_R')
  if name != 'cmps.u' and _field_or(fields, 'DST_HALF', 0): raise NotImplementedError('unsupported IR3 modifier DST_HALF')
  if name is not None and any(field == 'HALF' and value for field, value in fields):
    raise NotImplementedError('unsupported IR3 modifier HALF')

def _reg(value, half=False):
  constant = bool(value & 0x1000)
  value &= 0xfff
  if not constant and value // 4 in (61, 62): raise NotImplementedError('unsupported IR3 special register')
  kind = ('hc' if constant else 'hr') if half else ('c' if constant else 'r')
  return kind, value // 4, value % 4

def _decode_fields(fields):
  name = next((value for field, value in fields if field == 'NAME'), None)
  if name not in ('nop', 'add.u') and _field_or(fields, 'REPEAT', 0):
    raise NotImplementedError(f'repeated {name or "conversion"} is unsupported')
  _reject_unsupported_modifiers(fields, name)
  if name is None:
    types = (_field(fields, 'SRC_TYPE'), _field(fields, 'DST_TYPE'))
    if types == (mesa.TYPE_U32, mesa.TYPE_U32):
      return IR3Instruction('mov.u32u32', _reg(_field(fields, 'DST')), (_reg(_field(fields, 'SRC')),),
                            bool(_field(fields, 'SY')), 0)
    if types != (mesa.TYPE_U16, mesa.TYPE_S32):
      raise NotImplementedError(f'unsupported IR3 conversion {fields}')
    return IR3Instruction('cov.u16s32', _reg(_field(fields, 'DST')),
      (_reg(_field(fields, 'SRC'), bool(_field(fields, 'HALF'))),), bool(_field(fields, 'SY')), 0, _field_or(fields, 'REPEAT', 0))
  if name == 'nop':
    return IR3Instruction(name, None, (), bool(_field(fields, 'SY')), 0)
  if name == 'end':
    return IR3Instruction(name, None, (), bool(_field(fields, 'SY')), 0)
  if name in ('ashr.b', 'shl.b'):
    return IR3Instruction(name, _reg(_field(fields, 'DST')),
      (_reg(_field(fields, 'SRC1')), _field(fields, 'IMMED')),
      bool(_field(fields, 'SY')), _field_or(fields, 'NOP', 0), _field_or(fields, 'REPEAT', 0))
  if name == 'shrg':
    return IR3Instruction(name, _reg(_field(fields, 'DST')),
      (_field(fields, 'IMMED'), _reg(_field(fields, 'SRC2')), _reg(_field(fields, 'SRC3'))),
      bool(_field(fields, 'SY')), 0, _field_or(fields, 'REPEAT', 0))
  if name == 'mull.u':
    if not any(field == 'IMMED' for field, _ in fields): raise NotImplementedError('register mull.u is unsupported')
    return IR3Instruction(name, _reg(_field(fields, 'DST')), (_reg(_field(fields, 'SRC1')), _field(fields, 'IMMED')),
                          bool(_field(fields, 'SY')), _field_or(fields, 'NOP', 0))
  if name == 'madsh.m16':
    return IR3Instruction(name, _reg(_field(fields, 'DST')),
      tuple(_reg(_field(fields, src)) for src in ('SRC1', 'SRC2', 'SRC3')),
      bool(_field(fields, 'SY')), _field_or(fields, 'NOP', 0))
  if name == 'add.u':
    repeat = _field_or(fields, 'REPEAT', 0)
    immediate = any(field == 'IMMED' for field, _ in fields)
    if repeat and immediate: raise NotImplementedError('repeated add.u immediate is unsupported')
    src2 = _field(fields, 'IMMED') if immediate else _reg(_field(fields, 'SRC2'))
    repeat_srcs = tuple(bool(value) for field, value in fields if field == 'SRC_R')
    if len(repeat_srcs) != 2: raise ValueError('invalid add.u repeat fields')
    return IR3Instruction(name, _reg(_field(fields, 'DST')), (_reg(_field(fields, 'SRC1')), src2),
      bool(_field(fields, 'SY')), 0, repeat, repeat_srcs)
  if name == 'cmps.u':
    if _field(fields, 'COND') != mesa.IR3_COND_LT: raise NotImplementedError('unsupported cmps.u condition')
    return IR3Instruction('cmps.u.lt', _reg(_field(fields, 'DST'), bool(_field(fields, 'DST_HALF'))),
      tuple(_reg(_field(fields, src)) for src in ('SRC1', 'SRC2')),
      bool(_field(fields, 'SY')), 0, _field_or(fields, 'REPEAT', 0))
  if name == 'ldg':
    if _field(fields, 'TYPE') != mesa.TYPE_U32: raise NotImplementedError('unsupported ldg type')
    return IR3Instruction('ldg.u32', _reg(_field(fields, 'DST')),
      (_reg(_field(fields, 'SRC1')), _field(fields, 'OFF'), _field(fields, 'SIZE')),
      bool(_field(fields, 'SY')), 0)
  if name == 'stg':
    if _field(fields, 'TYPE') != mesa.TYPE_U32: raise NotImplementedError('unsupported stg type')
    return IR3Instruction('stg.u32', None,
      (_reg(_field(fields, 'SRC1')), _reg(_field(fields, 'SRC3')), _field(fields, 'OFF'), _field(fields, 'SIZE')),
      bool(_field(fields, 'SY')), 0)
  if name != 'add.f':
    raise NotImplementedError(f'unsupported IR3 instruction {name}')
  return IR3Instruction(name, _reg(_field(fields, 'DST')),
                        tuple(_reg(_field(fields, src)) for src in ('SRC1', 'SRC2')),
                        bool(_field(fields, 'SY')), _field_or(fields, 'NOP', 0))

def decode_ir3(code:bytes, gpu_id:int=630) -> list[IR3Instruction]:
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
    try: raw.append(_decode_fields(current.copy()))
    except Exception as exc: errors.append(exc)

  with tempfile.TemporaryFile('w+') as tf:
    fp = libc.fdopen(tf.fileno(), b'w')
    opts = mesa.struct_isa_decode_options(gpu_id, True, 0, False, field_cb=field,
                                          pre_instr_cb=pre, post_instr_cb=post)
    out = ctypes.cast(fp, ctypes.POINTER(mesa.struct__IO_FILE))
    mesa.ir3_isa_disasm(code, len(code), out, opts)
    libc.fflush(fp)
  if errors: raise errors[0]
  if len(raw) != len(code) // 8: raise ValueError('invalid IR3 instruction encoding')
  return raw

def _as_f32(bits): return ctypes.c_float.from_buffer_copy(ctypes.c_uint32(bits)).value

def _f32_bits(value): return ctypes.c_uint32.from_buffer_copy(ctypes.c_float(value)).value

def local_id_regs(local_size, register):
  x_size, y_size, z_size = local_size
  lanes = [(x, y, z) for z in range(z_size) for y in range(y_size) for x in range(x_size)]
  return {('r', register, component): [lane[component] for lane in lanes] for component in range(3)}

def workgroup_id_regs(workgroup_id, lane_count, register):
  return {('r', register, component): [workgroup_id[component]] * lane_count for component in range(3)}

def _next_reg(reg):
  kind, number, component = reg
  return kind, number + (component == 3), (component + 1) % 4

def _reg_offset(reg, offset):
  kind, number, component = reg
  return kind, number + (component + offset) // 4, (component + offset) % 4

def execute_ir3(code:bytes, regs:dict[tuple[str, int, int], list[int]], gpu_id:int=630, check_range=None):
  for inst in decode_ir3(code, gpu_id):
    if inst.name == 'nop': continue
    if inst.name == 'end': break
    if inst.name == 'ashr.b':
      src, shift = inst.srcs
      shift &= 31
      regs[inst.dst] = [(ctypes.c_int32(value).value >> shift) & 0xffffffff for value in regs[src]]
      continue
    if inst.name == 'shl.b':
      src, shift = inst.srcs
      shift &= 31
      regs[inst.dst] = [(value << shift) & 0xffffffff for value in regs[src]]
      continue
    if inst.name == 'shrg':
      shift, src2, src3 = inst.srcs
      regs[inst.dst] = [(((value & 0xffffffff) >> (shift & 31)) | other) & 0xffffffff
                        for value, other in zip(regs[src2], regs[src3], strict=True)]
      continue
    if inst.name == 'mull.u':
      src, multiplier = inst.srcs
      regs[inst.dst] = [(value * multiplier) & 0xffffffff for value in regs[src]]
      continue
    if inst.name == 'madsh.m16':
      src0, src1, src2 = (regs[src] for src in inst.srcs)
      regs[inst.dst] = [((((x & 0xffff) * ((y >> 16) & 0xffff)) << 16) + z) & 0xffffffff
                        for x, y, z in zip(src0, src1, src2, strict=True)]
      continue
    if inst.name == 'add.u':
      for component in range(inst.repeat + 1):
        srcs = tuple(_reg_offset(src, component) if repeated and not isinstance(src, int) else
                     src + component if repeated else src
                     for src, repeated in zip(inst.srcs, inst.repeat_srcs, strict=True))
        lhs = regs[srcs[0]]
        rhs = [srcs[1]] * len(lhs) if isinstance(srcs[1], int) else regs[srcs[1]]
        regs[_reg_offset(inst.dst, component)] = [(x + y) & 0xffffffff for x, y in zip(lhs, rhs, strict=True)]
      continue
    if inst.name == 'cmps.u.lt':
      lhs, rhs = (regs[src] for src in inst.srcs)
      regs[inst.dst] = [int((x & 0xffffffff) < (y & 0xffffffff)) for x, y in zip(lhs, rhs, strict=True)]
      continue
    if inst.name == 'cov.u16s32':
      regs[inst.dst] = [value & 0xffff for value in regs[inst.srcs[0]]]
      continue
    if inst.name == 'mov.u32u32':
      regs[inst.dst] = regs[inst.srcs[0]].copy()
      continue
    if inst.name == 'ldg.u32':
      address_reg, offset, size = inst.srcs
      if not 1 <= size <= 4: raise NotImplementedError(f'unsupported ldg size {size}')
      addresses = [low | high << 32 for low, high in zip(regs[address_reg], regs[_next_reg(address_reg)], strict=True)]
      if check_range is not None:
        for address in addresses: check_range(address + offset, size * 4)
      for component in range(size):
        regs[_reg_offset(inst.dst, component)] = [ctypes.c_uint32.from_address(address + offset + component * 4).value
                                                   for address in addresses]
      continue
    if inst.name == 'stg.u32':
      address_reg, value_reg, offset, size = inst.srcs
      if not 1 <= size <= 4: raise NotImplementedError(f'unsupported stg size {size}')
      addresses = [low | high << 32 for low, high in zip(regs[address_reg], regs[_next_reg(address_reg)], strict=True)]
      if check_range is not None:
        for address in addresses: check_range(address + offset, size * 4)
      for component in range(size):
        for address, value in zip(addresses, regs[_reg_offset(value_reg, component)], strict=True):
          ctypes.c_uint32.from_address(address + offset + component * 4).value = value
      continue
    if inst.name != 'add.f': raise NotImplementedError(f'unsupported IR3 execution {inst.name}')
    lhs, rhs = (regs[src] for src in inst.srcs)
    regs[inst.dst] = [_f32_bits(_as_f32(x) + _as_f32(y)) for x, y in zip(lhs, rhs, strict=True)]

def execute_dispatch(code, grid_size, local_size, local_id_register, initial_regs=None, check_range=None, workgroup_id_register=0xfc):
  lane_count = local_size[0] * local_size[1] * local_size[2]
  last_regs: dict[tuple[str, int, int], list[int]] = {}
  for z in range(grid_size[2]):
    for y in range(grid_size[1]):
      for x in range(grid_size[0]):
        regs = {} if initial_regs is None else initial_regs.copy()
        regs.update(local_id_regs(local_size, local_id_register))
        if workgroup_id_register != 0xfc: regs.update(workgroup_id_regs((x, y, z), lane_count, workgroup_id_register))
        execute_ir3(code, regs, check_range=check_range)
        last_regs = regs
  return last_regs
