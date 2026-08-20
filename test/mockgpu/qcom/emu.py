import ctypes, tempfile
from dataclasses import dataclass
from tinygrad.runtime.autogen import libc, mesa

@dataclass(frozen=True)
class IR3Instruction:
  name: str
  dst: tuple[str, int, int] | None
  srcs: tuple[tuple[str, int, int], ...]
  sy: bool
  nop: int

def _field(fields, name):
  try: return next(value for field, value in fields if field == name)
  except StopIteration as exc:
    raise ValueError(f'missing IR3 field {name}') from exc

def _reg(value):
  return 'r', value // 4, value % 4

def _decode_fields(fields):
  name = _field(fields, 'NAME')
  if name == 'nop':
    return IR3Instruction(name, None, (), bool(_field(fields, 'SY')), 0)
  if name != 'add.f':
    raise NotImplementedError(f'unsupported IR3 instruction {name}')
  return IR3Instruction(name, _reg(_field(fields, 'DST')),
                        tuple(_reg(_field(fields, src)) for src in ('SRC1', 'SRC2')),
                        bool(_field(fields, 'SY')), _field(fields, 'NOP'))
                 
def decode_ir3(code:bytes, gpu_id:int=630) -> list[IR3Instruction]:
  if len(code) % 8:
    raise ValueError('IR3 code size must be a multiple of 8 bytes')
  raw, current, errors = [], [], []

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
    try: raw.append(_decode_fields(current))
    except Exception as exc: errors.append(exc)
    
  with tempfile.TemporaryFile('w+') as tf:
    fp = libc.fdopen(tf.fileno(), b'w')
    opts = mesa.struct_isa_decode_options(gpu_id, True, 0, False, field_cb=field,
                                          pre_instr_cb=pre, post_instr_cb=post)
    out = ctypes.cast(fp, ctypes.POINTER(mesa.struct__IO_FILE))
    mesa.ir3_isa_disasm(code, len(code), out, opts)
    libc.fflush(fp)
  if errors: raise errors[0]
  return raw
  
def _as_f32(bits): return ctypes.c_float.from_buffer_copy(ctypes.c_uint32(bits)).value

def _f32_bits(value): return ctypes.c_uint32.from_buffer_copy(ctypes.c_float(value)).value

def execute_ir3(code:bytes, regs:dict[tuple[str, int, int], list[int]], gpu_id:int=630):
  for inst in decode_ir3(code, gpu_id):
    if inst.name == 'nop': continue
    if inst.name != 'add.f': raise NotImplementedError(f'unsupported IR3 execution {inst.name}')
    lhs, rhs = (regs[src] for src in inst.srcs)
    regs[inst.dst] = [_f32_bits(_as_f32(x) + _as_f32(y)) for x, y in zip(lhs, rhs, strict=True)]
