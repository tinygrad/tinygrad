"""A630 scalar ISA semantics evaluated over NumPy lanes of one workgroup."""

import ctypes, functools, itertools, math, os
from dataclasses import dataclass, replace
import numpy as np
from tinygrad.runtime.autogen import libc, mesa
from test.mockgpu.qcom.image import ImageBindings


# qcomgpu.Launch.apply supplies the validated shader image, constant bytes,
# dispatch dimensions and register/resource configuration recovered from PM4.
# This module turns that image into instructions, then evolves each workgroup:
# fetch at a lane's PC -> select operands -> compute -> write back state.
# Global writes use the caller's staged memory; the queue owns their commit.


# An operand identifies a scalar component in a register bank, a constant slot,
# or an immediate value. Precision and modifiers travel with that selection.
@dataclass(frozen=True)
class Operand:
  kind: str
  index: int
  half: bool = False
  repeat: bool = False
  modifier: int = 0
  relative: bool = False


# Keep encoding/flags alongside normalized operands: decoding establishes what
# an instruction may do, while execution supplies the current lane values.
@dataclass(frozen=True)
class Instruction:
  op: str
  fields: dict
  operands: dict
  raw: int


# Numeric and instruction policy. These tables describe the supported ISA
# forms and the conversions permitted between full and half register results.
# Despite its U8 name, cat1 conversion sign-extends byte inputs. Mesa emits
# AND 0xff instead for unsigned widening (ir3_compiler_nir.c:create_cov).
TYPES = (np.float16, np.float32, np.uint16, np.uint32, np.int16, np.int32, np.int8, np.uint8)
# Keep the ISA's FLUT index order, including the final 4.0 slot.
FLUT = (
  0.0, 0.5, 1.0, 2.0,
  math.e, math.pi, 1 / math.pi,
  math.log(2), math.log2(math.e), 1 / math.log2(10), math.log2(10), 4.0,
)
SUPPORTED = set(
  (
    'mov swz nop end predt predf prede bar fence br brao braa jump call ret ldg stg ldg.a stg.a ldl stl ldp stp '
    'add.u add.s add.f sub.u sub.s sub.f mul.f mul.s24 mul.u24 mull.u madsh.m16 mad.u16 mad.s24 mad.f32 mad.f16 '
    'shl.b shr.b ashr.b shrg shrm shlm shlg andg and.b or.b xor.b not.b absneg.s absneg.f sign.f clz.b getbit.b '
    'min.u min.s min.f max.u max.s max.f sel.b32 sel.b16 sel.s32 cmps.u cmps.s cmps.f cmpv.s cmpv.f cmpv.u '
    'floor.f ceil.f trunc.f rcp sqrt rsq exp2 log2 sin cos sad.s16 sad.s32'
  ).split()
)
FIELD_NAMES = set(
  (
    'SY SS JP SAT REPEAT NOP UL NAME EI DST_HALF DST DST0 DST1 SRC0 SRC1 SRC2 SRC3 SRC SRC_R LAST ABSNEG HALF '
    'GPR SWIZ CONST IMMED SRC_TYPE DST_TYPE ROUND TYPE TYPE_HALF OFF SIZE COND INV1 COMP1 INV2 COMP2 EQ G L R W REG '
    'SRC1_NEG SRC2_NEG SRC3_NEG SRC1_R SRC2_R SRC3_R FULL_SHIFT TYPE_SHIFT SRC2_SHIFT OFFSET'
  ).split()
)
BITWISE_OPS = {'and.b', 'or.b', 'not.b', 'xor.b', 'shl.b', 'shr.b', 'ashr.b'}
SIGNED_OUTPUTS = {'add.s', 'sub.s', 'min.s', 'max.s', 'absneg.s', 'mul.s24', 'mad.s24', 'sad.s16', 'sad.s32'}
FLOAT_OUTPUTS = {'add.f', 'mul.f', 'mad.f16', 'mad.f32'}
UNSIGNED_OUTPUTS = {
  'add.u', 'sub.u', 'min.u', 'max.u', 'mull.u', 'mul.u24', 'mad.u16',
  'shrg', 'shrm', 'shlm', 'shlg', 'andg',
  'cmps.u', 'cmps.s', 'cmps.f', 'cmpv.s', 'cmpv.f', 'cmpv.u', 'getbit.b',
} | BITWISE_OPS
UNARY_BASES = {
  'not', 'absneg', 'sign', 'clz',
  'floor', 'ceil', 'trunc',
  'rcp', 'sqrt', 'rsq',
  'exp2', 'log2',
  'sin', 'cos',
}


# Representation boundaries: register storage carries bits; ALU operations
# temporarily interpret those bits as the instruction's numeric type.
def _typed(bits, typ):
  dtype = np.dtype(TYPES[typ])
  unsigned = np.dtype(f'u{dtype.itemsize}')
  return bits.astype(unsigned).view(dtype)


def _signed(value, width):
  # Encoded immediates and offsets share two's-complement sign extension, but
  # retain their own field widths (11, 13, 24 or 32 bits at the call sites).
  sign = 1 << (width - 1)
  return ((value & (2 * sign - 1)) ^ sign) - sign


def _bits(value, typ, round_zero=False):
  dtype = np.dtype(TYPES[typ])
  with np.errstate(over='ignore', invalid='ignore'):
    converted = np.asarray(value).astype(dtype)
    if round_zero and dtype.kind == 'f':
      # Start from the nearest value and step toward zero only if it overshot.
      overshot = np.abs(converted.astype(np.float64)) > np.abs(np.asarray(value).astype(np.float64))
      converted = np.where(overshot, np.nextafter(converted, np.zeros_like(converted)), converted)
  return converted.view(np.dtype(f'u{dtype.itemsize}')).astype(np.uint32)


def _source_typed(bits, operand, typ, *, constant_demotion=True):
  # CONSTANT_DEMOTION_ENABLE keeps each constant slot 32 bits wide. A half
  # floating source converts its F32 value; an integer source takes low bits.
  # Model the implicit conversion as the default narrowing COV that Mesa folds
  # into ALU sources (ir3.h:is_const_mov), independently of output rounding.
  if constant_demotion and operand.kind == 'constant' and typ == 0:
    bits = _bits(_typed(bits, 1), 0, round_zero=True)
  return _typed(bits, typ)


# Decode one operand's nested Mesa fields before discarding their local order.
# The same field name can recur in another operand, so a whole-instruction
# dictionary alone cannot select register/constant components correctly.
def _operand(name, fields, top, source_half, raw):
  values = dict(fields)
  default_half = top.get('DST_HALF', top.get('TYPE_HALF', 0)) if name.startswith('DST') else source_half
  half = bool(values.get('HALF', values.get('DST_HALF', default_half)))
  modifier = values.get('ABSNEG', top.get(name + '_NEG', 0))
  repeat = bool(values.get('SRC_R', top.get(name + '_R', 0)))
  category, destination = raw >> 61, name.startswith('DST')
  relative = relative_constant = False
  if category == 1 and (raw >> 57) & 3 == 0:
    relative = bool(raw & (1 << 49)) if destination else bool(raw & (1 << 11)) and (raw >> 53) & 3 == 0
    relative_constant = not destination and bool(raw & (1 << 10))
  elif category in (2, 4) and not destination:
    encoded = (raw >> (16 if name == 'SRC2' else 0)) & 0xffff
    relative = (encoded >> 11) & 7 == 1
    relative_constant = bool(encoded & (1 << 10))
  if relative:
    # Cat1 destinations have unsigned8 offsets; relative sources use signed10.
    # Both offsets count scalar components in the bank selected by the operand.
    # The display callback omits OFFSET when it is zero, so use the mode bits.
    offset = (raw >> 32) & 255 if destination else _signed(values.get('OFFSET', 0), 10)
    return Operand('constant' if relative_constant else 'register', offset, half, repeat, modifier, relative=True)
  if 'OFFSET' in values:
    raise RuntimeError('IR3 unexpected relative-address field')
  if 'IMMED' in values:
    kind = 'float_immediate' if 'SRC_TYPE' not in top and (fields[0][1] >> 11) & 7 == 5 else 'immediate'
    if kind == 'float_immediate' and values['IMMED'] >= len(FLUT):
      raise RuntimeError('IR3 invalid floating immediate')
    if kind == 'float_immediate':
      half = bool(fields[0][1] & (1 << 10))
      if half != source_half:
        raise RuntimeError('IR3 mixed precision immediate is unsupported')
    if repeat:
      raise RuntimeError('IR3 repeated immediate operands are unsupported')
    immediate = values['IMMED']
    if 'SRC_TYPE' not in top and (fields[0][1] >> 11) & 7 == 4:
      immediate = _signed(immediate, 11)
    return Operand(kind, immediate, half, repeat, modifier)
  if 'CONST' in values:
    return Operand('constant', values['CONST'] * 4 + values.get('SWIZ', 0), half, repeat, modifier)
  index = values['GPR'] * 4 + values.get('SWIZ', 0) if 'GPR' in values else fields[0][1] & 255
  return Operand('register', index, half, repeat, modifier)


def memory_roles(op):
  # ld/st selects direction; g/l/p selects global, local/shared or private
  # storage. In a local/private store, DST names an address rather than a result.
  base = op.split('.')[0]
  loading = base.startswith('ld')
  address = 'SRC1' if base.endswith('g') else 'SRC' if loading else 'DST'
  return address, 'DST' if loading else 'SRC3' if base.endswith('g') else 'SRC'


# Reject unsupported operand forms before they can enter the state-update loop.
def _validate_operands(op, operands, fields, category):
  if op == 'swz':
    expected = {'DST0', 'DST1', 'SRC0', 'SRC1'}
  elif op in ('ldg', 'ldg.a'):
    expected = {'DST', 'SRC1'} | ({'SRC2'} if op.endswith('.a') else set())
  elif op in ('stg', 'stg.a'):
    expected = {'SRC1', 'SRC3'} | ({'SRC2'} if op.endswith('.a') else set())
  elif category in (0, 7):
    expected = set()
  elif category in (1, 4, 6):
    expected = {'DST', 'SRC'}
  elif op.split('.')[0] in UNARY_BASES:
    expected = {'DST', 'SRC1'}
  elif category == 3:
    expected = {'DST', 'SRC1', 'SRC2', 'SRC3'}
  else:
    expected = {'DST', 'SRC1', 'SRC2'}
  if set(operands) != expected:
    raise RuntimeError(f'IR3 {op}: invalid operand schema')
  # Mesa ir3_cf.c and ir3_validate.c forbid 16-bit inputs/outputs for MAD.x24.
  if op == 'mad.s24' and any(operand.half for operand in operands.values()):
    raise RuntimeError('IR3 mad.s24 requires full registers')
  if op == 'getbit.b':
    # QCOMCL restores loop-hoisted packed booleans with a half-register bit
    # test into p0.x..w. Other result representations are not qualified here.
    dst, source, bit = (operands[name] for name in ('DST', 'SRC1', 'SRC2'))
    if (dst.half or not 248 <= dst.index <= 251 or source.kind != 'register' or not source.half or
        bit.kind != 'immediate' or not 0 <= bit.index < 16 or fields.get('REPEAT', 0)):
      raise RuntimeError('IR3 unsupported GETBIT predicate form')
  if op.startswith(('cmps.', 'cmpv.')) and not 0 <= fields.get('COND', -1) <= 5:
    raise RuntimeError('IR3 invalid comparison condition')
  if op.startswith(('cmps.', 'cmpv.')) and fields.get('SAT', 0):
    # QCOMCL uses this bit to invert F32 LE/GE predicate comparisons. Other
    # comparison/destination forms need their own compiler evidence.
    if (op != 'cmps.f' or fields['COND'] not in (1, 3) or
        not 248 <= operands['DST'].index <= 251 or any(operand.half for operand in operands.values())):
      raise RuntimeError('IR3 unsupported comparison modifier')
  modifier_mask = (
    1
    if op in BITWISE_OPS or category == 3 and op.startswith('mad.f')
    else (3 if '.f' in op or category == 4 or op == 'absneg.s' else 0)
  )
  memory_data = memory_roles(op)[1] if category == 6 else None
  for name, operand in operands.items():
    allowed_modifier = int(name == 'SRC2') if op.startswith('sad.') else modifier_mask
    if operand.modifier & ~allowed_modifier:
      raise RuntimeError(f'IR3 {op}: unsupported source modifier')
    destination = name.startswith('DST') and op not in ('stl', 'stp')
    if destination and operand.kind != 'register':
      raise RuntimeError('IR3 invalid destination operand')
    last = fields.get('REPEAT', 0) if destination or operand.repeat else 0
    if category == 6:
      last = fields['SIZE'] - 1 if name == memory_data else int(name == 'SRC1' and op.startswith(('ldg', 'stg')))
      if operand.kind != 'register' or operand.relative:
        raise RuntimeError('IR3 invalid memory register operand')
    if operand.kind == 'register' and not operand.relative and not 0 <= operand.index <= 255 - last:
      raise RuntimeError(f'IR3 {"destination" if destination else "source"} register span out of bounds')
  if category in (2, 3, 4):
    source = next(operand for name, operand in operands.items() if name.startswith('SRC'))
    # Cat4 explicitly encodes opposite-precision destinations with DST_CONV.
    if operands['DST'].half != source.half and category != 4 and op not in SIGNED_OUTPUTS | UNSIGNED_OUTPUTS | FLOAT_OUTPUTS:
      raise RuntimeError(f'IR3 {op}: unsupported output conversion')


# One 64-bit encoding becomes a decoded instruction with checked operand roles.
def _instruction(fields, raw):
  top = dict(fields)
  # Mesa's MOVA display omits its implicit destination and types from callbacks.
  # Recover those exact cat1 fields before using the ordinary MOV machinery.
  category, destination = raw >> 61, (raw >> 32) & 255
  source_type, destination_type = (raw >> 50) & 7, (raw >> 46) & 7
  if category == 1 and (destination, source_type, destination_type) in ((244, 4, 4), (245, 2, 2)) and 'SRC_TYPE' not in top:
    fields = [('DST', destination), ('DST_HALF', 1), ('SRC_TYPE', source_type), ('DST_TYPE', destination_type), *fields]
    if 'IMMED' in top and not raw & (1 << 49):
      # SRC_R on an immediate MOVA is a hazard marker, not source repetition.
      fields = [(name, 0 if name == 'SRC_R' else value) for name, value in fields]
    top = dict(fields)
  op = top.get('NAME', 'swz' if 'DST0' in top else 'mov' if 'SRC_TYPE' in top else '')
  # Mesa uses distinct cat4 opcodes for these half special functions. Their
  # numeric operation is shared; FULL and DST_CONV still select the banks and
  # rounding in operand admission and ALU writeback (ir3-cat4.xml).
  if raw >> 61 == 4:
    op = {'hrsq': 'rsq', 'hlog2': 'log2', 'hexp2': 'exp2'}.get(op, op)
  if op in ('isam', 'stib.b', 'ldib.b'):
    return _image_instruction(top, raw)
  if op not in SUPPORTED:
    raise RuntimeError(f'IR3 unsupported instruction {op or hex(raw)}')
  if unknown := top.keys() - FIELD_NAMES:
    raise RuntimeError(f'IR3 unsupported fields {sorted(unknown)}')
  if top.get('ROUND', 0) and not (op == 'mov' and top['ROUND'] == 1 and top['DST_TYPE'] in (0, 1)):
    raise RuntimeError('IR3 conversion rounding mode is unsupported')
  if op == 'swz' and top['SRC_TYPE'] != top['DST_TYPE']:
    raise RuntimeError('IR3 converting swizzles are unsupported')
  if op == 'mov' and (
    (top['SRC_TYPE'] == 6 and top['DST_TYPE'] in (0, 1))
    or (top['DST_TYPE'] == 6 and top['SRC_TYPE'] in (0, 1))
  ):
    raise RuntimeError('IR3 direct byte/float conversion is unsupported')
  # EQ on NOP ends fragment helper invocations (ir3_legalize.c:helper_sched).
  # Compute has no helper lanes. Keep other EQ forms unsupported; some cat0
  # display callbacks omit EQ, so read its common encoding bit directly.
  end_of_quad = raw >> 61 == 0 and bool(raw & (1 << 48))
  # OpenCL hadd directly emits ADD.U(EI): retain the carry bit until the
  # mathematical sum is halved. Other EI forms are still unsupported.
  halving_add = op == 'add.u' and bool(raw & (1 << 52)) and not top.get('DST_HALF', 0)
  if (end_of_quad and op != 'nop') or (top.get('EI', 0) and not halving_add):
    raise RuntimeError('IR3 unsupported execution modifier')
  if top.get('SAT', 0) and '.f' not in op and raw >> 61 != 4:
    raise RuntimeError('IR3 integer saturation is unsupported')
  operands = {}
  # FULL is not printed for immediate operands, so it has no field callback.
  # Its shared cat2/cat4 encoding is bit 52 (ir3-cat2.xml, ir3-cat4.xml).
  source_half = not bool(raw & (1 << 52)) if raw >> 61 in (2, 4) else top.get('HALF', 0)
  # Nested field names repeat. Preserve the leaf values within each operand.
  names = (
    ('DST0', 'DST1', 'SRC0', 'SRC1')
    if op == 'swz'
    else ('DST', 'SRC')
    if 'SRC_TYPE' in top or raw >> 61 == 4
    else (('DST', 'SRC1', 'SRC2', 'SRC3') if raw >> 61 in (2, 3) else ('DST', 'SRC1', 'SRC2', 'SRC3', 'SRC'))
  )
  starts: list[tuple[int, str]] = []
  for i, (name, _) in enumerate(fields):
    if name in names and name not in [x[1] for x in starts]:
      starts.append((i, name))
  for k, (start, name) in enumerate(starts):
    stop = starts[k + 1][0] if k + 1 < len(starts) else len(fields)
    operands[name] = _operand(name, fields[start:stop], top, source_half, raw)
  if op == 'mad.u16':
    # The compiler's wide MAD.U16 uses full GPR/constant slots: only the two
    # multiplicands' low16 bits participate, while the accumulator is full32.
    # Mesa's derived HALF labels describe the digit width for this opcode.
    if operands['DST'].half:
      raise RuntimeError('IR3 MAD.U16 half destination is unsupported')
    for name in ('SRC1', 'SRC2', 'SRC3'):
      operands[name] = replace(operands[name], half=False)
  if op in ('ldg', 'stg', 'ldg.a', 'stg.a', 'ldl', 'stl', 'ldp', 'stp') and not 1 <= top.get('SIZE', 0) <= 4:
    raise RuntimeError('IR3 invalid memory vector width')
  if raw >> 61 == 6:
    address, data = memory_roles(op)
    operands[address] = replace(operands[address], half=False)
    operands[data] = replace(operands[data], half=top['TYPE'] in (0, 2, 4, 6))
    if op.endswith('.a'):
      # Mesa's display callbacks may emit only FULL_SHIFT when OFF is zero.
      # Retain the actual A6xx address-calculation fields independently of display.
      top['SRC2_SHIFT'], top['OFF'] = raw >> 12 & 3, raw >> 9 & 3
      operands['SRC2'] = replace(operands['SRC2'], half=False)
  _validate_operands(op, operands, top, raw >> 61)
  return Instruction(op, top, operands, raw)


def _image_instruction(fields, raw):
  # Image instructions have different operand roles from buffer cat6 operations.
  # Decode those roles explicitly: cat5's empty SRC2 is not a register, and the
  # cat6 OFFSET printed after SRC2 belongs to the instruction, not its register.
  op = fields['NAME']
  typ = fields['TYPE']
  if typ not in (0, 1):
    raise RuntimeError('IR3 image instructions require a floating data type')
  if op == 'isam':
    destination, coordinates = fields['DST'], fields['SRC1']
    mask = fields['WRMASK']
    if not mask or not 0 <= destination <= 256 - mask.bit_length() or not 0 <= coordinates <= 254:
      raise RuntimeError('IR3 image register span out of bounds')
    expected = (5 << 61) | (fields['SY'] << 60) | (fields['JP'] << 59)
    expected |= typ << 44 | mask << 40 | destination << 32 | coordinates << 1 | 1
    operands = {
      'DST': Operand('register', destination, half=typ == 0),
      'COORD': Operand('register', coordinates),
    }
    if raw & (1 << 51):
      # The non-bindless uniform S2EN form selects texture then sampler from a
      # pair of half registers. This is emitted when sampler index exceeds 15.
      indices = fields['SRC3']
      if not 0 <= indices <= 254:
        raise RuntimeError('IR3 image index register span out of bounds')
      expected |= (1 << 51) | indices << 21
      operands['INDICES'] = Operand('register', indices, half=True)
    else:
      expected |= fields['SAMP'] << 21 | fields['TEX'] << 25
    if raw != expected:
      raise RuntimeError('IR3 unsupported image sampling mode')
  else:
    data, coordinates, index = fields['SRC1'], fields['SRC2'], fields['SSBO']
    if fields['D'] != 2 or fields['TYPED'] != 1 or fields['TYPE_SIZE'] != 4:
      raise RuntimeError('IR3 unsupported image load/store shape')
    if not 0 <= data <= 252 or not 0 <= coordinates <= 254:
      raise RuntimeError('IR3 image register span out of bounds')
    # MODE=0 selects a bound descriptor by immediate index. The opcode's .b
    # suffix names the A6xx encoding; it does not itself select bindless access.
    expected = (6 << 61) | fields['SY'] << 60 | fields['JP'] << 59 | (2 << 52)
    expected |= typ << 49 | index << 41 | data << 32 | coordinates << 24
    opcode = 6 if op == 'ldib.b' else 29
    expected |= (6 << 20) | (opcode << 14) | (3 << 12) | (1 << 11) | (1 << 9)  # Four typed components, 2D.
    # Mesa marks LDIB bit 0 as dontcare; QCOMCL sets it, while IR3 clears it.
    if op == 'ldib.b': expected |= raw & 1
    if raw != expected:
      raise RuntimeError('IR3 unsupported image load/store mode')
    operands = {
      'DST' if op == 'ldib.b' else 'DATA': Operand('register', data, half=typ == 0),
      'COORD': Operand('register', coordinates),
    }
  return Instruction(op, fields, operands, raw)


# Decode is shared by identical images. All words, including padding after END,
# are checked before execute() creates any mutable workgroup state.
@functools.lru_cache(maxsize=256)
def decode(image):
  if not image or len(image) % 8:
    raise RuntimeError('IR3 image must contain complete 64-bit instructions')
  result, fields, errors = [], [], []

  # Consume structured callbacks rather than printed assembly. Callback errors
  # must cross the C boundary explicitly, since ctypes otherwise reports and
  # suppresses exceptions raised by a Python callback.
  @ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(mesa.struct_isa_decode_value)
  )
  def field(_, name, value):
    try:
      content = (
        ctypes.string_at(value.contents.str).decode() if value.contents.str else int(value.contents.num)
      )
      fields.append((ctypes.string_at(name).decode(), content))
    except Exception as error:
      errors.append(error)

  @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p)
  def post(_, index, raw):
    try:
      result.append(
        _instruction(fields.copy(), ctypes.cast(raw, ctypes.POINTER(ctypes.c_uint64)).contents.value)
      )
    except Exception as error:
      errors.append(error)
    finally:
      fields.clear()

  @ctypes.CFUNCTYPE(
    None, ctypes.POINTER(mesa.struct__IO_FILE), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint64
  )
  def unmatched(_, words, count):
    # The Qualcomm compiler emits legacy BAR/FENCE without Mesa's fixed bit49.
    # Decode only that explicit zero-reserved-bit layout, retaining the actual
    # word and all synchronization flags. Other unmatched encodings stay errors.
    try:
      flags = {'SS':44, 'W':51, 'R':52, 'L':53, 'G':54, 'JP':59, 'SY':60}
      allowed = (1 << 55) | sum(1 << shift for shift in flags.values())
      raw = int(words[0]) | int(words[1]) << 32 if count == 2 else 0
      if raw & ~allowed != 7 << 61:
        raise RuntimeError('IR3 unrecognized encoding')
      values: list[tuple[str, str|int]] = [('NAME', 'fence' if raw & (1 << 55) else 'bar')]
      values += [(name, raw >> shift & 1) for name, shift in flags.items()]
      result.append(_instruction(values, raw))
    except Exception as error:
      errors.append(error)
    finally:
      fields.clear()

  stream = libc.fopen(b'/dev/null', b'w')
  if not stream:
    raise RuntimeError('IR3 decoder output could not be opened')
  try:
    options = mesa.struct_isa_decode_options(
      gpu_id=630, field_cb=field, post_instr_cb=post, no_match_cb=unmatched
    )
    mesa.ir3_isa_disasm(image, len(image), ctypes.cast(stream, ctypes.POINTER(mesa.struct__IO_FILE)), options)
  finally:
    libc.fclose(stream)
  if errors:
    raise RuntimeError(f'IR3 decode failed: {errors[0]}') from errors[0]
  if len(result) * 8 != len(image):
    raise RuntimeError('IR3 decoder omitted an instruction')
  for pc, ins in enumerate(result):
    if ins.op in ('br', 'brao', 'braa', 'jump', 'call'):
      offset = _signed(ins.fields['IMMED'], 32)
      if not 0 <= pc + offset < len(result):
        raise RuntimeError('IR3 branch target out of bounds')
  return tuple(result)


# Each register-bank column is one invocation lane; rows are scalar register
# components. NumPy evaluates the selected lane columns together.
# PC, predicate mode/mask and barrier waiting have distinct transition rules.
class Workgroup:
  def __init__(self, constants, local_size, group, wgid, lid, wgsize, memory, shared_bytes, private_bytes, *,
               image_bindings=ImageBindings(), opencl=False, wgoffset=0xfc, groups=(1, 1, 1),
               constant_demotion=True, entry_pc=0):
    self.count = math.prod(local_size)
    self.registers = (np.zeros((256, self.count), np.uint32), np.zeros((256, self.count), np.uint16))
    self.constants = np.frombuffer(constants, dtype='<u4').copy()
    self.constant_demotion = constant_demotion
    self.opencl = opencl
    self.memory = memory
    self.image_bindings = image_bindings
    self.pc = np.full(self.count, entry_pc, np.int64)
    self.return_stack:list[list[int]] = [[] for _ in range(self.count)]
    self.predicate = np.ones(self.count, bool)
    self.predicate_mode = np.zeros(self.count, np.int8)
    self.waiting = np.zeros(self.count, bool)

    # Shared bytes belong to this workgroup; each private row belongs to one
    # lane. Only the supplied global-memory transaction spans workgroups.
    self.shared = np.zeros(shared_bytes, np.uint8)
    self.private = np.zeros((self.count, private_bytes), np.uint8)

    # The launch chooses where workgroup/local IDs and local dimensions live.
    # 0xfc means that the compiled shader does not consume that input.
    if wgid != 0xFC:
      self.registers[0][wgid : wgid + 3] = np.array(group, np.uint32)[:, None]
    if lid != 0xFC:
      lanes = np.arange(self.count, dtype=np.uint32)
      self.registers[0][lid : lid + 3] = np.array([
        lanes % local_size[0],
        lanes // local_size[0] % local_size[1],
        lanes // (local_size[0] * local_size[1]),
      ])
    if opencl:
      # QCOMCL builtin reads: local dimensions in r48.xyz, group-origin in
      # r51.xyz, global dimensions in c0.xyz, and group counts in c5.xyz.
      # The register positions are carried by the launch, not inferred from code.
      if wgsize != 0xFC:
        self.registers[0][wgsize : wgsize + 3] = np.array(local_size, np.uint32)[:, None]
      if wgoffset != 0xFC:
        self.registers[0][wgoffset : wgoffset + 3] = (np.array(group, np.uint32) * local_size)[:, None]
      self.constants[:3] = np.array(groups, np.uint32) * local_size
      self.constants[8:11] = 0 # get_global_offset: the admitted producer has zero offsets.
      self.constants[18] = 3 # get_work_dim: NDRANGE admission requires three dimensions.
      self.constants[20:23] = groups
      self.constants[28:31] = 0 # No dispatch splitting adds an extra group origin.
    elif wgsize != 0xFC:
      self.constants[wgsize : wgsize + 3] = local_size

  def _operand_index(self, operand, lanes, advance, limit):
    index = operand.index + advance
    if operand.relative:
      # a0.x is a per-lane signed16 address register, in scalar components.
      index = index + self.registers[True][244, lanes].view(np.int16).astype(np.int32)
      valid = np.all((index >= 0) & (index < limit))
    else:
      # Keep ordinary scalar indexing on its existing inexpensive path.
      valid = 0 <= index < limit
    if not valid:
      bank = 'constant register' if operand.kind == 'constant' else 'register'
      raise RuntimeError(f"IR3 {'relative ' if operand.relative else ''}{bank} index out of bounds")
    return index

  def read(self, operand, lanes, repeat=0):
    # A repeated instruction advances only sources carrying the (r) flag.
    mask = 0xFFFF if operand.half else 0xFFFFFFFF
    if operand.kind in ('immediate', 'float_immediate'):
      index = operand.index + repeat * operand.repeat
      return np.full(len(lanes), index & mask, np.uint32)
    if operand.kind == 'constant':
      constants = self.constants.view('<u2') if operand.half and not self.constant_demotion else self.constants
      index = self._operand_index(operand, lanes, repeat * operand.repeat, len(constants))
      # The instruction has not selected a numeric type yet. In particular,
      # discarding the upper bits here would destroy a floating constant.
      return constants[index].astype(np.uint32) if operand.relative else np.full(len(lanes), constants[index], np.uint32)
    index = self._operand_index(operand, lanes, repeat * operand.repeat, 244 if operand.relative else 256)
    return self.registers[operand.half][index, lanes].astype(np.uint32)

  def write(self, operand, lanes, value, repeat=0):
    # Destinations advance on every repeat, independently of source flags.
    index = self._operand_index(operand, lanes, repeat, 244 if operand.relative else 256)
    self.registers[operand.half][index, lanes] = value

  def refresh_predicate(self, lanes):
    # Mode 0 enables all lanes; +1/-1 select true/false p0.x. Keep the resulting
    # mask separately so ordinary register writes do not silently refresh it.
    mode = self.predicate_mode[lanes]
    self.predicate[lanes] = (mode == 0) | ((self.registers[0][248, lanes] != 0) == (mode == 1))

  def alu(self, ins, lanes, repeat):
    op, fields, operands = ins.op, ins.fields, ins.operands
    dst = operands['DST']
    if op == 'mov':
      # MOV/COV has explicit source and destination types and a rounding mode.
      src = operands['SRC']
      value = self.read(src, lanes, repeat)
      if fields['SRC_TYPE'] != fields['DST_TYPE'] or src.kind == 'constant' and fields['SRC_TYPE'] == 0:
        value = _bits(
          _source_typed(value, src, fields['SRC_TYPE'], constant_demotion=self.constant_demotion),
          fields['DST_TYPE'], round_zero=fields.get('ROUND', 0) == 0
        )
      elif src.half:
        value &= 0xFFFF
      self.write(dst, lanes, value, repeat)
      return
    base = op.split('.')[0]
    floating = '.f' in op or ins.raw >> 61 == 4
    signed = '.s' in op or op == 'ashr.b'
    values = []
    names = [name for name in operands if name.startswith('SRC')]

    # Select current values first, then apply the opcode's interpretation and
    # source modifiers. A NEG bit means bitwise inversion for bitwise opcodes.
    for name in names:
      src = operands[name]
      value = self.read(src, lanes, repeat)
      if floating:
        # cat2/cat4's FLUT immediate is a numeric table index, not IEEE bits.
        if src.kind == 'float_immediate':
          immediate = np.float16(FLUT[src.index]) if src.half else np.float32(FLUT[src.index])
          value = np.full(len(lanes), immediate, np.float32)
        else:
          value = _source_typed(value, src, 0 if src.half else 1, constant_demotion=self.constant_demotion).astype(np.float32)
      elif signed:
        value = _typed(value, 4 if src.half else 5).astype(np.int32)
      else:
        # Integer constant demotion is truncation before comparisons/shifts,
        # not merely a mask applied when the destination is written.
        value = _typed(value, 2 if src.half else 3).astype(np.uint32)
      if src.modifier & 2:
        value = np.abs(value)
      if src.modifier & 1:
        value = ~value if op in BITWISE_OPS else -value
        if src.half and op in BITWISE_OPS:
          # BNOT complements the selected source width before a shift can
          # consume its high bits. Preserve signed interpretation for ASHR.
          value = value.astype(np.int16 if signed else np.uint16).astype(value.dtype)
      values.append(value)
    source_half = operands[names[0]].half

    # Each case also states its operand count. Calculations operate on the
    # selected lane vectors; precision conversion happens at writeback below.
    match base, values:
      case 'add', [a, b]:
        if fields.get('EI'):
          value = ((a.astype(np.uint64) + b.astype(np.uint64)) >> 1).astype(np.uint32)
        else:
          value = a + b
      case 'sad', [a, b, c]:
        # QCOMCL emits SAD for three-source signed sums, including 3*index
        # and pointer high-word carry. Only its second source admits negation.
        value = a + b + c
      case 'sub', [a, b]:
        value = a - b
      case 'mul', [a, b]:
        if op == 'mul.s24':
          value = _signed(a, 24) * _signed(b, 24)
        elif op == 'mul.u24':
          value = (a & 0xffffff) * (b & 0xffffff)
        else:
          value = a * b
      case 'mull', [a, b]:
        # MULL uses the low 16-bit parts; MADSH adds a shifted high-half term.
        value = (a & 65535) * (b & 65535)
      case 'madsh', [a, b, c]:
        value = ((a * (b >> 16)) << 16) + c
      case 'mad', [a, b, c]:
        # Integer MAD narrows its multiplicands and keeps the full accumulator.
        # Floating MAD is unfused: round the product before the addition.
        if op == 'mad.u16':
          product = (a & 65535) * (b & 65535)
        elif op == 'mad.s24':
          product = _signed(a, 24) * _signed(b, 24)
        else:
          product = a * b
        if op == 'mad.f16':
          product = product.astype(np.float16).astype(np.float32)
        value = product + c
      case 'shl', [a, b]:
        value = a << (b & 31)
      case 'shr' | 'ashr', [a, b]:
        value = a >> (b & 31)
      case 'shrg', [a, b, c]:
        value = (b >> (a & 31)) | c
      case 'shrm', [a, b, c]:
        value = (b >> (a & 31)) & c
      case 'shlm', [a, b, c]:
        value = (b << (a & 31)) & c
      case 'shlg', [a, b, c]:
        value = (b << (a & 31)) | c
      case 'andg', [a, b, c]:
        value = (b & a) | c
      case 'and', [a, b]:
        value = a & b
      case 'or', [a, b]:
        value = a | b
      case 'xor', [a, b]:
        value = a ^ b
      case 'not', [a]:
        value = ~a
      case 'absneg', [a]:
        value = a
      case 'sign', [a]:
        value = np.where(np.isnan(a), np.float32(0), np.where(a == 0, a, np.copysign(np.float32(1), a)))
      case 'getbit', [a, b]:
        value = (a >> b) & 1
      case 'clz', [a]:
        value = np.where(a == 0, -1, (16 if source_half else 32) - np.frexp(a.astype(np.float64))[1])
      case 'min', [a, b]:
        value = np.fmin(a, b) if floating else np.minimum(a, b)
      case 'max', [a, b]:
        value = np.fmax(a, b) if floating else np.maximum(a, b)
      case 'sel', [a, b, c]:
        # SEL.S32 maps OpenCL vector-select's signed/MSB condition; SEL.B32
        # retains the scalar nonzero convention. Data bits are preserved.
        value = np.where(b >= 0 if op == 'sel.s32' else b != 0, a, c)
      case 'cmps' | 'cmpv', [a, b]:
        value = (a < b, a <= b, a > b, a >= b, a == b, a != b)[fields['COND']].astype(np.uint32)
        if fields.get('SAT', 0):
          value = np.uint32(1) - value
        if base == 'cmpv':
          # Vector compares provide bit masks for the compiler's following AND;
          # scalar compares retain their existing 0/1 predicate convention.
          value = np.uint32(0) - value
      case 'floor', [a]:
        value = np.floor(a)
      case 'ceil', [a]:
        value = np.ceil(a)
      case 'trunc', [a]:
        value = np.trunc(a)
      case 'rcp', [a]:
        value = 1 / a
      case 'sqrt', [a]:
        value = np.sqrt(a)
      case 'rsq', [a]:
        value = 1 / np.sqrt(a)
      case 'exp2', [a]:
        value = np.exp2(a)
      case 'log2', [a]:
        value = np.log2(a)
      case 'sin', [a]:
        value = np.sin(a)
      case 'cos', [a]:
        value = np.cos(a)
      case _:
        raise RuntimeError(f'IR3 unsupported arithmetic or operand count for {op}')

    # Most operations complete at source width before destination conversion.
    # The 24-bit multiplies and MAD.U16 retain their wider integer result even
    # with half sources; only their destination may truncate it.
    # Comparisons already produce integer 0/1 values.
    if floating and base not in ('cmps', 'cmpv'):
      if source_half:
        value = value.astype(np.float16).astype(np.float32)
      if fields.get('SAT', 0):
        value = np.clip(value, 0, 1)
      value = _bits(value, 0 if dst.half else 1, round_zero=source_half != dst.half)
    elif source_half and base not in ('cmps', 'cmpv') and op not in ('mul.s24', 'mul.u24', 'mad.u16'):
      value = value.astype(np.uint16)
      if op in SIGNED_OUTPUTS:
        value = value.view(np.int16).astype(np.int32)
    self.write(dst, lanes, value, repeat)

  def transfer(self, addresses, width, values=None, storage=None, lanes=None):
    # One word-access path serves all three memory spaces. Explicit storage
    # denotes this workgroup's shared/private bytes; None selects mapped global
    # regions through the caller. A supplied value vector selects stores.
    # Private offsets are checked per lane before selecting a flattened view.
    if storage is not None:
      limit = storage.shape[-1]
      if limit < width or np.any(addresses > limit - width):
        raise RuntimeError('IR3 local/private memory out of bounds')
      if storage.ndim == 2:
        addresses = addresses + lanes.astype(np.uint64) * limit
      storage = storage.reshape(-1)
    output = np.empty(len(addresses), np.uint32)
    remaining = np.ones(len(addresses), bool)
    while remaining.any():
      # Process one backing allocation at a time; lanes may address different
      # allocations even though they execute the same memory instruction.
      first = int(np.flatnonzero(remaining)[0])
      base, data = self.memory.region(int(addresses[first]), width) if storage is None else (0, storage)
      selected = remaining & (addresses >= base) & (addresses <= base + len(data) - width)
      offsets = (addresses[selected] - base).astype(np.int64)
      # This owned buffer view exposes a little-endian word at every byte offset,
      # including unaligned addresses, and never extends past its allocation.
      words: np.ndarray = np.ndarray((len(data) - width + 1,), dtype=f'<u{width}', buffer=data, strides=(1,))
      if values is not None and storage is None:
        # Resolving readable bytes does not imply permission to modify them.
        start, end = int(addresses[selected].min()), int(addresses[selected].max()) + width
        self.memory.validate(start, end - start, write=True)
      if values is None:
        output[selected] = words[offsets]
      else:
        words[offsets] = values[selected] & 0xFFFFFFFF
      remaining[selected] = False
    return output

  def memory_instruction(self, ins, lanes):
    op, fields, operands = ins.op, ins.fields, ins.operands
    space = op.split('.')[0][-1]
    loading = op.startswith('ld')
    address_name, data_name = memory_roles(op)
    typ = fields['TYPE']
    width = np.dtype(TYPES[typ]).itemsize
    pointer = operands[address_name]
    addresses = self.read(pointer, lanes).astype(np.uint64)

    # Global pointers use consecutive low/high full registers. Shared/private
    # addresses are byte offsets into the selected workgroup or lane storage.
    if space == 'g':
      addresses |= self.registers[0][pointer.index + 1, lanes].astype(np.uint64) << 32
    # Mesa's numeric callback preserves the encoded 13-bit offset bits.
    if op.endswith('.a'):
      # A6xx ldg.a/stg.a zero-extend before scaling: no intermediate u32 wrap.
      # Mesa ir3-cat6.xml specifies ((index << shift) + offset) * element_size.
      index = self.read(operands['SRC2'], lanes).astype(np.uint64)
      addresses += ((index << fields['SRC2_SHIFT']) + fields['OFF']) * width
    else:
      offset = _signed(fields.get('OFF', 0), 13)
      addresses = addresses + offset if offset >= 0 else addresses - (-offset)
    # The CL compiler's signed-byte form uses TYPE=7 and a half-register result;
    # generated consumers immediately read that half bank while full pointers
    # stay live. Mesa's GL convention instead names this type U8_32.
    signed_byte = self.opencl and typ == 7
    register = replace(operands[data_name], half=typ in (0, 2, 4, 6) or signed_byte)
    storage = None if space == 'g' else self.private if space == 'p' else self.shared

    # Preserve component order for vector loads/stores. The base addresses are
    # selected once, so a load cannot change its own remaining address operands.
    for component in range(fields['SIZE']):
      address = addresses + component * width
      value = None if loading else self.read(replace(register, index=register.index + component), lanes)
      result = self.transfer(address, width, value, storage, lanes)
      if loading:
        if signed_byte:
          result = result.astype(np.uint8).view(np.int8).astype(np.int16).view(np.uint16)
        self.write(register, lanes, result, component)

  def image_instruction(self, ins, lanes):
    bindings = self.image_bindings
    if ins.op == 'isam':
      if 'INDICES' in ins.operands:
        register = ins.operands['INDICES'].index
        textures = self.registers[1][register, lanes]
        samplers = self.registers[1][register + 1, lanes]
        if np.any(textures != textures[0]) or np.any(samplers != samplers[0]):
          raise RuntimeError('IR3 uniform image indices differ between lanes')
        texture_index, sampler_index = int(textures[0]), int(samplers[0])
      else:
        texture_index, sampler_index = ins.fields['TEX'], ins.fields['SAMP']
      if not 0 <= texture_index < len(bindings.textures) or not 0 <= sampler_index < bindings.sampler_count:
        raise RuntimeError('IR3 image texture or sampler index out of bounds')
      image = bindings.textures[texture_index]
    else:
      index = ins.fields['SSBO']
      if not 0 <= index < len(bindings.outputs):
        raise RuntimeError('IR3 image output index out of bounds')
      image = bindings.outputs[index]

    # Capture full-width signed coordinates before destination writes. A texture
    # coordinate outside either dimension selects black, never a wrapped row.
    coordinate = ins.operands['COORD'].index
    x = self.registers[0][coordinate, lanes].view(np.int32).astype(np.int64)
    y = self.registers[0][coordinate + 1, lanes].view(np.int32).astype(np.int64)
    valid = (x >= 0) & (x < image.width) & (y >= 0) & (y < image.height)
    # Formatted GL image loads return zero for invalid texels; stores have no
    # effect there. Descriptor and registered-allocation checks still precede
    # these coordinate rules, so an absent binding is never treated as padding.
    addresses = (image.base + y[valid] * image.pitch + x[valid] * 4 * image.component_bytes).astype(np.uint64)
    storage_type = 0 if image.component_bytes == 2 else 1

    for component in range(4):
      if ins.op in ('isam', 'ldib.b'):
        if ins.op == 'isam' and not ins.fields['WRMASK'] & (1 << component):
          continue
        values = np.zeros(len(lanes), np.float32)
        if valid.any():
          bits = self.transfer(addresses + component * image.component_bytes, image.component_bytes)
          values[valid] = _typed(bits, storage_type)
        self.write(ins.operands['DST'], lanes, _bits(values, ins.fields['TYPE']), component)
      else:
        data = replace(ins.operands['DATA'], index=ins.operands['DATA'].index + component)
        values = _typed(self.read(data, lanes[valid]), ins.fields['TYPE'])
        bits = _bits(values, storage_type)
        self.transfer(addresses + component * image.component_bytes, image.component_bytes, bits)

  def run(self, instructions):
    # This is the functional scheduling loop. Ended lanes have PC -1; waiting
    # lanes keep their PC but cannot run. Dependency hints model no extra latency.
    # Accumulated normalization gradients can exceed one million scheduler
    # steps. Keep a finite default and permit explicit limits for larger tests.
    max_steps = int(os.getenv('MOCK_QCOM_MAX_STEPS', '20000000'))
    if max_steps <= 0:
      raise ValueError('MOCK_QCOM_MAX_STEPS must be positive')
    for steps in range(max_steps):
      alive = self.pc >= 0
      if not alive.any():
        return
      runnable = alive & ~self.waiting
      if not runnable.any():
        # A shader barrier releases only when every lane is still alive and has
        # reached the same barrier. PM4 queue waits are handled outside this loop.
        if not alive.all() or len(np.unique(self.pc)) != 1:
          raise RuntimeError('IR3 divergent workgroup barrier')
        self.waiting[:] = False
        self.pc += 1
        continue

      # Fetch one instruction for the lanes currently at its PC. Choosing the
      # lowest runnable PC groups work without erasing per-lane branch history.
      pc = int(self.pc[runnable].min())
      if pc >= len(instructions):
        raise RuntimeError('IR3 execution escaped the program')
      ins = instructions[pc]
      lanes = np.flatnonzero((self.pc == pc) & runnable)

      # Fall-through is the default. A branch replaces selected PCs; a barrier
      # restores this PC while parked. Control handling precedes ordinary effects.
      self.pc[lanes] += 1
      op, fields, operands = ins.op, ins.fields, ins.operands
      if fields.get('JP', 0):
        # JP refreshes the saved mask while preserving true/false/none mode.
        self.refresh_predicate(lanes)
      if op == 'end':
        self.pc[lanes] = -1
      elif op in ('nop', 'fence'):
        # Memory operations already finish synchronously in staged storage.
        # A fence preserves compiler order; only BAR rendezvous parks lanes.
        pass
      elif op in ('predt', 'predf', 'prede'):
        self.predicate_mode[lanes] = {'predt': 1, 'predf': -1, 'prede': 0}[op]
        self.refresh_predicate(lanes)
      elif op == 'bar':
        self.waiting[lanes] = True
        self.pc[lanes] = pc
      else:
        # Predicated-off lanes still advance through the instruction stream but
        # do not perform these register or memory updates.
        lanes = lanes[self.predicate[lanes]]
        if not len(lanes):
          continue
        if op in ('br', 'brao', 'braa', 'jump'):
          taken = np.ones(len(lanes), bool)
          if op != 'jump':
            taken = (self.registers[0][248 + fields['COMP1'], lanes] != 0) ^ bool(fields['INV1'])
          if op in ('brao', 'braa'):
            other = (self.registers[0][248 + fields['COMP2'], lanes] != 0) ^ bool(fields['INV2'])
            taken = taken | other if op == 'brao' else taken & other
          offset = _signed(fields['IMMED'], 32)
          self.pc[lanes[taken]] = pc + offset
        elif op == 'call':
          # CALL offsets use the same full-program instruction coordinates as
          # branches. Each lane retains its own return path across divergence.
          for lane in lanes:
            if len(self.return_stack[lane]) >= 64:
              raise RuntimeError('IR3 call stack budget exhausted')
            self.return_stack[lane].append(pc + 1)
          self.pc[lanes] = pc + _signed(fields['IMMED'], 32)
        elif op == 'ret':
          for lane in lanes:
            if not self.return_stack[lane]:
              raise RuntimeError('IR3 return without a call')
            self.pc[lane] = self.return_stack[lane].pop()
        elif op in ('ldg', 'stg', 'ldg.a', 'stg.a', 'ldl', 'stl', 'ldp', 'stp'):
          self.memory_instruction(ins, lanes)
        elif op in ('isam', 'stib.b', 'ldib.b'):
          self.image_instruction(ins, lanes)
        elif op == 'swz':
          # Read both sides before writing either, including an in-place swap.
          values = [self.read(operands[name], lanes) for name in ('SRC0', 'SRC1')]
          for name, value in zip(('DST0', 'DST1'), values):
            self.write(operands[name], lanes, value)
        else:
          for repeat in range(fields.get('REPEAT', 0) + 1):
            self.alu(ins, lanes, repeat)
    if (self.pc < 0).all():
      return
    raise RuntimeError(f'IR3 instruction budget exhausted ({max_steps} steps)')


def validate_invocation_registers(wgid, lid):
  # Invocation coordinates occupy three scalar registers below the reserved
  # register bank; 0xfc means unused. Planning and execution share this contract
  # so a known invalid launch cannot follow an already published wait prefix.
  if any(reg != 0xFC and not 0 <= reg <= 241 for reg in (wgid, lid)):
    raise RuntimeError('IR3 invalid invocation register')


def validate_image_bindings(instructions, bindings):
  # Immediate resource references can be admitted before any command action.
  # S2EN's live register values remain an execution check, but it still requires
  # declared tables; it never creates a descriptor from an unregistered address.
  for ins in instructions:
    if ins.op == 'isam':
      if not bindings.textures or not bindings.sampler_count:
        raise RuntimeError('IR3 image sampling requires textures and samplers')
      if 'INDICES' not in ins.operands and (
        not 0 <= ins.fields['TEX'] < len(bindings.textures) or not 0 <= ins.fields['SAMP'] < bindings.sampler_count
      ):
        raise RuntimeError('IR3 image texture or sampler index out of bounds')
    elif ins.op in ('stib.b', 'ldib.b') and not 0 <= ins.fields['SSBO'] < len(bindings.outputs):
      raise RuntimeError('IR3 image output index out of bounds')


def validate_constant_footprint(instructions, slots, *, constant_demotion=True, entry_pc=0):
  # Admission uses immutable decoded indices, never live constant contents.
  # Track PC and whether predication is active: a predicated JUMP can fall
  # through. Mode zero always has a true saved mask, including after JP.
  if type(entry_pc) is not int or not 0 <= entry_pc < len(instructions):
    raise RuntimeError('IR3 entry point is outside the program')
  pending, visited = [(entry_pc, False, False)], set()
  while pending:
    pc, predicated, in_call = pending.pop()
    if (pc, predicated, in_call) in visited or pc == len(instructions):
      continue
    visited.add((pc, predicated, in_call))
    ins = instructions[pc]
    for operand in ins.operands.values():
      if operand.kind == 'constant':
        capacity = slots * 2 if operand.half and not constant_demotion else slots
        if operand.relative:
          # Its address depends on live lane values, like a global-memory load.
          # Direct indices remain checked here; relative lookups check the actual
          # upload bounds before reading or changing their destination.
          if capacity == 0:
            raise RuntimeError('IR3 relative constant source has no supplied slots')
          continue
        last = operand.index + (ins.fields.get('REPEAT', 0) if operand.repeat else 0)
        if not 0 <= operand.index <= last < capacity:
          raise RuntimeError('IR3 constant source span exceeds supplied slots')

    # All words still undergo decode/schema validation. Only structurally
    # unreachable words are excluded here: END has no successor, but an earlier
    # branch may reach code beyond it. Dynamic conditions take both paths.
    if ins.op == 'end':
      continue
    if ins.op == 'ret':
      if not in_call:
        raise RuntimeError('IR3 return is reachable without a call')
      if not predicated:
        continue
    if ins.op in ('predt', 'predf', 'prede'):
      predicated = ins.op != 'prede'
    if ins.op in ('br', 'brao', 'braa', 'jump', 'call'):
      pending.append((pc + _signed(ins.fields['IMMED'], 32), predicated, in_call or ins.op == 'call'))
      if ins.op == 'jump' and not predicated:
        continue
      if ins.op == 'call':
        # A callee can change predicate mode before RET. Both caller
        # continuations remain possible until actual lane data is executed.
        pending.append((pc + 1, not predicated, in_call))
    # A call explores both its callee and the continuation it may return to.
    # This bounded over-approximation checks constants without simulating data.
    pending.append((pc + 1, predicated, in_call))


# Launch entry: qcomgpu supplies the state; decoding may be reused, but every
# dispatched workgroup gets fresh lane, shared and private state.
def execute(
  image, constants, groups, local_size, wgid, lid, memory, *, shared_bytes=0, private_bytes=0, wgsize=0xFC,
  image_bindings=ImageBindings(),
  opencl=False, wgoffset=0xfc, constant_demotion=True, entry_pc=0,
):
  instructions = decode(image)
  if (
    len(groups) != 3
    or len(local_size) != 3
    or any(type(n) is not int or n <= 0 for n in (*groups, *local_size))
  ):
    raise RuntimeError('IR3 invalid launch dimensions')
  if math.prod(local_size) > 1024 or len(constants) % 4 or shared_bytes < 0 or private_bytes < 0:
    raise RuntimeError('IR3 invalid launch resources')
  validate_invocation_registers(wgid, lid)
  if opencl:
    validate_invocation_registers(wgsize, wgoffset)
    if len(constants) < 32 * 4:
      raise RuntimeError('IR3 OpenCL builtin constants are incomplete')
  elif wgsize != 0xFC and not 0 <= wgsize <= len(constants) // 4 - 3:
    raise RuntimeError('IR3 invalid workgroup-size constant')
  validate_constant_footprint(instructions, len(constants) // 4, constant_demotion=constant_demotion, entry_pc=entry_pc)
  validate_image_bindings(instructions, image_bindings)
  image_bindings.validate(memory)
  with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
    for group in itertools.product(*(range(n) for n in groups)):
      Workgroup(constants, local_size, group, wgid, lid, wgsize, memory, shared_bytes, private_bytes,
                image_bindings=image_bindings, opencl=opencl, wgoffset=wgoffset, groups=groups,
                constant_demotion=constant_demotion, entry_pc=entry_pc).run(
        instructions
      )
