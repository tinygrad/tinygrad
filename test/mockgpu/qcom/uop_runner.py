"""Lower decoded straight-line IR3 ALU blocks to CPU UOps."""
import array
from dataclasses import dataclass
from typing import Any, TypeGuard

from tinygrad.codegen import to_program
from tinygrad.device import Buffer, Device
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import get_runtime
from tinygrad.runtime.autogen import mesa
from tinygrad.uop.ops import KernelInfo, UOp

from test.mockgpu.qcom.decoder import IR3Instruction
from test.mockgpu.qcom.registers import _reg_offset


Register = tuple[str, int, int]
_UINT32_MASK = 0xffffffff
_HALF_TYPES = {mesa.TYPE_F16, mesa.TYPE_U16, mesa.TYPE_S16, mesa.TYPE_U8}
_FLOAT_TYPES = {mesa.TYPE_F16, mesa.TYPE_F32}
_SIGNED_TYPES = {mesa.TYPE_S16, mesa.TYPE_S32}
_BYTE_TYPES = {mesa.TYPE_U8, mesa.TYPE_U8_32}
# The interpreter takes a dedicated branch for these opcodes that reads raw
# operands, so encoded modifiers must be ignored rather than rejected.
_IGNORE_MODS = {'add.u', 'cmps.u', 'cmps.s', 'shrg', 'shrm', 'shlm', 'shlg', 'andg',
                'sel.b16', 'sel.b32', 'sel.s16', 'sel.s32', 'sel.f16', 'sel.f32',
                'sad.s16', 'sad.s32', 'mad.u16', 'madsh.u16', 'mad.s16', 'mad.u24', 'mad.s24', 'madsh.m16'}
_MOD_OPS = {'add.s', 'sub.u', 'sub.s', 'min.u', 'min.s', 'max.u', 'max.s', 'cmpv.u', 'cmpv.s', 'shl.b', 'shr.b', 'ashr.b',
            'and.b', 'or.b', 'xor.b', 'not.b', 'mul.u24', 'mul.s24', 'mull.u', 'getbit.b', 'absneg.s',
            'add.f', 'mul.f', 'min.f', 'max.f', 'cmps.f', 'cmpv.f', 'sign.f', 'absneg.f', 'floor.f', 'ceil.f', 'rndaz.f',
            'mad.f16', 'mad.f32', 'rcp', 'rsq', 'log2', 'exp2', 'sqrt', 'hrsq', 'hlog2', 'hexp2'}
_NATIVE = _IGNORE_MODS | _MOD_OPS | {'mov'}
_MAD_INT = {'mad.u16', 'madsh.u16', 'mad.s16', 'mad.u24', 'mad.s24'}


class UnsupportedIR3Block(Exception):
  """The decoded block needs the ordinary machine-code interpreter."""


def _register(value: Any) -> TypeGuard[Register]:
  return isinstance(value, tuple) and len(value) == 3 and value[0] not in {'rel', 'relr', 'relhr'}


def _advance(src: Register | int, component: int, repeated: bool) -> Register | int:
  if not repeated: return src
  return src + component if isinstance(src, int) else _reg_offset(src, component)

def _full_constant(src: Register | int) -> Register | None:
  return ('c', src[1], src[2]) if _register(src) and src[0] == 'hc' else None


def _mask(value: UOp, bits: int) -> UOp:
  return value.cast(dtypes.uint32) & UOp.const((1 << bits) - 1, dtypes.uint32)


def _sign_extend(value: UOp, bits: int) -> UOp:
  sign = UOp.const(1 << (bits - 1), dtypes.uint32)
  return (_mask(value, bits) ^ sign) - sign

def _as_int32(value: UOp) -> UOp: return _mask(value, 32).bitcast(dtypes.int32)


def _output_mask(value: UOp, dst: Register) -> UOp:
  return _mask(value, 16 if dst[0].startswith('h') else 32)


def _const(value: int | float, dtype) -> UOp: return UOp.const(value, dtype)

def _comparison(lhs: UOp, rhs: UOp, condition: int) -> UOp:
  if condition == 0: comparison = lhs < rhs
  elif condition == 1: comparison = lhs <= rhs
  elif condition == 2: comparison = lhs > rhs
  elif condition == 3: comparison = lhs >= rhs
  elif condition == 4: comparison = lhs.ne(rhs).logical_not()
  elif condition == 5: comparison = lhs.ne(rhs)
  else: raise UnsupportedIR3Block('invalid IR3 comparison condition')
  return comparison.cast(dtypes.uint32)


@dataclass
class IR3UOpBlock:
  """A cached native CPU program for one decoded IR3 basic block."""
  length: int
  lanes: int
  slots: tuple[Register, ...]
  dirty_slots: tuple[tuple[Register, int], ...]
  runtime: Any
  regfile: Buffer
  words: memoryview

  def run(self, regs: dict[Register, list[int]], mask: list[bool]) -> None:
    words = self.words
    for slot, reg in enumerate(self.slots):
      values = regs.get(reg)
      if values is None: values = [0] * self.lanes
      try: words[slot * self.lanes:(slot + 1) * self.lanes] = array.array('I', values)
      except OverflowError: words[slot * self.lanes:(slot + 1) * self.lanes] = array.array('I', (value & _UINT32_MASK for value in values))
    words[len(self.slots) * self.lanes:(len(self.slots) + 1) * self.lanes] = array.array('I', mask)
    self.runtime(self.regfile._buf, global_size=(1, 1, 1), local_size=(1, 1, 1), wait=True)
    for reg, slot in self.dirty_slots:
      regs[reg] = list(words[slot * self.lanes:(slot + 1) * self.lanes])


class _Lowerer:
  def __init__(self, instructions: tuple[IR3Instruction, ...], lanes: int):
    self.instructions, self.lanes = instructions, lanes
    self.keys = self._keys()
    self.slot = {reg: index for index, reg in enumerate(self.keys)}
    # One extra regfile slot holds the wave's write mask; masked stores select between old and new lanes.
    self.mask_slot = len(self.keys)
    self.regfile = UOp.placeholder(((self.mask_slot + 1) * lanes,), dtypes.uint32, slot=0, device='CPU')
    self.index = UOp.range(lanes, 0)
    self.mask = self.regfile.index(self.index + self.mask_slot * lanes).load().ne(_const(0, dtypes.uint32))
    self.values: dict[Register, UOp] = {}
    self.dirty: set[Register] = set()

  def _keys(self) -> tuple[Register, ...]:
    keys: set[Register] = set()
    for inst in self.instructions:
      if not _register(inst.dst): continue
      repeat_srcs = inst.repeat_srcs + (False,) * (len(inst.srcs) - len(inst.repeat_srcs))
      for component in range(inst.repeat + 1):
        keys.add(_reg_offset(inst.dst, component))
        for src, repeat in zip(inst.srcs, repeat_srcs, strict=True):
          advanced = _advance(src, component, repeat)
          if _register(advanced):
            keys.add(advanced)
            if (full := _full_constant(advanced)) is not None: keys.add(full)
    return tuple(sorted(keys))

  @staticmethod
  def sources(inst: IR3Instruction, component: int) -> tuple[Register | int, ...]:
    repeat_srcs = inst.repeat_srcs + (False,) * (len(inst.srcs) - len(inst.repeat_srcs))
    return tuple(_advance(src, component, repeat) for src, repeat in zip(inst.srcs, repeat_srcs, strict=True))

  def read(self, src: Register | int) -> UOp:
    if isinstance(src, int): return _const(src & _UINT32_MASK, dtypes.uint32)
    if src not in self.slot: raise UnsupportedIR3Block(f'unsupported IR3 source {src}')
    if src not in self.values:
      self.values[src] = self.regfile.index(self.index + self.slot[src] * self.lanes).load()
    return self.values[src]

  def write(self, dst: Register, value: UOp):
    if dst not in self.slot: raise UnsupportedIR3Block(f'unsupported IR3 destination {dst}')
    # IR3 predication writes through only where the wave mask is set, like the interpreter's _write.
    self.values[dst] = self.mask.where(_output_mask(value, dst), self.read(dst))
    self.dirty.add(dst)

  # ---- typed operand views -------------------------------------------------

  # The interpreter evaluates float ALU in Python doubles and repacks, so every
 # float lowering here computes in float64.  For f32/f16-sourced operands the
 # products and sums are exact in f64 (24+24 <= 53 bits), which also makes
 # FMA contraction harmless, and the final cast performs the same single
 # round-to-nearest-even that struct.pack applies.
  @staticmethod
  def unpack_float(value: UOp, half: bool) -> UOp:
    if half: return _mask(value, 16).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float64)
    return _mask(value, 32).bitcast(dtypes.float32).cast(dtypes.float64)

  @staticmethod
  def pack_float(value: UOp, half: bool) -> UOp:
    if half: return value.cast(dtypes.float16).bitcast(dtypes.uint16).cast(dtypes.uint32)
    return value.cast(dtypes.float32).bitcast(dtypes.uint32)

  def float_source(self, inst: IR3Instruction, index: int, component: int, half: bool) -> UOp:
    src = self.sources(inst, component)[index]
    raw = self.read(src)
    # Half float ALU constants may name a full-float constant slot. A6xx converts the full
    # value to f16 when its upper half is populated; integer/bit consumers still use raw hc.
    if half and (full_reg := _full_constant(src)) is not None:
      full = self.read(full_reg)
      converted = self.pack_float(self.unpack_float(full, False), True)
      raw = (full >> _const(16, dtypes.uint32)).ne(_const(0, dtypes.uint32)).where(converted, _mask(raw, 16))
    value = self.unpack_float(raw, half)
    modifier = inst.src_mods[index] if index < len(inst.src_mods) else 0
    if modifier:
      # neg/abs/-abs on a float equal sign-bit ops on the packed bits, including -0.0 and NaN payloads.
      bits = value.bitcast(dtypes.uint64)
      bits = bits ^ _const(1 << 63, dtypes.uint64) if modifier == 1 else \
        bits & _const((1 << 63) - 1, dtypes.uint64) if modifier == 2 else bits | _const(1 << 63, dtypes.uint64)
      value = bits.bitcast(dtypes.float64)
    return value

  def int_source(self, inst: IR3Instruction, index: int, component: int) -> UOp:
    value = self.read(self.sources(inst, component)[index])
    modifier = inst.src_mods[index] if index < len(inst.src_mods) else 0
    if not modifier: return value
    signed = _as_int32(_sign_extend(value, 16 if inst.source_half else 32))
    absolute = (signed < _const(0, dtypes.int32)).where(signed * _const(-1, dtypes.int32), signed)
    out = signed * _const(-1, dtypes.int32) if modifier == 1 else absolute if modifier == 2 else absolute * _const(-1, dtypes.int32)
    return out.cast(dtypes.uint32)

  @staticmethod
  def int_operand(value: UOp, signed: bool, bits: int) -> UOp:
    return _as_int32(_sign_extend(value, bits)) if signed else _mask(value, bits)

  def convert(self, value: UOp, src_type: int, dst_type: int) -> UOp:
    if src_type == dst_type: return _mask(value, 16 if src_type in _HALF_TYPES else 32)
    if src_type in _FLOAT_TYPES:
      number = self.unpack_float(value, src_type == mesa.TYPE_F16)
      if dst_type in _FLOAT_TYPES: return self.pack_float(number, dst_type == mesa.TYPE_F16)
      exponent = _const(0x7c00 if src_type == mesa.TYPE_F16 else 0x7f800000, dtypes.uint32)
      finite = (value.cast(dtypes.uint32) & exponent).ne(exponent)
      dst_bits = 8 if dst_type in _BYTE_TYPES else 16 if dst_type in _HALF_TYPES else 32
      # IR3 conversion truncates and then keeps the destination-width low bits. Avoid native float-to-int
      # saturation at signed/unsigned limits by reducing the integral value modulo 2**width before the cast.
      integral = (number < _const(0.0, dtypes.float64)).where(number.ceil(), number.floor())
      modulus = _const(float(1 << dst_bits), dtypes.float64)
      wrapped = integral - (integral / modulus).floor() * modulus
      return _mask(finite.where(wrapped.cast(dtypes.uint32), _const(0, dtypes.uint32)), dst_bits)
    src_bits = 8 if src_type in _BYTE_TYPES else 16 if src_type in _HALF_TYPES else 32
    if src_type == mesa.TYPE_U8 and dst_type in _SIGNED_TYPES: value = _sign_extend(value, 8)
    elif src_type in _SIGNED_TYPES: value = _sign_extend(value, src_bits)
    else: value = _mask(value, src_bits)
    if dst_type in _FLOAT_TYPES:
      number = _as_int32(value) if src_type in _SIGNED_TYPES else value.cast(dtypes.uint32)
      return self.pack_float(number.cast(dtypes.float64),
                             dst_type == mesa.TYPE_F16)
    dst_bits = 8 if dst_type in _BYTE_TYPES else 16 if dst_type in _HALF_TYPES else 32
    return _mask(value, dst_bits)

  # ---- opcode lowering -----------------------------------------------------

  def lower(self):
    for inst in self.instructions:
      if inst.name == 'nop': continue
      if inst.name not in _NATIVE or not _register(inst.dst) or inst.sat:
        raise UnsupportedIR3Block(f'unsupported IR3 opcode {inst.name}')
      for component in range(inst.repeat + 1):
        self.lower_one(inst, component)

  def lower_one(self, inst: IR3Instruction, component: int):
    name, dst = inst.name, _reg_offset(inst.dst, component)
    if name == 'mov':
      if len(inst.srcs) != 1: raise UnsupportedIR3Block('invalid mov source count')
      src, = self.sources(inst, component)
      self.write(dst, self.convert(self.read(src), *inst.types))
    elif name in {'mad.f16', 'mad.f32'}:
      half = name == 'mad.f16'
      a, b, c = (self.float_source(inst, index, component, half) for index in range(3))
      self.write(dst, self.pack_float(a * b + c, half))
    elif name in _MAD_INT:
      a, b, c = (self.read(src) for src in self.sources(inst, component))
      signed, bits = name in {'mad.s16', 'mad.s24'}, 16 if name.endswith('16') else 24
      if signed:
        product = self.int_operand(a, True, bits) * self.int_operand(b, True, bits)
        out = product + _as_int32(c)
      else: out = self.int_operand(a, False, bits) * self.int_operand(b, False, bits) + c
      self.write(dst, out)
    elif name == 'madsh.m16':
      if len(inst.srcs) != 3 or inst.dst[0].startswith('h'): raise UnsupportedIR3Block('unsupported madsh.m16 form')
      a, b, c = (self.read(src) for src in self.sources(inst, component))
      # Encoded IR3 source order is low(src0) * high(src1), inserted at bit 16.
      self.write(dst, ((_mask(a, 16) * _mask(b >> 16, 16)) << 16) + c)
    elif name.startswith('sad.'):
      a, b, c = (self.read(src) for src in self.sources(inst, component))
      difference = _as_int32(a) - _as_int32(b)
      absolute = (difference < _const(0, dtypes.int32)).where(difference * _const(-1, dtypes.int32), difference)
      self.write(dst, _as_int32(c) + absolute)
    elif name.startswith('sel.'):
      a, b, c = (self.read(src) for src in self.sources(inst, component))
      self.write(dst, b.ne(_const(0, dtypes.uint32)).where(a, c))
    elif name in {'shrm', 'shlm', 'shlg', 'andg'}:
      a, b, c = (self.read(src) for src in self.sources(inst, component))
      shift = a & _const(31, dtypes.uint32)
      out = _mask(b, 32) >> shift & c if name == 'shrm' else b << shift & c if name == 'shlm' else \
        (b << shift) | c if name == 'shlg' else (b & a) | c
      self.write(dst, out)
    elif name == 'shrg':
      if len(inst.srcs) != 3: raise UnsupportedIR3Block('invalid shrg source count')
      shift, value, other = (self.read(src) for src in self.sources(inst, component))
      self.write(dst, (_mask(value, 16 if inst.source_half else 32) >> (shift & _const(31, dtypes.uint32))) | other)
    elif name == 'not.b':
      if len(inst.srcs) != 1: raise UnsupportedIR3Block('invalid not.b source count')
      self.write(dst, self.int_source(inst, 0, component) ^ _const(_UINT32_MASK, dtypes.uint32))
    elif name in {'absneg.s', 'absneg.f', 'sign.f', 'floor.f', 'ceil.f', 'rndaz.f', 'rcp', 'rsq', 'log2', 'exp2', 'sqrt',
                  'hrsq', 'hlog2', 'hexp2'}:
      self.lower_unary(inst, component, dst)
    elif name in {'add.f', 'mul.f', 'min.f', 'max.f', 'cmps.f', 'cmpv.f'}:
      self.lower_binary_float(inst, component, dst)
    else:
      self.lower_binary_int(inst, component, dst)

  def lower_unary(self, inst: IR3Instruction, component: int, dst: Register):
    name = inst.name
    if name == 'absneg.s':
      if len(inst.srcs) != 1: raise UnsupportedIR3Block('invalid absneg.s source count')
      modifier = inst.src_mods[0] if inst.src_mods else 0
      value = _as_int32(_sign_extend(self.read(self.sources(inst, component)[0]), 16 if inst.source_half else 32))
      absolute = (value < _const(0, dtypes.int32)).where(value * _const(-1, dtypes.int32), value)
      self.write(dst, value * _const(-1, dtypes.int32) if modifier == 1 else absolute if modifier == 2 else
                 absolute * _const(-1, dtypes.int32) if modifier == 3 else value)
      return
    if len(inst.srcs) != 1: raise UnsupportedIR3Block(f'invalid {name} source count')
    value = self.float_source(inst, 0, component, inst.source_half)
    zero, one = _const(0.0, dtypes.float64), _const(1.0, dtypes.float64)
    exponent = _const(0x7ff0000000000000, dtypes.uint64)
    bits = value.bitcast(dtypes.uint64)
    finite = (bits & exponent).ne(exponent)
    is_zero = (bits & _const((1 << 63) - 1, dtypes.uint64)).ne(_const(0, dtypes.uint64)).logical_not()
    if name == 'absneg.f': out = value
    elif name == 'sign.f': out = (value < zero).where(_const(-1.0, dtypes.float64), (value > zero).where(one, value))
    elif name == 'floor.f': out = finite.where(is_zero.where(zero, value.floor()), value)
    elif name == 'ceil.f': out = finite.where(is_zero.where(zero, value.ceil()), value)
    elif name == 'rndaz.f':
      # round-toward-zero of the magnitude with the sign restored: copysign(ceil(|x|), x).
      magnitude = (value.bitcast(dtypes.uint64) & _const((1 << 63) - 1, dtypes.uint64)).bitcast(dtypes.float64)
      out = finite.where(
        (magnitude.ceil().bitcast(dtypes.uint64) | (value.bitcast(dtypes.uint64) & _const(1 << 63, dtypes.uint64)))
        .bitcast(dtypes.float64), value)
    elif name == 'rcp': out = one / value
    elif name in {'rsq', 'hrsq'}: out = one / value.sqrt()
    elif name in {'log2', 'hlog2'}: out = value.log2()
    elif name in {'exp2', 'hexp2'}: out = _const(2.0, dtypes.float64) ** value  # Python computes 2.0 ** x via libm pow
    else: out = value.sqrt()
    self.write(dst, self.pack_float(out, dst[0].startswith('h')))

  def lower_binary_float(self, inst: IR3Instruction, component: int, dst: Register):
    if len(inst.srcs) != 2: raise UnsupportedIR3Block(f'invalid {inst.name} source count')
    lhs, rhs = (self.float_source(inst, index, component, inst.source_half) for index in range(2))
    if inst.name in {'cmps.f', 'cmpv.f'}:
      self.write(dst, _comparison(lhs, rhs, inst.condition))
      return
    if inst.name == 'add.f': out = lhs + rhs
    elif inst.name == 'mul.f': out = lhs * rhs
    else:
      # The interpreter propagates the non-NaN operand when either side is NaN, then takes min/max.
      picked = (lhs <= rhs).where(lhs, rhs) if inst.name == 'min.f' else (lhs >= rhs).where(lhs, rhs)
      out = lhs.ne(lhs).where(rhs, rhs.ne(rhs).where(lhs, picked))
    self.write(dst, self.pack_float(out, dst[0].startswith('h')))

  def lower_binary_int(self, inst: IR3Instruction, component: int, dst: Register):
    name = inst.name
    if len(inst.srcs) != 2: raise UnsupportedIR3Block(f'invalid {name} source count')
    source_bits = 16 if inst.source_half else 32
    if name in {'cmps.u', 'cmps.s'}:  # dedicated interpreter branch reads raw operands, ignoring modifiers
      raw_a, raw_b = (self.read(src) for src in self.sources(inst, component))
      pair = (_mask(raw_a, source_bits), _mask(raw_b, source_bits)) if name == 'cmps.u' else \
        (self.int_operand(raw_a, True, source_bits), self.int_operand(raw_b, True, source_bits))
      self.write(dst, _comparison(*pair, inst.condition))
      return
    a, b = (self.int_source(inst, index, component) for index in range(2))
    if name in ('add.u', 'add.s', 'sub.u', 'sub.s'): out = a + b if name.startswith('add') else a - b
    elif name == 'cmpv.u':
      self.write(dst, _comparison(_mask(a, 32), _mask(b, 32), inst.condition))
      return
    elif name == 'cmpv.s':
      self.write(dst, _comparison(self.int_operand(a, True, source_bits), self.int_operand(b, True, source_bits), inst.condition))
      return
    elif name == 'min.u': out = (_mask(a, 32) <= _mask(b, 32)).where(a, b)
    elif name == 'max.u': out = (_mask(a, 32) >= _mask(b, 32)).where(a, b)
    elif name == 'min.s': out = self.int_minmax(a, b, source_bits, True)
    elif name == 'max.s': out = self.int_minmax(a, b, source_bits, False)
    elif name == 'and.b': out = a & b
    elif name == 'or.b': out = a | b
    elif name == 'xor.b': out = a ^ b
    elif name == 'mul.u24':
      bits = 16 if inst.source_half else 24
      out = _mask(a, bits) * _mask(b, bits)
    elif name == 'mul.s24':
      bits = 16 if inst.source_half else 24
      out = self.int_operand(a, True, bits) * self.int_operand(b, True, bits)
    elif name == 'mull.u': out = _mask(a, 16) * _mask(b, 16)
    elif name == 'shl.b': out = a << (b & _const(31, dtypes.uint32))
    elif name == 'shr.b': out = _mask(a, 32) >> (b & _const(31, dtypes.uint32))
    elif name == 'ashr.b': out = self.int_operand(a, True, source_bits) >> (b & _const(31, dtypes.uint32))
    elif name == 'getbit.b': out = (a >> (b & _const(31, dtypes.uint32))) & _const(1, dtypes.uint32)
    else: raise UnsupportedIR3Block(f'unsupported IR3 opcode {name}')
    self.write(dst, out)

  def int_minmax(self, a: UOp, b: UOp, bits: int, is_min: bool) -> UOp:
    sa, sb = self.int_operand(a, True, bits), self.int_operand(b, True, bits)
    return (sa <= sb).where(a, b) if is_min else (sa >= sb).where(a, b)

  def compile(self) -> IR3UOpBlock:
    self.lower()
    stores = tuple(self.regfile.index(self.index + self.slot[reg] * self.lanes).store(self.values[reg]) for reg in sorted(self.dirty))
    sink = UOp.group(*stores).end(self.index).sink(arg=KernelInfo('ir3_cpu_block', opts_to_apply=()))
    program = to_program(sink, Device['CPU'].renderer)
    runtime = get_runtime('CPU', program)
    regfile = Buffer('CPU', (self.mask_slot + 1) * self.lanes, dtypes.uint32).allocate()
    return IR3UOpBlock(len(self.instructions), self.lanes, self.keys,
                       tuple((reg, self.slot[reg]) for reg in sorted(self.dirty)), runtime, regfile,
                       regfile.as_memoryview(force_zero_copy=True).cast('I'))
