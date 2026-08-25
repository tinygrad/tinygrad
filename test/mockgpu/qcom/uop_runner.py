"""Optional native-CPU execution for decoded, straight-line IR3 ALU blocks.

This is deliberately a translator for :class:`IR3Instruction`, not a fallback
to the source Tensor/UOp graph.  Unsupported machine instructions return
``None`` from :meth:`IR3UOpRunner.try_run`, leaving the IR3 interpreter as the
single correctness authority.  Each lowered opcode mirrors the exact branch
the interpreter takes for it, including which branches ignore encoded source
modifiers (``_IGNORE_MODS``) and which apply them.
"""
import array
from collections import OrderedDict
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


class IR3UOpLoopTimeout(RuntimeError):
  """Internal signal: scalar-replay this staged loop through its exact fuel limit."""
  def __init__(self, start_pc: int):
    super().__init__(f'IR3 native loop exhausted fuel at PC {start_pc}')
    self.start_pc = start_pc


@dataclass
class IR3UOpRunnerStats:
  """Small, resettable counters for the optional native block and loop paths."""
  attempts: int = 0
  runs: int = 0
  native_calls: int = 0
  fallbacks: int = 0
  cache_hits: int = 0
  compiled: int = 0
  iterations: int = 0
  load_checks: int = 0
  load_rejections: int = 0
  block_attempts: int = 0
  block_runs: int = 0
  block_compiles: int = 0
  block_declines: int = 0
  cache_evictions: int = 0

  def reset(self) -> None:
    self.attempts = self.runs = self.native_calls = self.fallbacks = self.cache_hits = 0
    self.compiled = self.iterations = self.load_checks = self.load_rejections = 0
    self.block_attempts = self.block_runs = self.block_compiles = self.block_declines = self.cache_evictions = 0


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


class IR3UOpRunner:
  """Compile supported decoded IR3 ALU blocks and conservative single-lane natural loops to CPU UOps."""
  def __init__(self, min_instructions: int = 8, max_instructions: int = 64, max_register_slots: int = 128,
               max_average_register_slots: int = 40, max_compiled_blocks: int = 512, max_block_locations: int = 8192,
               max_programs: int = 128, max_compiled_loops: int = 32, max_loop_locations: int = 64,
               max_narrow_compiled_blocks: int | None = None, max_regular_narrow_compiled_blocks: int | None = None):
    if not 1 <= min_instructions <= max_instructions: raise ValueError('invalid native IR3 block instruction limits')
    if not 1 <= max_average_register_slots <= max_register_slots: raise ValueError('invalid native IR3 block register limits')
    if min(max_compiled_blocks, max_block_locations, max_programs, max_compiled_loops, max_loop_locations) < 1:
      raise ValueError('invalid native IR3 cache limits')
    if max_narrow_compiled_blocks is None: max_narrow_compiled_blocks = min(384, max_compiled_blocks)
    if not 1 <= max_narrow_compiled_blocks <= max_compiled_blocks: raise ValueError('invalid narrow IR3 compile budget')
    if max_regular_narrow_compiled_blocks is None: max_regular_narrow_compiled_blocks = min(128, max_narrow_compiled_blocks)
    if not 1 <= max_regular_narrow_compiled_blocks <= max_narrow_compiled_blocks:
      raise ValueError('invalid regular narrow IR3 compile budget')
    self.min_instructions, self.max_instructions = min_instructions, max_instructions
    self.max_register_slots, self.max_average_register_slots = max_register_slots, max_average_register_slots
    self.max_compiled_blocks, self.max_block_locations, self.max_programs = max_compiled_blocks, max_block_locations, max_programs
    self.max_compiled_loops, self.max_loop_locations = max_compiled_loops, max_loop_locations
    self.max_narrow_compiled_blocks = max_narrow_compiled_blocks
    self.max_regular_narrow_compiled_blocks = max_regular_narrow_compiled_blocks
    self.cache: OrderedDict[tuple[int, int, int, int], IR3UOpBlock] = OrderedDict()
    self.uncompilable: OrderedDict[tuple[int, int, int, int], None] = OrderedDict()
    self.program_blocks: OrderedDict[int, tuple[tuple[IR3Instruction, ...], dict[int, int]]] = OrderedDict()
    self.program_policy: OrderedDict[tuple[int, bool], tuple[tuple[IR3Instruction, ...], bool]] = OrderedDict()
    # Structurally identical blocks (same register layout, lanes, and instruction forms) share one
    # compiled program, so equivalent sequences in different shaders never recompile.
    self.compiled: OrderedDict[tuple, IR3UOpBlock] = OrderedDict()
    self.compiled_classes: dict[tuple, str] = {}
    self.loop_cache: OrderedDict[tuple[int, int], tuple[tuple[IR3Instruction, ...], Any]] = OrderedDict()
    self.loop_uncompilable: OrderedDict[tuple[int, int], None] = OrderedDict()
    self.compiled_loops: OrderedDict[tuple, Any] = OrderedDict()
    self.stats = IR3UOpRunnerStats()
    self._vmem: Buffer | None = None

  @staticmethod
  def _put_lru(cache: OrderedDict, key, value, limit: int):
    cache[key] = value
    cache.move_to_end(key)
    return cache.popitem(last=False) if len(cache) > limit else None

  @staticmethod
  def _get_lru(cache: OrderedDict, key):
    if key not in cache: return None
    cache.move_to_end(key)
    return cache[key]

  def _evict_compiled_class(self, block_class: str) -> bool:
    for signature in tuple(self.compiled):
      if self.compiled_classes.get(signature) != block_class: continue
      block = self.compiled.pop(signature)
      del self.compiled_classes[signature]
      for location, cached_block in tuple(self.cache.items()):
        if cached_block is block: del self.cache[location]
      self.stats.cache_evictions += 1
      return True
    return False

  @staticmethod
  def _supported(inst: IR3Instruction) -> bool:
    if inst.name == 'nop': return True
    if inst.name not in _NATIVE or not _register(inst.dst) or inst.sat or (inst.name == 'mov' and inst.rounding): return False
    return not any(isinstance(src, tuple) and src[0] in {'rel', 'relr', 'relhr'} for src in inst.srcs)

  @classmethod
  def _loop_shape(cls, program: tuple[IR3Instruction, ...], start_pc: int) -> Any:
    from test.mockgpu.qcom.loop_runner import loop_shape
    return loop_shape(program, start_pc, cls._supported)

  @classmethod
  def has_loop(cls, program: tuple[IR3Instruction, ...]) -> bool:
    from test.mockgpu.qcom.loop_runner import has_loop
    return has_loop(program, cls._supported)

  @staticmethod
  def _select_memory_bounds(regs: dict[Register, list[int]], bounds: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...] | None:
    from test.mockgpu.qcom.loop_runner import _select_memory_bounds
    return _select_memory_bounds(regs, bounds)

  def _vmem_buffer(self) -> Buffer:
    from test.mockgpu.qcom.loop_runner import _vmem_buffer
    return _vmem_buffer(self)

  def _loop_block(self, program: tuple[IR3Instruction, ...], start_pc: int) -> Any:
    from test.mockgpu.qcom.loop_runner import loop_block
    return loop_block(self, program, start_pc)

  def try_run_loop(self, program: tuple[IR3Instruction, ...], start_pc: int, regs: dict[Register, list[int]], exec_mask: list[bool], *,
                   check_range=None, memory_bounds: tuple[tuple[int, int], ...] | None = None,
                   max_steps: int | None = None) -> tuple[int, int] | None:
    """Run one eligible decoded-IR3 natural loop in a single native CPU call."""
    from test.mockgpu.qcom.loop_runner import try_run_loop
    return try_run_loop(self, program, start_pc, regs, exec_mask, check_range=check_range,
                        memory_bounds=memory_bounds, max_steps=max_steps)

  def _blocks(self, program: tuple[IR3Instruction, ...]) -> dict[int, int]:
    # decode_ir3 caches and reuses each exact program tuple.  Key by identity
    # here so the hot scheduler never hashes thousands of decoded instructions.
    program_id = id(program)
    if (cached := self._get_lru(self.program_blocks, program_id)) is not None and cached[0] is program: return cached[1]
    blocks: dict[int, int] = {}
    pc = 0
    while pc < len(program):
      if not self._supported(program[pc]):
        pc += 1
        continue
      start = pc
      while pc < len(program) and self._supported(program[pc]): pc += 1
      end = pc
      for candidate in range(start, end):
        stop, count = candidate, 0
        while stop < end and count < self.max_instructions:
          count += program[stop].name != 'nop'
          stop += 1
        if count >= self.min_instructions: blocks[candidate] = stop
    if (evicted := self._put_lru(self.program_blocks, program_id, (program, blocks), self.max_programs)) is not None:
      self.stats.cache_evictions += 1
      evicted_id = evicted[0]
      for policy_key in tuple(self.program_policy):
        if policy_key[0] == evicted_id: del self.program_policy[policy_key]
      for key in tuple(self.cache):
        if key[0] == evicted_id: del self.cache[key]
      for key in tuple(self.uncompilable):
        if key[0] == evicted_id: del self.uncompilable[key]
    return blocks

  def can_run_blocks(self, program: tuple[IR3Instruction, ...], lanes: int = 1) -> bool:
    """Reject programs whose native chunks spend more time marshalling registers than interpreting IR3."""
    program_id, wide = id(program), lanes >= 16
    policy_key = (program_id, wide)
    if (cached := self._get_lru(self.program_policy, policy_key)) is not None and cached[0] is program: return cached[1]
    blocks = self._blocks(program)
    ranges, covered_until = [], 0
    for start, end in sorted(blocks.items()):
      if start < covered_until: continue
      ranges.append((start, end))
      covered_until = end
    pressures = [len(_Lowerer(program[start:end], 1).keys) for start, end in ranges]
    average_limit = self.max_register_slots if wide else self.max_average_register_slots
    enabled = bool(pressures) and sum(pressures) <= average_limit * len(pressures) and \
      max(pressures) <= self.max_register_slots
    self._put_lru(self.program_policy, policy_key, (program, enabled), self.max_programs * 2)
    return enabled

  def try_run(self, program: tuple[IR3Instruction, ...], start_pc: int, regs: dict[Register, list[int]],
              exec_mask: list[bool], predication: list[bool] | None = None,
              mask_pcs: frozenset[int] | None = None, *, policy_checked: bool = False) -> int | None:
    """Return the next PC when accelerated, or ``None`` for exact interpreter fallback."""
    if not regs: return None
    lanes = len(next(iter(regs.values())))
    if not policy_checked and not self.can_run_blocks(program, lanes): return None
    self.stats.block_attempts += 1
    if (end_pc := self._blocks(program).get(start_pc)) is None: return None
    # A pending branch reconvergence inside the range would change the write mask mid-block;
    # leave those ranges to the interpreter, which re-enters native execution after the merge.
    if mask_pcs is not None and any(start_pc < target < end_pc for target in mask_pcs): return None
    if len(exec_mask) != lanes: return None
    mask = exec_mask if predication is None else [active and pred for active, pred in zip(exec_mask, predication, strict=True)]
    key = (id(program), start_pc, end_pc, lanes)
    try:
      if (block := self._get_lru(self.cache, key)) is None and key not in self.uncompilable:
        lowerer = _Lowerer(program[start_pc:end_pc], lanes)
        if len(lowerer.keys) > self.max_register_slots: raise UnsupportedIR3Block('native IR3 block register pressure is too high')
        signature = (lanes, lowerer.keys, tuple((inst.name, inst.dst, inst.srcs, inst.repeat, inst.repeat_srcs,
                                                inst.src_mods, inst.condition, inst.types, inst.sat, inst.rounding,
                                                inst.source_half) for inst in lowerer.instructions))
        block = self._get_lru(self.compiled, signature)
        if block is None:
          if lanes >= 16:
            block_class = 'wide'
            class_limit = self.max_compiled_blocks - self.max_narrow_compiled_blocks
          else:
            block_class = 'priority_narrow' if len(program) >= 256 else 'regular_narrow'
            class_limit = self.max_regular_narrow_compiled_blocks if block_class == 'regular_narrow' else \
              self.max_narrow_compiled_blocks - self.max_regular_narrow_compiled_blocks
          class_count = sum(value == block_class for value in self.compiled_classes.values())
          if class_count >= class_limit:
            if block_class == 'regular_narrow' or not self._evict_compiled_class(block_class):
              raise UnsupportedIR3Block(f'{block_class} native IR3 block compile budget exhausted')
          if len(self.compiled) >= self.max_compiled_blocks and not self._evict_compiled_class(block_class):
            raise UnsupportedIR3Block('native IR3 block compile budget exhausted')
          block = lowerer.compile()
          self.stats.block_compiles += 1
          self.compiled[signature] = block
          self.compiled_classes[signature] = block_class
        if self._put_lru(self.cache, key, block, self.max_block_locations) is not None:
          self.stats.cache_evictions += 1
      if block is None: return None
      block.run(regs, mask)
      self.stats.block_runs += 1
      return start_pc + block.length
    except UnsupportedIR3Block:
      self.stats.block_declines += 1
      self._put_lru(self.uncompilable, key, None, self.max_block_locations * 2)
      return None
