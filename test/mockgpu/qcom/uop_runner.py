"""Optional native-CPU execution for decoded, straight-line IR3 ALU blocks.

This is deliberately a translator for :class:`IR3Instruction`, not a fallback
to the source Tensor/UOp graph.  Unsupported machine instructions return
``None`` from :meth:`IR3UOpRunner.try_run`, leaving the IR3 interpreter as the
single correctness authority.  Each lowered opcode mirrors the exact branch
the interpreter takes for it, including which branches ignore encoded source
modifiers (``_IGNORE_MODS``) and which apply them.
"""
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


def _mask(value: UOp, bits: int) -> UOp:
  return value.cast(dtypes.uint32) & UOp.const((1 << bits) - 1, dtypes.uint32)


def _sign_extend(value: UOp, bits: int) -> UOp:
  sign = UOp.const(1 << (bits - 1), dtypes.uint32)
  return (_mask(value, bits) ^ sign) - sign


def _output_mask(value: UOp, dst: Register) -> UOp:
  return _mask(value, 16 if dst[0].startswith('h') else 32)


def _const(value: int | float, dtype) -> UOp: return UOp.const(value, dtype)

def _comparison(lhs: UOp, rhs: UOp, condition: int) -> UOp:
  comparisons = (lhs < rhs, lhs <= rhs, lhs > rhs, lhs >= rhs, lhs == rhs, lhs != rhs)
  if not 0 <= condition < len(comparisons) or not isinstance(comparison := comparisons[condition], UOp):
    raise UnsupportedIR3Block('invalid or constant-folded IR3 comparison')
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

  def run(self, regs: dict[Register, list[int]], mask: list[bool]) -> None:
    words = self.regfile.as_memoryview(force_zero_copy=True).cast('I')
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
          if _register(advanced): keys.add(advanced)
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
    value = self.unpack_float(self.read(self.sources(inst, component)[index]), half)
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
    signed = _sign_extend(value, 16 if inst.source_half else 32).cast(dtypes.int32)
    absolute = (signed < _const(0, dtypes.int32)).where(signed * _const(-1, dtypes.int32), signed)
    out = signed * _const(-1, dtypes.int32) if modifier == 1 else absolute if modifier == 2 else absolute * _const(-1, dtypes.int32)
    return out.cast(dtypes.uint32)

  @staticmethod
  def int_operand(value: UOp, signed: bool, bits: int) -> UOp:
    return _sign_extend(value, bits).cast(dtypes.int32) if signed else _mask(value, bits)

  def convert(self, value: UOp, src_type: int, dst_type: int) -> UOp:
    if src_type == dst_type: return _mask(value, 16 if src_type in _HALF_TYPES else 32)
    if src_type in _FLOAT_TYPES:
      number = self.unpack_float(value, src_type == mesa.TYPE_F16)
      if dst_type in _FLOAT_TYPES: return self.pack_float(number, dst_type == mesa.TYPE_F16)
      # NOTE: the interpreter truncates unrepresentable magnitudes with Python arbitrary precision; native C
      # casts saturate. Real kernels convert in-range indices, so this lowering accepts that divergence.
      finite = (number - number) == _const(0.0, dtypes.float64)
      return _mask(finite.where(number.cast(dtypes.int32), _const(0, dtypes.int32)).cast(dtypes.uint32),
                   8 if dst_type in _BYTE_TYPES else 16 if dst_type in _HALF_TYPES else 32)
    src_bits = 8 if src_type in _BYTE_TYPES else 16 if src_type in _HALF_TYPES else 32
    if src_type == mesa.TYPE_U8 and dst_type in _SIGNED_TYPES: value = _sign_extend(value, 8)
    elif src_type in _SIGNED_TYPES: value = _sign_extend(value, src_bits)
    else: value = _mask(value, src_bits)
    if dst_type in _FLOAT_TYPES:
      return self.pack_float(value.cast(dtypes.int32 if src_type in _SIGNED_TYPES else dtypes.uint32).cast(dtypes.float64),
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
        product = self.int_operand(a, True, bits).cast(dtypes.int32) * self.int_operand(b, True, bits).cast(dtypes.int32)
        out = product + c.cast(dtypes.int32)
      else: out = self.int_operand(a, False, bits) * self.int_operand(b, False, bits) + c
      self.write(dst, out)
    elif name == 'madsh.m16':
      if len(inst.srcs) != 3 or inst.dst[0].startswith('h'): raise UnsupportedIR3Block('unsupported madsh.m16 form')
      a, b, c = (self.read(src) for src in self.sources(inst, component))
      # Encoded IR3 source order is low(src0) * high(src1), inserted at bit 16.
      self.write(dst, ((_mask(a, 16) * _mask(b >> 16, 16)) << 16) + c)
    elif name.startswith('sad.'):
      a, b, c = (self.read(src) for src in self.sources(inst, component))
      difference = a.cast(dtypes.int32) - b.cast(dtypes.int32)
      absolute = (difference < _const(0, dtypes.int32)).where(difference * _const(-1, dtypes.int32), difference)
      self.write(dst, c.cast(dtypes.int32) + absolute)
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
      value = _sign_extend(self.read(self.sources(inst, component)[0]), 16 if inst.source_half else 32).cast(dtypes.int32)
      absolute = (value < _const(0, dtypes.int32)).where(value * _const(-1, dtypes.int32), value)
      self.write(dst, value * _const(-1, dtypes.int32) if modifier == 1 else absolute if modifier == 2 else
                 absolute * _const(-1, dtypes.int32) if modifier == 3 else value)
      return
    if len(inst.srcs) != 1: raise UnsupportedIR3Block(f'invalid {name} source count')
    value = self.float_source(inst, 0, component, inst.source_half)
    zero, one = _const(0.0, dtypes.float64), _const(1.0, dtypes.float64)
    if name == 'absneg.f': out = value
    elif name == 'sign.f': out = (value < zero).where(_const(-1.0, dtypes.float64), (value > zero).where(one, value))
    elif name == 'floor.f': out = (value - value == zero).where(value.floor(), value)
    elif name == 'ceil.f': out = (value - value == zero).where(value.ceil(), value)
    elif name == 'rndaz.f':
      # round-toward-zero of the magnitude with the sign restored: copysign(ceil(|x|), x).
      magnitude = (value.bitcast(dtypes.uint64) & _const((1 << 63) - 1, dtypes.uint64)).bitcast(dtypes.float64)
      out = (value - value == zero).where(
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
    elif name == 'cmpv.u': self.write(dst, _comparison(_mask(a, 32), _mask(b, 32), inst.condition)); return
    elif name == 'cmpv.s':
      self.write(dst, _comparison(self.int_operand(a, True, source_bits), self.int_operand(b, True, source_bits), inst.condition)); return
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
      out = self.int_operand(a, True, bits).cast(dtypes.int32) * self.int_operand(b, True, bits).cast(dtypes.int32)
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
                       tuple((reg, self.slot[reg]) for reg in sorted(self.dirty)), runtime, regfile)


class IR3UOpRunner:
  """Compile supported decoded, straight-line IR3 ALU runs to CPU UOps."""
  def __init__(self, min_instructions: int = 8):
    self.min_instructions = min_instructions
    self.cache: dict[tuple[int, int, int, int], IR3UOpBlock] = {}
    self.uncompilable: set[tuple[int, int, int, int]] = set()
    self.program_blocks: dict[int, tuple[tuple[IR3Instruction, ...], dict[int, int]]] = {}
    # Structurally identical blocks (same register layout, lanes, and instruction forms) share one
    # compiled program, so equivalent sequences in different shaders never recompile.
    self.compiled: dict[tuple, IR3UOpBlock] = {}

  @staticmethod
  def _supported(inst: IR3Instruction) -> bool:
    if inst.name == 'nop': return True
    if inst.name not in _NATIVE or not _register(inst.dst) or inst.sat: return False
    return not any(isinstance(src, tuple) and src[0] in {'rel', 'relr', 'relhr'} for src in inst.srcs)

  def _blocks(self, program: tuple[IR3Instruction, ...]) -> dict[int, int]:
    # decode_ir3 caches and reuses each exact program tuple.  Key by identity
    # here so the hot scheduler never hashes thousands of decoded instructions.
    program_id = id(program)
    if (cached := self.program_blocks.get(program_id)) is not None and cached[0] is program: return cached[1]
    blocks: dict[int, int] = {}
    pc = 0
    while pc < len(program):
      if not self._supported(program[pc]):
        pc += 1
        continue
      start = pc
      while pc < len(program) and self._supported(program[pc]): pc += 1
      end, remaining = pc, sum(inst.name != 'nop' for inst in program[start:pc])
      for candidate in range(start, end):
        if remaining >= self.min_instructions: blocks[candidate] = end
        remaining -= program[candidate].name != 'nop'
    self.program_blocks[program_id] = (program, blocks)
    return blocks

  def try_run(self, program: tuple[IR3Instruction, ...], start_pc: int, regs: dict[Register, list[int]],
              exec_mask: list[bool], predication: list[bool] | None = None,
              mask_pcs: frozenset[int] | None = None) -> int | None:
    """Return the next PC when accelerated, or ``None`` for exact interpreter fallback."""
    if not regs: return None
    if (end_pc := self._blocks(program).get(start_pc)) is None: return None
    # A pending branch reconvergence inside the range would change the write mask mid-block;
    # leave those ranges to the interpreter, which re-enters native execution after the merge.
    if mask_pcs is not None and any(start_pc < target < end_pc for target in mask_pcs): return None
    lanes = len(next(iter(regs.values())))
    if len(exec_mask) != lanes: return None
    mask = exec_mask if predication is None else [active and pred for active, pred in zip(exec_mask, predication, strict=True)]
    key = (id(program), start_pc, end_pc, lanes)
    try:
      if (block := self.cache.get(key)) is None and key not in self.uncompilable:
        lowerer = _Lowerer(program[start_pc:end_pc], lanes)
        signature = (lanes, lowerer.keys, tuple((inst.name, inst.dst, inst.srcs, inst.repeat, inst.repeat_srcs,
                                                inst.src_mods, inst.condition, inst.types, inst.sat, inst.rounding,
                                                inst.source_half) for inst in lowerer.instructions))
        block = self.compiled.get(signature)
        if block is None: block = self.compiled[signature] = lowerer.compile()
        self.cache[key] = block
      if block is None: return None
      block.run(regs, mask)
      return start_pc + block.length
    except UnsupportedIR3Block:
      self.uncompilable.add(key)
      return None
