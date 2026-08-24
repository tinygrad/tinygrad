"""Optional native-CPU execution for decoded, straight-line IR3 ALU blocks.

This is deliberately a translator for :class:`IR3Instruction`, not a fallback
to the source Tensor/UOp graph.  Unsupported machine instructions return
``None`` from :meth:`IR3UOpRunner.try_run`, leaving the IR3 interpreter as the
single correctness authority.
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
_SUPPORTED = {'mov', 'add.u', 'sub.u', 'cmps.u', 'cmps.s', 'shl.b', 'shr.b', 'ashr.b', 'mull.u', 'madsh.m16',
              'and.b', 'or.b', 'xor.b', 'not.b', 'shrg', 'sel.b32', 'andg', 'absneg.s',
              'add.f', 'mul.f', 'cmps.f', 'rcp'}


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


@dataclass
class IR3UOpBlock:
  """A cached native CPU program for one decoded IR3 basic block."""
  end_pc: int
  lanes: int
  slots: tuple[Register, ...]
  dirty_slots: tuple[tuple[Register, int], ...]
  runtime: Any
  regfile: Buffer

  def run(self, regs: dict[Register, list[int]]) -> int:
    words = self.regfile.as_memoryview(force_zero_copy=True).cast('I')
    for slot, reg in enumerate(self.slots):
      words[slot * self.lanes:(slot + 1) * self.lanes] = array.array('I', (value & _UINT32_MASK for value in regs.get(reg, [0] * self.lanes)))
    self.runtime(self.regfile._buf, global_size=(1, 1, 1), local_size=(1, 1, 1), wait=True)
    for reg, slot in self.dirty_slots:
      regs[reg] = list(words[slot * self.lanes:(slot + 1) * self.lanes])
    return self.end_pc


class _Lowerer:
  def __init__(self, instructions: tuple[IR3Instruction, ...], lanes: int):
    self.instructions, self.lanes = instructions, lanes
    self.keys = self._keys()
    self.slot = {reg: index for index, reg in enumerate(self.keys)}
    self.regfile = UOp.placeholder((len(self.keys) * lanes,), dtypes.uint32, slot=0, device='CPU')
    self.index = UOp.range(lanes, 0)
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
    if isinstance(src, int): return UOp.const(src & _UINT32_MASK, dtypes.uint32)
    if src not in self.slot: raise UnsupportedIR3Block(f'unsupported IR3 source {src}')
    if src not in self.values:
      self.values[src] = self.regfile.index(self.index + self.slot[src] * self.lanes).load()
    return self.values[src]

  def write(self, dst: Register, value: UOp):
    if dst not in self.slot: raise UnsupportedIR3Block(f'unsupported IR3 destination {dst}')
    self.values[dst] = _output_mask(value, dst)
    self.dirty.add(dst)

  def convert(self, value: UOp, src_type: int, dst_type: int) -> UOp:
    if src_type in _FLOAT_TYPES or dst_type in _FLOAT_TYPES: raise UnsupportedIR3Block('floating cat1 conversion')
    src_bits = 8 if src_type in _BYTE_TYPES else 16 if src_type in _HALF_TYPES else 32
    if src_type == mesa.TYPE_U8 and dst_type in _SIGNED_TYPES: value = _sign_extend(value, 8)
    elif src_type in _SIGNED_TYPES: value = _sign_extend(value, src_bits)
    else: value = _mask(value, src_bits)
    dst_bits = 8 if dst_type in _BYTE_TYPES else 16 if dst_type in _HALF_TYPES else 32
    return _mask(value, dst_bits)

  def binary_sources(self, inst: IR3Instruction, component: int) -> tuple[UOp, UOp]:
    if len(inst.srcs) != 2: raise UnsupportedIR3Block(f'unsupported IR3 source count for {inst.name}')
    return tuple(self.read(src) for src in self.sources(inst, component)) # type: ignore[return-value]

  def lower(self):
    for inst in self.instructions:
      if inst.name == 'nop': continue
      if inst.name not in _SUPPORTED or not _register(inst.dst) or (any(inst.src_mods) and inst.name != 'absneg.s') or inst.sat:
        raise UnsupportedIR3Block(f'unsupported IR3 opcode {inst.name}')
      if inst.name == 'mov':
        if len(inst.srcs) != 1: raise UnsupportedIR3Block('invalid mov source count')
        for component in range(inst.repeat + 1):
          src, = self.sources(inst, component)
          self.write(_reg_offset(inst.dst, component), self.convert(self.read(src), *inst.types))
        continue
      if inst.name == 'madsh.m16':
        if len(inst.srcs) != 3 or inst.dst[0].startswith('h'):
          raise UnsupportedIR3Block('unsupported madsh.m16 form')
        for component in range(inst.repeat + 1):
          a, b, c = (self.read(src) for src in self.sources(inst, component))
          # Encoded IR3 source order is low(src0) * high(src1), inserted at bit 16.
          self.write(_reg_offset(inst.dst, component), ((_mask(a, 16) * _mask(b >> 16, 16)) << 16) + c)
        continue
      if inst.name == 'shrg':
        if len(inst.srcs) != 3: raise UnsupportedIR3Block('invalid shrg source count')
        for component in range(inst.repeat + 1):
          shift, value, other = (self.read(src) for src in self.sources(inst, component))
          self.write(_reg_offset(inst.dst, component), (_mask(value, 16 if inst.source_half else 32) >> (shift & 31)) | other)
        continue
      if inst.name == 'not.b':
        if len(inst.srcs) != 1: raise UnsupportedIR3Block('invalid not.b source count')
        for component in range(inst.repeat + 1):
          src, = self.sources(inst, component)
          self.write(_reg_offset(inst.dst, component), self.read(src) ^ UOp.const(_UINT32_MASK, dtypes.uint32))
        continue
      if inst.name == 'absneg.s':
        if len(inst.srcs) != 1: raise UnsupportedIR3Block('invalid absneg.s source count')
        modifier = inst.src_mods[0] if inst.src_mods else 0
        for component in range(inst.repeat + 1):
          src, = self.sources(inst, component)
          value = _sign_extend(self.read(src), 16 if inst.source_half else 32).cast(dtypes.int32)
          absolute = (value < 0).where(-value, value)
          self.write(_reg_offset(inst.dst, component), -absolute if modifier == 3 else absolute if modifier == 2 else
                     -value if modifier == 1 else value)
        continue
      if inst.name in {'sel.b32', 'andg'}:
        if len(inst.srcs) != 3: raise UnsupportedIR3Block(f'invalid {inst.name} source count')
        for component in range(inst.repeat + 1):
          a, b, c = (self.read(src) for src in self.sources(inst, component))
          out = b.ne(0).where(a, c) if inst.name == 'sel.b32' else (b & a) | c
          self.write(_reg_offset(inst.dst, component), out)
        continue
      if inst.name == 'rcp':
        if len(inst.srcs) != 1 or inst.source_half or inst.dst[0].startswith('h'):
          raise UnsupportedIR3Block('unsupported rcp form')
        for component in range(inst.repeat + 1):
          src, = self.sources(inst, component)
          self.write(_reg_offset(inst.dst, component),
                     (UOp.const(1.0, dtypes.float32) / self.read(src).bitcast(dtypes.float32)).bitcast(dtypes.uint32))
        continue
      for component in range(inst.repeat + 1):
        a, b = self.binary_sources(inst, component)
        source_bits = 16 if inst.source_half else 32
        if inst.name == 'add.u': out = a + b
        elif inst.name == 'sub.u': out = a - b
        elif inst.name == 'and.b': out = a & b
        elif inst.name == 'or.b': out = a | b
        elif inst.name == 'xor.b': out = a ^ b
        elif inst.name == 'mull.u': out = _mask(a, 16) * _mask(b, 16)
        elif inst.name == 'shl.b': out = a << (b & 31)
        elif inst.name == 'shr.b': out = _mask(a, source_bits) >> (b & 31)
        elif inst.name == 'ashr.b': out = _sign_extend(a, source_bits).cast(dtypes.int32) >> (b & 31)
        elif inst.name in {'add.f', 'mul.f', 'cmps.f'}:
          if inst.source_half or inst.dst[0].startswith('h') and inst.name != 'cmps.f':
            raise UnsupportedIR3Block(f'unsupported half {inst.name}')
          lhs, rhs = a.bitcast(dtypes.float32), b.bitcast(dtypes.float32)
          if inst.name == 'add.f': out = (lhs + rhs).bitcast(dtypes.uint32)
          elif inst.name == 'mul.f': out = (lhs * rhs).bitcast(dtypes.uint32)
          else:
            comparisons = (lhs < rhs, lhs <= rhs, lhs > rhs, lhs >= rhs, lhs == rhs, lhs != rhs)
            if not 0 <= inst.condition < len(comparisons) or not isinstance(comparison := comparisons[inst.condition], UOp):
              raise UnsupportedIR3Block('invalid IR3 floating comparison condition')
            out = comparison.cast(dtypes.uint32)
        elif inst.name in {'cmps.u', 'cmps.s'}:
          lhs, rhs = (_mask(a, source_bits), _mask(b, source_bits)) if inst.name == 'cmps.u' else \
                     (_sign_extend(a, source_bits).cast(dtypes.int32), _sign_extend(b, source_bits).cast(dtypes.int32))
          comparisons = (lhs < rhs, lhs <= rhs, lhs > rhs, lhs >= rhs, lhs == rhs, lhs != rhs)
          if not 0 <= inst.condition < len(comparisons) or not isinstance(comparison := comparisons[inst.condition], UOp):
            raise UnsupportedIR3Block('invalid IR3 comparison condition')
          out = comparison.cast(dtypes.uint32)
        else: raise UnsupportedIR3Block(f'unsupported IR3 opcode {inst.name}')
        self.write(_reg_offset(inst.dst, component), out)

  def compile(self, end_pc: int) -> IR3UOpBlock:
    self.lower()
    stores = tuple(self.regfile.index(self.index + self.slot[reg] * self.lanes).store(self.values[reg]) for reg in sorted(self.dirty))
    sink = UOp.group(*stores).end(self.index).sink(arg=KernelInfo('ir3_cpu_block', opts_to_apply=()))
    program = to_program(sink, Device['CPU'].renderer)
    runtime = get_runtime('CPU', program)
    regfile = Buffer('CPU', len(self.keys) * self.lanes, dtypes.uint32).allocate()
    return IR3UOpBlock(end_pc, self.lanes, self.keys,
                       tuple((reg, self.slot[reg]) for reg in sorted(self.dirty)), runtime, regfile)


class IR3UOpRunner:
  """Compile supported decoded, straight-line IR3 ALU runs to CPU UOps."""
  def __init__(self, min_instructions: int = 8):
    self.min_instructions = min_instructions
    self.cache: dict[tuple[int, int, int, int], IR3UOpBlock] = {}
    self.program_blocks: dict[int, tuple[tuple[IR3Instruction, ...], dict[int, int]]] = {}

  @staticmethod
  def _supported(inst: IR3Instruction) -> bool:
    if inst.name == 'nop': return True
    if inst.name not in _SUPPORTED or not _register(inst.dst) or (any(inst.src_mods) and inst.name != 'absneg.s') or inst.sat: return False
    if any(isinstance(src, tuple) and src[0] in {'rel', 'relr', 'relhr'} for src in inst.srcs): return False
    if inst.name in {'add.f', 'mul.f', 'cmps.f', 'rcp'} and inst.source_half: return False
    return not (inst.name in {'madsh.m16', 'add.f', 'mul.f', 'rcp'} and inst.dst[0].startswith('h'))

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
              exec_mask: list[bool], predication: list[bool] | None = None) -> int | None:
    """Return the next PC when accelerated, or ``None`` for exact interpreter fallback."""
    if predication is not None or not regs: return None
    if (end_pc := self._blocks(program).get(start_pc)) is None: return None
    lanes = len(next(iter(regs.values())))
    if len(exec_mask) != lanes or not all(exec_mask) or any(len(values) != lanes for values in regs.values()): return None
    key = (id(program), start_pc, end_pc, lanes)
    try:
      if (block := self.cache.get(key)) is None:
        block = self.cache[key] = _Lowerer(program[start_pc:end_pc], lanes).compile(end_pc)
      return block.run(regs)
    except UnsupportedIR3Block:
      return None
