"""Native execution of conservative, reducible decoded-IR3 loops.

The straight-line ALU lowerer and :class:`IR3UOpRunner` facade deliberately
remain in :mod:`uop_runner`.  This module is imported by that facade only from
its loop methods, after generic lowering support has initialized; the one-way
dependency keeps loop-specific cache/runtime code out of the hot block runner
without a module-import cycle.
"""
import array
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tinygrad.codegen import to_program
from tinygrad.device import Buffer, BufferSpec, Device
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import get_runtime
from tinygrad.uop.ops import KernelInfo, UOp

from test.mockgpu.qcom.decoder import IR3Instruction
from test.mockgpu.qcom.registers import _itemsize, _next_reg, _reg_offset
from test.mockgpu.qcom.uop_runner import (Register, _Lowerer, _NATIVE, _UINT32_MASK, IR3UOpLoopTimeout,
                                           UnsupportedIR3Block, _advance, _const, _full_constant, _output_mask, _register)

if TYPE_CHECKING:
  from test.mockgpu.qcom.uop_runner import IR3UOpRunner


_U64_LIMIT = 1 << 64

# The native loop has one external virtual-memory parameter rooted at address
# zero.  Every access is guarded by executor-supplied mapping bounds before it
# is dereferenced, so the enormous logical extent is never an allocation.
_VMEM_BYTES = 1 << 48
_MAX_LOOP_MEMORY_RANGES = 16

# Fixed, dynamic staging slots after the decoded register file.  Keeping the
# range count/endpoints and fuel out of the UOp graph means one compiled runner
# can serve every work item and mapping snapshot of the same IR3 loop.
_LOOP_FAULT, _LOOP_FAULT_PC, _LOOP_FAULT_LO, _LOOP_FAULT_HI = range(4)
_LOOP_TIMEOUT, _LOOP_TIMEOUT_FUEL, _LOOP_ITERATIONS, _LOOP_LOAD_CHECKS, _LOOP_BOUNDS_COUNT = range(4, 9)
_LOOP_BOUNDS_BASE = 9
_LOOP_CONTROL_WORDS = _LOOP_BOUNDS_BASE + 4 * _MAX_LOOP_MEMORY_RANGES


def _instruction_signature(inst: IR3Instruction) -> tuple:
  return (inst.name, inst.dst, inst.srcs, inst.sy, inst.nop, inst.repeat, inst.repeat_srcs, inst.src_mods, inst.condition,
          inst.types, inst.sat, inst.rounding, inst.branch_offset, inst.invert, inst.source_half, inst.inverts)


@dataclass(frozen=True)
class _LoopShape:
  """One reducible `body; br exit; jump body` decoded-IR3 loop."""
  start_pc: int
  branch_pc: int
  jump_pc: int
  exit_pc: int
  instructions: tuple[IR3Instruction, ...]

  @property
  def branch_index(self) -> int: return self.branch_pc - self.start_pc
  @property
  def jump_index(self) -> int: return self.jump_pc - self.start_pc
  @property
  def exit_steps(self) -> int: return self.branch_index + 1
  @property
  def loop_steps(self) -> int: return self.jump_index + 1


def _loop_registers(shape: _LoopShape) -> tuple[Register, ...]:
  """All decoded registers consumed or written by the native loop, including ldg pointers."""
  keys: set[Register] = set()
  for inst in shape.instructions:
    if inst.name == 'nop': continue
    if inst.name == 'ldg':
      if not _register(inst.dst) or len(inst.srcs) != 3 or not _register(inst.srcs[0]):
        raise UnsupportedIR3Block('invalid ldg loop form')
      address, _, count = inst.srcs
      if not isinstance(count, int) or not 1 <= count <= 4: raise UnsupportedIR3Block('invalid ldg component count')
      keys.add(address)
      keys.add(_next_reg(address))
      for component in range(count): keys.add(_reg_offset(inst.dst, component))
      continue
    if inst.name == 'br':
      if len(inst.srcs) != 1 or not _register(inst.srcs[0]): raise UnsupportedIR3Block('invalid loop branch')
      keys.add(inst.srcs[0])
      continue
    if inst.name == 'jump': continue
    if inst.name not in _NATIVE or not _register(inst.dst) or inst.sat:
      raise UnsupportedIR3Block(f'unsupported IR3 loop opcode {inst.name}')
    repeat_srcs = inst.repeat_srcs + (False,) * (len(inst.srcs) - len(inst.repeat_srcs))
    for component in range(inst.repeat + 1):
      keys.add(_reg_offset(inst.dst, component))
      for src, repeat in zip(inst.srcs, repeat_srcs, strict=True):
        advanced = _advance(src, component, repeat)
        if _register(advanced):
          keys.add(advanced)
          if (full := _full_constant(advanced)) is not None: keys.add(full)
        elif isinstance(advanced, tuple): raise UnsupportedIR3Block(f'unsupported IR3 loop source {advanced}')
  return tuple(sorted(keys))


def _normalise_memory_bounds(bounds: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...] | None:
  """Validate and deduplicate mappings without widening any individual mapping."""
  try:
    raw = tuple(sorted({(int(start), int(end)) for start, end in bounds}))
  except (OverflowError, TypeError, ValueError):
    return None
  for start, end in raw:
    if not 0 <= start < end <= _U64_LIMIT: return None
  return raw if len(raw) <= _MAX_LOOP_MEMORY_RANGES else None


def _u64_parts(value: int) -> tuple[int, int]: return value & _UINT32_MASK, value >> 32


@dataclass
class _NativeLoopOutcome:
  fault: bool
  fault_pc: int
  fault_address: int
  timeout: bool
  timeout_fuel: int
  iterations: int
  load_checks: int


@dataclass
class _NativeLoopBlock:
  """One compiled CPU `for (;;)` runner, with all mutable state in a staging regfile."""
  shape: _LoopShape
  slots: tuple[Register, ...]
  dirty_slots: tuple[tuple[Register, int], ...]
  load_specs: dict[int, tuple[int, int]]
  runtime: Any
  regfile: Buffer
  words: memoryview

  @property
  def control_base(self) -> int: return len(self.slots)
  def control(self, offset: int) -> int: return self.control_base + offset

  def run(self, regs: dict[Register, list[int]], vmem: Buffer, bounds: tuple[tuple[int, int], ...], max_steps: int) -> _NativeLoopOutcome:
    # This is a private staging buffer.  No caller register is changed until
    # the generated loop exits normally and `commit` is called below.
    for slot, reg in enumerate(self.slots): self.words[slot] = regs.get(reg, [0])[0] & _UINT32_MASK
    controls = self.control_base
    self.words[controls:controls + _LOOP_CONTROL_WORDS] = array.array('I', [0]) * _LOOP_CONTROL_WORDS
    self.words[self.control(_LOOP_TIMEOUT_FUEL)] = max_steps
    self.words[self.control(_LOOP_BOUNDS_COUNT)] = len(bounds)
    for index, (start, end) in enumerate(bounds):
      base = self.control(_LOOP_BOUNDS_BASE + index * 4)
      self.words[base:base + 4] = array.array('I', (*_u64_parts(start), *_u64_parts(end)))
    buffers = (self.regfile._buf, vmem._buf) if self.load_specs else (self.regfile._buf,)
    self.runtime(*buffers, global_size=(1, 1, 1), local_size=(1, 1, 1), wait=True)
    return _NativeLoopOutcome(bool(self.words[self.control(_LOOP_FAULT)]), self.words[self.control(_LOOP_FAULT_PC)],
                              self.words[self.control(_LOOP_FAULT_LO)] | self.words[self.control(_LOOP_FAULT_HI)] << 32,
                              bool(self.words[self.control(_LOOP_TIMEOUT)]), self.words[self.control(_LOOP_TIMEOUT_FUEL)],
                              self.words[self.control(_LOOP_ITERATIONS)], self.words[self.control(_LOOP_LOAD_CHECKS)])

  def commit(self, regs: dict[Register, list[int]]) -> None:
    for reg, slot in self.dirty_slots:
      value = self.words[slot]
      if (old := regs.get(reg)) is None or old[0] != value: regs[reg] = [value]


class _LoopLowerer(_Lowerer):
  """Lower exactly one decoded natural loop into one CPU UOp loop and no source graph operations."""
  def __init__(self, shape: _LoopShape, slots: tuple[Register, ...]):
    self.shape, self.instructions, self.lanes = shape, shape.instructions, 1
    self.keys, self.slot = slots, {reg: index for index, reg in enumerate(slots)}
    self.control_base = len(slots)
    self.regfile = UOp.placeholder((self.control_base + _LOOP_CONTROL_WORDS,), dtypes.uint32, slot=0, device='CPU')
    self.vmem = UOp.param(1, dtypes.uint8, (_VMEM_BYTES,))
    self.loop = UOp.loop(0)
    self.values: dict[Register, UOp] = {}
    self.dirty: set[Register] = set()
    self.current_gate = _const(True, dtypes.bool)
    self.load_specs: dict[int, tuple[int, int]] = {}
    self.fault: UOp
    self.fault_pc: UOp
    self.fault_lo: UOp
    self.fault_hi: UOp
    self.load_checks: UOp

  def _control(self, offset: int) -> UOp:
    return self.regfile.after(self.loop).index(self.control_base + offset).load()

  @staticmethod
  def _u64(low: UOp, high: UOp) -> UOp:
    return low.cast(dtypes.uint64) | (high.cast(dtypes.uint64) << _const(32, dtypes.uint64))

  def read(self, src: Register | int) -> UOp:
    if isinstance(src, int): return _const(src & _UINT32_MASK, dtypes.uint32)
    if src not in self.slot: raise UnsupportedIR3Block(f'unsupported IR3 loop source {src}')
    if src not in self.values:
      self.values[src] = self.regfile.after(self.loop).index(self.slot[src]).load()
    return self.values[src]

  def write(self, dst: Register, value: UOp):
    if dst not in self.slot: raise UnsupportedIR3Block(f'unsupported IR3 loop destination {dst}')
    self.values[dst] = self.current_gate.where(_output_mask(value, dst), self.read(dst))
    self.dirty.add(dst)

  def _bounds_contains(self, target: UOp, span: int) -> UOp:
    count = self._control(_LOOP_BOUNDS_COUNT)
    inside = _const(False, dtypes.bool)
    for index in range(_MAX_LOOP_MEMORY_RANGES):
      base = _LOOP_BOUNDS_BASE + index * 4
      start, end = self._u64(self._control(base), self._control(base + 1)), self._u64(self._control(base + 2), self._control(base + 3))
      complete = end >= _const(span, dtypes.uint64)
      in_this = (count > _const(index, dtypes.uint32)) & complete & (target >= start) & (target <= end - _const(span, dtypes.uint64))
      inside = inside | in_this
    return inside

  def lower_ldg(self, inst: IR3Instruction, pc: int) -> None:
    if not _register(inst.dst) or len(inst.srcs) != 3 or not _register(inst.srcs[0]):
      raise UnsupportedIR3Block('invalid IR3 ldg loop form')
    address_reg, offset, count = inst.srcs
    if not isinstance(offset, int) or not isinstance(count, int) or not 1 <= count <= 4:
      raise UnsupportedIR3Block('invalid IR3 ldg loop operands')
    itemsize, span = _itemsize(inst.types[0]), count * _itemsize(inst.types[0])
    address = self._u64(self.read(address_reg), self.read(_next_reg(address_reg)))
    target = address + _const(offset & (_U64_LIMIT - 1), dtypes.uint64)
    in_bounds, active = self._bounds_contains(target, span), self.current_gate
    safe, bad = active & in_bounds, active & in_bounds.logical_not()
    first_bad = self.fault.eq(_const(0, dtypes.uint32)) & bad
    self.fault = bad.where(_const(1, dtypes.uint32), self.fault)
    self.fault_pc = first_bad.where(_const(pc, dtypes.uint32), self.fault_pc)
    self.fault_lo = first_bad.where(address.cast(dtypes.uint32), self.fault_lo)
    self.fault_hi = first_bad.where((address >> _const(32, dtypes.uint64)).cast(dtypes.uint32), self.fault_hi)
    self.load_checks = self.load_checks + active.cast(dtypes.uint32)
    self.load_specs[pc] = (offset, span)
    for component in range(count):
      value = _const(0, dtypes.uint32)
      component_address = target + _const(component * itemsize, dtypes.uint64)
      for byte in range(itemsize):
        index = self.vmem.index((component_address + _const(byte, dtypes.uint64)).cast(dtypes.int64).valid(safe))
        value = value | (safe.where(index.load(), _const(0, dtypes.uint8)).cast(dtypes.uint32) << _const(byte * 8, dtypes.uint32))
      self.write(_reg_offset(inst.dst, component), value)

  def lower_instruction(self, inst: IR3Instruction, pc: int) -> None:
    if inst.name == 'nop': return
    if inst.name == 'ldg':
      self.lower_ldg(inst, pc)
      return
    if inst.name not in _NATIVE or not _register(inst.dst) or inst.sat:
      raise UnsupportedIR3Block(f'unsupported IR3 loop opcode {inst.name}')
    for component in range(inst.repeat + 1): self.lower_one(inst, component)

  def compile_loop(self) -> tuple[Any, tuple[tuple[Register, int], ...], dict[int, tuple[int, int]]]:
    self.fault, self.fault_pc = self._control(_LOOP_FAULT), self._control(_LOOP_FAULT_PC)
    self.fault_lo, self.fault_hi = self._control(_LOOP_FAULT_LO), self._control(_LOOP_FAULT_HI)
    timeout_fuel = self._control(_LOOP_TIMEOUT_FUEL)
    iterations, self.load_checks = self._control(_LOOP_ITERATIONS), self._control(_LOOP_LOAD_CHECKS)
    can_reach_branch = timeout_fuel >= _const(self.shape.exit_steps, dtypes.uint32)
    self.current_gate = can_reach_branch
    for index, inst in enumerate(self.instructions[:self.shape.branch_index]): self.lower_instruction(inst, self.shape.start_pc + index)
    branch = self.instructions[self.shape.branch_index]
    taken = self.read(branch.srcs[0]).ne(_const(0, dtypes.uint32))
    if branch.invert: taken = taken.logical_not()
    fallthrough = can_reach_branch & taken.logical_not()
    can_complete_loop = timeout_fuel >= _const(self.shape.loop_steps, dtypes.uint32)
    run_backedge = fallthrough & can_complete_loop
    self.current_gate = run_backedge
    for index, inst in enumerate(self.instructions[self.shape.branch_index + 1:self.shape.jump_index], self.shape.branch_index + 1):
      self.lower_instruction(inst, self.shape.start_pc + index)
    timeout_now = can_reach_branch.logical_not() | (fallthrough & can_complete_loop.logical_not())
    consumed = taken.where(_const(self.shape.exit_steps, dtypes.uint32), _const(self.shape.loop_steps, dtypes.uint32))
    next_fuel = timeout_now.where(timeout_fuel, timeout_fuel - consumed)
    continue_loop = run_backedge & self.fault.eq(_const(0, dtypes.uint32))
    # A Mesa loop commonly performs its body before the forward exit branch;
    # a branch-at-header loop performs it after.  Count the gate that actually
    # admits body work, so both shapes report their logical trip count.
    body_entry = can_reach_branch if any(inst.name != 'nop' for inst in self.instructions[:self.shape.branch_index]) else continue_loop
    next_iterations = iterations + body_entry.cast(dtypes.uint32)
    stores = [self.regfile.index(self.slot[reg]).store(self.values[reg]) for reg in sorted(self.dirty)]
    stores += [self.regfile.index(self.control_base + _LOOP_FAULT).store(self.fault),
               self.regfile.index(self.control_base + _LOOP_FAULT_PC).store(self.fault_pc),
               self.regfile.index(self.control_base + _LOOP_FAULT_LO).store(self.fault_lo),
               self.regfile.index(self.control_base + _LOOP_FAULT_HI).store(self.fault_hi),
               self.regfile.index(self.control_base + _LOOP_TIMEOUT).store(timeout_now.cast(dtypes.uint32)),
               self.regfile.index(self.control_base + _LOOP_TIMEOUT_FUEL).store(timeout_now.where(timeout_fuel, next_fuel)),
               self.regfile.index(self.control_base + _LOOP_ITERATIONS).store(next_iterations),
               self.regfile.index(self.control_base + _LOOP_LOAD_CHECKS).store(self.load_checks)]
    sink = UOp.group(*stores).end(self.loop, continue_loop).sink(arg=KernelInfo('ir3_cpu_loop', opts_to_apply=()))
    program = to_program(sink, Device['CPU'].renderer)
    return get_runtime('CPU', program), tuple((reg, self.slot[reg]) for reg in sorted(self.dirty)), self.load_specs


def loop_shape(program: tuple[IR3Instruction, ...], start_pc: int, supported) -> _LoopShape | None:
  """Recognize only `supported body; br forward exit; jump backward header`."""
  if not 0 <= start_pc < len(program): return None
  branch_pc = None
  for pc in range(start_pc, len(program)):
    inst = program[pc]
    if inst.name == 'br' and inst.branch_offset > 0:
      branch_pc = pc
      break
    if inst.name in {'br', 'jump', 'bany', 'ball', 'brao', 'braa', 'predt', 'predf', 'prede', 'bar', 'fence', 'end'}:
      return None
  if branch_pc is None: return None
  exit_pc = branch_pc + program[branch_pc].branch_offset
  if not branch_pc + 1 < exit_pc <= len(program): return None
  jump_pc = exit_pc - 1
  jump = program[jump_pc]
  if jump.name != 'jump' or jump.branch_offset >= 0 or jump_pc + jump.branch_offset != start_pc: return None
  shape = _LoopShape(start_pc, branch_pc, jump_pc, exit_pc, program[start_pc:exit_pc])
  for index, inst in enumerate(shape.instructions):
    if index == shape.branch_index:
      if inst.name != 'br' or len(inst.srcs) != 1 or not _register(inst.srcs[0]): return None
    elif index == shape.jump_index:
      if inst.name != 'jump': return None
    elif inst.name == 'ldg':
      if not _register(inst.dst) or len(inst.srcs) != 3 or not _register(inst.srcs[0]) or inst.repeat or inst.sat: return None
      _, offset, count = inst.srcs
      if not isinstance(offset, int) or not isinstance(count, int) or not 1 <= count <= 4: return None
    elif not supported(inst):
      return None
  try: _loop_registers(shape)
  except UnsupportedIR3Block: return None
  return shape


def has_loop(program: tuple[IR3Instruction, ...], supported) -> bool:
  return any(inst.name == 'jump' and inst.branch_offset < 0 and
             loop_shape(program, pc + inst.branch_offset, supported) is not None for pc, inst in enumerate(program))


def _select_memory_bounds(regs: dict[Register, list[int]], bounds: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...] | None:
  """Keep only mappings anchored by a one-lane live-in adjacent register pair."""
  candidates: set[int] = set()
  for reg, values in regs.items():
    if not _register(reg) or len(values) != 1: continue
    if (high := regs.get(_next_reg(reg))) is None or len(high) != 1: continue
    candidates.add((values[0] & _UINT32_MASK) | ((high[0] & _UINT32_MASK) << 32))
  try: selected = tuple((start, end) for start, end in bounds if any(start <= address < end for address in candidates))
  except TypeError: return None
  return _normalise_memory_bounds(selected)


def _vmem_buffer(runner: 'IR3UOpRunner') -> Buffer:
  if runner._vmem is None:
    runner._vmem = Buffer('CPU', _VMEM_BYTES, dtypes.uint8, options=BufferSpec(external_ptr=0)).ensure_allocated()
  return runner._vmem


def loop_block(runner: 'IR3UOpRunner', program: tuple[IR3Instruction, ...], start_pc: int) -> _NativeLoopBlock | None:
  key = (id(program), start_pc)
  if (cached := runner._get_lru(runner.loop_cache, key)) is not None and cached[0] is program:
    runner.stats.cache_hits += 1
    return cached[1]
  if key in runner.loop_uncompilable: return None
  if (shape := loop_shape(program, start_pc, runner._supported)) is None:
    runner._put_lru(runner.loop_uncompilable, key, None, runner.max_loop_locations * 4)
    return None
  try:
    slots = _loop_registers(shape)
    # The generated fault metadata contains decoded PCs.  Keep the start PC
    # in this key unless runtime metadata is split from per-program shape.
    signature = (shape.start_pc, slots, shape.branch_index, shape.jump_index,
                 tuple(_instruction_signature(inst) for inst in shape.instructions))
    if (block := runner._get_lru(runner.compiled_loops, signature)) is None:
      if len(runner.compiled_loops) >= runner.max_compiled_loops:
        raise UnsupportedIR3Block('native IR3 loop compile budget exhausted')
      lowerer = _LoopLowerer(shape, slots)
      runtime, dirty_slots, load_specs = lowerer.compile_loop()
      regfile = Buffer('CPU', len(slots) + _LOOP_CONTROL_WORDS, dtypes.uint32).allocate()
      block = _NativeLoopBlock(shape, slots, dirty_slots, load_specs, runtime, regfile,
                               regfile.as_memoryview(force_zero_copy=True).cast('I'))
      runner.compiled_loops[signature] = block
      runner.stats.compiled += 1
    if runner._put_lru(runner.loop_cache, key, (program, block), runner.max_loop_locations) is not None:
      runner.stats.cache_evictions += 1
    return block
  except (AssertionError, RuntimeError, UnsupportedIR3Block):
    runner._put_lru(runner.loop_uncompilable, key, None, runner.max_loop_locations * 4)
    return None


def try_run_loop(runner: 'IR3UOpRunner', program: tuple[IR3Instruction, ...], start_pc: int, regs: dict[Register, list[int]],
                 exec_mask: list[bool], *, check_range=None, memory_bounds: tuple[tuple[int, int], ...] | None = None,
                 max_steps: int | None = None) -> tuple[int, int] | None:
  """Run one eligible decoded-IR3 natural loop in a single native CPU call."""
  runner.stats.attempts += 1
  if not regs or exec_mask != [True] or any(len(values) != 1 for values in regs.values()):
    runner.stats.fallbacks += 1
    return None
  if (block := loop_block(runner, program, start_pc)) is None:
    runner.stats.fallbacks += 1
    return None
  if block.load_specs:
    if check_range is None:
      # Mirror the scalar ldg prerequisite before touching staging state.
      first_pc = min(block.load_specs)
      raise RuntimeError(f'IR3 ldg requires a mapped-memory validator at PC {first_pc}')
    if memory_bounds is None or (bounds := _select_memory_bounds(regs, memory_bounds)) is None:
      runner.stats.fallbacks += 1
      return None
  else: bounds = ()
  if max_steps is None: max_steps = max(100000, len(program) * 65536)
  if not isinstance(max_steps, int) or not 0 <= max_steps <= _UINT32_MASK:
    runner.stats.fallbacks += 1
    return None
  outcome = block.run(regs, _vmem_buffer(runner), bounds, max_steps)
  runner.stats.native_calls += 1
  runner.stats.iterations += outcome.iterations
  runner.stats.load_checks += outcome.load_checks
  if outcome.fault:
    runner.stats.load_rejections += 1
    offset, span = block.load_specs[outcome.fault_pc]
    target = (outcome.fault_address + offset) & (_U64_LIMIT - 1)
    try: check_range(target, span)
    except Exception as exc:
      raise RuntimeError(f'IR3 ldg memory fault at PC {outcome.fault_pc}, lane 0, address={outcome.fault_address:#x}') from exc
    # The bounds snapshot was incomplete, not a license to dereference an
    # unselected mapping.  Leave the caller state untouched and let its
    # scalar decoded-IR3 path validate and execute this exact iteration.
    runner.stats.fallbacks += 1
    return None
  if outcome.timeout:
    # A scalar timeout may leave architectural registers updated through the
    # exact failing instruction.  Discard staging and replay it, rather than
    # exposing a different post-error register state from this fast path.
    runner.stats.fallbacks += 1
    raise IR3UOpLoopTimeout(block.shape.start_pc)
  block.commit(regs)
  runner.stats.runs += 1
  return block.shape.exit_pc, max_steps - outcome.timeout_fuel
