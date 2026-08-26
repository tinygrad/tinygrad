from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from test.mockgpu.qcom.decoder import IR3Instruction
from test.mockgpu.qcom.loop_runner import IR3UOpLoopTimeout as IR3UOpLoopTimeout, has_loop, try_run_loop
from test.mockgpu.qcom.registers import _lane_count
from test.mockgpu.qcom.uop_runner import _Lowerer, _NATIVE, IR3UOpBlock, UnsupportedIR3Block, _register


@dataclass
class IR3UOpRunnerStats:
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

  def reset(self):
    for key in self.__dataclass_fields__: setattr(self, key, 0)


class IR3UOpRunner:
  def __init__(self, min_instructions=8, max_instructions=64, max_register_slots=128, max_average_register_slots=40,
               max_compiled_blocks=512, max_block_locations=8192, max_programs=128, max_compiled_loops=32, max_loop_locations=64,
               max_narrow_compiled_blocks=None, max_regular_narrow_compiled_blocks=None):
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
    self.compiled: OrderedDict[tuple, IR3UOpBlock] = OrderedDict()
    self.compiled_classes: dict[tuple, str] = {}
    self.loop_cache: OrderedDict[tuple[int, int], tuple[tuple[IR3Instruction, ...], Any]] = OrderedDict()
    self.loop_uncompilable: OrderedDict[tuple[int, int], None] = OrderedDict()
    self.compiled_loops: OrderedDict[tuple, Any] = OrderedDict()
    self.stats, self._vmem = IR3UOpRunnerStats(), None

  @staticmethod
  def _put_lru(cache, key, value, limit):
    cache[key] = value
    cache.move_to_end(key)
    return cache.popitem(last=False) if len(cache) > limit else None

  @staticmethod
  def _get_lru(cache, key):
    if key not in cache: return None
    cache.move_to_end(key)
    return cache[key]

  def _evict_compiled_class(self, block_class):
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
  def _supported(inst):
    if inst.name == 'nop': return True
    if inst.name not in _NATIVE or not _register(inst.dst) or inst.sat or (inst.name == 'mov' and inst.rounding): return False
    return not any(isinstance(src, tuple) and src[0] in {'rel', 'relr', 'relhr'} for src in inst.srcs)

  @classmethod
  def has_loop(cls, program): return has_loop(program, cls._supported)
  def try_run_loop(self, program, start_pc, regs, exec_mask, *, check_range=None, memory_bounds=None, max_steps=None):
    return try_run_loop(self, program, start_pc, regs, exec_mask, check_range=check_range,
                        memory_bounds=memory_bounds, max_steps=max_steps)

  def _blocks(self, program):
    program_id = id(program)
    if (cached := self._get_lru(self.program_blocks, program_id)) is not None and cached[0] is program: return cached[1]
    blocks, pc = {}, 0
    while pc < len(program):
      if not self._supported(program[pc]):
        pc += 1
        continue
      start = pc
      while pc < len(program) and self._supported(program[pc]): pc += 1
      for candidate in range(start, pc):
        stop, count = candidate, 0
        while stop < pc and count < self.max_instructions:
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

  def can_run_blocks(self, program, lanes=1):
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
    enabled = bool(pressures) and sum(pressures) <= average_limit * len(pressures) and max(pressures) <= self.max_register_slots
    self._put_lru(self.program_policy, policy_key, (program, enabled), self.max_programs * 2)
    return enabled

  def try_run(self, program, start_pc, regs, exec_mask, predication=None, mask_pcs=None, *, policy_checked=False):
    lanes = _lane_count(regs)
    if not lanes: return None
    if not policy_checked and not self.can_run_blocks(program, lanes): return None
    self.stats.block_attempts += 1
    if (end_pc := self._blocks(program).get(start_pc)) is None: return None
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
          if lanes >= 16: block_class, class_limit = 'wide', self.max_compiled_blocks - self.max_narrow_compiled_blocks
          else:
            block_class = 'priority_narrow' if len(program) >= 256 else 'regular_narrow'
            class_limit = self.max_regular_narrow_compiled_blocks if block_class == 'regular_narrow' else \
              self.max_narrow_compiled_blocks - self.max_regular_narrow_compiled_blocks
          if sum(value == block_class for value in self.compiled_classes.values()) >= class_limit:
            if block_class == 'regular_narrow' or not self._evict_compiled_class(block_class):
              raise UnsupportedIR3Block(f'{block_class} native IR3 block compile budget exhausted')
          if len(self.compiled) >= self.max_compiled_blocks and not self._evict_compiled_class(block_class):
            raise UnsupportedIR3Block('native IR3 block compile budget exhausted')
          block = lowerer.compile()
          self.stats.block_compiles += 1
          self.compiled[signature], self.compiled_classes[signature] = block, block_class
        if self._put_lru(self.cache, key, block, self.max_block_locations) is not None: self.stats.cache_evictions += 1
      if block is None: return None
      block.run(regs, mask)
      self.stats.block_runs += 1
      return start_pc + block.length
    except UnsupportedIR3Block:
      self.stats.block_declines += 1
      self._put_lru(self.uncompilable, key, None, self.max_block_locations * 2)
      return None
