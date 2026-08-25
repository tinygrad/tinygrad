"""Workgroup construction and scheduling for the mock A630 IR3 executor.

This module deliberately receives the decoded-IR3 executor as a callback so
that it can build workgroups without importing :mod:`executor` back again.
"""
from collections.abc import Callable
from typing import Any

from test.mockgpu.qcom.decoder import IR3Instruction, decode_ir3
from test.mockgpu.qcom.registers import local_id_regs, workgroup_id_regs


Register = tuple[str, int, int]
Registers = dict[Register, list[int]]
ExecuteIR3 = Callable[..., int | None]
HasLoop = Callable[[tuple[IR3Instruction, ...]], bool]


def _no_native_loop(_program: tuple[IR3Instruction, ...]) -> bool: return False


def native_memory_bounds(check_range) -> tuple[tuple[int, int], ...] | None:
  """Snapshot active mock-GPU mappings for bounds-gated native execution."""
  owner = getattr(check_range, '__self__', None)
  ranges = getattr(owner, 'mapped_ranges', None)
  if not isinstance(ranges, dict): return None
  return tuple(sorted((start, start + size) for (start, size), count in ranges.items() if count > 0))


def use_native_blocks(grid_size, local_size) -> bool:
  return grid_size[0] * grid_size[1] * grid_size[2] * local_size[0] * local_size[1] * local_size[2] >= 4096


def workgroup_batch_size(program: tuple[IR3Instruction, ...], lane_count: int, has_loop: HasLoop) -> int:
  if not 1 <= lane_count <= 64: return 1
  if any(inst.name in {'bany', 'ball', 'brao', 'braa'} or inst.name.startswith('atomic.') for inst in program): return 1
  if has_loop(program): return 1
  return 256 // lane_count


def execute_dispatch(execute_ir3: ExecuteIR3, code: bytes, grid_size, local_size, local_id_register, initial_regs=None,
                     check_range=None, workgroup_id_register=0xfc, textures=(), ibos=(), global_id_register=0xfc,
                     linear_group_register=0xfc, local_id_order=(0, 1, 2), has_loop: HasLoop | None=None) -> Registers:
  """Run an IR3 dispatch using ``execute_ir3`` for each scheduled wave batch."""
  if has_loop is None: has_loop = _no_native_loop
  lane_count = local_size[0] * local_size[1] * local_size[2]
  program = decode_ir3(code)
  uses_private = any(inst.name in {'ldp', 'stp'} for inst in program)
  allow_native_blocks = use_native_blocks(grid_size, local_size)
  memory_bounds = native_memory_bounds(check_range)
  all_local_ids = local_id_regs(local_size, local_id_register, local_id_order)

  def make_regs(coord, wave_start, wave_lanes) -> Registers:
    x, y, z = coord
    regs = {} if initial_regs is None else {key: values[wave_start:wave_start + wave_lanes] for key, values in initial_regs.items()}
    regs.update({key: values[wave_start:wave_start + wave_lanes] for key, values in all_local_ids.items()})
    if workgroup_id_register != 0xfc: regs.update(workgroup_id_regs(coord, wave_lanes, workgroup_id_register))
    if global_id_register != 0xfc:
      regs.update(workgroup_id_regs((x * local_size[0], y * local_size[1], z * local_size[2]), wave_lanes, global_id_register))
    if linear_group_register != 0xfc:
      regs[('r', linear_group_register // 4, linear_group_register % 4)] = \
        [x + grid_size[0] * (y + grid_size[1] * z)] * wave_lanes
    return regs

  # Independent one-wave workgroups can share one lane-wise scheduler batch when all
  # workgroup-local backing remains separate. Wave votes and atomics observe wave
  # boundaries, so those programs stay on the exact per-workgroup path below. Non-atomic
  # cross-workgroup races are undefined on hardware and may expose another legal interleaving.
  batch_groups = workgroup_batch_size(program, lane_count, has_loop)
  if batch_groups > 1:
    coords = [(x, y, z) for z in range(grid_size[2]) for y in range(grid_size[1]) for x in range(grid_size[0])]
    batch_last_regs: Registers = {}
    for batch_start in range(0, len(coords), batch_groups):
      batch = coords[batch_start:batch_start + batch_groups]
      regs: Registers = {}
      shared_lanes: list[bytearray] = []
      private_lanes: list[bytearray] | None = [] if uses_private else None
      for coord in batch:
        for key, values in make_regs(coord, 0, lane_count).items(): regs.setdefault(key, []).extend(values)
        group_shared = bytearray(0x10000)
        shared_lanes.extend([group_shared] * lane_count)
        if private_lanes is not None: private_lanes.extend(bytearray(0x10000) for _ in range(lane_count))
      pc = 0
      resume_state: dict[str, Any] = {}
      while (next_pc := execute_ir3(code, regs, check_range=check_range, start_pc=pc, shared=shared_lanes,
                                    private=private_lanes, stop_at_barrier=True, textures=textures, ibos=ibos,
                                    memory_bounds=memory_bounds, allow_native_blocks=allow_native_blocks,
                                    resume_state=resume_state)) is not None: pc = next_pc
      offset = (len(batch) - 1) * lane_count
      batch_last_regs = {key: values[offset:offset + lane_count] for key, values in regs.items()}
    return batch_last_regs

  last_regs: Registers = {}
  for z in range(grid_size[2]):
    for y in range(grid_size[1]):
      for x in range(grid_size[0]):
        waves, privates = [], []
        resume_states: list[dict[str, Any]] = []
        for wave_start in range(0, lane_count, 64):
          wave_lanes = min(64, lane_count - wave_start)
          waves.append(make_regs((x, y, z), wave_start, wave_lanes))
          privates.append([bytearray(0x10000) for _ in range(wave_lanes)] if uses_private else None)
          resume_states.append({})
        shared, pcs, done = bytearray(0x10000), [0] * len(waves), [False] * len(waves)
        while not all(done):
          reached_barrier: list[int] = []
          for index, regs in enumerate(waves):
            if done[index]: continue
            next_pc = execute_ir3(code, regs, check_range=check_range, start_pc=pcs[index], shared=shared,
                                  private=privates[index], stop_at_barrier=True, textures=textures, ibos=ibos,
                                  memory_bounds=memory_bounds, allow_native_blocks=allow_native_blocks,
                                  resume_state=resume_states[index])
            if next_pc is None: done[index] = True
            else: pcs[index], reached_barrier = next_pc, [*reached_barrier, index]
          if reached_barrier and any(done): raise RuntimeError('IR3 barrier reached by only part of a workgroup')
        last_regs = waves[-1]
  return last_regs
