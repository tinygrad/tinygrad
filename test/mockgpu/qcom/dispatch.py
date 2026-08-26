"""Workgroup construction and scheduling for the mock A630 IR3 executor."""
from typing import Any

import test.mockgpu.qcom.executor as executor
from test.mockgpu.qcom.decoder import IR3Instruction, decode_ir3
from test.mockgpu.qcom.registers import IR3ConstantBank, IR3RegisterFile, local_id_regs, workgroup_id_regs


Register = tuple[str, int, int]
Registers = dict[Register, list[int]]


def use_native_blocks(grid_size, local_size) -> bool:
  return grid_size[0] * grid_size[1] * grid_size[2] * local_size[0] * local_size[1] * local_size[2] >= 4096


def workgroup_batch_size(program: tuple[IR3Instruction, ...], lane_count: int) -> int:
  if not 1 <= lane_count <= 64: return 1
  if any(inst.name in {'bany', 'ball', 'brao', 'braa'} or inst.name.startswith('atomic.') for inst in program): return 1
  if executor._UOP_RUNNER is not None and executor._UOP_RUNNER.has_loop(program): return 1
  return 256 // lane_count


def execute_dispatch(code: bytes, grid_size, local_size, local_id_register, initial_regs=None, check_range=None,
                     workgroup_id_register=0xfc, textures=(), ibos=(), global_id_register=0xfc, linear_group_register=0xfc,
                     local_id_order=(0, 1, 2), constant_words: tuple[int, ...] = ()) -> Registers:
  lane_count = local_size[0] * local_size[1] * local_size[2]
  program = decode_ir3(code)
  uses_private = any(inst.name in {'ldp', 'stp'} for inst in program)
  uses_shared = any(inst.name in {'ldl', 'stl'} for inst in program)
  allow_native_blocks = use_native_blocks(grid_size, local_size)
  memory_bounds = executor._native_memory_bounds(check_range)
  all_local_ids = local_id_regs(local_size, local_id_register, local_id_order)
  constants = IR3ConstantBank(constant_words)

  def make_regs(coord, wave_start, wave_lanes) -> IR3RegisterFile:
    x, y, z = coord
    regs = IR3RegisterFile(wave_lanes, None if initial_regs is None else
      {key: values[wave_start:wave_start + wave_lanes] for key, values in initial_regs.items()}, constants)
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
  batch_groups = workgroup_batch_size(program, lane_count)
  if batch_groups > 1:
    coords = [(x, y, z) for z in range(grid_size[2]) for y in range(grid_size[1]) for x in range(grid_size[0])]
    batch_last_regs: Registers = IR3RegisterFile(lane_count, constants=constants)
    for batch_start in range(0, len(coords), batch_groups):
      batch = coords[batch_start:batch_start + batch_groups]
      # One register row per batch, laid out group-major and lane-minor, instead of
      # one make_regs dict per workgroup. Override order matches make_regs: initial,
      # local ID, workgroup ID, global ID, linear group ID.
      regs = IR3RegisterFile(len(batch) * lane_count, constants=constants)
      if initial_regs is not None:
        for key, values in initial_regs.items(): regs[key] = values[:lane_count] * len(batch)
      for key, values in all_local_ids.items(): regs[key] = values[:lane_count] * len(batch)
      if workgroup_id_register != 0xfc:
        for component in range(3):
          regs[('r', workgroup_id_register, component)] = [coord[component] for coord in batch for _ in range(lane_count)]
      if global_id_register != 0xfc:
        for component in range(3):
          regs[('r', global_id_register, component)] = \
            [coord[component] * local_size[component] for coord in batch for _ in range(lane_count)]
      if linear_group_register != 0xfc:
        regs[('r', linear_group_register // 4, linear_group_register % 4)] = \
          [coord[0] + grid_size[0] * (coord[1] + grid_size[1] * coord[2]) for coord in batch for _ in range(lane_count)]
      shared_lanes: list[bytearray] | None = [] if uses_shared else None
      private_lanes: list[bytearray] | None = [] if uses_private else None
      if shared_lanes is not None or private_lanes is not None:
        for _ in batch:
          if shared_lanes is not None:
            group_shared = bytearray(0x10000)
            shared_lanes.extend([group_shared] * lane_count)
          if private_lanes is not None: private_lanes.extend(bytearray(0x10000) for _ in range(lane_count))
      pc = 0
      resume_state: dict[str, Any] = {}
      while (next_pc := executor.execute_ir3(code, regs, check_range=check_range, start_pc=pc, shared=shared_lanes,
                                             private=private_lanes, stop_at_barrier=True, textures=textures, ibos=ibos,
                                             memory_bounds=memory_bounds, allow_native_blocks=allow_native_blocks,
                                             resume_state=resume_state)) is not None: pc = next_pc
      offset = (len(batch) - 1) * lane_count
      batch_last_regs = IR3RegisterFile(lane_count,
        {key: values[offset:offset + lane_count] for key, values in regs.items()}, constants)
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
        shared, pcs, done = (bytearray(0x10000) if uses_shared else None), [0] * len(waves), [False] * len(waves)
        while not all(done):
          reached_barrier: list[int] = []
          for index, regs in enumerate(waves):
            if done[index]: continue
            next_pc = executor.execute_ir3(code, regs, check_range=check_range, start_pc=pcs[index], shared=shared,
                                           private=privates[index], stop_at_barrier=True, textures=textures, ibos=ibos,
                                           memory_bounds=memory_bounds, allow_native_blocks=allow_native_blocks,
                                           resume_state=resume_states[index])
            if next_pc is None: done[index] = True
            else: pcs[index], reached_barrier = next_pc, [*reached_barrier, index]
          if reached_barrier and any(done): raise RuntimeError('IR3 barrier reached by only part of a workgroup')
        last_regs = waves[-1]
  return last_regs
