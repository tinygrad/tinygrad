import ctypes, struct
from typing import Any
from tinygrad.runtime.autogen import mesa
from test.mockgpu.qcom.decoder import decode_ir3
from test.mockgpu.qcom.dispatch import (execute_dispatch as _execute_dispatch, native_memory_bounds as _native_memory_bounds,
                                        use_native_blocks as _dispatch_use_native_blocks,
                                        workgroup_batch_size as _dispatch_workgroup_batch_size)
from test.mockgpu.qcom.registers import (_alu_runner, _compare, _convert, _float, _float_bits, _itemsize, _mod_float, _next_reg, _reg_offset,
                                         _s32, _signed, _float_values, _values, _write)
from test.mockgpu.qcom.uop_runner import IR3UOpLoopTimeout, IR3UOpRunner

_UOP_RUNNER = IR3UOpRunner()

def _has_native_loop(program) -> bool:
  return _UOP_RUNNER is not None and getattr(_UOP_RUNNER, 'has_loop', lambda _program: False)(program)

def _use_native_blocks(grid_size, local_size) -> bool:
  return _dispatch_use_native_blocks(grid_size, local_size)

def _workgroup_batch_size(program, lane_count: int) -> int:
  return _dispatch_workgroup_batch_size(program, lane_count, _has_native_loop)

def _source_offset(src, offset):
  if isinstance(src, int): return src
  if src[0] in {'rel', 'relr', 'relhr'}: return (src[0], src[1] + offset, src[2])
  return _reg_offset(src, offset)

def _source_values(regs, src, lanes, half=False):
  if not isinstance(src, tuple) or src[0] != 'rel': return _values(regs, src, lanes)
  addresses = _values(regs, ('a', 61, 0), lanes)
  kind = ('hc' if src[2] else 'hr') if half else ('c' if src[2] else 'r')
  values = []
  for lane, address in enumerate(addresses):
    component = address + src[1]
    values.append(regs.get((kind, component // 4, component % 4), [0] * lanes)[lane])
  return values

def _write_destination(regs, dst, values, mask):
  if dst[0] not in {'relr', 'relhr'}: return _write(regs, dst, values, mask)
  addresses = _values(regs, ('a', 61, 0), len(values))
  kind = 'hr' if dst[0] == 'relhr' else 'r'
  for lane, (address, value, active) in enumerate(zip(addresses, values, mask, strict=True)):
    if not active: continue
    component = address + dst[1]
    target = (kind, component // 4, component % 4)
    previous = regs.setdefault(target, [0] * len(values))
    previous[lane] = value

def _check_access(check_range, address, size, name, pc):
  if check_range is None: raise RuntimeError(f'IR3 {name} requires a mapped-memory validator at PC {pc}')
  check_range(address, size)

def _private_lanes(private:bytearray|list[bytearray]|None, lanes:int, name:str) -> list[bytearray]:
  if isinstance(private, bytearray):
    if lanes == 1: return [private]
    raise RuntimeError(f'IR3 {name} requires one private backing per lane, got one backing for {lanes} lanes')
  if not isinstance(private, list) or len(private) != lanes or any(not isinstance(memory, bytearray) for memory in private):
    raise RuntimeError(f'IR3 {name} requires {lanes} bytearray private backings')
  return private

def _validate_targets(targets, mask, unit, check_range, name, pc):
  active = [target for target, active in zip(targets, mask, strict=True) if active]
  if active:
    # One aggregate span check usually covers every lane; per-lane checks reproduce exact faults otherwise.
    low, high = min(active), max(active)
    try:
      check_range(low, high - low + unit)
      return
    except Exception: pass
  for lane, target in enumerate(targets):
    if mask[lane]: _check_access(check_range, target, unit, name, pc)

def _read_targets(targets, mask, itemsize):
  char = {1: 'B', 2: 'H', 4: 'I'}[itemsize]
  out: list[int] = []
  lane = 0
  while lane < len(targets):
    if not mask[lane]:
      out.append(0)
      lane += 1
      continue
    run = lane + 1
    while run < len(targets) and mask[run] and targets[run] == targets[run - 1] + itemsize: run += 1
    if run > lane + 1: out.extend(struct.unpack(f'<{run - lane}{char}', ctypes.string_at(targets[lane], (run - lane) * itemsize)))
    else: out.append(int.from_bytes(ctypes.string_at(targets[lane], itemsize), 'little'))
    lane = run
  return out

def _write_targets(targets, mask, values, itemsize):
  unit_mask = (1 << (itemsize * 8)) - 1
  lane = 0
  while lane < len(targets):
    if not mask[lane]:
      lane += 1
      continue
    run = lane + 1
    while run < len(targets) and mask[run] and targets[run] == targets[run - 1] + itemsize: run += 1
    if run > lane + 1 and itemsize == 4:
      ctypes.memmove(targets[lane], b''.join((values[index] & 0xffffffff).to_bytes(4, 'little') for index in range(lane, run)), (run - lane) * 4)
    else:
      for index in range(lane, run):
        ctypes.memmove(targets[index], (values[index] & unit_mask).to_bytes(itemsize, 'little'), itemsize)
    lane = run

def execute_ir3(code:bytes, regs:dict[tuple[str, int, int], list[int]], gpu_id:int=630, check_range=None, start_pc=0,
                shared:bytearray|list[bytearray]|None=None, private:bytearray|list[bytearray]|None=None,
                stop_at_barrier=False, trace:dict|None=None,
                textures=(), ibos=(), memory_bounds:tuple[tuple[int, int], ...]|None=None, allow_native_blocks=True,
                resume_state:dict[str, Any]|None=None):
  program, pc = decode_ir3(code, gpu_id), start_pc
  step_limit = max(100000, len(program) * 65536)
  if memory_bounds is None: memory_bounds = _native_memory_bounds(check_range)
  lanes = len(next(iter(regs.values())))
  branch_frames: list[dict[str, Any]]
  if resume_state:
    steps = resume_state['steps']
    predication = resume_state['predication']
    exec_mask = resume_state['exec_mask']
    branch_frames = resume_state['branch_frames']
    native_loops_disabled = resume_state['native_loops_disabled']
  else:
    steps, predication = 0, None
    exec_mask = [True] * lanes
    branch_frames = []
    native_loops_disabled = False
  can_run_blocks = None if _UOP_RUNNER is None else getattr(_UOP_RUNNER, 'can_run_blocks', None)
  native_blocks_enabled = allow_native_blocks and trace is None and _UOP_RUNNER is not None and \
    (can_run_blocks is None or can_run_blocks(program, lanes))
  while pc < len(program):
    if branch_frames:
      frame = branch_frames[-1]
      if frame['reconv'] is not None and pc == frame['reconv'][0]:
        exec_mask = frame['reconv'][1]
        branch_frames.pop()
      elif frame['alternate'] is not None and pc == frame['alternate'][0]:
        exec_mask = [a or b for a, b in zip(exec_mask, frame['alternate'][1], strict=True)]
        branch_frames.pop()
    if trace is None and _UOP_RUNNER is not None and not native_loops_disabled and exec_mask == [True] and \
       predication is None and not branch_frames:
      try:
        loop_result = _UOP_RUNNER.try_run_loop(program, pc, regs, exec_mask, check_range=check_range,
                                              memory_bounds=memory_bounds, max_steps=step_limit - steps)
      except IR3UOpLoopTimeout:
        native_loops_disabled = True
      else:
        if loop_result is not None:
          pc, loop_steps = loop_result
          steps += loop_steps
          continue
    if native_blocks_enabled and \
       (next_pc := _UOP_RUNNER.try_run(program, pc, regs, exec_mask, predication,
         mask_pcs=None if not branch_frames else frozenset(target for frame in branch_frames for target in
           ((frame['reconv'] or (0,))[0], (frame['alternate'] or (0,))[0])), policy_checked=True)) is not None:
      steps += next_pc - pc
      if steps > step_limit: raise RuntimeError(f'IR3 execution did not terminate at PC {pc}')
      pc = next_pc
      continue
    inst_pc, inst = pc, program[pc]
    if trace is not None: trace[inst_pc] = {key: values.copy() for key, values in regs.items()}
    pc += 1
    steps += 1
    if steps > step_limit: raise RuntimeError(f'IR3 execution did not terminate at PC {inst_pc}')
    if inst.name == 'nop': continue
    if inst.name == 'end': break
    if inst.name == 'bar':
      if stop_at_barrier:
        if resume_state is not None:
          resume_state.clear()
          resume_state.update(steps=steps, predication=predication, exec_mask=exec_mask, branch_frames=branch_frames,
                              native_loops_disabled=native_loops_disabled)
        return pc
      continue
    if inst.name == 'fence': continue
    if inst.name == 'jump':
      target = inst_pc + inst.branch_offset
      if branch_frames and branch_frames[-1]['alternate'] is not None:
        frame = branch_frames[-1]
        alternate_pc, alternate_mask = frame['alternate']
        frame['alternate'] = None
        frame['reconv'] = (target, [a or b for a, b in zip(exec_mask, alternate_mask, strict=True)])
        pc, exec_mask = alternate_pc, alternate_mask
      else: pc = target
      continue
    if inst.name == 'br':
      cond = [not bool(x) if inst.invert else bool(x) for x in _values(regs, inst.srcs[0], lanes)]
      taken = [active and value for active, value in zip(exec_mask, cond, strict=True)]
      fallthrough = [active and not value for active, value in zip(exec_mask, cond, strict=True)]
      if any(taken) and any(fallthrough):
        branch_frames.append({'alternate':(inst_pc + inst.branch_offset, taken), 'reconv':None})
        exec_mask = fallthrough
      elif any(taken): pc = inst_pc + inst.branch_offset
      continue
    if inst.name in {'bany', 'ball'}:
      cond = [not bool(x) if inst.invert else bool(x) for x in _values(regs, inst.srcs[0], lanes)]
      active_cond = [value for value, active in zip(cond, exec_mask, strict=True) if active]
      if (any(active_cond) if inst.name == 'bany' else all(active_cond)): pc = inst_pc + inst.branch_offset
      continue
    if inst.name in {'brao', 'braa'}:
      predicates = [[not bool(x) if invert else bool(x) for x in _values(regs, src, lanes)]
                    for src, invert in zip(inst.srcs, inst.inverts, strict=True)]
      lane_cond = [a or b if inst.name == 'brao' else a and b for a, b in zip(*predicates, strict=True)]
      taken = [active and value for active, value in zip(exec_mask, lane_cond, strict=True)]
      fallthrough = [active and not value for active, value in zip(exec_mask, lane_cond, strict=True)]
      if any(taken) and any(fallthrough):
        branch_frames.append({'alternate':(inst_pc + inst.branch_offset, taken), 'reconv':None})
        exec_mask = fallthrough
      elif any(taken): pc = inst_pc + inst.branch_offset
      continue
    if inst.name in {'predt', 'predf'}:
      predicate = _values(regs, ('p', 62, 0), lanes)
      predication = [bool(value) == (inst.name == 'predt') for value in predicate]
      continue
    if inst.name == 'prede':
      predication = None
      continue
    write_mask = exec_mask if predication is None else [active and pred for active, pred in zip(exec_mask, predication, strict=True)]
    if inst.name in {'ashr.b', 'shl.b'}:
      lanes = len(next(iter(regs.values())))
      repeated = inst.repeat_srcs + (False,) * (2 - len(inst.repeat_srcs))
      runner = _alu_runner(inst.name, 0, inst.src_mods + (0,) * (2 - len(inst.src_mods)), inst.source_half, inst.dst[0].startswith('h'))
      for component in range(inst.repeat + 1):
        srcs = tuple(_source_offset(src, component) if repeat else src
                     for src, repeat in zip(inst.srcs, repeated, strict=True))
        values = [_values(regs, src, lanes) for src in srcs]
        _write(regs, _reg_offset(inst.dst, component), [runner(tuple(x)) for x in zip(*values, strict=True)], write_mask)
      continue
    if inst.name == 'shrg':
      repeated = inst.repeat_srcs + (False,) * (3 - len(inst.repeat_srcs))
      source_mask = 0xffff if inst.source_half else 0xffffffff
      dest_mask = 0xffff if inst.dst[0].startswith('h') else 0xffffffff
      for component in range(inst.repeat + 1):
        srcs = tuple(_source_offset(src, component) if repeat else src
                     for src, repeat in zip(inst.srcs, repeated, strict=True))
        shifts, values, others = (_source_values(regs, src, lanes, inst.source_half) for src in srcs)
        result = [(((value & source_mask) >> (shift & 31)) | other) & dest_mask
                  for shift, value, other in zip(shifts, values, others, strict=True)]
        _write(regs, _reg_offset(inst.dst, component), result, write_mask)
      continue
    if inst.name == 'add.u':
      mask = 0xffff if inst.dst[0].startswith('h') else 0xffffffff
      for component in range(inst.repeat + 1):
        srcs = tuple(_source_offset(src, component) if repeated else
                     src + component if repeated else src
                     for src, repeated in zip(inst.srcs, inst.repeat_srcs, strict=True))
        lhs = _values(regs, srcs[0], lanes)
        rhs = _values(regs, srcs[1], lanes)
        _write(regs, _reg_offset(inst.dst, component), [(x + y) & mask for x, y in zip(lhs, rhs, strict=True)], write_mask)
      continue
    if inst.name in ('cmps.u', 'cmps.s'):
      lanes = len(next(iter(regs.values())))
      unsigned_mask = 0xffff if inst.source_half else 0xffffffff
      for component in range(inst.repeat + 1):
        srcs = tuple(_source_offset(src, component) if repeated else src
                     for src, repeated in zip(inst.srcs, inst.repeat_srcs, strict=True))
        lhs, rhs = (_values(regs, src, lanes) for src in srcs)
        signed = inst.name == 'cmps.s'
        _write(regs, _reg_offset(inst.dst, component), [int(_compare(_signed(x, inst.source_half) if signed else x & unsigned_mask,
          _signed(y, inst.source_half) if signed else y & unsigned_mask, inst.condition)) for x, y in zip(lhs, rhs, strict=True)], write_mask)
      continue
    if inst.name == 'absneg.s':
      mask = 0xffff if inst.dst[0].startswith('h') else 0xffffffff
      for component in range(inst.repeat + 1):
        src = _source_offset(inst.srcs[0], component) if inst.repeat_srcs[0] else inst.srcs[0]
        mod = inst.src_mods[0]
        vals = _values(regs, src, lanes)
        result = [(-abs(_signed(x, inst.source_half)) if mod == 3 else abs(_signed(x, inst.source_half)) if mod == 2
                   else -_signed(x, inst.source_half) if mod == 1 else _signed(x, inst.source_half)) & mask
                  for x in vals]
        _write(regs, _reg_offset(inst.dst, component), result, write_mask)
      continue
    if inst.name == 'sel.b32':
      lanes = len(next(iter(regs.values())))
      for component in range(inst.repeat + 1):
        srcs = tuple(_source_offset(src, component) if repeated else src
                     for src, repeated in zip(inst.srcs, inst.repeat_srcs, strict=True))
        yes, cond, no = (_values(regs, src, lanes) for src in srcs)
        _write(regs, _reg_offset(inst.dst, component), [x if c else y for x, c, y in zip(yes, cond, no, strict=True)], write_mask)
      continue
    if inst.name in {'shrm', 'shlm', 'shlg', 'andg'}:
      repeated = inst.repeat_srcs + (False,) * (3 - len(inst.repeat_srcs))
      mask = 0xffff if inst.dst[0].startswith('h') else 0xffffffff
      for component in range(inst.repeat + 1):
        srcs = tuple(_source_offset(src, component) if repeat else src
                     for src, repeat in zip(inst.srcs, repeated, strict=True))
        first, second, third = (_values(regs, src, lanes) for src in srcs)
        result = []
        for a, b, c in zip(first, second, third, strict=True):
          if inst.name == 'shrm': value = ((b & 0xffffffff) >> (a & 31)) & c
          elif inst.name == 'shlm': value = (b << (a & 31)) & c
          elif inst.name == 'shlg': value = (b << (a & 31)) | c
          else: value = (b & a) | c
          result.append(value & mask)
        _write(regs, _reg_offset(inst.dst, component), result, write_mask)
      continue
    if inst.name.startswith(('mad.', 'madsh.', 'sel.', 'sad.')):
      lanes = len(next(iter(regs.values())))
      repeated = inst.repeat_srcs + (False,) * (3 - len(inst.repeat_srcs))
      half = inst.name.endswith('16')
      for component in range(inst.repeat + 1):
        srcs = tuple(_source_offset(src, component) if repeat else src
                     for src, repeat in zip(inst.srcs, repeated, strict=True))
        values = [_float_values(regs, src, lanes, half) if inst.name.startswith('mad.f') else _values(regs, src, lanes) for src in srcs]
        result = []
        for raw in zip(*values, strict=True):
          if inst.name.startswith('mad.f'):
            val = _mod_float(raw[0], inst.src_mods[0], half) * _mod_float(raw[1], inst.src_mods[1], half) + \
              _mod_float(raw[2], inst.src_mods[2], half)
            result.append(_float_bits(min(1.0, max(0.0, val)) if inst.sat else val, half))
          elif inst.name.startswith('sel.'):
            result.append(raw[0] if raw[1] else raw[2])
          elif inst.name.startswith('sad.'):
            result.append((abs(_s32(raw[0]) - _s32(raw[1])) + raw[2]) & 0xffffffff)
          elif inst.name == 'madsh.m16':
            result.append(((((raw[0] & 0xffff) * ((raw[1] >> 16) & 0xffff)) << 16) + raw[2]) & 0xffffffff)
          else:
            signed = '.s' in inst.name
            bits = 16 if '16' in inst.name else 24
            mask, sign = (1 << bits) - 1, 1 << (bits - 1)
            a, b = raw[0] & mask, raw[1] & mask
            if signed: a, b = a - (1 << bits) if a & sign else a, b - (1 << bits) if b & sign else b
            result.append((a * b + raw[2]) & 0xffffffff)
        _write(regs, _reg_offset(inst.dst, component), result, write_mask)
      continue
    if inst.name in {'swz', 'gat', 'sct'}:
      source_values = [_values(regs, src, lanes).copy() for src in inst.srcs]
      for dst, values in zip(inst.dst, source_values, strict=True):
        converted = [_convert(value, inst.types[0], inst.types[1], inst.rounding) for value in values]
        _write(regs, dst, converted, write_mask)
      continue
    if inst.name == 'mov':
      lanes = len(next(iter(regs.values())))
      for component in range(inst.repeat + 1):
        src = _source_offset(inst.srcs[0], component) if inst.repeat_srcs[0] else inst.srcs[0]
        converted = [_convert(x, inst.types[0], inst.types[1], inst.rounding)
                     for x in _source_values(regs, src, lanes, inst.source_half)]
        _write_destination(regs, _source_offset(inst.dst, component), converted, write_mask)
      continue
    if inst.name in {'ldl', 'ldp', 'stl', 'stp'}:
      itemsize, lanes = _itemsize(inst.types[0]), len(next(iter(regs.values())))
      if inst.name.endswith('l'):
        if shared is None: raise RuntimeError(f'IR3 {inst.name} has no backing memory')
        if isinstance(shared, list):
          if len(shared) != lanes or any(not isinstance(memory, bytearray) for memory in shared):
            raise RuntimeError(f'IR3 {inst.name} requires {lanes} shared-memory lane mappings')
          memories = shared
        else: memories = [shared] * lanes
      else: memories = _private_lanes(private, lanes, inst.name)
      if inst.name.startswith('ld'):
        address_reg, offset, size = inst.srcs
        addresses = _values(regs, address_reg, lanes)
        for component in range(size):
          out = []
          for lane, address in enumerate(addresses):
            pos = address + offset + component * itemsize
            if not write_mask[lane]:
              out.append(0)
              continue
            lane_memory = memories[lane]
            if pos < 0 or pos + itemsize > len(lane_memory): raise RuntimeError(f'IR3 {inst.name} out of bounds at {pos:#x}')
            out.append(int.from_bytes(lane_memory[pos:pos + itemsize], 'little'))
          _write(regs, _reg_offset(inst.dst, component), out, write_mask)
      else:
        address_reg, value_reg, offset, size = inst.srcs
        addresses = _values(regs, address_reg, lanes)
        for component in range(size):
          values = _values(regs, _reg_offset(value_reg, component), lanes)
          for lane, (address, value) in enumerate(zip(addresses, values, strict=True)):
            if not write_mask[lane]: continue
            pos = address + offset + component * itemsize
            lane_memory = memories[lane]
            if pos < 0 or pos + itemsize > len(lane_memory): raise RuntimeError(f'IR3 {inst.name} out of bounds at {pos:#x}')
            lane_memory[pos:pos + itemsize] = (value & ((1 << (itemsize * 8)) - 1)).to_bytes(itemsize, 'little')
      continue
    if inst.name == 'stib':
      coord_reg, value_reg, offset, resource_index, dimensions, components = inst.srcs
      if dimensions != 2 or not 1 <= components <= 4: raise NotImplementedError('unsupported IR3 image store shape')
      if resource_index >= len(ibos): raise RuntimeError(f'IR3 image store references missing IBO {resource_index}')
      image = ibos[resource_index]
      xs, ys = _values(regs, coord_reg, lanes), _values(regs, _next_reg(coord_reg), lanes)
      for lane, (x, y) in enumerate(zip(xs, ys, strict=True)):
        x, y = _s32(x), _s32(y)
        if not write_mask[lane] or not (0 <= x < image['width'] and 0 <= y < image['height']): continue
        address = image['address'] + y * image['pitch'] + x * components * image['itemsize'] + offset
        _check_access(check_range, address, components * image['itemsize'], inst.name, inst_pc)
        for component in range(components):
          value = _values(regs, _reg_offset(value_reg, component), lanes)[lane]
          if image['encoded_itemsize'] == 2:
            numeric = _float(value, inst.types[0] == mesa.TYPE_F16)
            value = _float_bits(_float(_float_bits(numeric, True), True))
          elif inst.types[0] == mesa.TYPE_F16: value = _float_bits(_float(value, True))
          ctypes.memmove(address + component * image['itemsize'], value.to_bytes(image['itemsize'], 'little'), image['itemsize'])
      continue
    if inst.name == 'isam':
      coord_reg, _sampler_index, texture_index, dimensions, write_components = inst.srcs
      if texture_index >= len(textures): raise RuntimeError(f'IR3 image sample references missing texture {texture_index}')
      image = textures[texture_index]
      coordinates = [_values(regs, _reg_offset(coord_reg, component), lanes) for component in range(dimensions)]
      selected = [component for component in range(4) if write_components & (1 << component)]
      outputs: list[list[int]] = [[] for _ in selected]
      for lane, raw_coords in enumerate(zip(*coordinates, strict=True)):
        coords = tuple(_s32(value) for value in raw_coords)
        if dimensions == 1:
          pixel = coords[0]
          x, y = pixel % image['width'], pixel // image['width']
        else: x, y = coords
        valid = write_mask[lane] and 0 <= x < image['width'] and 0 <= y < image['height']
        address = image['address'] + y * image['pitch'] + x * 4 * image['itemsize'] if valid else 0
        if valid: _check_access(check_range, address, 4 * image['itemsize'], inst.name, inst_pc)
        for output, component in zip(outputs, selected, strict=True):
          if not valid: output.append(0)
          else:
            value = int.from_bytes(ctypes.string_at(address + component * image['itemsize'], image['itemsize']), 'little')
            if inst.types[1] == mesa.TYPE_F16: value = _float_bits(_float(value), True)
            output.append(value)
      for offset, output in enumerate(outputs): _write(regs, _reg_offset(inst.dst, offset), output, write_mask)
      continue
    alu_ops = {'add.f', 'mul.f', 'min.f', 'max.f', 'cmps.f', 'cmpv.f', 'sign.f', 'absneg.f', 'floor.f', 'ceil.f', 'rndne.f',
      'rndaz.f', 'trunc.f', 'add.u', 'add.s', 'sub.u', 'sub.s', 'cmps.u', 'cmps.s', 'cmpv.u', 'cmpv.s', 'min.u', 'min.s', 'max.u',
      'max.s', 'absneg.s', 'and.b', 'or.b', 'xor.b', 'not.b', 'mul.u24', 'mul.s24', 'mull.u', 'bfrev.b', 'clz.s', 'clz.b', 'cbits.b',
      'shl.b', 'shr.b', 'ashr.b', 'getbit.b', 'rcp', 'rsq', 'log2', 'exp2', 'sin', 'cos', 'sqrt', 'hrsq', 'hlog2', 'hexp2'}
    if inst.name in alu_ops:
      lanes = len(next(iter(regs.values())))
      repeated = inst.repeat_srcs + (False,) * (len(inst.srcs) - len(inst.repeat_srcs))
      modifiers = inst.src_mods + (0,) * (len(inst.srcs) - len(inst.src_mods))
      half = inst.dst[0].startswith('h')
      runner = _alu_runner(inst.name, inst.condition, modifiers, inst.source_half, half)
      for component in range(inst.repeat + 1):
        srcs = tuple(_source_offset(src, component) if repeat else src
                     for src, repeat in zip(inst.srcs, repeated, strict=True))
        float_op = inst.name.endswith('.f') or inst.name in {'rcp', 'rsq', 'log2', 'exp2', 'sin', 'cos', 'sqrt', 'hrsq', 'hlog2', 'hexp2'}
        values = [_float_values(regs, src, lanes, inst.source_half) if float_op else _values(regs, src, lanes) for src in srcs]
        result = [runner(tuple(x)) for x in zip(*values, strict=True)]
        if inst.sat: result = [_float_bits(min(1.0, max(0.0, _float(x, half))), half) for x in result]
        _write(regs, _reg_offset(inst.dst, component), result, write_mask)
      continue
    if inst.name == 'cov.u16s32':
      _write(regs, inst.dst, [value & 0xffff for value in _values(regs, inst.srcs[0], lanes)], write_mask)
      continue
    if inst.name == 'mov.u32u32':
      src = inst.srcs[0]
      lanes = len(next(iter(regs.values())))
      _write(regs, inst.dst, _values(regs, src, lanes).copy(), write_mask)
      continue
    if inst.name.startswith('atomic.g.'):
      if len(inst.srcs) not in (2, 3): raise NotImplementedError(f'unsupported IR3 atomic operands for {inst.name}')
      itemsize = _itemsize(inst.types[0])
      if itemsize != 4: raise NotImplementedError(f'unsupported {itemsize * 8}-bit IR3 atomic {inst.name}')
      address_reg, value_reg, *compare_reg = inst.srcs
      addresses = [low | high << 32 for low, high in
                   zip(_values(regs, address_reg, lanes), _values(regs, _next_reg(address_reg), lanes), strict=True)]
      values = _values(regs, value_reg, lanes)
      compares = _values(regs, compare_reg[0], lanes) if compare_reg else [0] * lanes
      old_values = [0] * lanes
      operation = inst.name.removeprefix('atomic.g.')
      for lane, (address, value, compare) in enumerate(zip(addresses, values, compares, strict=True)):
        if not write_mask[lane]: continue
        _check_access(check_range, address, itemsize, inst.name, inst_pc)
        old = ctypes.c_uint32.from_address(address).value
        if operation == 'add': new = old + value
        elif operation == 'sub': new = old - value
        elif operation in {'xchg', 'exchange'}: new = value
        elif operation in {'cmpxchg', 'cas'}: new = value if old == compare else old
        elif operation == 'and': new = old & value
        elif operation == 'or': new = old | value
        elif operation == 'xor': new = old ^ value
        elif operation == 'min': new = min(_s32(old), _s32(value))
        elif operation == 'max': new = max(_s32(old), _s32(value))
        elif operation == 'umin': new = min(old, value)
        elif operation == 'umax': new = max(old, value)
        else: raise NotImplementedError(f'unsupported IR3 global atomic {operation}')
        ctypes.c_uint32.from_address(address).value = new & 0xffffffff
        old_values[lane] = old
      if inst.dst is not None: _write(regs, inst.dst, old_values, write_mask)
      continue
    if inst.name == 'ldg':
      address_reg, offset, size = inst.srcs
      if not 1 <= size <= 4: raise NotImplementedError(f'unsupported ldg size {size}')
      if check_range is None and any(write_mask): _check_access(check_range, 0, 0, inst.name, inst_pc)
      itemsize = _itemsize(inst.types[0])
      addresses = [low | high << 32 for low, high in
                   zip(_values(regs, address_reg, lanes), _values(regs, _next_reg(address_reg), lanes), strict=True)]
      span = size * itemsize
      valid_lanes = write_mask.copy()
      # Lanes usually access one mapped range together; validate the aggregate span once and fall
      # back to the exact per-lane checks when the span straddles mappings or faults.
      if predication is None and any(write_mask):
        active = [addresses[lane] + offset for lane in range(lanes) if write_mask[lane]]
        try:
          check_range(min(active), max(active) - min(active) + span)
          checked = True
        except Exception: checked = False
        if not checked:
          for lane, address in enumerate(addresses):
            if not write_mask[lane]: continue
            try: _check_access(check_range, address + offset, span, inst.name, inst_pc)
            except Exception as exc:
              raise RuntimeError(f'IR3 {inst.name} memory fault at PC {inst_pc}, lane {lane}, address={address:#x}') from exc
      else:
        for lane, address in enumerate(addresses):
          if not write_mask[lane]: continue
          try: _check_access(check_range, address + offset, span, inst.name, inst_pc)
          except Exception as exc:
            if predication is not None:
              valid_lanes[lane] = False
              continue
            raise RuntimeError(f'IR3 {inst.name} memory fault at PC {inst_pc}, lane {lane}, address={address:#x}') from exc
      fmt = '<%d%s' % (size, {1: 'B', 2: 'H', 4: 'I'}[itemsize])
      loaded_components: list[list[int]] = [[] for _ in range(size)]
      lane = 0
      while lane < lanes:
        address = addresses[lane] + offset
        if not valid_lanes[lane]:
          for component in range(size): loaded_components[component].append(0)
          lane += 1
          continue
        # Batch maximal runs of lanes accessing consecutive memory with one read.
        run = lane + 1
        if span >= 4:
          while run < lanes and valid_lanes[run] and addresses[run] + offset == addresses[run - 1] + offset + span: run += 1
        if run > lane + 1:
          data = struct.unpack(f'<{(run - lane) * size}{fmt[2]}', ctypes.string_at(address, (run - lane) * span))
          for index in range(run - lane):
            for component in range(size): loaded_components[component].append(data[index * size + component])
        else:
          data = struct.unpack(fmt, ctypes.string_at(address, span))
          for component in range(size): loaded_components[component].append(data[component])
        lane = run
      for component in range(size): _write(regs, _reg_offset(inst.dst, component), loaded_components[component], write_mask)
      continue
    if inst.name == 'ldg.a':
      address_reg, index_reg, shift, size = inst.srcs
      if not 1 <= size <= 4: raise NotImplementedError(f'unsupported ldg.a size {size}')
      itemsize = _itemsize(inst.types[0])
      addresses = [low | high << 32 for low, high in
                   zip(_values(regs, address_reg, lanes), _values(regs, _next_reg(address_reg), lanes), strict=True)]
      indices = _values(regs, index_reg, lanes)
      targets = [address + (_s32(index) << shift) for address, index in zip(addresses, indices, strict=True)]
      _validate_targets(targets, write_mask, itemsize, check_range, inst.name, inst_pc)
      for component in range(size):
        output = _read_targets([target + component * itemsize for target in targets], write_mask, itemsize)
        _write(regs, _reg_offset(inst.dst, component), output, write_mask)
      continue
    if inst.name == 'stg':
      address_reg, value_reg, offset, size = inst.srcs
      if not 1 <= size <= 4: raise NotImplementedError(f'unsupported stg size {size}')
      if check_range is None and any(write_mask): _check_access(check_range, 0, 0, inst.name, inst_pc)
      itemsize = _itemsize(inst.types[0])
      addresses = [low | high << 32 for low, high in
                   zip(_values(regs, address_reg, lanes), _values(regs, _next_reg(address_reg), lanes), strict=True)]
      span = size * itemsize
      if any(write_mask):
        active = [addresses[lane] + offset for lane in range(lanes) if write_mask[lane]]
        try:
          check_range(min(active), max(active) - min(active) + span)
          checked = True
        except Exception: checked = False
        if not checked:
          for lane, address in enumerate(addresses):
            if not write_mask[lane]: continue
            try: _check_access(check_range, address + offset, span, inst.name, inst_pc)
            except Exception as exc:
              raise RuntimeError(f'IR3 {inst.name} memory fault at PC {inst_pc}, lane {lane}, address={address:#x}, '
                f'c0={[_values(regs, ("c", 0, i), lanes)[lane] for i in range(4)]}, '
                f'r0={[_values(regs, ("r", 0, i), lanes)[lane] for i in range(4)]}') from exc
      columns = [_values(regs, _reg_offset(value_reg, component), lanes) for component in range(size)]
      unit_mask = (1 << (itemsize * 8)) - 1
      lane = 0
      while lane < lanes:
        if not write_mask[lane]:
          lane += 1
          continue
        address = addresses[lane] + offset
        # Consecutive lanes writing adjacent, disjoint slots go through one packed store.
        run = lane + 1
        while run < lanes and write_mask[run] and addresses[run] + offset == addresses[run - 1] + offset + span: run += 1
        if run > lane + 1 and itemsize == 4:
          payload = b''.join((column[index] & unit_mask).to_bytes(4, 'little') for index in range(lane, run) for column in columns)
          ctypes.memmove(address, payload, (run - lane) * span)
        else:
          for index in range(lane, run):
            ctypes.memmove(addresses[index] + offset,
              b''.join((column[index] & unit_mask).to_bytes(itemsize, 'little') for column in columns), span)
        lane = run
      continue
    if inst.name == 'stg.a':
      address_reg, index_reg, shift, value_reg, size = inst.srcs
      if not 1 <= size <= 4: raise NotImplementedError(f'unsupported stg.a size {size}')
      itemsize = _itemsize(inst.types[0])
      addresses = [low | high << 32 for low, high in
                   zip(_values(regs, address_reg, lanes), _values(regs, _next_reg(address_reg), lanes), strict=True)]
      indices = _values(regs, index_reg, lanes)
      targets = [address + (_s32(index) << shift) for address, index in zip(addresses, indices, strict=True)]
      _validate_targets(targets, write_mask, itemsize, check_range, inst.name, inst_pc)
      for component in range(size):
        _write_targets([target + component * itemsize for target in targets], write_mask,
                       _values(regs, _reg_offset(value_reg, component), lanes), itemsize)
      continue
    raise NotImplementedError(f'unsupported IR3 execution {inst.name}')
  if resume_state is not None: resume_state.clear()
  return None

def execute_dispatch(code, grid_size, local_size, local_id_register, initial_regs=None, check_range=None, workgroup_id_register=0xfc,
                     textures=(), ibos=(), global_id_register=0xfc, linear_group_register=0xfc, local_id_order=(0, 1, 2)):
  """Run a mock A630 dispatch with workgroup scheduling in :mod:`dispatch`."""
  return _execute_dispatch(execute_ir3, code, grid_size, local_size, local_id_register, initial_regs, check_range,
                           workgroup_id_register, textures, ibos, global_id_register, linear_group_register, local_id_order,
                           has_loop=_has_native_loop)
