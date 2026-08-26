from __future__ import annotations
import ctypes, time
from dataclasses import dataclass, field
from typing import cast

from tinygrad.runtime.autogen import mesa
from test.mockgpu.gpu import VirtGPU
from test.mockgpu.qcom.emu import execute_dispatch


_U32_MAX, _U64_LIMIT = (1 << 32) - 1, 1 << 64


def _range_end(address:int, size:int, name:str) -> int:
  if address < 0 or address >= _U64_LIMIT or size <= 0 or size > _U64_LIMIT - address:
    raise ValueError(f"{name} overflows the 64-bit address space")
  return address + size


def _field(value:int, name:str) -> int:
  return (value & getattr(mesa, f"{name}__MASK")) >> getattr(mesa, f"{name}__SHIFT")


def _u64(lo:int, hi:int) -> int: return lo | (hi << 32)


def _parity(value:int) -> int:
  for shift in range(4, 1, -1): value ^= value >> (1 << shift)
  return (~0x6996 >> (value & 0xf)) & 1


class A630Wait(RuntimeError):
  """The command stream is paused until a polled memory value changes."""


@dataclass
class A630ContextState:
  regs: dict[int, int]
  states: dict[tuple[int, int], tuple[int, int]]
  last_regs: dict[tuple[str, int, int], list[int]] = field(default_factory=dict)


@dataclass
class CommandStream:
  words: list[int]
  timestamp: int
  context_id: int
  engine: str = 'compute'
  cursor: int = 0
  state_before: tuple[dict[int, int], dict[tuple[int, int], tuple[int, int]], dict[tuple[str, int, int], list[int]]]|None = None
  memory_undo: list[tuple[int, bytes]] = field(default_factory=list)
  created_context_state: bool = False


class A630GPU(VirtGPU):
  def __init__(self, gpuid:int):
    super().__init__(gpuid)
    self.regs:dict[int, int] = {}
    self.mapped_ranges:dict[tuple[int, int], int] = {}
    self.mapped_pages:dict[int, set[tuple[int, int]]] = {}
    self.states:dict[tuple[int, int], tuple[int, int]] = {}
    self.last_regs:dict[tuple[str, int, int], list[int]] = {}
    self.pending:list[CommandStream] = []
    self.context_states:dict[int, A630ContextState] = {}
    self.completed_timestamps:dict[int, int] = {}
    self.completed_submissions:dict[int, set[int]] = {}
    self.completed_timestamp = 0
    self._active_stream:CommandStream|None = None

  def map_range(self, vaddr:int, size:int):
    _range_end(vaddr, size, "A630 mapping")
    key = (vaddr, size)
    self.mapped_ranges[key] = self.mapped_ranges.get(key, 0) + 1
    if self.mapped_ranges[key] == 1:
      for page in range(vaddr >> 12, ((vaddr + size - 1) >> 12) + 1): self.mapped_pages.setdefault(page, set()).add(key)
  def unmap_range(self, vaddr:int, size:int):
    _range_end(vaddr, size, "A630 mapping")
    key = (vaddr, size)
    if (count:=self.mapped_ranges.get(key, 0)) == 0: raise RuntimeError(f"A630 range {vaddr:#x}..{vaddr + size:#x} is not mapped")
    if count == 1:
      del self.mapped_ranges[key]
      for page in range(vaddr >> 12, ((vaddr + size - 1) >> 12) + 1):
        ranges = self.mapped_pages[page]
        ranges.remove(key)
        if not ranges: del self.mapped_pages[page]
    else: self.mapped_ranges[key] = count - 1

  def _mapped_bounds(self, address:int) -> tuple[int, int]:
    if containing := [(start, start + length) for start, length in self.mapped_pages.get(address >> 12, ())
                      if start <= address < start + length]:
      return max(containing, key=lambda bounds: bounds[1] - address)
    nearby = sorted((start, start + length) for start, length in self.mapped_ranges if abs(start - address) < 0x100000)[:8]
    raise RuntimeError(f"QCOM kernel referenced unmapped address {address:#x}; nearby mappings={nearby}")

  def _mapped_size(self, address:int) -> int:
    return self._mapped_bounds(address)[1] - address

  def _validate_memory(self, address:int, size:int):
    try: end = _range_end(address, size, "A630 memory range")
    except ValueError as exc: raise RuntimeError(str(exc)) from None
    try: bounds = self._mapped_bounds(address)
    except RuntimeError: bounds = (address, address)
    if end > bounds[1]:
      raise RuntimeError(f"A630 referenced unmapped range {address:#x}..{end:#x}")
    # Native mock executors may use these bounds to gate every dynamic access. Existing callers
    # only depend on validation and intentionally ignore the return value.
    return bounds

  def execute(self, address:int, size:int):
    if size % 4: raise ValueError(f"A630 command buffer size must be dword aligned, got {size}")
    if address % 4: raise ValueError(f"A630 command buffer address must be dword aligned, got {address:#x}")
    self._validate_memory(address, size)
    self.execute_words(list((ctypes.c_uint32 * (size // 4)).from_address(address)))

  def execute_words(self, words:list[int]):
    self.validate_words(words)
    cursor = 0
    while cursor < len(words): cursor = self._execute_one(words, cursor)

  @staticmethod
  def _packet_count(words:list[int], cursor:int) -> tuple[int, int]:
    if cursor >= len(words): raise ValueError("truncated A630 command stream")
    header = words[cursor]
    if not isinstance(header, int) or not 0 <= header <= _U32_MAX:
      raise ValueError(f"invalid A630 dword at {cursor}")
    packet_type = header >> 28
    if packet_type == 4:
      count, register = header & 0x7f, (header >> 8) & 0x3ffff
      if header & 0x04000000: raise ValueError(f"invalid A630 type-4 packet reserved bits at dword {cursor}")
      if (header >> 7) & 1 != _parity(count) or (header >> 27) & 1 != _parity(register):
        raise ValueError(f"invalid A630 type-4 packet parity at dword {cursor}")
      if count == 0: raise ValueError(f"A630 type-4 packet has zero dword count at dword {cursor}")
    elif packet_type == 7:
      count, opcode = header & 0x3fff, (header >> 16) & 0x7f
      if header & 0x0f004000: raise ValueError(f"invalid A630 type-7 packet reserved bits at dword {cursor}")
      if (header >> 15) & 1 != _parity(count) or (header >> 23) & 1 != _parity(opcode):
        raise ValueError(f"invalid A630 type-7 packet parity at dword {cursor}")
    else: raise ValueError(f"unsupported A630 packet type {packet_type} at dword {cursor}")
    if cursor + 1 + count > len(words): raise ValueError(f"truncated A630 type-{packet_type} packet")
    return packet_type, count

  def validate_words(self, words:list[int]):
    for index, word in enumerate(words):
      if not isinstance(word, int) or not 0 <= word <= _U32_MAX: raise ValueError(f"invalid A630 dword at {index}")
    cursor = 0
    while cursor < len(words):
      packet_type, count = self._packet_count(words, cursor)
      if packet_type == 7:
        opcode = (words[cursor] >> 16) & 0x7f
        self._validate_type7(opcode, words[cursor + 1:cursor + count + 1])
      cursor += count + 1

  @staticmethod
  def _validate_type7(opcode:int, values:list[int]):
    exact_counts = {
      mesa.CP_WAIT_FOR_IDLE: 0, mesa.CP_WAIT_MEM_WRITES: 0, mesa.CP_SET_MARKER: 1,
      mesa.CP_REG_TO_MEM: 3, mesa.CP_WAIT_REG_MEM: 6, mesa.CP_EXEC_CS: 4, mesa.CP_MEMCPY: 5,
    }
    if opcode in exact_counts:
      if len(values) != (expected:=exact_counts[opcode]):
        raise ValueError(f"{mesa.enum_adreno_pm4_type3_packets[opcode]} expects {expected} dwords, got {len(values)}")
      if opcode not in {mesa.CP_SET_MARKER, mesa.CP_WAIT_REG_MEM, mesa.CP_REG_TO_MEM, mesa.CP_EXEC_CS, mesa.CP_MEMCPY}: return
    if opcode == mesa.CP_SET_MARKER:
      if values != [mesa.RM6_COMPUTE]: raise NotImplementedError(f"unsupported A630 marker control {values[0]:#x}")
      return
    if opcode == mesa.CP_WAIT_REG_MEM:
      function = _field(values[0], "CP_WAIT_REG_MEM_0_FUNCTION")
      poll = _field(values[0], "CP_WAIT_REG_MEM_0_POLL")
      if function not in {mesa.WRITE_GE, mesa.WRITE_EQ}: raise NotImplementedError(f"unsupported A630 wait function {function}")
      if poll != mesa.POLL_MEMORY: raise NotImplementedError(f"unsupported A630 wait poll mode {poll}")
      return
    if opcode == mesa.CP_REG_TO_MEM:
      reg, count = _field(values[0], "CP_REG_TO_MEM_0_REG"), _field(values[0], "CP_REG_TO_MEM_0_CNT")
      if reg != mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER or count != 2 or not values[0] & mesa.CP_REG_TO_MEM_0_64B or \
         values[0] & mesa.CP_REG_TO_MEM_0_ACCUMULATE:
        raise NotImplementedError(f"unsupported A630 register-to-memory control {values[0]:#x}")
      return
    if opcode == mesa.CP_EXEC_CS:
      if values[0] != 0: raise ValueError(f"CP_EXEC_CS expects a zero control dword, got {values[0]:#x}")
      return
    if opcode == mesa.CP_MEMCPY:
      if values[0] & 0x7fffffff == 0: raise ValueError('CP_MEMCPY has zero copy count')
      return
    if opcode == mesa.CP_EVENT_WRITE:
      if not values: raise ValueError("CP_EVENT_WRITE expects at least 1 dword")
      event = _field(values[0], "CP_EVENT_WRITE_0_EVENT")
      if event not in {mesa.CACHE_FLUSH_TS, mesa.CACHE_INVALIDATE}:
        raise NotImplementedError(f"unsupported A630 event {event} ({mesa.enum_vgt_event_type.get(event, 'unknown')})")
      expected = 4 if event == mesa.CACHE_FLUSH_TS else 1
      if len(values) != expected:
        event_name = "CACHE_FLUSH_TS" if event == mesa.CACHE_FLUSH_TS else "CACHE_INVALIDATE"
        raise ValueError(f"{event_name} expects {expected} dwords, got {len(values)}")
      return
    if opcode == mesa.CP_LOAD_STATE6_FRAG:
      if len(values) != 3: raise ValueError(f"CP_LOAD_STATE6_FRAG expects 3 dwords, got {len(values)}")
      state_source = _field(values[0], "CP_LOAD_STATE6_0_STATE_SRC")
      if state_source != mesa.SS6_INDIRECT: raise NotImplementedError(f"unsupported A630 state source {state_source}")
      state_type = _field(values[0], "CP_LOAD_STATE6_0_STATE_TYPE")
      state_block = _field(values[0], "CP_LOAD_STATE6_0_STATE_BLOCK")
      supported = {
        (mesa.ST_CONSTANTS, mesa.SB6_CS_SHADER), (mesa.ST_SHADER, mesa.SB6_CS_SHADER),
        (mesa.ST_SHADER, mesa.SB6_CS_TEX), (mesa.ST_CONSTANTS, mesa.SB6_CS_TEX),
        (mesa.ST6_UAV, mesa.SB6_CS_SHADER),
      }
      if (state_type, state_block) not in supported:
        raise NotImplementedError(f"unsupported A630 state load type {state_type} block {state_block}")
      return
    if opcode == mesa.CP_RUN_OPENCL:
      if values != [0]: raise ValueError(f"CP_RUN_OPENCL expects [0], got {values}")
      return
    raise NotImplementedError(f"unsupported A630 type-7 opcode {opcode:#x} ({mesa.enum_adreno_pm4_type3_packets.get(opcode, 'unknown')})")

  def submit(self, words:list[int], timestamp:int, context_id:int=0):
    if not words: raise ValueError("A630 submission has no commands")
    if not 1 <= timestamp <= _U32_MAX: raise ValueError(f"invalid A630 timestamp {timestamp}")
    self.validate_words(words)
    engine = 'copy' if any(opcode == mesa.CP_MEMCPY for _cursor, opcode, _values in self._iter_type7(words)) else 'compute'
    self.pending.append(CommandStream(words.copy(), timestamp, context_id, engine))

  @classmethod
  def _iter_type7(cls, words:list[int]):
    cursor = 0
    while cursor < len(words):
      packet_type, count = cls._packet_count(words, cursor)
      if packet_type == 7: yield cursor, (words[cursor] >> 16) & 0x7f, words[cursor + 1:cursor + count + 1]
      cursor += count + 1

  def is_timestamp_complete(self, context_id:int, timestamp:int) -> bool:
    return timestamp == 0 or self.completed_timestamps.get(context_id, 0) >= timestamp

  def drop_context(self, context_id:int):
    state = self.context_states.pop(context_id, None)
    self.pending[:] = [stream for stream in self.pending if stream.context_id != context_id]
    self.completed_timestamps.pop(context_id, None)
    self.completed_submissions.pop(context_id, None)
    self.completed_timestamp = max(self.completed_timestamps.values(), default=0)
    if state is not None and self.regs is state.regs:
      self.regs, self.states, self.last_regs = {}, {}, {}

  def _activate_context(self, context_id:int) -> A630ContextState:
    state = self.context_states.setdefault(context_id, A630ContextState({}, {}))
    self.regs, self.states, self.last_regs = state.regs, state.states, state.last_regs
    return state

  @staticmethod
  def _copy_last_regs(last_regs:dict[tuple[str, int, int], list[int]]) -> dict[tuple[str, int, int], list[int]]:
    return {key: values.copy() for key, values in last_regs.items()}

  def _snapshot_stream_state(self, stream:CommandStream, state:A630ContextState):
    if stream.state_before is None:
      stream.state_before = (state.regs.copy(), state.states.copy(), self._copy_last_regs(state.last_regs))

  def _rollback_stream(self, stream:CommandStream, state:A630ContextState):
    if stream.state_before is not None:
      regs, states, last_regs = stream.state_before
      state.regs.clear()
      state.regs.update(regs)
      state.states.clear()
      state.states.update(states)
      state.last_regs.clear()
      state.last_regs.update(last_regs)
    for address, old_value in reversed(stream.memory_undo):
      try: self._validate_memory(address, len(old_value))
      except RuntimeError: continue
      ctypes.memmove(address, old_value, len(old_value))

  @staticmethod
  def _finish_stream(stream:CommandStream):
    stream.state_before = None
    stream.memory_undo.clear()

  def _retire_submission(self, stream:CommandStream):
    completed = self.completed_submissions.setdefault(stream.context_id, set())
    completed.add(stream.timestamp)
    contiguous = self.completed_timestamps.get(stream.context_id, 0)
    while contiguous + 1 in completed:
      contiguous += 1
      completed.remove(contiguous)
    self.completed_timestamps[stream.context_id] = contiguous
    self.completed_timestamp = max(self.completed_timestamps.values(), default=0)

  def _write_memory(self, address:int, value:bytes):
    self._validate_memory(address, len(value))
    if self._active_stream is not None: self._active_stream.memory_undo.append((address, ctypes.string_at(address, len(value))))
    ctypes.memmove(address, value, len(value))

  def progress(self) -> int:
    while self.pending:
      made_progress, blocked_queues, index = False, set(), 0
      while index < len(self.pending):
        stream = self.pending[index]
        if (stream.context_id, stream.engine) in blocked_queues:
          index += 1
          continue
        had_state = stream.context_id in self.context_states
        state = self._activate_context(stream.context_id)
        if stream.state_before is None: stream.created_context_state = not had_state
        self._snapshot_stream_state(stream, state)
        initial_cursor, waiting = stream.cursor, False
        self._active_stream = stream
        try:
          while stream.cursor < len(stream.words): stream.cursor = self._execute_one(stream.words, stream.cursor)
        except A630Wait: waiting = True
        except Exception:
          self._rollback_stream(stream, state)
          self.pending.pop(index)
          self._retire_submission(stream)
          if stream.created_context_state:
            self.context_states.pop(stream.context_id, None)
            self.regs, self.states, self.last_regs = {}, {}, {}
          raise
        finally: self._active_stream = None
        if waiting:
          made_progress |= stream.cursor != initial_cursor
          blocked_queues.add((stream.context_id, stream.engine))
          index += 1
          continue
        self._retire_submission(stream)
        self._finish_stream(stream)
        self.pending.pop(index)
        made_progress = True
      if not made_progress: break
    return self.completed_timestamp

  def _execute_one(self, words:list[int], cursor:int) -> int:
    header = words[cursor]
    start, cursor = cursor, cursor + 1
    packet_type, count = self._packet_count(words, start)
    if packet_type == 4:
      reg = (header >> 8) & 0x3ffff
      for offset, value in enumerate(words[cursor:cursor + count]): self.regs[reg + offset] = value
    elif packet_type == 7:
      opcode = (header >> 16) & 0x7f
      try: self._execute_type7(opcode, words[cursor:cursor + count])
      except A630Wait: raise
    return cursor + count

  def _execute_type7(self, opcode:int, values:list[int]):
    self._validate_type7(opcode, values)
    if opcode in {mesa.CP_WAIT_FOR_IDLE, mesa.CP_WAIT_MEM_WRITES, mesa.CP_SET_MARKER}: return
    if opcode == mesa.CP_EVENT_WRITE:
      if values and _field(values[0], "CP_EVENT_WRITE_0_EVENT") == mesa.CACHE_FLUSH_TS:
        address = _u64(values[1], values[2])
        self._write_memory(address, values[3].to_bytes(4, "little"))
      return
    if opcode == mesa.CP_REG_TO_MEM:
      destination = _u64(values[1], values[2])
      self._write_memory(destination, int(time.perf_counter() * 19_200_000).to_bytes(8, "little"))
      return
    if opcode == mesa.CP_WAIT_REG_MEM:
      address, reference, mask = _u64(values[1], values[2]), values[3], values[4]
      self._validate_memory(address, 4)
      actual = ctypes.c_uint32.from_address(address).value
      function = _field(values[0], "CP_WAIT_REG_MEM_0_FUNCTION")
      if function == mesa.WRITE_GE: satisfied = (actual & mask) >= (reference & mask)
      elif function == mesa.WRITE_EQ: satisfied = (actual & mask) == (reference & mask)
      else: raise NotImplementedError(f"unsupported A630 wait function {function}")
      if not satisfied: raise A630Wait(f"unsatisfied A630 memory wait at {address:#x}: {actual:#x} < {reference:#x}")
      return
    if opcode == mesa.CP_MEMCPY:
      count, source, destination = values[0], _u64(values[1], values[2]), _u64(values[3], values[4])
      size = count & 0x7fffffff if count & 1 << 31 else count * 4
      self._validate_memory(source, size)
      self._write_memory(destination, ctypes.string_at(source, size))
      return
    if opcode == mesa.CP_LOAD_STATE6_FRAG:
      state_type = _field(values[0], "CP_LOAD_STATE6_0_STATE_TYPE")
      state_block = _field(values[0], "CP_LOAD_STATE6_0_STATE_BLOCK")
      units = _field(values[0], "CP_LOAD_STATE6_0_NUM_UNIT")
      self.states[(state_block, state_type)] = (_u64(values[1], values[2]), units)
      return
    if opcode == mesa.CP_EXEC_CS:
      self._run_kernel((values[1], values[2], values[3]))
      return
    if opcode == mesa.CP_RUN_OPENCL:
      self._run_kernel(cast(tuple[int, int, int], tuple(self.regs.get(reg, 1) for reg in
                           (mesa.REG_A6XX_SP_CS_KERNEL_GROUP_X, mesa.REG_A6XX_SP_CS_KERNEL_GROUP_Y, mesa.REG_A6XX_SP_CS_KERNEL_GROUP_Z))))
      return

  def _run_kernel(self, global_size:tuple[int, int, int]):
    constants = self.states.get((mesa.SB6_CS_SHADER, mesa.ST_CONSTANTS))
    if constants is None: raise RuntimeError("A630 kernel launch has no constant-buffer address")
    constants_addr, constant_units = constants
    if constant_units == 0: raise RuntimeError("A630 kernel launch has an empty constant buffer")
    try:
      shader_addr = _u64(self.regs[mesa.REG_A6XX_SP_CS_BASE], self.regs[mesa.REG_A6XX_SP_CS_BASE + 1]) + \
        self.regs.get(mesa.REG_A6XX_SP_CS_PROGRAM_COUNTER_OFFSET, 0)
      shader_size = self.regs[mesa.REG_A6XX_SP_CS_INSTR_SIZE] * 128
      ndrange = self.regs[mesa.REG_A6XX_SP_CS_NDRANGE_0]
      mode = self.regs.get(mesa.REG_A6XX_SP_MODE_CNTL, 0)
    except KeyError as exc: raise RuntimeError(f"A630 kernel launch is missing register {exc.args[0]:#x}") from None
    if shader_size == 0: raise RuntimeError("A630 kernel launch has no shader instructions")
    self._validate_memory(shader_addr, shader_size)
    # CP_LOAD_STATE6 advertises the constant address space, but kernels only fault for
    # constants they actually read. Graph argument buffers can end inside a mapping,
    # so seed the mapped tail and let absent constant registers read as zero.
    self._validate_memory(constants_addr, 4)
    constant_bytes = min(constant_units * 16, self._mapped_size(constants_addr)) & ~3
    local_size = cast(tuple[int, int, int], tuple(_field(ndrange, f"A6XX_SP_CS_NDRANGE_0_LOCALSIZE{axis}") + 1 for axis in "XYZ"))
    opencl = _field(mode, 'A6XX_SP_MODE_CNTL_ISAMMODE') == mesa.ISAMMODE_CL
    config = self.regs.get(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0, 0)
    local_reg = 0 if opencl else _field(config, "A6XX_SP_CS_CONST_CONFIG_0_LOCALIDREGID")
    workgroup_reg = 0xfc if opencl else _field(config, "A6XX_SP_CS_CONST_CONFIG_0_WGIDCONSTID")
    if local_reg != 0xfc and local_reg % 4: raise ValueError(f"unaligned local ID register {local_reg:#x}")
    if workgroup_reg != 0xfc and workgroup_reg % 4: raise ValueError(f"unaligned workgroup ID register {workgroup_reg:#x}")
    words = (ctypes.c_uint32 * (constant_bytes // 4)).from_address(constants_addr)
    constant_words = tuple(words)
    def image_descriptors(block, state_type):
      state = self.states.get((block, state_type))
      if state is None: return []
      address, count = state
      self._validate_memory(address, count * 64)
      resources = []
      for index in range(count):
        descriptor = (ctypes.c_uint32 * 16).from_address(address + index * 64)
        fmt = _field(descriptor[0], 'A6XX_TEX_CONST_0_FMT')
        if fmt not in (mesa.FMT6_16_16_16_16_FLOAT, mesa.FMT6_32_32_32_32_FLOAT):
          raise NotImplementedError(f'unsupported A630 image format {fmt}')
        encoded_itemsize = 2 if fmt == mesa.FMT6_16_16_16_16_FLOAT else 4
        resources.append({'address':_u64(descriptor[4], descriptor[5]),
                          'width':_field(descriptor[1], 'A6XX_TEX_CONST_1_WIDTH'),
                          'height':_field(descriptor[1], 'A6XX_TEX_CONST_1_HEIGHT'),
                          'pitch':_field(descriptor[2], 'A6XX_TEX_CONST_2_PITCH') * (4 // encoded_itemsize),
                          'itemsize':4, 'encoded_itemsize':encoded_itemsize})
      return resources
    textures = image_descriptors(mesa.SB6_CS_TEX, mesa.ST_CONSTANTS)
    ibos = image_descriptors(mesa.SB6_CS_SHADER, mesa.ST6_UAV)
    self.last_regs.clear()
    self.last_regs.update(execute_dispatch(ctypes.string_at(shader_addr, shader_size), global_size, local_size,
      local_reg if local_reg == 0xfc else local_reg // 4, check_range=self._validate_memory,
      workgroup_id_register=workgroup_reg if workgroup_reg == 0xfc else workgroup_reg // 4, textures=textures, ibos=ibos,
      global_id_register=51 if opencl else 0xfc, linear_group_register=51 * 4 + 3 if opencl else 0xfc,
      constant_words=constant_words))
