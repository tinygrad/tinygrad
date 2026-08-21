from __future__ import annotations
import ctypes, time
from dataclasses import dataclass
from typing import cast

from tinygrad.runtime.autogen import mesa
from test.mockgpu.gpu import VirtGPU
from test.mockgpu.qcom.emu import find_program


def _field(value:int, name:str) -> int:
  return (value & getattr(mesa, f"{name}__MASK")) >> getattr(mesa, f"{name}__SHIFT")


def _u64(lo:int, hi:int) -> int: return lo | (hi << 32)


def _parity(value:int) -> int:
  for shift in range(4, 1, -1): value ^= value >> (1 << shift)
  return (~0x6996 >> (value & 0xf)) & 1


class A630Wait(RuntimeError):
  """The command stream is paused until a polled memory value changes."""


@dataclass
class CommandStream:
  words: list[int]
  timestamp: int
  context_id: int
  cursor: int = 0


@dataclass
class A630ContextState:
  regs: dict[int, int]
  constants_addr: int|None = None


class A630GPU(VirtGPU):
  def __init__(self, gpuid:int):
    super().__init__(gpuid)
    self.regs:dict[int, int] = {}
    self.mapped_ranges:dict[tuple[int, int], int] = {}
    self.constants_addr:int|None = None
    self.pending:list[CommandStream] = []
    self.context_states:dict[int, A630ContextState] = {}
    self.completed_timestamps:dict[int, int] = {}
    self.completed_timestamp = 0

  def map_range(self, vaddr:int, size:int):
    key = (vaddr, size)
    self.mapped_ranges[key] = self.mapped_ranges.get(key, 0) + 1
  def unmap_range(self, vaddr:int, size:int):
    key = (vaddr, size)
    if (count:=self.mapped_ranges.get(key, 0)) == 0: raise RuntimeError(f"A630 range {vaddr:#x}..{vaddr + size:#x} is not mapped")
    if count == 1: del self.mapped_ranges[key]
    else: self.mapped_ranges[key] = count - 1

  def _mapped_size(self, address:int) -> int:
    if containing := [start + length - address for start, length in self.mapped_ranges if start <= address < start + length]:
      return max(containing)
    nearby = sorted((start, start + length) for start, length in self.mapped_ranges if abs(start - address) < 0x100000)[:8]
    raise RuntimeError(f"QCOM kernel referenced unmapped address {address:#x}; nearby mappings={nearby}")

  def _validate_memory(self, address:int, size:int):
    try: available = self._mapped_size(address)
    except RuntimeError: available = 0
    if size > available:
      raise RuntimeError(f"A630 referenced unmapped range {address:#x}..{address + size:#x}")

  def execute(self, address:int, size:int):
    if size % 4: raise ValueError(f"A630 command buffer size must be dword aligned, got {size}")
    self._validate_memory(address, size)
    self.execute_words(list((ctypes.c_uint32 * (size // 4)).from_address(address)))

  def execute_words(self, words:list[int]):
    cursor = 0
    while cursor < len(words): cursor = self._execute_one(words, cursor)

  @staticmethod
  def _packet_count(words:list[int], cursor:int) -> tuple[int, int]:
    header, packet_type = words[cursor], words[cursor] >> 28
    if packet_type == 4:
      count, register = header & 0x7f, (header >> 8) & 0x3ffff
      if (header >> 7) & 1 != _parity(count) or (header >> 27) & 1 != _parity(register):
        raise ValueError(f"invalid A630 type-4 packet parity at dword {cursor}")
    elif packet_type == 7:
      count, opcode = header & 0x3fff, (header >> 16) & 0x7f
      if (header >> 15) & 1 != _parity(count) or (header >> 23) & 1 != _parity(opcode):
        raise ValueError(f"invalid A630 type-7 packet parity at dword {cursor}")
    else: raise ValueError(f"unsupported A630 packet type {packet_type} at dword {cursor}")
    if cursor + 1 + count > len(words): raise ValueError(f"truncated A630 type-{packet_type} packet")
    return packet_type, count

  def validate_words(self, words:list[int]):
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
      mesa.CP_REG_TO_MEM: 3, mesa.CP_WAIT_REG_MEM: 6, mesa.CP_EXEC_CS: 4,
    }
    if opcode in exact_counts:
      if len(values) != (expected:=exact_counts[opcode]):
        raise ValueError(f"{mesa.enum_adreno_pm4_type3_packets[opcode]} expects {expected} dwords, got {len(values)}")
      if opcode not in {mesa.CP_SET_MARKER, mesa.CP_WAIT_REG_MEM, mesa.CP_REG_TO_MEM, mesa.CP_EXEC_CS}: return
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
    self.pending.append(CommandStream(words, timestamp, context_id))

  def progress(self) -> int:
    while self.pending:
      made_progress, blocked_contexts, index = False, set(), 0
      while index < len(self.pending):
        stream = self.pending[index]
        if stream.context_id in blocked_contexts:
          index += 1
          continue
        state = self.context_states.setdefault(stream.context_id, A630ContextState({}))
        self.regs, self.constants_addr = state.regs, state.constants_addr
        initial_cursor, waiting = stream.cursor, False
        try:
          while stream.cursor < len(stream.words): stream.cursor = self._execute_one(stream.words, stream.cursor)
        except A630Wait: waiting = True
        except Exception:
          self.pending.pop(index)
          raise
        finally: state.constants_addr = self.constants_addr
        if waiting:
          made_progress |= stream.cursor != initial_cursor
          blocked_contexts.add(stream.context_id)
          index += 1
          continue
        self.completed_timestamps[stream.context_id] = stream.timestamp
        self.completed_timestamp = max(self.completed_timestamp, stream.timestamp)
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
        self._validate_memory(address, 4)
        ctypes.c_uint32.from_address(address).value = values[3]
      return
    if opcode == mesa.CP_REG_TO_MEM:
      destination = _u64(values[1], values[2])
      self._validate_memory(destination, 8)
      ctypes.c_uint64.from_address(destination).value = int(time.perf_counter() * 19_200_000)
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
    if opcode == mesa.CP_LOAD_STATE6_FRAG:
      state_type = _field(values[0], "CP_LOAD_STATE6_0_STATE_TYPE")
      state_block = _field(values[0], "CP_LOAD_STATE6_0_STATE_BLOCK")
      state_source = _field(values[0], "CP_LOAD_STATE6_0_STATE_SRC")
      if state_type == mesa.ST_CONSTANTS and state_block == mesa.SB6_CS_SHADER and state_source == mesa.SS6_INDIRECT:
        self.constants_addr = _u64(values[1], values[2])
      return
    if opcode == mesa.CP_EXEC_CS:
      self._run_kernel((values[1], values[2], values[3]))
      return
    if opcode == mesa.CP_RUN_OPENCL:
      self._run_kernel(cast(tuple[int, int, int], tuple(self.regs.get(reg, 1) for reg in
                           (mesa.REG_A6XX_SP_CS_KERNEL_GROUP_X, mesa.REG_A6XX_SP_CS_KERNEL_GROUP_Y, mesa.REG_A6XX_SP_CS_KERNEL_GROUP_Z))))
      return

  def _run_kernel(self, global_size:tuple[int, int, int]):
    if self.constants_addr is None: raise RuntimeError("A630 kernel launch has no constant-buffer address")
    shader_addr = _u64(self.regs[mesa.REG_A6XX_SP_CS_BASE], self.regs[mesa.REG_A6XX_SP_CS_BASE + 1])
    self._validate_memory(shader_addr, 1)
    self._validate_memory(self.constants_addr, 1)
    ndrange = self.regs[mesa.REG_A6XX_SP_CS_NDRANGE_0]
    local_size = cast(tuple[int, int, int], tuple(_field(ndrange, f"A6XX_SP_CS_NDRANGE_0_LOCALSIZE{axis}") + 1 for axis in "XYZ"))
    find_program(shader_addr, self._mapped_size(shader_addr)).execute(self.constants_addr, global_size, local_size, self._mapped_size)
