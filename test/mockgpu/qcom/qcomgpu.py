import ctypes, time
from tinygrad.runtime.autogen import mesa

def decode_pm4(words):
  packets, pc = [], 0
  while pc < len(words):
    header = words[pc]
    pc += 1
    packet_type = header >> 28
    if packet_type == 7:
      count, key = header & 0x3fff, (header >> 16) & 0x7f
    elif packet_type == 4:
      count, key = header & 0x7f, (header >> 8) & 0x3ffff
    else:
      raise NotImplementedError(f'unsupported PM4 packet type {packet_type}')
    if pc + count > len(words): raise ValueError('truncated PM4 packet')
    packets.append((packet_type, key, tuple(words[pc:pc+count])))
    pc += count
  return packets

class QCOMGPU:
  def __init__(self, executor=None, check_range=None):
    self.regs = {}
    self.state = {}
    self.state_units = {}
    self.executor = executor
    self.check_range = check_range

  def shader_image(self):
    address = self.state[(mesa.SB6_CS_SHADER, mesa.ST_SHADER)]
    size = self.regs[mesa.REG_A6XX_SP_CS_INSTR_SIZE] * 128
    if self.check_range is not None: self.check_range(address, size)
    return ctypes.string_at(address, size)

  def local_size(self):
    value = self.regs[mesa.REG_A6XX_SP_CS_NDRANGE_0]
    return tuple(1 + ((value & mask) >> shift) for mask, shift in (
      (mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEX__MASK, mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEX__SHIFT),
      (mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEY__MASK, mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEY__SHIFT),
      (mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEZ__MASK, mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEZ__SHIFT)))

  def constant_regs(self, lane_count):
    key = (mesa.SB6_CS_SHADER, mesa.ST_CONSTANTS)
    size = self.state_units[key] * 16
    if self.check_range is not None: self.check_range(self.state[key], size)
    words = (ctypes.c_uint32 * (size // 4)).from_address(self.state[key])
    return {('c', index // 4, index % 4): [value] * lane_count for index, value in enumerate(words)}

  def local_id_register(self):
    value = self.regs[mesa.REG_A6XX_SP_CS_CONST_CONFIG_0]
    encoded = (value & mesa.A6XX_SP_CS_CONST_CONFIG_0_LOCALIDREGID__MASK) >> mesa.A6XX_SP_CS_CONST_CONFIG_0_LOCALIDREGID__SHIFT
    if encoded != 0xfc and encoded % 4: raise ValueError(f'unaligned local ID register {encoded:#x}')
    return encoded if encoded == 0xfc else encoded // 4

  def workgroup_id_register(self):
    value = self.regs[mesa.REG_A6XX_SP_CS_CONST_CONFIG_0]
    encoded = (value & mesa.A6XX_SP_CS_CONST_CONFIG_0_WGIDCONSTID__MASK) >> mesa.A6XX_SP_CS_CONST_CONFIG_0_WGIDCONSTID__SHIFT
    if encoded != 0xfc and encoded % 4: raise ValueError(f'unaligned workgroup ID register {encoded:#x}')
    return encoded if encoded == 0xfc else encoded // 4

  def execute(self, words):
    for packet_type, key, payload in decode_pm4(words):
      if packet_type == 4:
        self.regs.update((key + index, value) for index, value in enumerate(payload))
      elif packet_type == 7 and key == mesa.CP_LOAD_STATE6_FRAG:
        if len(payload) != 3: raise ValueError('invalid CP_LOAD_STATE6_FRAG payload')
        control = payload[0]
        state_type = (control & mesa.CP_LOAD_STATE6_0_STATE_TYPE__MASK) >> mesa.CP_LOAD_STATE6_0_STATE_TYPE__SHIFT
        state_src = (control & mesa.CP_LOAD_STATE6_0_STATE_SRC__MASK) >> mesa.CP_LOAD_STATE6_0_STATE_SRC__SHIFT
        state_block = (control & mesa.CP_LOAD_STATE6_0_STATE_BLOCK__MASK) >> mesa.CP_LOAD_STATE6_0_STATE_BLOCK__SHIFT
        if state_src != mesa.SS6_INDIRECT: raise NotImplementedError(f'unsupported state source {state_src}')
        self.state[(state_block, state_type)] = payload[1] | payload[2] << 32
        self.state_units[(state_block, state_type)] = \
          (control & mesa.CP_LOAD_STATE6_0_NUM_UNIT__MASK) >> mesa.CP_LOAD_STATE6_0_NUM_UNIT__SHIFT
      elif packet_type == 7 and key == mesa.CP_WAIT_REG_MEM:
        if len(payload) != 6: raise ValueError('invalid CP_WAIT_REG_MEM payload')
        control = payload[0]
        function = (control & mesa.CP_WAIT_REG_MEM_0_FUNCTION__MASK) >> mesa.CP_WAIT_REG_MEM_0_FUNCTION__SHIFT
        poll = (control & mesa.CP_WAIT_REG_MEM_0_POLL__MASK) >> mesa.CP_WAIT_REG_MEM_0_POLL__SHIFT
        if function != mesa.WRITE_GE or poll != mesa.POLL_MEMORY: raise NotImplementedError('unsupported CP_WAIT_REG_MEM mode')
        address = payload[1] | payload[2] << 32
        mask, reference = payload[4], payload[3]
        if self.check_range is not None: self.check_range(address, 4)
        current = ctypes.c_uint32.from_address(address).value
        if current & mask < reference & mask: raise BlockingIOError('CP_WAIT_REG_MEM is blocked')
      elif packet_type == 7 and key == mesa.CP_EVENT_WRITE:
        event = payload[0] & mesa.CP_EVENT_WRITE_0_EVENT__MASK
        if len(payload) == 4 and event == mesa.CACHE_FLUSH_TS:
          address = payload[1] | payload[2] << 32
          if self.check_range is not None: self.check_range(address, 4)
          ctypes.c_uint32.from_address(address).value = payload[3]
        elif len(payload) == 1 and event == mesa.CACHE_INVALIDATE:
          pass
        else:
          raise NotImplementedError(f'unsupported CP_EVENT_WRITE event {event} with {len(payload)} words')
      elif packet_type == 7 and key == mesa.CP_REG_TO_MEM:
        if len(payload) != 3: raise ValueError('invalid CP_REG_TO_MEM payload')
        control = payload[0]
        reg = (control & mesa.CP_REG_TO_MEM_0_REG__MASK) >> mesa.CP_REG_TO_MEM_0_REG__SHIFT
        count = (control & mesa.CP_REG_TO_MEM_0_CNT__MASK) >> mesa.CP_REG_TO_MEM_0_CNT__SHIFT
        if reg != mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER or count != 2 or not control & mesa.CP_REG_TO_MEM_0_64B:
          raise NotImplementedError('unsupported CP_REG_TO_MEM mode')
        address = payload[1] | payload[2] << 32
        if self.check_range is not None: self.check_range(address, 8)
        ctypes.c_uint64.from_address(address).value = time.perf_counter_ns() * 19_200_000 // 1_000_000_000
      elif packet_type == 7 and key == mesa.CP_SET_MARKER:
        if len(payload) != 1: raise ValueError('invalid CP_SET_MARKER payload')
        mode = (payload[0] & mesa.A6XX_CP_SET_MARKER_0_MODE__MASK) >> mesa.A6XX_CP_SET_MARKER_0_MODE__SHIFT
        if mode != mesa.RM6_COMPUTE: raise NotImplementedError(f'unsupported CP_SET_MARKER mode {mode}')
      elif packet_type == 7 and key == mesa.CP_EXEC_CS:
        if len(payload) != 4: raise ValueError('invalid CP_EXEC_CS payload')
        if payload[0] != 0: raise NotImplementedError('unsupported CP_EXEC_CS control word')
        grid = (payload[1] & mesa.CP_EXEC_CS_1_NGROUPS_X__MASK,
          payload[2] & mesa.CP_EXEC_CS_2_NGROUPS_Y__MASK,
          payload[3] & mesa.CP_EXEC_CS_3_NGROUPS_Z__MASK)
        if self.executor is None: raise RuntimeError('IR3 executor is not connected')
        self.executor(self.shader_image(), grid, self.local_size(), self.regs, self.state)
      elif packet_type == 7 and key in (mesa.CP_WAIT_FOR_IDLE, mesa.CP_WAIT_MEM_WRITES):
        if payload: raise ValueError(f'invalid PM4 wait payload for opcode {key:#x}')
      else:
        raise NotImplementedError(f'unsupported PM4 opcode {key:#x}')
