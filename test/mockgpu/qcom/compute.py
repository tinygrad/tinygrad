from tinygrad.runtime.autogen import mesa as m
from dataclasses import dataclass
from collections.abc import Callable
import hashlib, itertools, math
from collections.abc import Mapping
from test.mockgpu.qcom.pm4 import decode
from test.mockgpu.qcom.errors import input_boundary
from test.mockgpu.qcom.state import QCOMGPU, QCOMGPUState, buffer_at, read_buffer
from test.mockgpu.qcom.ir3 import _decode_program, execute_group

FIXED = {m.REG_A6XX_SP_CS_CNTL_1: (0x41,), m.REG_A6XX_SP_CS_BOOLEAN_CF_MASK: (0,), m.REG_A6XX_SP_CS_PROGRAM_COUNTER_OFFSET: (0,),
         m.REG_A6XX_SP_CS_PVT_MEM_PARAM: (0,), m.REG_A6XX_SP_CS_PVT_MEM_SIZE: (0,), m.REG_A6XX_SP_CS_TSIZE: (0x80,),
         m.REG_A6XX_SP_CS_CONFIG: (0x100,), m.REG_A6XX_SP_CS_PVT_MEM_STACK_OFFSET: (0x1000,), m.REG_A6XX_SP_CS_USIZE: (0x40,),
         m.REG_A6XX_SP_MODE_CNTL: (5,), m.REG_A6XX_SP_PERFCTR_SHADER_MASK: (0x20,), m.REG_A6XX_TPL1_MODE_CNTL: (2,),
         m.REG_A6XX_TPL1_DBG_ECO_CNTL: (0,),
         m.REG_A6XX_SP_REG_PROG_ID_0: (0xfcfcfcfc,), m.REG_A6XX_SP_REG_PROG_ID_1: (0xfcfcfcfc,), m.REG_A6XX_SP_REG_PROG_ID_2: (0xfcfcfcfc,),
         m.REG_A6XX_SP_REG_PROG_ID_3: (0xfc,), m.REG_A6XX_SP_CS_NDRANGE_2: (0,),
         m.REG_A6XX_SP_CS_NDRANGE_4: (0,), m.REG_A6XX_SP_CS_NDRANGE_6: (0,), m.REG_A6XX_SP_CS_WGE_CNTL: (0xfc,), m.REG_A6XX_SP_UPDATE_CNTL: (0, 0x60)}
MASKS = {m.REG_A6XX_SP_CS_CNTL_0: 0x1ffe, m.REG_A6XX_SP_CS_BASE: 0xffffffff, m.REG_A6XX_SP_CS_BASE + 1: 0xffffffff,
         m.REG_A6XX_SP_CS_PVT_MEM_BASE: 0xffffffff, m.REG_A6XX_SP_CS_PVT_MEM_BASE + 1: 0xffffffff,
         m.REG_A6XX_SP_CS_INSTR_SIZE: 0x0fffffff, m.REG_A6XX_SP_CS_CONST_CONFIG: 0x1ff, m.REG_A6XX_SP_CS_NDRANGE_0: 0xffffffff,
         m.REG_A6XX_SP_CS_NDRANGE_1: 0xffffffff, m.REG_A6XX_SP_CS_NDRANGE_3: 0xffffffff,
         m.REG_A6XX_SP_CS_NDRANGE_5: 0xffffffff, m.REG_A6XX_SP_CS_CONST_CONFIG_0: 0xffffffff, m.REG_A6XX_SP_CS_KERNEL_GROUP_X: 0xffffffff,
         m.REG_A6XX_SP_CS_KERNEL_GROUP_Y: 0xffffffff, m.REG_A6XX_SP_CS_KERNEL_GROUP_Z: 0xffffffff}

@dataclass(frozen=True)
class QCOMComputeState(QCOMGPUState):
  mode: int
  idle_waits: int
  memory_waits: int
  flushes: int
  invalidations: int
  dispatches: int

class QCOMCompute(QCOMGPU):
  def __init__(self, instrumentation: Callable[[dict[str, object]], None] | None = None):
    super().__init__()
    self.mode = 0
    self.idle_waits = self.memory_waits = self.flushes = self.invalidations = 0
    self.dispatches = 0
    self.instrumentation = instrumentation

  def snapshot(self) -> QCOMComputeState:
    base = super().snapshot()
    return QCOMComputeState(base.shader, base.shader_addr, base.constants, base.registers, self.mode, self.idle_waits,
                             self.memory_waits, self.flushes, self.invalidations, self.dispatches)

  def restore(self, state: QCOMGPUState):
    if not isinstance(state, QCOMComputeState): raise TypeError(f'QCOM compute state: invalid snapshot {type(state).__name__}')
    super().restore(state)
    self.mode, self.idle_waits, self.memory_waits = state.mode, state.idle_waits, state.memory_waits
    self.flushes, self.invalidations, self.dispatches = state.flushes, state.invalidations, state.dispatches

  def _register(self, reg: int, value: int, ctx: str):
    if reg in FIXED:
      if value not in FIXED[reg]: raise ValueError(f'{ctx}: unsupported register {reg:#x} value {value:#x}')
    elif reg not in MASKS or value & ~MASKS[reg]: raise ValueError(f'{ctx}: unsupported register {reg:#x} bits {value:#x}')
    if reg == m.REG_A6XX_SP_UPDATE_CNTL and value & 0x20:
      self.shader, self.shader_addr, self.constants = None, None, {}
    self.registers[reg] = value

  def _dispatch(self, groups: tuple[int, ...], buffers: dict[int, bytearray], ctx: str):
    if self.mode != 8: raise ValueError(f'{ctx}: compute marker missing')
    for reg in FIXED.keys() | MASKS.keys():
      if reg not in self.registers: raise ValueError(f'{ctx}: missing compute register {reg:#x}')
    regs = self.registers
    if regs[m.REG_A6XX_SP_UPDATE_CNTL]: raise ValueError(f'{ctx}: state reset still asserted')
    control = regs[m.REG_A6XX_SP_CS_NDRANGE_0]
    if control & 3 != 3: raise ValueError(f'{ctx}: unsupported kernel dimension')
    local = tuple(((control >> shift) & 1023)+1 for shift in (2, 12, 22))
    if len(groups) != 3 or any(n <= 0 for n in groups) or math.prod(local) > 1024 or math.prod(groups)*math.prod(local) > 65536:
      raise ValueError(f'{ctx}: unsupported dispatch dimensions groups={groups} local={local}')
    if tuple(regs[reg] for reg in (m.REG_A6XX_SP_CS_KERNEL_GROUP_X, m.REG_A6XX_SP_CS_KERNEL_GROUP_Y, m.REG_A6XX_SP_CS_KERNEL_GROUP_Z)) != groups:
      raise ValueError(f'{ctx}: group registers disagree with EXEC_CS')
    if tuple(regs[reg] for reg in range(m.REG_A6XX_SP_CS_NDRANGE_1, m.REG_A6XX_SP_CS_NDRANGE_6, 2)) != tuple(g*l for g, l in zip(groups, local)):
      raise ValueError(f'{ctx}: global sizes disagree with group/local sizes')
    if (regs[m.REG_A6XX_SP_CS_CONST_CONFIG] & 255)*16 > 1024: raise ValueError(f'{ctx}: CONSTLEN exceeds constant RAM mode')
    ids = tuple((regs[m.REG_A6XX_SP_CS_CONST_CONFIG_0] >> shift) & 255 for shift in (0, 8, 16, 24))
    if ids[0] not in (192, 252) or ids[1:3] != (252, 252) or ids[3] not in (0, 252):
      raise ValueError(f'{ctx}: unsupported system registers {ids}')
    program = self.bind(buffers)
    words = _decode_program(program)
    if self.instrumentation is not None:
      self.instrumentation({'kind': 'dispatch', 'context': ctx, 'groups': list(groups),
                            'local': list(local), 'global': [regs[reg] for reg in (m.REG_A6XX_SP_CS_NDRANGE_1,
                                                                                     m.REG_A6XX_SP_CS_NDRANGE_3,
                                                                                     m.REG_A6XX_SP_CS_NDRANGE_5)],
                            'bytes': len(program), 'program_sha256': hashlib.sha256(program).hexdigest()})
    for group in itertools.product(*(range(n) for n in groups)):
      lanes = []
      for lane in itertools.product(*(range(n) for n in local)):
        initial: dict[int, int] = {}
        for index, values in ((ids[0], group), (ids[3], lane)):
          if index != 0xfc: initial.update(enumerate(values, index))
        lanes.append(initial)
      execute_group(program, lanes, self.constants, buffers, ((regs[m.REG_A6XX_SP_CS_CNTL_1] & 31)+1)*1024, padded=True, _words=words)
    self.dispatches += 1

  @input_boundary
  def run(self, data: bytes, memory: Mapping[int, bytearray]):
    for base, blob in memory.items():
      if type(base) is not int or type(blob) is not bytearray: raise ValueError('QCOM memory: invalid region')
      buffer_at(memory, base, len(blob), 'QCOM memory')
    if len({id(blob) for blob in memory.values()}) != len(memory): raise ValueError('QCOM memory: aliased host buffers')
    transaction = self.begin_transaction(memory)
    transaction.gpu.execute_command_stream(data, transaction.buffers)
    transaction.commit(memory)

  @input_boundary
  def execute_command_stream(self, data: bytes, buffers: dict[int, bytearray]):
    for packet in decode(data):
      ctx = f'PM4 dword {packet.offset} type={packet.type} target={packet.target:#x}'
      values = packet.payload
      if packet.type == 4:
        if not values: raise ValueError(f'{ctx}: empty register write unsupported')
        for reg, value in enumerate(values, packet.target): self._register(reg, value, ctx)
      elif packet.target == m.CP_LOAD_STATE6_FRAG:
        self.load_state(values, buffers, ctx)
      elif packet.target == m.CP_SET_MARKER:
        if values != (8,): raise ValueError(f'{ctx}: only compute marker supported')
        self.mode = 8
      elif packet.target == m.CP_EXEC_CS:
        if len(values) != 4 or values[0]: raise ValueError(f'{ctx}: unsupported EXEC_CS form')
        self._dispatch(values[1:], buffers, ctx)
      elif packet.target in (m.CP_WAIT_FOR_IDLE, m.CP_WAIT_MEM_WRITES):
        if values: raise ValueError(f'{ctx}: wait packet must have no payload')
        # All commands complete synchronously; these waits observe that completion.
        if packet.target == m.CP_WAIT_FOR_IDLE: self.idle_waits += 1
        else: self.memory_waits += 1
      elif packet.target == m.CP_WAIT_REG_MEM:
        if len(values) != 6 or values[0] != 0x15 or values[4:] != (0xffffffff, 32):
          raise ValueError(f'{ctx}: unsupported WAIT_REG_MEM form')
        addr = values[1] | (values[2] << 32)
        if addr & 3: raise ValueError(f'{ctx}: unaligned wait address {addr:#x}')
        observed = int.from_bytes(read_buffer(buffers, addr, 4, ctx), 'little')
        if observed < values[3]: raise ValueError(f'{ctx}: unsatisfied wait {observed} < {values[3]}')
      elif packet.target == m.CP_EVENT_WRITE:
        if values == (0x31,): self.invalidations += 1
        elif len(values) == 4 and values[0] == 4:
          addr = values[1] | (values[2] << 32)
          if addr & 3: raise ValueError(f'{ctx}: unaligned event address {addr:#x}')
          base = buffer_at(buffers, addr, 4, ctx)
          buffers[base][addr-base:addr-base+4] = values[3].to_bytes(4, 'little')
          self.flushes += 1
        else: raise ValueError(f'{ctx}: unsupported event or event flags')
      else: raise ValueError(f'{ctx}: unsupported packet')
