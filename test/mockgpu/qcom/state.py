from tinygrad.runtime.autogen import mesa as m
from dataclasses import dataclass
import copy, struct
from collections.abc import Mapping
from test.mockgpu.qcom.pm4 import decode
from test.mockgpu.qcom.errors import input_boundary

@dataclass(frozen=True)
class QCOMGPUState:
  shader: bytes | None
  shader_addr: int | None
  constants: tuple[tuple[int, int], ...]
  registers: tuple[tuple[int, int], ...]

# Run on copies; commit only after execution and publish checks pass.
class QCOMTransaction:
  def __init__(self, owner, memory: Mapping[int, bytes | bytearray]):
    self.owner = owner
    self.gpu = owner.fork()
    self.before = {base: bytes(blob) for base, blob in memory.items()}
    self.buffers = {base: bytearray(blob) for base, blob in memory.items()}

  def changed_bytes(self) -> int:
    return sum(len(blob) for base, blob in self.buffers.items() if blob != self.before[base])

  def commit_state(self):
    self.owner.restore(self.gpu.snapshot())

  def commit(self, memory: Mapping[int, bytearray]):
    for base, blob in self.buffers.items(): memory[base][:] = blob
    self.commit_state()

@input_boundary
def buffer_at(buffers: Mapping[int, bytes | bytearray], addr: int, size: int, ctx: str) -> int:
  if size <= 0 or addr < 0 or addr + size > 1 << 64: raise ValueError(f'{ctx}: invalid range {addr:#x}+{size}')
  overlaps = [(base, blob) for base, blob in buffers.items() if blob and base < addr+size and addr < base+len(blob)]
  if len(overlaps) != 1: raise ValueError(f'{ctx}: address {addr:#x}+{size} has {len(overlaps)} backing buffers')
  base, blob = overlaps[0]
  if base < 0 or base + len(blob) > 1 << 64 or not base <= addr < addr+size <= base+len(blob):
    raise ValueError(f'{ctx}: address {addr:#x}+{size} has 0 backing buffers covering the range')
  return base

def read_buffer(buffers: Mapping[int, bytes | bytearray], addr: int, size: int, ctx: str) -> bytes:
  base = buffer_at(buffers, addr, size, ctx)
  return bytes(buffers[base][addr-base:addr-base+size])

class QCOMGPU:
  def __init__(self):
    self.shader: bytes | None = None
    self.constants: dict[int, int] = {}
    self.registers: dict[int, int] = {}
    self.shader_addr: int | None = None

  def snapshot(self) -> QCOMGPUState:
    return QCOMGPUState(self.shader, self.shader_addr, tuple(sorted(self.constants.items())), tuple(sorted(self.registers.items())))

  def restore(self, state: QCOMGPUState):
    if not isinstance(state, QCOMGPUState): raise TypeError(f'QCOM state: invalid snapshot {type(state).__name__}')
    self.shader, self.shader_addr = state.shader, state.shader_addr
    self.constants, self.registers = dict(state.constants), dict(state.registers)

  def begin_transaction(self, memory: Mapping[int, bytes | bytearray]) -> QCOMTransaction:
    return QCOMTransaction(self, memory)

  def fork(self):
    gpu = copy.copy(self)
    gpu.restore(self.snapshot())
    return gpu

  @input_boundary
  def bind(self, buffers: Mapping[int, bytes | bytearray], required_constants: tuple[int, ...] = ()) -> bytes:
    # Readiness for the narrow source model, not a dispatch or hardware execution check.
    for reg in (m.REG_A6XX_SP_CS_BASE, m.REG_A6XX_SP_CS_BASE + 1, m.REG_A6XX_SP_CS_INSTR_SIZE, m.REG_A6XX_SP_CS_CONST_CONFIG):
      if reg not in self.registers: raise ValueError(f'QCOM binding: missing register {reg:#x}')
    addr = self.registers[m.REG_A6XX_SP_CS_BASE] | (self.registers[m.REG_A6XX_SP_CS_BASE + 1] << 32)
    size, config = self.registers[m.REG_A6XX_SP_CS_INSTR_SIZE]*128, self.registers[m.REG_A6XX_SP_CS_CONST_CONFIG]
    if addr & 31: raise ValueError(f'QCOM binding: program base {addr:#x} is not 32-byte aligned')
    if not config & 0x100: raise ValueError('QCOM binding: constants disabled')
    # CONSTLEN decodes to vec4s: the XML stores it shifted right by two.
    limit = (config & 0xff)*16
    if any(index >= limit for index in self.constants): raise ValueError('QCOM binding: loaded constants exceed CONSTLEN')
    for index in required_constants:
      if index < 0 or index >= limit: raise ValueError(f'QCOM binding: constant {index} exceeds CONSTLEN')
      if index not in self.constants: raise ValueError(f'QCOM binding: constant {index} uninitialized')
    program = read_buffer(buffers, addr, size, 'QCOM binding program')
    if self.shader is None: raise ValueError('QCOM binding: shader preload missing')
    if self.shader_addr != addr: raise ValueError('QCOM binding: shader preload address differs from program base')
    if len(self.shader) > size: raise ValueError('QCOM binding: shader preload exceeds program size')
    if program[:len(self.shader)] != self.shader: raise ValueError('QCOM binding: shader preload differs from program bytes')
    return program

  @input_boundary
  def apply_binding_packets(self, data: bytes, buffers: Mapping[int, bytes | bytearray]):
    transaction = self.begin_transaction({})
    for packet in decode(data):
      ctx = f'PM4 dword {packet.offset} type={packet.type} target={packet.target:#x}'
      if packet.type == 4:
        if not packet.payload: raise ValueError(f'{ctx}: empty register write unsupported')
        for reg, value in enumerate(packet.payload, packet.target):
          mask = {m.REG_A6XX_SP_CS_BASE: 0xffffffff, m.REG_A6XX_SP_CS_BASE + 1: 0xffffffff, m.REG_A6XX_SP_CS_INSTR_SIZE: 0x0fffffff,
                  m.REG_A6XX_SP_CS_CONST_CONFIG: 0x1ff}.get(reg)
          if mask is None: raise ValueError(f'{ctx}: unsupported register {reg:#x}')
          if value & ~mask: raise ValueError(f'{ctx}: unsupported bits in register {reg:#x}: {value:#x}')
          transaction.gpu.registers[reg] = value
        continue
      if packet.type != 7 or packet.target != m.CP_LOAD_STATE6_FRAG: raise ValueError(f'{ctx}: unsupported packet')
      transaction.gpu.load_state(packet.payload, buffers, ctx)
    transaction.commit_state()

  # Legacy raw entrypoint.
  def submit(self, data: bytes, buffers: Mapping[int, bytes | bytearray]):
    self.apply_binding_packets(data, buffers)

  @input_boundary
  def load_state(self, values: tuple[int, ...], buffers: Mapping[int, bytes | bytearray], ctx: str):
    if len(values) < 3: raise ValueError(f'{ctx}: LOAD_STATE6 needs three control dwords')
    control, lo, hi, *inline = values
    dst, kind, src, block, count = control & 0x3fff, (control >> 14) & 3, (control >> 16) & 3, (control >> 18) & 15, control >> 22
    ctx += f' state={kind} source={src} block={block:#x} count={count} dst={dst}'
    if block != 0xd or kind not in (0, 1) or src not in (0, 2): raise ValueError(f'{ctx}: unsupported state')
    if not count: raise ValueError(f'{ctx}: zero count unsupported')
    if kind == 0 and (src != 2 or dst): raise ValueError(f'{ctx}: only indirect shader preload at offset zero supported')
    if kind == 1 and dst + count > 0x4000: raise ValueError(f'{ctx}: constant destination exceeds encoded range')
    size = count * (128 if kind == 0 else 16)
    if src == 0:
      if lo or hi or len(inline)*4 != size: raise ValueError(f'{ctx}: invalid direct payload or address')
      contents = struct.pack(f'<{len(inline)}I', *inline)
    else:
      addr = lo | (hi << 32)
      if inline or addr & 3 or addr + size > 1 << 64: raise ValueError(f'{ctx}: invalid indirect address {addr:#x} or payload')
      contents = read_buffer(buffers, addr, size, ctx)
    if kind == 0: self.shader, self.shader_addr = contents, lo | (hi << 32)
    else: self.constants.update(enumerate(struct.unpack(f'<{size//4}I', contents), dst*4))
