"""SQTT (SQ Thread Trace) packet encoder and decoder for AMD GPUs.

This module provides encoding and decoding of raw SQTT byte streams.
The format is nibble-based with variable-width packets determined by a state machine.
Uses BitField infrastructure from dsl.py, similar to GPU instruction encoding.
"""
from __future__ import annotations
from enum import IntEnum
from typing import get_type_hints
from extra.assembly.amd.dsl import BitField, bits

# ═══════════════════════════════════════════════════════════════════════════════
# FIELD ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class MemSrc(IntEnum):
  LDS = 0
  LDS_ALT = 1
  VMEM = 2
  VMEM_ALT = 3

class AluSrc(IntEnum):
  NONE = 0
  SALU = 1
  VALU = 2
  VALU_ALT = 3

class InstOp(IntEnum):
  """SQTT instruction operation types.

  Memory ops appear in two ranges depending on which SIMD executes them:
  - 0x2x range: ops on traced SIMD
  - 0x5x range: ops on other SIMD (OTHER_ prefix)
  """
  SALU = 0x0
  SMEM = 0x1
  JUMP = 0x3
  MESSAGE = 0x9
  VALU = 0xb
  VALU_64 = 0xd
  VALU_MAD64 = 0xe
  VALU_64_2 = 0x10
  # Memory ops on traced SIMD (0x2x range)
  VMEM_LOAD = 0x21
  VMEM_STORE = 0x24
  VMEM_STORE_64 = 0x25
  VMEM_STORE_96 = 0x26
  VMEM_STORE_128 = 0x27
  LDS_LOAD = 0x29
  LDS_STORE = 0x2b
  LDS_STORE_64 = 0x2c
  LDS_STORE_128 = 0x2e
  OTHER_VALU_64 = 0x3a
  # Memory ops on other SIMD (0x5x range)
  OTHER_LDS_LOAD = 0x50
  OTHER_LDS_STORE = 0x51
  OTHER_LDS_STORE_64 = 0x52
  OTHER_LDS_STORE_128 = 0x54
  OTHER_VMEM_LOAD = 0x5a
  OTHER_VMEM_STORE = 0x5b
  OTHER_VMEM_STORE_64 = 0x5c
  OTHER_VMEM_STORE_96 = 0x5d
  OTHER_VMEM_STORE_128 = 0x5e
  VALU_CMPX = 0x73

# ═══════════════════════════════════════════════════════════════════════════════
# PACKET TYPE BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class PacketType:
  """Base class for SQTT packet types."""
  _encoding: tuple[BitField, int] | None = None
  _field_types: dict[str, type] = {}
  _values: dict[str, int]
  _raw: int
  _time: int

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    if 'encoding' in cls.__dict__ and isinstance(cls.__dict__['encoding'], tuple):
      cls._encoding = cls.__dict__['encoding']
    # Cache field type annotations for enum conversion
    try:
      cls._field_types = {k: v for k, v in get_type_hints(cls).items()
                          if isinstance(v, type) and issubclass(v, IntEnum)}
    except Exception:
      cls._field_types = {}

  @classmethod
  def fields(cls) -> dict[str, BitField]:
    return {k: v for k, v in cls.__dict__.items() if isinstance(v, BitField) and k != 'encoding'}

  @classmethod
  def size_bits(cls) -> int:
    max_bit = max((f.hi for f in cls.fields().values()), default=0)
    return ((max_bit + 4) // 4) * 4

  @classmethod
  def size_nibbles(cls) -> int:
    return cls.size_bits() // 4

  @classmethod
  def from_raw(cls, raw: int, time: int = 0):
    inst = object.__new__(cls)
    inst._raw = raw
    inst._time = time
    inst._values = {}
    for name, bf in cls.fields().items():
      val = (raw >> bf.lo) & bf.mask()
      # Convert to enum if annotated
      enum_type = cls._field_types.get(name)
      if enum_type is not None:
        try: val = enum_type(val)
        except ValueError: pass
      inst._values[name] = val
    return inst

  def __getattr__(self, name: str):
    if name.startswith('_'): raise AttributeError(name)
    return self._values.get(name, 0)

  def __repr__(self) -> str:
    fields_str = ", ".join(f"{k}={v}" for k, v in self._values.items() if not k.startswith('_'))
    return f"{self.__class__.__name__}({fields_str})"

# ═══════════════════════════════════════════════════════════════════════════════
# PACKET TYPE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class VALUINST(PacketType):
  encoding = bits[2:0] == 0b011
  delta = bits[5:3]
  flag = bits[6:6]
  wave = bits[11:7]

class VMEMEXEC(PacketType):
  encoding = bits[3:0] == 0b1111
  delta = bits[5:4]
  src: MemSrc = bits[7:6]

class ALUEXEC(PacketType):
  encoding = bits[3:0] == 0b1110
  delta = bits[5:4]
  src: AluSrc = bits[7:6]

class IMMEDIATE(PacketType):
  encoding = bits[3:0] == 0b1101
  delta = bits[6:4]
  wave = bits[11:7]

class IMMEDIATE_MASK(PacketType):
  encoding = bits[4:0] == 0b00100
  delta = bits[7:5]
  mask = bits[23:8]

class WAVERDY(PacketType):
  encoding = bits[4:0] == 0b10100
  delta = bits[7:5]
  mask = bits[23:8]

class TS_DELTA_S8_W3(PacketType):
  encoding = bits[6:0] == 0b0100001
  delta = bits[10:8]
  _padding = bits[63:11]

class WAVEEND(PacketType):
  encoding = bits[4:0] == 0b10101
  delta = bits[7:5]
  flag7 = bits[8:8]
  simd = bits[10:9]
  cu_lo = bits[13:11]
  wave = bits[19:15]
  @property
  def cu(self) -> int: return self.cu_lo | (self.flag7 << 3)

class WAVESTART(PacketType):
  encoding = bits[4:0] == 0b01100
  delta = bits[6:5]
  flag7 = bits[7:7]
  simd = bits[9:8]
  cu_lo = bits[12:10]
  wave = bits[17:13]
  id7 = bits[31:18]
  @property
  def cu(self) -> int: return self.cu_lo | (self.flag7 << 3)

class TS_DELTA_S5_W2(PacketType):
  encoding = bits[4:0] == 0b11100
  delta = bits[6:5]
  _padding = bits[47:7]

class WAVEALLOC(PacketType):
  encoding = bits[4:0] == 0b00101
  delta = bits[7:5]
  _padding = bits[19:8]

class TS_DELTA_S5_W3(PacketType):
  encoding = bits[4:0] == 0b00110
  delta = bits[7:5]
  _padding = bits[51:8]

class PERF(PacketType):
  encoding = bits[4:0] == 0b10110
  delta = bits[7:5]
  arg = bits[27:8]

class TS_DELTA_SHORT(PacketType):
  encoding = bits[3:0] == 0b1000
  delta = bits[7:4]

class NOP(PacketType):
  encoding = bits[3:0] == 0b0000
  delta = None  # type: ignore
  _padding = bits[3:0]

class TS_WAVE_STATE(PacketType):
  encoding = bits[6:0] == 0b1010001
  delta = bits[15:7]
  coarse = bits[23:16]
  @property
  def wave_interest(self) -> bool: return bool(self.coarse & 1)
  @property
  def terminate_all(self) -> bool: return bool(self.coarse & 8)

class EVENT(PacketType):
  encoding = bits[7:0] == 0b01100001
  delta = bits[10:8]
  event = bits[23:11]

class EVENT_BIG(PacketType):
  encoding = bits[7:0] == 0b11100001
  delta = bits[10:8]
  event = bits[31:11]

class REG(PacketType):
  encoding = bits[3:0] == 0b1001
  delta = bits[6:4]
  slot = bits[9:7]
  hi_byte = bits[15:8]
  subop = bits[31:16]
  val32 = bits[63:32]
  @property
  def is_config(self) -> bool: return bool(self.hi_byte & 0x80)

class SNAPSHOT(PacketType):
  encoding = bits[6:0] == 0b1110001
  delta = bits[9:7]
  snap = bits[63:10]

class TS_DELTA_OR_MARK(PacketType):
  encoding = bits[6:0] == 0b0000001
  delta = bits[47:12]
  bit8 = bits[8:8]
  bit9 = bits[9:9]
  @property
  def is_marker(self) -> bool: return bool(self.bit9 and not self.bit8)

class LAYOUT_HEADER(PacketType):
  encoding = bits[6:0] == 0b0010001
  delta = None  # type: ignore
  layout = bits[12:7]
  simd = bits[14:13]
  group = bits[17:15]
  sel_a = bits[31:28]
  sel_b = bits[36:33]
  flag4 = bits[59:59]
  _padding = bits[63:60]

class INST(PacketType):
  encoding = bits[2:0] == 0b010
  delta = bits[6:4]
  flag1 = bits[3:3]
  flag2 = bits[7:7]
  wave = bits[12:8]
  op: InstOp = bits[19:13]

class UTILCTR(PacketType):
  encoding = bits[6:0] == 0b0110001
  delta = bits[8:7]
  ctr = bits[47:9]

# All packet types in encoding priority order (more specific masks first, NOP last as fallback)
PACKET_TYPES: list[type[PacketType]] = [
  EVENT, EVENT_BIG,
  TS_DELTA_S8_W3, TS_WAVE_STATE, SNAPSHOT, TS_DELTA_OR_MARK, LAYOUT_HEADER, UTILCTR,
  IMMEDIATE_MASK, WAVERDY, WAVEEND, WAVESTART, TS_DELTA_S5_W2, WAVEALLOC, TS_DELTA_S5_W3, PERF,
  VMEMEXEC, ALUEXEC, IMMEDIATE, TS_DELTA_SHORT, REG,
  VALUINST, INST,
  NOP,
]

PACKET_BY_NAME: dict[str, type[PacketType]] = {cls.__name__: cls for cls in PACKET_TYPES}

def _build_state_table() -> tuple[bytes, dict[int, type[PacketType]]]:
  table = [len(PACKET_TYPES) - 1] * 256  # default to NOP
  opcode_to_class: dict[int, type[PacketType]] = {i: cls for i, cls in enumerate(PACKET_TYPES)}

  for byte_val in range(256):
    for opcode, pkt_cls in enumerate(PACKET_TYPES):
      if pkt_cls._encoding is None: continue
      mask_bf, pattern = pkt_cls._encoding
      if (byte_val & mask_bf.mask()) == pattern:
        table[byte_val] = opcode
        break

  return bytes(table), opcode_to_class

STATE_TO_OPCODE, OPCODE_TO_CLASS = _build_state_table()

OPCODE_TO_BYTES: dict[int, list[int]] = {}
for _byte_val, _opcode in enumerate(STATE_TO_OPCODE):
  if _opcode not in OPCODE_TO_BYTES: OPCODE_TO_BYTES[_opcode] = []
  OPCODE_TO_BYTES[_opcode].append(_byte_val)

BUDGET = {opcode: pkt_cls.size_nibbles() for opcode, pkt_cls in OPCODE_TO_CLASS.items()}

# ═══════════════════════════════════════════════════════════════════════════════
# DECODER
# ═══════════════════════════════════════════════════════════════════════════════

def decode(data: bytes) -> list[PacketType]:
  """Decode raw SQTT blob into list of packet instances."""
  packets: list[PacketType] = []
  n = len(data)
  reg = 0
  offset = 0
  nib_count = 16
  time = 0

  while (offset >> 3) < n:
    target = offset + nib_count * 4
    while offset < target and (offset >> 3) < n:
      byte = data[offset >> 3]
      nib = (byte >> (offset & 4)) & 0xF
      reg = ((reg >> 4) | (nib << 60)) & ((1 << 64) - 1)
      offset += 4
    if offset < target: break

    opcode = STATE_TO_OPCODE[reg & 0xFF]
    pkt_cls = OPCODE_TO_CLASS[opcode]
    nib_count = BUDGET[opcode]

    delta_field = getattr(pkt_cls, 'delta', None)
    delta = (reg >> delta_field.lo) & delta_field.mask() if delta_field is not None else 0

    if pkt_cls is TS_DELTA_OR_MARK:
      bit8 = (reg >> TS_DELTA_OR_MARK.bit8.lo) & TS_DELTA_OR_MARK.bit8.mask()
      bit9 = (reg >> TS_DELTA_OR_MARK.bit9.lo) & TS_DELTA_OR_MARK.bit9.mask()
      if bit9 and not bit8: delta = 0
    elif pkt_cls is TS_DELTA_SHORT:
      delta = delta + 8

    time += delta
    pkt = pkt_cls.from_raw(reg, time)
    packets.append(pkt)

  return packets

# ═══════════════════════════════════════════════════════════════════════════════
# ENCODER
# ═══════════════════════════════════════════════════════════════════════════════

def encode(packets: list[PacketType]) -> bytes:
  """Encode a list of packet instances into raw SQTT blob."""
  if not packets: return b''

  read_lengths = [16]
  for p in packets[:-1]:
    read_lengths.append(p.size_nibbles())

  total_nibbles = sum(read_lengths)
  bits_arr = [0] * (total_nibbles * 4)

  cumulative = 0
  for i, p in enumerate(packets):
    cumulative += read_lengths[i]
    pkt_cls = type(p)
    opcode = next(op for op, cls in OPCODE_TO_CLASS.items() if cls is pkt_cls)

    byte_vals = OPCODE_TO_BYTES.get(opcode)
    if not byte_vals: raise ValueError(f"No encoding for {pkt_cls.__name__}")
    opcode_byte = byte_vals[0]

    delta_field = getattr(pkt_cls, 'delta', None)
    if delta_field is not None and delta_field.hi < 8:
      delta = p._values.get('delta', 0)
      if isinstance(delta, IntEnum): delta = delta.value
      if pkt_cls is TS_DELTA_SHORT: delta = max(0, delta - 8)
      delta = delta & delta_field.mask()
      opcode_byte = (opcode_byte & ~(delta_field.mask() << delta_field.lo)) | (delta << delta_field.lo)

    opcode_nibble_pos = max(0, cumulative - 16)
    opcode_bit_pos = opcode_nibble_pos * 4

    for b in range(8):
      if opcode_bit_pos + b < len(bits_arr):
        bits_arr[opcode_bit_pos + b] = (opcode_byte >> b) & 1

  nibbles = [sum(bits_arr[i + j] << j for j in range(4) if i + j < len(bits_arr)) for i in range(0, len(bits_arr), 4)]
  while len(nibbles) % 2: nibbles.append(0)
  return bytes(nibbles[i] | (nibbles[i + 1] << 4) for i in range(0, len(nibbles), 2))
