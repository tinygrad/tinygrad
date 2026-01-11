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
  - 0x1x-0x2x range: ops on traced SIMD
  - 0x5x range: ops on other SIMD (OTHER_ prefix)

  GLOBAL memory ops encoding depends on addressing mode AND size:
  - Loads: 0x21 (saddr=SGPR) or 0x22 (saddr=NULL), all sizes same
  - Stores: base + size_offset, where VADDR is shifted +1 from SADDR
    SADDR: 0x24(32) 0x25(64) 0x26(96) 0x27(128)
    VADDR: 0x25(32) 0x26(64) 0x27(96) 0x28(128)

  OTHER_ range follows same pattern but values overlap differently.
  """
  SALU = 0x0
  SMEM = 0x1
  JUMP = 0x3              # branch taken
  JUMP_NO = 0x4           # branch not taken
  MESSAGE = 0x9
  VALU_TRANS = 0xb        # transcendental: exp, log, rcp, sqrt, sin, cos
  VALU_64_SHIFT = 0xd     # 64-bit shifts: lshl, lshr, ashr
  VALU_MAD64 = 0xe        # 64-bit multiply-add
  VALU_64 = 0xf           # 64-bit: add, mul, fma, rcp, sqrt, rounding, frexp, div helpers
  VINTERP = 0x12          # interpolation: v_interp_p10_f32, v_interp_p2_f32
  BARRIER = 0x13

  # FLAT memory ops on traced SIMD (0x1x range)
  FLAT_LOAD = 0x1c
  FLAT_STORE = 0x1d
  FLAT_STORE_64 = 0x1e
  FLAT_STORE_96 = 0x1f
  FLAT_STORE_128 = 0x20

  # GLOBAL memory ops on traced SIMD (0x2x range)
  GLOBAL_LOAD = 0x21             # saddr=SGPR, all sizes
  GLOBAL_LOAD_VADDR = 0x22       # saddr=NULL, all sizes
  GLOBAL_STORE = 0x24            # saddr=SGPR, 32-bit
  GLOBAL_STORE_64 = 0x25         # saddr=SGPR 64 or saddr=NULL 32
  GLOBAL_STORE_96 = 0x26         # saddr=SGPR 96 or saddr=NULL 64
  GLOBAL_STORE_128 = 0x27        # saddr=SGPR 128 or saddr=NULL 96
  GLOBAL_STORE_VADDR_128 = 0x28  # saddr=NULL, 128-bit

  # LDS ops on traced SIMD
  LDS_LOAD = 0x29
  LDS_STORE = 0x2b
  LDS_STORE_64 = 0x2c
  LDS_STORE_128 = 0x2e

  # Memory ops on other SIMD (0x5x range)
  OTHER_LDS_LOAD = 0x50
  OTHER_LDS_STORE = 0x51
  OTHER_LDS_STORE_64 = 0x52
  OTHER_LDS_STORE_128 = 0x54
  OTHER_FLAT_LOAD = 0x55
  OTHER_FLAT_STORE = 0x56
  OTHER_FLAT_STORE_64 = 0x57
  OTHER_FLAT_STORE_96 = 0x58
  OTHER_FLAT_STORE_128 = 0x59
  OTHER_GLOBAL_LOAD = 0x5a             # saddr=SGPR, all sizes
  OTHER_GLOBAL_LOAD_VADDR = 0x5b       # saddr=NULL or saddr=SGPR store 32
  OTHER_GLOBAL_STORE_64 = 0x5c         # saddr=SGPR 64 or saddr=NULL 32
  OTHER_GLOBAL_STORE_96 = 0x5d         # saddr=SGPR 96 or saddr=NULL 64
  OTHER_GLOBAL_STORE_128 = 0x5e        # saddr=SGPR 128 or saddr=NULL 96
  OTHER_GLOBAL_STORE_VADDR_128 = 0x5f  # saddr=NULL, 128-bit

  # EXEC-modifying ops (0x7x range)
  SALU_SAVEEXEC = 0x72    # s_*_saveexec_b32/b64
  VALU_CMPX = 0x73        # v_cmpx_*

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
    # Cache fields and precompute extraction info: (name, lo, mask, enum_type)
    cls._fields = {k: v for k, v in cls.__dict__.items() if isinstance(v, BitField) and k != 'encoding'}
    cls._extract_info = [(name, bf.lo, bf.mask(), cls._field_types.get(name)) for name, bf in cls._fields.items()]

  def __init__(self, _time: int = 0, **kwargs):
    """Construct packet from named fields (like assembly instructions)."""
    # Build raw value from encoding + fields
    raw = 0
    if self._encoding:
      bf, pattern = self._encoding
      raw |= pattern << bf.lo
    for name, bf in self._fields.items():
      val = kwargs.get(name, 0)
      if isinstance(val, IntEnum): val = val.value
      raw |= (val & bf.mask()) << bf.lo
    self._raw = raw
    self._time = _time
    # Extract values back (handles enum conversion)
    self._values = {}
    for name, lo, mask, enum_type in self._extract_info:
      val = (raw >> lo) & mask
      if enum_type is not None:
        try: val = enum_type(val)
        except ValueError: pass
      self._values[name] = val

  @classmethod
  def fields(cls) -> dict[str, BitField]:
    return cls._fields

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
    values = {}
    for name, lo, mask, enum_type in cls._extract_info:
      val = (raw >> lo) & mask
      if enum_type is not None:
        try: val = enum_type(val)
        except ValueError: pass
      values[name] = val
    inst._values = values
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

class VALUINST(PacketType):  # exclude: 1 << 2
  encoding = bits[2:0] == 0b011
  delta = bits[5:3]
  flag = bits[6:6]
  wave = bits[11:7]

class VMEMEXEC(PacketType):  # exclude: 1 << 0
  encoding = bits[3:0] == 0b1111
  delta = bits[5:4]
  src: MemSrc = bits[7:6]

class ALUEXEC(PacketType):  # exclude: 1 << 1
  encoding = bits[3:0] == 0b1110
  delta = bits[5:4]
  src: AluSrc = bits[7:6]

class IMMEDIATE(PacketType):  # exclude: 1 << 5
  encoding = bits[3:0] == 0b1101
  delta = bits[6:4]
  wave = bits[11:7]

class IMMEDIATE_MASK(PacketType):  # exclude: 1 << 5
  encoding = bits[4:0] == 0b00100
  delta = bits[7:5]
  mask = bits[23:8]

class WAVERDY(PacketType):  # exclude: 1 << 3
  encoding = bits[4:0] == 0b10100
  delta = bits[7:5]
  mask = bits[23:8]

class TS_DELTA_S8_W3(PacketType):
  encoding = bits[6:0] == 0b0100001
  delta = bits[10:8]
  _padding = bits[63:11]

class WAVEEND(PacketType):  # exclude: 1 << 4
  encoding = bits[4:0] == 0b10101
  delta = bits[7:5]
  flag7 = bits[8:8]
  simd = bits[10:9]
  cu_lo = bits[13:11]
  wave = bits[19:15]
  @property
  def cu(self) -> int: return self.cu_lo | (self.flag7 << 3)

class WAVESTART(PacketType):  # exclude: 1 << 4
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

class WAVEALLOC(PacketType):  # exclude: 1 << 10
  encoding = bits[4:0] == 0b00101
  delta = bits[7:5]
  _padding = bits[19:8]

class TS_DELTA_S5_W3(PacketType):
  encoding = bits[4:0] == 0b00110
  delta = bits[7:5]
  _padding = bits[51:8]

class PERF(PacketType):  # exclude: 1 << 11
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

class EVENT(PacketType):  # exclude: 1 << 7
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
BUDGET = {opcode: pkt_cls.size_nibbles() for opcode, pkt_cls in OPCODE_TO_CLASS.items()}

# Precompute special case opcodes
_TS_DELTA_OR_MARK_OPCODE = next(op for op, cls in OPCODE_TO_CLASS.items() if cls is TS_DELTA_OR_MARK)
_TS_DELTA_SHORT_OPCODE = next(op for op, cls in OPCODE_TO_CLASS.items() if cls is TS_DELTA_SHORT)
_TS_DELTA_OR_MARK_BIT8 = (TS_DELTA_OR_MARK.bit8.lo, TS_DELTA_OR_MARK.bit8.mask())
_TS_DELTA_OR_MARK_BIT9 = (TS_DELTA_OR_MARK.bit9.lo, TS_DELTA_OR_MARK.bit9.mask())

# Combined lookup: opcode -> (pkt_cls, nib_count, delta_lo, delta_mask, special_case)
# special_case: 0=none, 1=TS_DELTA_OR_MARK, 2=TS_DELTA_SHORT
_DECODE_INFO: dict[int, tuple] = {}
for _opcode, _pkt_cls in OPCODE_TO_CLASS.items():
  _delta_field = getattr(_pkt_cls, 'delta', None)
  _delta_lo = _delta_field.lo if _delta_field else 0
  _delta_mask = _delta_field.mask() if _delta_field else 0
  _special = 1 if _opcode == _TS_DELTA_OR_MARK_OPCODE else (2 if _opcode == _TS_DELTA_SHORT_OPCODE else 0)
  _DECODE_INFO[_opcode] = (_pkt_cls, BUDGET[_opcode], _delta_lo, _delta_mask, _special)

# ═══════════════════════════════════════════════════════════════════════════════
# DECODER
# ═══════════════════════════════════════════════════════════════════════════════

def decode(data: bytes) -> list[PacketType]:
  """Decode raw SQTT blob into list of packet instances."""
  packets: list[PacketType] = []
  packets_append = packets.append
  n = len(data)
  reg = 0
  offset = 0
  nib_count = 16
  time = 0
  state_to_opcode = STATE_TO_OPCODE
  decode_info = _DECODE_INFO
  mask64 = (1 << 64) - 1

  while (offset >> 3) < n:
    target = offset + nib_count * 4
    while offset < target and (offset >> 3) < n:
      byte = data[offset >> 3]
      nib = (byte >> (offset & 4)) & 0xF
      reg = ((reg >> 4) | (nib << 60)) & mask64
      offset += 4
    if offset < target: break

    opcode = state_to_opcode[reg & 0xFF]
    pkt_cls, nib_count, delta_lo, delta_mask, special = decode_info[opcode]

    delta = (reg >> delta_lo) & delta_mask

    if special == 1:  # TS_DELTA_OR_MARK
      bit8 = (reg >> _TS_DELTA_OR_MARK_BIT8[0]) & _TS_DELTA_OR_MARK_BIT8[1]
      bit9 = (reg >> _TS_DELTA_OR_MARK_BIT9[0]) & _TS_DELTA_OR_MARK_BIT9[1]
      if bit9 and not bit8: delta = 0
    elif special == 2:  # TS_DELTA_SHORT
      delta = delta + 8

    time += delta
    packets_append(pkt_cls.from_raw(reg, time))

  return packets


