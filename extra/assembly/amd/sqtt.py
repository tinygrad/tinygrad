"""SQTT (SQ Thread Trace) packet encoder and decoder for AMD GPUs.

This module provides encoding and decoding of raw SQTT byte streams.
The format is nibble-based with variable-width packets determined by a state machine.
Uses BitField infrastructure from dsl.py, similar to GPU instruction encoding.
"""
from __future__ import annotations
from typing import Iterator
from enum import Enum
from extra.assembly.amd.dsl import BitField, FixedBitField, bits

# ═══════════════════════════════════════════════════════════════════════════════
# FIELD ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class MemSrc(Enum):
  LDS = 0
  LDS_ALT = 1
  VMEM = 2
  VMEM_ALT = 3

class AluSrc(Enum):
  NONE = 0
  SALU = 1
  VALU = 2
  VALU_SALU = 3

class InstOp(Enum):
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
  encoding: FixedBitField
  _raw: int
  _time: int

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    cls._fields = {k: v for k, v in cls.__dict__.items() if isinstance(v, BitField)}
    cls._size_nibbles = ((max((f.hi for f in cls._fields.values()), default=0) + 4) // 4)

  @classmethod
  def from_raw(cls, raw: int, time: int = 0):
    inst = object.__new__(cls)
    inst._raw, inst._time = raw, time
    return inst

  def __repr__(self) -> str:
    fields_str = ", ".join(f"{k}={getattr(self, k)}" for k in self._fields if not k.startswith('_'))
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
  src = bits[7:6].enum(MemSrc)

class ALUEXEC(PacketType):  # exclude: 1 << 1
  encoding = bits[3:0] == 0b1110
  delta = bits[5:4]
  src = bits[7:6].enum(AluSrc)

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
  op = bits[19:13].enum(InstOp)

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

def _build_state_table() -> tuple[bytes, dict[int, type[PacketType]]]:
  table = [len(PACKET_TYPES) - 1] * 256  # default to NOP
  opcode_to_class: dict[int, type[PacketType]] = {i: cls for i, cls in enumerate(PACKET_TYPES)}

  for byte_val in range(256):
    for opcode, pkt_cls in enumerate(PACKET_TYPES):
      if (byte_val & pkt_cls.encoding.mask) == pkt_cls.encoding.default:
        table[byte_val] = opcode
        break

  return bytes(table), opcode_to_class

STATE_TO_OPCODE, OPCODE_TO_CLASS = _build_state_table()

# Precompute special case opcodes
_TS_DELTA_OR_MARK_OPCODE = next(op for op, cls in OPCODE_TO_CLASS.items() if cls is TS_DELTA_OR_MARK)
_TS_DELTA_SHORT_OPCODE = next(op for op, cls in OPCODE_TO_CLASS.items() if cls is TS_DELTA_SHORT)

# Combined lookup: opcode -> (pkt_cls, nib_count, delta_lo, delta_mask, special_case)
# special_case: 0=none, 1=TS_DELTA_OR_MARK, 2=TS_DELTA_SHORT
_DECODE_INFO: dict[int, tuple] = {}
for _opcode, _pkt_cls in OPCODE_TO_CLASS.items():
  _delta_field = getattr(_pkt_cls, 'delta', None)
  _delta_lo = _delta_field.lo if _delta_field else 0
  _delta_mask = _delta_field.mask if _delta_field else 0
  _special = 1 if _opcode == _TS_DELTA_OR_MARK_OPCODE else (2 if _opcode == _TS_DELTA_SHORT_OPCODE else 0)
  _DECODE_INFO[_opcode] = (_pkt_cls, _pkt_cls._size_nibbles, _delta_lo, _delta_mask, _special)

# ═══════════════════════════════════════════════════════════════════════════════
# DECODER
# ═══════════════════════════════════════════════════════════════════════════════

def decode(data: bytes) -> Iterator[PacketType]:
  """Decode raw SQTT blob, yielding packet instances."""
  n, reg, pos, nib_off, nib_count, time = len(data), 0, 0, 0, 16, 0

  while pos + ((nib_count + nib_off + 1) >> 1) <= n:
    need = nib_count - nib_off
    # 1. if unaligned, read high nibble to align
    if nib_off: reg, pos = (reg >> 4) | ((data[pos] >> 4) << 60), pos + 1
    # 2. read all full bytes at once
    if (byte_count := need >> 1):
      chunk = int.from_bytes(data[pos:pos + byte_count], 'little')
      reg, pos = (reg >> (byte_count * 8)) | (chunk << (64 - byte_count * 8)), pos + byte_count
    # 3. if odd, read low nibble
    if (nib_off := need & 1): reg = (reg >> 4) | ((data[pos] & 0xF) << 60)

    opcode = STATE_TO_OPCODE[reg & 0xFF]
    pkt_cls, nib_count, delta_lo, delta_mask, special = _DECODE_INFO[opcode]
    delta = (reg >> delta_lo) & delta_mask
    if special == 1 and (reg >> 9) & 1 and not (reg >> 8) & 1: delta = 0  # TS_DELTA_OR_MARK marker
    elif special == 2: delta += 8  # TS_DELTA_SHORT
    time += delta
    yield pkt_cls.from_raw(reg, time)

# ═══════════════════════════════════════════════════════════════════════════════
# PRINTER
# ═══════════════════════════════════════════════════════════════════════════════

PACKET_COLORS = {
  "INST": "WHITE", "VALUINST": "BLACK", "VMEMEXEC": "yellow", "ALUEXEC": "yellow",
  "IMMEDIATE": "YELLOW", "IMMEDIATE_MASK": "YELLOW", "WAVERDY": "cyan", "WAVEALLOC": "cyan",
  "WAVEEND": "blue", "WAVESTART": "blue", "PERF": "magenta", "EVENT": "red", "EVENT_BIG": "red",
  "REG": "green", "LAYOUT_HEADER": "white", "SNAPSHOT": "white", "UTILCTR": "green",
}

def format_packet(p) -> str:
  from tinygrad.helpers import colored
  name = type(p).__name__
  if isinstance(p, INST):
    op_name = p.op.name if isinstance(p.op, InstOp) else f"0x{p.op:02x}"
    fields = f"wave={p.wave} op={op_name}" + (" flag1" if p.flag1 else "") + (" flag2" if p.flag2 else "")
  elif isinstance(p, VALUINST): fields = f"wave={p.wave}" + (" flag" if p.flag else "")
  elif isinstance(p, ALUEXEC): fields = f"src={p.src.name if isinstance(p.src, AluSrc) else p.src}"
  elif isinstance(p, VMEMEXEC): fields = f"src={p.src.name if isinstance(p.src, MemSrc) else p.src}"
  elif isinstance(p, (WAVESTART, WAVEEND)): fields = f"wave={p.wave} simd={p.simd} cu={p.cu}"
  elif hasattr(p, '_fields'):
    fields = " ".join(f"{k}=0x{getattr(p, k):x}" if k in {'snap', 'val32'} else f"{k}={getattr(p, k)}"
                      for k in p._fields if not k.startswith('_') and k not in {'delta', 'encoding'})
  else: fields = ""
  return f"{p._time:8}: {colored(f'{name:18}', PACKET_COLORS.get(name, 'white'))} {fields}"

def print_packets(packets) -> None:
  skip = {"NOP", "TS_DELTA_SHORT", "TS_WAVE_STATE", "TS_DELTA_OR_MARK", "TS_DELTA_S5_W2", "TS_DELTA_S5_W3", "TS_DELTA_S8_W3", "REG", "EVENT"}
  for p in packets:
    if type(p).__name__ not in skip: print(format_packet(p))

if __name__ == "__main__":
  import sys, pickle
  if len(sys.argv) < 2:
    print("Usage: python sqtt.py <pkl_file>")
    sys.exit(1)
  with open(sys.argv[1], "rb") as f:
    data = pickle.load(f)
  sqtt_events = [e for e in data if type(e).__name__ == "ProfileSQTTEvent"]
  for i, event in enumerate(sqtt_events):
    print(f"\n=== event {i} ===")
    print_packets(decode(event.blob))
