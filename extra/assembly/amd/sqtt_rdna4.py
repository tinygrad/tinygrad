"""SQTT (SQ Thread Trace) packet decoder for RDNA4 (gfx1200) GPUs.

RDNA4 uses layout 4 which has different packet sizes compared to RDNA3 (layout 3).
Key differences from layout 3 (bits: L3→L4):
- TS_DELTA_S8_W3: 64→72 bits (16→18 nibbles)
- TS_DELTA_S5_W2: 48→40 bits (12→10 nibbles)
- WAVEALLOC: 20→24 bits (5→6 nibbles)
- TS_DELTA_S5_W3: 52→56 bits (13→14 nibbles)
- PERF: 28→32 bits (7→8 nibbles)
- TS_DELTA_OR_MARK: 48→64 bits (12→16 nibbles)
"""
from __future__ import annotations
from typing import Iterator
from enum import Enum
from extra.assembly.amd.dsl import BitField, FixedBitField, bits

# ═══════════════════════════════════════════════════════════════════════════════
# FIELD ENUMS (same as base sqtt.py)
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
  SALU = 0x0
  SMEM = 0x1
  JUMP = 0x3
  JUMP_NO = 0x4
  MESSAGE = 0x9
  VALU_TRANS = 0xb
  VALU_64_SHIFT = 0xd
  VALU_MAD64 = 0xe
  VALU_64 = 0xf
  VINTERP = 0x12
  BARRIER = 0x13
  FLAT_LOAD = 0x1c
  FLAT_STORE = 0x1d
  FLAT_STORE_64 = 0x1e
  FLAT_STORE_96 = 0x1f
  FLAT_STORE_128 = 0x20
  GLOBAL_LOAD = 0x21
  GLOBAL_LOAD_VADDR = 0x22
  GLOBAL_STORE = 0x24
  GLOBAL_STORE_64 = 0x25
  GLOBAL_STORE_96 = 0x26
  GLOBAL_STORE_128 = 0x27
  GLOBAL_STORE_VADDR_128 = 0x28
  LDS_LOAD = 0x29
  LDS_STORE = 0x2b
  LDS_STORE_64 = 0x2c
  LDS_STORE_128 = 0x2e
  OTHER_LDS_LOAD = 0x50
  OTHER_LDS_STORE = 0x51
  OTHER_LDS_STORE_64 = 0x52
  OTHER_LDS_STORE_128 = 0x54
  OTHER_FLAT_LOAD = 0x55
  OTHER_FLAT_STORE = 0x56
  OTHER_FLAT_STORE_64 = 0x57
  OTHER_FLAT_STORE_96 = 0x58
  OTHER_FLAT_STORE_128 = 0x59
  OTHER_GLOBAL_LOAD = 0x5a
  OTHER_GLOBAL_LOAD_VADDR = 0x5b
  OTHER_GLOBAL_STORE_64 = 0x5c
  OTHER_GLOBAL_STORE_96 = 0x5d
  OTHER_GLOBAL_STORE_128 = 0x5e
  OTHER_GLOBAL_STORE_VADDR_128 = 0x5f
  SALU_SAVEEXEC = 0x72
  VALU_CMPX = 0x73

# ═══════════════════════════════════════════════════════════════════════════════
# PACKET TYPE BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class PacketType:
  encoding: FixedBitField
  _raw: int
  _time: int
  _size_nibbles: int = 0  # overridden per class

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    cls._fields = {k: v for k, v in cls.__dict__.items() if isinstance(v, BitField)}
    if not hasattr(cls, '_size_nibbles') or cls._size_nibbles == 0:
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
# RDNA4 PACKET TYPE DEFINITIONS
# Layout 4 has different packet sizes and some new packet types
# ═══════════════════════════════════════════════════════════════════════════════

# Type 22 (0x16) - TS_DELTA_OR_MARK: 64 bits in layout 4 (was 48 in layout 3)
class TS_DELTA(PacketType):
  encoding = bits[6:0] == 0b0000001
  _size_nibbles = 16  # 64 bits
  delta = bits[47:12]
  bit8 = bits[8:8]
  bit9 = bits[9:9]
  @property
  def is_marker(self) -> bool: return bool(self.bit9 and not self.bit8)

# Type 0x0f (15) - Short timestamp delta
class TS_SHORT(PacketType):
  encoding = bits[3:0] == 0b1000
  _size_nibbles = 2
  delta = bits[7:4]

# Type 0x10 (16) - NOP/padding
class NOP(PacketType):
  encoding = bits[3:0] == 0b0000
  _size_nibbles = 1
  delta = None  # type: ignore

# Type 0x11 (17) - LAYOUT_HEADER (same encoding, 16 nibbles)
class LAYOUT_HEADER(PacketType):
  encoding = bits[6:0] == 0b0010001
  _size_nibbles = 16
  delta = None  # type: ignore
  layout = bits[12:7]
  simd = bits[14:13]
  group = bits[17:15]
  sel_a = bits[31:28]
  sel_b = bits[36:33]
  flag4 = bits[59:59]

# Type 0x12 (18) - SNAPSHOT
class SNAPSHOT(PacketType):
  encoding = bits[6:0] == 0b1110001
  _size_nibbles = 16
  delta = bits[9:7]
  snap = bits[63:10]

# Type 0x13 (19) - UTILCTR
class UTILCTR(PacketType):
  encoding = bits[6:0] == 0b0110001
  _size_nibbles = 12
  delta = bits[8:7]
  ctr = bits[47:9]

# Type 0x14 (20) - REG
class REG(PacketType):
  encoding = bits[3:0] == 0b1001
  _size_nibbles = 16
  delta = bits[6:4]
  slot = bits[9:7]
  hi_byte = bits[15:8]
  subop = bits[31:16]
  val32 = bits[63:32]

# Type 0x15 (21) - EVENT
class EVENT(PacketType):
  encoding = bits[7:0] == 0b01100001
  _size_nibbles = 6
  delta = bits[10:8]
  event = bits[23:11]

# Type 0x17 (23) - EVENT_BIG
class EVENT_BIG(PacketType):
  encoding = bits[7:0] == 0b11100001
  _size_nibbles = 8
  delta = bits[10:8]
  event = bits[31:11]

# Type 0x18 (24) - WAVE_STATE (TS_WAVE_STATE equivalent)
class WAVE_STATE(PacketType):
  encoding = bits[6:0] == 0b1010001
  _size_nibbles = 6
  delta = bits[15:7]
  coarse = bits[23:16]

# Type 13 (0x0d) - PERF: 32 bits in layout 4 (was 28 in layout 3)
class PERF(PacketType):
  encoding = bits[4:0] == 0b10110
  _size_nibbles = 8  # 32 bits
  delta = bits[7:5]
  arg = bits[27:8]

# Type 0x01 (1) - VALUINST
class VALUINST(PacketType):
  encoding = bits[2:0] == 0b011
  _size_nibbles = 3
  delta = bits[5:3]
  flag = bits[6:6]
  wave = bits[11:7]

# Type 0x02 (2) - VMEMEXEC
class VMEMEXEC(PacketType):
  encoding = bits[3:0] == 0b1111
  _size_nibbles = 2
  delta = bits[5:4]
  src = bits[7:6].enum(MemSrc)

# Type 0x03 (3) - ALUEXEC
class ALUEXEC(PacketType):
  encoding = bits[3:0] == 0b1110
  _size_nibbles = 2
  delta = bits[5:4]
  src = bits[7:6].enum(AluSrc)

# Type 0x04 (4) - IMMEDIATE
class IMMEDIATE(PacketType):
  encoding = bits[3:0] == 0b1101
  _size_nibbles = 3
  delta = bits[6:4]
  wave = bits[11:7]

# Type 0x05 (5) - IMMEDIATE_MASK
class IMMEDIATE_MASK(PacketType):
  encoding = bits[4:0] == 0b00100
  _size_nibbles = 6
  delta = bits[7:5]
  mask = bits[23:8]

# Type 0x06 (6) - WAVERDY
class WAVERDY(PacketType):
  encoding = bits[4:0] == 0b10100
  _size_nibbles = 6
  delta = bits[7:5]
  mask = bits[23:8]

# Type 11 (0x0b) - WAVEALLOC: 24 bits in layout 4 (was 20 in layout 3)
class WAVEALLOC(PacketType):
  encoding = bits[4:0] == 0b00101
  _size_nibbles = 6  # 24 bits
  delta = bits[7:5]

# Type 0x08 (8) - WAVEEND
class WAVEEND(PacketType):
  encoding = bits[4:0] == 0b10101
  _size_nibbles = 5
  delta = bits[7:5]
  flag7 = bits[8:8]
  simd = bits[10:9]
  cu_lo = bits[13:11]
  wave = bits[19:15]
  @property
  def cu(self) -> int: return self.cu_lo | (self.flag7 << 3)

# Type 0x09 (9) - INST (base, can have extension)
class INST(PacketType):
  encoding = bits[2:0] == 0b010
  _size_nibbles = 5
  delta = bits[6:4]
  flag1 = bits[3:3]
  flag2 = bits[7:7]
  wave = bits[12:8]
  op = bits[19:13].enum(InstOp)

# Type 9 (0x09) - WAVESTART: 32 bits (same as layout 3)
class WAVESTART(PacketType):
  encoding = bits[4:0] == 0b01100
  _size_nibbles = 8  # 32 bits (same as layout 3)
  delta = bits[6:5]
  flag7 = bits[7:7]
  simd = bits[9:8]
  cu_lo = bits[12:10]
  wave = bits[17:13]
  id7 = bits[31:18]
  @property
  def cu(self) -> int: return self.cu_lo | (self.flag7 << 3)

# Type 10 (0x0a) - TS_DELTA_S5_W2: 40 bits in layout 4 (was 48 in layout 3)
class TS_DELTA_S5_W2(PacketType):
  encoding = bits[4:0] == 0b11100
  _size_nibbles = 10  # 40 bits
  delta = bits[6:5]

# Type 12 (0x0c) - TS_DELTA_S5_W3: 56 bits in layout 4 (was 52 in layout 3)
class TS_DELTA_S5_W3(PacketType):
  encoding = bits[4:0] == 0b00110
  _size_nibbles = 14  # 56 bits
  delta = bits[7:5]

# Type 7 (0x07) - TS_DELTA_S8_W3: 72 bits in layout 4 (was 64 in layout 3)
class TS_DELTA_S8_W3(PacketType):
  encoding = bits[6:0] == 0b0100001
  _size_nibbles = 18  # 72 bits
  delta = bits[10:8]

# All packet types in encoding priority order
PACKET_TYPES: list[type[PacketType]] = [
  EVENT, EVENT_BIG,
  TS_DELTA_S8_W3, WAVE_STATE, SNAPSHOT, TS_DELTA, LAYOUT_HEADER, UTILCTR,
  IMMEDIATE_MASK, WAVERDY, WAVEEND, WAVESTART, TS_DELTA_S5_W2, WAVEALLOC, TS_DELTA_S5_W3, PERF,
  VMEMEXEC, ALUEXEC, IMMEDIATE, TS_SHORT, REG,
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
_TS_DELTA_OPCODE = next(op for op, cls in OPCODE_TO_CLASS.items() if cls is TS_DELTA)
_TS_SHORT_OPCODE = next(op for op, cls in OPCODE_TO_CLASS.items() if cls is TS_SHORT)
_INST_OPCODE = next(op for op, cls in OPCODE_TO_CLASS.items() if cls is INST)

# Combined lookup: opcode -> (pkt_cls, nib_count, delta_lo, delta_mask, special_case)
# special_case: 0=none, 1=TS_DELTA marker check, 2=TS_SHORT +8, 3=INST extension
_DECODE_INFO: dict[int, tuple] = {}
for _opcode, _pkt_cls in OPCODE_TO_CLASS.items():
  _delta_field = getattr(_pkt_cls, 'delta', None)
  _delta_lo = _delta_field.lo if _delta_field else 0
  _delta_mask = _delta_field.mask if _delta_field else 0
  _special = 1 if _opcode == _TS_DELTA_OPCODE else (2 if _opcode == _TS_SHORT_OPCODE else (3 if _opcode == _INST_OPCODE else 0))
  _DECODE_INFO[_opcode] = (_pkt_cls, _pkt_cls._size_nibbles, _delta_lo, _delta_mask, _special)

# ═══════════════════════════════════════════════════════════════════════════════
# DECODER
# ═══════════════════════════════════════════════════════════════════════════════

def decode(data: bytes) -> Iterator[PacketType]:
  """Decode raw SQTT blob for RDNA4/gfx1200, yielding packet instances."""
  n, reg, pos, nib_off, nib_count, time = len(data), 0, 0, 0, 16, 0
  prev_was_inst = False

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

    # RDNA4: INST packets can have 8-nibble extension if next nibble has bit 0 set
    if prev_was_inst and (reg & 1):
      nib_count = 8  # consume extension
      prev_was_inst = False
      continue

    opcode = STATE_TO_OPCODE[reg & 0xFF]
    pkt_cls, nib_count, delta_lo, delta_mask, special = _DECODE_INFO[opcode]
    delta = (reg >> delta_lo) & delta_mask
    if special == 1 and (reg >> 9) & 1 and not (reg >> 8) & 1: delta = 0  # TS_DELTA marker
    elif special == 2: delta += 8  # TS_SHORT
    time += delta

    # Track INST for extension check
    prev_was_inst = (special == 3)

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
    fields = " ".join(f"{k}=0x{getattr(p, k):x}" if k in {'snap', 'val32', 'ext_data'} else f"{k}={getattr(p, k)}"
                      for k in p._fields if not k.startswith('_') and k not in {'delta', 'encoding'})
  else: fields = ""
  return f"{p._time:8}: {colored(f'{name:18}', PACKET_COLORS.get(name, 'white'))} {fields}"

def print_packets(packets) -> None:
  skip = {"NOP", "TS_SHORT", "WAVE_STATE", "TS_DELTA", "TS_DELTA_S5_W2", "TS_DELTA_S5_W3", "TS_DELTA_S8_W3", "REG", "EVENT"}
  for p in packets:
    if type(p).__name__ not in skip: print(format_packet(p))

if __name__ == "__main__":
  import sys, pickle
  if len(sys.argv) < 2:
    print("Usage: python sqtt_rdna4.py <pkl_file>")
    sys.exit(1)
  with open(sys.argv[1], "rb") as f:
    data = pickle.load(f)
  sqtt_events = [e for e in data if type(e).__name__ == "ProfileSQTTEvent"]
  for i, event in enumerate(sqtt_events):
    print(f"\n=== event {i} ===")
    print_packets(decode(event.blob))
