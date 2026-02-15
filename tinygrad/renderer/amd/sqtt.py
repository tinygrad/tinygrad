"""SQTT (SQ Thread Trace) packet encoder and decoder for AMD GPUs.

This module provides encoding and decoding of raw SQTT byte streams.
The format is nibble-based with variable-width packets determined by a state machine.
Uses BitField infrastructure from dsl.py, similar to GPU instruction encoding.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator
from enum import Enum
from tinygrad.renderer.amd.dsl import BitField, FixedBitField, Inst, bits
from tinygrad.runtime.autogen.amd.rdna3.ins import SOPP, s_endpgm
from tinygrad.runtime.autogen.amd.rdna3.enum import SOPPOp

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
  """SQTT instruction operation types for RDNA3 (gfx1100).

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

class InstOpRDNA4(Enum):
  """SQTT instruction operation types for RDNA4 (gfx1200). Different encoding from RDNA3."""
  # TODO: we need to do discovery of all of these from instructions
  SALU = 0x0
  SMEM = 0x1
  UNK_02 = 0x2
  JUMP_NO = 0x4
  UNK_06 = 0x6
  VMEM = 0x10
  UNK_11 = 0x11
  VINTERP = 0x12
  UNK_14 = 0x14
  OTHER_VMEM = 0x5e
  UNK_60 = 0x60

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
    cls._fields = {k: v for k, v in cls.__dict__.items() if isinstance(v, BitField)}  # type: ignore[attr-defined]
    cls._size_nibbles = ((max((f.hi for f in cls._fields.values()), default=0) + 4) // 4)  # type: ignore[attr-defined]

  @classmethod
  def from_raw(cls, raw: int, time: int = 0):
    inst = object.__new__(cls)
    inst._raw, inst._time = raw, time
    return inst

  def __repr__(self) -> str:
    fields_str = ", ".join(f"{k}={getattr(self, k)}" for k in self._fields if not k.startswith('_') and k != 'encoding')  # type: ignore[attr-defined]
    return f"{self.__class__.__name__}({fields_str})"

# ═══════════════════════════════════════════════════════════════════════════════
# TS PACKET TYPE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TS_DELTA_S8_W3(PacketType):
  encoding = bits[6:0] == 0b0100001
  delta = bits[10:8]
  _padding = bits[63:11]

class TS_DELTA_S8_W3_RDNA4(PacketType):  # Layout 4: 64->72 bits
  encoding = bits[6:0] == 0b0100001
  delta = bits[10:8]
  _padding = bits[71:11]

class TS_DELTA_S5_W3(PacketType):
  encoding = bits[4:0] == 0b00110
  delta = bits[7:5]
  _padding = bits[51:8]

class TS_DELTA_S5_W3_RDNA4(PacketType):  # Layout 4: 52->56 bits
  encoding = bits[4:0] == 0b00110
  delta = bits[9:7]
  _padding = bits[55:10]

class TS_DELTA_SHORT(PacketType):
  encoding = bits[3:0] == 0b1000
  delta = bits[7:4]

class TS_DELTA_OR_MARK(PacketType):
  encoding = bits[6:0] == 0b0000001
  delta = bits[47:12]
  bit8 = bits[8:8]
  bit9 = bits[9:9]
  @property
  def is_marker(self) -> bool: return bool(self.bit9 and not self.bit8)

class TS_DELTA_OR_MARK_RDNA4(PacketType):  # Layout 4: 48->64 bits
  encoding = bits[6:0] == 0b0000001
  delta = bits[63:12]
  bit7 = bits[7:7]
  bit8 = bits[8:8]
  bit9 = bits[9:9]
  @property
  def is_marker(self) -> bool: return bool((self.bit9 and not self.bit8) or self.bit7)

class TS_DELTA_S5_W2(PacketType):
  encoding = bits[4:0] == 0b11100
  delta = bits[6:5]
  _padding = bits[47:7]

class TS_DELTA_S5_W2_RDNA4(PacketType):  # Layout 4: 48->40 bits
  encoding = bits[4:0] == 0b11100
  delta = bits[6:5]
  _padding = bits[39:7]

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

class WAVESTART_RDNA4(PacketType):  # Layout 4 has wave field at different position
  encoding = bits[4:0] == 0b01100
  delta = bits[6:5]
  flag7 = bits[7:7]
  simd = bits[9:8]
  cu_lo = bits[12:10]
  wave = bits[19:15]
  id7 = bits[31:20]
  @property
  def cu(self) -> int: return self.cu_lo | (self.flag7 << 3)

class WAVEALLOC(PacketType):  # exclude: 1 << 10
  encoding = bits[4:0] == 0b00101
  delta = bits[7:5]
  _padding = bits[19:8]

class WAVEALLOC_RDNA4(PacketType):  # Layout 4: 20->24 bits
  encoding = bits[4:0] == 0b00101
  delta = bits[7:5]
  _padding = bits[23:8]

class PERF(PacketType):  # exclude: 1 << 11
  encoding = bits[4:0] == 0b10110
  delta = bits[7:5]
  arg = bits[27:8]

class PERF_RDNA4(PacketType):  # Layout 4: 28->32 bits
  encoding = bits[4:0] == 0b10110
  delta = bits[9:7]
  arg = bits[31:10]

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

class INST_RDNA4(PacketType):  # Layout 4: different delta position and InstOp encoding
  encoding = bits[2:0] == 0b010
  delta = bits[5:3]
  flag1 = bits[6:6]
  flag2 = bits[7:7]
  wave = bits[12:8]
  op = bits[19:13].enum(InstOpRDNA4)

class UTILCTR(PacketType):
  encoding = bits[6:0] == 0b0110001
  delta = bits[8:7]
  ctr = bits[47:9]

# Packet types with rocprof type IDs as keys
PACKET_TYPES_RDNA3: dict[int, type[PacketType]] = {
  1: VALUINST, 2: VMEMEXEC, 3: ALUEXEC, 4: IMMEDIATE, 5: IMMEDIATE_MASK, 6: WAVERDY, 7: TS_DELTA_S8_W3, 8: WAVEEND,
  9: WAVESTART, 10: TS_DELTA_S5_W2, 11: WAVEALLOC, 12: TS_DELTA_S5_W3, 13: PERF, 14: UTILCTR, 15: TS_DELTA_SHORT,
  16: NOP, 17: TS_WAVE_STATE, 18: EVENT, 19: EVENT_BIG, 20: REG, 21: SNAPSHOT, 22: TS_DELTA_OR_MARK, 23: LAYOUT_HEADER, 24: INST,
}
PACKET_TYPES_RDNA4: dict[int, type[PacketType]] = {
  **PACKET_TYPES_RDNA3,
  7: TS_DELTA_S8_W3_RDNA4, 9: WAVESTART_RDNA4, 10: TS_DELTA_S5_W2_RDNA4, 11: WAVEALLOC_RDNA4,
  12: TS_DELTA_S5_W3_RDNA4, 13: PERF_RDNA4, 22: TS_DELTA_OR_MARK_RDNA4, 24: INST_RDNA4,
}

# ═══════════════════════════════════════════════════════════════════════════════
# CDNA PACKET TYPE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class CDNA_DELTA(PacketType):
  """pkt_fmt=0: 16-bit timestamp delta packet"""
  encoding = bits[3:0] == 0
  delta = bits[11:4]      # (data >> 4) & 0xff
  unk_0 = bits[12:12]     # (data >> 0xc) & 1
  unk_1 = bits[15:13]     # (data >> 0xd)

class CDNA_TIMESTAMP(PacketType):
  """pkt_fmt=1: 64-bit timestamp packet (case 0x0)"""
  encoding = bits[3:0] == 1
  unk_0 = bits[15:4]
  timestamp = bits[63:16]   # stored as (data_word >> 0x10) in low 46 bits of local_58

class CDNA_PKT_2(PacketType):
  """pkt_fmt=2: 64-bit packet (case 0x4)"""
  encoding = bits[3:0] == 2
  unk_0 = bits[6:5]       # (data >> 5) & 3
  unk_1 = bits[7:7]       # (data >> 7) + 1 & 1
  unk_padding = bits[63:8]

class CDNA_WAVESTART(PacketType):
  """pkt_fmt=3: 32-bit WAVESTART packet (case 0x8)"""
  encoding = bits[3:0] == 3
  unk_0 = bits[5:5]       # (data >> 5) & 1
  unk_1 = bits[9:6]       # (data >> 6) & 0xf
  wave = bits[13:10]      # (data >> 10) & 0xf
  simd = bits[15:14]      # (data >> 0xe) & 3
  cu = bits[17:16]        # (data >> 0x10) & 3
  unk_5 = bits[19:18]     # (data >> 0x12) & 3
  unk_6 = bits[28:22]     # (data >> 0x16) & 0x7f
  unk_padding = bits[31:29]

class CDNA_PKT_4(PacketType):
  """pkt_fmt=4: 16-bit packet (case 0xc, same as 0x8/0x14)"""
  encoding = bits[3:0] == 4
  unk_0 = bits[5:5]       # (data_word >> 5) & 1
  unk_1 = bits[9:6]       # (data_word >> 6) & 0xf
  unk_2 = bits[13:10]     # (data_word >> 10) & 0xf
  unk_3 = bits[15:14]     # (data_word >> 0xe)

class CDNA_PKT_5(PacketType):
  """pkt_fmt=5: 48-bit packet (case 0x10)"""
  encoding = bits[3:0] == 5
  unk_0 = bits[6:5]       # (data >> 5) & 3
  unk_1 = bits[7:7]       # (data >> 7) + 1 & 1
  unk_2 = bits[15:9]      # (data >> 9) & 0x7f
  unk_padding = bits[47:16]

class CDNA_WAVEEND(PacketType):
  """pkt_fmt=6: 16-bit WAVEEND packet (case 0x14, same as 0x8/0xc)"""
  encoding = bits[3:0] == 6
  unk_0 = bits[5:5]       # (data_word >> 5) & 1
  unk_1 = bits[9:6]       # (data_word >> 6) & 0xf
  wave = bits[13:10]      # (data_word >> 10) & 0xf
  simd = bits[15:14]      # (data_word >> 0xe)

class CDNA_EXEC(PacketType):
  """pkt_fmt=10: 16-bit EXEC packet (case 0x24)"""
  encoding = bits[3:0] == 10
  unk_0 = bits[8:5]       # (data_word >> 5) & 0xf
  unk_1 = bits[10:9]      # (data_word >> 9) & 3
  unk_2 = bits[15:11]     # (data_word >> 0xb)

class CDNA_PKT_11(PacketType):
  """pkt_fmt=11: 64-bit packet (case 0x28)"""
  encoding = bits[3:0] == 11
  unk_0 = bits[8:5]       # (data_word >> 5) & 0xf
  unk_1 = bits[10:9]      # (data_word >> 9) & 3
  unk_2 = bits[15:15]     # (data_word >> 0xf) & 1
  unk_padding = bits[63:16]

class CDNA_INST(PacketType):
  """pkt_fmt=13: 32-bit INST packet (case 0x30)"""
  encoding = bits[3:0] == 13
  unk_0 = bits[6:5]       # (data >> 5) & 3
  unk_1 = bits[9:8]       # (data >> 8) & 3
  unk_2 = bits[11:10]     # (data >> 10) & 3
  unk_3 = bits[13:12]     # (data >> 0xc) & 3
  unk_4 = bits[15:14]     # (data >> 0xe) & 3
  unk_5 = bits[19:18]     # (data >> 0x12) & 3
  unk_6 = bits[21:20]     # (data >> 0x14) & 3
  unk_7 = bits[23:22]     # (data >> 0x16) & 3
  unk_8 = bits[25:24]     # (data >> 0x18) & 3
  unk_9 = bits[27:26]     # (data >> 0x1a) & 3
  unk_padding = bits[31:28]

class CDNA_PKT_14(PacketType):
  """pkt_fmt=14: 64-bit packet (case 0x34)"""
  encoding = bits[3:0] == 14
  unk_0 = bits[5:5]       # (data >> 5) & 1
  unk_1 = bits[9:6]       # (data >> 6) & 0xf
  unk_2 = bits[11:10]     # (data >> 10) & 3
  unk_3 = bits[24:12]     # (data >> 0xc) & 0x1fff
  unk_4 = bits[37:25]     # (data >> 0x19) & 0x1fff
  unk_5 = bits[50:38]     # (data >> 0x26) & 0x1fff
  unk_6 = bits[51:51]     # (data >> 0x33) & 1
  unk_padding = bits[63:52]

class CDNA_PKT_7(PacketType):
  """pkt_fmt=7: 16-bit packet"""
  encoding = bits[3:0] == 7
  unk_padding = bits[15:4]

class CDNA_PKT_8(PacketType):
  """pkt_fmt=8: 16-bit packet"""
  encoding = bits[3:0] == 8
  unk_padding = bits[15:4]

class CDNA_PKT_9(PacketType):
  """pkt_fmt=9: 16-bit packet"""
  encoding = bits[3:0] == 9
  unk_padding = bits[15:4]

class CDNA_PKT_12(PacketType):
  """pkt_fmt=12: 48-bit packet"""
  encoding = bits[3:0] == 12
  unk_padding = bits[47:4]

class CDNA_PKT_15(PacketType):
  """pkt_fmt=15: 48-bit packet (case 0x38, same as 0x10)"""
  encoding = bits[3:0] == 15
  unk_0 = bits[6:5]       # (data >> 5) & 3
  unk_1 = bits[7:7]       # (data >> 7) + 1 & 1
  unk_2 = bits[15:9]      # (data >> 9) & 0x7f
  unk_padding = bits[47:16]

PACKET_TYPES_CDNA: dict[int, type[PacketType]] = {
  0: CDNA_DELTA, 1: CDNA_TIMESTAMP, 2: CDNA_PKT_2, 3: CDNA_WAVESTART, 4: CDNA_PKT_4, 5: CDNA_PKT_5, 6: CDNA_WAVEEND,
  7: CDNA_PKT_7, 8: CDNA_PKT_8, 9: CDNA_PKT_9, 10: CDNA_EXEC, 11: CDNA_PKT_11, 12: CDNA_PKT_12,
  13: CDNA_INST, 14: CDNA_PKT_14, 15: CDNA_PKT_15,
}

# ═══════════════════════════════════════════════════════════════════════════════
# DECODER
# ═══════════════════════════════════════════════════════════════════════════════

def _build_decode_tables(packet_types: dict[int, type[PacketType]]) -> tuple[dict[int, tuple], bytes]:
  # Build state table: byte -> opcode. Sort by mask specificity (more bits first), NOP last
  sorted_types = sorted(packet_types.items(), key=lambda x: (-bin(x[1].encoding.mask).count('1'), x[0] == 16))
  state_table = bytes(next((op for op, cls in sorted_types if (b & cls.encoding.mask) == cls.encoding.default), 16) for b in range(256))
  # Build decode info: opcode -> (pkt_cls, nib_count, delta_lo, delta_mask, special_case)
  # special_case: 0=none, 1=TS_DELTA_OR_MARK (check is_marker), 2=TS_DELTA_SHORT (add 8), 3=CDNA_DELTA (*4), 4=CDNA_TIMESTAMP (absolute)
  _special = {TS_DELTA_OR_MARK: 1, TS_DELTA_OR_MARK_RDNA4: 1, TS_DELTA_SHORT: 2, CDNA_DELTA: 3, CDNA_TIMESTAMP: 4}
  decode_info = {}
  for opcode, pkt_cls in packet_types.items():
    delta_field = getattr(pkt_cls, 'delta', None)
    special = _special.get(pkt_cls, 0)
    decode_info[opcode] = (pkt_cls, pkt_cls._size_nibbles, delta_field.lo if delta_field else 0, delta_field.mask if delta_field else 0, special)  # type: ignore[attr-defined]
  return decode_info, state_table

_DECODE_INFO_RDNA3, _STATE_TABLE_RDNA3 = _build_decode_tables(PACKET_TYPES_RDNA3)
_DECODE_INFO_RDNA4, _STATE_TABLE_RDNA4 = _build_decode_tables(PACKET_TYPES_RDNA4)
_DECODE_INFO_CDNA, _STATE_TABLE_CDNA = _build_decode_tables(PACKET_TYPES_CDNA)

def decode(data: bytes) -> Iterator[PacketType]:
  """Decode raw SQTT blob, yielding packet instances. Auto-detects RDNA (layout 3/4) vs CDNA."""
  n, reg, pos, nib_off, nib_count, time, ts_offset = len(data), 0, 0, 0, 16, 0, None
  decode_info, state_table = _DECODE_INFO_RDNA3, _STATE_TABLE_RDNA3  # start RDNA3, auto-detect switches if needed

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

    opcode = state_table[reg & 0xFF]
    pkt_cls, nib_count, delta_lo, delta_mask, special = decode_info[opcode]
    delta = (reg >> delta_lo) & delta_mask
    if special == 1:  # TS_DELTA_OR_MARK
      pkt = pkt_cls.from_raw(reg, 0)  # create packet to check is_marker
      if pkt.is_marker: delta = 0
    elif special == 2: delta += 8  # TS_DELTA_SHORT
    elif special == 3: delta *= 4  # CDNA_DELTA
    elif special == 4:  # CDNA_TIMESTAMP (absolute timestamp anchoring)
      if (reg >> 4) & 0xfff == 0:  # unk_0 == 0 means absolute timestamp
        abs_ts = reg >> 16
        if ts_offset is None: ts_offset = abs_ts - time
        else: time = ((abs_ts - ts_offset) & ~3) - 4
      delta = 0
    time += delta
    pkt = pkt_cls.from_raw(reg, time)
    # auto-detect: first packet is always LAYOUT_HEADER (RDNA layout 3/4) or misdetected (CDNA)
    if pkt_cls is LAYOUT_HEADER:
      if pkt.layout == 4: decode_info, state_table = _DECODE_INFO_RDNA4, _STATE_TABLE_RDNA4
      elif pkt.layout != 3:  # not a real LAYOUT_HEADER — switch to CDNA and re-decode first packet
        decode_info, state_table = _DECODE_INFO_CDNA, _STATE_TABLE_CDNA
        opcode = state_table[reg & 0xFF]
        pkt_cls, nib_count, delta_lo, delta_mask, special = decode_info[opcode]
        if special == 4 and (reg >> 4) & 0xfff == 0:  # CDNA_TIMESTAMP absolute
          ts_offset = (reg >> 16) - time
        pkt = pkt_cls.from_raw(reg, time)
    yield pkt

# ═══════════════════════════════════════════════════════════════════════════════
# MAPPER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class InstructionInfo:
  pc: int
  wave: int
  inst: Inst

def map_insts(data:bytes, lib:bytes, target:int) -> Iterator[tuple[PacketType, InstructionInfo|None]]:
  """maps SQTT packets to instructions, yields (packet, instruction_info or None)"""
  # map pcs to insts
  from tinygrad.viz.serve import amd_decode
  pc_map = amd_decode(lib, target)

  wave_pc:dict[int, int] = {}
  # only processing packets on one [CU, SIMD] unit
  def simd_select(p) -> bool: return getattr(p, "cu", 0) == 0 and getattr(p, "simd", 0) == 0
  for p in decode(data):
    if not simd_select(p): continue
    if isinstance(p, WAVESTART):
      assert p.wave not in wave_pc, "only one inflight wave per unit"
      wave_pc[p.wave] = next(iter(pc_map))
      continue
    if isinstance(p, WAVEEND):
      pc = wave_pc.pop(p.wave)
      yield (p, InstructionInfo(pc, p.wave, s_endpgm()))
      continue
    # skip OTHER_ instructions, they don't belong to this unit
    if isinstance(p, INST) and p.op.name.startswith("OTHER_"): continue
    if isinstance(p, IMMEDIATE_MASK):
      # immediate mask may yield multiple times per packet
      for wave in range(16):
        if p.mask & (1 << wave):
          inst = pc_map[pc:=wave_pc[wave]]
          # can this assert be more strict?
          assert isinstance(inst, SOPP), f"IMMEDIATE_MASK packet must map to SOPP, got {inst}"
          wave_pc[wave] += inst.size()
          yield (p, InstructionInfo(pc, wave, inst))
      continue
    if isinstance(p, (VALUINST, INST, IMMEDIATE)):
      inst = pc_map[pc:=wave_pc[p.wave]]
      # s_delay_alu doesn't get a packet?
      if isinstance(inst, SOPP) and inst.op in {SOPPOp.S_DELAY_ALU}:
        wave_pc[p.wave] += inst.size()
        inst = pc_map[pc:=wave_pc[p.wave]]
      # identify a branch instruction, only used for asserts
      is_branch = isinstance(inst, SOPP) and "BRANCH" in inst.op_name
      if is_branch: assert isinstance(p, INST) and p.op in {InstOp.JUMP_NO, InstOp.JUMP}, f"branch can only be folowed by jump packets, got {p}"
      # JUMP handling
      if isinstance(p, INST) and p.op is InstOp.JUMP:
        assert is_branch, f"JUMP packet must map to a branch instruction, got {inst}"
        x = inst.simm16 & 0xffff
        wave_pc[p.wave] += inst.size() + (x - 0x10000 if x & 0x8000 else x)*4
      else:
        if is_branch: assert inst.op != SOPPOp.S_BRANCH, f"S_BRANCH must have a JUMP packet, got {p}"
        wave_pc[p.wave] += inst.size()
      yield (p, InstructionInfo(pc, p.wave, inst))
      continue
    # for all other packets (VMEMEXEC, ALUEXEC, etc.), yield with None
    yield (p, None)

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
  if isinstance(p, (INST, INST_RDNA4)):
    op_name = p.op.name if isinstance(p.op, (InstOp, InstOpRDNA4)) else f"0x{p.op:02x}"
    fields = f"wave={p.wave} op={op_name}" + (" flag1" if p.flag1 else "") + (" flag2" if p.flag2 else "")
  elif isinstance(p, VALUINST): fields = f"wave={p.wave}" + (" flag" if p.flag else "")
  elif isinstance(p, ALUEXEC): fields = f"src={p.src.name if isinstance(p.src, AluSrc) else p.src}"
  elif isinstance(p, VMEMEXEC): fields = f"src={p.src.name if isinstance(p.src, MemSrc) else p.src}"
  elif isinstance(p, (WAVESTART, WAVESTART_RDNA4, WAVEEND)): fields = f"wave={p.wave} simd={p.simd} cu={p.cu}"
  elif hasattr(p, '_fields'):
    filt = {'delta', 'encoding'} if not isinstance(p, (TS_DELTA_OR_MARK, TS_DELTA_OR_MARK_RDNA4)) else {'encoding'}
    fields = " ".join(f"{k}=0x{getattr(p, k):x}" if k in {'snap', 'val32'} else f"{k}={getattr(p, k)}"
                      for k in p._fields if not k.startswith('_') and k not in filt)
  else: fields = ""
  return f"{p._time:8}: {colored(f'{name:18}', PACKET_COLORS.get(name.replace('_RDNA4', ''), 'white'))} {fields}"

def print_packets(packets) -> None:
  from tinygrad.helpers import getenv
  skip = {"NOP", "TS_DELTA_SHORT", "TS_WAVE_STATE", "TS_DELTA_OR_MARK",
          "TS_DELTA_S5_W2", "TS_DELTA_S5_W3", "TS_DELTA_S8_W3", "REG", "EVENT"} if not getenv("NOSKIP") else {"NOP"}
  for p in packets:
    if type(p).__name__.replace("_RDNA4", "") not in skip: print(format_packet(p))

if __name__ == "__main__":
  import sys, pickle
  if len(sys.argv) < 2:
    print("Usage: python sqtt.py <pkl_file>")
    sys.exit(1)
  with open(sys.argv[1], "rb") as f:
    data = pickle.load(f)
  prg_names = {e.tag: e.name for e in data if type(e).__name__ == "ProfileProgramEvent" and e.tag is not None}
  sqtt_events = [e for e in data if type(e).__name__ == "ProfileSQTTEvent"]
  for i, event in enumerate(sqtt_events):
    print(f"\n=== event {i} {prg_names.get(event.kern, '')} ===")
    print_packets(decode(event.blob))
