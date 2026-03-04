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
from tinygrad.runtime.autogen.amd.rdna3.ins import s_endpgm # same encoding as RDNA4

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
  CALL = 0x5              # s_call_b64
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
  LDS_ATOMIC = 0x2a        # ds_append, ds_consume, ds_store_addtid_b32
  LDS_STORE = 0x2b
  LDS_STORE_64 = 0x2c
  LDS_STORE_96 = 0x2d
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
  SALU = 0x0
  SMEM = 0x1
  JUMP = 0x3
  JUMP_NO = 0x4
  JUMP_UNCOND = 0x5
  MESSAGE = 0x9
  VALU_TRANS = 0xb
  VALU_B2 = 0xd
  VALU_B4 = 0xe
  VINTERP = 0x12
  VMEM_RD_1 = 0x21
  VMEM_WR_2 = 0x24
  VMEM_WR_3 = 0x25
  VMEM_WR_4 = 0x26
  VMEM_WR_5 = 0x27
  VMEM_WR_6 = 0x28
  LDS_RD = 0x29
  LDS_WR_1 = 0x2a
  LDS_WR_2 = 0x2b
  LDS_WR_3 = 0x2c
  LDS_WR_4 = 0x2d
  LDS_WR_5 = 0x2e
  WMMA_8 = 0x8c
  WMMA_16 = 0x8d
  VALU_DPFP = 0x92
  SALU_FLOAT3 = 0x98
  VALU_SCL_TRANS = 0x99
  SALU_2 = 0x9b
  SALU_5 = 0x9c
  OTHER_VMEM = 0xc1

class CDNAIssueStatus(Enum): NULL = 0; STALL = 1; INST = 2; IMMED = 3

class CDNAInstType(Enum):
  SMEM = 0; SALU_32 = 1; VMEM_RD = 2; VMEM_WR = 3; FLAT_WR = 4; VALU_32 = 5; LDS = 6; PC = 7
  EXPREQ_GDS = 8; EXPREQ_GFX = 9; EXPGNT_PAR_COL = 10; EXPGNT_POS_GDS = 11
  JUMP = 12; NEXT = 13; FLAT_RD = 14; OTHER_MSG = 15; SMEM_WR = 16; SALU_64 = 17; VALU_64 = 18; VALU_MAI = 28

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
  w64h = bits[6:6]
  wave = bits[11:7]
  op = bits[19:12].enum(InstOpRDNA4)

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
  9: WAVESTART_RDNA4, 10: TS_DELTA_S5_W2_RDNA4, 11: WAVEALLOC_RDNA4,
  12: TS_DELTA_S5_W3_RDNA4, 13: PERF_RDNA4, 22: TS_DELTA_OR_MARK_RDNA4, 24: INST_RDNA4,
}

# ═══════════════════════════════════════════════════════════════════════════════
# CDNA PACKET TYPE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class MISC_CDNA(PacketType):
  """type 0: 16-bit misc (timestamp delta, time reset, packet lost, etc.)"""
  encoding = bits[3:0] == 0
  delta = bits[11:4]
  sh = bits[12:12]
  misc_type = bits[15:13]

class TIMESTAMP_CDNA(PacketType):
  """type 1: 64-bit absolute timestamp"""
  encoding = bits[3:0] == 1
  _unk = bits[15:4]
  timestamp = bits[63:16]

class REG_CDNA(PacketType):
  """type 2: 64-bit register write (Reg)"""
  encoding = bits[3:0] == 2
  pipe = bits[6:5]
  _me_raw = bits[8:7]
  regaddr = bits[31:16]
  regdata = bits[63:32]

class WAVESTART_CDNA(PacketType):
  """type 3: 32-bit wave start (Wave/group_id)"""
  encoding = bits[3:0] == 3
  sh = bits[5:5]
  cu = bits[9:6]
  wave = bits[13:10]
  simd = bits[15:14]
  pipe = bits[17:16]
  me = bits[19:18]
  _gap = bits[21:20]
  count = bits[28:22]
  _padding = bits[31:29]

class WAVEALLOC_CDNA(PacketType):
  """type 4: 16-bit wave alloc (group_id)"""
  encoding = bits[3:0] == 4
  sh = bits[5:5]
  cu = bits[9:6]
  wave = bits[13:10]
  simd = bits[15:14]

class REGCS_CDNA(PacketType):
  """type 5: 48-bit register CS write (RegCs)"""
  encoding = bits[3:0] == 5
  pipe = bits[6:5]
  _me_raw = bits[8:7]
  regaddr = bits[15:9]
  regdata = bits[47:16]

class WAVEEND_CDNA(PacketType):
  """type 6: 16-bit wave end (group_id)"""
  encoding = bits[3:0] == 6
  sh = bits[5:5]
  cu = bits[9:6]
  wave = bits[13:10]
  simd = bits[15:14]

class EVENT_CDNA(PacketType):
  """type 7: 16-bit event"""
  encoding = bits[3:0] == 7
  _data = bits[15:4]

class EVENT_CS_CDNA(PacketType):
  """type 8: 16-bit event CS"""
  encoding = bits[3:0] == 8
  _data = bits[15:4]

class EVENT_GFX1_CDNA(PacketType):
  """type 9: 16-bit event GFX1"""
  encoding = bits[3:0] == 9
  _data = bits[15:4]

class INST_CDNA(PacketType):
  """type 10: 16-bit instruction classification (MsgInst/TOKEN_INST)"""
  encoding = bits[3:0] == 10
  wave = bits[8:5]
  simd = bits[10:9]
  op = bits[15:11].enum(CDNAInstType)

class INST_PC_CDNA(PacketType):
  """type 11: 64-bit instruction PC (MsgInstPc)"""
  encoding = bits[3:0] == 11
  wave = bits[8:5]
  simd = bits[10:9]
  err = bits[15:15]
  pc = bits[63:16]

class SHADERDATA_CDNA(PacketType):
  """type 12: 48-bit shader data (UserData/group_id)"""
  encoding = bits[3:0] == 12
  sh = bits[5:5]
  cu = bits[9:6]
  wave = bits[13:10]
  simd = bits[15:14]
  data = bits[47:16]

class ISSUE_CDNA(PacketType):
  """type 13: 32-bit issue status (Issue/TOKEN_ISSUE). Per-wave 2-bit status: 0=null, 1=stall, 2=inst, 3=immed."""
  encoding = bits[3:0] == 13
  simd = bits[6:5]
  _padding = bits[31:7]
  def wave_status(self, wave_id: int) -> int: return (self._raw >> (2 * wave_id + 8)) & 3

class PERF_CDNA(PacketType):
  """type 14: 64-bit perf counter (MsgPerf)"""
  encoding = bits[3:0] == 14
  sh = bits[5:5]
  cu = bits[9:6]
  cntr_bank = bits[11:10]
  cntr_0 = bits[24:12]
  cntr_1 = bits[37:25]
  cntr_2 = bits[50:38]
  _flag = bits[51:51]
  _padding = bits[63:52]

class REGCS_PRIV_CDNA(PacketType):
  """type 15: 48-bit register CS privileged (RegCs)"""
  encoding = bits[3:0] == 15
  pipe = bits[6:5]
  _me_raw = bits[8:7]
  regaddr = bits[15:9]
  regdata = bits[47:16]

class IMMEDIATE_CDNA(PacketType):
  """Synthesized: immediate instruction (ISSUE status=3). wave/simd/cu set as attributes."""

_CDNA_TOKEN_TYPES: dict[int, type[PacketType]] = {
  0: MISC_CDNA, 1: TIMESTAMP_CDNA, 2: REG_CDNA, 3: WAVESTART_CDNA, 4: WAVEALLOC_CDNA, 5: REGCS_CDNA,
  6: WAVEEND_CDNA, 7: EVENT_CDNA, 8: EVENT_CS_CDNA, 9: EVENT_GFX1_CDNA, 10: INST_CDNA, 11: INST_PC_CDNA,
  12: SHADERDATA_CDNA, 13: ISSUE_CDNA, 14: PERF_CDNA, 15: REGCS_PRIV_CDNA,
}
PACKET_TYPES_CDNA: dict[int, type[PacketType]] = {**_CDNA_TOKEN_TYPES, 16: IMMEDIATE_CDNA}

# ═══════════════════════════════════════════════════════════════════════════════
# DECODER
# ═══════════════════════════════════════════════════════════════════════════════

def _build_decode_tables(packet_types: dict[int, type[PacketType]]) -> tuple[dict[int, tuple], bytes]:
  # Build state table: byte -> opcode. Sort by mask specificity (more bits first), NOP last
  sorted_types = sorted(packet_types.items(), key=lambda x: (-bin(x[1].encoding.mask).count('1'), x[0] == 16))
  state_table = bytes(next((op for op, cls in sorted_types if (b & cls.encoding.mask) == cls.encoding.default), 16) for b in range(256))
  # Build decode info: opcode -> (pkt_cls, nib_count, delta_lo, delta_mask, special_case)
  # special_case: 0=none, 1=TS_DELTA_OR_MARK (check is_marker), 2=TS_DELTA_SHORT (add 8), 3=CDNA_DELTA (*4), 4=TIMESTAMP_CDNA (absolute)
  _special = {TS_DELTA_OR_MARK: 1, TS_DELTA_OR_MARK_RDNA4: 1, TS_DELTA_SHORT: 2,} # CDNA_DELTA: 3, TIMESTAMP_CDNA: 4}
  decode_info = {}
  for opcode, pkt_cls in packet_types.items():
    delta_field = getattr(pkt_cls, 'delta', None)
    special = _special.get(pkt_cls, 0)
    decode_info[opcode] = (pkt_cls, pkt_cls._size_nibbles, delta_field.lo if delta_field else 0, delta_field.mask if delta_field else 0, special)  # type: ignore[attr-defined]
  return decode_info, state_table

_DECODE_INFO_RDNA3, _STATE_TABLE_RDNA3 = _build_decode_tables(PACKET_TYPES_RDNA3)
_DECODE_INFO_RDNA4, _STATE_TABLE_RDNA4 = _build_decode_tables(PACKET_TYPES_RDNA4)

# CDNA replay instruction types (erased, not real instructions)
_CDNA_REPLAY_TYPES = {19, 20, 21, 22, 23, 24}

def decode_cdna(data: bytes) -> Iterator[PacketType]:
  """Decode CDNA (gfx9) SQTT blob. Byte-aligned tokens, matches rocprof-trace-decoder gfx9 logic."""
  header_raw = int.from_bytes(data[:8], 'little')
  target_cu = (header_raw >> 20) & 0x1f

  n, pos, globaltime, base_time = len(data), 8, 0, 0
  lookahead: list[tuple[type[PacketType], int, int]] = []  # (pkt_cls, raw, delta) buffer for time patching
  wave_issued: list[list[list[int]|None]] = [[None]*10 for _ in range(4)]  # per-simd per-wave issue time queue, None=inactive

  def _parse_one() -> tuple[type[PacketType], int, int] | None:
    nonlocal pos
    if pos + 2 > n: return None
    pkt_cls = _CDNA_TOKEN_TYPES[data[pos] & 0xf]
    sz = pkt_cls._size_nibbles // 2
    if pos + sz > n: return None
    raw = int.from_bytes(data[pos:pos + sz], 'little')
    pos += sz
    return (pkt_cls, raw, ((raw >> 4) & 0xff) if pkt_cls is MISC_CDNA else ((raw >> 4) & 1))

  def _patch_time() -> None:
    """Scan ahead for TIMESTAMP to recalibrate globaltime after TIME_RESET. Matches MITokenGenerator::patch_time."""
    nonlocal globaltime, base_time
    def _update(ts):
      nonlocal globaltime, base_time
      if base_time == 0: base_time = ts - globaltime + 4
      if ts > base_time: globaltime = (ts - base_time) & ~3
    for i, (pkt_cls, raw, _) in enumerate(lookahead):
      if pkt_cls is TIMESTAMP_CDNA: _update((raw >> 16) & ((1 << 48) - 1)); lookahead.pop(i); return
      elif pkt_cls is MISC_CDNA and ((raw >> 13) & 0x7) == 1: return
    while True:
      tok = _parse_one()
      if tok is None: return
      pkt_cls, raw, _ = tok
      if pkt_cls is TIMESTAMP_CDNA: _update((raw >> 16) & ((1 << 48) - 1)); return
      if pkt_cls is MISC_CDNA and ((raw >> 13) & 0x7) == 1: return
      lookahead.append(tok)

  def _next_token() -> tuple[type[PacketType], int, int] | None:
    return lookahead.pop(0) if lookahead else _parse_one()

  while True:
    if (tok := _next_token()) is None: break
    pkt_cls, raw, delta = tok
    globaltime += delta * 4
    p = pkt_cls.from_raw(raw, globaltime)
    if pkt_cls is MISC_CDNA:
      if ((raw >> 13) & 0x7) == 1: _patch_time()
    elif pkt_cls is WAVESTART_CDNA:
      if p.cu == target_cu and p.sh == 0 and p.count <= 64:
        wave_issued[p.simd][p.wave] = []
      yield p
    elif pkt_cls is WAVEEND_CDNA:
      if p.cu == target_cu and p.sh == 0:
        wave_issued[p.simd][p.wave] = None
      yield p
    elif pkt_cls is ISSUE_CDNA:
      for wave_id in range(10):
        status = CDNAIssueStatus(p.wave_status(wave_id))
        if status is CDNAIssueStatus.NULL: continue
        if status is CDNAIssueStatus.INST: wave_issued[p.simd][wave_id].append(globaltime)  # queue issue time
        elif status is CDNAIssueStatus.IMMED:
          new_pkt = IMMEDIATE_CDNA.from_raw(0, globaltime + 4)
          new_pkt.wave, new_pkt.simd = wave_id, p.simd
          yield new_pkt
    elif pkt_cls is INST_CDNA: # confirms queued ISSUE
      if p.op.value not in _CDNA_REPLAY_TYPES:
        p._time = wave_issued[p.simd][p.wave].pop(0)
        yield p
    else:
      yield pkt_cls.from_raw(raw, globaltime)

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
      read_bytes = min(byte_count, 8)
      chunk = int.from_bytes(data[pos:pos + read_bytes], 'little')
      reg, pos = (reg >> (read_bytes * 8)) | (chunk << (64 - read_bytes * 8)), pos + byte_count
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
    elif special == 4:  # TIMESTAMP_CDNA (absolute timestamp anchoring)
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
        yield from decode_cdna(data)
        return
    yield pkt

# ═══════════════════════════════════════════════════════════════════════════════
# MAPPER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class InstructionInfo:
  pc: int
  wave: int
  inst: Inst


def map_insts(data:bytes, lib:bytes, target:str) -> Iterator[tuple[PacketType, InstructionInfo|None]]:
  """maps SQTT packets to instructions, yields (packet, instruction_info or None)"""
  # map pcs to insts
  from tinygrad.viz.serve import amd_decode
  pc_map = amd_decode(lib, target)

  is_cdna = target.startswith("gfx9")
  wave_pc:dict[int, int] = {}
  # only processing packets on one [CU, SIMD] unit
  if is_cdna:
    header_raw = int.from_bytes(data[:8], 'little')
    target_cu = (header_raw >> 20) & 0x1f
    def simd_select(p) -> bool: return getattr(p, "simd", -1) == 0 and getattr(p, "cu", target_cu) == target_cu
  else:
    def simd_select(p) -> bool: return getattr(p, "cu", 0) == 0 and getattr(p, "simd", 0) == 0
  for p in decode(data):
    if not simd_select(p): continue
    if isinstance(p, (WAVESTART, WAVESTART_RDNA4, WAVESTART_CDNA)):
      assert p.wave not in wave_pc, "only one inflight wave per unit"
      wave_pc[p.wave] = next(iter(pc_map))
      continue
    if isinstance(p, (WAVEEND, WAVEEND_CDNA)):
      pc = wave_pc.pop(p.wave)
      yield (p, InstructionInfo(pc, p.wave, s_endpgm()) if isinstance(p, WAVEEND) else None)
      continue
    # CDNA decoded instructions
    if isinstance(p, INST_CDNA):
      inst = pc_map[pc:=wave_pc[p.wave]]
      if isinstance(p.op, CDNAInstType) and p.op is CDNAInstType.JUMP:
        simm16 = getattr(inst, 'simm16', None)
        assert simm16 is not None, f"CDNA JUMP must map to a branch instruction, got {inst}"
        x = simm16 & 0xffff
        wave_pc[p.wave] += inst.size() + (x - 0x10000 if x & 0x8000 else x)*4
      else:
        wave_pc[p.wave] += inst.size()
      yield (p, InstructionInfo(pc, p.wave, inst))
      continue
    if isinstance(p, IMMEDIATE_CDNA):
      inst = pc_map[pc:=wave_pc[p.wave]]
      wave_pc[p.wave] += inst.size()
      yield (p, InstructionInfo(pc, p.wave, inst))
      continue
    # skip OTHER_ instructions, they don't belong to this unit
    if isinstance(p, (INST, INST_RDNA4)) and p.op.name.startswith("OTHER_"): continue
    if isinstance(p, IMMEDIATE_MASK):
      # immediate mask may yield multiple times per packet
      for wave in range(16):
        if p.mask & (1 << wave):
          inst = pc_map[pc:=wave_pc[wave]]
          # can this assert be more strict?
          assert type(inst).__name__ == "SOPP", f"IMMEDIATE_MASK packet must map to SOPP, got {inst}"
          wave_pc[wave] += inst.size()
          yield (p, InstructionInfo(pc, wave, inst))
      continue
    if isinstance(p, (VALUINST, INST, INST_RDNA4, IMMEDIATE)):
      inst = pc_map[pc:=wave_pc[p.wave]]
      # s_delay_alu doesn't get a packet?
      while (inst_op:=getattr(inst, 'op_name', '')) in {"S_DELAY_ALU", "S_WAIT_ALU"}:
        wave_pc[p.wave] += inst.size()
        inst = pc_map[pc:=wave_pc[p.wave]]
      # identify a branch instruction, only used for asserts
      branch_inst = inst if "BRANCH" in inst_op else None
      if branch_inst is not None:
        assert isinstance(p, (INST, INST_RDNA4)) and p.op.name in {"JUMP_NO", "JUMP", "JUMP_UNCOND"}, f"branch can only be folowed by JUMP, got {p}"
      # JUMP handling
      if (isinstance(p, INST) and p.op is InstOp.JUMP) or (isinstance(p, INST_RDNA4) and p.op is InstOpRDNA4.JUMP):
        simm16 = getattr(branch_inst, 'simm16')
        assert branch_inst is not None and simm16 is not None, f"JUMP packet must map to a branch instruction, got {inst}"
        x = simm16 & 0xffff
        wave_pc[p.wave] += branch_inst.size() + (x - 0x10000 if x & 0x8000 else x)*4
      else:
        if branch_inst is not None: assert inst_op != "S_BRANCH", f"S_BRANCH must have a JUMP packet, got {p}"
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
    fields = f"wave={p.wave} op={op_name}" + ((" flag1" if p.flag1 else "") + (" flag2" if p.flag2 else "") if isinstance(p, INST) else "")
  elif isinstance(p, VALUINST): fields = f"wave={p.wave}" + (" flag" if p.flag else "")
  elif isinstance(p, ALUEXEC): fields = f"src={p.src.name if isinstance(p.src, AluSrc) else p.src}"
  elif isinstance(p, VMEMEXEC): fields = f"src={p.src.name if isinstance(p.src, MemSrc) else p.src}"
  elif isinstance(p, (WAVESTART, WAVESTART_RDNA4, WAVESTART_CDNA, WAVEEND_CDNA, WAVEEND)): fields = f"wave={p.wave} simd={p.simd} cu={p.cu}"
  elif hasattr(p, '_fields'):
    filt = {'delta', 'encoding'} if not isinstance(p, (TS_DELTA_OR_MARK, TS_DELTA_OR_MARK_RDNA4)) else {'encoding'}
    fields = " ".join(f"{k}=0x{getattr(p, k):x}" if k in {'snap', 'val32'} else f"{k}={getattr(p, k)}"
                      for k in p._fields if not k.startswith('_') and k not in filt)
  else: fields = ""
  return f"{p._time:8}: {colored(f'{name:18}', PACKET_COLORS.get(name.replace('_RDNA4', '').replace('_CDNA', ''), 'white'))} {fields}"

def print_packets(packets) -> None:
  from tinygrad.helpers import getenv
  skip = {"NOP", "TS_DELTA_SHORT", "TS_WAVE_STATE", "TS_DELTA_OR_MARK",
          "TS_DELTA_S5_W2", "TS_DELTA_S5_W3", "TS_DELTA_S8_W3", "REG", "EVENT"} if not getenv("NOSKIP") else {"NOP"}
  for data in packets:
    p, inst = data if isinstance(data, tuple) else (data, None)
    if type(p).__name__.replace("_RDNA4", "") not in skip: print(format_packet(p), f"inst={inst.inst}" if inst is not None else '')

if __name__ == "__main__":
  import sys, pickle
  from tinygrad.helpers import temp
  with open(temp("profile.pkl", append_user=True) if len(sys.argv) < 2 else sys.argv[1], "rb") as f:
    data = pickle.load(f)
  prg_events = {e.tag: e for e in data if type(e).__name__ == "ProfileProgramEvent" and e.tag is not None}
  sqtt_events = [e for e in data if type(e).__name__ == "ProfileSQTTEvent"]
  dev_targets = {e.device:f"gfx{e.props['gfx_target_version']//1000}" for e in data if type(e).__name__ == "ProfileDeviceEvent" and e.props}
  for i, event in enumerate(sqtt_events):
    prg = prg_events.get(event.kern)
    print(f"\n=== event {i} {prg.name if prg is not None else ''} ===")
    print_packets(map_insts(event.blob, prg.lib, dev_targets[prg.device]) if prg is not None else decode(event.blob))
