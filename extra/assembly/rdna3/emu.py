# RDNA3 emulator - pure Python implementation for testing
from __future__ import annotations
import ctypes, struct, math
from dataclasses import dataclass, field
from typing import Any
from tinygrad.helpers import DEBUG, colored
from extra.assembly.rdna3.lib import Inst32, Inst64, bits, RawImm
from extra.assembly.rdna3.autogen import (
  SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, VOP1, VOP2, VOP3, VOP3SD, VOP3P, VOPC, DS, FLAT, VOPD,
  SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, SMEMOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp, DSOp, FLATOp, GLOBALOp, VOPDOp
)

Inst, Program = Inst32 | Inst64 | VOP3P, dict[int, Inst32 | Inst64 | VOP3P]
WAVE_SIZE, SGPR_COUNT, VGPR_COUNT = 32, 128, 256
VCC_LO, VCC_HI, EXEC_LO, EXEC_HI, NULL_REG, M0 = 106, 107, 126, 127, 124, 125
FLOAT_BITS = {240: 0x3f000000, 241: 0xbf000000, 242: 0x3f800000, 243: 0xbf800000, 244: 0x40000000, 245: 0xc0000000, 246: 0x40800000, 247: 0xc0800000, 248: 0x3e22f983}
CTYPES = {1: ctypes.c_uint8, 2: ctypes.c_uint16, 4: ctypes.c_uint32}

_valid_mem_ranges: set[tuple[int, int]] = set()
def set_valid_mem_ranges(ranges: set[tuple[int, int]]) -> None: global _valid_mem_ranges; _valid_mem_ranges = ranges

def _check_addr(addr: int, size: int) -> None:
  if _valid_mem_ranges and not any(s <= addr and addr + size <= s + z for s, z in _valid_mem_ranges):
    raise RuntimeError(f"OOB memory access at 0x{addr:x} size={size}")

def mem_read(addr: int, size: int) -> int: _check_addr(addr, size); return CTYPES[size].from_address(addr).value
def mem_write(addr: int, size: int, val: int) -> None: _check_addr(addr, size); CTYPES[size].from_address(addr).value = val

# Memory op tables: op -> (count, size, signed)
FLAT_LOAD = {GLOBALOp.GLOBAL_LOAD_B32: (1,4,0), FLATOp.FLAT_LOAD_B32: (1,4,0), GLOBALOp.GLOBAL_LOAD_B64: (2,4,0), FLATOp.FLAT_LOAD_B64: (2,4,0),
  GLOBALOp.GLOBAL_LOAD_B96: (3,4,0), FLATOp.FLAT_LOAD_B96: (3,4,0), GLOBALOp.GLOBAL_LOAD_B128: (4,4,0), FLATOp.FLAT_LOAD_B128: (4,4,0),
  GLOBALOp.GLOBAL_LOAD_U8: (1,1,0), FLATOp.FLAT_LOAD_U8: (1,1,0), GLOBALOp.GLOBAL_LOAD_I8: (1,1,1), FLATOp.FLAT_LOAD_I8: (1,1,1),
  GLOBALOp.GLOBAL_LOAD_U16: (1,2,0), FLATOp.FLAT_LOAD_U16: (1,2,0), GLOBALOp.GLOBAL_LOAD_I16: (1,2,1), FLATOp.FLAT_LOAD_I16: (1,2,1)}
FLAT_STORE = {GLOBALOp.GLOBAL_STORE_B32: (1,4), FLATOp.FLAT_STORE_B32: (1,4), GLOBALOp.GLOBAL_STORE_B64: (2,4), FLATOp.FLAT_STORE_B64: (2,4),
  GLOBALOp.GLOBAL_STORE_B96: (3,4), FLATOp.FLAT_STORE_B96: (3,4), GLOBALOp.GLOBAL_STORE_B128: (4,4), FLATOp.FLAT_STORE_B128: (4,4),
  GLOBALOp.GLOBAL_STORE_B8: (1,1), FLATOp.FLAT_STORE_B8: (1,1), GLOBALOp.GLOBAL_STORE_B16: (1,2), FLATOp.FLAT_STORE_B16: (1,2)}
DS_LOAD = {DSOp.DS_LOAD_B32: (1,4,0), DSOp.DS_LOAD_B64: (2,4,0), DSOp.DS_LOAD_B128: (4,4,0),
  DSOp.DS_LOAD_U8: (1,1,0), DSOp.DS_LOAD_I8: (1,1,1), DSOp.DS_LOAD_U16: (1,2,0), DSOp.DS_LOAD_I16: (1,2,1)}
DS_STORE = {DSOp.DS_STORE_B32: (1,4), DSOp.DS_STORE_B64: (2,4), DSOp.DS_STORE_B128: (4,4), DSOp.DS_STORE_B8: (1,1), DSOp.DS_STORE_B16: (1,2)}

# Optimized float conversion using cached struct
_struct_I, _struct_f = struct.Struct('<I'), struct.Struct('<f')
def f32(i: int) -> float: return _struct_f.unpack(_struct_I.pack(i & 0xffffffff))[0]
def i32(f: float) -> int: return _struct_I.unpack(_struct_f.pack(f))[0]
def sext(v: int, b: int) -> int: return v - (1 << b) if v & (1 << (b-1)) else v
def clz(x: int) -> int: return 32 - x.bit_length() if x else 32
def cls(x: int) -> int: x &= 0xffffffff; return 31 if x in (0, 0xffffffff) else clz(~x & 0xffffffff if x >> 31 else x) - 1
def f16(i: int) -> float: return struct.unpack('<e', struct.pack('<H', i & 0xffff))[0]
def i16(f: float) -> int: return struct.unpack('<H', struct.pack('<e', f))[0]

# Shared transcendental/conversion functions
def alu_rcp(a): v = f32(a); return i32(1.0 / v if v != 0 else math.copysign(float('inf'), v))
def alu_rsq(a): v = f32(a); return i32(1.0 / math.sqrt(v) if v > 0 else (float('nan') if v < 0 else float('inf')))
def alu_sqrt(a): v = f32(a); return i32(math.sqrt(v) if v >= 0 else float('nan'))
def alu_log(a): v = f32(a); return i32(math.log2(v) if v > 0 else (float('-inf') if v == 0 else float('nan')))



@dataclass
class WaveState:
  sgpr: list[int] = field(default_factory=lambda: [0] * SGPR_COUNT)
  vgpr: list[list[int]] = field(default_factory=lambda: [[0] * VGPR_COUNT for _ in range(WAVE_SIZE)])
  scc: int = 0
  vcc: int = 0
  exec_mask: int = 0xffffffff
  pc: int = 0
  literal: int = 0
  _pend_vcc: int | None = None
  _pend_exec: int | None = None
  _pend_sgpr: dict[int, int] = field(default_factory=dict)

  def rsgpr(self, i: int) -> int:
    return {VCC_LO: self.vcc & 0xffffffff, VCC_HI: (self.vcc >> 32) & 0xffffffff, EXEC_LO: self.exec_mask & 0xffffffff,
            EXEC_HI: (self.exec_mask >> 32) & 0xffffffff, NULL_REG: 0, 253: self.scc}.get(i, self.sgpr[i] if i < SGPR_COUNT else 0)

  def wsgpr(self, i: int, v: int) -> None:
    v &= 0xffffffff
    if i == VCC_LO: self.vcc = (self.vcc & 0xffffffff00000000) | v
    elif i == VCC_HI: self.vcc = (self.vcc & 0xffffffff) | (v << 32)
    elif i == EXEC_LO: self.exec_mask = (self.exec_mask & 0xffffffff00000000) | v
    elif i == EXEC_HI: self.exec_mask = (self.exec_mask & 0xffffffff) | (v << 32)
    elif i < SGPR_COUNT: self.sgpr[i] = v

  def rsgpr64(self, i: int) -> int: return self.rsgpr(i) | (self.rsgpr(i+1) << 32)
  def wsgpr64(self, i: int, v: int) -> None: self.wsgpr(i, v & 0xffffffff); self.wsgpr(i+1, (v >> 32) & 0xffffffff)

  def rsrc(self, v: int, lane: int) -> int:
    if v <= 105: return self.sgpr[v]
    if v in (VCC_LO, VCC_HI): return (self.vcc >> (32 if v == VCC_HI else 0)) & 0xffffffff
    if 108 <= v <= 123 or v == M0: return self.sgpr[v]
    if v in (EXEC_LO, EXEC_HI): return (self.exec_mask >> (32 if v == EXEC_HI else 0)) & 0xffffffff
    if v == NULL_REG: return 0
    if 128 <= v <= 192: return v - 128
    if 193 <= v <= 208: return (-(v - 192)) & 0xffffffff
    if v in FLOAT_BITS: return FLOAT_BITS[v]
    if v == 255: return self.literal
    if 256 <= v <= 511: return self.vgpr[lane][v - 256]
    return 0

  def rsrc64(self, v: int, lane: int) -> int:
    return self.rsrc(v, lane) | ((self.rsrc(v+1, lane) if v <= 105 or 256 <= v <= 511 else 0) << 32)

  def pend_vcc_lane(self, lane: int, val: bool) -> None:
    if self._pend_vcc is None: self._pend_vcc = 0
    if val: self._pend_vcc |= (1 << lane)

  def pend_exec_lane(self, lane: int, val: bool) -> None:
    if self._pend_exec is None: self._pend_exec = 0
    if val: self._pend_exec |= (1 << lane)

  def pend_sgpr_lane(self, reg: int, lane: int, val: bool) -> None:
    if reg not in self._pend_sgpr: self._pend_sgpr[reg] = 0
    if val: self._pend_sgpr[reg] |= (1 << lane)

  def commit_pends(self) -> None:
    if self._pend_vcc is not None: self.vcc = self._pend_vcc; self._pend_vcc = None
    if self._pend_exec is not None: self.exec_mask = self._pend_exec; self._pend_exec = None
    for reg, val in self._pend_sgpr.items(): self.wsgpr(reg, val)
    self._pend_sgpr.clear()

def decode_format(word: int) -> tuple[type[Inst] | None, bool]:
  hi2 = (word >> 30) & 0x3
  if hi2 == 0b11:
    enc = (word >> 26) & 0xf
    if enc == 0b1101: return SMEM, True
    if enc == 0b0101:
      op = (word >> 16) & 0x3ff
      return (VOP3SD, True) if op in (288, 289, 290, 764, 765, 766, 767, 768, 769, 770) else (VOP3, True)
    return {0b0011: (VOP3P, True), 0b0110: (DS, True), 0b0111: (FLAT, True), 0b0010: (VOPD, True)}.get(enc, (None, True))
  if hi2 == 0b10:
    enc = (word >> 23) & 0x7f
    return {0b1111101: (SOP1, False), 0b1111110: (SOPC, False), 0b1111111: (SOPP, False)}.get(enc, (SOPK, False) if ((word >> 28) & 0xf) == 0b1011 else (SOP2, False))
  enc = (word >> 25) & 0x7f
  return (VOPC, False) if enc == 0b0111110 else (VOP1, False) if enc == 0b0111111 else (VOP2, False)

def _unwrap(v) -> int:
  """Unwrap RawImm/enum to plain int for fast field access."""
  return v.val if isinstance(v, RawImm) else v.value if hasattr(v, 'value') else v

def _cache_fields(inst) -> None:
  """Cache unwrapped field values on instruction for fast access."""
  for name, val in inst._values.items():
    setattr(inst, name, _unwrap(val))

def decode_program(data: bytes) -> Program:
  result: Program = {}
  i = 0
  while i < len(data):
    word = int.from_bytes(data[i:i+4], 'little')
    inst_class, is_64 = decode_format(word)
    if inst_class is None: i += 4; continue
    base_size = 8 if is_64 else 4
    inst = inst_class.from_bytes(data[i:i+base_size])
    _cache_fields(inst)
    has_literal = any(getattr(inst, fld, None) == 255 for fld in ('src0', 'src1', 'src2', 'ssrc0', 'ssrc1', 'srcx0', 'srcy0'))
    if inst_class == VOP2 and inst.op in (44, 45, 55, 56): has_literal = True
    if inst_class == VOPD and (inst.opx in (1, 2) or inst.opy in (1, 2)): has_literal = True
    if has_literal: inst._literal = int.from_bytes(data[i+base_size:i+base_size+4], 'little')
    result[i // 4] = inst
    i += inst.size()
    if inst_class == SOPP and inst.op == SOPPOp.S_ENDPGM: break
  return result

# SOP1 pure ops: fn(s0) -> result (no SCC, no special state)
def _s_cvt_i32_f32(s): v = f32(s); return 0x7fffffff if math.isinf(v) and v > 0 else ((-0x80000000 & 0xffffffff) if math.isinf(v) else max(-0x80000000, min(0x7fffffff, int(v))) & 0xffffffff)
def _s_cvt_u32_f32(s): v = f32(s); return 0xffffffff if math.isinf(v) and v > 0 else (0 if math.isinf(v) or math.isnan(v) or v < 0 else min(0xffffffff, int(v)))
SOP1_OPS = {
  SOP1Op.S_MOV_B32: lambda s: s, SOP1Op.S_BREV_B32: lambda s: int(f'{s:032b}'[::-1], 2),
  SOP1Op.S_CLZ_I32_U32: clz, SOP1Op.S_CLS_I32: cls,
  SOP1Op.S_SEXT_I32_I8: lambda s: sext(s & 0xff, 8) & 0xffffffff, SOP1Op.S_SEXT_I32_I16: lambda s: sext(s & 0xffff, 16) & 0xffffffff,
  SOP1Op.S_CEIL_F32: lambda s: i32(math.ceil(f32(s))), SOP1Op.S_FLOOR_F32: lambda s: i32(math.floor(f32(s))),
  SOP1Op.S_TRUNC_F32: lambda s: i32(math.trunc(f32(s))), SOP1Op.S_RNDNE_F32: lambda s: i32(round(f32(s))),
  SOP1Op.S_CVT_F32_I32: lambda s: i32(float(sext(s, 32))), SOP1Op.S_CVT_F32_U32: lambda s: i32(float(s)),
  SOP1Op.S_CVT_I32_F32: _s_cvt_i32_f32, SOP1Op.S_CVT_U32_F32: _s_cvt_u32_f32,
  SOP1Op.S_CVT_F16_F32: lambda s: i16(f32(s)), SOP1Op.S_CVT_F32_F16: lambda s: i32(f16(s)),
}
# SOP1 ops that set SCC = (result != 0)
SOP1_SCC = {SOP1Op.S_NOT_B32: lambda s: (~s) & 0xffffffff, SOP1Op.S_ABS_I32: lambda s: abs(sext(s, 32)) & 0xffffffff}

def exec_sop1(st: WaveState, inst: SOP1) -> int:
  s0, op = st.rsrc(inst.ssrc0, 0), inst.op
  if (fn := SOP1_OPS.get(op)): st.wsgpr(inst.sdst, fn(s0)); return 0
  if (fn := SOP1_SCC.get(op)): r = fn(s0); st.wsgpr(inst.sdst, r); st.scc = int(r != 0); return 0
  match op:
    case SOP1Op.S_MOV_B64:             st.wsgpr64(inst.sdst, st.rsrc64(inst.ssrc0, 0))
    case SOP1Op.S_BITSET0_B32:         st.wsgpr(inst.sdst, st.rsgpr(inst.sdst) & ~(1 << (s0 & 0x1f)))
    case SOP1Op.S_BITSET1_B32:         st.wsgpr(inst.sdst, st.rsgpr(inst.sdst) | (1 << (s0 & 0x1f)))
    case SOP1Op.S_NOT_B64:             r = (~st.rsrc64(inst.ssrc0, 0)) & 0xffffffffffffffff; st.wsgpr64(inst.sdst, r); st.scc = int(r != 0)
    case SOP1Op.S_AND_SAVEEXEC_B32:    old = st.exec_mask & 0xffffffff; st.exec_mask = s0 & old; st.scc = int(st.exec_mask != 0); st.wsgpr(inst.sdst, old)
    case SOP1Op.S_OR_SAVEEXEC_B32:     old = st.exec_mask & 0xffffffff; st.exec_mask = s0 | old; st.scc = int(st.exec_mask != 0); st.wsgpr(inst.sdst, old)
    case SOP1Op.S_AND_NOT1_SAVEEXEC_B32: old = st.exec_mask & 0xffffffff; st.exec_mask = s0 & (~old & 0xffffffff); st.scc = int(st.exec_mask != 0); st.wsgpr(inst.sdst, old)
    case _: raise NotImplementedError(f"SOP1 op {op}")
  return 0

# SOP2 pure ops: fn(s0, s1) -> result (no SCC, no special state)
SOP2_OPS = {
  SOP2Op.S_MUL_I32: lambda a, b: (sext(a, 32) * sext(b, 32)) & 0xffffffff,
  SOP2Op.S_MUL_HI_U32: lambda a, b: ((a * b) >> 32) & 0xffffffff,
  SOP2Op.S_MUL_HI_I32: lambda a, b: ((sext(a, 32) * sext(b, 32)) >> 32) & 0xffffffff,
  SOP2Op.S_PACK_LL_B32_B16: lambda a, b: (a & 0xffff) | ((b & 0xffff) << 16),
  SOP2Op.S_PACK_LH_B32_B16: lambda a, b: (a & 0xffff) | (b & 0xffff0000),
  SOP2Op.S_PACK_HH_B32_B16: lambda a, b: ((a >> 16) & 0xffff) | (b & 0xffff0000),
  SOP2Op.S_PACK_HL_B32_B16: lambda a, b: ((a >> 16) & 0xffff) | ((b & 0xffff) << 16),
  SOP2Op.S_ADD_F32: lambda a, b: i32(f32(a) + f32(b)), SOP2Op.S_SUB_F32: lambda a, b: i32(f32(a) - f32(b)),
  SOP2Op.S_MUL_F32: lambda a, b: i32(f32(a) * f32(b)),
}
# SOP2 ops that set SCC = (result != 0)
SOP2_SCC = {
  SOP2Op.S_LSHL_B32: lambda a, b: (a << (b & 0x1f)) & 0xffffffff, SOP2Op.S_LSHR_B32: lambda a, b: a >> (b & 0x1f),
  SOP2Op.S_ASHR_I32: lambda a, b: sext(a, 32) >> (b & 0x1f),
  SOP2Op.S_AND_B32: lambda a, b: a & b, SOP2Op.S_OR_B32: lambda a, b: a | b, SOP2Op.S_XOR_B32: lambda a, b: a ^ b,
  SOP2Op.S_AND_NOT1_B32: lambda a, b: a & (~b & 0xffffffff), SOP2Op.S_OR_NOT1_B32: lambda a, b: a | (~b & 0xffffffff),
  SOP2Op.S_BFE_U32: lambda a, b: ((a >> (b & 0x1f)) & ((1 << ((b >> 16) & 0x7f)) - 1)) if (b >> 16) & 0x7f else 0,
  SOP2Op.S_BFE_I32: lambda a, b: (sext((a >> (b & 0x1f)) & ((1 << ((b >> 16) & 0x7f)) - 1), (b >> 16) & 0x7f) & 0xffffffff) if (b >> 16) & 0x7f else 0,
}
# SOP2 min/max: fn(s0, s1) -> (result, scc)
SOP2_MINMAX = {
  SOP2Op.S_MIN_I32: lambda a, b: (a, 1) if sext(a, 32) < sext(b, 32) else (b, 0),
  SOP2Op.S_MIN_U32: lambda a, b: (a, 1) if a < b else (b, 0),
  SOP2Op.S_MAX_I32: lambda a, b: (a, 1) if sext(a, 32) > sext(b, 32) else (b, 0),
  SOP2Op.S_MAX_U32: lambda a, b: (a, 1) if a > b else (b, 0),
}

def exec_sop2(st: WaveState, inst: SOP2) -> int:
  s0, s1, op = st.rsrc(inst.ssrc0, 0), st.rsrc(inst.ssrc1, 0), inst.op
  if (fn := SOP2_OPS.get(op)): st.wsgpr(inst.sdst, fn(s0, s1)); return 0
  if (fn := SOP2_SCC.get(op)): r = fn(s0, s1); st.wsgpr(inst.sdst, r & 0xffffffff); st.scc = int(r != 0); return 0
  if (fn := SOP2_MINMAX.get(op)): r, st.scc = fn(s0, s1); st.wsgpr(inst.sdst, r); return 0
  match op:
    case SOP2Op.S_ADD_U32:      r = s0 + s1; st.wsgpr(inst.sdst, r & 0xffffffff); st.scc = int(r >= 0x100000000)
    case SOP2Op.S_SUB_U32:      st.wsgpr(inst.sdst, (s0 - s1) & 0xffffffff); st.scc = int(s1 > s0)
    case SOP2Op.S_ADD_I32:      r = sext(s0, 32) + sext(s1, 32); st.wsgpr(inst.sdst, r & 0xffffffff); st.scc = int(((s0 >> 31) == (s1 >> 31)) and ((s0 >> 31) != ((r >> 31) & 1)))
    case SOP2Op.S_SUB_I32:      r = sext(s0, 32) - sext(s1, 32); st.wsgpr(inst.sdst, r & 0xffffffff); st.scc = int(((s0 >> 31) != (s1 >> 31)) and ((s0 >> 31) != ((r >> 31) & 1)))
    case SOP2Op.S_ADDC_U32:     r = s0 + s1 + st.scc; st.wsgpr(inst.sdst, r & 0xffffffff); st.scc = int(r >= 0x100000000)
    case SOP2Op.S_SUBB_U32:     st.wsgpr(inst.sdst, (s0 - s1 - st.scc) & 0xffffffff); st.scc = int((s1 + st.scc) > s0)
    case SOP2Op.S_LSHL_B64:     r = (st.rsrc64(inst.ssrc0, 0) << (s1 & 0x3f)) & 0xffffffffffffffff; st.wsgpr64(inst.sdst, r); st.scc = int(r != 0)
    case SOP2Op.S_LSHR_B64:     r = st.rsrc64(inst.ssrc0, 0) >> (s1 & 0x3f); st.wsgpr64(inst.sdst, r); st.scc = int(r != 0)
    case SOP2Op.S_ASHR_I64:     r = sext(st.rsrc64(inst.ssrc0, 0), 64) >> (s1 & 0x3f); st.wsgpr64(inst.sdst, r & 0xffffffffffffffff); st.scc = int(r != 0)
    case SOP2Op.S_AND_B64:      r = st.rsrc64(inst.ssrc0, 0) & st.rsrc64(inst.ssrc1, 0); st.wsgpr64(inst.sdst, r); st.scc = int(r != 0)
    case SOP2Op.S_OR_B64:       r = st.rsrc64(inst.ssrc0, 0) | st.rsrc64(inst.ssrc1, 0); st.wsgpr64(inst.sdst, r); st.scc = int(r != 0)
    case SOP2Op.S_XOR_B64:      r = st.rsrc64(inst.ssrc0, 0) ^ st.rsrc64(inst.ssrc1, 0); st.wsgpr64(inst.sdst, r); st.scc = int(r != 0)
    case SOP2Op.S_CSELECT_B32:  st.wsgpr(inst.sdst, s0 if st.scc else s1)
    case SOP2Op.S_CSELECT_B64:  st.wsgpr64(inst.sdst, st.rsrc64(inst.ssrc0, 0) if st.scc else st.rsrc64(inst.ssrc1, 0))
    case _: raise NotImplementedError(f"SOP2 op {op}")
  return 0

# SOPC compare ops table
SOPC_CMP = {
  SOPCOp.S_CMP_EQ_I32: lambda a, b: sext(a, 32) == sext(b, 32), SOPCOp.S_CMP_LG_I32: lambda a, b: sext(a, 32) != sext(b, 32),
  SOPCOp.S_CMP_GT_I32: lambda a, b: sext(a, 32) > sext(b, 32), SOPCOp.S_CMP_GE_I32: lambda a, b: sext(a, 32) >= sext(b, 32),
  SOPCOp.S_CMP_LT_I32: lambda a, b: sext(a, 32) < sext(b, 32), SOPCOp.S_CMP_LE_I32: lambda a, b: sext(a, 32) <= sext(b, 32),
  SOPCOp.S_CMP_EQ_U32: lambda a, b: a == b, SOPCOp.S_CMP_LG_U32: lambda a, b: a != b,
  SOPCOp.S_CMP_GT_U32: lambda a, b: a > b, SOPCOp.S_CMP_GE_U32: lambda a, b: a >= b,
  SOPCOp.S_CMP_LT_U32: lambda a, b: a < b, SOPCOp.S_CMP_LE_U32: lambda a, b: a <= b,
  SOPCOp.S_CMP_LT_F32: lambda a, b: f32(a) < f32(b), SOPCOp.S_CMP_EQ_F32: lambda a, b: f32(a) == f32(b),
  SOPCOp.S_CMP_LE_F32: lambda a, b: f32(a) <= f32(b), SOPCOp.S_CMP_GT_F32: lambda a, b: f32(a) > f32(b),
  SOPCOp.S_CMP_LG_F32: lambda a, b: f32(a) != f32(b), SOPCOp.S_CMP_GE_F32: lambda a, b: f32(a) >= f32(b),
  SOPCOp.S_CMP_O_F32: lambda a, b: not (math.isnan(f32(a)) or math.isnan(f32(b))),
  SOPCOp.S_CMP_U_F32: lambda a, b: math.isnan(f32(a)) or math.isnan(f32(b)),
  SOPCOp.S_CMP_NGE_F32: lambda a, b: not (f32(a) >= f32(b)), SOPCOp.S_CMP_NLG_F32: lambda a, b: not (f32(a) != f32(b)),
  SOPCOp.S_CMP_NGT_F32: lambda a, b: not (f32(a) > f32(b)), SOPCOp.S_CMP_NLE_F32: lambda a, b: not (f32(a) <= f32(b)),
  SOPCOp.S_CMP_NEQ_F32: lambda a, b: not (f32(a) == f32(b)), SOPCOp.S_CMP_NLT_F32: lambda a, b: not (f32(a) < f32(b)),
  SOPCOp.S_BITCMP0_B32: lambda a, b: (a & (1 << (b & 0x1f))) == 0, SOPCOp.S_BITCMP1_B32: lambda a, b: (a & (1 << (b & 0x1f))) != 0,
}
def exec_sopc(st: WaveState, inst: SOPC) -> int:
  s0, s1, op = st.rsrc(inst.ssrc0, 0), st.rsrc(inst.ssrc1, 0), inst.op
  if (fn := SOPC_CMP.get(op)): st.scc = int(fn(s0, s1)); return 0
  if op == SOPCOp.S_CMP_EQ_U64: st.scc = int(st.rsrc64(inst.ssrc0, 0) == st.rsrc64(inst.ssrc1, 0)); return 0
  if op == SOPCOp.S_CMP_LG_U64: st.scc = int(st.rsrc64(inst.ssrc0, 0) != st.rsrc64(inst.ssrc1, 0)); return 0
  raise NotImplementedError(f"SOPC op {op}")

# SOPK compare ops - I32 uses signed compare with simm, U32 uses unsigned compare with (simm & 0xffff)
SOPK_CMP_I32 = {SOPKOp.S_CMPK_EQ_I32: lambda a, b: a == b, SOPKOp.S_CMPK_LG_I32: lambda a, b: a != b, SOPKOp.S_CMPK_GT_I32: lambda a, b: a > b,
  SOPKOp.S_CMPK_GE_I32: lambda a, b: a >= b, SOPKOp.S_CMPK_LT_I32: lambda a, b: a < b, SOPKOp.S_CMPK_LE_I32: lambda a, b: a <= b}
SOPK_CMP_U32 = {SOPKOp.S_CMPK_EQ_U32: lambda a, b: a == b, SOPKOp.S_CMPK_LG_U32: lambda a, b: a != b, SOPKOp.S_CMPK_GT_U32: lambda a, b: a > b,
  SOPKOp.S_CMPK_GE_U32: lambda a, b: a >= b, SOPKOp.S_CMPK_LT_U32: lambda a, b: a < b, SOPKOp.S_CMPK_LE_U32: lambda a, b: a <= b}
SOPK_WAIT = {SOPKOp.S_WAITCNT_VSCNT, SOPKOp.S_WAITCNT_VMCNT, SOPKOp.S_WAITCNT_EXPCNT, SOPKOp.S_WAITCNT_LGKMCNT}
def exec_sopk(st: WaveState, inst: SOPK) -> int:
  simm, s0, op = sext(inst.simm16, 16), st.rsgpr(inst.sdst), inst.op
  if op == SOPKOp.S_MOVK_I32: st.wsgpr(inst.sdst, simm & 0xffffffff); return 0
  if op == SOPKOp.S_MULK_I32: st.wsgpr(inst.sdst, (sext(s0, 32) * simm) & 0xffffffff); return 0
  if op in SOPK_WAIT: return 0
  if (fn := SOPK_CMP_I32.get(op)): st.scc = int(fn(sext(s0, 32), simm)); return 0
  if (fn := SOPK_CMP_U32.get(op)): st.scc = int(fn(s0, simm & 0xffff)); return 0
  if op == SOPKOp.S_ADDK_I32:
    r = sext(s0, 32) + simm
    st.scc = int(((s0 >> 31) == ((simm >> 15) & 1)) and ((s0 >> 31) != ((r >> 31) & 1)))
    st.wsgpr(inst.sdst, r & 0xffffffff); return 0
  raise NotImplementedError(f"SOPK op {op}")

# SOPP branch conditions: fn(scc, vcc, exec) -> bool
SOPP_CBRANCH = {
  SOPPOp.S_CBRANCH_SCC0: lambda s, v, e: s == 0, SOPPOp.S_CBRANCH_SCC1: lambda s, v, e: s == 1,
  SOPPOp.S_CBRANCH_VCCZ: lambda s, v, e: v == 0, SOPPOp.S_CBRANCH_VCCNZ: lambda s, v, e: v != 0,
  SOPPOp.S_CBRANCH_EXECZ: lambda s, v, e: e == 0, SOPPOp.S_CBRANCH_EXECNZ: lambda s, v, e: e != 0,
}
def exec_sopp(st: WaveState, inst: SOPP) -> int:
  if inst.op == SOPPOp.S_ENDPGM: return -1
  if inst.op == SOPPOp.S_BARRIER: return -2
  if inst.op == SOPPOp.S_BRANCH: return sext(inst.simm16, 16)
  if (fn := SOPP_CBRANCH.get(inst.op)): return sext(inst.simm16, 16) if fn(st.scc, st.vcc, st.exec_mask) else 0
  return 0

SMEM_LOAD = {SMEMOp.S_LOAD_B32: 1, SMEMOp.S_LOAD_B64: 2, SMEMOp.S_LOAD_B128: 4, SMEMOp.S_LOAD_B256: 8, SMEMOp.S_LOAD_B512: 16}
def exec_smem(st: WaveState, inst: SMEM) -> int:
  addr = st.rsgpr64(inst.sbase * 2) + sext(inst.offset, 21)
  if inst.soffset not in (NULL_REG, 0x7f): addr += st.rsrc(inst.soffset, 0)
  if (cnt := SMEM_LOAD.get(inst.op)) is None: raise NotImplementedError(f"SMEM op {inst.op}")
  for i in range(cnt): st.wsgpr(inst.sdata + i, mem_read((addr + i * 4) & 0xffffffffffffffff, 4))
  return 0

# VOP1 table: op -> fn(s0) -> result (1-arg ops)
VOP1_OPS = {
  VOP1Op.V_MOV_B32: lambda s: s, VOP1Op.V_CVT_F32_I32: lambda s: i32(float(sext(s, 32))), VOP1Op.V_CVT_F32_U32: lambda s: i32(float(s)),
  VOP1Op.V_CVT_U32_F32: lambda s: 0 if (v := f32(s)) != v else max(0, min(0xffffffff, int(v) if math.isfinite(v) else (0xffffffff if v > 0 else 0))),
  VOP1Op.V_CVT_I32_F32: lambda s: 0 if (v := f32(s)) != v else (max(-0x80000000, min(0x7fffffff, int(v) if math.isfinite(v) else (0x7fffffff if v > 0 else -0x80000000))) & 0xffffffff),
  VOP1Op.V_CVT_F16_F32: lambda s: i16(f32(s)), VOP1Op.V_CVT_F32_F16: lambda s: i32(f16(s)),
  VOP1Op.V_TRUNC_F32: lambda s: i32(math.trunc(f32(s))), VOP1Op.V_CEIL_F32: lambda s: i32(math.ceil(f32(s))),
  VOP1Op.V_RNDNE_F32: lambda s: i32(round(f32(s))), VOP1Op.V_FLOOR_F32: lambda s: i32(math.floor(f32(s))),
  VOP1Op.V_EXP_F32: lambda s: i32(math.pow(2.0, f32(s))), VOP1Op.V_LOG_F32: alu_log,
  VOP1Op.V_RCP_F32: alu_rcp, VOP1Op.V_RCP_IFLAG_F32: alu_rcp, VOP1Op.V_RSQ_F32: alu_rsq, VOP1Op.V_SQRT_F32: alu_sqrt,
  VOP1Op.V_SIN_F32: lambda s: i32(math.sin(f32(s) * 2 * math.pi)), VOP1Op.V_COS_F32: lambda s: i32(math.cos(f32(s) * 2 * math.pi)),
  VOP1Op.V_NOT_B32: lambda s: (~s) & 0xffffffff, VOP1Op.V_BFREV_B32: lambda s: int(f'{s:032b}'[::-1], 2),
  VOP1Op.V_CLZ_I32_U32: clz, VOP1Op.V_CLS_I32: cls,
  VOP1Op.V_CVT_F32_UBYTE0: lambda s: i32(float(s & 0xff)), VOP1Op.V_CVT_F32_UBYTE1: lambda s: i32(float((s >> 8) & 0xff)),
  VOP1Op.V_CVT_F32_UBYTE2: lambda s: i32(float((s >> 16) & 0xff)), VOP1Op.V_CVT_F32_UBYTE3: lambda s: i32(float((s >> 24) & 0xff)),
  VOP1Op.V_FREXP_MANT_F32: lambda s: i32(math.frexp(f32(s))[0] if f32(s) != 0 else 0.0),
  VOP1Op.V_FREXP_EXP_I32_F32: lambda s: (math.frexp(f32(s))[1] if f32(s) != 0 else 0) & 0xffffffff,
}
def exec_vop1(st: WaveState, inst: VOP1, lane: int) -> None:
  if inst.op == VOP1Op.V_NOP: return
  if inst.op == VOP1Op.V_READFIRSTLANE_B32:
    first = (st.exec_mask & -st.exec_mask).bit_length() - 1 if st.exec_mask else 0
    st.wsgpr(inst.vdst, st.rsrc(inst.src0, first) if inst.src0 >= 256 else st.rsrc(inst.src0, lane)); return
  st.vgpr[lane][inst.vdst] = VOP1_OPS[inst.op](st.rsrc(inst.src0, lane))

# VOP2 table: op -> fn(s0, s1) -> result
VOP2_OPS = {
  VOP2Op.V_ADD_F32: lambda a, b: i32(f32(a)+f32(b)), VOP2Op.V_SUB_F32: lambda a, b: i32(f32(a)-f32(b)),
  VOP2Op.V_SUBREV_F32: lambda a, b: i32(f32(b)-f32(a)), VOP2Op.V_MUL_F32: lambda a, b: i32(f32(a)*f32(b)),
  VOP2Op.V_MIN_F32: lambda a, b: i32(min(f32(a), f32(b))), VOP2Op.V_MAX_F32: lambda a, b: i32(max(f32(a), f32(b))),
  VOP2Op.V_ADD_NC_U32: lambda a, b: (a + b) & 0xffffffff, VOP2Op.V_SUB_NC_U32: lambda a, b: (a - b) & 0xffffffff,
  VOP2Op.V_SUBREV_NC_U32: lambda a, b: (b - a) & 0xffffffff,
  VOP2Op.V_LSHLREV_B32: lambda a, b: (b << (a & 0x1f)) & 0xffffffff, VOP2Op.V_LSHRREV_B32: lambda a, b: b >> (a & 0x1f),
  VOP2Op.V_ASHRREV_I32: lambda a, b: (sext(b, 32) >> (a & 0x1f)) & 0xffffffff,
  VOP2Op.V_AND_B32: lambda a, b: a & b, VOP2Op.V_OR_B32: lambda a, b: a | b, VOP2Op.V_XOR_B32: lambda a, b: a ^ b,
  VOP2Op.V_XNOR_B32: lambda a, b: ~(a ^ b) & 0xffffffff, VOP2Op.V_MIN_U32: min, VOP2Op.V_MAX_U32: max,
  VOP2Op.V_MIN_I32: lambda a, b: a if sext(a, 32) < sext(b, 32) else b, VOP2Op.V_MAX_I32: lambda a, b: a if sext(a, 32) > sext(b, 32) else b,
  VOP2Op.V_MUL_I32_I24: lambda a, b: (sext(a & 0xffffff, 24) * sext(b & 0xffffff, 24)) & 0xffffffff,
  VOP2Op.V_MUL_HI_I32_I24: lambda a, b: ((sext(a & 0xffffff, 24) * sext(b & 0xffffff, 24)) >> 32) & 0xffffffff,
  VOP2Op.V_MUL_U32_U24: lambda a, b: ((a & 0xffffff) * (b & 0xffffff)) & 0xffffffff,
  VOP2Op.V_MUL_HI_U32_U24: lambda a, b: (((a & 0xffffff) * (b & 0xffffff)) >> 32) & 0xffffffff,
  VOP2Op.V_LDEXP_F16: lambda a, b: i16(math.ldexp(f16(a), sext(b, 32))),
  VOP2Op.V_ADD_F16: lambda a, b: i16(f16(a)+f16(b)), VOP2Op.V_SUB_F16: lambda a, b: i16(f16(a)-f16(b)),
  VOP2Op.V_MUL_F16: lambda a, b: i16(f16(a)*f16(b)), VOP2Op.V_MIN_F16: lambda a, b: i16(min(f16(a), f16(b))), VOP2Op.V_MAX_F16: lambda a, b: i16(max(f16(a), f16(b))),
}
def exec_vop2(st: WaveState, inst: VOP2, lane: int) -> None:
  V, s0, s1 = st.vgpr[lane], st.rsrc(inst.src0, lane), st.vgpr[lane][inst.vsrc1]
  if (fn := VOP2_OPS.get(inst.op)): V[inst.vdst] = fn(s0, s1); return
  if inst.op == VOP2Op.V_CNDMASK_B32: V[inst.vdst] = s1 if (st.vcc >> lane) & 1 else s0; return
  if inst.op == VOP2Op.V_FMAC_F32: V[inst.vdst] = i32(f32(s0)*f32(s1)+f32(V[inst.vdst])); return
  if inst.op == VOP2Op.V_FMAMK_F32: V[inst.vdst] = i32(f32(s0)*f32(st.literal)+f32(s1)); return
  if inst.op == VOP2Op.V_FMAAK_F32: V[inst.vdst] = i32(f32(s0)*f32(s1)+f32(st.literal)); return
  if inst.op == VOP2Op.V_ADD_CO_CI_U32: r = s0+s1+((st.vcc>>lane)&1); st.pend_vcc_lane(lane, r >= 0x100000000); V[inst.vdst] = r & 0xffffffff; return
  if inst.op == VOP2Op.V_SUB_CO_CI_U32: b = (st.vcc>>lane)&1; st.pend_vcc_lane(lane, s1+b > s0); V[inst.vdst] = (s0-s1-b) & 0xffffffff; return
  raise NotImplementedError(f"VOP2 op {inst.op}")

def vop3_mod(val: int, neg: int, abs_: int, idx: int) -> int:
  if (abs_ >> idx) & 1: val = i32(abs(f32(val)))
  if (neg >> idx) & 1: val = i32(-f32(val))
  return val

# VOP3-only ops: true 3-source operations (not VOP1/VOP2 promoted)
VOP3_ONLY_OPS = {
  VOP3Op.V_FMA_F32: lambda a, b, c: i32(f32(a)*f32(b)+f32(c)), VOP3Op.V_ADD3_U32: lambda a, b, c: (a + b + c) & 0xffffffff,
  VOP3Op.V_LSHL_ADD_U32: lambda a, b, c: ((a << (b & 0x1f)) + c) & 0xffffffff, VOP3Op.V_ADD_LSHL_U32: lambda a, b, c: ((a + b) << (c & 0x1f)) & 0xffffffff,
  VOP3Op.V_MAD_U32_U24: lambda a, b, c: ((a & 0xffffff) * (b & 0xffffff) + c) & 0xffffffff,
  VOP3Op.V_MAD_I32_I24: lambda a, b, c: (sext(a & 0xffffff, 24) * sext(b & 0xffffff, 24) + sext(c, 32)) & 0xffffffff,
  VOP3Op.V_ALIGNBIT_B32: lambda a, b, c: (((a << 32) | b) >> (c & 0x1f)) & 0xffffffff,
  VOP3Op.V_XAD_U32: lambda a, b, c: ((a * b) + c) & 0xffffffff, VOP3Op.V_LSHL_OR_B32: lambda a, b, c: ((a << (b & 0x1f)) | c) & 0xffffffff,
  VOP3Op.V_XOR3_B32: lambda a, b, c: a ^ b ^ c, VOP3Op.V_DIV_FMAS_F32: lambda a, b, c: i32(f32(a) * f32(b) + f32(c)),
  VOP3Op.V_MIN3_I32: lambda a, b, c: sorted([sext(a, 32), sext(b, 32), sext(c, 32)])[0] & 0xffffffff,
  VOP3Op.V_MAX3_I32: lambda a, b, c: sorted([sext(a, 32), sext(b, 32), sext(c, 32)])[2] & 0xffffffff,
  VOP3Op.V_MED3_I32: lambda a, b, c: sorted([sext(a, 32), sext(b, 32), sext(c, 32)])[1] & 0xffffffff,
  VOP3Op.V_BFE_U32: lambda a, b, c: (a >> (b & 0x1f)) & ((1 << (c & 0x1f)) - 1) if c & 0x1f else 0,
  VOP3Op.V_BFE_I32: lambda a, b, c: sext((a >> (b & 0x1f)) & ((1 << (c & 0x1f)) - 1), c & 0x1f) & 0xffffffff if c & 0x1f else 0,
  VOP3Op.V_DIV_FIXUP_F32: lambda a, b, c: i32(math.copysign(float('inf'), f32(c)) if f32(b) == 0.0 else f32(c) / f32(b)),
  # VOP3-only 2-source ops (not in VOP2)
  VOP3Op.V_MUL_LO_U32: lambda a, b, c: (a * b) & 0xffffffff, VOP3Op.V_MUL_HI_U32: lambda a, b, c: ((a * b) >> 32) & 0xffffffff,
  VOP3Op.V_MUL_HI_I32: lambda a, b, c: ((sext(a, 32) * sext(b, 32)) >> 32) & 0xffffffff,
  VOP3Op.V_LDEXP_F32: lambda a, b, c: i32(math.ldexp(f32(a), sext(b, 32))),
}

# Build VOP3_OPS dynamically: VOP3Op = VOP1Op + 384 for VOP1, VOP3Op = VOP2Op + 256 for VOP2
VOP3_OPS: dict = {**VOP3_ONLY_OPS}
for vop1_op, fn in VOP1_OPS.items(): VOP3_OPS[vop1_op + 384] = lambda a, b, c, f=fn: f(a)  # wrap 1-arg as 3-arg
for vop2_op, fn in VOP2_OPS.items(): VOP3_OPS[vop2_op + 256] = lambda a, b, c, f=fn: f(a, b)  # wrap 2-arg as 3-arg

def exec_vop3(st: WaveState, inst: VOP3, lane: int) -> None:
  op, src0, src1, src2, vdst, neg, abs_ = inst.op, inst.src0, inst.src1, inst.src2, inst.vdst, inst.neg, getattr(inst, 'abs', 0)
  s0, s1, s2 = vop3_mod(st.rsrc(src0, lane), neg, abs_, 0), vop3_mod(st.rsrc(src1, lane), neg, abs_, 1), vop3_mod(st.rsrc(src2, lane), neg, abs_, 2)
  V = st.vgpr[lane]
  if 0 <= op <= 255: exec_vopc_vop3(st, op, s0, s1, vdst, lane); return  # VOPC encoded in VOP3
  if (fn := VOP3_OPS.get(op)): V[vdst] = fn(s0, s1, s2); return
  match op:
    case VOP3Op.V_FMAC_F32: V[vdst] = i32(f32(s0)*f32(s1)+f32(V[vdst]))
    case VOP3Op.V_CNDMASK_B32: V[vdst] = s1 if ((st.rsrc(src2, lane) if src2 < 256 else st.vcc) >> lane) & 1 else s0
    case VOP3Op.V_LSHLREV_B64 | VOP3Op.V_LSHRREV_B64 | VOP3Op.V_ASHRREV_I64:
      v64 = st.rsrc64(src1, lane)
      r = ((v64 << (s0 & 0x3f)) & 0xffffffffffffffff if op == VOP3Op.V_LSHLREV_B64 else
           v64 >> (s0 & 0x3f) if op == VOP3Op.V_LSHRREV_B64 else sext(v64, 64) >> (s0 & 0x3f))
      V[vdst], V[vdst+1] = r & 0xffffffff, (r >> 32) & 0xffffffff
    case _: raise NotImplementedError(f"VOP3 op {op}")

def cmp_class_f32(val: int, mask: int) -> bool:
  f = f32(val)
  if math.isnan(f): return bool(mask & 0x3)
  if math.isinf(f): return bool(mask & (0x4 if f < 0 else 0x200))
  if f == 0.0: return bool(mask & (0x20 if (val >> 31) & 1 else 0x40))
  exp, sign = (val >> 23) & 0xff, (val >> 31) & 1
  return bool(mask & ((0x10 if sign else 0x80) if exp == 0 else (0x8 if sign else 0x100)))

def vopc_compare(op: int, s0: int, s1: int) -> bool:
  base = op & 0x7f
  if base == 126: return cmp_class_f32(s0, s1)
  if 16 <= base <= 31:
    f0, f1, cmp, nan = f32(s0), f32(s1), base - 16, math.isnan(f32(s0)) or math.isnan(f32(s1))
    return [False, f0<f1, f0==f1, f0<=f1, f0>f1, f0!=f1, f0>=f1, not nan, nan, f0<f1 or nan, f0==f1 or nan, f0<=f1 or nan, f0>f1 or nan, f0!=f1 or nan, f0>=f1 or nan, True][cmp]
  if 64 <= base <= 79:
    cmp, s0s, s1s = (base - 64) % 8, sext(s0,32), sext(s1,32)
    return [False, s0s<s1s, s0s==s1s, s0s<=s1s, s0s>s1s, s0s!=s1s, s0s>=s1s, True][cmp] if base < 72 else [False, s0<s1, s0==s1, s0<=s1, s0>s1, s0!=s1, s0>=s1, True][cmp]
  raise NotImplementedError(f"VOPC op {op}")

def exec_vopc_vop3(st: WaveState, op: int, s0: int, s1: int, sdst: int, lane: int) -> None:
  result, is_cmpx = vopc_compare(op, s0, s1), op >= 128
  (st.pend_vcc_lane if sdst == VCC_LO else lambda l, v: st.pend_sgpr_lane(sdst, l, v))(lane, result)
  if is_cmpx: st.pend_exec_lane(lane, result)

def exec_vopc(st: WaveState, inst: VOPC, lane: int) -> None:
  result, is_cmpx = vopc_compare(inst.op, st.rsrc(inst.src0, lane), st.vgpr[lane][inst.vsrc1]), inst.op >= 128
  (st.pend_exec_lane if is_cmpx else st.pend_vcc_lane)(lane, result)

def exec_vop3sd(st: WaveState, inst: VOP3SD, lane: int) -> None:
  op, src0, src1, src2, vdst, sdst, neg = inst.op, inst.src0, inst.src1, inst.src2, inst.vdst, inst.sdst, inst.neg
  s0, s1, s2 = st.rsrc(src0, lane), st.rsrc(src1, lane), st.rsrc(src2, lane)
  if (neg >> 0) & 1: s0 = i32(-f32(s0))
  if (neg >> 1) & 1: s1 = i32(-f32(s1))
  if (neg >> 2) & 1: s2 = i32(-f32(s2))
  V = st.vgpr[lane]
  match op:
    case VOP3SDOp.V_ADD_CO_U32:
      r = s0 + s1
      V[vdst] = r & 0xffffffff
      st.pend_sgpr_lane(sdst, lane, r >= 0x100000000)
    case VOP3SDOp.V_SUB_CO_U32:
      V[vdst] = (s0 - s1) & 0xffffffff
      st.pend_sgpr_lane(sdst, lane, s1 > s0)
    case VOP3SDOp.V_SUBREV_CO_U32:
      V[vdst] = (s1 - s0) & 0xffffffff
      st.pend_sgpr_lane(sdst, lane, s0 > s1)
    case VOP3SDOp.V_ADD_CO_CI_U32:
      cin = (st.rsgpr(src2) >> lane) & 1 if src2 < 256 else (st.vcc >> lane) & 1
      r = s0 + s1 + cin
      V[vdst] = r & 0xffffffff
      st.pend_sgpr_lane(sdst, lane, r >= 0x100000000)
    case VOP3SDOp.V_MAD_U64_U32:
      s2_64 = s2 | (st.rsrc(src2 + 1, lane) << 32)
      r = s0 * s1 + s2_64
      V[vdst] = r & 0xffffffff
      V[vdst+1] = (r >> 32) & 0xffffffff
    case VOP3SDOp.V_MAD_I64_I32:
      s2_64 = sext(s2 | (st.rsrc(src2 + 1, lane) << 32), 64)
      r = (sext(s0, 32) * sext(s1, 32) + s2_64) & 0xffffffffffffffff
      V[vdst] = r & 0xffffffff
      V[vdst+1] = (r >> 32) & 0xffffffff
    case VOP3SDOp.V_DIV_SCALE_F32:
      V[vdst] = 0
      st.pend_sgpr_lane(sdst, lane, False)
    case VOP3SDOp.V_DIV_SCALE_F64:
      V[vdst] = s0
      V[vdst+1] = st.rsrc(src0 + 1, lane)
      st.pend_vcc_lane(lane, s0 == s2)
    case _: raise NotImplementedError(f"VOP3SD op {op}")

def exec_flat(st: WaveState, inst: FLAT, lane: int) -> None:
  op, addr_reg, data_reg, vdst, offset, saddr, V = inst.op, inst.addr, inst.data, inst.vdst, sext(inst.offset, 13), inst.saddr, st.vgpr[lane]
  addr = V[addr_reg] | (V[addr_reg+1] << 32)
  addr = (st.rsgpr64(saddr) + V[addr_reg] + offset) & 0xffffffffffffffff if saddr not in (NULL_REG, 0x7f) else (addr + offset) & 0xffffffffffffffff
  if op in FLAT_LOAD:
    cnt, sz, sign = FLAT_LOAD[op]
    for i in range(cnt):
      val = mem_read(addr + i * sz, sz)
      V[vdst + i] = sext(val, sz * 8) & 0xffffffff if sign else val
  elif op in FLAT_STORE:
    cnt, sz = FLAT_STORE[op]
    for i in range(cnt): mem_write(addr + i * sz, sz, V[data_reg + i] & ((1 << (sz * 8)) - 1))
  else: raise NotImplementedError(f"FLAT op {op}")

def exec_ds(st: WaveState, inst: DS, lane: int, lds: bytearray) -> None:
  op, addr, vdst, V = inst.op, (st.vgpr[lane][inst.addr] + inst.offset0) & 0xffff, inst.vdst, st.vgpr[lane]
  if op in DS_LOAD:
    cnt, sz, sign = DS_LOAD[op]
    for i in range(cnt):
      val = int.from_bytes(lds[addr+i*sz:addr+i*sz+sz], 'little')
      V[vdst + i] = sext(val, sz * 8) & 0xffffffff if sign else val
  elif op in DS_STORE:
    cnt, sz = DS_STORE[op]
    for i in range(cnt): lds[addr+i*sz:addr+i*sz+sz] = (V[inst.data0 + i] & ((1 << (sz * 8)) - 1)).to_bytes(sz, 'little')
  else: raise NotImplementedError(f"DS op {op}")

# VOPD ops: pure 2-arg ops (same signature as VOP2)
VOPD_OPS = {
  VOPDOp.V_DUAL_MUL_F32: lambda a, b: i32(f32(a)*f32(b)), VOPDOp.V_DUAL_ADD_F32: lambda a, b: i32(f32(a)+f32(b)),
  VOPDOp.V_DUAL_SUB_F32: lambda a, b: i32(f32(a)-f32(b)), VOPDOp.V_DUAL_SUBREV_F32: lambda a, b: i32(f32(b)-f32(a)),
  VOPDOp.V_DUAL_MAX_F32: lambda a, b: i32(max(f32(a), f32(b))), VOPDOp.V_DUAL_MIN_F32: lambda a, b: i32(min(f32(a), f32(b))),
  VOPDOp.V_DUAL_MUL_DX9_ZERO_F32: lambda a, b: i32(0.0 if f32(a) == 0.0 or f32(b) == 0.0 else f32(a)*f32(b)),
  VOPDOp.V_DUAL_MOV_B32: lambda a, b: a, VOPDOp.V_DUAL_ADD_NC_U32: lambda a, b: (a + b) & 0xffffffff,
  VOPDOp.V_DUAL_LSHLREV_B32: lambda a, b: (b << (a & 0x1f)) & 0xffffffff, VOPDOp.V_DUAL_AND_B32: lambda a, b: a & b,
}
def exec_vopd_op(st: WaveState, op: int, src0: int, src1: int, dst: int, lane: int) -> None:
  s0, s1, V = st.rsrc(src0, lane), st.vgpr[lane][src1], st.vgpr[lane]
  if (fn := VOPD_OPS.get(op)): V[dst] = fn(s0, s1); return
  # Special cases: ops that need dest register or literal
  if op == VOPDOp.V_DUAL_FMAC_F32: V[dst] = i32(f32(s0)*f32(s1)+f32(V[dst])); return
  if op == VOPDOp.V_DUAL_FMAAK_F32: V[dst] = i32(f32(s0)*f32(s1)+f32(st.literal)); return
  if op == VOPDOp.V_DUAL_FMAMK_F32: V[dst] = i32(f32(s0)*f32(st.literal)+f32(s1)); return
  if op == VOPDOp.V_DUAL_CNDMASK_B32: V[dst] = s1 if (st.vcc >> lane) & 1 else s0; return
  raise NotImplementedError(f"VOPD op {op}")

def exec_vopd(st: WaveState, inst: VOPD, lane: int) -> None:
  exec_vopd_op(st, inst.opx, inst.srcx0, inst.vsrcx1, inst.vdstx, lane)
  exec_vopd_op(st, inst.opy, inst.srcy0, inst.vsrcy1, (inst.vdsty << 1) | ((inst.vdstx & 1) ^ 1), lane)

def exec_vop3p(st: WaveState, inst: VOP3P, lane: int) -> None:
  op, vdst, V = inst.op, inst.vdst, st.vgpr[lane]
  s0, s1, s2 = st.rsrc(inst.src0, lane), st.rsrc(inst.src1, lane), st.rsrc(inst.src2, lane)
  opsel, opsel_hi = [(inst.opsel >> i) & 1 for i in range(3)], [(inst.opsel_hi >> i) & 1 for i in range(2)] + [inst.opsel_hi2]
  neg, neg_hi = inst.neg, inst.neg_hi

  def get_src(src: int, idx: int, for_mix: bool = False) -> float:
    if for_mix:
      if not opsel_hi[idx]: return abs(f32(src)) if (neg_hi >> idx) & 1 else f32(src)
      return float(f16((src >> 16) & 0xffff) if opsel[idx] else f16(src & 0xffff))
    use_hi = opsel_hi[idx] if for_mix else opsel[idx]
    val = ((src >> 16) & 0xffff) if use_hi else (src & 0xffff)
    f = f16(val)
    if use_hi and (neg >> idx) & 1: f = -f
    elif not use_hi and (neg_hi >> idx) & 1: f = -f
    return f

  match op:
    case VOP3POp.V_FMA_MIX_F32:    V[vdst] = i32(get_src(s0, 0, True) * get_src(s1, 1, True) + get_src(s2, 2, True))
    case VOP3POp.V_FMA_MIXLO_F16:  V[vdst] = (V[vdst] & 0xffff0000) | i16(get_src(s0, 0, True) * get_src(s1, 1, True) + get_src(s2, 2, True))
    case VOP3POp.V_FMA_MIXHI_F16:  V[vdst] = (V[vdst] & 0x0000ffff) | (i16(get_src(s0, 0, True) * get_src(s1, 1, True) + get_src(s2, 2, True)) << 16)
    case _: raise NotImplementedError(f"VOP3P op {op}")

SCALAR = {SOP1: exec_sop1, SOP2: exec_sop2, SOPC: exec_sopc, SOPK: exec_sopk, SOPP: exec_sopp, SMEM: exec_smem}
VECTOR = {VOP1: exec_vop1, VOP2: exec_vop2, VOP3: exec_vop3, VOP3SD: exec_vop3sd, VOPC: exec_vopc, FLAT: exec_flat, DS: exec_ds, VOPD: exec_vopd, VOP3P: exec_vop3p}

def step_wave(program: Program, st: WaveState, lds: bytearray, n_lanes: int) -> int:
  inst = program.get(st.pc)
  if inst is None: return 1
  inst_words = inst.size() // 4
  st.literal = inst._literal or 0
  inst_type = type(inst)
  handler = SCALAR.get(inst_type)
  if handler is not None:
    delta = handler(st, inst)
    if delta == -1: return -1
    if delta == -2: st.pc += inst_words; return -2
    st.pc += inst_words + delta
  else:
    handler = VECTOR[inst_type]
    exec_mask = st.exec_mask
    if inst_type is DS:
      for lane in range(n_lanes):
        if exec_mask & (1 << lane): handler(st, inst, lane, lds)
    else:
      for lane in range(n_lanes):
        if exec_mask & (1 << lane): handler(st, inst, lane)
    st.commit_pends()
    st.pc += inst_words
  return 0

def exec_wave(program: Program, st: WaveState, lds: bytearray, n_lanes: int, wg_id: tuple[int,int,int]=(0,0,0), local_size: tuple[int,int,int]=(1,1,1), wave_start: int=0) -> int:
  while st.pc in program:
    result = step_wave(program, st, lds, n_lanes)
    if result == -1: return 0
    if result == -2: return -2
  return 0

def exec_workgroup(program: Program, workgroup_id: tuple[int, int, int], local_size: tuple[int, int, int], args_ptr: int, dispatch_dim: int) -> None:
  lx, ly, lz = local_size
  total_threads, lds = lx * ly * lz, bytearray(65536)
  waves: list[tuple[WaveState, int, int]] = []
  for wave_start in range(0, total_threads, WAVE_SIZE):
    n_lanes = min(WAVE_SIZE, total_threads - wave_start)
    st = WaveState()
    st.exec_mask = (1 << n_lanes) - 1
    st.wsgpr64(0, args_ptr)
    gx, gy, gz = workgroup_id
    if dispatch_dim >= 3: st.sgpr[13], st.sgpr[14], st.sgpr[15] = gx, gy, gz
    elif dispatch_dim == 2: st.sgpr[14], st.sgpr[15] = gx, gy
    else: st.sgpr[15] = gx
    for i in range(n_lanes):
      tid = wave_start + i
      st.vgpr[i][0] = tid if local_size == (lx, 1, 1) else ((tid // (lx * ly)) << 20) | (((tid // lx) % ly) << 10) | (tid % lx)
    waves.append((st, n_lanes, wave_start))
  has_barrier = any(isinstance(inst, SOPP) and inst.op == SOPPOp.S_BARRIER for inst in program.values())
  for _ in range(2 if has_barrier else 1):
    for st, n_lanes, wave_start in waves: exec_wave(program, st, lds, n_lanes, workgroup_id, local_size, wave_start)

def run_asm(lib: int, lib_sz: int, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, args_ptr: int) -> int:
  data = (ctypes.c_char * lib_sz).from_address(lib).raw
  program = decode_program(data)
  if not program: return -1
  dispatch_dim = 3 if gz > 1 else (2 if gy > 1 else 1)
  for gidz in range(gz):
    for gidy in range(gy):
      for gidx in range(gx): exec_workgroup(program, (gidx, gidy, gidz), (lx, ly, lz), args_ptr, dispatch_dim)
  return 0
