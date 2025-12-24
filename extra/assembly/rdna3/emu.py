# RDNA3 emulator - pure Python implementation for testing
from __future__ import annotations
import ctypes, struct, math
from dataclasses import dataclass, field
from typing import Any
from extra.assembly.rdna3.lib import Inst32, Inst64, bits
from extra.assembly.rdna3.autogen import (
  SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, VOP1, VOP2, VOP3, VOP3SD, DS, FLAT, VOPD,
  SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, SMEMOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, DSOp, FLATOp, GLOBALOp, VOPDOp
)

class VOPC(Inst32):
  encoding = bits[31:25] == 0b0111110
  op, src0, vsrc1 = bits[24:17], bits[8:0], bits[16:9]

# Type aliases
Inst = Inst32 | Inst64
Program = list[tuple[Inst, int]]

# constants and helpers
WAVE_SIZE, SGPR_COUNT, VGPR_COUNT = 32, 128, 256
VCC_LO, VCC_HI, EXEC_LO, EXEC_HI, NULL_REG, M0 = 106, 107, 126, 127, 124, 125
FLOAT_BITS: dict[int, int] = {240: 0x3f000000, 241: 0xbf000000, 242: 0x3f800000, 243: 0xbf800000,
                               244: 0x40000000, 245: 0xc0000000, 246: 0x40800000, 247: 0xc0800000, 248: 0x3e22f983}

def f32(i: int) -> float: return struct.unpack('<f', struct.pack('<I', i & 0xffffffff))[0]
def i32(f: float) -> int: return struct.unpack('<I', struct.pack('<f', f))[0]
def sext(v: int, b: int) -> int: return v - (1 << b) if v & (1 << (b-1)) else v
def clz(x: int) -> int: return 32 - x.bit_length() if x else 32
def cls(x: int) -> int: x &= 0xffffffff; return 31 if x in (0, 0xffffffff) else clz(~x & 0xffffffff if x >> 31 else x) - 1

# aliases for test compatibility
bits_to_f32, f32_to_bits, sign_ext = f32, i32, sext

@dataclass
class WaveState:
  sgpr: list[int] = field(default_factory=lambda: [0] * SGPR_COUNT)
  vgpr: list[list[int]] = field(default_factory=lambda: [[0] * VGPR_COUNT for _ in range(WAVE_SIZE)])
  scc: int = 0
  vcc: int = 0
  exec_mask: int = 0xffffffff
  pc: int = 0
  literal: int = 0

def decode_format(word: int) -> tuple[type[Inst] | None, bool]:
  hi2 = (word >> 30) & 0x3
  if hi2 == 0b11:
    enc = (word >> 26) & 0xf
    if enc == 0b1101: return SMEM, True
    if enc == 0b0101:
      op = (word >> 16) & 0x3ff
      return (VOP3SD, True) if op in (288, 289, 290, 764, 765, 766, 767, 768, 769, 770) else (VOP3, True)
    return {0b0011: (None, True), 0b0110: (DS, True), 0b0111: (FLAT, True), 0b0010: (VOPD, True)}.get(enc, (None, True))
  if hi2 == 0b10:
    enc = (word >> 23) & 0x7f
    return {0b1111101: (SOP1, False), 0b1111110: (SOPC, False), 0b1111111: (SOPP, False)}.get(enc, (SOPK, False) if ((word >> 28) & 0xf) == 0b1011 else (SOP2, False))
  enc = (word >> 25) & 0x7f
  return (VOPC, False) if enc == 0b0111110 else (VOP1, False) if enc == 0b0111111 else (VOP2, False)

def decode_program(data: bytes) -> Program:
  result: Program = []
  i = 0
  while i < len(data):
    word = int.from_bytes(data[i:i+4], 'little')
    inst_class, is_64 = decode_format(word)
    if inst_class is None: i += 4; continue
    size = 8 if is_64 else 4
    inst = inst_class.from_bytes(data[i:i+size])
    for fld in ['src0', 'src1', 'src2', 'ssrc0', 'ssrc1', 'srcx0', 'srcy0']:
      if getattr(inst, fld, None) == 255: inst._literal = int.from_bytes(data[i+size:i+size+4], 'little'); size += 4; break
    else: inst._literal = None
    result.append((inst, size // 4)); i += size
    if inst_class == SOPP and inst.op == SOPPOp.S_ENDPGM: break
  return result

# register read/write helpers
def rsgpr(st: WaveState, i: int) -> int:
  return {VCC_LO: st.vcc & 0xffffffff, VCC_HI: (st.vcc >> 32) & 0xffffffff, EXEC_LO: st.exec_mask & 0xffffffff,
          EXEC_HI: (st.exec_mask >> 32) & 0xffffffff, NULL_REG: 0, 253: st.scc}.get(i, st.sgpr[i] if i < SGPR_COUNT else 0)

def wsgpr(st: WaveState, i: int, v: int) -> None:
  v &= 0xffffffff
  if i == VCC_LO: st.vcc = (st.vcc & 0xffffffff00000000) | v
  elif i == VCC_HI: st.vcc = (st.vcc & 0xffffffff) | (v << 32)
  elif i == EXEC_LO: st.exec_mask = (st.exec_mask & 0xffffffff00000000) | v
  elif i == EXEC_HI: st.exec_mask = (st.exec_mask & 0xffffffff) | (v << 32)
  elif i < SGPR_COUNT: st.sgpr[i] = v

def rsgpr64(st: WaveState, i: int) -> int: return rsgpr(st, i) | (rsgpr(st, i+1) << 32)
def wsgpr64(st: WaveState, i: int, v: int) -> None: wsgpr(st, i, v & 0xffffffff); wsgpr(st, i+1, (v >> 32) & 0xffffffff)

# aliases for test compatibility
write_sgpr, write_sgpr64, read_sgpr, read_sgpr64 = wsgpr, wsgpr64, rsgpr, rsgpr64

def rsrc(st: WaveState, v: int, lane: int) -> int:
  if v <= 105: return st.sgpr[v]
  if v in (VCC_LO, VCC_HI): return (st.vcc >> (32 if v == VCC_HI else 0)) & 0xffffffff
  if 108 <= v <= 123 or v == M0: return st.sgpr[v]
  if v in (EXEC_LO, EXEC_HI): return (st.exec_mask >> (32 if v == EXEC_HI else 0)) & 0xffffffff
  if v == NULL_REG: return 0
  if 128 <= v <= 192: return v - 128
  if 193 <= v <= 208: return (-(v - 192)) & 0xffffffff
  if v in FLOAT_BITS: return FLOAT_BITS[v]
  if v == 255: return st.literal
  if 256 <= v <= 511: return st.vgpr[lane][v - 256]
  return 0

def rsrc64(st: WaveState, v: int, lane: int) -> int:
  return rsrc(st, v, lane) | ((rsrc(st, v+1, lane) if v <= 105 or 256 <= v <= 511 else 0) << 32)

# instruction execution
def exec_sop1(st: WaveState, inst: SOP1) -> int:
  op, s0, sdst = inst.op, rsrc(st, inst.ssrc0, 0), inst.sdst
  OPS: dict[int, Any] = {
    SOP1Op.S_MOV_B32: lambda: wsgpr(st, sdst, s0), SOP1Op.S_MOV_B64: lambda: wsgpr64(st, sdst, rsrc64(st, inst.ssrc0, 0)),
    SOP1Op.S_BREV_B32: lambda: wsgpr(st, sdst, int(f'{s0:032b}'[::-1], 2)), SOP1Op.S_CLZ_I32_U32: lambda: wsgpr(st, sdst, clz(s0)),
    SOP1Op.S_CLS_I32: lambda: wsgpr(st, sdst, cls(s0)), SOP1Op.S_SEXT_I32_I8: lambda: wsgpr(st, sdst, sext(s0 & 0xff, 8) & 0xffffffff),
    SOP1Op.S_SEXT_I32_I16: lambda: wsgpr(st, sdst, sext(s0 & 0xffff, 16) & 0xffffffff),
    SOP1Op.S_BITSET0_B32: lambda: wsgpr(st, sdst, rsgpr(st, sdst) & ~(1 << (s0 & 0x1f))),
    SOP1Op.S_BITSET1_B32: lambda: wsgpr(st, sdst, rsgpr(st, sdst) | (1 << (s0 & 0x1f)))}
  if op in OPS: OPS[op]()
  elif op == SOP1Op.S_NOT_B32: r = (~s0) & 0xffffffff; st.scc = int(r != 0); wsgpr(st, sdst, r)
  elif op == SOP1Op.S_NOT_B64: r = (~rsrc64(st, inst.ssrc0, 0)) & 0xffffffffffffffff; st.scc = int(r != 0); wsgpr64(st, sdst, r)
  elif op == SOP1Op.S_ABS_I32: r = abs(sext(s0, 32)) & 0xffffffff; st.scc = int(r != 0); wsgpr(st, sdst, r)
  elif op == SOP1Op.S_AND_SAVEEXEC_B32: old = st.exec_mask & 0xffffffff; st.exec_mask = s0 & old; st.scc = int(st.exec_mask != 0); wsgpr(st, sdst, old)
  elif op == SOP1Op.S_OR_SAVEEXEC_B32: old = st.exec_mask & 0xffffffff; st.exec_mask = s0 | old; st.scc = int(st.exec_mask != 0); wsgpr(st, sdst, old)
  elif op == SOP1Op.S_AND_NOT1_SAVEEXEC_B32: old = st.exec_mask & 0xffffffff; st.exec_mask = s0 & (~old & 0xffffffff); st.scc = int(st.exec_mask != 0); wsgpr(st, sdst, old)
  else: raise NotImplementedError(f"SOP1 op {op}")
  return 0

def exec_sop2(st: WaveState, inst: SOP2) -> int:
  op, ssrc0, ssrc1, sdst = inst.op, inst.ssrc0, inst.ssrc1, inst.sdst
  s0, s1 = rsrc(st, ssrc0, 0), rsrc(st, ssrc1, 0)
  def w(r: int, scc: int | None = None) -> None: wsgpr(st, sdst, r & 0xffffffff); st.scc = scc if scc is not None else st.scc
  def w64(r: int, scc: int | None = None) -> None: wsgpr64(st, sdst, r & 0xffffffffffffffff); st.scc = scc if scc is not None else st.scc
  if op == SOP2Op.S_ADD_U32: r = s0 + s1; w(r, int(r >= 0x100000000))
  elif op == SOP2Op.S_SUB_U32: w(s0 - s1, int(s1 > s0))
  elif op == SOP2Op.S_ADD_I32: r = sext(s0,32) + sext(s1,32); w(r, int(((s0>>31)==(s1>>31)) and ((s0>>31)!=((r>>31)&1))))
  elif op == SOP2Op.S_SUB_I32: r = sext(s0,32) - sext(s1,32); w(r, int(((s0>>31)!=(s1>>31)) and ((s0>>31)!=((r>>31)&1))))
  elif op == SOP2Op.S_ADDC_U32: r = s0 + s1 + st.scc; w(r, int(r >= 0x100000000))
  elif op == SOP2Op.S_SUBB_U32: w(s0 - s1 - st.scc, int((s1 + st.scc) > s0))
  elif op == SOP2Op.S_MUL_I32: w(sext(s0,32) * sext(s1,32))
  elif op == SOP2Op.S_MUL_HI_U32: w((s0 * s1) >> 32)
  elif op == SOP2Op.S_MUL_HI_I32: w((sext(s0,32) * sext(s1,32)) >> 32)
  elif op == SOP2Op.S_LSHL_B32: r = (s0 << (s1 & 0x1f)) & 0xffffffff; w(r, int(r != 0))
  elif op == SOP2Op.S_LSHL_B64: r = (rsrc64(st, ssrc0, 0) << (s1 & 0x3f)) & 0xffffffffffffffff; w64(r, int(r != 0))
  elif op == SOP2Op.S_LSHR_B32: r = s0 >> (s1 & 0x1f); w(r, int(r != 0))
  elif op == SOP2Op.S_LSHR_B64: r = rsrc64(st, ssrc0, 0) >> (s1 & 0x3f); w64(r, int(r != 0))
  elif op == SOP2Op.S_ASHR_I32: r = sext(s0, 32) >> (s1 & 0x1f); w(r, int(r != 0))
  elif op == SOP2Op.S_ASHR_I64: r = sext(rsrc64(st, ssrc0, 0), 64) >> (s1 & 0x3f); w64(r, int(r != 0))
  elif op == SOP2Op.S_AND_B32: r = s0 & s1; w(r, int(r != 0))
  elif op == SOP2Op.S_AND_B64: r = rsrc64(st, ssrc0, 0) & rsrc64(st, ssrc1, 0); w64(r, int(r != 0))
  elif op == SOP2Op.S_OR_B32: r = s0 | s1; w(r, int(r != 0))
  elif op == SOP2Op.S_OR_B64: r = rsrc64(st, ssrc0, 0) | rsrc64(st, ssrc1, 0); w64(r, int(r != 0))
  elif op == SOP2Op.S_XOR_B32: r = s0 ^ s1; w(r, int(r != 0))
  elif op == SOP2Op.S_XOR_B64: r = rsrc64(st, ssrc0, 0) ^ rsrc64(st, ssrc1, 0); w64(r, int(r != 0))
  elif op == SOP2Op.S_AND_NOT1_B32: r = s0 & (~s1 & 0xffffffff); w(r, int(r != 0))
  elif op == SOP2Op.S_OR_NOT1_B32: r = s0 | (~s1 & 0xffffffff); w(r, int(r != 0))
  elif op == SOP2Op.S_MIN_I32: st.scc = int(sext(s0,32) < sext(s1,32)); w(s0 if st.scc else s1)
  elif op == SOP2Op.S_MIN_U32: st.scc = int(s0 < s1); w(min(s0, s1))
  elif op == SOP2Op.S_MAX_I32: st.scc = int(sext(s0,32) > sext(s1,32)); w(s0 if st.scc else s1)
  elif op == SOP2Op.S_MAX_U32: st.scc = int(s0 > s1); w(max(s0, s1))
  elif op == SOP2Op.S_CSELECT_B32: w(s0 if st.scc else s1)
  elif op == SOP2Op.S_CSELECT_B64: w64(rsrc64(st, ssrc0, 0) if st.scc else rsrc64(st, ssrc1, 0))
  elif op == SOP2Op.S_BFE_U32: off, wd = s1 & 0x1f, (s1 >> 16) & 0x7f; r = (s0 >> off) & ((1 << wd) - 1) if wd else 0; w(r, int(r != 0))
  elif op == SOP2Op.S_BFE_I32: off, wd = s1 & 0x1f, (s1 >> 16) & 0x7f; r = sext((s0 >> off) & ((1 << wd) - 1), wd) & 0xffffffff if wd else 0; w(r, int(r != 0))
  elif op == SOP2Op.S_PACK_LL_B32_B16: w((s0 & 0xffff) | ((s1 & 0xffff) << 16))
  elif op == SOP2Op.S_PACK_LH_B32_B16: w((s0 & 0xffff) | (s1 & 0xffff0000))
  elif op == SOP2Op.S_PACK_HH_B32_B16: w(((s0 >> 16) & 0xffff) | (s1 & 0xffff0000))
  elif op == SOP2Op.S_PACK_HL_B32_B16: w(((s0 >> 16) & 0xffff) | ((s1 & 0xffff) << 16))
  else: raise NotImplementedError(f"SOP2 op {op}")
  return 0

def exec_sopc(st: WaveState, inst: SOPC) -> int:
  op, s0, s1 = inst.op, rsrc(st, inst.ssrc0, 0), rsrc(st, inst.ssrc1, 0)
  I32_CMP: dict[int, Any] = {SOPCOp.S_CMP_EQ_I32: lambda: sext(s0,32)==sext(s1,32), SOPCOp.S_CMP_LG_I32: lambda: sext(s0,32)!=sext(s1,32),
              SOPCOp.S_CMP_GT_I32: lambda: sext(s0,32)>sext(s1,32), SOPCOp.S_CMP_GE_I32: lambda: sext(s0,32)>=sext(s1,32),
              SOPCOp.S_CMP_LT_I32: lambda: sext(s0,32)<sext(s1,32), SOPCOp.S_CMP_LE_I32: lambda: sext(s0,32)<=sext(s1,32)}
  U32_CMP: dict[int, bool] = {SOPCOp.S_CMP_EQ_U32: s0==s1, SOPCOp.S_CMP_LG_U32: s0!=s1, SOPCOp.S_CMP_GT_U32: s0>s1,
             SOPCOp.S_CMP_GE_U32: s0>=s1, SOPCOp.S_CMP_LT_U32: s0<s1, SOPCOp.S_CMP_LE_U32: s0<=s1}
  if op in I32_CMP: st.scc = int(I32_CMP[op]())
  elif op in U32_CMP: st.scc = int(U32_CMP[op])
  elif op == SOPCOp.S_BITCMP0_B32: st.scc = int((s0 & (1 << (s1 & 0x1f)))==0)
  elif op == SOPCOp.S_BITCMP1_B32: st.scc = int((s0 & (1 << (s1 & 0x1f)))!=0)
  elif op == SOPCOp.S_CMP_EQ_U64: st.scc = int(rsrc64(st, inst.ssrc0, 0) == rsrc64(st, inst.ssrc1, 0))
  elif op == SOPCOp.S_CMP_LG_U64: st.scc = int(rsrc64(st, inst.ssrc0, 0) != rsrc64(st, inst.ssrc1, 0))
  else: raise NotImplementedError(f"SOPC op {op}")
  return 0

def exec_sopk(st: WaveState, inst: SOPK) -> int:
  op, sdst, simm = inst.op, inst.sdst, sext(inst.simm16, 16)
  s0 = rsgpr(st, sdst)
  CMPK: dict[int, bool] = {SOPKOp.S_CMPK_EQ_I32: sext(s0,32)==simm, SOPKOp.S_CMPK_LG_I32: sext(s0,32)!=simm, SOPKOp.S_CMPK_GT_I32: sext(s0,32)>simm,
          SOPKOp.S_CMPK_GE_I32: sext(s0,32)>=simm, SOPKOp.S_CMPK_LT_I32: sext(s0,32)<simm, SOPKOp.S_CMPK_LE_I32: sext(s0,32)<=simm,
          SOPKOp.S_CMPK_EQ_U32: s0==(simm&0xffff), SOPKOp.S_CMPK_LG_U32: s0!=(simm&0xffff), SOPKOp.S_CMPK_GT_U32: s0>(simm&0xffff),
          SOPKOp.S_CMPK_GE_U32: s0>=(simm&0xffff), SOPKOp.S_CMPK_LT_U32: s0<(simm&0xffff), SOPKOp.S_CMPK_LE_U32: s0<=(simm&0xffff)}
  if op == SOPKOp.S_MOVK_I32: wsgpr(st, sdst, simm & 0xffffffff)
  elif op in CMPK: st.scc = int(CMPK[op])
  elif op == SOPKOp.S_ADDK_I32: r = sext(s0,32) + simm; st.scc = int(((s0>>31)==((simm>>15)&1)) and ((s0>>31)!=((r>>31)&1))); wsgpr(st, sdst, r & 0xffffffff)
  elif op == SOPKOp.S_MULK_I32: wsgpr(st, sdst, (sext(s0,32) * simm) & 0xffffffff)
  else: raise NotImplementedError(f"SOPK op {op}")
  return 0

def exec_sopp(st: WaveState, inst: SOPP) -> int:
  op, simm = inst.op, sext(inst.simm16, 16)
  BRANCH: dict[int, bool] = {SOPPOp.S_BRANCH: True, SOPPOp.S_CBRANCH_SCC0: st.scc==0, SOPPOp.S_CBRANCH_SCC1: st.scc==1,
            SOPPOp.S_CBRANCH_VCCZ: st.vcc==0, SOPPOp.S_CBRANCH_VCCNZ: st.vcc!=0,
            SOPPOp.S_CBRANCH_EXECZ: st.exec_mask==0, SOPPOp.S_CBRANCH_EXECNZ: st.exec_mask!=0}
  if op == SOPPOp.S_ENDPGM: return -1
  if op == SOPPOp.S_BARRIER: return -2
  if op in BRANCH: return simm if BRANCH[op] else 0
  return 0

def exec_smem(st: WaveState, inst: SMEM) -> int:
  op, sbase, sdata, offset, soff_idx = inst.op, inst.sbase * 2, inst.sdata, sext(inst.offset, 21), inst.soffset
  addr = (rsgpr64(st, sbase) + offset + (rsrc(st, soff_idx, 0) if soff_idx not in (NULL_REG, 0x7f) else 0)) & 0xffffffffffffffff
  LOADS: dict[int, int] = {SMEMOp.S_LOAD_B32: 1, SMEMOp.S_LOAD_B64: 2, SMEMOp.S_LOAD_B128: 4, SMEMOp.S_LOAD_B256: 8, SMEMOp.S_LOAD_B512: 16}
  if op in LOADS:
    for i in range(LOADS[op]): wsgpr(st, sdata + i, ctypes.c_uint32.from_address(addr + i * 4).value)
  else: raise NotImplementedError(f"SMEM op {op}")
  return 0

def exec_vop1(st: WaveState, inst: VOP1, lane: int) -> None:
  op, s0, vdst = inst.op, rsrc(st, inst.src0, lane), inst.vdst
  V = st.vgpr[lane]
  SIMPLE: dict[int, Any] = {
    VOP1Op.V_NOP: lambda: None, VOP1Op.V_MOV_B32: lambda: s0,
    VOP1Op.V_CVT_F32_I32: lambda: i32(float(sext(s0, 32))), VOP1Op.V_CVT_F32_U32: lambda: i32(float(s0)),
    VOP1Op.V_CVT_U32_F32: lambda: max(0, min(0xffffffff, int(f32(s0)))),
    VOP1Op.V_CVT_I32_F32: lambda: max(-0x80000000, min(0x7fffffff, int(f32(s0)))) & 0xffffffff,
    VOP1Op.V_CVT_F16_F32: lambda: struct.unpack('<H', struct.pack('<e', f32(s0)))[0],
    VOP1Op.V_CVT_F32_F16: lambda: i32(struct.unpack('<e', struct.pack('<H', s0 & 0xffff))[0]),
    VOP1Op.V_TRUNC_F32: lambda: i32(math.trunc(f32(s0))), VOP1Op.V_CEIL_F32: lambda: i32(math.ceil(f32(s0))),
    VOP1Op.V_RNDNE_F32: lambda: i32(round(f32(s0))), VOP1Op.V_FLOOR_F32: lambda: i32(math.floor(f32(s0))),
    VOP1Op.V_EXP_F32: lambda: i32(math.pow(2.0, f32(s0))), VOP1Op.V_LOG_F32: lambda: i32(math.log2(f32(s0)) if f32(s0) > 0 else float('-inf')),
    VOP1Op.V_RCP_F32: lambda: i32(1.0 / f32(s0) if f32(s0) != 0 else math.copysign(float('inf'), f32(s0))),
    VOP1Op.V_RCP_IFLAG_F32: lambda: i32(1.0 / f32(s0) if f32(s0) != 0 else math.copysign(float('inf'), f32(s0))),
    VOP1Op.V_RSQ_F32: lambda: i32(1.0 / math.sqrt(f32(s0)) if f32(s0) > 0 else float('inf')),
    VOP1Op.V_SQRT_F32: lambda: i32(math.sqrt(max(0, f32(s0)))),
    VOP1Op.V_SIN_F32: lambda: i32(math.sin(f32(s0) * 2 * math.pi)), VOP1Op.V_COS_F32: lambda: i32(math.cos(f32(s0) * 2 * math.pi)),
    VOP1Op.V_NOT_B32: lambda: (~s0) & 0xffffffff, VOP1Op.V_BFREV_B32: lambda: int(f'{s0:032b}'[::-1], 2),
    VOP1Op.V_CLZ_I32_U32: lambda: clz(s0), VOP1Op.V_CLS_I32: lambda: cls(s0),
    VOP1Op.V_CVT_F32_UBYTE0: lambda: i32(float(s0 & 0xff)), VOP1Op.V_CVT_F32_UBYTE1: lambda: i32(float((s0 >> 8) & 0xff)),
    VOP1Op.V_CVT_F32_UBYTE2: lambda: i32(float((s0 >> 16) & 0xff)), VOP1Op.V_CVT_F32_UBYTE3: lambda: i32(float((s0 >> 24) & 0xff))}
  if op == VOP1Op.V_READFIRSTLANE_B32:
    first = (st.exec_mask & 0xffffffff).bit_length() - 1 if st.exec_mask else 0
    wsgpr(st, vdst, rsrc(st, inst.src0, first) if inst.src0 >= 256 else s0)
  elif op in SIMPLE:
    r = SIMPLE[op]()
    if r is not None: V[vdst] = r
  else: raise NotImplementedError(f"VOP1 op {op}")

def exec_vop2(st: WaveState, inst: VOP2, lane: int) -> None:
  op, s0, s1, vdst = inst.op, rsrc(st, inst.src0, lane), st.vgpr[lane][inst.vsrc1], inst.vdst
  V = st.vgpr[lane]
  ARITH: dict[int, Any] = {
    VOP2Op.V_ADD_F32: lambda: i32(f32(s0)+f32(s1)), VOP2Op.V_SUB_F32: lambda: i32(f32(s0)-f32(s1)),
    VOP2Op.V_SUBREV_F32: lambda: i32(f32(s1)-f32(s0)), VOP2Op.V_MUL_F32: lambda: i32(f32(s0)*f32(s1)),
    VOP2Op.V_MIN_F32: lambda: i32(min(f32(s0),f32(s1))), VOP2Op.V_MAX_F32: lambda: i32(max(f32(s0),f32(s1))),
    VOP2Op.V_FMAC_F32: lambda: i32(f32(s0)*f32(s1)+f32(V[vdst])),
    VOP2Op.V_ADD_NC_U32: lambda: (s0+s1)&0xffffffff, VOP2Op.V_SUB_NC_U32: lambda: (s0-s1)&0xffffffff,
    VOP2Op.V_SUBREV_NC_U32: lambda: (s1-s0)&0xffffffff,
    VOP2Op.V_MUL_I32_I24: lambda: (sext(s0&0xffffff,24)*sext(s1&0xffffff,24))&0xffffffff,
    VOP2Op.V_MUL_HI_I32_I24: lambda: ((sext(s0&0xffffff,24)*sext(s1&0xffffff,24))>>32)&0xffffffff,
    VOP2Op.V_MUL_U32_U24: lambda: ((s0&0xffffff)*(s1&0xffffff))&0xffffffff,
    VOP2Op.V_MUL_HI_U32_U24: lambda: (((s0&0xffffff)*(s1&0xffffff))>>32)&0xffffffff,
    VOP2Op.V_MIN_I32: lambda: s0 if sext(s0,32)<sext(s1,32) else s1, VOP2Op.V_MAX_I32: lambda: s0 if sext(s0,32)>sext(s1,32) else s1,
    VOP2Op.V_MIN_U32: lambda: min(s0,s1), VOP2Op.V_MAX_U32: lambda: max(s0,s1),
    VOP2Op.V_LSHLREV_B32: lambda: (s1<<(s0&0x1f))&0xffffffff, VOP2Op.V_LSHRREV_B32: lambda: s1>>(s0&0x1f),
    VOP2Op.V_ASHRREV_I32: lambda: (sext(s1,32)>>(s0&0x1f))&0xffffffff,
    VOP2Op.V_AND_B32: lambda: s0&s1, VOP2Op.V_OR_B32: lambda: s0|s1, VOP2Op.V_XOR_B32: lambda: s0^s1, VOP2Op.V_XNOR_B32: lambda: ~(s0^s1)&0xffffffff}
  if op == VOP2Op.V_CNDMASK_B32: V[vdst] = s1 if (st.vcc >> lane) & 1 else s0
  elif op == VOP2Op.V_ADD_CO_CI_U32: cin = (st.vcc >> lane) & 1; r = s0 + s1 + cin; st.vcc = (st.vcc & ~(1 << lane)) | (int(r >= 0x100000000) << lane); V[vdst] = r & 0xffffffff
  elif op == VOP2Op.V_SUB_CO_CI_U32: bin_ = (st.vcc >> lane) & 1; r = s0 - s1 - bin_; st.vcc = (st.vcc & ~(1 << lane)) | (int(s1 + bin_ > s0) << lane); V[vdst] = r & 0xffffffff
  elif op == VOP2Op.V_SUBREV_CO_CI_U32: bin_ = (st.vcc >> lane) & 1; r = s1 - s0 - bin_; st.vcc = (st.vcc & ~(1 << lane)) | (int(s0 + bin_ > s1) << lane); V[vdst] = r & 0xffffffff
  elif op in ARITH: V[vdst] = ARITH[op]()
  else: raise NotImplementedError(f"VOP2 op {op}")

def vop3_mod(val: int, neg: int, abs_: int, idx: int) -> int:
  if (abs_ >> idx) & 1: val = i32(abs(f32(val)))
  if (neg >> idx) & 1: val = i32(-f32(val))
  return val

def exec_vop3(st: WaveState, inst: VOP3, lane: int) -> None:
  op, src0, src1, src2, vdst, neg, abs_ = inst.op, inst.src0, inst.src1, inst.src2, inst.vdst, inst.neg, getattr(inst, 'abs', 0)
  s0, s1, s2 = vop3_mod(rsrc(st, src0, lane), neg, abs_, 0), vop3_mod(rsrc(st, src1, lane), neg, abs_, 1), vop3_mod(rsrc(st, src2, lane), neg, abs_, 2)
  V = st.vgpr[lane]
  SIMPLE: dict[int, Any] = {
    VOP3Op.V_MOV_B32: lambda: s0, VOP3Op.V_ADD_F32: lambda: i32(f32(s0)+f32(s1)), VOP3Op.V_SUB_F32: lambda: i32(f32(s0)-f32(s1)),
    VOP3Op.V_SUBREV_F32: lambda: i32(f32(s1)-f32(s0)), VOP3Op.V_MUL_F32: lambda: i32(f32(s0)*f32(s1)),
    VOP3Op.V_MIN_F32: lambda: i32(min(f32(s0),f32(s1))), VOP3Op.V_MAX_F32: lambda: i32(max(f32(s0),f32(s1))),
    VOP3Op.V_FMAC_F32: lambda: i32(f32(s0)*f32(s1)+f32(V[vdst])), VOP3Op.V_FMA_F32: lambda: i32(f32(s0)*f32(s1)+f32(s2)),
    VOP3Op.V_AND_B32: lambda: s0&s1, VOP3Op.V_OR_B32: lambda: s0|s1, VOP3Op.V_XOR_B32: lambda: s0^s1,
    VOP3Op.V_LSHLREV_B32: lambda: (s1<<(s0&0x1f))&0xffffffff, VOP3Op.V_LSHRREV_B32: lambda: s1>>(s0&0x1f),
    VOP3Op.V_ASHRREV_I32: lambda: (sext(s1,32)>>(s0&0x1f))&0xffffffff,
    VOP3Op.V_ADD_NC_U32: lambda: (s0+s1)&0xffffffff, VOP3Op.V_SUB_NC_U32: lambda: (s0-s1)&0xffffffff,
    VOP3Op.V_MIN_I32: lambda: s0 if sext(s0,32)<sext(s1,32) else s1, VOP3Op.V_MAX_I32: lambda: s0 if sext(s0,32)>sext(s1,32) else s1,
    VOP3Op.V_MIN_U32: lambda: min(s0,s1), VOP3Op.V_MAX_U32: lambda: max(s0,s1),
    VOP3Op.V_ADD3_U32: lambda: (s0+s1+s2)&0xffffffff, VOP3Op.V_LSHL_ADD_U32: lambda: ((s0<<(s1&0x1f))+s2)&0xffffffff,
    VOP3Op.V_ADD_LSHL_U32: lambda: ((s0+s1)<<(s2&0x1f))&0xffffffff,
    VOP3Op.V_MUL_LO_U32: lambda: (s0*s1)&0xffffffff, VOP3Op.V_MUL_HI_U32: lambda: ((s0*s1)>>32)&0xffffffff,
    VOP3Op.V_MUL_HI_I32: lambda: ((sext(s0,32)*sext(s1,32))>>32)&0xffffffff,
    VOP3Op.V_MAD_U32_U24: lambda: ((s0&0xffffff)*(s1&0xffffff)+s2)&0xffffffff,
    VOP3Op.V_MAD_I32_I24: lambda: (sext(s0&0xffffff,24)*sext(s1&0xffffff,24)+sext(s2,32))&0xffffffff,
    VOP3Op.V_RCP_F32: lambda: i32(1.0/f32(s0) if f32(s0)!=0 else math.copysign(float('inf'),f32(s0))),
    VOP3Op.V_RSQ_F32: lambda: i32(1.0/math.sqrt(f32(s0)) if f32(s0)>0 else float('inf')),
    VOP3Op.V_SQRT_F32: lambda: i32(math.sqrt(max(0,f32(s0)))), VOP3Op.V_EXP_F32: lambda: i32(math.pow(2.0,f32(s0))),
    VOP3Op.V_LOG_F32: lambda: i32(math.log2(f32(s0)) if f32(s0)>0 else float('-inf')),
    VOP3Op.V_FLOOR_F32: lambda: i32(math.floor(f32(s0))), VOP3Op.V_CEIL_F32: lambda: i32(math.ceil(f32(s0))),
    VOP3Op.V_TRUNC_F32: lambda: i32(math.trunc(f32(s0))),
    VOP3Op.V_CVT_F32_I32: lambda: i32(float(sext(s0,32))), VOP3Op.V_CVT_F32_U32: lambda: i32(float(s0)),
    VOP3Op.V_CVT_I32_F32: lambda: max(-0x80000000,min(0x7fffffff,int(f32(s0))))&0xffffffff,
    VOP3Op.V_CVT_U32_F32: lambda: max(0,min(0xffffffff,int(f32(s0)))),
    VOP3Op.V_LDEXP_F32: lambda: i32(math.ldexp(f32(s0),sext(s1,32))),
    VOP3Op.V_FREXP_MANT_F32: lambda: i32(math.frexp(f32(s0))[0] if f32(s0)!=0 else 0.0),
    VOP3Op.V_FREXP_EXP_I32_F32: lambda: (math.frexp(f32(s0))[1] if f32(s0)!=0 else 0)&0xffffffff,
    VOP3Op.V_ALIGNBIT_B32: lambda: (((s0<<32)|s1)>>(s2&0x1f))&0xffffffff, VOP3Op.V_XAD_U32: lambda: ((s0*s1)+s2)&0xffffffff}
  if op == VOP3Op.V_BFE_U32: off, wd = s1 & 0x1f, s2 & 0x1f; V[vdst] = (s0 >> off) & ((1 << wd) - 1) if wd else 0
  elif op == VOP3Op.V_BFE_I32: off, wd = s1 & 0x1f, s2 & 0x1f; V[vdst] = sext((s0 >> off) & ((1 << wd) - 1), wd) & 0xffffffff if wd else 0
  elif op == VOP3Op.V_CNDMASK_B32: mask = rsrc(st, src2, lane) if src2 < 256 else st.vcc; V[vdst] = s1 if (mask >> lane) & 1 else s0
  elif op == VOP3Op.V_LSHLREV_B64: r = (rsrc64(st, src1, lane) << (s0 & 0x3f)) & 0xffffffffffffffff; V[vdst] = r & 0xffffffff; V[vdst+1] = (r >> 32) & 0xffffffff
  elif op == VOP3Op.V_DIV_FMAS_F32: vcc_bit = (st.vcc >> lane) & 1; V[vdst] = i32(f32(s0) * f32(s1) * (2.0**32 if vcc_bit else 1.0) + f32(s2))
  elif op == VOP3Op.V_DIV_FIXUP_F32:
    f0, f1, f2 = f32(s0), f32(s1), f32(s2)
    V[vdst] = i32(float('nan') if f1 == 0.0 and f2 == 0.0 or any(math.isnan(x) for x in (f0,f1,f2)) or (math.isinf(f2) and math.isinf(f1)) else math.copysign(float('inf'), f2) if f1 == 0.0 else f0)
  elif 0 <= op <= 255: exec_vopc_vop3(st, op, s0, s1, vdst, lane)
  elif op in SIMPLE: V[vdst] = SIMPLE[op]()
  else: raise NotImplementedError(f"VOP3 op {op}")

def exec_vopc_vop3(st: WaveState, op: int, s0: int, s1: int, sdst: int, lane: int) -> None:
  is_cmpx, base = op >= 128, op - 128 if op >= 128 else op
  result = False
  if 16 <= base <= 31:
    f0, f1, cmp = f32(s0), f32(s1), base - 16
    nan = math.isnan(f0) or math.isnan(f1)
    result = [False, f0<f1, f0==f1, f0<=f1, f0>f1, f0!=f1, f0>=f1, not nan, nan, f0<f1 or nan, f0==f1 or nan, f0<=f1 or nan, f0>f1 or nan, f0!=f1 or nan, f0>=f1 or nan, True][cmp]
  elif 64 <= base <= 79:
    cmp = (base - 64) % 8; s0s, s1s = sext(s0,32), sext(s1,32)
    result = [False, s0s<s1s, s0s==s1s, s0s<=s1s, s0s>s1s, s0s!=s1s, s0s>=s1s, True][cmp] if base < 72 else [False, s0<s1, s0==s1, s0<=s1, s0>s1, s0!=s1, s0>=s1, True][cmp]
  if sdst == VCC_LO: st.vcc = (st.vcc & ~(1 << lane)) | (int(result) << lane)
  else: wsgpr(st, sdst, (rsgpr(st, sdst) & ~(1 << lane)) | (int(result) << lane))
  if is_cmpx: st.exec_mask = (st.exec_mask & ~(1 << lane)) | (int(result) << lane)

def exec_vopc(st: WaveState, inst: VOPC, lane: int) -> None:
  op, s0, s1 = inst.op, rsrc(st, inst.src0, lane), st.vgpr[lane][inst.vsrc1]
  is_cmpx, base = op >= 128, op - 128 if op >= 128 else op
  result = False
  if 16 <= base <= 31:
    f0, f1, cmp = f32(s0), f32(s1), base - 16
    nan = math.isnan(f0) or math.isnan(f1)
    result = [False, f0<f1, f0==f1, f0<=f1, f0>f1, f0!=f1, f0>=f1, not nan, nan, f0<f1 or nan, f0==f1 or nan, f0<=f1 or nan, f0>f1 or nan, f0!=f1 or nan, f0>=f1 or nan, True][cmp]
  elif 64 <= base <= 79:
    cmp = (base - 64) % 8; s0s, s1s = sext(s0,32), sext(s1,32)
    result = [False, s0s<s1s, s0s==s1s, s0s<=s1s, s0s>s1s, s0s!=s1s, s0s>=s1s, True][cmp] if base < 72 else [False, s0<s1, s0==s1, s0<=s1, s0>s1, s0!=s1, s0>=s1, True][cmp]
  st.vcc = (st.vcc & ~(1 << lane)) | (int(result) << lane)
  if is_cmpx: st.exec_mask = (st.exec_mask & ~(1 << lane)) | (int(result) << lane)

def exec_vop3sd(st: WaveState, inst: VOP3SD, lane: int) -> None:
  op, src0, src1, src2, vdst, sdst, neg = inst.op, inst.src0, inst.src1, inst.src2, inst.vdst, inst.sdst, inst.neg
  s0, s1, s2 = rsrc(st, src0, lane), rsrc(st, src1, lane), rsrc(st, src2, lane)
  if (neg >> 0) & 1: s0 = i32(-f32(s0))
  if (neg >> 1) & 1: s1 = i32(-f32(s1))
  if (neg >> 2) & 1: s2 = i32(-f32(s2))
  V = st.vgpr[lane]
  if op == VOP3SDOp.V_ADD_CO_U32: r = s0 + s1; V[vdst] = r & 0xffffffff; wsgpr(st, sdst, (rsgpr(st, sdst) & ~(1 << lane)) | (int(r >= 0x100000000) << lane))
  elif op == VOP3SDOp.V_SUB_CO_U32: V[vdst] = (s0 - s1) & 0xffffffff; wsgpr(st, sdst, (rsgpr(st, sdst) & ~(1 << lane)) | (int(s1 > s0) << lane))
  elif op == VOP3SDOp.V_SUBREV_CO_U32: V[vdst] = (s1 - s0) & 0xffffffff; wsgpr(st, sdst, (rsgpr(st, sdst) & ~(1 << lane)) | (int(s0 > s1) << lane))
  elif op == VOP3SDOp.V_ADD_CO_CI_U32: cin = (rsgpr(st, src2) >> lane) & 1 if src2 < 256 else (st.vcc >> lane) & 1; r = s0 + s1 + cin; V[vdst] = r & 0xffffffff; wsgpr(st, sdst, (rsgpr(st, sdst) & ~(1 << lane)) | (int(r >= 0x100000000) << lane))
  elif op == VOP3SDOp.V_MAD_U64_U32: s2_64 = s2 | (rsrc(st, src2 + 1, lane) << 32); r = s0 * s1 + s2_64; V[vdst] = r & 0xffffffff; V[vdst+1] = (r >> 32) & 0xffffffff
  elif op == VOP3SDOp.V_MAD_I64_I32: s2_64 = sext(s2 | (rsrc(st, src2 + 1, lane) << 32), 64); r = (sext(s0,32) * sext(s1,32) + s2_64) & 0xffffffffffffffff; V[vdst] = r & 0xffffffff; V[vdst+1] = (r >> 32) & 0xffffffff
  elif op == VOP3SDOp.V_DIV_SCALE_F32: V[vdst] = s0; st.vcc = (st.vcc & ~(1 << lane)) | ((1 if s0 == s2 else 0) << lane)
  elif op == VOP3SDOp.V_DIV_SCALE_F64: V[vdst] = s0; V[vdst+1] = rsrc(st, src0 + 1, lane); st.vcc = (st.vcc & ~(1 << lane)) | ((1 if s0 == s2 else 0) << lane)
  else: raise NotImplementedError(f"VOP3SD op {op}")

def exec_flat(st: WaveState, inst: FLAT, lane: int) -> None:
  op, addr_reg, data_reg, vdst, offset, saddr = inst.op, inst.addr, inst.data, inst.vdst, sext(inst.offset, 13), inst.saddr
  V = st.vgpr[lane]
  addr = V[addr_reg] | (V[addr_reg+1] << 32)
  addr = (rsgpr64(st, saddr) + V[addr_reg] + offset) & 0xffffffffffffffff if saddr not in (NULL_REG, 0x7f) else (addr + offset) & 0xffffffffffffffff
  LOAD: dict[int, tuple[int, int] | tuple[int, int, str]] = {
    GLOBALOp.GLOBAL_LOAD_B32: (1,4), FLATOp.FLAT_LOAD_B32: (1,4), GLOBALOp.GLOBAL_LOAD_B64: (2,4), FLATOp.FLAT_LOAD_B64: (2,4),
    GLOBALOp.GLOBAL_LOAD_B96: (3,4), FLATOp.FLAT_LOAD_B96: (3,4), GLOBALOp.GLOBAL_LOAD_B128: (4,4), FLATOp.FLAT_LOAD_B128: (4,4),
    GLOBALOp.GLOBAL_LOAD_U8: (1,1,'u'), FLATOp.FLAT_LOAD_U8: (1,1,'u'), GLOBALOp.GLOBAL_LOAD_I8: (1,1,'i'), FLATOp.FLAT_LOAD_I8: (1,1,'i'),
    GLOBALOp.GLOBAL_LOAD_U16: (1,2,'u'), FLATOp.FLAT_LOAD_U16: (1,2,'u'), GLOBALOp.GLOBAL_LOAD_I16: (1,2,'i'), FLATOp.FLAT_LOAD_I16: (1,2,'i')}
  STORE: dict[int, tuple[int, int]] = {
    GLOBALOp.GLOBAL_STORE_B32: (1,4), FLATOp.FLAT_STORE_B32: (1,4), GLOBALOp.GLOBAL_STORE_B64: (2,4), FLATOp.FLAT_STORE_B64: (2,4),
    GLOBALOp.GLOBAL_STORE_B96: (3,4), FLATOp.FLAT_STORE_B96: (3,4), GLOBALOp.GLOBAL_STORE_B128: (4,4), FLATOp.FLAT_STORE_B128: (4,4),
    GLOBALOp.GLOBAL_STORE_B8: (1,1), FLATOp.FLAT_STORE_B8: (1,1), GLOBALOp.GLOBAL_STORE_B16: (1,2), FLATOp.FLAT_STORE_B16: (1,2)}
  CTYPES: dict[int, type[ctypes.c_uint8] | type[ctypes.c_uint16] | type[ctypes.c_uint32]] = {1: ctypes.c_uint8, 2: ctypes.c_uint16, 4: ctypes.c_uint32}
  if op in LOAD:
    info = LOAD[op]; cnt, sz = info[0], info[1]; sign = info[2] if len(info) > 2 else None
    for i in range(cnt):
      val = CTYPES[sz].from_address(addr + i * sz).value
      V[vdst + i] = sext(val, sz * 8) & 0xffffffff if sign == 'i' else val
  elif op in STORE:
    cnt, sz = STORE[op]
    for i in range(cnt): CTYPES[sz].from_address(addr + i * sz).value = V[data_reg + i] & ((1 << (sz * 8)) - 1)
  else: raise NotImplementedError(f"FLAT op {op}")

def exec_ds(st: WaveState, inst: DS, lane: int, lds: bytearray) -> None:
  op, addr, vdst, V = inst.op, (st.vgpr[lane][inst.addr] + inst.offset0) & 0xffff, inst.vdst, st.vgpr[lane]
  LOAD: dict[int, tuple[int, int] | tuple[int, int, str]] = {
    DSOp.DS_LOAD_B32: (1,4), DSOp.DS_LOAD_B64: (2,4), DSOp.DS_LOAD_B128: (4,4),
    DSOp.DS_LOAD_U8: (1,1,'u'), DSOp.DS_LOAD_I8: (1,1,'i'), DSOp.DS_LOAD_U16: (1,2,'u'), DSOp.DS_LOAD_I16: (1,2,'i')}
  STORE: dict[int, tuple[int, int]] = {DSOp.DS_STORE_B32: (1,4), DSOp.DS_STORE_B64: (2,4), DSOp.DS_STORE_B128: (4,4), DSOp.DS_STORE_B8: (1,1), DSOp.DS_STORE_B16: (1,2)}
  if op in LOAD:
    info = LOAD[op]; cnt, sz = info[0], info[1]; sign = info[2] if len(info) > 2 else None
    for i in range(cnt):
      val = int.from_bytes(lds[addr+i*sz:addr+i*sz+sz], 'little')
      V[vdst + i] = sext(val, sz * 8) & 0xffffffff if sign == 'i' else val
  elif op in STORE:
    cnt, sz = STORE[op]
    for i in range(cnt): lds[addr+i*sz:addr+i*sz+sz] = (V[inst.data0 + i] & ((1 << (sz * 8)) - 1)).to_bytes(sz, 'little')
  else: raise NotImplementedError(f"DS op {op}")

def exec_vopd_op(st: WaveState, op: int, src0: int, src1: int, dst: int, lane: int) -> None:
  s0, s1, V = rsrc(st, src0, lane), st.vgpr[lane][src1], st.vgpr[lane]
  OPS: dict[int, Any] = {
    VOPDOp.V_DUAL_FMAC_F32: lambda: i32(f32(s0)*f32(s1)+f32(V[dst])), VOPDOp.V_DUAL_MUL_F32: lambda: i32(f32(s0)*f32(s1)),
    VOPDOp.V_DUAL_ADD_F32: lambda: i32(f32(s0)+f32(s1)), VOPDOp.V_DUAL_SUB_F32: lambda: i32(f32(s0)-f32(s1)),
    VOPDOp.V_DUAL_SUBREV_F32: lambda: i32(f32(s1)-f32(s0)), VOPDOp.V_DUAL_MUL_DX9_ZERO_F32: lambda: i32(0.0 if f32(s0)==0.0 or f32(s1)==0.0 else f32(s0)*f32(s1)),
    VOPDOp.V_DUAL_MOV_B32: lambda: s0, VOPDOp.V_DUAL_CNDMASK_B32: lambda: s1 if (st.vcc >> lane) & 1 else s0,
    VOPDOp.V_DUAL_MAX_F32: lambda: i32(max(f32(s0), f32(s1))), VOPDOp.V_DUAL_MIN_F32: lambda: i32(min(f32(s0), f32(s1))),
    VOPDOp.V_DUAL_ADD_NC_U32: lambda: (s0 + s1) & 0xffffffff, VOPDOp.V_DUAL_LSHLREV_B32: lambda: (s1 << (s0 & 0x1f)) & 0xffffffff,
    VOPDOp.V_DUAL_AND_B32: lambda: s0 & s1}
  if op in OPS: V[dst] = OPS[op]()
  else: raise NotImplementedError(f"VOPD op {op}")

def exec_vopd(st: WaveState, inst: VOPD, lane: int) -> None:
  exec_vopd_op(st, inst.opx, inst.srcx0, inst.vsrcx1, inst.vdstx, lane)
  exec_vopd_op(st, inst.opy, inst.srcy0, inst.vsrcy1, (inst.vdsty << 1) | ((inst.vdstx & 1) ^ 1), lane)

def exec_wave(program: Program, st: WaveState, lds: bytearray, n_lanes: int) -> int:
  SCALAR: dict[type, Any] = {SOP1: exec_sop1, SOP2: exec_sop2, SOPC: exec_sopc, SOPK: exec_sopk, SOPP: exec_sopp, SMEM: exec_smem}
  VECTOR: dict[type, Any] = {VOP1: exec_vop1, VOP2: exec_vop2, VOP3: exec_vop3, VOP3SD: exec_vop3sd, VOPC: exec_vopc,
                              FLAT: exec_flat, DS: lambda s,i,l: exec_ds(s,i,l,lds), VOPD: exec_vopd}
  while st.pc < len(program):
    inst, _ = program[st.pc]
    st.literal = inst._literal or 0
    if type(inst) in SCALAR:
      delta = SCALAR[type(inst)](st, inst)
      if delta == -1: return 0
      if delta == -2: st.pc += 1; return -2
      st.pc += 1 + delta
    else:
      for lane in range(n_lanes):
        if st.exec_mask & (1 << lane): VECTOR[type(inst)](st, inst, lane)
      st.pc += 1
  return 0

def exec_workgroup(program: Program, workgroup_id: tuple[int, int, int], local_size: tuple[int, int, int], args_ptr: int, dispatch_dim: int) -> None:
  lx, ly, lz = local_size
  total_threads, lds = lx * ly * lz, bytearray(65536)
  waves: list[tuple[WaveState, int]] = []
  for wave_start in range(0, total_threads, WAVE_SIZE):
    n_lanes = min(WAVE_SIZE, total_threads - wave_start)
    st = WaveState()
    st.exec_mask = (1 << n_lanes) - 1
    wsgpr64(st, 0, args_ptr)
    gx, gy, gz = workgroup_id
    if dispatch_dim >= 3: st.sgpr[13], st.sgpr[14], st.sgpr[15] = gx, gy, gz
    elif dispatch_dim == 2: st.sgpr[14], st.sgpr[15] = gx, gy
    else: st.sgpr[15] = gx
    for i in range(n_lanes):
      tid = wave_start + i
      st.vgpr[i][0] = tid if local_size == (lx, 1, 1) else ((tid // (lx * ly)) << 20) | (((tid // lx) % ly) << 10) | (tid % lx)
    waves.append((st, n_lanes))
  has_barrier = any(isinstance(inst, SOPP) and inst.op == SOPPOp.S_BARRIER for inst, _ in program)
  for _ in range(2 if has_barrier else 1):
    for st, n_lanes in waves: exec_wave(program, st, lds, n_lanes)

def run_asm(lib: int, lib_sz: int, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, args_ptr: int) -> int:
  data = (ctypes.c_char * lib_sz).from_address(lib).raw
  program = decode_program(data)
  if not program: return -1
  dispatch_dim = 3 if gz > 1 else (2 if gy > 1 else 1)
  for gidz in range(gz):
    for gidy in range(gy):
      for gidx in range(gx): exec_workgroup(program, (gidx, gidy, gidz), (lx, ly, lz), args_ptr, dispatch_dim)
  return 0
