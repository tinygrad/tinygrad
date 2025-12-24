# RDNA3 emulator - operates on parsed Inst classes from lib.py
from __future__ import annotations
import ctypes, struct, math
from dataclasses import dataclass, field
from extra.assembly.rdna3.lib import Inst, Inst32, Inst64, RawImm, FLOAT_ENC, bits
from extra.assembly.rdna3.autogen import (
  SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, VOP1, VOP2, VOP3, VOP3SD, DS, FLAT, VOPD,
  SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, SMEMOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOPCOp, DSOp, FLATOp, GLOBALOp,
  SrcEnum
)

# VOPC instruction class (not in autogen, define here)
class VOPC(Inst32):
  encoding = bits[31:25] == 0b0111110
  op = bits[24:17]
  src0 = bits[8:0]
  vsrc1 = bits[16:9]

# constants
WAVE_SIZE = 32
SGPR_COUNT = 128
VGPR_COUNT = 256

# special register indices
VCC_LO, VCC_HI = 106, 107
EXEC_LO, EXEC_HI = 126, 127
NULL_REG = 124
M0 = 125

# float constants (value -> bits)
FLOAT_BITS = {240: 0x3f000000, 241: 0xbf000000, 242: 0x3f800000, 243: 0xbf800000,
              244: 0x40000000, 245: 0xc0000000, 246: 0x40800000, 247: 0xc0800000, 248: 0x3e22f983}

END_PRG = 0xbfb00000
S_BARRIER = 0xbfbd0000

# *** float helpers ***
def f32_to_bits(f: float) -> int: return struct.unpack('<I', struct.pack('<f', f))[0]
def bits_to_f32(i: int) -> float: return struct.unpack('<f', struct.pack('<I', i & 0xffffffff))[0]
def f64_to_bits(f: float) -> int: return struct.unpack('<Q', struct.pack('<d', f))[0]
def bits_to_f64(i: int) -> float: return struct.unpack('<d', struct.pack('<Q', i & 0xffffffffffffffff))[0]
def f16_to_bits(f: float) -> int: return struct.unpack('<H', struct.pack('<e', f))[0]
def bits_to_f16(i: int) -> float: return struct.unpack('<e', struct.pack('<H', i & 0xffff))[0]

def sign_ext(val: int, bits: int) -> int:
  if val & (1 << (bits - 1)): return val - (1 << bits)
  return val

def clz_i32_u32(x: int) -> int:
  if x == 0: return 32
  n = 0
  if (x & 0xffff0000) == 0: n += 16; x <<= 16
  if (x & 0xff000000) == 0: n += 8; x <<= 8
  if (x & 0xf0000000) == 0: n += 4; x <<= 4
  if (x & 0xc0000000) == 0: n += 2; x <<= 2
  if (x & 0x80000000) == 0: n += 1
  return n

def cls_i32(x: int) -> int:
  x = x & 0xffffffff
  if x == 0 or x == 0xffffffff: return 31
  sign = (x >> 31) & 1
  if sign: x = ~x & 0xffffffff
  return clz_i32_u32(x) - 1

# *** state ***
@dataclass
class WaveState:
  sgpr: list[int] = field(default_factory=lambda: [0] * SGPR_COUNT)
  vgpr: list[list[int]] = field(default_factory=lambda: [[0] * VGPR_COUNT for _ in range(WAVE_SIZE)])
  scc: int = 0
  vcc: int = 0
  exec_mask: int = 0xffffffff
  pc: int = 0
  literal: int = 0

# *** instruction decoding ***
def get_field(inst: Inst, name: str) -> int:
  val = inst._values.get(name, 0)
  return val.val if isinstance(val, RawImm) else (val.value if hasattr(val, 'value') else val)

def decode_program(data: bytes) -> list[tuple[Inst, int]]:
  """Decode bytes into list of (Inst, literal) pairs."""
  result = []
  i = 0
  while i < len(data):
    word = int.from_bytes(data[i:i+4], 'little')
    inst_class, is_64 = decode_format(word)
    if inst_class is None:
      i += 4
      continue
    size = 8 if is_64 else 4
    inst = inst_class.from_bytes(data[i:i+size])
    # check for literal
    literal = None
    for field in ['src0', 'src1', 'src2', 'ssrc0', 'ssrc1']:
      v = get_field(inst, field)
      if v == 255:  # literal
        literal = int.from_bytes(data[i+size:i+size+4], 'little')
        size += 4
        break
    inst._literal = literal
    result.append((inst, size // 4))  # size in dwords
    i += size
  return result

def decode_format(word: int) -> tuple[type | None, bool]:
  """Determine instruction format from first word."""
  hi2 = (word >> 30) & 0x3
  if hi2 == 0b11:
    enc = (word >> 26) & 0xf
    if enc == 0b1101: return SMEM, True
    if enc == 0b0101:
      # VOP3 and VOP3SD share encoding, differentiate by opcode
      op = (word >> 16) & 0x3ff
      if op in (288, 289, 290, 764, 765, 766, 767, 768, 769, 770): return VOP3SD, True
      return VOP3, True
    if enc == 0b0011: return None, True  # VOP3P - skip for now
    if enc == 0b0110: return DS, True
    if enc == 0b0111: return FLAT, True
    if enc == 0b0010: return VOPD, True
    return None, True
  if hi2 == 0b10:
    enc = (word >> 23) & 0x7f
    if enc == 0b1111101: return SOP1, False
    if enc == 0b1111110: return SOPC, False
    if enc == 0b1111111: return SOPP, False
    if ((word >> 28) & 0xf) == 0b1011: return SOPK, False
    return SOP2, False
  # hi2 == 0b00 or 0b01
  enc = (word >> 25) & 0x7f
  if enc == 0b0111110: return VOPC, False
  if enc == 0b0111111: return VOP1, False
  return VOP2, False

# *** operand read/write ***
def read_sgpr(state: WaveState, idx: int) -> int:
  if idx == VCC_LO: return state.vcc & 0xffffffff
  if idx == VCC_HI: return (state.vcc >> 32) & 0xffffffff
  if idx == EXEC_LO: return state.exec_mask & 0xffffffff
  if idx == EXEC_HI: return (state.exec_mask >> 32) & 0xffffffff
  if idx == NULL_REG: return 0
  if idx == 253: return state.scc  # SCC
  return state.sgpr[idx] if idx < SGPR_COUNT else 0

def write_sgpr(state: WaveState, idx: int, val: int):
  val = val & 0xffffffff
  if idx == VCC_LO: state.vcc = (state.vcc & 0xffffffff00000000) | val
  elif idx == VCC_HI: state.vcc = (state.vcc & 0xffffffff) | (val << 32)
  elif idx == EXEC_LO: state.exec_mask = (state.exec_mask & 0xffffffff00000000) | val
  elif idx == EXEC_HI: state.exec_mask = (state.exec_mask & 0xffffffff) | (val << 32)
  elif idx < SGPR_COUNT: state.sgpr[idx] = val

def read_sgpr64(state: WaveState, idx: int) -> int:
  lo = read_sgpr(state, idx)
  hi = read_sgpr(state, idx + 1)
  return lo | (hi << 32)

def write_sgpr64(state: WaveState, idx: int, val: int):
  write_sgpr(state, idx, val & 0xffffffff)
  write_sgpr(state, idx + 1, (val >> 32) & 0xffffffff)

def read_src(state: WaveState, val: int, lane: int) -> int:
  """Read source operand value."""
  if val <= 105: return state.sgpr[val]
  if val == VCC_LO: return state.vcc & 0xffffffff
  if val == VCC_HI: return (state.vcc >> 32) & 0xffffffff
  if 108 <= val <= 123: return state.sgpr[val]  # TTMP
  if val == NULL_REG: return 0
  if val == M0: return state.sgpr[M0]
  if val == EXEC_LO: return state.exec_mask & 0xffffffff
  if val == EXEC_HI: return (state.exec_mask >> 32) & 0xffffffff
  if 128 <= val <= 192: return val - 128
  if 193 <= val <= 208: return (-(val - 192)) & 0xffffffff
  if val in FLOAT_BITS: return FLOAT_BITS[val]
  if val == 255: return state.literal
  if 256 <= val <= 511: return state.vgpr[lane][val - 256]
  return 0

def read_src64(state: WaveState, val: int, lane: int) -> int:
  lo = read_src(state, val, lane)
  hi = read_src(state, val + 1, lane) if val <= 105 or 256 <= val <= 511 else 0
  return lo | (hi << 32)

# *** instruction execution ***
def exec_sop1(state: WaveState, inst: SOP1) -> int:
  op = get_field(inst, 'op')
  ssrc0 = get_field(inst, 'ssrc0')
  sdst = get_field(inst, 'sdst')
  s0 = read_src(state, ssrc0, 0)

  if op == SOP1Op.S_MOV_B32:
    write_sgpr(state, sdst, s0)
  elif op == SOP1Op.S_MOV_B64:
    s0_64 = read_src64(state, ssrc0, 0)
    write_sgpr64(state, sdst, s0_64)
  elif op == SOP1Op.S_NOT_B32:
    result = (~s0) & 0xffffffff
    state.scc = int(result != 0)
    write_sgpr(state, sdst, result)
  elif op == SOP1Op.S_NOT_B64:
    s0_64 = read_src64(state, ssrc0, 0)
    result = (~s0_64) & 0xffffffffffffffff
    state.scc = int(result != 0)
    write_sgpr64(state, sdst, result)
  elif op == SOP1Op.S_BREV_B32:
    write_sgpr(state, sdst, int(f'{s0:032b}'[::-1], 2))
  elif op == SOP1Op.S_CLZ_I32_U32:
    write_sgpr(state, sdst, clz_i32_u32(s0))
  elif op == SOP1Op.S_CLS_I32:
    write_sgpr(state, sdst, cls_i32(s0))
  elif op == SOP1Op.S_SEXT_I32_I8:
    write_sgpr(state, sdst, sign_ext(s0 & 0xff, 8) & 0xffffffff)
  elif op == SOP1Op.S_SEXT_I32_I16:
    write_sgpr(state, sdst, sign_ext(s0 & 0xffff, 16) & 0xffffffff)
  elif op == SOP1Op.S_BITSET0_B32:
    old = read_sgpr(state, sdst)
    write_sgpr(state, sdst, old & ~(1 << (s0 & 0x1f)))
  elif op == SOP1Op.S_BITSET1_B32:
    old = read_sgpr(state, sdst)
    write_sgpr(state, sdst, old | (1 << (s0 & 0x1f)))
  elif op == SOP1Op.S_ABS_I32:
    s0_signed = sign_ext(s0, 32)
    result = abs(s0_signed) & 0xffffffff
    state.scc = int(result != 0)
    write_sgpr(state, sdst, result)
  elif op == SOP1Op.S_AND_SAVEEXEC_B32:
    old_exec = state.exec_mask & 0xffffffff
    state.exec_mask = s0 & old_exec
    state.scc = int(state.exec_mask != 0)
    write_sgpr(state, sdst, old_exec)
  elif op == SOP1Op.S_OR_SAVEEXEC_B32:
    old_exec = state.exec_mask & 0xffffffff
    state.exec_mask = s0 | old_exec
    state.scc = int(state.exec_mask != 0)
    write_sgpr(state, sdst, old_exec)
  elif op == SOP1Op.S_AND_NOT1_SAVEEXEC_B32:
    old_exec = state.exec_mask & 0xffffffff
    state.exec_mask = s0 & (~old_exec & 0xffffffff)
    state.scc = int(state.exec_mask != 0)
    write_sgpr(state, sdst, old_exec)
  else:
    raise NotImplementedError(f"SOP1 op {op} ({SOP1Op(op).name})")
  return 0

def exec_sop2(state: WaveState, inst: SOP2) -> int:
  op = get_field(inst, 'op')
  ssrc0, ssrc1 = get_field(inst, 'ssrc0'), get_field(inst, 'ssrc1')
  sdst = get_field(inst, 'sdst')
  s0, s1 = read_src(state, ssrc0, 0), read_src(state, ssrc1, 0)

  if op == SOP2Op.S_ADD_U32:
    result = s0 + s1
    state.scc = int(result >= 0x100000000)
    write_sgpr(state, sdst, result & 0xffffffff)
  elif op == SOP2Op.S_SUB_U32:
    result = s0 - s1
    state.scc = int(s1 > s0)
    write_sgpr(state, sdst, result & 0xffffffff)
  elif op == SOP2Op.S_ADD_I32:
    s0_s, s1_s = sign_ext(s0, 32), sign_ext(s1, 32)
    result = s0_s + s1_s
    overflow = ((s0 >> 31) == (s1 >> 31)) and ((s0 >> 31) != ((result >> 31) & 1))
    state.scc = int(overflow)
    write_sgpr(state, sdst, result & 0xffffffff)
  elif op == SOP2Op.S_SUB_I32:
    s0_s, s1_s = sign_ext(s0, 32), sign_ext(s1, 32)
    result = s0_s - s1_s
    overflow = ((s0 >> 31) != (s1 >> 31)) and ((s0 >> 31) != ((result >> 31) & 1))
    state.scc = int(overflow)
    write_sgpr(state, sdst, result & 0xffffffff)
  elif op == SOP2Op.S_ADDC_U32:
    result = s0 + s1 + state.scc
    state.scc = int(result >= 0x100000000)
    write_sgpr(state, sdst, result & 0xffffffff)
  elif op == SOP2Op.S_SUBB_U32:
    result = s0 - s1 - state.scc
    state.scc = int((s1 + state.scc) > s0)
    write_sgpr(state, sdst, result & 0xffffffff)
  elif op == SOP2Op.S_LSHL_B32:
    result = (s0 << (s1 & 0x1f)) & 0xffffffff
    state.scc = int(result != 0)
    write_sgpr(state, sdst, result)
  elif op == SOP2Op.S_LSHL_B64:
    s0_64 = read_src64(state, ssrc0, 0)
    result = (s0_64 << (s1 & 0x3f)) & 0xffffffffffffffff
    state.scc = int(result != 0)
    write_sgpr64(state, sdst, result)
  elif op == SOP2Op.S_LSHR_B32:
    result = s0 >> (s1 & 0x1f)
    state.scc = int(result != 0)
    write_sgpr(state, sdst, result)
  elif op == SOP2Op.S_LSHR_B64:
    s0_64 = read_src64(state, ssrc0, 0)
    result = s0_64 >> (s1 & 0x3f)
    state.scc = int(result != 0)
    write_sgpr64(state, sdst, result)
  elif op == SOP2Op.S_ASHR_I32:
    result = sign_ext(s0, 32) >> (s1 & 0x1f)
    state.scc = int(result != 0)
    write_sgpr(state, sdst, result & 0xffffffff)
  elif op == SOP2Op.S_ASHR_I64:
    s0_64 = read_src64(state, ssrc0, 0)
    result = sign_ext(s0_64, 64) >> (s1 & 0x3f)
    state.scc = int(result != 0)
    write_sgpr64(state, sdst, result & 0xffffffffffffffff)
  elif op == SOP2Op.S_AND_B32:
    result = s0 & s1
    state.scc = int(result != 0)
    write_sgpr(state, sdst, result)
  elif op == SOP2Op.S_AND_B64:
    s0_64, s1_64 = read_src64(state, ssrc0, 0), read_src64(state, ssrc1, 0)
    result = s0_64 & s1_64
    state.scc = int(result != 0)
    write_sgpr64(state, sdst, result)
  elif op == SOP2Op.S_OR_B32:
    result = s0 | s1
    state.scc = int(result != 0)
    write_sgpr(state, sdst, result)
  elif op == SOP2Op.S_OR_B64:
    s0_64, s1_64 = read_src64(state, ssrc0, 0), read_src64(state, ssrc1, 0)
    result = s0_64 | s1_64
    state.scc = int(result != 0)
    write_sgpr64(state, sdst, result)
  elif op == SOP2Op.S_XOR_B32:
    result = s0 ^ s1
    state.scc = int(result != 0)
    write_sgpr(state, sdst, result)
  elif op == SOP2Op.S_XOR_B64:
    s0_64, s1_64 = read_src64(state, ssrc0, 0), read_src64(state, ssrc1, 0)
    result = s0_64 ^ s1_64
    state.scc = int(result != 0)
    write_sgpr64(state, sdst, result)
  elif op == SOP2Op.S_AND_NOT1_B32:
    result = s0 & (~s1 & 0xffffffff)
    state.scc = int(result != 0)
    write_sgpr(state, sdst, result)
  elif op == SOP2Op.S_OR_NOT1_B32:
    result = s0 | (~s1 & 0xffffffff)
    state.scc = int(result != 0)
    write_sgpr(state, sdst, result)
  elif op == SOP2Op.S_MIN_I32:
    s0_s, s1_s = sign_ext(s0, 32), sign_ext(s1, 32)
    result = s0 if s0_s < s1_s else s1
    state.scc = int(s0_s < s1_s)
    write_sgpr(state, sdst, result)
  elif op == SOP2Op.S_MIN_U32:
    state.scc = int(s0 < s1)
    write_sgpr(state, sdst, min(s0, s1))
  elif op == SOP2Op.S_MAX_I32:
    s0_s, s1_s = sign_ext(s0, 32), sign_ext(s1, 32)
    result = s0 if s0_s > s1_s else s1
    state.scc = int(s0_s > s1_s)
    write_sgpr(state, sdst, result)
  elif op == SOP2Op.S_MAX_U32:
    state.scc = int(s0 > s1)
    write_sgpr(state, sdst, max(s0, s1))
  elif op == SOP2Op.S_MUL_I32:
    result = (sign_ext(s0, 32) * sign_ext(s1, 32)) & 0xffffffff
    write_sgpr(state, sdst, result)
  elif op == SOP2Op.S_MUL_HI_U32:
    result = (s0 * s1) >> 32
    write_sgpr(state, sdst, result & 0xffffffff)
  elif op == SOP2Op.S_MUL_HI_I32:
    result = (sign_ext(s0, 32) * sign_ext(s1, 32)) >> 32
    write_sgpr(state, sdst, result & 0xffffffff)
  elif op == SOP2Op.S_CSELECT_B32:
    write_sgpr(state, sdst, s0 if state.scc else s1)
  elif op == SOP2Op.S_CSELECT_B64:
    s0_64, s1_64 = read_src64(state, ssrc0, 0), read_src64(state, ssrc1, 0)
    write_sgpr64(state, sdst, s0_64 if state.scc else s1_64)
  elif op == SOP2Op.S_BFE_U32:
    offset = s1 & 0x1f
    width = (s1 >> 16) & 0x7f
    result = (s0 >> offset) & ((1 << width) - 1) if width else 0
    state.scc = int(result != 0)
    write_sgpr(state, sdst, result)
  elif op == SOP2Op.S_BFE_I32:
    offset = s1 & 0x1f
    width = (s1 >> 16) & 0x7f
    if width == 0:
      result = 0
    else:
      result = (s0 >> offset) & ((1 << width) - 1)
      result = sign_ext(result, width) & 0xffffffff
    state.scc = int(result != 0)
    write_sgpr(state, sdst, result)
  elif op == SOP2Op.S_PACK_LL_B32_B16:
    write_sgpr(state, sdst, (s0 & 0xffff) | ((s1 & 0xffff) << 16))
  elif op == SOP2Op.S_PACK_LH_B32_B16:
    write_sgpr(state, sdst, (s0 & 0xffff) | (s1 & 0xffff0000))
  elif op == SOP2Op.S_PACK_HH_B32_B16:
    write_sgpr(state, sdst, ((s0 >> 16) & 0xffff) | (s1 & 0xffff0000))
  elif op == SOP2Op.S_PACK_HL_B32_B16:
    write_sgpr(state, sdst, ((s0 >> 16) & 0xffff) | ((s1 & 0xffff) << 16))
  else:
    raise NotImplementedError(f"SOP2 op {op} ({SOP2Op(op).name})")
  return 0

def exec_sopc(state: WaveState, inst: SOPC) -> int:
  op = get_field(inst, 'op')
  ssrc0, ssrc1 = get_field(inst, 'ssrc0'), get_field(inst, 'ssrc1')
  s0, s1 = read_src(state, ssrc0, 0), read_src(state, ssrc1, 0)

  if op == SOPCOp.S_CMP_EQ_I32: state.scc = int(sign_ext(s0, 32) == sign_ext(s1, 32))
  elif op == SOPCOp.S_CMP_LG_I32: state.scc = int(sign_ext(s0, 32) != sign_ext(s1, 32))
  elif op == SOPCOp.S_CMP_GT_I32: state.scc = int(sign_ext(s0, 32) > sign_ext(s1, 32))
  elif op == SOPCOp.S_CMP_GE_I32: state.scc = int(sign_ext(s0, 32) >= sign_ext(s1, 32))
  elif op == SOPCOp.S_CMP_LT_I32: state.scc = int(sign_ext(s0, 32) < sign_ext(s1, 32))
  elif op == SOPCOp.S_CMP_LE_I32: state.scc = int(sign_ext(s0, 32) <= sign_ext(s1, 32))
  elif op == SOPCOp.S_CMP_EQ_U32: state.scc = int(s0 == s1)
  elif op == SOPCOp.S_CMP_LG_U32: state.scc = int(s0 != s1)
  elif op == SOPCOp.S_CMP_GT_U32: state.scc = int(s0 > s1)
  elif op == SOPCOp.S_CMP_GE_U32: state.scc = int(s0 >= s1)
  elif op == SOPCOp.S_CMP_LT_U32: state.scc = int(s0 < s1)
  elif op == SOPCOp.S_CMP_LE_U32: state.scc = int(s0 <= s1)
  elif op == SOPCOp.S_BITCMP0_B32: state.scc = int((s0 & (1 << (s1 & 0x1f))) == 0)
  elif op == SOPCOp.S_BITCMP1_B32: state.scc = int((s0 & (1 << (s1 & 0x1f))) != 0)
  elif op == SOPCOp.S_CMP_EQ_U64:
    s0_64, s1_64 = read_src64(state, ssrc0, 0), read_src64(state, ssrc1, 0)
    state.scc = int(s0_64 == s1_64)
  elif op == SOPCOp.S_CMP_LG_U64:
    s0_64, s1_64 = read_src64(state, ssrc0, 0), read_src64(state, ssrc1, 0)
    state.scc = int(s0_64 != s1_64)
  else:
    raise NotImplementedError(f"SOPC op {op} ({SOPCOp(op).name})")
  return 0

def exec_sopk(state: WaveState, inst: SOPK) -> int:
  op = get_field(inst, 'op')
  sdst = get_field(inst, 'sdst')
  simm16 = sign_ext(get_field(inst, 'simm16'), 16)
  s0 = read_sgpr(state, sdst)

  if op == SOPKOp.S_MOVK_I32:
    write_sgpr(state, sdst, simm16 & 0xffffffff)
  elif op == SOPKOp.S_CMPK_EQ_I32: state.scc = int(sign_ext(s0, 32) == simm16)
  elif op == SOPKOp.S_CMPK_LG_I32: state.scc = int(sign_ext(s0, 32) != simm16)
  elif op == SOPKOp.S_CMPK_GT_I32: state.scc = int(sign_ext(s0, 32) > simm16)
  elif op == SOPKOp.S_CMPK_GE_I32: state.scc = int(sign_ext(s0, 32) >= simm16)
  elif op == SOPKOp.S_CMPK_LT_I32: state.scc = int(sign_ext(s0, 32) < simm16)
  elif op == SOPKOp.S_CMPK_LE_I32: state.scc = int(sign_ext(s0, 32) <= simm16)
  elif op == SOPKOp.S_CMPK_EQ_U32: state.scc = int(s0 == (simm16 & 0xffff))
  elif op == SOPKOp.S_CMPK_LG_U32: state.scc = int(s0 != (simm16 & 0xffff))
  elif op == SOPKOp.S_CMPK_GT_U32: state.scc = int(s0 > (simm16 & 0xffff))
  elif op == SOPKOp.S_CMPK_GE_U32: state.scc = int(s0 >= (simm16 & 0xffff))
  elif op == SOPKOp.S_CMPK_LT_U32: state.scc = int(s0 < (simm16 & 0xffff))
  elif op == SOPKOp.S_CMPK_LE_U32: state.scc = int(s0 <= (simm16 & 0xffff))
  elif op == SOPKOp.S_ADDK_I32:
    s0_s = sign_ext(s0, 32)
    result = s0_s + simm16
    overflow = ((s0 >> 31) == ((simm16 >> 15) & 1)) and ((s0 >> 31) != ((result >> 31) & 1))
    state.scc = int(overflow)
    write_sgpr(state, sdst, result & 0xffffffff)
  elif op == SOPKOp.S_MULK_I32:
    result = (sign_ext(s0, 32) * simm16) & 0xffffffff
    write_sgpr(state, sdst, result)
  else:
    raise NotImplementedError(f"SOPK op {op} ({SOPKOp(op).name})")
  return 0

def exec_sopp(state: WaveState, inst: SOPP) -> int:
  op = get_field(inst, 'op')
  simm16 = sign_ext(get_field(inst, 'simm16'), 16)

  if op == SOPPOp.S_NOP: pass
  elif op == SOPPOp.S_ENDPGM: return -1  # signal end
  elif op == SOPPOp.S_BRANCH: return simm16
  elif op == SOPPOp.S_CBRANCH_SCC0: return simm16 if state.scc == 0 else 0
  elif op == SOPPOp.S_CBRANCH_SCC1: return simm16 if state.scc == 1 else 0
  elif op == SOPPOp.S_CBRANCH_VCCZ: return simm16 if state.vcc == 0 else 0
  elif op == SOPPOp.S_CBRANCH_VCCNZ: return simm16 if state.vcc != 0 else 0
  elif op == SOPPOp.S_CBRANCH_EXECZ: return simm16 if state.exec_mask == 0 else 0
  elif op == SOPPOp.S_CBRANCH_EXECNZ: return simm16 if state.exec_mask != 0 else 0
  elif op == SOPPOp.S_BARRIER: return -2  # signal barrier
  elif op == SOPPOp.S_WAITCNT: pass  # ignore for now
  elif op == SOPPOp.S_SENDMSG: pass  # ignore
  elif op == SOPPOp.S_CLAUSE: pass  # ignore
  elif op == SOPPOp.S_DELAY_ALU: pass  # ignore
  elif op == SOPPOp.S_WAIT_IDLE: pass  # ignore
  elif op == SOPPOp.S_WAIT_EVENT: pass  # ignore
  elif op == SOPPOp.S_SENDMSGHALT: pass  # ignore
  # unknown ops - just ignore (may be undefined or new instructions)
  elif op not in SOPPOp._value2member_map_: pass
  return 0

def exec_smem(state: WaveState, inst: SMEM) -> int:
  op = get_field(inst, 'op')
  sbase = get_field(inst, 'sbase') * 2  # sbase is in pairs
  sdata = get_field(inst, 'sdata')
  offset = sign_ext(get_field(inst, 'offset'), 21)
  soffset_idx = get_field(inst, 'soffset')
  # soffset encoding: 0x7f (127) or NULL_REG (124) means OFF (no soffset)
  soffset = read_src(state, soffset_idx, 0) if soffset_idx not in (NULL_REG, 0x7f) else 0

  base_addr = read_sgpr64(state, sbase)
  addr = (base_addr + offset + soffset) & 0xffffffffffffffff

  if op == SMEMOp.S_LOAD_B32:
    val = ctypes.c_uint32.from_address(addr).value
    write_sgpr(state, sdata, val)
  elif op == SMEMOp.S_LOAD_B64:
    for i in range(2):
      val = ctypes.c_uint32.from_address(addr + i * 4).value
      write_sgpr(state, sdata + i, val)
  elif op == SMEMOp.S_LOAD_B128:
    for i in range(4):
      val = ctypes.c_uint32.from_address(addr + i * 4).value
      write_sgpr(state, sdata + i, val)
  elif op == SMEMOp.S_LOAD_B256:
    for i in range(8):
      val = ctypes.c_uint32.from_address(addr + i * 4).value
      write_sgpr(state, sdata + i, val)
  elif op == SMEMOp.S_LOAD_B512:
    for i in range(16):
      val = ctypes.c_uint32.from_address(addr + i * 4).value
      write_sgpr(state, sdata + i, val)
  else:
    raise NotImplementedError(f"SMEM op {op} ({SMEMOp(op).name})")
  return 0

def exec_vop1(state: WaveState, inst: VOP1, lane: int) -> None:
  op = get_field(inst, 'op')
  src0 = get_field(inst, 'src0')
  vdst = get_field(inst, 'vdst')
  s0 = read_src(state, src0, lane)

  if op == VOP1Op.V_NOP: return
  elif op == VOP1Op.V_MOV_B32: state.vgpr[lane][vdst] = s0
  elif op == VOP1Op.V_READFIRSTLANE_B32:
    # find first active lane
    first = (state.exec_mask & 0xffffffff).bit_length() - 1 if state.exec_mask else 0
    val = read_src(state, src0, first) if src0 >= 256 else s0
    write_sgpr(state, vdst, val)
  elif op == VOP1Op.V_CVT_F32_I32:
    state.vgpr[lane][vdst] = f32_to_bits(float(sign_ext(s0, 32)))
  elif op == VOP1Op.V_CVT_F32_U32:
    state.vgpr[lane][vdst] = f32_to_bits(float(s0))
  elif op == VOP1Op.V_CVT_U32_F32:
    f = bits_to_f32(s0)
    state.vgpr[lane][vdst] = max(0, min(0xffffffff, int(f)))
  elif op == VOP1Op.V_CVT_I32_F32:
    f = bits_to_f32(s0)
    state.vgpr[lane][vdst] = max(-0x80000000, min(0x7fffffff, int(f))) & 0xffffffff
  elif op == VOP1Op.V_CVT_F16_F32:
    state.vgpr[lane][vdst] = f16_to_bits(bits_to_f32(s0))
  elif op == VOP1Op.V_CVT_F32_F16:
    state.vgpr[lane][vdst] = f32_to_bits(bits_to_f16(s0 & 0xffff))
  elif op == VOP1Op.V_TRUNC_F32:
    state.vgpr[lane][vdst] = f32_to_bits(math.trunc(bits_to_f32(s0)))
  elif op == VOP1Op.V_CEIL_F32:
    state.vgpr[lane][vdst] = f32_to_bits(math.ceil(bits_to_f32(s0)))
  elif op == VOP1Op.V_RNDNE_F32:
    f = bits_to_f32(s0)
    state.vgpr[lane][vdst] = f32_to_bits(round(f))  # banker's rounding
  elif op == VOP1Op.V_FLOOR_F32:
    state.vgpr[lane][vdst] = f32_to_bits(math.floor(bits_to_f32(s0)))
  elif op == VOP1Op.V_EXP_F32:
    state.vgpr[lane][vdst] = f32_to_bits(math.pow(2.0, bits_to_f32(s0)))
  elif op == VOP1Op.V_LOG_F32:
    f = bits_to_f32(s0)
    state.vgpr[lane][vdst] = f32_to_bits(math.log2(f) if f > 0 else float('-inf'))
  elif op == VOP1Op.V_RCP_F32:
    f = bits_to_f32(s0)
    state.vgpr[lane][vdst] = f32_to_bits(1.0 / f if f != 0 else math.copysign(float('inf'), f))
  elif op == VOP1Op.V_RCP_IFLAG_F32:
    f = bits_to_f32(s0)
    state.vgpr[lane][vdst] = f32_to_bits(1.0 / f if f != 0 else math.copysign(float('inf'), f))
  elif op == VOP1Op.V_RSQ_F32:
    f = bits_to_f32(s0)
    state.vgpr[lane][vdst] = f32_to_bits(1.0 / math.sqrt(f) if f > 0 else float('inf'))
  elif op == VOP1Op.V_SQRT_F32:
    state.vgpr[lane][vdst] = f32_to_bits(math.sqrt(max(0, bits_to_f32(s0))))
  elif op == VOP1Op.V_SIN_F32:
    state.vgpr[lane][vdst] = f32_to_bits(math.sin(bits_to_f32(s0) * 2 * math.pi))
  elif op == VOP1Op.V_COS_F32:
    state.vgpr[lane][vdst] = f32_to_bits(math.cos(bits_to_f32(s0) * 2 * math.pi))
  elif op == VOP1Op.V_NOT_B32:
    state.vgpr[lane][vdst] = (~s0) & 0xffffffff
  elif op == VOP1Op.V_BFREV_B32:
    state.vgpr[lane][vdst] = int(f'{s0:032b}'[::-1], 2)
  elif op == VOP1Op.V_CLZ_I32_U32:
    state.vgpr[lane][vdst] = clz_i32_u32(s0)
  elif op == VOP1Op.V_CLS_I32:
    state.vgpr[lane][vdst] = cls_i32(s0)
  elif op == VOP1Op.V_CVT_F32_UBYTE0:
    state.vgpr[lane][vdst] = f32_to_bits(float(s0 & 0xff))
  elif op == VOP1Op.V_CVT_F32_UBYTE1:
    state.vgpr[lane][vdst] = f32_to_bits(float((s0 >> 8) & 0xff))
  elif op == VOP1Op.V_CVT_F32_UBYTE2:
    state.vgpr[lane][vdst] = f32_to_bits(float((s0 >> 16) & 0xff))
  elif op == VOP1Op.V_CVT_F32_UBYTE3:
    state.vgpr[lane][vdst] = f32_to_bits(float((s0 >> 24) & 0xff))
  else:
    raise NotImplementedError(f"VOP1 op {op} ({VOP1Op(op).name})")

def exec_vop2(state: WaveState, inst: VOP2, lane: int) -> None:
  op = get_field(inst, 'op')
  src0 = get_field(inst, 'src0')
  vsrc1 = get_field(inst, 'vsrc1')
  vdst = get_field(inst, 'vdst')
  s0 = read_src(state, src0, lane)
  s1 = state.vgpr[lane][vsrc1]

  if op == VOP2Op.V_CNDMASK_B32:
    mask = state.vcc if True else 0  # uses VCC by default
    state.vgpr[lane][vdst] = s1 if (mask >> lane) & 1 else s0
  elif op == VOP2Op.V_ADD_F32:
    state.vgpr[lane][vdst] = f32_to_bits(bits_to_f32(s0) + bits_to_f32(s1))
  elif op == VOP2Op.V_SUB_F32:
    state.vgpr[lane][vdst] = f32_to_bits(bits_to_f32(s0) - bits_to_f32(s1))
  elif op == VOP2Op.V_SUBREV_F32:
    state.vgpr[lane][vdst] = f32_to_bits(bits_to_f32(s1) - bits_to_f32(s0))
  elif op == VOP2Op.V_MUL_F32:
    state.vgpr[lane][vdst] = f32_to_bits(bits_to_f32(s0) * bits_to_f32(s1))
  elif op == VOP2Op.V_MUL_I32_I24:
    a = sign_ext(s0 & 0xffffff, 24)
    b = sign_ext(s1 & 0xffffff, 24)
    state.vgpr[lane][vdst] = (a * b) & 0xffffffff
  elif op == VOP2Op.V_MUL_HI_I32_I24:
    a = sign_ext(s0 & 0xffffff, 24)
    b = sign_ext(s1 & 0xffffff, 24)
    state.vgpr[lane][vdst] = ((a * b) >> 32) & 0xffffffff
  elif op == VOP2Op.V_MUL_U32_U24:
    a = s0 & 0xffffff
    b = s1 & 0xffffff
    state.vgpr[lane][vdst] = (a * b) & 0xffffffff
  elif op == VOP2Op.V_MUL_HI_U32_U24:
    a = s0 & 0xffffff
    b = s1 & 0xffffff
    state.vgpr[lane][vdst] = ((a * b) >> 32) & 0xffffffff
  elif op == VOP2Op.V_MIN_F32:
    state.vgpr[lane][vdst] = f32_to_bits(min(bits_to_f32(s0), bits_to_f32(s1)))
  elif op == VOP2Op.V_MAX_F32:
    state.vgpr[lane][vdst] = f32_to_bits(max(bits_to_f32(s0), bits_to_f32(s1)))
  elif op == VOP2Op.V_MIN_I32:
    state.vgpr[lane][vdst] = s0 if sign_ext(s0, 32) < sign_ext(s1, 32) else s1
  elif op == VOP2Op.V_MAX_I32:
    state.vgpr[lane][vdst] = s0 if sign_ext(s0, 32) > sign_ext(s1, 32) else s1
  elif op == VOP2Op.V_MIN_U32:
    state.vgpr[lane][vdst] = min(s0, s1)
  elif op == VOP2Op.V_MAX_U32:
    state.vgpr[lane][vdst] = max(s0, s1)
  elif op == VOP2Op.V_LSHLREV_B32:
    state.vgpr[lane][vdst] = (s1 << (s0 & 0x1f)) & 0xffffffff
  elif op == VOP2Op.V_LSHRREV_B32:
    state.vgpr[lane][vdst] = s1 >> (s0 & 0x1f)
  elif op == VOP2Op.V_ASHRREV_I32:
    state.vgpr[lane][vdst] = (sign_ext(s1, 32) >> (s0 & 0x1f)) & 0xffffffff
  elif op == VOP2Op.V_AND_B32:
    state.vgpr[lane][vdst] = s0 & s1
  elif op == VOP2Op.V_OR_B32:
    state.vgpr[lane][vdst] = s0 | s1
  elif op == VOP2Op.V_XOR_B32:
    state.vgpr[lane][vdst] = s0 ^ s1
  elif op == VOP2Op.V_XNOR_B32:
    state.vgpr[lane][vdst] = ~(s0 ^ s1) & 0xffffffff
  elif op == VOP2Op.V_ADD_CO_CI_U32:
    carry_in = (state.vcc >> lane) & 1
    result = s0 + s1 + carry_in
    carry_out = int(result >= 0x100000000)
    state.vgpr[lane][vdst] = result & 0xffffffff
    state.vcc = (state.vcc & ~(1 << lane)) | (carry_out << lane)
  elif op == VOP2Op.V_SUB_CO_CI_U32:
    borrow_in = (state.vcc >> lane) & 1
    result = s0 - s1 - borrow_in
    borrow_out = int(s1 + borrow_in > s0)
    state.vgpr[lane][vdst] = result & 0xffffffff
    state.vcc = (state.vcc & ~(1 << lane)) | (borrow_out << lane)
  elif op == VOP2Op.V_SUBREV_CO_CI_U32:
    borrow_in = (state.vcc >> lane) & 1
    result = s1 - s0 - borrow_in
    borrow_out = int(s0 + borrow_in > s1)
    state.vgpr[lane][vdst] = result & 0xffffffff
    state.vcc = (state.vcc & ~(1 << lane)) | (borrow_out << lane)
  elif op == VOP2Op.V_ADD_NC_U32:
    state.vgpr[lane][vdst] = (s0 + s1) & 0xffffffff
  elif op == VOP2Op.V_SUB_NC_U32:
    state.vgpr[lane][vdst] = (s0 - s1) & 0xffffffff
  elif op == VOP2Op.V_SUBREV_NC_U32:
    state.vgpr[lane][vdst] = (s1 - s0) & 0xffffffff
  elif op == VOP2Op.V_FMAC_F32:
    acc = bits_to_f32(state.vgpr[lane][vdst])
    state.vgpr[lane][vdst] = f32_to_bits(bits_to_f32(s0) * bits_to_f32(s1) + acc)
  else:
    raise NotImplementedError(f"VOP2 op {op} ({VOP2Op(op).name})")

def apply_vop3_modifiers(val: int, neg: int, abs_mod: int, idx: int) -> int:
  """Apply neg and abs modifiers to a float value."""
  if (abs_mod >> idx) & 1:
    f = abs(bits_to_f32(val))
    val = f32_to_bits(f)
  if (neg >> idx) & 1:
    f = -bits_to_f32(val)
    val = f32_to_bits(f)
  return val

def exec_vop3(state: WaveState, inst: VOP3, lane: int) -> None:
  op = get_field(inst, 'op')
  src0, src1, src2 = get_field(inst, 'src0'), get_field(inst, 'src1'), get_field(inst, 'src2')
  vdst = get_field(inst, 'vdst')
  neg = get_field(inst, 'neg')
  abs_mod = get_field(inst, 'abs')

  s0 = apply_vop3_modifiers(read_src(state, src0, lane), neg, abs_mod, 0)
  s1 = apply_vop3_modifiers(read_src(state, src1, lane), neg, abs_mod, 1)
  s2 = apply_vop3_modifiers(read_src(state, src2, lane), neg, abs_mod, 2)

  # VOP3 versions of VOP1/VOP2
  if op == VOP3Op.V_MOV_B32: state.vgpr[lane][vdst] = s0
  elif op == VOP3Op.V_ADD_F32: state.vgpr[lane][vdst] = f32_to_bits(bits_to_f32(s0) + bits_to_f32(s1))
  elif op == VOP3Op.V_SUB_F32: state.vgpr[lane][vdst] = f32_to_bits(bits_to_f32(s0) - bits_to_f32(s1))
  elif op == VOP3Op.V_SUBREV_F32: state.vgpr[lane][vdst] = f32_to_bits(bits_to_f32(s1) - bits_to_f32(s0))
  elif op == VOP3Op.V_MUL_F32: state.vgpr[lane][vdst] = f32_to_bits(bits_to_f32(s0) * bits_to_f32(s1))
  elif op == VOP3Op.V_MIN_F32: state.vgpr[lane][vdst] = f32_to_bits(min(bits_to_f32(s0), bits_to_f32(s1)))
  elif op == VOP3Op.V_MAX_F32: state.vgpr[lane][vdst] = f32_to_bits(max(bits_to_f32(s0), bits_to_f32(s1)))
  elif op == VOP3Op.V_FMAC_F32:
    acc = bits_to_f32(state.vgpr[lane][vdst])
    state.vgpr[lane][vdst] = f32_to_bits(bits_to_f32(s0) * bits_to_f32(s1) + acc)
  elif op == VOP3Op.V_FMA_F32:
    state.vgpr[lane][vdst] = f32_to_bits(bits_to_f32(s0) * bits_to_f32(s1) + bits_to_f32(s2))
  elif op == VOP3Op.V_AND_B32: state.vgpr[lane][vdst] = s0 & s1
  elif op == VOP3Op.V_OR_B32: state.vgpr[lane][vdst] = s0 | s1
  elif op == VOP3Op.V_XOR_B32: state.vgpr[lane][vdst] = s0 ^ s1
  elif op == VOP3Op.V_LSHLREV_B32: state.vgpr[lane][vdst] = (s1 << (s0 & 0x1f)) & 0xffffffff
  elif op == VOP3Op.V_LSHRREV_B32: state.vgpr[lane][vdst] = s1 >> (s0 & 0x1f)
  elif op == VOP3Op.V_ASHRREV_I32: state.vgpr[lane][vdst] = (sign_ext(s1, 32) >> (s0 & 0x1f)) & 0xffffffff
  elif op == VOP3Op.V_ADD_NC_U32: state.vgpr[lane][vdst] = (s0 + s1) & 0xffffffff
  elif op == VOP3Op.V_SUB_NC_U32: state.vgpr[lane][vdst] = (s0 - s1) & 0xffffffff
  elif op == VOP3Op.V_MIN_I32: state.vgpr[lane][vdst] = s0 if sign_ext(s0, 32) < sign_ext(s1, 32) else s1
  elif op == VOP3Op.V_MAX_I32: state.vgpr[lane][vdst] = s0 if sign_ext(s0, 32) > sign_ext(s1, 32) else s1
  elif op == VOP3Op.V_MIN_U32: state.vgpr[lane][vdst] = min(s0, s1)
  elif op == VOP3Op.V_MAX_U32: state.vgpr[lane][vdst] = max(s0, s1)
  elif op == VOP3Op.V_CNDMASK_B32:
    mask_src = src2  # in VOP3, mask comes from src2
    mask = read_src(state, mask_src, lane) if mask_src < 256 else state.vcc
    state.vgpr[lane][vdst] = s1 if (mask >> lane) & 1 else s0
  elif op == VOP3Op.V_BFE_U32:
    offset = s1 & 0x1f
    width = (s1 >> 16) & 0x1f if s1 > 255 else s2 & 0x1f
    state.vgpr[lane][vdst] = (s0 >> offset) & ((1 << width) - 1) if width else 0
  elif op == VOP3Op.V_BFE_I32:
    offset = s1 & 0x1f
    width = s2 & 0x1f
    if width == 0:
      state.vgpr[lane][vdst] = 0
    else:
      result = (s0 >> offset) & ((1 << width) - 1)
      state.vgpr[lane][vdst] = sign_ext(result, width) & 0xffffffff
  elif op == VOP3Op.V_ALIGNBIT_B32:
    shift = s2 & 0x1f
    combined = (s0 << 32) | s1
    state.vgpr[lane][vdst] = (combined >> shift) & 0xffffffff
  elif op == VOP3Op.V_ADD3_U32:
    state.vgpr[lane][vdst] = (s0 + s1 + s2) & 0xffffffff
  elif op == VOP3Op.V_LSHL_ADD_U32:
    state.vgpr[lane][vdst] = ((s0 << (s1 & 0x1f)) + s2) & 0xffffffff
  elif op == VOP3Op.V_ADD_LSHL_U32:
    state.vgpr[lane][vdst] = (((s0 + s1) << (s2 & 0x1f))) & 0xffffffff
  elif op == VOP3Op.V_LSHLREV_B64:
    s1_64 = read_src64(state, src1, lane)
    result = (s1_64 << (s0 & 0x3f)) & 0xffffffffffffffff
    state.vgpr[lane][vdst] = result & 0xffffffff
    state.vgpr[lane][vdst + 1] = (result >> 32) & 0xffffffff
  elif op == VOP3Op.V_MUL_LO_U32:
    state.vgpr[lane][vdst] = (s0 * s1) & 0xffffffff
  elif op == VOP3Op.V_MUL_HI_U32:
    state.vgpr[lane][vdst] = ((s0 * s1) >> 32) & 0xffffffff
  elif op == VOP3Op.V_MUL_HI_I32:
    result = sign_ext(s0, 32) * sign_ext(s1, 32)
    state.vgpr[lane][vdst] = (result >> 32) & 0xffffffff
  elif op == VOP3Op.V_MAD_U32_U24:
    a = s0 & 0xffffff
    b = s1 & 0xffffff
    state.vgpr[lane][vdst] = (a * b + s2) & 0xffffffff
  elif op == VOP3Op.V_MAD_I32_I24:
    a = sign_ext(s0 & 0xffffff, 24)
    b = sign_ext(s1 & 0xffffff, 24)
    state.vgpr[lane][vdst] = (a * b + sign_ext(s2, 32)) & 0xffffffff
  elif op == VOP3Op.V_RCP_F32:
    f = bits_to_f32(s0)
    state.vgpr[lane][vdst] = f32_to_bits(1.0 / f if f != 0 else math.copysign(float('inf'), f))
  elif op == VOP3Op.V_RSQ_F32:
    f = bits_to_f32(s0)
    state.vgpr[lane][vdst] = f32_to_bits(1.0 / math.sqrt(f) if f > 0 else float('inf'))
  elif op == VOP3Op.V_SQRT_F32:
    state.vgpr[lane][vdst] = f32_to_bits(math.sqrt(max(0, bits_to_f32(s0))))
  elif op == VOP3Op.V_EXP_F32:
    state.vgpr[lane][vdst] = f32_to_bits(math.pow(2.0, bits_to_f32(s0)))
  elif op == VOP3Op.V_LOG_F32:
    f = bits_to_f32(s0)
    state.vgpr[lane][vdst] = f32_to_bits(math.log2(f) if f > 0 else float('-inf'))
  elif op == VOP3Op.V_FLOOR_F32:
    state.vgpr[lane][vdst] = f32_to_bits(math.floor(bits_to_f32(s0)))
  elif op == VOP3Op.V_CEIL_F32:
    state.vgpr[lane][vdst] = f32_to_bits(math.ceil(bits_to_f32(s0)))
  elif op == VOP3Op.V_TRUNC_F32:
    state.vgpr[lane][vdst] = f32_to_bits(math.trunc(bits_to_f32(s0)))
  elif op == VOP3Op.V_CVT_F32_I32:
    state.vgpr[lane][vdst] = f32_to_bits(float(sign_ext(s0, 32)))
  elif op == VOP3Op.V_CVT_F32_U32:
    state.vgpr[lane][vdst] = f32_to_bits(float(s0))
  elif op == VOP3Op.V_CVT_I32_F32:
    f = bits_to_f32(s0)
    state.vgpr[lane][vdst] = max(-0x80000000, min(0x7fffffff, int(f))) & 0xffffffff
  elif op == VOP3Op.V_CVT_U32_F32:
    f = bits_to_f32(s0)
    state.vgpr[lane][vdst] = max(0, min(0xffffffff, int(f)))
  elif op == VOP3Op.V_LDEXP_F32:
    # ldexp(x, n) = x * 2^n
    f = bits_to_f32(s0)
    exp = sign_ext(s1, 32)
    state.vgpr[lane][vdst] = f32_to_bits(math.ldexp(f, exp))
  elif op == VOP3Op.V_FREXP_MANT_F32:
    f = bits_to_f32(s0)
    mant, _ = math.frexp(f) if f != 0 else (0.0, 0)
    state.vgpr[lane][vdst] = f32_to_bits(mant)
  elif op == VOP3Op.V_FREXP_EXP_I32_F32:
    f = bits_to_f32(s0)
    _, exp = math.frexp(f) if f != 0 else (0.0, 0)
    state.vgpr[lane][vdst] = exp & 0xffffffff
  # Division helpers
  elif op == VOP3Op.V_DIV_FMAS_F32:
    # FMA with special handling for division: D = S0 * S1 + S2, with VCC-based scaling
    f0, f1, f2 = bits_to_f32(s0), bits_to_f32(s1), bits_to_f32(s2)
    # If VCC[lane] is set, apply scale factor (typically 2^32 for double precision emulation)
    vcc_bit = (state.vcc >> lane) & 1
    if vcc_bit:
      result = f0 * f1 * (2.0**32) + f2
    else:
      result = f0 * f1 + f2
    state.vgpr[lane][vdst] = f32_to_bits(result)
  elif op == VOP3Op.V_DIV_FIXUP_F32:
    # Division fixup: handles special cases like div-by-zero, inf, etc.
    # D = fixup(S0/S1, S2) where S0=quotient, S1=divisor, S2=dividend
    f0, f1, f2 = bits_to_f32(s0), bits_to_f32(s1), bits_to_f32(s2)
    # Simple implementation: return S0 (the quotient) for normal cases
    if f1 == 0.0:
      # Division by zero
      result = math.copysign(float('inf'), f2) if f2 != 0 else float('nan')
    elif math.isnan(f0) or math.isnan(f1) or math.isnan(f2):
      result = float('nan')
    elif math.isinf(f2) and math.isinf(f1):
      result = float('nan')
    else:
      result = f0
    state.vgpr[lane][vdst] = f32_to_bits(result)
  # comparisons (VOP3 versions write to arbitrary SGPR)
  elif 0 <= op <= 127 or 128 <= op <= 255:  # V_CMP_* or V_CMPX_*
    exec_vopc_vop3(state, op, s0, s1, vdst, lane)
  else:
    raise NotImplementedError(f"VOP3 op {op} ({VOP3Op(op).name if op in VOP3Op._value2member_map_ else 'unknown'})")

def exec_vopc_vop3(state: WaveState, op: int, s0: int, s1: int, sdst: int, lane: int) -> None:
  """Execute VOPC-style comparison, writing to sdst."""
  is_cmpx = op >= 128
  base_op = op - 128 if is_cmpx else op
  result = False

  # float comparisons
  if 16 <= base_op <= 31:  # F32
    f0, f1 = bits_to_f32(s0), bits_to_f32(s1)
    cmp_op = base_op - 16
    if cmp_op == 0: result = False  # F
    elif cmp_op == 1: result = f0 < f1
    elif cmp_op == 2: result = f0 == f1
    elif cmp_op == 3: result = f0 <= f1
    elif cmp_op == 4: result = f0 > f1
    elif cmp_op == 5: result = f0 != f1
    elif cmp_op == 6: result = f0 >= f1
    elif cmp_op == 7: result = not (math.isnan(f0) or math.isnan(f1))  # O
    elif cmp_op == 8: result = math.isnan(f0) or math.isnan(f1)  # U
    elif cmp_op == 15: result = True  # T
  # int comparisons
  elif 64 <= base_op <= 79:  # I32
    cmp_op = (base_op - 64) % 8
    s0_s, s1_s = sign_ext(s0, 32), sign_ext(s1, 32)
    if cmp_op == 0: result = False
    elif cmp_op == 1: result = s0_s < s1_s
    elif cmp_op == 2: result = s0_s == s1_s
    elif cmp_op == 3: result = s0_s <= s1_s
    elif cmp_op == 4: result = s0_s > s1_s
    elif cmp_op == 5: result = s0_s != s1_s
    elif cmp_op == 6: result = s0_s >= s1_s
    elif cmp_op == 7: result = True
  elif 72 <= base_op <= 79:  # U32
    cmp_op = (base_op - 72) % 8
    if cmp_op == 0: result = False
    elif cmp_op == 1: result = s0 < s1
    elif cmp_op == 2: result = s0 == s1
    elif cmp_op == 3: result = s0 <= s1
    elif cmp_op == 4: result = s0 > s1
    elif cmp_op == 5: result = s0 != s1
    elif cmp_op == 6: result = s0 >= s1
    elif cmp_op == 7: result = True

  # write result
  old = read_sgpr(state, sdst) if sdst != VCC_LO else state.vcc
  new_val = (old & ~(1 << lane)) | (int(result) << lane)
  if sdst == VCC_LO:
    state.vcc = new_val
  else:
    write_sgpr(state, sdst, new_val)

  if is_cmpx:
    state.exec_mask = (state.exec_mask & ~(1 << lane)) | (int(result) << lane)

def exec_vopc(state: WaveState, inst: VOPC, lane: int) -> None:
  """Execute VOPC instruction - always writes to VCC."""
  op = get_field(inst, 'op')
  src0 = get_field(inst, 'src0')
  vsrc1 = get_field(inst, 'vsrc1')
  s0 = read_src(state, src0, lane)
  s1 = state.vgpr[lane][vsrc1]

  result = False
  is_cmpx = op >= 128
  base_op = op - 128 if is_cmpx else op

  # F32 comparisons
  if 16 <= base_op <= 31:
    f0, f1 = bits_to_f32(s0), bits_to_f32(s1)
    cmp_op = base_op - 16
    if cmp_op == 1: result = f0 < f1
    elif cmp_op == 2: result = f0 == f1
    elif cmp_op == 3: result = f0 <= f1
    elif cmp_op == 4: result = f0 > f1
    elif cmp_op == 5: result = f0 != f1
    elif cmp_op == 6: result = f0 >= f1
    elif cmp_op == 15: result = True
  # I32 comparisons
  elif 64 <= base_op <= 71:
    s0_s, s1_s = sign_ext(s0, 32), sign_ext(s1, 32)
    cmp_op = base_op - 64
    if cmp_op == 1: result = s0_s < s1_s
    elif cmp_op == 2: result = s0_s == s1_s
    elif cmp_op == 3: result = s0_s <= s1_s
    elif cmp_op == 4: result = s0_s > s1_s
    elif cmp_op == 5: result = s0_s != s1_s
    elif cmp_op == 6: result = s0_s >= s1_s
    elif cmp_op == 7: result = True
  # U32 comparisons
  elif 72 <= base_op <= 79:
    cmp_op = base_op - 72
    if cmp_op == 1: result = s0 < s1
    elif cmp_op == 2: result = s0 == s1
    elif cmp_op == 3: result = s0 <= s1
    elif cmp_op == 4: result = s0 > s1
    elif cmp_op == 5: result = s0 != s1
    elif cmp_op == 6: result = s0 >= s1
    elif cmp_op == 7: result = True

  state.vcc = (state.vcc & ~(1 << lane)) | (int(result) << lane)
  if is_cmpx:
    state.exec_mask = (state.exec_mask & ~(1 << lane)) | (int(result) << lane)

def exec_vop3sd(state: WaveState, inst: VOP3SD, lane: int) -> None:
  """Execute VOP3SD instruction (VOP3 with scalar dest for carry)."""
  op = get_field(inst, 'op')
  src0, src1, src2 = get_field(inst, 'src0'), get_field(inst, 'src1'), get_field(inst, 'src2')
  vdst = get_field(inst, 'vdst')
  sdst = get_field(inst, 'sdst')
  neg = get_field(inst, 'neg')

  s0 = read_src(state, src0, lane)
  s1 = read_src(state, src1, lane)
  s2 = read_src(state, src2, lane)

  if (neg >> 0) & 1: s0 = f32_to_bits(-bits_to_f32(s0))
  if (neg >> 1) & 1: s1 = f32_to_bits(-bits_to_f32(s1))
  if (neg >> 2) & 1: s2 = f32_to_bits(-bits_to_f32(s2))

  if op == VOP3SDOp.V_ADD_CO_U32:
    result = s0 + s1
    carry = int(result >= 0x100000000)
    state.vgpr[lane][vdst] = result & 0xffffffff
    old_sdst = read_sgpr(state, sdst)
    write_sgpr(state, sdst, (old_sdst & ~(1 << lane)) | (carry << lane))
  elif op == VOP3SDOp.V_SUB_CO_U32:
    result = s0 - s1
    borrow = int(s1 > s0)
    state.vgpr[lane][vdst] = result & 0xffffffff
    old_sdst = read_sgpr(state, sdst)
    write_sgpr(state, sdst, (old_sdst & ~(1 << lane)) | (borrow << lane))
  elif op == VOP3SDOp.V_SUBREV_CO_U32:
    result = s1 - s0
    borrow = int(s0 > s1)
    state.vgpr[lane][vdst] = result & 0xffffffff
    old_sdst = read_sgpr(state, sdst)
    write_sgpr(state, sdst, (old_sdst & ~(1 << lane)) | (borrow << lane))
  elif op == VOP3SDOp.V_ADD_CO_CI_U32:
    carry_in = (read_sgpr(state, src2) >> lane) & 1 if src2 < 256 else (state.vcc >> lane) & 1
    result = s0 + s1 + carry_in
    carry = int(result >= 0x100000000)
    state.vgpr[lane][vdst] = result & 0xffffffff
    old_sdst = read_sgpr(state, sdst)
    write_sgpr(state, sdst, (old_sdst & ~(1 << lane)) | (carry << lane))
  elif op == VOP3SDOp.V_MAD_U64_U32:
    s2_64 = s2 | (read_src(state, src2 + 1, lane) << 32)
    result = s0 * s1 + s2_64
    state.vgpr[lane][vdst] = result & 0xffffffff
    state.vgpr[lane][vdst + 1] = (result >> 32) & 0xffffffff
  elif op == VOP3SDOp.V_MAD_I64_I32:
    s0_s, s1_s = sign_ext(s0, 32), sign_ext(s1, 32)
    s2_64 = s2 | (read_src(state, src2 + 1, lane) << 32)
    s2_s = sign_ext(s2_64, 64)
    result = (s0_s * s1_s + s2_s) & 0xffffffffffffffff
    state.vgpr[lane][vdst] = result & 0xffffffff
    state.vgpr[lane][vdst + 1] = (result >> 32) & 0xffffffff
  elif op == VOP3SDOp.V_DIV_SCALE_F32:
    # DIV_SCALE is used for Newton-Raphson division: computes scaled dividend or divisor
    # D = S0 == S1 ? S0 * 2^n : S0, where n is chosen to normalize the divisor
    # Also sets VCC if S0 == S1 (i.e., if we're computing the scaled divisor)
    f0, f1, f2 = bits_to_f32(s0), bits_to_f32(s1), bits_to_f32(s2)
    # Simple implementation: return s0 as-is, set VCC based on S0 == S1
    state.vgpr[lane][vdst] = s0
    # Set VCC bit based on whether S0 == S2 (for double-size division)
    vcc_bit = 1 if s0 == s2 else 0
    state.vcc = (state.vcc & ~(1 << lane)) | (vcc_bit << lane)
  elif op == VOP3SDOp.V_DIV_SCALE_F64:
    # 64-bit version
    state.vgpr[lane][vdst] = s0
    state.vgpr[lane][vdst + 1] = read_src(state, src0 + 1, lane)
    vcc_bit = 1 if s0 == s2 else 0
    state.vcc = (state.vcc & ~(1 << lane)) | (vcc_bit << lane)
  else:
    raise NotImplementedError(f"VOP3SD op {op} ({VOP3SDOp(op).name})")

def exec_flat(state: WaveState, inst: FLAT, lane: int) -> None:
  """Execute FLAT/GLOBAL memory instruction."""
  op = get_field(inst, 'op')
  addr_reg = get_field(inst, 'addr')
  data_reg = get_field(inst, 'data')
  vdst = get_field(inst, 'vdst')
  offset = sign_ext(get_field(inst, 'offset'), 13)
  saddr = get_field(inst, 'saddr')
  seg = get_field(inst, 'seg')

  # compute address
  addr_lo = state.vgpr[lane][addr_reg]
  addr_hi = state.vgpr[lane][addr_reg + 1]
  addr = (addr_hi << 32) | addr_lo

  if saddr != NULL_REG and saddr != 0x7f:  # not OFF
    saddr_val = read_sgpr64(state, saddr)
    addr = (saddr_val + addr_lo + offset) & 0xffffffffffffffff
  else:
    addr = (addr + offset) & 0xffffffffffffffff

  # loads
  if op == GLOBALOp.GLOBAL_LOAD_B32 or op == FLATOp.FLAT_LOAD_B32:
    state.vgpr[lane][vdst] = ctypes.c_uint32.from_address(addr).value
  elif op == GLOBALOp.GLOBAL_LOAD_B64 or op == FLATOp.FLAT_LOAD_B64:
    state.vgpr[lane][vdst] = ctypes.c_uint32.from_address(addr).value
    state.vgpr[lane][vdst + 1] = ctypes.c_uint32.from_address(addr + 4).value
  elif op == GLOBALOp.GLOBAL_LOAD_B128 or op == FLATOp.FLAT_LOAD_B128:
    for i in range(4):
      state.vgpr[lane][vdst + i] = ctypes.c_uint32.from_address(addr + i * 4).value
  elif op == GLOBALOp.GLOBAL_LOAD_U8 or op == FLATOp.FLAT_LOAD_U8:
    state.vgpr[lane][vdst] = ctypes.c_uint8.from_address(addr).value
  elif op == GLOBALOp.GLOBAL_LOAD_I8 or op == FLATOp.FLAT_LOAD_I8:
    state.vgpr[lane][vdst] = sign_ext(ctypes.c_uint8.from_address(addr).value, 8) & 0xffffffff
  elif op == GLOBALOp.GLOBAL_LOAD_U16 or op == FLATOp.FLAT_LOAD_U16:
    state.vgpr[lane][vdst] = ctypes.c_uint16.from_address(addr).value
  elif op == GLOBALOp.GLOBAL_LOAD_I16 or op == FLATOp.FLAT_LOAD_I16:
    state.vgpr[lane][vdst] = sign_ext(ctypes.c_uint16.from_address(addr).value, 16) & 0xffffffff
  # stores
  elif op == GLOBALOp.GLOBAL_STORE_B32 or op == FLATOp.FLAT_STORE_B32:
    ctypes.c_uint32.from_address(addr).value = state.vgpr[lane][data_reg]
  elif op == GLOBALOp.GLOBAL_STORE_B64 or op == FLATOp.FLAT_STORE_B64:
    ctypes.c_uint32.from_address(addr).value = state.vgpr[lane][data_reg]
    ctypes.c_uint32.from_address(addr + 4).value = state.vgpr[lane][data_reg + 1]
  elif op == GLOBALOp.GLOBAL_STORE_B128 or op == FLATOp.FLAT_STORE_B128:
    for i in range(4):
      ctypes.c_uint32.from_address(addr + i * 4).value = state.vgpr[lane][data_reg + i]
  elif op == GLOBALOp.GLOBAL_STORE_B8 or op == FLATOp.FLAT_STORE_B8:
    ctypes.c_uint8.from_address(addr).value = state.vgpr[lane][data_reg] & 0xff
  elif op == GLOBALOp.GLOBAL_STORE_B16 or op == FLATOp.FLAT_STORE_B16:
    ctypes.c_uint16.from_address(addr).value = state.vgpr[lane][data_reg] & 0xffff
  else:
    raise NotImplementedError(f"FLAT op {op}")

def exec_ds(state: WaveState, inst: DS, lane: int, lds: bytearray) -> None:
  """Execute DS (LDS) instruction."""
  op = get_field(inst, 'op')
  addr_reg = get_field(inst, 'addr')
  data0_reg = get_field(inst, 'data0')
  vdst = get_field(inst, 'vdst')
  offset0 = get_field(inst, 'offset0')
  offset1 = get_field(inst, 'offset1')

  addr = (state.vgpr[lane][addr_reg] + offset0) & 0xffff

  if op == DSOp.DS_LOAD_B32:
    val = int.from_bytes(lds[addr:addr+4], 'little')
    state.vgpr[lane][vdst] = val
  elif op == DSOp.DS_LOAD_B64:
    state.vgpr[lane][vdst] = int.from_bytes(lds[addr:addr+4], 'little')
    state.vgpr[lane][vdst + 1] = int.from_bytes(lds[addr+4:addr+8], 'little')
  elif op == DSOp.DS_LOAD_B128:
    for i in range(4):
      state.vgpr[lane][vdst + i] = int.from_bytes(lds[addr+i*4:addr+i*4+4], 'little')
  elif op == DSOp.DS_LOAD_U8:
    state.vgpr[lane][vdst] = lds[addr]
  elif op == DSOp.DS_LOAD_I8:
    state.vgpr[lane][vdst] = sign_ext(lds[addr], 8) & 0xffffffff
  elif op == DSOp.DS_LOAD_U16:
    state.vgpr[lane][vdst] = int.from_bytes(lds[addr:addr+2], 'little')
  elif op == DSOp.DS_LOAD_I16:
    state.vgpr[lane][vdst] = sign_ext(int.from_bytes(lds[addr:addr+2], 'little'), 16) & 0xffffffff
  elif op == DSOp.DS_STORE_B32:
    val = state.vgpr[lane][data0_reg]
    lds[addr:addr+4] = val.to_bytes(4, 'little')
  elif op == DSOp.DS_STORE_B64:
    lds[addr:addr+4] = state.vgpr[lane][data0_reg].to_bytes(4, 'little')
    lds[addr+4:addr+8] = state.vgpr[lane][data0_reg + 1].to_bytes(4, 'little')
  elif op == DSOp.DS_STORE_B128:
    for i in range(4):
      lds[addr+i*4:addr+i*4+4] = state.vgpr[lane][data0_reg + i].to_bytes(4, 'little')
  elif op == DSOp.DS_STORE_B8:
    lds[addr] = state.vgpr[lane][data0_reg] & 0xff
  elif op == DSOp.DS_STORE_B16:
    lds[addr:addr+2] = (state.vgpr[lane][data0_reg] & 0xffff).to_bytes(2, 'little')
  else:
    raise NotImplementedError(f"DS op {op} ({DSOp(op).name})")

# VOPD opcode tables (from RDNA3 ISA)
VOPD_OPX = {0: 'FMAC', 1: 'FMAAK', 2: 'FMAMK', 3: 'MUL', 4: 'ADD', 5: 'SUB', 6: 'SUBREV', 7: 'MUL_LEGACY',
            8: 'MOV', 9: 'CNDMASK', 10: 'MAX', 11: 'MIN', 12: 'DOT2C', 13: 'ADD_U32', 14: 'LSHLREV', 15: 'AND'}
VOPD_OPY = {0: 'FMAC', 1: 'FMAAK', 2: 'FMAMK', 3: 'MUL', 4: 'ADD', 5: 'SUB', 6: 'SUBREV', 7: 'MUL_LEGACY',
            8: 'MOV', 9: 'CNDMASK', 10: 'MAX', 11: 'MIN', 16: 'ADD_U32', 17: 'LSHLREV', 18: 'AND'}

def exec_vopd_op(state: WaveState, op: int, src0: int, src1: int, dst: int, lane: int, is_x: bool) -> None:
  """Execute a single VOPD operation (X or Y slot)."""
  op_table = VOPD_OPX if is_x else VOPD_OPY
  op_name = op_table.get(op, 'UNKNOWN')

  s0 = read_src(state, src0, lane)
  s1 = state.vgpr[lane][src1]

  if op_name == 'MOV':
    state.vgpr[lane][dst] = s0
  elif op_name == 'ADD':
    state.vgpr[lane][dst] = f32_to_bits(bits_to_f32(s0) + bits_to_f32(s1))
  elif op_name == 'SUB':
    state.vgpr[lane][dst] = f32_to_bits(bits_to_f32(s0) - bits_to_f32(s1))
  elif op_name == 'SUBREV':
    state.vgpr[lane][dst] = f32_to_bits(bits_to_f32(s1) - bits_to_f32(s0))
  elif op_name == 'MUL':
    state.vgpr[lane][dst] = f32_to_bits(bits_to_f32(s0) * bits_to_f32(s1))
  elif op_name == 'MAX':
    state.vgpr[lane][dst] = f32_to_bits(max(bits_to_f32(s0), bits_to_f32(s1)))
  elif op_name == 'MIN':
    state.vgpr[lane][dst] = f32_to_bits(min(bits_to_f32(s0), bits_to_f32(s1)))
  elif op_name == 'FMAC':
    acc = bits_to_f32(state.vgpr[lane][dst])
    state.vgpr[lane][dst] = f32_to_bits(bits_to_f32(s0) * bits_to_f32(s1) + acc)
  elif op_name == 'ADD_U32':
    state.vgpr[lane][dst] = (s0 + s1) & 0xffffffff
  elif op_name == 'LSHLREV':
    state.vgpr[lane][dst] = (s1 << (s0 & 0x1f)) & 0xffffffff
  elif op_name == 'AND':
    state.vgpr[lane][dst] = s0 & s1
  elif op_name == 'CNDMASK':
    mask = state.vcc
    state.vgpr[lane][dst] = s1 if (mask >> lane) & 1 else s0
  else:
    raise NotImplementedError(f"VOPD op {op} ({op_name}) in {'X' if is_x else 'Y'} slot")

def exec_vopd(state: WaveState, inst: VOPD, lane: int) -> None:
  """Execute VOPD (dual-issue VOP) instruction."""
  opx = get_field(inst, 'opx')
  opy = get_field(inst, 'opy')
  srcx0 = get_field(inst, 'srcx0')
  vsrcx1 = get_field(inst, 'vsrcx1')
  vdstx = get_field(inst, 'vdstx')
  srcy0 = get_field(inst, 'srcy0')
  vsrcy1 = get_field(inst, 'vsrcy1')
  vdsty_enc = get_field(inst, 'vdsty')

  # VOPD Y destination encoding depends on the Y operation's component class:
  # Component B (odd VGPRs): opcodes 3 (MUL), 8 (MOV), 10 (MAX), 11 (MIN) - use vdsty*2+1
  # Component A (even VGPRs): opcodes 4 (ADD), 5 (SUB), 9 (CNDMASK), etc - use vdsty directly
  # Reference: RDNA3 ISA, VOPD instruction format
  component_b_ops = {3, 8, 10, 11}  # MUL, MOV, MAX, MIN
  if opy in component_b_ops:
    vdsty = vdsty_enc * 2 + 1  # Y writes to odd registers
  else:
    vdsty = vdsty_enc  # Y writes directly

  # Execute both operations
  exec_vopd_op(state, opx, srcx0, vsrcx1, vdstx, lane, is_x=True)
  exec_vopd_op(state, opy, srcy0, vsrcy1, vdsty, lane, is_x=False)

# *** wave execution ***
def exec_wave(program: list[tuple[Inst, int]], state: WaveState, lds: bytearray, n_lanes: int) -> int:
  """Execute a wave until end or barrier. Returns 0 on end, -2 on barrier."""
  while state.pc < len(program):
    inst, dwords = program[state.pc]
    state.literal = inst._literal if inst._literal else 0

    # scalar instructions execute once
    if isinstance(inst, (SOP1, SOP2, SOPC, SOPK, SOPP, SMEM)):
      if isinstance(inst, SOP1): delta = exec_sop1(state, inst)
      elif isinstance(inst, SOP2): delta = exec_sop2(state, inst)
      elif isinstance(inst, SOPC): delta = exec_sopc(state, inst)
      elif isinstance(inst, SOPK): delta = exec_sopk(state, inst)
      elif isinstance(inst, SOPP): delta = exec_sopp(state, inst)
      elif isinstance(inst, SMEM): delta = exec_smem(state, inst)
      if delta == -1: return 0  # end program
      if delta == -2: state.pc += 1; return -2  # barrier
      state.pc += 1 + delta
    else:
      # vector instructions execute per-lane
      for lane in range(n_lanes):
        if not (state.exec_mask & (1 << lane)):
          continue
        if isinstance(inst, VOP1): exec_vop1(state, inst, lane)
        elif isinstance(inst, VOP2): exec_vop2(state, inst, lane)
        elif isinstance(inst, VOP3): exec_vop3(state, inst, lane)
        elif isinstance(inst, VOP3SD): exec_vop3sd(state, inst, lane)
        elif isinstance(inst, VOPC): exec_vopc(state, inst, lane)
        elif isinstance(inst, FLAT): exec_flat(state, inst, lane)
        elif isinstance(inst, DS): exec_ds(state, inst, lane, lds)
        elif isinstance(inst, VOPD): exec_vopd(state, inst, lane)
        else: raise NotImplementedError(f"Unknown instruction type: {type(inst)}")
      state.pc += 1
  return 0

# *** workgroup execution ***
def exec_workgroup(program: list[tuple[Inst, int]], workgroup_id: tuple[int, int, int],
                   local_size: tuple[int, int, int], args_ptr: int, dispatch_dim: int) -> None:
  """Execute a workgroup."""
  lx, ly, lz = local_size
  total_threads = lx * ly * lz
  lds = bytearray(65536)  # 64KB LDS

  # create waves
  waves = []
  for wave_start in range(0, total_threads, WAVE_SIZE):
    wave_end = min(wave_start + WAVE_SIZE, total_threads)
    n_lanes = wave_end - wave_start

    state = WaveState()
    state.exec_mask = (1 << n_lanes) - 1

    # setup kernel args pointer
    write_sgpr64(state, 0, args_ptr)

    # setup workgroup ID
    gx, gy, gz = workgroup_id
    if dispatch_dim >= 3: state.sgpr[13], state.sgpr[14], state.sgpr[15] = gx, gy, gz
    elif dispatch_dim == 2: state.sgpr[14], state.sgpr[15] = gx, gy
    else: state.sgpr[15] = gx

    # setup thread IDs in v0
    for i in range(n_lanes):
      tid = wave_start + i
      if local_size == (lx, 1, 1):
        state.vgpr[i][0] = tid
      else:
        x = tid % lx
        y = (tid // lx) % ly
        z = tid // (lx * ly)
        state.vgpr[i][0] = (z << 20) | (y << 10) | x

    waves.append((state, n_lanes))

  # execute waves (handle barrier)
  has_barrier = any(isinstance(inst, SOPP) and get_field(inst, 'op') == SOPPOp.S_BARRIER for inst, _ in program)
  iterations = 2 if has_barrier else 1

  for _ in range(iterations):
    for state, n_lanes in waves:
      result = exec_wave(program, state, lds, n_lanes)
      if result == 0: break  # program ended

# *** main entry point ***
def run_asm(lib: int, lib_sz: int, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, args_ptr: int) -> int:
  """Main entry point - matches remu signature."""
  # read kernel bytes from memory
  kernel_bytes = (ctypes.c_char * lib_sz).from_address(lib).raw

  # decode program
  program = decode_program(kernel_bytes)
  if not program:
    return -1

  # determine dispatch dimension
  dispatch_dim = 3 if gz > 1 else (2 if gy > 1 else 1)

  # execute all workgroups
  for gidz in range(gz):
    for gidy in range(gy):
      for gidx in range(gx):
        exec_workgroup(program, (gidx, gidy, gidz), (lx, ly, lz), args_ptr, dispatch_dim)

  return 0
