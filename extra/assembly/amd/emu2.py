# RDNA3 emulator v2 - compiles pcode to UOps executed via tinygrad CPU backend
# Each instruction is compiled to a kernel that operates on buffers:
#   arg=0: vmem - base address 0, INDEX offsets directly to host memory
#   arg=1: lds - local data share
#   arg=2: vgpr - vgpr[reg * 32 + lane]
#   arg=3: sgpr - sgpr[reg], PC_LO=128, PC_HI=129, SCC=130
from __future__ import annotations
import ctypes, functools
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.dtype import dtypes
from tinygrad.codegen import get_program
from tinygrad.engine.realize import CompiledRunner
from tinygrad.device import Device, Buffer, BufferSpec
from tinygrad.runtime.autogen import hsa
from tinygrad.helpers import Context, DEBUG, colored

from extra.assembly.amd.decode import decode_inst
from extra.assembly.amd.autogen.rdna3.str_pcode import PCODE
from extra.assembly.amd.autogen.rdna3.ins import (SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, VOP1, VOP2, VOP3, VOP3SD, VOP3P, VOPC, DS, FLAT, GLOBAL, VOPD,
  SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, SMEMOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp, DSOp, FLATOp, GLOBALOp, SCRATCHOp, VOPDOp)
from extra.assembly.amd.dsl import NULL, SCC, VCC_LO, VCC_HI, EXEC_LO, EXEC_HI

MASK32, MASK64 = 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF
WAVE_SIZE = 32
PC_LO_IDX, PC_HI_IDX, SCC_IDX = 128, 129, 130
SGPR_COUNT, VGPR_SIZE = 131, 256 * 32

# Counter for unique axis IDs to avoid UOp caching issues
_axis_id_counter = 0
def _next_axis_id() -> int:
  global _axis_id_counter
  _axis_id_counter += 1
  return _axis_id_counter

# Buffers: sgpr=0, vgpr=1, vmem=2, lds=3
def _define_bufs():
  sgpr = UOp(Ops.DEFINE_GLOBAL, dtypes.uint32.ptr(SGPR_COUNT), arg=0)
  vgpr = UOp(Ops.DEFINE_GLOBAL, dtypes.uint32.ptr(VGPR_SIZE), arg=1)
  vmem = UOp(Ops.DEFINE_GLOBAL, dtypes.uint32.ptr(1 << 46), arg=2)
  lds = UOp(Ops.DEFINE_GLOBAL, dtypes.uint32.ptr(16384), arg=3)
  return sgpr, vgpr, vmem, lds

def _sext(v, bits): return v - (1 << bits) if v & (1 << (bits - 1)) else v

# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION COMPILER - converts decoded instruction to UOp SINK
# ═══════════════════════════════════════════════════════════════════════════════

@functools.cache
def _compile_inst_inner(inst_bytes: bytes) -> tuple[str, UOp]:
  """Compile instruction bytes to (name, SINK UOp)."""
  inst = decode_inst(inst_bytes)
  name = f"emu2_{inst_bytes[:inst.size()].hex()}"
  sgpr, vgpr, vmem, lds = _define_bufs()
  inst_words = inst.size() // 4
  literal = int.from_bytes(inst_bytes[4:8], 'little') if len(inst_bytes) >= 8 else 0

  # Helper: read SGPR
  def rsgpr(reg: int) -> UOp: return sgpr.index(UOp.const(dtypes.index, reg))
  def rsgpr64(reg: int) -> UOp:
    lo, hi = rsgpr(reg), rsgpr(reg + 1)
    return lo.cast(dtypes.uint64) | (hi.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))
  def wsgpr(reg: int, val: UOp) -> UOp: return sgpr.index(UOp.const(dtypes.index, reg)).store(val.cast(dtypes.uint32))

  # Helper: read VGPR
  def rvgpr(reg: int, lane: UOp) -> UOp: return vgpr.index(UOp.const(dtypes.index, reg * 32) + lane.cast(dtypes.index))
  def wvgpr(reg: int, lane: UOp, val: UOp, exec_mask: UOp) -> UOp:
    idx = vgpr.index(UOp.const(dtypes.index, reg * 32) + lane.cast(dtypes.index))
    exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
    active = exec_bit.ne(UOp.const(dtypes.uint32, 0))
    return idx.store(active.where(val.cast(dtypes.uint32), idx))

  # Helper: read source operand
  # Float inline constants (as uint32 bit patterns)
  FLOAT_CONSTS = {240: 0x3f000000,  # 0.5
                  241: 0xbf000000,  # -0.5
                  242: 0x3f800000,  # 1.0
                  243: 0xbf800000,  # -1.0
                  244: 0x40000000,  # 2.0
                  245: 0xc0000000,  # -2.0
                  246: 0x40800000,  # 4.0
                  247: 0xc0800000,  # -4.0
                  248: 0x3e22f983}  # 1/(2*pi)
  def rsrc(off: int, lane: UOp) -> UOp:
    if off < 128: return rsgpr(off)
    if off == 253: return rsgpr(SCC_IDX)
    if off == 255: return UOp.const(dtypes.uint32, literal)
    if off < 255:  # inline constants
      if off < 193: return UOp.const(dtypes.uint32, off - 128)  # 0-64
      if off < 209: return UOp.const(dtypes.uint32, (-(off - 192)) & MASK32)  # -1 to -16
      if off in FLOAT_CONSTS: return UOp.const(dtypes.uint32, FLOAT_CONSTS[off])
      return UOp.const(dtypes.uint32, 0)  # other inline
    return rvgpr(off - 256, lane)

  # Helper: increment PC
  def inc_pc() -> UOp:
    pc = rsgpr(PC_LO_IDX)
    return wsgpr(PC_LO_IDX, pc + UOp.const(dtypes.uint32, inst_words))

  # ═══════════════════════════════════════════════════════════════════════════
  # SOPP: s_endpgm, s_waitcnt, s_clause, etc.
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, SOPP):
    if inst.op == SOPPOp.S_ENDPGM:
      return name, UOp.sink(wsgpr(PC_LO_IDX, UOp.const(dtypes.uint32, 0xFFFFFFFF)), arg=KernelInfo(name=name))
    return name, UOp.sink(inc_pc(), arg=KernelInfo(name=name))

  # ═══════════════════════════════════════════════════════════════════════════
  # SMEM: scalar memory loads
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, SMEM):
    addr = rsgpr64(inst.sbase.offset)
    offset = _sext(inst.offset, 21)
    addr = addr + UOp.const(dtypes.uint64, offset)
    sdata_reg = inst.sdata.offset

    # Determine how many dwords to load
    ndwords = {SMEMOp.S_LOAD_B32: 1, SMEMOp.S_LOAD_B64: 2, SMEMOp.S_LOAD_B128: 4, SMEMOp.S_LOAD_B256: 8, SMEMOp.S_LOAD_B512: 16}.get(inst.op, 1)

    stores = []
    for i in range(ndwords):
      byte_addr = addr + UOp.const(dtypes.uint64, i * 4)
      # vmem base is 0, INDEX directly to host address (divide by 4 for uint32 ptr)
      val = vmem.index((byte_addr >> UOp.const(dtypes.uint64, 2)).cast(dtypes.index))
      stores.append(wsgpr(sdata_reg + i, val))
    return name, UOp.sink(*stores, inc_pc(), arg=KernelInfo(name=name))

  # ═══════════════════════════════════════════════════════════════════════════
  # SOP1: scalar unary ops (s_mov_b32, etc.)
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, SOP1):
    s0 = rsrc(inst.ssrc0.offset, UOp.const(dtypes.index, 0))
    dst_reg = inst.sdst.offset
    op_name = inst.op.name

    if 'MOV_B32' in op_name:
      return name, UOp.sink(wsgpr(dst_reg, s0), inc_pc(), arg=KernelInfo(name=name))

    if 'MOV_B64' in op_name:
      s0_hi = rsrc(inst.ssrc0.offset + 1, UOp.const(dtypes.index, 0))
      return name, UOp.sink(wsgpr(dst_reg, s0), wsgpr(dst_reg + 1, s0_hi), inc_pc(), arg=KernelInfo(name=name))

    # SAVEEXEC ops: save EXEC to dst, then modify EXEC
    if 'OR_SAVEEXEC_B32' in op_name:
      exec_lo = rsgpr(EXEC_LO.offset)
      new_exec = exec_lo | s0
      return name, UOp.sink(wsgpr(dst_reg, exec_lo), wsgpr(EXEC_LO.offset, new_exec), inc_pc(), arg=KernelInfo(name=name))

    if 'AND_SAVEEXEC_B32' in op_name:
      exec_lo = rsgpr(EXEC_LO.offset)
      new_exec = exec_lo & s0
      return name, UOp.sink(wsgpr(dst_reg, exec_lo), wsgpr(EXEC_LO.offset, new_exec), inc_pc(), arg=KernelInfo(name=name))

    assert False, f"unimplemented SOP1: {op_name}"

  # ═══════════════════════════════════════════════════════════════════════════
  # SOP2: scalar ALU (s_add_i32, s_lshl_b64, etc.)
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, SOP2):
    s0 = rsrc(inst.ssrc0.offset, UOp.const(dtypes.index, 0))
    s1 = rsrc(inst.ssrc1.offset, UOp.const(dtypes.index, 0))
    dst_reg = inst.sdst.offset
    op_name = inst.op.name

    if inst.op == SOP2Op.S_ADD_I32:
      result = s0.cast(dtypes.int) + s1.cast(dtypes.int)
      # SCC = signed overflow
      s0_sign = (s0 >> UOp.const(dtypes.uint32, 31)) & UOp.const(dtypes.uint32, 1)
      s1_sign = (s1 >> UOp.const(dtypes.uint32, 31)) & UOp.const(dtypes.uint32, 1)
      r_sign = (result.cast(dtypes.uint32) >> UOp.const(dtypes.uint32, 31)) & UOp.const(dtypes.uint32, 1)
      scc = s0_sign.eq(s1_sign) & s0_sign.ne(r_sign)
      return name, UOp.sink(wsgpr(dst_reg, result.cast(dtypes.uint32)), wsgpr(SCC_IDX, scc.cast(dtypes.uint32)), inc_pc(), arg=KernelInfo(name=name))

    if inst.op == SOP2Op.S_ADD_U32:
      result = s0 + s1
      # SCC = carry out (result < s0 means overflow)
      scc = UOp(Ops.CMPLT, dtypes.bool, (result, s0))
      return name, UOp.sink(wsgpr(dst_reg, result), wsgpr(SCC_IDX, scc.cast(dtypes.uint32)), inc_pc(), arg=KernelInfo(name=name))

    if inst.op == SOP2Op.S_ADDC_U32:
      scc_in = rsgpr(SCC_IDX)
      result = s0 + s1 + scc_in
      # SCC = carry out
      scc = UOp(Ops.CMPLT, dtypes.bool, (result, s0)) | (result.eq(s0) & scc_in.ne(UOp.const(dtypes.uint32, 0)))
      return name, UOp.sink(wsgpr(dst_reg, result), wsgpr(SCC_IDX, scc.cast(dtypes.uint32)), inc_pc(), arg=KernelInfo(name=name))

    if 'LSHL_B32' in op_name:
      shift = s1 & UOp.const(dtypes.uint32, 31)
      result = s0 << shift
      return name, UOp.sink(wsgpr(dst_reg, result), inc_pc(), arg=KernelInfo(name=name))

    if 'LSHL_B64' in op_name:
      # 64-bit shift left
      s0_hi = rsrc(inst.ssrc0.offset + 1, UOp.const(dtypes.index, 0))
      s0_64 = s0.cast(dtypes.uint64) | (s0_hi.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))
      shift = s1 & UOp.const(dtypes.uint32, 63)
      result = s0_64 << shift.cast(dtypes.uint64)
      result_lo = result.cast(dtypes.uint32)
      result_hi = (result >> UOp.const(dtypes.uint64, 32)).cast(dtypes.uint32)
      return name, UOp.sink(wsgpr(dst_reg, result_lo), wsgpr(dst_reg + 1, result_hi), inc_pc(), arg=KernelInfo(name=name))

    if 'ASHR_I32' in op_name:
      shift = s1 & UOp.const(dtypes.uint32, 31)
      result = s0.cast(dtypes.int) >> shift.cast(dtypes.int)
      return name, UOp.sink(wsgpr(dst_reg, result.cast(dtypes.uint32)), inc_pc(), arg=KernelInfo(name=name))

    if 'ASHR_I64' in op_name:
      # 64-bit arithmetic shift right
      s0_hi = rsrc(inst.ssrc0.offset + 1, UOp.const(dtypes.index, 0))
      s0_64 = s0.cast(dtypes.uint64) | (s0_hi.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))
      shift = s1 & UOp.const(dtypes.uint32, 63)
      result = s0_64.bitcast(dtypes.int64) >> shift.cast(dtypes.int64)
      result_lo = result.bitcast(dtypes.uint64).cast(dtypes.uint32)
      result_hi = (result.bitcast(dtypes.uint64) >> UOp.const(dtypes.uint64, 32)).cast(dtypes.uint32)
      return name, UOp.sink(wsgpr(dst_reg, result_lo), wsgpr(dst_reg + 1, result_hi), inc_pc(), arg=KernelInfo(name=name))

    if 'XOR_B32' in op_name:
      result = s0 ^ s1
      return name, UOp.sink(wsgpr(dst_reg, result), inc_pc(), arg=KernelInfo(name=name))

    if 'AND_B32' in op_name:
      result = s0 & s1
      return name, UOp.sink(wsgpr(dst_reg, result), inc_pc(), arg=KernelInfo(name=name))

    if 'OR_B32' in op_name:
      result = s0 | s1
      return name, UOp.sink(wsgpr(dst_reg, result), inc_pc(), arg=KernelInfo(name=name))

    if 'CSELECT_B32' in op_name:
      # D0 = SCC ? S0 : S1
      scc = rsgpr(SCC_IDX)
      result = scc.ne(UOp.const(dtypes.uint32, 0)).where(s0, s1)
      return name, UOp.sink(wsgpr(dst_reg, result), inc_pc(), arg=KernelInfo(name=name))

    assert False, f"unimplemented SOP2: {op_name}"

  # ═══════════════════════════════════════════════════════════════════════════
  # VOP1: v_mov_b32, etc.
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOP1):
    lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    exec_mask = rsgpr(EXEC_LO.offset)
    src0 = rsrc(inst.src0.offset, lane)
    vdst_reg = inst.vdst.offset - 256

    if 'MOV_B32' in inst.op.name:
      store = wvgpr(vdst_reg, lane, src0, exec_mask)
      return name, UOp.sink(store.end(lane), inc_pc(), arg=KernelInfo(name=name))

    if 'EXP_F32' in inst.op.name:
      # D0.f32 = pow(2.0, S0.f32) = exp2(S0)
      result = UOp(Ops.EXP2, dtypes.float32, (src0.bitcast(dtypes.float32),)).bitcast(dtypes.uint32)
      store = wvgpr(vdst_reg, lane, result, exec_mask)
      return name, UOp.sink(store.end(lane), inc_pc(), arg=KernelInfo(name=name))

    assert False, f"unimplemented VOP1: {inst.op.name}"

  # ═══════════════════════════════════════════════════════════════════════════
  # VOPC: vector compare, writes to VCC (or EXEC for CMPX)
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOPC):
    lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    exec_mask = rsgpr(EXEC_LO.offset)
    src0 = rsrc(inst.src0.offset, lane)
    src1 = rsrc(inst.vsrc1.offset, lane)
    op_name = inst.op.name
    is_cmpx = 'CMPX' in op_name

    # Compute comparison result per lane based on op type
    cmp_result = None
    # Float compares
    if 'CMP_GT_F32' in op_name or 'CMPX_GT_F32' in op_name:
      cmp_result = src0.bitcast(dtypes.float32) > src1.bitcast(dtypes.float32)
    elif 'CMP_LT_F32' in op_name or 'CMPX_LT_F32' in op_name:
      cmp_result = src0.bitcast(dtypes.float32) < src1.bitcast(dtypes.float32)
    elif 'CMP_GE_F32' in op_name or 'CMPX_GE_F32' in op_name:
      cmp_result = src0.bitcast(dtypes.float32) >= src1.bitcast(dtypes.float32)
    elif 'CMP_LE_F32' in op_name or 'CMPX_LE_F32' in op_name:
      cmp_result = src0.bitcast(dtypes.float32) <= src1.bitcast(dtypes.float32)
    elif 'CMP_EQ_F32' in op_name or 'CMPX_EQ_F32' in op_name:
      cmp_result = src0.bitcast(dtypes.float32).eq(src1.bitcast(dtypes.float32))
    elif 'CMP_NE_F32' in op_name or 'CMP_NEQ_F32' in op_name or 'CMPX_NE_F32' in op_name or 'CMPX_NEQ_F32' in op_name:
      cmp_result = src0.bitcast(dtypes.float32).ne(src1.bitcast(dtypes.float32))
    # Unsigned int compares
    elif 'CMP_LT_U32' in op_name or 'CMPX_LT_U32' in op_name:
      cmp_result = src0 < src1
    elif 'CMP_GT_U32' in op_name or 'CMPX_GT_U32' in op_name:
      cmp_result = src0 > src1
    elif 'CMP_LE_U32' in op_name or 'CMPX_LE_U32' in op_name:
      cmp_result = src0 <= src1
    elif 'CMP_GE_U32' in op_name or 'CMPX_GE_U32' in op_name:
      cmp_result = src0 >= src1
    elif 'CMP_EQ_U32' in op_name or 'CMPX_EQ_U32' in op_name:
      cmp_result = src0.eq(src1)
    elif 'CMP_NE_U32' in op_name or 'CMPX_NE_U32' in op_name:
      cmp_result = src0.ne(src1)
    # Signed int compares
    elif 'CMP_LT_I32' in op_name or 'CMPX_LT_I32' in op_name:
      cmp_result = src0.bitcast(dtypes.int) < src1.bitcast(dtypes.int)
    elif 'CMP_GT_I32' in op_name or 'CMPX_GT_I32' in op_name:
      cmp_result = src0.bitcast(dtypes.int) > src1.bitcast(dtypes.int)
    elif 'CMP_LE_I32' in op_name or 'CMPX_LE_I32' in op_name:
      cmp_result = src0.bitcast(dtypes.int) <= src1.bitcast(dtypes.int)
    elif 'CMP_GE_I32' in op_name or 'CMPX_GE_I32' in op_name:
      cmp_result = src0.bitcast(dtypes.int) >= src1.bitcast(dtypes.int)
    elif 'CMP_EQ_I32' in op_name or 'CMPX_EQ_I32' in op_name:
      cmp_result = src0.bitcast(dtypes.int).eq(src1.bitcast(dtypes.int))
    elif 'CMP_NE_I32' in op_name or 'CMPX_NE_I32' in op_name:
      cmp_result = src0.bitcast(dtypes.int).ne(src1.bitcast(dtypes.int))
    else:
      assert False, f"unimplemented VOPC: {op_name}"

    # Determine destination register (EXEC for CMPX, VCC otherwise)
    dst_reg = EXEC_LO.offset if is_cmpx else VCC_LO.offset
    dst = rsgpr(dst_reg)
    dst_mask = UOp.const(dtypes.uint32, 1) << lane.cast(dtypes.uint32)
    exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
    cmp_bit = cmp_result.cast(dtypes.uint32) << lane.cast(dtypes.uint32)
    dst_cleared = dst & (dst_mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))
    dst_new = exec_bit.ne(UOp.const(dtypes.uint32, 0)).where(dst_cleared | cmp_bit, dst)
    dst_store = wsgpr(dst_reg, dst_new)
    return name, UOp.sink(dst_store.end(lane), inc_pc(), arg=KernelInfo(name=name))

  # ═══════════════════════════════════════════════════════════════════════════
  # VOP2: v_add_f32, v_lshlrev_b32, etc.
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOP2):
    lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    exec_mask = rsgpr(EXEC_LO.offset)
    src0 = rsrc(inst.src0.offset, lane)
    src1 = rsrc(inst.vsrc1.offset, lane)
    vdst_reg = inst.vdst.offset - 256
    op_name = inst.op.name

    if 'ADD_F32' in op_name:
      result = (src0.bitcast(dtypes.float32) + src1.bitcast(dtypes.float32)).bitcast(dtypes.uint32)
      store = wvgpr(vdst_reg, lane, result, exec_mask)
      return name, UOp.sink(store.end(lane), inc_pc(), arg=KernelInfo(name=name))

    if 'SUB_F32' in op_name:
      result = (src0.bitcast(dtypes.float32) - src1.bitcast(dtypes.float32)).bitcast(dtypes.uint32)
      store = wvgpr(vdst_reg, lane, result, exec_mask)
      return name, UOp.sink(store.end(lane), inc_pc(), arg=KernelInfo(name=name))

    if 'MUL_F32' in op_name:
      result = (src0.bitcast(dtypes.float32) * src1.bitcast(dtypes.float32)).bitcast(dtypes.uint32)
      store = wvgpr(vdst_reg, lane, result, exec_mask)
      return name, UOp.sink(store.end(lane), inc_pc(), arg=KernelInfo(name=name))

    if 'ADD_NC_U32' in op_name or ('ADD_U32' in op_name and 'CO' not in op_name):
      result = src0 + src1
      store = wvgpr(vdst_reg, lane, result, exec_mask)
      return name, UOp.sink(store.end(lane), inc_pc(), arg=KernelInfo(name=name))

    if 'LSHLREV_B32' in op_name:
      shift = src0 & UOp.const(dtypes.uint32, 31)
      result = src1 << shift
      store = wvgpr(vdst_reg, lane, result, exec_mask)
      return name, UOp.sink(store.end(lane), inc_pc(), arg=KernelInfo(name=name))

    if 'MOV_B32' in op_name:
      store = wvgpr(vdst_reg, lane, src0, exec_mask)
      return name, UOp.sink(store.end(lane), inc_pc(), arg=KernelInfo(name=name))

    if 'ADD_CO_CI_U32' in op_name:
      # Add with carry-in from VCC and carry-out to VCC
      # tmp = src0 + src1 + VCC[lane]; VCC[lane] = tmp >= 0x100000000; D0 = tmp.u32
      vcc = rsgpr(VCC_LO.offset)
      carry_in = (vcc >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
      sum64 = src0.cast(dtypes.uint64) + src1.cast(dtypes.uint64) + carry_in.cast(dtypes.uint64)
      result = sum64.cast(dtypes.uint32)
      carry_out = (sum64 >> UOp.const(dtypes.uint64, 32)).cast(dtypes.uint32) & UOp.const(dtypes.uint32, 1)
      # Update VCC bit for this lane: clear bit then set if carry
      vcc_mask = UOp.const(dtypes.uint32, 1) << lane.cast(dtypes.uint32)
      vcc_cleared = vcc & (vcc_mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))
      vcc_new = vcc_cleared | (carry_out << lane.cast(dtypes.uint32))
      store = wvgpr(vdst_reg, lane, result, exec_mask)
      vcc_store = wsgpr(VCC_LO.offset, vcc_new)
      return name, UOp.sink(store.end(lane), vcc_store.end(lane), inc_pc(), arg=KernelInfo(name=name))

    assert False, f"unimplemented VOP2: {op_name}"

  # ═══════════════════════════════════════════════════════════════════════════
  # VOP3: 3-operand vector ALU (v_add_f32_e64, v_fma_f32, etc.)
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOP3):
    lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    exec_mask = rsgpr(EXEC_LO.offset)
    src0 = rsrc(inst.src0.offset, lane)
    src1 = rsrc(inst.src1.offset, lane)
    src2 = rsrc(inst.src2.offset, lane) if inst.src2 is not None else None
    vdst_reg = inst.vdst.offset - 256
    op_name = inst.op.name

    if 'ADD_F32' in op_name:
      result = (src0.bitcast(dtypes.float32) + src1.bitcast(dtypes.float32)).bitcast(dtypes.uint32)
      store = wvgpr(vdst_reg, lane, result, exec_mask)
      return name, UOp.sink(store.end(lane), inc_pc(), arg=KernelInfo(name=name))

    if 'MUL_F32' in op_name:
      result = (src0.bitcast(dtypes.float32) * src1.bitcast(dtypes.float32)).bitcast(dtypes.uint32)
      store = wvgpr(vdst_reg, lane, result, exec_mask)
      return name, UOp.sink(store.end(lane), inc_pc(), arg=KernelInfo(name=name))

    if 'FMA_F32' in op_name and src2 is not None:
      result = (src0.bitcast(dtypes.float32) * src1.bitcast(dtypes.float32) + src2.bitcast(dtypes.float32)).bitcast(dtypes.uint32)
      store = wvgpr(vdst_reg, lane, result, exec_mask)
      return name, UOp.sink(store.end(lane), inc_pc(), arg=KernelInfo(name=name))

    if 'MOV_B32' in op_name:
      store = wvgpr(vdst_reg, lane, src0, exec_mask)
      return name, UOp.sink(store.end(lane), inc_pc(), arg=KernelInfo(name=name))

    if 'LSHLREV_B64' in op_name:
      # 64-bit shift: dst = src1 << src0
      src1_hi = rsrc(inst.src1.offset + 1, lane) if inst.src1.offset < 256 else rvgpr(inst.src1.offset - 256 + 1, lane)
      src1_64 = src1.cast(dtypes.uint64) | (src1_hi.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))
      shift = src0 & UOp.const(dtypes.uint32, 63)
      result = src1_64 << shift.cast(dtypes.uint64)
      result_lo = result.cast(dtypes.uint32)
      result_hi = (result >> UOp.const(dtypes.uint64, 32)).cast(dtypes.uint32)
      # Need two separate loops for the two stores
      store_lo = wvgpr(vdst_reg, lane, result_lo, exec_mask)
      lane2 = UOp.range(32, _next_axis_id(), AxisType.LOOP)
      src1_2 = rsrc(inst.src1.offset, lane2)
      src1_hi_2 = rsrc(inst.src1.offset + 1, lane2) if inst.src1.offset < 256 else rvgpr(inst.src1.offset - 256 + 1, lane2)
      src1_64_2 = src1_2.cast(dtypes.uint64) | (src1_hi_2.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))
      src0_2 = rsrc(inst.src0.offset, lane2)
      shift_2 = src0_2 & UOp.const(dtypes.uint32, 63)
      result_2 = src1_64_2 << shift_2.cast(dtypes.uint64)
      result_hi_2 = (result_2 >> UOp.const(dtypes.uint64, 32)).cast(dtypes.uint32)
      store_hi = wvgpr(vdst_reg + 1, lane2, result_hi_2, exec_mask)
      return name, UOp.sink(store_lo.end(lane), store_hi.end(lane2), inc_pc(), arg=KernelInfo(name=name))

    if 'CNDMASK_B32' in op_name and src2 is not None:
      # D0 = mask[lane] ? S1 : S0  (src2 is the mask register, typically VCC)
      mask_bit = (src2 >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
      result = mask_bit.ne(UOp.const(dtypes.uint32, 0)).where(src1, src0)
      store = wvgpr(vdst_reg, lane, result, exec_mask)
      return name, UOp.sink(store.end(lane), inc_pc(), arg=KernelInfo(name=name))

    # VOP3 compares - write to scalar register (VCC_LO for VOP3_SDST)
    if 'CMP_' in op_name and '_F32' in op_name:
      s0f = src0.bitcast(dtypes.float32)
      s1f = src1.bitcast(dtypes.float32)
      if 'CMP_GT_F32' in op_name: cmp_result = s0f > s1f
      elif 'CMP_LT_F32' in op_name: cmp_result = s0f < s1f
      elif 'CMP_GE_F32' in op_name: cmp_result = s0f >= s1f
      elif 'CMP_LE_F32' in op_name: cmp_result = s0f <= s1f
      elif 'CMP_EQ_F32' in op_name: cmp_result = s0f.eq(s1f)
      elif 'CMP_NE_F32' in op_name or 'CMP_NEQ_F32' in op_name or 'CMP_LG_F32' in op_name: cmp_result = s0f.ne(s1f)
      else: assert False, f"unimplemented VOP3 compare: {op_name}"
      # Write to sdst (vdst in VOP3_SDST is the SGPR destination)
      sdst_reg = inst.vdst.offset  # For VOP3_SDST, vdst is an SGPR offset
      sdst = rsgpr(sdst_reg)
      sdst_mask = UOp.const(dtypes.uint32, 1) << lane.cast(dtypes.uint32)
      exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
      cmp_bit = cmp_result.cast(dtypes.uint32) << lane.cast(dtypes.uint32)
      sdst_cleared = sdst & (sdst_mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))
      sdst_new = exec_bit.ne(UOp.const(dtypes.uint32, 0)).where(sdst_cleared | cmp_bit, sdst)
      sdst_store = wsgpr(sdst_reg, sdst_new)
      return name, UOp.sink(sdst_store.end(lane), inc_pc(), arg=KernelInfo(name=name))

    if 'LDEXP_F32' in op_name:
      # D0.f32 = S0.f32 * 2^S1.i32
      s0f = src0.bitcast(dtypes.float32)
      s1i = src1.bitcast(dtypes.int)
      # ldexp(x, n) = x * 2^n
      two_pow_n = UOp(Ops.EXP2, dtypes.float32, (s1i.cast(dtypes.float32),))
      result = (s0f * two_pow_n).bitcast(dtypes.uint32)
      store = wvgpr(vdst_reg, lane, result, exec_mask)
      return name, UOp.sink(store.end(lane), inc_pc(), arg=KernelInfo(name=name))

    assert False, f"unimplemented VOP3: {op_name}"

  # ═══════════════════════════════════════════════════════════════════════════
  # VOPD: dual-issue v_dual_mov_b32, etc.
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOPD):
    exec_mask = rsgpr(EXEC_LO.offset)
    vdstx_reg = inst.vdstx.offset - 256
    vdsty_reg = (inst.vdsty << 1) | ((inst.vdstx.offset & 1) ^ 1)
    opx_name = inst.opx.name if hasattr(inst.opx, 'name') else str(inst.opx)
    opy_name = inst.opy.name if hasattr(inst.opy, 'name') else str(inst.opy)

    ended = []

    # X operation - separate loop
    lane_x = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    srcx0 = rsrc(inst.srcx0.offset, lane_x)
    vsrcx1 = rvgpr(inst.vsrcx1.offset - 256, lane_x)
    if 'MOV_B32' in opx_name:
      ended.append(wvgpr(vdstx_reg, lane_x, srcx0, exec_mask).end(lane_x))
    elif 'ADD_F32' in opx_name:
      result = (srcx0.bitcast(dtypes.float32) + vsrcx1.bitcast(dtypes.float32)).bitcast(dtypes.uint32)
      ended.append(wvgpr(vdstx_reg, lane_x, result, exec_mask).end(lane_x))
    elif 'SUB_F32' in opx_name:
      result = (srcx0.bitcast(dtypes.float32) - vsrcx1.bitcast(dtypes.float32)).bitcast(dtypes.uint32)
      ended.append(wvgpr(vdstx_reg, lane_x, result, exec_mask).end(lane_x))
    elif 'MUL_F32' in opx_name:
      result = (srcx0.bitcast(dtypes.float32) * vsrcx1.bitcast(dtypes.float32)).bitcast(dtypes.uint32)
      ended.append(wvgpr(vdstx_reg, lane_x, result, exec_mask).end(lane_x))
    elif 'MAX_F32' in opx_name:
      result = UOp(Ops.MAX, dtypes.float32, (srcx0.bitcast(dtypes.float32), vsrcx1.bitcast(dtypes.float32))).bitcast(dtypes.uint32)
      ended.append(wvgpr(vdstx_reg, lane_x, result, exec_mask).end(lane_x))
    elif 'ADD_NC_U32' in opx_name:
      ended.append(wvgpr(vdstx_reg, lane_x, srcx0 + vsrcx1, exec_mask).end(lane_x))
    elif 'LSHLREV_B32' in opx_name:
      shift = srcx0 & UOp.const(dtypes.uint32, 31)
      ended.append(wvgpr(vdstx_reg, lane_x, vsrcx1 << shift, exec_mask).end(lane_x))

    # Y operation - separate loop
    lane_y = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    srcy0 = rsrc(inst.srcy0.offset, lane_y)
    vsrcy1 = rvgpr(inst.vsrcy1.offset - 256, lane_y)
    if 'MOV_B32' in opy_name:
      ended.append(wvgpr(vdsty_reg, lane_y, srcy0, exec_mask).end(lane_y))
    elif 'ADD_F32' in opy_name:
      result = (srcy0.bitcast(dtypes.float32) + vsrcy1.bitcast(dtypes.float32)).bitcast(dtypes.uint32)
      ended.append(wvgpr(vdsty_reg, lane_y, result, exec_mask).end(lane_y))
    elif 'SUB_F32' in opy_name:
      result = (srcy0.bitcast(dtypes.float32) - vsrcy1.bitcast(dtypes.float32)).bitcast(dtypes.uint32)
      ended.append(wvgpr(vdsty_reg, lane_y, result, exec_mask).end(lane_y))
    elif 'MUL_F32' in opy_name:
      result = (srcy0.bitcast(dtypes.float32) * vsrcy1.bitcast(dtypes.float32)).bitcast(dtypes.uint32)
      ended.append(wvgpr(vdsty_reg, lane_y, result, exec_mask).end(lane_y))
    elif 'MAX_F32' in opy_name:
      result = UOp(Ops.MAX, dtypes.float32, (srcy0.bitcast(dtypes.float32), vsrcy1.bitcast(dtypes.float32))).bitcast(dtypes.uint32)
      ended.append(wvgpr(vdsty_reg, lane_y, result, exec_mask).end(lane_y))
    elif 'ADD_NC_U32' in opy_name:
      ended.append(wvgpr(vdsty_reg, lane_y, srcy0 + vsrcy1, exec_mask).end(lane_y))
    elif 'LSHLREV_B32' in opy_name:
      shift = srcy0 & UOp.const(dtypes.uint32, 31)
      ended.append(wvgpr(vdsty_reg, lane_y, vsrcy1 << shift, exec_mask).end(lane_y))

    assert len(ended) == 2, f"unimplemented VOPD: X={opx_name} Y={opy_name}"
    return name, UOp.sink(*ended, inc_pc(), arg=KernelInfo(name=name))

  # ═══════════════════════════════════════════════════════════════════════════
  # FLAT/GLOBAL: memory loads/stores
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, (FLAT, GLOBAL)):
    exec_mask = rsgpr(EXEC_LO.offset)
    addr_reg = inst.addr.offset - 256
    has_saddr = hasattr(inst, 'saddr') and inst.saddr != NULL and inst.saddr.offset < 128
    offset = _sext(getattr(inst, 'offset', 0), 13)
    op_name = inst.op.name if hasattr(inst.op, 'name') else str(inst.op)
    ndwords = 4 if '_B128' in op_name else 3 if '_B96' in op_name else 2 if '_B64' in op_name else 1

    # Helper to compute address for a lane
    def make_addr(lane: UOp) -> UOp:
      if has_saddr:
        vgpr_offset = rvgpr(addr_reg, lane).cast(dtypes.uint64)
        saddr = rsgpr64(inst.saddr.offset)
        return saddr + vgpr_offset + UOp.const(dtypes.uint64, offset)
      else:
        addr_lo = rvgpr(addr_reg, lane)
        addr_hi = rvgpr(addr_reg + 1, lane)
        return (addr_lo.cast(dtypes.uint64) | (addr_hi.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))) + UOp.const(dtypes.uint64, offset)

    if 'LOAD' in op_name:
      vdst_reg = inst.vdst.offset - 256
      ended = []
      # Each dword gets its own loop to avoid CFG issues
      for i in range(ndwords):
        lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
        addr = make_addr(lane)
        byte_addr = addr + UOp.const(dtypes.uint64, i * 4)
        val = vmem.index((byte_addr >> UOp.const(dtypes.uint64, 2)).cast(dtypes.index))
        store = wvgpr(vdst_reg + i, lane, val, exec_mask)
        ended.append(store.end(lane))
      return name, UOp.sink(*ended, inc_pc(), arg=KernelInfo(name=name))

    if 'STORE' in op_name:
      vdata_reg = inst.data.offset - 256
      ended = []
      # Each dword gets its own loop to avoid CFG issues
      for i in range(ndwords):
        lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
        addr = make_addr(lane)
        byte_addr = addr + UOp.const(dtypes.uint64, i * 4)
        val = rvgpr(vdata_reg + i, lane)
        idx = vmem.index((byte_addr >> UOp.const(dtypes.uint64, 2)).cast(dtypes.index))
        exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
        active = exec_bit.ne(UOp.const(dtypes.uint32, 0))
        ended.append(idx.store(active.where(val, idx)).end(lane))
      return name, UOp.sink(*ended, inc_pc(), arg=KernelInfo(name=name))

    return name, UOp.sink(inc_pc(), arg=KernelInfo(name=name))

  # Default: just increment PC
  return name, UOp.sink(inc_pc(), arg=KernelInfo(name=name))

def compile_inst(data: bytes) -> tuple[str, UOp]:
  inst = decode_inst(data)
  return _compile_inst_inner(bytes(data[:inst.size() + 4]))

# ═══════════════════════════════════════════════════════════════════════════════
# PROGRAM DECODE AND COMPILATION
# ═══════════════════════════════════════════════════════════════════════════════

def decode_program(data: bytes) -> dict[int, tuple[str, CompiledRunner|None, list[int]|None]]:
  """Decode program to {pc: (name, runner, globals)}."""
  result = {}
  renderer = Device['CPU'].renderer
  i = 0
  while i < len(data):
    inst = decode_inst(data[i:])
    if isinstance(inst, SOPP) and inst.op == SOPPOp.S_CODE_END: break

    name, sink = compile_inst(bytes(data[i:i + inst.size() + 4]))
    try:
      with Context(NOOPT=1):
        prg = get_program(sink, renderer)
      runner = CompiledRunner(prg)
      globals_list = prg.globals
      if DEBUG >= 2: print(f"[emu2] PC={i//4}: {repr(inst)}\n{colored(prg.src, 'BLACK')}")
    except Exception as e:
      if DEBUG >= 1: print(f"[emu2] Failed to compile PC={i//4} {repr(inst)}: {type(e).__name__}: {e}")
      if DEBUG >= 3:
        import traceback
        traceback.print_exc()
      runner, globals_list = None, None

    result[i // 4] = (name, runner, globals_list)
    i += inst.size()
  return result

# ═══════════════════════════════════════════════════════════════════════════════
# WAVE STATE
# ═══════════════════════════════════════════════════════════════════════════════

class WaveState:
  __slots__ = ('vgpr_buf', 'sgpr_buf', '_vgpr_mv', '_sgpr_mv', 'n_lanes')

  def __init__(self, n_lanes: int = WAVE_SIZE):
    self.n_lanes = n_lanes
    self.vgpr_buf = Buffer('CPU', VGPR_SIZE, dtypes.uint32).ensure_allocated()
    self.sgpr_buf = Buffer('CPU', SGPR_COUNT, dtypes.uint32).ensure_allocated()
    self._vgpr_mv = self.vgpr_buf.as_buffer(force_zero_copy=True).cast('I')
    self._sgpr_mv = self.sgpr_buf.as_buffer(force_zero_copy=True).cast('I')
    self._write_sgpr(EXEC_LO.offset, (1 << n_lanes) - 1)
    self._write_sgpr(PC_LO_IDX, 0)

  def _write_sgpr(self, idx: int, val: int): self._sgpr_mv[idx] = val & MASK32
  def _read_sgpr(self, idx: int) -> int: return self._sgpr_mv[idx]
  def _write_vgpr(self, reg: int, lane: int, val: int): self._vgpr_mv[reg * 32 + lane] = val & MASK32
  def _read_vgpr(self, reg: int, lane: int) -> int: return self._vgpr_mv[reg * 32 + lane]

  @property
  def pc(self) -> int: return self._read_sgpr(PC_LO_IDX)
  @pc.setter
  def pc(self, val: int): self._write_sgpr(PC_LO_IDX, val)

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

# Global vmem buffer: external_ptr=0 means INDEX offsets directly to host memory addresses
_vmem_buf: Buffer | None = None
def _get_vmem_buf() -> Buffer:
  global _vmem_buf
  if _vmem_buf is None:
    _vmem_buf = Buffer('CPU', 1 << 40, dtypes.uint32, options=BufferSpec(external_ptr=0)).ensure_allocated()
  return _vmem_buf

def run_asm(lib: int, lib_sz: int, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, args_ptr: int, rsrc2: int = 0x19c) -> int:
  """Execute AMD assembly program."""
  data = bytes((ctypes.c_char * lib_sz).from_address(lib).raw)
  program = decode_program(data)

  lds_size = ((rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_SHIFT) * 512
  total_threads = lx * ly * lz

  vmem_buf = _get_vmem_buf()
  lds_buf = Buffer('CPU', max(lds_size // 4, 1), dtypes.uint32).ensure_allocated()

  for gidz in range(gz):
    for gidy in range(gy):
      for gidx in range(gx):
        for wave_start in range(0, total_threads, WAVE_SIZE):
          n_lanes = min(WAVE_SIZE, total_threads - wave_start)
          st = WaveState(n_lanes)

          # s[0:1] = kernel args pointer
          st._write_sgpr(0, args_ptr & MASK32)
          st._write_sgpr(1, (args_ptr >> 32) & MASK32)

          # Workgroup IDs
          sgpr_idx = (rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT
          if rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X: st._write_sgpr(sgpr_idx, gidx); sgpr_idx += 1
          if rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y: st._write_sgpr(sgpr_idx, gidy); sgpr_idx += 1
          if rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z: st._write_sgpr(sgpr_idx, gidz)

          # v0 = packed workitem IDs
          for tid in range(wave_start, wave_start + n_lanes):
            lane = tid - wave_start
            z, y, x = tid // (lx * ly), (tid // lx) % ly, tid % lx
            st._write_vgpr(0, lane, (z << 20) | (y << 10) | x)

          # Execute wave: sgpr=0, vgpr=1, vmem=2, lds=3
          all_bufs = {0: st.sgpr_buf, 1: st.vgpr_buf, 2: vmem_buf, 3: lds_buf}
          while True:
            pc = st.pc
            if pc == 0xFFFFFFFF or pc not in program: break

            name, runner, globals_list = program[pc]
            if runner is None:
              if DEBUG >= 1: print(f"[emu2] No runner for {name} at PC={pc}")
              break

            bufs = [all_bufs[g] for g in globals_list]
            runner(bufs, {}, wait=True)

  return 0
