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
from extra.assembly.amd.emu2_pcode import parse_pcode

MASK32, MASK64 = 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF

# Map VOPD ops to VOP2 ops for pcode lookup
VOPD_TO_VOP2 = {
  VOPDOp.V_DUAL_FMAC_F32: VOP2Op.V_FMAC_F32_E32, VOPDOp.V_DUAL_MUL_F32: VOP2Op.V_MUL_F32_E32,
  VOPDOp.V_DUAL_ADD_F32: VOP2Op.V_ADD_F32_E32, VOPDOp.V_DUAL_SUB_F32: VOP2Op.V_SUB_F32_E32,
  VOPDOp.V_DUAL_SUBREV_F32: VOP2Op.V_SUBREV_F32_E32, VOPDOp.V_DUAL_MAX_F32: VOP2Op.V_MAX_F32_E32,
  VOPDOp.V_DUAL_MIN_F32: VOP2Op.V_MIN_F32_E32, VOPDOp.V_DUAL_ADD_NC_U32: VOP2Op.V_ADD_NC_U32_E32,
  VOPDOp.V_DUAL_LSHLREV_B32: VOP2Op.V_LSHLREV_B32_E32, VOPDOp.V_DUAL_AND_B32: VOP2Op.V_AND_B32_E32,
  VOPDOp.V_DUAL_MOV_B32: VOP1Op.V_MOV_B32_E32, VOPDOp.V_DUAL_CNDMASK_B32: VOP2Op.V_CNDMASK_B32_E32,
}
WAVE_SIZE = 32
PC_LO_IDX, PC_HI_IDX, SCC_IDX = 128, 129, 130
SGPR_COUNT, VGPR_SIZE = 131, 256 * 32

# Counter for unique axis IDs to avoid UOp caching issues
_axis_id_counter = 0
def _next_axis_id() -> int:
  global _axis_id_counter
  _axis_id_counter += 1
  return _axis_id_counter

def compile_sop_pcode(op, srcs: dict[str, UOp], wsgpr_fn, rsgpr_fn, sdst_reg: int, sdst_size: int, inc_pc_fn, name: str):
  """Compile a scalar instruction using pcode parser. Returns (name, sink) or None."""
  pcode = PCODE.get(op)
  if pcode is None: return None

  srcs['VCC'] = rsgpr_fn(VCC_LO.offset)
  srcs['EXEC'] = rsgpr_fn(EXEC_LO.offset)
  srcs['SCC'] = rsgpr_fn(SCC_IDX)
  _, assigns = parse_pcode(pcode, srcs, lane=None)

  stores = []
  for dest, val in assigns:
    if dest.startswith('D0'):
      # Scalar destination - write to sdst (handle 32-bit and 64-bit)
      if sdst_size == 2:
        # 64-bit destination: write lo and hi
        val64 = val.cast(dtypes.uint64) if val.dtype != dtypes.uint64 else val
        stores.append(wsgpr_fn(sdst_reg, val64.cast(dtypes.uint32)))
        stores.append(wsgpr_fn(sdst_reg + 1, (val64 >> UOp.const(dtypes.uint64, 32)).cast(dtypes.uint32)))
      else:
        if val.dtype != dtypes.uint32: val = val.cast(dtypes.uint32)
        stores.append(wsgpr_fn(sdst_reg, val))
    elif dest.startswith('SCC'):
      stores.append(wsgpr_fn(SCC_IDX, val.cast(dtypes.uint32)))

  if not stores: return None
  return name, UOp.sink(*stores, inc_pc_fn(), arg=KernelInfo(name=name))

def compile_vop_pcode_stores(op, srcs: dict[str, UOp], lane: UOp, wvgpr_fn, wsgpr_fn, rsgpr_fn, vdst_reg: int, exec_mask: UOp) -> list[UOp] | None:
  """Compile a VOP instruction using pcode parser. Returns list of store UOps or None."""
  pcode = PCODE.get(op)
  if pcode is None: return None

  # Manual implementations for instructions with complex pcode (if/elsif/else)
  op_name = op.name if hasattr(op, 'name') else str(op)
  if 'MAX_F32' in op_name or 'MIN_F32' in op_name:
    s0 = srcs['S0'].bitcast(dtypes.float32) if srcs['S0'].dtype == dtypes.uint32 else srcs['S0']
    s1 = srcs['S1'].bitcast(dtypes.float32) if srcs['S1'].dtype == dtypes.uint32 else srcs['S1']
    result = UOp(Ops.MAX, dtypes.float32, (s0, s1)) if 'MAX' in op_name else UOp(Ops.MIN, dtypes.float32, (s0, s1))
    return [wvgpr_fn(vdst_reg, lane, result.bitcast(dtypes.uint32), exec_mask).end(lane)]

  # Parse pcode with source operands
  srcs['VCC'] = rsgpr_fn(VCC_LO.offset)
  srcs['EXEC'] = exec_mask
  srcs['SCC'] = rsgpr_fn(SCC_IDX)
  _, assigns = parse_pcode(pcode, srcs, lane)

  stores = []
  for dest, val in assigns:
    if 'D0' in dest and '[laneId]' in dest:
      # D0.u64[laneId] means VCC bit write (for VOPC instructions)
      vcc = rsgpr_fn(VCC_LO.offset)
      vcc_mask = UOp.const(dtypes.uint32, 1) << lane.cast(dtypes.uint32)
      exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
      vcc_bit = val.cast(dtypes.uint32) << lane.cast(dtypes.uint32)
      vcc_cleared = vcc & (vcc_mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))
      vcc_new = exec_bit.ne(UOp.const(dtypes.uint32, 0)).where(vcc_cleared | vcc_bit, vcc)
      stores.append(wsgpr_fn(VCC_LO.offset, vcc_new).end(lane))
    elif dest.startswith('D0'):
      # Vector destination - write to vdst (bitcast floats to preserve bit pattern)
      if val.dtype in (dtypes.float32, dtypes.float64):
        result = val.bitcast(dtypes.uint32)
      elif val.dtype != dtypes.uint32:
        result = val.cast(dtypes.uint32)
      else:
        result = val
      stores.append(wvgpr_fn(vdst_reg, lane, result, exec_mask).end(lane))
    elif dest.startswith('VCC'):
      # VCC update - need to set bit for this lane
      vcc = rsgpr_fn(VCC_LO.offset)
      vcc_mask = UOp.const(dtypes.uint32, 1) << lane.cast(dtypes.uint32)
      exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
      vcc_bit = val.cast(dtypes.uint32) << lane.cast(dtypes.uint32)
      vcc_cleared = vcc & (vcc_mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))
      vcc_new = exec_bit.ne(UOp.const(dtypes.uint32, 0)).where(vcc_cleared | vcc_bit, vcc)
      stores.append(wsgpr_fn(VCC_LO.offset, vcc_new).end(lane))
    elif dest.startswith('EXEC'):
      # EXEC update (for CMPX)
      exec_val = rsgpr_fn(EXEC_LO.offset)
      exec_mask_bit = UOp.const(dtypes.uint32, 1) << lane.cast(dtypes.uint32)
      exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
      exec_new_bit = val.cast(dtypes.uint32) << lane.cast(dtypes.uint32)
      exec_cleared = exec_val & (exec_mask_bit ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))
      exec_new = exec_bit.ne(UOp.const(dtypes.uint32, 0)).where(exec_cleared | exec_new_bit, exec_val)
      stores.append(wsgpr_fn(EXEC_LO.offset, exec_new).end(lane))
    elif dest.startswith('SCC'):
      stores.append(wsgpr_fn(SCC_IDX, val.cast(dtypes.uint32)))

  return stores if stores else None

def compile_vop_pcode(op, srcs: dict[str, UOp], lane: UOp, wvgpr_fn, wsgpr_fn, rsgpr_fn, vdst_reg: int, exec_mask: UOp, inc_pc_fn, name: str):
  """Try to compile a VOP instruction using pcode parser. Returns (name, sink) or None."""
  stores = compile_vop_pcode_stores(op, srcs, lane, wvgpr_fn, wsgpr_fn, rsgpr_fn, vdst_reg, exec_mask)
  if stores is None: return None
  return name, UOp.sink(*stores, inc_pc_fn(), arg=KernelInfo(name=name))

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
    sizes = inst.op_regs
    s0 = rsgpr64(inst.ssrc0.offset) if sizes.get('ssrc0', 1) == 2 else rsrc(inst.ssrc0.offset, UOp.const(dtypes.index, 0))
    s1 = rsgpr64(inst.ssrc1.offset) if sizes.get('ssrc1', 1) == 2 else rsrc(inst.ssrc1.offset, UOp.const(dtypes.index, 0))
    dst_reg = inst.sdst.offset
    dst_size = sizes.get('sdst', 1)
    pcode_result = compile_sop_pcode(inst.op, {'S0': s0, 'S1': s1}, wsgpr, rsgpr, dst_reg, dst_size, inc_pc, name)
    assert pcode_result is not None, f"no pcode for SOP2: {inst.op.name}"
    return pcode_result

  # ═══════════════════════════════════════════════════════════════════════════
  # VOP1: v_mov_b32, etc.
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOP1):
    lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    exec_mask = rsgpr(EXEC_LO.offset)
    src0 = rsrc(inst.src0.offset, lane)
    vdst_reg = inst.vdst.offset - 256
    pcode_result = compile_vop_pcode(inst.op, {'S0': src0}, lane, wvgpr, wsgpr, rsgpr, vdst_reg, exec_mask, inc_pc, name)
    assert pcode_result is not None, f"no pcode for VOP1: {inst.op.name}"
    return pcode_result

  # ═══════════════════════════════════════════════════════════════════════════
  # VOPC: vector compare, writes to VCC (or EXEC for CMPX)
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOPC):
    lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    exec_mask = rsgpr(EXEC_LO.offset)
    src0 = rsrc(inst.src0.offset, lane)
    src1 = rsrc(inst.vsrc1.offset, lane)
    # For VOPC, D0 = VCC in pcode. Provide VCC as the initial D0 value for pcode parsing
    pcode_result = compile_vop_pcode(inst.op, {'S0': src0, 'S1': src1, 'D0': rsgpr(VCC_LO.offset)}, lane, wvgpr, wsgpr, rsgpr, 0, exec_mask, inc_pc, name)
    assert pcode_result is not None, f"no pcode for VOPC: {inst.op.name}"
    return pcode_result

  # ═══════════════════════════════════════════════════════════════════════════
  # VOP2: v_add_f32, v_lshlrev_b32, etc.
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOP2):
    lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    exec_mask = rsgpr(EXEC_LO.offset)
    src0 = rsrc(inst.src0.offset, lane)
    src1 = rsrc(inst.vsrc1.offset, lane)
    vdst_reg = inst.vdst.offset - 256
    pcode_result = compile_vop_pcode(inst.op, {'S0': src0, 'S1': src1}, lane, wvgpr, wsgpr, rsgpr, vdst_reg, exec_mask, inc_pc, name)
    assert pcode_result is not None, f"no pcode for VOP2: {inst.op.name}"
    return pcode_result

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
    srcs = {'S0': src0, 'S1': src1}
    if src2 is not None: srcs['S2'] = src2
    pcode_result = compile_vop_pcode(inst.op, srcs, lane, wvgpr, wsgpr, rsgpr, vdst_reg, exec_mask, inc_pc, name)
    assert pcode_result is not None, f"no pcode for VOP3: {inst.op.name}"
    return pcode_result

  # ═══════════════════════════════════════════════════════════════════════════
  # VOPD: dual-issue v_dual_mov_b32, etc.
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOPD):
    exec_mask = rsgpr(EXEC_LO.offset)
    vdstx_reg = inst.vdstx.offset - 256
    vdsty_reg = (inst.vdsty << 1) | ((inst.vdstx.offset & 1) ^ 1)
    ended = []

    # X operation
    lane_x = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    srcx0 = rsrc(inst.srcx0.offset, lane_x)
    vsrcx1 = rvgpr(inst.vsrcx1.offset - 256, lane_x)
    vop_opx = VOPD_TO_VOP2.get(inst.opx)
    assert vop_opx is not None, f"no VOP mapping for VOPD X: {inst.opx}"
    # For FMAC, D0 is both input and output: D0.f32 = fma(S0.f32, S1.f32, D0.f32)
    srcsx = {'S0': srcx0, 'S1': vsrcx1, 'D0': rvgpr(vdstx_reg, lane_x)}
    stores_x = compile_vop_pcode_stores(vop_opx, srcsx, lane_x, wvgpr, wsgpr, rsgpr, vdstx_reg, exec_mask)
    assert stores_x is not None, f"no pcode for VOPD X: {vop_opx}"
    ended.extend(stores_x)

    # Y operation
    lane_y = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    srcy0 = rsrc(inst.srcy0.offset, lane_y)
    vsrcy1 = rvgpr(inst.vsrcy1.offset - 256, lane_y)
    vop_opy = VOPD_TO_VOP2.get(inst.opy)
    assert vop_opy is not None, f"no VOP mapping for VOPD Y: {inst.opy}"
    srcsy = {'S0': srcy0, 'S1': vsrcy1, 'D0': rvgpr(vdsty_reg, lane_y)}
    stores_y = compile_vop_pcode_stores(vop_opy, srcsy, lane_y, wvgpr, wsgpr, rsgpr, vdsty_reg, exec_mask)
    assert stores_y is not None, f"no pcode for VOPD Y: {vop_opy}"
    ended.extend(stores_y)

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
