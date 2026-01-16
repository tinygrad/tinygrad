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
    elif dest.startswith('EXEC'):
      # EXEC update - write to EXEC_LO (and EXEC_HI for 64-bit)
      if val.dtype in (dtypes.uint64, dtypes.int64):
        stores.append(wsgpr_fn(EXEC_LO.offset, val.cast(dtypes.uint32)))
        stores.append(wsgpr_fn(EXEC_HI.offset, (val >> UOp.const(dtypes.uint64, 32)).cast(dtypes.uint32)))
      else:
        stores.append(wsgpr_fn(EXEC_LO.offset, val.cast(dtypes.uint32)))
    elif dest.startswith('VCC'):
      # VCC update
      if val.dtype in (dtypes.uint64, dtypes.int64):
        stores.append(wsgpr_fn(VCC_LO.offset, val.cast(dtypes.uint32)))
        stores.append(wsgpr_fn(VCC_HI.offset, (val >> UOp.const(dtypes.uint64, 32)).cast(dtypes.uint32)))
      else:
        stores.append(wsgpr_fn(VCC_LO.offset, val.cast(dtypes.uint32)))

  if not stores: return None
  return name, UOp.sink(*stores, inc_pc_fn(), arg=KernelInfo(name=name))

def compile_vop_pcode_stores(op, srcs: dict[str, UOp], lane: UOp, wvgpr_fn, wsgpr_fn, rsgpr_fn, vdst_reg: int, exec_mask: UOp,
                             opsel_dst_hi: bool = False, rvgpr_fn = None) -> list[UOp] | None:
  """Compile a VOP instruction using pcode parser. Returns list of store UOps or None."""
  pcode = PCODE.get(op)
  if pcode is None: return None

  # Manual implementations for instructions with complex pcode (if/elsif/else, for loops)
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

  # First pass: collect all stores without ending the loop yet
  raw_stores = []
  has_vgpr_write = False
  for dest, val in assigns:
    if 'D0' in dest and '[laneId]' in dest:
      # D0.u64[laneId] means VCC bit write (for VOPC instructions)
      vcc = rsgpr_fn(VCC_LO.offset)
      vcc_mask = UOp.const(dtypes.uint32, 1) << lane.cast(dtypes.uint32)
      exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
      vcc_bit = val.cast(dtypes.uint32) << lane.cast(dtypes.uint32)
      vcc_cleared = vcc & (vcc_mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))
      vcc_new = exec_bit.ne(UOp.const(dtypes.uint32, 0)).where(vcc_cleared | vcc_bit, vcc)
      raw_stores.append(('vcc', wsgpr_fn(VCC_LO.offset, vcc_new)))
    elif dest.startswith('D0'):
      # Vector destination - write to vdst (bitcast floats to preserve bit pattern)
      has_vgpr_write = True
      if val.dtype in (dtypes.uint64, dtypes.int64, dtypes.float64):
        # 64-bit destination: write both low and high 32-bit parts
        val64 = val.cast(dtypes.uint64) if val.dtype != dtypes.uint64 else val
        lo = val64.cast(dtypes.uint32)
        hi = (val64 >> UOp.const(dtypes.uint64, 32)).cast(dtypes.uint32)
        raw_stores.append(('vgpr', wvgpr_fn(vdst_reg, lane, lo, exec_mask)))
        raw_stores.append(('vgpr', wvgpr_fn(vdst_reg + 1, lane, hi, exec_mask)))
      elif val.dtype in (dtypes.float32,):
        result = val.bitcast(dtypes.uint32)
        raw_stores.append(('vgpr', wvgpr_fn(vdst_reg, lane, result, exec_mask)))
      elif val.dtype in (dtypes.half, dtypes.uint16, dtypes.int16):
        # 16-bit: cast to uint32 (bitcast for float, cast for int)
        result = val.bitcast(dtypes.uint16).cast(dtypes.uint32) if val.dtype == dtypes.half else val.cast(dtypes.uint32)
        if rvgpr_fn is not None:
          old_val = rvgpr_fn(vdst_reg, lane)
          if opsel_dst_hi:
            # Write to high 16 bits, preserving low 16 bits
            result = (old_val & UOp.const(dtypes.uint32, 0xFFFF)) | (result << UOp.const(dtypes.uint32, 16))
          else:
            # Write to low 16 bits, preserving high 16 bits
            result = (old_val & UOp.const(dtypes.uint32, 0xFFFF0000)) | (result & UOp.const(dtypes.uint32, 0xFFFF))
        raw_stores.append(('vgpr', wvgpr_fn(vdst_reg, lane, result, exec_mask)))
      elif val.dtype != dtypes.uint32:
        result = val.cast(dtypes.uint32)
        raw_stores.append(('vgpr', wvgpr_fn(vdst_reg, lane, result, exec_mask)))
      else:
        raw_stores.append(('vgpr', wvgpr_fn(vdst_reg, lane, val, exec_mask)))
    elif dest.startswith('VCC'):
      # VCC update - need to set bit for this lane
      vcc = rsgpr_fn(VCC_LO.offset)
      vcc_mask = UOp.const(dtypes.uint32, 1) << lane.cast(dtypes.uint32)
      exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
      vcc_bit = val.cast(dtypes.uint32) << lane.cast(dtypes.uint32)
      vcc_cleared = vcc & (vcc_mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))
      vcc_new = exec_bit.ne(UOp.const(dtypes.uint32, 0)).where(vcc_cleared | vcc_bit, vcc)
      raw_stores.append(('vcc', wsgpr_fn(VCC_LO.offset, vcc_new)))
    elif dest.startswith('EXEC'):
      # EXEC update (for CMPX)
      exec_val = rsgpr_fn(EXEC_LO.offset)
      exec_mask_bit = UOp.const(dtypes.uint32, 1) << lane.cast(dtypes.uint32)
      exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
      exec_new_bit = val.cast(dtypes.uint32) << lane.cast(dtypes.uint32)
      exec_cleared = exec_val & (exec_mask_bit ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))
      exec_new = exec_bit.ne(UOp.const(dtypes.uint32, 0)).where(exec_cleared | exec_new_bit, exec_val)
      raw_stores.append(('exec', wsgpr_fn(EXEC_LO.offset, exec_new)))
    elif dest.startswith('SCC'):
      raw_stores.append(('scc', wsgpr_fn(SCC_IDX, val.cast(dtypes.uint32))))

  # Second pass: combine all lane-dependent stores into a single END
  stores = []
  lane_stores = [s for t, s in raw_stores if t in ('vgpr', 'vcc', 'exec')]
  scalar_stores = [s for t, s in raw_stores if t == 'scc']
  if lane_stores:
    # Combine all lane-dependent stores and end with single END
    combined = UOp.sink(*lane_stores)
    stores.append(combined.end(lane))
  stores.extend(scalar_stores)

  return stores if stores else None

def compile_vop_pcode(op, srcs: dict[str, UOp], lane: UOp, wvgpr_fn, wsgpr_fn, rsgpr_fn, vdst_reg: int, exec_mask: UOp, inc_pc_fn, name: str,
                      opsel_dst_hi: bool = False, rvgpr_fn = None):
  """Try to compile a VOP instruction using pcode parser. Returns (name, sink) or None."""
  stores = compile_vop_pcode_stores(op, srcs, lane, wvgpr_fn, wsgpr_fn, rsgpr_fn, vdst_reg, exec_mask, opsel_dst_hi, rvgpr_fn)
  if stores is None: return None
  return name, UOp.sink(*stores, inc_pc_fn(), arg=KernelInfo(name=name))

def compile_ds_pcode(op, inst, lane: UOp, rvgpr_fn, wvgpr_fn, lds: UOp, exec_mask: UOp, inc_pc_fn, name: str):
  """Compile a DS instruction using pcode parser. Returns (name, sink) or None."""
  pcode = PCODE.get(op)
  if pcode is None: return None

  # Extract instruction fields
  addr_reg = inst.addr.offset - 256 if inst.addr.offset >= 256 else inst.addr.offset
  offset0 = getattr(inst, 'offset0', 0) or 0
  offset1 = getattr(inst, 'offset1', 0) or 0
  data0_reg = (inst.data0.offset - 256 if inst.data0.offset >= 256 else inst.data0.offset) if inst.data0 is not None else 0
  data1_reg = (inst.data1.offset - 256 if inst.data1.offset >= 256 else inst.data1.offset) if inst.data1 is not None else 0
  vdst_reg = (inst.vdst.offset - 256 if inst.vdst.offset >= 256 else inst.vdst.offset) if inst.vdst is not None else 0

  # Build source operands for pcode
  # ADDR/ADDR_BASE is base address from vgpr
  base_addr = rvgpr_fn(addr_reg, lane)
  # OFFSET/OFFSET0/OFFSET1 are immediate values from instruction
  # DATA is 64-bit from data0 vgpr pair, DATA2 from data1 vgpr pair
  data_lo = rvgpr_fn(data0_reg, lane)
  data_hi = rvgpr_fn(data0_reg + 1, lane) if data0_reg else UOp.const(dtypes.uint32, 0)
  data = data_lo.cast(dtypes.uint64) | (data_hi.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))
  data2_lo = rvgpr_fn(data1_reg, lane) if data1_reg else UOp.const(dtypes.uint32, 0)
  data2_hi = rvgpr_fn(data1_reg + 1, lane) if data1_reg else UOp.const(dtypes.uint32, 0)
  data2 = data2_lo.cast(dtypes.uint64) | (data2_hi.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))

  srcs = {
    'ADDR': base_addr, 'ADDR_BASE': base_addr,
    'OFFSET': UOp.const(dtypes.uint32, offset0), 'OFFSET0': UOp.const(dtypes.uint32, offset0), 'OFFSET1': UOp.const(dtypes.uint32, offset1),
    'DATA': data, 'DATA2': data2,
    '_lds': lds,
  }

  _, assigns = parse_pcode(pcode, srcs, lane)

  # Process assigns - handle MEM writes and RETURN_DATA
  stores = []
  return_data = {}  # Collect bit slices for RETURN_DATA

  for dest, val in assigns:
    if dest.startswith('MEM['):
      # LDS memory write: val is (addr, value) tuple
      addr, write_val = val
      if addr.dtype != dtypes.uint32: addr = addr.cast(dtypes.uint32)
      if write_val.dtype != dtypes.uint32: write_val = write_val.cast(dtypes.uint32)
      idx = lds.index((addr >> UOp.const(dtypes.uint32, 2)).cast(dtypes.index))
      exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
      active = exec_bit.ne(UOp.const(dtypes.uint32, 0))
      stores.append(idx.store(active.where(write_val, idx)))
    elif dest.startswith('RETURN_DATA['):
      # Bit slice assignment: RETURN_DATA[63:32] = val means write to vdst+1
      m = re.match(r'RETURN_DATA\[(\d+):(\d+)\]', dest)
      if m:
        high_bit, low_bit = int(m.group(1)), int(m.group(2))
        dword_idx = low_bit // 32
        return_data[dword_idx] = val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val
    elif dest.startswith('RETURN_DATA'):
      # Simple RETURN_DATA.type = val
      return_data[0] = val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val

  # Write RETURN_DATA to vgpr destination
  for dword_idx, val in sorted(return_data.items()):
    stores.append(wvgpr_fn(vdst_reg + dword_idx, lane, val, exec_mask))

  if not stores: return None
  # Combine all stores and end the lane loop
  return name, UOp.sink(UOp.sink(*stores).end(lane), inc_pc_fn(), arg=KernelInfo(name=name))

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
  # SOPP: s_endpgm, s_waitcnt, s_clause, branches
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, SOPP):
    if inst.op == SOPPOp.S_ENDPGM:
      return name, UOp.sink(wsgpr(PC_LO_IDX, UOp.const(dtypes.uint32, 0xFFFFFFFF)), arg=KernelInfo(name=name))

    # Branch instructions - simm16 is signed offset in dwords from PC+4
    simm16 = _sext(inst.simm16, 16)
    branch_target = rsgpr(PC_LO_IDX) + UOp.const(dtypes.uint32, simm16 + inst_words)  # PC + offset + inst_size
    fall_through = rsgpr(PC_LO_IDX) + UOp.const(dtypes.uint32, inst_words)

    if inst.op == SOPPOp.S_BRANCH:
      return name, UOp.sink(wsgpr(PC_LO_IDX, branch_target), arg=KernelInfo(name=name))

    if inst.op == SOPPOp.S_CBRANCH_SCC0:
      scc = rsgpr(SCC_IDX)
      cond = scc.eq(UOp.const(dtypes.uint32, 0))
      return name, UOp.sink(wsgpr(PC_LO_IDX, cond.where(branch_target, fall_through)), arg=KernelInfo(name=name))

    if inst.op == SOPPOp.S_CBRANCH_SCC1:
      scc = rsgpr(SCC_IDX)
      cond = scc.ne(UOp.const(dtypes.uint32, 0))
      return name, UOp.sink(wsgpr(PC_LO_IDX, cond.where(branch_target, fall_through)), arg=KernelInfo(name=name))

    if inst.op == SOPPOp.S_CBRANCH_VCCZ:
      vcc = rsgpr(VCC_LO.offset)
      cond = vcc.eq(UOp.const(dtypes.uint32, 0))
      return name, UOp.sink(wsgpr(PC_LO_IDX, cond.where(branch_target, fall_through)), arg=KernelInfo(name=name))

    if inst.op == SOPPOp.S_CBRANCH_VCCNZ:
      vcc = rsgpr(VCC_LO.offset)
      cond = vcc.ne(UOp.const(dtypes.uint32, 0))
      return name, UOp.sink(wsgpr(PC_LO_IDX, cond.where(branch_target, fall_through)), arg=KernelInfo(name=name))

    if inst.op == SOPPOp.S_CBRANCH_EXECZ:
      exec_lo = rsgpr(EXEC_LO.offset)
      cond = exec_lo.eq(UOp.const(dtypes.uint32, 0))
      return name, UOp.sink(wsgpr(PC_LO_IDX, cond.where(branch_target, fall_through)), arg=KernelInfo(name=name))

    if inst.op == SOPPOp.S_CBRANCH_EXECNZ:
      exec_lo = rsgpr(EXEC_LO.offset)
      cond = exec_lo.ne(UOp.const(dtypes.uint32, 0))
      return name, UOp.sink(wsgpr(PC_LO_IDX, cond.where(branch_target, fall_through)), arg=KernelInfo(name=name))

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
    src_off = inst.ssrc0.offset
    s0 = rsrc(src_off, UOp.const(dtypes.index, 0))
    dst_reg = inst.sdst.offset
    op_name = inst.op.name
    sizes = inst.op_regs if hasattr(inst, 'op_regs') else {}
    dst_size = sizes.get('sdst', 1)

    # Handle 64-bit source for B64 ops
    if sizes.get('ssrc0', 1) == 2:
      if src_off >= 128:  # inline constant - zero-extend to 64-bit
        s0 = s0.cast(dtypes.uint64)
      else:  # sgpr pair
        s0 = rsgpr64(src_off)

    # SAVEEXEC ops: save EXEC to dst, then modify EXEC (complex pcode with temp var)
    if 'OR_SAVEEXEC_B32' in op_name:
      exec_lo = rsgpr(EXEC_LO.offset)
      new_exec = exec_lo | s0
      return name, UOp.sink(wsgpr(dst_reg, exec_lo), wsgpr(EXEC_LO.offset, new_exec), inc_pc(), arg=KernelInfo(name=name))

    if 'AND_SAVEEXEC_B32' in op_name:
      exec_lo = rsgpr(EXEC_LO.offset)
      new_exec = exec_lo & s0
      return name, UOp.sink(wsgpr(dst_reg, exec_lo), wsgpr(EXEC_LO.offset, new_exec), inc_pc(), arg=KernelInfo(name=name))

    # S_BREV_B32: bit reverse (pcode uses reversed slice which parser can't handle)
    if op_name == 'S_BREV_B32':
      # Bit reverse using parallel swap approach
      v = s0
      v = ((v >> UOp.const(dtypes.uint32, 1)) & UOp.const(dtypes.uint32, 0x55555555)) | ((v & UOp.const(dtypes.uint32, 0x55555555)) << UOp.const(dtypes.uint32, 1))
      v = ((v >> UOp.const(dtypes.uint32, 2)) & UOp.const(dtypes.uint32, 0x33333333)) | ((v & UOp.const(dtypes.uint32, 0x33333333)) << UOp.const(dtypes.uint32, 2))
      v = ((v >> UOp.const(dtypes.uint32, 4)) & UOp.const(dtypes.uint32, 0x0F0F0F0F)) | ((v & UOp.const(dtypes.uint32, 0x0F0F0F0F)) << UOp.const(dtypes.uint32, 4))
      v = ((v >> UOp.const(dtypes.uint32, 8)) & UOp.const(dtypes.uint32, 0x00FF00FF)) | ((v & UOp.const(dtypes.uint32, 0x00FF00FF)) << UOp.const(dtypes.uint32, 8))
      v = (v >> UOp.const(dtypes.uint32, 16)) | (v << UOp.const(dtypes.uint32, 16))
      return name, UOp.sink(wsgpr(dst_reg, v), inc_pc(), arg=KernelInfo(name=name))

    # S_QUADMASK_B32: for each of 8 quads, if any bit in quad is set, set result bit (pcode uses for loop)
    if op_name == 'S_QUADMASK_B32':
      result = UOp.const(dtypes.uint32, 0)
      for i in range(8):
        quad = (s0 >> UOp.const(dtypes.uint32, i * 4)) & UOp.const(dtypes.uint32, 0xF)
        bit = quad.ne(UOp.const(dtypes.uint32, 0)).where(UOp.const(dtypes.uint32, 1), UOp.const(dtypes.uint32, 0))
        result = result | (bit << UOp.const(dtypes.uint32, i))
      scc = result.ne(UOp.const(dtypes.uint32, 0)).where(UOp.const(dtypes.uint32, 1), UOp.const(dtypes.uint32, 0))
      return name, UOp.sink(wsgpr(dst_reg, result), wsgpr(SCC_IDX, scc), inc_pc(), arg=KernelInfo(name=name))

    # S_WQM_B32: Whole Quad Mode - if any lane in quad active, set all lanes in that quad (pcode uses for loop)
    if op_name == 'S_WQM_B32':
      result = UOp.const(dtypes.uint32, 0)
      for i in range(8):
        quad = (s0 >> UOp.const(dtypes.uint32, i * 4)) & UOp.const(dtypes.uint32, 0xF)
        active = quad.ne(UOp.const(dtypes.uint32, 0)).where(UOp.const(dtypes.uint32, 0xF), UOp.const(dtypes.uint32, 0))
        result = result | (active << UOp.const(dtypes.uint32, i * 4))
      scc = result.ne(UOp.const(dtypes.uint32, 0)).where(UOp.const(dtypes.uint32, 1), UOp.const(dtypes.uint32, 0))
      return name, UOp.sink(wsgpr(dst_reg, result), wsgpr(SCC_IDX, scc), inc_pc(), arg=KernelInfo(name=name))

    # Try pcode for other SOP1 ops
    pcode_result = compile_sop_pcode(inst.op, {'S0': s0}, wsgpr, rsgpr, dst_reg, dst_size, inc_pc, name)
    assert pcode_result is not None, f"unimplemented SOP1: {op_name}"
    return pcode_result

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
  # SOPC: scalar compare (s_cmp_eq_u32, etc.) - only writes SCC
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, SOPC):
    sizes = inst.op_regs if hasattr(inst, 'op_regs') else {}
    s0 = rsgpr64(inst.ssrc0.offset) if sizes.get('ssrc0', 1) == 2 else rsrc(inst.ssrc0.offset, UOp.const(dtypes.index, 0))
    s1 = rsgpr64(inst.ssrc1.offset) if sizes.get('ssrc1', 1) == 2 else rsrc(inst.ssrc1.offset, UOp.const(dtypes.index, 0))
    # SOPC has no sdst, but compile_sop_pcode needs one - use dummy reg 0, it will only write SCC
    pcode_result = compile_sop_pcode(inst.op, {'S0': s0, 'S1': s1}, wsgpr, rsgpr, 0, 0, inc_pc, name)
    assert pcode_result is not None, f"no pcode for SOPC: {inst.op.name}"
    return pcode_result

  # ═══════════════════════════════════════════════════════════════════════════
  # VOP1: v_mov_b32, etc.
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOP1):
    lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    exec_mask = rsgpr(EXEC_LO.offset)
    src0 = rsrc(inst.src0.offset, lane)
    vdst_reg = inst.vdst.offset - 256
    pcode_result = compile_vop_pcode(inst.op, {'S0': src0}, lane, wvgpr, wsgpr, rsgpr, vdst_reg, exec_mask, inc_pc, name,
                                     rvgpr_fn=rvgpr)
    assert pcode_result is not None, f"no pcode for VOP1: {inst.op.name}"
    return pcode_result

  # ═══════════════════════════════════════════════════════════════════════════
  # VOPC: vector compare, writes to VCC (or EXEC for CMPX)
  # Uses unrolled computation to avoid loop-carried VCC dependency issues
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOPC):
    exec_mask = rsgpr(EXEC_LO.offset)
    old_vcc = rsgpr(VCC_LO.offset)
    op_name = inst.op.name if hasattr(inst.op, 'name') else str(inst.op)
    is_cmpx = 'CMPX' in op_name
    is_16bit = 'F16' in op_name or 'B16' in op_name or 'I16' in op_name or 'U16' in op_name

    # For 16-bit VOPC, vsrc1 index 128-255 means read high 16 bits of v[vsrc1-128]
    vsrc1_reg = inst.vsrc1.offset - 256  # Convert to vgpr index (0-255)
    vsrc1_hi = is_16bit and vsrc1_reg >= 128  # High-half access only for 16-bit ops
    vsrc1_actual_reg = vsrc1_reg - 128 if vsrc1_hi else vsrc1_reg
    vsrc1_offset = 256 + vsrc1_actual_reg  # Convert back to rsrc offset

    # Helper to get comparison result for a lane (returns 0 or 1)
    def get_cmp_result(lane_idx: int) -> UOp:
      lane_const = UOp.const(dtypes.index, lane_idx)
      s0 = rsrc(inst.src0.offset, lane_const)
      s1 = rsrc(vsrc1_offset, lane_const)
      # For 16-bit ops with vsrc1 high-half access, extract high 16 bits
      if is_16bit and vsrc1_hi:
        s1 = (s1 >> UOp.const(dtypes.uint32, 16)) & UOp.const(dtypes.uint32, 0xFFFF)
      # Get pcode and parse for this comparison
      pcode = PCODE.get(inst.op)
      if pcode is None: return UOp.const(dtypes.uint32, 0)
      _, assigns = parse_pcode(pcode, {'S0': s0, 'S1': s1}, lane=UOp.const(dtypes.uint32, lane_idx))
      # Find the D0.u64[laneId] assignment (the comparison result)
      for dest, val in assigns:
        if 'D0' in dest and '[laneId]' in dest:
          return val.cast(dtypes.uint32)
      return UOp.const(dtypes.uint32, 0)

    # Compute all 32 bits by unrolling
    new_vcc_bits = UOp.const(dtypes.uint32, 0)
    for i in range(32):
      cmp_result = get_cmp_result(i)
      bit = cmp_result << UOp.const(dtypes.uint32, i)
      new_vcc_bits = new_vcc_bits | bit

    # Apply EXEC mask: new_vcc = (old_vcc & ~exec) | (new_bits & exec)
    new_vcc = (old_vcc & (exec_mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))) | (new_vcc_bits & exec_mask)

    # Store to VCC (or EXEC for CMPX)
    if is_cmpx:
      stores = [wsgpr(EXEC_LO.offset, new_vcc), wsgpr(VCC_LO.offset, new_vcc)]
    else:
      stores = [wsgpr(VCC_LO.offset, new_vcc)]

    return name, UOp.sink(*stores, inc_pc(), arg=KernelInfo(name=name))

  # ═══════════════════════════════════════════════════════════════════════════
  # VOP2: v_add_f32, v_lshlrev_b32, etc.
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOP2):
    lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    exec_mask = rsgpr(EXEC_LO.offset)
    src0 = rsrc(inst.src0.offset, lane)
    src1 = rsrc(inst.vsrc1.offset, lane)
    vdst_reg = inst.vdst.offset - 256
    # For FMAC/accumulator instructions, D0 is also a source (read from vdst)
    srcs = {'S0': src0, 'S1': src1, 'D0': rvgpr(vdst_reg, lane)}
    # FMAAK/FMAMK use inline literal constant (SIMM32)
    op_name = inst.op.name if hasattr(inst.op, 'name') else str(inst.op)
    if 'FMAAK' in op_name or 'FMAMK' in op_name:
      srcs['SIMM32'] = UOp.const(dtypes.uint32, literal)
    pcode_result = compile_vop_pcode(inst.op, srcs, lane, wvgpr, wsgpr, rsgpr, vdst_reg, exec_mask, inc_pc, name)
    assert pcode_result is not None, f"no pcode for VOP2: {inst.op.name}"
    return pcode_result

  # ═══════════════════════════════════════════════════════════════════════════
  # VOP3: 3-operand vector ALU (v_add_f32_e64, v_fma_f32, etc.)
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOP3):
    exec_mask = rsgpr(EXEC_LO.offset)
    sizes = inst.op_regs if hasattr(inst, 'op_regs') else {}
    opsel = getattr(inst, 'opsel', 0) or 0
    op_name = inst.op.name if hasattr(inst.op, 'name') else str(inst.op)

    # Check if this is a VOP3 VOPC instruction (v_cmp_*_e64) - these write to scalar dest
    # VOP3 VOPC: writes to D0 (scalar dest), uses unrolled computation like VOPC
    is_vop3_vopc = 'V_CMP' in op_name or 'V_CMPX' in op_name
    if is_vop3_vopc:
      old_sdst = rsgpr(inst.vdst.offset)  # vdst is actually sdst for VOP3_SDST
      is_cmpx = 'CMPX' in op_name

      # Helper for opsel: extract high or low 16 bits
      def apply_opsel_scalar(val: UOp, sel_bit: int) -> UOp:
        if opsel & (1 << sel_bit):
          return (val >> UOp.const(dtypes.uint32, 16)) & UOp.const(dtypes.uint32, 0xFFFF)
        return val

      # Helper to get comparison result for a lane
      def get_cmp_result_vop3(lane_idx: int) -> UOp:
        lane_const = UOp.const(dtypes.index, lane_idx)
        s0 = rsrc(inst.src0.offset, lane_const)
        s1 = rsrc(inst.src1.offset, lane_const)
        # Apply opsel for 16-bit operations
        if 'F16' in op_name or 'B16' in op_name or 'I16' in op_name or 'U16' in op_name:
          s0 = apply_opsel_scalar(s0, 0)
          s1 = apply_opsel_scalar(s1, 1)
        pcode = PCODE.get(inst.op)
        if pcode is None: return UOp.const(dtypes.uint32, 0)
        _, assigns = parse_pcode(pcode, {'S0': s0, 'S1': s1}, lane=UOp.const(dtypes.uint32, lane_idx))
        for dest, val in assigns:
          if 'D0' in dest and '[laneId]' in dest:
            return val.cast(dtypes.uint32)
        return UOp.const(dtypes.uint32, 0)

      # Compute all 32 bits by unrolling
      new_bits = UOp.const(dtypes.uint32, 0)
      for i in range(32):
        cmp_result = get_cmp_result_vop3(i)
        bit = cmp_result << UOp.const(dtypes.uint32, i)
        new_bits = new_bits | bit

      # Apply EXEC mask
      new_sdst = (old_sdst & (exec_mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))) | (new_bits & exec_mask)

      # Store to scalar destination (and EXEC for CMPX)
      if is_cmpx:
        stores = [wsgpr(EXEC_LO.offset, new_sdst), wsgpr(inst.vdst.offset, new_sdst)]
      else:
        stores = [wsgpr(inst.vdst.offset, new_sdst)]

      return name, UOp.sink(*stores, inc_pc(), arg=KernelInfo(name=name))

    # Regular VOP3 handling
    lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)

    # Helper for 64-bit source reads
    def rsrc64(off: int, lane: UOp) -> UOp:
      lo = rsrc(off, lane)
      hi = rsrc(off + 1, lane)
      return lo.cast(dtypes.uint64) | (hi.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))

    # Helper for opsel: extract high or low 16 bits
    def apply_opsel(val: UOp, sel_bit: int) -> UOp:
      if opsel & (1 << sel_bit):
        return (val >> UOp.const(dtypes.uint32, 16)) & UOp.const(dtypes.uint32, 0xFFFF)
      return val

    src0 = rsrc64(inst.src0.offset, lane) if sizes.get('src0', 1) == 2 else rsrc(inst.src0.offset, lane)
    src1 = rsrc64(inst.src1.offset, lane) if sizes.get('src1', 1) == 2 else rsrc(inst.src1.offset, lane)
    src2 = (rsrc64(inst.src2.offset, lane) if sizes.get('src2', 1) == 2 else rsrc(inst.src2.offset, lane)) if inst.src2 is not None else None

    # Apply opsel to 16-bit operations (F16, etc.)
    if 'F16' in op_name or 'B16' in op_name or 'I16' in op_name or 'U16' in op_name:
      src0 = apply_opsel(src0, 0)
      src1 = apply_opsel(src1, 1)
      if src2 is not None: src2 = apply_opsel(src2, 2)

    vdst_reg = inst.vdst.offset - 256
    srcs = {'S0': src0, 'S1': src1}
    if src2 is not None: srcs['S2'] = src2

    # For 16-bit ops with opsel[3]=1, write to high half
    opsel_dst_hi = bool(opsel & 0b1000) and ('F16' in op_name or 'B16' in op_name or 'I16' in op_name or 'U16' in op_name)
    if opsel_dst_hi:
      stores = compile_vop_pcode_stores(inst.op, srcs, lane, wvgpr, wsgpr, rsgpr, vdst_reg, exec_mask,
                                        opsel_dst_hi=True, rvgpr_fn=rvgpr)
      if stores is not None:
        return name, UOp.sink(*stores, inc_pc(), arg=KernelInfo(name=name))

    pcode_result = compile_vop_pcode(inst.op, srcs, lane, wvgpr, wsgpr, rsgpr, vdst_reg, exec_mask, inc_pc, name)
    assert pcode_result is not None, f"no pcode for VOP3: {inst.op.name}"
    return pcode_result

  # ═══════════════════════════════════════════════════════════════════════════
  # VOP3SD: VOP3 with scalar destination (v_add_co_u32, etc.)
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOP3SD):
    lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    exec_mask = rsgpr(EXEC_LO.offset)
    sizes = inst.op_regs if hasattr(inst, 'op_regs') else {}

    # Helper for 64-bit source reads
    def rsrc64(off: int, lane: UOp) -> UOp:
      lo = rsrc(off, lane)
      hi = rsrc(off + 1, lane)
      return lo.cast(dtypes.uint64) | (hi.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))

    src0 = rsrc64(inst.src0.offset, lane) if sizes.get('src0', 1) == 2 else rsrc(inst.src0.offset, lane)
    src1 = rsrc64(inst.src1.offset, lane) if sizes.get('src1', 1) == 2 else rsrc(inst.src1.offset, lane)
    src2 = (rsrc64(inst.src2.offset, lane) if sizes.get('src2', 1) == 2 else rsrc(inst.src2.offset, lane)) if inst.src2 is not None else None
    vdst_reg = inst.vdst.offset - 256
    srcs = {'S0': src0, 'S1': src1}
    if src2 is not None: srcs['S2'] = src2
    pcode_result = compile_vop_pcode(inst.op, srcs, lane, wvgpr, wsgpr, rsgpr, vdst_reg, exec_mask, inc_pc, name)
    assert pcode_result is not None, f"no pcode for VOP3SD: {inst.op.name}"
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
  # DS: Local Data Share (LDS) operations
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, DS):
    exec_mask = rsgpr(EXEC_LO.offset)
    op_name = inst.op.name if hasattr(inst.op, 'name') else str(inst.op)
    addr_reg = inst.addr.offset - 256 if inst.addr.offset >= 256 else inst.addr.offset

    # LDS helper - reads/writes to lds buffer (uint32 indexed)
    def rlds(addr: UOp) -> UOp: return lds.index((addr >> UOp.const(dtypes.uint32, 2)).cast(dtypes.index))
    def wlds(addr: UOp, val: UOp, active: UOp) -> UOp:
      idx = lds.index((addr >> UOp.const(dtypes.uint32, 2)).cast(dtypes.index))
      return idx.store(active.where(val, idx))

    # 2ADDR B64 variants: DS_STORE_2ADDR_B64, DS_LOAD_2ADDR_B64 (not STOREXCHG)
    if '2ADDR' in op_name and 'B64' in op_name and 'XCHG' not in op_name:
      offset0 = getattr(inst, 'offset0', 0)
      offset1 = getattr(inst, 'offset1', 0)
      mult = 256 if 'STRIDE64' in op_name else 8  # STRIDE64 uses *256, normal uses *8 for B64

      if 'STORE' in op_name:
        data0_reg = inst.data0.offset - 256 if inst.data0.offset >= 256 else inst.data0.offset
        data1_reg = inst.data1.offset - 256 if inst.data1.offset >= 256 else inst.data1.offset
        ended = []
        for off, data_reg in [(offset0, data0_reg), (offset1, data1_reg)]:
          for j in range(2):  # 2 dwords per 64-bit value
            lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
            base_addr = rvgpr(addr_reg, lane)
            addr = base_addr + UOp.const(dtypes.uint32, off * mult + j * 4)
            val = rvgpr(data_reg + j, lane)
            exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
            active = exec_bit.ne(UOp.const(dtypes.uint32, 0))
            ended.append(wlds(addr, val, active).end(lane))
        return name, UOp.sink(*ended, inc_pc(), arg=KernelInfo(name=name))

      if 'LOAD' in op_name:
        vdst_reg = inst.vdst.offset - 256 if inst.vdst.offset >= 256 else inst.vdst.offset
        ended = []
        dst_idx = 0
        for off in [offset0, offset1]:
          for j in range(2):  # 2 dwords per 64-bit value
            lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
            base_addr = rvgpr(addr_reg, lane)
            addr = base_addr + UOp.const(dtypes.uint32, off * mult + j * 4)
            val = rlds(addr)
            store = wvgpr(vdst_reg + dst_idx, lane, val, exec_mask)
            ended.append(store.end(lane))
            dst_idx += 1
        return name, UOp.sink(*ended, inc_pc(), arg=KernelInfo(name=name))

    # 2ADDR variants: DS_STORE_2ADDR_B32, DS_LOAD_2ADDR_B32 (not STOREXCHG)
    if '2ADDR' in op_name and 'B32' in op_name and 'XCHG' not in op_name:
      offset0 = getattr(inst, 'offset0', 0)
      offset1 = getattr(inst, 'offset1', 0)
      mult = 256 if 'STRIDE64' in op_name else 4  # STRIDE64 uses *256, normal uses *4

      if 'STORE' in op_name:
        data0_reg = inst.data0.offset - 256 if inst.data0.offset >= 256 else inst.data0.offset
        data1_reg = inst.data1.offset - 256 if inst.data1.offset >= 256 else inst.data1.offset
        ended = []
        for i, (off, data_reg) in enumerate([(offset0, data0_reg), (offset1, data1_reg)]):
          lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
          base_addr = rvgpr(addr_reg, lane)
          addr = base_addr + UOp.const(dtypes.uint32, off * mult)
          val = rvgpr(data_reg, lane)
          exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
          active = exec_bit.ne(UOp.const(dtypes.uint32, 0))
          ended.append(wlds(addr, val, active).end(lane))
        return name, UOp.sink(*ended, inc_pc(), arg=KernelInfo(name=name))

      if 'LOAD' in op_name:
        vdst_reg = inst.vdst.offset - 256 if inst.vdst.offset >= 256 else inst.vdst.offset
        ended = []
        for i, off in enumerate([offset0, offset1]):
          lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
          base_addr = rvgpr(addr_reg, lane)
          addr = base_addr + UOp.const(dtypes.uint32, off * mult)
          val = rlds(addr)
          store = wvgpr(vdst_reg + i, lane, val, exec_mask)
          ended.append(store.end(lane))
        return name, UOp.sink(*ended, inc_pc(), arg=KernelInfo(name=name))

    # Simple B64 variants: DS_STORE_B64, DS_LOAD_B64 (not STOREXCHG)
    if 'STORE' in op_name and 'B64' in op_name and '2ADDR' not in op_name and 'XCHG' not in op_name:
      offset = getattr(inst, 'offset0', 0) or getattr(inst, 'offset', 0)
      data_reg = inst.data0.offset - 256 if inst.data0.offset >= 256 else inst.data0.offset
      ended = []
      for j in range(2):  # 2 dwords
        lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
        base_addr = rvgpr(addr_reg, lane)
        addr = base_addr + UOp.const(dtypes.uint32, offset + j * 4)
        val = rvgpr(data_reg + j, lane)
        exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
        active = exec_bit.ne(UOp.const(dtypes.uint32, 0))
        ended.append(wlds(addr, val, active).end(lane))
      return name, UOp.sink(*ended, inc_pc(), arg=KernelInfo(name=name))

    if 'LOAD' in op_name and 'B64' in op_name and '2ADDR' not in op_name:
      offset = getattr(inst, 'offset0', 0) or getattr(inst, 'offset', 0)
      vdst_reg = inst.vdst.offset - 256 if inst.vdst.offset >= 256 else inst.vdst.offset
      ended = []
      for j in range(2):  # 2 dwords
        lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
        base_addr = rvgpr(addr_reg, lane)
        addr = base_addr + UOp.const(dtypes.uint32, offset + j * 4)
        val = rlds(addr)
        store = wvgpr(vdst_reg + j, lane, val, exec_mask)
        ended.append(store.end(lane))
      return name, UOp.sink(*ended, inc_pc(), arg=KernelInfo(name=name))

    # Simple B32 variants: DS_STORE_B32, DS_LOAD_B32 (not STOREXCHG)
    if 'STORE' in op_name and 'B32' in op_name and '2ADDR' not in op_name and 'XCHG' not in op_name and 'CMP' not in op_name:
      offset = getattr(inst, 'offset0', 0) or getattr(inst, 'offset', 0)
      data_reg = inst.data0.offset - 256 if inst.data0.offset >= 256 else inst.data0.offset
      lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
      base_addr = rvgpr(addr_reg, lane)
      addr = base_addr + UOp.const(dtypes.uint32, offset)
      val = rvgpr(data_reg, lane)
      exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
      active = exec_bit.ne(UOp.const(dtypes.uint32, 0))
      return name, UOp.sink(wlds(addr, val, active).end(lane), inc_pc(), arg=KernelInfo(name=name))

    if 'LOAD' in op_name and 'B32' in op_name and '2ADDR' not in op_name:
      offset = getattr(inst, 'offset0', 0) or getattr(inst, 'offset', 0)
      vdst_reg = inst.vdst.offset - 256 if inst.vdst.offset >= 256 else inst.vdst.offset
      lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
      base_addr = rvgpr(addr_reg, lane)
      addr = base_addr + UOp.const(dtypes.uint32, offset)
      val = rlds(addr)
      store = wvgpr(vdst_reg, lane, val, exec_mask)
      return name, UOp.sink(store.end(lane), inc_pc(), arg=KernelInfo(name=name))

    # DS_STOREXCHG_RTN_B32: exchange, return old value
    if op_name == 'DS_STOREXCHG_RTN_B32':
      offset = getattr(inst, 'offset0', 0) or getattr(inst, 'offset', 0)
      data_reg = inst.data0.offset - 256 if inst.data0.offset >= 256 else inst.data0.offset
      vdst_reg = inst.vdst.offset - 256 if inst.vdst.offset >= 256 else inst.vdst.offset
      # Use separate lane loops for read and write
      lane1 = UOp.range(32, _next_axis_id(), AxisType.LOOP)
      addr1 = rvgpr(addr_reg, lane1) + UOp.const(dtypes.uint32, offset)
      old_val = rlds(addr1)
      vgpr_store = wvgpr(vdst_reg, lane1, old_val, exec_mask)
      lane2 = UOp.range(32, _next_axis_id(), AxisType.LOOP)
      addr2 = rvgpr(addr_reg, lane2) + UOp.const(dtypes.uint32, offset)
      new_val = rvgpr(data_reg, lane2)
      exec_bit = (exec_mask >> lane2.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
      active = exec_bit.ne(UOp.const(dtypes.uint32, 0))
      lds_store = wlds(addr2, new_val, active)
      return name, UOp.sink(vgpr_store.end(lane1), lds_store.end(lane2), inc_pc(), arg=KernelInfo(name=name))

    # DS_STOREXCHG_RTN_B64: exchange 64-bit, return old value
    if op_name == 'DS_STOREXCHG_RTN_B64':
      offset = getattr(inst, 'offset0', 0) or getattr(inst, 'offset', 0)
      data_reg = inst.data0.offset - 256 if inst.data0.offset >= 256 else inst.data0.offset
      vdst_reg = inst.vdst.offset - 256 if inst.vdst.offset >= 256 else inst.vdst.offset
      ended = []
      for j in range(2):
        lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
        base_addr = rvgpr(addr_reg, lane)
        addr = base_addr + UOp.const(dtypes.uint32, offset + j * 4)
        old_val = rlds(addr)
        new_val = rvgpr(data_reg + j, lane)
        exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
        active = exec_bit.ne(UOp.const(dtypes.uint32, 0))
        lds_store = wlds(addr, new_val, active)
        vgpr_store = wvgpr(vdst_reg + j, lane, old_val, exec_mask)
        ended.append(UOp.sink(lds_store, vgpr_store).end(lane))
      return name, UOp.sink(*ended, inc_pc(), arg=KernelInfo(name=name))

    # DS atomic operations with RTN (return old value)
    atomic_ops = {
      'DS_ADD_RTN_U32': lambda old, data: old + data,
      'DS_SUB_RTN_U32': lambda old, data: old - data,
      'DS_AND_RTN_B32': lambda old, data: old & data,
      'DS_OR_RTN_B32': lambda old, data: old | data,
      'DS_XOR_RTN_B32': lambda old, data: old ^ data,
      'DS_MIN_RTN_U32': lambda old, data: (old < data).where(old, data),
      'DS_MAX_RTN_U32': lambda old, data: (old > data).where(old, data),
      'DS_INC_RTN_U32': lambda old, data: (old >= data).where(UOp.const(dtypes.uint32, 0), old + UOp.const(dtypes.uint32, 1)),
      'DS_DEC_RTN_U32': lambda old, data: (old.eq(UOp.const(dtypes.uint32, 0)) | (old > data)).where(data, old - UOp.const(dtypes.uint32, 1)),
    }
    if op_name in atomic_ops:
      offset = getattr(inst, 'offset0', 0) or getattr(inst, 'offset', 0)
      data_reg = inst.data0.offset - 256 if inst.data0.offset >= 256 else inst.data0.offset
      vdst_reg = inst.vdst.offset - 256 if inst.vdst.offset >= 256 else inst.vdst.offset
      lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
      base_addr = rvgpr(addr_reg, lane)
      addr = base_addr + UOp.const(dtypes.uint32, offset)
      old_val = rlds(addr)
      data_val = rvgpr(data_reg, lane)
      new_val = atomic_ops[op_name](old_val, data_val)
      exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
      active = exec_bit.ne(UOp.const(dtypes.uint32, 0))
      lds_store = wlds(addr, new_val, active)
      vgpr_store = wvgpr(vdst_reg, lane, old_val, exec_mask)
      return name, UOp.sink(UOp.sink(lds_store, vgpr_store).end(lane), inc_pc(), arg=KernelInfo(name=name))

    # DS atomic operations without RTN (no return value)
    atomic_ops_no_rtn = {
      'DS_ADD_U32': lambda old, data: old + data,
      'DS_SUB_U32': lambda old, data: old - data,
      'DS_AND_B32': lambda old, data: old & data,
      'DS_OR_B32': lambda old, data: old | data,
      'DS_XOR_B32': lambda old, data: old ^ data,
      'DS_MIN_U32': lambda old, data: (old < data).where(old, data),
      'DS_MAX_U32': lambda old, data: (old > data).where(old, data),
      'DS_INC_U32': lambda old, data: (old >= data).where(UOp.const(dtypes.uint32, 0), old + UOp.const(dtypes.uint32, 1)),
      'DS_DEC_U32': lambda old, data: (old.eq(UOp.const(dtypes.uint32, 0)) | (old > data)).where(data, old - UOp.const(dtypes.uint32, 1)),
    }
    if op_name in atomic_ops_no_rtn:
      offset = getattr(inst, 'offset0', 0) or getattr(inst, 'offset', 0)
      data_reg = inst.data0.offset - 256 if inst.data0.offset >= 256 else inst.data0.offset
      lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
      base_addr = rvgpr(addr_reg, lane)
      addr = base_addr + UOp.const(dtypes.uint32, offset)
      old_val = rlds(addr)
      data_val = rvgpr(data_reg, lane)
      new_val = atomic_ops_no_rtn[op_name](old_val, data_val)
      exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
      active = exec_bit.ne(UOp.const(dtypes.uint32, 0))
      return name, UOp.sink(wlds(addr, new_val, active).end(lane), inc_pc(), arg=KernelInfo(name=name))

    # DS_CMPSTORE_B32: compare and swap
    if op_name == 'DS_CMPSTORE_B32':
      offset = getattr(inst, 'offset0', 0) or getattr(inst, 'offset', 0)
      data_reg = inst.data0.offset - 256 if inst.data0.offset >= 256 else inst.data0.offset
      data2_reg = inst.data1.offset - 256 if inst.data1.offset >= 256 else inst.data1.offset
      lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
      base_addr = rvgpr(addr_reg, lane)
      addr = base_addr + UOp.const(dtypes.uint32, offset)
      old_val = rlds(addr)
      src_val = rvgpr(data_reg, lane)
      cmp_val = rvgpr(data2_reg, lane)
      # Only store src_val if old_val == cmp_val, otherwise store old_val (no change)
      matches = old_val.eq(cmp_val)
      new_val = matches.where(src_val, old_val)
      exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
      active = exec_bit.ne(UOp.const(dtypes.uint32, 0))
      return name, UOp.sink(wlds(addr, new_val, active).end(lane), inc_pc(), arg=KernelInfo(name=name))

    # DS_STOREXCHG_2ADDR_RTN_B32: exchange at two addresses, return old values
    # Must use single loop to ensure addr reads happen before any vgpr writes
    if op_name == 'DS_STOREXCHG_2ADDR_RTN_B32':
      offset0 = getattr(inst, 'offset0', 0)
      offset1 = getattr(inst, 'offset1', 0)
      mult = 4
      data0_reg = inst.data0.offset - 256 if inst.data0.offset >= 256 else inst.data0.offset
      data1_reg = inst.data1.offset - 256 if inst.data1.offset >= 256 else inst.data1.offset
      vdst_reg = inst.vdst.offset - 256 if inst.vdst.offset >= 256 else inst.vdst.offset
      lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
      exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
      active = exec_bit.ne(UOp.const(dtypes.uint32, 0))
      base_addr = rvgpr(addr_reg, lane)
      # Read addresses and old LDS values first
      addr0 = base_addr + UOp.const(dtypes.uint32, offset0 * mult)
      addr1 = base_addr + UOp.const(dtypes.uint32, offset1 * mult)
      old0 = rlds(addr0)
      old1 = rlds(addr1)
      new0 = rvgpr(data0_reg, lane)
      new1 = rvgpr(data1_reg, lane)
      # All writes in single sink
      lds_store0 = wlds(addr0, new0, active)
      lds_store1 = wlds(addr1, new1, active)
      vgpr_store0 = wvgpr(vdst_reg, lane, old0, exec_mask)
      vgpr_store1 = wvgpr(vdst_reg + 1, lane, old1, exec_mask)
      return name, UOp.sink(UOp.sink(lds_store0, lds_store1, vgpr_store0, vgpr_store1).end(lane), inc_pc(), arg=KernelInfo(name=name))

    # DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32: exchange at two addresses with stride64, return old values
    # Must use single loop to ensure addr reads happen before any vgpr writes
    if op_name == 'DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32':
      offset0 = getattr(inst, 'offset0', 0)
      offset1 = getattr(inst, 'offset1', 0)
      mult = 256
      data0_reg = inst.data0.offset - 256 if inst.data0.offset >= 256 else inst.data0.offset
      data1_reg = inst.data1.offset - 256 if inst.data1.offset >= 256 else inst.data1.offset
      vdst_reg = inst.vdst.offset - 256 if inst.vdst.offset >= 256 else inst.vdst.offset
      lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
      exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
      active = exec_bit.ne(UOp.const(dtypes.uint32, 0))
      base_addr = rvgpr(addr_reg, lane)
      addr0 = base_addr + UOp.const(dtypes.uint32, offset0 * mult)
      addr1 = base_addr + UOp.const(dtypes.uint32, offset1 * mult)
      old0 = rlds(addr0)
      old1 = rlds(addr1)
      new0 = rvgpr(data0_reg, lane)
      new1 = rvgpr(data1_reg, lane)
      lds_store0 = wlds(addr0, new0, active)
      lds_store1 = wlds(addr1, new1, active)
      vgpr_store0 = wvgpr(vdst_reg, lane, old0, exec_mask)
      vgpr_store1 = wvgpr(vdst_reg + 1, lane, old1, exec_mask)
      return name, UOp.sink(UOp.sink(lds_store0, lds_store1, vgpr_store0, vgpr_store1).end(lane), inc_pc(), arg=KernelInfo(name=name))

    # DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64: exchange 64-bit at two addresses with stride64, return old values
    # Must use single loop to ensure addr reads happen before any vgpr writes
    if op_name == 'DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64':
      offset0 = getattr(inst, 'offset0', 0)
      offset1 = getattr(inst, 'offset1', 0)
      mult = 256
      data0_reg = inst.data0.offset - 256 if inst.data0.offset >= 256 else inst.data0.offset
      data1_reg = inst.data1.offset - 256 if inst.data1.offset >= 256 else inst.data1.offset
      vdst_reg = inst.vdst.offset - 256 if inst.vdst.offset >= 256 else inst.vdst.offset
      lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
      exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
      active = exec_bit.ne(UOp.const(dtypes.uint32, 0))
      base_addr = rvgpr(addr_reg, lane)
      # Read all addresses and old values first (4 dwords: 2 addresses × 2 dwords)
      addr0_lo = base_addr + UOp.const(dtypes.uint32, offset0 * mult)
      addr0_hi = base_addr + UOp.const(dtypes.uint32, offset0 * mult + 4)
      addr1_lo = base_addr + UOp.const(dtypes.uint32, offset1 * mult)
      addr1_hi = base_addr + UOp.const(dtypes.uint32, offset1 * mult + 4)
      old0_lo, old0_hi = rlds(addr0_lo), rlds(addr0_hi)
      old1_lo, old1_hi = rlds(addr1_lo), rlds(addr1_hi)
      new0_lo, new0_hi = rvgpr(data0_reg, lane), rvgpr(data0_reg + 1, lane)
      new1_lo, new1_hi = rvgpr(data1_reg, lane), rvgpr(data1_reg + 1, lane)
      # All stores in single sink
      stores = [wlds(addr0_lo, new0_lo, active), wlds(addr0_hi, new0_hi, active),
                wlds(addr1_lo, new1_lo, active), wlds(addr1_hi, new1_hi, active),
                wvgpr(vdst_reg, lane, old0_lo, exec_mask), wvgpr(vdst_reg + 1, lane, old0_hi, exec_mask),
                wvgpr(vdst_reg + 2, lane, old1_lo, exec_mask), wvgpr(vdst_reg + 3, lane, old1_hi, exec_mask)]
      return name, UOp.sink(UOp.sink(*stores).end(lane), inc_pc(), arg=KernelInfo(name=name))

    # Default: just increment PC (for unhandled DS ops)
    return name, UOp.sink(inc_pc(), arg=KernelInfo(name=name))

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
      if DEBUG >= 2:
        try:
          inst_str = repr(inst)
        except Exception:
          inst_str = f"<{type(inst).__name__} at PC={i//4}>"
        print(f"[emu2] PC={i//4}: {inst_str}\n{colored(prg.src, 'BLACK')}")
    except Exception as e:
      try:
        inst_str = repr(inst)
      except Exception:
        inst_str = f"<{type(inst).__name__}>"
      if DEBUG >= 1: print(f"[emu2] Failed to compile PC={i//4} {inst_str}: {type(e).__name__}: {e}")
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
