# RDNA3 emulator v2 - compiles pcode to UOps executed via tinygrad CPU backend
# Each instruction is compiled to a kernel that operates on buffers:
#   arg=0: vmem - base address 0, INDEX offsets directly to host memory
#   arg=1: lds - local data share
#   arg=2: vgpr - vgpr[reg * 32 + lane]
#   arg=3: sgpr - sgpr[reg], PC_LO=128, PC_HI=129, SCC=130
from __future__ import annotations
import ctypes, functools, re
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

# Inline float constants (as bit patterns) for GPU instructions
F32_INLINE = {240: 0x3f000000, 241: 0xbf000000, 242: 0x3f800000, 243: 0xbf800000,  # 0.5, -0.5, 1.0, -1.0
              244: 0x40000000, 245: 0xc0000000, 246: 0x40800000, 247: 0xc0800000, 248: 0x3e22f983}  # 2.0, -2.0, 4.0, -4.0, 1/(2*pi)
F64_INLINE = {240: 0x3fe0000000000000, 241: 0xbfe0000000000000, 242: 0x3ff0000000000000, 243: 0xbff0000000000000,
              244: 0x4000000000000000, 245: 0xc000000000000000, 246: 0x4010000000000000, 247: 0xc010000000000000, 248: 0x3fc45f306dc9c883}
F16_INLINE = {240: 0x3800, 241: 0xb800, 242: 0x3c00, 243: 0xbc00, 244: 0x4000, 245: 0xc000, 246: 0x4400, 247: 0xc400, 248: 0x3118}

def _u64(lo: UOp, hi: UOp) -> UOp:
  """Combine two 32-bit UOps into a 64-bit UOp."""
  return lo.cast(dtypes.uint64) | (hi.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))

def _split64(val: UOp) -> tuple[UOp, UOp]:
  """Split a 64-bit value into (lo, hi) 32-bit values."""
  v64 = val.bitcast(dtypes.uint64) if val.dtype == dtypes.float64 else val.cast(dtypes.uint64) if val.dtype != dtypes.uint64 else val
  return v64.cast(dtypes.uint32), (v64 >> UOp.const(dtypes.uint64, 32)).cast(dtypes.uint32)

def _apply_src_mods(val: UOp, mod_bit: int, abs_bits: int, neg_bits: int, is_16bit: bool = False, is_64bit: bool = False) -> UOp:
  """Apply abs/neg modifiers to source value based on operation type."""
  if not (abs_bits & (1 << mod_bit)) and not (neg_bits & (1 << mod_bit)): return val
  if is_16bit:
    f16_val = val.cast(dtypes.uint16).bitcast(dtypes.half)
    if abs_bits & (1 << mod_bit): f16_val = (f16_val.bitcast(dtypes.uint16) & UOp.const(dtypes.uint16, 0x7FFF)).bitcast(dtypes.half)
    if neg_bits & (1 << mod_bit): f16_val = f16_val.neg()
    return f16_val.bitcast(dtypes.uint16).cast(dtypes.uint32)
  if is_64bit:
    if val.dtype == dtypes.uint64: val = val.bitcast(dtypes.float64)
    if abs_bits & (1 << mod_bit): val = (val.bitcast(dtypes.uint64) & UOp.const(dtypes.uint64, 0x7FFFFFFFFFFFFFFF)).bitcast(dtypes.float64)
    if neg_bits & (1 << mod_bit): val = val.neg()
    return val.bitcast(dtypes.uint64)
  if val.dtype == dtypes.uint32: val = val.bitcast(dtypes.float32)
  if abs_bits & (1 << mod_bit): val = (val.bitcast(dtypes.uint32) & UOp.const(dtypes.uint32, 0x7FFFFFFF)).bitcast(dtypes.float32)
  if neg_bits & (1 << mod_bit): val = val.neg()
  return val.bitcast(dtypes.uint32)

# Map VOPD ops to VOP2 ops for pcode lookup
VOPD_TO_VOP2 = {
  VOPDOp.V_DUAL_FMAC_F32: VOP2Op.V_FMAC_F32_E32, VOPDOp.V_DUAL_MUL_F32: VOP2Op.V_MUL_F32_E32,
  VOPDOp.V_DUAL_ADD_F32: VOP2Op.V_ADD_F32_E32, VOPDOp.V_DUAL_SUB_F32: VOP2Op.V_SUB_F32_E32,
  VOPDOp.V_DUAL_SUBREV_F32: VOP2Op.V_SUBREV_F32_E32, VOPDOp.V_DUAL_MAX_F32: VOP2Op.V_MAX_F32_E32,
  VOPDOp.V_DUAL_MIN_F32: VOP2Op.V_MIN_F32_E32, VOPDOp.V_DUAL_ADD_NC_U32: VOP2Op.V_ADD_NC_U32_E32,
  VOPDOp.V_DUAL_LSHLREV_B32: VOP2Op.V_LSHLREV_B32_E32, VOPDOp.V_DUAL_AND_B32: VOP2Op.V_AND_B32_E32,
  VOPDOp.V_DUAL_MOV_B32: VOP1Op.V_MOV_B32_E32, VOPDOp.V_DUAL_CNDMASK_B32: VOP2Op.V_CNDMASK_B32_E32,
  VOPDOp.V_DUAL_FMAAK_F32: VOP2Op.V_FMAAK_F32_E32, VOPDOp.V_DUAL_FMAMK_F32: VOP2Op.V_FMAMK_F32_E32,
}
WAVE_SIZE = 32
PC_LO_IDX, PC_HI_IDX, SCC_IDX = 128, 129, 130
SGPR_COUNT, VGPR_SIZE = 131, 256 * 32

def _is_16bit_op(op_name: str) -> bool: return any(x in op_name for x in ('B16', 'F16', 'I16', 'U16'))
def _op_name(inst) -> str: return inst.op.name if hasattr(inst.op, 'name') else str(inst.op)
def _is_64bit_dest(dest: str) -> bool: return any(dest.endswith(x) for x in ('.b64', '.u64', '.i64', '.f64'))
def _to_u32(val: UOp) -> UOp: return val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val
def _lane_active(exec_mask: UOp, lane: UOp) -> UOp: return ((exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)).ne(UOp.const(dtypes.uint32, 0))
def _apply_opsel(val: UOp, sel_bit: int, opsel: int) -> UOp:
  return (val >> UOp.const(dtypes.uint32, 16)) & UOp.const(dtypes.uint32, 0xFFFF) if opsel & (1 << sel_bit) else val

def _set_lane_bit(old: UOp, lane: UOp, val: UOp, exec_mask: UOp) -> UOp:
  """Set/clear a single bit in a 32-bit mask based on lane index, respecting exec mask."""
  mask = UOp.const(dtypes.uint32, 1) << lane.cast(dtypes.uint32)
  new_bit = _to_u32(val) << lane.cast(dtypes.uint32)
  cleared = old & (mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))
  return _lane_active(exec_mask, lane).where(cleared | new_bit, old)

def _val_to_u32(val: UOp) -> UOp:
  """Convert any value to uint32 for storage (bitcast floats, cast ints)."""
  if val.dtype == dtypes.uint32: return val
  if val.dtype == dtypes.float32: return val.bitcast(dtypes.uint32)
  if val.dtype == dtypes.half: return val.bitcast(dtypes.uint16).cast(dtypes.uint32)
  if val.dtype in (dtypes.uint16, dtypes.int16): return val.cast(dtypes.uint32)
  return val.cast(dtypes.uint32)

def _write_64bit(val: UOp, wfn, reg_or_addr, *args) -> list[UOp]:
  """Write a 64-bit value as two 32-bit writes. args passed to wfn after reg/addr and lo/hi value."""
  lo, hi = _split64(val)
  incr = 4 if isinstance(reg_or_addr, UOp) else 1  # 4 bytes for memory addresses, 1 for register indices
  return [wfn(reg_or_addr, lo, *args), wfn(reg_or_addr + (UOp.const(reg_or_addr.dtype, incr) if isinstance(reg_or_addr, UOp) else incr), hi, *args)]

def _write_val(dest: str, val: UOp, wfn, reg_or_addr, *args) -> list[UOp]:
  """Write value, splitting 64-bit if needed based on dest type suffix."""
  return _write_64bit(val, wfn, reg_or_addr, *args) if _is_64bit_dest(dest) else [wfn(reg_or_addr, _to_u32(val), *args)]

def _mem_store(mem: UOp, addr: UOp, val: UOp, active: UOp, addr_bits: int = 32) -> UOp:
  """Conditional memory store: write val to mem[addr] if active, else keep old value."""
  shift = UOp.const(dtypes.uint64 if addr_bits == 64 else dtypes.uint32, 2)
  idx = mem.index((addr >> shift).cast(dtypes.index))
  return idx.store(active.where(_to_u32(val), idx.load()))

def _collect_data_slices(assigns: list, data_prefix: str, pcode_vars: dict = None, op_name: str = "") -> dict[int, UOp]:
  """Collect bit slices from assigns into {dword_idx: value} dict."""
  slices = {}
  for dest, val in assigns:
    if dest.startswith(f'{data_prefix}['):
      if (m := re.match(rf'{data_prefix}\[(\d+)\s*:\s*(\d+)\]', dest)):
        low_bit, dword_idx = int(m.group(2)), int(m.group(2)) // 32
        # D16 loads preserve bits - use final value from pcode_vars
        if pcode_vars and 'D16' in op_name and dword_idx == 0 and low_bit > 0:
          slices[0] = _to_u32(pcode_vars.get(data_prefix, val))
        else: slices[dword_idx] = _to_u32(val)
    elif dest.startswith(data_prefix): slices[0] = _to_u32(val)
  return slices

def _scalar_stores(assigns: list, wsgpr, sdst_reg: int, sdst_size: int = 1) -> list[UOp]:
  """Generate stores for scalar assigns (D0, SCC, EXEC, VCC)."""
  stores = []
  for dest, val in assigns:
    if dest.startswith('D0'):
      if sdst_size == 2:
        lo, hi = _split64(val)
        stores.extend([wsgpr(sdst_reg, lo), wsgpr(sdst_reg + 1, hi)])
      else: stores.append(wsgpr(sdst_reg, _to_u32(val)))
    elif dest.startswith('SCC'): stores.append(wsgpr(SCC_IDX, _to_u32(val)))
    elif dest.startswith('EXEC'):
      if val.dtype in (dtypes.uint64, dtypes.int64):
        lo, hi = _split64(val)
        stores.extend([wsgpr(EXEC_LO.offset, lo), wsgpr(EXEC_HI.offset, hi)])
      else: stores.append(wsgpr(EXEC_LO.offset, _to_u32(val)))
    elif dest.startswith('VCC'):
      if val.dtype in (dtypes.uint64, dtypes.int64):
        lo, hi = _split64(val)
        stores.extend([wsgpr(VCC_LO.offset, lo), wsgpr(VCC_HI.offset, hi)])
      else: stores.append(wsgpr(VCC_LO.offset, _to_u32(val)))
  return stores

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
  srcs.update({'VCC': rsgpr_fn(VCC_LO.offset), 'EXEC': rsgpr_fn(EXEC_LO.offset), 'SCC': rsgpr_fn(SCC_IDX), 'D0': rsgpr_fn(sdst_reg)})
  _, assigns = parse_pcode(pcode, srcs, lane=None)
  stores = _scalar_stores(assigns, wsgpr_fn, sdst_reg, sdst_size)
  if not stores: return None
  return name, UOp.sink(*stores, inc_pc_fn(), arg=KernelInfo(name=name))

def compile_lane_pcode(op, inst, vgpr, wsgpr_fn, rsgpr_fn, rsrc_fn, inc_pc_fn, name: str):
  """Compile READLANE/READFIRSTLANE/WRITELANE using pcode parser."""
  pcode = PCODE.get(op)
  if pcode is None: return None

  op_name = op.name if hasattr(op, 'name') else str(op)
  src0_reg = (inst.src0.offset - 256) if inst.src0.offset >= 256 else 0
  vdst_reg = (inst.vdst.offset - 256) if inst.vdst.offset >= 256 else inst.vdst.offset
  lane0 = UOp.const(dtypes.index, 0)

  # S0 = scalar value for WRITELANE, register index for others
  # S1 = lane select for READLANE/WRITELANE
  # VDST = vgpr register index for WRITELANE
  srcs = {
    'SRC0': UOp.const(dtypes.uint32, src0_reg),
    'S0': rsrc_fn(inst.src0.offset, lane0) if 'WRITELANE' in op_name else UOp.const(dtypes.uint32, src0_reg),
    'S1': rsrc_fn(inst.src1.offset, lane0) if hasattr(inst, 'src1') and inst.src1 is not None else UOp.const(dtypes.uint32, 0),
    'VDST': UOp.const(dtypes.uint32, vdst_reg),
    'EXEC_LO': rsgpr_fn(EXEC_LO.offset),
    '_vgpr': vgpr,
  }
  _, assigns = parse_pcode(pcode, srcs, lane=None)

  stores = []
  for dest, val in assigns:
    if dest.startswith('D0'):
      stores.append(wsgpr_fn(inst.vdst.offset, val.cast(dtypes.uint32)))
    elif dest.startswith('VGPR['):
      idx, write_val = val
      stores.append(vgpr.index(idx.cast(dtypes.index)).store(write_val.cast(dtypes.uint32)))

  if not stores: return None
  return name, UOp.sink(*stores, inc_pc_fn(), arg=KernelInfo(name=name))

def compile_vop_pcode_stores(op, srcs: dict[str, UOp], lane: UOp, wvgpr_fn, wsgpr_fn, rsgpr_fn, vdst_reg: int, exec_mask: UOp,
                             opsel_dst_hi: bool = False, rvgpr_fn = None, sdst_reg: int | None = None) -> list[UOp] | None:
  """Compile a VOP instruction using pcode parser. Returns list of store UOps or None.
  If sdst_reg is provided (for VOP3SD), VCC reads/writes are redirected to that register."""
  pcode = PCODE.get(op)
  if pcode is None: return None

  # For VOP3SD, use sdst for VCC (carry in/out), otherwise use actual VCC
  vcc_reg = sdst_reg if sdst_reg is not None else VCC_LO.offset

  # Parse pcode with source operands (VCC may already be set by caller for CNDMASK)
  if 'VCC' not in srcs: srcs['VCC'] = rsgpr_fn(vcc_reg)
  srcs['EXEC'] = exec_mask
  srcs['SCC'] = rsgpr_fn(SCC_IDX)
  _, assigns = parse_pcode(pcode, srcs, lane, op_name=op.name)

  # First pass: collect all stores without ending the loop yet
  raw_stores = []
  has_vgpr_write = False
  for dest, val in assigns:
    if 'D0' in dest and '[laneId]' in dest:
      # D0.u64[laneId] means VCC bit write (for VOPC instructions)
      raw_stores.append(('vcc', wsgpr_fn(VCC_LO.offset, _set_lane_bit(rsgpr_fn(VCC_LO.offset), lane, val, exec_mask))))
    elif dest.startswith('D0'):
      # Vector destination - write to vdst (bitcast floats to preserve bit pattern)
      has_vgpr_write = True
      # Check for bit slice assignment: D0[31:16].f16 or D0[15:0].f16 (but NOT D0[31:0] which is full width)
      if (slice_match := re.match(r'D0\[(\d+)\s*:\s*(\d+)\]', dest)):
        hi_bit, lo_bit = int(slice_match.group(1)), int(slice_match.group(2))
        if hi_bit != 31 or lo_bit != 0:
          # Partial slice - convert value to bits at correct position
          width = hi_bit - lo_bit + 1
          slice_mask = (1 << width) - 1
          if val.dtype == dtypes.half:
            val_bits = val.bitcast(dtypes.uint16).cast(dtypes.uint32)
          elif val.dtype in (dtypes.uint16, dtypes.int16):
            val_bits = val.cast(dtypes.uint32)
          else:
            val_bits = val.cast(dtypes.uint32) & UOp.const(dtypes.uint32, slice_mask)
          # Store with bit position and width (accumulated later)
          raw_stores.append(('vgpr_slice', (lo_bit, width, val_bits)))
          continue
      # Full 32-bit write (either D0, D0.type, or D0[31:0].type)
      if val.dtype in (dtypes.uint64, dtypes.int64, dtypes.float64):
        lo, hi = _split64(val)
        raw_stores.extend([('vgpr', wvgpr_fn(vdst_reg, lane, lo, exec_mask)), ('vgpr', wvgpr_fn(vdst_reg + 1, lane, hi, exec_mask))])
      elif val.dtype in (dtypes.half, dtypes.uint16, dtypes.int16) and rvgpr_fn is not None:
        # 16-bit with read-modify-write
        result, old_val = _val_to_u32(val), rvgpr_fn(vdst_reg, lane)
        if opsel_dst_hi: result = (old_val & UOp.const(dtypes.uint32, 0xFFFF)) | (result << UOp.const(dtypes.uint32, 16))
        else: result = (old_val & UOp.const(dtypes.uint32, 0xFFFF0000)) | (result & UOp.const(dtypes.uint32, 0xFFFF))
        raw_stores.append(('vgpr', wvgpr_fn(vdst_reg, lane, result, exec_mask)))
      else:
        raw_stores.append(('vgpr', wvgpr_fn(vdst_reg, lane, _val_to_u32(val), exec_mask)))
    elif dest.startswith('VCC'):
      raw_stores.append(('vcc', wsgpr_fn(vcc_reg, _set_lane_bit(rsgpr_fn(vcc_reg), lane, val, exec_mask))))
    elif dest.startswith('EXEC'):
      raw_stores.append(('exec', wsgpr_fn(EXEC_LO.offset, _set_lane_bit(rsgpr_fn(EXEC_LO.offset), lane, val, exec_mask))))
    elif dest.startswith('SCC'):
      raw_stores.append(('scc', wsgpr_fn(SCC_IDX, _to_u32(val))))

  # Second pass: combine all lane-dependent stores into a single END
  stores = []
  lane_stores = [s for t, s in raw_stores if t in ('vgpr', 'vcc', 'exec')]
  scalar_stores = [s for t, s in raw_stores if t == 'scc']
  # Handle bit slice stores: combine them into a single 32-bit write, preserving unwritten bits
  slice_stores = [s for t, s in raw_stores if t == 'vgpr_slice']
  if slice_stores:
    # Start with old value, mask out written bits, OR in new bits
    result = rvgpr_fn(vdst_reg, lane) if rvgpr_fn else UOp.const(dtypes.uint32, 0)
    for lo_bit, width, val_bits in slice_stores:
      mask = UOp.const(dtypes.uint32, ((1 << width) - 1) << lo_bit)
      result = (result & (mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))) | (val_bits << UOp.const(dtypes.uint32, lo_bit))
    lane_stores.append(wvgpr_fn(vdst_reg, lane, result, exec_mask))
  if lane_stores:
    # Combine all lane-dependent stores and end with single END
    combined = UOp.sink(*lane_stores)
    stores.append(combined.end(lane))
  stores.extend(scalar_stores)

  return stores if stores else None

def compile_vop_pcode(op, srcs: dict[str, UOp], lane: UOp, wvgpr_fn, wsgpr_fn, rsgpr_fn, vdst_reg: int, exec_mask: UOp, inc_pc_fn, name: str,
                      opsel_dst_hi: bool = False, rvgpr_fn = None, sdst_reg: int | None = None):
  """Try to compile a VOP instruction using pcode parser. Returns (name, sink) or None.
  If sdst_reg is provided (for VOP3SD), VCC reads/writes are redirected to that register."""
  stores = compile_vop_pcode_stores(op, srcs, lane, wvgpr_fn, wsgpr_fn, rsgpr_fn, vdst_reg, exec_mask, opsel_dst_hi, rvgpr_fn, sdst_reg)
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
  data = _u64(rvgpr_fn(data0_reg, lane), rvgpr_fn(data0_reg + 1, lane) if data0_reg else UOp.const(dtypes.uint32, 0))
  data2 = _u64(rvgpr_fn(data1_reg, lane) if data1_reg else UOp.const(dtypes.uint32, 0), rvgpr_fn(data1_reg + 1, lane) if data1_reg else UOp.const(dtypes.uint32, 0))

  srcs = {
    'ADDR': base_addr, 'ADDR_BASE': base_addr,
    'OFFSET': UOp.const(dtypes.uint32, offset0), 'OFFSET0': UOp.const(dtypes.uint32, offset0), 'OFFSET1': UOp.const(dtypes.uint32, offset1),
    'DATA': data, 'DATA2': data2,
    '_lds': lds,
  }

  _, assigns = parse_pcode(pcode, srcs, lane)

  active = _lane_active(exec_mask, lane)
  stores = [_mem_store(lds, val[0].cast(dtypes.uint32), val[1], active) for dest, val in assigns if dest.startswith('MEM[')]
  for dword_idx, val in sorted(_collect_data_slices(assigns, 'RETURN_DATA').items()):
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
  # Literal position depends on instruction type: 4-byte base (VOP1/VOP2/VOPC/SOP*) vs 8-byte base (VOP3/VOP3P/etc)
  is_8byte_base = isinstance(inst, (VOP3, VOP3SD, VOP3P, SMEM, DS, FLAT, GLOBAL, VOPD))
  lit_off = 8 if is_8byte_base else 4
  literal = int.from_bytes(inst_bytes[lit_off:lit_off+4], 'little') if len(inst_bytes) >= lit_off + 4 else 0

  # Helper: read SGPR
  def rsgpr(reg: int) -> UOp: return sgpr.index(UOp.const(dtypes.index, reg))
  def rsgpr64(off: int) -> UOp:
    if off >= 128:  # inline constant
      if off < 193: return UOp.const(dtypes.uint64, off - 128)  # 0-64
      if off < 209: return UOp.const(dtypes.int64, -(off - 192)).cast(dtypes.uint64)  # -1 to -16
      if off == 255: return UOp.const(dtypes.uint64, literal)  # literal constant
      return UOp.const(dtypes.uint64, 0)  # other inline constants
    return _u64(rsgpr(off), rsgpr(off + 1))
  def wsgpr(reg: int, val: UOp) -> UOp: return sgpr.index(UOp.const(dtypes.index, reg)).store(val.cast(dtypes.uint32))

  # Helper: read VGPR
  def rvgpr(reg: int, lane: UOp) -> UOp: return vgpr.index(UOp.const(dtypes.index, reg * 32) + lane.cast(dtypes.index))
  def wvgpr(reg: int, lane: UOp, val: UOp, exec_mask: UOp, after: UOp|None = None) -> UOp:
    buf = vgpr.after(after) if after is not None else vgpr
    idx = buf.index(UOp.const(dtypes.index, reg * 32) + lane.cast(dtypes.index))
    return idx.store(_lane_active(exec_mask, lane).where(val.cast(dtypes.uint32), idx.load()))

  # Helper: read source operand (32-bit, 64-bit with F64 inline, or 16-bit with F16 inline)
  def rsrc(off: int, lane: UOp, bits: int = 32) -> UOp:
    if bits == 64:
      if off in F64_INLINE: return UOp.const(dtypes.uint64, F64_INLINE[off])
      if 128 <= off < 256:
        if off < 193: return UOp.const(dtypes.uint64, off - 128)
        if off < 209: return UOp.const(dtypes.int64, -(off - 192)).cast(dtypes.uint64)
        if off == 255: return UOp.const(dtypes.uint64, literal) << UOp.const(dtypes.uint64, 32)  # literal is high 32 bits
      if off < 128: return _u64(rsgpr(off), rsgpr(off + 1))
      return _u64(rvgpr(off - 256, lane), rvgpr(off - 255, lane))
    if bits == 16 and off in F16_INLINE: return UOp.const(dtypes.uint32, F16_INLINE[off])
    if off < 128: return rsgpr(off)
    if off == 253: return rsgpr(SCC_IDX)
    if off == 255: return UOp.const(dtypes.uint32, literal)
    if off < 255:  # inline constants
      if off < 193: return UOp.const(dtypes.uint32, off - 128)  # 0-64
      if off < 209: return UOp.const(dtypes.uint32, (-(off - 192)) & MASK32)  # -1 to -16
      if off in F32_INLINE: return UOp.const(dtypes.uint32, F32_INLINE[off])
      return UOp.const(dtypes.uint32, 0)  # other inline
    return rvgpr(off - 256, lane)

  def rsrc_sized(off: int, lane: UOp, sizes: dict, key: str, f16: bool = False) -> UOp:
    """Read source with size from operand metadata."""
    return rsrc(off, lane, 64) if sizes.get(key, 1) == 2 else rsrc(off, lane, 16 if f16 else 32)

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

    # Branch instructions - use pcode parsing
    # NOTE: pcode uses byte offsets, but our PC is in 4-byte words, so we convert
    pcode = PCODE.get(inst.op)
    if pcode is not None:
      pc_words = rsgpr(PC_LO_IDX)
      pc_bytes = pc_words.cast(dtypes.int64) * UOp.const(dtypes.int64, 4)  # Convert to bytes for pcode
      vcc = rsgpr(VCC_LO.offset)
      exec_lo = rsgpr(EXEC_LO.offset)
      srcs = {
        'PC': pc_bytes,
        'SIMM16': UOp.const(dtypes.int16, _sext(inst.simm16, 16)),
        'SCC': rsgpr(SCC_IDX),
        'VCC': vcc,
        'VCCZ': vcc.eq(UOp.const(dtypes.uint32, 0)).cast(dtypes.uint32),
        'EXECZ': exec_lo.eq(UOp.const(dtypes.uint32, 0)).cast(dtypes.uint32),
      }
      _, assigns = parse_pcode(pcode, srcs, op_name=inst.op.name)
      for dest, val in assigns:
        if dest == 'PC' or dest.startswith('PC.'):
          # Convert back from bytes to words using integer right shift
          pc_new = val >> UOp.const(dtypes.int64, 2)
          return name, UOp.sink(wsgpr(PC_LO_IDX, pc_new.cast(dtypes.uint32)), arg=KernelInfo(name=name))

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
      s0 = rsgpr64(src_off)

    # Try pcode for SOP1 ops
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
  # SOPK: s_movk_i32, s_cmpk_*, s_addk_i32, etc. - 16-bit immediate
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, SOPK):
    sdst_reg = inst.sdst.offset
    simm16 = inst.simm16
    # Sign-extend SIMM16 to 32-bit for signed operations
    simm16_sext = simm16 if simm16 < 0x8000 else simm16 - 0x10000
    s0 = rsgpr(sdst_reg)
    srcs = {'S0': s0, 'SIMM16': UOp.const(dtypes.int32, simm16_sext), 'D0': s0}
    pcode_result = compile_sop_pcode(inst.op, srcs, wsgpr, rsgpr, sdst_reg, 1, inc_pc, name)
    assert pcode_result is not None, f"no pcode for SOPK: {inst.op.name}"
    return pcode_result

  # ═══════════════════════════════════════════════════════════════════════════
  # VOP1: v_mov_b32, etc.
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOP1):
    exec_mask = rsgpr(EXEC_LO.offset)
    op_name = _op_name(inst)

    # READFIRSTLANE: use pcode
    if op_name == 'V_READFIRSTLANE_B32_E32':
      pcode_result = compile_lane_pcode(inst.op, inst, vgpr, wsgpr, rsgpr, rsrc, inc_pc, name)
      assert pcode_result is not None, f"no pcode for VOP1: {op_name}"
      return pcode_result

    lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    sizes = inst.op_regs if hasattr(inst, 'op_regs') else {}
    src0 = rsrc_sized(inst.src0.offset, lane, sizes, 'src0')
    vdst_reg = inst.vdst.offset - 256

    # 16-bit ops: vdst_reg >= 128 means write to high half of vgpr[vdst_reg-128]
    is_16bit_op = _is_16bit_op(op_name)
    write_hi_half = is_16bit_op and vdst_reg >= 128
    if write_hi_half:
      vdst_reg = vdst_reg - 128

    pcode_result = compile_vop_pcode(inst.op, {'S0': src0}, lane, wvgpr, wsgpr, rsgpr, vdst_reg, exec_mask, inc_pc, name,
                                     opsel_dst_hi=write_hi_half, rvgpr_fn=rvgpr)
    assert pcode_result is not None, f"no pcode for VOP1: {inst.op.name}"
    return pcode_result

  # ═══════════════════════════════════════════════════════════════════════════
  # VOPC: vector compare, writes to VCC (or EXEC for CMPX)
  # Uses unrolled computation to avoid loop-carried VCC dependency issues
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOPC):
    exec_mask = rsgpr(EXEC_LO.offset)
    old_vcc = rsgpr(VCC_LO.offset)
    op_name = _op_name(inst)
    is_cmpx = 'CMPX' in op_name
    is_16bit = _is_16bit_op(op_name)

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
      # Find the comparison result assignment (D0 for CMP, EXEC for CMPX)
      for dest, val in assigns:
        if '[laneId]' in dest and ('D0' in dest or 'EXEC' in dest):
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
      stores = [wsgpr(EXEC_LO.offset, new_vcc)]  # CMPX E32 only writes EXEC, not VCC
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
    op_name = _op_name(inst)
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
    op_name = _op_name(inst)

    # Lane operations: READLANE, READFIRSTLANE, WRITELANE - use pcode
    if op_name in ('V_READLANE_B32', 'V_READFIRSTLANE_B32', 'V_READFIRSTLANE_B32_E64', 'V_WRITELANE_B32'):
      pcode_result = compile_lane_pcode(inst.op, inst, vgpr, wsgpr, rsgpr, rsrc, inc_pc, name)
      assert pcode_result is not None, f"no pcode for VOP3: {op_name}"
      return pcode_result

    # Check if this is a VOP3 VOPC instruction (v_cmp_*_e64) - these write to scalar dest
    # VOP3 VOPC: writes to D0 (scalar dest), uses unrolled computation like VOPC
    is_vop3_vopc = 'V_CMP' in op_name or 'V_CMPX' in op_name
    if is_vop3_vopc:
      old_sdst = rsgpr(inst.vdst.offset)  # vdst is actually sdst for VOP3_SDST
      is_cmpx = 'CMPX' in op_name

      # Get abs/neg modifiers and operand sizes from instruction metadata
      abs_bits = getattr(inst, 'abs', 0) or 0
      neg_bits = getattr(inst, 'neg', 0) or 0
      operands = inst.operands
      src0_fmt, src0_bits, _ = operands.get('src0', (None, 32, None))
      src1_fmt, src1_bits, _ = operands.get('src1', (None, 32, None))
      is_16bit_op = src0_bits == 16 or src1_bits == 16
      is_64bit_op = src0_bits == 64 or src1_bits == 64
      is_float_op = src0_fmt is not None and any(x in src0_fmt.name for x in ('F32', 'F64', 'F16'))

      # Helper to get comparison result for a lane
      def get_cmp_result_vop3(lane_idx: int) -> UOp:
        lc = UOp.const(dtypes.index, lane_idx)
        s0 = rsrc(inst.src0.offset, lc, src0_bits)
        s1 = rsrc(inst.src1.offset, lc, src1_bits)
        if is_16bit_op: s0, s1 = _apply_opsel(s0, 0, opsel), _apply_opsel(s1, 1, opsel)
        if is_float_op:
          s0, s1 = _apply_src_mods(s0, 0, abs_bits, neg_bits, is_16bit_op, src0_bits == 64), _apply_src_mods(s1, 1, abs_bits, neg_bits, is_16bit_op, src1_bits == 64)
        pcode = PCODE.get(inst.op)
        if pcode is None: return UOp.const(dtypes.uint32, 0)
        _, assigns = parse_pcode(pcode, {'S0': s0, 'S1': s1}, lane=UOp.const(dtypes.uint32, lane_idx))
        for dest, val in assigns:
          if 'D0' in dest and '[laneId]' in dest:
            return val.cast(dtypes.uint32)
        return UOp.const(dtypes.uint32, 0)

      # Compute all 32 bits by unrolling - VOP3 VOPC writes comparison bits for ALL lanes
      # EXEC mask affects which lanes participate (inactive lanes are 0)
      exec_mask = rsgpr(EXEC_LO.offset)
      new_sdst = UOp.const(dtypes.uint32, 0)
      for i in range(32):
        cmp_result = get_cmp_result_vop3(i)
        bit = cmp_result << UOp.const(dtypes.uint32, i)
        new_sdst = new_sdst | bit
      # Apply EXEC mask - inactive lanes are 0
      new_sdst = new_sdst & exec_mask

      # Store to scalar destination (and EXEC for CMPX)
      if is_cmpx:
        stores = [wsgpr(EXEC_LO.offset, new_sdst), wsgpr(inst.vdst.offset, new_sdst)]
      else:
        stores = [wsgpr(inst.vdst.offset, new_sdst)]

      return name, UOp.sink(*stores, inc_pc(), arg=KernelInfo(name=name))

    # Regular VOP3 handling
    lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    is_f16_op = 'F16' in op_name
    src0 = rsrc_sized(inst.src0.offset, lane, sizes, 'src0', is_f16_op)
    src1 = rsrc_sized(inst.src1.offset, lane, sizes, 'src1', is_f16_op)
    src2 = rsrc_sized(inst.src2.offset, lane, sizes, 'src2', is_f16_op) if inst.src2 is not None else None

    # Apply opsel to 16-bit operations (F16, etc.)
    if _is_16bit_op(op_name):
      src0, src1 = _apply_opsel(src0, 0, opsel), _apply_opsel(src1, 1, opsel)
      if src2 is not None: src2 = _apply_opsel(src2, 2, opsel)

    # Apply abs/neg modifiers
    abs_bits = getattr(inst, 'abs', 0) or 0
    neg_bits = getattr(inst, 'neg', 0) or 0

    is_16bit_op = _is_16bit_op(op_name)
    if abs_bits or neg_bits:
      src0 = _apply_src_mods(src0, 0, abs_bits, neg_bits, is_16bit_op, sizes.get('src0', 1) == 2)
      if src1 is not None: src1 = _apply_src_mods(src1, 1, abs_bits, neg_bits, is_16bit_op, sizes.get('src1', 1) == 2)
      if src2 is not None: src2 = _apply_src_mods(src2, 2, abs_bits, neg_bits, is_16bit_op, sizes.get('src2', 1) == 2)

    vdst_reg = inst.vdst.offset - 256
    srcs = {'S0': src0, 'S1': src1}
    if src2 is not None: srcs['S2'] = src2
    # For V_CNDMASK, src2 is the condition mask (used as VCC in pcode)
    if 'CNDMASK' in op_name and src2 is not None: srcs['VCC'] = src2

    # For 16-bit ops with opsel[3]=1, write to high half
    opsel_dst_hi = bool(opsel & 0b1000) and _is_16bit_op(op_name)
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
  # Carry operations use unrolled computation for sdst to avoid loop issues
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOP3SD):
    exec_mask = rsgpr(EXEC_LO.offset)
    sizes = inst.op_regs if hasattr(inst, 'op_regs') else {}
    sdst_reg = inst.sdst.offset
    op_name = _op_name(inst)
    vdst_reg = inst.vdst.offset - 256

    # Parse pcode to check if VCC is per-lane (VCC[laneId]) or scalar (VCC)
    pcode = PCODE.get(inst.op)
    assert pcode is not None, f"no pcode for VOP3SD: {op_name}"

    # Check if any VCC assignment is per-lane by looking for [laneId] in destination
    lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    src0, src1 = rsrc_sized(inst.src0.offset, lane, sizes, 'src0'), rsrc_sized(inst.src1.offset, lane, sizes, 'src1')
    src2 = rsrc_sized(inst.src2.offset, lane, sizes, 'src2') if inst.src2 is not None else None
    srcs = {'S0': src0, 'S1': src1, 'VCC': rsgpr(sdst_reg), 'EXEC': exec_mask, 'SCC': rsgpr(SCC_IDX)}
    if src2 is not None: srcs['S2'] = src2
    _, assigns = parse_pcode(pcode, srcs, lane, op_name=op_name)

    # Check if VCC is per-lane (needs unrolling) or scalar
    # D0.u64[laneId] is also per-lane VCC for comparison ops where D0 == sdst (VCC_LO)
    has_per_lane_vcc = any('[laneId]' in dest for dest, _ in assigns if dest.startswith('VCC') or dest.startswith('D0.u64'))

    if has_per_lane_vcc:
      # Per-lane VCC (e.g., V_ADD_CO_U32, V_CMP_CLASS_F32)
      # IMPORTANT: When src==dst, we must ensure VCC is computed from ORIGINAL source values, not updated values.
      # We fully unroll both VCC and D0 computation to avoid loop reordering issues.

      # Compute VCC bit and D0 value for each lane (unrolled)
      def get_lane_results(lane_idx: int) -> tuple[UOp, UOp|None]:
        lc = UOp.const(dtypes.index, lane_idx)
        s0, s1 = rsrc_sized(inst.src0.offset, lc, sizes, 'src0'), rsrc_sized(inst.src1.offset, lc, sizes, 'src1')
        s2 = rsrc_sized(inst.src2.offset, lc, sizes, 'src2') if inst.src2 is not None else None
        lane_srcs = {'S0': s0, 'S1': s1, 'VCC': rsgpr(sdst_reg), 'EXEC': exec_mask, 'SCC': rsgpr(SCC_IDX)}
        if s2 is not None: lane_srcs['S2'] = s2
        _, lane_assigns = parse_pcode(pcode, lane_srcs, UOp.const(dtypes.uint32, lane_idx), op_name=op_name)
        vcc_bit, d0_val = UOp.const(dtypes.uint32, 0), None
        for dest, val in lane_assigns:
          if dest.startswith('VCC') or (dest.startswith('D0.u64') and '[laneId]' in dest): vcc_bit = val.cast(dtypes.uint32)
          elif dest.startswith('D0') and '[laneId]' not in dest: d0_val = val
        return vcc_bit, d0_val

      # Compute all 32 lanes (this ensures reads happen before any writes through data dependencies)
      lane_results = [get_lane_results(i) for i in range(32)]

      # Combine VCC bits
      new_vcc = UOp.const(dtypes.uint32, 0)
      for i, (vcc_bit, _) in enumerate(lane_results):
        new_vcc = new_vcc | (vcc_bit << UOp.const(dtypes.uint32, i))

      # VOP3SD carry-out (sdst) is NOT EXEC-masked - all 32 bits are written
      # This matches hardware behavior where carry bits for inactive lanes are computed but zero
      final_vcc = new_vcc

      # Build vgpr stores (unrolled) - each store depends on final_vcc which depends on all source reads
      vgpr_stores = []
      for i, (_, d0_val) in enumerate(lane_results):
        if d0_val is None: continue
        lane_const = UOp.const(dtypes.index, i)
        exec_bit = (exec_mask >> UOp.const(dtypes.uint32, i)) & UOp.const(dtypes.uint32, 1)
        active = exec_bit.ne(UOp.const(dtypes.uint32, 0))
        if d0_val.dtype in (dtypes.uint64, dtypes.int64, dtypes.float64):
          lo, hi = _split64(d0_val)
          idx_lo, idx_hi = vgpr.after(final_vcc).index(UOp.const(dtypes.index, vdst_reg * 32 + i)), vgpr.after(final_vcc).index(UOp.const(dtypes.index, (vdst_reg + 1) * 32 + i))
          vgpr_stores.extend([idx_lo.store(active.where(lo, idx_lo.load())), idx_hi.store(active.where(hi, idx_hi.load()))])
        else:
          idx = vgpr.after(final_vcc).index(UOp.const(dtypes.index, vdst_reg * 32 + i))
          vgpr_stores.append(idx.store(active.where(d0_val.cast(dtypes.uint32), idx.load())))

      # Write VCC and vgpr
      vcc_write = wsgpr(sdst_reg, final_vcc)
      if vgpr_stores:
        return name, UOp.sink(*vgpr_stores, vcc_write, inc_pc(), arg=KernelInfo(name=name))
      else:
        return name, UOp.sink(vcc_write, inc_pc(), arg=KernelInfo(name=name))
    else:
      # Scalar VCC (e.g., V_DIV_SCALE_F32): use normal pcode handling with sdst redirection
      pcode_result = compile_vop_pcode(inst.op, srcs, lane, wvgpr, wsgpr, rsgpr, vdst_reg, exec_mask, inc_pc, name, sdst_reg=sdst_reg)
      assert pcode_result is not None, f"no pcode for VOP3SD: {op_name}"
      return pcode_result

  # ═══════════════════════════════════════════════════════════════════════════
  # VOP3P: packed 16-bit operations (v_pk_add_f16, v_pk_mul_f16, etc.)
  # Pcode uses S0[31:16], S0[15:0] etc. We remap halves based on opsel bits.
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOP3P):
    lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    exec_mask = rsgpr(EXEC_LO.offset)
    vdst_reg = inst.vdst.offset - 256

    # Read source operands (as uint32 with packed f16)
    src0 = rsrc(inst.src0.offset, lane, 16)
    src1 = rsrc(inst.src1.offset, lane, 16)
    src2 = rsrc(inst.src2.offset, lane, 16) if hasattr(inst, 'src2') and inst.src2 is not None else None

    # Get opsel bits for source selection
    opsel = getattr(inst, 'opsel', 0) or 0
    opsel_hi = getattr(inst, 'opsel_hi', 3) if getattr(inst, 'opsel_hi', 3) is not None else 3
    opsel_hi2 = getattr(inst, 'opsel_hi2', 1) if getattr(inst, 'opsel_hi2', 1) is not None else 1
    neg = getattr(inst, 'neg', 0) or 0
    neg_hi = getattr(inst, 'neg_hi', 0) or 0

    # Helper to extract 16-bit half as uint16 bits, then optionally negate as f16
    def get_half_bits(val: UOp, use_hi: bool, apply_neg: bool = False) -> UOp:
      bits = ((val >> UOp.const(dtypes.uint32, 16)) if use_hi else val) & UOp.const(dtypes.uint32, 0xFFFF)
      if apply_neg:
        f16_val = bits.cast(dtypes.uint16).bitcast(dtypes.half).neg()
        bits = f16_val.bitcast(dtypes.uint16).cast(dtypes.uint32)
      return bits

    # Build remapped sources: pcode expects [15:0] for lo, [31:16] for hi
    # opsel controls which actual half to put in each position
    # After remapping: S0_new[15:0] = selected lo half, S0_new[31:16] = selected hi half
    def build_remapped_src(src: UOp, opsel_lo_bit: int, opsel_hi_bit: int, neg_lo_bit: int, neg_hi_bit: int) -> UOp:
      lo_bits = get_half_bits(src, bool(opsel_lo_bit), bool(neg_lo_bit))
      hi_bits = get_half_bits(src, bool(opsel_hi_bit), bool(neg_hi_bit))
      return lo_bits | (hi_bits << UOp.const(dtypes.uint32, 16))

    s0_new = build_remapped_src(src0, opsel & 1, opsel_hi & 1, neg & 1, neg_hi & 1)
    s1_new = build_remapped_src(src1, opsel & 2, opsel_hi & 2, neg & 2, neg_hi & 2)
    s2_new = build_remapped_src(src2, opsel & 4, 1 if opsel_hi2 else 0, neg & 4, neg_hi & 4) if src2 is not None else None

    op_name = _op_name(inst)

    # WMMA: Wave Matrix Multiply-Accumulate - 16x16x16 matrix multiply across the wave
    if 'WMMA' in op_name and 'F32_16X16X16_F16' in op_name:
      src0, src1, src2 = inst.src0.offset - 256, inst.src1.offset - 256, inst.src2.offset - 256
      def f16_to_f32(bits: UOp) -> UOp: return bits.cast(dtypes.uint16).bitcast(dtypes.half).cast(dtypes.float32)
      def read_f16_mat(src):
        return [f for l in range(16) for r in range(8) for v in [rvgpr(src + r, UOp.const(dtypes.index, l))]
                for f in [f16_to_f32(v & UOp.const(dtypes.uint32, 0xFFFF)), f16_to_f32(v >> UOp.const(dtypes.uint32, 16))]]
      mat_a, mat_b = read_f16_mat(src0), read_f16_mat(src1)
      mat_c = [rvgpr(src2 + i // 32, UOp.const(dtypes.index, i % 32)).bitcast(dtypes.float32) for i in range(256)]
      mat_d = [sum(mat_a[row*16+k] * mat_b[col*16+k] for k in range(16)) + mat_c[row*16+col] for row in range(16) for col in range(16)]
      stores = [wvgpr(vdst_reg + i // 32, UOp.const(dtypes.index, i % 32), mat_d[i].bitcast(dtypes.uint32), exec_mask) for i in range(256)]
      return name, UOp.sink(*stores, inc_pc(), arg=KernelInfo(name=name))

    pcode = PCODE.get(inst.op)
    if pcode is not None:
      op_name = _op_name(inst)
      # For FMA_MIX ops, apply neg_hi by flipping sign bits, pass OPSEL_HI/OPSEL to pcode
      if 'FMA_MIX' in op_name:
        combined_opsel_hi = (opsel_hi & 0x3) | ((opsel_hi2 & 0x1) << 2)
        # neg_hi negates AFTER conversion: if f32 mode (opsel_hi=0) flip bit 31, if f16 mode flip bit 15 or 31 based on opsel
        def apply_neg(v, bit, opsel_hi_bit, opsel_bit):
          if not (neg_hi & bit): return v
          if not (combined_opsel_hi & opsel_hi_bit): return v ^ UOp.const(dtypes.uint32, 0x80000000)  # f32: flip bit 31
          if opsel & opsel_bit: return v ^ UOp.const(dtypes.uint32, 0x80000000)  # f16 hi: flip bit 31
          return v ^ UOp.const(dtypes.uint32, 0x00008000)  # f16 lo: flip bit 15
        srcs = {'S0': apply_neg(src0, 1, 1, 1), 'S1': apply_neg(src1, 2, 2, 2),
                'S2': apply_neg(src2, 4, 4, 4) if src2 is not None else UOp.const(dtypes.uint32, 0),
                'OPSEL_HI': UOp.const(dtypes.uint32, combined_opsel_hi), 'OPSEL': UOp.const(dtypes.uint32, opsel)}
      else:
        srcs = {'S0': s0_new, 'S1': s1_new}
        if s2_new is not None: srcs['S2'] = s2_new
      stores = compile_vop_pcode_stores(inst.op, srcs, lane, wvgpr, wsgpr, rsgpr, vdst_reg, exec_mask, rvgpr_fn=rvgpr)
      if stores is not None:
        return name, UOp.sink(*stores, inc_pc(), arg=KernelInfo(name=name))

    # No pcode or couldn't parse, skip
    return name, UOp.sink(inc_pc(), arg=KernelInfo(name=name))

  # ═══════════════════════════════════════════════════════════════════════════
  # VOPD: dual-issue v_dual_mov_b32, etc.
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOPD):
    exec_mask = rsgpr(EXEC_LO.offset)
    vdstx_reg = inst.vdstx.offset - 256
    vdsty_reg = (inst.vdsty << 1) | ((inst.vdstx.offset & 1) ^ 1)
    ended = []

    # Process X and Y operations
    for op, src0_off, vsrc1_off, vdst_reg, label in [
        (inst.opx, inst.srcx0.offset, inst.vsrcx1.offset, vdstx_reg, 'X'),
        (inst.opy, inst.srcy0.offset, inst.vsrcy1.offset, vdsty_reg, 'Y')]:
      lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
      vop = VOPD_TO_VOP2.get(op)
      assert vop is not None, f"no VOP mapping for VOPD {label}: {op}"
      srcs = {'S0': rsrc(src0_off, lane), 'S1': rvgpr(vsrc1_off - 256, lane), 'D0': rvgpr(vdst_reg, lane)}
      vop_name = vop.name if hasattr(vop, 'name') else str(vop)
      # FMAAK/FMAMK use inline literal constant (SIMM32)
      if 'FMAAK' in vop_name or 'FMAMK' in vop_name: srcs['SIMM32'] = UOp.const(dtypes.uint32, literal)
      # CNDMASK uses VCC as condition
      if 'CNDMASK' in vop_name: srcs['VCC'] = rsgpr(VCC_LO.offset)
      stores = compile_vop_pcode_stores(vop, srcs, lane, wvgpr, wsgpr, rsgpr, vdst_reg, exec_mask)
      assert stores is not None, f"no pcode for VOPD {label}: {vop}"
      ended.extend(stores)

    return name, UOp.sink(*ended, inc_pc(), arg=KernelInfo(name=name))

  # ═══════════════════════════════════════════════════════════════════════════
  # DS: Local Data Share (LDS) operations
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, DS):
    exec_mask = rsgpr(EXEC_LO.offset)
    op_name = _op_name(inst)
    addr_reg = inst.addr.offset - 256 if inst.addr.offset >= 256 else inst.addr.offset

    # LDS helper - reads/writes to lds buffer (uint32 indexed)
    def rlds(addr: UOp) -> UOp: return lds.index((addr >> UOp.const(dtypes.uint32, 2)).cast(dtypes.index))
    def wlds(addr: UOp, val: UOp, active: UOp) -> UOp:
      idx = lds.index((addr >> UOp.const(dtypes.uint32, 2)).cast(dtypes.index))
      return idx.store(active.where(val, idx.load()))

    # Get pcode for this DS instruction
    pcode = PCODE.get(inst.op)
    if pcode is None:
      return name, UOp.sink(inc_pc(), arg=KernelInfo(name=name))

    # Build source variables for pcode parsing
    offset0 = getattr(inst, 'offset0', 0) or 0
    offset1 = getattr(inst, 'offset1', 0) or 0
    offset = getattr(inst, 'offset', offset0) or offset0
    data0_reg = inst.data0.offset - 256 if hasattr(inst, 'data0') and inst.data0.offset >= 256 else (inst.data0.offset if hasattr(inst, 'data0') else 0)
    data1_reg = inst.data1.offset - 256 if hasattr(inst, 'data1') and inst.data1.offset >= 256 else (inst.data1.offset if hasattr(inst, 'data1') else 0)
    vdst_reg = inst.vdst.offset - 256 if hasattr(inst, 'vdst') and inst.vdst.offset >= 256 else (inst.vdst.offset if hasattr(inst, 'vdst') else 0)

    # Helper to build srcs dict for a given lane
    def make_srcs(lane: UOp) -> dict:
      base_addr = rvgpr(addr_reg, lane)
      return {
        'ADDR': base_addr, 'ADDR_BASE': base_addr,
        'OFFSET': UOp.const(dtypes.uint32, offset),
        'OFFSET0': UOp.const(dtypes.uint32, offset0),
        'OFFSET1': UOp.const(dtypes.uint32, offset1),
        'DATA': rvgpr(data0_reg, lane) if 'B32' in op_name else _u64(rvgpr(data0_reg, lane), rvgpr(data0_reg + 1, lane)),
        'DATA2': rvgpr(data1_reg, lane) if 'B32' in op_name else _u64(rvgpr(data1_reg, lane), rvgpr(data1_reg + 1, lane)),
        '_lds': lds,
      }

    # Parse once to get assignment structure
    dummy_lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    _, assigns = parse_pcode(pcode, make_srcs(dummy_lane), lane=dummy_lane, op_name=op_name)

    # Count distinct MEM addresses to determine if we need separate RANGEs
    # For 2ADDR ops, each MEM/RETURN_DATA uses a different address -> need separate RANGEs
    # For atomic ops, all operations share the same address -> use single RANGE
    # EXCEPTION: STOREXCHG ops need single RANGE to preserve read-before-write ordering
    mem_assigns = [d for d, _ in assigns if d.startswith('MEM[')]
    mem_addrs = set(re.match(r'MEM\[([^\]]+)\]', d).group(1) if re.match(r'MEM\[([^\]]+)\]', d) else d for d in mem_assigns)
    is_storexchg = 'STOREXCHG' in op_name
    use_separate_ranges = (len(mem_addrs) > 1 or '2ADDR' in op_name) and not is_storexchg

    def make_ds_stores(dest: str, val: UOp, lane: UOp, active: UOp, writes_return_data: bool = True) -> list[UOp]:
      """Generate store UOps for a DS assign (MEM or RETURN_DATA)."""
      if dest.startswith('MEM['):
        return _write_val(dest, val[1], wlds, val[0], active)
      if dest.startswith('RETURN_DATA') and writes_return_data:
        if (m := re.match(r'RETURN_DATA\[(\d+)\s*:\s*(\d+)\]', dest)):
          bit_width, dword_idx = int(m.group(1)) - int(m.group(2)) + 1, int(m.group(2)) // 32
          is_64 = '.b64' if bit_width == 64 else ''
          return _write_val(is_64, val, lambda r, v, l, e: wvgpr(r, l, v, e), vdst_reg + dword_idx, lane, exec_mask)
        return _write_val(dest, val, lambda r, v, l, e: wvgpr(r, l, v, e), vdst_reg, lane, exec_mask)
      return []

    if use_separate_ranges:
      ended = []
      for i, (dest, _) in enumerate(assigns):
        lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
        active = _lane_active(exec_mask, lane)
        _, lane_assigns = parse_pcode(pcode, make_srcs(lane), lane=lane, op_name=op_name)
        ended.extend(s.end(lane) for s in make_ds_stores(dest, lane_assigns[i][1], lane, active))
    else:
      writes_return_data = '_RTN' in op_name or op_name.startswith('DS_LOAD')
      lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
      active = _lane_active(exec_mask, lane)
      _, lane_assigns = parse_pcode(pcode, make_srcs(lane), lane=lane, op_name=op_name)
      stores = [s for dest, val in lane_assigns for s in make_ds_stores(dest, val, lane, active, writes_return_data)]
      ended = [UOp.sink(*stores).end(lane)] if stores else []

    return name, UOp.sink(*ended, inc_pc(), arg=KernelInfo(name=name)) if ended else (name, UOp.sink(inc_pc(), arg=KernelInfo(name=name)))

  # ═══════════════════════════════════════════════════════════════════════════
  # FLAT/GLOBAL: memory loads/stores
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, (FLAT, GLOBAL)):
    exec_mask = rsgpr(EXEC_LO.offset)
    addr_reg = inst.addr.offset - 256
    has_saddr = hasattr(inst, 'saddr') and inst.saddr != NULL and inst.saddr.offset < 128
    offset = _sext(getattr(inst, 'offset', 0), 13)
    op_name = _op_name(inst)
    ndwords = 4 if '_B128' in op_name else 3 if '_B96' in op_name else 2 if '_B64' in op_name else 1

    # Helper to compute address for a lane
    def make_addr(lane: UOp) -> UOp:
      if has_saddr:
        vgpr_offset = rvgpr(addr_reg, lane).cast(dtypes.uint64)
        saddr = rsgpr64(inst.saddr.offset)
        return saddr + vgpr_offset + UOp.const(dtypes.uint64, offset)
      else:
        return _u64(rvgpr(addr_reg, lane), rvgpr(addr_reg + 1, lane)) + UOp.const(dtypes.uint64, offset)

    pcode = PCODE.get(inst.op)
    if pcode is not None:
      vdata_reg = inst.data.offset - 256 if hasattr(inst, 'data') and inst.data else 0
      vdst_reg = inst.vdst.offset - 256 if hasattr(inst, 'vdst') and inst.vdst else vdata_reg
      is_atomic, glc = 'ATOMIC' in op_name, getattr(inst, 'glc', 0)
      is_64bit = '_B64' in op_name or '_U64' in op_name or '_I64' in op_name or '_F64' in op_name

      lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
      addr, active = make_addr(lane), _lane_active(exec_mask, lane)

      # Build source operands
      if is_atomic:
        srcs = {'ADDR': addr, 'DATA': _u64(rvgpr(vdata_reg, lane), rvgpr(vdata_reg + 1, lane)) if is_64bit else rvgpr(vdata_reg, lane), '_vmem': vmem}
      else:
        vdata = rvgpr(vdata_reg, lane).cast(dtypes.uint64) if 'STORE' in op_name else rvgpr(vdst_reg, lane) if 'D16' in op_name else UOp.const(dtypes.uint32, 0)
        if 'STORE' in op_name and ndwords >= 2: vdata = vdata | (rvgpr(vdata_reg + 1, lane).cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))
        srcs = {'ADDR': addr, 'VDATA': vdata, '_vmem': vmem}
        for i in range(ndwords): srcs[f'VDATA{i}'] = rvgpr(vdata_reg + i, lane) if 'STORE' in op_name else UOp.const(dtypes.uint32, 0)

      pcode_vars, assigns = parse_pcode(pcode, srcs, lane, op_name=op_name)

      # For atomics, use separate load to prevent CSE issues
      def wvmem(a: UOp, v: UOp, act: UOp) -> UOp:
        idx = vmem.index((a >> UOp.const(dtypes.uint64, 2)).cast(dtypes.index))
        return idx.store(act.where(v, vmem.index((a >> UOp.const(dtypes.uint64, 2)).cast(dtypes.index)).load()))

      stores = []
      for dest, val in assigns:
        if dest.startswith('MEM['):
          stores.extend(_write_val(dest, val[1], wvmem, val[0], active) if is_atomic else [_mem_store(vmem, val[0], val[1], active, 64)])
        elif dest.startswith('RETURN_DATA') and is_atomic and glc:
          stores.extend(_write_val(dest, val, lambda r, v, l, e: wvgpr(r, l, v, e), vdst_reg, lane, exec_mask))
      if not is_atomic:
        for dword_idx, val in sorted(_collect_data_slices(assigns, 'VDATA', pcode_vars, op_name).items()):
          stores.append(wvgpr(vdst_reg + dword_idx, lane, val, exec_mask))

      if stores:
        return name, UOp.sink(UOp.sink(*stores).end(lane), inc_pc(), arg=KernelInfo(name=name))

    return name, UOp.sink(inc_pc(), arg=KernelInfo(name=name))

  # Default: just increment PC
  return name, UOp.sink(inc_pc(), arg=KernelInfo(name=name))

def compile_inst(data: bytes) -> tuple[str, UOp]:
  inst = decode_inst(data)
  return _compile_inst_inner(bytes(data[:inst.size() + 4]))

# ═══════════════════════════════════════════════════════════════════════════════
# PROGRAM DECODE AND COMPILATION
# ═══════════════════════════════════════════════════════════════════════════════

@functools.cache
def decode_program(data: bytes) -> dict[int, tuple[str, ctypes.CFUNCTYPE|None, list[int]|None, CompiledRunner|None]]:
  """Decode program to {pc: (name, fxn, globals, runner)}. Runner is kept alive to prevent fxn memory from being freed."""
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
      fxn = runner._prg.fxn  # Extract raw ctypes function for direct calls (bypasses HCQ overhead)
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
      fxn, globals_list, runner = None, None, None

    result[i // 4] = (name, fxn, globals_list, runner)
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
    # Zero out all registers using ctypes memset (much faster than Python loop)
    ctypes.memset(ctypes.addressof(ctypes.c_uint32.from_buffer(self._sgpr_mv)), 0, SGPR_COUNT * 4)
    ctypes.memset(ctypes.addressof(ctypes.c_uint32.from_buffer(self._vgpr_mv)), 0, VGPR_SIZE * 4)
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
          # Pre-compute buffer addresses for direct ctypes calls (avoids per-instruction Buffer overhead)
          buf_addrs = {0: st.sgpr_buf._buf.va_addr, 1: st.vgpr_buf._buf.va_addr, 2: vmem_buf._buf.va_addr, 3: lds_buf._buf.va_addr}
          max_instructions = 100000
          inst_count = 0
          while True:
            pc = st.pc
            if pc == 0xFFFFFFFF or pc not in program: break

            name, fxn, globals_list, _runner = program[pc]
            if fxn is None:
              if DEBUG >= 1: print(f"[emu2] No fxn for {name} at PC={pc}")
              break

            # Direct ctypes call - bypasses HCQ queue/synchronization overhead (~75x faster)
            fxn(*[ctypes.c_uint64(buf_addrs[g]) for g in globals_list], ctypes.c_int32(0))

            inst_count += 1
            assert inst_count < max_instructions, f"exceeded {max_instructions} instructions, likely infinite loop"

  return 0
