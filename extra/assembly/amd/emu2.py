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

from extra.assembly.amd.decode import decode_inst
from extra.assembly.amd.autogen.rdna3.str_pcode import PCODE
from extra.assembly.amd.autogen.rdna3.ins import (SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, VOP1, VOP2, VOP3, VOP3SD, VOP3P, VOPC, DS, FLAT, GLOBAL, VOPD,
  SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, SMEMOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp, DSOp, FLATOp, GLOBALOp, SCRATCHOp, VOPDOp)
from extra.assembly.amd.dsl import NULL, SCC, VCC_LO, VCC_HI, EXEC_LO, EXEC_HI

MASK32, MASK64 = 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF
WAVE_SIZE = 32
PC_LO_IDX, PC_HI_IDX, SCC_IDX = 128, 129, 130
SGPR_COUNT, VGPR_SIZE = 131, 256 * 32

# Buffers: vmem at 0 (INDEX offsets to host addr), lds, vgpr, sgpr
def _define_bufs():
  vmem = UOp(Ops.DEFINE_GLOBAL, dtypes.uint32.ptr(1 << 46), arg=0)
  lds = UOp(Ops.DEFINE_GLOBAL, dtypes.uint32.ptr(16384), arg=1)
  vgpr = UOp(Ops.DEFINE_GLOBAL, dtypes.uint32.ptr(VGPR_SIZE), arg=2)
  sgpr = UOp(Ops.DEFINE_GLOBAL, dtypes.uint32.ptr(SGPR_COUNT), arg=3)
  return vmem, lds, vgpr, sgpr

def _sext(v, bits): return v - (1 << bits) if v & (1 << (bits - 1)) else v

# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION COMPILER - converts decoded instruction to UOp SINK
# ═══════════════════════════════════════════════════════════════════════════════

@functools.cache
def _compile_inst_inner(inst_bytes: bytes) -> tuple[str, UOp]:
  """Compile instruction bytes to (name, SINK UOp)."""
  inst = decode_inst(inst_bytes)
  name = f"emu2_{inst_bytes[:inst.size()].hex()}"
  vmem, lds, vgpr, sgpr = _define_bufs()
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
  def rsrc(off: int, lane: UOp) -> UOp:
    if off < 128: return rsgpr(off)
    if off == 253: return rsgpr(SCC_IDX)
    if off == 255: return UOp.const(dtypes.uint32, literal)
    if off < 255:  # inline constants
      if off < 193: return UOp.const(dtypes.uint32, off - 128)  # 0-64
      if off < 209: return UOp.const(dtypes.uint32, (-(off - 192)) & MASK32)  # -1 to -16
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
  # SOP2: scalar ALU (s_add_i32, etc.)
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, SOP2):
    s0 = rsrc(inst.ssrc0.offset, UOp.const(dtypes.index, 0))
    s1 = rsrc(inst.ssrc1.offset, UOp.const(dtypes.index, 0))
    dst_reg = inst.sdst.offset

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
      return name, UOp.sink(wsgpr(dst_reg, result), inc_pc(), arg=KernelInfo(name=name))

    return name, UOp.sink(inc_pc(), arg=KernelInfo(name=name))

  # ═══════════════════════════════════════════════════════════════════════════
  # VOP1: v_mov_b32, etc.
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOP1):
    lane = UOp.range(32, 0, AxisType.LOOP)
    exec_mask = rsgpr(EXEC_LO.offset)
    src0 = rsrc(inst.src0.offset, lane)
    vdst_reg = inst.vdst.offset - 256

    if 'MOV_B32' in inst.op.name:
      store = wvgpr(vdst_reg, lane, src0, exec_mask)
      return name, UOp.sink(store.end(lane), inc_pc(), arg=KernelInfo(name=name))

    return name, UOp.sink(inc_pc(), arg=KernelInfo(name=name))

  # ═══════════════════════════════════════════════════════════════════════════
  # VOP2: v_add_f32, v_lshlrev_b32, etc.
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOP2):
    lane = UOp.range(32, 0, AxisType.LOOP)
    exec_mask = rsgpr(EXEC_LO.offset)
    src0 = rsrc(inst.src0.offset, lane)
    src1 = rsrc(inst.vsrc1.offset, lane)
    vdst_reg = inst.vdst.offset - 256
    op_name = inst.op.name

    if 'ADD_F32' in op_name:
      result = (src0.bitcast(dtypes.float32) + src1.bitcast(dtypes.float32)).bitcast(dtypes.uint32)
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

    return name, UOp.sink(inc_pc(), arg=KernelInfo(name=name))

  # ═══════════════════════════════════════════════════════════════════════════
  # VOPD: dual-issue v_dual_mov_b32, etc.
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, VOPD):
    lane = UOp.range(32, 0, AxisType.LOOP)
    exec_mask = rsgpr(EXEC_LO.offset)

    srcx0 = rsrc(inst.srcx0.offset, lane)
    vsrcx1 = rvgpr(inst.vsrcx1.offset - 256, lane)
    vdstx_reg = inst.vdstx.offset - 256

    srcy0 = rsrc(inst.srcy0.offset, lane)
    vsrcy1 = rvgpr(inst.vsrcy1.offset - 256, lane)
    vdsty_reg = (inst.vdsty << 1) | ((inst.vdstx.offset & 1) ^ 1)

    stores = []
    opx_name = inst.opx.name if hasattr(inst.opx, 'name') else str(inst.opx)
    opy_name = inst.opy.name if hasattr(inst.opy, 'name') else str(inst.opy)

    # X operation
    if 'MOV_B32' in opx_name:
      stores.append(wvgpr(vdstx_reg, lane, srcx0, exec_mask))
    elif 'ADD_F32' in opx_name:
      result = (srcx0.bitcast(dtypes.float32) + vsrcx1.bitcast(dtypes.float32)).bitcast(dtypes.uint32)
      stores.append(wvgpr(vdstx_reg, lane, result, exec_mask))
    elif 'ADD_NC_U32' in opx_name:
      stores.append(wvgpr(vdstx_reg, lane, srcx0 + vsrcx1, exec_mask))

    # Y operation
    if 'MOV_B32' in opy_name:
      stores.append(wvgpr(vdsty_reg, lane, srcy0, exec_mask))
    elif 'ADD_F32' in opy_name:
      result = (srcy0.bitcast(dtypes.float32) + vsrcy1.bitcast(dtypes.float32)).bitcast(dtypes.uint32)
      stores.append(wvgpr(vdsty_reg, lane, result, exec_mask))
    elif 'ADD_NC_U32' in opy_name:
      stores.append(wvgpr(vdsty_reg, lane, srcy0 + vsrcy1, exec_mask))

    if stores:
      ended = [s.end(lane) for s in stores]
      return name, UOp.sink(*ended, inc_pc(), arg=KernelInfo(name=name))
    return name, UOp.sink(inc_pc(), arg=KernelInfo(name=name))

  # ═══════════════════════════════════════════════════════════════════════════
  # FLAT/GLOBAL: memory loads/stores
  # ═══════════════════════════════════════════════════════════════════════════
  if isinstance(inst, (FLAT, GLOBAL)):
    lane = UOp.range(32, 0, AxisType.LOOP)
    exec_mask = rsgpr(EXEC_LO.offset)

    addr_reg = inst.addr.offset - 256
    addr_lo = rvgpr(addr_reg, lane)
    addr_hi = rvgpr(addr_reg + 1, lane)
    addr = addr_lo.cast(dtypes.uint64) | (addr_hi.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))

    # Add saddr if present
    if hasattr(inst, 'saddr') and inst.saddr != NULL and inst.saddr.offset < 128:
      saddr = rsgpr64(inst.saddr.offset)
      addr = addr + saddr

    # Add signed offset
    offset = _sext(getattr(inst, 'offset', 0), 13)
    addr = addr + UOp.const(dtypes.uint64, offset)

    op_name = inst.op.name if hasattr(inst.op, 'name') else str(inst.op)
    ndwords = 4 if '_B128' in op_name else 3 if '_B96' in op_name else 2 if '_B64' in op_name else 1

    if 'LOAD' in op_name:
      vdst_reg = inst.vdst.offset - 256
      stores = []
      for i in range(ndwords):
        byte_addr = addr + UOp.const(dtypes.uint64, i * 4)
        val = vmem.index((byte_addr >> UOp.const(dtypes.uint64, 2)).cast(dtypes.index))
        stores.append(wvgpr(vdst_reg + i, lane, val, exec_mask))
      ended = [s.end(lane) for s in stores]
      return name, UOp.sink(*ended, inc_pc(), arg=KernelInfo(name=name))

    if 'STORE' in op_name:
      vdata_reg = inst.data.offset - 256
      stores = []
      for i in range(ndwords):
        byte_addr = addr + UOp.const(dtypes.uint64, i * 4)
        val = rvgpr(vdata_reg + i, lane)
        idx = vmem.index((byte_addr >> UOp.const(dtypes.uint64, 2)).cast(dtypes.index))
        exec_bit = (exec_mask >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
        active = exec_bit.ne(UOp.const(dtypes.uint32, 0))
        stores.append(idx.store(active.where(val, idx)))
      ended = [s.end(lane) for s in stores]
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
      prg = get_program(sink, renderer)
      runner = CompiledRunner(prg)
      globals_list = prg.globals
    except Exception as e:
      print(f"Failed to compile {name}: {e}")
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

def run_asm(lib: int, lib_sz: int, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, args_ptr: int, rsrc2: int = 0x19c) -> int:
  """Execute AMD assembly program."""
  data = bytes((ctypes.c_char * lib_sz).from_address(lib).raw)
  program = decode_program(data)

  lds_size = ((rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_SHIFT) * 512
  total_threads = lx * ly * lz

  # vmem_buf at address 0: INDEX directly offsets to host memory
  vmem_buf = Buffer('CPU', 1 << 40, dtypes.uint32, options=BufferSpec(external_ptr=0)).ensure_allocated()
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

          # Execute wave
          all_bufs = {0: vmem_buf, 1: lds_buf, 2: st.vgpr_buf, 3: st.sgpr_buf}
          while True:
            pc = st.pc
            if pc == 0xFFFFFFFF or pc not in program: break

            name, runner, globals_list = program[pc]
            if runner is None:
              print(f"No runner for {name} at PC={pc}")
              break

            bufs = [all_bufs[g] for g in globals_list]
            runner(bufs, {}, wait=True)

  return 0
