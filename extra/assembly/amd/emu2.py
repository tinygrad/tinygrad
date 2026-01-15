# RDNA3 emulator v2 - compiles AMD instructions to UOps executed via tinygrad
# Instructions are compiled to UOp SINKs that operate on the whole wave (32 lanes)
from __future__ import annotations
import struct, functools, ctypes, math
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.dtype import dtypes
from tinygrad.codegen import get_program
from tinygrad.engine.realize import CompiledRunner
from tinygrad.device import Device, Buffer
from tinygrad.runtime.autogen import hsa

from extra.assembly.amd.decode import decode_inst
from extra.assembly.amd.autogen.rdna3.str_pcode import PCODE
from extra.assembly.amd.autogen.rdna3.ins import (SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, VOP1, VOP2, VOP3, VOP3SD, VOP3P, VOPC, DS, FLAT, VOPD,
  SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, SMEMOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp, DSOp, FLATOp, GLOBALOp, SCRATCHOp, VOPDOp)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MASK32, MASK64 = 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF
WAVE_SIZE = 32

# SGPR layout: 0-127 regular, 128=PC_LO, 129=PC_HI, 130=SCC
PC_LO, PC_HI, SCC = 128, 129, 130
SGPR_COUNT = 131

# Special SGPR offsets (within 0-127 range)
EXEC_LO, EXEC_HI = 126, 127
VCC_LO, VCC_HI = 106, 107

# Inline constants for src operands 128-254 (except 253=SCC which maps to SGPR[130])
def _i32(f):
  if isinstance(f, int): f = float(f)
  if math.isnan(f): return 0xffc00000 if math.copysign(1.0, f) < 0 else 0x7fc00000
  if math.isinf(f): return 0x7f800000 if f > 0 else 0xff800000
  try: return struct.unpack('<I', struct.pack('<f', f))[0]
  except (OverflowError, struct.error): return 0x7f800000 if f > 0 else 0xff800000

FLOAT_CONSTS = {240: 0.5, 241: -0.5, 242: 1.0, 243: -1.0, 244: 2.0, 245: -2.0, 246: 4.0, 247: -4.0, 248: 1.0/(2*math.pi)}
INLINE_CONSTS = list(range(65)) + [((-i) & MASK32) for i in range(1, 17)] + [0] * (127 - 81)
for k, v in FLOAT_CONSTS.items(): INLINE_CONSTS[k - 128] = _i32(v)

# Buffer sizes
VGPR_SIZE = 256 * 32  # 256 regs * 32 lanes

# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION COMPILATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

class Ctx:
  """Compilation context with buffer UOps and helpers"""
  __slots__ = ('vmem', 'lds', 'vgpr', 'sgpr', 'inst', 'inst_words', 'literal', 'name')
  def __init__(self, inst, data: bytes):
    self.vmem = UOp(Ops.DEFINE_GLOBAL, dtypes.uint8.ptr(1 << 40), arg=0)
    self.lds = UOp(Ops.DEFINE_GLOBAL, dtypes.uint8.ptr(65536), arg=1)
    self.vgpr = UOp(Ops.DEFINE_GLOBAL, dtypes.uint32.ptr(VGPR_SIZE), arg=2)
    self.sgpr = UOp(Ops.DEFINE_GLOBAL, dtypes.uint32.ptr(SGPR_COUNT), arg=3)
    self.inst = inst
    self.inst_words = inst.size() // 4
    self.literal = int.from_bytes(data[4:8], 'little') if inst.size() > 4 else 0

  def read_src(self, src_offset: int, lane: UOp) -> UOp:
    """Read source operand based on offset"""
    if src_offset < 128:  # SGPR
      return self.sgpr.index(UOp.const(dtypes.index, src_offset))
    if src_offset == 253:  # SCC
      return self.sgpr.index(UOp.const(dtypes.index, SCC))
    if src_offset == 255:  # Literal
      return UOp.const(dtypes.uint32, self.literal)
    if src_offset < 255:  # Inline constant
      return UOp.const(dtypes.uint32, INLINE_CONSTS[src_offset - 128])
    # VGPR (256-511)
    reg = src_offset - 256
    return self.vgpr.index(UOp.const(dtypes.index, reg * 32) + lane)

  def read_vgpr(self, reg: int, lane: UOp) -> UOp:
    return self.vgpr.index(UOp.const(dtypes.index, reg * 32) + lane)

  def write_vgpr(self, reg: int, lane: UOp, val: UOp, exec_active: UOp) -> UOp:
    """Write to VGPR with EXEC mask"""
    idx = self.vgpr.index(UOp.const(dtypes.index, reg * 32) + lane)
    return idx.store(exec_active.where(val, idx))

  def read_sgpr(self, reg: int) -> UOp:
    return self.sgpr.index(UOp.const(dtypes.index, reg))

  def read_sgpr64(self, reg: int) -> UOp:
    lo = self.sgpr.index(UOp.const(dtypes.index, reg))
    hi = self.sgpr.index(UOp.const(dtypes.index, reg + 1))
    return lo.cast(dtypes.uint64) | (hi.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))

  def write_sgpr(self, reg: int, val: UOp) -> UOp:
    return self.sgpr.index(UOp.const(dtypes.index, reg)).store(val)

  def write_sgpr64(self, reg: int, val: UOp) -> tuple[UOp, UOp]:
    lo = self.sgpr.index(UOp.const(dtypes.index, reg)).store(val.cast(dtypes.uint32))
    hi = self.sgpr.index(UOp.const(dtypes.index, reg + 1)).store((val >> UOp.const(dtypes.uint64, 32)).cast(dtypes.uint32))
    return lo, hi

  def exec_active(self, lane: UOp) -> UOp:
    """Check if lane is active based on EXEC mask"""
    exec_lo = self.sgpr.index(UOp.const(dtypes.index, EXEC_LO))
    exec_bit = (exec_lo >> lane.cast(dtypes.uint32)) & UOp.const(dtypes.uint32, 1)
    return exec_bit.alu(Ops.CMPNE, UOp.const(dtypes.uint32, 0))

  def inc_pc(self) -> UOp:
    """Increment PC by instruction size"""
    pc_idx = self.sgpr.index(UOp.const(dtypes.index, PC_LO))
    return pc_idx.store(pc_idx + UOp.const(dtypes.uint32, self.inst_words))

  def set_pc_end(self) -> UOp:
    """Set PC to 0xFFFFFFFF (end program)"""
    return self.sgpr.index(UOp.const(dtypes.index, PC_LO)).store(UOp.const(dtypes.uint32, 0xFFFFFFFF))

# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

def _compile_sopp(ctx: Ctx) -> UOp:
  if ctx.inst.op == SOPPOp.S_ENDPGM:
    return UOp.sink(ctx.set_pc_end(), arg=KernelInfo(name=ctx.name))
  # NOP-like (waitcnt, nop, etc)
  return UOp.sink(ctx.inc_pc(), arg=KernelInfo(name=ctx.name))

def _compile_smem(ctx: Ctx) -> UOp:
  """Compile SMEM (scalar memory) instructions"""
  inst = ctx.inst
  # Calculate address: sbase + offset + soffset
  addr = ctx.read_sgpr64(inst.sbase.offset)
  # Sign-extend 21-bit offset
  offset = inst.offset
  if offset & (1 << 20): offset -= (1 << 21)
  addr = addr + UOp.const(dtypes.uint64, offset)

  if inst.op == SMEMOp.S_LOAD_B64:
    # Load 64 bits from memory
    # Cast vmem to uint64 ptr at the address
    vmem_u64 = ctx.vmem.cast(dtypes.uint64.ptr(1 << 37))
    val = vmem_u64.index(addr >> UOp.const(dtypes.uint64, 3))  # Divide by 8 for uint64 index
    lo, hi = ctx.write_sgpr64(inst.sdata.offset, val)
    return UOp.sink(lo, hi, ctx.inc_pc(), arg=KernelInfo(name=ctx.name))

  if inst.op == SMEMOp.S_LOAD_B32:
    vmem_u32 = ctx.vmem.cast(dtypes.uint32.ptr(1 << 38))
    val = vmem_u32.index(addr >> UOp.const(dtypes.uint64, 2))
    store = ctx.write_sgpr(inst.sdata.offset, val.cast(dtypes.uint32))
    return UOp.sink(store, ctx.inc_pc(), arg=KernelInfo(name=ctx.name))

  # Unsupported SMEM - just increment PC
  return UOp.sink(ctx.inc_pc(), arg=KernelInfo(name=ctx.name))

def _compile_vop_add_f32(ctx: Ctx) -> UOp:
  """Compile V_ADD_F32"""
  inst = ctx.inst
  src0 = getattr(inst, 'src0', None)
  src1 = getattr(inst, 'vsrc1', None) or getattr(inst, 'src1', None)
  vdst = getattr(inst, 'vdst', None)

  lane = UOp.range(32, 0, AxisType.LOOP)
  exec_active = ctx.exec_active(lane)

  s0 = ctx.read_src(src0.offset, lane)
  s1 = ctx.read_src(src1.offset, lane)
  result = (s0.bitcast(dtypes.float32) + s1.bitcast(dtypes.float32)).bitcast(dtypes.uint32)

  store = ctx.write_vgpr(vdst.offset - 256, lane, result, exec_active)
  return UOp.sink(store.end(lane), ctx.inc_pc(), arg=KernelInfo(name=ctx.name))

def _compile_vop_lshlrev_b32(ctx: Ctx) -> UOp:
  """Compile V_LSHLREV_B32: vdst = vsrc1 << (src0 & 31)"""
  inst = ctx.inst
  src0 = getattr(inst, 'src0', None)
  src1 = getattr(inst, 'vsrc1', None) or getattr(inst, 'src1', None)
  vdst = getattr(inst, 'vdst', None)

  lane = UOp.range(32, 0, AxisType.LOOP)
  exec_active = ctx.exec_active(lane)

  shift = ctx.read_src(src0.offset, lane) & UOp.const(dtypes.uint32, 31)
  val = ctx.read_src(src1.offset, lane)
  result = val << shift

  store = ctx.write_vgpr(vdst.offset - 256, lane, result, exec_active)
  return UOp.sink(store.end(lane), ctx.inc_pc(), arg=KernelInfo(name=ctx.name))

def _compile_flat(ctx: Ctx) -> UOp:
  """Compile FLAT/GLOBAL memory instructions"""
  inst = ctx.inst
  lane = UOp.range(32, 0, AxisType.LOOP)
  exec_active = ctx.exec_active(lane)

  # Calculate address from vaddr (64-bit: vaddr and vaddr+1)
  vaddr_reg = inst.vaddr.offset - 256
  addr_lo = ctx.read_vgpr(vaddr_reg, lane)
  addr_hi = ctx.read_vgpr(vaddr_reg + 1, lane)
  addr = addr_lo.cast(dtypes.uint64) | (addr_hi.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))

  # Add saddr if present (for GLOBAL instructions with scalar base)
  if isinstance(inst, (FLAT,)) and hasattr(inst, 'saddr') and inst.saddr.offset < 128:
    saddr = ctx.read_sgpr64(inst.saddr.offset)
    addr = addr + saddr

  # Add signed offset
  offset = getattr(inst, 'offset', 0)
  if offset & (1 << 12): offset -= (1 << 13)  # Sign extend 13-bit
  addr = addr + UOp.const(dtypes.uint64, offset)

  op_name = inst.op.name if hasattr(inst.op, 'name') else str(inst.op)

  if 'LOAD' in op_name and 'B32' in op_name:
    vmem_u32 = ctx.vmem.cast(dtypes.uint32.ptr(1 << 38))
    val = vmem_u32.index((addr >> UOp.const(dtypes.uint64, 2)).cast(dtypes.index))
    vdst_reg = inst.vdst.offset - 256
    store = ctx.write_vgpr(vdst_reg, lane, val, exec_active)
    return UOp.sink(store.end(lane), ctx.inc_pc(), arg=KernelInfo(name=ctx.name))

  if 'STORE' in op_name and 'B32' in op_name:
    vmem_u32 = ctx.vmem.cast(dtypes.uint32.ptr(1 << 38))
    vdata_reg = inst.vdata.offset - 256
    val = ctx.read_vgpr(vdata_reg, lane)
    idx = vmem_u32.index((addr >> UOp.const(dtypes.uint64, 2)).cast(dtypes.index))
    # Conditional store based on EXEC
    store = idx.store(exec_active.where(val, idx))
    return UOp.sink(store.end(lane), ctx.inc_pc(), arg=KernelInfo(name=ctx.name))

  # Unsupported FLAT - just increment PC
  return UOp.sink(ctx.inc_pc(), arg=KernelInfo(name=ctx.name))

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN COMPILE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def compile_inst(data: bytes) -> tuple[str, UOp]:
  """Compile instruction bytes to (name, SINK UOp)"""
  inst = decode_inst(data)
  name = f"ins_{data[:inst.size()].hex()}"
  ctx = Ctx(inst, data)
  ctx.name = name  # Store name for use in KernelInfo

  # SOPP instructions
  if isinstance(inst, SOPP):
    return name, _compile_sopp(ctx)

  # SMEM instructions
  if isinstance(inst, SMEM):
    return name, _compile_smem(ctx)

  # FLAT/GLOBAL memory
  if isinstance(inst, FLAT):
    return name, _compile_flat(ctx)

  # VOP instructions
  if isinstance(inst, (VOP1, VOP2, VOP3)):
    op_name = inst.op.name
    if 'ADD_F32' in op_name:
      return name, _compile_vop_add_f32(ctx)
    if 'LSHLREV_B32' in op_name:
      return name, _compile_vop_lshlrev_b32(ctx)

  # Default: just increment PC
  return name, UOp.sink(ctx.inc_pc(), arg=KernelInfo(name=ctx.name))

# ═══════════════════════════════════════════════════════════════════════════════
# PROGRAM DECODE AND COMPILATION
# ═══════════════════════════════════════════════════════════════════════════════

def decode_program(data: bytes) -> dict[int, tuple[str, UOp, CompiledRunner|None, list[int]|None]]:
  """Decode entire program to {pc: (name, sink, runner, globals)}"""
  result = {}
  renderer = Device['CPU'].renderer
  i = 0
  while i < len(data):
    inst = decode_inst(data[i:])
    inst_size = inst.size()

    if isinstance(inst, SOPP) and inst.op == SOPPOp.S_CODE_END:
      break

    name, sink = compile_inst(bytes(data[i:i + inst_size + 4]))

    try:
      prg = get_program(sink, renderer)
      runner = CompiledRunner(prg)
      globals_list = prg.globals
    except Exception as e:
      print(f"Failed to compile {name}: {e}")
      runner, globals_list = None, None

    result[i // 4] = (name, sink, runner, globals_list)
    i += inst_size

  return result

# ═══════════════════════════════════════════════════════════════════════════════
# WAVE STATE
# ═══════════════════════════════════════════════════════════════════════════════

class WaveState:
  """Wave execution state with VGPR and SGPR storage as tinygrad Buffers"""
  __slots__ = ('vgpr_buf', 'sgpr_buf', '_vgpr_mv', '_sgpr_mv', 'n_lanes')

  def __init__(self, n_lanes: int = WAVE_SIZE):
    self.n_lanes = n_lanes
    self.vgpr_buf = Buffer('CPU', VGPR_SIZE, dtypes.uint32).ensure_allocated()
    self.sgpr_buf = Buffer('CPU', SGPR_COUNT, dtypes.uint32).ensure_allocated()
    # Get zero-copy memoryviews for direct access
    self._vgpr_mv = self.vgpr_buf.as_buffer(force_zero_copy=True).cast('I')
    self._sgpr_mv = self.sgpr_buf.as_buffer(force_zero_copy=True).cast('I')
    # Initialize state
    self._write_sgpr(EXEC_LO, (1 << n_lanes) - 1)
    self._write_sgpr(PC_LO, 0)

  def _write_sgpr(self, idx: int, val: int):
    self._sgpr_mv[idx] = val & MASK32

  def _read_sgpr(self, idx: int) -> int:
    return self._sgpr_mv[idx]

  def _write_vgpr(self, reg: int, lane: int, val: int):
    self._vgpr_mv[reg * 32 + lane] = val & MASK32

  def _read_vgpr(self, reg: int, lane: int) -> int:
    return self._vgpr_mv[reg * 32 + lane]

  @property
  def pc(self) -> int: return self._read_sgpr(PC_LO)

  @pc.setter
  def pc(self, val: int): self._write_sgpr(PC_LO, val)

  @property
  def exec_mask(self) -> int: return self._read_sgpr(EXEC_LO)

  @exec_mask.setter
  def exec_mask(self, val: int): self._write_sgpr(EXEC_LO, val)

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_asm(lib: int, lib_sz: int, gx: int, gy: int, gz: int,
            lx: int, ly: int, lz: int, args_ptr: int, rsrc2: int = 0x19c) -> int:
  """Execute AMD assembly program"""
  data = bytes((ctypes.c_char * lib_sz).from_address(lib).raw)
  program = decode_program(data)

  lds_size = ((rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_SHIFT) * 512
  total_threads = lx * ly * lz

  vmem_buf = Buffer('CPU', 1 << 20, dtypes.uint8).ensure_allocated()
  lds_buf = Buffer('CPU', max(lds_size, 1), dtypes.uint8).ensure_allocated()

  for gidz in range(gz):
    for gidy in range(gy):
      for gidx in range(gx):
        for wave_start in range(0, total_threads, WAVE_SIZE):
          n_lanes = min(WAVE_SIZE, total_threads - wave_start)
          st = WaveState(n_lanes)

          # Initialize s[0:1] with kernel args pointer
          st._write_sgpr(0, args_ptr & MASK32)
          st._write_sgpr(1, (args_ptr >> 32) & MASK32)

          # Initialize workgroup IDs
          sgpr_idx = (rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT
          if rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X:
            st._write_sgpr(sgpr_idx, gidx); sgpr_idx += 1
          if rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y:
            st._write_sgpr(sgpr_idx, gidy); sgpr_idx += 1
          if rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z:
            st._write_sgpr(sgpr_idx, gidz)

          # Initialize VGPR[0] with packed workitem IDs
          for tid in range(wave_start, wave_start + n_lanes):
            lane = tid - wave_start
            z, y, x = tid // (lx * ly), (tid // lx) % ly, tid % lx
            st._write_vgpr(0, lane, (z << 20) | (y << 10) | x)

          # Execute wave
          all_bufs = {0: vmem_buf, 1: lds_buf, 2: st.vgpr_buf, 3: st.sgpr_buf}
          while True:
            pc = st.pc
            if pc == 0xFFFFFFFF or pc not in program:
              break

            name, sink, runner, globals_list = program[pc]
            if runner is None:
              print(f"No runner for {name} at PC={pc}")
              break

            # Pass buffers in order matching prg.globals
            bufs = [all_bufs[g] for g in globals_list]
            runner(bufs, {})

  return 0

# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_inst(data: bytes, st: WaveState, vmem_buf: Buffer = None):
  """Compile and run a single instruction"""
  name, sink = compile_inst(data)
  prg = get_program(sink, Device['CPU'].renderer)
  runner = CompiledRunner(prg)
  # Map globals to actual buffers
  all_bufs = {0: vmem_buf, 1: None, 2: st.vgpr_buf, 3: st.sgpr_buf}
  bufs = [all_bufs[g] for g in prg.globals]
  Device['CPU'].synchronize()  # Ensure previous work is done
  runner(bufs, {}, wait=True)
  return prg

def test_v_add_f32():
  """Test V_ADD_F32 v5, v3, v4"""
  print("TEST: V_ADD_F32 v5, v3, v4")
  data = bytes([0x03, 0x09, 0x0a, 0x06])

  st = WaveState(32)
  for lane in range(32):
    st._write_vgpr(3, lane, _i32(1.0))
    st._write_vgpr(4, lane, _i32(2.0))

  run_inst(data, st)

  expected = _i32(3.0)
  for lane in range(32):
    if st._read_vgpr(5, lane) != expected:
      print(f"  FAIL lane {lane}: got {st._read_vgpr(5, lane):#x}")
      return False
  if st.pc != 1:
    print(f"  FAIL: PC = {st.pc}, expected 1")
    return False
  print("  PASS")
  return True

def test_v_lshlrev_b32():
  """Test V_LSHLREV_B32 v2, 2, v1 (shift left by 2)"""
  print("TEST: V_LSHLREV_B32 v2, 2, v1")
  # VOP2 format: [31:31]=enc, [30:25]=op, [24:17]=vdst, [16:9]=vsrc1, [8:0]=src0
  # V_LSHLREV_B32 opcode = 0x18
  opcode = 0x18
  vdst = 2       # v2
  vsrc1 = 1      # v1
  src0 = 130     # inline constant 2 (128 + 2)
  word = (opcode << 25) | (vdst << 17) | (vsrc1 << 9) | src0
  data = word.to_bytes(4, 'little')

  st = WaveState(32)
  for lane in range(32):
    st._write_vgpr(1, lane, lane + 1)

  run_inst(data, st)

  for lane in range(32):
    expected = (lane + 1) << 2
    if st._read_vgpr(2, lane) != expected:
      print(f"  FAIL lane {lane}: got {st._read_vgpr(2, lane)}, expected {expected}")
      return False
  if st.pc != 1:
    print(f"  FAIL: PC = {st.pc}, expected 1")
    return False
  print("  PASS")
  return True

def test_s_endpgm():
  """Test S_ENDPGM"""
  print("TEST: S_ENDPGM")
  data = bytes.fromhex('0000b0bf')  # s_endpgm

  st = WaveState(32)
  run_inst(data, st)

  if st.pc != 0xFFFFFFFF:
    print(f"  FAIL: PC = {st.pc:#x}, expected 0xFFFFFFFF")
    return False
  print("  PASS")
  return True

def test_s_waitcnt():
  """Test S_WAITCNT (NOP)"""
  print("TEST: S_WAITCNT")
  data = bytes.fromhex('000089bf')  # s_waitcnt 0

  st = WaveState(32)
  run_inst(data, st)

  if st.pc != 1:
    print(f"  FAIL: PC = {st.pc}, expected 1")
    return False
  print("  PASS")
  return True

if __name__ == "__main__":
  results = []
  results.append(test_v_add_f32())
  results.append(test_v_lshlrev_b32())
  results.append(test_s_endpgm())
  results.append(test_s_waitcnt())

  print("\n" + "="*60)
  print(f"RESULTS: {sum(results)}/{len(results)} passed")
  print("="*60)
