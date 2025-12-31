# RDNA3 emulator - executes compiled pseudocode from AMD ISA PDF
# mypy: ignore-errors
from __future__ import annotations
import ctypes, struct
from extra.assembly.amd.dsl import Inst, RawImm, unwrap, FLOAT_ENC, MASK32, MASK64, _f32, _i32, _sext, _f16, _i16, _f64, _i64
from extra.assembly.amd.asm import detect_format
from extra.assembly.amd.autogen.rdna3.gen_pcode import get_compiled_functions
from extra.assembly.amd.autogen.rdna3.ins import (SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, VOP1, VOP2, VOP3, VOP3SD, VOP3P, VOPC, DS, FLAT, VOPD,
  SrcEnum, SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, SMEMOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp, DSOp, FLATOp, GLOBALOp, VOPDOp)

Program = dict[int, Inst]
WAVE_SIZE, SGPR_COUNT, VGPR_COUNT = 32, 128, 256
VCC_LO, VCC_HI, NULL, EXEC_LO, EXEC_HI, SCC = SrcEnum.VCC_LO, SrcEnum.VCC_HI, SrcEnum.NULL, SrcEnum.EXEC_LO, SrcEnum.EXEC_HI, SrcEnum.SCC

# Inline constants for src operands 128-254. Build tables for f32, f16, and f64 formats.
_FLOAT_CONSTS = {v: k for k, v in FLOAT_ENC.items()} | {248: 0.15915494309189535}  # INV_2PI
def _build_inline_consts(mask, to_bits):
  tbl = list(range(65)) + [((-i) & mask) for i in range(1, 17)] + [0] * (127 - 81)
  for k, v in _FLOAT_CONSTS.items(): tbl[k - 128] = to_bits(v)
  return tbl
_INLINE_CONSTS = _build_inline_consts(MASK32, _i32)
_INLINE_CONSTS_F16 = _build_inline_consts(0xffff, _i16)
_INLINE_CONSTS_F64 = _build_inline_consts(MASK64, _i64)

# Helper: extract/write 16-bit half from/to 32-bit value
def _src16(raw: int, is_hi: bool) -> int: return ((raw >> 16) & 0xffff) if is_hi else (raw & 0xffff)
def _dst16(cur: int, val: int, is_hi: bool) -> int: return (cur & 0x0000ffff) | ((val & 0xffff) << 16) if is_hi else (cur & 0xffff0000) | (val & 0xffff)
def _vgpr_hi(src: int) -> bool: return src >= 256 and ((src - 256) & 0x80) != 0
def _vgpr_masked(src: int) -> int: return ((src - 256) & 0x7f) + 256 if src >= 256 else src

# Memory access
_valid_mem_ranges: list[tuple[int, int]] = []
def set_valid_mem_ranges(ranges: set[tuple[int, int]]) -> None: _valid_mem_ranges.clear(); _valid_mem_ranges.extend(ranges)
def _mem_valid(addr: int, size: int) -> bool:
  return not _valid_mem_ranges or any(s <= addr and addr + size <= s + z for s, z in _valid_mem_ranges)
def _ctypes_at(addr: int, size: int): return (ctypes.c_uint8 if size == 1 else ctypes.c_uint16 if size == 2 else ctypes.c_uint32).from_address(addr)
def mem_read(addr: int, size: int) -> int: return _ctypes_at(addr, size).value if _mem_valid(addr, size) else 0
def mem_write(addr: int, size: int, val: int) -> None:
  if _mem_valid(addr, size): _ctypes_at(addr, size).value = val

# Memory op tables (not pseudocode - these are format descriptions)
def _mem_ops(ops, suffix_map):
  return {getattr(e, f"{p}_{s}"): v for e in ops for s, v in suffix_map.items() for p in [e.__name__.replace("Op", "")]}
_LOAD_MAP = {'LOAD_B32': (1,4,0), 'LOAD_B64': (2,4,0), 'LOAD_B96': (3,4,0), 'LOAD_B128': (4,4,0), 'LOAD_U8': (1,1,0), 'LOAD_I8': (1,1,1), 'LOAD_U16': (1,2,0), 'LOAD_I16': (1,2,1)}
_STORE_MAP = {'STORE_B32': (1,4), 'STORE_B64': (2,4), 'STORE_B96': (3,4), 'STORE_B128': (4,4), 'STORE_B8': (1,1), 'STORE_B16': (1,2)}
FLAT_LOAD, FLAT_STORE = _mem_ops([GLOBALOp, FLATOp], _LOAD_MAP), _mem_ops([GLOBALOp, FLATOp], _STORE_MAP)
# D16 ops: load/store 16-bit to lower or upper half of VGPR. Format: (size, sign, hi) where hi=1 means upper 16 bits
_D16_LOAD_MAP = {'LOAD_D16_U8': (1,0,0), 'LOAD_D16_I8': (1,1,0), 'LOAD_D16_B16': (2,0,0),
                 'LOAD_D16_HI_U8': (1,0,1), 'LOAD_D16_HI_I8': (1,1,1), 'LOAD_D16_HI_B16': (2,0,1)}
_D16_STORE_MAP = {'STORE_D16_HI_B8': (1,1), 'STORE_D16_HI_B16': (2,1)}  # (size, hi)
FLAT_D16_LOAD = _mem_ops([GLOBALOp, FLATOp], _D16_LOAD_MAP)
FLAT_D16_STORE = _mem_ops([GLOBALOp, FLATOp], _D16_STORE_MAP)
DS_LOAD = {DSOp.DS_LOAD_B32: (1,4,0), DSOp.DS_LOAD_B64: (2,4,0), DSOp.DS_LOAD_B128: (4,4,0), DSOp.DS_LOAD_U8: (1,1,0), DSOp.DS_LOAD_I8: (1,1,1), DSOp.DS_LOAD_U16: (1,2,0), DSOp.DS_LOAD_I16: (1,2,1)}
DS_STORE = {DSOp.DS_STORE_B32: (1,4), DSOp.DS_STORE_B64: (2,4), DSOp.DS_STORE_B128: (4,4), DSOp.DS_STORE_B8: (1,1), DSOp.DS_STORE_B16: (1,2)}
# 2ADDR ops: load/store two values using offset0 and offset1
DS_LOAD_2ADDR = {DSOp.DS_LOAD_2ADDR_B32: 4, DSOp.DS_LOAD_2ADDR_B64: 8}
DS_STORE_2ADDR = {DSOp.DS_STORE_2ADDR_B32: 4, DSOp.DS_STORE_2ADDR_B64: 8}
SMEM_LOAD = {SMEMOp.S_LOAD_B32: 1, SMEMOp.S_LOAD_B64: 2, SMEMOp.S_LOAD_B128: 4, SMEMOp.S_LOAD_B256: 8, SMEMOp.S_LOAD_B512: 16}

# VOPD op -> VOP3 op mapping (VOPD is dual-issue of VOP1/VOP2 ops, use VOP3 enums for pseudocode lookup)
_VOPD_TO_VOP = {
  VOPDOp.V_DUAL_FMAC_F32: VOP3Op.V_FMAC_F32, VOPDOp.V_DUAL_FMAAK_F32: VOP2Op.V_FMAAK_F32, VOPDOp.V_DUAL_FMAMK_F32: VOP2Op.V_FMAMK_F32,
  VOPDOp.V_DUAL_MUL_F32: VOP3Op.V_MUL_F32, VOPDOp.V_DUAL_ADD_F32: VOP3Op.V_ADD_F32, VOPDOp.V_DUAL_SUB_F32: VOP3Op.V_SUB_F32,
  VOPDOp.V_DUAL_SUBREV_F32: VOP3Op.V_SUBREV_F32, VOPDOp.V_DUAL_MUL_DX9_ZERO_F32: VOP3Op.V_MUL_DX9_ZERO_F32,
  VOPDOp.V_DUAL_MOV_B32: VOP3Op.V_MOV_B32, VOPDOp.V_DUAL_CNDMASK_B32: VOP3Op.V_CNDMASK_B32,
  VOPDOp.V_DUAL_MAX_F32: VOP3Op.V_MAX_F32, VOPDOp.V_DUAL_MIN_F32: VOP3Op.V_MIN_F32,
  VOPDOp.V_DUAL_ADD_NC_U32: VOP3Op.V_ADD_NC_U32, VOPDOp.V_DUAL_LSHLREV_B32: VOP3Op.V_LSHLREV_B32, VOPDOp.V_DUAL_AND_B32: VOP3Op.V_AND_B32,
}

# Compiled pseudocode functions (lazy loaded)
_COMPILED: dict | None = None

def _get_compiled() -> dict:
  global _COMPILED
  if _COMPILED is None: _COMPILED = get_compiled_functions()
  return _COMPILED

class WaveState:
  __slots__ = ('sgpr', 'vgpr', 'scc', 'pc', 'literal', '_pend_sgpr')
  def __init__(self):
    self.sgpr, self.vgpr = [0] * SGPR_COUNT, [[0] * VGPR_COUNT for _ in range(WAVE_SIZE)]
    self.sgpr[EXEC_LO], self.scc, self.pc, self.literal, self._pend_sgpr = 0xffffffff, 0, 0, 0, {}

  @property
  def vcc(self) -> int: return self.sgpr[VCC_LO] | (self.sgpr[VCC_HI] << 32)
  @vcc.setter
  def vcc(self, v: int): self.sgpr[VCC_LO], self.sgpr[VCC_HI] = v & MASK32, (v >> 32) & MASK32
  @property
  def exec_mask(self) -> int: return self.sgpr[EXEC_LO] | (self.sgpr[EXEC_HI] << 32)
  @exec_mask.setter
  def exec_mask(self, v: int): self.sgpr[EXEC_LO], self.sgpr[EXEC_HI] = v & MASK32, (v >> 32) & MASK32

  def rsgpr(self, i: int) -> int: return 0 if i == NULL else self.scc if i == SCC else self.sgpr[i] if i < SGPR_COUNT else 0
  def wsgpr(self, i: int, v: int):
    if i < SGPR_COUNT and i != NULL: self.sgpr[i] = v & MASK32
  def rsgpr64(self, i: int) -> int: return self.rsgpr(i) | (self.rsgpr(i+1) << 32)
  def wsgpr64(self, i: int, v: int): self.wsgpr(i, v & MASK32); self.wsgpr(i+1, (v >> 32) & MASK32)

  def _rsrc_base(self, v: int, lane: int, consts):
    if v < SGPR_COUNT: return self.sgpr[v]
    if v == SCC: return self.scc
    if v < 255: return consts[v - 128]
    if v == 255: return self.literal
    return self.vgpr[lane][v - 256] if v <= 511 else 0
  def rsrc(self, v: int, lane: int) -> int: return self._rsrc_base(v, lane, _INLINE_CONSTS)
  def rsrc_f16(self, v: int, lane: int) -> int: return self._rsrc_base(v, lane, _INLINE_CONSTS_F16)
  def rsrc64(self, v: int, lane: int) -> int:
    if 128 <= v < 255: return _INLINE_CONSTS_F64[v - 128]
    if v == 255: return self.literal  # literal is already shifted in from_bytes for 64-bit ops
    return self.rsrc(v, lane) | ((self.rsrc(v+1, lane) if v < VCC_LO or 256 <= v <= 511 else 0) << 32)

  def pend_sgpr_lane(self, reg: int, lane: int, val: int):
    if reg not in self._pend_sgpr: self._pend_sgpr[reg] = 0
    if val: self._pend_sgpr[reg] |= (1 << lane)
  def commit_pends(self):
    for reg, val in self._pend_sgpr.items(): self.sgpr[reg] = val
    self._pend_sgpr.clear()


def decode_program(data: bytes) -> Program:
  result: Program = {}
  i = 0
  while i < len(data):
    try: inst_class = detect_format(data[i:])
    except ValueError: break  # stop at invalid instruction (padding/metadata after code)
    if inst_class is None: i += 4; continue
    base_size = inst_class._size()
    # Pass enough data for potential 64-bit literal (base + 8 bytes max)
    inst = inst_class.from_bytes(data[i:i+base_size+8])
    for name, val in inst._values.items():
      if name != 'op': setattr(inst, name, unwrap(val))  # skip op to preserve property access
    inst._words = inst.size() // 4
    result[i // 4] = inst
    i += inst._words * 4
  return result

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION - All ALU ops use pseudocode from PDF
# ═══════════════════════════════════════════════════════════════════════════════

def exec_scalar(st: WaveState, inst: Inst) -> int:
  """Execute scalar instruction. Returns PC delta or negative for special cases."""
  compiled = _get_compiled()
  inst_type = type(inst)

  # SOPP: special cases for control flow that has no pseudocode
  if inst_type is SOPP:
    if inst.op == SOPPOp.S_ENDPGM: return -1
    if inst.op == SOPPOp.S_BARRIER: return -2

  # SMEM: memory loads (not ALU)
  if inst_type is SMEM:
    addr = st.rsgpr64(inst.sbase * 2) + _sext(inst.offset, 21)
    if inst.soffset not in (NULL, 0x7f): addr += st.rsrc(inst.soffset, 0)
    if (cnt := SMEM_LOAD.get(inst.op)) is None: raise NotImplementedError(f"SMEM op {inst.op}")
    for i in range(cnt): st.wsgpr(inst.sdata + i, mem_read((addr + i * 4) & MASK64, 4))
    return 0

  # Get op enum and lookup compiled function
  if inst_type is SOP1: ssrc0, sdst = inst.ssrc0, inst.sdst
  elif inst_type is SOP2: ssrc0, sdst = inst.ssrc0, inst.sdst
  elif inst_type is SOPC: ssrc0, sdst = inst.ssrc0, None
  elif inst_type is SOPK: ssrc0, sdst = inst.sdst, inst.sdst  # sdst is both src and dst
  elif inst_type is SOPP: ssrc0, sdst = None, None
  else: raise NotImplementedError(f"Unknown scalar type {inst_type}")

  # SOPP has gaps in the opcode enum - treat unknown opcodes as no-ops
  try: op = inst.op
  except ValueError:
    if inst_type is SOPP: return 0
    raise
  fn = compiled.get(type(op), {}).get(op)
  if fn is None:
    # SOPP instructions without pseudocode (waits, hints, nops) are no-ops
    if inst_type is SOPP: return 0
    raise NotImplementedError(f"{op.name} not in pseudocode")

  # Build context - use inst methods to determine operand sizes
  s0 = st.rsrc64(ssrc0, 0) if inst.is_src_64(0) else (st.rsrc(ssrc0, 0) if inst_type not in (SOPK, SOPP) else (st.rsgpr(inst.sdst) if inst_type is SOPK else 0))
  s1 = st.rsrc64(inst.ssrc1, 0) if inst.is_src_64(1) else (st.rsrc(inst.ssrc1, 0) if inst_type in (SOP2, SOPC) else inst.simm16 if inst_type is SOPK else 0)
  d0 = st.rsgpr64(sdst) if inst.dst_regs() == 2 and sdst is not None else (st.rsgpr(sdst) if sdst is not None else 0)
  exec_mask = st.exec_mask
  literal = inst.simm16 if inst_type in (SOPK, SOPP) else st.literal

  # Execute compiled function - pass PC in bytes for instructions that need it
  # For wave32, mask VCC and EXEC to 32 bits since only the lower 32 bits are relevant
  pc_bytes = st.pc * 4
  vcc32, exec32 = st.vcc & MASK32, exec_mask & MASK32
  result = fn(s0, s1, 0, d0, st.scc, vcc32, 0, exec32, literal, None, {}, pc=pc_bytes)

  # Apply results
  if sdst is not None:
    (st.wsgpr64 if result.get('d0_64') else st.wsgpr)(sdst, result['d0'])
  if 'scc' in result: st.scc = result['scc']
  if 'exec' in result: st.exec_mask = result['exec']
  if 'new_pc' in result:
    # Convert absolute byte address to word delta
    # new_pc is where we want to go, st.pc is current position, inst._words will be added after
    new_pc_words = result['new_pc'] // 4
    return new_pc_words - st.pc - 1  # -1 because emulator adds inst_words (1 for scalar)
  return 0

def exec_vector(st: WaveState, inst: Inst, lane: int, lds: bytearray | None = None) -> None:
  """Execute vector instruction for one lane."""
  compiled = _get_compiled()
  inst_type, V = type(inst), st.vgpr[lane]

  # Memory ops (not ALU pseudocode)
  if inst_type is FLAT:
    op, addr_reg, data_reg, vdst, offset, saddr = inst.op, inst.addr, inst.data, inst.vdst, _sext(inst.offset, 13), inst.saddr
    addr = V[addr_reg] | (V[addr_reg+1] << 32)
    addr = (st.rsgpr64(saddr) + V[addr_reg] + offset) & MASK64 if saddr not in (NULL, 0x7f) else (addr + offset) & MASK64
    if op in FLAT_LOAD:
      cnt, sz, sign = FLAT_LOAD[op]
      for i in range(cnt): val = mem_read(addr + i * sz, sz); V[vdst + i] = _sext(val, sz * 8) & MASK32 if sign else val
    elif op in FLAT_STORE:
      cnt, sz = FLAT_STORE[op]
      for i in range(cnt): mem_write(addr + i * sz, sz, V[data_reg + i] & ((1 << (sz * 8)) - 1))
    elif op in FLAT_D16_LOAD:
      sz, sign, hi = FLAT_D16_LOAD[op]
      val = mem_read(addr, sz)
      if sign: val = _sext(val, sz * 8) & 0xffff
      V[vdst] = _dst16(V[vdst], val, hi)
    elif op in FLAT_D16_STORE:
      sz, hi = FLAT_D16_STORE[op]
      mem_write(addr, sz, _src16(V[data_reg], hi) & ((1 << (sz * 8)) - 1))
    else: raise NotImplementedError(f"FLAT op {op}")
    return

  if inst_type is DS:
    op, addr0, vdst = inst.op, (V[inst.addr] + inst.offset0) & 0xffff, inst.vdst
    if op in DS_LOAD:
      cnt, sz, sign = DS_LOAD[op]
      for i in range(cnt): val = int.from_bytes(lds[addr0+i*sz:addr0+i*sz+sz], 'little'); V[vdst + i] = _sext(val, sz * 8) & MASK32 if sign else val
    elif op in DS_STORE:
      cnt, sz = DS_STORE[op]
      for i in range(cnt): lds[addr0+i*sz:addr0+i*sz+sz] = (V[inst.data0 + i] & ((1 << (sz * 8)) - 1)).to_bytes(sz, 'little')
    elif op in DS_LOAD_2ADDR:
      # Load two values from addr+offset0*sz and addr+offset1*sz into vdst (B32: 1 dword each, B64: 2 dwords each)
      # Note: offsets are scaled by data size (4 for B32, 8 for B64) per AMD ISA
      sz = DS_LOAD_2ADDR[op]
      addr0 = (V[inst.addr] + inst.offset0 * sz) & 0xffff
      addr1 = (V[inst.addr] + inst.offset1 * sz) & 0xffff
      cnt = sz // 4  # 1 for B32, 2 for B64
      for i in range(cnt): V[vdst + i] = int.from_bytes(lds[addr0+i*4:addr0+i*4+4], 'little')
      for i in range(cnt): V[vdst + cnt + i] = int.from_bytes(lds[addr1+i*4:addr1+i*4+4], 'little')
    elif op in DS_STORE_2ADDR:
      # Store two values from data0 and data1 to addr+offset0*sz and addr+offset1*sz
      # Note: offsets are scaled by data size (4 for B32, 8 for B64) per AMD ISA
      sz = DS_STORE_2ADDR[op]
      addr0 = (V[inst.addr] + inst.offset0 * sz) & 0xffff
      addr1 = (V[inst.addr] + inst.offset1 * sz) & 0xffff
      cnt = sz // 4
      for i in range(cnt): lds[addr0+i*4:addr0+i*4+4] = (V[inst.data0 + i] & MASK32).to_bytes(4, 'little')
      for i in range(cnt): lds[addr1+i*4:addr1+i*4+4] = (V[inst.data1 + i] & MASK32).to_bytes(4, 'little')
    else: raise NotImplementedError(f"DS op {op}")
    return

  # VOPD: dual-issue, execute two ops simultaneously (read all inputs before writes)
  if inst_type is VOPD:
    vdsty = (inst.vdsty << 1) | ((inst.vdstx & 1) ^ 1)
    inputs = [(inst.opx, st.rsrc(inst.srcx0, lane), V[inst.vsrcx1], V[inst.vdstx], inst.vdstx),
              (inst.opy, st.rsrc(inst.srcy0, lane), V[inst.vsrcy1], V[vdsty], vdsty)]
    results = [(dst, fn(s0, s1, 0, d0, st.scc, st.vcc, lane, st.exec_mask, st.literal, None, {})['d0'])
               for vopd_op, s0, s1, d0, dst in inputs if (op := _VOPD_TO_VOP.get(vopd_op)) and (fn := compiled.get(type(op), {}).get(op))]
    for dst, val in results: V[dst] = val
    return

  # VOP3SD: has extra scalar dest for carry output
  if inst_type is VOP3SD:
    fn = compiled.get(VOP3SDOp, {}).get(inst.op)
    if fn is None: raise NotImplementedError(f"{inst.op.name} not in pseudocode")
    # Read sources based on register counts from inst properties
    def rsrc_n(src, regs): return st.rsrc64(src, lane) if regs == 2 else st.rsrc(src, lane)
    s0, s1, s2 = rsrc_n(inst.src0, inst.src_regs(0)), rsrc_n(inst.src1, inst.src_regs(1)), rsrc_n(inst.src2, inst.src_regs(2))
    # Carry-in ops use src2 as carry bitmask instead of VCC
    vcc = st.rsgpr64(inst.src2) if 'CO_CI' in inst.op_name else st.vcc
    result = fn(s0, s1, s2, V[inst.vdst], st.scc, vcc, lane, st.exec_mask, st.literal, None, {})
    V[inst.vdst] = result['d0'] & MASK32
    if result.get('d0_64'): V[inst.vdst + 1] = (result['d0'] >> 32) & MASK32
    if result.get('vcc_lane') is not None: st.pend_sgpr_lane(inst.sdst, lane, result['vcc_lane'])
    return

  # Get op enum and sources (None means "no source" for that operand)
  # dst_hi: for VOP1/VOP2 16-bit dst ops, bit 7 of vdst indicates .h (high 16-bit) destination
  dst_hi = False
  if inst_type is VOP1:
    if inst.op == VOP1Op.V_NOP: return
    src0, src1, src2 = inst.src0, None, None
    dst_hi = (inst.vdst & 0x80) != 0 and inst.is_dst_16()
    vdst = inst.vdst & 0x7f if inst.is_dst_16() else inst.vdst
  elif inst_type is VOP2:
    src0, src1, src2 = inst.src0, inst.vsrc1 + 256, None
    dst_hi = (inst.vdst & 0x80) != 0 and inst.is_dst_16()
    vdst = inst.vdst & 0x7f if inst.is_dst_16() else inst.vdst
  elif inst_type is VOP3:
    # VOP3 ops 0-255 are VOPC comparisons encoded as VOP3 - inst.op returns VOPCOp for these
    src0, src1, src2, vdst = inst.src0, inst.src1, (None if inst.op.value < 256 else inst.src2), inst.vdst
  elif inst_type is VOPC:
    # For 16-bit VOPC, vsrc1 uses same encoding as VOP2 16-bit: bit 7 selects hi(1) or lo(0) half
    # vsrc1 field is 8 bits: [6:0] = VGPR index, [7] = hi flag
    src0, src1, src2, vdst = inst.src0, inst.vsrc1 + 256, None, VCC_LO
  elif inst_type is VOP3P:
    # VOP3P: Packed 16-bit operations using compiled functions
    # WMMA: wave-level matrix multiply-accumulate (special handling - needs cross-lane access)
    if 'WMMA' in inst.op_name:
      if lane == 0:  # Only execute once per wave, write results for all lanes
        exec_wmma(st, inst, inst.op)
      return
    # V_FMA_MIX: Mixed precision FMA - opsel_hi controls f32(0) vs f16(1), opsel selects which f16 half
    if 'FMA_MIX' in inst.op_name:
      opsel, opsel_hi, opsel_hi2 = getattr(inst, 'opsel', 0), getattr(inst, 'opsel_hi', 0), getattr(inst, 'opsel_hi2', 0)
      neg, abs_ = getattr(inst, 'neg', 0), getattr(inst, 'neg_hi', 0)  # neg_hi reused as abs
      raws = [st.rsrc(inst.src0, lane), st.rsrc(inst.src1, lane), st.rsrc(inst.src2, lane) if inst.src2 is not None else 0]
      is_f16 = [opsel_hi & 1, opsel_hi & 2, opsel_hi2]
      srcs = [_f16(_src16(raws[i], bool(opsel & (1<<i)))) if is_f16[i] else _f32(raws[i]) for i in range(3)]
      for i in range(3):
        if abs_ & (1<<i): srcs[i] = abs(srcs[i])
        if neg & (1<<i): srcs[i] = -srcs[i]
      result = srcs[0] * srcs[1] + srcs[2]
      V = st.vgpr[lane]
      V[inst.vdst] = _i32(result) if inst.op == VOP3POp.V_FMA_MIX_F32 else _dst16(V[inst.vdst], _i16(result), inst.op == VOP3POp.V_FMA_MIXHI_F16)
      return
    # VOP3P packed ops: opsel selects halves for lo, opsel_hi for hi; neg toggles f16 sign
    raws = [st.rsrc_f16(inst.src0, lane), st.rsrc_f16(inst.src1, lane), st.rsrc_f16(inst.src2, lane) if inst.src2 is not None else 0]
    opsel, opsel_hi, opsel_hi2 = getattr(inst, 'opsel', 0), getattr(inst, 'opsel_hi', 3), getattr(inst, 'opsel_hi2', 1)
    neg, neg_hi = getattr(inst, 'neg', 0), getattr(inst, 'neg_hi', 0)
    hi_sels = [opsel_hi & 1, opsel_hi & 2, opsel_hi2]
    srcs = [((_src16(raws[i], hi_sels[i]) ^ (0x8000 if neg_hi & (1<<i) else 0)) << 16) |
            (_src16(raws[i], opsel & (1<<i)) ^ (0x8000 if neg & (1<<i) else 0)) for i in range(3)]
    fn = compiled.get(VOP3POp, {}).get(inst.op)
    if fn is None: raise NotImplementedError(f"{inst.op.name} not in pseudocode")
    st.vgpr[lane][inst.vdst] = fn(srcs[0], srcs[1], srcs[2], 0, st.scc, st.vcc, lane, st.exec_mask, st.literal, None, {})['d0'] & MASK32
    return
  else: raise NotImplementedError(f"Unknown vector type {inst_type}")

  op_cls = type(inst.op)
  fn = compiled.get(op_cls, {}).get(inst.op)
  if fn is None: raise NotImplementedError(f"{inst.op_name} not in pseudocode")

  # Read sources (with VOP3 modifiers if applicable)
  neg, abs_ = (getattr(inst, 'neg', 0), getattr(inst, 'abs', 0)) if inst_type is VOP3 else (0, 0)
  opsel = getattr(inst, 'opsel', 0) if inst_type is VOP3 else 0
  def mod_src(val: int, idx: int, is64=False) -> int:
    to_f, to_i = (_f64, _i64) if is64 else (_f32, _i32)
    if (abs_ >> idx) & 1: val = to_i(abs(to_f(val)))
    if (neg >> idx) & 1: val = to_i(-to_f(val))
    return val

  # Use inst methods to determine operand sizes (inst.is_src_16, inst.is_src_64, etc.)
  is_vop2_16bit = inst_type is VOP2 and inst.is_16bit()

  # Read sources based on register counts and dtypes from inst properties
  def read_src(src, idx, regs, is_src_16):
    if src is None: return 0
    if regs == 2: return mod_src(st.rsrc64(src, lane), idx, is64=True)
    if is_src_16 and inst_type is VOP3:
      raw = st.rsrc_f16(src, lane) if 128 <= src < 255 else st.rsrc(src, lane)
      val = _src16(raw, bool(opsel & (1 << idx)))
      if abs_ & (1 << idx): val &= 0x7fff
      if neg & (1 << idx): val ^= 0x8000
      return val
    if is_src_16 and inst_type in (VOP1, VOP2, VOPC):
      if src >= 256: return _src16(mod_src(st.rsrc(_vgpr_masked(src), lane), idx), _vgpr_hi(src))
      return mod_src(st.rsrc_f16(src, lane), idx) & 0xffff
    return mod_src(st.rsrc(src, lane), idx)

  s0 = read_src(src0, 0, inst.src_regs(0), inst.is_src_16(0))
  s1 = read_src(src1, 1, inst.src_regs(1), inst.is_src_16(1)) if src1 is not None else 0
  s2 = read_src(src2, 2, inst.src_regs(2), inst.is_src_16(2)) if src2 is not None else 0
  # Read destination (accumulator for VOP2 f16, 64-bit for 64-bit ops)
  d0 = _src16(V[vdst], dst_hi) if is_vop2_16bit else (V[vdst] | (V[vdst + 1] << 32)) if inst.dst_regs() == 2 else V[vdst]

  # V_CNDMASK_B32/B16: VOP3 encoding uses src2 as mask (not VCC); VOP2 uses VCC implicitly
  # Pass the correct mask as vcc to the function so pseudocode VCC.u64[laneId] works correctly
  vcc_for_fn = st.rsgpr64(src2) if inst.op in (VOP3Op.V_CNDMASK_B32, VOP3Op.V_CNDMASK_B16) and inst_type is VOP3 and src2 is not None and src2 < 256 else st.vcc

  # Execute compiled function - pass src0_idx and vdst_idx for lane instructions
  # For VGPR access: src0 index is the VGPR number (src0 - 256 if VGPR, else src0 for SGPR)
  src0_idx = (src0 - 256) if src0 is not None and src0 >= 256 else (src0 if src0 is not None else 0)
  result = fn(s0, s1, s2, d0, st.scc, vcc_for_fn, lane, st.exec_mask, st.literal, st.vgpr, {}, src0_idx, vdst)

  # Apply results
  if 'vgpr_write' in result:
    # Lane instruction wrote to VGPR: (lane, vgpr_idx, value)
    wr_lane, wr_idx, wr_val = result['vgpr_write']
    st.vgpr[wr_lane][wr_idx] = wr_val
  if 'vcc_lane' in result:
    # VOP2 carry ops write to VCC implicitly; VOPC/VOP3 write to vdst
    st.pend_sgpr_lane(VCC_LO if inst_type is VOP2 and 'CO_CI' in inst.op_name else vdst, lane, result['vcc_lane'])
  if 'exec_lane' in result:
    # V_CMPX instructions write to EXEC per-lane
    st.pend_sgpr_lane(EXEC_LO, lane, result['exec_lane'])
  if 'd0' in result and op_cls is not VOPCOp and 'vgpr_write' not in result:
    writes_to_sgpr = 'READFIRSTLANE' in inst.op_name or 'READLANE' in inst.op_name
    d0_val = result['d0']
    if writes_to_sgpr: st.wsgpr(vdst, d0_val & MASK32)
    elif result.get('d0_64'): V[vdst], V[vdst + 1] = d0_val & MASK32, (d0_val >> 32) & MASK32
    elif inst.is_dst_16(): V[vdst] = _dst16(V[vdst], d0_val, bool(opsel & 8) if inst_type is VOP3 else dst_hi)
    else: V[vdst] = d0_val & MASK32

# ═══════════════════════════════════════════════════════════════════════════════
# WMMA (Wave Matrix Multiply-Accumulate)
# ═══════════════════════════════════════════════════════════════════════════════

def exec_wmma(st: WaveState, inst, op: VOP3POp) -> None:
  """Execute WMMA instruction - 16x16x16 matrix multiply across the wave."""
  src0, src1, src2, vdst = inst.src0, inst.src1, inst.src2, inst.vdst
  # Read 16x16 f16 matrix from 16 lanes × 8 VGPRs (2 f16 per VGPR)
  def read_f16_mat(src):
    return [f for l in range(16) for r in range(8) for v in [st.vgpr[l][src-256+r] if src >= 256 else st.rsgpr(src+r)] for f in [_f16(v&0xffff), _f16((v>>16)&0xffff)]]
  mat_a, mat_b = read_f16_mat(src0), read_f16_mat(src1)
  # Read matrix C (16x16 f32) from lanes 0-31, VGPRs src2 to src2+7
  mat_c = [_f32(st.vgpr[i % 32][src2 - 256 + i // 32] if src2 >= 256 else st.rsgpr(src2 + i // 32)) for i in range(256)]
  # Compute D = A × B + C (16x16 matrix multiply)
  mat_d = [sum(mat_a[row*16+k] * mat_b[col*16+k] for k in range(16)) + mat_c[row*16+col] for row in range(16) for col in range(16)]
  # Write result - f16 packed or f32
  if op == VOP3POp.V_WMMA_F16_16X16X16_F16:
    for i in range(0, 256, 2):
      st.vgpr[(i//2) % 32][vdst + (i//2)//32] = ((_i16(mat_d[i+1]) & 0xffff) << 16) | (_i16(mat_d[i]) & 0xffff)
  else:
    for i in range(256): st.vgpr[i % 32][vdst + i//32] = _i32(mat_d[i])

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

SCALAR_TYPES = {SOP1, SOP2, SOPC, SOPK, SOPP, SMEM}
VECTOR_TYPES = {VOP1, VOP2, VOP3, VOP3SD, VOPC, FLAT, DS, VOPD, VOP3P}

def step_wave(program: Program, st: WaveState, lds: bytearray, n_lanes: int) -> int:
  inst = program.get(st.pc)
  if inst is None: return 1
  inst_words, st.literal, inst_type = inst._words, getattr(inst, '_literal', None) or 0, type(inst)

  if inst_type in SCALAR_TYPES:
    delta = exec_scalar(st, inst)
    if delta == -1: return -1  # endpgm
    if delta == -2: st.pc += inst_words; return -2  # barrier
    st.pc += inst_words + delta
  else:
    # V_READFIRSTLANE/V_READLANE write to SGPR, execute once; others execute per-lane with exec_mask
    is_readlane = inst_type in (VOP1, VOP3) and ('READFIRSTLANE' in inst.op_name or 'READLANE' in inst.op_name)
    exec_mask = 1 if is_readlane else st.exec_mask
    for lane in range(1 if is_readlane else n_lanes):
      if exec_mask & (1 << lane): exec_vector(st, inst, lane, lds)
    st.commit_pends()
    st.pc += inst_words
  return 0

def exec_wave(program: Program, st: WaveState, lds: bytearray, n_lanes: int) -> int:
  while st.pc in program:
    result = step_wave(program, st, lds, n_lanes)
    if result == -1: return 0
    if result == -2: return -2
  return 0

def exec_workgroup(program: Program, workgroup_id: tuple[int, int, int], local_size: tuple[int, int, int], args_ptr: int,
                   wg_id_sgpr_base: int, wg_id_enables: tuple[bool, bool, bool]) -> None:
  lx, ly, lz = local_size
  total_threads, lds = lx * ly * lz, bytearray(65536)
  waves: list[tuple[WaveState, int, int]] = []
  for wave_start in range(0, total_threads, WAVE_SIZE):
    n_lanes, st = min(WAVE_SIZE, total_threads - wave_start), WaveState()
    st.exec_mask = (1 << n_lanes) - 1
    st.wsgpr64(0, args_ptr)
    # Set workgroup IDs in SGPRs based on USER_SGPR_COUNT and enable flags from COMPUTE_PGM_RSRC2
    sgpr_idx = wg_id_sgpr_base
    for wg_id, enabled in zip(workgroup_id, wg_id_enables):
      if enabled: st.sgpr[sgpr_idx] = wg_id; sgpr_idx += 1
    # Set workitem IDs in VGPR0 using packed method: v0 = (Z << 20) | (Y << 10) | X
    for i in range(n_lanes):
      tid = wave_start + i
      st.vgpr[i][0] = ((tid // (lx * ly)) << 20) | (((tid // lx) % ly) << 10) | (tid % lx)
    waves.append((st, n_lanes, wave_start))
  has_barrier = any(isinstance(inst, SOPP) and inst.op == SOPPOp.S_BARRIER for inst in program.values())
  for _ in range(2 if has_barrier else 1):
    for st, n_lanes, _ in waves: exec_wave(program, st, lds, n_lanes)

def run_asm(lib: int, lib_sz: int, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, args_ptr: int, rsrc2: int = 0x19c) -> int:
  program = decode_program((ctypes.c_char * lib_sz).from_address(lib).raw)
  if not program: return -1
  wg_id_enables = tuple(bool((rsrc2 >> (7+i)) & 1) for i in range(3))
  for gidz in range(gz):
    for gidy in range(gy):
      for gidx in range(gx): exec_workgroup(program, (gidx, gidy, gidz), (lx, ly, lz), args_ptr, (rsrc2 >> 1) & 0x1f, wg_id_enables)
  return 0
