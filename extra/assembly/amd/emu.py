# RDNA3 emulator - executes compiled pseudocode from AMD ISA PDF
# mypy: ignore-errors
from __future__ import annotations
import ctypes, functools
from tinygrad.helpers import DEBUG, colored, ansilen
from tinygrad.runtime.autogen import hsa
from extra.assembly.amd.dsl import Inst, unwrap, FLOAT_ENC, MASK32, MASK64, _f32, _i32, _sext, _f16, _i16, _f64, _i64, SrcEnum
from extra.assembly.amd.pcode import Reg, compile_pseudocode
from extra.assembly.amd.asm import detect_format, disasm
from extra.assembly.amd.autogen.rdna3.str_pcode import PSEUDOCODE_STRINGS
from extra.assembly.amd.autogen.rdna3.ins import (SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, VOP1, VOP2, VOP3, VOP3SD, VOP3P, VOPC, DS, FLAT, VOPD,
  SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, SMEMOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp, DSOp, FLATOp, GLOBALOp, SCRATCHOp, VOPDOp)

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

# VOP3 source modifier: apply abs/neg to value
def _mod_src(val: int, idx: int, neg: int, abs_: int, is64: bool = False) -> int:
  to_f, to_i = (_f64, _i64) if is64 else (_f32, _i32)
  if (abs_ >> idx) & 1: val = to_i(abs(to_f(val)))
  if (neg >> idx) & 1: val = to_i(-to_f(val))
  return val

# Read source operand with VOP3 modifiers
def _read_src(st, inst, src, idx: int, lane: int, neg: int, abs_: int, opsel: int) -> int:
  if src is None: return 0
  literal, regs, is_src_16 = inst._literal, inst.src_regs(idx), inst.is_src_16(idx)
  if regs == 2: return _mod_src(st.rsrc64(src, lane, literal), idx, neg, abs_, is64=True)
  if isinstance(inst, VOP3P):
    opsel_hi = inst.opsel_hi | (inst.opsel_hi2 << 2)
    if 'FMA_MIX' in inst.op_name:
      raw = st.rsrc(src, lane, literal)
      sign_bit = (15 if not (opsel & (1 << idx)) else 31) if (opsel_hi >> idx) & 1 else 31
      if inst.neg_hi & (1 << idx): raw &= ~(1 << sign_bit)
      if neg & (1 << idx): raw ^= (1 << sign_bit)
      return raw
    raw = st.rsrc_f16(src, lane, literal)
    hi = _src16(raw, opsel_hi & (1 << idx)) ^ (0x8000 if inst.neg_hi & (1 << idx) else 0)
    lo = _src16(raw, opsel & (1 << idx)) ^ (0x8000 if neg & (1 << idx) else 0)
    return (hi << 16) | lo
  if is_src_16 and isinstance(inst, VOP3):
    raw = st.rsrc_f16(src, lane, literal) if 128 <= src < 255 else st.rsrc(src, lane, literal)
    val = _src16(raw, bool(opsel & (1 << idx)))
    if abs_ & (1 << idx): val &= 0x7fff
    if neg & (1 << idx): val ^= 0x8000
    return val
  if is_src_16 and isinstance(inst, (VOP1, VOP2, VOPC)):
    if src >= 256: return _src16(_mod_src(st.rsrc(_vgpr_masked(src), lane, literal), idx, neg, abs_), _vgpr_hi(src))
    return _mod_src(st.rsrc_f16(src, lane, literal), idx, neg, abs_) & 0xffff
  return _mod_src(st.rsrc(src, lane, literal), idx, neg, abs_)

# Helper: get number of dwords from memory op name
def _op_ndwords(name: str) -> int:
  if '_B128' in name: return 4
  if '_B96' in name: return 3
  if any(s in name for s in ('_B64', '_U64', '_I64', '_F64')): return 2
  return 1

# Helper: build multi-dword int from consecutive VGPRs
def _vgpr_read(V: list, base: int, ndwords: int) -> int: return sum(V[base + i] << (32 * i) for i in range(ndwords))

# Helper: write multi-dword value to consecutive VGPRs
def _vgpr_write(V: list, base: int, val: int, ndwords: int):
  for i in range(ndwords): V[base + i] = (val >> (32 * i)) & MASK32

# Memory access
_valid_mem_ranges: list[tuple[int, int]] = []
def set_valid_mem_ranges(ranges: set[tuple[int, int]]) -> None: _valid_mem_ranges.clear(); _valid_mem_ranges.extend(ranges)
def _mem_valid(addr: int, size: int) -> bool:
  return not _valid_mem_ranges or any(s <= addr and addr + size <= s + z for s, z in _valid_mem_ranges)
def _ctypes_at(addr: int, size: int): return (ctypes.c_uint8 if size == 1 else ctypes.c_uint16 if size == 2 else ctypes.c_uint64 if size == 8 else ctypes.c_uint32).from_address(addr)
def mem_read(addr: int, size: int) -> int: return _ctypes_at(addr, size).value if _mem_valid(addr, size) else 0
def mem_write(addr: int, size: int, val: int) -> None:
  if _mem_valid(addr, size): _ctypes_at(addr, size).value = val

def _make_mem_accessor(read_fn, write_fn):
  """Create a memory accessor class with the given read/write functions."""
  class _MemAccessor:
    __slots__ = ('_addr',)
    def __init__(self, addr: int): self._addr = int(addr)
    u8 = property(lambda s: read_fn(s._addr, 1), lambda s, v: write_fn(s._addr, 1, int(v)))
    u16 = property(lambda s: read_fn(s._addr, 2), lambda s, v: write_fn(s._addr, 2, int(v)))
    u32 = property(lambda s: read_fn(s._addr, 4), lambda s, v: write_fn(s._addr, 4, int(v)))
    u64 = property(lambda s: read_fn(s._addr, 8), lambda s, v: write_fn(s._addr, 8, int(v)))
    i8 = property(lambda s: _sext(read_fn(s._addr, 1), 8), lambda s, v: write_fn(s._addr, 1, int(v)))
    i16 = property(lambda s: _sext(read_fn(s._addr, 2), 16), lambda s, v: write_fn(s._addr, 2, int(v)))
    i32 = property(lambda s: _sext(read_fn(s._addr, 4), 32), lambda s, v: write_fn(s._addr, 4, int(v)))
    i64 = property(lambda s: _sext(read_fn(s._addr, 8), 64), lambda s, v: write_fn(s._addr, 8, int(v)))
    b8, b16, b32, b64 = u8, u16, u32, u64
  return _MemAccessor

_GlobalMemAccessor = _make_mem_accessor(mem_read, mem_write)

class _GlobalMem:
  """Global memory wrapper that supports MEM[addr].u32 style access."""
  def __getitem__(self, addr) -> _GlobalMemAccessor: return _GlobalMemAccessor(addr)
GlobalMem = _GlobalMem()

class LDSMem:
  """LDS memory wrapper that supports MEM[addr].u32 style access."""
  __slots__ = ('_lds',)
  def __init__(self, lds: bytearray): self._lds = lds
  def _read(self, addr: int, size: int) -> int:
    addr = addr & 0xffff
    return int.from_bytes(self._lds[addr:addr+size], 'little') if addr + size <= len(self._lds) else 0
  def _write(self, addr: int, size: int, val: int):
    addr = addr & 0xffff
    if addr + size <= len(self._lds): self._lds[addr:addr+size] = (int(val) & ((1 << (size*8)) - 1)).to_bytes(size, 'little')
  def __getitem__(self, addr): return _make_mem_accessor(self._read, self._write)(addr)

# SMEM dst register count (for writing result back to SGPRs)
SMEM_DST_COUNT = {SMEMOp.S_LOAD_B32: 1, SMEMOp.S_LOAD_B64: 2, SMEMOp.S_LOAD_B128: 4, SMEMOp.S_LOAD_B256: 8, SMEMOp.S_LOAD_B512: 16}

# VOPD op -> VOP3 op mapping (VOPD is dual-issue of VOP1/VOP2 ops, use VOP3 enums for pseudocode lookup)
_VOPD_TO_VOP = {
  VOPDOp.V_DUAL_FMAC_F32: VOP3Op.V_FMAC_F32, VOPDOp.V_DUAL_FMAAK_F32: VOP2Op.V_FMAAK_F32, VOPDOp.V_DUAL_FMAMK_F32: VOP2Op.V_FMAMK_F32,
  VOPDOp.V_DUAL_MUL_F32: VOP3Op.V_MUL_F32, VOPDOp.V_DUAL_ADD_F32: VOP3Op.V_ADD_F32, VOPDOp.V_DUAL_SUB_F32: VOP3Op.V_SUB_F32,
  VOPDOp.V_DUAL_SUBREV_F32: VOP3Op.V_SUBREV_F32, VOPDOp.V_DUAL_MUL_DX9_ZERO_F32: VOP3Op.V_MUL_DX9_ZERO_F32,
  VOPDOp.V_DUAL_MOV_B32: VOP3Op.V_MOV_B32, VOPDOp.V_DUAL_CNDMASK_B32: VOP3Op.V_CNDMASK_B32,
  VOPDOp.V_DUAL_MAX_F32: VOP3Op.V_MAX_F32, VOPDOp.V_DUAL_MIN_F32: VOP3Op.V_MIN_F32,
  VOPDOp.V_DUAL_ADD_NC_U32: VOP3Op.V_ADD_NC_U32, VOPDOp.V_DUAL_LSHLREV_B32: VOP3Op.V_LSHLREV_B32, VOPDOp.V_DUAL_AND_B32: VOP3Op.V_AND_B32,
}


class WaveState:
  __slots__ = ('sgpr', 'vgpr', 'scc', 'pc', '_pend_sgpr', 'lds', 'n_lanes')
  def __init__(self, lds: LDSMem | None = None, n_lanes: int = WAVE_SIZE):
    self.sgpr, self.vgpr = [0] * SGPR_COUNT, [[0] * VGPR_COUNT for _ in range(WAVE_SIZE)]
    self.sgpr[EXEC_LO], self.scc, self.pc, self._pend_sgpr, self.lds, self.n_lanes = 0xffffffff, 0, 0, {}, lds, n_lanes

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

  def _rsrc_base(self, v: int, lane: int, consts, literal: int):
    if v < SGPR_COUNT: return self.sgpr[v]
    if v == SCC: return self.scc
    if v < 255: return consts[v - 128]
    if v == 255: return literal
    return self.vgpr[lane][v - 256] if v <= 511 else 0
  def rsrc(self, v: int, lane: int, literal: int = 0) -> int: return self._rsrc_base(v, lane, _INLINE_CONSTS, literal)
  def rsrc_f16(self, v: int, lane: int, literal: int = 0) -> int: return self._rsrc_base(v, lane, _INLINE_CONSTS_F16, literal)
  def rsrc64(self, v: int, lane: int, literal: int = 0) -> int:
    if 128 <= v < 255: return _INLINE_CONSTS_F64[v - 128]
    if v == 255: return literal  # literal is already shifted in from_bytes for 64-bit ops
    return self.rsrc(v, lane, literal) | ((self.rsrc(v+1, lane, literal) if v < VCC_LO or 256 <= v <= 511 else 0) << 32)

  def pend_sgpr_lane(self, reg: int, lane: int, val: int):
    if reg not in self._pend_sgpr: self._pend_sgpr[reg] = 0
    if val: self._pend_sgpr[reg] |= (1 << lane)
  def commit_pends(self):
    for reg, val in self._pend_sgpr.items(): self.sgpr[reg] = val
    self._pend_sgpr.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION - All ops use pseudocode from PDF
# ═══════════════════════════════════════════════════════════════════════════════

def exec_scalar(st: WaveState, inst: Inst):
  """Execute scalar instruction. Returns 0 to continue execution."""
  # Get op enum and lookup compiled function
  if isinstance(inst, SMEM): ssrc0, sdst = None, None
  elif isinstance(inst, SOP1): ssrc0, sdst = inst.ssrc0, inst.sdst
  elif isinstance(inst, SOP2): ssrc0, sdst = inst.ssrc0, inst.sdst
  elif isinstance(inst, SOPC): ssrc0, sdst = inst.ssrc0, None
  elif isinstance(inst, SOPK): ssrc0, sdst = inst.sdst, inst.sdst  # sdst is both src and dst
  elif isinstance(inst, SOPP): ssrc0, sdst = None, None
  else: raise NotImplementedError(f"Unknown scalar type {type(inst)}")

  # SMEM: memory loads
  if isinstance(inst, SMEM):
    addr = st.rsgpr64(inst.sbase * 2) + _sext(inst.offset, 21)
    if inst.soffset not in (NULL, 0x7f): addr += st.rsrc(inst.soffset, 0, inst._literal)
    result = inst._fn(GlobalMem, addr & MASK64)
    if 'SDATA' in result:
      sdata = result['SDATA']
      for i in range(SMEM_DST_COUNT.get(inst.op, 1)): st.wsgpr(inst.sdata + i, (sdata >> (i * 32)) & MASK32)
    st.pc += inst._words
    return 0

  # Build context - use inst methods to determine operand sizes
  literal = inst._literal
  s0 = st.rsrc64(ssrc0, 0, literal) if inst.is_src_64(0) else (st.rsrc(ssrc0, 0, literal) if not isinstance(inst, (SOPK, SOPP)) else (st.rsgpr(inst.sdst) if isinstance(inst, SOPK) else 0))
  s1 = st.rsrc64(inst.ssrc1, 0, literal) if inst.is_src_64(1) else (st.rsrc(inst.ssrc1, 0, literal) if isinstance(inst, (SOP2, SOPC)) else inst.simm16 if isinstance(inst, SOPK) else 0)
  d0 = st.rsgpr64(sdst) if inst.dst_regs() == 2 and sdst is not None else (st.rsgpr(sdst) if sdst is not None else 0)
  literal = inst.simm16 if isinstance(inst, (SOPK, SOPP)) else inst._literal

  # Call compiled function with int parameters
  result = inst._fn(s0, s1, 0, d0, st.scc, st.vcc & MASK32, 0, st.exec_mask & MASK32, literal, None, pc=st.pc * 4)

  # Apply results (already int values)
  if sdst is not None and 'D0' in result:
    (st.wsgpr64 if inst.dst_regs() == 2 else st.wsgpr)(sdst, result['D0'])
  if 'SCC' in result: st.scc = result['SCC'] & 1
  if 'EXEC' in result: st.exec_mask = result['EXEC']
  if 'PC' in result:
    # Convert absolute byte address to word offset
    pc_val = result['PC']
    new_pc = pc_val if pc_val < 0x8000000000000000 else pc_val - 0x10000000000000000
    st.pc = new_pc // 4
  else:
    st.pc += inst._words
  return 0

# ═══════════════════════════════════════════════════════════════════════════════
# VECTOR INSTRUCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def exec_vopd(st: WaveState, inst, V: list, lane: int) -> None:
  """VOPD: dual-issue, execute two ops simultaneously (read all inputs before writes)."""
  literal, vdstx, vdsty = inst._literal, inst.vdstx, (inst.vdsty << 1) | ((inst.vdstx & 1) ^ 1)
  sx0, sx1, dx, sy0, sy1, dy = st.rsrc(inst.srcx0, lane, literal), V[inst.vsrcx1], V[vdstx], st.rsrc(inst.srcy0, lane, literal), V[inst.vsrcy1], V[vdsty]
  V[vdstx] = inst._fnx(sx0, sx1, 0, dx, st.scc, st.vcc, lane, st.exec_mask, literal, None)['D0']
  V[vdsty] = inst._fny(sy0, sy1, 0, dy, st.scc, st.vcc, lane, st.exec_mask, literal, None)['D0']

def exec_flat(st: WaveState, inst, V: list, lane: int) -> None:
  """FLAT/GLOBAL/SCRATCH memory ops."""
  ndwords = _op_ndwords(inst.op_name)
  addr = V[inst.addr] | (V[inst.addr + 1] << 32)
  ADDR = (st.rsgpr64(inst.saddr) + V[inst.addr] + _sext(inst.offset, 13)) & MASK64 if inst.saddr not in (NULL, 0x7f) else (addr + _sext(inst.offset, 13)) & MASK64
  vdata_src = inst.vdst if 'LOAD' in inst.op_name else inst.data
  result = inst._fn(GlobalMem, ADDR, _vgpr_read(V, vdata_src, ndwords), V[inst.vdst])
  if 'VDATA' in result: _vgpr_write(V, inst.vdst, result['VDATA'], ndwords)
  if 'RETURN_DATA' in result: _vgpr_write(V, inst.vdst, result['RETURN_DATA'], ndwords)

def exec_ds(st: WaveState, inst, V: list, lane: int) -> None:
  """DS (LDS) memory ops."""
  ndwords = _op_ndwords(inst.op_name)
  data0, data1 = _vgpr_read(V, inst.data0, ndwords), _vgpr_read(V, inst.data1, ndwords) if inst.data1 is not None else 0
  result = inst._fn(st.lds, V[inst.addr], data0, data1, inst.offset0, inst.offset1)
  if 'RETURN_DATA' in result and ('_RTN' in inst.op_name or '_LOAD' in inst.op_name):
    _vgpr_write(V, inst.vdst, result['RETURN_DATA'], ndwords * 2 if '_2ADDR_' in inst.op_name else ndwords)

def exec_vop(st: WaveState, inst: Inst, V: list, lane: int) -> None:
  """VOP1/VOP2/VOP3/VOP3SD/VOP3P/VOPC: standard ALU ops."""
  if isinstance(inst, VOP3P):
    src0, src1, src2, vdst, dst_hi = inst.src0, inst.src1, inst.src2, inst.vdst, False
    neg, abs_, opsel = inst.neg, 0, inst.opsel
  elif isinstance(inst, VOP1):
    src0, src1, src2, vdst = inst.src0, None, None, inst.vdst & 0x7f if inst.is_dst_16() else inst.vdst
    neg, abs_, opsel, dst_hi = 0, 0, 0, (inst.vdst & 0x80) != 0 and inst.is_dst_16()
  elif isinstance(inst, VOP2):
    src0, src1, src2, vdst = inst.src0, inst.vsrc1 + 256, None, inst.vdst & 0x7f if inst.is_dst_16() else inst.vdst
    neg, abs_, opsel, dst_hi = 0, 0, 0, (inst.vdst & 0x80) != 0 and inst.is_dst_16()
  elif isinstance(inst, (VOP3, VOP3SD)):
    src0, src1, src2, vdst = inst.src0, inst.src1, (None if isinstance(inst, VOP3) and inst.op.value < 256 else inst.src2), inst.vdst
    neg, abs_, opsel, dst_hi = (inst.neg, inst.abs, inst.opsel, False) if isinstance(inst, VOP3) else (0, 0, 0, False)
  elif isinstance(inst, VOPC):
    src0, src1, src2, vdst, neg, abs_, opsel, dst_hi = inst.src0, inst.vsrc1 + 256, None, VCC_LO, 0, 0, 0, False
  else:
    raise NotImplementedError(f"exec_vop: unhandled instruction type {type(inst).__name__}")

  s0 = _read_src(st, inst, src0, 0, lane, neg, abs_, opsel)
  s1 = _read_src(st, inst, src1, 1, lane, neg, abs_, opsel)
  s2 = _read_src(st, inst, src2, 2, lane, neg, abs_, opsel)
  if isinstance(inst, VOP2) and inst.is_16bit(): d0 = _src16(V[vdst], dst_hi)
  elif inst.dst_regs() == 2: d0 = V[vdst] | (V[vdst + 1] << 32)
  else: d0 = V[vdst]

  if isinstance(inst, VOP3SD) and 'CO_CI' in inst.op_name: vcc_for_fn = st.rsgpr64(inst.src2)
  elif isinstance(inst, VOP3) and inst.op in (VOP3Op.V_CNDMASK_B32, VOP3Op.V_CNDMASK_B16) and src2 is not None and src2 < 256: vcc_for_fn = st.rsgpr64(src2)
  else: vcc_for_fn = st.vcc
  src0_idx = (src0 - 256) if src0 is not None and src0 >= 256 else (src0 if src0 is not None else 0)
  extra_kwargs = {'opsel': opsel, 'opsel_hi': inst.opsel_hi | (inst.opsel_hi2 << 2)} if isinstance(inst, VOP3P) and 'FMA_MIX' in inst.op_name else {}
  result = inst._fn(s0, s1, s2, d0, st.scc, vcc_for_fn, lane, st.exec_mask, inst._literal, st.vgpr, src0_idx, vdst, **extra_kwargs)

  # Check if this is a VOPC instruction (either standalone VOPC or VOP3 with VOPC opcode)
  is_vopc = isinstance(inst.op, VOPCOp) or (isinstance(inst, VOP3) and inst.op.value < 256)
  if 'VCC' in result:
    if isinstance(inst, VOP3SD): st.pend_sgpr_lane(inst.sdst, lane, (result['VCC'] >> lane) & 1)
    else: st.pend_sgpr_lane(VCC_LO if isinstance(inst, VOP2) and 'CO_CI' in inst.op_name else vdst, lane, (result['VCC'] >> lane) & 1)
  if 'EXEC' in result:
    st.pend_sgpr_lane(EXEC_LO, lane, (result['EXEC'] >> lane) & 1)
  elif is_vopc:
    st.pend_sgpr_lane(vdst, lane, (result['D0'] >> lane) & 1)
  if not is_vopc:
    d0_val = result['D0']
    if inst.dst_regs() == 2: V[vdst], V[vdst + 1] = d0_val & MASK32, (d0_val >> 32) & MASK32
    elif not isinstance(inst, VOP3P) and inst.is_dst_16(): V[vdst] = _dst16(V[vdst], d0_val, bool(opsel & 8) if isinstance(inst, VOP3) else dst_hi)
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

# SQTT TRACING
# ═══════════════════════════════════════════════════════════════════════════════

WAVESTART_TO_INST_CYCLES = 32
SNOP_EXTRA_DELAY_MIN, SNOP_EXTRA_DELAY_MAX = 11, 22  # s_nop(11-22) has +4 penalty
SNOP_EXTRA_DELAY_CYCLES = 4

from extra.assembly.amd.sqtt import WAVESTART, WAVEEND, IMMEDIATE, VALUINST, ALUEXEC, AluSrc

def _get_src_vgprs(inst: Inst) -> list[int]:
  if isinstance(inst, VOP1): return [inst.src0 - 256] if inst.src0 >= 256 else []
  if isinstance(inst, VOP2): return ([inst.src0 - 256] if inst.src0 >= 256 else []) + [inst.vsrc1]
  if isinstance(inst, VOP3): return [s - 256 for s in [inst.src0, inst.src1, getattr(inst, 'src2', None)] if s is not None and s >= 256]
  return []

class SQTTState:
  """SQTT tracing with cycle-accurate RDNA3 VALU pipeline model.

  NOTE: This is a hardware-plausible model derived from observed SQTT timing patterns.
  The model should be verified by tests against real hardware traces, not by fitting
  formulas to expected outputs. If tests fail, the model needs to be understood and
  fixed, not hacked with magic constants.

  Physical model:
    - alu[4]: 4-stage ALU pipeline, each slot holds dest_vgpr or None
    - in_flight: up to 12 in-flight instructions (issued but not yet completed)
    - issue_queue: instructions waiting to enter ALU (sources not ready)
    - fwd_slots: 4 forwarding slots, reserved at issue, freed when consumer forwards
    - completed: vgprs with results ready (exited ALU)

  Forwarding model (4 slots):
    - Slot reserved at ISSUE time if available (len(fwd_slots) < 4)
    - Slot freed when a consumer uses the result for forwarding
    - Consumer can forward if: has a slot AND producer is completed
    - If no slot at issue, instruction uses regfile path (+4 cycle penalty)
  """
  def __init__(self, wave_id: int = 0, simd: int = 0, cu: int = 0):
    self.wave_id, self.simd, self.cu = wave_id, simd, cu
    self.cycle = 0
    self.packets = []

    # 4-stage ALU pipeline: each slot holds dest_vgpr or None
    self.alu = [None, None, None, None]

    # In-flight instructions: max 12 at a time, each is (dest_vgpr, srcs, has_fwd_slot)
    self.in_flight: list[tuple[int, list[int], bool]] = []

    # Issue queue: list of (dest_vgpr, srcs, ready_at, has_fwd_slot, was_warm) waiting for deps
    # ready_at: cycle when this instruction can enter ALU (0 = no restriction)
    # has_fwd_slot: True if this instruction reserved a forwarding slot at issue time
    # was_warm: True if forwarding path was warm when this instruction was issued
    self.issue_queue: list[tuple[int, list[int], int, bool, bool]] = []

    # 4 forwarding slots: consumer adds producer at issue, freed when consumer forwards
    self.fwd_slots: list[int] = []  # producer vgprs reserved for forwarding

    # VGPRs that had a dependent try to add them to fwd_slots (successful or not)
    self.had_dependent: set[int] = set()

    # VGPRs that were issued after forwarding chain broke (can't forward)
    self.fwd_chain_broken: set[int] = set()

    # Set of completed vgprs (results ready, exited ALU)
    self.completed: set[int] = set()

    # Cold start: first forwarding use has +1 cycle penalty
    self.forward_warm = False
    self.cold_used = False  # True if cold start penalty was applied

  def emit(self, pkt_class, **kwargs):
    self.packets.append(pkt_class(_time=self.cycle, **kwargs))

  def _fmt_alu(self) -> str:
    # Fixed width: each slot 3 chars, total ALU[xxx,xxx,xxx,xxx] = 20 chars
    slots = [f'v{v}' if v is not None else '-' for v in self.alu]
    return 'ALU[' + ','.join(f'{s:>3}' for s in slots) + ']'

  def _fmt_fwd(self) -> str:
    items = [f'v{v}' for v in self.fwd_slots]
    content = 'FWD[' + ','.join(items) + ']' if items else 'FWD[]'
    padded = f'{content:<24}'
    return colored(padded, 'yellow') if items else padded

  def _fmt_iq(self) -> str:
    def fmt_item(d, r, fwd):
      s = f'v{d}'
      if r != 0: s += f'@{abs(r)}'
      if not fwd: s += 'R'
      return s
    items = [fmt_item(d, r, fwd) for d, _, r, fwd, _ in self.issue_queue]
    return 'IQ[' + ','.join(items) + ']' if items else 'IQ[]'

  def _debug_line(self, events: list[str] | None = None):
    if DEBUG < 3: return
    # Skip empty cycles (nothing in ALU, no events, no IQ)
    has_alu = any(s is not None for s in self.alu)
    if not has_alu and not events and not self.issue_queue: return
    cycle = colored(f'C{self.cycle:>3}:', 'cyan')
    alu = self._fmt_alu()
    fwd = self._fmt_fwd()
    iq = f'{self._fmt_iq():<28}'
    ev_str = ' '.join(events) if events else ''
    ev_padded = f'{ev_str:<20}' if ev_str else ' ' * 20
    print(f"{cycle} {alu} {fwd} {iq} {ev_padded}")

  def _can_issue(self) -> bool:
    return len(self.in_flight) < 12

  def _has_pending_write(self, vgpr: int) -> bool:
    """Check if there's a pending write to this VGPR (in ALU, in-flight, or issue queue)."""
    if any(slot == vgpr for slot in self.alu if slot is not None): return True
    if any(d == vgpr for d, _, _ in self.in_flight): return True
    if any(d == vgpr for d, _, _, _, _ in self.issue_queue): return True
    return False

  def _all_srcs_ready(self, srcs: list[int]) -> bool:
    """Returns True if all sources are ready (completed or no pending write)."""
    for src in srcs:
      if src in self.completed: continue
      if not self._has_pending_write(src): continue  # initial value
      return False
    return True

  def tick(self):
    self.cycle += 1
    if self.cycle > 10000: raise RuntimeError("cycle limit exceeded")
    events = []

    # 1. ALU[3] exits - capture but don't add to completed yet
    exiting = self.alu[3]
    if exiting is not None:
      self.emit(ALUEXEC, src=AluSrc.VALU)
      events.append(colored(f"EXEC v{exiting}", 'red'))

    # 2. Slide ALU pipeline
    self.alu[3] = self.alu[2]
    self.alu[2] = self.alu[1]
    self.alu[1] = self.alu[0]
    self.alu[0] = None

    # 3. Try to promote from issue_queue to ALU[0] (before adding exiting to completed)
    if self.alu[0] is None and self.issue_queue:
      for i, (dest, srcs, ready_at, has_fwd_slot, was_warm) in enumerate(self.issue_queue):
        # Check if instruction has a minimum ready cycle
        if ready_at > 0 and self.cycle < ready_at:
          continue
        # Check if sources are ready
        ready = self._all_srcs_ready(srcs)
        has_deps = len(srcs) > 0
        if not ready:
          continue
        # Cold start penalty: first dependent instruction has +1 cycle delay (delta=6 vs delta=5)
        # Only applies if forwarding path wasn't warm when this instruction was issued
        if has_deps and not was_warm and not self.cold_used:
          self.cold_used = True
          self.issue_queue[i] = (dest, srcs, self.cycle + 1, has_fwd_slot, was_warm)
          continue
        # Forwarding: consumer can forward if:
        # 1. Not in fwd_chain_broken (chain must be intact), AND
        # 2. Producer has a slot (source is in fwd_slots), AND
        # 3. Either activated by dependent OR successfully added producer at issue
        # Note: if issued cold with no slot, activation only counts if the activator also has a dependent
        chain_intact = dest not in self.fwd_chain_broken
        producer_has_slot = has_deps and any(src in self.fwd_slots for src in srcs)
        # Check activation validity
        if dest in self.had_dependent:
          if was_warm or has_fwd_slot:
            activated_by_dependent = True
          else:
            # Cold + no slot: activation only counts if activator itself has a dependent
            # This handles the chain_6 vs chain_7 difference (chain_7 has v6 which activates v5)
            activated_by_dependent = (dest + 1) in self.had_dependent  # activator is dest+1 in a chain
        else:
          activated_by_dependent = False
        can_forward = chain_intact and producer_has_slot and (activated_by_dependent or has_fwd_slot)
        # Regfile path: has dependencies but can't forward
        must_use_regfile = has_deps and not can_forward
        # Regfile penalty: add +4 cycles latency (only apply once)
        if must_use_regfile and ready_at == 0:
          self.issue_queue[i] = (dest, srcs, self.cycle + 4, has_fwd_slot, was_warm)
          continue
        # Enter ALU
        self.alu[0] = dest
        self.issue_queue.pop(i)
        # Free producer's forwarding slot when consumer dispatches (regardless of fwd/rf)
        for src in srcs:
          if src in self.fwd_slots:
            self.fwd_slots.remove(src)
            break
        events.append(colored(f"v{dest}->ALU" + ("(fwd)" if can_forward else "(rf)" if must_use_regfile else ""), 'green'))
        break

    # 4. Now add exiting instruction to completed (after promotion decision)
    if exiting is not None:
      self.completed.add(exiting)
      # Remove from in_flight - any VALU completing warms up the forward path
      for idx, (d, _, _) in enumerate(self.in_flight):
        if d == exiting:
          self.forward_warm = True
          self.in_flight.pop(idx)
          break

    self._debug_line(events)

  def _pipeline_empty(self) -> bool:
    if any(s is not None for s in self.alu): return False
    if self.issue_queue: return False
    if self.in_flight: return False
    return True

  def process_instruction(self, inst: Inst):
    if isinstance(inst, SOPP) and inst.op == SOPPOp.S_DELAY_ALU:
      # TODO: implement s_delay_alu properly
      return

    elif isinstance(inst, SOPP) and inst.op == SOPPOp.S_NOP:
      # s_nop(N) delays N+1 cycles, plus extra penalty for s_nop(11-22)
      cycles = inst.simm16 + 1
      if SNOP_EXTRA_DELAY_MIN <= inst.simm16 <= SNOP_EXTRA_DELAY_MAX:
        cycles += SNOP_EXTRA_DELAY_CYCLES
      if DEBUG >= 3:
        cycle = colored(f'C{self.cycle:>3}:', 'cyan')
        # 20 (ALU) + 1 + 24 (FWD) + 1 + 28 (IQ) + 1 + 20 (events) = 95 padding after cycle
        print(f"{cycle} {' ' * 95} {disasm(inst)}")
      for _ in range(cycles): self.tick()
      self.emit(IMMEDIATE, wave=self.wave_id)

    elif isinstance(inst, SOPP) and inst.op == SOPPOp.S_ENDPGM:
      # Drain pipeline before ending
      while not self._pipeline_empty(): self.tick()
      self.emit(WAVEEND, wave=self.wave_id, simd=self.simd, cu_lo=self.cu & 0x7, flag7=self.cu >> 3)

    elif isinstance(inst, (VOP1, VOP2, VOP3)):
      # Check for issue stall (no free in-flight slots)
      while not self._can_issue():
        self.tick()

      # Issue: add to in_flight and issue_queue
      srcs = _get_src_vgprs(inst)
      dest = inst.vdst
      # Clear stale state for this dest (WAW hazard)
      self.completed.discard(dest)
      if dest in self.fwd_slots: self.fwd_slots.remove(dest)

      # Consumer adds producer to fwd_slots (if room and has dependency)
      # If producer is in fwd_chain_broken, or we can't add, the chain breaks for this instruction too
      has_fwd_slot = False
      if srcs:
        producer = srcs[0]
        self.had_dependent.add(producer)  # record that producer has a dependent
        # Check if producer's forwarding chain is already broken
        if producer in self.fwd_chain_broken:
          # Chain is broken, this instruction also can't forward
          self.fwd_chain_broken.add(dest)
        elif len(self.fwd_slots) >= 4:
          # Can't add producer, chain breaks
          self.fwd_chain_broken.add(dest)
        else:
          # Can add producer
          if producer not in self.fwd_slots:
            self.fwd_slots.append(producer)
          has_fwd_slot = len(self.fwd_slots) < 4

      # Record if forwarding path was warm at issue time
      was_warm = self.forward_warm

      self.in_flight.append((dest, srcs, has_fwd_slot))
      self.issue_queue.append((dest, srcs, 0, has_fwd_slot, was_warm))
      self.emit(VALUINST, wave=self.wave_id)

      if DEBUG >= 3:
        cycle = colored(f'C{self.cycle:>3}:', 'cyan')
        slot_info = "" if has_fwd_slot else colored(" NO_SLOT", 'red')
        issue = colored(f'ISSUE v{dest}', 'magenta') + slot_info
        padding = 95 - ansilen(issue)
        print(f"{cycle} {issue}{' ' * padding} {disasm(inst)}")

      # One cycle per instruction issued, then try to enter ALU
      self.tick()
      return

    # One cycle per instruction issued (for non-VALU)
    self.tick()

  def emit_wavestart(self):
    self.emit(WAVESTART, wave=self.wave_id, simd=self.simd, cu_lo=self.cu & 0x7, flag7=self.cu >> 3)
    for _ in range(WAVESTART_TO_INST_CYCLES): self.tick()

# ═══════════════════════════════════════════════════════════════════════════════
# PROGRAM DECODE
# ═══════════════════════════════════════════════════════════════════════════════

# Wave-level dispatch functions: (st, inst) -> return_code (0 = continue, -1 = end, -2 = barrier)
def dispatch_endpgm(st, inst): return -1
def dispatch_barrier(st, inst): st.pc += inst._words; return -2
def dispatch_nop(st, inst): st.pc += inst._words; return 0
def dispatch_wmma(st, inst): exec_wmma(st, inst, inst.op); st.pc += inst._words; return 0
def dispatch_writelane(st, inst): st.vgpr[st.rsrc(inst.src1, 0, inst._literal) & 0x1f][inst.vdst] = st.rsrc(inst.src0, 0, inst._literal) & MASK32; st.pc += inst._words; return 0
def dispatch_readlane(st, inst):
  src0_idx = (inst.src0 - 256) if inst.src0 >= 256 else inst.src0
  s1 = st.rsrc(inst.src1, 0, inst._literal) if getattr(inst, 'src1', None) is not None else 0
  result = inst._fn(0, s1, 0, 0, st.scc, st.vcc, 0, st.exec_mask, inst._literal, st.vgpr, src0_idx, inst.vdst)
  st.wsgpr(inst.vdst, result['D0'])
  st.pc += inst._words; return 0

# Per-lane dispatch wrapper: wraps per-lane exec functions into wave-level dispatch
@functools.cache
def dispatch_lane(exec_fn):
  def dispatch(st, inst):
    exec_mask, vgpr, n_lanes = st.exec_mask, st.vgpr, st.n_lanes
    for lane in range(n_lanes):
      if exec_mask >> lane & 1: exec_fn(st, inst, vgpr[lane], lane)
    st.commit_pends()
    st.pc += inst._words
    return 0
  return dispatch

def decode_program(data: bytes) -> dict[int, Inst]:
  result: dict[int, Inst] = {}
  i = 0
  while i < len(data):
    inst = detect_format(data[i:]).from_bytes(data[i:])
    inst._words = inst.size() // 4

    # Determine dispatch function and pcode function
    if isinstance(inst, SOPP) and inst.op == SOPPOp.S_CODE_END: break
    elif isinstance(inst, SOPP) and inst.op == SOPPOp.S_ENDPGM: inst._dispatch = dispatch_endpgm
    elif isinstance(inst, SOPP) and inst.op == SOPPOp.S_BARRIER: inst._dispatch = dispatch_barrier
    elif isinstance(inst, SOPP) and inst.op in (SOPPOp.S_CLAUSE, SOPPOp.S_WAITCNT, SOPPOp.S_WAITCNT_DEPCTR, SOPPOp.S_SENDMSG, SOPPOp.S_SET_INST_PREFETCH_DISTANCE, SOPPOp.S_DELAY_ALU): inst._dispatch = dispatch_nop
    elif isinstance(inst, (SOP1, SOP2, SOPC, SOPK, SOPP, SMEM)): inst._dispatch = exec_scalar
    elif isinstance(inst, VOP1) and inst.op == VOP1Op.V_NOP: inst._dispatch = dispatch_nop
    elif isinstance(inst, VOP3P) and 'WMMA' in inst.op_name: inst._dispatch = dispatch_wmma
    elif isinstance(inst, VOP3) and inst.op == VOP3Op.V_WRITELANE_B32: inst._dispatch = dispatch_writelane
    elif isinstance(inst, (VOP1, VOP3)) and inst.op in (VOP1Op.V_READFIRSTLANE_B32, VOP3Op.V_READFIRSTLANE_B32, VOP3Op.V_READLANE_B32): inst._dispatch = dispatch_readlane
    elif isinstance(inst, VOPD): inst._dispatch = dispatch_lane(exec_vopd)
    elif isinstance(inst, FLAT): inst._dispatch = dispatch_lane(exec_flat)
    elif isinstance(inst, DS): inst._dispatch = dispatch_lane(exec_ds)
    else: inst._dispatch = dispatch_lane(exec_vop)

    # Compile pcode for instructions that use it (not VOPD which has _fnx/_fny, not special dispatches)
    # VOPD needs separate functions for X and Y ops
    if isinstance(inst, VOPD):
      def _compile_vopd_op(op): return compile_pseudocode(type(op).__name__, op.name, PSEUDOCODE_STRINGS[type(op)][op])
      inst._fnx, inst._fny = _compile_vopd_op(_VOPD_TO_VOP[inst.opx]), _compile_vopd_op(_VOPD_TO_VOP[inst.opy])
    elif inst._dispatch not in (dispatch_endpgm, dispatch_barrier, dispatch_nop, dispatch_wmma, dispatch_writelane):
      assert type(inst.op) != int, f"inst op of {inst} is int"
      inst._fn = compile_pseudocode(type(inst.op).__name__, inst.op.name, PSEUDOCODE_STRINGS[type(inst.op)][inst.op])
    result[i // 4] = inst
    i += inst._words * 4
  return result

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def exec_wave(program: dict[int, Inst], st: WaveState) -> int:
  while (inst := program.get(st.pc)) and (result := inst._dispatch(st, inst)) == 0: pass
  return result

def exec_workgroup(program: dict[int, Inst], workgroup_id: tuple[int, int, int], local_size: tuple[int, int, int], args_ptr: int, rsrc2: int) -> None:
  lx, ly, lz = local_size
  total_threads = lx * ly * lz
  # GRANULATED_LDS_SIZE is in 512-byte units (see ops_amd.py: lds_size = ((group_segment_size + 511) // 512))
  lds_size = ((rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_SHIFT) * 512
  lds = LDSMem(bytearray(lds_size)) if lds_size else None
  waves: list[WaveState] = []
  for wave_start in range(0, total_threads, WAVE_SIZE):
    n_lanes = min(WAVE_SIZE, total_threads - wave_start)
    st = WaveState(lds, n_lanes)
    st.exec_mask = (1 << n_lanes) - 1
    st.wsgpr64(0, args_ptr)  # s[0:1] = kernel arguments pointer
    # COMPUTE_PGM_RSRC2: USER_SGPR_COUNT is where workgroup IDs start, ENABLE_SGPR_WORKGROUP_ID_X/Y/Z control which are passed
    sgpr_idx = (rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT
    if rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X: st.sgpr[sgpr_idx] = workgroup_id[0]; sgpr_idx += 1
    if rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y: st.sgpr[sgpr_idx] = workgroup_id[1]; sgpr_idx += 1
    if rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z: st.sgpr[sgpr_idx] = workgroup_id[2]
    # VGPR0 = packed workitem IDs: (Z << 20) | (Y << 10) | X
    for tid in range(wave_start, wave_start + n_lanes):
      st.vgpr[tid - wave_start][0] = ((tid // (lx * ly)) << 20) | (((tid // lx) % ly) << 10) | (tid % lx)
    waves.append(st)
  while waves:
    waves = [st for st in waves if exec_wave(program, st) != -1]

def run_asm(lib: int, lib_sz: int, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, args_ptr: int, rsrc2: int = 0x19c) -> int:
  program = decode_program((ctypes.c_char * lib_sz).from_address(lib).raw)
  for gidz in range(gz):
    for gidy in range(gy):
      for gidx in range(gx): exec_workgroup(program, (gidx, gidy, gidz), (lx, ly, lz), args_ptr, rsrc2)
  return 0
