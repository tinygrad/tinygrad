# RDNA3 emulator - executes compiled pseudocode from AMD ISA PDF
# mypy: ignore-errors
from __future__ import annotations
import ctypes, functools
from enum import IntEnum
from tinygrad.runtime.autogen import hsa
from extra.assembly.amd.dsl import Inst, NULL, SCC, VCC_LO, VCC_HI, EXEC_LO, EXEC_HI
from extra.assembly.amd.pcode import _f32, _i32, _sext, _f16, _i16, _f64, _i64
from extra.assembly.amd.decode import decode_inst
from extra.assembly.amd.pcode import compile_pseudocode
from extra.assembly.amd.autogen.rdna3.str_pcode import PCODE
from extra.assembly.amd.autogen.rdna3.ins import (SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, VOP1, VOP2, VOP3, VOP3SD, VOP3P, VOPC, DS, FLAT, VOPD,
  SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, SMEMOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp, DSOp, FLATOp, GLOBALOp, SCRATCHOp, VOPDOp)

# Constants and helpers defined locally (not imported from dsl.py)
MASK32, MASK64 = 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF
FLOAT_ENC = {0.5: 240, -0.5: 241, 1.0: 242, -1.0: 243, 2.0: 244, -2.0: 245, 4.0: 246, -4.0: 247}
def unwrap(v): return v.offset if hasattr(v, 'offset') else v

class SGPRArray:
  """SGPR array that accepts Reg or int index. Validates SGPR range (0-127)."""
  __slots__ = ('_data',)
  def __init__(self, size: int): self._data = [0] * size
  def _idx(self, key) -> int:
    i = key.offset if hasattr(key, 'offset') else key
    assert 0 <= i < 128, f"SGPR index {i} out of range 0-127"
    return i
  def __getitem__(self, key): return self._data[self._idx(key)]
  def __setitem__(self, key, val): self._data[self._idx(key)] = val
  def __len__(self): return len(self._data)
  def __iter__(self): return iter(self._data)

class VGPRLane:
  """Single lane of VGPRs that accepts Reg or int index. Validates VGPR range (256-511)."""
  __slots__ = ('_data',)
  def __init__(self, size: int): self._data = [0] * size
  def _idx(self, key) -> int:
    i = key.offset if hasattr(key, 'offset') else key
    if i >= 256: i -= 256  # convert from src encoding to VGPR index
    assert 0 <= i < 256, f"VGPR index {i} out of range 0-255"
    return i
  def __getitem__(self, key): return self._data[self._idx(key)]
  def __setitem__(self, key, val): self._data[self._idx(key)] = val
  def __len__(self): return len(self._data)
  def __iter__(self): return iter(self._data)

WAVE_SIZE, SGPR_COUNT, VGPR_COUNT = 32, 128, 256

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
def _vgpr_hi(src) -> bool:
  off = src.offset if hasattr(src, 'offset') else src
  return off >= 256 and ((off - 256) & 0x80) != 0
def _vgpr_masked(src) -> int:
  off = src.offset if hasattr(src, 'offset') else src
  return ((off - 256) & 0x7f) + 256 if off >= 256 else off

# VOP3 source modifier: apply abs/neg to value
def _mod_src(val: int, idx: int, neg: int, abs_: int, is64: bool = False) -> int:
  to_f, to_i = (_f64, _i64) if is64 else (_f32, _i32)
  if (abs_ >> idx) & 1: val = to_i(abs(to_f(val)))
  if (neg >> idx) & 1: val = to_i(-to_f(val))
  return val

# Read source operand with VOP3 modifiers
def _read_src(st, inst, src, idx: int, lane: int, neg: int, abs_: int, opsel: int) -> int:
  if src is None: return 0
  src_off = src.offset if hasattr(src, 'offset') else src
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
    raw = st.rsrc_f16(src, lane, literal) if 128 <= src_off < 255 else st.rsrc(src, lane, literal)
    val = _src16(raw, bool(opsel & (1 << idx)))
    if abs_ & (1 << idx): val &= 0x7fff
    if neg & (1 << idx): val ^= 0x8000
    return val
  if is_src_16 and isinstance(inst, (VOP1, VOP2, VOPC)):
    if src_off >= 256: return _src16(_mod_src(st.rsrc(_vgpr_masked(src), lane, literal), idx, neg, abs_), _vgpr_hi(src))
    return _mod_src(st.rsrc_f16(src, lane, literal), idx, neg, abs_) & 0xffff
  return _mod_src(st.rsrc(src, lane, literal), idx, neg, abs_)

# Helper: get number of dwords from memory op name
def _op_ndwords(name: str) -> int:
  if '_B128' in name: return 4
  if '_B96' in name: return 3
  if any(s in name for s in ('_B64', '_U64', '_I64', '_F64')): return 2
  return 1

# Helper: build multi-dword int from consecutive VGPRs
def _vgpr_read(V: VGPRLane, reg, ndwords: int) -> int:
  base = reg.offset if hasattr(reg, 'offset') else reg
  return sum(V[base + i] << (32 * i) for i in range(ndwords))

# Helper: write multi-dword value to consecutive VGPRs
def _vgpr_write(V: VGPRLane, reg, val: int, ndwords: int):
  base = reg.offset if hasattr(reg, 'offset') else reg
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
  VOPDOp.V_DUAL_FMAC_F32: VOP3Op.V_FMAC_F32_E64, VOPDOp.V_DUAL_FMAAK_F32: VOP2Op.V_FMAAK_F32_E32, VOPDOp.V_DUAL_FMAMK_F32: VOP2Op.V_FMAMK_F32_E32,
  VOPDOp.V_DUAL_MUL_F32: VOP3Op.V_MUL_F32_E64, VOPDOp.V_DUAL_ADD_F32: VOP3Op.V_ADD_F32_E64, VOPDOp.V_DUAL_SUB_F32: VOP3Op.V_SUB_F32_E64,
  VOPDOp.V_DUAL_SUBREV_F32: VOP3Op.V_SUBREV_F32_E64, VOPDOp.V_DUAL_MUL_DX9_ZERO_F32: VOP3Op.V_MUL_DX9_ZERO_F32_E64,
  VOPDOp.V_DUAL_MOV_B32: VOP3Op.V_MOV_B32_E64, VOPDOp.V_DUAL_CNDMASK_B32: VOP3Op.V_CNDMASK_B32_E64,
  VOPDOp.V_DUAL_MAX_F32: VOP3Op.V_MAX_F32_E64, VOPDOp.V_DUAL_MIN_F32: VOP3Op.V_MIN_F32_E64,
  VOPDOp.V_DUAL_ADD_NC_U32: VOP3Op.V_ADD_NC_U32_E64, VOPDOp.V_DUAL_LSHLREV_B32: VOP3Op.V_LSHLREV_B32_E64, VOPDOp.V_DUAL_AND_B32: VOP3Op.V_AND_B32_E64,
}


class WaveState:
  __slots__ = ('sgpr', 'vgpr', 'scc', 'pc', '_pend_sgpr', 'lds', 'n_lanes')
  def __init__(self, lds: LDSMem | None = None, n_lanes: int = WAVE_SIZE):
    self.sgpr, self.vgpr = SGPRArray(SGPR_COUNT), [VGPRLane(VGPR_COUNT) for _ in range(WAVE_SIZE)]
    self.sgpr[EXEC_LO], self.scc, self.pc, self._pend_sgpr, self.lds, self.n_lanes = 0xffffffff, 0, 0, {}, lds, n_lanes

  @property
  def vcc(self) -> int: return self.sgpr[VCC_LO] | (self.sgpr[VCC_HI] << 32)
  @vcc.setter
  def vcc(self, v: int): self.sgpr[VCC_LO], self.sgpr[VCC_HI] = v & MASK32, (v >> 32) & MASK32
  @property
  def exec_mask(self) -> int: return self.sgpr[EXEC_LO] | (self.sgpr[EXEC_HI] << 32)
  @exec_mask.setter
  def exec_mask(self, v: int): self.sgpr[EXEC_LO], self.sgpr[EXEC_HI] = v & MASK32, (v >> 32) & MASK32

  def rsgpr(self, reg) -> int:
    if reg == NULL: return 0
    if reg == SCC: return self.scc
    return self.sgpr[reg]
  def wsgpr(self, reg, v: int):
    if reg != NULL: self.sgpr[reg] = v & MASK32
  def rsgpr64(self, reg) -> int:
    off = reg.offset if hasattr(reg, 'offset') else reg
    return self.rsgpr(off) | (self.rsgpr(off + 1) << 32)
  def wsgpr64(self, reg, v: int):
    off = reg.offset if hasattr(reg, 'offset') else reg
    self.wsgpr(off, v & MASK32); self.wsgpr(off + 1, (v >> 32) & MASK32)

  def _rsrc_base(self, reg, lane: int, consts, literal: int):
    v = reg.offset if hasattr(reg, 'offset') else reg
    if v < SGPR_COUNT: return self.sgpr[v]
    if v == SCC.offset: return self.scc
    if v < 255: return consts[v - 128]
    if v == 255: return literal
    return self.vgpr[lane][v] if v <= 511 else 0
  def rsrc(self, reg, lane: int, literal: int = 0) -> int: return self._rsrc_base(reg, lane, _INLINE_CONSTS, literal)
  def rsrc_f16(self, reg, lane: int, literal: int = 0) -> int: return self._rsrc_base(reg, lane, _INLINE_CONSTS_F16, literal)
  def rsrc64(self, reg, lane: int, literal: int = 0) -> int:
    v = reg.offset if hasattr(reg, 'offset') else reg
    if 128 <= v < 255: return _INLINE_CONSTS_F64[v - 128]
    if v == 255: return literal << 32  # 32-bit literal forms upper 32 bits of 64-bit value
    return self.rsrc(v, lane, literal) | ((self.rsrc(v+1, lane, literal) if v < VCC_LO.offset or 256 <= v <= 511 else 0) << 32)

  def pend_sgpr_lane(self, reg, lane: int, val: int):
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
    addr = st.rsgpr64(inst.sbase) + _sext(inst.offset, 21)
    if inst.soffset != NULL: addr += st.rsrc(inst.soffset, 0, inst._literal)
    result = inst._fn(GlobalMem, addr & MASK64)
    if 'SDATA' in result:
      sdata = result['SDATA']
      for i in range(SMEM_DST_COUNT.get(inst.op, 1)): st.wsgpr(inst.sdata.offset + i, (sdata >> (i * 32)) & MASK32)
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

def exec_vopd(st: WaveState, inst, V: VGPRLane, lane: int) -> None:
  """VOPD: dual-issue, execute two ops simultaneously (read all inputs before writes)."""
  literal, vdstx = inst._literal, inst.vdstx
  vdsty = (inst.vdsty << 1) | ((inst.vdstx.offset & 1) ^ 1)  # vdsty is raw int from VDSTYField.decode
  sx0, sx1, dx, sy0, sy1, dy = st.rsrc(inst.srcx0, lane, literal), V[inst.vsrcx1], V[vdstx], st.rsrc(inst.srcy0, lane, literal), V[inst.vsrcy1], V[vdsty]
  V[vdstx] = inst._fnx(sx0, sx1, 0, dx, st.scc, st.vcc, lane, st.exec_mask, literal, None)['D0']
  V[vdsty] = inst._fny(sy0, sy1, 0, dy, st.scc, st.vcc, lane, st.exec_mask, literal, None)['D0']

def exec_flat(st: WaveState, inst, V: VGPRLane, lane: int) -> None:
  """FLAT/GLOBAL/SCRATCH memory ops."""
  ndwords = _op_ndwords(inst.op_name)
  addr = V[inst.addr] | (V[inst.addr + 1] << 32)
  ADDR = (st.rsgpr64(inst.saddr) + V[inst.addr] + _sext(inst.offset, 13)) & MASK64 if inst.saddr != NULL else (addr + _sext(inst.offset, 13)) & MASK64
  vdata_src = inst.vdst if 'LOAD' in inst.op_name else inst.data
  result = inst._fn(GlobalMem, ADDR, _vgpr_read(V, vdata_src, ndwords), V[inst.vdst])
  if 'VDATA' in result: _vgpr_write(V, inst.vdst, result['VDATA'], ndwords)
  if 'RETURN_DATA' in result: _vgpr_write(V, inst.vdst, result['RETURN_DATA'], ndwords)

def exec_ds(st: WaveState, inst, V: VGPRLane, lane: int) -> None:
  """DS (LDS) memory ops."""
  ndwords = _op_ndwords(inst.op_name)
  data0, data1 = _vgpr_read(V, inst.data0, ndwords), _vgpr_read(V, inst.data1, ndwords) if inst.data1 is not None else 0
  result = inst._fn(st.lds, V[inst.addr], data0, data1, inst.offset0, inst.offset1)
  if 'RETURN_DATA' in result and ('_RTN' in inst.op_name or '_LOAD' in inst.op_name):
    _vgpr_write(V, inst.vdst, result['RETURN_DATA'], ndwords * 2 if '_2ADDR_' in inst.op_name else ndwords)

def exec_vop(st: WaveState, inst: Inst, V: VGPRLane, lane: int) -> None:
  """VOP1/VOP2/VOP3/VOP3SD/VOP3P/VOPC: standard ALU ops."""
  if isinstance(inst, VOP3P):
    src0, src1, src2, vdst, dst_hi = inst.src0, inst.src1, inst.src2, inst.vdst, False
    neg, abs_, opsel = inst.neg, 0, inst.opsel
  elif isinstance(inst, VOP1):
    src0, src1, src2, vdst = inst.src0, None, None, inst.vdst
    neg, abs_, opsel, dst_hi = 0, 0, 0, (inst.vdst.offset & 0x80) != 0 and inst.is_dst_16()
    if inst.is_dst_16(): vdst = inst.vdst.offset & 0x7f
  elif isinstance(inst, VOP2):
    src0, src1, src2, vdst = inst.src0, inst.vsrc1, None, inst.vdst
    neg, abs_, opsel, dst_hi = 0, 0, 0, (inst.vdst.offset & 0x80) != 0 and inst.is_dst_16()
    if inst.is_dst_16(): vdst = inst.vdst.offset & 0x7f
  elif isinstance(inst, (VOP3, VOP3SD)):
    src0, src1, src2, vdst = inst.src0, inst.src1, (None if isinstance(inst, VOP3) and inst.op.value < 256 else inst.src2), inst.vdst
    neg, abs_, opsel, dst_hi = (inst.neg, inst.abs, inst.opsel, False) if isinstance(inst, VOP3) else (0, 0, 0, False)
  elif isinstance(inst, VOPC):
    src0, src1, src2, vdst, neg, abs_, opsel, dst_hi = inst.src0, inst.vsrc1, None, VCC_LO, 0, 0, 0, False
  else:
    raise NotImplementedError(f"exec_vop: unhandled instruction type {type(inst).__name__}")

  s0 = _read_src(st, inst, src0, 0, lane, neg, abs_, opsel)
  s1 = _read_src(st, inst, src1, 1, lane, neg, abs_, opsel)
  s2 = _read_src(st, inst, src2, 2, lane, neg, abs_, opsel)
  if isinstance(inst, VOP2) and inst.is_dst_16(): d0 = _src16(V[vdst], dst_hi)
  elif inst.dst_regs() == 2: d0 = V[vdst] | (V[vdst + 1] << 32)
  else: d0 = V[vdst]

  if isinstance(inst, VOP3SD) and 'CO_CI' in inst.op_name: vcc_for_fn = st.rsgpr64(inst.src2)
  elif isinstance(inst, VOP3) and inst.op in (VOP3Op.V_CNDMASK_B32_E64, VOP3Op.V_CNDMASK_B16) and src2 is not None and src2.offset < 256: vcc_for_fn = st.rsgpr64(src2)
  else: vcc_for_fn = st.vcc
  src0_off = src0.offset if src0 is not None else 0
  src0_idx = (src0_off - 256) if src0_off >= 256 else src0_off
  vdst_off = vdst.offset if hasattr(vdst, 'offset') else vdst
  extra_kwargs = {'opsel': opsel, 'opsel_hi': inst.opsel_hi | (inst.opsel_hi2 << 2)} if isinstance(inst, VOP3P) and 'FMA_MIX' in inst.op_name else {}
  result = inst._fn(s0, s1, s2, d0, st.scc, vcc_for_fn, lane, st.exec_mask, inst._literal, st.vgpr, src0_idx, vdst_off, **extra_kwargs)

  # Check if this is a VOPC instruction (either standalone VOPC or VOP3 with VOPC opcode)
  is_vopc = isinstance(inst.op, VOPCOp) or (isinstance(inst, VOP3) and inst.op.value < 256)
  if 'VCC' in result:
    if isinstance(inst, VOP3SD): st.pend_sgpr_lane(inst.sdst, lane, (result['VCC'] >> lane) & 1)
    elif isinstance(inst, VOP2) and 'CO_CI' in inst.op_name: st.pend_sgpr_lane(VCC_LO, lane, (result['VCC'] >> lane) & 1)
    elif is_vopc: st.pend_sgpr_lane(vdst, lane, (result['VCC'] >> lane) & 1)  # vdst is VCC_LO for VOPC
    else: st.pend_sgpr_lane(VCC_LO, lane, (result['VCC'] >> lane) & 1)
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
  src0, src1, src2, vdst = inst.src0.offset, inst.src1.offset, inst.src2.offset, inst.vdst.offset
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
      st.vgpr[(i//2) % 32][vdst - 256 + (i//2)//32] = ((_i16(mat_d[i+1]) & 0xffff) << 16) | (_i16(mat_d[i]) & 0xffff)
  else:
    for i in range(256): st.vgpr[i % 32][vdst - 256 + i//32] = _i32(mat_d[i])

# ═══════════════════════════════════════════════════════════════════════════════
# PROGRAM DECODE
# ═══════════════════════════════════════════════════════════════════════════════

# Wave-level dispatch functions: (st, inst) -> return_code (0 = continue, -1 = end, -2 = barrier)
def dispatch_endpgm(st, inst): return -1
def dispatch_barrier(st, inst): st.pc += inst._words; return -2
def dispatch_nop(st, inst): st.pc += inst._words; return 0
def dispatch_wmma(st, inst): exec_wmma(st, inst, inst.op); st.pc += inst._words; return 0
def dispatch_writelane(st, inst): st.vgpr[st.rsrc(inst.src1, 0, inst._literal) & 0x1f][inst.vdst.offset - 256] = st.rsrc(inst.src0, 0, inst._literal) & MASK32; st.pc += inst._words; return 0
def dispatch_readlane(st, inst):
  src0_off = inst.src0.offset
  src0_idx = (src0_off - 256) if src0_off >= 256 else src0_off
  s1 = st.rsrc(inst.src1, 0, inst._literal) if getattr(inst, 'src1', None) is not None else 0
  result = inst._fn(0, s1, 0, 0, st.scc, st.vcc, 0, st.exec_mask, inst._literal, st.vgpr, src0_idx, inst.vdst.offset)
  st.wsgpr(inst.vdst.offset, result['D0'])
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
    inst = decode_inst(data[i:])
    inst._words = inst.size() // 4

    # Determine dispatch function and pcode function
    if isinstance(inst, SOPP) and inst.op == SOPPOp.S_CODE_END: break
    elif isinstance(inst, SOPP) and inst.op == SOPPOp.S_ENDPGM: inst._dispatch = dispatch_endpgm
    elif isinstance(inst, SOPP) and inst.op == SOPPOp.S_BARRIER: inst._dispatch = dispatch_barrier
    elif isinstance(inst, SOPP) and inst.op in (SOPPOp.S_CLAUSE, SOPPOp.S_WAITCNT, SOPPOp.S_WAITCNT_DEPCTR, SOPPOp.S_SENDMSG, SOPPOp.S_SET_INST_PREFETCH_DISTANCE, SOPPOp.S_DELAY_ALU): inst._dispatch = dispatch_nop
    elif isinstance(inst, (SOP1, SOP2, SOPC, SOPK, SOPP, SMEM)): inst._dispatch = exec_scalar
    elif isinstance(inst, VOP1) and inst.op == VOP1Op.V_NOP_E32: inst._dispatch = dispatch_nop
    elif isinstance(inst, VOP3P) and 'WMMA' in inst.op_name: inst._dispatch = dispatch_wmma
    elif isinstance(inst, VOP3) and inst.op == VOP3Op.V_WRITELANE_B32: inst._dispatch = dispatch_writelane
    elif isinstance(inst, (VOP1, VOP3)) and inst.op in (VOP1Op.V_READFIRSTLANE_B32_E32, VOP3Op.V_READFIRSTLANE_B32, VOP3Op.V_READLANE_B32): inst._dispatch = dispatch_readlane
    elif isinstance(inst, VOPD): inst._dispatch = dispatch_lane(exec_vopd)
    elif isinstance(inst, FLAT): inst._dispatch = dispatch_lane(exec_flat)
    elif isinstance(inst, DS): inst._dispatch = dispatch_lane(exec_ds)
    else: inst._dispatch = dispatch_lane(exec_vop)

    # Compile pcode for instructions that use it (not VOPD which has _fnx/_fny, not special dispatches)
    # VOPD needs separate functions for X and Y ops
    if isinstance(inst, VOPD):
      def _compile_vopd_op(op): return compile_pseudocode(type(op).__name__, op.name, PCODE[op])
      inst._fnx, inst._fny = _compile_vopd_op(_VOPD_TO_VOP[inst.opx]), _compile_vopd_op(_VOPD_TO_VOP[inst.opy])
    elif inst._dispatch not in (dispatch_endpgm, dispatch_barrier, dispatch_nop, dispatch_wmma, dispatch_writelane):
      assert type(inst.op) != int, f"inst op of {inst} is int"
      inst._fn = compile_pseudocode(type(inst.op).__name__, inst.op.name, PCODE[inst.op])
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
