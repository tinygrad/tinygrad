# RDNA3 emulator - executes compiled pseudocode from AMD ISA PDF
# mypy: ignore-errors
from __future__ import annotations
import ctypes
from extra.assembly.amd.dsl import Inst, unwrap, FLOAT_ENC, MASK32, MASK64, _f32, _i32, _sext, _f16, _i16, _f64, _i64
from extra.assembly.amd.pcode import Reg
from extra.assembly.amd.asm import detect_format
from extra.assembly.amd.autogen.rdna3.gen_pcode import get_compiled_functions
from extra.assembly.amd.autogen.rdna3.ins import (SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, VOP1, VOP2, VOP3, VOP3SD, VOP3P, VOPC, DS, FLAT, VOPD,
  SrcEnum, SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, SMEMOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp, DSOp, FLATOp, GLOBALOp, SCRATCHOp, VOPDOp)

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

# Helper: get number of dwords from memory op name
def _op_ndwords(name: str) -> int:
  if '_B128' in name: return 4
  if '_B96' in name: return 3
  if any(s in name for s in ('_B64', '_U64', '_I64', '_F64')): return 2
  return 1

# Helper: build multi-dword Reg from consecutive VGPRs
def _vgpr_read(V: list, base: int, ndwords: int) -> Reg: return Reg(sum(V[base + i] << (32 * i) for i in range(ndwords)))

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

  # SOPP: special cases for control flow that has no pseudocode
  if isinstance(inst, SOPP):
    if inst.op == SOPPOp.S_ENDPGM: return -1
    if inst.op == SOPPOp.S_BARRIER: return -2

  # SMEM: memory loads (not ALU)
  if isinstance(inst, SMEM):
    addr = st.rsgpr64(inst.sbase * 2) + _sext(inst.offset, 21)
    if inst.soffset not in (NULL, 0x7f): addr += st.rsrc(inst.soffset, 0)
    if (cnt := SMEM_LOAD.get(inst.op)) is None: raise NotImplementedError(f"SMEM op {inst.op}")
    for i in range(cnt): st.wsgpr(inst.sdata + i, mem_read((addr + i * 4) & MASK64, 4))
    return 0

  # Get op enum and lookup compiled function
  if isinstance(inst, SOP1): ssrc0, sdst = inst.ssrc0, inst.sdst
  elif isinstance(inst, SOP2): ssrc0, sdst = inst.ssrc0, inst.sdst
  elif isinstance(inst, SOPC): ssrc0, sdst = inst.ssrc0, None
  elif isinstance(inst, SOPK): ssrc0, sdst = inst.sdst, inst.sdst  # sdst is both src and dst
  elif isinstance(inst, SOPP): ssrc0, sdst = None, None
  else: raise NotImplementedError(f"Unknown scalar type {type(inst)}")

  # SOPP has gaps in the opcode enum - treat unknown opcodes as no-ops
  try: op = inst.op
  except ValueError:
    if isinstance(inst, SOPP): return 0
    raise
  fn = compiled.get(type(op), {}).get(op)
  if fn is None:
    # SOPP instructions without pseudocode (waits, hints, nops) are no-ops
    if isinstance(inst, SOPP): return 0
    raise NotImplementedError(f"{op.name} not in pseudocode")

  # Build context - use inst methods to determine operand sizes
  s0 = st.rsrc64(ssrc0, 0) if inst.is_src_64(0) else (st.rsrc(ssrc0, 0) if not isinstance(inst, (SOPK, SOPP)) else (st.rsgpr(inst.sdst) if isinstance(inst, SOPK) else 0))
  s1 = st.rsrc64(inst.ssrc1, 0) if inst.is_src_64(1) else (st.rsrc(inst.ssrc1, 0) if isinstance(inst, (SOP2, SOPC)) else inst.simm16 if isinstance(inst, SOPK) else 0)
  d0 = st.rsgpr64(sdst) if inst.dst_regs() == 2 and sdst is not None else (st.rsgpr(sdst) if sdst is not None else 0)
  literal = inst.simm16 if isinstance(inst, (SOPK, SOPP)) else st.literal

  # Create Reg objects for compiled function - mask VCC/EXEC to 32 bits for wave32
  result = fn(Reg(s0), Reg(s1), None, Reg(d0), Reg(st.scc), Reg(st.vcc & MASK32), 0, Reg(st.exec_mask & MASK32), literal, None, PC=Reg(st.pc * 4))

  # Apply results - extract values from returned Reg objects
  if sdst is not None and 'D0' in result:
    (st.wsgpr64 if inst.dst_regs() == 2 else st.wsgpr)(sdst, result['D0']._val)
  if 'SCC' in result: st.scc = result['SCC']._val & 1
  if 'EXEC' in result: st.exec_mask = result['EXEC']._val
  if 'PC' in result:
    # Convert absolute byte address to word delta
    pc_val = result['PC']._val
    new_pc = pc_val if pc_val < 0x8000000000000000 else pc_val - 0x10000000000000000
    new_pc_words = new_pc // 4
    return new_pc_words - st.pc - 1  # -1 because emulator adds inst_words (1 for scalar)
  return 0

def exec_vector(st: WaveState, inst: Inst, lane: int, lds: LDSMem | None = None) -> None:
  """Execute vector instruction for one lane."""
  compiled = _get_compiled()
  V = st.vgpr[lane]

  # Memory ops (FLAT/GLOBAL/SCRATCH and DS) - use generated pcode
  if isinstance(inst, (FLAT, DS)):
    op, vdst, op_name = inst.op, inst.vdst, inst.op.name
    fn, ndwords = compiled[type(op)][op], _op_ndwords(op_name)
    if isinstance(inst, FLAT):
      addr = V[inst.addr] | (V[inst.addr + 1] << 32)
      ADDR = (st.rsgpr64(inst.saddr) + V[inst.addr] + _sext(inst.offset, 13)) & MASK64 if inst.saddr not in (NULL, 0x7f) else (addr + _sext(inst.offset, 13)) & MASK64
      # For loads, VDATA comes from vdst (preserves unwritten bits); for stores, from inst.data
      vdata_src = vdst if 'LOAD' in op_name else inst.data
      result = fn(GlobalMem, ADDR, _vgpr_read(V, vdata_src, ndwords), Reg(V[vdst]), Reg(0))
      if 'VDATA' in result: _vgpr_write(V, vdst, result['VDATA']._val, ndwords)
      if 'RETURN_DATA' in result: _vgpr_write(V, vdst, result['RETURN_DATA']._val, ndwords)
    else:  # DS
      DATA0, DATA1 = _vgpr_read(V, inst.data0, ndwords), _vgpr_read(V, inst.data1, ndwords) if inst.data1 is not None else Reg(0)
      result = fn(lds, Reg(V[inst.addr]), DATA0, DATA1, Reg(inst.offset0), Reg(inst.offset1), Reg(0))
      if 'RETURN_DATA' in result and ('_RTN' in op_name or '_LOAD' in op_name):
        _vgpr_write(V, vdst, result['RETURN_DATA']._val, ndwords * 2 if '_2ADDR_' in op_name else ndwords)
    return

  # VOPD: dual-issue, execute two ops simultaneously (read all inputs before writes)
  if isinstance(inst, VOPD):
    vdsty = (inst.vdsty << 1) | ((inst.vdstx & 1) ^ 1)
    inputs = [(inst.opx, st.rsrc(inst.srcx0, lane), V[inst.vsrcx1], V[inst.vdstx], inst.vdstx),
              (inst.opy, st.rsrc(inst.srcy0, lane), V[inst.vsrcy1], V[vdsty], vdsty)]
    def exec_vopd(vopd_op, s0, s1, d0):
      op = _VOPD_TO_VOP[vopd_op]
      return compiled[type(op)][op](Reg(s0), Reg(s1), None, Reg(d0), Reg(st.scc), Reg(st.vcc), lane, Reg(st.exec_mask), st.literal, None)['D0']._val
    for vopd_op, s0, s1, d0, dst in inputs: V[dst] = exec_vopd(vopd_op, s0, s1, d0)
    return

  # VOP3SD: has extra scalar dest for carry output
  if isinstance(inst, VOP3SD):
    fn = compiled[VOP3SDOp][inst.op]
    # Read sources based on register counts from inst properties
    def rsrc_n(src, regs): return st.rsrc64(src, lane) if regs == 2 else st.rsrc(src, lane)
    s0, s1, s2 = rsrc_n(inst.src0, inst.src_regs(0)), rsrc_n(inst.src1, inst.src_regs(1)), rsrc_n(inst.src2, inst.src_regs(2))
    # Carry-in ops use src2 as carry bitmask instead of VCC
    vcc = st.rsgpr64(inst.src2) if 'CO_CI' in inst.op_name else st.vcc
    result = fn(Reg(s0), Reg(s1), Reg(s2), Reg(V[inst.vdst]), Reg(st.scc), Reg(vcc), lane, Reg(st.exec_mask), st.literal, None)
    d0_val = result['D0']._val
    V[inst.vdst] = d0_val & MASK32
    if inst.dst_regs() == 2: V[inst.vdst + 1] = (d0_val >> 32) & MASK32
    if 'VCC' in result: st.pend_sgpr_lane(inst.sdst, lane, (result['VCC']._val >> lane) & 1)
    return

  # Get op enum and sources (None means "no source" for that operand)
  # dst_hi: for VOP1/VOP2 16-bit dst ops, bit 7 of vdst indicates .h (high 16-bit) destination
  dst_hi = False
  if isinstance(inst, VOP1):
    if inst.op == VOP1Op.V_NOP: return
    src0, src1, src2 = inst.src0, None, None
    dst_hi = (inst.vdst & 0x80) != 0 and inst.is_dst_16()
    vdst = inst.vdst & 0x7f if inst.is_dst_16() else inst.vdst
  elif isinstance(inst, VOP2):
    src0, src1, src2 = inst.src0, inst.vsrc1 + 256, None
    dst_hi = (inst.vdst & 0x80) != 0 and inst.is_dst_16()
    vdst = inst.vdst & 0x7f if inst.is_dst_16() else inst.vdst
  elif isinstance(inst, VOP3):
    # VOP3 ops 0-255 are VOPC comparisons encoded as VOP3 - inst.op returns VOPCOp for these
    src0, src1, src2, vdst = inst.src0, inst.src1, (None if inst.op.value < 256 else inst.src2), inst.vdst
  elif isinstance(inst, VOPC):
    # For 16-bit VOPC, vsrc1 uses same encoding as VOP2 16-bit: bit 7 selects hi(1) or lo(0) half
    # vsrc1 field is 8 bits: [6:0] = VGPR index, [7] = hi flag
    src0, src1, src2, vdst = inst.src0, inst.vsrc1 + 256, None, VCC_LO
  elif isinstance(inst, VOP3P):
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
      st.vgpr[lane][inst.vdst] = _i32(result) if inst.op == VOP3POp.V_FMA_MIX_F32 else _dst16(V[inst.vdst], _i16(result), inst.op == VOP3POp.V_FMA_MIXHI_F16)
      return
    # VOP3P packed ops: opsel selects halves for lo, opsel_hi for hi; neg toggles f16 sign
    raws = [st.rsrc_f16(inst.src0, lane), st.rsrc_f16(inst.src1, lane), st.rsrc_f16(inst.src2, lane) if inst.src2 is not None else 0]
    opsel, opsel_hi, opsel_hi2 = getattr(inst, 'opsel', 0), getattr(inst, 'opsel_hi', 3), getattr(inst, 'opsel_hi2', 1)
    neg, neg_hi = getattr(inst, 'neg', 0), getattr(inst, 'neg_hi', 0)
    hi_sels = [opsel_hi & 1, opsel_hi & 2, opsel_hi2]
    srcs = [((_src16(raws[i], hi_sels[i]) ^ (0x8000 if neg_hi & (1<<i) else 0)) << 16) |
            (_src16(raws[i], opsel & (1<<i)) ^ (0x8000 if neg & (1<<i) else 0)) for i in range(3)]
    result = compiled[VOP3POp][inst.op](Reg(srcs[0]), Reg(srcs[1]), Reg(srcs[2]), Reg(0), Reg(st.scc), Reg(st.vcc), lane, Reg(st.exec_mask), st.literal, None)
    st.vgpr[lane][inst.vdst] = result['D0']._val & MASK32
    return
  else: raise NotImplementedError(f"Unknown vector type {type(inst)}")

  op_cls = type(inst.op)
  if (fn := compiled.get(op_cls, {}).get(inst.op)) is None: raise NotImplementedError(f"{inst.op_name} not in pseudocode")

  # Read sources (with VOP3 modifiers if applicable)
  neg, abs_ = (getattr(inst, 'neg', 0), getattr(inst, 'abs', 0)) if isinstance(inst, VOP3) else (0, 0)
  opsel = getattr(inst, 'opsel', 0) if isinstance(inst, VOP3) else 0
  def mod_src(val: int, idx: int, is64=False) -> int:
    to_f, to_i = (_f64, _i64) if is64 else (_f32, _i32)
    if (abs_ >> idx) & 1: val = to_i(abs(to_f(val)))
    if (neg >> idx) & 1: val = to_i(-to_f(val))
    return val

  # Use inst methods to determine operand sizes (inst.is_src_16, inst.is_src_64, etc.)
  is_vop2_16bit = isinstance(inst, VOP2) and inst.is_16bit()

  # Read sources based on register counts and dtypes from inst properties
  def read_src(src, idx, regs, is_src_16):
    if src is None: return 0
    if regs == 2: return mod_src(st.rsrc64(src, lane), idx, is64=True)
    if is_src_16 and isinstance(inst, VOP3):
      raw = st.rsrc_f16(src, lane) if 128 <= src < 255 else st.rsrc(src, lane)
      val = _src16(raw, bool(opsel & (1 << idx)))
      if abs_ & (1 << idx): val &= 0x7fff
      if neg & (1 << idx): val ^= 0x8000
      return val
    if is_src_16 and isinstance(inst, (VOP1, VOP2, VOPC)):
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
  vcc_for_fn = st.rsgpr64(src2) if inst.op in (VOP3Op.V_CNDMASK_B32, VOP3Op.V_CNDMASK_B16) and isinstance(inst, VOP3) and src2 is not None and src2 < 256 else st.vcc

  # Execute compiled function - pass src0_idx and vdst_idx for lane instructions
  # For VGPR access: src0 index is the VGPR number (src0 - 256 if VGPR, else src0 for SGPR)
  src0_idx = (src0 - 256) if src0 is not None and src0 >= 256 else (src0 if src0 is not None else 0)
  result = fn(Reg(s0), Reg(s1), Reg(s2), Reg(d0), Reg(st.scc), Reg(vcc_for_fn), lane, Reg(st.exec_mask), st.literal, st.vgpr, src0_idx, vdst)

  # Apply results - extract values from returned Reg objects
  if 'vgpr_write' in result:
    # Lane instruction wrote to VGPR: (lane, vgpr_idx, value)
    wr_lane, wr_idx, wr_val = result['vgpr_write']
    st.vgpr[wr_lane][wr_idx] = wr_val
  if 'VCC' in result:
    # VOP2 carry ops write to VCC implicitly; VOPC/VOP3 write to vdst
    st.pend_sgpr_lane(VCC_LO if isinstance(inst, VOP2) and 'CO_CI' in inst.op_name else vdst, lane, (result['VCC']._val >> lane) & 1)
  if 'EXEC' in result:
    # V_CMPX instructions write to EXEC per-lane (not to vdst)
    st.pend_sgpr_lane(EXEC_LO, lane, (result['EXEC']._val >> lane) & 1)
  elif op_cls is VOPCOp:
    # VOPC comparison result stored in D0 bitmask, extract lane bit (non-CMPX only)
    st.pend_sgpr_lane(vdst, lane, (result['D0']._val >> lane) & 1)
  if op_cls is not VOPCOp and 'vgpr_write' not in result:
    writes_to_sgpr = 'READFIRSTLANE' in inst.op_name or 'READLANE' in inst.op_name
    d0_val = result['D0']._val
    if writes_to_sgpr: st.wsgpr(vdst, d0_val & MASK32)
    elif inst.dst_regs() == 2: V[vdst], V[vdst + 1] = d0_val & MASK32, (d0_val >> 32) & MASK32
    elif inst.is_dst_16(): V[vdst] = _dst16(V[vdst], d0_val, bool(opsel & 8) if isinstance(inst, VOP3) else dst_hi)
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
# SQTT TRACING
# ═══════════════════════════════════════════════════════════════════════════════

WAVESTART_TO_INST_CYCLES = 32  # cycles from WAVESTART to first instruction

# Issue intervals (fixed, independent of lane count)
VALU_ISSUE_CYCLES = 1
TRANS_ISSUE_CYCLES = 4
DP_ISSUE_CYCLES = 32
SALU_ISSUE_CYCLES = 1

# ALU latencies (cycles from dispatch to result ready / ALUEXEC)
VALU_LATENCY = 6
SALU_LATENCY = 2
TRANS_LATENCY = 9
DP_LATENCY = 38

# Pipeline delay from last ALU dispatch to first s_nop IMMEDIATE
SNOP_PIPELINE_DELAY = 3

# Forwarding latencies (cycles until result available for dependent instruction)
VALU_FORWARD_LATENCY = 5  # result available 5 cycles after dispatch (writeback at 6)
TRANS_FORWARD_LATENCY = 13  # result available 13 cycles after dispatch
SALU_FORWARD_LATENCY = 1  # result available 1 cycle after dispatch (writeback at 2)

# Transcendental ops (use TRANS unit)
_TRANS_OPS = {'V_RCP_F32', 'V_RCP_F64', 'V_RSQ_F32', 'V_RSQ_F64', 'V_SQRT_F32', 'V_SQRT_F64',
              'V_LOG_F32', 'V_EXP_F32', 'V_SIN_F32', 'V_COS_F32', 'V_RCP_F16', 'V_RSQ_F16', 'V_SQRT_F16'}

# Double precision ops (use DP unit)
_DP_OPS = {'V_ADD_F64', 'V_MUL_F64', 'V_FMA_F64', 'V_DIV_F64', 'V_MIN_F64', 'V_MAX_F64',
           'V_LDEXP_F64', 'V_FREXP_MANT_F64', 'V_FREXP_EXP_I32_F64', 'V_FRACT_F64',
           'V_TRUNC_F64', 'V_CEIL_F64', 'V_RNDNE_F64', 'V_FLOOR_F64', 'V_DIV_SCALE_F64',
           'V_DIV_FMAS_F64', 'V_DIV_FIXUP_F64', 'V_CVT_F64_I32', 'V_CVT_F64_U32',
           'V_CVT_I32_F64', 'V_CVT_U32_F64', 'V_CVT_F32_F64', 'V_CVT_F64_F32'}

class SQTTState:
  """SQTT tracing state - emits packets when instructions dispatch."""

  def __init__(self, wave_id: int = 0, simd: int = 0, cu: int = 0):
    self.packets = []
    self.wave_id, self.simd, self.cu = wave_id, simd, cu
    self.cycle = 0

  def emit(self, pkt_class, **kwargs):
    self.packets.append(pkt_class(_time=self.cycle, **kwargs))

  def emit_wavestart(self):
    from extra.assembly.amd.sqtt import WAVESTART
    self.emit(WAVESTART, wave=self.wave_id, simd=self.simd, cu_lo=self.cu & 0x7, flag7=self.cu >> 3)
    for _ in range(WAVESTART_TO_INST_CYCLES):
      self.tick()

  def emit_waveend(self):
    from extra.assembly.amd.sqtt import WAVEEND
    self.emit(WAVEEND, wave=self.wave_id, simd=self.simd, cu_lo=self.cu & 0x7, flag7=self.cu >> 3)

  def tick(self):
    """Process one cycle: emit any completing ALUEXECs, then advance cycle."""
    self.cycle += 1

  def process_instruction(self, inst: Inst):
    """Simulate cycles until instruction dispatches, emitting SQTT packets."""
    pass

  def finalize(self):
    """Emit pending ALUEXECs and WAVEEND."""
    # Emit any remaining ALUEXECs
    self.emit_waveend()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def step_wave(program: Program, st: WaveState, lds: LDSMem, n_lanes: int, trace: SQTTState | None = None) -> int:
  inst = program.get(st.pc)
  if inst is None: return 1
  inst_words, st.literal = inst._words, getattr(inst, '_literal', None) or 0

  # TODO: add ALUEXEC emits if anything completed

  # Emit SQTT packets for this instruction
  if trace is not None:
    trace.process_instruction(inst)

  if isinstance(inst, (SOP1, SOP2, SOPC, SOPK, SOPP, SMEM)):
    delta = exec_scalar(st, inst)
    if delta == -1: return -1  # endpgm
    if delta == -2: st.pc += inst_words; return -2  # barrier
    st.pc += inst_words + delta
  else:
    # V_READFIRSTLANE/V_READLANE write to SGPR, execute once; others execute per-lane with exec_mask
    is_readlane = isinstance(inst, (VOP1, VOP3)) and ('READFIRSTLANE' in inst.op_name or 'READLANE' in inst.op_name)
    exec_mask = 1 if is_readlane else st.exec_mask
    for lane in range(1 if is_readlane else n_lanes):
      if exec_mask & (1 << lane): exec_vector(st, inst, lane, lds)
    st.commit_pends()
    st.pc += inst_words
  return 0

def exec_wave(program: Program, st: WaveState, lds: LDSMem, n_lanes: int, trace: SQTTState | None = None) -> int:
  if trace is not None:
    trace.emit_wavestart()
  while st.pc in program:
    result = step_wave(program, st, lds, n_lanes, trace)
    if result == -1: break
    if result == -2: return -2
  if trace is not None:
    trace.finalize()
  return 0

def exec_workgroup(program: Program, workgroup_id: tuple[int, int, int], local_size: tuple[int, int, int], args_ptr: int,
                   wg_id_sgpr_base: int, wg_id_enables: tuple[bool, bool, bool]) -> None:
  lx, ly, lz = local_size
  total_threads, lds = lx * ly * lz, LDSMem(bytearray(65536))
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
