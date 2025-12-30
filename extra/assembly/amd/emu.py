# RDNA3 emulator - executes compiled pseudocode from AMD ISA PDF
# mypy: ignore-errors
from __future__ import annotations
import ctypes, os
from extra.assembly.amd.dsl import Inst, RawImm
from extra.assembly.amd.pcode import _f32, _i32, _sext, _f16, _i16, _f64, _i64, Reg
from extra.assembly.amd.autogen.rdna3.gen_pcode import get_compiled_functions
from extra.assembly.amd.autogen.rdna3 import (
  SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, VOP1, VOP2, VOP3, VOP3SD, VOP3P, VOPC, DS, FLAT, VOPD, SrcEnum,
  SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, SMEMOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp, DSOp, FLATOp, GLOBALOp, VOPDOp
)

Program = dict[int, Inst]
WAVE_SIZE, SGPR_COUNT, VGPR_COUNT = 32, 128, 256
VCC_LO, VCC_HI, NULL, EXEC_LO, EXEC_HI, SCC = SrcEnum.VCC_LO, SrcEnum.VCC_HI, SrcEnum.NULL, SrcEnum.EXEC_LO, SrcEnum.EXEC_HI, SrcEnum.SCC

# VOP3 ops that use 64-bit operands (and thus 64-bit literals when src is 255)
# Exception: V_LDEXP_F64 has 32-bit integer src1, so literal should NOT be 64-bit when src1=255
_VOP3_64BIT_OPS = {op.value for op in VOP3Op if op.name.endswith(('_F64', '_B64', '_I64', '_U64'))}
# Ops where src1 is 32-bit (exponent/shift amount) even though the op name suggests 64-bit
_VOP3_64BIT_OPS_32BIT_SRC1 = {VOP3Op.V_LDEXP_F64.value}
# Ops with 16-bit types in name (for source/dest handling)
# Exception: SAD/MSAD ops take 32-bit packed sources and extract 16-bit/8-bit chunks internally
_VOP3_16BIT_OPS = {op for op in VOP3Op if any(s in op.name for s in ('_F16', '_B16', '_I16', '_U16')) and 'SAD' not in op.name}
_VOP1_16BIT_OPS = {op for op in VOP1Op if any(s in op.name for s in ('_F16', '_B16', '_I16', '_U16'))}
_VOP2_16BIT_OPS = {op for op in VOP2Op if any(s in op.name for s in ('_F16', '_B16', '_I16', '_U16'))}
# CVT ops with 32/64-bit source (despite 16-bit in name)
_CVT_32_64_SRC_OPS = {op for op in VOP3Op if op.name.startswith('V_CVT_') and op.name.endswith(('_F32', '_I32', '_U32', '_F64', '_I64', '_U64'))} | \
                     {op for op in VOP1Op if op.name.startswith('V_CVT_') and op.name.endswith(('_F32', '_I32', '_U32', '_F64', '_I64', '_U64'))}
# 16-bit dst ops (PACK has 32-bit dst despite F16 in name)
_VOP3_16BIT_DST_OPS = {op for op in _VOP3_16BIT_OPS if 'PACK' not in op.name}
_VOP1_16BIT_DST_OPS = {op for op in _VOP1_16BIT_OPS if 'PACK' not in op.name}

# Inline constants for src operands 128-254. Build tables for f32, f16, and f64 formats.
import struct as _struct
_FLOAT_CONSTS = {SrcEnum.POS_HALF: 0.5, SrcEnum.NEG_HALF: -0.5, SrcEnum.POS_ONE: 1.0, SrcEnum.NEG_ONE: -1.0,
                 SrcEnum.POS_TWO: 2.0, SrcEnum.NEG_TWO: -2.0, SrcEnum.POS_FOUR: 4.0, SrcEnum.NEG_FOUR: -4.0, SrcEnum.INV_2PI: 0.15915494309189535}
def _build_inline_consts(neg_mask, float_to_bits):
  tbl = list(range(65)) + [((-i) & neg_mask) for i in range(1, 17)] + [0] * (127 - 81)
  for k, v in _FLOAT_CONSTS.items(): tbl[k - 128] = float_to_bits(v)
  return tbl
_INLINE_CONSTS = _build_inline_consts(0xffffffff, lambda f: _struct.unpack('<I', _struct.pack('<f', f))[0])
_INLINE_CONSTS_F16 = _build_inline_consts(0xffff, lambda f: _struct.unpack('<H', _struct.pack('<e', f))[0])
_INLINE_CONSTS_F64 = _build_inline_consts(0xffffffffffffffff, lambda f: _struct.unpack('<Q', _struct.pack('<d', f))[0])

# Memory access
_valid_mem_ranges: list[tuple[int, int]] = []
def set_valid_mem_ranges(ranges: set[tuple[int, int]]) -> None: _valid_mem_ranges.clear(); _valid_mem_ranges.extend(ranges)
def _mem_valid(addr: int, size: int) -> bool:
  for s, z in _valid_mem_ranges:
    if s <= addr and addr + size <= s + z: return True
  return not _valid_mem_ranges
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
  __slots__ = ('sgpr', 'vgpr', 'scc', 'pc', 'literal', '_pend_sgpr', '_scc_reg', '_vcc_reg', '_exec_reg')
  def __init__(self):
    self.sgpr = [Reg(0) for _ in range(SGPR_COUNT)]
    self.vgpr = [[Reg(0) for _ in range(VGPR_COUNT)] for _ in range(WAVE_SIZE)]
    self.sgpr[EXEC_LO]._val = 0xffffffff
    self.scc, self.pc, self.literal, self._pend_sgpr = 0, 0, 0, {}
    # Reg wrappers for pseudocode access
    self._scc_reg = Reg(0)
    self._vcc_reg = self.sgpr[VCC_LO]
    self._exec_reg = self.sgpr[EXEC_LO]

  @property
  def vcc(self) -> int: return self.sgpr[VCC_LO]._val | (self.sgpr[VCC_HI]._val << 32)
  @vcc.setter
  def vcc(self, v: int): self.sgpr[VCC_LO]._val, self.sgpr[VCC_HI]._val = v & 0xffffffff, (v >> 32) & 0xffffffff
  @property
  def exec_mask(self) -> int: return self.sgpr[EXEC_LO]._val | (self.sgpr[EXEC_HI]._val << 32)
  @exec_mask.setter
  def exec_mask(self, v: int): self.sgpr[EXEC_LO]._val, self.sgpr[EXEC_HI]._val = v & 0xffffffff, (v >> 32) & 0xffffffff

  def rsgpr(self, i: int) -> int: return 0 if i == NULL else self.scc if i == SCC else self.sgpr[i]._val if i < SGPR_COUNT else 0
  def wsgpr(self, i: int, v: int):
    if i < SGPR_COUNT and i != NULL: self.sgpr[i]._val = v & 0xffffffff
  def rsgpr64(self, i: int) -> int: return self.rsgpr(i) | (self.rsgpr(i+1) << 32)
  def wsgpr64(self, i: int, v: int): self.wsgpr(i, v & 0xffffffff); self.wsgpr(i+1, (v >> 32) & 0xffffffff)

  def rsrc(self, v: int, lane: int) -> int:
    if v < SGPR_COUNT: return self.sgpr[v]._val
    if v == SCC: return self.scc
    if v < 255: return _INLINE_CONSTS[v - 128]
    if v == 255: return self.literal
    return self.vgpr[lane][v - 256]._val if v <= 511 else 0

  def rsrc_reg(self, v: int, lane: int) -> Reg:
    """Return the Reg object for a source operand."""
    if v < SGPR_COUNT: return self.sgpr[v]
    if v == SCC: self._scc_reg._val = self.scc; return self._scc_reg
    if v < 255: return Reg(_INLINE_CONSTS[v - 128])
    if v == 255: return Reg(self.literal)
    return self.vgpr[lane][v - 256] if v <= 511 else Reg(0)

  def rsrc_f16(self, v: int, lane: int) -> int:
    """Read source operand for VOP3P packed f16 operations. Uses f16 inline constants."""
    if v < SGPR_COUNT: return self.sgpr[v]._val
    if v == SCC: return self.scc
    if v < 255: return _INLINE_CONSTS_F16[v - 128]
    if v == 255: return self.literal
    return self.vgpr[lane][v - 256]._val if v <= 511 else 0

  def rsrc_reg_f16(self, v: int, lane: int) -> Reg:
    """Return Reg for VOP3P source. Inline constants are f16 in low 16 bits only."""
    if v < SGPR_COUNT: return self.sgpr[v]
    if v == SCC: self._scc_reg._val = self.scc; return self._scc_reg
    if v < 255: return Reg(_INLINE_CONSTS_F16[v - 128])  # f16 inline constant
    if v == 255: return Reg(self.literal)
    return self.vgpr[lane][v - 256] if v <= 511 else Reg(0)

  def rsrc64(self, v: int, lane: int) -> int:
    """Read 64-bit source operand. For inline constants, returns 64-bit representation."""
    if 128 <= v < 255: return _INLINE_CONSTS_F64[v - 128]
    if v == 255: return self.literal
    return self.rsrc(v, lane) | ((self.rsrc(v+1, lane) if v < VCC_LO or 256 <= v <= 511 else 0) << 32)

  def rsrc_reg64(self, v: int, lane: int) -> Reg:
    """Return Reg for 64-bit source operand. For inline constants, returns 64-bit f64 value."""
    if 128 <= v < 255: return Reg(_INLINE_CONSTS_F64[v - 128])
    if v == 255: return Reg(self.literal)
    if v < SGPR_COUNT: return Reg(self.sgpr[v]._val | (self.sgpr[v+1]._val << 32))
    if 256 <= v <= 511:
      vgpr_idx = v - 256
      return Reg(self.vgpr[lane][vgpr_idx]._val | (self.vgpr[lane][vgpr_idx + 1]._val << 32))
    return Reg(0)

  def pend_sgpr_lane(self, reg: int, lane: int, val: int):
    if reg not in self._pend_sgpr: self._pend_sgpr[reg] = 0
    if val: self._pend_sgpr[reg] |= (1 << lane)
  def commit_pends(self):
    for reg, val in self._pend_sgpr.items(): self.sgpr[reg]._val = val
    self._pend_sgpr.clear()

# Instruction decode
def decode_format(word: int) -> tuple[type[Inst] | None, bool]:
  hi2 = (word >> 30) & 0x3
  if hi2 == 0b11:
    enc = (word >> 26) & 0xf
    if enc == 0b1101: return SMEM, True
    if enc == 0b0101:
      op = (word >> 16) & 0x3ff
      return (VOP3SD, True) if op in (288, 289, 290, 764, 765, 766, 767, 768, 769, 770) else (VOP3, True)
    return {0b0011: (VOP3P, True), 0b0110: (DS, True), 0b0111: (FLAT, True), 0b0010: (VOPD, True)}.get(enc, (None, True))
  if hi2 == 0b10:
    enc = (word >> 23) & 0x7f
    return {0b1111101: (SOP1, False), 0b1111110: (SOPC, False), 0b1111111: (SOPP, False)}.get(enc, (SOPK, False) if ((word >> 28) & 0xf) == 0b1011 else (SOP2, False))
  enc = (word >> 25) & 0x7f
  return (VOPC, False) if enc == 0b0111110 else (VOP1, False) if enc == 0b0111111 else (VOP2, False)

def _unwrap(v) -> int: return v.val if isinstance(v, RawImm) else v.value if hasattr(v, 'value') else v

def decode_program(data: bytes) -> Program:
  result: Program = {}
  i = 0
  while i < len(data):
    word = int.from_bytes(data[i:i+4], 'little')
    inst_class, is_64 = decode_format(word)
    if inst_class is None: i += 4; continue
    base_size = 8 if is_64 else 4
    # Pass enough data for potential 64-bit literal (base + 8 bytes max)
    inst = inst_class.from_bytes(data[i:i+base_size+8])
    for name, val in inst._values.items(): setattr(inst, name, _unwrap(val))
    # from_bytes already handles literal reading - only need fallback for cases it doesn't handle
    if inst._literal is None:
      has_literal = any(getattr(inst, fld, None) == 255 for fld in ('src0', 'src1', 'src2', 'ssrc0', 'ssrc1', 'srcx0', 'srcy0'))
      if inst_class == VOP2 and inst.op in (44, 45, 55, 56): has_literal = True
      if inst_class == VOPD and (inst.opx in (1, 2) or inst.opy in (1, 2)): has_literal = True
      if inst_class == SOP2 and inst.op in (69, 70): has_literal = True
      if has_literal:
        # For 64-bit ops, the 32-bit literal is placed in HIGH 32 bits (low 32 bits = 0)
        # Exception: some ops have mixed src sizes (e.g., V_LDEXP_F64 has 32-bit src1)
        op_val = inst._values.get('op')
        if hasattr(op_val, 'value'): op_val = op_val.value
        is_64bit = inst_class is VOP3 and op_val in _VOP3_64BIT_OPS
        # Don't treat literal as 64-bit if the op has 32-bit src1 and src1 is the literal
        if is_64bit and op_val in _VOP3_64BIT_OPS_32BIT_SRC1 and getattr(inst, 'src1', None) == 255:
          is_64bit = False
        lit32 = int.from_bytes(data[i+base_size:i+base_size+4], 'little')
        inst._literal = (lit32 << 32) if is_64bit else lit32
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

  # SOPP: control flow (not ALU)
  if inst_type is SOPP:
    op = inst.op
    if op == SOPPOp.S_ENDPGM: return -1
    if op == SOPPOp.S_BARRIER: return -2
    if op == SOPPOp.S_BRANCH: return _sext(inst.simm16, 16)
    if op == SOPPOp.S_CBRANCH_SCC0: return _sext(inst.simm16, 16) if st.scc == 0 else 0
    if op == SOPPOp.S_CBRANCH_SCC1: return _sext(inst.simm16, 16) if st.scc == 1 else 0
    if op == SOPPOp.S_CBRANCH_VCCZ: return _sext(inst.simm16, 16) if (st.vcc & 0xffffffff) == 0 else 0
    if op == SOPPOp.S_CBRANCH_VCCNZ: return _sext(inst.simm16, 16) if (st.vcc & 0xffffffff) != 0 else 0
    if op == SOPPOp.S_CBRANCH_EXECZ: return _sext(inst.simm16, 16) if st.exec_mask == 0 else 0
    if op == SOPPOp.S_CBRANCH_EXECNZ: return _sext(inst.simm16, 16) if st.exec_mask != 0 else 0
    # Valid SOPP range is 0-61 (max defined opcode); anything above is invalid
    if op > 61: raise NotImplementedError(f"Invalid SOPP opcode {op}")
    return 0  # waits, hints, nops

  # SMEM: memory loads (not ALU)
  if inst_type is SMEM:
    addr = st.rsgpr64(inst.sbase * 2) + _sext(inst.offset, 21)
    if inst.soffset not in (NULL, 0x7f): addr += st.rsrc(inst.soffset, 0)
    if (cnt := SMEM_LOAD.get(inst.op)) is None: raise NotImplementedError(f"SMEM op {inst.op}")
    for i in range(cnt): st.wsgpr(inst.sdata + i, mem_read((addr + i * 4) & 0xffffffffffffffff, 4))
    return 0

  # SOP1: special handling for ops not in pseudocode
  if inst_type is SOP1:
    op = SOP1Op(inst.op)
    # S_GETPC_B64: Get program counter (PC is stored as byte offset, convert from words)
    if op == SOP1Op.S_GETPC_B64:
      pc_bytes = st.pc * 4  # PC is in words, convert to bytes
      st.wsgpr64(inst.sdst, pc_bytes)
      return 0
    # S_SETPC_B64: Set program counter to source value (indirect jump)
    # Returns delta such that st.pc + inst_words + delta = target_words
    if op == SOP1Op.S_SETPC_B64:
      target_bytes = st.rsrc64(inst.ssrc0, 0)
      target_words = target_bytes // 4
      inst_words = 1  # SOP1 is always 1 word
      return target_words - st.pc - inst_words

  # Get op enum and lookup compiled function
  if inst_type is SOP1: op_cls, ssrc0, sdst = SOP1Op, inst.ssrc0, inst.sdst
  elif inst_type is SOP2: op_cls, ssrc0, sdst = SOP2Op, inst.ssrc0, inst.sdst
  elif inst_type is SOPC: op_cls, ssrc0, sdst = SOPCOp, inst.ssrc0, None
  elif inst_type is SOPK: op_cls, ssrc0, sdst = SOPKOp, inst.sdst, inst.sdst  # sdst is both src and dst
  else: raise NotImplementedError(f"Unknown scalar type {inst_type}")

  op = op_cls(inst.op)
  fn = compiled.get(op_cls, {}).get(op)
  if fn is None: raise NotImplementedError(f"{op.name} not in pseudocode")

  # Build context - handle 64-bit ops that need 64-bit source reads
  # 64-bit source ops: name ends with _B64, _I64, _U64 or contains _U64, _I64 before last underscore
  is_64bit_s0 = op.name.endswith(('_B64', '_I64', '_U64')) or '_U64_' in op.name or '_I64_' in op.name
  is_64bit_s0s1 = op_cls is SOPCOp and op in (SOPCOp.S_CMP_EQ_U64, SOPCOp.S_CMP_LG_U64)
  s0 = st.rsrc64(ssrc0, 0) if is_64bit_s0 or is_64bit_s0s1 else (st.rsrc(ssrc0, 0) if inst_type != SOPK else st.rsgpr(inst.sdst))
  is_64bit_sop2 = is_64bit_s0 and inst_type is SOP2
  s1 = st.rsrc64(inst.ssrc1, 0) if (is_64bit_sop2 or is_64bit_s0s1) else (st.rsrc(inst.ssrc1, 0) if inst_type in (SOP2, SOPC) else inst.simm16 if inst_type is SOPK else 0)
  d0 = st.rsgpr64(sdst) if (is_64bit_s0 or is_64bit_s0s1) and sdst is not None else (st.rsgpr(sdst) if sdst is not None else 0)
  literal = inst.simm16 if inst_type is SOPK else st.literal

  # Create Reg objects for new calling convention
  S0, S1, S2, D0 = Reg(s0), Reg(s1), Reg(0), Reg(d0)
  SCC, VCC, EXEC = Reg(st.scc), Reg(st.vcc), Reg(st.exec_mask)

  # Execute compiled function - fn(S0, S1, S2, D0, SCC, VCC, laneId, EXEC, SIMM16, VGPR, SRC0, VDST)
  fn(S0, S1, S2, D0, SCC, VCC, 0, EXEC, Reg(literal), None, 0, 0)

  # Apply results from Reg objects
  is_64bit_d0 = is_64bit_s0 or is_64bit_s0s1
  if sdst is not None:
    if is_64bit_d0:
      st.wsgpr64(sdst, D0._val)
    else:
      st.wsgpr(sdst, D0._val)
  st.scc = SCC._val
  st.exec_mask = EXEC._val
  return 0

def exec_vector(st: WaveState, inst: Inst, lane: int, lds: bytearray | None = None,
                d0_override: 'Reg | None' = None, vcc_override: 'Reg | None' = None) -> None:
  """Execute vector instruction for one lane.
  d0_override: For VOPC/VOP3-VOPC, use this Reg instead of st.sgpr[vdst] for D0 output.
  vcc_override: For VOP3SD, use this Reg instead of st.sgpr[sdst] for VCC output.
  """
  compiled = _get_compiled()
  inst_type, V = type(inst), st.vgpr[lane]

  # Memory ops (not ALU pseudocode)
  if inst_type is FLAT:
    op, addr_reg, data_reg, vdst, offset, saddr = inst.op, inst.addr, inst.data, inst.vdst, _sext(inst.offset, 13), inst.saddr
    addr = V[addr_reg]._val | (V[addr_reg+1]._val << 32)
    addr = (st.rsgpr64(saddr) + V[addr_reg]._val + offset) & 0xffffffffffffffff if saddr not in (NULL, 0x7f) else (addr + offset) & 0xffffffffffffffff
    if op in FLAT_LOAD:
      cnt, sz, sign = FLAT_LOAD[op]
      for i in range(cnt): val = mem_read(addr + i * sz, sz); V[vdst + i]._val = _sext(val, sz * 8) & 0xffffffff if sign else val
    elif op in FLAT_STORE:
      cnt, sz = FLAT_STORE[op]
      for i in range(cnt): mem_write(addr + i * sz, sz, V[data_reg + i]._val & ((1 << (sz * 8)) - 1))
    elif op in FLAT_D16_LOAD:
      sz, sign, hi = FLAT_D16_LOAD[op]
      val = mem_read(addr, sz)
      if sign: val = _sext(val, sz * 8) & 0xffff
      if hi: V[vdst]._val = (V[vdst]._val & 0xffff) | (val << 16)
      else: V[vdst]._val = (V[vdst]._val & 0xffff0000) | (val & 0xffff)
    elif op in FLAT_D16_STORE:
      sz, hi = FLAT_D16_STORE[op]
      val = (V[data_reg]._val >> 16) & 0xffff if hi else V[data_reg]._val & 0xffff
      mem_write(addr, sz, val & ((1 << (sz * 8)) - 1))
    else: raise NotImplementedError(f"FLAT op {op}")
    return

  if inst_type is DS:
    op, addr, vdst = inst.op, (V[inst.addr]._val + inst.offset0) & 0xffff, inst.vdst
    if op in DS_LOAD:
      cnt, sz, sign = DS_LOAD[op]
      for i in range(cnt): val = int.from_bytes(lds[addr+i*sz:addr+i*sz+sz], 'little'); V[vdst + i]._val = _sext(val, sz * 8) & 0xffffffff if sign else val
    elif op in DS_STORE:
      cnt, sz = DS_STORE[op]
      for i in range(cnt): lds[addr+i*sz:addr+i*sz+sz] = (V[inst.data0 + i]._val & ((1 << (sz * 8)) - 1)).to_bytes(sz, 'little')
    else: raise NotImplementedError(f"DS op {op}")
    return

  # VOPD: dual-issue, execute two ops using VOP2/VOP3 compiled functions
  if inst_type is VOPD:
    vdsty = (inst.vdsty << 1) | ((inst.vdstx & 1) ^ 1)
    # Read all source operands BEFORE any writes (dual-issue semantics)
    sx0, sx1 = Reg(st.rsrc(inst.srcx0, lane)), Reg(V[inst.vsrcx1]._val)
    sy0, sy1 = Reg(st.rsrc(inst.srcy0, lane)), Reg(V[inst.vsrcy1]._val)
    dx0, dy0 = Reg(V[inst.vdstx]._val), Reg(V[vdsty]._val)
    st._scc_reg._val = st.scc
    if (op_x := _VOPD_TO_VOP.get(inst.opx)):
      if (fn_x := compiled.get(type(op_x), {}).get(op_x)):
        fn_x(sx0, sx1, Reg(0), dx0, st._scc_reg, st.sgpr[VCC_LO], lane, st.sgpr[EXEC_LO], Reg(st.literal), None, Reg(0), Reg(inst.vdstx))
    if (op_y := _VOPD_TO_VOP.get(inst.opy)):
      if (fn_y := compiled.get(type(op_y), {}).get(op_y)):
        fn_y(sy0, sy1, Reg(0), dy0, st._scc_reg, st.sgpr[VCC_LO], lane, st.sgpr[EXEC_LO], Reg(st.literal), None, Reg(0), Reg(vdsty))
    V[inst.vdstx]._val, V[vdsty]._val = dx0._val, dy0._val
    st.scc = st._scc_reg._val
    return

  # Determine instruction format and get function
  is_vop3_vopc = False
  is_readlane = False
  if inst_type is VOP1:
    if inst.op == VOP1Op.V_NOP: return
    op_cls, op, src0, src1, src2, vdst = VOP1Op, VOP1Op(inst.op), inst.src0, None, None, inst.vdst
    # V_READFIRSTLANE_B32 writes to SGPR, not VGPR
    is_readlane = inst.op == VOP1Op.V_READFIRSTLANE_B32
  elif inst_type is VOP2:
    op_cls, op, src0, src1, src2, vdst = VOP2Op, VOP2Op(inst.op), inst.src0, inst.vsrc1 + 256, None, inst.vdst
  elif inst_type is VOP3:
    if inst.op < 256:
      # VOP3-encoded VOPC - destination is an SGPR (vdst field)
      op_cls, op, src0, src1, src2, vdst = VOPCOp, VOPCOp(inst.op), inst.src0, inst.src1, None, inst.vdst
      is_vop3_vopc = True
    else:
      op_cls, op, src0, src1, src2, vdst = VOP3Op, VOP3Op(inst.op), inst.src0, inst.src1, inst.src2, inst.vdst
      # V_READFIRSTLANE_B32 and V_READLANE_B32 write to SGPR
      is_readlane = inst.op in (VOP3Op.V_READFIRSTLANE_B32, VOP3Op.V_READLANE_B32)
  elif inst_type is VOP3SD:
    op_cls, op, src0, src1, src2, vdst = VOP3SDOp, VOP3SDOp(inst.op), inst.src0, inst.src1, inst.src2, inst.vdst
  elif inst_type is VOPC:
    op_cls, op, src0, src1, src2, vdst = VOPCOp, VOPCOp(inst.op), inst.src0, inst.vsrc1 + 256, None, VCC_LO
  elif inst_type is VOP3P:
    op_cls, op, src0, src1, src2, vdst = VOP3POp, VOP3POp(inst.op), inst.src0, inst.src1, inst.src2, inst.vdst
    # WMMA instructions are handled specially (only execute for lane 0)
    if op in (VOP3POp.V_WMMA_F32_16X16X16_F16, VOP3POp.V_WMMA_F16_16X16X16_F16):
      if lane == 0: exec_wmma(st, inst, op)
      return
  else: raise NotImplementedError(f"Unknown vector type {inst_type}")

  fn = compiled.get(op_cls, {}).get(op)
  if fn is None: raise NotImplementedError(f"{op.name} not in pseudocode")

  # Build source Regs - get the actual register or create temp for inline constants
  # VOP3P uses f16 inline constants (16-bit value in low half only)
  if inst_type is VOP3P:
    S0 = st.rsrc_reg_f16(src0, lane)
    S1 = st.rsrc_reg_f16(src1, lane) if src1 is not None else Reg(0)
    S2 = st.rsrc_reg_f16(src2, lane) if src2 is not None else Reg(0)
    # Apply op_sel_hi modifiers: control which half is used for hi-half computation
    # opsel_hi[0]=0 means src0 hi comes from lo half, =1 means from hi half (default)
    # opsel_hi[1]=0 means src1 hi comes from lo half, =1 means from hi half (default)
    # opsel_hi2=0 means src2 hi comes from lo half, =1 means from hi half (default)
    opsel_hi = getattr(inst, 'opsel_hi', 3)  # default 0b11
    opsel_hi2 = getattr(inst, 'opsel_hi2', 1)  # default 1
    # If opsel_hi bit is 0, replicate lo half to hi half
    if not (opsel_hi & 1):  # src0 hi from lo
      lo = S0._val & 0xffff
      S0 = Reg((lo << 16) | lo)
    if not (opsel_hi & 2):  # src1 hi from lo
      lo = S1._val & 0xffff
      S1 = Reg((lo << 16) | lo)
    if not opsel_hi2:  # src2 hi from lo
      lo = S2._val & 0xffff
      S2 = Reg((lo << 16) | lo)
  else:
    # Check if this is a 64-bit F64 op - needs 64-bit source reads for f64 operands
    # V_LDEXP_F64: S0 is f64, S1 is i32 (exponent)
    # V_ADD_F64, V_MUL_F64, etc: S0 and S1 are f64
    # VOP1 F64 ops (V_TRUNC_F64, V_FLOOR_F64, etc): S0 is f64
    is_f64_op = hasattr(op, 'name') and '_F64' in op.name
    is_ldexp_f64 = hasattr(op, 'name') and op.name == 'V_LDEXP_F64'
    if is_f64_op:
      S0 = st.rsrc_reg64(src0, lane)
      # V_LDEXP_F64: S1 is i32 exponent, not f64
      if is_ldexp_f64:
        S1 = st.rsrc_reg(src1, lane) if src1 is not None else Reg(0)
      else:
        S1 = st.rsrc_reg64(src1, lane) if src1 is not None else Reg(0)
      S2 = st.rsrc_reg64(src2, lane) if src2 is not None else Reg(0)
    else:
      S0 = st.rsrc_reg(src0, lane)
      S1 = st.rsrc_reg(src1, lane) if src1 is not None else Reg(0)
      S2 = st.rsrc_reg(src2, lane) if src2 is not None else Reg(0)
    # VOP3SD V_MAD_U64_U32 and V_MAD_I64_I32 need S2 as 64-bit from VGPR pair
    if inst_type is VOP3SD and op in (VOP3SDOp.V_MAD_U64_U32, VOP3SDOp.V_MAD_I64_I32) and src2 is not None:
      if 256 <= src2 <= 511:  # VGPR
        vgpr_idx = src2 - 256
        S2 = Reg(V[vgpr_idx]._val | (V[vgpr_idx + 1]._val << 32))

  # Apply source modifiers (neg, abs) for VOP3/VOP3SD
  if inst_type in (VOP3, VOP3SD):
    neg, abs_mod = getattr(inst, 'neg', 0), getattr(inst, 'abs', 0)
    if neg or abs_mod:
      # Apply to f32 values - need to handle as float
      import struct
      def apply_mods(reg, neg_bit, abs_bit):
        val = reg._val
        f = struct.unpack('<f', struct.pack('<I', val & 0xffffffff))[0]
        if abs_bit: f = abs(f)
        if neg_bit: f = -f
        return Reg(struct.unpack('<I', struct.pack('<f', f))[0])
      if neg & 1 or abs_mod & 1: S0 = apply_mods(S0, neg & 1, abs_mod & 1)
      if neg & 2 or abs_mod & 2: S1 = apply_mods(S1, neg & 2, abs_mod & 2)
      if neg & 4 or abs_mod & 4: S2 = apply_mods(S2, neg & 4, abs_mod & 4)

  # Apply opsel for VOP3 f16 operations - select which half to use
  # opsel[0]: src0, opsel[1]: src1, opsel[2]: src2 (0=lo, 1=hi)
  if inst_type is VOP3:
    opsel = getattr(inst, 'opsel', 0)
    if opsel:
      # If opsel bit is set, swap lo and hi so that .f16 reads the hi half
      if opsel & 1:  # src0 from hi
        S0 = Reg(((S0._val >> 16) & 0xffff) | (S0._val << 16))
      if opsel & 2:  # src1 from hi
        S1 = Reg(((S1._val >> 16) & 0xffff) | (S1._val << 16))
      if opsel & 4:  # src2 from hi
        S2 = Reg(((S2._val >> 16) & 0xffff) | (S2._val << 16))

  # For VOPC and VOP3-encoded VOPC, D0 is an SGPR (VCC_LO for VOPC, vdst for VOP3 VOPC)
  # V_READFIRSTLANE_B32 and V_READLANE_B32 also write to SGPR
  # Use d0_override if provided (for batch execution with shared output register)
  is_vopc = inst_type is VOPC or (inst_type is VOP3 and is_vop3_vopc)
  if is_vopc:
    D0 = d0_override if d0_override is not None else st.sgpr[VCC_LO if inst_type is VOPC else vdst]
  elif is_readlane:
    D0 = st.sgpr[vdst]
  else:
    D0 = V[vdst]

  # Execute compiled function - D0 is modified in place
  st._scc_reg._val = st.scc
  # For VOP3SD, pass sdst register as VCC parameter (carry-out destination)
  # Use vcc_override if provided (for batch execution with shared output register)
  # For VOP3 V_CNDMASK_B32, src2 specifies the condition selector (not VCC)
  if inst_type is VOP3SD:
    vcc_reg = vcc_override if vcc_override is not None else st.sgpr[inst.sdst]
  elif inst_type is VOP3 and op == VOP3Op.V_CNDMASK_B32 and src2 is not None:
    vcc_reg = st.rsrc_reg(src2, lane)  # Use src2 as condition
  else:
    vcc_reg = st.sgpr[VCC_LO]
  # SRC0/VDST are VGPR indices (0-255), not hardware encoding (256-511)
  src0_idx = (src0 - 256) if src0 and src0 >= 256 else (src0 if src0 else 0)
  result = fn(S0, S1, S2, D0, st._scc_reg, vcc_reg, lane, st.sgpr[EXEC_LO], Reg(st.literal), st.vgpr, Reg(src0_idx), Reg(vdst))
  st.scc = st._scc_reg._val

  # Handle special results
  if result:
    if 'vgpr_write' in result:
      wr_lane, wr_idx, wr_val = result['vgpr_write']
      st.vgpr[wr_lane][wr_idx]._val = wr_val

  # 64-bit destination: write high 32 bits to next VGPR (determined from op name)
  is_64bit_dst = not is_vopc and not is_readlane and hasattr(op, 'name') and \
                 any(s in op.name for s in ('_B64', '_I64', '_U64', '_F64'))
  if is_64bit_dst:
    V[vdst + 1]._val = (D0._val >> 32) & 0xffffffff
    D0._val = D0._val & 0xffffffff  # Keep only low 32 bits in D0

# ═══════════════════════════════════════════════════════════════════════════════
# WMMA (Wave Matrix Multiply-Accumulate)
# ═══════════════════════════════════════════════════════════════════════════════

def exec_wmma(st: WaveState, inst, op: VOP3POp) -> None:
  """Execute WMMA instruction - 16x16x16 matrix multiply across the wave."""
  src0, src1, src2, vdst = inst.src0, inst.src1, inst.src2, inst.vdst
  # Read matrix A (16x16 f16/bf16) from lanes 0-15, VGPRs src0 to src0+7 (2 f16 per VGPR = 16 values per lane)
  # Layout: A[row][k] where row = lane (0-15), k comes from 8 VGPRs × 2 halves
  mat_a = []
  for lane in range(16):
    for reg in range(8):
      val = st.vgpr[lane][src0 - 256 + reg] if src0 >= 256 else st.rsgpr(src0 + reg)
      mat_a.append(_f16(val & 0xffff))
      mat_a.append(_f16((val >> 16) & 0xffff))
  # Read matrix B (16x16 f16/bf16) - same layout, B[col][k] where col comes from lane
  mat_b = []
  for lane in range(16):
    for reg in range(8):
      val = st.vgpr[lane][src1 - 256 + reg] if src1 >= 256 else st.rsgpr(src1 + reg)
      mat_b.append(_f16(val & 0xffff))
      mat_b.append(_f16((val >> 16) & 0xffff))

  # Read matrix C (16x16 f32) from lanes 0-31, VGPRs src2 to src2+7
  # Layout: element i is at lane (i % 32), VGPR (i // 32) + src2
  mat_c = []
  for i in range(256):
    lane, reg = i % 32, i // 32
    val = st.vgpr[lane][src2 - 256 + reg] if src2 >= 256 else st.rsgpr(src2 + reg)
    mat_c.append(_f32(val))

  # Compute D = A × B + C (16x16 matrix multiply)
  mat_d = [0.0] * 256
  for row in range(16):
    for col in range(16):
      acc = 0.0
      for k in range(16):
        a_val = mat_a[row * 16 + k]
        b_val = mat_b[col * 16 + k]
        acc += a_val * b_val
      mat_d[row * 16 + col] = acc + mat_c[row * 16 + col]

  # Write result matrix D back - same layout as C
  if op == VOP3POp.V_WMMA_F16_16X16X16_F16:
    # Output is f16, pack 2 values per VGPR
    for i in range(0, 256, 2):
      lane, reg = (i // 2) % 32, (i // 2) // 32
      lo = _i16(mat_d[i]) & 0xffff
      hi = _i16(mat_d[i + 1]) & 0xffff
      st.vgpr[lane][vdst + reg]._val = (hi << 16) | lo
  else:
    # Output is f32
    for i in range(256):
      lane, reg = i % 32, i // 32
      st.vgpr[lane][vdst + reg]._val = _i32(mat_d[i])

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

SCALAR_TYPES = {SOP1, SOP2, SOPC, SOPK, SOPP, SMEM}
VECTOR_TYPES = {VOP1, VOP2, VOP3, VOP3SD, VOPC, FLAT, DS, VOPD, VOP3P}

# Pre-cache compiled functions for fast lookup
_COMPILED_CACHE: dict | None = None
def _get_fn(op_cls, op):
  global _COMPILED_CACHE
  if _COMPILED_CACHE is None: _COMPILED_CACHE = _get_compiled()
  return _COMPILED_CACHE.get(op_cls, {}).get(op)

def exec_vector_batch(st: WaveState, inst: Inst, exec_mask: int, n_lanes: int, lds: bytearray | None = None) -> None:
  """Execute vector instruction for all active lanes at once."""
  compiled = _get_compiled()
  inst_type = type(inst)
  vgpr = st.vgpr

  # Memory ops - still per-lane but inlined
  if inst_type is FLAT:
    op, addr_reg, data_reg, vdst, offset, saddr = inst.op, inst.addr, inst.data, inst.vdst, _sext(inst.offset, 13), inst.saddr
    if op in FLAT_LOAD:
      cnt, sz, sign = FLAT_LOAD[op]
      for lane in range(n_lanes):
        if not (exec_mask & (1 << lane)): continue
        V = vgpr[lane]
        addr = V[addr_reg]._val | (V[addr_reg+1]._val << 32)
        addr = (st.rsgpr64(saddr) + V[addr_reg]._val + offset) & 0xffffffffffffffff if saddr not in (NULL, 0x7f) else (addr + offset) & 0xffffffffffffffff
        for i in range(cnt): val = mem_read(addr + i * sz, sz); V[vdst + i]._val = _sext(val, sz * 8) & 0xffffffff if sign else val
    elif op in FLAT_STORE:
      cnt, sz = FLAT_STORE[op]
      for lane in range(n_lanes):
        if not (exec_mask & (1 << lane)): continue
        V = vgpr[lane]
        addr = V[addr_reg]._val | (V[addr_reg+1]._val << 32)
        addr = (st.rsgpr64(saddr) + V[addr_reg]._val + offset) & 0xffffffffffffffff if saddr not in (NULL, 0x7f) else (addr + offset) & 0xffffffffffffffff
        for i in range(cnt): mem_write(addr + i * sz, sz, V[data_reg + i]._val & ((1 << (sz * 8)) - 1))
    elif op in FLAT_D16_LOAD:
      sz, sign, hi = FLAT_D16_LOAD[op]
      for lane in range(n_lanes):
        if not (exec_mask & (1 << lane)): continue
        V = vgpr[lane]
        addr = V[addr_reg]._val | (V[addr_reg+1]._val << 32)
        addr = (st.rsgpr64(saddr) + V[addr_reg]._val + offset) & 0xffffffffffffffff if saddr not in (NULL, 0x7f) else (addr + offset) & 0xffffffffffffffff
        val = mem_read(addr, sz)
        if sign: val = _sext(val, sz * 8) & 0xffff
        if hi: V[vdst]._val = (V[vdst]._val & 0xffff) | (val << 16)
        else: V[vdst]._val = (V[vdst]._val & 0xffff0000) | (val & 0xffff)
    elif op in FLAT_D16_STORE:
      sz, hi = FLAT_D16_STORE[op]
      for lane in range(n_lanes):
        if not (exec_mask & (1 << lane)): continue
        V = vgpr[lane]
        addr = V[addr_reg]._val | (V[addr_reg+1]._val << 32)
        addr = (st.rsgpr64(saddr) + V[addr_reg]._val + offset) & 0xffffffffffffffff if saddr not in (NULL, 0x7f) else (addr + offset) & 0xffffffffffffffff
        val = (V[data_reg]._val >> 16) & 0xffff if hi else V[data_reg]._val & 0xffff
        mem_write(addr, sz, val & ((1 << (sz * 8)) - 1))
    else: raise NotImplementedError(f"FLAT op {op}")
    return

  if inst_type is DS:
    op, vdst = inst.op, inst.vdst
    if op in DS_LOAD:
      cnt, sz, sign = DS_LOAD[op]
      for lane in range(n_lanes):
        if not (exec_mask & (1 << lane)): continue
        V = vgpr[lane]
        addr = (V[inst.addr]._val + inst.offset0) & 0xffff
        for i in range(cnt): val = int.from_bytes(lds[addr+i*sz:addr+i*sz+sz], 'little'); V[vdst + i]._val = _sext(val, sz * 8) & 0xffffffff if sign else val
    elif op in DS_STORE:
      cnt, sz = DS_STORE[op]
      for lane in range(n_lanes):
        if not (exec_mask & (1 << lane)): continue
        V = vgpr[lane]
        addr = (V[inst.addr]._val + inst.offset0) & 0xffff
        for i in range(cnt): lds[addr+i*sz:addr+i*sz+sz] = (V[inst.data0 + i]._val & ((1 << (sz * 8)) - 1)).to_bytes(sz, 'little')
    else: raise NotImplementedError(f"DS op {op}")
    return

  # For VOPC, VOP3-encoded VOPC, and VOP3SD, we write per-lane bits to an SGPR.
  # The pseudocode does D0.u64[laneId] = bit or VCC.u64[laneId] = bit.
  # To avoid corrupting reads from the same SGPR, use a shared output Reg(0).
  # Exception: CMPX instructions write to EXEC (not D0/VCC).
  d0_override, vcc_override = None, None
  vopc_dst, vop3sd_dst = None, None
  is_cmpx = False
  if inst_type is VOPC:
    op = VOPCOp(inst.op)
    is_cmpx = 'CMPX' in op.name
    if not is_cmpx:  # Regular CMP writes to VCC
      d0_override, vopc_dst = Reg(0), VCC_LO
    else:  # CMPX writes to EXEC - clear it first, accumulate per-lane
      st.sgpr[EXEC_LO]._val = 0
  elif inst_type is VOP3 and inst.op < 256:  # VOP3-encoded VOPC
    op = VOPCOp(inst.op)
    is_cmpx = 'CMPX' in op.name
    if not is_cmpx:  # Regular CMP writes to destination SGPR
      d0_override, vopc_dst = Reg(0), inst.vdst
    else:  # CMPX writes to EXEC - clear it first, accumulate per-lane
      st.sgpr[EXEC_LO]._val = 0
  if inst_type is VOP3SD:
    vcc_override, vop3sd_dst = Reg(0), inst.sdst

  # For other vector ops, dispatch to exec_vector per lane (can optimize later)
  for lane in range(n_lanes):
    if exec_mask & (1 << lane): exec_vector(st, inst, lane, lds, d0_override, vcc_override)

  # Write accumulated per-lane bit results to destination SGPRs
  # (CMPX writes directly to EXEC in the pseudocode, so no separate write needed)
  if vopc_dst is not None: st.sgpr[vopc_dst]._val = d0_override._val
  if vop3sd_dst is not None: st.sgpr[vop3sd_dst]._val = vcc_override._val

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
    # V_READFIRSTLANE_B32 and V_READLANE_B32 write to SGPR, so they should only execute once per wave (lane 0)
    is_readlane = (inst_type is VOP1 and inst.op == VOP1Op.V_READFIRSTLANE_B32) or \
                  (inst_type is VOP3 and inst.op in (VOP3Op.V_READFIRSTLANE_B32, VOP3Op.V_READLANE_B32))
    if is_readlane:
      exec_vector(st, inst, 0, lds)  # Execute once with lane 0
    else:
      exec_vector_batch(st, inst, st.exec_mask, n_lanes, lds)
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
    gx, gy, gz = workgroup_id
    # Set workgroup IDs in SGPRs based on USER_SGPR_COUNT and enable flags from COMPUTE_PGM_RSRC2
    sgpr_idx = wg_id_sgpr_base
    if wg_id_enables[0]: st.sgpr[sgpr_idx]._val = gx; sgpr_idx += 1
    if wg_id_enables[1]: st.sgpr[sgpr_idx]._val = gy; sgpr_idx += 1
    if wg_id_enables[2]: st.sgpr[sgpr_idx]._val = gz
    for i in range(n_lanes):
      tid = wave_start + i
      st.vgpr[i][0]._val = tid if local_size == (lx, 1, 1) else ((tid // (lx * ly)) << 20) | (((tid // lx) % ly) << 10) | (tid % lx)
    waves.append((st, n_lanes, wave_start))
  has_barrier = any(isinstance(inst, SOPP) and inst.op == SOPPOp.S_BARRIER for inst in program.values())
  for _ in range(2 if has_barrier else 1):
    for st, n_lanes, _ in waves: exec_wave(program, st, lds, n_lanes)

def run_asm(lib: int, lib_sz: int, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, args_ptr: int, rsrc2: int = 0x19c) -> int:
  data = (ctypes.c_char * lib_sz).from_address(lib).raw
  program = decode_program(data)
  if not program: return -1
  # Parse COMPUTE_PGM_RSRC2 for SGPR layout
  user_sgpr_count = (rsrc2 >> 1) & 0x1f
  enable_wg_id_x = bool((rsrc2 >> 7) & 1)
  enable_wg_id_y = bool((rsrc2 >> 8) & 1)
  enable_wg_id_z = bool((rsrc2 >> 9) & 1)
  wg_id_enables = (enable_wg_id_x, enable_wg_id_y, enable_wg_id_z)
  for gidz in range(gz):
    for gidy in range(gy):
      for gidx in range(gx): exec_workgroup(program, (gidx, gidy, gidz), (lx, ly, lz), args_ptr, user_sgpr_count, wg_id_enables)
  return 0
