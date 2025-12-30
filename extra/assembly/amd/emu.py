# RDNA3 emulator - executes compiled pseudocode from AMD ISA PDF
# mypy: ignore-errors
from __future__ import annotations
import ctypes, os
from extra.assembly.amd.dsl import Inst, RawImm
from extra.assembly.amd.asm import detect_format
from extra.assembly.amd.pcode import _f32, _i32, _sext, _f16, _i16, _f64, _i64
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
_VOPC_16BIT_OPS = {op for op in VOPCOp if any(s in op.name for s in ('_F16', '_B16', '_I16', '_U16'))}
# CVT ops with 32/64-bit source (despite 16-bit in name)
_CVT_32_64_SRC_OPS = {op for op in VOP3Op if op.name.startswith('V_CVT_') and op.name.endswith(('_F32', '_I32', '_U32', '_F64', '_I64', '_U64'))} | \
                     {op for op in VOP1Op if op.name.startswith('V_CVT_') and op.name.endswith(('_F32', '_I32', '_U32', '_F64', '_I64', '_U64'))}
# CVT ops with 32-bit destination (convert FROM 16-bit TO 32-bit): V_CVT_F32_F16, V_CVT_I32_I16, V_CVT_U32_U16
_CVT_32_DST_OPS = {op for op in VOP3Op if op.name.startswith('V_CVT_') and any(s in op.name for s in ('F32_F16', 'I32_I16', 'U32_U16', 'I32_F16', 'U32_F16'))} | \
                  {op for op in VOP1Op if op.name.startswith('V_CVT_') and any(s in op.name for s in ('F32_F16', 'I32_I16', 'U32_U16', 'I32_F16', 'U32_F16'))}
# 16-bit dst ops (PACK has 32-bit dst despite F16 in name, CVT to 32-bit has 32-bit dst)
_VOP3_16BIT_DST_OPS = {op for op in _VOP3_16BIT_OPS if 'PACK' not in op.name} - _CVT_32_DST_OPS
_VOP1_16BIT_DST_OPS = {op for op in _VOP1_16BIT_OPS if 'PACK' not in op.name} - _CVT_32_DST_OPS
# VOP1 16-bit source ops (excluding CVT ops with 32/64-bit source) - for VOP1 e32, .h encoded in register index
_VOP1_16BIT_SRC_OPS = _VOP1_16BIT_OPS - _CVT_32_64_SRC_OPS

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
  __slots__ = ('sgpr', 'vgpr', 'scc', 'pc', 'literal', '_pend_sgpr')
  def __init__(self):
    self.sgpr, self.vgpr = [0] * SGPR_COUNT, [[0] * VGPR_COUNT for _ in range(WAVE_SIZE)]
    self.sgpr[EXEC_LO], self.scc, self.pc, self.literal, self._pend_sgpr = 0xffffffff, 0, 0, 0, {}

  @property
  def vcc(self) -> int: return self.sgpr[VCC_LO] | (self.sgpr[VCC_HI] << 32)
  @vcc.setter
  def vcc(self, v: int): self.sgpr[VCC_LO], self.sgpr[VCC_HI] = v & 0xffffffff, (v >> 32) & 0xffffffff
  @property
  def exec_mask(self) -> int: return self.sgpr[EXEC_LO] | (self.sgpr[EXEC_HI] << 32)
  @exec_mask.setter
  def exec_mask(self, v: int): self.sgpr[EXEC_LO], self.sgpr[EXEC_HI] = v & 0xffffffff, (v >> 32) & 0xffffffff

  def rsgpr(self, i: int) -> int: return 0 if i == NULL else self.scc if i == SCC else self.sgpr[i] if i < SGPR_COUNT else 0
  def wsgpr(self, i: int, v: int):
    if i < SGPR_COUNT and i != NULL: self.sgpr[i] = v & 0xffffffff
  def rsgpr64(self, i: int) -> int: return self.rsgpr(i) | (self.rsgpr(i+1) << 32)
  def wsgpr64(self, i: int, v: int): self.wsgpr(i, v & 0xffffffff); self.wsgpr(i+1, (v >> 32) & 0xffffffff)

  def rsrc(self, v: int, lane: int) -> int:
    if v < SGPR_COUNT: return self.sgpr[v]
    if v == SCC: return self.scc
    if v < 255: return _INLINE_CONSTS[v - 128]
    if v == 255: return self.literal
    return self.vgpr[lane][v - 256] if v <= 511 else 0

  def rsrc_f16(self, v: int, lane: int) -> int:
    """Read source operand for VOP3P packed f16 operations. Uses f16 inline constants."""
    if v < SGPR_COUNT: return self.sgpr[v]
    if v == SCC: return self.scc
    if v < 255: return _INLINE_CONSTS_F16[v - 128]
    if v == 255: return self.literal
    return self.vgpr[lane][v - 256] if v <= 511 else 0

  def rsrc64(self, v: int, lane: int) -> int:
    """Read 64-bit source operand. For inline constants, returns 64-bit representation."""
    # Inline constants 128-254 need special handling for 64-bit ops
    if 128 <= v < 255: return _INLINE_CONSTS_F64[v - 128]
    if v == 255: return self.literal  # 32-bit literal, caller handles extension
    return self.rsrc(v, lane) | ((self.rsrc(v+1, lane) if v < VCC_LO or 256 <= v <= 511 else 0) << 32)

  def pend_sgpr_lane(self, reg: int, lane: int, val: int):
    if reg not in self._pend_sgpr: self._pend_sgpr[reg] = 0
    if val: self._pend_sgpr[reg] |= (1 << lane)
  def commit_pends(self):
    for reg, val in self._pend_sgpr.items(): self.sgpr[reg] = val
    self._pend_sgpr.clear()



def _unwrap(v) -> int: return v.val if isinstance(v, RawImm) else v.value if hasattr(v, 'value') else v

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

  # SOPP: special cases for control flow that has no pseudocode
  if inst_type is SOPP:
    op = inst.op
    if op == SOPPOp.S_ENDPGM: return -1
    if op == SOPPOp.S_BARRIER: return -2

  # SMEM: memory loads (not ALU)
  if inst_type is SMEM:
    addr = st.rsgpr64(inst.sbase * 2) + _sext(inst.offset, 21)
    if inst.soffset not in (NULL, 0x7f): addr += st.rsrc(inst.soffset, 0)
    if (cnt := SMEM_LOAD.get(inst.op)) is None: raise NotImplementedError(f"SMEM op {inst.op}")
    for i in range(cnt): st.wsgpr(inst.sdata + i, mem_read((addr + i * 4) & 0xffffffffffffffff, 4))
    return 0

  # Get op enum and lookup compiled function
  if inst_type is SOP1: op_cls, ssrc0, sdst = SOP1Op, inst.ssrc0, inst.sdst
  elif inst_type is SOP2: op_cls, ssrc0, sdst = SOP2Op, inst.ssrc0, inst.sdst
  elif inst_type is SOPC: op_cls, ssrc0, sdst = SOPCOp, inst.ssrc0, None
  elif inst_type is SOPK: op_cls, ssrc0, sdst = SOPKOp, inst.sdst, inst.sdst  # sdst is both src and dst
  elif inst_type is SOPP: op_cls, ssrc0, sdst = SOPPOp, None, None
  else: raise NotImplementedError(f"Unknown scalar type {inst_type}")

  # SOPP has gaps in the opcode enum - treat unknown opcodes as no-ops
  try: op = op_cls(inst.op)
  except ValueError:
    if inst_type is SOPP: return 0
    raise
  fn = compiled.get(op_cls, {}).get(op)
  if fn is None:
    # SOPP instructions without pseudocode (waits, hints, nops) are no-ops
    if inst_type is SOPP: return 0
    raise NotImplementedError(f"{op.name} not in pseudocode")

  # Build context - handle 64-bit ops that need 64-bit source reads
  # 64-bit source ops: name ends with _B64, _I64, _U64 or contains _U64, _I64 before last underscore
  is_64bit_s0 = op.name.endswith(('_B64', '_I64', '_U64')) or '_U64_' in op.name or '_I64_' in op.name
  is_64bit_s0s1 = op_cls is SOPCOp and op in (SOPCOp.S_CMP_EQ_U64, SOPCOp.S_CMP_LG_U64)
  s0 = st.rsrc64(ssrc0, 0) if is_64bit_s0 or is_64bit_s0s1 else (st.rsrc(ssrc0, 0) if inst_type not in (SOPK, SOPP) else (st.rsgpr(inst.sdst) if inst_type is SOPK else 0))
  is_64bit_sop2 = is_64bit_s0 and inst_type is SOP2
  s1 = st.rsrc64(inst.ssrc1, 0) if (is_64bit_sop2 or is_64bit_s0s1) else (st.rsrc(inst.ssrc1, 0) if inst_type in (SOP2, SOPC) else inst.simm16 if inst_type is SOPK else 0)
  d0 = st.rsgpr64(sdst) if (is_64bit_s0 or is_64bit_s0s1) and sdst is not None else (st.rsgpr(sdst) if sdst is not None else 0)
  exec_mask = st.exec_mask
  literal = inst.simm16 if inst_type in (SOPK, SOPP) else st.literal

  # Execute compiled function - pass PC in bytes for instructions that need it
  pc_bytes = st.pc * 4
  result = fn(s0, s1, 0, d0, st.scc, st.vcc, 0, exec_mask, literal, None, {}, pc=pc_bytes)

  # Apply results
  if sdst is not None:
    if result.get('d0_64'):
      st.wsgpr64(sdst, result['d0'])
    else:
      st.wsgpr(sdst, result['d0'])
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
    addr = (st.rsgpr64(saddr) + V[addr_reg] + offset) & 0xffffffffffffffff if saddr not in (NULL, 0x7f) else (addr + offset) & 0xffffffffffffffff
    if op in FLAT_LOAD:
      cnt, sz, sign = FLAT_LOAD[op]
      for i in range(cnt): val = mem_read(addr + i * sz, sz); V[vdst + i] = _sext(val, sz * 8) & 0xffffffff if sign else val
    elif op in FLAT_STORE:
      cnt, sz = FLAT_STORE[op]
      for i in range(cnt): mem_write(addr + i * sz, sz, V[data_reg + i] & ((1 << (sz * 8)) - 1))
    elif op in FLAT_D16_LOAD:
      sz, sign, hi = FLAT_D16_LOAD[op]
      val = mem_read(addr, sz)
      if sign: val = _sext(val, sz * 8) & 0xffff
      if hi: V[vdst] = (V[vdst] & 0xffff) | (val << 16)  # upper 16 bits
      else: V[vdst] = (V[vdst] & 0xffff0000) | (val & 0xffff)  # lower 16 bits
    elif op in FLAT_D16_STORE:
      sz, hi = FLAT_D16_STORE[op]
      val = (V[data_reg] >> 16) & 0xffff if hi else V[data_reg] & 0xffff
      mem_write(addr, sz, val & ((1 << (sz * 8)) - 1))
    else: raise NotImplementedError(f"FLAT op {op}")
    return

  if inst_type is DS:
    op, addr, vdst = inst.op, (V[inst.addr] + inst.offset0) & 0xffff, inst.vdst
    if op in DS_LOAD:
      cnt, sz, sign = DS_LOAD[op]
      for i in range(cnt): val = int.from_bytes(lds[addr+i*sz:addr+i*sz+sz], 'little'); V[vdst + i] = _sext(val, sz * 8) & 0xffffffff if sign else val
    elif op in DS_STORE:
      cnt, sz = DS_STORE[op]
      for i in range(cnt): lds[addr+i*sz:addr+i*sz+sz] = (V[inst.data0 + i] & ((1 << (sz * 8)) - 1)).to_bytes(sz, 'little')
    else: raise NotImplementedError(f"DS op {op}")
    return

  # VOPD: dual-issue, execute two ops using VOP2/VOP3 compiled functions
  # Both ops execute simultaneously using pre-instruction values, so read all inputs first
  if inst_type is VOPD:
    vdsty = (inst.vdsty << 1) | ((inst.vdstx & 1) ^ 1)
    # Read all source operands BEFORE any writes (dual-issue semantics)
    sx0, sx1 = st.rsrc(inst.srcx0, lane), V[inst.vsrcx1]
    sy0, sy1 = st.rsrc(inst.srcy0, lane), V[inst.vsrcy1]
    dx0, dy0 = V[inst.vdstx], V[vdsty]
    # Execute X op
    res_x = None
    if (op_x := _VOPD_TO_VOP.get(inst.opx)):
      if (fn_x := compiled.get(type(op_x), {}).get(op_x)):
        res_x = fn_x(sx0, sx1, 0, dx0, st.scc, st.vcc, lane, st.exec_mask, st.literal, None, {})
    # Execute Y op
    res_y = None
    if (op_y := _VOPD_TO_VOP.get(inst.opy)):
      if (fn_y := compiled.get(type(op_y), {}).get(op_y)):
        res_y = fn_y(sy0, sy1, 0, dy0, st.scc, st.vcc, lane, st.exec_mask, st.literal, None, {})
    # Write results after both ops complete
    if res_x is not None: V[inst.vdstx] = res_x['d0']
    if res_y is not None: V[vdsty] = res_y['d0']
    return

  # VOP3SD: has extra scalar dest for carry output
  if inst_type is VOP3SD:
    op = VOP3SDOp(inst.op)
    fn = compiled.get(VOP3SDOp, {}).get(op)
    if fn is None: raise NotImplementedError(f"{op.name} not in pseudocode")
    s0, s1, s2 = st.rsrc(inst.src0, lane), st.rsrc(inst.src1, lane), st.rsrc(inst.src2, lane)
    # For 64-bit src2 ops (V_MAD_U64_U32, V_MAD_I64_I32), read from consecutive registers
    mad64_ops = (VOP3SDOp.V_MAD_U64_U32, VOP3SDOp.V_MAD_I64_I32)
    if op in mad64_ops:
      if inst.src2 >= 256:  # VGPR
        s2 = V[inst.src2 - 256] | (V[inst.src2 - 256 + 1] << 32)
      else:  # SGPR - read 64-bit from consecutive SGPRs
        s2 = st.rsgpr64(inst.src2)
    d0 = V[inst.vdst]
    # For carry-in operations (V_*_CO_CI_*), src2 register contains the carry bitmask (not VCC).
    # The pseudocode uses VCC but in VOP3SD encoding, the actual carry source is inst.src2.
    # We pass the src2 register value as 'vcc' to the interpreter so it reads the correct carry.
    carry_ops = (VOP3SDOp.V_ADD_CO_CI_U32, VOP3SDOp.V_SUB_CO_CI_U32, VOP3SDOp.V_SUBREV_CO_CI_U32)
    vcc_for_exec = st.rsgpr64(inst.src2) if op in carry_ops else st.vcc
    result = fn(s0, s1, s2, d0, st.scc, vcc_for_exec, lane, st.exec_mask, st.literal, None, {})
    # Write result - handle 64-bit destinations
    if result.get('d0_64'):
      V[inst.vdst] = result['d0'] & 0xffffffff
      V[inst.vdst + 1] = (result['d0'] >> 32) & 0xffffffff
    else:
      V[inst.vdst] = result['d0'] & 0xffffffff
    if result.get('vcc_lane') is not None:
      st.pend_sgpr_lane(inst.sdst, lane, result['vcc_lane'])
    return



  # Get op enum and sources (None means "no source" for that operand)
  # vop1_dst_hi/vop2_dst_hi: for VOP1/VOP2 16-bit dst ops, bit 7 of vdst indicates .h (high 16-bit) destination
  vop1_dst_hi, vop2_dst_hi = False, False
  if inst_type is VOP1:
    if inst.op == VOP1Op.V_NOP: return
    op_cls, op, src0, src1, src2 = VOP1Op, VOP1Op(inst.op), inst.src0, None, None
    # For 16-bit dst ops, vdst encodes .h in bit 7
    if op in _VOP1_16BIT_DST_OPS:
      vop1_dst_hi = (inst.vdst & 0x80) != 0
      vdst = inst.vdst & 0x7f
    else:
      vdst = inst.vdst
  elif inst_type is VOP2:
    op_cls, op, src0, src1, src2 = VOP2Op, VOP2Op(inst.op), inst.src0, inst.vsrc1 + 256, None
    # For 16-bit dst ops, vdst encodes .h in bit 7
    if op in _VOP2_16BIT_OPS:
      vop2_dst_hi = (inst.vdst & 0x80) != 0
      vdst = inst.vdst & 0x7f
    else:
      vdst = inst.vdst
  elif inst_type is VOP3:
    # VOP3 ops 0-255 are VOPC comparisons encoded as VOP3 (use VOPCOp pseudocode)
    if inst.op < 256:
      op_cls, op, src0, src1, src2, vdst = VOPCOp, VOPCOp(inst.op), inst.src0, inst.src1, None, inst.vdst
    else:
      op_cls, op, src0, src1, src2, vdst = VOP3Op, VOP3Op(inst.op), inst.src0, inst.src1, inst.src2, inst.vdst
  elif inst_type is VOPC:
    op = VOPCOp(inst.op)
    # For 16-bit VOPC, vsrc1 uses same encoding as VOP2 16-bit: bit 7 selects hi(1) or lo(0) half
    # vsrc1 field is 8 bits: [6:0] = VGPR index, [7] = hi flag
    src1 = inst.vsrc1 + 256  # convert to standard VGPR encoding (256 + vgpr_idx)
    op_cls, src0, src2, vdst = VOPCOp, inst.src0, None, VCC_LO
  elif inst_type is VOP3P:
    # VOP3P: Packed 16-bit operations using compiled functions
    op = VOP3POp(inst.op)
    # WMMA: wave-level matrix multiply-accumulate (special handling - needs cross-lane access)
    if op in (VOP3POp.V_WMMA_F32_16X16X16_F16, VOP3POp.V_WMMA_F32_16X16X16_BF16, VOP3POp.V_WMMA_F16_16X16X16_F16):
      if lane == 0:  # Only execute once per wave, write results for all lanes
        exec_wmma(st, inst, op)
      return
    # V_FMA_MIX: Mixed precision FMA - inputs can be f16 or f32 controlled by opsel_hi/opsel_hi2
    # opsel_hi[0]: src0 is f32 (0) or f16 from hi bits (1)
    # opsel_hi[1]: src1 is f32 (0) or f16 from hi bits (1)
    # opsel_hi2: src2 is f32 (0) or f16 from hi bits (1)
    # opsel[i]: when source is f16, use lo (0) or hi (1) 16 bits - BUT for V_FMA_MIX, opsel selects lo/hi when opsel_hi=1
    # neg_hi[i]: abs modifier for source i (reuses neg_hi field for abs in V_FMA_MIX)
    if op in (VOP3POp.V_FMA_MIX_F32, VOP3POp.V_FMA_MIXLO_F16, VOP3POp.V_FMA_MIXHI_F16):
      opsel = getattr(inst, 'opsel', 0)
      opsel_hi = getattr(inst, 'opsel_hi', 0)
      opsel_hi2 = getattr(inst, 'opsel_hi2', 0)
      neg = getattr(inst, 'neg', 0)
      abs_ = getattr(inst, 'neg_hi', 0)  # neg_hi field is reused as abs for V_FMA_MIX
      vdst = inst.vdst
      # Read raw 32-bit values
      s0_raw = st.rsrc(inst.src0, lane)
      s1_raw = st.rsrc(inst.src1, lane)
      s2_raw = st.rsrc(inst.src2, lane) if inst.src2 is not None else 0
      # Decode sources based on opsel_hi (controls f32 vs f16) and opsel (controls which half for f16)
      # src0: opsel_hi[0]=1 means f16, opsel[0] selects hi(1) or lo(0) half
      if opsel_hi & 1:
        s0 = _f16((s0_raw >> 16) & 0xffff) if (opsel & 1) else _f16(s0_raw & 0xffff)
      else:
        s0 = _f32(s0_raw)
      # src1: opsel_hi[1]=1 means f16, opsel[1] selects hi(1) or lo(0) half
      if opsel_hi & 2:
        s1 = _f16((s1_raw >> 16) & 0xffff) if (opsel & 2) else _f16(s1_raw & 0xffff)
      else:
        s1 = _f32(s1_raw)
      # src2: opsel_hi2=1 means f16, opsel[2] selects hi(1) or lo(0) half
      if opsel_hi2:
        s2 = _f16((s2_raw >> 16) & 0xffff) if (opsel & 4) else _f16(s2_raw & 0xffff)
      else:
        s2 = _f32(s2_raw)
      # Apply abs modifiers (abs_ field reuses neg_hi position)
      if abs_ & 1: s0 = abs(s0)
      if abs_ & 2: s1 = abs(s1)
      if abs_ & 4: s2 = abs(s2)
      # Apply neg modifiers
      if neg & 1: s0 = -s0
      if neg & 2: s1 = -s1
      if neg & 4: s2 = -s2
      # Compute FMA: d = s0 * s1 + s2
      result = s0 * s1 + s2
      V = st.vgpr[lane]
      if op == VOP3POp.V_FMA_MIX_F32:
        V[vdst] = _i32(result)
      elif op == VOP3POp.V_FMA_MIXLO_F16:
        lo = _i16(result) & 0xffff
        V[vdst] = (V[vdst] & 0xffff0000) | lo
      else:  # V_FMA_MIXHI_F16
        hi = _i16(result) & 0xffff
        V[vdst] = (V[vdst] & 0x0000ffff) | (hi << 16)
      return
    # Use rsrc_f16 for VOP3P to get correct f16 inline constants
    s0_raw = st.rsrc_f16(inst.src0, lane)
    s1_raw = st.rsrc_f16(inst.src1, lane)
    s2_raw = st.rsrc_f16(inst.src2, lane) if inst.src2 is not None else 0
    # Handle opsel (which 16-bit halves to use for each source)
    opsel = getattr(inst, 'opsel', 0)
    opsel_hi = getattr(inst, 'opsel_hi', 3)  # Default: use hi for hi result
    opsel_hi2 = getattr(inst, 'opsel_hi2', 1)  # Default for src2
    # Handle neg modifiers for VOP3P
    # neg applies to lo result inputs, neg_hi applies to hi result inputs
    neg = getattr(inst, 'neg', 0)
    neg_hi = getattr(inst, 'neg_hi', 0)
    # Build "virtual" sources with halves arranged for pseudocode: lo half goes to [15:0], hi half goes to [31:16]
    # opsel bit 0/1/2 selects which half of src0/1/2 goes to the LO result
    # opsel_hi bit 0/1 selects which half of src0/1 goes to the HI result
    s0_lo = (s0_raw >> 16) & 0xffff if (opsel & 1) else s0_raw & 0xffff
    s1_lo = (s1_raw >> 16) & 0xffff if (opsel & 2) else s1_raw & 0xffff
    s2_lo = (s2_raw >> 16) & 0xffff if (opsel & 4) else s2_raw & 0xffff
    s0_hi = (s0_raw >> 16) & 0xffff if (opsel_hi & 1) else s0_raw & 0xffff
    s1_hi = (s1_raw >> 16) & 0xffff if (opsel_hi & 2) else s1_raw & 0xffff
    s2_hi = (s2_raw >> 16) & 0xffff if opsel_hi2 else s2_raw & 0xffff
    # Apply neg to lo result inputs (toggle f16 sign bit)
    if neg & 1: s0_lo ^= 0x8000
    if neg & 2: s1_lo ^= 0x8000
    if neg & 4: s2_lo ^= 0x8000
    # Apply neg_hi to hi result inputs
    if neg_hi & 1: s0_hi ^= 0x8000
    if neg_hi & 2: s1_hi ^= 0x8000
    if neg_hi & 4: s2_hi ^= 0x8000
    # Pack into format expected by pseudocode: [31:16] = hi input, [15:0] = lo input
    s0 = (s0_hi << 16) | s0_lo
    s1 = (s1_hi << 16) | s1_lo
    s2 = (s2_hi << 16) | s2_lo
    op_cls, vdst = VOP3POp, inst.vdst
    fn = compiled.get(op_cls, {}).get(op)
    if fn is None: raise NotImplementedError(f"{op.name} not in pseudocode")
    result = fn(s0, s1, s2, 0, st.scc, st.vcc, lane, st.exec_mask, st.literal, None, {})
    st.vgpr[lane][vdst] = result['d0'] & 0xffffffff
    return
  else: raise NotImplementedError(f"Unknown vector type {inst_type}")

  fn = compiled.get(op_cls, {}).get(op)
  if fn is None: raise NotImplementedError(f"{op.name} not in pseudocode")

  # Read sources (with VOP3 modifiers if applicable)
  neg, abs_ = (getattr(inst, 'neg', 0), getattr(inst, 'abs', 0)) if inst_type is VOP3 else (0, 0)
  opsel = getattr(inst, 'opsel', 0) if inst_type is VOP3 else 0
  def mod_src(val: int, idx: int) -> int:
    if (abs_ >> idx) & 1: val = _i32(abs(_f32(val)))
    if (neg >> idx) & 1: val = _i32(-_f32(val))
    return val
  def mod_src64(val: int, idx: int) -> int:
    if (abs_ >> idx) & 1: val = _i64(abs(_f64(val)))
    if (neg >> idx) & 1: val = _i64(-_f64(val))
    return val

  # Determine if sources are 64-bit based on instruction type
  # For 64-bit shift ops: src0 is 32-bit (shift amount), src1 is 64-bit (value to shift)
  # For most other _B64/_I64/_U64/_F64 ops: all sources are 64-bit
  is_64bit_op = op.name.endswith(('_B64', '_I64', '_U64', '_F64'))
  # V_LDEXP_F64: src0 is 64-bit float, src1 is 32-bit integer exponent
  is_ldexp_64 = op in (VOP3Op.V_LDEXP_F64,)
  is_shift_64 = op in (VOP3Op.V_LSHLREV_B64, VOP3Op.V_LSHRREV_B64, VOP3Op.V_ASHRREV_I64)
  # 16-bit source ops: use precomputed sets instead of string checks
  # Note: must check op_cls to avoid cross-enum value collisions
  is_16bit_src = op_cls is VOP3Op and op in _VOP3_16BIT_OPS and op not in _CVT_32_64_SRC_OPS
  # VOP2 16-bit ops use f16 inline constants for src0 (vsrc1 is always a VGPR, no inline constants)
  is_vop2_16bit = op_cls is VOP2Op and op in _VOP2_16BIT_OPS

  if is_shift_64:
    s0 = mod_src(st.rsrc(src0, lane), 0)  # shift amount is 32-bit
    s1 = st.rsrc64(src1, lane) if src1 is not None else 0  # value to shift is 64-bit
    s2 = mod_src(st.rsrc(src2, lane), 2) if src2 is not None else 0
  elif is_ldexp_64:
    s0 = mod_src64(st.rsrc64(src0, lane), 0)  # mantissa is 64-bit float
    s1 = mod_src(st.rsrc(src1, lane), 1) if src1 is not None else 0  # exponent is 32-bit int
    s2 = mod_src(st.rsrc(src2, lane), 2) if src2 is not None else 0
  elif is_64bit_op:
    # 64-bit ops: apply neg/abs modifiers using f64 interpretation for float ops
    s0 = mod_src64(st.rsrc64(src0, lane), 0)
    s1 = mod_src64(st.rsrc64(src1, lane), 1) if src1 is not None else 0
    s2 = mod_src64(st.rsrc64(src2, lane), 2) if src2 is not None else 0
  elif is_16bit_src:
    # For 16-bit source ops, opsel bits select which half to use
    # Inline constants (128-254) must use f16 encoding, not f32
    def rsrc_16bit(src, lane): return st.rsrc_f16(src, lane) if 128 <= src < 255 else st.rsrc(src, lane)
    s0_raw = rsrc_16bit(src0, lane)
    s1_raw = rsrc_16bit(src1, lane) if src1 is not None else 0
    s2_raw = rsrc_16bit(src2, lane) if src2 is not None else 0
    # opsel[0] selects hi(1) or lo(0) for src0, opsel[1] for src1, opsel[2] for src2
    s0 = ((s0_raw >> 16) & 0xffff) if (opsel & 1) else (s0_raw & 0xffff)
    s1 = ((s1_raw >> 16) & 0xffff) if (opsel & 2) else (s1_raw & 0xffff)
    s2 = ((s2_raw >> 16) & 0xffff) if (opsel & 4) else (s2_raw & 0xffff)
    # Apply abs/neg modifiers as f16 operations (toggle sign bit 15)
    if abs_ & 1: s0 &= 0x7fff
    if abs_ & 2: s1 &= 0x7fff
    if abs_ & 4: s2 &= 0x7fff
    if neg & 1: s0 ^= 0x8000
    if neg & 2: s1 ^= 0x8000
    if neg & 4: s2 ^= 0x8000
  elif is_vop2_16bit:
    # VOP2 16-bit ops: src0 uses f16 inline constants, or VGPR where v128+ = hi half of v0-v127
    # RDNA3 encoding: for VGPRs, bit 7 of VGPR index (src0-256) selects hi(1) or lo(0) half
    if src0 >= 256:  # VGPR
      src0_hi = (src0 - 256) & 0x80 != 0
      src0_masked = ((src0 - 256) & 0x7f) + 256  # mask out hi bit to get actual VGPR
      s0_raw = mod_src(st.rsrc(src0_masked, lane), 0)
      s0 = ((s0_raw >> 16) & 0xffff) if src0_hi else (s0_raw & 0xffff)
    else:  # SGPR or inline constant
      s0_raw = mod_src(st.rsrc_f16(src0, lane), 0)
      s0 = s0_raw & 0xffff
    # vsrc1: .h suffix encoded in bit 7 of VGPR index (src1 = 256 + vgpr_idx + 0x80 if hi)
    if src1 is not None:
      src1_hi = (src1 - 256) & 0x80 != 0
      src1_masked = ((src1 - 256) & 0x7f) + 256
      s1_raw = mod_src(st.rsrc(src1_masked, lane), 1)
      s1 = ((s1_raw >> 16) & 0xffff) if src1_hi else (s1_raw & 0xffff)
    else:
      s1 = 0
    s2 = mod_src(st.rsrc(src2, lane), 2) if src2 is not None else 0
  elif op_cls is VOP1Op and op in _VOP1_16BIT_SRC_OPS:
    # VOP1 16-bit source ops: .h encoded in bit 7 of VGPR index (src0 >= 384 means hi half)
    # For VGPRs: src0 = 256 + vgpr_idx + (0x80 if hi else 0), so bit 7 of (src0-256) is the hi flag
    src0_hi = src0 >= 256 and ((src0 - 256) & 0x80) != 0
    src0_masked = ((src0 - 256) & 0x7f) + 256 if src0 >= 256 else src0  # mask out hi bit for VGPR
    s0_raw = mod_src(st.rsrc(src0_masked, lane), 0)
    s0 = ((s0_raw >> 16) & 0xffff) if src0_hi else (s0_raw & 0xffff)
    s1, s2 = 0, 0
  elif op_cls is VOPCOp and op in _VOPC_16BIT_OPS:
    # VOPC 16-bit ops: src0 and vsrc1 use same encoding as VOP2 16-bit
    # For VGPRs, bit 7 of VGPR index selects hi(1) or lo(0) half
    if src0 >= 256:  # VGPR
      src0_hi = (src0 - 256) & 0x80 != 0
      src0_masked = ((src0 - 256) & 0x7f) + 256
      s0_raw = mod_src(st.rsrc(src0_masked, lane), 0)
      s0 = ((s0_raw >> 16) & 0xffff) if src0_hi else (s0_raw & 0xffff)
    else:  # SGPR or inline constant
      s0_raw = mod_src(st.rsrc_f16(src0, lane), 0)
      s0 = s0_raw & 0xffff
    # vsrc1: bit 7 of VGPR index selects hi(1) or lo(0) half
    if src1 is not None:
      if src1 >= 256:  # VGPR - use hi/lo encoding
        src1_hi = (src1 - 256) & 0x80 != 0
        src1_masked = ((src1 - 256) & 0x7f) + 256
        s1_raw = mod_src(st.rsrc(src1_masked, lane), 1)
        s1 = ((s1_raw >> 16) & 0xffff) if src1_hi else (s1_raw & 0xffff)
      else:  # SGPR or inline constant - read as 32-bit, use low 16 bits
        s1_raw = mod_src(st.rsrc(src1, lane), 1)
        s1 = s1_raw & 0xffffffff  # V_CMP_CLASS uses full 32-bit mask
    else:
      s1 = 0
    s2 = 0
  else:
    s0 = mod_src(st.rsrc(src0, lane), 0)
    s1 = mod_src(st.rsrc(src1, lane), 1) if src1 is not None else 0
    s2 = mod_src(st.rsrc(src2, lane), 2) if src2 is not None else 0
  # For VOP2 16-bit ops (like V_FMAC_F16), the destination is used as an accumulator.
  # The pseudocode reads D0.f16 from low 16 bits, so we need to shift hi->lo when vop2_dst_hi is True.
  if is_vop2_16bit:
    d0 = ((V[vdst] >> 16) & 0xffff) if vop2_dst_hi else (V[vdst] & 0xffff)
  else:
    d0 = V[vdst] if not is_64bit_op else (V[vdst] | (V[vdst + 1] << 32))

  # V_CNDMASK_B32/B16: VOP3 encoding uses src2 as mask (not VCC); VOP2 uses VCC implicitly
  # Pass the correct mask as vcc to the function so pseudocode VCC.u64[laneId] works correctly
  vcc_for_fn = st.rsgpr64(src2) if op in (VOP3Op.V_CNDMASK_B32, VOP3Op.V_CNDMASK_B16) and inst_type is VOP3 and src2 is not None and src2 < 256 else st.vcc

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
    # VOP2 carry instructions (V_ADD_CO_CI_U32, V_SUB_CO_CI_U32, V_SUBREV_CO_CI_U32) write carry to VCC implicitly
    # VOPC and VOP3-encoded VOPC write to vdst (which is VCC_LO for VOPC, inst.sdst for VOP3)
    vcc_dst = VCC_LO if op_cls is VOP2Op and op in (VOP2Op.V_ADD_CO_CI_U32, VOP2Op.V_SUB_CO_CI_U32, VOP2Op.V_SUBREV_CO_CI_U32) else vdst
    st.pend_sgpr_lane(vcc_dst, lane, result['vcc_lane'])
  if 'exec_lane' in result:
    # V_CMPX instructions write to EXEC per-lane
    st.pend_sgpr_lane(EXEC_LO, lane, result['exec_lane'])
  if 'd0' in result and op_cls not in (VOPCOp,) and 'vgpr_write' not in result:
    # V_READFIRSTLANE_B32 and V_READLANE_B32 write to SGPR, not VGPR
    # V_WRITELANE_B32 uses vgpr_write for cross-lane writes, don't overwrite with d0
    writes_to_sgpr = op in (VOP1Op.V_READFIRSTLANE_B32,) or \
                     (op_cls is VOP3Op and op in (VOP3Op.V_READFIRSTLANE_B32, VOP3Op.V_READLANE_B32))
    # Check for 16-bit destination ops (opsel[3] controls hi/lo write)
    # Must check op_cls to avoid cross-enum value collisions (e.g., VOP1Op.V_MOV_B32=1 vs VOP3Op.V_CMP_LT_F16=1)
    is_16bit_dst = (op_cls is VOP3Op and op in _VOP3_16BIT_DST_OPS) or (op_cls is VOP1Op and op in _VOP1_16BIT_DST_OPS)
    if writes_to_sgpr:
      st.wsgpr(vdst, result['d0'] & 0xffffffff)
    elif result.get('d0_64') or is_64bit_op:
      V[vdst] = result['d0'] & 0xffffffff
      V[vdst + 1] = (result['d0'] >> 32) & 0xffffffff
    elif is_16bit_dst and inst_type is VOP3:
      # VOP3 16-bit ops: opsel[3] (bit 3 of opsel field) controls hi/lo destination
      if opsel & 8:  # opsel[3] = 1: write to high 16 bits
        V[vdst] = (V[vdst] & 0x0000ffff) | ((result['d0'] & 0xffff) << 16)
      else:  # opsel[3] = 0: write to low 16 bits
        V[vdst] = (V[vdst] & 0xffff0000) | (result['d0'] & 0xffff)
    elif is_16bit_dst and inst_type is VOP1:
      # VOP1 16-bit ops: .h suffix encoded in bit 7 of vdst (extracted as vop1_dst_hi)
      if vop1_dst_hi:  # .h: write to high 16 bits
        V[vdst] = (V[vdst] & 0x0000ffff) | ((result['d0'] & 0xffff) << 16)
      else:  # .l: write to low 16 bits
        V[vdst] = (V[vdst] & 0xffff0000) | (result['d0'] & 0xffff)
    elif is_vop2_16bit:
      # VOP2 16-bit ops: .h suffix encoded in bit 7 of vdst (extracted as vop2_dst_hi)
      if vop2_dst_hi:  # .h: write to high 16 bits
        V[vdst] = (V[vdst] & 0x0000ffff) | ((result['d0'] & 0xffff) << 16)
      else:  # .l: write to low 16 bits
        V[vdst] = (V[vdst] & 0xffff0000) | (result['d0'] & 0xffff)
    else:
      V[vdst] = result['d0'] & 0xffffffff

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
      st.vgpr[lane][vdst + reg] = (hi << 16) | lo
  else:
    # Output is f32
    for i in range(256):
      lane, reg = i % 32, i // 32
      st.vgpr[lane][vdst + reg] = _i32(mat_d[i])

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
    # V_READFIRSTLANE_B32 and V_READLANE_B32 write to SGPR, so they should only execute once per wave (lane 0)
    is_readlane = (inst_type is VOP1 and inst.op == VOP1Op.V_READFIRSTLANE_B32) or \
                  (inst_type is VOP3 and inst.op in (VOP3Op.V_READFIRSTLANE_B32, VOP3Op.V_READLANE_B32))
    if is_readlane:
      exec_vector(st, inst, 0, lds)  # Execute once with lane 0
    else:
      exec_mask = st.exec_mask
      for lane in range(n_lanes):
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
    gx, gy, gz = workgroup_id
    # Set workgroup IDs in SGPRs based on USER_SGPR_COUNT and enable flags from COMPUTE_PGM_RSRC2
    sgpr_idx = wg_id_sgpr_base
    if wg_id_enables[0]: st.sgpr[sgpr_idx] = gx; sgpr_idx += 1
    if wg_id_enables[1]: st.sgpr[sgpr_idx] = gy; sgpr_idx += 1
    if wg_id_enables[2]: st.sgpr[sgpr_idx] = gz
    for i in range(n_lanes):
      tid = wave_start + i
      st.vgpr[i][0] = tid if local_size == (lx, 1, 1) else ((tid // (lx * ly)) << 20) | (((tid // lx) % ly) << 10) | (tid % lx)
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
