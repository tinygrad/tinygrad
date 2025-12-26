# RDNA3 emulator v2 - executes pseudocode from AMD ISA PDF
from __future__ import annotations
import ctypes, struct, math
from extra.assembly.rdna3.lib import Inst, RawImm
from extra.assembly.rdna3.pseudocode import get_pseudocode, PseudocodeInterpreter, _f32, _i32, _sext
from extra.assembly.rdna3.autogen import (
  SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, VOP1, VOP2, VOP3, VOP3SD, VOP3P, VOPC, DS, FLAT, VOPD, SrcEnum,
  SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, SMEMOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp, DSOp, FLATOp, GLOBALOp, VOPDOp
)

Program = dict[int, Inst]
WAVE_SIZE, SGPR_COUNT, VGPR_COUNT = 32, 128, 256
VCC_LO, VCC_HI, NULL, EXEC_LO, EXEC_HI, SCC = SrcEnum.VCC_LO, SrcEnum.VCC_HI, SrcEnum.NULL, SrcEnum.EXEC_LO, SrcEnum.EXEC_HI, SrcEnum.SCC

# Inline constants for src operands 128-254
_INLINE_CONSTS = [0] * 127
for _i in range(65): _INLINE_CONSTS[_i] = _i
for _i in range(1, 17): _INLINE_CONSTS[64 + _i] = ((-_i) & 0xffffffff)
for _k, _v in {SrcEnum.POS_HALF: 0x3f000000, SrcEnum.NEG_HALF: 0xbf000000, SrcEnum.POS_ONE: 0x3f800000, SrcEnum.NEG_ONE: 0xbf800000,
               SrcEnum.POS_TWO: 0x40000000, SrcEnum.NEG_TWO: 0xc0000000, SrcEnum.POS_FOUR: 0x40800000, SrcEnum.NEG_FOUR: 0xc0800000,
               SrcEnum.INV_2PI: 0x3e22f983}.items(): _INLINE_CONSTS[_k - 128] = _v

# Memory access
_valid_mem_ranges: list[tuple[int, int]] = []
def set_valid_mem_ranges(ranges: set[tuple[int, int]]) -> None: global _valid_mem_ranges; _valid_mem_ranges = list(ranges)
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

# Pseudocode cache: {enum_cls: {op: [pseudocode_lines]}}
_PSEUDOCODE: dict | None = None
_INTERP: PseudocodeInterpreter | None = None
def _get_pseudocode() -> tuple[dict, PseudocodeInterpreter]:
  global _PSEUDOCODE, _INTERP
  if _PSEUDOCODE is None: _PSEUDOCODE, _INTERP = get_pseudocode(), PseudocodeInterpreter()
  return _PSEUDOCODE, _INTERP

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

  def rsrc64(self, v: int, lane: int) -> int:
    return self.rsrc(v, lane) | ((self.rsrc(v+1, lane) if v < VCC_LO or 256 <= v <= 511 else 0) << 32)

  def pend_sgpr_lane(self, reg: int, lane: int, val: int):
    if reg not in self._pend_sgpr: self._pend_sgpr[reg] = 0
    if val: self._pend_sgpr[reg] |= (1 << lane)
  def commit_pends(self):
    for reg, val in self._pend_sgpr.items(): self.sgpr[reg] = val
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
    inst = inst_class.from_bytes(data[i:i+base_size])
    for name, val in inst._values.items(): setattr(inst, name, _unwrap(val))
    has_literal = any(getattr(inst, fld, None) == 255 for fld in ('src0', 'src1', 'src2', 'ssrc0', 'ssrc1', 'srcx0', 'srcy0'))
    if inst_class == VOP2 and inst.op in (44, 45, 55, 56): has_literal = True
    if inst_class == VOPD and (inst.opx in (1, 2) or inst.opy in (1, 2)): has_literal = True
    if inst_class == SOP2 and inst.op in (69, 70): has_literal = True
    if has_literal: inst._literal = int.from_bytes(data[i+base_size:i+base_size+4], 'little')
    inst._words = inst.size() // 4
    result[i // 4] = inst
    i += inst._words * 4
  return result

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION - All ALU ops use pseudocode from PDF
# ═══════════════════════════════════════════════════════════════════════════════

def exec_scalar(st: WaveState, inst: Inst) -> int:
  """Execute scalar instruction. Returns PC delta or negative for special cases."""
  pc_dict, interp = _get_pseudocode()
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
    return 0  # waits, hints, nops

  # SMEM: memory loads (not ALU)
  if inst_type is SMEM:
    addr = st.rsgpr64(inst.sbase * 2) + _sext(inst.offset, 21)
    if inst.soffset not in (NULL, 0x7f): addr += st.rsrc(inst.soffset, 0)
    if (cnt := SMEM_LOAD.get(inst.op)) is None: raise NotImplementedError(f"SMEM op {inst.op}")
    for i in range(cnt): st.wsgpr(inst.sdata + i, mem_read((addr + i * 4) & 0xffffffffffffffff, 4))
    return 0

  # Get op enum and lookup pseudocode
  if inst_type is SOP1: op_cls, ssrc0, sdst = SOP1Op, inst.ssrc0, inst.sdst
  elif inst_type is SOP2: op_cls, ssrc0, sdst = SOP2Op, inst.ssrc0, inst.sdst
  elif inst_type is SOPC: op_cls, ssrc0, sdst = SOPCOp, inst.ssrc0, None
  elif inst_type is SOPK: op_cls, ssrc0, sdst = SOPKOp, inst.sdst, inst.sdst  # sdst is both src and dst
  else: raise NotImplementedError(f"Unknown scalar type {inst_type}")

  op = op_cls(inst.op)
  pc = pc_dict.get(op_cls, {}).get(op)
  if pc is None: raise NotImplementedError(f"{op.name} not in pseudocode")

  # Build interpreter context
  s0 = st.rsrc(ssrc0, 0) if inst_type != SOPK else st.rsgpr(inst.sdst)
  s1 = st.rsrc(inst.ssrc1, 0) if inst_type is SOP2 else inst.simm16 if inst_type is SOPK else 0
  d0 = st.rsgpr(sdst) if sdst is not None else 0
  exec_mask = st.exec_mask

  # Execute pseudocode
  result = interp.execute(pc, s0, s1, scc=st.scc, d0=d0, exec_mask=exec_mask)

  # Apply results
  if sdst is not None:
    if result.get('d0_64'):
      st.wsgpr64(sdst, result['d0'])
    else:
      st.wsgpr(sdst, result['d0'])
  if 'scc' in result: st.scc = result['scc']
  if 'exec' in result: st.exec_mask = result['exec']
  if 'pc_delta' in result: return result['pc_delta']
  return 0

def exec_vector(st: WaveState, inst: Inst, lane: int, lds: bytearray | None = None) -> None:
  """Execute vector instruction for one lane."""
  pc_dict, interp = _get_pseudocode()
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

  # VOPD: dual-issue, execute two ops using VOP2/VOP3 pseudocode
  if inst_type is VOPD:
    vdsty = (inst.vdsty << 1) | ((inst.vdstx & 1) ^ 1)
    # Execute X op
    if (op_x := _VOPD_TO_VOP.get(inst.opx)):
      s0, s1 = st.rsrc(inst.srcx0, lane), V[inst.vsrcx1]
      if op_x in (VOP3Op.V_MAX_F32, VOP2Op.V_MAX_F32): V[inst.vdstx] = _i32(max(_f32(s0), _f32(s1)))
      elif op_x in (VOP3Op.V_MIN_F32, VOP2Op.V_MIN_F32): V[inst.vdstx] = _i32(min(_f32(s0), _f32(s1)))
      elif (pc_x := pc_dict.get(type(op_x), {}).get(op_x)):
        res = interp.execute(pc_x, s0, s1, scc=st.scc, d0=V[inst.vdstx], vcc=st.vcc, lane=lane, literal=st.literal)
        V[inst.vdstx] = res['d0']
    # Execute Y op
    if (op_y := _VOPD_TO_VOP.get(inst.opy)):
      s0, s1 = st.rsrc(inst.srcy0, lane), V[inst.vsrcy1]
      if op_y in (VOP3Op.V_MAX_F32, VOP2Op.V_MAX_F32): V[vdsty] = _i32(max(_f32(s0), _f32(s1)))
      elif op_y in (VOP3Op.V_MIN_F32, VOP2Op.V_MIN_F32): V[vdsty] = _i32(min(_f32(s0), _f32(s1)))
      elif (pc_y := pc_dict.get(type(op_y), {}).get(op_y)):
        res = interp.execute(pc_y, s0, s1, scc=st.scc, d0=V[vdsty], vcc=st.vcc, lane=lane, literal=st.literal)
        V[vdsty] = res['d0']
    return

  # VOP3SD: has extra scalar dest for carry output
  if inst_type is VOP3SD:
    op = VOP3SDOp(inst.op)
    pc = pc_dict.get(VOP3SDOp, {}).get(op)
    if pc is None: raise NotImplementedError(f"{op.name} not in pseudocode")
    s0, s1, s2 = st.rsrc(inst.src0, lane), st.rsrc(inst.src1, lane), st.rsrc(inst.src2, lane)
    # For 64-bit src2, read from consecutive VGPRs
    if inst.src2 >= 256:  # VGPR
      s2 = V[inst.src2 - 256] | (V[inst.src2 - 256 + 1] << 32)
    d0 = V[inst.vdst]
    result = interp.execute(pc, s0, s1, s2, scc=st.scc, d0=d0, vcc=st.vcc, lane=lane, exec_mask=st.exec_mask)
    # Write result - handle 64-bit destinations
    if result.get('d0_64'):
      V[inst.vdst] = result['d0'] & 0xffffffff
      V[inst.vdst + 1] = (result['d0'] >> 32) & 0xffffffff
    else:
      V[inst.vdst] = result['d0'] & 0xffffffff
    if result.get('vcc_lane') is not None:
      st.pend_sgpr_lane(inst.sdst, lane, result['vcc_lane'])
    return

  # Get op enum and sources
  if inst_type is VOP1:
    if inst.op == VOP1Op.V_NOP: return
    op_cls, op, src0, src1, src2, vdst = VOP1Op, VOP1Op(inst.op), inst.src0, 0, 0, inst.vdst
  elif inst_type is VOP2:
    op_cls, op, src0, src1, src2, vdst = VOP2Op, VOP2Op(inst.op), inst.src0, inst.vsrc1 + 256, 0, inst.vdst
  elif inst_type is VOP3:
    # VOP3 ops 0-255 are VOPC comparisons encoded as VOP3 (use VOPCOp pseudocode)
    if inst.op < 256:
      op_cls, op, src0, src1, src2, vdst = VOPCOp, VOPCOp(inst.op), inst.src0, inst.src1, 0, inst.sdst if hasattr(inst, 'sdst') else VCC_LO
    else:
      op_cls, op, src0, src1, src2, vdst = VOP3Op, VOP3Op(inst.op), inst.src0, inst.src1, inst.src2, inst.vdst
  elif inst_type is VOPC:
    op_cls, op, src0, src1, src2, vdst = VOPCOp, VOPCOp(inst.op), inst.src0, inst.vsrc1 + 256, 0, VCC_LO
  else: raise NotImplementedError(f"Unknown vector type {inst_type}")

  pc = pc_dict.get(op_cls, {}).get(op)
  if pc is None: raise NotImplementedError(f"{op.name} not in pseudocode")

  # Read sources (with VOP3 modifiers if applicable)
  neg, abs_ = (getattr(inst, 'neg', 0), getattr(inst, 'abs', 0)) if inst_type is VOP3 else (0, 0)
  def mod_src(val: int, idx: int) -> int:
    if (abs_ >> idx) & 1: val = _i32(abs(_f32(val)))
    if (neg >> idx) & 1: val = _i32(-_f32(val))
    return val

  # Determine if sources are 64-bit based on instruction type
  # For 64-bit shift ops: src0 is 32-bit (shift amount), src1 is 64-bit (value to shift)
  # For most other _B64/_I64/_U64/_F64 ops: all sources are 64-bit
  is_64bit_op = op_cls in (VOP3Op, VOP1Op) and hasattr(op, 'name') and any(op.name.endswith(s) for s in ('_B64', '_I64', '_U64', '_F64'))
  is_shift_64 = op_cls is VOP3Op and op in (VOP3Op.V_LSHLREV_B64, VOP3Op.V_LSHRREV_B64, VOP3Op.V_ASHRREV_I64)

  if is_shift_64:
    s0 = mod_src(st.rsrc(src0, lane), 0)  # shift amount is 32-bit
    s1 = st.rsrc64(src1, lane) if src1 else 0  # value to shift is 64-bit
    s2 = mod_src(st.rsrc(src2, lane), 2) if src2 else 0
  elif is_64bit_op:
    s0 = st.rsrc64(src0, lane)
    s1 = st.rsrc64(src1, lane) if src1 else 0
    s2 = st.rsrc64(src2, lane) if src2 else 0
  else:
    s0 = mod_src(st.rsrc(src0, lane), 0)
    s1 = mod_src(st.rsrc(src1, lane), 1) if src1 else 0
    s2 = mod_src(st.rsrc(src2, lane), 2) if src2 else 0
  d0 = V[vdst] if not is_64bit_op else (V[vdst] | (V[vdst + 1] << 32))

  # Special case: ops with control flow that the interpreter can't handle
  if op in (VOP3Op.V_MAX_F32, VOP2Op.V_MAX_F32, VOP1Op.V_MAX_F32 if hasattr(VOP1Op, 'V_MAX_F32') else None):
    V[vdst] = _i32(max(_f32(s0), _f32(s1))); return
  if op in (VOP3Op.V_MIN_F32, VOP2Op.V_MIN_F32, VOP1Op.V_MIN_F32 if hasattr(VOP1Op, 'V_MIN_F32') else None):
    V[vdst] = _i32(min(_f32(s0), _f32(s1))); return

  # Execute pseudocode
  result = interp.execute(pc, s0, s1, s2, scc=st.scc, d0=d0, vcc=st.vcc, lane=lane, exec_mask=st.exec_mask, literal=st.literal)

  # Apply results
  if 'vcc_lane' in result:
    # VOP2 carry instructions (V_ADD_CO_CI_U32, V_SUB_CO_CI_U32, V_SUBREV_CO_CI_U32) write carry to VCC implicitly
    # VOPC and VOP3-encoded VOPC write to vdst (which is VCC_LO for VOPC, inst.sdst for VOP3)
    vcc_dst = VCC_LO if op_cls is VOP2Op and op in (VOP2Op.V_ADD_CO_CI_U32, VOP2Op.V_SUB_CO_CI_U32, VOP2Op.V_SUBREV_CO_CI_U32) else vdst
    st.pend_sgpr_lane(vcc_dst, lane, result['vcc_lane'])
  if 'exec_lane' in result:
    # V_CMPX instructions write to EXEC per-lane
    st.pend_sgpr_lane(EXEC_LO, lane, result['exec_lane'])
  if 'd0' in result and op_cls not in (VOPCOp,):
    if result.get('d0_64') or is_64bit_op:
      V[vdst] = result['d0'] & 0xffffffff
      V[vdst + 1] = (result['d0'] >> 32) & 0xffffffff
    else:
      V[vdst] = result['d0'] & 0xffffffff

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

def exec_workgroup(program: Program, workgroup_id: tuple[int, int, int], local_size: tuple[int, int, int], args_ptr: int, dispatch_dim: int) -> None:
  lx, ly, lz = local_size
  total_threads, lds = lx * ly * lz, bytearray(65536)
  waves: list[tuple[WaveState, int, int]] = []
  for wave_start in range(0, total_threads, WAVE_SIZE):
    n_lanes, st = min(WAVE_SIZE, total_threads - wave_start), WaveState()
    st.exec_mask = (1 << n_lanes) - 1
    st.wsgpr64(0, args_ptr)
    gx, gy, gz = workgroup_id
    if dispatch_dim >= 3: st.sgpr[13], st.sgpr[14], st.sgpr[15] = gx, gy, gz
    elif dispatch_dim == 2: st.sgpr[14], st.sgpr[15] = gx, gy
    else: st.sgpr[15] = gx
    for i in range(n_lanes):
      tid = wave_start + i
      st.vgpr[i][0] = tid if local_size == (lx, 1, 1) else ((tid // (lx * ly)) << 20) | (((tid // lx) % ly) << 10) | (tid % lx)
    waves.append((st, n_lanes, wave_start))
  has_barrier = any(isinstance(inst, SOPP) and inst.op == SOPPOp.S_BARRIER for inst in program.values())
  for _ in range(2 if has_barrier else 1):
    for st, n_lanes, _ in waves: exec_wave(program, st, lds, n_lanes)

def run_asm(lib: int, lib_sz: int, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, args_ptr: int) -> int:
  data = (ctypes.c_char * lib_sz).from_address(lib).raw
  program = decode_program(data)
  if not program: return -1
  dispatch_dim = 3 if gz > 1 else (2 if gy > 1 else 1)
  for gidz in range(gz):
    for gidy in range(gy):
      for gidx in range(gx): exec_workgroup(program, (gidx, gidy, gidz), (lx, ly, lz), args_ptr, dispatch_dim)
  return 0
