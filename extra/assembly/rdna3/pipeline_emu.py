# Cycle-accurate RDNA3 pipeline emulator
# Models real hardware timing for eventual Verilog translation
from __future__ import annotations
import ctypes, math
from dataclasses import dataclass, field
from typing import Any
from extra.assembly.rdna3.lib import Inst32, Inst64, RawImm
from extra.assembly.rdna3.autogen import (
  SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, VOP1, VOP2, VOP3, VOP3SD, VOP3P, VOPC, DS, FLAT, VOPD,
  SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, SMEMOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp, DSOp, FLATOp, GLOBALOp, VOPDOp
)
from extra.assembly.rdna3.alu import (
  salu, valu, vopc, f32, i32, f16, i16, sext, FLOAT_BITS,
  SOP1_BASE, SOP2_BASE, SOPC_BASE, SOPK_BASE, VOP1_BASE, VOP2_BASE
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS - from LLVM GFX11 SISchedule.td
# ═══════════════════════════════════════════════════════════════════════════════
VALU_LATENCY = 5       # V_ADD, V_MUL, etc. - uses HWVALU + HWRC
SALU_LATENCY = 2       # S_ADD, S_MUL, etc.
TRANS_LATENCY = 10     # V_RCP, V_SQRT, V_SIN, etc. (quarter rate transcendentals)
LDS_LATENCY = 20       # DS_* operations
SMEM_LATENCY = 20      # S_LOAD_* operations
VMEM_LATENCY = 320     # GLOBAL_LOAD/STORE, FLAT_*
BRANCH_LATENCY = 1     # S_BRANCH, S_CBRANCH_*
FORWARD_CYCLES = 2     # ReadAdvance<MIVGPRRead, -2> - results available 2 cycles early

WAVE_SIZE = 32
SGPR_COUNT = 128
VGPR_COUNT = 256
VCC_LO, VCC_HI, EXEC_LO, EXEC_HI, NULL_REG, M0 = 106, 107, 126, 127, 124, 125

# Transcendental ops (10 cycles instead of 5)
TRANS_OPS = {VOP1Op.V_RCP_F32, VOP1Op.V_RSQ_F32, VOP1Op.V_SQRT_F32, VOP1Op.V_LOG_F32, VOP1Op.V_EXP_F32, VOP1Op.V_SIN_F32, VOP1Op.V_COS_F32}

# ═══════════════════════════════════════════════════════════════════════════════
# DECODER - reuses autogen/lib
# ═══════════════════════════════════════════════════════════════════════════════
Inst = Inst32 | Inst64 | VOP3P
Program = dict[int, Inst]

def decode_format(word: int) -> tuple[type[Inst] | None, bool]:
  """Identify instruction format from first word. Returns (class, is_64bit)."""
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

def _unwrap(v) -> int:
  return v.val if isinstance(v, RawImm) else v.value if hasattr(v, 'value') else v

def decode_program(data: bytes) -> Program:
  """Decode binary into instruction dictionary keyed by PC (in words)."""
  result: Program = {}
  i = 0
  while i < len(data):
    word = int.from_bytes(data[i:i+4], 'little')
    inst_class, is_64 = decode_format(word)
    if inst_class is None: i += 4; continue
    base_size = 8 if is_64 else 4
    inst = inst_class.from_bytes(data[i:i+base_size])
    # Cache unwrapped field values
    for name, val in inst._values.items(): setattr(inst, name, _unwrap(val))
    # Check for literal
    has_literal = any(getattr(inst, fld, None) == 255 for fld in ('src0', 'src1', 'src2', 'ssrc0', 'ssrc1', 'srcx0', 'srcy0'))
    if inst_class == VOP2 and inst.op in (44, 45, 55, 56): has_literal = True
    if inst_class == VOPD and (inst.opx in (1, 2) or inst.opy in (1, 2)): has_literal = True
    if has_literal and len(data) >= i + base_size + 4: inst._literal = int.from_bytes(data[i+base_size:i+base_size+4], 'little')
    result[i // 4] = inst
    i += inst.size()
  return result

# ═══════════════════════════════════════════════════════════════════════════════
# REGISTER FILES
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class ScalarRegs:
  sgpr: list[int] = field(default_factory=lambda: [0] * SGPR_COUNT)
  scc: int = 0
  vcc: int = 0
  exec_mask: int = 0xffffffff
  m0: int = 0

  def read(self, idx: int) -> int:
    if idx == VCC_LO: return self.vcc & 0xffffffff
    if idx == VCC_HI: return (self.vcc >> 32) & 0xffffffff
    if idx == EXEC_LO: return self.exec_mask & 0xffffffff
    if idx == EXEC_HI: return (self.exec_mask >> 32) & 0xffffffff
    if idx == NULL_REG: return 0
    if idx == M0: return self.m0
    if idx == 253: return self.scc
    return self.sgpr[idx] if idx < SGPR_COUNT else 0

  def write(self, idx: int, val: int) -> None:
    val &= 0xffffffff
    if idx == VCC_LO: self.vcc = (self.vcc & 0xffffffff00000000) | val
    elif idx == VCC_HI: self.vcc = (self.vcc & 0xffffffff) | (val << 32)
    elif idx == EXEC_LO: self.exec_mask = (self.exec_mask & 0xffffffff00000000) | val
    elif idx == EXEC_HI: self.exec_mask = (self.exec_mask & 0xffffffff) | (val << 32)
    elif idx == M0: self.m0 = val
    elif idx == NULL_REG: pass
    elif idx < SGPR_COUNT: self.sgpr[idx] = val

  def read64(self, idx: int) -> int: return self.read(idx) | (self.read(idx + 1) << 32)
  def write64(self, idx: int, val: int) -> None: self.write(idx, val & 0xffffffff); self.write(idx + 1, (val >> 32) & 0xffffffff)

@dataclass
class VectorRegs:
  vgpr: list[list[int]] = field(default_factory=lambda: [[0] * VGPR_COUNT for _ in range(WAVE_SIZE)])

  def read(self, lane: int, idx: int) -> int: return self.vgpr[lane][idx] if idx < VGPR_COUNT else 0
  def write(self, lane: int, idx: int, val: int) -> None:
    if idx < VGPR_COUNT: self.vgpr[lane][idx] = val & 0xffffffff

# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY - simplified fixed-latency model
# ═══════════════════════════════════════════════════════════════════════════════
CTYPES = {1: ctypes.c_uint8, 2: ctypes.c_uint16, 4: ctypes.c_uint32}

_valid_mem_ranges: set[tuple[int, int]] = set()
def set_valid_mem_ranges(ranges: set[tuple[int, int]]) -> None: global _valid_mem_ranges; _valid_mem_ranges = ranges

def mem_read(addr: int, size: int) -> int:
  if _valid_mem_ranges and not any(s <= addr and addr + size <= s + z for s, z in _valid_mem_ranges):
    raise RuntimeError(f"OOB memory access at 0x{addr:x} size={size}")
  return CTYPES[size].from_address(addr).value

def mem_write(addr: int, size: int, val: int) -> None:
  if _valid_mem_ranges and not any(s <= addr and addr + size <= s + z for s, z in _valid_mem_ranges):
    raise RuntimeError(f"OOB memory access at 0x{addr:x} size={size}")
  CTYPES[size].from_address(addr).value = val

# Memory op tables
FLAT_LOAD = {GLOBALOp.GLOBAL_LOAD_B32: (1,4,0), FLATOp.FLAT_LOAD_B32: (1,4,0), GLOBALOp.GLOBAL_LOAD_B64: (2,4,0), FLATOp.FLAT_LOAD_B64: (2,4,0),
  GLOBALOp.GLOBAL_LOAD_B96: (3,4,0), FLATOp.FLAT_LOAD_B96: (3,4,0), GLOBALOp.GLOBAL_LOAD_B128: (4,4,0), FLATOp.FLAT_LOAD_B128: (4,4,0),
  GLOBALOp.GLOBAL_LOAD_U8: (1,1,0), FLATOp.FLAT_LOAD_U8: (1,1,0), GLOBALOp.GLOBAL_LOAD_I8: (1,1,1), FLATOp.FLAT_LOAD_I8: (1,1,1),
  GLOBALOp.GLOBAL_LOAD_U16: (1,2,0), FLATOp.FLAT_LOAD_U16: (1,2,0), GLOBALOp.GLOBAL_LOAD_I16: (1,2,1), FLATOp.FLAT_LOAD_I16: (1,2,1)}
FLAT_STORE = {GLOBALOp.GLOBAL_STORE_B32: (1,4), FLATOp.FLAT_STORE_B32: (1,4), GLOBALOp.GLOBAL_STORE_B64: (2,4), FLATOp.FLAT_STORE_B64: (2,4),
  GLOBALOp.GLOBAL_STORE_B96: (3,4), FLATOp.FLAT_STORE_B96: (3,4), GLOBALOp.GLOBAL_STORE_B128: (4,4), FLATOp.FLAT_STORE_B128: (4,4),
  GLOBALOp.GLOBAL_STORE_B8: (1,1), FLATOp.FLAT_STORE_B8: (1,1), GLOBALOp.GLOBAL_STORE_B16: (1,2), FLATOp.FLAT_STORE_B16: (1,2)}
DS_LOAD = {DSOp.DS_LOAD_B32: (1,4,0), DSOp.DS_LOAD_B64: (2,4,0), DSOp.DS_LOAD_B128: (4,4,0),
  DSOp.DS_LOAD_U8: (1,1,0), DSOp.DS_LOAD_I8: (1,1,1), DSOp.DS_LOAD_U16: (1,2,0), DSOp.DS_LOAD_I16: (1,2,1)}
DS_STORE = {DSOp.DS_STORE_B32: (1,4), DSOp.DS_STORE_B64: (2,4), DSOp.DS_STORE_B128: (4,4), DSOp.DS_STORE_B8: (1,1), DSOp.DS_STORE_B16: (1,2)}
SMEM_LOAD = {SMEMOp.S_LOAD_B32: 1, SMEMOp.S_LOAD_B64: 2, SMEMOp.S_LOAD_B128: 4, SMEMOp.S_LOAD_B256: 8, SMEMOp.S_LOAD_B512: 16}

# ═══════════════════════════════════════════════════════════════════════════════
# SCOREBOARD - hazard detection with forwarding
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class InFlight:
  """Instruction in pipeline with destination info."""
  pc: int
  inst: Any
  unit: str  # 'SALU', 'VALU', 'SMEM', 'VMEM', 'LDS', 'BRANCH'
  issue_cycle: int
  complete_cycle: int
  dst_sregs: list[int] = field(default_factory=list)
  dst_vregs: list[int] = field(default_factory=list)
  writes_vcc: bool = False
  writes_exec: bool = False
  writes_scc: bool = False

class Scoreboard:
  def __init__(self):
    self.sreg_ready: dict[int, int] = {}
    self.vreg_ready: dict[int, int] = {}
    self.vcc_ready: int = 0
    self.exec_ready: int = 0
    self.scc_ready: int = 0

  def add_write(self, inflight: InFlight) -> None:
    ready = inflight.complete_cycle - FORWARD_CYCLES
    for r in inflight.dst_sregs: self.sreg_ready[r] = inflight.complete_cycle
    for r in inflight.dst_vregs: self.vreg_ready[r] = ready
    if inflight.writes_vcc: self.vcc_ready = ready
    if inflight.writes_exec: self.exec_ready = ready
    if inflight.writes_scc: self.scc_ready = inflight.complete_cycle

  def sreg_available(self, reg: int, cycle: int) -> bool: return self.sreg_ready.get(reg, 0) <= cycle
  def vreg_available(self, reg: int, cycle: int) -> bool: return self.vreg_ready.get(reg, 0) <= cycle
  def cleanup(self, cycle: int) -> None:
    self.sreg_ready = {r: c for r, c in self.sreg_ready.items() if c > cycle}
    self.vreg_ready = {r: c for r, c in self.vreg_ready.items() if c > cycle}

# ═══════════════════════════════════════════════════════════════════════════════
# TRACE OUTPUT - SQTT-compatible format
# ═══════════════════════════════════════════════════════════════════════════════
TRACE_VALUINST, TRACE_VMEMEXEC, TRACE_ALUEXEC, TRACE_IMMEDIATE = 0x01, 0x02, 0x03, 0x04
TRACE_IMMEDIATE_MASK, TRACE_WAVEEND, TRACE_WAVESTART, TRACE_INST = 0x05, 0x08, 0x09, 0x18
INST_SALU, INST_SMEM, INST_JUMP, INST_NEXT, INST_MESSAGE = 0x00, 0x01, 0x03, 0x04, 0x09
INST_VALU, INST_VALU_SHIFT, INST_VALU_MAD, INST_VALU_CMPX = 0x0b, 0x0d, 0x0e, 0x73
INST_VMEM_LOAD, INST_VMEM_LOAD2, INST_VMEM_STORE = 0x21, 0x22, 0x24
INST_VMEM_STORE2, INST_VMEM_STORE3, INST_VMEM_STORE4 = 0x25, 0x27, 0x28
INST_LDS_LOAD, INST_LDS_STORE, INST_LDS_STORE2 = 0x29, 0x2b, 0x2e
ALUSRC_SALU, ALUSRC_VALU, ALUSRC_VALU_ALT = 1, 2, 3
MEMSRC_LDS, MEMSRC_VMEM = 0, 2

@dataclass
class TraceEvent:
  cycle: int
  opcode: int
  wave: int = 0
  inst_op: int = 0
  pc: int = 0
  src: int = 0
  mask: int = 0
  extra: dict = field(default_factory=dict)

  def __str__(self) -> str:
    names = {TRACE_WAVESTART: "WAVESTART", TRACE_WAVEEND: "WAVEEND", TRACE_INST: "INST",
             TRACE_ALUEXEC: "ALUEXEC", TRACE_VMEMEXEC: "VMEMEXEC", TRACE_VALUINST: "VALUINST",
             TRACE_IMMEDIATE: "IMMEDIATE", TRACE_IMMEDIATE_MASK: "IMMEDIATE_MASK"}
    inst_names = {INST_SALU: "SALU", INST_SMEM: "SMEM", INST_JUMP: "JUMP", INST_NEXT: "NEXT",
                  INST_VALU: "VALU", INST_VALU_SHIFT: "VALU", INST_VALU_MAD: "VALU", INST_VALU_CMPX: "VALU_CMPX",
                  INST_VMEM_LOAD: "VMEM_LOAD", INST_VMEM_LOAD2: "VMEM_LOAD",
                  INST_VMEM_STORE: "VMEM_STORE", INST_VMEM_STORE2: "VMEM_STORE", INST_VMEM_STORE3: "VMEM_STORE", INST_VMEM_STORE4: "VMEM_STORE",
                  INST_LDS_LOAD: "LDS_LOAD", INST_LDS_STORE: "LDS_STORE", INST_LDS_STORE2: "LDS_STORE"}
    alusrc_names = {ALUSRC_SALU: "SALU", ALUSRC_VALU: "VALU", ALUSRC_VALU_ALT: "VALU_ALT"}
    memsrc_names = {MEMSRC_LDS: "LDS", MEMSRC_VMEM: "VMEM"}
    s = f"{self.cycle:8d} : {names.get(self.opcode, f'0x{self.opcode:02x}'):14s}"
    if self.opcode == TRACE_INST: s += f" op=0x{self.inst_op:02x} [{inst_names.get(self.inst_op, '')}] pc={self.pc}"
    elif self.opcode == TRACE_VALUINST: s += f" wave={self.wave}"
    elif self.opcode == TRACE_ALUEXEC: s += f" src={self.src} [{alusrc_names.get(self.src, '')}]"
    elif self.opcode == TRACE_VMEMEXEC: s += f" src={self.src} [{memsrc_names.get(self.src, '')}]"
    elif self.opcode == TRACE_IMMEDIATE_MASK: s += f" mask={self.mask:016b}"
    elif self.opcode in (TRACE_WAVESTART, TRACE_WAVEEND): s += f" wave={self.wave}"
    return s

# ═══════════════════════════════════════════════════════════════════════════════
# WAVE STATE
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Wave:
  pc: int = 0
  sregs: ScalarRegs = field(default_factory=ScalarRegs)
  vregs: VectorRegs = field(default_factory=VectorRegs)
  literal: int = 0
  n_lanes: int = WAVE_SIZE
  done: bool = False
  barrier: bool = False
  _pend_vcc: int | None = None
  _pend_exec: int | None = None
  _pend_sgpr: dict[int, int] = field(default_factory=dict)

  def rsrc(self, v: int, lane: int) -> int:
    if v <= 105: return self.sregs.sgpr[v]
    if v in (VCC_LO, VCC_HI): return (self.sregs.vcc >> (32 if v == VCC_HI else 0)) & 0xffffffff
    if 108 <= v <= 123 or v == M0: return self.sregs.sgpr[v] if v < SGPR_COUNT else self.sregs.m0
    if v in (EXEC_LO, EXEC_HI): return (self.sregs.exec_mask >> (32 if v == EXEC_HI else 0)) & 0xffffffff
    if v == NULL_REG: return 0
    if 128 <= v <= 192: return v - 128
    if 193 <= v <= 208: return (-(v - 192)) & 0xffffffff
    if v in FLOAT_BITS: return FLOAT_BITS[v]
    if v == 255: return self.literal
    if 256 <= v <= 511: return self.vregs.vgpr[lane][v - 256]
    return 0

  def rsrc64(self, v: int, lane: int) -> int:
    return self.rsrc(v, lane) | ((self.rsrc(v+1, lane) if v <= 105 or 256 <= v <= 511 else 0) << 32)

  def pend_vcc_lane(self, lane: int, val: bool) -> None:
    if self._pend_vcc is None: self._pend_vcc = 0
    if val: self._pend_vcc |= (1 << lane)

  def pend_exec_lane(self, lane: int, val: bool) -> None:
    if self._pend_exec is None: self._pend_exec = 0
    if val: self._pend_exec |= (1 << lane)

  def pend_sgpr_lane(self, reg: int, lane: int, val: bool) -> None:
    if reg not in self._pend_sgpr: self._pend_sgpr[reg] = 0
    if val: self._pend_sgpr[reg] |= (1 << lane)

  def commit_pends(self) -> None:
    if self._pend_vcc is not None: self.sregs.vcc = self._pend_vcc; self._pend_vcc = None
    if self._pend_exec is not None: self.sregs.exec_mask = self._pend_exec; self._pend_exec = None
    for reg, val in self._pend_sgpr.items(): self.sregs.write(reg, val)
    self._pend_sgpr.clear()

# ═══════════════════════════════════════════════════════════════════════════════
# CU - Compute Unit (main simulation loop)
# ═══════════════════════════════════════════════════════════════════════════════
class CU:
  def __init__(self, program: Program, wave: Wave, lds: bytearray | None = None):
    self.program = program
    self.wave = wave
    self.lds = lds if lds is not None else bytearray(65536)
    self.cycle = 0
    self.scoreboard = Scoreboard()
    self.in_flight: list[InFlight] = []
    self.trace: list[TraceEvent] = []
    self._emit_trace(TRACE_WAVESTART, wave=0)

  def _emit_trace(self, opcode: int, **kwargs) -> None:
    self.trace.append(TraceEvent(cycle=self.cycle, opcode=opcode, **kwargs))

  def _get_inst_type(self, inst) -> int:
    t = type(inst)
    if t in (SOP1, SOP2, SOPC, SOPK): return INST_SALU
    if t == SMEM: return INST_SMEM
    if t == SOPP:
      if inst.op in (SOPPOp.S_BRANCH, SOPPOp.S_CBRANCH_SCC0, SOPPOp.S_CBRANCH_SCC1,
                     SOPPOp.S_CBRANCH_VCCZ, SOPPOp.S_CBRANCH_VCCNZ,
                     SOPPOp.S_CBRANCH_EXECZ, SOPPOp.S_CBRANCH_EXECNZ):
        return INST_JUMP
      return INST_SALU
    if t in (VOP1, VOP2, VOP3, VOP3SD, VOP3P, VOPC, VOPD): return INST_VALU
    if t == FLAT: return INST_VMEM_LOAD if inst.op in FLAT_LOAD else INST_VMEM_STORE
    if t == DS: return INST_LDS_LOAD if inst.op in DS_LOAD else INST_LDS_STORE
    return INST_SALU

  def _get_latency(self, inst) -> int:
    t = type(inst)
    if t in (SOP1, SOP2, SOPC, SOPK, SOPP): return SALU_LATENCY
    if t == SMEM: return SMEM_LATENCY
    if t in (VOP1, VOP2, VOP3, VOP3SD, VOPC, VOPD):
      if t == VOP1 and inst.op in TRANS_OPS: return TRANS_LATENCY
      if t == VOP3 and (inst.op - 384) in TRANS_OPS: return TRANS_LATENCY
      return VALU_LATENCY
    if t == VOP3P: return VALU_LATENCY
    if t == FLAT: return VMEM_LATENCY
    if t == DS: return LDS_LATENCY
    return VALU_LATENCY

  def _get_unit(self, inst) -> str:
    t = type(inst)
    if t in (SOP1, SOP2, SOPC, SOPK, SOPP): return 'SALU'
    if t == SMEM: return 'SMEM'
    if t in (VOP1, VOP2, VOP3, VOP3SD, VOP3P, VOPC, VOPD): return 'VALU'
    if t == FLAT: return 'VMEM'
    if t == DS: return 'LDS'
    return 'SALU'

  def _exec_scalar(self, inst) -> int:
    """Execute scalar instruction. Returns PC delta (for branches)."""
    w = self.wave
    t = type(inst)

    if t == SOP1:
      s0, op = w.rsrc(inst.ssrc0, 0), inst.op
      if op == SOP1Op.S_MOV_B64: w.sregs.write64(inst.sdst, w.rsrc64(inst.ssrc0, 0)); return 0
      if op == SOP1Op.S_NOT_B64: r = (~w.rsrc64(inst.ssrc0, 0)) & 0xffffffffffffffff; w.sregs.write64(inst.sdst, r); w.sregs.scc = int(r != 0); return 0
      if op == SOP1Op.S_BITSET0_B32: w.sregs.write(inst.sdst, w.sregs.read(inst.sdst) & ~(1 << (s0 & 0x1f))); return 0
      if op == SOP1Op.S_BITSET1_B32: w.sregs.write(inst.sdst, w.sregs.read(inst.sdst) | (1 << (s0 & 0x1f))); return 0
      if op == SOP1Op.S_AND_SAVEEXEC_B32: old = w.sregs.exec_mask & 0xffffffff; w.sregs.exec_mask = s0 & old; w.sregs.scc = int(w.sregs.exec_mask != 0); w.sregs.write(inst.sdst, old); return 0
      if op == SOP1Op.S_OR_SAVEEXEC_B32: old = w.sregs.exec_mask & 0xffffffff; w.sregs.exec_mask = s0 | old; w.sregs.scc = int(w.sregs.exec_mask != 0); w.sregs.write(inst.sdst, old); return 0
      if op == SOP1Op.S_AND_NOT1_SAVEEXEC_B32: old = w.sregs.exec_mask & 0xffffffff; w.sregs.exec_mask = s0 & (~old & 0xffffffff); w.sregs.scc = int(w.sregs.exec_mask != 0); w.sregs.write(inst.sdst, old); return 0
      r, scc = salu(SOP1_BASE + op, s0, 0, w.sregs.scc)
      w.sregs.write(inst.sdst, r); w.sregs.scc = scc; return 0

    if t == SOP2:
      s0, s1, op = w.rsrc(inst.ssrc0, 0), w.rsrc(inst.ssrc1, 0), inst.op
      if op == SOP2Op.S_LSHL_B64: r = (w.rsrc64(inst.ssrc0, 0) << (s1 & 0x3f)) & 0xffffffffffffffff; w.sregs.write64(inst.sdst, r); w.sregs.scc = int(r != 0)
      elif op == SOP2Op.S_LSHR_B64: r = w.rsrc64(inst.ssrc0, 0) >> (s1 & 0x3f); w.sregs.write64(inst.sdst, r); w.sregs.scc = int(r != 0)
      elif op == SOP2Op.S_ASHR_I64: r = sext(w.rsrc64(inst.ssrc0, 0), 64) >> (s1 & 0x3f); w.sregs.write64(inst.sdst, r & 0xffffffffffffffff); w.sregs.scc = int(r != 0)
      elif op == SOP2Op.S_AND_B64: r = w.rsrc64(inst.ssrc0, 0) & w.rsrc64(inst.ssrc1, 0); w.sregs.write64(inst.sdst, r); w.sregs.scc = int(r != 0)
      elif op == SOP2Op.S_OR_B64: r = w.rsrc64(inst.ssrc0, 0) | w.rsrc64(inst.ssrc1, 0); w.sregs.write64(inst.sdst, r); w.sregs.scc = int(r != 0)
      elif op == SOP2Op.S_XOR_B64: r = w.rsrc64(inst.ssrc0, 0) ^ w.rsrc64(inst.ssrc1, 0); w.sregs.write64(inst.sdst, r); w.sregs.scc = int(r != 0)
      elif op == SOP2Op.S_CSELECT_B64: w.sregs.write64(inst.sdst, w.rsrc64(inst.ssrc0, 0) if w.sregs.scc else w.rsrc64(inst.ssrc1, 0))
      else: r, scc = salu(SOP2_BASE + op, s0, s1, w.sregs.scc); w.sregs.write(inst.sdst, r); w.sregs.scc = scc
      return 0

    if t == SOPC:
      s0, s1, op = w.rsrc(inst.ssrc0, 0), w.rsrc(inst.ssrc1, 0), inst.op
      if op == SOPCOp.S_CMP_EQ_U64: w.sregs.scc = int(w.rsrc64(inst.ssrc0, 0) == w.rsrc64(inst.ssrc1, 0))
      elif op == SOPCOp.S_CMP_LG_U64: w.sregs.scc = int(w.rsrc64(inst.ssrc0, 0) != w.rsrc64(inst.ssrc1, 0))
      else: _, scc = salu(SOPC_BASE + op, s0, s1, w.sregs.scc); w.sregs.scc = scc
      return 0

    if t == SOPK:
      simm, s0, op = inst.simm16, w.sregs.read(inst.sdst), inst.op
      if op in (SOPKOp.S_WAITCNT_VSCNT, SOPKOp.S_WAITCNT_VMCNT, SOPKOp.S_WAITCNT_EXPCNT, SOPKOp.S_WAITCNT_LGKMCNT): return 0
      r, scc = salu(SOPK_BASE + op, s0, simm, w.sregs.scc)
      if op not in (SOPKOp.S_CMPK_EQ_I32, SOPKOp.S_CMPK_LG_I32, SOPKOp.S_CMPK_GT_I32, SOPKOp.S_CMPK_GE_I32,
                    SOPKOp.S_CMPK_LT_I32, SOPKOp.S_CMPK_LE_I32, SOPKOp.S_CMPK_EQ_U32, SOPKOp.S_CMPK_LG_U32,
                    SOPKOp.S_CMPK_GT_U32, SOPKOp.S_CMPK_GE_U32, SOPKOp.S_CMPK_LT_U32, SOPKOp.S_CMPK_LE_U32):
        w.sregs.write(inst.sdst, r)
      w.sregs.scc = scc; return 0

    if t == SOPP:
      op = inst.op
      if op == SOPPOp.S_ENDPGM: w.done = True; return 0
      if op == SOPPOp.S_BARRIER: w.barrier = True; return 0
      if op == SOPPOp.S_BRANCH: return sext(inst.simm16, 16)
      if op == SOPPOp.S_CBRANCH_SCC0: return sext(inst.simm16, 16) if w.sregs.scc == 0 else 0
      if op == SOPPOp.S_CBRANCH_SCC1: return sext(inst.simm16, 16) if w.sregs.scc == 1 else 0
      if op == SOPPOp.S_CBRANCH_VCCZ: return sext(inst.simm16, 16) if w.sregs.vcc == 0 else 0
      if op == SOPPOp.S_CBRANCH_VCCNZ: return sext(inst.simm16, 16) if w.sregs.vcc != 0 else 0
      if op == SOPPOp.S_CBRANCH_EXECZ: return sext(inst.simm16, 16) if w.sregs.exec_mask == 0 else 0
      if op == SOPPOp.S_CBRANCH_EXECNZ: return sext(inst.simm16, 16) if w.sregs.exec_mask != 0 else 0
      if op in (SOPPOp.S_NOP, SOPPOp.S_WAITCNT, SOPPOp.S_DELAY_ALU, SOPPOp.S_CLAUSE): return 0
      raise NotImplementedError(f"SOPP op {op}")

    if t == SMEM:
      addr = w.sregs.read64(inst.sbase * 2) + sext(inst.offset, 21)
      if inst.soffset not in (NULL_REG, 0x7f): addr += w.rsrc(inst.soffset, 0)
      cnt = SMEM_LOAD.get(inst.op)
      if cnt is None: raise NotImplementedError(f"SMEM op {inst.op}")
      for i in range(cnt): w.sregs.write(inst.sdata + i, mem_read((addr + i * 4) & 0xffffffffffffffff, 4))
      return 0

    raise NotImplementedError(f"Unknown scalar inst type {t}")

  def _exec_vector_lane(self, inst, lane: int) -> None:
    """Execute vector instruction for one lane."""
    w = self.wave
    V = w.vregs.vgpr[lane]
    t = type(inst)

    if t == VOP1:
      if inst.op == VOP1Op.V_NOP: return
      s0 = w.rsrc(inst.src0, lane)
      if inst.op == VOP1Op.V_READFIRSTLANE_B32:
        first = (w.sregs.exec_mask & -w.sregs.exec_mask).bit_length() - 1 if w.sregs.exec_mask else 0
        w.sregs.write(inst.vdst, w.rsrc(inst.src0, first) if inst.src0 >= 256 else s0); return
      r = valu(VOP1_BASE + inst.op, s0, 0, 0)
      if r is not None: V[inst.vdst] = r
      return

    if t == VOP2:
      s0, s1, op = w.rsrc(inst.src0, lane), V[inst.vsrc1], inst.op
      if op == VOP2Op.V_CNDMASK_B32: V[inst.vdst] = s1 if (w.sregs.vcc >> lane) & 1 else s0; return
      if op == VOP2Op.V_FMAC_F32: V[inst.vdst] = i32(f32(s0)*f32(s1)+f32(V[inst.vdst])); return
      if op == VOP2Op.V_FMAMK_F32: V[inst.vdst] = i32(f32(s0)*f32(w.literal)+f32(s1)); return
      if op == VOP2Op.V_FMAAK_F32: V[inst.vdst] = i32(f32(s0)*f32(s1)+f32(w.literal)); return
      # Carry ops
      if op == VOP2Op.V_ADD_CO_CI_U32:
        vcc_bit = (w.sregs.vcc >> lane) & 1; r = s0 + s1 + vcc_bit
        w.pend_vcc_lane(lane, r >= 0x100000000); V[inst.vdst] = r & 0xffffffff; return
      if op == VOP2Op.V_SUB_CO_CI_U32:
        vcc_bit = (w.sregs.vcc >> lane) & 1
        w.pend_vcc_lane(lane, (s1 + vcc_bit) > s0); V[inst.vdst] = (s0 - s1 - vcc_bit) & 0xffffffff; return
      if op == VOP2Op.V_SUBREV_CO_CI_U32:
        vcc_bit = (w.sregs.vcc >> lane) & 1
        w.pend_vcc_lane(lane, (s0 + vcc_bit) > s1); V[inst.vdst] = (s1 - s0 - vcc_bit) & 0xffffffff; return
      r = valu(VOP2_BASE + op, s0, s1, 0)
      if r is not None: V[inst.vdst] = r
      return

    if t == VOP3:
      op, neg, abs_ = inst.op, inst.neg, getattr(inst, 'abs', 0)
      def mod(val, idx):
        if (abs_ >> idx) & 1: val = i32(abs(f32(val)))
        if (neg >> idx) & 1: val = i32(-f32(val))
        return val
      s0, s1, s2 = mod(w.rsrc(inst.src0, lane), 0), mod(w.rsrc(inst.src1, lane), 1), mod(w.rsrc(inst.src2, lane), 2)
      # VOPC in VOP3 encoding (0-255)
      if 0 <= op <= 255:
        cmp_result = vopc(op, s0, s1)
        is_cmpx = op >= 128
        if inst.vdst == VCC_LO: w.pend_vcc_lane(lane, cmp_result)
        else: w.pend_sgpr_lane(inst.vdst, lane, cmp_result)
        if is_cmpx: w.pend_exec_lane(lane, cmp_result)
        return
      # Special ops
      if op == VOP3Op.V_CNDMASK_B32:
        mask = w.sregs.read(inst.src2) if inst.src2 < 256 else w.sregs.vcc
        V[inst.vdst] = s1 if (mask >> lane) & 1 else s0; return
      if op == VOP3Op.V_FMAC_F32: V[inst.vdst] = i32(f32(s0)*f32(s1)+f32(V[inst.vdst])); return
      if op == VOP3Op.V_READLANE_B32: w.sregs.write(inst.vdst, V[inst.src0 - 256] if inst.src0 >= 256 else s0); return
      if op == VOP3Op.V_WRITELANE_B32: w.vregs.vgpr[s1 & 0x1f][inst.vdst] = s0; return
      if op == VOP3Op.V_PACK_B32_F16: V[inst.vdst] = (s0 & 0xffff) | ((s1 & 0xffff) << 16); return
      # Use VALU dict
      r = valu(op, s0, s1, s2)
      if r is not None: V[inst.vdst] = r
      return

    if t == VOP3SD:
      op, s0, s1, s2 = inst.op, w.rsrc(inst.src0, lane), w.rsrc(inst.src1, lane), w.rsrc(inst.src2, lane)
      if op == VOP3SDOp.V_ADD_CO_U32: r = s0 + s1; V[inst.vdst] = r & 0xffffffff; w.pend_sgpr_lane(inst.sdst, lane, r >= 0x100000000)
      elif op == VOP3SDOp.V_SUB_CO_U32: V[inst.vdst] = (s0 - s1) & 0xffffffff; w.pend_sgpr_lane(inst.sdst, lane, s1 > s0)
      elif op == VOP3SDOp.V_SUBREV_CO_U32: V[inst.vdst] = (s1 - s0) & 0xffffffff; w.pend_sgpr_lane(inst.sdst, lane, s0 > s1)
      elif op == VOP3SDOp.V_ADD_CO_CI_U32:
        cin = (w.sregs.read(inst.src2) >> lane) & 1 if inst.src2 < 256 else (w.sregs.vcc >> lane) & 1
        r = s0 + s1 + cin; V[inst.vdst] = r & 0xffffffff; w.pend_sgpr_lane(inst.sdst, lane, r >= 0x100000000)
      elif op == VOP3SDOp.V_SUB_CO_CI_U32:
        cin = (w.sregs.read(inst.src2) >> lane) & 1 if inst.src2 < 256 else (w.sregs.vcc >> lane) & 1
        V[inst.vdst] = (s0 - s1 - cin) & 0xffffffff; w.pend_sgpr_lane(inst.sdst, lane, s1 + cin > s0)
      elif op == VOP3SDOp.V_MAD_U64_U32:
        s2_64 = s2 | (w.rsrc(inst.src2 + 1, lane) << 32); r = s0 * s1 + s2_64
        V[inst.vdst] = r & 0xffffffff; V[inst.vdst + 1] = (r >> 32) & 0xffffffff
      elif op == VOP3SDOp.V_MAD_I64_I32:
        s2_64 = sext(s2 | (w.rsrc(inst.src2 + 1, lane) << 32), 64)
        r = (sext(s0, 32) * sext(s1, 32) + s2_64) & 0xffffffffffffffff
        V[inst.vdst] = r & 0xffffffff; V[inst.vdst + 1] = (r >> 32) & 0xffffffff
      elif op == VOP3SDOp.V_DIV_SCALE_F32: V[inst.vdst] = 0; w.pend_sgpr_lane(inst.sdst, lane, False)
      else: raise NotImplementedError(f"VOP3SD op {op}")
      return

    if t == VOPC:
      cmp_result = vopc(inst.op, w.rsrc(inst.src0, lane), V[inst.vsrc1])
      is_cmpx = inst.op >= 128
      (w.pend_exec_lane if is_cmpx else w.pend_vcc_lane)(lane, cmp_result)
      return

    if t == VOPD:
      vdsty = (inst.vdsty << 1) | ((inst.vdstx & 1) ^ 1)
      sx0, sx1, sy0, sy1 = w.rsrc(inst.srcx0, lane), V[inst.vsrcx1], w.rsrc(inst.srcy0, lane), V[inst.vsrcy1]
      opx, dstx = inst.opx, inst.vdstx
      if opx == VOPDOp.V_DUAL_MOV_B32: V[dstx] = sx0
      elif opx == VOPDOp.V_DUAL_ADD_F32: V[dstx] = i32(f32(sx0) + f32(sx1))
      elif opx == VOPDOp.V_DUAL_MUL_F32: V[dstx] = i32(f32(sx0) * f32(sx1))
      elif opx == VOPDOp.V_DUAL_FMAC_F32: V[dstx] = i32(f32(sx0)*f32(sx1)+f32(V[dstx]))
      elif opx == VOPDOp.V_DUAL_FMAAK_F32: V[dstx] = i32(f32(sx0)*f32(sx1)+f32(w.literal))
      elif opx == VOPDOp.V_DUAL_FMAMK_F32: V[dstx] = i32(f32(sx0)*f32(w.literal)+f32(sx1))
      elif opx == VOPDOp.V_DUAL_CNDMASK_B32: V[dstx] = sx1 if (w.sregs.vcc >> lane) & 1 else sx0
      elif opx == VOPDOp.V_DUAL_ADD_NC_U32: V[dstx] = (sx0 + sx1) & 0xffffffff
      elif opx == VOPDOp.V_DUAL_LSHLREV_B32: V[dstx] = (sx1 << (sx0 & 0x1f)) & 0xffffffff
      elif opx == VOPDOp.V_DUAL_AND_B32: V[dstx] = sx0 & sx1
      else: raise NotImplementedError(f"VOPD opx {opx}")
      opy = inst.opy
      if opy == VOPDOp.V_DUAL_MOV_B32: V[vdsty] = sy0
      elif opy == VOPDOp.V_DUAL_ADD_F32: V[vdsty] = i32(f32(sy0) + f32(sy1))
      elif opy == VOPDOp.V_DUAL_MUL_F32: V[vdsty] = i32(f32(sy0) * f32(sy1))
      elif opy == VOPDOp.V_DUAL_FMAC_F32: V[vdsty] = i32(f32(sy0)*f32(sy1)+f32(V[vdsty]))
      elif opy == VOPDOp.V_DUAL_FMAAK_F32: V[vdsty] = i32(f32(sy0)*f32(sy1)+f32(w.literal))
      elif opy == VOPDOp.V_DUAL_FMAMK_F32: V[vdsty] = i32(f32(sy0)*f32(w.literal)+f32(sy1))
      elif opy == VOPDOp.V_DUAL_CNDMASK_B32: V[vdsty] = sy1 if (w.sregs.vcc >> lane) & 1 else sy0
      elif opy == VOPDOp.V_DUAL_ADD_NC_U32: V[vdsty] = (sy0 + sy1) & 0xffffffff
      elif opy == VOPDOp.V_DUAL_LSHLREV_B32: V[vdsty] = (sy1 << (sy0 & 0x1f)) & 0xffffffff
      elif opy == VOPDOp.V_DUAL_AND_B32: V[vdsty] = sy0 & sy1
      else: raise NotImplementedError(f"VOPD opy {opy}")
      return

    if t == VOP3P:
      op, s0, s1, s2 = inst.op, w.rsrc(inst.src0, lane), w.rsrc(inst.src1, lane), w.rsrc(inst.src2, lane)
      opsel = [(inst.opsel >> i) & 1 for i in range(3)]
      opsel_hi = [(inst.opsel_hi >> i) & 1 for i in range(2)] + [inst.opsel_hi2]
      neg, neg_hi = inst.neg, inst.neg_hi
      def get_src(src: int, idx: int, for_mix: bool = False) -> float:
        if for_mix:
          if not opsel_hi[idx]: return abs(f32(src)) if (neg_hi >> idx) & 1 else f32(src)
          return float(f16((src >> 16) & 0xffff) if opsel[idx] else f16(src & 0xffff))
        use_hi = opsel[idx]
        val = ((src >> 16) & 0xffff) if use_hi else (src & 0xffff)
        f = f16(val)
        if use_hi and (neg >> idx) & 1: f = -f
        elif not use_hi and (neg_hi >> idx) & 1: f = -f
        return f
      if op == VOP3POp.V_FMA_MIX_F32: V[inst.vdst] = i32(get_src(s0, 0, True) * get_src(s1, 1, True) + get_src(s2, 2, True))
      elif op == VOP3POp.V_FMA_MIXLO_F16: V[inst.vdst] = (V[inst.vdst] & 0xffff0000) | i16(get_src(s0, 0, True) * get_src(s1, 1, True) + get_src(s2, 2, True))
      elif op == VOP3POp.V_FMA_MIXHI_F16: V[inst.vdst] = (V[inst.vdst] & 0x0000ffff) | (i16(get_src(s0, 0, True) * get_src(s1, 1, True) + get_src(s2, 2, True)) << 16)
      elif op == VOP3POp.V_PK_ADD_F16: V[inst.vdst] = i16(f16(s0 & 0xffff) + f16(s1 & 0xffff)) | (i16(f16((s0 >> 16) & 0xffff) + f16((s1 >> 16) & 0xffff)) << 16)
      elif op == VOP3POp.V_PK_MUL_F16: V[inst.vdst] = i16(f16(s0 & 0xffff) * f16(s1 & 0xffff)) | (i16(f16((s0 >> 16) & 0xffff) * f16((s1 >> 16) & 0xffff)) << 16)
      elif op == VOP3POp.V_PK_FMA_F16: V[inst.vdst] = i16(f16(s0 & 0xffff) * f16(s1 & 0xffff) + f16(s2 & 0xffff)) | (i16(f16((s0 >> 16) & 0xffff) * f16((s1 >> 16) & 0xffff) + f16((s2 >> 16) & 0xffff)) << 16)
      else: raise NotImplementedError(f"VOP3P op {op}")
      return

    if t == FLAT:
      op, offset, saddr = inst.op, sext(inst.offset, 13), inst.saddr
      addr = V[inst.addr] | (V[inst.addr + 1] << 32)
      if saddr not in (NULL_REG, 0x7f): addr = (w.sregs.read64(saddr) + V[inst.addr] + offset) & 0xffffffffffffffff
      else: addr = (addr + offset) & 0xffffffffffffffff
      if op in FLAT_LOAD:
        cnt, sz, sign = FLAT_LOAD[op]
        for i in range(cnt):
          val = mem_read(addr + i * sz, sz)
          V[inst.vdst + i] = sext(val, sz * 8) & 0xffffffff if sign else val
      elif op in FLAT_STORE:
        cnt, sz = FLAT_STORE[op]
        for i in range(cnt): mem_write(addr + i * sz, sz, V[inst.data + i] & ((1 << (sz * 8)) - 1))
      else: raise NotImplementedError(f"FLAT op {op}")
      return

    if t == DS:
      op, addr = inst.op, (V[inst.addr] + inst.offset0) & 0xffff
      if op in DS_LOAD:
        cnt, sz, sign = DS_LOAD[op]
        for i in range(cnt):
          val = int.from_bytes(self.lds[addr+i*sz:addr+i*sz+sz], 'little')
          V[inst.vdst + i] = sext(val, sz * 8) & 0xffffffff if sign else val
      elif op in DS_STORE:
        cnt, sz = DS_STORE[op]
        for i in range(cnt): self.lds[addr+i*sz:addr+i*sz+sz] = (V[inst.data0 + i] & ((1 << (sz * 8)) - 1)).to_bytes(sz, 'little')
      else: raise NotImplementedError(f"DS op {op}")
      return

    raise NotImplementedError(f"Unknown vector inst type {t}")

  def tick(self) -> bool:
    if self.wave.done:
      self._emit_trace(TRACE_WAVEEND, wave=0)
      return False
    completed = [inf for inf in self.in_flight if inf.complete_cycle <= self.cycle]
    for inf in completed:
      self.in_flight.remove(inf)
      if inf.unit == 'VALU': self._emit_trace(TRACE_ALUEXEC, wave=0)
      elif inf.unit in ('VMEM', 'SMEM'): self._emit_trace(TRACE_VMEMEXEC, wave=0)
    inst = self.program.get(self.wave.pc)
    if inst is None: self.wave.done = True; return True
    inst_words = inst.size() // 4
    self.wave.literal = inst._literal if hasattr(inst, '_literal') and inst._literal else 0
    inst_type = self._get_inst_type(inst)
    self._emit_trace(TRACE_INST, inst_op=inst_type, pc=self.wave.pc * 4)
    t = type(inst)
    if t in (SOP1, SOP2, SOPC, SOPK, SOPP, SMEM):
      delta = self._exec_scalar(inst)
      if self.wave.done: return True
      self.wave.pc += inst_words + delta
    else:
      exec_mask = self.wave.sregs.exec_mask
      for lane in range(self.wave.n_lanes):
        if exec_mask & (1 << lane): self._exec_vector_lane(inst, lane)
      self.wave.commit_pends()
      self.wave.pc += inst_words
    latency, unit = self._get_latency(inst), self._get_unit(inst)
    inflight = InFlight(pc=self.wave.pc - inst_words, inst=inst, unit=unit, issue_cycle=self.cycle, complete_cycle=self.cycle + latency)
    self.in_flight.append(inflight)
    self.scoreboard.add_write(inflight)
    self.cycle += 1
    return True

  def run(self, max_cycles: int = 1_000_000) -> int:
    while self.cycle < max_cycles:
      if not self.tick(): break
    return self.cycle

  def print_trace(self) -> None:
    last_cycle = 0
    for evt in self.trace:
      delta = evt.cycle - last_cycle
      print(f"{evt.cycle:8d} +{delta:8d} : {evt}")
      last_cycle = evt.cycle

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════════════
def run_program(data: bytes, n_lanes: int = WAVE_SIZE, args_ptr: int = 0,
                workgroup_id: tuple[int, int, int] = (0, 0, 0)) -> tuple[Wave, int, list[TraceEvent]]:
  program = decode_program(data)
  wave = Wave(n_lanes=n_lanes)
  wave.sregs.exec_mask = (1 << n_lanes) - 1
  wave.sregs.write64(0, args_ptr)
  wave.sregs.sgpr[13], wave.sregs.sgpr[14], wave.sregs.sgpr[15] = workgroup_id
  cu = CU(program, wave)
  cycles = cu.run()
  return wave, cycles, cu.trace

def compare_with_functional(data: bytes, n_lanes: int = WAVE_SIZE, args_ptr: int = 0) -> tuple[bool, str]:
  from extra.assembly.rdna3.emu import WaveState as FuncWaveState, decode_program as func_decode, step_wave
  func_program = func_decode(data)
  func_state = FuncWaveState()
  func_state.exec_mask = (1 << n_lanes) - 1
  func_state.sgpr[0] = args_ptr & 0xffffffff
  func_state.sgpr[1] = (args_ptr >> 32) & 0xffffffff
  func_lds = bytearray(65536)
  while func_state.pc in func_program:
    result = step_wave(func_program, func_state, func_lds, n_lanes)
    if result == -1: break
  pipe_wave, cycles, trace = run_program(data, n_lanes, args_ptr)
  diffs = []
  if func_state.scc != pipe_wave.sregs.scc: diffs.append(f"scc: {func_state.scc} vs {pipe_wave.sregs.scc}")
  if func_state.vcc != pipe_wave.sregs.vcc: diffs.append(f"vcc: 0x{func_state.vcc:x} vs 0x{pipe_wave.sregs.vcc:x}")
  if func_state.exec_mask != pipe_wave.sregs.exec_mask: diffs.append(f"exec: 0x{func_state.exec_mask:x} vs 0x{pipe_wave.sregs.exec_mask:x}")
  for i in range(SGPR_COUNT):
    if func_state.sgpr[i] != pipe_wave.sregs.sgpr[i]: diffs.append(f"sgpr[{i}]: 0x{func_state.sgpr[i]:x} vs 0x{pipe_wave.sregs.sgpr[i]:x}")
  for lane in range(n_lanes):
    for i in range(VGPR_COUNT):
      if func_state.vgpr[lane][i] != pipe_wave.vregs.vgpr[lane][i]:
        diffs.append(f"vgpr[{lane}][{i}]: 0x{func_state.vgpr[lane][i]:x} vs 0x{pipe_wave.vregs.vgpr[lane][i]:x}")
  if diffs: return False, f"Mismatch after {cycles} cycles:\n  " + "\n  ".join(diffs[:20])
  return True, f"Match after {cycles} cycles"
