# Cycle-stepped RDNA3 (gfx1100, wave32) Compute Unit / WGP timing model for SQTT emulation.
# Models two SIMDs sharing an L0 instruction cache, 32-bank LDS, and workgroup barrier unit:
#   fetch (ICache -> 4-line wave buffer) -> Wave sequencer ->
#     - Vector pipe: vector/matrix queues -> 4-bank VGPR file + 4-way operand cache -> VALU / TFU / WMMA -> VGPR/SGPR writeback
#     - Scalar pipe: SALU queue -> SGPR pair-latched file -> SALU -> SGPR writeback
#     - LDS pipe: per-SIMD request queue -> shared burst-arbitrated port -> 32-bank array -> return bus
from __future__ import annotations
import functools
from collections import Counter, deque
from enum import Enum, auto
from typing import NamedTuple
from tinygrad.helpers import dedup
from tinygrad.renderer.amd.dsl import Reg, OPERANDS, EXEC_LO, VCC_LO, SCC
from tinygrad.runtime.autogen.amd.common import Fmt
from tinygrad.runtime.autogen.amd.rdna3.ins import (DS, FLAT, GLOBAL, SCRATCH, SMEM, SOP1, SOP2, SOPC, SOPK, SOPP, VINTERP,
                                                    VOP3, VOP3P, VOP3SD, VOPD, VOP1_DPP16, VOP2_DPP16)
from tinygrad.runtime.autogen.amd.rdna3.enum import DSOp, SOPPOp, SOPKOp, SOP2Op, VOP1Op, VOP3Op, VOP3POp, VOP3SDOp, VOPCOp
from tinygrad.runtime.autogen.amd.rdna3.str_pcode import PCODE
from test.mockgpu.amd.emu import VOPD_TO_VOP2

# Front-end / instruction cache (bytes & shader cycles)
LINE, BUF_LINES, BLOCK = 64, 4, 256
FETCH_START, FILL, DELIVER, ICACHE_HIT = 765, 624, 3, 16
BRANCH_TAKEN, BAR_RELEASE, WAITCNT_SEE, SEQ = 10, 11, 1, 3
SLEEP_TICK, TICK_PHASE = 64, 21

# Vector side (depths & shader cycles)
VQ_DEPTH, MATRIX_DEPTH = 13, 8
VGPR_READ, NO_VGPR, EXEC_READ, VALU_LAT, VGPR_WB, ALU_STAGES = 6, 5, 1, 5, 4, 4
VGPR_BANKS, OPERAND_CACHE_WAYS, VGPR_MISS = 4, 4, 1
XLANE, WMMA_READ, WMMA_ALU = 2, 3, 34
TRANS_PORT, TRANS_LAT = 4, 10
F64_PORT, F64_ALU, F64_LAT = 32, 30, 9

# Scalar side
SALU_DEPTH, SGPR_WRITE_DEPTH = 7, 11
SALU_READ, SALU_LAT, SGPR_WB = 1, 2, 3
SGPR_READ, SGPR_MISS = 5, 2

# Shared LDS & off-CU flat latencies
LDS_REQ_DEPTH, LDS_BURST, LDS_BANKS = 9, 3, 32
LDS_SEND, LDS_LAT, DS_HOLD, FLOAT_RMW = 2, 25, 5, 3
SMEM_LAT, VMEM_LAT = 740, 700

class Unit(Enum):
  VALU = auto(); TRANS = auto(); WMMA = auto(); SALU = auto(); LDS = auto(); SMEM = auto(); VMEM = auto()  # noqa: E702
  BRANCH = auto(); CALL = auto(); BARRIER = auto(); NOP = auto(); WAIT = auto(); SLEEP = auto(); MSG = auto(); OTHER = auto()  # noqa: E702

VECTOR = (Unit.VALU, Unit.TRANS, Unit.WMMA)
EXEC, VCC, NEVER = EXEC_LO.offset, VCC_LO.offset, 1 << 62

SOPP_UNIT = {SOPPOp.S_NOP: Unit.NOP, SOPPOp.S_SLEEP: Unit.SLEEP, SOPPOp.S_BARRIER: Unit.BARRIER,
             SOPPOp.S_WAITCNT: Unit.WAIT, SOPPOp.S_WAITCNT_DEPCTR: Unit.WAIT, SOPPOp.S_SENDMSG: Unit.MSG,
             **{op: Unit.BRANCH for op in SOPPOp if "BRANCH" in op.name}}
WAITCNTS = {SOPKOp.S_WAITCNT_VSCNT, SOPKOp.S_WAITCNT_VMCNT, SOPKOp.S_WAITCNT_EXPCNT, SOPKOp.S_WAITCNT_LGKMCNT}
TRANS_OPS = {getattr(E, f"V_{f}_{t}_{e}") for E, e in ((VOP1Op, "E32"), (VOP3Op, "E64")) for t in ("F32", "F16")
             for f in ("EXP", "LOG", "RCP", "RSQ", "SQRT", "SIN", "COS")} | {VOP1Op.V_RCP_IFLAG_F32_E32, VOP3Op.V_RCP_IFLAG_F32_E64}
LANE_READS = {VOP1Op.V_READFIRSTLANE_B32_E32, VOP3Op.V_READFIRSTLANE_B32_E64, VOP3Op.V_READLANE_B32}
VALUB4_OPS = {VOP3Op.V_MUL_LO_U32, VOP3Op.V_MUL_HI_U32, VOP3Op.V_MUL_HI_I32, VOP3SDOp.V_MAD_U64_U32, VOP3SDOp.V_MAD_I64_I32}
VALUB2_NAMES = {"V_LSHLREV_B64", "V_LSHRREV_B64", "V_ASHRREV_I64"}
WMMAS = {op for op in VOP3POp if op.name.startswith("V_WMMA")}
SMULS = {SOP2Op.S_MUL_I32, SOP2Op.S_MUL_HI_U32, SOP2Op.S_MUL_HI_I32}
DS_FLOAT = {op for op in DSOp if op.name.endswith("_F32")}
PERMUTES = {DSOp.DS_PERMUTE_B32, DSOp.DS_BPERMUTE_B32}
DPP = (VOP1_DPP16, VOP2_DPP16)

FLAG_READS = {SCC.offset: ("CMOV", "CSELECT", "ADDC", "SUBB", "CBRANCH_SCC"),
              VCC: ("CNDMASK_B32_E32", "CO_CI_U32_E32", "CBRANCH_VCC"), EXEC: ("CBRANCH_EXEC",)}
FLAG_WRITES = {VCC: ("CO_CI_U32_E32",), EXEC: ("CMPX", "SAVEEXEC", "WREXEC")}

def classify_unit(inst, op) -> Unit:
  if isinstance(inst, SOPP): return SOPP_UNIT.get(op, Unit.OTHER)
  if isinstance(inst, SOPK):
    if op in WAITCNTS: return Unit.WAIT
    if op is SOPKOp.S_CALL_B64: return Unit.CALL
    return Unit.SALU
  if isinstance(inst, (SOP1, SOP2, SOPC)): return Unit.SALU
  if isinstance(inst, DS): return Unit.LDS
  if isinstance(inst, SMEM): return Unit.SMEM
  if isinstance(inst, (GLOBAL, FLAT, SCRATCH)): return Unit.VMEM
  if isinstance(inst, VINTERP): return Unit.VALU
  if op in TRANS_OPS: return Unit.TRANS
  return Unit.WMMA if op in WMMAS else Unit.VALU

class PipeCost(NamedTuple):
  vector: int; valu: int; alu: int; slow: int = 0; slow_unit: int = 0; lat: int = VALU_LAT  # noqa: E702

def pipe_cost(inst, u: Unit, f64: bool) -> PipeCost:
  if u is Unit.TRANS: return PipeCost(vector=1, valu=0, slow=TRANS_PORT, alu=1, slow_unit=TRANS_PORT, lat=TRANS_LAT)
  if u is Unit.WMMA:
    dur = WMMA_ALU // 2 if getattr(inst, "op_name", "").endswith("IU4") else WMMA_ALU
    return PipeCost(vector=1, valu=0, slow=1, alu=dur, slow_unit=1)
  if f64: return PipeCost(vector=F64_PORT, valu=1, slow=F64_PORT, alu=F64_ALU, slow_unit=F64_PORT, lat=F64_LAT)
  if isinstance(inst, DPP) and inst.dpp >= 0x100: return PipeCost(vector=1, valu=XLANE, alu=XLANE)
  op = getattr(inst, "op", None)
  if op in VALUB4_OPS: return PipeCost(vector=4, valu=4, alu=4)
  if getattr(op, "name", "") in VALUB2_NAMES: return PipeCost(vector=2, valu=2, alu=2)
  return PipeCost(vector=1, valu=1, alu=1)

def decode_delay_alu(d: int) -> tuple[Unit, int, int]:
  if d <= 4: return Unit.VALU, d, VALU_LAT
  if d <= 7: return Unit.TRANS, d - 4, TRANS_LAT
  return Unit.SALU, 1, max(1, d - 8)

@functools.cache
def implicit_flags(op) -> tuple[list[int], list[int]]:
  rd = [r for r, names in FLAG_READS.items() if any(n in op.name for n in names)]
  wr = [r for r, names in FLAG_WRITES.items() if any(n in op.name for n in names)]
  if "SCC = " in PCODE.get(op, ""): wr.append(SCC.offset)
  if isinstance(op, VOPCOp) and EXEC not in wr: wr.append(VCC)
  return rd, wr

def reg_list(inst, names) -> list[int]:
  rs = [x for f in names if isinstance(x := getattr(inst, f, None), Reg) and not 128 <= x.offset < 256]
  return dedup(r for x in rs for r in range(x.offset, x.offset + x.sz))

def sgprs(rs: list[int]) -> list[int]: return [r for r in rs if r < EXEC]
def vgprs(rs: list[int]) -> list[int]: return [r for r in rs if r >= 256]

def operand_fields(inst, opcodes) -> tuple[list[str], list[str]]:
  if isinstance(inst, VOPD):
    rd: list[str] = []
    for half, op in zip("xy", opcodes):
      rd.append(f"src{half}0")
      if "vsrc1" in OPERANDS[op]: rd.append(f"vsrc{half}1")
      if "FMAC" in op.name or "DOT2ACC" in op.name: rd.append(f"vdst{half}")
    return rd, ["vdstx", "vdsty"]
  if isinstance(inst, SOPK):
    opn = inst.op.name
    if "CMPK" in opn or "SETREG" in opn: return ["sdst"], []
    if any(k in opn for k in ("ADDK", "MULK", "CMOVK")): return ["sdst"], ["sdst"]
    return [], ["sdst"] if "sdst" in inst.operands else []
  wr = [f for f in inst.operands if f in ("vdst", "sdst") or (f == "sdata" and isinstance(inst, SMEM) and "STORE" not in inst.op_name)]
  rd = [f for f in inst.operands if f not in wr] + (["vsrc0"] if isinstance(inst, DPP) else [])
  opn = getattr(inst, "op_name", "")
  if "vdst" in wr and any(k in opn for k in ("FMAC", "DOT2ACC")): rd.append("vdst")
  return rd, wr

class MicroOp:
  w: Wave
  sent = False
  def __init__(self, pc: int, inst, exec_mask: int, addrs: list[int]):
    self.pc, self.inst, self.exec_mask, self.addrs = pc, inst, exec_mask, addrs
    self.lines = range(pc // LINE, (pc + inst.size() - 1) // LINE + 1)
    opcodes = [VOPD_TO_VOP2[inst.opx], VOPD_TO_VOP2[inst.opy]] if isinstance(inst, VOPD) else [inst.op]
    self.unit = classify_unit(inst, opcodes[0])
    self.f64 = self.unit is Unit.VALU and any(f is Fmt.FMT_NUM_F64 for f, *_ in OPERANDS.get(opcodes[0], {}).values())
    if self.unit in VECTOR: self.cost = pipe_cost(inst, self.unit, self.f64)
    rd, wr = operand_fields(inst, opcodes)
    reads = reg_list(inst, rd) + [r for op in opcodes for r in implicit_flags(op)[0]]
    self.unused = [r for r in sgprs(reg_list(inst, ("src0", "src1", "src2"))) if r not in reads] if isinstance(inst, (VOP3, VOP3SD, VOP3P)) else []
    self.reads = dedup(reads + self.unused)
    self.writes = dedup(reg_list(inst, wr) + [r for op in opcodes for r in implicit_flags(op)[1]])
    self.xlane = opcodes[0] in LANE_READS or isinstance(inst, DPP)
    self.t = self.retire = self.out = self.back = self.stamp = NEVER
    self.srcs: dict[int, MicroOp] = {}
    self.overwrites: dict[int, MicroOp] = {}

  @property
  def written(self) -> int: return self.out + (SGPR_WB if self.unit is Unit.SALU else VGPR_WB)
  def seen(self, by: MicroOp, c: int) -> bool: return c >= (self.out if (self.unit is Unit.SALU) == (by.unit is Unit.SALU) else self.written)

class DelayPipe:
  def __init__(self): self.q: deque[tuple[MicroOp, int]] = deque()
  def __iter__(self): return (x for x, _ in self.q)
  def push(self, x: MicroOp, at: int): self.q.append((x, at))
  def head(self, c: int) -> MicroOp | None: return self.q[0][0] if self.q and c >= self.q[0][1] else None
  def pop(self) -> MicroOp: return self.q.popleft()[0]

class BoundedQueue(deque):
  def __init__(self, depth: int):
    super().__init__()
    self.depth = depth
  def full(self) -> bool: return len(self) >= self.depth

class Port:
  x: MicroOp | None = None
  def __init__(self, until: int = 0): self.until = until
  def take(self, until: int, x: MicroOp | None = None): self.until, self.x = until, x
  def busy(self, c: int) -> bool: return c < self.until
  def done(self, c: int) -> MicroOp | None:
    if c < self.until: return None
    x, self.x = self.x, None
    return x

class ICache:
  def __init__(self):
    self.fill, self.deliver = Port(FETCH_START), Port()
    self.have: set[int] = set()
    self.wants: dict[int, list[Wave]] = {}
    self.landed: deque[tuple[Wave, int]] = deque()
    self.hits: deque[tuple[Wave, int]] = deque()

  def request(self, w: Wave, line: int, c: int):
    if (blk := line * LINE // BLOCK) in self.have: self.hits.append((w, line))
    elif w not in (ws := self.wants.setdefault(blk, [])): ws.append(w)

  def _serve_hits(self, c: int):
    while self.hits and not self.fill.busy(c):
      w, line = self.hits.popleft()
      w.buf[line] = c + ICACHE_HIT

  def step(self, c: int):
    if c == self.fill.until and self.wants:
      self.have.add(blk := next(iter(self.wants)))
      self.landed.extend((w, blk) for w in self.wants.pop(blk))
    if self.wants and not self.fill.busy(c): self.fill.take(c + FILL)
    if self.landed and not self.deliver.busy(c):
      w, blk = self.landed.popleft()
      for line, t in w.buf.items():
        if t == NEVER and line * LINE // BLOCK == blk: w.buf[line] = c
      self.deliver.take(c + DELIVER)
    self._serve_hits(c)

class Wave:
  def __init__(self, n: int, raw_ops: list[MicroOp], simd: SIMDUnit, sgpr_file: SGPRFile, cu: RDNA3CU):
    self.n, self.simd, self.sgpr_file, self.cu, self.i = n, simd, sgpr_file, cu, 0
    self.ops: list[MicroOp] = []
    self.delays: dict[int, list[tuple[Unit, int, int]]] = {}
    for x in raw_ops:
      x.w = self
      if isinstance(x.inst, SOPP) and x.inst.op in (SOPPOp.S_DELAY_ALU,):
        for d, skip in ((x.inst.simm16 & 0xf, 0), ((x.inst.simm16 >> 7) & 0xf, (x.inst.simm16 >> 4) & 7)):
          if d: self.delays.setdefault(len(self.ops) + skip, []).append(decode_delay_alu(d))
      else:
        self.ops.append(x)
    self.buf: dict[int, int] = {}
    self.fetch: int | None = None
    self.resume = 0
    self.seq, self.barriers, self.ds_hold = Port(), 0, 0
    self.lgkmcnt: list[MicroOp] = []
    self.vmcnt: list[MicroOp] = []
    self.recent: dict[Unit, deque[MicroOp]] = {u: deque(maxlen=4) for u in (Unit.VALU, Unit.TRANS, Unit.SALU)}
    self.writer: dict[int, MicroOp] = {}
    self.exec_changed = True

  def unwritten(self, c: int) -> int: return len({p for p in self.writer.values() if p.unit in VECTOR and c < p.written})
  def end(self) -> int: return max([self.n + 1] + [t for x in self.ops for t in (x.t, x.stamp, x.back) if t != NEVER])

  def refill(self, first: int, c: int):
    old, self.buf = self.buf, {}
    for line in range(first, first + BUF_LINES):
      self.buf[line] = old.get(line, NEVER)
      if line not in old: self.cu.icache.request(self, line, c)

  def redirect(self, c: int):
    if self.fetch is None or c < self.resume: return
    first = self.fetch // LINE
    if all(c >= t for line, t in self.buf.items() if line % BUF_LINES == first % BUF_LINES):
      self.refill(first, c)
      self.fetch = None

  def _waitcnt_busy(self, x: MicroOp, c: int) -> bool:
    op, n, lgkm, vm = x.inst.op, x.inst.simm16, 0x3f, 0x3f
    if op is SOPPOp.S_WAITCNT: lgkm, vm = (n >> 4) & 0x3f, (n >> 10) & 0x3f
    elif op is SOPKOp.S_WAITCNT_LGKMCNT: lgkm = n & 0x3f
    elif op is SOPKOp.S_WAITCNT_VMCNT: vm = n & 0x3f
    if len(self.lgkmcnt) > lgkm or len(self.vmcnt) > vm: return True
    return op is SOPPOp.S_WAITCNT_DEPCTR and self.unwritten(c) > (n >> 12)

  def _sends_to_pipe(self, x: MicroOp) -> bool:
    if x.unit in VECTOR: return x.exec_mask != 0
    if x.unit is Unit.LDS: return bool(x.exec_mask or self.lgkmcnt)
    return x.unit in (Unit.SALU, Unit.CALL, Unit.SMEM, Unit.VMEM, Unit.BARRIER)

  def _retire_at(self, x: MicroOp, c: int) -> int:
    c = max(c, self.seq.until)
    if x.unit is Unit.NOP: c += x.inst.simm16 & 0xf
    elif x.unit is Unit.SLEEP and (ticks := x.inst.simm16 & 0x7f): c += (TICK_PHASE - c) % SLEEP_TICK + SLEEP_TICK * (ticks - 1)
    return c

  def _stalled(self, x: MicroOp, c: int) -> bool:
    if c < self.resume or any(c < self.buf.get(line, NEVER) for line in x.lines): return True
    if self.cu.barrier.released < self.barriers: return True
    for u, k, lat in self.delays.get(self.i, ()):
      if len(q := self.recent[u]) >= k and c - q[-k].t < lat: return True
    if x.unit is Unit.WAIT and self._waitcnt_busy(x, c): return True
    if (ex := self.writer.get(EXEC)) and c < ex.written and (x.unit is Unit.LDS or x.unit in VECTOR): return True
    if x.unit in VECTOR and (c < self.ds_hold or any(p.w is self for p in self.simd.lds_requests)): return True
    if x.unit in (Unit.BRANCH, Unit.SMEM) and any(c < self.writer[r].written for r in x.reads if r in self.writer): return True
    if x.unit is Unit.VMEM and self.unwritten(c): return True
    return False

  def _issue(self, x: MicroOp, c: int):
    x.t = c
    self.i += 1
    after = x.pc + x.inst.size()
    nxt = self.ops[self.i].pc if self.i < len(self.ops) else after
    if x.unit in (Unit.BRANCH, Unit.CALL): self.resume = c + (BRANCH_TAKEN if nxt != after else SEQ)
    if x.unit in (Unit.BRANCH, Unit.CALL) and not (0 <= nxt // LINE - x.pc // LINE < BUF_LINES): self.fetch = nxt
    elif nxt // LINE != x.pc // LINE: self.refill(nxt // LINE, c)
    if x.unit in self.recent: self.recent[x.unit].append(x)
    if x.sent: self.seq.take(c + SEQ)

    if x.unit is Unit.BARRIER:
      self.barriers += 1
      self.cu.barrier.arrived += 1
    elif x.unit is Unit.SMEM:
      x.stamp = max(c + 4, self.simd.salu.alu.until)
      self.simd.salu.alu.take(x.stamp + 1)
      if x.writes:
        x.back = c + SMEM_LAT
        self.lgkmcnt.append(x)
    elif x.unit is Unit.VMEM:
      x.stamp = max(c + 3, self.cu.vmem_port.until)
      self.cu.vmem_port.take(x.stamp + 1)
      if x.writes:
        x.back = c + VMEM_LAT
        self.vmcnt.append(x)
    elif x.sent and x.unit is Unit.LDS:
      self.lgkmcnt.append(x)
      self.simd.lds_issue(x)
    elif x.sent:
      x.srcs = {r: self.writer[r] for r in x.reads if r in self.writer}
      x.overwrites = {r: self.writer[r] for r in x.writes if r in self.writer}
      self.writer.update({r: x for r in x.writes})
      if x.unit in (Unit.SALU, Unit.CALL): self.simd.salu.issue(x)
      else: self.simd.issue(x)

    if x.unit in VECTOR: self.exec_changed = False
    elif EXEC in x.writes: self.exec_changed = True

  def step(self, c: int):
    self.lgkmcnt = [x for x in self.lgkmcnt if c < x.back + WAITCNT_SEE]
    self.vmcnt = [x for x in self.vmcnt if c < x.back + WAITCNT_SEE]
    self.redirect(c)
    if self.i >= len(self.ops): return
    x = self.ops[self.i]
    if self._stalled(x, c): return
    if x.retire == NEVER: x.sent, x.retire = self._sends_to_pipe(x), self._retire_at(x, c)
    if not x.sent:
      if c < x.retire: return
    elif x.unit in (Unit.SALU, Unit.CALL):
      if self.simd.salu.busy(x, c): return
    elif x.unit is Unit.LDS:
      if self.simd.lds_busy(x, c): return
    elif x.unit in VECTOR:
      if self.simd.busy(x, c): return
    self._issue(x, c)

class SIMDUnit:
  def __init__(self):
    self.vector_issue, self.valu_issue, self.slow_issue, self.ds_issue = Port(), Port(), Port(), Port()
    self.queue, self.matrix_queue, self.lds_requests = BoundedQueue(VQ_DEPTH), BoundedQueue(MATRIX_DEPTH), BoundedQueue(LDS_REQ_DEPTH)
    self.vgpr_reads, self.ready, self.alu, self.slow_unit, self.salu = DelayPipe(), DelayPipe(), Port(), Port(), SALUUnit()
    self.sgpr_latch: dict[int, SGPRFile] = {}
    self.operand_cache: list[deque[tuple[int, int]]] = [deque(maxlen=OPERAND_CACHE_WAYS) for _ in range(VGPR_BANKS)]

  def busy(self, x: MicroOp, c: int) -> bool:
    if self.vector_issue.busy(c) or (x.cost.valu and self.valu_issue.busy(c)) or (x.cost.slow and self.slow_issue.busy(c)): return True
    if self.queue.full() or self.matrix_queue.full(): return True
    return bool(sgprs(x.writes) and x.w.sgpr_file.writes.full())

  def issue(self, x: MicroOp):
    self.vector_issue.take(x.t + x.cost.vector)
    if x.cost.valu: self.valu_issue.take(x.t + x.cost.valu)
    if x.cost.slow: self.slow_issue.take(x.t + x.cost.slow)
    self.queue.append(x)
    if x.unit is Unit.WMMA or self.matrix_queue: self.matrix_queue.append(x)
    self.vgpr_reads.push(x, x.t + (VGPR_READ if vgprs(x.reads) else NO_VGPR) + (EXEC_READ if x.w.exec_changed else 0))
    if sgprs(x.reads): x.w.sgpr_file.reads.push(x, x.t + SGPR_READ)
    if sgprs(x.writes): x.w.sgpr_file.writes.append(x)

  def lds_busy(self, x: MicroOp, c: int) -> bool:
    return self.ds_issue.busy(c) or self.lds_requests.full() or bool(x.w.unwritten(c))

  def lds_issue(self, x: MicroOp):
    self.ds_issue.take(x.t + 1)
    self.lds_requests.append(x)

  def _read_vgprs(self, c: int):
    if (x := self.vgpr_reads.head(c)) is None or x in x.w.sgpr_file.reads: return
    late = XLANE if x.xlane else 0
    banks: list[int] = []
    for r in vgprs(x.reads):
      lru, tag = self.operand_cache[r % VGPR_BANKS], (x.w.n, r)
      late = max(late, banks.count(r % VGPR_BANKS) + (0 if tag in lru else VGPR_MISS))
      if tag in lru: lru.remove(tag)
      lru.append(tag)
      banks.append(r % VGPR_BANKS)
    self.ready.push(self.vgpr_reads.pop(), c + 1 + (WMMA_READ if x.unit is Unit.WMMA else late))

  def _to_alu(self, c: int):
    if (x := self.ready.head(c)) is None or self.alu.busy(c) or (x.cost.slow_unit and self.slow_unit.busy(c)): return
    done = c + x.cost.alu - 1
    if not all(p.seen(x, c) for p in x.srcs.values()): return
    if any(p.out >= done + x.cost.lat for p in x.overwrites.values()): return
    self.ready.pop()
    self.alu.take(done + 1)
    if x.cost.slow_unit: self.slow_unit.take(c + x.cost.slow_unit)
    x.stamp, x.out = done, done + x.cost.lat

  def evict(self, w: Wave, r: int | None = None):
    for lru in self.operand_cache:
      for tag in [t for t in lru if t[0] == w.n and r in (None, t[1])]: lru.remove(tag)

  def step(self, c: int):
    if self.matrix_queue and c > self.matrix_queue[0].stamp:
      self.matrix_queue.popleft()
      while self.matrix_queue and self.matrix_queue[0].unit is not Unit.WMMA: self.matrix_queue.popleft()
    if self.queue and c >= self.queue[0].stamp + ALU_STAGES: self.queue.popleft()
    self._to_alu(c)
    self._read_vgprs(c)

class SGPRFile:
  def __init__(self, latch: dict[int, SGPRFile]):
    self.reads, self.port, self.writes, self.latch = DelayPipe(), Port(), BoundedQueue(SGPR_WRITE_DEPTH), latch
    self.stale: set[int] = set()

  def reading(self, c: int) -> list[MicroOp]: return [*self.reads, self.port.x] if self.port.busy(c) and self.port.x else [*self.reads]

  def step(self, c: int):
    while self.writes and c >= self.writes[0].written: self.writes.popleft()
    if (x := self.reads.head(c)) is None or self.port.busy(c): return
    miss = 0
    for r in sgprs(x.reads):
      if r in x.srcs and c < x.srcs[r].written: continue
      pair = r >> 1
      if r not in x.unused and (self.latch.get(pair) is not self or r in self.stale):
        miss = SGPR_MISS
        self.stale -= {2 * pair, 2 * pair + 1}
      self.latch[pair] = self
      self.stale.discard(r)
    self.port.take(c + 1 + miss, self.reads.pop())

class SALUUnit:
  def __init__(self): self.issue_port, self.queue, self.alu = Port(), BoundedQueue(SALU_DEPTH), Port()

  def busy(self, x: MicroOp, c: int) -> bool:
    if self.issue_port.busy(c) or self.queue.full() or x.w.sgpr_file.writes: return True
    return any(p.unit in VECTOR for p in x.w.sgpr_file.reading(c + SALU_READ))

  def issue(self, x: MicroOp):
    self.issue_port.take(x.t + 1)
    self.queue.append(x)
    x.w.sgpr_file.reads.push(x, x.t + SALU_READ)
    if EXEC in x.writes: x.w.simd.evict(x.w)

  def step(self, c: int):
    if not self.queue or self.alu.busy(c): return
    x = self.queue[0]
    if x in x.w.sgpr_file.reading(c) or not all(p.seen(x, c) for p in x.srcs.values()): return
    self.queue.popleft()
    done = c + 1 if getattr(x.inst, "op", None) in SMULS else c
    self.alu.take(done + 1)
    x.stamp, x.out = done, done + SALU_LAT
    x.w.sgpr_file.stale.update(sgprs(x.writes))

def _lds_bank_cycles(x: MicroOp) -> int:
  op = x.inst.op
  if op in PERMUTES: return 1 + XLANE
  if op is DSOp.DS_SWIZZLE_B32 or not x.addrs: return 1
  off = 0 if "2ADDR" in op.name else (getattr(x.inst, "offset1", 0) << 8) | getattr(x.inst, "offset0", 0)
  dwords = [((x.addrs[lane] + off) >> 2) for lane in range(min(32, len(x.addrs))) if (x.exec_mask >> lane) & 1]
  if not dwords: return 1
  if op in DS_FLOAT: return FLOAT_RMW * len(set(dwords))
  return max(Counter(d % LDS_BANKS for d in set(dwords)).values(), default=1)

class LDSUnit:
  def __init__(self, simds: list[SIMDUnit]):
    self.simds, self.port, self.bank_queue, self.banks, self.back = simds, Port(), DelayPipe(), Port(), DelayPipe()
    self.turn, self.burst, self.idle = simds[0], 0, True

  def step(self, c: int):
    if x := self.back.head(c):
      self.back.pop()
      for r in vgprs(x.writes): x.w.simd.evict(x.w, r)
    if x := self.banks.done(c):
      x.back = c + LDS_LAT
      self.back.push(x, x.back)
    if (x := self.bank_queue.head(c)) and not self.banks.busy(c):
      self.banks.take(c + _lds_bank_cycles(x), self.bank_queue.pop())
    if x := self.port.done(c):
      self.bank_queue.push(x, c + LDS_SEND + (len(x.reads) > 1))
      x.w.simd.lds_requests.popleft()
    want = [sm for sm in self.simds if sm.lds_requests]
    if want and not self.port.busy(c):
      if self.idle: sm = want[-1]
      elif self.turn in want and (self.burst < LDS_BURST or len(want) == 1): sm = self.turn
      else: sm = next(sm for sm in want if sm is not self.turn)
      if self.idle or sm is not self.turn: self.turn, self.burst = sm, 0
      x = sm.lds_requests[0]
      n = max(1, len(x.reads))
      self.burst += n
      self.port.take(c + n, x)
      x.w.ds_hold = c + n + DS_HOLD
      x.stamp = c + LDS_SEND
    self.idle = not self.port.busy(c)

class BarrierUnit:
  def __init__(self, n_waves: int):
    self.n_waves, self.arrived, self.count, self.released, self.release_at = max(1, n_waves), 0, 0, 0, -1
    self.release_cycles: list[int] = []

  def step(self, c: int):
    if c == self.release_at:
      self.released += 1
      self.release_cycles.append(c)
    if self.arrived:
      self.arrived -= 1
      self.count = (self.count + 1) % self.n_waves
      if self.count == 0: self.release_at = c + BAR_RELEASE

class RDNA3CU:
  def __init__(self, wave_ops: list[list[MicroOp]], simd_ids: list[int]):
    self.simds = [SIMDUnit(), SIMDUnit()]
    self.lds, self.barrier, self.icache, self.vmem_port = LDSUnit(self.simds), BarrierUnit(len(wave_ops)), ICache(), Port()
    self.waves = [Wave(k, xs, self.simds[s], SGPRFile(self.simds[s].sgpr_latch), self) for k, (xs, s) in enumerate(zip(wave_ops, simd_ids))]
    for w in self.waves:
      if w.ops: w.refill(w.ops[0].pc // LINE, 0)

  def run(self, max_cycles: int = 5_000_000):
    c = 0
    while c < max_cycles and (any(w.i < len(w.ops) for w in self.waves) or
                              any(sm.queue or sm.salu.queue or sm.lds_requests for sm in self.simds)):
      self.lds.step(c)
      for sm in self.simds: sm.step(c)
      for w in self.waves: w.sgpr_file.step(c)
      for sm in self.simds: sm.salu.step(c)
      self.barrier.step(c)
      self.icache.step(c)
      for w in self.waves: w.step(c)
      c += 1
