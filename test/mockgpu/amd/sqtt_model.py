# timing of one gfx1100 compute unit running wave32 code: two SIMDs sharing an LDS and an instruction cache
# fetch -> the wave's 4-line buffer -> sequencer, which retires an op itself or issues it down one of
#   VGPR read -> ALU -> VGPR file. it holds a vector queue entry until it's out of the ALU, and a matrix queue entry if it's a WMMA or behind one
#   SALU queue -> SGPR read -> SALU -> SGPR file, vector ops that write SGPRs wait in a queue for the file too
#   the SIMD's LDS requests -> the LDS port, one for both SIMDs -> banks
# each cycle every stage moves its oldest op if there's room ahead, then each wave issues its next instruction
# past the ALU it's fixed latency, kept as times on the op
from __future__ import annotations
import functools
from collections import deque, Counter
from enum import Enum, auto
from typing import NamedTuple
from tinygrad.helpers import dedup
from tinygrad.renderer.amd.dsl import Reg, OPERANDS, EXEC_LO, VCC_LO, SCC
from tinygrad.runtime.autogen.amd.common import Fmt
from tinygrad.runtime.autogen.amd.rdna3.ins import DS, FLAT, GLOBAL, SCRATCH, SMEM, SOP1, SOP2, SOPC, SOPK, SOPP, VOP3, VOP3P, VOP3SD, VOPD, \
  VOP1_DPP16, VOP2_DPP16
from tinygrad.runtime.autogen.amd.rdna3.enum import DSOp, SOPPOp, SOPKOp, SOP2Op, VOP1Op, VOP3Op, VOP3POp, VOPCOp
from tinygrad.runtime.autogen.amd.rdna3.str_pcode import PCODE
from test.mockgpu.amd.emu import VOPD_TO_VOP2

# measured from SQTT on a 7900 XTX, cycles unless it's a depth or a size. the structure is what fits the traces, not known
# fetch: a miss fills a block, one at a time, then the waves waiting get their lines DELIVER apart
LINE, BUF_LINES, BLOCK = 64, 4, 256
FILL, DELIVER, ICACHE_HIT = 624, 3, 16

BRANCH, BAR_RELEASE, WAITCNT_SEE = 10, 11, 1
SEQ = 3  # the sequencer is busy this long after it issues to a unit
SLEEP_TICK, TICK_PHASE = 64, 21  # s_sleep counts ticks

# vector side
VQ_DEPTH, MATRIX_DEPTH = 13, 8
VGPR_READ, NO_VGPR, EXEC_READ, VALU_LAT, VGPR_WB = 6, 5, 1, 5, 4  # NO_VGPR is an op with no VGPR to read
ALU_STAGES = 4  # ALU done to queue entry free, a guess that fits
VGPR_BANKS, OPERAND_CACHE_WAYS, VGPR_MISS = 4, 4, 1
XLANE, WMMA_READ, WMMA_ALU = 2, 3, 34
TRANS_PORT, TRANS_LAT = 4, 10
F64_PORT, F64_ALU, F64_LAT = 32, 30, 9

# scalar side, same stages
SALU_DEPTH = 7  # could be 8
SGPR_WRITE_DEPTH = 11
SALU_READ, SALU_LAT, SGPR_WB = 1, 2, 3
SGPR_READ, SGPR_MISS = 5, 2

# LDS
LDS_REQ_DEPTH, LDS_BURST, LDS_BANKS = 9, 3, 32
LDS_SEND, LDS_LAT, DS_HOLD, FLOAT_RMW = 2, 25, 5, 3

FETCH_START, SMEM_LAT, VMEM_LAT = 765, 740, 700  # nothing below the CU is modeled, these are flat

class Unit(Enum):
  VALU = auto(); TRANS = auto(); WMMA = auto(); SALU = auto(); LDS = auto(); SMEM = auto(); VMEM = auto() # noqa: E702
  BRANCH = auto(); BARRIER = auto(); NOP = auto(); WAIT = auto(); SLEEP = auto(); OTHER = auto() # noqa: E702
VECTOR = (Unit.VALU, Unit.TRANS, Unit.WMMA)
EXEC, VCC, NEVER = EXEC_LO.offset, VCC_LO.offset, 1 << 62

SOPP_UNIT = {SOPPOp.S_NOP: Unit.NOP, SOPPOp.S_SLEEP: Unit.SLEEP, SOPPOp.S_BARRIER: Unit.BARRIER, SOPPOp.S_WAITCNT: Unit.WAIT,
             SOPPOp.S_WAITCNT_DEPCTR: Unit.WAIT, **{op: Unit.BRANCH for op in SOPPOp if "BRANCH" in op.name}}
WAITCNTS = {SOPKOp.S_WAITCNT_VSCNT, SOPKOp.S_WAITCNT_VMCNT, SOPKOp.S_WAITCNT_EXPCNT, SOPKOp.S_WAITCNT_LGKMCNT}
TRANS_OPS = {getattr(E, f"V_{f}_{t}_{e}") for E, e in ((VOP1Op, "E32"), (VOP3Op, "E64")) for t in ("F32", "F16")
             for f in ("EXP", "LOG", "RCP", "RSQ", "SQRT", "SIN", "COS")} | {VOP1Op.V_RCP_IFLAG_F32_E32, VOP3Op.V_RCP_IFLAG_F32_E64}
LANE_READS = {VOP1Op.V_READFIRSTLANE_B32_E32, VOP3Op.V_READFIRSTLANE_B32_E64, VOP3Op.V_READLANE_B32}
WMMAS = {op for op in VOP3POp if op.name.startswith("V_WMMA")}
SMULS = {SOP2Op.S_MUL_I32, SOP2Op.S_MUL_HI_U32, SOP2Op.S_MUL_HI_I32}
DS_FLOAT = {op for op in DSOp if op.name.endswith("_F32")}
PERMUTES = {DSOp.DS_PERMUTE_B32, DSOp.DS_BPERMUTE_B32}
# ops that read or write a flag they don't name as an operand. a VOP3 names its carry or mask, so E32 only
FLAG_READS = {SCC.offset: ("CMOV", "CSELECT", "ADDC", "SUBB", "CBRANCH_SCC"), VCC: ("CNDMASK_B32_E32", "CO_CI_U32_E32", "CBRANCH_VCC"),
              EXEC: ("CBRANCH_EXEC",)}
FLAG_WRITES = {VCC: ("CO_CI_U32_E32",), EXEC: ("CMPX", "SAVEEXEC", "WREXEC")}
DPP = (VOP1_DPP16, VOP2_DPP16)

def unit(inst, op) -> Unit:
  if isinstance(inst, SOPP): return SOPP_UNIT.get(op, Unit.OTHER)
  if isinstance(inst, SOPK) and op in WAITCNTS: return Unit.WAIT
  if isinstance(inst, (SOP1, SOP2, SOPC, SOPK)): return Unit.SALU
  if isinstance(inst, DS): return Unit.LDS
  if isinstance(inst, SMEM): return Unit.SMEM
  if isinstance(inst, (GLOBAL, FLAT, SCRATCH)): return Unit.VMEM
  if op in TRANS_OPS: return Unit.TRANS
  return Unit.WMMA if op in WMMAS else Unit.VALU

# cycles each issue port (vector, valu, slow) and unit (alu, slow_unit) is held, 0 is not held. lat is ALU to forwarding
class Cost(NamedTuple): vector: int; valu: int; alu: int; slow: int = 0; slow_unit: int = 0; lat: int = VALU_LAT # noqa: E702
def cost(inst, u:Unit, f64:bool) -> Cost:
  if u is Unit.TRANS: return Cost(vector=1, valu=0, slow=TRANS_PORT, alu=1, slow_unit=TRANS_PORT, lat=TRANS_LAT)
  if u is Unit.WMMA: return Cost(vector=1, valu=0, slow=1, alu=WMMA_ALU, slow_unit=1)
  if f64: return Cost(vector=F64_PORT, valu=1, slow=F64_PORT, alu=F64_ALU, slow_unit=F64_PORT, lat=F64_LAT)
  return Cost(vector=1, valu=XLANE, alu=XLANE) if isinstance(inst, DPP) and inst.dpp >= 0x100 else Cost(vector=1, valu=1, alu=1)

# 1-4 VALU_DEP_n, 5-7 TRANS32_DEP_n, 9-11 SALU_CYCLE_n: the nth last op of a unit and the cycles to leave after its issue
def delay_alu(d:int) -> tuple[Unit, int, int]:
  if d <= 4: return Unit.VALU, d, VALU_LAT
  if d <= 7: return Unit.TRANS, d - 4, TRANS_LAT
  return Unit.SALU, 1, d - 8

@functools.cache
def flags(op) -> tuple[list[int], list[int]]:
  rd = [r for r, names in FLAG_READS.items() if any(n in op.name for n in names)]
  wr = [r for r, names in FLAG_WRITES.items() if any(n in op.name for n in names)]
  if "SCC = " in PCODE.get(op, ""): wr.append(SCC.offset)
  if isinstance(op, VOPCOp) and EXEC not in wr: wr.append(VCC)
  return rd, wr

def regs(inst, names) -> list[int]:
  rs = [x for f in names if isinstance(x := getattr(inst, f, None), Reg) and not 128 <= x.offset < 256]
  return dedup(r for x in rs for r in range(x.offset, x.offset + x.sz))
def sgprs(rs:list[int]) -> list[int]: return [r for r in rs if r < EXEC]
def vgprs(rs:list[int]) -> list[int]: return [r for r in rs if r >= 256]

def fields(inst, opcodes) -> tuple[list[str], list[str]]:
  if isinstance(inst, VOPD):
    rd = []
    for half, op in zip("xy", opcodes):
      rd.append(f"src{half}0")
      if "vsrc1" in OPERANDS[op]: rd.append(f"vsrc{half}1")
    return rd, ["vdstx", "vdsty"]
  wr = [f for f in inst.operands if f in ("vdst", "sdst") or f == "sdata" and isinstance(inst, SMEM)]
  return [f for f in inst.operands if f not in wr] + (["vsrc0"] if isinstance(inst, DPP) else []), wr

class Op:
  w: Wave
  waits: Counter
  sent = False  # to a unit, not retired by the sequencer
  def __init__(self, pc:int, inst, exec_mask:int, addrs):
    self.pc, self.inst, self.exec_mask, self.addrs = pc, inst, exec_mask, addrs
    self.lines = range(pc // LINE, (pc + inst.size() - 1) // LINE + 1)
    opcodes = [VOPD_TO_VOP2[inst.opx], VOPD_TO_VOP2[inst.opy]] if isinstance(inst, VOPD) else [inst.op]
    self.name = "+".join(op.name.lower() for op in opcodes)
    self.unit = unit(inst, opcodes[0])
    self.f64 = self.unit is Unit.VALU and any(f is Fmt.FMT_NUM_F64 for f, *_ in OPERANDS.get(opcodes[0], {}).values())
    if self.unit in VECTOR: self.cost = cost(inst, self.unit, self.f64)
    rd, wr = fields(inst, opcodes)
    reads = regs(inst, rd) + [r for op in opcodes for r in flags(op)[0]]
    # a VOP3 reads every SGPR in its source fields, one it doesn't use is never a miss
    self.unused: list[int] = []
    if isinstance(inst, (VOP3, VOP3SD, VOP3P)): self.unused = [r for r in sgprs(regs(inst, ("src0", "src1", "src2"))) if r not in reads]
    self.reads = dedup(reads + self.unused)
    self.writes = dedup(regs(inst, wr) + [r for op in opcodes for r in flags(op)[1]])
    self.xlane = opcodes[0] in LANE_READS or isinstance(inst, DPP)
    self.t = self.retire = self.out = self.back = NEVER
    self.stamp = NEVER  # in the ALU or sent to the LDS, the cycle SQTT stamps
    self.srcs: dict[int, Op] = {}
    self.overwrites: dict[int, Op] = {}
  @property
  def written(self) -> int: return self.out + (SGPR_WB if self.unit is Unit.SALU else VGPR_WB)
  # results forward inside the vector side and inside the SALU, everything else reads the file
  def seen(self, by:Op, c:int) -> bool: return c >= (self.out if (self.unit is Unit.SALU) == (by.unit is Unit.SALU) else self.written)

# delay line, never full
class Pipe:
  def __init__(self): self.q: deque[tuple[Op, int]] = deque()
  def __iter__(self): return (x for x, _ in self.q)
  def push(self, x:Op, at:int): self.q.append((x, at))
  def head(self, c:int): return self.q[0][0] if self.q and c >= self.q[0][1] else None
  def pop(self): return self.q.popleft()[0]

class Queue(deque):
  def __init__(self, depth:int):
    super().__init__()
    self.depth = depth
  def full(self): return len(self) >= self.depth

class Port:
  x: Op|None = None
  def __init__(self, until=0): self.until = until
  def take(self, until:int, x:Op|None=None): self.until, self.x = until, x
  def busy(self, c:int): return c < self.until
  def done(self, c:int):
    if c < self.until: return None
    x, self.x = self.x, None
    return x

class ICache:
  def __init__(self):
    self.fill, self.deliver = Port(FETCH_START), Port()
    self.have: set[int] = set()
    self.wants: dict[int, list[Wave]] = {}  # the fill queue, block -> waves waiting
    self.landed: deque[tuple[Wave, int]] = deque()
    self.hits: deque[tuple[Wave, int]] = deque()  # a hit waits for the fill in progress

  def request(self, w:Wave, line:int, c:int):
    if (blk := line * LINE // BLOCK) in self.have: self.hits.append((w, line))
    elif w not in (ws := self.wants.setdefault(blk, [])): ws.append(w)
    self.hit(c)

  def hit(self, c:int):
    while self.hits and not self.fill.busy(c):
      w, line = self.hits.popleft()
      w.buf[line] = c + ICACHE_HIT

  def step(self, c:int):
    if c == self.fill.until:
      self.have.add(blk := next(iter(self.wants)))
      self.landed.extend((w, blk) for w in self.wants.pop(blk))
    if self.wants and not self.fill.busy(c): self.fill.take(c + FILL)
    if self.landed and not self.deliver.busy(c):
      w, blk = self.landed.popleft()
      for line, t in w.buf.items():
        if t == NEVER and line * LINE // BLOCK == blk: w.buf[line] = c
      self.deliver.take(c + DELIVER)
    self.hit(c)

class Wave:
  def __init__(self, n:int, ops:list[Op], simd:SIMD, sgpr_file:SGPRFile, cu:CU):
    self.n, self.simd, self.sgpr_file, self.cu, self.i = n, simd, sgpr_file, cu, 0
    self.ops: list[Op] = []
    self.delays: dict[int, list[tuple[Unit, int, int]]] = {}
    # s_delay_alu takes no cycle
    for x in ops:
      x.w = self
      if not (isinstance(x.inst, SOPP) and x.inst.op is SOPPOp.S_DELAY_ALU): self.ops.append(x)
      else:
        for d, skip in ((x.inst.simm16 & 0xf, 0), (x.inst.simm16 >> 7 & 0xf, x.inst.simm16 >> 4 & 7)):
          if d: self.delays.setdefault(len(self.ops) + skip, []).append(delay_alu(d))
    self.waits: Counter = Counter()
    self.buf: dict[int, int] = {}  # line -> cycle it's in the buffer
    self.fetch: int|None = None
    self.resume = 0  # after a branch
    self.seq, self.barriers, self.ds_hold = Port(), 0, 0
    self.lgkmcnt: list[Op] = []
    self.vmcnt: list[Op] = []
    self.recent: dict[Unit, deque[Op]] = {u: deque(maxlen=4) for u in (Unit.VALU, Unit.TRANS, Unit.SALU)}
    self.writer: dict[int, Op] = {}
    self.exec_changed = True  # the first vector op after the SALU wrote EXEC reads it again, so does the wave's first

  def unwritten(self, c:int) -> int: return len({p for p in self.writer.values() if p.unit in VECTOR and c < p.written})
  def end(self) -> int: return max(t for x in self.ops for t in (x.t, x.stamp, x.back) if t != NEVER)

  # TODO: all 4 lines land together here. in rand4w_2600 and ext1w_2003 hw waits another 300-400 cycles for the third, don't know why
  def refill(self, first:int, c:int):
    old, self.buf = self.buf, {}
    for line in range(first, first + BUF_LINES):
      self.buf[line] = old.get(line, NEVER)
      if line not in old: self.cu.icache.request(self, line, c)

  # the target of a branch out of the buffer is fetched once the line in its slot has arrived
  def redirect(self, c:int):
    if self.fetch is None or c < self.resume: return
    first = self.fetch // LINE
    if all(c >= t for line, t in self.buf.items() if line % BUF_LINES == first % BUF_LINES):
      self.refill(first, c)
      self.fetch = None

  def waitcnt(self, x:Op, c:int) -> str|None:
    op, n, lgkm, vm = x.inst.op, x.inst.simm16, 0x3f, 0x3f
    if op is SOPPOp.S_WAITCNT: lgkm, vm = n >> 4 & 0x3f, n >> 10 & 0x3f
    elif op is SOPKOp.S_WAITCNT_LGKMCNT: lgkm = n & 0x3f
    elif op is SOPKOp.S_WAITCNT_VMCNT: vm = n & 0x3f
    if len(self.lgkmcnt) > lgkm: return "lgkmcnt"
    if len(self.vmcnt) > vm: return "vmcnt"
    if op is SOPPOp.S_WAITCNT_DEPCTR and self.unwritten(c) > n >> 12: return "depctr"
    return None

  def sends(self, x:Op) -> bool:
    if x.unit in VECTOR: return x.exec_mask != 0
    # a DS op still goes while lgkmcnt is up (rand1w_1809)
    if x.unit is Unit.LDS: return bool(x.exec_mask or self.lgkmcnt)
    return x.unit in (Unit.SALU, Unit.BARRIER)

  def retire_at(self, x:Op, c:int) -> int:
    c = max(c, self.seq.until)
    if x.unit is Unit.NOP: c += x.inst.simm16 & 0xf
    if x.unit is Unit.SLEEP and (ticks := x.inst.simm16 & 0x7f): c += (TICK_PHASE - c) % SLEEP_TICK + SLEEP_TICK * (ticks - 1)
    return c

  def wait(self, x:Op, c:int) -> str|None:
    if c < self.resume: return "branch"
    if any(c < self.buf.get(line, NEVER) for line in x.lines): return "icache"
    if self.cu.barrier.released < self.barriers: return "barrier"
    for u, k, lat in self.delays.get(self.i, ()):
      if len(q := self.recent[u]) >= k and c - q[-k].t < lat: return "s_delay_alu"
    if x.unit is Unit.WAIT and (why := self.waitcnt(x, c)): return why
    if (ex := self.writer.get(EXEC)) and c < ex.written and (x.unit is Unit.LDS or x.unit in VECTOR): return "exec"
    # a vector op can't overwrite VGPRs a DS op of its wave hasn't read yet
    if x.unit in VECTOR and (c < self.ds_hold or any(p.w is self for p in self.simd.lds_requests)): return "DS operands"
    if x.unit is Unit.BRANCH and any(c < self.writer[r].written for r in x.reads if r in self.writer): return "branch operand"
    if x.unit is Unit.VMEM and self.unwritten(c): return "VGPR file"
    return None

  def issue(self, x:Op, c:int):
    x.t, x.waits, self.waits = c, self.waits, Counter()
    self.i += 1
    after = x.pc + x.inst.size()
    nxt = self.ops[self.i].pc if self.i < len(self.ops) else after
    if x.unit is Unit.BRANCH: self.resume = c + (BRANCH if nxt != after else SEQ)
    if x.unit is Unit.BRANCH and not 0 <= nxt // LINE - x.pc // LINE < BUF_LINES: self.fetch = nxt
    elif nxt // LINE != x.pc // LINE: self.refill(nxt // LINE, c)
    if x.unit in self.recent: self.recent[x.unit].append(x)
    if x.sent: self.seq.take(c + SEQ)
    if x.unit is Unit.BARRIER:
      self.barriers += 1
      self.cu.barrier.arrived += 1
    elif x.unit in (Unit.SMEM, Unit.VMEM) and x.writes:
      x.back = c + (SMEM_LAT if x.unit is Unit.SMEM else VMEM_LAT)
      (self.lgkmcnt if x.unit is Unit.SMEM else self.vmcnt).append(x)
    elif x.sent and x.unit is Unit.LDS:
      self.lgkmcnt.append(x)
      self.simd.lds_issue(x)
    elif x.sent:
      x.srcs = {r: self.writer[r] for r in x.reads if r in self.writer}
      x.overwrites = {r: self.writer[r] for r in x.writes if r in self.writer}
      self.writer.update({r: x for r in x.writes})
      if x.unit is Unit.SALU: self.simd.salu.issue(x)
      else: self.simd.issue(x)
    if x.unit in VECTOR: self.exec_changed = False
    elif EXEC in x.writes: self.exec_changed = True

  def step(self, c:int):
    self.lgkmcnt = [x for x in self.lgkmcnt if c < x.back + WAITCNT_SEE]
    self.vmcnt = [x for x in self.vmcnt if c < x.back + WAITCNT_SEE]
    self.redirect(c)
    if self.i == len(self.ops): return
    x = self.ops[self.i]
    if (why := self.wait(x, c)) is None:
      # the sequencer looks at an op once, when it's clear to go
      if x.retire == NEVER: x.sent, x.retire = self.sends(x), self.retire_at(x, c)
      if not x.sent: why = "sequencer" if c < x.retire else None
      elif x.unit is Unit.SALU: why = self.simd.salu.wait(x, c)
      elif x.unit is Unit.LDS: why = self.simd.lds_wait(x, c)
      elif x.unit in VECTOR: why = self.simd.wait(x, c)
    if why: self.waits[why] += 1
    else: self.issue(x, c)

class SIMD:
  def __init__(self):
    self.vector_issue, self.valu_issue, self.slow_issue, self.ds_issue = Port(), Port(), Port(), Port()
    self.queue, self.matrix_queue, self.lds_requests = Queue(VQ_DEPTH), Queue(MATRIX_DEPTH), Queue(LDS_REQ_DEPTH)
    self.vgpr_reads, self.ready, self.alu, self.slow_unit, self.salu = Pipe(), Pipe(), Port(), Port(), SALU()
    self.sgpr_latch: dict[int, SGPRFile] = {}
    self.operand_cache: list[deque[tuple]] = [deque(maxlen=OPERAND_CACHE_WAYS) for _ in range(VGPR_BANKS)]

  def wait(self, x:Op, c:int) -> str|None:
    if self.vector_issue.busy(c) or x.cost.valu and self.valu_issue.busy(c) or x.cost.slow and self.slow_issue.busy(c): return "vector issue"
    if self.queue.full(): return "vector queue"
    if self.matrix_queue.full(): return "matrix queue"
    if sgprs(x.writes) and x.w.sgpr_file.writes.full(): return "SGPR write queue"
    return None

  def issue(self, x:Op):
    self.vector_issue.take(x.t + x.cost.vector)
    if x.cost.valu: self.valu_issue.take(x.t + x.cost.valu)
    if x.cost.slow: self.slow_issue.take(x.t + x.cost.slow)
    self.queue.append(x)
    if x.unit is Unit.WMMA or self.matrix_queue: self.matrix_queue.append(x)
    self.vgpr_reads.push(x, x.t + (VGPR_READ if vgprs(x.reads) else NO_VGPR) + (EXEC_READ if x.w.exec_changed else 0))
    if sgprs(x.reads): x.w.sgpr_file.reads.push(x, x.t + SGPR_READ)
    if sgprs(x.writes): x.w.sgpr_file.writes.append(x)

  def lds_wait(self, x:Op, c:int) -> str|None:
    if self.ds_issue.busy(c): return "DS issue"
    if self.lds_requests.full(): return "LDS requests"
    if x.w.unwritten(c): return "VGPR file"
    return None

  def lds_issue(self, x:Op):
    self.ds_issue.take(x.t + 1)
    self.lds_requests.append(x)

  def read_vgprs(self, c:int):
    if (x := self.vgpr_reads.head(c)) is None or x in x.w.sgpr_file.reads: return
    late = XLANE if x.xlane else 0
    banks: list[int] = []
    for r in vgprs(x.reads):
      # a bank reads one VGPR a cycle and keeps the last operands it read
      lru, tag = self.operand_cache[r % VGPR_BANKS], (x.w.n, r)
      late = max(late, banks.count(r % VGPR_BANKS) + (0 if tag in lru else VGPR_MISS))
      if tag in lru: lru.remove(tag)
      lru.append(tag)
      banks.append(r % VGPR_BANKS)
    self.ready.push(self.vgpr_reads.pop(), c + 1 + (WMMA_READ if x.unit is Unit.WMMA else late))

  def to_alu(self, c:int):
    if (x := self.ready.head(c)) is None or self.alu.busy(c) or x.cost.slow_unit and self.slow_unit.busy(c): return
    done = c + x.cost.alu - 1
    if not all(p.seen(x, c) for p in x.srcs.values()): return
    if any(p.out >= done + x.cost.lat for p in x.overwrites.values()): return  # results to one register forward in order
    self.ready.pop()
    self.alu.take(done + 1)
    if x.cost.slow_unit: self.slow_unit.take(c + x.cost.slow_unit)
    x.stamp, x.out = done, done + x.cost.lat

  def evict(self, w:Wave, r:int|None=None):
    for lru in self.operand_cache:
      for tag in [t for t in lru if t[0] == w.n and r in (None, t[1])]: lru.remove(tag)

  def step(self, c:int):
    # back to front. a WMMA leaves the matrix queue when it's out of the ALU, with the ops behind it up to the next WMMA
    if self.matrix_queue and c > self.matrix_queue[0].stamp:
      self.matrix_queue.popleft()
      while self.matrix_queue and self.matrix_queue[0].unit is not Unit.WMMA: self.matrix_queue.popleft()
    if self.queue and c >= self.queue[0].stamp + ALU_STAGES: self.queue.popleft()
    self.to_alu(c)
    self.read_vgprs(c)

# a wave's SGPRs. its vector ops and its SALU ops read through one port, VALU results wait in writes to go in
class SGPRFile:
  def __init__(self, latch:dict[int, SGPRFile]):
    self.reads, self.port, self.writes, self.latch = Pipe(), Port(), Queue(SGPR_WRITE_DEPTH), latch
    self.stale: set[int] = set()

  # a vector op only waits to get the port, a SALU op waits for the read
  def reading(self, c:int): return [*self.reads, self.port.x] if self.port.busy(c) else [*self.reads]

  def step(self, c:int):
    while self.writes and c >= self.writes[0].written: self.writes.popleft()
    if (x := self.reads.head(c)) is None or self.port.busy(c): return
    # SGPRs are latched in pairs for one wave. a pair the other wave has, or one written since, is a miss
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

class SALU:
  def __init__(self): self.issue_port, self.queue, self.alu = Port(), Queue(SALU_DEPTH), Port()

  def wait(self, x:Op, c:int) -> str|None:
    if self.issue_port.busy(c): return "SALU issue"
    if self.queue.full(): return "SALU queue"
    # a SALU op waits for the wave's vector ops to read their SGPRs and for VALU writes in flight to land
    if any(p.unit in VECTOR for p in x.w.sgpr_file.reading(c + SALU_READ)): return "SGPR read"
    if x.w.sgpr_file.writes: return "SGPR writes"
    return None

  def issue(self, x:Op):
    self.issue_port.take(x.t + 1)
    self.queue.append(x)
    x.w.sgpr_file.reads.push(x, x.t + SALU_READ)
    if EXEC in x.writes: x.w.simd.evict(x.w)  # its cached operands are no good under a new EXEC

  def step(self, c:int):
    if not self.queue or self.alu.busy(c): return
    x = self.queue[0]
    if x in x.w.sgpr_file.reading(c) or not all(p.seen(x, c) for p in x.srcs.values()): return
    self.queue.popleft()
    done = c + 1 if x.inst.op in SMULS else c
    self.alu.take(done + 1)
    x.stamp, x.out = done, done + SALU_LAT
    x.w.sgpr_file.stale.update(sgprs(x.writes))

def bank_cycles(x:Op) -> int:
  if x.inst.op in PERMUTES: return 1 + XLANE
  if x.inst.op is DSOp.DS_SWIZZLE_B32: return 1
  dwords = {x.addrs[lane] >> 2 for lane in range(32) if x.exec_mask >> lane & 1}
  if x.inst.op in DS_FLOAT: return FLOAT_RMW * len(dwords)
  return max(Counter(d % LDS_BANKS for d in dwords).values(), default=1)

class LDS:
  def __init__(self, simds:list[SIMD]):
    self.simds, self.port, self.bank_queue, self.banks, self.back = simds, Port(), Pipe(), Port(), Pipe()
    self.turn, self.burst, self.idle = simds[0], 0, True

  def step(self, c:int):
    if x := self.back.head(c):
      self.back.pop()
      for r in vgprs(x.writes): x.w.simd.evict(x.w, r)
    if x := self.banks.done(c):
      x.back = c + LDS_LAT
      self.back.push(x, x.back)
    if (x := self.bank_queue.head(c)) and not self.banks.busy(c): self.banks.take(c + bank_cycles(x), self.bank_queue.pop())
    if x := self.port.done(c):
      self.bank_queue.push(x, c + LDS_SEND + (len(x.reads) > 1))  # a cycle more with data
      x.w.simd.lds_requests.popleft()
    want = [sm for sm in self.simds if sm.lds_requests]
    if want and not self.port.busy(c):
      # a SIMD keeps the port for LDS_BURST dwords, the last SIMD goes first after an idle cycle
      if self.idle: sm = want[-1]
      elif self.turn in want and (self.burst < LDS_BURST or len(want) == 1): sm = self.turn
      else: sm = next(sm for sm in want if sm is not self.turn)
      if self.idle or sm is not self.turn: self.turn, self.burst = sm, 0
      x = sm.lds_requests[0]
      self.burst += (n := len(x.reads))
      self.port.take(c + n, x)
      x.w.ds_hold = c + n + DS_HOLD
      x.stamp = c + LDS_SEND
    self.idle = not self.port.busy(c)

# arrivals are counted one a cycle, the waves go BAR_RELEASE after the last one
class Barrier:
  def __init__(self, waves:int): self.waves, self.arrived, self.count, self.released, self.release_at = waves, 0, 0, 0, -1
  def step(self, c:int):
    if c == self.release_at: self.released += 1
    if self.arrived:
      self.arrived -= 1
      self.count = (self.count + 1) % self.waves
      if self.count == 0: self.release_at = c + BAR_RELEASE

class CU:
  def __init__(self, ops:list[list[Op]], simd:list[int]):
    self.simds = [SIMD(), SIMD()]
    self.lds, self.barrier, self.icache = LDS(self.simds), Barrier(len(ops)), ICache()
    self.waves = [Wave(k, xs, sm, SGPRFile(sm.sgpr_latch), self) for k, (xs, sm) in enumerate(zip(ops, [self.simds[i] for i in simd]))]
    for w in self.waves: w.refill(w.ops[0].pc // LINE, 0)

  def run(self):
    c = 0
    while any(w.i < len(w.ops) for w in self.waves) or any(sm.queue or sm.salu.queue or sm.lds_requests for sm in self.simds):
      self.lds.step(c)
      for sm in self.simds: sm.step(c)
      for w in self.waves: w.sgpr_file.step(c)
      for sm in self.simds: sm.salu.step(c)
      self.barrier.step(c)
      self.icache.step(c)
      for w in self.waves: w.step(c)  # the older wave wins a port
      c += 1

def report(cu:CU):
  for w in cu.waves:
    for x in w.ops:
      print(f"{x.t:<6} w{w.n} {x.pc:#x} {x.name:32} " + ", ".join(f"{n} {why}" for why, n in x.waits.items()) +
            (f" | exec +{x.stamp - x.t}" if x.stamp != NEVER else "") + (f" | back +{x.back - x.t}" if x.back != NEVER else ""))
