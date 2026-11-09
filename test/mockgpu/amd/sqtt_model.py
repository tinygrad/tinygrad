# cycle-stepped timing model of one RDNA3 WGP (gfx1100, wave32) for SQTT traces of SRAM-only kernels. nothing is read from a
# recording: instruction fetch follows a law by fetch class and memory loads take a fixed latency
import re
from itertools import islice
from typing import NamedTuple
from tinygrad.renderer.amd.dsl import Reg
from tinygrad.runtime.autogen.amd.rdna3.ins import DS, FLAT, GLOBAL, SCRATCH, SMEM, SOP1, SOP2, SOPC, SOPK, SOPP, VOPC, VOPD
from tinygrad.runtime.autogen.amd.rdna3.enum import VOP1Op, VOP2Op, VOP3Op, VOP3SDOp
from tinygrad.runtime.autogen.amd.rdna3.str_pcode import PCODE

# times are SIMD clock cycles
EXEC_AFTER_ISSUE = 7
IMM_AFTER_ALU = 3        # an IMMEDIATE or idle op after an ALU op, 1 after anything else
VALU_LAT = 5
VQ_DEPTH = 13
VQ_SLOT_FREE = 4         # after its op left
VQ_CATCH = 4             # ops at the queue's tail an SALU EXEC write or a lane read can still move
VALU_OUT = 4             # a vector result reaches the SALU, LDS and branches this long after its write
SGPR_READ_QPOS = 10      # a VALU reads its SGPRs once the op this many ahead in the queue executed
VC_ENTRIES, VC_MISS = 4, 1  # VGPR operand cache per bank of 4
BYPASS = 2               # a value ready at most this long before the read is bypassed
TRANS_LAT, TRANS_II = 10, 4
WMMA_BUSY = 34
WMMA_QUEUE = 8           # a WMMA this many back in the queue holds issue until it leaves
WMMA_READ = 2            # sources read this long before exec
F64_BUSY, F64_LAT, F64_AFTER_TRANS = 32, 6, 2  # F64_LAT after the unit's busy cycles
XLANE = 2                # a DPP or lane op crosses lanes this long after its exec
SALU_LAT = 2
SALU_READ = 2
SALU_READ_TO_EXEC, SMUL_READ_TO_EXEC = 1, 6
SALU_READS_AHEAD, SALU_READ_LEAD = 3, -3  # an ALU op issues no earlier than LEAD after the read of the SALU op AHEAD back
SALU_EXECS_AHEAD, SALU_EXEC_LEAD = 7, -2  # nor LEAD after the exec of the SIMD's SALU op AHEAD back
SGPR_READ = 5            # a VALU's SGPR read after issue
SGPR_COLD = 2            # cold: written and unread, or another wave of the SIMD read the file since
SGPR_WQ, VCC_WQ = 11, 7  # VALU SGPR writes in flight, VCC ones of them
LDS_BURST = 3            # request port cycles a SIMD keeps while the other waits
LDS_REQS = 9             # DS ops in flight per SIMD
LDS_SLOT_BACK = 3        # a request slot comes back this long after the stretch it was served in
LDS_LAT = 29             # after the bank stage
LDS_STORE_LAND = 2       # a store's dwords are readable this long after its bank start
LDS_RET_DWORD = 4        # per extra dword on a return channel
VALU_AFTER_LDS = 7       # after a DS op's VGPR reads were granted
WAITCNT_SEE = 1
BAR_RELEASE, BAR_II = 12, 7  # after the last arrival; a SIMD's arrivals apart
BR_IN_BUFFER = 10
BR_REFETCH = 27
BR_FETCH_GAP, BR_FETCH_SIMD_GAP = 9, 2  # between two waves' branch target fetches on a SIMD, and across SIMDs
FETCH_AHEAD = 14         # a line refills this long after the wave's first op in the line before it
BLOCK, LINE, ICL = 256, 64, 128  # bytes: the buffer, its lines, an icache fetch
# when an icache miss's line reaches the buffer
FETCH_START = 766        # a workgroup's first line, after its first wave's dispatch
FETCH_START_SKEW = {2: (0, 4), 4: (0, 6, 4, 12)}  # by dispatch order, 4 apart for other sizes
FETCH_NEXT_BLOCK = 666   # a block's first line, after the workgroup first issued in the first line of the block before it
FETCH_SECOND_LINE = 60   # a block's second line, after the workgroup first issued in its first
FETCH_REFILL = 680       # a branch back this soon after the wave first issued in the target line resumes then
FETCH_MISS = 630         # a later miss waits this long
# on hardware a VMEM load takes 90 cycles for a cache hit to 1470 for a first touch
SMEM_LAT, VMEM_LAT = 740, 700
# exec packets: an SALU op's after its exec, a DS op's after its grant
SALU_EXEC_PACKET, LDS_EXEC_PACKET = -5, 3

FOUR_PASS = {VOP3Op.V_MUL_LO_U32, VOP3Op.V_MUL_HI_U32, VOP3Op.V_MUL_HI_I32, VOP3SDOp.V_MAD_U64_U32, VOP3SDOp.V_MAD_I64_I32}

EXEC, SCC, VCC = 126, 253, 106  # operand codes of EXEC_LO, SCC, VCC_LO
VALU, VEC, ALU = ("VALU", "TRANS"), ("VALU", "TRANS", "WMMA"), ("VALU", "TRANS", "WMMA", "SALU", "LDS")
RESULT_LAT = {"VALU": VALU_LAT, "TRANS": TRANS_LAT, "SALU": SALU_LAT, "WMMA": VALU_LAT}
REDIRECT = {"taken": "branch: taken", "not_taken": "branch: not taken", "fetch": "branch: refetch"}
SOPP_UNIT = {"S_NOP": "NOP", "S_SLEEP": "SLEEP", "S_BARRIER": "BARRIER", "S_CLAUSE": "CLAUSE", "S_WAITCNT": "WAIT", "S_WAITCNT_DEPCTR": "WAIT"}

class Step:  # deps: the s_delay_alu instids that apply to it
  def __init__(self, pc, inst, exec_mask=0xffffffff, lanes=None, deps=()):
    self.pc, self.inst, self.exec_mask, self.deps = pc, inst, exec_mask, list(deps)
    self.lds_degree, self.misaligned, self.dwords, self.bank_cycle = 1, False, None, None
    if lanes is not None: lds_lanes(self, lanes)

class Timeline(NamedTuple):
  issue: list[int]
  end: int
  kinds: list[str]
  execs: list[int|None]
  stalls: list[tuple[str, int]]  # the rule that set the issue cycle ("unit: rule") and the cycles past back-to-back issue

def op_name(inst) -> str: return f"{inst.opx.name}+{inst.opy.name}" if hasattr(inst, "opx") else inst.op_name

def unit(inst) -> str:
  name = op_name(inst)
  if isinstance(inst, SOPP): return "BRANCH" if name.startswith(("S_CBRANCH", "S_BRANCH")) else SOPP_UNIT.get(name, "IMM")
  if isinstance(inst, (SOP1, SOP2, SOPC, SOPK)): return "WAIT" if name.startswith("S_WAITCNT") else "SALU"
  if isinstance(inst, DS): return "LDS"
  if isinstance(inst, SMEM): return "SMEM"
  if isinstance(inst, (GLOBAL, FLAT, SCRATCH)): return "VMEM"
  return "TRANS" if re.match("V_(RCP|RSQ|SQRT|EXP|LOG|SIN|COS)_", name) else "WMMA" if name.startswith("V_WMMA") else "VALU"

def regs(inst, *names) -> set[int]:  # operand codes 128-255 are constants and special sources, not registers
  return {r for n in names if isinstance(x := getattr(inst, n, None), Reg) and not 128 <= x.offset < 256 for r in range(x.offset, x.offset + x.sz)}

def ds_words(inst) -> int: return int(m[1]) // 32 if (m := re.search(r"_B(32|64|96|128)$", inst.op_name)) else 1

def ds_accesses(inst) -> list[int]:  # a 2addr op's offsets are in units of the access
  return [o * ds_words(inst) * (64 if "STRIDE64" in inst.op_name else 1) for o in (inst.offset0, inst.offset1)] if "2ADDR" in inst.op_name else [0]

def bank_order(passes: list[list[int]]) -> tuple[dict[int, int], int]:
  # a pass takes a cycle per dword on its busiest bank, at least one
  out, deg = dict[int, int](), 0
  for dw in passes:
    n = [0] * 32
    for d in dict.fromkeys(dw):
      out.setdefault(d, deg + n[d % 32])
      n[d % 32] += 1
    deg += max(n + [1])
  return out, deg

def lds_lanes(s: Step, lanes: list[int]):
  inst, name, n = s.inst, op_name(s.inst), ds_words(s.inst)
  off = 0 if "2ADDR" in name else inst.offset1 << 8 | inst.offset0
  dw = {l: (lanes[l] + off) >> 2 for l in range(32) if s.exec_mask >> l & 1}
  # a b64/b128 takes n passes of 32/n lanes, each at least a cycle even with no lane active; a 2addr op's second address goes after
  s.bank_cycle, s.lds_degree = bank_order([[a + dw[l] + j for l in range(q, q + 32 // n) if l in dw for j in range(n)]
                                           for a in ds_accesses(inst) for q in range(0, 32, 32 // n)])
  s.dwords = frozenset(s.bank_cycle)
  if name.startswith(("DS_SWIZZLE", "DS_PERMUTE", "DS_BPERMUTE")): s.lds_degree, s.dwords, s.bank_cycle = 1, None, None  # lanes, not banks
  elif not name.startswith(("DS_LOAD", "DS_STORE")):  # atomic: lanes on one dword take 2 cycles each after the first
    banks = [[d for d in dw.values() if d % 32 == b] for b in range(32)]  # a float add 3 a lane, 4 if an earlier lane had its dword
    s.lds_degree = max(1, 4 * len(dw) - len(set(dw.values())) if name.endswith("_F32") else
                       max(len(set(x)) + 2 * (len(x) - len(set(x))) for x in banks))
  size = int(m[1]) // 8 if (m := re.search(r"_[BUI](8|16)$", name)) else 4 * n  # bytes per lane
  s.misaligned = name.startswith(("DS_LOAD", "DS_STORE")) and any((lanes[l] + off) % size for l in dw)

def pcode(inst) -> str:
  if not isinstance(inst, VOPD): return re.sub("//.*", "", PCODE.get(inst.op, ""))
  return "\n".join(re.sub("//.*", "", PCODE.get(e[x.name.replace("V_DUAL_", "V_") + "_E32"], "")) for x in (inst.opx, inst.opy)
                   for e in (VOP1Op, VOP2Op) if x.name.replace("V_DUAL_", "V_") + "_E32" in e.__members__)

def flags(inst) -> tuple[set[int], set[int]]:  # the SCC, VCC and EXEC reads and writes no operand field names
  code, reads, writes = pcode(inst), set(), set()
  lhs = {m[1] for m in re.finditer(r"^\s*(SCC|VCC|EXEC|D0)\b[^=\n]*=(?!=)", code, re.M)}
  rhs = {m[1] for m in re.finditer(r"\b(SCC|VCC|EXEC)Z?\b", re.sub(r"^\s*[\w.\[\]]+\s*=(?!=)", "", code, flags=re.M))}
  if "SCC" in rhs: reads.add(SCC)
  if "SCC" in lhs: writes.add(SCC)
  if "EXEC" in lhs: writes.add(EXEC)
  if "EXEC" in rhs and isinstance(inst, SOPP): reads.add(EXEC)  # vector ops read EXEC too, the EXEC write rules time that
  if "VOP3" not in type(inst).__name__:  # VOP3 names VCC in a field
    if "VCC" in rhs: reads.add(VCC)
    if "VCC" in lhs or "D0" in lhs and isinstance(inst, VOPC): writes.add(VCC)  # a VOPC op's D0 is VCC
  return reads, writes

def waitcnt(inst) -> tuple[int, int]:  # (lgkmcnt, vmcnt)
  v, name = inst.simm16, inst.op_name
  if name == "S_WAITCNT": return (v >> 4) & 0x3f, (v >> 10) & 0x3f
  return (v & 0x3f if name == "S_WAITCNT_LGKMCNT" else 0x3f), (v & 0x3f if name == "S_WAITCNT_VMCNT" else 0x3f)

class Op:
  def __init__(self, inst):
    self.unit = u = unit(inst)
    n, ty = op_name(inst), type(inst).__name__
    self.smul, self.writelane, self.readlane = n.startswith("S_MUL"), n.startswith("V_WRITELANE"), n.startswith(("V_READLANE", "V_READFIRSTLANE"))
    self.f64, self.iu4, self.e64, self.dpp = "F64" in n, n.endswith("IU4"), n.endswith("_E64"), "DPP" in ty
    self.row_dpp = self.dpp and inst.dpp >= 0x100
    self.s_branch, self.store, self.depctr = n == "S_BRANCH", "STORE" in n, n == "S_WAITCNT_DEPCTR"
    self.passes = 1 if u != "VALU" else 4 if getattr(inst, "op", None) in FOUR_PASS else 2 if n[-4:] in ("_B64", "_I64", "_U64") else 1
    self.simm = getattr(inst, "simm16", 0) if u in ("NOP", "SLEEP", "WAIT") else 0
    mac = re.compile("FMAC|DOT2ACC")
    if hasattr(inst, "opx"):  # a v_dual_mov half reads only src0
      fs = tuple(f for h, x in (("x", inst.opx.name), ("y", inst.opy.name)) for f in (f"src{h}0",) + (f"vsrc{h}1",) * ("MOV" not in x) +
                 (f"vdst{h}",) * bool(mac.search(x)))
    else: fs = ("src0", "vsrc0", "src1", "vsrc1", "src2") + ("vdst",) * bool(mac.search(n))  # a multiply-accumulate reads its destination
    self.vsrcs = [r for f in fs if isinstance(x := getattr(inst, f, None), Reg) and x.offset >= 256 for r in range(x.offset, x.offset + x.sz)]
    self.load = n.startswith(("DS_LOAD", "DS_SWIZZLE"))
    if u == "LDS":
      self.words, self.perm = ds_words(inst) * (1 + ("2ADDR" in n)), n.startswith(("DS_PERMUTE", "DS_BPERMUTE"))
      data = () if self.load else ("data0", "data1") if "2ADDR" in n else ("data0",)
      ss, ds = regs(inst, "addr", *data), regs(inst, "vdst") if self.load or self.perm or "_RTN" in n else set()
      self.lds_reads = [(r not in regs(inst, "addr"), r % 4) for r in ss if r >= 256]  # (is data, bank)
    else:
      reads, writes = flags(inst)
      ss, ds = regs(inst, *fs, "ssrc0", "ssrc1") | reads, regs(inst, "vdst", "sdst", "vdstx", "vdsty") | writes
    self.srcs, self.dsts = frozenset(ss), frozenset(ds)
    self.vdsts, self.vdst_banks = sorted(r for r in ds if r >= 256), [r % 4 for r in ds if r >= 256]
    self.sgpr_dsts = {r for r in ds if r < 128 and r not in (EXEC, EXEC + 1)}
    self.unused = regs(inst, *("src0", "src1", "src2")[inst.num_srcs():]) if "VOP3" in ty and inst.operands else set()
    self.src2 = sorted(regs(inst, "src2"))
    self.salu_srcs = [x.offset if isinstance(x := getattr(inst, f, None), Reg) else None for f in ("ssrc0", "ssrc1")]
    if u == "WMMA": self.wmma_c0, self.wmma_first = inst.src2.offset, {inst.src0.offset, inst.src1.offset, inst.src2.offset, inst.src2.offset + 1}

def fetch_classes(steps: list[Step]) -> list[str]:
  out, seen, entered, refetched = [], set(), set(), set()
  for i, s in enumerate(steps):
    blk = (s.pc + s.inst.size() - 1) // ICL  # an 8-byte instruction at 0x17c is in the next line
    refetch = s.pc in seen and blk not in refetched and unit(steps[i - 1].inst) == "BRANCH"
    out.append("wave_start" if i == 0 else "block_entry" if blk not in entered else "refetch" if refetch else "")
    refetched |= {blk} if out[-1] == "refetch" else set()
    seen.add(s.pc)
    entered.add(blk)
  return out

def nth(xs: list, n: int) -> list: return xs[-n:][:1] if len(xs) >= n else []

class Bound:
  def __init__(self, t: int, why: str): self.t, self.why = t, why
  def at(self, why: str, *ts: int):
    for t in ts:
      if t > self.t: self.t, self.why = t, why

class AluOp:
  def __init__(self, wave, unit, ex, written=0, leave=0, idx=0, passes=1, pinned=False, srcs=None):
    self.wave, self.unit, self.exec, self.idx, self.passes = wave, unit, ex, idx, passes
    self.written, self.leave, self.slot = written, leave, 0
    self.pinned = pinned
    self.srcs = srcs or {}
    self.ready: dict[int, int] = {}

def last_vec(ops: list[AluOp], n: int) -> list[AluOp]: return list(islice((f for f in reversed(ops) if f.unit != "SALU"), n))[::-1]

class LDSRequest:
  def __init__(self, step, op, issued, simd, operands, short, regfile):
    self.step, self.op, self.issued, self.simd, self.operands = step, op, issued, simd, operands
    self.short, self.regfile = short, regfile
    self.grant = self.done = self.stretch = None

class LDSPort:
  # the request port takes a DS op's VGPR reads one a cycle, for one SIMD at a time: the owner keeps it while its burst is under
  # LDS_BURST or the other SIMD has nothing waiting, an overrun lengthens the next burst, and a cycle with no grant ends it
  def __init__(self):
    self.pending: list[LDSRequest] = []
    self.free, self.owner, self.burst = 0, None, 0
    self.stretch, self.stretch_used = [0], 0  # the request slot stretch's end, shared with its requests
    self.sent = self.banks_free = self.data_free = self.xbar_free = 0
    self.stores = []  # (lands, dwords, bank cycles)
    self.returns = [0, 0]

  def pick(self, c) -> LDSRequest|None:
    heads: dict[int, LDSRequest] = {}
    for r in self.pending:
      if r.issued <= c and r.simd not in heads: heads[r.simd] = r
    if c < self.free or not heads: return None
    own = None if c > self.free else self.owner
    if own in heads and (self.burst < LDS_BURST or len(heads) == 1): return heads[own]
    if own is not None: return next(r for x, r in heads.items() if x != own)
    return heads[max(heads)]  # after a rest SIMD 2 goes first

  def vgpr_reads(self, r: LDSRequest, c: int) -> set:  # the address, then data dwords; SIMD 2 a cycle later
    return {(c + EXEC_AFTER_ISSUE + 1 + r.simd + d, b) for d, b in r.op.lds_reads}

  def step(self, c) -> bool:
    if (r := self.pick(c)) is None: return False
    fresh = c > self.free or self.owner is None
    if fresh or r.simd != self.owner:
      self.burst = (LDS_BURST - 1 if r.simd == 0 else int(r.short)) if fresh else -max(0, self.burst - LDS_BURST)
      self.stretch_used, self.owner = LDS_BURST, r.simd
    if self.stretch_used >= LDS_BURST: self.stretch, self.stretch_used = [0], 0
    self.pending.remove(r)
    o, st, cost = r.op, r.step, len(r.op.lds_reads)
    self.burst += cost
    self.stretch_used += cost
    self.free = self.stretch[0] = c + cost
    r.stretch = self.stretch
    r.regfile.lds_vgpr_reads |= self.vgpr_reads(r, c)
    self.sent = max(c, r.operands, self.sent) + (0 if o.load else cost)
    # a load reads a dword a store wrote once it landed
    bank = max([self.sent, self.banks_free] +
               [lt + max([so[d] - st.bank_cycle[d] for d in a & st.dwords] if so and st.bank_cycle else [0])
                for lt, a, so in self.stores if o.load and not (a and st.dwords and not a & st.dwords)])
    self.banks_free = done = bank + (1 if st.misaligned else st.lds_degree)
    if o.load:
      self.data_free = done = max(done, self.data_free) + o.words - 1
      ch = self.returns.index(min(self.returns))
      done = self.returns[ch] = max(done + LDS_LAT, self.returns[ch] + LDS_RET_DWORD * (o.words - 1))
    else:
      self.stores = [x for x in self.stores if x[0] > bank] + [(bank + LDS_STORE_LAND, st.dwords, st.bank_cycle)]
      if o.perm:  # crosses lanes after the banks one at a time, holding the bank stage
        done = max(done, self.xbar_free) + XLANE
        self.xbar_free, self.banks_free = done + 1, done
      done += LDS_LAT
    if st.misaligned: self.free, done = done + 1, done + 2 + LDS_LAT  # a second pass once the first is back
    r.grant, r.done = c, done
    if o.load: r.regfile.vgpr_write_port |= {(done + (r.simd == 0), b) for b in o.vdst_banks}  # SIMD 0 a cycle later
    return True

class Barrier:  # one arrival counted a cycle, oldest first
  def __init__(self, size): self.size, self.arrivals, self.count, self.release = size, [], {}, {}
  def step(self, c) -> bool:
    if not self.arrivals: return False
    i, simd = self.arrivals.pop(0)
    self.count[i], simd.bar_counted = self.count.get(i, 0) + 1, c
    if self.count[i] == self.size: self.release[i] = c + BAR_RELEASE
    return True

class SIMD:
  def __init__(self):
    self.vec_issue = self.salu_issue = self.lds_issue = self.trans_issue = 0
    self.bar_issue = self.bar_counted = -99
    self.f64_hold = self.dpp_hold = 0  # an f64 op holds vector issue, a row DPP op VALU issue
    self.exec_free = dict(VALU=0, TRANS=0, WMMA=0, SALU=0)
    self.vec_free = self.read_free = self.vop_done = self.last_vexec = 0
    self.f64_free = self.f64_valu_free = self.f64_trans_extra = 0
    self.vq: list[AluOp] = []
    self.vcache: dict[int, list] = {}  # bank -> LRU of (wave, VGPR[, "dead"])
    self.vgpr_write_port: set[tuple[int, int]] = set()
    self.lds_vgpr_reads: set[tuple[int, int]] = set()
    self.salu_execs: list[tuple[int, int]] = []
    self.salu_exec_chain = self.salu_start = 0
    self.lds_reqs: list[LDSRequest] = []

  def operand_read(self, w, k, o, t, x, lds) -> int:
    # a second VGPR in a bank reads a slot late, a cache miss VC_MISS late. a bypassed value still allocates, LDS data landing kills
    # the entry in place, and a miss waits out a DS read of its bank that cycle
    nxt = lds.pick(t)
    soon = lds.vgpr_reads(nxt, t) if nxt is not None and nxt.regfile is self else set()
    late, slip, seen = 0, 0, list[int]()
    for r in o.vsrcs:
      landed = [q for q in w.lds_landing.get(r, []) if q.done is not None and q.done + WAITCNT_SEE <= x - EXEC_AFTER_ISSUE + SGPR_READ]
      if landed: w.lds_landing[r] = [q for q in w.lds_landing[r] if q not in landed]
      b, tag = self.vcache.setdefault(r % 4, []), (k, r)
      miss = (bool(landed) or tag not in b) and w.ready.get(r, 0) < x - BYPASS
      if tag in b and landed: b[b.index(tag)] = (k, r, "dead")
      elif tag in b: b.remove(tag)
      b.append(tag)
      del b[:-VC_ENTRIES]
      slot, d = sum(q % 4 == r % 4 and q != r for q in seen), 0
      while miss and ((x + slot + d, r % 4) in self.lds_vgpr_reads or (x + slot + d, r % 4) in soon): d += 1
      late, slip = max(late, slot + VC_MISS * miss), max(slip, d)
      seen.append(r)
    return late + slip

class Wave:
  def __init__(self, steps, start):
    self.steps, self.start, self.idx, self.last = steps, start, 0, 0
    self.after, self.prev, self.redirect, self.bars = "", "", 0, 0
    self.zero_exec_real = False
    self.issue: list[int] = []
    self.kinds: list[str] = []
    self.execs: list = []
    self.stalls: list[tuple[str, int]] = []
    self.alu_ops: list[AluOp] = []
    self.lgkm: list = []
    self.vm: list[int] = []
    self.delay_alu = (list[int](), list[int]())  # VALU, TRANS
    self.last_lds: LDSRequest|None = None
    self.ready: dict[int, int] = {}
    self.written_exec: dict[int, int] = {}
    self.producer: dict[int, AluOp] = {}
    self.lds_landing: dict[int, list[LDSRequest]] = {}
    self.wmma_out: dict[int, int] = {}
    self.vq_slot = 0
    self.cold, self.warming, self.evicted, self.valu_sgprs = set(range(VCC)), {}, {}, set()
    self.sgpr_wq: list[tuple[int, bool]] = []
    self.valu_sgpr_read = self.sgpr_read_free = -99
    self.salu_read_port, self.salu_reads, self.salu_execs, self.salu_chain, self.salu_fwd = -99, [], [], -99, {}
    self.salu_start = self.salu_sgpr_write = 0
    self.exec_start, self.exec_cold, self.nobypass, self.exec_new = 0, True, {}, False
    self.valu_exec_mask = 0

class InstFetch:
  def __init__(self, waves: list[Wave]): self.waves = waves

  def entered(self, k: int, line: int, own=False) -> int|None:
    firsts = [next((t for s, t in zip(w.steps, w.issue) if (s.pc + s.inst.size() - 1) // ICL == line), None)
              for q, w in enumerate(self.waves) if q == k or not own]
    return min((t for t in firsts if t is not None), default=None)

  def arrival(self, k: int, cl: str) -> Bound|None:  # None: in the buffer
    w, n = self.waves[k], len(self.waves)
    s = w.steps[w.idx]
    line = (s.pc + s.inst.size() - 1) // ICL
    if cl == "wave_start":
      skew = FETCH_START_SKEW.get(n, tuple(4 * r for r in range(n)))[k]
      return Bound(min(q.start for q in self.waves) + FETCH_START + skew, "fetch: wave start")
    if cl == "block_entry":
      second = line % (BLOCK // ICL) == 1
      t = self.entered(k, line - 1 if second else line - BLOCK // ICL)
      lat = FETCH_SECOND_LINE if second and t is not None else FETCH_NEXT_BLOCK
      return Bound(max(w.last + 1, (w.last if t is None else t) + lat), "fetch: second line" if second else "fetch: next block")
    # a branch back stays in the buffer if the target is in the branch's buffer slot and that is an even line
    br = w.steps[w.idx - 1].pc // LINE
    if (br - s.pc // LINE) % (BLOCK // LINE) == 0 and br % 2 == 0: return None
    t = self.entered(k, line, own=True)
    return Bound(max(w.last + BR_REFETCH, t + FETCH_REFILL if t is not None and w.last < t + FETCH_REFILL else w.last + FETCH_MISS),
                 "fetch: branch back")

def runs(w: Wave, s: Step, o: Op) -> bool:  # a v_cmpx's EXEC is not seen until it lands; an SGPR write always runs
  if o.unit in VALU and (o.sgpr_dsts or EXEC in w.valu_sgprs and w.last + 1 < w.ready.get(EXEC, 0)): return bool(o.sgpr_dsts) or w.valu_exec_mask != 0
  return s.exec_mask != 0

def emulate(waves_steps: list[list[Step]], simds: list[int]|None=None) -> list[Timeline]:
  # one workgroup on one CU, waves starting a cycle apart
  n = len(waves_steps)
  sids = simds or [0] * n
  per = {sm: SIMD() for sm in set(sids)}
  sims, classes, bar = [per[sm] for sm in sids], [fetch_classes(st) for st in waves_steps], Barrier(n)
  ops = [[Op(s.inst) for s in st] for st in waves_steps]
  waves = [Wave(st, k) for k, st in enumerate(waves_steps)]
  fetch, lds = InstFetch(waves), LDSPort()
  fetch_simd, fetch_cu = {sm: (-99, -1) for sm in per}, (-99, -1)  # (cycle, wave)

  def after_last(k) -> Bound|None:  # None while it waits on a barrier
    w = waves[k]
    if w.after == "barrier" and n > 1:
      return None if (rel := bar.release.get(w.bars - 1)) is None else Bound(rel, "barrier: release")
    if w.after not in ("taken", "not_taken", "fetch"): return Bound(w.last + 1, "")
    b = Bound(w.last + w.redirect, REDIRECT[w.after])
    if w.after == "fetch":
      for (bt, bk), gap in ((fetch_simd[sids[k]], BR_FETCH_GAP), (fetch_cu, BR_FETCH_SIMD_GAP)):
        if bk != k: b.at("branch: another wave's fetch", bt + gap)
    return b

  def vgpr_out(w, keep=0) -> list[int]:
    wrs = [f.written + VALU_OUT for f in w.alu_ops[-VQ_DEPTH:] if f.unit != "SALU"]
    return wrs[:len(wrs) - keep]

  def ready(k) -> Bound|None:  # None while it waits on an event
    w, sm = waves[k], sims[k]
    s, o = w.steps[w.idx], ops[k][w.idx]
    u, ss, ds, real = o.unit, o.srcs, o.dsts, runs(w, s, o)
    b = (fetch.arrival(k, cl) if (cl := classes[k][w.idx]) else None) or after_last(k)
    if b is None: return None
    if u == "NOP" and w.after in ("barrier", "taken", "fetch"):
      b.at("s_nop", b.t + (o.simm & 0xf))
      return b
    for d in s.deps:  # VALU_DEP_n counts no TRANS ops
      if d <= 7: b.at("s_delay_alu", *[x + RESULT_LAT["TRANS" if d > 4 else "VALU"] for x in nth(w.delay_alu[d > 4], (d - 1) % 4 + 1)])
    if u in VEC and real:
      b.at("f64: issue hold", sm.f64_hold)
      if u == "VALU": b.at("VALU: row DPP issue hold", sm.dpp_hold)
    if u in VEC or u == "LDS": b.at("SALU: EXEC write", w.exec_start)
    if u in VALU and o.sgpr_dsts:  # SGPR writes go in order after the last SALU's
      b.at("SALU: SGPR write order", w.salu_sgpr_write if any(r < VCC for r in ds) else w.salu_start)
      b.at("SGPR file: write queue", *[c for c, _ in nth(w.sgpr_wq, SGPR_WQ)])
      if VCC in ds and not o.e64: b.at("SGPR file: VCC write queue", *nth([c for c, vcc in w.sgpr_wq if vcc], VCC_WQ))
    if u in ("LDS", "VMEM"): b.at("VALU: results out", *vgpr_out(w))
    if u == "SALU":
      b.at("SGPR file: VALU read", w.sgpr_read_free)
      if w.prev in VALU: b.at("vector queue: slot", w.vq_slot)
      b.at("SGPR file: VALU write", *[w.ready[r] + VALU_OUT for r in w.valu_sgprs if r not in (VCC, VCC + 1, EXEC, EXEC + 1) or r in ss | ds])
    if u in VEC and (q := w.last_lds) and q.op.lds_reads:
      if q.grant is None: return None
      b.at("LDS: VGPR reads", q.grant + VALU_AFTER_LDS + len(q.op.lds_reads) - 1)
    # with EXEC==0 a DS op is still real while lgkmcnt counts an LDS op, anything else vector issues as an IMMEDIATE
    w.zero_exec_real = u == "LDS" and not real and any(isinstance(x, LDSRequest) and (x.done is None or x.done + WAITCNT_SEE >= b.t) for x in w.lgkm)
    if (u in VEC or u == "LDS") and not real and not w.zero_exec_real: b.at("issue: IMMEDIATE", w.last + 1 + (IMM_AFTER_ALU - 1) * (w.prev in ALU))
    if u in VEC and real or u == "SALU":
      vq = sm.vq if u != "SALU" else last_vec(w.alu_ops, max(VQ_DEPTH, WMMA_QUEUE))
      b.at("vector queue: WMMA", *[f.leave for f in nth(vq, WMMA_QUEUE) if f.unit == "WMMA"])
      b.at("SALU: reads in flight", *[x + SALU_READ_LEAD for x in nth(w.salu_reads, SALU_READS_AHEAD)])
      # a vector op does not count an SALU op issued this cycle
      sx = sm.salu_execs[-SALU_EXECS_AHEAD - 1:]
      if u != "SALU" and sx and sx[-1][0] >= b.t: sx.pop()
      if u == "SALU": b.at("SALU: issue port", sm.salu_issue)
      else: b.at("VALU: issue port", sm.vec_issue)
      b.at("SALU: execs in flight", *[x + SALU_EXEC_LEAD for _, x in nth(sx, SALU_EXECS_AHEAD)])
      if u == "SALU": b.at("SALU: pipe", *[f.exec for f in nth(vq, VQ_DEPTH)])
      else: b.at("vector queue: slot", *[f.slot + VQ_SLOT_FREE for f in nth(vq, VQ_DEPTH)])
      if u in ("TRANS", "WMMA") or o.f64 or o.passes > 1: b.at("TRANS: port", sm.trans_issue)
    elif u == "LDS" and (real or w.zero_exec_real):
      reqs = nth(sm.lds_reqs, LDS_REQS)
      if any(r.grant is None for r in reqs): return None
      b.at("LDS: issue port", sm.lds_issue)
      b.at("LDS: request slots", *[r.stretch[0] + LDS_SLOT_BACK for r in reqs])
    if u in ("WAIT", "NOP", "CLAUSE", "SLEEP") or u == "BARRIER" and n == 1:
      idle = w.last + (o.simm & 0xf if u == "NOP" else 0) + 1 + (IMM_AFTER_ALU - 1) * (w.prev in ALU + ("ZDS", "VMEM"))
      b.at("s_nop" if u == "NOP" else "issue: IMMEDIATE", idle)
    if u == "WAIT":
      lg, vm = waitcnt(s.inst)
      lgkm = [x.done if isinstance(x, LDSRequest) else x for x in w.lgkm[:max(0, len(w.lgkm) - lg)]]
      if None in lgkm: return None
      b.at("s_waitcnt: lgkmcnt", *[d + WAITCNT_SEE for d in lgkm if d is not None])
      b.at("s_waitcnt: vmcnt", *[d + WAITCNT_SEE for d in w.vm[:max(0, len(w.vm) - vm)]])
      if o.depctr: b.at("VALU: results out", *vgpr_out(w, o.simm >> 12))  # depctr va_vdst(n) reads the file like a DS op
    elif u == "BRANCH":  # on SCC or VCC also the SALU drained
      if ss & {SCC, VCC} and w.salu_execs: b.at("SALU: drain", w.salu_execs[-1])
      b.at("branch: condition", *[w.ready[r] + VALU_OUT if r in w.valu_sgprs else w.written_exec.get(r, 0) for r in ss])
    elif u == "BARRIER":  # one not ready by the cycle after the SIMD's last waits BAR_II after that one counted
      b.at("barrier: port", sm.bar_issue + 1 if b.t <= sm.bar_issue + 1 else sm.bar_counted + BAR_II)
    return b

  def evict(k, rs, when):  # a read takes the SIMD's warm entry of those pairs from the other waves
    for o in (waves[j] for j in range(n) if j != k and sims[j] is sims[k]):
      for r in rs: o.evicted[r >> 1] = min(o.evicted.get(r >> 1, when), when)

  def thaw(w, rd):
    w.cold |= {r for q, c in w.evicted.items() if c <= rd for r in (2 * q, 2 * q + 1)}
    w.evicted = {q: c for q, c in w.evicted.items() if c > rd}

  def catch(k, t):
    # after an SALU EXEC write or a lane read, queued ops whose producer has not executed read from the file a cycle after it is ready
    sm, prev = sims[k], None
    for c in sm.vq[-VQ_CATCH:]:
      e = c.exec if prev is None else max(c.exec, prev.exec + prev.passes)
      for r, q in c.srcs.items():
        if c.wave == k and q.exec > t: e = max(e, q.ready[r] + 1)
      if not c.pinned and (d := e - c.exec) > 0:
        o, tr = waves[c.wave], c.unit == "TRANS"
        for r in c.ready:
          if o.ready.get(r) == c.ready[r]: o.ready[r], o.written_exec[r] = o.ready[r] + d, e
          c.ready[r] += d
        o.execs[c.idx], c.exec, c.written, c.leave, c.slot = e, e, c.written + d, c.leave + d, c.slot + d
        sm.exec_free[c.unit] = max(sm.exec_free[c.unit], e + (TRANS_II if tr else 1))
        sm.vec_free, sm.vop_done, sm.last_vexec = max(sm.vec_free, e + c.passes), max(sm.vop_done, e + 1 + tr), max(sm.last_vexec, e)
      prev = c

  def salu_read(w, o, t) -> int:  # one SGPR file read a cycle, in order
    start = max(t + SALU_READ, w.salu_read_port)
    reads = [r for r in set(o.salu_srcs) - {None} if r < EXEC and not (r in w.ready and start < w.ready[r])]  # forwarded values take none
    w.salu_read_port = start + max(1, len(reads))
    return start

  def salu(k, w, sm, o, t) -> int:
    ss, ds, mul = o.srcs, o.dsts, o.smul
    sm.salu_issue = t + 1
    r0, rd = salu_read(w, o, t), t + SGPR_READ
    thaw(w, rd)
    rdy = {r: w.ready[r] for r in ss if r < 128 and r in w.ready}
    file_reads = {r for r in ss if r < EXEC and (r not in rdy or rd >= rdy[r])}
    cold_reads = file_reads & w.cold
    cold = SGPR_COLD * bool(cold_reads)
    e = max([max(t + EXEC_AFTER_ISSUE, r0 + (SMUL_READ_TO_EXEC if mul else SALU_READ_TO_EXEC)) + cold, sm.exec_free["SALU"]] +
            [w.ready.get(r, 0) for r in ss])
    sm.exec_free["SALU"] = e + 1
    for r in ds: w.ready[r], w.written_exec[r] = e + SALU_LAT + mul, e
    w.salu_reads.append(r0)
    w.salu_execs.append(e)
    w.alu_ops.append(AluOp(k, "SALU", e, written=e + SALU_LAT, leave=e - 1, idx=len(w.execs)))
    # a cold second operand paired with a warm first takes a cycle more
    s0, s1 = o.salu_srcs
    pm = s0 is not None and s1 is not None and s1 in cold_reads and s0 not in cold_reads and s0 != s1 and s0 >> 1 == s1 >> 1
    lo = max([t + EXEC_AFTER_ISSUE + cold + pm, r0 + SALU_READ_TO_EXEC] +
             [w.salu_fwd[r] if r in w.salu_fwd else rdy[r] for r in ss if r in w.salu_fwd or r in rdy])
    se, so = max(lo, sm.salu_exec_chain), max(lo, w.salu_chain)
    start = max(t + EXEC_AFTER_ISSUE + cold, sm.salu_start, min(e - mul, se))
    sm.salu_exec_chain, w.salu_chain, sm.salu_start, w.salu_start = se + 1 + mul, so + 1 + mul, start + 1 + mul, start + mul
    sm.salu_execs.append((t, se + mul))
    w.salu_fwd.update({r: se + SALU_LAT + mul for r in ds})
    any_cold = bool(cold_reads) or any(r in w.cold for r in ss if r < EXEC)
    w.salu_sgpr_write = max(t + EXEC_AFTER_ISSUE, w.salu_sgpr_write) + SGPR_COLD * any_cold + mul
    if EXEC in ds:  # the operand cache goes cold and results in flight lose the bypass
      w.exec_cold, w.exec_start, w.exec_new = True, max(start, so), True
      w.nobypass = {r: x for r, x in w.ready.items() if r >= 256 and x + VALU_OUT > start}
      for b in sm.vcache.values(): b[:] = [tg for tg in b if tg[0] != k]
    w.valu_sgprs -= ds
    if file_reads:  # a read warms its aligned pair partner while that still holds its wave-start value
      pair = {r ^ 1 for r in cold_reads if (r ^ 1) in w.cold and ((r ^ 1) not in w.ready or r < VCC and w.ready[r ^ 1] <= rd)}
      w.cold -= cold_reads | pair
      w.warming.update({r: rd + SGPR_COLD for r in cold_reads | pair})
      evict(k, file_reads, rd + SGPR_COLD)
    w.cold |= {r for r in ds if r < VCC}
    for r in ds: w.warming.pop(r, None)
    return se + mul + SALU_EXEC_PACKET

  def vector(k, w, sm, s, o, t) -> int|None:
    u, ss, ds, lane, dpx, n = o.unit, o.srcs, o.dsts, o.readlane, o.dpp, o.passes
    dp = u in VALU and o.f64
    sm.vec_issue = t + n
    if u == "TRANS": sm.trans_issue = t + TRANS_II
    if o.row_dpp: sm.dpp_hold = t + XLANE
    if dp: sm.f64_hold = t + F64_BUSY
    # the first read after a wave start or an SALU EXEC write is VC_MISS late; a WMMA or DPP op skips the read stage
    rs = max(t + EXEC_AFTER_ISSUE + VC_MISS * w.exec_cold * (not lane and u != "WMMA"), sm.read_free * (u != "WMMA" and not dpx))
    sm.read_free, w.exec_cold, x = max(rs + 1, sm.read_free), False, rs if o.vsrcs else t + EXEC_AFTER_ISSUE - 1
    if u == "WMMA":  # waits on VALU/TRANS results only in A[0], B[0], C[0], C[1]
      e = max([x, sm.exec_free["WMMA"], sm.vop_done - WMMA_READ] +
              [w.wmma_out[r] + (r in o.src2) if r in w.wmma_out else w.ready.get(r, 0) - WMMA_READ
               for r in ss if r >= 256 and (r in w.wmma_out or r in o.wmma_first)])
      if o.wmma_c0 in w.wmma_out and sm.last_vexec > w.wmma_out[o.wmma_c0]: e = max(e, min(w.ready[o.wmma_c0], sm.last_vexec))
    else:
      e0, own = max(t + EXEC_AFTER_ISSUE, sm.exec_free[u] * (not lane), sm.vec_free * (not lane)), not (lane or dpx)
      sgpr_rd = t + SGPR_READ
      cold = sgpr_rd + SGPR_COLD if not o.writelane and any(r < VCC and r in w.cold and (r not in w.ready or sgpr_rd >= w.ready[r])
                                                            for r in ss - o.unused) else 0
      e = max([x + (sm.operand_read(w, k, o, t, x, lds) if own else 0), sm.exec_free[u] * own, sm.vec_free * own, cold] +
              [w.ready.get(r, 0) + (r >= 256 and lane) for r in ss if r >= 128] +
              [w.ready[r] for r in ss & w.valu_sgprs if r not in (VCC, VCC + 1)] +
              # a result an SALU EXEC write caught in flight comes from the file
              [max(x + VC_MISS + 1, w.nobypass[r] + 1)
               for r in ss if r in w.nobypass and w.nobypass[r] >= x - BYPASS] +
              [w.written_exec[r] for r in ss - w.valu_sgprs if r < VCC and r in w.written_exec and w.written_exec[r] > x])
    if dp: e = max(e, sm.f64_free + sm.f64_trans_extra * (u != "TRANS"))  # a VALU f64 op waits more after a transcendental
    elif u != "WMMA": e = max(e, sm.f64_valu_free)
    if u == "VALU" and not s.exec_mask and any(r < 128 for r in ds): e = max(e, t + EXEC_AFTER_ISSUE + 1)  # a v_cmp with no lane on
    if lane: e = max(e, e0 + XLANE, sm.last_vexec + 1)
    if u != "WMMA":
      e = max([e] + [w.ready[r] - (F64_BUSY + F64_LAT if dp else RESULT_LAT[u]) + 1 for r in o.vdsts if r in w.ready])
    busy, lat = WMMA_BUSY // (2 if o.iu4 else 1), RESULT_LAT[u]
    sm.exec_free[u] = e + (TRANS_II if u == "TRANS" else busy if u == "WMMA" else 1)
    if dp:
      tr = F64_AFTER_TRANS * (u == "TRANS")
      lat, sm.f64_free, sm.f64_valu_free, sm.f64_trans_extra = F64_BUSY + F64_LAT + tr - n + 1, e + F64_BUSY, e + F64_BUSY + 1 + tr, tr
    lat += XLANE * dpx
    wr = e + n - 1 + lat + o.row_dpp  # forwarded at e + lat, written once its bank's write port is free
    if u != "WMMA":
      while any((wr, r % 4) in sm.vgpr_write_port for r in o.vdsts): wr += 1
      sm.vgpr_write_port |= {(wr, r % 4) for r in o.vdsts}
    land = {r: n - len(o.vdsts) + j for j, r in enumerate(o.vdsts)} if n > 1 else {}  # a multi-pass op writes a dword a pass
    rdy = {r: w.ready[r] for r in ss if r < 128 and r in w.ready}
    op = AluOp(k, u, e, idx=len(w.execs), passes=n, pinned=u == "WMMA" or dp,
               srcs={r: w.producer[r] for r in ss if r >= 256 and r in w.producer})
    for r in ds: w.ready[r], w.written_exec[r] = e + land.get(r, n - 1) + lat, e
    w.wmma_out = {r: v for r, v in w.wmma_out.items() if r not in ds}
    w.nobypass = {r: v for r, v in w.nobypass.items() if r not in ds}
    op.ready = {r: w.ready[r] for r in o.vdsts}
    w.producer.update({r: op for r in op.ready})
    w.salu_fwd = {r: v for r, v in w.salu_fwd.items() if r not in ds}
    if u == "WMMA":
      for i, r in enumerate(o.vdsts): w.ready[r], w.wmma_out[r] = e + busy + WMMA_READ + 2 + i // 2, e + busy
      wr = w.ready[o.vdsts[-1]]
    op.written = wr + (o.writelane and any(r in w.cold for r in ss if r < VCC))  # a writelane reading a cold SGPR lands a cycle later
    op.leave, op.slot = e + busy + 1 if u == "WMMA" else e - 1, e + F64_BUSY * dp
    w.alu_ops.append(op)
    sm.vec_free = e + (busy + WMMA_READ + 1 if u == "WMMA" else n)
    if u != "WMMA": sm.vop_done, sm.last_vexec = e + 1 + (u == "TRANS"), e  # a transcendental holds the read stage two cycles for a WMMA
    sm.vq.append(op)
    w.vq_slot = max([0] + [f.slot + VQ_SLOT_FREE for f in nth(sm.vq, VQ_DEPTH)])
    w.valu_sgprs |= {r for r in ds if r < 128}
    if o.sgpr_dsts: w.sgpr_wq = w.sgpr_wq[-SGPR_WQ:] + [(max(w.ready.get(r, t) for r in o.sgpr_dsts) + VALU_OUT, VCC in o.sgpr_dsts and not o.e64)]
    if sr := {r for r in ss if r < 128}:
      # a forwarded read leaves it cold
      rd = max([max([t] + [f.slot + 1 for f in nth(sm.vq, SGPR_READ_QPOS + 1)]) + SGPR_READ] + [w.warming.pop(r) for r in sr if r in w.warming])
      thaw(w, rd)
      cold_reads = sr & w.cold - o.unused  # a VOP3 field past its operands is not read
      rd = max([rd, w.valu_sgpr_read + 1] + [w.written_exec[r] for r in cold_reads if r in rdy and rd < rdy[r] and r not in w.valu_sgprs])
      fwd = {r for r in sr if r in rdy and rd < rdy[r]}
      w.sgpr_read_free = max(w.sgpr_read_free + (rd == w.sgpr_read_free > w.valu_sgpr_read),
                             rd + dp + SGPR_COLD * (bool(cold_reads - fwd) or rd < w.sgpr_read_free))
      w.valu_sgpr_read = rd
      paid = {r for r in cold_reads - fwd if r < VCC}  # a cold read fills its aligned pair
      pf = {r ^ 1 for r in sr if r ^ 1 not in w.ready} | {r ^ 1 for r in paid if w.ready.get(r ^ 1, 0) <= rd}
      w.cold -= (sr - fwd) | pf
      evict(k, sr, rd)
    w.cold |= {r for r in ds if r < EXEC}
    return e if u != "WMMA" else None

  def branch(k, w, sm, s, o, t):
    nxt = w.steps[w.idx + 1].pc if w.idx + 1 < len(w.steps) else s.pc + s.inst.size()
    if not o.s_branch and nxt == s.pc + s.inst.size():
      w.after, w.redirect = "not_taken", IMM_AFTER_ALU
      return
    # a cycle more for a target straddling 16 B, a refetch a cycle sooner in the low half line, and a target in the same buffer slot
    # as the line before the branch's waits for that refill
    w.after, w.redirect = "taken", BR_IN_BUFFER + (nxt % 16 + w.steps[w.idx + 1].inst.size() > 16)
    if 0 <= nxt // LINE - s.pc // LINE < BLOCK // LINE: return
    w.after, w.redirect = "fetch", w.redirect + BR_REFETCH - BR_IN_BUFFER - (nxt % LINE < LINE // 2)
    if (s.pc // LINE - nxt // LINE) % (BLOCK // LINE) == 1 and (prev := [j for j in range(1, w.idx) if w.steps[j].pc // LINE == s.pc // LINE - 1]):
      w.redirect += max(0, w.issue[prev[-1] - 1] + FETCH_AHEAD - t)

  def issue(k, t, b: Bound):
    nonlocal fetch_cu
    w, sm = waves[k], sims[k]
    s, o = w.steps[w.idx], ops[k][w.idx]
    u, real = o.unit, runs(w, s, o)
    w.stalls.append((b.why, t - (w.start if w.idx == 0 else w.last + 1)))
    if w.after == "fetch": fetch_simd[sids[k]] = fetch_cu = (t, k)
    if u == "VALU" and o.readlane: catch(k, t)
    elif u in VALU:
      if w.exec_new and real: catch(k, t)
      w.exec_new = False
    if u in VALU: w.delay_alu[u == "TRANS"].append(t + o.passes - 1)
    if u == "VALU" and EXEC in o.dsts: w.valu_exec_mask = s.exec_mask
    rel, w.after = w.after == "barrier", ""
    ex: int|LDSRequest|None = None
    if u == "SALU": ex = salu(k, w, sm, o, t)
    elif u in VEC and real: ex = vector(k, w, sm, s, o, t)
    elif u == "LDS" and (s.exec_mask != 0 or w.zero_exec_real):
      sm.lds_issue = t + 1
      ex = q = LDSRequest(s, o, t, sids[k] % 2, max([t] + [w.ready.get(r, 0) for r in o.srcs]), w.prev != "NOP", sm)
      for xs in (lds.pending, sm.lds_reqs, w.lgkm): xs.append(q)
      w.last_lds, w.producer = q, {r: v for r, v in w.producer.items() if r not in o.dsts}
      # no VGPR scoreboard on LDS returns: s_waitcnt is the program's
      for r in o.dsts: w.ready[r], w.lds_landing[r] = 0, w.lds_landing.get(r, []) + [q]
    if u == "WAIT":
      lg, vm = waitcnt(s.inst)
      del w.lgkm[:max(0, len(w.lgkm) - lg)], w.vm[:max(0, len(w.vm) - vm)]
    elif u == "BRANCH": branch(k, w, sm, s, o, t)
    elif u == "BARRIER":
      sm.bar_issue = sm.bar_counted = t
      bar.arrivals.append((w.bars, sm))
      w.bars, w.after = w.bars + 1, "barrier"
    elif u == "SMEM":
      w.lgkm.append(t + SMEM_LAT)
    elif u == "VMEM" and not o.store: w.vm.append(t + VMEM_LAT)
    w.last, w.idx = t, w.idx + 1
    w.prev = "BNOP" if u == "NOP" and rel else u if real or u not in ("VALU", "TRANS", "LDS") else "ZDS" if w.zero_exec_real else "IMM"
    w.issue.append(t)
    w.kinds.append("IMM" if w.prev == "IMM" else u)
    w.execs.append(ex)

  def tick(c) -> bool: return lds.step(c) | bar.step(c)
  c = 0
  while lds.pending or any(w.idx < len(w.steps) for w in waves):
    ts = []
    for k, w in enumerate(waves):  # oldest first takes the ports
      if w.idx < len(w.steps) and (b := ready(k)) is not None:
        if b.t <= c: issue(k, c, b)
        ts.append(b.t)
    moved = tick(c)
    if not (moved or ts) and not lds.pending: raise RuntimeError("barrier deadlock")
    c += 1
    if not moved and ts and (nxt := min(ts)) >= c:  # until nxt only the LDS and the barrier can act
      while c < nxt and not tick(c): c += 1
      if c < nxt: c += 1
  return [Timeline(w.issue, max([w.last] + [f.exec for f in w.alu_ops] + [x.done for x in w.lgkm if isinstance(x, LDSRequest)]), w.kinds,
                   [x.grant + LDS_EXEC_PACKET if isinstance(x, LDSRequest) else x for x in w.execs], w.stalls) for w in waves]  # type: ignore[operator]
