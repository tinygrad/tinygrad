# SQTT trace encoder for the emulator (the decoder lives in tinygrad/renderer/amd/sqtt.py).
# run_asm emits packets inline as instructions execute; finished traces end up in emu.sqtt_traces.
from __future__ import annotations
from tinygrad.renderer.amd.dsl import Inst
from tinygrad.renderer.amd.sqtt import (_build_decode_tables, PACKET_TYPES_RDNA3, PacketType, InstOp,
                                        LAYOUT_HEADER, WAVESTART, WAVEEND, INST, IMMEDIATE, IMMEDIATE_MASK, VALUINST, TS_DELTA_OR_MARK)

_NIB_COUNTS = {cls: nc for _, (cls, nc, *_) in _build_decode_tables(PACKET_TYPES_RDNA3)[0].items()}

def _emit_nibbles(nibbles: list[int], pkt_cls: type[PacketType], **kwargs):
  raw = pkt_cls.encoding.default
  for k, v in kwargs.items(): raw = pkt_cls.__dict__[k].set(raw, v)
  nibbles.extend((raw >> (i * 4)) & 0xF for i in range(_NIB_COUNTS[pkt_cls]))

def make_encoder():
  """Build an SQTT trace encoder for the emulator. Returns (emit, finish, finalize)."""
  from tinygrad.runtime.autogen.amd.rdna3.enum import SOPPOp as SOPPOp3
  from tinygrad.runtime.autogen.amd.rdna4.enum import SOPPOp as SOPPOp4
  from tinygrad.runtime.autogen.amd.rdna3 import ins as ir3
  from tinygrad.runtime.autogen.amd.rdna4 import ins as ir4
  from tinygrad.runtime.autogen.amd.cdna import ins as irc
  import re
  from tinygrad.helpers import getenv
  SIMD_PARITY = getenv("SQTT_SIMD_PARITY", 0)  # which SIMD of the pair gets wave 0: the dispatcher's round robin carries over

  def _kinds(*names: str) -> tuple[type[Inst], ...]:
    return tuple(getattr(m, n) for m in (ir3, ir4, irc) for n in names if hasattr(m, n))
  _SOPP, _SOPK, _SMEM, _DS = _kinds('SOPP'), _kinds('SOPK'), _kinds('SMEM'), _kinds('DS')
  _GLOBAL, _FLAT, _SCRATCH = _kinds('GLOBAL', 'VGLOBAL'), _kinds('FLAT', 'VFLAT'), _kinds('SCRATCH', 'VSCRATCH')
  _VALU = _kinds('VOP1', 'VOP2', 'VOP3', 'VOP3P', 'VOP3PX2', 'VOPC', 'VOPD', 'VOP3SD', 'VOP3_SDST', 'VOP1_SDST')

  # SOPP classification sets
  _SOPP_SKIP = {SOPPOp3.S_ENDPGM.value, SOPPOp3.S_ENDPGM_SAVED.value, SOPPOp3.S_ENDPGM_ORDERED_PS_DONE.value, SOPPOp3.S_DELAY_ALU.value}
  _SOPP_IMMEDIATE = {SOPPOp3.S_NOP.value, SOPPOp3.S_CLAUSE.value, SOPPOp3.S_WAITCNT.value, SOPPOp3.S_WAITCNT_DEPCTR.value,
                     SOPPOp3.S_WAIT_IDLE.value, SOPPOp3.S_WAIT_EVENT.value, SOPPOp3.S_SLEEP.value, SOPPOp3.S_SET_INST_PREFETCH_DISTANCE.value}
  for _op in (SOPPOp4.S_WAIT_ALU, SOPPOp4.S_WAIT_LOADCNT, SOPPOp4.S_WAIT_STORECNT, SOPPOp4.S_WAIT_SAMPLECNT,
              SOPPOp4.S_WAIT_BVHCNT, SOPPOp4.S_WAIT_EXPCNT, SOPPOp4.S_WAIT_DSCNT, SOPPOp4.S_WAIT_KMCNT,
              SOPPOp4.S_WAIT_LOADCNT_DSCNT, SOPPOp4.S_WAIT_STORECNT_DSCNT):
    _SOPP_IMMEDIATE.add(_op.value)
  _SOPP_BARRIER = {SOPPOp3.S_BARRIER.value}
  if hasattr(SOPPOp4, 'S_BARRIER_WAIT'): _SOPP_BARRIER.add(SOPPOp4.S_BARRIER_WAIT.value)
  if hasattr(SOPPOp4, 'S_BARRIER_LEAVE'): _SOPP_BARRIER.add(SOPPOp4.S_BARRIER_LEAVE.value)
  _SOPP_BRANCH = {SOPPOp3.S_BRANCH.value, SOPPOp3.S_CBRANCH_SCC0.value, SOPPOp3.S_CBRANCH_SCC1.value,
                  SOPPOp3.S_CBRANCH_VCCZ.value, SOPPOp3.S_CBRANCH_VCCNZ.value,
                  SOPPOp3.S_CBRANCH_EXECZ.value, SOPPOp3.S_CBRANCH_EXECNZ.value}

  # VALU sub-classification patterns
  _VALUT_4_RE = re.compile(r'V_(EXP|LOG|RCP|RSQ|SQRT|SIN|COS|CEIL|FLOOR|TRUNC|RNDNE|FRACT|FREXP)_')
  _VALUB_2_RE = re.compile(r'V_(LSHLREV|LSHRREV|ASHRREV)_(B|I)64')
  _VALUB_4_RE = re.compile(r'V_MAD_(U|I)64')
  _VALUB_16_RE = re.compile(r'V_\w+_F64')

  def _valu_op(op_name: str) -> InstOp|None:
    if 'CMPX' in op_name: return InstOp.VALU1_WR_EXEC
    if _VALUB_2_RE.search(op_name): return InstOp.VALUB_2
    if _VALUB_4_RE.search(op_name): return InstOp.VALUB_4
    if _VALUB_16_RE.search(op_name): return InstOp.VALUB_16
    if _VALUT_4_RE.search(op_name): return InstOp.VALUT_4
    return None

  def _mem_op(t: type[Inst], op_name: str, saddr: bool) -> InstOp:  # saddr: SGPR base, else 64-bit VGPR address
    is_store = "STORE" in op_name
    if issubclass(t, _DS):  # stores: one cycle plus one per dword (b32 -> LDS_WR_2, b64 -> LDS_WR_3)
      if not is_store: return InstOp.LDS_RD
      m = re.search(r'_B(\d+)$', op_name)
      return InstOp[f"LDS_WR_{min(5, 1 + max(1, int(m.group(1)) // 32 if m else 1))}"]
    if issubclass(t, _GLOBAL): return InstOp.SGMEM_WR_2 if is_store else InstOp.SGMEM_RD_1 if saddr else InstOp.SGMEM_RD_2
    if issubclass(t, _FLAT) or issubclass(t, _SCRATCH): return InstOp.FLAT_WR_3 if is_store else InstOp.FLAT_RD_2
    return InstOp.SALU

  # timing is decided after execution: emit()/finish() record what each wave issued, schedule() replays the traced
  # workgroups cycle by cycle on one WGP pair (simd 0 traced, simd 2 paired), all numbers measured on gfx1100 traces
  # front end: first fetch, wave end, 64 B icache lines fetched in blocks of 4, half-block prefetch
  FETCH_LAT, END_LAT, ICACHE_LINE_LAT = 416, 16, 530
  # VALU: issue -> exec 8 (9 shifts/fma, 6 no register read), result +5, per-wave queue of 13
  VALU_EXEC_LAT, VALU_EXEC_LAT_SLOW, VALU_EXEC_LAT_NOREAD, VALU_DEP_LAT, VALU_QUEUE = 8, 9, 6, 5, 13
  WMMA_LAT, WMMA_INFLIGHT = 34, 6  # a WMMA holds the pipe 34 cycles; issue stalls with 6 in flight
  VALU_AFTER_LDS_WAIT = 2  # VALU right after a wait blocked on LDS executes 2 earlier
  TRANS_OCC, TRANS_DEP_LAT = 4, 10  # transcendental execs at least 4 apart, result +10
  # SALU and branches: exec +4, dependent +2, branch waits for its SCC/VCC producer, taken +10 (+20 into another line)
  SALU_EXEC_LAT, SALU_DEP_LAT, BRANCH_AFTER_SALU, BRANCH_TAKEN, BRANCH_NOT_TAKEN, BRANCH_NEW_LINE = 4, 2, 5, 10, 3, 20
  # crossings between units: VALU-written SGPR/VCC read by SALU at exec + 9, VALU after SALU exec + 5, s_nop after ALU + 3
  VALU_TO_SCALAR, VALU_AFTER_SALU, IMM_AFTER_ALU, SLEEP_CYCLES = 9, 5, 3, 24  # s_sleep 1 measured once
  # memory issue after the last VALU exec, SMEM after SALU, waitcnt token after memory issue
  MEM_AFTER_EXEC, SMEM_AFTER_SALU, IMM_AFTER_MEM, WAITCNT_FREE_PENALTY, BARRIER_RELEASE = 9, 9, 3, 4, 12
  # LDS: one server, accept +3 (+8 for a store racing another wave's on the same SIMD), 8 queued per SIMD
  LDS_ACCEPT, LDS_ACCEPT_LATE, LDS_RET, LDS_QUEUE = 3, 8, 28, 8
  # load return per width: (extra latency, cycles on the return path)
  LDS_RET_WIDE = {"32": (0, 1), "64": (4, 4), "96": (4, 5), "128": (4, 6)}
  # DRAM: rough, not exact targets
  SMEM_LAT, VMEM_LOAD_LAT, STORE_ACK = 880, 785, 325
  _SLOW_VALU = re.compile(r'V_(LSHLREV|LSHRREV|ASHRREV|FMA|LSHL_OR|LSHL_ADD)_')
  _REG = re.compile(r'([vs])\[(\d+)(?::(\d+))?\]')

  def _port(op: InstOp|None) -> str:  # transcendentals issue on their own port, every 4 cycles, beside plain VALU
    return "TRANS" if op is not None and op.name.startswith("VALUT_") else "VALU"

  def _cycles(op: InstOp) -> int:
    """Issue cycles encoded in the op name (LDS_WR_2 -> 2); 1 when the name carries none."""
    return int(m.group(1)) if (m := re.search(r'_(\d+)$', op.name)) else 1

  def _regs(inst: Inst, names: tuple[str, ...]) -> set[str]:
    out: set[str] = set()
    for n in names:
      for m in _REG.finditer(repr(getattr(inst, n, ""))):
        lo = int(m.group(2))
        out |= {f"{m.group(1)}{r}" for r in range(lo, int(m.group(3) or lo) + 1)}
    return out

  def _waits(inst: Inst) -> tuple[int, int]:
    """(lgkmcnt, vmcnt) targets of s_waitcnt or the SOPK s_waitcnt_<counter> form (null sdst: the immediate is the count)."""
    name, v = _name(inst), inst.simm16
    if name == "S_WAITCNT_LGKMCNT": return v & 0x3f, 0x3f
    if name == "S_WAITCNT_VMCNT": return 0x3f, v & 0x3f
    if name.startswith("S_WAITCNT_"): return 0x3f, 0x3f
    return (v >> 4) & 0x3f, (v >> 10) & 0x3f

  def _name(inst: Inst) -> str:
    """Op name; a VOPD pair is named by both halves."""
    return inst.op.name if hasattr(inst, 'op') else f"{inst.opx.name}::{inst.opy.name}" if hasattr(inst, 'opx') else ""

  def _classify(inst: Inst, branch_taken: bool|None) -> tuple[str, InstOp|None]:
    """Issue class and INST op of one instruction; class NONE is not traced."""
    inst_type, inst_op, op_name = type(inst), inst.op.value if hasattr(inst, 'op') else 0, _name(inst)
    if issubclass(inst_type, _SOPK) and op_name.startswith("S_WAITCNT_"): return "IMM", None
    if issubclass(inst_type, _SOPP):
      if inst_op in _SOPP_SKIP: return "NONE", None
      if inst_op in _SOPP_IMMEDIATE: return "IMM", None
      if inst_op in _SOPP_BARRIER: return "BARRIER", InstOp.BARRIER
      if inst_op in _SOPP_BRANCH: return "BR", InstOp.JUMP if branch_taken else InstOp.JUMP_NO
      return "SALU", InstOp.SALU
    if issubclass(inst_type, _VALU): return "VALU", _valu_op(op_name)
    if issubclass(inst_type, _SMEM): return "SMEM", InstOp.SMEM_RD
    op = _mem_op(inst_type, op_name, bool(_regs(inst, ("saddr",))))
    return ("LDS" if issubclass(inst_type, _DS) else "VMEM" if op != InstOp.SALU else "SALU"), op

  events: list[tuple[int, Inst|None, bool|None, int, tuple[int, int]]] = []  # (wave_id, inst, branch_taken, pc, (exec, bank degree))

  def emit(wave_id: int, inst: Inst, branch_taken: bool|None, pc: int, exec_mask: int = -1, bank_degree: int = 1):
    """Record one executed instruction; timing is assigned later by schedule()."""
    events.append((wave_id, inst, branch_taken, pc, (exec_mask, bank_degree)))

  def finish(wave_id: int):
    """Record the end of a wave."""
    if any(e[0] == wave_id for e in events): events.append((wave_id, None, None, 0, (-1, 1)))

  def schedule() -> list[tuple[int, int, type[PacketType], dict]]:
    """Assign a cycle to every recorded instruction. Returns (cycle, seq, packet class, fields)."""
    out: list[tuple[int, int, type[PacketType], dict]] = []
    def put(t: int, pkt_cls: type[PacketType], **kw): out.append((t, len(out), pkt_cls, kw))
    recs: dict[int, list[list[tuple[Inst|None, bool|None]]]] = {}  # wave -> barrier phases -> records
    pcs: dict[int, list[list[int]]] = {}  # same shape as recs: byte offset of each record in the code
    lanes: dict[int, list[list[tuple[int, int]]]] = {}  # same shape: (exec mask, LDS bank conflict degree)
    for wave_id, inst, bt, pc, ln in events:
      pcs.setdefault(wave_id, [[]])[-1].append(pc)
      lanes.setdefault(wave_id, [[]])[-1].append(ln)
      phases = recs.setdefault(wave_id, [[]])
      phases[-1].append((inst, bt))
      if inst is not None and _classify(inst, bt)[0] == "BARRIER":
        phases.append([])
        pcs[wave_id].append([])
        lanes[wave_id].append([])
    waves = sorted(recs)
    rank = {w: k for k, w in enumerate(waves)}  # wave id = workgroup * 64 + wave; placement goes by rank across workgroups
    simd, slot = {w: 2 * ((rank[w] + SIMD_PARITY) % 2) for w in waves}, {w: rank[w] // 2 for w in waves}
    st = {w: dict(i=0, ready=rank[w] + FETCH_LAT, blocked=False, valu_exec=None, salu_issue=None, salu_exec=None, mem_issue=None, reg_ready={},
                  reg_issue={}, lgkm=[], vm=[], vs=[], arrival=None, lds_wait=0, valu_execs=[], valu_issues=[],
                  early_valu=False, last_mem=None, lds_dst=set(),
                  wmma_done=[], wmma_ready={}, vsgpr_exec={}, pend=[], last_issue=0,
                  last_cls=None) for w in waves}
    valu_free, trans_free, unit, port_free = {0: 0, 2: 0}, {0: 0, 2: 0}, dict(lds=0), {}
    # instruction cache: 64 B lines fetched in aligned blocks of ahead + 1 lines (s_set_inst_prefetch_distance, default 3);
    # a miss fetches the line's block, entering the second half of a block prefetches the next one
    icache = dict(ahead=3, ready={line: FETCH_LAT for line in range(4)})
    def line_ready(line: int, cycle: int) -> int:
      n = icache["ahead"] + 1
      def fetch(block: int):
        for x in range(block * n, block * n + n): icache["ready"].setdefault(x, cycle + ICACHE_LINE_LAT)
      if line not in icache["ready"]: fetch(line // n)
      if line % n >= n // 2: fetch(line // n + 1)
      return icache["ready"][line]
    lds_issued, lds_starts = {0: 0, 2: 0}, {0: [], 2: []}  # per SIMD: requests issued, service start times in order
    def lds_credits(sm: int, cycle: int) -> int:  # slots free in pairs at the second service's start (the first one alone)
      st_ = lds_starts[sm]
      return sum(1 for n in range(len(st_)) if (i := ((n + 1) | 1) - 1) < len(st_) and st_[i] <= cycle)
    lds_q: list[tuple[int, int, bool, int, str|None, int]] = []  # (accept, wave, store, occupancy, load width, conflict cycles)
    ret_free, lds_last = {0: 0, 2: 0}, {}

    def serve_lds(upto: int):
      """Start every queued LDS request whose service slot is decidable by cycle upto (accept = issue + LDS_ACCEPT)."""
      while lds_q and (t := max(unit["lds"], min(r[0] for r in lds_q))) < upto + LDS_ACCEPT:
        cand = [r for r in lds_q if r[0] <= t]  # the traced SIMD's port wins whenever it has a ready request, FIFO within a SIMD
        idle_tie = t > unit["lds"] and len({simd[r[1]] for r in cand}) > 1  # both ports became ready while the server sat idle
        r = next((r for r in cand if simd[r[1]] == (2 if idle_tie else 0)), cand[0])
        lds_q.remove(r)
        lds_starts[simd[r[1]]].append(t)
        unit["lds"] = t + r[3]
        t += r[5]
        if r[2]: st[r[1]]["lgkm"].append(t + LDS_RET + 2)
        else:  # loads return their data one after another per SIMD
          extra, beat = LDS_RET_WIDE[r[4]]
          ret_free[simd[r[1]]] = done = max(t + LDS_RET + extra, ret_free[simd[r[1]]] + beat)
          st[r[1]]["lgkm"].append(done)
        st[r[1]]["lds_wait"] -= 1
    for w in waves: put(w, WAVESTART, simd=simd[w], wgp=0, wave=slot[w], id7=rank[w])

    def issue(w: int, cls: str, op: InstOp|None, inst: Inst, cycle: int) -> bool:
      """Issue one record of wave w at cycle if its readiness rules allow; else push its ready time and return False."""
      s, primary, sl = st[w], simd[w] == 0, slot[w]
      reads = _regs(inst, ("src0", "src1", "src2", "vsrc1", "addr", "data0", "data1", "data", "saddr", "ssrc0", "ssrc1",
                         "srcx0", "vsrcx1", "srcy0", "vsrcy1"))
      writes = _regs(inst, ("vdst", "sdst", "vdstx", "vdsty"))
      need = max(cycle, line_ready(pcs[w][ph][s["i"]] // 64, cycle))
      if cls == "VALU" and len(busy_wmma := [d for d in s["wmma_done"] if d - 1 > cycle]) >= WMMA_INFLIGHT:
        need = max(need, min(busy_wmma) - 1)
      if cls == "VALU" and len(s["valu_execs"]) >= VALU_QUEUE:  # the per-wave VALU queue holds 13; a slot frees at exec + 4
        need = max(need, s["valu_execs"][-VALU_QUEUE] + 4)
      if cls == "LDS" and lds_issued[simd[w]] - lds_credits(simd[w], cycle) >= LDS_QUEUE: need = max(need, cycle + 1)
      if (cls in ("LDS", "VMEM") or (cls == "BR" and _name(inst).startswith("S_CBRANCH_EXEC"))) and s["valu_exec"] is not None:
        need = max(need, s["valu_exec"] + MEM_AFTER_EXEC)
      if cls == "BR" and "CBRANCH" in _name(inst) and "EXEC" not in _name(inst) and s["salu_exec"] is not None:
        need = max(need, s["salu_exec"] + BRANCH_AFTER_SALU)  # waits for its SCC/VCC producer
      if cls in ("SALU", "SMEM", "BR"):  # an SGPR or VCC written by a VALU stalls the scalar reader until exec + 9
        need = max([need] + [s["vsgpr_exec"][r] + VALU_TO_SCALAR for r in reads | ({"s106"} if "VCC" in _name(inst) else set())
                             if r in s["vsgpr_exec"]])
      if cls == "SMEM" and s["salu_issue"] is not None: need = max(need, s["salu_issue"] + SMEM_AFTER_SALU)
      for countdown, dep in s["pend"]:  # s_delay_alu: VALU_DEP_n / TRANS32_DEP_n wait for the n-th newest VALU, SALU_CYCLE_n
        if countdown: continue
        if 1 <= dep <= 7 and len(s["valu_issues"]) >= (n := dep if dep <= 4 else dep - 4):
          need = max(need, s["valu_issues"][-n] + VALU_DEP_LAT)
        elif 9 <= dep <= 11 and s["salu_issue"] is not None: need = max(need, s["salu_issue"] + 1 + dep - 8)
      if cls == "IMM" and s["last_cls"] in ("SALU", "VALU"): need = max(need, s["last_issue"] + IMM_AFTER_ALU)
      if cls == "VALU" and s["salu_exec"] is not None: need = max(need, s["salu_exec"] + VALU_AFTER_SALU)
      if cls == "IMM":
        if s["mem_issue"] is not None: need = max(need, s["mem_issue"] + IMM_AFTER_MEM)
        if "WAITCNT" in _name(inst):
          lgkm_n, vm_n = _waits(inst)
          if s["lds_wait"] and len(s["lgkm"]) < len(s["lgkm"]) + s["lds_wait"] - lgkm_n:
            s["ready"], s["blocked"] = cycle + 1, True
            return False  # a counted LDS request is not served yet
          for q, n in ((s["lgkm"], lgkm_n - s["lds_wait"]), (s["vm"], vm_n)):  # unserved LDS are newest
            q.sort()
            if len(q) > n:
              sat = q[len(q) - n - 1]
              if sat > need: need, s["blocked"] = sat, True
      if need > cycle:
        s["ready"] = need
        return False
      s["i"] += 1
      s["ready"] = cycle + 1
      s["pend"] = [[c - 1, d] for c, d in s["pend"] if c > 0]
      s["last_cls"], s["last_issue"] = cls, cycle
      if cls == "IMM":
        if _name(inst) == "S_SET_INST_PREFETCH_DISTANCE": icache["ahead"] = inst.simm16 & 3
        if "WAITCNT" in _name(inst):
          for q, n in zip((s["lgkm"], s["vm"]), _waits(inst)): del q[:max(0, len(q) - n)]
          if not s["blocked"]: s["ready"] = cycle + WAITCNT_FREE_PENALTY
          s["early_valu"] = s["blocked"] and s["last_mem"] == "LDS"  # an LDS return is known ahead: the next VALU starts early
        s["blocked"] = False
        # s_nop N / s_sleep N hold the wave; the token is stamped when they end
        hold = inst.simm16 & 0xf if _name(inst) == "S_NOP" else SLEEP_CYCLES * (inst.simm16 & 0x7f) if _name(inst) == "S_SLEEP" else 0
        if hold: s["ready"] = max(s["ready"], cycle + hold + 1)
        if primary: put(cycle + hold, IMMEDIATE, wave=sl)
      elif cls == "VALU":
        lat = VALU_EXEC_LAT_NOREAD if not reads else VALU_EXEC_LAT_SLOW if _SLOW_VALU.search(_name(inst)) else VALU_EXEC_LAT
        if s["early_valu"] and not reads & s["lds_dst"]: lat = max(VALU_EXEC_LAT_NOREAD, lat - VALU_AFTER_LDS_WAIT)
        s["early_valu"] = False
        base = cycle + lat
        wmma = _name(inst).startswith("V_WMMA")
        ready, trans = s["wmma_ready"] if wmma else s["reg_ready"], _port(op) == "TRANS"
        ex = max([base, valu_free[simd[w]], trans_free[simd[w]] if trans else 0] + [ready.get(r, 0) for r in reads])
        if trans: trans_free[simd[w]] = ex + TRANS_OCC
        # accumulating in place: the forwarded register is also the destination and another source shares its VGPR bank
        if ex == base and any(s["reg_ready"].get(d) == ex and any(o != d and o[0] == "v" and int(o[1:]) % 4 == int(d[1:]) % 4
                                                                  for o in reads) for d in writes & reads if d[0] == "v"): ex += 1
        valu_free[simd[w]], s["valu_exec"] = ex + 1, ex
        if wmma:  # holds the pipe WMMA_LAT cycles; its exec token marks completion
          ex += WMMA_LAT
          valu_free[simd[w]], s["valu_exec"] = ex + 1, ex
          s["wmma_done"].append(ex)
        s["valu_execs"].append(ex)
        for r in writes | ({"s106"} if re.search(r"V_CMP|V_ADD_CO|V_SUB_CO", _name(inst)) else set()):
          if r[0] == "s": s["vsgpr_exec"][r] = ex
        s["valu_issues"].append(cycle)
        if op is not None and op.name.startswith(("VALUB_", "VALUT_")): port_free[(simd[w], _port(op))] = cycle + _cycles(op)
        for r in writes:
          s["reg_ready"][r], s["reg_issue"][r] = ex + (2 if wmma else TRANS_DEP_LAT if trans else VALU_DEP_LAT), cycle
          s["wmma_ready"][r] = ex + 1  # a WMMA takes its operands one cycle after the producer, VALU or WMMA
        if primary: put(cycle, VALUINST, wave=sl) if op is None else put(cycle, INST, wave=sl, op=op)
      elif cls == "LDS":
        store = "STORE" in _name(inst)
        occ = _cycles(op) if store else 1
        width = (m.group(1) if (m := re.search(r'_B(32|64|96|128)$', _name(inst))) else "32") if not store else None
        last = lds_last.get(simd[w])  # another wave's request on this SIMD within 3 cycles: accepted late (sync s0w1 store, issue + 8)
        acc = LDS_ACCEPT_LATE if last is not None and last[0] != w and store and cycle - last[1] < 3 else LDS_ACCEPT
        lds_last[simd[w]] = (w, cycle)
        if not store: s["lds_dst"] |= writes
        conflict = lanes[w][ph][s["i"] - 1][1] - 1  # bank conflicts serialize the access, the data follows it
        lds_q.append((cycle + acc, w, store, occ + conflict, width, conflict))
        lds_issued[simd[w]] += 1
        s["lds_wait"] += 1
        s["mem_issue"] = cycle
        s["last_mem"] = cls
        put(cycle, INST, wave=sl, op=op if primary else InstOp(0x50 + occ - 1))
      elif cls == "VMEM":
        store = "STORE" in _name(inst)
        (s["vs"] if store else s["vm"]).append(cycle + (STORE_ACK if store else VMEM_LOAD_LAT))
        s["mem_issue"] = cycle
        s["last_mem"] = cls
        put(cycle, INST, wave=sl, op=op if primary else InstOp(0x5a + _cycles(op) - 1))
      else:
        if cls == "SMEM":
          s["lgkm"].append(cycle + SMEM_LAT)
          s["mem_issue"] = cycle
          s["last_mem"] = cls
        if cls == "SALU":
          s["salu_issue"] = s["salu_exec"] = max([cycle + SALU_EXEC_LAT] + [s["reg_ready"].get(r, 0) for r in reads if r[0] == "s"])
          s["salu_issue"] = cycle
          for r in writes: s["reg_ready"][r] = s["salu_exec"] + SALU_DEP_LAT
        if cls == "BR":
          s["ready"] = cycle + (BRANCH_TAKEN if op == InstOp.JUMP else BRANCH_NOT_TAKEN)
          nxt = pcs[w][ph][s["i"]] if s["i"] < len(pcs[w][ph]) else None  # s["i"] already points past the branch
          if op == InstOp.JUMP and nxt is not None and nxt // 64 != pcs[w][ph][s["i"] - 1] // 64: s["ready"] += BRANCH_NEW_LINE
        if cls == "BARRIER": s["arrival"] = cycle
        if primary: put(cycle, INST, wave=sl, op=op)
      return True

    for ph in range(max(len(p) for p in recs.values())):
      live = [w for w in waves if ph < len(recs[w]) and recs[w][ph]]
      if not live: continue
      if ph > 0:
        for w in live:  # a barrier releases its own workgroup only
          release = max(st[x]["arrival"] for x in waves if x // 64 == w // 64 and st[x]["arrival"] is not None) + BARRIER_RELEASE
          st[w]["ready"] = max(st[w]["ready"], release)
      for w in live: st[w]["i"] = 0
      cycle = min(st[w]["ready"] for w in live)
      while live:
        serve_lds(cycle)
        busy: set[tuple[int, str]] = set()  # one issue per (simd, class) per cycle, one per wave per cycle
        for w in sorted(live):
          s = st[w]
          while (rec := recs[w][ph][s["i"]])[0] is not None and _classify(*rec)[0] == "NONE":  # hints cost no issue slot
            if _name(rec[0]) == "S_DELAY_ALU":  # id0 bits 3:0 -> next instruction; id1 bits 10:7 -> `skip` (6:4) later
              v = rec[0].simm16  # a newer s_delay_alu replaces the pending one
              s["pend"] = [[0, v & 0xf]] + ([[(v >> 4) & 7, (v >> 7) & 0xf]] if (v >> 7) & 0xf else [])
              if s["last_cls"] == "IMM": s["ready"] = max(s["ready"], cycle + 1)
            s["i"] += 1
          inst, bt = recs[w][ph][s["i"]]
          if inst is None:
            if cycle >= s["ready"]:
              put(max(cycle + END_LAT, max(s["vs"], default=0)), WAVEEND, simd=simd[w], wgp=0, wave=slot[w])
              s["i"] += 1
          else:
            cls, op = _classify(inst, bt)
            port = _port(op) if cls == "VALU" else cls
            if cycle >= s["ready"] and cycle >= port_free.get((simd[w], port), 0) and (simd[w], cls) not in busy \
                 and issue(w, cls, op, inst, cycle): busy.add((simd[w], cls))
          if s["i"] >= len(recs[w][ph]): live.remove(w)
        cycle = max(cycle + 1, min((st[w]["ready"] for w in live), default=cycle + 1))
    return out

  def finalize() -> bytes:
    """Schedule the recorded events, sort by cycle, encode the gaps, pad and return the SQTT blob."""
    nibbles: list[int] = []
    _emit_nibbles(nibbles, LAYOUT_HEADER, layout=3, sel_a=6)
    now = 0
    recs, imm = [], {}
    for r in sorted(schedule(), key=lambda r: (r[0], r[1])):  # waits of several waves done in one cycle share an IMMEDIATE_MASK
      if r[2] is not IMMEDIATE: recs.append(r)
      elif r[0] in imm: imm[r[0]][3]["mask"] |= 1 << r[3]["wave"]
      else: recs.append(imm.setdefault(r[0], (r[0], r[1], IMMEDIATE_MASK, {"mask": 1 << r[3]["wave"]})))
    recs = [(t, q, IMMEDIATE, {"wave": kw["mask"].bit_length() - 1}) if c is IMMEDIATE_MASK and kw["mask"] & (kw["mask"] - 1) == 0
            else (t, q, c, kw) for t, q, c, kw in recs]
    for t, _, pkt_cls, kw in recs:
      gap, room = t - now, pkt_cls.__dict__['delta'].mask
      if gap > room:  # a long gap goes in a timestamp packet ahead of the instruction packet
        _emit_nibbles(nibbles, TS_DELTA_OR_MARK, delta=gap - room)
        gap = room
      _emit_nibbles(nibbles, pkt_cls, delta=gap, **kw)
      now = t
    while len(nibbles) % 2 != 0: nibbles.append(0)
    nibbles.extend([0] * 32)
    while len(nibbles) % 64 != 0: nibbles.append(0)
    return bytes(nibbles[i] | ((nibbles[i + 1] if i + 1 < len(nibbles) else 0) << 4) for i in range(0, len(nibbles), 2))

  return emit, finish, finalize
