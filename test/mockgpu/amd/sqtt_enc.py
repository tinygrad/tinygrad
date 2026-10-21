# SQTT trace encoder for the emulator (the decoder lives in tinygrad/renderer/amd/sqtt.py).
# run_asm emits packets inline as instructions execute; finished traces end up in emu.sqtt_traces.
from __future__ import annotations
from tinygrad.renderer.amd.dsl import Inst
from tinygrad.renderer.amd.sqtt import (_build_decode_tables, PACKET_TYPES_RDNA3, PacketType, InstOp,
                                        LAYOUT_HEADER, WAVESTART, WAVEEND, INST, IMMEDIATE, VALUINST, TS_DELTA_OR_MARK)

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

  def _mem_op(t: type[Inst], op_name: str) -> InstOp:
    is_store = "STORE" in op_name
    if issubclass(t, _DS):  # stores: one cycle plus one per dword (b32 -> LDS_WR_2, b64 -> LDS_WR_3)
      if not is_store: return InstOp.LDS_RD
      m = re.search(r'_B(\d+)$', op_name)
      return InstOp[f"LDS_WR_{min(5, 1 + max(1, int(m.group(1)) // 32 if m else 1))}"]
    if issubclass(t, _GLOBAL): return InstOp.SGMEM_WR_2 if is_store else InstOp.SGMEM_RD_1
    if issubclass(t, _FLAT) or issubclass(t, _SCRATCH): return InstOp.FLAT_WR_3 if is_store else InstOp.FLAT_RD_2
    return InstOp.SALU

  # Timing is decided after execution, not during it. The emulator runs each wave until s_barrier/s_endpgm, so emit()
  # and finish() only record what each wave issued; schedule() replays all waves of a workgroup cycle by cycle, one
  # barrier phase at a time, with per-SIMD issue ports, a shared in-order VALU pipe and one LDS unit for both SIMDs of
  # the pair. Constants come from the gfx1100 references in extra/sqtt/examples/gfx1100; DRAM latencies are rough.
  FETCH_LAT, END_LAT, SMEM_LAT, SMEM_AFTER_SALU = 416, 160, 880, 9
  VALU_EXEC_LAT, VALU_EXEC_LAT_SLOW, VALU_DEP_LAT, FP_ISSUE_DEP, MEM_AFTER_EXEC = 8, 9, 5, 5, 9
  LDS_ACCEPT, LDS_RET, IMM_AFTER_MEM, WAITCNT_FREE_PENALTY, BARRIER_RELEASE = 3, 28, 3, 4, 14
  VMEM_LOAD_LAT, STORE_ACK = 785, 325
  _SLOW_VALU = re.compile(r'V_(LSHLREV|LSHRREV|ASHRREV|FMA|LSHL_OR|LSHL_ADD)_')
  _REG = re.compile(r'([vs])\[(\d+)(?::(\d+))?\]')

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
    op = _mem_op(inst_type, op_name)
    return ("LDS" if issubclass(inst_type, _DS) else "VMEM" if op != InstOp.SALU else "SALU"), op

  events: list[tuple[int, Inst|None, bool|None]] = []  # (wave_id, inst, branch_taken); inst None = end of wave

  def emit(wave_id: int, inst: Inst, branch_taken: bool|None):
    """Record one executed instruction; timing is assigned later by schedule()."""
    events.append((wave_id, inst, branch_taken))

  def finish(wave_id: int):
    """Record the end of a wave."""
    if any(e[0] == wave_id for e in events): events.append((wave_id, None, None))

  def schedule() -> list[tuple[int, int, type[PacketType], dict]]:
    """Assign a cycle to every recorded instruction. Returns (cycle, seq, packet class, fields)."""
    out: list[tuple[int, int, type[PacketType], dict]] = []
    def put(t: int, pkt_cls: type[PacketType], **kw): out.append((t, len(out), pkt_cls, kw))
    recs: dict[int, list[list[tuple[Inst|None, bool|None]]]] = {}  # wave -> barrier phases -> records
    for wave_id, inst, bt in events:
      phases = recs.setdefault(wave_id, [[]])
      phases[-1].append((inst, bt))
      if inst is not None and _classify(inst, bt)[0] == "BARRIER": phases.append([])
    waves = sorted(recs)
    simd, slot = {w: 2 * (w % 2) for w in waves}, {w: w // 2 for w in waves}
    st = {w: dict(i=0, ready=w + FETCH_LAT, blocked=False, valu_exec=None, salu_issue=None, mem_issue=None, reg_ready={},
                  reg_issue={}, lgkm=[], vm=[], vs=[], arrival=None, lds_wait=0, valu_issues=[], pend=[], last_cls=None) for w in waves}
    valu_free, unit, port_free = {0: 0, 2: 0}, dict(lds=0), {}
    lds_q: list[tuple[int, int, bool, int]] = []  # (accept, wave, store, occupancy), served by serve_lds

    lds_tie = dict(store=None, n=0)  # same-slot ties between SIMDs alternate, simd 2 first, within a run of stores or loads

    def serve_lds(upto: int):
      """Start every queued LDS request whose service slot is decidable by cycle upto (accept = issue + LDS_ACCEPT)."""
      while lds_q and (t := max(unit["lds"], min(r[0] for r in lds_q))) < upto + LDS_ACCEPT:
        cand = [r for r in lds_q if r[0] <= t]
        top = min(slot[r[1]] for r in cand)
        cand = [r for r in cand if slot[r[1]] == top]
        if len({simd[r[1]] for r in cand}) > 1:
          if lds_tie["store"] != cand[0][2]: lds_tie.update(store=cand[0][2], n=0)
          want = 2 if lds_tie["n"] % 2 == 0 else 0
          lds_tie["n"] += 1
          cand = [r for r in cand if simd[r[1]] == want]
        r = cand[0]
        lds_q.remove(r)
        unit["lds"] = t + r[3]
        st[r[1]]["lgkm"].append(t + LDS_RET + (2 if r[2] else 0))
        st[r[1]]["lds_wait"] -= 1
    for w in waves: put(w, WAVESTART, simd=simd[w], wgp=0, wave=slot[w], id7=w)

    def issue(w: int, cls: str, op: InstOp|None, inst: Inst, cycle: int) -> bool:
      """Issue one record of wave w at cycle if its readiness rules allow; else push its ready time and return False."""
      s, primary, sl = st[w], simd[w] == 0, slot[w]
      reads = _regs(inst, ("src0", "src1", "src2", "vsrc1", "addr", "data0", "data1", "data", "saddr", "ssrc0", "ssrc1"))
      writes = _regs(inst, ("vdst", "sdst"))
      need = cycle
      if cls == "VALU" and re.search(r'_F(16|32|64)', _name(inst)):  # fp consumer of a VALU result stalls at issue
        need = max([need] + [s["reg_issue"][r] + FP_ISSUE_DEP for r in reads if r in s["reg_issue"]])
      if (cls in ("LDS", "VMEM") or (cls == "BR" and _name(inst).startswith("S_CBRANCH_EXEC"))) and s["valu_exec"] is not None:
        need = max(need, s["valu_exec"] + MEM_AFTER_EXEC)
      if cls == "SMEM" and s["salu_issue"] is not None: need = max(need, s["salu_issue"] + SMEM_AFTER_SALU)
      for countdown, dep in s["pend"]:  # s_delay_alu: VALU_DEP_n / TRANS32_DEP_n wait for the n-th newest VALU, SALU_CYCLE_n
        if countdown: continue
        if 1 <= dep <= 7 and len(s["valu_issues"]) >= (n := dep if dep <= 4 else dep - 4):
          need = max(need, s["valu_issues"][-n] + VALU_DEP_LAT)
        elif 9 <= dep <= 11 and s["salu_issue"] is not None: need = max(need, s["salu_issue"] + 1 + dep - 8)
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
      s["last_cls"] = cls
      if cls == "IMM":
        if "WAITCNT" in _name(inst):
          for q, n in zip((s["lgkm"], s["vm"]), _waits(inst)): del q[:max(0, len(q) - n)]
          if not s["blocked"]: s["ready"] = cycle + WAITCNT_FREE_PENALTY
        s["blocked"] = False
        if primary: put(cycle, IMMEDIATE, wave=sl)
      elif cls == "VALU":
        base = cycle + (VALU_EXEC_LAT_SLOW if _SLOW_VALU.search(_name(inst)) else VALU_EXEC_LAT)
        ex = max([base, valu_free[simd[w]]] + [s["reg_ready"].get(r, 0) for r in reads])
        # a source forwarded the cycle its producer writes back collides with another source read from the same VGPR bank
        vreads = [r for r in reads if r[0] == "v"]
        if ex == base and any(s["reg_ready"].get(f) == ex and any(o != f and int(o[1:]) % 4 == int(f[1:]) % 4 for o in vreads)
                              for f in vreads): ex += 1
        valu_free[simd[w]], s["valu_exec"] = ex + 1, ex
        s["valu_issues"].append(cycle)
        if op is not None and op.name.startswith(("VALUB_", "VALUT_")): port_free[(simd[w], "VALU")] = cycle + _cycles(op)
        for r in writes: s["reg_ready"][r], s["reg_issue"][r] = ex + VALU_DEP_LAT, cycle
        if primary: put(cycle, VALUINST, wave=sl) if op is None else put(cycle, INST, wave=sl, op=op)
      elif cls == "LDS":
        store = "STORE" in _name(inst)
        occ = _cycles(op) if store else 1
        lds_q.append((cycle + LDS_ACCEPT, w, store, occ))
        s["lds_wait"] += 1
        s["mem_issue"] = cycle
        put(cycle, INST, wave=sl, op=op if primary else InstOp(0x50 + occ - 1))
      elif cls == "VMEM":
        store = "STORE" in _name(inst)
        (s["vs"] if store else s["vm"]).append(cycle + (STORE_ACK if store else VMEM_LOAD_LAT))
        s["mem_issue"] = cycle
        put(cycle, INST, wave=sl, op=op if primary else InstOp(0x5a + _cycles(op) - 1))
      else:
        if cls == "SMEM":
          s["lgkm"].append(cycle + SMEM_LAT)
          s["mem_issue"] = cycle
        if cls == "SALU": s["salu_issue"] = cycle
        if cls == "BARRIER": s["arrival"] = cycle
        if primary: put(cycle, INST, wave=sl, op=op)
      return True

    for ph in range(max(len(p) for p in recs.values())):
      live = [w for w in waves if ph < len(recs[w]) and recs[w][ph]]
      if not live: continue
      if ph > 0:
        release = max(st[w]["arrival"] for w in waves if st[w]["arrival"] is not None) + BARRIER_RELEASE
        for w in live: st[w]["ready"] = max(st[w]["ready"], release)
      for w in live: st[w]["i"] = 0
      cycle = min(st[w]["ready"] for w in live)
      while live:
        serve_lds(cycle)
        busy: set[tuple[int, str]] = set()  # one issue per (simd, class) per cycle, one per wave per cycle
        for w in sorted(live, key=lambda w: (simd[w] == 0, w)):  # paired SIMD first: it wins LDS ties (sync loads)
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
            if cycle >= s["ready"] and cycle >= port_free.get((simd[w], cls), 0) and (simd[w], cls) not in busy \
                 and issue(w, cls, op, inst, cycle): busy.add((simd[w], cls))
          if s["i"] >= len(recs[w][ph]): live.remove(w)
        cycle = max(cycle + 1, min((st[w]["ready"] for w in live), default=cycle + 1))
    return out

  def finalize() -> bytes:
    """Schedule the recorded events, sort by cycle, encode the gaps, pad and return the SQTT blob."""
    nibbles: list[int] = []
    _emit_nibbles(nibbles, LAYOUT_HEADER, layout=3, sel_a=6)
    now = 0
    for t, _, pkt_cls, kw in sorted(schedule(), key=lambda r: (r[0], r[1])):
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
