# Cycle-accurate RDNA3 (gfx1100/gfx1102) SQTT emulator for SRAM-only and general kernels
from dataclasses import dataclass, field
from collections import defaultdict
from tinygrad.renderer.amd.dsl import Inst, Reg, VCC_LO, EXEC_LO, NULL
from tinygrad.renderer.amd.sqtt import (encode, LAYOUT_HEADER, WAVESTART, WAVEEND, VALUINST, INST, IMMEDIATE, IMMEDIATE_MASK,
                                        ALUEXEC, VMEMEXEC, WAVERDY, TS_DELTA_OR_MARK, InstOp, AluSrc, MemSrc, PacketType)
from tinygrad.runtime.autogen.amd.rdna3 import ins as ir3
from tinygrad.viz.serve import amd_decode, get_cfg, COND_NOT_TAKEN

# ═══════════════════════════════════════════════════════════════════════════════
# RDNA3 HARDWARE TIMING CONSTANTS (calibrated against real gfx1100 SQTT traces)
# ═══════════════════════════════════════════════════════════════════════════════

SHADER_CLK_HZ = 2_482_000_000   # matches _gpu_props_rdna max_engine_clk_fcompute
RT_CLK_HZ = 100_000_000         # 100 MHz realtime counter (10 ns/tick)
RT_MARKER_PERIOD = 4096         # SQ_TT_RT_FREQ_4096_CLK

VALU_DISPATCH_TO_EXEC = 8       # 1-cycle VALU completion relative to dispatch
VALU_MOV_IMM_TO_EXEC = 6        # v_mov_b32 with inline/literal constant
VALU_RAW_LAT = 5                # VALU -> VALU RAW dependency (forwarding)
VALU_TO_MEM_RAW_LAT = 17        # VALU -> LDS/VMEM RAW dependency
VALU_EXEC_WR_TO_SALU_LAT = 16   # v_cmpx -> s_cbranch_execz
TFU_DISPATCH_TO_EXEC = 5        # VALUT_4 exec start relative to dispatch
SALU_DISPATCH_TO_EXEC = 4       # SALU/SMEM exec start relative to dispatch
SALU_RAW_LAT = 2                # SALU -> SALU RAW dependency
SALU_SCC_RAW_LAT = 1            # SALU -> SALU carry/SCC bypass
SALU_TO_VALU_RAW_LAT = 4        # SALU -> VALU RAW dependency
LDS_DISPATCH_TO_EXEC = 3        # DS op -> VMEMEXEC(LDS)
LDS_LOAD_RETURN_LAT_32 = 28     # 32-bit LDS load -> s_waitcnt lgkmcnt(0)
LDS_STORE_RETURN_LAT_32 = 30    # 32-bit LDS store -> s_waitcnt lgkmcnt(0)
LDS_RETURN_LAT_128 = 37         # 128-bit LDS load -> s_waitcnt lgkmcnt(0)
SMEM_RETURN_LAT = 28            # K$ hit SMEM load -> s_waitcnt lgkmcnt(0)
VMEM_DISPATCH_TO_EXEC = 3       # GLOBAL/FLAT -> VMEMEXEC(VMEM)
VMEM_RETURN_LAT = 160           # Global load -> s_waitcnt vmcnt(0)
BARRIER_RELEASE_LAT = 14        # cycles after last WG wave hits s_barrier
BRANCH_TAKEN_LAT = 4            # taken branch fetch penalty

_TRANSCENDENTAL_PREFIXES = ("V_RCP_", "V_RSQ_", "V_EXP_", "V_LOG_", "V_SIN_", "V_COS_", "V_SQRT_")
_VALUB2_OPS = {"V_LSHLREV_B64", "V_LSHRREV_B64", "V_ASHRREV_I64", "V_LSHL_B64", "V_LSHR_B64", "V_ASHR_I64"}
_VALUB4_PREFIXES = ("V_MAD_U64_U32", "V_MAD_I64_I32", "V_MUL_LO_U32", "V_MUL_HI_U32", "V_MUL_HI_I32")
_IMMEDIATE_OPS = {"S_NOP", "S_WAITCNT", "S_WAITCNT_VMCNT", "S_WAITCNT_EXPCNT", "S_WAITCNT_LGKMCNT", "S_WAITCNT_VSCNT",
                  "S_CLAUSE", "S_SLEEP", "S_SETPRIO"}
_SKIP_OPS = {"S_DELAY_ALU", "S_WAIT_ALU", "S_BARRIER_WAIT"}

# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION CLASSIFICATION & OPERAND EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class InstDesc:
  kind: str                 # 'valu', 'tfu', 'salu', 'smem', 'lds', 'vmem', 'branch', 'barrier', 'imm', 'delay', 'endpgm', 'msg'
  pkt_cls: type[PacketType] | None
  inst_op: InstOp | None
  pipe_dur: int             # execution pipe occupancy cycles
  vgpr_r: frozenset[int]
  vgpr_w: frozenset[int]
  sgpr_r: frozenset[int]
  sgpr_w: frozenset[int]
  reads_scc: bool = False
  writes_scc: bool = False
  reads_vcc: bool = False
  writes_vcc: bool = False
  reads_exec: bool = False
  writes_exec: bool = False
  is_mov_imm: bool = False
  is_store: bool = False
  mem_dwords: int = 1
  wmma_dur: int = 0

def _reg_indices(val, n_regs: int = 1) -> tuple[set[int], set[int]]:
  """Extract (vgpr_indices, sgpr_indices) from a decoded operand field."""
  vgprs: set[int] = set()
  sgprs: set[int] = set()
  if isinstance(val, Reg):
    off, sz = val.offset, max(val.sz, n_regs)
    if 256 <= off < 512: vgprs.update(range(off - 256, min(256, off - 256 + sz)))
    elif 0 <= off < 128 and off != NULL.offset: sgprs.update(range(off, min(128, off + sz)))
  elif isinstance(val, int):
    if 256 <= val < 512: vgprs.update(range(val - 256, min(256, val - 256 + n_regs)))
    elif 0 <= val < 128 and val != NULL.offset: sgprs.update(range(val, min(128, val + n_regs)))
  return vgprs, sgprs

def _dwords_from_name(name: str) -> int:
  if "B128" in name or "X4" in name: return 4
  if "B96" in name or "X3" in name: return 3
  if "B64" in name or "2ADDR_B32" in name or "X2" in name or "F64" in name or "I64" in name or "U64" in name: return 2
  return 1

def classify_inst(inst: Inst) -> InstDesc:
  op_name = getattr(inst, "op_name", "")
  if (opx := getattr(inst, "opx", None)) is not None and (opy := getattr(inst, "opy", None)) is not None:
    op_name = f"{opx.name}_{opy.name}"
  vgpr_r: set[int] = set()
  vgpr_w: set[int] = set()
  sgpr_r: set[int] = set()
  sgpr_w: set[int] = set()
  dwords = _dwords_from_name(op_name)

  # extract destination / source registers
  for fname in ("vdst", "vdstx", "vdsty"):
    if hasattr(inst, fname) and (val := getattr(inst, fname)) is not None:
      v, s = _reg_indices(val, 8 if "WMMA" in op_name else dwords)
      vgpr_w |= v
      sgpr_w |= s
  for fname in ("sdst", "sdata"):
    if hasattr(inst, fname) and (val := getattr(inst, fname)) is not None:
      v, s = _reg_indices(val, dwords)
      if "STORE" in op_name:
        vgpr_r |= v
        sgpr_r |= s
      else:
        vgpr_w |= v
        sgpr_w |= s
  for fname in ("src0", "src1", "src2", "vsrc0", "vsrc1", "srcx0", "vsrcx1", "srcy0", "vsrcy1",
                "ssrc0", "ssrc1", "sbase", "soffset", "saddr", "addr", "data", "data0", "data1"):
    if hasattr(inst, fname) and (val := getattr(inst, fname)) is not None:
      v, s = _reg_indices(val, 2 if fname in ("sbase", "saddr") else dwords)
      vgpr_r |= v
      sgpr_r |= s
  # V_FMAC / V_WMMA / V_DUAL_FMAC also read their destination accumulator
  if any(k in op_name for k in ("FMAC", "WMMA", "DOT2ACC")): vgpr_r |= vgpr_w

  reads_scc = "SCC" in op_name or "ADDC" in op_name or "SUBB" in op_name or "CSELECT" in op_name
  writes_scc = op_name.startswith(("S_CMP_", "S_ADD_", "S_SUB_", "S_AND_", "S_OR_", "S_XOR_", "S_LSHL_", "S_LSHR_", "S_ASHR_", "S_MIN_", "S_MAX_"))
  reads_vcc = "VCC" in op_name or "_CO_CI_" in op_name or "CNDMASK" in op_name or (VCC_LO.offset in sgpr_r)
  writes_vcc = op_name.startswith("V_CMP_") or "_CO_" in op_name or (VCC_LO.offset in sgpr_w)
  reads_exec = "EXEC" in op_name or (EXEC_LO.offset in sgpr_r)
  writes_exec = "SAVEEXEC" in op_name or op_name.startswith("V_CMPX_") or (EXEC_LO.offset in sgpr_w)

  # S_ENDPGM
  if op_name in ("S_ENDPGM", "S_ENDPGM_SAVED"):
    return InstDesc("endpgm", None, None, 1, frozenset(), frozenset(), frozenset(), frozenset())
  # S_DELAY_ALU (skipped by map_insts, delays wave issue)
  if op_name in _SKIP_OPS:
    return InstDesc("delay", None, None, 1, frozenset(), frozenset(), frozenset(), frozenset())
  # S_BARRIER
  if op_name == "S_BARRIER":
    return InstDesc("barrier", INST, InstOp.BARRIER, 1, frozenset(), frozenset(), frozenset(), frozenset())
  # S_SENDMSG
  if op_name.startswith("S_SENDMSG"):
    return InstDesc("msg", INST, InstOp.MESSAGE, 1, frozenset(), frozenset(), frozenset(sgpr_r), frozenset())
  # Branches
  if "BRANCH" in op_name:
    return InstDesc("branch", INST, InstOp.JUMP, 1, frozenset(), frozenset(), frozenset(sgpr_r), frozenset(),
                    reads_scc=reads_scc, reads_vcc=reads_vcc, reads_exec=reads_exec)
  # Immediates (s_nop, s_waitcnt*, s_clause, s_sleep, s_setprio)
  if op_name in _IMMEDIATE_OPS:
    return InstDesc("imm", IMMEDIATE, None, 1, frozenset(), frozenset(), frozenset(), frozenset())
  # SMEM
  if isinstance(inst, ir3.SMEM):
    return InstDesc("smem", INST, InstOp.SMEM_RD, 1, frozenset(), frozenset(), frozenset(sgpr_r), frozenset(sgpr_w), mem_dwords=dwords)
  # DS (LDS)
  if isinstance(inst, ir3.DS):
    is_store = "STORE" in op_name or "WRITE" in op_name
    if is_store:
      op = {1: InstOp.LDS_WR_2, 2: InstOp.LDS_WR_3, 3: InstOp.LDS_WR_4, 4: InstOp.LDS_WR_5}.get(dwords, InstOp.LDS_WR_2)
      dur = op.value - InstOp.LDS_WR_1.value + 1
    else:
      op, dur = InstOp.LDS_RD, 1
    return InstDesc("lds", INST, op, dur, frozenset(vgpr_r), frozenset(vgpr_w), frozenset(sgpr_r), frozenset(),
                    is_store=is_store, mem_dwords=dwords)
  # GLOBAL / SCRATCH
  if isinstance(inst, (ir3.GLOBAL, ir3.SCRATCH)):
    is_store = "STORE" in op_name
    saddr = getattr(inst, "saddr", NULL)
    has_saddr = isinstance(saddr, Reg) and saddr.offset != NULL.offset
    if is_store:
      base = InstOp.SGMEM_WR_2.value if has_saddr else InstOp.SGMEM_WR_3.value
      op = InstOp(min(InstOp.SGMEM_WR_6.value, base + dwords - 1))
      dur = op.value - InstOp.SGMEM_WR_2.value + 2
    else:
      op = InstOp.SGMEM_RD_1 if has_saddr else InstOp.SGMEM_RD_2
      dur = 1 if has_saddr else 2
    return InstDesc("vmem", INST, op, dur, frozenset(vgpr_r), frozenset(vgpr_w), frozenset(sgpr_r), frozenset(),
                    is_store=is_store, mem_dwords=dwords)
  # FLAT
  if isinstance(inst, ir3.FLAT):
    is_store = "STORE" in op_name
    if is_store:
      op = InstOp(min(InstOp.FLAT_WR_6.value, InstOp.FLAT_WR_3.value + dwords - 1))
      dur = op.value - InstOp.FLAT_WR_3.value + 3
    else:
      op, dur = InstOp.FLAT_RD_2, 2
    return InstDesc("vmem", INST, op, dur, frozenset(vgpr_r), frozenset(vgpr_w), frozenset(sgpr_r), frozenset(),
                    is_store=is_store, mem_dwords=dwords)
  # SALU (SOP1, SOP2, SOPC, SOPK)
  if isinstance(inst, (ir3.SOP1, ir3.SOP2, ir3.SOPC, ir3.SOPK)):
    op = InstOp.SALU_WR_EXEC if writes_exec else InstOp.SALU
    return InstDesc("salu", INST, op, 1, frozenset(), frozenset(), frozenset(sgpr_r), frozenset(sgpr_w),
                    reads_scc=reads_scc, writes_scc=writes_scc, reads_vcc=reads_vcc, writes_vcc=writes_vcc,
                    reads_exec=reads_exec, writes_exec=writes_exec)
  # VINTERP
  if isinstance(inst, ir3.VINTERP):
    return InstDesc("valu", INST, InstOp.VINTERP, 1, frozenset(vgpr_r), frozenset(vgpr_w), frozenset(sgpr_r), frozenset())
  # VALU variants
  if op_name.startswith("V_CMPX_"):
    return InstDesc("valu", INST, InstOp.VALU1_WR_EXEC, 1, frozenset(vgpr_r), frozenset(vgpr_w), frozenset(sgpr_r), frozenset(),
                    reads_vcc=reads_vcc, writes_exec=True)
  if op_name.startswith("V_WMMA"):
    wmma_dur = 16 if "IU4" in op_name else 32
    return InstDesc("valu", VALUINST, None, wmma_dur, frozenset(vgpr_r), frozenset(vgpr_w), frozenset(sgpr_r), frozenset(), wmma_dur=wmma_dur)
  if op_name.startswith(_TRANSCENDENTAL_PREFIXES) and "F64" not in op_name:
    return InstDesc("tfu", INST, InstOp.VALUT_4, 4, frozenset(vgpr_r), frozenset(vgpr_w), frozenset(sgpr_r), frozenset())
  if any(op_name.startswith(p) for p in _VALUB2_OPS):
    return InstDesc("valu", INST, InstOp.VALUB_2, 2, frozenset(vgpr_r), frozenset(vgpr_w), frozenset(sgpr_r), frozenset())
  if any(op_name.startswith(p) for p in _VALUB4_PREFIXES):
    return InstDesc("valu", INST, InstOp.VALUB_4, 4, frozenset(vgpr_r), frozenset(vgpr_w), frozenset(sgpr_r), frozenset())
  if "F64" in op_name and op_name.startswith("V_"):
    return InstDesc("valu", INST, InstOp.VALUB_16, 16, frozenset(vgpr_r), frozenset(vgpr_w), frozenset(sgpr_r), frozenset())
  is_mov_imm = op_name.startswith("V_MOV_B32") and not vgpr_r and not sgpr_r
  return InstDesc("valu", VALUINST, None, 1, frozenset(vgpr_r), frozenset(vgpr_w), frozenset(sgpr_r), frozenset(sgpr_w),
                  reads_vcc=reads_vcc, writes_vcc=writes_vcc, reads_exec=reads_exec, writes_exec=writes_exec, is_mov_imm=is_mov_imm)

# ═══════════════════════════════════════════════════════════════════════════════
# CYCLE-ACCURATE RDNA3 WGP / SIMD SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimWave:
  wg_id: int
  local_wave: int
  se: int
  sa: int
  wgp: int
  simd: int
  wave_slot: int
  pcs: list[int]                       # dynamic PC stream to execute
  idx: int = 0
  start_time: int = 0
  ready_cycle: int = 0
  at_barrier: bool = False
  done: bool = False
  vgpr_ready: dict[int, int] = field(default_factory=dict)         # vgpr -> exec completion cycle (for VALU exec-stage RAW)
  vgpr_mem_ready: dict[int, int] = field(default_factory=dict)     # vgpr -> cycle ready for LDS/VMEM issue
  sgpr_ready: dict[int, int] = field(default_factory=dict)         # sgpr -> cycle ready for SALU
  sgpr_valu_ready: dict[int, int] = field(default_factory=dict)    # sgpr -> cycle ready for VALU
  scc_ready: int = 0
  vcc_ready: int = 0
  exec_ready: int = 0
  lgkm_done: list[int] = field(default_factory=list)               # sorted completion cycles of outstanding SMEM/LDS ops
  vm_done: list[int] = field(default_factory=list)                 # sorted completion cycles of outstanding VMEM ops
  valu_history: list[int] = field(default_factory=list)            # recent VALU issue cycles (for s_delay_alu)
  trans_history: list[int] = field(default_factory=list)           # recent TFU issue cycles
  salu_history: list[int] = field(default_factory=list)            # recent SALU issue cycles
  delay_pending: int = 0                                           # cycle from s_delay_alu
  pending_skip_delay: tuple[int, int] | None = None                # (skip_count, instid1)

def _build_static_wave_pcs(insts: dict[int, Inst], cfg: dict, block_trace: list[int] | None) -> list[int]:
  """Expand a dynamic block trace (or acyclic static CFG fallback) into a flat list of instruction PCs."""
  if block_trace:
    return [off for bpc in block_trace if bpc in cfg["blocks"] for off in cfg["blocks"][bpc]]
  # fallback for simple acyclic kernels without branch trace
  pcs: list[int] = []
  pc, visited = 0, 0
  while pc in cfg["blocks"] and visited < 10000:
    pcs.extend(cfg["blocks"][pc])
    visited += 1
    nxt = cfg["paths"].get(pc, {})
    if not nxt: break
    # prefer fallthrough if present, else first target
    pc = next((d for d, k in nxt.items() if k == COND_NOT_TAKEN), next(iter(nxt)))
  return pcs

def _parse_waitcnt(inst: Inst) -> tuple[int, int, int]:
  """Return (vmcnt_max, lgkmcnt_max, expcnt_max) thresholds for RDNA3 s_waitcnt*."""
  op = getattr(inst, "op_name", "")
  simm = getattr(inst, "simm16", 0) & 0xFFFF
  if op == "S_WAITCNT":
    expcnt = simm & 0x7
    lgkmcnt = (simm >> 4) & 0x3F
    vmcnt = (simm >> 10) & 0x3F
    return vmcnt, lgkmcnt, expcnt
  if op == "S_WAITCNT_VMCNT": return simm & 0x3F, 63, 7
  if op == "S_WAITCNT_LGKMCNT": return 63, simm & 0x3F, 7
  if op == "S_WAITCNT_EXPCNT": return 63, 63, simm & 0x7
  return 63, 63, 7

def _delay_alu_target(instid: int, w: SimWave, cur: int) -> int:
  """Compute the cycle a wave must wait until for a single s_delay_alu instid."""
  if instid == 0: return cur
  if 1 <= instid <= 4:  # VALU_DEP_1..4
    if len(w.valu_history) >= instid: return w.valu_history[-instid] + VALU_RAW_LAT
  elif 5 <= instid <= 7:  # TRANS32_DEP_1..3
    idx = instid - 4
    if len(w.trans_history) >= idx: return w.trans_history[-idx] + TFU_DISPATCH_TO_EXEC + 4
  elif instid == 8:  # FMA_ACCUM_CYCLE_1
    if w.valu_history: return w.valu_history[-1] + 1
  elif 9 <= instid <= 11:  # SALU_CYCLE_1..3
    if w.salu_history: return w.salu_history[-1] + (instid - 8)
  return cur

def simulate_rdna3_sqtt(code: bytes, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int,
                        n_se: int = 6, itrace_se_mask: int = 0b11, simd_sel: int = 0,
                        block_traces: dict[tuple[int, int], list[int]] | None = None) -> dict[int, bytes]:
  """Run cycle-accurate RDNA3 SQTT simulation and return {se: raw_sqtt_bytes}."""
  insts = amd_decode(code, "rdna3")
  cfg = get_cfg(insts)
  descs = {pc: classify_inst(inst) for pc, inst in insts.items()}

  total_wgs = gx * gy * gz
  n_waves_per_wg = (lx * ly * lz + 31) // 32
  # Distribute workgroups across SEs (round-robin), then SA (0..1), WGP (0..3)
  # Inside a WGP, waves of a WG distribute across SIMDs in hardware order (0, 2, 0, 2, 1, 3, 1, 3)
  simd_order = (0, 2, 0, 2, 1, 3, 1, 3)
  se_waves: dict[int, list[SimWave]] = {se: [] for se in range(n_se)}
  slot_counter: dict[tuple[int, int, int, int], int] = defaultdict(int)

  base_t = 1000
  for wg_id in range(total_wgs):
    se = wg_id % n_se
    wg_in_se = wg_id // n_se
    wgp = (wg_in_se % 4)
    sa = (wg_in_se // 4) % 2
    for lw in range(n_waves_per_wg):
      simd = simd_order[lw % len(simd_order)]
      slot = slot_counter[(se, sa, wgp, simd)] % 16
      slot_counter[(se, sa, wgp, simd)] += 1
      btrace = block_traces.get((wg_id, lw)) if block_traces else None
      pcs = _build_static_wave_pcs(insts, cfg, btrace)
      st = base_t + wg_in_se * 6 + lw
      se_waves[se].append(SimWave(wg_id=wg_id, local_wave=lw, se=se, sa=sa, wgp=wgp, simd=simd, wave_slot=slot,
                                  pcs=pcs, start_time=st, ready_cycle=st + 10))

  blobs: dict[int, bytes] = {}
  for se in range(n_se):
    itrace = bool((itrace_se_mask >> se) & 1)
    pkts = _simulate_se(se, se_waves[se], insts, descs, itrace, simd_sel, n_waves_per_wg)
    blobs[se] = encode(pkts)
  return blobs

def _simulate_se(se: int, waves: list[SimWave], insts: dict[int, Inst], descs: dict[int, InstDesc],
                 itrace: bool, simd_sel: int, n_waves_per_wg: int) -> list[tuple[int, type[PacketType], dict]]:
  events: list[tuple[int, int, type[PacketType], dict]] = []  # (time, seq, cls, fields)
  seq = 0
  def push(t: int, cls: type[PacketType], **fields):
    nonlocal seq
    events.append((t, seq, cls, fields))
    seq += 1

  # 1. LAYOUT_HEADER + initial realtime clock marker
  push(0, LAYOUT_HEADER, layout=3, simd=simd_sel, group=0, sel_a=6, sel_b=0, flag4=0)
  rt_base = 10_000_000_000
  push(100, TS_DELTA_OR_MARK, rt=1, pl=0, delta=rt_base)

  if not waves:
    push(100 + RT_MARKER_PERIOD, TS_DELTA_OR_MARK, rt=1, pl=0, delta=rt_base + int(RT_MARKER_PERIOD * RT_CLK_HZ // SHADER_CLK_HZ))
    return [(t, cls, f) for t, _, cls, f in sorted(events, key=lambda x: (x[0], x[1]))]

  # Separate WGP0/SA0 waves (for full instruction tracing if itrace) vs other WGPs (coarse lifecycle)
  wgp0_waves = [w for w in waves if w.sa == 0 and w.wgp == 0]
  other_waves = [w for w in waves if not (w.sa == 0 and w.wgp == 0)]

  # Emit WAVESTART for all waves
  for w in waves:
    push(w.start_time, WAVESTART, sa=w.sa, simd=w.simd, wgp=w.wgp, wave=w.wave_slot, id7=w.wg_id & 0x3FFF)

  # Simulate WGP0 cycle-by-cycle if itrace is enabled
  max_end = max(w.start_time for w in waves) + 20
  if wgp0_waves:
    if itrace:
      max_end = max(max_end, _simulate_wgp0(wgp0_waves, insts, descs, simd_sel, n_waves_per_wg, push))
    else:
      for w in wgp0_waves:
        et = w.start_time + max(20, len(w.pcs) * 4)
        push(et, WAVEEND, sa=w.sa, simd=w.simd, wgp=w.wgp, wave=w.wave_slot)
        max_end = max(max_end, et)

  # Non-WGP0 waves emit WAVEEND at approximate completion
  for w in other_waves:
    et = max(w.start_time + 20, max_end - 10 + w.local_wave)
    push(et, WAVEEND, sa=w.sa, simd=w.simd, wgp=w.wgp, wave=w.wave_slot)
    max_end = max(max_end, et)

  # Emit realtime clock markers every RT_MARKER_PERIOD shader cycles so sqtt_timeline computes shader freq_hz
  t_mark = 100 + RT_MARKER_PERIOD
  while t_mark <= max_end + RT_MARKER_PERIOD:
    rt_val = rt_base + int((t_mark - 100) * RT_CLK_HZ // SHADER_CLK_HZ)
    push(t_mark, TS_DELTA_OR_MARK, rt=1, pl=0, delta=rt_val)
    t_mark += RT_MARKER_PERIOD

  events.sort(key=lambda x: (x[0], x[1]))
  return [(t, cls, f) for t, _, cls, f in events]

def _simulate_wgp0(waves: list[SimWave], insts: dict[int, Inst], descs: dict[int, InstDesc],
                   simd_sel: int, n_waves_per_wg: int, push) -> int:
  # Per-SIMD wave lists and pipe busy cycles
  simd_waves: dict[int, list[SimWave]] = defaultdict(list)
  wg_waves: dict[int, list[SimWave]] = defaultdict(list)
  for w in waves:
    simd_waves[w.simd].append(w)
    wg_waves[w.wg_id].append(w)

  # Pipe availability clocks (earliest cycle next op can execute on that pipe)
  valu_free: dict[int, int] = defaultdict(int)
  tfu_free: dict[int, int] = defaultdict(int)
  salu_free: dict[int, int] = defaultdict(int)
  lds_free = 0
  vmem_free = 0

  # Pending exec events on traced SIMD to merge simultaneous VALU+SALU -> VALU_SALU
  alu_execs: dict[int, set[str]] = defaultdict(set)
  vmem_execs: list[tuple[int, MemSrc]] = []
  imm_retires: dict[int, list[int]] = defaultdict(list)  # cycle -> list of wave_slots retiring IMMEDIATE on traced SIMD

  cur = min(w.ready_cycle for w in waves)
  active = [w for w in waves if not w.done]
  max_cycles = 2_000_000

  while active and cur < max_cycles:
    # Check barrier releases per workgroup
    for wg_id, wlist in wg_waves.items():
      if any(w.at_barrier for w in wlist) and all(w.at_barrier or w.done for w in wlist):
        rel_t = cur + BARRIER_RELEASE_LAT
        woke_mask = 0
        for w in wlist:
          if w.at_barrier:
            w.at_barrier = False
            w.ready_cycle = rel_t
            if w.simd == simd_sel: woke_mask |= (1 << w.wave_slot)
        # WAVERDY for waking waves on traced SIMD (when >1 wave or non-0 wave wakes)
        if woke_mask & ~1: push(rel_t + 1, WAVERDY, mask=woke_mask & ~1)

    next_wakeup = max_cycles
    issued_any = False

    for simd, swaves in simd_waves.items():
      live = [w for w in swaves if not w.done]
      if not live: continue
      is_traced = (simd == simd_sel)

      # First, consume any zero-packet s_delay_alu instructions on ready waves
      for w in live:
        while not w.at_barrier and w.idx < len(w.pcs) and cur >= w.ready_cycle:
          pc = w.pcs[w.idx]
          desc = descs[pc]
          if desc.kind != "delay": break
          inst = insts[pc]
          simm = getattr(inst, "simm16", 0) & 0xFFFF
          id0, skip, id1 = simm & 0xF, (simm >> 4) & 0x7, (simm >> 7) & 0xF
          w.delay_pending = max(w.delay_pending, _delay_alu_target(id0, w, cur))
          if id1: w.pending_skip_delay = (skip, id1)
          w.idx += 1

      # Retire ready IMMEDIATE instructions (s_nop, s_waitcnt, s_clause, s_sleep) in parallel across waves on the SIMD
      imm_retired: set[int] = set()
      for w in live:
        if w.at_barrier or w.idx >= len(w.pcs): continue
        pc = w.pcs[w.idx]
        desc = descs[pc]
        if desc.kind != "imm": continue
        inst = insts[pc]
        req = _wave_ready_cycle(w, inst, desc, cur)
        if req <= cur:
          w.idx += 1
          opn = getattr(inst, "op_name", "")
          if opn == "S_NOP": w.ready_cycle = cur + (getattr(inst, "simm16", 0) & 0xF) + 1
          elif opn == "S_SLEEP": w.ready_cycle = cur + max(1, (getattr(inst, "simm16", 0) & 0x7F) * 64)
          else: w.ready_cycle = cur + 1
          if is_traced: imm_retires[cur].append(w.wave_slot)
          imm_retired.add(w.wave_slot)
          issued_any = True
        elif req < next_wakeup:
          next_wakeup = req

      # Oldest-wave-first (age-based) arbitration on this SIMD (matches RDNA3 hardware: wave 0 issues back-to-back until it stalls)
      picked: SimWave | None = None
      for w in live:
        if w.at_barrier or w.wave_slot in imm_retired: continue
        if w.idx >= len(w.pcs):
          w.done = True
          push(cur, WAVEEND, sa=w.sa, simd=w.simd, wgp=w.wgp, wave=w.wave_slot)
          continue
        desc = descs[w.pcs[w.idx]]
        if desc.kind == "imm": continue
        req = _wave_ready_cycle(w, insts[w.pcs[w.idx]], desc, cur)
        if req <= cur:
          picked = w
          break
        if req < next_wakeup: next_wakeup = req

      if picked is None: continue
      issued_any = True
      w = picked
      pc = w.pcs[w.idx]
      inst = insts[pc]
      desc = descs[pc]
      w.idx += 1

      # Apply deferred s_delay_alu instid1 if skip counter hits 0
      if w.pending_skip_delay is not None:
        sk, id1 = w.pending_skip_delay
        if sk <= 0:
          w.delay_pending = max(w.delay_pending, _delay_alu_target(id1, w, cur))
          w.pending_skip_delay = None
        else:
          w.pending_skip_delay = (sk - 1, id1)

      # Execute instruction & schedule SQTT packets
      if desc.kind == "endpgm":
        w.done = True
        push(cur, WAVEEND, sa=w.sa, simd=w.simd, wgp=w.wgp, wave=w.wave_slot)
      elif desc.kind == "barrier":
        w.at_barrier = True
        if is_traced: push(cur, INST, wave=w.wave_slot, op=InstOp.BARRIER)
      elif desc.kind == "imm":
        opn = getattr(inst, "op_name", "")
        if opn == "S_NOP": w.ready_cycle = cur + (getattr(inst, "simm16", 0) & 0xF) + 1
        elif opn == "S_SLEEP": w.ready_cycle = cur + max(1, (getattr(inst, "simm16", 0) & 0x7F) * 64)
        else: w.ready_cycle = cur + 1
        if is_traced: imm_retires[cur].append(w.wave_slot)
      elif desc.kind == "msg":
        w.ready_cycle = cur + 1
        if is_traced: push(cur, INST, wave=w.wave_slot, op=InstOp.MESSAGE)
      elif desc.kind == "branch":
        nxt_pc = w.pcs[w.idx] if w.idx < len(w.pcs) else -1
        taken = (nxt_pc != pc + inst.size())
        w.ready_cycle = cur + (BRANCH_TAKEN_LAT if taken else 1)
        if is_traced: push(cur, INST, wave=w.wave_slot, op=InstOp.JUMP if taken else InstOp.JUMP_NO)
      elif desc.kind == "salu":
        exec_t = max(cur + SALU_DISPATCH_TO_EXEC, salu_free[simd])
        salu_free[simd] = exec_t + 1
        w.salu_history.append(cur)
        for r in desc.sgpr_w:
          w.sgpr_ready[r] = cur + SALU_RAW_LAT
          w.sgpr_valu_ready[r] = cur + SALU_TO_VALU_RAW_LAT
        if desc.writes_scc: w.scc_ready = cur + SALU_SCC_RAW_LAT
        if desc.writes_vcc: w.vcc_ready = cur + SALU_RAW_LAT
        if desc.writes_exec: w.exec_ready = cur + SALU_TO_VALU_RAW_LAT
        w.ready_cycle = cur + 1
        if is_traced:
          push(cur, INST, wave=w.wave_slot, op=desc.inst_op)
          alu_execs[exec_t].add("SALU")
      elif desc.kind == "smem":
        exec_t = max(cur + SALU_DISPATCH_TO_EXEC, salu_free[simd])
        salu_free[simd] = exec_t + 1
        done_t = exec_t + SMEM_RETURN_LAT
        w.lgkm_done.append(done_t)
        w.ready_cycle = cur + 1
        if is_traced:
          push(cur, INST, wave=w.wave_slot, op=InstOp.SMEM_RD)
          alu_execs[exec_t].add("SALU")
      elif desc.kind == "tfu":
        exec_t = max(cur + TFU_DISPATCH_TO_EXEC, tfu_free[simd])
        tfu_free[simd] = exec_t + 4
        w.trans_history.append(cur)
        for r in desc.vgpr_w:
          w.vgpr_ready[r] = exec_t + 4
          w.vgpr_mem_ready[r] = exec_t + 12
        w.ready_cycle = cur + 1
        if is_traced:
          push(cur, INST, wave=w.wave_slot, op=InstOp.VALUT_4)
          alu_execs[exec_t].add("VALU")
      elif desc.kind == "valu":
        base_lat = VALU_MOV_IMM_TO_EXEC if desc.is_mov_imm else VALU_DISPATCH_TO_EXEC
        dur = desc.pipe_dur
        exec_start = max(cur + base_lat - 1, valu_free[simd])
        exec_end = exec_start + dur
        valu_free[simd] = exec_end
        w.valu_history.append(cur)
        raw_rdy = cur + (desc.wmma_dur + 4 if desc.wmma_dur else VALU_RAW_LAT)
        mem_rdy = exec_end + 10
        for r in desc.vgpr_w:
          w.vgpr_ready[r] = max(w.vgpr_ready.get(r, 0), raw_rdy)
          w.vgpr_mem_ready[r] = max(w.vgpr_mem_ready.get(r, 0), mem_rdy)
        if desc.writes_vcc: w.vcc_ready = max(w.vcc_ready, exec_end)
        if desc.writes_exec: w.exec_ready = max(w.exec_ready, cur + VALU_EXEC_WR_TO_SALU_LAT)
        w.ready_cycle = cur + 1
        if is_traced:
          if desc.pkt_cls is VALUINST: push(cur, VALUINST, wave=w.wave_slot, flag=0)
          else: push(cur, INST, wave=w.wave_slot, op=desc.inst_op)
          alu_execs[exec_end].add("VALU")
      elif desc.kind == "lds":
        exec_t = max(cur + LDS_DISPATCH_TO_EXEC, lds_free)
        lds_free = exec_t + desc.pipe_dur
        ret_lat = LDS_RETURN_LAT_128 if desc.mem_dwords >= 4 else (LDS_STORE_RETURN_LAT_32 if desc.is_store else LDS_LOAD_RETURN_LAT_32)
        w.lgkm_done.append(exec_t + ret_lat)
        w.ready_cycle = cur + 1
        if is_traced:
          push(cur, INST, wave=w.wave_slot, op=desc.inst_op)
          vmem_execs.append((exec_t, MemSrc.LDS))
        else:
          other_op = InstOp(min(0x54, 0x50 + desc.pipe_dur - 1))
          push(cur, INST, wave=w.wave_slot, op=other_op)
          vmem_execs.append((exec_t, MemSrc.LDS_ALT))
      elif desc.kind == "vmem":
        exec_t = max(cur + VMEM_DISPATCH_TO_EXEC, vmem_free)
        vmem_free = exec_t + desc.pipe_dur
        w.vm_done.append(exec_t + VMEM_RETURN_LAT)
        w.ready_cycle = cur + 1
        if is_traced:
          push(cur, INST, wave=w.wave_slot, op=desc.inst_op)
          vmem_execs.append((exec_t, MemSrc.VMEM))
        else:
          other_op = InstOp(min(0x66, 0x5a + desc.pipe_dur - 1))
          push(cur, INST, wave=w.wave_slot, op=other_op)
          vmem_execs.append((exec_t, MemSrc.VMEM_ALT))

    active = [w for w in waves if not w.done]
    if not active: break
    if issued_any: cur += 1
    elif next_wakeup < max_cycles: cur = max(cur + 1, next_wakeup)
    else: cur += 1

  # Emit merged IMMEDIATE / IMMEDIATE_MASK packets
  for t, wslots in imm_retires.items():
    if len(wslots) == 1: push(t, IMMEDIATE, wave=wslots[0])
    else:
      mask = 0
      for ws in wslots: mask |= (1 << ws)
      push(t, IMMEDIATE_MASK, mask=mask)

  # Emit merged ALUEXEC packets (VALU + SALU on same cycle -> VALU_SALU)
  for t, srcs in alu_execs.items():
    if "VALU" in srcs and "SALU" in srcs: push(t, ALUEXEC, src=AluSrc.VALU_SALU)
    elif "VALU" in srcs: push(t, ALUEXEC, src=AluSrc.VALU)
    elif "SALU" in srcs: push(t, ALUEXEC, src=AluSrc.SALU)

  # Emit VMEMEXEC packets
  for t, msrc in vmem_execs: push(t, VMEMEXEC, src=msrc)

  return max([cur] + list(alu_execs.keys()) + [t for t, _ in vmem_execs])

def _wave_ready_cycle(w: SimWave, inst: Inst, desc: InstDesc, cur: int) -> int:
  req = max(w.ready_cycle, w.delay_pending)
  # s_endpgm waits for outstanding memory ops to finish draining before wave termination
  if desc.kind == "endpgm":
    if w.vm_done: req = max(req, max(w.vm_done))
    if w.lgkm_done: req = max(req, max(w.lgkm_done))
    return req
  # Check s_waitcnt*
  if desc.kind == "imm" and getattr(inst, "op_name", "").startswith("S_WAITCNT"):
    vm_max, lgkm_max, _ = _parse_waitcnt(inst)
    w.vm_done = [t for t in w.vm_done if t > cur]
    w.lgkm_done = [t for t in w.lgkm_done if t > cur]
    if len(w.vm_done) > vm_max: req = max(req, sorted(w.vm_done)[len(w.vm_done) - vm_max - 1])
    if len(w.lgkm_done) > lgkm_max: req = max(req, sorted(w.lgkm_done)[len(w.lgkm_done) - lgkm_max - 1])
    return req
  # Check register RAW hazards at issue stage
  # Note: on RDNA3, VALU VGPR RAW hazards enforce 5-cycle issue spacing, while VCC carry-in (v_add_co_ci)
  # forwards without issue stall. SGPR/EXEC hazards and VALU->LDS/VMEM hazards stall at issue.
  if desc.kind in ("valu", "tfu"):
    for r in desc.vgpr_r: req = max(req, w.vgpr_ready.get(r, 0))
    for r in desc.sgpr_r: req = max(req, w.sgpr_valu_ready.get(r, 0))
    if desc.reads_exec: req = max(req, w.exec_ready)
  elif desc.kind in ("lds", "vmem"):
    for r in desc.vgpr_r: req = max(req, w.vgpr_mem_ready.get(r, 0))
    for r in desc.sgpr_r: req = max(req, w.sgpr_valu_ready.get(r, 0))
  elif desc.kind in ("salu", "smem", "branch"):
    for r in desc.sgpr_r: req = max(req, w.sgpr_ready.get(r, 0))
    if desc.reads_scc: req = max(req, w.scc_ready)
    if desc.reads_vcc: req = max(req, w.vcc_ready)
    if desc.reads_exec: req = max(req, w.exec_ready)
  return req
