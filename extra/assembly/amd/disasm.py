# RDNA3/RDNA4/CDNA disassembler
from __future__ import annotations
import re, struct
from extra.assembly.amd.dsl import Inst, Reg, src

# CDNA-specific special registers (not in Reg._NAMES)
_CDNA_GPRS = {102: 'flat_scratch_lo', 103: 'flat_scratch_hi', 104: 'xnack_mask_lo', 105: 'xnack_mask_hi'}
_CDNA_PAIRS = {102: 'flat_scratch', 104: 'xnack_mask'}

def _d(r: Reg, cdna: bool = False) -> str:
  """Format register for disasm output. Handles CDNA-specific registers."""
  if cdna and r.sz == 1 and r.offset in _CDNA_GPRS: return _CDNA_GPRS[r.offset]
  if cdna and r.sz == 2 and r.offset in _CDNA_PAIRS: return _CDNA_PAIRS[r.offset]
  return r.disasm

def _off(v) -> int: return v.offset if isinstance(v, Reg) else v
def _vi(v) -> int: off = _off(v); return off - 256 if off >= 256 else off

def _lit(inst, v, neg=0) -> str:
  """Format literal/inline constant value."""
  off = _off(v)
  if off == 255: return f"-0x{inst._literal:x}" if neg else (f"0x{inst._literal:x}" if inst._literal else "0")
  s = src[off].disasm
  return f"-{s}" if neg else s

# Backwards compatibility aliases
def decode_src(v, cdna: bool = False) -> str: return _d(src[_off(v)], cdna)
def _unwrap(v) -> int: return _off(v)

# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION METADATA - fallback functions when inst.num_srcs()/inst.operands unavailable
# ═══════════════════════════════════════════════════════════════════════════════

def _num_srcs(inst) -> int:
  """Fallback: get number of source operands from instruction name."""
  name = getattr(inst, 'op_name', '') or ''
  n = name.upper()
  # FMAC/MAC ops are 2-source (dst is implicit accumulator), but FMA/MAD ops are 3-source
  if 'FMAC' in n or 'V_MAC_' in n: return 2
  if any(x in n for x in ('FMA', 'MAD', 'CNDMASK', 'BFE', 'BFI', 'LERP', 'MED3', 'SAD', 'DIV_FMAS', 'DIV_FIXUP', 'DIV_SCALE', 'CUBE')): return 3
  # PERMLANE_VAR ops are 2-source, but PERMLANE (non-VAR) are 3-source
  if 'PERMLANE' in n and '_VAR' not in n: return 3
  if any(x in n for x in ('_ADD3', '_LSHL_ADD', '_ADD_LSHL', '_LSHL_OR', '_AND_OR', 'OR3_B32', 'AND_OR_B32', 'ALIGNBIT', 'ALIGNBYTE', 'V_PERM_', 'XOR3', 'XAD', 'MULLIT', 'MINMAX', 'MAXMIN', 'MINIMUMMAXIMUM', 'MAXIMUMMINIMUM', 'MINIMUM3', 'MAXIMUM3', 'MIN3', 'MAX3', 'DOT2', 'CVT_PK_U8_F32', 'DOT4', 'DOT8', 'WMMA', 'SWMMAC')): return 3
  return 2

# SWMMAC register counts: (dst, src0, src1, src2)
def _swmmac_regs(name: str) -> tuple[int, int, int, int]:
  """Return (dst, src0, src1, src2) register counts for SWMMAC instructions."""
  if 'f16_16x16x32' in name or 'bf16_16x16x32' in name: return (4, 4, 8, 1)
  if 'f32_16x16x32_f16' in name or 'f32_16x16x32_bf16' in name: return (8, 4, 8, 1)
  if 'i32_16x16x32_iu4' in name: return (8, 1, 2, 1)
  if 'i32_16x16x64_iu4' in name: return (8, 2, 4, 1)
  if 'i32_16x16x32_iu8' in name or 'f32_16x16x32_fp8' in name or 'f32_16x16x32_bf8' in name: return (8, 2, 4, 1)
  return (8, 8, 8, 8)

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

from extra.assembly.amd.autogen.rdna3.ins import (VOP1, VOP1_SDST, VOP2, VOP3, VOP3_SDST, VOP3SD, VOP3P, VOPC, VOPD, VINTERP, SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, DS, FLAT, GLOBAL, SCRATCH, MUBUF, MTBUF, MIMG, EXP,
  VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOPDOp, SOP1Op, SOPKOp, SOPPOp, SMEMOp, DSOp, MUBUFOp)
from extra.assembly.amd.autogen.rdna4.ins import (VOP1 as R4_VOP1, VOP1_SDST as R4_VOP1_SDST, VOP2 as R4_VOP2, VOP3 as R4_VOP3, VOP3_SDST as R4_VOP3_SDST, VOP3SD as R4_VOP3SD, VOP3P as R4_VOP3P,
  VOPC as R4_VOPC, VOPD as R4_VOPD, VINTERP as R4_VINTERP, SOP1 as R4_SOP1, SOP2 as R4_SOP2, SOPC as R4_SOPC, SOPK as R4_SOPK, SOPP as R4_SOPP,
  SMEM as R4_SMEM, DS as R4_DS, VBUFFER as R4_VBUFFER, VEXPORT as R4_VEXPORT, VOPDOp as R4_VOPDOp)
from extra.assembly.amd.autogen.cdna.ins import FLAT as C_FLAT, MUBUF as C_MUBUF, MTBUF as C_MTBUF

def _is_cdna(inst: Inst) -> bool: return 'cdna' in inst.__class__.__module__

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

HWREG = {1: 'HW_REG_MODE', 2: 'HW_REG_STATUS', 3: 'HW_REG_TRAPSTS', 4: 'HW_REG_HW_ID', 5: 'HW_REG_GPR_ALLOC',
         6: 'HW_REG_LDS_ALLOC', 7: 'HW_REG_IB_STS', 15: 'HW_REG_SH_MEM_BASES', 18: 'HW_REG_PERF_SNAPSHOT_PC_LO',
         19: 'HW_REG_PERF_SNAPSHOT_PC_HI', 20: 'HW_REG_FLAT_SCR_LO', 21: 'HW_REG_FLAT_SCR_HI', 22: 'HW_REG_XNACK_MASK',
         23: 'HW_REG_HW_ID1', 24: 'HW_REG_HW_ID2', 25: 'HW_REG_POPS_PACKER', 28: 'HW_REG_IB_STS2'}
HWREG_RDNA4 = {1: 'HW_REG_MODE', 2: 'HW_REG_STATUS', 4: 'HW_REG_STATE_PRIV', 5: 'HW_REG_GPR_ALLOC',
               6: 'HW_REG_LDS_ALLOC', 7: 'HW_REG_IB_STS', 10: 'HW_REG_PERF_SNAPSHOT_DATA', 11: 'HW_REG_PERF_SNAPSHOT_PC_LO',
               12: 'HW_REG_PERF_SNAPSHOT_PC_HI', 15: 'HW_REG_PERF_SNAPSHOT_DATA1', 16: 'HW_REG_PERF_SNAPSHOT_DATA2',
               17: 'HW_REG_EXCP_FLAG_PRIV', 18: 'HW_REG_EXCP_FLAG_USER', 19: 'HW_REG_TRAP_CTRL',
               20: 'HW_REG_SCRATCH_BASE_LO', 21: 'HW_REG_SCRATCH_BASE_HI', 23: 'HW_REG_HW_ID1',
               24: 'HW_REG_HW_ID2', 26: 'HW_REG_SCHED_MODE', 29: 'HW_REG_SHADER_CYCLES_LO',
               30: 'HW_REG_SHADER_CYCLES_HI', 31: 'HW_REG_DVGPR_ALLOC_LO', 32: 'HW_REG_DVGPR_ALLOC_HI'}
MSG = {128: 'MSG_RTN_GET_DOORBELL', 129: 'MSG_RTN_GET_DDID', 130: 'MSG_RTN_GET_TMA',
       131: 'MSG_RTN_GET_REALTIME', 132: 'MSG_RTN_SAVE_WAVE', 133: 'MSG_RTN_GET_TBA',
       134: 'MSG_RTN_GET_TBA_TO_PC', 135: 'MSG_RTN_GET_SE_AID_ID'}
# CDNA opcode name aliases for disasm (new name -> old name expected by tests)
_CDNA_DISASM_ALIASES = {'v_fmac_f64': 'v_mul_legacy_f32', 'v_dot2c_f32_bf16': 'v_mac_f32', 'v_fmamk_f32': 'v_madmk_f32', 'v_fmaak_f32': 'v_madak_f32'}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _sreg(b, n: int = 1) -> str: off = _off(b); return src[off:off+n-1].disasm if n > 1 else f"s{off}"
def _vreg(b, n: int = 1) -> str: off = _off(b); idx = off - 256 if off >= 256 else off; return f"v{idx}" if n == 1 else f"v[{idx}:{idx+n-1}]"
def _areg(b, n: int = 1) -> str: off = _off(b); idx = off - 256 if off >= 256 else off; return f"a{idx}" if n == 1 else f"a[{idx}:{idx+n-1}]"
def _ttmp(b, n: int = 1) -> str: off = _off(b); return (f"ttmp{off-108}" if n == 1 else f"ttmp[{off-108}:{off-108+n-1}]") if 108 <= off <= 123 else None
def _sreg_or_ttmp(b, n: int = 1) -> str: return _ttmp(b, n) or _sreg(b, n)
def _fmt_sdst(v, n: int = 1, cdna: bool = False) -> str: return _d(src[_off(v):_off(v)+n-1] if n > 1 else src[_off(v)], cdna)
def _fmt_src(v, n: int = 1, cdna: bool = False) -> str:
  off = _off(v)
  if off == 253: return "src_scc"
  return _d(src[off:off+n-1] if n > 1 else src[off], cdna)

def _fmt_v16(v, base: int = 256, hi_thresh: int = 384) -> str:
  off = _off(v)
  return f"v{(off - base) & 0x7f}.{'h' if off >= hi_thresh else 'l'}"

def _has(op: str, *subs) -> bool: return any(s in op for s in subs)
def _omod(v: int) -> str: return {1: " mul:2", 2: " mul:4", 3: " div:2"}.get(v, "")
def _src16(inst, v) -> str: off = _off(v); return _fmt_v16(off) if off >= 256 else _lit(inst, v)
def _mods(*pairs) -> str: return " ".join(m for c, m in pairs if c)
def _fmt_bits(label: str, val: int, count: int) -> str: return f"{label}:[{','.join(str((val >> i) & 1) for i in range(count))}]"

def _vop3_src(inst, v, neg: int, abs_: int, hi: int, n: int, f16: bool) -> str:
  """Format VOP3 source operand with modifiers."""
  off = _off(v)
  if off == 255: s = _lit(inst, v)
  elif n > 1: s = _fmt_src(v, n)
  elif f16 and off >= 256: s = f"v{off - 256}.{'h' if hi else 'l'}"
  elif off == 253: s = "src_scc"
  else: s = _lit(inst, v)
  if abs_: s = f"|{s}|"
  return f"-{s}" if neg else s

def _opsel_str(opsel: int, n: int, need: bool, is16_d: bool) -> str:
  """Format op_sel modifier string."""
  if not need: return ""
  dst_hi = (opsel >> 3) & 1
  if n == 1: return f" op_sel:[{opsel & 1},{dst_hi}]"
  if n == 2: return f" op_sel:[{opsel & 1},{(opsel >> 1) & 1},{dst_hi}]"
  return f" op_sel:[{opsel & 1},{(opsel >> 1) & 1},{(opsel >> 2) & 1},{dst_hi}]"

# ═══════════════════════════════════════════════════════════════════════════════
# DISASSEMBLER
# ═══════════════════════════════════════════════════════════════════════════════

def _disasm_vop1(inst: VOP1) -> str:
  name, cdna = inst.op_name.lower() or f'vop1_op_{inst.op}', _is_cdna(inst)
  name = name.replace('_e32', '')  # Strip _e32 suffix
  if any(x in name for x in ('v_nop', 'v_pipeflush', 'v_clrexcp')): return name  # no operands
  if 'readfirstlane' in name:
    src = _vreg(inst.src0) if _unwrap(inst.src0) >= 256 else decode_src(_unwrap(inst.src0), cdna)
    vdst_raw = _unwrap(inst.vdst)
    return f"{name} {_fmt_sdst(vdst_raw - 256 if vdst_raw >= 256 else vdst_raw, 1, cdna)}, {src}"
  # Use get_field_bits for register sizes and 16-bit detection
  bits = inst.canonical_op_bits
  dregs, sregs = max(1, bits['d'] // 32), max(1, bits['s0'] // 32)
  is16_dst, is16_src = not cdna and bits['d'] == 16, not cdna and bits['s0'] == 16
  # v_cvt_pk_f32_fp8/bf8: pcode has None dst type but outputs 2 VGPRs
  if 'cvt_pk_f32_fp8' in name or 'cvt_pk_f32_bf8' in name: dregs, is16_src = 2, True
  # Format dst
  if is16_dst:
    vdst = _unwrap(inst.vdst) - 256
    dst = f"v{vdst & 0x7f}.{'h' if vdst >= 128 else 'l'}"
  else:
    dst = _vreg(inst.vdst, dregs)
  # Format src
  src0 = _unwrap(inst.src0)
  if src0 == 255: src = _lit(inst, inst.src0)
  elif is16_src and src0 >= 256:
    s = src0 - 256
    src = f"v{s & 0x7f}.{'h' if s >= 128 else 'l'}"
  elif sregs > 1: src = _fmt_src(inst.src0, sregs, cdna)
  else: src = _lit(inst, inst.src0)
  return f"{name} {dst}, {src}"

_VOP2_CARRY_OUT = {'v_add_co_u32', 'v_sub_co_u32', 'v_subrev_co_u32'}  # carry out only
_VOP2_CARRY_INOUT = {'v_addc_co_u32', 'v_subb_co_u32', 'v_subbrev_co_u32'}  # carry in and out (CDNA)
_VOP2_CARRY_INOUT_RDNA = {'v_add_co_ci_u32', 'v_sub_co_ci_u32', 'v_subrev_co_ci_u32'}  # carry in and out (RDNA)
def _disasm_vop2(inst: VOP2) -> str:
  name, cdna = inst.op_name.lower(), _is_cdna(inst)
  if cdna: name = _CDNA_DISASM_ALIASES.get(name, name)  # apply CDNA aliases
  suf = "" if cdna or name.endswith('_e32') or (not cdna and inst.op == VOP2Op.V_DOT2ACC_F32_F16_E32) else "_e32"
  lit = getattr(inst, '_literal', None)
  # Use get_field_bits for 16-bit detection
  bits = inst.canonical_op_bits
  is16 = not cdna and bits['d'] == 16
  # fmaak/madak: dst = src0 * vsrc1 + K, fmamk/madmk: dst = src0 * K + vsrc1
  if 'fmaak' in name or 'madak' in name or (not cdna and inst.op in (VOP2Op.V_FMAAK_F32_E32, VOP2Op.V_FMAAK_F16_E32)):
    if lit is None: return f"op_{inst.op.value if hasattr(inst.op, 'value') else inst.op}"
    if is16: return f"{name}{suf} {_fmt_v16(inst.vdst)}, {_src16(inst, inst.src0)}, {_fmt_v16(inst.vsrc1)}, 0x{lit:x}"
    return f"{name}{suf} {_vreg(inst.vdst)}, {_lit(inst, inst.src0)}, {_vreg(inst.vsrc1)}, 0x{lit:x}"
  if 'fmamk' in name or 'madmk' in name or (not cdna and inst.op in (VOP2Op.V_FMAMK_F32_E32, VOP2Op.V_FMAMK_F16_E32)):
    if lit is None: return f"op_{inst.op.value if hasattr(inst.op, 'value') else inst.op}"
    if is16: return f"{name}{suf} {_fmt_v16(inst.vdst)}, {_src16(inst, inst.src0)}, 0x{lit:x}, {_fmt_v16(inst.vsrc1)}"
    return f"{name}{suf} {_vreg(inst.vdst)}, {_lit(inst, inst.src0)}, 0x{lit:x}, {_vreg(inst.vsrc1)}"
  if is16: return f"{name}{suf} {_fmt_v16(inst.vdst)}, {_src16(inst, inst.src0)}, {_fmt_v16(inst.vsrc1)}"
  vcc = "vcc" if cdna else "vcc_lo"
  # CDNA carry ops output vcc after vdst
  if cdna and name in _VOP2_CARRY_OUT: return f"{name}{suf} {_vreg(inst.vdst)}, {vcc}, {_lit(inst, inst.src0)}, {_vreg(inst.vsrc1)}"
  if cdna and name in _VOP2_CARRY_INOUT: return f"{name}{suf} {_vreg(inst.vdst)}, {vcc}, {_lit(inst, inst.src0)}, {_vreg(inst.vsrc1)}, {vcc}"
  # RDNA carry-in/out ops: v_add_co_ci_u32, etc.
  if not cdna and name in _VOP2_CARRY_INOUT_RDNA: return f"{name}{suf} {_vreg(inst.vdst)}, {vcc}, {_lit(inst, inst.src0)}, {_vreg(inst.vsrc1)}, {vcc}"
  # Use get_field_bits for register sizes
  regs = inst.canonical_op_regs
  dn, sn0, sn1 = regs.get('d', 1), regs.get('s0', 1), regs.get('s1', 1)
  if dn > 1 or sn0 > 1 or sn1 > 1:
    dst = _vreg(inst.vdst, dn)
    src0 = _lit(inst, inst.src0) if _unwrap(inst.src0) == 255 else _fmt_src(inst.src0, sn0, cdna)
    src1 = _vreg(inst.vsrc1, sn1)
    return f"{name.replace('_e32', '')} {dst}, {src0}, {src1}"
  return f"{name}{suf} {_vreg(inst.vdst)}, {_lit(inst, inst.src0)}, {_vreg(inst.vsrc1)}" + (f", {vcc}" if name == 'v_cndmask_b32' else "")

def _disasm_vopc(inst: VOPC) -> str:
  name, cdna = inst.op_name.lower(), _is_cdna(inst)
  # Use get_field_bits for register sizes and 16-bit detection
  bits = inst.canonical_op_bits
  r0, r1 = max(1, bits['s0'] // 32), max(1, bits['s1'] // 32)
  is16 = bits['s0'] == 16
  if cdna:
    s0 = _lit(inst, inst.src0) if _unwrap(inst.src0) == 255 else _fmt_src(inst.src0, r0, cdna)
    s1 = _vreg(inst.vsrc1, r1) if r1 > 1 else _vreg(inst.vsrc1)
    return f"{name} vcc, {s0}, {s1}"  # CDNA VOPC always outputs vcc
  # RDNA: v_cmpx_* writes to exec (no vcc), v_cmp_* writes to vcc_lo
  has_vcc = 'cmpx' not in name
  s0 = _lit(inst, inst.src0) if _unwrap(inst.src0) == 255 else _fmt_src(inst.src0, r0) if r0 > 1 else _src16(inst, _unwrap(inst.src0)) if is16 else _lit(inst, inst.src0)
  s1 = _vreg(inst.vsrc1, r1) if r1 > 1 else _fmt_v16(inst.vsrc1) if is16 else _vreg(inst.vsrc1)
  suf = "" if name.endswith('_e32') else "_e32"
  return f"{name}{suf} vcc_lo, {s0}, {s1}" if has_vcc else f"{name}{suf} {s0}, {s1}"

NO_ARG_SOPP = {SOPPOp.S_BARRIER, SOPPOp.S_WAKEUP, SOPPOp.S_ICACHE_INV,
               SOPPOp.S_WAIT_IDLE, SOPPOp.S_ENDPGM_SAVED, SOPPOp.S_CODE_END, SOPPOp.S_ENDPGM_ORDERED_PS_DONE, SOPPOp.S_TTRACEDATA}
_CDNA_NO_ARG_SOPP = {'s_endpgm', 's_barrier', 's_wakeup', 's_icache_inv', 's_ttracedata', 's_nop', 's_sethalt', 's_sleep',
                     's_setprio', 's_trap', 's_incperflevel', 's_decperflevel', 's_sendmsg', 's_sendmsghalt'}

def _disasm_sopp(inst: SOPP) -> str:
  name, cdna = inst.op_name.lower(), _is_cdna(inst)
  is_rdna4 = 'rdna4' in inst.__class__.__module__
  # Ops that have no argument when simm16 == 0
  no_arg_zero = {'s_barrier', 's_wakeup', 's_icache_inv', 's_ttracedata', 's_wait_idle', 's_endpgm_saved',
                 's_endpgm_ordered_ps_done', 's_code_end'}
  if name in no_arg_zero: return name if inst.simm16 == 0 else f"{name} {inst.simm16}"
  if name == 's_endpgm': return name if inst.simm16 == 0 else f"{name} {inst.simm16}"
  if cdna:
    if name == 's_waitcnt':
      vm, lgkm, exp = inst.simm16 & 0xf, (inst.simm16 >> 8) & 0x3f, (inst.simm16 >> 4) & 0x7
      p = [f"vmcnt({vm})" if vm != 0xf else "", f"expcnt({exp})" if exp != 7 else "", f"lgkmcnt({lgkm})" if lgkm != 0x3f else ""]
      return f"s_waitcnt {' '.join(x for x in p if x) or '0'}"
    if name.startswith(('s_cbranch', 's_branch')): return f"{name} {inst.simm16}"
    return f"{name} 0x{inst.simm16:x}" if inst.simm16 else name
  # RDNA (use name-based checks instead of enum-based for cross-arch compatibility)
  if name == 's_waitcnt':
    if is_rdna4:
      return f"{name} {inst.simm16}" if inst.simm16 else f"{name} 0"
    vm, exp, lgkm = (inst.simm16 >> 10) & 0x3f, inst.simm16 & 0xf, (inst.simm16 >> 4) & 0x3f
    p = [f"vmcnt({vm})" if vm != 0x3f else "", f"expcnt({exp})" if exp != 7 else "", f"lgkmcnt({lgkm})" if lgkm != 0x3f else ""]
    return f"s_waitcnt {' '.join(x for x in p if x) or '0'}"
  if name == 's_delay_alu':
    deps = ['VALU_DEP_1','VALU_DEP_2','VALU_DEP_3','VALU_DEP_4','TRANS32_DEP_1','TRANS32_DEP_2','TRANS32_DEP_3','FMA_ACCUM_CYCLE_1','SALU_CYCLE_1','SALU_CYCLE_2','SALU_CYCLE_3']
    skips = ['SAME','NEXT','SKIP_1','SKIP_2','SKIP_3','SKIP_4']
    id0, skip, id1 = inst.simm16 & 0xf, (inst.simm16 >> 4) & 0x7, (inst.simm16 >> 7) & 0xf
    dep = lambda v: deps[v-1] if 0 < v <= len(deps) else str(v)
    p = [f"instid0({dep(id0)})" if id0 else "", f"instskip({skips[skip]})" if skip else "", f"instid1({dep(id1)})" if id1 else ""]
    return f"s_delay_alu {' | '.join(x for x in p if x) or '0'}"
  if name.startswith(('s_cbranch', 's_branch')): return f"{name} 0x{inst.simm16:x}"
  return f"{name} 0x{inst.simm16:x}"

def _disasm_smem(inst: SMEM) -> str:
  name, cdna = inst.op_name.lower(), _is_cdna(inst)
  if name in ('s_gl1_inv', 's_dcache_inv'): return name
  soe, imm = getattr(inst, 'soe', 0), getattr(inst, 'imm', 1)
  is_rdna4 = 'rdna4' in inst.__class__.__module__
  offset = inst.ioffset if is_rdna4 else getattr(inst, 'offset', 0)
  if cdna:
    if soe and imm: off_s = f"{decode_src(inst.soffset, cdna)} offset:0x{offset:x}"
    elif imm: off_s = f"0x{offset:x}"
    elif offset < 256: off_s = decode_src(offset, cdna)
    else: off_s = decode_src(inst.soffset, cdna)
  elif offset and inst.soffset != 124: off_s = f"{decode_src(inst.soffset, cdna)} offset:0x{offset:x}"
  elif offset: off_s = f"0x{offset:x}"
  else: off_s = decode_src(inst.soffset, cdna)
  is_buffer = 'buffer' in name or 's_atc_probe_buffer' == name
  sbase_idx, sbase_count = _unwrap(inst.sbase), 4 if is_buffer else 2
  sbase_str = _fmt_src(sbase_idx, sbase_count, cdna) if sbase_count == 2 else _sreg(sbase_idx, sbase_count) if sbase_idx <= 105 else _reg("ttmp", sbase_idx - 108, sbase_count)
  if name in ('s_atc_probe', 's_atc_probe_buffer'): return f"{name} {_unwrap(inst.sdata)}, {sbase_str}, {off_s}"
  if 'prefetch' in name:
    off = getattr(inst, 'ioffset', getattr(inst, 'offset', 0))
    if off >= 0x800000: off = off - 0x1000000
    off_s = f"0x{off:x}" if off > 255 else str(off)
    soff_s = decode_src(inst.soffset, cdna) if inst.soffset != 124 else "null"
    if 'pc_rel' in name: return f"{name} {off_s}, {soff_s}, {_unwrap(inst.sdata)}"
    return f"{name} {sbase_str}, {off_s}, {soff_s}, {_unwrap(inst.sdata)}"
  # Use get_field_bits for register count
  dst_n = inst.canonical_op_regs.get('d', 1)
  th, scope = getattr(inst, 'th', 0), getattr(inst, 'scope', 0)
  if is_rdna4:  # RDNA4 uses th/scope instead of glc/dlc
    th_names = ['TH_LOAD_RT', 'TH_LOAD_NT', 'TH_LOAD_HT', 'TH_LOAD_LU']
    scope_names = ['SCOPE_CU', 'SCOPE_SE', 'SCOPE_DEV', 'SCOPE_SYS']
    mods = (f" th:{th_names[th]}" if th else "") + (f" scope:{scope_names[scope]}" if scope else "")
    return f"{name} {_fmt_sdst(inst.sdata, dst_n, cdna)}, {sbase_str}, {off_s}{mods}"
  if th or scope:
    th_names = ['TH_LOAD_RT', 'TH_LOAD_NT', 'TH_LOAD_HT', 'TH_LOAD_LU']
    scope_names = ['SCOPE_CU', 'SCOPE_SE', 'SCOPE_DEV', 'SCOPE_SYS']
    mods = (f" th:{th_names[th]}" if th else "") + (f" scope:{scope_names[scope]}" if scope else "")
    return f"{name} {_fmt_sdst(inst.sdata, dst_n, cdna)}, {sbase_str}, {off_s}{mods}"
  return f"{name} {_fmt_sdst(inst.sdata, dst_n, cdna)}, {sbase_str}, {off_s}" + _mods((inst.glc, " glc"), (getattr(inst, 'dlc', 0), " dlc"))

def _disasm_flat(inst: FLAT) -> str:
  name, cdna = inst.op_name.lower(), _is_cdna(inst)
  acc = getattr(inst, 'acc', 0)
  reg_fn = _areg if acc else _vreg
  seg = ['flat', 'scratch', 'global'][inst.seg] if inst.seg < 3 else 'flat'
  instr = f"{seg}_{name.split('_', 1)[1] if '_' in name else name}"
  off_val = inst.offset if seg == 'flat' else (inst.offset if inst.offset < 4096 else inst.offset - 8192)
  # Use get_field_bits: data for stores/atomics, d for loads
  regs = inst.canonical_op_regs
  w = regs.get('data', regs.get('d', 1)) if 'store' in name or 'atomic' in name else regs.get('d', 1)
  off_s = f" offset:{off_val}" if off_val else ""
  if cdna: mods = f"{off_s}{' glc' if inst.sc0 else ''}{' slc' if inst.nt else ''}"
  else: mods = f"{off_s}{' glc' if inst.glc else ''}{' slc' if inst.slc else ''}{' dlc' if inst.dlc else ''}"
  if seg == 'flat' or _unwrap(inst.saddr) == 0x7F: saddr_s = ""
  elif _unwrap(inst.saddr) == 124: saddr_s = ", off"
  elif seg == 'scratch': saddr_s = f", {decode_src(inst.saddr, cdna)}"
  elif t := _ttmp(inst.saddr, 2): saddr_s = f", {t}"
  else: saddr_s = f", {_d(src[_off(inst.saddr):_off(inst.saddr)+1], cdna)}"
  if 'addtid' in name: return f"{instr} {reg_fn(inst.data if 'store' in name else inst.vdst)}{saddr_s}{mods}"
  if cdna: addr_w = 1 if seg == 'scratch' else 2
  else: addr_w = 1 if seg == 'scratch' or (_unwrap(inst.saddr) not in (0x7F, 124)) else 2
  addr_s = "off" if not inst.sve and seg == 'scratch' else _vreg(inst.addr, addr_w)
  data_s, vdst_s = reg_fn(inst.data, w), reg_fn(inst.vdst, w // 2 if 'cmpswap' in name else w)
  glc_or_sc0 = inst.sc0 if cdna else inst.glc
  if 'atomic' in name:
    return f"{instr} {vdst_s}, {addr_s}, {data_s}{saddr_s if seg != 'flat' else ''}{mods}" if glc_or_sc0 else f"{instr} {addr_s}, {data_s}{saddr_s if seg != 'flat' else ''}{mods}"
  if 'store' in name: return f"{instr} {addr_s}, {data_s}{saddr_s}{mods}"
  return f"{instr} {reg_fn(inst.vdst, w)}, {addr_s}{saddr_s}{mods}"

def _disasm_ds(inst: DS) -> str:
  op, name = inst.op, inst.op_name.lower()
  acc = getattr(inst, 'acc', 0)
  reg_fn = _areg if acc else _vreg
  gds = " gds" if getattr(inst, 'gds', 0) else ""
  off = f" offset:{inst.offset0 | (inst.offset1 << 8)}" if inst.offset0 or inst.offset1 else ""
  off2 = (" offset0:" + str(inst.offset0) if inst.offset0 else "") + (" offset1:" + str(inst.offset1) if inst.offset1 else "")
  # Use get_field_bits: data for stores/writes/atomics, d for loads
  regs = inst.canonical_op_regs
  w = regs.get('data', regs.get('d', 1)) if 'store' in name or 'write' in name or ('load' not in name and 'read' not in name) else regs.get('d', 1)
  d0, d1, dst, addr = reg_fn(inst.data0, w), reg_fn(inst.data1, w), reg_fn(inst.vdst, w), _vreg(inst.addr)

  if name == 'ds_nop': return name
  if name == 'ds_bvh_stack_rtn_b32': return f"{name} {_vreg(inst.vdst)}, {addr}, {_vreg(inst.data0)}, {_vreg(inst.data1, 4)}{off}{gds}"
  if 'bvh_stack_push' in name:
    d1_regs = 8 if 'push8' in name else 4
    vdst_regs = 2 if 'pop2' in name else 1
    vdst_s = _vreg(inst.vdst, vdst_regs) if vdst_regs > 1 else _vreg(inst.vdst)
    return f"{name} {vdst_s}, {addr}, {_vreg(inst.data0)}, {_vreg(inst.data1, d1_regs)}{off}{gds}"
  if 'gws_sema' in name and 'sema_br' not in name: return f"{name}{off}{gds}"
  if 'gws_' in name: return f"{name} {addr}{off}{gds}"
  if name in ('ds_consume', 'ds_append'): return f"{name} {reg_fn(inst.vdst)}{off}{gds}"
  if 'gs_reg' in name: return f"{name} {reg_fn(inst.vdst, 2)}, {reg_fn(inst.data0)}{off}{gds}"
  if '2addr' in name:
    if 'load' in name: return f"{name} {reg_fn(inst.vdst, regs.get('d', 1))}, {addr}{off2}{gds}"
    if 'store' in name and 'xchg' not in name: return f"{name} {addr}, {d0}, {d1}{off2}{gds}"
    return f"{name} {reg_fn(inst.vdst, regs.get('d', 1))}, {addr}, {d0}, {d1}{off2}{gds}"
  if 'write2' in name: return f"{name} {addr}, {d0}, {d1}{off2}{gds}"
  if 'read2' in name: return f"{name} {reg_fn(inst.vdst, regs.get('d', 1))}, {addr}{off2}{gds}"
  if 'load' in name: return f"{name} {reg_fn(inst.vdst)}{off}{gds}" if 'addtid' in name else f"{name} {dst}, {addr}{off}{gds}"
  if 'store' in name and not _has(name, 'cmp', 'xchg'):
    return f"{name} {reg_fn(inst.data0)}{off}{gds}" if 'addtid' in name else f"{name} {addr}, {d0}{off}{gds}"
  if 'swizzle' in name or name == 'ds_ordered_count': return f"{name} {reg_fn(inst.vdst)}, {addr}{off}{gds}"
  if 'permute' in name: return f"{name} {reg_fn(inst.vdst)}, {addr}, {reg_fn(inst.data0)}{off}{gds}"
  if 'condxchg' in name: return f"{name} {reg_fn(inst.vdst, 2)}, {addr}, {reg_fn(inst.data0, 2)}{off}{gds}"
  if _has(name, 'cmpstore', 'mskor', 'wrap'):
    return f"{name} {dst}, {addr}, {d0}, {d1}{off}{gds}" if '_rtn' in name else f"{name} {addr}, {d0}, {d1}{off}{gds}"
  return f"{name} {dst}, {addr}, {d0}{off}{gds}" if '_rtn' in name else f"{name} {addr}, {d0}{off}{gds}"

def _disasm_vop3(inst: VOP3) -> str:
  op, name = inst.op, inst.op_name.lower()
  n_up = name.upper()
  bits = inst.canonical_op_bits

  # RDNA4 v_s_* scalar VOP3 instructions - vdst is SGPR (VGPRField adds 256)
  if name.startswith('v_s_'):
    src = _lit(inst, inst.src0) if _unwrap(inst.src0) == 255 else ("src_scc" if _unwrap(inst.src0) == 253 else _fmt_src(inst.src0, max(1, bits['s0'] // 32)))
    if inst.neg & 1: src = f"-{src}"
    if inst.abs & 1: src = f"|{src}|"
    clamp = getattr(inst, 'cm', None) or getattr(inst, 'clmp', 0)
    vdst_raw = _unwrap(inst.vdst)
    return f"{name} s{vdst_raw - 256 if vdst_raw >= 256 else vdst_raw}, {src}" + (" clamp" if clamp else "") + _omod(inst.omod)

  # Use get_field_bits for register sizes and 16-bit detection
  r0, r1, r2 = max(1, bits['s0'] // 32), max(1, bits['s1'] // 32), max(1, bits['s2'] // 32)
  dn = max(1, bits['d'] // 32)
  is16_d, is16_s, is16_s2 = bits['d'] == 16, bits['s0'] == 16, bits['s2'] == 16

  s0 = _vop3_src(inst, inst.src0, inst.neg&1, inst.abs&1, inst.opsel&1, r0, is16_s)
  s1 = _vop3_src(inst, inst.src1, inst.neg&2, inst.abs&2, inst.opsel&2, r1, is16_s)
  s2 = _vop3_src(inst, inst.src2, inst.neg&4, inst.abs&4, inst.opsel&4, r2, is16_s2)

  # Format destination
  if 'readlane' in name:
    vdst_raw = _unwrap(inst.vdst)
    dst = _fmt_sdst(vdst_raw - 256 if vdst_raw >= 256 else vdst_raw, 1)
  elif dn > 1: dst = _vreg(inst.vdst, dn)
  elif is16_d: dst = f"{_vreg(inst.vdst)}.h" if (inst.opsel & 8) else f"{_vreg(inst.vdst)}.l"
  else: dst = _vreg(inst.vdst)

  clamp = getattr(inst, 'cm', None) or getattr(inst, 'clmp', 0)
  cl, om = " clamp" if clamp else "", _omod(inst.omod)
  nonvgpr_opsel = (_unwrap(inst.src0) < 256 and (inst.opsel & 1)) or (_unwrap(inst.src1) < 256 and (inst.opsel & 2)) or (_unwrap(inst.src2) < 256 and (inst.opsel & 4))
  need_opsel = nonvgpr_opsel or (inst.opsel and not is16_s)

  op_val = inst.op.value if hasattr(inst.op, 'value') else inst.op
  e64 = "" if name.endswith('_e64') else "_e64"
  if op_val < 256:  # VOPC
    vdst_raw = _unwrap(inst.vdst) - 256 if _unwrap(inst.vdst) >= 256 else _unwrap(inst.vdst)
    return f"{name}{e64} {s0}, {s1}{cl}" if name.startswith('v_cmpx') else f"{name}{e64} {_fmt_sdst(vdst_raw, 1)}, {s0}, {s1}{cl}"
  if op_val < 384:  # VOP2
    n = inst.num_srcs() or 2
    os = _opsel_str(inst.opsel, n, need_opsel, is16_d)
    return f"{name}{e64} {dst}, {s0}, {s1}, {s2}{os}{cl}{om}" if n == 3 else f"{name}{e64} {dst}, {s0}, {s1}{os}{cl}{om}"
  if op_val < 512:  # VOP1
    if re.match(r'v_cvt_f32_(bf|fp)8', name) and inst.opsel:
      os = f" byte_sel:{((inst.opsel & 1) << 1) | ((inst.opsel >> 1) & 1)}"
    else:
      os = _opsel_str(inst.opsel, 1, need_opsel, is16_d)
    if 'v_nop' in name or 'v_pipeflush' in name: return f"{name}{e64}"
    return f"{name}{e64} {dst}, {s0}{os}{cl}{om}"
  # Native VOP3
  n = inst.num_srcs() or 2
  os = f" byte_sel:{inst.opsel >> 2}" if 'cvt_sr' in name and inst.opsel else _opsel_str(inst.opsel, n, need_opsel, is16_d)
  return f"{name} {dst}, {s0}, {s1}, {s2}{os}{cl}{om}" if n == 3 else f"{name} {dst}, {s0}, {s1}{os}{cl}{om}"

def _disasm_vop3sd(inst: VOP3SD) -> str:
  name, regs = inst.op_name.lower(), inst.canonical_op_regs
  dn, sr0, sr1, sr2 = regs['d'], regs['s0'], regs['s1'], regs['s2']
  def src(v, neg, n):
    v = _unwrap(v)
    s = _lit(inst, v) if v == 255 else ("src_scc" if v == 253 else (_fmt_src(v, n) if n > 1 else _lit(inst, v)))
    return f"neg({s})" if neg and v == 255 else (f"-{s}" if neg else s)
  s0, s1, s2 = src(inst.src0, inst.neg & 1, sr0), src(inst.src1, inst.neg & 2, sr1), src(inst.src2, inst.neg & 4, sr2)
  # VOP3SD: _co_ ops (add/sub) without _ci_ have only 2 sources, all others (mad, div_scale, _co_ci_) have 3 sources
  has_only_two_srcs = '_co_' in name and '_ci_' not in name and 'mad' not in name
  srcs = f"{s0}, {s1}" if has_only_two_srcs else f"{s0}, {s1}, {s2}"
  clamp = getattr(inst, 'cm', None) or getattr(inst, 'clmp', 0)
  return f"{name} {_vreg(inst.vdst, dn)}, {_fmt_sdst(inst.sdst, 1)}, {srcs}{' clamp' if clamp else ''}{_omod(inst.omod)}"

def _disasm_vopd(inst: VOPD) -> str:
  lit = inst._literal or getattr(inst, 'literal', None)
  is_rdna4 = 'rdna4' in inst.__class__.__module__
  op_enum = R4_VOPDOp if is_rdna4 else VOPDOp
  vdst_y, nx, ny = (_unwrap(inst.vdsty) << 1) | ((_unwrap(inst.vdstx) & 1) ^ 1), op_enum(inst.opx).name.lower(), op_enum(inst.opy).name.lower()
  def half(n, vd, s0, vs1):
    vd, vs1 = _vi(vd), _vi(vs1)
    if 'mov' in n: return f"{n} v{vd}, {_lit(inst, s0)}"
    if 'fmamk' in n and lit: return f"{n} v{vd}, {_lit(inst, s0)}, 0x{lit:x}, v{vs1}"
    if 'fmaak' in n and lit: return f"{n} v{vd}, {_lit(inst, s0)}, v{vs1}, 0x{lit:x}"
    return f"{n} v{vd}, {_lit(inst, s0)}, v{vs1}"
  return f"{half(nx, inst.vdstx, inst.srcx0, inst.vsrcx1)} :: {half(ny, vdst_y, inst.srcy0, inst.vsrcy1)}"

def _disasm_vop3p(inst: VOP3P) -> str:
  name = inst.op_name.lower()
  is_wmma, is_swmmac, n, is_fma_mix = 'wmma' in name, 'swmmac' in name, inst.num_srcs() or 2, 'fma_mix' in name
  def get_src(v, sc):
    uv = _unwrap(v)
    return _lit(inst, uv) if uv == 255 else _fmt_src(uv, sc)
  # Use get_field_bits for register sizes
  regs = inst.canonical_op_regs
  src0, src1, src2, dst = get_src(inst.src0, regs['s0']), get_src(inst.src1, regs['s1']), get_src(inst.src2, regs['s2']), _vreg(inst.vdst, regs['d'])
  opsel_hi = inst.opsel_hi | (inst.opsel_hi2 << 2)
  clamp = getattr(inst, 'cm', None) or getattr(inst, 'clmp', 0)
  if is_fma_mix:
    def m(s, neg, abs_): return f"-{f'|{s}|' if abs_ else s}" if neg else (f"|{s}|" if abs_ else s)
    src0, src1, src2 = m(src0, inst.neg & 1, inst.neg_hi & 1), m(src1, inst.neg & 2, inst.neg_hi & 2), m(src2, inst.neg & 4, inst.neg_hi & 4)
    mods = ([_fmt_bits("op_sel", inst.opsel, n)] if inst.opsel else []) + ([_fmt_bits("op_sel_hi", opsel_hi, n)] if opsel_hi else []) + (["clamp"] if clamp else [])
  elif is_swmmac:
    mods = ([f"index_key:{inst.opsel}"] if inst.opsel else []) + ([_fmt_bits("neg_lo", inst.neg, n)] if inst.neg else []) + \
           ([_fmt_bits("neg_hi", inst.neg_hi, n)] if inst.neg_hi else []) + (["clamp"] if clamp else [])
  else:
    opsel_hi_default = 7 if n == 3 else 3
    mods = ([_fmt_bits("op_sel", inst.opsel, n)] if inst.opsel else []) + ([_fmt_bits("op_sel_hi", opsel_hi, n)] if opsel_hi != opsel_hi_default else []) + \
           ([_fmt_bits("neg_lo", inst.neg, n)] if inst.neg else []) + ([_fmt_bits("neg_hi", inst.neg_hi, n)] if inst.neg_hi else []) + (["clamp"] if clamp else [])
  return f"{name} {dst}, {src0}, {src1}, {src2}{' ' + ' '.join(mods) if mods else ''}" if n == 3 else f"{name} {dst}, {src0}, {src1}{' ' + ' '.join(mods) if mods else ''}"

def _disasm_buf(inst: MUBUF | MTBUF) -> str:
  name, cdna = inst.op_name.lower(), _is_cdna(inst)
  acc = getattr(inst, 'acc', 0)
  reg_fn = _areg if acc else _vreg
  if cdna and name in ('buffer_wbl2', 'buffer_inv'): return name
  if not cdna and inst.op in (MUBUFOp.BUFFER_GL0_INV, MUBUFOp.BUFFER_GL1_INV): return name
  w = (2 if _has(name, 'xyz', 'xyzw') else 1) if 'd16' in name else \
      ((2 if _has(name, 'b64', 'u64', 'i64') else 1) * (2 if 'cmpswap' in name else 1)) if 'atomic' in name else \
      {'b32':1,'b64':2,'b96':3,'b128':4,'b16':1,'x':1,'xy':2,'xyz':3,'xyzw':4}.get(name.split('_')[-1], 1)
  if hasattr(inst, 'tfe') and inst.tfe: w += 1
  vaddr = _vreg(inst.vaddr, 2) if inst.offen and inst.idxen else _vreg(inst.vaddr) if inst.offen or inst.idxen else "off"
  srsrc = _sreg_or_ttmp(_unwrap(inst.srsrc), 4)
  is_mtbuf = isinstance(inst, MTBUF) or isinstance(inst, C_MTBUF)
  if is_mtbuf:
    dfmt, nfmt = inst.format & 0xf, (inst.format >> 4) & 0x7
    if acc: fmt_s = f"  dfmt:{dfmt}, nfmt:{nfmt},"
    elif not cdna: fmt_s = f" format:{inst.format}" if inst.format else ""
    else:
      dfmt_names = ['INVALID', '8', '16', '8_8', '32', '16_16', '10_11_11', '11_11_10', '10_10_10_2', '2_10_10_10', '8_8_8_8', '32_32', '16_16_16_16', '32_32_32', '32_32_32_32', 'RESERVED_15']
      nfmt_names = ['UNORM', 'SNORM', 'USCALED', 'SSCALED', 'UINT', 'SINT', 'RESERVED_6', 'FLOAT']
      if dfmt == 1 and nfmt == 0: fmt_s = ""
      elif nfmt == 0: fmt_s = f" format:[BUF_DATA_FORMAT_{dfmt_names[dfmt]}]"
      elif dfmt == 1: fmt_s = f" format:[BUF_NUM_FORMAT_{nfmt_names[nfmt]}]"
      else: fmt_s = f" format:[BUF_DATA_FORMAT_{dfmt_names[dfmt]},BUF_NUM_FORMAT_{nfmt_names[nfmt]}]"
  else: fmt_s = ""
  if cdna: mods = [m for c, m in [(inst.idxen,"idxen"),(inst.offen,"offen"),(inst.offset,f"offset:{inst.offset}"),(inst.sc0,"glc"),(inst.nt,"slc"),(inst.sc1,"sc1")] if c]
  else: mods = [m for c, m in [(inst.idxen,"idxen"),(inst.offen,"offen"),(inst.offset,f"offset:{inst.offset}"),(inst.glc,"glc"),(inst.dlc,"dlc"),(inst.slc,"slc"),(inst.tfe,"tfe")] if c]
  soffset_s = decode_src(inst.soffset, cdna)
  if cdna and not acc and is_mtbuf: return f"{name} {reg_fn(inst.vdata, w)}, {vaddr}, {srsrc}, {soffset_s}{fmt_s}{' ' + ' '.join(mods) if mods else ''}"
  return f"{name} {reg_fn(inst.vdata, w)}, {vaddr}, {srsrc},{fmt_s} {soffset_s}{' ' + ' '.join(mods) if mods else ''}"

def _mimg_vaddr_width(name: str, dim: int, a16: bool) -> int:
  base =              [1, 2, 3, 3,   2,     3,     3,      4][dim]
  grad =              [1, 2, 3, 2,   1,     2,     2,      2][dim]
  if 'get_resinfo' in name: return 1
  packed, unpacked = 0, 0
  if '_mip' in name: packed += 1
  elif 'sample' in name or 'gather' in name:
    if '_o' in name: unpacked += 1
    if re.search(r'_c(_|$)', name): unpacked += 1
    if '_d' in name: unpacked += (grad + 1) & ~1 if '_g16' in name else grad*2
    if '_b' in name: unpacked += 1
    if '_l' in name and '_cl' not in name and '_lz' not in name: packed += 1
    if '_cl' in name: packed += 1
  return (base + packed + 1) // 2 + unpacked if a16 else base + packed + unpacked

def _disasm_mimg(inst: MIMG) -> str:
  name = inst.op_name.lower()
  srsrc_base = _unwrap(inst.srsrc)
  srsrc_str = _sreg_or_ttmp(srsrc_base, 8)
  if 'bvh' in name:
    vaddr = (9 if '64' in name else 8) if inst.a16 else (12 if '64' in name else 11)
    return f"{name} {_vreg(inst.vdata, 4)}, {_vreg(inst.vaddr, vaddr)}, {_sreg_or_ttmp(srsrc_base, 4)}{' a16' if inst.a16 else ''}"
  vdata = 4 if 'gather4' in name or 'msaa_load' in name else (bin(inst.dmask).count('1') or 1)
  if inst.d16: vdata = (vdata + 1) // 2
  if inst.tfe: vdata += 1
  dim_names = ['1d', '2d', '3d', 'cube', '1d_array', '2d_array', '2d_msaa', '2d_msaa_array']
  dim = dim_names[inst.dim] if inst.dim < len(dim_names) else f"dim_{inst.dim}"
  vaddr = _mimg_vaddr_width(name, inst.dim, inst.a16)
  vaddr_str = _vreg(inst.vaddr) if vaddr == 1 else _vreg(inst.vaddr, vaddr)
  mods = [f"dmask:0x{inst.dmask:x}"] if inst.dmask and (inst.dmask != 15 or 'atomic' in name) else []
  mods.append(f"dim:SQ_RSRC_IMG_{dim.upper()}")
  for flag, mod in [(inst.unrm,"unorm"),(inst.glc,"glc"),(inst.slc,"slc"),(inst.dlc,"dlc"),(inst.r128,"r128"),
                    (inst.a16,"a16"),(inst.tfe,"tfe"),(inst.lwe,"lwe"),(inst.d16,"d16")]:
    if flag: mods.append(mod)
  ssamp_str = ""
  if 'sample' in name or 'gather' in name or 'get_lod' in name:
    ssamp_str = ", " + _sreg_or_ttmp(_unwrap(inst.ssamp), 4)
  return f"{name} {_vreg(inst.vdata, vdata)}, {vaddr_str}, {srsrc_str}{ssamp_str} {' '.join(mods)}"

def _disasm_sop1(inst: SOP1) -> str:
  op, name, cdna = inst.op, inst.op_name.lower(), _is_cdna(inst)
  # Use get_field_bits for register sizes
  regs = inst.canonical_op_regs
  dst_regs, src_regs = regs.get('d', 1), regs.get('s0', 1)
  src = _lit(inst, inst.ssrc0) if _unwrap(inst.ssrc0) == 255 else _fmt_src(inst.ssrc0, src_regs, cdna)
  if not cdna:
    if 'getpc_b64' in name: return f"{name} {_fmt_sdst(inst.sdst, 2)}"
    if 'setpc_b64' in name or 'rfe_b64' in name: return f"{name} {src}"
    if 'swappc_b64' in name: return f"{name} {_fmt_sdst(inst.sdst, 2)}, {src}"
    if 'sendmsg_rtn' in name:
      v = _unwrap(inst.ssrc0)
      msg_str = MSG.get(v)
      return f"{name} {_fmt_sdst(inst.sdst, dst_regs)}, sendmsg({msg_str})" if msg_str else f"{name} {_fmt_sdst(inst.sdst, dst_regs)}, 0x{v:x}"
  sop1_src_only = ('S_ALLOC_VGPR', 'S_SLEEP_VAR', 'S_BARRIER_SIGNAL', 'S_BARRIER_SIGNAL_ISFIRST', 'S_BARRIER_INIT', 'S_BARRIER_JOIN')
  if inst.op_name in sop1_src_only: return f"{name} {src}"
  return f"{name} {_fmt_sdst(inst.sdst, dst_regs, cdna)}, {src}"

def _disasm_sop2(inst: SOP2) -> str:
  cdna, name = _is_cdna(inst), inst.op_name.lower()
  lit = getattr(inst, '_literal', None)
  # Use get_field_bits for register sizes
  regs = inst.canonical_op_regs
  dn, s0n, s1n = regs['d'], regs['s0'], regs['s1']
  s0 = _lit(inst, inst.ssrc0) if _unwrap(inst.ssrc0) == 255 else _fmt_src(inst.ssrc0, s0n, cdna)
  s1 = _lit(inst, inst.ssrc1) if _unwrap(inst.ssrc1) == 255 else _fmt_src(inst.ssrc1, s1n, cdna)
  dst = _fmt_sdst(inst.sdst, dn, cdna)
  if 'fmamk' in name and lit is not None: return f"{name} {dst}, {s0}, 0x{lit:x}, {s1}"
  if 'fmaak' in name and lit is not None: return f"{name} {dst}, {s0}, {s1}, 0x{lit:x}"
  return f"{name} {dst}, {s0}, {s1}"

def _disasm_sopc(inst: SOPC) -> str:
  cdna, regs = _is_cdna(inst), inst.canonical_op_regs
  s0 = _lit(inst, inst.ssrc0) if _unwrap(inst.ssrc0) == 255 else _fmt_src(inst.ssrc0, regs['s0'], cdna)
  s1 = _lit(inst, inst.ssrc1) if _unwrap(inst.ssrc1) == 255 else _fmt_src(inst.ssrc1, regs['s1'], cdna)
  return f"{inst.op_name.lower()} {s0}, {s1}"

def _disasm_sopk(inst: SOPK) -> str:
  op, name, cdna = inst.op, inst.op_name.lower(), _is_cdna(inst)
  is_rdna4 = 'rdna4' in inst.__class__.__module__
  hw = HWREG_RDNA4 if is_rdna4 else HWREG
  def fmt_hwreg(hid, hoff, hsz):
    if hid not in hw: return f"0x{inst.simm16:x}"
    hr_name = hw[hid]
    return f"hwreg({hr_name})" if hoff == 0 and hsz == 32 else f"hwreg({hr_name}, {hoff}, {hsz})"
  if name == 's_setreg_imm32_b32':
    hid, hoff, hsz = inst.simm16 & 0x3f, (inst.simm16 >> 6) & 0x1f, ((inst.simm16 >> 11) & 0x1f) + 1
    return f"{name} {fmt_hwreg(hid, hoff, hsz)}, 0x{inst._literal:x}"
  if name == 's_version': return f"{name} 0x{inst.simm16:x}"
  if name in ('s_setreg_b32', 's_getreg_b32'):
    hid, hoff, hsz = inst.simm16 & 0x3f, (inst.simm16 >> 6) & 0x1f, ((inst.simm16 >> 11) & 0x1f) + 1
    hs = fmt_hwreg(hid, hoff, hsz)
    return f"{name} {hs}, {_fmt_sdst(inst.sdst, 1, cdna)}" if 'setreg' in name else f"{name} {_fmt_sdst(inst.sdst, 1, cdna)}, {hs}"
  if name in ('s_subvector_loop_begin', 's_subvector_loop_end'):
    return f"{name} {_fmt_sdst(inst.sdst, 1)}, 0x{inst.simm16:x}"
  return f"{name} {_fmt_sdst(inst.sdst, inst.canonical_op_regs['d'], cdna)}, 0x{inst.simm16:x}"

def _disasm_vinterp(inst: VINTERP) -> str:
  mods = _mods((inst.waitexp, f"wait_exp:{inst.waitexp}"), (inst.clmp, "clamp"))
  return f"{inst.op_name.lower()} {_vreg(inst.vdst)}, {_lit(inst, inst.src0, inst.neg & 1)}, {_lit(inst, inst.src1, inst.neg & 2)}, {_lit(inst, inst.src2, inst.neg & 4)}" + (" " + mods if mods else "")

EXP_TARGETS = {0: 'mrt0', 1: 'mrt1', 2: 'mrt2', 3: 'mrt3', 4: 'mrt4', 5: 'mrt5', 6: 'mrt6', 7: 'mrt7',
               8: 'mrtz', 9: 'null', 12: 'pos0', 13: 'pos1', 14: 'pos2', 15: 'pos3', 16: 'pos4',
               32: 'param0', 33: 'param1', 34: 'param2', 35: 'param3', 36: 'param4', 37: 'param5'}
def _disasm_vexport(inst) -> str:
  tgt = EXP_TARGETS.get(inst.target, f'{inst.target}')
  srcs = [f'{_vreg(getattr(inst, f"vsrc{i}"))}' if inst.en & (1 << i) else 'off' for i in range(4)]
  mods = _mods((inst.done, "done"), (inst.row, "row_en"))
  return f"export {tgt} {', '.join(srcs)}" + (" " + mods if mods else "")

def _disasm_vbuffer(inst) -> str:
  name = inst.op_name.lower().replace('buffer_', 'buffer_').replace('tbuffer_', 'tbuffer_')
  w = (2 if _has(name, 'xyz', 'xyzw') else 1) if 'd16' in name else \
      ((2 if _has(name, 'b64', 'u64', 'i64') else 1) * (2 if 'cmpswap' in name else 1)) if 'atomic' in name else \
      {'b32':1,'b64':2,'b96':3,'b128':4,'b16':1,'x':1,'xy':2,'xyz':3,'xyzw':4}.get(name.split('_')[-1], inst.canonical_op_regs['d'])
  if getattr(inst, 'tfe', 0): w += 1
  vdata = _vreg(inst.vdata, w) if w else _vreg(inst.vdata)
  vaddr = _vreg(inst.vaddr, 2) if inst.offen and inst.idxen else (_vreg(inst.vaddr) if inst.offen or inst.idxen else 'off')
  srsrc = f'ttmp[{inst.rsrc - 108}:{inst.rsrc - 108 + 3}]' if inst.rsrc >= 108 else f's[{inst.rsrc}:{inst.rsrc + 3}]'
  soff = decode_src(inst.soffset) if _unwrap(inst.soffset) >= 106 else f's{_unwrap(inst.soffset)}'
  fmt = getattr(inst, 'format', 0)
  from extra.assembly.amd.asm import BUF_FMT
  fmt_names = {v: k for k, v in BUF_FMT.items()}
  fmt_s = f" format:[{fmt_names[fmt]}]" if fmt > 1 and fmt in fmt_names else (f" format:{fmt}" if fmt > 1 else "")
  if 'atomic' in name: th_names = {1: 'TH_ATOMIC_RETURN', 6: 'TH_ATOMIC_CASCADE_NT'}
  elif 'store' in name: th_names = {3: 'TH_STORE_BYPASS', 6: 'TH_STORE_NT_HT'}
  else: th_names = {3: 'TH_LOAD_BYPASS', 6: 'TH_LOAD_NT_HT'}
  scope_names = {1: 'SCOPE_SE', 2: 'SCOPE_DEV', 3: 'SCOPE_SYS'}
  mods = _mods((inst.idxen, "idxen"), (inst.offen, "offen"), (inst.ioffset, f"offset:{inst.ioffset}"),
               (inst.th in th_names, f"th:{th_names.get(inst.th, '')}"), (inst.scope in scope_names, f"scope:{scope_names.get(inst.scope, '')}"))
  return f"{name} {vdata}, {vaddr}, {srsrc}, {soff}{fmt_s}" + (" " + mods if mods else "")

DISASM_HANDLERS: dict[type, callable] = {
  VOP1: _disasm_vop1, VOP1_SDST: _disasm_vop1, VOP2: _disasm_vop2, VOPC: _disasm_vopc, VOP3: _disasm_vop3, VOP3_SDST: _disasm_vop3, VOP3SD: _disasm_vop3sd, VOPD: _disasm_vopd, VOP3P: _disasm_vop3p,
  VINTERP: _disasm_vinterp, SOPP: _disasm_sopp, SMEM: _disasm_smem, DS: _disasm_ds, FLAT: _disasm_flat, GLOBAL: _disasm_flat, SCRATCH: _disasm_flat,
  MUBUF: _disasm_buf, MTBUF: _disasm_buf, MIMG: _disasm_mimg, SOP1: _disasm_sop1, SOP2: _disasm_sop2, SOPC: _disasm_sopc, SOPK: _disasm_sopk,
  # RDNA4
  R4_VOP1: _disasm_vop1, R4_VOP1_SDST: _disasm_vop1, R4_VOP2: _disasm_vop2, R4_VOPC: _disasm_vopc, R4_VOP3: _disasm_vop3, R4_VOP3_SDST: _disasm_vop3, R4_VOP3SD: _disasm_vop3sd,
  R4_VOPD: _disasm_vopd, R4_VOP3P: _disasm_vop3p, R4_VINTERP: _disasm_vinterp, R4_SOPP: _disasm_sopp, R4_SMEM: _disasm_smem,
  R4_DS: _disasm_ds, R4_SOP1: _disasm_sop1, R4_SOP2: _disasm_sop2, R4_SOPC: _disasm_sopc, R4_SOPK: _disasm_sopk,
  R4_VEXPORT: _disasm_vexport, R4_VBUFFER: _disasm_vbuffer}

def disasm(inst: Inst) -> str: return DISASM_HANDLERS[type(inst)](inst)

# ═══════════════════════════════════════════════════════════════════════════════
# CDNA DISASSEMBLER SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════

try:
  from extra.assembly.amd.autogen.cdna.ins import (VOP1 as CDNA_VOP1, VOP2 as CDNA_VOP2, VOPC as CDNA_VOPC, VOP3A, VOP3B, VOP3P as CDNA_VOP3P,
    SOP1 as CDNA_SOP1, SOP2 as CDNA_SOP2, SOPC as CDNA_SOPC, SOPK as CDNA_SOPK, SOPP as CDNA_SOPP, SMEM as CDNA_SMEM, DS as CDNA_DS,
    FLAT as CDNA_FLAT, MUBUF as CDNA_MUBUF, MTBUF as CDNA_MTBUF, VOP1Op as CDNA_VOP1Op, VOP2Op as CDNA_VOP2Op, VOPCOp as CDNA_VOPCOp)

  def _cdna_src(inst, v, neg, abs_=0, n=1):
    s = _lit(inst, v) if v == 255 else _fmt_src(v, n, cdna=True)
    if abs_: s = f"|{s}|"
    return f"neg({s})" if neg and v == 255 else (f"-{s}" if neg else s)

  _CDNA_VOP3_ALIASES = {'v_fmac_f64': 'v_mul_legacy_f32', 'v_dot2c_f32_bf16': 'v_mac_f32'}

  def _disasm_vop3a(inst) -> str:
    op_val = inst._values.get('op', 0)
    if hasattr(op_val, 'value'): op_val = op_val.value
    name = inst.op_name.lower() or f'vop3a_op_{op_val}'
    n = inst.num_srcs() or _num_srcs(inst)
    cl, om = " clamp" if inst.clmp else "", _omod(inst.omod)
    orig_name = name
    name = _CDNA_VOP3_ALIASES.get(name, name)
    if name != orig_name:
      s0, s1 = _cdna_src(inst, inst.src0, inst.neg&1, inst.abs&1, 1), _cdna_src(inst, inst.src1, inst.neg&2, inst.abs&2, 1)
      s2 = ""
      dst = _vreg(inst.vdst)
    else:
      regs = inst.canonical_op_regs
      dregs, r0, r1, r2 = regs['d'], regs['s0'], regs['s1'], regs['s2']
      s0, s1, s2 = _cdna_src(inst, inst.src0, inst.neg&1, inst.abs&1, r0), _cdna_src(inst, inst.src1, inst.neg&2, inst.abs&2, r1), _cdna_src(inst, inst.src2, inst.neg&4, inst.abs&4, r2)
      dst = _vreg(inst.vdst, dregs) if dregs > 1 else _vreg(inst.vdst)
    if op_val >= 512:
      return f"{name} {dst}, {s0}, {s1}, {s2}{cl}{om}" if n == 3 else f"{name} {dst}, {s0}, {s1}{cl}{om}"
    if op_val < 256:
      sdst = _fmt_sdst(inst.vdst, 2, cdna=True)
      return f"{name}_e64 {sdst}, {s0}, {s1}{cl}"
    if 320 <= op_val < 512:
      if name in ('v_nop', 'v_clrexcp'): return f"{name}_e64"
      return f"{name}_e64 {dst}, {s0}{cl}{om}"
    if name == 'v_cndmask_b32':
      s2 = _fmt_src(inst.src2, 2, cdna=True)
      return f"{name}_e64 {dst}, {s0}, {s1}, {s2}{cl}{om}"
    if name in ('v_mul_legacy_f32', 'v_mac_f32'):
      return f"{name}_e64 {dst}, {s0}, {s1}{cl}{om}"
    suf = "_e64" if op_val < 512 else ""
    return f"{name}{suf} {dst}, {s0}, {s1}, {s2}{cl}{om}" if n == 3 else f"{name}{suf} {dst}, {s0}, {s1}{cl}{om}"

  def _disasm_vop3b(inst) -> str:
    op_val = inst._values.get('op', 0)
    if hasattr(op_val, 'value'): op_val = op_val.value
    name = inst.op_name.lower() or f'vop3b_op_{op_val}'
    n = inst.num_srcs() or _num_srcs(inst)
    regs = inst.canonical_op_regs
    dregs, r0, r1, r2 = regs['d'], regs['s0'], regs['s1'], regs['s2']
    s0, s1, s2 = _cdna_src(inst, inst.src0, inst.neg&1, n=r0), _cdna_src(inst, inst.src1, inst.neg&2, n=r1), _cdna_src(inst, inst.src2, inst.neg&4, n=r2)
    dst = _vreg(inst.vdst, dregs) if dregs > 1 else _vreg(inst.vdst)
    sdst = _fmt_sdst(inst.sdst, 2, cdna=True)
    cl, om = " clamp" if inst.clmp else "", _omod(inst.omod)
    if name in ('v_addc_co_u32', 'v_subb_co_u32', 'v_subbrev_co_u32'):
      s2 = _fmt_src(inst.src2, 2, cdna=True)
      return f"{name}_e64 {dst}, {sdst}, {s0}, {s1}, {s2}{cl}{om}"
    suf = "_e64" if 'co_' in name else ""
    return f"{name}{suf} {dst}, {sdst}, {s0}, {s1}, {s2}{cl}{om}" if n == 3 else f"{name}{suf} {dst}, {sdst}, {s0}, {s1}{cl}{om}"

  def _disasm_cdna_vop3p(inst) -> str:
    name, n, is_mfma = inst.op_name.lower(), inst.num_srcs() or 2, 'mfma' in inst.op_name.lower() or 'smfmac' in inst.op_name.lower()
    get_src = lambda v, sc: _lit(inst, v) if v == 255 else _fmt_src(v, sc, cdna=True)
    if is_mfma: sc = 2 if 'iu4' in name else 4 if 'iu8' in name or 'i4' in name else 8 if 'f16' in name or 'bf16' in name else 4; src0, src1, src2, dst = get_src(inst.src0, sc), get_src(inst.src1, sc), get_src(inst.src2, 16), _vreg(inst.vdst, 16)
    else: src0, src1, src2, dst = get_src(inst.src0, 1), get_src(inst.src1, 1), get_src(inst.src2, 1), _vreg(inst.vdst)
    opsel_hi = inst.opsel_hi | (inst.opsel_hi2 << 2)
    mods = ([_fmt_bits("op_sel", inst.opsel, n)] if inst.opsel else []) + ([_fmt_bits("op_sel_hi", opsel_hi, n)] if opsel_hi != (7 if n == 3 else 3) else []) + \
           ([_fmt_bits("neg_lo", inst.neg, n)] if inst.neg else []) + ([_fmt_bits("neg_hi", inst.neg_hi, n)] if inst.neg_hi else []) + (["clamp"] if inst.clmp else [])
    return f"{name} {dst}, {src0}, {src1}, {src2}{' ' + ' '.join(mods) if mods else ''}" if n == 3 else f"{name} {dst}, {src0}, {src1}{' ' + ' '.join(mods) if mods else ''}"

  DISASM_HANDLERS.update({CDNA_VOP1: _disasm_vop1, CDNA_VOP2: _disasm_vop2, CDNA_VOPC: _disasm_vopc,
    CDNA_SOP1: _disasm_sop1, CDNA_SOP2: _disasm_sop2, CDNA_SOPC: _disasm_sopc, CDNA_SOPK: _disasm_sopk, CDNA_SOPP: _disasm_sopp,
    CDNA_SMEM: _disasm_smem, CDNA_DS: _disasm_ds, CDNA_FLAT: _disasm_flat, CDNA_MUBUF: _disasm_buf, CDNA_MTBUF: _disasm_buf,
    VOP3A: _disasm_vop3a, VOP3B: _disasm_vop3b, CDNA_VOP3P: _disasm_cdna_vop3p})
except ImportError:
  pass
