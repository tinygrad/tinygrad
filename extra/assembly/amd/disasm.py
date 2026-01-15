# RDNA3/RDNA4/CDNA disassembler
from __future__ import annotations
import re
from extra.assembly.amd.dsl import Inst, Reg, src

# CDNA-specific registers not in Reg._NAMES
_CDNA = {102: 'flat_scratch_lo', 103: 'flat_scratch_hi', 104: 'xnack_mask_lo', 105: 'xnack_mask_hi'}
_CDNA2 = {102: 'flat_scratch', 104: 'xnack_mask'}

def _off(v) -> int: return v.offset if isinstance(v, Reg) else v
def _vi(v) -> int: off = _off(v); return off - 256 if off >= 256 else off
def _cdna(i) -> bool: return 'cdna' in i.__class__.__module__
def _r4(i) -> bool: return 'rdna4' in i.__class__.__module__

def _s(v, n=1, cdna=False) -> str:
  """Format source operand."""
  off = _off(v)
  if off == 253: return "src_scc"
  r = src[off:off+n-1] if n > 1 else src[off]
  if cdna and r.sz == 1 and off in _CDNA: return _CDNA[off]
  if cdna and r.sz == 2 and off in _CDNA2: return _CDNA2[off]
  return r.disasm

def _v(b, n=1) -> str: off = _off(b); idx = off - 256 if off >= 256 else off; return f"v{idx}" if n == 1 else f"v[{idx}:{idx+n-1}]"
def _v16(b) -> str: off = _off(b); idx = (off - 256) & 0x7f; return f"v{idx}.{'h' if (off - 256) >= 128 else 'l'}"
def _a(b, n=1) -> str: off = _off(b); idx = off - 256 if off >= 256 else off; return f"a{idx}" if n == 1 else f"a[{idx}:{idx+n-1}]"
def _t(b, n=1) -> str: off = _off(b); return (f"ttmp{off-108}" if n == 1 else f"ttmp[{off-108}:{off-108+n-1}]") if 108 <= off <= 123 else None
def _st(b, n=1) -> str: return _t(b, n) or _s(b, n)

def _lit(i, v, neg=0) -> str:
  off = _off(v)
  if off == 255: return f"-0x{i._literal:x}" if neg else (f"0x{i._literal:x}" if i._literal else "0")
  s = src[off].disasm
  return f"-{s}" if neg else s

def _omod(v) -> str: return {1: " mul:2", 2: " mul:4", 3: " div:2"}.get(v, "")
def _mods(*p) -> str: return " ".join(m for c, m in p if c)
def _bits(l, v, n) -> str: return f"{l}:[{','.join(str((v >> i) & 1) for i in range(n))}]"

def _vop3_src(i, v, neg, abs_, hi, n, f16) -> str:
  off = _off(v)
  if off == 255: s = _lit(i, v)
  elif n > 1: s = _s(v, n)
  elif f16 and off >= 256: s = f"v{off - 256}.{'h' if hi else 'l'}"
  elif off == 253: s = "src_scc"
  else: s = _lit(i, v)
  if abs_: s = f"|{s}|"
  return f"-{s}" if neg else s

# Imports
from extra.assembly.amd.autogen.rdna3.ins import (VOP1, VOP1_SDST, VOP2, VOP3, VOP3_SDST, VOP3SD, VOP3P, VOPC, VOPD, VINTERP,
  SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, DS, FLAT, GLOBAL, SCRATCH, MUBUF, MTBUF, MIMG, EXP, VOP2Op, VOPDOp, SOPPOp)
from extra.assembly.amd.autogen.rdna4.ins import (VOP1 as R4_VOP1, VOP1_SDST as R4_VOP1_SDST, VOP2 as R4_VOP2, VOP3 as R4_VOP3,
  VOP3_SDST as R4_VOP3_SDST, VOP3SD as R4_VOP3SD, VOP3P as R4_VOP3P, VOPC as R4_VOPC, VOPD as R4_VOPD, VINTERP as R4_VINTERP,
  SOP1 as R4_SOP1, SOP2 as R4_SOP2, SOPC as R4_SOPC, SOPK as R4_SOPK, SOPP as R4_SOPP, SMEM as R4_SMEM, DS as R4_DS,
  VBUFFER as R4_VBUFFER, VEXPORT as R4_VEXPORT, VOPDOp as R4_VOPDOp)
from extra.assembly.amd.autogen.cdna.ins import FLAT as C_FLAT, MUBUF as C_MUBUF, MTBUF as C_MTBUF

HWREG = {1:'HW_REG_MODE',2:'HW_REG_STATUS',3:'HW_REG_TRAPSTS',4:'HW_REG_HW_ID',5:'HW_REG_GPR_ALLOC',6:'HW_REG_LDS_ALLOC',
  7:'HW_REG_IB_STS',15:'HW_REG_SH_MEM_BASES',18:'HW_REG_PERF_SNAPSHOT_PC_LO',19:'HW_REG_PERF_SNAPSHOT_PC_HI',
  20:'HW_REG_FLAT_SCR_LO',21:'HW_REG_FLAT_SCR_HI',22:'HW_REG_XNACK_MASK',23:'HW_REG_HW_ID1',24:'HW_REG_HW_ID2',25:'HW_REG_POPS_PACKER',28:'HW_REG_IB_STS2'}
HWREG_RDNA4 = {1:'HW_REG_MODE',2:'HW_REG_STATUS',4:'HW_REG_STATE_PRIV',5:'HW_REG_GPR_ALLOC',6:'HW_REG_LDS_ALLOC',7:'HW_REG_IB_STS',
  10:'HW_REG_PERF_SNAPSHOT_DATA',11:'HW_REG_PERF_SNAPSHOT_PC_LO',12:'HW_REG_PERF_SNAPSHOT_PC_HI',15:'HW_REG_PERF_SNAPSHOT_DATA1',
  16:'HW_REG_PERF_SNAPSHOT_DATA2',17:'HW_REG_EXCP_FLAG_PRIV',18:'HW_REG_EXCP_FLAG_USER',19:'HW_REG_TRAP_CTRL',20:'HW_REG_SCRATCH_BASE_LO',
  21:'HW_REG_SCRATCH_BASE_HI',23:'HW_REG_HW_ID1',24:'HW_REG_HW_ID2',26:'HW_REG_SCHED_MODE',29:'HW_REG_SHADER_CYCLES_LO',
  30:'HW_REG_SHADER_CYCLES_HI',31:'HW_REG_DVGPR_ALLOC_LO',32:'HW_REG_DVGPR_ALLOC_HI'}
_MSG = {128:'MSG_RTN_GET_DOORBELL',129:'MSG_RTN_GET_DDID',130:'MSG_RTN_GET_TMA',131:'MSG_RTN_GET_REALTIME',
  132:'MSG_RTN_SAVE_WAVE',133:'MSG_RTN_GET_TBA',134:'MSG_RTN_GET_TBA_TO_PC',135:'MSG_RTN_GET_SE_AID_ID'}

# VOP1/VOP2/VOPC
def _disasm_vop1(i) -> str:
  nm, cdna = i.op_name.lower().replace('_e32', ''), _cdna(i)
  if any(x in nm for x in ('v_nop', 'v_pipeflush', 'v_clrexcp')): return nm
  if 'readfirstlane' in nm:
    s = _v(i.src0) if _off(i.src0) >= 256 else _s(_off(i.src0), 1, cdna)
    return f"{nm} {_s(_off(i.vdst) - 256 if _off(i.vdst) >= 256 else _off(i.vdst), 1, cdna)}, {s}"
  b = i.canonical_op_bits
  dn, sn = max(1, b['d']//32), max(1, b['s0']//32)
  is16d, is16s = not cdna and b['d'] == 16, not cdna and b['s0'] == 16
  if 'cvt_pk_f32_fp8' in nm or 'cvt_pk_f32_bf8' in nm: dn, is16s = 2, True
  if is16d: vd = _off(i.vdst) - 256; dst = f"v{vd & 0x7f}.{'h' if vd >= 128 else 'l'}"
  else: dst = _v(i.vdst, dn)
  s0 = _off(i.src0)
  if s0 == 255: s = _lit(i, i.src0)
  elif is16s and s0 >= 256: s = f"v{(s0-256) & 0x7f}.{'h' if s0-256 >= 128 else 'l'}"
  elif sn > 1: s = _s(i.src0, sn, cdna)
  else: s = _lit(i, i.src0)
  return f"{nm} {dst}, {s}"

_VOP2_CI = {'v_add_co_ci_u32', 'v_sub_co_ci_u32', 'v_subrev_co_ci_u32', 'v_addc_co_u32', 'v_subb_co_u32', 'v_subbrev_co_u32'}
def _disasm_vop2(i) -> str:
  nm, cdna = i.op_name.lower(), _cdna(i)
  suf = "" if cdna or nm.endswith('_e32') or (not cdna and i.op == VOP2Op.V_DOT2ACC_F32_F16_E32) else "_e32"
  lit, b = getattr(i, '_literal', None), i.canonical_op_bits
  is16 = not cdna and b['d'] == 16
  def v16s(v): return _v16(v) if is16 and _off(v) >= 256 else _lit(i, v)  # 16-bit src
  if 'fmaak' in nm or 'madak' in nm:
    if lit is None: return f"op_{i.op.value if hasattr(i.op,'value') else i.op}"
    return f"{nm}{suf} {_v16(i.vdst) if is16 else _v(i.vdst)}, {v16s(i.src0)}, {_v16(i.vsrc1) if is16 else _v(i.vsrc1)}, 0x{lit:x}"
  if 'fmamk' in nm or 'madmk' in nm:
    if lit is None: return f"op_{i.op.value if hasattr(i.op,'value') else i.op}"
    return f"{nm}{suf} {_v16(i.vdst) if is16 else _v(i.vdst)}, {v16s(i.src0)}, 0x{lit:x}, {_v16(i.vsrc1) if is16 else _v(i.vsrc1)}"
  vcc = "vcc" if cdna else "vcc_lo"
  if nm in _VOP2_CI: return f"{nm}{suf} {_v(i.vdst)}, {vcc}, {_lit(i, i.src0)}, {_v(i.vsrc1)}, {vcc}"
  r = i.canonical_op_regs
  dn, s0n, s1n = r.get('d',1), r.get('s0',1), r.get('s1',1)
  if dn > 1 or s0n > 1 or s1n > 1:
    return f"{nm.replace('_e32','')} {_v(i.vdst, dn)}, {_lit(i, i.src0) if _off(i.src0) == 255 else _s(i.src0, s0n, cdna)}, {_v(i.vsrc1, s1n)}"
  if is16: return f"{nm}{suf} {_v16(i.vdst)}, {v16s(i.src0)}, {_v16(i.vsrc1)}"
  return f"{nm}{suf} {_v(i.vdst)}, {_lit(i, i.src0)}, {_v(i.vsrc1)}" + (f", {vcc}" if nm == 'v_cndmask_b32' else "")

def _disasm_vopc(i) -> str:
  nm, cdna, b = i.op_name.lower(), _cdna(i), i.canonical_op_bits
  r0, r1, is16 = max(1, b['s0']//32), max(1, b['s1']//32), b['s0'] == 16
  s0 = _lit(i, i.src0) if _off(i.src0) == 255 else (_v16(i.src0) if is16 and _off(i.src0) >= 256 else _s(i.src0, r0, cdna))
  s1 = _v16(i.vsrc1) if is16 else _v(i.vsrc1, r1)
  if cdna: return f"{nm} vcc, {s0}, {s1}"
  suf = "" if nm.endswith('_e32') else "_e32"
  return f"{nm}{suf} vcc_lo, {s0}, {s1}" if 'cmpx' not in nm else f"{nm}{suf} {s0}, {s1}"

# Scalar ops
_NO_ARG_SOPP = {'s_barrier','s_wakeup','s_icache_inv','s_ttracedata','s_wait_idle','s_endpgm_saved','s_endpgm_ordered_ps_done','s_code_end','s_endpgm'}
def _disasm_sopp(i) -> str:
  nm, cdna, r4 = i.op_name.lower(), _cdna(i), _r4(i)
  if nm in _NO_ARG_SOPP: return nm if i.simm16 == 0 else f"{nm} {i.simm16}"
  if nm == 's_waitcnt':
    if r4: return f"{nm} {i.simm16}" if i.simm16 else f"{nm} 0"
    if cdna: vm, lgkm, exp = i.simm16 & 0xf, (i.simm16 >> 8) & 0x3f, (i.simm16 >> 4) & 0x7
    else: vm, exp, lgkm = (i.simm16 >> 10) & 0x3f, i.simm16 & 0xf, (i.simm16 >> 4) & 0x3f
    p = [f"vmcnt({vm})" if vm != (0xf if cdna else 0x3f) else "", f"expcnt({exp})" if exp != 7 else "", f"lgkmcnt({lgkm})" if lgkm != 0x3f else ""]
    return f"s_waitcnt {' '.join(x for x in p if x) or '0'}"
  if nm == 's_delay_alu':
    deps = ['VALU_DEP_1','VALU_DEP_2','VALU_DEP_3','VALU_DEP_4','TRANS32_DEP_1','TRANS32_DEP_2','TRANS32_DEP_3','FMA_ACCUM_CYCLE_1','SALU_CYCLE_1','SALU_CYCLE_2','SALU_CYCLE_3']
    skips = ['SAME','NEXT','SKIP_1','SKIP_2','SKIP_3','SKIP_4']
    id0, skip, id1 = i.simm16 & 0xf, (i.simm16 >> 4) & 0x7, (i.simm16 >> 7) & 0xf
    dep = lambda v: deps[v-1] if 0 < v <= len(deps) else str(v)
    p = [f"instid0({dep(id0)})" if id0 else "", f"instskip({skips[skip]})" if skip else "", f"instid1({dep(id1)})" if id1 else ""]
    return f"s_delay_alu {' | '.join(x for x in p if x) or '0'}"
  if nm.startswith(('s_cbranch', 's_branch')): return f"{nm} {'0x' if not cdna else ''}{i.simm16:x}" if not cdna else f"{nm} {i.simm16}"
  return f"{nm} 0x{i.simm16:x}" if i.simm16 or not cdna else nm

def _disasm_sop1(i) -> str:
  nm, cdna, r = i.op_name.lower(), _cdna(i), i.canonical_op_regs
  dn, sn = r.get('d', 1), r.get('s0', 1)
  s = _lit(i, i.ssrc0) if _off(i.ssrc0) == 255 else _s(i.ssrc0, sn, cdna)
  if 'getpc_b64' in nm: return f"{nm} {_s(i.sdst, 2)}"
  if 'setpc_b64' in nm or 'rfe_b64' in nm: return f"{nm} {s}"
  if 'swappc_b64' in nm: return f"{nm} {_s(i.sdst, 2)}, {s}"
  if 'sendmsg_rtn' in nm and not cdna:
    v = _off(i.ssrc0)
    return f"{nm} {_s(i.sdst, dn)}, sendmsg({_MSG.get(v, f'0x{v:x}')})"
  if i.op_name.upper() in ('S_ALLOC_VGPR','S_SLEEP_VAR','S_BARRIER_SIGNAL','S_BARRIER_SIGNAL_ISFIRST','S_BARRIER_INIT','S_BARRIER_JOIN'): return f"{nm} {s}"
  return f"{nm} {_s(i.sdst, dn)}, {s}"

def _disasm_sop2(i) -> str:
  nm, cdna, r, lit = i.op_name.lower(), _cdna(i), i.canonical_op_regs, getattr(i, '_literal', None)
  dn, s0n, s1n = r['d'], r['s0'], r['s1']
  s0 = _lit(i, i.ssrc0) if _off(i.ssrc0) == 255 else _s(i.ssrc0, s0n, cdna)
  s1 = _lit(i, i.ssrc1) if _off(i.ssrc1) == 255 else _s(i.ssrc1, s1n, cdna)
  dst = _s(i.sdst, dn, cdna)
  if 'fmamk' in nm and lit is not None: return f"{nm} {dst}, {s0}, 0x{lit:x}, {s1}"
  if 'fmaak' in nm and lit is not None: return f"{nm} {dst}, {s0}, {s1}, 0x{lit:x}"
  return f"{nm} {dst}, {s0}, {s1}"

def _disasm_sopc(i) -> str:
  cdna, r = _cdna(i), i.canonical_op_regs
  s0 = _lit(i, i.ssrc0) if _off(i.ssrc0) == 255 else _s(i.ssrc0, r['s0'], cdna)
  s1 = _lit(i, i.ssrc1) if _off(i.ssrc1) == 255 else _s(i.ssrc1, r['s1'], cdna)
  return f"{i.op_name.lower()} {s0}, {s1}"

def _disasm_sopk(i) -> str:
  nm, cdna, r4 = i.op_name.lower(), _cdna(i), _r4(i)
  r = i.canonical_op_regs
  dn = r.get('d', 1)
  simm = i.simm16 if i.simm16 < 32768 else i.simm16 - 65536
  if 'setreg' in nm:
    hw = HWREG_RDNA4 if r4 else HWREG
    id_, off, sz = i.simm16 & 0x3f, (i.simm16 >> 6) & 0x1f, ((i.simm16 >> 11) & 0x1f) + 1
    hwreg_s = f"hwreg({hw.get(id_, id_)}, {off}, {sz})"
    if 'imm32' in nm: return f"{nm} {hwreg_s}, {_lit(i, 255)}"  # s_setreg_imm32_b32 takes literal, not register
    return f"{nm} {hwreg_s}, {_s(i.sdst, dn)}"
  if 'getreg' in nm:
    hw = HWREG_RDNA4 if r4 else HWREG
    id_, off, sz = i.simm16 & 0x3f, (i.simm16 >> 6) & 0x1f, ((i.simm16 >> 11) & 0x1f) + 1
    return f"{nm} {_s(i.sdst, dn)}, hwreg({hw.get(id_, id_)}, {off}, {sz})"
  if nm in ('s_call_b64', 's_subvector_loop_begin', 's_subvector_loop_end', 's_cbranch_i_fork'):
    return f"{nm} {_s(i.sdst, dn)}, 0x{i.simm16:x}"
  if nm == 's_version': return f"{nm} 0x{i.simm16:x}"  # s_version takes only immediate
  if 'waitcnt' in nm: return f"{nm} null, 0x{i.simm16:x}"  # s_waitcnt_* needs null prefix
  if 'addk' in nm or 'mulk' in nm: return f"{nm} {_s(i.sdst, dn)}, {simm}"
  return f"{nm} {_s(i.sdst, dn)}, 0x{i.simm16:x}"

# Memory ops
def _disasm_smem(i) -> str:
  nm, cdna, r4 = i.op_name.lower(), _cdna(i), _r4(i)
  if nm in ('s_gl1_inv', 's_dcache_inv'): return nm
  soe, imm = getattr(i, 'soe', 0), getattr(i, 'imm', 1)
  offset = i.ioffset if r4 else getattr(i, 'offset', 0)
  if cdna:
    if soe and imm: off_s = f"{_s(i.soffset, 1, cdna)} offset:0x{offset:x}"
    elif imm: off_s = f"0x{offset:x}"
    elif offset < 256: off_s = _s(offset, 1, cdna)
    else: off_s = _s(i.soffset, 1, cdna)
  elif offset and i.soffset != 124: off_s = f"{_s(i.soffset, 1, cdna)} offset:0x{offset:x}"
  elif offset: off_s = f"0x{offset:x}"
  else: off_s = _s(i.soffset, 1, cdna)
  is_buf = 'buffer' in nm or nm == 's_atc_probe_buffer'
  sbase_n = 4 if is_buf else 2
  sbase_s = _s(_off(i.sbase), sbase_n, cdna) if sbase_n == 2 else (_st(_off(i.sbase), sbase_n))
  if nm in ('s_atc_probe', 's_atc_probe_buffer'): return f"{nm} {_off(i.sdata)}, {sbase_s}, {off_s}"
  if 'prefetch' in nm:
    off = getattr(i, 'ioffset', getattr(i, 'offset', 0))
    if off >= 0x800000: off = off - 0x1000000
    off_s = f"0x{off:x}" if off > 255 else str(off)
    soff_s = _s(i.soffset, 1, cdna) if i.soffset != 124 else "null"
    if 'pc_rel' in nm: return f"{nm} {off_s}, {soff_s}, {_off(i.sdata)}"
    return f"{nm} {sbase_s}, {off_s}, {soff_s}, {_off(i.sdata)}"
  dn, th, scope = i.canonical_op_regs.get('d', 1), getattr(i, 'th', 0), getattr(i, 'scope', 0)
  dst = f"{nm} {_s(i.sdata, dn, cdna)}, {sbase_s}, {off_s}"
  if th or scope or r4:
    return dst + (f" th:{['TH_LOAD_RT','TH_LOAD_NT','TH_LOAD_HT','TH_LOAD_LU'][th]}" if th else "") + (f" scope:{['SCOPE_CU','SCOPE_SE','SCOPE_DEV','SCOPE_SYS'][scope]}" if scope else "")
  return dst + _mods((i.glc, " glc"), (getattr(i, 'dlc', 0), " dlc"))

def _disasm_flat(i) -> str:
  nm, cdna = i.op_name.lower(), _cdna(i)
  acc, reg = getattr(i, 'acc', 0), _a if getattr(i, 'acc', 0) else _v
  seg = ['flat', 'scratch', 'global'][i.seg] if i.seg < 3 else 'flat'
  instr = f"{seg}_{nm.split('_', 1)[1] if '_' in nm else nm}"
  off_val = i.offset if seg == 'flat' else (i.offset if i.offset < 4096 else i.offset - 8192)
  r = i.canonical_op_regs
  w = r.get('data', r.get('d', 1)) if 'store' in nm or 'atomic' in nm else r.get('d', 1)
  off_s = f" offset:{off_val}" if off_val else ""
  mods = f"{off_s}{' glc' if (i.sc0 if cdna else i.glc) else ''}{' slc' if (i.nt if cdna else i.slc) else ''}" + ("" if cdna else f"{' dlc' if i.dlc else ''}")
  saddr = _off(i.saddr)
  if seg == 'flat' or saddr == 0x7F: saddr_s = ""
  elif saddr == 124: saddr_s = ", off"
  elif seg == 'scratch': saddr_s = f", {_s(i.saddr, 1, cdna)}"
  elif t := _t(i.saddr, 2): saddr_s = f", {t}"
  else: saddr_s = f", {_s(saddr, 2, cdna)}"
  if 'addtid' in nm: return f"{instr} {reg(i.data if 'store' in nm else i.vdst)}{saddr_s}{mods}"
  addr_w = 1 if seg == 'scratch' else (2 if cdna else (1 if saddr not in (0x7F, 124) else 2))
  addr_s = "off" if not i.sve and seg == 'scratch' else _v(i.addr, addr_w)
  if 'atomic' in nm:
    return f"{instr} {reg(i.vdst, w // 2 if 'cmpswap' in nm else w)}, {addr_s}, {reg(i.data, w)}{saddr_s if seg != 'flat' else ''}{mods}" if (i.sc0 if cdna else i.glc) else f"{instr} {addr_s}, {reg(i.data, w)}{saddr_s if seg != 'flat' else ''}{mods}"
  if 'store' in nm: return f"{instr} {addr_s}, {reg(i.data, w)}{saddr_s}{mods}"
  return f"{instr} {reg(i.vdst, w)}, {addr_s}{saddr_s}{mods}"

def _disasm_ds(i) -> str:
  nm, reg = i.op_name.lower(), _a if getattr(i, 'acc', 0) else _v
  gds = " gds" if getattr(i, 'gds', 0) else ""
  off = f" offset:{i.offset0 | (i.offset1 << 8)}" if i.offset0 or i.offset1 else ""
  off2 = (f" offset0:{i.offset0}" if i.offset0 else "") + (f" offset1:{i.offset1}" if i.offset1 else "")
  r = i.canonical_op_regs
  w = r.get('data', r.get('d', 1)) if 'store' in nm or 'write' in nm or ('load' not in nm and 'read' not in nm) else r.get('d', 1)
  d0, d1, dst, addr = reg(i.data0, w), reg(i.data1, w), reg(i.vdst, w), _v(i.addr)
  if nm == 'ds_nop': return nm
  if 'bvh_stack' in nm:
    d1n = 8 if 'push8' in nm else 4
    vdstn = 2 if 'pop2' in nm else 1
    return f"{nm} {_v(i.vdst, vdstn)}, {addr}, {_v(i.data0)}, {_v(i.data1, d1n)}{off}{gds}"
  if 'gws_sema' in nm and 'sema_br' not in nm: return f"{nm}{off}{gds}"
  if 'gws_' in nm: return f"{nm} {addr}{off}{gds}"
  if nm in ('ds_consume', 'ds_append'): return f"{nm} {reg(i.vdst)}{off}{gds}"
  if 'gs_reg' in nm: return f"{nm} {reg(i.vdst, 2)}, {reg(i.data0)}{off}{gds}"
  if '2addr' in nm or 'write2' in nm or 'read2' in nm:
    if 'load' in nm or 'read' in nm: return f"{nm} {reg(i.vdst, r.get('d', 1))}, {addr}{off2}{gds}"
    if 'store' in nm and 'xchg' not in nm: return f"{nm} {addr}, {d0}, {d1}{off2}{gds}"
    return f"{nm} {reg(i.vdst, r.get('d', 1))}, {addr}, {d0}, {d1}{off2}{gds}"
  if 'load' in nm: return f"{nm} {reg(i.vdst)}{off}{gds}" if 'addtid' in nm else f"{nm} {dst}, {addr}{off}{gds}"
  if 'store' in nm and not any(x in nm for x in ('cmp', 'xchg')):
    return f"{nm} {reg(i.data0)}{off}{gds}" if 'addtid' in nm else f"{nm} {addr}, {d0}{off}{gds}"
  if 'swizzle' in nm or nm == 'ds_ordered_count': return f"{nm} {reg(i.vdst)}, {addr}{off}{gds}"
  if 'permute' in nm: return f"{nm} {reg(i.vdst)}, {addr}, {reg(i.data0)}{off}{gds}"
  if 'condxchg' in nm: return f"{nm} {reg(i.vdst, 2)}, {addr}, {reg(i.data0, 2)}{off}{gds}"
  if any(x in nm for x in ('cmpstore', 'mskor', 'wrap')):
    return f"{nm} {dst}, {addr}, {d0}, {d1}{off}{gds}" if '_rtn' in nm else f"{nm} {addr}, {d0}, {d1}{off}{gds}"
  return f"{nm} {dst}, {addr}, {d0}{off}{gds}" if '_rtn' in nm else f"{nm} {addr}, {d0}{off}{gds}"

# VOP3/VOP3P
def _disasm_vop3(i) -> str:
  nm, b = i.op_name.lower(), i.canonical_op_bits
  if nm.startswith('v_s_'):  # RDNA4 scalar VOP3
    s = _lit(i, i.src0) if _off(i.src0) == 255 else ("src_scc" if _off(i.src0) == 253 else _s(i.src0, max(1, b['s0']//32)))
    if i.neg & 1: s = f"-{s}"
    if i.abs & 1: s = f"|{s}|"
    vd = _off(i.vdst)
    return f"{nm} s{vd - 256 if vd >= 256 else vd}, {s}" + (" clamp" if getattr(i, 'cm', None) or getattr(i, 'clmp', 0) else "") + _omod(i.omod)
  r0, r1, r2, dn = max(1, b['s0']//32), max(1, b['s1']//32), max(1, b['s2']//32), max(1, b['d']//32)
  is16d, is16s = b['d'] == 16, b['s0'] == 16
  s0 = _vop3_src(i, i.src0, i.neg&1, i.abs&1, i.opsel&1, r0, is16s)
  s1 = _vop3_src(i, i.src1, i.neg&2, i.abs&2, i.opsel&2, r1, is16s)
  s2 = _vop3_src(i, i.src2, i.neg&4, i.abs&4, i.opsel&4, r2, b['s2'] == 16)
  if 'readlane' in nm: vd = _off(i.vdst); dst = _s(vd - 256 if vd >= 256 else vd, 1)
  elif dn > 1: dst = _v(i.vdst, dn)
  elif is16d: dst = f"{_v(i.vdst)}.h" if (i.opsel & 8) else f"{_v(i.vdst)}.l"
  else: dst = _v(i.vdst)
  cl, om = " clamp" if getattr(i, 'cm', None) or getattr(i, 'clmp', 0) else "", _omod(i.omod)
  nonvgpr_opsel = (_off(i.src0) < 256 and (i.opsel & 1)) or (_off(i.src1) < 256 and (i.opsel & 2)) or (_off(i.src2) < 256 and (i.opsel & 4))
  need_opsel = nonvgpr_opsel or (i.opsel and not is16s)
  def opsel_s(n):
    if not need_opsel: return ""
    dh = (i.opsel >> 3) & 1
    if n == 1: return f" op_sel:[{i.opsel & 1},{dh}]"
    if n == 2: return f" op_sel:[{i.opsel & 1},{(i.opsel >> 1) & 1},{dh}]"
    return f" op_sel:[{i.opsel & 1},{(i.opsel >> 1) & 1},{(i.opsel >> 2) & 1},{dh}]"
  op_val = i.op.value if hasattr(i.op, 'value') else i.op
  e64 = "" if nm.endswith('_e64') else "_e64"
  if op_val < 256:  # VOPC
    vd = _off(i.vdst) - 256 if _off(i.vdst) >= 256 else _off(i.vdst)
    return f"{nm}{e64} {s0}, {s1}{cl}" if nm.startswith('v_cmpx') else f"{nm}{e64} {_s(vd, 1)}, {s0}, {s1}{cl}"
  n = i.num_srcs() or 2
  if op_val < 384: os = opsel_s(n); return f"{nm}{e64} {dst}, {s0}, {s1}, {s2}{os}{cl}{om}" if n == 3 else f"{nm}{e64} {dst}, {s0}, {s1}{os}{cl}{om}"
  if op_val < 512:
    os = f" byte_sel:{((i.opsel & 1) << 1) | ((i.opsel >> 1) & 1)}" if re.match(r'v_cvt_f32_(bf|fp)8', nm) and i.opsel else opsel_s(1)
    if 'v_nop' in nm or 'v_pipeflush' in nm: return f"{nm}{e64}"
    return f"{nm}{e64} {dst}, {s0}{os}{cl}{om}"
  os = f" byte_sel:{i.opsel >> 2}" if 'cvt_sr' in nm and i.opsel else opsel_s(n)
  return f"{nm} {dst}, {s0}, {s1}, {s2}{os}{cl}{om}" if n == 3 else f"{nm} {dst}, {s0}, {s1}{os}{cl}{om}"

def _disasm_vop3sd(i) -> str:
  nm, r = i.op_name.lower(), i.canonical_op_regs
  dn, sr0, sr1, sr2 = r['d'], r['s0'], r['s1'], r['s2']
  def s(v, neg, n):
    v = _off(v)
    r = _lit(i, v) if v == 255 else ("src_scc" if v == 253 else (_s(v, n) if n > 1 else _lit(i, v)))
    return f"neg({r})" if neg and v == 255 else (f"-{r}" if neg else r)
  s0, s1, s2 = s(i.src0, i.neg & 1, sr0), s(i.src1, i.neg & 2, sr1), s(i.src2, i.neg & 4, sr2)
  has2 = '_co_' in nm and '_ci_' not in nm and 'mad' not in nm
  srcs = f"{s0}, {s1}" if has2 else f"{s0}, {s1}, {s2}"
  return f"{nm} {_v(i.vdst, dn)}, {_s(i.sdst, 1)}, {srcs}{' clamp' if getattr(i, 'cm', None) or getattr(i, 'clmp', 0) else ''}{_omod(i.omod)}"

def _disasm_vopd(i) -> str:
  lit, r4 = i._literal or getattr(i, 'literal', None), _r4(i)
  op_enum = R4_VOPDOp if r4 else VOPDOp
  vdst_y, nx, ny = (_off(i.vdsty) << 1) | ((_off(i.vdstx) & 1) ^ 1), op_enum(i.opx).name.lower(), op_enum(i.opy).name.lower()
  def half(n, vd, s0, vs1):
    vd, vs1 = _vi(vd), _vi(vs1)
    if 'mov' in n: return f"{n} v{vd}, {_lit(i, s0)}"
    if 'fmamk' in n: return f"{n} v{vd}, {_lit(i, s0)}, 0x{lit:x}, v{vs1}"
    if 'fmaak' in n: return f"{n} v{vd}, {_lit(i, s0)}, v{vs1}, 0x{lit:x}"
    return f"{n} v{vd}, {_lit(i, s0)}, v{vs1}"
  return f"{half(nx, i.vdstx, i.srcx0, i.vsrcx1)} :: {half(ny, vdst_y, i.srcy0, i.vsrcy1)}"

def _disasm_vop3p(i) -> str:
  nm, n = i.op_name.lower(), i.num_srcs() or 2
  is_wmma, is_swmmac, is_mix = 'wmma' in nm, 'swmmac' in nm, 'fma_mix' in nm
  r = i.canonical_op_regs
  sc0, sc1, sc2, scd = r['s0'], r['s1'], r['s2'], r['d']
  if is_swmmac:
    if 'f16_16x16x32' in nm or 'bf16_16x16x32' in nm: scd, sc0, sc1, sc2 = 4, 4, 8, 1
    elif 'f32_16x16x32_f16' in nm or 'f32_16x16x32_bf16' in nm: scd, sc0, sc1, sc2 = 8, 4, 8, 1
    elif 'i32_16x16x32_iu4' in nm: scd, sc0, sc1, sc2 = 8, 1, 2, 1
    elif 'i32_16x16x64_iu4' in nm: scd, sc0, sc1, sc2 = 8, 2, 4, 1
    elif 'i32_16x16x32_iu8' in nm or 'f32_16x16x32_fp8' in nm or 'f32_16x16x32_bf8' in nm: scd, sc0, sc1, sc2 = 8, 2, 4, 1
  get_s = lambda v, sc: _lit(i, v) if _off(v) == 255 else _s(v, sc)
  opsel_hi = i.opsel_hi | (i.opsel_hi2 << 2) if hasattr(i, 'opsel_hi2') else i.opsel_hi
  mods = []
  if is_mix:
    # FMA_MIX uses -src prefix for neg, |src| for abs (neg_hi bit encodes abs), default opsel_hi is 0
    def fmt_mix(s, neg, abs_): return f"-|{s}|" if neg and abs_ else (f"|{s}|" if abs_ else (f"-{s}" if neg else s))
    src0 = fmt_mix(get_s(i.src0, sc0), i.neg & 1, i.neg_hi & 1)
    src1 = fmt_mix(get_s(i.src1, sc1), i.neg & 2, i.neg_hi & 2)
    src2 = fmt_mix(get_s(i.src2, sc2), i.neg & 4, i.neg_hi & 4)
    dst = _v(i.vdst, scd)
    if i.opsel: mods.append(_bits("op_sel", i.opsel, n))
    if opsel_hi: mods.append(_bits("op_sel_hi", opsel_hi, n))  # FMA_MIX default opsel_hi is 0, not 7
  elif not is_wmma and not is_swmmac:
    src0, src1, src2, dst = get_s(i.src0, sc0), get_s(i.src1, sc1), get_s(i.src2, sc2), _v(i.vdst, scd)
    if i.opsel: mods.append(_bits("op_sel", i.opsel, n))
    if opsel_hi != (7 if n == 3 else 3): mods.append(_bits("op_sel_hi", opsel_hi, n))
    if i.neg: mods.append(_bits("neg_lo", i.neg, n))
    if i.neg_hi: mods.append(_bits("neg_hi", i.neg_hi, n))
  else:
    # WMMA/SWMMAC
    src0, src1, src2, dst = get_s(i.src0, sc0), get_s(i.src1, sc1), get_s(i.src2, sc2), _v(i.vdst, scd)
    # Only WMMA f16/bf16 output supports op_sel; SWMMAC uses opsel for index_key
    has_opsel = is_wmma and ('f16_16x16x16_f16' in nm or 'bf16_16x16x16_bf16' in nm)
    if has_opsel and i.opsel: mods.append(f"op_sel:[{i.opsel & 1},{(i.opsel >> 1) & 1},{(i.opsel >> 2) & 1}]")
    if hasattr(i, 'neg') and i.neg: mods.append(_bits("neg_lo", i.neg, n))
    if hasattr(i, 'neg_hi') and i.neg_hi: mods.append(_bits("neg_hi", i.neg_hi, n))
    index_key = getattr(i, 'index_key', None) or (i.opsel if is_swmmac else 0)  # SWMMAC uses opsel for index_key
    if index_key: mods.append(f"index_key:{index_key}")
  if getattr(i, 'clmp', 0): mods.append("clamp")
  return f"{nm} {dst}, {src0}, {src1}, {src2}{' ' + ' '.join(mods) if mods else ''}" if n == 3 else f"{nm} {dst}, {src0}, {src1}{' ' + ' '.join(mods) if mods else ''}"

# Buffer/image ops
def _disasm_buf(i) -> str:
  nm, cdna = i.op_name.lower(), _cdna(i)
  if nm in ('buffer_gl0_inv', 'buffer_gl1_inv'): return nm  # no operands
  r = i.canonical_op_regs
  w = r.get('data', r.get('d', 1))  # use canonical_op_regs data field for width
  if hasattr(i, 'tfe') and i.tfe: w += 1
  reg = _a if getattr(i, 'acc', 0) else _v
  vaddr = _v(i.vaddr, 2) if i.offen and i.idxen else _v(i.vaddr) if i.offen or i.idxen else "off"
  srsrc = _st(_off(i.srsrc), 4)
  is_mtbuf = 'mtbuf' in i.__class__.__name__.lower()
  if is_mtbuf:
    from extra.assembly.amd.asm import BUF_FMT
    fmt_names = {v: k for k, v in BUF_FMT.items()}
    fmt_s = f" format:[{fmt_names.get(i.format, f'fmt{i.format}')}]" if i.format else ""
  else: fmt_s = ""
  soffset_s = _s(i.soffset, 1, cdna)
  mods = _mods((i.idxen, "idxen"), (i.offen, "offen"), (i.offset, f"offset:{i.offset}"), (i.glc, "glc"), (getattr(i, 'slc', 0), "slc"), (getattr(i, 'dlc', 0), "dlc"), (getattr(i, 'tfe', 0), "tfe"))
  # MUBUF always has vdata, vaddr order for RDNA
  return f"{nm} {reg(i.vdata, w)}, {vaddr}, {srsrc}, {soffset_s}{fmt_s}" + (" " + mods if mods else "")

def _disasm_mimg(i) -> str:
  nm = i.op_name.lower()
  srsrc_s = _st(_off(i.srsrc), 8)
  if 'bvh' in nm:  # BVH uses 4-reg resource descriptor
    va = (9 if '64' in nm else 8) if i.a16 else (12 if '64' in nm else 11)
    return f"{nm} {_v(i.vdata, 4)}, {_v(i.vaddr, va)}, {_st(_off(i.srsrc), 4)}" + (" a16" if i.a16 else "")
  # vdata size: count bits in dmask, halve if d16, add 1 if tfe
  dmask_bits = i.dmask.bit_count() if hasattr(i.dmask, 'bit_count') else bin(i.dmask).count('1')
  d16, tfe = getattr(i, 'd16', 0), getattr(i, 'tfe', 0)
  is_sample, is_gather, is_msaa = 'sample' in nm or 'get_lod' in nm, 'gather' in nm, 'msaa' in nm
  vdata = 4 if is_gather or is_msaa else max(1, dmask_bits)  # gather/msaa always 4 channels
  if d16: vdata = (vdata + 1) // 2  # d16 packs 2 f16 per dword
  if tfe: vdata += 1  # tfe adds 1 dword for status
  # Build modifiers (emit all flags - hardware valid even if LLVM rejects some combos)
  parts = set(nm.split('_'))
  mods = [f"dmask:0x{i.dmask:x}", f"dim:SQ_RSRC_IMG_{['1D','2D','3D','CUBE','1D_ARRAY','2D_ARRAY','2D_MSAA','2D_MSAA_ARRAY'][i.dim]}"]
  for flag, mod in [(i.unrm, "unorm"), (i.glc, "glc"), (getattr(i, 'slc', 0), "slc"), (getattr(i, 'dlc', 0), "dlc"), (i.r128, "r128"),
                    (i.a16, "a16"), (tfe, "tfe"), (getattr(i, 'lwe', 0), "lwe"), (d16, "d16")]:
    if flag: mods.append(mod)
  # get_resinfo only takes MIP level (1 vaddr), not coordinates
  if 'resinfo' in parts: return f"{nm} {_v(i.vdata, vdata)}, {_v(i.vaddr)}, {srsrc_s} {' '.join(mods)}"
  # vaddr size: bias/compare/offset are f32, coords/lod are a16-packable
  dim_coords = [1, 2, 3, 3, 2, 3, 3, 4][i.dim]  # 1D, 2D, 3D, CUBE, 1D_ARRAY, 2D_ARRAY, 2D_MSAA, 2D_MSAA_ARRAY
  grad_coords = [1, 2, 3, 2, 1, 2, 2, 2][i.dim]  # coords for gradient (CUBE uses 2, not 3)
  # Packable components (affected by a16): coords and lod (not get_lod - it's a query)
  packable = dim_coords
  if ('l' in parts or ('lod' in parts and 'get' not in parts) or 'mip' in parts) and 'lz' not in parts and 'cl' not in parts: packable += 1
  if 'cl' in parts: packable += 1  # lod clamp
  coord_addr = (packable + 1) // 2 if i.a16 else packable  # a16 packs pairs
  # Non-packable components (always f32): compare, bias, offset
  if 'c' in parts and (is_sample or is_gather): coord_addr += 1  # compare
  if 'b' in parts: coord_addr += 1  # bias
  if 'o' in parts and (is_sample or is_gather): coord_addr += 1  # offset
  # Derivatives
  deriv_addr = 0
  if 'd' in parts and 'atomic' not in nm:
    g16 = 'g16' in parts
    deriv_addr = ((grad_coords + 1) // 2) * 2 if g16 else grad_coords * 2
  vaddr_w = coord_addr + deriv_addr
  ssamp_s = f", {_st(_off(i.ssamp), 4)}" if is_sample or is_gather else ""
  return f"{nm} {_v(i.vdata, vdata)}, {_v(i.vaddr, vaddr_w)}, {srsrc_s}{ssamp_s} {' '.join(mods)}"

def _disasm_vinterp(i) -> str:
  nm, r, b = i.op_name.lower(), i.canonical_op_regs, i.canonical_op_bits
  r0, r1, r2 = max(1, b['s0']//32), max(1, b['s1']//32), max(1, b['s2']//32)
  # vinterp uses op_sel for hi/lo selection, not .l/.h suffix
  s0 = _vop3_src(i, i.src0, i.neg&1, 0, 0, r0, False)
  s1 = _vop3_src(i, i.src1, i.neg&2, 0, 0, r1, False)
  s2 = _vop3_src(i, i.src2, i.neg&4, 0, 0, r2, False)
  mods = f" clamp" if getattr(i, 'clmp', 0) else ""
  if i.opsel: mods += f" op_sel:[{i.opsel&1},{(i.opsel>>1)&1},{(i.opsel>>2)&1},{(i.opsel>>3)&1}]"
  mods += f" wait_exp:{i.waitexp}"
  return f"{nm} {_v(i.vdst, r.get('d', 1))}, {s0}, {s1}, {s2}{mods}"

def _disasm_vexport(i) -> str:
  nm = 'export' if not hasattr(i, 'op_name') else i.op_name.lower()
  tgt = getattr(i, 'tgt', None) or getattr(i, 'target', 0)  # RDNA3 uses tgt, RDNA4 uses target
  if tgt == 0: tgt_s = "mrt0"
  elif tgt <= 7: tgt_s = f"mrt{tgt}"
  elif tgt == 8: tgt_s = "mrtz"
  elif tgt == 9: tgt_s = "null"
  elif 12 <= tgt <= 15: tgt_s = f"pos{tgt - 12}"
  elif 32 <= tgt <= 63: tgt_s = f"param{tgt - 32}"
  else: tgt_s = f"tgt{tgt}"
  vsrcs = [f"v{_vi(i.vsrc0)}" if i.en & 1 else "off", f"v{_vi(i.vsrc1)}" if i.en & 2 else "off",
           f"v{_vi(i.vsrc2)}" if i.en & 4 else "off", f"v{_vi(i.vsrc3)}" if i.en & 8 else "off"]
  row_s = f" row" if getattr(i, 'row', 0) else ""
  return f"{nm} {tgt_s}, {', '.join(vsrcs)}" + (" done" if i.done else "") + (" vm" if getattr(i, 'vm', 0) else "") + row_s

def _disasm_vbuffer(i) -> str:
  nm = i.op_name.lower()
  r = i.canonical_op_regs
  w = r.get('data', r.get('d', 1))  # use data field for width
  vdata = _v(i.vdata, w) if w else _v(i.vdata)
  vaddr = _v(i.vaddr, 2) if i.offen and i.idxen else (_v(i.vaddr) if i.offen or i.idxen else 'off')
  srsrc = f'ttmp[{i.rsrc - 108}:{i.rsrc - 108 + 3}]' if i.rsrc >= 108 else f's[{i.rsrc}:{i.rsrc + 3}]'
  soff = _s(i.soffset, 1) if _off(i.soffset) >= 106 else f's{_off(i.soffset)}'
  # Only tbuffer operations have format
  fmt_s = ""
  if 'tbuffer' in nm:
    fmt = getattr(i, 'format', 0)
    from extra.assembly.amd.asm import BUF_FMT
    fmt_names = {v: k for k, v in BUF_FMT.items()}
    fmt_s = f" format:[{fmt_names.get(fmt, str(fmt))}]" if fmt else ""
  th_load = {0: '', 1: 'TH_LOAD_NT', 2: 'TH_LOAD_HT', 3: 'TH_LOAD_BYPASS', 6: 'TH_LOAD_NT_HT'}
  th_store = {0: '', 1: 'TH_STORE_NT', 2: 'TH_STORE_HT', 3: 'TH_STORE_BYPASS', 6: 'TH_STORE_NT_HT'}
  th_atomic = {0: '', 1: 'TH_ATOMIC_RETURN', 2: 'TH_ATOMIC_NT', 3: 'TH_ATOMIC_CASCADE', 6: 'TH_ATOMIC_CASCADE_NT'}
  scope_names = {1: 'SCOPE_SE', 2: 'SCOPE_DEV', 3: 'SCOPE_SYS'}
  is_atomic, is_store = 'atomic' in nm, 'store' in nm
  th_names = th_atomic if is_atomic else (th_store if is_store else th_load)
  mods = _mods((i.idxen, "idxen"), (i.offen, "offen"), (i.ioffset, f"offset:{i.ioffset}"),
               (i.th in th_names and th_names[i.th], f"th:{th_names.get(i.th, '')}"),
               (i.scope in scope_names, f"scope:{scope_names.get(i.scope, '')}"))
  return f"{nm} {vdata}, {vaddr}, {srsrc}, {soff}{fmt_s}" + (" " + mods if mods else "")

# Handler dispatch
DISASM_HANDLERS: dict[type, callable] = {
  VOP1: _disasm_vop1, VOP1_SDST: _disasm_vop1, VOP2: _disasm_vop2, VOPC: _disasm_vopc, VOP3: _disasm_vop3, VOP3_SDST: _disasm_vop3,
  VOP3SD: _disasm_vop3sd, VOPD: _disasm_vopd, VOP3P: _disasm_vop3p, VINTERP: _disasm_vinterp, SOPP: _disasm_sopp, SMEM: _disasm_smem,
  DS: _disasm_ds, FLAT: _disasm_flat, GLOBAL: _disasm_flat, SCRATCH: _disasm_flat, MUBUF: _disasm_buf, MTBUF: _disasm_buf, MIMG: _disasm_mimg,
  SOP1: _disasm_sop1, SOP2: _disasm_sop2, SOPC: _disasm_sopc, SOPK: _disasm_sopk,
  R4_VOP1: _disasm_vop1, R4_VOP1_SDST: _disasm_vop1, R4_VOP2: _disasm_vop2, R4_VOPC: _disasm_vopc, R4_VOP3: _disasm_vop3, R4_VOP3_SDST: _disasm_vop3,
  R4_VOP3SD: _disasm_vop3sd, R4_VOPD: _disasm_vopd, R4_VOP3P: _disasm_vop3p, R4_VINTERP: _disasm_vinterp, R4_SOPP: _disasm_sopp, R4_SMEM: _disasm_smem,
  R4_DS: _disasm_ds, R4_SOP1: _disasm_sop1, R4_SOP2: _disasm_sop2, R4_SOPC: _disasm_sopc, R4_SOPK: _disasm_sopk,
  R4_VEXPORT: _disasm_vexport, R4_VBUFFER: _disasm_vbuffer}

def disasm(inst: Inst) -> str: return DISASM_HANDLERS[type(inst)](inst)

# CDNA support
from extra.assembly.amd.autogen.cdna.ins import (VOP1 as CDNA_VOP1, VOP2 as CDNA_VOP2, VOPC as CDNA_VOPC, VOP3A, VOP3B, VOP3P as CDNA_VOP3P,
  SOP1 as CDNA_SOP1, SOP2 as CDNA_SOP2, SOPC as CDNA_SOPC, SOPK as CDNA_SOPK, SOPP as CDNA_SOPP, SMEM as CDNA_SMEM, DS as CDNA_DS,
  FLAT as CDNA_FLAT, MUBUF as CDNA_MUBUF, MTBUF as CDNA_MTBUF)

def _cdna_src(i, v, neg, abs_=0, n=1):
  s = _lit(i, v) if v == 255 else _s(v, n, cdna=True)
  if abs_: s = f"|{s}|"
  return f"neg({s})" if neg and v == 255 else (f"-{s}" if neg else s)

def _disasm_vop3a(i) -> str:
  op_val = i._values.get('op', 0)
  if hasattr(op_val, 'value'): op_val = op_val.value
  nm = i.op_name.lower() or f'vop3a_op_{op_val}'
  n, cl, om = i.num_srcs() or 2, " clamp" if i.clmp else "", _omod(i.omod)
  r = i.canonical_op_regs
  dn, r0, r1, r2 = r['d'], r['s0'], r['s1'], r['s2']
  s0, s1, s2 = _cdna_src(i, i.src0, i.neg&1, i.abs&1, r0), _cdna_src(i, i.src1, i.neg&2, i.abs&2, r1), _cdna_src(i, i.src2, i.neg&4, i.abs&4, r2)
  dst = _v(i.vdst, dn) if dn > 1 else _v(i.vdst)
  if op_val < 256: return f"{nm}_e64 {_s(i.vdst, 2, cdna=True)}, {s0}, {s1}{cl}"
  if 320 <= op_val < 512:
    if nm in ('v_nop', 'v_clrexcp'): return f"{nm}_e64"
    return f"{nm}_e64 {dst}, {s0}{cl}{om}"
  suf = "_e64" if op_val < 512 else ""
  return f"{nm}{suf} {dst}, {s0}, {s1}, {s2}{cl}{om}" if n == 3 else f"{nm}{suf} {dst}, {s0}, {s1}{cl}{om}"

def _disasm_vop3b(i) -> str:
  op_val = i._values.get('op', 0)
  if hasattr(op_val, 'value'): op_val = op_val.value
  nm, n = i.op_name.lower() or f'vop3b_op_{op_val}', i.num_srcs() or 2
  r = i.canonical_op_regs
  dn, r0, r1, r2 = r['d'], r['s0'], r['s1'], r['s2']
  s0, s1, s2 = _cdna_src(i, i.src0, i.neg&1, n=r0), _cdna_src(i, i.src1, i.neg&2, n=r1), _cdna_src(i, i.src2, i.neg&4, n=r2)
  dst, sdst = _v(i.vdst, dn) if dn > 1 else _v(i.vdst), _s(i.sdst, 2, cdna=True)
  cl, om = " clamp" if i.clmp else "", _omod(i.omod)
  if nm in ('v_addc_co_u32', 'v_subb_co_u32', 'v_subbrev_co_u32'):
    return f"{nm}_e64 {dst}, {sdst}, {s0}, {s1}, {_s(i.src2, 2, cdna=True)}{cl}{om}"
  suf = "_e64" if 'co_' in nm else ""
  return f"{nm}{suf} {dst}, {sdst}, {s0}, {s1}, {s2}{cl}{om}" if n == 3 else f"{nm}{suf} {dst}, {sdst}, {s0}, {s1}{cl}{om}"

def _disasm_cdna_vop3p(i) -> str:
  nm, n, is_mfma = i.op_name.lower(), i.num_srcs() or 2, 'mfma' in i.op_name.lower() or 'smfmac' in i.op_name.lower()
  get_s = lambda v, sc: _lit(i, v) if v == 255 else _s(v, sc, cdna=True)
  if is_mfma:
    sc = 2 if 'iu4' in nm else 4 if 'iu8' in nm or 'i4' in nm else 8 if 'f16' in nm or 'bf16' in nm else 4
    src0, src1, src2, dst = get_s(i.src0, sc), get_s(i.src1, sc), get_s(i.src2, 16), _v(i.vdst, 16)
  else: src0, src1, src2, dst = get_s(i.src0, 1), get_s(i.src1, 1), get_s(i.src2, 1), _v(i.vdst)
  opsel_hi = i.opsel_hi | (i.opsel_hi2 << 2)
  mods = ([_bits("op_sel", i.opsel, n)] if i.opsel else []) + ([_bits("op_sel_hi", opsel_hi, n)] if opsel_hi != (7 if n == 3 else 3) else []) + \
         ([_bits("neg_lo", i.neg, n)] if i.neg else []) + ([_bits("neg_hi", i.neg_hi, n)] if i.neg_hi else []) + (["clamp"] if i.clmp else [])
  return f"{nm} {dst}, {src0}, {src1}, {src2}{' ' + ' '.join(mods) if mods else ''}" if n == 3 else f"{nm} {dst}, {src0}, {src1}{' ' + ' '.join(mods) if mods else ''}"

DISASM_HANDLERS.update({CDNA_VOP1: _disasm_vop1, CDNA_VOP2: _disasm_vop2, CDNA_VOPC: _disasm_vopc,
  CDNA_SOP1: _disasm_sop1, CDNA_SOP2: _disasm_sop2, CDNA_SOPC: _disasm_sopc, CDNA_SOPK: _disasm_sopk, CDNA_SOPP: _disasm_sopp,
  CDNA_SMEM: _disasm_smem, CDNA_DS: _disasm_ds, CDNA_FLAT: _disasm_flat, CDNA_MUBUF: _disasm_buf, CDNA_MTBUF: _disasm_buf,
  VOP3A: _disasm_vop3a, VOP3B: _disasm_vop3b, CDNA_VOP3P: _disasm_cdna_vop3p})
