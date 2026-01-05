# RDNA3/CDNA assembler and disassembler
from __future__ import annotations
import re
from extra.assembly.amd.dsl import Inst, RawImm, Reg, SrcMod, SGPR, VGPR, TTMP, s, v, ttmp, _RegFactory
from extra.assembly.amd.dsl import VCC_LO, VCC_HI, VCC, EXEC_LO, EXEC_HI, EXEC, SCC, M0, NULL, OFF
from extra.assembly.amd.dsl import SPECIAL_GPRS, SPECIAL_PAIRS, SPECIAL_PAIRS_CDNA, FLOAT_DEC, FLOAT_ENC, decode_src
from extra.assembly.amd.autogen.rdna3 import ins
from extra.assembly.amd.autogen.rdna3.ins import (VOP1, VOP2, VOP3, VOP3SD, VOP3P, VOPC, VOPD, VINTERP, SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, DS, FLAT, MUBUF, MTBUF, MIMG, EXP,
  VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOPDOp, SOP1Op, SOPKOp, SOPPOp, SMEMOp, DSOp, MUBUFOp, MTBUFOp)
from extra.assembly.amd.autogen.rdna3.enum import BufFmt

def _is_cdna(inst: Inst) -> bool: return 'cdna' in inst.__class__.__module__

from extra.assembly.amd.autogen.rdna4 import ins
from extra.assembly.amd.autogen.rdna4.ins import (VOP1, VOP2, VOP3, VOP3SD, VOP3P, VOPC, VOPD, VINTERP, SOP1, SOP2, SOPC, SOPK, SOPP, SMEM,
  VBUFFER, VDS, VBUFFEROp, DSOp)

def _matches_encoding(word: int, cls: type[Inst]) -> bool:
  """Check if word matches the encoding pattern of an instruction class."""
  if cls._encoding is None: return False
  bf, val = cls._encoding
  return ((word >> bf.lo) & bf.mask()) == val

# Order matters: more specific encodings first, VOP2 last (it's a catch-all for bit31=0)
_RDNA_FORMATS_64 = [VOPD, VOP3P, VINTERP, VOP3, DS, FLAT, MUBUF, MTBUF, MIMG, SMEM, EXP]
_RDNA_FORMATS_32 = [SOP1, SOPC, SOPP, SOPK, VOPC, VOP1, SOP2, VOP2]  # SOP2/VOP2 are catch-alls
from extra.assembly.amd.autogen.cdna.ins import (VOP1 as C_VOP1, VOP2 as C_VOP2, VOPC as C_VOPC, VOP3A, VOP3B, VOP3P as C_VOP3P,
  SOP1 as C_SOP1, SOP2 as C_SOP2, SOPC as C_SOPC, SOPK as C_SOPK, SOPP as C_SOPP, SMEM as C_SMEM, DS as C_DS,
  FLAT as C_FLAT, MUBUF as C_MUBUF, MTBUF as C_MTBUF, SDWA, DPP)
_CDNA_FORMATS_64 = [C_VOP3P, VOP3A, C_DS, C_FLAT, C_MUBUF, C_MTBUF, C_SMEM]
_CDNA_FORMATS_32 = [SDWA, DPP, C_SOP1, C_SOPC, C_SOPP, C_SOPK, C_VOPC, C_VOP1, C_SOP2, C_VOP2]
_CDNA_VOP3B_OPS = {281, 282, 283, 284, 285, 286, 480, 481, 488, 489}  # VOP3B opcodes
# CDNA opcode name aliases for disasm (new name -> old name expected by tests)
_CDNA_DISASM_ALIASES = {'v_fmac_f64': 'v_mul_legacy_f32', 'v_dot2c_f32_bf16': 'v_mac_f32', 'v_fmamk_f32': 'v_madmk_f32', 'v_fmaak_f32': 'v_madak_f32'}

def detect_format(data: bytes, arch: str = "rdna3") -> type[Inst]:
  """Detect instruction format from machine code bytes."""
  assert len(data) >= 4, f"need at least 4 bytes, got {len(data)}"
  word = int.from_bytes(data[:4], 'little')
  if arch == "cdna":
    if (word >> 30) == 0b11:
      for cls in _CDNA_FORMATS_64:
        if _matches_encoding(word, cls):
          return VOP3B if cls is VOP3A and ((word >> 16) & 0x3ff) in _CDNA_VOP3B_OPS else cls
      raise ValueError(f"unknown CDNA 64-bit format word={word:#010x}")
    for cls in _CDNA_FORMATS_32:
      if _matches_encoding(word, cls): return cls
    raise ValueError(f"unknown CDNA 32-bit format word={word:#010x}")
  # RDNA (default)
  if (word >> 30) == 0b11:
    for cls in _RDNA_FORMATS_64:
      if _matches_encoding(word, cls):
        return VOP3SD if cls is VOP3 and ((word >> 16) & 0x3ff) in Inst._VOP3SD_OPS else cls
    raise ValueError(f"unknown 64-bit format word={word:#010x}")
  for cls in _RDNA_FORMATS_32:
    if _matches_encoding(word, cls): return cls
  raise ValueError(f"unknown 32-bit format word={word:#010x}")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

HWREG = {1: 'HW_REG_MODE', 2: 'HW_REG_STATUS', 3: 'HW_REG_TRAPSTS', 4: 'HW_REG_HW_ID', 5: 'HW_REG_GPR_ALLOC',
         6: 'HW_REG_LDS_ALLOC', 7: 'HW_REG_IB_STS', 15: 'HW_REG_SH_MEM_BASES', 18: 'HW_REG_PERF_SNAPSHOT_PC_LO',
         19: 'HW_REG_PERF_SNAPSHOT_PC_HI', 20: 'HW_REG_FLAT_SCR_LO', 21: 'HW_REG_FLAT_SCR_HI', 22: 'HW_REG_XNACK_MASK',
         23: 'HW_REG_HW_ID1', 24: 'HW_REG_HW_ID2', 25: 'HW_REG_POPS_PACKER', 28: 'HW_REG_IB_STS2'}
HWREG_IDS = {v.lower(): k for k, v in HWREG.items()}
# RDNA unified buffer format - extracted from PDF, use enum for name->value lookup
BUF_FMT = {e.name: e.value for e in BufFmt}
def _parse_buf_fmt_combo(s: str) -> int:  # parse format:[BUF_DATA_FORMAT_X, BUF_NUM_FORMAT_Y]
  parts = [p.strip().replace('BUF_DATA_FORMAT_', '').replace('BUF_NUM_FORMAT_', '') for p in s.split(',')]
  return BUF_FMT.get(f'BUF_FMT_{parts[0]}_{parts[1]}') if len(parts) == 2 else None
MSG = {128: 'MSG_RTN_GET_DOORBELL', 129: 'MSG_RTN_GET_DDID', 130: 'MSG_RTN_GET_TMA',
       131: 'MSG_RTN_GET_REALTIME', 132: 'MSG_RTN_SAVE_WAVE', 133: 'MSG_RTN_GET_TBA'}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _reg(p: str, b: int, n: int = 1) -> str: return f"{p}{b}" if n == 1 else f"{p}[{b}:{b+n-1}]"
def _sreg(b: int, n: int = 1) -> str: return _reg("s", b, n)
def _vreg(b: int, n: int = 1) -> str: return _reg("v", b, n)
def _areg(b: int, n: int = 1) -> str: return _reg("a", b, n)  # accumulator registers for GFX90a
def _ttmp(b: int, n: int = 1) -> str: return _reg("ttmp", b - 108, n) if 108 <= b <= 123 else None
def _sreg_or_ttmp(b: int, n: int = 1) -> str: return _ttmp(b, n) or _sreg(b, n)

def _fmt_sdst(v: int, n: int = 1, cdna: bool = False) -> str:
  from extra.assembly.amd.dsl import SPECIAL_PAIRS_CDNA, SPECIAL_GPRS_CDNA
  if t := _ttmp(v, n): return t
  pairs = SPECIAL_PAIRS_CDNA if cdna else SPECIAL_PAIRS
  gprs = SPECIAL_GPRS_CDNA if cdna else SPECIAL_GPRS
  if n > 1: return pairs.get(v) or gprs.get(v) or _sreg(v, n)  # also check gprs for null/m0
  return gprs.get(v, f"s{v}")

def _fmt_src(v: int, n: int = 1, cdna: bool = False) -> str:
  from extra.assembly.amd.dsl import SPECIAL_PAIRS_CDNA
  if n == 1: return decode_src(v, cdna)
  if v >= 256: return _vreg(v - 256, n)
  if v <= 101: return _sreg(v, n)  # s0-s101 can be pairs, but 102+ are special on CDNA
  pairs = SPECIAL_PAIRS_CDNA if cdna else SPECIAL_PAIRS
  if n == 2 and v in pairs: return pairs[v]
  if v <= 105: return _sreg(v, n)  # s102-s105 regular pairs for RDNA
  if t := _ttmp(v, n): return t
  return decode_src(v, cdna)

def _fmt_v16(v: int, base: int = 256, hi_thresh: int = 384) -> str:
  return f"v{(v - base) & 0x7f}.{'h' if v >= hi_thresh else 'l'}"

def waitcnt(vmcnt: int = 0x3f, expcnt: int = 0x7, lgkmcnt: int = 0x3f) -> int:
  return (expcnt & 0x7) | ((lgkmcnt & 0x3f) << 4) | ((vmcnt & 0x3f) << 10)

def _has(op: str, *subs) -> bool: return any(s in op for s in subs)
def _omod(v: int) -> str: return {1: " mul:2", 2: " mul:4", 3: " div:2"}.get(v, "")
def _src16(inst, v: int) -> str: return _fmt_v16(v) if v >= 256 else inst.lit(v)  # format 16-bit src: vgpr.h/l or literal
def _mods(*pairs) -> str: return " ".join(m for c, m in pairs if c)
def _fmt_bits(label: str, val: int, count: int) -> str: return f"{label}:[{','.join(str((val >> i) & 1) for i in range(count))}]"

def _vop3_src(inst, v: int, neg: int, abs_: int, hi: int, n: int, f16: bool) -> str:
  """Format VOP3 source operand with modifiers. VOP3 uses op_sel for hi/lo, not .h/.l suffix."""
  if v == 255: s = inst.lit(v)  # literal constant takes priority
  elif n > 1: s = _fmt_src(v, n)
  else: s = inst.lit(v)  # VOP3: no .h/.l suffix, use op_sel instead
  if abs_: s = f"|{s}|"
  return f"-{s}" if neg else s

def _opsel_str(opsel: int, n: int, need: bool, is16_d: bool) -> str:
  """Format op_sel modifier string."""
  if not need: return ""
  # For VOP1 (n=1): op_sel:[src0_hi, dst_hi], for VOP2 (n=2): op_sel:[src0_hi, src1_hi, dst_hi], for VOP3 (n=3): op_sel:[src0_hi, src1_hi, src2_hi, dst_hi]
  dst_hi = (opsel >> 3) & 1
  if n == 1: return f" op_sel:[{opsel & 1},{dst_hi}]"
  if n == 2: return f" op_sel:[{opsel & 1},{(opsel >> 1) & 1},{dst_hi}]"
  return f" op_sel:[{opsel & 1},{(opsel >> 1) & 1},{(opsel >> 2) & 1},{dst_hi}]"

# ═══════════════════════════════════════════════════════════════════════════════
# DISASSEMBLER
# ═══════════════════════════════════════════════════════════════════════════════

def _disasm_vop1(inst: VOP1) -> str:
  name, cdna = inst.op_name.lower() or f'vop1_op_{inst.op}', _is_cdna(inst)
  suf = "" if cdna else "_e32"
  if name in ('v_nop', 'v_pipeflush', 'v_clrexcp'): return name  # no operands
  if 'readfirstlane' in name:
    src = f"v{inst.src0 - 256}" if inst.src0 >= 256 else decode_src(inst.src0, cdna)
    return f"{name} {_fmt_sdst(inst.vdst, 1, cdna)}, {src}"
  # 16-bit dst: uses .h/.l suffix for RDNA (CDNA uses plain vN)
  parts = name.split('_')
  is_16d = not cdna and (any(p in ('f16','i16','u16','b16') for p in parts[-2:-1]) or (len(parts) >= 2 and parts[-1] in ('f16','i16','u16','b16') and 'cvt' not in name))
  dst = _vreg(inst.vdst, inst.dst_regs()) if inst.dst_regs() > 1 else _fmt_v16(inst.vdst, 0, 128) if is_16d else f"v{inst.vdst}"
  src = inst.lit(inst.src0) if inst.src0 == 255 else _fmt_src(inst.src0, inst.src_regs(0), cdna) if inst.src_regs(0) > 1 else _src16(inst, inst.src0) if not cdna and inst.is_src_16(0) and 'sat_pk' not in name else inst.lit(inst.src0)
  return f"{name}{suf} {dst}, {src}"

_VOP2_CARRY_OUT = {'v_add_co_u32', 'v_sub_co_u32', 'v_subrev_co_u32'}  # carry out only
_VOP2_CARRY_INOUT = {'v_addc_co_u32', 'v_subb_co_u32', 'v_subbrev_co_u32'}  # carry in and out
def _disasm_vop2(inst: VOP2) -> str:
  name, cdna = inst.op_name.lower(), _is_cdna(inst)
  if cdna: name = _CDNA_DISASM_ALIASES.get(name, name)  # apply CDNA aliases
  suf = "" if cdna or (not cdna and inst.op == VOP2Op.V_DOT2ACC_F32_F16) else "_e32"
  lit = getattr(inst, '_literal', None)
  is16 = not cdna and inst.is_16bit()
  # fmaak/madak: dst = src0 * vsrc1 + K, fmamk/madmk: dst = src0 * K + vsrc1
  if 'fmaak' in name or 'madak' in name or (not cdna and inst.op in (VOP2Op.V_FMAAK_F32, VOP2Op.V_FMAAK_F16)):
    if is16: return f"{name}{suf} {_fmt_v16(inst.vdst, 0, 128)}, {_src16(inst, inst.src0)}, {_fmt_v16(inst.vsrc1, 0, 128)}, 0x{lit:x}"
    return f"{name}{suf} v{inst.vdst}, {inst.lit(inst.src0)}, v{inst.vsrc1}, 0x{lit:x}"
  if 'fmamk' in name or 'madmk' in name or (not cdna and inst.op in (VOP2Op.V_FMAMK_F32, VOP2Op.V_FMAMK_F16)):
    if is16: return f"{name}{suf} {_fmt_v16(inst.vdst, 0, 128)}, {_src16(inst, inst.src0)}, 0x{lit:x}, {_fmt_v16(inst.vsrc1, 0, 128)}"
    return f"{name}{suf} v{inst.vdst}, {inst.lit(inst.src0)}, 0x{lit:x}, v{inst.vsrc1}"
  if is16: return f"{name}{suf} {_fmt_v16(inst.vdst, 0, 128)}, {_src16(inst, inst.src0)}, {_fmt_v16(inst.vsrc1, 0, 128)}"
  # Handle 64-bit ops (f64) for RDNA
  if not cdna and 'f64' in name:
    dst = _vreg(inst.vdst, 2)
    src0 = _vreg(inst.src0 - 256, 2) if inst.src0 >= 256 else _fmt_src(inst.src0, 2)
    src1 = _vreg(inst.vsrc1, 2)
    return f"{name}{suf} {dst}, {src0}, {src1}"
  vcc = "vcc" if cdna else "vcc_lo"
  # CDNA carry ops output vcc after vdst
  if cdna and name in _VOP2_CARRY_OUT: return f"{name}{suf} v{inst.vdst}, {vcc}, {inst.lit(inst.src0)}, v{inst.vsrc1}"
  if cdna and name in _VOP2_CARRY_INOUT: return f"{name}{suf} v{inst.vdst}, {vcc}, {inst.lit(inst.src0)}, v{inst.vsrc1}, {vcc}"
  return f"{name}{suf} v{inst.vdst}, {inst.lit(inst.src0)}, v{inst.vsrc1}" + (f", {vcc}" if name == 'v_cndmask_b32' else "")

def _disasm_vopc(inst: VOPC) -> str:
  name, cdna = inst.op_name.lower(), _is_cdna(inst)
  if cdna:
    s0 = inst.lit(inst.src0) if inst.src0 == 255 else _fmt_src(inst.src0, inst.src_regs(0), cdna)
    s1 = _vreg(inst.vsrc1, inst.src_regs(1)) if inst.src_regs(1) > 1 else f"v{inst.vsrc1}"
    return f"{name} vcc, {s0}, {s1}"  # CDNA VOPC always outputs vcc
  # RDNA: v_cmpx_* writes to exec (no vcc), v_cmp_* writes to vcc_lo
  has_vcc = 'cmpx' not in name
  s0 = inst.lit(inst.src0) if inst.src0 == 255 else _fmt_src(inst.src0, inst.src_regs(0)) if inst.src_regs(0) > 1 else _src16(inst, inst.src0) if inst.is_16bit() else inst.lit(inst.src0)
  s1 = _vreg(inst.vsrc1, inst.src_regs(1)) if inst.src_regs(1) > 1 else _fmt_v16(inst.vsrc1, 0, 128) if inst.is_16bit() else f"v{inst.vsrc1}"
  return f"{name}_e32 vcc_lo, {s0}, {s1}" if has_vcc else f"{name}_e32 {s0}, {s1}"

NO_ARG_SOPP = {SOPPOp.S_BARRIER, SOPPOp.S_WAKEUP, SOPPOp.S_ICACHE_INV,
               SOPPOp.S_WAIT_IDLE, SOPPOp.S_ENDPGM_SAVED, SOPPOp.S_CODE_END, SOPPOp.S_ENDPGM_ORDERED_PS_DONE, SOPPOp.S_TTRACEDATA}
# CDNA uses name-based matching since opcode values differ from RDNA
_CDNA_NO_ARG_SOPP = {'s_endpgm', 's_barrier', 's_wakeup', 's_icache_inv', 's_ttracedata', 's_nop', 's_sethalt', 's_sleep',
                     's_setprio', 's_trap', 's_incperflevel', 's_decperflevel', 's_sendmsg', 's_sendmsghalt'}

def _disasm_sopp(inst: SOPP) -> str:
  name, cdna = inst.op_name.lower(), _is_cdna(inst)
  if cdna:
    # CDNA: use name-based matching
    if name == 's_endpgm': return name if inst.simm16 == 0 else f"{name} {inst.simm16}"
    if name in ('s_barrier', 's_wakeup', 's_icache_inv', 's_ttracedata'): return name
    if name == 's_waitcnt':
      vm, lgkm, exp = inst.simm16 & 0xf, (inst.simm16 >> 8) & 0x3f, (inst.simm16 >> 4) & 0x7
      p = [f"vmcnt({vm})" if vm != 0xf else "", f"expcnt({exp})" if exp != 7 else "", f"lgkmcnt({lgkm})" if lgkm != 0x3f else ""]
      return f"s_waitcnt {' '.join(x for x in p if x) or '0'}"
    if name.startswith(('s_cbranch', 's_branch')): return f"{name} {inst.simm16}"
    return f"{name} 0x{inst.simm16:x}" if inst.simm16 else name
  # RDNA
  if inst.op in NO_ARG_SOPP: return name
  if inst.op == SOPPOp.S_ENDPGM: return name if inst.simm16 == 0 else f"{name} {inst.simm16}"
  if inst.op == SOPPOp.S_WAITCNT:
    vm, exp, lgkm = (inst.simm16 >> 10) & 0x3f, inst.simm16 & 0xf, (inst.simm16 >> 4) & 0x3f
    p = [f"vmcnt({vm})" if vm != 0x3f else "", f"expcnt({exp})" if exp != 7 else "", f"lgkmcnt({lgkm})" if lgkm != 0x3f else ""]
    return f"s_waitcnt {' '.join(x for x in p if x) or '0'}"
  if inst.op == SOPPOp.S_DELAY_ALU:
    deps, skips = ['VALU_DEP_1','VALU_DEP_2','VALU_DEP_3','VALU_DEP_4','TRANS32_DEP_1','TRANS32_DEP_2','TRANS32_DEP_3','FMA_ACCUM_CYCLE_1','SALU_CYCLE_1','SALU_CYCLE_2','SALU_CYCLE_3'], ['SAME','NEXT','SKIP_1','SKIP_2','SKIP_3','SKIP_4']
    id0, skip, id1 = inst.simm16 & 0xf, (inst.simm16 >> 4) & 0x7, (inst.simm16 >> 7) & 0xf
    dep = lambda v: deps[v-1] if 0 < v <= len(deps) else str(v)
    p = [f"instid0({dep(id0)})" if id0 else "", f"instskip({skips[skip]})" if skip else "", f"instid1({dep(id1)})" if id1 else ""]
    return f"s_delay_alu {' | '.join(x for x in p if x) or '0'}"
  return f"{name} {inst.simm16}" if name.startswith(('s_cbranch', 's_branch')) else f"{name} 0x{inst.simm16:x}"

def _disasm_smem(inst: SMEM) -> str:
  name, cdna = inst.op_name.lower(), _is_cdna(inst)
  if inst.op in (SMEMOp.S_GL1_INV, SMEMOp.S_DCACHE_INV): return name
  # GFX9 SMEM: soe and imm bits determine offset interpretation
  # soe=1, imm=1: soffset is SGPR, offset is immediate (both used)
  # soe=0, imm=1: offset is immediate
  # soe=0, imm=0: offset field is SGPR encoding (0-255)
  soe, imm = getattr(inst, 'soe', 0), getattr(inst, 'imm', 1)
  # RDNA4 uses ioffset, RDNA3 uses offset
  off = getattr(inst, 'ioffset', None) or getattr(inst, 'offset', 0)
  if cdna:
    if soe and imm:
      off_s = f"{decode_src(inst.soffset, cdna)} offset:0x{inst.offset:x}"  # SGPR + immediate
    elif imm:
      off_s = f"0x{inst.offset:x}"  # Immediate offset only
    elif inst.offset < 256:
      off_s = decode_src(inst.offset, cdna)  # SGPR encoding in offset field
    else:
      off_s = decode_src(inst.soffset, cdna)
  elif off and inst.soffset != 124:
    off_s = f"{decode_src(inst.soffset, cdna)} offset:0x{off:x}"
  elif off:
    off_s = f"0x{off:x}"
  else:
    off_s = decode_src(inst.soffset, cdna)
  # s_buffer_* instructions use 4 SGPRs for sbase (buffer descriptor)
  is_buffer = 'buffer' in name or 's_atc_probe_buffer' == name
  sbase_idx, sbase_count = inst.sbase * 2, 4 if is_buffer else 2
  sbase_str = _fmt_src(sbase_idx, sbase_count, cdna) if sbase_count == 2 else _sreg(sbase_idx, sbase_count) if sbase_idx <= 105 else _reg("ttmp", sbase_idx - 108, sbase_count)
  if name in ('s_atc_probe', 's_atc_probe_buffer'): return f"{name} {inst.sdata}, {sbase_str}, {off_s}"
  return f"{name} {_fmt_sdst(inst.sdata, inst.dst_regs(), cdna)}, {sbase_str}, {off_s}" + _mods((inst.glc, " glc"), (getattr(inst, 'dlc', 0), " dlc"))

def _disasm_flat(inst: FLAT) -> str:
  name, cdna = inst.op_name.lower(), _is_cdna(inst)
  acc = getattr(inst, 'acc', 0)  # GFX90a accumulator register flag
  reg_fn = _areg if acc else _vreg  # use a[n] for acc=1, v[n] for acc=0
  seg = ['flat', 'scratch', 'global'][inst.seg] if inst.seg < 3 else 'flat'
  instr = f"{seg}_{name.split('_', 1)[1] if '_' in name else name}"
  off_val = inst.offset if seg == 'flat' else (inst.offset if inst.offset < 4096 else inst.offset - 8192)
  w = inst.dst_regs() * (2 if '_x2' in name else 1) * (2 if 'cmpswap' in name else 1)
  off_s = f" offset:{off_val}" if off_val else ""  # Omit offset:0
  if cdna: mods = f"{off_s}{' glc' if inst.sc0 else ''}{' slc' if inst.nt else ''}"  # GFX9: sc0->glc, nt->slc
  else: mods = f"{off_s}{' glc' if inst.glc else ''}{' slc' if inst.slc else ''}{' dlc' if inst.dlc else ''}"
  # saddr
  if seg == 'flat' or inst.saddr == 0x7F: saddr_s = ""
  elif inst.saddr == 124: saddr_s = ", off"
  elif seg == 'scratch': saddr_s = f", {decode_src(inst.saddr, cdna)}"
  elif inst.saddr in (SPECIAL_PAIRS_CDNA if cdna else SPECIAL_PAIRS): saddr_s = f", {(SPECIAL_PAIRS_CDNA if cdna else SPECIAL_PAIRS)[inst.saddr]}"
  elif t := _ttmp(inst.saddr, 2): saddr_s = f", {t}"
  else: saddr_s = f", {_sreg(inst.saddr, 2) if inst.saddr < 106 else decode_src(inst.saddr, cdna)}"
  # addtid: no addr
  if 'addtid' in name: return f"{instr} {'a' if acc else 'v'}{inst.data if 'store' in name else inst.vdst}{saddr_s}{mods}"
  # addr width: CDNA flat always uses 2 VGPRs (64-bit), scratch uses 1, RDNA uses 2 only when no saddr
  if cdna:
    addr_w = 1 if seg == 'scratch' else 2  # CDNA: flat/global always 64-bit addr
  else:
    addr_w = 1 if seg == 'scratch' or (inst.saddr not in (0x7F, 124)) else 2
  addr_s = "off" if not inst.sve and seg == 'scratch' else _vreg(inst.addr, addr_w)
  data_s, vdst_s = reg_fn(inst.data, w), reg_fn(inst.vdst, w // 2 if 'cmpswap' in name else w)
  glc_or_sc0 = inst.sc0 if cdna else inst.glc
  if 'atomic' in name:
    return f"{instr} {vdst_s}, {addr_s}, {data_s}{saddr_s if seg != 'flat' else ''}{mods}" if glc_or_sc0 else f"{instr} {addr_s}, {data_s}{saddr_s if seg != 'flat' else ''}{mods}"
  if 'store' in name: return f"{instr} {addr_s}, {data_s}{saddr_s}{mods}"
  return f"{instr} {reg_fn(inst.vdst, w)}, {addr_s}{saddr_s}{mods}"

def _disasm_ds(inst: DS) -> str:
  op, name = inst.op, inst.op_name.lower()
  acc = getattr(inst, 'acc', 0)  # GFX90a accumulator register flag
  reg_fn = _areg if acc else _vreg  # use a[n] for acc=1, v[n] for acc=0
  rp = 'a' if acc else 'v'  # register prefix for single regs
  gds = " gds" if inst.gds else ""
  off = f" offset:{inst.offset0 | (inst.offset1 << 8)}" if inst.offset0 or inst.offset1 else ""
  off2 = (" offset0:" + str(inst.offset0) if inst.offset0 else "") + (" offset1:" + str(inst.offset1) if inst.offset1 else "")
  w = inst.dst_regs()
  d0, d1, dst, addr = reg_fn(inst.data0, w), reg_fn(inst.data1, w), reg_fn(inst.vdst, w), f"v{inst.addr}"

  if op == DSOp.DS_NOP: return name
  if op == DSOp.DS_BVH_STACK_RTN_B32: return f"{name} v{inst.vdst}, {addr}, v{inst.data0}, {_vreg(inst.data1, 4)}{off}{gds}"
  if 'gws_sema' in name and op != DSOp.DS_GWS_SEMA_BR: return f"{name}{off}{gds}"
  if 'gws_' in name: return f"{name} {addr}{off}{gds}"
  if op in (DSOp.DS_CONSUME, DSOp.DS_APPEND): return f"{name} {rp}{inst.vdst}{off}{gds}"
  if 'gs_reg' in name: return f"{name} {reg_fn(inst.vdst, 2)}, {rp}{inst.data0}{off}{gds}"
  if '2addr' in name:
    if 'load' in name: return f"{name} {reg_fn(inst.vdst, w*2)}, {addr}{off2}{gds}"
    if 'store' in name and 'xchg' not in name: return f"{name} {addr}, {d0}, {d1}{off2}{gds}"
    return f"{name} {reg_fn(inst.vdst, w*2)}, {addr}, {d0}, {d1}{off2}{gds}"
  if 'write2' in name: return f"{name} {addr}, {d0}, {d1}{off2}{gds}"
  if 'read2' in name: return f"{name} {reg_fn(inst.vdst, w*2)}, {addr}{off2}{gds}"
  if 'load' in name: return f"{name} {rp}{inst.vdst}{off}{gds}" if 'addtid' in name else f"{name} {dst}, {addr}{off}{gds}"
  if 'store' in name and not _has(name, 'cmp', 'xchg'):
    return f"{name} {rp}{inst.data0}{off}{gds}" if 'addtid' in name else f"{name} {addr}, {d0}{off}{gds}"
  if 'swizzle' in name or op == DSOp.DS_ORDERED_COUNT: return f"{name} {rp}{inst.vdst}, {addr}{off}{gds}"
  if 'permute' in name: return f"{name} {rp}{inst.vdst}, {addr}, {rp}{inst.data0}{off}{gds}"
  if 'condxchg' in name: return f"{name} {reg_fn(inst.vdst, 2)}, {addr}, {reg_fn(inst.data0, 2)}{off}{gds}"
  if _has(name, 'cmpstore', 'mskor', 'wrap'):
    return f"{name} {dst}, {addr}, {d0}, {d1}{off}{gds}" if '_rtn' in name else f"{name} {addr}, {d0}, {d1}{off}{gds}"
  return f"{name} {dst}, {addr}, {d0}{off}{gds}" if '_rtn' in name else f"{name} {addr}, {d0}{off}{gds}"

def _disasm_vop3(inst: VOP3) -> str:
  op, name = inst.op, inst.op_name.lower()

  # VOP3SD (shared encoding)
  if isinstance(op, VOP3SDOp):
    sdst = (inst.clmp << 7) | (inst.opsel << 3) | inst.abs
    def src(v, neg, n):
      s = inst.lit(v) if v == 255 else (_fmt_src(v, n) if n > 1 else inst.lit(v))
      return f"neg({s})" if neg and v == 255 else (f"-{s}" if neg else s)
    s0, s1, s2 = src(inst.src0, inst.neg & 1, inst.src_regs(0)), src(inst.src1, inst.neg & 2, inst.src_regs(1)), src(inst.src2, inst.neg & 4, inst.src_regs(2))
    dst = _vreg(inst.vdst, inst.dst_regs()) if inst.dst_regs() > 1 else f"v{inst.vdst}"
    srcs = f"{s0}, {s1}, {s2}" if inst.num_srcs() == 3 else f"{s0}, {s1}"
    return f"{name} {dst}, {_fmt_sdst(sdst, 1)}, {srcs}" + _omod(inst.omod)

  # Detect 16-bit operand sizes (for .h/.l suffix handling)
  is16_d = is16_s = is16_s2 = False
  if 'cvt_pk' in name: is16_s = name.endswith('16')
  elif m := re.match(r'v_(?:cvt|frexp_exp)_([a-z0-9_]+)_([a-z0-9]+)', name):
    is16_d, is16_s = _has(m.group(1), 'f16','i16','u16','b16'), _has(m.group(2), 'f16','i16','u16','b16')
    is16_s2 = is16_s
  elif re.match(r'v_mad_[iu]32_[iu]16', name): is16_s = True
  elif 'pack_b32' in name: is16_s = is16_s2 = True
  elif 'sat_pk' in name: is16_d = True  # v_sat_pk_* writes to 16-bit dest but takes 32-bit src
  else: is16_d = is16_s = is16_s2 = inst.is_16bit()

  s0 = _vop3_src(inst, inst.src0, inst.neg&1, inst.abs&1, inst.opsel&1, inst.src_regs(0), is16_s)
  s1 = _vop3_src(inst, inst.src1, inst.neg&2, inst.abs&2, inst.opsel&2, inst.src_regs(1), is16_s)
  s2 = _vop3_src(inst, inst.src2, inst.neg&4, inst.abs&4, inst.opsel&4, inst.src_regs(2), is16_s2)

  # Destination - VOP3 uses op_sel for hi/lo, not .h/.l suffix
  dn = inst.dst_regs()
  if op == VOP3Op.V_READLANE_B32: dst = _fmt_sdst(inst.vdst, 1)
  elif dn > 1: dst = _vreg(inst.vdst, dn)
  else: dst = f"v{inst.vdst}"  # VOP3: no .h/.l suffix, use op_sel instead

  cl, om = " clamp" if inst.clmp else "", _omod(inst.omod)
  nonvgpr_opsel = (inst.src0 < 256 and (inst.opsel & 1)) or (inst.src1 < 256 and (inst.opsel & 2)) or (inst.src2 < 256 and (inst.opsel & 4))
  # Only output op_sel when there's actual selection to do (non-zero opsel or 16-bit sources)
  # Conversions like v_cvt_f16_f32 (32-bit src, 16-bit dst) don't support op_sel
  need_opsel = nonvgpr_opsel or (inst.opsel and not is16_s)

  if inst.op < 256:  # VOPC
    return f"{name}_e64 {s0}, {s1}{cl}" if name.startswith('v_cmpx') else f"{name}_e64 {_fmt_sdst(inst.vdst, 1)}, {s0}, {s1}{cl}"
  if inst.op < 384:  # VOP2
    n = inst.num_srcs()
    os = _opsel_str(inst.opsel, n, need_opsel, is16_d)
    return f"{name}_e64 {dst}, {s0}, {s1}, {s2}{os}{cl}{om}" if n == 3 else f"{name}_e64 {dst}, {s0}, {s1}{os}{cl}{om}"
  if inst.op < 512:  # VOP1
    return f"{name}_e64" if op in (VOP3Op.V_NOP, VOP3Op.V_PIPEFLUSH) else f"{name}_e64 {dst}, {s0}{_opsel_str(inst.opsel, 1, need_opsel, is16_d)}{cl}{om}"
  # Native VOP3
  n = inst.num_srcs()
  os = _opsel_str(inst.opsel, n, need_opsel, is16_d)
  return f"{name} {dst}, {s0}, {s1}, {s2}{os}{cl}{om}" if n == 3 else f"{name} {dst}, {s0}, {s1}{os}{cl}{om}"

def _disasm_vop3sd(inst: VOP3SD) -> str:
  name = inst.op_name.lower()
  def src(v, neg, n):
    s = inst.lit(v) if v == 255 else (_fmt_src(v, n) if n > 1 else inst.lit(v))
    return f"neg({s})" if neg and v == 255 else (f"-{s}" if neg else s)
  s0, s1, s2 = src(inst.src0, inst.neg & 1, inst.src_regs(0)), src(inst.src1, inst.neg & 2, inst.src_regs(1)), src(inst.src2, inst.neg & 4, inst.src_regs(2))
  dst = _vreg(inst.vdst, inst.dst_regs()) if inst.dst_regs() > 1 else f"v{inst.vdst}"
  srcs = f"{s0}, {s1}, {s2}" if inst.num_srcs() == 3 else f"{s0}, {s1}"
  suffix = "_e64" if name.startswith('v_') and 'co_' in name else ""
  return f"{name}{suffix} {dst}, {_fmt_sdst(inst.sdst, 1)}, {srcs}{' clamp' if inst.clmp else ''}{_omod(inst.omod)}"

def _disasm_vopd(inst: VOPD) -> str:
  lit = inst._literal or inst.literal
  vdst_y, nx, ny = (inst.vdsty << 1) | ((inst.vdstx & 1) ^ 1), VOPDOp(inst.opx).name.lower(), VOPDOp(inst.opy).name.lower()
  def half(n, vd, s0, vs1):
    if 'mov' in n: return f"{n} v{vd}, {inst.lit(s0)}"
    # fmamk: dst = src0 * K + vsrc1, fmaak: dst = src0 * vsrc1 + K
    if 'fmamk' in n and lit: return f"{n} v{vd}, {inst.lit(s0)}, 0x{lit:x}, v{vs1}"
    if 'fmaak' in n and lit: return f"{n} v{vd}, {inst.lit(s0)}, v{vs1}, 0x{lit:x}"
    return f"{n} v{vd}, {inst.lit(s0)}, v{vs1}"
  return f"{half(nx, inst.vdstx, inst.srcx0, inst.vsrcx1)} :: {half(ny, vdst_y, inst.srcy0, inst.vsrcy1)}"

def _disasm_vop3p(inst: VOP3P) -> str:
  name = inst.op_name.lower()
  is_wmma, n, is_fma_mix = 'wmma' in name, inst.num_srcs(), 'fma_mix' in name
  def get_src(v, sc): return inst.lit(v) if v == 255 else _fmt_src(v, sc)
  if is_wmma:
    # src count: iu4=2, iu8=4, f16/bf16=4 (16 values per 4 VGPRs), f32=8
    sc = 2 if 'iu4' in name else 4 if 'iu8' in name or 'f16' in name or 'bf16' in name else 8
    src0, src1, src2, dst = get_src(inst.src0, sc), get_src(inst.src1, sc), get_src(inst.src2, 8), _vreg(inst.vdst, 8)
  else: src0, src1, src2, dst = get_src(inst.src0, 1), get_src(inst.src1, 1), get_src(inst.src2, 1), f"v{inst.vdst}"
  opsel_hi = inst.opsel_hi | (inst.opsel_hi2 << 2)
  if is_fma_mix:
    def m(s, neg, abs_): return f"-{f'|{s}|' if abs_ else s}" if neg else (f"|{s}|" if abs_ else s)
    src0, src1, src2 = m(src0, inst.neg & 1, inst.neg_hi & 1), m(src1, inst.neg & 2, inst.neg_hi & 2), m(src2, inst.neg & 4, inst.neg_hi & 4)
    mods = ([_fmt_bits("op_sel", inst.opsel, n)] if inst.opsel else []) + ([_fmt_bits("op_sel_hi", opsel_hi, n)] if opsel_hi else []) + (["clamp"] if inst.clmp else [])
  else:
    mods = ([_fmt_bits("op_sel", inst.opsel, n)] if inst.opsel else []) + ([_fmt_bits("op_sel_hi", opsel_hi, n)] if opsel_hi != (7 if n == 3 else 3) else []) + \
           ([_fmt_bits("neg_lo", inst.neg, n)] if inst.neg else []) + ([_fmt_bits("neg_hi", inst.neg_hi, n)] if inst.neg_hi else []) + (["clamp"] if inst.clmp else [])
  return f"{name} {dst}, {src0}, {src1}, {src2}{' ' + ' '.join(mods) if mods else ''}" if n == 3 else f"{name} {dst}, {src0}, {src1}{' ' + ' '.join(mods) if mods else ''}"

def _disasm_buf(inst: MUBUF | MTBUF) -> str:
  name, cdna = inst.op_name.lower(), _is_cdna(inst)
  acc = getattr(inst, 'acc', 0)  # GFX90a accumulator register flag
  reg_fn = _areg if acc else _vreg  # use a[n] for acc=1, v[n] for acc=0
  if cdna and name in ('buffer_wbl2', 'buffer_inv'): return name
  if not cdna and inst.op in (MUBUFOp.BUFFER_GL0_INV, MUBUFOp.BUFFER_GL1_INV): return name
  w = (2 if _has(name, 'xyz', 'xyzw') else 1) if 'd16' in name else \
      ((2 if _has(name, 'b64', 'u64', 'i64') else 1) * (2 if 'cmpswap' in name else 1)) if 'atomic' in name else \
      {'b32':1,'b64':2,'b96':3,'b128':4,'b16':1,'x':1,'xy':2,'xyz':3,'xyzw':4}.get(name.split('_')[-1], 1)
  if hasattr(inst, 'tfe') and inst.tfe: w += 1
  vaddr = _vreg(inst.vaddr, 2) if inst.offen and inst.idxen else f"v{inst.vaddr}" if inst.offen or inst.idxen else "off"
  srsrc = _sreg_or_ttmp(inst.srsrc*4, 4)
  is_mtbuf = isinstance(inst, MTBUF) or isinstance(inst, C_MTBUF)
  if is_mtbuf:
    dfmt, nfmt = inst.format & 0xf, (inst.format >> 4) & 0x7
    if acc:  # GFX90a accumulator style: show dfmt/nfmt as numbers
      fmt_s = f"  dfmt:{dfmt}, nfmt:{nfmt},"  # double space before dfmt per LLVM format
    elif not cdna:  # RDNA style: show combined format number
      fmt_s = f" format:{inst.format}" if inst.format else ""
    else:  # CDNA: show format:[BUF_DATA_FORMAT_X] or format:[BUF_NUM_FORMAT_X]
      dfmt_names = ['INVALID', '8', '16', '8_8', '32', '16_16', '10_11_11', '11_11_10', '10_10_10_2', '2_10_10_10', '8_8_8_8', '32_32', '16_16_16_16', '32_32_32', '32_32_32_32', 'RESERVED_15']
      nfmt_names = ['UNORM', 'SNORM', 'USCALED', 'SSCALED', 'UINT', 'SINT', 'RESERVED_6', 'FLOAT']
      if dfmt == 1 and nfmt == 0: fmt_s = ""  # default, no format shown
      elif nfmt == 0: fmt_s = f" format:[BUF_DATA_FORMAT_{dfmt_names[dfmt]}]"  # only dfmt differs
      elif dfmt == 1: fmt_s = f" format:[BUF_NUM_FORMAT_{nfmt_names[nfmt]}]"  # only nfmt differs
      else: fmt_s = f" format:[BUF_DATA_FORMAT_{dfmt_names[dfmt]},BUF_NUM_FORMAT_{nfmt_names[nfmt]}]"  # both differ
  else:
    fmt_s = ""
  if cdna: mods = [m for c, m in [(inst.idxen,"idxen"),(inst.offen,"offen"),(inst.offset,f"offset:{inst.offset}"),(inst.sc0,"glc"),(inst.nt,"slc"),(inst.sc1,"sc1")] if c]
  else: mods = [m for c, m in [(inst.idxen,"idxen"),(inst.offen,"offen"),(inst.offset,f"offset:{inst.offset}"),(inst.glc,"glc"),(inst.dlc,"dlc"),(inst.slc,"slc"),(inst.tfe,"tfe")] if c]
  soffset_s = decode_src(inst.soffset, cdna)
  if cdna and not acc and is_mtbuf: return f"{name} {reg_fn(inst.vdata, w)}, {vaddr}, {srsrc}, {soffset_s}{fmt_s}{' ' + ' '.join(mods) if mods else ''}"
  return f"{name} {reg_fn(inst.vdata, w)}, {vaddr}, {srsrc},{fmt_s} {soffset_s}{' ' + ' '.join(mods) if mods else ''}"

def _mimg_vaddr_width(name: str, dim: int, a16: bool) -> int:
  """Calculate vaddr register count for MIMG sample/gather operations."""
  #                    1d,2d,3d,cube,1d_arr,2d_arr,2d_msaa,2d_msaa_arr
  base =              [1, 2, 3, 3,   2,     3,     3,      4][dim]  # address coords
  grad =              [1, 2, 3, 2,   1,     2,     2,      2][dim]  # gradient coords (for derivatives)
  if 'get_resinfo' in name: return 1  # only mip level
  packed, unpacked = 0, 0
  if '_mip' in name: packed += 1
  elif 'sample' in name or 'gather' in name:
    if '_o' in name: unpacked += 1                                              # offset
    if re.search(r'_c(_|$)', name): unpacked += 1                               # compare (not _cl)
    if '_d' in name: unpacked += (grad + 1) & ~1 if '_g16' in name else grad*2  # derivatives
    if '_b' in name: unpacked += 1                                              # bias
    if '_l' in name and '_cl' not in name and '_lz' not in name: packed += 1    # LOD
    if '_cl' in name: packed += 1                                               # clamp
  return (base + packed + 1) // 2 + unpacked if a16 else base + packed + unpacked

def _disasm_mimg(inst: MIMG) -> str:
  name = inst.op_name.lower()
  srsrc_base = inst.srsrc * 4
  srsrc_str = _sreg_or_ttmp(srsrc_base, 8)
  # BVH intersect ray: special case with 4 SGPR srsrc
  if 'bvh' in name:
    vaddr = (9 if '64' in name else 8) if inst.a16 else (12 if '64' in name else 11)
    return f"{name} {_vreg(inst.vdata, 4)}, {_vreg(inst.vaddr, vaddr)}, {_sreg_or_ttmp(srsrc_base, 4)}{' a16' if inst.a16 else ''}"
  # vdata width from dmask (gather4/msaa_load always 4), d16 packs, tfe adds 1
  vdata = 4 if 'gather4' in name or 'msaa_load' in name else (bin(inst.dmask).count('1') or 1)
  if inst.d16: vdata = (vdata + 1) // 2
  if inst.tfe: vdata += 1
  # vaddr width
  dim_names = ['1d', '2d', '3d', 'cube', '1d_array', '2d_array', '2d_msaa', '2d_msaa_array']
  dim = dim_names[inst.dim] if inst.dim < len(dim_names) else f"dim_{inst.dim}"
  vaddr = _mimg_vaddr_width(name, inst.dim, inst.a16)
  vaddr_str = f"v{inst.vaddr}" if vaddr == 1 else _vreg(inst.vaddr, vaddr)
  # modifiers
  mods = [f"dmask:0x{inst.dmask:x}"] if inst.dmask and (inst.dmask != 15 or 'atomic' in name) else []
  mods.append(f"dim:SQ_RSRC_IMG_{dim.upper()}")
  for flag, mod in [(inst.unrm,"unorm"),(inst.glc,"glc"),(inst.slc,"slc"),(inst.dlc,"dlc"),(inst.r128,"r128"),
                    (inst.a16,"a16"),(inst.tfe,"tfe"),(inst.lwe,"lwe"),(inst.d16,"d16")]:
    if flag: mods.append(mod)
  # ssamp for sample/gather/get_lod
  ssamp_str = ""
  if 'sample' in name or 'gather' in name or 'get_lod' in name:
    ssamp_str = ", " + _sreg_or_ttmp(inst.ssamp * 4, 4)
  return f"{name} {_vreg(inst.vdata, vdata)}, {vaddr_str}, {srsrc_str}{ssamp_str} {' '.join(mods)}"

def _disasm_sop1(inst: SOP1) -> str:
  op, name, cdna = inst.op, inst.op_name.lower(), _is_cdna(inst)
  src = inst.lit(inst.ssrc0) if inst.ssrc0 == 255 else _fmt_src(inst.ssrc0, inst.src_regs(0), cdna)
  if not cdna:
    if op == SOP1Op.S_GETPC_B64: return f"{name} {_fmt_sdst(inst.sdst, 2)}"
    if op in (SOP1Op.S_SETPC_B64, SOP1Op.S_RFE_B64): return f"{name} {src}"
    if op == SOP1Op.S_SWAPPC_B64: return f"{name} {_fmt_sdst(inst.sdst, 2)}, {src}"
    if op in (SOP1Op.S_SENDMSG_RTN_B32, SOP1Op.S_SENDMSG_RTN_B64): return f"{name} {_fmt_sdst(inst.sdst, inst.dst_regs())}, sendmsg({MSG.get(inst.ssrc0, str(inst.ssrc0))})"
    # barrier_signal only takes ssrc0 (barrier index), not sdst
    if 'barrier_signal' in name: return f"{name} {inst.lit(inst.ssrc0)}"
  return f"{name} {_fmt_sdst(inst.sdst, inst.dst_regs(), cdna)}, {src}"

def _disasm_sop2(inst: SOP2) -> str:
  cdna = _is_cdna(inst)
  return f"{inst.op_name.lower()} {_fmt_sdst(inst.sdst, inst.dst_regs(), cdna)}, {inst.lit(inst.ssrc0) if inst.ssrc0 == 255 else _fmt_src(inst.ssrc0, inst.src_regs(0), cdna)}, {inst.lit(inst.ssrc1) if inst.ssrc1 == 255 else _fmt_src(inst.ssrc1, inst.src_regs(1), cdna)}"

def _disasm_sopc(inst: SOPC) -> str:
  cdna = _is_cdna(inst)
  s0 = inst.lit(inst.ssrc0) if inst.ssrc0 == 255 else _fmt_src(inst.ssrc0, inst.src_regs(0), cdna)
  s1 = inst.lit(inst.ssrc1) if inst.ssrc1 == 255 else _fmt_src(inst.ssrc1, inst.src_regs(1), cdna)
  return f"{inst.op_name.lower()} {s0}, {s1}"

def _disasm_sopk(inst: SOPK) -> str:
  op, name, cdna = inst.op, inst.op_name.lower(), _is_cdna(inst)
  # s_setreg_imm32_b32 has a 32-bit literal value
  if name == 's_setreg_imm32_b32' or (not cdna and op == SOPKOp.S_SETREG_IMM32_B32):
    hid, hoff, hsz = inst.simm16 & 0x3f, (inst.simm16 >> 6) & 0x1f, ((inst.simm16 >> 11) & 0x1f) + 1
    hs = f"0x{inst.simm16:x}" if hid in (16, 17) else f"hwreg({HWREG.get(hid, str(hid))}, {hoff}, {hsz})"
    return f"{name} {hs}, 0x{inst._literal:x}"
  if not cdna and op == SOPKOp.S_VERSION: return f"{name} 0x{inst.simm16:x}"
  if (not cdna and op in (SOPKOp.S_SETREG_B32, SOPKOp.S_GETREG_B32)) or (cdna and name in ('s_setreg_b32', 's_getreg_b32')):
    hid, hoff, hsz = inst.simm16 & 0x3f, (inst.simm16 >> 6) & 0x1f, ((inst.simm16 >> 11) & 0x1f) + 1
    hs = f"0x{inst.simm16:x}" if hid in (16, 17) else f"hwreg({HWREG.get(hid, str(hid))}, {hoff}, {hsz})"
    return f"{name} {hs}, {_fmt_sdst(inst.sdst, 1, cdna)}" if 'setreg' in name else f"{name} {_fmt_sdst(inst.sdst, 1, cdna)}, {hs}"
  if not cdna and op in (SOPKOp.S_SUBVECTOR_LOOP_BEGIN, SOPKOp.S_SUBVECTOR_LOOP_END):
    return f"{name} {_fmt_sdst(inst.sdst, 1)}, 0x{inst.simm16:x}"
  return f"{name} {_fmt_sdst(inst.sdst, inst.dst_regs(), cdna)}, 0x{inst.simm16:x}"

def _disasm_vinterp(inst: VINTERP) -> str:
  mods = _mods((inst.waitexp, f"wait_exp:{inst.waitexp}"), (inst.clmp, "clamp"))
  return f"{inst.op_name.lower()} v{inst.vdst}, {inst.lit(inst.src0, inst.neg & 1)}, {inst.lit(inst.src1, inst.neg & 2)}, {inst.lit(inst.src2, inst.neg & 4)}" + (" " + mods if mods else "")

def _disasm_vbuffer(inst: VBUFFER) -> str:
  name = inst.op_name.lower()
  w = {'b32':1,'b64':2,'b96':3,'b128':4}.get(name.split('_')[-1], 1)
  vaddr = f"v{inst.vaddr}" if inst.offen or inst.idxen else "off"
  srsrc = _sreg_or_ttmp(inst.rsrc, 4)  # rsrc is already register number, not divided by 4
  mods = [m for c, m in [(inst.offen,"offen"),(inst.idxen,"idxen")] if c]
  return f"{name} {_vreg(inst.vdata, w)}, {vaddr}, {srsrc}, {decode_src(inst.soffset)}{' ' + ' '.join(mods) if mods else ''}"

def _disasm_vds(inst: VDS) -> str:
  name = inst.op_name.lower()
  w = {'b32':1,'b64':2,'b96':3,'b128':4}.get(name.split('_')[-1], 1)
  off = f" offset:{inst.offset0 | (inst.offset1 << 8)}" if inst.offset0 or inst.offset1 else ""
  if 'load' in name:
    return f"{name} {_vreg(inst.vdst, w)}, v{inst.addr}{off}"
  else:
    return f"{name} v{inst.addr}, {_vreg(inst.data0, w)}{off}"

DISASM_HANDLERS = {VOP1: _disasm_vop1, VOP2: _disasm_vop2, VOPC: _disasm_vopc, VOP3: _disasm_vop3, VOP3SD: _disasm_vop3sd, VOPD: _disasm_vopd, VOP3P: _disasm_vop3p,
                   VINTERP: _disasm_vinterp, SOPP: _disasm_sopp, SMEM: _disasm_smem, DS: _disasm_ds, FLAT: _disasm_flat, MUBUF: _disasm_buf, MTBUF: _disasm_buf,
                   MIMG: _disasm_mimg, SOP1: _disasm_sop1, SOP2: _disasm_sop2, SOPC: _disasm_sopc, SOPK: _disasm_sopk, VBUFFER: _disasm_vbuffer, VDS: _disasm_vds}

def disasm(inst: Inst) -> str: return DISASM_HANDLERS[type(inst)](inst)

# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLER
# ═══════════════════════════════════════════════════════════════════════════════

SPEC_REGS = {'vcc_lo': RawImm(106), 'vcc_hi': RawImm(107), 'vcc': RawImm(106), 'null': RawImm(124), 'off': RawImm(124), 'm0': RawImm(125),
             'exec_lo': RawImm(126), 'exec_hi': RawImm(127), 'exec': RawImm(126), 'scc': RawImm(253), 'src_scc': RawImm(253)}
FLOATS = {str(k): k for k in FLOAT_ENC}  # Valid float literal strings: '0.5', '-0.5', '1.0', etc.
REG_MAP: dict[str, _RegFactory] = {'s': s, 'v': v, 't': ttmp, 'ttmp': ttmp}
SMEM_OPS = {'s_load_b32', 's_load_b64', 's_load_b128', 's_load_b256', 's_load_b512',
            's_buffer_load_b32', 's_buffer_load_b64', 's_buffer_load_b128', 's_buffer_load_b256', 's_buffer_load_b512',
            's_atc_probe', 's_atc_probe_buffer'}
SPEC_DSL = {'vcc_lo': 'VCC_LO', 'vcc_hi': 'VCC_HI', 'vcc': 'VCC_LO', 'null': 'NULL', 'off': 'OFF', 'm0': 'M0',
            'exec_lo': 'EXEC_LO', 'exec_hi': 'EXEC_HI', 'exec': 'EXEC_LO', 'scc': 'SCC', 'src_scc': 'SCC'}

def _op2dsl(op: str) -> str:
  op = op.strip()
  neg = op.startswith('-') and not (op[1:2].isdigit() or (len(op) > 2 and op[1] == '0' and op[2] in 'xX'))
  if neg: op = op[1:]
  abs_ = (op.startswith('|') and op.endswith('|')) or (op.startswith('abs(') and op.endswith(')'))
  if abs_: op = op[1:-1] if op.startswith('|') else op[4:-1]
  hi = ".h" if op.endswith('.h') else ".l" if op.endswith('.l') else ""
  if hi: op = op[:-2]
  lo = op.lower()
  def wrap(b): return f"{'-' if neg else ''}abs({b}){hi}" if abs_ else f"-{b}{hi}" if neg else f"{b}{hi}"
  if lo in SPEC_DSL: return wrap(SPEC_DSL[lo])
  if op in FLOATS: return wrap(op)
  rp = {'s': 's', 'v': 'v', 't': 'ttmp', 'ttmp': 'ttmp'}
  if m := re.match(r'^([svt](?:tmp)?)\[(\d+):(\d+)\]$', lo): return wrap(f"{rp[m.group(1)]}[{m.group(2)}:{m.group(3)}]")
  if m := re.match(r'^([svt](?:tmp)?)(\d+)$', lo): return wrap(f"{rp[m.group(1)]}[{m.group(2)}]")
  if re.match(r'^-?\d+$|^-?0x[0-9a-fA-F]+$', op): return f"SrcMod({op}, neg={neg}, abs_={abs_})" if neg or abs_ else op
  return wrap(op)

def _parse_ops(s: str) -> list[str]:
  ops, cur, depth, pipe = [], "", 0, False
  for c in s:
    if c in '[(': depth += 1
    elif c in '])': depth -= 1
    elif c == '|': pipe = not pipe
    if c == ',' and depth == 0 and not pipe: ops.append(cur.strip()); cur = ""
    else: cur += c
  if cur.strip(): ops.append(cur.strip())
  return ops

def _extract(text: str, pat: str, flags=re.I):
  if m := re.search(pat, text, flags): return m, text[:m.start()] + ' ' + text[m.end():]
  return None, text

# Instruction aliases: LLVM uses different names for some instructions
_ALIASES = {
  'v_cmp_tru_f16': 'v_cmp_t_f16', 'v_cmp_tru_f32': 'v_cmp_t_f32', 'v_cmp_tru_f64': 'v_cmp_t_f64',
  'v_cmpx_tru_f16': 'v_cmpx_t_f16', 'v_cmpx_tru_f32': 'v_cmpx_t_f32', 'v_cmpx_tru_f64': 'v_cmpx_t_f64',
  'v_cvt_flr_i32_f32': 'v_cvt_floor_i32_f32', 'v_cvt_rpi_i32_f32': 'v_cvt_nearest_i32_f32',
  'v_ffbh_i32': 'v_cls_i32', 'v_ffbh_u32': 'v_clz_i32_u32', 'v_ffbl_b32': 'v_ctz_i32_b32',
  'v_cvt_pkrtz_f16_f32': 'v_cvt_pk_rtz_f16_f32', 'v_fmac_legacy_f32': 'v_fmac_dx9_zero_f32', 'v_mul_legacy_f32': 'v_mul_dx9_zero_f32',
  # SMEM aliases (dword -> b32, dwordx2 -> b64, etc.)
  's_load_dword': 's_load_b32', 's_load_dwordx2': 's_load_b64', 's_load_dwordx4': 's_load_b128',
  's_load_dwordx8': 's_load_b256', 's_load_dwordx16': 's_load_b512',
  's_buffer_load_dword': 's_buffer_load_b32', 's_buffer_load_dwordx2': 's_buffer_load_b64',
  's_buffer_load_dwordx4': 's_buffer_load_b128', 's_buffer_load_dwordx8': 's_buffer_load_b256',
  's_buffer_load_dwordx16': 's_buffer_load_b512',
  # VOP3 aliases
  'v_cvt_pknorm_i16_f16': 'v_cvt_pk_norm_i16_f16', 'v_cvt_pknorm_u16_f16': 'v_cvt_pk_norm_u16_f16',
  'v_add3_nc_u32': 'v_add3_u32', 'v_xor_add_u32': 'v_xad_u32',
  # VINTERP aliases
  'v_interp_p2_new_f32': 'v_interp_p2_f32',
  # SOP1 aliases
  's_ff1_i32_b32': 's_ctz_i32_b32', 's_ff1_i32_b64': 's_ctz_i32_b64',
  's_flbit_i32_b32': 's_clz_i32_u32', 's_flbit_i32_b64': 's_clz_i32_u64', 's_flbit_i32': 's_cls_i32', 's_flbit_i32_i64': 's_cls_i32_i64',
  's_andn1_saveexec_b32': 's_and_not0_saveexec_b32', 's_andn1_saveexec_b64': 's_and_not0_saveexec_b64',
  's_andn1_wrexec_b32': 's_and_not0_wrexec_b32', 's_andn1_wrexec_b64': 's_and_not0_wrexec_b64',
  's_andn2_saveexec_b32': 's_and_not1_saveexec_b32', 's_andn2_saveexec_b64': 's_and_not1_saveexec_b64',
  's_andn2_wrexec_b32': 's_and_not1_wrexec_b32', 's_andn2_wrexec_b64': 's_and_not1_wrexec_b64',
  's_orn1_saveexec_b32': 's_or_not0_saveexec_b32', 's_orn1_saveexec_b64': 's_or_not0_saveexec_b64',
  's_orn2_saveexec_b32': 's_or_not1_saveexec_b32', 's_orn2_saveexec_b64': 's_or_not1_saveexec_b64',
  # SOP2 aliases
  's_andn2_b32': 's_and_not1_b32', 's_andn2_b64': 's_and_not1_b64',
  's_orn2_b32': 's_or_not1_b32', 's_orn2_b64': 's_or_not1_b64',
  # VOP2 aliases
  'v_dot2c_f32_f16': 'v_dot2acc_f32_f16',
  # More VOP3 aliases
  'v_fma_legacy_f32': 'v_fma_dx9_zero_f32',
}

def _apply_alias(text: str) -> str:
  mn = text.split()[0].lower() if ' ' in text else text.lower().rstrip('_')
  # Try exact match first, then strip _e32/_e64 suffix
  for m in (mn, mn.removesuffix('_e32'), mn.removesuffix('_e64')):
    if m in _ALIASES: return _ALIASES[m] + text[len(m):]
  return text

def get_dsl(text: str) -> str:
  text, kw = _apply_alias(text.strip()), []
  # Extract modifiers
  for pat, val in [(r'\s+mul:2(?:\s|$)', 1), (r'\s+mul:4(?:\s|$)', 2), (r'\s+div:2(?:\s|$)', 3)]:
    if (m := _extract(text, pat))[0]: kw.append(f'omod={val}'); text = m[1]; break
  if (m := _extract(text, r'\s+clamp(?:\s|$)'))[0]: kw.append('clmp=1'); text = m[1]
  opsel, m, text = None, *_extract(text, r'\s+op_sel:\[([^\]]+)\]')
  if m:
    bits, mn = [int(x.strip()) for x in m.group(1).split(',')], text.split()[0].lower()
    is3p = mn.startswith(('v_pk_', 'v_wmma_', 'v_dot'))
    opsel = (bits[0] | (bits[1] << 1) | (bits[2] << 2)) if len(bits) == 3 and is3p else \
            (bits[0] | (bits[1] << 1) | (bits[2] << 3)) if len(bits) == 3 else sum(b << i for i, b in enumerate(bits))
  m, text = _extract(text, r'\s+wait_exp:(\d+)'); waitexp = m.group(1) if m else None
  m, text = _extract(text, r'\s+offset:(0x[0-9a-fA-F]+|-?\d+)'); off_val = m.group(1) if m else None
  m, text = _extract(text, r'\s+dlc(?:\s|$)'); dlc = 1 if m else None
  m, text = _extract(text, r'\s+glc(?:\s|$)'); glc = 1 if m else None
  m, text = _extract(text, r'\s+slc(?:\s|$)'); slc = 1 if m else None
  m, text = _extract(text, r'\s+tfe(?:\s|$)'); tfe = 1 if m else None
  m, text = _extract(text, r'\s+offen(?:\s|$)'); offen = 1 if m else None
  m, text = _extract(text, r'\s+idxen(?:\s|$)'); idxen = 1 if m else None
  m, text = _extract(text, r'\s+format:\[([^\]]+)\]'); fmt_val = m.group(1) if m else None
  m, text = _extract(text, r'\s+format:(\d+)'); fmt_val = m.group(1) if m and not fmt_val else fmt_val
  m, text = _extract(text, r'\s+neg_lo:\[([^\]]+)\]'); neg_lo = sum(int(x.strip()) << i for i, x in enumerate(m.group(1).split(','))) if m else None
  m, text = _extract(text, r'\s+neg_hi:\[([^\]]+)\]'); neg_hi = sum(int(x.strip()) << i for i, x in enumerate(m.group(1).split(','))) if m else None
  if waitexp: kw.append(f'waitexp={waitexp}')

  parts = text.replace(',', ' ').split()
  if not parts: raise ValueError("empty instruction")
  mn, op_str = parts[0].lower(), text[len(parts[0]):].strip()
  ops, args = _parse_ops(op_str), [_op2dsl(o) for o in _parse_ops(op_str)]

  # s_waitcnt
  if mn == 's_waitcnt':
    vm, exp, lgkm = 0x3f, 0x7, 0x3f
    for p in op_str.replace(',', ' ').split():
      if m := re.match(r'vmcnt\((\d+)\)', p): vm = int(m.group(1))
      elif m := re.match(r'expcnt\((\d+)\)', p): exp = int(m.group(1))
      elif m := re.match(r'lgkmcnt\((\d+)\)', p): lgkm = int(m.group(1))
      elif re.match(r'^0x[0-9a-f]+$|^\d+$', p): return f"s_waitcnt(simm16={int(p, 0)})"
    return f"s_waitcnt(simm16={waitcnt(vm, exp, lgkm)})"

  # VOPD
  if '::' in text:
    xp, yp = text.split('::')
    xps, yps = xp.strip().replace(',', ' ').split(), yp.strip().replace(',', ' ').split()
    xo, yo = [_op2dsl(p) for p in xps[1:]], [_op2dsl(p) for p in yps[1:]]
    vdx, sx0, vsx1 = xo[0], xo[1] if len(xo) > 1 else '0', xo[2] if len(xo) > 2 else 'v[0]'
    vdy, sy0, vsy1 = yo[0], yo[1] if len(yo) > 1 else '0', yo[2] if len(yo) > 2 else 'v[0]'
    lit = xo[3] if 'fmaak' in xps[0].lower() and len(xo) > 3 else yo[3] if 'fmaak' in yps[0].lower() and len(yo) > 3 else None
    if 'fmamk' in xps[0].lower() and len(xo) > 3: lit, vsx1 = xo[2], xo[3]
    elif 'fmamk' in yps[0].lower() and len(yo) > 3: lit, vsy1 = yo[2], yo[3]
    return f"VOPD(VOPDOp.{xps[0].upper()}, VOPDOp.{yps[0].upper()}, vdstx={vdx}, vdsty={vdy}, srcx0={sx0}, vsrcx1={vsx1}, srcy0={sy0}, vsrcy1={vsy1}{f', literal={lit}' if lit else ''})"

  # Special instructions
  if mn == 's_setreg_imm32_b32': raise ValueError(f"unsupported: {mn}")
  if mn in ('s_setpc_b64', 's_rfe_b64'): return f"{mn}(ssrc0={args[0]})"
  if mn in ('s_sendmsg_rtn_b32', 's_sendmsg_rtn_b64'): return f"{mn}(sdst={args[0]}, ssrc0=RawImm({args[1].strip()}))"
  if mn == 's_version': return f"{mn}(simm16={args[0]})"
  if mn == 's_setreg_b32': return f"{mn}(simm16={args[0]}, sdst={args[1]})"

  # SMEM
  if mn in SMEM_OPS:
    gs, ds = ", glc=1" if glc else "", ", dlc=1" if dlc else ""
    if len(ops) >= 3 and re.match(r'^-?[0-9]|^-?0x', ops[2].strip().lower()):
      return f"{mn}(sdata={args[0]}, sbase={args[1]}, offset={args[2]}, soffset=RawImm(124){gs}{ds})"
    if off_val and len(ops) >= 3: return f"{mn}(sdata={args[0]}, sbase={args[1]}, offset={off_val}, soffset={args[2]}{gs}{ds})"
    if len(ops) >= 3: return f"{mn}(sdata={args[0]}, sbase={args[1]}, soffset={args[2]}{gs}{ds})"

  # Buffer (MUBUF/MTBUF) instructions
  if mn.startswith(('buffer_', 'tbuffer_')):
    is_tbuf = mn.startswith('tbuffer_')
    # Parse format value for tbuffer
    fmt_num = None
    if fmt_val is not None:
      if fmt_val.isdigit(): fmt_num = int(fmt_val)
      else: fmt_num = BUF_FMT.get(fmt_val.replace(' ', '')) or _parse_buf_fmt_combo(fmt_val)
    # Handle special no-arg buffer ops
    if mn in ('buffer_gl0_inv', 'buffer_gl1_inv', 'buffer_wbl2', 'buffer_inv'): return f"{mn}()"
    # Build modifiers string
    buf_mods = "".join([f", offset={off_val}" if off_val else "", ", glc=1" if glc else "", ", dlc=1" if dlc else "",
                        ", slc=1" if slc else "", ", tfe=1" if tfe else "", ", offen=1" if offen else "", ", idxen=1" if idxen else ""])
    if is_tbuf and fmt_num is not None: buf_mods = f", format={fmt_num}" + buf_mods
    # Determine vaddr value (v[0] for 'off', actual register otherwise)
    vaddr_idx = 1
    if len(ops) > vaddr_idx and ops[vaddr_idx].strip().lower() == 'off': vaddr_val = "v[0]"
    else: vaddr_val = args[vaddr_idx] if len(args) > vaddr_idx else "v[0]"
    # srsrc and soffset indices depend on whether vaddr is 'off'
    srsrc_idx, soff_idx = (2, 3) if len(ops) > 1 else (1, 2)
    srsrc_val = args[srsrc_idx] if len(args) > srsrc_idx else "s[0:3]"
    soff_val = args[soff_idx] if len(args) > soff_idx else "0"
    # soffset: integers are inline constants, don't wrap in RawImm
    return f"{mn}(vdata={args[0]}, vaddr={vaddr_val}, srsrc={srsrc_val}, soffset={soff_val}{buf_mods})"

  # FLAT/GLOBAL/SCRATCH load/store/atomic - saddr needs RawImm(124) for off/null
  def _saddr(a): return 'RawImm(124)' if a in ('OFF', 'NULL') else a
  flat_mods = f"{f', offset={off_val}' if off_val else ''}{', glc=1' if glc else ''}{', slc=1' if slc else ''}{', dlc=1' if dlc else ''}"
  for pre, flds in [('flat_load','vdst,addr,saddr'), ('global_load','vdst,addr,saddr'), ('scratch_load','vdst,addr,saddr'),
                    ('flat_store','addr,data,saddr'), ('global_store','addr,data,saddr'), ('scratch_store','addr,data,saddr')]:
    if mn.startswith(pre) and len(args) >= 2:
      f0, f1, f2 = flds.split(',')
      return f"{mn}({f0}={args[0]}, {f1}={args[1]}{f', {f2}={_saddr(args[2])}' if len(args) >= 3 else ', saddr=RawImm(124)'}{flat_mods})"
  for pre in ('flat_atomic', 'global_atomic', 'scratch_atomic'):
    if mn.startswith(pre):
      if glc and len(args) >= 3: return f"{mn}(vdst={args[0]}, addr={args[1]}, data={args[2]}{f', saddr={_saddr(args[3])}' if len(args) >= 4 else ', saddr=RawImm(124)'}{flat_mods})"
      if len(args) >= 2: return f"{mn}(addr={args[0]}, data={args[1]}{f', saddr={_saddr(args[2])}' if len(args) >= 3 else ', saddr=RawImm(124)'}{flat_mods})"

  # DS instructions
  if mn.startswith('ds_'):
    off0, off1 = (str(int(off_val, 0) & 0xff), str((int(off_val, 0) >> 8) & 0xff)) if off_val else ("0", "0")
    gds_s = ", gds=1" if 'gds' in text.lower().split()[-1:] else ""
    off_kw = f", offset0={off0}, offset1={off1}{gds_s}"
    if mn == 'ds_nop' or mn in ('ds_gws_sema_v', 'ds_gws_sema_p', 'ds_gws_sema_release_all'): return f"{mn}({off_kw.lstrip(', ')})"
    if 'gws_' in mn: return f"{mn}(addr={args[0]}{off_kw})"
    if 'consume' in mn or 'append' in mn: return f"{mn}(vdst={args[0]}{off_kw})"
    if 'gs_reg' in mn: return f"{mn}(vdst={args[0]}, data0={args[1]}{off_kw})"
    if '2addr' in mn:
      if 'load' in mn: return f"{mn}(vdst={args[0]}, addr={args[1]}{off_kw})"
      if 'store' in mn and 'xchg' not in mn: return f"{mn}(addr={args[0]}, data0={args[1]}, data1={args[2]}{off_kw})"
      return f"{mn}(vdst={args[0]}, addr={args[1]}, data0={args[2]}, data1={args[3]}{off_kw})"
    if 'load' in mn: return f"{mn}(vdst={args[0]}{off_kw})" if 'addtid' in mn else f"{mn}(vdst={args[0]}, addr={args[1]}{off_kw})"
    if 'store' in mn and not _has(mn, 'cmp', 'xchg'):
      return f"{mn}(data0={args[0]}{off_kw})" if 'addtid' in mn else f"{mn}(addr={args[0]}, data0={args[1]}{off_kw})"
    if 'swizzle' in mn or 'ordered_count' in mn: return f"{mn}(vdst={args[0]}, addr={args[1]}{off_kw})"
    if 'permute' in mn: return f"{mn}(vdst={args[0]}, addr={args[1]}, data0={args[2]}{off_kw})"
    if 'bvh' in mn: return f"{mn}(vdst={args[0]}, addr={args[1]}, data0={args[2]}, data1={args[3]}{off_kw})"
    if 'condxchg' in mn: return f"{mn}(vdst={args[0]}, addr={args[1]}, data0={args[2]}{off_kw})"
    if _has(mn, 'cmpstore', 'mskor', 'wrap'):
      return f"{mn}(vdst={args[0]}, addr={args[1]}, data0={args[2]}, data1={args[3]}{off_kw})" if '_rtn' in mn else f"{mn}(addr={args[0]}, data0={args[1]}, data1={args[2]}{off_kw})"
    return f"{mn}(vdst={args[0]}, addr={args[1]}, data0={args[2]}{off_kw})" if '_rtn' in mn else f"{mn}(addr={args[0]}, data0={args[1]}{off_kw})"

  # v_fmaak/v_fmamk literal extraction
  lit_s = ""
  if mn in ('v_fmaak_f32', 'v_fmaak_f16') and len(args) == 4: lit_s, args = f", literal={args[3].strip()}", args[:3]
  elif mn in ('v_fmamk_f32', 'v_fmamk_f16') and len(args) == 4: lit_s, args = f", literal={args[2].strip()}", [args[0], args[1], args[3]]

  # VCC ops cleanup
  vcc_ops = {'v_add_co_ci_u32', 'v_sub_co_ci_u32', 'v_subrev_co_ci_u32'}
  if mn.replace('_e32', '') in vcc_ops and len(args) >= 5: mn, args = mn.replace('_e32', '') + '_e32', [args[0], args[2], args[3]]
  if mn.replace('_e64', '') in vcc_ops and mn.endswith('_e64'): mn = mn.replace('_e64', '')
  if mn.startswith('v_cmp') and not mn.endswith('_e64') and len(args) >= 3 and ops[0].strip().lower() in ('vcc_lo', 'vcc_hi', 'vcc'): args = args[1:]
  if 'cmpx' in mn and mn.endswith('_e64') and len(args) == 2: args = ['RawImm(126)'] + args
  # v_cmp_*_e64 has SGPR destination in vdst field - encode as RawImm
  _SGPR_NAMES = {'vcc_lo': 106, 'vcc_hi': 107, 'vcc': 106, 'null': 124, 'm0': 125, 'exec_lo': 126, 'exec_hi': 127}
  if mn.startswith('v_cmp') and 'cmpx' not in mn and mn.endswith('_e64') and len(args) >= 1:
    dst = ops[0].strip().lower()
    if dst.startswith('s') and dst[1:].isdigit(): args[0] = f'RawImm({int(dst[1:])})'
    elif dst.startswith('s[') and ':' in dst: args[0] = f'RawImm({int(dst[2:].split(":")[0])})'
    elif dst.startswith('ttmp') and dst[4:].isdigit(): args[0] = f'RawImm({108 + int(dst[4:])})'
    elif dst.startswith('ttmp[') and ':' in dst: args[0] = f'RawImm({108 + int(dst[5:].split(":")[0])})'
    elif dst in _SGPR_NAMES: args[0] = f'RawImm({_SGPR_NAMES[dst]})'

  fn = mn.replace('.', '_')
  if opsel is not None: args = [re.sub(r'\.[hl]$', '', a) for a in args]

  # v_fma_mix*: extract inline neg/abs modifiers
  if 'fma_mix' in mn and neg_lo is None and neg_hi is None:
    inline_neg, inline_abs, clean_args = 0, 0, [args[0]]
    for i, op in enumerate(ops[1:4]):
      op = op.strip()
      neg = op.startswith('-') and not (op[1:2].isdigit() or (len(op) > 2 and op[1] == '0' and op[2] in 'xX'))
      if neg: op = op[1:]
      abs_ = op.startswith('|') and op.endswith('|')
      if abs_: op = op[1:-1]
      if neg: inline_neg |= (1 << i)
      if abs_: inline_abs |= (1 << i)
      clean_args.append(_op2dsl(op))
    args = clean_args + args[4:]
    if inline_neg: neg_lo = inline_neg
    if inline_abs: neg_hi = inline_abs

  all_kw = list(kw)
  if lit_s: all_kw.append(lit_s.lstrip(', '))
  if opsel is not None: all_kw.append(f'opsel={opsel}')
  if neg_lo is not None: all_kw.append(f'neg={neg_lo}')
  if neg_hi is not None: all_kw.append(f'neg_hi={neg_hi}')
  if 'bvh' in mn and 'intersect_ray' in mn: all_kw.extend(['dmask=15', 'unrm=1', 'r128=1'])

  a_str, kw_str = ', '.join(args), ', '.join(all_kw)
  return f"{fn}({a_str}, {kw_str})" if kw_str and a_str else f"{fn}({kw_str})" if kw_str else f"{fn}({a_str})"

def asm(text: str) -> Inst:
  dsl = get_dsl(text)
  ns = {n: getattr(ins, n) for n in dir(ins) if not n.startswith('_')}
  ns.update({'s': s, 'v': v, 'ttmp': ttmp, 'abs': abs, 'RawImm': RawImm, 'SrcMod': SrcMod, 'VGPR': VGPR, 'SGPR': SGPR, 'TTMP': TTMP,
             'VCC_LO': VCC_LO, 'VCC_HI': VCC_HI, 'VCC': VCC, 'EXEC_LO': EXEC_LO, 'EXEC_HI': EXEC_HI, 'EXEC': EXEC, 'SCC': SCC, 'M0': M0, 'NULL': NULL, 'OFF': OFF})
  try: return eval(dsl, ns)
  except NameError:
    if m := re.match(r'^(v_\w+)(\(.*\))$', dsl): return eval(f"{m.group(1)}_e32{m.group(2)}", ns)
    raise

# ═══════════════════════════════════════════════════════════════════════════════
# CDNA DISASSEMBLER SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════

try:
  from extra.assembly.amd.autogen.cdna.ins import (VOP1 as CDNA_VOP1, VOP2 as CDNA_VOP2, VOPC as CDNA_VOPC, VOP3A, VOP3B, VOP3P as CDNA_VOP3P,
    SOP1 as CDNA_SOP1, SOP2 as CDNA_SOP2, SOPC as CDNA_SOPC, SOPK as CDNA_SOPK, SOPP as CDNA_SOPP, SMEM as CDNA_SMEM, DS as CDNA_DS,
    FLAT as CDNA_FLAT, MUBUF as CDNA_MUBUF, MTBUF as CDNA_MTBUF, SDWA, DPP, VOP1Op as CDNA_VOP1Op, VOP2Op as CDNA_VOP2Op, VOPCOp as CDNA_VOPCOp)

  def _cdna_src(inst, v, neg, abs_=0, n=1):
    s = inst.lit(v) if v == 255 else _fmt_src(v, n, cdna=True)
    if abs_: s = f"|{s}|"
    return f"neg({s})" if neg and v == 255 else (f"-{s}" if neg else s)

  # CDNA VOP2 aliases: new opcode name -> old name expected by LLVM tests
  _CDNA_VOP3_ALIASES = {'v_fmac_f64': 'v_mul_legacy_f32', 'v_dot2c_f32_bf16': 'v_mac_f32'}

  def _disasm_vop3a(inst) -> str:
    op_val = inst._values.get('op', 0)  # get raw opcode value, not enum value
    if hasattr(op_val, 'value'): op_val = op_val.value  # in case it's stored as enum
    name = inst.op_name.lower() or f'vop3a_op_{op_val}'
    from extra.assembly.amd.dsl import spec_num_srcs, spec_regs
    n = spec_num_srcs(name) if name else inst.num_srcs()
    cl, om = " clamp" if inst.clmp else "", _omod(inst.omod)
    orig_name = name
    name = _CDNA_VOP3_ALIASES.get(name, name)  # apply CDNA aliases
    # For aliased ops, recalculate sources without 64-bit assumption
    if name != orig_name:
      s0, s1 = _cdna_src(inst, inst.src0, inst.neg&1, inst.abs&1, 1), _cdna_src(inst, inst.src1, inst.neg&2, inst.abs&2, 1)
      s2 = ""
      dst = f"v{inst.vdst}"
    else:
      dregs, r0, r1, r2 = spec_regs(name) if name else (inst.dst_regs(), inst.src_regs(0), inst.src_regs(1), inst.src_regs(2))
      s0, s1, s2 = _cdna_src(inst, inst.src0, inst.neg&1, inst.abs&1, r0), _cdna_src(inst, inst.src1, inst.neg&2, inst.abs&2, r1), _cdna_src(inst, inst.src2, inst.neg&4, inst.abs&4, r2)
      dst = _vreg(inst.vdst, dregs) if dregs > 1 else f"v{inst.vdst}"
    # True VOP3 instructions (512+) - 3-source ops
    if op_val >= 512:
      return f"{name} {dst}, {s0}, {s1}, {s2}{cl}{om}" if n == 3 else f"{name} {dst}, {s0}, {s1}{cl}{om}"
    # VOPC (0-255): writes to SGPR pair, VOP2 (256-319): 2-3 src, VOP1 (320-511): 1 src
    if op_val < 256:
      sdst = _fmt_sdst(inst.vdst, 2, cdna=True)  # VOPC writes to 64-bit SGPR pair
      # v_cmpx_ also writes to sdst in CDNA VOP3 (unlike VOP32 where it writes to exec)
      return f"{name}_e64 {sdst}, {s0}, {s1}{cl}"
    if 320 <= op_val < 512:  # VOP1 promoted
      if name in ('v_nop', 'v_clrexcp'): return f"{name}_e64"
      return f"{name}_e64 {dst}, {s0}{cl}{om}"
    # VOP2 promoted (256-319)
    if name == 'v_cndmask_b32':
      s2 = _fmt_src(inst.src2, 2, cdna=True)  # src2 is 64-bit SGPR pair
      return f"{name}_e64 {dst}, {s0}, {s1}, {s2}{cl}{om}"
    if name in ('v_mul_legacy_f32', 'v_mac_f32'):
      return f"{name}_e64 {dst}, {s0}, {s1}{cl}{om}"
    suf = "_e64" if op_val < 512 else ""
    return f"{name}{suf} {dst}, {s0}, {s1}, {s2}{cl}{om}" if n == 3 else f"{name}{suf} {dst}, {s0}, {s1}{cl}{om}"

  # GFX9-specific VOP3B opcodes not in CDNA enum
  def _disasm_vop3b(inst) -> str:
    op_val = inst._values.get('op', 0)
    if hasattr(op_val, 'value'): op_val = op_val.value
    name = inst.op_name.lower() or f'vop3b_op_{op_val}'
    from extra.assembly.amd.dsl import spec_num_srcs, spec_regs
    n = spec_num_srcs(name) if name else inst.num_srcs()
    dregs, r0, r1, r2 = spec_regs(name) if name else (inst.dst_regs(), inst.src_regs(0), inst.src_regs(1), inst.src_regs(2))
    s0, s1, s2 = _cdna_src(inst, inst.src0, inst.neg&1, n=r0), _cdna_src(inst, inst.src1, inst.neg&2, n=r1), _cdna_src(inst, inst.src2, inst.neg&4, n=r2)
    dst = _vreg(inst.vdst, dregs) if dregs > 1 else f"v{inst.vdst}"
    sdst = _fmt_sdst(inst.sdst, 2, cdna=True)  # VOP3B sdst is always 64-bit SGPR pair
    cl, om = " clamp" if inst.clmp else "", _omod(inst.omod)
    # Carry ops need special handling
    if name in ('v_addc_co_u32', 'v_subb_co_u32', 'v_subbrev_co_u32'):
      s2 = _fmt_src(inst.src2, 2, cdna=True)  # src2 is carry-in (64-bit SGPR pair)
      return f"{name}_e64 {dst}, {sdst}, {s0}, {s1}, {s2}{cl}{om}"
    suf = "_e64" if 'co_' in name else ""
    return f"{name}{suf} {dst}, {sdst}, {s0}, {s1}, {s2}{cl}{om}" if n == 3 else f"{name}{suf} {dst}, {sdst}, {s0}, {s1}{cl}{om}"

  def _disasm_cdna_vop3p(inst) -> str:
    name, n, is_mfma = inst.op_name.lower(), inst.num_srcs(), 'mfma' in inst.op_name.lower() or 'smfmac' in inst.op_name.lower()
    get_src = lambda v, sc: inst.lit(v) if v == 255 else _fmt_src(v, sc, cdna=True)
    if is_mfma: sc = 2 if 'iu4' in name else 4 if 'iu8' in name or 'i4' in name else 8 if 'f16' in name or 'bf16' in name else 4; src0, src1, src2, dst = get_src(inst.src0, sc), get_src(inst.src1, sc), get_src(inst.src2, 16), _vreg(inst.vdst, 16)
    else: src0, src1, src2, dst = get_src(inst.src0, 1), get_src(inst.src1, 1), get_src(inst.src2, 1), f"v{inst.vdst}"
    opsel_hi = inst.opsel_hi | (inst.opsel_hi2 << 2)
    mods = ([_fmt_bits("op_sel", inst.opsel, n)] if inst.opsel else []) + ([_fmt_bits("op_sel_hi", opsel_hi, n)] if opsel_hi != (7 if n == 3 else 3) else []) + \
           ([_fmt_bits("neg_lo", inst.neg, n)] if inst.neg else []) + ([_fmt_bits("neg_hi", inst.neg_hi, n)] if inst.neg_hi else []) + (["clamp"] if inst.clmp else [])
    return f"{name} {dst}, {src0}, {src1}, {src2}{' ' + ' '.join(mods) if mods else ''}" if n == 3 else f"{name} {dst}, {src0}, {src1}{' ' + ' '.join(mods) if mods else ''}"

  _SEL = {0: 'BYTE_0', 1: 'BYTE_1', 2: 'BYTE_2', 3: 'BYTE_3', 4: 'WORD_0', 5: 'WORD_1', 6: 'DWORD'}
  _UNUSED = {0: 'UNUSED_PAD', 1: 'UNUSED_SEXT', 2: 'UNUSED_PRESERVE'}
  _DPP = {0x130: "wave_shl:1", 0x134: "wave_rol:1", 0x138: "wave_shr:1", 0x13c: "wave_ror:1", 0x140: "row_mirror", 0x141: "row_half_mirror", 0x142: "row_bcast:15", 0x143: "row_bcast:31"}

  def _sdwa_src0(v, is_sgpr, sext=0, neg=0, abs_=0):
    # s0=0: VGPR (v is VGPR number), s0=1: SGPR/constant (v is encoded like normal src)
    s = decode_src(v, cdna=True) if is_sgpr else f"v{v}"
    if sext: s = f"sext({s})"
    if abs_: s = f"|{s}|"
    return f"-{s}" if neg else s

  def _sdwa_vsrc1(v, sext=0, neg=0, abs_=0):
    # For VOP2 SDWA, vsrc1 is in vop_op field as raw VGPR number
    s = f"v{v}"
    if sext: s = f"sext({s})"
    if abs_: s = f"|{s}|"
    return f"-{s}" if neg else s

  _OMOD_SDWA = {0: "", 1: " mul:2", 2: " mul:4", 3: " div:2"}

  def _disasm_sdwa(inst) -> str:
    # SDWA format: vop2_op=63 -> VOP1, vop2_op=62 -> VOPC, vop2_op=0-61 -> VOP2
    vop2_op = inst.vop2_op
    src0 = _sdwa_src0(inst.src0, inst.s0, inst.src0_sext, inst.src0_neg, inst.src0_abs)
    clamp = " clamp" if inst.clmp else ""
    omod = _OMOD_SDWA.get(inst.omod, "")
    if vop2_op == 63:  # VOP1
      try: name = CDNA_VOP1Op(inst.vop_op).name.lower()
      except ValueError: name = f"vop1_op_{inst.vop_op}"
      dst = f"v{inst.vdst}"
      mods = [f"dst_sel:{_SEL[inst.dst_sel]}", f"dst_unused:{_UNUSED[inst.dst_u]}", f"src0_sel:{_SEL[inst.src0_sel]}"]
      return f"{name}_sdwa {dst}, {src0}{clamp}{omod} " + " ".join(mods)
    elif vop2_op == 62:  # VOPC
      try: name = CDNA_VOPCOp(inst.vdst).name.lower()  # opcode is in vdst field for VOPC SDWA
      except ValueError: name = f"vopc_op_{inst.vdst}"
      src1 = _sdwa_vsrc1(inst.vop_op, inst.src1_sext, inst.src1_neg, inst.src1_abs)  # vsrc1 is in vop_op field
      # VOPC SDWA: dst encoded in byte 5 (bits 47:40): 0=vcc, 128+n=s[n:n+1]
      sdst_enc = inst.dst_sel | (inst.dst_u << 3) | (inst.clmp << 5) | (inst.omod << 6)
      if sdst_enc == 0:
        sdst = "vcc"
      else:
        sdst_val = sdst_enc - 128 if sdst_enc >= 128 else sdst_enc
        sdst = _fmt_sdst(sdst_val, 2, cdna=True)
      mods = [f"src0_sel:{_SEL[inst.src0_sel]}", f"src1_sel:{_SEL[inst.src1_sel]}"]
      return f"{name}_sdwa {sdst}, {src0}, {src1} " + " ".join(mods)
    else:  # VOP2
      try: name = CDNA_VOP2Op(vop2_op).name.lower()
      except ValueError: name = f"vop2_op_{vop2_op}"
      name = _CDNA_DISASM_ALIASES.get(name, name)  # apply aliases (v_fmac -> v_mac, etc.)
      dst = f"v{inst.vdst}"
      src1 = _sdwa_vsrc1(inst.vop_op, inst.src1_sext, inst.src1_neg, inst.src1_abs)  # vsrc1 is in vop_op field
      mods = [f"dst_sel:{_SEL[inst.dst_sel]}", f"dst_unused:{_UNUSED[inst.dst_u]}", f"src0_sel:{_SEL[inst.src0_sel]}", f"src1_sel:{_SEL[inst.src1_sel]}"]
      # v_cndmask_b32 needs vcc as third operand
      if name == 'v_cndmask_b32':
        return f"{name}_sdwa {dst}, {src0}, {src1}, vcc{clamp}{omod} " + " ".join(mods)
      # Carry ops need vcc - v_addc/subb also need vcc as carry-in
      if name in ('v_addc_co_u32', 'v_subb_co_u32', 'v_subbrev_co_u32'):
        return f"{name}_sdwa {dst}, vcc, {src0}, {src1}, vcc{clamp}{omod} " + " ".join(mods)
      if '_co_' in name:
        return f"{name}_sdwa {dst}, vcc, {src0}, {src1}{clamp}{omod} " + " ".join(mods)
      return f"{name}_sdwa {dst}, {src0}, {src1}{clamp}{omod} " + " ".join(mods)

  def _dpp_src(v, neg=0, abs_=0):
    s = f"v{v}" if v < 256 else f"v{v - 256}"
    if abs_: s = f"|{s}|"
    return f"-{s}" if neg else s

  def _disasm_dpp(inst) -> str:
    # DPP format: vop2_op=63 -> VOP1, vop2_op=0-62 -> VOP2
    vop2_op = inst.vop2_op
    ctrl = inst.dpp_ctrl
    dpp = f"quad_perm:[{ctrl&3},{(ctrl>>2)&3},{(ctrl>>4)&3},{(ctrl>>6)&3}]" if ctrl < 0x100 else f"row_shl:{ctrl&0xf}" if ctrl < 0x110 else f"row_shr:{ctrl&0xf}" if ctrl < 0x120 else f"row_ror:{ctrl&0xf}" if ctrl < 0x130 else _DPP.get(ctrl, f"dpp_ctrl:0x{ctrl:x}")
    src0 = _dpp_src(inst.src0, inst.src0_neg, inst.src0_abs)
    # DPP modifiers: row_mask and bank_mask always shown, bound_ctrl:0 when bit=1
    mods = [dpp, f"row_mask:0x{inst.row_mask:x}", f"bank_mask:0x{inst.bank_mask:x}"] + (["bound_ctrl:0"] if inst.bound_ctrl else [])
    if vop2_op == 63:  # VOP1
      try: name = CDNA_VOP1Op(inst.vop_op).name.lower()
      except ValueError: name = f"vop1_op_{inst.vop_op}"
      return f"{name}_dpp v{inst.vdst}, {src0} " + " ".join(mods)
    else:  # VOP2
      try: name = CDNA_VOP2Op(vop2_op).name.lower()
      except ValueError: name = f"vop2_op_{vop2_op}"
      name = _CDNA_DISASM_ALIASES.get(name, name)
      src1 = _dpp_src(inst.vop_op, inst.src1_neg, inst.src1_abs)  # vsrc1 is in vop_op field
      if name == 'v_cndmask_b32':
        return f"{name}_dpp v{inst.vdst}, {src0}, {src1}, vcc " + " ".join(mods)
      if name in ('v_addc_co_u32', 'v_subb_co_u32', 'v_subbrev_co_u32'):
        return f"{name}_dpp v{inst.vdst}, vcc, {src0}, {src1}, vcc " + " ".join(mods)
      if '_co_' in name:
        return f"{name}_dpp v{inst.vdst}, vcc, {src0}, {src1} " + " ".join(mods)
      return f"{name}_dpp v{inst.vdst}, {src0}, {src1} " + " ".join(mods)

  # Register CDNA handlers - shared formats use merged disassemblers, CDNA-only formats use dedicated ones
  DISASM_HANDLERS.update({CDNA_VOP1: _disasm_vop1, CDNA_VOP2: _disasm_vop2, CDNA_VOPC: _disasm_vopc,
    CDNA_SOP1: _disasm_sop1, CDNA_SOP2: _disasm_sop2, CDNA_SOPC: _disasm_sopc, CDNA_SOPK: _disasm_sopk, CDNA_SOPP: _disasm_sopp,
    CDNA_SMEM: _disasm_smem, CDNA_DS: _disasm_ds, CDNA_FLAT: _disasm_flat, CDNA_MUBUF: _disasm_buf, CDNA_MTBUF: _disasm_buf,
    VOP3A: _disasm_vop3a, VOP3B: _disasm_vop3b, CDNA_VOP3P: _disasm_cdna_vop3p, SDWA: _disasm_sdwa, DPP: _disasm_dpp})
except ImportError:
  pass
