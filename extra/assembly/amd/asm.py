# RDNA3/CDNA assembler and disassembler
from __future__ import annotations
import re
from extra.assembly.amd.dsl import Inst, RawImm, Reg, SrcMod, SGPR, VGPR, TTMP, s, v, ttmp, _RegFactory
from extra.assembly.amd.dsl import VCC_LO, VCC_HI, VCC, EXEC_LO, EXEC_HI, EXEC, SCC, M0, NULL, OFF
from extra.assembly.amd.dsl import SPECIAL_GPRS, SPECIAL_GPRS_CDNA, SPECIAL_PAIRS, SPECIAL_PAIRS_CDNA, FLOAT_DEC, FLOAT_ENC, decode_src
from extra.assembly.amd.autogen.rdna3 import ins
from extra.assembly.amd.autogen.rdna3.ins import (VOP1, VOP2, VOP3, VOP3SD, VOP3P, VOPC, VOPD, VINTERP, SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, DS, FLAT, MUBUF, MTBUF, MIMG, EXP,
  VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOPDOp, SOP1Op, SOPKOp, SOPPOp, SMEMOp, DSOp, MUBUFOp, MTBUFOp)
from extra.assembly.amd.autogen.rdna3.enum import BufFmt

def _is_cdna(inst: Inst) -> bool: return 'cdna' in inst.__class__.__module__

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
  if arch in ("cdna", "gfx90a", "gfx942"):
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
# GFX942-specific HWREG values
_HWREG_GFX942 = {'HW_REG_XCC_ID': 20, 'HW_REG_SQ_PERF_SNAPSHOT_DATA': 21, 'HW_REG_SQ_PERF_SNAPSHOT_DATA1': 22,
                 'HW_REG_SQ_PERF_SNAPSHOT_PC_LO': 23, 'HW_REG_SQ_PERF_SNAPSHOT_PC_HI': 24}
HWREG_IDS = {v.lower(): k for k, v in HWREG.items()}
HWREG_IDS.update({k.lower(): v for k, v in _HWREG_GFX942.items()})
def hwreg(name, offset=0, size=32):
  """Encode hwreg(name[, offset[, size]]) -> simm16 value. id[5:0], offset[10:6], size-1[15:11]"""
  if isinstance(name, int): hid = name
  else: hid = HWREG_IDS.get(name.lower(), HWREG_IDS.get(name.lower().replace('hw_reg_', ''), None))
  if hid is None: raise ValueError(f"unknown hwreg: {name}")
  return hid | (offset << 6) | ((size - 1) << 11)
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
  """Format VOP3 source operand with modifiers."""
  if v == 255: s = inst.lit(v)  # literal constant takes priority
  elif n > 1: s = _fmt_src(v, n)
  elif f16 and v >= 256: s = f"v{v - 256}.h" if hi else f"v{v - 256}.l"
  else: s = inst.lit(v)
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
  if cdna:
    if soe and imm:
      off_s = f"{decode_src(inst.soffset, cdna)} offset:0x{inst.offset:x}"  # SGPR + immediate
    elif imm:
      off_s = f"0x{inst.offset:x}"  # Immediate offset only
    elif inst.offset < 256:
      off_s = decode_src(inst.offset, cdna)  # SGPR encoding in offset field
    else:
      off_s = decode_src(inst.soffset, cdna)
  elif inst.offset and inst.soffset != 124:
    off_s = f"{decode_src(inst.soffset, cdna)} offset:0x{inst.offset:x}"
  elif inst.offset:
    off_s = f"0x{inst.offset:x}"
  else:
    off_s = decode_src(inst.soffset, cdna)
  op_val = inst.op.value if hasattr(inst.op, 'value') else inst.op
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

  # Destination
  dn = inst.dst_regs()
  if op == VOP3Op.V_READLANE_B32: dst = _fmt_sdst(inst.vdst, 1)
  elif dn > 1: dst = _vreg(inst.vdst, dn)
  elif is16_d: dst = f"v{inst.vdst}.h" if (inst.opsel & 8) else f"v{inst.vdst}.l"
  else: dst = f"v{inst.vdst}"

  cl, om = " clamp" if inst.clmp else "", _omod(inst.omod)
  nonvgpr_opsel = (inst.src0 < 256 and (inst.opsel & 1)) or (inst.src1 < 256 and (inst.opsel & 2)) or (inst.src2 < 256 and (inst.opsel & 4))
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
    sc = 2 if 'iu4' in name else 4 if 'iu8' in name else 8
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

DISASM_HANDLERS = {VOP1: _disasm_vop1, VOP2: _disasm_vop2, VOPC: _disasm_vopc, VOP3: _disasm_vop3, VOP3SD: _disasm_vop3sd, VOPD: _disasm_vopd, VOP3P: _disasm_vop3p,
                   VINTERP: _disasm_vinterp, SOPP: _disasm_sopp, SMEM: _disasm_smem, DS: _disasm_ds, FLAT: _disasm_flat, MUBUF: _disasm_buf, MTBUF: _disasm_buf,
                   MIMG: _disasm_mimg, SOP1: _disasm_sop1, SOP2: _disasm_sop2, SOPC: _disasm_sopc, SOPK: _disasm_sopk}

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
            's_scratch_load_dword', 's_scratch_load_dwordx2', 's_scratch_load_dwordx4',
            's_scratch_store_dword', 's_scratch_store_dwordx2', 's_scratch_store_dwordx4',
            's_store_dword', 's_store_dwordx2', 's_store_dwordx4',
            's_buffer_store_dword', 's_buffer_store_dwordx2', 's_buffer_store_dwordx4',
            's_atc_probe', 's_atc_probe_buffer'}
SPEC_DSL = {'vcc_lo': 'VCC_LO', 'vcc_hi': 'VCC_HI', 'vcc': 'VCC_LO', 'null': 'NULL', 'off': 'OFF', 'm0': 'M0',
            'exec_lo': 'EXEC_LO', 'exec_hi': 'EXEC_HI', 'exec': 'EXEC_LO', 'scc': 'SCC', 'src_scc': 'SCC'}
SPEC_DSL_CDNA = {**SPEC_DSL, 'src_scc': 'SRC_SCC', 'flat_scratch_lo': 'FLAT_SCRATCH_LO', 'flat_scratch_hi': 'FLAT_SCRATCH_HI',
                 'flat_scratch': 'FLAT_SCRATCH', 'xnack_mask_lo': 'XNACK_MASK_LO', 'xnack_mask_hi': 'XNACK_MASK_HI', 'xnack_mask': 'XNACK_MASK',
                 'src_vccz': 'SRC_VCCZ', 'src_execz': 'SRC_EXECZ', 'vccz': 'SRC_VCCZ', 'execz': 'SRC_EXECZ',
                 'src_lds_direct': 'SRC_LDS_DIRECT', 'lds_direct': 'SRC_LDS_DIRECT'}

def _op2dsl(op: str, arch: str = "rdna3") -> str:
  op = op.strip()
  neg = op.startswith('-') and not (op[1:2].isdigit() or (len(op) > 2 and op[1] == '0' and op[2] in 'xX'))
  if neg: op = op[1:]
  abs_ = (op.startswith('|') and op.endswith('|')) or (op.startswith('abs(') and op.endswith(')'))
  if abs_: op = op[1:-1] if op.startswith('|') else op[4:-1]
  hi = ".h" if op.endswith('.h') else ".l" if op.endswith('.l') else ""
  if hi: op = op[:-2]
  lo = op.lower()
  spec_dsl = SPEC_DSL_CDNA if arch == "cdna" else SPEC_DSL
  def wrap(b): return f"{'-' if neg else ''}abs({b}){hi}" if abs_ else f"-{b}{hi}" if neg else f"{b}{hi}"
  if lo in spec_dsl: return wrap(spec_dsl[lo])
  if op in FLOATS: return wrap(op)
  rp = {'s': 's', 'v': 'v', 't': 'ttmp', 'ttmp': 'ttmp'}
  if m := re.match(r'^([asvt](?:tmp)?)\[(\d+):(\d+)\]$', lo): return wrap(f"{rp.get(m.group(1), m.group(1))}[{m.group(2)}:{m.group(3)}]")
  if m := re.match(r'^([asvt](?:tmp)?)(\d+)$', lo): return wrap(f"{rp.get(m.group(1), m.group(1))}[{m.group(2)}]")
  if re.match(r'^-?\d+$|^-?0x[0-9a-fA-F]+$', op): return f"SrcMod({op}, neg={neg}, abs_={abs_})" if neg or abs_ else op
  # Floating-point literal: convert to IEEE 754 32-bit integer representation
  import struct
  try:
    f = float(op)
    as_int = struct.unpack('<I', struct.pack('<f', f))[0]
    return f"SrcMod({as_int}, neg={neg}, abs_={abs_})" if neg or abs_ else str(as_int)
  except ValueError: pass
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

def _parse_src_mods(raw: str) -> tuple[str, bool, bool, bool]:
  """Parse neg/abs/sext modifiers from operand string. Returns (stripped_op, neg, abs_, sext)."""
  neg = raw.startswith('-') and not raw[1:2].isdigit() and raw[1:3] != '0.'
  if neg: raw = raw[1:]
  abs_ = raw.startswith('|') and raw.endswith('|')
  if abs_: raw = raw[1:-1]
  sext = raw.startswith('sext(') and raw.endswith(')')
  if sext: raw = raw[5:-1]
  return raw, neg, abs_, sext

_SGPR_BY_NAME = {v: k for k, v in SPECIAL_GPRS_CDNA.items()}
_SGPR_BY_NAME.update({'src_vccz': 251, 'src_execz': 252, 'src_scc': 253, 'vcc': 106})

def _parse_sdwa_src(raw: str) -> tuple[int, int]:
  """Parse SDWA source operand. Returns (value, s_flag) where s_flag=1 for SGPR/literal."""
  if raw.startswith('v') and (raw[1:].isdigit() or raw[1] == '['): return int(raw.split('[')[1].split(']')[0]) if '[' in raw else int(raw[1:]), 0
  if raw.startswith('s') and (raw[1:].isdigit() or raw[1] == '['): return int(raw.split('[')[1].split(':')[0]) if '[' in raw else int(raw[1:]), 1
  if raw.startswith('ttmp') and raw[4:].isdigit(): return 108 + int(raw[4:]), 1
  if raw in _SGPR_BY_NAME: return _SGPR_BY_NAME[raw], 1
  # Inline constants: integers 0-64 -> 128+N, -1 to -16 -> 192+abs(N), floats use FLOAT_ENC
  if raw.lstrip('-').replace('.', '', 1).isdigit():
    if '.' in raw: return FLOAT_ENC.get(float(raw), 128), 1
    ival = int(raw)
    if 0 <= ival <= 64: return 128 + ival, 1
    if -16 <= ival < 0: return 192 + (-ival), 1
  return 0, 0

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
# RDNA3-only aliases (should NOT be applied to CDNA) - CDNA uses OLD names, RDNA3 uses NEW names
_RDNA3_ONLY_ALIASES = {'v_mul_legacy_f32', 'v_fmac_legacy_f32', 'v_fma_legacy_f32',
  's_load_dword', 's_load_dwordx2', 's_load_dwordx4', 's_load_dwordx8', 's_load_dwordx16',
  's_buffer_load_dword', 's_buffer_load_dwordx2', 's_buffer_load_dwordx4', 's_buffer_load_dwordx8', 's_buffer_load_dwordx16',
  # SOP: CDNA uses s_andn2/s_orn2, RDNA3 uses s_and_not1/s_or_not1
  's_andn2_b32', 's_andn2_b64', 's_orn2_b32', 's_orn2_b64',
  's_andn1_saveexec_b32', 's_andn1_saveexec_b64', 's_andn1_wrexec_b32', 's_andn1_wrexec_b64',
  's_andn2_saveexec_b32', 's_andn2_saveexec_b64', 's_andn2_wrexec_b32', 's_andn2_wrexec_b64',
  's_orn1_saveexec_b32', 's_orn1_saveexec_b64', 's_orn2_saveexec_b32', 's_orn2_saveexec_b64',
  # VOP1: CDNA uses old names
  'v_cvt_flr_i32_f32', 'v_cvt_rpi_i32_f32', 'v_ffbh_i32', 'v_ffbh_u32', 'v_ffbl_b32',
  # VOPC: CDNA uses tru suffix for float comparisons
  'v_cmp_tru_f16', 'v_cmp_tru_f32', 'v_cmp_tru_f64', 'v_cmpx_tru_f16', 'v_cmpx_tru_f32', 'v_cmpx_tru_f64'}
# CDNA-specific aliases (GFX9 uses different names for some instructions)
# CDNA-specific aliases - CDNA uses dword naming, not b32
_CDNA_ALIASES = {
  # VOP aliases: madmk/madak -> fmamk/fmaak (same encoding, different name in CDNA enum)
  'v_cvt_pkrtz_f16_f32': 'v_cvt_pk_rtz_f16_f32', 'v_madmk_f32': 'v_fmamk_f32', 'v_madak_f32': 'v_fmaak_f32',
  # VOPC: v_cmp_t_fXX -> v_cmp_tru_fXX for CDNA float comparisons
  'v_cmp_t_f16': 'v_cmp_tru_f16', 'v_cmp_t_f32': 'v_cmp_tru_f32', 'v_cmp_t_f64': 'v_cmp_tru_f64',
  'v_cmpx_t_f16': 'v_cmpx_tru_f16', 'v_cmpx_t_f32': 'v_cmpx_tru_f32', 'v_cmpx_t_f64': 'v_cmpx_tru_f64',
}

def _apply_alias(text: str, arch: str = "rdna3") -> str:
  mn = text.split()[0].lower() if ' ' in text else text.lower().rstrip('_')
  aliases = _CDNA_ALIASES if arch == "cdna" else _ALIASES
  # Try exact match first, then strip _e32/_e64 suffix
  for m in (mn, mn.removesuffix('_e32'), mn.removesuffix('_e64')):
    if m in aliases: return aliases[m] + text[len(m):]
    # Also check common aliases, but skip RDNA3-only ones for CDNA
    if m in _ALIASES and not (arch == "cdna" and m in _RDNA3_ONLY_ALIASES): return _ALIASES[m] + text[len(m):]
  return text

def get_dsl(text: str, arch: str = "rdna3", gfx942: bool = False) -> str:
  text, kw = _apply_alias(text.strip(), arch), []
  # Extract modifiers
  for pat, val in [(r'\s+mul:2(?:\s|$)', 1), (r'\s+mul:4(?:\s|$)', 2), (r'\s+div:2(?:\s|$)', 3)]:
    if (m := _extract(text, pat))[0]: kw.append(f'omod={val}'); text = m[1]; break
  if (m := _extract(text, r'\s+clamp(?:\s|$)'))[0]: kw.append('clmp=1'); text = m[1]
  opsel, m, text = None, *_extract(text, r'\s+op_sel:\[([^\]]+)\]')
  if m:
    bits, mn = [int(x.strip()) for x in m.group(1).split(',')], text.split()[0].lower()
    is3p = mn.startswith(('v_pk_', 'v_wmma_', 'v_dot', 'v_mad_mix', 'v_fma_mix'))
    opsel = (bits[0] | (bits[1] << 1) | (bits[2] << 2)) if len(bits) == 3 and is3p else \
            (bits[0] | (bits[1] << 1) | (bits[2] << 3)) if len(bits) == 3 else sum(b << i for i, b in enumerate(bits))
  m, text = _extract(text, r'\s+wait_exp:(\d+)'); waitexp = m.group(1) if m else None
  m, text = _extract(text, r'\s+offset:(0x[0-9a-fA-F]+|-?\d+)'); off_val = m.group(1) if m else None
  # Flag modifiers: extract presence/absence
  flags = {}
  for f in ('dlc', 'glc', 'slc', 'tfe', 'offen', 'idxen', 'gds', 'lds'):
    m, text = _extract(text, rf'\s+{f}(?:\s|$)'); flags[f] = 1 if m else None
  dlc, glc, slc, tfe, offen, idxen, gds, lds = [flags[f] for f in ('dlc', 'glc', 'slc', 'tfe', 'offen', 'idxen', 'gds', 'lds')]
  # GFX942: sc0, sc1, nt with negation variants
  for f in ('sc0', 'sc1', 'nt'):
    m, text = _extract(text, rf'\s+{f}(?:\s|$)'); flags[f] = 1 if m else None
    m, text = _extract(text, rf'\s+no{f}(?:\s|$)'); flags[f] = 0 if m else flags[f]
  sc0, sc1, nt = flags['sc0'], flags['sc1'], flags['nt']
  m, text = _extract(text, r'\s+format:\[([^\]]+)\]'); fmt_val = m.group(1) if m else None
  m, text = _extract(text, r'\s+format:(\d+)'); fmt_val = m.group(1) if m and not fmt_val else fmt_val
  # dfmt:N, nfmt:N can appear as comma-separated items (CDNA ACC style) or as space-separated modifiers
  m, text = _extract(text, r',\s*dfmt:(\d+)'); dfmt_val = int(m.group(1)) if m else None
  if not m: m, text = _extract(text, r'\s+dfmt:(\d+)'); dfmt_val = int(m.group(1)) if m else dfmt_val
  m, text = _extract(text, r',\s*nfmt:(\d+)'); nfmt_val = int(m.group(1)) if m else None
  if not m: m, text = _extract(text, r'\s+nfmt:(\d+)'); nfmt_val = int(m.group(1)) if m else nfmt_val
  m, text = _extract(text, r'\s+neg_lo:\[([^\]]+)\]'); neg_lo = sum(int(x.strip()) << i for i, x in enumerate(m.group(1).split(','))) if m else None
  m, text = _extract(text, r'\s+neg_hi:\[([^\]]+)\]'); neg_hi = sum(int(x.strip()) << i for i, x in enumerate(m.group(1).split(','))) if m else None
  m, text = _extract(text, r'\s+op_sel_hi:\[([^\]]+)\]')
  opsel_hi, opsel_hi_count = (sum(int(x.strip()) << i for i, x in enumerate(m.group(1).split(','))), len(m.group(1).split(','))) if m else (None, 0)
  m, text = _extract(text, r'\s+offset0:(\d+)'); offset0 = m.group(1) if m else None
  m, text = _extract(text, r'\s+offset1:(\d+)'); offset1 = m.group(1) if m else None
  # MAI instruction modifiers (MFMA cbsz/abid/blgp)
  m, text = _extract(text, r'\s+cbsz:(\d+)'); cbsz = int(m.group(1)) if m else None
  m, text = _extract(text, r'\s+abid:(\d+)'); abid = int(m.group(1)) if m else None
  m, text = _extract(text, r'\s+blgp:(\d+)'); blgp = int(m.group(1)) if m else None
  # MFMA neg:[x,y,z] modifier -> sets neg field (same as blgp for MFMA)
  m, text = _extract(text, r'\s+neg:\[([^\]]+)\]'); mfma_neg = sum(int(x.strip()) << i for i, x in enumerate(m.group(1).split(','))) if m else None
  # SDWA modifiers: sel values are BYTE_0-3=0-3, WORD_0-1=4-5, DWORD=6; dst_unused PAD=0, SEXT=1, PRESERVE=2
  def _sel(s): return {'BYTE_0': 0, 'BYTE_1': 1, 'BYTE_2': 2, 'BYTE_3': 3, 'WORD_0': 4, 'WORD_1': 5, 'DWORD': 6}.get(s, 6)
  m, text = _extract(text, r'\s+dst_sel:(\w+)'); sdwa_dst_sel = _sel(m.group(1)) if m else None
  m, text = _extract(text, r'\s+dst_unused:(\w+)'); sdwa_dst_unused = {'UNUSED_PAD': 0, 'UNUSED_SEXT': 1, 'UNUSED_PRESERVE': 2}.get(m.group(1), 0) if m else None
  m, text = _extract(text, r'\s+src0_sel:(\w+)'); sdwa_src0_sel = _sel(m.group(1)) if m else None
  m, text = _extract(text, r'\s+src1_sel:(\w+)'); sdwa_src1_sel = _sel(m.group(1)) if m else None
  m, text = _extract(text, r'\s+sext\(src0\)'); sdwa_src0_sext = 1 if m else None
  m, text = _extract(text, r'\s+sext\(src1\)'); sdwa_src1_sext = 1 if m else None
  # DPP modifiers: quad_perm, row_shl/shr/ror, wave_shl/rol/shr/ror, row_mirror, row_bcast, etc.
  m, text = _extract(text, r'\s+quad_perm:\[(\d+),(\d+),(\d+),(\d+)\]')
  dpp_ctrl = int(m.group(1)) | (int(m.group(2)) << 2) | (int(m.group(3)) << 4) | (int(m.group(4)) << 6) if m else None
  for pat, base in [('row_shl', 0x100), ('row_shr', 0x110), ('row_ror', 0x120), ('row_newbcast', 0x150)]:
    m, text = _extract(text, rf'\s+{pat}:(\d+)'); dpp_ctrl = base + int(m.group(1)) if m else dpp_ctrl
  for pat, val in [('wave_shl:1', 0x130), ('wave_rol:1', 0x134), ('wave_shr:1', 0x138), ('wave_ror:1', 0x13c),
                   ('row_mirror', 0x140), ('row_half_mirror', 0x141), ('row_bcast:15', 0x142), ('row_bcast:31', 0x143)]:
    m, text = _extract(text, rf'\s+{pat}(?:\s|$)'); dpp_ctrl = val if m else dpp_ctrl
  m, text = _extract(text, r'\s+row_mask:(0x[0-9a-fA-F]+|\d+)'); dpp_row_mask = int(m.group(1), 0) if m else None; dpp_row_mask_specified = m is not None
  m, text = _extract(text, r'\s+bank_mask:(0x[0-9a-fA-F]+|\d+)'); dpp_bank_mask = int(m.group(1), 0) if m else None; dpp_bank_mask_specified = m is not None
  m, text = _extract(text, r'\s+bound_ctrl:([01])'); dpp_bound_ctrl = 1 if m else None
  if waitexp: kw.append(f'waitexp={waitexp}')

  parts = text.replace(',', ' ').split()
  if not parts: raise ValueError("empty instruction")
  mn, op_str = parts[0].lower(), text[len(parts[0]):].strip()
  ops, args = _parse_ops(op_str), [_op2dsl(o, arch) for o in _parse_ops(op_str)]

  # s_waitcnt
  if mn == 's_waitcnt':
    vm, exp, lgkm = 0x3f, 0x7, 0x3f
    for p in op_str.replace(',', ' ').split():
      if m := re.match(r'vmcnt\((\d+)\)', p): vm = int(m.group(1))
      elif m := re.match(r'expcnt\((\d+)\)', p): exp = int(m.group(1))
      elif m := re.match(r'lgkmcnt\((\d+)\)', p): lgkm = int(m.group(1))
      elif re.match(r'^0x[0-9a-f]+$|^\d+$', p): return f"s_waitcnt(simm16={int(p, 0)})"
    return f"s_waitcnt(simm16={waitcnt(vm, exp, lgkm)})"

  # MAI instructions (CDNA): v_mfma_*, v_accvgpr_*, v_smfmac_*
  if arch == "cdna" and mn.startswith(('v_mfma_', 'v_accvgpr_', 'v_smfmac_')):
    from extra.assembly.amd.autogen.cdna.ins import VOP3POp, VOP1Op
    # Handle aliases: v_accvgpr_read_b32 -> v_accvgpr_read, v_accvgpr_write_b32 -> v_accvgpr_write
    fn = mn.replace('_b32', '').upper()
    # MFMA/SMFMAC name mapping: LLVM v_mfma_f32_32x32x1f32 -> enum V_MFMA_F32_32X32X1_2B_F32
    def _mfma_alias(n):
      n = n.replace('_1K', '')  # Strip _1K suffix first (same opcodes)
      # FP8/BF8 pairs: insert underscore between AND before: 64BF8BF8 -> 64_BF8_BF8
      for t in ('BF8BF8', 'BF8FP8', 'FP8BF8', 'FP8FP8'):
        if t in n: n = n.replace(t, '_' + t[:3] + '_' + t[3:])
      # Insert underscore before dtype suffix
      for t in ('F32', 'F16', 'BF16', 'I8', 'F64', 'XF32'):
        n = re.sub(rf'(X\d+)({t})$', rf'\1_{t}', n)
      # Block sizes for specific shapes
      for pat, blk in [('32X32X1_', '32X32X1_2B_'), ('16X16X1_', '16X16X1_4B_'), ('4X4X1_', '4X4X1_16B_'),
                       ('32X32X4_F16', '32X32X4_2B_F16'), ('16X16X4_F16', '16X16X4_4B_F16'), ('4X4X4_F16', '4X4X4_16B_F16'),
                       ('32X32X4_I8', '32X32X4_2B_I8'), ('16X16X4_I8', '16X16X4_4B_I8'), ('4X4X4_I8', '4X4X4_16B_I8'),
                       ('32X32X4_BF16', '32X32X4_2B_BF16'), ('16X16X4_BF16', '16X16X4_4B_BF16'), ('4X4X4_BF16', '4X4X4_16B_BF16'),
                       ('4X4X4_F64', '4X4X4_4B_F64')]:
        n = n.replace(pat, blk)
      return n
    _MFMA_ALIASES = {n: _mfma_alias(n) for n in [
      'V_MFMA_F32_32X32X1F32', 'V_MFMA_F32_16X16X1F32', 'V_MFMA_F32_4X4X1F32', 'V_MFMA_F32_32X32X2F32', 'V_MFMA_F32_16X16X4F32',
      'V_MFMA_F32_32X32X4F16', 'V_MFMA_F32_16X16X4F16', 'V_MFMA_F32_4X4X4F16', 'V_MFMA_F32_32X32X8F16', 'V_MFMA_F32_16X16X16F16',
      'V_MFMA_F32_32X32X16F16', 'V_MFMA_F32_16X16X32F16',
      'V_MFMA_I32_32X32X4I8', 'V_MFMA_I32_16X16X4I8', 'V_MFMA_I32_4X4X4I8', 'V_MFMA_I32_32X32X16I8', 'V_MFMA_I32_16X16X32I8',
      'V_MFMA_F64_16X16X4F64', 'V_MFMA_F64_4X4X4F64',
      'V_MFMA_F32_32X32X4BF16', 'V_MFMA_F32_16X16X4BF16', 'V_MFMA_F32_4X4X4BF16', 'V_MFMA_F32_32X32X8BF16', 'V_MFMA_F32_16X16X16BF16',
      'V_MFMA_F32_32X32X16BF16', 'V_MFMA_F32_16X16X32BF16',
      'V_MFMA_F32_32X32X4BF16_1K', 'V_MFMA_F32_16X16X4BF16_1K', 'V_MFMA_F32_4X4X4BF16_1K', 'V_MFMA_F32_32X32X8BF16_1K', 'V_MFMA_F32_16X16X16BF16_1K',
      'V_MFMA_F32_16X16X8XF32', 'V_MFMA_F32_32X32X4XF32',
      'V_SMFMAC_F32_16X16X32F16', 'V_SMFMAC_F32_32X32X16F16', 'V_SMFMAC_F32_16X16X32BF16', 'V_SMFMAC_F32_32X32X16BF16',
      'V_SMFMAC_I32_16X16X64I8', 'V_SMFMAC_I32_32X32X32I8',
      'V_SMFMAC_F32_16X16X64BF8BF8', 'V_SMFMAC_F32_16X16X64BF8FP8', 'V_SMFMAC_F32_16X16X64FP8BF8', 'V_SMFMAC_F32_16X16X64FP8FP8',
      'V_SMFMAC_F32_32X32X32BF8BF8', 'V_SMFMAC_F32_32X32X32BF8FP8', 'V_SMFMAC_F32_32X32X32FP8BF8', 'V_SMFMAC_F32_32X32X32FP8FP8']}
    # GFX90a-specific opcodes (different from gfx942 enum): map to raw opcode values
    # These instructions use opcodes that don't match our gfx942-based enum
    _MFMA_GFX90A_OPS = {
      'V_MFMA_I32_32X32X8I8': 84, 'V_MFMA_I32_16X16X16I8': 85,
      'V_MFMA_F32_32X32X2BF16': 104, 'V_MFMA_F32_16X16X2BF16': 105, 'V_MFMA_F32_4X4X2BF16': 107,
      'V_MFMA_F32_32X32X4BF16': 108, 'V_MFMA_F32_16X16X8BF16': 109,
      'V_MFMA_F32_32X32X4BF16_1K': 99, 'V_MFMA_F32_16X16X4BF16_1K': 100, 'V_MFMA_F32_4X4X4BF16_1K': 101,
      'V_MFMA_F32_32X32X8BF16_1K': 102, 'V_MFMA_F32_16X16X16BF16_1K': 103,
    }
    # Check for gfx90a-specific opcodes BEFORE applying aliases (gfx90a uses different opcodes than gfx942)
    gfx90a_op = _MFMA_GFX90A_OPS.get(fn.upper()) if not gfx942 else None
    # Apply aliases for gfx942 (or when no gfx90a-specific opcode exists)
    if gfx90a_op is None:
      fn = _MFMA_ALIASES.get(fn, fn)
    # v_accvgpr_mov_b32 is VOP1, not VOP3P
    if mn.startswith('v_accvgpr_mov'):
      vop1_op = getattr(VOP1Op, fn, None)
      if vop1_op is None: raise ValueError(f"unknown MAI instruction: {mn}")
      # v_accvgpr_mov_b32 a1, a2 -> VOP1 with dst=a1, src=a2
      # dst is ACC (a[N]), src is ACC (a[N])
      # In VOP1 encoding: vdst uses VGPR number, src0 uses 256+N for ACC registers
      dst_m = re.match(r'a\[?(\d+)', ops[0]) if ops else None
      src_m = re.match(r'a\[?(\d+)', ops[1]) if len(ops) > 1 else None
      if not dst_m or not src_m: raise ValueError(f"v_accvgpr_mov requires ACC registers: {mn} {ops}")
      return f"v_accvgpr_mov_b32_e32(vdst=v[{dst_m.group(1)}], src0=RawImm({256 + int(src_m.group(1))}))"
    # VOP3P MAI instructions
    vop3p_op = getattr(VOP3POp, fn, None)
    if vop3p_op is None and gfx90a_op is None: raise ValueError(f"unknown MAI instruction: {mn}")
    # Parse operands: vdst, src0, src1[, src2] - can be VGPRs (v[N]) or ACCs (a[N])
    # ACC encoding: VGPRs and ACCs both use 256+N in src fields, ACC flags in opsel_hi/clmp
    # NOTE: we detect ACC from the ORIGINAL operand string, not the converted DSL
    def parse_mai_reg(orig_op, dsl_arg):
      """Parse MAI register from original operand, return (reg_num, is_acc)"""
      orig = orig_op.strip().lower()
      # Check original operand for ACC (a[N] or aN)
      if m := re.match(r'a\[?(\d+)', orig): return int(m.group(1)), True
      if m := re.match(r'v\[?(\d+)', orig): return int(m.group(1)), False
      # For literals, use DSL conversion
      if m := re.match(r'v\[(\d+)(?::\d+)?\]', dsl_arg): return int(m.group(1)), False
      # Handle literals and other sources
      return dsl_arg, False
    vdst_num, vdst_acc = parse_mai_reg(ops[0], args[0]) if ops else (0, False)
    src0_num, src0_acc = parse_mai_reg(ops[1], args[1]) if len(ops) > 1 else (0, False)
    src1_num, src1_acc = parse_mai_reg(ops[2], args[2]) if len(ops) > 2 else (0, False)
    src2_num, src2_acc = parse_mai_reg(ops[3], args[3]) if len(ops) > 3 else (0, False)
    # Build src values: VGPRs/ACCs use 256+N encoding
    def mai_src(num, is_acc):
      if isinstance(num, str): return num  # literal or special
      return f"RawImm({256 + num})"
    # Build VOP3P call with proper ACC flags
    # opsel_hi[0] (bit 59) = src0 is ACC, opsel_hi[1] (bit 60) = src1 is ACC
    # clmp (bit 15) = vdst is ACC (only for MFMA/SMFMAC, not for v_accvgpr_*)
    # For MFMA/SMFMAC, src2 ACC is encoded in the src2 value (256+N if ACC)
    opsel_hi_val = (1 if src0_acc else 0) | ((1 if src1_acc else 0) << 1)
    is_mfma = 'mfma' in mn or 'smfmac' in mn
    clmp_val = 1 if vdst_acc and is_mfma else 0  # Only set clmp for MFMA with ACC vdst
    # cbsz -> neg_hi, abid -> opsel, blgp -> neg
    mai_mods = []
    # MFMA/SMFMAC need explicit opsel_hi based on ACC flags, v_accvgpr_* use VOP3P defaults
    if is_mfma:
      mai_mods = [f'opsel_hi={opsel_hi_val}', 'opsel_hi2=0']
    if clmp_val: mai_mods.append(f'clmp={clmp_val}')
    if cbsz is not None: mai_mods.append(f'neg_hi={cbsz}')
    if abid is not None: mai_mods.append(f'opsel={abid}')
    # blgp and neg:[x,y,z] both set the neg field
    neg_val = mfma_neg if mfma_neg is not None else blgp
    if neg_val is not None: mai_mods.append(f'neg={neg_val}')
    # v_accvgpr_read/write have 2 operands, MFMA/SMFMAC have 4
    if mn.startswith('v_accvgpr_read'):
      # v_accvgpr_read vdst, src0 (src0 is ACC register)
      return f"{fn.lower()}(vdst=v[{vdst_num}], src0={mai_src(src0_num, src0_acc)}, src1=RawImm(0), src2=RawImm(0){', ' + ', '.join(mai_mods) if mai_mods else ''})"
    if mn.startswith('v_accvgpr_write'):
      # v_accvgpr_write vdst, src0 (vdst is ACC register)
      return f"{fn.lower()}(vdst=v[{vdst_num}], src0={mai_src(src0_num, src0_acc)}, src1=RawImm(0), src2=RawImm(0){', ' + ', '.join(mai_mods) if mai_mods else ''})"
    # MFMA/SMFMAC: 4 operands
    src2_val = mai_src(src2_num, src2_acc)
    # Use raw VOP3P with explicit op for gfx90a-specific opcodes
    if gfx90a_op is not None:
      return f"VOP3P(op={gfx90a_op}, vdst=v[{vdst_num}], src0={mai_src(src0_num, src0_acc)}, src1={mai_src(src1_num, src1_acc)}, src2={src2_val}{', ' + ', '.join(mai_mods) if mai_mods else ''})"
    return f"{fn.lower()}(vdst=v[{vdst_num}], src0={mai_src(src0_num, src0_acc)}, src1={mai_src(src1_num, src1_acc)}, src2={src2_val}{', ' + ', '.join(mai_mods) if mai_mods else ''})"

  # SDWA instructions (CDNA)
  if mn.endswith('_sdwa') and arch == "cdna":
    base_mn = mn[:-5]  # strip _sdwa
    from extra.assembly.amd.autogen.cdna.ins import VOP1Op, VOP2Op, VOPCOp, SDWA
    vop1_op, vop2_op, vopc_op = getattr(VOP1Op, base_mn.upper(), None), getattr(VOP2Op, base_mn.upper(), None), getattr(VOPCOp, base_mn.upper(), None)
    if vop1_op is None and vop2_op is None and vopc_op is None: raise ValueError(f"unknown SDWA instruction: {mn}")
    # Operand layout: vdst, [vcc,] src0[, vsrc1] - carry-out ops have vcc at index 1
    carry_out = {'v_add_co_u32', 'v_sub_co_u32', 'v_subrev_co_u32', 'v_addc_co_u32', 'v_subb_co_u32', 'v_subbrev_co_u32'}
    src0_idx, src1_idx = (2, 3) if base_mn in carry_out else (1, 2)
    # VOPC SDWA: sdst at [0], src0 at [1], src1 at [2]
    if vopc_op is not None:
      _SDWA_SDST = {'vcc': 0, 'vcc_lo': 0, 'flat_scratch': 230, 'flat_scratch_lo': 230}
      sdst_raw = ops[0].strip().lower()
      sdst_enc = _SDWA_SDST.get(sdst_raw, 128 + int(sdst_raw[2:].split(':')[0]) if sdst_raw.startswith('s[') else
                  128 + int(sdst_raw[1:]) if sdst_raw.startswith('s') and sdst_raw[1:].isdigit() else
                  128 + 108 + int(sdst_raw[5:].split(':')[0]) if sdst_raw.startswith('ttmp[') else 0)
      src0_raw, src0_neg, src0_abs, src0_sext = _parse_src_mods(ops[1].strip().lower() if len(ops) > 1 else 'v0')
      src1_raw, src1_neg, src1_abs, src1_sext = _parse_src_mods(ops[2].strip().lower() if len(ops) > 2 else 'v0')
      src0_val, s0 = _parse_sdwa_src(src0_raw)
      vsrc1_val, s1 = _parse_sdwa_src(src1_raw)
      sdwa_kw = [f'vop_op={vsrc1_val}', 'vop2_op=62', f'vdst=RawImm({vopc_op.value})', f'src0=RawImm({src0_val})',
                 f'dst_sel={sdst_enc & 7}', f'dst_u={(sdst_enc >> 3) & 3}', f'clmp={(sdst_enc >> 5) & 1}', f'omod={(sdst_enc >> 6) & 3}',
                 f'src0_sel={sdwa_src0_sel if sdwa_src0_sel is not None else 6}', f'src1_sel={sdwa_src1_sel if sdwa_src1_sel is not None else 6}']
      if src0_sext or sdwa_src0_sext: sdwa_kw.append('src0_sext=1')
      if src0_neg: sdwa_kw.append('src0_neg=1')
      if src0_abs: sdwa_kw.append('src0_abs=1')
      if s0: sdwa_kw.append('s0=1')
      if src1_sext or sdwa_src1_sext: sdwa_kw.append('src1_sext=1')
      if src1_neg: sdwa_kw.append('src1_neg=1')
      if src1_abs: sdwa_kw.append('src1_abs=1')
      if s1: sdwa_kw.append('s1=1')
      return f"SDWA({', '.join(sdwa_kw)})"
    # VOP1/VOP2 SDWA
    src0_raw, src0_neg, src0_abs, src0_sext = _parse_src_mods(ops[src0_idx].strip().lower() if len(ops) > src0_idx else 'v0')
    src0_val, s0 = _parse_sdwa_src(src0_raw)
    vsrc1_val, src1_neg, src1_abs, src1_sext, s1 = 0, False, False, False, 0
    if vop2_op is not None and len(ops) > src1_idx:
      src1_raw, src1_neg, src1_abs, src1_sext = _parse_src_mods(ops[src1_idx].strip().lower())
      vsrc1_val, s1 = _parse_sdwa_src(src1_raw)
    sdwa_kw = [f'vop_op={vop1_op.value if vop1_op else vsrc1_val}', f'vop2_op={63 if vop1_op else vop2_op.value}',
               f'vdst={args[0]}', f'src0=RawImm({src0_val})', f'dst_sel={sdwa_dst_sel if sdwa_dst_sel is not None else 6}',
               f'dst_u={sdwa_dst_unused if sdwa_dst_unused is not None else 2}', f'src0_sel={sdwa_src0_sel if sdwa_src0_sel is not None else 6}']
    if src0_sext or sdwa_src0_sext: sdwa_kw.append('src0_sext=1')
    if src0_neg: sdwa_kw.append('src0_neg=1')
    if src0_abs: sdwa_kw.append('src0_abs=1')
    if s0: sdwa_kw.append('s0=1')
    if vop2_op is not None:
      sdwa_kw.append(f'src1_sel={sdwa_src1_sel if sdwa_src1_sel is not None else 6}')
      if src1_sext or sdwa_src1_sext: sdwa_kw.append('src1_sext=1')
      if src1_neg: sdwa_kw.append('src1_neg=1')
      if src1_abs: sdwa_kw.append('src1_abs=1')
      if s1: sdwa_kw.append('s1=1')
    for k in kw:
      if k.startswith('clmp=') or k.startswith('omod='): sdwa_kw.append(k)
    return f"SDWA({', '.join(sdwa_kw)})"

  # DPP instructions (CDNA)
  if mn.endswith('_dpp') and arch == "cdna" and dpp_ctrl is not None:
    base_mn = mn[:-4]  # strip _dpp
    from extra.assembly.amd.autogen.cdna.ins import VOP1Op, VOP2Op, DPP
    vop1_op, vop2_op = getattr(VOP1Op, base_mn.upper(), None), getattr(VOP2Op, base_mn.upper(), None)
    if vop1_op is None and vop2_op is None: raise ValueError(f"unknown DPP instruction: {mn}")
    carry_out = {'v_add_co_u32', 'v_sub_co_u32', 'v_subrev_co_u32', 'v_addc_co_u32', 'v_subb_co_u32', 'v_subbrev_co_u32'}
    src0_idx, src1_idx = (2, 3) if base_mn in carry_out else (1, 2)
    src0_raw, src0_neg, src0_abs, _ = _parse_src_mods(ops[src0_idx].strip().lower() if len(ops) > src0_idx else 'v0')
    src0_val = int(src0_raw[1:]) if src0_raw.startswith('v') and src0_raw[1:].isdigit() else int(src0_raw.split('[')[1].split(']')[0]) if 'v[' in src0_raw else 0
    vsrc1_val, src1_neg, src1_abs = 0, False, False
    if vop2_op is not None and len(ops) > src1_idx:
      src1_raw, src1_neg, src1_abs, _ = _parse_src_mods(ops[src1_idx].strip().lower())
      vsrc1_val = int(src1_raw[1:]) if src1_raw.startswith('v') and src1_raw[1:].isdigit() else int(src1_raw.split('[')[1].split(']')[0]) if 'v[' in src1_raw else 0
    dpp_kw = [f'vop_op={vop1_op.value if vop1_op else vsrc1_val}', f'vop2_op={63 if vop1_op else vop2_op.value}',
              f'vdst={args[0]}', f'src0=RawImm({src0_val})', f'dpp_ctrl={dpp_ctrl}']
    if dpp_bound_ctrl: dpp_kw.append('bound_ctrl=1')
    if src0_neg: dpp_kw.append('src0_neg=1')
    if src0_abs: dpp_kw.append('src0_abs=1')
    if src1_neg: dpp_kw.append('src1_neg=1')
    if src1_abs: dpp_kw.append('src1_abs=1')
    if dpp_bank_mask_specified or dpp_row_mask_specified:
      dpp_kw.extend([f'bank_mask={dpp_bank_mask if dpp_bank_mask is not None else 0xf}', f'row_mask={dpp_row_mask if dpp_row_mask is not None else 0xf}'])
    return f"DPP({', '.join(dpp_kw)})"

  # VOPD (RDNA3 only)
  if '::' in text:
    xp, yp = text.split('::')
    xps, yps = xp.strip().replace(',', ' ').split(), yp.strip().replace(',', ' ').split()
    xo, yo = [_op2dsl(p, arch) for p in xps[1:]], [_op2dsl(p, arch) for p in yps[1:]]
    vdx, sx0, vsx1 = xo[0], xo[1] if len(xo) > 1 else '0', xo[2] if len(xo) > 2 else 'v[0]'
    vdy, sy0, vsy1 = yo[0], yo[1] if len(yo) > 1 else '0', yo[2] if len(yo) > 2 else 'v[0]'
    lit = xo[3] if 'fmaak' in xps[0].lower() and len(xo) > 3 else yo[3] if 'fmaak' in yps[0].lower() and len(yo) > 3 else None
    if 'fmamk' in xps[0].lower() and len(xo) > 3: lit, vsx1 = xo[2], xo[3]
    elif 'fmamk' in yps[0].lower() and len(yo) > 3: lit, vsy1 = yo[2], yo[3]
    return f"VOPD(VOPDOp.{xps[0].upper()}, VOPDOp.{yps[0].upper()}, vdstx={vdx}, vdsty={vdy}, srcx0={sx0}, vsrcx1={vsx1}, srcy0={sy0}, vsrcy1={vsy1}{f', literal={lit}' if lit else ''})"

  # Special instructions
  if mn == 's_setreg_imm32_b32': raise ValueError(f"unsupported: {mn}")
  # v_readfirstlane_b32 has SGPR dest but encoded in vdst field - use RawImm
  if mn == 'v_readfirstlane_b32' and len(args) >= 2:
    dst = ops[0].strip().lower()
    if dst.startswith('s') and dst[1:].isdigit(): dst_val = int(dst[1:])
    elif dst.startswith('ttmp') and dst[4:].isdigit(): dst_val = 108 + int(dst[4:])
    else:
      sgpr_map = {'vcc_lo': 106, 'vcc_hi': 107, 'm0': 124, 'exec_lo': 126, 'exec_hi': 127,
                  'flat_scratch_lo': 102, 'flat_scratch_hi': 103, 'xnack_mask_lo': 104, 'xnack_mask_hi': 105,
                  'null': 124}  # null register for RDNA3
      dst_val = sgpr_map.get(dst, int(dst) if dst.isdigit() else 0)
    return f"v_readfirstlane_b32_e32(vdst=RawImm({dst_val}), src0={args[1]})"
  if mn in ('s_setpc_b64', 's_rfe_b64'): return f"{mn}(ssrc0={args[0]})"
  if mn in ('s_cbranch_join', 's_set_gpr_idx_idx'): return f"{mn}(ssrc0={args[0]}, sdst=RawImm(0))"  # No destination, only source
  if mn == 's_cbranch_g_fork': return f"{mn}(ssrc0={args[0]}, ssrc1={args[1]}, sdst=RawImm(0))"  # Two sources, no dest
  if mn == 's_set_gpr_idx_on': return f"{mn}(ssrc0={args[0]}, ssrc1=RawImm({int(args[1], 0)}))"  # Mode bits as raw value
  if mn in ('s_sendmsg_rtn_b32', 's_sendmsg_rtn_b64'): return f"{mn}(sdst={args[0]}, ssrc0=RawImm({args[1].strip()}))"
  if mn == 's_version': return f"{mn}(simm16={args[0]})"
  if mn == 's_setreg_b32': return f"{mn}(simm16={args[0]}, sdst={args[1]})"

  # SMEM: s_dcache_discard has swapped operand layout (saddr→sbase, soffset→sdata)
  if arch == "cdna" and mn.startswith('s_dcache_discard'):
    gs = ", glc=1" if glc else ""
    # Syntax: s_dcache_discard saddr, soffset [offset:imm]
    if off_val and len(ops) >= 2:
      # SGPR + immediate offset: soe=1, imm=1, soffset=SGPR, offset=imm
      return f"{mn}(sbase={args[0]}, sdata=RawImm(0), offset={off_val}, soffset={args[1]}, soe=1, imm=1{gs})"
    if len(ops) >= 2 and re.match(r'^-?[0-9]|^-?0x', ops[1].strip().lower()):
      # Immediate offset only: imm=1
      return f"{mn}(sbase={args[0]}, sdata=RawImm(0), offset={args[1]}, soffset=RawImm(0), imm=1{gs})"
    # SGPR offset only: imm=0, offset=SGPR
    return f"{mn}(sbase={args[0]}, sdata=RawImm(0), offset={args[1]}, soffset=RawImm(0){gs})"

  # SMEM: s_atomic_*/s_buffer_atomic_* uses offset field for SGPR (imm=0), not soffset
  if arch == "cdna" and (mn.startswith('s_buffer_atomic') or (mn.startswith('s_atomic') and not mn.startswith('s_atc'))):
    gs = ", glc=1" if glc else ""
    if len(ops) >= 3:
      # Syntax: s_atomic_* sdata, sbase, soffset [offset:imm]
      if off_val:
        # SGPR + immediate offset: soe=1, imm=1
        return f"{mn}(sdata={args[0]}, sbase={args[1]}, offset={off_val}, soffset={args[2]}, soe=1, imm=1{gs})"
      if re.match(r'^-?[0-9]|^-?0x', ops[2].strip().lower()):
        # Immediate offset only: imm=1
        return f"{mn}(sdata={args[0]}, sbase={args[1]}, offset={args[2]}, soffset=RawImm(0), imm=1{gs})"
      # SGPR offset only: imm=0, offset=SGPR
      return f"{mn}(sdata={args[0]}, sbase={args[1]}, offset={args[2]}, soffset=RawImm(0){gs})"

  # SMEM
  if mn in SMEM_OPS or (arch == "cdna" and mn.startswith(('s_load_dword', 's_buffer_load_dword'))):
    gs, ds = ", glc=1" if glc else "", ", dlc=1" if dlc else ""
    if arch == "cdna":
      # CDNA SMEM encoding: imm=1 for immediate, soe=1 for sgpr+offset combo
      if len(ops) >= 3 and re.match(r'^-?[0-9]|^-?0x', ops[2].strip().lower()):
        # Immediate offset only
        return f"{mn}(sdata={args[0]}, sbase={args[1]}, offset={args[2]}, soffset=RawImm(0), imm=1{gs}{ds})"
      if off_val and len(ops) >= 3:
        # SGPR + immediate offset: soe=1, soffset=SGPR, offset=imm
        return f"{mn}(sdata={args[0]}, sbase={args[1]}, offset={off_val}, soffset={args[2]}, soe=1, imm=1{gs}{ds})"
      if len(ops) >= 3:
        # SGPR offset only: offset=SGPR index, soffset=0
        return f"{mn}(sdata={args[0]}, sbase={args[1]}, offset={args[2]}, soffset=RawImm(0){gs}{ds})"
      if len(ops) == 2:
        # No offset specified: imm=1, offset=0
        return f"{mn}(sdata={args[0]}, sbase={args[1]}, offset=0, soffset=RawImm(0), imm=1{gs}{ds})"
    else:
      # RDNA3 encoding
      if len(ops) >= 3 and re.match(r'^-?[0-9]|^-?0x', ops[2].strip().lower()):
        return f"{mn}(sdata={args[0]}, sbase={args[1]}, offset={args[2]}, soffset=RawImm(124){gs}{ds})"
      if off_val and len(ops) >= 3: return f"{mn}(sdata={args[0]}, sbase={args[1]}, offset={off_val}, soffset={args[2]}{gs}{ds})"
      if len(ops) >= 3: return f"{mn}(sdata={args[0]}, sbase={args[1]}, soffset={args[2]}{gs}{ds})"

  # ACC register handling for CDNA: detect a[N] and set acc=1, convert to v[N]
  def _has_acc(args): return any(a.startswith('a[') for a in args if isinstance(a, str))
  def _acc_to_vgpr(a): return 'v' + a[1:] if isinstance(a, str) and a.startswith('a[') else a

  # Buffer (MUBUF/MTBUF) instructions
  if mn.startswith(('buffer_', 'tbuffer_')):
    is_tbuf = mn.startswith('tbuffer_')
    # Parse format value for tbuffer
    fmt_num = None
    # Handle dfmt:N nfmt:N style (CDNA ACC style)
    if dfmt_val is not None or nfmt_val is not None:
      fmt_num = (dfmt_val if dfmt_val is not None else 1) | ((nfmt_val if nfmt_val is not None else 0) << 4)
    elif fmt_val is not None:
      if fmt_val.isdigit(): fmt_num = int(fmt_val)
      else:
        fmt_num = BUF_FMT.get(fmt_val.replace(' ', '')) or _parse_buf_fmt_combo(fmt_val)
        # CDNA-style: BUF_DATA_FORMAT_X or BUF_NUM_FORMAT_X (or comma-separated pair)
        if fmt_num is None and arch == "cdna":
          _dfmt = {'INVALID': 0, '8': 1, '16': 2, '8_8': 3, '32': 4, '16_16': 5, '10_11_11': 6, '11_11_10': 7,
                   '10_10_10_2': 8, '2_10_10_10': 9, '8_8_8_8': 10, '32_32': 11, '16_16_16_16': 12,
                   '32_32_32': 13, '32_32_32_32': 14, 'RESERVED_15': 15}
          _nfmt = {'UNORM': 0, 'SNORM': 1, 'USCALED': 2, 'SSCALED': 3, 'UINT': 4, 'SINT': 5, 'RESERVED_6': 6, 'FLOAT': 7}
          parts = [p.strip() for p in fmt_val.split(',')]
          dfmt, nfmt = 1, 0  # defaults
          for p in parts:
            if p.startswith('BUF_DATA_FORMAT_'): dfmt = _dfmt.get(p[16:], 1)
            elif p.startswith('BUF_NUM_FORMAT_'): nfmt = _nfmt.get(p[15:], 0)
          fmt_num = dfmt | (nfmt << 4)
    # Handle special no-arg buffer ops (with optional sc0/sc1 for CDNA)
    if mn in ('buffer_gl0_inv', 'buffer_gl1_inv'): return f"{mn}()"
    if mn in ('buffer_inv', 'buffer_wbl2'):
      _buf_sc0 = 1 if sc0 else None
      _buf_sc1 = 1 if sc1 else None
      mods = [x for x in ['sc0=1' if _buf_sc0 else '', 'sc1=1' if _buf_sc1 else ''] if x]
      return f"{mn}({', '.join(mods)})"
    # ACC register support for CDNA: detect a[N] registers and set acc=1
    acc_mod = ', acc=1' if arch == 'cdna' and _has_acc(args) else ''
    args = [_acc_to_vgpr(a) for a in args]  # convert a[N] to v[N] for encoding
    # Build modifiers string - CDNA uses sc0/nt for glc/slc; GFX942 uses sc0/sc1/nt directly
    if arch == "cdna":
      _buf_sc0 = 1 if (sc0 or glc) else None
      _buf_sc1 = 1 if sc1 else None
      _buf_nt = 1 if (nt or slc) else None
      buf_mods = "".join([f", offset={off_val}" if off_val else "", ", sc0=1" if _buf_sc0 else "", ", sc1=1" if _buf_sc1 else "",
                          ", nt=1" if _buf_nt else "", ", offen=1" if offen else "", ", idxen=1" if idxen else "", ", lds=1" if lds else "", acc_mod])
    else:
      buf_mods = "".join([f", offset={off_val}" if off_val else "", ", glc=1" if glc else "", ", dlc=1" if dlc else "",
                          ", slc=1" if slc else "", ", tfe=1" if tfe else "", ", offen=1" if offen else "", ", idxen=1" if idxen else ""])
    # Default format for tbuffer is dfmt=1, nfmt=0 (format=8 after encoding as (nfmt<<4)|dfmt becomes just dfmt=1)
    # Actually format is (dfmt | (nfmt << 4)), so dfmt=1, nfmt=0 -> format=1
    if is_tbuf: buf_mods = f", format={fmt_num if fmt_num is not None else 1}" + buf_mods
    # Handle LDS mode: first operand is 'off' meaning no vdata, it goes to LDS
    if len(ops) >= 1 and ops[0].strip().lower() == 'off':
      # LDS mode: buffer_load_format_x off, srsrc, soffset -> no vdata, just vaddr=off
      srsrc_val = args[1] if len(args) > 1 else "s[0:3]"
      soff_val = args[2] if len(args) > 2 else "0"
      return f"{mn}(vdata=v[0], vaddr=v[0], srsrc={srsrc_val}, soffset={soff_val}{buf_mods})"
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

  # FLAT/GLOBAL/SCRATCH load/store/atomic - saddr needs RawImm for off/null
  # CDNA: flat uses saddr=0 for off, global/scratch use saddr=0x7F (127) for off
  # RDNA: uses saddr=124 (NULL)
  # CDNA: uses sc0/sc1 for glc/slc
  def _saddr_off(seg): return 'RawImm(0)' if arch == 'cdna' and seg == 'flat' else ('RawImm(127)' if arch == 'cdna' else 'RawImm(124)')
  def _saddr(a, seg='global'): return _saddr_off(seg) if a in ('OFF', 'NULL') else a
  if arch == "cdna":
    # GFX942 uses sc0/sc1/nt directly; older CDNA uses glc->sc0, slc->nt
    _sc0 = 1 if (sc0 or glc) else None
    _sc1 = 1 if sc1 else None
    _nt = 1 if (nt or slc) else None
    flat_mods = f"{f', offset={off_val}' if off_val else ''}{', sc0=1' if _sc0 else ''}{', sc1=1' if _sc1 else ''}{', nt=1' if _nt else ''}{', lds=1' if lds else ''}"
  else:
    flat_mods = f"{f', offset={off_val}' if off_val else ''}{', glc=1' if glc else ''}{', slc=1' if slc else ''}{', dlc=1' if dlc else ''}{', lds=1' if lds else ''}"
  for pre, flds in [('flat_load','vdst,addr,saddr'), ('global_load','vdst,addr,saddr'), ('scratch_load','vdst,addr,saddr'),
                    ('flat_store','addr,data,saddr'), ('global_store','addr,data,saddr'), ('scratch_store','addr,data,saddr')]:
    if mn.startswith(pre) and len(args) >= 2:
      f0, f1, f2 = flds.split(',')
      seg = pre.split('_')[0]  # 'flat', 'global', or 'scratch'
      # ACC register support for CDNA: detect a[N] registers and set acc=1
      acc_mod = ', acc=1' if arch == 'cdna' and _has_acc(args) else ''
      args = [_acc_to_vgpr(a) for a in args]  # convert a[N] to v[N] for encoding
      # LDS mode: args=[addr, saddr], vdst=0, data goes to LDS
      # Triggered by 'lds' modifier OR '_lds_' in mnemonic (e.g. global_load_lds_dword)
      is_lds_instr = '_lds_' in mn
      if (lds or is_lds_instr) and 'load' in pre:
        addr_off = args[0] == 'OFF'
        addr_val = 'v[0]' if seg == 'scratch' and addr_off else args[0]
        saddr_val = _saddr(args[1], seg) if len(args) >= 2 else _saddr_off(seg)
        # For scratch_load_lds_* with vaddr (not off), lds=1 is needed in encoding
        # For global_load_lds_*, lds is implicit in opcode (no lds bit needed)
        need_lds_bit = is_lds_instr and seg == 'scratch' and not addr_off
        lds_flat_mods = flat_mods + ', lds=1' if need_lds_bit else flat_mods
        return f"{mn}(vdst=v[0], addr={addr_val}, saddr={saddr_val}{lds_flat_mods}{acc_mod})"
      # For scratch, 'off' as vaddr means vaddr=0 (no offset), not null register
      # For load: args=[vdst, addr, saddr], for store: args=[addr, data, saddr]
      # For RDNA3 scratch with 'off' as vaddr, set sve=0 (no VGPR address)
      # For GFX942 scratch: lds=1 when vaddr is used (not 'off')
      if 'store' in pre:
        addr_off = seg == 'scratch' and args[0] == 'OFF'
        addr_val = 'v[0]' if addr_off else args[0]
        sve_mod = ', sve=0' if addr_off and arch == 'rdna3' else ''
        gfx942_lds = ', lds=1' if gfx942 and seg == 'scratch' and not addr_off else ''
        return f"{mn}({f0}={addr_val}, {f1}={args[1]}{f', {f2}={_saddr(args[2], seg)}' if len(args) >= 3 else f', saddr={_saddr_off(seg)}'}{sve_mod}{flat_mods}{gfx942_lds}{acc_mod})"
      else:
        addr_off = seg == 'scratch' and args[1] == 'OFF'
        addr_val = 'v[0]' if addr_off else args[1]
        sve_mod = ', sve=0' if addr_off and arch == 'rdna3' else ''
        gfx942_lds = ', lds=1' if gfx942 and seg == 'scratch' and not addr_off else ''
        return f"{mn}({f0}={args[0]}, {f1}={addr_val}{f', {f2}={_saddr(args[2], seg)}' if len(args) >= 3 else f', saddr={_saddr_off(seg)}'}{sve_mod}{flat_mods}{gfx942_lds}{acc_mod})"
  for pre in ('flat_atomic', 'global_atomic', 'scratch_atomic'):
    if mn.startswith(pre):
      seg = pre.split('_')[0]  # 'flat', 'global', or 'scratch'
      # ACC register support for CDNA: detect a[N] registers and set acc=1
      acc_mod = ', acc=1' if arch == 'cdna' and _has_acc(args) else ''
      args = [_acc_to_vgpr(a) for a in args]  # convert a[N] to v[N] for encoding
      # For atomics with return value: vdst, addr, data, [saddr] - triggered by glc (or sc0 for GFX942)
      has_return = glc or sc0
      if has_return and len(args) >= 3: return f"{mn}(vdst={args[0]}, addr={args[1]}, data={args[2]}{f', saddr={_saddr(args[3], seg)}' if len(args) >= 4 else f', saddr={_saddr_off(seg)}'}{flat_mods}{acc_mod})"
      if len(args) >= 2: return f"{mn}(addr={args[0]}, data={args[1]}{f', saddr={_saddr(args[2], seg)}' if len(args) >= 3 else f', saddr={_saddr_off(seg)}'}{flat_mods}{acc_mod})"

  # DS instructions
  if mn.startswith('ds_'):
    # Handle offset formats: offset:N (combined), offset0:N offset1:N (separate), or none
    if offset0 is not None or offset1 is not None:
      off0, off1 = offset0 or "0", offset1 or "0"
    elif off_val:
      off0, off1 = str(int(off_val, 0) & 0xff), str((int(off_val, 0) >> 8) & 0xff)
    else:
      off0, off1 = "0", "0"
    # ACC register support for CDNA DS instructions
    acc_mod = ', acc=1' if arch == 'cdna' and _has_acc(args) else ''
    args = [_acc_to_vgpr(a) for a in args]  # convert a[N] to v[N] for encoding
    gds_s = ", gds=1" if gds else ""
    off_kw = f", offset0={off0}, offset1={off1}{gds_s}{acc_mod}"
    if mn == 'ds_nop' or mn in ('ds_gws_sema_v', 'ds_gws_sema_p', 'ds_gws_sema_release_all'): return f"{mn}({off_kw.lstrip(', ')})"
    if 'gws_' in mn: return f"{mn}(addr={args[0]}{off_kw})"
    if 'consume' in mn or 'append' in mn: return f"{mn}(vdst={args[0]}{off_kw})"
    if 'gs_reg' in mn: return f"{mn}(vdst={args[0]}, data0={args[1]}{off_kw})"
    if '2addr' in mn:
      if 'load' in mn: return f"{mn}(vdst={args[0]}, addr={args[1]}{off_kw})"
      if 'store' in mn and 'xchg' not in mn: return f"{mn}(addr={args[0]}, data0={args[1]}, data1={args[2]}{off_kw})"
      return f"{mn}(vdst={args[0]}, addr={args[1]}, data0={args[2]}, data1={args[3]}{off_kw})"
    if 'load' in mn or ('read' in mn and 'read2' not in mn): return f"{mn}(vdst={args[0]}{off_kw})" if 'addtid' in mn else f"{mn}(vdst={args[0]}, addr={args[1]}{off_kw})"
    if 'read2' in mn: return f"{mn}(vdst={args[0]}, addr={args[1]}{off_kw})"
    if 'write2' in mn: return f"{mn}(addr={args[0]}, data0={args[1]}, data1={args[2]}{off_kw})"
    if 'xchg2' in mn: return f"{mn}(vdst={args[0]}, addr={args[1]}, data0={args[2]}, data1={args[3]}{off_kw})" if '_rtn' in mn else f"{mn}(addr={args[0]}, data0={args[1]}, data1={args[2]}{off_kw})"
    if 'store' in mn and not _has(mn, 'cmp', 'xchg'):
      return f"{mn}(data0={args[0]}{off_kw})" if 'addtid' in mn else f"{mn}(addr={args[0]}, data0={args[1]}{off_kw})"
    if 'swizzle' in mn or 'ordered_count' in mn: return f"{mn}(vdst={args[0]}, addr={args[1]}{off_kw})"
    if 'permute' in mn: return f"{mn}(vdst={args[0]}, addr={args[1]}, data0={args[2]}{off_kw})"
    if 'bvh' in mn: return f"{mn}(vdst={args[0]}, addr={args[1]}, data0={args[2]}, data1={args[3]}{off_kw})"
    if 'condxchg' in mn: return f"{mn}(vdst={args[0]}, addr={args[1]}, data0={args[2]}{off_kw})"
    if _has(mn, 'cmpst', 'mskor', 'wrap'):
      return f"{mn}(vdst={args[0]}, addr={args[1]}, data0={args[2]}, data1={args[3]}{off_kw})" if '_rtn' in mn else f"{mn}(addr={args[0]}, data0={args[1]}, data1={args[2]}{off_kw})"
    return f"{mn}(vdst={args[0]}, addr={args[1]}, data0={args[2]}{off_kw})" if '_rtn' in mn else f"{mn}(addr={args[0]}, data0={args[1]}{off_kw})"

  # v_fmaak/v_fmamk/v_madak/v_madmk literal handling - need literal= keyword for VOP2
  # fmamk/madmk: dst = src0 * K + vsrc1, fmaak/madak: dst = src0 * vsrc1 + K
  lit_s = ""
  _ak_ops = ('v_fmaak_f32', 'v_fmaak_f16', 'v_madak_f32', 'v_madak_f16')
  _mk_ops = ('v_fmamk_f32', 'v_fmamk_f16', 'v_madmk_f32', 'v_madmk_f16')
  if mn in _ak_ops and len(args) == 4: lit_s, args = f", literal={args[3].strip()}", args[:3]
  elif mn in _mk_ops and len(args) == 4: lit_s, args = f", literal={args[2].strip()}", [args[0], args[1], args[3]]

  # VCC ops cleanup
  vcc_ops = {'v_add_co_ci_u32', 'v_sub_co_ci_u32', 'v_subrev_co_ci_u32'}
  if mn.replace('_e32', '') in vcc_ops and len(args) >= 5: mn, args = mn.replace('_e32', '') + '_e32', [args[0], args[2], args[3]]
  if mn.replace('_e64', '') in vcc_ops and mn.endswith('_e64'): mn = mn.replace('_e64', '')
  if mn.startswith('v_cmp') and not mn.endswith('_e64') and len(args) >= 3 and ops[0].strip().lower() in ('vcc_lo', 'vcc_hi', 'vcc'): args = args[1:]
  # For RDNA3 v_cmpx, destination is implicitly exec (126)
  if 'cmpx' in mn and mn.endswith('_e64') and len(args) == 2 and arch == 'rdna3': args = ['RawImm(126)'] + args
  # v_cmp_*_e64 and v_cmpx_*_e64 have SGPR destination in vdst field - encode as RawImm
  # For CDNA, v_cmpx also writes to SGPR pair (first operand)
  _SGPR_NAMES = {'vcc_lo': 106, 'vcc_hi': 107, 'vcc': 106, 'null': 124, 'm0': 125, 'exec_lo': 126, 'exec_hi': 127}
  if mn.startswith('v_cmp') and mn.endswith('_e64') and len(args) >= 1:
    # For CDNA v_cmpx with 3 operands (sdst, src0, src1), convert sdst to RawImm
    # For RDNA3, v_cmpx only has 2 operands (src0, src1) - already handled above
    is_cmpx = 'cmpx' in mn
    if not is_cmpx or arch == 'cdna':
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
      clean_args.append(_op2dsl(op, arch))
    args = clean_args + args[4:]
    if inline_neg: neg_lo = inline_neg
    if inline_abs: neg_hi = inline_abs

  all_kw = list(kw)
  if lit_s: all_kw.append(lit_s.lstrip(', '))
  if opsel is not None: all_kw.append(f'opsel={opsel}')
  if opsel_hi is not None:
    all_kw.append(f'opsel_hi={opsel_hi & 3}')
    if opsel_hi_count >= 3: all_kw.append(f'opsel_hi2={(opsel_hi >> 2) & 1}')  # only set opsel_hi2 if 3 elements specified
  if neg_lo is not None: all_kw.append(f'neg={neg_lo}')
  if neg_hi is not None: all_kw.append(f'neg_hi={neg_hi}')
  if 'bvh' in mn and 'intersect_ray' in mn: all_kw.extend(['dmask=15', 'unrm=1', 'r128=1'])

  # For CDNA _e64 VOP instructions: use keyword args (VOP3 layout)
  # Pattern: v_xxx_e64 dst, src0[, src1[, src2]] -> VOP3A with promoted opcode
  # VOP1 to VOP3 promotion: VOP3 op = 384 + (VOP1_op - 64) for VOP1_op >= 64, else 256 + VOP1_op
  if fn.endswith('_e64') and fn.startswith('v_') and arch == "cdna":
    fn_base = fn[:-4].upper()  # strip _e64 and uppercase for enum lookup
    from extra.assembly.amd.autogen.cdna.ins import VOP1Op, VOP2Op, VOP3AOp, VOP3BOp
    # Check if this is a VOP3B instruction (has sdst for carry-out)
    vop3b_op = getattr(VOP3BOp, fn_base, None)
    if vop3b_op is not None:
      # VOP3B: v_xxx_e64 vdst, sdst, src0, src1[, src2]
      vop3_args = []
      if len(args) >= 1: vop3_args.append(f'vdst={args[0]}')
      if len(args) >= 2: vop3_args.append(f'sdst={args[1]}')
      if len(args) >= 3: vop3_args.append(f'src0={args[2]}')
      if len(args) >= 4: vop3_args.append(f'src1={args[3]}')
      if len(args) >= 5: vop3_args.append(f'src2={args[4]}')
      a_str = ', '.join(vop3_args + all_kw)
      return f"{fn[:-4]}({a_str})"
    # Check if this is a VOP1 instruction that needs promotion
    vop1_op = getattr(VOP1Op, fn_base, None)
    vop2_op = getattr(VOP2Op, fn_base, None)
    vop3a_op = getattr(VOP3AOp, fn_base, None)
    if vop1_op is not None and vop3a_op is None:
      # VOP1 -> VOP3 promotion: calculate promoted opcode
      promoted_op = 384 + (vop1_op.value - 64) if vop1_op.value >= 64 else 256 + vop1_op.value
      vop3_args = [f'op={promoted_op}']
      if len(args) >= 1: vop3_args.append(f'vdst={args[0]}')
      if len(args) >= 2: vop3_args.append(f'src0={args[1]}')
      if len(args) >= 3: vop3_args.append(f'src1={args[2]}')
      if len(args) >= 4: vop3_args.append(f'src2={args[3]}')
      return f"VOP3A({', '.join(vop3_args + all_kw)})"
    # Otherwise try normal VOP3 lookup
    vop3_args = ['_vop3=True']  # marker for asm() to force VOP3
    if len(args) >= 1: vop3_args.append(f'vdst={args[0]}')
    if len(args) >= 2: vop3_args.append(f'src0={args[1]}')
    if len(args) >= 3: vop3_args.append(f'src1={args[2]}')
    if len(args) >= 4: vop3_args.append(f'src2={args[3]}')
    a_str = ', '.join(vop3_args + all_kw)
    return f"{fn[:-4]}({a_str})"

  # CDNA VOP1 with modifiers: auto-promote to VOP3A/SDWA/DPP
  # Check if this is a VOP1 instruction needing extended encoding (not already _e64/_sdwa/_dpp)
  has_vop3_mods = any(k.startswith(('omod=', 'clmp=')) for k in all_kw)
  has_sdwa_mods = sdwa_src0_sel is not None or sdwa_src1_sel is not None or sdwa_dst_sel is not None
  has_dpp_mods = dpp_ctrl is not None
  if arch == "cdna" and fn.startswith('v_') and not fn.endswith(('_e64', '_sdwa', '_dpp')) and (has_vop3_mods or has_sdwa_mods or has_dpp_mods):
    from extra.assembly.amd.autogen.cdna.ins import VOP1Op, VOP2Op, SDWA, DPP
    fn_upper = fn.upper()
    vop1_op = getattr(VOP1Op, fn_upper, None)
    vop2_op = getattr(VOP2Op, fn_upper, None)
    if vop1_op is not None or vop2_op is not None:
      if has_sdwa_mods:
        # SDWA encoding for VOP1/VOP2 with src0_sel/src1_sel/dst_sel
        sdwa_kw = []
        src0_orig = ops[1].strip().lower() if len(ops) > 1 else ''
        src0_is_sgpr = src0_orig.startswith('s') and not src0_orig.startswith('src')
        src0_is_literal = src0_orig.isdigit() or (len(src0_orig) > 2 and src0_orig[:2] == '0x')
        if vop1_op is not None:
          sdwa_kw.append(f'vop_op={vop1_op.value}')
          sdwa_kw.append('vop2_op=63')  # 0x3f indicates VOP1 mode
          sdwa_kw.append(f'vdst={args[0]}' if args else 'vdst=v[0]')
          sdwa_kw.append(f'src0={args[1]}' if len(args) > 1 else 'src0=v[0]')
        else:
          sdwa_kw.append(f'vop_op={args[1] if len(args) > 1 else "v[0]"}')
          sdwa_kw.append(f'vop2_op={vop2_op.value}')
          sdwa_kw.append(f'vdst={args[0]}' if args else 'vdst=v[0]')
          sdwa_kw.append(f'src0={args[2] if len(args) > 2 else "v[0]"}')
        sdwa_kw.append(f'dst_sel={sdwa_dst_sel if sdwa_dst_sel is not None else 6}')
        sdwa_kw.append('dst_u=0')
        sdwa_kw.append(f'src0_sel={sdwa_src0_sel if sdwa_src0_sel is not None else 6}')
        sdwa_kw.append('src0_sext=0')
        sdwa_kw.append('src0_neg=0')
        sdwa_kw.append('src0_abs=0')
        sdwa_kw.append(f's0={1 if src0_is_sgpr or src0_is_literal else 0}')  # s0=1 for SGPR/literal
        sdwa_kw.append(f'src1_sel={sdwa_src1_sel if sdwa_src1_sel is not None else 0}')  # 0 for VOP1
        sdwa_kw.append('src1_sext=0')
        sdwa_kw.append('src1_neg=0')
        sdwa_kw.append('src1_abs=0')
        sdwa_kw.append('s1=0')
        # Add clamp and omod if present
        if any(k == 'clmp=1' for k in all_kw): sdwa_kw.append('clmp=1')
        for k in all_kw:
          if k.startswith('omod='): sdwa_kw.append(k); break
        return f"SDWA({', '.join(sdwa_kw)})"
      elif has_dpp_mods:
        # DPP encoding for VOP1/VOP2 with quad_perm/row_shl/etc.
        dpp_kw = []
        if vop1_op is not None:
          dpp_kw.append(f'vop_op={vop1_op.value}')
          dpp_kw.append('vop2_op=63')  # 0x3f indicates VOP1 mode
          dpp_kw.append(f'vdst={args[0]}' if args else 'vdst=v[0]')
          dpp_kw.append(f'src0={args[1]}' if len(args) > 1 else 'src0=v[0]')
        else:
          # VOP2 DPP: vop_op is vsrc1 (second source), src0 is DPP source (first source)
          dpp_kw.append(f'vop_op={args[2] if len(args) > 2 else "v[0]"}')
          dpp_kw.append(f'vop2_op={vop2_op.value}')
          dpp_kw.append(f'vdst={args[0]}' if args else 'vdst=v[0]')
          dpp_kw.append(f'src0={args[1] if len(args) > 1 else "v[0]"}')
        dpp_kw.append(f'dpp_ctrl={dpp_ctrl}')
        dpp_kw.append(f'row_mask={dpp_row_mask if dpp_row_mask is not None else 15}')
        dpp_kw.append(f'bank_mask={dpp_bank_mask if dpp_bank_mask is not None else 15}')
        dpp_kw.append(f'bound_ctrl={dpp_bound_ctrl if dpp_bound_ctrl is not None else 0}')
        return f"DPP({', '.join(dpp_kw)})"
      elif has_vop3_mods and vop1_op is not None:
        # VOP3A encoding for VOP1 with clamp/omod
        from extra.assembly.amd.autogen.cdna.ins import VOP3AOp
        # Calculate promoted opcode: VOP3 op = 320 + VOP1_op
        promoted_op = 320 + vop1_op.value
        vop3_kw = [f'op={promoted_op}']
        vop3_kw.append(f'vdst={args[0]}' if args else 'vdst=v[0]')
        vop3_kw.append(f'src0={args[1]}' if len(args) > 1 else 'src0=v[0]')
        vop3_kw.append('src1=RawImm(0)')
        vop3_kw.append('src2=RawImm(0)')
        vop3_kw.extend(all_kw)
        return f"VOP3A({', '.join(vop3_kw)})"

  # GFX942-specific VOP3A opcode adjustments: some instructions need +64 offset
  _GFX942_VOP3A_OFFSET64 = {'V_CVT_PK_BF8_F32', 'V_CVT_PK_FP8_F32', 'V_CVT_SR_BF8_F32', 'V_CVT_SR_FP8_F32', 'V_LSHL_ADD_U64'}
  if gfx942 and fn.upper() in _GFX942_VOP3A_OFFSET64:
    from extra.assembly.amd.autogen.cdna.ins import VOP3AOp
    base_op = getattr(VOP3AOp, fn.upper(), None)
    if base_op is not None:
      vop3_kw = [f'op={base_op + 64}']
      vop3_kw.append(f'vdst={args[0]}' if args else 'vdst=v[0]')
      vop3_kw.append(f'src0={args[1]}' if len(args) > 1 else 'src0=v[0]')
      vop3_kw.append(f'src1={args[2]}' if len(args) > 2 else 'src1=RawImm(0)')
      vop3_kw.append(f'src2={args[3]}' if len(args) > 3 else 'src2=RawImm(0)')
      vop3_kw.extend(all_kw)
      return f"VOP3A({', '.join(vop3_kw)})"

  a_str, kw_str = ', '.join(args), ', '.join(all_kw)
  return f"{fn}({a_str}, {kw_str})" if kw_str and a_str else f"{fn}({kw_str})" if kw_str else f"{fn}({a_str})"

# CDNA VOP3A opcodes that need +64 offset on gfx90a/gfx942 (autogen has wrong values)
_CDNA_VOP3A_OPCODE_FIX = {'v_mul_legacy_f32': 64}  # gfx90a opcode is 0x2a1, autogen has 0x261

def _fix_cdna_opcode(inst, mnemonic: str, is_gfx90a_or_942: bool):
  """Fix opcode for CDNA instructions where autogen has wrong values (gfx90a/gfx942 only)."""
  if not is_gfx90a_or_942: return inst
  base = mnemonic.removesuffix('_e64').removesuffix('_e32')
  if base in _CDNA_VOP3A_OPCODE_FIX and hasattr(inst, '_values') and 'op' in inst._values:
    op = inst._values['op']
    offset = _CDNA_VOP3A_OPCODE_FIX[base]
    inst._values['op'] = (op.value + offset) if hasattr(op, 'value') else (op + offset)
  return inst

def asm(text: str, arch: str = "rdna3") -> Inst:
  # Normalize arch: gfx90a and gfx942 are CDNA variants
  is_gfx942 = arch == "gfx942"
  is_gfx90a = arch == "gfx90a"
  is_gfx90a_or_942 = is_gfx942 or is_gfx90a
  if is_gfx90a_or_942: arch = "cdna"
  mnemonic = text.split()[0].lower()
  dsl = get_dsl(text, arch, gfx942=is_gfx942)
  if arch == "cdna":
    from extra.assembly.amd.autogen.cdna import ins as cdna_ins
    ns = {n: getattr(cdna_ins, n) for n in dir(cdna_ins) if not n.startswith('_')}
    # CDNA special registers: m0=124, flat_scratch=102-103, xnack_mask=104-105, no NULL (use m0 for off)
    # HWREG symbolic names for s_getreg_b32/s_setreg_b32
    _hwreg_names = {k: v for k, v in _HWREG_GFX942.items()}
    _hwreg_names.update({v: k for k, v in HWREG.items()})  # standard names: id -> name
    _hwreg_ids = {v: k for k, v in _hwreg_names.items()}  # reverse: name -> id
    ns.update({'s': s, 'v': v, 'ttmp': ttmp, 'abs': abs, 'RawImm': RawImm, 'SrcMod': SrcMod, 'VGPR': VGPR, 'SGPR': SGPR, 'TTMP': TTMP,
               'VCC_LO': RawImm(106), 'VCC_HI': RawImm(107), 'VCC': RawImm(106), 'EXEC_LO': RawImm(126), 'EXEC_HI': RawImm(127), 'EXEC': RawImm(126),
               'SCC': RawImm(253), 'M0': RawImm(124), 'NULL': RawImm(124), 'OFF': RawImm(124), 'hwreg': hwreg,
               'HW_REG_XCC_ID': 20, 'HW_REG_SQ_PERF_SNAPSHOT_DATA': 21, 'HW_REG_SQ_PERF_SNAPSHOT_DATA1': 22,
               'HW_REG_SQ_PERF_SNAPSHOT_PC_LO': 23, 'HW_REG_SQ_PERF_SNAPSHOT_PC_HI': 24,
               'FLAT_SCRATCH_LO': RawImm(102), 'FLAT_SCRATCH_HI': RawImm(103), 'FLAT_SCRATCH': RawImm(102),
               'XNACK_MASK_LO': RawImm(104), 'XNACK_MASK_HI': RawImm(105), 'XNACK_MASK': RawImm(104),
               'SRC_VCCZ': RawImm(251), 'SRC_EXECZ': RawImm(252), 'SRC_SCC': RawImm(253), 'SRC_LDS_DIRECT': RawImm(254)})
  else:
    ns = {n: getattr(ins, n) for n in dir(ins) if not n.startswith('_')}
    ns.update({'s': s, 'v': v, 'ttmp': ttmp, 'abs': abs, 'RawImm': RawImm, 'SrcMod': SrcMod, 'VGPR': VGPR, 'SGPR': SGPR, 'TTMP': TTMP,
               'VCC_LO': VCC_LO, 'VCC_HI': VCC_HI, 'VCC': VCC, 'EXEC_LO': EXEC_LO, 'EXEC_HI': EXEC_HI, 'EXEC': EXEC, 'SCC': SCC, 'M0': M0, 'NULL': NULL, 'OFF': OFF})
  fix = (lambda inst: _fix_cdna_opcode(inst, mnemonic, is_gfx90a_or_942)) if arch == "cdna" else (lambda inst: inst)
  try:
    # Generic CDNA (not gfx90a/gfx942): v_mul_legacy_f32 uses VOP2 opcode 4, _e64 uses VOP3A opcode 0x104
    if arch == "cdna" and not is_gfx90a_or_942 and mnemonic.startswith('v_mul_legacy_f32'):
      from extra.assembly.amd.autogen.cdna.ins import VOP2, VOP3A
      args = _parse_ops(text[len(mnemonic):])
      dsl_args = [_op2dsl(a, arch) for a in args]
      if mnemonic == 'v_mul_legacy_f32_e64':
        return eval(f"VOP3A(op=0x104, vdst={dsl_args[0]}, src0={dsl_args[1]}, src1={dsl_args[2]}, src2=RawImm(0))", ns)
      return eval(f"VOP2(op=4, vdst={dsl_args[0]}, src0={dsl_args[1]}, vsrc1={dsl_args[2]})", ns)
    # For CDNA, prefer _e32 variants for VOP1/VOP2 when available (bare names map to VOP3)
    # But skip if:
    #   - already has _e64 suffix (explicit VOP3 request)
    #   - uses keyword args like vdst=/src0= (VOP3 layout from _e64 instructions)
    #   - has _vop3=True marker (from _e64 instructions without operands)
    uses_vop3_kwargs = 'vdst=' in dsl or 'src0=' in dsl or '_vop3=True' in dsl
    if arch == "cdna" and (m := re.match(r'^(v_\w+)(\(.*\))$', dsl)) and not m.group(1).endswith('_e64') and not uses_vop3_kwargs:
      fn_name, args_str = m.group(1), m.group(2)
      e32_name = f"{fn_name}_e32"
      # VOP2 carry ops: v_add_co_u32(vdst, vcc, src0, vsrc1) -> v_add_co_u32_e32(vdst, src0, vsrc1)
      # Strip VCC argument (2nd arg) for VOP2 carry operations when using _e32
      if e32_name in ns and fn_name in _VOP2_CARRY_OUT | _VOP2_CARRY_INOUT:
        args_match = re.match(r'\(([^,]+),\s*[^,]+,\s*(.+)\)$', args_str)
        if args_match: args_str = f"({args_match.group(1)}, {args_match.group(2)})"
      if e32_name in ns: return fix(eval(f"{e32_name}{args_str}", ns))
    # For CDNA, _e64 suffix maps to base name (VOP3)
    if arch == "cdna" and (m := re.match(r'^(v_\w+)_e64(\(.*\))$', dsl)):
      base_name = m.group(1)
      if base_name in ns: return fix(eval(f"{base_name}{m.group(2)}", ns))
    # Strip _vop3=True marker before eval
    eval_dsl = dsl.replace('_vop3=True, ', '').replace('_vop3=True', '')
    return fix(eval(eval_dsl, ns))
  except NameError:
    # For CDNA, try stripping _e64 to get VOP3 base name
    if arch == "cdna" and (m := re.match(r'^(v_\w+)_e64(\(.*\))$', dsl)):
      return fix(eval(f"{m.group(1)}{m.group(2)}", ns))
    # Don't try _e32 if already _e64
    if (m := re.match(r'^(v_\w+)(\(.*\))$', dsl)) and not m.group(1).endswith('_e64'):
      return fix(eval(f"{m.group(1)}_e32{m.group(2)}", ns))
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
