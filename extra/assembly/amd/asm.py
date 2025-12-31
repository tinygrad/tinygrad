# RDNA3 assembler and disassembler
from __future__ import annotations
import re
from extra.assembly.amd.dsl import Inst, RawImm, Reg, SrcMod, SGPR, VGPR, TTMP, s, v, ttmp, _RegFactory, SRC_FIELDS, unwrap
from extra.assembly.amd.dsl import VCC_LO, VCC_HI, VCC, EXEC_LO, EXEC_HI, EXEC, SCC, M0, NULL, OFF
from extra.assembly.amd.dsl import SPECIAL_GPRS, SPECIAL_PAIRS, FLOAT_DEC, FLOAT_ENC, decode_src
from extra.assembly.amd.autogen.rdna3 import VOP1, VOP2, VOP3, VOP3SD, VOP3P, VOPC, VOPD, VINTERP, SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, DS, FLAT, MUBUF, MTBUF, MIMG, EXP
from extra.assembly.amd.autogen.rdna3 import VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp, VOPDOp, VINTERPOp
from extra.assembly.amd.autogen.rdna3 import SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, SMEMOp, DSOp, FLATOp, MUBUFOp, MTBUFOp, MIMGOp

# VOP3SD opcodes that share VOP3 encoding
VOP3SD_OPS = {288, 289, 290, 764, 765, 766, 767, 768, 769, 770}

def _matches_encoding(word: int, cls: type[Inst]) -> bool:
  """Check if word matches the encoding pattern of an instruction class."""
  if cls._encoding is None: return False
  bf, val = cls._encoding
  return ((word >> bf.lo) & bf.mask()) == val

# Order matters: more specific encodings first, VOP2 last (it's a catch-all for bit31=0)
_FORMATS_64 = [VOPD, VOP3P, VINTERP, VOP3, DS, FLAT, MUBUF, MTBUF, MIMG, SMEM, EXP]
_FORMATS_32 = [SOP1, SOPC, SOPP, SOPK, VOPC, VOP1, SOP2, VOP2]  # SOP2/VOP2 are catch-alls

def detect_format(data: bytes) -> type[Inst]:
  """Detect instruction format from machine code bytes."""
  assert len(data) >= 4, f"need at least 4 bytes, got {len(data)}"
  word = int.from_bytes(data[:4], 'little')
  # Check 64-bit formats first (bits[31:30] == 0b11)
  if (word >> 30) == 0b11:
    for cls in _FORMATS_64:
      if _matches_encoding(word, cls):
        return VOP3SD if cls is VOP3 and ((word >> 16) & 0x3ff) in VOP3SD_OPS else cls
    raise ValueError(f"unknown 64-bit format word={word:#010x}")
  # 32-bit formats
  for cls in _FORMATS_32:
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
MSG = {128: 'MSG_RTN_GET_DOORBELL', 129: 'MSG_RTN_GET_DDID', 130: 'MSG_RTN_GET_TMA',
       131: 'MSG_RTN_GET_REALTIME', 132: 'MSG_RTN_SAVE_WAVE', 133: 'MSG_RTN_GET_TBA'}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _reg(p: str, b: int, n: int = 1) -> str: return f"{p}{b}" if n == 1 else f"{p}[{b}:{b+n-1}]"
def _sreg(b: int, n: int = 1) -> str: return _reg("s", b, n)
def _vreg(b: int, n: int = 1) -> str: return _reg("v", b, n)
def _ttmp(b: int, n: int = 1) -> str: return _reg("ttmp", b - 108, n) if 108 <= b <= 123 else None
def _sreg_or_ttmp(b: int, n: int = 1) -> str: return _ttmp(b, n) or _sreg(b, n)

def _fmt_sdst(v: int, n: int = 1, *, is_cdna: bool = False) -> str:
  if v == 124: return "m0" if is_cdna else "null"  # CDNA: M0=124, RDNA: null=124
  if t := _ttmp(v, n): return t
  if n > 1: return SPECIAL_PAIRS.get(v) or _sreg(v, n)
  return SPECIAL_GPRS.get(v, f"s{v}")

def _fmt_src(v: int, n: int = 1) -> str:
  if n == 1: return decode_src(v)
  if v >= 256: return _vreg(v - 256, n)
  if v <= 105: return _sreg(v, n)
  if n == 2 and v in SPECIAL_PAIRS: return SPECIAL_PAIRS[v]
  if t := _ttmp(v, n): return t
  return decode_src(v)

def _fmt_v16(v: int, base: int = 256, hi_thresh: int = 384) -> str:
  return f"v{(v - base) & 0x7f}.{'h' if v >= hi_thresh else 'l'}"

def waitcnt(vmcnt: int = 0x3f, expcnt: int = 0x7, lgkmcnt: int = 0x3f) -> int:
  return (expcnt & 0x7) | ((lgkmcnt & 0x3f) << 4) | ((vmcnt & 0x3f) << 10)

def _has(op: str, *subs) -> bool: return any(s in op for s in subs)
def _is16(op: str) -> bool: return _has(op, 'f16', 'i16', 'u16', 'b16') and not _has(op, '_f32', '_i32')
def _is64(op: str) -> bool: return _has(op, 'f64', 'i64', 'u64', 'b64')
def _omod(v: int) -> str: return {1: " mul:2", 2: " mul:4", 3: " div:2"}.get(v, "")
def _src16(inst, v: int) -> str: return _fmt_v16(v) if v >= 256 else inst.lit(v)  # format 16-bit src: vgpr.h/l or literal
def _mods(*pairs) -> str: return " ".join(m for c, m in pairs if c)
def _fmt_bits(label: str, val: int, count: int) -> str: return f"{label}:[{','.join(str((val >> i) & 1) for i in range(count))}]"

def _vop3_src(inst, v: int, neg: int, abs_: int, hi: int, n: int, f16: bool, any_hi: bool) -> str:
  """Format VOP3 source operand with modifiers."""
  if n > 1: s = _fmt_src(v, n)
  elif f16 and v >= 256: s = f"v{v - 256}.h" if hi else (f"v{v - 256}.l" if any_hi else inst.lit(v))
  else: s = inst.lit(v)
  if abs_: s = f"|{s}|"
  return f"-{s}" if neg else s

def _opsel_str(opsel: int, n: int, need: bool, is16_d: bool) -> str:
  """Format op_sel modifier string."""
  if not need: return ""
  if is16_d and (opsel & 8): return f" op_sel:[1,1,1{',1' if n == 3 else ''}]"
  if n == 3: return f" op_sel:[{opsel & 1},{(opsel >> 1) & 1},{(opsel >> 2) & 1},{(opsel >> 3) & 1}]"
  return f" op_sel:[{opsel & 1},{(opsel >> 1) & 1},{(opsel >> 2) & 1}]"

# ═══════════════════════════════════════════════════════════════════════════════
# DISASSEMBLER
# ═══════════════════════════════════════════════════════════════════════════════

_VOP1_F64 = {VOP1Op.V_CEIL_F64, VOP1Op.V_FLOOR_F64, VOP1Op.V_FRACT_F64, VOP1Op.V_FREXP_MANT_F64, VOP1Op.V_RCP_F64, VOP1Op.V_RNDNE_F64, VOP1Op.V_RSQ_F64, VOP1Op.V_SQRT_F64, VOP1Op.V_TRUNC_F64}

def _disasm_vop1(inst: VOP1) -> str:
  stored_op = inst._values.get('op')
  op = stored_op if stored_op is not None and hasattr(stored_op, 'name') else VOP1Op(inst.op)
  name = op.name.lower()
  if name in ('v_nop', 'v_pipeflush'): return name
  if name == 'v_readfirstlane_b32': return f"v_readfirstlane_b32 {decode_src(inst.vdst)}, v{inst.src0 - 256 if inst.src0 >= 256 else inst.src0}"
  _f64_names = {'v_ceil_f64', 'v_floor_f64', 'v_fract_f64', 'v_frexp_mant_f64', 'v_rcp_f64', 'v_rndne_f64', 'v_rsq_f64', 'v_sqrt_f64', 'v_trunc_f64'}
  parts, is_f64_d = name.split('_'), name in _f64_names or name in ('v_cvt_f64_f32', 'v_cvt_f64_i32', 'v_cvt_f64_u32')
  is_f64_s = name in _f64_names or name in ('v_cvt_f32_f64', 'v_cvt_i32_f64', 'v_cvt_u32_f64', 'v_frexp_exp_i32_f64')
  is_16d = any(p in ('f16','i16','u16','b16') for p in parts[-2:-1]) or (len(parts) >= 2 and parts[-1] in ('f16','i16','u16','b16') and 'cvt' not in name)
  is_16s = parts[-1] in ('f16','i16','u16','b16') and 'sat_pk' not in name
  dst = _vreg(inst.vdst, 2) if is_f64_d else _fmt_v16(inst.vdst, 0, 128) if is_16d else f"v{inst.vdst}"
  src = _fmt_src(inst.src0, 2) if is_f64_s else _src16(inst, inst.src0) if is_16s else inst.lit(inst.src0)
  return f"{name}_e32 {dst}, {src}"

def _disasm_vop2(inst: VOP2) -> str:
  stored_op = inst._values.get('op')
  op = stored_op if stored_op is not None and hasattr(stored_op, 'name') else VOP2Op(inst.op)
  name = op.name.lower()
  suf, is16 = "" if name == 'v_dot2acc_f32_f16' else "_e32", _is16(name) and 'pk_' not in name
  # fmaak: dst = src0 * vsrc1 + K, fmamk: dst = src0 * K + vsrc1
  if name in ('v_fmaak_f32', 'v_fmaak_f16'): return f"{name}{suf} v{inst.vdst}, {inst.lit(inst.src0)}, v{inst.vsrc1}, 0x{inst._literal:x}"
  if name in ('v_fmamk_f32', 'v_fmamk_f16'): return f"{name}{suf} v{inst.vdst}, {inst.lit(inst.src0)}, 0x{inst._literal:x}, v{inst.vsrc1}"
  if is16: return f"{name}{suf} {_fmt_v16(inst.vdst, 0, 128)}, {_src16(inst, inst.src0)}, {_fmt_v16(inst.vsrc1, 0, 128)}"
  return f"{name}{suf} v{inst.vdst}, {inst.lit(inst.src0)}, v{inst.vsrc1}" + (", vcc_lo" if name == 'v_cndmask_b32' else "")

VOPC_CLASS_NAMES = {'v_cmp_class_f16', 'v_cmp_class_f32', 'v_cmp_class_f64',
                    'v_cmpx_class_f16', 'v_cmpx_class_f32', 'v_cmpx_class_f64'}

def _disasm_vopc(inst: VOPC) -> str:
  stored_op = inst._values.get('op')
  op = stored_op if stored_op is not None and hasattr(stored_op, 'name') else VOPCOp(inst.op)
  name = op.name.lower()
  is64, is16 = _is64(name), _is16(name)
  s0 = _fmt_src(inst.src0, 2) if is64 else _src16(inst, inst.src0) if is16 else inst.lit(inst.src0)
  s1 = _vreg(inst.vsrc1, 2) if is64 and name not in VOPC_CLASS_NAMES else _fmt_v16(inst.vsrc1, 0, 128) if is16 else f"v{inst.vsrc1}"
  return f"{name}_e32 {s0}, {s1}" if op.value >= 128 else f"{name}_e32 vcc_lo, {s0}, {s1}"

NO_ARG_SOPP = {SOPPOp.S_ENDPGM, SOPPOp.S_BARRIER, SOPPOp.S_WAKEUP, SOPPOp.S_ICACHE_INV,
               SOPPOp.S_WAIT_IDLE, SOPPOp.S_ENDPGM_SAVED, SOPPOp.S_CODE_END, SOPPOp.S_ENDPGM_ORDERED_PS_DONE}

def _disasm_sopp(inst: SOPP) -> str:
  stored_op = inst._values.get('op')
  name = stored_op.name.lower() if hasattr(stored_op, 'name') else SOPPOp(inst.op).name.lower()
  if name in {x.name.lower() for x in NO_ARG_SOPP}: return name
  if name == 's_waitcnt':
    # CDNA uses different bit layout, just output raw hex value for safety
    is_cdna = hasattr(stored_op, 'name') and 'cdna' in type(stored_op).__module__
    if is_cdna: return f"s_waitcnt 0x{inst.simm16:x}"
    vm, exp, lgkm = (inst.simm16 >> 10) & 0x3f, inst.simm16 & 0xf, (inst.simm16 >> 4) & 0x3f
    p = [f"vmcnt({vm})" if vm != 0x3f else "", f"expcnt({exp})" if exp != 7 else "", f"lgkmcnt({lgkm})" if lgkm != 0x3f else ""]
    return f"s_waitcnt {' '.join(x for x in p if x) or '0'}"
  if name == 's_delay_alu':
    deps, skips = ['VALU_DEP_1','VALU_DEP_2','VALU_DEP_3','VALU_DEP_4','TRANS32_DEP_1','TRANS32_DEP_2','TRANS32_DEP_3','FMA_ACCUM_CYCLE_1','SALU_CYCLE_1','SALU_CYCLE_2','SALU_CYCLE_3'], ['SAME','NEXT','SKIP_1','SKIP_2','SKIP_3','SKIP_4']
    id0, skip, id1 = inst.simm16 & 0xf, (inst.simm16 >> 4) & 0x7, (inst.simm16 >> 7) & 0xf
    dep = lambda v: deps[v-1] if 0 < v <= len(deps) else str(v)
    p = [f"instid0({dep(id0)})" if id0 else "", f"instskip({skips[skip]})" if skip else "", f"instid1({dep(id1)})" if id1 else ""]
    return f"s_delay_alu {' | '.join(x for x in p if x) or '0'}"
  return f"{name} {inst.simm16}" if name.startswith(('s_cbranch', 's_branch')) else f"{name} 0x{inst.simm16:x}"

def _disasm_smem(inst: SMEM) -> str:
  # Use stored enum if available (preserves CDNA names), otherwise fall back to RDNA3 SMEMOp
  stored_op = inst._values.get('op')
  name = stored_op.name.lower() if hasattr(stored_op, 'name') else SMEMOp(inst.op).name.lower()
  if name in ('s_gl1_inv', 's_dcache_inv'): return name
  # CDNA uses imm bit to indicate immediate offset mode; RDNA3 uses soffset==124 (null)
  is_imm = inst._values.get('imm', 0) or inst.soffset == 124
  off_s = f"{decode_src(inst.soffset)} offset:0x{inst.offset:x}" if inst.offset and not is_imm else f"0x{inst.offset:x}" if is_imm else decode_src(inst.soffset)
  sbase_idx, sbase_count = inst.sbase * 2, 4 if (8 <= inst.op <= 12 or name == 's_atc_probe_buffer') else 2
  sbase_str = _fmt_src(sbase_idx, sbase_count) if sbase_count == 2 else _sreg(sbase_idx, sbase_count) if sbase_idx <= 105 else _reg("ttmp", sbase_idx - 108, sbase_count)
  if name in ('s_atc_probe', 's_atc_probe_buffer'): return f"{name} {inst.sdata}, {sbase_str}, {off_s}"
  width = {0:1, 1:2, 2:4, 3:8, 4:16, 8:1, 9:2, 10:4, 11:8, 12:16}.get(inst.op, 1)
  dlc = inst._values.get('dlc', 0)  # CDNA may not have dlc field
  return f"{name} {_fmt_sdst(inst.sdata, width)}, {sbase_str}, {off_s}" + _mods((inst.glc, " glc"), (dlc, " dlc"))

def _disasm_flat(inst: FLAT) -> str:
  name = FLATOp(inst.op).name.lower()
  seg = ['flat', 'scratch', 'global'][inst.seg] if inst.seg < 3 else 'flat'
  instr = f"{seg}_{name.split('_', 1)[1] if '_' in name else name}"
  off_val = inst.offset if seg == 'flat' else (inst.offset if inst.offset < 4096 else inst.offset - 8192)
  suffix = name.split('_')[-1]
  w = {'b32':1,'b64':2,'b96':3,'b128':4,'u8':1,'i8':1,'u16':1,'i16':1,'u32':1,'i32':1,'u64':2,'i64':2,'f32':1,'f64':2}.get(suffix, 1)
  if 'cmpswap' in name: w *= 2
  if name.endswith('_x2') or 'x2' in suffix: w = max(w, 2)
  mods = f"{f' offset:{off_val}' if off_val else ''}{' glc' if inst.glc else ''}{' slc' if inst.slc else ''}{' dlc' if inst.dlc else ''}"
  # saddr
  if seg == 'flat' or inst.saddr == 0x7F: saddr_s = ""
  elif inst.saddr == 124: saddr_s = ", off"
  elif seg == 'scratch': saddr_s = f", {decode_src(inst.saddr)}"
  elif inst.saddr in SPECIAL_PAIRS: saddr_s = f", {SPECIAL_PAIRS[inst.saddr]}"
  elif t := _ttmp(inst.saddr, 2): saddr_s = f", {t}"
  else: saddr_s = f", {_sreg(inst.saddr, 2) if inst.saddr < 106 else decode_src(inst.saddr)}"
  # addtid: no addr
  if 'addtid' in name: return f"{instr} v{inst.data if 'store' in name else inst.vdst}{saddr_s}{mods}"
  # addr width
  addr_s = "off" if not inst.sve and seg == 'scratch' else _vreg(inst.addr, 1 if seg == 'scratch' or (inst.saddr not in (0x7F, 124)) else 2)
  data_s, vdst_s = _vreg(inst.data, w), _vreg(inst.vdst, w // 2 if 'cmpswap' in name else w)
  if 'atomic' in name:
    return f"{instr} {vdst_s}, {addr_s}, {data_s}{saddr_s if seg != 'flat' else ''}{mods}" if inst.glc else f"{instr} {addr_s}, {data_s}{saddr_s if seg != 'flat' else ''}{mods}"
  if 'store' in name: return f"{instr} {addr_s}, {data_s}{saddr_s}{mods}"
  return f"{instr} {_vreg(inst.vdst, w)}, {addr_s}{saddr_s}{mods}"

def _disasm_ds(inst: DS) -> str:
  op, name = DSOp(inst.op), DSOp(inst.op).name.lower()
  gds = " gds" if inst.gds else ""
  off = f" offset:{inst.offset0 | (inst.offset1 << 8)}" if inst.offset0 or inst.offset1 else ""
  off2 = f" offset0:{inst.offset0} offset1:{inst.offset1}" if inst.offset0 or inst.offset1 else ""
  w = 4 if '128' in name else 3 if '96' in name else 2 if (name.endswith('64') or 'gs_reg' in name) else 1
  d0, d1, dst, addr = _vreg(inst.data0, w), _vreg(inst.data1, w), _vreg(inst.vdst, w), f"v{inst.addr}"

  if op == DSOp.DS_NOP: return name
  if op == DSOp.DS_BVH_STACK_RTN_B32: return f"{name} v{inst.vdst}, {addr}, v{inst.data0}, {_vreg(inst.data1, 4)}{off}{gds}"
  if 'gws_sema' in name and op != DSOp.DS_GWS_SEMA_BR: return f"{name}{off}{gds}"
  if 'gws_' in name: return f"{name} {addr}{off}{gds}"
  if op in (DSOp.DS_CONSUME, DSOp.DS_APPEND): return f"{name} v{inst.vdst}{off}{gds}"
  if 'gs_reg' in name: return f"{name} {_vreg(inst.vdst, 2)}, v{inst.data0}{off}{gds}"
  if '2addr' in name:
    if 'load' in name: return f"{name} {_vreg(inst.vdst, w*2)}, {addr}{off2}{gds}"
    if 'store' in name and 'xchg' not in name: return f"{name} {addr}, {d0}, {d1}{off2}{gds}"
    return f"{name} {_vreg(inst.vdst, w*2)}, {addr}, {d0}, {d1}{off2}{gds}"
  if 'load' in name: return f"{name} v{inst.vdst}{off}{gds}" if 'addtid' in name else f"{name} {dst}, {addr}{off}{gds}"
  if 'store' in name and not _has(name, 'cmp', 'xchg'):
    return f"{name} v{inst.data0}{off}{gds}" if 'addtid' in name else f"{name} {addr}, {d0}{off}{gds}"
  if 'swizzle' in name or op == DSOp.DS_ORDERED_COUNT: return f"{name} v{inst.vdst}, {addr}{off}{gds}"
  if 'permute' in name: return f"{name} v{inst.vdst}, {addr}, v{inst.data0}{off}{gds}"
  if 'condxchg' in name: return f"{name} {_vreg(inst.vdst, 2)}, {addr}, {_vreg(inst.data0, 2)}{off}{gds}"
  if _has(name, 'cmpstore', 'mskor', 'wrap'):
    return f"{name} {dst}, {addr}, {d0}, {d1}{off}{gds}" if '_rtn' in name else f"{name} {addr}, {d0}, {d1}{off}{gds}"
  return f"{name} {dst}, {addr}, {d0}{off}{gds}" if '_rtn' in name else f"{name} {addr}, {d0}{off}{gds}"

def _disasm_vop3(inst: VOP3) -> str:
  stored_op = inst._values.get('op')
  if stored_op is not None and hasattr(stored_op, 'name'):
    op, name = stored_op, stored_op.name.lower()
  else:
    op = VOP3SDOp(inst.op) if inst.op in VOP3SD_OPS else VOP3Op(inst.op)
    name = op.name.lower()

  # VOP3SD (shared encoding)
  if inst.op in VOP3SD_OPS:
    sdst = (inst.clmp << 7) | (inst.opsel << 3) | inst.abs
    is64, mad64 = 'f64' in name, _has(name, 'mad_i64_i32', 'mad_u64_u32')
    def src(v, neg, ext=False): s = _fmt_src(v, 2) if ext or is64 else inst.lit(v); return f"-{s}" if neg else s
    s0, s1, s2 = src(inst.src0, inst.neg & 1), src(inst.src1, inst.neg & 2), src(inst.src2, inst.neg & 4, mad64)
    dst = _vreg(inst.vdst, 2) if is64 or mad64 else f"v{inst.vdst}"
    if op in (VOP3SDOp.V_ADD_CO_U32, VOP3SDOp.V_SUB_CO_U32, VOP3SDOp.V_SUBREV_CO_U32): return f"{name} {dst}, {_fmt_sdst(sdst, 1)}, {s0}, {s1}"
    if op in (VOP3SDOp.V_ADD_CO_CI_U32, VOP3SDOp.V_SUB_CO_CI_U32, VOP3SDOp.V_SUBREV_CO_CI_U32): return f"{name} {dst}, {_fmt_sdst(sdst, 1)}, {s0}, {s1}, {s2}"
    return f"{name} {dst}, {_fmt_sdst(sdst, 1)}, {s0}, {s1}, {s2}" + _omod(inst.omod)

  # Detect operand sizes
  is64 = _is64(name)
  is64_src, is64_dst = False, False
  is16_d = is16_s = is16_s2 = False
  if 'cvt_pk' in name: is16_s = name.endswith('16')
  elif m := re.match(r'v_(?:cvt|frexp_exp)_([a-z0-9_]+)_([a-z0-9]+)', name):
    is16_d, is16_s = _has(m.group(1), 'f16','i16','u16','b16'), _has(m.group(2), 'f16','i16','u16','b16')
    is64_src, is64_dst = '64' in m.group(2), '64' in m.group(1)
    is16_s2, is64 = is16_s, False
  elif re.match(r'v_mad_[iu]32_[iu]16', name): is16_s = True
  elif 'pack_b32' in name: is16_s = is16_s2 = True
  else: is16_d = is16_s = is16_s2 = _is16(name) and not _has(name, 'dot2', 'pk_', 'sad', 'msad', 'qsad', 'mqsad')

  # Source counts
  shift64 = 'rev' in name and '64' in name and name.startswith('v_')
  ldexp64 = op == VOP3Op.V_LDEXP_F64
  trig = op == VOP3Op.V_TRIG_PREOP_F64
  sad64, mqsad = _has(name, 'qsad_pk', 'mqsad_pk'), 'mqsad_u32' in name
  s0n = 2 if ((is64 and not shift64) or sad64 or mqsad or is64_src) else 1
  s1n = 2 if (is64 and not _has(name, 'class') and not ldexp64 and not trig) else 1
  s2n = 4 if mqsad else 2 if (is64 or sad64) else 1

  any_hi = inst.opsel != 0
  s0 = _vop3_src(inst, inst.src0, inst.neg&1, inst.abs&1, inst.opsel&1, s0n, is16_s, any_hi)
  s1 = _vop3_src(inst, inst.src1, inst.neg&2, inst.abs&2, inst.opsel&2, s1n, is16_s, any_hi)
  s2 = _vop3_src(inst, inst.src2, inst.neg&4, inst.abs&4, inst.opsel&4, s2n, is16_s2, any_hi)

  # Destination
  dn = 4 if mqsad else 2 if (is64 or sad64 or is64_dst) else 1
  if op == VOP3Op.V_READLANE_B32: dst = _fmt_sdst(inst.vdst, 1)
  elif dn > 1: dst = _vreg(inst.vdst, dn)
  elif is16_d: dst = f"v{inst.vdst}.h" if (inst.opsel & 8) else f"v{inst.vdst}.l" if any_hi else f"v{inst.vdst}"
  else: dst = f"v{inst.vdst}"

  cl, om = " clamp" if inst.clmp else "", _omod(inst.omod)
  nonvgpr_opsel = (inst.src0 < 256 and (inst.opsel & 1)) or (inst.src1 < 256 and (inst.opsel & 2)) or (inst.src2 < 256 and (inst.opsel & 4))
  need_opsel = nonvgpr_opsel or (inst.opsel and not is16_s)

  # RDNA3 VOP3 encodes VOPC/VOP2/VOP1 in lower opcode ranges - skip these for CDNA VOP3A
  is_cdna_vop3a = stored_op is not None and type(stored_op).__name__ == 'VOP3AOp'
  if not is_cdna_vop3a:
    if inst.op < 256:  # VOPC
      return f"{name}_e64 {s0}, {s1}" if name.startswith('v_cmpx') else f"{name}_e64 {_fmt_sdst(inst.vdst, 1)}, {s0}, {s1}"
    if inst.op < 384:  # VOP2
      os = _opsel_str(inst.opsel, 3, need_opsel, is16_d) if 'cndmask' in name else _opsel_str(inst.opsel, 2, need_opsel, is16_d)
      return f"{name}_e64 {dst}, {s0}, {s1}, {s2}{os}{cl}{om}" if 'cndmask' in name else f"{name}_e64 {dst}, {s0}, {s1}{os}{cl}{om}"
    if inst.op < 512:  # VOP1
      return f"{name}_e64" if name in ('v_nop', 'v_pipeflush') else f"{name}_e64 {dst}, {s0}{_opsel_str(inst.opsel, 1, need_opsel, is16_d)}{cl}{om}"
  # Native VOP3
  is3 = _has(name, 'fma', 'mad', 'min3', 'max3', 'med3', 'div_fix', 'div_fmas', 'sad', 'lerp', 'align', 'cube', 'bfe', 'bfi',
             'perm_b32', 'permlane', 'cndmask', 'xor3', 'or3', 'add3', 'lshl_or', 'and_or', 'lshl_add', 'add_lshl', 'xad', 'maxmin', 'minmax', 'dot2', 'cvt_pk_u8', 'mullit')
  os = _opsel_str(inst.opsel, 3 if is3 else 2, need_opsel, is16_d)
  return f"{name} {dst}, {s0}, {s1}, {s2}{os}{cl}{om}" if is3 else f"{name} {dst}, {s0}, {s1}{os}{cl}{om}"

def _disasm_vop3sd(inst: VOP3SD) -> str:
  op, name = VOP3SDOp(inst.op), VOP3SDOp(inst.op).name.lower()
  is64, mad64 = 'f64' in name, _has(name, 'mad_i64_i32', 'mad_u64_u32')
  def src(v, neg, ext=False): s = _fmt_src(v, 2) if ext or is64 else inst.lit(v); return f"-{s}" if neg else s
  s0, s1, s2 = src(inst.src0, inst.neg & 1), src(inst.src1, inst.neg & 2), src(inst.src2, inst.neg & 4, mad64)
  dst, is2src = _vreg(inst.vdst, 2) if is64 or mad64 else f"v{inst.vdst}", op in (VOP3SDOp.V_ADD_CO_U32, VOP3SDOp.V_SUB_CO_U32, VOP3SDOp.V_SUBREV_CO_U32)
  suffix = "_e64" if name.startswith('v_') and 'co_' in name else ""
  return f"{name}{suffix} {dst}, {_fmt_sdst(inst.sdst, 1)}, {s0}, {s1}{'' if is2src else f', {s2}'}{' clamp' if inst.clmp else ''}{_omod(inst.omod)}"

def _disasm_vopd(inst: VOPD) -> str:
  lit = inst._literal or inst.literal
  vdst_y, nx, ny = (inst.vdsty << 1) | ((inst.vdstx & 1) ^ 1), VOPDOp(inst.opx).name.lower(), VOPDOp(inst.opy).name.lower()
  def half(n, vd, s0, vs1): return f"{n} v{vd}, {inst.lit(s0)}{f', 0x{lit:x}' if lit and _has(n, 'fmaak', 'fmamk') else ''}" if 'mov' in n else f"{n} v{vd}, {inst.lit(s0)}, v{vs1}{f', 0x{lit:x}' if lit and _has(n, 'fmaak', 'fmamk') else ''}"
  return f"{half(nx, inst.vdstx, inst.srcx0, inst.vsrcx1)} :: {half(ny, vdst_y, inst.srcy0, inst.vsrcy1)}"

def _disasm_vop3p(inst: VOP3P) -> str:
  name = VOP3POp(inst.op).name.lower()
  is_wmma, is_3src, is_fma_mix = 'wmma' in name, _has(name, 'fma', 'mad', 'dot', 'wmma'), 'fma_mix' in name
  if is_wmma:
    sc = 2 if 'iu4' in name else 4 if 'iu8' in name else 8
    src0, src1, src2, dst = _fmt_src(inst.src0, sc), _fmt_src(inst.src1, sc), _fmt_src(inst.src2, 8), _vreg(inst.vdst, 8)
  else: src0, src1, src2, dst = _fmt_src(inst.src0, 1), _fmt_src(inst.src1, 1), _fmt_src(inst.src2, 1), f"v{inst.vdst}"
  n, opsel_hi = 3 if is_3src else 2, inst.opsel_hi | (inst.opsel_hi2 << 2)
  if is_fma_mix:
    def m(s, neg, abs_): return f"-{f'|{s}|' if abs_ else s}" if neg else (f"|{s}|" if abs_ else s)
    src0, src1, src2 = m(src0, inst.neg & 1, inst.neg_hi & 1), m(src1, inst.neg & 2, inst.neg_hi & 2), m(src2, inst.neg & 4, inst.neg_hi & 4)
    mods = ([_fmt_bits("op_sel", inst.opsel, n)] if inst.opsel else []) + ([_fmt_bits("op_sel_hi", opsel_hi, n)] if opsel_hi else []) + (["clamp"] if inst.clmp else [])
  else:
    mods = ([_fmt_bits("op_sel", inst.opsel, n)] if inst.opsel else []) + ([_fmt_bits("op_sel_hi", opsel_hi, n)] if opsel_hi != (7 if is_3src else 3) else []) + \
           ([_fmt_bits("neg_lo", inst.neg, n)] if inst.neg else []) + ([_fmt_bits("neg_hi", inst.neg_hi, n)] if inst.neg_hi else []) + (["clamp"] if inst.clmp else [])
  return f"{name} {dst}, {src0}, {src1}, {src2}{' ' + ' '.join(mods) if mods else ''}" if is_3src else f"{name} {dst}, {src0}, {src1}{' ' + ' '.join(mods) if mods else ''}"

def _disasm_buf(inst: MUBUF | MTBUF) -> str:
  op = MTBUFOp(inst.op) if isinstance(inst, MTBUF) else MUBUFOp(inst.op)
  name = op.name.lower()
  if op in (MUBUFOp.BUFFER_GL0_INV, MUBUFOp.BUFFER_GL1_INV): return name
  w = (2 if _has(name, 'xyz', 'xyzw') else 1) if 'd16' in name else \
      ((2 if _has(name, 'b64', 'u64', 'i64') else 1) * (2 if 'cmpswap' in name else 1)) if 'atomic' in name else \
      {'b32':1,'b64':2,'b96':3,'b128':4,'b16':1,'x':1,'xy':2,'xyz':3,'xyzw':4}.get(name.split('_')[-1], 1)
  if inst.tfe: w += 1
  vaddr = _vreg(inst.vaddr, 2) if inst.offen and inst.idxen else f"v{inst.vaddr}" if inst.offen or inst.idxen else "off"
  srsrc = _sreg_or_ttmp(inst.srsrc*4, 4)
  mods = ([f"format:{inst.format}"] if isinstance(inst, MTBUF) else []) + [m for c, m in [(inst.idxen,"idxen"),(inst.offen,"offen"),(inst.offset,f"offset:{inst.offset}"),(inst.glc,"glc"),(inst.dlc,"dlc"),(inst.slc,"slc"),(inst.tfe,"tfe")] if c]
  return f"{name} {_vreg(inst.vdata, w)}, {vaddr}, {srsrc}, {decode_src(inst.soffset)}{' ' + ' '.join(mods) if mods else ''}"

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
  name = MIMGOp(inst.op).name.lower()
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

def _sop_widths(name: str) -> tuple[int, int, int]:
  """Return (dst_width, src0_width, src1_width) in register count for SOP instructions."""
  if name in ('s_bitset0_b64', 's_bitset1_b64', 's_bfm_b64'): return 2, 1, 1
  if name in ('s_lshl_b64', 's_lshr_b64', 's_ashr_i64', 's_bfe_u64', 's_bfe_i64'): return 2, 2, 1
  if name in ('s_bitcmp0_b64', 's_bitcmp1_b64'): return 1, 2, 1
  if m := re.search(r'_(b|i|u)(32|64)_(b|i|u)(32|64)$', name): return 2 if m.group(2) == '64' else 1, 2 if m.group(4) == '64' else 1, 1
  if m := re.search(r'_(b|i|u)(32|64)$', name): sz = 2 if m.group(2) == '64' else 1; return sz, sz, sz
  return 1, 1, 1

def _disasm_sop1(inst: SOP1) -> str:
  stored_op = inst._values.get('op')
  name = stored_op.name.lower() if hasattr(stored_op, 'name') else SOP1Op(inst.op).name.lower()
  is_cdna = hasattr(stored_op, 'name') and 'cdna' in type(stored_op).__module__
  if name == 's_getpc_b64': return f"{name} {_fmt_sdst(inst.sdst, 2, is_cdna=is_cdna)}"
  if name in ('s_setpc_b64', 's_rfe_b64'): return f"{name} {_fmt_src(inst.ssrc0, 2)}"
  if name == 's_swappc_b64': return f"{name} {_fmt_sdst(inst.sdst, 2, is_cdna=is_cdna)}, {_fmt_src(inst.ssrc0, 2)}"
  if name in ('s_sendmsg_rtn_b32', 's_sendmsg_rtn_b64'): return f"{name} {_fmt_sdst(inst.sdst, 2 if 'b64' in name else 1, is_cdna=is_cdna)}, sendmsg({MSG.get(inst.ssrc0, str(inst.ssrc0))})"
  dn, s0n, _ = _sop_widths(name)
  return f"{name} {_fmt_sdst(inst.sdst, dn, is_cdna=is_cdna)}, {inst.lit(inst.ssrc0) if s0n == 1 else _fmt_src(inst.ssrc0, s0n)}"

def _disasm_sop2(inst: SOP2) -> str:
  stored_op = inst._values.get('op')
  name = stored_op.name.lower() if hasattr(stored_op, 'name') else SOP2Op(inst.op).name.lower()
  dn, s0n, s1n = _sop_widths(name)
  return f"{name} {_fmt_sdst(inst.sdst, dn)}, {inst.lit(inst.ssrc0) if inst.ssrc0 == 255 else _fmt_src(inst.ssrc0, s0n)}, {inst.lit(inst.ssrc1) if inst.ssrc1 == 255 else _fmt_src(inst.ssrc1, s1n)}"

def _disasm_sopc(inst: SOPC) -> str:
  name = SOPCOp(inst.op).name.lower()
  _, s0n, s1n = _sop_widths(name)
  return f"{name} {_fmt_src(inst.ssrc0, s0n)}, {_fmt_src(inst.ssrc1, s1n)}"

def _disasm_sopk(inst: SOPK) -> str:
  op, name = SOPKOp(inst.op), SOPKOp(inst.op).name.lower()
  if op == SOPKOp.S_VERSION: return f"{name} 0x{inst.simm16:x}"
  if op in (SOPKOp.S_SETREG_B32, SOPKOp.S_GETREG_B32):
    hid, hoff, hsz = inst.simm16 & 0x3f, (inst.simm16 >> 6) & 0x1f, ((inst.simm16 >> 11) & 0x1f) + 1
    hs = f"0x{inst.simm16:x}" if hid in (16, 17) else f"hwreg({HWREG.get(hid, str(hid))}, {hoff}, {hsz})"
    return f"{name} {hs}, {_fmt_sdst(inst.sdst, 1)}" if op == SOPKOp.S_SETREG_B32 else f"{name} {_fmt_sdst(inst.sdst, 1)}, {hs}"
  dn, _, _ = _sop_widths(name)
  return f"{name} {_fmt_sdst(inst.sdst, dn)}, 0x{inst.simm16:x}"

def _disasm_vinterp(inst: VINTERP) -> str:
  name = VINTERPOp(inst.op).name.lower()
  src0 = f"-{inst.lit(inst.src0)}" if inst.neg & 1 else inst.lit(inst.src0)
  src1 = f"-{inst.lit(inst.src1)}" if inst.neg & 2 else inst.lit(inst.src1)
  src2 = f"-{inst.lit(inst.src2)}" if inst.neg & 4 else inst.lit(inst.src2)
  mods = _mods((inst.waitexp, f"wait_exp:{inst.waitexp}"), (inst.clmp, "clamp"))
  return f"{name} v{inst.vdst}, {src0}, {src1}, {src2}" + (" " + mods if mods else "")

def _disasm_generic(inst: Inst) -> str:
  name = f"op_{inst.op}"
  def format_field(field_name, val):
    val = unwrap(val)
    if field_name in SRC_FIELDS: return inst.lit(val) if val != 255 else "0xff"
    return f"{'s' if field_name == 'sdst' else 'v'}{val}" if field_name in ('sdst', 'vdst') else f"v{val}" if field_name == 'vsrc1' else f"0x{val:x}" if field_name == 'simm16' else str(val)
  operands = [format_field(field_name, inst._values.get(field_name, 0)) for field_name in inst._fields if field_name not in ('encoding', 'op')]
  return f"{name} {', '.join(operands)}" if operands else name

DISASM_HANDLERS = {VOP1: _disasm_vop1, VOP2: _disasm_vop2, VOPC: _disasm_vopc, VOP3: _disasm_vop3, VOP3SD: _disasm_vop3sd, VOPD: _disasm_vopd, VOP3P: _disasm_vop3p,
                   VINTERP: _disasm_vinterp, SOPP: _disasm_sopp, SMEM: _disasm_smem, DS: _disasm_ds, FLAT: _disasm_flat, MUBUF: _disasm_buf, MTBUF: _disasm_buf,
                   MIMG: _disasm_mimg, SOP1: _disasm_sop1, SOP2: _disasm_sop2, SOPC: _disasm_sopc, SOPK: _disasm_sopk}
DISASM_BY_NAME = {cls.__name__: handler for cls, handler in DISASM_HANDLERS.items()}
DISASM_BY_NAME['VOP3A'] = _disasm_vop3  # CDNA VOP3A uses same handler as VOP3

def disasm(inst: Inst) -> str: return DISASM_HANDLERS.get(type(inst), DISASM_BY_NAME.get(type(inst).__name__, _disasm_generic))(inst)

# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLER
# ═══════════════════════════════════════════════════════════════════════════════

SPEC_REGS = {'vcc_lo': RawImm(106), 'vcc_hi': RawImm(107), 'vcc': RawImm(106), 'null': RawImm(124), 'off': RawImm(124), 'm0': RawImm(125),
             'exec_lo': RawImm(126), 'exec_hi': RawImm(127), 'exec': RawImm(126), 'scc': RawImm(253), 'src_scc': RawImm(253)}
FLOATS = {str(k): k for k in FLOAT_ENC}  # Valid float literal strings: '0.5', '-0.5', '1.0', etc.
REG_MAP: dict[str, _RegFactory] = {'s': s, 'v': v, 't': ttmp, 'ttmp': ttmp}
# RDNA3/4 uses _b32/_b64 naming, CDNA uses _dword/_dwordx2 naming
SMEM_OPS = {'s_load_b32', 's_load_b64', 's_load_b128', 's_load_b256', 's_load_b512',
            's_buffer_load_b32', 's_buffer_load_b64', 's_buffer_load_b128', 's_buffer_load_b256', 's_buffer_load_b512',
            's_load_dword', 's_load_dwordx2', 's_load_dwordx4', 's_load_dwordx8', 's_load_dwordx16',
            's_buffer_load_dword', 's_buffer_load_dwordx2', 's_buffer_load_dwordx4', 's_buffer_load_dwordx8', 's_buffer_load_dwordx16'}
CDNA_SMEM_OPS = {'s_load_dword', 's_load_dwordx2', 's_load_dwordx4', 's_load_dwordx8', 's_load_dwordx16',
                 's_buffer_load_dword', 's_buffer_load_dwordx2', 's_buffer_load_dwordx4', 's_buffer_load_dwordx8', 's_buffer_load_dwordx16'}
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
  if m := re.search(pat, text, flags): return m, text[:m.start()] + text[m.end():]
  return None, text

def get_dsl(text: str) -> str:
  text, kw = text.strip(), []
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

  # SMEM - RDNA3/4 uses soffset=RawImm(124) for immediate offset, CDNA uses imm=1, soffset=RawImm(0)
  if mn in SMEM_OPS:
    is_cdna = mn in CDNA_SMEM_OPS
    gs, ds = ", glc=1" if glc else "", ", dlc=1" if dlc else ""
    imm_kw = ", imm=1, soffset=RawImm(0)" if is_cdna else ", soffset=RawImm(124)"
    if len(ops) >= 3 and re.match(r'^-?[0-9]|^-?0x', ops[2].strip().lower()):
      return f"{mn}(sdata={args[0]}, sbase={args[1]}, offset={args[2]}{imm_kw}{gs}{ds})"
    if off_val and len(ops) >= 3: return f"{mn}(sdata={args[0]}, sbase={args[1]}, offset={off_val}, soffset={args[2]}{gs}{ds})"
    if len(ops) >= 3: return f"{mn}(sdata={args[0]}, sbase={args[1]}, soffset={args[2]}{gs}{ds})"

  # Buffer
  if mn.startswith('buffer_') and len(ops) >= 2 and ops[1].strip().lower() == 'off':
    return f"{mn}(vdata={args[0]}, vaddr=0, srsrc={args[2]}, soffset={f'RawImm({args[3].strip()})' if len(args) > 3 else 'RawImm(0)'})"

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

  fn = mn.replace('.', '_')
  if opsel is not None: args = [re.sub(r'\.[hl]$', '', a) for a in args]

  # VOP3A (CDNA native VOP3) needs keyword args since op is passed via partial
  is_vop3a_3src = _has(mn, 'lshl_add', 'add_lshl', 'lshl_or', 'and_or', 'xor3', 'or3', 'add3') and not mn.endswith(('_e32', '_e64'))
  is_vop3a_2src = _has(mn, 'mul_lo_u32', 'mul_hi_u32', 'mul_lo_i32', 'mul_hi_i32') and not mn.endswith(('_e32', '_e64'))
  if is_vop3a_3src and len(args) == 4:
    args = [f'vdst={args[0]}', f'src0={args[1]}', f'src1={args[2]}', f'src2={args[3]}']
  elif is_vop3a_2src and len(args) == 3:
    args = [f'vdst={args[0]}', f'src0={args[1]}', f'src1={args[2]}']

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

def asm(text: str, arch: str = 'rdna3') -> Inst:
  if arch == 'cdna': from extra.assembly.amd.autogen import cdna as ag
  elif arch == 'rdna4': from extra.assembly.amd.autogen import rdna4 as ag
  else: from extra.assembly.amd.autogen import rdna3 as ag
  dsl = get_dsl(text)
  ns = {n: getattr(ag, n) for n in dir(ag) if not n.startswith('_')}
  ns.update({'s': s, 'v': v, 'ttmp': ttmp, 'abs': abs, 'RawImm': RawImm, 'SrcMod': SrcMod, 'VGPR': VGPR, 'SGPR': SGPR, 'TTMP': TTMP,
             'VCC_LO': VCC_LO, 'VCC_HI': VCC_HI, 'VCC': VCC, 'EXEC_LO': EXEC_LO, 'EXEC_HI': EXEC_HI, 'EXEC': EXEC, 'SCC': SCC, 'NULL': NULL, 'OFF': OFF})
  try: return eval(dsl, ns)
  except NameError:
    if m := re.match(r'^(v_\w+)(\(.*\))$', dsl): return eval(f"{m.group(1)}_e32{m.group(2)}", ns)
    raise
