# RDNA3 assembler and disassembler
from __future__ import annotations
import re
from extra.assembly.amd.dsl import Inst, RawImm, Reg, SrcMod, SGPR, VGPR, TTMP, s, v, ttmp, _RegFactory, FLOAT_ENC, SRC_FIELDS, unwrap
from extra.assembly.amd.dsl import VCC_LO, VCC_HI, VCC, EXEC_LO, EXEC_HI, EXEC, SCC, M0, NULL, OFF
from extra.assembly.amd.autogen import rdna3
from extra.assembly.amd.autogen.rdna3 import VOP1, VOP2, VOP3, VOP3SD, VOP3P, VOPC, VOPD, VINTERP, SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, DS, FLAT, MUBUF, MTBUF
from extra.assembly.amd.autogen.rdna3 import VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp, VOPDOp, VINTERPOp
from extra.assembly.amd.autogen.rdna3 import SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, SMEMOp, FLATOp, MUBUFOp, MTBUFOp

# VOP3SD opcodes that share VOP3 encoding
VOP3SD_OPS = {288, 289, 290, 764, 765, 766, 767, 768, 769, 770}

def detect_format(data: bytes) -> type[Inst]:
  """Detect instruction format from machine code bytes."""
  assert len(data) >= 4, f"need at least 4 bytes, got {len(data)}"
  word = int.from_bytes(data[:4], 'little')
  hi2 = (word >> 30) & 0x3
  if hi2 == 0b11:
    enc = (word >> 26) & 0xf
    if enc == 0b1101: return SMEM
    if enc == 0b0101: return VOP3SD if ((word >> 16) & 0x3ff) in VOP3SD_OPS else VOP3
    if enc == 0b0011: return VOP3P
    if enc == 0b0110: return DS
    if enc == 0b0111: return FLAT
    if enc == 0b0010: return VOPD
    if enc == 0b0100: return VINTERP
    raise ValueError(f"unknown 64-bit format enc={enc:#06b} word={word:#010x}")
  if hi2 == 0b10:
    enc = (word >> 23) & 0x7f
    if enc == 0b1111101: return SOP1
    if enc == 0b1111110: return SOPC
    if enc == 0b1111111: return SOPP
    return SOPK if ((word >> 28) & 0xf) == 0b1011 else SOP2
  # hi2 == 0b00 or 0b01: VOP1/VOP2/VOPC (bit 31 = 0)
  assert (word >> 31) == 0, f"expected bit 31 = 0 for VOP, got word={word:#010x}"
  enc = (word >> 25) & 0x7f
  if enc == 0b0111110: return VOPC
  if enc == 0b0111111: return VOP1
  if enc <= 0b0111101: return VOP2  # bits 31:25 = 0xxxxxx where xxxxxx <= 0b111101
  raise ValueError(f"unknown VOP format enc={enc:#09b} word={word:#010x}")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SPECIAL_GPRS = {106: "vcc_lo", 107: "vcc_hi", 124: "null", 125: "m0", 126: "exec_lo", 127: "exec_hi", 253: "scc"}
SPECIAL_DEC = {**SPECIAL_GPRS, **{v: str(k) for k, v in FLOAT_ENC.items()}}
SPECIAL_PAIRS = {106: "vcc", 126: "exec"}
HWREG = {1: 'HW_REG_MODE', 2: 'HW_REG_STATUS', 3: 'HW_REG_TRAPSTS', 4: 'HW_REG_HW_ID', 5: 'HW_REG_GPR_ALLOC',
         6: 'HW_REG_LDS_ALLOC', 7: 'HW_REG_IB_STS', 15: 'HW_REG_SH_MEM_BASES', 18: 'HW_REG_PERF_SNAPSHOT_PC_LO',
         19: 'HW_REG_PERF_SNAPSHOT_PC_HI', 20: 'HW_REG_FLAT_SCR_LO', 21: 'HW_REG_FLAT_SCR_HI', 22: 'HW_REG_XNACK_MASK',
         23: 'HW_REG_HW_ID1', 24: 'HW_REG_HW_ID2', 25: 'HW_REG_POPS_PACKER', 28: 'HW_REG_IB_STS2'}
HWREG_IDS = {v.lower(): k for k, v in HWREG.items()}
MSG = {128: 'MSG_RTN_GET_DOORBELL', 129: 'MSG_RTN_GET_DDID', 130: 'MSG_RTN_GET_TMA',
       131: 'MSG_RTN_GET_REALTIME', 132: 'MSG_RTN_SAVE_WAVE', 133: 'MSG_RTN_GET_TBA'}
VOP3SD_OPS = {288, 289, 290, 764, 765, 766, 767, 768, 769, 770}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def decode_src(val: int) -> str:
  if val <= 105: return f"s{val}"
  if val in SPECIAL_DEC: return SPECIAL_DEC[val]
  if 108 <= val <= 123: return f"ttmp{val - 108}"
  if 128 <= val <= 192: return str(val - 128)
  if 193 <= val <= 208: return str(-(val - 192))
  if 256 <= val <= 511: return f"v{val - 256}"
  return "lit" if val == 255 else f"?{val}"

def _reg(p: str, b: int, n: int = 1) -> str: return f"{p}{b}" if n == 1 else f"{p}[{b}:{b+n-1}]"
def _sreg(b: int, n: int = 1) -> str: return _reg("s", b, n)
def _vreg(b: int, n: int = 1) -> str: return _reg("v", b, n)
def _hl(v: int, hi_thresh: int = 128) -> str: return 'h' if v >= hi_thresh else 'l'

def _fmt_sdst(v: int, n: int = 1) -> str:
  if v == 124: return "null"
  if 108 <= v <= 123: return _reg("ttmp", v - 108, n)
  if n > 1: return SPECIAL_PAIRS.get(v) or _sreg(v, n)
  return {126: "exec_lo", 127: "exec_hi", 106: "vcc_lo", 107: "vcc_hi", 125: "m0"}.get(v, f"s{v}")

def _fmt_src(v: int, n: int = 1) -> str:
  if n == 1: return decode_src(v)
  if v >= 256: return _vreg(v - 256, n)
  if v <= 105: return _sreg(v, n)
  if n == 2 and v in SPECIAL_PAIRS: return SPECIAL_PAIRS[v]
  if 108 <= v <= 123: return _reg("ttmp", v - 108, n)
  return decode_src(v)

def _fmt_v16(v: int, base: int = 256, hi_thresh: int = 384) -> str:
  return f"v{(v - base) & 0x7f}.{_hl(v, hi_thresh)}"

def waitcnt(vmcnt: int = 0x3f, expcnt: int = 0x7, lgkmcnt: int = 0x3f) -> int:
  return (expcnt & 0x7) | ((lgkmcnt & 0x3f) << 4) | ((vmcnt & 0x3f) << 10)

def _has(op: str, *subs) -> bool: return any(s in op for s in subs)
def _is16(op: str) -> bool: return _has(op, 'f16', 'i16', 'u16', 'b16') and not _has(op, '_f32', '_i32')
def _is64(op: str) -> bool: return _has(op, 'f64', 'i64', 'u64', 'b64')
def _omod(v: int) -> str: return {1: " mul:2", 2: " mul:4", 3: " div:2"}.get(v, "")
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

def _disasm_vop1(inst: VOP1) -> str:
  op = VOP1Op(inst.op)
  if op in (VOP1Op.V_NOP, VOP1Op.V_PIPEFLUSH): return op.name.lower()
  F64_OPS = {VOP1Op.V_CEIL_F64, VOP1Op.V_FLOOR_F64, VOP1Op.V_FRACT_F64, VOP1Op.V_FREXP_MANT_F64, VOP1Op.V_RCP_F64, VOP1Op.V_RNDNE_F64, VOP1Op.V_RSQ_F64, VOP1Op.V_SQRT_F64, VOP1Op.V_TRUNC_F64}
  is_f64_d = op in F64_OPS or op in (VOP1Op.V_CVT_F64_F32, VOP1Op.V_CVT_F64_I32, VOP1Op.V_CVT_F64_U32)
  is_f64_s = op in F64_OPS or op in (VOP1Op.V_CVT_F32_F64, VOP1Op.V_CVT_I32_F64, VOP1Op.V_CVT_U32_F64, VOP1Op.V_FREXP_EXP_I32_F64)
  name = op.name.lower()
  parts = name.split('_')
  is_16d = any(p in ('f16','i16','u16','b16') for p in parts[-2:-1]) or (len(parts) >= 2 and parts[-1] in ('f16','i16','u16','b16') and 'cvt' not in name)
  is_16s = parts[-1] in ('f16','i16','u16','b16') and 'sat_pk' not in name
  if op == VOP1Op.V_READFIRSTLANE_B32: return f"v_readfirstlane_b32 {decode_src(inst.vdst)}, v{inst.src0 - 256 if inst.src0 >= 256 else inst.src0}"
  dst = _vreg(inst.vdst, 2) if is_f64_d else _fmt_v16(inst.vdst, 0, 128) if is_16d else f"v{inst.vdst}"
  src = _fmt_src(inst.src0, 2) if is_f64_s else _fmt_v16(inst.src0) if is_16s and inst.src0 >= 256 else inst.lit(inst.src0)
  return f"{name}_e32 {dst}, {src}"

def _disasm_vop2(inst: VOP2) -> str:
  op = VOP2Op(inst.op)
  name = op.name.lower()
  suf = "" if op == VOP2Op.V_DOT2ACC_F32_F16 else "_e32"
  is16 = _is16(name) and 'pk_' not in name
  # fmaak: dst = src0 * vsrc1 + K, fmamk: dst = src0 * K + vsrc1
  if op in (VOP2Op.V_FMAAK_F32, VOP2Op.V_FMAAK_F16): return f"{name}{suf} v{inst.vdst}, {inst.lit(inst.src0)}, v{inst.vsrc1}, 0x{inst._literal:x}"
  if op in (VOP2Op.V_FMAMK_F32, VOP2Op.V_FMAMK_F16): return f"{name}{suf} v{inst.vdst}, {inst.lit(inst.src0)}, 0x{inst._literal:x}, v{inst.vsrc1}"
  if is16: return f"{name}{suf} {_fmt_v16(inst.vdst, 0, 128)}, {_fmt_v16(inst.src0) if inst.src0 >= 256 else inst.lit(inst.src0)}, {_fmt_v16(inst.vsrc1, 0, 128)}"
  return f"{name}{suf} v{inst.vdst}, {inst.lit(inst.src0)}, v{inst.vsrc1}" + (", vcc_lo" if op == VOP2Op.V_CNDMASK_B32 else "")

def _disasm_vopc(inst: VOPC) -> str:
  name = VOPCOp(inst.op).name.lower()
  is64, is16 = _is64(name), _is16(name)
  s0 = _fmt_src(inst.src0, 2) if is64 else _fmt_v16(inst.src0) if is16 and inst.src0 >= 256 else inst.lit(inst.src0)
  s1 = _vreg(inst.vsrc1, 2) if is64 and 'class' not in name else _fmt_v16(inst.vsrc1, 0, 128) if is16 else f"v{inst.vsrc1}"
  return f"{name}_e32 {s0}, {s1}" if name.startswith('v_cmpx') else f"{name}_e32 vcc_lo, {s0}, {s1}"

def _disasm_sopp(inst: SOPP) -> str:
  op = SOPPOp(inst.op)
  name = op.name.lower()
  NO_ARG_NAMES = {'s_endpgm', 's_barrier', 's_wakeup', 's_icache_inv', 's_ttracedata', 's_ttracedata_imm',
                  's_wait_idle', 's_endpgm_saved', 's_code_end', 's_endpgm_ordered_ps_done'}
  if name in NO_ARG_NAMES: return name
  if op == SOPPOp.S_WAITCNT:
    vm, exp, lgkm = (inst.simm16 >> 10) & 0x3f, inst.simm16 & 0xf, (inst.simm16 >> 4) & 0x3f
    p = [f"vmcnt({vm})" if vm != 0x3f else "", f"expcnt({exp})" if exp != 7 else "", f"lgkmcnt({lgkm})" if lgkm != 0x3f else ""]
    return f"s_waitcnt {' '.join(x for x in p if x)}" if any(p) else "s_waitcnt 0"
  if op == SOPPOp.S_DELAY_ALU:
    deps = ['VALU_DEP_1','VALU_DEP_2','VALU_DEP_3','VALU_DEP_4','TRANS32_DEP_1','TRANS32_DEP_2','TRANS32_DEP_3','FMA_ACCUM_CYCLE_1','SALU_CYCLE_1','SALU_CYCLE_2','SALU_CYCLE_3']
    skips = ['SAME','NEXT','SKIP_1','SKIP_2','SKIP_3','SKIP_4']
    id0, skip, id1 = inst.simm16 & 0xf, (inst.simm16 >> 4) & 0x7, (inst.simm16 >> 7) & 0xf
    dep = lambda v: deps[v-1] if 0 < v <= len(deps) else str(v)
    p = [f"instid0({dep(id0)})" if id0 else "", f"instskip({skips[skip]})" if skip else "", f"instid1({dep(id1)})" if id1 else ""]
    return f"s_delay_alu {' | '.join(x for x in p if x)}" if any(p) else "s_delay_alu 0"
  if name.startswith('s_cbranch') or name.startswith('s_branch'): return f"{name} {inst.simm16}"
  return f"{name} 0x{inst.simm16:x}"

def _disasm_smem(inst: SMEM) -> str:
  op = SMEMOp(inst.op)
  name = op.name.lower()
  if name in ('s_gl1_inv', 's_dcache_inv'): return name
  off_s = f"{decode_src(inst.soffset)} offset:0x{inst.offset:x}" if inst.offset and inst.soffset != 124 else f"0x{inst.offset:x}" if inst.offset else decode_src(inst.soffset)
  sbase_idx, sbase_count = inst.sbase * 2, 4 if (8 <= inst.op <= 12 or name == 's_atc_probe_buffer') else 2
  sbase_str = _fmt_src(sbase_idx, sbase_count) if sbase_count == 2 else _sreg(sbase_idx, sbase_count) if sbase_idx <= 105 else _reg("ttmp", sbase_idx - 108, sbase_count)
  if name in ('s_atc_probe', 's_atc_probe_buffer'): return f"{name} {inst.sdata}, {sbase_str}, {off_s}"
  width = {0:1, 1:2, 2:4, 3:8, 4:16, 8:1, 9:2, 10:4, 11:8, 12:16}.get(inst.op, 1)
  return f"{name} {_fmt_sdst(inst.sdata, width)}, {sbase_str}, {off_s}" + _mods((inst.glc, " glc"), (inst.dlc, " dlc"))

def _disasm_flat(inst: FLAT) -> str:
  op = FLATOp(inst.op)
  name = op.name.lower()
  instr = f"{['flat', 'scratch', 'global'][inst.seg] if inst.seg < 3 else 'flat'}_{name.split('_', 1)[1] if '_' in name else name}"
  w = {'b32':1, 'b64':2, 'b96':3, 'b128':4, 'u8':1, 'i8':1, 'u16':1, 'i16':1}.get(name.split('_')[-1], 1)
  addr_s = _vreg(inst.addr, 2 if inst.saddr == 0x7F else 1)
  saddr_s = "" if inst.saddr == 0x7F else f", {_sreg(inst.saddr, 2)}" if inst.saddr < 106 else ", off" if inst.saddr == 124 else f", {decode_src(inst.saddr)}"
  off_s = f" offset:{inst.offset}" if inst.offset else ""
  vdata = _vreg(inst.data if 'store' in name else inst.vdst, w)
  return f"{instr} {addr_s}, {vdata}{saddr_s}{off_s}" if 'store' in name else f"{instr} {vdata}, {addr_s}{saddr_s}{off_s}"

def _disasm_vop3(inst: VOP3) -> str:
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

  if inst.op < 256:  # VOPC
    return f"{name}_e64 {s0}, {s1}" if name.startswith('v_cmpx') else f"{name}_e64 {_fmt_sdst(inst.vdst, 1)}, {s0}, {s1}"
  if inst.op < 384:  # VOP2
    os = _opsel_str(inst.opsel, 3, need_opsel, is16_d) if 'cndmask' in name else _opsel_str(inst.opsel, 2, need_opsel, is16_d)
    return f"{name}_e64 {dst}, {s0}, {s1}, {s2}{os}{cl}{om}" if 'cndmask' in name else f"{name}_e64 {dst}, {s0}, {s1}{os}{cl}{om}"
  if inst.op < 512:  # VOP1
    return f"{name}_e64" if op in (VOP3Op.V_NOP, VOP3Op.V_PIPEFLUSH) else f"{name}_e64 {dst}, {s0}{_opsel_str(inst.opsel, 1, need_opsel, is16_d)}{cl}{om}"
  # Native VOP3
  is3 = _has(name, 'fma', 'mad', 'min3', 'max3', 'med3', 'div_fix', 'div_fmas', 'sad', 'lerp', 'align', 'cube', 'bfe', 'bfi',
             'perm_b32', 'permlane', 'cndmask', 'xor3', 'or3', 'add3', 'lshl_or', 'and_or', 'lshl_add', 'add_lshl', 'xad', 'maxmin', 'minmax', 'dot2', 'cvt_pk_u8', 'mullit')
  os = _opsel_str(inst.opsel, 3 if is3 else 2, need_opsel, is16_d)
  return f"{name} {dst}, {s0}, {s1}, {s2}{os}{cl}{om}" if is3 else f"{name} {dst}, {s0}, {s1}{os}{cl}{om}"

def _disasm_vop3sd(inst: VOP3SD) -> str:
  op = VOP3SDOp(inst.op)
  name = op.name.lower()
  is64, mad64 = 'f64' in name, _has(name, 'mad_i64_i32', 'mad_u64_u32')
  def src(v, neg, ext=False): s = _fmt_src(v, 2) if ext or is64 else inst.lit(v); return f"-{s}" if neg else s
  s0, s1, s2 = src(inst.src0, inst.neg & 1), src(inst.src1, inst.neg & 2), src(inst.src2, inst.neg & 4, mad64)
  dst = _vreg(inst.vdst, 2) if is64 or mad64 else f"v{inst.vdst}"
  is2src = op in (VOP3SDOp.V_ADD_CO_U32, VOP3SDOp.V_SUB_CO_U32, VOP3SDOp.V_SUBREV_CO_U32)
  suffix = "_e64" if name.startswith('v_') and 'co_' in name else ""
  srcs = f"{s0}, {s1}" if is2src else f"{s0}, {s1}, {s2}"
  return f"{name}{suffix} {dst}, {_fmt_sdst(inst.sdst, 1)}, {srcs}" + (" clamp" if inst.clmp else "") + _omod(inst.omod)

def _disasm_vopd(inst: VOPD) -> str:
  literal = inst._literal or inst.literal
  vdst_y = (inst.vdsty << 1) | ((inst.vdstx & 1) ^ 1)
  op_x, op_y = VOPDOp(inst.opx), VOPDOp(inst.opy)
  name_x, name_y = op_x.name.lower(), op_y.name.lower()
  lit_x = f", 0x{literal:x}" if literal and _has(name_x, 'fmaak', 'fmamk') else ""
  lit_y = f", 0x{literal:x}" if literal and _has(name_y, 'fmaak', 'fmamk') else ""
  half_x = f"{name_x} v{inst.vdstx}, {inst.lit(inst.srcx0)}{lit_x}" if 'mov' in name_x else f"{name_x} v{inst.vdstx}, {inst.lit(inst.srcx0)}, v{inst.vsrcx1}{lit_x}"
  half_y = f"{name_y} v{vdst_y}, {inst.lit(inst.srcy0)}{lit_y}" if 'mov' in name_y else f"{name_y} v{vdst_y}, {inst.lit(inst.srcy0)}, v{inst.vsrcy1}{lit_y}"
  return f"{half_x} :: {half_y}"

def _disasm_vop3p(inst: VOP3P) -> str:
  name = VOP3POp(inst.op).name.lower()
  is_wmma = 'wmma' in name
  is_3src = _has(name, 'fma', 'mad', 'dot', 'wmma')
  if is_wmma:
    src_count = 2 if 'iu4' in name else 4 if 'iu8' in name else 8
    src0, src1, src2, dst = _fmt_src(inst.src0, src_count), _fmt_src(inst.src1, src_count), _fmt_src(inst.src2, 8), _vreg(inst.vdst, 8)
  else: src0, src1, src2, dst = _fmt_src(inst.src0, 1), _fmt_src(inst.src1, 1), _fmt_src(inst.src2, 1), f"v{inst.vdst}"
  num_srcs = 3 if is_3src else 2
  opsel_hi_combined = inst.opsel_hi | (inst.opsel_hi2 << 2)
  mods = ([_fmt_bits("op_sel", inst.opsel, num_srcs)] if inst.opsel else []) + \
         ([_fmt_bits("op_sel_hi", opsel_hi_combined, num_srcs)] if opsel_hi_combined != (7 if is_3src else 3) else []) + \
         ([_fmt_bits("neg_lo", inst.neg, num_srcs)] if inst.neg else []) + \
         ([_fmt_bits("neg_hi", inst.neg_hi, num_srcs)] if inst.neg_hi else []) + (["clamp"] if inst.clmp else [])
  mods_str = ' ' + ' '.join(mods) if mods else ''
  return f"{name} {dst}, {src0}, {src1}, {src2}{mods_str}" if is_3src else f"{name} {dst}, {src0}, {src1}{mods_str}"

def _disasm_buf(inst: MUBUF | MTBUF) -> str:
  op = MTBUFOp(inst.op) if isinstance(inst, MTBUF) else MUBUFOp(inst.op)
  name = op.name.lower()
  if op in (MUBUFOp.BUFFER_GL0_INV, MUBUFOp.BUFFER_GL1_INV): return name
  if 'd16' in name: width = 2 if _has(name, 'xyz', 'xyzw') else 1
  elif 'atomic' in name: base = 2 if _has(name, 'b64', 'u64', 'i64') else 1; width = base * 2 if 'cmpswap' in name else base
  else: width = {'b32':1, 'b64':2, 'b96':3, 'b128':4, 'b16':1, 'x':1, 'xy':2, 'xyz':3, 'xyzw':4}.get(name.split('_')[-1], 1)
  if inst.tfe: width += 1
  vaddr_str = _vreg(inst.vaddr, 2) if inst.offen and inst.idxen else f"v{inst.vaddr}" if inst.offen or inst.idxen else "off"
  srsrc_base = inst.srsrc * 4
  srsrc_str = _reg("ttmp", srsrc_base - 108, 4) if 108 <= srsrc_base <= 123 else _sreg(srsrc_base, 4)
  fmt = inst.format if isinstance(inst, MTBUF) else None
  mods = ([f"format:{fmt}"] if fmt else []) + [m for cond, m in [(inst.idxen, "idxen"), (inst.offen, "offen"), (inst.offset, f"offset:{inst.offset}"), (inst.glc, "glc"), (inst.dlc, "dlc"), (inst.slc, "slc"), (inst.tfe, "tfe")] if cond]
  return f"{name} {_vreg(inst.vdata, width)}, {vaddr_str}, {srsrc_str}, {decode_src(inst.soffset)}" + (" " + " ".join(mods) if mods else "")

def _sop_widths(name: str) -> tuple[int, int, int]:
  """Return (dst_width, src0_width, src1_width) in register count for SOP instructions."""
  if name in ('s_bitset0_b64', 's_bitset1_b64', 's_bfm_b64'): return 2, 1, 1
  if name in ('s_lshl_b64', 's_lshr_b64', 's_ashr_i64', 's_bfe_u64', 's_bfe_i64'): return 2, 2, 1
  if name in ('s_bitcmp0_b64', 's_bitcmp1_b64'): return 1, 2, 1
  if m := re.search(r'_(b|i|u)(32|64)_(b|i|u)(32|64)$', name): return 2 if m.group(2) == '64' else 1, 2 if m.group(4) == '64' else 1, 1
  if m := re.search(r'_(b|i|u)(32|64)$', name): sz = 2 if m.group(2) == '64' else 1; return sz, sz, sz
  return 1, 1, 1

def _disasm_sop1(inst: SOP1) -> str:
  op, name = SOP1Op(inst.op), SOP1Op(inst.op).name.lower()
  if op == SOP1Op.S_GETPC_B64: return f"{name} {_fmt_sdst(inst.sdst, 2)}"
  if op in (SOP1Op.S_SETPC_B64, SOP1Op.S_RFE_B64): return f"{name} {_fmt_src(inst.ssrc0, 2)}"
  if op == SOP1Op.S_SWAPPC_B64: return f"{name} {_fmt_sdst(inst.sdst, 2)}, {_fmt_src(inst.ssrc0, 2)}"
  if op in (SOP1Op.S_SENDMSG_RTN_B32, SOP1Op.S_SENDMSG_RTN_B64): return f"{name} {_fmt_sdst(inst.sdst, 2 if 'b64' in name else 1)}, sendmsg({MSG.get(inst.ssrc0, str(inst.ssrc0))})"
  dn, s0n, _ = _sop_widths(name)
  return f"{name} {_fmt_sdst(inst.sdst, dn)}, {inst.lit(inst.ssrc0) if s0n == 1 else _fmt_src(inst.ssrc0, s0n)}"

def _disasm_sop2(inst: SOP2) -> str:
  name = SOP2Op(inst.op).name.lower()
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
                   VINTERP: _disasm_vinterp, SOPP: _disasm_sopp, SMEM: _disasm_smem, FLAT: _disasm_flat, MUBUF: _disasm_buf, MTBUF: _disasm_buf,
                   SOP1: _disasm_sop1, SOP2: _disasm_sop2, SOPC: _disasm_sopc, SOPK: _disasm_sopk}

def disasm(inst: Inst) -> str: return DISASM_HANDLERS.get(type(inst), _disasm_generic)(inst)

# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLER
# ═══════════════════════════════════════════════════════════════════════════════

SPEC_REGS = {'vcc_lo': RawImm(106), 'vcc_hi': RawImm(107), 'vcc': RawImm(106), 'null': RawImm(124), 'off': RawImm(124), 'm0': RawImm(125),
             'exec_lo': RawImm(126), 'exec_hi': RawImm(127), 'exec': RawImm(126), 'scc': RawImm(253), 'src_scc': RawImm(253)}
FLOATS = {'0.5': 0.5, '-0.5': -0.5, '1.0': 1.0, '-1.0': -1.0, '2.0': 2.0, '-2.0': -2.0, '4.0': 4.0, '-4.0': -4.0}
REG_MAP: dict[str, _RegFactory] = {'s': s, 'v': v, 't': ttmp, 'ttmp': ttmp}
SMEM_OPS = {'s_load_b32', 's_load_b64', 's_load_b128', 's_load_b256', 's_load_b512',
            's_buffer_load_b32', 's_buffer_load_b64', 's_buffer_load_b128', 's_buffer_load_b256', 's_buffer_load_b512'}
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
  reg_prefix = {'s': 's', 'v': 'v', 't': 'ttmp', 'ttmp': 'ttmp'}
  if m := re.match(r'^([svt](?:tmp)?)\[(\d+):(\d+)\]$', lo): return wrap(f"{reg_prefix[m.group(1)]}[{m.group(2)}:{m.group(3)}]")
  if m := re.match(r'^([svt](?:tmp)?)(\d+)$', lo): return wrap(f"{reg_prefix[m.group(1)]}[{m.group(2)}]")
  if re.match(r'^-?\d+$', op) or re.match(r'^-?0x[0-9a-fA-F]+$', op): return f"SrcMod({op}, neg={neg}, abs_={abs_})" if neg or abs_ else op
  if lo.startswith('hwreg(') or lo.startswith('sendmsg('): return wrap(op)
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
    m, text = _extract(text, pat)
    if m: kw.append(f'omod={val}'); break
  m, text = _extract(text, r'\s+clamp(?:\s|$)')
  if m: kw.append('clmp=1')

  opsel = None
  m, text = _extract(text, r'\s+op_sel:\[([^\]]+)\]')
  if m:
    bits = [int(x.strip()) for x in m.group(1).split(',')]
    mn = text.split()[0].lower()
    is3p = mn.startswith(('v_pk_', 'v_wmma_', 'v_dot'))
    opsel = (bits[0] | (bits[1] << 1) | (bits[2] << 2)) if len(bits) == 3 and is3p else (bits[0] | (bits[1] << 1) | (bits[2] << 3)) if len(bits) == 3 else sum(b << i for i, b in enumerate(bits))

  m, text = _extract(text, r'\s+wait_exp:(\d+)')
  if m: kw.append(f'waitexp={m.group(1)}')
  m, text = _extract(text, r'\s+offset:(0x[0-9a-fA-F]+|-?\d+)')
  off_val = m.group(1) if m else None
  m, text = _extract(text, r'\s+dlc(?:\s|$)')
  dlc = 1 if m else None
  m, text = _extract(text, r'\s+glc(?:\s|$)')
  glc = 1 if m else None
  m, text = _extract(text, r'\s+neg_lo:\[([^\]]+)\]')
  neg_lo = sum(int(x.strip()) << i for i, x in enumerate(m.group(1).split(','))) if m else None
  m, text = _extract(text, r'\s+neg_hi:\[([^\]]+)\]')
  neg_hi = sum(int(x.strip()) << i for i, x in enumerate(m.group(1).split(','))) if m else None

  parts = text.replace(',', ' ').split()
  if not parts: raise ValueError("empty instruction")
  mn, op_str = parts[0].lower(), text[len(parts[0]):].strip()

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
    lit = None
    if 'fmaak' in xps[0].lower() and len(xo) > 3: lit = xo[3]
    elif 'fmamk' in xps[0].lower() and len(xo) > 3: lit, vsx1 = xo[2], xo[3]
    elif 'fmaak' in yps[0].lower() and len(yo) > 3: lit = yo[3]
    elif 'fmamk' in yps[0].lower() and len(yo) > 3: lit, vsy1 = yo[2], yo[3]
    ls = f", literal={lit}" if lit else ""
    return f"VOPD(VOPDOp.{xps[0].upper()}, VOPDOp.{yps[0].upper()}, vdstx={vdx}, vdsty={vdy}, srcx0={sx0}, vsrcx1={vsx1}, srcy0={sy0}, vsrcy1={vsy1}{ls})"

  ops = _parse_ops(op_str)
  args = [_op2dsl(o) for o in ops]

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

  # Buffer
  if mn.startswith('buffer_') and len(ops) >= 2 and ops[1].strip().lower() == 'off':
    soff = f"RawImm({args[3].strip()})" if len(args) > 3 else "RawImm(0)"
    return f"{mn}(vdata={args[0]}, vaddr=0, srsrc={args[2]}, soffset={soff})"

  # FLAT/GLOBAL/SCRATCH
  for pre, flds in [('flat_load', 'vdst,addr,saddr'), ('global_load', 'vdst,addr,saddr'), ('scratch_load', 'vdst,addr,saddr'),
                    ('flat_store', 'addr,data,saddr'), ('global_store', 'addr,data,saddr'), ('scratch_store', 'addr,data,saddr')]:
    if mn.startswith(pre) and len(args) >= 3:
      fs = flds.split(',')
      return f"{mn}({fs[0]}={args[0]}, {fs[1]}={args[1]}, {fs[2]}={args[2]}{', offset=' + off_val if off_val else ''})"

  # v_fmaak/v_fmamk
  lit_s = ""
  if mn in ('v_fmaak_f32', 'v_fmaak_f16') and len(args) == 4: lit_s, args = f", literal={args[3].strip()}", args[:3]
  elif mn in ('v_fmamk_f32', 'v_fmamk_f16') and len(args) == 4: lit_s, args = f", literal={args[2].strip()}", [args[0], args[1], args[3]]

  # VCC ops
  vcc_ops = {'v_add_co_ci_u32', 'v_sub_co_ci_u32', 'v_subrev_co_ci_u32'}
  if mn.replace('_e32', '') in vcc_ops and len(args) >= 5: mn, args = mn.replace('_e32', '') + '_e32', [args[0], args[2], args[3]]
  if mn.replace('_e64', '') in vcc_ops and mn.endswith('_e64'): mn = mn.replace('_e64', '')

  # v_cmp strip implicit vcc_lo
  if mn.startswith('v_cmp') and not mn.endswith('_e64') and len(args) >= 3 and ops[0].strip().lower() in ('vcc_lo', 'vcc_hi', 'vcc'):
    args = args[1:]

  # CMPX e64
  if 'cmpx' in mn and mn.endswith('_e64') and len(args) == 2: args = ['RawImm(126)'] + args

  fn = mn.replace('.', '_')
  if opsel is not None: args = [re.sub(r'\.[hl]$', '', a) for a in args]

  all_kw = list(kw)
  if lit_s: all_kw.append(lit_s.lstrip(', '))
  if opsel is not None: all_kw.append(f'opsel={opsel}')
  if neg_lo is not None: all_kw.append(f'neg={neg_lo}')
  if neg_hi is not None: all_kw.append(f'neg_hi={neg_hi}')

  a_str, kw_str = ', '.join(args), ', '.join(all_kw)
  return f"{fn}({a_str}, {kw_str})" if kw_str and a_str else f"{fn}({kw_str})" if kw_str else f"{fn}({a_str})"

def asm(text: str) -> Inst:
  from extra.assembly.amd.autogen import rdna3 as ag
  dsl = get_dsl(text)
  ns = {n: getattr(ag, n) for n in dir(ag) if not n.startswith('_')}
  ns.update({'s': s, 'v': v, 'ttmp': ttmp, 'abs': abs, 'RawImm': RawImm, 'SrcMod': SrcMod, 'VGPR': VGPR, 'SGPR': SGPR, 'TTMP': TTMP,
             'VCC_LO': VCC_LO, 'VCC_HI': VCC_HI, 'VCC': VCC, 'EXEC_LO': EXEC_LO, 'EXEC_HI': EXEC_HI, 'EXEC': EXEC, 'SCC': SCC, 'M0': M0, 'NULL': NULL, 'OFF': OFF})
  try: return eval(dsl, ns)
  except NameError:
    if m := re.match(r'^(v_\w+)(\(.*\))$', dsl): return eval(f"{m.group(1)}_e32{m.group(2)}", ns)
    raise
