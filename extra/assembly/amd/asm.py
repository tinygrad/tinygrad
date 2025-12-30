# RDNA3 assembler and disassembler
from __future__ import annotations
import re
from extra.assembly.amd.dsl import Inst, RawImm, Reg, SrcMod, SGPR, VGPR, TTMP, s, v, ttmp, _RegFactory, FLOAT_ENC, SRC_FIELDS, unwrap
from extra.assembly.amd.dsl import VCC_LO, VCC_HI, VCC, EXEC_LO, EXEC_HI, EXEC, SCC, M0, NULL, OFF

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SPECIAL_GPRS = {106: "vcc_lo", 107: "vcc_hi", 124: "null", 125: "m0", 126: "exec_lo", 127: "exec_hi", 253: "scc"}
SPECIAL_DEC = {**SPECIAL_GPRS, **{v: str(k) for k, v in FLOAT_ENC.items()}}
SPECIAL_PAIRS = {106: "vcc", 126: "exec"}
HWREG_NAMES = {1: 'HW_REG_MODE', 2: 'HW_REG_STATUS', 3: 'HW_REG_TRAPSTS', 4: 'HW_REG_HW_ID', 5: 'HW_REG_GPR_ALLOC',
               6: 'HW_REG_LDS_ALLOC', 7: 'HW_REG_IB_STS', 15: 'HW_REG_SH_MEM_BASES', 18: 'HW_REG_PERF_SNAPSHOT_PC_LO',
               19: 'HW_REG_PERF_SNAPSHOT_PC_HI', 20: 'HW_REG_FLAT_SCR_LO', 21: 'HW_REG_FLAT_SCR_HI',
               22: 'HW_REG_XNACK_MASK', 23: 'HW_REG_HW_ID1', 24: 'HW_REG_HW_ID2', 25: 'HW_REG_POPS_PACKER', 28: 'HW_REG_IB_STS2'}
HWREG_IDS = {v.lower(): k for k, v in HWREG_NAMES.items()}
MSG_NAMES = {128: 'MSG_RTN_GET_DOORBELL', 129: 'MSG_RTN_GET_DDID', 130: 'MSG_RTN_GET_TMA',
             131: 'MSG_RTN_GET_REALTIME', 132: 'MSG_RTN_SAVE_WAVE', 133: 'MSG_RTN_GET_TBA'}
VOP3SD_OPCODES = {288, 289, 290, 764, 765, 766, 767, 768, 769, 770}
_16BIT_TYPES = ('f16', 'i16', 'u16', 'b16')

# ═══════════════════════════════════════════════════════════════════════════════
# DISASSEMBLER HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def decode_src(val: int) -> str:
  if val <= 105: return f"s{val}"
  if val in SPECIAL_DEC: return SPECIAL_DEC[val]
  if 108 <= val <= 123: return f"ttmp{val - 108}"
  if 128 <= val <= 192: return str(val - 128)
  if 193 <= val <= 208: return str(-(val - 192))
  if 256 <= val <= 511: return f"v{val - 256}"
  return "lit" if val == 255 else f"?{val}"

def _reg(prefix: str, base: int, cnt: int = 1) -> str: return f"{prefix}{base}" if cnt == 1 else f"{prefix}[{base}:{base+cnt-1}]"
def _sreg(base: int, cnt: int = 1) -> str: return _reg("s", base, cnt)
def _vreg(base: int, cnt: int = 1) -> str: return _reg("v", base, cnt)

def _fmt_sdst(v: int, cnt: int = 1) -> str:
  if v == 124: return "null"
  if 108 <= v <= 123: return _reg("ttmp", v - 108, cnt)
  if cnt > 1 and v in SPECIAL_PAIRS: return SPECIAL_PAIRS[v]
  if cnt > 1: return _sreg(v, cnt)
  return {126: "exec_lo", 127: "exec_hi", 106: "vcc_lo", 107: "vcc_hi", 125: "m0"}.get(v, f"s{v}")

def _fmt_ssrc(v: int, cnt: int = 1) -> str:
  if cnt == 2:
    if v in SPECIAL_PAIRS: return SPECIAL_PAIRS[v]
    if v <= 105: return _sreg(v, 2)
    if 108 <= v <= 123: return _reg("ttmp", v - 108, 2)
  return decode_src(v)

def _fmt_src_n(v: int, cnt: int) -> str:
  if cnt == 1: return decode_src(v)
  if v >= 256: return _vreg(v - 256, cnt)
  if v <= 105: return _sreg(v, cnt)
  if cnt == 2 and v in SPECIAL_PAIRS: return SPECIAL_PAIRS[v]
  if 108 <= v <= 123: return _reg("ttmp", v - 108, cnt)
  return decode_src(v)

def waitcnt(vmcnt: int = 0x3f, expcnt: int = 0x7, lgkmcnt: int = 0x3f) -> int:
  return (expcnt & 0x7) | ((lgkmcnt & 0x3f) << 4) | ((vmcnt & 0x3f) << 10)

def decode_waitcnt(val: int) -> tuple[int, int, int]:
  return (val >> 10) & 0x3f, val & 0xf, (val >> 4) & 0x3f

# ═══════════════════════════════════════════════════════════════════════════════
# DISASSEMBLER - INSTRUCTION HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

class DisasmCtx:
  """Context for disassembling an instruction."""
  def __init__(self, inst: Inst, op_name: str):
    self.inst, self.op_name = inst, op_name
    self._cache: dict = {}
  def get(self, field: str, default: int = 0) -> int:
    if field not in self._cache: self._cache[field] = unwrap(self.inst._values.get(field, default))
    return self._cache[field]
  def fmt_src(self, v: int) -> str: return f"0x{self.inst._literal:x}" if v == 255 and self.inst._literal is not None else decode_src(v)
  def is_16bit(self) -> bool: return any(t in self.op_name for t in _16BIT_TYPES) and '_f32' not in self.op_name and '_i32' not in self.op_name
  def is_64bit(self) -> bool: return any(t in self.op_name for t in ('f64', 'i64', 'u64', 'b64'))
  def omod_str(self) -> str: return {1: " mul:2", 2: " mul:4", 3: " div:2"}.get(self.get('omod'), "")

def _disasm_vop1(ctx: DisasmCtx) -> str:
  vdst, src0, op = ctx.get('vdst'), ctx.get('src0'), ctx.op_name
  if op in ('v_nop', 'v_pipeflush'): return op
  parts = op.split('_')
  is_16bit_dst = any(p in _16BIT_TYPES for p in parts[-2:-1]) or (len(parts) >= 2 and parts[-1] in _16BIT_TYPES and 'cvt' not in op)
  is_16bit_src = parts[-1] in _16BIT_TYPES and 'sat_pk' not in op
  _F64_OPS = ('v_ceil_f64', 'v_floor_f64', 'v_fract_f64', 'v_frexp_mant_f64', 'v_rcp_f64', 'v_rndne_f64', 'v_rsq_f64', 'v_sqrt_f64', 'v_trunc_f64')
  is_f64_dst = op in _F64_OPS or op in ('v_cvt_f64_f32', 'v_cvt_f64_i32', 'v_cvt_f64_u32')
  is_f64_src = op in _F64_OPS or op in ('v_cvt_f32_f64', 'v_cvt_i32_f64', 'v_cvt_u32_f64', 'v_frexp_exp_i32_f64')
  if op == 'v_readfirstlane_b32': return f"v_readfirstlane_b32 {decode_src(vdst)}, v{src0 - 256 if src0 >= 256 else src0}"
  dst = _vreg(vdst, 2) if is_f64_dst else f"v{vdst & 0x7f}.{'h' if vdst >= 128 else 'l'}" if is_16bit_dst else f"v{vdst}"
  src = _fmt_src_n(src0, 2) if is_f64_src else f"v{(src0 - 256) & 0x7f}.{'h' if src0 >= 384 else 'l'}" if is_16bit_src and src0 >= 256 else ctx.fmt_src(src0)
  return f"{op}_e32 {dst}, {src}"

def _disasm_vop2(ctx: DisasmCtx) -> str:
  vdst, src0, vsrc1, op = ctx.get('vdst'), ctx.get('src0'), ctx.get('vsrc1'), ctx.op_name
  suffix = "" if op == "v_dot2acc_f32_f16" else "_e32"
  is_16bit = ctx.is_16bit() and 'pk_' not in op
  if is_16bit:
    dst = f"v{vdst & 0x7f}.{'h' if vdst >= 128 else 'l'}"
    s0 = f"v{(src0 - 256) & 0x7f}.{'h' if src0 >= 384 else 'l'}" if src0 >= 256 else ctx.fmt_src(src0)
    s1 = f"v{vsrc1 & 0x7f}.{'h' if vsrc1 >= 128 else 'l'}"
  else:
    dst, s0, s1 = f"v{vdst}", ctx.fmt_src(src0), f"v{vsrc1}"
  return f"{op}{suffix} {dst}, {s0}, {s1}" + (", vcc_lo" if op == "v_cndmask_b32" else "")

def _disasm_vopc(ctx: DisasmCtx) -> str:
  src0, vsrc1, op = ctx.get('src0'), ctx.get('vsrc1'), ctx.op_name
  is_64bit = ctx.is_64bit()
  is_64bit_vsrc1 = is_64bit and 'class' not in op
  is_16bit = ctx.is_16bit()
  is_cmpx = op.startswith('v_cmpx')
  s0 = _fmt_src_n(src0, 2) if is_64bit else f"v{(src0 - 256) & 0x7f}.{'h' if src0 >= 384 else 'l'}" if is_16bit and src0 >= 256 else ctx.fmt_src(src0)
  s1 = _vreg(vsrc1, 2) if is_64bit_vsrc1 else f"v{vsrc1 & 0x7f}.{'h' if vsrc1 >= 128 else 'l'}" if is_16bit else f"v{vsrc1}"
  return f"{op}_e32 {s0}, {s1}" if is_cmpx else f"{op}_e32 vcc_lo, {s0}, {s1}"

def _disasm_sopp(ctx: DisasmCtx) -> str:
  simm16, op = ctx.get('simm16'), ctx.op_name
  no_imm = ('s_endpgm', 's_barrier', 's_wakeup', 's_icache_inv', 's_ttracedata', 's_ttracedata_imm',
            's_wait_idle', 's_endpgm_saved', 's_code_end', 's_endpgm_ordered_ps_done')
  if op in no_imm: return op
  if op == 's_waitcnt':
    vmcnt, expcnt, lgkmcnt = decode_waitcnt(simm16)
    parts = [f"vmcnt({vmcnt})" if vmcnt != 0x3f else "", f"expcnt({expcnt})" if expcnt != 0x7 else "", f"lgkmcnt({lgkmcnt})" if lgkmcnt != 0x3f else ""]
    return f"s_waitcnt {' '.join(p for p in parts if p)}" if any(parts) else "s_waitcnt 0"
  if op == 's_delay_alu':
    deps = ['VALU_DEP_1','VALU_DEP_2','VALU_DEP_3','VALU_DEP_4','TRANS32_DEP_1','TRANS32_DEP_2','TRANS32_DEP_3','FMA_ACCUM_CYCLE_1','SALU_CYCLE_1','SALU_CYCLE_2','SALU_CYCLE_3']
    skips = ['SAME','NEXT','SKIP_1','SKIP_2','SKIP_3','SKIP_4']
    id0, skip, id1 = simm16 & 0xf, (simm16 >> 4) & 0x7, (simm16 >> 7) & 0xf
    dep = lambda v: deps[v-1] if 0 < v <= len(deps) else str(v)
    parts = [f"instid0({dep(id0)})" if id0 else "", f"instskip({skips[skip]})" if skip else "", f"instid1({dep(id1)})" if id1 else ""]
    return f"s_delay_alu {' | '.join(p for p in parts if p)}" if any(parts) else "s_delay_alu 0"
  if op.startswith('s_cbranch') or op.startswith('s_branch'): return f"{op} {simm16}"
  return f"{op} 0x{simm16:x}"

def _disasm_smem(ctx: DisasmCtx) -> str:
  op, op_val = ctx.op_name, ctx.get('op')
  if op in ('s_gl1_inv', 's_dcache_inv'): return op
  sdata, sbase, soffset, offset = ctx.get('sdata'), ctx.get('sbase'), ctx.get('soffset'), ctx.get('offset')
  glc, dlc = ctx.get('glc'), ctx.get('dlc')
  off = f"{decode_src(soffset)} offset:0x{offset:x}" if offset and soffset != 124 else f"0x{offset:x}" if offset else decode_src(soffset)
  sbase_idx, sbase_cnt = sbase * 2, 4 if (8 <= op_val <= 12 or op == 's_atc_probe_buffer') else 2
  sb = _fmt_ssrc(sbase_idx, sbase_cnt) if sbase_cnt == 2 else _sreg(sbase_idx, sbase_cnt) if sbase_idx <= 105 else _reg("ttmp", sbase_idx - 108, sbase_cnt)
  if op in ('s_atc_probe', 's_atc_probe_buffer'): return f"{op} {sdata}, {sb}, {off}"
  width = {0:1, 1:2, 2:4, 3:8, 4:16, 8:1, 9:2, 10:4, 11:8, 12:16}.get(op_val, 1)
  mods = " ".join(m for m in ["glc" if glc else "", "dlc" if dlc else ""] if m)
  return f"{op} {_fmt_sdst(sdata, width)}, {sb}, {off}" + (" " + mods if mods else "")

def _disasm_flat(ctx: DisasmCtx) -> str:
  vdst, addr, data, saddr, offset, seg = [ctx.get(f) for f in ('vdst', 'addr', 'data', 'saddr', 'offset', 'seg')]
  op = ctx.op_name
  instr = f"{['flat', 'scratch', 'global'][seg] if seg < 3 else 'flat'}_{op.split('_', 1)[1] if '_' in op else op}"
  width = {'b32':1, 'b64':2, 'b96':3, 'b128':4, 'u8':1, 'i8':1, 'u16':1, 'i16':1}.get(op.split('_')[-1], 1)
  addr_s = _vreg(addr, 2) if saddr == 0x7F else _vreg(addr)
  saddr_s = "" if saddr == 0x7F else f", {_sreg(saddr, 2)}" if saddr < 106 else ", off" if saddr == 124 else f", {decode_src(saddr)}"
  off_s = f" offset:{offset}" if offset else ""
  vdata = _vreg(data if 'store' in op else vdst, width)
  return f"{instr} {addr_s}, {vdata}{saddr_s}{off_s}" if 'store' in op else f"{instr} {vdata}, {addr_s}{saddr_s}{off_s}"

def _disasm_vop3(ctx: DisasmCtx) -> str:
  op, op_val = ctx.op_name, ctx.get('op')
  vdst = ctx.get('vdst')
  src0, src1, src2 = ctx.get('src0'), ctx.get('src1'), ctx.get('src2')
  neg, abs_, clmp, opsel, omod = ctx.get('neg'), ctx.get('abs'), ctx.get('clmp'), ctx.get('opsel'), ctx.get('omod')

  # VOP3SD handling (shared encoding)
  if op_val in VOP3SD_OPCODES:
    sdst = (clmp << 7) | (opsel << 3) | abs_
    is_f64, is_mad64 = 'f64' in op, 'mad_i64_i32' in op or 'mad_u64_u32' in op
    def fmt_sd(v, neg_bit, is_64=False):
      s = _fmt_src_n(v, 2) if (is_64 or is_f64) else ctx.fmt_src(v)
      return f"-{s}" if neg_bit else s
    s0, s1, s2 = fmt_sd(src0, neg & 1), fmt_sd(src1, neg & 2), fmt_sd(src2, neg & 4, is_mad64)
    dst = _vreg(vdst, 2) if (is_f64 or is_mad64) else f"v{vdst}"
    omod_s = {1: " mul:2", 2: " mul:4", 3: " div:2"}.get(omod, "")
    if op in ('v_add_co_u32', 'v_sub_co_u32', 'v_subrev_co_u32'): return f"{op} {dst}, {_fmt_sdst(sdst, 1)}, {s0}, {s1}"
    if op in ('v_add_co_ci_u32', 'v_sub_co_ci_u32', 'v_subrev_co_ci_u32'): return f"{op} {dst}, {_fmt_sdst(sdst, 1)}, {s0}, {s1}, {s2}"
    return f"{op} {dst}, {_fmt_sdst(sdst, 1)}, {s0}, {s1}, {s2}" + omod_s

  # Regular VOP3
  is_f64 = ctx.is_64bit()
  is_class, is_shift64 = 'class' in op, 'rev' in op and '64' in op and op.startswith('v_')
  is_ldexp64, is_trig_preop = op == 'v_ldexp_f64', op == 'v_trig_preop_f64'
  is_readlane = op == 'v_readlane_b32'
  is_sad64, is_mqsad_u32 = any(x in op for x in ('qsad_pk', 'mqsad_pk')), 'mqsad_u32' in op

  # Detect 16/64-bit operands for CVT and other mixed-precision ops
  is_f64_src, is_f64_dst = False, False
  if 'cvt_pk' in op: is_f16_dst, is_f16_src, is_f16_src2 = False, op.endswith('16'), False
  elif m := re.match(r'v_(?:cvt|frexp_exp)_([a-z0-9_]+)_([a-z0-9]+)', op):
    is_f16_dst, is_f16_src = any(t in m.group(1) for t in _16BIT_TYPES), any(t in m.group(2) for t in _16BIT_TYPES)
    is_f64_src = '64' in m.group(2)  # source is 64-bit (e.g. v_cvt_u32_f64, v_frexp_exp_i32_f64)
    is_f64_dst = '64' in m.group(1)  # dest is 64-bit (e.g. v_cvt_f64_u32)
    is_f16_src2, is_f64 = is_f16_src, False
  elif re.match(r'v_mad_[iu]32_[iu]16', op): is_f16_dst, is_f16_src, is_f16_src2 = False, True, False
  elif 'pack_b32' in op: is_f16_dst, is_f16_src, is_f16_src2 = False, True, True
  else:
    is_16bit = ctx.is_16bit() and not any(x in op for x in ('dot2', 'pk_', 'sad', 'msad', 'qsad', 'mqsad'))
    is_f16_dst = is_f16_src = is_f16_src2 = is_16bit

  any_hi = opsel != 0
  def fmt_vop3(v, neg_bit, abs_bit, hi_bit=False, cnt=1, is_16=False):
    if cnt > 1: s = _fmt_src_n(v, cnt)
    elif is_16 and v >= 256: s = f"v{v - 256}.h" if hi_bit else f"v{v - 256}.l" if any_hi else ctx.fmt_src(v)
    else: s = ctx.fmt_src(v)
    if abs_bit: s = f"|{s}|"
    return f"-{s}" if neg_bit else s

  # Source register counts
  is_src0_64 = (is_f64 and not is_shift64) or is_sad64 or is_mqsad_u32 or is_f64_src
  is_src1_64 = is_f64 and not is_class and not is_ldexp64 and not is_trig_preop
  src0_cnt, src1_cnt = 2 if is_src0_64 else 1, 2 if is_src1_64 else 1
  src2_cnt = 4 if is_mqsad_u32 else 2 if (is_f64 or is_sad64) else 1

  s0 = fmt_vop3(src0, neg & 1, abs_ & 1, opsel & 1, src0_cnt, is_f16_src)
  s1 = fmt_vop3(src1, neg & 2, abs_ & 2, opsel & 2, src1_cnt, is_f16_src)
  s2 = fmt_vop3(src2, neg & 4, abs_ & 4, opsel & 4, src2_cnt, is_f16_src2)

  # Destination
  cvt_dst_64 = is_f64_dst  # CVT ops with 64-bit destination
  is_dst_64 = is_f64 or is_sad64 or cvt_dst_64
  dst_cnt = 4 if is_mqsad_u32 else 2 if is_dst_64 else 1
  if is_readlane: dst = _fmt_sdst(vdst, 1)
  elif dst_cnt > 1: dst = _vreg(vdst, dst_cnt)
  elif is_f16_dst: dst = f"v{vdst}.h" if (opsel & 8) else f"v{vdst}.l" if any_hi else f"v{vdst}"
  else: dst = f"v{vdst}"

  clamp_s, omod_s = " clamp" if clmp else "", ctx.omod_str()
  has_nonvgpr_opsel = (src0 < 256 and (opsel & 1)) or (src1 < 256 and (opsel & 2)) or (src2 < 256 and (opsel & 4))
  need_opsel = has_nonvgpr_opsel or (opsel and not is_f16_src)

  def fmt_opsel(n):
    if not need_opsel: return ""
    if is_f16_dst and (opsel & 8): return f" op_sel:[1,1,1{',1' if n == 3 else ''}]"
    if n == 3: return f" op_sel:[{opsel & 1},{(opsel >> 1) & 1},{(opsel >> 2) & 1},{(opsel >> 3) & 1}]"
    return f" op_sel:[{opsel & 1},{(opsel >> 1) & 1},{(opsel >> 2) & 1}]"

  # Dispatch by opcode range
  if op_val < 256:  # VOPC promoted
    return f"{op}_e64 {s0}, {s1}" if op.startswith('v_cmpx') else f"{op}_e64 {_fmt_sdst(vdst, 1)}, {s0}, {s1}"
  if op_val < 384:  # VOP2 promoted
    if 'cndmask' in op: return f"{op}_e64 {dst}, {s0}, {s1}, {s2}" + fmt_opsel(3) + clamp_s + omod_s
    return f"{op}_e64 {dst}, {s0}, {s1}" + fmt_opsel(2) + clamp_s + omod_s
  if op_val < 512:  # VOP1 promoted
    if op in ('v_nop', 'v_pipeflush'): return f"{op}_e64"
    return f"{op}_e64 {dst}, {s0}" + fmt_opsel(1) + clamp_s + omod_s
  # Native VOP3
  is_3src = any(x in op for x in ('fma', 'mad', 'min3', 'max3', 'med3', 'div_fix', 'div_fmas', 'sad', 'lerp', 'align', 'cube',
                                   'bfe', 'bfi', 'perm_b32', 'permlane', 'cndmask', 'xor3', 'or3', 'add3', 'lshl_or', 'and_or',
                                   'lshl_add', 'add_lshl', 'xad', 'maxmin', 'minmax', 'dot2', 'cvt_pk_u8', 'mullit'))
  if is_3src: return f"{op} {dst}, {s0}, {s1}, {s2}" + fmt_opsel(3) + clamp_s + omod_s
  return f"{op} {dst}, {s0}, {s1}" + fmt_opsel(2) + clamp_s + omod_s

def _disasm_vop3sd(ctx: DisasmCtx) -> str:
  op = ctx.op_name
  vdst, sdst = ctx.get('vdst'), ctx.get('sdst')
  src0, src1, src2 = ctx.get('src0'), ctx.get('src1'), ctx.get('src2')
  neg, omod, clmp = ctx.get('neg'), ctx.get('omod'), ctx.get('clmp')
  is_f64, is_mad64 = 'f64' in op, 'mad_i64_i32' in op or 'mad_u64_u32' in op
  def fmt_neg(v, neg_bit, is_64=False):
    s = _fmt_src_n(v, 2) if (is_64 or is_f64) else ctx.fmt_src(v)
    return f"-{s}" if neg_bit else s
  srcs = [fmt_neg(src0, neg & 1), fmt_neg(src1, neg & 2), fmt_neg(src2, neg & 4, is_mad64)]
  dst = _vreg(vdst, 2) if (is_f64 or is_mad64) else f"v{vdst}"
  clamp_s, omod_s = " clamp" if clmp else "", ctx.omod_str()
  is_2src = op in ('v_add_co_u32', 'v_sub_co_u32', 'v_subrev_co_u32')
  suffix = "_e64" if op.startswith('v_') and 'co_' in op else ""
  return f"{op}{suffix} {dst}, {_fmt_sdst(sdst, 1)}, {', '.join(srcs[:2] if is_2src else srcs)}" + clamp_s + omod_s

def _disasm_vopd(ctx: DisasmCtx) -> str:
  from extra.assembly.amd.autogen import rdna3 as autogen
  opx, opy, vdstx, vdsty_enc = [ctx.get(f) for f in ('opx', 'opy', 'vdstx', 'vdsty')]
  srcx0, vsrcx1, srcy0, vsrcy1 = [ctx.get(f) for f in ('srcx0', 'vsrcx1', 'srcy0', 'vsrcy1')]
  literal = ctx.inst._literal if hasattr(ctx.inst, '_literal') and ctx.inst._literal else ctx.get('literal')
  vdsty = (vdsty_enc << 1) | ((vdstx & 1) ^ 1)
  def fmt(op, vdst, src0, vsrc1, with_lit):
    try: name = autogen.VOPDOp(op).name.lower()
    except (ValueError, KeyError): name = f"op_{op}"
    lit = f", 0x{literal:x}" if with_lit and literal is not None and ('fmaak' in name or 'fmamk' in name) else ""
    return f"{name} v{vdst}, {ctx.fmt_src(src0)}{lit}" if 'mov' in name else f"{name} v{vdst}, {ctx.fmt_src(src0)}, v{vsrc1}{lit}"
  x_lit = 'fmaak' in autogen.VOPDOp(opx).name.lower() or 'fmamk' in autogen.VOPDOp(opx).name.lower()
  y_lit = 'fmaak' in autogen.VOPDOp(opy).name.lower() or 'fmamk' in autogen.VOPDOp(opy).name.lower()
  return f"{fmt(opx, vdstx, srcx0, vsrcx1, x_lit)} :: {fmt(opy, vdsty, srcy0, vsrcy1, y_lit)}"

def _disasm_vop3p(ctx: DisasmCtx) -> str:
  op = ctx.op_name
  vdst, clmp = ctx.get('vdst'), ctx.get('clmp')
  src0, src1, src2 = ctx.get('src0'), ctx.get('src1'), ctx.get('src2')
  neg, neg_hi = ctx.get('neg'), ctx.get('neg_hi')
  opsel, opsel_hi, opsel_hi2 = ctx.get('opsel'), ctx.get('opsel_hi'), ctx.get('opsel_hi2')
  is_wmma, is_3src = 'wmma' in op, any(x in op for x in ('fma', 'mad', 'dot', 'wmma'))
  def fmt_bits(name, val, n): return f"{name}:[{','.join(str((val >> i) & 1) for i in range(n))}]"
  if is_wmma:
    src_cnt = 2 if 'iu4' in op else 4 if 'iu8' in op else 8
    s0, s1, s2, dst = _fmt_src_n(src0, src_cnt), _fmt_src_n(src1, src_cnt), _fmt_src_n(src2, 8), _vreg(vdst, 8)
  else:
    s0, s1, s2, dst = _fmt_src_n(src0, 1), _fmt_src_n(src1, 1), _fmt_src_n(src2, 1), f"v{vdst}"
  n = 3 if is_3src else 2
  full_opsel_hi = opsel_hi | (opsel_hi2 << 2)
  mods = [fmt_bits("op_sel", opsel, n)] if opsel else []
  if full_opsel_hi != (0b111 if is_3src else 0b11): mods.append(fmt_bits("op_sel_hi", full_opsel_hi, n))
  if neg: mods.append(fmt_bits("neg_lo", neg, n))
  if neg_hi: mods.append(fmt_bits("neg_hi", neg_hi, n))
  if clmp: mods.append("clamp")
  mod_s = " " + " ".join(mods) if mods else ""
  return f"{op} {dst}, {s0}, {s1}, {s2}{mod_s}" if is_3src else f"{op} {dst}, {s0}, {s1}{mod_s}"

def _disasm_vinterp(ctx: DisasmCtx) -> str:
  op = ctx.op_name
  vdst = ctx.get('vdst')
  src0, src1, src2 = ctx.get('src0'), ctx.get('src1'), ctx.get('src2')
  neg, waitexp, clmp = ctx.get('neg'), ctx.get('waitexp'), ctx.get('clmp')
  def fmt_neg(v, i): s = f"v{v - 256}" if v >= 256 else ctx.fmt_src(v); return f"-{s}" if neg & (1 << i) else s
  srcs = [fmt_neg(src0, 0), fmt_neg(src1, 1), fmt_neg(src2, 2)]
  mods = " ".join(m for m in [f"wait_exp:{waitexp}" if waitexp else "", "clamp" if clmp else ""] if m)
  return f"{op} v{vdst}, {', '.join(srcs)}" + (" " + mods if mods else "")

def _disasm_mubuf(ctx: DisasmCtx) -> str:
  op = ctx.op_name
  if op in ('buffer_gl0_inv', 'buffer_gl1_inv'): return op
  vdata, vaddr, srsrc, soffset = [ctx.get(f) for f in ('vdata', 'vaddr', 'srsrc', 'soffset')]
  offset, offen, idxen = ctx.get('offset'), ctx.get('offen'), ctx.get('idxen')
  glc, dlc, slc, tfe = ctx.get('glc'), ctx.get('dlc'), ctx.get('slc'), ctx.get('tfe')
  if 'd16' in op: width = 2 if any(x in op for x in ('xyz', 'xyzw')) else 1
  elif 'atomic' in op:
    base = 2 if any(x in op for x in ('b64', 'u64', 'i64')) else 1
    width = base * 2 if 'cmpswap' in op else base
  else: width = {'b32':1, 'b64':2, 'b96':3, 'b128':4, 'b16':1, 'x':1, 'xy':2, 'xyz':3, 'xyzw':4}.get(op.split('_')[-1], 1)
  if tfe: width += 1
  vaddr_s = _vreg(vaddr, 2) if offen and idxen else f"v{vaddr}" if offen or idxen else "off"
  srsrc_base = srsrc * 4
  srsrc_s = _reg("ttmp", srsrc_base - 108, 4) if 108 <= srsrc_base <= 123 else _sreg(srsrc_base, 4)
  mods = " ".join(m for m in ["offen" if offen else "", "idxen" if idxen else "", f"offset:{offset}" if offset else "",
                              "glc" if glc else "", "dlc" if dlc else "", "slc" if slc else "", "tfe" if tfe else ""] if m)
  return f"{op} {_vreg(vdata, width)}, {vaddr_s}, {srsrc_s}, {decode_src(soffset)}" + (" " + mods if mods else "")

def _disasm_mtbuf(ctx: DisasmCtx) -> str:
  op = ctx.op_name
  vdata, vaddr, srsrc, soffset = [ctx.get(f) for f in ('vdata', 'vaddr', 'srsrc', 'soffset')]
  offset, tbuf_fmt, offen, idxen = ctx.get('offset'), ctx.get('format'), ctx.get('offen'), ctx.get('idxen')
  glc, dlc, slc = ctx.get('glc'), ctx.get('dlc'), ctx.get('slc')
  vaddr_s = _vreg(vaddr, 2) if offen and idxen else f"v{vaddr}" if offen or idxen else "off"
  srsrc_base = srsrc * 4
  srsrc_s = _reg("ttmp", srsrc_base - 108, 4) if 108 <= srsrc_base <= 123 else _sreg(srsrc_base, 4)
  width = 2 if 'd16' in op and any(x in op for x in ('xyz', 'xyzw')) else 1 if 'd16' in op else {'x':1, 'xy':2, 'xyz':3, 'xyzw':4}.get(op.split('_')[-1], 1)
  mods = " ".join([f"format:{tbuf_fmt}"] + [m for m in ["idxen" if idxen else "", "offen" if offen else "", f"offset:{offset}" if offset else "",
                                                        "glc" if glc else "", "dlc" if dlc else "", "slc" if slc else ""] if m])
  return f"{op} {_vreg(vdata, width)}, {vaddr_s}, {srsrc_s}, {decode_src(soffset)} {mods}"

def _parse_sop_sizes(op_name: str) -> tuple[int, ...]:
  if op_name in ('s_bitset0_b64', 's_bitset1_b64'): return (2, 1)
  if op_name in ('s_lshl_b64', 's_lshr_b64', 's_ashr_i64', 's_bfe_u64', 's_bfe_i64'): return (2, 2, 1)
  if op_name in ('s_bfm_b64',): return (2, 1, 1)
  if op_name in ('s_bitcmp0_b64', 's_bitcmp1_b64'): return (1, 2, 1)
  if m := re.search(r'_(b|i|u)(32|64)_(b|i|u)(32|64)$', op_name): return (2 if m.group(2) == '64' else 1, 2 if m.group(4) == '64' else 1)
  if m := re.search(r'_(b|i|u)(32|64)$', op_name): sz = 2 if m.group(2) == '64' else 1; return (sz, sz)
  return (1, 1)

def _disasm_sop(ctx: DisasmCtx) -> str:
  cls_name, op = ctx.inst.__class__.__name__, ctx.op_name
  sizes = _parse_sop_sizes(op)
  dst_cnt, src0_cnt = sizes[0], sizes[1]
  src1_cnt = sizes[2] if len(sizes) > 2 else src0_cnt
  if cls_name == 'SOP1':
    sdst, ssrc0 = ctx.get('sdst'), ctx.get('ssrc0')
    if op == 's_getpc_b64': return f"{op} {_fmt_sdst(sdst, 2)}"
    if op in ('s_setpc_b64', 's_rfe_b64'): return f"{op} {_fmt_ssrc(ssrc0, 2)}"
    if op == 's_swappc_b64': return f"{op} {_fmt_sdst(sdst, 2)}, {_fmt_ssrc(ssrc0, 2)}"
    if op in ('s_sendmsg_rtn_b32', 's_sendmsg_rtn_b64'):
      return f"{op} {_fmt_sdst(sdst, 2 if 'b64' in op else 1)}, sendmsg({MSG_NAMES.get(ssrc0, str(ssrc0))})"
    return f"{op} {_fmt_sdst(sdst, dst_cnt)}, {ctx.fmt_src(ssrc0) if src0_cnt == 1 else _fmt_ssrc(ssrc0, src0_cnt)}"
  if cls_name == 'SOP2':
    sdst, ssrc0, ssrc1 = ctx.get('sdst'), ctx.get('ssrc0'), ctx.get('ssrc1')
    s0 = ctx.fmt_src(ssrc0) if ssrc0 == 255 else _fmt_ssrc(ssrc0, src0_cnt)
    s1 = ctx.fmt_src(ssrc1) if ssrc1 == 255 else _fmt_ssrc(ssrc1, src1_cnt)
    return f"{op} {_fmt_sdst(sdst, dst_cnt)}, {s0}, {s1}"
  if cls_name == 'SOPC':
    return f"{op} {_fmt_ssrc(ctx.get('ssrc0'), src0_cnt)}, {_fmt_ssrc(ctx.get('ssrc1'), src1_cnt)}"
  if cls_name == 'SOPK':
    sdst, simm16 = ctx.get('sdst'), ctx.get('simm16')
    if op == 's_version': return f"{op} 0x{simm16:x}"
    if op in ('s_setreg_b32', 's_getreg_b32'):
      hwreg_id, hwreg_offset, hwreg_size = simm16 & 0x3f, (simm16 >> 6) & 0x1f, ((simm16 >> 11) & 0x1f) + 1
      hwreg_s = f"0x{simm16:x}" if hwreg_id in (16, 17) else f"hwreg({HWREG_NAMES.get(hwreg_id, str(hwreg_id))}, {hwreg_offset}, {hwreg_size})"
      return f"{op} {hwreg_s}, {_fmt_sdst(sdst, 1)}" if op == 's_setreg_b32' else f"{op} {_fmt_sdst(sdst, 1)}, {hwreg_s}"
    return f"{op} {_fmt_sdst(sdst, dst_cnt)}, 0x{simm16:x}"
  return op

def _disasm_generic(ctx: DisasmCtx) -> str:
  def fmt(n, v):
    v = unwrap(v)
    if n in SRC_FIELDS: return ctx.fmt_src(v) if v != 255 else "0xff"
    if n in ('sdst', 'vdst'): return f"{'s' if n == 'sdst' else 'v'}{v}"
    return f"v{v}" if n == 'vsrc1' else f"0x{v:x}" if n == 'simm16' else str(v)
  ops = [fmt(n, ctx.inst._values.get(n, 0)) for n in ctx.inst._fields if n not in ('encoding', 'op')]
  return f"{ctx.op_name} {', '.join(ops)}" if ops else ctx.op_name

DISASM_HANDLERS = {
  'VOP1': _disasm_vop1, 'VOP2': _disasm_vop2, 'VOPC': _disasm_vopc, 'VOP3': _disasm_vop3, 'VOP3SD': _disasm_vop3sd,
  'VOPD': _disasm_vopd, 'VOP3P': _disasm_vop3p, 'VINTERP': _disasm_vinterp,
  'SOPP': _disasm_sopp, 'SMEM': _disasm_smem, 'FLAT': _disasm_flat,
  'MUBUF': _disasm_mubuf, 'MTBUF': _disasm_mtbuf,
  'SOP1': _disasm_sop, 'SOP2': _disasm_sop, 'SOPC': _disasm_sop, 'SOPK': _disasm_sop,
}

def disasm(inst: Inst) -> str:
  op_val = unwrap(inst._values.get('op', 0))
  cls_name = inst.__class__.__name__
  is_vop3sd = cls_name == 'VOP3' and op_val in VOP3SD_OPCODES
  try:
    from extra.assembly.amd.autogen import rdna3 as autogen
    if is_vop3sd: op_name = autogen.VOP3SDOp(op_val).name.lower()
    else: op_name = getattr(autogen, f"{cls_name}Op")(op_val).name.lower() if hasattr(autogen, f"{cls_name}Op") else f"op_{op_val}"
  except (ValueError, KeyError): op_name = f"op_{op_val}"
  ctx = DisasmCtx(inst, op_name)
  handler = DISASM_HANDLERS.get(cls_name, _disasm_generic)
  return handler(ctx)

# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLER
# ═══════════════════════════════════════════════════════════════════════════════

SPECIAL_REGS = {'vcc_lo': RawImm(106), 'vcc_hi': RawImm(107), 'vcc': RawImm(106), 'null': RawImm(124), 'off': RawImm(124), 'm0': RawImm(125),
                'exec_lo': RawImm(126), 'exec_hi': RawImm(127), 'exec': RawImm(126), 'scc': RawImm(253), 'src_scc': RawImm(253)}
FLOAT_CONSTS = {'0.5': 0.5, '-0.5': -0.5, '1.0': 1.0, '-1.0': -1.0, '2.0': 2.0, '-2.0': -2.0, '4.0': 4.0, '-4.0': -4.0}
REG_MAP: dict[str, _RegFactory] = {'s': s, 'v': v, 't': ttmp, 'ttmp': ttmp}
SMEM_OPS = {'s_load_b32', 's_load_b64', 's_load_b128', 's_load_b256', 's_load_b512',
            's_buffer_load_b32', 's_buffer_load_b64', 's_buffer_load_b128', 's_buffer_load_b256', 's_buffer_load_b512'}

def parse_operand(op: str) -> tuple:
  op = op.strip().lower()
  neg = op.startswith('-') and not op[1:2].isdigit(); op = op[1:] if neg else op
  abs_ = op.startswith('|') and op.endswith('|') or op.startswith('abs(') and op.endswith(')')
  op = op[1:-1] if op.startswith('|') else op[4:-1] if op.startswith('abs(') else op
  hi_half = op.endswith('.h')
  op = re.sub(r'\.[lh]$', '', op)
  if op in FLOAT_CONSTS: return (FLOAT_CONSTS[op], neg, abs_, hi_half)
  if re.match(r'^-?\d+$', op): return (int(op), neg, abs_, hi_half)
  if m := re.match(r'^-?0x([0-9a-f]+)$', op):
    v = -int(m.group(1), 16) if op.startswith('-') else int(m.group(1), 16)
    return (v, neg, abs_, hi_half)
  if op in SPECIAL_REGS: return (SPECIAL_REGS[op], neg, abs_, hi_half)
  if op == 'lit': return (RawImm(255), neg, abs_, hi_half)
  if m := re.match(r'^([svt](?:tmp)?)\[(\d+):(\d+)\]$', op): return (REG_MAP[m.group(1)][int(m.group(2)):int(m.group(3))], neg, abs_, hi_half)
  if m := re.match(r'^([svt](?:tmp)?)(\d+)$', op):
    reg = REG_MAP[m.group(1)][int(m.group(2))]
    reg.hi = hi_half
    return (reg, neg, abs_, hi_half)
  if m := re.match(r'^hwreg\((\w+)(?:,\s*(\d+),\s*(\d+))?\)$', op):
    name_str = m.group(1).lower()
    hwreg_id = HWREG_IDS.get(name_str, int(name_str) if name_str.isdigit() else None)
    if hwreg_id is None: raise ValueError(f"unknown hwreg name: {name_str}")
    offset, size = int(m.group(2)) if m.group(2) else 0, int(m.group(3)) if m.group(3) else 32
    return (((size - 1) << 11) | (offset << 6) | hwreg_id, neg, abs_, hi_half)
  raise ValueError(f"cannot parse operand: {op}")

def _operand_to_dsl(op: str) -> str:
  op = op.strip()
  neg = op.startswith('-') and not (op[1:2].isdigit() or (len(op) > 2 and op[1] == '0' and op[2] in 'xX'))
  if neg: op = op[1:]
  abs_ = op.startswith('|') and op.endswith('|') or op.startswith('abs(') and op.endswith(')')
  if abs_: op = op[1:-1] if op.startswith('|') else op[4:-1]
  hi_suffix = ""
  if op.endswith('.h'): hi_suffix, op = ".h", op[:-2]
  elif op.endswith('.l'): hi_suffix, op = ".l", op[:-2]
  op_lower = op.lower()

  def apply(base: str) -> str:
    if not neg and not abs_: return f"{base}{hi_suffix}"
    if abs_: return f"{'-' if neg else ''}abs({base}){hi_suffix}"
    return f"-{base}{hi_suffix}"

  special = {'vcc_lo': 'VCC_LO', 'vcc_hi': 'VCC_HI', 'vcc': 'VCC_LO', 'null': 'NULL', 'off': 'OFF', 'm0': 'M0',
             'exec_lo': 'EXEC_LO', 'exec_hi': 'EXEC_HI', 'exec': 'EXEC_LO', 'scc': 'SCC', 'src_scc': 'SCC'}
  if op_lower in special: return apply(special[op_lower])
  floats = {'0.5': '0.5', '-0.5': '-0.5', '1.0': '1.0', '-1.0': '-1.0', '2.0': '2.0', '-2.0': '-2.0', '4.0': '4.0', '-4.0': '-4.0'}
  if op in floats: return apply(floats[op])
  if m := re.match(r'^([svt](?:tmp)?)\[(\d+):(\d+)\]$', op_lower):
    prefix = {'s': 's', 'v': 'v', 't': 'ttmp', 'ttmp': 'ttmp'}[m.group(1)]
    return apply(f"{prefix}[{m.group(2)}:{m.group(3)}]")
  if m := re.match(r'^([svt](?:tmp)?)(\d+)$', op_lower):
    prefix = {'s': 's', 'v': 'v', 't': 'ttmp', 'ttmp': 'ttmp'}[m.group(1)]
    return apply(f"{prefix}[{m.group(2)}]")
  if re.match(r'^-?\d+$', op) or re.match(r'^-?0x([0-9a-fA-F]+)$', op):
    return f"SrcMod({op}, neg={neg}, abs_={abs_})" if neg or abs_ else op
  if op_lower.startswith('hwreg(') or op_lower.startswith('sendmsg('): return apply(op)
  return apply(op)

def _parse_operands(op_str: str) -> list[str]:
  operands, current, depth, in_pipe = [], "", 0, False
  for ch in op_str:
    if ch in '[(': depth += 1
    elif ch in '])': depth -= 1
    elif ch == '|': in_pipe = not in_pipe
    if ch == ',' and depth == 0 and not in_pipe: operands.append(current.strip()); current = ""
    else: current += ch
  if current.strip(): operands.append(current.strip())
  return operands

def get_dsl(text: str) -> str:
  text = text.strip()
  kwargs = []

  # Extract modifiers
  omod_val = 0
  for pat, val in [(r'\s+mul:2(?:\s|$)', 1), (r'\s+mul:4(?:\s|$)', 2), (r'\s+div:2(?:\s|$)', 3)]:
    if m := re.search(pat, text, re.I): omod_val = val; text = text[:m.start()] + text[m.end():]
  if omod_val: kwargs.append(f'omod={omod_val}')

  if m := re.search(r'\s+clamp(?:\s|$)', text, re.I): kwargs.append('clmp=1'); text = text[:m.start()] + text[m.end():]

  opsel_explicit = None
  if m := re.search(r'\s+op_sel:\[([^\]]+)\]', text, re.I):
    bits = [int(x.strip()) for x in m.group(1).split(',')]
    mnemonic = text.split()[0].lower()
    is_vop3p = mnemonic.startswith(('v_pk_', 'v_wmma_', 'v_dot'))
    if len(bits) == 3:
      opsel_explicit = bits[0] | (bits[1] << 1) | (bits[2] << 2) if is_vop3p else bits[0] | (bits[1] << 1) | (bits[2] << 3)
    else: opsel_explicit = sum(b << i for i, b in enumerate(bits))
    text = text[:m.start()] + text[m.end():]

  if m := re.search(r'\s+wait_exp:(\d+)', text, re.I): kwargs.append(f'waitexp={m.group(1)}'); text = text[:m.start()] + text[m.end():]

  offset_val = None
  if m := re.search(r'\s+offset:(0x[0-9a-fA-F]+|-?\d+)', text, re.I): offset_val = m.group(1); text = text[:m.start()] + text[m.end():]

  dlc_val = glc_val = None
  if m := re.search(r'\s+dlc(?:\s|$)', text, re.I): dlc_val = 1; text = text[:m.start()] + text[m.end():]
  if m := re.search(r'\s+glc(?:\s|$)', text, re.I): glc_val = 1; text = text[:m.start()] + text[m.end():]

  neg_lo_val = neg_hi_val = None
  if m := re.search(r'\s+neg_lo:\[([^\]]+)\]', text, re.I):
    neg_lo_val = sum(int(x.strip()) << i for i, x in enumerate(m.group(1).split(','))); text = text[:m.start()] + text[m.end():]
  if m := re.search(r'\s+neg_hi:\[([^\]]+)\]', text, re.I):
    neg_hi_val = sum(int(x.strip()) << i for i, x in enumerate(m.group(1).split(','))); text = text[:m.start()] + text[m.end():]

  parts = text.replace(',', ' ').split()
  if not parts: raise ValueError("empty instruction")
  mnemonic, op_str = parts[0].lower(), text[len(parts[0]):].strip()

  # Special cases
  if mnemonic == 's_waitcnt':
    vmcnt, expcnt, lgkmcnt = 0x3f, 0x7, 0x3f
    for part in op_str.replace(',', ' ').split():
      if m := re.match(r'vmcnt\((\d+)\)', part): vmcnt = int(m.group(1))
      elif m := re.match(r'expcnt\((\d+)\)', part): expcnt = int(m.group(1))
      elif m := re.match(r'lgkmcnt\((\d+)\)', part): lgkmcnt = int(m.group(1))
      elif re.match(r'^0x[0-9a-f]+$|^\d+$', part): return f"s_waitcnt(simm16={int(part, 0)})"
    return f"s_waitcnt(simm16={waitcnt(vmcnt, expcnt, lgkmcnt)})"

  # VOPD dual-issue
  if '::' in text:
    x_part, y_part = text.split('::')
    x_parts, y_parts = x_part.strip().replace(',', ' ').split(), y_part.strip().replace(',', ' ').split()
    opx_name, opy_name = x_parts[0].upper(), y_parts[0].upper()
    x_ops, y_ops = [_operand_to_dsl(p) for p in x_parts[1:]], [_operand_to_dsl(p) for p in y_parts[1:]]
    vdstx, srcx0, vsrcx1 = x_ops[0], x_ops[1] if len(x_ops) > 1 else '0', x_ops[2] if len(x_ops) > 2 else 'v[0]'
    vdsty, srcy0, vsrcy1 = y_ops[0], y_ops[1] if len(y_ops) > 1 else '0', y_ops[2] if len(y_ops) > 2 else 'v[0]'
    lit = None
    if 'fmaak' in opx_name.lower() and len(x_ops) > 3: lit = x_ops[3]
    elif 'fmamk' in opx_name.lower() and len(x_ops) > 3: lit, vsrcx1 = x_ops[2], x_ops[3]
    elif 'fmaak' in opy_name.lower() and len(y_ops) > 3: lit = y_ops[3]
    elif 'fmamk' in opy_name.lower() and len(y_ops) > 3: lit, vsrcy1 = y_ops[2], y_ops[3]
    lit_s = f", literal={lit}" if lit else ""
    return f"VOPD(VOPDOp.{opx_name}, VOPDOp.{opy_name}, vdstx={vdstx}, vdsty={vdsty}, srcx0={srcx0}, vsrcx1={vsrcx1}, srcy0={srcy0}, vsrcy1={vsrcy1}{lit_s})"

  operands = _parse_operands(op_str)
  dsl_args = [_operand_to_dsl(op) for op in operands]

  # Instruction-specific handling
  if mnemonic == 's_setreg_imm32_b32': raise ValueError(f"unsupported instruction: {mnemonic}")
  if mnemonic in ('s_setpc_b64', 's_rfe_b64'): return f"{mnemonic}(ssrc0={dsl_args[0]})"
  if mnemonic in ('s_sendmsg_rtn_b32', 's_sendmsg_rtn_b64'): return f"{mnemonic}(sdst={dsl_args[0]}, ssrc0=RawImm({dsl_args[1].strip()}))"
  if mnemonic == 's_version': return f"{mnemonic}(simm16={dsl_args[0]})"
  if mnemonic == 's_setreg_b32': return f"{mnemonic}(simm16={dsl_args[0]}, sdst={dsl_args[1]})"

  # SMEM
  if mnemonic in SMEM_OPS:
    glc_s, dlc_s = ", glc=1" if glc_val else "", ", dlc=1" if dlc_val else ""
    if len(operands) >= 3 and re.match(r'^-?[0-9]|^-?0x', operands[2].strip().lower()):
      return f"{mnemonic}(sdata={dsl_args[0]}, sbase={dsl_args[1]}, offset={dsl_args[2]}, soffset=RawImm(124){glc_s}{dlc_s})"
    if offset_val and len(operands) >= 3:
      return f"{mnemonic}(sdata={dsl_args[0]}, sbase={dsl_args[1]}, offset={offset_val}, soffset={dsl_args[2]}{glc_s}{dlc_s})"
    if len(operands) >= 3:
      return f"{mnemonic}(sdata={dsl_args[0]}, sbase={dsl_args[1]}, soffset={dsl_args[2]}{glc_s}{dlc_s})"

  # Buffer ops
  if mnemonic.startswith('buffer_') and len(operands) >= 2 and operands[1].strip().lower() == 'off':
    soff = f"RawImm({dsl_args[3].strip()})" if len(dsl_args) > 3 else "RawImm(0)"
    return f"{mnemonic}(vdata={dsl_args[0]}, vaddr=0, srsrc={dsl_args[2]}, soffset={soff})"

  # FLAT/GLOBAL/SCRATCH
  if (mnemonic.startswith('flat_load') or mnemonic.startswith('global_load') or mnemonic.startswith('scratch_load')) and len(dsl_args) >= 3:
    off = f", offset={offset_val}" if offset_val else ""
    return f"{mnemonic}(vdst={dsl_args[0]}, addr={dsl_args[1]}, saddr={dsl_args[2]}{off})"
  if (mnemonic.startswith('flat_store') or mnemonic.startswith('global_store') or mnemonic.startswith('scratch_store')) and len(dsl_args) >= 3:
    off = f", offset={offset_val}" if offset_val else ""
    return f"{mnemonic}(addr={dsl_args[0]}, data={dsl_args[1]}, saddr={dsl_args[2]}{off})"

  # v_fmaak/v_fmamk literals
  lit_s = ""
  if mnemonic in ('v_fmaak_f32', 'v_fmaak_f16') and len(dsl_args) == 4:
    lit_s, dsl_args = f", literal={dsl_args[3].strip()}", dsl_args[:3]
  elif mnemonic in ('v_fmamk_f32', 'v_fmamk_f16') and len(dsl_args) == 4:
    lit_s, dsl_args = f", literal={dsl_args[2].strip()}", [dsl_args[0], dsl_args[1], dsl_args[3]]

  # VCC ops
  vcc_ops = {'v_add_co_ci_u32', 'v_sub_co_ci_u32', 'v_subrev_co_ci_u32'}
  if mnemonic.replace('_e32', '') in vcc_ops and len(dsl_args) >= 5:
    mnemonic = mnemonic.replace('_e32', '') + '_e32'
    dsl_args = [dsl_args[0], dsl_args[2], dsl_args[3]]
  if mnemonic.replace('_e64', '') in vcc_ops and mnemonic.endswith('_e64'):
    mnemonic = mnemonic.replace('_e64', '')

  # v_cmp strip implicit vcc_lo
  if mnemonic.startswith('v_cmp') and not mnemonic.endswith('_e64') and len(dsl_args) >= 3 and operands[0].strip().lower() in ('vcc_lo', 'vcc_hi', 'vcc'):
    dsl_args = dsl_args[1:]

  # CMPX with _e64
  if 'cmpx' in mnemonic and mnemonic.endswith('_e64') and len(dsl_args) == 2:
    dsl_args = ['RawImm(126)'] + dsl_args

  func_name = mnemonic.replace('.', '_')
  if opsel_explicit is not None:
    dsl_args = [re.sub(r'\.[hl]$', '', a) for a in dsl_args]

  args_str = ', '.join(dsl_args)
  all_kwargs = list(kwargs)
  if lit_s: all_kwargs.append(lit_s.lstrip(', '))
  if opsel_explicit is not None: all_kwargs.append(f'opsel={opsel_explicit}')
  if neg_lo_val is not None: all_kwargs.append(f'neg={neg_lo_val}')
  if neg_hi_val is not None: all_kwargs.append(f'neg_hi={neg_hi_val}')
  kwargs_str = ', '.join(all_kwargs)
  if kwargs_str: return f"{func_name}({args_str}, {kwargs_str})" if args_str else f"{func_name}({kwargs_str})"
  return f"{func_name}({args_str})"

def asm(text: str) -> Inst:
  from extra.assembly.amd.autogen import rdna3 as autogen
  dsl_expr = get_dsl(text)
  namespace = {name: getattr(autogen, name) for name in dir(autogen) if not name.startswith('_')}
  namespace.update({'s': s, 'v': v, 'ttmp': ttmp, 'abs': abs, 'RawImm': RawImm, 'SrcMod': SrcMod, 'VGPR': VGPR, 'SGPR': SGPR, 'TTMP': TTMP,
                    'VCC_LO': VCC_LO, 'VCC_HI': VCC_HI, 'VCC': VCC, 'EXEC_LO': EXEC_LO, 'EXEC_HI': EXEC_HI, 'EXEC': EXEC,
                    'SCC': SCC, 'M0': M0, 'NULL': NULL, 'OFF': OFF})
  try: return eval(dsl_expr, namespace)
  except NameError:
    if m := re.match(r'^(v_\w+)(\(.*\))$', dsl_expr): return eval(f"{m.group(1)}_e32{m.group(2)}", namespace)
    raise
