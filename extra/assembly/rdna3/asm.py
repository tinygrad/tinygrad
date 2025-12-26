# RDNA3 assembler and disassembler
from __future__ import annotations
import re
from extra.assembly.rdna3.lib import Inst, RawImm, Reg, SGPR, VGPR, TTMP, FLOAT_ENC, SRC_FIELDS, unwrap

# Decoding helpers
SPECIAL_GPRS = {106: "vcc_lo", 107: "vcc_hi", 124: "null", 125: "m0", 126: "exec_lo", 127: "exec_hi", 253: "scc"}
SPECIAL_DEC = {**SPECIAL_GPRS, **{v: str(k) for k, v in FLOAT_ENC.items()}}

def decode_src(val: int) -> str:
  if val <= 105: return f"s{val}"
  if val in SPECIAL_DEC: return SPECIAL_DEC[val]
  if 108 <= val <= 123: return f"ttmp{val - 108}"
  if 128 <= val <= 192: return str(val - 128)
  if 193 <= val <= 208: return str(-(val - 192))
  if 256 <= val <= 511: return f"v{val - 256}"
  return "lit" if val == 255 else f"?{val}"

def _sreg(base: int, cnt: int = 1) -> str: return f"s{base}" if cnt == 1 else f"s[{base}:{base+cnt-1}]"
def _vreg(base: int, cnt: int = 1) -> str: return f"v{base}" if cnt == 1 else f"v[{base}:{base+cnt-1}]"

def _fmt_sdst(v: int, cnt: int = 1) -> str:
  """Format SGPR destination with special register names."""
  if v == 124: return "null"
  if 108 <= v <= 123: return f"ttmp[{v-108}:{v-108+cnt-1}]" if cnt > 1 else f"ttmp{v-108}"
  if cnt > 1:
    if v == 126 and cnt == 2: return "exec"
    if v == 106 and cnt == 2: return "vcc"
    return _sreg(v, cnt)
  return {126: "exec_lo", 127: "exec_hi", 106: "vcc_lo", 107: "vcc_hi", 125: "m0"}.get(v, f"s{v}")

def _fmt_ssrc(v: int, cnt: int = 1) -> str:
  """Format SGPR source with special register names and pairs."""
  if cnt == 2:
    if v == 126: return "exec"
    if v == 106: return "vcc"
    if v <= 105: return _sreg(v, 2)
    if 108 <= v <= 123: return f"ttmp[{v-108}:{v-108+1}]"
  return decode_src(v)

def _parse_sop_sizes(op_name: str) -> tuple[int, ...]:
  """Parse dst and src sizes from SOP instruction name. Returns (dst_cnt, src0_cnt) or (dst_cnt, src0_cnt, src1_cnt)."""
  if op_name in ('s_bitset0_b64', 's_bitset1_b64'): return (2, 1)
  if op_name in ('s_lshl_b64', 's_lshr_b64', 's_ashr_i64', 's_bfe_u64', 's_bfe_i64'): return (2, 2, 1)
  if op_name in ('s_bfm_b64',): return (2, 1, 1)
  # SOPC: s_bitcmp0_b64, s_bitcmp1_b64 - 64-bit src0, 32-bit src1 (bit index)
  if op_name in ('s_bitcmp0_b64', 's_bitcmp1_b64'): return (1, 2, 1)
  if m := re.search(r'_(b|i|u)(32|64)_(b|i|u)(32|64)$', op_name):
    return (2 if m.group(2) == '64' else 1, 2 if m.group(4) == '64' else 1)
  if m := re.search(r'_(b|i|u)(32|64)$', op_name):
    sz = 2 if m.group(2) == '64' else 1
    return (sz, sz)
  return (1, 1)

# Waitcnt helpers (RDNA3 format: bits 15:10=vmcnt, bits 9:4=lgkmcnt, bits 3:0=expcnt)
def waitcnt(vmcnt: int = 0x3f, expcnt: int = 0x7, lgkmcnt: int = 0x3f) -> int:
  return (expcnt & 0x7) | ((lgkmcnt & 0x3f) << 4) | ((vmcnt & 0x3f) << 10)
def decode_waitcnt(val: int) -> tuple[int, int, int]:
  return (val >> 10) & 0x3f, val & 0xf, (val >> 4) & 0x3f  # vmcnt, expcnt, lgkmcnt

# VOP3SD opcodes (shared encoding with VOP3 but different field layout)
# Note: opcodes 0-255 are VOPC promoted to VOP3 - never treat as VOP3SD
VOP3SD_OPCODES = {288, 289, 290, 764, 765, 766, 767, 768, 769, 770}

# Disassembler
def disasm(inst: Inst) -> str:
  op_val = unwrap(inst._values.get('op', 0))
  cls_name = inst.__class__.__name__
  # VOP3 and VOP3SD share encoding - check opcode to determine which
  is_vop3sd = cls_name == 'VOP3' and op_val in VOP3SD_OPCODES
  try:
    from extra.assembly.rdna3 import autogen
    if is_vop3sd:
      op_name = autogen.VOP3SDOp(op_val).name.lower()
    else:
      op_name = getattr(autogen, f"{cls_name}Op")(op_val).name.lower() if hasattr(autogen, f"{cls_name}Op") else f"op_{op_val}"
  except (ValueError, KeyError): op_name = f"op_{op_val}"
  def fmt_src(v): return f"0x{inst._literal:x}" if v == 255 and getattr(inst, '_literal', None) else decode_src(v)

  # VOP1
  if cls_name == 'VOP1':
    vdst, src0 = unwrap(inst._values['vdst']), unwrap(inst._values['src0'])
    if op_name == 'v_nop': return 'v_nop'
    if op_name == 'v_pipeflush': return 'v_pipeflush'
    parts = op_name.split('_')
    is_16bit_dst = any(p in ('f16', 'i16', 'u16', 'b16') for p in parts[-2:-1]) or (len(parts) >= 2 and parts[-1] in ('f16', 'i16', 'u16', 'b16') and 'cvt' not in op_name)
    is_16bit_src = parts[-1] in ('f16', 'i16', 'u16', 'b16') and 'sat_pk' not in op_name
    is_f64_dst = op_name in ('v_ceil_f64', 'v_floor_f64', 'v_fract_f64', 'v_frexp_mant_f64', 'v_rcp_f64', 'v_rndne_f64', 'v_rsq_f64', 'v_sqrt_f64', 'v_trunc_f64', 'v_cvt_f64_f32', 'v_cvt_f64_i32', 'v_cvt_f64_u32')
    is_f64_src = op_name in ('v_ceil_f64', 'v_floor_f64', 'v_fract_f64', 'v_frexp_mant_f64', 'v_rcp_f64', 'v_rndne_f64', 'v_rsq_f64', 'v_sqrt_f64', 'v_trunc_f64', 'v_cvt_f32_f64', 'v_cvt_i32_f64', 'v_cvt_u32_f64', 'v_frexp_exp_i32_f64')
    if op_name == 'v_readfirstlane_b32':
      return f"v_readfirstlane_b32 {decode_src(vdst)}, v{src0 - 256 if src0 >= 256 else src0}"
    dst_str = _vreg(vdst, 2) if is_f64_dst else f"v{vdst & 0x7f}.{'h' if vdst >= 128 else 'l'}" if is_16bit_dst else f"v{vdst}"
    if is_f64_src:
      src_str = _vreg(src0 - 256, 2) if src0 >= 256 else _sreg(src0, 2) if src0 <= 105 else "vcc" if src0 == 106 else "exec" if src0 == 126 else f"ttmp[{src0-108}:{src0-108+1}]" if 108 <= src0 <= 123 else fmt_src(src0)
    elif is_16bit_src and src0 >= 256:
      src_str = f"v{(src0 - 256) & 0x7f}.{'h' if src0 >= 384 else 'l'}"
    else:
      src_str = fmt_src(src0)
    return f"{op_name}_e32 {dst_str}, {src_str}"

  # VOP2
  if cls_name == 'VOP2':
    vdst, src0_raw, vsrc1 = unwrap(inst._values['vdst']), unwrap(inst._values['src0']), unwrap(inst._values['vsrc1'])
    suffix = "" if op_name == "v_dot2acc_f32_f16" else "_e32"
    is_16bit_op = ('_f16' in op_name or '_i16' in op_name or '_u16' in op_name) and '_f32' not in op_name and '_i32' not in op_name and 'pk_' not in op_name
    if is_16bit_op:
      dst_str = f"v{vdst & 0x7f}.{'h' if vdst >= 128 else 'l'}"
      src0_str = f"v{(src0_raw - 256) & 0x7f}.{'h' if src0_raw >= 384 else 'l'}" if src0_raw >= 256 else fmt_src(src0_raw)
      vsrc1_str = f"v{vsrc1 & 0x7f}.{'h' if vsrc1 >= 128 else 'l'}"
    else:
      dst_str, src0_str, vsrc1_str = f"v{vdst}", fmt_src(src0_raw), f"v{vsrc1}"
    return f"{op_name}{suffix} {dst_str}, {src0_str}, {vsrc1_str}" + (", vcc_lo" if op_name == "v_cndmask_b32" else "")

  # VOPC
  if cls_name == 'VOPC':
    src0, vsrc1 = unwrap(inst._values['src0']), unwrap(inst._values['vsrc1'])
    is_64bit = any(x in op_name for x in ('f64', 'i64', 'u64'))
    is_64bit_vsrc1 = is_64bit and 'class' not in op_name
    is_16bit = any(x in op_name for x in ('_f16', '_i16', '_u16')) and 'f32' not in op_name
    is_cmpx = op_name.startswith('v_cmpx')  # VOPCX writes to exec, no vcc destination
    if is_64bit:
      src0_str = _vreg(src0 - 256, 2) if src0 >= 256 else _sreg(src0, 2) if src0 <= 105 else "vcc" if src0 == 106 else "exec" if src0 == 126 else f"ttmp[{src0-108}:{src0-108+1}]" if 108 <= src0 <= 123 else fmt_src(src0)
    elif is_16bit and src0 >= 256:
      src0_str = f"v{(src0 - 256) & 0x7f}.{'h' if src0 >= 384 else 'l'}"
    else:
      src0_str = fmt_src(src0)
    vsrc1_str = _vreg(vsrc1, 2) if is_64bit_vsrc1 else f"v{vsrc1 & 0x7f}.{'h' if vsrc1 >= 128 else 'l'}" if is_16bit else f"v{vsrc1}"
    if is_cmpx:
      return f"{op_name}_e32 {src0_str}, {vsrc1_str}"
    return f"{op_name}_e32 vcc_lo, {src0_str}, {vsrc1_str}"

  # SOPP
  if cls_name == 'SOPP':
    simm16 = unwrap(inst._values.get('simm16', 0))
    # No-operand instructions (simm16 is ignored)
    no_imm_ops = ('s_endpgm', 's_barrier', 's_wakeup', 's_icache_inv', 's_ttracedata', 's_ttracedata_imm',
                  's_wait_idle', 's_endpgm_saved', 's_code_end', 's_endpgm_ordered_ps_done')
    if op_name in no_imm_ops: return op_name
    if op_name == 's_waitcnt':
      vmcnt, expcnt, lgkmcnt = decode_waitcnt(simm16)
      parts = []
      if vmcnt != 0x3f: parts.append(f"vmcnt({vmcnt})")
      if expcnt != 0x7: parts.append(f"expcnt({expcnt})")
      if lgkmcnt != 0x3f: parts.append(f"lgkmcnt({lgkmcnt})")
      return f"s_waitcnt {' '.join(parts)}" if parts else "s_waitcnt 0"
    if op_name == 's_delay_alu':
      dep_names = ['VALU_DEP_1','VALU_DEP_2','VALU_DEP_3','VALU_DEP_4','TRANS32_DEP_1','TRANS32_DEP_2','TRANS32_DEP_3','FMA_ACCUM_CYCLE_1','SALU_CYCLE_1','SALU_CYCLE_2','SALU_CYCLE_3']
      skip_names = ['SAME','NEXT','SKIP_1','SKIP_2','SKIP_3','SKIP_4']
      id0, skip, id1 = simm16 & 0xf, (simm16 >> 4) & 0x7, (simm16 >> 7) & 0xf
      def dep_name(v): return dep_names[v-1] if 0 < v <= len(dep_names) else str(v)
      parts = [f"instid0({dep_name(id0)})"] if id0 else []
      if skip: parts.append(f"instskip({skip_names[skip]})")
      if id1: parts.append(f"instid1({dep_name(id1)})")
      return f"s_delay_alu {' | '.join(p for p in parts if p)}" if parts else "s_delay_alu 0"
    if op_name.startswith('s_cbranch') or op_name.startswith('s_branch'):
      return f"{op_name} {simm16}"
    # Most SOPP ops require immediate (s_nop, s_setkill, s_sethalt, s_sleep, s_setprio, s_sendmsg*, etc.)
    return f"{op_name} 0x{simm16:x}"

  # SMEM
  if cls_name == 'SMEM':
    # No-operand instructions
    if op_name in ('s_gl1_inv', 's_dcache_inv'): return op_name
    sdata, sbase, soffset, offset = unwrap(inst._values['sdata']), unwrap(inst._values['sbase']), unwrap(inst._values['soffset']), unwrap(inst._values.get('offset', 0))
    glc, dlc = unwrap(inst._values.get('glc', 0)), unwrap(inst._values.get('dlc', 0))
    # s_atc_probe/s_atc_probe_buffer: sdata is the probe mode (0-7), not a register
    if op_name in ('s_atc_probe', 's_atc_probe_buffer'):
      sbase_idx = sbase * 2
      sbase_cnt = 4 if op_name == 's_atc_probe_buffer' else 2
      sbase_str = _sreg(sbase_idx, sbase_cnt)
      if offset and soffset != 124:
        off_str = f"{decode_src(soffset)} offset:0x{offset:x}"
      elif offset:
        off_str = f"0x{offset:x}"
      else:
        off_str = decode_src(soffset)
      return f"{op_name} {sdata}, {sbase_str}, {off_str}"
    width = {0:1, 1:2, 2:4, 3:8, 4:16, 8:1, 9:2, 10:4, 11:8, 12:16}.get(op_val, 1)
    # Offset handling: if offset is set, we need "soffset offset:X" format, otherwise just soffset or imm
    if offset and soffset != 124:  # both soffset register and offset immediate
      off_str = f"{decode_src(soffset)} offset:0x{offset:x}"
    elif offset:  # only offset immediate (soffset=null)
      off_str = f"0x{offset:x}"
    elif soffset == 124:  # null
      off_str = "null"
    else:  # only soffset register
      off_str = decode_src(soffset)
    # sbase is stored as register pair index, multiply by 2 for actual register number
    # s_buffer_load_* (op 8-12) use 4-reg sbase (buffer descriptor), s_load_* (op 0-4) use 2-reg sbase
    sbase_idx = sbase * 2
    sbase_cnt = 4 if 8 <= op_val <= 12 else 2
    # Format sbase with special register names
    if sbase_idx == 106 and sbase_cnt == 2: sbase_str = "vcc"
    elif sbase_idx == 126 and sbase_cnt == 2: sbase_str = "exec"
    elif 108 <= sbase_idx <= 123: sbase_str = f"ttmp[{sbase_idx-108}:{sbase_idx-108+sbase_cnt-1}]"
    else: sbase_str = _sreg(sbase_idx, sbase_cnt)
    # Build modifiers
    mods = []
    if glc: mods.append("glc")
    if dlc: mods.append("dlc")
    mod_str = " " + " ".join(mods) if mods else ""
    return f"{op_name} {_fmt_sdst(sdata, width)}, {sbase_str}, {off_str}{mod_str}"

  # FLAT
  if cls_name == 'FLAT':
    vdst, addr, data, saddr, offset, seg = [unwrap(inst._values.get(f, 0)) for f in ['vdst', 'addr', 'data', 'saddr', 'offset', 'seg']]
    prefix = {0: 'flat', 1: 'scratch', 2: 'global'}.get(seg, 'flat')
    op_suffix = op_name.split('_', 1)[1] if '_' in op_name else op_name
    instr = f"{prefix}_{op_suffix}"
    is_store = 'store' in op_name
    width = {'b32':1, 'b64':2, 'b96':3, 'b128':4, 'u8':1, 'i8':1, 'u16':1, 'i16':1}.get(op_name.split('_')[-1], 1)
    if saddr == 0x7F:
      addr_str, saddr_str = _vreg(addr, 2), ""
    else:
      addr_str = _vreg(addr)
      saddr_str = f", {_sreg(saddr, 2)}" if saddr < 106 else f", off" if saddr == 124 else f", {decode_src(saddr)}"
    off_str = f" offset:{offset}" if offset else ""
    if is_store: return f"{instr} {addr_str}, {_vreg(data, width)}{saddr_str}{off_str}"
    return f"{instr} {_vreg(vdst, width)}, {addr_str}{saddr_str}{off_str}"

  # VOP3: vector ops with modifiers (can be 1, 2, or 3 sources depending on opcode range)
  if cls_name == 'VOP3':
    # Handle VOP3SD opcodes (same encoding, different field layout)
    if is_vop3sd:
      vdst = unwrap(inst._values.get('vdst', 0))
      # VOP3SD: sdst is at bits [14:8], but VOP3 decodes opsel at [14:11], abs at [10:8], clmp at [15]
      # We need to reconstruct sdst from these fields
      opsel_raw = unwrap(inst._values.get('opsel', 0))
      abs_raw = unwrap(inst._values.get('abs', 0))
      clmp_raw = unwrap(inst._values.get('clmp', 0))
      sdst = (clmp_raw << 7) | (opsel_raw << 3) | abs_raw
      src0, src1, src2 = [unwrap(inst._values.get(f, 0)) for f in ('src0', 'src1', 'src2')]
      neg = unwrap(inst._values.get('neg', 0))
      omod = unwrap(inst._values.get('omod', 0))
      omod_str = {1: " mul:2", 2: " mul:4", 3: " div:2"}.get(omod, "")
      is_f64 = 'f64' in op_name
      # v_mad_i64_i32/v_mad_u64_u32: 64-bit dst and src2, 32-bit src0/src1
      is_mad64 = 'mad_i64_i32' in op_name or 'mad_u64_u32' in op_name
      def fmt_sd_src(v, neg_bit, is_64bit=False):
        s = fmt_src(v)
        if is_64bit or is_f64:
          if v >= 256: s = _vreg(v - 256, 2)
          elif v <= 105: s = _sreg(v, 2)
          elif v == 106: s = "vcc"
          elif v == 126: s = "exec"
          elif 108 <= v <= 123: s = f"ttmp[{v-108}:{v-108+1}]"
        if neg_bit: s = f"-{s}"
        return s
      src0_str = fmt_sd_src(src0, neg & 1, False)  # 32-bit for mad64
      src1_str = fmt_sd_src(src1, neg & 2, False)  # 32-bit for mad64
      src2_str = fmt_sd_src(src2, neg & 4, is_mad64)  # 64-bit for mad64
      dst_str = _vreg(vdst, 2) if (is_f64 or is_mad64) else f"v{vdst}"
      sdst_str = _fmt_sdst(sdst, 1)
      # v_add_co_u32, v_sub_co_u32, v_subrev_co_u32, v_add_co_ci_u32, etc. only use 2 sources
      if op_name in ('v_add_co_u32', 'v_sub_co_u32', 'v_subrev_co_u32', 'v_add_co_ci_u32', 'v_sub_co_ci_u32', 'v_subrev_co_ci_u32'):
        return f"{op_name} {dst_str}, {sdst_str}, {src0_str}, {src1_str}"
      # v_div_scale uses 3 sources
      return f"{op_name} {dst_str}, {sdst_str}, {src0_str}, {src1_str}, {src2_str}" + omod_str

    vdst = unwrap(inst._values.get('vdst', 0))
    src0, src1, src2 = [unwrap(inst._values.get(f, 0)) for f in ('src0', 'src1', 'src2')]
    neg, abs_, clmp = unwrap(inst._values.get('neg', 0)), unwrap(inst._values.get('abs', 0)), unwrap(inst._values.get('clmp', 0))
    opsel = unwrap(inst._values.get('opsel', 0))
    # Check if 64-bit op (needs register pairs)
    is_f64 = 'f64' in op_name or 'i64' in op_name or 'u64' in op_name or 'b64' in op_name
    # v_cmp_class_* has 64-bit src0 but 32-bit src1 (class mask)
    is_class = 'class' in op_name
    # Shift ops: v_*rev_*64 have 32-bit shift amount (src0), 64-bit value (src1)
    is_shift64 = 'rev' in op_name and '64' in op_name and op_name.startswith('v_')
    # v_ldexp_f64: 64-bit src0 (mantissa), 32-bit src1 (exponent)
    is_ldexp64 = op_name == 'v_ldexp_f64'
    # v_trig_preop_f64: 64-bit dst/src0, 32-bit src1 (exponent/scale)
    is_trig_preop = op_name == 'v_trig_preop_f64'
    # v_readlane_b32: destination is SGPR (despite vdst field)
    is_readlane = op_name == 'v_readlane_b32'
    # SAD/QSAD/MQSAD instructions have mixed sizes
    # v_qsad_pk_u16_u8, v_mqsad_pk_u16_u8: 64-bit dst/src0/src2, 32-bit src1
    # v_mqsad_u32_u8: 128-bit (4 reg) dst/src2, 64-bit src0, 32-bit src1
    is_sad64 = any(x in op_name for x in ('qsad_pk', 'mqsad_pk'))
    is_mqsad_u32 = 'mqsad_u32' in op_name
    # Detect conversion ops: v_cvt_{dst_type}_{src_type} - each side may have different size
    # Also handle v_cvt_pk_* which packs two values into one
    if 'cvt_pk' in op_name:
      # Pack ops: dst is packed 16-bit, src is determined by last type in name
      # e.g., v_cvt_pk_i16_f32, v_cvt_pk_norm_i16_f32
      is_f16_dst = is_f16_src = is_f16_src2 = False  # dst is 32-bit, srcs depend on op
      is_f16_src = op_name.endswith('16')  # only if final type is 16-bit
    elif m := re.match(r'v_cvt_([a-z0-9_]+)_([a-z0-9]+)', op_name):
      dst_type, src_type = m.group(1), m.group(2)
      # Check if dst/src ends with a 16-bit type suffix
      is_f16_dst = any(dst_type.endswith(x) for x in ('f16', 'i16', 'u16', 'b16'))
      is_f16_src = is_f16_src2 = any(src_type.endswith(x) for x in ('f16', 'i16', 'u16', 'b16'))
      # Override is_f64 for conversion ops - check if dst or src is 64-bit
      is_f64_dst = '64' in dst_type
      is_f64_src = '64' in src_type
      is_f64 = False  # Don't use default is_f64 detection for cvt ops
    elif m := re.match(r'v_frexp_exp_([a-z0-9]+)_([a-z0-9]+)', op_name):
      # v_frexp_exp_i32_f64: 32-bit dst (exponent), 64-bit src
      # v_frexp_exp_i16_f16: 16-bit dst, 16-bit src
      dst_type, src_type = m.group(1), m.group(2)
      is_f16_dst = any(dst_type.endswith(x) for x in ('f16', 'i16', 'u16', 'b16'))
      is_f16_src = is_f16_src2 = any(src_type.endswith(x) for x in ('f16', 'i16', 'u16', 'b16'))
      is_f64_dst = '64' in dst_type
      is_f64_src = '64' in src_type
      is_f64 = False
    elif m := re.match(r'v_mad_([iu])32_([iu])16', op_name):
      # v_mad_i32_i16, v_mad_u32_u16: 32-bit dst, 16-bit src0/src1, 32-bit src2
      is_f16_dst = False
      is_f16_src = True  # src0 and src1 are 16-bit
      is_f16_src2 = False  # src2 is 32-bit
    elif 'pack_b32' in op_name:
      # v_pack_b32_f16: 32-bit dst, 16-bit sources
      is_f16_dst = False
      is_f16_src = is_f16_src2 = True
    else:
      # 16-bit ops need .h/.l suffix, but packed ops (dot2, pk_, sad, msad, qsad, mqsad) don't
      is_16bit_op = ('f16' in op_name or 'i16' in op_name or 'u16' in op_name or 'b16' in op_name) and not any(x in op_name for x in ('dot2', 'pk_', 'sad', 'msad', 'qsad', 'mqsad'))
      is_f16_dst = is_f16_src = is_f16_src2 = is_16bit_op
    def fmt_vop3_src(v, neg_bit, abs_bit, hi_bit=False, reg_cnt=1, is_16=False):
      s = fmt_src(v)
      # Add register pair/quad for 64/128-bit, or .h suffix for f16 VGPRs with opsel
      if reg_cnt > 1 and v >= 256: s = _vreg(v - 256, reg_cnt)
      elif reg_cnt > 1 and v <= 105: s = _sreg(v, reg_cnt)
      elif reg_cnt == 2 and v == 106: s = "vcc"
      elif reg_cnt == 2 and v == 126: s = "exec"
      elif reg_cnt > 1 and 108 <= v <= 123: s = f"ttmp[{v-108}:{v-108+reg_cnt-1}]"
      elif is_16 and v >= 256: s = f"v{v - 256}.h" if hi_bit else f"v{v - 256}.l"
      if abs_bit: s = f"|{s}|"
      if neg_bit: s = f"-{s}"
      return s
    # Determine register count for each source (check for cvt-specific 64-bit flags first)
    is_src0_64 = locals().get('is_f64_src', is_f64 and not is_shift64) or is_sad64 or is_mqsad_u32
    is_src1_64 = is_f64 and not is_class and not is_ldexp64 and not is_trig_preop
    src0_cnt = 2 if is_src0_64 else 1
    src1_cnt = 2 if is_src1_64 else 1
    src2_cnt = 4 if is_mqsad_u32 else 2 if (is_f64 or is_sad64) else 1
    src0_str = fmt_vop3_src(src0, neg & 1, abs_ & 1, opsel & 1, src0_cnt, is_f16_src)
    src1_str = fmt_vop3_src(src1, neg & 2, abs_ & 2, opsel & 2, src1_cnt, is_f16_src)
    src2_str = fmt_vop3_src(src2, neg & 4, abs_ & 4, opsel & 4, src2_cnt, is_f16_src2)
    # Format destination - for 16-bit ops, use .h/.l suffix; readlane uses SGPR dest
    is_dst_64 = locals().get('is_f64_dst', is_f64) or is_sad64
    dst_cnt = 4 if is_mqsad_u32 else 2 if is_dst_64 else 1
    if is_readlane:
      dst_str = _fmt_sdst(vdst, 1)
    elif dst_cnt > 1:
      dst_str = _vreg(vdst, dst_cnt)
    elif is_f16_dst:
      dst_str = f"v{vdst}.h" if (opsel & 8) else f"v{vdst}.l"
    else:
      dst_str = f"v{vdst}"
    clamp_str = " clamp" if clmp else ""
    omod = unwrap(inst._values.get('omod', 0))
    omod_str = {1: " mul:2", 2: " mul:4", 3: " div:2"}.get(omod, "")
    # op_sel for non-VGPR sources (when opsel bits are set but source is not a VGPR)
    # For 16-bit ops with VGPR sources, opsel is encoded in .h/.l suffix
    # For non-VGPR sources or non-16-bit ops, we need explicit op_sel
    has_nonvgpr_opsel = (src0 < 256 and (opsel & 1)) or (src1 < 256 and (opsel & 2)) or (src2 < 256 and (opsel & 4))
    need_opsel = has_nonvgpr_opsel or (opsel and not is_f16_src)
    # Helper to format opsel string based on source count
    def fmt_opsel(num_src):
      if not need_opsel: return ""
      # When dst is .h (for 16-bit ops) and non-VGPR sources have opsel, use all 1s
      if is_f16_dst and (opsel & 8):  # dst is .h
        return f" op_sel:[1,1,1{',1' if num_src == 3 else ''}]"
      # Otherwise output actual opsel values
      if num_src == 3:
        return f" op_sel:[{opsel & 1},{(opsel >> 1) & 1},{(opsel >> 2) & 1},{(opsel >> 3) & 1}]"
      return f" op_sel:[{opsel & 1},{(opsel >> 1) & 1},{(opsel >> 2) & 1}]"
    # Determine number of sources based on opcode range:
    # 0-255: VOPC promoted (comparison, 2 src, sdst)
    # 256-383: VOP2 promoted (2 src)
    # 384-511: VOP1 promoted (1 src)
    # 512+: Native VOP3 (2 or 3 src depending on instruction)
    if op_val < 256:  # VOPC promoted
      # VOPCX (v_cmpx_*) writes to exec, no explicit destination
      if op_name.startswith('v_cmpx'):
        return f"{op_name}_e64 {src0_str}, {src1_str}"
      return f"{op_name}_e64 {_fmt_sdst(vdst, 1)}, {src0_str}, {src1_str}"
    elif op_val < 384:  # VOP2 promoted
      # v_cndmask_b32 in VOP3 format has 3 sources (src2 is mask selector)
      if 'cndmask' in op_name:
        return f"{op_name}_e64 {dst_str}, {src0_str}, {src1_str}, {src2_str}" + fmt_opsel(3) + clamp_str + omod_str
      return f"{op_name}_e64 {dst_str}, {src0_str}, {src1_str}" + fmt_opsel(2) + clamp_str + omod_str
    elif op_val < 512:  # VOP1 promoted
      if op_name in ('v_nop', 'v_pipeflush'): return f"{op_name}_e64"
      return f"{op_name}_e64 {dst_str}, {src0_str}" + fmt_opsel(1) + clamp_str + omod_str
    else:  # Native VOP3 - determine 2 vs 3 sources based on instruction name
      # 3-source ops: fma, mad, min3, max3, med3, div_fixup, div_fmas, sad, msad, qsad, mqsad, lerp, alignbit/byte, cubeid/sc/tc/ma, bfe, bfi, perm_b32, permlane, cndmask
      # Note: v_writelane_b32 is 2-src (src0, src1 with vdst as 3rd operand - read-modify-write)
      is_3src = any(x in op_name for x in ('fma', 'mad', 'min3', 'max3', 'med3', 'div_fix', 'div_fmas', 'sad', 'lerp', 'align', 'cube',
                                            'bfe', 'bfi', 'perm_b32', 'permlane', 'cndmask', 'xor3', 'or3', 'add3', 'lshl_or', 'and_or', 'lshl_add',
                                            'add_lshl', 'xad', 'maxmin', 'minmax', 'dot2', 'cvt_pk_u8', 'mullit'))
      if is_3src:
        return f"{op_name} {dst_str}, {src0_str}, {src1_str}, {src2_str}" + fmt_opsel(3) + clamp_str + omod_str
      return f"{op_name} {dst_str}, {src0_str}, {src1_str}" + fmt_opsel(2) + clamp_str + omod_str

  # VOP3SD: 3-source with scalar destination (v_div_scale_*, v_add_co_u32, v_mad_*64_*32, etc.)
  if cls_name == 'VOP3SD':
    vdst, sdst = unwrap(inst._values.get('vdst', 0)), unwrap(inst._values.get('sdst', 0))
    src0, src1, src2 = [unwrap(inst._values.get(f, 0)) for f in ('src0', 'src1', 'src2')]
    neg = unwrap(inst._values.get('neg', 0))
    omod = unwrap(inst._values.get('omod', 0))
    clmp = unwrap(inst._values.get('clmp', 0))
    is_f64 = 'f64' in op_name
    is_mad64 = 'mad_i64_i32' in op_name or 'mad_u64_u32' in op_name
    def fmt_sd_src(v, neg_bit, is_64bit=False):
      s = fmt_src(v)
      if is_64bit or is_f64:
        if v >= 256: s = _vreg(v - 256, 2)
        elif v <= 105: s = _sreg(v, 2)
        elif v == 106: s = "vcc"
        elif v == 126: s = "exec"
        elif 108 <= v <= 123: s = f"ttmp[{v-108}:{v-108+1}]"
      if neg_bit: s = f"-{s}"
      return s
    src0_str = fmt_sd_src(src0, neg & 1, False)
    src1_str = fmt_sd_src(src1, neg & 2, False)
    src2_str = fmt_sd_src(src2, neg & 4, is_mad64)
    dst_str = _vreg(vdst, 2) if (is_f64 or is_mad64) else f"v{vdst}"
    sdst_str = _fmt_sdst(sdst, 1)
    clamp_str = " clamp" if clmp else ""
    omod_str = {1: " mul:2", 2: " mul:4", 3: " div:2"}.get(omod, "")
    # v_add_co_u32, v_sub_co_u32, v_subrev_co_u32 only use 2 sources
    if op_name in ('v_add_co_u32', 'v_sub_co_u32', 'v_subrev_co_u32'):
      return f"{op_name}_e64 {dst_str}, {sdst_str}, {src0_str}, {src1_str}" + clamp_str
    # v_add_co_ci_u32, v_sub_co_ci_u32, v_subrev_co_ci_u32 use 3 sources (src2 is carry-in)
    if op_name in ('v_add_co_ci_u32', 'v_sub_co_ci_u32', 'v_subrev_co_ci_u32'):
      return f"{op_name}_e64 {dst_str}, {sdst_str}, {src0_str}, {src1_str}, {src2_str}" + clamp_str
    # v_div_scale, v_mad_*64_*32 use 3 sources
    return f"{op_name} {dst_str}, {sdst_str}, {src0_str}, {src1_str}, {src2_str}" + clamp_str + omod_str

  # VOPD: dual-issue instructions
  if cls_name == 'VOPD':
    from extra.assembly.rdna3 import autogen
    opx, opy = unwrap(inst._values.get('opx', 0)), unwrap(inst._values.get('opy', 0))
    vdstx, vdsty_enc = unwrap(inst._values.get('vdstx', 0)), unwrap(inst._values.get('vdsty', 0))
    srcx0, vsrcx1 = unwrap(inst._values.get('srcx0', 0)), unwrap(inst._values.get('vsrcx1', 0))
    srcy0, vsrcy1 = unwrap(inst._values.get('srcy0', 0)), unwrap(inst._values.get('vsrcy1', 0))
    # Decode vdsty: actual = (encoded << 1) | ((vdstx & 1) ^ 1)
    vdsty = (vdsty_enc << 1) | ((vdstx & 1) ^ 1)
    try:
      opx_name = autogen.VOPDOp(opx).name.lower()
      opy_name = autogen.VOPDOp(opy).name.lower()
    except (ValueError, KeyError):
      opx_name, opy_name = f"opx_{opx}", f"opy_{opy}"
    # v_dual_mov_b32 only has 1 source
    opx_str = f"{opx_name} v{vdstx}, {fmt_src(srcx0)}" if 'mov' in opx_name else f"{opx_name} v{vdstx}, {fmt_src(srcx0)}, v{vsrcx1}"
    opy_str = f"{opy_name} v{vdsty}, {fmt_src(srcy0)}" if 'mov' in opy_name else f"{opy_name} v{vdsty}, {fmt_src(srcy0)}, v{vsrcy1}"
    return f"{opx_str} :: {opy_str}"

  # VOP3P: packed vector ops
  if cls_name == 'VOP3P':
    vdst = unwrap(inst._values.get('vdst', 0))
    src0, src1, src2 = [unwrap(inst._values.get(f, 0)) for f in ('src0', 'src1', 'src2')]
    neg = unwrap(inst._values.get('neg', 0))  # neg_lo
    neg_hi = unwrap(inst._values.get('neg_hi', 0))
    opsel = unwrap(inst._values.get('opsel', 0))
    opsel_hi = unwrap(inst._values.get('opsel_hi', 0))
    opsel_hi2 = unwrap(inst._values.get('opsel_hi2', 0))
    clmp = unwrap(inst._values.get('clmp', 0))
    # WMMA ops have special register widths
    is_wmma = 'wmma' in op_name
    # Determine number of sources (dot ops are 3-src, most are 2-src)
    is_3src = any(x in op_name for x in ('fma', 'mad', 'dot', 'wmma'))
    # Format source operands
    def fmt_vop3p_src(v, reg_cnt=1):
      if v >= 256: return _vreg(v - 256, reg_cnt)
      if v <= 105: return _sreg(v, reg_cnt) if reg_cnt > 1 else f"s{v}"
      if v == 106 and reg_cnt == 2: return "vcc"
      if v == 126 and reg_cnt == 2: return "exec"
      return fmt_src(v)
    # WMMA: f16/bf16 use 8-reg sources, iu8 uses 4-reg, iu4 uses 2-reg; all have 8-reg dst
    if is_wmma:
      src_cnt = 2 if 'iu4' in op_name else 4 if 'iu8' in op_name else 8
      src0_str = _vreg(src0 - 256, src_cnt) if src0 >= 256 else fmt_vop3p_src(src0, src_cnt)
      src1_str = _vreg(src1 - 256, src_cnt) if src1 >= 256 else fmt_vop3p_src(src1, src_cnt)
      src2_str = _vreg(src2 - 256, 8) if src2 >= 256 else fmt_vop3p_src(src2, 8)
      dst_str = _vreg(vdst, 8)
    else:
      src0_str = fmt_vop3p_src(src0)
      src1_str = fmt_vop3p_src(src1)
      src2_str = fmt_vop3p_src(src2)
      dst_str = f"v{vdst}"
    # Build modifiers - VOP3P uses op_sel, op_sel_hi, neg_lo, neg_hi
    mods = []
    # op_sel: selects high/low half of each source
    if opsel:
      if is_3src:
        mods.append(f"op_sel:[{opsel & 1},{(opsel >> 1) & 1},{(opsel >> 2) & 1}]")
      else:
        mods.append(f"op_sel:[{opsel & 1},{(opsel >> 1) & 1}]")
    # op_sel_hi: selects high half for upper result lane (default [1,1] or [1,1,1])
    # opsel_hi is bits 0-1, opsel_hi2 is bit 2 (for src2)
    full_opsel_hi = opsel_hi | (opsel_hi2 << 2)
    default_opsel_hi = 0b111 if is_3src else 0b11
    if full_opsel_hi != default_opsel_hi:
      if is_3src:
        mods.append(f"op_sel_hi:[{full_opsel_hi & 1},{(full_opsel_hi >> 1) & 1},{(full_opsel_hi >> 2) & 1}]")
      else:
        mods.append(f"op_sel_hi:[{full_opsel_hi & 1},{(full_opsel_hi >> 1) & 1}]")
    # neg_lo: negate lower half of source
    if neg:
      if is_3src:
        mods.append(f"neg_lo:[{neg & 1},{(neg >> 1) & 1},{(neg >> 2) & 1}]")
      else:
        mods.append(f"neg_lo:[{neg & 1},{(neg >> 1) & 1}]")
    # neg_hi: negate upper half of source
    if neg_hi:
      if is_3src:
        mods.append(f"neg_hi:[{neg_hi & 1},{(neg_hi >> 1) & 1},{(neg_hi >> 2) & 1}]")
      else:
        mods.append(f"neg_hi:[{neg_hi & 1},{(neg_hi >> 1) & 1}]")
    if clmp: mods.append("clamp")
    mod_str = " " + " ".join(mods) if mods else ""
    if is_3src:
      return f"{op_name} {dst_str}, {src0_str}, {src1_str}, {src2_str}{mod_str}"
    return f"{op_name} {dst_str}, {src0_str}, {src1_str}{mod_str}"

  # VINTERP: interpolation instructions
  if cls_name == 'VINTERP':
    vdst = unwrap(inst._values.get('vdst', 0))
    src0, src1, src2 = [unwrap(inst._values.get(f, 0)) for f in ('src0', 'src1', 'src2')]
    waitexp = unwrap(inst._values.get('waitexp', 0))
    neg = unwrap(inst._values.get('neg', 0))
    clmp = unwrap(inst._values.get('clmp', 0))
    opsel = unwrap(inst._values.get('opsel', 0))
    def fmt_vi_src(v, neg_bit):
      s = f"v{v - 256}" if v >= 256 else fmt_src(v)
      if neg_bit: s = f"-{s}"
      return s
    src0_str = fmt_vi_src(src0, neg & 1)
    src1_str = fmt_vi_src(src1, neg & 2)
    src2_str = fmt_vi_src(src2, neg & 4)
    # LLVM doesn't use .l/.h suffix for vinterp dst
    dst_str = f"v{vdst}"
    mods = []
    if waitexp: mods.append(f"wait_exp:{waitexp}")
    if clmp: mods.append("clamp")
    mod_str = " " + " ".join(mods) if mods else ""
    return f"{op_name} {dst_str}, {src0_str}, {src1_str}, {src2_str}{mod_str}"

  # MUBUF: buffer load/store
  if cls_name == 'MUBUF':
    vdata, vaddr = unwrap(inst._values.get('vdata', 0)), unwrap(inst._values.get('vaddr', 0))
    srsrc, soffset = unwrap(inst._values.get('srsrc', 0)), unwrap(inst._values.get('soffset', 0))
    offset = unwrap(inst._values.get('offset', 0))
    offen, idxen = unwrap(inst._values.get('offen', 0)), unwrap(inst._values.get('idxen', 0))
    glc, dlc, slc = unwrap(inst._values.get('glc', 0)), unwrap(inst._values.get('dlc', 0)), unwrap(inst._values.get('slc', 0))
    tfe = unwrap(inst._values.get('tfe', 0))
    # Special ops with no operands
    if op_name in ('buffer_gl0_inv', 'buffer_gl1_inv'): return op_name
    # Determine data width from op name
    # d16 formats: _x and _xy use 1 reg, _xyz and _xyzw use 2 regs
    # regular formats: _x=1, _xy=2, _xyz=3, _xyzw=4
    # atomic u64 uses 2 regs, cmpswap doubles width (compare + swap)
    if 'd16' in op_name:
      width = 2 if any(x in op_name for x in ('xyz', 'xyzw')) else 1
    elif 'atomic' in op_name:
      # cmpswap uses 2 regs for b32, 4 for b64; other atomics use 1 for b32, 2 for b64/u64/i64
      base_width = 2 if any(x in op_name for x in ('b64', 'u64', 'i64')) else 1
      width = base_width * 2 if 'cmpswap' in op_name else base_width
    else:
      width = {'b32':1, 'b64':2, 'b96':3, 'b128':4, 'b16':1, 'x':1, 'xy':2, 'xyz':3, 'xyzw':4}.get(op_name.split('_')[-1], 1)
    # tfe adds 1 extra VGPR for texture fault status
    if tfe: width += 1
    is_store = 'store' in op_name
    # Format vaddr
    if offen and idxen: vaddr_str = f"v[{vaddr}:{vaddr+1}]"
    elif offen or idxen: vaddr_str = f"v{vaddr}"
    else: vaddr_str = "off"
    # Format srsrc (4-aligned SGPR quad)
    srsrc_base = srsrc * 4
    srsrc_str = f"s[{srsrc_base}:{srsrc_base+3}]"
    # Format soffset - use decode_src for proper constant handling
    soff_str = decode_src(soffset)
    # Build modifiers
    mods = []
    if offen: mods.append("offen")
    if idxen: mods.append("idxen")
    if offset: mods.append(f"offset:{offset}")
    if glc: mods.append("glc")
    if dlc: mods.append("dlc")
    if slc: mods.append("slc")
    if tfe: mods.append("tfe")
    mod_str = " " + " ".join(mods) if mods else ""
    if is_store:
      return f"{op_name} {_vreg(vdata, width)}, {vaddr_str}, {srsrc_str}, {soff_str}{mod_str}"
    return f"{op_name} {_vreg(vdata, width)}, {vaddr_str}, {srsrc_str}, {soff_str}{mod_str}"

  # MTBUF: typed buffer load/store
  if cls_name == 'MTBUF':
    vdata, vaddr = unwrap(inst._values.get('vdata', 0)), unwrap(inst._values.get('vaddr', 0))
    srsrc, soffset = unwrap(inst._values.get('srsrc', 0)), unwrap(inst._values.get('soffset', 0))
    offset, fmt = unwrap(inst._values.get('offset', 0)), unwrap(inst._values.get('format', 0))
    offen, idxen = unwrap(inst._values.get('offen', 0)), unwrap(inst._values.get('idxen', 0))
    glc, dlc, slc = unwrap(inst._values.get('glc', 0)), unwrap(inst._values.get('dlc', 0)), unwrap(inst._values.get('slc', 0))
    # Format vaddr
    if offen and idxen: vaddr_str = f"v[{vaddr}:{vaddr+1}]"
    elif offen or idxen: vaddr_str = f"v{vaddr}"
    else: vaddr_str = "off"
    # Format srsrc (4-aligned SGPR quad, or ttmp)
    srsrc_base = srsrc * 4
    if 108 <= srsrc_base <= 123:
      srsrc_str = f"ttmp[{srsrc_base-108}:{srsrc_base-108+3}]"
    else:
      srsrc_str = f"s[{srsrc_base}:{srsrc_base+3}]"
    # Format soffset - use decode_src for proper special register handling
    soff_str = decode_src(soffset)
    # Build modifiers - idxen must come before offen for LLVM
    mods = [f"format:{fmt}"]
    if idxen: mods.append("idxen")
    if offen: mods.append("offen")
    if offset: mods.append(f"offset:{offset}")
    if glc: mods.append("glc")
    if dlc: mods.append("dlc")
    if slc: mods.append("slc")
    # Determine vdata width: d16 xyz/xyzw use 2 regs, d16 x/xy use 1 reg
    if 'd16' in op_name:
      width = 2 if any(x in op_name for x in ('xyz', 'xyzw')) else 1
    else:
      width = {'x':1, 'xy':2, 'xyz':3, 'xyzw':4}.get(op_name.split('_')[-1], 1)
    return f"{op_name} {_vreg(vdata, width)}, {vaddr_str}, {srsrc_str}, {soff_str} {' '.join(mods)}"

  # SOP1/SOP2/SOPC/SOPK
  if cls_name in ('SOP1', 'SOP2', 'SOPC', 'SOPK'):
    sizes = _parse_sop_sizes(op_name)
    dst_cnt, src0_cnt = sizes[0], sizes[1]
    src1_cnt = sizes[2] if len(sizes) > 2 else src0_cnt
    if cls_name == 'SOP1':
      if op_name == 's_getpc_b64': return f"{op_name} {_fmt_sdst(unwrap(inst._values.get('sdst', 0)), 2)}"
      if op_name in ('s_setpc_b64', 's_rfe_b64'): return f"{op_name} {_fmt_ssrc(unwrap(inst._values.get('ssrc0', 0)), 2)}"
      if op_name == 's_swappc_b64': return f"{op_name} {_fmt_sdst(unwrap(inst._values.get('sdst', 0)), 2)}, {_fmt_ssrc(unwrap(inst._values.get('ssrc0', 0)), 2)}"
      if op_name in ('s_sendmsg_rtn_b32', 's_sendmsg_rtn_b64'):
        msg_id = unwrap(inst._values.get('ssrc0', 0))
        msg_names = {128: 'MSG_RTN_GET_DOORBELL', 129: 'MSG_RTN_GET_DDID', 130: 'MSG_RTN_GET_TMA', 131: 'MSG_RTN_GET_REALTIME', 132: 'MSG_RTN_SAVE_WAVE', 133: 'MSG_RTN_GET_TBA'}
        msg = msg_names.get(msg_id, str(msg_id))
        return f"{op_name} {_fmt_sdst(unwrap(inst._values.get('sdst', 0)), 2 if 'b64' in op_name else 1)}, sendmsg({msg})"
      return f"{op_name} {_fmt_sdst(unwrap(inst._values.get('sdst', 0)), dst_cnt)}, {_fmt_ssrc(unwrap(inst._values.get('ssrc0', 0)), src0_cnt)}"
    if cls_name == 'SOP2':
      sdst, ssrc0, ssrc1 = [unwrap(inst._values.get(f, 0)) for f in ('sdst', 'ssrc0', 'ssrc1')]
      return f"{op_name} {_fmt_sdst(sdst, dst_cnt)}, {_fmt_ssrc(ssrc0, src0_cnt)}, {_fmt_ssrc(ssrc1, src1_cnt)}"
    if cls_name == 'SOPC':
      return f"{op_name} {_fmt_ssrc(unwrap(inst._values.get('ssrc0', 0)), src0_cnt)}, {_fmt_ssrc(unwrap(inst._values.get('ssrc1', 0)), src1_cnt)}"
    if cls_name == 'SOPK':
      sdst, simm16 = unwrap(inst._values.get('sdst', 0)), unwrap(inst._values.get('simm16', 0))
      if op_name == 's_version': return f"{op_name} 0x{simm16:x}"
      if op_name in ('s_setreg_b32', 's_getreg_b32'):
        # Decode hwreg: (size-1) << 11 | offset << 6 | id
        hwreg_id, hwreg_offset, hwreg_size = simm16 & 0x3f, (simm16 >> 6) & 0x1f, ((simm16 >> 11) & 0x1f) + 1
        # GFX11+ hwreg names (IDs 16-17 are TBA which are not supported on GFX11, IDs 18-19 are PERF_SNAPSHOT)
        hwreg_names = {1: 'HW_REG_MODE', 2: 'HW_REG_STATUS', 3: 'HW_REG_TRAPSTS', 4: 'HW_REG_HW_ID',
                       5: 'HW_REG_GPR_ALLOC', 6: 'HW_REG_LDS_ALLOC', 7: 'HW_REG_IB_STS',
                       15: 'HW_REG_SH_MEM_BASES',
                       18: 'HW_REG_PERF_SNAPSHOT_PC_LO', 19: 'HW_REG_PERF_SNAPSHOT_PC_HI',
                       20: 'HW_REG_FLAT_SCR_LO', 21: 'HW_REG_FLAT_SCR_HI', 22: 'HW_REG_XNACK_MASK',
                       23: 'HW_REG_HW_ID1', 24: 'HW_REG_HW_ID2', 25: 'HW_REG_POPS_PACKER',
                       28: 'HW_REG_IB_STS2'}
        # For unsupported registers (TBA_LO/HI, TMA_LO/HI on GFX11), output raw simm16 value
        if hwreg_id in (16, 17, 18, 19) and hwreg_id not in hwreg_names:
          # Unsupported on GFX11 - use raw encoding
          hwreg_str = f"0x{simm16:x}"
        else:
          hwreg_name = hwreg_names.get(hwreg_id, str(hwreg_id))
          hwreg_str = f"hwreg({hwreg_name}, {hwreg_offset}, {hwreg_size})"
        if op_name == 's_setreg_b32':
          return f"{op_name} {hwreg_str}, {_fmt_sdst(sdst, 1)}"
        return f"{op_name} {_fmt_sdst(sdst, 1)}, {hwreg_str}"
      return f"{op_name} {_fmt_sdst(sdst, dst_cnt)}, 0x{simm16:x}"

  # Generic fallback
  def fmt(n, v):
    v = unwrap(v)
    if n in SRC_FIELDS: return fmt_src(v) if v != 255 else "0xff"
    if n in ('sdst', 'vdst'): return f"{'s' if n == 'sdst' else 'v'}{v}"
    return f"v{v}" if n == 'vsrc1' else f"0x{v:x}" if n == 'simm16' else str(v)
  ops = [fmt(n, inst._values.get(n, 0)) for n in inst._fields if n not in ('encoding', 'op')]
  return f"{op_name} {', '.join(ops)}" if ops else op_name

# Assembler
SPECIAL_REGS = {'vcc_lo': RawImm(106), 'vcc_hi': RawImm(107), 'null': RawImm(124), 'off': RawImm(124), 'm0': RawImm(125), 'exec_lo': RawImm(126), 'exec_hi': RawImm(127), 'scc': RawImm(253)}
FLOAT_CONSTS = {'0.5': 0.5, '-0.5': -0.5, '1.0': 1.0, '-1.0': -1.0, '2.0': 2.0, '-2.0': -2.0, '4.0': 4.0, '-4.0': -4.0}
REG_MAP = {'s': SGPR, 'v': VGPR, 't': TTMP, 'ttmp': TTMP}

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
  if m := re.match(r'^([svt](?:tmp)?)\[(\d+):(\d+)\]$', op): return (REG_MAP[m.group(1)][int(m.group(2)):int(m.group(3))+1], neg, abs_, hi_half)
  if m := re.match(r'^([svt](?:tmp)?)(\d+)$', op):
    return (REG_MAP[m.group(1)](int(m.group(2)), 1, hi_half), neg, abs_, hi_half)
  # hwreg(name, offset, size) -> simm16 encoding
  if m := re.match(r'^hwreg\((\w+),\s*(\d+),\s*(\d+)\)$', op):
    hwreg_names = {'hw_reg_mode': 1, 'hw_reg_status': 2, 'hw_reg_trapsts': 3, 'hw_reg_hw_id': 4,
                   'hw_reg_gpr_alloc': 5, 'hw_reg_lds_alloc': 6, 'hw_reg_ib_sts': 7,
                   'hw_reg_sh_mem_bases': 15, 'hw_reg_tba_lo': 16, 'hw_reg_tba_hi': 17, 'hw_reg_tma_lo': 18,
                   'hw_reg_tma_hi': 19, 'hw_reg_flat_scr_lo': 20, 'hw_reg_flat_scr_hi': 21, 'hw_reg_xnack_mask': 22,
                   'hw_reg_hw_id1': 23, 'hw_reg_hw_id2': 24, 'hw_reg_pops_packer': 25, 'hw_reg_ib_sts2': 28}
    name_str = m.group(1).lower()
    hwreg_id = hwreg_names.get(name_str, int(name_str) if name_str.isdigit() else None)
    if hwreg_id is None: raise ValueError(f"unknown hwreg name: {name_str}")
    offset, size = int(m.group(2)), int(m.group(3))
    simm16 = ((size - 1) << 11) | (offset << 6) | hwreg_id
    return (simm16, neg, abs_, hi_half)
  raise ValueError(f"cannot parse operand: {op}")

SMEM_OPS = {'s_load_b32', 's_load_b64', 's_load_b128', 's_load_b256', 's_load_b512',
            's_buffer_load_b32', 's_buffer_load_b64', 's_buffer_load_b128', 's_buffer_load_b256', 's_buffer_load_b512'}
SOP1_SRC_ONLY = {'s_setpc_b64', 's_rfe_b64'}
SOP1_MSG_IMM = {'s_sendmsg_rtn_b32', 's_sendmsg_rtn_b64'}
SOPK_IMM_ONLY = {'s_version'}
SOPK_IMM_FIRST = {'s_setreg_b32'}
SOPK_UNSUPPORTED = {'s_setreg_imm32_b32'}

def asm(text: str) -> Inst:
  from extra.assembly.rdna3 import autogen
  text = text.strip()
  clamp = 'clamp' in text.lower()
  if clamp: text = re.sub(r'\s+clamp\s*$', '', text, flags=re.I)
  modifiers = {}
  if m := re.search(r'\s+wait_exp:(\d+)', text, re.I): modifiers['waitexp'] = int(m.group(1)); text = text[:m.start()] + text[m.end():]
  parts = text.replace(',', ' ').split()
  if not parts: raise ValueError("empty instruction")
  mnemonic, op_str = parts[0].lower(), text[len(parts[0]):].strip()
  # Handle s_waitcnt specially before operand parsing
  if mnemonic == 's_waitcnt':
    vmcnt, expcnt, lgkmcnt = 0x3f, 0x7, 0x3f
    for part in op_str.replace(',', ' ').split():
      if m := re.match(r'vmcnt\((\d+)\)', part): vmcnt = int(m.group(1))
      elif m := re.match(r'expcnt\((\d+)\)', part): expcnt = int(m.group(1))
      elif m := re.match(r'lgkmcnt\((\d+)\)', part): lgkmcnt = int(m.group(1))
      elif re.match(r'^0x[0-9a-f]+$|^\d+$', part): return autogen.s_waitcnt(simm16=int(part, 0))
    return autogen.s_waitcnt(simm16=waitcnt(vmcnt, expcnt, lgkmcnt))
  # Handle VOPD dual-issue instructions: opx dst, src :: opy dst, src
  if '::' in text:
    x_part, y_part = text.split('::')
    x_parts, y_parts = x_part.strip().replace(',', ' ').split(), y_part.strip().replace(',', ' ').split()
    opx_name, opy_name = x_parts[0].upper(), y_parts[0].upper()
    opx, opy = autogen.VOPDOp[opx_name], autogen.VOPDOp[opy_name]
    x_ops, y_ops = [parse_operand(p)[0] for p in x_parts[1:]], [parse_operand(p)[0] for p in y_parts[1:]]
    vdstx, srcx0 = x_ops[0], x_ops[1] if len(x_ops) > 1 else 0
    vsrcx1 = x_ops[2] if len(x_ops) > 2 else VGPR(0)
    vdsty, srcy0 = y_ops[0], y_ops[1] if len(y_ops) > 1 else 0
    vsrcy1 = y_ops[2] if len(y_ops) > 2 else VGPR(0)
    # Handle fmaak/fmamk literals (4th operand on x or y side)
    lit = None
    if 'fmaak' in opx_name.lower() and len(x_ops) > 3: lit = unwrap(x_ops[3])
    elif 'fmamk' in opx_name.lower() and len(x_ops) > 3: lit, vsrcx1 = unwrap(x_ops[2]), x_ops[3]
    elif 'fmaak' in opy_name.lower() and len(y_ops) > 3: lit = unwrap(y_ops[3])
    elif 'fmamk' in opy_name.lower() and len(y_ops) > 3: lit, vsrcy1 = unwrap(y_ops[2]), y_ops[3]
    return autogen.VOPD(opx, opy, vdstx=vdstx, vdsty=vdsty, srcx0=srcx0, vsrcx1=vsrcx1, srcy0=srcy0, vsrcy1=vsrcy1, literal=lit)
  operands, current, depth, in_pipe = [], "", 0, False
  for ch in op_str:
    if ch in '[(': depth += 1
    elif ch in '])': depth -= 1
    elif ch == '|': in_pipe = not in_pipe
    if ch == ',' and depth == 0 and not in_pipe: operands.append(current.strip()); current = ""
    else: current += ch
  if current.strip(): operands.append(current.strip())
  parsed = [parse_operand(op) for op in operands]
  values = [p[0] for p in parsed]
  neg_bits = sum((1 << (i-1)) for i, p in enumerate(parsed) if i > 0 and p[1])
  abs_bits = sum((1 << (i-1)) for i, p in enumerate(parsed) if i > 0 and p[2])
  opsel_bits = (8 if len(parsed) > 0 and parsed[0][3] else 0) | sum((1 << i) for i, p in enumerate(parsed[1:4]) if p[3])
  lit = None
  if mnemonic in ('v_fmaak_f32', 'v_fmaak_f16') and len(values) == 4: lit, values = unwrap(values[3]), values[:3]
  elif mnemonic in ('v_fmamk_f32', 'v_fmamk_f16') and len(values) == 4: lit, values = unwrap(values[2]), [values[0], values[1], values[3]]
  vcc_ops = {'v_add_co_ci_u32', 'v_sub_co_ci_u32', 'v_subrev_co_ci_u32', 'v_add_co_u32', 'v_sub_co_u32', 'v_subrev_co_u32'}
  if mnemonic.replace('_e32', '') in vcc_ops and len(values) >= 5: values = [values[0], values[2], values[3]]
  if mnemonic.startswith('v_cmp') and len(values) >= 3 and operands[0].strip().lower() in ('vcc_lo', 'vcc_hi', 'vcc'):
    values = values[1:]
  vop3sd_ops = {'v_div_scale_f32', 'v_div_scale_f64'}
  if mnemonic in vop3sd_ops and len(parsed) >= 5:
    neg_bits = sum((1 << i) for i, p in enumerate(parsed[2:5]) if p[1])
    abs_bits = sum((1 << i) for i, p in enumerate(parsed[2:5]) if p[2])
  if mnemonic in SOPK_UNSUPPORTED: raise ValueError(f"unsupported instruction: {mnemonic}")
  elif mnemonic in SOP1_SRC_ONLY:
    return getattr(autogen, mnemonic)(ssrc0=values[0])
  elif mnemonic in SOP1_MSG_IMM:
    return getattr(autogen, mnemonic)(sdst=values[0], ssrc0=RawImm(unwrap(values[1])))
  elif mnemonic in SOPK_IMM_ONLY:
    return getattr(autogen, mnemonic)(simm16=values[0])
  elif mnemonic in SOPK_IMM_FIRST:
    return getattr(autogen, mnemonic)(simm16=values[0], sdst=values[1])
  elif mnemonic in SMEM_OPS and len(operands) >= 3 and re.match(r'^-?[0-9]|^-?0x', operands[2].strip().lower()):
    return getattr(autogen, mnemonic)(sdata=values[0], sbase=values[1], offset=values[2], soffset=RawImm(124))
  elif mnemonic.startswith('buffer_') and len(operands) >= 2 and operands[1].strip().lower() == 'off':
    return getattr(autogen, mnemonic)(vdata=values[0], vaddr=0, srsrc=values[2], soffset=RawImm(unwrap(values[3])) if len(values) > 3 else RawImm(0))
  elif (mnemonic.startswith('flat_load') or mnemonic.startswith('global_load') or mnemonic.startswith('scratch_load')) and len(values) >= 3:
    offset = int(m.group(1)) if (m := re.search(r'offset:(-?\d+)', op_str)) else 0
    return getattr(autogen, mnemonic)(vdst=values[0], addr=values[1], saddr=values[2], offset=offset)
  elif (mnemonic.startswith('flat_store') or mnemonic.startswith('global_store') or mnemonic.startswith('scratch_store')) and len(values) >= 3:
    offset = int(m.group(1)) if (m := re.search(r'offset:(-?\d+)', op_str)) else 0
    return getattr(autogen, mnemonic)(addr=values[0], data=values[1], saddr=values[2], offset=offset)
  for suffix in (['_e32', ''] if not (neg_bits or abs_bits or clamp) else ['', '_e32']):
    if hasattr(autogen, name := mnemonic.replace('.', '_') + suffix):
      use_opsel = 'opsel' in getattr(autogen, name).func._fields
      vals = [type(v)(v.idx, v.count, False) if isinstance(v, Reg) and v.hi and use_opsel else v for v in values]
      inst = getattr(autogen, name)(*vals, literal=lit, **modifiers)
      if neg_bits and 'neg' in inst._fields: inst._values['neg'] = neg_bits
      if opsel_bits and use_opsel: inst._values['opsel'] = opsel_bits
      if abs_bits and 'abs' in inst._fields: inst._values['abs'] = abs_bits
      if clamp and 'clmp' in inst._fields: inst._values['clmp'] = 1
      return inst
  raise ValueError(f"unknown instruction: {mnemonic}")
