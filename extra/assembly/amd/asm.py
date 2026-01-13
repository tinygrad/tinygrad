# RDNA3/RDNA4/CDNA assembler
from __future__ import annotations
import re
from extra.assembly.amd.dsl import Reg, s, v, ttmp
from extra.assembly.amd.dsl import VCC_LO, VCC_HI, VCC, EXEC_LO, EXEC_HI, EXEC, SCC, M0, NULL

# Assembler-specific types (not part of clean DSL)
class RawImm(Reg):
  """Raw immediate value - bypasses normal encoding, used for special register encodings."""
  def __init__(self, val: int): super().__init__(val, 1)

class SrcMod(Reg):
  """Source with modifiers - wraps a value with neg/abs flags."""
  def __init__(self, val: int, neg: bool = False, abs_: bool = False):
    super().__init__(255 if not (-16 <= val <= 64) else (128 + val if val >= 0 else 192 - val), 1)
    self.val, self.neg, self.abs_ = val, neg, abs_

# Type aliases for register factories
_RegFactory = type(s)
SGPR, VGPR, TTMP = s, v, ttmp
OFF = NULL  # OFF is alias for NULL (encoding 124)

# Float encoding constants
FLOAT_ENC = {0.5: 240, -0.5: 241, 1.0: 242, -1.0: 243, 2.0: 244, -2.0: 245, 4.0: 246, -4.0: 247}
from extra.assembly.amd.autogen.rdna3 import ins
from extra.assembly.amd.autogen.rdna3.ins import VOP2Op, VOPDOp, SOPKOp
from extra.assembly.amd.autogen.rdna3.enum import BufFmt
from extra.assembly.amd.autogen.rdna4 import ins as rdna4_ins

# Re-export disasm for backwards compatibility
from extra.assembly.amd.disasm import disasm, HWREG, HWREG_RDNA4

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# RDNA unified buffer format
BUF_FMT = {e.name: e.value for e in BufFmt}
_BUF_FMT_EXT = {'BUF_FMT_32_32_32_32_SINT': 62, 'BUF_FMT_32_32_32_32_FLOAT': 63, 'BUF_FMT_8_FLOAT': 108}
BUF_FMT.update(_BUF_FMT_EXT)
def _parse_buf_fmt_combo(s: str) -> int:
  parts = [p.strip().replace('BUF_DATA_FORMAT_', '').replace('BUF_NUM_FORMAT_', '') for p in s.split(',')]
  return BUF_FMT.get(f'BUF_FMT_{parts[0]}_{parts[1]}') if len(parts) == 2 else None

def waitcnt(vmcnt: int = 0x3f, expcnt: int = 0x7, lgkmcnt: int = 0x3f) -> int:
  return (expcnt & 0x7) | ((lgkmcnt & 0x3f) << 4) | ((vmcnt & 0x3f) << 10)

# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLER
# ═══════════════════════════════════════════════════════════════════════════════

SPEC_REGS = {'vcc_lo': RawImm(106), 'vcc_hi': RawImm(107), 'vcc': RawImm(106), 'null': RawImm(124), 'off': RawImm(124), 'm0': RawImm(125),
             'exec_lo': RawImm(126), 'exec_hi': RawImm(127), 'exec': RawImm(126), 'scc': RawImm(253), 'src_scc': RawImm(253)}
FLOATS = {str(k): k for k in FLOAT_ENC}  # Valid float literal strings: '0.5', '-0.5', '1.0', etc.
REG_MAP: dict[str, _RegFactory] = {'s': s, 'v': v, 't': ttmp, 'ttmp': ttmp}
SMEM_OPS = {'s_load_b32', 's_load_b64', 's_load_b96', 's_load_b128', 's_load_b256', 's_load_b512',
            's_load_i8', 's_load_u8', 's_load_i16', 's_load_u16',
            's_buffer_load_b32', 's_buffer_load_b64', 's_buffer_load_b96', 's_buffer_load_b128', 's_buffer_load_b256', 's_buffer_load_b512',
            's_buffer_load_i8', 's_buffer_load_u8', 's_buffer_load_i16', 's_buffer_load_u16',
            's_atc_probe', 's_atc_probe_buffer'}
SPEC_DSL = {'vcc_lo': 'VCC_LO', 'vcc_hi': 'VCC_HI', 'vcc': 'VCC_LO', 'null': 'NULL', 'off': 'OFF', 'm0': 'M0',
            'exec_lo': 'EXEC_LO', 'exec_hi': 'EXEC_HI', 'exec': 'EXEC_LO', 'scc': 'SCC', 'src_scc': 'SCC'}

def _op2dsl(op: str) -> str:
  op = op.strip()
  neg = op.startswith('-') and not (op[1:2].isdigit() or (len(op) > 2 and op[1] == '0' and op[2] in 'xX'))
  if neg: op = op[1:]
  if op.startswith('neg(') and op.endswith(')'): neg = True; op = op[4:-1]
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
  's_load_dword': 's_load_b32', 's_load_dwordx2': 's_load_b64', 's_load_dwordx4': 's_load_b128',
  's_load_dwordx8': 's_load_b256', 's_load_dwordx16': 's_load_b512',
  's_buffer_load_dword': 's_buffer_load_b32', 's_buffer_load_dwordx2': 's_buffer_load_b64',
  's_buffer_load_dwordx4': 's_buffer_load_b128', 's_buffer_load_dwordx8': 's_buffer_load_b256',
  's_buffer_load_dwordx16': 's_buffer_load_b512',
  'v_cvt_pknorm_i16_f16': 'v_cvt_pk_norm_i16_f16', 'v_cvt_pknorm_u16_f16': 'v_cvt_pk_norm_u16_f16',
  'v_add3_nc_u32': 'v_add3_u32', 'v_xor_add_u32': 'v_xad_u32',
  'v_interp_p2_new_f32': 'v_interp_p2_f32',
  's_ff1_i32_b32': 's_ctz_i32_b32', 's_ff1_i32_b64': 's_ctz_i32_b64',
  's_flbit_i32_b32': 's_clz_i32_u32', 's_flbit_i32_b64': 's_clz_i32_u64', 's_flbit_i32': 's_cls_i32', 's_flbit_i32_i64': 's_cls_i32_i64',
  's_andn1_saveexec_b32': 's_and_not0_saveexec_b32', 's_andn1_saveexec_b64': 's_and_not0_saveexec_b64',
  's_andn1_wrexec_b32': 's_and_not0_wrexec_b32', 's_andn1_wrexec_b64': 's_and_not0_wrexec_b64',
  's_andn2_saveexec_b32': 's_and_not1_saveexec_b32', 's_andn2_saveexec_b64': 's_and_not1_saveexec_b64',
  's_andn2_wrexec_b32': 's_and_not1_wrexec_b32', 's_andn2_wrexec_b64': 's_and_not1_wrexec_b64',
  's_orn1_saveexec_b32': 's_or_not0_saveexec_b32', 's_orn1_saveexec_b64': 's_or_not0_saveexec_b64',
  's_orn2_saveexec_b32': 's_or_not1_saveexec_b32', 's_orn2_saveexec_b64': 's_or_not1_saveexec_b64',
  's_andn2_b32': 's_and_not1_b32', 's_andn2_b64': 's_and_not1_b64',
  's_orn2_b32': 's_or_not1_b32', 's_orn2_b64': 's_or_not1_b64',
  'v_dot2c_f32_f16': 'v_dot2acc_f32_f16',
  'v_fma_legacy_f32': 'v_fma_dx9_zero_f32',
  'ds_read_b32': 'ds_load_b32', 'ds_read_b64': 'ds_load_b64', 'ds_read_b96': 'ds_load_b96', 'ds_read_b128': 'ds_load_b128',
  'ds_read_i8': 'ds_load_i8', 'ds_read_u8': 'ds_load_u8', 'ds_read_i16': 'ds_load_i16', 'ds_read_u16': 'ds_load_u16',
  'ds_read_i8_d16': 'ds_load_i8_d16', 'ds_read_u8_d16': 'ds_load_u8_d16', 'ds_read_i8_d16_hi': 'ds_load_i8_d16_hi', 'ds_read_u8_d16_hi': 'ds_load_u8_d16_hi',
  'ds_read_u16_d16': 'ds_load_u16_d16', 'ds_read_u16_d16_hi': 'ds_load_u16_d16_hi',
  'ds_read2_b32': 'ds_load_2addr_b32', 'ds_read2_b64': 'ds_load_2addr_b64',
  'ds_read2st64_b32': 'ds_load_2addr_stride64_b32', 'ds_read2st64_b64': 'ds_load_2addr_stride64_b64',
  'ds_read_addtid_b32': 'ds_load_addtid_b32', 'ds_write_addtid_b32': 'ds_store_addtid_b32',
  'ds_write_b32': 'ds_store_b32', 'ds_write_b64': 'ds_store_b64', 'ds_write_b96': 'ds_store_b96', 'ds_write_b128': 'ds_store_b128',
  'ds_write_b8': 'ds_store_b8', 'ds_write_b16': 'ds_store_b16',
  'ds_write_b8_d16_hi': 'ds_store_b8_d16_hi', 'ds_write_b16_d16_hi': 'ds_store_b16_d16_hi',
  'ds_write2_b32': 'ds_store_2addr_b32', 'ds_write2_b64': 'ds_store_2addr_b64',
  'ds_write2st64_b32': 'ds_store_2addr_stride64_b32', 'ds_write2st64_b64': 'ds_store_2addr_stride64_b64',
  'ds_wrxchg_rtn_b32': 'ds_storexchg_rtn_b32', 'ds_wrxchg_rtn_b64': 'ds_storexchg_rtn_b64',
  'ds_wrxchg2_rtn_b32': 'ds_storexchg_2addr_rtn_b32', 'ds_wrxchg2_rtn_b64': 'ds_storexchg_2addr_rtn_b64',
  'ds_wrxchg2st64_rtn_b32': 'ds_storexchg_2addr_stride64_rtn_b32', 'ds_wrxchg2st64_rtn_b64': 'ds_storexchg_2addr_stride64_rtn_b64',
}

def _apply_alias(text: str) -> str:
  mn = text.split()[0].lower() if ' ' in text else text.lower().rstrip('_')
  for m in (mn, mn.removesuffix('_e32'), mn.removesuffix('_e64')):
    if m in _ALIASES: return _ALIASES[m] + text[len(m):]
  return text

def _has(op: str, *subs) -> bool: return any(s in op for s in subs)

def get_dsl(text: str, arch: str = "rdna3") -> str:
  text, kw = _apply_alias(text.strip()), []
  # Extract modifiers
  for pat, val in [(r'\s+mul:2(?:\s|$)', 1), (r'\s+mul:4(?:\s|$)', 2), (r'\s+div:2(?:\s|$)', 3)]:
    if (m := _extract(text, pat))[0]: kw.append(f'omod={val}'); text = m[1]; break
  clamp_found = False
  if (m := _extract(text, r'\s+clamp(?:\s|$)'))[0]: clamp_found = True; text = m[1]
  opsel, m, text = None, *_extract(text, r'\s+op_sel:\[([^\]]+)\]')
  if m:
    bits, mn = [int(x.strip()) for x in m.group(1).split(',')], text.split()[0].lower()
    is3p = mn.startswith(('v_pk_', 'v_wmma_', 'v_dot', 'v_fma_mix'))
    opsel = (bits[0] | (bits[1] << 1) | (bits[2] << 2)) if len(bits) == 3 and is3p else \
            (bits[0] | (bits[1] << 1) | (bits[2] << 3)) if len(bits) == 3 else sum(b << i for i, b in enumerate(bits))
  opsel_hi_val, m, text = None, *_extract(text, r'\s+op_sel_hi:\[([^\]]+)\]')
  if m: opsel_hi_val = [int(x.strip()) for x in m.group(1).split(',')]
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
  m, text = _extract(text, r'\s+byte_sel:(\d+)'); byte_sel = int(m.group(1)) if m else None
  m, text = _extract(text, r'\s+offset0:(\d+)'); ds_off0 = int(m.group(1)) if m else None
  m, text = _extract(text, r'\s+offset1:(\d+)'); ds_off1 = int(m.group(1)) if m else None
  m, text = _extract(text, r'\s+index_key:(\d+)'); index_key = int(m.group(1)) if m else None
  if waitexp: kw.append(f'waitexp={waitexp}')
  if byte_sel is not None:
    if opsel is None: opsel = 0
    opsel |= (byte_sel << 2)
  if ds_off0 is not None: kw.append(f'offset0={ds_off0}')
  if ds_off1 is not None: kw.append(f'offset1={ds_off1}')
  if index_key is not None: kw.append(f'opsel={index_key}')

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
  sop1_no_dest = ('s_alloc_vgpr', 's_barrier_init', 's_barrier_join', 's_barrier_signal', 's_barrier_signal_isfirst', 's_sleep_var')
  if mn in sop1_no_dest:
    return f"{mn}(sdst=RawImm(128), ssrc0={args[0]})"
  if mn in ('s_setpc_b64', 's_rfe_b64'): return f"{mn}(ssrc0={args[0]})"
  if mn in ('s_sendmsg_rtn_b32', 's_sendmsg_rtn_b64'): return f"{mn}(sdst={args[0]}, ssrc0=RawImm({args[1].strip()}))"
  if mn == 's_version': return f"{mn}(simm16={args[0]})"
  if mn == 's_setreg_b32': return f"{mn}(simm16={args[0]}, sdst={args[1]})"

  # Export instructions (RDNA4 VEXPORT)
  if mn == 'export':
    target_map = {**{f'mrt{i}': i for i in range(8)}, 'mrtz': 8, **{f'pos{i}': 12+i for i in range(4)}}
    m, exp_str = _extract(op_str, r'\s+done(?:\s|$)')
    done_val = 1 if m else 0
    exp_parts = exp_str.replace(',', ' ').split()
    target_name = exp_parts[0].lower().strip()
    target = target_map.get(target_name, 0)
    vsrcs, en = [], 0
    for i, o in enumerate(exp_parts[1:5]):
      o = o.strip().lower()
      if o == 'off': vsrcs.append('v[0]')
      else: vsrcs.append(_op2dsl(o)); en |= (1 << i)
    return f"VEXPORT(target={target}, en={en}, vsrc0={vsrcs[0]}, vsrc1={vsrcs[1]}, vsrc2={vsrcs[2]}, vsrc3={vsrcs[3]}, done={done_val})"

  # SMEM
  if mn in SMEM_OPS:
    gs, ds = ", glc=1" if glc else "", ", dlc=1" if dlc else ""
    off_field = "ioffset" if arch == "rdna4" else "offset"
    th_s, scope_s, smem_str = "", "", op_str
    if arch == "rdna4":
      m, smem_str = _extract(op_str, r'\s+th:TH_(\w+)')
      th_val = {'LOAD_RT': 0, 'LOAD_NT': 1, 'LOAD_HT': 2, 'LOAD_LU': 3, 'STORE_RT': 0, 'STORE_NT': 1, 'STORE_HT': 2, 'STORE_LU': 3}.get(m.group(1), 0) if m else None
      m, smem_str = _extract(smem_str, r'\s+scope:SCOPE_(\w+)')
      scope_val = {'CU': 0, 'SE': 1, 'DEV': 2, 'SYS': 3}.get(m.group(1), 0) if m else None
      if scope_val is None:
        m, smem_str = _extract(smem_str, r'\s+scope:(0?x?[0-9a-fA-F]+)')
        scope_val = int(m.group(1), 0) if m else None
      th_s = f", th={th_val}" if th_val else ""
      scope_s = f", scope={scope_val}" if scope_val else ""
    smem_ops = _parse_ops(smem_str)
    smem_args = [_op2dsl(o) for o in smem_ops]
    if len(smem_ops) >= 3 and re.match(r'^-?[0-9]|^-?0x', smem_ops[2].strip().lower()):
      return f"{mn}(sdata={smem_args[0]}, sbase={smem_args[1]}, {off_field}={smem_ops[2].strip()}, soffset=RawImm(124){gs}{ds}{th_s}{scope_s})"
    if off_val and len(smem_ops) >= 3: return f"{mn}(sdata={smem_args[0]}, sbase={smem_args[1]}, {off_field}={off_val}, soffset={smem_args[2]}{gs}{ds}{th_s}{scope_s})"
    if len(smem_ops) >= 3: return f"{mn}(sdata={smem_args[0]}, sbase={smem_args[1]}, soffset={smem_args[2]}{gs}{ds}{th_s}{scope_s})"

  # Buffer (MUBUF/MTBUF/VBUFFER) instructions
  if mn.startswith(('buffer_', 'tbuffer_')):
    is_tbuf = mn.startswith('tbuffer_')
    fmt_num = None
    if fmt_val is not None:
      if fmt_val.isdigit(): fmt_num = int(fmt_val)
      else: fmt_num = BUF_FMT.get(fmt_val.replace(' ', '')) or _parse_buf_fmt_combo(fmt_val)
    if mn in ('buffer_gl0_inv', 'buffer_gl1_inv', 'buffer_wbl2', 'buffer_inv'): return f"{mn}()"
    if arch == "rdna4":
      m, buf_text = _extract(op_str, r'\s+th:TH_(\w+)')
      th_val = {'LOAD_RT': 0, 'LOAD_NT': 1, 'LOAD_HT': 2, 'LOAD_BYPASS': 3, 'LOAD_LU': 4, 'LOAD_RT_NT': 5, 'LOAD_NT_HT': 6, 'LOAD_RT_WB': 7,
                'STORE_RT': 0, 'STORE_NT': 1, 'STORE_HT': 2, 'STORE_BYPASS': 3, 'STORE_LU': 4, 'STORE_RT_NT': 5, 'STORE_NT_HT': 6,
                'ATOMIC_RT': 0, 'ATOMIC_NT': 1, 'ATOMIC_RETURN': 1, 'ATOMIC_RT_RETURN': 1, 'ATOMIC_NT_RETURN': 3, 'ATOMIC_CASCADE_RT': 6, 'ATOMIC_CASCADE_NT': 6}.get(m.group(1), 0) if m else 0
      m, buf_text = _extract(buf_text, r'\s+scope:SCOPE_(\w+)')
      scope_val = {'CU': 0, 'SE': 1, 'DEV': 2, 'SYS': 3}.get(m.group(1), 0) if m else 0
      buf_ops = _parse_ops(buf_text)
      buf_args = [_op2dsl(o) for o in buf_ops]
      vbuf_mods = "".join([f", ioffset={off_val}" if off_val else "", ", offen=1" if offen else "", ", idxen=1" if idxen else "",
                          f", th={th_val}" if th_val else "", f", scope={scope_val}" if scope_val else "",
                          ", tfe=1" if tfe else ""])
      if is_tbuf and fmt_num is not None: vbuf_mods = f", format={fmt_num}" + vbuf_mods
      elif is_tbuf: vbuf_mods = ", format=1" + vbuf_mods
      else: vbuf_mods = ", format=1" + vbuf_mods
      vaddr_idx = 1
      if len(buf_ops) > vaddr_idx and buf_ops[vaddr_idx].strip().lower() == 'off': vaddr_val = "v[0]"
      else: vaddr_val = buf_args[vaddr_idx] if len(buf_args) > vaddr_idx else "v[0]"
      rsrc_idx, soff_idx = (2, 3) if len(buf_ops) > 1 else (1, 2)
      rsrc_raw = buf_ops[rsrc_idx].strip() if len(buf_ops) > rsrc_idx else "s[0:3]"
      if m := re.match(r's\[(\d+):\d+\]', rsrc_raw.lower()): rsrc_val = m.group(1)
      elif m := re.match(r's(\d+)', rsrc_raw.lower()): rsrc_val = m.group(1)
      elif m := re.match(r'ttmp\[(\d+):\d+\]', rsrc_raw.lower()): rsrc_val = str(108 + int(m.group(1)))
      elif m := re.match(r'ttmp(\d+)', rsrc_raw.lower()): rsrc_val = str(108 + int(m.group(1)))
      else: rsrc_val = "0"
      soff_raw = buf_ops[soff_idx].strip() if len(buf_ops) > soff_idx else "0"
      soff_lower = soff_raw.lower()
      if soff_lower == 'm0': soff_val = "RawImm(125)"
      elif soff_lower in ('null', 'off'): soff_val = "RawImm(124)"
      elif m := re.match(r's(\d+)', soff_lower): soff_val = f"RawImm({m.group(1)})"
      else: soff_val = f"RawImm({soff_raw})"
      return f"{mn}(vdata={buf_args[0]}, vaddr={vaddr_val}, rsrc={rsrc_val}, soffset={soff_val}{vbuf_mods})"
    buf_mods = "".join([f", offset={off_val}" if off_val else "", ", glc=1" if glc else "", ", dlc=1" if dlc else "",
                        ", slc=1" if slc else "", ", tfe=1" if tfe else "", ", offen=1" if offen else "", ", idxen=1" if idxen else ""])
    if is_tbuf and fmt_num is not None: buf_mods = f", format={fmt_num}" + buf_mods
    vaddr_idx = 1
    if len(ops) > vaddr_idx and ops[vaddr_idx].strip().lower() == 'off': vaddr_val = "v[0]"
    else: vaddr_val = args[vaddr_idx] if len(args) > vaddr_idx else "v[0]"
    srsrc_idx, soff_idx = (2, 3) if len(ops) > 1 else (1, 2)
    srsrc_val = args[srsrc_idx] if len(args) > srsrc_idx else "s[0:3]"
    soff_val = args[soff_idx] if len(args) > soff_idx else "0"
    return f"{mn}(vdata={args[0]}, vaddr={vaddr_val}, srsrc={srsrc_val}, soffset={soff_val}{buf_mods})"

  # FLAT/GLOBAL/SCRATCH load/store/atomic
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
    if ds_off0 is not None or ds_off1 is not None:
      off0, off1 = str(ds_off0 or 0), str(ds_off1 or 0)
    elif off_val:
      off0, off1 = str(int(off_val, 0) & 0xff), str((int(off_val, 0) >> 8) & 0xff)
    else:
      off0, off1 = "0", "0"
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
  mn_base = mn.replace('_e32', '').replace('_e64', '')
  if mn_base in ('v_fmaak_f32', 'v_fmaak_f16') and len(args) == 4: lit_s, args = f", literal={args[3].strip()}", args[:3]
  elif mn_base in ('v_fmamk_f32', 'v_fmamk_f16') and len(args) == 4: lit_s, args = f", literal={args[2].strip()}", [args[0], args[1], args[3]]
  elif mn_base in ('s_fmaak_f32',) and len(args) == 4: lit_s, args = f", literal={args[3].strip()}", args[:3]
  elif mn_base in ('s_fmamk_f32',) and len(args) == 4: lit_s, args = f", literal={args[2].strip()}", [args[0], args[1], args[3]]
  elif mn in ('v_cndmask_b32', 'v_cndmask_b32_e32') and len(args) == 4 and ops[3].strip().lower() in ('vcc_lo', 'vcc'):
    mn, args = 'v_cndmask_b32_e32', args[:3]

  _SGPR_NAMES = {'vcc_lo': 106, 'vcc_hi': 107, 'vcc': 106, 'null': 124, 'm0': 125, 'exec_lo': 126, 'exec_hi': 127}
  vcc_ops = {'v_add_co_ci_u32', 'v_sub_co_ci_u32', 'v_subrev_co_ci_u32'}
  if mn.replace('_e32', '') in vcc_ops and len(args) >= 5:
    carry_in = ops[4].strip().lower() if len(ops) > 4 else 'vcc_lo'
    carry_out = ops[1].strip().lower() if len(ops) > 1 else 'vcc_lo'
    if carry_in in ('vcc_lo', 'vcc') and carry_out in ('vcc_lo', 'vcc'):
      mn, args = mn.replace('_e32', '') + '_e32', [args[0], args[2], args[3]]
    else:
      mn_base = mn.replace('_e32', '').replace('_e64', '')
      sdst = _SGPR_NAMES.get(carry_out, 124) if carry_out in _SGPR_NAMES else (int(carry_out[1:]) if carry_out.startswith('s') and carry_out[1:].isdigit() else 124)
      src2 = _SGPR_NAMES.get(carry_in, 0) if carry_in in _SGPR_NAMES else (int(carry_in[1:]) if carry_in.startswith('s') and carry_in[1:].isdigit() else 0)
      return f"{mn_base}(vdst={args[0]}, sdst=RawImm({sdst}), src0={args[2]}, src1={args[3]}, src2=RawImm({src2}))"
  if mn.replace('_e64', '') in vcc_ops and mn.endswith('_e64'): mn = mn.replace('_e64', '')
  if mn.startswith('v_cmp') and not mn.endswith('_e64') and len(args) >= 3 and ops[0].strip().lower() in ('vcc_lo', 'vcc_hi', 'vcc'): args = args[1:]
  if 'cmpx' in mn and mn.endswith('_e64') and len(args) == 2: args = ['RawImm(126)'] + args
  if ((mn.startswith('v_cmp') and 'cmpx' not in mn and mn.endswith('_e64')) or mn.startswith('v_s_') or mn in ('v_readlane_b32', 'v_readfirstlane_b32')) and len(args) >= 1:
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
  vop3p_ops = {'v_pk_', 'v_dot2', 'v_dot4', 'v_dot8', 'v_wmma', 'v_swmmac'}
  is_vop3p = any(mn.startswith(p) for p in vop3p_ops)
  is_fma_mix = 'fma_mix' in mn
  if opsel_hi_val is not None:
    opsel_hi_enc = opsel_hi_val[0] | (opsel_hi_val[1] << 1) if len(opsel_hi_val) >= 2 else opsel_hi_val[0]
    opsel_hi2_enc = opsel_hi_val[2] if len(opsel_hi_val) >= 3 else (0 if is_fma_mix else 1)
    all_kw.extend([f'opsel_hi={opsel_hi_enc}', f'opsel_hi2={opsel_hi2_enc}'])
  elif is_vop3p and not is_fma_mix:
    all_kw.extend(['opsel_hi=3', 'opsel_hi2=1'])
  if clamp_found:
    if arch == 'rdna4': all_kw.append('cm=1')
    else: all_kw.append('clmp=1')

  a_str, kw_str = ', '.join(args), ', '.join(all_kw)
  return f"{fn}({a_str}, {kw_str})" if kw_str and a_str else f"{fn}({kw_str})" if kw_str else f"{fn}({a_str})"

def _hwreg(id_, offset=0, size=32): return id_ | (offset << 6) | ((size - 1) << 11)
def _sendmsg(id_, op=0, stream=0): return id_ | (op << 4) | (stream << 8)

_HWREG_NAMES = {'HW_REG_MODE': 1, 'HW_REG_STATUS': 2, 'HW_REG_TRAPSTS': 3, 'HW_REG_HW_ID': 4, 'HW_REG_GPR_ALLOC': 5,
  'HW_REG_LDS_ALLOC': 6, 'HW_REG_IB_STS': 7, 'HW_REG_PC_LO': 8, 'HW_REG_PC_HI': 9, 'HW_REG_INST_DW0': 10, 'HW_REG_INST_DW1': 11,
  'HW_REG_IB_DBG0': 12, 'HW_REG_IB_DBG1': 13, 'HW_REG_FLUSH_IB': 14, 'HW_REG_SH_MEM_BASES': 15, 'HW_REG_SQ_SHADER_TBA_LO': 16,
  'HW_REG_SQ_SHADER_TBA_HI': 17, 'HW_REG_SQ_SHADER_TMA_LO': 18, 'HW_REG_SQ_SHADER_TMA_HI': 19, 'HW_REG_FLAT_SCR_LO': 20,
  'HW_REG_FLAT_SCR_HI': 21, 'HW_REG_XNACK_MASK': 22, 'HW_REG_HW_ID1': 23, 'HW_REG_HW_ID2': 24, 'HW_REG_POPS_PACKER': 25,
  'HW_REG_PERF_SNAPSHOT_DATA': 26, 'HW_REG_PERF_SNAPSHOT_PC_LO': 27, 'HW_REG_PERF_SNAPSHOT_PC_HI': 28, 'HW_REG_SHADER_CYCLES': 29,
  'HW_REG_SHADER_CYCLES_HI': 30, 'HW_REG_WAVE_MODE': 31, 'HW_REG_WAVE_SCRATCH_BASE': 32}
_HWREG_NAMES_RDNA4 = {v: k for k, v in HWREG_RDNA4.items()}
_SENDMSG_NAMES = {'MSG_INTERRUPT': 1, 'MSG_GS': 2, 'MSG_GS_DONE': 3, 'MSG_SAVEWAVE': 4, 'MSG_STALL_WAVE_GEN': 5,
  'MSG_HALT_WAVES': 6, 'MSG_ORDERED_PS_DONE': 7, 'MSG_EARLY_PRIM_DEALLOC': 8, 'MSG_GS_ALLOC_REQ': 9, 'MSG_GET_DOORBELL': 10,
  'MSG_GET_DDID': 11, 'MSG_HS_TESSFACTOR': 2, 'MSG_DEALLOC_VGPRS': 10, 'MSG_RTN_GET_DOORBELL': 128, 'MSG_RTN_GET_DDID': 129,
  'MSG_RTN_GET_TMA': 130, 'MSG_RTN_GET_REALTIME': 131, 'MSG_RTN_SAVE_WAVE': 132, 'MSG_RTN_GET_TBA': 133,
  'MSG_RTN_GET_TBA_TO_PC': 134, 'MSG_RTN_GET_SE_AID_ID': 135}

def asm(text: str, arch: str = "rdna3") -> Inst:
  dsl = get_dsl(text, arch)
  if arch == "rdna4":
    ns = {n: getattr(rdna4_ins, n) for n in dir(rdna4_ins) if not n.startswith('_')}
    hwreg_names = _HWREG_NAMES_RDNA4
  else:
    ns = {n: getattr(ins, n) for n in dir(ins) if not n.startswith('_')}
    hwreg_names = _HWREG_NAMES
  def hwreg(id_, offset=0, size=32): return _hwreg(hwreg_names.get(id_, id_) if isinstance(id_, str) else id_, offset, size)
  def sendmsg(id_, op=0, stream=0): return _sendmsg(_SENDMSG_NAMES.get(id_, id_) if isinstance(id_, str) else id_, op, stream)
  ns.update({'s': s, 'v': v, 'ttmp': ttmp, 'abs': abs, 'RawImm': RawImm, 'SrcMod': SrcMod, 'VGPR': VGPR, 'SGPR': SGPR, 'TTMP': TTMP,
             'VCC_LO': VCC_LO, 'VCC_HI': VCC_HI, 'VCC': VCC, 'EXEC_LO': EXEC_LO, 'EXEC_HI': EXEC_HI, 'EXEC': EXEC, 'SCC': SCC, 'M0': M0, 'NULL': NULL, 'OFF': OFF,
             'hwreg': hwreg, 'sendmsg': sendmsg, **{k: k for k in hwreg_names}, **{k: k for k in _SENDMSG_NAMES}})
  try: return eval(dsl, ns)
  except NameError:
    if m := re.match(r'^(v_\w+)(\(.*\))$', dsl): return eval(f"{m.group(1)}_e32{m.group(2)}", ns)
    raise
