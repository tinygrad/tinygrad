# RDNA3 emulator v2 - compiles pcode to UOps executed via tinygrad CPU backend
# Each instruction is compiled to a kernel that operates on buffers:
#   arg=0: vmem - base address 0, INDEX offsets directly to host memory
#   arg=1: lds - local data share
#   arg=2: vgpr - vgpr[reg * 32 + lane]
#   arg=3: sgpr - sgpr[reg], PC_LO=128, PC_HI=129, SCC=130
from __future__ import annotations
import ctypes, functools, re, platform, subprocess, tempfile

# Set/restore DAZ+FTZ (denormals-are-zero + flush-to-zero) in MXCSR to match RDNA3 default float mode
# Only applied during emulator execution, restored afterward to avoid breaking hypothesis tests
@functools.cache
def _get_mxcsr_lib():
  if platform.machine() not in ('x86_64', 'AMD64'): return None
  try:
    src = b'''
unsigned int get_mxcsr(void){unsigned int m;__asm__ __volatile__("stmxcsr %0":"=m"(m));return m;}
void set_mxcsr(unsigned int m){__asm__ __volatile__("ldmxcsr %0"::"m"(m));}
'''
    with tempfile.NamedTemporaryFile(suffix='.so', delete=False) as f:
      subprocess.check_output(['clang', '-shared', '-O2', '-x', 'c', '-', '-o', f.name], input=src)
      lib = ctypes.CDLL(f.name)
      lib.get_mxcsr.restype = ctypes.c_uint32
      lib.set_mxcsr.argtypes = [ctypes.c_uint32]
      return lib
  except Exception: return None

class _MXCSRContext:
  """Context manager to set DAZ+FTZ during emulator execution and restore afterward."""
  __slots__ = ('_saved',)
  def __enter__(self):
    lib = _get_mxcsr_lib()
    if lib is None: return self
    self._saved = lib.get_mxcsr()
    lib.set_mxcsr(self._saved | 0x8040)  # DAZ (bit 6) + FTZ (bit 15)
    return self
  def __exit__(self, *args):
    lib = _get_mxcsr_lib()
    if lib is None or not hasattr(self, '_saved'): return
    lib.set_mxcsr(self._saved)
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.dtype import dtypes
from tinygrad.codegen import get_program
from tinygrad.device import Device, Buffer, BufferSpec
from tinygrad.runtime.autogen import hsa
from tinygrad.helpers import Context, DEBUG, colored, TUPLE_ORDER, getenv
from tinygrad.renderer import ProgramSpec

from extra.assembly.amd.decode import decode_inst
from extra.assembly.amd.autogen.rdna3.str_pcode import PCODE
from extra.assembly.amd.autogen.rdna3.ins import (SOP1, SOP2, SOPC, SOPK, SOPP, SMEM, VOP1, VOP1_SDST, VOP2, VOP3, VOP3_SDST, VOP3SD, VOP3P, VOPC,
  DS, FLAT, GLOBAL, SCRATCH, VOPD, SOPPOp, SMEMOp, VOP1Op, VOP2Op, VOP3Op, VOPDOp)
from extra.assembly.amd.dsl import NULL, VCC_LO, EXEC_LO
from extra.assembly.amd.autogen.common import OpType
from extra.assembly.amd.expr_parser import parse_block

MASK32 = 0xFFFFFFFF

# Common UOp constants (avoid repeated allocation)
def _c(val, dtype=dtypes.uint32): return UOp.const(dtype, val)
U32_0, U32_1, U32_16, U32_MASK = _c(0), _c(1), _c(16), _c(MASK32)
IDX_0 = _c(0, dtypes.index)

# Inline float constants (as bit patterns) for GPU instructions
F32_INLINE = {240: 0x3f000000, 241: 0xbf000000, 242: 0x3f800000, 243: 0xbf800000,  # 0.5, -0.5, 1.0, -1.0
              244: 0x40000000, 245: 0xc0000000, 246: 0x40800000, 247: 0xc0800000, 248: 0x3e22f983}  # 2.0, -2.0, 4.0, -4.0, 1/(2*pi)
F64_INLINE = {240: 0x3fe0000000000000, 241: 0xbfe0000000000000, 242: 0x3ff0000000000000, 243: 0xbff0000000000000,
              244: 0x4000000000000000, 245: 0xc000000000000000, 246: 0x4010000000000000, 247: 0xc010000000000000, 248: 0x3fc45f306dc9c883}
F16_INLINE = {240: 0x3800, 241: 0xb800, 242: 0x3c00, 243: 0xbc00, 244: 0x4000, 245: 0xc000, 246: 0x4400, 247: 0xc400, 248: 0x3118}

def _u64(lo: UOp, hi: UOp) -> UOp:
  """Combine two 32-bit UOps into a 64-bit UOp."""
  return lo.cast(dtypes.uint64) | (hi.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))

def _split64(val: UOp) -> tuple[UOp, UOp]:
  """Split a 64-bit value into (lo, hi) 32-bit values."""
  v64 = val.bitcast(dtypes.uint64) if val.dtype == dtypes.float64 else val.cast(dtypes.uint64) if val.dtype != dtypes.uint64 else val
  return v64.cast(dtypes.uint32), (v64 >> UOp.const(dtypes.uint64, 32)).cast(dtypes.uint32)

def _apply_src_mods(val: UOp, mod_bit: int, abs_bits: int, neg_bits: int, is_16bit: bool = False, is_64bit: bool = False) -> UOp:
  """Apply abs/neg modifiers to source value based on operation type."""
  if not (abs_bits & (1 << mod_bit)) and not (neg_bits & (1 << mod_bit)): return val
  if is_16bit:
    f16_val = val.cast(dtypes.uint16).bitcast(dtypes.half)
    if abs_bits & (1 << mod_bit): f16_val = (f16_val.bitcast(dtypes.uint16) & UOp.const(dtypes.uint16, 0x7FFF)).bitcast(dtypes.half)
    if neg_bits & (1 << mod_bit): f16_val = f16_val.neg()
    return f16_val.bitcast(dtypes.uint16).cast(dtypes.uint32)
  if is_64bit:
    if val.dtype == dtypes.uint64: val = val.bitcast(dtypes.float64)
    if abs_bits & (1 << mod_bit): val = (val.bitcast(dtypes.uint64) & UOp.const(dtypes.uint64, 0x7FFFFFFFFFFFFFFF)).bitcast(dtypes.float64)
    if neg_bits & (1 << mod_bit): val = val.neg()
    return val.bitcast(dtypes.uint64)
  if val.dtype == dtypes.uint32: val = val.bitcast(dtypes.float32)
  if abs_bits & (1 << mod_bit): val = (val.bitcast(dtypes.uint32) & UOp.const(dtypes.uint32, 0x7FFFFFFF)).bitcast(dtypes.float32)
  if neg_bits & (1 << mod_bit): val = val.neg()
  return val.bitcast(dtypes.uint32)

# Map VOPD ops to VOP2 ops for pcode lookup
VOPD_TO_VOP2 = {
  VOPDOp.V_DUAL_FMAC_F32: VOP2Op.V_FMAC_F32_E32, VOPDOp.V_DUAL_MUL_F32: VOP2Op.V_MUL_F32_E32,
  VOPDOp.V_DUAL_ADD_F32: VOP2Op.V_ADD_F32_E32, VOPDOp.V_DUAL_SUB_F32: VOP2Op.V_SUB_F32_E32,
  VOPDOp.V_DUAL_SUBREV_F32: VOP2Op.V_SUBREV_F32_E32, VOPDOp.V_DUAL_MAX_F32: VOP2Op.V_MAX_F32_E32,
  VOPDOp.V_DUAL_MIN_F32: VOP2Op.V_MIN_F32_E32, VOPDOp.V_DUAL_ADD_NC_U32: VOP2Op.V_ADD_NC_U32_E32,
  VOPDOp.V_DUAL_LSHLREV_B32: VOP2Op.V_LSHLREV_B32_E32, VOPDOp.V_DUAL_AND_B32: VOP2Op.V_AND_B32_E32,
  VOPDOp.V_DUAL_MOV_B32: VOP1Op.V_MOV_B32_E32, VOPDOp.V_DUAL_CNDMASK_B32: VOP2Op.V_CNDMASK_B32_E32,
  VOPDOp.V_DUAL_FMAAK_F32: VOP2Op.V_FMAAK_F32_E32, VOPDOp.V_DUAL_FMAMK_F32: VOP2Op.V_FMAMK_F32_E32,
}
WAVE_SIZE = 32
PC_LO_IDX, SCC_IDX, SCRATCH_STRIDE_IDX = 128, 130, 131
SGPR_COUNT, VGPR_SIZE = 132, 256 * 32

def _is_16bit_op(op_name: str) -> bool: return any(x in op_name for x in ('B16', 'F16', 'I16', 'U16'))
def _op_name(inst) -> str:
  if hasattr(inst, 'opx'): return f"{inst.opx.name}_{inst.opy.name}"  # VOPD has opx/opy not op
  return inst.op.name if hasattr(inst.op, 'name') else str(inst.op)
def _is_64bit_dest(dest: str) -> bool: return any(dest.endswith(x) for x in ('.b64', '.u64', '.i64', '.f64'))
def _to_u32(val: UOp) -> UOp:
  if val.dtype == dtypes.uint32: return val
  if val.dtype.itemsize == 4: return val.bitcast(dtypes.uint32)  # same size: bitcast (float32->uint32)
  return val.cast(dtypes.uint32)  # different size: cast (bool, int16, etc)
def _lane_active(exec_mask: UOp, lane: UOp) -> UOp: return ((exec_mask >> lane.cast(dtypes.uint32)) & U32_1).ne(U32_0)
def _apply_opsel(val: UOp, sel_bit: int, opsel: int) -> UOp:
  return (val >> U32_16) & _c(0xFFFF) if opsel & (1 << sel_bit) else val

def _unroll_lanes(get_lane_bit, exec_mask: UOp, apply_exec: bool = True) -> UOp:
  """Combine 32 lane bits into a 32-bit mask using RANGE+REDUCE. Optionally apply EXEC mask."""
  lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
  bit = get_lane_bit(lane).cast(dtypes.uint32) << lane.cast(dtypes.uint32)
  result = bit.reduce(lane, arg=Ops.ADD)
  return result & exec_mask if apply_exec else result

def _set_lane_bit(old: UOp, lane: UOp, val: UOp, exec_mask: UOp) -> UOp:
  """Set/clear a single bit in a 32-bit mask based on lane index, respecting exec mask."""
  mask = U32_1 << lane.cast(dtypes.uint32)
  new_bit = _to_u32(val) << lane.cast(dtypes.uint32)
  cleared = old & (mask ^ U32_MASK)
  return _lane_active(exec_mask, lane).where(cleared | new_bit, old)

def _val_to_u32(val: UOp) -> UOp:
  """Convert any value to uint32 for storage (bitcast floats, cast ints)."""
  if val.dtype == dtypes.uint32: return val
  if val.dtype == dtypes.float32: return val.bitcast(dtypes.uint32)
  if val.dtype == dtypes.half: return val.bitcast(dtypes.uint16).cast(dtypes.uint32)
  if val.dtype in (dtypes.uint16, dtypes.int16): return val.cast(dtypes.uint32)
  return val.cast(dtypes.uint32)

# Pcode parser
def _apply_pseudocode_fixes(op_name: str, pcode: str) -> str:
  fixes = {
    'V_DIV_FMAS_F32': ('D0.f32 = 2.0F ** 32 * fma(S0.f32, S1.f32, S2.f32)',
      'D0.f32 = (exponent(S2.f32) > 127) ? (2.0F ** 64 * fma(S0.f32, S1.f32, S2.f32)) : (2.0F ** -64 * fma(S0.f32, S1.f32, S2.f32))'),
    'V_DIV_FMAS_F64': ('D0.f64 = 2.0 ** 64 * fma(S0.f64, S1.f64, S2.f64)',
      'D0.f64 = (exponent(S2.f64) > 1023) ? (2.0 ** 128 * fma(S0.f64, S1.f64, S2.f64)) : (2.0 ** -128 * fma(S0.f64, S1.f64, S2.f64))'),
    'V_DIV_FIXUP_F32': ('D0.f32 = sign_out ? -abs(S0.f32) : abs(S0.f32)',
      'D0.f32 = isNAN(S0.f32) ? (sign_out ? -INF.f32 : +INF.f32) : (sign_out ? -abs(S0.f32) : abs(S0.f32))'),
    'V_DIV_FIXUP_F64': ('D0.f64 = sign_out ? -abs(S0.f64) : abs(S0.f64)',
      'D0.f64 = isNAN(S0.f64) ? (sign_out ? -INF : +INF) : (sign_out ? -abs(S0.f64) : abs(S0.f64))'),
    'V_TRIG_PREOP_F64': ("result = 64'F((1201'B(2.0 / PI)[1200 : 0] << shift.u32) & 1201'0x1fffffffffffff)", "result = trig_preop_result(shift)"),
  }
  if op_name in fixes: pcode = pcode.replace(fixes[op_name][0], fixes[op_name][1])
  if 'V_DIV_SCALE' in op_name:
    dt, exp_lim, ldexp_val = ('f32', '23', '64') if 'F32' in op_name else ('f64', '52', '128')
    for old, new in [(f'S2.{dt} / S1.{dt} == DENORM.{dt}', f'divWouldBeDenorm(S2.{dt}, S1.{dt})'), (f"1.0 / 64'F(S1.{dt}) == DENORM.f64", '0'),
                     (f'1.0 / S1.{dt} == DENORM.{dt}', '0'), (f'S1.{dt} == DENORM.{dt}', f'isDENORM(S1.{dt})'),
                     (f'D0.{dt} = NAN.{dt}', f'VCC = 0x1LL;\nD0.{dt} = NAN.{dt}'),
                     (f'elsif isDENORM(S1.{dt}) then\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})', f'elsif 1 == 0 then\nD0.{dt} = S0.{dt}'),
                     (f'elsif exponent(S2.{dt}) <= {exp_lim} then\n// Numerator is tiny\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})',
                      f'elsif exponent(S2.{dt}) <= {exp_lim} then\nVCC = 0x1LL;\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})'),
                     (f'elsif divWouldBeDenorm(S2.{dt}, S1.{dt}) then\nVCC = 0x1LL;\nif S0.{dt} == S2.{dt} then\n// Only scale the numerator\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nendif',
                      f'elsif divWouldBeDenorm(S2.{dt}, S1.{dt}) then\nVCC = 0x1LL;\nD0.{dt} = S0.{dt}'),
                     (f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nendif\nelsif', f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nelse\nD0.{dt} = S0.{dt}\nendif\nelsif')]:
      pcode = pcode.replace(old, new)
    lines = pcode.rstrip().split('\n')
    for i in range(len(lines) - 1, -1, -1):
      if lines[i].strip() == 'endif': lines.insert(i, f'else\nD0.{dt} = S0.{dt}'); break
    pcode = '\n'.join(lines) + f';\nif isDENORM(S1.{dt}) then\nD0.{dt} = NAN.{dt}\nendif'
    pcode = pcode.replace('VCC = 0x0LL', 'VCC.u64[laneId] = 0').replace('VCC = 0x1LL', 'VCC.u64[laneId] = 1')
  return pcode

def parse_pcode(pcode: str, srcs: dict[str, UOp] | None = None, lane: UOp | None = None, op_name: str | None = None) -> tuple[dict[str, UOp], list[tuple[str, UOp]]]:
  if op_name: pcode = _apply_pseudocode_fixes(op_name, pcode)
  vars: dict[str, UOp] = {n: UOp(Ops.DEFINE_VAR, dtypes.uint32, (), (n, U32_0, U32_MASK)) for n in ['S0', 'S1', 'S2', 'D0', 'VCC', 'EXEC', 'SCC', 'SIMM32']}
  if srcs: vars.update(srcs)
  vars.update({'laneId': lane if lane is not None else U32_0, 'WAVE_MODE': {'IEEE': U32_1}, 'WAVE32': _c(True, dtypes.bool), 'WAVE64': _c(False, dtypes.bool)})
  assigns: list[tuple[str, UOp]] = []
  lines = [l.strip().rstrip(';') for l in pcode.split('\n') if l.strip() and not l.strip().startswith('//')]
  _, final, _ = parse_block(lines, 0, vars, assigns=assigns)
  sliced = set(d.split('[')[0] for d, _ in assigns if '[' in d)
  for var, val in final.items():
    if var in ['D0', 'SCC', 'VCC', 'EXEC', 'PC', 'RETURN_DATA', 'VDATA']:
      if var in sliced and not any(re.match(rf'{var}\.\w+\s*=', l) for l in lines): continue
      for l in lines:
        if (m := re.match(rf'{var}\.(\w+(?:\[\w+\])?)', l)): assigns.append((f'{var}.{m.group(1)}', val)); break
      else: assigns.append((var, val))
  return vars, assigns

def _write_64bit(val: UOp, wfn, reg_or_addr, *args) -> list[UOp]:
  """Write a 64-bit value as two 32-bit writes. args passed to wfn after reg/addr and lo/hi value."""
  lo, hi = _split64(val)
  incr = 4 if isinstance(reg_or_addr, UOp) else 1  # 4 bytes for memory addresses, 1 for register indices
  return [wfn(reg_or_addr, lo, *args), wfn(reg_or_addr + (UOp.const(reg_or_addr.dtype, incr) if isinstance(reg_or_addr, UOp) else incr), hi, *args)]

def _write_val(dest: str, val: UOp, wfn, reg_or_addr, *args) -> list[UOp]:
  """Write value, splitting 64-bit if needed based on dest type suffix."""
  return _write_64bit(val, wfn, reg_or_addr, *args) if _is_64bit_dest(dest) else [wfn(reg_or_addr, _to_u32(val), *args)]

def _mem_store(mem: UOp, addr: UOp, val: UOp, active: UOp, addr_bits: int = 32, data_bits: int = 32) -> list[UOp]:
  """Conditional memory store: write val to mem[addr] if active, else keep old value. Handles sub-word stores. Returns list of store UOps."""
  adt = dtypes.uint64 if addr_bits == 64 else dtypes.uint32
  shift = UOp.const(adt, 2)
  word_addr = addr >> shift
  # Use .valid(active) to skip load from garbage address when lane is inactive
  idx = mem.index(word_addr.cast(dtypes.index).valid(active))
  # NOTE: Don't call idx.load() - use idx directly as the value. pm_add_loads will add the load op later.
  # Calling .load() here causes LOAD(LOAD) after pm_add_loads runs.
  val_u32 = val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val
  if data_bits == 8:
    byte_pos = (addr.cast(dtypes.uint32) & UOp.const(dtypes.uint32, 3))  # 0-3
    byte_shift = byte_pos << UOp.const(dtypes.uint32, 3)  # *8
    mask = UOp.const(dtypes.uint32, 0xFF) << byte_shift
    new_word = (idx & (mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))) | ((val_u32 & UOp.const(dtypes.uint32, 0xFF)) << byte_shift)
    return [idx.store(active.where(new_word, idx))]
  elif data_bits == 16:
    # 16-bit stores. byte_pos (0-3) determines placement within 4-byte word.
    # byte_pos 0,1,2: both bytes fit in current word
    # byte_pos 3: crosses word boundary - low byte to byte 3, high byte to next word's byte 0
    byte_pos = addr.cast(dtypes.uint32) & UOp.const(dtypes.uint32, 3)
    byte_shift = byte_pos << UOp.const(dtypes.uint32, 3)  # *8
    low_byte = val_u32 & UOp.const(dtypes.uint32, 0xFF)
    high_byte = (val_u32 >> UOp.const(dtypes.uint32, 8)) & UOp.const(dtypes.uint32, 0xFF)
    # Same-word value (for byte_pos 0,1,2): write 16 bits at byte_pos
    mask_16 = UOp.const(dtypes.uint32, 0xFFFF) << byte_shift
    same_word = (idx & (mask_16 ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))) | ((val_u32 & UOp.const(dtypes.uint32, 0xFFFF)) << byte_shift)
    # Cross-word value for current word (byte_pos=3): write low byte to byte 3
    cross_word0 = (idx & UOp.const(dtypes.uint32, 0x00FFFFFF)) | (low_byte << UOp.const(dtypes.uint32, 24))
    # Detect cross-word case: byte_pos == 3 <=> (byte_pos & 2) && (byte_pos & 1)
    is_cross = ((byte_pos >> UOp.const(dtypes.uint32, 1)) & byte_pos & UOp.const(dtypes.uint32, 1)).cast(dtypes.bool)
    # Select value for current word
    new_word = is_cross.where(cross_word0, same_word)
    store0 = idx.store(active.where(new_word, idx))
    # Next word store for cross-word case: write high byte to byte 0 of next word
    active_cross = active & is_cross
    # Use .valid(active_cross) to skip load from garbage address when lane is inactive or not cross-word
    next_word_addr = (word_addr + UOp.const(adt, 1)).cast(dtypes.index).valid(active_cross)
    next_idx = mem.index(next_word_addr)
    cross_word1 = (next_idx & UOp.const(dtypes.uint32, 0xFFFFFF00)) | high_byte
    store1 = next_idx.store(active_cross.where(cross_word1, next_idx))
    return [store0, store1]
  else:
    new_word = _to_u32(val)
    return [idx.store(active.where(new_word, idx))]

def _mem_store_bytes(mem: UOp, addr: UOp, val: UOp, active: UOp, data_bits: int = 32) -> list[UOp]:
  """Store to byte-addressable memory (scratch). addr is byte offset, mem is uint8 buffer."""
  stores = []
  val_u32 = val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val
  for i in range(data_bits // 8):
    byte_val = (val_u32 >> UOp.const(dtypes.uint32, i * 8)) & UOp.const(dtypes.uint32, 0xFF)
    idx = (addr + UOp.const(dtypes.uint64, i)).cast(dtypes.index).valid(active)
    stores.append(mem.index(idx).store(byte_val.cast(dtypes.uint8)))
  return stores

def _collect_data_slices(assigns: list, data_prefix: str, pcode_vars: dict = None, op_name: str = "") -> dict[int, UOp]:
  """Collect bit slices from assigns into {dword_idx: value} dict."""
  slices = {}
  for dest, val in assigns:
    if dest.startswith(f'{data_prefix}['):
      if (m := re.match(rf'{data_prefix}\[(\d+)\s*:\s*(\d+)\]', dest)):
        hi_bit, low_bit = int(m.group(1)), int(m.group(2))
        dword_idx = low_bit // 32
        # D16 loads preserve bits - use final value from pcode_vars which has hi bits preserved
        if pcode_vars and 'D16' in op_name and dword_idx == 0 and hi_bit < 32:
          slices[0] = _to_u32(pcode_vars.get(data_prefix, val))
        else: slices[dword_idx] = _to_u32(val)
    elif dest.startswith(data_prefix): slices[0] = _to_u32(val)
  return slices

def _scalar_stores(assigns: list, wsgpr, sdst_reg: int, sdst_size: int = 1) -> list[UOp]:
  """Generate stores for scalar assigns (D0, SCC, EXEC, VCC)."""
  def w64(reg, val):
    if val.dtype in (dtypes.uint64, dtypes.int64):
      lo, hi = _split64(val)
      return [wsgpr(reg, lo), wsgpr(reg + 1, hi)]
    return [wsgpr(reg, _to_u32(val))]
  stores = []
  for dest, val in assigns:
    if dest.startswith('D0'): stores.extend(w64(sdst_reg, val) if sdst_size == 2 else [wsgpr(sdst_reg, _to_u32(val))])
    elif dest.startswith('SCC'): stores.append(wsgpr(SCC_IDX, _to_u32(val)))
    elif dest.startswith('EXEC'): stores.extend(w64(EXEC_LO.offset, val))
    elif dest.startswith('VCC'): stores.extend(w64(VCC_LO.offset, val))
  return stores

# Counter for unique axis IDs to avoid UOp caching issues
_axis_id_counter = 0
def _next_axis_id() -> int:
  global _axis_id_counter
  _axis_id_counter += 1
  return _axis_id_counter

def compile_sop_pcode(op, srcs: dict[str, UOp], wsgpr_fn, rsgpr_fn, sdst_reg: int, sdst_size: int, inc_pc_fn, name: str):
  """Compile a scalar instruction using pcode parser. Returns (name, sink) or None."""
  pcode = PCODE.get(op)
  if pcode is None: return None
  srcs.update({'VCC': rsgpr_fn(VCC_LO.offset), 'EXEC': rsgpr_fn(EXEC_LO.offset), 'SCC': rsgpr_fn(SCC_IDX), 'D0': rsgpr_fn(sdst_reg)})
  _, assigns = parse_pcode(pcode, srcs, lane=None)
  stores = _scalar_stores(assigns, wsgpr_fn, sdst_reg, sdst_size)
  if not stores: return None
  return name, UOp.sink(*stores, inc_pc_fn(), arg=KernelInfo(name=name))

def compile_lane_pcode(op, inst, vgpr, wsgpr_fn, rsgpr_fn, rsrc_fn, inc_pc_fn, name: str):
  """Compile READLANE/READFIRSTLANE/WRITELANE using pcode parser."""
  pcode = PCODE.get(op)
  if pcode is None: return None

  op_name = op.name if hasattr(op, 'name') else str(op)
  src0_reg = (inst.src0.offset - 256) if inst.src0.offset >= 256 else 0
  vdst_reg = (inst.vdst.offset - 256) if inst.vdst.offset >= 256 else inst.vdst.offset
  # S0 = scalar value for WRITELANE, register index for others; S1 = lane select for READLANE/WRITELANE
  srcs = {
    'SRC0': _c(src0_reg), 'VDST': _c(vdst_reg), 'EXEC_LO': rsgpr_fn(EXEC_LO.offset), '_vgpr': vgpr,
    'S0': rsrc_fn(inst.src0.offset, IDX_0) if 'WRITELANE' in op_name else _c(src0_reg),
    'S1': rsrc_fn(inst.src1.offset, IDX_0) if hasattr(inst, 'src1') and inst.src1 is not None else U32_0,
  }
  _, assigns = parse_pcode(pcode, srcs, lane=None)

  stores = []
  for dest, val in assigns:
    if dest.startswith('D0'):
      stores.append(wsgpr_fn(inst.vdst.offset, val.cast(dtypes.uint32)))
    elif dest.startswith('VGPR['):
      idx, write_val = val
      stores.append(vgpr.index(idx.cast(dtypes.index)).store(write_val.cast(dtypes.uint32)))

  if not stores: return None
  return name, UOp.sink(*stores, inc_pc_fn(), arg=KernelInfo(name=name))

def compile_vop_pcode(op, srcs: dict[str, UOp], lane: UOp, wvgpr_fn, wsgpr_fn, rsgpr_fn, vdst_reg: int, exec_mask: UOp,
                      inc_pc_fn=None, name: str = None, opsel_dst_hi: bool = False, rvgpr_fn=None, sdst_reg: int | None = None):
  """Compile a VOP instruction using pcode parser. Returns (name, sink) if inc_pc_fn/name provided, else list of store UOps, or None."""
  pcode = PCODE.get(op)
  if pcode is None: return None
  vcc_reg = sdst_reg if sdst_reg is not None else VCC_LO.offset
  if 'VCC' not in srcs: srcs['VCC'] = rsgpr_fn(vcc_reg)
  srcs['EXEC'], srcs['SCC'] = exec_mask, rsgpr_fn(SCC_IDX)
  _, assigns = parse_pcode(pcode, srcs, lane, op_name=op.name)

  raw_stores, vcc_val, exec_val = [], None, None
  for dest, val in assigns:
    if 'D0' in dest and '[laneId]' in dest:
      raw_stores.append(('vcc', wsgpr_fn(VCC_LO.offset, _set_lane_bit(rsgpr_fn(VCC_LO.offset), lane, val, exec_mask))))
    elif dest.startswith('D0'):
      if (slice_match := re.match(r'D0\[(\d+)\s*:\s*(\d+)\]', dest)):
        hi_bit, lo_bit = int(slice_match.group(1)), int(slice_match.group(2))
        if hi_bit != 31 or lo_bit != 0:
          width, slice_mask = hi_bit - lo_bit + 1, (1 << (hi_bit - lo_bit + 1)) - 1
          val_bits = val.bitcast(dtypes.uint16).cast(dtypes.uint32) if val.dtype == dtypes.half else \
                     val.cast(dtypes.uint32) if val.dtype in (dtypes.uint16, dtypes.int16) else val.cast(dtypes.uint32) & UOp.const(dtypes.uint32, slice_mask)
          raw_stores.append(('vgpr_slice', (lo_bit, width, val_bits)))
          continue
      if val.dtype in (dtypes.uint64, dtypes.int64, dtypes.float64):
        lo, hi = _split64(val)
        raw_stores.extend([('vgpr', wvgpr_fn(vdst_reg, lane, lo, exec_mask)), ('vgpr', wvgpr_fn(vdst_reg + 1, lane, hi, exec_mask))])
      elif val.dtype in (dtypes.half, dtypes.uint16, dtypes.int16) and rvgpr_fn is not None:
        result, old_val = _val_to_u32(val), rvgpr_fn(vdst_reg, lane)
        result = (old_val & UOp.const(dtypes.uint32, 0xFFFF)) | (result << UOp.const(dtypes.uint32, 16)) if opsel_dst_hi else \
                 (old_val & UOp.const(dtypes.uint32, 0xFFFF0000)) | (result & UOp.const(dtypes.uint32, 0xFFFF))
        raw_stores.append(('vgpr', wvgpr_fn(vdst_reg, lane, result, exec_mask)))
      else: raw_stores.append(('vgpr', wvgpr_fn(vdst_reg, lane, _val_to_u32(val), exec_mask)))
    elif dest.startswith('VCC'): vcc_val = val  # Collect VCC value to reduce across lanes
    elif dest.startswith('EXEC'): exec_val = val  # Collect EXEC value to reduce across lanes
    elif dest.startswith('SCC'): raw_stores.append(('scc', wsgpr_fn(SCC_IDX, _to_u32(val))))

  stores, lane_stores, scalar_stores = [], [s for t, s in raw_stores if t == 'vgpr'], [s for t, s in raw_stores if t == 'scc']
  slice_stores = [s for t, s in raw_stores if t == 'vgpr_slice']
  if slice_stores:
    result = rvgpr_fn(vdst_reg, lane) if rvgpr_fn else UOp.const(dtypes.uint32, 0)
    for lo_bit, width, val_bits in slice_stores:
      mask = UOp.const(dtypes.uint32, ((1 << width) - 1) << lo_bit)
      result = (result & (mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))) | (val_bits << UOp.const(dtypes.uint32, lo_bit))
    lane_stores.append(wvgpr_fn(vdst_reg, lane, result, exec_mask))
  if lane_stores: stores.append(UOp.sink(*lane_stores).end(lane))
  # VCC/EXEC writes use reduce to combine all lane bits, then write once (fixes multi-lane carry bug)
  # Must use _unroll_lanes pattern with fresh lambda to avoid graph issues with the main lane range
  # VOP2 carry instructions write ALL 32 VCC bits (hardware verified), not just active lane bits
  if vcc_val is not None:
    def get_vcc_bit(l): return (_to_u32(vcc_val.substitute({lane: l})) & U32_1).cast(dtypes.uint32)
    stores.append(wsgpr_fn(vcc_reg, _unroll_lanes(get_vcc_bit, exec_mask, apply_exec=False)))
  if exec_val is not None:
    def get_exec_bit(l): return (_to_u32(exec_val.substitute({lane: l})) & U32_1).cast(dtypes.uint32)
    stores.append(wsgpr_fn(EXEC_LO.offset, _unroll_lanes(get_exec_bit, exec_mask, apply_exec=False)))
  stores.extend(scalar_stores)
  if not stores: return None
  return (name, UOp.sink(*stores, inc_pc_fn(), arg=KernelInfo(name=name))) if inc_pc_fn else stores

# Buffers: sgpr=0, vgpr=1, vmem=2, lds=3, scratch=4

def _define_bufs():
  sgpr = UOp(Ops.DEFINE_GLOBAL, dtypes.uint32.ptr(SGPR_COUNT), arg=0)
  vgpr = UOp(Ops.DEFINE_GLOBAL, dtypes.uint32.ptr(VGPR_SIZE), arg=1)
  vmem = UOp(Ops.DEFINE_GLOBAL, dtypes.uint32.ptr(1 << 46), arg=2)
  lds = UOp(Ops.DEFINE_GLOBAL, dtypes.uint32.ptr(16384), arg=3)
  scratch = UOp(Ops.DEFINE_GLOBAL, dtypes.uint8.ptr(1 << 30), arg=4)
  return sgpr, vgpr, vmem, lds, scratch

def _sext(v, bits): return v - (1 << bits) if v & (1 << (bits - 1)) else v

# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION COMPILER - converts decoded instruction to UOp SINK
# ═══════════════════════════════════════════════════════════════════════════════

class _Ctx:
  """Context for instruction compilation - holds buffers and helpers."""
  __slots__ = ('sgpr', 'vgpr', 'vmem', 'lds', 'scratch', 'literal', 'inst_words')

  def __init__(self, sgpr, vgpr, vmem, lds, scratch, literal, inst_words):
    self.sgpr, self.vgpr, self.vmem, self.lds, self.scratch = sgpr, vgpr, vmem, lds, scratch
    self.literal, self.inst_words = literal, inst_words

  def rsgpr(self, reg: int) -> UOp:
    if reg == 124: return UOp.const(dtypes.uint32, 0)  # NULL register reads as 0
    return self.sgpr.index(UOp.const(dtypes.index, reg), ptr=True).load()
  def rsgpr64(self, off: int) -> UOp:
    if off >= 128:  # inline constant
      if off < 193: return UOp.const(dtypes.uint64, off - 128)  # 0-64
      if off < 209: return UOp.const(dtypes.int64, -(off - 192)).cast(dtypes.uint64)  # -1 to -16
      if off == 255: return UOp.const(dtypes.uint64, self.literal)  # literal constant
      return UOp.const(dtypes.uint64, 0)  # other inline constants
    return _u64(self.rsgpr(off), self.rsgpr(off + 1))
  def wsgpr(self, reg: int, val: UOp) -> UOp:
    if reg == 124: return UOp(Ops.GROUP)  # NULL register - discard write
    return self.sgpr.index(UOp.const(dtypes.index, reg)).store(val.cast(dtypes.uint32))

  def rvgpr(self, reg: int, lane: UOp) -> UOp: return self.vgpr.index(UOp.const(dtypes.index, reg * 32) + lane.cast(dtypes.index), ptr=True).load()
  def wvgpr(self, reg: int, lane: UOp, val: UOp, exec_mask: UOp, after: UOp|None = None) -> UOp:
    buf = self.vgpr.after(after) if after is not None else self.vgpr
    offset = (UOp.const(dtypes.index, reg * 32) + lane.cast(dtypes.index)).valid(_lane_active(exec_mask, lane))
    return buf.index(offset).store(val.cast(dtypes.uint32))

  def rsrc(self, off: int, lane: UOp, bits: int = 32) -> UOp:
    """Read source operand (32-bit, 64-bit with F64 inline, or 16-bit with F16 inline)."""
    if bits == 64:
      if off in F64_INLINE: return UOp.const(dtypes.uint64, F64_INLINE[off])
      if 128 <= off < 256:
        if off < 193: return UOp.const(dtypes.uint64, off - 128)
        if off < 209: return UOp.const(dtypes.int64, -(off - 192)).cast(dtypes.uint64)
        if off == 255: return UOp.const(dtypes.uint64, self.literal) << UOp.const(dtypes.uint64, 32)  # literal is high 32 bits
      if off < 128: return _u64(self.rsgpr(off), self.rsgpr(off + 1))
      return _u64(self.rvgpr(off - 256, lane), self.rvgpr(off - 255, lane))
    if bits == 16 and off in F16_INLINE: return UOp.const(dtypes.uint32, F16_INLINE[off])
    if off < 128: return self.rsgpr(off)
    if off == 253: return self.rsgpr(SCC_IDX)
    if off == 255: return UOp.const(dtypes.uint32, self.literal)
    if off < 255:  # inline constants
      if off < 193: return UOp.const(dtypes.uint32, off - 128)  # 0-64
      if off < 209: return UOp.const(dtypes.uint32, (-(off - 192)) & MASK32)  # -1 to -16
      if off in F32_INLINE: return UOp.const(dtypes.uint32, F32_INLINE[off])
      return UOp.const(dtypes.uint32, 0)  # other inline
    return self.rvgpr(off - 256, lane)

  def rsrc_sized(self, off: int, lane: UOp, sizes: dict, key: str, f16: bool = False) -> UOp:
    """Read source with size from operand metadata."""
    return self.rsrc(off, lane, 64) if sizes.get(key, 1) == 2 else self.rsrc(off, lane, 16 if f16 else 32)

  def inc_pc(self) -> UOp: return self.wsgpr(PC_LO_IDX, self.rsgpr(PC_LO_IDX) + UOp.const(dtypes.uint32, self.inst_words))

# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

def _compile_sopp(inst: SOPP, ctx: _Ctx, name: str) -> tuple[str, UOp]:
  if inst.op == SOPPOp.S_ENDPGM:
    return name, UOp.sink(ctx.wsgpr(PC_LO_IDX, UOp.const(dtypes.uint32, 0xFFFFFFFF)), arg=KernelInfo(name=name))
  pcode = PCODE.get(inst.op)
  if pcode is not None:
    pc_words = ctx.rsgpr(PC_LO_IDX)
    pc_bytes = pc_words.cast(dtypes.int64) * UOp.const(dtypes.int64, 4)
    vcc, exec_lo = ctx.rsgpr(VCC_LO.offset), ctx.rsgpr(EXEC_LO.offset)
    srcs = {'PC': pc_bytes, 'SIMM16': UOp.const(dtypes.int16, _sext(inst.simm16, 16)), 'SCC': ctx.rsgpr(SCC_IDX), 'VCC': vcc,
            'VCCZ': vcc.eq(UOp.const(dtypes.uint32, 0)).cast(dtypes.uint32), 'EXECZ': exec_lo.eq(UOp.const(dtypes.uint32, 0)).cast(dtypes.uint32)}
    for dest, val in parse_pcode(pcode, srcs, op_name=inst.op.name)[1]:
      if dest == 'PC' or dest.startswith('PC.'):
        return name, UOp.sink(ctx.wsgpr(PC_LO_IDX, (val >> UOp.const(dtypes.int64, 2)).cast(dtypes.uint32)), arg=KernelInfo(name=name))
  return name, UOp.sink(ctx.inc_pc(), arg=KernelInfo(name=name))

def _compile_smem(inst: SMEM, ctx: _Ctx, name: str) -> tuple[str, UOp]:
  addr = ctx.rsgpr64(inst.sbase.offset) + UOp.const(dtypes.uint64, _sext(inst.offset, 21))
  sdata_reg = inst.sdata.offset
  ndwords = {SMEMOp.S_LOAD_B32: 1, SMEMOp.S_LOAD_B64: 2, SMEMOp.S_LOAD_B128: 4, SMEMOp.S_LOAD_B256: 8, SMEMOp.S_LOAD_B512: 16}.get(inst.op, 1)
  stores = [ctx.wsgpr(sdata_reg + i, ctx.vmem.index((addr + UOp.const(dtypes.uint64, i * 4) >> UOp.const(dtypes.uint64, 2)).cast(dtypes.index)))
            for i in range(ndwords)]
  return name, UOp.sink(*stores, ctx.inc_pc(), arg=KernelInfo(name=name))

def _compile_sop(inst, ctx: _Ctx, name: str) -> tuple[str, UOp]:
  sizes = getattr(inst, 'op_regs', {})
  def rsrc_sz(off, key): return ctx.rsgpr64(off) if sizes.get(key, 1) == 2 else ctx.rsrc(off, IDX_0)
  if isinstance(inst, SOPK):
    simm16_sext = inst.simm16 if inst.simm16 < 0x8000 else inst.simm16 - 0x10000
    srcs = {'S0': ctx.rsgpr(inst.sdst.offset), 'SIMM16': UOp.const(dtypes.int32, simm16_sext), 'D0': ctx.rsgpr(inst.sdst.offset)}
    dst_reg, dst_size = inst.sdst.offset, 1
  else:
    srcs = {'S0': rsrc_sz(inst.ssrc0.offset, 'ssrc0')}
    if hasattr(inst, 'ssrc1'): srcs['S1'] = rsrc_sz(inst.ssrc1.offset, 'ssrc1')
    # SOP2 instructions like s_fmamk_f32 use SIMM32 literal
    if ctx.literal: srcs['SIMM32'] = UOp.const(dtypes.uint32, ctx.literal)
    dst_reg, dst_size = (0, 0) if isinstance(inst, SOPC) else (inst.sdst.offset, sizes.get('sdst', 1))
  pcode_result = compile_sop_pcode(inst.op, srcs, ctx.wsgpr, ctx.rsgpr, dst_reg, dst_size, ctx.inc_pc, name)
  assert pcode_result is not None, f"no pcode for {type(inst).__name__}: {inst.op.name}"
  return pcode_result

def _compile_vop12(inst, ctx: _Ctx, name: str) -> tuple[str, UOp]:
  op_name = _op_name(inst)
  if op_name == 'V_READFIRSTLANE_B32_E32':
    pcode_result = compile_lane_pcode(inst.op, inst, ctx.vgpr, ctx.wsgpr, ctx.rsgpr, ctx.rsrc, ctx.inc_pc, name)
    assert pcode_result is not None, f"no pcode for VOP1: {op_name}"
    return pcode_result
  lane, exec_mask, sizes = UOp.range(32, _next_axis_id(), AxisType.LOOP), ctx.rsgpr(EXEC_LO.offset), getattr(inst, 'op_regs', {})
  is_16bit = _is_16bit_op(op_name)
  vdst_reg = inst.vdst.offset - 256
  write_hi_half = is_16bit and vdst_reg >= 128
  if write_hi_half: vdst_reg -= 128
  if isinstance(inst, VOP1):
    # Handle VOP1 hi-half source operand (src0 >= v[128] for 16-bit ops)
    src0_off = inst.src0.offset
    if is_16bit and src0_off >= 384:  # v[128]+ in src encoding
      src0_reg = src0_off - 256 - 128  # actual VGPR index
      s0 = ctx.rvgpr(src0_reg, lane)
      s0 = (s0 >> U32_16) & _c(0xFFFF)  # extract hi 16 bits
    else:
      # For 16-bit ops, use f16=True to get F16 inline constants
      s0 = ctx.rsrc_sized(src0_off, lane, sizes, 'src0', f16=is_16bit)
    srcs = {'S0': s0}
  else:
    vsrc1_reg = inst.vsrc1.offset - 256
    vsrc1_hi = is_16bit and vsrc1_reg >= 128
    vsrc1_actual = vsrc1_reg - 128 if vsrc1_hi else vsrc1_reg
    s1 = ctx.rvgpr(vsrc1_actual, lane)
    if vsrc1_hi: s1 = (s1 >> U32_16) & _c(0xFFFF)  # extract hi 16 bits
    # For FMAC/FMAMK hi-half dest, D0 must also read from hi-half (accumulator is in same half as dest)
    d0 = ctx.rvgpr(vdst_reg, lane)
    if write_hi_half: d0 = (d0 >> U32_16) & _c(0xFFFF)  # extract hi 16 bits for accumulator
    # Handle VOP2 hi-half src0 operand (src0 >= v[128] for 16-bit ops)
    src0_off = inst.src0.offset
    if is_16bit and src0_off >= 384:  # v[128]+ in src encoding
      src0_reg = src0_off - 256 - 128  # actual VGPR index
      s0 = ctx.rvgpr(src0_reg, lane)
      s0 = (s0 >> U32_16) & _c(0xFFFF)  # extract hi 16 bits
    else:
      # For 16-bit ops, use bits=16 to get F16 inline constants (e.g. 1.0 -> 0x3c00 not 0x3f800000)
      s0 = ctx.rsrc(src0_off, lane, bits=16 if is_16bit else 32)
    srcs = {'S0': s0, 'S1': s1, 'D0': d0}
    if inst.op in (VOP2Op.V_FMAAK_F32_E32, VOP2Op.V_FMAMK_F32_E32, VOP2Op.V_FMAAK_F16_E32, VOP2Op.V_FMAMK_F16_E32):
      srcs['SIMM32'] = UOp.const(dtypes.uint32, ctx.literal)
  pcode_result = compile_vop_pcode(inst.op, srcs, lane, ctx.wvgpr, ctx.wsgpr, ctx.rsgpr, vdst_reg, exec_mask, ctx.inc_pc, name,
                                   opsel_dst_hi=write_hi_half, rvgpr_fn=ctx.rvgpr)
  assert pcode_result is not None, f"no pcode for {type(inst).__name__}: {inst.op.name}"
  return pcode_result

def _compile_vopc(inst, ctx: _Ctx, name: str, opsel: int = 0, abs_bits: int = 0, neg_bits: int = 0) -> tuple[str, UOp]:
  exec_mask, op_name = ctx.rsgpr(EXEC_LO.offset), _op_name(inst)
  is_cmpx, is_16bit, is_64bit = 'CMPX' in op_name, _is_16bit_op(op_name), 'F64' in op_name
  is_vopc = hasattr(inst, 'vsrc1')  # VOPC (e32) vs VOP3 (e64) format

  # Handle both VOPC (vsrc1) and VOP3 (src1) instruction formats
  if is_vopc:
    vsrc1_reg = inst.vsrc1.offset - 256
    vsrc1_hi, src1_off = is_16bit and vsrc1_reg >= 128, 256 + (vsrc1_reg - 128 if is_16bit and vsrc1_reg >= 128 else vsrc1_reg)
    src0_bits, src1_bits, dst_reg = (64, 64, VCC_LO.offset) if is_64bit else (32, 32, VCC_LO.offset)
  else:
    src1_off, vsrc1_hi, dst_reg = inst.src1.offset, False, inst.vdst.offset
    _, src0_bits, _ = inst.operands.get('src0', (None, 32, None))
    _, src1_bits, _ = inst.operands.get('src1', (None, 32, None))
    is_16bit = src0_bits == 16 or src1_bits == 16

  is_float, pcode = any(x in op_name for x in ('_F32', '_F64', '_F16')), PCODE.get(inst.op)
  def get_cmp_bit(lane) -> UOp:
    lc = lane.cast(dtypes.index) if isinstance(lane, UOp) else _c(lane, dtypes.index)
    s0, s1 = ctx.rsrc(inst.src0.offset, lc, src0_bits), ctx.rsrc(src1_off, lc, src1_bits)
    if is_16bit:
      if vsrc1_hi: s1 = (s1 >> U32_16) & _c(0xFFFF)
      if opsel: s0, s1 = _apply_opsel(s0, 0, opsel), _apply_opsel(s1, 1, opsel)
    if is_float and (abs_bits or neg_bits):
      s0 = _apply_src_mods(s0, 0, abs_bits, neg_bits, is_16bit, src0_bits == 64)
      s1 = _apply_src_mods(s1, 1, abs_bits, neg_bits, is_16bit, src1_bits == 64)
    if pcode is None: return U32_0
    for dest, val in parse_pcode(pcode, {'S0': s0, 'S1': s1}, lane=lc)[1]:
      if '[laneId]' in dest and ('D0' in dest or 'EXEC' in dest): return val.cast(dtypes.uint32)
    return U32_0

  new_bits = _unroll_lanes(get_cmp_bit, exec_mask, apply_exec=False)
  # Both VOPC and VOP3 clear inactive lane bits (hardware verified)
  new_result = new_bits & exec_mask

  # CMPX e32: writes EXEC only; CMPX e64: writes both EXEC and SDST; non-CMPX: writes dst only
  if is_cmpx: stores = [ctx.wsgpr(EXEC_LO.offset, new_result)] + ([] if is_vopc else [ctx.wsgpr(dst_reg, new_result)])
  else: stores = [ctx.wsgpr(dst_reg, new_result)]
  return name, UOp.sink(*stores, ctx.inc_pc(), arg=KernelInfo(name=name))

def _compile_vop3(inst: VOP3, ctx: _Ctx, name: str) -> tuple[str, UOp]:
  exec_mask = ctx.rsgpr(EXEC_LO.offset)
  sizes = getattr(inst, 'op_regs', {})
  opsel, op_name = getattr(inst, 'opsel', 0) or 0, _op_name(inst)

  # Lane operations
  if op_name in ('V_READLANE_B32', 'V_READFIRSTLANE_B32', 'V_READFIRSTLANE_B32_E64', 'V_WRITELANE_B32'):
    pcode_result = compile_lane_pcode(inst.op, inst, ctx.vgpr, ctx.wsgpr, ctx.rsgpr, ctx.rsrc, ctx.inc_pc, name)
    assert pcode_result is not None, f"no pcode for VOP3: {op_name}"
    return pcode_result

  # VOP3 VOPC (v_cmp_*_e64) - delegate to unified VOPC handler
  if 'V_CMP' in op_name or 'V_CMPX' in op_name:
    return _compile_vopc(inst, ctx, name, opsel=opsel, abs_bits=getattr(inst, 'abs', 0) or 0, neg_bits=getattr(inst, 'neg', 0) or 0)

  # Regular VOP3
  lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
  is_f16_op = 'F16' in op_name
  src0 = ctx.rsrc_sized(inst.src0.offset, lane, sizes, 'src0', is_f16_op)
  src1 = ctx.rsrc_sized(inst.src1.offset, lane, sizes, 'src1', is_f16_op)
  src2 = ctx.rsrc_sized(inst.src2.offset, lane, sizes, 'src2', is_f16_op) if inst.src2 is not None else None
  if _is_16bit_op(op_name):
    src0, src1 = _apply_opsel(src0, 0, opsel), _apply_opsel(src1, 1, opsel)
    if src2 is not None: src2 = _apply_opsel(src2, 2, opsel)
  abs_bits, neg_bits = getattr(inst, 'abs', 0) or 0, getattr(inst, 'neg', 0) or 0
  is_16bit_op = _is_16bit_op(op_name)
  if abs_bits or neg_bits:
    src0 = _apply_src_mods(src0, 0, abs_bits, neg_bits, is_16bit_op, sizes.get('src0', 1) == 2)
    if src1 is not None: src1 = _apply_src_mods(src1, 1, abs_bits, neg_bits, is_16bit_op, sizes.get('src1', 1) == 2)
    if src2 is not None: src2 = _apply_src_mods(src2, 2, abs_bits, neg_bits, is_16bit_op, sizes.get('src2', 1) == 2)
  vdst_reg = inst.vdst.offset - 256
  srcs = {'S0': src0, 'S1': src1}
  if src2 is not None: srcs['S2'] = src2
  if inst.op in (VOP3Op.V_CNDMASK_B32_E64, VOP3Op.V_CNDMASK_B16) and src2 is not None: srcs['VCC'] = src2
  # FMAC instructions need D0 (accumulator) from destination register
  if 'FMAC' in op_name: srcs['D0'] = ctx.rvgpr(vdst_reg, lane)
  opsel_dst_hi = bool(opsel & 0b1000) and _is_16bit_op(op_name)
  if opsel_dst_hi:
    stores = compile_vop_pcode(inst.op, srcs, lane, ctx.wvgpr, ctx.wsgpr, ctx.rsgpr, vdst_reg, exec_mask, opsel_dst_hi=True, rvgpr_fn=ctx.rvgpr)
    if stores is not None:
      return name, UOp.sink(*stores, ctx.inc_pc(), arg=KernelInfo(name=name))
  pcode_result = compile_vop_pcode(inst.op, srcs, lane, ctx.wvgpr, ctx.wsgpr, ctx.rsgpr, vdst_reg, exec_mask, ctx.inc_pc, name, rvgpr_fn=ctx.rvgpr)
  assert pcode_result is not None, f"no pcode for VOP3: {inst.op.name}"
  return pcode_result

def _compile_vop3sd(inst: VOP3SD, ctx: _Ctx, name: str) -> tuple[str, UOp]:
  exec_mask = ctx.rsgpr(EXEC_LO.offset)
  sizes = getattr(inst, 'op_regs', {})
  sdst_reg, op_name, vdst_reg = inst.sdst.offset, _op_name(inst), inst.vdst.offset - 256
  pcode = PCODE.get(inst.op)
  assert pcode is not None, f"no pcode for VOP3SD: {op_name}"

  has_carry_in = 'src2' in inst.operands and inst.operands['src2'][2] == OpType.OPR_SREG
  vcc_in_reg = inst.src2.offset if has_carry_in and inst.src2 is not None else sdst_reg

  lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
  src0, src1 = ctx.rsrc_sized(inst.src0.offset, lane, sizes, 'src0'), ctx.rsrc_sized(inst.src1.offset, lane, sizes, 'src1')
  src2 = ctx.rsrc_sized(inst.src2.offset, lane, sizes, 'src2') if inst.src2 is not None else None
  srcs = {'S0': src0, 'S1': src1, 'VCC': ctx.rsgpr(vcc_in_reg), 'EXEC': exec_mask, 'SCC': ctx.rsgpr(SCC_IDX)}
  if src2 is not None: srcs['S2'] = src2
  _, assigns = parse_pcode(pcode, srcs, lane, op_name=op_name)

  has_per_lane_vcc = any('[laneId]' in dest for dest, _ in assigns if dest.startswith('VCC') or dest.startswith('D0.u64'))
  if has_per_lane_vcc:
    # VCC computation: RANGE+REDUCE gets axis ID first (lower ID = runs first)
    # This ensures VCC reads source values BEFORE VGPR stores modify them
    def get_vcc_bit(lane_uop) -> UOp:
      s0, s1 = ctx.rsrc_sized(inst.src0.offset, lane_uop, sizes, 'src0'), ctx.rsrc_sized(inst.src1.offset, lane_uop, sizes, 'src1')
      s2 = ctx.rsrc_sized(inst.src2.offset, lane_uop, sizes, 'src2') if inst.src2 is not None else None
      lane_srcs = {'S0': s0, 'S1': s1, 'VCC': ctx.rsgpr(vcc_in_reg), 'EXEC': exec_mask, 'SCC': ctx.rsgpr(SCC_IDX)}
      if s2 is not None: lane_srcs['S2'] = s2
      vcc_bit = U32_0
      for dest, val in parse_pcode(pcode, lane_srcs, lane_uop, op_name=op_name)[1]:
        if dest.startswith('VCC') or (dest.startswith('D0.u64') and '[laneId]' in dest): vcc_bit = val.cast(dtypes.uint32)
      return vcc_bit
    final_vcc = _unroll_lanes(get_vcc_bit, exec_mask)
    # VGPR stores: RANGE gets axis ID second (higher ID = runs after VCC loop)
    lane3 = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    s0, s1 = ctx.rsrc_sized(inst.src0.offset, lane3, sizes, 'src0'), ctx.rsrc_sized(inst.src1.offset, lane3, sizes, 'src1')
    s2 = ctx.rsrc_sized(inst.src2.offset, lane3, sizes, 'src2') if inst.src2 is not None else None
    lane_srcs = {'S0': s0, 'S1': s1, 'VCC': ctx.rsgpr(vcc_in_reg), 'EXEC': exec_mask, 'SCC': ctx.rsgpr(SCC_IDX)}
    if s2 is not None: lane_srcs['S2'] = s2
    d0_val = None
    for dest, val in parse_pcode(pcode, lane_srcs, lane3, op_name=op_name)[1]:
      if dest.startswith('D0') and '[laneId]' not in dest: d0_val = val
    vgpr_stores = []
    if d0_val is not None:
      if d0_val.dtype in (dtypes.uint64, dtypes.int64, dtypes.float64):
        lo, hi = _split64(d0_val)
        vgpr_stores.extend([ctx.wvgpr(vdst_reg, lane3, lo, exec_mask), ctx.wvgpr(vdst_reg + 1, lane3, hi, exec_mask)])
      else:
        d0_u32 = d0_val.bitcast(dtypes.uint32) if d0_val.dtype in (dtypes.float32, dtypes.half) else d0_val.cast(dtypes.uint32)
        vgpr_stores.append(ctx.wvgpr(vdst_reg, lane3, d0_u32, exec_mask))
    # Write carry output (wsgpr returns GROUP no-op for NULL register 124)
    vcc_write = ctx.wsgpr(sdst_reg, final_vcc)
    if vgpr_stores:
      # VCC write must come first in sink to ensure VCC loop runs before VGPR loop
      return name, UOp.sink(vcc_write, UOp.sink(*vgpr_stores).end(lane3), ctx.inc_pc(), arg=KernelInfo(name=name))
    return name, UOp.sink(vcc_write, ctx.inc_pc(), arg=KernelInfo(name=name))
  else:
    pcode_result = compile_vop_pcode(inst.op, srcs, lane, ctx.wvgpr, ctx.wsgpr, ctx.rsgpr, vdst_reg, exec_mask, ctx.inc_pc, name, sdst_reg=sdst_reg)
    assert pcode_result is not None, f"no pcode for VOP3SD: {op_name}"
    return pcode_result

def _compile_vop3p(inst: VOP3P, ctx: _Ctx, name: str) -> tuple[str, UOp]:
  lane, exec_mask, vdst_reg = UOp.range(32, _next_axis_id(), AxisType.LOOP), ctx.rsgpr(EXEC_LO.offset), inst.vdst.offset - 256
  src0, src1 = ctx.rsrc(inst.src0.offset, lane, 16), ctx.rsrc(inst.src1.offset, lane, 16)
  src2 = ctx.rsrc(inst.src2.offset, lane, 16) if hasattr(inst, 'src2') and inst.src2 is not None else None
  opsel, opsel_hi = getattr(inst, 'opsel', 0) or 0, getattr(inst, 'opsel_hi', 3) if getattr(inst, 'opsel_hi', 3) is not None else 3
  opsel_hi2 = getattr(inst, 'opsel_hi2', 1) if getattr(inst, 'opsel_hi2', 1) is not None else 1
  neg, neg_hi = getattr(inst, 'neg', 0) or 0, getattr(inst, 'neg_hi', 0) or 0
  def get_half_bits(val: UOp, use_hi: bool, apply_neg: bool = False) -> UOp:
    bits = ((val >> UOp.const(dtypes.uint32, 16)) if use_hi else val) & UOp.const(dtypes.uint32, 0xFFFF)
    if apply_neg: bits = bits.cast(dtypes.uint16).bitcast(dtypes.half).neg().bitcast(dtypes.uint16).cast(dtypes.uint32)
    return bits
  def build_remapped_src(src: UOp, opsel_lo_bit: int, opsel_hi_bit: int, neg_lo_bit: int, neg_hi_bit: int) -> UOp:
    return get_half_bits(src, bool(opsel_lo_bit), bool(neg_lo_bit)) | (get_half_bits(src, bool(opsel_hi_bit), bool(neg_hi_bit)) << UOp.const(dtypes.uint32, 16))
  s0_new = build_remapped_src(src0, opsel & 1, opsel_hi & 1, neg & 1, neg_hi & 1)
  s1_new = build_remapped_src(src1, opsel & 2, opsel_hi & 2, neg & 2, neg_hi & 2)
  s2_new = build_remapped_src(src2, opsel & 4, 1 if opsel_hi2 else 0, neg & 4, neg_hi & 4) if src2 is not None else None
  op_name = _op_name(inst)

  # WMMA: Wave Matrix Multiply-Accumulate
  if 'WMMA' in op_name and 'F32_16X16X16_F16' in op_name:
    src0_r, src1_r, src2_r = inst.src0.offset - 256, inst.src1.offset - 256, inst.src2.offset - 256
    def f16_to_f32(bits: UOp) -> UOp: return bits.cast(dtypes.uint16).bitcast(dtypes.half).cast(dtypes.float32)
    def read_f16_mat(src):
      return [f for l in range(16) for r in range(8) for v in [ctx.rvgpr(src + r, UOp.const(dtypes.index, l))]
              for f in [f16_to_f32(v & UOp.const(dtypes.uint32, 0xFFFF)), f16_to_f32(v >> UOp.const(dtypes.uint32, 16))]]
    mat_a, mat_b = read_f16_mat(src0_r), read_f16_mat(src1_r)
    mat_c = [ctx.rvgpr(src2_r + i // 32, UOp.const(dtypes.index, i % 32)).bitcast(dtypes.float32) for i in range(256)]
    mat_d = [sum(mat_a[row*16+k] * mat_b[col*16+k] for k in range(16)) + mat_c[row*16+col] for row in range(16) for col in range(16)]
    stores = [ctx.wvgpr(vdst_reg + i // 32, UOp.const(dtypes.index, i % 32), mat_d[i].bitcast(dtypes.uint32), exec_mask) for i in range(256)]
    return name, UOp.sink(*stores, ctx.inc_pc(), arg=KernelInfo(name=name))

  pcode = PCODE.get(inst.op)
  if pcode is not None:
    if 'FMA_MIX' in op_name:
      combined_opsel_hi = (opsel_hi & 0x3) | ((opsel_hi2 & 0x1) << 2)
      # For FMA_MIX: neg_hi is ABS (not neg!), neg is actual negation
      def apply_abs(v, bit, opsel_hi_bit, opsel_bit):
        if not (neg_hi & bit): return v
        # Apply abs based on whether source is f32 or f16
        if not (combined_opsel_hi & opsel_hi_bit): return v & UOp.const(dtypes.uint32, 0x7FFFFFFF)  # f32 abs
        if opsel & opsel_bit: return v & UOp.const(dtypes.uint32, 0x7FFF0000)  # f16 hi abs (preserve lo)
        return v & UOp.const(dtypes.uint32, 0xFFFF7FFF)  # f16 lo abs (preserve hi)
      def apply_neg_mix(v, bit, opsel_hi_bit, opsel_bit):
        if not (neg & bit): return v
        if not (combined_opsel_hi & opsel_hi_bit): return v ^ UOp.const(dtypes.uint32, 0x80000000)  # f32 neg
        if opsel & opsel_bit: return v ^ UOp.const(dtypes.uint32, 0x80000000)  # f16 hi neg
        return v ^ UOp.const(dtypes.uint32, 0x00008000)  # f16 lo neg
      s0_mod = apply_neg_mix(apply_abs(src0, 1, 1, 1), 1, 1, 1)
      s1_mod = apply_neg_mix(apply_abs(src1, 2, 2, 2), 2, 2, 2)
      s2_mod = apply_neg_mix(apply_abs(src2, 4, 4, 4), 4, 4, 4) if src2 is not None else UOp.const(dtypes.uint32, 0)
      srcs = {'S0': s0_mod, 'S1': s1_mod, 'S2': s2_mod,
              'OPSEL_HI': UOp.const(dtypes.uint32, combined_opsel_hi), 'OPSEL': UOp.const(dtypes.uint32, opsel)}
    else:
      srcs = {'S0': s0_new, 'S1': s1_new}
      if s2_new is not None: srcs['S2'] = s2_new
    stores = compile_vop_pcode(inst.op, srcs, lane, ctx.wvgpr, ctx.wsgpr, ctx.rsgpr, vdst_reg, exec_mask, rvgpr_fn=ctx.rvgpr)
    if stores is not None:
      return name, UOp.sink(*stores, ctx.inc_pc(), arg=KernelInfo(name=name))
  return name, UOp.sink(ctx.inc_pc(), arg=KernelInfo(name=name))

def _compile_vopd(inst: VOPD, ctx: _Ctx, name: str) -> tuple[str, UOp]:
  exec_mask = ctx.rsgpr(EXEC_LO.offset)
  vdstx_reg, vdsty_reg = inst.vdstx.offset - 256, (inst.vdsty << 1) | ((inst.vdstx.offset & 1) ^ 1)
  lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
  srcy0, srcy1 = ctx.rsrc(inst.srcy0.offset, lane), ctx.rvgpr(inst.vsrcy1.offset - 256, lane)
  def wvgpr_after_y(reg: int, lane: UOp, val: UOp, exec_mask: UOp) -> UOp:
    return ctx.wvgpr(reg, lane, val, exec_mask, after=srcy1)
  all_stores = []
  for op, src0_off, vsrc1_off, vdst_reg, label in [(inst.opx, inst.srcx0.offset, inst.vsrcx1.offset, vdstx_reg, 'X'),
                                                     (inst.opy, inst.srcy0.offset, inst.vsrcy1.offset, vdsty_reg, 'Y')]:
    vop = VOPD_TO_VOP2.get(op)
    assert vop is not None, f"no VOP mapping for VOPD {label}: {op}"
    if label == 'Y': srcs = {'S0': srcy0, 'S1': srcy1, 'D0': ctx.rvgpr(vdst_reg, lane)}
    else: srcs = {'S0': ctx.rsrc(src0_off, lane), 'S1': ctx.rvgpr(vsrc1_off - 256, lane), 'D0': ctx.rvgpr(vdst_reg, lane)}
    if op in (VOPDOp.V_DUAL_FMAAK_F32, VOPDOp.V_DUAL_FMAMK_F32): srcs['SIMM32'] = UOp.const(dtypes.uint32, ctx.literal)
    if op == VOPDOp.V_DUAL_CNDMASK_B32: srcs['VCC'] = ctx.rsgpr(VCC_LO.offset)
    pcode = PCODE.get(vop)
    assert pcode is not None, f"no pcode for VOPD {label}: {vop}"
    srcs.update({'VCC': ctx.rsgpr(VCC_LO.offset), 'EXEC': exec_mask, 'SCC': ctx.rsgpr(SCC_IDX)})
    for dest, val in parse_pcode(pcode, srcs, lane, op_name=vop.name)[1]:
      if dest.startswith('D0'): all_stores.append(wvgpr_after_y(vdst_reg, lane, _val_to_u32(val), exec_mask))
  return name, UOp.sink(UOp.group(*all_stores).end(lane), ctx.inc_pc(), arg=KernelInfo(name=name))

def _compile_mem_op(inst, ctx: _Ctx, name: str) -> tuple[str, UOp]:
  """Unified memory operation compiler for DS, FLAT, GLOBAL, SCRATCH."""
  exec_mask, op_name = ctx.rsgpr(EXEC_LO.offset), _op_name(inst)
  pcode = PCODE.get(inst.op)
  if pcode is None: return name, UOp.sink(ctx.inc_pc(), arg=KernelInfo(name=name))

  is_lds = isinstance(inst, DS)
  is_scratch = isinstance(inst, SCRATCH)
  mem = ctx.lds if is_lds else ctx.scratch if is_scratch else ctx.vmem
  addr_shift = UOp.const(dtypes.uint32 if is_lds else dtypes.uint64, 2)

  # Extract register info
  def vreg(attr): return (getattr(inst, attr).offset - 256) if hasattr(inst, attr) and getattr(inst, attr).offset >= 256 else (getattr(inst, attr).offset if hasattr(inst, attr) else 0)
  addr_reg = vreg('addr')
  vdata_reg = vreg('data0') if is_lds else (inst.data.offset - 256 if hasattr(inst, 'data') and inst.data else 0)
  vdst_reg = vreg('vdst') if is_lds else (inst.vdst.offset - 256 if hasattr(inst, 'vdst') and inst.vdst else vdata_reg)
  has_saddr = not is_lds and hasattr(inst, 'saddr') and inst.saddr != NULL and inst.saddr.offset < 128

  # Offset handling
  offset0, offset1 = (getattr(inst, 'offset0', 0) or 0, getattr(inst, 'offset1', 0) or 0) if is_lds else (0, 0)
  offset = (getattr(inst, 'offset', offset0) or offset0) if is_lds else _sext(getattr(inst, 'offset', 0), 13)

  # Data width
  ndwords = 4 if '_B128' in op_name or 'B128' in op_name else 3 if '_B96' in op_name or 'B96' in op_name else 2 if '_B64' in op_name or 'B64' in op_name else 1
  is_64bit = ndwords >= 2 or '_U64' in op_name or '_I64' in op_name or '_F64' in op_name
  is_atomic, glc = 'ATOMIC' in op_name, getattr(inst, 'glc', 0)
  has_data1 = is_lds and hasattr(inst, 'data1') and inst.data1 is not None
  data1_reg = vreg('data1') if is_lds else 0

  def make_addr(lane: UOp) -> UOp:
    if is_lds: return ctx.rvgpr(addr_reg, lane)
    if is_scratch:
      scratch_stride = ctx.rsgpr(SCRATCH_STRIDE_IDX).cast(dtypes.uint64)
      base = lane.cast(dtypes.uint64) * scratch_stride
      addr_offset = ctx.rvgpr(addr_reg, lane).cast(dtypes.uint64)
      if has_saddr: addr_offset = addr_offset + ctx.rsgpr(inst.saddr.offset).cast(dtypes.uint64)
      return base + addr_offset + UOp.const(dtypes.uint64, offset)
    if has_saddr: return ctx.rsgpr64(inst.saddr.offset) + ctx.rvgpr(addr_reg, lane).cast(dtypes.uint64) + UOp.const(dtypes.uint64, offset)
    return _u64(ctx.rvgpr(addr_reg, lane), ctx.rvgpr(addr_reg + 1, lane)) + UOp.const(dtypes.uint64, offset)

  def wmem(addr: UOp, val: UOp, active: UOp) -> UOp:
    idx = mem.index((addr >> addr_shift).cast(dtypes.index))
    return idx.store(active.where(val, idx.load()))

  def make_srcs(lane: UOp) -> dict:
    addr = make_addr(lane)
    if is_lds:
      if 'B128' in op_name or 'B96' in op_name:
        data = {'DATA': ctx.rvgpr(vdata_reg, lane), 'DATA1': ctx.rvgpr(vdata_reg + 1, lane),
                'DATA2': ctx.rvgpr(vdata_reg + 2, lane), 'DATA3': ctx.rvgpr(vdata_reg + 3, lane)}
      elif 'B32' in op_name:
        data = {'DATA': ctx.rvgpr(vdata_reg, lane), 'DATA2': ctx.rvgpr(data1_reg, lane) if has_data1 else UOp.const(dtypes.uint32, 0)}
      else:
        data = {'DATA': _u64(ctx.rvgpr(vdata_reg, lane), ctx.rvgpr(vdata_reg + 1, lane)),
                'DATA2': _u64(ctx.rvgpr(data1_reg, lane), ctx.rvgpr(data1_reg + 1, lane)) if has_data1 else UOp.const(dtypes.uint64, 0)}
      return {'ADDR': addr, 'ADDR_BASE': addr, 'OFFSET': UOp.const(dtypes.uint32, offset),
              'OFFSET0': UOp.const(dtypes.uint32, offset0), 'OFFSET1': UOp.const(dtypes.uint32, offset1), '_lds': mem, **data}
    active = _lane_active(exec_mask, lane)
    if is_atomic:
      return {'ADDR': addr, 'DATA': _u64(ctx.rvgpr(vdata_reg, lane), ctx.rvgpr(vdata_reg + 1, lane)) if is_64bit else ctx.rvgpr(vdata_reg, lane),
              '_vmem': mem, '_active': active}
    vdata = ctx.rvgpr(vdata_reg, lane).cast(dtypes.uint64) if 'STORE' in op_name else ctx.rvgpr(vdst_reg, lane) if 'D16' in op_name else UOp.const(dtypes.uint32, 0)
    if 'STORE' in op_name and ndwords >= 2: vdata = vdata | (ctx.rvgpr(vdata_reg + 1, lane).cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))
    srcs = {'ADDR': addr, 'VDATA': vdata, '_vmem': mem, '_active': active}
    for i in range(ndwords): srcs[f'VDATA{i}'] = ctx.rvgpr(vdata_reg + i, lane) if 'STORE' in op_name else UOp.const(dtypes.uint32, 0)
    return srcs

  def make_stores(dest: str, val: UOp, lane: UOp, active: UOp, writes_return_data: bool, pcode_vars: dict) -> list[UOp]:
    if dest.startswith('MEM['):
      if is_lds or is_atomic: return _write_val(dest, val[1], wmem, val[0], active)
      data_bits = 8 if '.b8' in dest else 16 if '.b16' in dest else 64 if '.b64' in dest else 32
      if is_scratch: return _mem_store_bytes(mem, val[0], val[1], active, data_bits)
      return _mem_store(mem, val[0], val[1], active, 64, data_bits)
    if dest.startswith('RETURN_DATA') and writes_return_data:
      if (m := re.match(r'RETURN_DATA\[(\d+)\s*:\s*(\d+)\]', dest)):
        bit_width, dword_idx = int(m.group(1)) - int(m.group(2)) + 1, int(m.group(2)) // 32
        is_64 = '.b64' if bit_width == 64 else ''
        return _write_val(is_64, val, lambda r, v, l, e: ctx.wvgpr(r, l, v, e), vdst_reg + dword_idx, lane, exec_mask)
      return _write_val(dest, val, lambda r, v, l, e: ctx.wvgpr(r, l, v, e), vdst_reg, lane, exec_mask)
    return []

  # DS-specific: check for 2ADDR pattern needing separate ranges
  if is_lds:
    dummy_lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
    _, assigns = parse_pcode(pcode, make_srcs(dummy_lane), lane=dummy_lane, op_name=op_name)
    mem_assigns = [d for d, _ in assigns if d.startswith('MEM[')]
    mem_addrs = set(re.match(r'MEM\[([^\]]+)\]', d).group(1) if re.match(r'MEM\[([^\]]+)\]', d) else d for d in mem_assigns)
    use_separate_ranges = (len(mem_addrs) > 1 or '2ADDR' in op_name) and 'STOREXCHG' not in op_name
    if use_separate_ranges:
      ended = []
      for i, (dest, _) in enumerate(assigns):
        lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
        active = _lane_active(exec_mask, lane)
        _, lane_assigns = parse_pcode(pcode, make_srcs(lane), lane=lane, op_name=op_name)
        ended.extend(s.end(lane) for s in make_stores(dest, lane_assigns[i][1], lane, active, True, {}))
      return (name, UOp.sink(*ended, ctx.inc_pc(), arg=KernelInfo(name=name))) if ended else (name, UOp.sink(ctx.inc_pc(), arg=KernelInfo(name=name)))

  # Standard path: single lane range
  writes_return_data = '_RTN' in op_name or (is_lds and op_name.startswith('DS_LOAD')) or (is_atomic and glc)
  lane = UOp.range(32, _next_axis_id(), AxisType.LOOP)
  active = _lane_active(exec_mask, lane)
  pcode_vars, assigns = parse_pcode(pcode, make_srcs(lane), lane=lane, op_name=op_name)
  stores = [s for dest, val in assigns for s in make_stores(dest, val, lane, active, writes_return_data, pcode_vars)]

  # FLAT/GLOBAL/SCRATCH: collect VDATA slices for loads
  if not is_lds and not is_atomic:
    for dword_idx, val in sorted(_collect_data_slices(assigns, 'VDATA', pcode_vars, op_name).items()):
      stores.append(ctx.wvgpr(vdst_reg + dword_idx, lane, val, exec_mask))

  if stores: return name, UOp.sink(UOp.sink(*stores).end(lane), ctx.inc_pc(), arg=KernelInfo(name=name))
  return name, UOp.sink(ctx.inc_pc(), arg=KernelInfo(name=name))

# Dispatch table: instruction type -> handler function
_INST_HANDLERS: dict[type, callable] = {
  SOPP: _compile_sopp, SMEM: _compile_smem, SOP1: _compile_sop, SOP2: _compile_sop, SOPC: _compile_sop, SOPK: _compile_sop,
  VOP1: _compile_vop12, VOP1_SDST: _compile_vop12, VOP2: _compile_vop12, VOPC: _compile_vopc, VOP3: _compile_vop3, VOP3_SDST: _compile_vop3,
  VOP3SD: _compile_vop3sd, VOP3P: _compile_vop3p, VOPD: _compile_vopd,
  DS: _compile_mem_op, FLAT: _compile_mem_op, GLOBAL: _compile_mem_op, SCRATCH: _compile_mem_op,
}

# ═══════════════════════════════════════════════════════════════════════════════
# PROGRAM DECODE AND COMPILATION
# ═══════════════════════════════════════════════════════════════════════════════

# Backend selection: EMU2_BACKEND=python, clang (default), or llvm
EMU2_BACKEND = getenv("EMU2_BACKEND", "clang")

def _get_backend():
  """Get renderer, compiler, and program class based on EMU2_BACKEND."""
  if EMU2_BACKEND == "python":
    from tinygrad.runtime.ops_python import PythonRenderer, PythonCompiler, PythonProgram
    return PythonRenderer(), PythonCompiler(), PythonProgram
  elif EMU2_BACKEND == "llvm":
    from tinygrad.renderer.llvmir import LLVMRenderer
    from tinygrad.runtime.support.compiler_cpu import CPULLVMCompiler
    from tinygrad.runtime.ops_cpu import CPUProgram
    return LLVMRenderer(), CPULLVMCompiler(), CPUProgram
  else:  # clang (default)
    from tinygrad.renderer.cstyle import ClangRenderer
    from tinygrad.runtime.support.compiler_cpu import ClangJITCompiler
    from tinygrad.runtime.ops_cpu import CPUProgram
    return ClangRenderer(), ClangJITCompiler(), CPUProgram

_emu_renderer, _emu_compiler, _ProgramClass = _get_backend()

def _elf_symbol_offsets(obj: bytes) -> dict[str, int]:
  """Parse ELF object file and return {symbol_name: offset} for all defined symbols."""
  from tinygrad.runtime.support.elf import elf_loader, libc
  def _strtab(blob: bytes, idx: int) -> str: return blob[idx:blob.find(b'\x00', idx)].decode('utf-8')
  _, sections, _ = elf_loader(obj)
  symtab_sec = next((s for s in sections if s.header.sh_type == libc.SHT_SYMTAB), None)
  if symtab_sec is None: return {}
  strtab_sec = sections[symtab_sec.header.sh_link] if symtab_sec.header.sh_link < len(sections) else None
  if strtab_sec is None: return {}
  symbols = (libc.Elf64_Sym * (symtab_sec.header.sh_size // symtab_sec.header.sh_entsize)).from_buffer_copy(symtab_sec.content)
  return {name: sections[sym.st_shndx].header.sh_addr + sym.st_value
          for sym in symbols if 0 < sym.st_shndx < len(sections) and (name := _strtab(strtab_sec.content, sym.st_name))}

@functools.cache
def _get_inst_sink(inst_bytes: bytes) -> UOp:
  """Build UOp sink for instruction bytes. Cached by instruction bytes."""
  inst = decode_inst(inst_bytes)
  name = f"{_op_name(inst).lower()}_{inst_bytes[:inst.size()].hex()}"
  sgpr, vgpr, vmem, lds, scratch = _define_bufs()
  inst_words = inst.size() // 4
  is_8byte_base = isinstance(inst, (VOP3, VOP3SD, VOP3P, SMEM, DS, FLAT, GLOBAL, VOPD, SCRATCH))
  lit_off = 8 if is_8byte_base else 4
  literal = int.from_bytes(inst_bytes[lit_off:lit_off+4], 'little') if len(inst_bytes) >= lit_off + 4 else 0
  ctx = _Ctx(sgpr, vgpr, vmem, lds, scratch, literal, inst_words)

  # Look up handler by type, falling back to base classes for _LIT variants
  handler = _INST_HANDLERS.get(type(inst))
  if handler is None:
    for base in type(inst).__mro__:
      if base in _INST_HANDLERS:
        handler = _INST_HANDLERS[base]
        break
  if handler is None: raise RuntimeError(f"[emu2] unimplemented instruction type: {type(inst).__name__} {_op_name(inst)}")
  _, sink = handler(inst, ctx, name)
  return sink

@functools.cache
def _get_inst_prg(inst_bytes: bytes) -> ProgramSpec:
  """Compile instruction bytes to ProgramSpec. Cached by instruction bytes."""
  with Context(NOOPT=1, IGNORE_OOB=1, TUPLE_ORDER=0):
    return get_program(_get_inst_sink(inst_bytes), _emu_renderer)

@functools.cache
def decode_program(data: bytes) -> dict[int, tuple[str, object, list[int], object]]:
  """Decode program to {pc: (name, program, globals, holder)}."""

  # Collect all instruction programs
  inst_info: list[tuple[int, ProgramSpec]] = []  # (pc, prg)
  i = 0
  while i < len(data):
    inst = decode_inst(data[i:])
    if isinstance(inst, SOPP) and inst.op == SOPPOp.S_CODE_END: break
    try:
      prg = _get_inst_prg(bytes(data[i:i + inst.size() + 4]))
      inst_info.append((i // 4, prg))
      if DEBUG >= 2:
        try: inst_str = repr(inst)
        except Exception: inst_str = f"<{type(inst).__name__} at PC={i//4}>"
        print(f"[emu2] PC={i//4}: {inst_str}")
        if DEBUG >= 3: print(f"{colored(prg.src, 'BLACK')}")
    except Exception as e:
      try: inst_str = repr(inst)
      except Exception: inst_str = f"<{type(inst).__name__}>"
      raise RuntimeError(f"[emu2] Failed to compile PC={i//4} {inst_str}: {type(e).__name__}: {e}") from e
    i += inst.size()

  if not inst_info: return {}

  if EMU2_BACKEND == "python":
    # Python backend: create PythonProgram instances directly
    return {pc: (prg.function_name, _ProgramClass(prg.function_name, _emu_compiler.compile(prg.src)), prg.globals, None)
            for pc, prg in inst_info}
  else:
    # Clang/LLVM backend: batch compile and create function pointers
    from tinygrad.runtime.support.elf import jit_loader
    seen_funcs: set[str] = set()
    combined_src_parts: list[str] = []
    for pc, prg in inst_info:
      if prg.function_name not in seen_funcs:
        seen_funcs.add(prg.function_name)
        combined_src_parts.append(prg.src)
    obj = _emu_compiler.compile_to_obj("\n".join(combined_src_parts))
    sym_offsets = _elf_symbol_offsets(obj)
    cpu_prg = _ProgramClass(Device['CPU'], "emu2_batch", jit_loader(obj))
    base_addr = ctypes.cast(cpu_prg.fxn, ctypes.c_void_p).value
    return {pc: (prg.function_name, ctypes.CFUNCTYPE(None)(base_addr + sym_offsets.get(prg.function_name, 0)), prg.globals, cpu_prg)
            for pc, prg in inst_info}

# ═══════════════════════════════════════════════════════════════════════════════
# WAVE STATE
# ═══════════════════════════════════════════════════════════════════════════════

class WaveState:
  __slots__ = ('vgpr_buf', 'sgpr_buf', '_vgpr_mv', '_sgpr_mv', 'n_lanes')

  def __init__(self, n_lanes: int = WAVE_SIZE):
    self.n_lanes = n_lanes
    self.vgpr_buf = Buffer('CPU', VGPR_SIZE, dtypes.uint32).ensure_allocated()
    self.sgpr_buf = Buffer('CPU', SGPR_COUNT, dtypes.uint32).ensure_allocated()
    self._vgpr_mv = self.vgpr_buf.as_buffer(force_zero_copy=True).cast('I')
    self._sgpr_mv = self.sgpr_buf.as_buffer(force_zero_copy=True).cast('I')
    # Zero memory using ctypes memset (much faster than Python loops)
    ctypes.memset(self.vgpr_buf._buf.va_addr, 0, VGPR_SIZE * 4)
    ctypes.memset(self.sgpr_buf._buf.va_addr, 0, SGPR_COUNT * 4)
    self._write_sgpr(EXEC_LO.offset, (1 << n_lanes) - 1)
    self._write_sgpr(PC_LO_IDX, 0)

  def _write_sgpr(self, idx: int, val: int): self._sgpr_mv[idx] = val & MASK32
  def _read_sgpr(self, idx: int) -> int: return self._sgpr_mv[idx]
  def _write_vgpr(self, reg: int, lane: int, val: int): self._vgpr_mv[reg * 32 + lane] = val & MASK32
  def _read_vgpr(self, reg: int, lane: int) -> int: return self._vgpr_mv[reg * 32 + lane]

  @property
  def pc(self) -> int: return self._read_sgpr(PC_LO_IDX)
  @pc.setter
  def pc(self, val: int): self._write_sgpr(PC_LO_IDX, val)

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_asm(lib: int, lib_sz: int, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, args_ptr: int, rsrc2: int = 0x19c,
            scratch_size: int = 0) -> int:
  """Execute AMD assembly program. scratch_size is private_segment_fixed_size from kernel descriptor (per-lane)."""
  program = decode_program(bytes((ctypes.c_char * lib_sz).from_address(lib).raw))
  lds_size = ((rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_SHIFT) * 512
  total_threads = lx * ly * lz

  if EMU2_BACKEND == "python":
    # Python backend: use memoryview buffers with address-0 trick for vmem
    vmem_mv = memoryview((ctypes.c_char * (1 << 47)).from_address(0)).cast('B')  # 128TB at address 0
    lds_mv = memoryview(bytearray(max(lds_size, 4)))
    scratch_mv = memoryview(bytearray(scratch_size * WAVE_SIZE)) if scratch_size else None
  else:
    # Clang backend: use Buffer objects with external_ptr=0 for vmem
    vmem_buf = Buffer('CPU', 1 << 40, dtypes.uint32, options=BufferSpec(external_ptr=0)).ensure_allocated()
    lds_buf = Buffer('CPU', max(lds_size // 4, 1), dtypes.uint32).ensure_allocated()
    scratch_buf = Buffer('CPU', scratch_size * WAVE_SIZE, dtypes.uint8).ensure_allocated() if scratch_size else None

  # Set DAZ+FTZ during emulator execution, restore afterward to avoid breaking hypothesis tests
  with _MXCSRContext():
    for gidx in range(gx):
      for gidy in range(gy):
        for gidz in range(gz):
          for wave_start in range(0, total_threads, WAVE_SIZE):
            n_lanes, st = min(WAVE_SIZE, total_threads - wave_start), WaveState(min(WAVE_SIZE, total_threads - wave_start))
            st._write_sgpr(0, args_ptr & MASK32)
            st._write_sgpr(1, (args_ptr >> 32) & MASK32)

            # Workgroup IDs in SGPRs after user SGPRs
            sgpr_idx = (rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT
            for enabled, gid in [(hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X, gidx),
                                 (hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y, gidy),
                                 (hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z, gidz)]:
              if rsrc2 & enabled: st._write_sgpr(sgpr_idx, gid); sgpr_idx += 1

            # v0 = packed workitem IDs, scratch stride in secret SGPR
            for lane in range(n_lanes):
              tid = wave_start + lane
              st._write_vgpr(0, lane, ((tid // (lx * ly)) << 20) | (((tid // lx) % ly) << 10) | (tid % lx))
            st._write_sgpr(SCRATCH_STRIDE_IDX, scratch_size)

            if EMU2_BACKEND == "python":
              # Python backend: pass raw byte memoryviews
              sgpr_raw = st.sgpr_buf.as_buffer(force_zero_copy=True)
              vgpr_raw = st.vgpr_buf.as_buffer(force_zero_copy=True)
              all_bufs = [sgpr_raw, vgpr_raw, vmem_mv, lds_mv, scratch_mv]
              for inst_count in range(1_000_000):
                if (pc := st.pc) == 0xFFFFFFFF or pc not in program: break
                name, prg, globals_list, _ = program[pc]
                assert prg is not None, f"[emu2] No program for {name} at PC={pc}"
                assert 4 not in globals_list or scratch_mv, f"SCRATCH instruction {name} but scratch_size=0"
                bufs = [all_bufs[g] for g in globals_list]
                prg(*bufs, global_size=(1,1,1), local_size=(1,1,1))
              else: raise RuntimeError("exceeded 1M instructions, likely infinite loop")
            else:
              # Clang/LLVM backend: pass buffer addresses via ctypes (pre-create to avoid allocation in loop)
              c_bufs = [ctypes.c_uint64(st.sgpr_buf._buf.va_addr), ctypes.c_uint64(st.vgpr_buf._buf.va_addr),
                        ctypes.c_uint64(vmem_buf._buf.va_addr), ctypes.c_uint64(lds_buf._buf.va_addr),
                        ctypes.c_uint64(scratch_buf._buf.va_addr if scratch_buf else 0)]
              c_lane = ctypes.c_int32(0)
              for inst_count in range(1_000_000):
                if (pc := st.pc) == 0xFFFFFFFF or pc not in program: break
                name, fxn, globals_list, _ = program[pc]
                assert fxn is not None, f"[emu2] No fxn for {name} at PC={pc}"
                assert 4 not in globals_list or scratch_buf, f"SCRATCH instruction {name} but scratch_size=0"
                fxn(*[c_bufs[g] for g in globals_list], c_lane)
              else: raise RuntimeError("exceeded 1M instructions, likely infinite loop")
  return 0
