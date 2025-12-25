# RDNA3 instruction rendering functions and string_rewrite PatternMatcher
import struct
from typing import Callable
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat, GroupOp
from tinygrad.dtype import dtypes, DType, PtrDType

# *** Utility functions ***
def get_reg_base(reg: str) -> int: return int(reg[2:reg.index(':')]) if '[' in reg else int(reg[1:])
def render_val(x, dtype):
  if dtypes.is_float(dtype):
    if dtype == dtypes.half: return "0x%04X" % struct.unpack("H", struct.pack("e", x))[0]
    if float(x) in (0.0, 0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0, -4.0): return str(float(x)) if float(x) != 0.0 else "0"
    if dtype == dtypes.double: return "0x%016X" % struct.unpack("Q", struct.pack("d", x))[0]
    return "0x%08X" % struct.unpack("I", struct.pack("f", x))[0]
  val = int(x) & 0xFFFFFFFF
  return f"0x{val:08X}" if val > 0x7FFFFFFF else str(val)
def can_inline_const(val, dtype) -> bool:
  if dtype in (dtypes.float64, dtypes.long, dtypes.ulong, dtypes.half): return False
  if dtypes.is_float(dtype): return float(val) in (0.0, 0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0, -4.0)
  try: return -16 <= int(val) <= 64
  except (TypeError, ValueError): return False
def extract_low_32(reg: str) -> str: return f"v{get_reg_base(reg)}" if isinstance(reg, str) and '[' in reg else reg
def get_32bit_reg(ctx, uop) -> str: return extract_low_32(ctx.r[uop]) if uop.dtype in (dtypes.long, dtypes.ulong) else ctx.r[uop]

# *** Type conversion mixin - provides render_* methods for RDNARenderer ***
class RDNARenderMixin:
  """Mixin providing type conversion helpers and render methods for RDNA3."""
  # Type classifications for cleaner dispatch
  SIGNED_INTS = (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.long)
  UNSIGNED_INTS = (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.ulong)
  SMALL_SIGNED_INTS = (dtypes.int8, dtypes.int16)
  SMALL_UNSIGNED_INTS = (dtypes.uint8, dtypes.uint16)
  SMALL_INTS = SMALL_SIGNED_INTS + SMALL_UNSIGNED_INTS
  INT64_TYPES = (dtypes.long, dtypes.ulong)
  FLOAT_TYPES = (dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64)

  def _get_dst_pair(self, rx: str) -> tuple[int, int]:
    base = get_reg_base(rx); return base, base + 1
  def _get_src_pair(self, ra: str) -> tuple[str, str]:
    base = get_reg_base(ra); return f"v{base}", f"v{base+1}"
  def _sign_extend_32_to_64(self, dst_lo: int, dst_hi: int, src: str) -> list[str]:
    return [f"v_mov_b32 v{dst_lo}, {src}", f"v_ashrrev_i32 v{dst_hi}, 31, {src}"]
  def _zero_extend_32_to_64(self, dst_lo: int, dst_hi: int, src: str) -> list[str]:
    return [f"v_mov_b32 v{dst_lo}, {src}", f"v_mov_b32 v{dst_hi}, 0"]
  def _sign_extend_small_int(self, dst: str, src: str, src_dtype) -> str:
    return f"v_bfe_i32 {dst}, {src}, 0, {8 if src_dtype.itemsize == 1 else 16}"
  def _bf16_to_f32(self, dst: str, src: str) -> str: return f"v_lshlrev_b32 {dst}, 16, {src}"
  def _f32_to_bf16(self, dst: str, src: str) -> str: return f"v_lshrrev_b32 {dst}, 16, {src}"
  def _cvt_f32_to_int(self, dst: str, src: str, signed: bool) -> str: return f"v_cvt_{'i' if signed else 'u'}32_f32 {dst}, {src}"
  def _cvt_int_to_f32(self, dst: str, src: str, signed: bool) -> str: return f"v_cvt_f32_{'i' if signed else 'u'}32 {dst}, {src}"
  def _cvt_f64_to_f32(self, dst: str, src_lo: int) -> str: return f"v_cvt_f32_f64 {dst}, v[{src_lo}:{src_lo+1}]"
  def _cvt_f32_to_f64(self, dst_lo: int, src: str) -> str: return f"v_cvt_f64_f32 v[{dst_lo}:{dst_lo+1}], {src}"
  def _cvt_f64_to_int(self, dst: str, src_lo: int, signed: bool) -> str: return f"v_cvt_{'i' if signed else 'u'}32_f64 {dst}, v[{src_lo}:{src_lo+1}]"
  def _cvt_i64_to_f32(self, dst: str, src_lo: str, src_hi: str, signed: bool) -> list[str]:
    s = self.get_scratch_vgpr()
    return [f"v_cvt_f64_u32 v[{s}:{s+1}], {src_lo}", f"v_cvt_f64_{'i' if signed else 'u'}32 v[{s+2}:{s+3}], {src_hi}",
            f"v_ldexp_f64 v[{s+2}:{s+3}], v[{s+2}:{s+3}], 32", f"v_add_f64 v[{s}:{s+1}], v[{s}:{s+1}], v[{s+2}:{s+3}]", f"v_cvt_f32_f64 {dst}, v[{s}:{s+1}]"]

  def render_special(self, x: UOp) -> str:
    dim = int(x.arg[-1])
    if x.arg[0] == 'g': return f"v_mov_b32 {self.r[x]}, s{2+dim}"
    if x.arg[0] == 'l': return f"v_and_b32 {self.r[x]}, 0x3ff, v0" if dim == 0 else f"v_bfe_u32 {self.r[x]}, v0, {dim*10}, 10"
    return f"v_mov_b32 {self.r[x]}, 0"

  def render_cast_from_bool(self, x: UOp, a: UOp) -> str|list[str]:
    rx, ra, dst_t = self.r[x], self.r[a], x.dtype
    if ra.startswith('s'):
      if dst_t == dtypes.float64:
        s, dst_lo = self.get_scratch_vgpr(), self._get_dst_pair(rx)[0]
        return [f"v_cndmask_b32 v{s}, 0, 1, {ra}", f"v_cvt_f32_u32 v{s}, v{s}", self._cvt_f32_to_f64(dst_lo, f"v{s}")]
      if dst_t in self.INT64_TYPES: dst_lo, dst_hi = self._get_dst_pair(rx); return [f"v_cndmask_b32 v{dst_lo}, 0, 1, {ra}", f"v_mov_b32 v{dst_hi}, 0"]
      return f"v_cndmask_b32 {rx}, {render_val(0, dst_t)}, {render_val(1, dst_t)}, {ra}"
    if dst_t == dtypes.float64:
      s, dst_lo = self.get_scratch_vgpr(), self._get_dst_pair(rx)[0]
      return [f"v_cvt_f32_u32 v{s}, {ra}", self._cvt_f32_to_f64(dst_lo, f"v{s}")]
    if dst_t in self.INT64_TYPES: dst_lo, dst_hi = self._get_dst_pair(rx); return [f"v_mov_b32 v{dst_lo}, {ra}", f"v_mov_b32 v{dst_hi}, 0"]
    if dtypes.is_float(dst_t):
      if dst_t == dtypes.float32: return f"v_cvt_f32_u32 {rx}, {ra}"
      s = self.get_scratch_vgpr()
      if dst_t == dtypes.float16: return [f"v_cvt_f32_u32 v{s}, {ra}", f"v_cvt_f16_f32 {rx}, v{s}"]
      if dst_t == dtypes.bfloat16: return [f"v_cvt_f32_u32 v{s}, {ra}", self._f32_to_bf16(rx, f"v{s}")]
    return f"v_mov_b32 {rx}, {ra}"

  def render_cast_to_bool(self, x: UOp, a: UOp) -> str|list[str]:
    rx, ra, src_t = self.r[x], self.r[a], a.dtype
    if src_t in self.INT64_TYPES:
      s, (src_lo, src_hi) = self.get_scratch_vgpr(), self._get_src_pair(ra)
      return [f"v_or_b32 v{s}, {src_lo}, {src_hi}", f"v_cmp_ne_u32 vcc_lo, v{s}, 0", f"v_cndmask_b32 {rx}, 0, 1, vcc_lo"]
    if src_t == dtypes.float64:
      src_base = get_reg_base(ra)
      return [f"v_cmp_neq_f64 vcc_lo, v[{src_base}:{src_base+1}], 0", f"v_cndmask_b32 {rx}, 0, 1, vcc_lo"]
    cmp_op = "neq" if dtypes.is_float(src_t) else "ne"
    return [f"v_cmp_{cmp_op}_{self.types[src_t]} vcc_lo, {ra}, 0", f"v_cndmask_b32 {rx}, 0, 1, vcc_lo"]

  def render_cast(self, x: UOp, a: UOp) -> str|list[str]:
    dst_t, src_t, rx, ra = x.dtype, a.dtype, self.r[x], self.r[a]
    src_signed = src_t in self.SIGNED_INTS
    if dst_t == dtypes.float32 and src_t in (dtypes.int32, dtypes.uint32): return self._cvt_int_to_f32(rx, ra, signed=src_t == dtypes.int32)
    if dst_t in (dtypes.int32, dtypes.uint32) and src_t == dtypes.float32: return self._cvt_f32_to_int(rx, ra, signed=dst_t == dtypes.int32)
    if dst_t in self.SMALL_INTS and src_t == dtypes.float32: return self._cvt_f32_to_int(rx, ra, signed=dst_t in self.SMALL_SIGNED_INTS)
    if dst_t == dtypes.float32 and src_t in self.SMALL_INTS:
      return [self._sign_extend_small_int(rx, ra, src_t), self._cvt_int_to_f32(rx, rx, signed=True)] if src_t in self.SMALL_SIGNED_INTS else self._cvt_int_to_f32(rx, ra, signed=False)
    if (dst_t, src_t) == (dtypes.float16, dtypes.float32): return f"v_cvt_f16_f32 {rx}, {ra}"
    if (dst_t, src_t) == (dtypes.float32, dtypes.float16): return f"v_cvt_f32_f16 {rx}, {ra}"
    if (dst_t, src_t) == (dtypes.bfloat16, dtypes.float32): return self._f32_to_bf16(rx, ra)
    if (dst_t, src_t) == (dtypes.float32, dtypes.bfloat16): return self._bf16_to_f32(rx, ra)
    if {dst_t, src_t} == {dtypes.float16, dtypes.bfloat16}:
      s = self.get_scratch_vgpr()
      return [f"v_cvt_f32_f16 v{s}, {ra}", self._f32_to_bf16(rx, f"v{s}")] if src_t == dtypes.float16 else [self._bf16_to_f32(f"v{s}", ra), f"v_cvt_f16_f32 {rx}, v{s}"]
    if (dst_t, src_t) == (dtypes.float64, dtypes.float32): return self._cvt_f32_to_f64(self._get_dst_pair(rx)[0], ra)
    if (dst_t, src_t) == (dtypes.float32, dtypes.float64): return self._cvt_f64_to_f32(rx, get_reg_base(ra))
    if dst_t in self.INT64_TYPES:
      dst_lo, dst_hi = self._get_dst_pair(rx); signed_dst = dst_t == dtypes.long; s = self.get_scratch_vgpr()
      hi_extend = [f"v_ashrrev_i32 v{dst_hi}, 31, v{dst_lo}"] if signed_dst else [f"v_mov_b32 v{dst_hi}, 0"]
      if src_t == dtypes.float32: return [self._cvt_f32_to_int(f"v{dst_lo}", ra, signed_dst)] + hi_extend
      if src_t in (dtypes.float16, dtypes.bfloat16):
        to_f32 = f"v_cvt_f32_f16 v{s}, {ra}" if src_t == dtypes.float16 else self._bf16_to_f32(f"v{s}", ra)
        return [to_f32, self._cvt_f32_to_int(f"v{dst_lo}", f"v{s}", signed_dst)] + hi_extend
      if src_t == dtypes.float64: return [self._cvt_f64_to_int(f"v{dst_lo}", get_reg_base(ra), signed_dst)] + hi_extend
      if src_t in self.SMALL_SIGNED_INTS: return [self._sign_extend_small_int(f"v{dst_lo}", ra, src_t)] + hi_extend
      if src_t == dtypes.int32: return self._sign_extend_32_to_64(dst_lo, dst_hi, ra)
      if src_t in (dtypes.uint32, *self.SMALL_UNSIGNED_INTS): return self._zero_extend_32_to_64(dst_lo, dst_hi, ra)
    if src_t in self.INT64_TYPES:
      src_lo, src_hi = self._get_src_pair(ra); signed, s = src_t == dtypes.long, self.get_scratch_vgpr()
      if dst_t == dtypes.float32: return self._cvt_i64_to_f32(rx, src_lo, src_hi, signed)
      if dst_t in (dtypes.float16, dtypes.bfloat16):
        instrs = self._cvt_i64_to_f32(f"v{s}", src_lo, src_hi, signed)
        instrs.append(f"v_cvt_f16_f32 {rx}, v{s}" if dst_t == dtypes.float16 else self._f32_to_bf16(rx, f"v{s}")); return instrs
      if dst_t.itemsize <= 4: return f"v_mov_b32 {rx}, {src_lo}"
    if dtypes.is_int(dst_t) and src_t in (dtypes.float16, dtypes.bfloat16):
      s = self.get_scratch_vgpr()
      return [f"v_cvt_f32_f16 v{s}, {ra}" if src_t == dtypes.float16 else self._bf16_to_f32(f"v{s}", ra), self._cvt_f32_to_int(rx, f"v{s}", dst_t in self.SIGNED_INTS)]
    if dst_t in (dtypes.float16, dtypes.bfloat16) and dtypes.is_int(src_t):
      s, src_reg, instrs = self.get_scratch_vgpr(), ra, []
      if src_t in self.SMALL_SIGNED_INTS: instrs.append(self._sign_extend_small_int(f"v{s}", ra, src_t)); src_reg = f"v{s}"
      instrs.append(self._cvt_int_to_f32(f"v{s}", src_reg, src_signed))
      instrs.append(f"v_cvt_f16_f32 {rx}, v{s}" if dst_t == dtypes.float16 else self._f32_to_bf16(rx, f"v{s}")); return instrs
    if dst_t == dtypes.float64 and src_t in (dtypes.float16, dtypes.bfloat16):
      s, dst_lo = self.get_scratch_vgpr(), self._get_dst_pair(rx)[0]
      return [f"v_cvt_f32_f16 v{s}, {ra}" if src_t == dtypes.float16 else self._bf16_to_f32(f"v{s}", ra), self._cvt_f32_to_f64(dst_lo, f"v{s}")]
    if dst_t in (dtypes.float16, dtypes.bfloat16) and src_t == dtypes.float64:
      s = self.get_scratch_vgpr()
      return [self._cvt_f64_to_f32(f"v{s}", get_reg_base(ra)), f"v_cvt_f16_f32 {rx}, v{s}" if dst_t == dtypes.float16 else self._f32_to_bf16(rx, f"v{s}")]
    if dtypes.is_int(dst_t) and src_t == dtypes.float64: return self._cvt_f64_to_int(rx, get_reg_base(ra), dst_t in self.SIGNED_INTS)
    if dst_t == dtypes.float64:
      dst_lo = self._get_dst_pair(rx)[0]
      if src_t in self.INT64_TYPES:
        src_lo, src_hi = self._get_src_pair(ra); s, signed = self.get_scratch_vgpr(), src_t == dtypes.long
        return [f"v_cvt_f64_u32 v[{s}:{s+1}], {src_lo}", f"v_cvt_f64_{'i' if signed else 'u'}32 v[{s+2}:{s+3}], {src_hi}",
                f"v_ldexp_f64 v[{s+2}:{s+3}], v[{s+2}:{s+3}], 32", f"v_add_f64 v[{dst_lo}:{dst_lo+1}], v[{s}:{s+1}], v[{s+2}:{s+3}]"]
      if src_t == dtypes.int32: return f"v_cvt_f64_i32 v[{dst_lo}:{dst_lo+1}], {ra}"
      if src_t in self.SMALL_SIGNED_INTS:
        s = self.get_scratch_vgpr(); return [self._sign_extend_small_int(f"v{s}", ra, src_t), f"v_cvt_f64_i32 v[{dst_lo}:{dst_lo+1}], v{s}"]
      return f"v_cvt_f64_u32 v[{dst_lo}:{dst_lo+1}], {ra}"
    if src_t in self.SMALL_SIGNED_INTS and dtypes.is_int(dst_t) and dst_t.itemsize > src_t.itemsize and dst_t not in self.INT64_TYPES:
      return self._sign_extend_small_int(rx, ra, src_t)
    return f"v_mov_b32 {rx}, {ra}"

  def render_bool_logic(self, x: UOp, a: UOp, b: UOp, op: str) -> str|list[str]:
    ra, rb, rx = self.r[a], self.r[b], self.r[x]; s_op, v_op = f"s_{op}_b32", f"v_{op}_b32"
    if ra.startswith('s') and rb.startswith('s'): return [f"{s_op} vcc_lo, {ra}, {rb}", f"v_cndmask_b32 {rx}, 0, 1, vcc_lo"]
    if ra.startswith('s'): return [f"v_cndmask_b32 v{self.get_scratch_vgpr()}, 0, 1, {ra}", f"{v_op} {rx}, v{self.get_scratch_vgpr()}, {rb}"]
    if rb.startswith('s'): return [f"v_cndmask_b32 v{self.get_scratch_vgpr()}, 0, 1, {rb}", f"{v_op} {rx}, {ra}, v{self.get_scratch_vgpr()}"]
    return f"{v_op} {rx}, {ra}, {rb}"

  def render_where(self, x: UOp, cond: UOp, true_val: UOp, false_val: UOp) -> str|list[str]:
    rc, rt, rf, rx = self.r[cond], self.r[true_val], self.r[false_val], self.r[x]
    is_cond_sgpr = rc.startswith('s') or rc == 'vcc_lo'
    is_true_sgpr_mask, is_false_sgpr_mask = rt.startswith('s') and true_val.dtype == dtypes.bool, rf.startswith('s') and false_val.dtype == dtypes.bool
    is_false_zero = rf == '0' or (false_val.op is Ops.CONST and false_val.arg == 0)
    def is_noninline_const(val: str, uop: UOp) -> bool:
      if not val.startswith('0x') and not (val.lstrip('-').isdigit() and not can_inline_const(int(val), uop.dtype)): return False
      return uop.op is Ops.CONST and not can_inline_const(uop.arg, uop.dtype)
    if is_cond_sgpr and is_true_sgpr_mask and is_false_zero: return [f"s_and_b32 vcc_lo, {rc}, {rt}", f"v_cndmask_b32 {rx}, 0, 1, vcc_lo"]
    result, scratch = [], None
    if is_true_sgpr_mask: scratch = self.get_scratch_vgpr(); result.append(f"v_cndmask_b32 v{scratch}, 0, 1, {rt}"); rt = f"v{scratch}"
    if is_false_sgpr_mask: scratch = scratch or self.get_scratch_vgpr(); idx = scratch + 1 if is_true_sgpr_mask else scratch; result.append(f"v_cndmask_b32 v{idx}, 0, 1, {rf}"); rf = f"v{idx}"
    if is_noninline_const(rt, true_val) and is_noninline_const(rf, false_val): scratch = scratch or self.get_scratch_vgpr(); result.append(f"v_mov_b32 v{scratch}, {rt}"); rt = f"v{scratch}"
    if is_cond_sgpr: result.append(f"v_cndmask_b32 {rx}, {rf}, {rt}, {rc}")
    else: result.extend([f"v_cmp_ne_u32 vcc_lo, {rc}, 0", f"v_cndmask_b32 {rx}, {rf}, {rt}, vcc_lo"])
    return result if len(result) > 1 else result[0]

  def render_mov_64(self, x: UOp, a: UOp) -> list[str]:
    dst, src = get_reg_base(self.r[x]), get_reg_base(self.r[a])
    return [f"v_mov_b32 v{dst}, v{src}", f"v_mov_b32 v{dst+1}, v{src+1}"]

  def render_cast_to_64(self, x: UOp, a: UOp, signed: bool = False) -> str|list[str]:
    rx, ra, src_t = self.r[x], self.r[a], a.dtype
    if src_t in self.INT64_TYPES: return self.render_mov_64(x, a)
    if '[' not in rx: return f"v_cndmask_b32 {rx}, 0, 1, {ra}" if ra.startswith('s') else f"v_mov_b32 {rx}, {extract_low_32(ra)}"
    dst_num, src_reg = get_reg_base(rx), extract_low_32(ra)
    if ra.startswith('s'): return [f"v_cndmask_b32 v{dst_num}, 0, 1, {ra}", f"v_mov_b32 v{dst_num+1}, 0"]
    if src_t in self.SMALL_SIGNED_INTS and signed: return [self._sign_extend_small_int(f"v{dst_num}", src_reg, src_t), f"v_ashrrev_i32 v{dst_num+1}, 31, v{dst_num}"]
    return [f"v_mov_b32 v{dst_num}, {src_reg}", f"v_ashrrev_i32 v{dst_num+1}, 31, {src_reg}" if signed else f"v_mov_b32 v{dst_num+1}, 0"]

  def render_kernel(self, kernel, function_name, bufs, v_cnt, s_cnt, uops) -> str:
    args: list[dict[str, str|int|bool]] = []
    for name, dtype in bufs:
      if name.startswith("data"):
        i = int(name[4:])
        args.append({'.address_space': 'global', '.name': f'buf_{i}', '.offset': i*8, '.size': 8, '.type_name': 'void*', '.value_kind': 'global_buffer'})
      else:
        args.append({'.name': name, '.offset': len(args)*8, '.size': 8, '.value_kind': 'by_value'})
    kernarg_size = (int(args[-1][".offset"]) + int(args[-1][".size"])) if args else 0
    args_yaml = ['  - ' + '\n    '.join(f'{k}: {repr(v) if isinstance(v, str) else str(v).lower() if isinstance(v, bool) else v}' for k, v in arg.items()) for arg in args]
    metadata_yaml = f"""amdhsa.kernels:
- .args:
{chr(10).join(args_yaml)}
  .group_segment_fixed_size: {self.lds_size}
  .kernarg_segment_align: 8
  .kernarg_segment_size: {kernarg_size}
  .language: OpenCL C
  .language_version:
  - 1
  - 2
  .max_flat_workgroup_size: 256
  .name: {function_name}
  .private_segment_fixed_size: 0
  .sgpr_count: {s_cnt}
  .sgpr_spill_count: 0
  .symbol: {function_name}.kd
  .uses_dynamic_stack: false
  .vgpr_count: {v_cnt}
  .vgpr_spill_count: 0
  .wavefront_size: 32
amdhsa.target: amdgcn-amd-amdhsa--{self.arch}
amdhsa.version:
- 1
- 2
"""
    return ".text\n" + f'.amdgcn_target "amdgcn-amd-amdhsa--{self.arch}"\n' + f".global {function_name}\n" + \
           f".type {function_name},@function\n" + ".p2align 8\n" + f"{function_name}:\n" + '\n'.join(kernel) + \
           f"\n.size {function_name}, .-{function_name}\n\n" + ".rodata\n" + ".p2align 6\n" + f".amdhsa_kernel {function_name}\n" + \
           f"  .amdhsa_group_segment_fixed_size {self.lds_size}\n" + f"  .amdhsa_kernarg_size {kernarg_size}\n" + \
           "  .amdhsa_user_sgpr_count 2\n" + f"  .amdhsa_next_free_vgpr {v_cnt}\n" + f"  .amdhsa_next_free_sgpr {s_cnt}\n" + \
           "  .amdhsa_wavefront_size32 1\n" + "  .amdhsa_user_sgpr_kernarg_segment_ptr 1\n" + "  .amdhsa_system_sgpr_workgroup_id_x 1\n" + \
           "  .amdhsa_system_sgpr_workgroup_id_y 1\n" + "  .amdhsa_system_sgpr_workgroup_id_z 1\n" + "  .amdhsa_system_vgpr_workitem_id 2\n" + \
           ".end_amdhsa_kernel\n\n" + ".amdgpu_metadata\n" + metadata_yaml + ".end_amdgpu_metadata"

# *** Pre-render analysis functions ***
from tinygrad.dtype import AddrSpace

def analyze_const_usage(uops: list[UOp]) -> tuple[set[UOp], set[UOp]]:
  """Analyze constants to find which can skip VGPR allocation."""
  from collections import defaultdict
  const_use_count: dict[UOp, int] = defaultdict(int)
  reg_index_const_uses: dict[UOp, int] = defaultdict(int)
  add_mul_const_uses: dict[UOp, int] = defaultdict(int)
  store_const_uses: set[UOp] = set()
  vectorize_const_uses: set[UOp] = set()
  for u in uops:
    for src in u.src:
      if src.op is Ops.CONST: const_use_count[src] += 1
    if u.op is Ops.INDEX and len(u.src) > 1:
      if isinstance(u.src[0].dtype, PtrDType) and u.src[0].dtype.addrspace == AddrSpace.REG and u.src[1].op is Ops.CONST:
        reg_index_const_uses[u.src[1]] += 1
    if u.op in {Ops.ADD, Ops.MUL}:
      for src in u.src:
        if src.op is Ops.CONST: add_mul_const_uses[src] += 1
    if u.op is Ops.STORE and len(u.src) >= 2 and u.src[1].op is Ops.CONST: store_const_uses.add(u.src[1])
    if u.op is Ops.VECTORIZE:
      for src in u.src:
        if src.op is Ops.CONST: vectorize_const_uses.add(src)
  skip_alloc_consts: set[UOp] = set()
  for c, uses in reg_index_const_uses.items():
    if uses == const_use_count[c]: skip_alloc_consts.add(c)
  for c, uses in add_mul_const_uses.items():
    if uses == const_use_count[c] and c not in store_const_uses and c not in vectorize_const_uses: skip_alloc_consts.add(c)
  return skip_alloc_consts, store_const_uses

def analyze_half16_vectorize(uops: list[UOp], regalloc) -> tuple[dict, dict, dict, dict, dict]:
  """Analyze half16 VECTORIZE ops for look-ahead packing optimization."""
  half16_vectorize_sources: dict[UOp, tuple[UOp, int]] = {}
  half16_vectorize_ranges: dict[UOp, str] = {}
  half16_packed: dict[UOp, set[int]] = {}
  half16_temp_regs: dict[UOp, str] = {}
  half16_direct_loads: dict[UOp, tuple[UOp, int]] = {}
  for u in uops:
    if u.op is Ops.VECTORIZE and u.dtype.scalar() == dtypes.half and u.dtype.count == 16:
      for pos, src in enumerate(u.src): half16_vectorize_sources[src] = (u, pos)
      half16_packed[u] = set()
      half16_vectorize_ranges[u] = regalloc.alloc_vgpr_range(u, 8)
      load_positions: dict[UOp, list[tuple[int, int]]] = {}
      for pos, src in enumerate(u.src):
        if src.op is Ops.GEP and src.src[0].op is Ops.LOAD:
          load = src.src[0]
          if hasattr(load.dtype, 'count') and load.dtype.count == 4 and load.dtype.scalar() == dtypes.half:
            gep_idx = src.arg[0] if isinstance(src.arg, tuple) else src.arg
            load_positions.setdefault(load, []).append((pos, gep_idx))
      for load, positions in load_positions.items():
        if len(positions) == 4:
          positions.sort(key=lambda x: x[0])
          if [g for _, g in positions] == [0, 1, 2, 3] and positions[0][0] % 4 == 0:
            base_vgpr_idx = positions[0][0] // 2
            half16_direct_loads[load] = (u, base_vgpr_idx)
            half16_packed[u].update([base_vgpr_idx, base_vgpr_idx + 1])
  return half16_vectorize_sources, half16_vectorize_ranges, half16_packed, half16_temp_regs, half16_direct_loads

def analyze_deferred_stores(uops: list[UOp], regalloc) -> tuple[set[UOp], set[UOp]]:
  """Analyze store-only indices and recomputable SHL ops for deferred address computation."""
  store_only_indices: set[UOp] = set()
  index_users: dict[UOp, list[UOp]] = {}
  for u in uops:
    if u.op is Ops.INDEX and len(u.src) >= 1:
      if isinstance(u.src[0].dtype, PtrDType) and u.src[0].dtype.addrspace == AddrSpace.GLOBAL: index_users[u] = []
    for src in u.src:
      if src in index_users: index_users[src].append(u)
  for idx_op, users in index_users.items():
    if users and all(u.op is Ops.STORE for u in users): store_only_indices.add(idx_op)
  recomputable_shls: set[UOp] = set()
  uop_users: dict[UOp, list[UOp]] = {}
  for u in uops:
    for src in u.src: uop_users.setdefault(src, []).append(u)
  recompute_base_uops: set[UOp] = set()
  for idx_op in store_only_indices:
    if len(idx_op.src) > 1:
      byte_offset = idx_op.src[1]
      if byte_offset.op is Ops.SHL and len(byte_offset.src) == 2 and byte_offset.src[0].op is Ops.ADD and byte_offset.src[1].op is Ops.CONST:
        add_op = byte_offset.src[0]
        if all(user in store_only_indices for user in uop_users.get(byte_offset, [])):
          recomputable_shls.add(byte_offset)
          if len(add_op.src) == 2 and (add_op.src[0].op is Ops.CONST or add_op.src[1].op is Ops.CONST):
            if all(user == byte_offset or user in store_only_indices for user in uop_users.get(add_op, [])): recomputable_shls.add(add_op)
            recompute_base_uops.add(add_op.src[1] if add_op.src[0].op is Ops.CONST else add_op.src[0])
          else:
            for src in add_op.src:
              if src.op is not Ops.CONST: recompute_base_uops.add(src)
  for base_uop in recompute_base_uops: regalloc.extend_lifetime(base_uop, len(uops) - 1)
  return store_only_indices, recomputable_shls

# *** Float64 MULACC (FMA) ***
def render_f64_mulacc(ctx, x, a, b, c):
  rx, s = get_reg_base(ctx.r[x]), ctx.get_scratch_vgpr()
  scratch_used = 0
  def get_f64_operand(uop):
    nonlocal scratch_used
    r = ctx.r[uop]
    if isinstance(r, str) and ('[' in r or r.startswith('v')): return f"v[{get_reg_base(r)}:{get_reg_base(r)+1}]", []
    bits = struct.pack('<d', float(uop.arg)); lo, hi = int.from_bytes(bits[0:4], 'little'), int.from_bytes(bits[4:8], 'little')
    base = s + scratch_used; scratch_used += 2
    return f"v[{base}:{base+1}]", [f"v_mov_b32 v{base}, 0x{lo:08x}", f"v_mov_b32 v{base+1}, 0x{hi:08x}"]
  a_reg, a_load = get_f64_operand(a); b_reg, b_load = get_f64_operand(b); c_reg, c_load = get_f64_operand(c)
  return a_load + b_load + c_load + [f"v_fma_f64 v[{rx}:{rx+1}], {a_reg}, {b_reg}, {c_reg}"]

_f64_cmp_ops = {Ops.CMPLT: "lt", Ops.CMPEQ: "eq", Ops.CMPNE: "neq"}
def render_f64_cmp(ctx, x, a, b):
  ra, rb, dest, cmp_op = get_reg_base(ctx.r[a]), get_reg_base(ctx.r[b]), ctx.r[x], _f64_cmp_ops[x.op]
  cmp_instr = f"v_cmp_{cmp_op}_f64 {dest if dest.startswith('s') else 'vcc_lo'}, v[{ra}:{ra+1}], v[{rb}:{rb+1}]"
  return [cmp_instr, f"v_cndmask_b32 {dest}, 0, 1, vcc_lo"] if dest.startswith('v') else cmp_instr

# *** Signed int8/int16 operations ***
def _get_bits(dtype): return 8 if dtype == dtypes.int8 else 16
_signed_small_int_ops = {Ops.MAX: "v_max_i32", Ops.MUL: "v_mul_lo_u32", Ops.SUB: "v_sub_nc_u32"}
def render_signed_small_int_binop(ctx, x, a, b):
  bits, dest, s, op_instr = _get_bits(x.dtype), ctx.r[x], ctx.get_scratch_vgpr(), _signed_small_int_ops[x.op]
  return [f"v_bfe_i32 v{s}, {ctx.r[a]}, 0, {bits}", f"v_bfe_i32 v{s+1}, {ctx.r[b]}, 0, {bits}", f"{op_instr} {dest}, v{s}, v{s+1}"]

# *** 64-bit integer operations ***
def render_64bit_mul(ctx, x):
  rx, a, b = ctx.r[x], x.src[0], x.src[1]
  a_is_signed_cast = a.op is Ops.CAST and a.src[0].dtype == dtypes.int32
  if a.op is Ops.CAST and a.src[0].dtype.itemsize == 4: ra, a_is_32bit = ctx.r[a.src[0]], True
  else: ra, a_is_32bit = ctx.r[a], a.op is Ops.CAST and a.src[0].dtype.itemsize == 4
  if b.op is Ops.CONST:
    b_val, b_lo_val, b_hi_val = b.arg, b.arg & 0xFFFFFFFF, (b.arg >> 32) & 0xFFFFFFFF
    rb_lo, rb_hi = render_val(b_lo_val, dtypes.uint32), render_val(b_hi_val, dtypes.uint32) if b_hi_val != 0 else None
    b_const_hi_bit_set, b_is_32bit = (b_val & 0x80000000) != 0, b_hi_val == 0
  else:
    rb = ctx.r[b]; rb_lo = f"v{get_reg_base(rb)}" if '[' in rb else rb; rb_hi = f"v{get_reg_base(rb) + 1}" if '[' in rb else None
    b_const_hi_bit_set, b_is_32bit = False, '[' not in rb
  ra_lo = f"v{get_reg_base(ra)}" if '[' in ra else ra; ra_hi = f"v{get_reg_base(ra) + 1}" if '[' in ra else None
  if '[' in rx:
    rx_lo, rx_hi = f"v{get_reg_base(rx)}", f"v{get_reg_base(rx) + 1}"
    if a_is_32bit or b_is_32bit:
      if x.dtype == dtypes.long and a_is_signed_cast and b.op is Ops.CONST and b_const_hi_bit_set:
        s = ctx.get_scratch_vgpr()
        return [f"v_mul_lo_u32 {rx_lo}, {ra_lo}, {rb_lo}", f"v_mul_hi_u32 {rx_hi}, {ra_lo}, {rb_lo}",
                f"v_cmp_lt_i32 vcc_lo, {ra_lo}, 0", f"v_cndmask_b32 v{s}, 0, {rb_lo}, vcc_lo", f"v_sub_nc_u32 {rx_hi}, {rx_hi}, v{s}"]
      return [f"v_mul_lo_u32 {rx_lo}, {ra_lo}, {rb_lo}", f"{'v_mul_hi_i32' if x.dtype == dtypes.long else 'v_mul_hi_u32'} {rx_hi}, {ra_lo}, {rb_lo}"]
    s = ctx.get_scratch_vgpr()
    instrs = [f"v_mul_lo_u32 {rx_lo}, {ra_lo}, {rb_lo}", f"v_mul_hi_u32 {rx_hi}, {ra_lo}, {rb_lo}"]
    if ra_hi: instrs.extend([f"v_mul_lo_u32 v{s}, {ra_hi}, {rb_lo}", f"v_add_nc_u32 {rx_hi}, {rx_hi}, v{s}"])
    if rb_hi: instrs.extend([f"v_mul_lo_u32 v{s}, {ra_lo}, {rb_hi}", f"v_add_nc_u32 {rx_hi}, {rx_hi}, v{s}"])
    return instrs
  return f"v_mul_lo_u32 {rx}, {ra_lo}, {rb_lo}"

def render_64bit_shr(ctx, x, a, b):
  if b.op is not Ops.CONST: raise RuntimeError("64-bit SHR requires constant shift amount")
  rx, ra, shift_amt = ctx.r[x], ctx.r[a], b.arg
  dst_lo, dst_hi, src_lo = get_reg_base(rx), get_reg_base(rx) + 1, get_reg_base(ra)
  src_hi = get_reg_base(ra) + 1 if '[' in ra else get_reg_base(ra)
  shift_instr = "v_ashrrev_i32" if x.dtype == dtypes.long else "v_lshrrev_b32"
  if shift_amt == 0: return [f"v_mov_b32 v{dst_lo}, v{src_lo}", f"v_mov_b32 v{dst_hi}, v{src_hi}"]
  if shift_amt >= 32:
    remaining = shift_amt - 32
    if x.dtype == dtypes.long:
      return [f"v_mov_b32 v{dst_lo}, v{src_hi}" if remaining == 0 else f"v_ashrrev_i32 v{dst_lo}, {remaining}, v{src_hi}", f"v_ashrrev_i32 v{dst_hi}, 31, v{src_hi}"]
    return [f"v_mov_b32 v{dst_lo}, v{src_hi}" if remaining == 0 else f"v_lshrrev_b32 v{dst_lo}, {remaining}, v{src_hi}", f"v_mov_b32 v{dst_hi}, 0"]
  s = ctx.get_scratch_vgpr()
  return [f"v_lshrrev_b32 v{dst_lo}, {shift_amt}, v{src_lo}", f"v_lshlrev_b32 v{s}, {32-shift_amt}, v{src_hi}",
          f"v_or_b32 v{dst_lo}, v{dst_lo}, v{s}", f"{shift_instr} v{dst_hi}, {shift_amt}, v{src_hi}"]

def render_64bit_shl(ctx, x, a, b):
  if b.op is not Ops.CONST: raise RuntimeError("64-bit SHL requires constant shift amount")
  rx, ra, shift_amt = ctx.r[x], ctx.r[a], b.arg
  dst_lo, dst_hi = get_reg_base(rx), get_reg_base(rx) + 1
  src_lo = get_reg_base(ra) if '[' in ra else int(ra[1:]); src_hi = src_lo + 1 if '[' in ra else src_lo
  if shift_amt == 0: return [f"v_mov_b32 v{dst_lo}, v{src_lo}", f"v_mov_b32 v{dst_hi}, v{src_hi}"]
  if shift_amt >= 32:
    remaining = shift_amt - 32
    return [f"v_mov_b32 v{dst_hi}, v{src_lo}" if remaining == 0 else f"v_lshlrev_b32 v{dst_hi}, {remaining}, v{src_lo}", f"v_mov_b32 v{dst_lo}, 0"]
  s = ctx.get_scratch_vgpr()
  return [f"v_lshlrev_b32 v{dst_lo}, {shift_amt}, v{src_lo}", f"v_lshlrev_b32 v{dst_hi}, {shift_amt}, v{src_hi}",
          f"v_lshrrev_b32 v{s}, {32-shift_amt}, v{src_lo}", f"v_or_b32 v{dst_hi}, v{dst_hi}, v{s}"]

def _get_64bit_lo_hi(r, uop=None):
  if '[' in r: base = get_reg_base(r); return f"v{base}", f"v{base+1}"
  if r.startswith('v') or r.startswith('s'): num = int(r[1:]); return f"v{num}", f"v{num}"
  return (r, "-1") if uop is not None and uop.op is Ops.CONST and isinstance(uop.arg, int) and uop.arg < 0 else (r, "0")

def render_64bit_add(ctx, x, a, b):
  dst_lo, dst_hi = get_reg_base(ctx.r[x]), get_reg_base(ctx.r[x]) + 1
  a_lo, a_hi = _get_64bit_lo_hi(ctx.r[a], a); b_lo, b_hi = _get_64bit_lo_hi(ctx.r[b], b)
  return [f"v_add_co_u32 v{dst_lo}, vcc_lo, {a_lo}, {b_lo}", f"v_add_co_ci_u32 v{dst_hi}, vcc_lo, {a_hi}, {b_hi}, vcc_lo"]

def render_64bit_sub(ctx, x, a, b):
  dst_lo, dst_hi = get_reg_base(ctx.r[x]), get_reg_base(ctx.r[x]) + 1
  a_lo, a_hi = _get_64bit_lo_hi(ctx.r[a], a); b_lo, b_hi = _get_64bit_lo_hi(ctx.r[b], b)
  return [f"v_sub_co_u32 v{dst_lo}, vcc_lo, {a_lo}, {b_lo}", f"v_sub_co_ci_u32 v{dst_hi}, vcc_lo, {a_hi}, {b_hi}, vcc_lo"]

_64bit_bitwise_ops = {Ops.OR: "or", Ops.XOR: "xor", Ops.AND: "and"}
def render_64bit_bitwise(ctx, x, a, b):
  dst_lo, dst_hi = get_reg_base(ctx.r[x]), get_reg_base(ctx.r[x]) + 1
  a_lo, a_hi = get_reg_base(ctx.r[a]), get_reg_base(ctx.r[a]) + 1; b_lo, b_hi = get_reg_base(ctx.r[b]), get_reg_base(ctx.r[b]) + 1
  op = _64bit_bitwise_ops[x.op]
  return [f"v_{op}_b32 v{dst_lo}, v{a_lo}, v{b_lo}", f"v_{op}_b32 v{dst_hi}, v{a_hi}, v{b_hi}"]

def render_64bit_max(ctx, x, a, b):
  dst_lo, dst_hi = get_reg_base(ctx.r[x]), get_reg_base(ctx.r[x]) + 1
  a_lo, a_hi = get_reg_base(ctx.r[a]), get_reg_base(ctx.r[a]) + 1; b_lo, b_hi = get_reg_base(ctx.r[b]), get_reg_base(ctx.r[b]) + 1
  hi_cmp = "i32" if x.dtype == dtypes.long else "u32"; ctx.scratch_sgpr_used = True
  ss0, ss1 = f"s{ctx.gated_sgpr}", f"s{ctx.gated_sgpr+1}"
  return [f"v_cmp_gt_{hi_cmp} vcc_lo, v{a_hi}, v{b_hi}", f"s_mov_b32 {ss0}, vcc_lo",
          f"v_cmp_eq_u32 vcc_lo, v{a_hi}, v{b_hi}", f"s_mov_b32 {ss1}, vcc_lo",
          f"v_cmp_gt_u32 vcc_lo, v{a_lo}, v{b_lo}", f"s_and_b32 {ss1}, {ss1}, vcc_lo", f"s_or_b32 vcc_lo, {ss0}, {ss1}",
          f"v_cndmask_b32 v{dst_lo}, v{b_lo}, v{a_lo}, vcc_lo", f"v_cndmask_b32 v{dst_hi}, v{b_hi}, v{a_hi}, vcc_lo"]

def render_where_64(ctx, x, cond, true_val, false_val):
  dst, ra_t, ra_f = ctx.r[x], ctx.r[true_val], ctx.r[false_val]
  dst_lo, dst_hi = get_reg_base(dst), get_reg_base(dst) + 1
  t_lo, t_hi = get_reg_base(ra_t), get_reg_base(ra_t) + 1; f_lo, f_hi = get_reg_base(ra_f), get_reg_base(ra_f) + 1
  cond_reg = ctx.r[cond]
  if cond_reg.startswith('s'): return [f"v_cndmask_b32 v{dst_lo}, v{f_lo}, v{t_lo}, {cond_reg}", f"v_cndmask_b32 v{dst_hi}, v{f_hi}, v{t_hi}, {cond_reg}"]
  return [f"v_cmp_ne_u32 vcc_lo, {cond_reg}, 0", f"v_cndmask_b32 v{dst_lo}, v{f_lo}, v{t_lo}, vcc_lo", f"v_cndmask_b32 v{dst_hi}, v{f_hi}, v{t_hi}, vcc_lo"]

# *** ASM op table ***
asm_for_op: dict[Ops, Callable] = {
  Ops.RECIPROCAL: lambda d,a,dt,name: None, Ops.EXP2: lambda d,a,dt,name: f"v_exp_{name} {d}, {a}",
  Ops.LOG2: lambda d,a,dt,name: f"v_log_{name} {d}, {a}", Ops.SIN: lambda d,a,dt,name: None,
  Ops.SQRT: lambda d,a,dt,name: f"v_sqrt_{name} {d}, {a}", Ops.TRUNC: lambda d,a,dt,name: f"v_trunc_{name} {d}, {a}",
  Ops.NEG: lambda d,a,dt,name: f"v_sub_{name} {d}, 0, {a}" if dtypes.is_float(dt) else f"v_sub_nc_u32 {d}, 0, {a}",
  Ops.SHR: lambda d,a,b,dt,name: f"v_ashrrev_i32 {d}, {b}, {a}" if dtypes.is_int(dt) and not dtypes.is_unsigned(dt) else f"v_lshrrev_b32 {d}, {b}, {a}",
  Ops.SHL: lambda d,a,b,dt,name: f"v_lshlrev_b32 {d}, {b}, {a}",
  Ops.ADD: lambda d,a,b,dt,name: f"v_add_{name} {d}, {a}, {b}" if dtypes.is_float(dt) else f"v_add_nc_u32 {d}, {a}, {b}",
  Ops.SUB: lambda d,a,b,dt,name: f"v_sub_{name} {d}, {a}, {b}" if dtypes.is_float(dt) else f"v_sub_nc_u32 {d}, {a}, {b}",
  Ops.MUL: lambda d,a,b,dt,name: f"v_mul_{name} {d}, {a}, {b}" if dtypes.is_float(dt) else f"v_mul_lo_u32 {d}, {a}, {b}",
  Ops.XOR: lambda d,a,b,dt,name: f"v_xor_b32 {d}, {a}, {b}", Ops.AND: lambda d,a,b,dt,name: f"v_and_b32 {d}, {a}, {b}",
  Ops.OR: lambda d,a,b,dt,name: f"v_or_b32 {d}, {a}, {b}", Ops.MAX: lambda d,a,b,dt,name: f"v_max_{name} {d}, {a}, {b}",
  Ops.MOD: lambda d,a,b,dt,name: f"v_mod_{name} {d}, {a}, {b}" if dtypes.is_float(dt) else None,
  Ops.CMPEQ: lambda d,a,b,dt,name: f"v_cmp_eq_{name} {d}, {a}, {b}",
  Ops.CMPLT: lambda d,a,b,dt,name: f"v_cmp_lt_{name} {d}, {a}, {b}",
  Ops.CMPNE: lambda d,a,b,dt,name: f"v_cmp_neq_{name} {d}, {a}, {b}" if dtypes.is_float(dt) else f"v_cmp_ne_{name} {d}, {a}, {b}",
  Ops.MULACC: lambda d,a,b,c,dt,name: f"v_fma_{name} {d}, {a}, {b}, {c}" if dtypes.is_float(dt) else f"v_mad_i32_i24 {d}, {a}, {b}, {c}",
  Ops.WHERE: lambda d,a,b,c,dt,name: f"v_cndmask_b32 {d}, {c}, {b}, {a}",
}

# *** Memory operations ***
def global_store(addr:str, data:str, base:str, dt:DType) -> str:
  return f"global_store_{({1: 'byte', 2: 'b16', 4: 'b32', 8: 'b64', 16: 'b128'}[dt.itemsize])} {addr}, {data}, {base}"

def global_load(dest:str, addr:str, base:str, dt:DType) -> str:
  if dt.itemsize == 1: sz = 'ubyte'
  elif dt.itemsize == 2: sz = 'u16'
  else: sz = {4: 'b32', 8: 'b64', 16: 'b128'}[dt.itemsize]
  return f"global_load_{sz} {dest}, {addr}, {base}"

def gated_load(ctx, x, idx, alt, gate, buf, index_op) -> list[str]:
  ctx.scratch_sgpr_used = True
  addr_reg, dest_reg, alt_reg = ctx.r[index_op], ctx.r[x], ctx.r[alt]
  gate_instr = f"v_cmp_ne_u32 vcc_lo, {ctx.r[gate]}, 0" if ctx.r[gate].startswith('v') else f"s_and_b32 vcc_lo, exec_lo, {ctx.r[gate]}"
  result = [gate_instr, f"s_mov_b32 s{ctx.gated_sgpr}, vcc_lo", f"v_cndmask_b32 {addr_reg}, 0, {addr_reg}, vcc_lo",
            global_load(dest_reg, addr_reg, ctx.r[buf], x.dtype), "s_waitcnt vmcnt(0)"]
  if '[' in dest_reg:
    base, end, alt_base = get_reg_base(dest_reg), int(dest_reg[dest_reg.index(':')+1:-1]), get_reg_base(alt_reg)
    result.extend(f"v_cndmask_b32 v{base+i}, v{alt_base+i}, v{base+i}, s{ctx.gated_sgpr}" for i in range(end - base + 1))
  else:
    result.append(f"v_cndmask_b32 {dest_reg}, {alt_reg}, {dest_reg}, s{ctx.gated_sgpr}")
  return result

def ds_read(dest:str, addr:str, dt:DType) -> str:
  if dt.itemsize == 1: sz = 'u8'
  elif dt.itemsize == 2: sz = 'u16'
  else: sz = {4: 'b32', 8: 'b64', 16: 'b128'}[dt.itemsize]
  return f"ds_read_{sz} {dest}, {addr}"

def ds_write(addr:str, data:str, dt:DType) -> str:
  return f"ds_write_{({1: 'b8', 2: 'b16', 4: 'b32', 8: 'b64', 16: 'b128'}[dt.itemsize])} {addr}, {data}"

# *** Render helpers ***
def render_define_var(ctx, x):
  if ctx.r[x].startswith('s'): return f"s_load_b32 {ctx.r[x]}, s[0:1], {ctx.kernarg_offset[x]}"
  ctx.scratch_sgpr_used = True
  return [f"s_load_b32 s{ctx.gated_sgpr}, s[0:1], {ctx.kernarg_offset[x]}", f"v_mov_b32 {ctx.r[x]}, s{ctx.gated_sgpr}"]

def render_const_64(ctx, x):
  reg_num = get_reg_base(ctx.r[x])
  bits = struct.unpack("Q", struct.pack("d", x.arg))[0] if x.dtype == dtypes.float64 else int(x.arg) & 0xFFFFFFFFFFFFFFFF
  return [f"v_mov_b32 v{reg_num}, 0x{bits & 0xFFFFFFFF:08X}", f"v_mov_b32 v{reg_num+1}, 0x{(bits >> 32) & 0xFFFFFFFF:08X}"]

def render_comparison(ctx, x, src0):
  dest, dtype, instrs = ctx.r[x], src0.dtype, []
  if dtype in (dtypes.int64, dtypes.uint64, dtypes.long, dtypes.ulong):
    regs = [ctx.r[v] for v in x.src]
    a_lo, a_hi = f"v{get_reg_base(regs[0])}", f"v{get_reg_base(regs[0])+1}"
    b_lo, b_hi = f"v{get_reg_base(regs[1])}", f"v{get_reg_base(regs[1])+1}"
    hi_suffix = "i32" if dtype in (dtypes.int64, dtypes.long) else "u32"
    ctx.scratch_sgpr_used = True; ss0, ss1 = f"s{ctx.gated_sgpr}", f"s{ctx.gated_sgpr+1}"
    if x.op is Ops.CMPEQ:
      instrs = [f"v_cmp_eq_u32 vcc_lo, {a_hi}, {b_hi}", f"s_mov_b32 {ss0}, vcc_lo", f"v_cmp_eq_u32 vcc_lo, {a_lo}, {b_lo}", f"s_and_b32 vcc_lo, {ss0}, vcc_lo"]
    elif x.op is Ops.CMPNE:
      instrs = [f"v_cmp_ne_u32 vcc_lo, {a_hi}, {b_hi}", f"s_mov_b32 {ss0}, vcc_lo", f"v_cmp_ne_u32 vcc_lo, {a_lo}, {b_lo}", f"s_or_b32 vcc_lo, {ss0}, vcc_lo"]
    else:
      instrs = [f"v_cmp_lt_{hi_suffix} vcc_lo, {a_hi}, {b_hi}", f"s_mov_b32 {ss0}, vcc_lo", f"v_cmp_eq_u32 vcc_lo, {a_hi}, {b_hi}",
                f"s_mov_b32 {ss1}, vcc_lo", f"v_cmp_lt_u32 vcc_lo, {a_lo}, {b_lo}", f"s_and_b32 {ss1}, {ss1}, vcc_lo", f"s_or_b32 vcc_lo, {ss0}, {ss1}"]
    instrs.append(f"v_cndmask_b32 {dest}, 0, 1, vcc_lo" if dest.startswith('v') else f"s_cselect_b32 {dest}, 1, 0")
    return instrs
  if dtype in (dtypes.int8, dtypes.int16):
    s, bits = ctx.get_scratch_vgpr(), 8 if dtype == dtypes.int8 else 16
    srcs = [f"v{s+i}" for i, v in enumerate(x.src)]
    instrs = [f"v_bfe_i32 {srcs[i]}, {ctx.r[v]}, 0, {bits}" for i, v in enumerate(x.src)]
  else:
    srcs = [extract_low_32(ctx.r[v]) for v in x.src]
  instrs.append(ctx.code_for_op[x.op]("vcc_lo" if dest.startswith('v') else dest, *srcs, dtype, ctx.types[dtype]))
  if dest.startswith('v'): instrs.append(f"v_cndmask_b32 {dest}, 0, 1, vcc_lo")
  return instrs

def render_sin(ctx, x, a):
  dest, src, s = ctx.r[x], ctx.r[a], ctx.get_scratch_vgpr()
  return [f"v_mul_f32 {dest}, 0x3E22F983, {src}", f"v_fma_f32 v{s}, {src}, 0x3E22F983, -{dest}",
          f"v_fract_f32 {dest}, {dest}", f"v_add_f32 {dest}, {dest}, v{s}",
          f"v_fma_f32 {dest}, {src}, 0x31DC9C88, {dest}", f"v_fract_f32 {dest}, {dest}", f"v_sin_f32 {dest}, {dest}"]

def render_recip(ctx, x, a):
  dest, src, s = ctx.r[x], ctx.r[a], ctx.get_scratch_vgpr()
  return [f"v_rcp_f32 v{s+1}, {src}", f"v_mov_b32 {dest}, v{s+1}",
          f"v_fma_f32 v{s}, -{src}, {dest}, 1.0", f"v_fma_f32 {dest}, {dest}, v{s}, {dest}",
          f"v_fma_f32 v{s}, -{src}, {dest}, 1.0", f"v_fma_f32 {dest}, {dest}, v{s}, {dest}",
          f"v_cmp_class_f32 vcc_lo, {dest}, 0x3", f"v_cndmask_b32 {dest}, {dest}, v{s+1}, vcc_lo"]

def render_recip_f64(ctx, x, a):
  dest, src, s = ctx.r[x], ctx.r[a], ctx.get_scratch_vgpr()
  dst_lo, dst_hi, src_lo = get_reg_base(dest), get_reg_base(dest) + 1, get_reg_base(src)
  return [f"v_rcp_f64 v[{s}:{s+1}], v[{src_lo}:{src_lo+1}]",
          f"v_fma_f64 v[{s+2}:{s+3}], -v[{src_lo}:{src_lo+1}], v[{s}:{s+1}], 2.0",
          f"v_mul_f64 v[{s}:{s+1}], v[{s}:{s+1}], v[{s+2}:{s+3}]",
          f"v_fma_f64 v[{s+2}:{s+3}], -v[{src_lo}:{src_lo+1}], v[{s}:{s+1}], 2.0",
          f"v_mul_f64 v[{dst_lo}:{dst_hi}], v[{s}:{s+1}], v[{s+2}:{s+3}]"]

def render_if(ctx, x):
  ctx.scratch_sgpr_used = True
  save_sgpr = ctx.if_sgpr_base + len(ctx.if_save_stack); ctx.if_save_stack.append(save_sgpr)
  ctx.max_if_depth = max(ctx.max_if_depth, len(ctx.if_save_stack)); cond_reg = ctx.r[x.src[0]]
  instrs = [f"v_cmp_ne_u32 vcc_lo, {cond_reg}, 0", "s_and_b32 vcc_lo, exec_lo, vcc_lo"] if cond_reg.startswith('v') else [f"s_and_b32 vcc_lo, exec_lo, {cond_reg}"]
  return instrs + [f"s_and_saveexec_b32 s{save_sgpr}, vcc_lo", f"s_cbranch_execz IF_END_{ctx.uops.index(x)}"]

def render_endif(ctx, x): return [f"IF_END_{ctx.uops.index(x.src[0])}:", f"s_mov_b32 exec_lo, s{ctx.if_save_stack.pop()}"]

def render_wmma(ctx, x):
  dtype_in, dtype_out = x.arg[2], x.arg[3]
  def to_range(r): return f"v[{min(n:=[int(s[1:]) for s in r])}:{max(n)}]" if isinstance(r, list) else r
  ra, rb, rc, rd = [to_range(ctx.r[s]) for s in x.src] + [to_range(ctx.r[x])]
  if dtype_out == dtypes.float: instr = f"v_wmma_f32_16x16x16_{'f16' if dtype_in == dtypes.half else 'bf16'}"
  elif dtype_out == dtypes.half: instr = "v_wmma_f16_16x16x16_f16"
  else: raise RuntimeError(f"Unsupported WMMA dtype_out: {dtype_out}")
  return f"{instr} {rd}, {ra}, {rb}, {rc}"

def render_add_with_literal(ctx, x, a, b):
  if x.dtype in (dtypes.long, dtypes.ulong, dtypes.float64): return None
  ra = ctx.r[a]
  if not isinstance(ra, str) or not ra.startswith('v'): return None
  const_val = render_val(b.arg, b.dtype)
  return f"v_add_{ctx.types[x.dtype]} {ctx.r[x]}, {const_val}, {ra}" if dtypes.is_float(x.dtype) else f"v_add_nc_u32 {ctx.r[x]}, {const_val}, {ra}"

def render_mul_with_literal(ctx, x, a, b):
  if x.dtype in (dtypes.long, dtypes.ulong): return None
  ra = ctx.r[a]
  if not isinstance(ra, str) or not ra.startswith('v'): return None
  const_val = render_val(b.arg, b.dtype)
  return f"v_mul_{ctx.types[x.dtype]} {ctx.r[x]}, {const_val}, {ra}" if dtypes.is_float(x.dtype) else f"v_mul_lo_u32 {ctx.r[x]}, {const_val}, {ra}"

# *** string_rewrite PatternMatcher ***
string_rewrite = PatternMatcher([
  (UPat(Ops.WMMA, name="x"), render_wmma),
  (UPat(Ops.ADD, name="x", src=(UPat.var("a"), UPat.cvar("b"))), render_add_with_literal),
  (UPat(Ops.ADD, name="x", src=(UPat.cvar("b"), UPat.var("a"))), render_add_with_literal),
  (UPat(Ops.MUL, name="x", src=(UPat.var("a"), UPat.cvar("b"))), render_mul_with_literal),
  (UPat(Ops.MUL, name="x", src=(UPat.cvar("b"), UPat.var("a"))), render_mul_with_literal),
  (UPat.cvar("x", dtypes.bool), lambda ctx, x: f"v_mov_b32 {ctx.r[x]}, {1 if x.arg else 0}"),
  (UPat.cvar("x", dtypes.float64), render_const_64),
  (UPat.cvar("x", (dtypes.long, dtypes.ulong)), lambda ctx, x: (render_const_64(ctx, x) if '[' in ctx.r[x]
    else f"v_mov_b32 {ctx.r[x]}, {render_val(x.arg, dtypes.int32 if x.dtype == dtypes.long else dtypes.uint32)}")),
  (UPat.cvar("x"), lambda ctx, x: f"v_mov_b32 {ctx.r[x]}, {render_val(x.arg, x.dtype)}"),
  (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: ctx.render_special(x)),
  (UPat(Ops.DEFINE_GLOBAL, name="x"), lambda ctx, x: f"s_load_b64 {ctx.r[x]}, s[0:1], {ctx.kernarg_offset[x]}"),
  (UPat(Ops.DEFINE_VAR, name="x"), render_define_var),
  (UPat(Ops.CMPNE, name="x", src=(UPat(dtype=dtypes.bool, name="a"), UPat.cvar("b"))),
   lambda ctx, x, a, b: f"s_not_b32 {ctx.r[x]}, {ctx.r[a]}" if b.arg == 1 and ctx.r[a].startswith('s') else None),
  (UPat((Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE), name="x", src=(UPat.var("a", dtype=dtypes.float64), UPat.var("b"))), render_f64_cmp),
  (UPat((Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ), name="x", allow_any_len=True, src=(UPat.var("src0"),)), render_comparison),
  (UPat(Ops.WHERE, name="x", dtype=(dtypes.float64, dtypes.long, dtypes.ulong),
    src=(UPat.var("cond"), UPat.var("true_val"), UPat.var("false_val"))), render_where_64),
  (UPat(Ops.WHERE, name="x", src=(UPat.var("cond"), UPat.var("true_val"), UPat.var("false_val"))),
   lambda ctx, x, cond, true_val, false_val: ctx.render_where(x, cond, true_val, false_val)),
  (UPat(Ops.RECIPROCAL, name="x", dtype=dtypes.float64, src=(UPat.var("a"),)), render_recip_f64),
  (UPat(Ops.RECIPROCAL, name="x", src=(UPat.var("a"),)), render_recip),
  (UPat(Ops.SIN, name="x", src=(UPat.var("a"),)), render_sin),
  (UPat((Ops.AND, Ops.OR, Ops.XOR), name="x", dtype=dtypes.bool, src=(UPat.var("a", dtype=dtypes.bool), UPat.var("b", dtype=dtypes.bool))),
   lambda ctx, x, a, b: ctx.render_bool_logic(x, a, b, {Ops.AND: "and", Ops.OR: "or", Ops.XOR: "xor"}[x.op])),
  (UPat((Ops.MUL, Ops.SUB), name="x", dtype=(dtypes.int8, dtypes.int16), src=(UPat.var("a"), UPat.var("b"))), render_signed_small_int_binop),
  (UPat(Ops.MUL, name="x", dtype=(dtypes.long, dtypes.ulong)), render_64bit_mul),
  (UPat(Ops.SHR, name="x", dtype=(dtypes.long, dtypes.ulong), src=(UPat.var("a"), UPat.cvar("b"))), render_64bit_shr),
  (UPat(Ops.SHL, name="x", dtype=(dtypes.long, dtypes.ulong), src=(UPat.var("a"), UPat.cvar("b"))), render_64bit_shl),
  (UPat(Ops.ADD, name="x", dtype=(dtypes.long, dtypes.ulong), src=(UPat.var("a"), UPat.var("b"))), render_64bit_add),
  (UPat(Ops.SUB, name="x", dtype=(dtypes.long, dtypes.ulong), src=(UPat.var("a"), UPat.var("b"))), render_64bit_sub),
  (UPat((Ops.OR, Ops.XOR, Ops.AND), name="x", dtype=(dtypes.long, dtypes.ulong), src=(UPat.var("a"), UPat.var("b"))), render_64bit_bitwise),
  (UPat(Ops.MULACC, name="x", dtype=dtypes.float64, src=(UPat.var("a"), UPat.var("b"), UPat.var("c"))), render_f64_mulacc),
  (UPat(Ops.MAX, name="x", dtype=(dtypes.int8, dtypes.int16), src=(UPat.var("a"), UPat.var("b"))), render_signed_small_int_binop),
  (UPat(Ops.MAX, name="x", dtype=(dtypes.long, dtypes.ulong), src=(UPat.var("a"), UPat.var("b"))), render_64bit_max),
  (UPat(GroupOp.ALU, name="x"), lambda ctx, x: ctx.code_for_op[x.op](
    get_32bit_reg(ctx, x), *[get_32bit_reg(ctx, v) for v in x.src], x.dtype, ctx.types[x.dtype])),
  (UPat(Ops.BITCAST, name="x", dtype=(dtypes.float64, dtypes.long, dtypes.ulong), src=(UPat.var("a"),), allow_any_len=True),
   lambda ctx, x, a: ctx.render_mov_64(x, a)),
  (UPat(Ops.BITCAST, name="x", src=(UPat.var("a", dtype=(dtypes.float64, dtypes.long, dtypes.ulong)),), allow_any_len=True),
   lambda ctx, x, a: ctx.render_mov_64(x, a)),
  (UPat(Ops.BITCAST, name="x", src=(UPat.var("a"),), allow_any_len=True),
   lambda ctx, x, a: (ctx.render_mov_64(x, a) if isinstance(ctx.r[x], str) and ctx.r[x].startswith('v[') else
                      f"v_mov_b32 {ctx.r[x]}, v{get_reg_base(ctx.r[a])}" if isinstance(ctx.r[a], str) and ctx.r[a].startswith('v[') else
                      f"v_mov_b32 {ctx.r[x]}, {ctx.r[a]}")),
  (UPat(Ops.CAST, name="x", dtype=dtypes.float16, src=(UPat(dtype=dtypes.bool, name="a"),)),
   lambda ctx, x, a: (lambda s=ctx.get_scratch_vgpr(): [f"v_cndmask_b32 v{s}, 0, 1, {ctx.r[a]}", f"v_cvt_f32_u32 v{s}, v{s}", f"v_cvt_f16_f32 {ctx.r[x]}, v{s}"]
     if ctx.r[a].startswith('s') else [f"v_cvt_f32_u32 v{s}, {ctx.r[a]}", f"v_cvt_f16_f32 {ctx.r[x]}, v{s}"])()),
  (UPat(Ops.CAST, name="x", dtype=dtypes.bfloat16, src=(UPat(dtype=dtypes.bool, name="a"),)),
   lambda ctx, x, a: (lambda s=ctx.get_scratch_vgpr(): [f"v_cndmask_b32 v{s}, 0, 1, {ctx.r[a]}", f"v_cvt_f32_u32 v{s}, v{s}", f"v_lshrrev_b32 {ctx.r[x]}, 16, v{s}"]
     if ctx.r[a].startswith('s') else [f"v_cvt_f32_u32 v{s}, {ctx.r[a]}", f"v_lshrrev_b32 {ctx.r[x]}, 16, v{s}"])()),
  (UPat(Ops.CAST, name="x", src=(UPat(dtype=dtypes.bool, name="a"),)), lambda ctx, x, a: ctx.render_cast_from_bool(x, a)),
  (UPat(Ops.CAST, name="x", dtype=dtypes.bool, src=(UPat.var("a"),)), lambda ctx, x, a: ctx.render_cast_to_bool(x, a)),
  (UPat(Ops.CAST, name="x", dtype=(dtypes.long, dtypes.ulong), src=(UPat.var("a", dtypes.sints),)),
   lambda ctx, x, a: ctx.render_cast_to_64(x, a, signed=True)),
  (UPat(Ops.CAST, name="x", dtype=(dtypes.long, dtypes.ulong), src=(UPat.var("a", (*dtypes.uints, dtypes.bool)),)),
   lambda ctx, x, a: ctx.render_cast_to_64(x, a, signed=False)),
  (UPat(Ops.CAST, name="x", src=(UPat.var("a"),)), lambda ctx, x, a: ctx.render_cast(x, a)),
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_GLOBAL, name="buf"), UPat.var("idx")), name="index_op", allow_any_len=True), UPat.var("var", dtype=dtypes.bool))),
   lambda ctx, idx, var, buf, index_op: [f"v_cndmask_b32 v{ctx.get_scratch_vgpr()}, 0, 1, {ctx.r[var]}", f"global_store_byte {ctx.r[index_op]}, v{ctx.get_scratch_vgpr()}, {ctx.r[buf]}"]
     if ctx.r[var].startswith('s') else f"global_store_byte {ctx.r[index_op]}, {ctx.r[var]}, {ctx.r[buf]}"),
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_GLOBAL, name="buf"), UPat.var("idx")), name="index_op", allow_any_len=True), UPat.var("var"))),
   lambda ctx, idx, var, buf, index_op: global_store(ctx.r[index_op], ctx.r[var], ctx.r[buf], var.dtype)),
  (UPat(Ops.LOAD, name="x", src=(UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_GLOBAL, name="buf"), UPat.var("idx"), UPat.var("gate")), name="index_op"), UPat.var("alt")), allow_any_len=True),
    lambda ctx, x, idx, alt, gate, buf, index_op: gated_load(ctx, x, idx, alt, gate, buf, index_op)),
  (UPat(Ops.LOAD, name="x", src=(UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_GLOBAL, name="buf"), UPat.var("idx")), name="index_op"),), allow_any_len=True),
    lambda ctx, x, idx, buf, index_op: global_load(ctx.r[x], ctx.r[index_op], ctx.r[buf], x.dtype)),
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_LOCAL), UPat.var("idx")), name="index_op", allow_any_len=True), UPat.var("var"))),
   lambda ctx, idx, var, index_op: ds_write(ctx.r[index_op], ctx.r[var], var.dtype)),
  (UPat(Ops.LOAD, name="x", src=(UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_LOCAL), UPat.var("idx")), name="index_op"),), allow_any_len=True),
    lambda ctx, x, idx, index_op: ds_read(ctx.r[x], ctx.r[index_op], x.dtype)),
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat(Ops.AFTER, src=(UPat(Ops.DEFINE_LOCAL),), allow_any_len=True), UPat.var("idx")), name="index_op", allow_any_len=True), UPat.var("var"))),
   lambda ctx, idx, var, index_op: ds_write(ctx.r[index_op], ctx.r[var], var.dtype)),
  (UPat(Ops.LOAD, name="x", src=(UPat(Ops.INDEX, src=(UPat(Ops.AFTER, src=(UPat(Ops.DEFINE_LOCAL),), allow_any_len=True), UPat.var("idx")), name="index_op"),), allow_any_len=True),
    lambda ctx, x, idx, index_op: ds_read(ctx.r[x], ctx.r[index_op], x.dtype)),
  (UPat(Ops.DEFINE_REG, src=()), lambda ctx: []),
  (UPat(Ops.RANGE, name="r"), lambda ctx, r: [f"v_mov_b32 {ctx.r[r]}, -1", f"s_branch LOOP_END_{ctx.uops.index(r)}", f"LOOP_{ctx.uops.index(r)}:"]),
  (UPat(Ops.END, name="x", src=(UPat(), UPat(Ops.RANGE, name="r"))), lambda ctx, x, r: [
    f"LOOP_END_{ctx.uops.index(r)}:", f"v_add_nc_u32 {ctx.r[r]}, {ctx.r[r]}, 1",
    f"v_cmp_lt_i32 vcc_lo, {ctx.r[r]}, {ctx.r[r.src[0]]}", f"s_cbranch_vccnz LOOP_{ctx.uops.index(r)}"]),
  (UPat(Ops.DEFINE_LOCAL, name="x"), lambda ctx, x: []),
  (UPat(Ops.IF, name="x"), render_if),
  (UPat(Ops.ENDIF, name="x"), render_endif),
  (UPat(Ops.BARRIER, name="x"), lambda ctx, x: "s_barrier"),
])
