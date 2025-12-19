from typing import Callable, cast
import struct, yaml
from collections import defaultdict
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat, GroupOp
from tinygrad.dtype import dtypes, DType, PtrDType, AddrSpace
from tinygrad.renderer import Renderer
from tinygrad.helpers import prod, get_single_element, getenv
from tinygrad.codegen.late.devectorizer import no_vectorized_alu
from tinygrad.codegen.opt import tc

def get_reg_base(reg: str) -> int:
  """Extract base register number from register string (e.g., 'v5' -> 5, 'v[10:17]' -> 10)."""
  return int(reg[2:reg.index(':')]) if '[' in reg else int(reg[1:])

def render_val(x, dtype):
  if dtypes.is_float(dtype):
    fval = float(x)
    # For half precision, always use hex format since literals are float32
    if dtype == dtypes.half: return "0x%04X" % struct.unpack("H", struct.pack("e", x))[0]
    # For float32, use inline literals for common values (RDNA3 supports these as immediate operands)
    if fval in (0.0, 0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0, -4.0): return str(fval) if fval != 0.0 else "0"
    if dtype == dtypes.double: return "0x%016X" % struct.unpack("Q", struct.pack("d", x))[0]
    return "0x%08X" % struct.unpack("I", struct.pack("f", x))[0]
  val = int(x) & 0xFFFFFFFF
  return f"0x{val:08X}" if val > 0x7FFFFFFF else str(val)

def can_inline_const(val, dtype) -> bool:
  if dtypes.is_float(dtype): return float(val) in (0.0, 0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0, -4.0)
  try: return -16 <= int(val) <= 64
  except (TypeError, ValueError): return False

# RDNA3 uses different instruction names and formats than PTX
# NOTE: These are used via string_rewrite which passes r[v] (register strings), not UOps
# For literal constant embedding, we handle ADD/MUL specially in string_rewrite
asm_for_op: dict[Ops, Callable] = {
  # v_rcp_f32 is approximate - handled specially in string_rewrite's render_recip with Newton-Raphson refinement
  Ops.RECIPROCAL: lambda d,a,dt,name: None,  # placeholder - actual render is in string_rewrite
  Ops.EXP2: lambda d,a,dt,name: f"v_exp_{name} {d}, {a}", Ops.LOG2: lambda d,a,dt,name: f"v_log_{name} {d}, {a}",
  # v_sin_f32 expects input in turns (x / 2π) - handled specially in string_rewrite's render_sin
  Ops.SIN: lambda d,a,dt,name: None,  # placeholder - actual render is in string_rewrite
  Ops.SQRT: lambda d,a,dt,name: f"v_sqrt_{name} {d}, {a}",
  Ops.TRUNC: lambda d,a,dt,name: f"v_trunc_{name} {d}, {a}",
  Ops.NEG: lambda d,a,dt,name: f"v_sub_{name} {d}, 0, {a}" if dtypes.is_float(dt) else f"v_sub_nc_u32 {d}, 0, {a}",
  # SHR: Use arithmetic shift for signed types, logical shift for unsigned
  Ops.SHR: lambda d,a,b,dt,name: f"v_ashrrev_i32 {d}, {b}, {a}" if dtypes.is_int(dt) and not dtypes.is_unsigned(dt) else f"v_lshrrev_b32 {d}, {b}, {a}",
  Ops.SHL: lambda d,a,b,dt,name: f"v_lshlrev_b32 {d}, {b}, {a}",  # Note: operand order is reversed
  Ops.ADD: lambda d,a,b,dt,name: f"v_add_{name} {d}, {a}, {b}" if dtypes.is_float(dt) else f"v_add_nc_u32 {d}, {a}, {b}",
  Ops.SUB: lambda d,a,b,dt,name: f"v_sub_{name} {d}, {a}, {b}" if dtypes.is_float(dt) else f"v_sub_nc_u32 {d}, {a}, {b}",
  Ops.MUL: lambda d,a,b,dt,name: f"v_mul_{name} {d}, {a}, {b}" if dtypes.is_float(dt) else f"v_mul_lo_u32 {d}, {a}, {b}",
  Ops.XOR: lambda d,a,b,dt,name: f"v_xor_b32 {d}, {a}, {b}",
  Ops.AND: lambda d,a,b,dt,name: f"v_and_b32 {d}, {a}, {b}",
  Ops.OR: lambda d,a,b,dt,name: f"v_or_b32 {d}, {a}, {b}",
  Ops.MAX: lambda d,a,b,dt,name: f"v_max_{name} {d}, {a}, {b}",
  Ops.MOD: lambda d,a,b,dt,name: f"v_mod_{name} {d}, {a}, {b}" if dtypes.is_float(dt) else None,  # int mod handled separately
  Ops.CMPEQ: lambda d,a,b,dt,name: f"v_cmp_eq_{name} {d}, {a}, {b}",
  Ops.CMPLT: lambda d,a,b,dt,name: f"v_cmp_lt_{name} {d}, {a}, {b}",
  # RDNA uses v_cmp_ne for integers, v_cmp_neq for floats
  Ops.CMPNE: lambda d,a,b,dt,name: f"v_cmp_neq_{name} {d}, {a}, {b}" if dtypes.is_float(dt) else f"v_cmp_ne_{name} {d}, {a}, {b}",
  # v_mad_u32_u24 only handles 24-bit unsigned operands; for signed ints, use v_mad_i32_i24
  Ops.MULACC: lambda d,a,b,c,dt,name: f"v_fma_{name} {d}, {a}, {b}, {c}" if dtypes.is_float(dt) else f"v_mad_i32_i24 {d}, {a}, {b}, {c}",
  Ops.WHERE: lambda d,a,b,c,dt,name: f"v_cndmask_b32 {d}, {c}, {b}, {a}",
}

# Pattern matcher for RDNA3-specific rewrites
# NOTE: By the time rdna_matcher runs, gated loads have already been created by devectorize
# (WHERE+LOAD → LOAD(INDEX(buf, idx, gate), alt)). We don't need to do that transformation here.
rdna_matcher = PatternMatcher([
  # cast void does nothing
  (UPat(Ops.CAST, name="x"), lambda x: x.src[0] if isinstance(x.dtype, PtrDType) or x.src[0].dtype == dtypes.void else None),
  # devectorize ALU operations - RDNA doesn't have vector float ALU
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST), name="alu"), no_vectorized_alu),
  # compute byte offset for INDEX operations at UOp level (like PTX)
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx")), name="op", allow_any_len=True), lambda buf, idx, op:
    UOp(Ops.INDEX, dtype=dtypes.int32, src=(buf, idx.cast(dtypes.int32)*buf.dtype.itemsize)+op.src[2:])
      if op.dtype != dtypes.int32 and isinstance(buf.dtype, PtrDType) and buf.dtype.addrspace != AddrSpace.REG else None),
])

def global_store(addr:str, data:str, base:str, dt:DType) -> str:
  sz = {1: 'byte', 2: 'b16', 4: 'b32', 8: 'b64', 16: 'b128'}[dt.itemsize]
  return f"global_store_{sz} {addr}, {data}, {base}"

def global_load(dest:str, addr:str, base:str, dt:DType) -> str:
  sz = {1: 'ubyte', 2: 'u16', 4: 'b32', 8: 'b64', 16: 'b128'}[dt.itemsize]
  return f"global_load_{sz} {dest}, {addr}, {base}"

def gated_load(ctx, x, idx, alt, gate, buf, index_op) -> list[str]:
  """Generate gated load using v_cndmask for address selection.
  Instead of exec masking (which can still fault on invalid addresses in masked lanes),
  we use v_cndmask to select a safe address (0) when gate is false, then unconditionally
  load, and finally select between loaded value and alt based on gate."""
  ctx.scratch_sgpr_used = True
  result = []
  # Get gate comparison result in vcc_lo, save to scratch_sgpr (vcc_lo may be clobbered)
  if ctx.r[gate].startswith('v'):
    result.append(f"v_cmp_ne_u32 vcc_lo, {ctx.r[gate]}, 0")
  else:
    result.append(f"s_and_b32 vcc_lo, exec_lo, {ctx.r[gate]}")
  result.append(f"s_mov_b32 s{ctx.gated_sgpr}, vcc_lo")  # save mask
  # Select address: use computed address if gate is true, else use 0 (safe address)
  addr_reg = ctx.r[index_op]
  result.append(f"v_cndmask_b32 {addr_reg}, 0, {addr_reg}, vcc_lo")
  result.append(global_load(ctx.r[x], addr_reg, ctx.r[buf], x.dtype))
  result.append("s_waitcnt vmcnt(0)")
  dest_reg, alt_reg = ctx.r[x], ctx.r[alt]
  if '[' in dest_reg:
    base, end = get_reg_base(dest_reg), int(dest_reg[dest_reg.index(':')+1:-1])
    alt_base = get_reg_base(alt_reg)
    for i in range(end - base + 1):
      result.append(f"v_cndmask_b32 v{base+i}, v{alt_base+i}, v{base+i}, s{ctx.gated_sgpr}")
  else:
    result.append(f"v_cndmask_b32 {dest_reg}, {alt_reg}, {dest_reg}, s{ctx.gated_sgpr}")
  return result

def ds_read(dest:str, addr:str, dt:DType) -> str:
  sz = {1: 'u8', 2: 'u16', 4: 'b32', 8: 'b64', 16: 'b128'}[dt.itemsize]
  return f"ds_read_{sz} {dest}, {addr}"

def ds_write(addr:str, data:str, dt:DType) -> str:
  sz = {1: 'b8', 2: 'b16', 4: 'b32', 8: 'b64', 16: 'b128'}[dt.itemsize]
  return f"ds_write_{sz} {addr}, {data}"

def render_define_var(ctx, x):
  """Render DEFINE_VAR - load from kernarg buffer. Uses SGPR if available, else VGPR via scratch."""
  if ctx.r[x].startswith('s'):
    return f"s_load_b32 {ctx.r[x]}, s[0:1], {ctx.kernarg_offset[x]}"
  # VGPR fallback - use gated_sgpr for the load, then move to VGPR
  ctx.scratch_sgpr_used = True
  return [f"s_load_b32 s{ctx.gated_sgpr}, s[0:1], {ctx.kernarg_offset[x]}",
          f"v_mov_b32 {ctx.r[x]}, s{ctx.gated_sgpr}"]

def render_const_64(ctx, x):
  """Render 64-bit constant as two v_mov_b32 instructions"""
  reg_num = get_reg_base(ctx.r[x])
  bits = struct.unpack("Q", struct.pack("d", x.arg))[0] if x.dtype == dtypes.float64 else int(x.arg) & 0xFFFFFFFFFFFFFFFF
  return [f"v_mov_b32 v{reg_num}, 0x{bits & 0xFFFFFFFF:08X}", f"v_mov_b32 v{reg_num+1}, 0x{(bits >> 32) & 0xFFFFFFFF:08X}"]

def extract_low_32(reg: str) -> str:
  """Extract low 32 bits from a register (handles 64-bit register pairs v[n:m])"""
  return f"v{get_reg_base(reg)}" if isinstance(reg, str) and '[' in reg else reg

def render_idiv(ctx, x, a, b):
  """Render integer division via float conversion with correction for rounding errors.
  v_rcp_f32 is approximate, so 6/-3 might compute as -1.999... truncating to -1 instead of -2.
  Correction: if |remainder| >= |divisor|, we undershot and need to adjust q away from zero.
  IMPORTANT: Uses output register {rx} for quotient. Scratch registers are only used for
  temporary float values and intermediate computations that don't need to persist."""
  ra = extract_low_32(ctx.r[a]) if a.dtype in (dtypes.long, dtypes.ulong) else ctx.r[a]
  rb = extract_low_32(ctx.r[b]) if b.dtype in (dtypes.long, dtypes.ulong) else ctx.r[b]
  rx = extract_low_32(ctx.r[x]) if x.dtype in (dtypes.long, dtypes.ulong) else ctx.r[x]
  s = ctx.get_scratch_vgpr()
  # Use rx for the quotient throughout. Scratch registers s, s+1 are for float temporaries.
  # We save/restore values carefully to avoid collisions when rx is in the scratch range.
  return [
    f"v_cvt_f32_i32 v{s}, {ra}",           # s = float(a)
    f"v_cvt_f32_i32 v{s+1}, {rb}",         # s+1 = float(b)
    f"v_rcp_f32 v{s+1}, v{s+1}",           # s+1 = 1/float(b) (approx)
    f"v_mul_f32 v{s}, v{s}, v{s+1}",       # s = float(a)/float(b) (approx)
    f"v_trunc_f32 v{s}, v{s}",             # s = trunc(a/b)
    f"v_cvt_i32_f32 {rx}, v{s}",           # rx = q (quotient in output register)
    # Now compute remainder r = a - q*b to check if correction is needed
    # Use s for q*b, s+1 for remainder
    f"v_mul_lo_u32 v{s}, {rx}, {rb}",      # s = q*b
    f"v_sub_nc_u32 v{s+1}, {ra}, v{s}",    # s+1 = r = a - q*b (remainder)
    # Check if |r| >= |b| - this means we undershot and need to add/subtract 1
    # Get |r|: abs(r) = (r xor sign(r)) - sign(r)
    f"v_ashrrev_i32 v{s}, 31, v{s+1}",     # s = sign(r): -1 if r<0, 0 if r>=0
    f"v_xor_b32 v{s+2}, v{s+1}, v{s}",     # s+2 = r xor sign(r)
    f"v_sub_nc_u32 v{s+2}, v{s+2}, v{s}",  # s+2 = |r| = (r xor sign) - sign
    # Get |b|: abs(b) = (b xor sign(b)) - sign(b)
    f"v_ashrrev_i32 v{s}, 31, {rb}",       # s = sign(b)
    f"v_xor_b32 v{s+3}, {rb}, v{s}",       # s+3 = b xor sign(b)
    f"v_sub_nc_u32 v{s+3}, v{s+3}, v{s}",  # s+3 = |b|
    # Compare |r| >= |b|
    f"v_cmp_ge_u32 vcc_lo, v{s+2}, v{s+3}",# vcc = |r| >= |b| (need correction)
    # Correction direction: sign(a) xor sign(b) gives sign of quotient
    # If quotient is positive, add 1. If negative, subtract 1.
    f"v_xor_b32 v{s}, {ra}, {rb}",         # s = a xor b
    f"v_ashrrev_i32 v{s}, 31, v{s}",       # s = sign(a xor b): -1 if neg quotient, 0 if pos
    f"v_or_b32 v{s}, v{s}, 1",             # s = -1 if neg, 1 if pos (correction)
    f"v_cndmask_b32 v{s}, 0, v{s}, vcc_lo",# s = correction if needed, else 0
    f"v_add_nc_u32 {rx}, {rx}, v{s}",      # result = q + correction
  ]

def render_mod(ctx, x, a, b):
  """Render integer modulo: a % b = a - (a // b) * b. For 64-bit types, operates on low 32 bits only."""
  ra = extract_low_32(ctx.r[a]) if a.dtype in (dtypes.long, dtypes.ulong) else ctx.r[a]
  rb = extract_low_32(ctx.r[b]) if b.dtype in (dtypes.long, dtypes.ulong) else ctx.r[b]
  rx = extract_low_32(ctx.r[x]) if x.dtype in (dtypes.long, dtypes.ulong) else ctx.r[x]
  scratch = ctx.get_scratch_vgpr()
  # Compute a // b via float conversion
  # Then compute a - (a // b) * b
  return [
    f"v_cvt_f32_i32 v{scratch}, {ra}",
    f"v_cvt_f32_i32 v{scratch+1}, {rb}",
    f"v_rcp_f32 v{scratch+1}, v{scratch+1}",
    f"v_mul_f32 v{scratch}, v{scratch}, v{scratch+1}",
    f"v_trunc_f32 v{scratch}, v{scratch}",
    f"v_cvt_i32_f32 v{scratch}, v{scratch}",      # scratch = a // b
    f"v_mul_lo_u32 v{scratch}, v{scratch}, {rb}", # scratch = (a // b) * b
    f"v_sub_nc_u32 {rx}, {ra}, v{scratch}"]       # result = a - (a // b) * b

def render_comparison(ctx, x, src0):
  """Render comparison op. If dest is SGPR, use directly. If VGPR (fallback), use vcc_lo + v_cndmask_b32."""
  dest = ctx.r[x]
  # For 64-bit integer operands, extract low 32 bits (RDNA has limited 64-bit int ALU)
  srcs = [extract_low_32(ctx.r[v]) for v in x.src]
  dtype, typename = src0.dtype, ctx.types[src0.dtype]
  cmp_instr = ctx.code_for_op[x.op]("vcc_lo" if dest.startswith('v') else dest, *srcs, dtype, typename)
  if dest.startswith('v'):
    # VGPR fallback: compare to vcc_lo, then convert to 0/1 in VGPR
    return [cmp_instr, f"v_cndmask_b32 {dest}, 0, 1, vcc_lo"]
  return cmp_instr

def render_sin(ctx, x, a):
  """Render SIN using v_sin_f32 with extended precision range reduction.
  v_sin_f32 expects input in turns (fraction of 2π cycle).
  For precision with large inputs, we use the two-product algorithm:
    1. t_hi = x * INV_2PI_HI (main turns value)
    2. t_lo = fma(x, INV_2PI_HI, -t_hi) (error term from multiplication)
    3. frac_hi = fract(t_hi)  (fractional part of main term)
    4. frac = frac_hi + t_lo + x * INV_2PI_LO  (combined with corrections)
    5. sin(fract(frac))"""
  dest, src = ctx.r[x], ctx.r[a]
  s = ctx.get_scratch_vgpr()
  # Extended precision 1/(2π):
  # INV_2PI_HI = 0.15915493667125702 -> 0x3E22F983
  # INV_2PI_LO = 6.42e-9 -> 0x31DC9C88
  return [f"v_mul_f32 {dest}, 0x3E22F983, {src}",         # dest = t_hi = x * INV_2PI_HI
          f"v_fma_f32 v{s}, {src}, 0x3E22F983, -{dest}",  # s = t_lo = fma(x, INV_HI, -t_hi) = error
          f"v_fract_f32 {dest}, {dest}",                  # dest = fract(t_hi)
          f"v_add_f32 {dest}, {dest}, v{s}",              # dest += t_lo (add error term)
          f"v_fma_f32 {dest}, {src}, 0x31DC9C88, {dest}", # dest += x * INV_2PI_LO
          f"v_fract_f32 {dest}, {dest}",                  # ensure in [0, 1)
          f"v_sin_f32 {dest}, {dest}"]

def render_recip(ctx, x, a):
  """Render RECIPROCAL using v_rcp_f32 with Newton-Raphson refinement using FMA for precision.
  v_rcp_f32 is approximate (~1 ULP error), which causes issues when the result is truncated.
  Using FMA: y' = y + y*(1 - x*y) = y*(2 - x*y) with better precision.
  FMA computes a*b+c with only one rounding, avoiding intermediate precision loss.
  Special handling: N-R corrupts special values (rcp(inf)=0, but inf*0=nan in N-R), so we
  restore the raw rcp result when N-R produces NaN from a valid (non-NaN) rcp result."""
  dest, src = ctx.r[x], ctx.r[a]
  s = ctx.get_scratch_vgpr()  # get_scratch_vgpr returns base of 4 scratch regs
  # s = N-R temp, s+1 = saved raw rcp result
  # Newton-Raphson with FMA for maximum precision:
  # y' = y + y*(1 - x*y) = y*(2 - x*y)
  # Using FMA: tmp = fma(-x, y, 1.0), y' = fma(y, tmp, y)
  return [f"v_rcp_f32 v{s+1}, {src}",                  # s+1 = y0 = rcp(x) (save raw result)
          f"v_mov_b32 {dest}, v{s+1}",                 # dest = y0 (working copy)
          # First N-R iteration with FMA
          f"v_fma_f32 v{s}, -{src}, {dest}, 1.0",      # s = 1 - x*y0 = fma(-x, y0, 1)
          f"v_fma_f32 {dest}, {dest}, v{s}, {dest}",   # dest = y0 + y0*s = y0*(2-x*y0)
          # Second N-R iteration with FMA
          f"v_fma_f32 v{s}, -{src}, {dest}, 1.0",      # s = 1 - x*y1 = fma(-x, y1, 1)
          f"v_fma_f32 {dest}, {dest}, v{s}, {dest}",   # dest = y1 + y1*s = y1*(2-x*y1)
          # Restore raw rcp if N-R produced NaN but raw rcp was valid (handles inf -> 0 case)
          f"v_cmp_class_f32 vcc_lo, {dest}, 0x3",      # check if dest is NaN (class 0x1|0x2)
          f"v_cndmask_b32 {dest}, {dest}, v{s+1}, vcc_lo"]  # if NaN, use raw rcp result

def render_if(ctx, x):
  """Render IF with stack-based exec save for proper nesting."""
  ctx.scratch_sgpr_used = True
  # Push a new SGPR for this IF level
  save_sgpr = ctx.if_sgpr_base + len(ctx.if_save_stack)
  ctx.if_save_stack.append(save_sgpr)
  ctx.max_if_depth = max(ctx.max_if_depth, len(ctx.if_save_stack))
  return [
    f"s_and_b32 vcc_lo, exec_lo, {ctx.r[x.src[0]]}",
    f"s_and_saveexec_b32 s{save_sgpr}, vcc_lo",
    f"s_cbranch_execz IF_END_{ctx.uops.index(x)}"]

def render_endif(ctx, x):
  """Render ENDIF by popping the exec save stack."""
  # Pop the SGPR for this IF level
  save_sgpr = ctx.if_save_stack.pop()
  return [
    f"IF_END_{ctx.uops.index(x.src[0])}:",
    f"s_mov_b32 exec_lo, s{save_sgpr}"]

def render_64bit_mul(ctx, x):
  """Render 64-bit integer multiplication for division-by-multiplication pattern.
  Stores hi bits in HIGH register of pair for subsequent SHR.
  For signed (long) type with a positive constant that has bit 31 set, we need special handling
  because v_mul_hi_i32 would misinterpret the constant as negative."""
  rx = ctx.r[x]
  rx_hi = f"v{get_reg_base(rx) + 1}" if '[' in rx else rx
  a, b = x.src[0], x.src[1]
  ra = ctx.r[a.src[0]] if a.op is Ops.CAST and a.src[0].dtype.itemsize == 4 else ctx.r[a]
  ra = f"v{get_reg_base(ra)}" if '[' in ra else ra
  rb = render_val(b.arg, dtypes.uint32) if b.op is Ops.CONST else (f"v{get_reg_base(ctx.r[b])}" if '[' in ctx.r[b] else ctx.r[b])
  scratch = ctx.get_scratch_vgpr()

  # For signed multiply (dtypes.long) with a constant that has bit 31 set,
  # v_mul_hi_i32 treats the constant as negative (m - 2^32).
  # We want: (a * m_unsigned) >> 32
  # v_mul_hi_i32 gives: (a * m_signed) >> 32 = (a * (m - 2^32)) >> 32 = (a*m)>>32 - a
  # Therefore: (a*m)>>32 = v_mul_hi_i32(a, m) + a
  if x.dtype == dtypes.long and b.op is Ops.CONST and b.arg >= 0x80000000:
    return [
      f"v_mul_lo_u32 v{scratch}, {ra}, {rb}",
      f"v_mul_hi_i32 {rx_hi}, {ra}, {rb}",
      f"v_add_nc_u32 {rx_hi}, {rx_hi}, {ra}"]

  mul_hi_instr = "v_mul_hi_i32" if x.dtype == dtypes.long else "v_mul_hi_u32"
  return [f"v_mul_lo_u32 v{scratch}, {ra}, {rb}", f"{mul_hi_instr} {rx_hi}, {ra}, {rb}"]

def render_64bit_shr(ctx, x, a, b):
  """Render 64-bit right shift. For shifts >= 32, just shift the high 32 bits.
  Use arithmetic shift (ashrrev) for signed types, logical shift (lshrrev) for unsigned."""
  if b.op is not Ops.CONST: raise RuntimeError("64-bit SHR requires constant shift amount")
  dst_num, src_hi = get_reg_base(ctx.r[x]), get_reg_base(ctx.r[a]) + (1 if '[' in ctx.r[a] else 0)
  shift_amt = b.arg
  # Use arithmetic shift for signed types (long), logical for unsigned (ulong)
  shift_instr = "v_ashrrev_i32" if x.dtype == dtypes.long else "v_lshrrev_b32"
  if shift_amt >= 32:
    remaining = shift_amt - 32
    return f"v_mov_b32 v{dst_num}, v{src_hi}" if remaining == 0 else f"{shift_instr} v{dst_num}, {remaining}, v{src_hi}"
  return f"{shift_instr} v{dst_num}, {shift_amt}, v{src_hi}"

def render_where_64(ctx, x, cond, true_val, false_val):
  """Render WHERE for 64-bit types using two v_cndmask_b32 instructions.
  v_cndmask_b32 only works with 32-bit registers, so we need to handle each half separately."""
  dst, ra_t, ra_f = ctx.r[x], ctx.r[true_val], ctx.r[false_val]
  dst_lo, dst_hi = get_reg_base(dst), get_reg_base(dst) + 1
  t_lo, t_hi = get_reg_base(ra_t), get_reg_base(ra_t) + 1
  f_lo, f_hi = get_reg_base(ra_f), get_reg_base(ra_f) + 1
  cond_reg = ctx.r[cond]
  if cond_reg.startswith('s'):
    return [f"v_cndmask_b32 v{dst_lo}, v{f_lo}, v{t_lo}, {cond_reg}",
            f"v_cndmask_b32 v{dst_hi}, v{f_hi}, v{t_hi}, {cond_reg}"]
  return [f"v_cmp_ne_u32 vcc_lo, {cond_reg}, 0",
          f"v_cndmask_b32 v{dst_lo}, v{f_lo}, v{t_lo}, vcc_lo",
          f"v_cndmask_b32 v{dst_hi}, v{f_hi}, v{t_hi}, vcc_lo"]

def render_wmma(ctx, x):
  """Render WMMA instruction for RDNA3.
  RDNA3 WMMA: v_wmma_f32_16x16x16_f16 dst[0:7], src_a[0:7], src_b[0:7], src_c[0:7]
  - dtype_in: half (16 elements per thread = 8 VGPRs)
  - dtype_out: float (8 elements per thread = 8 VGPRs) or half (8 elements = 4 VGPRs, packed)
  """
  # x.arg[2] = dtype_in, x.arg[3] = dtype_out
  dtype_in, dtype_out = x.arg[2], x.arg[3]
  # Get the register ranges for the three sources
  # src[0] = A matrix (half16 = 8 VGPRs), src[1] = B matrix (half16 = 8 VGPRs), src[2] = accumulator
  def get_reg_range(reg_list, count):
    """Convert a list of register names to a v[start:end] range."""
    if isinstance(reg_list, list):
      # Extract register numbers from v0, v1, etc.
      nums = [int(r[1:]) for r in reg_list]
      return f"v[{min(nums)}:{max(nums)}]"
    # Single register string like v[10:17]
    return reg_list

  ra = ctx.r[x.src[0]]  # A matrix
  rb = ctx.r[x.src[1]]  # B matrix
  rc = ctx.r[x.src[2]]  # accumulator
  rd = ctx.r[x]         # destination (same size as accumulator)

  # Convert register lists to ranges
  ra_range = get_reg_range(ra, 8)
  rb_range = get_reg_range(rb, 8)
  rc_range = get_reg_range(rc, 8)
  rd_range = get_reg_range(rd, 8)

  # Select instruction based on input/output types
  if dtype_out == dtypes.float:
    if dtype_in == dtypes.half:
      instr = "v_wmma_f32_16x16x16_f16"
    elif dtype_in == dtypes.bfloat16:
      instr = "v_wmma_f32_16x16x16_bf16"
    else:
      raise RuntimeError(f"Unsupported WMMA dtype_in: {dtype_in}")
  elif dtype_out == dtypes.half:
    instr = "v_wmma_f16_16x16x16_f16"
  else:
    raise RuntimeError(f"Unsupported WMMA dtype_out: {dtype_out}")

  return f"{instr} {rd_range}, {ra_range}, {rb_range}, {rc_range}"

def render_add_with_literal(ctx, x, a, b):
  """Render ADD using literal constant instead of register for the constant operand.
  RDNA3 VOP2 constraints: dst, src0, src1 where src0 can be literal but src1 must be VGPR."""
  # Only use literal optimization if `a` has a proper VGPR (not a literal string)
  # This can happen if `a` is also a CONST that was optimized away
  ra = ctx.r[a]
  if not isinstance(ra, str) or not ra.startswith('v'):
    return None  # Fall back to default rendering
  rd = ctx.r[x]
  # Use the constant value directly as a literal (not from register)
  # IMPORTANT: literal must be src0, VGPR must be src1 for VOP2 instructions
  const_val = render_val(b.arg, b.dtype)
  if dtypes.is_float(x.dtype):
    return f"v_add_{ctx.types[x.dtype]} {rd}, {const_val}, {ra}"
  return f"v_add_nc_u32 {rd}, {const_val}, {ra}"

def render_mul_with_literal(ctx, x, a, b):
  """Render MUL using literal constant instead of register.
  RDNA3 VOP2 constraints: dst, src0, src1 where src0 can be literal but src1 must be VGPR."""
  # Skip 64-bit integer types - they need special handling (render_64bit_mul)
  if x.dtype in (dtypes.long, dtypes.ulong):
    return None  # Fall through to render_64bit_mul
  # Only use literal optimization if `a` has a proper VGPR
  ra = ctx.r[a]
  if not isinstance(ra, str) or not ra.startswith('v'):
    return None  # Fall back to default rendering
  rd = ctx.r[x]
  const_val = render_val(b.arg, b.dtype)
  # IMPORTANT: literal must be src0, VGPR must be src1
  if dtypes.is_float(x.dtype):
    return f"v_mul_{ctx.types[x.dtype]} {rd}, {const_val}, {ra}"
  return f"v_mul_lo_u32 {rd}, {const_val}, {ra}"

string_rewrite = PatternMatcher([
  # WMMA for tensor cores
  (UPat(Ops.WMMA, name="x"), render_wmma),
  # ADD/MUL with constant operand - use literal instead of register (saves VGPRs)
  # Note: literal goes in src0 (first operand after dst), VGPR in src1 (VOP2 constraint)
  (UPat(Ops.ADD, name="x", src=(UPat.var("a"), UPat.cvar("b"))), render_add_with_literal),
  (UPat(Ops.ADD, name="x", src=(UPat.cvar("b"), UPat.var("a"))), render_add_with_literal),
  (UPat(Ops.MUL, name="x", src=(UPat.var("a"), UPat.cvar("b"))), render_mul_with_literal),
  (UPat(Ops.MUL, name="x", src=(UPat.cvar("b"), UPat.var("a"))), render_mul_with_literal),
  # const rendering
  (UPat.cvar("x", dtypes.bool), lambda ctx, x: f"v_mov_b32 {ctx.r[x]}, {1 if x.arg else 0}"),
  # 64-bit float constants need two mov instructions
  (UPat.cvar("x", dtypes.float64), render_const_64),
  # 64-bit integer constants: use render_const_64 for pairs, single mov for scalar
  (UPat.cvar("x", dtypes.long), lambda ctx, x: render_const_64(ctx, x) if '[' in ctx.r[x] else f"v_mov_b32 {ctx.r[x]}, {render_val(x.arg, dtypes.int32)}"),
  (UPat.cvar("x", dtypes.ulong), lambda ctx, x: render_const_64(ctx, x) if '[' in ctx.r[x] else f"v_mov_b32 {ctx.r[x]}, {render_val(x.arg, dtypes.uint32)}"),
  (UPat.cvar("x"), lambda ctx, x: f"v_mov_b32 {ctx.r[x]}, {render_val(x.arg, x.dtype)}"),
  # special registers
  (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: ctx.render_special(x)),
  # define global
  (UPat(Ops.DEFINE_GLOBAL, name="x"), lambda ctx, x: f"s_load_b64 {ctx.r[x]}, s[0:1], {ctx.kernarg_offset[x]}"),
  # define var - load variable from kernarg buffer
  (UPat(Ops.DEFINE_VAR, name="x"), render_define_var),
  # Boolean inversion: CMPNE(bool, 1) -> s_not_b32 (invert wave mask)
  (UPat(Ops.CMPNE, name="x", src=(UPat(dtype=dtypes.bool, name="a"), UPat.cvar("b"))),
   lambda ctx, x, a, b: f"s_not_b32 {ctx.r[x]}, {ctx.r[a]}" if b.arg == 1 and ctx.r[a].startswith('s') else None),
  # comparison ops - uses SGPR if available, falls back to VGPR with vcc_lo + v_cndmask_b32
  (UPat((Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ), name="x", allow_any_len=True, src=(UPat.var("src0"),)), render_comparison),
  # NOTE: WHERE wrapping LOAD is transformed to gated LOAD at UOp level by rdna_matcher
  # WHERE 64-bit: need two v_cndmask_b32 for each half
  (UPat(Ops.WHERE, name="x", dtype=dtypes.float64, src=(UPat.var("cond"), UPat.var("true_val"), UPat.var("false_val"))), render_where_64),
  (UPat(Ops.WHERE, name="x", dtype=dtypes.long, src=(UPat.var("cond"), UPat.var("true_val"), UPat.var("false_val"))), render_where_64),
  (UPat(Ops.WHERE, name="x", dtype=dtypes.ulong, src=(UPat.var("cond"), UPat.var("true_val"), UPat.var("false_val"))), render_where_64),
  # WHERE: if condition is SGPR or vcc_lo, use directly; if VGPR, compare to 0 first to get VCC
  # CRITICAL: When both cond AND true_val are SGPR lane masks (from comparisons), we must use s_and_b32
  # to properly combine the masks, because v_cndmask_b32 treats SGPR operands as scalar values.
  (UPat(Ops.WHERE, name="x", src=(UPat.var("cond"), UPat.var("true_val"), UPat.var("false_val"))),
   lambda ctx, x, cond, true_val, false_val: ctx.render_where(x, cond, true_val, false_val)),
  # RECIPROCAL: v_rcp_f32 is approximate, use Newton-Raphson refinement for better precision
  # This ensures division results are exact when they should be (e.g., 6.0/3.0 = 2.0, not 1.999...)
  (UPat(Ops.RECIPROCAL, name="x", src=(UPat.var("a"),)), render_recip),
  # SIN: v_sin_f32 expects input in turns (1 turn = 2π radians), so we multiply by 1/(2π) first
  # NOTE: v_sin_f32 has limited range (~256 turns) but we accept this for native instruction benefits
  (UPat(Ops.SIN, name="x", src=(UPat.var("a"),)), render_sin),
  # IDIV: integer division via float conversion (a // b = trunc(float(a) / float(b)))
  (UPat(Ops.IDIV, name="x", src=(UPat.var("a"), UPat.var("b"))), render_idiv),
  # Integer MOD: a % b = a - (a // b) * b (floats use v_mod_f32 via code_for_op)
  (UPat(Ops.MOD, name="x", src=(UPat.var("a"), UPat.var("b"))),
   lambda ctx, x, a, b: render_mod(ctx, x, a, b) if dtypes.is_int(x.dtype) else None),
  # Boolean AND/OR with SGPR sources: need to convert to VGPR first
  (UPat(Ops.AND, name="x", dtype=dtypes.bool, src=(UPat.var("a", dtype=dtypes.bool), UPat.var("b", dtype=dtypes.bool))),
   lambda ctx, x, a, b: ctx.render_bool_logic(x, a, b, "and")),
  (UPat(Ops.OR, name="x", dtype=dtypes.bool, src=(UPat.var("a", dtype=dtypes.bool), UPat.var("b", dtype=dtypes.bool))),
   lambda ctx, x, a, b: ctx.render_bool_logic(x, a, b, "or")),
  # 64-bit integer MUL: need full 64-bit product for division-by-multiplication pattern
  (UPat(Ops.MUL, name="x", dtype=dtypes.long), render_64bit_mul),
  (UPat(Ops.MUL, name="x", dtype=dtypes.ulong), render_64bit_mul),
  # 64-bit integer SHR: for shifts >= 32, use high bits only
  (UPat(Ops.SHR, name="x", dtype=dtypes.long, src=(UPat.var("a"), UPat.cvar("b"))), render_64bit_shr),
  (UPat(Ops.SHR, name="x", dtype=dtypes.ulong, src=(UPat.var("a"), UPat.cvar("b"))), render_64bit_shr),
  # alu ops - extract low 32 bits for 64-bit integer operands since RDNA has limited 64-bit int ALU
  (UPat(GroupOp.ALU, name="x"), lambda ctx, x: ctx.code_for_op[x.op](
    extract_low_32(ctx.r[x]) if x.dtype in (dtypes.long, dtypes.ulong) else ctx.r[x],
    *[extract_low_32(ctx.r[v]) if v.dtype in (dtypes.long, dtypes.ulong) else ctx.r[v] for v in x.src],
    x.dtype, ctx.types[x.dtype])),
  # bitcast/cast
  # BITCAST - need special handling for 64-bit float types
  (UPat(Ops.BITCAST, name="x", dtype=dtypes.float64, src=(UPat.var("a"),), allow_any_len=True),
   lambda ctx, x, a: ctx.render_mov_64(x, a)),
  (UPat(Ops.BITCAST, name="x", src=(UPat.var("a"),), allow_any_len=True), lambda ctx, x, a: f"v_mov_b32 {ctx.r[x]}, {ctx.r[a]}"),
  # cast from bool: if bool is in SGPR (comparison result), use v_cndmask_b32; if in VGPR (loaded from memory), convert directly
  (UPat(Ops.CAST, name="x", src=(UPat(dtype=dtypes.bool, name="a"),)),
   lambda ctx, x, a: f"v_cndmask_b32 {ctx.r[x]}, {render_val(0, x.dtype)}, {render_val(1, x.dtype)}, {ctx.r[a]}"
     if ctx.r[a].startswith('s') else f"v_cvt_f32_u32 {ctx.r[x]}, {ctx.r[a]}" if dtypes.is_float(x.dtype) else f"v_mov_b32 {ctx.r[x]}, {ctx.r[a]}"),
  # cast TO bool: compare to 0, result to vcc_lo, then cndmask to VGPR
  (UPat(Ops.CAST, name="x", dtype=dtypes.bool, src=(UPat.var("a"),)),
   lambda ctx, x, a: [
     f"v_cmp_{'neq' if dtypes.is_float(a.dtype) else 'ne'}_{ctx.types[a.dtype]} vcc_lo, {ctx.r[a]}, 0",
     f"v_cndmask_b32 {ctx.r[x]}, 0, 1, vcc_lo"]),
  # cast TO 64-bit int: move low 32 bits, zero high 32 bits (for register pairs)
  (UPat(Ops.CAST, name="x", dtype=dtypes.long, src=(UPat.var("a"),)),
   lambda ctx, x, a: ctx.render_cast_to_64(x, a, signed=True)),
  (UPat(Ops.CAST, name="x", dtype=dtypes.ulong, src=(UPat.var("a"),)),
   lambda ctx, x, a: ctx.render_cast_to_64(x, a, signed=False)),
  (UPat(Ops.CAST, name="x", src=(UPat.var("a"),)), lambda ctx, x, a: ctx.render_cast(x, a)),
  # store / load for global memory
  # store boolean value - if SGPR (comparison result), convert via cndmask; if VGPR, store directly
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_GLOBAL, name="buf"), UPat.var("idx")), name="index_op", allow_any_len=True), UPat.var("var", dtype=dtypes.bool))),
   lambda ctx, idx, var, buf, index_op: [
     f"v_cndmask_b32 v{ctx.get_scratch_vgpr()}, 0, 1, {ctx.r[var]}",
     f"global_store_byte {ctx.r[index_op]}, v{ctx.get_scratch_vgpr()}, {ctx.r[buf]}"]
       if ctx.r[var].startswith('s') else f"global_store_byte {ctx.r[index_op]}, {ctx.r[var]}, {ctx.r[buf]}"),
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_GLOBAL, name="buf"), UPat.var("idx")), name="index_op", allow_any_len=True), UPat.var("var"))),
   lambda ctx, idx, var, buf, index_op: global_store(ctx.r[index_op], ctx.r[var], ctx.r[buf], var.dtype)),
  (UPat(Ops.LOAD, name="x", src=(UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_GLOBAL, name="buf"), UPat.var("idx"), UPat.var("gate")), name="index_op"), UPat.var("alt")), allow_any_len=True),
    lambda ctx, x, idx, alt, gate, buf, index_op: gated_load(ctx, x, idx, alt, gate, buf, index_op)),
  (UPat(Ops.LOAD, name="x", src=(UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_GLOBAL, name="buf"), UPat.var("idx")), name="index_op"),), allow_any_len=True),
    lambda ctx, x, idx, buf, index_op: global_load(ctx.r[x], ctx.r[index_op], ctx.r[buf], x.dtype)),
  # store / load for local memory (LDS) - DEFINE_LOCAL directly
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_LOCAL), UPat.var("idx")), name="index_op", allow_any_len=True), UPat.var("var"))),
   lambda ctx, idx, var, index_op: ds_write(ctx.r[index_op], ctx.r[var], var.dtype)),
  (UPat(Ops.LOAD, name="x", src=(UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_LOCAL), UPat.var("idx")), name="index_op"),), allow_any_len=True),
    lambda ctx, x, idx, index_op: ds_read(ctx.r[x], ctx.r[index_op], x.dtype)),
  # store / load for local memory (LDS) - DEFINE_LOCAL wrapped in AFTER
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat(Ops.AFTER, src=(UPat(Ops.DEFINE_LOCAL),), allow_any_len=True), UPat.var("idx")), name="index_op", allow_any_len=True), UPat.var("var"))),
   lambda ctx, idx, var, index_op: ds_write(ctx.r[index_op], ctx.r[var], var.dtype)),
  (UPat(Ops.LOAD, name="x", src=(UPat(Ops.INDEX, src=(UPat(Ops.AFTER, src=(UPat(Ops.DEFINE_LOCAL),), allow_any_len=True), UPat.var("idx")), name="index_op"),), allow_any_len=True),
    lambda ctx, x, idx, index_op: ds_read(ctx.r[x], ctx.r[index_op], x.dtype)),
  # simple
  (UPat(Ops.DEFINE_REG, src=()), lambda ctx: []),
  (UPat(Ops.RANGE, name="r"), lambda ctx, r: [
    f"v_mov_b32 {ctx.r[r]}, -1",  # Start at -1, incremented to 0 on first iteration
    f"s_branch LOOP_END_{ctx.uops.index(r)}",
    f"LOOP_{ctx.uops.index(r)}:"]),
  (UPat(Ops.END, name="x", src=(UPat(), UPat(Ops.RANGE, name="r"))), lambda ctx, x, r: [
    f"LOOP_END_{ctx.uops.index(r)}:",
    f"v_add_nc_u32 {ctx.r[r]}, {ctx.r[r]}, 1",
    f"v_cmp_lt_i32 vcc_lo, {ctx.r[r]}, {ctx.r[r.src[0]]}",
    f"s_cbranch_vccnz LOOP_{ctx.uops.index(r)}"]),
  (UPat(Ops.DEFINE_LOCAL, name="x"),
   lambda ctx, x: []),  # local memory is handled differently in RDNA
  (UPat(Ops.IF, name="x"), render_if),
  (UPat(Ops.ENDIF, name="x"), render_endif),
  (UPat(Ops.BARRIER, name="x"), lambda ctx, x: "s_barrier"),
])

class RDNARenderer(Renderer):
  device = "AMD"
  suffix = "RDNA"
  supports_float4 = True  # Vectorized loads for better register efficiency
  global_max = (2147483647, 65535, 65535)
  local_max = (1024, 1024, 1024)
  shared_max = 65536
  max_upcast_size = 16  # RDNA3 has 256 VGPRs max, limit unrolling to reduce register pressure
  code_for_op = asm_for_op
  extra_matcher = rdna_matcher
  tensor_cores = tc.amd_rdna3  # RDNA3 WMMA tensor cores

  def __init__(self, arch:str="gfx1100"):
    self.arch = arch
    # gfx1100 = RDNA3, gfx1201 = RDNA4
    # wavefront size is 32 for RDNA3
    self.wave32 = True

  def __reduce__(self): return self.__class__, (self.arch,)

  # NOTE: 64-bit integers lowered to 32-bit since RDNA3 has limited 64-bit integer ALU support
  # This affects precision for operations like sin polynomial range reduction for very large inputs
  types: dict[DType, str] = {
    dtypes.int8: "i32", dtypes.int16: "i32", dtypes.int32: "i32", dtypes.int64: "i32",
    dtypes.uint8: "u32", dtypes.uint16: "u32", dtypes.uint32: "u32", dtypes.uint64: "u32",
    dtypes.float16: "f16", dtypes.float32: "f32", dtypes.float64: "f64", dtypes.bool: "i32"
  }

  def is_local(self, index_op: UOp) -> bool:
    """Check if an INDEX operation is for local memory"""
    return isinstance(index_op.dtype, PtrDType) and index_op.dtype.addrspace == AddrSpace.LOCAL

  def render_special(self, x: UOp) -> str:
    # SPECIAL arg is like 'g0', 'g1', 'g2' for global dims or 'l0', 'l1', 'l2' for local dims
    dim = int(x.arg[-1])
    if x.arg[0] == 'g':
      # With .amdhsa_user_sgpr_count 2, system SGPRs start at s2
      # s[0:1] = kernarg ptr, s2 = workgroup_id_x, s3 = y, s4 = z
      return f"v_mov_b32 {self.r[x]}, s{2+dim}"
    elif x.arg[0] == 'l':
      # local id: v0 contains packed xyz (v0.x = bits 0-9, v0.y = bits 10-19, v0.z = bits 20-29)
      if dim == 0:
        return f"v_and_b32 {self.r[x]}, 0x3ff, v0"
      elif dim == 1:
        return f"v_bfe_u32 {self.r[x]}, v0, 10, 10"
      elif dim == 2:
        return f"v_bfe_u32 {self.r[x]}, v0, 20, 10"
    else:
      # 'i' for global index = gid * local_size + lid - this is computed later
      return f"v_mov_b32 {self.r[x]}, 0"  # placeholder
    raise RuntimeError(f"Unknown special: {x.arg}")

  def render_cast(self, x: UOp, a: UOp) -> str|list[str]:
    # RDNA3 cast instructions
    if x.dtype == dtypes.float32 and a.dtype == dtypes.int32:
      return f"v_cvt_f32_i32 {self.r[x]}, {self.r[a]}"
    elif x.dtype == dtypes.float32 and a.dtype == dtypes.uint32:
      return f"v_cvt_f32_u32 {self.r[x]}, {self.r[a]}"
    elif x.dtype == dtypes.int32 and a.dtype == dtypes.float32:
      return f"v_cvt_i32_f32 {self.r[x]}, {self.r[a]}"
    elif x.dtype == dtypes.uint32 and a.dtype == dtypes.float32:
      return f"v_cvt_u32_f32 {self.r[x]}, {self.r[a]}"
    # float32 -> smaller int types: convert to int32 first, truncation happens via store
    elif x.dtype in (dtypes.int8, dtypes.int16) and a.dtype == dtypes.float32:
      return f"v_cvt_i32_f32 {self.r[x]}, {self.r[a]}"
    elif x.dtype in (dtypes.uint8, dtypes.uint16) and a.dtype == dtypes.float32:
      return f"v_cvt_u32_f32 {self.r[x]}, {self.r[a]}"
    elif x.dtype == dtypes.float16 and a.dtype == dtypes.float32:
      return f"v_cvt_f16_f32 {self.r[x]}, {self.r[a]}"
    elif x.dtype == dtypes.float32 and a.dtype == dtypes.float16:
      return f"v_cvt_f32_f16 {self.r[x]}, {self.r[a]}"
    # float64 conversions
    elif x.dtype == dtypes.float64 and a.dtype == dtypes.float32:
      return f"v_cvt_f64_f32 {self.r[x]}, {self.r[a]}"
    elif x.dtype == dtypes.float32 and a.dtype == dtypes.float64:
      src = get_reg_base(self.r[a])
      return f"v_cvt_f32_f64 {self.r[x]}, v[{src}:{src+1}]"
    elif x.dtype == dtypes.float64:
      return self.render_mov_64(x, a)  # TODO: proper int->f64 conversion
    elif x.dtype.itemsize == 4 and a.dtype in (dtypes.long, dtypes.ulong):
      return f"v_mov_b32 {self.r[x]}, v{get_reg_base(self.r[a])}" if '[' in self.r[a] else f"v_mov_b32 {self.r[x]}, {self.r[a]}"
    # fallback: just move (same size types)
    return f"v_mov_b32 {self.r[x]}, {self.r[a]}"

  def render_bool_logic(self, x: UOp, a: UOp, b: UOp, op: str) -> str|list[str]:
    """Render boolean AND/OR with proper SGPR to VGPR handling."""
    ra, rb, rx = self.r[a], self.r[b], self.r[x]
    s_op, v_op = f"s_{op}_b32", f"v_{op}_b32"
    if ra.startswith('s') and rb.startswith('s'):
      return [f"{s_op} vcc_lo, {ra}, {rb}", f"v_cndmask_b32 {rx}, 0, 1, vcc_lo"]
    if ra.startswith('s'):
      return [f"v_cndmask_b32 v{self.get_scratch_vgpr()}, 0, 1, {ra}", f"{v_op} {rx}, v{self.get_scratch_vgpr()}, {rb}"]
    if rb.startswith('s'):
      return [f"v_cndmask_b32 v{self.get_scratch_vgpr()}, 0, 1, {rb}", f"{v_op} {rx}, {ra}, v{self.get_scratch_vgpr()}"]
    return f"{v_op} {rx}, {ra}, {rb}"

  def render_where(self, x: UOp, cond: UOp, true_val: UOp, false_val: UOp) -> str|list[str]:
    """Render WHERE with proper handling of SGPR lane masks.

    CRITICAL: When true_val or false_val is an SGPR lane mask (from comparisons), v_cndmask_b32
    would incorrectly treat it as a scalar. We must expand SGPR lane masks to VGPR 0/1 values first.

    Cases:
    - cond=SGPR, true_val=SGPR, false_val=0: use s_and_b32 to combine masks
    - true_val or false_val is SGPR lane mask: expand to VGPR first
    - Standard case: use v_cndmask_b32 directly
    """
    rc, rt, rf, rx = self.r[cond], self.r[true_val], self.r[false_val], self.r[x]
    is_cond_sgpr = rc.startswith('s') or rc == 'vcc_lo'
    is_true_sgpr_mask = rt.startswith('s') and true_val.dtype == dtypes.bool
    is_false_sgpr_mask = rf.startswith('s') and false_val.dtype == dtypes.bool
    is_false_zero = rf == '0' or (false_val.op is Ops.CONST and false_val.arg == 0)

    # Special case: both cond and true_val are SGPR masks, false_val is 0
    # This is equivalent to AND of the two masks, which we can do in SGPR
    if is_cond_sgpr and is_true_sgpr_mask and is_false_zero:
      return [f"s_and_b32 vcc_lo, {rc}, {rt}", f"v_cndmask_b32 {rx}, 0, 1, vcc_lo"]

    # If true_val or false_val is an SGPR lane mask, expand it to VGPR 0/1 values first
    # This is necessary because v_cndmask_b32 treats SGPR operands as scalar values
    result = []
    scratch = None
    if is_true_sgpr_mask:
      scratch = self.get_scratch_vgpr()
      result.append(f"v_cndmask_b32 v{scratch}, 0, 1, {rt}")
      rt = f"v{scratch}"
    if is_false_sgpr_mask:
      scratch = scratch if scratch else self.get_scratch_vgpr()
      idx = scratch + 1 if is_true_sgpr_mask else scratch
      result.append(f"v_cndmask_b32 v{idx}, 0, 1, {rf}")
      rf = f"v{idx}"

    # Now handle the condition
    if is_cond_sgpr:
      result.append(f"v_cndmask_b32 {rx}, {rf}, {rt}, {rc}")
    else:
      result.append(f"v_cmp_ne_u32 vcc_lo, {rc}, 0")
      result.append(f"v_cndmask_b32 {rx}, {rf}, {rt}, vcc_lo")

    return result if len(result) > 1 else result[0]

  def render_mov_64(self, x: UOp, a: UOp) -> list[str]:
    """Render 64-bit move using two v_mov_b32 instructions"""
    dst, src = get_reg_base(self.r[x]), get_reg_base(self.r[a])
    return [f"v_mov_b32 v{dst}, v{src}", f"v_mov_b32 v{dst+1}, v{src+1}"]

  def render_cast_to_64(self, x: UOp, a: UOp, signed: bool = False) -> str|list[str]:
    """Render cast from 32-bit to 64-bit integer (sign or zero extend)"""
    rx, ra = self.r[x], self.r[a]
    if '[' not in rx:
      src_reg = extract_low_32(ra)
      if ra.startswith('s'): return f"v_cndmask_b32 {rx}, 0, 1, {ra}"
      return f"v_mov_b32 {rx}, {src_reg}"
    dst_num, src_reg = get_reg_base(rx), extract_low_32(ra)
    if ra.startswith('s'):
      return [f"v_cndmask_b32 v{dst_num}, 0, 1, {ra}", f"v_mov_b32 v{dst_num+1}, 0"]
    if signed:
      return [f"v_mov_b32 v{dst_num}, {src_reg}", f"v_ashrrev_i32 v{dst_num+1}, 31, {src_reg}"]
    else:
      # Zero extend
      return [
        f"v_mov_b32 v{dst_num}, {src_reg}",
        f"v_mov_b32 v{dst_num+1}, 0"
      ]

  def render_kernel(self, kernel, function_name, bufs, v_cnt, s_cnt, uops) -> str:
    # Build metadata for kernel
    args = []
    for name, dtype in bufs:
      if name.startswith("data"):
        i = int(name[4:])
        args.append({'.address_space': 'global', '.name': f'buf_{i}', '.offset': i*8, '.size': 8,
                     '.type_name': 'void*', '.value_kind': 'global_buffer'})
      else:
        # variable
        args.append({'.name': name, '.offset': len(args)*8, '.size': 8, '.value_kind': 'by_value'})

    kernarg_size = (args[-1][".offset"] + args[-1][".size"]) if args else 0

    metadata = {
      'amdhsa.kernels': [{
        '.args': args,
        '.group_segment_fixed_size': self.lds_size, '.kernarg_segment_align': 8, '.kernarg_segment_size': kernarg_size,
        '.language': 'OpenCL C', '.language_version': [1, 2], '.max_flat_workgroup_size': 256,
        '.name': function_name, '.private_segment_fixed_size': 0, '.sgpr_count': s_cnt, '.sgpr_spill_count': 0,
        '.symbol': f'{function_name}.kd', '.uses_dynamic_stack': False, '.vgpr_count': v_cnt, '.vgpr_spill_count': 0,
        '.wavefront_size': 32
      }],
      'amdhsa.target': f'amdgcn-amd-amdhsa--{self.arch}', 'amdhsa.version': [1, 2]
    }

    kernel_str = '\n'.join(kernel)
    # NOTE: .text must be first line for HIPCompiler to detect as assembly
    return ".text\n" + \
           f'.amdgcn_target "amdgcn-amd-amdhsa--{self.arch}"\n' + \
           ".amdhsa_code_object_version 6\n" + \
           f".global {function_name}\n" + \
           f".type {function_name},@function\n" + \
           ".p2align 8\n" + \
           f"{function_name}:\n" + \
           kernel_str + f"\n.size {function_name}, .-{function_name}\n\n" + \
           ".rodata\n" + \
           ".p2align 6\n" + \
           f".amdhsa_kernel {function_name}\n" + \
           f"  .amdhsa_group_segment_fixed_size {self.lds_size}\n" + \
           f"  .amdhsa_kernarg_size {kernarg_size}\n" + \
           f"  .amdhsa_user_sgpr_count 2\n" + \
           f"  .amdhsa_next_free_vgpr {v_cnt}\n" + \
           f"  .amdhsa_next_free_sgpr {s_cnt}\n" + \
           "  .amdhsa_wavefront_size32 1\n" + \
           "  .amdhsa_user_sgpr_kernarg_segment_ptr 1\n" + \
           "  .amdhsa_system_sgpr_workgroup_id_x 1\n" + \
           "  .amdhsa_system_sgpr_workgroup_id_y 1\n" + \
           "  .amdhsa_system_sgpr_workgroup_id_z 1\n" + \
           "  .amdhsa_system_vgpr_workitem_id 2\n" + \
           ".end_amdhsa_kernel\n\n" + \
           ".amdgpu_metadata\n" + yaml.dump(metadata) + ".end_amdgpu_metadata"

  def render(self, uops:list[UOp]) -> str:
    kernel:list[str] = []
    bufs = []
    kernarg_offset: dict[UOp, int] = {}  # Track kernarg offset for each DEFINE_GLOBAL/DEFINE_VAR
    current_kernarg_offset = 0

    r: dict[UOp, str|list[str]] = {}
    self.r = r
    self.kernarg_offset = kernarg_offset
    self.uops = uops
    # Separate SGPRs for different exec save contexts to avoid collisions
    self.gated_sgpr = 100  # for gated_load (atomic save/restore, doesn't nest)
    self.if_sgpr_base = 101  # for IF/ENDIF (can nest, uses stack)
    self.if_save_stack: list[int] = []  # stack of IF save SGPRs
    self.max_if_depth = 0  # track max nesting depth for SGPR count
    self.scratch_sgpr_used = False  # track if any scratch SGPRs are used
    MAX_SGPR = 100  # RDNA3 limit ~106, reserve some for scratch
    self.lds_size = 0  # track local memory (LDS) usage
    # Scratch VGPR will be allocated after we know how many VGPRs the kernel uses
    self.scratch_vgpr = -1  # will be set after register allocation
    self._deferred_store_vgpr = -1  # reset for each kernel - must NOT persist from previous kernel

    # === LIVENESS ANALYSIS ===
    # Compute last use position for each UOp
    last_use: dict[UOp, int] = {}
    # Track which UOps alias to other UOps (share registers)
    aliases: dict[UOp, UOp] = {}

    # First pass: find all RANGE/END pairs to identify loop ranges
    # Map from RANGE position to END position
    loop_ranges: dict[int, int] = {}
    range_positions: dict[UOp, int] = {}
    for i, u in enumerate(uops):
      if u.op is Ops.RANGE:
        range_positions[u] = i
      if u.op is Ops.END and len(u.src) >= 2 and u.src[1].op is Ops.RANGE:
        range_op = u.src[1]
        if range_op in range_positions:
          loop_ranges[range_positions[range_op]] = i

    for i, u in enumerate(uops):
      for src in u.src:
        last_use[src] = i
      # For INDEX inside LOAD/STORE, track both the INDEX and its sources
      if u.op in {Ops.LOAD, Ops.STORE} and u.src[0].op is Ops.INDEX:
        last_use[u.src[0]] = i  # INDEX's liveness extends to the LOAD/STORE that uses it
        for src in u.src[0].src:
          last_use[src] = i
      # For END, track RANGE.src[0] (loop bound) which is used in rendering
      if u.op is Ops.END and len(u.src) >= 2 and u.src[1].op is Ops.RANGE:
        range_op = u.src[1]
        if len(range_op.src) > 0:
          last_use[range_op.src[0]] = i
      # Track aliases - UOps that share registers
      if u.op is Ops.AFTER:
        aliases[u] = u.src[0]
      if u.op in {Ops.CAST, Ops.BITCAST} and (u.src[0].dtype == u.dtype or isinstance(u.src[0].dtype, PtrDType)):
        aliases[u] = u.src[0]
      # GEP on vector types aliases to source - element extraction shares the source's registers
      if u.op is Ops.GEP and isinstance(u.src[0].dtype, DType) and u.src[0].dtype.count > 1:
        aliases[u] = u.src[0]
      # LOAD/STORE/INDEX on REG addrspace alias to the DEFINE_REG
      if u.op in {Ops.INDEX, Ops.LOAD, Ops.STORE} and len(u.src) > 0:
        if isinstance(u.src[0].dtype, PtrDType) and u.src[0].dtype.addrspace == AddrSpace.REG:
          if u.op is Ops.INDEX:
            aliases[u] = u.src[0]  # INDEX on REG aliases to DEFINE_REG
          elif u.op is Ops.LOAD:
            aliases[u] = u.src[0]  # LOAD from REG aliases to INDEX (which aliases to DEFINE_REG)

    # Second pass: extend last_use for values DEFINED OUTSIDE a loop but USED INSIDE
    # Only these need their lifetime extended to the END of the loop (for loop-carried dependencies)
    # Values defined INSIDE a loop can be freed normally
    uop_positions = {u: i for i, u in enumerate(uops)}
    for uop, use_pos in list(last_use.items()):
      if uop not in uop_positions:
        continue
      def_pos = uop_positions[uop]
      # Check if defined OUTSIDE loop but used INSIDE loop
      for range_pos, end_pos in loop_ranges.items():
        # Value defined before the loop starts
        if def_pos <= range_pos:
          # And used inside the loop
          if range_pos < use_pos <= end_pos:
            # Extend lifetime to end of loop
            last_use[uop] = max(last_use[uop], end_pos)

    # Third pass: extend SPECIAL (thread ID) lifetimes to end of kernel
    # Thread IDs are used to compute addresses throughout the kernel, including at the very end.
    # Their derived values may be used after the SPECIAL itself is last referenced, so we must
    # keep SPECIAL registers alive for the entire kernel to prevent register reuse corruption.
    max_uop_pos = len(uops) - 1
    for u in uops:
      if u.op is Ops.SPECIAL:
        last_use[u] = max_uop_pos

    # === REGISTER ALLOCATOR ===
    # Track free registers (available for reuse)
    free_vgprs: list[int] = []
    free_vgpr_pairs: list[int] = []  # Track free aligned pairs (base register numbers)
    free_vgpr_ranges: list[tuple[int, int]] = []  # Track free contiguous ranges (base, count)
    free_sgprs: list[int] = []
    # Track SGPR pairs to prevent individual registers from being freed
    sgpr_pairs: set[int] = set()
    vgpr_pairs: set[int] = set()  # Track which VGPRs are part of pairs
    vgpr_ranges: dict[int, int] = {}  # base -> count for 8-register ranges
    # Track which UOp uses which register (for freeing)
    vgpr_owner: dict[int, UOp] = {}
    sgpr_owner: dict[int, UOp] = {}
    range_owner: dict[int, UOp] = {}  # base -> owner for 8-register ranges
    # Constant deduplication: (dtype, value) -> register
    const_cache: dict[tuple, str] = {}
    # v[0:2] is local_xyz, we start allocating from v3
    next_vgpr = 3
    # s[0:1] is kernarg ptr, s[2:4] is group id xyz, we start from s5
    next_sgpr = 5
    max_vgpr = 3
    max_sgpr = 5

    # Build reverse alias map: owner -> all UOps that alias to it (directly or transitively)
    def get_root_owner(u: UOp) -> UOp:
      while u in aliases:
        u = aliases[u]
      return u

    alias_groups: dict[UOp, list[UOp]] = defaultdict(list)
    for u in aliases:
      root = get_root_owner(u)
      alias_groups[root].append(u)

    # Precompute effective death position for each root owner (max of all aliases' last_use)
    effective_death: dict[UOp, int] = {}
    for root, alias_list in alias_groups.items():
      death_pos = last_use.get(root, -1)
      for alias in alias_list:
        death_pos = max(death_pos, last_use.get(alias, -1))
      effective_death[root] = death_pos

    # Pending register deaths: position -> list of (reg_type, reg_num) to free
    # This avoids O(n²) scanning of all registers at each position
    pending_vgpr_deaths: dict[int, list[int]] = defaultdict(list)
    pending_sgpr_deaths: dict[int, list[int]] = defaultdict(list)
    pending_range_deaths: dict[int, list[int]] = defaultdict(list)  # base register of range

    def schedule_vgpr_death(reg: int, owner: UOp):
      """Schedule a VGPR to be freed AFTER its owner's last use (death_pos + 1)"""
      root = get_root_owner(owner)
      death_pos = effective_death.get(root, last_use.get(owner, -1))
      if death_pos >= 0:
        pending_vgpr_deaths[death_pos + 1].append(reg)  # +1: free AFTER last use

    def schedule_sgpr_death(reg: int, owner: UOp):
      """Schedule an SGPR to be freed AFTER its owner's last use (death_pos + 1)"""
      root = get_root_owner(owner)
      death_pos = effective_death.get(root, last_use.get(owner, -1))
      if death_pos >= 0:
        pending_sgpr_deaths[death_pos + 1].append(reg)  # +1: free AFTER last use

    def schedule_range_death(base: int, owner: UOp):
      """Schedule an 8-VGPR range to be freed AFTER its owner's last use (death_pos + 1)"""
      root = get_root_owner(owner)
      death_pos = effective_death.get(root, last_use.get(owner, -1))
      if death_pos >= 0:
        pending_range_deaths[death_pos + 1].append(base)  # +1: free AFTER last use

    # === IDENTIFY COMPILE-TIME CONSTANTS ===
    # Constants don't need VGPRs if they're only used in contexts where literals are allowed:
    # 1. REG indices (compile-time array indices)
    # 2. ADD/MUL operands (RDNA3 supports 32-bit literal operands)
    # First count all uses of each CONST
    const_use_count: dict[UOp, int] = defaultdict(int)
    reg_index_const_uses: dict[UOp, int] = defaultdict(int)
    add_mul_const_uses: dict[UOp, int] = defaultdict(int)  # Constants used in ADD/MUL (can use literals)
    store_const_uses: set[UOp] = set()  # Constants used in STORE (must have VGPR)
    vectorize_const_uses: set[UOp] = set()  # Constants used in VECTORIZE sources (must have VGPR)
    for u in uops:
      for src in u.src:
        if src.op is Ops.CONST:
          const_use_count[src] += 1
      # Track uses as REG indices specifically
      if u.op is Ops.INDEX and len(u.src) > 1:
        buf = u.src[0]
        idx = u.src[1]
        if isinstance(buf.dtype, PtrDType) and buf.dtype.addrspace == AddrSpace.REG:
          if idx.op is Ops.CONST:
            reg_index_const_uses[idx] += 1
      # Track constants used in ADD/MUL - can use 32-bit literals instead of VGPRs
      if u.op in {Ops.ADD, Ops.MUL}:
        for src in u.src:
          if src.op is Ops.CONST:
            add_mul_const_uses[src] += 1
      # Track constants used in STORE - these MUST have VGPRs (can't use immediate in store data)
      if u.op is Ops.STORE and len(u.src) >= 2:
        if u.src[1].op is Ops.CONST:
          store_const_uses.add(u.src[1])
      # Track constants used in VECTORIZE sources - these MUST have VGPRs for v_mov_b32
      if u.op is Ops.VECTORIZE:
        for src in u.src:
          if src.op is Ops.CONST:
            vectorize_const_uses.add(src)
    # Skip allocation for constants that are ONLY used in literal-allowed contexts
    skip_alloc_consts: set[UOp] = set()
    for const_uop, reg_uses in reg_index_const_uses.items():
      if reg_uses == const_use_count[const_uop]:
        skip_alloc_consts.add(const_uop)
    # Also skip constants only used in ADD/MUL (use literals instead)
    for const_uop, add_mul_uses in add_mul_const_uses.items():
      if add_mul_uses == const_use_count[const_uop] and const_uop not in store_const_uses and const_uop not in vectorize_const_uses:
        skip_alloc_consts.add(const_uop)

    def free_dead_regs(pos: int):
      """Free registers scheduled to die at position pos - O(1) lookup instead of O(n) scan"""
      nonlocal free_vgprs, free_vgpr_pairs, free_vgpr_ranges, free_sgprs
      # Free ranges scheduled to die at this position
      for base in pending_range_deaths.get(pos, []):
        if base in range_owner:
          del range_owner[base]
          count = vgpr_ranges.pop(base, 8)
          free_vgpr_ranges.append((base, count))
      # Free VGPRs scheduled to die at this position
      dead_vgprs = pending_vgpr_deaths.get(pos, [])
      dead_vgprs_set = set(dead_vgprs)
      for reg in dead_vgprs:
        if reg not in vgpr_owner: continue  # Already freed (e.g., as part of pair)
        owner = vgpr_owner[reg]
        del vgpr_owner[reg]
        # If this is part of a pair, check if both regs are dead and return as pair
        if reg in vgpr_pairs:
          base_reg = reg if reg % 2 == 0 else reg - 1
          other_reg = base_reg + 1 if reg == base_reg else base_reg
          if other_reg in dead_vgprs_set and base_reg not in free_vgpr_pairs:
            # Both regs of pair are dead - return as pair
            free_vgpr_pairs.append(base_reg)
            vgpr_pairs.discard(base_reg)
            vgpr_pairs.discard(other_reg)
            # Also delete other_reg from vgpr_owner to prevent double-free
            if other_reg in vgpr_owner: del vgpr_owner[other_reg]
            if getenv('DEBUG_RDNA_REG'): print(f"  pos {pos}: freed pair v[{base_reg}:{base_reg+1}] (owner: {owner.op})")
          # Don't add to free_vgprs - pairs are handled separately
        else:
          free_vgprs.append(reg)
          if getenv('DEBUG_RDNA_REG'): print(f"  pos {pos}: freed v{reg} (owner: {owner.op})")
      # Free SGPRs scheduled to die at this position
      for reg in pending_sgpr_deaths.get(pos, []):
        if reg not in sgpr_owner: continue  # Already freed
        # Don't free SGPRs that are part of a pair (used for 64-bit values like buffer addresses)
        if reg in sgpr_pairs:
          continue
        del sgpr_owner[reg]
        free_sgprs.append(reg)

    def alloc_vgpr(owner: UOp) -> str:
      nonlocal next_vgpr, max_vgpr
      if free_vgprs:
        reg = free_vgprs.pop()
        if getenv('DEBUG_RDNA_REG'): print(f"    alloc_vgpr({owner.op}): reused v{reg} from free_vgprs={free_vgprs}")
      elif free_vgpr_ranges:
        # Take one VGPR from a free range and put the remainder back
        base, count = free_vgpr_ranges.pop()
        reg = base
        if count > 1:
          free_vgpr_ranges.append((base + 1, count - 1))
        if getenv('DEBUG_RDNA_REG'): print(f"    alloc_vgpr({owner.op}): took v{reg} from range")
      else:
        reg = next_vgpr
        next_vgpr += 1
        max_vgpr = max(max_vgpr, next_vgpr)
        if getenv('DEBUG_RDNA_REG'): print(f"    alloc_vgpr({owner.op}): new v{reg}")
      vgpr_owner[reg] = owner
      schedule_vgpr_death(reg, owner)
      return f"v{reg}"

    def alloc_vgpr_pair(owner: UOp) -> str:
      """Allocate an aligned pair of VGPRs for 64-bit values (float64, int64, uint64)"""
      nonlocal next_vgpr, max_vgpr
      # Try to reuse a free pair first
      if free_vgpr_pairs:
        reg = free_vgpr_pairs.pop()
      else:
        # Align to even for 64-bit values
        if next_vgpr % 2 != 0:
          next_vgpr += 1
        reg = next_vgpr
        next_vgpr += 2
        max_vgpr = max(max_vgpr, next_vgpr)
      vgpr_owner[reg] = owner
      vgpr_owner[reg+1] = owner
      vgpr_pairs.add(reg)
      vgpr_pairs.add(reg+1)
      schedule_vgpr_death(reg, owner)
      schedule_vgpr_death(reg+1, owner)
      return f"v[{reg}:{reg+1}]"

    def needs_vgpr_pair(dtype: DType) -> bool:
      """Check if a dtype needs a VGPR pair (64-bit types)"""
      # 64-bit types need register pairs for load/store
      # Note: ALU operations will extract low 32 bits for int64/uint64 since RDNA has limited 64-bit int ALU
      return dtype in (dtypes.float64, dtypes.long, dtypes.ulong) or (hasattr(dtype, 'itemsize') and dtype.itemsize == 8)

    def alloc_sgpr(owner: UOp) -> str|None:
      nonlocal next_sgpr, max_sgpr
      if free_sgprs:
        reg = free_sgprs.pop()
      elif next_sgpr < MAX_SGPR:
        reg = next_sgpr
        next_sgpr += 1
        max_sgpr = max(max_sgpr, next_sgpr)
      else:
        return None  # SGPR exhausted, caller should fall back to VGPR
      sgpr_owner[reg] = owner
      schedule_sgpr_death(reg, owner)
      return f"s{reg}"

    def alloc_sgpr_pair(owner: UOp) -> str:
      """Allocate an aligned pair of SGPRs for 64-bit values"""
      nonlocal next_sgpr, max_sgpr
      # Align to even
      if next_sgpr % 2 != 0:
        next_sgpr += 1
      reg = next_sgpr
      next_sgpr += 2
      max_sgpr = max(max_sgpr, next_sgpr)
      sgpr_owner[reg] = owner
      sgpr_owner[reg+1] = owner
      # Mark these as part of a pair - never free individually
      sgpr_pairs.add(reg)
      sgpr_pairs.add(reg+1)
      return f"s[{reg}:{reg+1}]"

    def alloc_vgpr_range(owner: UOp, count: int = 8) -> str:
      """Allocate a contiguous range of VGPRs (for WMMA/VECTORIZE)"""
      nonlocal next_vgpr, max_vgpr
      # Try to reuse a free range of sufficient size
      for i, (base, range_count) in enumerate(free_vgpr_ranges):
        if range_count >= count:
          free_vgpr_ranges.pop(i)
          # If range is larger than needed, put remainder back
          if range_count > count:
            free_vgpr_ranges.append((base + count, range_count - count))
          range_owner[base] = owner
          vgpr_ranges[base] = count
          schedule_range_death(base, owner)
          return f"v[{base}:{base+count-1}]"
      # No free range - allocate fresh
      base = next_vgpr
      if base % 2 != 0:  # Align to even for better access
        base = next_vgpr = next_vgpr + 1
      next_vgpr = base + count
      max_vgpr = max(max_vgpr, next_vgpr)
      range_owner[base] = owner
      vgpr_ranges[base] = count
      schedule_range_death(base, owner)
      return f"v[{base}:{base+count-1}]"

    def get_scratch_vgpr(count:int=1) -> int:
      """Get or allocate scratch VGPRs for temporary operations. Returns the base register number."""
      nonlocal next_vgpr, max_vgpr
      if self.scratch_vgpr < 0:
        # Allocate scratch VGPRs (not tracked in owner map, stays allocated)
        # IDIV uses s, s+1, s+2, s+3 (4 registers) for float temps and abs value computation
        self.scratch_vgpr = next_vgpr
        next_vgpr += 4  # Allocate 4 scratch registers for IDIV etc.
        max_vgpr = max(max_vgpr, next_vgpr)
      return self.scratch_vgpr

    def get_deferred_store_vgpr() -> str:
      """Get or allocate a scratch VGPR for deferred store address computation. Never freed."""
      nonlocal next_vgpr, max_vgpr
      if not hasattr(self, '_deferred_store_vgpr') or self._deferred_store_vgpr < 0:
        # Allocate a dedicated VGPR for deferred store addresses (never freed, won't conflict with INDEX regs)
        self._deferred_store_vgpr = next_vgpr
        next_vgpr += 1
        max_vgpr = max(max_vgpr, next_vgpr)
      return f"v{self._deferred_store_vgpr}"

    # Make get_scratch_vgpr available to pattern matcher via self
    self.get_scratch_vgpr = get_scratch_vgpr

    name = "test"
    pending_waits = set()  # track which ops need waits

    # === LOOK-AHEAD PACKING FOR HALF16 VECTORIZE ===
    # Pre-scan to find LOADs that will be packed into half16 VECTORIZE
    # Key optimization: half.vec(4) LOADs can go directly into half16 destination range
    half16_vectorize_sources: dict[UOp, tuple[UOp, int]] = {}  # source -> (vectorize_uop, position)
    half16_vectorize_ranges: dict[UOp, str] = {}  # vectorize_uop -> allocated v[base:end] range
    half16_packed: dict[UOp, set[int]] = {}  # vectorize_uop -> set of packed VGPR indices (0-7)
    half16_temp_regs: dict[UOp, str] = {}  # source_uop -> temp VGPR holding the loaded value
    # Map half.vec(4) LOADs directly to their destination in half16 range
    half16_direct_loads: dict[UOp, tuple[UOp, int]] = {}  # load -> (vectorize_uop, base_vgpr_idx)

    for u in uops:
      if u.op is Ops.VECTORIZE and u.dtype.scalar() == dtypes.half and u.dtype.count == 16:
        # This is a half16 VECTORIZE for WMMA input
        # Map each source to its position
        for pos, src in enumerate(u.src):
          half16_vectorize_sources[src] = (u, pos)
        half16_packed[u] = set()
        # Pre-allocate the 8-VGPR range NOW, before scalar temp allocations inflate next_vgpr
        half16_vectorize_ranges[u] = alloc_vgpr_range(u, 8)

        # Check if sources are GEPs from half.vec(4) LOADs - can load directly into destination
        # Group sources by their underlying LOAD
        load_positions: dict[UOp, list[tuple[int, int]]] = {}  # load -> [(pos, gep_idx), ...]
        for pos, src in enumerate(u.src):
          if src.op is Ops.GEP and src.src[0].op is Ops.LOAD:
            load_op = src.src[0]
            if hasattr(load_op.dtype, 'count') and load_op.dtype.count == 4 and load_op.dtype.scalar() == dtypes.half:
              gep_idx = src.arg[0] if isinstance(src.arg, tuple) else src.arg
              if load_op not in load_positions:
                load_positions[load_op] = []
              load_positions[load_op].append((pos, gep_idx))

        # For LOADs that feed exactly 4 consecutive positions (0,1,2,3 -> vgprs 0,1), mark for direct load
        for load_op, positions in load_positions.items():
          if len(positions) == 4:
            positions.sort(key=lambda x: x[0])
            pos_list = [p for p, _ in positions]
            gep_list = [g for _, g in positions]
            # Check if positions are consecutive groups of 4 and GEP indices are 0,1,2,3
            if gep_list == [0, 1, 2, 3] and pos_list[0] % 4 == 0:
              # This LOAD can go directly into half16 range
              # positions 0-3 -> vgprs 0-1, positions 4-7 -> vgprs 2-3, etc.
              base_vgpr_idx = pos_list[0] // 2  # First of the 2 VGPRs this LOAD fills
              half16_direct_loads[load_op] = (u, base_vgpr_idx)
              # Mark all VGPRs as packed (since LOAD writes them directly)
              half16_packed[u].add(base_vgpr_idx)
              half16_packed[u].add(base_vgpr_idx + 1)

    # === DEFERRED STORE ADDRESS COMPUTATION ===
    # Pre-scan to identify INDEX ops that are ONLY used by global STOREs
    # These can defer address allocation to save VGPRs during computation
    store_only_indices: set[UOp] = set()
    index_users: dict[UOp, list[UOp]] = {}  # INDEX -> list of users

    # First pass: collect all INDEX ops for global memory and track users
    for u in uops:
      if u.op is Ops.INDEX and len(u.src) >= 1:
        src0_dtype = u.src[0].dtype
        # Check if src[0] is a global buffer (PtrDType with GLOBAL addrspace)
        if isinstance(src0_dtype, PtrDType) and src0_dtype.addrspace == AddrSpace.GLOBAL:
          index_users[u] = []
      # Track users of INDEX ops
      for src in u.src:
        if src in index_users:
          index_users[src].append(u)

    # Second pass: identify INDEX ops used ONLY by STOREs (not LOADs)
    for idx_op, users in index_users.items():
      if not users:
        continue
      # Check if all users are STOREs
      all_stores = all(u.op is Ops.STORE for u in users)
      if all_stores:
        store_only_indices.add(idx_op)

    # Third pass: identify SHL(ADD(base, const), shift) ops used ONLY by store-only indices
    # These can be recomputed inline at store time, saving VGPRs
    recomputable_shls: set[UOp] = set()
    uop_users: dict[UOp, list[UOp]] = {}  # Track UOp users
    for u in uops:
      for src in u.src:
        if src not in uop_users:
          uop_users[src] = []
        uop_users[src].append(u)
    # Track base UOps that need extended liveness (for recomputation at store time)
    recompute_base_uops: set[UOp] = set()
    for idx_op in store_only_indices:
      if len(idx_op.src) > 1:
        byte_offset = idx_op.src[1]
        # Check if it's SHL(ADD(a, b), shift) - handles both ADD(base, CONST) and ADD(reg, reg) patterns
        if (byte_offset.op is Ops.SHL and len(byte_offset.src) == 2 and
            byte_offset.src[0].op is Ops.ADD and byte_offset.src[1].op is Ops.CONST):
          add_op = byte_offset.src[0]
          # Check if this SHL is ONLY used by store-only indices
          shl_users = uop_users.get(byte_offset, [])
          if all(user in store_only_indices for user in shl_users):
            recomputable_shls.add(byte_offset)
            # Check if ADD has a constant operand (either position) - these can skip allocation
            if len(add_op.src) == 2 and (add_op.src[0].op is Ops.CONST or add_op.src[1].op is Ops.CONST):
              # Also mark the ADD as skippable if only used by this SHL
              add_users = uop_users.get(add_op, [])
              if all(user == byte_offset or user in store_only_indices for user in add_users):
                recomputable_shls.add(add_op)
              # Track the base UOp (non-constant operand of ADD) for liveness extension
              base_uop = add_op.src[1] if add_op.src[0].op is Ops.CONST else add_op.src[0]
              recompute_base_uops.add(base_uop)
            else:
              # ADD(reg, reg) pattern - need to extend liveness of BOTH operands
              for src in add_op.src:
                if src.op is not Ops.CONST:
                  recompute_base_uops.add(src)

    # Extend liveness of base UOps to the end of the kernel (they're needed for store-time recomputation)
    if recompute_base_uops:
      max_uop_pos = len(uops) - 1
      for base_uop in recompute_base_uops:
        last_use[base_uop] = max_uop_pos

    # Shared temp VGPR for deferred store address computation (allocated lazily, reused for all stores)
    deferred_store_addr_vgpr: str | None = None

    def get_half16_range(vec_uop: UOp) -> str:
      """Lazily allocate the 8-VGPR range for a half16 VECTORIZE"""
      if vec_uop not in half16_vectorize_ranges:
        half16_vectorize_ranges[vec_uop] = alloc_vgpr_range(vec_uop, 8)
      return half16_vectorize_ranges[vec_uop]

    for i, u in enumerate(uops):
      # Free registers that are no longer needed
      free_dead_regs(i)
      if u.op in {Ops.NOOP, Ops.GROUP}: continue
      if u.op is Ops.AFTER:
        r[u] = r[u.src[0]]
        continue
      if u.op is Ops.SINK:
        if u.arg is not None: name = u.arg.function_name
        continue
      if u.op is Ops.VECTORIZE:
        # Check if any source needs a wait (from pending loads)
        for src in u.src:
          if src in pending_waits:
            kernel.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
            pending_waits.clear()
            break
        # For WMMA inputs (half16), we need contiguous packed VGPRs (2 halfs per 32-bit VGPR)
        if u.dtype.scalar() == dtypes.half and u.dtype.count == 16:
          if u in half16_vectorize_ranges:
            r[u] = half16_vectorize_ranges[u]
          else:
            r[u] = alloc_vgpr_range(u, 8)
          base = get_reg_base(r[u])
          for j in range(8):
            if j not in half16_packed.get(u, set()):
              kernel.append(f"v_pack_b32_f16 v{base+j}, {r[u.src[j*2]]}, {r[u.src[j*2+1]]}")
        # For float8 (WMMA accumulator), check if sources are contiguous
        elif u.dtype.scalar() == dtypes.float and u.dtype.count == 8:
          src_regs = [int(r[src][1:]) for src in u.src if isinstance(r[src], str) and r[src].startswith('v') and '[' not in r[src]]
          if len(src_regs) == 8 and src_regs == list(range(src_regs[0], src_regs[0] + 8)):
            r[u] = f"v[{src_regs[0]}:{src_regs[0]+7}]"
            continue
          r[u] = alloc_vgpr_range(u, 8)
          base = get_reg_base(r[u])
          for j, src in enumerate(u.src):
            kernel.append(f"v_mov_b32 v{base+j}, {r[src]}")
        # For other vector types, allocate contiguous VGPRs based on size
        elif isinstance(u.dtype, DType) and u.dtype.count > 1:
          vgpr_count = (u.dtype.itemsize + 3) // 4
          r[u] = alloc_vgpr_range(u, vgpr_count)
          base = get_reg_base(r[u])
          # Check if this is a packed half type (2 halfs per VGPR)
          if u.dtype.scalar() == dtypes.half and u.dtype.count > 1:
            # Pack pairs of half values into each VGPR using v_pack_b32_f16
            for j in range(vgpr_count):
              lo_idx = j * 2
              hi_idx = j * 2 + 1
              lo_reg = r[u.src[lo_idx]] if lo_idx < len(u.src) else "0"
              hi_reg = r[u.src[hi_idx]] if hi_idx < len(u.src) else "0"
              kernel.append(f"v_pack_b32_f16 v{base+j}, {lo_reg}, {hi_reg}")
          else:
            for j, src in enumerate(u.src):
              src_reg = r[src]
              if isinstance(src_reg, str) and src_reg.startswith('v['):
                src_base, src_end = get_reg_base(src_reg), int(src_reg[src_reg.index(':')+1:-1])
                for k in range(src_end - src_base + 1):
                  kernel.append(f"v_mov_b32 v{base+j+k}, v{src_base+k}")
              else:
                kernel.append(f"v_mov_b32 v{base+j}, {src_reg}")
        else:
          r[u] = [cast(str,r[x]) for x in u.src]
        continue
      if u.op is Ops.GEP:
        # Check if source needs wait (from pending loads)
        if u.src[0] in pending_waits:
          kernel.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
          pending_waits.clear()
        src_reg = r[u.src[0]]
        idx = get_single_element(u.arg)
        src_dtype = u.src[0].dtype

        # Check if source LOAD is a direct-load into half16 range
        if u.src[0] in half16_direct_loads:
          vec_uop, base_vgpr_idx = half16_direct_loads[u.src[0]]
          range_base = get_reg_base(half16_vectorize_ranges[vec_uop])
          r[u] = f"v{range_base + base_vgpr_idx + idx // 2}"
          continue

        if isinstance(src_reg, str) and src_reg.startswith('v['):
          base = get_reg_base(src_reg)
          end = int(src_reg[src_reg.index(':')+1:-1])
          num_vgprs = end - base + 1
          # Check if this is a packed type (multiple elements per VGPR)
          if isinstance(src_dtype, DType) and src_dtype.count > 1:
            elements_per_vgpr = src_dtype.count // num_vgprs
            if elements_per_vgpr > 1:
              # Packed type (e.g., half4 = 4 elements in 2 VGPRs)
              vgpr_idx = idx // elements_per_vgpr
              element_in_vgpr = idx % elements_per_vgpr
              src_vgpr = f"v{base + vgpr_idx}"
              if element_in_vgpr == 0:
                # Low bits - can use directly
                r[u] = src_vgpr
              else:
                # High bits - need to extract with shift
                dst = alloc_vgpr(u)
                shift_amount = element_in_vgpr * (32 // elements_per_vgpr)
                kernel.append(f"v_lshrrev_b32 {dst}, {shift_amount}, {src_vgpr}")
                r[u] = dst
            else:
              # 1:1 mapping (e.g., float4 = 4 elements in 4 VGPRs)
              r[u] = f"v{base + idx}"
          else:
            r[u] = f"v{base + idx}"
        elif isinstance(src_reg, list):
          r[u] = src_reg[idx]
        else:
          # Single register - GEP index must be 0
          r[u] = src_reg
        continue
      if u.op in {Ops.CAST, Ops.BITCAST} and (u.src[0].dtype == u.dtype or isinstance(u.src[0].dtype, PtrDType)):
        r[u] = r[u.src[0]]
        continue

      # Register allocation with liveness-based reuse
      if u.op is Ops.DEFINE_GLOBAL:
        r[u] = alloc_sgpr_pair(u)
        kernarg_offset[u] = current_kernarg_offset
        current_kernarg_offset += 8  # Pointers are 8 bytes
        bufs.append((f"data{u.arg}", u.dtype))
      elif u.op is Ops.DEFINE_LOCAL:
        # Local memory - DEFINE_LOCAL.dtype contains the LDS size in the ptr size
        lds_size = u.dtype.size * u.dtype.itemsize if isinstance(u.dtype, PtrDType) else 0
        self.lds_size = max(getattr(self, 'lds_size', 0), lds_size)
        r[u] = u.arg  # Store the offset (arg is the offset in bytes)
        continue
      elif u.op is Ops.DEFINE_VAR:
        sgpr = alloc_sgpr(u)
        kernarg_offset[u] = current_kernarg_offset
        current_kernarg_offset += 4  # Variables are 4 bytes (int32)
        r[u] = sgpr if sgpr else alloc_vgpr(u)  # Fall back to VGPR if SGPRs exhausted
        bufs.append((u.arg[0], u.dtype))
      elif u.op is Ops.SPECIAL:
        r[u] = alloc_vgpr(u)
      elif u.op is Ops.INDEX:
        # REG addrspace INDEX: index into register array from DEFINE_REG
        if isinstance(u.src[0].dtype, PtrDType) and u.src[0].dtype.addrspace == AddrSpace.REG:
          assert u.src[1].op == Ops.CONST, f"REG INDEX requires CONST index, not {u.src[1].op}"
          r[u] = r[u.src[0]][u.src[1].arg]  # Get specific register from array
          continue  # Skip rendering - handled inline
        # Skip VGPR allocation for store-only indices - address computed inline at store time
        if u in store_only_indices:
          r[u] = "DEFERRED_STORE_ADDR"
        else:
          r[u] = alloc_vgpr(u)
      elif u.op is Ops.LOAD:
        # REG addrspace LOAD: handled by REG handling code below, skip allocation
        if u.src[0].op is Ops.INDEX and isinstance(u.src[0].src[0].dtype, PtrDType) and u.src[0].src[0].dtype.addrspace == AddrSpace.REG:
          r[u] = r[u.src[0]]  # Use register from INDEX (which came from DEFINE_REG array)
          continue  # Skip rendering - handled inline
        if u in half16_direct_loads:
          vec_uop, base_vgpr_idx = half16_direct_loads[u]
          range_base = get_reg_base(half16_vectorize_ranges[vec_uop])
          r[u] = f"v[{range_base + base_vgpr_idx}:{range_base + base_vgpr_idx + 1}]"
        elif isinstance(u.dtype, DType) and u.dtype.count > 1:
          # Calculate number of 32-bit VGPRs needed
          vgpr_count = (u.dtype.itemsize + 3) // 4  # Round up to 32-bit chunks
          r[u] = alloc_vgpr_range(u, vgpr_count)
        elif u in half16_vectorize_sources:
          # Scalar half LOAD destined for half16 VECTORIZE - use look-ahead packing
          # Allocate a temp VGPR for the load, will pack immediately after
          r[u] = alloc_vgpr(u)
          half16_temp_regs[u] = r[u]
        else:
          r[u] = alloc_vgpr_pair(u) if needs_vgpr_pair(u.dtype) else alloc_vgpr(u)
        pending_waits.add(u)
      elif u.op is Ops.CONST:
        # Skip allocation for constants used as compile-time REG indices
        if u in skip_alloc_consts:
          r[u] = render_val(u.arg, u.dtype)  # Store the value as string, no register needed
          continue  # Skip rendering - it's a compile-time index
        # Check if constant can be inlined (no VGPR needed) - but not if used in STORE
        if can_inline_const(u.arg, u.dtype) and u not in store_const_uses:
          r[u] = render_val(u.arg, u.dtype)  # Store value string, not register
          continue  # Skip rendering - will be inlined
        # Deduplicate non-inlineable constants - reuse register if same value already loaded
        const_key = (u.dtype, u.arg)
        if const_key in const_cache:
          r[u] = const_cache[const_key]
          # Extend the original owner's lifetime to include this use
          # by updating last_use for the original constant
          reg_str = const_cache[const_key]
          reg_num = int(reg_str[1:]) if reg_str.startswith('v') and '[' not in reg_str else None
          if reg_num is not None and reg_num in vgpr_owner:
            original_owner = vgpr_owner[reg_num]
            # Extend last_use to include all uses of this UOp
            last_use[original_owner] = max(last_use.get(original_owner, -1), last_use.get(u, i))
          continue  # Skip rendering - already loaded
        # For 64-bit types used in STORE, need a pair for global_store_b64
        needs_pair = needs_vgpr_pair(u.dtype) or (u in store_const_uses and u.dtype.itemsize == 8)
        r[u] = alloc_vgpr_pair(u) if needs_pair else alloc_vgpr(u)
        const_cache[const_key] = r[u]
      elif u.op is Ops.RANGE:
        r[u] = alloc_vgpr(u)
      elif u.op is Ops.END:
        r[u] = "vcc_lo"  # comparison result
      elif u.op in GroupOp.ALU or u.op in {Ops.CAST, Ops.BITCAST}:
        # Skip allocation for SHL/ADD ops that will be recomputed inline at store time
        if u in recomputable_shls:
          r[u] = "RECOMPUTE_AT_STORE"
          continue
        # Check if any source needs a wait
        for src in u.src:
          if src in pending_waits:
            kernel.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
            pending_waits.clear()
            break
        # CAST optimization: reuse source register when this is its last use and dest fits
        # This avoids allocating 128 extra VGPRs for float32→half conversions
        cast_reused_src = False
        if u.op is Ops.CAST and len(u.src) == 1:
          src = u.src[0]
          src_reg = r.get(src)
          # Can reuse if: single VGPR source, this is last use, dest fits in 32 bits
          if (src_reg and isinstance(src_reg, str) and src_reg.startswith('v') and '[' not in src_reg
              and last_use.get(src, -1) == i and isinstance(u.dtype, DType) and u.dtype.itemsize <= 4):
            r[u] = src_reg  # Reuse source register - in-place conversion
            # Mark source as freed since we're taking over its register
            reg_num = int(src_reg[1:])
            if reg_num in vgpr_owner:
              del vgpr_owner[reg_num]
            cast_reused_src = True
        # Only allocate if we didn't already reuse source register for CAST
        if not cast_reused_src:
          # Only direct comparison ops go to SGPR; other bool ops stay in VGPR
          # because their inputs might be in VGPR (from memory loads)
          if u.dtype == dtypes.bool and u.op in {Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ}:
            sgpr = alloc_sgpr(u)
            r[u] = sgpr if sgpr is not None else alloc_vgpr(u)  # fall back to VGPR if SGPRs exhausted
          elif isinstance(u.dtype, DType) and u.dtype.count > 1:
            # Vector types need contiguous register ranges - size based on total bytes
            vgpr_count = (u.dtype.itemsize + 3) // 4  # Round up to 32-bit chunks
            r[u] = alloc_vgpr_range(u, vgpr_count)
          else:
            r[u] = alloc_vgpr_pair(u) if needs_vgpr_pair(u.dtype) else alloc_vgpr(u)
      elif u.op is Ops.IF:
        r[u] = alloc_sgpr(u)
      elif u.op is Ops.WMMA:
        # WMMA outputs a vector of floats (8 for RDNA3)
        # For RDNA3 WMMA, we can do in-place accumulation if dst == C source
        # Check if we can reuse the accumulator register range
        acc_src = u.src[2]  # accumulator input
        acc_reg = r.get(acc_src)
        if isinstance(acc_reg, str) and acc_reg.startswith('v['):
          # Accumulator is already a contiguous range - reuse it for output
          r[u] = acc_reg
        else:
          # Allocate contiguous VGPRs for the output using range allocator
          r[u] = alloc_vgpr_range(u, 8)
      elif u.op is Ops.DEFINE_REG:
        # For 64-bit types, allocate VGPR pairs
        if needs_vgpr_pair(u.ptrdtype.base):
          r[u] = [alloc_vgpr_pair(u) for _ in range(u.ptrdtype.size)]
        else:
          r[u] = [alloc_vgpr(u) for _ in range(u.ptrdtype.size)]
        continue

      # Handle register-based INDEX/LOAD/STORE (accumulator spills)
      if u.op in {Ops.INDEX, Ops.LOAD, Ops.STORE} and isinstance(u.src[0].dtype, PtrDType) and u.src[0].dtype.addrspace == AddrSpace.REG:
        if u.op is Ops.INDEX:
          assert u.src[1].op == Ops.CONST, f"index on REG in rdna only supported on CONST, not {u.src[1].op}"
          r[u] = r[u.src[0]][u.src[1].arg]
        else:
          r[u] = r[u.src[0]]
          if u.op is Ops.STORE:
            dst_reg, src_reg = r[u.src[0]], r[u.src[1]]
            if '[' in dst_reg:
              dst_num, src_num = get_reg_base(dst_reg), get_reg_base(src_reg)
              kernel.extend([f"v_mov_b32 v{dst_num}, v{src_num}", f"v_mov_b32 v{dst_num+1}, v{src_num+1}"])
            else:
              kernel.append(f"v_mov_b32 {dst_reg}, {src_reg}")
        continue

      # Skip INDEX as it's handled inline
      if u.op is Ops.INDEX:
        # If this is a store-only index, skip - address computed inline at store time
        if u in store_only_indices:
          continue

        # The byte offset is already computed at UOp level by rdna_matcher
        # We just need to move it to the allocated register (and optionally add base offset for local)
        buf, idx = u.src[0], u.src[1]
        if len(u.src) > 2:  # gated
          pass  # handled in LOAD
        is_local = isinstance(u.dtype, PtrDType) and u.dtype.addrspace == AddrSpace.LOCAL
        base_offset = r[buf] if is_local else 0  # local memory has base offset from DEFINE_LOCAL

        if idx.op is Ops.CONST and idx.arg == 0:
          if is_local and base_offset:
            kernel.append(f"v_mov_b32 {r[u]}, {base_offset}")
          else:
            kernel.append(f"v_mov_b32 {r[u]}, 0")
        else:
          # Wait for idx if needed
          if idx in pending_waits:
            kernel.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
            pending_waits.clear()
          # idx is already the byte offset (computed at UOp level by rdna_matcher)
          if is_local and base_offset:
            kernel.append(f"v_add_nc_u32 {r[u]}, {base_offset}, {r[idx]}")
          else:
            kernel.append(f"v_mov_b32 {r[u]}, {r[idx]}")
        continue

      # Check if STORE needs a wait for any pending loads
      if u.op is Ops.STORE:
        for src in u.src:
          if src in pending_waits or (src.op is Ops.INDEX and any(s in pending_waits for s in src.src)):
            kernel.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
            pending_waits.clear()
            break

        # Handle deferred store address computation - reuse single shared temp VGPR
        if u.src[0].op is Ops.INDEX and r.get(u.src[0]) == "DEFERRED_STORE_ADDR":
          index_op = u.src[0]
          buf, idx = index_op.src[0], index_op.src[1]
          # Use a dedicated scratch VGPR for all deferred store addresses (never freed, won't conflict)
          if deferred_store_addr_vgpr is None:
            deferred_store_addr_vgpr = get_deferred_store_vgpr()
          # Compute the byte offset - detect pattern SHL(ADD(base, const), shift) for inline recompute
          # Only recompute if idx is marked as RECOMPUTE_AT_STORE - otherwise it has a valid register
          if idx.op is Ops.CONST and idx.arg == 0:
            kernel.append(f"v_mov_b32 {deferred_store_addr_vgpr}, 0")
          elif r.get(idx) == "RECOMPUTE_AT_STORE" and idx.op is Ops.SHL and idx.src[0].op is Ops.ADD and idx.src[1].op is Ops.CONST:
            # Pattern: SHL(ADD(base, offset), shift) - recompute inline to avoid holding SHL result
            add_op = idx.src[0]
            shift_val = idx.src[1].arg
            add_src0, add_src1 = add_op.src[0], add_op.src[1]
            # Handle both ADD(base, const) and ADD(const, base)
            if add_src1.op is Ops.CONST:
              const_val, base_uop = add_src1.arg, add_src0
            elif add_src0.op is Ops.CONST:
              const_val, base_uop = add_src0.arg, add_src1
            else:
              const_val, base_uop = None, None
            if const_val is not None:
              # ADD(base, const) - use v_lshl_add_u32 to compute (base << shift) + (const << shift)
              base_reg = r.get(base_uop)
              if base_reg and isinstance(base_reg, str) and base_reg.startswith('v'):
                kernel.append(f"v_lshl_add_u32 {deferred_store_addr_vgpr}, {base_reg}, {shift_val}, {const_val << shift_val}")
              else:
                # Fallback: copy from pre-computed
                kernel.append(f"v_mov_b32 {deferred_store_addr_vgpr}, {r[idx]}")
            else:
              # ADD(reg, reg) - compute ADD then SHL
              kernel.append(f"v_add_nc_u32 {deferred_store_addr_vgpr}, {r[add_src0]}, {r[add_src1]}")
              kernel.append(f"v_lshlrev_b32 {deferred_store_addr_vgpr}, {shift_val}, {deferred_store_addr_vgpr}")
          else:
            kernel.append(f"v_mov_b32 {deferred_store_addr_vgpr}, {r[idx]}")
          # Update r[index_op] to point to the temp VGPR so pattern matcher can use it
          r[index_op] = deferred_store_addr_vgpr

      # Render the instruction
      if (l:=cast(str|list[str], string_rewrite.rewrite(u, ctx=self))) is None:
        raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
      if isinstance(l, str):
        kernel.append(l)
      else:
        kernel.extend(l)

      # === LOOK-AHEAD PACKING: pack halfs immediately after load ===
      if u.op is Ops.LOAD and u in half16_vectorize_sources:
        vec_uop, pos = half16_vectorize_sources[u]
        vgpr_idx, is_high_half = pos // 2, pos % 2 == 1
        dst_vgpr = f"v{get_reg_base(get_half16_range(vec_uop)) + vgpr_idx}"

        # Find the pair (the other half that shares this destination VGPR)
        pair_pos = pos ^ 1  # XOR to toggle between even/odd
        pair_src = vec_uop.src[pair_pos]

        # Check if pair is already loaded (its temp reg exists in half16_temp_regs and it's been processed)
        pair_loaded = pair_src in half16_temp_regs and pair_src in r

        if pair_loaded:
          # Both halves ready - wait for loads and pack immediately
          kernel.append("s_waitcnt vmcnt(0)")
          pending_waits.discard(u)
          pending_waits.discard(pair_src)

          # Determine which is low and which is high
          if is_high_half:
            lo_reg = half16_temp_regs[pair_src]
            hi_reg = half16_temp_regs[u]
          else:
            lo_reg = half16_temp_regs[u]
            hi_reg = half16_temp_regs[pair_src]

          # Pack two halfs into destination VGPR
          kernel.append(f"v_pack_b32_f16 {dst_vgpr}, {lo_reg}, {hi_reg}")
          half16_packed[vec_uop].add(vgpr_idx)

          # Immediately free the temp VGPRs since values are now in packed destination
          # This is critical for reducing register pressure
          for temp_src in [u, pair_src]:
            temp_reg = half16_temp_regs[temp_src]
            if temp_reg.startswith('v') and '[' not in temp_reg:
              reg_num = int(temp_reg[1:])
              if reg_num in vgpr_owner:
                del vgpr_owner[reg_num]
                free_vgprs.append(reg_num)

      # Add wait after loads
      if u.op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR}:
        kernel.append("s_waitcnt lgkmcnt(0)")

    # Final waitcnt and end program - always wait to ensure stores complete
    kernel.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
    kernel.extend(['s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)', 's_endpgm', 's_code_end'])

    # Compute actual max VGPR from assembly (not allocation high-water mark)
    # This accounts for VGPRs that were allocated but then freed and reused
    import re
    actual_max_vgpr = 3  # v[0:2] are reserved for local_xyz
    for line in kernel:
      for m in re.finditer(r'v(\d+)', line):
        actual_max_vgpr = max(actual_max_vgpr, int(m.group(1)) + 1)
      # Also check for VGPR ranges v[a:b]
      for m in re.finditer(r'v\[(\d+):(\d+)\]', line):
        actual_max_vgpr = max(actual_max_vgpr, int(m.group(2)) + 1)

    # If scratch SGPRs were used, update max_sgpr to include them
    if self.scratch_sgpr_used:
      # gated_sgpr (s100) + IF stack SGPRs (s101, s102, ...)
      max_sgpr = max(max_sgpr, self.gated_sgpr + 1)
      if self.max_if_depth > 0:
        max_sgpr = max(max_sgpr, self.if_sgpr_base + self.max_if_depth)

    return self.render_kernel(kernel, name, bufs, actual_max_vgpr, max_sgpr, uops)
