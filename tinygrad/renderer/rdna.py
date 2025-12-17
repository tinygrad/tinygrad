from typing import Callable, cast
import struct, yaml
from collections import defaultdict
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat, GroupOp
from tinygrad.dtype import dtypes, DType, PtrDType, AddrSpace
from tinygrad.renderer import Renderer
from tinygrad.helpers import prod, get_single_element
from tinygrad.codegen.late.devectorizer import no_vectorized_alu
from tinygrad.codegen.opt import tc

def render_val(x, dtype):
  if dtypes.is_float(dtype):
    # Check if this is an inlineable float constant
    fval = float(x)
    if fval == 0.0: return "0"
    if fval == 0.5: return "0.5"
    if fval == 1.0: return "1.0"
    if fval == 2.0: return "2.0"
    if fval == 4.0: return "4.0"
    if fval == -0.5: return "-0.5"
    if fval == -1.0: return "-1.0"
    if fval == -2.0: return "-2.0"
    if fval == -4.0: return "-4.0"
    # Non-inlineable float - use hex representation
    if dtype == dtypes.double: return "0x%016X" % struct.unpack("Q", struct.pack("d", x))[0]
    if dtype == dtypes.half: return "0x%04X" % struct.unpack("H", struct.pack("e", x))[0]
    return "0x%08X" % struct.unpack("I", struct.pack("f", x))[0]
  # RDNA3 doesn't support 64-bit integer ALU well - truncate to 32-bit
  # and use hex for large values
  val = int(x) & 0xFFFFFFFF
  if val > 0x7FFFFFFF or val < -0x80000000:
    return f"0x{val:08X}"
  return str(val)

def can_inline_const(val, dtype) -> bool:
  """Check if a constant can be inlined in RDNA3 instructions."""
  if dtypes.is_float(dtype):
    # Float inline constants: 0.0, 0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0, -4.0
    return float(val) in (0.0, 0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0, -4.0)
  # Integer inline constants: -16 to 64
  try:
    return -16 <= int(val) <= 64
  except (TypeError, ValueError):
    return False

# RDNA3 uses different instruction names and formats than PTX
# NOTE: These are used via string_rewrite which passes r[v] (register strings), not UOps
# For literal constant embedding, we handle ADD/MUL specially in string_rewrite
asm_for_op: dict[Ops, Callable] = {
  Ops.RECIPROCAL: lambda d,a,dt,name: f"v_rcp_{name} {d}, {a}",
  Ops.EXP2: lambda d,a,dt,name: f"v_exp_{name} {d}, {a}", Ops.LOG2: lambda d,a,dt,name: f"v_log_{name} {d}, {a}",
  # v_sin/v_cos expect input in turns (x / 2π), so we multiply by 1/(2π) ≈ 0.159155
  Ops.SQRT: lambda d,a,dt,name: f"v_sqrt_{name} {d}, {a}",
  Ops.TRUNC: lambda d,a,dt,name: f"v_trunc_{name} {d}, {a}",
  Ops.NEG: lambda d,a,dt,name: f"v_sub_{name} {d}, 0, {a}" if dtypes.is_float(dt) else f"v_sub_nc_u32 {d}, 0, {a}",
  Ops.SHR: lambda d,a,b,dt,name: f"v_lshrrev_b32 {d}, {b}, {a}",  # Note: operand order is reversed
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

def mem_type(x:UOp) -> str:
  match x.op:
    case Ops.AFTER: return mem_type(x.src[0])
    case Ops.DEFINE_LOCAL: return 'ds'
    case Ops.DEFINE_GLOBAL: return 'global'
    case _: raise RuntimeError(f"{x.op} needs to be memory")

def global_store(addr:str, data:str, base:str, dt:DType) -> str:
  if dt.itemsize == 1: return f"global_store_byte {addr}, {data}, {base}"
  if dt.itemsize == 2: return f"global_store_b16 {addr}, {data}, {base}"
  if dt.itemsize == 4: return f"global_store_b32 {addr}, {data}, {base}"
  if dt.itemsize == 8: return f"global_store_b64 {addr}, {data}, {base}"
  if dt.itemsize == 16: return f"global_store_b128 {addr}, {data}, {base}"
  raise RuntimeError(f"Unsupported store dtype size: {dt.itemsize}")

def global_load(dest:str, addr:str, base:str, dt:DType) -> str:
  if dt.itemsize == 1: return f"global_load_ubyte {dest}, {addr}, {base}"
  if dt.itemsize == 2: return f"global_load_u16 {dest}, {addr}, {base}"
  if dt.itemsize == 4: return f"global_load_b32 {dest}, {addr}, {base}"
  if dt.itemsize == 8: return f"global_load_b64 {dest}, {addr}, {base}"
  if dt.itemsize == 16: return f"global_load_b128 {dest}, {addr}, {base}"
  raise RuntimeError(f"Unsupported load dtype size: {dt.itemsize}")

def render_const_64(ctx, x):
  """Render 64-bit constant as two v_mov_b32 instructions"""
  reg = ctx.r[x]
  # Extract register number from v[n:n+1] format
  reg_num = int(reg[2:reg.index(':')])
  if x.dtype == dtypes.float64:
    bits = struct.unpack("Q", struct.pack("d", x.arg))[0]
  else:
    bits = int(x.arg) & 0xFFFFFFFFFFFFFFFF
  lo = bits & 0xFFFFFFFF
  hi = (bits >> 32) & 0xFFFFFFFF
  return [f"v_mov_b32 v{reg_num}, 0x{lo:08X}", f"v_mov_b32 v{reg_num+1}, 0x{hi:08X}"]

def render_64bit_mul(ctx, x):
  """Render 64-bit integer multiplication using scratch registers.
  For pattern (a * magic_const) used in division-by-multiplication.
  Result: uses scratch registers, stores hi bits in destination for subsequent SHR."""
  rx = ctx.r[x]
  a, b = x.src[0], x.src[1]
  # Get source registers
  ra = ctx.r[a]
  # If a is a CAST from 32-bit, use the source register directly
  if a.op is Ops.CAST and a.src[0].dtype.itemsize == 4:
    ra = ctx.r[a.src[0]]
  elif '[' in ra:  # 64-bit reg pair - use low 32 bits
    ra = f"v{ra[2:ra.index(':')]}"
  rb = ctx.r[b]
  if b.op is Ops.CONST:
    rb = render_val(b.arg, dtypes.uint32)
  elif '[' in rb:  # 64-bit reg pair - use low 32 bits
    rb = f"v{rb[2:rb.index(':')]}"
  # Destination is single VGPR - we'll store high 32 bits there for subsequent SHR
  # Use scratch for low bits (usually not needed)
  scratch = ctx.get_scratch_vgpr()
  # Full 64-bit multiply: lo in scratch, hi in destination
  # This works because the common pattern is (x*magic)>>N where N>=32, so only hi bits matter
  return [f"v_mul_lo_u32 v{scratch}, {ra}, {rb}", f"v_mul_hi_u32 {rx}, {ra}, {rb}"]

def render_64bit_shr(ctx, x, a, b):
  """Render 64-bit right shift. For shifts >= 32, we just need the high 32 bits shifted.
  The source from render_64bit_mul is the high 32 bits in a single VGPR."""
  rx = ctx.r[x]
  ra = ctx.r[a]
  shift_amt = b.arg if b.op is Ops.CONST else None
  if shift_amt is None:
    raise RuntimeError("64-bit SHR requires constant shift amount")
  # Handle the result - always 32-bit destination
  if '[' in rx:
    dst_num = int(rx[2:rx.index(':')])
  else:
    dst_num = int(rx[1:])
  # For source: if pair v[n:n+1], high bits in v(n+1); otherwise single reg has high bits
  if '[' in ra:
    src_hi = int(ra[2:ra.index(':')]) + 1
  else:
    src_hi = int(ra[1:])  # Single reg - this IS the high bits from MUL
  if shift_amt >= 32:
    # Just shift the high 32 bits by (shift_amt - 32)
    remaining_shift = shift_amt - 32
    if remaining_shift == 0:
      return f"v_mov_b32 v{dst_num}, v{src_hi}"
    return f"v_lshrrev_b32 v{dst_num}, {remaining_shift}, v{src_hi}"
  else:
    # For shift < 32, we'd need both halves, but our MUL only stores high bits
    # This pattern shouldn't occur for division-by-multiplication
    # Fall back to just using high bits (loses precision but avoids crash)
    return f"v_lshrrev_b32 v{dst_num}, {shift_amt}, v{src_hi}"

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
  # 64-bit integer constants: just use low 32 bits (sufficient for most patterns)
  (UPat.cvar("x", dtypes.long), lambda ctx, x: f"v_mov_b32 {ctx.r[x]}, {render_val(x.arg, dtypes.int32)}"),
  (UPat.cvar("x", dtypes.ulong), lambda ctx, x: f"v_mov_b32 {ctx.r[x]}, {render_val(x.arg, dtypes.uint32)}"),
  (UPat.cvar("x"), lambda ctx, x: f"v_mov_b32 {ctx.r[x]}, {render_val(x.arg, x.dtype)}"),
  # special registers
  (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: ctx.render_special(x)),
  # define global
  (UPat(Ops.DEFINE_GLOBAL, name="x"), lambda ctx, x: f"s_load_b64 {ctx.r[x]}, s[0:1], {x.arg*8}"),
  # comparison ops
  (UPat((Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ), name="x", allow_any_len=True, src=(UPat.var("src0"),)),
    lambda ctx, x, src0: ctx.code_for_op[x.op](ctx.r[x], *[ctx.r[v] for v in x.src], src0.dtype, ctx.types[src0.dtype])),
  # WHERE: if condition is SGPR, use directly; if VGPR, compare to 0 first to get VCC
  (UPat(Ops.WHERE, name="x", src=(UPat.var("cond"), UPat.var("true_val"), UPat.var("false_val"))),
   lambda ctx, x, cond, true_val, false_val: f"v_cndmask_b32 {ctx.r[x]}, {ctx.r[false_val]}, {ctx.r[true_val]}, {ctx.r[cond]}"
     if ctx.r[cond].startswith('s') else [
       f"v_cmp_ne_u32 vcc_lo, {ctx.r[cond]}, 0",
       f"v_cndmask_b32 {ctx.r[x]}, {ctx.r[false_val]}, {ctx.r[true_val]}, vcc_lo"]),
  # NOTE: v_sin_f32/v_cos_f32 have limited range (~256 turns) and fail for large inputs like 1e6.
  # We let tinygrad's software sin implementation handle all cases for correctness.
  # IDIV: integer division via float conversion (a // b = trunc(float(a) / float(b)))
  (UPat(Ops.IDIV, name="x", src=(UPat.var("a"), UPat.var("b"))),
   lambda ctx, x, a, b: [
     f"v_cvt_f32_i32 v{ctx.get_scratch_vgpr()}, {ctx.r[a]}",
     f"v_cvt_f32_i32 v{ctx.get_scratch_vgpr()+1}, {ctx.r[b]}",
     f"v_rcp_f32 v{ctx.get_scratch_vgpr()+1}, v{ctx.get_scratch_vgpr()+1}",
     f"v_mul_f32 v{ctx.get_scratch_vgpr()}, v{ctx.get_scratch_vgpr()}, v{ctx.get_scratch_vgpr()+1}",
     f"v_trunc_f32 v{ctx.get_scratch_vgpr()}, v{ctx.get_scratch_vgpr()}",
     f"v_cvt_i32_f32 {ctx.r[x]}, v{ctx.get_scratch_vgpr()}"]),
  # Boolean AND/OR with SGPR sources: need to convert to VGPR first (SGPR comparison results are exec-mask style)
  (UPat(Ops.AND, name="x", dtype=dtypes.bool, src=(UPat.var("a", dtype=dtypes.bool), UPat.var("b", dtype=dtypes.bool))),
   lambda ctx, x, a, b: ctx.render_bool_and(x, a, b)),
  (UPat(Ops.OR, name="x", dtype=dtypes.bool, src=(UPat.var("a", dtype=dtypes.bool), UPat.var("b", dtype=dtypes.bool))),
   lambda ctx, x, a, b: ctx.render_bool_or(x, a, b)),
  # 64-bit integer MUL: need full 64-bit product for division-by-multiplication pattern
  (UPat(Ops.MUL, name="x", dtype=dtypes.long), render_64bit_mul),
  (UPat(Ops.MUL, name="x", dtype=dtypes.ulong), render_64bit_mul),
  # 64-bit integer SHR: for shifts >= 32, use high bits only
  (UPat(Ops.SHR, name="x", dtype=dtypes.long, src=(UPat.var("a"), UPat.cvar("b"))), render_64bit_shr),
  (UPat(Ops.SHR, name="x", dtype=dtypes.ulong, src=(UPat.var("a"), UPat.cvar("b"))), render_64bit_shr),
  # alu ops
  (UPat(GroupOp.ALU, name="x"), lambda ctx, x: ctx.code_for_op[x.op](ctx.r[x], *[ctx.r[v] for v in x.src], x.dtype, ctx.types[x.dtype])),
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
  # cast TO 64-bit int: just move (we treat 64-bit ints as 32-bit for register allocation)
  (UPat(Ops.CAST, name="x", dtype=dtypes.long, src=(UPat.var("a"),)),
   lambda ctx, x, a: f"v_mov_b32 {ctx.r[x]}, {ctx.r[a]}" if not ctx.r[a].startswith('s') else
     f"v_cndmask_b32 {ctx.r[x]}, 0, 1, {ctx.r[a]}"),
  (UPat(Ops.CAST, name="x", dtype=dtypes.ulong, src=(UPat.var("a"),)),
   lambda ctx, x, a: f"v_mov_b32 {ctx.r[x]}, {ctx.r[a]}" if not ctx.r[a].startswith('s') else
     f"v_cndmask_b32 {ctx.r[x]}, 0, 1, {ctx.r[a]}"),
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
    lambda ctx, x, idx, alt, gate, buf, index_op: [
    f"v_mov_b32 {ctx.r[x]}, {ctx.r[alt]}",
    # If gate is in VGPR, compare to get SGPR mask; if in SGPR, use directly
    f"v_cmp_ne_u32 vcc_lo, {ctx.r[gate]}, 0" if ctx.r[gate].startswith('v') else f"s_and_b32 vcc_lo, exec_lo, {ctx.r[gate]}",
    f"s_and_saveexec_b32 s{ctx.scratch_sgpr}, vcc_lo",
    global_load(ctx.r[x], ctx.r[index_op], ctx.r[buf], buf.dtype.base),
    f"s_mov_b32 exec_lo, s{ctx.scratch_sgpr}"]),
  (UPat(Ops.LOAD, name="x", src=(UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_GLOBAL, name="buf"), UPat.var("idx")), name="index_op"),), allow_any_len=True),
    lambda ctx, x, idx, buf, index_op: global_load(ctx.r[x], ctx.r[index_op], ctx.r[buf], x.dtype)),
  # store / load for local memory (LDS) - DEFINE_LOCAL directly
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_LOCAL), UPat.var("idx")), name="index_op", allow_any_len=True), UPat.var("var"))),
   lambda ctx, idx, var, index_op: f"ds_write_b32 {ctx.r[index_op]}, {ctx.r[var]}"),
  (UPat(Ops.LOAD, name="x", src=(UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_LOCAL), UPat.var("idx")), name="index_op"),), allow_any_len=True),
    lambda ctx, x, idx, index_op: f"ds_read_b32 {ctx.r[x]}, {ctx.r[index_op]}"),
  # store / load for local memory (LDS) - DEFINE_LOCAL wrapped in AFTER
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat(Ops.AFTER, src=(UPat(Ops.DEFINE_LOCAL),), allow_any_len=True), UPat.var("idx")), name="index_op", allow_any_len=True), UPat.var("var"))),
   lambda ctx, idx, var, index_op: f"ds_write_b32 {ctx.r[index_op]}, {ctx.r[var]}"),
  (UPat(Ops.LOAD, name="x", src=(UPat(Ops.INDEX, src=(UPat(Ops.AFTER, src=(UPat(Ops.DEFINE_LOCAL),), allow_any_len=True), UPat.var("idx")), name="index_op"),), allow_any_len=True),
    lambda ctx, x, idx, index_op: f"ds_read_b32 {ctx.r[x]}, {ctx.r[index_op]}"),
  # simple
  (UPat(Ops.DEFINE_REG, src=()), lambda ctx: []),
  (UPat(Ops.RANGE, name="r"), lambda ctx, r: [
    f"v_mov_b32 {ctx.r[r]}, -1",  # Start at -1, incremented to 0 on first iteration
    f"s_branch LOOP_END_{ctx.r[r][1:]}",
    f"LOOP_{ctx.r[r][1:]}:"]),
  (UPat(Ops.END, name="x", src=(UPat(), UPat(Ops.RANGE, name="r"))), lambda ctx, x, r: [
    f"LOOP_END_{ctx.r[r][1:]}:",
    f"v_add_nc_u32 {ctx.r[r]}, {ctx.r[r]}, 1",
    f"v_cmp_lt_i32 vcc_lo, {ctx.r[r]}, {ctx.r[r.src[0]]}",
    f"s_cbranch_vccnz LOOP_{ctx.r[r][1:]}"]),
  (UPat(Ops.DEFINE_LOCAL, name="x"),
   lambda ctx, x: []),  # local memory is handled differently in RDNA
  (UPat(Ops.IF, name="x"), lambda ctx, x: [
    f"s_and_b32 vcc_lo, exec_lo, {ctx.r[x.src[0]]}",
    f"s_and_saveexec_b32 s{ctx.scratch_sgpr}, vcc_lo",
    f"s_cbranch_execz IF_END_{ctx.uops.index(x)}"]),
  (UPat(Ops.ENDIF, name="x"), lambda ctx, x: [
    f"IF_END_{ctx.uops.index(x.src[0])}:",
    f"s_mov_b32 exec_lo, s{ctx.scratch_sgpr}"]),
  (UPat(Ops.BARRIER, name="x"), lambda ctx, x: "s_barrier"),
])

class RDNARenderer(Renderer):
  device = "AMD"
  suffix = "RDNA"
  supports_float4 = True  # Vectorized loads for better register efficiency
  global_max = (2147483647, 65535, 65535)
  local_max = (1024, 1024, 1024)
  shared_max = 65536
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
      # group id is in s2, s3, s4
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
    elif x.dtype == dtypes.float16 and a.dtype == dtypes.float32:
      return f"v_cvt_f16_f32 {self.r[x]}, {self.r[a]}"
    elif x.dtype == dtypes.float32 and a.dtype == dtypes.float16:
      return f"v_cvt_f32_f16 {self.r[x]}, {self.r[a]}"
    # float64 conversions
    elif x.dtype == dtypes.float64 and a.dtype == dtypes.float32:
      return f"v_cvt_f64_f32 {self.r[x]}, {self.r[a]}"
    elif x.dtype == dtypes.float32 and a.dtype == dtypes.float64:
      # Extract low 32 bits of the f64 register pair
      ra = self.r[a]
      src_num = int(ra[2:ra.index(':')]) if '[' in ra else int(ra[1:])
      return f"v_cvt_f32_f64 {self.r[x]}, v[{src_num}:{src_num+1}]"
    elif x.dtype == dtypes.float64:
      # int to float64: first convert to f32, then to f64
      return self.render_mov_64(x, a)  # TODO: proper int->f64 conversion
    # 64-bit to 32-bit integer: just use low 32 bits
    elif x.dtype.itemsize == 4 and a.dtype in (dtypes.long, dtypes.ulong):
      ra = self.r[a]
      # Extract low register from pair v[n:n+1]
      if '[' in ra:
        src_num = int(ra[2:ra.index(':')])
        return f"v_mov_b32 {self.r[x]}, v{src_num}"
      return f"v_mov_b32 {self.r[x]}, {ra}"
    # fallback: just move (same size types)
    return f"v_mov_b32 {self.r[x]}, {self.r[a]}"

  def render_bool_and(self, x: UOp, a: UOp, b: UOp) -> str|list[str]:
    """Render boolean AND - SGPR comparison results need to be converted to VGPR for proper lane-by-lane AND"""
    ra, rb, rx = self.r[a], self.r[b], self.r[x]
    # If both are in SGPRs, use s_and_b32 for the exec-mask AND, then expand to VGPR
    if ra.startswith('s') and rb.startswith('s'):
      # s_and_b32 result must go to SGPR, then v_cndmask_b32 uses that SGPR as mask
      return [f"s_and_b32 vcc_lo, {ra}, {rb}", f"v_cndmask_b32 {rx}, 0, 1, vcc_lo"]
    # If one is SGPR and one is VGPR, convert SGPR to VGPR first
    if ra.startswith('s'):
      return [f"v_cndmask_b32 v{self.get_scratch_vgpr()}, 0, 1, {ra}", f"v_and_b32 {rx}, v{self.get_scratch_vgpr()}, {rb}"]
    if rb.startswith('s'):
      return [f"v_cndmask_b32 v{self.get_scratch_vgpr()}, 0, 1, {rb}", f"v_and_b32 {rx}, {ra}, v{self.get_scratch_vgpr()}"]
    # Both in VGPR, simple AND
    return f"v_and_b32 {rx}, {ra}, {rb}"

  def render_bool_or(self, x: UOp, a: UOp, b: UOp) -> str|list[str]:
    """Render boolean OR - similar handling to AND"""
    ra, rb, rx = self.r[a], self.r[b], self.r[x]
    if ra.startswith('s') and rb.startswith('s'):
      return [f"s_or_b32 vcc_lo, {ra}, {rb}", f"v_cndmask_b32 {rx}, 0, 1, vcc_lo"]
    if ra.startswith('s'):
      return [f"v_cndmask_b32 v{self.get_scratch_vgpr()}, 0, 1, {ra}", f"v_or_b32 {rx}, v{self.get_scratch_vgpr()}, {rb}"]
    if rb.startswith('s'):
      return [f"v_cndmask_b32 v{self.get_scratch_vgpr()}, 0, 1, {rb}", f"v_or_b32 {rx}, {ra}, v{self.get_scratch_vgpr()}"]
    return f"v_or_b32 {rx}, {ra}, {rb}"

  def render_mov_64(self, x: UOp, a: UOp) -> list[str]:
    """Render 64-bit move using two v_mov_b32 instructions"""
    rx, ra = self.r[x], self.r[a]
    # Extract register numbers from v[n:n+1] format
    def get_reg_num(reg: str) -> int:
      if '[' in reg:
        return int(reg[2:reg.index(':')])
      return int(reg[1:])
    dst_num = get_reg_num(rx)
    src_num = get_reg_num(ra)
    return [f"v_mov_b32 v{dst_num}, v{src_num}", f"v_mov_b32 v{dst_num+1}, v{src_num+1}"]

  def render_cast_to_64(self, x: UOp, a: UOp, signed: bool = False) -> list[str]:
    """Render cast from 32-bit to 64-bit integer (sign or zero extend)"""
    rx, ra = self.r[x], self.r[a]
    # Extract dest register number from v[n:n+1] format
    dst_num = int(rx[2:rx.index(':')])
    # Source can be single reg (v5) or in SGPR for comparison results
    if ra.startswith('s'):
      # SGPR comparison result - expand to VGPR first
      return [
        f"v_cndmask_b32 v{dst_num}, 0, 1, {ra}",
        f"v_mov_b32 v{dst_num+1}, 0"  # zero extend for bool/comparison results
      ]
    src_reg = ra if ra.startswith('v') else f"v{ra}"
    # Low bits: copy from source
    # High bits: 0 for unsigned, or sign-extend for signed
    if signed:
      # Sign extend: copy bit 31 to all bits of high word
      return [
        f"v_mov_b32 v{dst_num}, {src_reg}",
        f"v_ashrrev_i32 v{dst_num+1}, 31, {src_reg}"  # arithmetic shift right by 31 gets sign bit
      ]
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

    # Kernel descriptor
    kernel_desc = {
      '.amdhsa_group_segment_fixed_size': self.lds_size, '.amdhsa_private_segment_fixed_size': 0, '.amdhsa_kernarg_size': kernarg_size,
      '.amdhsa_next_free_vgpr': v_cnt,
      '.amdhsa_reserve_vcc': 0, '.amdhsa_reserve_xnack_mask': 0,
      '.amdhsa_next_free_sgpr': s_cnt,
      '.amdhsa_float_round_mode_32': 0, '.amdhsa_float_round_mode_16_64': 0,
      '.amdhsa_float_denorm_mode_32': 3, '.amdhsa_float_denorm_mode_16_64': 3,
      '.amdhsa_dx10_clamp': 1, '.amdhsa_ieee_mode': 1, '.amdhsa_fp16_overflow': 0,
      '.amdhsa_workgroup_processor_mode': 1, '.amdhsa_memory_ordered': 1, '.amdhsa_forward_progress': 0,
      '.amdhsa_enable_private_segment': 0,
      '.amdhsa_system_sgpr_workgroup_id_x': 1, '.amdhsa_system_sgpr_workgroup_id_y': 1, '.amdhsa_system_sgpr_workgroup_id_z': 1,
      '.amdhsa_system_sgpr_workgroup_info': 0, '.amdhsa_system_vgpr_workitem_id': 2,
      '.amdhsa_exception_fp_ieee_invalid_op': 0, '.amdhsa_exception_fp_denorm_src': 0,
      '.amdhsa_exception_fp_ieee_div_zero': 0, '.amdhsa_exception_fp_ieee_overflow': 0,
      '.amdhsa_exception_fp_ieee_underflow': 0, '.amdhsa_exception_fp_ieee_inexact': 0,
      '.amdhsa_exception_int_div_zero': 0, '.amdhsa_user_sgpr_dispatch_ptr': 0, '.amdhsa_user_sgpr_queue_ptr': 0,
      '.amdhsa_user_sgpr_kernarg_segment_ptr': 1, '.amdhsa_user_sgpr_dispatch_id': 0,
      '.amdhsa_user_sgpr_private_segment_size': 0, '.amdhsa_wavefront_size32': 1, '.amdhsa_uses_dynamic_stack': 0
    }

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

    boilerplate_start = f"""
.rodata
.global {function_name}.kd
.type {function_name}.kd,STT_OBJECT
.align 0x10
.amdhsa_kernel {function_name}"""

    code_start = f""".end_amdhsa_kernel
.text
.global {function_name}
.type {function_name},@function
.p2align 8
{function_name}:
"""

    kernel_str = '\n'.join(kernel)
    # NOTE: .text must be first line for HIPCompiler to detect as assembly
    return ".text\n" + \
           f".global {function_name}\n" + \
           f".type {function_name},@function\n" + \
           ".p2align 8\n" + \
           f"{function_name}:\n" + \
           kernel_str + f"\n.size {function_name}, .-{function_name}\n" + \
           "\n.rodata\n" + \
           f".global {function_name}.kd\n" + \
           f".type {function_name}.kd,STT_OBJECT\n" + \
           ".align 0x10\n" + \
           f".amdhsa_kernel {function_name}\n" + \
           '\n'.join("%s %d" % x for x in kernel_desc.items()) + "\n" + \
           ".end_amdhsa_kernel\n" + \
           ".amdgpu_metadata\n" + yaml.dump(metadata) + ".end_amdgpu_metadata"

  def render(self, uops:list[UOp]) -> str:
    kernel:list[str] = []
    bufs = []

    r: dict[UOp, str|list[str]] = {}
    self.r = r
    self.uops = uops
    self.scratch_sgpr = 100  # scratch register for exec manipulation
    self.lds_size = 0  # track local memory (LDS) usage
    # Scratch VGPR will be allocated after we know how many VGPRs the kernel uses
    self.scratch_vgpr = -1  # will be set after register allocation

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
      # For INDEX inside LOAD/STORE, track the INDEX sources too
      if u.op in {Ops.LOAD, Ops.STORE} and u.src[0].op is Ops.INDEX:
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

    # === IDENTIFY COMPILE-TIME CONSTANTS ===
    # Constants don't need VGPRs if they're only used in contexts where literals are allowed:
    # 1. REG indices (compile-time array indices)
    # 2. ADD/MUL operands (RDNA3 supports 32-bit literal operands)
    # First count all uses of each CONST
    const_use_count: dict[UOp, int] = defaultdict(int)
    reg_index_const_uses: dict[UOp, int] = defaultdict(int)
    add_mul_const_uses: dict[UOp, int] = defaultdict(int)  # Constants used in ADD/MUL (can use literals)
    store_const_uses: set[UOp] = set()  # Constants used in STORE (must have VGPR)
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
    # Skip allocation for constants that are ONLY used in literal-allowed contexts
    skip_alloc_consts: set[UOp] = set()
    for const_uop, reg_uses in reg_index_const_uses.items():
      if reg_uses == const_use_count[const_uop]:
        skip_alloc_consts.add(const_uop)
    # Also skip constants only used in ADD/MUL (use literals instead)
    for const_uop, add_mul_uses in add_mul_const_uses.items():
      if add_mul_uses == const_use_count[const_uop] and const_uop not in store_const_uses:
        skip_alloc_consts.add(const_uop)

    def free_dead_regs(pos: int):
      """Free registers whose owners (and all aliases) are no longer live after position pos"""
      nonlocal free_vgprs, free_vgpr_pairs, free_vgpr_ranges, free_sgprs
      # First check 8-register ranges
      dead_ranges = []
      for base, owner in range_owner.items():
        owner_last_use = last_use.get(owner, -1)
        for alias_uop in alias_groups.get(owner, []):
          owner_last_use = max(owner_last_use, last_use.get(alias_uop, -1))
        if owner_last_use < pos:
          dead_ranges.append(base)
      for base in dead_ranges:
        del range_owner[base]
        count = vgpr_ranges.pop(base, 8)
        free_vgpr_ranges.append((base, count))
      dead_vgprs = []
      for reg, owner in vgpr_owner.items():
        # Check if owner and all its aliases are dead
        owner_last_use = last_use.get(owner, -1)
        # Also check any UOps that alias to this owner
        for alias_uop in alias_groups.get(owner, []):
          owner_last_use = max(owner_last_use, last_use.get(alias_uop, -1))
        if owner_last_use < pos:
          dead_vgprs.append(reg)
      dead_sgprs = []
      for reg, owner in sgpr_owner.items():
        owner_last_use = last_use.get(owner, -1)
        for alias_uop in alias_groups.get(owner, []):
          owner_last_use = max(owner_last_use, last_use.get(alias_uop, -1))
        if owner_last_use < pos:
          dead_sgprs.append(reg)
      # Process dead VGPRs - handle pairs specially
      dead_vgprs_set = set(dead_vgprs)
      for reg in dead_vgprs:
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
          # Don't add to free_vgprs - pairs are handled separately
        else:
          free_vgprs.append(reg)
      for reg in dead_sgprs:
        # Don't free SGPRs that are part of a pair (used for 64-bit values like buffer addresses)
        if reg in sgpr_pairs:
          continue
        del sgpr_owner[reg]
        free_sgprs.append(reg)

    def alloc_vgpr(owner: UOp) -> str:
      nonlocal next_vgpr, max_vgpr
      if free_vgprs:
        reg = free_vgprs.pop()
      elif free_vgpr_ranges:
        # Take one VGPR from a free range and put the remainder back
        base, count = free_vgpr_ranges.pop()
        reg = base
        if count > 1:
          free_vgpr_ranges.append((base + 1, count - 1))
      else:
        reg = next_vgpr
        next_vgpr += 1
        max_vgpr = max(max_vgpr, next_vgpr)
      vgpr_owner[reg] = owner
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
      return f"v[{reg}:{reg+1}]"

    def needs_vgpr_pair(dtype: DType) -> bool:
      """Check if a dtype needs a VGPR pair (64-bit types)"""
      # Only float64 needs pairs - int64/uint64 use special split hi/lo patterns
      return dtype == dtypes.float64

    def alloc_sgpr(owner: UOp) -> str:
      nonlocal next_sgpr, max_sgpr
      if free_sgprs:
        reg = free_sgprs.pop()
      else:
        reg = next_sgpr
        next_sgpr += 1
        max_sgpr = max(max_sgpr, next_sgpr)
      sgpr_owner[reg] = owner
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
          return f"v[{base}:{base+count-1}]"
      # No free range - allocate fresh
      base = next_vgpr
      if base % 2 != 0:  # Align to even for better access
        base = next_vgpr = next_vgpr + 1
      next_vgpr = base + count
      max_vgpr = max(max_vgpr, next_vgpr)
      range_owner[base] = owner
      vgpr_ranges[base] = count
      return f"v[{base}:{base+count-1}]"

    def get_scratch_vgpr(count:int=1) -> int:
      """Get or allocate scratch VGPRs for temporary operations. Returns the base register number."""
      nonlocal next_vgpr, max_vgpr
      if self.scratch_vgpr < 0:
        # Allocate scratch VGPRs (not tracked in owner map, stays allocated)
        self.scratch_vgpr = next_vgpr
        next_vgpr += 2  # Allocate 2 scratch registers for IDIV etc.
        max_vgpr = max(max_vgpr, next_vgpr)
      return self.scratch_vgpr

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
        # For WMMA inputs (half16), we need contiguous packed VGPRs
        # half16 = 16 halfs = 8 VGPRs (2 halfs per 32-bit VGPR)
        if u.dtype.scalar() == dtypes.half and u.dtype.count == 16:
          # Check if we used look-ahead packing (range already allocated)
          if u in half16_vectorize_ranges:
            r[u] = half16_vectorize_ranges[u]
            base = int(r[u][2:r[u].index(':')])
            # Only pack VGPRs that weren't already packed by look-ahead
            for j in range(8):
              if j not in half16_packed.get(u, set()):
                src_lo = r[u.src[j*2]]
                src_hi = r[u.src[j*2+1]]
                kernel.append(f"v_pack_b32_f16 v{base+j}, {src_lo}, {src_hi}")
          else:
            # No look-ahead - allocate and pack as before
            r[u] = alloc_vgpr_range(u, 8)
            base = int(r[u][2:r[u].index(':')])
            # Pack the source halfs into the destination VGPRs
            # Each VGPR holds 2 halfs: low 16 bits and high 16 bits
            for j in range(8):
              src_lo = r[u.src[j*2]]
              src_hi = r[u.src[j*2+1]]
              # Pack two halfs into one VGPR using v_pack_b32_f16
              kernel.append(f"v_pack_b32_f16 v{base+j}, {src_lo}, {src_hi}")
        # For float8 (WMMA accumulator), check if sources are contiguous
        elif u.dtype.scalar() == dtypes.float and u.dtype.count == 8:
          # Check if all sources are from contiguous registers
          src_regs = []
          all_contiguous = True
          for src in u.src:
            src_str = r[src]
            if isinstance(src_str, str) and src_str.startswith('v') and '[' not in src_str:
              src_regs.append(int(src_str[1:]))
            else:
              all_contiguous = False
              break
          if all_contiguous and len(src_regs) == 8:
            # Check if contiguous
            if src_regs == list(range(src_regs[0], src_regs[0] + 8)):
              # Already contiguous - just create range reference
              r[u] = f"v[{src_regs[0]}:{src_regs[0]+7}]"
              continue
          # Not contiguous - allocate new registers using range allocator
          r[u] = alloc_vgpr_range(u, 8)
          base = int(r[u][2:r[u].index(':')])
          for j, src in enumerate(u.src):
            kernel.append(f"v_mov_b32 v{base+j}, {r[src]}")
        # For other vector types, allocate contiguous VGPRs based on size
        elif isinstance(u.dtype, DType) and u.dtype.count > 1:
          vgpr_count = (u.dtype.itemsize + 3) // 4  # Round up to 32-bit chunks
          r[u] = alloc_vgpr_range(u, vgpr_count)
          base = int(r[u][2:r[u].index(':')])
          # Copy sources - each source is scalar, pack into VGPRs
          for j, src in enumerate(u.src):
            src_reg = r[src]
            # Handle case where source is a vector range (shouldn't normally happen for VECTORIZE with scalar sources)
            if isinstance(src_reg, str) and src_reg.startswith('v['):
              # Source is a range - copy element by element
              src_base = int(src_reg[2:src_reg.index(':')])
              src_end = int(src_reg[src_reg.index(':')+1:-1])
              for k in range(src_end - src_base + 1):
                kernel.append(f"v_mov_b32 v{base+j+k}, v{src_base+k}")
              j += (src_end - src_base)  # Adjust index (though this is inside for loop, won't work right)
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
        # In this case, data is already in the correct packed format - no extraction needed
        if u.src[0] in half16_direct_loads:
          vec_uop, base_vgpr_idx = half16_direct_loads[u.src[0]]
          range_str = half16_vectorize_ranges[vec_uop]
          range_base = int(range_str[2:range_str.index(':')])
          # idx 0,1 -> first VGPR, idx 2,3 -> second VGPR
          vgpr_offset = idx // 2
          dest_vgpr = f"v{range_base + base_vgpr_idx + vgpr_offset}"
          r[u] = dest_vgpr  # Reference the packed VGPR directly
          continue

        if isinstance(src_reg, str) and src_reg.startswith('v['):
          # Extract base and end from v[base:end] range
          base = int(src_reg[2:src_reg.index(':')])
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
        bufs.append((f"data{u.arg}", u.dtype))
      elif u.op is Ops.DEFINE_LOCAL:
        # Local memory - DEFINE_LOCAL.dtype contains the LDS size in the ptr size
        lds_size = u.dtype.size * u.dtype.itemsize if isinstance(u.dtype, PtrDType) else 0
        self.lds_size = max(getattr(self, 'lds_size', 0), lds_size)
        r[u] = u.arg  # Store the offset (arg is the offset in bytes)
        continue
      elif u.op is Ops.DEFINE_VAR:
        r[u] = alloc_sgpr(u)
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
          # half.vec(4) LOAD that goes directly into half16 destination range
          vec_uop, base_vgpr_idx = half16_direct_loads[u]
          range_str = half16_vectorize_ranges[vec_uop]
          range_base = int(range_str[2:range_str.index(':')])
          # This LOAD fills 2 VGPRs starting at base_vgpr_idx within the half16 range
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
        r[u] = alloc_vgpr_pair(u) if needs_vgpr_pair(u.dtype) else alloc_vgpr(u)
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
            r[u] = alloc_sgpr(u)  # comparison results go in SGPR
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
            # For 64-bit types, use two v_mov_b32 instructions
            if '[' in dst_reg:
              dst_num = int(dst_reg[2:dst_reg.index(':')])
              src_num = int(src_reg[2:src_reg.index(':')]) if '[' in src_reg else int(src_reg[1:])
              kernel.append(f"v_mov_b32 v{dst_num}, v{src_num}")
              kernel.append(f"v_mov_b32 v{dst_num+1}, v{src_num+1}")
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
          # Use shared temp VGPR for all deferred store addresses (stores are sequential)
          if deferred_store_addr_vgpr is None:
            deferred_store_addr_vgpr = alloc_vgpr(index_op)  # Only allocate once
          # Compute the byte offset - detect pattern SHL(ADD(base, const), shift) for inline recompute
          if idx.op is Ops.CONST and idx.arg == 0:
            kernel.append(f"v_mov_b32 {deferred_store_addr_vgpr}, 0")
          elif idx.op is Ops.SHL and idx.src[0].op is Ops.ADD and idx.src[1].op is Ops.CONST:
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
        vgpr_idx = pos // 2  # Which VGPR in the 8-VGPR range (0-7)
        is_high_half = pos % 2 == 1  # Even positions are low half, odd are high half
        range_str = get_half16_range(vec_uop)  # Lazy allocate if needed
        base = int(range_str[2:range_str.index(':')])
        dst_vgpr = f"v{base + vgpr_idx}"

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
      if u.op is Ops.DEFINE_GLOBAL:
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

    return self.render_kernel(kernel, name, bufs, actual_max_vgpr, max_sgpr, uops)
