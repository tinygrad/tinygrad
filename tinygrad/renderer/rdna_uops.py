# RDNA3-specific UOp-level rewrites
# These transformations run before rendering to lower operations without hardware support

from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat, GroupOp
from tinygrad.dtype import dtypes, PtrDType, AddrSpace
from tinygrad.codegen.late.devectorizer import no_vectorized_alu

# *** Fix fast_idiv output when shift >= 32 ***
# fast_idiv generates (x * magic) >> shift expecting 64-bit multiply, but we only have 32-bit.
# When shift >= 32, we need to use 64-bit arithmetic: cast to 64-bit, multiply, shift, cast back.

def _fix_fast_idiv_unsigned(x: UOp, c: UOp, shift: UOp) -> UOp | None:
  """Fix fast_idiv for unsigned: (x * magic) >> shift where shift >= 32."""
  if not (c.op is Ops.CONST and shift.op is Ops.CONST): return None
  s = shift.arg
  if s < 32: return None  # Regular shift, no fix needed
  # fast_idiv already promotes to int64/uint64 for safety - just return None to let it work
  # The 64-bit ops will be properly lowered by other patterns
  if x.dtype in (dtypes.int64, dtypes.uint64, dtypes.long, dtypes.ulong): return None
  # For 32-bit types, use 64-bit arithmetic: cast to uint64, multiply, shift, cast back
  x64 = x.cast(dtypes.uint64)
  m64 = UOp.const(dtypes.uint64, c.arg)
  result = (x64 * m64).alu(Ops.SHR, UOp.const(dtypes.uint64, s))
  return result.cast(x.dtype)

def _fix_fast_idiv_signed(x: UOp, c: UOp, shift: UOp, add: UOp) -> UOp | None:
  """Fix fast_idiv for signed: ((x * magic) >> shift) + correction where shift >= 32.

  Note: fast_idiv uses UNSIGNED multiply semantics even for signed division.
  The magic constant is computed for unsigned division, and sign handling is separate.
  We must use uint64 to avoid the magic being reinterpreted as negative.
  """
  if not (c.op is Ops.CONST and shift.op is Ops.CONST and x.dtype in dtypes.sints): return None
  s = shift.arg
  if s < 32: return None  # Regular shift, no fix needed
  # Use 64-bit UNSIGNED arithmetic: the magic constant must stay positive
  # Cast x to uint64 (zero-extend the bit pattern), multiply, shift, cast back
  x64 = x.bitcast(dtypes.uint32).cast(dtypes.uint64)  # zero-extend the bits
  m64 = UOp.const(dtypes.uint64, c.arg & 0xFFFFFFFF)  # ensure magic is treated as unsigned
  result = (x64 * m64).alu(Ops.SHR, UOp.const(dtypes.uint64, s))
  # Cast back to signed int32 and add the sign correction
  return result.cast(dtypes.uint32).bitcast(x.dtype) + add

# *** UOp-level lowering for operations without hardware support ***
# RDNA3 lacks hardware integer division - lower to float approximation with correction

def _udiv_correction(q: UOp, a: UOp, b: UOp, rcp: UOp) -> UOp:
  """One correction pass for unsigned division: adjust q based on remainder error."""
  r = a - q * b
  rf = r.cast(dtypes.float32)
  adj = (rf * rcp).alu(Ops.TRUNC).cast(dtypes.uint32)
  return q + adj

def lower_udiv(a: UOp, b: UOp) -> UOp:
  """Lower unsigned 32-bit division to float approximation with corrections."""
  af, bf = a.cast(dtypes.float32), b.cast(dtypes.float32)
  rcp = UOp(Ops.RECIPROCAL, dtypes.float32, (bf,))
  q = (af * rcp).alu(Ops.TRUNC).cast(dtypes.uint32)
  for _ in range(3): q = _udiv_correction(q, a, b, rcp)  # correction passes
  r = a - q * b
  return UOp(Ops.WHERE, dtypes.uint32, ((r.alu(Ops.CMPLT, b)).ne(True), q + UOp.const(dtypes.uint32, 1), q))

def lower_umod(a: UOp, b: UOp) -> UOp:
  """Lower unsigned 32-bit modulo: a % b = a - (a // b) * b."""
  q = lower_udiv(a, b)
  return a - q * b

def lower_idiv(a: UOp, b: UOp) -> UOp:
  """Lower signed 32-bit division using unsigned division on absolute values."""
  zero = UOp.const(dtypes.int32, 0)
  a_neg, b_neg = a.alu(Ops.CMPLT, zero), b.alu(Ops.CMPLT, zero)
  a_abs = UOp(Ops.WHERE, dtypes.int32, (a_neg, zero - a, a)).bitcast(dtypes.uint32)
  b_abs = UOp(Ops.WHERE, dtypes.int32, (b_neg, zero - b, b)).bitcast(dtypes.uint32)
  q_abs = lower_udiv(a_abs, b_abs).bitcast(dtypes.int32)
  sign_diff = a_neg ^ b_neg  # result is negative if signs differ (XOR is reliable, ne() is buggy on bools)
  return UOp(Ops.WHERE, dtypes.int32, (sign_diff, zero - q_abs, q_abs))

def lower_imod(a: UOp, b: UOp) -> UOp:
  """Lower signed 32-bit modulo: result has same sign as dividend."""
  zero = UOp.const(dtypes.int32, 0)
  a_neg = a.alu(Ops.CMPLT, zero)
  a_abs = UOp(Ops.WHERE, dtypes.int32, (a_neg, zero - a, a)).bitcast(dtypes.uint32)
  b_abs = UOp(Ops.WHERE, dtypes.int32, (b.alu(Ops.CMPLT, zero), zero - b, b)).bitcast(dtypes.uint32)
  r_abs = lower_umod(a_abs, b_abs).bitcast(dtypes.int32)
  return UOp(Ops.WHERE, dtypes.int32, (a_neg, zero - r_abs, r_abs))

# 64-bit division lowering using f64 arithmetic (more precise than f32)
def _udiv64_correction(q: UOp, a: UOp, b: UOp, rcp: UOp) -> UOp:
  """One correction pass for 64-bit unsigned division."""
  r = a - q * b
  rf = r.cast(dtypes.float64)
  adj = (rf * rcp).alu(Ops.TRUNC).cast(dtypes.uint64)
  return q + adj

def lower_udiv64(a: UOp, b: UOp) -> UOp:
  """Lower unsigned 64-bit division to f64 approximation with corrections."""
  af, bf = a.cast(dtypes.float64), b.cast(dtypes.float64)
  rcp = UOp(Ops.RECIPROCAL, dtypes.float64, (bf,))
  q = (af * rcp).alu(Ops.TRUNC).cast(dtypes.uint64)
  for _ in range(5): q = _udiv64_correction(q, a, b, rcp)  # more correction passes for 64-bit
  r = a - q * b
  # Final adjustment: if r >= b, increment q
  return UOp(Ops.WHERE, dtypes.uint64, (r.alu(Ops.CMPLT, b), q, q + UOp.const(dtypes.uint64, 1)))

def lower_umod64(a: UOp, b: UOp) -> UOp:
  """Lower unsigned 64-bit modulo: a % b = a - (a // b) * b."""
  q = lower_udiv64(a, b)
  return a - q * b

def lower_idiv64(a: UOp, b: UOp) -> UOp:
  """Lower signed 64-bit division using unsigned division on absolute values."""
  zero = UOp.const(dtypes.int64, 0)
  a_neg, b_neg = a.alu(Ops.CMPLT, zero), b.alu(Ops.CMPLT, zero)
  a_abs = UOp(Ops.WHERE, dtypes.int64, (a_neg, zero - a, a)).bitcast(dtypes.uint64)
  b_abs = UOp(Ops.WHERE, dtypes.int64, (b_neg, zero - b, b)).bitcast(dtypes.uint64)
  q_abs = lower_udiv64(a_abs, b_abs).bitcast(dtypes.int64)
  sign_diff = a_neg ^ b_neg
  return UOp(Ops.WHERE, dtypes.int64, (sign_diff, zero - q_abs, q_abs))

# *** Float16/BFloat16 ALU lowering ***
# RDNA3 lacks scalar float16 ALU (only has packed f16), so we convert to f32, operate, convert back
_small_floats = (dtypes.float16, dtypes.bfloat16, dtypes.half)

def _lower_f16_add(a: UOp, b: UOp, x: UOp) -> UOp:
  return (a.cast(dtypes.float32) + b.cast(dtypes.float32)).cast(x.dtype)

def _lower_f16_sub(a: UOp, b: UOp, x: UOp) -> UOp:
  return (a.cast(dtypes.float32) - b.cast(dtypes.float32)).cast(x.dtype)

def _lower_f16_mul(a: UOp, b: UOp, x: UOp) -> UOp:
  return (a.cast(dtypes.float32) * b.cast(dtypes.float32)).cast(x.dtype)

def _lower_f16_max(a: UOp, b: UOp, x: UOp) -> UOp:
  return a.cast(dtypes.float32).alu(Ops.MAX, b.cast(dtypes.float32)).cast(x.dtype)

def _lower_f16_reciprocal(a: UOp, x: UOp) -> UOp:
  return UOp(Ops.RECIPROCAL, dtypes.float32, (a.cast(dtypes.float32),)).cast(x.dtype)

def _lower_f16_sqrt(a: UOp, x: UOp) -> UOp:
  return UOp(Ops.SQRT, dtypes.float32, (a.cast(dtypes.float32),)).cast(x.dtype)

def _lower_f16_exp2(a: UOp, x: UOp) -> UOp:
  return UOp(Ops.EXP2, dtypes.float32, (a.cast(dtypes.float32),)).cast(x.dtype)

def _lower_f16_log2(a: UOp, x: UOp) -> UOp:
  return UOp(Ops.LOG2, dtypes.float32, (a.cast(dtypes.float32),)).cast(x.dtype)

def _lower_f16_trunc(a: UOp, x: UOp) -> UOp:
  return UOp(Ops.TRUNC, dtypes.float32, (a.cast(dtypes.float32),)).cast(x.dtype)

def _lower_f16_sin(a: UOp, x: UOp) -> UOp:
  return UOp(Ops.SIN, dtypes.float32, (a.cast(dtypes.float32),)).cast(x.dtype)

def _lower_f16_neg(a: UOp, x: UOp) -> UOp:
  return UOp(Ops.NEG, dtypes.float32, (a.cast(dtypes.float32),)).cast(x.dtype)

def _lower_f16_where(cond: UOp, a: UOp, b: UOp, x: UOp) -> UOp:
  return UOp(Ops.WHERE, dtypes.float32, (cond, a.cast(dtypes.float32), b.cast(dtypes.float32))).cast(x.dtype)

def _lower_f16_cmplt(a: UOp, b: UOp) -> UOp:
  return UOp(Ops.CMPLT, dtypes.bool, (a.cast(dtypes.float32), b.cast(dtypes.float32)))

def _lower_f16_cmpeq(a: UOp, b: UOp) -> UOp:
  return UOp(Ops.CMPEQ, dtypes.bool, (a.cast(dtypes.float32), b.cast(dtypes.float32)))

def _lower_f16_cmpne(a: UOp, b: UOp) -> UOp:
  return UOp(Ops.CMPNE, dtypes.bool, (a.cast(dtypes.float32), b.cast(dtypes.float32)))

def _lower_same_size_int_cast(x: UOp) -> UOp | None:
  """Convert same-size signed/unsigned integer casts to bitcasts (they're just bit reinterpretations)."""
  src, dst = x.src[0].dtype, x.dtype
  # Only match integer-to-integer casts of same size (e.g., int32 <-> uint32, int16 <-> uint16)
  # NOT int <-> float (those need actual conversion instructions)
  if src.itemsize == dst.itemsize and src != dst and dtypes.is_int(src) and dtypes.is_int(dst):
    return x.src[0].bitcast(dst)
  return None

def _lower_f16_to_bf16(x: UOp) -> UOp:
  """float16 -> bfloat16: go through float32."""
  return x.src[0].cast(dtypes.float32).cast(dtypes.bfloat16)

def _lower_bf16_to_f16(x: UOp) -> UOp:
  """bfloat16 -> float16: go through float32."""
  return x.src[0].cast(dtypes.float32).cast(dtypes.float16)

# *** Cast lowerings: multi-step casts go via intermediate types ***
_small_ints = (dtypes.int8, dtypes.int16, dtypes.uint8, dtypes.uint16)

# Pattern matcher for RDNA3-specific rewrites
# NOTE: By the time rdna_matcher runs, gated loads have already been created by devectorize
# (WHERE+LOAD -> LOAD(INDEX(buf, idx, gate), alt)). We don't need to do that transformation here.
rdna_matcher = PatternMatcher([
  # cast void does nothing
  (UPat(Ops.CAST, name="x"), lambda x: x.src[0] if isinstance(x.dtype, PtrDType) or x.src[0].dtype == dtypes.void else None),
  # same-size integer casts (signed <-> unsigned) are just bitcasts
  (UPat(Ops.CAST, name="x"), _lower_same_size_int_cast),
  # float16 <-> bfloat16 via float32
  (UPat(Ops.CAST, dtype=dtypes.bfloat16, src=(UPat(dtype=dtypes.float16),), name="x"), _lower_f16_to_bf16),
  (UPat(Ops.CAST, dtype=dtypes.bfloat16, src=(UPat(dtype=dtypes.half),), name="x"), _lower_f16_to_bf16),
  (UPat(Ops.CAST, dtype=dtypes.float16, src=(UPat(dtype=dtypes.bfloat16),), name="x"), _lower_bf16_to_f16),
  (UPat(Ops.CAST, dtype=dtypes.half, src=(UPat(dtype=dtypes.bfloat16),), name="x"), _lower_bf16_to_f16),
  # small ints <-> float16/bfloat16 via float32
  (UPat(Ops.CAST, dtype=_small_floats, src=(UPat(dtype=_small_ints),), name="x"), lambda x: x.src[0].cast(dtypes.float32).cast(x.dtype)),
  (UPat(Ops.CAST, dtype=_small_ints, src=(UPat(dtype=_small_floats),), name="x"), lambda x: x.src[0].cast(dtypes.float32).cast(x.dtype)),
  # int32/uint32 <-> float16/bfloat16 via float32
  (UPat(Ops.CAST, dtype=_small_floats, src=(UPat(dtype=(dtypes.int32, dtypes.uint32)),), name="x"),
   lambda x: x.src[0].cast(dtypes.float32).cast(x.dtype)),
  (UPat(Ops.CAST, dtype=(dtypes.int32, dtypes.uint32), src=(UPat(dtype=_small_floats),), name="x"),
   lambda x: x.src[0].cast(dtypes.float32).cast(x.dtype)),
  # int64/uint64 <-> float32 via float64
  (UPat(Ops.CAST, dtype=dtypes.float32, src=(UPat(dtype=(dtypes.int64, dtypes.uint64)),), name="x"),
   lambda x: x.src[0].cast(dtypes.float64).cast(dtypes.float32)),
  (UPat(Ops.CAST, dtype=(dtypes.int64, dtypes.uint64), src=(UPat(dtype=dtypes.float32),), name="x"),
   lambda x: x.src[0].cast(dtypes.float64).cast(x.dtype)),
  # int64/uint64 <-> float16/bfloat16 via float64 -> float32
  (UPat(Ops.CAST, dtype=_small_floats, src=(UPat(dtype=(dtypes.int64, dtypes.uint64)),), name="x"),
   lambda x: x.src[0].cast(dtypes.float64).cast(dtypes.float32).cast(x.dtype)),
  (UPat(Ops.CAST, dtype=(dtypes.int64, dtypes.uint64), src=(UPat(dtype=_small_floats),), name="x"),
   lambda x: x.src[0].cast(dtypes.float32).cast(dtypes.float64).cast(x.dtype)),
  # small ints <-> float64 via float32
  (UPat(Ops.CAST, dtype=dtypes.float64, src=(UPat(dtype=_small_ints),), name="x"), lambda x: x.src[0].cast(dtypes.float32).cast(dtypes.float64)),
  (UPat(Ops.CAST, dtype=_small_ints, src=(UPat(dtype=dtypes.float64),), name="x"), lambda x: x.src[0].cast(dtypes.float32).cast(x.dtype)),
  # float16/bfloat16 <-> float64 via float32
  (UPat(Ops.CAST, dtype=dtypes.float64, src=(UPat(dtype=_small_floats),), name="x"), lambda x: x.src[0].cast(dtypes.float32).cast(dtypes.float64)),
  (UPat(Ops.CAST, dtype=_small_floats, src=(UPat(dtype=dtypes.float64),), name="x"), lambda x: x.src[0].cast(dtypes.float32).cast(x.dtype)),
  # bool <-> float16/bfloat16 via float32
  (UPat(Ops.CAST, dtype=_small_floats, src=(UPat(dtype=dtypes.bool),), name="x"), lambda x: x.src[0].cast(dtypes.float32).cast(x.dtype)),
  (UPat(Ops.CAST, dtype=dtypes.bool, src=(UPat(dtype=_small_floats),), name="x"), lambda x: x.src[0].cast(dtypes.float32).cast(dtypes.bool)),
  # bool <-> int64/uint64 (need to handle 64-bit extension)
  (UPat(Ops.CAST, dtype=(dtypes.int64, dtypes.uint64), src=(UPat(dtype=dtypes.bool),), name="x"),
   lambda x: x.src[0].cast(dtypes.int32).cast(x.dtype)),
  (UPat(Ops.CAST, dtype=dtypes.bool, src=(UPat(dtype=(dtypes.int64, dtypes.uint64)),), name="x"),
   lambda x: x.src[0].cast(dtypes.int32).cast(dtypes.bool)),
  # float64 comparisons: lower to float32 for now (VOP3 CMP needs special handling)
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.float64), UPat.var("b")), name="x"),
   lambda x, a, b: UOp(Ops.CMPLT, dtypes.bool, (a.cast(dtypes.float32), b.cast(dtypes.float32)))),
  (UPat(Ops.CMPEQ, src=(UPat.var("a", dtypes.float64), UPat.var("b")), name="x"),
   lambda x, a, b: UOp(Ops.CMPEQ, dtypes.bool, (a.cast(dtypes.float32), b.cast(dtypes.float32)))),
  (UPat(Ops.CMPNE, src=(UPat.var("a", dtypes.float64), UPat.var("b")), name="x"),
   lambda x, a, b: UOp(Ops.CMPNE, dtypes.bool, (a.cast(dtypes.float32), b.cast(dtypes.float32)))),
  # devectorize ALU operations - RDNA doesn't have vector float ALU
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST), name="alu"), no_vectorized_alu),
  # SIN: normalize input by 1/(2π) for v_sin_f32 (expects [0,1) -> [0,2π))
  (UPat(Ops.SIN, dtype=dtypes.float32, src=(UPat.var("x"),), name="u"),
   lambda u, x: None if u.tag == "normalized" else  # skip already normalized
   UOp(Ops.SIN, dtypes.float32, (x * UOp.const(dtypes.float32, 0.15915494309189535),)).rtag("normalized")),
  # Fix fast_idiv output when shift >= 32 (needs 64-bit multiply)
  # Pattern: (x * const) >> shift for unsigned
  (UPat(Ops.SHR, src=(UPat(Ops.MUL, src=(UPat.var("x"), UPat.cvar("c"))), UPat.cvar("shift"))), _fix_fast_idiv_unsigned),
  (UPat(Ops.SHR, src=(UPat(Ops.MUL, src=(UPat.cvar("c"), UPat.var("x"))), UPat.cvar("shift"))),
   lambda x, c, shift: _fix_fast_idiv_unsigned(x, c, shift)),
  # Pattern: ((x * const) >> shift) + correction for signed (fast_idiv adds correction for negative x)
  (UPat(Ops.ADD, src=(UPat(Ops.SHR, src=(UPat(Ops.MUL, src=(UPat.var("x"), UPat.cvar("c"))), UPat.cvar("shift"))), UPat.var("add"))),
   _fix_fast_idiv_signed),
  (UPat(Ops.ADD, src=(UPat(Ops.SHR, src=(UPat(Ops.MUL, src=(UPat.cvar("c"), UPat.var("x"))), UPat.cvar("shift"))), UPat.var("add"))),
   lambda x, c, shift, add: _fix_fast_idiv_signed(x, c, shift, add)),
  # Lower integer division/modulo to float approximation (RDNA3 lacks hardware div)
  (UPat(Ops.IDIV, dtype=dtypes.uint32, src=(UPat.var("a"), UPat.var("b"))), lower_udiv),
  (UPat(Ops.IDIV, dtype=dtypes.int32, src=(UPat.var("a"), UPat.var("b"))), lower_idiv),
  (UPat(Ops.MOD, dtype=dtypes.uint32, src=(UPat.var("a"), UPat.var("b"))), lower_umod),
  (UPat(Ops.MOD, dtype=dtypes.int32, src=(UPat.var("a"), UPat.var("b"))), lower_imod),
  # Small int div/mod: sign-extend to 32-bit, divide, truncate back
  (UPat(Ops.IDIV, dtype=(dtypes.int8, dtypes.int16), src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda a, b, x: lower_idiv(a.cast(dtypes.int32), b.cast(dtypes.int32)).cast(x.dtype)),
  (UPat(Ops.IDIV, dtype=(dtypes.uint8, dtypes.uint16), src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda a, b, x: lower_udiv(a.cast(dtypes.uint32), b.cast(dtypes.uint32)).cast(x.dtype)),
  (UPat(Ops.MOD, dtype=(dtypes.int8, dtypes.int16), src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda a, b, x: lower_imod(a.cast(dtypes.int32), b.cast(dtypes.int32)).cast(x.dtype)),
  (UPat(Ops.MOD, dtype=(dtypes.uint8, dtypes.uint16), src=(UPat.var("a"), UPat.var("b")), name="x"),
   lambda a, b, x: lower_umod(a.cast(dtypes.uint32), b.cast(dtypes.uint32)).cast(x.dtype)),
  # 64-bit division/modulo using f64 approximation
  (UPat(Ops.IDIV, dtype=dtypes.uint64, src=(UPat.var("a"), UPat.var("b"))), lower_udiv64),
  (UPat(Ops.IDIV, dtype=dtypes.int64, src=(UPat.var("a"), UPat.var("b"))), lower_idiv64),
  (UPat(Ops.MOD, dtype=dtypes.uint64, src=(UPat.var("a"), UPat.var("b"))), lower_umod64),
  # compute byte offset for INDEX operations at UOp level (like PTX)
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx")), name="op", allow_any_len=True), lambda buf, idx, op:
    UOp(Ops.INDEX, dtype=dtypes.int32, src=(buf, idx.cast(dtypes.int32)*buf.dtype.itemsize)+op.src[2:])
      if op.dtype != dtypes.int32 and isinstance(buf.dtype, PtrDType) and buf.dtype.addrspace != AddrSpace.REG else None),
  # f64 ADD/SUB/MUL -> MULACC (RDNA3 lacks native v_add_f64/v_sub_f64/v_mul_f64, use v_fma_f64)
  # ADD: a + b = FMA(1.0, a, b) = 1.0 * a + b
  (UPat(Ops.ADD, dtype=dtypes.float64, src=(UPat.var("a"), UPat.var("b"))),
   lambda a, b: UOp(Ops.MULACC, dtypes.float64, (UOp.const(dtypes.float64, 1.0), a, b))),
  # SUB: a - b = FMA(-1.0, b, a) = -1.0 * b + a
  (UPat(Ops.SUB, dtype=dtypes.float64, src=(UPat.var("a"), UPat.var("b"))),
   lambda a, b: UOp(Ops.MULACC, dtypes.float64, (UOp.const(dtypes.float64, -1.0), b, a))),
  # MUL: a * b = FMA(a, b, 0.0) = a * b + 0.0
  (UPat(Ops.MUL, dtype=dtypes.float64, src=(UPat.var("a"), UPat.var("b"))),
   lambda a, b: UOp(Ops.MULACC, dtypes.float64, (a, b, UOp.const(dtypes.float64, 0.0)))),
  # float16/bfloat16 ALU lowering - convert to f32, operate, convert back
  # Binary ops: ADD, SUB, MUL, MAX
  (UPat(Ops.ADD, dtype=_small_floats, src=(UPat.var("a"), UPat.var("b")), name="x"), _lower_f16_add),
  (UPat(Ops.SUB, dtype=_small_floats, src=(UPat.var("a"), UPat.var("b")), name="x"), _lower_f16_sub),
  (UPat(Ops.MUL, dtype=_small_floats, src=(UPat.var("a"), UPat.var("b")), name="x"), _lower_f16_mul),
  (UPat(Ops.MAX, dtype=_small_floats, src=(UPat.var("a"), UPat.var("b")), name="x"), _lower_f16_max),
  # Unary ops: RECIPROCAL, SQRT, EXP2, LOG2, TRUNC, SIN, NEG
  (UPat(Ops.RECIPROCAL, dtype=_small_floats, src=(UPat.var("a"),), name="x"), _lower_f16_reciprocal),
  (UPat(Ops.SQRT, dtype=_small_floats, src=(UPat.var("a"),), name="x"), _lower_f16_sqrt),
  (UPat(Ops.EXP2, dtype=_small_floats, src=(UPat.var("a"),), name="x"), _lower_f16_exp2),
  (UPat(Ops.LOG2, dtype=_small_floats, src=(UPat.var("a"),), name="x"), _lower_f16_log2),
  (UPat(Ops.TRUNC, dtype=_small_floats, src=(UPat.var("a"),), name="x"), _lower_f16_trunc),
  (UPat(Ops.SIN, dtype=_small_floats, src=(UPat.var("a"),), name="x"), _lower_f16_sin),
  (UPat(Ops.NEG, dtype=_small_floats, src=(UPat.var("a"),), name="x"), _lower_f16_neg),
  # WHERE for float16
  (UPat(Ops.WHERE, dtype=_small_floats, src=(UPat.var("cond"), UPat.var("a"), UPat.var("b")), name="x"), _lower_f16_where),
  # Comparisons on float16 inputs - note: result dtype is bool, but inputs are float16
  (UPat(Ops.CMPLT, src=(UPat(dtype=_small_floats, name="a"), UPat(dtype=_small_floats, name="b"))), _lower_f16_cmplt),
  (UPat(Ops.CMPEQ, src=(UPat(dtype=_small_floats, name="a"), UPat(dtype=_small_floats, name="b"))), _lower_f16_cmpeq),
  (UPat(Ops.CMPNE, src=(UPat(dtype=_small_floats, name="a"), UPat(dtype=_small_floats, name="b"))), _lower_f16_cmpne),
])
