# RDNA3-specific UOp-level rewrites
# These transformations run before rendering to lower operations without hardware support

from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat, GroupOp
from tinygrad.dtype import dtypes, PtrDType, AddrSpace
from tinygrad.codegen.late.devectorizer import no_vectorized_alu

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
  zero, one = UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, 1)
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

# Pattern matcher for RDNA3-specific rewrites
# NOTE: By the time rdna_matcher runs, gated loads have already been created by devectorize
# (WHERE+LOAD -> LOAD(INDEX(buf, idx, gate), alt)). We don't need to do that transformation here.
rdna_matcher = PatternMatcher([
  # cast void does nothing
  (UPat(Ops.CAST, name="x"), lambda x: x.src[0] if isinstance(x.dtype, PtrDType) or x.src[0].dtype == dtypes.void else None),
  # devectorize ALU operations - RDNA doesn't have vector float ALU
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST), name="alu"), no_vectorized_alu),
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
])
