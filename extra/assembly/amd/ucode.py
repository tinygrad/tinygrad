# UOp-based pseudocode compiler for AMD GPU instruction emulation
# Transforms pseudocode -> qcode AST -> UOps -> execution via simplify

import functools, struct, math
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes, DType
from extra.assembly.amd.qcode import parse, Const, Var, Typed, Slice, Index, Cast, Unary, Binary, Ternary, Call, Pack, Assign, Declare, If, For
from extra.assembly.amd.qcode import DType as QDType

# ═══════════════════════════════════════════════════════════════════════════════
# TYPE MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

QDTYPE_MAP = {
  QDType.F64: dtypes.float64, QDType.F32: dtypes.float32, QDType.F16: dtypes.float16,
  QDType.U64: dtypes.uint64, QDType.U32: dtypes.uint32, QDType.U24: dtypes.uint24, QDType.U16: dtypes.uint16, QDType.U8: dtypes.uint8,
  QDType.I64: dtypes.int64, QDType.I32: dtypes.int32, QDType.I24: dtypes.int24, QDType.I16: dtypes.int16, QDType.I8: dtypes.int8,
  QDType.B128: dtypes.uint64, QDType.B64: dtypes.uint64, QDType.B32: dtypes.uint32, QDType.B16: dtypes.uint16, QDType.B8: dtypes.uint8,
  QDType.U1: dtypes.uint32, QDType.I1: dtypes.int32, QDType.U3: dtypes.uint32, QDType.U4: dtypes.uint32, QDType.I4: dtypes.int32,
}

def _is_float(dtype: DType) -> bool: return dtype in (dtypes.float16, dtypes.float32, dtypes.float64)
def _qdt(qd: QDType) -> DType: return QDTYPE_MAP.get(qd, dtypes.uint32)

# ═══════════════════════════════════════════════════════════════════════════════
# UOP GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

class UOpBuilder:
  """Builds a UOp graph from qcode AST at compile time."""

  def __init__(self):
    self.input_vars = {
      'S0': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('S0', 0, 0xffffffff)),
      'S1': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('S1', 0, 0xffffffff)),
      'S2': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('S2', 0, 0xffffffff)),
      'D0': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('D0', 0, 0xffffffff)),
      'S0_64': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('S0_64', 0, 0xffffffffffffffff)),
      'S1_64': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('S1_64', 0, 0xffffffffffffffff)),
      'S2_64': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('S2_64', 0, 0xffffffffffffffff)),
      'D0_64': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('D0_64', 0, 0xffffffffffffffff)),
      'SCC': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('SCC', 0, 1)),
      'VCC': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('VCC', 0, 0xffffffffffffffff)),
      'EXEC': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('EXEC', 0, 0xffffffffffffffff)),
      'laneId': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('laneId', 0, 31)),
      'SIMM16': UOp(Ops.DEFINE_VAR, dtypes.int32, (), ('SIMM16', -32768, 32767)),
      'SIMM32': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('SIMM32', 0, 0xffffffff)),
      'PC': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('PC', 0, 0xffffffffffffffff)),
    }
    self.vars: dict[str, UOp] = dict(self.input_vars)
    self.outputs: list[tuple[str, UOp, DType]] = []

  def const(self, val, dtype: DType) -> UOp: return UOp(Ops.CONST, dtype, (), val)

  def cast(self, x: UOp, dtype: DType) -> UOp:
    if x.dtype == dtype: return x
    return UOp(Ops.BITCAST if dtype.itemsize == x.dtype.itemsize else Ops.CAST, dtype, (x,))

  def build_sink(self) -> UOp:
    if not self.outputs: return UOp(Ops.SINK, dtypes.void, ())
    return UOp(Ops.SINK, dtypes.void, tuple(uop for _, uop, _ in self.outputs))

# ═══════════════════════════════════════════════════════════════════════════════
# AST -> UOP TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════════════

def _get_var_dtype(name: str, qdt: QDType|None = None) -> tuple[str, DType]:
  """Get variable name and dtype, handling 64-bit variants."""
  dt = _qdt(qdt) if qdt else dtypes.uint32
  if qdt in (QDType.F64, QDType.U64, QDType.I64, QDType.B64) and name.isupper():
    return name + '_64', dt
  return name, dt

def transform_expr(node, b: UOpBuilder, hint: DType = None) -> tuple[UOp, DType]:
  """Transform qcode AST expression to UOp."""
  match node:
    case Const(val, qdt):
      dt = _qdt(qdt) if qdt != QDType.I32 or hint is None else hint
      if isinstance(val, float) and not _is_float(dt): dt = dtypes.float32
      return b.const(val, dt), dt

    case Var(name):
      # Special constants
      if name == 'PI': return b.const(math.pi, hint or dtypes.float64), hint or dtypes.float64
      if name in ('INF', '+INF'): return b.const(float('inf'), hint or dtypes.float64), hint or dtypes.float64
      if name == '-INF': return b.const(float('-inf'), hint or dtypes.float64), hint or dtypes.float64
      if name == 'WAVE_MODE.IEEE': return b.const(1, dtypes.uint32), dtypes.uint32
      if name == 'WAVE32': return b.const(1, dtypes.uint32), dtypes.uint32
      if name == 'WAVE64': return b.const(0, dtypes.uint32), dtypes.uint32
      if name == 'ROUND_MODE': return b.const(0, dtypes.uint32), dtypes.uint32
      if name == 'VCCZ':
        vcc = b.vars.get('VCC')
        cmp = UOp(Ops.CMPEQ, dtypes.bool, (vcc, b.const(0, dtypes.uint64)))
        return b.cast(cmp, dtypes.uint32), dtypes.uint32
      if name == 'EXECZ':
        ex = b.vars.get('EXEC')
        cmp = UOp(Ops.CMPEQ, dtypes.bool, (ex, b.const(0, dtypes.uint64)))
        return b.cast(cmp, dtypes.uint32), dtypes.uint32
      if name.startswith('eval '): return b.vars.get('_eval', b.const(0, dtypes.uint32)), dtypes.uint32
      # Regular variable
      if name not in b.vars: raise ValueError(f"Unknown variable: {name}")
      uop = b.vars[name]
      dt = hint or uop.dtype
      return b.cast(uop, dt), dt

    case Typed(expr, qdt):
      dt = _qdt(qdt)
      var_name = expr.name if isinstance(expr, Var) else None
      # Handle typed variable access
      if var_name:
        if var_name == 'VCCZ':
          vcc = b.vars.get('VCC')
          cmp = UOp(Ops.CMPEQ, dtypes.bool, (vcc, b.const(0, dtypes.uint64)))
          # Cast to uint32 for integer comparisons
          return b.cast(cmp, dtypes.uint32), dt
        if var_name == 'EXECZ':
          ex = b.vars.get('EXEC')
          cmp = UOp(Ops.CMPEQ, dtypes.bool, (ex, b.const(0, dtypes.uint64)))
          return b.cast(cmp, dtypes.uint32), dt
        # For 64-bit types, use _64 variant
        vn, vdt = _get_var_dtype(var_name, qdt)
        base = b.vars.get(vn) if vn in b.vars else b.vars.get(var_name)
        if base is None: raise ValueError(f"Unknown variable: {var_name}")
        # Handle 24-bit types
        if qdt == QDType.U24:
          masked = UOp(Ops.AND, dtypes.uint32, (base, b.const(0xffffff, dtypes.uint32)))
          return masked, dtypes.uint32
        if qdt == QDType.I24:
          masked = UOp(Ops.AND, dtypes.uint32, (base, b.const(0xffffff, dtypes.uint32)))
          xored = UOp(Ops.XOR, dtypes.int32, (masked, b.const(0x800000, dtypes.int32)))
          return UOp(Ops.SUB, dtypes.int32, (xored, b.const(0x800000, dtypes.int32))), dtypes.int32
        # Float types need bitcast
        if _is_float(dt):
          if dt == dtypes.float16:
            # Mask to 16 bits and bitcast to f16
            masked = UOp(Ops.AND, dtypes.uint16, (b.cast(base, dtypes.uint16), b.const(0xffff, dtypes.uint16)))
            return UOp(Ops.BITCAST, dtypes.float16, (masked,)), dtypes.float16
          return UOp(Ops.BITCAST, dt, (base,)), dt
        # For signed integer types, keep as unsigned to avoid overflow issues during simplify
        # Return the unsigned base but report the signed dtype for semantic purposes
        if dt == dtypes.int32: return base, dtypes.int32
        if dt == dtypes.int64:
          base64 = b.vars.get(var_name + '_64') if (var_name + '_64') in b.vars else base
          return base64, dtypes.int64
        if dt == dtypes.int16: return base, dtypes.int16
        if dt == dtypes.int8: return base, dtypes.int8
        return b.cast(base, dt), dt
      # Non-variable typed expression
      inner, _ = transform_expr(expr, b, dt)
      if _is_float(dt):
        if dt == dtypes.float16:
          # For f16, need to cast to u16 first then bitcast
          inner_u16 = b.cast(inner, dtypes.uint16)
          return UOp(Ops.BITCAST, dt, (inner_u16,)), dt
        return UOp(Ops.BITCAST, dt, (inner,)), dt
      return b.cast(inner, dt), dt

    case Slice(expr, hi, lo):
      base, base_dt = transform_expr(expr, b)
      hi_uop, _ = transform_expr(hi, b)
      lo_uop, _ = transform_expr(lo, b)
      # For constant hi/lo, compute mask directly
      if hi_uop.op == Ops.CONST and lo_uop.op == Ops.CONST:
        hi_val, lo_val = int(hi_uop.arg), int(lo_uop.arg)
        if hi_val < lo_val: hi_val, lo_val = lo_val, hi_val
        mask = (1 << (hi_val - lo_val + 1)) - 1
        shifted = UOp(Ops.SHR, base_dt, (base, b.const(lo_val, base_dt))) if lo_val > 0 else base
        masked = UOp(Ops.AND, dtypes.uint32, (b.cast(shifted, dtypes.uint32), b.const(mask, dtypes.uint32)))
        return masked, hint or dtypes.uint32
      raise ValueError(f"Non-constant slice bounds not supported: {node}")

    case Index(expr, idx):
      base, base_dt = transform_expr(expr, b)
      idx_uop, _ = transform_expr(idx, b)
      # Single bit extraction: (base >> idx) & 1
      shifted = UOp(Ops.SHR, base_dt, (base, b.cast(idx_uop, base_dt)))
      masked = UOp(Ops.AND, dtypes.uint32, (b.cast(shifted, dtypes.uint32), b.const(1, dtypes.uint32)))
      return masked, dtypes.uint32

    case Cast(bits, typ, expr):
      dtype_map = {
        (16, 'I'): dtypes.int16, (16, 'U'): dtypes.uint16, (16, 'F'): dtypes.float16,
        (32, 'I'): dtypes.int32, (32, 'U'): dtypes.uint32, (32, 'F'): dtypes.float32, (32, 'B'): dtypes.uint32,
        (64, 'I'): dtypes.int64, (64, 'U'): dtypes.uint64, (64, 'F'): dtypes.float64, (64, 'B'): dtypes.uint64,
      }
      dt = dtype_map.get((bits, typ), dtypes.uint32)
      inner, inner_dt = transform_expr(expr, b, dt)
      if typ == 'F': return UOp(Ops.CAST, dt, (inner,)), dt
      if inner_dt in (dtypes.uint32, dtypes.int32) and bits == 32: return inner, dt
      if inner_dt in (dtypes.uint64, dtypes.int64) and bits == 64: return inner, dt
      # For signed widening cast, first cast to signed type to get sign extension
      if typ == 'I' and inner_dt in (dtypes.int32, dtypes.int16, dtypes.int8):
        signed_inner = b.cast(inner, inner_dt)  # BITCAST to signed
        return UOp(Ops.CAST, dt, (signed_inner,)), dt
      return b.cast(inner, dt), dt

    case Unary(op, expr):
      val, dt = transform_expr(expr, b, hint)
      if op == '-': return UOp(Ops.NEG, dt, (val,)), dt
      if op == '~': return UOp(Ops.XOR, dt, (val, b.const(-1, dt))), dt
      if op == '!': return UOp(Ops.CMPEQ, dtypes.bool, (val, b.const(0, dt))), dtypes.bool
      raise ValueError(f"Unknown unary op: {op}")

    case Binary(op, left, right):
      l, l_dt = transform_expr(left, b, hint)
      r, r_dt = transform_expr(right, b, l_dt if _is_float(l_dt) else hint)
      # Use actual UOp dtype for arithmetic to avoid type mismatches
      # The semantic dtype (l_dt/r_dt) may be signed but UOp is unsigned
      result_dt = l.dtype if _is_float(l.dtype) else r.dtype if _is_float(r.dtype) else l.dtype

      binop_map = {'+': Ops.ADD, '-': Ops.SUB, '*': Ops.MUL, '/': Ops.FDIV, '&': Ops.AND, '|': Ops.OR, '^': Ops.XOR,
                   '<<': Ops.SHL, '==': Ops.CMPEQ, '!=': Ops.CMPNE, '<>': Ops.CMPNE, '<': Ops.CMPLT}
      # >> is logical shift for unsigned, arithmetic shift for signed
      if op == '>>':
        if l_dt in (dtypes.int32, dtypes.int64, dtypes.int16, dtypes.int8):
          signed_l = b.cast(l, l_dt)
          shifted = UOp(Ops.SHR, l_dt, (signed_l, r))
          return b.cast(shifted, l.dtype), l_dt
        return UOp(Ops.SHR, result_dt, (l, r)), result_dt
      if op == '||':
        one, zero = b.const(1, dtypes.uint32), b.const(0, dtypes.uint32)
        inner = UOp(Ops.WHERE, dtypes.uint32, (r, one, zero))
        return UOp(Ops.WHERE, dtypes.uint32, (l, one, inner)), dtypes.uint32
      if op == '&&':
        one, zero = b.const(1, dtypes.uint32), b.const(0, dtypes.uint32)
        inner = UOp(Ops.WHERE, dtypes.uint32, (r, one, zero))
        return UOp(Ops.WHERE, dtypes.uint32, (l, inner, zero)), dtypes.uint32
      # For signed comparisons, use the semantic dtype (l_dt) for comparison
      def _cmp_operands():
        if l_dt in (dtypes.int32, dtypes.int64, dtypes.int16, dtypes.int8):
          return b.cast(l, l_dt), b.cast(r, l_dt)
        return l, r
      if op == '>':
        cmp_l, cmp_r = _cmp_operands()
        return UOp(Ops.CMPLT, dtypes.bool, (cmp_r, cmp_l)), dtypes.bool
      if op == '>=':
        cmp_l, cmp_r = _cmp_operands()
        lt = UOp(Ops.CMPLT, dtypes.bool, (cmp_l, cmp_r))
        return UOp(Ops.XOR, dtypes.bool, (lt, b.const(True, dtypes.bool))), dtypes.bool
      if op == '<=':
        cmp_l, cmp_r = _cmp_operands()
        lt = UOp(Ops.CMPLT, dtypes.bool, (cmp_r, cmp_l))
        return UOp(Ops.XOR, dtypes.bool, (lt, b.const(True, dtypes.bool))), dtypes.bool
      if op == '**':
        if l.op == Ops.CONST and l.arg == 2.0:
          # For signed exponents, cast to signed first to get correct sign extension
          if r_dt in (dtypes.int32, dtypes.int64, dtypes.int16, dtypes.int8):
            r_signed = b.cast(r, r_dt)
            exp = UOp(Ops.CAST, result_dt, (r_signed,))
          else:
            exp = UOp(Ops.CAST, result_dt, (r,)) if r.dtype != result_dt else r
          return UOp(Ops.EXP2, result_dt, (exp,)), result_dt
        log_a = UOp(Ops.LOG2, result_dt, (l,))
        exp = UOp(Ops.CAST, result_dt, (r,)) if r.dtype != result_dt else r
        return UOp(Ops.EXP2, result_dt, (UOp(Ops.MUL, result_dt, (exp, log_a)),)), result_dt
      if op == '%':
        # a % b = a - (a / b) * b (integer modulo)
        div = UOp(Ops.IDIV, result_dt, (l, r))
        return UOp(Ops.SUB, result_dt, (l, UOp(Ops.MUL, result_dt, (div, r)))), result_dt
      uop_op = binop_map.get(op)
      if uop_op is None: raise ValueError(f"Unknown binary op: {op}")
      out_dt = dtypes.bool if uop_op in (Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE) else result_dt
      # For signed < comparison, cast operands to signed type
      if op == '<' and l_dt in (dtypes.int32, dtypes.int64, dtypes.int16, dtypes.int8):
        cmp_l, cmp_r = b.cast(l, l_dt), b.cast(r, l_dt)
        return UOp(Ops.CMPLT, dtypes.bool, (cmp_l, cmp_r)), dtypes.bool
      return UOp(uop_op, out_dt, (l, r)), out_dt

    case Ternary(cond, t, f):
      c, _ = transform_expr(cond, b)
      tv, t_dt = transform_expr(t, b, hint)
      fv, f_dt = transform_expr(f, b, t_dt)
      return UOp(Ops.WHERE, t_dt, (c, tv, fv)), t_dt

    case Call(name, args):
      return _transform_call(name, args, b, hint)

    case Pack(exprs):
      if len(exprs) == 2:
        hi, hi_dt = transform_expr(exprs[0], b)
        lo, lo_dt = transform_expr(exprs[1], b)
        if lo_dt.itemsize >= 4:
          hi_ext = b.cast(hi, dtypes.uint64)
          lo_ext = b.cast(lo, dtypes.uint64)
          hi_shifted = UOp(Ops.SHL, dtypes.uint64, (hi_ext, b.const(32, dtypes.uint64)))
          return UOp(Ops.OR, dtypes.uint64, (hi_shifted, lo_ext)), dtypes.uint64
        else:
          hi_shifted = UOp(Ops.SHL, dtypes.uint32, (b.cast(hi, dtypes.uint32), b.const(16, dtypes.uint32)))
          lo_masked = UOp(Ops.AND, dtypes.uint32, (b.cast(lo, dtypes.uint32), b.const(0xffff, dtypes.uint32)))
          return UOp(Ops.OR, dtypes.uint32, (hi_shifted, lo_masked)), dtypes.uint32
      raise ValueError(f"Pack with {len(exprs)} elements not supported")

  raise ValueError(f"Cannot transform expression: {node}")

def _transform_call(name: str, args: tuple, b: UOpBuilder, hint: DType) -> tuple[UOp, DType]:
  """Transform function call to UOp."""
  def arg(i, h=None): return transform_expr(args[i], b, h)

  # Memory access
  if name == 'MEM':
    addr, _ = arg(0)
    return addr, hint or dtypes.uint32

  # Math functions
  if name == 'fma' and len(args) == 3:
    a, _ = arg(0, hint); bv, _ = arg(1, hint); c, dt = arg(2, hint)
    return UOp(Ops.MULACC, dt, (a, bv, c)), dt
  if name == 'trunc' and len(args) == 1:
    v, dt = arg(0, hint); return UOp(Ops.TRUNC, dt, (v,)), dt
  if name == 'floor' and len(args) == 1:
    v, dt = arg(0, hint); return UOp(Ops.TRUNC, dt, (v,)), dt  # TODO: proper floor
  if name == 'sqrt' and len(args) == 1:
    v, dt = arg(0, hint); return UOp(Ops.SQRT, dt, (v,)), dt
  if name == 'abs' and len(args) == 1:
    v, dt = arg(0, hint)
    neg = UOp(Ops.NEG, dt, (v,))
    cond = UOp(Ops.CMPLT, dtypes.bool, (v, b.const(0, dt)))
    return UOp(Ops.WHERE, dt, (cond, neg, v)), dt
  if name == 'exp2' and len(args) == 1:
    v, dt = arg(0, hint); return UOp(Ops.EXP2, dt, (v,)), dt
  if name == 'log2' and len(args) == 1:
    v, dt = arg(0, hint); return UOp(Ops.LOG2, dt, (v,)), dt
  if name == 'sin' and len(args) == 1:
    v, dt = arg(0, hint); return UOp(Ops.SIN, dt, (v,)), dt
  if name == 'cos' and len(args) == 1:
    v, dt = arg(0, hint)
    pi_2 = b.const(1.5707963267948966, dt)
    return UOp(Ops.SIN, dt, (UOp(Ops.ADD, dt, (v, pi_2)),)), dt
  if name == 'rcp' and len(args) == 1:
    v, dt = arg(0, hint); return UOp(Ops.RECIPROCAL, dt, (v,)), dt
  if name == 'rsqrt' and len(args) == 1:
    v, dt = arg(0, hint)
    return UOp(Ops.RECIPROCAL, dt, (UOp(Ops.SQRT, dt, (v,)),)), dt
  if name == 'min' and len(args) == 2:
    a, dt = arg(0, hint); bv, _ = arg(1, hint)
    # For signed types, compare with signed dtype
    cmp_a, cmp_b = (b.cast(a, dt), b.cast(bv, dt)) if dt in (dtypes.int32, dtypes.int64, dtypes.int16, dtypes.int8) else (a, bv)
    return UOp(Ops.WHERE, a.dtype, (UOp(Ops.CMPLT, dtypes.bool, (cmp_a, cmp_b)), a, bv)), dt
  if name == 'max' and len(args) == 2:
    a, dt = arg(0, hint); bv, _ = arg(1, hint)
    cmp_a, cmp_b = (b.cast(a, dt), b.cast(bv, dt)) if dt in (dtypes.int32, dtypes.int64, dtypes.int16, dtypes.int8) else (a, bv)
    return UOp(Ops.WHERE, a.dtype, (UOp(Ops.CMPLT, dtypes.bool, (cmp_b, cmp_a)), a, bv)), dt
  if name == 'clamp' and len(args) == 3:
    x, dt = arg(0, hint); lo, _ = arg(1, hint); hi, _ = arg(2, hint)
    cond_lo = UOp(Ops.CMPLT, dtypes.bool, (x, lo))
    max_val = UOp(Ops.WHERE, dt, (cond_lo, lo, x))
    cond_hi = UOp(Ops.CMPLT, dtypes.bool, (hi, max_val))
    return UOp(Ops.WHERE, dt, (cond_hi, hi, max_val)), dt
  if name == 'fract' and len(args) == 1:
    v, dt = arg(0, hint)
    truncated = UOp(Ops.TRUNC, dt, (v,))
    return UOp(Ops.SUB, dt, (v, truncated)), dt

  # NaN/Inf checks
  if name == 'isNAN' and len(args) == 1:
    v, dt = arg(0, hint)
    return UOp(Ops.CMPNE, dtypes.bool, (v, v)), dtypes.bool
  if name == 'isQuietNAN' and len(args) == 1:
    v, dt = arg(0, hint)
    return UOp(Ops.CMPNE, dtypes.bool, (v, v)), dtypes.bool
  if name == 'isSignalNAN' and len(args) == 1:
    return b.const(0, dtypes.bool), dtypes.bool
  if name == 'cvtToQuietNAN' and len(args) == 1:
    v, dt = arg(0, hint); return v, dt
  if name == 'isINF' and len(args) == 1:
    v, dt = arg(0, hint)
    inf = b.const(float('inf'), dt)
    neg_inf = b.const(float('-inf'), dt)
    is_pos = UOp(Ops.CMPEQ, dtypes.bool, (v, inf))
    is_neg = UOp(Ops.CMPEQ, dtypes.bool, (v, neg_inf))
    return UOp(Ops.OR, dtypes.bool, (is_pos, is_neg)), dtypes.bool

  # Type conversions
  cvt_map = {
    'u32_to_f32': (dtypes.float32, False), 'i32_to_f32': (dtypes.float32, False),
    'f32_to_u32': (dtypes.uint32, True), 'f32_to_i32': (dtypes.int32, False),
    'f16_to_f32': (dtypes.float32, False), 'f32_to_f16': (dtypes.float16, False),
    'f32_to_u8': (dtypes.uint8, False), 'f32_to_i8': (dtypes.int8, False),
    'f32_to_u16': (dtypes.uint16, False), 'f32_to_i16': (dtypes.int16, False),
    'v_cvt_u16_f32': (dtypes.uint16, False), 'v_cvt_i16_f32': (dtypes.int16, False),
    'f64_to_i32': (dtypes.int32, False), 'f64_to_u32': (dtypes.uint32, True),
    'i32_to_f64': (dtypes.float64, False), 'u32_to_f64': (dtypes.float64, False),
    'f64_to_f32': (dtypes.float32, False), 'f32_to_f64': (dtypes.float64, False),
    'u16_to_f16': (dtypes.float16, False), 'i16_to_f16': (dtypes.float16, False),
    'f16_to_u16': (dtypes.uint16, False), 'f16_to_i16': (dtypes.int16, False),
  }
  if name in cvt_map and len(args) == 1:
    v, v_dt = arg(0)
    dt, clamp_neg = cvt_map[name]
    if clamp_neg:
      zero = b.const(0.0, v_dt)
      v = UOp(Ops.WHERE, v_dt, (UOp(Ops.CMPLT, dtypes.bool, (v, zero)), zero, v))
    return UOp(Ops.CAST, dt, (v,)), dt

  if name == 'f16_to_snorm' and len(args) == 1:
    v, dt = arg(0)
    clamped = UOp(Ops.WHERE, dt, (UOp(Ops.CMPLT, dtypes.bool, (v, b.const(-1.0, dt))), b.const(-1.0, dt), v))
    clamped = UOp(Ops.WHERE, dt, (UOp(Ops.CMPLT, dtypes.bool, (b.const(1.0, dt), clamped)), b.const(1.0, dt), clamped))
    scaled = UOp(Ops.MUL, dt, (clamped, b.const(32767.0, dt)))
    return UOp(Ops.CAST, dtypes.int16, (scaled,)), dtypes.int16
  if name == 'f16_to_unorm' and len(args) == 1:
    v, dt = arg(0)
    clamped = UOp(Ops.WHERE, dt, (UOp(Ops.CMPLT, dtypes.bool, (v, b.const(0.0, dt))), b.const(0.0, dt), v))
    clamped = UOp(Ops.WHERE, dt, (UOp(Ops.CMPLT, dtypes.bool, (b.const(1.0, dt), clamped)), b.const(1.0, dt), clamped))
    scaled = UOp(Ops.MUL, dt, (clamped, b.const(65535.0, dt)))
    return UOp(Ops.CAST, dtypes.uint16, (scaled,)), dtypes.uint16

  # Sign/exponent/mantissa extraction
  if name == 'sign' and len(args) == 1:
    v, dt = arg(0, hint)
    if dt == dtypes.float64:
      bits = UOp(Ops.BITCAST, dtypes.uint64, (v,))
      sign = UOp(Ops.SHR, dtypes.uint64, (bits, b.const(63, dtypes.uint64)))
      return UOp(Ops.AND, dtypes.uint32, (b.cast(sign, dtypes.uint32), b.const(1, dtypes.uint32))), dtypes.uint32
    elif dt == dtypes.float16:
      bits = UOp(Ops.BITCAST, dtypes.uint16, (v,))
      sign = UOp(Ops.SHR, dtypes.uint16, (bits, b.const(15, dtypes.uint16)))
      return UOp(Ops.AND, dtypes.uint32, (b.cast(sign, dtypes.uint32), b.const(1, dtypes.uint32))), dtypes.uint32
    else:
      bits = UOp(Ops.BITCAST, dtypes.uint32, (v,))
      sign = UOp(Ops.SHR, dtypes.uint32, (bits, b.const(31, dtypes.uint32)))
      return UOp(Ops.AND, dtypes.uint32, (sign, b.const(1, dtypes.uint32))), dtypes.uint32

  if name == 'exponent' and len(args) == 1:
    v, dt = arg(0, hint)
    if dt == dtypes.float64:
      bits = UOp(Ops.BITCAST, dtypes.uint64, (v,))
      exp = UOp(Ops.SHR, dtypes.uint64, (bits, b.const(52, dtypes.uint64)))
      return UOp(Ops.AND, dtypes.uint32, (b.cast(exp, dtypes.uint32), b.const(0x7ff, dtypes.uint32))), dtypes.uint32
    elif dt == dtypes.float16:
      bits = UOp(Ops.BITCAST, dtypes.uint16, (v,))
      exp = UOp(Ops.SHR, dtypes.uint16, (bits, b.const(10, dtypes.uint16)))
      return UOp(Ops.AND, dtypes.uint32, (b.cast(exp, dtypes.uint32), b.const(0x1f, dtypes.uint32))), dtypes.uint32
    else:
      bits = UOp(Ops.BITCAST, dtypes.uint32, (v,))
      exp = UOp(Ops.SHR, dtypes.uint32, (bits, b.const(23, dtypes.uint32)))
      return UOp(Ops.AND, dtypes.uint32, (exp, b.const(0xff, dtypes.uint32))), dtypes.uint32

  if name == 'mantissa' and len(args) == 1:
    v, dt = arg(0, hint)
    if dt == dtypes.float64:
      bits = UOp(Ops.BITCAST, dtypes.uint64, (v,))
      return UOp(Ops.AND, dtypes.uint64, (bits, b.const(0xfffffffffffff, dtypes.uint64))), dtypes.uint64
    elif dt == dtypes.float16:
      bits = UOp(Ops.BITCAST, dtypes.uint16, (v,))
      return UOp(Ops.AND, dtypes.uint32, (b.cast(bits, dtypes.uint32), b.const(0x3ff, dtypes.uint32))), dtypes.uint32
    else:
      bits = UOp(Ops.BITCAST, dtypes.uint32, (v,))
      return UOp(Ops.AND, dtypes.uint32, (bits, b.const(0x7fffff, dtypes.uint32))), dtypes.uint32

  if name == 'isEven' and len(args) == 1:
    v, dt = arg(0, hint)
    int_val = UOp(Ops.CAST, dtypes.int64, (v,))
    bit0 = UOp(Ops.AND, dtypes.int64, (int_val, b.const(1, dtypes.int64)))
    return UOp(Ops.CMPEQ, dtypes.bool, (bit0, b.const(0, dtypes.int64))), dtypes.bool

  if name == 'signext' and len(args) == 1:
    v, dt = arg(0)
    return b.cast(v, dtypes.int64), dtypes.int64

  if name == 'signext_from_bit' and len(args) == 2:
    val, dt = arg(0, hint)
    width, _ = arg(1)
    one = b.const(1, dt)
    width_minus_1 = UOp(Ops.SUB, dt, (b.cast(width, dt), one))
    sign_bit = UOp(Ops.SHL, dt, (one, width_minus_1))
    xored = UOp(Ops.XOR, dt, (val, sign_bit))
    result = UOp(Ops.SUB, dt, (xored, sign_bit))
    width_is_zero = UOp(Ops.CMPEQ, dtypes.bool, (width, b.const(0, width.dtype)))
    return UOp(Ops.WHERE, dt, (width_is_zero, b.const(0, dt), result)), dt

  if name == 'ABSDIFF' and len(args) == 2:
    a, _ = arg(0); bv, _ = arg(1)
    a_gt_b = UOp(Ops.CMPLT, dtypes.bool, (bv, a))
    max_v = UOp(Ops.WHERE, dtypes.uint32, (a_gt_b, b.cast(a, dtypes.uint32), b.cast(bv, dtypes.uint32)))
    min_v = UOp(Ops.WHERE, dtypes.uint32, (a_gt_b, b.cast(bv, dtypes.uint32), b.cast(a, dtypes.uint32)))
    return UOp(Ops.SUB, dtypes.uint32, (max_v, min_v)), dtypes.uint32

  if name == 'pow' and len(args) == 2:
    base, base_dt = arg(0, hint); exp, _ = arg(1, hint)
    result_dt = base_dt if _is_float(base_dt) else hint or dtypes.float32
    if base.op == Ops.CONST and base.arg == 2.0:
      exp_uop = UOp(Ops.CAST, result_dt, (exp,)) if exp.dtype != result_dt else exp
      return UOp(Ops.EXP2, result_dt, (exp_uop,)), result_dt
    base_cast = UOp(Ops.CAST, result_dt, (base,)) if base.dtype != result_dt else base
    exp_cast = UOp(Ops.CAST, result_dt, (exp,)) if exp.dtype != result_dt else exp
    log_a = UOp(Ops.LOG2, result_dt, (base_cast,))
    return UOp(Ops.EXP2, result_dt, (UOp(Ops.MUL, result_dt, (exp_cast, log_a)),)), result_dt

  if name == 'LT_NEG_ZERO' and len(args) == 2:
    a, dt = arg(0, hint); bv, _ = arg(1, hint)
    if dt == dtypes.float64:
      a_bits = UOp(Ops.BITCAST, dtypes.int64, (a,))
      b_bits = UOp(Ops.BITCAST, dtypes.int64, (bv,))
    elif dt == dtypes.float16:
      a_bits = UOp(Ops.BITCAST, dtypes.int16, (a,))
      b_bits = UOp(Ops.BITCAST, dtypes.int16, (bv,))
    else:
      a_bits = UOp(Ops.BITCAST, dtypes.int32, (a,))
      b_bits = UOp(Ops.BITCAST, dtypes.int32, (bv,))
    return UOp(Ops.CMPLT, dtypes.bool, (a_bits, b_bits)), dtypes.bool

  if name == 'GT_NEG_ZERO' and len(args) == 2:
    a, dt = arg(0, hint); bv, _ = arg(1, hint)
    if dt == dtypes.float64:
      a_bits = UOp(Ops.BITCAST, dtypes.int64, (a,))
      b_bits = UOp(Ops.BITCAST, dtypes.int64, (bv,))
    elif dt == dtypes.float16:
      a_bits = UOp(Ops.BITCAST, dtypes.int16, (a,))
      b_bits = UOp(Ops.BITCAST, dtypes.int16, (bv,))
    else:
      a_bits = UOp(Ops.BITCAST, dtypes.int32, (a,))
      b_bits = UOp(Ops.BITCAST, dtypes.int32, (bv,))
    return UOp(Ops.CMPLT, dtypes.bool, (b_bits, a_bits)), dtypes.bool

  if name == 'SAT8' and len(args) == 1:
    v, dt = arg(0, hint)
    lo, hi = b.const(-128, dt), b.const(127, dt)
    clamped_lo = UOp(Ops.WHERE, dt, (UOp(Ops.CMPLT, dtypes.bool, (v, lo)), lo, v))
    return UOp(Ops.WHERE, dt, (UOp(Ops.CMPLT, dtypes.bool, (hi, clamped_lo)), hi, clamped_lo)), dt

  # v_min/v_max functions
  if name.startswith('v_min_') and len(args) == 2:
    a, dt = arg(0, hint); bv, _ = arg(1, hint)
    return UOp(Ops.WHERE, dt, (UOp(Ops.CMPLT, dtypes.bool, (a, bv)), a, bv)), dt
  if name.startswith('v_max_') and len(args) == 2:
    a, dt = arg(0, hint); bv, _ = arg(1, hint)
    return UOp(Ops.WHERE, dt, (UOp(Ops.CMPLT, dtypes.bool, (bv, a)), a, bv)), dt

  raise ValueError(f"Unknown function: {name}")

# ═══════════════════════════════════════════════════════════════════════════════
# STATEMENT TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════════════

def _get_lhs_info(lhs, b: UOpBuilder) -> tuple[str, DType, int|None, int|None, str|None]:
  """Extract assignment target info: (var_name, dtype, hi_bit, lo_bit, idx_var)"""
  match lhs:
    case Typed(Var(name), qdt): return name, _qdt(qdt), None, None, None
    case Typed(Slice(Var(name), Const(hi, _), Const(lo, _)), qdt): return name, _qdt(qdt), int(hi), int(lo), None
    case Typed(Index(Typed(Var(name), _), Var(idx)), _): return name, dtypes.uint64, None, None, idx
    case Typed(Index(Var(name), Var(idx)), qdt): return name, _qdt(qdt), None, None, idx
    case Slice(Typed(Var(name), _), Const(hi, _), Const(lo, _)): return name, dtypes.uint32, int(hi), int(lo), None
    case Slice(Var(name), Const(hi, _), Const(lo, _)): return name, dtypes.uint32, int(hi), int(lo), None
    case Index(Typed(Var(name), qdt), Var(idx)): return name, _qdt(qdt), None, None, idx
    case Var(name): return name, dtypes.uint32, None, None, None
  raise ValueError(f"Cannot parse LHS: {lhs}")

def transform_stmt(stmt, b: UOpBuilder):
  """Transform statement and update builder state."""
  match stmt:
    case Declare(_, _): pass  # Skip declarations

    case Assign(lhs, rhs):
      var, dtype, hi, lo, idx_var = _get_lhs_info(lhs, b)

      # Bit index assignment: D0.u64[laneId] = expr
      if idx_var is not None:
        base = b.vars.get(var)
        idx = b.vars.get(idx_var)
        if base is None: raise ValueError(f"Unknown variable: {var}")
        if idx is None: raise ValueError(f"Unknown index variable: {idx_var}")
        cond, _ = transform_expr(rhs, b)
        one = b.const(1, dtype)
        bit_mask = UOp(Ops.SHL, dtype, (one, b.cast(idx, dtype)))
        inv_mask = UOp(Ops.XOR, dtype, (bit_mask, b.const(-1, dtype)))
        cleared = UOp(Ops.AND, dtype, (base, inv_mask))
        cond_ext = b.cast(cond, dtype)
        cond_bit = UOp(Ops.SHL, dtype, (UOp(Ops.AND, dtype, (cond_ext, one)), b.cast(idx, dtype)))
        result = UOp(Ops.OR, dtype, (cleared, cond_bit))
        b.vars[var] = result
        if var in ('D0', 'D1', 'SCC', 'VCC', 'EXEC', 'PC'):
          b.outputs.append((var, result, dtype))
        return

      # Bit range assignment: D0[31:16].f16 = expr
      if hi is not None and lo is not None:
        if hi < lo: hi, lo = lo, hi
        base = b.vars[var] if var in b.vars else b.const(0, dtypes.uint32)
        rhs_uop, _ = transform_expr(rhs, b, dtype)
        if _is_float(dtype):
          if dtype == dtypes.float16:
            rhs_bits = UOp(Ops.BITCAST, dtypes.uint16, (rhs_uop,))
            rhs_bits = b.cast(rhs_bits, dtypes.uint32)
          else:
            rhs_bits = UOp(Ops.BITCAST, dtypes.uint32 if dtype == dtypes.float32 else dtypes.uint64, (rhs_uop,))
        else:
          rhs_bits = b.cast(rhs_uop, dtypes.uint32)
        width = hi - lo + 1
        mask = (1 << width) - 1
        shifted_val = UOp(Ops.SHL, dtypes.uint32, (UOp(Ops.AND, dtypes.uint32, (rhs_bits, b.const(mask, dtypes.uint32))), b.const(lo, dtypes.uint32)))
        inv_mask = ~(mask << lo) & 0xffffffff
        cleared = UOp(Ops.AND, dtypes.uint32, (b.cast(base, dtypes.uint32), b.const(inv_mask, dtypes.uint32)))
        result = UOp(Ops.OR, dtypes.uint32, (cleared, shifted_val))
        b.vars[var] = result
        if var in ('D0', 'D1', 'SCC', 'VCC'):
          b.outputs = [(n, u, d) for n, u, d in b.outputs if n != var]
          b.outputs.append((var, result, dtypes.uint32))
        return

      # Simple assignment
      rhs_uop, _ = transform_expr(rhs, b, dtype)
      b.vars[var] = rhs_uop
      if dtype.itemsize == 8 and var in ('D0', 'D1', 'S0', 'S1'):
        b.vars[var + '_64'] = rhs_uop
      if var in ('D0', 'D1', 'SCC', 'VCC', 'EXEC', 'PC'):
        b.outputs.append((var, rhs_uop, dtype))

    case If(branches):
      _transform_if(branches, b)

    case For(var, start, end, body):
      _transform_for(var, start, end, body, b)

def _transform_if(branches: tuple, b: UOpBuilder):
  """Transform if/elsif/else to nested WHERE expressions."""
  # Parse all conditions
  parsed_branches = []
  for cond, body in branches:
    cond_uop = transform_expr(cond, b)[0] if cond else None
    parsed_branches.append((cond_uop, body))

  # Collect all assigned variables
  assigned_vars = set()
  for _, body in parsed_branches:
    for stmt in body:
      if isinstance(stmt, Assign):
        var, _, _, _, _ = _get_lhs_info(stmt.lhs, b)
        assigned_vars.add(var)

  # Build nested WHERE for each variable
  for var in assigned_vars:
    # Determine dtype from first assignment
    dtype = dtypes.uint32
    for _, body in parsed_branches:
      for stmt in body:
        if isinstance(stmt, Assign):
          v, dt, _, _, _ = _get_lhs_info(stmt.lhs, b)
          if v == var: dtype = dt; break

    curr_val = b.vars[var] if var in b.vars else b.const(0, dtype)
    result = curr_val

    # Process branches in reverse order
    for cond_uop, body in reversed(parsed_branches):
      branch_val = None
      for stmt in body:
        if isinstance(stmt, Assign):
          v, dt, _, _, _ = _get_lhs_info(stmt.lhs, b)
          if v == var:
            branch_val, _ = transform_expr(stmt.rhs, b, dt)
            dtype = dt
            break

      if branch_val is not None:
        if cond_uop is None:
          result = branch_val
        else:
          if result.dtype != branch_val.dtype:
            result = b.cast(result, branch_val.dtype)
          result = UOp(Ops.WHERE, branch_val.dtype, (cond_uop, branch_val, result))

    # Store result
    if _is_float(dtype):
      result_bits = UOp(Ops.BITCAST, dtypes.uint32 if dtype == dtypes.float32 else dtypes.uint64, (result,))
      b.vars[var] = result_bits
      if var in ('D0', 'D1', 'SCC', 'VCC', 'EXEC', 'PC'):
        b.outputs = [(n, u, d) for n, u, d in b.outputs if n != var]
        b.outputs.append((var, result_bits, dtypes.uint32 if dtype == dtypes.float32 else dtypes.uint64))
    else:
      b.vars[var] = result
      if var in ('D0', 'D1', 'SCC', 'VCC', 'EXEC', 'PC'):
        b.outputs = [(n, u, d) for n, u, d in b.outputs if n != var]
        b.outputs.append((var, result, dtype))

def _transform_for(var: str, start, end, body: tuple, b: UOpBuilder):
  """Unroll for loop and transform body."""
  start_val = start.value if isinstance(start, Const) else int(transform_expr(start, b)[0].arg)
  end_val = end.value if isinstance(end, Const) else int(transform_expr(end, b)[0].arg)

  for loop_val in range(int(start_val), int(end_val) + 1):
    # Set loop variable
    b.vars[var] = b.const(loop_val, dtypes.uint32)

    for stmt in body:
      if isinstance(stmt, If):
        _transform_if(stmt.branches, b)
      elif isinstance(stmt, Assign):
        transform_stmt(stmt, b)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _float_to_bits(val: float, dtype: DType) -> int:
  if dtype == dtypes.float32:
    return struct.unpack('<I', struct.pack('<f', val))[0]
  elif dtype == dtypes.float16:
    if math.isnan(val): return 0x7e00
    if math.isinf(val): return 0x7c00 if val > 0 else 0xfc00
    if abs(val) > 65504.0: return 0x7c00 if val > 0 else 0xfc00
    if abs(val) < 6.103515625e-05 and val != 0: return 0x0000 if val > 0 else 0x8000
    return struct.unpack('<H', struct.pack('<e', val))[0]
  elif dtype == dtypes.float64:
    return struct.unpack('<Q', struct.pack('<d', val))[0]
  return int(val)

# ═══════════════════════════════════════════════════════════════════════════════
# COMPILED FUNCTION GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _compile_pseudocode(pseudocode: str) -> tuple[UOp, list[tuple[str, DType]], dict[str, UOp]]:
  """Compile pseudocode to UOp graph using qcode parser."""
  ast = parse(pseudocode)
  b = UOpBuilder()

  for stmt in ast:
    transform_stmt(stmt, b)

  sink = b.build_sink()
  output_info = [(name, dtype) for name, _, dtype in b.outputs]
  return sink, output_info, b.input_vars

def _make_uop_fn(sink: UOp, output_info: list[tuple[str, DType]], input_vars: dict[str, UOp]):
  """Create a runtime function that evaluates the UOp graph via simplify."""
  def fn(s0, s1, s2, d0, scc, vcc, laneId, exec_mask, literal, VGPR, src0_idx=0, vdst_idx=0, pc=None):
    if literal is not None:
      simm16 = literal if -32768 <= literal <= 32767 else (literal - 65536 if literal < 65536 else 0)
    else:
      simm16 = 0
    dvars = {
      input_vars['S0']: UOp.const(dtypes.uint32, s0 & 0xffffffff),
      input_vars['S1']: UOp.const(dtypes.uint32, s1 & 0xffffffff),
      input_vars['S2']: UOp.const(dtypes.uint32, s2 & 0xffffffff),
      input_vars['D0']: UOp.const(dtypes.uint32, d0 & 0xffffffff),
      input_vars['S0_64']: UOp.const(dtypes.uint64, s0),
      input_vars['S1_64']: UOp.const(dtypes.uint64, s1),
      input_vars['S2_64']: UOp.const(dtypes.uint64, s2),
      input_vars['D0_64']: UOp.const(dtypes.uint64, d0),
      input_vars['SCC']: UOp.const(dtypes.uint32, scc),
      input_vars['VCC']: UOp.const(dtypes.uint64, vcc),
      input_vars['EXEC']: UOp.const(dtypes.uint64, exec_mask),
      input_vars['laneId']: UOp.const(dtypes.uint32, laneId),
      input_vars['SIMM16']: UOp.const(dtypes.int32, simm16),
      input_vars['SIMM32']: UOp.const(dtypes.uint32, literal if literal is not None else 0),
      input_vars['PC']: UOp.const(dtypes.uint64, pc if pc is not None else 0),
    }

    simplified_sink = sink.substitute(dvars).simplify()
    assert simplified_sink.op == Ops.SINK, f"expected SINK, got {simplified_sink.op}"

    result = {}
    for i, (name, dtype) in enumerate(output_info):
      out_uop = simplified_sink.src[i]
      assert out_uop.op == Ops.CONST, f"simplify did not produce CONST for {name}, got {out_uop.op}"
      val = out_uop.arg
      if _is_float(dtype):
        bits = _float_to_bits(val, dtype)
      else:
        bits = int(val) & (0xffffffff if dtype in (dtypes.uint32, dtypes.int32) else 0xffffffffffffffff)
      result[name] = bits

    return result
  return fn

# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

SUPPORTED_OPS: set[str] = {
  # VOP (153 ops)
  'V_ADD3_U32', 'V_ADD_CO_CI_U32', 'V_ADD_CO_U32', 'V_ADD_F16', 'V_ADD_F32', 'V_ADD_F64', 'V_ADD_LSHL_U32', 'V_ADD_NC_I16', 'V_ADD_NC_I32', 'V_ADD_NC_U16', 'V_ADD_NC_U32',
  'V_ALIGNBIT_B32', 'V_ALIGNBYTE_B32', 'V_AND_B16', 'V_AND_B32', 'V_AND_OR_B32', 'V_ASHRREV_I16', 'V_ASHRREV_I32', 'V_ASHRREV_I64',
  'V_BFE_I32', 'V_BFE_U32', 'V_BFI_B32', 'V_BFM_B32',
  'V_CNDMASK_B16', 'V_CNDMASK_B32', 'V_COS_F16', 'V_COS_F32', 'V_CUBEID_F32', 'V_CUBESC_F32',
  'V_CVT_F16_F32', 'V_CVT_F32_F16', 'V_CVT_F32_I32', 'V_CVT_F32_U32',
  'V_CVT_F32_UBYTE0', 'V_CVT_F32_UBYTE1', 'V_CVT_F32_UBYTE2', 'V_CVT_F32_UBYTE3', 'V_CVT_FLOOR_I32_F32',
  'V_CVT_I32_F32', 'V_CVT_I32_I16', 'V_CVT_NEAREST_I32_F32', 'V_CVT_PK_I16_F32', 'V_CVT_PK_U16_F32',
  'V_CVT_PK_U8_F32', 'V_CVT_U32_F32', 'V_CVT_U32_U16',
  'V_DOT2_F16_F16', 'V_DOT2_F32_F16', 'V_DOT2ACC_F32_F16',
  'V_FMA_DX9_ZERO_F32', 'V_FMA_F16', 'V_FMA_F32', 'V_FMA_F64', 'V_FMAAK_F16', 'V_FMAAK_F32',
  'V_FMAC_DX9_ZERO_F32', 'V_FMAC_F16', 'V_FMAC_F32', 'V_FMAMK_F16', 'V_FMAMK_F32',
  'V_FREXP_EXP_I16_F16', 'V_FREXP_EXP_I32_F32', 'V_FREXP_EXP_I32_F64',
  'V_LERP_U8', 'V_LOG_F16', 'V_LOG_F32',
  'V_LSHL_ADD_U32', 'V_LSHL_OR_B32', 'V_LSHLREV_B16', 'V_LSHLREV_B32', 'V_LSHLREV_B64', 'V_LSHRREV_B16', 'V_LSHRREV_B32', 'V_LSHRREV_B64',
  'V_MAD_I16', 'V_MAD_I32_I16', 'V_MAD_I32_I24', 'V_MAD_U16', 'V_MAD_U32_U16', 'V_MAD_U32_U24',
  'V_MAX_I16', 'V_MAX_I32', 'V_MAX_U16', 'V_MAX_U32', 'V_MIN_I16', 'V_MIN_I32', 'V_MIN_U16', 'V_MIN_U32',
  'V_MOV_B16', 'V_MOV_B32', 'V_MSAD_U8', 'V_MUL_DX9_ZERO_F32', 'V_MUL_F16', 'V_MUL_F32', 'V_MUL_F64',
  'V_MUL_HI_I32', 'V_MUL_HI_I32_I24', 'V_MUL_HI_U32', 'V_MUL_HI_U32_U24', 'V_MUL_I32_I24', 'V_MUL_LO_U16', 'V_MUL_LO_U32', 'V_MUL_U32_U24',
  'V_NOT_B16', 'V_NOT_B32', 'V_OR3_B32', 'V_OR_B16', 'V_OR_B32', 'V_PACK_B32_F16', 'V_PK_FMAC_F16', 'V_RCP_F16',
  'V_RCP_F32', 'V_RCP_F64', 'V_RCP_IFLAG_F32', 'V_RSQ_F16', 'V_RSQ_F32', 'V_RSQ_F64',
  'V_PK_ADD_F16', 'V_PK_ADD_I16', 'V_PK_ADD_U16', 'V_PK_ASHRREV_I16', 'V_PK_FMA_F16',
  'V_PK_LSHLREV_B16', 'V_PK_LSHRREV_B16', 'V_PK_MAD_I16', 'V_PK_MAD_U16',
  'V_PK_MAX_I16', 'V_PK_MAX_U16', 'V_PK_MIN_I16', 'V_PK_MIN_U16', 'V_PK_MUL_F16', 'V_PK_MUL_LO_U16',
  'V_PK_SUB_I16', 'V_PK_SUB_U16',
  'V_RNDNE_F16', 'V_RNDNE_F32', 'V_RNDNE_F64',
  'V_SAD_U8', 'V_SAD_U16', 'V_SAD_U32', 'V_SIN_F16', 'V_SIN_F32', 'V_SQRT_F16', 'V_SQRT_F32', 'V_SQRT_F64',
  'V_CVT_F32_F64', 'V_CVT_F64_F32', 'V_CVT_F64_I32', 'V_CVT_F64_U32', 'V_CVT_I32_F64', 'V_CVT_U32_F64',
  'V_CVT_NORM_I16_F16', 'V_CVT_NORM_U16_F16', 'V_CVT_PK_NORM_I16_F16', 'V_CVT_PK_NORM_U16_F16', 'V_CVT_PK_RTZ_F16_F32',
  'V_SUB_CO_CI_U32', 'V_SUB_CO_U32', 'V_SUB_F16', 'V_SUB_F32', 'V_SUB_NC_I16', 'V_SUB_NC_I32', 'V_SUB_NC_U16', 'V_SUB_NC_U32',
  'V_SUBREV_CO_CI_U32', 'V_SUBREV_CO_U32', 'V_SUBREV_F16', 'V_SUBREV_F32', 'V_SUBREV_NC_U32', 'V_SWAP_B16', 'V_SWAP_B32',
  'V_TRUNC_F16', 'V_TRUNC_F32', 'V_TRUNC_F64', 'V_WRITELANE_B32', 'V_XAD_U32', 'V_XNOR_B32', 'V_XOR3_B32', 'V_XOR_B16', 'V_XOR_B32',
  'V_CVT_F16_I16', 'V_CVT_F16_U16', 'V_CVT_I16_F16', 'V_CVT_U16_F16',
  'V_EXP_F16', 'V_EXP_F32',
  'V_LDEXP_F16', 'V_LDEXP_F32', 'V_LDEXP_F64',
  'V_CUBEMA_F32', 'V_CUBETC_F32',
  'V_SAT_PK_U8_I16',
  'V_MAX3_I16', 'V_MAX3_I32', 'V_MAX3_U16', 'V_MAX3_U32',
  'V_MIN3_I16', 'V_MIN3_I32', 'V_MIN3_U16', 'V_MIN3_U32',
  'V_MAXMIN_I32', 'V_MAXMIN_U32', 'V_MINMAX_I32', 'V_MINMAX_U32',
  # VOPC (112 ops)
  'V_CMP_EQ_F16', 'V_CMP_EQ_F32', 'V_CMP_EQ_F64', 'V_CMP_EQ_I16', 'V_CMP_EQ_I32', 'V_CMP_EQ_I64', 'V_CMP_EQ_U16', 'V_CMP_EQ_U32',
  'V_CMP_EQ_U64', 'V_CMP_F_F16', 'V_CMP_F_F32', 'V_CMP_F_F64', 'V_CMP_F_I32', 'V_CMP_F_I64', 'V_CMP_F_U32', 'V_CMP_F_U64',
  'V_CMP_GE_F16', 'V_CMP_GE_F32', 'V_CMP_GE_F64', 'V_CMP_GE_I16', 'V_CMP_GE_I32', 'V_CMP_GE_I64', 'V_CMP_GE_U16', 'V_CMP_GE_U32', 'V_CMP_GE_U64',
  'V_CMP_GT_F16', 'V_CMP_GT_F32', 'V_CMP_GT_F64', 'V_CMP_GT_I16', 'V_CMP_GT_I32', 'V_CMP_GT_I64', 'V_CMP_GT_U16', 'V_CMP_GT_U32', 'V_CMP_GT_U64',
  'V_CMP_LE_F16', 'V_CMP_LE_F32', 'V_CMP_LE_F64', 'V_CMP_LE_I16', 'V_CMP_LE_I32', 'V_CMP_LE_I64', 'V_CMP_LE_U16', 'V_CMP_LE_U32', 'V_CMP_LE_U64',
  'V_CMP_LG_F16', 'V_CMP_LG_F32', 'V_CMP_LG_F64',
  'V_CMP_LT_F16', 'V_CMP_LT_F32', 'V_CMP_LT_F64', 'V_CMP_LT_I16', 'V_CMP_LT_I32', 'V_CMP_LT_I64', 'V_CMP_LT_U16', 'V_CMP_LT_U32', 'V_CMP_LT_U64',
  'V_CMP_NE_I16', 'V_CMP_NE_I32', 'V_CMP_NE_I64', 'V_CMP_NE_U16', 'V_CMP_NE_U32', 'V_CMP_NE_U64',
  'V_CMP_NEQ_F16', 'V_CMP_NEQ_F32', 'V_CMP_NEQ_F64',
  'V_CMP_NGE_F16', 'V_CMP_NGE_F32', 'V_CMP_NGE_F64', 'V_CMP_NGT_F16', 'V_CMP_NGT_F32', 'V_CMP_NGT_F64',
  'V_CMP_NLE_F16', 'V_CMP_NLE_F32', 'V_CMP_NLE_F64', 'V_CMP_NLG_F16', 'V_CMP_NLG_F32', 'V_CMP_NLG_F64',
  'V_CMP_NLT_F16', 'V_CMP_NLT_F32', 'V_CMP_NLT_F64',
  'V_CMP_O_F16', 'V_CMP_O_F32', 'V_CMP_O_F64', 'V_CMP_T_F16', 'V_CMP_T_F32', 'V_CMP_T_F64',
  'V_CMP_T_I32', 'V_CMP_T_I64', 'V_CMP_T_U32', 'V_CMP_T_U64', 'V_CMP_U_F16', 'V_CMP_U_F32', 'V_CMP_U_F64',
  # VOPCX (112 ops)
  'V_CMPX_EQ_F16', 'V_CMPX_EQ_F32', 'V_CMPX_EQ_F64', 'V_CMPX_EQ_I16', 'V_CMPX_EQ_I32', 'V_CMPX_EQ_I64', 'V_CMPX_EQ_U16', 'V_CMPX_EQ_U32',
  'V_CMPX_EQ_U64', 'V_CMPX_F_F16', 'V_CMPX_F_F32', 'V_CMPX_F_F64', 'V_CMPX_F_I32', 'V_CMPX_F_I64', 'V_CMPX_F_U32', 'V_CMPX_F_U64',
  'V_CMPX_GE_F16', 'V_CMPX_GE_F32', 'V_CMPX_GE_F64', 'V_CMPX_GE_I16', 'V_CMPX_GE_I32', 'V_CMPX_GE_I64', 'V_CMPX_GE_U16', 'V_CMPX_GE_U32', 'V_CMPX_GE_U64',
  'V_CMPX_GT_F16', 'V_CMPX_GT_F32', 'V_CMPX_GT_F64', 'V_CMPX_GT_I16', 'V_CMPX_GT_I32', 'V_CMPX_GT_I64', 'V_CMPX_GT_U16', 'V_CMPX_GT_U32', 'V_CMPX_GT_U64',
  'V_CMPX_LE_F16', 'V_CMPX_LE_F32', 'V_CMPX_LE_F64', 'V_CMPX_LE_I16', 'V_CMPX_LE_I32', 'V_CMPX_LE_I64', 'V_CMPX_LE_U16', 'V_CMPX_LE_U32', 'V_CMPX_LE_U64',
  'V_CMPX_LG_F16', 'V_CMPX_LG_F32', 'V_CMPX_LG_F64',
  'V_CMPX_LT_F16', 'V_CMPX_LT_F32', 'V_CMPX_LT_F64', 'V_CMPX_LT_I16', 'V_CMPX_LT_I32', 'V_CMPX_LT_I64', 'V_CMPX_LT_U16', 'V_CMPX_LT_U32', 'V_CMPX_LT_U64',
  'V_CMPX_NE_I16', 'V_CMPX_NE_I32', 'V_CMPX_NE_I64', 'V_CMPX_NE_U16', 'V_CMPX_NE_U32', 'V_CMPX_NE_U64',
  'V_CMPX_NEQ_F16', 'V_CMPX_NEQ_F32', 'V_CMPX_NEQ_F64',
  'V_CMPX_NGE_F16', 'V_CMPX_NGE_F32', 'V_CMPX_NGE_F64', 'V_CMPX_NGT_F16', 'V_CMPX_NGT_F32', 'V_CMPX_NGT_F64',
  'V_CMPX_NLE_F16', 'V_CMPX_NLE_F32', 'V_CMPX_NLE_F64', 'V_CMPX_NLG_F16', 'V_CMPX_NLG_F32', 'V_CMPX_NLG_F64',
  'V_CMPX_NLT_F16', 'V_CMPX_NLT_F32', 'V_CMPX_NLT_F64',
  'V_CMPX_O_F16', 'V_CMPX_O_F32', 'V_CMPX_O_F64', 'V_CMPX_T_F16', 'V_CMPX_T_F32', 'V_CMPX_T_F64',
  'V_CMPX_T_I32', 'V_CMPX_T_I64', 'V_CMPX_T_U32', 'V_CMPX_T_U64', 'V_CMPX_U_F16', 'V_CMPX_U_F32', 'V_CMPX_U_F64',
  # SOP (134 ops)
  'S_ABSDIFF_I32', 'S_ABS_I32', 'S_ADD_F16', 'S_ADD_F32', 'S_ADD_I32', 'S_ADD_U32', 'S_ADDC_U32', 'S_ADDK_I32',
  'S_AND_B32', 'S_AND_B64', 'S_AND_NOT0_SAVEEXEC_B32', 'S_AND_NOT0_SAVEEXEC_B64', 'S_AND_NOT0_WREXEC_B32', 'S_AND_NOT0_WREXEC_B64',
  'S_AND_NOT1_B32', 'S_AND_NOT1_B64', 'S_AND_NOT1_SAVEEXEC_B32', 'S_AND_NOT1_SAVEEXEC_B64', 'S_AND_NOT1_WREXEC_B32', 'S_AND_NOT1_WREXEC_B64',
  'S_AND_SAVEEXEC_B32', 'S_AND_SAVEEXEC_B64', 'S_ASHR_I32', 'S_ASHR_I64',
  'S_BCNT0_I32_B32', 'S_BCNT0_I32_B64', 'S_BCNT1_I32_B32', 'S_BCNT1_I32_B64',
  'S_BFE_I32', 'S_BFE_I64', 'S_BFE_U32', 'S_BFE_U64',
  'S_BFM_B32', 'S_BFM_B64', 'S_BITSET0_B32', 'S_BITSET0_B64', 'S_BITSET1_B32', 'S_BITSET1_B64',
  'S_CMOVK_I32', 'S_CMOV_B32', 'S_CMOV_B64', 'S_CSELECT_B32', 'S_CSELECT_B64',
  'S_CVT_F16_F32', 'S_CVT_F32_F16', 'S_CVT_F32_I32', 'S_CVT_F32_U32', 'S_CVT_HI_F32_F16', 'S_CVT_I32_F32', 'S_CVT_PK_RTZ_F16_F32', 'S_CVT_U32_F32',
  'S_DELAY_ALU', 'S_FMAAK_F32', 'S_FMAC_F16', 'S_FMAC_F32', 'S_FMAMK_F32',
  'S_LSHL_B32', 'S_LSHL_B64', 'S_LSHL1_ADD_U32', 'S_LSHL2_ADD_U32', 'S_LSHL3_ADD_U32', 'S_LSHL4_ADD_U32',
  'S_LSHR_B32', 'S_LSHR_B64', 'S_MAX_I32', 'S_MAX_U32', 'S_MIN_I32', 'S_MIN_U32', 'S_MOVK_I32', 'S_MOV_B32',
  'S_MOV_B64', 'S_MULK_I32', 'S_MUL_F16', 'S_MUL_F32', 'S_MUL_HI_I32', 'S_MUL_HI_U32', 'S_MUL_I32',
  'S_NAND_B32', 'S_NAND_B64', 'S_NAND_SAVEEXEC_B32', 'S_NAND_SAVEEXEC_B64',
  'S_NOP', 'S_NOR_B32', 'S_NOR_B64', 'S_NOR_SAVEEXEC_B32', 'S_NOR_SAVEEXEC_B64',
  'S_NOT_B32', 'S_NOT_B64', 'S_OR_B32', 'S_OR_B64',
  'S_OR_NOT0_SAVEEXEC_B32', 'S_OR_NOT0_SAVEEXEC_B64', 'S_OR_NOT1_B32', 'S_OR_NOT1_B64', 'S_OR_NOT1_SAVEEXEC_B32', 'S_OR_NOT1_SAVEEXEC_B64',
  'S_OR_SAVEEXEC_B32', 'S_OR_SAVEEXEC_B64',
  'S_PACK_HH_B32_B16', 'S_PACK_HL_B32_B16', 'S_PACK_LH_B32_B16', 'S_PACK_LL_B32_B16', 'S_RFE_B64', 'S_RNDNE_F16', 'S_RNDNE_F32',
  'S_SENDMSG_RTN_B32', 'S_SENDMSG_RTN_B64', 'S_SETPC_B64', 'S_SEXT_I32_I16', 'S_SEXT_I32_I8',
  'S_SUB_F16', 'S_SUB_F32', 'S_SUB_I32', 'S_SUB_U32', 'S_SUBB_U32',
  'S_TRUNC_F16', 'S_TRUNC_F32', 'S_VERSION',
  'S_BITCMP0_B32', 'S_BITCMP0_B64', 'S_BITCMP1_B32', 'S_BITCMP1_B64',
  'S_MAX_F16', 'S_MAX_F32', 'S_MIN_F16', 'S_MIN_F32',
  'S_WAITCNT_EXPCNT', 'S_WAITCNT_LGKMCNT', 'S_WAITCNT_VMCNT', 'S_WAITCNT_VSCNT',
  'S_BRANCH', 'S_CALL_B64', 'S_CBRANCH_EXECNZ', 'S_CBRANCH_EXECZ', 'S_CBRANCH_SCC0', 'S_CBRANCH_SCC1',
  'S_CBRANCH_VCCNZ', 'S_CBRANCH_VCCZ', 'S_GETPC_B64',
  'S_XNOR_B32', 'S_XNOR_B64', 'S_XNOR_SAVEEXEC_B32', 'S_XNOR_SAVEEXEC_B64',
  'S_XOR_B32', 'S_XOR_B64', 'S_XOR_SAVEEXEC_B32', 'S_XOR_SAVEEXEC_B64',
  # SOPC (54 ops)
  'S_CMPK_EQ_I32', 'S_CMPK_EQ_U32', 'S_CMPK_GE_I32', 'S_CMPK_GE_U32', 'S_CMPK_GT_I32', 'S_CMPK_GT_U32', 'S_CMPK_LE_I32', 'S_CMPK_LE_U32',
  'S_CMPK_LG_I32', 'S_CMPK_LG_U32', 'S_CMPK_LT_I32', 'S_CMPK_LT_U32',
  'S_CMP_EQ_F16', 'S_CMP_EQ_F32', 'S_CMP_EQ_I32', 'S_CMP_EQ_U32', 'S_CMP_EQ_U64',
  'S_CMP_GE_F16', 'S_CMP_GE_F32', 'S_CMP_GE_I32', 'S_CMP_GE_U32',
  'S_CMP_GT_F16', 'S_CMP_GT_F32', 'S_CMP_GT_I32', 'S_CMP_GT_U32',
  'S_CMP_LE_F16', 'S_CMP_LE_F32', 'S_CMP_LE_I32', 'S_CMP_LE_U32',
  'S_CMP_LG_F16', 'S_CMP_LG_F32', 'S_CMP_LG_I32', 'S_CMP_LG_U32', 'S_CMP_LG_U64',
  'S_CMP_LT_F16', 'S_CMP_LT_F32', 'S_CMP_LT_I32', 'S_CMP_LT_U32',
  'S_CMP_NEQ_F16', 'S_CMP_NEQ_F32', 'S_CMP_NGE_F16', 'S_CMP_NGE_F32', 'S_CMP_NGT_F16', 'S_CMP_NGT_F32',
  'S_CMP_NLE_F16', 'S_CMP_NLE_F32', 'S_CMP_NLG_F16', 'S_CMP_NLG_F32', 'S_CMP_NLT_F16', 'S_CMP_NLT_F32',
  'S_CMP_O_F16', 'S_CMP_O_F32', 'S_CMP_U_F16', 'S_CMP_U_F32',
}

@functools.cache
def compile_uop(cls_name: str, op_name: str, pseudocode: str):
  """Compile pseudocode to UOp-based function. Returns None if unsupported."""
  if op_name not in SUPPORTED_OPS: return None
  sink, output_info, input_vars = _compile_pseudocode(pseudocode)
  return _make_uop_fn(sink, output_info, input_vars)
