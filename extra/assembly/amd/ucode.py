# UOp-based pseudocode compiler for AMD GPU instruction emulation
import functools, struct, math
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes, DType
from extra.assembly.amd.qcode import parse, Const, Var, Typed, Slice, Index, Cast, Unary, Binary, Ternary, Call, Pack, Assign, Declare, If, For, QDType

# Type mapping from qcode types to tinygrad dtypes
QDTYPE_MAP = {
  QDType.F64: dtypes.float64, QDType.F32: dtypes.float32, QDType.F16: dtypes.float16,
  QDType.U64: dtypes.uint64, QDType.U32: dtypes.uint32, QDType.U24: dtypes.uint24, QDType.U16: dtypes.uint16, QDType.U8: dtypes.uint8,
  QDType.I64: dtypes.int64, QDType.I32: dtypes.int32, QDType.I24: dtypes.int24, QDType.I16: dtypes.int16, QDType.I8: dtypes.int8,
  QDType.B128: dtypes.uint64, QDType.B64: dtypes.uint64, QDType.B32: dtypes.uint32, QDType.B16: dtypes.uint16, QDType.B8: dtypes.uint8,
  QDType.U1: dtypes.uint32, QDType.I1: dtypes.int32, QDType.U3: dtypes.uint32, QDType.U4: dtypes.uint32, QDType.I4: dtypes.int32,
}
SIGNED = (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64)
FLOATS = (dtypes.float16, dtypes.float32, dtypes.float64)
def _qdt(qd: QDType) -> DType: return QDTYPE_MAP.get(qd, dtypes.uint32)

def _cast(x: UOp, dtype: DType) -> UOp:
  return x if x.dtype == dtype else UOp(Ops.BITCAST if dtype.itemsize == x.dtype.itemsize else Ops.CAST, dtype, (x,))

# Input variables for the UOp graph
INPUT_VARS = {
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

class Ctx:
  """Compilation context - tracks variables and outputs."""
  def __init__(self):
    self.vars: dict[str, UOp] = dict(INPUT_VARS)
    self.outputs: list[tuple[str, UOp, DType]] = []

def _expr(node, ctx: Ctx, hint: DType = None) -> UOp:
  """Transform qcode AST expression to UOp."""
  match node:
    case Const(val, qdt):
      dt = _qdt(qdt) if qdt != QDType.I32 or hint is None else hint
      if isinstance(val, float) and dt not in FLOATS: dt = dtypes.float32
      return UOp.const(dt, val)

    case Var(name):
      if name == 'PI': return UOp.const(hint or dtypes.float64, math.pi)
      if name in ('INF', '+INF', '-INF'): return UOp.const(hint or dtypes.float64, float('-inf') if name == '-INF' else float('inf'))
      if name in ('WAVE_MODE.IEEE', 'WAVE32'): return UOp.const(dtypes.uint32, 1)
      if name in ('WAVE64', 'ROUND_MODE'): return UOp.const(dtypes.uint32, 0)
      if name in ('VCCZ', 'EXECZ'):
        return _cast(UOp(Ops.CMPEQ, dtypes.bool, (ctx.vars.get('VCC' if name == 'VCCZ' else 'EXEC'), UOp.const(dtypes.uint64, 0))), dtypes.uint32)
      if name.startswith('eval '): return ctx.vars.get('_eval', UOp.const(dtypes.uint32, 0))
      if name not in ctx.vars: raise ValueError(f"Unknown variable: {name}")
      return _cast(ctx.vars[name], hint or ctx.vars[name].dtype)

    case Typed(expr, qdt):
      dt = _qdt(qdt)
      if isinstance(expr, Var):
        if expr.name in ('VCCZ', 'EXECZ'):
          return _cast(UOp(Ops.CMPEQ, dtypes.bool, (ctx.vars.get('VCC' if expr.name == 'VCCZ' else 'EXEC'), UOp.const(dtypes.uint64, 0))), dtypes.uint32)
        vn = expr.name + '_64' if qdt in (QDType.F64, QDType.U64, QDType.I64, QDType.B64) and expr.name.isupper() else expr.name
        base = ctx.vars.get(vn) if vn in ctx.vars else ctx.vars.get(expr.name)
        if base is None: raise ValueError(f"Unknown variable: {expr.name}")
        if qdt == QDType.U24: return UOp(Ops.AND, dtypes.uint32, (base, UOp.const(dtypes.uint32, 0xffffff)))
        if qdt == QDType.I24:
          masked = UOp(Ops.AND, dtypes.uint32, (base, UOp.const(dtypes.uint32, 0xffffff)))
          return UOp(Ops.SUB, dtypes.int32, (UOp(Ops.XOR, dtypes.int32, (masked, UOp.const(dtypes.int32, 0x800000))), UOp.const(dtypes.int32, 0x800000)))
        if dt == dtypes.float16:
          return UOp(Ops.BITCAST, dtypes.float16, (UOp(Ops.AND, dtypes.uint16, (_cast(base, dtypes.uint16), UOp.const(dtypes.uint16, 0xffff))),))
        if dt in FLOATS: return UOp(Ops.BITCAST, dt, (base,))
        if dt in SIGNED:
          base64 = ctx.vars.get(expr.name + '_64') if (expr.name + '_64') in ctx.vars else base
          return _cast(base64 if dt == dtypes.int64 else base, dt)
        return _cast(base, dt)
      inner = _expr(expr, ctx, dt)
      if dt == dtypes.float16: return UOp(Ops.BITCAST, dt, (_cast(inner, dtypes.uint16),))
      if dt in FLOATS: return UOp(Ops.BITCAST, dt, (inner,))
      return _cast(inner, dt)

    case Slice(expr, hi, lo):
      base, hi_uop, lo_uop = _expr(expr, ctx), _expr(hi, ctx), _expr(lo, ctx)
      if hi_uop.op == Ops.CONST and lo_uop.op == Ops.CONST:
        hi_val, lo_val = int(hi_uop.arg), int(lo_uop.arg)
        if hi_val < lo_val: hi_val, lo_val = lo_val, hi_val
        shifted = UOp(Ops.SHR, base.dtype, (base, UOp.const(base.dtype, lo_val))) if lo_val else base
        return UOp(Ops.AND, dtypes.uint32, (_cast(shifted, dtypes.uint32), UOp.const(dtypes.uint32, (1 << (hi_val - lo_val + 1)) - 1)))
      raise ValueError(f"Non-constant slice bounds: {node}")

    case Index(expr, idx):
      base, idx_uop = _expr(expr, ctx), _expr(idx, ctx)
      return UOp(Ops.AND, dtypes.uint32, (_cast(UOp(Ops.SHR, base.dtype, (base, _cast(idx_uop, base.dtype))), dtypes.uint32), UOp.const(dtypes.uint32, 1)))

    case Cast(bits, typ, expr):
      dt = {(16,'I'): dtypes.int16, (16,'U'): dtypes.uint16, (16,'F'): dtypes.float16, (32,'I'): dtypes.int32, (32,'U'): dtypes.uint32,
            (32,'F'): dtypes.float32, (32,'B'): dtypes.uint32, (64,'I'): dtypes.int64, (64,'U'): dtypes.uint64, (64,'F'): dtypes.float64,
            (64,'B'): dtypes.uint64}.get((bits, typ), dtypes.uint32)
      inner = _expr(expr, ctx, dt)
      if typ == 'F': return UOp(Ops.CAST, dt, (inner,))
      if inner.dtype in (dtypes.uint32, dtypes.int32, dtypes.uint64, dtypes.int64) and inner.dtype.itemsize * 8 == bits: return _cast(inner, dt)
      if typ == 'I' and inner.dtype in SIGNED: return UOp(Ops.CAST, dt, (inner,))
      return _cast(inner, dt)

    case Unary(op, expr):
      val = _expr(expr, ctx, hint)
      if op == '-': return UOp(Ops.NEG, val.dtype, (val,))
      if op == '~': return UOp(Ops.XOR, val.dtype, (val, UOp.const(val.dtype, -1)))
      if op == '!': return UOp(Ops.CMPEQ, dtypes.bool, (val, UOp.const(val.dtype, 0)))
      raise ValueError(f"Unknown unary op: {op}")

    case Binary(op, left, right):
      l = _expr(left, ctx, hint)
      r = _expr(right, ctx, l.dtype if l.dtype in FLOATS else hint)
      # For 32/64-bit signed arithmetic, use unsigned to avoid overflow issues during constant folding
      # Unwrap bitcasts to avoid double-cast chains that confuse the simplifier
      if op in ('+', '-', '*') and l.dtype in (dtypes.int32, dtypes.int64):
        udt = {dtypes.int32: dtypes.uint32, dtypes.int64: dtypes.uint64}[l.dtype]
        lu = l.src[0] if l.op == Ops.BITCAST and l.src[0].dtype == udt else _cast(l, udt)
        ru = r.src[0] if r.op == Ops.BITCAST and r.src[0].dtype == udt else _cast(r, udt)
        return UOp(Ops.ADD if op == '+' else Ops.SUB if op == '-' else Ops.MUL, udt, (lu, ru))
      result_dt = l.dtype if l.dtype in FLOATS else r.dtype if r.dtype in FLOATS else l.dtype

      ops = {'+': Ops.ADD, '-': Ops.SUB, '*': Ops.MUL, '/': Ops.FDIV, '&': Ops.AND, '|': Ops.OR, '^': Ops.XOR, '<<': Ops.SHL,
             '==': Ops.CMPEQ, '!=': Ops.CMPNE, '<>': Ops.CMPNE}
      if op in ops:
        uop_op = ops[op]
        return UOp(uop_op, dtypes.bool if uop_op in (Ops.CMPEQ, Ops.CMPNE) else result_dt, (l, r))
      if op == '>>': return UOp(Ops.SHR, result_dt, (l, r))
      if op == '<': return UOp(Ops.CMPLT, dtypes.bool, (l, r))
      if op == '>': return UOp(Ops.CMPLT, dtypes.bool, (r, l))
      if op == '>=': return UOp(Ops.XOR, dtypes.bool, (UOp(Ops.CMPLT, dtypes.bool, (l, r)), UOp.const(dtypes.bool, True)))
      if op == '<=': return UOp(Ops.XOR, dtypes.bool, (UOp(Ops.CMPLT, dtypes.bool, (r, l)), UOp.const(dtypes.bool, True)))
      if op in ('||', '&&'):
        one, zero = UOp.const(dtypes.uint32, 1), UOp.const(dtypes.uint32, 0)
        inner = UOp(Ops.WHERE, dtypes.uint32, (r, one, zero))
        return UOp(Ops.WHERE, dtypes.uint32, (l, one if op == '||' else inner, inner if op == '||' else zero))
      if op == '**':
        exp = UOp(Ops.CAST, result_dt, (r,)) if r.dtype != result_dt else r
        if l.op == Ops.CONST and l.arg == 2.0: return UOp(Ops.EXP2, result_dt, (exp,))
        return UOp(Ops.EXP2, result_dt, (UOp(Ops.MUL, result_dt, (exp, UOp(Ops.LOG2, result_dt, (l,)))),))
      if op == '%':
        div = UOp(Ops.IDIV, result_dt, (l, r))
        return UOp(Ops.SUB, result_dt, (l, UOp(Ops.MUL, result_dt, (div, r))))
      raise ValueError(f"Unknown binary op: {op}")

    case Ternary(cond, t, f):
      c, tv = _expr(cond, ctx), _expr(t, ctx, hint)
      fv = _expr(f, ctx, tv.dtype)
      return UOp(Ops.WHERE, tv.dtype, (c, tv, fv))

    case Call(name, args): return _transform_call(name, [_expr(a, ctx, hint) for a in args], hint)

    case Pack(exprs):
      if len(exprs) == 2:
        hi, lo = _expr(exprs[0], ctx), _expr(exprs[1], ctx)
        if lo.dtype.itemsize >= 4:
          return UOp(Ops.OR, dtypes.uint64, (UOp(Ops.SHL, dtypes.uint64, (_cast(hi, dtypes.uint64), UOp.const(dtypes.uint64, 32))), _cast(lo, dtypes.uint64)))
        return UOp(Ops.OR, dtypes.uint32, (UOp(Ops.SHL, dtypes.uint32, (_cast(hi, dtypes.uint32), UOp.const(dtypes.uint32, 16))),
                                           UOp(Ops.AND, dtypes.uint32, (_cast(lo, dtypes.uint32), UOp.const(dtypes.uint32, 0xffff)))))
      raise ValueError(f"Pack with {len(exprs)} elements not supported")
  raise ValueError(f"Cannot transform expression: {node}")

# Float bit layout: (uint_type, sign_shift, exp_shift, exp_mask, mantissa_mask)
FP_INFO = {dtypes.float64: (dtypes.uint64, 63, 52, 0x7ff, 0xfffffffffffff),
           dtypes.float32: (dtypes.uint32, 31, 23, 0xff, 0x7fffff), dtypes.float16: (dtypes.uint16, 15, 10, 0x1f, 0x3ff)}
# Type conversions: target_dtype, clamp_negative
CVT_MAP = {'u32_to_f32': (dtypes.float32, False), 'i32_to_f32': (dtypes.float32, False), 'f32_to_u32': (dtypes.uint32, True),
           'f32_to_i32': (dtypes.int32, False), 'f16_to_f32': (dtypes.float32, False), 'f32_to_f16': (dtypes.float16, False),
           'f32_to_u8': (dtypes.uint8, False), 'f32_to_i8': (dtypes.int8, False), 'f32_to_u16': (dtypes.uint16, False),
           'f32_to_i16': (dtypes.int16, False), 'v_cvt_u16_f32': (dtypes.uint16, False), 'v_cvt_i16_f32': (dtypes.int16, False),
           'f64_to_i32': (dtypes.int32, False), 'f64_to_u32': (dtypes.uint32, True), 'i32_to_f64': (dtypes.float64, False),
           'u32_to_f64': (dtypes.float64, False), 'f64_to_f32': (dtypes.float32, False), 'f32_to_f64': (dtypes.float64, False),
           'u16_to_f16': (dtypes.float16, False), 'i16_to_f16': (dtypes.float16, False), 'f16_to_u16': (dtypes.uint16, False),
           'f16_to_i16': (dtypes.int16, False)}
MATH_OPS = {'trunc': Ops.TRUNC, 'floor': Ops.TRUNC, 'sqrt': Ops.SQRT, 'exp2': Ops.EXP2, 'log2': Ops.LOG2, 'sin': Ops.SIN, 'rcp': Ops.RECIPROCAL}

def _call_MEM(v): return v
def _call_fma(a, b, c): return UOp(Ops.MULACC, c.dtype, (a, b, c))
def _call_abs(v): return UOp(Ops.WHERE, v.dtype, (UOp(Ops.CMPLT, dtypes.bool, (v, UOp.const(v.dtype, 0))), UOp(Ops.NEG, v.dtype, (v,)), v))
def _call_cos(v): return UOp(Ops.SIN, v.dtype, (UOp(Ops.ADD, v.dtype, (v, UOp.const(v.dtype, 1.5707963267948966))),))
def _call_rsqrt(v): return UOp(Ops.RECIPROCAL, v.dtype, (UOp(Ops.SQRT, v.dtype, (v,)),))
def _call_clamp(x, lo, hi):
  clamped = UOp(Ops.WHERE, x.dtype, (UOp(Ops.CMPLT, dtypes.bool, (x, lo)), lo, x))
  return UOp(Ops.WHERE, x.dtype, (UOp(Ops.CMPLT, dtypes.bool, (hi, clamped)), hi, clamped))
def _call_fract(v): return UOp(Ops.SUB, v.dtype, (v, UOp(Ops.TRUNC, v.dtype, (v,))))
def _call_isNAN(v): return UOp(Ops.CMPNE, dtypes.bool, (v, v))
def _call_isSignalNAN(v): return UOp.const(dtypes.bool, 0)
def _call_cvtToQuietNAN(v): return v
def _call_isINF(v):
  return UOp(Ops.OR, dtypes.bool, (UOp(Ops.CMPEQ, dtypes.bool, (v, UOp.const(v.dtype, float('inf')))),
                                   UOp(Ops.CMPEQ, dtypes.bool, (v, UOp.const(v.dtype, float('-inf'))))))
def _call_sign(v):
  uint_dt, sign_shift, _, _, _ = FP_INFO.get(v.dtype, FP_INFO[dtypes.float32])
  bits = UOp(Ops.BITCAST, uint_dt, (v,))
  return UOp(Ops.AND, dtypes.uint32, (_cast(UOp(Ops.SHR, uint_dt, (bits, UOp.const(uint_dt, sign_shift))), dtypes.uint32), UOp.const(dtypes.uint32, 1)))
def _call_exponent(v):
  uint_dt, _, exp_shift, exp_mask, _ = FP_INFO.get(v.dtype, FP_INFO[dtypes.float32])
  bits = UOp(Ops.BITCAST, uint_dt, (v,))
  return UOp(Ops.AND, dtypes.uint32, (_cast(UOp(Ops.SHR, uint_dt, (bits, UOp.const(uint_dt, exp_shift))), dtypes.uint32), UOp.const(dtypes.uint32, exp_mask)))
def _call_mantissa(v):
  uint_dt, _, _, _, mant_mask = FP_INFO.get(v.dtype, FP_INFO[dtypes.float32])
  bits, out_dt = UOp(Ops.BITCAST, uint_dt, (v,)), dtypes.uint64 if v.dtype == dtypes.float64 else dtypes.uint32
  return UOp(Ops.AND, out_dt, (_cast(bits, out_dt) if out_dt != uint_dt else bits, UOp.const(out_dt, mant_mask)))
def _call_isEven(v):
  int_val = UOp(Ops.CAST, dtypes.int64, (v,))
  return UOp(Ops.CMPEQ, dtypes.bool, (UOp(Ops.AND, dtypes.int64, (int_val, UOp.const(dtypes.int64, 1))), UOp.const(dtypes.int64, 0)))
def _call_signext(v): return _cast(v, dtypes.int64)
def _call_signext_from_bit(val, width):
  sign_bit = UOp(Ops.SHL, val.dtype, (UOp.const(val.dtype, 1), UOp(Ops.SUB, val.dtype, (_cast(width, val.dtype), UOp.const(val.dtype, 1)))))
  result = UOp(Ops.SUB, val.dtype, (UOp(Ops.XOR, val.dtype, (val, sign_bit)), sign_bit))
  return UOp(Ops.WHERE, val.dtype, (UOp(Ops.CMPEQ, dtypes.bool, (width, UOp.const(width.dtype, 0))), UOp.const(val.dtype, 0), result))
def _call_ABSDIFF(a, b):
  a_gt_b = UOp(Ops.CMPLT, dtypes.bool, (b, a))
  max_v = UOp(Ops.WHERE, dtypes.uint32, (a_gt_b, _cast(a, dtypes.uint32), _cast(b, dtypes.uint32)))
  min_v = UOp(Ops.WHERE, dtypes.uint32, (a_gt_b, _cast(b, dtypes.uint32), _cast(a, dtypes.uint32)))
  return UOp(Ops.SUB, dtypes.uint32, (max_v, min_v))
def _call_SAT8(v):
  clamped = UOp(Ops.WHERE, v.dtype, (UOp(Ops.CMPLT, dtypes.bool, (v, UOp.const(v.dtype, -128))), UOp.const(v.dtype, -128), v))
  return UOp(Ops.WHERE, v.dtype, (UOp(Ops.CMPLT, dtypes.bool, (UOp.const(v.dtype, 127), clamped)), UOp.const(v.dtype, 127), clamped))

CALL_DISPATCH = {
  'MEM': _call_MEM, 'fma': _call_fma, 'abs': _call_abs, 'cos': _call_cos, 'rsqrt': _call_rsqrt,
  'clamp': _call_clamp, 'fract': _call_fract, 'isNAN': _call_isNAN, 'isQuietNAN': _call_isNAN,
  'isSignalNAN': _call_isSignalNAN, 'cvtToQuietNAN': _call_cvtToQuietNAN, 'isINF': _call_isINF,
  'sign': _call_sign, 'exponent': _call_exponent, 'mantissa': _call_mantissa, 'isEven': _call_isEven,
  'signext': _call_signext, 'signext_from_bit': _call_signext_from_bit, 'ABSDIFF': _call_ABSDIFF, 'SAT8': _call_SAT8,
}

def _transform_call(name: str, a: list[UOp], hint: DType) -> UOp:
  """Transform function call to UOp."""
  if name in CALL_DISPATCH: return CALL_DISPATCH[name](*a)
  if name == 'pow':  # pow(2.0, x) -> exp2(x)
    assert a[0].op == Ops.CONST and a[0].arg == 2.0, f"pow only supports base=2, got {a[0]}"
    result_dt = a[0].dtype if a[0].dtype in FLOATS else hint or dtypes.float32
    return UOp(Ops.EXP2, result_dt, (a[1] if a[1].dtype == result_dt else UOp(Ops.CAST, result_dt, (a[1],)),))
  if name in MATH_OPS: return UOp(MATH_OPS[name], a[0].dtype, (a[0],))
  if name in ('min', 'max'):
    return UOp(Ops.WHERE, a[0].dtype, (UOp(Ops.CMPLT, dtypes.bool, ((a[0], a[1]) if name == 'min' else (a[1], a[0]))), a[0], a[1]))
  if name in CVT_MAP:
    dt, clamp_neg = CVT_MAP[name]
    v = UOp(Ops.WHERE, a[0].dtype, (UOp(Ops.CMPLT, dtypes.bool, (a[0], UOp.const(a[0].dtype, 0.0))), UOp.const(a[0].dtype, 0.0), a[0])) if clamp_neg else a[0]
    return UOp(Ops.CAST, dt, (v,))
  if name in ('f16_to_snorm', 'f16_to_unorm'):
    lo, scale, out_dt = (-1.0, 32767.0, dtypes.int16) if name == 'f16_to_snorm' else (0.0, 65535.0, dtypes.uint16)
    clamped = UOp(Ops.WHERE, a[0].dtype, (UOp(Ops.CMPLT, dtypes.bool, (a[0], UOp.const(a[0].dtype, lo))), UOp.const(a[0].dtype, lo), a[0]))
    clamped = UOp(Ops.WHERE, a[0].dtype, (UOp(Ops.CMPLT, dtypes.bool, (UOp.const(a[0].dtype, 1.0), clamped)), UOp.const(a[0].dtype, 1.0), clamped))
    return UOp(Ops.CAST, out_dt, (UOp(Ops.MUL, a[0].dtype, (clamped, UOp.const(a[0].dtype, scale))),))
  if name in ('LT_NEG_ZERO', 'GT_NEG_ZERO'):
    int_dt = {dtypes.float64: dtypes.int64, dtypes.float16: dtypes.int16}.get(a[0].dtype, dtypes.int32)
    a_bits, b_bits = UOp(Ops.BITCAST, int_dt, (a[0],)), UOp(Ops.BITCAST, int_dt, (a[1],))
    return UOp(Ops.CMPLT, dtypes.bool, ((a_bits, b_bits) if name == 'LT_NEG_ZERO' else (b_bits, a_bits)))
  if name.startswith('v_min_') or name.startswith('v_max_'):
    return UOp(Ops.WHERE, a[0].dtype, (UOp(Ops.CMPLT, dtypes.bool, ((a[0], a[1]) if 'min' in name else (a[1], a[0]))), a[0], a[1]))

  raise ValueError(f"Unknown function: {name}")

def _get_lhs_info(lhs, ctx: Ctx) -> tuple[str, DType, int|None, int|None, str|None]:
  """Extract assignment target: (var_name, dtype, hi_bit, lo_bit, idx_var)"""
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

def _stmt(stmt, ctx: Ctx):
  """Transform statement and update context state."""
  match stmt:
    case Declare(_, _): pass
    case Assign(lhs, rhs):
      var, dtype, hi, lo, idx_var = _get_lhs_info(lhs, ctx)
      out_vars = ('D0', 'D1', 'SCC', 'VCC', 'EXEC', 'PC')

      # Bit index: D0.u64[laneId] = expr
      if idx_var is not None:
        base, idx = ctx.vars.get(var), ctx.vars.get(idx_var)
        if base is None or idx is None: raise ValueError(f"Unknown variable: {var} or {idx_var}")
        cond = _expr(rhs, ctx)
        one, bit_mask = UOp.const(dtype, 1), UOp(Ops.SHL, dtype, (UOp.const(dtype, 1), _cast(idx, dtype)))
        result = UOp(Ops.OR, dtype, (UOp(Ops.AND, dtype, (base, UOp(Ops.XOR, dtype, (bit_mask, UOp.const(dtype, -1))))),
                                     UOp(Ops.SHL, dtype, (UOp(Ops.AND, dtype, (_cast(cond, dtype), one)), _cast(idx, dtype)))))
        ctx.vars[var] = result
        if var in out_vars: ctx.outputs.append((var, result, dtype))
        return

      # Bit range: D0[31:16].f16 = expr
      if hi is not None and lo is not None:
        if hi < lo: hi, lo = lo, hi
        base = ctx.vars.get(var, UOp.const(dtypes.uint32, 0))
        rhs_uop = _expr(rhs, ctx, dtype)
        rhs_bits = UOp(Ops.BITCAST, dtypes.uint16 if dtype == dtypes.float16 else (dtypes.uint32 if dtype == dtypes.float32 else dtypes.uint64), (rhs_uop,)) if dtype in FLOATS else _cast(rhs_uop, dtypes.uint32)
        if dtype == dtypes.float16: rhs_bits = _cast(rhs_bits, dtypes.uint32)
        mask = (1 << (hi - lo + 1)) - 1
        shifted = UOp(Ops.SHL, dtypes.uint32, (UOp(Ops.AND, dtypes.uint32, (rhs_bits, UOp.const(dtypes.uint32, mask))), UOp.const(dtypes.uint32, lo)))
        result = UOp(Ops.OR, dtypes.uint32, (UOp(Ops.AND, dtypes.uint32, (_cast(base, dtypes.uint32), UOp.const(dtypes.uint32, ~(mask << lo) & 0xffffffff))), shifted))
        ctx.vars[var] = result
        if var in out_vars:
          ctx.outputs = [(n, u, d) for n, u, d in ctx.outputs if n != var]
          ctx.outputs.append((var, result, dtypes.uint32))
        return

      # Simple assignment
      rhs_uop = _expr(rhs, ctx, dtype)
      ctx.vars[var] = rhs_uop
      if dtype.itemsize == 8 and var in ('D0', 'D1', 'S0', 'S1'): ctx.vars[var + '_64'] = rhs_uop
      if var in out_vars: ctx.outputs.append((var, rhs_uop, dtype))

    case If(branches): _transform_if(branches, ctx)
    case For(var, start, end, body): _transform_for(var, start, end, body, ctx)

def _transform_if(branches: tuple, ctx: Ctx):
  """Transform if/elsif/else to nested WHERE expressions."""
  parsed = [(_expr(c, ctx) if c else None, body) for c, body in branches]
  assigned = {_get_lhs_info(s.lhs, ctx)[0] for _, body in parsed for s in body if isinstance(s, Assign)}

  for var in assigned:
    dtype = next((_get_lhs_info(s.lhs, ctx)[1] for _, body in parsed for s in body if isinstance(s, Assign) and _get_lhs_info(s.lhs, ctx)[0] == var), dtypes.uint32)
    result = ctx.vars.get(var, UOp.const(dtype, 0))
    for cond_uop, body in reversed(parsed):
      branch_val = next((_expr(s.rhs, ctx, _get_lhs_info(s.lhs, ctx)[1]) for s in body if isinstance(s, Assign) and _get_lhs_info(s.lhs, ctx)[0] == var), None)
      if branch_val is not None:
        result = branch_val if cond_uop is None else UOp(Ops.WHERE, branch_val.dtype, (cond_uop, branch_val, _cast(result, branch_val.dtype)))

    if dtype in FLOATS:
      result_bits = UOp(Ops.BITCAST, dtypes.uint32 if dtype == dtypes.float32 else dtypes.uint64, (result,))
      ctx.vars[var] = result_bits
      if var in ('D0', 'D1', 'SCC', 'VCC', 'EXEC', 'PC'):
        ctx.outputs = [(n, u, d) for n, u, d in ctx.outputs if n != var]
        ctx.outputs.append((var, result_bits, dtypes.uint32 if dtype == dtypes.float32 else dtypes.uint64))
    else:
      ctx.vars[var] = result
      if var in ('D0', 'D1', 'SCC', 'VCC', 'EXEC', 'PC'):
        ctx.outputs = [(n, u, d) for n, u, d in ctx.outputs if n != var]
        ctx.outputs.append((var, result, dtype))

def _transform_for(var: str, start, end, body: tuple, ctx: Ctx):
  """Unroll for loop and transform body."""
  start_val = start.value if isinstance(start, Const) else int(_expr(start, ctx).arg)
  end_val = end.value if isinstance(end, Const) else int(_expr(end, ctx).arg)
  for i in range(int(start_val), int(end_val) + 1):
    ctx.vars[var] = UOp.const(dtypes.uint32, i)
    for s in body:
      if isinstance(s, If): _transform_if(s.branches, ctx)
      elif isinstance(s, Assign): _stmt(s, ctx)

def _float_to_bits(val: float, dtype: DType) -> int:
  if dtype == dtypes.float32: return struct.unpack('<I', struct.pack('<f', val))[0]
  if dtype == dtypes.float16:
    if math.isnan(val): return 0x7e00
    if math.isinf(val): return 0x7c00 if val > 0 else 0xfc00
    if abs(val) > 65504.0: return 0x7c00 if val > 0 else 0xfc00
    if abs(val) < 6.103515625e-05 and val != 0: return 0x0000 if val > 0 else 0x8000
    return struct.unpack('<H', struct.pack('<e', val))[0]
  if dtype == dtypes.float64: return struct.unpack('<Q', struct.pack('<d', val))[0]
  return int(val)

def _compile_pseudocode(pseudocode: str) -> tuple[UOp, list[tuple[str, DType]], dict[str, UOp]]:
  """Compile pseudocode to UOp graph."""
  ctx = Ctx()
  for stmt in parse(pseudocode): _stmt(stmt, ctx)
  return UOp(Ops.SINK, dtypes.void, tuple(u for _, u, _ in ctx.outputs) or ()), [(n, d) for n, _, d in ctx.outputs], INPUT_VARS

def _make_uop_fn(sink: UOp, output_info: list[tuple[str, DType]], input_vars: dict[str, UOp]):
  """Create runtime function that evaluates UOp graph via simplify."""
  def fn(s0, s1, s2, d0, scc, vcc, laneId, exec_mask, literal, VGPR, src0_idx=0, vdst_idx=0, pc=None):
    simm16 = (literal if -32768 <= literal <= 32767 else (literal - 65536 if literal < 65536 else 0)) if literal is not None else 0
    dvars = {
      input_vars['S0']: UOp.const(dtypes.uint32, s0 & 0xffffffff), input_vars['S1']: UOp.const(dtypes.uint32, s1 & 0xffffffff),
      input_vars['S2']: UOp.const(dtypes.uint32, s2 & 0xffffffff), input_vars['D0']: UOp.const(dtypes.uint32, d0 & 0xffffffff),
      input_vars['S0_64']: UOp.const(dtypes.uint64, s0), input_vars['S1_64']: UOp.const(dtypes.uint64, s1),
      input_vars['S2_64']: UOp.const(dtypes.uint64, s2), input_vars['D0_64']: UOp.const(dtypes.uint64, d0),
      input_vars['SCC']: UOp.const(dtypes.uint32, scc), input_vars['VCC']: UOp.const(dtypes.uint64, vcc),
      input_vars['EXEC']: UOp.const(dtypes.uint64, exec_mask), input_vars['laneId']: UOp.const(dtypes.uint32, laneId),
      input_vars['SIMM16']: UOp.const(dtypes.int32, simm16), input_vars['SIMM32']: UOp.const(dtypes.uint32, literal or 0),
      input_vars['PC']: UOp.const(dtypes.uint64, pc or 0),
    }
    simplified = sink.substitute(dvars).simplify()
    assert simplified.op == Ops.SINK, f"expected SINK, got {simplified.op}"
    result = {}
    for i, (name, dtype) in enumerate(output_info):
      out = simplified.src[i]
      assert out.op == Ops.CONST, f"simplify did not produce CONST for {name}, got {out.op}"
      result[name] = _float_to_bits(out.arg, dtype) if dtype in FLOATS else int(out.arg) & (0xffffffff if dtype.itemsize <= 4 else 0xffffffffffffffff)
    return result
  return fn

SUPPORTED_OPS: set[str] = {
  'V_ADD3_U32', 'V_ADD_CO_CI_U32', 'V_ADD_CO_U32', 'V_ADD_F16', 'V_ADD_F32', 'V_ADD_F64', 'V_ADD_LSHL_U32', 'V_ADD_NC_I16', 'V_ADD_NC_I32', 'V_ADD_NC_U16', 'V_ADD_NC_U32',
  'V_ALIGNBIT_B32', 'V_ALIGNBYTE_B32', 'V_AND_B16', 'V_AND_B32', 'V_AND_OR_B32', 'V_ASHRREV_I16', 'V_ASHRREV_I32', 'V_ASHRREV_I64',
  'V_BFE_I32', 'V_BFE_U32', 'V_BFI_B32', 'V_BFM_B32', 'V_CNDMASK_B16', 'V_CNDMASK_B32', 'V_COS_F16', 'V_COS_F32', 'V_CUBEID_F32', 'V_CUBESC_F32',
  'V_CVT_F16_F32', 'V_CVT_F32_F16', 'V_CVT_F32_I32', 'V_CVT_F32_U32', 'V_CVT_F32_UBYTE0', 'V_CVT_F32_UBYTE1', 'V_CVT_F32_UBYTE2', 'V_CVT_F32_UBYTE3',
  'V_CVT_FLOOR_I32_F32', 'V_CVT_I32_F32', 'V_CVT_I32_I16', 'V_CVT_NEAREST_I32_F32', 'V_CVT_PK_I16_F32', 'V_CVT_PK_U16_F32', 'V_CVT_PK_U8_F32', 'V_CVT_U32_F32', 'V_CVT_U32_U16',
  'V_DOT2_F16_F16', 'V_DOT2_F32_F16', 'V_DOT2ACC_F32_F16', 'V_FMA_DX9_ZERO_F32', 'V_FMA_F16', 'V_FMA_F32', 'V_FMA_F64', 'V_FMAAK_F16', 'V_FMAAK_F32',
  'V_FMAC_DX9_ZERO_F32', 'V_FMAC_F16', 'V_FMAC_F32', 'V_FMAMK_F16', 'V_FMAMK_F32', 'V_FREXP_EXP_I16_F16', 'V_FREXP_EXP_I32_F32', 'V_FREXP_EXP_I32_F64',
  'V_LERP_U8', 'V_LOG_F16', 'V_LOG_F32', 'V_LSHL_ADD_U32', 'V_LSHL_OR_B32', 'V_LSHLREV_B16', 'V_LSHLREV_B32', 'V_LSHLREV_B64', 'V_LSHRREV_B16', 'V_LSHRREV_B32', 'V_LSHRREV_B64',
  'V_MAD_I16', 'V_MAD_I32_I16', 'V_MAD_I32_I24', 'V_MAD_U16', 'V_MAD_U32_U16', 'V_MAD_U32_U24', 'V_MAX_I16', 'V_MAX_I32', 'V_MAX_U16', 'V_MAX_U32',
  'V_MIN_I16', 'V_MIN_I32', 'V_MIN_U16', 'V_MIN_U32', 'V_MOV_B16', 'V_MOV_B32', 'V_MSAD_U8', 'V_MUL_DX9_ZERO_F32', 'V_MUL_F16', 'V_MUL_F32', 'V_MUL_F64',
  'V_MUL_HI_I32', 'V_MUL_HI_I32_I24', 'V_MUL_HI_U32', 'V_MUL_HI_U32_U24', 'V_MUL_I32_I24', 'V_MUL_LO_U16', 'V_MUL_LO_U32', 'V_MUL_U32_U24',
  'V_NOT_B16', 'V_NOT_B32', 'V_OR3_B32', 'V_OR_B16', 'V_OR_B32', 'V_PACK_B32_F16', 'V_PK_FMAC_F16', 'V_RCP_F16', 'V_RCP_F32', 'V_RCP_F64', 'V_RCP_IFLAG_F32',
  'V_RSQ_F16', 'V_RSQ_F32', 'V_RSQ_F64', 'V_PK_ADD_F16', 'V_PK_ADD_I16', 'V_PK_ADD_U16', 'V_PK_ASHRREV_I16', 'V_PK_FMA_F16', 'V_PK_LSHLREV_B16', 'V_PK_LSHRREV_B16',
  'V_PK_MAD_I16', 'V_PK_MAD_U16', 'V_PK_MAX_I16', 'V_PK_MAX_U16', 'V_PK_MIN_I16', 'V_PK_MIN_U16', 'V_PK_MUL_F16', 'V_PK_MUL_LO_U16', 'V_PK_SUB_I16', 'V_PK_SUB_U16',
  'V_RNDNE_F16', 'V_RNDNE_F32', 'V_RNDNE_F64', 'V_SAD_U8', 'V_SAD_U16', 'V_SAD_U32', 'V_SIN_F16', 'V_SIN_F32', 'V_SQRT_F16', 'V_SQRT_F32', 'V_SQRT_F64',
  'V_CVT_F32_F64', 'V_CVT_F64_F32', 'V_CVT_F64_I32', 'V_CVT_F64_U32', 'V_CVT_I32_F64', 'V_CVT_U32_F64', 'V_CVT_NORM_I16_F16', 'V_CVT_NORM_U16_F16',
  'V_CVT_PK_NORM_I16_F16', 'V_CVT_PK_NORM_U16_F16', 'V_CVT_PK_RTZ_F16_F32', 'V_SUB_CO_CI_U32', 'V_SUB_CO_U32', 'V_SUB_F16', 'V_SUB_F32',
  'V_SUB_NC_I16', 'V_SUB_NC_I32', 'V_SUB_NC_U16', 'V_SUB_NC_U32', 'V_SUBREV_CO_CI_U32', 'V_SUBREV_CO_U32', 'V_SUBREV_F16', 'V_SUBREV_F32', 'V_SUBREV_NC_U32',
  'V_SWAP_B16', 'V_SWAP_B32', 'V_TRUNC_F16', 'V_TRUNC_F32', 'V_TRUNC_F64', 'V_WRITELANE_B32', 'V_XAD_U32', 'V_XNOR_B32', 'V_XOR3_B32', 'V_XOR_B16', 'V_XOR_B32',
  'V_CVT_F16_I16', 'V_CVT_F16_U16', 'V_CVT_I16_F16', 'V_CVT_U16_F16', 'V_EXP_F16', 'V_EXP_F32', 'V_LDEXP_F16', 'V_LDEXP_F32', 'V_LDEXP_F64',
  'V_CUBEMA_F32', 'V_CUBETC_F32', 'V_SAT_PK_U8_I16', 'V_MAX3_I16', 'V_MAX3_I32', 'V_MAX3_U16', 'V_MAX3_U32', 'V_MIN3_I16', 'V_MIN3_I32', 'V_MIN3_U16', 'V_MIN3_U32',
  'V_MAXMIN_I32', 'V_MAXMIN_U32', 'V_MINMAX_I32', 'V_MINMAX_U32',
  'V_CMP_EQ_F16', 'V_CMP_EQ_F32', 'V_CMP_EQ_F64', 'V_CMP_EQ_I16', 'V_CMP_EQ_I32', 'V_CMP_EQ_I64', 'V_CMP_EQ_U16', 'V_CMP_EQ_U32', 'V_CMP_EQ_U64',
  'V_CMP_F_F16', 'V_CMP_F_F32', 'V_CMP_F_F64', 'V_CMP_F_I32', 'V_CMP_F_I64', 'V_CMP_F_U32', 'V_CMP_F_U64',
  'V_CMP_GE_F16', 'V_CMP_GE_F32', 'V_CMP_GE_F64', 'V_CMP_GE_I16', 'V_CMP_GE_I32', 'V_CMP_GE_I64', 'V_CMP_GE_U16', 'V_CMP_GE_U32', 'V_CMP_GE_U64',
  'V_CMP_GT_F16', 'V_CMP_GT_F32', 'V_CMP_GT_F64', 'V_CMP_GT_I16', 'V_CMP_GT_I32', 'V_CMP_GT_I64', 'V_CMP_GT_U16', 'V_CMP_GT_U32', 'V_CMP_GT_U64',
  'V_CMP_LE_F16', 'V_CMP_LE_F32', 'V_CMP_LE_F64', 'V_CMP_LE_I16', 'V_CMP_LE_I32', 'V_CMP_LE_I64', 'V_CMP_LE_U16', 'V_CMP_LE_U32', 'V_CMP_LE_U64',
  'V_CMP_LG_F16', 'V_CMP_LG_F32', 'V_CMP_LG_F64', 'V_CMP_LT_F16', 'V_CMP_LT_F32', 'V_CMP_LT_F64', 'V_CMP_LT_I16', 'V_CMP_LT_I32', 'V_CMP_LT_I64',
  'V_CMP_LT_U16', 'V_CMP_LT_U32', 'V_CMP_LT_U64', 'V_CMP_NE_I16', 'V_CMP_NE_I32', 'V_CMP_NE_I64', 'V_CMP_NE_U16', 'V_CMP_NE_U32', 'V_CMP_NE_U64',
  'V_CMP_NEQ_F16', 'V_CMP_NEQ_F32', 'V_CMP_NEQ_F64', 'V_CMP_NGE_F16', 'V_CMP_NGE_F32', 'V_CMP_NGE_F64', 'V_CMP_NGT_F16', 'V_CMP_NGT_F32', 'V_CMP_NGT_F64',
  'V_CMP_NLE_F16', 'V_CMP_NLE_F32', 'V_CMP_NLE_F64', 'V_CMP_NLG_F16', 'V_CMP_NLG_F32', 'V_CMP_NLG_F64', 'V_CMP_NLT_F16', 'V_CMP_NLT_F32', 'V_CMP_NLT_F64',
  'V_CMP_O_F16', 'V_CMP_O_F32', 'V_CMP_O_F64', 'V_CMP_T_F16', 'V_CMP_T_F32', 'V_CMP_T_F64', 'V_CMP_T_I32', 'V_CMP_T_I64', 'V_CMP_T_U32', 'V_CMP_T_U64',
  'V_CMP_U_F16', 'V_CMP_U_F32', 'V_CMP_U_F64',
  'V_CMPX_EQ_F16', 'V_CMPX_EQ_F32', 'V_CMPX_EQ_F64', 'V_CMPX_EQ_I16', 'V_CMPX_EQ_I32', 'V_CMPX_EQ_I64', 'V_CMPX_EQ_U16', 'V_CMPX_EQ_U32', 'V_CMPX_EQ_U64',
  'V_CMPX_F_F16', 'V_CMPX_F_F32', 'V_CMPX_F_F64', 'V_CMPX_F_I32', 'V_CMPX_F_I64', 'V_CMPX_F_U32', 'V_CMPX_F_U64',
  'V_CMPX_GE_F16', 'V_CMPX_GE_F32', 'V_CMPX_GE_F64', 'V_CMPX_GE_I16', 'V_CMPX_GE_I32', 'V_CMPX_GE_I64', 'V_CMPX_GE_U16', 'V_CMPX_GE_U32', 'V_CMPX_GE_U64',
  'V_CMPX_GT_F16', 'V_CMPX_GT_F32', 'V_CMPX_GT_F64', 'V_CMPX_GT_I16', 'V_CMPX_GT_I32', 'V_CMPX_GT_I64', 'V_CMPX_GT_U16', 'V_CMPX_GT_U32', 'V_CMPX_GT_U64',
  'V_CMPX_LE_F16', 'V_CMPX_LE_F32', 'V_CMPX_LE_F64', 'V_CMPX_LE_I16', 'V_CMPX_LE_I32', 'V_CMPX_LE_I64', 'V_CMPX_LE_U16', 'V_CMPX_LE_U32', 'V_CMPX_LE_U64',
  'V_CMPX_LG_F16', 'V_CMPX_LG_F32', 'V_CMPX_LG_F64', 'V_CMPX_LT_F16', 'V_CMPX_LT_F32', 'V_CMPX_LT_F64', 'V_CMPX_LT_I16', 'V_CMPX_LT_I32', 'V_CMPX_LT_I64',
  'V_CMPX_LT_U16', 'V_CMPX_LT_U32', 'V_CMPX_LT_U64', 'V_CMPX_NE_I16', 'V_CMPX_NE_I32', 'V_CMPX_NE_I64', 'V_CMPX_NE_U16', 'V_CMPX_NE_U32', 'V_CMPX_NE_U64',
  'V_CMPX_NEQ_F16', 'V_CMPX_NEQ_F32', 'V_CMPX_NEQ_F64', 'V_CMPX_NGE_F16', 'V_CMPX_NGE_F32', 'V_CMPX_NGE_F64', 'V_CMPX_NGT_F16', 'V_CMPX_NGT_F32', 'V_CMPX_NGT_F64',
  'V_CMPX_NLE_F16', 'V_CMPX_NLE_F32', 'V_CMPX_NLE_F64', 'V_CMPX_NLG_F16', 'V_CMPX_NLG_F32', 'V_CMPX_NLG_F64', 'V_CMPX_NLT_F16', 'V_CMPX_NLT_F32', 'V_CMPX_NLT_F64',
  'V_CMPX_O_F16', 'V_CMPX_O_F32', 'V_CMPX_O_F64', 'V_CMPX_T_F16', 'V_CMPX_T_F32', 'V_CMPX_T_F64', 'V_CMPX_T_I32', 'V_CMPX_T_I64', 'V_CMPX_T_U32', 'V_CMPX_T_U64',
  'V_CMPX_U_F16', 'V_CMPX_U_F32', 'V_CMPX_U_F64',
  'S_ABSDIFF_I32', 'S_ABS_I32', 'S_ADD_F16', 'S_ADD_F32', 'S_ADD_I32', 'S_ADD_U32', 'S_ADDC_U32', 'S_ADDK_I32', 'S_AND_B32', 'S_AND_B64',
  'S_AND_NOT0_SAVEEXEC_B32', 'S_AND_NOT0_SAVEEXEC_B64', 'S_AND_NOT0_WREXEC_B32', 'S_AND_NOT0_WREXEC_B64', 'S_AND_NOT1_B32', 'S_AND_NOT1_B64',
  'S_AND_NOT1_SAVEEXEC_B32', 'S_AND_NOT1_SAVEEXEC_B64', 'S_AND_NOT1_WREXEC_B32', 'S_AND_NOT1_WREXEC_B64', 'S_AND_SAVEEXEC_B32', 'S_AND_SAVEEXEC_B64',
  'S_ASHR_I32', 'S_ASHR_I64', 'S_BCNT0_I32_B32', 'S_BCNT0_I32_B64', 'S_BCNT1_I32_B32', 'S_BCNT1_I32_B64', 'S_BFE_I32', 'S_BFE_I64', 'S_BFE_U32', 'S_BFE_U64',
  'S_BFM_B32', 'S_BFM_B64', 'S_BITSET0_B32', 'S_BITSET0_B64', 'S_BITSET1_B32', 'S_BITSET1_B64', 'S_CMOVK_I32', 'S_CMOV_B32', 'S_CMOV_B64',
  'S_CSELECT_B32', 'S_CSELECT_B64', 'S_CVT_F16_F32', 'S_CVT_F32_F16', 'S_CVT_F32_I32', 'S_CVT_F32_U32', 'S_CVT_HI_F32_F16', 'S_CVT_I32_F32',
  'S_CVT_PK_RTZ_F16_F32', 'S_CVT_U32_F32', 'S_DELAY_ALU', 'S_FMAAK_F32', 'S_FMAC_F16', 'S_FMAC_F32', 'S_FMAMK_F32', 'S_LSHL_B32', 'S_LSHL_B64',
  'S_LSHL1_ADD_U32', 'S_LSHL2_ADD_U32', 'S_LSHL3_ADD_U32', 'S_LSHL4_ADD_U32', 'S_LSHR_B32', 'S_LSHR_B64', 'S_MAX_I32', 'S_MAX_U32', 'S_MIN_I32', 'S_MIN_U32',
  'S_MOVK_I32', 'S_MOV_B32', 'S_MOV_B64', 'S_MULK_I32', 'S_MUL_F16', 'S_MUL_F32', 'S_MUL_HI_I32', 'S_MUL_HI_U32', 'S_MUL_I32', 'S_NAND_B32', 'S_NAND_B64',
  'S_NAND_SAVEEXEC_B32', 'S_NAND_SAVEEXEC_B64', 'S_NOP', 'S_NOR_B32', 'S_NOR_B64', 'S_NOR_SAVEEXEC_B32', 'S_NOR_SAVEEXEC_B64', 'S_NOT_B32', 'S_NOT_B64',
  'S_OR_B32', 'S_OR_B64', 'S_OR_NOT0_SAVEEXEC_B32', 'S_OR_NOT0_SAVEEXEC_B64', 'S_OR_NOT1_B32', 'S_OR_NOT1_B64', 'S_OR_NOT1_SAVEEXEC_B32', 'S_OR_NOT1_SAVEEXEC_B64',
  'S_OR_SAVEEXEC_B32', 'S_OR_SAVEEXEC_B64', 'S_PACK_HH_B32_B16', 'S_PACK_HL_B32_B16', 'S_PACK_LH_B32_B16', 'S_PACK_LL_B32_B16', 'S_RFE_B64',
  'S_RNDNE_F16', 'S_RNDNE_F32', 'S_SENDMSG_RTN_B32', 'S_SENDMSG_RTN_B64', 'S_SETPC_B64', 'S_SEXT_I32_I16', 'S_SEXT_I32_I8',
  'S_SUB_F16', 'S_SUB_F32', 'S_SUB_I32', 'S_SUB_U32', 'S_SUBB_U32', 'S_TRUNC_F16', 'S_TRUNC_F32', 'S_VERSION', 'S_BITCMP0_B32', 'S_BITCMP0_B64',
  'S_BITCMP1_B32', 'S_BITCMP1_B64', 'S_MAX_F16', 'S_MAX_F32', 'S_MIN_F16', 'S_MIN_F32', 'S_WAITCNT_EXPCNT', 'S_WAITCNT_LGKMCNT', 'S_WAITCNT_VMCNT',
  'S_WAITCNT_VSCNT', 'S_BRANCH', 'S_CALL_B64', 'S_CBRANCH_EXECNZ', 'S_CBRANCH_EXECZ', 'S_CBRANCH_SCC0', 'S_CBRANCH_SCC1', 'S_CBRANCH_VCCNZ', 'S_CBRANCH_VCCZ',
  'S_GETPC_B64', 'S_XNOR_B32', 'S_XNOR_B64', 'S_XNOR_SAVEEXEC_B32', 'S_XNOR_SAVEEXEC_B64', 'S_XOR_B32', 'S_XOR_B64', 'S_XOR_SAVEEXEC_B32', 'S_XOR_SAVEEXEC_B64',
  'S_CMPK_EQ_I32', 'S_CMPK_EQ_U32', 'S_CMPK_GE_I32', 'S_CMPK_GE_U32', 'S_CMPK_GT_I32', 'S_CMPK_GT_U32', 'S_CMPK_LE_I32', 'S_CMPK_LE_U32',
  'S_CMPK_LG_I32', 'S_CMPK_LG_U32', 'S_CMPK_LT_I32', 'S_CMPK_LT_U32', 'S_CMP_EQ_F16', 'S_CMP_EQ_F32', 'S_CMP_EQ_I32', 'S_CMP_EQ_U32', 'S_CMP_EQ_U64',
  'S_CMP_GE_F16', 'S_CMP_GE_F32', 'S_CMP_GE_I32', 'S_CMP_GE_U32', 'S_CMP_GT_F16', 'S_CMP_GT_F32', 'S_CMP_GT_I32', 'S_CMP_GT_U32',
  'S_CMP_LE_F16', 'S_CMP_LE_F32', 'S_CMP_LE_I32', 'S_CMP_LE_U32', 'S_CMP_LG_F16', 'S_CMP_LG_F32', 'S_CMP_LG_I32', 'S_CMP_LG_U32', 'S_CMP_LG_U64',
  'S_CMP_LT_F16', 'S_CMP_LT_F32', 'S_CMP_LT_I32', 'S_CMP_LT_U32', 'S_CMP_NEQ_F16', 'S_CMP_NEQ_F32', 'S_CMP_NGE_F16', 'S_CMP_NGE_F32',
  'S_CMP_NGT_F16', 'S_CMP_NGT_F32', 'S_CMP_NLE_F16', 'S_CMP_NLE_F32', 'S_CMP_NLG_F16', 'S_CMP_NLG_F32', 'S_CMP_NLT_F16', 'S_CMP_NLT_F32',
  'S_CMP_O_F16', 'S_CMP_O_F32', 'S_CMP_U_F16', 'S_CMP_U_F32',
}

@functools.cache
def compile_uop(cls_name: str, op_name: str, pseudocode: str):
  """Compile pseudocode to UOp-based function. Returns None if unsupported."""
  if op_name not in SUPPORTED_OPS: return None
  sink, output_info, input_vars = _compile_pseudocode(pseudocode)
  return _make_uop_fn(sink, output_info, input_vars)
