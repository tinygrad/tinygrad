# UOp-based pseudocode compiler for AMD GPU instruction emulation
import functools, struct, math
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes, DType, AddrSpace
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
  # Memory-related variables (for SMEM/FLAT/GLOBAL ops)
  'ADDR': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('ADDR', 0, 0xffffffffffffffff)),
  'SDATA': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('SDATA', 0, 0xffffffffffffffff)),
  'VDATA': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('VDATA', 0, 0xffffffffffffffff)),
  'VDST': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('VDST', 0, 0xffffffffffffffff)),
  'RETURN_DATA': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('RETURN_DATA', 0, 0xffffffffffffffff)),
  # DS (LDS) op variables - DATA/DATA2 are the source data registers, OFFSET is address offset
  'DATA': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('DATA', 0, 0xffffffffffffffff)),
  'DATA2': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('DATA2', 0, 0xffffffffffffffff)),
  'OFFSET': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('OFFSET', 0, 0xffff)),
  'OFFSET0': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('OFFSET0', 0, 0xff)),
  'OFFSET1': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('OFFSET1', 0, 0xff)),
}

# Global memory buffer for MEM[] accesses - DEFINE_GLOBAL with byte pointer
MEM_BUF = UOp(Ops.DEFINE_GLOBAL, dtypes.uint8.ptr(addrspace=AddrSpace.GLOBAL), arg=0)
# LDS (local) memory buffer for DS ops - DEFINE_LOCAL with byte pointer
LDS_BUF = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(addrspace=AddrSpace.LOCAL), arg=0)

class Ctx:
  """Compilation context - tracks variables and outputs."""
  def __init__(self, mem_buf: UOp = MEM_BUF):
    self.vars: dict[str, UOp] = dict(INPUT_VARS)
    self.outputs: list[tuple[str, UOp, DType]] = []
    self.mem_stores: list[UOp] = []  # STORE UOps for MEM/LDS
    self.mem_buf = mem_buf  # MEM_BUF for global, LDS_BUF for local

def _expr(node, ctx: Ctx, hint: DType = None) -> UOp:
  """Transform qcode AST expression to UOp."""
  match node:
    case Const(val, qdt):
      dt = _qdt(qdt) if qdt != QDType.I32 or hint is None else hint
      if isinstance(val, float) and dt not in FLOATS: dt = dtypes.float32
      return UOp.const(dt, val)

    case Var(name):
      if name == 'PI': return UOp.const(hint or dtypes.float64, math.pi)
      if 'INF' in name and name.replace('+', '').replace('-', '').replace('.f16', '').replace('.f32', '').replace('.f64', '') == 'INF':
        dt = dtypes.float16 if '.f16' in name else dtypes.float32 if '.f32' in name else hint or dtypes.float64
        return UOp.const(dt, float('-inf') if name.startswith('-') else float('inf'))
      if name in ('WAVE_MODE.IEEE', 'WAVE32'): return UOp.const(dtypes.uint32, 1)
      if name in ('WAVE64', 'ROUND_MODE', 'WAVE_STATUS.COND_DBG_SYS', 'WAVE_STATUS.COND_DBG_USER'): return UOp.const(dtypes.uint32, 0)
      if name == 'MAX_FLOAT_F32': return UOp.const(dtypes.float32, 3.402823466e+38)
      if name == 'OVERFLOW_F32': return UOp.const(dtypes.float32, float('inf'))
      if name == 'OVERFLOW_F64': return UOp.const(dtypes.float64, float('inf'))
      if name == 'UNDERFLOW_F32': return UOp.const(dtypes.float32, 0.0)
      if name == 'UNDERFLOW_F64': return UOp.const(dtypes.float64, 0.0)
      if name == 'DENORM.f32': return UOp.const(dtypes.float32, 1.17549435e-38)
      if name == 'DENORM.f64': return UOp.const(dtypes.float64, 2.2250738585072014e-308)
      if name == 'NAN.f32': return UOp.const(dtypes.float32, float('nan'))
      if name in ('VCCZ', 'EXECZ'):
        return _cast(UOp(Ops.CMPEQ, dtypes.bool, (ctx.vars.get('VCC' if name == 'VCCZ' else 'EXEC'), UOp.const(dtypes.uint64, 0))), dtypes.uint32)
      if name.startswith('eval '): return ctx.vars.get('_eval', UOp.const(dtypes.uint32, 0))
      if name not in ctx.vars: raise ValueError(f"Unknown variable: {name}")
      return _cast(ctx.vars[name], hint or ctx.vars[name].dtype)

    case Typed(expr, qdt):
      dt = _qdt(qdt)
      # Handle MEM[addr].type -> memory load using INDEX + LOAD (buffer set by caller)
      if isinstance(expr, Call) and expr.name == 'MEM':
        addr_uop = _expr(expr.args[0], ctx, dtypes.uint64)
        buf = ctx.mem_buf
        idx = UOp(Ops.INDEX, dt.ptr(0, buf.dtype.addrspace), (buf, addr_uop))
        return UOp(Ops.LOAD, dt, (idx,))
      if isinstance(expr, Var):
        if expr.name in ('VCCZ', 'EXECZ'):
          return _cast(UOp(Ops.CMPEQ, dtypes.bool, (ctx.vars.get('VCC' if expr.name == 'VCCZ' else 'EXEC'), UOp.const(dtypes.uint64, 0))), dtypes.uint32)
        if expr.name.startswith('WAVE_STATUS.COND_DBG'): return UOp.const(dtypes.uint32, 0)
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
  if name in ('f16_to_snorm', 'f16_to_unorm', 'f32_to_snorm', 'f32_to_unorm'):
    lo, scale, out_dt = (-1.0, 32767.0, dtypes.int16) if 'snorm' in name else (0.0, 65535.0, dtypes.uint16)
    clamped = UOp(Ops.WHERE, a[0].dtype, (UOp(Ops.CMPLT, dtypes.bool, (a[0], UOp.const(a[0].dtype, lo))), UOp.const(a[0].dtype, lo), a[0]))
    clamped = UOp(Ops.WHERE, a[0].dtype, (UOp(Ops.CMPLT, dtypes.bool, (UOp.const(a[0].dtype, 1.0), clamped)), UOp.const(a[0].dtype, 1.0), clamped))
    return UOp(Ops.CAST, out_dt, (UOp(Ops.MUL, a[0].dtype, (clamped, UOp.const(a[0].dtype, scale))),))
  if name == 'u32_to_u16': return UOp(Ops.AND, dtypes.uint32, (a[0], UOp.const(dtypes.uint32, 0xffff)))
  if name == 'i32_to_i16': return _cast(UOp(Ops.AND, dtypes.uint32, (_cast(a[0], dtypes.uint32), UOp.const(dtypes.uint32, 0xffff))), dtypes.int16)

  if name in ('LT_NEG_ZERO', 'GT_NEG_ZERO'):
    int_dt = {dtypes.float64: dtypes.int64, dtypes.float16: dtypes.int16}.get(a[0].dtype, dtypes.int32)
    a_bits, b_bits = UOp(Ops.BITCAST, int_dt, (a[0],)), UOp(Ops.BITCAST, int_dt, (a[1],))
    return UOp(Ops.CMPLT, dtypes.bool, ((a_bits, b_bits) if name == 'LT_NEG_ZERO' else (b_bits, a_bits)))
  if name.startswith('v_min_') or name.startswith('v_max_'):
    return UOp(Ops.WHERE, a[0].dtype, (UOp(Ops.CMPLT, dtypes.bool, ((a[0], a[1]) if 'min' in name else (a[1], a[0]))), a[0], a[1]))
  if name.startswith('v_max3_') or name.startswith('v_min3_'):
    cmp = lambda x, y: UOp(Ops.CMPLT, dtypes.bool, ((x, y) if 'min' in name else (y, x)))
    m01 = UOp(Ops.WHERE, a[0].dtype, (cmp(a[0], a[1]), a[0], a[1]))
    return UOp(Ops.WHERE, a[0].dtype, (cmp(m01, a[2]), m01, a[2]))
  if name in ('v_sad_u8', 'v_msad_u8'):  # sum of absolute differences
    result = a[2] if len(a) > 2 else UOp.const(dtypes.uint32, 0)
    for i in range(4):
      byte_a = UOp(Ops.AND, dtypes.uint32, (UOp(Ops.SHR, dtypes.uint32, (a[0], UOp.const(dtypes.uint32, i*8))), UOp.const(dtypes.uint32, 0xff)))
      byte_b = UOp(Ops.AND, dtypes.uint32, (UOp(Ops.SHR, dtypes.uint32, (a[1], UOp.const(dtypes.uint32, i*8))), UOp.const(dtypes.uint32, 0xff)))
      diff = UOp(Ops.SUB, dtypes.uint32, (byte_a, byte_b))
      abs_diff = UOp(Ops.WHERE, dtypes.uint32, (UOp(Ops.CMPLT, dtypes.bool, (diff, UOp.const(dtypes.uint32, 0x80000000))), diff,
                                                 UOp(Ops.SUB, dtypes.uint32, (UOp.const(dtypes.uint32, 0), diff))))
      result = UOp(Ops.ADD, dtypes.uint32, (result, abs_diff))
    return result
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
      # Handle MEM[addr].type = value -> memory store using INDEX + STORE (buffer set by caller)
      if isinstance(lhs, Typed) and isinstance(lhs.expr, Call) and lhs.expr.name == 'MEM':
        dt = _qdt(lhs.dtype)
        addr_uop = _expr(lhs.expr.args[0], ctx, dtypes.uint64)
        val_uop = _expr(rhs, ctx, dt)
        buf = ctx.mem_buf
        idx = UOp(Ops.INDEX, dt.ptr(0, buf.dtype.addrspace), (buf, addr_uop))
        ctx.mem_stores.append(UOp(Ops.STORE, dtypes.void, (idx, val_uop)))
        return

      var, dtype, hi, lo, idx_var = _get_lhs_info(lhs, ctx)
      out_vars = ('D0', 'D1', 'SCC', 'VCC', 'EXEC', 'PC', 'SDATA', 'VDATA', 'RETURN_DATA')

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

      # Bit range: D0[31:16].f16 = expr or SDATA[63:32] = expr
      if hi is not None and lo is not None:
        if hi < lo: hi, lo = lo, hi
        # Use 64-bit operations if slice extends beyond 32 bits
        use64 = hi >= 32
        op_dt = dtypes.uint64 if use64 else dtypes.uint32
        base = ctx.vars.get(var, UOp.const(op_dt, 0))
        if base.dtype != op_dt: base = _cast(base, op_dt)
        rhs_uop = _expr(rhs, ctx, dtype)
        rhs_bits = UOp(Ops.BITCAST, dtypes.uint16 if dtype == dtypes.float16 else (dtypes.uint32 if dtype == dtypes.float32 else dtypes.uint64), (rhs_uop,)) if dtype in FLOATS else _cast(rhs_uop, op_dt)
        if dtype == dtypes.float16: rhs_bits = _cast(rhs_bits, op_dt)
        mask = (1 << (hi - lo + 1)) - 1
        shifted = UOp(Ops.SHL, op_dt, (UOp(Ops.AND, op_dt, (rhs_bits, UOp.const(op_dt, mask))), UOp.const(op_dt, lo)))
        clear_mask = ~(mask << lo) & (0xffffffffffffffff if use64 else 0xffffffff)
        result = UOp(Ops.OR, op_dt, (UOp(Ops.AND, op_dt, (base, UOp.const(op_dt, clear_mask))), shifted))
        ctx.vars[var] = result
        if var in out_vars:
          ctx.outputs = [(n, u, d) for n, u, d in ctx.outputs if n != var]
          ctx.outputs.append((var, result, op_dt))
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

def _compile_pseudocode(pseudocode: str, mem_buf: UOp = MEM_BUF) -> tuple[UOp, list[tuple[str, DType]], dict[str, UOp], list[UOp]]:
  """Compile pseudocode to UOp graph. Returns (sink, outputs, input_vars, mem_stores)."""
  ctx = Ctx(mem_buf=mem_buf)
  for stmt in parse(pseudocode): _stmt(stmt, ctx)
  sink = UOp(Ops.SINK, dtypes.void, tuple(u for _, u, _ in ctx.outputs) or ())
  return sink, [(n, d) for n, _, d in ctx.outputs], INPUT_VARS, ctx.mem_stores

# Map dtype to memory accessor name for emu.py
_DTYPE_ACCESSOR = {
  dtypes.uint8: 'u8', dtypes.int8: 'i8', dtypes.uint16: 'u16', dtypes.int16: 'i16',
  dtypes.uint32: 'u32', dtypes.int32: 'i32', dtypes.uint64: 'u64', dtypes.int64: 'i64',
  dtypes.float32: 'u32', dtypes.float64: 'u64',
}

def _make_fn(sink: UOp, output_info: list[tuple[str, DType]], input_vars: dict[str, UOp], mem_stores: list[UOp]):
  """Create runtime function using substitute+simplify, with memory ops on sink."""
  # Add stores to sink first so they're included in detection
  if mem_stores: sink = UOp(Ops.SINK, dtypes.void, sink.src + tuple(mem_stores))
  # Detect memory type from UOps: check if any INDEX uses DEFINE_LOCAL
  topo = sink.toposort()
  is_lds = any(u.op == Ops.DEFINE_LOCAL for u in topo)
  is_mem = bool(mem_stores) or any(u.op == Ops.LOAD for u in topo)

  def _extract_results(s, MEM=None):
    # Execute stores and extract output values from simplified sink
    for u in s.src:
      if u.op == Ops.STORE:
        idx_uop, val_uop = u.src[0], u.src[1]
        addr, val, dt = int(idx_uop.src[1].arg), val_uop.arg, idx_uop.dtype.base
        acc = _DTYPE_ACCESSOR.get(dt, 'u32')
        if dt == dtypes.float32: val = struct.unpack('<I', struct.pack('<f', val))[0]
        elif dt == dtypes.float64: val = struct.unpack('<Q', struct.pack('<d', val))[0]
        setattr(MEM[addr], acc, int(val))
    result = {}
    for i, (name, dtype) in enumerate(output_info):
      if i >= len(s.src) or s.src[i].op != Ops.CONST: continue
      result[name] = _float_to_bits(s.src[i].arg, dtype) if dtype in FLOATS else int(s.src[i].arg) & (0xffffffff if dtype.itemsize <= 4 else 0xffffffffffffffff)
    return result

  if is_lds:
    # DS (LDS) ops: fn(MEM, addr, data0, data1, offset0, offset1)
    def fn(MEM, addr, data0=0, data1=0, offset0=0, offset1=0):
      dvars = {input_vars['ADDR']: UOp.const(dtypes.uint64, addr), input_vars['DATA']: UOp.const(dtypes.uint64, data0),
               input_vars['DATA2']: UOp.const(dtypes.uint64, data1), input_vars['OFFSET']: UOp.const(dtypes.uint32, offset0),
               input_vars['OFFSET0']: UOp.const(dtypes.uint32, offset0), input_vars['OFFSET1']: UOp.const(dtypes.uint32, offset1),
               input_vars['RETURN_DATA']: UOp.const(dtypes.uint64, 0)}
      s1 = sink.substitute(dvars).simplify()
      # Replace LOADs with actual values from LDS, then simplify again
      loads = {}
      for u in s1.toposort():
        if u.op == Ops.LOAD:
          idx_uop = u.src[0]
          load_addr, dt = int(idx_uop.src[1].arg), idx_uop.dtype.base
          acc = _DTYPE_ACCESSOR.get(dt, 'u32')
          loads[u] = UOp.const(dt, getattr(MEM[load_addr], acc))
      s2 = s1.substitute(loads).simplify() if loads else s1
      return _extract_results(s2, MEM)
    return fn
  elif is_mem:
    # SMEM/FLAT/GLOBAL ops: fn(MEM, addr, vdata, vdst)
    def fn(MEM, addr, vdata=0, vdst=0):
      dvars = {input_vars['ADDR']: UOp.const(dtypes.uint64, addr), input_vars['SDATA']: UOp.const(dtypes.uint64, 0),
               input_vars['VDATA']: UOp.const(dtypes.uint64, vdata), input_vars['VDST']: UOp.const(dtypes.uint64, vdst),
               input_vars['DATA']: UOp.const(dtypes.uint64, vdata), input_vars['DATA2']: UOp.const(dtypes.uint64, 0),
               input_vars['RETURN_DATA']: UOp.const(dtypes.uint64, 0)}
      s1 = sink.substitute(dvars).simplify()
      # Replace LOADs with actual values from MEM, then simplify again
      loads = {}
      for u in s1.toposort():
        if u.op == Ops.LOAD:
          idx_uop = u.src[0]
          load_addr, dt = int(idx_uop.src[1].arg), idx_uop.dtype.base
          acc = _DTYPE_ACCESSOR.get(dt, 'u32')
          loads[u] = UOp.const(dt, getattr(MEM[load_addr], acc))
      s2 = s1.substitute(loads).simplify() if loads else s1
      return _extract_results(s2, MEM)
    return fn
  else:
    # ALU ops: fn(s0, s1, s2, d0, scc, vcc, laneId, exec_mask, literal, VGPR, ...)
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
      return _extract_results(sink.substitute(dvars).simplify())
    return fn

# Ops with known issues (subtle float semantics, register array access, unimplemented features)
_SKIP_OPS = {
  # Float ops with subtle semantics (neg zero, NaN handling)
  'V_CEIL_F16', 'V_CEIL_F32', 'V_CEIL_F64', 'V_FLOOR_F16', 'V_FLOOR_F32', 'V_FLOOR_F64',
  'V_FRACT_F16', 'V_FRACT_F32', 'V_FRACT_F64', 'V_DIV_FMAS_F32', 'V_DIV_FMAS_F64',
  'V_CMP_CLASS_F16', 'V_CMP_CLASS_F32', 'V_CMP_CLASS_F64', 'V_CMPX_CLASS_F16', 'V_CMPX_CLASS_F32', 'V_CMPX_CLASS_F64',
  'V_FREXP_MANT_F16', 'V_FREXP_MANT_F32', 'V_FREXP_MANT_F64',
  'V_DIV_FIXUP_F16', 'V_DIV_FIXUP_F32', 'V_DIV_SCALE_F32', 'V_DIV_SCALE_F64',  # complex NaN/inf/denorm handling
  'V_TRIG_PREOP_F64',  # lookup table for 2/PI mantissa bits
  'V_MIN_F16', 'V_MIN_F32', 'V_MIN_F64', 'V_MAX_F16', 'V_MAX_F32', 'V_MAX_F64',  # neg zero handling: -0 < +0
  'V_SIN_F16', 'V_SIN_F32', 'V_COS_F16', 'V_COS_F32',  # transcendental with special range reduction
  # Bit manipulation ops (need CLZ/CTZ/BREV intrinsics)
  'V_CLZ_I32_U32', 'V_CTZ_I32_B32', 'S_BREV_B32', 'S_BREV_B64',
  'S_FF0_I32_B32', 'S_FF0_I32_B64', 'S_FF1_I32_B32', 'S_FF1_I32_B64',
  'S_FLBIT_I32', 'S_FLBIT_I32_B32', 'S_FLBIT_I32_B64', 'S_FLBIT_I32_I32', 'S_FLBIT_I32_I64',
  'S_BITREPLICATE_B64_B32',  # bit replication loop
  'S_BITSET0_B32', 'S_BITSET0_B64', 'S_BITSET1_B32', 'S_BITSET1_B64',  # D0[S0[n:0]] = val (dynamic index into output)
  'S_QUADMASK_B32', 'S_QUADMASK_B64', 'S_WQM_B32', 'S_WQM_B64',  # quad/wave mask ops with +: slice syntax
  # Register array access (SRC0, DST, VGPR[], SGPR[])
  'S_GETREG_B32', 'S_SETREG_B32', 'S_SETREG_IMM32_B32',
  'S_MOVRELD_B32', 'S_MOVRELD_B64', 'S_MOVRELS_B32', 'S_MOVRELS_B64', 'S_MOVRELSD_2_B32',
  'V_MOVRELD_B32', 'V_MOVRELS_B32', 'V_MOVRELSD_B32', 'V_MOVRELSD_2_B32',
  'V_READFIRSTLANE_B32', 'V_WRITELANE_B32', 'V_READLANE_B32',
  'V_PERMLANE16_B32', 'V_PERMLANE64_B32', 'V_PERMLANEX16_B32',
  'V_MBCNT_HI_U32_B32', 'V_MBCNT_LO_U32_B32',
  'S_SENDMSG_RTN_B32', 'S_SENDMSG_RTN_B64',
  'V_SWAPREL_B32',  # VGPR[laneId][addr] register array access
  'V_PERM_B32',  # BYTE_PERMUTE function not implemented
  # Control flow / special ops (no actual computation)
  'S_NOP', 'S_SETHALT', 'S_TRAP',
  # 65-bit intermediate results / multi-output with carry
  'V_MAD_U64_U32', 'V_MAD_I64_I32',  # { D1.u1, D0.u64 } = 65-bit result
  # Dot product ops (need bf16/u4 conversion functions or array declarations)
  'V_DOT2_F32_BF16',  # bf16_to_f32 not implemented
  'V_DOT4_I32_IU8', 'V_DOT4_U32_U8',  # u8_to_u32 array access pattern
  'V_DOT8_I32_IU4', 'V_DOT8_U32_U4',  # u4_to_u32 array access pattern
  # VOP3P mixed precision (S[i] array access in loop)
  'V_FMA_MIX_F32', 'V_FMA_MIXLO_F16', 'V_FMA_MIXHI_F16',
  # Lookup table ops
  'V_CVT_OFF_F32_I4',  # CVT_OFF_TABLE lookup
}

# Patterns that still need pcode (register arrays, special ops, complex atomics)
_PCODE_PATTERNS = ('LDS[', 'LDS(', 'VGPR[', 'SGPR[', 'GPR[', 'GS_REGS', 'thread_in[', 'thread_out[', 'thread_valid[',
                   'DATA2', 'OFFSET0', 'OFFSET1')  # DS ops with dual offsets or DATA2
# Wide outputs (>64 bit) that need pcode's arbitrary-precision integers
_WIDE_OUTPUT_PATTERNS = ('SDATA[95', 'SDATA[127', 'SDATA[159', 'SDATA[191', 'SDATA[223', 'SDATA[255',  # SMEM B128, B256, B512
                         'VDATA[95', 'VDATA[127')  # FLAT B128

@functools.cache
def compile_uop(op_name: str, pseudocode: str):
  """Compile pseudocode to UOp-based function. Returns None if unsupported."""
  if op_name in _SKIP_OPS: return None
  if any(p in pseudocode for p in _PCODE_PATTERNS): return None  # these patterns still need pcode
  if any(p in pseudocode for p in _WIDE_OUTPUT_PATTERNS): return None  # >64-bit outputs need pcode
  # DS ops use LDS (local memory), others use global MEM
  is_ds = op_name.startswith('DS_')
  mem_buf = LDS_BUF if is_ds else MEM_BUF
  sink, output_info, input_vars, mem_stores = _compile_pseudocode(pseudocode, mem_buf)
  return _make_fn(sink, output_info, input_vars, mem_stores)
