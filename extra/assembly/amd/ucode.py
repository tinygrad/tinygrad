# UOp-based pseudocode compiler for AMD GPU instruction emulation
import functools, struct, math
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes, DType, AddrSpace
from extra.assembly.amd.qcode import parse, Assign, Declare, If, For

SIGNED = (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64)
FLOATS = (dtypes.float16, dtypes.float32, dtypes.float64)

# FTZ (Flush To Zero): RDNA3 default mode flushes f32 denormals to ±0
def _ftz32(bits: int) -> float:
  bits = bits & 0xffffffff
  if (bits & 0x7f800000) == 0 and (bits & 0x007fffff) != 0:  # denormal
    return 0.0
  return struct.unpack('<f', struct.pack('<I', bits))[0]

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
  'ADDR': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('ADDR', 0, 0xffffffffffffffff)),
  'ADDR_BASE': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('ADDR', 0, 0xffffffffffffffff)),  # Alias for ADDR
  'SDATA': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('SDATA', 0, 0xffffffffffffffff)),
  'VDATA': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('VDATA', 0, 0xffffffffffffffff)),
  'VDST': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('VDST', 0, 0xffffffffffffffff)),
  'RETURN_DATA': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('RETURN_DATA', 0, 0xffffffffffffffff)),
  'DATA': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('DATA', 0, 0xffffffffffffffff)),
  'DATA2': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('DATA2', 0, 0xffffffffffffffff)),
  'OFFSET': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('OFFSET', 0, 0xffff)),
  'OFFSET0': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('OFFSET0', 0, 0xff)),
  'OFFSET1': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('OFFSET1', 0, 0xff)),
  'OPSEL': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('OPSEL', 0, 7)),
  'OPSEL_HI': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('OPSEL_HI', 0, 7)),
}

MEM_BUF = UOp(Ops.DEFINE_GLOBAL, dtypes.uint8.ptr(addrspace=AddrSpace.GLOBAL), arg=0)
LDS_BUF = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(addrspace=AddrSpace.LOCAL), arg=0)

class Ctx:
  def __init__(self, mem_buf: UOp = MEM_BUF):
    self.vars: dict[str, UOp] = dict(INPUT_VARS)
    self.decls: dict[str, DType] = {}
    self.outputs: list[tuple[str, UOp, DType]] = []
    self.mem_stores: list[UOp] = []
    self.mem_buf = mem_buf

def _expr(node: UOp, ctx: Ctx, hint: DType = None) -> UOp:
  """Transform parsed UOp expression to resolved UOp."""
  match node:
    case UOp(Ops.CONST, dt, _, val):
      dt = dt if dt != dtypes.int32 or hint is None else hint
      if isinstance(val, float) and dt not in FLOATS: dt = dtypes.float32
      return UOp.const(dt, val)

    case UOp(Ops.DEFINE_VAR, _, _, (name, None, None)):
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

    case UOp(Ops.BITCAST, dt, (inner,)):
      # Handle MEM[addr].type -> memory load
      if inner.op == Ops.CUSTOM and inner.arg == 'MEM':
        addr_uop = _expr(inner.src[0], ctx, dtypes.uint64)
        buf = ctx.mem_buf
        idx = UOp(Ops.INDEX, dt.ptr(0, buf.dtype.addrspace), (buf, addr_uop))
        return UOp(Ops.LOAD, dt, (idx,))
      # Handle Var.type
      if inner.op == Ops.DEFINE_VAR and inner.arg[1] is None:
        name = inner.arg[0]
        # Handle INF.f32, INF.f64, NAN.f32, NAN.f64, etc.
        if name == 'INF' or name in ('+INF', '-INF'):
          return UOp.const(dt, float('-inf') if name.startswith('-') else float('inf'))
        if name == 'NAN':
          return UOp.const(dt, float('nan'))
        if name == 'DENORM':
          denorm = {dtypes.float32: 1.17549435e-38, dtypes.float64: 2.2250738585072014e-308}.get(dt, 1.17549435e-38)
          return UOp.const(dt, denorm)
        if name in ('VCCZ', 'EXECZ'):
          return _cast(UOp(Ops.CMPEQ, dtypes.bool, (ctx.vars.get('VCC' if name == 'VCCZ' else 'EXEC'), UOp.const(dtypes.uint64, 0))), dtypes.uint32)
        if name.startswith('WAVE_STATUS.COND_DBG'): return UOp.const(dtypes.uint32, 0)
        vn = name + '_64' if dt.itemsize == 8 and name.isupper() else name
        base = ctx.vars.get(vn) if vn in ctx.vars else ctx.vars.get(name)
        if base is None: raise ValueError(f"Unknown variable: {name}")
        if dt.itemsize == 3 and 'int' in dt.name:
          masked = UOp(Ops.AND, dtypes.uint32, (base, UOp.const(dtypes.uint32, 0xffffff)))
          if 'uint' not in dt.name:
            return UOp(Ops.SUB, dtypes.int32, (UOp(Ops.XOR, dtypes.int32, (masked, UOp.const(dtypes.int32, 0x800000))), UOp.const(dtypes.int32, 0x800000)))
          return masked
        if dt == dtypes.float16:
          return UOp(Ops.BITCAST, dtypes.float16, (UOp(Ops.AND, dtypes.uint16, (_cast(base, dtypes.uint16), UOp.const(dtypes.uint16, 0xffff))),))
        if dt in FLOATS: return UOp(Ops.BITCAST, dt, (base,))
        if dt in SIGNED:
          base64 = ctx.vars.get(name + '_64') if (name + '_64') in ctx.vars else base
          return _cast(base64 if dt == dtypes.int64 else base, dt)
        return _cast(base, dt)
      inner_resolved = _expr(inner, ctx, dt)
      if dt == dtypes.float16: return UOp(Ops.BITCAST, dt, (_cast(inner_resolved, dtypes.uint16),))
      if dt == dtypes.bfloat16: return UOp(Ops.BITCAST, dt, (_cast(inner_resolved, dtypes.uint16),))
      if dt in FLOATS: return UOp(Ops.BITCAST, dt, (inner_resolved,))
      return _cast(inner_resolved, dt)

    case UOp(Ops.CUSTOMI, _, (base_expr, hi_expr, lo_expr)):  # Slice or array access
      # Check for array element access first: arr[idx] where arr is a vector type
      if base_expr.op == Ops.DEFINE_VAR and base_expr.arg[1] is None and hi_expr is lo_expr:
        name = base_expr.arg[0]
        var_dtype = ctx.decls.get(name)
        if var_dtype is not None and var_dtype.count > 1:
          # Array element access - look up stored element
          idx_uop = _expr(hi_expr, ctx)
          idx_uop = idx_uop.simplify()
          if idx_uop.op == Ops.CONST:
            arr_key = f"{name}_{int(idx_uop.arg)}"
            if arr_key in ctx.vars:
              return ctx.vars[arr_key]
            # Element not set, return default value
            return UOp.const(var_dtype.scalar(), 0)
      base, hi_uop, lo_uop = _expr(base_expr, ctx), _expr(hi_expr, ctx), _expr(lo_expr, ctx)
      # Single-bit slice: base[idx:idx] -> (base >> idx) & 1
      if hi_expr is lo_expr:
        return UOp(Ops.AND, dtypes.uint32, (_cast(UOp(Ops.SHR, base.dtype, (base, _cast(lo_uop, base.dtype))), dtypes.uint32), UOp.const(dtypes.uint32, 1)))
      # Simplify the bounds to get constant values (needed when loop variables are substituted)
      hi_uop, lo_uop = hi_uop.simplify(), lo_uop.simplify()
      if hi_uop.op == Ops.CONST and lo_uop.op == Ops.CONST:
        hi_val, lo_val = int(hi_uop.arg), int(lo_uop.arg)
        if hi_val < lo_val:
          width = lo_val - hi_val + 1
          if width == 32:
            result = UOp.const(dtypes.uint32, 0)
            for i in range(32):
              bit = UOp(Ops.AND, dtypes.uint32, (UOp(Ops.SHR, dtypes.uint32, (_cast(base, dtypes.uint32), UOp.const(dtypes.uint32, i))), UOp.const(dtypes.uint32, 1)))
              result = UOp(Ops.OR, dtypes.uint32, (result, UOp(Ops.SHL, dtypes.uint32, (bit, UOp.const(dtypes.uint32, 31 - i)))))
            return result
          elif width == 64:
            result = UOp.const(dtypes.uint64, 0)
            for i in range(64):
              bit = UOp(Ops.AND, dtypes.uint64, (UOp(Ops.SHR, dtypes.uint64, (_cast(base, dtypes.uint64), UOp.const(dtypes.uint64, i))), UOp.const(dtypes.uint64, 1)))
              result = UOp(Ops.OR, dtypes.uint64, (result, UOp(Ops.SHL, dtypes.uint64, (bit, UOp.const(dtypes.uint64, 63 - i)))))
            return result
          hi_val, lo_val = lo_val, hi_val
        shifted = UOp(Ops.SHR, base.dtype, (base, UOp.const(base.dtype, lo_val))) if lo_val else base
        return UOp(Ops.AND, dtypes.uint32, (_cast(shifted, dtypes.uint32), UOp.const(dtypes.uint32, (1 << (hi_val - lo_val + 1)) - 1)))
      raise ValueError(f"Non-constant slice bounds: {node}")

    case UOp(Ops.CAST, dt, (inner,)):
      inner_resolved = _expr(inner, ctx, dt)
      if dt in FLOATS:
        # For 32'F(0xffc00000) etc, treat integer constants as BITCAST (interpret bits as float)
        if inner_resolved.op == Ops.CONST and inner_resolved.dtype not in FLOATS:
          return UOp(Ops.BITCAST, dt, (inner_resolved,))
        return UOp(Ops.CAST, dt, (inner_resolved,))
      if inner_resolved.dtype.itemsize == dt.itemsize:
        return _cast(inner_resolved, dt)
      if dt in SIGNED and inner_resolved.dtype in SIGNED: return UOp(Ops.CAST, dt, (inner_resolved,))
      return _cast(inner_resolved, dt)

    case UOp(Ops.NEG, _, (src,)):
      val = _expr(src, ctx, hint)
      return UOp(Ops.NEG, val.dtype, (val,))

    case UOp(Ops.XOR, _, (src,)) if len(node.src) == 1:  # Unary ~ (bitwise not)
      val = _expr(src, ctx, hint)
      return UOp(Ops.XOR, val.dtype, (val, UOp.const(val.dtype, -1)))

    case UOp(Ops.CMPEQ, _, (src,)) if len(node.src) == 1:  # Unary ! (logical not)
      val = _expr(src, ctx, hint)
      return UOp(Ops.CMPEQ, dtypes.bool, (val, UOp.const(val.dtype, 0)))

    case UOp(Ops.WHERE, _, (cond, tv, fv)):
      c, t = _expr(cond, ctx), _expr(tv, ctx, hint)
      f = _expr(fv, ctx, t.dtype)
      return UOp(Ops.WHERE, t.dtype, (c, t, f))

    case UOp(op, _, (l_expr, r_expr)) if op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.AND, Ops.OR, Ops.XOR, Ops.SHL, Ops.SHR,
                                                 Ops.CMPLT, Ops.CMPLE, Ops.CMPEQ, Ops.CMPNE, Ops.POW, Ops.MOD):
      l = _expr(l_expr, ctx, hint)
      r = _expr(r_expr, ctx, l.dtype if l.dtype in FLOATS else hint)
      if op in (Ops.ADD, Ops.SUB, Ops.MUL) and l.dtype in (dtypes.int32, dtypes.int64):
        udt = {dtypes.int32: dtypes.uint32, dtypes.int64: dtypes.uint64}[l.dtype]
        lu = l.src[0] if l.op == Ops.BITCAST and l.src[0].dtype == udt else _cast(l, udt)
        ru = r.src[0] if r.op == Ops.BITCAST and r.src[0].dtype == udt else _cast(r, udt)
        return UOp(op, udt, (lu, ru))
      result_dt = l.dtype if l.dtype in FLOATS else r.dtype if r.dtype in FLOATS else l.dtype
      if op in (Ops.CMPLT, Ops.CMPLE, Ops.CMPEQ, Ops.CMPNE): return UOp(op, dtypes.bool, (l, r))
      if op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.AND, Ops.OR, Ops.XOR, Ops.SHL, Ops.SHR): return UOp(op, result_dt, (l, r))
      if op is Ops.POW:
        exp = UOp(Ops.CAST, result_dt, (r,)) if r.dtype != result_dt else r
        if l.op == Ops.CONST and l.arg == 2.0: return UOp(Ops.EXP2, result_dt, (exp,))
        return UOp(Ops.EXP2, result_dt, (UOp(Ops.MUL, result_dt, (exp, UOp(Ops.LOG2, result_dt, (l,)))),))
      if op is Ops.MOD:
        div = UOp(Ops.IDIV, result_dt, (l, r))
        return UOp(Ops.SUB, result_dt, (l, UOp(Ops.MUL, result_dt, (div, r))))

    case UOp(Ops.CUSTOM, _, args, name):  # Call
      resolved_args = [_expr(a, ctx, hint) for a in args]
      return _transform_call(name, resolved_args, hint)

    case UOp(Ops.CAT, _, exprs):  # Pack
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
CVT_MAP = {'u32_to_f32': (dtypes.float32, False), 'i32_to_f32': (dtypes.float32, False), 'f32_to_u32': (dtypes.uint32, True),
           'f32_to_i32': (dtypes.int32, False), 'f16_to_f32': (dtypes.float32, False), 'f32_to_f16': (dtypes.float16, False),
           'f32_to_u8': (dtypes.uint8, False), 'f32_to_i8': (dtypes.int8, False), 'f32_to_u16': (dtypes.uint16, False),
           'f32_to_i16': (dtypes.int16, False), 'v_cvt_u16_f32': (dtypes.uint16, False), 'v_cvt_i16_f32': (dtypes.int16, False),
           'f64_to_i32': (dtypes.int32, False), 'f64_to_u32': (dtypes.uint32, True), 'i32_to_f64': (dtypes.float64, False),
           'u32_to_f64': (dtypes.float64, False), 'f64_to_f32': (dtypes.float32, False), 'f32_to_f64': (dtypes.float64, False),
           'u16_to_f16': (dtypes.float16, False), 'i16_to_f16': (dtypes.float16, False), 'f16_to_u16': (dtypes.uint16, False),
           'f16_to_i16': (dtypes.int16, False)}
MATH_OPS = {'trunc': Ops.TRUNC, 'sqrt': Ops.SQRT, 'exp2': Ops.EXP2, 'log2': Ops.LOG2, 'sin': Ops.SIN, 'rcp': Ops.RECIPROCAL}

def _call_MEM(v): return v
def _call_bf16_to_f32(v):
  bits = _cast(v, dtypes.uint32)
  shifted = UOp(Ops.SHL, dtypes.uint32, (bits, UOp.const(dtypes.uint32, 16)))
  return UOp(Ops.BITCAST, dtypes.float32, (shifted,))
def _call_fma(a, b, c): return UOp(Ops.MULACC, c.dtype, (a, b, c))
def _call_abs(v): return UOp(Ops.WHERE, v.dtype, (UOp(Ops.CMPLT, dtypes.bool, (v, UOp.const(v.dtype, 0))), UOp(Ops.NEG, v.dtype, (v,)), v))
def _call_cos(v): return UOp(Ops.SIN, v.dtype, (UOp(Ops.ADD, v.dtype, (v, UOp.const(v.dtype, 1.5707963267948966))),))
def _call_rsqrt(v): return UOp(Ops.RECIPROCAL, v.dtype, (UOp(Ops.SQRT, v.dtype, (v,)),))
def _call_clamp(x, lo, hi):
  clamped = UOp(Ops.WHERE, x.dtype, (UOp(Ops.CMPLT, dtypes.bool, (x, lo)), lo, x))
  return UOp(Ops.WHERE, x.dtype, (UOp(Ops.CMPLT, dtypes.bool, (hi, clamped)), hi, clamped))
def _call_floor(v):
  truncated = UOp(Ops.TRUNC, v.dtype, (v,))
  needs_adjust = UOp(Ops.CMPLT, dtypes.bool, (v, truncated))
  return UOp(Ops.WHERE, v.dtype, (needs_adjust, UOp(Ops.SUB, v.dtype, (truncated, UOp.const(v.dtype, 1))), truncated))
def _call_fract(v): return UOp(Ops.SUB, v.dtype, (v, _call_floor(v)))
def _call_isNAN(v): return UOp(Ops.CMPNE, dtypes.bool, (v, v))
def _call_isSignalNAN(v):
  # Signaling NaN: exponent all 1s, mantissa non-zero, MSB of mantissa is 0
  # Unwrap CAST to check on original float type
  while v.op == Ops.CAST and v.dtype in FLOATS: v = v.src[0]
  uint_dt, _, exp_shift, exp_mask, mant_mask = FP_INFO.get(v.dtype, FP_INFO[dtypes.float32])
  bits = UOp(Ops.BITCAST, uint_dt, (v,))
  exp = UOp(Ops.AND, uint_dt, (UOp(Ops.SHR, uint_dt, (bits, UOp.const(uint_dt, exp_shift))), UOp.const(uint_dt, exp_mask)))
  mant = UOp(Ops.AND, uint_dt, (bits, UOp.const(uint_dt, mant_mask)))
  quiet_bit = {dtypes.float64: 0x8000000000000, dtypes.float32: 0x400000, dtypes.float16: 0x200}.get(v.dtype, 0x400000)
  is_exp_all_ones = UOp(Ops.CMPEQ, dtypes.bool, (exp, UOp.const(uint_dt, exp_mask)))
  is_mant_nonzero = UOp(Ops.CMPNE, dtypes.bool, (mant, UOp.const(uint_dt, 0)))
  is_quiet_bit_clear = UOp(Ops.CMPEQ, dtypes.bool, (UOp(Ops.AND, uint_dt, (mant, UOp.const(uint_dt, quiet_bit))), UOp.const(uint_dt, 0)))
  return UOp(Ops.AND, dtypes.bool, (UOp(Ops.AND, dtypes.bool, (is_exp_all_ones, is_mant_nonzero)), is_quiet_bit_clear))
def _call_isQuietNAN(v):
  # Quiet NaN: exponent all 1s, MSB of mantissa is 1
  while v.op == Ops.CAST and v.dtype in FLOATS: v = v.src[0]
  uint_dt, _, exp_shift, exp_mask, mant_mask = FP_INFO.get(v.dtype, FP_INFO[dtypes.float32])
  bits = UOp(Ops.BITCAST, uint_dt, (v,))
  exp = UOp(Ops.AND, uint_dt, (UOp(Ops.SHR, uint_dt, (bits, UOp.const(uint_dt, exp_shift))), UOp.const(uint_dt, exp_mask)))
  quiet_bit = {dtypes.float64: 0x8000000000000, dtypes.float32: 0x400000, dtypes.float16: 0x200}.get(v.dtype, 0x400000)
  is_exp_all_ones = UOp(Ops.CMPEQ, dtypes.bool, (exp, UOp.const(uint_dt, exp_mask)))
  is_quiet_bit_set = UOp(Ops.CMPNE, dtypes.bool, (UOp(Ops.AND, uint_dt, (bits, UOp.const(uint_dt, quiet_bit))), UOp.const(uint_dt, 0)))
  return UOp(Ops.AND, dtypes.bool, (is_exp_all_ones, is_quiet_bit_set))
def _call_cvtToQuietNAN(v): return v
def _call_isINF(v):
  return UOp(Ops.OR, dtypes.bool, (UOp(Ops.CMPEQ, dtypes.bool, (v, UOp.const(v.dtype, float('inf')))),
                                   UOp(Ops.CMPEQ, dtypes.bool, (v, UOp.const(v.dtype, float('-inf'))))))
def _call_isDENORM(v):
  # Denormalized float: exponent is 0, mantissa is non-zero, value is not zero
  uint_dt, _, exp_shift, exp_mask, mant_mask = FP_INFO.get(v.dtype, FP_INFO[dtypes.float32])
  bits = UOp(Ops.BITCAST, uint_dt, (v,))
  exp = UOp(Ops.AND, uint_dt, (UOp(Ops.SHR, uint_dt, (bits, UOp.const(uint_dt, exp_shift))), UOp.const(uint_dt, exp_mask)))
  mant = UOp(Ops.AND, uint_dt, (bits, UOp.const(uint_dt, mant_mask)))
  is_exp_zero = UOp(Ops.CMPEQ, dtypes.bool, (exp, UOp.const(uint_dt, 0)))
  is_mant_nonzero = UOp(Ops.CMPNE, dtypes.bool, (mant, UOp.const(uint_dt, 0)))
  return UOp(Ops.AND, dtypes.bool, (is_exp_zero, is_mant_nonzero))
def _call_sign(v):
  uint_dt, sign_shift, _, _, _ = FP_INFO.get(v.dtype, FP_INFO[dtypes.float32])
  bits = UOp(Ops.BITCAST, uint_dt, (v,))
  return UOp(Ops.AND, dtypes.uint32, (_cast(UOp(Ops.SHR, uint_dt, (bits, UOp.const(uint_dt, sign_shift))), dtypes.uint32), UOp.const(dtypes.uint32, 1)))
def _call_exponent(v):
  uint_dt, _, exp_shift, exp_mask, _ = FP_INFO.get(v.dtype, FP_INFO[dtypes.float32])
  bits = UOp(Ops.BITCAST, uint_dt, (v,))
  return UOp(Ops.AND, dtypes.uint32, (_cast(UOp(Ops.SHR, uint_dt, (bits, UOp.const(uint_dt, exp_shift))), dtypes.uint32), UOp.const(dtypes.uint32, exp_mask)))
def _call_mantissa(v):
  # AMD V_FREXP_MANT returns mantissa in [0.5, 1.0) range like math.frexp()[0]
  # For normalized floats: set exponent to bias-1 (makes value in [0.5,1.0))
  # For zero: return zero; for inf/nan: should be handled by caller
  uint_dt, sign_shift, exp_shift, exp_mask, mant_mask = FP_INFO.get(v.dtype, FP_INFO[dtypes.float32])
  bias = {dtypes.float64: 1023, dtypes.float32: 127, dtypes.float16: 15}.get(v.dtype, 127)
  bits = UOp(Ops.BITCAST, uint_dt, (v,))
  sign_and_mant = UOp(Ops.AND, uint_dt, (bits, UOp.const(uint_dt, (1 << sign_shift) | mant_mask)))
  new_exp = UOp.const(uint_dt, (bias - 1) << exp_shift)  # exponent = -1 in biased form
  result_bits = UOp(Ops.OR, uint_dt, (sign_and_mant, new_exp))
  result = UOp(Ops.BITCAST, v.dtype, (result_bits,))
  is_zero = UOp(Ops.CMPEQ, dtypes.bool, (v, UOp.const(v.dtype, 0.0)))
  return UOp(Ops.WHERE, v.dtype, (is_zero, v, result))
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
def _call_BYTE_PERMUTE(src, sel):
  # src is {S0, S1} = (S0 << 32) | S1, where bytes 0-3 are S1, bytes 4-7 are S0
  src64 = _cast(src, dtypes.uint64)
  sel_val = UOp(Ops.AND, dtypes.uint32, (_cast(sel, dtypes.uint32), UOp.const(dtypes.uint32, 0xff)))
  sel_idx = UOp(Ops.AND, dtypes.uint32, (sel_val, UOp.const(dtypes.uint32, 7)))
  sel_nibble = UOp(Ops.AND, dtypes.uint32, (sel_val, UOp.const(dtypes.uint32, 0xf)))
  # Normal byte select (sel 0-7): extract byte at index
  shift = UOp(Ops.SHL, dtypes.uint32, (sel_idx, UOp.const(dtypes.uint32, 3)))
  byte_val = _cast(UOp(Ops.AND, dtypes.uint64, (UOp(Ops.SHR, dtypes.uint64, (src64, _cast(shift, dtypes.uint64))), UOp.const(dtypes.uint64, 0xff))), dtypes.uint32)
  # Sign extension (sel 8-11): check bit 15/31/47/63 respectively
  def sign_ext_bit(bit_pos):
    bit = UOp(Ops.AND, dtypes.uint64, (UOp(Ops.SHR, dtypes.uint64, (src64, UOp.const(dtypes.uint64, bit_pos))), UOp.const(dtypes.uint64, 1)))
    return UOp(Ops.WHERE, dtypes.uint32, (UOp(Ops.CMPNE, dtypes.bool, (bit, UOp.const(dtypes.uint64, 0))), UOp.const(dtypes.uint32, 0xff), UOp.const(dtypes.uint32, 0)))
  sign8, sign9, sign10, sign11 = sign_ext_bit(15), sign_ext_bit(31), sign_ext_bit(47), sign_ext_bit(63)
  # Build result based on selector
  is_sel8 = UOp(Ops.CMPEQ, dtypes.bool, (sel_nibble, UOp.const(dtypes.uint32, 8)))
  is_sel9 = UOp(Ops.CMPEQ, dtypes.bool, (sel_nibble, UOp.const(dtypes.uint32, 9)))
  is_sel10 = UOp(Ops.CMPEQ, dtypes.bool, (sel_nibble, UOp.const(dtypes.uint32, 10)))
  is_sel11 = UOp(Ops.CMPEQ, dtypes.bool, (sel_nibble, UOp.const(dtypes.uint32, 11)))
  is_sel12 = UOp(Ops.CMPEQ, dtypes.bool, (sel_nibble, UOp.const(dtypes.uint32, 12)))
  is_sel_gt12 = UOp(Ops.CMPLT, dtypes.bool, (UOp.const(dtypes.uint32, 12), sel_nibble))
  result = byte_val
  result = UOp(Ops.WHERE, dtypes.uint32, (is_sel8, sign8, result))
  result = UOp(Ops.WHERE, dtypes.uint32, (is_sel9, sign9, result))
  result = UOp(Ops.WHERE, dtypes.uint32, (is_sel10, sign10, result))
  result = UOp(Ops.WHERE, dtypes.uint32, (is_sel11, sign11, result))
  result = UOp(Ops.WHERE, dtypes.uint32, (is_sel12, UOp.const(dtypes.uint32, 0), result))
  result = UOp(Ops.WHERE, dtypes.uint32, (is_sel_gt12, UOp.const(dtypes.uint32, 0xff), result))
  # High bit of selector (0x80) means return 0
  sel_hi = UOp(Ops.AND, dtypes.uint32, (sel_val, UOp.const(dtypes.uint32, 0x80)))
  return UOp(Ops.WHERE, dtypes.uint32, (UOp(Ops.CMPNE, dtypes.bool, (sel_hi, UOp.const(dtypes.uint32, 0))), UOp.const(dtypes.uint32, 0), result))

def _call_trig_preop_result(shift):
  # Returns CUSTOM op that gets evaluated at runtime with the 1201-bit constant
  return UOp(Ops.CUSTOM, dtypes.float64, (shift,), arg='trig_preop_result')

CALL_DISPATCH = {
  'MEM': _call_MEM, 'fma': _call_fma, 'abs': _call_abs, 'cos': _call_cos, 'rsqrt': _call_rsqrt,
  'clamp': _call_clamp, 'floor': _call_floor, 'fract': _call_fract, 'isNAN': _call_isNAN, 'isQuietNAN': _call_isQuietNAN,
  'isSignalNAN': _call_isSignalNAN, 'cvtToQuietNAN': _call_cvtToQuietNAN, 'isINF': _call_isINF, 'isDENORM': _call_isDENORM,
  'sign': _call_sign, 'exponent': _call_exponent, 'mantissa': _call_mantissa, 'isEven': _call_isEven,
  'signext': _call_signext, 'signext_from_bit': _call_signext_from_bit, 'ABSDIFF': _call_ABSDIFF, 'SAT8': _call_SAT8,
  'BYTE_PERMUTE': _call_BYTE_PERMUTE, 'bf16_to_f32': _call_bf16_to_f32, 'trig_preop_result': _call_trig_preop_result,
}

def _transform_call(name: str, a: list[UOp], hint: DType) -> UOp:
  if name in CALL_DISPATCH: return CALL_DISPATCH[name](*a)
  if name == 'pow':
    assert a[0].op == Ops.CONST and a[0].arg == 2.0, f"pow only supports base=2, got {a[0]}"
    result_dt = a[0].dtype if a[0].dtype in FLOATS else hint or dtypes.float32
    return UOp(Ops.EXP2, result_dt, (a[1] if a[1].dtype == result_dt else UOp(Ops.CAST, result_dt, (a[1],)),))
  if name in MATH_OPS: return UOp(MATH_OPS[name], a[0].dtype, (a[0],))
  if name == 'ldexp':
    # ldexp(x, exp) = x * 2^exp
    exp_float = UOp(Ops.CAST, a[0].dtype, (a[1],)) if a[1].dtype != a[0].dtype else a[1]
    return UOp(Ops.MUL, a[0].dtype, (a[0], UOp(Ops.EXP2, a[0].dtype, (exp_float,))))
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
  if name in ('v_sad_u8', 'v_msad_u8'):
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

def _get_lhs_info(lhs: UOp, ctx: Ctx) -> tuple[str, DType, int|None, int|None, str|None, int|None]:
  """Extract assignment target: (var_name, dtype, hi_bit, lo_bit, idx_var, array_idx)"""
  match lhs:
    case UOp(Ops.BITCAST, dt, (UOp(Ops.DEFINE_VAR, _, _, (name, None, None)),)): return name, dt, None, None, None, None
    case UOp(Ops.BITCAST, dt, (UOp(Ops.CUSTOMI, _, (UOp(Ops.DEFINE_VAR, _, _, (name, None, None)), UOp(Ops.CONST, _, _, hi), UOp(Ops.CONST, _, _, lo))),)):
      return name, dt, int(hi), int(lo), None, None
    case UOp(Ops.BITCAST, _, (UOp(Ops.CUSTOMI, _, (UOp(Ops.BITCAST, _, (UOp(Ops.DEFINE_VAR, _, _, (name, None, None)),)), UOp(Ops.DEFINE_VAR, _, _, (idx, None, None)), idx2)),)) if lhs.src[0].src[1] is lhs.src[0].src[2]:
      return name, dtypes.uint64, None, None, idx, None
    case UOp(Ops.BITCAST, dt, (UOp(Ops.CUSTOMI, _, (UOp(Ops.DEFINE_VAR, _, _, (name, None, None)), UOp(Ops.DEFINE_VAR, _, _, (idx, None, None)), idx2)),)) if lhs.src[0].src[1] is lhs.src[0].src[2]:
      return name, dt, None, None, idx, None
    case UOp(Ops.CUSTOMI, _, (UOp(Ops.BITCAST, _, (UOp(Ops.DEFINE_VAR, _, _, (name, None, None)),)), UOp(Ops.CONST, _, _, hi), UOp(Ops.CONST, _, _, lo))):
      return name, dtypes.uint32, int(hi), int(lo), None, None
    case UOp(Ops.CUSTOMI, _, (UOp(Ops.DEFINE_VAR, _, _, (name, None, None)), UOp(Ops.CONST, _, _, idx), _)) if lhs.src[1] is lhs.src[2]:
      # Check if this is array element access (variable is a vector type)
      var_dtype = ctx.decls.get(name)
      if var_dtype is not None and var_dtype.count > 1:
        return name, var_dtype.scalar(), None, None, None, int(idx)
      return name, dtypes.uint32, int(idx), int(idx), None, None
    case UOp(Ops.CUSTOMI, _, (UOp(Ops.DEFINE_VAR, _, _, (name, None, None)), UOp(Ops.CONST, _, _, hi), UOp(Ops.CONST, _, _, lo))):
      return name, dtypes.uint32, int(hi), int(lo), None, None
    case UOp(Ops.CUSTOMI, _, (UOp(Ops.BITCAST, dt, (UOp(Ops.DEFINE_VAR, _, _, (name, None, None)),)), UOp(Ops.DEFINE_VAR, _, _, (idx, None, None)), idx2)) if lhs.src[1] is lhs.src[2]:
      return name, dt, None, None, idx, None
    # Handle arr[i] where i is a variable - check if it's array element or bit index
    case UOp(Ops.CUSTOMI, _, (UOp(Ops.DEFINE_VAR, _, _, (name, None, None)), UOp(Ops.DEFINE_VAR, _, _, (idx, None, None)), idx2)) if lhs.src[1] is lhs.src[2]:
      var_dtype = ctx.decls.get(name)
      if var_dtype is not None and var_dtype.count > 1:
        # Array element access with variable index
        return name, var_dtype.scalar(), None, None, None, idx  # Return idx as variable name for array_idx
      return name, dtypes.uint32, None, None, idx, None
    case UOp(Ops.DEFINE_VAR, _, _, (name, None, None)):
      # If the variable already exists, use its dtype; otherwise default to uint32
      existing = ctx.vars.get(name)
      dtype = existing.dtype if existing is not None else dtypes.uint32
      return name, dtype, None, None, None, None
  raise ValueError(f"Cannot parse LHS: {lhs}")

def _stmt(stmt, ctx: Ctx):
  match stmt:
    case Declare(name, dtype):
      ctx.decls[name] = dtype
      # Special handling for S array - it maps to source operands S0, S1, S2
      if name == 'S' and dtype.count == 3:
        ctx.vars['S_0'] = ctx.vars['S0']
        ctx.vars['S_1'] = ctx.vars['S1']
        ctx.vars['S_2'] = ctx.vars['S2']
      else:
        # Initialize declared variable with zero value
        ctx.vars[name] = UOp.const(dtype, 0)
    case Assign(lhs, rhs):
      # Handle MEM[addr].type = value -> memory store
      if lhs.op == Ops.BITCAST and lhs.src[0].op == Ops.CUSTOM and lhs.src[0].arg == 'MEM':
        dt = lhs.dtype
        addr_uop = _expr(lhs.src[0].src[0], ctx, dtypes.uint64)
        val_uop = _expr(rhs, ctx, dt)
        buf = ctx.mem_buf
        idx = UOp(Ops.INDEX, dt.ptr(0, buf.dtype.addrspace), (buf, addr_uop))
        ctx.mem_stores.append(UOp(Ops.STORE, dtypes.void, (idx, val_uop)))
        return

      # Handle CAT (multi-output assignment) like {D1.u1, D0.u64} = ...
      if lhs.op == Ops.CAT:
        rhs_uop = _expr(rhs, ctx)
        out_vars = ('D0', 'D1', 'SCC', 'VCC', 'EXEC', 'PC', 'SDATA', 'VDATA', 'RETURN_DATA')
        offset = 0
        for part in reversed(lhs.src):  # CAT is hi, lo order, so reverse to get lo first
          if part.op == Ops.BITCAST and part.src[0].op == Ops.DEFINE_VAR:
            dt, name = part.dtype, part.src[0].arg[0]
            # Map non-standard dtypes to real dtypes
            if dt.name == 'u1': bits, real_dt = 1, dtypes.uint32
            elif dt == dtypes.ulong or dt.name == 'ulong': bits, real_dt = 64, dtypes.uint64
            else: bits, real_dt = dt.itemsize * 8, dt
            mask = (1 << bits) - 1
            extracted = UOp(Ops.AND, rhs_uop.dtype, (UOp(Ops.SHR, rhs_uop.dtype, (rhs_uop, UOp.const(rhs_uop.dtype, offset))), UOp.const(rhs_uop.dtype, mask)))
            val = _cast(extracted, real_dt)
            ctx.vars[name] = val
            if name in out_vars: ctx.outputs.append((name, val, real_dt))
            offset += bits
        return

      var, dtype, hi, lo, idx_var, array_idx = _get_lhs_info(lhs, ctx)
      out_vars = ('D0', 'D1', 'SCC', 'VCC', 'EXEC', 'PC', 'SDATA', 'VDATA', 'RETURN_DATA')

      # Handle array element assignment: arr[idx] = value
      if array_idx is not None:
        var_dtype = ctx.decls.get(var)
        if var_dtype is None: raise ValueError(f"Unknown array variable: {var}")
        rhs_uop = _expr(rhs, ctx, dtype)
        # array_idx can be an int or a variable name (str)
        if isinstance(array_idx, str):
          # Variable index - resolve it
          idx_uop = ctx.vars.get(array_idx)
          if idx_uop is not None and idx_uop.op == Ops.CONST:
            arr_key = f"{var}_{int(idx_uop.arg)}"
          else:
            raise ValueError(f"Non-constant array index: {array_idx}")
        else:
          arr_key = f"{var}_{array_idx}"
        ctx.vars[arr_key] = rhs_uop
        return

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

      if hi is not None and lo is not None:
        if hi < lo: hi, lo = lo, hi
        # Select dtype based on highest bit needed
        if hi >= 128: op_dt = dtypes.uint256
        elif hi >= 64: op_dt = dtypes.uint128
        elif hi >= 32: op_dt = dtypes.uint64
        else: op_dt = dtypes.uint32
        base = ctx.vars.get(var, UOp.const(op_dt, 0))
        if base.dtype != op_dt: base = _cast(base, op_dt)
        rhs_uop = _expr(rhs, ctx, dtype)
        rhs_bits = UOp(Ops.BITCAST, dtypes.uint16 if dtype == dtypes.float16 else (dtypes.uint32 if dtype == dtypes.float32 else dtypes.uint64), (rhs_uop,)) if dtype in FLOATS else _cast(rhs_uop, op_dt)
        if dtype == dtypes.float16: rhs_bits = _cast(rhs_bits, op_dt)
        mask = (1 << (hi - lo + 1)) - 1
        shifted = UOp(Ops.SHL, op_dt, (UOp(Ops.AND, op_dt, (rhs_bits, UOp.const(op_dt, mask))), UOp.const(op_dt, lo)))
        full_mask = (1 << (op_dt.itemsize * 8)) - 1  # itemsize is bytes, need bits
        clear_mask = ~(mask << lo) & full_mask
        result = UOp(Ops.OR, op_dt, (UOp(Ops.AND, op_dt, (base, UOp.const(op_dt, clear_mask))), shifted))
        ctx.vars[var] = result
        if var in out_vars:
          ctx.outputs = [(n, u, d) for n, u, d in ctx.outputs if n != var]
          ctx.outputs.append((var, result, op_dt))
        return

      rhs_uop = _expr(rhs, ctx, dtype)
      ctx.vars[var] = rhs_uop
      if dtype.itemsize == 8 and var in ('D0', 'D1', 'S0', 'S1'): ctx.vars[var + '_64'] = rhs_uop
      if var in out_vars: ctx.outputs.append((var, rhs_uop, dtype))

    case If(branches): _transform_if(branches, ctx)
    case For(var, start, end, body): _transform_for(var, start, end, body, ctx)

def _transform_if(branches: tuple, ctx: Ctx):
  # Process each branch by executing its body statements in a sub-context
  parsed = []
  for cond, body in branches:
    cond_uop = _expr(cond, ctx) if cond is not None else None
    # Create a sub-context that shares vars but has its own outputs
    sub_ctx = Ctx(mem_buf=ctx.mem_buf)
    sub_ctx.vars = dict(ctx.vars)
    sub_ctx.decls = dict(ctx.decls)
    for s in body: _stmt(s, sub_ctx)
    parsed.append((cond_uop, sub_ctx))

  # Collect all assigned variables across all branches (both outputs and locals)
  out_vars = ('D0', 'D1', 'SCC', 'VCC', 'EXEC', 'PC', 'SDATA', 'VDATA', 'RETURN_DATA')
  assigned_outputs = set()
  assigned_locals = set()
  for _, sub_ctx in parsed:
    for name, _, _ in sub_ctx.outputs:
      if name in out_vars: assigned_outputs.add(name)
    # Track local variables that were modified in branches
    for name, val in sub_ctx.vars.items():
      if name not in ctx.vars or ctx.vars[name] is not val:
        if name not in out_vars and name not in INPUT_VARS:
          assigned_locals.add(name)

  # Merge output variables
  for var in assigned_outputs:
    dtype = next((d for _, sub_ctx in parsed for n, _, d in sub_ctx.outputs if n == var), dtypes.uint32)
    result = ctx.vars.get(var, UOp.const(dtype, 0))
    for cond_uop, sub_ctx in reversed(parsed):
      branch_val = next((u for n, u, _ in sub_ctx.outputs if n == var), None)
      if branch_val is not None:
        result = branch_val if cond_uop is None else UOp(Ops.WHERE, branch_val.dtype, (cond_uop, branch_val, _cast(result, branch_val.dtype)))
    ctx.vars[var] = result
    ctx.outputs = [(n, u, d) for n, u, d in ctx.outputs if n != var]
    ctx.outputs.append((var, result, dtype))

  # Merge local variables (like 'result')
  for var in assigned_locals:
    dtype = ctx.decls.get(var, dtypes.uint32)
    result = ctx.vars.get(var, UOp.const(dtype, 0))
    for cond_uop, sub_ctx in reversed(parsed):
      if var in sub_ctx.vars and (var not in ctx.vars or sub_ctx.vars[var] is not ctx.vars[var]):
        branch_val = sub_ctx.vars[var]
        result = branch_val if cond_uop is None else UOp(Ops.WHERE, branch_val.dtype, (cond_uop, branch_val, _cast(result, branch_val.dtype)))
    ctx.vars[var] = result

def _transform_for(var: str, start: UOp, end: UOp, body: tuple, ctx: Ctx):
  start_val = start.arg if start.op == Ops.CONST else int(_expr(start, ctx).arg)
  end_val = end.arg if end.op == Ops.CONST else int(_expr(end, ctx).arg)
  var_dtype = ctx.decls.get(var, dtypes.uint32)
  for i in range(int(end_val), int(start_val) - 1, -1):
    ctx.vars[var] = UOp.const(var_dtype, i)
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
  ctx = Ctx(mem_buf=mem_buf)
  for stmt in parse(pseudocode): _stmt(stmt, ctx)
  sink = UOp(Ops.SINK, dtypes.void, tuple(u for _, u, _ in ctx.outputs) or ())
  return sink, [(n, d) for n, _, d in ctx.outputs], INPUT_VARS, ctx.mem_stores

_DTYPE_ACCESSOR = {
  dtypes.uint8: 'u8', dtypes.int8: 'i8', dtypes.uint16: 'u16', dtypes.int16: 'i16',
  dtypes.uint32: 'u32', dtypes.int32: 'i32', dtypes.uint64: 'u64', dtypes.int64: 'i64',
  dtypes.float32: 'u32', dtypes.float64: 'u64',
}

def _make_fn(sink: UOp, output_info: list[tuple[str, DType]], input_vars: dict[str, UOp], mem_stores: list[UOp]):
  if mem_stores: sink = UOp(Ops.SINK, dtypes.void, sink.src + tuple(mem_stores))
  topo = sink.toposort()
  is_lds = any(u.op == Ops.DEFINE_LOCAL for u in topo)
  is_mem = bool(mem_stores) or any(u.op == Ops.LOAD for u in topo)

  def _eval_uop(u: UOp) -> int|float|None:
    """Recursively evaluate a UOp tree to a constant value."""
    if u.op == Ops.CONST: return u.arg
    if u.op == Ops.CAST:
      v = _eval_uop(u.src[0])
      return v if v is not None else None
    if u.op == Ops.BITCAST:
      v = _eval_uop(u.src[0])
      if v is None: return None
      # Convert between int and float bit representations
      if u.dtype == dtypes.float64 and u.src[0].dtype in (dtypes.uint64, dtypes.int64):
        return struct.unpack('<d', struct.pack('<Q', int(v) & 0xffffffffffffffff))[0]
      if u.dtype == dtypes.float32 and u.src[0].dtype in (dtypes.uint32, dtypes.int32):
        return _ftz32(int(v))  # Apply FTZ for f32
      if u.dtype in (dtypes.uint64, dtypes.int64) and u.src[0].dtype == dtypes.float64:
        return struct.unpack('<Q', struct.pack('<d', float(v)))[0]
      if u.dtype in (dtypes.uint32, dtypes.int32) and u.src[0].dtype == dtypes.float32:
        return struct.unpack('<I', struct.pack('<f', float(v)))[0]
      return v
    if u.op == Ops.MULACC:
      a, b, c = _eval_uop(u.src[0]), _eval_uop(u.src[1]), _eval_uop(u.src[2])
      if a is None or b is None or c is None: return None
      return math.fma(float(a), float(b), float(c))
    if u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.AND, Ops.OR, Ops.XOR, Ops.SHR, Ops.SHL):
      l, r = _eval_uop(u.src[0]), _eval_uop(u.src[1])
      if l is None or r is None: return None
      if u.op == Ops.ADD: return l + r
      if u.op == Ops.SUB: return l - r
      if u.op == Ops.MUL: return l * r
      if u.op == Ops.AND: return int(l) & int(r)
      if u.op == Ops.OR: return int(l) | int(r)
      if u.op == Ops.XOR: return int(l) ^ int(r)
      if u.op == Ops.SHR: return int(l) >> int(r)
      if u.op == Ops.SHL: return int(l) << int(r)
    if u.op == Ops.NEG:
      v = _eval_uop(u.src[0])
      return -v if v is not None else None
    if u.op in (Ops.CMPEQ, Ops.CMPNE, Ops.CMPLT, Ops.CMPLE):
      l, r = _eval_uop(u.src[0]), _eval_uop(u.src[1])
      if l is None or r is None: return None
      if u.op == Ops.CMPEQ: return l == r
      if u.op == Ops.CMPNE: return l != r
      if u.op == Ops.CMPLT: return l < r
      if u.op == Ops.CMPLE: return l <= r
    if u.op == Ops.WHERE:
      c, t, f = _eval_uop(u.src[0]), _eval_uop(u.src[1]), _eval_uop(u.src[2])
      if c is None or t is None or f is None: return None
      return t if c else f
    if u.op == Ops.CUSTOM and u.arg == 'trig_preop_result':
      # Compute result from 1201-bit 2/PI constant
      shift = _eval_uop(u.src[0])
      if shift is None: return None
      TWO_OVER_PI_1201 = 0x0145f306dc9c882a53f84eafa3ea69bb81b6c52b3278872083fca2c757bd778ac36e48dc74849ba5c00c925dd413a32439fc3bd63962534e7dd1046bea5d768909d338e04d68befc827323ac7306a673e93908bf177bf250763ff12fffbc0b301fde5e2316b414da3eda6cfd9e4f96136e9e8c7ecd3cbfd45aea4f758fd7cbe2f67a0e73ef14a525d4d7f6bf623f1aba10ac06608df8f6
      # Extract 53 bits starting from position (1200 - shift) from the MSB
      shifted = (TWO_OVER_PI_1201 << int(shift)) >> (1201 - 53)
      mantissa = shifted & 0x1fffffffffffff
      return float(mantissa)
    return None

  def _extract_results(s, MEM=None):
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
      if i >= len(s.src): continue
      if s.src[i].op == Ops.CONST:
        val = s.src[i].arg
      else:
        val = _eval_uop(s.src[i])
        if val is None: continue
      if dtype in FLOATS:
        result[name] = _float_to_bits(val, dtype)
      else:
        # Mask to appropriate size: 32-bit, 64-bit, or wider (128/256/512 bits)
        result[name] = int(val) & ((1 << (dtype.itemsize * 8)) - 1)
    return result

  if is_lds:
    def fn(MEM, addr, data0=0, data1=0, offset0=0, offset1=0):
      dvars = {input_vars['ADDR']: UOp.const(dtypes.uint64, addr), input_vars['DATA']: UOp.const(dtypes.uint64, data0),
               input_vars['DATA2']: UOp.const(dtypes.uint64, data1), input_vars['OFFSET']: UOp.const(dtypes.uint32, offset0),
               input_vars['OFFSET0']: UOp.const(dtypes.uint32, offset0), input_vars['OFFSET1']: UOp.const(dtypes.uint32, offset1),
               input_vars['RETURN_DATA']: UOp.const(dtypes.uint64, 0)}
      s1 = sink.substitute(dvars).simplify()
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
    def fn(MEM, addr, vdata=0, vdst=0):
      dvars = {input_vars['ADDR']: UOp.const(dtypes.uint64, addr), input_vars['SDATA']: UOp.const(dtypes.uint64, 0),
               input_vars['VDATA']: UOp.const(dtypes.uint64, vdata), input_vars['VDST']: UOp.const(dtypes.uint64, vdst),
               input_vars['DATA']: UOp.const(dtypes.uint64, vdata), input_vars['DATA2']: UOp.const(dtypes.uint64, 0),
               input_vars['RETURN_DATA']: UOp.const(dtypes.uint64, 0)}
      s1 = sink.substitute(dvars).simplify()
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
    def fn(s0, s1, s2, d0, scc, vcc, laneId, exec_mask, literal, VGPR, src0_idx=0, vdst_idx=0, pc=None, opsel=0, opsel_hi=0):
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
        input_vars['OPSEL']: UOp.const(dtypes.uint32, opsel), input_vars['OPSEL_HI']: UOp.const(dtypes.uint32, opsel_hi),
      }
      return _extract_results(sink.substitute(dvars).simplify())
    return fn

# Ops that need Python exec features (inline conditionals, complex PDF fixes) - fall back to pcode.py
_SKIP_OPS: set[str] = set()

_PCODE_PATTERNS: tuple[str, ...] = ()
_WIDE_OUTPUT_PATTERNS: tuple[str, ...] = ()

def _apply_pseudocode_fixes(op_name: str, pcode: str) -> str:
  """Apply known fixes for PDF pseudocode bugs - same as pcode.py but for raw pseudocode."""
  if op_name == 'V_DIV_FMAS_F32':
    pcode = pcode.replace('D0.f32 = 2.0F ** 32 * fma(S0.f32, S1.f32, S2.f32)',
      'D0.f32 = (exponent(S2.f32) > 127) ? (2.0F ** 64 * fma(S0.f32, S1.f32, S2.f32)) : (2.0F ** -64 * fma(S0.f32, S1.f32, S2.f32))')
  if op_name == 'V_DIV_FMAS_F64':
    pcode = pcode.replace('D0.f64 = 2.0 ** 64 * fma(S0.f64, S1.f64, S2.f64)',
      'D0.f64 = (exponent(S2.f64) > 1023) ? (2.0 ** 128 * fma(S0.f64, S1.f64, S2.f64)) : (2.0 ** -128 * fma(S0.f64, S1.f64, S2.f64))')
  if op_name == 'V_DIV_FIXUP_F32':
    # When S0 (estimate) is NaN but inputs are valid, return OVERFLOW instead of NaN
    pcode = pcode.replace('D0.f32 = sign_out ? -abs(S0.f32) : abs(S0.f32)',
      'D0.f32 = isNAN(S0.f32) ? (sign_out ? -OVERFLOW_F32 : OVERFLOW_F32) : (sign_out ? -abs(S0.f32) : abs(S0.f32))')
  if op_name == 'V_DIV_FIXUP_F64':
    pcode = pcode.replace('D0.f64 = sign_out ? -abs(S0.f64) : abs(S0.f64)',
      'D0.f64 = isNAN(S0.f64) ? (sign_out ? -OVERFLOW_F64 : OVERFLOW_F64) : (sign_out ? -abs(S0.f64) : abs(S0.f64))')
  if op_name == 'V_DIV_SCALE_F32':
    # Fix 0: Replace DENORM comparisons with isDENORM() calls (order matters - do longer patterns first)
    pcode = pcode.replace('S2.f32 / S1.f32 == DENORM.f32', 'isDENORM(S2.f32 / S1.f32)')
    pcode = pcode.replace('1.0 / 64\'F(S1.f32) == DENORM.f64', 'isDENORM(1.0 / 64\'F(S1.f32))')
    pcode = pcode.replace('S1.f32 == DENORM.f32', 'isDENORM(S1.f32)')
    # Fix 1: Set VCC=1 when returning NAN for zero inputs
    pcode = pcode.replace('D0.f32 = NAN.f32', 'VCC = 0x1LL;\nD0.f32 = NAN.f32')
    # Fix 2: Remove the S1==DENORM branch (it's wrong), handle at end
    pcode = pcode.replace('elsif isDENORM(S1.f32) then\nD0.f32 = ldexp(S0.f32, 64)',
                          'elsif 1 == 0 then\nD0.f32 = S0.f32')
    # Fix 3: Set VCC=1 for tiny numerator case
    pcode = pcode.replace('elsif exponent(S2.f32) <= 23 then\n// Numerator is tiny\nD0.f32 = ldexp(S0.f32, 64)',
                          'elsif exponent(S2.f32) <= 23 then\nVCC = 0x1LL;\nD0.f32 = ldexp(S0.f32, 64)')
    # Fix 4: Simplify S2/S1==DENORM case (just set VCC, don't check S0==S2)
    pcode = pcode.replace('elsif isDENORM(S2.f32 / S1.f32) then\nVCC = 0x1LL;\nif S0.f32 == S2.f32 then\n// Only scale the numerator\nD0.f32 = ldexp(S0.f32, 64)\nendif',
                          'elsif isDENORM(S2.f32 / S1.f32) then\nVCC = 0x1LL;\nD0.f32 = S0.f32')
    # Fix 5: Add else to nested ifs that don't have D0 assignment
    pcode = pcode.replace('D0.f32 = ldexp(S0.f32, 64)\nendif\nelsif', 'D0.f32 = ldexp(S0.f32, 64)\nelse\nD0.f32 = S0.f32\nendif\nelsif')
    # Fix 6: Add else clause to outermost if before final endif, and check for S1==DENORM at end
    lines = pcode.rstrip().split('\n')
    for i in range(len(lines) - 1, -1, -1):
      if lines[i].strip() == 'endif':
        lines.insert(i, 'else\nD0.f32 = S0.f32')
        break
    pcode = '\n'.join(lines) + ';\nif isDENORM(S1.f32) then\nD0.f32 = NAN.f32\nendif'
  if op_name == 'V_DIV_SCALE_F64':
    pcode = pcode.replace('S2.f64 / S1.f64 == DENORM.f64', 'isDENORM(S2.f64 / S1.f64)')
    pcode = pcode.replace('1.0 / S1.f64 == DENORM.f64', 'isDENORM(1.0 / S1.f64)')
    pcode = pcode.replace('S1.f64 == DENORM.f64', 'isDENORM(S1.f64)')
    pcode = pcode.replace('D0.f64 = NAN.f64', 'VCC = 0x1LL;\nD0.f64 = NAN.f64')
    pcode = pcode.replace('elsif isDENORM(S1.f64) then\nD0.f64 = ldexp(S0.f64, 128)',
                          'elsif 1 == 0 then\nD0.f64 = S0.f64')
    pcode = pcode.replace('elsif exponent(S2.f64) <= 52 then\n// Numerator is tiny\nD0.f64 = ldexp(S0.f64, 128)',
                          'elsif exponent(S2.f64) <= 52 then\nVCC = 0x1LL;\nD0.f64 = ldexp(S0.f64, 128)')
    pcode = pcode.replace('elsif isDENORM(S2.f64 / S1.f64) then\nVCC = 0x1LL;\nif S0.f64 == S2.f64 then\n// Only scale the numerator\nD0.f64 = ldexp(S0.f64, 128)\nendif',
                          'elsif isDENORM(S2.f64 / S1.f64) then\nVCC = 0x1LL;\nD0.f64 = S0.f64')
    pcode = pcode.replace('D0.f64 = ldexp(S0.f64, 128)\nendif\nelsif', 'D0.f64 = ldexp(S0.f64, 128)\nelse\nD0.f64 = S0.f64\nendif\nelsif')
    lines = pcode.rstrip().split('\n')
    for i in range(len(lines) - 1, -1, -1):
      if lines[i].strip() == 'endif':
        lines.insert(i, 'else\nD0.f64 = S0.f64')
        break
    pcode = '\n'.join(lines) + ';\nif isDENORM(S1.f64) then\nD0.f64 = NAN.f64\nendif'
  if op_name == 'V_TRIG_PREOP_F64':
    # Replace the complex 1201-bit computation with a function call
    pcode = pcode.replace("result = 64'F((1201'B(2.0 / PI)[1200 : 0] << shift.u32) & 1201'0x1fffffffffffff)",
                          "result = trig_preop_result(shift)")
  return pcode

@functools.cache
def compile_uop(op_name: str, pseudocode: str):
  if op_name in _SKIP_OPS: return None
  if any(p in pseudocode for p in _PCODE_PATTERNS): return None
  if any(p in pseudocode for p in _WIDE_OUTPUT_PATTERNS): return None
  pseudocode = _apply_pseudocode_fixes(op_name, pseudocode)
  is_ds = op_name.startswith('DS_')
  mem_buf = LDS_BUF if is_ds else MEM_BUF
  sink, output_info, input_vars, mem_stores = _compile_pseudocode(pseudocode, mem_buf)
  return _make_fn(sink, output_info, input_vars, mem_stores)
