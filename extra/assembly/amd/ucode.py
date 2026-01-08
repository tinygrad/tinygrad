# UOp-based pseudocode compiler for AMD GPU instruction emulation
import functools, struct, math
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes, DType, AddrSpace
from extra.assembly.amd.pcode_parse import parse, Assign, Declare, If, For

SIGNED, FLOATS = (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64), (dtypes.float16, dtypes.float32, dtypes.float64)
MASK32, MASK64 = 0xffffffff, 0xffffffffffffffff

def _ftz32(bits: int) -> float:
  """Flush f32 denormals to zero (RDNA3 default mode)."""
  bits = bits & MASK32
  return 0.0 if (bits & 0x7f800000) == 0 and (bits & 0x007fffff) != 0 else struct.unpack('<f', struct.pack('<I', bits))[0]

def _cast(x: UOp, dtype: DType) -> UOp:
  return x if x.dtype == dtype else UOp(Ops.BITCAST if dtype.itemsize == x.dtype.itemsize else Ops.CAST, dtype, (x,))

# Input variables
def _var(name, dt, lo=0, hi=None): return UOp(Ops.DEFINE_VAR, dt, (), (name, lo, hi if hi else (1 << dt.itemsize*8) - 1))
INPUT_VARS = {n: _var(n, dt) for n, dt in [
  ('S0', dtypes.uint32), ('S1', dtypes.uint32), ('S2', dtypes.uint32), ('D0', dtypes.uint32),
  ('S0_64', dtypes.uint64), ('S1_64', dtypes.uint64), ('S2_64', dtypes.uint64), ('D0_64', dtypes.uint64),
  ('SCC', dtypes.uint32), ('VCC', dtypes.uint64), ('EXEC', dtypes.uint64), ('laneId', dtypes.uint32),
  ('SIMM16', dtypes.int32), ('SIMM32', dtypes.uint32), ('PC', dtypes.uint64),
  ('ADDR', dtypes.uint64), ('SDATA', dtypes.uint64), ('VDATA', dtypes.uint64), ('VDST', dtypes.uint32),
  ('RETURN_DATA', dtypes.uint64), ('DATA', dtypes.uint64), ('DATA2', dtypes.uint64),
  ('OFFSET', dtypes.uint32), ('OFFSET0', dtypes.uint32), ('OFFSET1', dtypes.uint32),
  ('OPSEL', dtypes.uint32), ('OPSEL_HI', dtypes.uint32), ('SRC0', dtypes.uint32), ('M0', dtypes.uint32),
]}
INPUT_VARS['ADDR_BASE'] = INPUT_VARS['ADDR']

MEM_BUF = UOp(Ops.DEFINE_GLOBAL, dtypes.uint8.ptr(addrspace=AddrSpace.GLOBAL), arg=0)
LDS_BUF = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(addrspace=AddrSpace.LOCAL), arg=0)

# Float bit layout: (uint_type, sign_shift, exp_shift, exp_mask, mantissa_mask, bias)
FP_INFO = {
  dtypes.float64: (dtypes.uint64, 63, 52, 0x7ff, 0xfffffffffffff, 1023),
  dtypes.float32: (dtypes.uint32, 31, 23, 0xff, 0x7fffff, 127),
  dtypes.float16: (dtypes.uint16, 15, 10, 0x1f, 0x3ff, 15),
}

class Ctx:
  def __init__(self, mem_buf: UOp = MEM_BUF):
    self.vars, self.decls, self.outputs, self.mem_stores, self.mem_buf = dict(INPUT_VARS), {}, [], [], mem_buf

# ═══════════════════════════════════════════════════════════════════════════════
# EXPRESSION TRANSFORM
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_special_var(name: str, ctx: Ctx, hint: DType = None) -> UOp | None:
  """Resolve special variables and constants."""
  if name == 'PI': return UOp.const(hint or dtypes.float64, math.pi)
  if name == 'MAX_FLOAT_F32': return UOp.const(dtypes.float32, 3.402823466e+38)
  if name in ('OVERFLOW_F32', 'UNDERFLOW_F32'): return UOp.const(dtypes.float32, float('inf') if 'OVER' in name else 0.0)
  if name in ('OVERFLOW_F64', 'UNDERFLOW_F64'): return UOp.const(dtypes.float64, float('inf') if 'OVER' in name else 0.0)
  if name == 'NAN.f32': return UOp.const(dtypes.float32, float('nan'))
  if name.startswith('DENORM.'): return UOp.const(dtypes.float64 if '64' in name else dtypes.float32, 2.2250738585072014e-308 if '64' in name else 1.17549435e-38)
  if name in ('WAVE_MODE.IEEE', 'WAVE32'): return UOp.const(dtypes.uint32, 1)
  if name in ('WAVE64', 'ROUND_MODE') or name.startswith('WAVE_STATUS.COND_DBG'): return UOp.const(dtypes.uint32, 0)
  if 'INF' in name and name.replace('+', '').replace('-', '').replace('.f16', '').replace('.f32', '').replace('.f64', '') == 'INF':
    dt = dtypes.float16 if '.f16' in name else dtypes.float32 if '.f32' in name else hint or dtypes.float64
    return UOp.const(dt, float('-inf') if name.startswith('-') else float('inf'))
  # Register aliases
  if name in ('VCCZ', 'EXECZ'):
    return _cast(UOp(Ops.CMPEQ, dtypes.bool, (ctx.vars['VCC' if 'VCC' in name else 'EXEC'], UOp.const(dtypes.uint64, 0))), dtypes.uint32)
  if name in ('EXEC_LO', 'VCC_LO'):
    return _cast(UOp(Ops.AND, dtypes.uint64, (ctx.vars['EXEC' if 'EXEC' in name else 'VCC'], UOp.const(dtypes.uint64, MASK32))), hint or dtypes.uint32)
  if name in ('EXEC_HI', 'VCC_HI'):
    return _cast(UOp(Ops.SHR, dtypes.uint64, (ctx.vars['EXEC' if 'EXEC' in name else 'VCC'], UOp.const(dtypes.uint64, 32))), hint or dtypes.uint32)
  if name in ('laneID', 'laneId'): return ctx.vars.get('laneId', UOp.const(dtypes.uint32, 0))
  if name == 'ThreadMask': return _cast(ctx.vars.get('EXEC'), hint or dtypes.uint32)
  if name == 'DST': return ctx.vars.get('VDST', UOp.const(dtypes.uint32, 0))
  if name == 'LDS': return UOp.const(dtypes.uint64, 0)
  return None

def _expr(node: UOp, ctx: Ctx, hint: DType = None) -> UOp:
  """Transform parsed UOp expression to resolved UOp."""
  match node:
    case UOp(Ops.CONST, dt, _, val):
      dt = dt if dt != dtypes.int32 or hint is None else hint
      return UOp.const(dtypes.float32 if isinstance(val, float) and dt not in FLOATS else dt, val)

    case UOp(Ops.DEFINE_VAR, _, _, (name, None, None)):
      if (resolved := _resolve_special_var(name, ctx, hint)) is not None: return resolved
      if name.startswith('eval '): return ctx.vars.get('_eval', UOp.const(dtypes.uint32, 0))
      if name not in ctx.vars: raise ValueError(f"Unknown variable: {name}")
      return _cast(ctx.vars[name], hint or ctx.vars[name].dtype)

    case UOp(Ops.BITCAST, dt, (inner,)):
      # Memory load: MEM[addr].type
      if inner.op == Ops.CUSTOM and inner.arg == 'MEM':
        addr = _expr(inner.src[0], ctx, dtypes.uint64)
        idx = UOp(Ops.INDEX, dt.ptr(0, ctx.mem_buf.dtype.addrspace), (ctx.mem_buf, addr))
        return UOp(Ops.LOAD, dt, (idx,))
      # Typed variable access: Var.type
      if inner.op == Ops.DEFINE_VAR and inner.arg[1] is None:
        name = inner.arg[0]
        if name in ('INF', '+INF', '-INF'): return UOp.const(dt, float('-inf') if '-' in name else float('inf'))
        if name == 'NAN': return UOp.const(dt, float('nan'))
        if name == 'DENORM': return UOp.const(dt, FP_INFO.get(dt, FP_INFO[dtypes.float32])[4] * 2**(-FP_INFO.get(dt, FP_INFO[dtypes.float32])[5]))
        if (resolved := _resolve_special_var(name, ctx, dt)) is not None: return _cast(resolved, dt)
        vn = name + '_64' if dt.itemsize == 8 and name.isupper() else name
        base = ctx.vars.get(vn) if vn in ctx.vars else ctx.vars.get(name)
        if base is None: raise ValueError(f"Unknown variable: {name}")
        if dt.itemsize == 3 and 'int' in dt.name:
          masked = UOp(Ops.AND, dtypes.uint32, (base, UOp.const(dtypes.uint32, 0xffffff)))
          return masked if 'uint' in dt.name else UOp(Ops.SUB, dtypes.int32, (UOp(Ops.XOR, dtypes.int32, (masked, UOp.const(dtypes.int32, 0x800000))), UOp.const(dtypes.int32, 0x800000)))
        if dt == dtypes.float16: return UOp(Ops.BITCAST, dtypes.float16, (UOp(Ops.AND, dtypes.uint16, (_cast(base, dtypes.uint16), UOp.const(dtypes.uint16, 0xffff))),))
        if dt in FLOATS: return UOp(Ops.BITCAST, dt, (base,))
        if dt in SIGNED: return _cast(ctx.vars.get(name + '_64', base) if dt == dtypes.int64 else base, dt)
        return _cast(base, dt)
      inner_resolved = _expr(inner, ctx, dt)
      if dt in (dtypes.float16, dtypes.bfloat16): return UOp(Ops.BITCAST, dt, (_cast(inner_resolved, dtypes.uint16),))
      if dt in FLOATS: return UOp(Ops.BITCAST, dt, (inner_resolved,))
      return _cast(inner_resolved, dt)

    case UOp(Ops.CUSTOMI, _, (base_expr, hi_expr, lo_expr)):  # Slice or array access
      # VGPR[lane][reg] read
      if base_expr.op == Ops.CUSTOMI and hi_expr is lo_expr:
        inner_base, inner_idx, _ = base_expr.src
        if inner_base.op == Ops.DEFINE_VAR and inner_base.arg[0] == 'VGPR':
          lane, reg = _expr(inner_idx, ctx, dtypes.uint32), _expr(hi_expr, ctx, dtypes.uint32)
          return UOp(Ops.CUSTOM, dtypes.uint32, (UOp(Ops.ADD, dtypes.uint32, (UOp(Ops.MUL, dtypes.uint32, (lane, UOp.const(dtypes.uint32, 256))), reg)),), arg='vgpr_read')
      # SGPR[idx] read
      if base_expr.op == Ops.DEFINE_VAR and base_expr.arg[0] == 'SGPR' and hi_expr is lo_expr:
        return UOp(Ops.CUSTOM, dtypes.uint32, (_expr(hi_expr, ctx, dtypes.uint32),), arg='sgpr_read')
      # Array element access
      if base_expr.op == Ops.DEFINE_VAR and base_expr.arg[1] is None and hi_expr is lo_expr:
        name, var_dtype = base_expr.arg[0], ctx.decls.get(base_expr.arg[0])
        if var_dtype is not None and var_dtype.count > 1:
          idx_uop = _expr(hi_expr, ctx).simplify()
          if idx_uop.op == Ops.CONST:
            return ctx.vars.get(f"{name}_{int(idx_uop.arg)}", UOp.const(var_dtype.scalar(), 0))
      base, hi_uop, lo_uop = _expr(base_expr, ctx), _expr(hi_expr, ctx).simplify(), _expr(lo_expr, ctx).simplify()
      # Single bit: base[idx]
      if hi_expr is lo_expr:
        return UOp(Ops.AND, dtypes.uint32, (_cast(UOp(Ops.SHR, base.dtype, (base, _cast(lo_uop, base.dtype))), dtypes.uint32), UOp.const(dtypes.uint32, 1)))
      # Bit slice: base[hi:lo]
      if hi_uop.op == Ops.CONST and lo_uop.op == Ops.CONST:
        hi_val, lo_val = int(hi_uop.arg), int(lo_uop.arg)
        if hi_val < lo_val:  # Reversed slice - bit reverse
          width = lo_val - hi_val + 1
          result_dt = dtypes.uint64 if width == 64 else dtypes.uint32
          result = UOp.const(result_dt, 0)
          for i in range(width):
            bit = UOp(Ops.AND, result_dt, (UOp(Ops.SHR, result_dt, (_cast(base, result_dt), UOp.const(result_dt, i))), UOp.const(result_dt, 1)))
            result = UOp(Ops.OR, result_dt, (result, UOp(Ops.SHL, result_dt, (bit, UOp.const(result_dt, width - 1 - i)))))
          return result
        shifted = UOp(Ops.SHR, base.dtype, (base, UOp.const(base.dtype, lo_val))) if lo_val else base
        return UOp(Ops.AND, dtypes.uint32, (_cast(shifted, dtypes.uint32), UOp.const(dtypes.uint32, (1 << (hi_val - lo_val + 1)) - 1)))
      raise ValueError(f"Non-constant slice bounds: {node}")

    case UOp(Ops.CAST, dt, (inner,)):
      inner_resolved = _expr(inner, ctx, dt)
      if dt in FLOATS and inner_resolved.op == Ops.CONST and inner_resolved.dtype not in FLOATS:
        return UOp(Ops.BITCAST, dt, (inner_resolved,))
      if inner_resolved.dtype.itemsize == dt.itemsize: return _cast(inner_resolved, dt)
      return UOp(Ops.CAST, dt, (inner_resolved,))

    case UOp(Ops.NEG, _, (src,)): val = _expr(src, ctx, hint); return UOp(Ops.NEG, val.dtype, (val,))
    case UOp(Ops.XOR, _, (src,)) if len(node.src) == 1: val = _expr(src, ctx, hint); return UOp(Ops.XOR, val.dtype, (val, UOp.const(val.dtype, -1)))
    case UOp(Ops.CMPEQ, _, (src,)) if len(node.src) == 1: val = _expr(src, ctx, hint); return UOp(Ops.CMPEQ, dtypes.bool, (val, UOp.const(val.dtype, 0)))

    # Unary math ops
    case UOp(op, _, (src,)) if op in (Ops.TRUNC, Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.RECIPROCAL):
      val = _expr(src, ctx, hint); return UOp(op, val.dtype, (val,))

    # MULACC (fma)
    case UOp(Ops.MULACC, _, (a, b, c)): return UOp(Ops.MULACC, _expr(c, ctx, hint).dtype, (_expr(a, ctx, hint), _expr(b, ctx, hint), _expr(c, ctx, hint)))

    case UOp(Ops.WHERE, _, (cond, tv, fv)):
      c, t = _expr(cond, ctx), _expr(tv, ctx, hint)
      return UOp(Ops.WHERE, t.dtype, (c, t, _expr(fv, ctx, t.dtype)))

    case UOp(op, _, (l_expr, r_expr)) if op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.AND, Ops.OR, Ops.XOR, Ops.SHL, Ops.SHR,
                                                 Ops.CMPLT, Ops.CMPLE, Ops.CMPEQ, Ops.CMPNE, Ops.POW, Ops.MOD):
      l, r = _expr(l_expr, ctx, hint), _expr(r_expr, ctx, hint)
      if op in (Ops.ADD, Ops.SUB, Ops.MUL) and l.dtype in (dtypes.int32, dtypes.int64):
        udt = {dtypes.int32: dtypes.uint32, dtypes.int64: dtypes.uint64}[l.dtype]
        return UOp(op, udt, (_cast(l, udt), _cast(r, udt)))
      if op in (Ops.CMPLT, Ops.CMPLE, Ops.CMPEQ, Ops.CMPNE): return UOp(op, dtypes.bool, (l, r))
      result_dt = l.dtype if l.dtype in FLOATS else r.dtype if r.dtype in FLOATS else l.dtype
      if op is Ops.POW:
        if l.op == Ops.CONST and l.arg == 2.0: return UOp(Ops.EXP2, result_dt, (UOp(Ops.CAST, result_dt, (r,)) if r.dtype != result_dt else r,))
        return UOp(Ops.EXP2, result_dt, (UOp(Ops.MUL, result_dt, (UOp(Ops.CAST, result_dt, (r,)), UOp(Ops.LOG2, result_dt, (l,)))),))
      if op is Ops.MOD: return UOp(Ops.SUB, result_dt, (l, UOp(Ops.MUL, result_dt, (UOp(Ops.IDIV, result_dt, (l, r)), r))))
      return UOp(op, result_dt, (l, r))

    case UOp(Ops.CUSTOM, _, args, name): return _transform_call(name, [_expr(a, ctx, hint) for a in args], hint)

    case UOp(Ops.CAT, _, exprs):  # Pack {hi, lo}
      hi, lo = _expr(exprs[0], ctx), _expr(exprs[1], ctx)
      if lo.dtype.itemsize >= 4:
        return UOp(Ops.OR, dtypes.uint64, (UOp(Ops.SHL, dtypes.uint64, (_cast(hi, dtypes.uint64), UOp.const(dtypes.uint64, 32))), _cast(lo, dtypes.uint64)))
      return UOp(Ops.OR, dtypes.uint32, (UOp(Ops.SHL, dtypes.uint32, (_cast(hi, dtypes.uint32), UOp.const(dtypes.uint32, 16))),
                                         UOp(Ops.AND, dtypes.uint32, (_cast(lo, dtypes.uint32), UOp.const(dtypes.uint32, 0xffff)))))
  raise ValueError(f"Cannot transform expression: {node}")

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTION CALLS
# ═══════════════════════════════════════════════════════════════════════════════

CVT_MAP = {'u32_to_f32': (dtypes.float32, False), 'i32_to_f32': (dtypes.float32, False), 'f32_to_u32': (dtypes.uint32, True),
           'f32_to_i32': (dtypes.int32, False), 'f16_to_f32': (dtypes.float32, False), 'f32_to_f16': (dtypes.float16, False),
           'f32_to_u8': (dtypes.uint8, False), 'f32_to_i8': (dtypes.int8, False), 'f32_to_u16': (dtypes.uint16, False),
           'f32_to_i16': (dtypes.int16, False), 'v_cvt_u16_f32': (dtypes.uint16, False), 'v_cvt_i16_f32': (dtypes.int16, False),
           'f64_to_i32': (dtypes.int32, False), 'f64_to_u32': (dtypes.uint32, True), 'i32_to_f64': (dtypes.float64, False),
           'u32_to_f64': (dtypes.float64, False), 'f64_to_f32': (dtypes.float32, False), 'f32_to_f64': (dtypes.float64, False),
           'u16_to_f16': (dtypes.float16, False), 'i16_to_f16': (dtypes.float16, False), 'f16_to_u16': (dtypes.uint16, False), 'f16_to_i16': (dtypes.int16, False)}

def _fp_bits(v: UOp) -> tuple[UOp, int, int, int]:
  """Get float as bits with its layout info. Unwraps CAST to check original float type."""
  # For NaN checking, we need to use the original float's bit layout (not the casted one)
  # because Python's float cast doesn't preserve signaling vs quiet NaN
  while v.op == Ops.CAST and v.src[0].dtype in FLOATS: v = v.src[0]
  uint_dt, _, exp_shift, exp_mask, mant_mask, _ = FP_INFO.get(v.dtype, FP_INFO[dtypes.float32])
  return UOp(Ops.BITCAST, uint_dt, (v,)), exp_shift, exp_mask, mant_mask

def _minmax(args: list[UOp], is_min: bool) -> UOp:
  """Build min/max expression for 2 or 3 arguments."""
  cmp = lambda x, y: UOp(Ops.CMPLT, dtypes.bool, (x, y) if is_min else (y, x))
  result = UOp(Ops.WHERE, args[0].dtype, (cmp(args[0], args[1]), args[0], args[1]))
  return UOp(Ops.WHERE, args[0].dtype, (cmp(result, args[2]), result, args[2])) if len(args) > 2 else result

def _transform_call(name: str, a: list[UOp], hint: DType) -> UOp:
  if name == 'MEM': return a[0]
  if name == 'abs': return UOp(Ops.WHERE, a[0].dtype, (UOp(Ops.CMPLT, dtypes.bool, (a[0], UOp.const(a[0].dtype, 0))), UOp(Ops.NEG, a[0].dtype, (a[0],)), a[0]))
  if name == 'cos': return UOp(Ops.SIN, a[0].dtype, (UOp(Ops.ADD, a[0].dtype, (a[0], UOp.const(a[0].dtype, 1.5707963267948966))),))
  if name == 'rsqrt': return UOp(Ops.RECIPROCAL, a[0].dtype, (UOp(Ops.SQRT, a[0].dtype, (a[0],)),))
  if name == 'floor':
    trunc = UOp(Ops.TRUNC, a[0].dtype, (a[0],))
    return UOp(Ops.WHERE, a[0].dtype, (UOp(Ops.CMPLT, dtypes.bool, (a[0], trunc)), UOp(Ops.SUB, a[0].dtype, (trunc, UOp.const(a[0].dtype, 1))), trunc))
  if name == 'fract': return UOp(Ops.SUB, a[0].dtype, (a[0], _transform_call('floor', a, hint)))
  if name == 'clamp':
    c = UOp(Ops.WHERE, a[0].dtype, (UOp(Ops.CMPLT, dtypes.bool, (a[0], a[1])), a[1], a[0]))
    return UOp(Ops.WHERE, a[0].dtype, (UOp(Ops.CMPLT, dtypes.bool, (a[2], c)), a[2], c))
  if name == 'isNAN': return UOp(Ops.CMPNE, dtypes.bool, (a[0], a[0]))
  if name == 'isINF': return UOp(Ops.OR, dtypes.bool, (UOp(Ops.CMPEQ, dtypes.bool, (a[0], UOp.const(a[0].dtype, float('inf')))),
                                                        UOp(Ops.CMPEQ, dtypes.bool, (a[0], UOp.const(a[0].dtype, float('-inf'))))))
  if name in ('isQuietNAN', 'isSignalNAN'):
    bits, exp_shift, exp_mask, mant_mask = _fp_bits(a[0])
    # Use the dtype from bits (uint32/uint64/uint16) to determine which quiet bit to use
    float_dt = {dtypes.uint64: dtypes.float64, dtypes.uint32: dtypes.float32, dtypes.uint16: dtypes.float16}.get(bits.dtype, dtypes.float32)
    quiet_bit = {dtypes.float64: 0x8000000000000, dtypes.float32: 0x400000, dtypes.float16: 0x200}.get(float_dt, 0x400000)
    exp = UOp(Ops.AND, bits.dtype, (UOp(Ops.SHR, bits.dtype, (bits, UOp.const(bits.dtype, exp_shift))), UOp.const(bits.dtype, exp_mask)))
    is_exp_all = UOp(Ops.CMPEQ, dtypes.bool, (exp, UOp.const(bits.dtype, exp_mask)))
    quiet_check = UOp(Ops.AND, bits.dtype, (bits, UOp.const(bits.dtype, quiet_bit)))
    if name == 'isQuietNAN': return UOp(Ops.AND, dtypes.bool, (is_exp_all, UOp(Ops.CMPNE, dtypes.bool, (quiet_check, UOp.const(bits.dtype, 0)))))
    mant = UOp(Ops.AND, bits.dtype, (bits, UOp.const(bits.dtype, mant_mask)))
    return UOp(Ops.AND, dtypes.bool, (UOp(Ops.AND, dtypes.bool, (is_exp_all, UOp(Ops.CMPNE, dtypes.bool, (mant, UOp.const(bits.dtype, 0))))),
                                      UOp(Ops.CMPEQ, dtypes.bool, (quiet_check, UOp.const(bits.dtype, 0)))))
  if name == 'isDENORM':
    bits, exp_shift, exp_mask, mant_mask = _fp_bits(a[0])
    exp = UOp(Ops.AND, bits.dtype, (UOp(Ops.SHR, bits.dtype, (bits, UOp.const(bits.dtype, exp_shift))), UOp.const(bits.dtype, exp_mask)))
    mant = UOp(Ops.AND, bits.dtype, (bits, UOp.const(bits.dtype, mant_mask)))
    return UOp(Ops.AND, dtypes.bool, (UOp(Ops.CMPEQ, dtypes.bool, (exp, UOp.const(bits.dtype, 0))), UOp(Ops.CMPNE, dtypes.bool, (mant, UOp.const(bits.dtype, 0)))))
  if name == 'sign':
    uint_dt, sign_shift, _, _, _, _ = FP_INFO.get(a[0].dtype, FP_INFO[dtypes.float32])
    return UOp(Ops.AND, dtypes.uint32, (_cast(UOp(Ops.SHR, uint_dt, (UOp(Ops.BITCAST, uint_dt, (a[0],)), UOp.const(uint_dt, sign_shift))), dtypes.uint32), UOp.const(dtypes.uint32, 1)))
  if name == 'exponent':
    bits, exp_shift, exp_mask, _ = _fp_bits(a[0])
    return UOp(Ops.AND, dtypes.uint32, (_cast(UOp(Ops.SHR, bits.dtype, (bits, UOp.const(bits.dtype, exp_shift))), dtypes.uint32), UOp.const(dtypes.uint32, exp_mask)))
  if name == 'mantissa':
    uint_dt, sign_shift, exp_shift, _, mant_mask, bias = FP_INFO.get(a[0].dtype, FP_INFO[dtypes.float32])
    bits = UOp(Ops.BITCAST, uint_dt, (a[0],))
    result = UOp(Ops.BITCAST, a[0].dtype, (UOp(Ops.OR, uint_dt, (UOp(Ops.AND, uint_dt, (bits, UOp.const(uint_dt, (1 << sign_shift) | mant_mask))),
                                                                  UOp.const(uint_dt, (bias - 1) << exp_shift))),))
    return UOp(Ops.WHERE, a[0].dtype, (UOp(Ops.CMPEQ, dtypes.bool, (a[0], UOp.const(a[0].dtype, 0.0))), a[0], result))
  if name == 'cvtToQuietNAN': return a[0]
  if name == 'isEven':
    return UOp(Ops.CMPEQ, dtypes.bool, (UOp(Ops.AND, dtypes.int64, (UOp(Ops.CAST, dtypes.int64, (a[0],)), UOp.const(dtypes.int64, 1))), UOp.const(dtypes.int64, 0)))
  if name == 'signext': return _cast(a[0], dtypes.int64)
  if name == 'signext_from_bit':
    sign = UOp(Ops.SHL, a[0].dtype, (UOp.const(a[0].dtype, 1), UOp(Ops.SUB, a[0].dtype, (_cast(a[1], a[0].dtype), UOp.const(a[0].dtype, 1)))))
    result = UOp(Ops.SUB, a[0].dtype, (UOp(Ops.XOR, a[0].dtype, (a[0], sign)), sign))
    return UOp(Ops.WHERE, a[0].dtype, (UOp(Ops.CMPEQ, dtypes.bool, (a[1], UOp.const(a[1].dtype, 0))), UOp.const(a[0].dtype, 0), result))
  if name == 'ABSDIFF':
    gt = UOp(Ops.CMPLT, dtypes.bool, (a[1], a[0]))
    return UOp(Ops.SUB, dtypes.uint32, (UOp(Ops.WHERE, dtypes.uint32, (gt, _cast(a[0], dtypes.uint32), _cast(a[1], dtypes.uint32))),
                                        UOp(Ops.WHERE, dtypes.uint32, (gt, _cast(a[1], dtypes.uint32), _cast(a[0], dtypes.uint32)))))
  if name == 'SAT8':
    c = UOp(Ops.WHERE, a[0].dtype, (UOp(Ops.CMPLT, dtypes.bool, (a[0], UOp.const(a[0].dtype, -128))), UOp.const(a[0].dtype, -128), a[0]))
    return UOp(Ops.WHERE, a[0].dtype, (UOp(Ops.CMPLT, dtypes.bool, (UOp.const(a[0].dtype, 127), c)), UOp.const(a[0].dtype, 127), c))
  if name == 'bf16_to_f32': return UOp(Ops.BITCAST, dtypes.float32, (UOp(Ops.SHL, dtypes.uint32, (_cast(a[0], dtypes.uint32), UOp.const(dtypes.uint32, 16))),))
  if name == 'BYTE_PERMUTE':
    src64, sel = _cast(a[0], dtypes.uint64), UOp(Ops.AND, dtypes.uint32, (_cast(a[1], dtypes.uint32), UOp.const(dtypes.uint32, 0xff)))
    sel_idx, sel_nib = UOp(Ops.AND, dtypes.uint32, (sel, UOp.const(dtypes.uint32, 7))), UOp(Ops.AND, dtypes.uint32, (sel, UOp.const(dtypes.uint32, 0xf)))
    byte_val = _cast(UOp(Ops.AND, dtypes.uint64, (UOp(Ops.SHR, dtypes.uint64, (src64, _cast(UOp(Ops.SHL, dtypes.uint32, (sel_idx, UOp.const(dtypes.uint32, 3))), dtypes.uint64))), UOp.const(dtypes.uint64, 0xff))), dtypes.uint32)
    def sign_bit(pos): return UOp(Ops.WHERE, dtypes.uint32, (UOp(Ops.CMPNE, dtypes.bool, (UOp(Ops.AND, dtypes.uint64, (UOp(Ops.SHR, dtypes.uint64, (src64, UOp.const(dtypes.uint64, pos))), UOp.const(dtypes.uint64, 1))), UOp.const(dtypes.uint64, 0))), UOp.const(dtypes.uint32, 0xff), UOp.const(dtypes.uint32, 0)))
    result = byte_val
    for i, pos in enumerate([15, 31, 47, 63], 8):
      result = UOp(Ops.WHERE, dtypes.uint32, (UOp(Ops.CMPEQ, dtypes.bool, (sel_nib, UOp.const(dtypes.uint32, i))), sign_bit(pos), result))
    result = UOp(Ops.WHERE, dtypes.uint32, (UOp(Ops.CMPEQ, dtypes.bool, (sel_nib, UOp.const(dtypes.uint32, 12))), UOp.const(dtypes.uint32, 0), result))
    result = UOp(Ops.WHERE, dtypes.uint32, (UOp(Ops.CMPLT, dtypes.bool, (UOp.const(dtypes.uint32, 12), sel_nib)), UOp.const(dtypes.uint32, 0xff), result))
    return UOp(Ops.WHERE, dtypes.uint32, (UOp(Ops.CMPNE, dtypes.bool, (UOp(Ops.AND, dtypes.uint32, (sel, UOp.const(dtypes.uint32, 0x80))), UOp.const(dtypes.uint32, 0))), UOp.const(dtypes.uint32, 0), result))
  if name == 'trig_preop_result': return UOp(Ops.CUSTOM, dtypes.float64, (a[0],), arg='trig_preop_result')
  if name == 's_ff1_i32_b32': return UOp(Ops.CUSTOM, dtypes.int32, (_cast(a[0], dtypes.uint32),), arg='s_ff1_i32_b32')
  if name == 's_ff1_i32_b64': return UOp(Ops.CUSTOM, dtypes.int32, (_cast(a[0], dtypes.uint64),), arg='s_ff1_i32_b64')
  if name in ('u8_to_u32', 'u4_to_u32'):
    mask = 0xff if '8' in name else 0xf
    return UOp(Ops.AND, dtypes.uint32, (_cast(a[0], dtypes.uint32), UOp.const(dtypes.uint32, mask)))
  if name == 'pow':
    assert a[0].op == Ops.CONST and a[0].arg == 2.0
    return UOp(Ops.EXP2, a[0].dtype, (a[1] if a[1].dtype == a[0].dtype else UOp(Ops.CAST, a[0].dtype, (a[1],)),))
  if name == 'ldexp': return UOp(Ops.MUL, a[0].dtype, (a[0], UOp(Ops.EXP2, a[0].dtype, (UOp(Ops.CAST, a[0].dtype, (a[1],)),))))
  if name in ('min', 'max'): return _minmax(a, is_min=(name == 'min'))
  if name in CVT_MAP:
    dt, clamp = CVT_MAP[name]
    v = UOp(Ops.WHERE, a[0].dtype, (UOp(Ops.CMPLT, dtypes.bool, (a[0], UOp.const(a[0].dtype, 0.0))), UOp.const(a[0].dtype, 0.0), a[0])) if clamp else a[0]
    return UOp(Ops.CAST, dt, (v,))
  if 'snorm' in name or 'unorm' in name:
    lo, scale, out = (-1.0, 32767.0, dtypes.int16) if 'snorm' in name else (0.0, 65535.0, dtypes.uint16)
    c = UOp(Ops.WHERE, a[0].dtype, (UOp(Ops.CMPLT, dtypes.bool, (a[0], UOp.const(a[0].dtype, lo))), UOp.const(a[0].dtype, lo), a[0]))
    c = UOp(Ops.WHERE, a[0].dtype, (UOp(Ops.CMPLT, dtypes.bool, (UOp.const(a[0].dtype, 1.0), c)), UOp.const(a[0].dtype, 1.0), c))
    return UOp(Ops.CAST, out, (UOp(Ops.MUL, a[0].dtype, (c, UOp.const(a[0].dtype, scale))),))
  if name == 'u32_to_u16': return UOp(Ops.AND, dtypes.uint32, (a[0], UOp.const(dtypes.uint32, 0xffff)))
  if name == 'i32_to_i16': return _cast(UOp(Ops.AND, dtypes.uint32, (_cast(a[0], dtypes.uint32), UOp.const(dtypes.uint32, 0xffff))), dtypes.int16)
  if name in ('LT_NEG_ZERO', 'GT_NEG_ZERO'):
    int_dt = {dtypes.float64: dtypes.int64, dtypes.float16: dtypes.int16}.get(a[0].dtype, dtypes.int32)
    return UOp(Ops.CMPLT, dtypes.bool, ((UOp(Ops.BITCAST, int_dt, (a[0],)), UOp(Ops.BITCAST, int_dt, (a[1],))) if 'LT' in name else (UOp(Ops.BITCAST, int_dt, (a[1],)), UOp(Ops.BITCAST, int_dt, (a[0],)))))
  if name.startswith('v_min') or name.startswith('v_max'): return _minmax(a, is_min=('min' in name))
  if name in ('v_sad_u8', 'v_msad_u8'):
    result = a[2] if len(a) > 2 else UOp.const(dtypes.uint32, 0)
    for i in range(4):
      ba = UOp(Ops.AND, dtypes.uint32, (UOp(Ops.SHR, dtypes.uint32, (a[0], UOp.const(dtypes.uint32, i*8))), UOp.const(dtypes.uint32, 0xff)))
      bb = UOp(Ops.AND, dtypes.uint32, (UOp(Ops.SHR, dtypes.uint32, (a[1], UOp.const(dtypes.uint32, i*8))), UOp.const(dtypes.uint32, 0xff)))
      diff = UOp(Ops.SUB, dtypes.uint32, (ba, bb))
      result = UOp(Ops.ADD, dtypes.uint32, (result, UOp(Ops.WHERE, dtypes.uint32, (UOp(Ops.CMPLT, dtypes.bool, (diff, UOp.const(dtypes.uint32, 0x80000000))), diff, UOp(Ops.SUB, dtypes.uint32, (UOp.const(dtypes.uint32, 0), diff))))))
    return result
  raise ValueError(f"Unknown function: {name}")

# ═══════════════════════════════════════════════════════════════════════════════
# STATEMENT PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

OUT_VARS = ('D0', 'D1', 'SCC', 'VCC', 'EXEC', 'PC', 'SDATA', 'VDATA', 'RETURN_DATA')

def _get_lhs_info(lhs: UOp, ctx: Ctx) -> tuple[str, DType, int|None, int|None, str|None, int|str|None, UOp|tuple|None]:
  """Extract: (var_name, dtype, hi_bit, lo_bit, idx_var, array_idx, dynamic_idx)"""
  match lhs:
    case UOp(Ops.BITCAST, dt, (UOp(Ops.DEFINE_VAR, _, _, (name, None, None)),)): return name, dt, None, None, None, None, None
    case UOp(Ops.BITCAST, dt, (UOp(Ops.CUSTOMI, _, (UOp(Ops.DEFINE_VAR, _, _, (name, None, None)), UOp(Ops.CONST, _, _, hi), UOp(Ops.CONST, _, _, lo))),)):
      return name, dt, int(hi), int(lo), None, None, None
    case UOp(Ops.BITCAST, _, (UOp(Ops.CUSTOMI, _, (UOp(Ops.BITCAST, _, (UOp(Ops.DEFINE_VAR, _, _, (name, None, None)),)), UOp(Ops.DEFINE_VAR, _, _, (idx, None, None)), idx2)),)) if lhs.src[0].src[1] is lhs.src[0].src[2]:
      return name, dtypes.uint64, None, None, idx, None, None
    case UOp(Ops.BITCAST, dt, (UOp(Ops.CUSTOMI, _, (UOp(Ops.DEFINE_VAR, _, _, (name, None, None)), UOp(Ops.DEFINE_VAR, _, _, (idx, None, None)), idx2)),)) if lhs.src[0].src[1] is lhs.src[0].src[2]:
      return name, dt, None, None, idx, None, None
    case UOp(Ops.CUSTOMI, _, (UOp(Ops.BITCAST, _, (UOp(Ops.DEFINE_VAR, _, _, (name, None, None)),)), UOp(Ops.CONST, _, _, hi), UOp(Ops.CONST, _, _, lo))):
      return name, dtypes.uint32, int(hi), int(lo), None, None, None
    case UOp(Ops.CUSTOMI, _, (UOp(Ops.DEFINE_VAR, _, _, (name, None, None)), UOp(Ops.CONST, _, _, idx), _)) if lhs.src[1] is lhs.src[2]:
      var_dtype = ctx.decls.get(name)
      if var_dtype is not None and var_dtype.count > 1: return name, var_dtype.scalar(), None, None, None, int(idx), None
      return name, dtypes.uint32, int(idx), int(idx), None, None, None
    case UOp(Ops.CUSTOMI, _, (UOp(Ops.DEFINE_VAR, _, _, (name, None, None)), UOp(Ops.CONST, _, _, hi), UOp(Ops.CONST, _, _, lo))):
      return name, dtypes.uint32, int(hi), int(lo), None, None, None
    case UOp(Ops.CUSTOMI, _, (UOp(Ops.BITCAST, dt, (UOp(Ops.DEFINE_VAR, _, _, (name, None, None)),)), UOp(Ops.DEFINE_VAR, _, _, (idx, None, None)), idx2)) if lhs.src[1] is lhs.src[2]:
      return name, dt, None, None, idx, None, None
    case UOp(Ops.CUSTOMI, _, (UOp(Ops.DEFINE_VAR, _, _, (name, None, None)), UOp(Ops.DEFINE_VAR, _, _, (idx, None, None)), idx2)) if lhs.src[1] is lhs.src[2]:
      var_dtype = ctx.decls.get(name)
      if var_dtype is not None and var_dtype.count > 1: return name, var_dtype.scalar(), None, None, None, idx, None
      return name, dtypes.uint32, None, None, idx, None, None
    case UOp(Ops.CUSTOMI, _, (UOp(Ops.BITCAST, dt, (UOp(Ops.DEFINE_VAR, _, _, (name, None, None)),)), idx_expr, idx_expr2)) if lhs.src[1] is lhs.src[2]:
      return name, dt, None, None, None, None, idx_expr
    case UOp(Ops.CUSTOMI, _, (UOp(Ops.CUSTOMI, _, (UOp(Ops.DEFINE_VAR, _, _, ('VGPR', None, None)), lane, _)), reg, _)):
      return 'VGPR', dtypes.uint32, None, None, None, None, (lane, reg)
    case UOp(Ops.BITCAST, dt, (UOp(Ops.CUSTOMI, _, (UOp(Ops.CUSTOMI, _, (UOp(Ops.DEFINE_VAR, _, _, ('VGPR', None, None)), lane, _)), reg, _)),)):
      return 'VGPR', dt, None, None, None, None, (lane, reg)
    case UOp(Ops.BITCAST, dt, (UOp(Ops.CUSTOMI, _, (UOp(Ops.DEFINE_VAR, _, _, ('SGPR', None, None)), reg, _)),)):
      return 'SGPR', dt, None, None, None, None, reg
    case UOp(Ops.DEFINE_VAR, _, _, (name, None, None)):
      return name, ctx.vars.get(name, UOp.const(dtypes.uint32, 0)).dtype, None, None, None, None, None
  raise ValueError(f"Cannot parse LHS: {lhs}")

def _stmt(stmt, ctx: Ctx):
  match stmt:
    case Declare(name, dtype):
      ctx.decls[name] = dtype
      if name == 'S' and dtype.count == 3:
        ctx.vars['S_0'], ctx.vars['S_1'], ctx.vars['S_2'] = ctx.vars['S0'], ctx.vars['S1'], ctx.vars['S2']
      else:
        ctx.vars[name] = UOp.const(dtype, 0)

    case Assign(lhs, rhs):
      # Memory store
      if lhs.op == Ops.BITCAST and lhs.src[0].op == Ops.CUSTOM and lhs.src[0].arg == 'MEM':
        addr, val = _expr(lhs.src[0].src[0], ctx, dtypes.uint64), _expr(rhs, ctx, lhs.dtype)
        idx = UOp(Ops.INDEX, lhs.dtype.ptr(0, ctx.mem_buf.dtype.addrspace), (ctx.mem_buf, addr))
        ctx.mem_stores.append(UOp(Ops.STORE, dtypes.void, (idx, val)))
        return

      # CAT assignment: {D1.u1, D0.u64} = ...
      if lhs.op == Ops.CAT:
        rhs_uop, offset = _expr(rhs, ctx), 0
        for part in reversed(lhs.src):
          if part.op == Ops.BITCAST and part.src[0].op == Ops.DEFINE_VAR:
            dt, name = part.dtype, part.src[0].arg[0]
            bits = 1 if dt.name == 'u1' else 64 if dt == dtypes.ulong or dt.name == 'ulong' else dt.itemsize * 8
            real_dt = dtypes.uint32 if bits == 1 else dtypes.uint64 if bits == 64 else dt
            val = _cast(UOp(Ops.AND, rhs_uop.dtype, (UOp(Ops.SHR, rhs_uop.dtype, (rhs_uop, UOp.const(rhs_uop.dtype, offset))), UOp.const(rhs_uop.dtype, (1 << bits) - 1))), real_dt)
            ctx.vars[name] = val
            if name in OUT_VARS: ctx.outputs.append((name, val, real_dt))
            offset += bits
        return

      var, dtype, hi, lo, idx_var, array_idx, dynamic_idx = _get_lhs_info(lhs, ctx)

      # VGPR write
      if var == 'VGPR' and isinstance(dynamic_idx, tuple):
        lane, reg = _expr(dynamic_idx[0], ctx, dtypes.uint32), _expr(dynamic_idx[1], ctx, dtypes.uint32)
        idx = UOp(Ops.ADD, dtypes.uint32, (UOp(Ops.MUL, dtypes.uint32, (lane, UOp.const(dtypes.uint32, 256))), reg))
        ctx.outputs.append(('VGPR_WRITE', UOp(Ops.CUSTOM, dtypes.uint32, (idx, _cast(_expr(rhs, ctx, dtype), dtypes.uint32)), arg='vgpr_write'), dtypes.uint32))
        return

      # SGPR write
      if var == 'SGPR' and dynamic_idx is not None and not isinstance(dynamic_idx, tuple):
        ctx.outputs.append(('SGPR_WRITE', UOp(Ops.CUSTOM, dtypes.uint32, (_expr(dynamic_idx, ctx, dtypes.uint32), _cast(_expr(rhs, ctx, dtype), dtypes.uint32)), arg='sgpr_write'), dtypes.uint32))
        return

      # Dynamic bit index: D0.u32[expr] = value
      if dynamic_idx is not None and not isinstance(dynamic_idx, tuple):
        idx_uop, rhs_uop = _expr(dynamic_idx, ctx, dtypes.uint32), _expr(rhs, ctx, dtypes.uint32)
        op_dt = dtypes.uint64 if dtype.itemsize == 8 else dtypes.uint32
        base = _cast(ctx.vars.get(var, UOp.const(op_dt, 0)), op_dt)
        one, bit_mask = UOp.const(op_dt, 1), UOp(Ops.SHL, op_dt, (UOp.const(op_dt, 1), _cast(idx_uop, op_dt)))
        result = UOp(Ops.OR, op_dt, (UOp(Ops.AND, op_dt, (base, UOp(Ops.XOR, op_dt, (bit_mask, UOp.const(op_dt, -1))))),
                                     UOp(Ops.SHL, op_dt, (UOp(Ops.AND, op_dt, (_cast(rhs_uop, op_dt), one)), _cast(idx_uop, op_dt)))))
        ctx.vars[var] = result
        if var in OUT_VARS:
          ctx.outputs = [(n, u, d) for n, u, d in ctx.outputs if n != var]
          ctx.outputs.append((var, result, op_dt))
        return

      # Array element: arr[idx] = value
      if array_idx is not None:
        rhs_uop = _expr(rhs, ctx, dtype)
        if isinstance(array_idx, str):
          idx_uop = ctx.vars.get(array_idx)
          if idx_uop is None or idx_uop.op != Ops.CONST: raise ValueError(f"Non-constant array index: {array_idx}")
          array_idx = int(idx_uop.arg)
        ctx.vars[f"{var}_{array_idx}"] = rhs_uop
        return

      # Variable bit index: var[idx_var] = cond
      if idx_var is not None:
        base, idx, cond = ctx.vars.get(var), ctx.vars.get(idx_var), _expr(rhs, ctx)
        one, bit_mask = UOp.const(dtype, 1), UOp(Ops.SHL, dtype, (UOp.const(dtype, 1), _cast(idx, dtype)))
        result = UOp(Ops.OR, dtype, (UOp(Ops.AND, dtype, (base, UOp(Ops.XOR, dtype, (bit_mask, UOp.const(dtype, -1))))),
                                     UOp(Ops.SHL, dtype, (UOp(Ops.AND, dtype, (_cast(cond, dtype), one)), _cast(idx, dtype)))))
        ctx.vars[var] = result
        if var in OUT_VARS: ctx.outputs.append((var, result, dtype))
        return

      # Bit slice: var[hi:lo] = value
      if hi is not None and lo is not None:
        if hi < lo: hi, lo = lo, hi
        op_dt = dtypes.uint256 if hi >= 128 else dtypes.uint128 if hi >= 64 else dtypes.uint64 if hi >= 32 else dtypes.uint32
        base = _cast(ctx.vars.get(var, UOp.const(op_dt, 0)), op_dt)
        rhs_uop = _expr(rhs, ctx, dtype)
        rhs_bits = UOp(Ops.BITCAST, dtypes.uint16 if dtype == dtypes.float16 else dtypes.uint32 if dtype == dtypes.float32 else dtypes.uint64, (rhs_uop,)) if dtype in FLOATS else _cast(rhs_uop, op_dt)
        if dtype == dtypes.float16: rhs_bits = _cast(rhs_bits, op_dt)
        mask, width = (1 << (hi - lo + 1)) - 1, op_dt.itemsize * 8
        result = UOp(Ops.OR, op_dt, (UOp(Ops.AND, op_dt, (base, UOp.const(op_dt, ~(mask << lo) & ((1 << width) - 1)))),
                                     UOp(Ops.SHL, op_dt, (UOp(Ops.AND, op_dt, (rhs_bits, UOp.const(op_dt, mask))), UOp.const(op_dt, lo)))))
        ctx.vars[var] = result
        if var in OUT_VARS:
          ctx.outputs = [(n, u, d) for n, u, d in ctx.outputs if n != var]
          ctx.outputs.append((var, result, op_dt))
        return

      # Simple assignment
      rhs_uop = _expr(rhs, ctx, dtype)
      ctx.vars[var] = rhs_uop
      if dtype.itemsize == 8 and var in ('D0', 'D1', 'S0', 'S1'): ctx.vars[var + '_64'] = rhs_uop
      if var in OUT_VARS: ctx.outputs.append((var, rhs_uop, dtype))

    case If(branches): _transform_if(branches, ctx)
    case For(var, start, end, body): _transform_for(var, start, end, body, ctx)

def _transform_if(branches: tuple, ctx: Ctx):
  parsed = []
  for cond, body in branches:
    sub_ctx = Ctx(mem_buf=ctx.mem_buf)
    sub_ctx.vars, sub_ctx.decls = dict(ctx.vars), dict(ctx.decls)
    for s in body: _stmt(s, sub_ctx)
    parsed.append((_expr(cond, ctx) if cond is not None else None, sub_ctx))

  assigned = {n for _, sc in parsed for n, _, _ in sc.outputs if n in OUT_VARS}
  assigned |= {n for _, sc in parsed for n, v in sc.vars.items() if n not in ctx.vars or ctx.vars[n] is not v if n not in OUT_VARS and n not in INPUT_VARS}

  for var in assigned:
    is_out = var in OUT_VARS
    dtype = next((d for _, sc in parsed for n, _, d in sc.outputs if n == var), ctx.decls.get(var, dtypes.uint32))
    result = ctx.vars.get(var, UOp.const(dtype, 0))
    for cond_uop, sub_ctx in reversed(parsed):
      val = next((u for n, u, _ in sub_ctx.outputs if n == var), None) if is_out else sub_ctx.vars.get(var) if var in sub_ctx.vars and sub_ctx.vars[var] is not ctx.vars.get(var) else None
      if val is not None:
        result = val if cond_uop is None else UOp(Ops.WHERE, val.dtype, (cond_uop, val, _cast(result, val.dtype)))
    ctx.vars[var] = result
    if is_out:
      ctx.outputs = [(n, u, d) for n, u, d in ctx.outputs if n != var]
      ctx.outputs.append((var, result, dtype))

def _transform_for(var: str, start: UOp, end: UOp, body: tuple, ctx: Ctx):
  start_val = start.arg if start.op == Ops.CONST else int(_expr(start, ctx).arg)
  end_val = end.arg if end.op == Ops.CONST else int(_expr(end, ctx).arg)
  for i in range(int(end_val), int(start_val) - 1, -1):
    ctx.vars[var] = UOp.const(ctx.decls.get(var, dtypes.uint32), i)
    for s in body:
      if isinstance(s, If): _transform_if(s.branches, ctx)
      elif isinstance(s, Assign): _stmt(s, ctx)

# ═══════════════════════════════════════════════════════════════════════════════
# CODE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

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

TWO_OVER_PI_1201 = 0x0145f306dc9c882a53f84eafa3ea69bb81b6c52b3278872083fca2c757bd778ac36e48dc74849ba5c00c925dd413a32439fc3bd63962534e7dd1046bea5d768909d338e04d68befc827323ac7306a673e93908bf177bf250763ff12fffbc0b301fde5e2316b414da3eda6cfd9e4f96136e9e8c7ecd3cbfd45aea4f758fd7cbe2f67a0e73ef14a525d4d7f6bf623f1aba10ac06608df8f6

def _eval_uop(u: UOp) -> int|float|None:
  """Recursively evaluate a UOp tree to a constant value."""
  if u.op == Ops.CONST: return u.arg
  if u.op == Ops.CAST: return _eval_uop(u.src[0])
  if u.op == Ops.BITCAST:
    v = _eval_uop(u.src[0])
    if v is None: return None
    if u.dtype == dtypes.float64 and u.src[0].dtype in (dtypes.uint64, dtypes.int64): return struct.unpack('<d', struct.pack('<Q', int(v) & MASK64))[0]
    if u.dtype == dtypes.float32 and u.src[0].dtype in (dtypes.uint32, dtypes.int32): return _ftz32(int(v))
    if u.dtype in (dtypes.uint64, dtypes.int64) and u.src[0].dtype == dtypes.float64: return struct.unpack('<Q', struct.pack('<d', float(v)))[0]
    if u.dtype in (dtypes.uint32, dtypes.int32) and u.src[0].dtype == dtypes.float32: return struct.unpack('<I', struct.pack('<f', float(v)))[0]
    return v
  if u.op == Ops.MULACC:
    a, b, c = _eval_uop(u.src[0]), _eval_uop(u.src[1]), _eval_uop(u.src[2])
    return math.fma(float(a), float(b), float(c)) if None not in (a, b, c) else None
  if u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.AND, Ops.OR, Ops.XOR, Ops.SHR, Ops.SHL):
    l, r = _eval_uop(u.src[0]), _eval_uop(u.src[1])
    if l is None or r is None: return None
    ops = {Ops.ADD: lambda a, b: a + b, Ops.SUB: lambda a, b: a - b, Ops.MUL: lambda a, b: a * b,
           Ops.AND: lambda a, b: int(a) & int(b), Ops.OR: lambda a, b: int(a) | int(b), Ops.XOR: lambda a, b: int(a) ^ int(b),
           Ops.SHR: lambda a, b: int(a) >> int(b), Ops.SHL: lambda a, b: int(a) << int(b)}
    return ops[u.op](l, r)
  if u.op == Ops.NEG: v = _eval_uop(u.src[0]); return -v if v is not None else None
  if u.op in (Ops.CMPEQ, Ops.CMPNE, Ops.CMPLT, Ops.CMPLE):
    l, r = _eval_uop(u.src[0]), _eval_uop(u.src[1])
    if l is None or r is None: return None
    return {Ops.CMPEQ: l == r, Ops.CMPNE: l != r, Ops.CMPLT: l < r, Ops.CMPLE: l <= r}[u.op]
  if u.op == Ops.WHERE:
    c, t, f = _eval_uop(u.src[0]), _eval_uop(u.src[1]), _eval_uop(u.src[2])
    return t if c else f if None not in (c, t, f) else None
  if u.op == Ops.CUSTOM:
    if u.arg == 'trig_preop_result':
      shift = _eval_uop(u.src[0])
      return float(((TWO_OVER_PI_1201 << int(shift)) >> (1201 - 53)) & 0x1fffffffffffff) if shift is not None else None
    if u.arg in ('s_ff1_i32_b32', 's_ff1_i32_b64'):
      v = _eval_uop(u.src[0])
      if v is None: return None
      mask = MASK64 if 'b64' in u.arg else MASK32
      v = int(v) & mask
      if v == 0: return 64 if 'b64' in u.arg else 32
      n = 0
      while (v & 1) == 0: v >>= 1; n += 1
      return n
    if u.arg == 'vgpr_read': return None  # Needs runtime substitution
  return None

_DTYPE_ACCESSOR = {dtypes.uint8: 'u8', dtypes.int8: 'i8', dtypes.uint16: 'u16', dtypes.int16: 'i16',
                   dtypes.uint32: 'u32', dtypes.int32: 'i32', dtypes.uint64: 'u64', dtypes.int64: 'i64',
                   dtypes.float32: 'u32', dtypes.float64: 'u64'}

def _compile_pseudocode(pseudocode: str, mem_buf: UOp = MEM_BUF) -> tuple[UOp, list[tuple[str, DType]], dict[str, UOp], list[UOp]]:
  ctx = Ctx(mem_buf=mem_buf)
  try:
    stmts = parse(pseudocode)
  except AssertionError as e:
    print("issue parsing")
    print(pseudocode)
    print(e)
    raise
  for stmt in stmts: _stmt(stmt, ctx)
  return UOp(Ops.SINK, dtypes.void, tuple(u for _, u, _ in ctx.outputs) or ()), [(n, d) for n, _, d in ctx.outputs], INPUT_VARS, ctx.mem_stores

def _make_fn(sink: UOp, output_info: list[tuple[str, DType]], input_vars: dict[str, UOp], mem_stores: list[UOp]):
  if mem_stores: sink = UOp(Ops.SINK, dtypes.void, sink.src + tuple(mem_stores))
  topo = sink.toposort()
  is_lds, is_mem = any(u.op == Ops.DEFINE_LOCAL for u in topo), bool(mem_stores) or any(u.op == Ops.LOAD for u in topo)

  def _extract_results(s, MEM=None):
    for u in s.src:
      if u.op == Ops.STORE:
        addr, val, dt = int(u.src[0].src[1].arg), u.src[1].arg, u.src[0].dtype.base
        if dt == dtypes.float32: val = struct.unpack('<I', struct.pack('<f', val))[0]
        elif dt == dtypes.float64: val = struct.unpack('<Q', struct.pack('<d', val))[0]
        setattr(MEM[addr], _DTYPE_ACCESSOR.get(dt, 'u32'), int(val))
    result = {}
    for i, (name, dtype) in enumerate(output_info):
      if i >= len(s.src): continue
      val = s.src[i].arg if s.src[i].op == Ops.CONST else _eval_uop(s.src[i])
      if val is None: continue
      result[name] = _float_to_bits(val, dtype) if dtype in FLOATS else int(val) & ((1 << (dtype.itemsize * 8)) - 1)
    return result

  def _do_loads(s, MEM):
    loads = {}
    for u in s.toposort():
      if u.op == Ops.LOAD:
        addr, dt = int(u.src[0].src[1].arg), u.src[0].dtype.base
        loads[u] = UOp.const(dt, getattr(MEM[addr], _DTYPE_ACCESSOR.get(dt, 'u32')))
    return s.substitute(loads).simplify() if loads else s

  if is_lds:
    def fn(MEM, addr, data0=0, data1=0, offset0=0, offset1=0):
      dvars = {input_vars['ADDR']: UOp.const(dtypes.uint64, addr), input_vars['DATA']: UOp.const(dtypes.uint64, data0),
               input_vars['DATA2']: UOp.const(dtypes.uint64, data1), input_vars['OFFSET']: UOp.const(dtypes.uint32, offset0),
               input_vars['OFFSET0']: UOp.const(dtypes.uint32, offset0), input_vars['OFFSET1']: UOp.const(dtypes.uint32, offset1),
               input_vars['RETURN_DATA']: UOp.const(dtypes.uint64, 0)}
      return _extract_results(_do_loads(sink.substitute(dvars).simplify(), MEM), MEM)
    return fn
  elif is_mem:
    def fn(MEM, addr, vdata=0, vdst=0):
      dvars = {input_vars['ADDR']: UOp.const(dtypes.uint64, addr), input_vars['SDATA']: UOp.const(dtypes.uint64, 0),
               input_vars['VDATA']: UOp.const(dtypes.uint64, vdata), input_vars['VDST']: UOp.const(dtypes.uint64, vdst),
               input_vars['DATA']: UOp.const(dtypes.uint64, vdata), input_vars['DATA2']: UOp.const(dtypes.uint64, 0),
               input_vars['RETURN_DATA']: UOp.const(dtypes.uint64, 0)}
      return _extract_results(_do_loads(sink.substitute(dvars).simplify(), MEM), MEM)
    return fn
  else:
    def fn(s0, s1, s2, d0, scc, vcc, laneId, exec_mask, literal, VGPR, src0_idx=0, vdst_idx=0, pc=None, opsel=0, opsel_hi=0):
      simm16 = (literal if -32768 <= literal <= 32767 else (literal - 65536 if literal < 65536 else 0)) if literal else 0
      dvars = {
        input_vars['S0']: UOp.const(dtypes.uint32, s0 & MASK32), input_vars['S1']: UOp.const(dtypes.uint32, s1 & MASK32),
        input_vars['S2']: UOp.const(dtypes.uint32, s2 & MASK32), input_vars['D0']: UOp.const(dtypes.uint32, d0 & MASK32),
        input_vars['S0_64']: UOp.const(dtypes.uint64, s0), input_vars['S1_64']: UOp.const(dtypes.uint64, s1),
        input_vars['S2_64']: UOp.const(dtypes.uint64, s2), input_vars['D0_64']: UOp.const(dtypes.uint64, d0),
        input_vars['SCC']: UOp.const(dtypes.uint32, scc), input_vars['VCC']: UOp.const(dtypes.uint64, vcc),
        input_vars['EXEC']: UOp.const(dtypes.uint64, exec_mask), input_vars['laneId']: UOp.const(dtypes.uint32, laneId),
        input_vars['SIMM16']: UOp.const(dtypes.int32, simm16), input_vars['SIMM32']: UOp.const(dtypes.uint32, literal or 0),
        input_vars['PC']: UOp.const(dtypes.uint64, pc or 0), input_vars['OPSEL']: UOp.const(dtypes.uint32, opsel),
        input_vars['OPSEL_HI']: UOp.const(dtypes.uint32, opsel_hi), input_vars['SRC0']: UOp.const(dtypes.uint32, src0_idx),
      }
      s1_sub = sink.substitute(dvars).simplify()
      if VGPR is not None:
        vgpr_subs = {}
        for u in s1_sub.toposort():
          if u.op == Ops.CUSTOM and u.arg == 'vgpr_read':
            idx = _eval_uop(u.src[0])
            if idx is not None:
              lane, reg = int(idx) // 256, int(idx) % 256
              vgpr_subs[u] = UOp.const(dtypes.uint32, VGPR[lane][reg] if lane < len(VGPR) and reg < len(VGPR[lane]) else 0)
        if vgpr_subs: s1_sub = s1_sub.substitute(vgpr_subs).simplify()
      return _extract_results(s1_sub)
    return fn

# ═══════════════════════════════════════════════════════════════════════════════
# PSEUDOCODE FIXES
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_pseudocode_fixes(op_name: str, pcode: str) -> str:
  """Apply known fixes for PDF pseudocode bugs."""
  if op_name == 'V_DIV_FMAS_F32':
    pcode = pcode.replace('D0.f32 = 2.0F ** 32 * fma(S0.f32, S1.f32, S2.f32)',
      'D0.f32 = (exponent(S2.f32) > 127) ? (2.0F ** 64 * fma(S0.f32, S1.f32, S2.f32)) : (2.0F ** -64 * fma(S0.f32, S1.f32, S2.f32))')
  if op_name == 'V_DIV_FMAS_F64':
    pcode = pcode.replace('D0.f64 = 2.0 ** 64 * fma(S0.f64, S1.f64, S2.f64)',
      'D0.f64 = (exponent(S2.f64) > 1023) ? (2.0 ** 128 * fma(S0.f64, S1.f64, S2.f64)) : (2.0 ** -128 * fma(S0.f64, S1.f64, S2.f64))')
  if op_name == 'V_DIV_FIXUP_F32':
    pcode = pcode.replace('D0.f32 = sign_out ? -abs(S0.f32) : abs(S0.f32)',
      'D0.f32 = isNAN(S0.f32) ? (sign_out ? -OVERFLOW_F32 : OVERFLOW_F32) : (sign_out ? -abs(S0.f32) : abs(S0.f32))')
  if op_name == 'V_DIV_FIXUP_F64':
    pcode = pcode.replace('D0.f64 = sign_out ? -abs(S0.f64) : abs(S0.f64)',
      'D0.f64 = isNAN(S0.f64) ? (sign_out ? -OVERFLOW_F64 : OVERFLOW_F64) : (sign_out ? -abs(S0.f64) : abs(S0.f64))')
  if 'V_DIV_SCALE' in op_name:
    dt = 'f32' if 'F32' in op_name else 'f64'
    exp_lim, ldexp_val = ('23', '64') if dt == 'f32' else ('52', '128')
    pcode = pcode.replace(f'S2.{dt} / S1.{dt} == DENORM.{dt}', f'isDENORM(S2.{dt} / S1.{dt})')
    pcode = pcode.replace(f"1.0 / 64'F(S1.{dt}) == DENORM.f64", f"isDENORM(1.0 / 64'F(S1.{dt}))")
    pcode = pcode.replace(f'1.0 / S1.{dt} == DENORM.{dt}', f'isDENORM(1.0 / S1.{dt})')
    pcode = pcode.replace(f'S1.{dt} == DENORM.{dt}', f'isDENORM(S1.{dt})')
    pcode = pcode.replace(f'D0.{dt} = NAN.{dt}', f'VCC = 0x1LL;\nD0.{dt} = NAN.{dt}')
    pcode = pcode.replace(f'elsif isDENORM(S1.{dt}) then\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})', f'elsif 1 == 0 then\nD0.{dt} = S0.{dt}')
    pcode = pcode.replace(f'elsif exponent(S2.{dt}) <= {exp_lim} then\n// Numerator is tiny\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})',
                          f'elsif exponent(S2.{dt}) <= {exp_lim} then\nVCC = 0x1LL;\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})')
    pcode = pcode.replace(f'elsif isDENORM(S2.{dt} / S1.{dt}) then\nVCC = 0x1LL;\nif S0.{dt} == S2.{dt} then\n// Only scale the numerator\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nendif',
                          f'elsif isDENORM(S2.{dt} / S1.{dt}) then\nVCC = 0x1LL;\nD0.{dt} = S0.{dt}')
    pcode = pcode.replace(f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nendif\nelsif', f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nelse\nD0.{dt} = S0.{dt}\nendif\nelsif')
    lines = pcode.rstrip().split('\n')
    for i in range(len(lines) - 1, -1, -1):
      if lines[i].strip() == 'endif':
        lines.insert(i, f'else\nD0.{dt} = S0.{dt}')
        break
    pcode = '\n'.join(lines) + f';\nif isDENORM(S1.{dt}) then\nD0.{dt} = NAN.{dt}\nendif'
  if op_name == 'V_TRIG_PREOP_F64':
    pcode = pcode.replace("result = 64'F((1201'B(2.0 / PI)[1200 : 0] << shift.u32) & 1201'0x1fffffffffffff)", "result = trig_preop_result(shift)")
  return pcode

@functools.cache
def compile_uop(op_name: str, pseudocode: str):
  pseudocode = _apply_pseudocode_fixes(op_name, pseudocode)
  mem_buf = LDS_BUF if op_name.startswith('DS_') else MEM_BUF
  sink, output_info, input_vars, mem_stores = _compile_pseudocode(pseudocode, mem_buf)
  return _make_fn(sink, output_info, input_vars, mem_stores)
