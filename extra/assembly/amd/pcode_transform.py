# Transform parsed pcode CUSTOM ops to UOps using PatternMatcher
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, graph_rewrite
from tinygrad.uop.spec import program_spec, type_verify
from tinygrad.dtype import dtypes, DType
from extra.assembly.amd.pcode_parse import parse, If, For, Lambda, Break, Return
import math

# Placeholder buffer for MEM operations - substituted in ucode with actual buffer
MEM_BUF = UOp(Ops.DEFINE_GLOBAL, dtypes.uint8.ptr(0), arg=0)

# ═══════════════════════════════════════════════════════════════════════════════
# TYPE MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════════

_DT_SUFFIX = {
  'f16': dtypes.float16, 'f32': dtypes.float32, 'f64': dtypes.float64, 'bf16': dtypes.bfloat16,
  'i8': dtypes.int8, 'i16': dtypes.int16, 'i32': dtypes.int32, 'i64': dtypes.int64,
  'u8': dtypes.uint8, 'u16': dtypes.uint16, 'u32': dtypes.uint32, 'u64': dtypes.uint64,
}
_SPECIAL_CASTS = {'f32_to_u32', 'f64_to_u32', 'f16_to_u32', 'f32_to_u64', 'f64_to_u64', 'f16_to_u64',
                  'bf16_to_f32', 'u32_to_u16', 'i32_to_i16'}
_CAST_MAP = {f'{s}_to_{d}': _DT_SUFFIX[d] for s in _DT_SUFFIX for d in _DT_SUFFIX if s != d and f'{s}_to_{d}' not in _SPECIAL_CASTS}
_CAST_MAP.update({f'v_cvt_{d}_{s}': _DT_SUFFIX[d] for s in _DT_SUFFIX for d in _DT_SUFFIX if s != d and f'{s}_to_{d}' not in _SPECIAL_CASTS})

# CUSTOM op return types (ops that stay as CUSTOM after transformation)
# sign/exponent/mantissa stay as CUSTOM when args are void-typed (e.g., in lambda params)
_CUSTOM_TYPES = {
  'sign': dtypes.uint32, 'exponent': dtypes.uint32, 'mantissa': dtypes.float32,
  'count_ones': dtypes.uint32, 'countbits': dtypes.uint32, 'reverse_bits': dtypes.uint32,
  's_ff1_i32_b32': dtypes.uint32, 's_ff1_i32_b64': dtypes.uint32,
  'ConvertFromFormat': dtypes.uint32, 'nop': dtypes.uint32,
  'trig_preop_result': dtypes.float64,
}
# Constants: name -> (dtype, value) - replaced with CONST during transformation
_CONSTS = {
  'PI': (dtypes.float64, math.pi), 'INF': (dtypes.float64, math.inf),
  'MAX_FLOAT_F32': (dtypes.float32, 3.4028235e+38), 'MAX_FLOAT_F64': (dtypes.float64, 1.7976931348623157e+308),
  'OVERFLOW_F32': (dtypes.float32, math.inf), 'OVERFLOW_F64': (dtypes.float64, math.inf),
  'UNDERFLOW_F32': (dtypes.float32, 0.0), 'UNDERFLOW_F64': (dtypes.float64, 0.0),
  'WAVE32': (dtypes.uint32, 1), 'WAVE64': (dtypes.uint32, 0),  # RDNA3 is wave32 mode
  'WAVE_MODE.IEEE': (dtypes.uint32, 1), 'ROUND_MODE': (dtypes.uint32, 0),
  'NAN.f32': (dtypes.float32, float('nan')), 'NAN.f64': (dtypes.float64, float('nan')),
  'DENORM.f32': (dtypes.float32, 1.17549435e-38), 'DENORM.f64': (dtypes.float64, 2.2250738585072014e-308),
  'LDS': (dtypes.uint64, 0),
  # Debug condition flags (typically 0 in normal execution)
  'WAVE_STATUS.COND_DBG_SYS': (dtypes.uint32, 0), 'WAVE_STATUS.COND_DBG_USER': (dtypes.uint32, 0),
}

# Float bit layout: (uint_type, sign_shift, exp_shift, exp_mask, mantissa_mask, bias, quiet_bit)
FP_INFO = {
  dtypes.float64: (dtypes.uint64, 63, 52, 0x7ff, 0xfffffffffffff, 1023, 0x8000000000000),
  dtypes.float32: (dtypes.uint32, 31, 23, 0xff, 0x7fffff, 127, 0x400000),
  dtypes.float16: (dtypes.uint16, 15, 10, 0x1f, 0x3ff, 15, 0x200),
}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _tc(src: UOp, val) -> UOp:  # typed const
  return UOp.const(src.dtype, val) if src.dtype != dtypes.void else UOp(Ops.CONST, dtypes.void, (src,), val)

def _floor(x: UOp) -> UOp:
  trunc = UOp(Ops.TRUNC, x.dtype, (x,))
  return (x < trunc).where(trunc - 1, trunc)

def _minmax(a: UOp, b: UOp, is_min: bool, dt: DType|None = None) -> UOp:
  dt = dt or (a.dtype if a.dtype != dtypes.void else b.dtype)
  return ((a < b) if is_min else (b < a)).where(a, b)

def _vn(u: UOp) -> str|None:  # var name
  if u.op == Ops.DEFINE_VAR: return u.arg[0] if isinstance(u.arg, tuple) else u.arg
  return _vn(u.src[0]) if u.op == Ops.CUSTOMI and u.src[0].op == Ops.DEFINE_VAR else None

def _signext_from_bit(x: UOp, n: UOp) -> UOp:
  sign = UOp(Ops.SHL, x.dtype, (_tc(x, 1), UOp(Ops.SUB, x.dtype, (UOp(Ops.CAST, x.dtype, (n,)), _tc(x, 1)))))
  return UOp(Ops.WHERE, x.dtype, (n.eq(0), _tc(x, 0), UOp(Ops.SUB, x.dtype, (UOp(Ops.XOR, x.dtype, (x, sign)), sign))))

def _fp_bits(x: UOp) -> tuple[UOp, UOp, UOp, int, int]:
  """Extract float bit components. Returns (exp, mant, bits, exp_mask, quiet_bit)."""
  uint_dt, _, exp_shift, exp_mask, mant_mask, _, quiet_bit = FP_INFO.get(x.dtype, FP_INFO[dtypes.float32])
  bits = x.bitcast(uint_dt)
  return (bits >> exp_shift) & exp_mask, bits & mant_mask, bits, exp_mask, quiet_bit

def _is_denorm(x: UOp) -> UOp:
  exp, mant, _, _, _ = _fp_bits(x)
  return exp.eq(0) & mant.ne(0)

def _is_quiet_nan(x: UOp) -> UOp:
  exp, _, bits, exp_mask, quiet_bit = _fp_bits(x)
  return exp.eq(exp_mask) & (bits & quiet_bit).ne(0)

def _is_signal_nan(x: UOp) -> UOp:
  exp, mant, bits, exp_mask, quiet_bit = _fp_bits(x)
  return exp.eq(exp_mask) & mant.ne(0) & (bits & quiet_bit).eq(0)

def _fp_extract(x: UOp, field: str) -> UOp:
  uint_dt, sign_shift, exp_shift, exp_mask, mant_mask, bias, _ = FP_INFO.get(x.dtype, FP_INFO[dtypes.float32])
  bits = x.bitcast(uint_dt)
  if field == 'sign': return ((bits >> sign_shift) & 1).cast(dtypes.uint32)
  if field == 'exp': return ((bits >> exp_shift) & exp_mask).cast(dtypes.uint32)
  # mantissa: preserve sign, set exponent to bias-1, keep mantissa bits
  mant = ((bits & ((1 << sign_shift) | mant_mask)) | ((bias - 1) << exp_shift)).bitcast(x.dtype)
  return x.eq(0.0).where(x, mant)

def _v_sad_u8(a: UOp, b: UOp, c: UOp) -> UOp:
  result = c
  for i in range(4):
    ba, bb = (a >> (i * 8)) & 0xff, (b >> (i * 8)) & 0xff
    diff = ba - bb
    result = result + (diff < 0x80000000).where(diff, 0 - diff)
  return result

def _byte_permute(data: UOp, sel: UOp) -> UOp:
  u32, u64 = dtypes.uint32, dtypes.uint64
  src64 = data.cast(u64)
  sel_m = sel.cast(u32) & 0xff
  sel_idx, sel_nib = sel_m & 7, sel_m & 0xf
  result = ((src64 >> (sel_idx << 3).cast(u64)) & 0xff).cast(u32)
  for i, pos in enumerate([15, 31, 47, 63], 8):  # sign extension from byte boundaries
    sbit = ((src64 >> pos) & 1).ne(0).where(UOp.const(u32, 0xff), UOp.const(u32, 0))
    result = sel_nib.eq(i).where(sbit, result)
  result = sel_nib.eq(12).where(UOp.const(u32, 0), result)
  result = (sel_nib > 12).where(UOp.const(u32, 0xff), result)
  return (sel_m & 0x80).ne(0).where(UOp.const(u32, 0), result)

# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

def _typed_minmax(op, *args):
  if not isinstance(op.arg, str): return None
  n, suffix = len(args), op.arg.split('_')[-1]
  if suffix not in _DT_SUFFIX: return None
  if n == 2 and (op.arg.startswith('v_min_') or op.arg.startswith('v_max_')):
    return _minmax(args[0], args[1], op.arg.startswith('v_min_'), _DT_SUFFIX[suffix])
  if n == 3 and (op.arg.startswith('v_min3_') or op.arg.startswith('v_max3_')):
    return _minmax(_minmax(args[0], args[1], op.arg.startswith('v_min3_'), _DT_SUFFIX[suffix]), args[2], op.arg.startswith('v_min3_'), _DT_SUFFIX[suffix])
  return None

def _track_var(ctx, u):
  if ctx is None or u.dtype == dtypes.void: return None
  name = u.arg[0] if isinstance(u.arg, tuple) else u.arg
  if name in ctx: assert ctx[name] == u.dtype, f"variable '{name}' type conflict: {ctx[name]} vs {u.dtype}"
  else: ctx[name] = u.dtype

def _prop_var(ctx, u):
  if ctx is None: return None
  name = u.arg[0] if isinstance(u.arg, tuple) else u.arg
  if name not in ctx: return None
  dt = ctx[name]
  return UOp(Ops.DEFINE_VAR, dt, arg=(name, dtypes.min(dt), dtypes.max(dt)))

def _prop_assign(ctx, lhs, rhs):
  if ctx is None or rhs.dtype == dtypes.void or lhs.op != Ops.DEFINE_VAR: return None
  if (name := _vn(lhs)) is None or name in ctx: return None
  ctx[name] = rhs.dtype
  return UOp(Ops.ASSIGN, rhs.dtype, (UOp(Ops.DEFINE_VAR, rhs.dtype, arg=(name, dtypes.min(rhs.dtype), dtypes.max(rhs.dtype))), rhs))

def _prop_binop(l, r, __OP__):
  # Don't infer type if either operand is void - wait for variable typing first
  if l.dtype == dtypes.void or r.dtype == dtypes.void: return None
  if __OP__.op in {Ops.SHL, Ops.SHR}: dt = l.dtype
  else: dt = l.dtype if l.dtype.itemsize >= r.dtype.itemsize else r.dtype
  return UOp(__OP__.op, dt, (l, r), __OP__.arg)

def _prop_custom(x):
  if x.arg in _CUSTOM_TYPES: return UOp(Ops.CUSTOM, _CUSTOM_TYPES[x.arg], x.src, x.arg)
  if x.arg in {'MEM', 'abs', 'cvtToQuietNAN'}: return None  # wrapped by BITCAST/CAST
  dt = next((s.dtype for s in x.src if s.dtype != dtypes.void), dtypes.void)
  if dt == dtypes.void: return None  # wait for sources to be typed first
  return UOp(Ops.CUSTOM, dt, x.src, x.arg)

def _prop_customi(base, hi, lo):
  if base.dtype == dtypes.void: return None  # wait for base to be typed
  if hi is lo:
    # Array element access: use scalar element type
    dt = base.dtype.scalar() if base.dtype.count > 1 else base.dtype
    return UOp(Ops.CUSTOMI, dt, (base, hi, lo))
  if hi.op == Ops.CONST and lo.op == Ops.CONST:
    return UOp(Ops.CUSTOMI, dtypes.uint64 if abs(int(hi.arg) - int(lo.arg)) + 1 > 32 else dtypes.uint32, (base, hi, lo))
  return UOp(Ops.CUSTOMI, dtypes.uint32, (base, hi, lo))

def _backprop(ctx, op, v, t):
  if t.dtype == dtypes.void: return None
  name = v.arg[0] if isinstance(v.arg, tuple) else v.arg
  # Don't back-propagate if variable already has a type in context
  if ctx is not None and name in ctx: return None
  new_var = UOp(Ops.DEFINE_VAR, t.dtype, arg=v.arg)
  return UOp(op.op, op.dtype, (new_var, t) if op.src[0] is v else (t, new_var), op.arg)

def _prop_unop(x, __OP__):
  return UOp(__OP__.op, x.dtype, (x,), __OP__.arg) if x.dtype != dtypes.void else None

def _prop_mulacc(a, b, c):
  dt = next((x.dtype for x in (a, b, c) if x.dtype != dtypes.void), dtypes.void)
  return UOp(Ops.MULACC, dt, (a, b, c)) if dt != dtypes.void else None

def _prop_where(cond, t, f):
  dt = t.dtype if t.dtype != dtypes.void else f.dtype
  return UOp(Ops.WHERE, dt, (cond, t, f)) if dt != dtypes.void else None

def _prop_cat(x):
  if not x.src: return None
  bits = sum(s.dtype.itemsize * 8 if s.dtype != dtypes.void else 0 for s in x.src)
  return UOp(Ops.CAT, dtypes.uint64 if bits > 32 else dtypes.uint32, x.src) if bits > 0 else None

def _lower_cat(hi, lo):
  """Lower CAT {hi, lo} to (hi << shift) | lo."""
  if lo.dtype.itemsize >= 4:  # 32-bit lo -> 64-bit result
    return (hi.cast(dtypes.uint64) << 32) | lo.cast(dtypes.uint64)
  # 16-bit lo -> 32-bit result
  return (hi.cast(dtypes.uint32) << 16) | (lo.cast(dtypes.uint32) & 0xffff)

def _lower_cat_assign(lhs_parts, rhs):
  """Lower {hi, lo} = rhs into GROUP(ASSIGN(lo, rhs & mask), ASSIGN(hi, rhs >> shift))."""
  rhs_bits = rhs.cast(dtypes.uint64) if rhs.dtype != dtypes.uint64 else rhs
  assigns, offset = [], 0
  for part in reversed(lhs_parts):  # lo first, then hi
    bits = 1 if part.dtype.name == 'u1' else 64 if part.dtype == dtypes.uint64 else part.dtype.itemsize * 8
    mask = (1 << bits) - 1
    val = ((rhs_bits >> offset) & mask).cast(part.dtype)
    assigns.append(UOp(Ops.ASSIGN, part.dtype, (part, val)))
    offset += bits
  return UOp(Ops.GROUP, dtypes.void, tuple(assigns))

def _typed_cast(x, op):
  if op.arg not in _CAST_MAP: return None
  return UOp(Ops.CAST, _CAST_MAP[op.arg], (x,))

# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN MATCHER
# ═══════════════════════════════════════════════════════════════════════════════

_fpat = UPat.var('x', dtype=dtypes.floats)

def _cast_const_to_bitcast(bc, c):
  """CAST(float, CONST(uint, val)) -> CONST with proper float bit pattern (preserves NaN)."""
  import struct
  val = c.arg
  if bc.dtype == dtypes.float32 and c.dtype == dtypes.uint32:
    return UOp(Ops.CONST, dtypes.float32, arg=struct.unpack('f', struct.pack('I', val))[0])
  if bc.dtype == dtypes.float64 and c.dtype == dtypes.uint64:
    return UOp(Ops.CONST, dtypes.float64, arg=struct.unpack('d', struct.pack('Q', val))[0])
  if bc.dtype == dtypes.float16 and c.dtype == dtypes.uint16:
    import numpy as np
    return UOp(Ops.CONST, dtypes.float16, arg=float(np.frombuffer(struct.pack('H', val), dtype=np.float16)[0]))
  return None

pcode_pm = PatternMatcher([
  # CAST(float, CONST(uint)) -> proper bitcast to preserve NaN bit patterns (e.g., 32'F(0xffc00000))
  (UPat(Ops.CAST, dtype=dtypes.floats, src=(UPat(Ops.CONST, dtype=dtypes.uints, name='c'),), name='bc'), _cast_const_to_bitcast),
  # Eliminate round-trip BITCAST: BITCAST(dtype, BITCAST(_, x)) -> x when x.dtype == dtype (avoids NaN canonicalization)
  (UPat(Ops.BITCAST, src=(UPat(Ops.BITCAST, src=(UPat.var('x'),)),), name='bc'), lambda bc, x: x if x.dtype == bc.dtype else None),
  # MEM read: BITCAST(CUSTOM('MEM', addr)) -> INDEX(buf, addr) with element type
  (UPat(Ops.BITCAST, name='bc', src=(UPat(Ops.CUSTOM, arg='MEM', src=(UPat.var('addr'),)),)),
   lambda bc, addr: UOp(Ops.INDEX, bc.dtype, (MEM_BUF, addr))),
  # MEM write: ASSIGN(INDEX, val) -> STORE (INDEX created by MEM read pattern above)
  (UPat(Ops.ASSIGN, src=(UPat(Ops.INDEX, name='idx'), UPat.var('val'))),
   lambda idx, val: UOp(Ops.STORE, dtypes.void, (idx, val))),
  # Float ops (preserve input type)
  (UPat(Ops.CUSTOM, arg='trunc', src=(_fpat,)), lambda x: UOp(Ops.TRUNC, x.dtype, (x,))),
  (UPat(Ops.CUSTOM, arg='sqrt', src=(_fpat,)), lambda x: UOp(Ops.SQRT, x.dtype, (x,))),
  (UPat(Ops.CUSTOM, arg='exp2', src=(_fpat,)), lambda x: UOp(Ops.EXP2, x.dtype, (x,))),
  (UPat(Ops.CUSTOM, arg='log2', src=(_fpat,)), lambda x: UOp(Ops.LOG2, x.dtype, (x,))),
  (UPat(Ops.CUSTOM, arg='sin', src=(_fpat,)), lambda x: UOp(Ops.SIN, x.dtype, (x,))),
  (UPat(Ops.CUSTOM, arg='rcp', src=(_fpat,)), lambda x: UOp(Ops.RECIPROCAL, x.dtype, (x,))),
  (UPat(Ops.CUSTOM, arg='fma', src=(_fpat, UPat.var('b'), UPat.var('c'))), lambda x, b, c: UOp(Ops.MULACC, x.dtype, (x, b, c))),
  (UPat(Ops.CUSTOM, arg='abs', src=(_fpat,)), lambda x: UOp(Ops.WHERE, x.dtype, (UOp(Ops.CMPLT, dtypes.bool, (x, _tc(x, 0))), UOp(Ops.NEG, x.dtype, (x,)), x))),
  (UPat(Ops.CUSTOM, arg='cos', src=(_fpat,)), lambda x: UOp(Ops.SIN, x.dtype, (UOp(Ops.ADD, x.dtype, (x, _tc(x, 1.5707963267948966))),))),
  (UPat(Ops.CUSTOM, arg='floor', src=(_fpat,)), lambda x: _floor(x)),
  (UPat(Ops.CUSTOM, arg='fract', src=(_fpat,)), lambda x: UOp(Ops.SUB, x.dtype, (x, _floor(x)))),
  (UPat(Ops.CUSTOM, arg='rsqrt', src=(_fpat,)), lambda x: UOp(Ops.RECIPROCAL, x.dtype, (UOp(Ops.SQRT, x.dtype, (x,)),))),
  # pow() function: pow(2.0, x) -> EXP2(x), pow(a, b) -> EXP2(b * LOG2(a))
  (UPat(Ops.CUSTOM, arg='pow', src=(UPat(Ops.CONST, arg=2.0), UPat.var('x'))), lambda x: UOp(Ops.EXP2, x.dtype, (x,))),
  (UPat(Ops.CUSTOM, arg='pow', src=(UPat.var('a', dtype=dtypes.floats), UPat.var('b'))),
   lambda a, b: UOp(Ops.EXP2, a.dtype, (UOp(Ops.MUL, a.dtype, (UOp(Ops.CAST, a.dtype, (b,)) if b.dtype != a.dtype else b, UOp(Ops.LOG2, a.dtype, (a,)))),))),
  # POW operator: 2.0 ** x -> EXP2(x), a ** b -> EXP2(b * LOG2(a))
  (UPat(Ops.POW, src=(UPat(Ops.CONST, arg=2.0), UPat.var('x'))), lambda x: UOp(Ops.EXP2, x.dtype, (x,)) if x.dtype in dtypes.floats else UOp(Ops.EXP2, dtypes.float32, (UOp(Ops.CAST, dtypes.float32, (x,)),))),
  (UPat(Ops.POW, dtype=dtypes.floats, src=(UPat.var('a'), UPat.var('b')), name='p'), lambda p, a, b: UOp(Ops.EXP2, p.dtype, (UOp(Ops.MUL, p.dtype, (UOp(Ops.CAST, p.dtype, (b,)) if b.dtype != p.dtype else b, UOp(Ops.LOG2, p.dtype, (a,)))),))),
  # MOD: a % b -> a - (a / b) * b
  (UPat(Ops.MOD, dtype=dtypes.ints, src=(UPat.var('a'), UPat.var('b')), name='m'), lambda m, a, b: UOp(Ops.SUB, m.dtype, (a, UOp(Ops.MUL, m.dtype, (UOp(Ops.IDIV, m.dtype, (a, b)), b))))),
  # ldexp(x, n) = x * 2^n
  (UPat(Ops.CUSTOM, arg='ldexp', src=(UPat.var('x', dtype=dtypes.floats), UPat.var('n'))),
   lambda x, n: UOp(Ops.MUL, x.dtype, (x, UOp(Ops.EXP2, x.dtype, (UOp(Ops.CAST, x.dtype, (n,)) if n.dtype != x.dtype else n,))))),
  # Mask conversions
  (UPat(Ops.CUSTOM, arg='u8_to_u32', src=(UPat.var('x'),)), lambda x: UOp(Ops.AND, dtypes.uint32, (UOp(Ops.CAST, dtypes.uint32, (x,)), UOp.const(dtypes.uint32, 0xff)))),
  (UPat(Ops.CUSTOM, arg='u4_to_u32', src=(UPat.var('x'),)), lambda x: UOp(Ops.AND, dtypes.uint32, (UOp(Ops.CAST, dtypes.uint32, (x,)), UOp.const(dtypes.uint32, 0xf)))),
  # isEven: check if integer part is even
  (UPat(Ops.CUSTOM, arg='isEven', src=(UPat.var('x'),)),
   lambda x: UOp(Ops.CMPEQ, dtypes.bool, (UOp(Ops.AND, dtypes.int64, (UOp(Ops.CAST, dtypes.int64, (x,)), UOp.const(dtypes.int64, 1))), UOp.const(dtypes.int64, 0)))),
  # Boolean functions
  (UPat(Ops.CUSTOM, arg='isNAN', src=(UPat.var('x'),)), lambda x: UOp(Ops.CMPNE, dtypes.bool, (x, x))),
  (UPat(Ops.CUSTOM, arg='isINF', src=(UPat.var('x'),)), lambda x: UOp(Ops.OR, dtypes.bool, (
    UOp(Ops.CMPEQ, dtypes.bool, (x, _tc(x, float('inf')))), UOp(Ops.CMPEQ, dtypes.bool, (x, _tc(x, float('-inf'))))))),
  (UPat(Ops.CUSTOM, arg='isDENORM', src=(_fpat,)), _is_denorm),
  # isQuietNAN/isSignalNAN with CAST to float64: check the original float directly (avoids NaN canonicalization through Python)
  (UPat(Ops.CUSTOM, arg='isQuietNAN', src=(UPat(Ops.CAST, dtype=dtypes.float64, src=(UPat.var('x', dtype=dtypes.floats),)),)), _is_quiet_nan),
  (UPat(Ops.CUSTOM, arg='isSignalNAN', src=(UPat(Ops.CAST, dtype=dtypes.float64, src=(UPat.var('x', dtype=dtypes.floats),)),)), _is_signal_nan),
  (UPat(Ops.CUSTOM, arg='isQuietNAN', src=(_fpat,)), _is_quiet_nan),
  (UPat(Ops.CUSTOM, arg='isSignalNAN', src=(_fpat,)), _is_signal_nan),
  # Float component extraction
  (UPat(Ops.CUSTOM, arg='sign', src=(_fpat,)), lambda x: _fp_extract(x, 'sign')),
  (UPat(Ops.CUSTOM, arg='exponent', src=(_fpat,)), lambda x: _fp_extract(x, 'exp')),
  (UPat(Ops.CUSTOM, arg='mantissa', src=(_fpat,)), lambda x: _fp_extract(x, 'mant')),
  # ABSDIFF: |a - b| for unsigned
  (UPat(Ops.CUSTOM, arg='ABSDIFF', src=(UPat.var('a'), UPat.var('b'))),
   lambda a, b: UOp(Ops.WHERE, a.dtype, (UOp(Ops.CMPLT, dtypes.bool, (b, a)),
     UOp(Ops.SUB, a.dtype, (a, b)), UOp(Ops.SUB, a.dtype, (b, a))))),
  # SAT8: clamp to signed 8-bit range [-128, 127]
  (UPat(Ops.CUSTOM, arg='SAT8', src=(UPat.var('x'),)),
   lambda x: UOp(Ops.WHERE, x.dtype, (UOp(Ops.CMPLT, dtypes.bool, (x, _tc(x, -128))), _tc(x, -128),
     UOp(Ops.WHERE, x.dtype, (UOp(Ops.CMPLT, dtypes.bool, (_tc(x, 127), x)), _tc(x, 127), x))))),
  # cvtToQuietNAN: passthrough (just returns the arg)
  (UPat(Ops.CUSTOM, arg='cvtToQuietNAN', src=(UPat.var('x'),)), lambda x: x),
  # LT/GT_NEG_ZERO: signed comparison via bitcast to int (handles f16 fallback to f32 int type)
  (UPat(Ops.CUSTOM, arg='LT_NEG_ZERO', src=(UPat.var('a', dtype=dtypes.floats), UPat.var('b', dtype=dtypes.floats))),
   lambda a, b: (idt := {dtypes.float64: dtypes.int64, dtypes.float16: dtypes.int16}.get(a.dtype, dtypes.int32)) and
     UOp(Ops.CMPLT, dtypes.bool, (UOp(Ops.BITCAST, idt, (a,)), UOp(Ops.BITCAST, idt, (b,))))),
  (UPat(Ops.CUSTOM, arg='GT_NEG_ZERO', src=(UPat.var('a', dtype=dtypes.floats), UPat.var('b', dtype=dtypes.floats))),
   lambda a, b: (idt := {dtypes.float64: dtypes.int64, dtypes.float16: dtypes.int16}.get(a.dtype, dtypes.int32)) and
     UOp(Ops.CMPLT, dtypes.bool, (UOp(Ops.BITCAST, idt, (b,)), UOp(Ops.BITCAST, idt, (a,))))),
  # min/max
  (UPat(Ops.CUSTOM, arg='min', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _minmax(a, b, True)),
  (UPat(Ops.CUSTOM, arg='max', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _minmax(a, b, False)),
  (UPat(Ops.CUSTOM, arg='clamp', src=(UPat.var('x'), UPat.var('lo'), UPat.var('hi'))), lambda x, lo, hi: _minmax(_minmax(x, lo, False), hi, True)),
  (UPat(Ops.CUSTOM, src=(UPat.var('a'), UPat.var('b')), name='op'), lambda op, a, b: _typed_minmax(op, a, b)),
  (UPat(Ops.CUSTOM, src=(UPat.var('a'), UPat.var('b'), UPat.var('c')), name='op'), lambda op, a, b, c: _typed_minmax(op, a, b, c)),
  # Type conversions
  (UPat(Ops.CUSTOM, src=(UPat.var('x'),), name='op'), _typed_cast),
  (UPat(Ops.CUSTOM, arg='signext', src=(UPat.var('x', dtype=dtypes.ints),)), lambda x: UOp(Ops.CAST, dtypes.int64, (x,))),
  (UPat(Ops.CUSTOM, arg='bf16_to_f32', src=(UPat.var('x', dtype=dtypes.bfloat16),)),
   lambda x: UOp(Ops.BITCAST, dtypes.float32, (UOp(Ops.SHL, dtypes.uint32, (UOp(Ops.CAST, dtypes.uint32, (x,)), UOp.const(dtypes.uint32, 16))),))),
  (UPat(Ops.CUSTOM, arg='u32_to_u16', src=(UPat.var('x', dtype=dtypes.uint32),)), lambda x: UOp(Ops.AND, dtypes.uint32, (x, UOp.const(dtypes.uint32, 0xffff)))),
  (UPat(Ops.CUSTOM, arg='i32_to_i16', src=(UPat.var('x', dtype=dtypes.int32),)),
   lambda x: UOp(Ops.CAST, dtypes.int16, (UOp(Ops.AND, dtypes.uint32, (UOp(Ops.CAST, dtypes.uint32, (x,)), UOp.const(dtypes.uint32, 0xffff))),))),
  # f32_to_u32, f64_to_u32: clamp negative to 0, then cast
  (UPat(Ops.CUSTOM, arg='f32_to_u32', src=(UPat.var('x', dtype=dtypes.float32),)),
   lambda x: UOp(Ops.CAST, dtypes.uint32, (UOp(Ops.WHERE, x.dtype, (UOp(Ops.CMPLT, dtypes.bool, (x, _tc(x, 0.0))), _tc(x, 0.0), x)),))),
  (UPat(Ops.CUSTOM, arg='f64_to_u32', src=(UPat.var('x', dtype=dtypes.float64),)),
   lambda x: UOp(Ops.CAST, dtypes.uint32, (UOp(Ops.WHERE, x.dtype, (UOp(Ops.CMPLT, dtypes.bool, (x, _tc(x, 0.0))), _tc(x, 0.0), x)),))),
  # snorm/unorm: clamp to [-1,1] or [0,1], scale, cast
  (UPat(Ops.CUSTOM, arg='f16_to_snorm', src=(UPat.var('x', dtype=dtypes.float16),)),
   lambda x: UOp(Ops.CAST, dtypes.int16, (UOp(Ops.MUL, x.dtype, (_minmax(_minmax(x, _tc(x, -1.0), False), _tc(x, 1.0), True), _tc(x, 32767.0))),))),
  (UPat(Ops.CUSTOM, arg='f32_to_snorm', src=(UPat.var('x', dtype=dtypes.float32),)),
   lambda x: UOp(Ops.CAST, dtypes.int16, (UOp(Ops.MUL, x.dtype, (_minmax(_minmax(x, _tc(x, -1.0), False), _tc(x, 1.0), True), _tc(x, 32767.0))),))),
  (UPat(Ops.CUSTOM, arg='f16_to_unorm', src=(UPat.var('x', dtype=dtypes.float16),)),
   lambda x: UOp(Ops.CAST, dtypes.uint16, (UOp(Ops.MUL, x.dtype, (_minmax(_minmax(x, _tc(x, 0.0), False), _tc(x, 1.0), True), _tc(x, 65535.0))),))),
  (UPat(Ops.CUSTOM, arg='f32_to_unorm', src=(UPat.var('x', dtype=dtypes.float32),)),
   lambda x: UOp(Ops.CAST, dtypes.uint16, (UOp(Ops.MUL, x.dtype, (_minmax(_minmax(x, _tc(x, 0.0), False), _tc(x, 1.0), True), _tc(x, 65535.0))),))),
  # signext_from_bit: sign extend from bit position
  (UPat(Ops.CUSTOM, arg='signext_from_bit', src=(UPat.var('x'), UPat.var('n'))), _signext_from_bit),
  # v_sad_u8/v_msad_u8: sum of absolute differences of packed bytes
  (UPat(Ops.CUSTOM, arg='v_sad_u8', src=(UPat.var('a'), UPat.var('b'), UPat.var('c'))), _v_sad_u8),
  (UPat(Ops.CUSTOM, arg='v_msad_u8', src=(UPat.var('a'), UPat.var('b'), UPat.var('c'))), _v_sad_u8),  # same impl
  (UPat(Ops.CUSTOM, arg='v_sad_u8', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _v_sad_u8(a, b, UOp.const(dtypes.uint32, 0))),
  (UPat(Ops.CUSTOM, arg='v_msad_u8', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _v_sad_u8(a, b, UOp.const(dtypes.uint32, 0))),
  # BYTE_PERMUTE: byte selection from 64-bit value
  (UPat(Ops.CUSTOM, arg='BYTE_PERMUTE', src=(UPat.var('data'), UPat.var('sel'))), _byte_permute),
]) + PatternMatcher([
  # Named constants from _CONSTS dict
  (UPat(Ops.DEFINE_VAR, name='u'), lambda u: UOp.const(*_CONSTS[u.arg[0]]) if isinstance(u.arg, tuple) and u.arg[0] in _CONSTS else None),
  # Typed constants: INF.f32, NAN.f64, DENORM.f32, etc. (BITCAST of untyped var to target type)
  (UPat(Ops.BITCAST, dtype=dtypes.floats, src=(UPat(Ops.DEFINE_VAR, arg=('INF', -float('inf'), float('inf'))),), name='x'), lambda x: UOp.const(x.dtype, float('inf'))),
  (UPat(Ops.BITCAST, dtype=dtypes.floats, src=(UPat(Ops.DEFINE_VAR, arg=('NAN', -float('inf'), float('inf'))),), name='x'), lambda x: UOp.const(x.dtype, float('nan'))),
  (UPat(Ops.BITCAST, dtype=dtypes.floats, src=(UPat(Ops.DEFINE_VAR, arg=('DENORM', -float('inf'), float('inf'))),), name='x'),
   lambda x: UOp.const(x.dtype, FP_INFO.get(x.dtype, FP_INFO[dtypes.float32])[4] * 2**(-FP_INFO.get(x.dtype, FP_INFO[dtypes.float32])[5]))),
  # Variable type tracking and propagation
  (UPat(Ops.DEFINE_VAR, name='u'), _track_var),
  (UPat(Ops.DEFINE_VAR, dtype=dtypes.void, name='u'), _prop_var),
  (UPat(Ops.ASSIGN, src=(UPat(Ops.DEFINE_VAR, dtype=dtypes.void, name='lhs'), UPat.var('rhs'))), _prop_assign),
  # Propagate dtype for ASSIGN from rhs, or propagate lhs type to void DEFINE_VAR rhs
  (UPat(Ops.ASSIGN, dtype=dtypes.void, src=(UPat.var('lhs'), UPat.var('rhs'))),
   lambda lhs, rhs: UOp(Ops.ASSIGN, rhs.dtype, (lhs, rhs)) if rhs.dtype != dtypes.void else
                    UOp(Ops.ASSIGN, lhs.dtype, (lhs, rhs.replace(dtype=lhs.dtype))) if lhs.dtype != dtypes.void and rhs.op == Ops.DEFINE_VAR else None),
  # Dtype propagation for void-typed ops
  (UPat((Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.AND, Ops.OR, Ops.XOR, Ops.SHL, Ops.SHR, Ops.MOD, Ops.POW),
        dtype=dtypes.void, src=(UPat.var('l'), UPat.var('r')), name='__OP__'), _prop_binop),
  (UPat((Ops.NEG, Ops.TRUNC, Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.RECIPROCAL),
        dtype=dtypes.void, src=(UPat.var('x'),), name='__OP__'), _prop_unop),
  # Unary XOR (NOT) -> binary XOR with all ones
  (UPat(Ops.XOR, src=(UPat.var('x'),)),
   lambda x: UOp(Ops.XOR, x.dtype, (x, UOp.const(x.dtype, -1))) if x.dtype != dtypes.void else None),
  # Unary CMPEQ (logical NOT) -> CMPEQ(x, 0) with matching type (default to uint32 for void)
  (UPat(Ops.CMPEQ, dtype=dtypes.bool, src=(UPat.var('x'),)),
   lambda x: UOp(Ops.CMPEQ, dtypes.bool, (x, UOp.const(x.dtype if x.dtype != dtypes.void else dtypes.uint32, 0)))),
  (UPat(Ops.MULACC, dtype=dtypes.void, src=(UPat.var('a'), UPat.var('b'), UPat.var('c'))), _prop_mulacc),
  (UPat(Ops.WHERE, dtype=dtypes.void, src=(UPat.var('cond'), UPat.var('t'), UPat.var('f'))), _prop_where),
  # Lower ASSIGN with CAT LHS to GROUP of ASSIGNs (bottom_up=True matches ASSIGN before CAT is lowered)
  (UPat(Ops.ASSIGN, src=(UPat(Ops.CAT, src=(UPat.var('hi'), UPat.var('lo'))), UPat.var('rhs'))),
   lambda hi, lo, rhs: _lower_cat_assign((hi, lo), rhs)),
  # Type void CAT
  (UPat(Ops.CAT, dtype=dtypes.void, name='x'), _prop_cat),
  # Lower typed CAT (RHS) to SHL+OR
  (UPat(Ops.CAT, dtype={dtypes.uint32, dtypes.uint64}, src=(UPat.var('hi'), UPat.var('lo'))), _lower_cat),
  (UPat(Ops.CUSTOMI, dtype=dtypes.void, src=(UPat.var('base'), UPat.var('hi'), UPat.var('lo'))), _prop_customi),
  (UPat(Ops.CUSTOM, dtype=dtypes.void, name='x'), _prop_custom),
  # Fix comparison type mismatches: cast to larger type
  (UPat((Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ, Ops.CMPLE), src=(UPat.var('x'), UPat.var('y')), name='cmp'),
   lambda cmp, x, y: UOp(cmp.op, dtypes.bool, (x, UOp(Ops.CAST, x.dtype, (y,)))) if x.dtype != dtypes.void and y.dtype != dtypes.void and x.dtype != y.dtype and x.dtype.itemsize >= y.dtype.itemsize else None),
  (UPat((Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ, Ops.CMPLE), src=(UPat.var('x'), UPat.var('y')), name='cmp'),
   lambda cmp, x, y: UOp(cmp.op, dtypes.bool, (UOp(Ops.CAST, y.dtype, (x,)), y)) if x.dtype != dtypes.void and y.dtype != dtypes.void and x.dtype != y.dtype and y.dtype.itemsize > x.dtype.itemsize else None),
  # Fix WHERE with non-bool condition: cast int condition to bool (test != 0)
  (UPat(Ops.WHERE, src=(UPat.var('c', dtype=dtypes.ints), UPat.var('t'), UPat.var('f'))),
   lambda c, t, f: UOp(Ops.WHERE, t.dtype if t.dtype != dtypes.void else f.dtype, (UOp(Ops.CMPNE, dtypes.bool, (c, UOp.const(c.dtype, 0))), t, f))),
  # Fix logical AND/OR with bool and int: convert int to bool (!= 0)
  (UPat((Ops.AND, Ops.OR), src=(UPat.var('x', dtype=dtypes.bool), UPat.var('y', dtype=dtypes.ints))),
   lambda x, y: UOp(Ops.AND, dtypes.bool, (x, UOp(Ops.CMPNE, dtypes.bool, (y, UOp.const(y.dtype, 0)))))),
  (UPat((Ops.AND, Ops.OR), src=(UPat.var('x', dtype=dtypes.ints), UPat.var('y', dtype=dtypes.bool))),
   lambda x, y: UOp(Ops.AND, dtypes.bool, (UOp(Ops.CMPNE, dtypes.bool, (x, UOp.const(x.dtype, 0))), y))),
  # Fix binary op type mismatches: cast smaller to larger (excluding POW which allows int exponent)
  (UPat((Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.AND, Ops.OR, Ops.XOR), src=(UPat.var('x'), UPat.var('y')), name='op'),
   lambda op, x, y: UOp(op.op, x.dtype, (x, UOp(Ops.CAST, x.dtype, (y,)))) if x.dtype != dtypes.void and y.dtype != dtypes.void and x.dtype != y.dtype and x.dtype.itemsize >= y.dtype.itemsize else None),
  (UPat((Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.AND, Ops.OR, Ops.XOR), src=(UPat.var('x'), UPat.var('y')), name='op'),
   lambda op, x, y: UOp(op.op, y.dtype, (UOp(Ops.CAST, y.dtype, (x,)), y)) if x.dtype != dtypes.void and y.dtype != dtypes.void and x.dtype != y.dtype and y.dtype.itemsize > x.dtype.itemsize else None),
  # Back-propagate types to void DEFINE_VAR sources (skip if var already has type in ctx)
  (UPat((Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.AND, Ops.OR, Ops.XOR),
        src=(UPat(Ops.DEFINE_VAR, dtype=dtypes.void, name='v'), UPat.var('t')), name='op'),
   lambda ctx, op, v, t: _backprop(ctx, op, v, t)),
  (UPat((Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.AND, Ops.OR, Ops.XOR),
        src=(UPat.var('t'), UPat(Ops.DEFINE_VAR, dtype=dtypes.void, name='v')), name='op'),
   lambda ctx, op, t, v: _backprop(ctx, op, v, t)),
])

# ═══════════════════════════════════════════════════════════════════════════════
# PCODE SPEC (extends program_spec with pcode-specific exceptions)
# ═══════════════════════════════════════════════════════════════════════════════

pcode_spec = PatternMatcher([
  # DEFINE_VAR: pcode uses (name, min, max) tuples for all vars
  (UPat(Ops.DEFINE_VAR, name="x"), lambda x: isinstance(x.arg, tuple) and len(x.arg) == 3),
  # ASSIGN: pcode assignment statement (dtype must match rhs)
  (UPat(Ops.ASSIGN, src=(UPat.var("lhs"), UPat.var("rhs")), name="a"),
   lambda a, lhs, rhs: a.dtype == rhs.dtype and rhs.dtype != dtypes.void),
]) + program_spec

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORM
# ═══════════════════════════════════════════════════════════════════════════════

def _transform_uop(u: UOp, ctx: dict) -> UOp:
  result = graph_rewrite(u, pcode_pm, ctx=ctx, bottom_up=True).simplify()
  type_verify(result, pcode_spec)
  return result

def _transform_stmt(stmt, ctx: dict):
  match stmt:
    case If(branches): return If(tuple((_transform_uop(c, ctx) if c is not None else None, tuple(_transform_stmt(s, ctx) for s in b)) for c, b in branches))
    case For(var, start, end, body): return For(var, _transform_uop(start, ctx), _transform_uop(end, ctx), tuple(_transform_stmt(s, ctx) for s in body))
    case Lambda(name, params, body): return Lambda(name, params, _transform_uop(body, ctx) if isinstance(body, UOp) else tuple(_transform_stmt(s, ctx) for s in body))
    case Return(v): return Return(_transform_uop(v, ctx))
    case UOp(): return _transform_uop(stmt, ctx)
    case _: return stmt

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

def parse_transform(pcode: str, op_name: str | None = None) -> tuple:
  if op_name is not None: pcode = _apply_pseudocode_fixes(op_name, pcode)
  ctx: dict[str, DType] = {'SCC': dtypes.bool, 'VCC': dtypes.uint64, 'EXEC': dtypes.uint64,
                           'VDATA': dtypes.uint64, 'SDATA': dtypes.uint64, 'ADDR': dtypes.uint64, 'VDST': dtypes.uint32,
                           'ROUND_MODE': dtypes.uint32, 'ROUND_TOWARD_ZERO': dtypes.uint32, 'HW_REGISTERS': dtypes.uint32,
                           'SGPR': dtypes.uint32, 'VGPR': dtypes.uint32}  # register files are uint32 arrays
  return tuple(_transform_stmt(s, ctx) for s in parse(pcode))
