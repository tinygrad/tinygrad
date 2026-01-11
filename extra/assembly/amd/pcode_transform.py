# Transform parsed pcode CUSTOM ops to UOps using PatternMatcher
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, graph_rewrite
from tinygrad.uop.spec import shared_spec, type_verify
from tinygrad.dtype import dtypes, DType
from extra.assembly.amd.pcode_parse import parse, If, For, Lambda, Break, Return
import math

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

# CUSTOM op return types
_CUSTOM_TYPES = {
  'isDENORM': dtypes.bool, 'isQuietNAN': dtypes.bool, 'isSignalNAN': dtypes.bool, 'isEven': dtypes.bool,
  'LT_NEG_ZERO': dtypes.bool, 'GT_NEG_ZERO': dtypes.bool,
  'sign': dtypes.uint32, 'exponent': dtypes.uint32, 'ABSDIFF': dtypes.uint32, 'SAT8': dtypes.uint32,
  'BYTE_PERMUTE': dtypes.uint32, 'count_ones': dtypes.uint32, 'countbits': dtypes.uint32, 'reverse_bits': dtypes.uint32,
  'u8_to_u32': dtypes.uint32, 'u4_to_u32': dtypes.uint32, 's_ff1_i32_b32': dtypes.uint32, 's_ff1_i32_b64': dtypes.uint32,
  'v_sad_u8': dtypes.uint32, 'v_msad_u8': dtypes.uint32, 'ConvertFromFormat': dtypes.uint32, 'nop': dtypes.uint32,
  'f32_to_u32': dtypes.uint32, 'f64_to_u32': dtypes.uint32, 'signext_from_bit': dtypes.int64,
  'f16_to_snorm': dtypes.int16, 'f16_to_unorm': dtypes.uint16, 'f32_to_snorm': dtypes.int16, 'f32_to_unorm': dtypes.uint16,
  'trig_preop_result': dtypes.float64,
}
# Constants that get replaced with CONST
_CONSTS = {
  'PI': (dtypes.float64, math.pi), 'INF': (dtypes.float64, math.inf),
  'MAX_FLOAT_F32': (dtypes.float32, 3.4028235e+38), 'MAX_FLOAT_F64': (dtypes.float64, 1.7976931348623157e+308),
  'OVERFLOW_F32': (dtypes.float32, math.inf), 'OVERFLOW_F64': (dtypes.float64, math.inf),
  'UNDERFLOW_F32': (dtypes.float32, 0.0), 'UNDERFLOW_F64': (dtypes.float64, 0.0),
}
# Well-known register types
_REGS = {'SCC': dtypes.bool, 'VCC': dtypes.uint64, 'EXEC': dtypes.uint64, 'VDATA': dtypes.uint64, 'SDATA': dtypes.uint64,
         'ADDR': dtypes.uint64, 'VDST': dtypes.uint32, 'ROUND_MODE': dtypes.uint32, 'ROUND_TOWARD_ZERO': dtypes.uint32,
         'HW_REGISTERS': dtypes.uint32, 'SGPR': dtypes.uint32, 'VGPR': dtypes.uint32}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _tc(src: UOp, val) -> UOp:  # typed const
  return UOp.const(src.dtype, val) if src.dtype != dtypes.void else UOp(Ops.CONST, dtypes.void, (src,), val)

def _floor(x: UOp) -> UOp:
  trunc = UOp(Ops.TRUNC, x.dtype, (x,))
  return UOp(Ops.WHERE, x.dtype, (UOp(Ops.CMPLT, dtypes.bool, (x, trunc)), UOp(Ops.SUB, x.dtype, (trunc, _tc(x, 1))), trunc))

def _minmax(a: UOp, b: UOp, is_min: bool, dt: DType|None = None) -> UOp:
  dt = dt or (a.dtype if a.dtype != dtypes.void else b.dtype)
  return UOp(Ops.WHERE, dt, (UOp(Ops.CMPLT, dtypes.bool, (a, b) if is_min else (b, a)), a, b))

def _vn(u: UOp) -> str|None:  # var name
  if u.op == Ops.DEFINE_VAR: return u.arg[0] if isinstance(u.arg, tuple) else u.arg
  return _vn(u.src[0]) if u.op == Ops.CUSTOMI and u.src[0].op == Ops.DEFINE_VAR else None

def _typed_const(src: UOp, val) -> UOp:  # const matching src type
  return UOp.const(src.dtype, val) if src.dtype != dtypes.void else UOp(Ops.CONST, dtypes.void, (src,), val)

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
  return UOp(Ops.DEFINE_VAR, ctx[name], arg=u.arg) if name in ctx else None

def _prop_assign(ctx, lhs, rhs):
  if ctx is None or rhs.dtype == dtypes.void or lhs.op != Ops.DEFINE_VAR: return None
  if (name := _vn(lhs)) is None or name in ctx: return None
  ctx[name] = rhs.dtype
  return UOp(Ops.ASSIGN, rhs.dtype, (UOp(Ops.DEFINE_VAR, rhs.dtype, arg=lhs.arg), rhs))

def _prop_binop(l, r, __OP__):
  if __OP__.op in {Ops.SHL, Ops.SHR}: dt = l.dtype if l.dtype != dtypes.void else r.dtype
  elif l.dtype != dtypes.void and r.dtype != dtypes.void: dt = l.dtype if l.dtype.itemsize >= r.dtype.itemsize else r.dtype
  else: dt = l.dtype if l.dtype != dtypes.void else r.dtype
  return UOp(__OP__.op, dt, (l, r), __OP__.arg) if dt != dtypes.void else None

def _prop_custom(x):
  if x.arg in _CUSTOM_TYPES: return UOp(Ops.CUSTOM, _CUSTOM_TYPES[x.arg], x.src, x.arg)
  if x.arg in {'MEM', 'abs', 'cvtToQuietNAN'}: return None  # wrapped by BITCAST/CAST
  dt = next((s.dtype for s in x.src if s.dtype != dtypes.void), dtypes.void)
  assert dt != dtypes.void, f"cannot infer type for CUSTOM op '{x.arg}'"
  return UOp(Ops.CUSTOM, dt, x.src, x.arg)

def _prop_customi(base, hi, lo):
  if hi is lo: return UOp(Ops.CUSTOMI, base.dtype if base.dtype != dtypes.void else dtypes.uint32, (base, hi, lo))
  if hi.op == Ops.CONST and lo.op == Ops.CONST:
    return UOp(Ops.CUSTOMI, dtypes.uint64 if abs(int(hi.arg) - int(lo.arg)) + 1 > 32 else dtypes.uint32, (base, hi, lo))
  return UOp(Ops.CUSTOMI, dtypes.uint32, (base, hi, lo))

def _fix_binop(op, x, y):
  if x.dtype == dtypes.void or y.dtype == dtypes.void or x.dtype == y.dtype: return None
  if x.dtype.itemsize >= y.dtype.itemsize: return UOp(op.op, op.dtype, (x, UOp(Ops.CAST, x.dtype, (y,))), op.arg)
  return UOp(op.op, op.dtype, (UOp(Ops.CAST, y.dtype, (x,)), y), op.arg)

def _backprop(op, v, t):
  if t.dtype == dtypes.void: return None
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

def _typed_cast(x, op):
  if op.arg not in _CAST_MAP: return None
  return UOp(Ops.CAST, _CAST_MAP[op.arg], (x,))

# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN MATCHER
# ═══════════════════════════════════════════════════════════════════════════════

_fpat = UPat.var('x', dtype=dtypes.floats)

pcode_pm = PatternMatcher([
  # Float ops (preserve input type)
  (UPat(Ops.CUSTOM, arg='trunc', src=(_fpat,)), lambda x: UOp(Ops.TRUNC, x.dtype, (x,))),
  (UPat(Ops.CUSTOM, arg='sqrt', src=(_fpat,)), lambda x: UOp(Ops.SQRT, x.dtype, (x,))),
  (UPat(Ops.CUSTOM, arg='exp2', src=(_fpat,)), lambda x: UOp(Ops.EXP2, x.dtype, (x,))),
  (UPat(Ops.CUSTOM, arg='log2', src=(_fpat,)), lambda x: UOp(Ops.LOG2, x.dtype, (x,))),
  (UPat(Ops.CUSTOM, arg='sin', src=(_fpat,)), lambda x: UOp(Ops.SIN, x.dtype, (x,))),
  (UPat(Ops.CUSTOM, arg='rcp', src=(_fpat,)), lambda x: UOp(Ops.RECIPROCAL, x.dtype, (x,))),
  (UPat(Ops.CUSTOM, arg='fma', src=(_fpat, UPat.var('b'), UPat.var('c'))), lambda x, b, c: UOp(Ops.MULACC, x.dtype, (x, b, c))),
  (UPat(Ops.CUSTOM, arg='abs', src=(_fpat,)), lambda x: UOp(Ops.WHERE, x.dtype, (UOp(Ops.CMPLT, dtypes.bool, (x, _typed_const(x, 0))), UOp(Ops.NEG, x.dtype, (x,)), x))),
  (UPat(Ops.CUSTOM, arg='cos', src=(_fpat,)), lambda x: UOp(Ops.SIN, x.dtype, (UOp(Ops.ADD, x.dtype, (x, _typed_const(x, 1.5707963267948966))),))),
  (UPat(Ops.CUSTOM, arg='floor', src=(_fpat,)), lambda x: _floor(x)),
  (UPat(Ops.CUSTOM, arg='fract', src=(_fpat,)), lambda x: UOp(Ops.SUB, x.dtype, (x, _floor(x)))),
  (UPat(Ops.CUSTOM, arg='rsqrt', src=(_fpat,)), lambda x: UOp(Ops.RECIPROCAL, x.dtype, (UOp(Ops.SQRT, x.dtype, (x,)),))),
  # Boolean functions
  (UPat(Ops.CUSTOM, arg='isNAN', src=(UPat.var('x'),)), lambda x: UOp(Ops.CMPNE, dtypes.bool, (x, x))),
  (UPat(Ops.CUSTOM, arg='isINF', src=(UPat.var('x'),)), lambda x: UOp(Ops.OR, dtypes.bool, (
    UOp(Ops.CMPEQ, dtypes.bool, (x, _typed_const(x, float('inf')))), UOp(Ops.CMPEQ, dtypes.bool, (x, _typed_const(x, float('-inf'))))))),
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
]) + PatternMatcher([
  # Math constants
  (UPat(Ops.DEFINE_VAR, arg=('PI', None, None)), lambda: UOp.const(dtypes.float64, 3.141592653589793)),
  (UPat(Ops.DEFINE_VAR, arg=('INF', None, None)), lambda: UOp.const(dtypes.float64, float('inf'))),
  # Float special values
  (UPat(Ops.DEFINE_VAR, arg=('MAX_FLOAT_F32', None, None)), lambda: UOp.const(dtypes.float32, 3.4028235e+38)),
  (UPat(Ops.DEFINE_VAR, arg=('MAX_FLOAT_F64', None, None)), lambda: UOp.const(dtypes.float64, 1.7976931348623157e+308)),
  (UPat(Ops.DEFINE_VAR, arg=('OVERFLOW_F32', None, None)), lambda: UOp.const(dtypes.float32, float('inf'))),
  (UPat(Ops.DEFINE_VAR, arg=('OVERFLOW_F64', None, None)), lambda: UOp.const(dtypes.float64, float('inf'))),
  (UPat(Ops.DEFINE_VAR, arg=('UNDERFLOW_F32', None, None)), lambda: UOp.const(dtypes.float32, 0.0)),
  (UPat(Ops.DEFINE_VAR, arg=('UNDERFLOW_F64', None, None)), lambda: UOp.const(dtypes.float64, 0.0)),
  # Variable type tracking and propagation
  (UPat(Ops.DEFINE_VAR, name='u'), _track_var),
  (UPat(Ops.DEFINE_VAR, dtype=dtypes.void, name='u'), _prop_var),
  (UPat(Ops.ASSIGN, src=(UPat(Ops.DEFINE_VAR, dtype=dtypes.void, name='lhs'), UPat.var('rhs'))), _prop_assign),
  # Propagate dtype for ASSIGN from rhs, or infer rhs dtype from lhs if rhs is void
  (UPat(Ops.ASSIGN, dtype=dtypes.void, src=(UPat.var('lhs'), UPat.var('rhs'))),
   lambda lhs, rhs: UOp(Ops.ASSIGN, rhs.dtype, (lhs, rhs)) if rhs.dtype != dtypes.void else
                    UOp(Ops.ASSIGN, lhs.dtype, (lhs, rhs.replace(dtype=lhs.dtype))) if lhs.dtype != dtypes.void else None),
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
  (UPat(Ops.CAT, dtype=dtypes.void, name='x'), _prop_cat),
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
   lambda op, x, y: UOp(op.op, op.dtype, (x, UOp(Ops.CAST, x.dtype, (y,)))) if x.dtype != dtypes.void and y.dtype != dtypes.void and x.dtype != y.dtype and x.dtype.itemsize >= y.dtype.itemsize else None),
  (UPat((Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.AND, Ops.OR, Ops.XOR), src=(UPat.var('x'), UPat.var('y')), name='op'),
   lambda op, x, y: UOp(op.op, op.dtype, (UOp(Ops.CAST, y.dtype, (x,)), y)) if x.dtype != dtypes.void and y.dtype != dtypes.void and x.dtype != y.dtype and y.dtype.itemsize > x.dtype.itemsize else None),
  # Back-propagate types to void DEFINE_VAR sources
  (UPat((Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.AND, Ops.OR, Ops.XOR),
        src=(UPat(Ops.DEFINE_VAR, dtype=dtypes.void, name='v'), UPat.var('t')), name='op'),
   lambda op, v, t: _backprop(op, v, t)),
  (UPat((Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.AND, Ops.OR, Ops.XOR),
        src=(UPat.var('t'), UPat(Ops.DEFINE_VAR, dtype=dtypes.void, name='v')), name='op'),
   lambda op, t, v: _backprop(op, v, t)),
])

# ═══════════════════════════════════════════════════════════════════════════════
# PCODE SPEC (extends shared_spec with pcode-specific patterns)
# ═══════════════════════════════════════════════════════════════════════════════

pcode_spec = PatternMatcher([
  # DEFINE_VAR: pcode uses string names, not (name, min, max) tuples with ints
  (UPat(Ops.DEFINE_VAR, name="x"), lambda x: isinstance(x.arg, (str, tuple))),
  # ASSIGN: dtype matches rhs (unless both void)
  (UPat(Ops.ASSIGN, src=(UPat.var("lhs"), UPat.var("rhs")), name="a"),
   lambda a, lhs, rhs: a.dtype == rhs.dtype and (rhs.dtype != dtypes.void or lhs.dtype == dtypes.void)),
  # BITCAST: void source allowed (type view on untyped register)
  (UPat(Ops.BITCAST, src=(UPat(),)), lambda: True),
  # CUSTOMI/CAT: must be typed (slice bounds or bit concat determine type)
  (UPat(Ops.CUSTOMI, name="x"), lambda x: x.dtype != dtypes.void),
  (UPat(Ops.CAT, name="x"), lambda x: x.dtype != dtypes.void),
  # CUSTOM: MEM and passthrough ops (abs, cvtToQuietNAN) can be void (wrapped by BITCAST/CAST)
  (UPat(Ops.CUSTOM, name="x"), lambda x: x.dtype != dtypes.void or x.arg in {'MEM', 'abs', 'cvtToQuietNAN'}),
  # POW allows int exponent with float base
  (UPat(Ops.POW, dtype=dtypes.floats, src=(UPat(dtype=dtypes.floats), UPat(dtype=dtypes.ints))), lambda: True),
]) + shared_spec

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORM
# ═══════════════════════════════════════════════════════════════════════════════

def _transform_uop(u: UOp, ctx: dict) -> UOp:
  result = graph_rewrite(u, pcode_pm, ctx=ctx)
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

def parse_transform(pcode: str) -> tuple:
  ctx: dict[str, DType] = {'SCC': dtypes.bool, 'VCC': dtypes.uint64, 'EXEC': dtypes.uint64,
                           'VDATA': dtypes.uint64, 'SDATA': dtypes.uint64, 'ADDR': dtypes.uint64, 'VDST': dtypes.uint32,
                           'ROUND_MODE': dtypes.uint32, 'ROUND_TOWARD_ZERO': dtypes.uint32, 'HW_REGISTERS': dtypes.uint32,
                           'SGPR': dtypes.uint32, 'VGPR': dtypes.uint32}  # register files are uint32 arrays
  return tuple(_transform_stmt(s, ctx) for s in parse(pcode))
