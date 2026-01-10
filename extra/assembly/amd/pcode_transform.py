# Transform parsed pcode CUSTOM ops to UOps using PatternMatcher
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, graph_rewrite
from tinygrad.dtype import dtypes, DType
from extra.assembly.amd.pcode_parse import parse, If, For, Lambda, Break, Return

# ═══════════════════════════════════════════════════════════════════════════════
# TYPE MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════════

_DT_SUFFIX = {
  'f16': dtypes.float16, 'f32': dtypes.float32, 'f64': dtypes.float64, 'bf16': dtypes.bfloat16,
  'i8': dtypes.int8, 'i16': dtypes.int16, 'i32': dtypes.int32, 'i64': dtypes.int64,
  'u8': dtypes.uint8, 'u16': dtypes.uint16, 'u32': dtypes.uint32, 'u64': dtypes.uint64,
}
# Special conversions needing custom handling - excluded from auto-generated casts
_SPECIAL_CASTS = {'f32_to_u32', 'f64_to_u32', 'f16_to_u32', 'f32_to_u64', 'f64_to_u64', 'f16_to_u64',  # clamping
                  'bf16_to_f32', 'u32_to_u16', 'i32_to_i16'}  # bit manipulation
# Auto-generate all {src}_to_{dst} and v_cvt_{dst}_{src} cast mappings
_CAST_MAP = {f'{s}_to_{d}': _DT_SUFFIX[d] for s in _DT_SUFFIX for d in _DT_SUFFIX if s != d and f'{s}_to_{d}' not in _SPECIAL_CASTS}
_CAST_MAP.update({f'v_cvt_{d}_{s}': _DT_SUFFIX[d] for s in _DT_SUFFIX for d in _DT_SUFFIX if s != d and f'{s}_to_{d}' not in _SPECIAL_CASTS})

# Remaining CUSTOM ops that need dtype inference (not transformed to real UOps)
_BOOL_FNS = {'isDENORM', 'isQuietNAN', 'isSignalNAN', 'isEven', 'LT_NEG_ZERO', 'GT_NEG_ZERO'}
_U32_FNS = {'sign', 'exponent', 'ABSDIFF', 'SAT8', 'BYTE_PERMUTE', 'count_ones', 'countbits', 'reverse_bits',
            'u8_to_u32', 'u4_to_u32', 's_ff1_i32_b32', 's_ff1_i32_b64', 'v_sad_u8', 'v_msad_u8'}
_CVT_FNS = {'f32_to_u32': dtypes.uint32, 'f64_to_u32': dtypes.uint32, 'signext_from_bit': dtypes.int64,
            'f16_to_snorm': dtypes.int16, 'f16_to_unorm': dtypes.uint16, 'f32_to_snorm': dtypes.int16, 'f32_to_unorm': dtypes.uint16}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _typed_const(src: UOp, val) -> UOp:
  return UOp.const(src.dtype, val) if src.dtype != dtypes.void else UOp(Ops.CONST, dtypes.void, (src,), val)

def _floor(x: UOp, dt: DType) -> UOp:
  trunc = UOp(Ops.TRUNC, dt, (x,))
  return UOp(Ops.WHERE, dt, (UOp(Ops.CMPLT, dtypes.bool, (x, trunc)), UOp(Ops.SUB, dt, (trunc, _typed_const(x, 1))), trunc))

def _minmax(a: UOp, b: UOp, is_min: bool, dt: DType|None = None) -> UOp:
  dt = dt or (a.dtype if a.dtype != dtypes.void else b.dtype)
  return UOp(Ops.WHERE, dt, (UOp(Ops.CMPLT, dtypes.bool, (a, b) if is_min else (b, a)), a, b))

def _minmax3(a: UOp, b: UOp, c: UOp, is_min: bool, dt: DType|None = None) -> UOp:
  dt = dt or (a.dtype if a.dtype != dtypes.void else b.dtype if b.dtype != dtypes.void else c.dtype)
  return _minmax(_minmax(a, b, is_min, dt), c, is_min, dt)

def _first_nonvoid(*srcs: UOp) -> DType:
  return next((s.dtype for s in srcs if s.dtype != dtypes.void), dtypes.void)

def _var_name(u: UOp) -> str|None:
  if u.op == Ops.DEFINE_VAR: return u.arg[0] if isinstance(u.arg, tuple) else u.arg
  if u.op == Ops.CUSTOMI and u.src[0].op == Ops.DEFINE_VAR: return _var_name(u.src[0])
  return None

# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

def _typed_minmax2(a, b, op):
  if not isinstance(op.arg, str) or not (op.arg.startswith('v_min_') or op.arg.startswith('v_max_')): return None
  if (suffix := op.arg.split('_')[-1]) not in _DT_SUFFIX: return None
  return _minmax(a, b, op.arg.startswith('v_min_'), _DT_SUFFIX[suffix])

def _typed_minmax3(a, b, c, op):
  if not isinstance(op.arg, str) or not (op.arg.startswith('v_min3_') or op.arg.startswith('v_max3_')): return None
  if (suffix := op.arg.split('_')[-1]) not in _DT_SUFFIX: return None
  return _minmax3(a, b, c, op.arg.startswith('v_min3_'), _DT_SUFFIX[suffix])

def _typed_cast(x, op):
  return UOp(Ops.CAST, _CAST_MAP[op.arg], (x,)) if op.arg in _CAST_MAP else None

# Variable type tracking/propagation
def _track_var(ctx, u):
  if ctx is None or u.dtype == dtypes.void: return None
  name = u.arg[0] if isinstance(u.arg, tuple) else u.arg
  if name in ctx: assert ctx[name] == u.dtype, f"variable '{name}' declared with conflicting types: {ctx[name]} vs {u.dtype}"
  else: ctx[name] = u.dtype
  return None

def _prop_var(ctx, u):
  if ctx is None: return None
  name = u.arg[0] if isinstance(u.arg, tuple) else u.arg
  return UOp(Ops.DEFINE_VAR, ctx[name], arg=u.arg) if name in ctx else None

def _prop_assign(ctx, lhs, rhs):
  if ctx is None or rhs.dtype == dtypes.void or lhs.op != Ops.DEFINE_VAR: return None
  if (name := _var_name(lhs)) is None or name in ctx: return None
  ctx[name] = rhs.dtype
  return UOp(Ops.ASSIGN, dtypes.void, (UOp(Ops.DEFINE_VAR, rhs.dtype, arg=lhs.arg), rhs))

# Dtype propagation for void-typed ops
def _prop_binop(l, r, __OP__, **kw):
  dt = l.dtype if l.dtype != dtypes.void else r.dtype
  return UOp(__OP__.op, dt, (l, r), kw.get('arg')) if dt != dtypes.void else None

def _prop_unop(x, __OP__, **kw):
  return UOp(__OP__.op, x.dtype, (x,), kw.get('arg')) if x.dtype != dtypes.void else None

def _prop_mulacc(a, b, c, **kw):
  return UOp(Ops.MULACC, c.dtype, (a, b, c), kw.get('arg')) if c.dtype != dtypes.void else None

def _prop_where(cond, t, f, **kw):
  dt = _first_nonvoid(t, f)
  return UOp(Ops.WHERE, dt, (cond, t, f), kw.get('arg')) if dt != dtypes.void else None

def _prop_cat(src, **kw):
  total_bits = sum(p.dtype.itemsize * 8 for p in src if p.dtype != dtypes.void)
  dt = dtypes.uint64 if total_bits > 32 else dtypes.uint32 if total_bits > 0 else dtypes.void
  return UOp(Ops.CAT, dt, src, kw.get('arg')) if dt != dtypes.void else None

def _prop_customi(base, hi, lo, **kw):
  if hi is lo:  # array element access
    if base.dtype == dtypes.void: return None
    dt = base.dtype.scalar() if base.dtype.count > 1 else base.dtype
  else:  # slice - infer from bounds
    dt = dtypes.uint64 if hi.op == Ops.CONST and lo.op == Ops.CONST and abs(int(hi.arg) - int(lo.arg)) + 1 > 32 else dtypes.uint32
  return UOp(Ops.CUSTOMI, dt, (base, hi, lo), kw.get('arg'))

def _prop_custom(src, arg, **kw):
  if arg in _BOOL_FNS: dt = dtypes.bool
  elif arg in _U32_FNS: dt = dtypes.uint32
  elif arg in _CVT_FNS: dt = _CVT_FNS[arg]
  elif arg == 'trig_preop_result': dt = dtypes.float64
  else: dt = _first_nonvoid(*src) if src else dtypes.void
  return UOp(Ops.CUSTOM, dt, src, arg) if dt != dtypes.void else None

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
  (UPat(Ops.CUSTOM, arg='floor', src=(_fpat,)), lambda x: _floor(x, x.dtype)),
  (UPat(Ops.CUSTOM, arg='fract', src=(_fpat,)), lambda x: UOp(Ops.SUB, x.dtype, (x, _floor(x, x.dtype)))),
  (UPat(Ops.CUSTOM, arg='rsqrt', src=(_fpat,)), lambda x: UOp(Ops.RECIPROCAL, x.dtype, (UOp(Ops.SQRT, x.dtype, (x,)),))),
  # Boolean functions
  (UPat(Ops.CUSTOM, arg='isNAN', src=(UPat.var('x'),)), lambda x: UOp(Ops.CMPNE, dtypes.bool, (x, x))),
  (UPat(Ops.CUSTOM, arg='isINF', src=(UPat.var('x'),)), lambda x: UOp(Ops.OR, dtypes.bool, (
    UOp(Ops.CMPEQ, dtypes.bool, (x, _typed_const(x, float('inf')))), UOp(Ops.CMPEQ, dtypes.bool, (x, _typed_const(x, float('-inf'))))))),
  # min/max
  (UPat(Ops.CUSTOM, arg='min', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _minmax(a, b, True)),
  (UPat(Ops.CUSTOM, arg='max', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _minmax(a, b, False)),
  (UPat(Ops.CUSTOM, arg='clamp', src=(UPat.var('x'), UPat.var('lo'), UPat.var('hi'))), lambda x, lo, hi: _minmax(_minmax(x, lo, False), hi, True)),
  (UPat(Ops.CUSTOM, src=(UPat.var('a'), UPat.var('b')), name='op'), _typed_minmax2),
  (UPat(Ops.CUSTOM, src=(UPat.var('a'), UPat.var('b'), UPat.var('c')), name='op'), _typed_minmax3),
  # Type conversions
  (UPat(Ops.CUSTOM, src=(UPat.var('x'),), name='op'), _typed_cast),
  (UPat(Ops.CUSTOM, arg='signext', src=(UPat.var('x', dtype=dtypes.ints),)), lambda x: UOp(Ops.CAST, dtypes.int64, (x,))),
  (UPat(Ops.CUSTOM, arg='bf16_to_f32', src=(UPat.var('x', dtype=dtypes.bfloat16),)),
   lambda x: UOp(Ops.BITCAST, dtypes.float32, (UOp(Ops.SHL, dtypes.uint32, (UOp(Ops.CAST, dtypes.uint32, (x,)), UOp.const(dtypes.uint32, 16))),))),
  (UPat(Ops.CUSTOM, arg='u32_to_u16', src=(UPat.var('x', dtype=dtypes.uint32),)), lambda x: UOp(Ops.AND, dtypes.uint32, (x, UOp.const(dtypes.uint32, 0xffff)))),
  (UPat(Ops.CUSTOM, arg='i32_to_i16', src=(UPat.var('x', dtype=dtypes.int32),)),
   lambda x: UOp(Ops.CAST, dtypes.int16, (UOp(Ops.AND, dtypes.uint32, (UOp(Ops.CAST, dtypes.uint32, (x,)), UOp.const(dtypes.uint32, 0xffff))),))),
]) + PatternMatcher([
  # Variable type tracking and propagation
  (UPat(Ops.DEFINE_VAR, name='u'), _track_var),
  (UPat(Ops.DEFINE_VAR, dtype=dtypes.void, name='u'), _prop_var),
  (UPat(Ops.ASSIGN, src=(UPat(Ops.DEFINE_VAR, dtype=dtypes.void, name='lhs'), UPat.var('rhs'))), _prop_assign),
  # Dtype propagation for void-typed ops
  (UPat((Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.AND, Ops.OR, Ops.XOR, Ops.SHL, Ops.SHR, Ops.MOD, Ops.POW),
        dtype=dtypes.void, src=(UPat.var('l'), UPat.var('r')), name='__OP__'), _prop_binop),
  (UPat((Ops.NEG, Ops.TRUNC, Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.RECIPROCAL),
        dtype=dtypes.void, src=(UPat.var('x'),), name='__OP__'), _prop_unop),
  (UPat(Ops.MULACC, dtype=dtypes.void, src=(UPat.var('a'), UPat.var('b'), UPat.var('c'))), _prop_mulacc),
  (UPat(Ops.WHERE, dtype=dtypes.void, src=(UPat.var('cond'), UPat.var('t'), UPat.var('f'))), _prop_where),
  (UPat(Ops.CAT, dtype=dtypes.void, src=UPat.var('src')), _prop_cat),
  (UPat(Ops.CUSTOMI, dtype=dtypes.void, src=(UPat.var('base'), UPat.var('hi'), UPat.var('lo'))), _prop_customi),
  (UPat(Ops.CUSTOM, dtype=dtypes.void, src=UPat.var('src'), arg=UPat.var('arg')), _prop_custom),
])

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORM
# ═══════════════════════════════════════════════════════════════════════════════

def _transform_uop(u: UOp, ctx: dict) -> UOp: return graph_rewrite(u, pcode_pm, ctx=ctx)

def _transform_stmt(stmt, ctx: dict):
  match stmt:
    case If(branches): return If(tuple((_transform_uop(c, ctx) if c is not None else None, tuple(_transform_stmt(s, ctx) for s in b)) for c, b in branches))
    case For(var, start, end, body): return For(var, _transform_uop(start, ctx), _transform_uop(end, ctx), tuple(_transform_stmt(s, ctx) for s in body))
    case Lambda(name, params, body): return Lambda(name, params, _transform_uop(body, ctx) if isinstance(body, UOp) else tuple(_transform_stmt(s, ctx) for s in body))
    case Return(v): return Return(_transform_uop(v, ctx))
    case UOp(): return _transform_uop(stmt, ctx)
    case _: return stmt

def parse_transform(pcode: str) -> tuple:
  ctx: dict[str, DType] = {}
  return tuple(_transform_stmt(s, ctx) for s in parse(pcode))
