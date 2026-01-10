# Transform parsed pcode CUSTOM ops to UOps using PatternMatcher
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, graph_rewrite
from tinygrad.dtype import dtypes, DType
from extra.assembly.amd.pcode_parse import parse, If, For, Lambda, Break, Return

def _typed_const(src: UOp, val) -> UOp:
  """Create a const with same dtype as src, or a deferred const if src.dtype is void."""
  return UOp.const(src.dtype, val) if src.dtype != dtypes.void else UOp(Ops.CONST, dtypes.void, (src,), val)

def _floor(x: UOp, dt: DType) -> UOp:
  trunc = UOp(Ops.TRUNC, dt, (x,))
  return UOp(Ops.WHERE, dt, (UOp(Ops.CMPLT, dtypes.bool, (x, trunc)), UOp(Ops.SUB, dt, (trunc, _typed_const(x, 1))), trunc))

def _minmax(a: UOp, b: UOp, is_min: bool, dt: DType|None = None) -> UOp:
  if dt is None: dt = a.dtype if a.dtype != dtypes.void else b.dtype
  cmp = UOp(Ops.CMPLT, dtypes.bool, (a, b) if is_min else (b, a))
  return UOp(Ops.WHERE, dt, (cmp, a, b))

def _minmax3(a: UOp, b: UOp, c: UOp, is_min: bool, dt: DType|None = None) -> UOp:
  if dt is None: dt = a.dtype if a.dtype != dtypes.void else b.dtype if b.dtype != dtypes.void else c.dtype
  return _minmax(_minmax(a, b, is_min, dt), c, is_min, dt)

# ═══════════════════════════════════════════════════════════════════════════════
# DTYPE PROPAGATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _first_nonvoid(*srcs: UOp) -> DType:
  """Get first non-void dtype from sources."""
  for s in srcs:
    if s.dtype != dtypes.void: return s.dtype
  return dtypes.void

def _infer_binop_dtype(l: UOp, r: UOp) -> DType:
  """Infer dtype for binary ops from operands."""
  return l.dtype if l.dtype != dtypes.void else r.dtype

def _infer_slice_dtype(hi: UOp, lo: UOp) -> DType:
  """Infer dtype for CUSTOMI slices from bounds."""
  if hi.op == Ops.CONST and lo.op == Ops.CONST:
    width = abs(int(hi.arg) - int(lo.arg)) + 1
    return dtypes.uint64 if width > 32 else dtypes.uint32
  return dtypes.uint32

def _infer_cat_dtype(*parts: UOp) -> DType:
  """Infer dtype for CAT from parts' bitsizes."""
  total_bits = sum(p.dtype.itemsize * 8 for p in parts if p.dtype != dtypes.void)
  return dtypes.uint64 if total_bits > 32 else dtypes.uint32 if total_bits > 0 else dtypes.void

# Function return type inference for remaining CUSTOM ops (ones not transformed by pcode_pm)
_BOOL_FNS = {'isDENORM', 'isQuietNAN', 'isSignalNAN', 'isEven', 'LT_NEG_ZERO', 'GT_NEG_ZERO'}
_U32_FNS = {'sign', 'exponent', 'ABSDIFF', 'SAT8', 'BYTE_PERMUTE', 'count_ones', 'countbits', 'reverse_bits',
            'u8_to_u32', 'u4_to_u32', 's_ff1_i32_b32', 's_ff1_i32_b64', 'v_sad_u8', 'v_msad_u8'}
_CVT_FNS = {
  'f32_to_u32': dtypes.uint32, 'f64_to_u32': dtypes.uint32,  # special clamping conversions, not simple casts
  'signext_from_bit': dtypes.int64,
  'f16_to_snorm': dtypes.int16, 'f16_to_unorm': dtypes.uint16, 'f32_to_snorm': dtypes.int16, 'f32_to_unorm': dtypes.uint16,
}

def _infer_custom_dtype(name: str, srcs: tuple[UOp, ...]) -> DType:
  """Infer output dtype for CUSTOM function calls."""
  if name in _BOOL_FNS: return dtypes.bool
  if name in _U32_FNS: return dtypes.uint32
  if name in _CVT_FNS: return _CVT_FNS[name]
  if name == 'trig_preop_result': return dtypes.float64
  # Passthrough functions inherit from first source
  return _first_nonvoid(*srcs) if srcs else dtypes.void

# Dtype propagation patterns - match void dtype and replace with inferred
def _prop_binop(l: UOp, r: UOp, **kw) -> UOp|None:
  dt = _infer_binop_dtype(l, r)
  if dt == dtypes.void: return None
  return UOp(kw['__OP__'].op, dt, (l, r), kw.get('arg'), kw.get('tag'))

def _prop_unop(x: UOp, **kw) -> UOp|None:
  if x.dtype == dtypes.void: return None
  return UOp(kw['__OP__'].op, x.dtype, (x,), kw.get('arg'), kw.get('tag'))

def _prop_mulacc(a: UOp, b: UOp, c: UOp, **kw) -> UOp|None:
  if c.dtype == dtypes.void: return None
  return UOp(Ops.MULACC, c.dtype, (a, b, c), kw.get('arg'), kw.get('tag'))

def _prop_where(cond: UOp, t: UOp, f: UOp, **kw) -> UOp|None:
  dt = _first_nonvoid(t, f)  # infer from true/false branches, not gate
  if dt == dtypes.void: return None
  return UOp(Ops.WHERE, dt, (cond, t, f), kw.get('arg'), kw.get('tag'))

def _prop_cat(src: tuple[UOp, ...], **kw) -> UOp|None:
  dt = _infer_cat_dtype(*src)
  if dt == dtypes.void: return None
  return UOp(Ops.CAT, dt, src, kw.get('arg'), kw.get('tag'))

def _prop_customi(base: UOp, hi: UOp, lo: UOp, **kw) -> UOp|None:
  # CUSTOMI(base, hi, lo) - for slices, infer from bounds; for array access, infer from base
  if hi is lo:  # single index (hi == lo) - array element access
    if base.dtype == dtypes.void: return None  # can't infer yet
    if base.dtype.count > 1: return UOp(Ops.CUSTOMI, base.dtype.scalar(), (base, hi, lo), kw.get('arg'), kw.get('tag'))
    return UOp(Ops.CUSTOMI, base.dtype, (base, hi, lo), kw.get('arg'), kw.get('tag'))
  # Slice - infer from bounds
  dt = _infer_slice_dtype(hi, lo)
  return UOp(Ops.CUSTOMI, dt, (base, hi, lo), kw.get('arg'), kw.get('tag'))

def _prop_custom(src: tuple[UOp, ...], arg, **kw) -> UOp|None:
  dt = _infer_custom_dtype(arg, src)
  if dt == dtypes.void: return None
  return UOp(Ops.CUSTOM, dt, src, arg, kw.get('tag'))

def _prop_var(ctx, u) -> UOp|None:
  """Propagate variable type from ctx declarations."""
  if ctx is None: return None
  name = u.arg[0] if isinstance(u.arg, tuple) else u.arg
  if name not in ctx: return None
  return UOp(Ops.DEFINE_VAR, ctx[name], arg=u.arg)

def _track_var(ctx, u) -> UOp|None:
  """Track typed DEFINE_VAR in ctx, assert on conflict."""
  if ctx is None: return None
  name = u.arg[0] if isinstance(u.arg, tuple) else u.arg
  if name in ctx:
    assert ctx[name] == u.dtype, f"variable '{name}' declared with conflicting types: {ctx[name]} vs {u.dtype}"
    return None  # already tracked, no change needed
  ctx[name] = u.dtype
  return None  # no rewrite, just tracking

def _get_var_name(u: UOp) -> str|None:
  """Extract variable name from a UOp (DEFINE_VAR or CUSTOMI array access)."""
  if u.op == Ops.DEFINE_VAR: return u.arg[0] if isinstance(u.arg, tuple) else u.arg
  if u.op == Ops.CUSTOMI and u.src[0].op == Ops.DEFINE_VAR: return _get_var_name(u.src[0])
  return None

def _prop_assign(ctx, lhs, rhs) -> UOp|None:
  """Infer LHS variable type from RHS in ASSIGN."""
  if ctx is None or rhs.dtype == dtypes.void: return None
  # Only infer for bare DEFINE_VAR (not BITCAST wrapped)
  if lhs.op != Ops.DEFINE_VAR: return None
  name = _get_var_name(lhs)
  if name is None or name in ctx: return None
  ctx[name] = rhs.dtype
  return UOp(Ops.ASSIGN, dtypes.void, (UOp(Ops.DEFINE_VAR, rhs.dtype, arg=lhs.arg), rhs))

# ═══════════════════════════════════════════════════════════════════════════════
# TYPE MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Float dtype pattern for matching float inputs
_fpat = UPat.var('x', dtype=dtypes.floats)

# Lookup tables for typed operations
_MINMAX_DTYPES = {'f16': dtypes.float16, 'f32': dtypes.float32, 'i16': dtypes.int16, 'i32': dtypes.int32, 'u16': dtypes.uint16, 'u32': dtypes.uint32}

# dtype suffix to DType mapping - generate all cast functions from this
_DT_SUFFIX = {
  'f16': dtypes.float16, 'f32': dtypes.float32, 'f64': dtypes.float64, 'bf16': dtypes.bfloat16,
  'i8': dtypes.int8, 'i16': dtypes.int16, 'i32': dtypes.int32, 'i64': dtypes.int64,
  'u8': dtypes.uint8, 'u16': dtypes.uint16, 'u32': dtypes.uint32, 'u64': dtypes.uint64,
}
# Special conversions that need custom handling (not simple casts) - excluded from _CAST_MAP
_SPECIAL_CASTS = {
  'f32_to_u32', 'f64_to_u32', 'f16_to_u32', 'f32_to_u64', 'f64_to_u64', 'f16_to_u64',  # clamping casts
  'bf16_to_f32',  # shift left by 16
  'u32_to_u16', 'i32_to_i16',  # masking casts
}
# Generate all {src}_to_{dst} cast mappings (excluding special casts)
_CAST_MAP = {f'{s}_to_{d}': (_DT_SUFFIX[s], _DT_SUFFIX[d]) for s in _DT_SUFFIX for d in _DT_SUFFIX
             if s != d and f'{s}_to_{d}' not in _SPECIAL_CASTS}
# Add v_cvt_* aliases
_CAST_MAP.update({f'v_cvt_{d}_{s}': (_DT_SUFFIX[s], _DT_SUFFIX[d]) for s in _DT_SUFFIX for d in _DT_SUFFIX
                  if s != d and f'{s}_to_{d}' not in _SPECIAL_CASTS})

# Handler for typed minmax ops - looks up dtype from arg name
def _typed_minmax2(a, b, op):
  arg = op.arg
  if not isinstance(arg, str) or not (arg.startswith('v_min_') or arg.startswith('v_max_')): return None
  suffix = arg.split('_')[-1]
  if suffix not in _MINMAX_DTYPES: return None
  return _minmax(a, b, arg.startswith('v_min_'), _MINMAX_DTYPES[suffix])

def _typed_minmax3(a, b, c, op):
  arg = op.arg
  if not isinstance(arg, str) or not (arg.startswith('v_min3_') or arg.startswith('v_max3_')): return None
  suffix = arg.split('_')[-1]
  if suffix not in _MINMAX_DTYPES: return None
  return _minmax3(a, b, c, arg.startswith('v_min3_'), _MINMAX_DTYPES[suffix])

# Handler for cast ops - looks up types from arg name
def _typed_cast(x, op):
  arg = op.arg
  if arg not in _CAST_MAP: return None
  return UOp(Ops.CAST, _CAST_MAP[arg][1], (x,))

# Transform CUSTOM ops to real UOps
pcode_pm = PatternMatcher([
  # Direct UOp mappings (float ops preserve input type)
  (UPat(Ops.CUSTOM, arg='trunc', src=(_fpat,)), lambda x: UOp(Ops.TRUNC, x.dtype, (x,))),
  (UPat(Ops.CUSTOM, arg='sqrt', src=(_fpat,)), lambda x: UOp(Ops.SQRT, x.dtype, (x,))),
  (UPat(Ops.CUSTOM, arg='exp2', src=(_fpat,)), lambda x: UOp(Ops.EXP2, x.dtype, (x,))),
  (UPat(Ops.CUSTOM, arg='log2', src=(_fpat,)), lambda x: UOp(Ops.LOG2, x.dtype, (x,))),
  (UPat(Ops.CUSTOM, arg='sin', src=(_fpat,)), lambda x: UOp(Ops.SIN, x.dtype, (x,))),
  (UPat(Ops.CUSTOM, arg='rcp', src=(_fpat,)), lambda x: UOp(Ops.RECIPROCAL, x.dtype, (x,))),
  (UPat(Ops.CUSTOM, arg='fma', src=(_fpat, UPat.var('b'), UPat.var('c'))), lambda x, b, c: UOp(Ops.MULACC, x.dtype, (x, b, c))),

  # Boolean functions
  (UPat(Ops.CUSTOM, arg='isNAN', src=(UPat.var('x'),)), lambda x: UOp(Ops.CMPNE, dtypes.bool, (x, x))),
  (UPat(Ops.CUSTOM, arg='isINF', src=(UPat.var('x'),)), lambda x: UOp(Ops.OR, dtypes.bool, (
    UOp(Ops.CMPEQ, dtypes.bool, (x, _typed_const(x, float('inf')))),
    UOp(Ops.CMPEQ, dtypes.bool, (x, _typed_const(x, float('-inf'))))))),

  # Math functions (float ops preserve input type)
  (UPat(Ops.CUSTOM, arg='abs', src=(_fpat,)), lambda x: UOp(Ops.WHERE, x.dtype, (
    UOp(Ops.CMPLT, dtypes.bool, (x, _typed_const(x, 0))), UOp(Ops.NEG, x.dtype, (x,)), x))),
  (UPat(Ops.CUSTOM, arg='cos', src=(_fpat,)), lambda x: UOp(Ops.SIN, x.dtype, (
    UOp(Ops.ADD, x.dtype, (x, _typed_const(x, 1.5707963267948966))),))),
  (UPat(Ops.CUSTOM, arg='floor', src=(_fpat,)), lambda x: _floor(x, x.dtype)),
  (UPat(Ops.CUSTOM, arg='fract', src=(_fpat,)), lambda x: UOp(Ops.SUB, x.dtype, (x, _floor(x, x.dtype)))),
  (UPat(Ops.CUSTOM, arg='rsqrt', src=(_fpat,)), lambda x: UOp(Ops.RECIPROCAL, x.dtype, (UOp(Ops.SQRT, x.dtype, (x,)),))),

  # min/max (2 args) - generic (inherits type from inputs)
  (UPat(Ops.CUSTOM, arg='min', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _minmax(a, b, True)),
  (UPat(Ops.CUSTOM, arg='max', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _minmax(a, b, False)),
  # min/max - typed versions (v_min_f32, v_max_i16, etc) - handler returns None if arg doesn't match
  (UPat(Ops.CUSTOM, src=(UPat.var('a'), UPat.var('b')), name='op'), _typed_minmax2),
  # min3/max3 - typed versions (v_min3_f32, v_max3_i16, etc)
  (UPat(Ops.CUSTOM, src=(UPat.var('a'), UPat.var('b'), UPat.var('c')), name='op'), _typed_minmax3),
  # clamp(x, lo, hi) = min(max(x, lo), hi)
  (UPat(Ops.CUSTOM, arg='clamp', src=(UPat.var('x'), UPat.var('lo'), UPat.var('hi'))), lambda x, lo, hi: _minmax(_minmax(x, lo, False), hi, True)),
  # Type conversions (lookup from _CAST_MAP) - handler returns None if arg doesn't match
  (UPat(Ops.CUSTOM, src=(UPat.var('x'),), name='op'), _typed_cast),
  # signext to int64
  (UPat(Ops.CUSTOM, arg='signext', src=(UPat.var('x', dtype=dtypes.ints),)), lambda x: UOp(Ops.CAST, dtypes.int64, (x,))),
  (UPat(Ops.CUSTOM, arg='bf16_to_f32', src=(UPat.var('x', dtype=dtypes.bfloat16),)), lambda x: UOp(Ops.BITCAST, dtypes.float32, (
    UOp(Ops.SHL, dtypes.uint32, (UOp(Ops.CAST, dtypes.uint32, (x,)), UOp.const(dtypes.uint32, 16))),))),
  (UPat(Ops.CUSTOM, arg='u32_to_u16', src=(UPat.var('x', dtype=dtypes.uint32),)), lambda x: UOp(Ops.AND, dtypes.uint32, (x, UOp.const(dtypes.uint32, 0xffff)))),
  (UPat(Ops.CUSTOM, arg='i32_to_i16', src=(UPat.var('x', dtype=dtypes.int32),)), lambda x: UOp(Ops.CAST, dtypes.int16, (
    UOp(Ops.AND, dtypes.uint32, (UOp(Ops.CAST, dtypes.uint32, (x,)), UOp.const(dtypes.uint32, 0xffff))),))),
]) + PatternMatcher([
  # Track typed DEFINE_VAR in ctx (must come before propagation)
  (UPat(Ops.DEFINE_VAR, name='u'), lambda ctx, u: _track_var(ctx, u) if u.dtype != dtypes.void else None),
  # Propagate variable type from ctx to void DEFINE_VAR
  (UPat(Ops.DEFINE_VAR, dtype=dtypes.void, name='u'), _prop_var),
  # Infer LHS type from RHS in ASSIGN
  (UPat(Ops.ASSIGN, src=(UPat(Ops.DEFINE_VAR, dtype=dtypes.void, name='lhs'), UPat.var('rhs'))), _prop_assign),
  # Dtype propagation patterns - match void dtype and replace with inferred
  # Binary ops with void dtype
  (UPat((Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.AND, Ops.OR, Ops.XOR, Ops.SHL, Ops.SHR, Ops.MOD, Ops.POW), dtype=dtypes.void,
        src=(UPat.var('l'), UPat.var('r')), name='__OP__'), _prop_binop),
  # Unary ops with void dtype
  (UPat((Ops.NEG, Ops.TRUNC, Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.RECIPROCAL), dtype=dtypes.void,
        src=(UPat.var('x'),), name='__OP__'), _prop_unop),
  # MULACC inherits from accumulator (third source)
  (UPat(Ops.MULACC, dtype=dtypes.void, src=(UPat.var('a'), UPat.var('b'), UPat.var('c'))), _prop_mulacc),
  # WHERE inherits from true/false branches
  (UPat(Ops.WHERE, dtype=dtypes.void, src=(UPat.var('cond'), UPat.var('t'), UPat.var('f'))), _prop_where),
  # CAT computes from parts
  (UPat(Ops.CAT, dtype=dtypes.void, src=UPat.var('src')), _prop_cat),
  # CUSTOMI (slices/array access)
  (UPat(Ops.CUSTOMI, dtype=dtypes.void, src=(UPat.var('base'), UPat.var('hi'), UPat.var('lo'))), _prop_customi),
  # CUSTOM functions
  (UPat(Ops.CUSTOM, dtype=dtypes.void, src=UPat.var('src'), arg=UPat.var('arg')), _prop_custom),
])

def _transform_uop(u: UOp, ctx: dict) -> UOp: return graph_rewrite(u, pcode_pm, ctx=ctx)

def _transform_stmt(stmt, ctx: dict):
  """Transform a statement, rewriting UOps and updating ctx."""
  match stmt:
    case If(branches):
      return If(tuple((_transform_uop(cond, ctx) if cond is not None else None, tuple(_transform_stmt(s, ctx) for s in body)) for cond, body in branches))
    case For(var, start, end, body):
      return For(var, _transform_uop(start, ctx), _transform_uop(end, ctx), tuple(_transform_stmt(s, ctx) for s in body))
    case Lambda(name, params, body):
      return Lambda(name, params, _transform_uop(body, ctx) if isinstance(body, UOp) else tuple(_transform_stmt(s, ctx) for s in body))
    case Return(v): return Return(_transform_uop(v, ctx))
    case UOp(): return _transform_uop(stmt, ctx)
    case _: return stmt

def parse_transform(pcode: str) -> tuple:
  """Parse pseudocode and transform CUSTOM ops to UOps."""
  ctx: dict[str, DType] = {}
  return tuple(_transform_stmt(s, ctx) for s in parse(pcode))
