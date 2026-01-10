# Transform parsed pcode CUSTOM ops to UOps using PatternMatcher
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, graph_rewrite
from tinygrad.dtype import dtypes, DType
from extra.assembly.amd.pcode_parse import parse, If, For, Lambda, Break, Return

def _typed_const(src: UOp, val) -> UOp:
  """Create a const with same dtype as src, or a deferred const if src.dtype is void."""
  return UOp.const(src.dtype, val) if src.dtype != dtypes.void else UOp(Ops.CONST, dtypes.void, (src,), val)

def _floor(x: UOp) -> UOp:
  trunc = UOp(Ops.TRUNC, dtypes.void, (x,))
  return UOp(Ops.WHERE, dtypes.void, (UOp(Ops.CMPLT, dtypes.bool, (x, trunc)), UOp(Ops.SUB, dtypes.void, (trunc, _typed_const(x, 1))), trunc))

def _minmax(a: UOp, b: UOp, is_min: bool) -> UOp:
  cmp = UOp(Ops.CMPLT, dtypes.bool, (a, b) if is_min else (b, a))
  return UOp(Ops.WHERE, dtypes.void, (cmp, a, b))

def _minmax3(a: UOp, b: UOp, c: UOp, is_min: bool) -> UOp:
  return _minmax(_minmax(a, b, is_min), c, is_min)

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
  'f32_to_u32': dtypes.uint32, 'f64_to_u32': dtypes.uint32,
  'f32_to_bf16': dtypes.bfloat16,
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
  # CUSTOMI(base, hi, lo) - for slices, infer from bounds; for array access, use uint32
  if hi is lo:  # single index (hi == lo)
    # Array element access - check if base has vector dtype
    if base.dtype != dtypes.void and base.dtype.count > 1:
      return UOp(Ops.CUSTOMI, base.dtype.scalar(), (base, hi, lo), kw.get('arg'), kw.get('tag'))
    return UOp(Ops.CUSTOMI, dtypes.uint32, (base, hi, lo), kw.get('arg'), kw.get('tag'))
  # Slice - infer from bounds
  dt = _infer_slice_dtype(hi, lo)
  return UOp(Ops.CUSTOMI, dt, (base, hi, lo), kw.get('arg'), kw.get('tag'))

def _prop_custom(src: tuple[UOp, ...], arg, **kw) -> UOp|None:
  dt = _infer_custom_dtype(arg, src)
  if dt == dtypes.void: return None
  return UOp(Ops.CUSTOM, dt, src, arg, kw.get('tag'))

# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED PATTERN MATCHER
# ═══════════════════════════════════════════════════════════════════════════════

# Transform CUSTOM ops to real UOps
pcode_pm = PatternMatcher([
  # Direct UOp mappings
  (UPat(Ops.CUSTOM, arg='trunc', src=(UPat.var('x'),)), lambda x: UOp(Ops.TRUNC, dtypes.void, (x,))),
  (UPat(Ops.CUSTOM, arg='sqrt', src=(UPat.var('x'),)), lambda x: UOp(Ops.SQRT, dtypes.void, (x,))),
  (UPat(Ops.CUSTOM, arg='exp2', src=(UPat.var('x'),)), lambda x: UOp(Ops.EXP2, dtypes.void, (x,))),
  (UPat(Ops.CUSTOM, arg='log2', src=(UPat.var('x'),)), lambda x: UOp(Ops.LOG2, dtypes.void, (x,))),
  (UPat(Ops.CUSTOM, arg='sin', src=(UPat.var('x'),)), lambda x: UOp(Ops.SIN, dtypes.void, (x,))),
  (UPat(Ops.CUSTOM, arg='rcp', src=(UPat.var('x'),)), lambda x: UOp(Ops.RECIPROCAL, dtypes.void, (x,))),
  (UPat(Ops.CUSTOM, arg='fma', src=(UPat.var('a'), UPat.var('b'), UPat.var('c'))), lambda a, b, c: UOp(Ops.MULACC, dtypes.void, (a, b, c))),

  # Boolean functions
  (UPat(Ops.CUSTOM, arg='isNAN', src=(UPat.var('x'),)), lambda x: UOp(Ops.CMPNE, dtypes.bool, (x, x))),
  (UPat(Ops.CUSTOM, arg='isINF', src=(UPat.var('x'),)), lambda x: UOp(Ops.OR, dtypes.bool, (
    UOp(Ops.CMPEQ, dtypes.bool, (x, _typed_const(x, float('inf')))),
    UOp(Ops.CMPEQ, dtypes.bool, (x, _typed_const(x, float('-inf'))))))),

  # Math functions
  (UPat(Ops.CUSTOM, arg='abs', src=(UPat.var('x'),)), lambda x: UOp(Ops.WHERE, dtypes.void, (
    UOp(Ops.CMPLT, dtypes.bool, (x, _typed_const(x, 0))), UOp(Ops.NEG, dtypes.void, (x,)), x))),
  (UPat(Ops.CUSTOM, arg='cos', src=(UPat.var('x'),)), lambda x: UOp(Ops.SIN, dtypes.void, (
    UOp(Ops.ADD, dtypes.void, (x, _typed_const(x, 1.5707963267948966))),))),
  (UPat(Ops.CUSTOM, arg='floor', src=(UPat.var('x'),)), lambda x: _floor(x)),
  (UPat(Ops.CUSTOM, arg='fract', src=(UPat.var('x'),)), lambda x: UOp(Ops.SUB, dtypes.void, (x, _floor(x)))),
  (UPat(Ops.CUSTOM, arg='rsqrt', src=(UPat.var('x'),)), lambda x: UOp(Ops.RECIPROCAL, dtypes.void, (UOp(Ops.SQRT, dtypes.void, (x,)),))),

  # min/max (2 args)
  (UPat(Ops.CUSTOM, arg='min', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _minmax(a, b, True)),
  (UPat(Ops.CUSTOM, arg='max', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _minmax(a, b, False)),
  (UPat(Ops.CUSTOM, arg='v_min_f16', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _minmax(a, b, True)),
  (UPat(Ops.CUSTOM, arg='v_min_f32', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _minmax(a, b, True)),
  (UPat(Ops.CUSTOM, arg='v_min_i16', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _minmax(a, b, True)),
  (UPat(Ops.CUSTOM, arg='v_min_i32', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _minmax(a, b, True)),
  (UPat(Ops.CUSTOM, arg='v_min_u16', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _minmax(a, b, True)),
  (UPat(Ops.CUSTOM, arg='v_min_u32', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _minmax(a, b, True)),
  (UPat(Ops.CUSTOM, arg='v_max_f16', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _minmax(a, b, False)),
  (UPat(Ops.CUSTOM, arg='v_max_f32', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _minmax(a, b, False)),
  (UPat(Ops.CUSTOM, arg='v_max_i16', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _minmax(a, b, False)),
  (UPat(Ops.CUSTOM, arg='v_max_i32', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _minmax(a, b, False)),
  (UPat(Ops.CUSTOM, arg='v_max_u16', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _minmax(a, b, False)),
  (UPat(Ops.CUSTOM, arg='v_max_u32', src=(UPat.var('a'), UPat.var('b'))), lambda a, b: _minmax(a, b, False)),

  # min3/max3 (3 args)
  (UPat(Ops.CUSTOM, arg='v_min3_f16', src=(UPat.var('a'), UPat.var('b'), UPat.var('c'))), lambda a, b, c: _minmax3(a, b, c, True)),
  (UPat(Ops.CUSTOM, arg='v_min3_f32', src=(UPat.var('a'), UPat.var('b'), UPat.var('c'))), lambda a, b, c: _minmax3(a, b, c, True)),
  (UPat(Ops.CUSTOM, arg='v_min3_i16', src=(UPat.var('a'), UPat.var('b'), UPat.var('c'))), lambda a, b, c: _minmax3(a, b, c, True)),
  (UPat(Ops.CUSTOM, arg='v_min3_i32', src=(UPat.var('a'), UPat.var('b'), UPat.var('c'))), lambda a, b, c: _minmax3(a, b, c, True)),
  (UPat(Ops.CUSTOM, arg='v_min3_u16', src=(UPat.var('a'), UPat.var('b'), UPat.var('c'))), lambda a, b, c: _minmax3(a, b, c, True)),
  (UPat(Ops.CUSTOM, arg='v_min3_u32', src=(UPat.var('a'), UPat.var('b'), UPat.var('c'))), lambda a, b, c: _minmax3(a, b, c, True)),
  (UPat(Ops.CUSTOM, arg='v_max3_f16', src=(UPat.var('a'), UPat.var('b'), UPat.var('c'))), lambda a, b, c: _minmax3(a, b, c, False)),
  (UPat(Ops.CUSTOM, arg='v_max3_f32', src=(UPat.var('a'), UPat.var('b'), UPat.var('c'))), lambda a, b, c: _minmax3(a, b, c, False)),
  (UPat(Ops.CUSTOM, arg='v_max3_i16', src=(UPat.var('a'), UPat.var('b'), UPat.var('c'))), lambda a, b, c: _minmax3(a, b, c, False)),
  (UPat(Ops.CUSTOM, arg='v_max3_i32', src=(UPat.var('a'), UPat.var('b'), UPat.var('c'))), lambda a, b, c: _minmax3(a, b, c, False)),
  (UPat(Ops.CUSTOM, arg='v_max3_u16', src=(UPat.var('a'), UPat.var('b'), UPat.var('c'))), lambda a, b, c: _minmax3(a, b, c, False)),
  (UPat(Ops.CUSTOM, arg='v_max3_u32', src=(UPat.var('a'), UPat.var('b'), UPat.var('c'))), lambda a, b, c: _minmax3(a, b, c, False)),

  # clamp(x, lo, hi) = min(max(x, lo), hi)
  (UPat(Ops.CUSTOM, arg='clamp', src=(UPat.var('x'), UPat.var('lo'), UPat.var('hi'))), lambda x, lo, hi: _minmax(_minmax(x, lo, False), hi, True)),

  # Float/int conversions (type-checked casts)
  (UPat(Ops.CUSTOM, arg='f32_to_i32', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.int32, (x,))),
  (UPat(Ops.CUSTOM, arg='f32_to_f16', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.float16, (x,))),
  (UPat(Ops.CUSTOM, arg='f32_to_f64', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.float64, (x,))),
  (UPat(Ops.CUSTOM, arg='f32_to_i8', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.int8, (x,))),
  (UPat(Ops.CUSTOM, arg='f32_to_u8', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.uint8, (x,))),
  (UPat(Ops.CUSTOM, arg='f32_to_i16', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.int16, (x,))),
  (UPat(Ops.CUSTOM, arg='f32_to_u16', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.uint16, (x,))),
  (UPat(Ops.CUSTOM, arg='f64_to_i32', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.int32, (x,))),
  (UPat(Ops.CUSTOM, arg='f64_to_f32', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.float32, (x,))),
  (UPat(Ops.CUSTOM, arg='f16_to_f32', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.float32, (x,))),
  (UPat(Ops.CUSTOM, arg='f16_to_i16', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.int16, (x,))),
  (UPat(Ops.CUSTOM, arg='f16_to_u16', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.uint16, (x,))),
  (UPat(Ops.CUSTOM, arg='i32_to_f32', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.float32, (x,))),
  (UPat(Ops.CUSTOM, arg='i32_to_f64', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.float64, (x,))),
  (UPat(Ops.CUSTOM, arg='u32_to_f32', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.float32, (x,))),
  (UPat(Ops.CUSTOM, arg='u32_to_f64', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.float64, (x,))),
  (UPat(Ops.CUSTOM, arg='i16_to_f16', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.float16, (x,))),
  (UPat(Ops.CUSTOM, arg='u16_to_f16', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.float16, (x,))),
  (UPat(Ops.CUSTOM, arg='v_cvt_u16_f32', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.uint16, (x,))),
  (UPat(Ops.CUSTOM, arg='v_cvt_i16_f32', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.int16, (x,))),

  # Bit manipulation conversions
  (UPat(Ops.CUSTOM, arg='signext', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.int64, (x,))),
  (UPat(Ops.CUSTOM, arg='bf16_to_f32', src=(UPat.var('x'),)), lambda x: UOp(Ops.BITCAST, dtypes.float32, (
    UOp(Ops.SHL, dtypes.uint32, (UOp(Ops.CAST, dtypes.uint32, (x,)), UOp.const(dtypes.uint32, 16))),))),
  (UPat(Ops.CUSTOM, arg='u32_to_u16', src=(UPat.var('x'),)), lambda x: UOp(Ops.AND, dtypes.uint32, (x, UOp.const(dtypes.uint32, 0xffff)))),
  (UPat(Ops.CUSTOM, arg='i32_to_i16', src=(UPat.var('x'),)), lambda x: UOp(Ops.CAST, dtypes.int16, (
    UOp(Ops.AND, dtypes.uint32, (UOp(Ops.CAST, dtypes.uint32, (x,)), UOp.const(dtypes.uint32, 0xffff))),))),
]) + PatternMatcher([
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

def _transform_uop(u: UOp) -> UOp:
  """Transform a UOp tree, rewriting CUSTOM ops and propagating dtypes."""
  return graph_rewrite(u, pcode_pm)

def _transform_stmt(stmt):
  """Transform a statement, rewriting all UOps within it."""
  match stmt:
    # Chained assignment: ASSIGN(lhs, ASSIGN(rhs_lhs, rhs_rhs))
    case UOp(Ops.ASSIGN, _, (lhs, UOp(Ops.ASSIGN, _, (rhs_lhs, rhs_rhs)))):
      return UOp(Ops.ASSIGN, dtypes.void, (_transform_uop(lhs), UOp(Ops.ASSIGN, dtypes.void, (_transform_uop(rhs_lhs), _transform_uop(rhs_rhs)))))
    # Simple assignment: ASSIGN(lhs, rhs)
    case UOp(Ops.ASSIGN, _, (lhs, rhs)):
      return UOp(Ops.ASSIGN, dtypes.void, (_transform_uop(lhs), _transform_uop(rhs)))
    # Declaration: DEFINE_VAR with dtype and name arg
    case UOp(Ops.DEFINE_VAR, dtype, arg=name) if name is not None and dtype != dtypes.void:
      return UOp(Ops.DEFINE_VAR, dtype, arg=name)
    case If(branches):
      new_branches = tuple((_transform_uop(cond) if cond is not None else None, tuple(_transform_stmt(s) for s in body)) for cond, body in branches)
      return If(new_branches)
    case For(var, start, end, body):
      return For(var, _transform_uop(start), _transform_uop(end), tuple(_transform_stmt(s) for s in body))
    case Lambda(name, params, body):
      if isinstance(body, UOp): return Lambda(name, params, _transform_uop(body))
      return Lambda(name, params, tuple(_transform_stmt(s) for s in body))
    case Return(v):
      return Return(_transform_uop(v))
    case UOp():  # Other UOps (like bare function calls)
      return _transform_uop(stmt)
    case _:
      return stmt

def parse_transform(pcode: str) -> tuple:
  """Parse pseudocode and transform CUSTOM ops to UOps."""
  stmts = parse(pcode)
  return tuple(_transform_stmt(s) for s in stmts)
