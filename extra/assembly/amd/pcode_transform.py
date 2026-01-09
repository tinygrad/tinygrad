# Transform parsed pcode CUSTOM ops to UOps using PatternMatcher
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, graph_rewrite
from tinygrad.dtype import dtypes, DType
from extra.assembly.amd.pcode_parse import parse, Assign, Declare, If, For, Lambda, Break, Return

def _typed_const(src: UOp, val) -> UOp:
  """Create a const with same dtype as src, or a deferred const if src.dtype is void."""
  return UOp.const(src.dtype, val) if src.dtype != dtypes.void else UOp(Ops.CONST, dtypes.void, (src,), val)

def _floor(x: UOp) -> UOp:
  trunc = UOp(Ops.TRUNC, x.dtype, (x,))
  return UOp(Ops.WHERE, x.dtype, (UOp(Ops.CMPLT, dtypes.bool, (x, trunc)), UOp(Ops.SUB, x.dtype, (trunc, _typed_const(x, 1))), trunc))

def _minmax(a: UOp, b: UOp, is_min: bool) -> UOp:
  cmp = UOp(Ops.CMPLT, dtypes.bool, (a, b) if is_min else (b, a))
  return UOp(Ops.WHERE, a.dtype, (cmp, a, b))

def _minmax3(a: UOp, b: UOp, c: UOp, is_min: bool) -> UOp:
  return _minmax(_minmax(a, b, is_min), c, is_min)

# PatternMatcher for transforming CUSTOM ops to UOps
pcode_pm = PatternMatcher([
  # Boolean functions
  (UPat(Ops.CUSTOM, arg='isNAN', src=(UPat.var('x'),)), lambda x: UOp(Ops.CMPNE, dtypes.bool, (x, x))),
  (UPat(Ops.CUSTOM, arg='isINF', src=(UPat.var('x'),)), lambda x: UOp(Ops.OR, dtypes.bool, (
    UOp(Ops.CMPEQ, dtypes.bool, (x, _typed_const(x, float('inf')))),
    UOp(Ops.CMPEQ, dtypes.bool, (x, _typed_const(x, float('-inf'))))))),

  # Math functions
  (UPat(Ops.CUSTOM, arg='abs', src=(UPat.var('x'),)), lambda x: UOp(Ops.WHERE, x.dtype, (
    UOp(Ops.CMPLT, dtypes.bool, (x, _typed_const(x, 0))), UOp(Ops.NEG, x.dtype, (x,)), x))),
  (UPat(Ops.CUSTOM, arg='cos', src=(UPat.var('x'),)), lambda x: UOp(Ops.SIN, x.dtype, (
    UOp(Ops.ADD, x.dtype, (x, _typed_const(x, 1.5707963267948966))),))),
  (UPat(Ops.CUSTOM, arg='floor', src=(UPat.var('x'),)), lambda x: _floor(x)),
  (UPat(Ops.CUSTOM, arg='fract', src=(UPat.var('x'),)), lambda x: UOp(Ops.SUB, x.dtype, (x, _floor(x)))),
  (UPat(Ops.CUSTOM, arg='rsqrt', src=(UPat.var('x'),)), lambda x: UOp(Ops.RECIPROCAL, x.dtype, (UOp(Ops.SQRT, x.dtype, (x,)),))),

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
])

def _transform_uop(u: UOp) -> UOp:
  """Transform a UOp tree, rewriting CUSTOM ops."""
  return graph_rewrite(u, pcode_pm)

def _transform_stmt(stmt):
  """Transform a statement, rewriting all UOps within it."""
  match stmt:
    case Assign(lhs, rhs) if isinstance(rhs, Assign):
      return Assign(_transform_uop(lhs), Assign(_transform_uop(rhs.lhs), _transform_uop(rhs.rhs)))
    case Assign(lhs, rhs):
      return Assign(_transform_uop(lhs), _transform_uop(rhs))
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
    case _:
      return stmt

def parse_transform(pcode: str) -> tuple:
  """Parse pseudocode and transform CUSTOM ops to UOps."""
  stmts = parse(pcode)
  return tuple(_transform_stmt(s) for s in stmts)
