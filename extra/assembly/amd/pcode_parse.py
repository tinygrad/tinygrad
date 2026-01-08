# Minimal parser for AMD GPU pseudocode -> UOps
from __future__ import annotations
import re
from dataclasses import dataclass
from tinygrad.dtype import dtypes, DType
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp
# DType lookup table for AMD pseudocode type suffixes
from tinygrad.dtype import INVERSE_DTYPES_DICT
_QDTYPES: dict[str, DType] = {
  'f64': dtypes.float64, 'f32': dtypes.float32, 'f16': dtypes.float16, 'bf16': dtypes.bfloat16,
  'fp8': DType.new(4, 8, "fp8", None), 'bf8': DType.new(4, 8, "bf8", None),
  'fp6': DType.new(4, 6, "fp6", None), 'bf6': DType.new(4, 6, "bf6", None),
  'fp4': DType.new(4, 4, "fp4", None), 'i4': DType.new(5, 4, "i4", None),
  'u64': dtypes.uint64, 'u32': dtypes.uint32, 'u16': dtypes.uint16, 'u8': dtypes.uint8,
  'i64': dtypes.int64, 'i32': dtypes.int32, 'i16': dtypes.int16, 'i8': dtypes.int8,
  'b1201': DType.new(6, 1201, "b1201", None), 'b1024': DType.new(6, 1024, "b1024", None), 'b512': DType.new(6, 512, "b512", None),
  'b192': DType.new(6, 192, "b192", None), 'b128': DType.new(6, 128, "b128", None),
  'b65': DType.new(6, 65, "b65", None), 'b64': dtypes.uint64, 'b32': dtypes.uint32, 'b23': DType.new(6, 23, "b23", None), 'b16': dtypes.uint16, 'b8': dtypes.uint8, 'b4': DType.new(6, 4, "b4", None),
  'u1201': DType.new(6, 1201, "u1201", None), 'u65': DType.new(6, 65, "u65", None), 'u24': DType.new(6, 24, "u24", None), 'u23': DType.new(6, 23, "u23", None),
  'u6': DType.new(6, 6, "u6", None), 'u4': DType.new(6, 4, "u4", None),
  # i1/u1 are used for carry/overflow bits in 64-bit multiply-add ops (e.g., { D1.u1, D0.u64 } = 65-bit result)
  'u3': DType.new(6, 3, "u3", None), 'u1': DType.new(6, 1, "u1", None),
  'i65': DType.new(5, 65, "i65", None), 'i24': DType.new(5, 24, "i24", None),
  'i1': DType.new(5, 1, "i1", None),
  'u': dtypes.uint32, 'i': dtypes.int32, 'f': dtypes.float32,
}
# Register custom dtypes for repr
for k, v in _QDTYPES.items():
  if v.name not in INVERSE_DTYPES_DICT: INVERSE_DTYPES_DICT[v.name] = k

# String to Ops mapping
_BINOPS: dict[str, Ops] = {
  '+': Ops.ADD, '-': Ops.SUB, '*': Ops.MUL, '/': Ops.FDIV, '%': Ops.MOD, '**': Ops.POW,
  '&': Ops.AND, '|': Ops.OR, '^': Ops.XOR, '<<': Ops.SHL, '>>': Ops.SHR,
  '==': Ops.CMPEQ, '!=': Ops.CMPNE, '<>': Ops.CMPNE,
  '<': Ops.CMPLT, '>': Ops.CMPLT, '<=': Ops.CMPLE, '>=': Ops.CMPLE,
  '||': Ops.OR, '&&': Ops.AND,
}
# NOTE: ~ is bitwise NOT (XOR with -1), ! is logical NOT (compare == 0). NEG is arithmetic negation, not suitable here.
_UNOPS: dict[str, Ops] = {'-': Ops.NEG, '~': Ops.XOR, '!': Ops.CMPEQ}

def _typed_const(src: UOp, val) -> UOp:
  """Create a const with same dtype as src, or a deferred const if src.dtype is void."""
  return UOp.const(src.dtype, val) if src.dtype != dtypes.void else UOp(Ops.CONST, dtypes.void, (src,), val)

def _floor(x):
  trunc = UOp(Ops.TRUNC, x.dtype, (x,))
  return UOp(Ops.WHERE, x.dtype, (UOp(Ops.CMPLT, dtypes.bool, (x, trunc)), UOp(Ops.SUB, x.dtype, (trunc, _typed_const(x, 1))), trunc))

def _cvt(src_dt: DType, dst_dt: DType):
  """Create a conversion function that asserts input type and casts to output type."""
  def convert(x: UOp) -> UOp:
    assert x.dtype == src_dt, f"Expected {src_dt}, got {x.dtype}"
    return UOp(Ops.CAST, dst_dt, (x,))
  return convert

def _minmax(dt: DType, is_min: bool):
  """Create a min/max function that asserts input types."""
  def fn(*args: UOp) -> UOp:
    for i, a in enumerate(args):
      assert a.dtype == dt, f"Expected {dt} for arg {i}, got {a.dtype}"
    cmp = lambda x, y: UOp(Ops.CMPLT, dtypes.bool, (x, y) if is_min else (y, x))
    result = UOp(Ops.WHERE, dt, (cmp(args[0], args[1]), args[0], args[1]))
    return UOp(Ops.WHERE, dt, (cmp(result, args[2]), result, args[2])) if len(args) > 2 else result
  return fn

# Function expansions: name -> lambda(*srcs) -> UOp
_FN_EXPAND: dict[str, callable] = {
  'trunc': lambda x: UOp(Ops.TRUNC, x.dtype, (x,)),
  'sqrt': lambda x: UOp(Ops.SQRT, x.dtype, (x,)),
  'exp2': lambda x: UOp(Ops.EXP2, x.dtype, (x,)),
  'log2': lambda x: UOp(Ops.LOG2, x.dtype, (x,)),
  'sin': lambda x: UOp(Ops.SIN, x.dtype, (x,)),
  'rcp': lambda x: UOp(Ops.RECIPROCAL, x.dtype, (x,)),
  'fma': lambda a, b, c: UOp(Ops.MULACC, c.dtype, (a, b, c)),
  'isNAN': lambda x: UOp(Ops.CMPNE, dtypes.bool, (x, x)),
  'rsqrt': lambda x: UOp(Ops.RECIPROCAL, x.dtype, (UOp(Ops.SQRT, x.dtype, (x,)),)),
  'abs': lambda x: UOp(Ops.WHERE, x.dtype, (UOp(Ops.CMPLT, dtypes.bool, (x, _typed_const(x, 0))), UOp(Ops.NEG, x.dtype, (x,)), x)),
  'cos': lambda x: UOp(Ops.SIN, x.dtype, (UOp(Ops.ADD, x.dtype, (x, _typed_const(x, 1.5707963267948966))),)),
  'floor': _floor,
  'fract': lambda x: UOp(Ops.SUB, x.dtype, (x, _floor(x))),
  'isINF': lambda x: UOp(Ops.OR, dtypes.bool, (UOp(Ops.CMPEQ, dtypes.bool, (x, _typed_const(x, float('inf')))),
                                               UOp(Ops.CMPEQ, dtypes.bool, (x, _typed_const(x, float('-inf')))))),
  'min': lambda a, b: UOp(Ops.WHERE, a.dtype, (UOp(Ops.CMPLT, dtypes.bool, (a, b)), a, b)),
  'max': lambda a, b: UOp(Ops.WHERE, a.dtype, (UOp(Ops.CMPLT, dtypes.bool, (b, a)), a, b)),
  'clamp': lambda x, lo, hi: (c := UOp(Ops.WHERE, x.dtype, (UOp(Ops.CMPLT, dtypes.bool, (x, lo)), lo, x)),
                              UOp(Ops.WHERE, x.dtype, (UOp(Ops.CMPLT, dtypes.bool, (hi, c)), hi, c)))[1],
  # Conversions (type-checked casts)
  'f32_to_i32': _cvt(dtypes.float32, dtypes.int32), 'f32_to_f16': _cvt(dtypes.float32, dtypes.float16),
  'f32_to_f64': _cvt(dtypes.float32, dtypes.float64), 'f32_to_i8': _cvt(dtypes.float32, dtypes.int8),
  'f32_to_u8': _cvt(dtypes.float32, dtypes.uint8), 'f32_to_i16': _cvt(dtypes.float32, dtypes.int16),
  'f32_to_u16': _cvt(dtypes.float32, dtypes.uint16), 'f64_to_i32': _cvt(dtypes.float64, dtypes.int32),
  'f64_to_f32': _cvt(dtypes.float64, dtypes.float32), 'f16_to_f32': _cvt(dtypes.float16, dtypes.float32),
  'f16_to_i16': _cvt(dtypes.float16, dtypes.int16), 'f16_to_u16': _cvt(dtypes.float16, dtypes.uint16),
  'i32_to_f32': _cvt(dtypes.int32, dtypes.float32), 'i32_to_f64': _cvt(dtypes.int32, dtypes.float64),
  'u32_to_f32': _cvt(dtypes.uint32, dtypes.float32), 'u32_to_f64': _cvt(dtypes.uint32, dtypes.float64),
  'i16_to_f16': _cvt(dtypes.int16, dtypes.float16), 'u16_to_f16': _cvt(dtypes.uint16, dtypes.float16),
  'v_cvt_u16_f32': _cvt(dtypes.float32, dtypes.uint16), 'v_cvt_i16_f32': _cvt(dtypes.float32, dtypes.int16),
  # v_min/v_max (2 args, type-checked)
  'v_min_f16': _minmax(dtypes.float16, True), 'v_min_f32': _minmax(dtypes.float32, True),
  'v_min_i16': _minmax(dtypes.int16, True), 'v_min_i32': _minmax(dtypes.int32, True),
  'v_min_u16': _minmax(dtypes.uint16, True), 'v_min_u32': _minmax(dtypes.uint32, True),
  'v_max_f16': _minmax(dtypes.float16, False), 'v_max_f32': _minmax(dtypes.float32, False),
  'v_max_i16': _minmax(dtypes.int16, False), 'v_max_i32': _minmax(dtypes.int32, False),
  'v_max_u16': _minmax(dtypes.uint16, False), 'v_max_u32': _minmax(dtypes.uint32, False),
  # v_min3/v_max3 (3 args, type-checked)
  'v_min3_f16': _minmax(dtypes.float16, True), 'v_min3_f32': _minmax(dtypes.float32, True),
  'v_min3_i16': _minmax(dtypes.int16, True), 'v_min3_i32': _minmax(dtypes.int32, True),
  'v_min3_u16': _minmax(dtypes.uint16, True), 'v_min3_u32': _minmax(dtypes.uint32, True),
  'v_max3_f16': _minmax(dtypes.float16, False), 'v_max3_f32': _minmax(dtypes.float32, False),
  'v_max3_i16': _minmax(dtypes.int16, False), 'v_max3_i32': _minmax(dtypes.int32, False),
  'v_max3_u16': _minmax(dtypes.uint16, False), 'v_max3_u32': _minmax(dtypes.uint32, False),
}

# Function return type inference for CUSTOM ops
_BOOL_FNS = {'isNAN', 'isINF', 'isDENORM', 'isQuietNAN', 'isSignalNAN', 'isEven', 'LT_NEG_ZERO', 'GT_NEG_ZERO'}
_PASSTHRU_FNS = {'abs', 'floor', 'fract', 'sqrt', 'sin', 'cos', 'trunc', 'fma', 'clamp', 'min', 'max', 'ldexp',
                 'cvtToQuietNAN', 'pow', 'rcp', 'rsqrt', 'exp2', 'log2', 'mantissa'}
_U32_FNS = {'sign', 'exponent', 'ABSDIFF', 'SAT8', 'BYTE_PERMUTE', 'count_ones', 'countbits', 'reverse_bits',
            'u8_to_u32', 'u4_to_u32', 'u32_to_u16', 's_ff1_i32_b32', 's_ff1_i32_b64', 'v_sad_u8', 'v_msad_u8'}
_CVT_FNS = {  # conversion functions: name -> output dtype (only those not in _FN_EXPAND)
  'f32_to_u32': dtypes.uint32, 'f64_to_u32': dtypes.uint32,  # need clamping
  'i32_to_i16': dtypes.int16, 'u32_to_u16': dtypes.uint32,  # need masking
  'bf16_to_f32': dtypes.float32, 'f32_to_bf16': dtypes.bfloat16,  # bit manipulation
  'f16_to_snorm': dtypes.int16, 'f16_to_unorm': dtypes.uint16, 'f32_to_snorm': dtypes.int16, 'f32_to_unorm': dtypes.uint16,  # scaling
  'signext': dtypes.int64, 'signext_from_bit': dtypes.int64,  # special handling
}

def _infer_fn_dtype(name: str, srcs: tuple[UOp, ...]) -> DType:
  """Infer output dtype for a function call based on function name and input types."""
  if name in _BOOL_FNS: return dtypes.bool
  if name in _PASSTHRU_FNS: return srcs[0].dtype if srcs and srcs[0].dtype != dtypes.void else dtypes.void
  if name in _U32_FNS: return dtypes.uint32
  if name in _CVT_FNS: return _CVT_FNS[name]
  if name == 'trig_preop_result': return dtypes.float64
  # Default: inherit from first non-void source, or void
  for s in srcs:
    if s.dtype != dtypes.void: return s.dtype
  return dtypes.void

# Statement types (control flow, not expressions)
@dataclass(frozen=True)
class Assign: lhs: UOp; rhs: UOp
@dataclass(frozen=True)
class Declare: name: str; dtype: DType
@dataclass(frozen=True)
class If: branches: tuple[tuple[UOp|None, tuple[Stmt, ...]], ...]
@dataclass(frozen=True)
class For: var: str; start: UOp; end: UOp; body: tuple[Stmt, ...]
@dataclass(frozen=True)
class Lambda: name: str; params: tuple[str, ...]; body: tuple[Stmt, ...]|UOp
@dataclass(frozen=True)
class Break: pass
@dataclass(frozen=True)
class Return: value: UOp
Stmt = Assign|Declare|If|For|Lambda|Break|Return

# Parse context for tracking variable dtypes (module-level, set during parse())
_var_dtypes: dict[str, DType] = {}

def _match(s, i, o, c):
  d = 1
  for j in range(i+1, len(s)):
    if s[j] == o: d += 1
    elif s[j] == c: d -= 1
    if d == 0: return j
  return -1

def _split(s):
  r, d, l = [], 0, 0
  for i, c in enumerate(s):
    if c in '([{': d += 1
    elif c in ')]}': d -= 1
    elif c == ',' and d == 0: r.append(s[l:i].strip()); l = i+1
  if s[l:].strip(): r.append(s[l:].strip())
  return r

def _fop(s, ops):
  d = b = 0
  for i in range(len(s)-1, -1, -1):
    c = s[i]
    if c == ')': d += 1
    elif c == '(': d -= 1
    elif c == ']': b += 1
    elif c == '[': b -= 1
    elif d == 0 and b == 0:
      for op in sorted(ops, key=len, reverse=True):
        if s[i:i+len(op)] == op:
          if op in ('<', '>') and (i+1 < len(s) and s[i+1] in '<>=' or i > 0 and s[i-1] in '<>='): continue
          if op == '*' and (i+1 < len(s) and s[i+1] == '*' or i > 0 and s[i-1] == '*'): continue
          if op == '-' and (not s[:i].rstrip() or s[:i].rstrip()[-1] in '+-*/(<>=&|^,'): continue
          return i
  return -1

def _get_dtype(name: str) -> DType | None: return _QDTYPES.get(name.lower())

def expr(s: str) -> UOp:
  s = s.strip().rstrip(';')
  if s.endswith('.') and not (len(s) > 1 and s[-2].isdigit()): s = s[:-1]
  s = s.strip()
  if not s: raise ValueError("Empty expression")
  if s == '+INF': s = 'INF'
  # Parentheses
  if s[0] == '(' and (e := _match(s, 0, '(', ')')) == len(s)-1: return expr(s[1:e])
  # Pack -> CAT: { hi, lo } concatenates to larger type
  if s[0] == '{' and s[-1] == '}':
    parts = tuple(expr(a) for a in _split(s[1:-1]))
    # Infer combined bitwidth from parts (e.g., {u32, u32} -> u64)
    total_bits = sum(p.dtype.bitsize for p in parts if p.dtype != dtypes.void)
    cat_dtype = dtypes.uint64 if total_bits > 32 else dtypes.uint32 if total_bits > 0 else dtypes.void
    return UOp(Ops.CAT, cat_dtype, parts)
  # Typed cast: 32'U(expr) - value conversion (vs .type which is bit reinterpretation)
  if m := re.match(r"^(\d+)'([IUFB])\(", s):
    if (e := _match(s, m.end()-1, '(', ')')) == len(s)-1:
      cast_dtype = _QDTYPES[f"{m[2].lower()}{m[1]}"]
      assert cast_dtype != dtypes.void, f"CAST target type should not be void"
      return UOp(Ops.CAST, cast_dtype, (expr(s[m.end():e]),))
  # Typed constant: 32'-5I
  if m := re.match(r"^(\d+)'(-?\d+)([IUFB])?$", s):
    return UOp(Ops.CONST, _QDTYPES[f"{(m[3] or 'I').lower()}{m[1]}"], arg=int(m[2]))
  if m := re.match(r"^(\d+)'(-?[\d.]+)$", s):
    return UOp(Ops.CONST, _QDTYPES[f"f{m[1]}"], arg=float(m[2]))
  if m := re.match(r"^(\d+)'(0x[0-9a-fA-F]+)$", s):
    return UOp(Ops.CONST, _QDTYPES[f"u{m[1]}"], arg=int(m[2], 16))
  # Function call -> direct UOp or CUSTOM
  if m := re.match(r"^([A-Za-z_]\w*)\(", s):
    if (e := _match(s, m.end()-1, '(', ')')) == len(s)-1:
      a = _split(s[m.end():e])
      srcs = tuple(expr(x) for x in a) if a != [''] else ()
      name = m[1]
      if name in _FN_EXPAND: return _FN_EXPAND[name](*srcs)
      output_dtype = _infer_fn_dtype(name, srcs)
      return UOp(Ops.CUSTOM, output_dtype, srcs, arg=name)
  # MEM[addr] -> CUSTOM('MEM', addr), MEM[addr].type -> BITCAST
  if s[:4] == 'MEM[' and (e := _match(s, 3, '[', ']')) != -1:
    r, b = s[e+1:], UOp(Ops.CUSTOM, dtypes.void, (expr(s[4:e]),), arg='MEM')
    if not r: return b
    if r[:1] == '.' and (dt := _get_dtype(r[1:])):
      assert dt != dtypes.void, f"BITCAST target type should not be void"
      return UOp(Ops.BITCAST, dt, (b,))
  # Ternary: cond ? t : f -> WHERE
  if (q := _fop(s, ('?',))) > 0:
    d = b = 0
    for i in range(q+1, len(s)):
      if s[i] == '(': d += 1
      elif s[i] == ')': d -= 1
      elif s[i] == '[': b += 1
      elif s[i] == ']': b -= 1
      elif s[i] == ':' and d == 0 and b == 0:
        gate, lhs, rhs = expr(s[:q]), expr(s[q+1:i]), expr(s[i+1:])
        # Infer output dtype from lhs/rhs (prefer non-void)
        out_dtype = lhs.dtype if lhs.dtype != dtypes.void else rhs.dtype
        # Gate can be bool, void (unresolved), or integer (C-style truthiness: 0=false, non-zero=true)
        assert gate.dtype == dtypes.void or gate.dtype == dtypes.bool or dtypes.is_int(gate.dtype), \
          f"gate on WHERE must be bool or int, got {gate.dtype}"
        return UOp(Ops.WHERE, out_dtype, (gate, lhs, rhs))
  # Binary ops
  for ops in [('||',),('&&',),('|',),('^',),('&',),('==','!=','<>'),('<=','>=','<','>'),('<<','>>'),
              ('+','-'),('*','/','%'),('**',)]:
    if (p := _fop(s, ops)) > 0:
      op = next(o for o in sorted(ops, key=len, reverse=True) if s[p:p+len(o)] == o)
      l, r = s[:p].strip(), s[p+len(op):].strip()
      if l and r:
        lhs, rhs = expr(l), expr(r)
        flipped = op in ('>', '>=')
        if flipped: lhs, rhs = rhs, lhs
        tag = 'flipped' if flipped else ('<>' if op == '<>' else None)
        uop_op = _BINOPS[op]
        output_dtype = lhs.dtype if lhs.dtype != dtypes.void else rhs.dtype
        if uop_op in {Ops.CMPNE, Ops.CMPEQ, Ops.CMPLE, Ops.CMPLT}: output_dtype = dtypes.bool
        return UOp(uop_op, output_dtype, (lhs, rhs), tag=tag)
  # Unary ops: - (negate), ~ (bitwise NOT), ! (logical NOT)
  if s[0] in '-~!' and len(s) > 1 and (s[0] != '!' or s[1] != '='):
    src = expr(s[1:])
    # ! is logical NOT (compare to 0), returns bool; - and ~ preserve dtype
    out_dtype = dtypes.bool if s[0] == '!' else src.dtype
    return UOp(_UNOPS[s[0]], out_dtype, (src,))
  # Slice/Index -> CUSTOMI
  if '[' in s and s[-1] == ']':
    d = 0
    for i in range(len(s)-1, -1, -1):
      if s[i] == ']': d += 1
      elif s[i] == '[': d -= 1
      if d == 0 and s[i] == '[': break
    b, n = s[:i], s[i+1:-1]
    if '+:' in n:  # Verilog [start +: width]
      st, w = expr(n.split('+:', 1)[0]), expr(n.split('+:', 1)[1])
      # hi = start + width - 1; use int32 for index computations
      idx_dt = st.dtype if st.dtype != dtypes.void else dtypes.int32
      hi = UOp(Ops.SUB, idx_dt, (UOp(Ops.ADD, idx_dt, (st, w)), UOp(Ops.CONST, dtypes.int32, arg=1)))
      # Infer slice dtype from width (if constant)
      slice_dtype = dtypes.uint64 if (w.op == Ops.CONST and w.arg > 32) else dtypes.uint32
      # NOTE: CUSTOMI is used for bit slicing; SHRINK would be for tensor operations
      return UOp(Ops.CUSTOMI, slice_dtype, (expr(b), hi, st))
    if ':' in n and '?' not in n:
      d = 0
      for j, c in enumerate(n):
        if c in '([{': d += 1
        elif c in ')]}': d -= 1
        elif c == ':' and d == 0:
          hi_expr, lo_expr = expr(n[:j]), expr(n[j+1:])
          # Infer slice dtype from constant bounds (hi - lo + 1 bits)
          if hi_expr.op == Ops.CONST and lo_expr.op == Ops.CONST:
            width = abs(int(hi_expr.arg) - int(lo_expr.arg)) + 1
            slice_dtype = dtypes.uint64 if width > 32 else dtypes.uint32
          else:
            slice_dtype = dtypes.uint32  # default for dynamic slices
          return UOp(Ops.CUSTOMI, slice_dtype, (expr(b), hi_expr, lo_expr))
    idx = expr(n)
    base = expr(b)
    # For array element access, use scalar type of the array; for bit index, use uint32
    elem_dtype = base.dtype.scalar() if base.dtype != dtypes.void and base.dtype.count > 1 else dtypes.uint32
    return UOp(Ops.CUSTOMI, elem_dtype, (base, idx, idx))
  # Bitcast: expr.type
  if '.' in s:
    for i in range(len(s)-1, 0, -1):
      if s[i] == '.' and (dt := _get_dtype(s[i+1:])):
        assert dt != dtypes.void, f"BITCAST target type should not be void: {s}"
        return UOp(Ops.BITCAST, dt, (expr(s[:i]),))
  # Variable
  if s[:5] == 'eval ': return UOp(Ops.DEFINE_VAR, dtypes.void, arg=(s, None, None))
  if re.match(r'^[A-Za-z_][\w.]*$', s):
    var_dtype = _var_dtypes.get(s, dtypes.void)
    return UOp(Ops.DEFINE_VAR, var_dtype, arg=(s, None, None))
  # Numeric literal
  # NOTE: hex constants are unsigned (uint32) even without U suffix
  try:
    if s[:2].lower() == '0x':
      m = re.match(r'0[xX]([0-9a-fA-F]+)([UuLl]*)$', s)
      if m:
        val, suf = int(m[1], 16), m[2].lower()
        if 'll' in suf: return UOp(Ops.CONST, dtypes.uint64 if 'u' in suf else dtypes.int64, arg=val)
        if 'u' in suf: return UOp(Ops.CONST, dtypes.uint32, arg=val)
        return UOp(Ops.CONST, dtypes.uint32, arg=val)
    suffix = re.search(r'([UuLlFf]+)$', s)
    suf = suffix[1].lower() if suffix else ''
    c = re.sub(r'[FfLlUu]+$', '', s)
    if '.' in c or 'e' in c.lower() or 'f' in suf: return UOp(Ops.CONST, dtypes.float32, arg=float(c))
    if 'u' in suf: return UOp(Ops.CONST, dtypes.uint64 if 'll' in suf else dtypes.uint32, arg=int(c))
    if 'll' in suf: return UOp(Ops.CONST, dtypes.int64, arg=int(c))
    return UOp(Ops.CONST, dtypes.int32, arg=int(c))
  except ValueError: pass
  raise ValueError(f"Cannot parse expression: {s}")

def stmt(line: str) -> Stmt|None:
  # NOTE: variable dtypes are resolved in ucode.py via INPUT_VARS (SCC=uint32, ADDR=uint64, etc.)
  line = line.split('//')[0].strip().rstrip(';').rstrip('.')
  if not line: return None
  if line == 'break': return Break()
  if line[:7] == 'return ': return Return(expr(line[7:]))
  if line[:5] == 'eval ': return Assign(UOp(Ops.DEFINE_VAR, dtypes.void, arg=('_eval', None, None)), UOp(Ops.DEFINE_VAR, dtypes.void, arg=(line, None, None)))
  if line[:8] == 'declare ' and ':' in line:
    n, t = line[8:].split(':', 1)
    t = t.strip()
    vec_count = int(m[1]) if (m := re.search(r'\[(\d+)\]$', t)) else 1
    t = t.split('[')[0]  # strip array suffix like [64]
    if m := re.match(r"^(\d+)'([IUFB])$", t):
      dt = _QDTYPES[f"{m[2].lower()}{m[1]}"]
      final_dt = dt.vec(vec_count) if vec_count > 1 else dt
      _var_dtypes[n.strip()] = final_dt  # track dtype for subsequent expr() calls
      return Declare(n.strip(), final_dt)
    return None  # unsupported declare type
  for op, uop in [('+=', Ops.ADD), ('-=', Ops.SUB), ('|=', Ops.OR), ('&=', Ops.AND), ('^=', Ops.XOR), ('<<=', Ops.SHL), ('>>=', Ops.SHR)]:
    if op in line:
      l, r = line.split(op, 1)
      lhs, rhs = expr(l), expr(r)
      # Infer result dtype from operands (dtype resolution happens in ucode._stmt)
      result_dtype = lhs.dtype if lhs.dtype != dtypes.void else rhs.dtype
      return Assign(lhs, UOp(uop, result_dtype, (lhs, rhs)))
  if '=' in line and not any(line[:k] == p for k, p in [(3,'if '),(6,'elsif '),(4,'for ')]):
    # Find leftmost assignment = (not ==, <=, >=, !=) for chained assignment support
    eq = -1
    for i in range(1, len(line) - 1):
      if line[i] == '=' and line[i-1] not in '!<>=' and line[i+1] != '=':
        eq = i
        break
    if eq > 0:
      rhs = line[eq+1:].strip()
      # Check if RHS contains another assignment = (not ==, <=, >=, !=)
      has_assign = False
      for i in range(1, len(rhs) - 1):
        if rhs[i] == '=' and rhs[i-1] not in '!<>=' and rhs[i+1] != '=':
          has_assign = True
          break
      if has_assign:
        rhs_parsed = stmt(rhs)
        if isinstance(rhs_parsed, Assign):
          lhs = expr(line[:eq])
          return Assign(lhs, rhs_parsed)
      lhs, rhs_expr = expr(line[:eq]), expr(rhs)
      # Track dtype for bare variable assignments (e.g., tmp = S0.u32)
      if lhs.op == Ops.DEFINE_VAR and lhs.dtype == dtypes.void and rhs_expr.dtype != dtypes.void:
        _var_dtypes[lhs.arg[0]] = rhs_expr.dtype
        lhs = UOp(Ops.DEFINE_VAR, rhs_expr.dtype, arg=lhs.arg)
      return Assign(lhs, rhs_expr)
  # Bare function call (e.g., nop())
  if re.match(r'\w+\([^)]*\)$', line):
    return expr(line)
  raise ValueError(f"Cannot parse statement: {line}")

def parse(code: str, _toplevel: bool = True) -> tuple[Stmt, ...]:
  global _var_dtypes
  if _toplevel: _var_dtypes = {}  # only reset at top level, preserve for recursive calls
  lines = [l.split('//')[0].strip() for l in code.strip().split('\n') if l.split('//')[0].strip()]
  # Join continuation lines (unbalanced parens) - but not for control flow or lambdas
  joined, j = [], 0
  while j < len(lines):
    ln = lines[j]
    # Don't join lambda lines - they have their own multiline handling
    if '= lambda(' not in ln:
      while ln.count('(') > ln.count(')') and j + 1 < len(lines):
        next_ln = lines[j + 1]
        # Don't join if next line is control flow or looks like a new statement
        if next_ln[:3] == 'if ' or next_ln[:4] == 'for ' or next_ln[:6] == 'elsif ' or next_ln == 'else' or \
           next_ln == 'endif' or next_ln == 'endfor' or '= lambda(' in next_ln: break
        j += 1
        ln += ' ' + next_ln
    joined.append(ln)
    j += 1
  lines = joined
  stmts, i = [], 0
  while i < len(lines):
    ln = lines[i].rstrip(';')
    # Lambda: NAME = lambda(params) ( body );
    if '= lambda(' in ln and (m := re.match(r'(\w+)\s*=\s*lambda\(([^)]*)\)\s*\(', ln)):
      name, params = m[1], tuple(p.strip() for p in m[2].split(',')) if m[2].strip() else ()
      # Collect lambda body until closing );
      body_lines = [ln[m.end():]]
      i += 1
      while i < len(lines) and not lines[i-1].rstrip().endswith(');'):
        body_lines.append(lines[i])
        i += 1
      body_text = '\n'.join(body_lines).strip()
      if body_text.endswith(');'): body_text = body_text[:-2]
      # Try to parse as expression first, then as statements
      try:
        body = expr(body_text)
      except ValueError:
        body = parse(body_text, _toplevel=False)
      stmts.append(Lambda(name, params, body)); continue
    if ln[:4] == 'for ' and ' do' in ln and (m := re.match(r'for\s+(\w+)\s+in\s+(.+?)\s*:\s*(.+?)\s+do', ln)):
      i, body, d = i+1, [], 1
      while i < len(lines) and d > 0:
        line_i = lines[i].rstrip(';').rstrip('.')
        if line_i[:4] == 'for ' and ' do' in line_i: d += 1
        elif line_i == 'endfor': d -= 1
        if d > 0: body.append(lines[i])
        i += 1
      stmts.append(For(m[1], expr(m[2]), expr(m[3]), parse('\n'.join(body), _toplevel=False))); continue
    if ln[:3] == 'if ':
      cond = ln[3:ln.index(' then')] if ' then' in ln else ln[3:]
      br, body, i, depth = [], [], i+1, 1
      while i < len(lines) and depth > 0:
        line_i = lines[i].rstrip(';').rstrip('.')
        if line_i[:3] == 'if ': depth += 1; body.append(lines[i])
        elif line_i == 'endif':
          depth -= 1
          if depth > 0: body.append(lines[i])
        elif depth == 1 and line_i[:6] == 'elsif ':
          cond_end = line_i.index(' then') if ' then' in line_i else len(line_i)
          br.append((expr(cond), parse('\n'.join(body), _toplevel=False))); cond, body = line_i[6:cond_end], []
        elif depth == 1 and line_i == 'else':
          br.append((expr(cond), parse('\n'.join(body), _toplevel=False))); cond, body = None, []
        else: body.append(lines[i])
        i += 1
      br.append((expr(cond) if cond else None, parse('\n'.join(body), _toplevel=False))); stmts.append(If(tuple(br))); continue
    if ln == 'else' or ln[:6] == 'elsif ': raise ValueError(f"Unexpected {ln.split()[0]} without matching if")
    s = stmt(ln)
    if s is not None: stmts.append(s)
    i += 1
  return tuple(stmts)
