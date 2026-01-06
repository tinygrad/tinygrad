# Simple pseudocode executor - transforms UOps to Python code strings, then exec()
import functools, math
from tinygrad.uop.ops import UOp, Ops
from extra.assembly.amd.pcode_parse import parse, Assign, Declare, If, For, Break, Return
from extra.assembly.amd.ucode import _apply_pseudocode_fixes
from extra.assembly.amd.pcode import (
  MASK32, MASK64, _f32, _i32, _f64, _i64, _f16, _i16, _bf16, _ibf16, _sext, _brev, _div,
  fma, ldexp, sqrt, log2, fract, sin, cos, trunc, floor, ceil, exponent, sign, mantissa, signext_from_bit,
  isNAN, isQuietNAN, isSignalNAN, cvtToQuietNAN, v_min_f32, v_max_f32, v_min_f16, v_max_f16, v_min_i32, v_max_i32,
  f32_to_i32, f32_to_u32, f64_to_i32, f64_to_u32, i32_to_f32, u32_to_f32, f32_to_f16, f16_to_f32,
  f32_to_f64, f64_to_f32, i32_to_f64, u32_to_f64, bf16_to_f32, f32_to_bf16, f16_to_i16, f16_to_u16,
  ABSDIFF, BYTE_PERMUTE, v_sad_u8, v_msad_u8, s_ff1_i32_b32, s_ff1_i32_b64,
  PI, DENORM, INF, OVERFLOW_F32, OVERFLOW_F64, UNDERFLOW_F32, UNDERFLOW_F64, TWO_OVER_PI_1201,
)

_PARAM = {'OPSEL': 'opsel', 'OPSEL_HI': 'opsel_hi', 'PC': 'pc'}  # pseudocode name -> param name
_INPUTS = {'S0', 'S1', 'S2', 'D0', 'SCC', 'VCC', 'EXEC', 'laneId', 'SIMM16', 'SIMM32', 'PC', 'OPSEL', 'OPSEL_HI', 'SRC0', 'VDST'}
_OUTPUTS = {'D0', 'D1', 'SCC', 'VCC', 'EXEC', 'PC'}

from tinygrad.dtype import dtypes, DType

def _dtype_info(dt: DType) -> tuple[str|None, int, bool, bool]:
  """Returns (to_float_fn, bits, is_float, is_signed) for dtype."""
  if dt == dtypes.double: return ('_f64', 64, True, False)
  if dt == dtypes.float: return ('_f32', 32, True, False)
  if dt == dtypes.half: return ('_f16', 16, True, False)
  if dt == dtypes.bfloat16: return ('_bf16', 16, True, False)
  # Integer types - use itemsize and check signed via fmt
  bits = dt.itemsize * 8 if hasattr(dt, 'itemsize') else 32
  signed = dt.fmt is not None and dt.fmt.islower() if hasattr(dt, 'fmt') else False
  return (None, bits, False, signed)

# Binary ops: op -> format string
_BINOPS = {Ops.ADD: '({a}+{b})', Ops.SUB: '({a}-{b})', Ops.MUL: '({a}*{b})', Ops.FDIV: '_div({a},{b})', Ops.MOD: '({a}%{b})',
           Ops.AND: '({a}&{b})', Ops.OR: '({a}|{b})', Ops.XOR: '({a}^{b})', Ops.SHL: '({a}<<int({b}))', Ops.SHR: '({a}>>int({b}))',
           Ops.POW: '({a}**{b})', Ops.CMPLT: '(1 if {a}<{b} else 0)', Ops.CMPLE: '(1 if {a}<={b} else 0)',
           Ops.CMPEQ: '(1 if {a}=={b} else 0)', Ops.CMPNE: '(1 if {a}!={b} else 0)'}

def _gen_expr_raw(u: UOp, ctx: set) -> str:
  """Generate Python expression from UOp, yielding raw bits for float types (no float conversion)."""
  if u.op == Ops.BITCAST:
    _, bits, _, _ = _dtype_info(u.dtype)
    src = _gen_expr(u.src[0], ctx)
    return f'(int({src})&{hex((1<<bits)-1)})'
  return _gen_expr(u, ctx)

def _gen_expr(u: UOp, ctx: set) -> str:
  """Generate Python expression from UOp."""
  if u.op == Ops.CONST:
    if isinstance(u.arg, float):
      if math.isnan(u.arg): return "float('nan')"
      if math.isinf(u.arg): return f"float('{'-' if u.arg < 0 else ''}inf')"
    return repr(u.arg)
  if u.op == Ops.DEFINE_VAR:
    name = u.arg[0]
    return _PARAM.get(name, name)
  if u.op == Ops.BITCAST:
    to_fn, bits, is_float, is_signed = _dtype_info(u.dtype)
    src = _gen_expr(u.src[0], ctx)
    if to_fn: return f'{to_fn}({src})'
    if is_signed: return f'_sext(int({src})&{hex((1<<bits)-1)},{bits})'  # signed int
    return f'(int({src})&{hex((1<<bits)-1)})'  # unsigned int
  if u.op == Ops.CAST:
    to_fn, bits, is_float, is_signed = _dtype_info(u.dtype)
    src = _gen_expr(u.src[0], ctx)
    if to_fn: return f'float({src})'  # float cast
    return f'(int({src})&{hex((1<<bits)-1)})'  # int cast
  if u.op in _BINOPS and len(u.src) == 2:
    return _BINOPS[u.op].format(a=_gen_expr(u.src[0], ctx), b=_gen_expr(u.src[1], ctx))
  if u.op == Ops.XOR and len(u.src) == 1: return f'(~{_gen_expr(u.src[0], ctx)})'
  if u.op == Ops.CMPEQ and len(u.src) == 1: return f'(1 if not {_gen_expr(u.src[0], ctx)} else 0)'
  if u.op == Ops.NEG: return f'(-{_gen_expr(u.src[0], ctx)})'
  if u.op == Ops.WHERE: return f'({_gen_expr(u.src[1], ctx)} if {_gen_expr(u.src[0], ctx)} else {_gen_expr(u.src[2], ctx)})'
  if u.op == Ops.CUSTOMI:  # slice: base[hi:lo]
    base = _gen_expr(u.src[0], ctx)
    # Check if this is a reversed slice (hi < lo means bit-reverse)
    if u.src[1].op == Ops.CONST and u.src[2].op == Ops.CONST and u.src[1].arg < u.src[2].arg:
      hi, lo = u.src[2].arg, u.src[1].arg  # swap
      nbits = hi - lo + 1
      return f'_brev((({base}>>{lo})&{hex((1<<nbits)-1)}),{nbits})'
    hi, lo = _gen_expr(u.src[1], ctx), _gen_expr(u.src[2], ctx)
    return f'(({base}>>{lo})&((1<<({hi}-{lo}+1))-1))'
  if u.op == Ops.CUSTOM:  # function call
    fn = 'int' if u.arg == 'signext' else u.arg  # signext -> int
    # Functions expecting bits need raw values, not float-converted values
    args = [_gen_expr_raw(a, ctx) if u.arg in ('f16_to_i16', 'f16_to_u16') else _gen_expr(a, ctx) for a in u.src]
    return f'{fn}({",".join(args)})'
  if u.op == Ops.CAT and len(u.src) == 2:  # {hi, lo} concatenation
    return f'(({_gen_expr(u.src[0], ctx)}<<32)|({_gen_expr(u.src[1], ctx)}&0xffffffff))'
  return f'0  # unhandled {u.op}'

# Float-to-int conversion functions for LHS assignments (use _toi* helpers that handle int passthrough)
_FLOAT_TO_INT = {dtypes.double: '_toi64', dtypes.float: '_toi32', dtypes.half: '_toi16', dtypes.bfloat16: '_toibf16'}

def _extract_lhs(lhs: UOp) -> tuple[str, int, str|None, UOp|None, UOp|None]:
  """Extract (var_name, bits, float_conv, hi, lo) from LHS. float_conv is None for int types."""
  match lhs:
    case UOp(Ops.BITCAST, dt, (UOp(Ops.CUSTOMI, _, (UOp(Ops.DEFINE_VAR, _, _, (name, _, _)), hi, lo)),)):  # tmp[31:16].f16
      _, bits, _, _ = _dtype_info(dt)
      return (name, bits, _FLOAT_TO_INT.get(dt), hi, lo)
    case UOp(Ops.BITCAST, dt, (UOp(Ops.DEFINE_VAR, _, _, (name, _, _)),)):  # D0.f32
      _, bits, _, _ = _dtype_info(dt)
      return (name, bits, _FLOAT_TO_INT.get(dt), None, None)
    case UOp(Ops.DEFINE_VAR, _, _, (name, _, _)):  # tmp
      return (name, 64, None, None, None)
    case UOp(Ops.CUSTOMI, _, (UOp(Ops.BITCAST, dt, (UOp(Ops.DEFINE_VAR, _, _, (name, _, _)),)), hi, lo)):  # D0.u32[31:16]
      _, bits, _, _ = _dtype_info(dt)
      return (name, bits, _FLOAT_TO_INT.get(dt), hi, lo)
    case UOp(Ops.CUSTOMI, _, (UOp(Ops.DEFINE_VAR, _, _, (name, _, _)), hi, lo)):  # tmp[31:16]
      return (name, 32, None, hi, lo)
  return ('_unknown', 32, None, None, None)

def _extract_cat_lhs(lhs: UOp) -> list[tuple[str, int]]:
  """Extract list of (var_name, bits) from CAT LHS like { D1.u1, D0.u64 }."""
  if lhs.op != Ops.CAT: return None
  result = []
  for src in lhs.src:
    match src:
      case UOp(Ops.BITCAST, dt, (UOp(Ops.DEFINE_VAR, _, _, (name, _, _)),)):
        _, bits, _, _ = _dtype_info(dt)
        result.append((name, bits))
      case _: return None
  return result

def _gen_stmt(stmt, ctx: set, indent: int = 0) -> list[str]:
  """Generate Python statements."""
  p = '  ' * indent
  match stmt:
    case Assign(lhs, rhs) if lhs.op == Ops.CAT:
      # Handle concatenation assignment like { D1.u1, D0.u64 } = expr
      parts = _extract_cat_lhs(lhs)
      if parts is None: return [f'{p}# unhandled CAT assignment']
      rhs_expr = _gen_expr(rhs, ctx)
      lines = [f'{p}_cat_tmp = int({rhs_expr})']
      offset = 0
      for name, bits in reversed(parts):  # reversed because low bits come last in { hi, lo }
        ctx.add(name)
        mask = hex((1 << bits) - 1)
        lines.append(f'{p}{name} = (_cat_tmp >> {offset}) & {mask}')
        offset += bits
      return lines
    case Assign(lhs, rhs):
      name, bits, fconv, hi, lo = _extract_lhs(lhs)
      # Initialize variable if first use
      init = [f'{p}{name} = 0'] if name not in ctx and name not in _INPUTS else []
      ctx.add(name)
      rhs_expr = _gen_expr(rhs, ctx)
      mask = hex((1 << bits) - 1)
      # Convert to int: for float types use _toi32/etc (handles NaN), for int types use int()
      to_int = f'{fconv}({rhs_expr})' if fconv else f'int({rhs_expr})'
      # Handle slice assignment
      if hi is not None:
        h, l = _gen_expr(hi, ctx), _gen_expr(lo, ctx)
        if hi is lo:  # single bit
          return init + [f'{p}{name} = ({name} & ~(1 << {h})) | (({to_int} & 1) << {h})']
        return init + [f'{p}m = ((1 << ({h} - {l} + 1)) - 1) << {l}',
                       f'{p}{name} = ({name} & ~m) | (({to_int} << {l}) & m)']
      # Regular assignment - convert and mask to bit width
      if bits < 64: return init + [f'{p}{name} = {to_int} & {mask}']
      if fconv: return init + [f'{p}{name} = {to_int}']  # 64-bit float still needs conversion
      return init + [f'{p}{name} = {rhs_expr}']
    case Declare(name, _):
      ctx.add(name)
      return [f'{p}{name} = 0']
    case If(branches):
      lines = []
      for i, (cond, body) in enumerate(branches):
        kw = 'if' if i == 0 else ('elif' if cond else 'else')
        lines.append(f'{p}{kw} {_gen_expr(cond, ctx)}:' if cond else f'{p}else:')
        body_lines = [l for s in body for l in _gen_stmt(s, ctx, indent + 1)]
        lines.extend(body_lines or [f'{p}  pass'])
      return lines
    case For(var, start, end, body):
      ctx.add(var)
      lines = [f'{p}for {var} in range(int({_gen_expr(start, ctx)}), int({_gen_expr(end, ctx)}) + 1):']
      body_lines = [l for s in body for l in _gen_stmt(s, ctx, indent + 1)]
      lines.extend(body_lines or [f'{p}  pass'])
      return lines
    case Break(): return [f'{p}break']
    case Return(value): return [f'{p}return {_gen_expr(value, ctx)}']
  return []

def _gen_function(op_name: str, stmts: tuple, pcode: str) -> str:
  """Generate complete function."""
  ctx = set()  # tracks variables assigned in the body
  sig = f'def _{op_name}(S0, S1, S2, D0, SCC, VCC, laneId, EXEC, SIMM32, VGPR, SRC0=0, VDST=0, pc=0, opsel=0, opsel_hi=0):'
  # Generate body - add SIMM16 alias for SIMM32
  body = ['  SIMM16 = SIMM32'] + [l for s in stmts for l in _gen_stmt(s, ctx, 1)]
  # Generate return - use _PARAM mapping for variable names
  ret_items = [f"'{out}': {_PARAM.get(out, out)}" for out in _OUTPUTS if out in ctx]
  ret = f"  return {{{', '.join(ret_items)}}}"
  return '\n'.join([sig] + body + [ret])

# Sign extension helpers
_sext8 = lambda v: _sext(int(v) & 0xff, 8)
_sext16 = lambda v: _sext(int(v) & 0xffff, 16)
_sext32 = lambda v: _sext(int(v) & 0xffffffff, 32)
_sext64 = lambda v: _sext(int(v) & 0xffffffffffffffff, 64)

# Float-to-int with passthrough for already-int values (like f32_to_f16 returns)
_toi64 = lambda v: v if isinstance(v, int) else _i64(float(v))
_toi32 = lambda v: v if isinstance(v, int) else _i32(float(v))
_toi16 = lambda v: v if isinstance(v, int) else _i16(float(v))
_toibf16 = lambda v: v if isinstance(v, int) else _ibf16(float(v))

_GLOBALS = {
  'MASK32': MASK32, 'MASK64': MASK64, '_f32': _f32, '_i32': _i32, '_f64': _f64, '_i64': _i64,
  '_f16': _f16, '_i16': _i16, '_bf16': _bf16, '_ibf16': _ibf16, '_sext': _sext, '_brev': _brev, '_div': _div,
  '_sext8': _sext8, '_sext16': _sext16, '_sext32': _sext32, '_sext64': _sext64,
  '_toi64': _toi64, '_toi32': _toi32, '_toi16': _toi16, '_toibf16': _toibf16,
  'fma': fma, 'ldexp': ldexp, 'sqrt': sqrt, 'log2': log2, 'fract': fract, 'sin': sin, 'cos': cos,
  'trunc': trunc, 'floor': floor, 'ceil': ceil, 'exponent': exponent, 'sign': sign, 'mantissa': mantissa,
  'signext_from_bit': signext_from_bit, 'isNAN': isNAN, 'isQuietNAN': isQuietNAN, 'isSignalNAN': isSignalNAN,
  'cvtToQuietNAN': cvtToQuietNAN, 'v_min_f32': v_min_f32, 'v_max_f32': v_max_f32, 'v_min_f16': v_min_f16, 'v_max_f16': v_max_f16,
  'v_min_i32': v_min_i32, 'v_max_i32': v_max_i32,
  'f32_to_i32': f32_to_i32, 'f32_to_u32': f32_to_u32, 'f64_to_i32': f64_to_i32, 'f64_to_u32': f64_to_u32,
  'i32_to_f32': i32_to_f32, 'u32_to_f32': u32_to_f32, 'f32_to_f16': f32_to_f16, 'f16_to_f32': f16_to_f32,
  'f32_to_f64': f32_to_f64, 'f64_to_f32': f64_to_f32, 'i32_to_f64': i32_to_f64, 'u32_to_f64': u32_to_f64,
  'bf16_to_f32': bf16_to_f32, 'f32_to_bf16': f32_to_bf16, 'f16_to_i16': f16_to_i16, 'f16_to_u16': f16_to_u16,
  'ABSDIFF': ABSDIFF, 'BYTE_PERMUTE': BYTE_PERMUTE,
  'v_sad_u8': v_sad_u8, 'v_msad_u8': v_msad_u8, 's_ff1_i32_b32': s_ff1_i32_b32, 's_ff1_i32_b64': s_ff1_i32_b64,
  'abs': abs, 'min': min, 'max': max, 'int': int, 'float': float,
  'PI': PI, 'DENORM': DENORM, 'INF': INF, 'OVERFLOW_F32': OVERFLOW_F32, 'OVERFLOW_F64': OVERFLOW_F64,
  'UNDERFLOW_F32': UNDERFLOW_F32, 'UNDERFLOW_F64': UNDERFLOW_F64, 'TWO_OVER_PI_1201': TWO_OVER_PI_1201,
}

@functools.cache
def compile_exec(op_name: str, pseudocode: str):
  """Compile pseudocode to executable function. Returns None if can't handle."""
  if 'MEM[' in pseudocode or 'LDS[' in pseudocode or 'VGPR[' in pseudocode: return None  # skip memory ops
  try:
    pcode = _apply_pseudocode_fixes(op_name, pseudocode)
    stmts = parse(pcode)
    fn_code = _gen_function(op_name, stmts, pcode)
    local_ns = {}
    exec(fn_code, _GLOBALS, local_ns)
    return local_ns[f'_{op_name}']
  except Exception:
    return None
