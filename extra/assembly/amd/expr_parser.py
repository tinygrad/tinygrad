# Tokenizer-based expression parser for AMD pcode
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp

DTYPES = {'u32': dtypes.uint32, 'i32': dtypes.int, 'f32': dtypes.float32, 'b32': dtypes.uint32, 'u64': dtypes.uint64, 'i64': dtypes.int64,
          'f64': dtypes.float64, 'b64': dtypes.uint64, 'u16': dtypes.uint16, 'i16': dtypes.short, 'f16': dtypes.half, 'b16': dtypes.uint16,
          'u8': dtypes.uint8, 'i8': dtypes.int8, 'b8': dtypes.uint8, 'u1': dtypes.uint32}
_BITS_DT = {8: dtypes.uint8, 16: dtypes.uint16, 32: dtypes.uint32, 64: dtypes.uint64}

def _const(dt, v): return UOp.const(dt, v)
def _u32(v): return _const(dtypes.uint32, v)
def _u64(v): return _const(dtypes.uint64, v)
def _to_u32(v): return v if v.dtype == dtypes.uint32 else v.bitcast(dtypes.uint32) if v.dtype.itemsize == 4 else v.cast(dtypes.uint32)
def _to_bool(v): return v if v.dtype == dtypes.bool else v.ne(_const(v.dtype, 0))
def _cast_to(v, dt):
  if v.dtype == dt: return v
  if dt == dtypes.half: return v.cast(dtypes.uint16).bitcast(dtypes.half)
  return v.cast(dt) if dt.itemsize != v.dtype.itemsize else v.bitcast(dt)

# Float bit extraction - returns (bits, exp_mask, mant_mask, quiet_bit, exp_shift) based on float type
def _float_info(v: UOp) -> tuple[UOp, UOp, UOp, UOp, int]:
  if v.dtype in (dtypes.float64, dtypes.uint64):
    bits = v.bitcast(dtypes.uint64) if v.dtype == dtypes.float64 else v.cast(dtypes.uint64)
    return bits, _u64(0x7FF0000000000000), _u64(0x000FFFFFFFFFFFFF), _u64(0x0008000000000000), 52
  if v.dtype in (dtypes.half, dtypes.uint16):
    bits = (v.bitcast(dtypes.uint16) if v.dtype == dtypes.half else (v & _u32(0xFFFF)).cast(dtypes.uint16)).cast(dtypes.uint32)
    return bits, _u32(0x7C00), _u32(0x03FF), _u32(0x0200), 10
  bits = v.bitcast(dtypes.uint32) if v.dtype == dtypes.float32 else v.cast(dtypes.uint32)
  return bits, _u32(0x7F800000), _u32(0x007FFFFF), _u32(0x00400000), 23

def _isnan(v: UOp) -> UOp:
  bits, exp_m, mant_m, _, _ = _float_info(v.cast(dtypes.float32) if v.dtype == dtypes.half else v)
  return (bits & exp_m).eq(exp_m) & (bits & mant_m).ne(_const(bits.dtype, 0))

def _bitreverse(v: UOp, bits: int) -> UOp:
  dt, masks = (dtypes.uint64, [(0x5555555555555555,1),(0x3333333333333333,2),(0x0F0F0F0F0F0F0F0F,4),(0x00FF00FF00FF00FF,8),(0x0000FFFF0000FFFF,16)]) \
    if bits == 64 else (dtypes.uint32, [(0x55555555,1),(0x33333333,2),(0x0F0F0F0F,4),(0x00FF00FF,8)])
  v = v.cast(dt) if v.dtype != dt else v
  for m, s in masks: v = ((v >> _const(dt, s)) & _const(dt, m)) | ((v & _const(dt, m)) << _const(dt, s))
  return (v >> _const(dt, 32 if bits == 64 else 16)) | (v << _const(dt, 32 if bits == 64 else 16))

def _extract_bits(val: UOp, hi: int, lo: int) -> UOp:
  dt = dtypes.uint64 if val.dtype in (dtypes.uint64, dtypes.int64) else dtypes.uint32
  return ((val >> _const(dt, lo)) if lo > 0 else val) & _const(val.dtype, (1 << (hi - lo + 1)) - 1)

def _set_bit(old, pos, val):
  mask = _u32(1) << pos
  return (old & (mask ^ _u32(0xFFFFFFFF))) | ((val.cast(dtypes.uint32) & _u32(1)) << pos)

def _val_to_bits(val):
  if val.dtype == dtypes.half: return val.bitcast(dtypes.uint16).cast(dtypes.uint32)
  if val.dtype == dtypes.float32: return val.bitcast(dtypes.uint32)
  if val.dtype == dtypes.float64: return val.bitcast(dtypes.uint64)
  return val if val.dtype == dtypes.uint32 else val.cast(dtypes.uint32)

def _floor(x): t = UOp(Ops.TRUNC, x.dtype, (x,)); return ((x < _const(x.dtype, 0)) & x.ne(t)).where(t - _const(x.dtype, 1), t)
def _f16_extract(v): return (v & _u32(0xFFFF)).cast(dtypes.uint16).bitcast(dtypes.half) if v.dtype == dtypes.uint32 else v

def _check_nan(v: UOp, quiet: bool) -> UOp:
  if v.op == Ops.CAST and v.dtype == dtypes.float64: v = v.src[0]
  bits, exp_m, mant_m, qb, _ = _float_info(v)
  is_nan_exp, has_mant, is_q = (bits & exp_m).eq(exp_m), (bits & mant_m).ne(_const(bits.dtype, 0)), (bits & qb).ne(_const(bits.dtype, 0))
  return (is_nan_exp & is_q) if quiet else (is_nan_exp & has_mant & is_q.logical_not())

def _minmax_reduce(is_max, dt, args):
  def cast(v): return v.bitcast(dt) if dt == dtypes.float32 and v.dtype == dtypes.uint32 else v.cast(dt)
  def minmax(a, b):
    if dt in (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64):
      return (a > b).where(a, b) if is_max else (a < b).where(a, b)
    return a.maximum(b) if is_max else a.minimum(b)
  result = cast(args[0])
  for a in args[1:]:
    b = cast(a)
    if dt == dtypes.float32: result = _isnan(result).where(b, _isnan(b).where(result, minmax(result, b)))
    else: result = minmax(result, b)
  return result

# Token types
class Token:
  __slots__ = ('type', 'val')
  def __init__(self, type: str, val: str): self.type, self.val = type, val
  def __repr__(self): return f'{self.type}:{self.val}'

def tokenize(s: str) -> list[Token]:
  tokens, i, n = [], 0, len(s)
  while i < n:
    c = s[i]
    if c.isspace(): i += 1; continue
    if i + 1 < n and s[i:i+2] in ('||', '&&', '>=', '<=', '==', '!=', '<>', '>>', '<<', '**', '+:', '-:'):
      tokens.append(Token('OP', s[i:i+2])); i += 2; continue
    if c in '|^&><+-*/~!%': tokens.append(Token('OP', c)); i += 1; continue
    if c == '(': tokens.append(Token('LPAREN', c)); i += 1; continue
    if c == ')': tokens.append(Token('RPAREN', c)); i += 1; continue
    if c == '[': tokens.append(Token('LBRACKET', c)); i += 1; continue
    if c == ']': tokens.append(Token('RBRACKET', c)); i += 1; continue
    if c == '{': tokens.append(Token('LBRACE', c)); i += 1; continue
    if c == '}': tokens.append(Token('RBRACE', c)); i += 1; continue
    if c == ':': tokens.append(Token('COLON', c)); i += 1; continue
    if c == ',': tokens.append(Token('COMMA', c)); i += 1; continue
    if c == '?': tokens.append(Token('QUESTION', c)); i += 1; continue
    if c == '.': tokens.append(Token('DOT', c)); i += 1; continue
    if c == '=': tokens.append(Token('EQUALS', c)); i += 1; continue
    if c == "'": tokens.append(Token('QUOTE', c)); i += 1; continue
    if c == ';': i += 1; continue
    if c.isdigit() or (c == '-' and i + 1 < n and s[i+1].isdigit()):
      start = i
      if c == '-': i += 1
      if i + 1 < n and s[i] == '0' and s[i+1] in 'xX':
        i += 2
        while i < n and s[i] in '0123456789abcdefABCDEF': i += 1
      else:
        while i < n and s[i].isdigit(): i += 1
        if i < n and s[i] == '.' and i + 1 < n and s[i+1].isdigit():
          i += 1
          while i < n and s[i].isdigit(): i += 1
      for sfx in ('ULL', 'LL', 'UL', 'U', 'L', 'F', 'f'):
        if s[i:i+len(sfx)] == sfx: i += len(sfx); break
      tokens.append(Token('NUM', s[start:i])); continue
    if c.isalpha() or c == '_':
      start = i
      while i < n and (s[i].isalnum() or s[i] == '_'): i += 1
      tokens.append(Token('IDENT', s[start:i])); continue
    raise RuntimeError(f"unexpected char '{c}' at pos {i} in: {s}")
  tokens.append(Token('EOF', ''))
  return tokens

class Parser:
  def __init__(self, tokens: list[Token], vars: dict, funcs: dict, expr: str):
    self.tokens, self.vars, self.funcs, self.expr, self.pos = tokens, vars, funcs, expr, 0

  def peek(self, offset=0) -> Token: return self.tokens[min(self.pos + offset, len(self.tokens) - 1)]
  def at(self, *types) -> bool: return self.peek().type in types
  def at_val(self, *vals) -> bool: return self.peek().val in vals
  def eat(self, type: str) -> Token:
    if self.peek().type != type: raise RuntimeError(f"expected {type}, got {self.peek()} in: {self.expr}")
    tok = self.tokens[self.pos]; self.pos += 1; return tok
  def try_eat(self, type: str) -> Token | None:
    if self.peek().type == type: return self.eat(type)
    return None
  def try_eat_val(self, val: str) -> Token | None:
    if self.peek().val == val: tok = self.tokens[self.pos]; self.pos += 1; return tok
    return None

  def parse(self) -> UOp: return self.ternary()
  def expr_top(self) -> UOp: return self.ternary()

  def ternary(self) -> UOp:
    cond = self.binop(0)
    if self.try_eat('QUESTION'):
      then_val, else_val = self.ternary(), (self.eat('COLON'), self.ternary())[1]
      return _to_bool(cond).where(then_val, else_val)
    return cond

  def _apply_binop(self, left, right, op):
    if op in ('||', '&&', '|', '^', '&'): left, right = self._coerce_bitwise(left, right)
    elif op in ('>=', '<=', '>', '<', '==', '!=', '<>', '>>', '<<'): left, right = self._coerce_cmp(left, right)
    elif left.dtype != right.dtype: right = right.cast(left.dtype)
    match op:
      case '||' | '|': return left | right
      case '&&' | '&': return left & right
      case '^': return left ^ right
      case '==' | '<>': return left.eq(right) if op == '==' else left.ne(right)
      case '!=' : return left.ne(right)
      case '>=' | '<=' | '>' | '<': return self._cmp_nan(left, right, {'>=':(lambda a,b:a>=b),'<=':(lambda a,b:a<=b),'>':(lambda a,b:a>b),'<':(lambda a,b:a<b)}[op])
      case '>>' | '<<': return (left >> right) if op == '>>' else (left << right)
      case '+' | '-':
        if op == '-' and left.op == Ops.CONST and right.op == Ops.CONST: return _const(left.dtype, left.arg - right.arg)
        return (left + right) if op == '+' else (left - right)
      case '*' | '/': return (left * right) if op == '*' else (left / right)
      case '**': return UOp(Ops.EXP2, left.dtype, (right.cast(left.dtype),)) if left.op == Ops.CONST and left.arg == 2.0 else left

  _PREC = [('||',), ('&&',), ('|',), ('^',), ('&',), ('==', '!=', '<>'), ('>=', '<=', '>', '<'), ('>>', '<<'), ('+', '-'), ('*', '/'), ('**',)]

  def binop(self, prec: int) -> UOp:
    if prec >= len(self._PREC): return self.unary()
    left = self.binop(prec + 1)
    ops = self._PREC[prec]
    while self.at('OP') and self.peek().val in ops:
      op = self.eat('OP').val
      left = self._apply_binop(left, self.binop(prec + 1), op)
    return left

  def unary(self) -> UOp:
    if self.at('OP') and self.peek().val == '~':
      self.eat('OP'); inner = self.unary()
      return inner ^ _const(inner.dtype, (1 << (inner.dtype.itemsize * 8)) - 1)
    if self.at('OP') and self.peek().val == '!':
      self.eat('OP'); inner = self.unary()
      return inner.eq(_const(inner.dtype, 0))
    if self.at('OP') and self.peek().val == '-':
      self.eat('OP'); inner = self.unary()
      if inner.op == Ops.CONST:
        return _const(dtypes.int if inner.dtype == dtypes.uint32 else inner.dtype, -inner.arg)
      return inner.neg()
    if self.at('OP') and self.peek().val == '+':
      self.eat('OP')
      return self.unary()
    return self.postfix()

  def postfix(self) -> UOp:
    base = self.primary()
    while True:
      if self.try_eat('DOT'):
        field = self.eat('IDENT').val
        base = self._handle_dot(base, field)
      elif self.at('LBRACKET'):
        base = self._handle_bracket(base)
      elif self.at('LBRACE'):
        base = self._handle_brace_index(base)
      else:
        break
    return base

  def primary(self) -> UOp:
    if self.try_eat('LPAREN'):
      e = self.expr_top()
      self.eat('RPAREN')
      return e
    if self.try_eat('LBRACE'):
      hi = self.expr_top()
      self.eat('COMMA')
      lo = self.expr_top()
      self.eat('RBRACE')
      return (hi.cast(dtypes.uint64) << _u64(32)) | lo.cast(dtypes.uint64)
    if self.at('NUM'):
      num = self.eat('NUM').val
      if self.try_eat('QUOTE'):
        return self._sized_literal(int(num.rstrip('ULlf')))
      return self._parse_number(num)
    if self.at('IDENT'):
      name = self.eat('IDENT').val
      if name == 'MEM':
        self.eat('LBRACKET')
        addr = self.expr_top()
        self.eat('RBRACKET')
        self.eat('DOT')
        dt_name = self.eat('IDENT').val
        return self._handle_mem_load(addr, DTYPES.get(dt_name, dtypes.uint32))
      if name == 'VGPR':
        self.eat('LBRACKET')
        lane = self.expr_top()
        self.eat('RBRACKET')
        self.eat('LBRACKET')
        reg = self.expr_top()
        self.eat('RBRACKET')
        vgpr = self.vars.get('_vgpr')
        if vgpr is None: return _u32(0)
        return vgpr.index((_to_u32(reg) * _u32(32) + _to_u32(lane)).cast(dtypes.index), ptr=True).load()
      if self.try_eat('LPAREN'):
        args = self._parse_args()
        self.eat('RPAREN')
        return self._call_func(name, args)
      if name == 'PI': return _const(dtypes.float32, 3.141592653589793)
      if name == 'INF': return _const(dtypes.float64, float('inf'))
      if name == 'NAN': return _const(dtypes.uint32, 0x7FC00000).bitcast(dtypes.float32)
      if name == 'UNDERFLOW_F32': return _const(dtypes.uint32, 1).bitcast(dtypes.float32)
      if name == 'OVERFLOW_F32': return _const(dtypes.uint32, 0x7F7FFFFF).bitcast(dtypes.float32)
      if name == 'UNDERFLOW_F64': return _const(dtypes.uint64, 1).bitcast(dtypes.float64)
      if name == 'OVERFLOW_F64': return _const(dtypes.uint64, 0x7FEFFFFFFFFFFFFF).bitcast(dtypes.float64)
      if self.at('LBRACE'):
        self.eat('LBRACE')
        idx = self.eat('NUM').val
        self.eat('RBRACE')
        elem = self.vars.get(f'{name}{idx}', _u32(0))
        if self.try_eat('DOT'):
          dt_name = self.eat('IDENT').val
          return _cast_to(elem, DTYPES.get(dt_name, dtypes.uint32))
        if self.at('LBRACKET'):
          return self._handle_bracket_with_name(elem, name + idx)
        return elem
      if self.at('LBRACKET') and name not in self.vars:
        self.eat('LBRACKET')
        if self.at('NUM'):
          idx = int(self.peek().val)
          if f'{name}{idx}' in self.vars:
            self.eat('NUM')
            self.eat('RBRACKET')
            elem = self.vars[f'{name}{idx}']
            if self.try_eat('DOT'):
              dt_name = self.eat('IDENT').val
              return _cast_to(elem, DTYPES.get(dt_name, dtypes.uint32))
            return elem
        first = self.expr_top()
        return self._handle_bracket_rest(first, _u32(0), name)
      if name in self.vars:
        v = self.vars[name]
        return v if isinstance(v, UOp) else _u32(0) if isinstance(v, dict) else _u32(0)
      return _u32(0)
    raise RuntimeError(f"unexpected token in primary: {self.peek()} in: {self.expr}")

  def _handle_dot(self, base, field: str) -> UOp:
    if isinstance(base, str): return _u32(0)
    if not isinstance(base, UOp):
      if isinstance(base, dict): return base.get(field, _u32(0))
      return _u32(0)
    if field == 'u64' and self.at('LBRACKET') and self.peek(1).type == 'IDENT' and self.peek(1).val == 'laneId':
      self.eat('LBRACKET')
      self.eat('IDENT')
      self.eat('RBRACKET')
      result = (base >> _to_u32(self.vars['laneId'])) & _u32(1)
      if self.try_eat('DOT'):
        dt_name = self.eat('IDENT').val
        return result.cast(DTYPES.get(dt_name, dtypes.uint32))
      return result
    dt = DTYPES.get(field)
    if dt is None: return base
    if dt == base.dtype: return base
    if dt.itemsize == 2 and base.dtype.itemsize == 4:
      return (base & _const(base.dtype, 0xFFFF)).cast(dtypes.uint16) if dt == dtypes.uint16 else (base & _const(base.dtype, 0xFFFF)).cast(dtypes.uint16).bitcast(dt)
    return _cast_to(base, dt)

  def _handle_bracket(self, base, var_name: str | None = None) -> UOp:
    self.eat('LBRACKET')
    first = self.expr_top()
    return self._handle_bracket_rest(first, base, var_name)

  def _handle_bracket_with_name(self, base, var_name: str) -> UOp:
    self.eat('LBRACKET')
    first = self.expr_top()
    return self._handle_bracket_rest(first, base, var_name)

  def _handle_bracket_rest(self, first: UOp, base: UOp, var_name: str | None = None) -> UOp:
    if self.at('OP') and self.peek().val in ('+:', '-:'):
      op = self.eat('OP').val
      width = self.expr_top()
      self.eat('RBRACKET')
      if width.op == Ops.CONST:
        w = int(width.arg)
        return (base >> _to_u32(first)) & _const(base.dtype, (1 << w) - 1)
      return base
    if self.try_eat('COLON'):
      second = self.expr_top()
      self.eat('RBRACKET')
      if first.op == Ops.CONST and second.op == Ops.CONST:
        a, b = int(first.arg), int(second.arg)
        if a < b: return _bitreverse(base, b - a + 1)
        hi, lo = a, b
        if lo >= base.dtype.itemsize * 8:
          vn = var_name or self._find_var_name(base)
          if vn and f'{vn}{lo // 32}' in self.vars:
            base = self.vars[f'{vn}{lo // 32}']
            lo, hi = lo % 32, (hi % 32) + (lo % 32)
        return _extract_bits(base, hi, lo)
      return base
    self.eat('RBRACKET')
    dt_suffix = None
    if self.try_eat('DOT'):
      dt_suffix = DTYPES.get(self.eat('IDENT').val, dtypes.uint32)
    if var_name is None:
      var_name = self._find_var_name(base)
    if first.op == Ops.CONST:
      idx = int(first.arg)
      if var_name and f'{var_name}{idx}' in self.vars:
        v = self.vars[f'{var_name}{idx}']
        return _cast_to(v, dt_suffix) if dt_suffix else v
      dt = dtypes.uint64 if base.dtype in (dtypes.uint64, dtypes.int64) else dtypes.uint32
      base_cast = base.cast(dt) if base.dtype != dt else base
      result = ((base_cast >> _const(dt, idx)) & _const(dt, 1))
      return _cast_to(result, dt_suffix) if dt_suffix else result
    if var_name:
      idx_u32 = _to_u32(first)
      elems = [(i, self.vars[f'{var_name}{i}']) for i in range(256) if f'{var_name}{i}' in self.vars]
      if elems:
        result = elems[-1][1]
        for ei, ev in reversed(elems[:-1]):
          if ev.dtype != result.dtype and ev.dtype.itemsize == result.dtype.itemsize: result = result.cast(ev.dtype)
          elif ev.dtype != result.dtype: ev = ev.cast(result.dtype)
          result = idx_u32.eq(_u32(ei)).where(ev, result)
        return result
    dt = dtypes.uint64 if base.dtype in (dtypes.uint64, dtypes.int64) else dtypes.uint32
    base_cast = base.cast(dt) if base.dtype != dt else base
    result = (base_cast >> first.cast(dt)) & _const(dt, 1)
    return _cast_to(result, dt_suffix) if dt_suffix else result

  def _handle_brace_index(self, base) -> UOp:
    self.eat('LBRACE')
    idx = self.eat('NUM').val
    self.eat('RBRACE')
    var_name = self._find_var_name(base)
    if var_name:
      elem = self.vars.get(f'{var_name}{idx}', _u32(0))
      if self.try_eat('DOT'):
        dt_name = self.eat('IDENT').val
        return _cast_to(elem, DTYPES.get(dt_name, dtypes.uint32))
      if self.at('LBRACKET'):
        return self._handle_bracket(elem)
      return elem
    return _u32(0)

  def _find_var_name(self, base: UOp) -> str | None:
    if base.op == Ops.DEFINE_VAR and base.arg: return base.arg[0]
    for name, v in self.vars.items():
      if isinstance(v, UOp) and v is base: return name
    return None

  def _sized_literal(self, bits: int) -> UOp:
    if self.at('IDENT') and self.peek().val in ('U', 'I', 'F', 'B'):
      type_char = self.eat('IDENT').val
      self.eat('LPAREN')
      inner = self.expr_top()
      self.eat('RPAREN')
      dt = {('U',32): dtypes.uint32, ('U',64): dtypes.uint64, ('I',32): dtypes.int, ('I',64): dtypes.int64,
            ('F',32): dtypes.float32, ('F',64): dtypes.float64, ('B',32): dtypes.uint32, ('B',64): dtypes.uint64}.get((type_char, bits), dtypes.uint64 if bits > 32 else dtypes.uint32)
      if type_char == 'F' and inner.dtype in (dtypes.uint32, dtypes.uint64, dtypes.ulong, dtypes.int, dtypes.int64):
        if inner.dtype.itemsize != dt.itemsize: inner = inner.cast(dtypes.uint32 if dt.itemsize == 4 else dtypes.uint64)
        return inner.bitcast(dt)
      return inner.cast(dt)
    if self.at('IDENT'):
      ident = self.peek().val
      fmt = ident[0].lower()
      if fmt in ('h', 'b', 'd'):
        self.eat('IDENT')
        if len(ident) > 1: num = ident[1:]
        elif self.at('NUM'): num = self.eat('NUM').val
        elif self.at('IDENT'): num = self.eat('IDENT').val
        else: raise RuntimeError(f"expected number after {bits}'{fmt} in: {self.expr}")
        if fmt == 'h': val = int(num, 16)
        elif fmt == 'b': val = int(num, 2)
        else: val = int(num)
        return _const(_BITS_DT.get(bits, dtypes.uint32), val)
    if self.at('NUM') and self.peek().val.startswith('0x'):
      num = self.eat('NUM').val
      return _const(_BITS_DT.get(bits, dtypes.uint32), int(num, 16))
    if self.at('NUM') or (self.at('OP') and self.peek().val == '-'):
      neg = self.try_eat_val('-') is not None
      num = self.eat('NUM').val
      suffix = ''
      for sfx in ('ULL', 'LL', 'UL', 'U', 'L', 'F', 'f'):
        if num.endswith(sfx): suffix, num = sfx, num[:-len(sfx)]; break
      if num.startswith('0x'):
        val = int(num, 16)
        if neg: val = -val
      elif '.' in num:
        val = float(num)
        if neg: val = -val
        return _const({16: dtypes.half, 32: dtypes.float32, 64: dtypes.float64}.get(bits, dtypes.float32), val)
      else:
        val = int(num)
        if neg: val = -val
      dt = {1: dtypes.uint32, 8: dtypes.uint8, 16: dtypes.int16 if 'U' not in suffix else dtypes.uint16,
            32: dtypes.int if 'U' not in suffix else dtypes.uint32, 64: dtypes.int64 if 'U' not in suffix else dtypes.uint64}.get(bits, dtypes.uint32)
      return _const(dt, val)
    raise RuntimeError(f"unexpected token after {bits}': {self.peek()} in: {self.expr}")

  def _parse_number(self, num: str) -> UOp:
    suffix = ''
    if num.startswith('0x') or num.startswith('0X'):
      for sfx in ('ULL', 'LL', 'UL', 'U', 'L'):
        if num.endswith(sfx): suffix, num = sfx, num[:-len(sfx)]; break
      return _const(dtypes.uint64, int(num, 16))
    for sfx in ('ULL', 'LL', 'UL', 'U', 'L', 'F', 'f'):
      if num.endswith(sfx): suffix, num = sfx, num[:-len(sfx)]; break
    if '.' in num or suffix in ('F', 'f'):
      return _const(dtypes.float32 if suffix in ('F', 'f') else dtypes.float64, float(num))
    val = int(num)
    if 'ULL' in suffix: return _const(dtypes.uint64, val)
    if 'LL' in suffix or 'L' in suffix: return _const(dtypes.uint64, val)
    if 'U' in suffix: return _const(dtypes.uint32, val)
    return _const(dtypes.int if val < 0 else dtypes.uint32, val)

  def _parse_args(self) -> list[UOp]:
    if self.at('RPAREN'): return []
    args = [self.expr_top()]
    while self.try_eat('COMMA'):
      args.append(self.expr_top())
    return args

  def _call_func(self, name: str, args: list[UOp]) -> UOp:
    if name in self.vars and isinstance(self.vars[name], tuple) and self.vars[name][0] == 'lambda':
      _, params, body = self.vars[name]
      lv = {**self.vars, **{p: a for p, a in zip(params, args)}}
      if ';' in body or '\n' in body or 'return' in body.lower():
        return _parse_lambda_body(body, lv, self.funcs)
      return parse_expr(body, lv, self.funcs)
    if name in self.funcs:
      return self.funcs[name](args, self.vars)
    raise RuntimeError(f"unknown function: {name} in: {self.expr}")

  def _handle_mem_load(self, addr: UOp, dt) -> UOp:
    mem = self.vars.get('_vmem') if '_vmem' in self.vars else self.vars.get('_lds')
    if mem is None: return _const(dt, 0)
    adt = dtypes.uint64 if addr.dtype == dtypes.uint64 else dtypes.uint32
    active = self.vars.get('_active')
    byte_mem = mem.dtype.base == dtypes.uint8
    if byte_mem:
      idx = addr.cast(dtypes.index)
      idx = idx.valid(active) if active is not None else idx
      if dt in (dtypes.uint64, dtypes.int64, dtypes.float64):
        val = _u32(0).cast(dtypes.uint64)
        for i in range(8): val = val | (mem.index(idx + _const(dtypes.index, i), ptr=True).load().cast(dtypes.uint64) << _u64(i * 8))
      elif dt in (dtypes.uint8, dtypes.int8):
        val = mem.index(idx, ptr=True).load().cast(dt)
      elif dt in (dtypes.uint16, dtypes.int16, dtypes.short):
        val = (mem.index(idx, ptr=True).load().cast(dtypes.uint32) | (mem.index(idx + _const(dtypes.index, 1), ptr=True).load().cast(dtypes.uint32) << _u32(8))).cast(dt)
      else:
        val = _u32(0)
        for i in range(4): val = val | (mem.index(idx + _const(dtypes.index, i), ptr=True).load().cast(dtypes.uint32) << _u32(i * 8))
    else:
      idx = (addr >> _const(addr.dtype, 2)).cast(dtypes.index)
      idx = idx.valid(active) if active is not None else idx
      val = mem.index(idx)
      if dt in (dtypes.uint64, dtypes.int64, dtypes.float64):
        idx2 = ((addr + _const(adt, 4)) >> _const(adt, 2)).cast(dtypes.index)
        idx2 = idx2.valid(active) if active is not None else idx2
        val = val.cast(dtypes.uint64) | (mem.index(idx2).cast(dtypes.uint64) << _u64(32))
      elif dt in (dtypes.uint8, dtypes.int8): val = (val >> ((addr & _const(adt, 3)).cast(dtypes.uint32) * _u32(8))) & _u32(0xFF)
      elif dt in (dtypes.uint16, dtypes.int16): val = (val >> (((addr >> _const(adt, 1)) & _const(adt, 1)).cast(dtypes.uint32) * _u32(16))) & _u32(0xFFFF)
    return val

  def _coerce_cmp(self, l: UOp, r: UOp) -> tuple[UOp, UOp]:
    if l.dtype != r.dtype:
      if r.dtype == dtypes.int and r.op == Ops.CONST and r.arg < 0: l = l.cast(dtypes.int)
      else: r = r.cast(l.dtype)
    return l, r

  def _coerce_bitwise(self, l: UOp, r: UOp) -> tuple[UOp, UOp]:
    if l.dtype != r.dtype:
      if l.dtype.itemsize == r.dtype.itemsize:
        t = dtypes.uint32 if l.dtype.itemsize == 4 else dtypes.uint64 if l.dtype.itemsize == 8 else l.dtype
        l, r = l.bitcast(t), r.bitcast(t)
      else: r = r.cast(l.dtype)
    return l, r

  def _cmp_nan(self, l: UOp, r: UOp, fn) -> UOp:
    result = fn(l, r)
    if l.dtype in (dtypes.float32, dtypes.float64, dtypes.half):
      return result & _isnan(l).logical_not() & _isnan(r).logical_not()
    return result

# Lambda body parser using tokenizer
def _parse_lambda_body(body: str, vars: dict[str, UOp], funcs: dict) -> UOp:
  lines = [l.strip() for l in body.replace(';', '\n').split('\n') if l.strip() and not l.strip().startswith('//')]
  return _parse_lambda_block(lines, 0, vars, funcs)[1]

def _line_starts_with(line: str, keyword: str) -> bool:
  """Check if line starts with keyword (case insensitive)"""
  tokens = tokenize(line)
  return tokens[0].type == 'IDENT' and tokens[0].val.lower() == keyword.lower()

def _parse_lambda_block(lines: list[str], start: int, vars: dict[str, UOp], funcs: dict) -> tuple[int, UOp]:
  i = start
  while i < len(lines):
    line = lines[i]
    tokens = tokenize(line)
    if tokens[0].type != 'IDENT': i += 1; continue
    first = tokens[0].val.lower()

    # Block terminators
    if first in ('elsif', 'else', 'endif', 'endfor'): break

    # return expr
    if first == 'return':
      rest = line[line.lower().find('return') + 6:].strip()
      return i + 1, parse_expr(rest, vars, funcs)

    # for var in start:end do
    if first == 'for':
      # Parse: for VAR in NUM : NUM do
      p = Parser(tokens, vars, funcs, line)
      p.eat('IDENT')  # for
      loop_var = p.eat('IDENT').val
      p.eat('IDENT')  # in
      start_val = int(p.eat('NUM').val.rstrip('U'))
      p.eat('COLON')
      end_val = int(p.eat('NUM').val.rstrip('U'))
      # Collect body
      i += 1; body_lines, depth = [], 1
      while i < len(lines) and depth > 0:
        btoks = tokenize(lines[i])
        if btoks[0].type == 'IDENT':
          if btoks[0].val.lower() == 'for': depth += 1
          elif btoks[0].val.lower() == 'endfor': depth -= 1
        if depth > 0: body_lines.append(lines[i])
        i += 1
      # Execute loop
      for li in range(start_val, end_val + 1):
        for bl in body_lines:
          # Substitute loop var everywhere (like old regex: re.sub(rf'\b{lv}\b', str(li), bl))
          toks = tokenize(bl)
          subst_parts = [str(li) if t.type == 'IDENT' and t.val == loop_var else t.val for t in toks if t.type != 'EOF']
          subst = ' '.join(subst_parts)
          # Evaluate bracket expressions (like old: re.sub(r'\[([^\]\[]+?)\s*:\s*([^\]\[]+?)\]', ...))
          stoks = tokenize(subst)
          eval_parts = []
          j = 0
          while j < len(stoks):
            if stoks[j].type == 'LBRACKET':
              # Find matching RBRACKET and evaluate any : range
              depth, start_j = 1, j
              j += 1
              while j < len(stoks) and depth > 0:
                if stoks[j].type == 'LBRACKET': depth += 1
                elif stoks[j].type == 'RBRACKET': depth -= 1
                j += 1
              inner = ' '.join(t.val for t in stoks[start_j+1:j-1] if t.type != 'EOF')
              if ':' in inner and '+:' not in inner and '-:' not in inner:
                # Evaluate range like "i * 8 + 7 : i * 8"
                parts = inner.split(':')
                if len(parts) == 2:
                  inner = f'{_try_eval(parts[0].strip())} : {_try_eval(parts[1].strip())}'
              eval_parts.append(f'[{inner}]')
            elif stoks[j].type != 'EOF':
              eval_parts.append(stoks[j].val)
              j += 1
            else:
              j += 1
          subst = ' '.join(eval_parts)
          # Now parse assignment: VAR[IDX] = EXPR
          stoks = tokenize(subst)
          if len(stoks) >= 5 and stoks[0].type == 'IDENT' and stoks[1].type == 'LBRACKET' and stoks[2].type == 'NUM' and stoks[3].type == 'RBRACKET':
            var_name, idx = stoks[0].val, int(stoks[2].val)
            for k, t in enumerate(stoks):
              if t.type == 'EQUALS':
                rhs = subst[subst.find('=') + 1:].strip()
                vars[f'{var_name}{idx}'] = parse_expr(rhs, vars, funcs)
                break
      continue

    # declare
    if first == 'declare': i += 1; continue

    # if cond then
    if first == 'if':
      # Extract condition between 'if' and 'then'
      line_lower = line.lower()
      if_idx = line_lower.find('if')
      then_idx = line_lower.rfind('then')
      cond_str = line[if_idx + 2:then_idx].strip() if then_idx > 0 else line[if_idx + 2:].strip()
      conds = [(_to_bool(parse_expr(cond_str, vars, funcs)), None)]
      i += 1; i, rv = _parse_lambda_block(lines, i, vars, funcs); conds[0] = (conds[0][0], rv)
      while i < len(lines):
        ltoks = tokenize(lines[i])
        if ltoks[0].type != 'IDENT': break
        lf = ltoks[0].val.lower()
        if lf == 'elsif':
          ll = lines[i].lower()
          ei, ti = ll.find('elsif'), ll.rfind('then')
          cond_str = lines[i][ei + 5:ti].strip() if ti > 0 else lines[i][ei + 5:].strip()
          i += 1; i, rv = _parse_lambda_block(lines, i, vars, funcs)
          conds.append((_to_bool(parse_expr(cond_str, vars, funcs)), rv))
        elif lf == 'else':
          i += 1; i, er = _parse_lambda_block(lines, i, vars, funcs)
          result = er
          for c, rv in reversed(conds):
            if rv is not None:
              if rv.dtype != result.dtype and rv.dtype.itemsize == result.dtype.itemsize: result = result.cast(rv.dtype)
              result = c.where(rv, result)
          return i, result
        elif lf == 'endif': i += 1; break
        else: break
      continue
    i += 1
  return i, _u32(0)

# Built-in function registry
_FUNCS: dict[str, callable] = {}

def _register_funcs():
  def _find_two_pi_mul(x):
    if x.op != Ops.MUL or len(x.src) != 2: return None
    for i, s in enumerate(x.src):
      if s.op == Ops.CONST and abs(s.arg - 6.283185307179586) < 1e-5: return (x.src[1-i], 6.283185307179586)
      if s.op == Ops.MUL and len(s.src) == 2:
        vals = [ss.arg for ss in s.src if ss.op == Ops.CONST] + [ss.src[0].arg for ss in s.src if ss.op == Ops.CAST and ss.src[0].op == Ops.CONST]
        if len(vals) == 2 and abs(vals[0] * vals[1] - 6.283185307179586) < 1e-5: return (x.src[1-i], vals[0] * vals[1])
    return None

  def _trig_reduce(x, phase=0.0):
    match = _find_two_pi_mul(x)
    if match is not None:
      turns, two_pi = match
      if phase: turns = turns + _const(turns.dtype, phase)
      n = _floor(turns + _const(turns.dtype, 0.5))
      return UOp(Ops.SIN, turns.dtype, ((turns - n) * _const(turns.dtype, two_pi),))
    if phase: x = x + _const(x.dtype, phase * 6.283185307179586)
    n = _floor(x * _const(x.dtype, 0.15915494309189535) + _const(x.dtype, 0.5))
    return UOp(Ops.SIN, x.dtype, (x - n * _const(x.dtype, 6.283185307179586),))

  def _signext(a):
    val = a[0]
    for bits, mask, ext in [(8, 0xFF, 0xFFFFFF00), (16, 0xFFFF, 0xFFFF0000)]:
      if (val.op == Ops.AND and len(val.src) == 2 and val.src[1].op == Ops.CONST and val.src[1].arg == mask) or val.dtype.itemsize == bits // 8:
        v32 = val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val
        sb = (v32 >> _u32(bits - 1)) & _u32(1)
        return sb.ne(_u32(0)).where(v32 | _u32(ext), v32).cast(dtypes.int)
    return val.cast(dtypes.int64) if val.dtype in (dtypes.int, dtypes.int32) else val

  def _abs(a):
    if a[0].dtype not in (dtypes.float32, dtypes.float64, dtypes.half): return a[0]
    _, _, _, _, shift = _float_info(a[0])
    sign_mask = {10: 0x7FFF, 23: 0x7FFFFFFF, 52: 0x7FFFFFFFFFFFFFFF}[shift]
    bt, ft = {10: (dtypes.uint16, dtypes.half), 23: (dtypes.uint32, dtypes.float32), 52: (dtypes.uint64, dtypes.float64)}[shift]
    return (a[0].bitcast(bt) & _const(bt, sign_mask)).bitcast(ft)

  def _f_to_u(f, dt): return UOp(Ops.TRUNC, f.dtype, ((f < _const(f.dtype, 0.0)).where(_const(f.dtype, 0.0), f),)).cast(dt)

  def _cvt_quiet(a):
    bits, _, _, qb, _ = _float_info(a[0])
    bt, ft = (dtypes.uint64, dtypes.float64) if a[0].dtype == dtypes.float64 else (dtypes.uint16, dtypes.half) if a[0].dtype == dtypes.half else (dtypes.uint32, dtypes.float32)
    return (a[0].bitcast(bt) | qb).bitcast(ft)

  def _is_denorm(a):
    bits, exp_m, mant_m, _, _ = _float_info(a[0])
    return (bits & exp_m).eq(_const(bits.dtype, 0)) & (bits & mant_m).ne(_const(bits.dtype, 0))

  _EXP_BITS = {10: 0x1F, 23: 0xFF, 52: 0x7FF}
  def _get_exp(bits, shift): return ((bits >> _const(bits.dtype, shift)) & _const(bits.dtype, _EXP_BITS[shift])).cast(dtypes.int)

  def _exponent(a):
    bits, _, _, _, shift = _float_info(a[0])
    return _get_exp(bits, shift)

  def _div_would_be_denorm(a):
    bits_n, _, _, _, shift = _float_info(a[0])
    bits_d, _, _, _, _ = _float_info(a[1])
    min_exp = {10: -14, 23: -126, 52: -1022}[shift]
    return (_get_exp(bits_n, shift) - _get_exp(bits_d, shift)) < _const(dtypes.int, min_exp)

  def _sign(a):
    bits, _, _, _, shift = _float_info(a[0])
    sign_shift = {10: 15, 23: 31, 52: 63}[shift]
    return ((bits >> _const(bits.dtype, sign_shift)) & _const(bits.dtype, 1)).cast(dtypes.uint32)

  def _signext_from_bit(a):
    val, w = a[0], a[1]
    is_64bit = val.dtype in (dtypes.uint64, dtypes.int64)
    dt = dtypes.uint64 if is_64bit else dtypes.uint32
    mask_all = _const(dt, 0xFFFFFFFFFFFFFFFF if is_64bit else 0xFFFFFFFF)
    one = _const(dt, 1)
    val_u = val.cast(dt) if val.dtype != dt else val
    w_val = w.cast(dt) if w.dtype != dt else w
    sign_bit = (val_u >> (w_val - one)) & one
    ext_mask = ((one << w_val) - one) ^ mask_all
    return sign_bit.ne(_const(dt, 0)).where(val_u | ext_mask, val_u)

  def _ldexp(a):
    val, exp = a[0], a[1]
    if val.dtype == dtypes.uint32: val = val.bitcast(dtypes.float32)
    elif val.dtype == dtypes.uint64: val = val.bitcast(dtypes.float64)
    if exp.dtype in (dtypes.uint32, dtypes.uint64): exp = exp.cast(dtypes.int if exp.dtype == dtypes.uint32 else dtypes.int64)
    return val * UOp(Ops.EXP2, val.dtype, (exp.cast(val.dtype),))

  def _frexp_mant(a):
    val = a[0].bitcast(dtypes.float32) if a[0].dtype == dtypes.uint32 else a[0].bitcast(dtypes.float64) if a[0].dtype == dtypes.uint64 else a[0]
    if val.dtype == dtypes.float32: return ((val.bitcast(dtypes.uint32) & _u32(0x807FFFFF)) | _u32(0x3f000000)).bitcast(dtypes.float32)
    return ((val.bitcast(dtypes.uint64) & _const(dtypes.uint64, 0x800FFFFFFFFFFFFF)) | _const(dtypes.uint64, 0x3fe0000000000000)).bitcast(dtypes.float64)

  def _frexp_exp(a):
    val = a[0].bitcast(dtypes.float32) if a[0].dtype == dtypes.uint32 else a[0].bitcast(dtypes.float64) if a[0].dtype == dtypes.uint64 else a[0]
    if val.dtype == dtypes.float32: return ((val.bitcast(dtypes.uint32) >> _u32(23)) & _u32(0xFF)).cast(dtypes.int) - _const(dtypes.int, 126)
    return ((val.bitcast(dtypes.uint64) >> _const(dtypes.uint64, 52)) & _const(dtypes.uint64, 0x7FF)).cast(dtypes.int) - _const(dtypes.int, 1022)

  TWO_OVER_PI = 0x0145f306dc9c882a53f84eafa3ea69bb81b6c52b3278872083fca2c757bd778ac36e48dc74849ba5c00c925dd413a32439fc3bd63962534e7dd1046bea5d768909d338e04d68befc827323ac7306a673e93908bf177bf250763ff12fffbc0b301fde5e2316b414da3eda6cfd9e4f96136e9e8c7ecd3cbfd45aea4f758fd7cbe2f67a0e73ef14a525d4d7f6bf623f1aba10ac06608df8f6
  _PREOP = {s: float(((TWO_OVER_PI << s) >> (1201 - 53)) & 0x1fffffffffffff) for s in range(1149)}
  def _trig_preop(a):
    if a[0].op == Ops.CONST: return _const(dtypes.float64, _PREOP.get(int(a[0].arg), float(((TWO_OVER_PI << int(a[0].arg)) >> (1201 - 53)) & 0x1fffffffffffff)))
    result = _const(dtypes.float64, _PREOP[0])
    for s in range(1148, -1, -1): result = a[0].eq(_const(a[0].dtype, s)).where(_const(dtypes.float64, _PREOP[s]), result)
    return result

  def _ff1(a, bits):
    dt = dtypes.uint64 if bits == 64 else dtypes.uint32
    val = a[0].cast(dt) if a[0].dtype != dt else a[0]
    result = _const(dtypes.int, -1)
    for i in range(bits):
      cond = ((val >> _const(dt, i)) & _const(dt, 1)).ne(_const(dt, 0)) & result.eq(_const(dtypes.int, -1))
      result = cond.where(_const(dtypes.int, i), result)
    return result

  _FUNCS.update({
    'sqrt': lambda a, v: UOp(Ops.SQRT, a[0].dtype, (a[0],)), 'trunc': lambda a, v: UOp(Ops.TRUNC, a[0].dtype, (a[0],)),
    'log2': lambda a, v: UOp(Ops.LOG2, a[0].dtype, (a[0],)), 'sin': lambda a, v: _trig_reduce(a[0]),
    'cos': lambda a, v: _trig_reduce(a[0], 0.25), 'floor': lambda a, v: _floor(a[0]), 'fract': lambda a, v: a[0] - _floor(a[0]),
    'signext': lambda a, v: _signext(a), 'abs': lambda a, v: _abs(a),
    'isEven': lambda a, v: (UOp(Ops.TRUNC, a[0].dtype, (a[0],)).cast(dtypes.int) & _const(dtypes.int, 1)).eq(_const(dtypes.int, 0)),
    'max': lambda a, v: UOp(Ops.MAX, a[0].dtype, (a[0], a[1])),
    'min': lambda a, v: UOp(Ops.MAX, a[0].dtype, (a[0].neg(), a[1].neg())).neg(),
    'pow': lambda a, v: UOp(Ops.EXP2, dtypes.float32, (a[1].bitcast(dtypes.float32),)),
    'fma': lambda a, v: a[0] * a[1] + a[2],
    'i32_to_f32': lambda a, v: a[0].cast(dtypes.int).cast(dtypes.float32),
    'u32_to_f32': lambda a, v: a[0].cast(dtypes.uint32).cast(dtypes.float32),
    'f32_to_i32': lambda a, v: UOp(Ops.TRUNC, dtypes.float32, (a[0].bitcast(dtypes.float32),)).cast(dtypes.int),
    'f32_to_u32': lambda a, v: _f_to_u(a[0].bitcast(dtypes.float32), dtypes.uint32),
    'f64_to_i32': lambda a, v: UOp(Ops.TRUNC, dtypes.float64, (a[0].bitcast(dtypes.float64),)).cast(dtypes.int),
    'f64_to_u32': lambda a, v: _f_to_u(a[0].bitcast(dtypes.float64), dtypes.uint32),
    'f16_to_f32': lambda a, v: _f16_extract(a[0]).cast(dtypes.float32),
    'f32_to_f16': lambda a, v: a[0].cast(dtypes.half),
    'f32_to_f64': lambda a, v: a[0].bitcast(dtypes.float32).cast(dtypes.float64),
    'f64_to_f32': lambda a, v: a[0].bitcast(dtypes.float64).cast(dtypes.float32),
    'i32_to_f64': lambda a, v: a[0].cast(dtypes.int).cast(dtypes.float64),
    'u32_to_f64': lambda a, v: a[0].cast(dtypes.uint32).cast(dtypes.float64),
    'f16_to_i16': lambda a, v: UOp(Ops.TRUNC, dtypes.half, (_f16_extract(a[0]),)).cast(dtypes.int16),
    'f16_to_u16': lambda a, v: UOp(Ops.TRUNC, dtypes.half, (_f16_extract(a[0]),)).cast(dtypes.uint16),
    'i16_to_f16': lambda a, v: a[0].cast(dtypes.int16).cast(dtypes.half),
    'u16_to_f16': lambda a, v: a[0].cast(dtypes.uint16).cast(dtypes.half),
    'bf16_to_f32': lambda a, v: (((a[0].cast(dtypes.uint32) if a[0].dtype != dtypes.uint32 else a[0]) & _u32(0xFFFF)) << _u32(16)).bitcast(dtypes.float32),
    'isNAN': lambda a, v: _isnan(a[0]), 'isSignalNAN': lambda a, v: _check_nan(a[0], False),
    'isQuietNAN': lambda a, v: _check_nan(a[0], True), 'cvtToQuietNAN': lambda a, v: _cvt_quiet(a),
    'isDENORM': lambda a, v: _is_denorm(a), 'exponent': lambda a, v: _exponent(a),
    'divWouldBeDenorm': lambda a, v: _div_would_be_denorm(a), 'sign': lambda a, v: _sign(a),
    'signext_from_bit': lambda a, v: _signext_from_bit(a), 'ldexp': lambda a, v: _ldexp(a),
    'frexp_mant': lambda a, v: _frexp_mant(a), 'mantissa': lambda a, v: _frexp_mant(a),
    'frexp_exp': lambda a, v: _frexp_exp(a), 'trig_preop_result': lambda a, v: _trig_preop(a),
    's_ff1_i32_b32': lambda a, v: _ff1(a, 32), 's_ff1_i32_b64': lambda a, v: _ff1(a, 64),
  })
  for is_max, name in [(False, 'min'), (True, 'max')]:
    for dt, sfx in [(dtypes.float32, 'f32'), (dtypes.int, 'i32'), (dtypes.uint32, 'u32'), (dtypes.int16, 'i16'), (dtypes.uint16, 'u16')]:
      _FUNCS[f'v_{name}_{sfx}'] = lambda a, v, im=is_max, d=dt: _minmax_reduce(im, d, a)
      _FUNCS[f'v_{name}3_{sfx}'] = lambda a, v, im=is_max, d=dt: _minmax_reduce(im, d, a)

_register_funcs()

def parse_expr(expr: str, vars: dict, funcs: dict = None) -> UOp:
  if funcs is None: funcs = _FUNCS
  expr = expr.strip().rstrip(';')
  tokens = tokenize(expr)
  parser = Parser(tokens, vars, funcs, expr)
  return parser.parse()

# Helper for evaluating simple arithmetic expressions with variable substitution
import re as _re
def _try_eval(s: str) -> str:
  try: return str(int(eval(_re.sub(r'(\d+)U', r'\1', s))))
  except Exception: return s
