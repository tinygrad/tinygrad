# Clean tokenizer-based expression parser for AMD pcode
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
    # Two-char operators
    if i + 1 < n and s[i:i+2] in ('||', '&&', '>=', '<=', '==', '!=', '<>', '>>', '<<', '**', '+:', '-:'):
      tokens.append(Token('OP', s[i:i+2])); i += 2; continue
    # Single-char tokens
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
    if c == "'": tokens.append(Token('QUOTE', c)); i += 1; continue
    if c == ';': i += 1; continue
    # Number (including hex, float with suffix)
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
      # Suffix
      for sfx in ('ULL', 'LL', 'UL', 'U', 'L', 'F', 'f'):
        if s[i:i+len(sfx)] == sfx: i += len(sfx); break
      tokens.append(Token('NUM', s[start:i])); continue
    # Identifier
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

  # Precedence table: (ops, apply_fn) - lowest precedence first
  # apply_fn(left, right, op) -> result
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
      # Fold constant negation to produce negative literals
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
    # Parenthesized expression
    if self.try_eat('LPAREN'):
      e = self.expr_top()
      self.eat('RPAREN')
      return e

    # Brace concat: {hi, lo}
    if self.try_eat('LBRACE'):
      hi = self.expr_top()
      self.eat('COMMA')
      lo = self.expr_top()
      self.eat('RBRACE')
      return (hi.cast(dtypes.uint64) << _u64(32)) | lo.cast(dtypes.uint64)

    # Number with optional size prefix: 64'U(x) or 1'1U or plain 123
    if self.at('NUM'):
      num = self.eat('NUM').val
      if self.try_eat('QUOTE'):
        return self._sized_literal(int(num.rstrip('ULlf')))
      return self._parse_number(num)

    # Identifier: variable, function call, MEM, VGPR, or special constant
    if self.at('IDENT'):
      name = self.eat('IDENT').val

      # MEM[addr].type
      if name == 'MEM':
        self.eat('LBRACKET')
        addr = self.expr_top()
        self.eat('RBRACKET')
        self.eat('DOT')
        dt_name = self.eat('IDENT').val
        return self._handle_mem_load(addr, DTYPES.get(dt_name, dtypes.uint32))

      # VGPR[lane][reg]
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

      # Function call
      if self.try_eat('LPAREN'):
        args = self._parse_args()
        self.eat('RPAREN')
        return self._call_func(name, args)

      # Special constants
      if name == 'PI': return _const(dtypes.float32, 3.141592653589793)
      if name == 'INF': return _const(dtypes.float64, float('inf'))
      if name == 'NAN': return _const(dtypes.uint32, 0x7FC00000).bitcast(dtypes.float32)
      if name == 'UNDERFLOW_F32': return _const(dtypes.uint32, 1).bitcast(dtypes.float32)
      if name == 'OVERFLOW_F32': return _const(dtypes.uint32, 0x7F7FFFFF).bitcast(dtypes.float32)
      if name == 'UNDERFLOW_F64': return _const(dtypes.uint64, 1).bitcast(dtypes.float64)
      if name == 'OVERFLOW_F64': return _const(dtypes.uint64, 0x7FEFFFFFFFFFFFFF).bitcast(dtypes.float64)

      # Array element access: name{idx} or name{idx}.type
      if self.at('LBRACE'):
        self.eat('LBRACE')
        idx = self.eat('NUM').val
        self.eat('RBRACE')
        elem = self.vars.get(f'{name}{idx}', _u32(0))
        # Check for type suffix or bit slice
        if self.try_eat('DOT'):
          dt_name = self.eat('IDENT').val
          return _cast_to(elem, DTYPES.get(dt_name, dtypes.uint32))
        if self.at('LBRACKET'):
          return self._handle_bracket_with_name(elem, name + idx)
        return elem

      # Array element access with bracket: name[idx] - check if name{idx} exists
      if self.at('LBRACKET') and name not in self.vars:
        # Peek to see if this is a constant index and we have array elements
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
        # Not a simple array access, parse as expression and handle
        first = self.expr_top()
        return self._handle_bracket_rest(first, _u32(0), name)

      # Variable lookup
      if name in self.vars:
        v = self.vars[name]
        return v if isinstance(v, UOp) else _u32(0) if isinstance(v, dict) else _u32(0)
      return _u32(0)

    raise RuntimeError(f"unexpected token in primary: {self.peek()} in: {self.expr}")

  def _handle_dot(self, base, field: str) -> UOp:
    """Handle base.field - type cast, dict access, or special constants"""
    # Special float constants with type suffix
    if isinstance(base, str):  # This shouldn't happen normally
      return _u32(0)
    if not isinstance(base, UOp):
      if isinstance(base, dict): return base.get(field, _u32(0))
      return _u32(0)

    # Check for .u64[laneId] pattern - peek ahead
    if field == 'u64' and self.at('LBRACKET') and self.peek(1).type == 'IDENT' and self.peek(1).val == 'laneId':
      self.eat('LBRACKET')
      self.eat('IDENT')  # laneId
      self.eat('RBRACKET')
      result = (base >> _to_u32(self.vars['laneId'])) & _u32(1)
      # Check for another type suffix
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
    """Handle base[...] - slice, single bit, Verilog +:/-, or array index"""
    self.eat('LBRACKET')
    first = self.expr_top()
    return self._handle_bracket_rest(first, base, var_name)

  def _handle_bracket_with_name(self, base, var_name: str) -> UOp:
    """Handle base[...] with known variable name"""
    self.eat('LBRACKET')
    first = self.expr_top()
    return self._handle_bracket_rest(first, base, var_name)

  def _handle_bracket_rest(self, first: UOp, base: UOp, var_name: str | None = None) -> UOp:
    """Handle the rest of bracket parsing after first expr is parsed"""
    # Verilog +: or -: slice
    if self.at('OP') and self.peek().val in ('+:', '-:'):
      op = self.eat('OP').val
      width = self.expr_top()
      self.eat('RBRACKET')
      if width.op == Ops.CONST:
        w = int(width.arg)
        return (base >> _to_u32(first)) & _const(base.dtype, (1 << w) - 1)
      return base

    # Range slice [hi:lo] or [lo:hi] for bit reverse
    if self.try_eat('COLON'):
      second = self.expr_top()
      self.eat('RBRACKET')
      if first.op == Ops.CONST and second.op == Ops.CONST:
        a, b = int(first.arg), int(second.arg)
        if a < b:  # bit reverse
          from extra.assembly.amd.emu2_pcode import _bitreverse
          return _bitreverse(base, b - a + 1)
        from extra.assembly.amd.emu2_pcode import _extract_bits
        hi, lo = a, b
        # Check if we need to access array elements
        if lo >= base.dtype.itemsize * 8:
          vn = var_name or self._find_var_name(base)
          if vn and f'{vn}{lo // 32}' in self.vars:
            base = self.vars[f'{vn}{lo // 32}']
            lo, hi = lo % 32, (hi % 32) + (lo % 32)
        return _extract_bits(base, hi, lo)
      return base

    self.eat('RBRACKET')

    # Check for optional type suffix after bracket
    dt_suffix = None
    if self.try_eat('DOT'):
      dt_suffix = DTYPES.get(self.eat('IDENT').val, dtypes.uint32)

    # Get var_name if not provided
    if var_name is None:
      var_name = self._find_var_name(base)

    # Single bit [idx] or array element
    if first.op == Ops.CONST:
      idx = int(first.arg)
      # Check if this is array element access
      if var_name and f'{var_name}{idx}' in self.vars:
        v = self.vars[f'{var_name}{idx}']
        return _cast_to(v, dt_suffix) if dt_suffix else v
      # Single bit extraction - cast base to uint32/64 for consistent shift types
      dt = dtypes.uint64 if base.dtype in (dtypes.uint64, dtypes.int64) else dtypes.uint32
      base_cast = base.cast(dt) if base.dtype != dt else base
      result = ((base_cast >> _const(dt, idx)) & _const(dt, 1))
      return _cast_to(result, dt_suffix) if dt_suffix else result

    # Dynamic index - try array lookup first
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

    # Dynamic bit extraction - cast to uint32 for consistent shift types
    dt = dtypes.uint64 if base.dtype in (dtypes.uint64, dtypes.int64) else dtypes.uint32
    base_cast = base.cast(dt) if base.dtype != dt else base
    result = (base_cast >> first.cast(dt)) & _const(dt, 1)
    return _cast_to(result, dt_suffix) if dt_suffix else result

  def _handle_brace_index(self, base) -> UOp:
    """Handle base{idx} - array element access"""
    self.eat('LBRACE')
    idx = self.eat('NUM').val
    self.eat('RBRACE')
    var_name = self._find_var_name(base)
    if var_name:
      elem = self.vars.get(f'{var_name}{idx}', _u32(0))
      # Check for type suffix or bit slice
      if self.try_eat('DOT'):
        dt_name = self.eat('IDENT').val
        return _cast_to(elem, DTYPES.get(dt_name, dtypes.uint32))
      if self.at('LBRACKET'):
        return self._handle_bracket(elem)
      return elem
    return _u32(0)

  def _find_var_name(self, base: UOp) -> str | None:
    """Try to find the variable name for a UOp (for array access)"""
    if base.op == Ops.DEFINE_VAR and base.arg: return base.arg[0]
    # Search vars for matching UOp
    for name, v in self.vars.items():
      if isinstance(v, UOp) and v is base: return name
    return None

  def _sized_literal(self, bits: int) -> UOp:
    """Parse literal after N' - e.g., 64'U(x) or 1'1U or 64'h1234"""
    # Type cast: bits'T(expr) where T is U, I, F, B
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

    # Sized literal with radix: bits'hXXX or bits'bXXX or bits'dXXX
    # The tokenizer may produce 'h', 'b', 'd' alone or 'hFF', 'b101', 'd123' as single identifiers
    if self.at('IDENT'):
      ident = self.peek().val
      fmt = ident[0].lower()
      if fmt in ('h', 'b', 'd'):
        self.eat('IDENT')
        # Rest of the identifier is the number, or next token is the number
        if len(ident) > 1:
          num = ident[1:]
        elif self.at('NUM'):
          num = self.eat('NUM').val
        elif self.at('IDENT'):
          num = self.eat('IDENT').val
        else:
          raise RuntimeError(f"expected number after {bits}'{fmt} in: {self.expr}")
        if fmt == 'h': val = int(num, 16)
        elif fmt == 'b': val = int(num, 2)
        else: val = int(num)
        return _const(_BITS_DT.get(bits, dtypes.uint32), val)

    # Sized hex literal: bits'0xXXX
    if self.at('NUM') and self.peek().val.startswith('0x'):
      num = self.eat('NUM').val
      return _const(_BITS_DT.get(bits, dtypes.uint32), int(num, 16))

    # Sized literal: bits'value or bits'-value
    if self.at('NUM') or (self.at('OP') and self.peek().val == '-'):
      neg = self.try_eat_val('-') is not None
      num = self.eat('NUM').val
      # Strip suffix
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
    """Parse a standalone number like 123, 0x100, 1.5F, 0x100ULL"""
    suffix = ''
    # Handle hex - strip only non-hex suffixes (ULL, LL, UL, L, U - but NOT F since it's a hex digit)
    if num.startswith('0x') or num.startswith('0X'):
      for sfx in ('ULL', 'LL', 'UL', 'U', 'L'):
        if num.endswith(sfx): suffix, num = sfx, num[:-len(sfx)]; break
      return _const(dtypes.uint64, int(num, 16))
    # For non-hex, strip all suffixes including F
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
    """Parse comma-separated arguments"""
    if self.at('RPAREN'): return []
    args = [self.expr_top()]
    while self.try_eat('COMMA'):
      args.append(self.expr_top())
    return args

  def _call_func(self, name: str, args: list[UOp]) -> UOp:
    """Call a built-in function"""
    # Lambda call
    if name in self.vars and isinstance(self.vars[name], tuple) and self.vars[name][0] == 'lambda':
      _, params, body = self.vars[name]
      lv = {**self.vars, **{p: a for p, a in zip(params, args)}}
      if ';' in body or '\n' in body or 'return' in body.lower():
        from extra.assembly.amd.emu2_pcode import _parse_lambda_body
        return _parse_lambda_body(body, lv)
      return parse_expr(body, lv, self.funcs)
    # Built-in function
    if name in self.funcs:
      return self.funcs[name](args, self.vars)
    raise RuntimeError(f"unknown function: {name} in: {self.expr}")

  def _handle_mem_load(self, addr: UOp, dt) -> UOp:
    """Handle MEM[addr].type load"""
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
    from extra.assembly.amd.emu2_pcode import _isnan
    result = fn(l, r)
    if l.dtype in (dtypes.float32, dtypes.float64, dtypes.half):
      return result & _isnan(l).logical_not() & _isnan(r).logical_not()
    return result

def parse_expr(expr: str, vars: dict, funcs: dict) -> UOp:
  expr = expr.strip().rstrip(';')
  tokens = tokenize(expr)
  parser = Parser(tokens, vars, funcs, expr)
  return parser.parse()
