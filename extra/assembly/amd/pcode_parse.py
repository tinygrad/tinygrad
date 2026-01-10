# Minimal parser for AMD GPU pseudocode -> UOps
from __future__ import annotations
import re
from dataclasses import dataclass
from tinygrad.dtype import dtypes, DType, INVERSE_DTYPES_DICT
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp
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

# Statement types (control flow, not expressions)
# Assign is UOp(Ops.ASSIGN, dtypes.void, (lhs, rhs))
# Declare is UOp(Ops.DEFINE_VAR, dtype, arg=name)

def Assign(lhs: UOp, rhs: UOp) -> UOp: return UOp(Ops.ASSIGN, dtypes.void, (lhs, rhs))
def Declare(name: str, dtype: DType) -> UOp: return UOp(Ops.DEFINE_VAR, dtype, arg=name)

# Control flow statements (can be late substituted)
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
Stmt = UOp|If|For|Lambda|Break|Return

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
    return UOp(Ops.CAT, dtypes.void, parts)
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
  # Function call -> CUSTOM (all functions, transformed in pcode_transform.py)
  if m := re.match(r"^([A-Za-z_]\w*)\(", s):
    if (e := _match(s, m.end()-1, '(', ')')) == len(s)-1:
      a = _split(s[m.end():e])
      srcs = tuple(expr(x) for x in a) if a != [''] else ()
      return UOp(Ops.CUSTOM, dtypes.void, srcs, arg=m[1])
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
        return UOp(Ops.WHERE, dtypes.void, (gate, lhs, rhs))
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
        output_dtype = dtypes.bool if uop_op in {Ops.CMPNE, Ops.CMPEQ, Ops.CMPLE, Ops.CMPLT} else dtypes.void
        return UOp(uop_op, output_dtype, (lhs, rhs), tag=tag)
  # Unary ops: - (negate), ~ (bitwise NOT), ! (logical NOT)
  if s[0] in '-~!' and len(s) > 1 and (s[0] != '!' or s[1] != '='):
    src = expr(s[1:])
    # ! is logical NOT (compare to 0), returns bool; - and ~ use void (dtype resolved later)
    out_dtype = dtypes.bool if s[0] == '!' else dtypes.void
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
      # hi = start + width - 1
      hi = UOp(Ops.SUB, dtypes.void, (UOp(Ops.ADD, dtypes.void, (st, w)), UOp(Ops.CONST, dtypes.int32, arg=1)))
      # NOTE: CUSTOMI is used for bit slicing; SHRINK would be for tensor operations
      return UOp(Ops.CUSTOMI, dtypes.void, (expr(b), hi, st))
    if ':' in n and '?' not in n:
      d = 0
      for j, c in enumerate(n):
        if c in '([{': d += 1
        elif c in ')]}': d -= 1
        elif c == ':' and d == 0:
          hi_expr, lo_expr = expr(n[:j]), expr(n[j+1:])
          return UOp(Ops.CUSTOMI, dtypes.void, (expr(b), hi_expr, lo_expr))
    idx = expr(n)
    base = expr(b)
    return UOp(Ops.CUSTOMI, dtypes.void, (base, idx, idx))
  # Bitcast: expr.type
  if '.' in s:
    for i in range(len(s)-1, 0, -1):
      if s[i] == '.' and (dt := _get_dtype(s[i+1:])):
        assert dt != dtypes.void, f"BITCAST target type should not be void: {s}"
        return UOp(Ops.BITCAST, dt, (expr(s[:i]),))
  # Variable
  if s[:5] == 'eval ': return UOp(Ops.DEFINE_VAR, dtypes.void, arg=(s, None, None))
  if re.match(r'^[A-Za-z_][\w.]*$', s):
    return UOp(Ops.DEFINE_VAR, dtypes.void, arg=(s, None, None))
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
      return Declare(n.strip(), final_dt)
    return None  # unsupported declare type
  for op, uop in [('+=', Ops.ADD), ('-=', Ops.SUB), ('|=', Ops.OR), ('&=', Ops.AND), ('^=', Ops.XOR), ('<<=', Ops.SHL), ('>>=', Ops.SHR)]:
    if op in line:
      l, r = line.split(op, 1)
      lhs, rhs = expr(l), expr(r)
      return Assign(lhs, UOp(uop, dtypes.void, (lhs, rhs)))
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
        if isinstance(rhs_parsed, UOp) and rhs_parsed.op == Ops.ASSIGN:
          lhs = expr(line[:eq])
          return Assign(lhs, rhs_parsed)
      lhs, rhs_expr = expr(line[:eq]), expr(rhs)
      return Assign(lhs, rhs_expr)
  # Bare function call (e.g., nop())
  if re.match(r'\w+\([^)]*\)$', line):
    return expr(line)
  raise ValueError(f"Cannot parse statement: {line}")

def parse(code: str, _toplevel: bool = True) -> tuple[Stmt, ...]:
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
