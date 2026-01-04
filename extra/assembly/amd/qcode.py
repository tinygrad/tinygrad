# Minimal AST parser for AMD GPU pseudocode
from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum, auto

class DType(Enum):
  F64 = auto(); F32 = auto(); F16 = auto(); BF16 = auto()
  U64 = auto(); U32 = auto(); U24 = auto(); U16 = auto(); U8 = auto()
  I64 = auto(); I32 = auto(); I24 = auto(); I16 = auto(); I8 = auto()
  B128 = auto(); B64 = auto(); B32 = auto(); B16 = auto(); B8 = auto()
  U1 = auto(); I1 = auto(); U3 = auto(); U4 = auto(); I4 = auto()
  U = U32; I = I32; F = F32  # aliases

DTYPES = {d.name.lower(): d for d in DType}

@dataclass(frozen=True)
class Const:
  value: int | float
  dtype: DType = DType.I32

@dataclass(frozen=True)
class Var:
  name: str

@dataclass(frozen=True)
class Typed:
  expr: Expr
  dtype: DType

@dataclass(frozen=True)
class Slice:
  expr: Expr
  high: Expr
  low: Expr

@dataclass(frozen=True)
class VSlice:  # Verilog-style [start +: width]
  expr: Expr
  start: Expr
  width: Expr

@dataclass(frozen=True)
class Index:
  expr: Expr
  idx: Expr

@dataclass(frozen=True)
class Cast:
  bits: int
  typ: str  # 'I', 'U', 'F', 'B'
  expr: Expr

@dataclass(frozen=True)
class Unary:
  op: str
  expr: Expr

@dataclass(frozen=True)
class Binary:
  op: str
  left: Expr
  right: Expr

@dataclass(frozen=True)
class Ternary:
  cond: Expr
  true_val: Expr
  false_val: Expr

@dataclass(frozen=True)
class Call:
  name: str
  args: tuple[Expr, ...]

@dataclass(frozen=True)
class Pack:
  exprs: tuple[Expr, ...]

Expr = Const | Var | Typed | Slice | VSlice | Index | Cast | Unary | Binary | Ternary | Call | Pack

@dataclass(frozen=True)
class Assign:
  lhs: Expr
  rhs: Expr

@dataclass(frozen=True)
class Compound:
  op: str
  lhs: Expr
  rhs: Expr

@dataclass(frozen=True)
class Declare:
  name: str
  dtype: str  # raw string from pseudocode

@dataclass(frozen=True)
class If:
  branches: tuple[tuple[Expr | None, tuple[Stmt, ...]], ...]

@dataclass(frozen=True)
class For:
  var: str
  start: Expr
  end: Expr
  body: tuple[Stmt, ...]

Stmt = Assign | Compound | Declare | If | For

def _match(s: str, start: int, open: str, close: str) -> int:
  depth = 1
  for i in range(start + 1, len(s)):
    if s[i] == open: depth += 1
    elif s[i] == close: depth -= 1
    if depth == 0: return i
  return -1

def _split(s: str) -> list[str]:
  args, depth, last = [], 0, 0
  for i, c in enumerate(s):
    if c in '([{': depth += 1
    elif c in ')]}': depth -= 1
    elif c == ',' and depth == 0: args.append(s[last:i].strip()); last = i + 1
  if s[last:].strip(): args.append(s[last:].strip())
  return args

def _findop(s: str, ops: tuple[str, ...]) -> int:
  depth = bracket = 0
  for i in range(len(s) - 1, -1, -1):
    c = s[i]
    if c == ')': depth += 1
    elif c == '(': depth -= 1
    elif c == ']': bracket += 1
    elif c == '[': bracket -= 1
    elif depth == 0 and bracket == 0:
      for op in sorted(ops, key=len, reverse=True):
        if s[i:i+len(op)] == op:
          if op in ('<', '>') and (i + 1 < len(s) and s[i+1] in '<>=' or i > 0 and s[i-1] in '<>='): continue
          if op == '*' and (i + 1 < len(s) and s[i+1] == '*' or i > 0 and s[i-1] == '*'): continue
          if op == '-' and (not s[:i].rstrip() or s[:i].rstrip()[-1] in '+-*/(<>=&|^,'): continue
          return i
  return -1

def parse_expr(s: str) -> Expr:
  s = s.strip()
  if not s: raise ValueError("Empty expression")
  if s.endswith('.') and not (len(s) > 1 and s[-2].isdigit()): s = s[:-1]
  if s.endswith(';'): s = s[:-1]
  s = s.strip()

  if s.startswith('(') and (end := _match(s, 0, '(', ')')) == len(s) - 1: return parse_expr(s[1:end])
  if s.startswith('{') and s.endswith('}'): return Pack(tuple(parse_expr(a) for a in _split(s[1:-1])))
  if m := re.match(r"^(\d+)'([IUFB])\(", s):
    if (end := _match(s, m.end() - 1, '(', ')')) == len(s) - 1: return Cast(int(m[1]), m[2], parse_expr(s[m.end():end]))
  if m := re.match(r"^(\d+)'(-?\d+)([IUFB])?$", s):
    return Const(int(m[2]), DTYPES.get(f"{m[3] or 'i'}{m[1]}".lower(), DType.I32))
  if m := re.match(r"^(\d+)'(-?[\d.]+)$", s): return Const(float(m[2]), DTYPES.get(f"f{m[1]}", DType.F32))
  if m := re.match(r"^(\d+)'(0x[0-9a-fA-F]+)$", s): return Const(int(m[2], 16), DTYPES.get(f"u{m[1]}", DType.U32))
  if m := re.match(r"^([A-Za-z_]\w*)\(", s):
    if (end := _match(s, m.end() - 1, '(', ')')) == len(s) - 1:
      args = _split(s[m.end():end])
      return Call(m[1], tuple(parse_expr(a) for a in args) if args != [''] else ())
  if s.startswith('MEM[') and (end := _match(s, 3, '[', ']')) != -1:
    rest, base = s[end+1:], Call('MEM', (parse_expr(s[4:end]),))
    return Typed(base, DTYPES[rest[1:]]) if rest.startswith('.') and rest[1:] in DTYPES else base if not rest else base
  if (q := _findop(s, ('?',))) > 0:
    depth = bracket = 0
    for i in range(q + 1, len(s)):
      if s[i] == '(': depth += 1
      elif s[i] == ')': depth -= 1
      elif s[i] == '[': bracket += 1
      elif s[i] == ']': bracket -= 1
      elif s[i] == ':' and depth == 0 and bracket == 0:
        return Ternary(parse_expr(s[:q]), parse_expr(s[q+1:i]), parse_expr(s[i+1:]))
  for ops in [('||',), ('&&',), ('|',), ('^',), ('&',), ('==', '!=', '<>'), ('<=', '>=', '<', '>'), ('<<', '>>'), ('+', '-'), ('*', '/', '%'), ('**',)]:
    if (pos := _findop(s, ops)) > 0:
      op = next(o for o in sorted(ops, key=len, reverse=True) if s[pos:pos+len(o)] == o)
      left, right = s[:pos].strip(), s[pos+len(op):].strip()
      if left and right: return Binary(op, parse_expr(left), parse_expr(right))
  if s[0] in '-~!' and len(s) > 1 and (s[0] != '!' or s[1] != '='): return Unary(s[0], parse_expr(s[1:]))
  if '[' in s and s.endswith(']'):
    depth = 0
    for i in range(len(s) - 1, -1, -1):
      if s[i] == ']': depth += 1
      elif s[i] == '[': depth -= 1
      if depth == 0 and s[i] == '[': break
    base, inner = s[:i], s[i+1:-1]
    if '+:' in inner:
      st, w = inner.split('+:', 1)
      return VSlice(parse_expr(base), parse_expr(st), parse_expr(w))
    if ':' in inner and '?' not in inner:
      depth = 0
      for ci, cc in enumerate(inner):
        if cc in '([{': depth += 1
        elif cc in ')]}': depth -= 1
        elif cc == ':' and depth == 0: return Slice(parse_expr(base), parse_expr(inner[:ci]), parse_expr(inner[ci+1:]))
    return Index(parse_expr(base), parse_expr(inner))
  if '.' in s:
    for i in range(len(s) - 1, 0, -1):
      if s[i] == '.' and s[i+1:] in DTYPES: return Typed(parse_expr(s[:i]), DTYPES[s[i+1:]])
  if s in ('INF', '+INF'): return Const(float('inf'), DType.F64)
  if s == '-INF': return Const(float('-inf'), DType.F64)
  if s == 'PI': return Const(3.141592653589793, DType.F64)
  if s.startswith('eval '): return Var(s)
  if ':' in s and '?' not in s and '[' not in s:
    parts = s.split(':')
    if len(parts) == 2 and all(re.match(r'^[A-Za-z_]\w*$', p.strip()) for p in parts):
      return Binary(':', parse_expr(parts[0]), parse_expr(parts[1]))
  if re.match(r'^[A-Za-z_][\w.]*$', s): return Var(s)
  try:
    if s.startswith('0x') or s.startswith('0X'): return Const(int(re.sub(r'[UuLl]+$', '', s), 16), DType.U32)
    c = re.sub(r'[FfLlUu]+$', '', s)
    if '.' in c or 'e' in c.lower(): return Const(float(c), DType.F32)
    return Const(int(re.sub(r'[UuLl]+$', '', s)), DType.I32)
  except ValueError: pass
  raise ValueError(f"Cannot parse expression: {s}")

def parse_stmt(line: str) -> Stmt | None:
  line = line.split('//')[0].strip().rstrip(';')
  if not line: return None
  if line.startswith('eval '): return Assign(Var('_eval'), Var(line))
  if line.startswith('declare ') and ':' in line:
    name, typ = line[8:].split(':', 1)
    return Declare(name.strip(), typ.strip())
  for op in ('+=', '-=', '|=', '&=', '^=', '<<=', '>>='):
    if op in line:
      lhs, rhs = line.split(op, 1)
      return Compound(op[:-1], parse_expr(lhs), parse_expr(rhs))
  if '=' in line and not any(line.startswith(k) for k in ('if ', 'elsif ', 'for ')):
    eq = line.index('=')
    if eq > 0 and line[eq-1] not in '!<>=' and eq < len(line) - 1 and line[eq+1] != '=':
      return Assign(parse_expr(line[:eq]), parse_expr(line[eq+1:]))
  return None

def parse(pseudocode: str) -> tuple[Stmt, ...]:
  lines = [l.split('//')[0].strip().rstrip(';') for l in pseudocode.strip().split('\n') if l.split('//')[0].strip()]
  stmts: list[Stmt] = []
  i = 0
  while i < len(lines):
    line = lines[i]
    if line.startswith('for ') and ' do' in line and (m := re.match(r'for\s+(\w+)\s+in\s+(.+?)\s*:\s*(.+?)\s+do', line)):
      i, body, depth = i + 1, [], 1
      while i < len(lines) and depth > 0:
        if lines[i].startswith('for ') and ' do' in lines[i]: depth += 1
        elif lines[i] == 'endfor': depth -= 1
        if depth > 0: body.append(lines[i])
        i += 1
      stmts.append(For(m[1], parse_expr(m[2]), parse_expr(m[3]), parse('\n'.join(body))))
      continue
    if line.startswith('if ') and ' then' in line:
      branches: list[tuple[Expr | None, tuple[Stmt, ...]]] = []
      cond, body, i = line[3:line.index(' then')], [], i + 1
      while i < len(lines) and lines[i] != 'endif':
        if lines[i].startswith('elsif ') and ' then' in lines[i]:
          branches.append((parse_expr(cond), parse('\n'.join(body))))
          cond, body = lines[i][6:lines[i].index(' then')], []
        elif lines[i] == 'else':
          branches.append((parse_expr(cond), parse('\n'.join(body))))
          cond, body = None, []  # type: ignore
        else: body.append(lines[i])
        i += 1
      branches.append((parse_expr(cond) if cond else None, parse('\n'.join(body))))
      stmts.append(If(tuple(branches)))
      i += 1
      continue
    if stmt := parse_stmt(line): stmts.append(stmt)
    i += 1
  return tuple(stmts)

def _pretty(node, indent=0) -> str:
  pad = "  " * indent
  match node:
    case Const(v, d): return f"{pad}Const({v}, {d.name})"
    case Var(n): return f"{pad}Var({n})"
    case Typed(e, d): return f"{pad}Typed .{d.name}\n{_pretty(e, indent+1)}"
    case Slice(e, h, l): return f"{pad}Slice\n{_pretty(e, indent+1)}\n{pad}  [{_pretty(h).strip()}:{_pretty(l).strip()}]"
    case VSlice(e, s, w): return f"{pad}VSlice\n{_pretty(e, indent+1)}\n{pad}  [{_pretty(s).strip()} +: {_pretty(w).strip()}]"
    case Index(e, i): return f"{pad}Index\n{_pretty(e, indent+1)}\n{pad}  [{_pretty(i).strip()}]"
    case Cast(b, t, e): return f"{pad}Cast {b}'{t}\n{_pretty(e, indent+1)}"
    case Unary(o, e): return f"{pad}Unary {o}\n{_pretty(e, indent+1)}"
    case Binary(o, l, r): return f"{pad}Binary {o}\n{_pretty(l, indent+1)}\n{_pretty(r, indent+1)}"
    case Ternary(c, t, f): return f"{pad}Ternary\n{_pretty(c, indent+1)}\n{pad}  ?\n{_pretty(t, indent+1)}\n{pad}  :\n{_pretty(f, indent+1)}"
    case Call(n, a): return f"{pad}Call {n}\n" + "\n".join(_pretty(x, indent+1) for x in a) if a else f"{pad}Call {n}()"
    case Pack(e): return f"{pad}Pack\n" + "\n".join(_pretty(x, indent+1) for x in e)
    case Assign(l, r): return f"{pad}Assign\n{_pretty(l, indent+1)}\n{pad}  =\n{_pretty(r, indent+1)}"
    case Compound(o, l, r): return f"{pad}Compound {o}=\n{_pretty(l, indent+1)}\n{_pretty(r, indent+1)}"
    case Declare(n, d): return f"{pad}Declare {n}: {d}"
    case If(br): return f"{pad}If\n" + "\n".join(f"{pad}  {'else' if c is None else 'elif' if i else 'if'}{'' if c is None else chr(10)+_pretty(c,indent+2)}\n" + "\n".join(_pretty(s,indent+2) for s in b) for i,(c,b) in enumerate(br))
    case For(v, s, e, b): return f"{pad}For {v} in {_pretty(s).strip()}:{_pretty(e).strip()}\n" + "\n".join(_pretty(x, indent+1) for x in b)
    case tuple(): return "\n".join(_pretty(x, indent) for x in node)
    case _: return f"{pad}{node}"

if __name__ == "__main__":
  import os
  from extra.assembly.amd.autogen.rdna3.str_pcode import PSEUDOCODE_STRINGS
  DEBUG = os.getenv("DEBUG", "0") == "1"
  success, fail, errors = 0, 0, {}
  for cls, ops in PSEUDOCODE_STRINGS.items():
    for op, pcode in ops.items():
      try:
        ast = parse(pcode)
        success += 1
        if DEBUG: print(f"{'='*60}\n{op.name}\n{'='*60}\n{pcode}\n{'-'*60}\n{_pretty(ast)}\n")
      except Exception as e: fail += 1; err = str(e)[:60]; errors[err] = errors.get(err, 0) + 1
  print(f"Parsed: {success}/{success+fail} ({100*success/(success+fail):.1f}%)")
  for err, cnt in sorted(errors.items(), key=lambda x: -x[1])[:10]: print(f"  {cnt}: {err}")
