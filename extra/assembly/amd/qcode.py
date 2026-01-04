# Minimal AST parser for AMD GPU pseudocode
from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum, auto

class QDType(Enum):
  F64=auto(); F32=auto(); F16=auto(); BF16=auto(); U64=auto(); U32=auto(); U24=auto(); U16=auto(); U8=auto()
  I64=auto(); I32=auto(); I24=auto(); I16=auto(); I8=auto(); B128=auto(); B64=auto(); B32=auto(); B16=auto()
  B8=auto(); U1=auto(); I1=auto(); U3=auto(); U4=auto(); I4=auto(); U=U32; I=I32; F=F32
DTYPES = {d.name.lower(): d for d in QDType}

@dataclass(frozen=True)
class Const: value: int|float; dtype: QDType = QDType.I32
@dataclass(frozen=True)
class Var: name: str
@dataclass(frozen=True)
class Typed: expr: Expr; dtype: QDType
@dataclass(frozen=True)
class Slice: expr: Expr; hi: Expr; lo: Expr
@dataclass(frozen=True)
class Index: expr: Expr; idx: Expr
@dataclass(frozen=True)
class Cast: bits: int; typ: str; expr: Expr
@dataclass(frozen=True)
class Unary: op: str; expr: Expr
@dataclass(frozen=True)
class Binary: op: str; left: Expr; right: Expr
@dataclass(frozen=True)
class Ternary: cond: Expr; t: Expr; f: Expr
@dataclass(frozen=True)
class Call: name: str; args: tuple[Expr, ...]
@dataclass(frozen=True)
class Pack: exprs: tuple[Expr, ...]
Expr = Const|Var|Typed|Slice|Index|Cast|Unary|Binary|Ternary|Call|Pack

@dataclass(frozen=True)
class Assign: lhs: Expr; rhs: Expr

@dataclass(frozen=True)
class Declare: name: str; dtype: str
@dataclass(frozen=True)
class If: branches: tuple[tuple[Expr|None, tuple[Stmt, ...]], ...]
@dataclass(frozen=True)
class For: var: str; start: Expr; end: Expr; body: tuple[Stmt, ...]
Stmt = Assign|Declare|If|For

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

def expr(s: str) -> Expr:
  s = s.strip().rstrip(';')
  if s.endswith('.') and not (len(s) > 1 and s[-2].isdigit()): s = s[:-1]
  s = s.strip()
  if not s: raise ValueError("Empty expression")
  if s == '+INF': s = 'INF'
  if s[0] == '(' and (e := _match(s, 0, '(', ')')) == len(s)-1: return expr(s[1:e])
  if s[0] == '{' and s[-1] == '}': return Pack(tuple(expr(a) for a in _split(s[1:-1])))
  if m := re.match(r"^(\d+)'([IUFB])\(", s):
    if (e := _match(s, m.end()-1, '(', ')')) == len(s)-1: return Cast(int(m[1]), m[2], expr(s[m.end():e]))
  if m := re.match(r"^(\d+)'(-?\d+)([IUFB])?$", s): return Const(int(m[2]), DTYPES.get(f"{m[3]or'i'}{m[1]}".lower(), QDType.I32))
  if m := re.match(r"^(\d+)'(-?[\d.]+)$", s): return Const(float(m[2]), DTYPES.get(f"f{m[1]}", QDType.F32))
  if m := re.match(r"^(\d+)'(0x[0-9a-fA-F]+)$", s): return Const(int(m[2], 16), DTYPES.get(f"u{m[1]}", QDType.U32))
  if m := re.match(r"^([A-Za-z_]\w*)\(", s):
    if (e := _match(s, m.end()-1, '(', ')')) == len(s)-1:
      a = _split(s[m.end():e]); return Call(m[1], tuple(expr(x) for x in a) if a != [''] else ())
  if s[:4] == 'MEM[' and (e := _match(s, 3, '[', ']')) != -1:
    r, b = s[e+1:], Call('MEM', (expr(s[4:e]),))
    return Typed(b, DTYPES[r[1:]]) if r[:1] == '.' and r[1:] in DTYPES else b
  if (q := _fop(s, ('?',))) > 0:
    d = b = 0
    for i in range(q+1, len(s)):
      if s[i] == '(': d += 1
      elif s[i] == ')': d -= 1
      elif s[i] == '[': b += 1
      elif s[i] == ']': b -= 1
      elif s[i] == ':' and d == 0 and b == 0: return Ternary(expr(s[:q]), expr(s[q+1:i]), expr(s[i+1:]))
  for ops in [('||',),('&&',),('|',),('^',),('&',),('==','!=','<>'),('<=','>=','<','>'),('<<','>>'),
              ('+','-'),('*','/','%'),('**',)]:
    if (p := _fop(s, ops)) > 0:
      op = next(o for o in sorted(ops, key=len, reverse=True) if s[p:p+len(o)] == o)
      l, r = s[:p].strip(), s[p+len(op):].strip()
      if l and r: return Binary(op, expr(l), expr(r))
  if s[0] in '-~!' and len(s) > 1 and (s[0] != '!' or s[1] != '='): return Unary(s[0], expr(s[1:]))
  if '[' in s and s[-1] == ']':
    d = 0
    for i in range(len(s)-1, -1, -1):
      if s[i] == ']': d += 1
      elif s[i] == '[': d -= 1
      if d == 0 and s[i] == '[': break
    b, n = s[:i], s[i+1:-1]
    if '+:' in n:  # Verilog [start +: width] -> [start+width-1 : start]
      st, w = expr(n.split('+:', 1)[0]), expr(n.split('+:', 1)[1])
      return Slice(expr(b), Binary('-', Binary('+', st, w), Const(1)), st)
    if ':' in n and '?' not in n:
      d = 0
      for j, c in enumerate(n):
        if c in '([{': d += 1
        elif c in ')]}': d -= 1
        elif c == ':' and d == 0: return Slice(expr(b), expr(n[:j]), expr(n[j+1:]))
    return Index(expr(b), expr(n))
  if '.' in s:
    for i in range(len(s)-1, 0, -1):
      if s[i] == '.' and s[i+1:] in DTYPES: return Typed(expr(s[:i]), DTYPES[s[i+1:]])
  if s[:5] == 'eval ': return Var(s)
  if ':' in s and '?' not in s and '[' not in s:
    p = s.split(':')
    if len(p) == 2 and all(re.match(r'^[A-Za-z_]\w*$', x.strip()) for x in p): return Binary(':', expr(p[0]), expr(p[1]))
  if re.match(r'^[A-Za-z_][\w.]*$', s): return Var(s)
  try:
    if s[:2].lower() == '0x': return Const(int(re.sub(r'[UuLl]+$', '', s), 16), QDType.U32)
    c = re.sub(r'[FfLlUu]+$', '', s)
    return Const(float(c), QDType.F32) if '.' in c or 'e' in c.lower() else Const(int(re.sub(r'[UuLl]+$', '', s)), QDType.I32)
  except ValueError: pass
  raise ValueError(f"Cannot parse expression: {s}")

def stmt(line: str) -> Stmt|None:
  line = line.split('//')[0].strip().rstrip(';')
  if not line: return None
  if line[:5] == 'eval ': return Assign(Var('_eval'), Var(line))
  if line[:8] == 'declare ' and ':' in line: n, t = line[8:].split(':', 1); return Declare(n.strip(), t.strip())
  for op in ('+=', '-=', '|=', '&=', '^=', '<<=', '>>='):
    if op in line:
      l, r = line.split(op, 1)
      lhs = expr(l)
      return Assign(lhs, Binary(op[:-1], lhs, expr(r)))
  if '=' in line and not any(line[:k] == p for k, p in [(3,'if '),(6,'elsif '),(4,'for ')]):
    eq = line.index('=')
    if eq > 0 and line[eq-1] not in '!<>=' and eq < len(line)-1 and line[eq+1] != '=':
      return Assign(expr(line[:eq]), expr(line[eq+1:]))
  return None

def parse(code: str) -> tuple[Stmt, ...]:
  lines = [l.split('//')[0].strip().rstrip(';') for l in code.strip().split('\n') if l.split('//')[0].strip()]
  stmts, i = [], 0
  while i < len(lines):
    ln = lines[i]
    if ln[:4] == 'for ' and ' do' in ln and (m := re.match(r'for\s+(\w+)\s+in\s+(.+?)\s*:\s*(.+?)\s+do', ln)):
      i, body, d = i+1, [], 1
      while i < len(lines) and d > 0:
        if lines[i][:4] == 'for ' and ' do' in lines[i]: d += 1
        elif lines[i] == 'endfor': d -= 1
        if d > 0: body.append(lines[i])
        i += 1
      stmts.append(For(m[1], expr(m[2]), expr(m[3]), parse('\n'.join(body)))); continue
    if ln[:3] == 'if ' and ' then' in ln:
      br, cond, body, i = [], ln[3:ln.index(' then')], [], i+1
      while i < len(lines) and lines[i] != 'endif':
        if lines[i][:6] == 'elsif ' and ' then' in lines[i]:
          br.append((expr(cond), parse('\n'.join(body)))); cond, body = lines[i][6:lines[i].index(' then')], []
        elif lines[i] == 'else': br.append((expr(cond), parse('\n'.join(body)))); cond, body = None, []
        else: body.append(lines[i])
        i += 1
      br.append((expr(cond) if cond else None, parse('\n'.join(body)))); stmts.append(If(tuple(br))); i += 1; continue
    if s := stmt(ln): stmts.append(s)
    i += 1
  return tuple(stmts)

if __name__ == "__main__":
  import os
  from extra.assembly.amd.autogen.rdna3.str_pcode import PSEUDOCODE_STRINGS
  DEBUG = os.getenv("DEBUG", "0") == "1"
  ok, fail, errs = 0, 0, {}
  for cls, ops in PSEUDOCODE_STRINGS.items():
    for op, pc in ops.items():
      try: ast = parse(pc); ok += 1
      except Exception as e: fail += 1; errs[str(e)[:60]] = errs.get(str(e)[:60], 0) + 1
      else:
        if DEBUG:
          def pr(n, d=0):
            p = "  "*d
            match n:
              case Const(v, t): return f"{v}" if t == QDType.I32 else f"{v}:{t.name}"
              case Var(x): return x
              case Typed(e, t): return f"{pr(e)}.{t.name}"
              case Slice(e,h,l): return f"{pr(e)}[{pr(h)}:{pr(l)}]"
              case Index(e, i): return f"{pr(e)}[{pr(i)}]"
              case Cast(b, t, e): return f"{b}'{t}({pr(e)})"
              case Unary(o, e): return f"{o}{pr(e)}"
              case Binary(o, l, r): return f"({pr(l)} {o} {pr(r)})"
              case Ternary(c, t, f): return f"({pr(c)} ? {pr(t)} : {pr(f)})"
              case Call(n, a): return f"{n}({', '.join(pr(x) for x in a)})"
              case Pack(e): return f"{{{', '.join(pr(x) for x in e)}}}"
              case Assign(l, r): return f"{p}{pr(l)} = {pr(r)}"
              case Declare(n, t): return f"{p}declare {n}: {t}"
              case If(br): return f"{p}if " + " elif ".join(f"({pr(c) if c else 'else'}) {{\n" + "\n".join(pr(s,d+1) for s in b) + f"\n{p}}}" for c,b in br)
              case For(v,s,e,b): return f"{p}for {v} in {pr(s)}:{pr(e)} {{\n" + "\n".join(pr(x,d+1) for x in b) + f"\n{p}}}"
              case tuple(): return "\n".join(pr(x, d) for x in n)
              case _: return f"{p}{n}"
          print(f"{'='*60}\n{op.name}\n{'='*60}\n{pc}\n{'-'*60}\n{pr(ast)}\n")
  print(f"Parsed: {ok}/{ok+fail} ({100*ok/(ok+fail):.1f}%)")
  for e, c in sorted(errs.items(), key=lambda x: -x[1])[:10]: print(f"  {c}: {e}")
