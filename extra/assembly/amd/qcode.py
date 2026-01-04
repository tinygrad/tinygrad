# Minimal parser for AMD GPU pseudocode -> UOps
from __future__ import annotations
import re
from dataclasses import dataclass
from tinygrad.dtype import dtypes, DType
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp
from tinygrad.helpers import DEBUG

# DType lookup table for AMD pseudocode type suffixes
from tinygrad.dtype import INVERSE_DTYPES_DICT
_QDTYPES: dict[str, DType] = {
  'f64': dtypes.float64, 'f32': dtypes.float32, 'f16': dtypes.float16, 'bf16': dtypes.bfloat16,
  'u64': dtypes.uint64, 'u32': dtypes.uint32, 'u16': dtypes.uint16, 'u8': dtypes.uint8,
  'i64': dtypes.int64, 'i32': dtypes.int32, 'i16': dtypes.int16, 'i8': dtypes.int8,
  'b1201': DType.new(6, 1201, "b1201", None), 'b128': DType.new(6, 128, "b128", None),
  'b65': DType.new(6, 65, "b65", None), 'b64': dtypes.uint64, 'b32': dtypes.uint32, 'b16': dtypes.uint16, 'b8': dtypes.uint8,
  'u65': DType.new(6, 65, "u65", None), 'u24': DType.new(6, 24, "u24", None),
  'u6': DType.new(6, 6, "u6", None), 'u4': DType.new(6, 4, "u4", None),
  'u3': DType.new(6, 3, "u3", None), 'u1': DType.new(6, 1, "u1", None),
  'i65': DType.new(5, 65, "i65", None), 'i24': DType.new(5, 24, "i24", None), 'i1': DType.new(5, 1, "i1", None),
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
_UNOPS: dict[str, Ops] = {'-': Ops.NEG, '~': Ops.XOR, '!': Ops.CMPEQ}

# Statement types (control flow, not expressions)
@dataclass(frozen=True)
class Assign: lhs: UOp; rhs: UOp
@dataclass(frozen=True)
class Declare: name: str; dtype: DType
@dataclass(frozen=True)
class If: branches: tuple[tuple[UOp|None, tuple[Stmt, ...]], ...]
@dataclass(frozen=True)
class For: var: str; start: UOp; end: UOp; body: tuple[Stmt, ...]
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

def _get_dtype(name: str) -> DType | None: return _QDTYPES.get(name.lower())

def expr(s: str) -> UOp:
  s = s.strip().rstrip(';')
  if s.endswith('.') and not (len(s) > 1 and s[-2].isdigit()): s = s[:-1]
  s = s.strip()
  if not s: raise ValueError("Empty expression")
  if s == '+INF': s = 'INF'
  # Parentheses
  if s[0] == '(' and (e := _match(s, 0, '(', ')')) == len(s)-1: return expr(s[1:e])
  # Pack -> CAT
  if s[0] == '{' and s[-1] == '}': return UOp(Ops.CAT, dtypes.void, tuple(expr(a) for a in _split(s[1:-1])))
  # Typed cast: 32'U(expr)
  if m := re.match(r"^(\d+)'([IUFB])\(", s):
    if (e := _match(s, m.end()-1, '(', ')')) == len(s)-1: return UOp(Ops.CAST, _QDTYPES[f"{m[2].lower()}{m[1]}"], (expr(s[m.end():e]),))
  # Typed constant: 32'-5I
  if m := re.match(r"^(\d+)'(-?\d+)([IUFB])?$", s):
    return UOp(Ops.CONST, _QDTYPES[f"{(m[3] or 'I').lower()}{m[1]}"], arg=int(m[2]))
  if m := re.match(r"^(\d+)'(-?[\d.]+)$", s):
    return UOp(Ops.CONST, _QDTYPES[f"f{m[1]}"], arg=float(m[2]))
  if m := re.match(r"^(\d+)'(0x[0-9a-fA-F]+)$", s):
    return UOp(Ops.CONST, _QDTYPES[f"u{m[1]}"], arg=int(m[2], 16))
  # Function call -> CUSTOM
  if m := re.match(r"^([A-Za-z_]\w*)\(", s):
    if (e := _match(s, m.end()-1, '(', ')')) == len(s)-1:
      a = _split(s[m.end():e])
      return UOp(Ops.CUSTOM, dtypes.void, tuple(expr(x) for x in a) if a != [''] else (), arg=m[1])
  # MEM[addr] -> CUSTOM('MEM', addr), MEM[addr].type -> BITCAST
  if s[:4] == 'MEM[' and (e := _match(s, 3, '[', ']')) != -1:
    r, b = s[e+1:], UOp(Ops.CUSTOM, dtypes.void, (expr(s[4:e]),), arg='MEM')
    if not r: return b
    if r[:1] == '.' and (dt := _get_dtype(r[1:])): return UOp(Ops.BITCAST, dt, (b,))
  # Ternary: cond ? t : f -> WHERE
  if (q := _fop(s, ('?',))) > 0:
    d = b = 0
    for i in range(q+1, len(s)):
      if s[i] == '(': d += 1
      elif s[i] == ')': d -= 1
      elif s[i] == '[': b += 1
      elif s[i] == ']': b -= 1
      elif s[i] == ':' and d == 0 and b == 0: return UOp(Ops.WHERE, dtypes.void, (expr(s[:q]), expr(s[q+1:i]), expr(s[i+1:])))
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
        return UOp(_BINOPS[op], dtypes.void, (lhs, rhs), tag=tag)
  # Unary ops
  if s[0] in '-~!' and len(s) > 1 and (s[0] != '!' or s[1] != '='): return UOp(_UNOPS[s[0]], dtypes.void, (expr(s[1:]),))
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
      hi = UOp(Ops.SUB, dtypes.void, (UOp(Ops.ADD, dtypes.void, (st, w)), UOp(Ops.CONST, dtypes.int32, arg=1)))
      return UOp(Ops.CUSTOMI, dtypes.void, (expr(b), hi, st))
    if ':' in n and '?' not in n:
      d = 0
      for j, c in enumerate(n):
        if c in '([{': d += 1
        elif c in ')]}': d -= 1
        elif c == ':' and d == 0: return UOp(Ops.CUSTOMI, dtypes.void, (expr(b), expr(n[:j]), expr(n[j+1:])))
    idx = expr(n)
    return UOp(Ops.CUSTOMI, dtypes.void, (expr(b), idx, idx))
  # Bitcast: expr.type
  if '.' in s:
    for i in range(len(s)-1, 0, -1):
      if s[i] == '.' and (dt := _get_dtype(s[i+1:])): return UOp(Ops.BITCAST, dt, (expr(s[:i]),))
  # Variable
  if s[:5] == 'eval ': return UOp(Ops.DEFINE_VAR, dtypes.void, arg=(s, None, None))
  if re.match(r'^[A-Za-z_][\w.]*$', s): return UOp(Ops.DEFINE_VAR, dtypes.void, arg=(s, None, None))
  # Numeric literal
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
  line = line.split('//')[0].strip().rstrip(';')
  if not line: return None
  if line[:5] == 'eval ': return Assign(UOp(Ops.DEFINE_VAR, dtypes.void, arg=('_eval', None, None)), UOp(Ops.DEFINE_VAR, dtypes.void, arg=(line, None, None)))
  if line[:8] == 'declare ' and ':' in line:
    n, t = line[8:].split(':', 1)
    t = t.strip().split('[')[0]  # strip array suffix like [64]
    if m := re.match(r"^(\d+)'([IUFB])$", t): return Declare(n.strip(), _QDTYPES[f"{m[2].lower()}{m[1]}"])
    return None  # unsupported declare type
  for op, uop in [('+=', Ops.ADD), ('-=', Ops.SUB), ('|=', Ops.OR), ('&=', Ops.AND), ('^=', Ops.XOR), ('<<=', Ops.SHL), ('>>=', Ops.SHR)]:
    if op in line:
      l, r = line.split(op, 1)
      lhs = expr(l)
      return Assign(lhs, UOp(uop, dtypes.void, (lhs, expr(r))))
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

_OP_SYMS = {Ops.ADD: '+', Ops.SUB: '-', Ops.MUL: '*', Ops.FDIV: '/', Ops.MOD: '%', Ops.AND: '&', Ops.OR: '|',
             Ops.XOR: '^', Ops.SHL: '<<', Ops.SHR: '>>', Ops.CMPEQ: '==', Ops.CMPNE: '!=', Ops.CMPLT: '<', Ops.CMPLE: '<='}
_DT_STR = {dtypes.uint64: 'u64', dtypes.uint32: 'u32', dtypes.uint16: 'u16', dtypes.uint8: 'u8',
           dtypes.int64: 'i64', dtypes.int32: 'i32', dtypes.int16: 'i16', dtypes.int8: 'i8',
           dtypes.float64: 'f64', dtypes.float32: 'f32', dtypes.float16: 'f16'}

def _dt_bits(dt: DType) -> int:
  if m := re.search(r'(\d+)', dt.name): return int(m[1])
  return dt.itemsize * 8

def _pr(n, d=0):
  p = "  "*d
  match n:
    case UOp(Ops.CONST, dt, _, v):
      if dt == dtypes.int32: return str(int(v))
      if dt == dtypes.int64: return f"{int(v)}LL"
      if dt == dtypes.float32: return f"{v}F"
      if dt == dtypes.float16: return f"16'{v}"
      if dt == dtypes.uint32: return f"{int(v)}U"
      if dt == dtypes.uint64: return f"{int(v)}ULL"
      if dt == dtypes.int16: return f"16'{int(v)}"
      bits = _dt_bits(dt)
      if 'u' in dt.name or 'b' in dt.name: return f"{bits}'{int(v)}U"
      if 'f' in dt.name or 'float' in dt.name: return f"{bits}'{v}"
      if 'i' in dt.name or 'int' in dt.name: return f"{bits}'{int(v)}"
      return f"{v}"
    case UOp(Ops.DEFINE_VAR, _, _, (name, _, _)): return name
    case UOp(Ops.BITCAST, dt, (e,)): return f"{_pr(e)}.{_DT_STR.get(dt, dt.name)}"
    case UOp(Ops.CUSTOMI, _, (e, h, l)):
      if h is l: return f"{_pr(e)}[{_pr(h)}]"
      # Detect [start +: width] pattern: hi = (start + width) - 1, lo = start
      if h.op == Ops.SUB and h.src[1].op == Ops.CONST and h.src[1].arg == 1 and h.src[0].op == Ops.ADD and h.src[0].src[0] == l:
        return f"{_pr(e)}[{_pr(l)} +: {_pr(h.src[0].src[1])}]"
      return f"{_pr(e)}[{_pr(h)} : {_pr(l)}]"
    case UOp(Ops.CAST, dt, (e,)): return f"{_dt_bits(dt)}'{_DT_STR.get(dt, dt.name)[0].upper()}({_pr(e)})"
    case UOp(Ops.NEG, _, (x,)): return f"-{_pr(x)}"
    case UOp(Ops.XOR, _, (x,)) if len(n.src) == 1: return f"~{_pr(x)}"
    case UOp(Ops.CMPEQ, _, (x,)) if len(n.src) == 1: return f"!{_pr(x)}"
    case UOp(_, _, (l, r), _) if n.op in _OP_SYMS:
      sym = _OP_SYMS[n.op]
      left, right = l, r
      # Restore > / >= if tag indicates flipped
      if n.tag == 'flipped' and n.op == Ops.CMPLT: sym, left, right = '>', r, l
      if n.tag == 'flipped' and n.op == Ops.CMPLE: sym, left, right = '>=', r, l
      # Use <> for unordered not-equal
      if n.tag == '<>' and n.op == Ops.CMPNE: sym = '<>'
      # Use && / || for logical ops when operands are comparisons or bool-returning calls
      bool_ops = (Ops.CMPLT, Ops.CMPLE, Ops.CMPEQ, Ops.CMPNE, Ops.CUSTOM, Ops.AND, Ops.OR)
      if n.op == Ops.AND and l.op in bool_ops and r.op in bool_ops: sym = '&&'
      if n.op == Ops.OR and l.op in bool_ops and r.op in bool_ops: sym = '||'
      return f"{_pr(left)} {sym} {_pr(right)}"
    case UOp(Ops.WHERE, _, (c, t, f)): return f"{_pr(c)} ? {_pr(t)} : {_pr(f)}"
    case UOp(Ops.CUSTOM, _, args, 'MEM'): return f"MEM[{_pr(args[0])}]"
    case UOp(Ops.CUSTOM, _, args, name): return f"{name}({', '.join(_pr(x) for x in args)})"
    case UOp(Ops.CAT, _, exprs): return f"{{{', '.join(_pr(x) for x in exprs)}}}"
    case Assign(l, r):
      # Detect compound assignment: lhs = lhs op rhs -> lhs op= rhs
      compound = {Ops.ADD: '+=', Ops.SUB: '-=', Ops.OR: '|=', Ops.AND: '&=', Ops.XOR: '^=', Ops.SHL: '<<=', Ops.SHR: '>>='}
      if r.op in compound and len(r.src) == 2 and r.src[0] == l:
        return f"{p}{_pr(l)} {compound[r.op]} {_pr(r.src[1])}"
      return f"{p}{_pr(l)} = {_pr(r)}"
    case Declare(name, dt): return f"{p}declare {name} : {_dt_bits(dt)}'{_DT_STR.get(dt, dt.name)[0].upper()}"
    case If(br):
      parts = []
      for i, (c, b) in enumerate(br):
        kw = "if" if i == 0 else "elsif" if c is not None else "else"
        cond = f" {_pr(c)} then" if c is not None else ""
        body = "\n".join(_pr(s, d) for s in b)
        parts.append(f"{p}{kw}{cond}\n{body}")
      return "\n".join(parts) + f"\n{p}endif"
    case For(v, s, e, b): return f"{p}for {v} in {_pr(s)} : {_pr(e)} do\n" + "\n".join(_pr(x, d) for x in b) + f"\n{p}endfor"
    case tuple(): return "\n".join(_pr(x, d) for x in n)
    case _: return f"{p}{n}"

if __name__ == "__main__":
  from extra.assembly.amd.autogen.rdna3.str_pcode import PSEUDOCODE_STRINGS
  ok, fail, match, errs = 0, 0, 0, {}
  for cls, ops in PSEUDOCODE_STRINGS.items():
    for op, pc in ops.items():
      try: ast = parse(pc); ok += 1
      except Exception as e: fail += 1; errs[str(e)[:60]] = errs.get(str(e)[:60], 0) + 1
      else:
        rendered = _pr(ast)
        # Normalize for comparison
        def norm(s):
          s = re.sub(r'^[^=;]+(?=\n)', '', s)  # strip leading description lines (no = or ;)
          s = re.sub(r'//[^\n]*', '', s)  # strip comments
          s = re.sub(r'[;()\s]', '', s)  # strip semicolons, parens, whitespace
          s = re.sub(r'0x[0-9a-fA-F]+', lambda m: str(int(m[0], 16)), s)  # hex to decimal
          s = re.sub(r'\.b(\d+)', r'.u\1', s)  # .bXX -> .uXX
          s = re.sub(r"'B", "'U", s)  # 'B -> 'U
          s = re.sub(r'(\d+\.\d+)F', r'\1', s)  # strip F suffix from floats
          return s
        if norm(rendered) == norm(pc):
          match += 1
          if DEBUG >= 2: print(f"\033[32m{op.name}\033[0m")
        elif DEBUG: print(f"\033[31m{'='*60}\n{op.name}\n{'='*60}\n{pc}\n{'-'*60}\n{rendered}\n\033[0m")
  print(f"Parsed: {ok}/{ok+fail} ({100*ok/(ok+fail):.1f}%), Match: {match}/{ok} ({100*match/ok:.1f}%)")
  for e, c in sorted(errs.items(), key=lambda x: -x[1])[:10]: print(f"  {c}: {e}")
