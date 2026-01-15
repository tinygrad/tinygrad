import unittest, re, os, random
from tinygrad.dtype import dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp
from extra.assembly.amd.pcode_parse import parse, _BINOPS, _QDTYPES, If, For, Lambda, Break, Return
from extra.assembly.amd.pcode_transform import parse_transform
from extra.assembly.amd.autogen.rdna3.str_pcode import PCODE as RDNA3_PCODE
from extra.assembly.amd.autogen.rdna4.str_pcode import PCODE as RDNA4_PCODE
from extra.assembly.amd.autogen.cdna.str_pcode import PCODE as CDNA_PCODE

DEBUG = int(os.getenv("DEBUG", "0"))

# NOTE: After refactor, parse() outputs void dtypes for most ops.
# Dtype inference happens in parse_transform() via pcode_pm PatternMatcher.
# Void checking is no longer done at parse time - it's expected to have voids.

_OP_SYMS = {v: k for k, v in _BINOPS.items() if k not in ('>', '>=', '<>', '||', '&&')}
_DT_STR = {v: k for k, v in _QDTYPES.items() if k in ('u64', 'u32', 'u16', 'u8', 'i64', 'i32', 'i16', 'i8', 'f64', 'f32', 'f16', 'bf16')}

def _dt_bits(dt):
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
    case UOp(Ops.DEFINE_VAR, dt, _, (name, _, _)):
      if dt == dtypes.void: return name  # reference
      # declaration
      bits = _dt_bits(dt.scalar()) if dt.count > 1 else _dt_bits(dt)
      base = dt.scalar() if dt.count > 1 else dt
      tchar = 'U' if 'uint' in base.name or base.name.startswith('u') else 'I' if 'int' in base.name or base.name.startswith('i') else 'F' if 'float' in base.name else 'B'
      arr = f"[{dt.count}]" if dt.count > 1 else ""
      return f"declare {name} : {bits}'{tchar}{arr}"
    case UOp(Ops.BITCAST, dt, (e,)): return f"{_pr(e)}.{_DT_STR.get(dt, dt.name)}"
    case UOp(Ops.CUSTOMI, _, (e, h, l)):
      if h is l: return f"{_pr(e)}[{_pr(h)}]"
      if h.op == Ops.SUB and h.src[1].op == Ops.CONST and h.src[1].arg == 1 and h.src[0].op == Ops.ADD and h.src[0].src[0] == l:
        return f"{_pr(e)}[{_pr(l)} +: {_pr(h.src[0].src[1])}]"
      return f"{_pr(e)}[{_pr(h)} : {_pr(l)}]"
    case UOp(Ops.CAST, dt, (e,)): return f"{_dt_bits(dt)}'{_DT_STR.get(dt, dt.name)[0].upper()}({_pr(e)})"
    case UOp(Ops.NEG, _, (x,)): return f"-{_pr(x)}"
    case UOp(Ops.XOR, _, (x,)) if len(n.src) == 1: return f"~{_pr(x)}"
    case UOp(Ops.CMPEQ, _, (x,)) if len(n.src) == 1: return f"!{_pr(x)}"
    case UOp(Ops.CMPNE, dtypes.bool, (a, b)) if a == b: return f"isNAN({_pr(a)})"
    # isINF(x) -> OR(CMPEQ(x, +inf), CMPEQ(x, -inf))
    case UOp(Ops.OR, dtypes.bool, (UOp(Ops.CMPEQ, _, (x1, UOp(Ops.CONST, _, _, c1))), UOp(Ops.CMPEQ, _, (x2, UOp(Ops.CONST, _, _, c2))))) if x1 == x2 and c1 == float('inf') and c2 == float('-inf'):
      return f"isINF({_pr(x1)})"
    # fract(x) -> SUB(x, floor(x)) where floor(x) = WHERE(CMPLT(x, TRUNC(x)), SUB(TRUNC(x), 1), TRUNC(x))
    case UOp(Ops.SUB, _, (x1, UOp(Ops.WHERE, _, (UOp(Ops.CMPLT, _, (x2, UOp(Ops.TRUNC, _, (x3,)))), UOp(Ops.SUB, _, (UOp(Ops.TRUNC, _, (x4,)), UOp(Ops.CONST, _, _, c))), UOp(Ops.TRUNC, _, (x5,)))))) if c in (1, 1.0) and x1 == x2 == x3 == x4 == x5:
      return f"fract({_pr(x1)})"
    case UOp(_, _, (l, r), _) if n.op in _OP_SYMS:
      sym = _OP_SYMS[n.op]
      left, right = l, r
      if n.tag == 'flipped' and n.op == Ops.CMPLT: sym, left, right = '>', r, l
      if n.tag == 'flipped' and n.op == Ops.CMPLE: sym, left, right = '>=', r, l
      if n.tag == '<>' and n.op == Ops.CMPNE: sym = '<>'
      return f"{_pr(left)} {sym} {_pr(right)}"
    # clamp(x, lo, hi) -> WHERE(CMPLT(hi, WHERE(CMPLT(x, lo), lo, x)), hi, WHERE(CMPLT(x, lo), lo, x))
    case UOp(Ops.WHERE, _, (UOp(Ops.CMPLT, _, (hi, UOp(Ops.WHERE, _, (UOp(Ops.CMPLT, _, (x1, lo1)), lo2, x2)))), hi2, UOp(Ops.WHERE, _, (UOp(Ops.CMPLT, _, (x3, lo3)), lo4, x4)))) if hi == hi2 and x1 == x2 == x3 == x4 and lo1 == lo2 == lo3 == lo4:
      return f"clamp({_pr(x1)}, {_pr(lo1)}, {_pr(hi)})"
    # abs(x) -> WHERE(CMPLT(x, 0), NEG(x), x)
    case UOp(Ops.WHERE, _, (UOp(Ops.CMPLT, _, (x1, UOp(Ops.CONST, _, _, c))), UOp(Ops.NEG, _, (x2,)), x3)) if c in (0, 0.0) and x1 == x2 == x3:
      return f"abs({_pr(x1)})"
    # floor(x) -> WHERE(CMPLT(x, TRUNC(x)), SUB(TRUNC(x), 1), TRUNC(x))
    case UOp(Ops.WHERE, _, (UOp(Ops.CMPLT, _, (x1, UOp(Ops.TRUNC, _, (x2,)))), UOp(Ops.SUB, _, (UOp(Ops.TRUNC, _, (x3,)), UOp(Ops.CONST, _, _, c))), UOp(Ops.TRUNC, _, (x4,)))) if c in (1, 1.0) and x1 == x2 == x3 == x4:
      return f"floor({_pr(x1)})"
    case UOp(Ops.WHERE, _, (c, t, f)): return f"{_pr(c)} ? {_pr(t)} : {_pr(f)}"
    case UOp(Ops.TRUNC, _, (x,)): return f"trunc({_pr(x)})"
    case UOp(Ops.SQRT, _, (x,)): return f"sqrt({_pr(x)})"
    case UOp(Ops.EXP2, _, (x,)): return f"exp2({_pr(x)})"
    case UOp(Ops.LOG2, _, (x,)): return f"log2({_pr(x)})"
    # cos(x) -> SIN(ADD(x, π/2))
    case UOp(Ops.SIN, _, (UOp(Ops.ADD, _, (x, UOp(Ops.CONST, _, _, c))),)) if abs(c - 1.5707963267948966) < 1e-10: return f"cos({_pr(x)})"
    case UOp(Ops.SIN, _, (x,)): return f"sin({_pr(x)})"
    case UOp(Ops.RECIPROCAL, _, (UOp(Ops.SQRT, _, (x,)),)): return f"rsqrt({_pr(x)})"
    case UOp(Ops.RECIPROCAL, _, (x,)): return f"rcp({_pr(x)})"
    case UOp(Ops.MULACC, _, (a, b, c)): return f"fma({_pr(a)}, {_pr(b)}, {_pr(c)})"
    case UOp(Ops.CUSTOM, _, args, 'MEM'): return f"MEM[{_pr(args[0])}]"
    case UOp(Ops.CUSTOM, _, args, name): return f"{name}({', '.join(_pr(x) for x in args)})"
    case UOp(Ops.CAT, _, exprs): return f"{{{', '.join(_pr(x) for x in exprs)}}}"
    case UOp(Ops.ASSIGN, _, (l, r)):
      compound = {Ops.ADD: '+=', Ops.SUB: '-=', Ops.OR: '|=', Ops.AND: '&=', Ops.XOR: '^=', Ops.SHL: '<<=', Ops.SHR: '>>='}
      is_pc = l.op == Ops.DEFINE_VAR and l.arg[0] == 'PC'
      if isinstance(r, UOp) and r.op in compound and len(r.src) == 2 and r.src[0] == l and not is_pc:
        return f"{p}{_pr(l)} {compound[r.op]} {_pr(r.src[1])}"
      # Chained assignment: render without prefix for RHS
      rhs = _pr(r) if (isinstance(r, UOp) and r.op == Ops.ASSIGN) else _pr(r)
      return f"{p}{_pr(l)} = {rhs}"
    case UOp(Ops.DEFINE_VAR, dt, _, name) if isinstance(name, str):
      base = dt.scalar() if dt.count > 1 else dt
      suffix = f"[{dt.count}]" if dt.count > 1 else ""
      return f"{p}declare {name} : {_dt_bits(base)}'{_DT_STR.get(base, base.name)[0].upper()}{suffix}"
    case If(br):
      parts = []
      for i, (c, b) in enumerate(br):
        kw = "if" if i == 0 else "elsif" if c is not None else "else"
        cond = f" {_pr(c)} then" if c is not None else ""
        body = "\n".join(_pr(s, d) for s in b)
        parts.append(f"{p}{kw}{cond}\n{body}")
      return "\n".join(parts) + f"\n{p}endif"
    case For(v, s, e, b): return f"{p}for {v} in {_pr(s)} : {_pr(e)} do\n" + "\n".join(_pr(x, d) for x in b) + f"\n{p}endfor"
    case Break(): return f"{p}break"
    case Return(v): return f"{p}return {_pr(v)}"
    case Lambda(name, params, body):
      body_str = _pr(body) if isinstance(body, UOp) else "\n".join(_pr(x, d) for x in body)
      return f"{p}{name} = lambda({', '.join(params)}) (\n{body_str});"
    case tuple(): return "\n".join(_pr(x, d) for x in n)
    case _: return f"{p}{n}"

def _norm(s, keep_structure=False):
  s = re.sub(r'//[^\n]*', '', s)  # comments
  s = re.sub(r'0x[0-9a-fA-F]+', lambda m: str(int(m[0], 16)), s)  # hex literals
  s = re.sub(r"(\d+)U(?!LL)", r"\1", s)  # strip U suffix (but not ULL)
  s = re.sub(r'\.b(\d+)', r'.u\1', s)  # .b32 -> .u32
  s = re.sub(r"'B", "'U", s)  # 'B -> 'U
  s = re.sub(r'(\d+\.\d+)F', r'\1', s)  # 0.5F -> 0.5
  s = re.sub(r'\+INF', 'INF', s)  # +INF -> INF
  s = re.sub(r'&&', '&', s)  # && -> &
  s = re.sub(r'\|\|', '|', s)  # || -> |
  if keep_structure:
    s = re.sub(r';', '', s)
    s = re.sub(r'\n\s*\n', '\n', s)
  else:
    s = re.sub(r'[;()\s]', '', s)
  return s.strip()

def _pp(stmt, indent=0) -> str:
  """Pretty print a parsed statement with proper indentation."""
  pad = "  " * indent
  match stmt:
    case UOp():
      return f"{stmt}"
    case If(branches):
      lines = [f"{pad}If("]
      for cond, body in branches:
        lines.append(f"{pad}  ({cond},")
        lines.append(f"{pad}   [")
        for s in body: lines.append(_pp(s, indent + 2) + ",")
        lines.append(f"{pad}   ]),")
      lines.append(f"{pad})")
      return "\n".join(lines)
    case For(var, start, end, body):
      lines = [f"{pad}For({var!r}, {start}, {end},"]
      lines.append(f"{pad}  [")
      for s in body: lines.append(_pp(s, indent + 1) + ",")
      lines.append(f"{pad}  ])")
      return "\n".join(lines)
    case Break(): return f"{pad}Break()"
    case Return(v): return f"{pad}Return({v})"
    case Lambda(name, params, body):
      if isinstance(body, UOp): return f"{pad}Lambda({name!r}, {params}, {body})"
      lines = [f"{pad}Lambda({name!r}, {params}, ["]
      for s in body: lines.append(_pp(s, indent + 1) + ",")
      lines.append(f"{pad}])")
      return "\n".join(lines)
    case _: return f"{pad}{stmt}"

def _test_arch(test, pcode_strings, min_parse, min_roundtrip, min_transform, name=""):
  ok, fail, match, transform_ok, transform_fail = 0, 0, 0, 0, 0
  parse_errs: dict[str, list[str]] = {}
  transform_errs: dict[str, list[str]] = {}

  # test in random order - flat dict {op: pcode}
  triples = [(type(op), op, pc) for op, pc in pcode_strings.items()]
  random.shuffle(triples)

  for cls, op, pc in triples:
    # Phase 1: parse
    try:
      ast = parse(pc)
      ok += 1
    except Exception as e:
      fail += 1
      key = str(e)[:60]
      if key not in parse_errs: parse_errs[key] = []
      parse_errs[key].append(f"{cls.__name__}.{op.name}")
      continue
    # Phase 2: transform (separate metric)
    try:
      ast_pt = parse_transform(pc)
      transform_ok += 1
    except Exception as e:
      transform_fail += 1
      key = str(e)[:60]
      if key not in transform_errs: transform_errs[key] = []
      transform_errs[key].append(f"{cls.__name__}.{op.name}")
      ast_pt = None
    # Phase 3: roundtrip
    rendered = _pr(ast)
    if _norm(pc) == _norm(rendered):
      match += 1
      if DEBUG >= 2:
        print(f"{'='*60}\n\033[32m{op.name}\033[0m\n{'='*60}")
        print(pc)
        if DEBUG >= 3 and ast_pt:
          for stmt in ast_pt: print(_pp(stmt))
    elif DEBUG:
      orig_lines = [l for l in _norm(pc, keep_structure=True).split('\n') if l.strip()]
      rend_lines = [l for l in rendered.split('\n') if l.strip()]
      max_lines = max(len(orig_lines), len(rend_lines))
      print(f"{'='*60}\n\033[31m{op.name}\033[0m\n{'='*60}")
      w = 60
      for i in range(max_lines):
        oline = orig_lines[i] if i < len(orig_lines) else ''
        rline = rend_lines[i] if i < len(rend_lines) else ''
        line_match = _norm(oline) == _norm(rline)
        color = '' if line_match else '\033[31m'
        reset = '' if line_match else '\033[0m'
        print(f"{color}{oline:<{w}} | {rline}{reset}")
  total = ok + fail
  parse_rate = 100 * ok / total
  roundtrip_rate = 100 * match / total
  transform_rate = 100 * transform_ok / total
  print(f"{name}: Parsed: {ok}/{total} ({parse_rate:.1f}%), Roundtrip: {match}/{total} ({roundtrip_rate:.1f}%), Transform: {transform_ok}/{total} ({transform_rate:.1f}%)")
  if DEBUG:
    if parse_errs:
      print("Parse errors:")
      for e, ops in sorted(parse_errs.items(), key=lambda x: -len(x[1])): print(f"  {len(ops)}: {e} ({ops[0]})")
    if transform_errs:
      print("Transform errors:")
      for e, ops in sorted(transform_errs.items(), key=lambda x: -len(x[1])): print(f"  {len(ops)}: {e} ({ops[0]})")
  test.assertGreater(parse_rate, min_parse, f"Parse rate {parse_rate:.1f}% should be >{min_parse}%")
  test.assertGreater(roundtrip_rate, min_roundtrip, f"Roundtrip rate {roundtrip_rate:.1f}% should be >{min_roundtrip}%")
  test.assertGreater(transform_rate, min_transform, f"Transform rate {transform_rate:.1f}% should be >{min_transform}%")

class TestQcodeParseAndRoundtrip(unittest.TestCase):
  def test_rdna3(self): _test_arch(self, RDNA3_PCODE, min_parse=98, min_roundtrip=97, min_transform=97, name="RDNA3")
  def test_rdna4(self): _test_arch(self, RDNA4_PCODE, min_parse=96, min_roundtrip=96, min_transform=96, name="RDNA4")
  def test_cdna(self): _test_arch(self, CDNA_PCODE, min_parse=94, min_roundtrip=93, min_transform=92, name="CDNA")

if __name__ == "__main__":
  unittest.main()
