import unittest, re, os
from tinygrad.dtype import dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp
from extra.assembly.amd.qcode import parse, _BINOPS, _QDTYPES, Assign, Declare, If, For, Lambda, Break, Return
from extra.assembly.amd.autogen.rdna3.str_pcode import PSEUDOCODE_STRINGS as RDNA3_PCODE
from extra.assembly.amd.autogen.rdna4.str_pcode import PSEUDOCODE_STRINGS as RDNA4_PCODE
from extra.assembly.amd.autogen.cdna.str_pcode import PSEUDOCODE_STRINGS as CDNA_PCODE

DEBUG = int(os.getenv("DEBUG", "0"))

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
    case UOp(Ops.DEFINE_VAR, _, _, (name, _, _)): return name
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
    case UOp(_, _, (l, r), _) if n.op in _OP_SYMS:
      sym = _OP_SYMS[n.op]
      left, right = l, r
      if n.tag == 'flipped' and n.op == Ops.CMPLT: sym, left, right = '>', r, l
      if n.tag == 'flipped' and n.op == Ops.CMPLE: sym, left, right = '>=', r, l
      if n.tag == '<>' and n.op == Ops.CMPNE: sym = '<>'
      return f"{_pr(left)} {sym} {_pr(right)}"
    case UOp(Ops.WHERE, _, (c, t, f)): return f"{_pr(c)} ? {_pr(t)} : {_pr(f)}"
    case UOp(Ops.CUSTOM, _, args, 'MEM'): return f"MEM[{_pr(args[0])}]"
    case UOp(Ops.CUSTOM, _, args, name): return f"{name}({', '.join(_pr(x) for x in args)})"
    case UOp(Ops.CAT, _, exprs): return f"{{{', '.join(_pr(x) for x in exprs)}}}"
    case Assign(l, r):
      compound = {Ops.ADD: '+=', Ops.SUB: '-=', Ops.OR: '|=', Ops.AND: '&=', Ops.XOR: '^=', Ops.SHL: '<<=', Ops.SHR: '>>='}
      is_pc = l.op == Ops.DEFINE_VAR and l.arg[0] == 'PC'
      if isinstance(r, UOp) and r.op in compound and len(r.src) == 2 and r.src[0] == l and not is_pc:
        return f"{p}{_pr(l)} {compound[r.op]} {_pr(r.src[1])}"
      # Chained assignment: render without prefix for RHS
      rhs = _pr(r) if isinstance(r, Assign) else _pr(r)
      return f"{p}{_pr(l)} = {rhs}"
    case Declare(name, dt):
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
  while True:
    m = re.match(r'^(?!declare|if |for )[^=;\n]+\n', s)
    if not m: break
    s = s[m.end():]
  s = re.sub(r'//[^\n]*', '', s)
  s = re.sub(r'0x[0-9a-fA-F]+', lambda m: str(int(m[0], 16)), s)  # convert hex before stripping whitespace
  s = re.sub(r"(\d+)U(?!LL)", r"\1", s)  # strip U suffix early before whitespace removal
  if keep_structure:
    s = re.sub(r';', '', s)
    s = re.sub(r'\n\s*\n', '\n', s)
  else:
    s = re.sub(r'[;()\s]', '', s)
  s = re.sub(r'_eval=', '', s)
  s = re.sub(r'\.b(\d+)', r'.u\1', s)
  s = re.sub(r"'B", "'U", s)
  s = re.sub(r'(\d+\.\d+)F', r'\1', s)
  s = re.sub(r'\+INF', 'INF', s)
  s = re.sub(r'&&', '&', s)
  s = re.sub(r'\|\|', '|', s)
  return s.strip()

def _test_arch(test, pcode_strings, min_parse=98, min_roundtrip=98):
  ok, fail, match = 0, 0, 0
  errs: dict[str, list[str]] = {}
  for cls, ops in pcode_strings.items():
    for op, pc in ops.items():
      try:
        ast = parse(pc)
        ok += 1
      except Exception as e:
        fail += 1
        key = str(e)[:60]
        if key not in errs: errs[key] = []
        errs[key].append(f"{cls.__name__}.{op.name}")
        continue
      rendered = _pr(ast)
      if _norm(pc) == _norm(rendered):
        match += 1
        if DEBUG >= 2: print(f"\033[32m{op.name}\033[0m")
      elif DEBUG:
        orig_lines = [l for l in _norm(pc, keep_structure=True).split('\n') if l.strip()]
        rend_lines = [l for l in rendered.split('\n') if l.strip()]
        max_lines = max(len(orig_lines), len(rend_lines))
        print(f"{'='*60}\n{op.name}\n{'='*60}")
        w = 50
        for i in range(max_lines):
          oline = orig_lines[i] if i < len(orig_lines) else ''
          rline = rend_lines[i] if i < len(rend_lines) else ''
          line_match = _norm(oline) == _norm(rline)
          color = '' if line_match else '\033[31m'
          reset = '' if line_match else '\033[0m'
          print(f"{color}{oline:<{w}} | {rline}{reset}")
  total = ok + fail
  parse_rate = 100 * ok / total
  roundtrip_rate = 100 * match / ok if ok > 0 else 0
  if DEBUG:
    print(f"Parsed: {ok}/{total} ({parse_rate:.1f}%), Match: {match}/{ok} ({roundtrip_rate:.1f}%)")
    for e, ops in sorted(errs.items(), key=lambda x: -len(x[1])):
      print(f"  {len(ops)}: {e} ({ops[0]})")
  test.assertGreater(parse_rate, min_parse, f"Parse rate {parse_rate:.1f}% should be >{min_parse}%")
  test.assertGreater(roundtrip_rate, min_roundtrip, f"Roundtrip rate {roundtrip_rate:.1f}% should be >{min_roundtrip}%")

class TestQcodeParseAndRoundtrip(unittest.TestCase):
  def test_rdna3(self): _test_arch(self, RDNA3_PCODE)
  def test_rdna4(self): _test_arch(self, RDNA4_PCODE, min_parse=96)
  def test_cdna(self): _test_arch(self, CDNA_PCODE, min_parse=95)

if __name__ == "__main__":
  unittest.main()
