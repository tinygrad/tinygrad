import re
from enum import Enum
from dataclasses import is_dataclass, fields
from tinygrad.dtype import dtypes, DType
from tinygrad.uop import Ops, GroupOp
from tinygrad.uop.ops import UOp, PatternMatcher, UPat, KernelInfo, range_str, sint
from tinygrad.helpers import strip_parens, colored

def pretty_print(x:UOp, cache=None, d=0)->str:
  def dfs(x:UOp, cache:dict):
    for s in x.src:
      cache.setdefault(s, [len(cache), 0, False])[1] += 1
      if cache[s][1] == 1: dfs(s, cache)
  if cache is None: dfs(x, cache:={})
  if (cx:=cache.setdefault(x, [0,0,False]))[2]: return f"{' '*d}x{cx[0]}"
  cx[2], srcs = True, (''.join(f'\n{pretty_print(s, cache, d+2)},' for s in x.src))
  return f"{' '*d}{f'x{cx[0]}:=' * (cx[1]>1)}{type(x).__name__}({x.op}, arg={x.argstr()}, src=({srcs}))"

# ***** SSA wire format *****

uops_colors = {Ops.LOAD: "#ffc0c0", Ops.STORE: "#87CEEB", Ops.CONST: "#e0e0e0", Ops.REDUCE: "#FF5B5B",
               Ops.RANGE: "#c8a0e0", Ops.BARRIER: "#ff8080", Ops.IF: "#c8b0c0", Ops.SPECIAL: "#c0c0ff",
               Ops.INDEX: "#CEF9B7", Ops.STACK: "#D8F9E4",
               Ops.WMMA: "#efefc0", Ops.UNSHARD: "#f6ccff", Ops.INS: "#eec4ff",
               **{x:"#D8F9E4" for x in GroupOp.Movement}, **{x:"#ffffc0" for x in GroupOp.ALU}, Ops.THREEFRY:"#ffff80",
               Ops.BUFFER: "#B0BDFF", Ops.GETADDR: "#9DB1F0", Ops.COPY: "#ff90c0", Ops.CUSTOM_FUNCTION: "#bf71b6",
               Ops.CALL: "#00B7C8", Ops.PARAM: "#14686F", Ops.SOURCE: "#c0c0c0", Ops.BINARY: "#404040",
               Ops.LINEAR: "#7DF4FF", Ops.ALLOC: "#C07788", Ops.PROGRAM: "#a0a0a0",
               Ops.ALLREDUCE: "#ff40a0", Ops.MSELECT: "#d040a0", Ops.MSTACK: "#d040a0",
               Ops.STAGE: "#FFC14D", Ops.REWRITE_ERROR: "#1a1b26", Ops.AFTER: "#8A7866", Ops.END: "#524C46", Ops.BACKEDGE: "#464752"}

def _const_str(x:UOp) -> str:
  """CONSTs are weak or bool, so literals are bare: 1, 2.5, -inf, true, invalid. typed values are CASTs of these"""
  if x.is_invalid: return "invalid"
  return str(x.val).lower() if isinstance(x.val, bool) else repr(x.val) if isinstance(x.val, int) else repr(float(x.val))

def _val(v) -> str:
  """an arg value: tuples are [a b], enums and dtypes by name, one-word strings bare, anything else quoted"""
  if isinstance(v, bool): return str(v).lower()
  if isinstance(v, Ops): return v.name.lower()
  if isinstance(v, Enum): return v.name
  if isinstance(v, DType): return v.name
  if isinstance(v, (int, float)): return repr(v)
  if isinstance(v, tuple): return "[" + " ".join(map(_val, v)) + "]"
  if isinstance(v, str) and re.fullmatch(r"[\w:.\-]+", v): return v
  return repr(v if isinstance(v, str) else repr(v))

# single-value args get a name, structured args are key=value per field. runtime objects (Buffer, BufferSpec, grad_fxn) are not rendered
_arg_key = {Ops.CAST:"dtype", Ops.BITCAST:"dtype", Ops.COPY:"device", Ops.GETADDR:"device", Ops.SPECIAL:"name", Ops.UNSHARD:"axes",
            Ops.MSELECT:"index"}
_skip_fields = {"buffer", "spec", "grad_fxn"}

def _render_arg(x:UOp) -> str:
  """args are key=value, written after the srcs"""
  if (a:=x.arg) is None: return ""
  match x.op:
    case Ops.CONST: return _const_str(x)
    case Ops.RANGE: return f"axis={a[0].name} id={_val(a[1] if len(a) == 2 else a[1:])}"   # flatten_range merges ids: id=[1 2]
    case Ops.REDUCE: return f"op={_val(a[0])}" + (f" pop={a[1]}" if a[1] else "")
    case Ops.ALLREDUCE: return f"op={_val(a[0])} device={_val(a[1])}"
    case Ops.SINK if isinstance(a, KernelInfo): return f"name={_val(a.name)}"
  if x.op not in _arg_key and is_dataclass(a):
    return " ".join(f"{f.name}={_val(v)}" for f in fields(a) if f.name not in _skip_fields and (v:=getattr(a, f.name)) != f.default)
  return f"{_arg_key.get(x.op, 'arg')}={_val(a)}"

# CONSTs never get lines (inline literals, no %id); concrete all-const STACKs inline into their parent as [a b]
def _inline(u:UOp) -> bool: return u.op is Ops.CONST or (u.op is Ops.STACK and all(s.op is Ops.CONST for s in u.src))

def render_uir(root:UOp|list[UOp]) -> str:
  """one line per UOp: %id = op src src key=value"""
  nodes = [u for u in (list(root.toposort(enter_calls=True)) if isinstance(root, UOp) else list(root)) if not _inline(u)]
  table = {u:i for i,u in enumerate(nodes)}
  def src_str(u:UOp) -> str:
    if not _inline(u): return f"%{table[u]}"
    return _const_str(u) if u.op is Ops.CONST else "[" + " ".join(src_str(s) for s in u.src) + "]"
  return "\n".join(" ".join(filter(None, [f"%{i} = {colored(u.op.name.lower(), uops_colors.get(u.op))}", *map(src_str, u.src), _render_arg(u)]))
                   for i,u in enumerate(nodes))


# for debug
syms = { Ops.ADD: "+", Ops.SUB: "-", Ops.DIV: "//", Ops.MOD: "%", Ops.SHL: "<<", Ops.SHR: ">>",
         Ops.MUL: "*", Ops.CMPLT: "<", Ops.CMPNE: "!=", Ops.CMPEQ: "==", Ops.AND: "&", Ops.OR: "|", Ops.XOR: "^"}
# comparison operators are not in here because they are chained in python, not left-associative
precedence = {Ops.MUL:1, Ops.DIV:1, Ops.MOD:1, Ops.ADD:2, Ops.SUB:2, Ops.SHL:3, Ops.SHR:3, Ops.AND:4, Ops.XOR:5, Ops.OR:6}
def strip_binary_parens(x:UOp, left:str, right:str, code_for_op) -> str:
  if x.op not in precedence: return code_for_op(left, right)
  return code_for_op(strip_parens(left) if precedence.get(x.src[0].op,99)<=precedence[x.op] else left, strip_parens(right) if
    precedence.get(x.src[1].op,99)<precedence[x.op] else right)

# marg is ssimplify'd, so a bound can be a node this graph never contained
def marg_str(ctx, a:sint) -> str: return str(a) if not isinstance(a, UOp) else ctx[a] if a in ctx else a.render()

def render_marg(ctx,x:UOp):
  if x.op is Ops.PERMUTE: return str(x.marg)
  pieces = []
  if x.op in {Ops.RESHAPE, Ops.EXPAND}: pieces = [marg_str(ctx, a) for a in x.marg]
  if x.op in {Ops.PAD, Ops.SHRINK}: pieces = [f"({marg_str(ctx, a[0])}, {marg_str(ctx, a[1])})" for a in x.marg]
  return f"({','.join(pieces)})" if len(pieces) != 1 else f"({pieces[0]},)"

def render_index(srcs) -> str: return ''.join(f"[{strip_parens(src)}]" for src in srcs)

renderer = PatternMatcher([
  (UPat(Ops.PARAM, name="x"), lambda x: x.arg.name if x.arg.name is not None else f"p{x.arg.slot}"),
  (UPat((Ops.BUFFER, Ops.ALLOC), name="x"), lambda x:
   x.arg.name if x.arg.name is not None else f"{'a' if x.op is Ops.ALLOC else 'b'}{x.arg.slot}"),
  (UPat(Ops.AFTER, name="x"), lambda ctx,x: ctx[x.src[0]]),
  (UPat((Ops.SPECIAL), name="x"), lambda x: x.arg),
  (UPat(Ops.RANGE, dtypes.void, name="x"), lambda x: f"loop{x.axis_id[0]}"),
  (UPat(Ops.RANGE, name="x"), lambda x: f"r{range_str(x)}"),
  (UPat(Ops.CONST, name="x"), lambda x: str(x.val)),
  # CAST states the width, the weak CONST carries the value
  (UPat.cvar("c").cast(), lambda c: str(c.val)),
  (UPat(Ops.CAST, name="x"), lambda ctx,x: f"({str(x.dtype)[7:]})({ctx[x.src[0]]})"),
  (UPat(Ops.NEG, name="x"), lambda ctx,x: f"(-{ctx[x.src[0]]})"),
  (UPat(Ops.RECIP, name="x"), lambda ctx,x: f"(1/{ctx[x.src[0]]})"),
  (UPat((Ops.MAX, Ops.TRUNCDIV, Ops.TRUNCMOD), name="x"), lambda ctx,x: f"{x.op.name.lower()}({ctx[x.src[0]]}, {ctx[x.src[1]]})"),
  (UPat(Ops.MULACC, name="x"), lambda ctx,x: f"({ctx[x.src[0]]}*{ctx[x.src[1]]}+{ctx[x.src[2]]})"),
  (UPat(Ops.WHERE, name="x"), lambda ctx,x: f"({ctx[x.src[1]]} if {ctx[x.src[0]]} else {ctx[x.src[2]]})"),
  (UPat(GroupOp.Movement, name="x"), lambda ctx,x: f"{ctx[x.src[0]]}.{x.op.name.lower()}({render_marg(ctx, x)})"),
  (UPat(set(syms.keys()), name="x"), lambda ctx,x: strip_binary_parens(x, ctx[x.src[0]], ctx[x.src[1]], lambda a,b: f"({a}{syms[x.op]}{b})")),
  (UPat((Ops.INDEX, Ops.STAGE), name="x"), lambda x, ctx: render_index(ctx[y] for y in x.src[1:])),
  (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, name="idx"),)), lambda ctx,idx: f"{ctx[idx.src[0]]}{ctx[idx]}"),
  (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, name="idx"), UPat(name="alt"), UPat(name="gate"))),
   lambda ctx,idx,alt,gate: f"({ctx[idx.src[0]]}{ctx[idx]} if {ctx[gate]} else {ctx[alt]})"),
  (UPat(Ops.STACK, name="x"), lambda ctx,x: f"{{{','.join([ctx[y] for y in x.src])}}}"),
  (UPat(GroupOp.All, name="x"), lambda x: str(x)),
])

renderer_infer = PatternMatcher([
  (UPat((Ops.MOD, Ops.DIV), name="x"), lambda ctx,x: f"{x.op.name.lower()}({ctx[x.src[0]]}, {ctx[x.src[1]]})"),
  (UPat(Ops.CAST, name="x"),
    lambda ctx,x: f"{'float' if dtypes.is_float(x.dtype) else 'bool' if x.dtype is dtypes.bool else 'int'}({ctx[x.src[0]]})"),
  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"bitcast({ctx[x.src[0]]}, {x.src[0].dtype!r}, {x.dtype!r})"),
]) + renderer

def _render_with_splits(lst:list[UOp], pm:PatternMatcher, to_render:set[UOp], split_depth:int=100) -> dict[str, str]:
  r: dict[UOp, str] = {}
  ret: dict[str, str] = {}
  depth: dict[UOp, int] = {}
  for i,u in enumerate(lst):
    # limit inline depth to avoid "too many nested parentheses" in Python parser
    op_depth = 1 + max([depth.get(s, 0) for s in u.src], default=0)
    if op_depth > split_depth: to_render.add(u)
    depth[u] = 0 if u in to_render else op_depth
    ren = pm.rewrite(u, ctx=r)
    assert isinstance(ren, str)
    if u not in to_render: r[u] = ren
    else:
      r[u] = f"c{i}" if u is not lst[-1] else "ast"
      ret[r[u]] = ren
  return ret
