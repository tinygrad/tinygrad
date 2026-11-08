"""UIR: the text form of a UOp graph. one line per UOp, `%id = op src src key=value`, like studentgrad.
parse_uir also reads studentgrad's hand-written form: %names, nested (op ...), [stacks] of expressions, and {scoped blocks}.

  %0 = param 4 slot=0 dtype=f32 device=CPU
  %1 = range 4 slot=0
  %2 = index %0 %1
  %3 = store %2 1.0
  %4 = end %3 %1
  %5 = sink %4 name=test

srcs are %refs or inline literals (CONSTs are never lines, all-CONST STACKs are [a b]). args come after the srcs as key=value.
a value is a literal (1, 2.5, -inf, true), a bare word, a 'quoted string', a [list], a %ref, or a `python object` for anything else.
parse_uir(render_uir(x)) is x: UOps are hash-consed, so a lossless round trip gives back the identical UOp.
"""
import re, sys, math, ast, functools, importlib, pickle, base64
from collections import deque
from enum import Enum
from dataclasses import is_dataclass, fields
from typing import Any, Callable
from tinygrad.dtype import dtypes, DType, AddrSpace, Invalid
from tinygrad.uop import Ops, GroupOp
from tinygrad.uop.ops import UOp, KernelInfo, CallInfo, CustomFunction, ProgramInfo, ParamArg, AxisType
from tinygrad.helpers import colored, ansistrip, Target

# ***** python objects *****
# an arg value with no UIR form is a `python expression`. it's built from the value's structure (containers, enums, classes, functools.partial,
# dataclasses, UOps as %refs), and anything else is its repr, evaluated in its own module when written `module:expr`.
# every expression is checked to rebuild the value when it's rendered, and a value no expression rebuilds is pickled.

_names_cache: list = [-1, {}]
def _names_index() -> dict[str, list[Any]]:
  """name -> the distinct objects loaded tinygrad modules define with that name, rebuilt when modules load"""
  if _names_cache[0] != len(sys.modules):
    idx: dict[str, dict[int, Any]] = {}
    for m in list(sys.modules.values()):
      if getattr(m, "__name__", "").startswith("tinygrad"):
        for k, v in list(vars(m).items()): idx.setdefault(k, {})[id(v)] = v
    _names_cache[:] = [len(sys.modules), {k: list(v.values()) for k,v in idx.items()}]
  return _names_cache[1]

class _Names(dict):
  """the names an expression can use: a name defined by exactly one object in loaded tinygrad modules, or a loaded module"""
  def __missing__(self, k:str):
    if len(objs:=_names_index().get(k, [])) == 1: return objs[0]
    if k in sys.modules: return sys.modules[k]
    raise KeyError(k)   # a KeyError falls through to builtins

# modules generated per target (rdna3, rdna4, ...) define the same names, so their names are always written with their module
_per_target = "tinygrad.runtime.autogen."

def _name(obj) -> str:
  """a class or function by its name if that's unambiguous, else by its module path"""
  if not obj.__module__.startswith(_per_target) and len(objs:=_names_index().get(obj.__name__, [])) == 1 and objs[0] is obj: return obj.__name__
  return f"{obj.__module__}.{obj.__qualname__}"

def _py(v, ref:Callable[[UOp], str|None]) -> str:
  """the python expression for v, from its structure where it has one"""
  if v is None or type(v) in (bool, int, float, str, bytes): return repr(v)
  if isinstance(v, UOp): return r if (r:=ref(v)) is not None else f"UOp({_py(v.op, ref)}, {_py(v.src, ref)}, {_py(v.arg, ref)})"
  if isinstance(v, tuple): return f"({_py(v[0], ref)},)" if len(v) == 1 else "(" + ", ".join(_py(x, ref) for x in v) + ")"
  if isinstance(v, DType): return repr(v)
  if isinstance(v, list): return "[" + ", ".join(_py(x, ref) for x in v) + "]"
  if isinstance(v, (set, frozenset)): return f"{type(v).__name__}([{', '.join(sorted(_py(x, ref) for x in v))}])"
  if isinstance(v, dict): return "{" + ", ".join(f"{_py(k, ref)}: {_py(x, ref)}" for k,x in v.items()) + "}"
  if isinstance(v, Enum): return f"{_name(type(v))}.{v.name}"
  if isinstance(v, (int, float)): return f"{_name(type(v))}({repr(int(v) if isinstance(v, int) else float(v))})"   # ConstFloat
  if isinstance(v, type) or callable(v) and hasattr(v, "__qualname__") and hasattr(v, "__module__"): return _name(v)
  if isinstance(v, functools.partial):
    args = [_py(v.func, ref), *(_py(x, ref) for x in v.args), *(f"{k}={_py(x, ref)}" for k,x in v.keywords.items())]
    return f"functools.partial({', '.join(args)})"
  if is_dataclass(v):
    kw = [f"{f.name}={_py(x, ref)}" for f in fields(v) if f.init and not _same(x:=getattr(v, f.name), f.default)]
    return f"{_name(type(v))}({', '.join(kw)})"
  return repr(v)

def _same(a, b) -> bool:
  """b rebuilds a: identical, or equal and of the same type, comparing structure where equality is identity"""
  if a is b: return True
  if type(a) is not type(b) or isinstance(a, UOp): return False
  if isinstance(a, functools.partial): return _same(a.func, b.func) and _same(a.args, b.args) and _same(a.keywords, b.keywords)
  if isinstance(a, (tuple, list)): return len(a) == len(b) and all(map(_same, a, b))
  if isinstance(a, dict): return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
  if is_dataclass(a) and type(a).__eq__ is object.__eq__: return all(_same(getattr(a, f.name), getattr(b, f.name)) for f in fields(a))
  try: return bool(a == b)
  except Exception: return False

def _eval(expr:str, refs) -> Any:
  """evaluate a `python expression`. NOTE: this is eval, only parse UIR you trust"""
  ns = _Names(inf=math.inf, nan=math.nan, unpickle=lambda b: pickle.loads(base64.b64decode(b)), __refs__=refs)
  if (m:=re.match(r"([\w.]+):(?!=)", expr)):   # module:expr evaluates in that module
    ns.update(vars(importlib.import_module(m[1])), __refs__=refs)
    expr = expr[m.end():]
  # a %ref outside a string literal is a UOp
  expr = re.sub(r"""('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")|%(\w+)""", lambda x: x[1] or f"__refs__['%{x[2]}']", expr)
  return eval(expr, {}, ns)  # pylint: disable=eval-used

def _pyobj(v, ref:Callable[[UOp], str|None], refs) -> str:
  """the checked `python expression` for v"""
  py = _py(v, ref).replace("\n", " ").replace("`", "\\x60")   # backticks and newlines are only inside string literals
  mod = type(v).__module__
  for expr in ((f"{mod}:{py}",) if mod.startswith(_per_target) else (py, f"{mod}:{py}")):
    try:
      if _same(v, _eval(expr, refs)): return f"`{expr}`"
    except Exception: pass
  try: return f"`unpickle({base64.b64encode(pickle.dumps(v)).decode()!r})`"
  except Exception: return "`" + repr(v).replace("\n", " ").replace("`", "\\x60") + "`"   # nothing rebuilds it, show the repr

# ***** render *****

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

# keys holding an enum or dtype, rendered by name: dtype=f32 addrspace=REG axis=LOOP op=add
_named_keys = {"dtype", "addrspace", "axis", "op"}
_keywords = {"true", "false", "inf", "nan", "invalid"}

def _val(v, named=False, ref:Callable[[UOp], str|None]=lambda _: None, refs=None) -> str:
  """an arg value: literals bare, tuples [a b], strings bare if one word else quoted, UOps in the graph as %id, other python objects as `expr`"""
  if named and isinstance(v, (Enum, DType)): return v.name.lower() if isinstance(v, Ops) else v.name
  if isinstance(v, bool): return str(v).lower()
  if type(v) in (int, float): return repr(v)   # exact types: enums and ConstFloat are python objects
  if isinstance(v, tuple): return "[" + " ".join(_val(x, ref=ref, refs=refs) for x in v) + "]"
  if isinstance(v, UOp) and (r:=ref(v)) is not None: return r
  if isinstance(v, str): return v if re.fullmatch(r"[A-Za-z_][\w:.,+\-]*", v) and v not in _keywords else repr(v)
  if isinstance(v, Target): return _val(repr(v))
  return _pyobj(v, ref, refs)

# single-value args get a name, these ops have a dataclass arg rendered as key=value per field. runtime objects aren't rendered
_arg_key = {Ops.CAST:"dtype", Ops.BITCAST:"dtype", Ops.COPY:"device", Ops.GETADDR:"device", Ops.SPECIAL:"name", Ops.UNSHARD:"axes",
            Ops.MSELECT:"index"}
_skip_fields = {"buffer", "grad_fxn"}

@functools.cache
def _struct_args() -> dict[Ops, type]:
  from tinygrad.schedule.indexing import BufferizeOpts  # schedule imports codegen, which imports this
  return {Ops.PARAM: ParamArg, Ops.REFPARAM: ParamArg, Ops.BUFFER: ParamArg, Ops.ALLOC: ParamArg, Ops.SINK: KernelInfo,
          Ops.CALL: CallInfo, Ops.CUSTOM_FUNCTION: CustomFunction, Ops.STAGE: BufferizeOpts, Ops.PROGRAM: ProgramInfo}

def _fields_str(a, always=(), ref:Callable[[UOp], str|None]=lambda _: None, refs=None) -> str:
  return " ".join(f"{f.name}={_val(v, f.name in _named_keys, ref, refs)}" for f in fields(a)
                  if f.name not in _skip_fields and (not _same(v:=getattr(a, f.name), f.default) or f.name in always))

def _render_arg(x:UOp, ref:Callable[[UOp], str|None]=lambda _: None, refs=None) -> str:
  """args are key=value, written after the srcs"""
  if (a:=x.arg) is None: return ""
  match x.op:
    case Ops.CONST: return _const_str(x)
    case Ops.RANGE:   # like UOp.range, axis defaults to WEAK. flatten_range merges slots: slot=[1 2]
      return f"slot={_val(a[1] if len(a) == 2 else a[1:])}" + (f" axis={a[0].name}" if a[0] is not AxisType.WEAK else "")
    case Ops.REDUCE: return f"op={_val(a[0], True)}" + (f" pop={a[1]}" if a[1] else "")
    case Ops.ALLREDUCE: return f"op={_val(a[0], True)} device={_val(a[1])}"
  if type(a) is _struct_args().get(x.op):   # name always, so a SINK's KernelInfo() is not None
    return _fields_str(a, always=("name",) if x.op is Ops.SINK else (), ref=ref, refs=refs)
  return f"{(k:=_arg_key.get(x.op, 'arg'))}={_val(a, k in _named_keys, ref, refs)}"

# CONSTs never get lines (inline literals, no %id); concrete all-const STACKs inline into their parent as [a b]
def _inline(u:UOp) -> bool: return u.op is Ops.CONST or (u.op is Ops.STACK and all(s.op is Ops.CONST for s in u.src))

def _arg_uops(a) -> list[UOp]:
  """the UOps inside an arg"""
  if isinstance(a, UOp): return [a]
  if isinstance(a, (tuple, list, set, frozenset)): return [u for x in a for u in _arg_uops(x)]
  if isinstance(a, dict): return _arg_uops(tuple(a.values()))
  if isinstance(a, functools.partial): return _arg_uops((a.args, a.keywords))
  if is_dataclass(a) and not isinstance(a, type): return _arg_uops(tuple(getattr(a, f.name) for f in fields(a) if f.name not in _skip_fields))
  return []

def render_uir(root:UOp|list[UOp]) -> str:
  """one line per UOp: %id = op src src key=value. a UOp in an arg (like a symbolic size) gets its line before the arg's UOp"""
  order: dict[UOp, None] = {}
  def add(r:UOp):
    for u in r.toposort(enter_calls=True):
      if u in order: continue
      for au in _arg_uops(u.arg): add(au)
      order[u] = None
  if isinstance(root, UOp): add(root)
  nodes = [u for u in (order if isinstance(root, UOp) else root) if not _inline(u)]
  table = {u:i for i,u in enumerate(nodes)}
  refs = {f"%{i}":u for i,u in enumerate(nodes)}
  def src_str(u:UOp) -> str:
    if not _inline(u): return f"%{table[u]}"
    return _const_str(u) if u.op is Ops.CONST else "[" + " ".join(src_str(s) for s in u.src) + "]"
  def line(i:int, u:UOp) -> str:
    # a UOp in an arg is a %ref if it's an earlier line
    arg = _render_arg(u, lambda v: f"%{table[v]}" if table.get(v, i) < i else None, refs)
    return " ".join(filter(None, [f"%{i} = {colored(u.op.name.lower(), uops_colors.get(u.op))}", *map(src_str, u.src), arg]))
  return "\n".join(line(i, u) for i,u in enumerate(nodes))

# ***** parse *****

# a token is a comment, an assignment '%name =', a key=, a %ref, a bracket, a quoted string, a `python object`, or a bare word
_tokens = re.compile(r"""#[^\n]*|%\w+\s*=|\w+=|%\w+|[\[\](){}]|'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|`[^`]*`|[^\s\[\](){}#]+""", re.ASCII)
_ops = {o.name.lower(): o for o in Ops}
_dtypes = {v.name: v for v in vars(type(dtypes)).values() if isinstance(v, DType)}
_named_parse: dict[str, Callable[[str], Any]] = {"dtype": lambda v: _dtypes[v], "addrspace": lambda v: AddrSpace[v],
                                                  "axis": lambda v: AxisType[v], "op": lambda v: _ops[v]}

def _lit(t:str) -> Any:
  """literals: true false 0x2a -3 1.5 1e+20 -inf nan. anything else is a bare word"""
  if t in ("true", "false"): return t == "true"
  if re.fullmatch(r"0x[0-9a-fA-F]+", t): return int(t, 16)
  if re.fullmatch(r"-?[0-9]+", t): return int(t)
  if re.fullmatch(r"-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?|-?inf|nan", t): return float(t)
  return t

def _build(name:str, srcs:tuple[UOp, ...], kv:dict[str, Any]) -> UOp:
  """make the UOp for `name srcs key=value`"""
  if (op:=_ops.get(name)) is None: raise ValueError(f"unknown op {name!r}")
  if "target" in kv: kv["target"] = Target.parse(kv["target"])   # a Target is written as its string
  arg: Any = None
  if op is Ops.RANGE: arg = (kv.get("axis", AxisType.WEAK), *(s if isinstance(s:=kv["slot"], tuple) else (s,)))
  elif op is Ops.REDUCE: arg = (kv["op"], kv.get("pop", 0))
  elif op is Ops.ALLREDUCE: arg = (kv["op"], kv["device"])
  elif "arg" in kv and len(kv) == 1: arg = kv["arg"]
  elif op in _struct_args() and (kv or op is Ops.CALL): arg = _struct_args()[op](**kv)   # a CALL always has a CallInfo
  elif len(kv) == 1: arg = next(iter(kv.values()))
  elif kv: raise ValueError(f"{name} takes one arg, got {list(kv)}")
  return UOp(op, srcs, arg)

def parse_uir(text:str) -> UOp:
  """parse UIR text into a UOp graph: render_uir's output, or hand-written UIR like studentgrad's.
  a program is `%name = value` assignments and its root is the last one. a value is `op srcs key=value` (only right after the `=`),
  `(op srcs key=value)`, `[values]` for a STACK, `{assignments}` for the last of a scoped block, a %name, or a literal.
  NOTE: a `python object` is eval'd, only parse UIR you trust"""
  toks = deque(t for t in _tokens.findall(ansistrip(text)) if not t.startswith("#"))   # op names may carry ANSI color from render_uir
  def is_assign(t:str) -> bool: return t[0] == "%" and t[-1] == "="
  def at_end() -> bool: return not toks or toks[0] in (")", "]", "}") or is_assign(toks[0])
  def pop(what:str) -> str:
    if not toks: raise ValueError(f"expected {what}, got end of input")
    return toks.popleft()
  def ref(names:dict[str, UOp], t:str) -> UOp:
    if t not in names: raise ValueError(f"{t} is not defined")
    return names[t]
  def attr(names:dict[str, UOp], k:str) -> Any:
    if (t:=pop(f"a value for {k}=")) == "[":
      ret = []
      while toks and toks[0] != "]": ret.append(attr(names, k))
      pop("']'")
      return tuple(ret)
    if t[0] == "`": return _eval(t[1:-1], names)
    if t[0] in "'\"": return ast.literal_eval(t)
    return ref(names, t) if t[0] == "%" else _named_parse.get(k, _lit)(t)
  def expr(names:dict[str, UOp], name:str, close:str|None) -> UOp:
    srcs: list[UOp] = []
    while not at_end() and not toks[0].endswith("="): srcs.append(value(names))
    kv: dict[str, Any] = {}
    while not at_end():
      if not (k:=pop("key=")).endswith("="): raise ValueError(f"srcs must come before key=value args in {name}, got {k!r}")
      kv[k[:-1]] = attr(names, k[:-1])
    if close is not None and pop(repr(close)) != close: raise ValueError(f"expected {close!r} to end {name}")
    return _build(name, tuple(srcs), kv)
  def value(names:dict[str, UOp], bare=False) -> UOp:
    t = pop("a value")
    if t == "{": return scope(dict(names), "}")
    if t == "(": return expr(names, pop("an op"), ")")
    if t == "[":
      srcs = []
      while toks and toks[0] != "]": srcs.append(value(names))
      pop("']'")
      return UOp(Ops.STACK, tuple(srcs))
    if t[0] == "%": return ref(names, t)
    if t == "invalid": return UOp(Ops.CONST, arg=Invalid)
    if isinstance(v:=_lit(t), str):
      if bare and re.fullmatch(r"[a-z_0-9]+", t): return expr(names, t, None)   # an assignment's value can be a bare op
      raise ValueError(f"{t!r} is not a %name or a literal" + (f", write ({t} ...)" if t in _ops else ""))
    return UOp(Ops.CONST, arg=v if isinstance(v, int) else dtypes.weakfloat.const(v))
  def scope(names:dict[str, UOp], close:str|None) -> UOp:
    root: UOp|None = None
    while toks and toks[0] != "}":
      if not is_assign(t:=pop("'%name ='")): raise ValueError(f"expected '%name =', got {t!r}")
      if (name:=t[:-1].rstrip()) in names: raise ValueError(f"{name} is already defined")
      names[name] = root = value(names, bare=True)
    if (toks.popleft() if toks else None) != close: raise ValueError(f"expected {close or 'end of input'!r}")
    if root is None: raise ValueError("expected an assignment")
    return root
  return scope({}, None)
