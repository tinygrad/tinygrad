from typing import Any, Callable
import itertools, inspect, functools, types, hashlib, marshal, sys
from tinygrad.helpers import partition, dedup, Context, diskcache_get, diskcache_put
from tinygrad.dtype import Invalid
from tinygrad.uop.ops import UPat, UOp, Ops, PatternMatcher, graph_rewrite, deconstruct_function, upat_interpret

class UPatCompileError(Exception): pass

def flatands(x:UOp) -> list[UOp]: return [x] if x.op is not Ops.AND else [y for s in x.src for y in flatands(s)]

# **** UPat compiled ****
# This file builds an IR of match predicates and compiles them to Python source.
# Ops used: CUSTOM (format-string predicate over operands), CUSTOMI (inline string fragment),
#           STORE (bind a matched UOp to a name), PYLITERAL (Python literal for CUSTOM operands),
#           AND/OR (clause combininers).

def _get_clause(self:UPat, base:UOp, depth=0, skip_op=False) -> UOp:
  if self.is_any:
    assert len(self.src) == 1
    return UOp(Ops.AND, src=(UOp(Ops.OR, src=tuple(_get_clause(s, base, depth) for s in self.src[0])),))
  # build the and_clause for acceptance
  and_clause:list[UOp] = []
  # skip_op is used when the caller already dispatched on uop.op
  if self.op is not None and not skip_op:
    if len(self.op) > 1:
      and_clause.append(UOp.custom_function("{0}.op in {1}", fmt=True).call(base, UOp(Ops.PYLITERAL, arg=frozenset(self.op))))
    else: and_clause.append(UOp.custom_function("{0}.op is {1}", fmt=True).call(base, UOp(Ops.PYLITERAL, arg=self.op[0])))
  if self.arg is not None:
    # NOTE: Invalid is a singleton whose == is identity, `is` doesn't call into InvalidType.__eq__
    if self.arg is Invalid: and_clause.append(UOp.custom_function("{0}.arg is {1}", fmt=True).call(base, UOp(Ops.PYLITERAL, arg=self.arg)))
    elif isinstance(self.arg, int): and_clause.append(UOp.custom_function("{0}.arg == "+str(int(self.arg)), fmt=True).call(base))
    else: and_clause.append(UOp.custom_function("{0}.arg == {1}", fmt=True).call(base, UOp(Ops.PYLITERAL, arg=self.arg)))
  if self.strict_length or self.required_len > 0:
    and_clause.append(UOp.custom_function("len({0}.src)"+(" == " if self.strict_length else " >= ")+str(self.required_len), fmt=True).call(base))
  if self.name is not None: and_clause.append(UOp(Ops.STORE, src=(UOp.custom_function(self.name, fmt=True).call(inline=True), base)))
  if self.match_dtype is not None:
    if len(self.match_dtype) > 1:
      and_clause.append(UOp.custom_function("{0}.dtype in {1}", fmt=True).call(base, UOp(Ops.PYLITERAL, arg=frozenset(self.match_dtype))))
    else:
      and_clause.append(UOp.custom_function("{0}.dtype is {1}", fmt=True).call(base, UOp(Ops.PYLITERAL, arg=self.match_dtype[0])))
  if self.src is not None:
    # single match
    if len(self.src) == 1 and isinstance(self.src[0], tuple):
      and_clause += [_get_clause(s, base.index(i), depth) for i,s in enumerate(self.src[0])]
    # repeat match
    elif len(self.src) == 1 and isinstance(self.src[0], itertools.repeat):
      it = UOp.custom_function(f"ituop{depth}", fmt=True).call(inline=True)
      match = _get_clause(next(self.src[0]), it, depth+1)
      # lift named binds out of the repeat: they bind the first src, and every element must be identical to it
      stores, pred = partition(flatands(match), lambda x: x.op is Ops.STORE)
      for st in stores:
        first = st.src[1].substitute({it: base.index(0)})
        pred.append(UOp.custom_function("{0} is {1}", fmt=True).call(st.src[1], first))
        and_clause.append(UOp(Ops.STORE, src=(st.src[0], first)))
      if len(stores): and_clause.append(UOp.custom_function("len({0}.src) != 0", fmt=True).call(base))
      and_clause.append(UOp.custom_function("all([{0} for {1} in {2}.src])", fmt=True).call(UOp(Ops.AND, src=tuple(pred)), it, base))
    # multi match (fork)
    elif len(self.src) > 1 and all(isinstance(x, tuple) for x in self.src):
      fork_cond = [UOp(Ops.AND, src=tuple([_get_clause(s, base.index(i), depth) for i,s in enumerate(ss)])) for ss in self.src]
      and_clause.append(UOp(Ops.OR, src=tuple(fork_cond)))
    else: raise RuntimeError("broken")
  return UOp(Ops.AND, src=tuple(and_clause)) if and_clause else UOp.custom_function("True", fmt=True).call(inline=True)

# *** pattern matcher ***

def do_process_and(a:UOp) -> UOp|None:
  found = False
  new_src:list[UOp] = []
  or_clause:list[UOp] = []

  # remove any nested ANDs, extract or clauses
  for x in a.src:
    if x.op is Ops.AND:
      new_src.extend(x.src)
      found = True
    elif x.op is Ops.OR: or_clause.append(x)
    else: new_src.append(x)

  # too big to compile
  if len(or_clause) >= 4: raise UPatCompileError("too big to compile")

  # one or clause max
  if len(or_clause) > 1:
    # need the product of the or clauses
    or_clause = [UOp(Ops.OR, src=tuple([UOp(Ops.AND, src=x) for x in itertools.product(*[x.src for x in or_clause])]))]
    found = True

  # handle stores
  stores, new_src = partition(new_src, lambda x: x.op is Ops.STORE)
  if len(stores):
    if len(or_clause):
      # push stores to the top if we have an or_clause
      assert len(or_clause) == 1 and all(x.op is Ops.AND for x in or_clause[0].src)
      or_clause = [UOp(Ops.OR, src=tuple([x.replace(src=x.src+tuple(stores)) for x in or_clause[0].src]))]
      found = True
    else:
      # check for duplicate stores
      dict_stores: dict[UOp, UOp] = {}
      for store in stores:
        if store.src[0] in dict_stores:
          # duplicate store is an identity compare
          new_src.append(UOp.custom_function("{0} is {1}", fmt=True).call(dict_stores[store.src[0]], store.src[1]))
          found = True
        else:
          dict_stores[store.src[0]] = store.src[1]
      # put the stores back
      for k,v in dict_stores.items(): new_src.append(UOp(Ops.STORE, src=(k,v)))

  # reassemble, if there's any deduping to do, do it
  if len(dretand:=dedup(new_src+or_clause)) != len(new_src)+len(or_clause): found = True
  return UOp(Ops.AND, src=tuple(dretand)) if found else None

# processor
pm_proc = PatternMatcher([(UPat(Ops.AND, name="a"), do_process_and)], compiled=False)

# renderer
# literal names are global (and deduped) so code from many UPats can share one namespace and identical checks render identically
_lit_names: dict[Any, str] = {}
def wrap(ctx, x) -> UOp:
  key: tuple
  try: key = (type(x.arg), x.arg, hash(x.arg))
  except TypeError: key = (type(x.arg), id(x.arg))
  if (ret:=_lit_names.get(key)) is None: _lit_names[key] = ret = f"_l{len(_lit_names)}"
  ctx[ret] = x.arg
  return UOp.custom_function(ret, fmt=True).call(inline=True)

pm_renderer = PatternMatcher([
  (UPat(Ops.PYLITERAL, name="x"), wrap),

  # AND/OR of CUSTOMI fragments becomes a single CUSTOMI
  (UPat(Ops.AND, src=UPat(Ops.CALL), name="x"), lambda x:
   UOp.custom_function("(" + ' and '.join(y.body.arg.name for y in x.src) + ")", fmt=True).call(inline=True)
   if all(y.arg.inline for y in x.src) else None),
  (UPat(Ops.OR, src=UPat(Ops.CALL), name="x"), lambda x:
   UOp.custom_function("(" + ' or '.join(y.body.arg.name for y in x.src) + ")", fmt=True).call(inline=True)
   if all(y.arg.inline for y in x.src) else None),

  (UPat(Ops.CALL, name="x"), lambda x: UOp.custom_function(x.body.arg.name.format(*[y.body.arg.name for y in x.src[1:]]), fmt=True).call(inline=True)
   if not x.arg.inline and all(y.op is Ops.CALL and y.arg.inline for y in x.src[1:]) else None),
  (UPat(Ops.INDEX, src=(UPat(Ops.CALL, name="x"), UPat(Ops.CONST, name="c"))),
   lambda x,c: UOp.custom_function(x.body.arg.name + f".src[{c.val}]", fmt=True).call(inline=True) if x.arg.inline else None)
], compiled=False)

# leaf modes: "single" returns from a standalone matcher, "merged" falls through to the next pattern if the fxn returns uop,
# "merged_loop" is "merged" for patterns with forks: it's wrapped in a `while True:` so `break` exits the pattern
def _call_args(stores:list[tuple[str, str]], has_ctx:bool, params:tuple[str, ...]|None) -> str:
  """the args for calling the pattern fxn. positional (faster) if params is the fxn's positional args, else keywords"""
  named = dict(stores)
  if has_ctx: named["ctx"] = "ctx"
  if params is not None and set(params) == set(named): return ', '.join(named[p] for p in params)
  return ', '.join(f"{k}={v}" for k,v in named.items())

def _positional_params(fxn:Callable) -> tuple[str, ...]|None:
  # the names of the args of fxn if they are all positional (no defaults, *args, **kwargs or keyword only args)
  co = fxn.__code__
  if co.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS) or co.co_kwonlyargcount or fxn.__defaults__: return None
  return co.co_varnames[:co.co_argcount]

def _final_render(x:UOp, has_ctx:bool, depth=1, fxn_name="_fxn", leaf="single", params:tuple[str, ...]|None=None) -> list[str]:
  # if the whole clause collapsed to a single predicate (no binds), rewrap it
  if x.op is Ops.CALL and x.arg.inline: x = UOp(Ops.AND, (x,))
  assert x.op is Ops.AND
  and_pieces, store_pieces = [], []
  or_pieces: list[str] = []
  for s in x.src:
    if s.op is Ops.OR:
      assert len(or_pieces) == 0 and len(s.src) >= 1
      for ss in s.src: or_pieces.extend(_final_render(ss, has_ctx, depth+1, fxn_name, leaf, params))
    elif s.op is Ops.STORE:
      assert all(y.op is Ops.CALL and y.arg.inline for y in s.src)
      store_pieces.append((s.src[0].body.arg.name, s.src[1].body.arg.name))
    elif s.op is Ops.CALL and s.arg.inline: and_pieces.append(s.body.arg.name)
    else: raise UPatCompileError(f"can't compile this {s}")
  # if we have an or, render it
  if len(or_pieces):
    assert len(store_pieces) == 0
    and_clause = ' and '.join(and_pieces)
    return [f"{'  '*depth}if {and_clause if len(and_clause) else 'True'}:"] + or_pieces
  # if we don't, this is a final return
  store_clause = _call_args(store_pieces, has_ctx, params)
  and_clause = ' and '.join(and_pieces + [f"(_ret:={fxn_name}({store_clause})) is not None"])
  if leaf == "single": return [f"{'  '*depth}if {and_clause}: return _ret"]
  if leaf == "merged": return [f"{'  '*depth}if {and_clause} and _ret is not uop: return _ret"]
  return [f"{'  '*depth}if {and_clause}:", f"{'  '*depth}  if _ret is not uop: return _ret", f"{'  '*depth}  break"]

def _get_ir(self:UPat, skip_op=False) -> tuple[UOp, dict[str, Any]]:
  ret = _get_clause(self, UOp.custom_function("uop", fmt=True).call(inline=True), skip_op=skip_op)
  # TODO: this should be tracked in a "system" rewrite, not untracked or tracked with kernel
  with Context(TRACK_MATCH_STATS=0):
    ret = graph_rewrite(ret, pm_proc, name="process UPat")
    dyn_lookup: dict[str, Any] = {}
    out = graph_rewrite(ret, pm_renderer, ctx=dyn_lookup, name="compile UPat")
  return out, dyn_lookup

def _get_code(self:UPat, has_ctx:bool):
  try:
    out, dyn_lookup = _get_ir(self)
    rendered = _final_render(out, has_ctx)
  except UPatCompileError:
    #print("FAILED", self, self.location)
    return None
  return '\n'.join([f"# match for {self.location}", "def compiled_match(uop, ctx):"] + rendered + ["  return None"]), dyn_lookup

@functools.cache
def upat_compile(self:UPat, fxn) -> Callable:
  real_fxn = types.FunctionType(*deconstruct_function(fxn))
  code = _get_code(self, 'ctx' in inspect.signature(real_fxn).parameters)
  if code is None: raise UPatCompileError(f"can't compile pattern defined at {self.location[0]}:{self.location[1]}")
  code_str, dyn_lookup = code
  globs = dyn_lookup.copy()
  globs["_fxn"] = real_fxn
  namespace: dict = {}
  exec(code_str, globs, namespace)  # pylint: disable=W0122
  return namespace["compiled_match"]

def compile_cached(code:str, filename:str) -> types.CodeType:
  """compile Python source, with the code object cached on disk (like compiled kernels)"""
  key = hashlib.sha256(f"{sys.version}\n{filename}\n{code}".encode()).hexdigest()
  if (ret:=diskcache_get("python_code", key)) is not None: return marshal.loads(ret)
  co = compile(code, filename, "exec")
  diskcache_put("python_code", key, marshal.dumps(co))
  return co

# **** whole PatternMatcher compiled ****
# All the patterns for one Ops are rendered into a single function, in order. This removes the per pattern call, the (redundant) op check
# and the early_reject check when it's implied by the src checks.

_fxn_cnt = itertools.count()

@functools.cache
def _pattern_code(p:UPat, fxn) -> tuple[tuple[str, ...], str, list[str], dict[str, Any]]:
  """returns (conds, leaf, fork_lines, globals) to try pattern p on `uop`. Assumes uop.op in p.op.
  if fork_lines is empty, the pattern is `if all(conds) and leaf: return _ret`, else it's `if all(conds): while True: fork_lines`"""
  real_fxn = types.FunctionType(*deconstruct_function(fxn))
  globs: dict[str, Any] = {(fname:=f"_f{next(_fxn_cnt)}"): real_fxn}
  has_ctx = 'ctx' in inspect.signature(real_fxn).parameters
  try:
    out, dyn_lookup = _get_ir(p, skip_op=True)
    globs.update(dyn_lookup)
  except UPatCompileError:
    # fall back to the interpreted matcher
    globs[fname] = upat_interpret(p, fxn)
    return (), f"(_ret:={fname}(uop, ctx)) is not None", [], globs
  if out.op is Ops.CALL and out.arg.inline: out = UOp(Ops.AND, (out,))
  conds = tuple(s.body.arg.name for s in out.src if s.op is Ops.CALL)
  if any(s.op is Ops.OR for s in out.src):
    fork_lines = [l for s in out.src if s.op is Ops.OR for ss in s.src
                  for l in _final_render(ss, has_ctx, 0, fname, "merged_loop", _positional_params(real_fxn))]
    return conds, "", fork_lines + ["break"], globs
  stores = _call_args([(s.src[0].body.arg.name, s.src[1].body.arg.name) for s in out.src if s.op is Ops.STORE], has_ctx,
                      _positional_params(real_fxn))
  return conds, f"(_ret:={fname}({stores})) is not None", [], globs

def _early_reject_implied(p:UPat) -> bool:
  # the default early_reject is the set of single ops in the src, if the src is a tuple (or permutations) these are all checked by the matcher
  return p.custom_early_reject is None and p.src is not None and all(isinstance(x, tuple) for x in p.src)

def _render_pats(pats:list[tuple[tuple[str, ...], str, list[str], str]], depth:int) -> list[str]:
  # consecutive patterns sharing a leading condition are nested under one `if`. order is preserved, so first match still wins
  ret: list[str] = []
  ind = "  "*depth
  i = 0
  while i < len(pats):
    conds, leaf, fork_lines, comment = pats[i]
    j = i+1
    if len(conds): j = next((k for k in range(i+1, len(pats)) if not pats[k][0] or pats[k][0][0] != conds[0]), len(pats))
    if j-i > 1:
      ret.append(f"{ind}if {conds[0]}:")
      ret.extend(_render_pats([(c[1:], l, f, m) for c,l,f,m in pats[i:j]], depth+1))
    else:
      ret.append(f"{ind}# {comment}")
      if len(fork_lines):
        if len(conds): ret.append(f"{ind}if {' and '.join(conds)}:")
        ret.append(f"{ind}{'  ' if len(conds) else ''}while True:")
        ret.extend(f"{ind}{'    ' if len(conds) else '  '}{l}" for l in fork_lines)
      else: ret.append(f"{ind}if {' and '.join(conds+(leaf,))} and _ret is not uop: return _ret")
    i = j
  return ret

@functools.cache
def pm_compile_op(op:Ops, pats:tuple[tuple[UPat, Callable], ...]) -> Callable:
  globs: dict[str, Any] = {}
  rpats: list[tuple[tuple[str, ...], str, list[str], str]] = []
  need_ler = False
  for i,(p,fxn) in enumerate(pats):
    conds, leaf, fork_lines, pglobs = _pattern_code(p, fxn)
    globs.update(pglobs)
    if len(p.early_reject) and not _early_reject_implied(p):
      need_ler = True
      globs[ername:=f"_er{i}"] = frozenset(p.early_reject)
      conds = (f"{ername}.issubset(_ler)",) + conds
    rpats.append((conds, leaf, fork_lines, f"{p.location[0].split('/')[-1]}:{p.location[1]}"))
  pre = ["if (_ler:=uop.__dict__.get('_src_ops')) is None: uop.__dict__['_src_ops'] = _ler = {u.op for u in uop.src}"] if need_ler else []
  code = '\n'.join([f"def compiled_pm_{op.name}(uop, ctx):"] + ["  "+l for l in pre] + _render_pats(rpats, 1) + ["  return None"])
  namespace: dict = {}
  exec(compile_cached(code, f"<compiled_pm_{op.name}>"), globs, namespace)  # pylint: disable=W0122
  fxn = namespace[f"compiled_pm_{op.name}"]
  fxn.__source__ = code
  return fxn

# **** compiled graph_rewrite ****
# The unified_rewrite loop is generated for each mode (top down pm, bottom up bpm, compiled matchers, enter_calls), so the hot loop
# has no branches on the mode and calls the compiled per-op matchers directly.

def _matcher_call(name:str, fd:bool, x:str) -> str:
  return f"(_f({x}, ctx) if (_f:={name}_fd_get({x}.op)) is not None else None)" if fd else f"{name}_rewrite({x}, ctx)"

# the loops over srcs are unrolled for the common small number of srcs
UNROLL = 3

def _push_srcs(ind:str, srcs:str) -> list[str]:
  # push the srcs not on the stack (last one first), and before the first push, the node itself to come back to after its srcs
  def push(x:str, i:str) -> list[str]:
    return [f"{i}if {x} not in on_stack:",
            f"{i}  if not pushed:",
            f"{i}    stack.append((n, 1, new_n))",
            f"{i}    pushed = True",
            f"{i}  stack.append(({x}, 0, {x}))",
            f"{i}  on_stack.add({x})"]
  L = [f"{ind}pushed = False", f"{ind}ps = {srcs}", f"{ind}ns = len(ps)"]
  srcs = "ps"
  for k in range(1, UNROLL+1):
    L.append(f"{ind}{'if' if k == 1 else 'elif'} ns == {k}:")
    if k == 1: L.append(f"{ind}  s0 = {srcs}[0]")
    else: L.append(f"{ind}  {', '.join(f's{i}' for i in range(k))} = {srcs}")
    for i in reversed(range(k)): L += push(f"s{i}", ind+"  ")
  L += [f"{ind}elif ns:", f"{ind}  for x in reversed({srcs}):"] + push("x", ind+"    ")
  return L

def _rebuild_srcs(ind:str, enter_calls:bool) -> list[str]:
  # sets changed and new_src from replace, or waits for the first src that isn't rewritten yet
  def wait(x:str, i:str) -> list[str]: return [f"{i}waitlist.setdefault({x}, []).append((n, 1, new_n))", f"{i}continue"]
  L = [f"{ind}src = new_n.src", f"{ind}ns = len(src)"]
  # NOTE: a CALL body is not rewritten, unless enter_calls
  L.append(f"{ind}if ns == 0: changed = False")
  for k in range(1, UNROLL+1):
    L.append(f"{ind}elif ns == {k}{'' if enter_calls else ' and new_n.op is not CALL'}:")
    L.append(f"{ind}  s0 = src[0]" if k == 1 else f"{ind}  {', '.join(f's{i}' for i in range(k))} = src")
    for i in range(k): L += [f"{ind}  if (r{i}:=replace_get(s{i})) is None:"] + wait(f"s{i}", ind+"    ")
    L += [f"{ind}  if changed:={' or '.join(f'r{i} is not s{i}' for i in range(k))}: new_src = ({', '.join(f'r{i}' for i in range(k))},)"]
  L += [f"{ind}else:",
        f"{ind}  # NOTE: replace never has None values",
        f"{ind}  new_src = tuple(map(replace_get, src))" if enter_calls else
        f"{ind}  new_src = (src[0],)+tuple(map(replace_get, src[1:])) if new_n.op is CALL else tuple(map(replace_get, src))",
        f"{ind}  if None in new_src:"] + wait("src[new_src.index(None)]", ind+"    ") + [f"{ind}  changed = not (new_src == src)"]
  return L

def _finalize(ind:str, v:str) -> list[str]: return [f"{ind}replace[n] = {v}", f"{ind}if n in waitlist: stack.extend(waitlist.pop(n))"]

_rewrite_sources: dict = {}
@functools.cache
def compile_unified_rewrite(has_pm:bool, pm_fd:bool, has_bpm:bool, bpm_fd:bool, enter_calls:bool) -> Callable:
  L = ["def unified_rewrite(self, root):",
       "  replace, ctx = self.replace, self.ctx",
       "  replace_get = replace.get",
       "  limit = REWRITE_STACK_LIMIT.value"]
  if has_pm: L.append("  pm_fd_get = self.pm.fdict.get" if pm_fd else "  pm_rewrite = self.pm.rewrite")
  if has_bpm:
    L += ["  bpm_cache = self.bpm_cache", "  bpm_cache_get = bpm_cache.get"]
    L.append("  bpm_fd_get = self.bpm.fdict.get" if bpm_fd else "  bpm_rewrite = self.bpm.rewrite")
  L += ["  stack = [(root, 0, root)]",
        "  on_stack = {root}  # all UOps either on the stack or in replace, i.e. dont have to be placed again",
        "  waitlist = {}  # UOps waiting on a dependency to be in replace",
        "  while stack:",
        "    if len(stack) > limit: raise RuntimeError('infinite loop in graph_rewrite (stack too big)')",
        "    n, stage, new_n = stack.pop()",
        "    if n in replace: continue  # skip any nodes we have seen",
        "    if stage == 0:"]
  if has_bpm:
    # apply rewrite rules until a fixed point is reached. may return `uop` itself if PatternMatcher doesn't match
    L += ["      test_n = n",
          "      seen = set()",
          "      try:",
          "        while test_n is not None:",
          "          if test_n in seen: raise RuntimeError('infinite loop in fixed_point_rewrite')",
          "          seen.add(test_n)",
          "          if (nxt:=bpm_cache_get(test_n, SENTINEL)) is SENTINEL: nxt = bpm_cache[test_n] = "+_matcher_call("bpm", bpm_fd, "test_n"),
          "          new_n, test_n = test_n, nxt",
          "      except BottomUpGate:",
          "        # if the bpm matching raised a gate, we are done with this node and dont continue down the srcs"] + \
         _finalize("        ", "test_n") + ["        continue"]
  # NOTE: CALLs are handled as a special case: their bodies are not included in the graph_rewrite, rewrites that need them pass enter_calls
  L += _push_srcs("      ", "new_n.src" if enter_calls else "(new_n.src[1:] if new_n.op is CALL else new_n.src)")
  L += ["      # if no srcs were pushed, stage 1 is the next thing popped, so do it now",
        "      if pushed: continue",
        "      stage = 1",
        "    if stage == 1:"]
  L += _rebuild_srcs("      ", enter_calls)
  L += ["      # in stage 1, once all srcs are rewritten, rebuild (if changed) or run top-down rewrite",
        "      if not changed:"]
  if has_pm:
    L += ["        # if top down, do the rewrite. if no rewrite, we are done rewriting this node so we add it to the dict",
          "        if (new_src_n:="+_matcher_call("pm", pm_fd, "new_n")+") is None:"] + _finalize("          ", "new_n") + ["          continue"]
  else: L += _finalize("        ", "new_n") + ["        continue"]
  L += ["      else:",
        "        # if srcs changed from rewrites, construct a new UOp with the new srcs",
        "        new_src_n = UOp(new_n.op, new_src, new_n.arg)",
        "      # if new_src_n is already rewritten, link it now. this is what stage 2 does right after skipping new_src_n",
        "      if (replaced_new_n:=replace_get(new_src_n)) is not None:"] + _finalize("        ", "replaced_new_n") + ["        continue",
        "      # trigger a rewrite of new_src_n, then after that rewrite is done, link it back to n",
        "      stack.append((n, 2, new_src_n))",
        "      stack.append((new_src_n, 0, new_src_n))",
        "    else:",
        "      # in stage 2, we link the result of new_n to the result of n",
        "      if (replaced_new_n:=replace_get(new_n)) is None:",
        "        # not ready: register in waitlist instead of spinning",
        "        waitlist.setdefault(new_n, []).append((n, 2, new_n))",
        "      else:"] + _finalize("        ", "replaced_new_n") + [
        "  if root not in replace: self.stalled(waitlist)",
        "  return replace[root]"]
  from tinygrad.uop.ops import REWRITE_STACK_LIMIT, SENTINEL, BottomUpGate
  globs = {"Ops":Ops, "CALL":Ops.CALL, "UOp":UOp, "REWRITE_STACK_LIMIT":REWRITE_STACK_LIMIT, "SENTINEL":SENTINEL, "BottomUpGate":BottomUpGate}
  code = '\n'.join(L)
  _rewrite_sources[(has_pm, pm_fd, has_bpm, bpm_fd, enter_calls)] = code
  namespace: dict = {}
  exec(compile(code, f"<unified_rewrite pm={has_pm}{'fd' if pm_fd else ''} bpm={has_bpm}{'fd' if bpm_fd else ''} calls={enter_calls}>", "exec"),
       globs, namespace)  # pylint: disable=W0122
  return namespace["unified_rewrite"]

# **** compiled order free graph_rewrite ****
# For rewrites where the result of a node only depends on the node (no ctx side effects), the nodes are processed in toposort order
# and the (usually small) new subgraphs created by rewrites are rewritten recursively.

def _topo_srcs(ind:str, n:str, enter_calls:bool) -> list[str]:
  return [f"{ind}src = {n}.src" if enter_calls else f"{ind}src = {n}.src[1:] if {n}.op is CALL else {n}.src", f"{ind}ns = len(src)"]

@functools.cache
def compile_topo_rewrite(pm_fd:bool, enter_calls:bool, memo:str|None=None) -> Callable:
  # memo: the result of rewriting a node is stored on the node with this name (None means unchanged), and reused by later rewrites
  match = _matcher_call("pm", pm_fd, "n")
  # rebuild n from its rewritten srcs (all in replace) into new_src/changed, the srcs are unrolled
  def rebuild(ind:str, get:str) -> list[str]:
    L = _topo_srcs(ind, "n", enter_calls) + [f"{ind}if ns == 0: changed = False"]
    for k in range(1, UNROLL+1):
      L.append(f"{ind}elif ns == {k}:")
      L.append(f"{ind}  s0 = src[0]" if k == 1 else f"{ind}  {', '.join(f's{i}' for i in range(k))} = src")
      L += [f"{ind}  r{i} = {get.format(f's{i}')}" for i in range(k)]
      L.append(f"{ind}  if changed:={' or '.join(f'r{i} is not s{i}' for i in range(k))}: new_src = ({', '.join(f'r{i}' for i in range(k))},)")
    L += [f"{ind}else:", f"{ind}  new_src = tuple([{get.format('x')} for x in src])", f"{ind}  changed = not (new_src == src)"]
    if not enter_calls: L.append(f"{ind}if changed and n.op is CALL: new_src = (n.src[0],)+new_src")
    return L
  def process(ind:str, get:str, ret:str) -> list[str]:
    # rewrite n (whose srcs are rewritten), the result is in replace[n]
    return rebuild(ind, get) + [
      f"{ind}if not changed:",
      f"{ind}  if (nr:={match}) is None:",
      f"{ind}    replace[n] = n",
      f"{ind}    {ret}",
      f"{ind}else: nr = UOp(n.op, new_src, n.arg)",
      f"{ind}replace[n] = r = rw(nr)"]
  L = ["def topo_rewrite(self, root):",
       "  replace, ctx = self.replace, self.ctx",
       "  replace_get = replace.get",
       "  pm_fd_get = self.pm.fdict.get" if pm_fd else "  pm_rewrite = self.pm.rewrite",
       "  def rw(n):",
       "    # recursive rewrite for the (usually small) new subgraphs created by rewrites",
       "    if (r:=replace_get(n)) is not None: return r"] + \
      ([f"    if (r:=n.__dict__.get('{memo}', SENTINEL)) is not SENTINEL:", "      replace[n] = r = n if r is None else r", "      return r"]
       if memo else []) + \
      process("    ", "rw({})", "return n") + [
       "    return r",
       "  try:",
       "    # post order DFS (same order as toposort), each node is processed once all its srcs are in replace",
       "    stack = [(root, False)]" if memo else "    stack = [root]",
       "    while stack:",
       "      n, visited = stack.pop()" if memo else "      n = stack.pop()",
       "      if n in replace: continue"] + \
      (["      if not visited and (r:=n.__dict__.get('{memo}', SENTINEL)) is not SENTINEL:".replace("{memo}", memo),
        "        replace[n] = n if r is None else r", "        continue"] if memo else [])
  def entry(x:str, visited:str) -> str: return f"({x}, {visited})" if memo else x
  # the srcs are looked up once: if they are all rewritten, n is processed now (it would be popped next), else push the missing ones
  L += _topo_srcs("      ", "n", enter_calls) + ["      if ns == 0: changed = False"]
  for k in range(1, UNROLL+1):
    L.append(f"      elif ns == {k}:")
    L.append("        s0 = src[0]" if k == 1 else f"        {', '.join(f's{i}' for i in range(k))} = src")
    L += [f"        r{i} = replace_get(s{i})" for i in range(k)]
    L += [f"        if {' or '.join(f'r{i} is None' for i in range(k))}:", "          stack.append("+entry("n", "True")+")"]
    for i in reversed(range(k)): L.append(f"          if r{i} is None: stack.append("+entry(f"s{i}", "False")+")")
    L.append("          continue")
    L.append(f"        if changed:={' or '.join(f'r{i} is not s{i}' for i in range(k))}: new_src = ({', '.join(f'r{i}' for i in range(k))},)")
  L += ["      else:",
        "        new_src = tuple(map(replace_get, src))",
        "        if None in new_src:",
        "          stack.append("+entry("n", "True")+")",
        "          for x in reversed(src):",
        "            if x not in replace: stack.append("+entry("x", "False")+")",
        "          continue",
        "        changed = not (new_src == src)"]
  if not enter_calls: L.append("      if changed and n.op is CALL: new_src = (n.src[0],)+new_src")
  # NOTE: the results already in replace are final, so on deep (or infinite) recursion the general rewrite finishes the job
  L += ["      try:",
        "        if not changed:",
        f"          if (nr:={match}) is None:",
        "            replace[n] = n",
        "            continue",
        "        else: nr = UOp(n.op, new_src, n.arg)",
        "        replace[n] = rw(nr)",
        "      except RecursionError: replace[n] = self.unified_rewrite(nr)",
        "  finally:",
        "    # NOTE: rw is a recursive closure, this breaks the reference cycle so the refcounter can free everything",
        "    rw = None"]

  # NOTE: a result never contains the node it's the result of, so this doesn't make reference cycles
  if memo: L += ["  for k,v in replace.items():", f"    if '{memo}' not in (d:=k.__dict__): d['{memo}'] = None if v is k else v"]
  L += ["  return replace[root]"]
  from tinygrad.uop.ops import SENTINEL
  globs = {"CALL":Ops.CALL, "UOp":UOp, "SENTINEL":SENTINEL}
  code = '\n'.join(L)
  _rewrite_sources[("topo", pm_fd, enter_calls, memo)] = code
  namespace: dict = {}
  exec(compile_cached(code, f"<topo_rewrite pm={'fd' if pm_fd else ''} calls={enter_calls} memo={memo}>"), globs, namespace)  # pylint: disable=W0122
  return namespace["topo_rewrite"]

@functools.cache
def compile_bottom_up_rewrite(bpm_fd:bool, enter_calls:bool, inline:str|None=None) -> Callable:
  """order free bottom up rewrite, same semantics as unified_rewrite with only a bpm. inline is the bpm as an expression of x and ctx"""
  def bpm_call(x:str, ind:str) -> list[str]:
    # the cached bpm rewrite of x into nxt
    if inline is not None: return [f"{ind}nxt = {inline.format(x=x)}"]
    return [f"{ind}if (nxt:=bpm_cache_get({x}, SENTINEL)) is SENTINEL: nxt = bpm_cache[{x}] = "+_matcher_call("bpm", bpm_fd, x)]
  L = ["def bottom_up_rewrite(self, root):",
       "  replace, ctx, bpm_cache = self.replace, self.ctx, self.bpm_cache",
       "  replace_get, bpm_cache_get = replace.get, bpm_cache.get",
       "  bpm_fd_get = self.bpm.fdict.get" if bpm_fd else "  bpm_rewrite = self.bpm.rewrite",
       "  def rw(n):",
       "    if (r:=replace_get(n)) is not None: return r",
       "    # apply rewrite rules until a fixed point is reached",
       "    m = n",
       "    try:"] + bpm_call("n", "      ") + [
       "      if nxt is not None:",
       "        seen = {n}",
       "        while nxt is not None:",
       "          if nxt in seen: raise RuntimeError('infinite loop in fixed_point_rewrite')",
       "          seen.add(nxt)",
       "          m = nxt"] + bpm_call("m", "          ") + [
       "    except BottomUpGate:",
       "      # if the bpm matching raised a gate, we are done with this node and dont continue down the srcs",
       "      replace[n] = m",
       "      return m"]
  L += _topo_srcs("    ", "m", enter_calls) + ["    if ns == 0: changed = False"]
  for k in range(1, UNROLL+1):
    L.append(f"    elif ns == {k}:")
    L.append("      s0 = src[0]" if k == 1 else f"      {', '.join(f's{i}' for i in range(k))} = src")
    L += [f"      r{i} = rw(s{i})" for i in range(k)]
    L.append(f"      if changed:={' or '.join(f'r{i} is not s{i}' for i in range(k))}: new_src = ({', '.join(f'r{i}' for i in range(k))},)")
  L += ["    else:", "      new_src = tuple([rw(x) for x in src])", "      changed = not (new_src == src)"]
  if not enter_calls: L.append("    if changed and m.op is CALL: new_src = (m.src[0],)+new_src")
  L += ["    replace[n] = r = rw(UOp(m.op, new_src, m.arg)) if changed else m",
        "    return r",
        "  # NOTE: the results already in replace are final, so on deep (or infinite) recursion the general rewrite finishes the job",
        "  try: return rw(root)",
        "  except RecursionError: return self.unified_rewrite(root)",
        "  finally:",
        "    # NOTE: rw is a recursive closure, this breaks the reference cycle so the refcounter can free everything",
        "    rw = None"]
  from tinygrad.uop.ops import SENTINEL, BottomUpGate
  globs = {"CALL":Ops.CALL, "UOp":UOp, "SENTINEL":SENTINEL, "BottomUpGate":BottomUpGate}
  code = '\n'.join(L)
  _rewrite_sources[("bottom_up", bpm_fd, enter_calls, inline)] = code
  namespace: dict = {}
  exec(compile_cached(code, f"<bottom_up_rewrite bpm={'fd' if bpm_fd else ''} calls={enter_calls} inline={inline}>"), globs, namespace)  # pylint: disable=W0122
  return namespace["bottom_up_rewrite"]
