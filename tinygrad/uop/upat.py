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
def _call_args(stores:list[tuple[str, str]], has_ctx:bool) -> str: return ', '.join((["ctx=ctx"] if has_ctx else [])+[f"{k}={v}" for k,v in stores])

def _final_render(x:UOp, has_ctx:bool, depth=1, fxn_name="_fxn", leaf="single") -> list[str]:
  # if the whole clause collapsed to a single predicate (no binds), rewrap it
  if x.op is Ops.CALL and x.arg.inline: x = UOp(Ops.AND, (x,))
  assert x.op is Ops.AND
  and_pieces, store_pieces = [], []
  or_pieces: list[str] = []
  for s in x.src:
    if s.op is Ops.OR:
      assert len(or_pieces) == 0 and len(s.src) >= 1
      for ss in s.src: or_pieces.extend(_final_render(ss, has_ctx, depth+1, fxn_name, leaf))
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
  and_clause = ' and '.join(and_pieces + [f"(_ret:={fxn_name}({_call_args(store_pieces, has_ctx)})) is not None"])
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
  diskcache_put("python_code", key, marshal.dumps(co:=compile(code, filename, "exec")))
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
    fork_lines = [l for s in out.src if s.op is Ops.OR for ss in s.src for l in _final_render(ss, has_ctx, 0, fname, "merged_loop")]
    return conds, "", fork_lines + ["break"], globs
  stores = [(s.src[0].body.arg.name, s.src[1].body.arg.name) for s in out.src if s.op is Ops.STORE]
  return conds, f"(_ret:={fname}({_call_args(stores, has_ctx)})) is not None", [], globs

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
