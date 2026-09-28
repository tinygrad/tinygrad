from typing import Any, Callable
import itertools, inspect, functools, types
from tinygrad.helpers import partition, dedup, Context
from tinygrad.uop.ops import UPat, UOp, Ops, PatternMatcher, graph_rewrite, deconstruct_function
from tinygrad.dtype import dtypes

class UPatCompileError(Exception): pass

# **** UPat compiled ****
# This file builds an IR of match predicates and compiles them to Python source.
# Ops used: CUSTOM (format-string predicate over operands), CUSTOMI (inline string fragment),
#           STORE (bind a matched UOp to a name), PYLITERAL (Python literal for CUSTOM operands),
#           AND/OR (clause combininers).

def _get_clause(self:UPat, base:UOp, depth=0) -> UOp:
  if self.is_any:
    assert len(self.src) == 1
    return UOp(Ops.AND, src=(UOp(Ops.OR, src=tuple(_get_clause(s, base, depth) for s in self.src[0])),))
  # build the and_clause for acceptance
  and_clause:list[UOp] = []
  if self.op is not None:
    if len(self.op) > 1:
      and_clause.append(UOp(Ops.CUSTOM, src=(base, UOp(Ops.PYLITERAL, arg=tuple(int(x) for x in self.op))), arg=("{0}.op in {1}", dtypes.void)))
    else: and_clause.append(UOp(Ops.CUSTOM, src=(base,), arg=("{0}.op == "+str(self.op[0].value), dtypes.void)))
  if self.arg is not None:
    if isinstance(self.arg, int): and_clause.append(UOp(Ops.CUSTOM, src=(base,), arg=("{0}.arg == "+str(int(self.arg)), dtypes.void)))
    else: and_clause.append(UOp(Ops.CUSTOM, src=(base, UOp(Ops.PYLITERAL, arg=self.arg)), arg=("{0}.arg == {1}", dtypes.void)))
  if self.strict_length or self.required_len > 0:
    and_clause.append(UOp(Ops.CUSTOM, src=(base,),
      arg=("len({0}.src)"+(" == " if self.strict_length else " >= ")+str(self.required_len), dtypes.void)))
  if self.name is not None: and_clause.append(UOp(Ops.STORE, src=(UOp(Ops.CUSTOMI, arg=(self.name, dtypes.void)), base)))
  if self.match_dtype is not None:
    if len(self.match_dtype) > 1:
      and_clause.append(UOp(Ops.CUSTOM, src=(base, UOp(Ops.PYLITERAL, arg=tuple(self.match_dtype))),
                            arg=("{0}.dtype in {1}", dtypes.void)))
    else:
      and_clause.append(UOp(Ops.CUSTOM, src=(base, UOp(Ops.PYLITERAL, arg=self.match_dtype[0])),
                            arg=("{0}.dtype == {1}", dtypes.void)))
  if self.match_tag is not None:
    if len(self.match_tag) > 1:
      and_clause.append(UOp(Ops.CUSTOM, src=(base, UOp(Ops.PYLITERAL, arg=tuple(self.match_tag))), arg=("{0}.tag in {1}", dtypes.void)))
    else: and_clause.append(UOp(Ops.CUSTOM, src=(base, UOp(Ops.PYLITERAL, arg=self.match_tag[0])), arg=("{0}.tag == {1}", dtypes.void)))
  if self.src is not None:
    # single match
    if len(self.src) == 1 and isinstance(self.src[0], tuple):
      and_clause += [_get_clause(s, base.index(i), depth) for i,s in enumerate(self.src[0])]
    # repeat match
    elif len(self.src) == 1 and isinstance(self.src[0], itertools.repeat):
      it = UOp(Ops.CUSTOMI, arg=(f"ituop{depth}", dtypes.void))
      match = _get_clause(next(self.src[0]), it, depth+1)
      and_clause.append(UOp(Ops.CUSTOM, src=(match, it, base), arg=("all([{0} for {1} in {2}.src])", dtypes.void)))
    # multi match (fork)
    elif len(self.src) > 1 and all(isinstance(x, tuple) for x in self.src):
      fork_cond = [UOp(Ops.AND, src=tuple([_get_clause(s, base.index(i), depth) for i,s in enumerate(ss)])) for ss in self.src]
      and_clause.append(UOp(Ops.OR, src=tuple(fork_cond)))
    else: raise RuntimeError("broken")
  return UOp(Ops.AND, src=tuple(and_clause)) if and_clause else UOp(Ops.CUSTOMI, arg=("True", dtypes.void))

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
          new_src.append(UOp(Ops.CUSTOM, src=(dict_stores[store.src[0]], store.src[1]), arg=("{0} is {1}", dtypes.void)))
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
def wrap(ctx, x) -> UOp:
  ctx[ret:=f"a{len(ctx)}"] = x.arg
  return UOp(Ops.CUSTOMI, arg=(ret, dtypes.void))

pm_renderer = PatternMatcher([
  (UPat(Ops.PYLITERAL, name="x"), wrap),

  # AND/OR of CUSTOMI fragments becomes a single CUSTOMI
  (UPat(Ops.AND, src=UPat(Ops.CUSTOMI), name="x"), lambda x: UOp(Ops.CUSTOMI, arg=("(" + ' and '.join(y.arg[0] for y in x.src) + ")", dtypes.void))),
  (UPat(Ops.OR, src=UPat(Ops.CUSTOMI), name="x"), lambda x: UOp(Ops.CUSTOMI, arg=("(" + ' or '.join(y.arg[0] for y in x.src) + ")", dtypes.void))),

  (UPat(Ops.CUSTOM, src=UPat(Ops.CUSTOMI), name="x"), lambda x: UOp(Ops.CUSTOMI, arg=(x.arg[0].format(*[y.arg[0] for y in x.src]), dtypes.void))),
  (UPat(Ops.INDEX, src=(UPat(Ops.CUSTOMI, name="x"), UPat(Ops.CONST, name="c")), name="g"),
   lambda x,c,g: x.replace(arg=(x.arg[0]+f".src[{c.val}]", dtypes.void)))
], compiled=False)

def _final_render(x:UOp, has_ctx:bool, depth=1) -> list[str]:
  # if the whole clause collapsed to a single predicate (no binds), rewrap it
  if x.op is Ops.CUSTOMI: x = UOp(Ops.AND, (x,))
  assert x.op is Ops.AND
  and_pieces: list[str] = []
  bound: dict[str, str] = {}  # rebinding a name renders an identity compare (setdefault semantics in the interpreter)
  or_pieces: list[str] = []
  def bind(nm:str, path:str):
    if nm in bound: and_pieces.append(f"{bound[nm]} is {path}")
    else: bound[nm] = path
  for s in x.src:
    if s.op is Ops.OR:
      assert len(or_pieces) == 0 and len(s.src) >= 1
      for ss in s.src: or_pieces.extend(_final_render(ss, has_ctx, depth+1))
    elif s.op is Ops.STORE:
      assert s.src[0].op is Ops.CUSTOMI and s.src[1].op is Ops.CUSTOMI
      bind(s.src[0].arg[0], s.src[1].arg[0])
    # repeat with named binds: binds come from the first src, and every element must be identical to it (setdefault semantics)
    elif s.op is Ops.CUSTOM and s.src[0].op is Ops.AND and all(y.op in (Ops.CUSTOMI, Ops.STORE) for y in s.src[0].src):
      stores, pred = partition(s.src[0].src, lambda x: x.op is Ops.STORE)
      it, base = s.src[1].arg[0], s.src[2].arg[0]
      for st in stores:
        # st.src[1] is the path this name binds, written as {it}.src[...]; rebase it on the first src
        first = f"{base}.src[0]{st.src[1].arg[0].removeprefix(it)}"
        pred.append(UOp(Ops.CUSTOMI, arg=(f"{st.src[1].arg[0]} is {first}", dtypes.void)))
        bind(st.src[0].arg[0], first)
      and_pieces.append(f"all([{' and '.join(y.arg[0] for y in pred)} for {it} in {base}.src])")
      if len(stores): and_pieces.append(f"len({base}.src) != 0")
    elif s.op is Ops.CUSTOMI: and_pieces.append(s.arg[0])
    else: raise UPatCompileError(f"can't compile this {s}")
  # if we have an or, render it
  if len(or_pieces):
    assert len(bound) == 0
    and_clause = ' and '.join(and_pieces)
    return [f"{'  '*depth}if {and_clause if len(and_clause) else 'True'}:"] + or_pieces
  # if we don't, this is a final return
  store_clause = ', '.join((["ctx=ctx"] if has_ctx else [])+[f"{k}={v}" for k,v in bound.items()])
  and_clause = ' and '.join(and_pieces + [f"(_ret:=_fxn({store_clause})) is not None"])
  return [f"{'  '*depth}if {and_clause}: return _ret"]

def _get_code(self:UPat, has_ctx:bool):
  ret = _get_clause(self, UOp(Ops.CUSTOMI, arg=("uop", dtypes.void)))
  try:
    # TODO: this should be tracked in a "system" rewrite, not untracked or tracked with kernel
    with Context(TRACK_MATCH_STATS=0):
      ret = graph_rewrite(ret, pm_proc, name="process UPat")
      dyn_lookup: dict[str, Any] = {}
      out = graph_rewrite(ret, pm_renderer, ctx=dyn_lookup, name="compile UPat")
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
