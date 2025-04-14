from typing import Any, Callable
import itertools, inspect, functools
from tinygrad.helpers import partition, flatten, dedup
from tinygrad.ops import UPat, UPatAny, UOp, Ops, PatternMatcher, graph_rewrite

# **** UPat compiled ****

def _get_clause(self:UPat, base:UOp, depth=0) -> UOp:
  if isinstance(self, UPatAny):
    assert len(self.src) == 1
    return UOp(Ops.AND, src=(UOp(Ops.OR, src=tuple(_get_clause(s, base, depth) for s in self.src[0])),))
  # build the and_clause for acceptance
  and_clause = []
  if self.op is not None:
    if len(self.op) > 1:
      and_clause.append(UOp(Ops.CUSTOM, src=(base, UOp(Ops.BIND, arg=tuple(int(x) for x in self.op))), arg="{0}.op in {1}"))
    else:
      and_clause.append(UOp(Ops.CUSTOM, src=(base,), arg="{0}.op == "+str(self.op[0].value)))
  if self.arg is not None:
    if isinstance(self.arg, int):
      and_clause.append(UOp(Ops.CUSTOM, src=(base,), arg="{0}.arg == "+str(int(self.arg))))
    else:
      and_clause.append(UOp(Ops.CUSTOM, src=(base, UOp(Ops.BIND, arg=self.arg)), arg="{0}.arg == {1}"))
  if self.strict_length or self.required_len > 0:
    and_clause.append(UOp(Ops.CUSTOM, src=(base,),
                          arg=("len({0}.src) == " if self.strict_length else "len({0}.src) >= ")+str(self.required_len)))
  if self.name is not None:
    and_clause.append(UOp(Ops.ASSIGN, src=(UOp(Ops.DEFINE_VAR, arg=self.name), base)))
  if self.dtype is not None:
    if len(self.dtype) > 1:
      and_clause.append(UOp(Ops.CUSTOM, src=(base, UOp(Ops.BIND, arg=tuple(self.dtype))),
        arg="({0}.dtype in {1} or {0}.dtype._scalar in {1})"))
    else:
      and_clause.append(UOp(Ops.CUSTOM, src=(base, UOp(Ops.BIND, arg=self.dtype[0])),
        arg="({0}.dtype == {1} or {0}.dtype._scalar == {1})"))
  and_uop = UOp(Ops.AND, src=tuple(and_clause))
  if self.src is None: return and_uop
  if len(self.src) == 1 and isinstance(self.src[0], tuple):
    more_cond = [_get_clause(s, base.gep(i), depth) for i,s in enumerate(self.src[0])]
    return UOp(Ops.AND, src=tuple([and_uop]+more_cond))
  if len(self.src) == 1 and isinstance(self.src[0], itertools.repeat):
    it = UOp(Ops.NOOP, arg=f"ituop{depth}")
    match = _get_clause(next(self.src[0]), it, depth+1)
    # NOTE: use of a generator here is slow
    rep = UOp(Ops.RANGE, src=(match, it, base), arg="all([{0} for {1} in {2}.src])")
    return UOp(Ops.AND, src=(and_uop, rep))
  if len(self.src) > 1 and all(isinstance(x, tuple) for x in self.src):
    fork_cond = []
    for ss in self.src:
      more_cond = [_get_clause(s, base.gep(i), depth) for i,s in enumerate(ss)]
      fork_cond.append(UOp(Ops.AND, src=tuple(more_cond)))
    rep = UOp(Ops.OR, src=tuple(fork_cond))
    return UOp(Ops.AND, src=(and_uop, rep))
  raise RuntimeError("broken")

# *** pattern matcher ***

def wrap(ctx, x):
  ctx[ret:=f"a{len(ctx)}"] = x.arg
  return UOp(Ops.NOOP, arg=ret)

def do_catand(a):
  found = False
  new_src = []
  or_clause = []
  for x in a.src:
    if x.op is Ops.AND:
      new_src.extend(x.src)
      found = True
    elif x.op is Ops.OR:
      or_clause.append(x)
    else: new_src.append(x)

  # one or clause max
  if len(or_clause) > 1:
    found = True
    # need the product of the or clauses
    new_or = []
    for x in itertools.product(*[x.src for x in or_clause]):
      new_or.append(UOp(Ops.AND, src=x))
    or_clause = [UOp(Ops.OR, src=tuple(new_or))]

  # push assigns to the top
  if any(x.op is Ops.ASSIGN for x in new_src) and len(or_clause):
    assert len(or_clause) == 1
    assigns, new_src = partition(new_src, lambda x: x.op is Ops.ASSIGN)
    assert len(assigns) >= 1
    new_or_srcs = []
    for x in or_clause[0].src:
      assert x.op is Ops.AND
      new_or_srcs.append(x.replace(src=x.src+tuple(assigns)))
    or_clause = [UOp(Ops.OR, src=tuple(new_or_srcs))]
    found = True

  return UOp(Ops.AND, src=tuple(new_src+or_clause)) if found else None

def do_fixassigns(a:UOp):
  assigns, new_src = partition(a.src, lambda x: x.op is Ops.ASSIGN)
  dict_assigns: dict[UOp, UOp] = {}
  found = False
  for a in assigns:
    if a.src[0] in dict_assigns:
      new_src.append(UOp(Ops.CMPNE, src=(dict_assigns[a.src[0]], a.src[1])))
      found = True
    else:
      dict_assigns[a.src[0]] = a.src[1]
  if found:
    for k,v in dict_assigns.items():
      new_src.append(UOp(Ops.ASSIGN, src=(k,v)))
    return UOp(Ops.AND, src=tuple(new_src))

def do_pullfromor(a):
  in_all = []
  for x in a.src[0].src:
    if x.op is not Ops.ASSIGN and all(x in s.src for s in a.src[1:]):
      in_all.append(x)
  if len(in_all):
    new_ands = []
    for x in a.src:
      new_ands.append(UOp(Ops.AND, src=tuple(y for y in x.src if y not in in_all)))
    return UOp(Ops.AND, src=tuple(in_all)+(UOp(Ops.OR, src=tuple(new_ands)),))

def do_collapseor(a:UOp):
  if all(x.op is Ops.AND and len(x.src) == 1 and x.src[0].op is Ops.OR for x in a.src):
    all_srcs = flatten([x.src[0].src for x in a.src])
    return UOp(Ops.OR, src=tuple(all_srcs))

# processor
pm_proc = PatternMatcher([
  # clean up ANDs
  (UPat(Ops.AND, name="a"), do_catand),

  # do_fixassigns
  (UPat(Ops.AND, name="a"), do_fixassigns),

  # pull dups from or
  (UPat(Ops.OR, name="a"), do_pullfromor),

  # collapse or maybe
  (UPat(Ops.OR, name="a"), do_collapseor),

  # dedup all (needed?)
  (UPat((Ops.AND, Ops.OR), name="a"), lambda a: a.replace(src=s) if len(s:=dedup(a.src)) != len(a.src) else None),
], compiled=False)

# renderer
pm_renderer = PatternMatcher([
  (UPat(Ops.BIND, name="x"), wrap),

  # CMPNE is actually equal
  (UPat(Ops.CMPNE, name="x"), lambda x: UOp(Ops.CUSTOM, src=x.src, arg="{0} is {1}")),

  # RANGE can't have OR inside it
  (UPat(Ops.RANGE, src=(UPat(Ops.AND, src=UPat(Ops.NOOP), name="x"), UPat(), UPat()), name="r"),
    lambda r,x: r.replace(op=Ops.CUSTOM, src=(UOp(Ops.NOOP, arg="(" + ' and '.join(y.arg for y in x.src) + ")"),)+r.src[1:])),

  (UPat(Ops.CUSTOM, src=UPat(Ops.NOOP), name="x"), lambda x: UOp(Ops.NOOP, arg=x.arg.format(*[y.arg for y in x.src]))),
  (UPat(Ops.GEP, src=UPat(Ops.NOOP, name="x"), name="g"), lambda x,g: x.replace(arg=x.arg+f".src[{g.arg[0]}]"))
], compiled=False)

def _get_code(self:UPat, has_ctx:bool):
  ret = _get_clause(self, UOp(Ops.NOOP, arg="uop"))
  ret = graph_rewrite(ret, pm_proc)

  dyn_lookup: dict[str, Any] = {}
  out = graph_rewrite(ret, pm_renderer, ctx=dyn_lookup, name="compile UPat")

  # build the function, try 2
  # renderer
  def render(x:UOp, depth=1) -> list[str]:
    assert x.op is Ops.AND
    and_clause = ' and '.join([s.arg for s in x.src if s.op is Ops.NOOP])
    ret = [f"{'  '*depth}if {and_clause if len(and_clause) else 'True'}:"]
    has_or = False
    assign_dict = ["ctx=ctx"] if has_ctx else []
    for s in x.src:
      if s.op is Ops.OR:
        assert has_or is False
        assert len(s.src) >= 1
        for ss in s.src: ret.extend(render(ss, depth+1))
        has_or = True
      elif s.op is Ops.ASSIGN:
        assert s.src[0].op is Ops.DEFINE_VAR
        assert s.src[1].op is Ops.NOOP
        assign_dict.append(f"{s.src[0].arg}={s.src[1].arg}")
      elif s.op is not Ops.NOOP:
        raise NotImplementedError(f"can't compile this {s}")
    if not has_or:
      and_clause = ' and '.join([s.arg for s in x.src if s.op is Ops.NOOP] + ["(_ret:=_fxn("+', '.join(assign_dict)+")) is not None"])
      ret = [f"{'  '*depth}if {and_clause if len(and_clause) else 'True'}:"]
      ret[-1] += " return _ret"
      ret[-1] = ret[-1].replace("if True: ", "")
    return ret

  try:
    rendered = render(out)
  except NotImplementedError:
    #print("FAIL2", self, self.location)
    return None

  code = [f"# match for {self.location}", "def match(uop, ctx):"]
  code += rendered
  code += ["  return None"]
  return '\n'.join(code), dyn_lookup

@functools.cache
def upat_compile(self:UPat, fxn) -> Callable|None:
  has_ctx = 'ctx' in inspect.signature(fxn).parameters
  code = _get_code(self, has_ctx)
  if code is None: return None
  code_str, dyn_lookup = code
  globs = dyn_lookup.copy()
  globs["_fxn"] = fxn
  namespace: dict = {}
  #print(code_str)
  exec(code_str, globs, namespace)  # pylint: disable=W0122
  return namespace["match"]
