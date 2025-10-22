import heapq
from typing import cast
from collections import defaultdict
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, UPat
from tinygrad.helpers import panic

# only needed if device doesn't support gated stores
pm_linearize_cleanups = PatternMatcher([
  # if statements are not allowed in the graph
  (UPat((Ops.IF, Ops.ENDIF)), lambda: panic(RuntimeError("if not allowed in graph"))),
  # gated INDEX becomes IF-STORE-ENDIF. this is the only use of IF-ENDIF
  (UPat(Ops.STORE, name="u", src=(UPat(Ops.INDEX, src=(UPat(), UPat(), UPat(name="gate", dtype=dtypes.bool))).or_casted(), UPat()),
        allow_any_len=True), lambda u, gate: (u, [mif:=UOp(Ops.IF, src=(gate, u.src[0])), u, UOp(Ops.ENDIF, src=(mif,))]))
])

# requires lst be toposorted. like graph rewrite, but for lines
def line_rewrite(lst:list[UOp], pm:PatternMatcher) -> list[UOp]:
  newlst = []
  replaced: dict[UOp, UOp] = {}
  for u in lst:
    nu = u.replace(src=tuple([replaced[x] for x in u.src]))
    ret: tuple[UOp, list[UOp]] = cast(tuple[UOp, list[UOp]]|None, pm.rewrite(nu)) or (nu, [nu])
    replaced[u] = ret[0]
    newlst.extend(ret[1])
  return newlst

def linearize(u:UOp) -> list[UOp]:
  lst = list(u.toposort())
  in_this_block = set(lst)
  local_children: defaultdict[UOp, list[UOp]] = defaultdict(list)
  in_degree:dict[UOp, int] = {}
  priorities:dict[UOp, int] = {}

  # get local children and assign priorities
  # NOTE: this requires the lst be locally toposorted
  for u in reversed(lst):
    in_degree[u] = 0
    for s in u.src:
      if s in in_this_block:
        local_children[s].append(u)
        in_degree[u] += 1
    # put loads in the beginning of the block and prevent priority inversion. hack for BARRIER grouping too
    priority = [0] + [priorities[x] for x in local_children[u]]
    if u.op is Ops.LOAD: priority.append(-1000)
    if u.op is Ops.BARRIER: priority.append(-1500)
    # ranges are scheduled as late as possible so anything that can be outside is
    #if u.op is Ops.RANGE: priority = [2000]
    # move defines and consts to the top
    if u.op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.DEFINE_REG, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.CONST}: priority.append(-2000)
    priorities[u] = min(priority)

  # number the uops in "ideal" order
  nkey = {u:i for i,u in enumerate(sorted(lst, key=lambda x: (priorities[x],)+x.tuplize))}

  # then force then to be toposorted in as close to the ideal order as possible
  heapq.heapify(heap:=[(nkey[u],u) for u in lst if in_degree[u] == 0])
  newlst = []
  while heap:
    newlst.append(u:=heapq.heappop(heap)[1])
    for v in local_children[u]:
      in_degree[v] -= 1
      if in_degree[v] == 0: heapq.heappush(heap, (nkey[v],v))

  assert len(newlst) == len(lst), f"len mismatch {len(newlst)} != {len(lst)}"
  return line_rewrite(newlst, pm_linearize_cleanups)

class CFGContext:
  def __init__(self, sink:UOp):
    # there are 3 relationships between ranges:
    # nested, meaning endrange y is a dependency of endrange x and range x is a dependency of endrange y
    # dependent, meaning endrange y is a dependency of endrange x and range x is not a dependency of endrange y
    # independent, endrange y is not a dependency of endrange x
    # everything is nested inside the sink
    deps: dict[UOp, dict[UOp, None]] = {}
    nesting: dict[UOp, UOp] = {}
    for u in sink.toposort():
      # get the deps from the src
      deps[u] = {}
      for s in u.src: deps[u] |= deps[s]

      if u.op in (Ops.END, Ops.SINK):
        nesting |= {x:u for x in deps[u] if x.op is Ops.END and (u.op is Ops.SINK or u.src[0] in deps[x]) and x not in nesting}
      if u.op in (Ops.RANGE, Ops.END): deps[u][u] = None

    self.edges: dict[UOp, UOp] = {}
    siblings: dict[UOp, list[UOp]] = {}
    for k,vv in nesting.items(): siblings.setdefault(vv, []).append(k)
    for k,v in siblings.items():
      # range/if that have dependencies on other siblings need to run after them
      order = sorted(v, key=lambda x: len([u for u in v if u in deps[x]]))
      zipped = zip(order, order[1:]) if k.op is Ops.SINK else zip([k.src[0]] + order, order)
      for x,y in zipped:
        # TODO: is this check correct?
        if y.src[0] not in x.backward_slice_with_self:
          self.edges[y.src[0]] = x

pm_add_control_flow = PatternMatcher([
  (UPat((Ops.RANGE, Ops.IF), name="x"), lambda ctx,x: x.replace(src=x.src+(y,)) if (y:=ctx.edges.get(x)) is not None else None),
])

def do_merge_ends(s:UOp):
  # NOTE: this can fail
  stacked: dict[UOp, list[UOp]] = {}
  for x in s.toposort():
    if x.op is Ops.END:
      assert x.arg == 1, "ends must be single ends for linearizer"
      stacked.setdefault(x.src[0], []).append(x)
  replaces = {}
  for k,v in stacked.items():
    if len(v) == 1: continue
    rep = UOp(v[0].op, src=tuple([k] + [y for x in v for y in x.src[1:]]), arg=v[0].arg)
    for x in v: replaces[x] = rep
    if not len(replaces): return None
  return s.substitute(replaces)

pm_add_ends = PatternMatcher([
  # put the end on the store
  (UPat(Ops.STORE, name="s"), lambda s: s.replace(src=s.src[:2]).end(ends=s.src[2:]) if len(s.src) > 2 else None),
  # END is only on RANGES
  (UPat(Ops.END, name="e"), lambda e: UOp.end(*e.src[e.arg:], ends=sorted(UOp.sink(*e.src[:e.arg]).ranges, key=lambda x: x.arg))),
  # for renderering and linearizing, all ends must end one loop
  (UPat(Ops.END, name="e"), lambda e: e.replace(src=e.src[e.arg-1:], arg=1).end(ends=e.src[:e.arg-1]) if e.arg > 1 else None),
])

pm_merge_ends = PatternMatcher([
  (UPat(Ops.SINK, name="s"), do_merge_ends),
])