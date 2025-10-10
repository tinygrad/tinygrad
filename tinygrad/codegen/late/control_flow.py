from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.helpers import dedup
from collections import defaultdict
from itertools import groupby
from functools import reduce
import heapq

def linearize(sink:UOp) -> list[UOp]:
  lst = list(sink.toposort())
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
  return newlst

def add_endrange(x:UOp):
  if not ((x.op is Ops.LOAD and x.src[-1].op is Ops.STORE) or all(s.op is Ops.STORE and any(n.op is Ops.RANGE for n in s.src) for s in x.src)):
    return None
  src: list[UOp] = []
  for k,g in groupby(x.src, key=lambda k: tuple(dedup(s for s in k.src if s.op is Ops.RANGE))):
    if not k: src.extend(g)
    else: src.extend(reduce(lambda acc,rng: (UOp(Ops.ENDRANGE, src=(rng,) + acc),), reversed(k), tuple(g))) # type: ignore
  return x.replace(src=tuple(src))

def add_endif(x:UOp):
  groups = {k: tuple(g) for k,g in groupby(x.src, key=lambda k: k.src[2] if len(k.src) >= 3 and k.src[2].op is Ops.IF else k)}
  if not any(k.op is Ops.IF for k in groups): return None
  return x.replace(src=tuple(UOp(Ops.ENDIF, src=(k,) + g) if k.op is Ops.IF else k for k,g in groups.items()))

# some Ops.IF aren't closed by an Ops.STORE, in that case the Ops.SINK closes it
def close_ifs(x:UOp):
  consumers = x.get_consumer_map()
  if (y:=next((s for s in consumers if s.op is Ops.IF and all(n.op is not Ops.ENDIF for n in consumers[s])), None)) is not None:
    return x.replace(src=(UOp(Ops.ENDIF, src=(y,) + x.src),))
  return None

pm_control_flow_ends = PatternMatcher([
  (UPat((Ops.SINK, Ops.NOOP, Ops.LOAD), name="x"), add_endrange),
  (UPat((Ops.SINK, Ops.ENDRANGE, Ops.BARRIER), name="x"), add_endif),
  (UPat(Ops.SINK, name="x"), close_ifs),
])

class CFGContext:
  def __init__(self, sink:UOp):
    # there are 3 relationships between ranges:
    # nested, meaning endrange y is a dependency of endrange x and range x is a dependency of endrange y
    # dependent, meaning endrange y is a dependency of endrange x and range x is not a dependency of endrange y
    # independent, endrange y is not a dependency of endrange x
    deps: dict[UOp, set[UOp]] = {}
    nesting: dict[UOp, UOp] = {}
    for u in sink.toposort():
      deps[u] = set().union(*(deps[s] for s in u.src))
      if u.op in (Ops.ENDRANGE, Ops.ENDIF):
        for n in [x for x in deps[u] if x.op in (Ops.ENDRANGE, Ops.ENDIF) and u.src[0] in deps[x] and x not in nesting]: nesting[n] = u
      if u.op is Ops.SINK:
        for n in [x for x in deps[u] if x.op in (Ops.ENDRANGE, Ops.ENDIF) and x not in nesting]: nesting[n] = u
      if u.op in (Ops.RANGE, Ops.ENDRANGE, Ops.IF, Ops.ENDIF): deps[u] |= {u}

    self.edges: dict[UOp, UOp] = {}
    siblings: dict[UOp, list[UOp]] = {}
    for k,vv in nesting.items(): siblings.setdefault(vv, []).append(k)
    for k,v in siblings.items():
      # range/if that have dependencies on other siblings need to run after them
      order = sorted(v, key=lambda x: len([y for y in v if y in deps[x]]))
      zipped = zip(order, order[1:]) if k.op is Ops.SINK else zip([k.src[0]] + order, order)
      for x,y in zipped: self.edges[y.src[0]] = x

pm_control_flow_starts = PatternMatcher([
  (UPat((Ops.RANGE, Ops.IF), src=(UPat(),), name="x"), lambda ctx,x: x.replace(src=x.src+(y,)) if (y:=ctx.edges.get(x)) is not None else None),
  (UPat(Ops.IF, src=(UPat(), UPat(Ops.BARRIER)), name="x"), lambda ctx,x: x.replace(src=x.src+(y,)) if (y:=ctx.edges.get(x)) is not None else None),
])
