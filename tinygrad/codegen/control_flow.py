import heapq
from collections import defaultdict
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, UPat

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
  return newlst

class CFGContext:
  def __init__(self, sink:UOp):
    # there are 3 relationships between ranges:
    # nested, meaning endrange y is a dependency of endrange x and range x is a dependency of endrange y
    # dependent, meaning endrange y is a dependency of endrange x and range x is not a dependency of endrange y
    # independent, endrange y is not a dependency of endrange x
    # everything is nested inside the sink
    deps: dict[UOp, set[UOp]] = {}
    nesting: dict[UOp, UOp] = {}
    for u in sink.toposort():
      deps[u] = set().union(*(deps[s] for s in u.src))
      if u.op in (Ops.END, Ops.ENDIF, Ops.SINK):
        nesting |= {x:u for x in deps[u] if x.op in (Ops.END, Ops.ENDIF) and (u.op is Ops.SINK or u.src[0] in deps[x]) and x not in nesting}
      if u.op in (Ops.RANGE, Ops.END, Ops.IF, Ops.ENDIF): deps[u] |= {u}

    self.edges: dict[UOp, UOp] = {}
    siblings: dict[UOp, list[UOp]] = {}
    for k,vv in nesting.items(): siblings.setdefault(vv, []).append(k)
    for k,v in siblings.items():
      # range/if that have dependencies on other siblings need to run after them
      order = sorted(v, key=lambda x: len(deps[x].intersection(v)))
      zipped = zip(order, order[1:]) if k.op is Ops.SINK else zip([k.src[0]] + order, order)
      for x,y in zipped: self.edges[y.src[0]] = x

# put all endifs on the sink
def add_endifs(sink:UOp):
  if sink.src[0].op is Ops.ENDIF: return None
  ifs = [x for x in sink.toposort() if x.op is Ops.IF]
  if len(ifs) == 0: return None
  assert len(ifs) == 1, "we only support one if"
  return sink.replace(src=(UOp(Ops.ENDIF, src=(ifs[0],*sink.src))))

pm_add_control_flow = PatternMatcher([
  (UPat((Ops.RANGE, Ops.IF), src=(UPat(),), name="x"), lambda ctx,x: x.replace(src=x.src+(y,)) if (y:=ctx.edges.get(x)) is not None else None),
  (UPat(Ops.IF, src=(UPat(), UPat(Ops.BARRIER)), name="x"), lambda ctx,x: x.replace(src=x.src+(y,)) if (y:=ctx.edges.get(x)) is not None else None),
  (UPat(Ops.SINK, name="sink"), add_endifs),
])
