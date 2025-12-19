import heapq
from collections import defaultdict
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, UPat, multirange_str
from tinygrad.helpers import prod, getenv, TUPLE_ORDER

def linearize(sink:UOp, for_scheduling:bool=False) -> list[UOp]:
  lst = list(sink.toposort())
  consumers:defaultdict[UOp, list[UOp]] = defaultdict(list)
  in_degree:dict[UOp, int] = {}
  priorities:dict[UOp, tuple] = {}

  for u in reversed(lst):
    for s in u.src: consumers[s].append(u)
    in_degree[u] = len(u.src)

  if for_scheduling:
    topo_idx = {u:i for i,u in enumerate(lst)}
    depth:dict[UOp, int] = {}
    for u in lst: depth[u] = max((depth.get(s, 0) for s in u.src), default=0) + 1 if u.src else 0
    for u in lst:
      if u.op is Ops.KERNEL:
        radj = 1000 if any(s.op is Ops.BIND and len(s.src) > 1 and s.src[1].op is Ops.RANGE for s in u.src) else 0
      elif u.op is Ops.RANGE: radj = 500
      elif u.op is Ops.END: radj = 2000
      elif u.op is Ops.AFTER:
        w = u.src[1]
        if w.op is Ops.KERNEL: radj = 1000 if any(s.op is Ops.BIND and len(s.src) > 1 and s.src[1].op is Ops.RANGE for s in w.src) else 0
        elif w.op is Ops.END: radj = 2000
        elif w.op is Ops.RANGE: radj = 500
        else: radj = 0
      else: radj = 0
      priorities[u] = (radj, depth[u], topo_idx[u])
    in_deg = dict(in_degree)
    heapq.heapify(heap:=[(priorities[u], topo_idx[u], u) for u in lst if in_deg[u] == 0])
    out:list[UOp] = []
    while heap:
      out.append(u:=heapq.heappop(heap)[2])
      for v in consumers[u]:
        in_deg[v] -= 1
        if in_deg[v] == 0: heapq.heappush(heap, (priorities[v], topo_idx[v], v))
    return out

  # codegen path
  out_degree:dict[UOp, int] = {}
  for u in reversed(lst): out_degree[u] = len(consumers[u])
  for u in reversed(lst):
    run_count = prod([int(r.vmax)+1 for r in u.ranges])
    extra = None
    match u.op:
      case Ops.DEFINE_GLOBAL: priority, extra = -20, u.arg
      case Ops.DEFINE_VAR: priority, extra = -19, u.arg
      case Ops.DEFINE_LOCAL: priority = -18
      case Ops.DEFINE_REG: priority = -17
      case Ops.CONST: priority = -10
      case Ops.LOAD: priority = -1
      case Ops.STORE: priority = 1
      case Ops.RANGE: priority = 5
      case Ops.END: priority = -5
      case _: priority = 0
    priorities[u] = (run_count, priority, extra)
  nkey = {u:i for i,u in enumerate(sorted(lst, key=lambda x: priorities[x]+(x.tuplize if TUPLE_ORDER else ())))}
  cg_heap:list[tuple[int, UOp]] = [(-nkey[sink], sink)]
  newlst:list[UOp] = []
  while cg_heap:
    newlst.append(u:=heapq.heappop(cg_heap)[1])
    for v in u.src:
      out_degree[v] -= 1
      if out_degree[v] == 0: heapq.heappush(cg_heap, (-nkey[v], v))
  newlst = newlst[::-1]
  if getenv("DEBUG_LINEARIZE"):
    for i,u in enumerate(newlst): print(f"{i:4d} {str(u.op):20s} {multirange_str(u.ranges, color=True, pad=10)} {priorities[u]}")
  return newlst

class CFGContext:
  def __init__(self, sink:UOp):
    deps:dict[UOp, dict[UOp, None]] = {}
    nesting:dict[UOp, UOp] = {}
    for u in sink.toposort():
      deps[u] = {}
      for s in u.src: deps[u] |= deps[s]
      if u.op in (Ops.END, Ops.SINK):
        nesting |= {x:u for x in deps[u] if x.op is Ops.END and (u.op is Ops.SINK or u.src[1] in deps[x]) and x not in nesting}
      if u.op in (Ops.RANGE, Ops.END): deps[u][u] = None
    self.edges:dict[UOp, UOp] = {}
    siblings:dict[UOp, list[UOp]] = {}
    for k,vv in nesting.items(): siblings.setdefault(vv, []).append(k)
    for k,v in siblings.items():
      order = sorted(v, key=lambda x: len([u for u in v if u in deps[x]]))
      zipped = zip(order, order[1:]) if k.op is Ops.SINK else zip([k.src[1]] + order, order)
      for x,y in zipped:
        assert y.src[1] not in x.backward_slice_with_self
        self.edges[y.src[1]] = x

pm_add_control_flow = PatternMatcher([
  (UPat(Ops.RANGE, name="x"), lambda ctx,x: x.replace(src=x.src+(y,)) if (y:=ctx.edges.get(x)) is not None else None),
])

def do_split_ends(e:UOp):
  ret = e.src[0]
  for r in sorted(UOp.sink(*e.src[1:]).ranges, key=lambda x: x.arg, reverse=True): ret = ret.end(r)
  return ret

pm_split_ends = PatternMatcher([
  (UPat(Ops.END, name="e"), do_split_ends),
])
