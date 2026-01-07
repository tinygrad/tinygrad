import heapq
from typing import Any
from collections import defaultdict
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, UPat, multirange_str
from tinygrad.helpers import prod, getenv, TUPLE_ORDER

def linearize(sink:UOp, schedule_mode:bool=False) -> list[UOp]:
  # this is a toposort with priority
  lst = list(sink.toposort())
  consumers: defaultdict[UOp, list[UOp]] = defaultdict(list)
  in_degree:dict[UOp, int] = {}
  out_degree:dict[UOp, int] = {}
  priorities:dict[UOp, tuple[int, int, Any]] = {}

  if schedule_mode:
    # construct the KERNEL children graph based on assigns
    for u in lst:
      if u.op is Ops.RANGE:
        in_degree.setdefault(u, 0)
        out_degree.setdefault(u, 0)
        priorities[u] = (0, 0, None)
        continue
      if u.op is not Ops.AFTER or u.src[1].op is Ops.RANGE: continue
      k = u.src[1]
      in_degree.setdefault(k, 0)
      out_degree.setdefault(k, 0)
      priorities[k] = (0, 0, None)
      for s in k.src[0].src if k.op is Ops.END else k.src:
        if s.op is Ops.AFTER:
          consumers[s.src[1]].append(k)
          out_degree[s.src[1]] = out_degree.get(s.src[1], 0) + 1
          in_degree[k] += 1
        elif s.op in {Ops.MSELECT, Ops.MSTACK}:
          for ss in s.src:
            if ss.op is Ops.MSELECT: ss = ss.src[0]
            if ss.op is not Ops.BUFFER:
              assert ss.op is Ops.AFTER, f"ss.op is not AFTER, it's {ss.op}"
              consumers[ss.src[1]].append(k)
              out_degree[ss.src[1]] = out_degree.get(ss.src[1], 0) + 1
              in_degree[k] += 1
        elif s.op in {Ops.BUFFER, Ops.BIND}:
          pass  # a BUFFER is already realized, BINDs are handled in complete_create_schedule_with_vars
        else:
          raise RuntimeError(f"input to kernel must be AFTER or BUFFER, not {s.op}")
    lst = list(in_degree.keys())
  else:
    for u in reversed(lst):
      for s in u.src: consumers[s].append(u)
      in_degree[u] = len(u.src)
      out_degree[u] = len(consumers[u])

      # we place UOps with higher run_counts later
      run_count = prod([int(r.vmax)+1 for r in u.ranges])

      # simple priority override. this is all bottom up now, smaller numbers will be closer to the top
      extra = None
      match u.op:
        # the order and placement of these defines is important
        case Ops.DEFINE_GLOBAL: priority, extra = -20, u.arg
        case Ops.DEFINE_VAR: priority, extra = -19, u.arg
        case Ops.DEFINE_LOCAL: priority = -18
        case Ops.DEFINE_REG: priority = -17
        case Ops.CONST: priority = -10  # early consts
        case Ops.LOAD: priority = -1    # place loads early
        case Ops.STORE: priority = 1    # place stores late
        case Ops.RANGE: priority = 5    # placing RANGE is good
        case Ops.END: priority = -5     # placing END is bad
        case _: priority = 0            # everything else has priority 0
      priorities[u] = (run_count, priority, extra)

  # number the uops in "ideal" order (schedule_mode skips tuplize to avoid comparing Kernel objects)
  use_tuplize = TUPLE_ORDER and not schedule_mode
  nkey = {u:i for i,u in enumerate(sorted(lst, key=lambda x: priorities[x]+(x.tuplize if use_tuplize else ())))}

  # then force them to be toposorted in as close to the ideal order as possible
  # schedule_mode: forward (in_degree, consumers), normal: backward then reverse (out_degree, u.src)
  degree = in_degree if schedule_mode else out_degree
  # schedule_mode uses positive nkey (no reversal), normal uses negative nkey (will reverse at end)
  heap = [(nkey[u], u) if schedule_mode else (-nkey[u], u) for u in lst if degree.get(u, 0) == 0]
  heapq.heapify(heap)
  newlst: list[UOp] = []
  while heap:
    newlst.append(u:=heapq.heappop(heap)[1])
    for v in (consumers[u] if schedule_mode else u.src):
      degree[v] -= 1
      if degree[v] == 0: heapq.heappush(heap, (nkey[v], v) if schedule_mode else (-nkey[v], v))
  if not schedule_mode: newlst = newlst[::-1]

  if getenv("DEBUG_LINEARIZE"):
    for i,u in enumerate(newlst):
      print(f"{i:4d} {str(u.op):20s} {multirange_str(u.ranges, color=True, pad=10)} {priorities[u]}")
  return newlst

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
        nesting |= {x:u for x in deps[u] if x.op is Ops.END and (u.op is Ops.SINK or u.src[1] in deps[x]) and x not in nesting}
      if u.op in (Ops.RANGE, Ops.END): deps[u][u] = None

    self.edges: dict[UOp, UOp] = {}
    siblings: dict[UOp, list[UOp]] = {}
    for k,vv in nesting.items(): siblings.setdefault(vv, []).append(k)
    for k,v in siblings.items():
      # ranges that have dependencies on other siblings need to be scheduled after them
      order = sorted(v, key=lambda x: len([u for u in v if u in deps[x]]))
      zipped = zip(order, order[1:]) if k.op is Ops.SINK else zip([k.src[1]] + order, order)
      for x,y in zipped:
        # TODO: this can happen! it causes infinite loop in shufflenet
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
  # split the ends
  (UPat(Ops.END, name="e"), do_split_ends),
])