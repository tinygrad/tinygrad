import heapq
from collections import defaultdict
from typing import Any

from tinygrad.helpers import TUPLE_ORDER, getenv, prod
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat, multirange_str


def linearize(sink:UOp) -> list[UOp]:
  # this is a toposort with priority
  lst = list(sink.toposort())
  consumers: defaultdict[UOp, list[UOp]] = defaultdict(list)
  in_degree:dict[UOp, int] = {}
  out_degree:dict[UOp, int] = {}
  priorities:dict[UOp, tuple[int, int, Any]] = {}

  # for schedule-level linearization, compute depth in KERNEL dependency graph (via AFTER ops)
  # this ensures kernels are ordered like BFS: all depth-0 before depth-1, etc.
  kernel_depth: dict[UOp, int] = {}
  kernel_first_after_pos: dict[UOp, int] = {}  # position of first AFTER pointing to each kernel
  for i, u in enumerate(lst):
    if u.op is Ops.KERNEL:
      max_dep_depth = -1
      for s in u.src:
        if s.op is Ops.AFTER and len(s.src) > 1 and s.src[1].op is Ops.KERNEL:
          max_dep_depth = max(max_dep_depth, kernel_depth.get(s.src[1], 0))
      kernel_depth[u] = max_dep_depth + 1
    # track the first AFTER op that points to each kernel (for BFS-like ordering)
    if u.op is Ops.AFTER and len(u.src) > 1 and u.src[1].op is Ops.KERNEL:
      kernel_first_after_pos.setdefault(u.src[1], i)

  # toposort position for stable ordering within same depth
  toposort_pos = {u:i for i,u in enumerate(lst)}

  # get consumers and assign priorities
  # NOTE: this requires the lst be locally toposorted
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
      case Ops.KERNEL: priority, extra = 6, (kernel_depth[u], kernel_first_after_pos.get(u, toposort_pos[u]))
      case Ops.END: priority = -5     # placing END is bad
      case _: priority = 0            # everything else has priority 0
    priorities[u] = (run_count, priority, extra)

  # number the uops in "ideal" order
  nkey = {u:i for i,u in enumerate(sorted(lst, key=lambda x: priorities[x]+(x.tuplize if TUPLE_ORDER else ())))}

  # then force them to be toposorted in as close to the ideal order as possible
  heap = [(-nkey[sink], sink)]
  newlst = []
  while heap:
    newlst.append(u:=heapq.heappop(heap)[1])
    for v in u.src:
      out_degree[v] -= 1
      if out_degree[v] == 0: heapq.heappush(heap, (-nkey[v],v))
  newlst = newlst[::-1]

  # for schedule-level linearization, reorder KERNELs by depth to match BFS behavior
  # this is necessary because the heap-based algorithm processes kernels one at a time
  if kernel_depth:
    # extract kernel positions and sort by (depth, first_after_position)
    kernel_positions = [(i, u) for i, u in enumerate(newlst) if u.op is Ops.KERNEL]
    sorted_kernels = sorted([u for _, u in kernel_positions],
                            key=lambda u: (kernel_depth[u], kernel_first_after_pos.get(u, toposort_pos[u])))
    # place sorted kernels back in their original positions
    for (pos, _), kernel in zip(kernel_positions, sorted_kernels):
      newlst[pos] = kernel

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
