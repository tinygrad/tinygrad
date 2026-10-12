import heapq
from typing import Any
from collections import defaultdict
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, GroupOp, UPat, multirange_str
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import prod, getenv, dedup, TUPLE_ORDER

def linearize(sink:UOp) -> list[UOp]:
  # this is a toposort with priority
  lst = list(sink.toposort())
  out_degree:defaultdict[UOp, int] = defaultdict(int)
  priorities:dict[UOp, tuple[int, int, int, Any]] = {}

  # Scalar expressions derived only from constants and parameters can precede every control-flow scope.
  invariant:set[UOp] = set()
  for u in lst:
    if u.op in (Ops.CONST, Ops.PARAM) or (u.op in GroupOp.ALU | {Ops.CAST} and all(s in invariant for s in u.src)):
      invariant.add(u)

  # get consumers and assign priorities
  # NOTE: this requires the lst be locally toposorted
  for u in reversed(lst):
    for s in u.src_without_body: out_degree[s] += 1

    # we place UOps with higher run_counts later
    run_count = prod([int(r.vmax)+1 for r in u.ranges])

    # simple priority override. this is all bottom up now, smaller numbers will be closer to the top
    extra = None
    match u.op:
      # the order and placement of these defines is important
      case Ops.PARAM: priority, extra = -20, u.arg.slot
      case _ if u in invariant: priority = -19
      case Ops.BUFFER | Ops.ALLOC: priority = -17 if u.addrspace == AddrSpace.LOCAL else -18
      case Ops.LOAD: priority = -1    # place loads early
      case Ops.STORE: priority = 1    # place stores late
      case Ops.RANGE: priority = 5    # placing RANGE is good
      case Ops.END | Ops.BACKEDGE: priority = -5     # placing loop exits is bad
      case _: priority = 0            # everything else has priority 0
    priorities[u] = (run_count, len(u.ranges), priority, extra)

  # number the uops in "ideal" order
  nkey = {u:i for i,u in enumerate(sorted(lst, key=lambda x: priorities[x]+(x.tuplize if TUPLE_ORDER else ())))}

  # then force them to be toposorted in as close to the ideal order as possible
  heap = [(-nkey[sink], sink)]
  newlst = []
  while heap:
    newlst.append(u:=heapq.heappop(heap)[1])
    for v in u.src_without_body:
      out_degree[v] -= 1
      if out_degree[v] == 0: heapq.heappush(heap, (-nkey[v],v))
  newlst = newlst[::-1]

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
    # Only control-flow nodes matter here. Bitsets avoid copying growing dependency dictionaries at every UOp.
    deps: dict[UOp, int] = {}
    indices: dict[UOp, int] = {}
    controls: list[UOp] = []
    ends = assigned = 0
    siblings: dict[UOp, list[UOp]] = {}
    for u in sink.toposort():
      deps[u] = 0
      for s in u.src_without_body: deps[u] |= deps[s]

      if u.op in (Ops.END, Ops.BACKEDGE, Ops.SINK):
        pending = deps[u] & ends & ~assigned
        parent = 1 << indices[u.src[1]] if u.op is not Ops.SINK else 0
        # An end preceding the parent range in topological order cannot be nested inside it.
        if parent: pending &= -parent
        while pending:
          bit = pending & -pending
          pending -= bit
          x = controls[bit.bit_length()-1]
          if not parent or deps[x] & parent:
            siblings.setdefault(u, []).append(x)
            assigned |= bit
      if u.op in (Ops.RANGE, Ops.END, Ops.BACKEDGE):
        indices[u] = len(controls)
        controls.append(u)
        bit = 1 << indices[u]
        deps[u] |= bit
        if u.op is not Ops.RANGE: ends |= bit

    self.edges: dict[UOp, UOp] = {}
    for k,v in siblings.items():
      # ranges that have dependencies on other siblings need to be scheduled after them
      siblings_mask = sum(1 << indices[u] for u in v)
      order = sorted(v, key=lambda x: (deps[x] & siblings_mask).bit_count())
      zipped = zip(order, order[1:]) if k.op is Ops.SINK else zip([k.src[1]] + order, order)
      for x,y in zipped:
        # TODO: this can happen! it causes infinite loop in shufflenet
        assert not deps[x] & (1 << indices[y.src[1]])
        self.edges[y.src[1]] = x

pm_add_control_flow = PatternMatcher([
  # the ordering dep goes on the bound wrapped in AFTER, so a RANGE always has exactly one src
  (UPat(Ops.RANGE, name="x"), lambda ctx,x: x.replace(src=(x.src[0].after(y),)) if (y:=ctx.edges.get(x)) is not None else None),
])

def do_split_ends(e:UOp):
  ret = e.src[0]
  rngs = dedup(r for s in e.src[1:] for r in ((s,) if s.op is Ops.RANGE else s.ranges))
  for r in sorted(rngs, key=lambda x: x.arg, reverse=True): ret = ret.end(r)
  return ret

pm_split_ends = PatternMatcher([
  # split the ends
  (UPat(Ops.END, name="e"), do_split_ends),
])
