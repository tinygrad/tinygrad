import heapq
from typing import Any
from collections import defaultdict
from tinygrad.uop import X86Ops, X86GroupOp
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, UPat, multirange_str
from tinygrad.helpers import prod, getenv, TUPLE_ORDER

def linearize(sink:UOp) -> list[UOp]:
  from tinygrad.renderer.x86 import RSP
  # this is a toposort with priority
  lst = list(sink.toposort())
  consumers: defaultdict[UOp, list[UOp]] = defaultdict(list)
  in_degree:dict[UOp, int] = {}
  out_degree:dict[UOp, int] = {}
  priorities:dict[UOp, tuple[int, int, Any]] = {}

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
      case Ops.LOAD: priority = -1    # place loads early
      case Ops.STORE: priority = 1    # place stores late
      case Ops.RANGE: priority = 5    # placing RANGE is good
      case Ops.END: priority = -5     # placing END is bad
      # x86 op version
      # stack pointer needs to be scheduled at the top of the kernel
      case X86Ops.DEFINE_REG: priority = -21 if u.arg == RSP else -20
      case X86Ops.IMM: priority = -10
      case _: priority = 0            # everything else has priority 0
    priorities[u] = (run_count, priority, extra)

  # number the uops in "ideal" order
  nkey = {u:i for i,u in enumerate(sorted(lst, key=lambda x: priorities[x]+(x.tuplize if TUPLE_ORDER and not getenv("CPU_X86") else ())))}

  # then force them to be toposorted in as close to the ideal order as possible
  heap = [(-nkey[sink], sink)]
  newlst = []
  lock: UOp|None = None
  stupid: int = 0
  clobbers: set[UOp] = set()
  while heap or clobbers:
    # if heap is empty we have a cycle and the flag producer must be rematerialized
    # we schedule the flag producer and free the clobbers
    if not heap:
      assert lock is not None and clobbers
      newlst.append(lock)
      for c in clobbers: heapq.heappush(heap, (-nkey[c],c))
      clobbers.clear()
      lock, stupid = None, 0

    u = heapq.heappop(heap)[1]

    # flags introduce state that must be dealt with, can't overwrite the flag until all its users and producer are scheduled
    if lock is not None:
      # if this is the flag producer we free the flag clobbers and release the lock
      if lock is u:
        for c in clobbers: heapq.heappush(heap, (-nkey[c],c))
        clobbers.clear()
        lock, stupid = None, 0
      # if this is the user of or is another flag producer it can't be scheduled
      # if this is a loop boundry or has a lower run count than the flag user that introduced the lock we also don't schedule
      # loop boundries do clobber but we also don't want to insert stuff from outside the loop into the loop
      # if there's no loop we also don't want to add IMM and DEFINE_REG in the middle of the kernel
      elif u.op in X86GroupOp.ReadFlags and lock is not u.src[-1] or u.op in X86GroupOp.WriteFlags or \
        u.op in {Ops.RANGE, Ops.END, X86Ops.IMM, X86Ops.DEFINE_REG} or priorities[u][0] < stupid:
        clobbers.add(u)
        continue
    # if there's no lock and this is a flag user its flag producer becomes the lock
    elif u.op in X86GroupOp.ReadFlags: lock, stupid = u.src[-1], priorities[u][0]

    newlst.append(u)

    for v in u.src:
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