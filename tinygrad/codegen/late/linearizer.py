import heapq
from typing import Any
from collections import defaultdict
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, UPat, multirange_str, ParamArg, AxisType
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import prod, getenv, TUPLE_ORDER

def linearize(sink:UOp) -> list[UOp]:
  # this is a toposort with priority
  lst = list(sink.toposort())
  out_degree:defaultdict[UOp, int] = defaultdict(int)
  priorities:dict[UOp, tuple[int, int, Any]] = {}

  # get consumers and assign priorities
  # NOTE: this requires the lst be locally toposorted
  for u in reversed(lst):
    for s in u.src: out_degree[s] += 1

    # we place UOps with higher run_counts later
    run_count = prod([int(r.vmax)+1 for r in u.ranges])

    # simple priority override. this is all bottom up now, smaller numbers will be closer to the top
    extra = None
    match u.op:
      # the order and placement of these defines is important
      case Ops.PARAM: priority, extra = -20, u.arg.slot
      case Ops.BUFFER: priority = -17 if u.addrspace == AddrSpace.LOCAL else -18
      case Ops.LOAD: priority = -1    # place loads early
      case Ops.STORE: priority = 1    # place stores late
      case Ops.RANGE: priority = 5    # placing RANGE is good
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
  ret, backedge = e.src[0], tuple(x for x in e.src[1:] if x.dtype in (dtypes.void, dtypes.bool))
  for r in sorted(UOp.sink(*[x for x in e.src[1:] if x not in backedge]).ranges, key=lambda x: x.arg, reverse=True): ret = ret.end(r)
  return ret.end(*backedge) if len(backedge) else ret

pm_split_ends = PatternMatcher([
  # split the ends
  (UPat(Ops.END, name="e"), do_split_ends),
])

def ranges_to_loops(sink:UOp) -> UOp:
  # rewrite bounded ranges to bound-less loops with a register counter: i = 0; loop { body; i += 1; loop again while i < bound }
  slot = max((u.arg.slot for u in sink.toposort() if u.op is Ops.BUFFER and u.addrspace == AddrSpace.REG), default=-1) + 1
  ends = [u for u in sink.toposort() if u.op is Ops.END and any(x.op is Ops.RANGE and x.dtype is not dtypes.void for x in u.src[1:])]
  # e.ranges over-approximates nesting (it flows ranges through ordering deps), so compute true nesting from the body slices
  # NOTE: uop identity is not stable (the uop cache is weak), all lookups are by uop key
  end_for_range = {r.key: e for e in ends for r in e.src[1:] if r.op is Ops.RANGE and r.dtype is not dtypes.void}
  body_ends = {e.key: {u.key for u in e.src[0].toposort()} for e in ends}
  repl: dict[UOp, UOp] = {}
  range_to_loop: dict[bytes, UOp] = {}
  for e in ends:
    # the counter init is placed after the enclosing loops so it resets every outer iteration, the loop header depends on it so it runs first
    enclosing = tuple(r for r in e.ranges if (er:=end_for_range.get(r.key)) is not None and e.key in body_ends[er.key])
    e = e.substitute(repl)
    assert len(e.src) == 2, f"expected a split END with one range, got {len(e.src)-1} ranges"
    r = e.src[1]
    i = UOp(Ops.BUFFER, src=(UOp.const(dtypes.int, 1),), arg=ParamArg(slot, r.dtype, addrspace=AddrSpace.REG))
    slot += 1
    z = UOp.const(dtypes.int, 0)
    init = i.after(*enclosing).index(z).store(UOp.const(r.dtype, 0))
    i = i.after(init)
    # a do-while can't skip its first iteration, so a range with a possibly zero bound gets a one-time entry guard on the loop header
    guard = () if r.src[0].vmin >= 1 else (UOp.const(r.dtype, 0) < r.src[0],)
    l = range_to_loop[r.key] = UOp(Ops.RANGE, dtypes.void, src=(init,)+guard, arg=(r.arg[0], AxisType.LOOP))
    iv = i.after(l).index(z).load()
    inc = iv + UOp.const(r.dtype, 1)
    body = e.src[0].substitute({r: iv})
    # the counter store is part of the loop body, an AFTER body can't be in a GROUP so sequence it with a dep instead
    ret = body.after(i.index(z).store(inc)) if body.op is Ops.AFTER else UOp.group(body, i.index(z).store(inc))
    repl[e] = ret.end(l, inc < r.src[0])
    # keep the tracked loop headers up to date: their init deps on enclosing ranges get rewritten by the same substitution
    for k in range_to_loop: range_to_loop[k] = range_to_loop[k].substitute({r: iv})
  if not len(repl): return sink
  out = sink.substitute(repl)
  # ordering deps on the old ranges (scope AFTERs outside the loop bodies) point at the loop headers
  fix = {a: a.replace(src=(a.src[0],) + tuple(range_to_loop[s.key] if s.key in range_to_loop else s for s in a.src[1:]))
         for a in out.toposort() if a.op is Ops.AFTER and any(s.key in range_to_loop for s in a.src[1:])}
  return out.substitute(fix) if len(fix) else out
