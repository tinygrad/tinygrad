from tinygrad.uop import Ops
from tinygrad.uop.ops import AddrSpace, PatternMatcher, UPat, UOp

# heuristics:
#   - scheduled by fixed shared operand (A) of WMMAs
#   - track other load dependencies and schedule lazily
#   - also track WMMA reg stores to schedule at end of each group
#   to end reg lifetimes (bit of a hack, would be solved if reg buf semantics were correct)
class WMMASchedulePolicy:
  def __init__(self, sink:UOp):
    self.schedule_past: dict[UOp, tuple[UOp,...]] = {}

    wmma_deps: dict[UOp, list[set[UOp]]] = {}
    wmma_consumers: dict[UOp, list[UOp]] = {}
    # find loads used by wmma
    for u in sink.toposort():
      if u.op is Ops.WMMA:
        for s in u.src:
          deps = set(l for l in s.toposort() if l.op is Ops.LOAD and l.src[0].addrspace is not AddrSpace.REG)
          wmma_deps.setdefault(u, []).append(deps)
      # dont rely on store? use last src edge to the WMMA?
      if u.op is Ops.STORE:
        if u.src[1].op is Ops.INDEX:
          w = u.src[1].src[0]
          if w.op is not Ops.WMMA: continue
          wmma_consumers.setdefault(w, []).append(u)
    if len(wmma_deps) < 4: return

    sched_groups: dict[frozenset[UOp], list[UOp]] = {}
    for w in wmma_deps.keys():
      sched_groups.setdefault(frozenset(wmma_deps[w][0]), []).append(w)

    # transitive, only schedule past last block not all preceding
    carry: list[UOp] = []
    scheduled: set[UOp] = set()
    for i,g in enumerate(sched_groups.values()):
      deps = set()
      for w in g:
        for ls in wmma_deps[w]: deps.update(ls)
      if i > 0:
        for l in deps:
          if l not in scheduled: self.schedule_past[l] = tuple(carry)
      scheduled.update(deps)
      carry = list(g)
      for w in g: carry.extend(wmma_consumers[w])

pm_schedule_interleave_wmma = PatternMatcher([
  (UPat((Ops.LOAD, Ops.STORE), name="x"), lambda ctx,x: x.replace(src=(x.src[0].after(*ctx.schedule_past[x]),)+x.src[1:]) if x in ctx.schedule_past else None),
])
