import heapq, functools
from typing import Any
from collections import defaultdict
from tinygrad.uop.ops import UOp, Ops, GroupOp, multirange_str, consumer_map_from_toposort
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import prod, getenv, TUPLE_ORDER

def linearize2(sink:UOp) -> list[UOp]:
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







# the lowest common ancestor of the blocks
def lca(blocks:tuple[UOp|None, ...]) -> UOp|None:
  def _lca(a:UOp|None, b:UOp|None) -> UOp|None:
    while a is not b:
      if block_depth(a) >= block_depth(b): a = idom(a)
      else: b = idom(b)
    return a
  return functools.reduce(_lca, blocks)

# the immediate dominator of the block
@functools.cache
def idom(x:UOp|None) -> UOp|None:
  if x is None: return None
  # ENDIF merges multiple blocks so idom is their lca
  if x.op is Ops.ENDIF: return lca(x.src)
  if x.op is Ops.START: return None
  return x.src[0]

@functools.cache
def block_depth(x:UOp|None) -> int:
  if x is None: return 0
  return block_depth(idom(x)) + 1

@functools.cache
def loop_depth(x:UOp|None) -> int:
  if x is None: return 0
  if x.op is Ops.RANGE: return loop_depth(idom(x)) + 1
  if x.op is Ops.END: return loop_depth(idom(x)) - 1
  return loop_depth(idom(x))

def linearize(sink:UOp) -> list[UOp]:
  topo = sink.toposort()
  users = consumer_map_from_toposort(topo)
  sched: dict[UOp, UOp|None] = {}
  cfg: dict[UOp|None, list[UOp]] = {None: []}

  # early schedule, here we find the earliest/highest block u can go in
  for u in topo:
    # control flow ops are pinned to themselves
    if u.op in GroupOp.ControlFlow: cfg[u] = []
    # the highest block for u is the lowest block of all its srcs
    else: sched[u] = max((sched.get(s, s) for s in u.src), key=block_depth, default=None)

  # late schedule, here we find the latest/lowest block u can go in, this is the lowest block that dominates all uses of u
  # then we pick the best block in the range of earliest to latest
  for u in reversed(sched):
    best = last = lca(tuple(sched.get(s, idom(s) if s.op in (Ops.START, Ops.END, Ops.SINK) else s) for s in users[u]))
    # we pick the block with lowest loop nest and most depth, so we hoist out of loops and into branches
    while True:
      if loop_depth(last) < loop_depth(best) or block_depth(last) > block_depth(best) or best is not None and best.op is Ops.IF: best = last
      if last is sched[u]: break
      assert last is not None
      last = idom(last)
    #print(u.op, sched[u].op if sched[u] is not None else None, best.op if best is not None else None)
    sched[u] = best

  # get the uops in each block
  for k,v in sched.items(): cfg[v].append(k)
  print("BEFORE BLOCK SCHEDULE")
  for k,v in cfg.items():
    print("BLOCK: ", (k.op, k.arg) if k is not None else None)
    for x in v: print("  ", x.op)

  # schedule the uops in each block
  for k in cfg:
    cfg[k] = block_linearize(cfg[k], users)

  print("AFTER BLOCK SCHEDULE")
  for k,v in cfg.items():
    print("BLOCK: ", (k.op, k.arg) if k is not None else None)
    for x in v: print("  ", x.op)

  ret = []
  for k,v in cfg.items(): ret.extend(([k] if k is not None else []) + v)
  return ret

def block_linearize(lst:list[UOp], users:dict[UOp, dict[UOp, None]]) -> list[UOp]:
  count:dict[UOp, int] = {}
  for u in lst: count[u] = len([s for s in u.src if s in count])

  # number the uops in "ideal" order
  nkey = {u:i for i,u in enumerate(sorted(lst, key=lambda x: x.tuplize if TUPLE_ORDER else ()))}

  # then force them to be toposorted in as close to the ideal order as possible
  heap = [(nkey[u],u) for u in lst if count[u] == 0]
  heapq.heapify(heap)
  newlst = []
  while heap:
    newlst.append(u:=heapq.heappop(heap)[1])
    for v in users[u]:
      if v not in count: continue
      count[v] -= 1
      if count[v] == 0: heapq.heappush(heap, (nkey[v],v))
  return newlst