from typing import List, Set, Dict, Tuple, Any, Optional
import functools, heapq
from tinygrad.ops import type_verify, END_FOR_UOP, UOp, UOps
from tinygrad.dtype import dtypes, DType
from tinygrad.helpers import DEBUG

def get_children_dfs(u:UOp, children:Dict[UOp, List[UOp]], srcs:Dict[UOp, Dict[UOp, None]], in_degree:Dict[UOp, int]):
  if u in children: return srcs[u]
  srcs[u] = {}
  children[u] = []
  for x in u.src:
    srcs[u].update(get_children_dfs(x, children, srcs, in_degree))
    if x.op is UOps.RANGE and x.arg[1]: srcs[u][x] = None
    children[x].append(u)
  in_degree[u] = len(u.src)
  return srcs[u]

def linearize_uop(sink:UOp, skip_check:bool=not __debug__) -> List[UOp]:
  assert sink.op is UOps.SINK, f"sink isn't sink, it's {sink.op}"
  # filter nodes that don't link to a sink
  # BFS toposort
  children: Dict[UOp, List[UOp]] = {}
  range_srcs: Dict[UOp, Dict[UOp, None]] = {}
  in_degree: Dict[UOp, int] = {}
  get_children_dfs(sink, children, range_srcs, in_degree)

  @functools.lru_cache(None)
  def get_recursive_children(x:UOp, end:UOps, include_self=False) -> Set[UOp]:
    if x.op is UOps.SINK: return set()
    return set.union({x} if include_self else set(), *([get_recursive_children(u, end, True) for u in children[x] if x.op is not end]))

  # scope children impact the toposort and END* insertion
  scope_children = {p:get_recursive_children(p, END_FOR_UOP[p.op][0]) for p in reversed(in_degree) if p.op in END_FOR_UOP}
  range_phi = {r:[p for p in scope_children[r] if p.op is UOps.ASSIGN] for r in scope_children if r.op is UOps.RANGE}

  # assign priorities
  def get_priority(u:UOp):
    priority = 0
    # prefer ranges that depend on the least number of independent ranges
    if u.op is UOps.RANGE and u.arg[1]:
      priority += u.arg[0]
      for p in range_phi[u]:
        priority += 10000*len([r for r in range_srcs[p] if not any(i in range_phi[u] for i in range_phi[r])])
    # prefer uops that are loop children
    else:
      priority -= sum([(l.arg[0]+1) + 1000*l.arg[1] for l,ss in scope_children.items() if l.op is UOps.RANGE and u in ss])
    if u.op is UOps.IF and len(u.src) == 1: priority += 10000000 # if penalty
    return priority
  priorities:Dict[UOp, int] = {u:get_priority(u) for u in children}

  # prevent priority inversion
  @functools.lru_cache(None)
  def fix_priority(u:UOp, lowest_priority):
    if u.op in {UOps.CAST, UOps.BITCAST, UOps.ALU, UOps.VECTORIZE, UOps.GEP, UOps.SPECIAL, UOps.DEFINE_LOCAL, UOps.LOAD}:
      priorities[u] = min(priorities[u], lowest_priority)
      if u.op is UOps.LOAD: priorities[u] += 100 # load penalty (here)
    for x in u.src: fix_priority(x, priorities[u])
  fix_priority(sink, 0)

  @functools.lru_cache(None)
  def tuplize(u:UOp) -> Tuple[int, Any, Optional[DType], Tuple]:
    # NOTE: this sort of DEFINE_VAR shouldn't have to be here. only for PTX
    if u.op is UOps.DEFINE_VAR: arg = u.arg[0]
    elif u.op is UOps.ALU: arg = u.arg.value
    else: arg = u.arg
    return (u.op.value, arg, u.dtype, tuple(tuplize(x) for x in u.src))

  # NOTE: the compare should never make it all the way to u
  queue:List[Tuple[int, Tuple, UOp]] = []
  def push(u:UOp): heapq.heappush(queue, (priorities[u], tuplize(u), u))

  for u in children:
    if in_degree[u] == 0: push(u)

  scope_end: Dict[UOp, UOp] = {}
  _uops: List[UOp] = []
  while queue:
    p,_,x = heapq.heappop(queue)
    if DEBUG >= 7: print(f"{p:5d}", x.op, x.dtype, x.arg)
    if x in scope_children: scope_end[x] = x
    if x.op is UOps.DEFINE_ACC:
      idx = min([_uops.index(l) for l in x.src if l.op is UOps.RANGE])
      _uops.insert(idx, x)
    else: _uops.append(x)
    for u, ss in scope_children.items():
      if x in ss:
        ss.remove(x)
        if len(ss) == 0: scope_end[u] = x
    for u in children[x]:
      in_degree[u] -= 1
      if in_degree[u] == 0: push(u)

  # end scopes in toposort order
  for u, x in scope_end.items(): _uops.insert(_uops.index(x)+1, UOp(END_FOR_UOP[u.op][1], dtypes.void, (u,)))

  # sanity checks (NOTE: these can cause things to be skipped in BEAM)
  if not skip_check: type_verify(_uops)

  # strip the SINK
  return _uops[:-1]
