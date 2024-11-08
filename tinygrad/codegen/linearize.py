from typing import List, Set, Dict, Tuple
import functools, heapq
from collections import defaultdict
from tinygrad.ops import type_verify, END_FOR_UOP, UOp, Ops, GroupOp
from tinygrad.dtype import dtypes
from tinygrad.helpers import DEBUG, dedup, flatten

from tinygrad.ops import PatternMatcher, UPat, graph_rewrite
class BasicBlock:
  def __init__(self, rngs, lst):
    self.rngs = tuple(rngs)
    self.lst = tuple(lst)
  def __hash__(self): return hash((self.rngs, self.lst))
  def __eq__(self, x): return self.rngs == x.rngs and self.lst == x.lst
  def __repr__(self):
    return f"{[y.arg[0] for y in self.rngs]}\n{'\n'.join([str(x.op) for x in self.lst])}"
  #def __add__(self, x):
  #  assert self.rngs == x.rngs
  #  return BasicBlock(self.rngs, self.lst+x.lst)
  def add(self, x):
    if len(x) == 0: return self
    return BasicBlock(self.rngs, tuple(x)+self.lst)

@functools.lru_cache(None)
def get_ranges_in_parents(x:UOp) -> Tuple[UOp]:
  ret = []
  for u in x.src:
    if u.op is Ops.RANGE: ret.append([u])
    # don't flow through assign and store
    if u.op is Ops.STORE: continue
    if u.op is Ops.ASSIGN:
      assert u.src[0].op is Ops.DEFINE_ACC
      ret.append([x for x in get_ranges_in_parents(u.src[1]) if x not in u.src[0].src[1:]])
    else:
      ret.append(get_ranges_in_parents(u))
  return tuple(dedup(sorted(flatten(ret), key=lambda x: x.arg)))

def append_to_block(x:UOp):
  new_srcs = []
  to_append = []
  old_blocks = {u.arg.rngs:u for u in x.src if u.op is Ops.BASICBLOCK}
  new_blocks = defaultdict(list)
  for u in x.src:
    if u.op is Ops.BASICBLOCK:
      continue
    elif u.op in {Ops.RANGE, Ops.CONST, Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.SPECIAL}: #, Ops.DEFINE_ACC}:
      new_srcs.append(u)
    else:
      if (rngs:=get_ranges_in_parents(u)) == x.arg.rngs:
        # fine to put it in this block
        new_srcs += list(u.src)
        to_append.append(u)
      elif rngs in old_blocks:
        new_blocks[rngs].extend(old_blocks[rngs].arg.lst)
        del old_blocks[rngs]
      else:
        # need to create a new block
        new_blocks[rngs].append(u)
  new_srcs = list(old_blocks.values()) + new_srcs
  if len(to_append) == 0 and len(new_blocks) == 0: return None
  for rng,lst in new_blocks.items():
    new_srcs.append(UOp(Ops.BASICBLOCK, dtypes.void, tuple(dedup(sum([x.src for x in lst], ()))), BasicBlock(rng, lst)))
  return UOp(Ops.BASICBLOCK, dtypes.void, tuple(dedup(new_srcs)), x.arg.add(to_append))

  """
  new_srcs = []
  to_append = []
  new_assigns = defaultdict(list)
  for u in x.src:
    if u.op not in {Ops.BASICBLOCK, Ops.ASSIGN, Ops.RANGE, Ops.IF}:
      new_srcs += list(u.src)
      to_append.append(u)
    elif u.op is Ops.ASSIGN:
      # ASSIGN creates a new block
      assert u.src[0].op is Ops.DEFINE_ACC
      new_assigns[u.src[0].src[1:]].append(u)
    else:
      new_srcs.append(u)
  new_new_srcs = []
  for v in new_assigns.values():
    new_new_srcs.append(UOp(Ops.BASICBLOCK, dtypes.void, tuple(dedup(sum([x.src for x in v], ()))), BasicBlock(v)))
  if len(to_append) == 0 and len(new_new_srcs) == 0: return None
  return UOp(Ops.BASICBLOCK, dtypes.void, tuple(dedup(new_new_srcs+new_srcs)), x.arg.add(to_append))
  """

make_basic_blocks = PatternMatcher([
  (UPat(Ops.SINK, name="x"), lambda x: UOp(Ops.BASICBLOCK, dtypes.void, x.src, BasicBlock(get_ranges_in_parents(x), [x]))),
  (UPat(Ops.BASICBLOCK, name="x"), append_to_block),
])

def get_children_dfs(u:UOp, children:Dict[UOp, List[UOp]], srcs:Dict[UOp, Dict[UOp, None]], in_degree:Dict[UOp, int]):
  if u in children: return srcs[u]
  srcs[u] = {}
  children[u] = []
  for x in u.src:
    srcs[u].update(get_children_dfs(x, children, srcs, in_degree))
    if x.op is Ops.RANGE and x.arg[1]: srcs[u][x] = None
    children[x].append(u)
  in_degree[u] = len(u.src)
  return srcs[u]

def linearize_uop(sink:UOp, skip_check:bool=not __debug__) -> List[UOp]:
  assert sink.op is Ops.SINK, f"sink isn't sink, it's {sink.op}"

  sink_bb = graph_rewrite(sink, make_basic_blocks)

  # filter nodes that don't link to a sink
  # BFS toposort
  children: Dict[UOp, List[UOp]] = {}
  range_srcs: Dict[UOp, Dict[UOp, None]] = {}
  in_degree: Dict[UOp, int] = {}
  get_children_dfs(sink, children, range_srcs, in_degree)

  @functools.lru_cache(None)
  def get_recursive_children(x:UOp, end:Ops, include_self=False) -> Set[UOp]:
    if x.op is Ops.SINK: return set()
    return set.union({x} if include_self else set(), *([get_recursive_children(u, end, True) for u in children[x] if x.op is not end]))

  # scope children impact the toposort and END* insertion
  scope_children = {p:get_recursive_children(p, END_FOR_UOP[p.op][0]) for p in reversed(in_degree) if p.op in END_FOR_UOP}
  range_phi = {r:[p for p in scope_children[r] if p.op is Ops.ASSIGN] for r in scope_children if r.op is Ops.RANGE}

  # assign priorities
  def get_priority(u:UOp):
    priority = 0
    # prefer ranges that depend on the least number of independent ranges
    if u.op is Ops.RANGE and u.arg[1]:
      priority += u.arg[0]
      for p in range_phi[u]:
        priority += 10000*len([r for r in range_srcs[p] if not any(i in range_phi[u] for i in range_phi[r])])
    elif u.op is Ops.CONST:
      # place consts first here, they don't do anything and it can cause issues with DEFINE_ACC
      priority -= 100000000000
    else:
      # prefer uops that are loop children
      priority -= sum([(l.arg[0]+1) + 1000*l.arg[1] for l,ss in scope_children.items() if l.op is Ops.RANGE and u in ss])
    if u.op is Ops.IF and len(u.src) == 1: priority += 10000000 # if penalty
    return priority
  priorities:Dict[UOp, int] = {u:get_priority(u) for u in children}

  # prevent priority inversion
  @functools.lru_cache(None)
  def fix_priority(u:UOp, lowest_priority):
    if u.op in {Ops.CAST, Ops.BITCAST, *GroupOp.ALU, Ops.VECTORIZE, Ops.GEP, Ops.SPECIAL, Ops.DEFINE_LOCAL, Ops.LOAD}:
      priorities[u] = min(priorities[u], lowest_priority)
      if u.op is Ops.LOAD: priorities[u] += 100 # load penalty (here)
    for x in u.src: fix_priority(x, priorities[u])
  fix_priority(sink, 0)

  # NOTE: the compare should never make it all the way to u
  queue:List[Tuple[int, Tuple, UOp]] = []
  def push(u:UOp): heapq.heappush(queue, (priorities[u], u.tuplize, u))

  for u in children:
    if in_degree[u] == 0: push(u)

  scope_end: Dict[UOp, UOp] = {}
  _uops: List[UOp] = []
  while queue:
    p,_,x = heapq.heappop(queue)
    print(x.op, [y.arg[0] for y in get_ranges_in_parents(x)])
    if DEBUG >= 7: print(f"{p:5d}", x.op, x.dtype, x.arg)
    if x in scope_children: scope_end[x] = x
    if x.op is Ops.DEFINE_ACC:
      idx = min([_uops.index(l) for l in x.src if l.op is Ops.RANGE])
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
