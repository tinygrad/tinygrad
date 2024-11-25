from typing import List, Set, Dict, Tuple, DefaultDict
from collections import defaultdict
import functools, heapq
from tinygrad.ops import type_verify, END_FOR_UOP, UOp, Ops, GroupOp, PatternMatcher, UPat, graph_rewrite
from tinygrad.dtype import dtypes
from tinygrad.helpers import DEBUG, dedup, flatten

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

def disp(y): return y.arg[0] if y.op is Ops.RANGE else f'IF{id(y)}'
class BasicBlock:
  def __init__(self, ctx, lst, ends=[]):
    self.ctx, self.lst, self.ends = ctx, lst, ends
  def __repr__(self):
    return f"{[disp(y) for y in self.ctx]} {[disp(y) for y in self.ends] if len(self.ends) else ""} "+\
           f"{len(self.lst)}" + "\n" + '\n'.join([str(x.op) for x in self.lst])

@functools.lru_cache(None)
def get_block_ctx(x:UOp) -> Tuple[UOp, ...]:
  ret: List[Tuple[UOp, ...]] = []
  for u in x.src:
    if u.op in {Ops.RANGE, Ops.IF}: ret.append((u,))
    # don't flow through assign and store
    if u.op is Ops.STORE: continue
    if u.op is Ops.ASSIGN:
      assert u.src[0].op is Ops.DEFINE_ACC
      ret.append(tuple(x for x in get_block_ctx(u.src[1]) if x not in u.src[0].src[1:]))
    else:
      ret.append(get_block_ctx(u))
  return tuple(dedup(sorted(flatten(ret), key=lambda x: x.tuplize)))

DONT_PLACE_IN_BLOCK = {Ops.RANGE, Ops.ENDRANGE, Ops.CONST,
                       Ops.DEFINE_ACC, Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.BLOCK, Ops.BLOCKEND}
def append_to_block(ctx, x:UOp):
  new_srcs = []
  to_append = []
  new_blocks: DefaultDict[Tuple[UOp, ...], List[UOp]] = defaultdict(list)
  updated = False
  in_this_block = set(x.arg.lst)
  for u in x.src:
    if u.op in DONT_PLACE_IN_BLOCK or len([x for x in ctx[u] if x not in in_this_block]) > 0: new_srcs.append(u)
    elif (block_ctx:=get_block_ctx(u)) == x.arg.ctx:
      # if it's the same context, we place in this block and append the parents
      new_srcs += list(u.src)
      to_append.append(u)
      updated = True
    else:
      # otherwise, we create a new block with this
      new_blocks[block_ctx].append(u)
      updated = True
  if not updated: return None
  for rng,lst in new_blocks.items():
    new_block = UOp(Ops.BLOCK, dtypes.void, tuple(dedup(flatten(y.src for y in lst))), BasicBlock(rng, lst))
    lrng = list(rng)
    for r in rng[::-1]:
      if r not in x.arg.ctx:
        new_block = UOp(Ops.BLOCKEND, dtypes.void, (new_block,), BasicBlock(lrng, [UOp(Ops.ENDRANGE, dtypes.void, (r,))], [r]))
        lrng = lrng[:]
        lrng.remove(r)
    new_srcs.append(new_block)
  return UOp(Ops.BLOCK, dtypes.void, tuple(dedup(new_srcs)), BasicBlock(x.arg.ctx, to_append+x.arg.lst))

make_basic_blocks = PatternMatcher([
  (UPat(Ops.SINK, name="x"), lambda x: UOp(Ops.BLOCKEND, src=(UOp(Ops.BLOCK, src=x.src, arg=BasicBlock([], [x])),), arg=BasicBlock([], [], []))),
  (UPat(Ops.BLOCK, name="x"), append_to_block),
])

def blockend_gobble(x:UOp):
  pass

blockend = PatternMatcher([
  (UPat(Ops.BLOCKEND, name="x"), blockend_gobble),
])

def linearize_uop(sink:UOp, skip_check:bool=not __debug__) -> List[UOp]:
  assert sink.op is Ops.SINK, f"sink isn't sink, it's {sink.op}"
  # filter nodes that don't link to a sink
  # BFS toposort
  children: Dict[UOp, List[UOp]] = {}
  range_srcs: Dict[UOp, Dict[UOp, None]] = {}
  in_degree: Dict[UOp, int] = {}
  get_children_dfs(sink, children, range_srcs, in_degree)

  sink = graph_rewrite(sink, make_basic_blocks, ctx=children)
  # TODO: combine matching BLOCKENDS
  sink = graph_rewrite(sink, blockend)


  @functools.lru_cache(None)
  def topoplace(u:UOp) -> List[UOp]: return flatten(topoplace(x) for x in u.src) + (u.arg.lst if u.op in {Ops.BLOCK, Ops.BLOCKEND} else [u])

  _uops = topoplace(sink)

  from tinygrad.ops import print_uops
  print_uops(_uops)

  # sanity checks (NOTE: these can cause things to be skipped in BEAM)
  #if not skip_check: type_verify(_uops)

  # strip the SINK
  return _uops[:-1]

  """
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
  """
