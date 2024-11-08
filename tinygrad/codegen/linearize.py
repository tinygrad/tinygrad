from typing import List, Set, Dict, Tuple, DefaultDict
import functools, heapq
from collections import defaultdict
from tinygrad.ops import type_verify, END_FOR_UOP, UOp, Ops, GroupOp
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

BlockType = Tuple[Tuple[int, ...], Tuple[int, ...]]
def order_blocks(blocks:List[BlockType]) -> List[BlockType]:
  # order the loops
  loop_deps: DefaultDict[int, List[int]] = defaultdict(list)
  for b in blocks:
    for o in b[0]: loop_deps[o] = sorted(dedup(loop_deps[o] + [x for x in b[0] if x != o] + list(b[1])))
  #print(loop_deps)
  loop_order: List[int] = []
  while len(loop_order) < len(loop_deps):
    to_place = []
    others = []
    for k,v in sorted(loop_deps.items()):
      # already placed
      if k in loop_order: continue
      if all(x in loop_order for x in v): to_place.append(k)
      else: others.append(k)
    if len(to_place) == 0:
      # sad panda :(
      loop_order.append(others[0])
    else:
      loop_order += to_place
  #print(loop_order)
  open_loops: List[int] = []
  seen_loops: List[int] = []
  placed_blocks: List[BlockType] = []
  blocks = sorted(blocks, key=lambda x: (len(x[1]), x[1], len(x[0]), x[0]))
  while len(blocks):
    # can we place any blocks without opening or closing loops
    for b in blocks:
      if b in placed_blocks: continue
      if all(x in open_loops for x in b[0]) and all(x in seen_loops for x in b[1]):
        placed_blocks.append(b)
    if len(placed_blocks) == len(blocks): break
    # see if we can close any no longer required loops
    loops_still_required = flatten([b[0] for b in blocks if b not in placed_blocks])
    closable_loops = [x for x in open_loops if x not in loops_still_required]
    if len(closable_loops):
      # there are closable loops
      open_loops = [x for x in open_loops if x not in closable_loops]
      seen_loops += closable_loops
    else:
      # we have to open a loop because no blocks are currently placable
      open_loops.append(loop_order.pop(0))
  #for x in blocks: print(x)
  return placed_blocks

def linearize_uop(sink:UOp, skip_check:bool=not __debug__) -> List[UOp]:
  assert sink.op is Ops.SINK, f"sink isn't sink, it's {sink.op}"

  # break uops into basic blocks
  blocks: DefaultDict[BlockType, List[UOp]] = defaultdict(list)
  @functools.lru_cache(None)
  def place_with_scope(u:UOp, rng:Tuple[int, ...]=()):
    if u.op is Ops.ASSIGN:
      assert u.src[0].op is Ops.DEFINE_ACC
      rng = tuple(sorted(dedup(rng+tuple(x.arg[0] for x in u.src[0].src[1:]))))
    if u.op is Ops.STORE:
      rng = tuple(sorted(dedup(rng+tuple(x.arg[0] for x in u.src[0].sparents if x.op is Ops.RANGE))))
    parent_rng = tuple(sorted([x.arg[0] for x in u.sparents if x.op is Ops.RANGE]))
    loop_inside = tuple(x for x in rng if x in parent_rng)
    blocks[(loop_inside, tuple(x for x in parent_rng if x not in loop_inside))].append(u)
    for x in u.src: place_with_scope(x, rng)
  place_with_scope(sink)

  #for block,ops in blocks.items():
  #for block in order_blocks(list(blocks.keys())):
    #ops = blocks[block]
    #print(block, len(ops))
    #from tinygrad.ops import print_uops
    #print_uops(dedup(sorted(ops, key=lambda x: x.tuplize)))

  block_priority: DefaultDict[UOp, int] = defaultdict(lambda: -1)
  for block_num, block in enumerate(order_blocks(list(blocks.keys()))):
    if DEBUG >= 5: print(block)
    for u in blocks[block]:
      if block_priority[u] == -1: block_priority[u] = block_num

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
  queue:List[Tuple[int, int, Tuple, UOp]] = []
  def push(u:UOp): heapq.heappush(queue, (block_priority[u], priorities[u], u.tuplize, u))

  for u in children:
    if in_degree[u] == 0: push(u)

  scope_end: Dict[UOp, UOp] = {}
  _uops: List[UOp] = []
  while queue:
    bp,p,_,x = heapq.heappop(queue)
    if DEBUG >= 7: print(f"{bp:3d} {p:5d}", x.op, x.dtype, x.arg)
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
