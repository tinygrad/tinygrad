from typing import List, Set, Dict, Tuple, DefaultDict
import functools, heapq
from collections import defaultdict
from dataclasses import dataclass
from tinygrad.ops import type_verify, END_FOR_UOP, UOp, Ops, GroupOp, print_uops
from tinygrad.dtype import dtypes
from tinygrad.helpers import DEBUG, dedup

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

@dataclass(eq=True, unsafe_hash=True)
class Block:
  loop_inside: Tuple[UOp, ...]
  loop_after: Tuple[UOp, ...]
  @property
  def key(self): return len(self.loop_after), [x.arg for x in self.loop_after], len(self.loop_inside), [x.arg for x in self.loop_inside]
  def __lt__(self, o): return self.key < o.key
  def __repr__(self): return f"{[x.arg for x in self.loop_inside]}-{[x.arg for x in self.loop_after]}"

def linearize_uop(sink:UOp, skip_check:bool=not __debug__) -> List[UOp]:
  assert sink.op is Ops.SINK, f"sink isn't sink, it's {sink.op}"

  # break uops into basic blocks
  blocks: DefaultDict[Block, List[UOp]] = defaultdict(list)
  placed: Dict[UOp, Block] = {}
  @functools.lru_cache(None)
  def place_with_scope(u:UOp, rng:Tuple[UOp]=()):
    if u.op is Ops.ASSIGN:
      assert u.src[0].op is Ops.DEFINE_ACC
      rng = rng+u.src[0].src[1:]
    parent_rng = tuple(sorted([x for x in u.sparents if x.op is Ops.RANGE], key=lambda x: x.tuplize))
    # (<loop inside>, <loop after>)
    block = Block(tuple(x for x in rng if x in parent_rng), parent_rng)
    blocks[block].append(u)
    placed[u] = block
    for x in u.src: place_with_scope(x, rng)
  place_with_scope(sink)

  """
  open_loops = set()
  seen_loops = set()

  # block placement
  placed_blocks = []
  while len(placed_blocks) < len(blocks):
    placable_blocks = []
    for block in blocks.keys():
      if block in placed_blocks: continue
      if not all(x in open_loops for x in block.loop_inside) or not all(x in seen_loops for x in block.loop_after): continue
      placable_blocks.append(block)
    if len(placable_blocks) > 0:
      placed_blocks.append(placable_blocks[0])
      continue

    # we have to open a loop
    placable_blocks_with_open = []
    for block in blocks.keys():
      if block in placed_blocks: continue
      if not all(x in seen_loops for x in block.loop_after): continue
      placable_blocks_with_open.append(block)
    placable_blocks_with_open = sorted(placable_blocks_with_open)
    if len(placable_blocks_with_open) > 0:
      for x in placable_blocks_with_open[0].loop_inside: open_loops.add(x)
      placed_blocks.append(placable_blocks_with_open[0])
      continue
    raise RuntimeError("no placable blocks")
  """

  #for block in placed_blocks:
  for block in sorted(blocks.keys()):
    v = blocks[block]
    print(block, len(v))
    v = sorted(dedup(v), key=lambda x: x.tuplize)
    print_uops(v)


  """
  # get block deps
  deps = {b:set() for b in blocks}
  for block,ops in blocks.items():
    for u in ops:
      for s in u.src:
        if placed[s] != block: deps[block].add(placed[s])

  for block,v in sorted(blocks.items()):
    print(block, len(v))
    #print(deps[block])
    v = sorted(dedup(v), key=lambda x: x.tuplize)
    print_uops(v)
  """


  """
  ops_to_range = {u:tuple(sorted([x for x in u.sparents if x.op is Ops.RANGE], key=lambda x: x.tuplize)) for u in sink.sparents}
  ranges_to_ops = defaultdict(list)
  for k,v in ops_to_range.items(): ranges_to_ops[v].append(k)
  for k,v in ranges_to_ops.items():
    print([x.arg for x in k], len(v))
  """

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
