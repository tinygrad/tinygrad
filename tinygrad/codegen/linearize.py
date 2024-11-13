from __future__ import annotations
from typing import List, Dict, Tuple, DefaultDict
import functools, heapq
from collections import defaultdict
from tinygrad.ops import type_verify, UOp, Ops
from tinygrad.dtype import dtypes
from tinygrad.helpers import dedup, flatten, DEBUG

from tinygrad.ops import PatternMatcher, UPat, graph_rewrite
class BasicBlock:
  def __init__(self, rngs, lst):
    self.rngs = tuple(rngs)
    self.lst = tuple(lst)
  def __hash__(self): return hash((self.rngs, self.lst))
  def __eq__(self, x):
    if x is None: return False
    return self.rngs == x.rngs and self.lst == x.lst
  def __repr__(self):
    return f"{[y.arg[0] if y.op is Ops.RANGE else f'IF{id(y)}' for y in self.rngs]} {len(self.lst)}" + "\n" + '\n'.join([str(x.op) for x in self.lst])
  @functools.cached_property
  def lst_tuplize(self): return tuple(y.tuplize for y in self.lst)
  @functools.cached_property
  def rngs_tuplize(self): return tuple(y.tuplize for y in self.rngs)
  def __lt__(self, x:BasicBlock):
    if self.rngs == x.rngs: return False # self.lst_tuplize < x.lst_tuplize
    return self.rngs_tuplize < x.rngs_tuplize
  def add(self, x):
    if len(x) == 0: return self
    return BasicBlock(self.rngs, tuple(x)+self.lst)

@functools.lru_cache(None)
def get_ranges_in_parents(x:UOp) -> Tuple[UOp, ...]:
  ret: List[Tuple[UOp, ...]] = []
  for u in x.src:
    if u.op in {Ops.RANGE, Ops.IF}: ret.append((u,))
    # don't flow through assign and store
    if u.op is Ops.STORE: continue
    if u.op is Ops.ASSIGN:
      assert u.src[0].op is Ops.DEFINE_ACC
      ret.append(tuple(x for x in get_ranges_in_parents(u.src[1]) if x not in u.src[0].src[1:]))
    else:
      ret.append(get_ranges_in_parents(u))
  return tuple(dedup(sorted(flatten(ret), key=lambda x: x.tuplize)))

DONT_PLACE_IN_BLOCK = {Ops.RANGE, Ops.CONST, Ops.DEFINE_ACC, Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.BLOCK, Ops.BLOCKEND, Ops.BLOCKIF, Ops.BLOCKFORK}
def append_to_block(ctx, x:UOp):
  children, forks = ctx
  new_srcs = []
  to_append = []
  new_blocks: DefaultDict[Tuple[UOp, ...], List[UOp]] = defaultdict(list)
  updated = False
  block_uop_set = set(x.arg.lst)
  pend_if = None
  for u in x.src:
    if u.op is Ops.BLOCK:
      return None
      #if len(new_block_list:=new_blocks[u.arg.rngs]): updated = True
      #new_block_list.append(u)
    elif u in forks:
      updated = True
      #new_blocks[get_ranges_in_parents(u)].append(u)
      new_srcs.append(forks[u])
    elif u.op is Ops.IF:
      pend_if = u
      blk = UOp(Ops.BLOCK, dtypes.void, u.src, BasicBlock(get_ranges_in_parents(u), []))
      #new_srcs.append(blk)
      new_srcs.append(UOp(Ops.BLOCKIF, dtypes.void, (blk,), BasicBlock(get_ranges_in_parents(u)+(u,), [u])))
    elif u.op in DONT_PLACE_IN_BLOCK:
      # it stays in srcs if it has children not in the basic or is RANGE/CONST
      new_srcs.append(u)
    elif (len([y for y in children[u] if y not in block_uop_set]) and u not in forks):
      new_srcs.append(u)
      #blk = UOp(Ops.BLOCK, dtypes.void, u.src, BasicBlock(get_ranges_in_parents(u), []))
      #new_srcs.append(UOp(Ops.BLOCKFORK, dtypes.void, (blk,), BasicBlock(get_ranges_in_parents(u)+(u,), [u])))
      #pass
    else:
      updated = True
      if (rngs:=get_ranges_in_parents(u)) == x.arg.rngs: # and u not in forks:
        # fine to put it in this block
        new_srcs += list(u.src)
        to_append.append(u)
      else:
        # need to create a new block
        new_blocks[rngs].append(u)
  if not updated: return None
  for rng,lst in new_blocks.items():
    if len(lst) == 1 and lst[0].op is Ops.BLOCK: new_src = lst[0]
    else:
      new_lst = flatten([y.arg.lst if y.op is Ops.BLOCK else [y] for y in lst])
      new_src = UOp(Ops.BLOCK, dtypes.void, tuple(dedup(flatten(y.src for y in lst))), BasicBlock(rng, new_lst))
    closed = tuple([y for y in new_src.arg.rngs if y not in x.arg.rngs])
    if len(closed):
      # TODO: is this order always right?
      for c in closed[::-1]: new_src = UOp(Ops.BLOCKEND, dtypes.void, (new_src,), arg=BasicBlock([c], []))
    new_srcs.append(new_src)
  if pend_if is not None:
    new_srcs = [UOp(Ops.BLOCKIF, dtypes.void, tuple(dedup(new_srcs)), BasicBlock(get_ranges_in_parents(u), [u]))]
  return UOp(Ops.BLOCK, dtypes.void, tuple(dedup(new_srcs)), x.arg.add(to_append))


def simple_append_to_block(ctx, x:UOp):
  # we are in a block
  children, ranges, placed = ctx
  new_blocks: DefaultDict[Tuple[UOp, ...], List[UOp]] = defaultdict(list)
  new_srcs = []
  to_append = []
  updated = False
  for u in x.src:
    if u.op in DONT_PLACE_IN_BLOCK:
      # it stays in the block srcs if it's an unplaced type (including a parent block)
      new_srcs.append(u)
      continue

    if u.op is Ops.IF:
      updated = True
      blk = UOp(Ops.BLOCK, dtypes.void, u.src, BasicBlock(get_ranges_in_parents(u), []))
      placed[u] = blk.arg
      blk_if = UOp(Ops.BLOCKIF, dtypes.void, (blk,), BasicBlock(get_ranges_in_parents(u)+(u,), [u]))
      new_srcs.append(blk_if)
      continue

    # see where it's children are placed
    children_block_location = dedup([placed.get(c, None) for c in children[u]])
    if len(children_block_location) == 1 and children_block_location[0] == x.arg:
      # if all children are in this block
      updated = True
      if (rngs:=ranges[u]) == x.arg.rngs:
        # and it's the same range, we place it in this block
        new_srcs += list(u.src)
        to_append.append(u)
      else:
        # and it's a different range, we place it in a new block
        new_blocks[rngs].append(u)
    elif None in children_block_location:
      # this has an unplaced child, it stays in srcs
      print(f"unplaced {u.op}")
      new_srcs.append(u)
    else:
      #print(f"FAILURE on {u.op} len {len(children_block_location)} {id(children_block_location[0])}, {id(x)}")
      #new_srcs.append(u)
      updated = True
      # this has two children in different blocks, create a new (empty) block to replace it

      new_srcs.append(UOp(Ops.BLOCKFORK, dtypes.void, (u,)))
      #new_srcs.append(UOp(Ops.BLOCK, dtypes.void, (u,), BasicBlock(ranges[u], [])))

  if not updated: return None

  # create new blocks, sometimes ending things
  for rng,lst in new_blocks.items():
    new_src = UOp(Ops.BLOCK, dtypes.void, tuple(dedup(flatten(y.src for y in lst))), BasicBlock(rng, lst))
    for l in new_src.arg.lst: placed[l] = new_src.arg
    closed = tuple([y for y in new_src.arg.rngs if y not in x.arg.rngs])
    if len(closed):
      # TODO: is this order always right?
      for c in closed[::-1]: new_src = UOp(Ops.BLOCKEND, dtypes.void, (new_src,), arg=BasicBlock([c], []))
    new_srcs.append(new_src)

  ret = UOp(Ops.BLOCK, dtypes.void, tuple(dedup(new_srcs)), x.arg.add(to_append))
  for l in ret.arg.lst: placed[l] = ret.arg
  return ret

def place_sink(ctx, x:UOp):
  children, ranges, placed = ctx
  ret = UOp(Ops.BLOCK, dtypes.void, x.src, BasicBlock([], [x]))
  placed[x] = ret.arg
  return ret

make_basic_blocks = PatternMatcher([
  (UPat(Ops.SINK, name="x"), place_sink),
  #(UPat(Ops.BLOCK, name="x"), append_to_block),
  (UPat(Ops.BLOCK, name="x"), simple_append_to_block),
])

def fix_fork(ctx, x):
  assert len(x.src) == 1
  if x.src[0].op is Ops.BLOCK: return None

  children, ranges, placed = ctx
  ret = UOp(Ops.BLOCK, dtypes.void, x.src[0].src, BasicBlock(ranges[x.src[0]], [x.src[0]]))
  placed[x.src[0]] = ret.arg
  return x.replace(src=(ret,))

fix_forks = PatternMatcher([
  (UPat(Ops.BLOCKFORK, name="x"), fix_fork),
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

def block_uop(sink:UOp) -> UOp:
  children: Dict[UOp, List[UOp]] = {}
  range_srcs: Dict[UOp, Dict[UOp, None]] = {}
  in_degree: Dict[UOp, int] = {}
  get_children_dfs(sink, children, range_srcs, in_degree)

  ranges = {u:get_ranges_in_parents(u) for u in sink.sparents}

  # find ifs
  # if you aren't in a parent of the IF, you are in the IF (FALSE!)
  """
  for u in sink.sparents:
    if u.op is Ops.IF:
      not_in_if = u.sparents
      in_if = [s for s in sink.parents if s not in not_in_if]
      for x in in_if:
        ranges[x] = tuple(dedup(sorted(ranges[x] + (u,), key=lambda y: y.tuplize)))
  """

  placed = {}
  sink = graph_rewrite(sink, make_basic_blocks, ctx=(children, ranges, placed))

  while any(u.op is Ops.BLOCKFORK and u.src[0].op is not Ops.BLOCK for u in sink.sparents):
    # link up the FORKs
    sink = graph_rewrite(sink, make_basic_blocks, ctx=(children, ranges, placed))

    # rewrite the FORKs as blocks
    sink = graph_rewrite(sink, fix_forks, ctx=(children, ranges, placed))

    # more rewrite
    sink = graph_rewrite(sink, make_basic_blocks, ctx=(children, ranges, placed))


  #ranges = []
  #stores = []
  #for u in sink.sparents:
    #if u.op is Ops.RANGE: ranges.append(u)
    #if u.op is Ops.IF: ifs.append(u)
    #if u.op is Ops.IF: ifs.append(u)



  """

  forks: Dict[UOp, UOp] = {}
  while 1:
    sink = graph_rewrite(sink, make_basic_blocks, ctx=(children, forks))

    # some blocks have two children, find them and mark them as okay to fork
    forks: Dict[UOp, UOp] = {}
    non_forks: Dict[UOp, None] = {}
    for block in sink.sparents:
      #if block.op is not Ops.BLOCK: continue
      for u in block.src:
        if u.op is Ops.BLOCK or u.op in DONT_PLACE_IN_BLOCK: continue
        if block.op is Ops.BLOCK:
          blk = UOp(Ops.BLOCK, dtypes.void, u.src, BasicBlock(get_ranges_in_parents(u), [u]))
          forks[u] = UOp(Ops.BLOCKFORK, dtypes.void, (blk,))
        else:
          non_forks[u] = None
    for k in non_forks:
      if k in forks:
        del forks[k]
    if len(forks) == 0: break
  """

  # terrible O(n^2) algo
  cc: DefaultDict[BasicBlock, List[UOp]] = defaultdict(list)
  for block in sink.sparents:
    if block.op is Ops.BLOCKEND:
      cc[block.arg].append(block)
  for k,v in cc.items():
    if len(v) > 1:
      rep = UOp(Ops.BLOCKEND, dtypes.void, tuple(dedup(flatten(x.src for x in v))), k)
      sink = sink.substitute({u:rep for u in v})

  """
  # move ifs
  def move_if(x):
    move_me = []
    dont_move_me = []
    for u in x.src:
      if u.op is Ops.BLOCK and u.arg.rngs == x.arg.rngs:
        move_me.append(u)
      else:
        dont_move_me.append(u)
    if len(move_me) == 0: return None
    lst = []
    for u in move_me: lst += u.arg.lst
    bif = UOp(Ops.BLOCKIF, dtypes.void, tuple(dont_move_me), x.arg)
    return UOp(Ops.BLOCK, dtypes.void, (bif,), BasicBlock(x.arg.rngs, lst))

  move_ifs = PatternMatcher([
    (UPat(Ops.BLOCKIF, name="x"), move_if),
  ])
  """
  #sink = graph_rewrite(sink, move_ifs)

  # show graph for VIZ
  sink = graph_rewrite(sink, PatternMatcher([]))
  return sink

def linearize_uop(sink:UOp, skip_check:bool=not __debug__) -> List[UOp]:
  assert sink.op is Ops.SINK, f"sink isn't sink, it's {sink.op}"

  sink_bb = block_uop(sink)

  # filter nodes that don't link to a sink
  # BFS toposort
  children: Dict[UOp, List[UOp]] = {}
  range_srcs: Dict[UOp, Dict[UOp, None]] = {}
  in_degree: Dict[UOp, int] = {}
  get_children_dfs(sink_bb, children, range_srcs, in_degree)

  """
  ret: List[UOp] = []
  placed: Set[UOp] = {}
  placable = []

  def place(x):
    placed.add(x)
    ret.append(x)
    for u in x.src:
      if all(y in placed for y in children[u]):
        placable.append(u)

  place(sink_bb)

  return ret
  """

  # NOTE: the compare should never make it all the way to u
  queue:List[Tuple[int, Tuple, UOp]] = []
  def push(u:UOp):
    p = 0
    if u.op is Ops.CONST: p = -10  # TODO: put CONST earlier
    heapq.heappush(queue, (p, u.tuplize, u))

  for u in children:
    if in_degree[u] == 0: push(u)

  _uops: List[UOp] = []
  open_loops: List[UOp] = []
  if DEBUG >= 6: print("*** PLACE")
  while queue:
    _,_,x = heapq.heappop(queue)
    if DEBUG >= 6: print(x.op, x.dtype, str(x.arg).split("\n")[0] if x.op in {Ops.BLOCK, Ops.BLOCKIF, Ops.BLOCKEND} else x.arg)
    if x.op is Ops.BLOCK:
      for u in x.arg.lst: assert u not in _uops, f"replacing {u.op}"
      _uops.extend(x.arg.lst)
    elif x.op is Ops.BLOCKFORK:
      pass
    elif x.op is Ops.BLOCKIF:
      _uops.extend(x.arg.lst)
      open_loops.append(x.arg.lst[0])
    elif x.op is Ops.BLOCKEND:
      for r in x.arg.rngs:
        assert r in open_loops
        if r.op is Ops.RANGE: _uops.append(UOp(Ops.ENDRANGE, dtypes.void, (r,)))
        elif r.op is Ops.IF: _uops.append(UOp(Ops.ENDIF, dtypes.void, (r,)))
        else: raise RuntimeError(f"unknown op {r.op}")
    elif x.op is Ops.DEFINE_ACC:
      idx = min([_uops.index(l) for l in x.src if l.op is Ops.RANGE])
      _uops.insert(idx, x)
    else:
      if x.op is Ops.RANGE: open_loops.append(x)
      _uops.append(x)
    for u in children[x]:
      in_degree[u] -= 1
      if in_degree[u] == 0: push(u)

  # sanity checks (NOTE: these can cause things to be skipped in BEAM)
  if not skip_check: type_verify(_uops)

  # strip the SINK
  assert _uops[-1].op is Ops.SINK
  return _uops[:-1]
