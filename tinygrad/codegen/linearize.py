from __future__ import annotations
from typing import List, Dict, Tuple, DefaultDict
import functools, heapq
from collections import defaultdict
from tinygrad.ops import type_verify, UOp, Ops
from tinygrad.dtype import dtypes
from tinygrad.helpers import dedup, flatten, DEBUG, partition

from tinygrad.ops import PatternMatcher, UPat, graph_rewrite
def disp(y): return y.arg[0] if y.op is Ops.RANGE else f'IF{id(y)}'

class BasicBlock:
  def __init__(self, rngs, lst, closed=()):
    self.rngs = tuple(rngs)
    self.lst = tuple(lst)
    self.closed = tuple(closed)
  def __hash__(self): return hash((self.rngs, self.lst))
  def __eq__(self, x):
    if x is None: return False
    return self.rngs == x.rngs and self.lst == x.lst and self.closed == x.closed
  def __repr__(self):
    return f"{[disp(y) for y in self.rngs]} {[disp(y) for y in self.closed] if len(self.closed) else ''} {len(self.lst)}" + "\n" + '\n'.join([str(x.op) for x in self.lst])
  @functools.cached_property
  def lst_tuplize(self): return tuple(y.tuplize for y in self.lst)
  @functools.cached_property
  def rngs_tuplize(self): return tuple(y.tuplize for y in self.rngs)
  def __lt__(self, x:BasicBlock):
    if self.rngs == x.rngs: return False # self.lst_tuplize < x.lst_tuplize
    return self.rngs_tuplize < x.rngs_tuplize
  def add(self, x):
    if len(x) == 0: return self
    return BasicBlock(self.rngs, tuple(x)+self.lst, self.closed)

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

DONT_PLACE_IN_BLOCK = {Ops.RANGE, Ops.CONST, Ops.DEFINE_ACC, Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.BLOCK, Ops.BLOCKEND, Ops.BLOCKIF, Ops.BLOCKFORK}
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
      blk = UOp(Ops.BLOCK, dtypes.void, u.src, BasicBlock(ranges[u], []))
      placed[u] = blk.arg
      blk_if = UOp(Ops.BLOCKIF, dtypes.void, (blk,), BasicBlock(get_ranges_in_parents(u)+(u,), [u]))
      new_srcs.append(blk_if)
      continue

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
      # this has an unplaced child, it stays in srcs because it might not need a fork (diamond)
      new_srcs.append(u)
    else:
      updated = True
      new_srcs.append(UOp(Ops.BLOCKFORK, dtypes.void, (u,), len(children[u])))

  if not updated: return None

  # create new blocks, sometimes ending things
  for rng,lst in new_blocks.items():
    new_src = UOp(Ops.BLOCK, dtypes.void, tuple(dedup(flatten(y.src for y in lst))), BasicBlock(rng, lst))
    for l in new_src.arg.lst: placed[l] = new_src.arg
    closed = tuple([y for y in new_src.arg.rngs if y not in x.arg.rngs])
    if len(closed):
      # TODO: is this order always right?
      for c in closed[::-1]:
        new_src = UOp(Ops.BLOCKEND, dtypes.void, (new_src,), arg=BasicBlock(rng, [], [c]))
        rng = tuple(x for x in rng if x is not c)
    new_srcs.append(new_src)

  ret = UOp(Ops.BLOCK, dtypes.void, tuple(dedup(new_srcs)), x.arg.add(to_append))
  for l in ret.arg.lst: placed[l] = ret.arg
  return ret

def place_sink(ctx, x:UOp):
  children, ranges, placed = ctx
  ret = UOp(Ops.BLOCK, dtypes.void, x.src, BasicBlock([], [x]))
  placed[x] = ret.arg
  return ret

def fix_fork(ctx, x):
  assert len(x.src) == 1
  if x.src[0].op is Ops.BLOCK: return None
  children, ranges, placed = ctx
  ret = UOp(Ops.BLOCK, dtypes.void, x.src[0].src, BasicBlock(ranges[x.src[0]], [x.src[0]]))
  placed[x.src[0]] = ret.arg
  return x.replace(src=(ret,))

make_basic_blocks = PatternMatcher([
  (UPat(Ops.SINK, name="x"), place_sink),
  (UPat(Ops.BLOCK, name="x"), simple_append_to_block),
])

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

  # add ifs to parents that only feed into things already in the if (awful algo is n^2)
  changed = True
  while changed:
    changed = False
    for k,v in children.items():
      if len(v) == 0: continue
      if k.op in {Ops.IF, Ops.ASSIGN}: continue
      # if all children have an if, add the if
      to_add = [u for u in set.intersection(*[set(ranges[c]) for c in v]) if u.op is Ops.IF and u not in ranges[k]]
      if len(to_add):
        changed = True
        ranges[k] = tuple(dedup(sorted(ranges[k]+tuple(to_add), key=lambda x: x.tuplize)))

  # terrible algo for rewriting
  placed = {}
  sink = graph_rewrite(sink, make_basic_blocks, ctx=(children, ranges, placed))
  while any(u.op is Ops.BLOCKFORK and u.src[0].op is not Ops.BLOCK for u in sink.sparents):
    # link up the FORKs
    sink = graph_rewrite(sink, make_basic_blocks, ctx=(children, ranges, placed))

    # rewrite the FORKs as blocks
    sink = graph_rewrite(sink, fix_forks, ctx=(children, ranges, placed))

    # more rewrite
    sink = graph_rewrite(sink, make_basic_blocks, ctx=(children, ranges, placed))

  # terrible O(n^2) algo for END merging
  cc: DefaultDict[BasicBlock, List[UOp]] = defaultdict(list)
  for block in sink.sparents:
    if block.op is Ops.BLOCKEND:
      cc[block.arg].append(block)
  for k,v in cc.items():
    if len(v) > 1:
      rep = UOp(Ops.BLOCKEND, dtypes.void, tuple(dedup(flatten(x.src for x in v))), k)
      sink = sink.substitute({u:rep for u in v})

  # blockend gobble
  def blockend_gobble(x:UOp):
    blocks, non_blocks = partition(x.src, lambda y: y.op is Ops.BLOCK and x.arg.closed[0] in y.arg.rngs) # and x.arg.rngs == y.arg.rngs)
    #for b in blocks: assert x.arg.closed[0] in b.arg.rngs
    if len(blocks) == 0:
      # find BLOCKFORK and merge it
      for b in non_blocks:
        if b.op is Ops.BLOCKFORK:
          print(len([x for x in non_blocks if x is b]), b.arg)
          if len([x for x in non_blocks if x is b]) == b.arg:
            new_srcs = tuple([x for x in non_blocks if x is not b]) + b.src
            return UOp(Ops.BLOCKEND, dtypes.void, new_srcs, x.arg)
      # find matching BLOCKIF and merge it
      # find matching RANGE and merge it
      return None
    return UOp(Ops.BLOCKEND, dtypes.void, tuple(flatten([x.src for x in blocks])+non_blocks), x.arg.add(flatten([x.arg.lst for x in blocks])))

  gobble = PatternMatcher([
    (UPat(Ops.BLOCKEND, name="x"), blockend_gobble),
  ])
  sink = graph_rewrite(sink, gobble)

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
      for u in x.arg.lst: assert u not in _uops, f"replacing {u.op}"
      _uops.extend(x.arg.lst)

      assert len(x.arg.closed) == 1
      r = x.arg.closed[0]
      """
      if r is not open_loops[-1] and r.op is Ops.RANGE:
        flip_ifs = open_loops[open_loops.index(r)+1:][::-1]
        for rr in flip_ifs:
          assert rr.op is Ops.IF
          if rr.op is Ops.IF: _uops.append(UOp(Ops.ENDIF, dtypes.void, (rr,)))
        _uops.append(UOp(Ops.ENDRANGE, dtypes.void, (r,)))
        for rr in flip_ifs:
          _uops.append(rr)
      else:
        assert r is open_loops[-1]
      """
      if r.op is Ops.RANGE: _uops.append(UOp(Ops.ENDRANGE, dtypes.void, (r,)))
      elif r.op is Ops.IF: _uops.append(UOp(Ops.ENDIF, dtypes.void, (r,)))
      open_loops.remove(r)
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
