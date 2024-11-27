from typing import List, Dict, Tuple, DefaultDict
from collections import defaultdict
import functools
from tinygrad.ops import type_verify, UOp, Ops, PatternMatcher, UPat, graph_rewrite
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.helpers import dedup, flatten, partition

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

def disp(y) -> str:
  if y.op is Ops.BLOCKSTART: return "w"+disp(y.src[0])
  if y.op is Ops.IF: return f'IF{id(y)}'
  if y.op is Ops.RANGE: return str(y.arg[0])
  return "<NONE>"

class BasicBlock:
  def __init__(self, ctx, lst, end=None):
    self.ctx, self.lst, self.end = ctx, lst, end
  def __repr__(self):
    return f"{(str(disp(self.end))+' ') if self.end is not None else ''}"+\
           f"{[disp(y) for y in self.ctx]} {len(self.lst)}" + "\n" + '\n'.join([str(x.op) for x in self.lst])

@functools.lru_cache(None)
def _get_block_ctx(x:UOp) -> Tuple[UOp, ...]:
  ret: List[Tuple[UOp, ...]] = []
  for u in x.src:
    if u.op in {Ops.RANGE, Ops.IF}: ret.append((u,))
    # don't flow through assign and store
    elif u.op is Ops.STORE:
      # ugh, deal with non-reduce locals. probably wrong
      if isinstance(u.src[0].dtype, PtrDType) and u.src[0].dtype.local:
        idx_context, store_context = _get_block_ctx(u.src[0]), _get_block_ctx(u)
        ret.append(tuple(x for x in store_context if x not in idx_context and x.op is Ops.RANGE))
    elif u.op is Ops.ASSIGN:
      assert u.src[0].op is Ops.DEFINE_ACC
      ret.append(tuple(x for x in _get_block_ctx(u.src[1]) if x not in u.src[0].src[1:]))
    else:
      ret.append(_get_block_ctx(u))
  return tuple(dedup(sorted(flatten(ret), key=lambda x: x.tuplize)))

def get_block_ctx(x:UOp) -> Tuple[UOp, ...]:
  ret = _get_block_ctx(x)
  if x.op in {Ops.IF, Ops.RANGE}: return (UOp(Ops.BLOCKSTART, src=(x,)),) + ret
  return ret

DONT_PLACE_IN_BLOCK = {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.CONST,
                       Ops.BLOCK, Ops.BLOCKEND, Ops.BLOCKFORK, Ops.BLOCKSTART}
def append_to_block(ctx, x:UOp):
  new_srcs = []
  to_append = []
  new_blocks: DefaultDict[Tuple[UOp, ...], List[UOp]] = defaultdict(list)
  updated = False
  in_this_block = set(x.arg.lst)
  for u in x.src:
    if u.op in DONT_PLACE_IN_BLOCK or len([y for y in ctx[u] if y not in in_this_block]) > 0: new_srcs.append(u)
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
      if r not in x.arg.ctx and r.op is not Ops.BLOCKSTART:
        lrng = lrng[:]
        lrng.remove(r)
        new_block = UOp(Ops.BLOCKEND, dtypes.void, (new_block,),
                        BasicBlock(lrng, [UOp(Ops.ENDIF if r.op is Ops.IF else Ops.ENDRANGE, dtypes.void, (r,))], r))
    new_srcs.append(new_block)
  return UOp(Ops.BLOCK, dtypes.void, tuple(dedup(new_srcs)), BasicBlock(x.arg.ctx, to_append+x.arg.lst))

make_basic_blocks = PatternMatcher([
  (UPat(Ops.SINK, name="x"), lambda x: UOp(Ops.BLOCK, src=x.src, arg=BasicBlock([], [x]))),
  (UPat(Ops.BLOCK, name="x"), append_to_block),
])

def blockend_gobble(ctx, x:UOp):
  # x is a blockend
  new_srcs = []
  to_append = []

  # first, see if we are done with placement. if all the children of the range are in here
  in_this_block = set(x.arg.lst)
  if len([y for y in ctx[x.arg.end] if y not in in_this_block]) == 0:
    # find the parent block that has the BLOCKSTART in the ctx
    search_block_start = UOp(Ops.BLOCKSTART, src=(x.arg.end,))
    parent_blocks = [y for y in x.src if y.op is Ops.BLOCK and search_block_start in y.arg.ctx]
    if len(parent_blocks) == 1:
      parent_block = parent_blocks[0]
      # range needs DEFINE_ACC to be before the range (never in DEFINE_ACC for if)
      early_ops, late_ops = partition(x.arg.lst, lambda y: y.op is Ops.DEFINE_ACC and x.arg.end in y.src)
      return UOp(Ops.BLOCK, dtypes.void, tuple(y for y in x.src if y is not parent_block)+parent_block.src,
                BasicBlock([y for y in x.arg.ctx if y is not x.arg.end], early_ops+parent_block.arg.lst+late_ops))
    assert not len(parent_blocks)

  updated = False
  new_ctx = x.arg.ctx[:]
  placed = set()
  for u in x.src:
    if u.op is Ops.BLOCK and x.arg.end in u.arg.ctx:
      new_ctx += u.arg.ctx
      new_srcs += list(u.src)
      to_append += u.arg.lst
      updated = True
    elif u.op is Ops.BLOCKFORK:
      # block fork appears # of times in srcs
      seen_count = len([y for y in x.src if y is u])
      if seen_count == u.arg:
        if u not in placed:
          new_srcs += list(u.src)
          placed.add(u)
          updated = True
      else:
        # keep it
        new_srcs.append(u)
    else:
      new_srcs.append(u)
  if not updated: return None
  return UOp(Ops.BLOCKEND, dtypes.void, tuple(new_srcs), BasicBlock(dedup(new_ctx), to_append+x.arg.lst, x.arg.end))

def block_merge(ctx, x:UOp):
  updated = False
  new_srcs = []
  to_append = []
  placed = set()
  for u in x.src:
    if u.op is Ops.BLOCK and tuple(u.arg.ctx) == tuple(x.arg.ctx):
      new_srcs += list(u.src)
      to_append += u.arg.lst
      updated = True
    # TODO: copied from above
    elif u.op is Ops.BLOCKFORK:
      # block fork appears # of times in srcs
      seen_count = len([y for y in x.src if y is u])
      if seen_count == u.arg:
        if u not in placed:
          new_srcs += list(u.src)
          placed.add(u)
          updated = True
      else:
        # keep it
        new_srcs.append(u)
    else:
      new_srcs.append(u)
  if not updated: return None
  return UOp(Ops.BLOCK, dtypes.void, tuple(new_srcs), BasicBlock(x.arg.ctx, to_append+x.arg.lst))

blockend = PatternMatcher([
  (UPat(Ops.BLOCKEND, name="x"), blockend_gobble),
  (UPat(Ops.BLOCK, name="x"), block_merge),
])

def linearize_uop(sink:UOp, skip_check:bool=not __debug__) -> List[UOp]:
  assert sink.op is Ops.SINK, f"sink isn't sink, it's {sink.op}"
  # filter nodes that don't link to a sink
  # BFS toposort
  children: Dict[UOp, List[UOp]] = {}
  range_srcs: Dict[UOp, Dict[UOp, None]] = {}
  in_degree: Dict[UOp, int] = {}
  get_children_dfs(sink, children, range_srcs, in_degree)

  # TODO: there's probably a clever way to remove this while loop
  while 1:
    sink = graph_rewrite(sink, make_basic_blocks, ctx=children)

    # add BLOCKFORK
    block_parents = flatten([x.src for x in sink.sparents if x.op is Ops.BLOCK])
    non_block_parents = flatten([x.src for x in sink.sparents if x.op is not Ops.BLOCK])
    forks = {}
    for u in block_parents:
      if u in non_block_parents: continue
      child_count = len([x for x in block_parents if x is u])
      if child_count > 1 and u.op not in DONT_PLACE_IN_BLOCK:
        forks[u] = UOp(Ops.BLOCKFORK, src=(UOp(Ops.BLOCK, src=u.src, arg=BasicBlock(get_block_ctx(u), [u])),), arg=child_count)

    if not len(forks): break
    sink = sink.substitute(forks)

  # combine matching BLOCKENDS
  blockends = [x for x in sink.sparents if x.op is Ops.BLOCKEND]
  blockends_to_arg: DefaultDict[UOp, List[UOp]] = defaultdict(list)
  for be in blockends: blockends_to_arg[be.arg.end].append(be)
  new_forks = {}
  for k,v in blockends_to_arg.items():
    # parents fixup
    """
    to_remove = []
    for kv in v:
      for kvv in v:
        if kvv in kv.parents:
          new_forks[kvv] = kvv.src[0]
          to_remove.append(kvv)
    v = [x for x in v if x not in to_remove]
    """
    if len(v) > 1:
      new_be = UOp(Ops.BLOCKEND, src=tuple(flatten(x.src for x in v)), arg=BasicBlock(dedup(flatten([y.arg.ctx for y in v])), v[0].arg.lst, k))
      out = UOp(Ops.BLOCKFORK, src=(new_be,), arg=len(v))
      for u in v: new_forks[u] = out
  sink = sink.substitute(new_forks)

  # final rewrite to linearizer
  sink = graph_rewrite(sink, blockend, ctx=children)

  # there should just be one block left
  assert sink.op is Ops.BLOCK
  _uops = sorted(dedup(sink.src), key=lambda x: x.tuplize)
  assert all(len(x.src) == 0 and x.op not in {Ops.BLOCK, Ops.BLOCKSTART, Ops.BLOCKEND, Ops.BLOCKFORK} for x in _uops)
  _uops += sink.arg.lst

  # sanity checks (NOTE: these can cause things to be skipped in BEAM)
  if not skip_check: type_verify(_uops)

  # strip the SINK
  return _uops[:-1]
