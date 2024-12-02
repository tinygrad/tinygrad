from typing import List, Dict, Tuple
import collections
from tinygrad.ops import type_verify, UOp, Ops, PatternMatcher, UPat, graph_rewrite
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.helpers import dedup, flatten, partition

DONT_PLACE_IN_BLOCK = {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.CONST,
                       Ops.BLOCK, Ops.BLOCKEND, Ops.BLOCKFORK, Ops.BLOCKSTART}

def disp(y:UOp) -> str:
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

def append_to_block(ctx, x:UOp):
  block_ctxs, children = ctx
  new_srcs: List[UOp] = []
  to_append: List[UOp] = []
  new_blocks: Dict[Tuple[UOp, ...], List[UOp]] = {}
  in_this_block = set(x.arg.lst)
  for u in x.src:
    if u.op in DONT_PLACE_IN_BLOCK or len([y for y in children[u] if y not in in_this_block]) > 0:
      # if it's a fork or not placed, we don't place it
      new_srcs.append(u)
    elif (block_ctx:=block_ctxs[u]) == x.arg.ctx:
      # if it's the same context, we place the UOp in this block and append the parents to it's srcs
      new_srcs += list(u.src)
      to_append.append(u)
    else:
      # otherwise, we create a new block with this UOp
      new_blocks.setdefault(block_ctx, []).append(u)
  if len(to_append) == 0 and len(new_blocks) == 0: return None

  for rng,lst in new_blocks.items():
    new_block = UOp(Ops.BLOCK, dtypes.void, tuple(dedup(flatten(y.src for y in lst))), BasicBlock(rng, lst))
    lrng = list(rng)
    for r in rng[::-1]:
      if r not in x.arg.ctx and r.op is not Ops.BLOCKSTART:
        lrng.remove(r)
        new_block = UOp(Ops.BLOCKEND, src=(new_block,), arg=BasicBlock(lrng[:], [UOp(Ops.ENDIF if r.op is Ops.IF else Ops.ENDRANGE, src=(r,))], r))
    new_srcs.append(new_block)
  return UOp(Ops.BLOCK, dtypes.void, tuple(dedup(new_srcs)), BasicBlock(x.arg.ctx, to_append+x.arg.lst))

make_basic_blocks = PatternMatcher([
  (UPat(Ops.SINK, name="x"), lambda x: UOp(Ops.BLOCK, src=x.src, arg=BasicBlock([], [x]))),
  (UPat(Ops.BLOCK, name="x"), append_to_block),
])

def block_merge(ctx, x:UOp):
  # ctx is children here
  if x.op is Ops.BLOCKEND:
    # if it's a BLOCKEND, see if we are done with placement. if all the children of the range are in here
    in_this_block = set(x.arg.lst)
    if len([y for y in ctx[x.arg.end] if y not in in_this_block]) == 0:
      # find the parent block that has the BLOCKSTART in the ctx
      parent_blocks = [y for y in x.src if y.op is Ops.BLOCK and UOp(Ops.BLOCKSTART, src=(x.arg.end,)) in y.arg.ctx]
      if len(parent_blocks) == 1:
        parent_block = parent_blocks[0]
        # range needs DEFINE_ACC to be before the range (never in DEFINE_ACC for if)
        early_ops, late_ops = partition(x.arg.lst, lambda y: y.op is Ops.DEFINE_ACC and x.arg.end in y.src)
        return UOp(Ops.BLOCK, dtypes.void, tuple(y for y in x.src if y is not parent_block)+parent_block.src,
                  BasicBlock([y for y in x.arg.ctx if y is not x.arg.end], early_ops+parent_block.arg.lst+late_ops))
      assert not len(parent_blocks)

  new_srcs: List[UOp] = []
  to_append: List[UOp] = []
  new_ctx = list(x.arg.ctx[:])
  placed = set()
  for u in x.src:
    if u.op is Ops.BLOCK and (tuple(u.arg.ctx) == tuple(x.arg.ctx) or (x.arg.end is not None and x.arg.end in u.arg.ctx)):
      # NOTE: this can't appear in srcs twice or it would be a BLOCKFORK
      new_ctx += u.arg.ctx
      new_srcs += list(u.src)
      to_append += u.arg.lst
    elif u.op is Ops.BLOCKFORK and len([y for y in x.src if y is u]) == u.arg: # block fork appears # of times in srcs
      if u not in placed:
        new_srcs += list(u.src)
        placed.add(u)
    else:
      # keep it in srcs
      new_srcs.append(u)
  if len(to_append) == 0 and len(placed) == 0: return None
  return UOp(x.op, dtypes.void, tuple(new_srcs), BasicBlock(dedup(new_ctx), to_append+x.arg.lst, x.arg.end))

pm_block_merge = PatternMatcher([(UPat((Ops.BLOCKEND, Ops.BLOCK), name="x"), block_merge),])

def linearize_uop(sink:UOp, skip_check:bool=not __debug__) -> List[UOp]:
  assert sink.op is Ops.SINK, f"sink isn't sink, it's {sink.op}"

  # get children and all block contexts
  temp_block_ctxs: Dict[UOp, List[UOp]] = {}
  children: Dict[UOp, List[UOp]] = {}
  for u in sink.toposort:
    this_block_ctx: List[UOp] = []
    for s in u.src:
      # save children
      children.setdefault(s, []).append(u)
      # compute block ctx
      if s.op in {Ops.RANGE, Ops.IF}: this_block_ctx.append(s)
      # don't flow (fully) through assign and store
      elif s.op is Ops.STORE:
        # ugh, deal with non-reduce locals. probably wrong
        if isinstance(s.src[0].dtype, PtrDType) and s.src[0].dtype.local:
          idx_context, store_context = temp_block_ctxs[s.src[0]], temp_block_ctxs[s]
          this_block_ctx += [x for x in store_context if x not in idx_context and x.op is Ops.RANGE]
      elif s.op is Ops.ASSIGN:
        # flow though assign, but remove the ranges used in the assign
        assert s.src[0].op is Ops.DEFINE_ACC
        this_block_ctx += [x for x in temp_block_ctxs[s.src[1]] if x not in s.src[0].src[1:]]
      else:
        # flow though everything else
        this_block_ctx += temp_block_ctxs[s]
    temp_block_ctxs[u] = dedup(sorted(this_block_ctx, key=lambda x: x.tuplize))

  # make final block_ctxs, add BLOCKSTART to block_ctxs for IF and RANGE
  block_ctxs: Dict[UOp, Tuple[UOp, ...]] = {}
  for u in sink.toposort:
    block_ctxs[u] = ((UOp(Ops.BLOCKSTART, src=(u,)),) + tuple(temp_block_ctxs[u])) if u.op in {Ops.IF, Ops.RANGE} else tuple(temp_block_ctxs[u])

  # TODO: there's probably a clever way to remove this while loop
  while 1:
    sink = graph_rewrite(sink, make_basic_blocks, ctx=(block_ctxs, children))

    # add BLOCKFORK (slow!)
    block_parent_count = collections.Counter(flatten([x.src for x in sink.sparents if x.op is Ops.BLOCK]))
    non_block_parents = flatten([x.src for x in sink.sparents if x.op is not Ops.BLOCK])
    forks = {}
    for u,child_count in block_parent_count.items():
      if u.op not in DONT_PLACE_IN_BLOCK and child_count > 1 and u not in non_block_parents:
        forks[u] = UOp(Ops.BLOCKFORK, src=(UOp(Ops.BLOCK, src=u.src, arg=BasicBlock(block_ctxs[u], [u])),), arg=child_count)

    if not len(forks): break
    sink = sink.substitute(forks)

  # combine matching BLOCKENDS
  blockends_to_arg: Dict[UOp, List[UOp]] = {}
  for be in sink.sparents:
    if be.op is Ops.BLOCKEND: blockends_to_arg.setdefault(be.arg.end, []).append(be)
  new_forks = {}
  for k,v in blockends_to_arg.items():
    # NOTE: if any BLOCKEND is the parent of any other with the same arg, this algo fails
    if len(v) > 1:
      new_blockend = UOp(Ops.BLOCKEND, src=tuple(flatten(x.src for x in v)), arg=BasicBlock(dedup(flatten([y.arg.ctx for y in v])), v[0].arg.lst, k))
      out = UOp(Ops.BLOCKFORK, src=(new_blockend,), arg=len(v))
      for u in v: new_forks[u] = out
  sink = sink.substitute(new_forks)

  # final rewrite to merge all blocks into one
  sink = graph_rewrite(sink, pm_block_merge, ctx=children)

  # there should just be one block left, with a few parents with 0 srcs
  assert sink.op is Ops.BLOCK
  _uops = sorted(dedup(sink.src), key=lambda x: x.tuplize)
  assert all(len(x.src) == 0 and x.op not in {Ops.BLOCK, Ops.BLOCKSTART, Ops.BLOCKEND, Ops.BLOCKFORK} for x in _uops)
  _uops += sink.arg.lst

  # sanity checks (NOTE: these can cause things to be skipped in BEAM)
  if not skip_check: type_verify(_uops)

  # strip the SINK
  return _uops[:-1]
