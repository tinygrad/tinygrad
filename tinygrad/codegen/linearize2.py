from tinygrad.ops import UOp, Ops, graph_rewrite, PatternMatcher, UPat, GroupOp
from tinygrad.dtype import PtrDType
from tinygrad.helpers import dedup, partition, all_same, flatten
from tinygrad.codegen.linearize import BasicBlock

def _start_strip(y:UOp): return [z.src[0] if z.op is Ops.BLOCKSTART else z for z in y]

def make_block(ctx, x:UOp):
  # NOTE: we don't update the children
  assert len(x.src) > 0 and all(x.op is Ops.BLOCK for x in x.src)

  # this is key to making this algorithm work. we recover the old x by replacing the srcs
  fixed_x = x.replace(src=tuple(y.arg.lst[-1] for y in x.src))

  # compute the new context
  # if this is a RANGE or an IF, we add it to this block context
  _new_ctx = [UOp(Ops.BLOCKSTART, src=(fixed_x,))] if fixed_x.op in {Ops.RANGE, Ops.IF} else []
  _old_ctxs = []
  for y in x.src:
    _old_ctx = _start_strip(y.arg.ctx)
    uop: UOp = y.arg.lst[-1]

    # if it's an ASSIGN, remove any RANGEs used
    if uop.op is Ops.ASSIGN:
      assert uop.src[0].op is Ops.DEFINE_ACC
      _old_ctx = [x for x in _old_ctx if x not in uop.src[0].src[1:]]

    # if it's a STORE remove everything if non local
    if uop.op is Ops.STORE:
      if isinstance(uop.src[0].dtype, PtrDType) and uop.src[0].dtype.local:
        # get these ctxs from the blocks
        idx_context, store_context = x.src[0].arg.ctx, x.src[1].arg.ctx
        _old_ctx = [x for x in store_context if x not in idx_context and x.op is Ops.RANGE]
      else:
        _old_ctx = []
    _old_ctxs.append(_old_ctx)

  new_ctx = sorted(dedup(_new_ctx+flatten(_old_ctxs)), key=lambda x: x.tuplize)

  # use a tuple
  tuple_ctx = tuple(new_ctx)

  # a block is unmergable if it has children or it has a different context
  unmergable_blocks, mergable_blocks = partition(x.src,
    lambda y: len(ctx.children[y.arg.lst[-1]]) > 1 or y.arg.ctx != tuple_ctx or y.op is Ops.BLOCKEND)

  # for all unmergable blocks, if they aren't mergable because of ctx changes, we have to insert BLOCKENDs
  final_unmergable_blocks = []
  for x in unmergable_blocks:
    um_ctx = tuple(_start_strip(x.arg.ctx))
    if um_ctx != tuple_ctx:
      ends_to_add = [y for y in um_ctx if y not in tuple_ctx][::-1]
      while len(ends_to_add):
        r = ends_to_add[0]
        end_uop = UOp(Ops.ENDIF if r.op is Ops.IF else Ops.ENDRANGE, src=(r,))
        x = UOp(Ops.BLOCKEND, src=(x,), arg=BasicBlock(tuple(ends_to_add[1:]), (end_uop,), r, cnt=1))
        ends_to_add = ends_to_add[1:]
    final_unmergable_blocks.append(x)

  # count the number of times this is referenced in srcs, a little different from children
  cnt = len(ctx.children[fixed_x])
  #child_srcs = flatten([y.src for y in ctx.children[fixed_x]])
  #cnt = len([y for y in child_srcs if y is fixed_x])

  # create the block (TODO: we don't actually need to track that list)
  arg = BasicBlock(tuple_ctx, tuple(flatten([y.arg.lst for y in mergable_blocks])+[fixed_x]), cnt=cnt)
  new_srcs = flatten([y.src for y in mergable_blocks])+final_unmergable_blocks
  return UOp(Ops.BLOCK, src=tuple(new_srcs), arg=arg)

"""
def merge_block(ctx, x:UOp):
  sole_child_blocks, multi_child_blocks = partition(dedup(x.src), lambda y: len(ctx.children[y]) == 1)
  if len(sole_child_blocks) == 0: return None
  print("MERGE")
  assert all_same([y.arg.ctx for y in sole_child_blocks])
  arg = BasicBlock(sole_child_blocks[0].arg.ctx, tuple(flatten([y.arg.lst for y in sole_child_blocks]))+x.arg.lst)
  new_srcs = flatten([y.src for y in sole_child_blocks])+multi_child_blocks
  return UOp(Ops.BLOCK, src=tuple(new_srcs), arg=arg)
"""

def merge_blockend(x:UOp):
  unmergable_blocks, mergable_blocks = [], []
  #real_ctx = tuple(sorted(dedup(x.arg.ctx+(x.arg.end,)), key=lambda x: x.tuplize))
  for y in x.src:
    if y.op is Ops.BLOCK and x.src.count(y) == y.arg.cnt and x.arg.end in y.arg.ctx:
      if y not in mergable_blocks: mergable_blocks.append(y)
    else:
      #print(y.op is Ops.BLOCK, x.src.count(y) == y.arg.cnt, real_ctx == y.arg.ctx)
      #print([z.arg for z in real_ctx], [z.arg for z in y.arg.ctx])
      unmergable_blocks.append(y)

  if len(mergable_blocks) == 0:
    #print("FAIL TO MERGE", x.arg.end, len(real_ctx))
    return None
  #print("MERGING", x.arg.end)

  # create the block (TODO: we don't actually need to track that list)
  new_lst = tuple(flatten([y.arg.lst for y in mergable_blocks]))+x.arg.lst
  arg = BasicBlock(x.arg.ctx, new_lst, end=x.arg.end, cnt=x.arg.cnt)
  new_srcs = flatten([y.src for y in mergable_blocks])+unmergable_blocks
  return UOp(Ops.BLOCKEND, src=tuple(new_srcs), arg=arg)

def remove_blockend(x:UOp):
  # if there's any
  if any(x.arg.end in y.arg.ctx for y in x.src): return None

  parent_blocks = [y for y in x.src if y.op is Ops.BLOCK and UOp(Ops.BLOCKSTART, src=(x.arg.end,)) in y.arg.ctx]
  assert all_same(parent_blocks), f"should never have two parent blocks (has {len(parent_blocks)})"
  if len(parent_blocks) > 0:
    parent_block = parent_blocks[0]
    assert len(parent_blocks) == parent_block.arg.cnt
    # range needs DEFINE_ACC to be before the range (never in DEFINE_ACC for if)
    early_ops, late_ops = partition(x.arg.lst, lambda y: y.op is Ops.DEFINE_ACC and x.arg.end in y.src)
    # NOTE: we have to add a barrier at the start if barrier is used in the range
    if x.op is Ops.BLOCKEND and any(y.op is Ops.BARRIER for y in late_ops) and late_ops[-1].op is Ops.ENDRANGE:
      late_ops = [UOp(Ops.BARRIER)] + late_ops
    return UOp(Ops.BLOCK, src=tuple(y for y in x.src if y is not parent_block)+parent_block.src,
      arg=BasicBlock(tuple(sorted([y for y in x.arg.ctx if y is not x.arg.end], key=lambda x: x.tuplize)),
                     tuple(early_ops)+parent_block.arg.lst+tuple(late_ops), cnt=x.arg.cnt))

def merge_block(x:UOp):
  unmergable_blocks, mergable_blocks = [], []
  #print("merge block", len(x.src))
  for y in x.src:
    # if ctxs match and count is correct, we merge in this block
    if y.op is Ops.BLOCK and x.src.count(y) == y.arg.cnt and x.arg.ctx == y.arg.ctx:
      if y not in mergable_blocks: mergable_blocks.append(y)
    else:
      unmergable_blocks.append(y)
  if len(mergable_blocks) == 0: return None

  # create the block (TODO: we don't actually need to track that list)
  new_lst = tuple(flatten([y.arg.lst for y in mergable_blocks]))+x.arg.lst
  arg = BasicBlock(x.arg.ctx, new_lst, cnt=x.arg.cnt)
  new_srcs = flatten([y.src for y in mergable_blocks])+unmergable_blocks
  return UOp(Ops.BLOCK, src=tuple(new_srcs), arg=arg)

block_merge = PatternMatcher([
  (UPat(Ops.BLOCK, name="x"), merge_block),
  (UPat(Ops.BLOCKEND, name="x"), merge_blockend),
  (UPat(Ops.BLOCKEND, name="x"), remove_blockend),
])

# this will wrap every UOp in a BLOCK, top down
blocks_in_context = PatternMatcher([
  # make blocks from non block and const
  (UPat(GroupOp.All-{Ops.BLOCK, Ops.BLOCKEND}, src=(), name="x"), lambda ctx,x: UOp(Ops.BLOCK, arg=BasicBlock((), (x,), cnt=len(ctx.children[x])))),
  # NOTE: this pattern doesn't match 0 due to the early reject
  (UPat(GroupOp.All-{Ops.BLOCK, Ops.BLOCKEND}, src=UPat((Ops.BLOCK, Ops.BLOCKEND)), name="x"), make_block),

  # merge block
  #(UPat(Ops.BLOCK, name="x"), merge_block),
])


def linearize_uop(sink:UOp, skip_check:bool=not __debug__) -> list[UOp]:
  assert sink.op is Ops.SINK, f"sink isn't sink, it's {sink.op}"

  # TODO: do we really need track_children. just precompute them
  sink = graph_rewrite(sink, blocks_in_context, track_children=True, name="Linearizer: Create Blocks")

  # combine matching BLOCKENDS, the keys of this dictionary are the RANGE UOps, values are the BLOCKENDs
  blockends_to_arg: dict[UOp, list[UOp]] = {}
  for be in sink.toposort:
    if be.op is Ops.BLOCKEND: blockends_to_arg.setdefault(be.arg.end, []).append(be)
  new_forks = {}
  for k,v in blockends_to_arg.items():
    # NOTE: if any BLOCKEND is the parent of any other with the same arg, this algo fails
    if len(v) > 1:
      # TODO: don't use BLOCKFORK here, track the count in BLOCKEND
      out = UOp(Ops.BLOCKEND, src=tuple(flatten(x.src for x in v)),
                arg=BasicBlock(tuple(dedup(flatten([y.arg.ctx for y in v]))), v[0].arg.lst, k, cnt=len(v)))
      for u in v: new_forks[u] = out
  sink = sink.substitute(new_forks)

  # merge blocks
  sink = graph_rewrite(sink, block_merge, name="Linearizer: Merge Blocks")

  assert sink.op is Ops.BLOCK and len(sink.src) == 0
  return list(sink.arg.lst)
