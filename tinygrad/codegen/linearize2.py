from tinygrad.ops import UOp, Ops, graph_rewrite, PatternMatcher, UPat, GroupOp
from tinygrad.dtype import PtrDType
from tinygrad.helpers import dedup, partition, all_same, flatten
from tinygrad.codegen.linearize import BasicBlock
from tinygrad.upat import _get_code

def make_block(ctx, x:UOp):
  # NOTE: we don't update the children
  assert len(x.src) > 0 and all(x.op is Ops.BLOCK for x in x.src)

  # this is key to making this algorithm work. we recover the old x by replacing the srcs
  fixed_x = x.replace(src=tuple(y.arg.lst[-1] for y in x.src))

  # compute the new context
  # if this is a RANGE or an IF, we add it to this block context
  _new_ctx = [UOp(Ops.BLOCKSTART, src=(fixed_x,))] if fixed_x.op in {Ops.RANGE, Ops.IF} else []
  _old_ctxs = [[z.src[0] if z.op is Ops.BLOCKSTART else z for z in y.arg.ctx] for y in x.src]
  new_ctx = sorted(dedup(_new_ctx+flatten(_old_ctxs)), key=lambda x: x.tuplize)

  # if it's an ASSIGN, remove any RANGEs used
  if fixed_x.op is Ops.ASSIGN:
    assert fixed_x.src[0].op is Ops.DEFINE_ACC
    new_ctx = [x for x in new_ctx if x not in fixed_x.src[0].src[1:]]

  # if it's a STORE remove everything if non local
  if fixed_x.op is Ops.STORE:
    if isinstance(fixed_x.src[0].dtype, PtrDType) and fixed_x.src[0].dtype.local:
      # get these ctxs from the blocks
      idx_context, store_context = x.src[0].arg.ctx, x.src[1].arg.ctx
      new_ctx = [x for x in store_context if x not in idx_context and x.op is Ops.RANGE]
    else:
      new_ctx = []

  # use a tuple
  tuple_ctx = tuple(new_ctx)

  # a block is unmergable if it has children or it has a different context
  unmergable_blocks, mergable_blocks = partition(x.src, lambda y: len(ctx.children[y.arg.lst[-1]]) > 1 or y.arg.ctx != tuple_ctx)

  # create the block (TODO: we don't actually need to track that list)
  arg = BasicBlock(tuple_ctx, tuple(flatten([y.arg.lst for y in mergable_blocks])+[fixed_x]))
  new_srcs = flatten([y.src for y in mergable_blocks])+unmergable_blocks
  return UOp(Ops.BLOCK, src=tuple(new_srcs), arg=arg)

# this will wrap every UOp in a BLOCK, top down
blocks_in_context = PatternMatcher([
  # make blocks from non block and const
  (UPat(GroupOp.All-{Ops.BLOCK}, src=(), name="x"), lambda x: UOp(Ops.BLOCK, arg=BasicBlock((), (x,)))),
  (UPat(GroupOp.All-{Ops.BLOCK}, src=UPat(Ops.BLOCK), name="x"), make_block),

  #(UPat(Ops.RANGE, name="r"), lambda r: UOp(Ops.BLOCK, BasicBlock())),
])

def merge_block(ctx, x:UOp):
  sole_child_blocks, multi_child_blocks = partition(dedup(x.src), lambda y: len(ctx.children[y]) == 1)
  if len(sole_child_blocks) == 0: return None
  print("MERGE")
  assert all_same([y.arg.ctx for y in sole_child_blocks])
  arg = BasicBlock(sole_child_blocks[0].arg.ctx, tuple(flatten([y.arg.lst for y in sole_child_blocks]))+x.arg.lst)
  new_srcs = flatten([y.src for y in sole_child_blocks])+multi_child_blocks
  return UOp(Ops.BLOCK, src=tuple(new_srcs), arg=arg)

block_merge = PatternMatcher([
  (UPat(Ops.BLOCK, src=UPat(Ops.BLOCK), name="x"), merge_block),
])

def linearize_uop(sink:UOp, skip_check:bool=not __debug__) -> list[UOp]:
  assert sink.op is Ops.SINK, f"sink isn't sink, it's {sink.op}"

  print(_get_code(blocks_in_context.patterns[0][0], True)[0])

  sink = graph_rewrite(sink, blocks_in_context, track_children=True, name="Linearizer: Create Blocks")
  print("rewritten")

  for i in range(10):
    out_sink = graph_rewrite(sink, block_merge, track_children=True)
    if out_sink is sink: break
    sink = out_sink

  assert sink.op is Ops.BLOCK and len(sink.src) == 0
  return list(sink.arg.lst)


  """
  # get block contexts
  temp_block_ctxs: dict[UOp, list[UOp]] = {}
  for u in sink.toposort:
    for s in u.src:
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
    temp_block_ctxs[u] = sorted(dedup(this_block_ctx), key=lambda x: x.tuplize)

  # make final block_ctxs, add BLOCKSTART to block_ctxs for IF and RANGE
  block_ctxs: dict[UOp, tuple[UOp, ...]] = {}
  for u in sink.toposort:
    block_ctxs[u] = ((UOp(Ops.BLOCKSTART, src=(u,)),) + tuple(temp_block_ctxs[u])) if u.op in {Ops.IF, Ops.RANGE} else tuple(temp_block_ctxs[u])
  """
