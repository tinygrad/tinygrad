from tinygrad.ops import UOp, Ops, graph_rewrite, PatternMatcher, UPat, GroupOp
from tinygrad.dtype import PtrDType
from tinygrad.helpers import dedup, partition, all_same, flatten
from tinygrad.codegen.linearize import BasicBlock
from line_profiler import profile
from collections import defaultdict

def _strip_start(y:UOp): return [z.src[0] if z.op is Ops.BLOCKSTART else z for z in y]
def _sort_ctx(inp): return tuple(sorted(dedup(inp), key=lambda x: x.tuplize))

def make_block(ctx, x:UOp):
  # this is key to making this algorithm work. we recover the old x by replacing the srcs
  fixed_x = x.replace(src=tuple(y.arg.lst[-1] for y in x.src))

  # compute the new context
  # if this is a RANGE or an IF, we add it to this block context
  _new_ctx = [UOp(Ops.BLOCKSTART, src=(fixed_x,))] if fixed_x.op in {Ops.RANGE, Ops.IF} else []
  _old_ctxs = []
  for y in x.src:
    uop: UOp = y.arg.lst[-1]
    # if it's a STORE remove everything if non local
    if uop.op is Ops.STORE:
      if isinstance(uop.src[0].dtype, PtrDType) and uop.src[0].dtype.local:
        # get these ctxs from the blocks
        idx_context, store_context = y.src[0].arg.ctx, y.src[1].arg.ctx
        _old_ctx = [z for z in store_context if z not in idx_context and z.op is Ops.RANGE]
      else:
        # for nonlocal store, we clear the context
        _old_ctx = []
    else:
      _old_ctx = _strip_start(y.arg.ctx)
      # if it's an ASSIGN, remove any RANGEs used
      if uop.op is Ops.ASSIGN:
        assert uop.src[0].op is Ops.DEFINE_ACC
        _old_ctx = [z for z in _old_ctx if z not in uop.src[0].src[1:]]
    _old_ctxs.append(_old_ctx)
  tuple_ctx = _sort_ctx(_new_ctx+flatten(_old_ctxs))

  # a block is unmergable if it has children or it has a different context
  unmergable_blocks, mergable_blocks = [], []
  seen = set()
  for y in x.src:
    if y in seen: continue
    seen.add(y)
    if y.arg.cnt == 1 and y.arg.ctx == tuple_ctx:
      mergable_blocks.append(y)
    else:
      # block is unmergable
      um_ctx = tuple(_strip_start(y.arg.ctx))
      if um_ctx != tuple_ctx:
        ends_to_add = [z for z in um_ctx if z not in tuple_ctx][::-1]
        extra_ends = [z for z in um_ctx if z in tuple_ctx]
        while len(ends_to_add):
          r = ends_to_add[0]
          ends_to_add = ends_to_add[1:]
          end_uop = UOp(Ops.ENDIF if r.op is Ops.IF else Ops.ENDRANGE, src=(r,))
          y = UOp(Ops.BLOCKEND, src=(y,), arg=BasicBlock(_sort_ctx(ends_to_add+extra_ends), (end_uop,), r, cnt=1))
      unmergable_blocks.append(y)

  # create the block (TODO: we don't actually need to track that list)
  arg = BasicBlock(tuple_ctx, tuple(flatten([y.arg.lst for y in mergable_blocks])+[fixed_x]), cnt=len(ctx[fixed_x]))
  new_srcs = flatten([y.src for y in mergable_blocks])+unmergable_blocks
  return UOp(Ops.BLOCK, src=tuple(new_srcs), arg=arg)

# this will wrap every UOp in a BLOCK, top down
blocks_in_context = PatternMatcher([
  # NOTE: this pattern doesn't match 0 due to the early reject
  (UPat(GroupOp.All-{Ops.BLOCK, Ops.BLOCKEND}, src=(), name="x"), make_block),
  # make blocks from non block and const
  (UPat(GroupOp.All-{Ops.BLOCK, Ops.BLOCKEND}, src=UPat((Ops.BLOCK, Ops.BLOCKEND)), name="x"), make_block),
])

def merge_blockend(x:UOp):
  unmergable_blocks, mergable_blocks = [], []
  mergable_dict = defaultdict(int)
  for y in x.src:
    if y.op is Ops.BLOCK and x.arg.end in y.arg.ctx: mergable_dict[y] += 1
    else: unmergable_blocks.append(y)
  for k,v in mergable_dict.items():
    if v == k.arg.cnt: mergable_blocks.append(k)
    else: unmergable_blocks.extend([k]*v)
  if len(mergable_blocks) == 0: return None

  # create the block (TODO: we don't actually need to track that list)
  parent_lst = tuple(flatten([y.arg.lst for y in mergable_blocks]))
  new_srcs = flatten([y.src for y in mergable_blocks])+unmergable_blocks
  return UOp(Ops.BLOCKEND, src=tuple(new_srcs), arg=x.arg.replace(lst=parent_lst+x.arg.lst))

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
      arg=BasicBlock(_sort_ctx([y for y in x.arg.ctx if y is not x.arg.end]),
                     tuple(early_ops)+parent_block.arg.lst+tuple(late_ops), cnt=x.arg.cnt))

def merge_block(x:UOp):
  unmergable_blocks, mergable_blocks = [], []
  mergable_dict = defaultdict(int)
  for y in x.src:
    if y.op is Ops.BLOCK and x.arg.ctx == y.arg.ctx: mergable_dict[y] += 1
    else: unmergable_blocks.append(y)
  for k,v in mergable_dict.items():
    if v == k.arg.cnt: mergable_blocks.append(k)
    else: unmergable_blocks.extend([k]*v)
  if len(mergable_blocks) == 0: return None

  # create the block (TODO: we don't actually need to track that list)
  parent_lst = tuple(flatten([y.arg.lst for y in mergable_blocks]))
  new_srcs = flatten([y.src for y in mergable_blocks])+unmergable_blocks
  return UOp(Ops.BLOCK, src=tuple(new_srcs), arg=x.arg.replace(lst=parent_lst+x.arg.lst))

block_merge = PatternMatcher([
  (UPat(Ops.BLOCK, name="x"), profile(merge_block)),
  (UPat(Ops.BLOCKEND, name="x"), merge_blockend),
  (UPat(Ops.BLOCKEND, name="x"), remove_blockend),
])

def linearize_uop(sink:UOp, skip_check:bool=not __debug__) -> list[UOp]:
  assert sink.op is Ops.SINK, f"sink isn't sink, it's {sink.op}"

  sink = graph_rewrite(sink, blocks_in_context, ctx=sink.get_children_map(), name="Linearizer: Create Blocks")

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
