from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, replace
from tinygrad.ops import UOp, Ops, graph_rewrite, PatternMatcher, UPat
from tinygrad.dtype import PtrDType
from tinygrad.helpers import dedup, partition, all_same, flatten

def disp(y:UOp) -> str:
  if y.op is Ops.IF: return f'IF{id(y)}'
  if y.op is Ops.RANGE: return str(y.arg)
  return "<NONE>"

def _sort_ctx(inp): return sorted(dedup(inp), key=lambda x: x.tuplize)
def _get_ctx(deduped_blocks:list[UOp]):
  list_ctx = []
  for y in deduped_blocks:
    list_ctx.extend(list(y.arg.child_ctx if y.arg.child_ctx is not None else y.arg.ctx))
  return _sort_ctx(list_ctx)

@dataclass(frozen=True, eq=False)
class BasicBlock2:
  lst: list[UOp]
  ctx: list[UOp]
  end: UOp|None = None
  cnt: int = 0
  child_ctx: list[UOp]|None = None
  def __lt__(self, o:BasicBlock2): raise Exception("no comparing basic blocks")
  def __repr__(self):
    return f"{(str(disp(self.end))+' ') if self.end is not None else ''}"+f'f{self.cnt} '+\
           f"{[disp(y) for y in self.ctx]} {[disp(y) for y in self.child_ctx] if self.child_ctx is not None else '-'} "+\
           f"{len(self.lst)}" + "\n" + '\n'.join([str(x.op) for x in self.lst])

def make_block(fixed_x:UOp, child_len:dict[UOp, int], blocks:dict[UOp, UOp]):
  # all input blocks to this one, deduped
  deduped_blocks = list({blocks[y]:None for y in fixed_x.src})

  # compute the new context
  tuple_ctx = _get_ctx(deduped_blocks)

  # does this block modify the ctx?
  # RANGE/IF add to the next ctx
  # STORE/ASSIGN subtract from the next ctx
  child_ctx = None
  if fixed_x.op in {Ops.RANGE, Ops.IF}: child_ctx = [fixed_x]
  elif fixed_x.op is Ops.STORE:
    # ugh, deal with non-reduce locals. probably wrong
    if isinstance(fixed_x.src[0].dtype, PtrDType) and fixed_x.src[0].dtype.local:
      idx_context, store_context = blocks[fixed_x.src[0]].arg.ctx, blocks[fixed_x.src[1]].arg.ctx
      child_ctx = [y for y in store_context if y not in idx_context and y.op is Ops.RANGE]
    else: child_ctx = []
  elif fixed_x.op is Ops.ASSIGN:
    assert fixed_x.src[0].op is Ops.DEFINE_ACC
    child_ctx = [y for y in blocks[fixed_x.src[1]].arg.ctx if y not in fixed_x.src[0].src[1:]]

  # a block is unmergable if it has children or it has a different context
  unmergable_blocks = []
  lst = []
  merged_srcs = []
  for y in deduped_blocks:
    if y.arg.cnt == 1 and y.arg.ctx == tuple_ctx:
      lst += y.arg.lst
      merged_srcs += list(y.src)
    else:
      # block is unmergable
      if y.arg.ctx != tuple_ctx:
        ends_to_add = [z for z in y.arg.ctx if z not in tuple_ctx]
        new_ctx = y.arg.ctx
        while len(ends_to_add):
          r:UOp = ends_to_add.pop(-1)
          new_ctx = [z for z in new_ctx if z is not r]
          end_uop = UOp(Ops.ENDIF if r.op is Ops.IF else Ops.ENDRANGE, src=(r,))
          y = UOp(Ops.BLOCKEND, src=(y,), arg=BasicBlock2([end_uop], new_ctx, end=r, cnt=1))
      unmergable_blocks.append(y)

  # create the block
  arg = BasicBlock2(lst+[fixed_x], tuple_ctx, cnt=child_len[fixed_x], child_ctx=child_ctx)
  return UOp(Ops.BLOCK, src=tuple(merged_srcs+unmergable_blocks), arg=arg)

def merge_block(x:UOp):
  unmergable_blocks, mergable_blocks = [], []
  mergable_dict: defaultdict[UOp, int] = defaultdict(int)
  for y in x.src:
    if y.op is Ops.BLOCK and x.op is Ops.BLOCK and x.arg.ctx == y.arg.ctx: mergable_dict[y] += 1
    elif y.op is Ops.BLOCK and x.op is Ops.BLOCKEND and x.arg.end in y.arg.ctx: mergable_dict[y] += 1
    else: unmergable_blocks.append(y)
  for k,v in mergable_dict.items():
    if v == k.arg.cnt: mergable_blocks.append(k)
    else: unmergable_blocks.extend([k]*v)
  if len(mergable_blocks) == 0: return None
  del mergable_dict

  # create the block
  lst = flatten([y.arg.lst for y in mergable_blocks])+x.arg.lst
  arg = replace(x.arg, lst=lst)
  return UOp(x.op, src=tuple(flatten([y.src for y in mergable_blocks])+unmergable_blocks), arg=arg)

def remove_blockend(x:UOp):
  # if there's any
  if any(x.arg.end in y.arg.ctx for y in x.src): return None

  parent_blocks = [y for y in x.src if y.op is Ops.BLOCK and y.arg.child_ctx is not None and x.arg.end in y.arg.child_ctx]
  assert all_same(parent_blocks), f"should never have two parent blocks (has {len(parent_blocks)})"
  if len(parent_blocks) > 0:
    parent_block = parent_blocks[0]
    assert len(parent_blocks) == parent_block.arg.cnt
    # range needs DEFINE_ACC to be before the range (never in DEFINE_ACC for if)
    early_ops, late_ops = partition(x.arg.lst, lambda y: y.op is Ops.DEFINE_ACC and x.arg.end in y.src)
    # NOTE: we have to add a barrier at the start if barrier is used in the range
    if x.op is Ops.BLOCKEND and any(y.op is Ops.BARRIER for y in late_ops) and late_ops[-1].op is Ops.ENDRANGE:
      late_ops = [UOp(Ops.BARRIER)] + late_ops
    lst = early_ops+parent_block.arg.lst+late_ops
    arg = BasicBlock2(lst, [y for y in x.arg.ctx if y is not x.arg.end], cnt=x.arg.cnt)
    return UOp(Ops.BLOCK, src=tuple(y for y in x.src if y is not parent_block)+parent_block.src, arg=arg)

block_merge = PatternMatcher([
  (UPat((Ops.BLOCK, Ops.BLOCKEND), name="x"), merge_block),
  (UPat(Ops.BLOCKEND, name="x"), remove_blockend),
])

def get_child_counts(toposink:dict[UOp, None]) -> dict[UOp, int]:
  children: dict[UOp, dict[UOp, None]] = {}
  for u in toposink:
    for s in u.src: children.setdefault(s, {})[u] = None
  return {k:len(v) for k,v in children.items()}

def wrap_in_blocks(sink:UOp):
  # do toposort
  toposink = sink.toposort

  # get children
  child_len = get_child_counts(toposink)
  child_len[sink] = 0

  # this will wrap every UOp in a BLOCK, top down
  blocks = {}
  for x in toposink: blocks[x] = make_block(x, child_len, blocks)
  return blocks[sink]

def linearize_uop(sink:UOp, skip_check:bool=not __debug__) -> list[UOp]:
  assert sink.op is Ops.SINK, f"sink isn't sink, it's {sink.op}"

  # wrap all uops in blocks
  sink = wrap_in_blocks(sink)

  # combine matching BLOCKENDS, the keys of this dictionary are the RANGE UOps, values are the BLOCKENDs
  blockends_to_arg: dict[UOp, list[UOp]] = {}
  for be in sink.toposort:
    if be.op is Ops.BLOCKEND: blockends_to_arg.setdefault(be.arg.end, []).append(be)
  new_forks = {}
  for k,v in blockends_to_arg.items():
    # NOTE: if any BLOCKEND is the parent of any other with the same arg, this algo fails
    if len(v) > 1:
      bb = BasicBlock2(v[0].arg.lst, _sort_ctx(flatten([y.arg.ctx for y in v])), k, cnt=len(v))
      out = UOp(Ops.BLOCKEND, src=tuple(flatten([x.src for x in v])), arg=bb)
      for u in v: new_forks[u] = out
  sink = sink.substitute(new_forks)

  # merge blocks
  sink = graph_rewrite(sink, block_merge, name="Linearizer: Merge Blocks")

  assert sink.op is Ops.BLOCK and len(sink.src) == 0
  return list(sink.arg.lst)
