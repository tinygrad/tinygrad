from __future__ import annotations
import heapq
from collections import defaultdict
from dataclasses import dataclass, replace
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, GroupOp
from tinygrad.helpers import dedup, partition, all_same, flatten, getenv

# NOTE: any toposort should be valid here, unlike last time this isn't required, it's just for speed
def block_reorder(lst:list[UOp]) -> list[UOp]:
  in_this_block = set(lst)
  local_children: defaultdict[UOp, list[UOp]] = defaultdict(list)
  in_degree:dict[UOp, int] = {}
  priorities:dict[UOp, int] = {}

  # get local children and assign priorities
  # NOTE: this requires the lst be locally toposorted
  for u in reversed(lst):
    in_degree[u] = 0
    for s in u.src:
      if s in in_this_block:
        local_children[s].append(u)
        in_degree[u] += 1
    # put loads in the beginning of the block and prevent priority inversion. hack for BARRIER grouping too
    priority = [0] + [priorities[x] for x in local_children[u]]
    if u.op is Ops.LOAD: priority.append(-1000)
    if u.op is Ops.BARRIER: priority.append(-1500)
    priorities[u] = min(priority)

  # number the uops in "ideal" order
  nkey = {u:i for i,u in enumerate(sorted(lst, key=lambda x: (priorities[x],)+x.tuplize))}

  # then force then to be toposorted in as close to the ideal order as possible
  heapq.heapify(heap:=[(nkey[u],u) for u in lst if in_degree[u] == 0])
  newlst = []
  while heap:
    newlst.append(u:=heapq.heappop(heap)[1])
    for v in local_children[u]:
      in_degree[v] -= 1
      if in_degree[v] == 0: heapq.heappush(heap, (nkey[v],v))

  assert len(newlst) == len(lst), f"len mismatch {len(newlst)} != {len(lst)}"
  return newlst

# ***** basic block *****

def disp(y:UOp) -> str:
  if y.op is Ops.IF: return f'IF{id(y)}'
  if y.op is Ops.RANGE: return str(y.arg)
  return "<NONE>"

@dataclass(frozen=True, eq=False)
class BasicBlock:
  lst: tuple[UOp, ...]
  ctx: tuple[UOp, ...] = ()
  end: UOp|None = None
  cnt: int = 0
  child_ctx: tuple[UOp, ...]|None = None
  def __lt__(self, _:BasicBlock): raise RuntimeError("no comparing basic blocks")
  def __repr__(self):
    return f"{(str(disp(self.end))+' ') if self.end is not None else ''}"+f'f{self.cnt} '+\
           f"{[disp(y) for y in self.ctx]} {[disp(y) for y in self.child_ctx] if self.child_ctx is not None else '-'} "+\
           f"{len(self.lst)}" + "\n" + '\n'.join([str(x.op) for x in self.lst])
  def last_ctx(self): return self.child_ctx if self.child_ctx is not None else self.ctx

def _sort_ctx(inp): return tuple(sorted(dedup(inp), key=lambda x: x.tuplize))

# ***** block context *****

@dataclass
class BlockContext:
  child_count: dict[UOp, int]
  block_ctxs: dict[UOp, tuple[UOp, ...]]
  child_ctxs: dict[UOp, tuple[UOp, ...]]
  def last_ctx(self, u): return self.child_ctxs.get(u, self.block_ctxs[u])
  @staticmethod
  def from_sink(sink:UOp) -> BlockContext:
    # get children and all block contexts
    ctx = BlockContext({}, {}, {})
    for u in sink.toposort():
      this_block_ctx: list[UOp] = []
      ctx.child_count[u] = 0

      # get children and accumulate the last_ctx
      for s in u.src:
        # NOTE: if a parent appears multiple times in the src, it counts multiple times as a child
        ctx.child_count[s] += 1
        this_block_ctx += ctx.last_ctx(s)

      # save the block ctx
      ctx.block_ctxs[u] = _sort_ctx(this_block_ctx)

      # RANGE/IF add to the next ctx
      # STORE subtract from the next ctx
      if u.op in {Ops.RANGE, Ops.IF}: ctx.child_ctxs[u] = _sort_ctx(ctx.block_ctxs[u] + (u,))
      elif u.op is Ops.STORE:
        # handle STORE to registers (replacement for ASSIGN)
        if u.src[0].op is Ops.DEFINE_REG and u.src[0].arg[0] == "register":
          ctx.child_ctxs[u] = tuple([y for y in ctx.last_ctx(u.src[1]) if y not in u.src[0].src[1:]])
        # ugh, deal with non-reduce locals. probably wrong
        elif any(x.op is Ops.DEFINE_REG and x.arg[0] == "local" for x in u.src[0].toposort()):
          idx_context, store_context = ctx.last_ctx(u.src[0]), ctx.last_ctx(u.src[1])
          ctx.child_ctxs[u] = tuple([y for y in store_context if y not in idx_context and y.op is Ops.RANGE])
        else: ctx.child_ctxs[u] = ()
    return ctx

# ***** make blocks *****

DONT_PLACE_IN_BLOCK = {Ops.DEFINE_REG, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.CONST}

def is_store_to_register(u:UOp) -> bool:
  return u.op is Ops.STORE and u.src[0].op is Ops.DEFINE_REG and u.src[0].arg[0] == "register"

def add_blockends(base_block:UOp, new_ctx:tuple[UOp, ...], current_ctx:tuple[UOp, ...], cnt:int=1) -> UOp:
  ends_to_add = [z for z in new_ctx if z not in current_ctx]
  while len(ends_to_add):
    r:UOp = ends_to_add.pop(-1)
    new_ctx = tuple([z for z in new_ctx if z is not r])
    end_uop = UOp(Ops.ENDIF if r.op is Ops.IF else Ops.ENDRANGE, src=(r,))
    base_block = UOp(Ops.BLOCKEND, src=(base_block,)*cnt, arg=BasicBlock((end_uop,), tuple(new_ctx), end=r, cnt=cnt))
  return base_block

def make_block_bottom_up(ctx:BlockContext, x:UOp):
  if x.op is Ops.BLOCKSTART:
    current_ctx, child_ctx = x.arg
    lst = list(x.src)
    child_count = 1
  else:
    current_ctx, child_count, child_ctx = ctx.block_ctxs[x], ctx.child_count[x], ctx.child_ctxs.get(x, None)
    lst = [x]

  # count of times we've seen this block, or a seed for a new block if we can't merge it
  unmergable: defaultdict[UOp, int] = defaultdict(int)
  blockseeds = defaultdict(list)

  # add the srcs of this to the frontier
  # NOTE: things may be in here multiple times, that's okay
  frontier_nodes = list(flatten(y.src[::-1] for y in lst))
  while len(frontier_nodes):
    u = frontier_nodes.pop(0)
    if u.op not in DONT_PLACE_IN_BLOCK and ctx.child_count[u] == unmergable[u]+1:
      # count is correct
      if (newctx:=ctx.block_ctxs[u]) == current_ctx:
        # block has same context, merge it, and put the srcs on the frontier
        lst.append(u)
        frontier_nodes.extend(u.src[::-1])
      else:
        # block has different context, add it to blockseeds
        blockseeds[(newctx, ctx.child_ctxs.get(u, None))].append(u)
      del unmergable[u]
    else:
      # count is incorrect (or it's DONT_PLACE_IN_BLOCK), add it to unmergable
      unmergable[u] += 1

  # add unmergables to sources
  srcs = []
  for u,cnt in unmergable.items(): srcs += [add_blockends(u, ctx.block_ctxs[u], current_ctx, cnt=cnt)]*cnt

  # add blockseeds, with blockends as needed
  for (new_ctx, new_child_ctx), v in blockseeds.items():
    base_block = UOp(Ops.BLOCKSTART, src=tuple(v), arg=(new_ctx, new_child_ctx))
    srcs.append(add_blockends(base_block, new_ctx, current_ctx))

  lst = lst[::-1]
  if getenv("BLOCK_REORDER", 1): lst = block_reorder(lst)
  bb = BasicBlock(tuple(lst), ctx=current_ctx, cnt=child_count, child_ctx=child_ctx)
  return UOp(Ops.BLOCK, src=tuple(srcs), arg=bb)

block_create = PatternMatcher([
  (UPat(GroupOp.All-DONT_PLACE_IN_BLOCK.union({Ops.BLOCK, Ops.BLOCKEND, Ops.BLOCKFINAL}), name="x"), make_block_bottom_up),
])

# ***** blockend merging ****

def merge_blockends(sink:UOp) -> UOp|None:
  # only run on the final BLOCK with the SINK in it
  if sink.arg.lst[-1].op is not Ops.SINK: return None
  # combine matching BLOCKENDS, the keys of this dictionary are the RANGE UOps, values are the BLOCKENDs
  blockends_to_arg: dict[UOp, list[UOp]] = {}
  for be in sink.toposort():
    if be.op is Ops.BLOCKEND: blockends_to_arg.setdefault(be.arg.end, []).append(be)
  new_forks = {}
  for k,v in blockends_to_arg.items():
    # NOTE: if any BLOCKEND is the parent of any other with the same arg, this algo fails
    if len(v) > 1:
      bb = BasicBlock(v[0].arg.lst, _sort_ctx(flatten([y.arg.ctx for y in v])), k, cnt=sum(y.arg.cnt for y in v))
      out = UOp(Ops.BLOCKEND, src=tuple(flatten([x.src for x in v])), arg=bb)
      # NOTE: bb.ctx != u.arg.ctx can cause problems here
      for u in v: new_forks[u] = out
  if len(new_forks) == 0: return None
  return sink.substitute(new_forks)

pm_blockend_merge = PatternMatcher([(UPat(Ops.BLOCK, name="sink"), merge_blockends)])

# ***** block merging ****

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
  arg = replace(x.arg, lst=tuple(flatten([y.arg.lst for y in mergable_blocks]))+x.arg.lst)
  return UOp(x.op, src=tuple(flatten([y.src for y in mergable_blocks])+unmergable_blocks), arg=arg)

def remove_blockend(x:UOp):
  # if there's any remaining blocks that need to go in this BLOCKEND, we don't remove it
  if any(x.arg.end in y.arg.ctx for y in x.src if y.op in {Ops.BLOCK, Ops.BLOCKEND}): return None

  if (parent_blocks := [y for y in x.src if y.op is Ops.BLOCK and y.arg.child_ctx is not None and x.arg.end in y.arg.child_ctx]):
    assert all_same(parent_blocks), f"should never have two parent blocks (has {len(parent_blocks)})"
    parent_block = parent_blocks[0]
    # When ASSIGN was replaced with STORE, the count might be off
    # Use the actual count of parent blocks found
    actual_cnt = len(parent_blocks)
    # range needs DEFINE_ACC to be before the range (never in DEFINE_ACC for if)
    # Check if x.arg.end is in the sources, including inside BLOCK sources
    def has_range_in_sources(u:UOp) -> bool:
      if u.op != Ops.DEFINE_REG: return False
      for s in u.src:
        if s is x.arg.end: return True
        # Check if the source is a BLOCK that contains the range
        if s.op is Ops.BLOCK and x.arg.end in s.toposort(): return True
      return False
    early_ops, late_ops = partition(x.arg.lst, has_range_in_sources)
    # NOTE: we have to add a barrier at the start if barrier is used in the range
    if x.op is Ops.BLOCKEND and any(y.op is Ops.BARRIER for y in late_ops) and late_ops[-1].op is Ops.ENDRANGE:
      late_ops = [UOp(Ops.BARRIER)] + late_ops
    arg = BasicBlock(tuple(early_ops)+parent_block.arg.lst+tuple(late_ops), tuple([y for y in x.arg.ctx if y is not x.arg.end]), cnt=actual_cnt)
    return UOp(Ops.BLOCK, src=tuple(y for y in x.src if y is not parent_block)+parent_block.src, arg=arg)

block_merge = PatternMatcher([
  (UPat((Ops.BLOCK, Ops.BLOCKEND), name="x"), merge_block),
  (UPat(Ops.BLOCKEND, name="x"), remove_blockend),
])

# ****** finalize ******

def finalize(sink:UOp) -> UOp:
  if sink.op is not Ops.BLOCK or not all(x.op in DONT_PLACE_IN_BLOCK for x in sink.src):
    if sink.op is not Ops.BLOCK:
      raise RuntimeError(f"linearize failure: sink is {sink.op} not BLOCK")
    bad_ops = [x.op for x in sink.src if x.op not in DONT_PLACE_IN_BLOCK]
    raise RuntimeError(f"linearize failure: found ops not in DONT_PLACE_IN_BLOCK: {bad_ops}")

  # Fix DEFINE_REG operations that have BLOCK or BLOCKFINAL sources
  # This can happen when RANGE operations are wrapped in BLOCKs during block creation
  fixed_sources = []
  additional_sources = []  # Sources that need to be added to the list

  for src in sink.src:
    if src.op == Ops.DEFINE_REG and src.arg[0] == "register":
      # Check if any source is a BLOCK or BLOCKFINAL that contains a RANGE
      new_src = []
      for s in src.src:
        if s.op == Ops.BLOCK and len(s.arg.lst) > 0 and s.arg.lst[0].op == Ops.RANGE:
          # Replace BLOCK source with the RANGE inside it
          new_src.append(s.arg.lst[0])
        elif s.op == Ops.BLOCKFINAL and hasattr(s.arg, 'lst') and len(s.arg.lst) > 0:
          # Check if this BLOCKFINAL contains just CONST and RANGE
          if len(s.arg.lst) == 2 and s.arg.lst[0].op == Ops.CONST and s.arg.lst[1].op == Ops.RANGE:
            # This is likely the RANGE we need, but we also need the CONST
            new_src.append(s.arg.lst[1])
            # Make sure the CONST is in our sources
            if s.arg.lst[0] not in fixed_sources and s.arg.lst[0] not in sink.src:
              additional_sources.append(s.arg.lst[0])
          else:
            new_src.append(s)
        else:
          new_src.append(s)
      if new_src != list(src.src):
        src = src.replace(src=tuple(new_src))
    fixed_sources.append(src)

  # Add any additional sources we found
  fixed_sources.extend(additional_sources)

  # place the early things
  # For DEFINE_REG with sources, ensure sources come before the DEFINE_REG
  early_things = dedup(fixed_sources)

  # Separate DEFINE_REGs with sources from other operations
  define_regs_with_sources = []
  other_ops = []
  for op in early_things:
    if op.op == Ops.DEFINE_REG and len(op.src) > 0:
      define_regs_with_sources.append(op)
    else:
      other_ops.append(op)

  # Sort other ops
  sorted_others = sorted(other_ops, key=lambda x: x.tuplize)

  # For each DEFINE_REG, ensure its sources are in the sorted list
  final_early = []
  added = set()
  for op in sorted_others:
    if op not in added:
      final_early.append(op)
      added.add(op)

  # Collect all ops that will be in the main list to avoid duplicates
  main_lst = list(sink.arg.lst)
  main_ops_set = set(main_lst)

  # Now add DEFINE_REGs, ensuring their sources are already added
  for dr in sorted(define_regs_with_sources, key=lambda x: x.tuplize):
    # Add any sources that aren't already added and aren't in main list
    for src in dr.src:
      if src not in added and src.op in {Ops.CONST, Ops.RANGE} and src not in main_ops_set:
        final_early.append(src)
        added.add(src)
    # Add the DEFINE_REG itself
    if dr not in added:
      final_early.append(dr)
      added.add(dr)

  lst = final_early + main_lst

  return UOp(Ops.BLOCKFINAL, arg=BasicBlock(tuple(lst)))

pm_finalize = PatternMatcher([(UPat(Ops.BLOCK, name="sink"), finalize)])
