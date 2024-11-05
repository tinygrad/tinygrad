import sys
from collections import defaultdict, deque
from typing import Set, Tuple, List, Dict, DefaultDict
from tinygrad.ops import GroupOp, MetaOps, ReduceOps, UOp, UnaryOps
from tinygrad.helpers import FUSE_CONV_BW, FUSE_ARANGE, dedup, merge_dicts
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.engine.lazy import LazyBuffer
from tinygrad.device import Buffer

# creation can recurse a lot
sys.setrecursionlimit(10000)

def _recurse_lb(buf:LazyBuffer, realizes:Dict[LazyBuffer, None], allbufs:Dict[LazyBuffer, None],
  children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]], assign_targets:Set[Buffer], double_reduces:Dict[LazyBuffer, None], ctx):
  """recursively search the entire graph for all LazyBuffers, insert realizes after expands"""
  if buf in allbufs or buf.base.op is MetaOps.CONST: return None
  if buf.base.realized is not None: return realizes.setdefault(buf.base)
  if ctx.buf_uops[buf.buffer] in ctx.realizes: realizes[buf] = None
  if buf.op in GroupOp.Reduce and buf.srcs[0].base.op is buf.op and buf.srcs[0] is not buf.srcs[0].base: double_reduces[buf] = None
  allbufs[buf] = None
  if buf.op is MetaOps.ASSIGN: assign_targets.add(buf.buffer)
  for x in buf.srcs:
    if x.base.realized is None: children[x.base][buf] = None
    _recurse_lb(x.base, realizes, allbufs, children, assign_targets, double_reduces, ctx)

def _recursive_group(tr:LazyBuffer, st:ShapeTracker, r:LazyBuffer, children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]],
                     realizes:Dict[LazyBuffer, None], reduce_for_op:Dict[LazyBuffer, LazyBuffer], group:Dict[LazyBuffer, None],
                     cache:Dict[Tuple[LazyBuffer, ShapeTracker], None]) -> None:
  """recursively search the LazyBuffer for groupable children, realize the LazyBuffer if a child can't group"""
  if (tr, st) in cache: return
  cache.setdefault((tr, st))
  if tr in realizes and tr is not r:
    # can only fuse contiguous
    # max one reduceop per kernel
    if not st.contiguous or st.size != r.st.size or tr in reduce_for_op: group.setdefault(r)
    return group.setdefault(tr)
  for tr_next in children[tr]:
    # max one reduceop per kernel
    if tr_next.op in GroupOp.Reduce: return group.setdefault(r)
    # can only fuse contiguous
    if len(st_childs:=dedup(s for s in tr_next.srcs if s.base == tr)) > 1: return group.setdefault(r)
    _recursive_group(tr_next, st+st_childs[0].st, r, children, realizes, reduce_for_op, group, cache)

def _get_isolated_children(r:LazyBuffer, reduce_for_op:Dict[LazyBuffer, LazyBuffer], children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]],\
    realizes:Dict[LazyBuffer, None], group:Dict[LazyBuffer, None]) -> Dict[LazyBuffer, None]:
  rc_parents, cache = deque(group), set()
  while rc_parents:
    if (p:=rc_parents.pop()) in cache: continue
    cache.add(p)
    # max one reduceop per kernel
    if p.op in GroupOp.Reduce: return {}
    rc_parents.extend(x.base for x in p.srcs if x.base.realized is None and x.base is not r)
  # search descendants of the reduceop that can cleanly group
  descendants: Dict[LazyBuffer, None] = {}
  for tr in group: _recursive_group(tr, tr.st, tr, children, realizes, reduce_for_op, descendants, cache={})
  return merge_dicts([group, {} if any(tr in group for tr in descendants) else descendants])

def get_realizes(outs:List[LazyBuffer], ctx) -> Tuple[List[List[UOp]], Dict[Buffer, LazyBuffer], Set[Buffer]]:
  """search the graph for all the LazyBuffers that need to realize"""
  realizes: Dict[LazyBuffer, None] = {}
  allbufs: Dict[LazyBuffer, None] = {}
  children: DefaultDict[LazyBuffer, Dict[LazyBuffer, None]] = defaultdict(dict)
  assign_targets: Set[Buffer] = set()
  double_reduces: Dict[LazyBuffer, None] = {}
  for out in outs: _recurse_lb(out, realizes, allbufs, children, assign_targets, double_reduces, ctx)

  # find all reduces, and pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce (or a contig child)
  reduce_for_op: Dict[LazyBuffer, LazyBuffer] = {}
  reduce_of_const: List[LazyBuffer] = []
  for r in allbufs:
    if r.op not in GroupOp.Reduce or r in realizes: continue

    group: Dict[LazyBuffer, None] = {}
    _recursive_group(r, r.st, r, children, realizes, reduce_for_op, group, cache={})
    # max one reduceop per kernel
    can_chase = all(tr not in reduce_for_op for tr in group)
    # TODO: forced_realize exists because the scheduler is incapable of checking for self-contained DAGs
    forced_realize = r in group
    if not forced_realize and len(group) > 1:
      group = _get_isolated_children(r, reduce_for_op, children, realizes, group)
    # can only fuse assign if no other assign_target is used in the kernel
    if not forced_realize and any(x.op is MetaOps.ASSIGN for x in group):
      parents = deque((r, *group))
      while parents and not forced_realize:
        if (p:=parents.pop().base).is_realized() or p in realizes:
          if p.is_realized() and p.buffer in assign_targets and not any(x.buffer is p.buffer for x in group): forced_realize, can_chase = True, False
          continue
        parents.extend(p.srcs)
    if forced_realize or not group:
      tr = r
      if can_chase:
        # can chase this down to contiguous children
        st = tr.st
        while len(children[tr]) == 1:
          tr_next = next(iter(children[tr]))
          st_childs = dedup(s for s in tr_next.srcs if s.base is tr)
          if len(st_childs) > 1: break
          if st.size != st_childs[0].st.size: break
          st = st + st_childs[0].st
          if not st.contiguous or tr_next.op in GroupOp.Reduce: break
          tr = tr_next
        # don't cast to higher size before store (tr cannot be realized if forced_realize)
        if tr.op is UnaryOps.CAST and tr.arg.itemsize > tr.srcs[0].dtype.itemsize:
          tr = tr.srcs[0].base
        reduce_for_op[tr] = r
      realizes[tr] = None
    else: reduce_for_op.update((tr, r) for tr in group)
    if FUSE_ARANGE and r.op is ReduceOps.SUM and r.srcs[0].base.op is MetaOps.CONST: reduce_of_const.append(r)

  # fuse double reduces with no other child
  if FUSE_CONV_BW:
    for reduceop in double_reduces:
      top_reduce = reduceop.base.srcs[0].base
      if len(children[top_reduce]) == 1:
        del realizes[top_reduce]
        if (ubuf:=ctx.buf_uops[top_reduce.buffer]) in ctx.realizes: del ctx.realizes[ubuf]

  for r in reduce_of_const:
    group = {tr:None for tr,rop in reduce_for_op.items() if rop is r}
    if any(tr.forced_realize for tr in group) or any(x.base in group for x in outs): continue
    kernel_children = {c for tr in group for c in children[tr] if c.op not in {MetaOps.COPY, MetaOps.BUFFER_VIEW}}
    if len(kernel_children) == 0: continue
    for tr in group:
      del realizes[tr]
      if (ubuf:=ctx.buf_uops[tr.buffer]) in ctx.realizes: del ctx.realizes[ubuf]
  output_groups: DefaultDict[LazyBuffer, List[UOp]] = defaultdict(list)
  lazybufs_to_realize: Dict[Buffer, LazyBuffer] = {}
  for buf in realizes:
    if buf.realized is None:
      if (dup:=lazybufs_to_realize.get(buf.buffer)) is not None:
        raise RuntimeError(f"can't double realize in one schedule, Buffer is realizing both {dup} and {buf}")
      lazybufs_to_realize[buf.buffer] = buf
      output_groups[reduce_for_op.get(buf, buf)].append(ubuf:=ctx.buf_uops[buf.buffer])
      ctx.realizes[ubuf] = ubuf
  return list(output_groups.values()), lazybufs_to_realize, assign_targets
