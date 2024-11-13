from collections import defaultdict, deque
from typing import Set, Tuple, List, Dict, DefaultDict
from tinygrad.device import Buffer
from tinygrad.ops import UOp, Ops
from tinygrad.helpers import FUSE_CONV_BW, FUSE_ARANGE, dedup, merge_dicts
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.engine.lazy import LazyBuffer

def _recursive_group(tr:LazyBuffer, st:ShapeTracker, r:LazyBuffer, children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]],
                     realizes:Dict[LazyBuffer, None], reduce_for_op:Dict[LazyBuffer, UOp], group:Dict[LazyBuffer, None],
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
    if tr_next.op is Ops.REDUCE_AXIS: return group.setdefault(r)
    # can only fuse contiguous
    if len(st_childs:=dedup(s.st for s in tr_next.srcs if s.base == tr)) > 1: return group.setdefault(r)
    _recursive_group(tr_next, st+st_childs[0], r, children, realizes, reduce_for_op, group, cache)

def _get_isolated_children(r:LazyBuffer, reduce_for_op:Dict[LazyBuffer, UOp], children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]],\
    realizes:Dict[LazyBuffer, None], group:Dict[LazyBuffer, None]) -> Dict[LazyBuffer, None]:
  rc_parents, cache = deque(group), set()
  while rc_parents:
    if (p:=rc_parents.pop()) in cache: continue
    cache.add(p)
    # max one reduceop per kernel
    if p.op is Ops.REDUCE_AXIS: return {}
    rc_parents.extend(x.base for x in p.srcs if x.base.realized is None and x.base is not r)
  # search descendants of the reduceop that can cleanly group
  descendants: Dict[LazyBuffer, None] = {}
  for tr in group: _recursive_group(tr, tr.st, tr, children, realizes, reduce_for_op, descendants, cache={})
  return merge_dicts([group, {} if any(tr in group for tr in descendants) else descendants])

def get_realizes(children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]], allbufs:Dict[LazyBuffer, None], double_reduces:Dict[LazyBuffer, None],
                 ubuf_realizes:Dict[UOp, UOp], assigns:Set[UOp], buf_uops:Dict[Buffer, UOp]) -> List[List[UOp]]:
  """search the graph for all the LazyBuffers that need to realize"""
  # get all the realizes from big graph
  realizes: Dict[LazyBuffer, None] = {}
  for r in allbufs:
    if (ubuf:=buf_uops[r.buffer]) in ubuf_realizes: realizes[r] = None
  # find all reduces, and pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce (or a contig child)
  reduce_for_op: Dict[LazyBuffer, UOp] = {}
  reduce_of_const: List[UOp] = []
  for r in allbufs:
    if r in realizes or r.op is not Ops.REDUCE_AXIS: continue
    group: Dict[LazyBuffer, None] = {}
    _recursive_group(r, r.st, r, children, realizes, reduce_for_op, group, cache={})
    # max one reduceop per kernel
    can_chase = all(tr not in reduce_for_op for tr in group)
    # TODO: forced_realize exists because the scheduler is incapable of checking for self-contained DAGs
    forced_realize = r in group
    if not forced_realize and len(group) > 1:
      group = _get_isolated_children(r, reduce_for_op, children, realizes, group)
    # can only fuse assign if no other assign_target is used in the kernel
    if not forced_realize and any(x.op is Ops.ASSIGN for x in group):
      parents = deque((r, *group))
      while parents and not forced_realize:
        if (p:=parents.pop().base).is_realized() or p in realizes:
          if p.is_realized() and buf_uops[(b:=p.buffer)] in assigns and not any(x.buffer is b for x in group): forced_realize, can_chase = True, False
          continue
        parents.extend(p.srcs)
    if forced_realize or not group:
      tr = r
      if can_chase:
        # can chase this down to contiguous children
        st = tr.st
        while len(children[tr]) == 1:
          tr_next = next(iter(children[tr]))
          st_childs = dedup(s.st for s in tr_next.srcs if s.base is tr)
          if len(st_childs) > 1: break
          if st.size != st_childs[0].size: break
          st = st + st_childs[0]
          if not st.contiguous or tr_next.op is Ops.REDUCE_AXIS: break
          tr = tr_next
        # don't cast to higher size before store (tr cannot be realized if forced_realize)
        if tr.op is Ops.CAST and tr.dtype.base.itemsize > tr.srcs[0].dtype.base.itemsize:
          tr = tr.srcs[0].base
      group = {tr: None}
      realizes[tr] = None
    rbuf = buf_uops[r.buffer]
    reduce_for_op.update((tr, rbuf) for tr in group)
    if FUSE_ARANGE and r.arg[0] is Ops.ADD and r.srcs[0].base.op is Ops.CONST: reduce_of_const.append(rbuf)

  # fuse double reduces with no other child
  if FUSE_CONV_BW:
    for reduceop in double_reduces:
      top_reduce = reduceop.base.srcs[0].base
      if len(children[top_reduce]) == 1:
        del realizes[top_reduce]
        if (ubuf:=buf_uops[top_reduce.buffer]) in ubuf_realizes: del ubuf_realizes[ubuf]

  for rbuf in reduce_of_const:
    group = {tr:None for tr,rop in reduce_for_op.items() if rop is rbuf}
    if any(tr.forced_realize for tr in group): continue
    kernel_children = {c for tr in group for c in children[tr] if c.op not in {Ops.COPY, Ops.BUFFER_VIEW}}
    if len(kernel_children) == 0: continue
    for tr in group:
      del realizes[tr]
      if (ubuf:=buf_uops[tr.buffer]) in ubuf_realizes: del ubuf_realizes[ubuf]

  output_groups: DefaultDict[UOp, List[UOp]] = defaultdict(list)
  for buf in realizes:
    output_groups[reduce_for_op.get(buf, ubuf:=buf_uops[buf.buffer])].append(ubuf)
    ubuf_realizes[ubuf] = ubuf
  return list(output_groups.values())
