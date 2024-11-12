from collections import defaultdict, deque
from typing import Set, Tuple, List, Dict, DefaultDict
from tinygrad.device import Buffer
from tinygrad.ops import GroupOp, MetaOps, Ops, ReduceOps, UOp, UnaryOps
from tinygrad.helpers import FUSE_CONV_BW, FUSE_ARANGE, dedup, merge_dicts, unwrap
from tinygrad.shape.shapetracker import ShapeTracker

def _recursive_group(tr:UOp, st:ShapeTracker, r:UOp, children:DefaultDict[UOp, Dict[UOp, None]], realizes:Dict[UOp, UOp],
                     reduce_for_op:Dict[UOp, UOp], allbufs:Dict[UOp, UOp], group:Dict[UOp, None], cache:Dict[Tuple[UOp, ShapeTracker], None]) -> None:
  """recursively search the UOp for groupable children, realize the UOp if a child can't group"""
  if (tr, st) in cache: return
  cache.setdefault((tr, st))
  tr_uop = allbufs[tr]
  if tr in realizes and tr_uop is not r:
    # can only fuse contiguous
    # max one reduceop per kernel
    if not st.contiguous or st.size != (unwrap(tr_uop.st)).size or tr in reduce_for_op: group.setdefault(r)
    return group.setdefault(tr)
  for tr_next in children[tr]:
    if (tr_next_uop:=allbufs[tr_next]).op is Ops.CONTIGUOUS and tr_next_uop.src[0].op is not Ops.LOAD: tr_next_uop = tr_next_uop.src[0]
    # max one reduceop per kernel
    if tr_next_uop.op is Ops.REDUCE_AXIS: return group.setdefault(r)
    # can only fuse contiguous
    if len(st_childs:=dedup(s for s in tr_next_uop.src if s.base.src[0] == tr)) > 1: return group.setdefault(r)
    _recursive_group(tr_next, st+unwrap(st_childs[0].st), r, children, realizes, reduce_for_op, allbufs, group, cache)

def _get_isolated_children(r:UOp, reduce_for_op:Dict[UOp, UOp], children:DefaultDict[UOp, Dict[UOp, None]], realizes:Dict[UOp, UOp],
                           allbufs:Dict[UOp, UOp], group:Dict[UOp, None]) -> Dict[UOp, None]:
  rc_parents, cache = deque(group), set()
  while rc_parents:
    if (p:=rc_parents.pop()) in cache: continue
    cache.add(p)
    # max one reduceop per kernel
    if p.op in GroupOp.Reduce: return {}
    rc_parents.extend(x.base for x in p.srcs if x.base.realized is None and x.base is not r)
  # search descendants of the reduceop that can cleanly group
  descendants: Dict[UOp, None] = {}
  for tr in group: _recursive_group(tr, tr.st, tr, children, realizes, reduce_for_op, allbufs, descendants, cache={})
  return merge_dicts([group, {} if any(tr in group for tr in descendants) else descendants])

def get_realizes(children:DefaultDict[UOp, Dict[UOp, None]], allbufs:Dict[UOp, UOp], double_reduces:Dict[UOp, None],
                 realizes:Dict[UOp, UOp], assigns:Set[UOp], buf_uops:Dict[Buffer, UOp]) -> List[List[UOp]]:
  """search the graph for all the LazyBuffers that need to realize"""
  # find all reduces, and pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce (or a contig child)
  reduce_for_op: Dict[UOp, UOp] = {}
  reduce_of_const: List[UOp] = []
  for rbuf,r in allbufs.items():
    if rbuf in realizes or r.op is not Ops.REDUCE_AXIS: continue
    group: Dict[UOp, None] = {}
    _recursive_group(rbuf, r.st, r, children, realizes, reduce_for_op, allbufs, group, cache={})
    # max one reduceop per kernel
    can_chase = all(tr not in reduce_for_op for tr in group)
    # TODO: forced_realize exists because the scheduler is incapable of checking for self-contained DAGs
    forced_realize = r in group
    if not forced_realize and len(group) > 1:
      group = _get_isolated_children(r, reduce_for_op, children, realizes, allbufs, group)
    # can only fuse assign if no other assign_target is used in the kernel
    if not forced_realize and any(x.op is MetaOps.ASSIGN for x in group):
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
          st_childs = dedup(s for s in tr_next.srcs if s.base is tr)
          if len(st_childs) > 1: break
          if st.size != st_childs[0].st.size: break
          st = st + st_childs[0].st
          if not st.contiguous or tr_next.op in GroupOp.Reduce: break
          tr = tr_next
        # don't cast to higher size before store (tr cannot be realized if forced_realize)
        if tr.op is UnaryOps.CAST and tr.arg.itemsize > tr.srcs[0].dtype.itemsize:
          tr = tr.srcs[0].base
      group = {tr: None}
      realizes[tr] = tr
    reduce_for_op.update((tr, rbuf) for tr in group)
    if FUSE_ARANGE and r.op is ReduceOps.SUM and r.srcs[0].base.op is MetaOps.CONST: reduce_of_const.append(rbuf)

  # fuse double reduces with no other child
  if FUSE_CONV_BW:
    for reduceop in double_reduces:
      top_reduce = reduceop.base.srcs[0].base
      if len(children[top_reduce]) == 1: del realizes[top_reduce]

  for rbuf in reduce_of_const:
    group = {tr:None for tr,rop in reduce_for_op.items() if rop is rbuf}
    if any(tr.forced_realize for tr in group): continue
    kernel_children = {c for tr in group for c in children[tr] if c.op not in {MetaOps.COPY, MetaOps.BUFFER_VIEW}}
    if len(kernel_children) == 0: continue
    for tr in group: del realizes[tr]

  output_groups: DefaultDict[UOp, List[UOp]] = defaultdict(list)
  for ubuf in realizes: output_groups[reduce_for_op.get(ubuf, ubuf)].append(ubuf)
  return list(output_groups.values())
