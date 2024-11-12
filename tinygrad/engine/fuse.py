from collections import defaultdict, deque
from typing import Set, Tuple, List, Dict, DefaultDict
from tinygrad.device import Buffer
from tinygrad.ops import MetaOps, Ops, UOp
from tinygrad.helpers import FUSE_CONV_BW, FUSE_ARANGE, dedup, merge_dicts, unwrap
from tinygrad.shape.shapetracker import ShapeTracker

def _recursive_group(tr:UOp, st:ShapeTracker, r:UOp, children:DefaultDict[UOp, Dict[UOp, None]], realizes:Dict[UOp, UOp], allbufs:Dict[UOp, UOp],
                     reduce_for_op:Dict[UOp, UOp], group:Dict[UOp, None], cache:Dict[Tuple[UOp, ShapeTracker], None]) -> None:
  """recursively search the UOp for groupable children, realize the UOp if a child can't group"""
  if (tr, st) in cache: return
  cache.setdefault((tr, st))
  reduceop_size = unwrap(allbufs[r].st).size
  if tr in realizes and tr is not r:
    # can only fuse contiguous
    # max one reduceop per kernel
    if not st.contiguous or st.size != reduceop_size or tr in reduce_for_op: group.setdefault(r)
    return group.setdefault(tr)
  for tr_next in children[tr]:
    # max one reduceop per kernel
    if (tr_next_uop:=allbufs[tr_next]).op is Ops.REDUCE_AXIS: return group.setdefault(r)
    # can only fuse contiguous
    if len(st_childs:=dedup(s for s in tr_next_uop.src if s.base.op is Ops.LOAD and s.base.src[0] == tr)) > 1: return group.setdefault(r)
    _recursive_group(tr_next, st+unwrap(st_childs[0].st), r, children, realizes, allbufs, reduce_for_op, group, cache)

def _get_isolated_children(rbuf:UOp, reduce_for_op:Dict[UOp, UOp], children:DefaultDict[UOp, Dict[UOp, None]], realizes:Dict[UOp, UOp],
                           allbufs:Dict[UOp, UOp], group:Dict[UOp, None]) -> Dict[UOp, None]:
  rc_parents, cache = deque(group), set()
  while rc_parents:
    if (p:=rc_parents.pop()) in cache: continue
    cache.add(p)
    # max one reduceop per kernel
    if (p_uop:=allbufs[p]).op is Ops.REDUCE_AXIS: return {}
    rc_parents.extend(next_p for x in p_uop.src if x.base.op is Ops.LOAD and (next_p:=x.base.src[0]) is not rbuf)
  # search descendants of the reduceop that can cleanly group
  descendants: Dict[UOp, None] = {}
  for tr in group: _recursive_group(tr, unwrap(allbufs[tr].st), tr, children, realizes, allbufs, reduce_for_op, descendants, cache={})
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
    _recursive_group(rbuf, unwrap(r.st), rbuf, children, realizes, allbufs, reduce_for_op, group, cache={})
    # max one reduceop per kernel
    can_chase = all(tr not in reduce_for_op for tr in group)
    # TODO: forced_realize exists because the scheduler is incapable of checking for self-contained DAGs
    forced_realize = rbuf in group
    if not forced_realize and len(group) > 1:
      group = _get_isolated_children(rbuf, reduce_for_op, children, realizes, allbufs, group)
    # can only fuse assign if no other assign_target is used in the kernel
    if not forced_realize and any(x.op is MetaOps.ASSIGN for x in group):
      parents = deque((r, *group))
      while parents and not forced_realize:
        if (p:=parents.pop().base).is_realized() or p in realizes:
          if p.is_realized() and buf_uops[(b:=p.buffer)] in assigns and not any(x.buffer is b for x in group): forced_realize, can_chase = True, False
          continue
        parents.extend(p.srcs)
    if forced_realize or not group:
      tr = rbuf
      if can_chase:
        # can chase this down to contiguous children
        st = unwrap(r.st)
        while len(children[tr]) == 1:
          tr_next_uop = allbufs[tr_next:=next(iter(children[tr]))]
          st_childs = dedup(s for s in tr_next_uop.src if s.base.op is Ops.LOAD and s.base.src[0] == tr)
          if len(st_childs) > 1: break
          if st.size != unwrap(st_childs[0].st).size: break
          st = st + unwrap(st_childs[0].st)
          if not st.contiguous or tr_next_uop.op is Ops.REDUCE_AXIS: break
          tr = tr_next
        # don't cast to higher size before store (tr cannot be realized if forced_realize)
        if (tr_uop:=allbufs[tr]).op is Ops.CAST and tr_uop.dtype.itemsize > tr_uop.src[0].dtype.itemsize:
          tr = tr_uop.src[0].base.src[0]
      group = {tr: None}
      realizes[tr] = tr
    reduce_for_op.update((tr, rbuf) for tr in group)
    if FUSE_ARANGE and r.arg[0] is Ops.ADD and r.src[0].base.op not in {Ops.LOAD, Ops.PRELOAD}: reduce_of_const.append(rbuf)

  # fuse double reduces with no other child
  if FUSE_CONV_BW:
    for rbuf in double_reduces:
      top_reduce = allbufs[rbuf].base.src[0].base.src[0]
      if len(children[top_reduce]) == 1: del realizes[top_reduce]

  for rbuf in reduce_of_const:
    group = {tr:None for tr,rop in reduce_for_op.items() if rop is rbuf}
    kernel_children = {c for tr in group for c in children[tr] if allbufs[c].op not in {MetaOps.COPY, MetaOps.BUFFER_VIEW}}
    if len(kernel_children) == 0: continue
    for tr in group: del realizes[tr]

  output_groups: DefaultDict[UOp, List[UOp]] = defaultdict(list)
  for ubuf in realizes: output_groups[reduce_for_op.get(ubuf, ubuf)].append(ubuf)
  return list(output_groups.values())
