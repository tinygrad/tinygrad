from collections import defaultdict, deque
from typing import Set, Tuple, List, Dict, DefaultDict
from tinygrad.device import Buffer
from tinygrad.ops import UOp, Ops
from tinygrad.helpers import FUSE_CONV_BW, FUSE_ARANGE, dedup, merge_dicts, unwrap
from tinygrad.shape.shapetracker import ShapeTracker

def uval(u:UOp) -> UOp:
  assert u.op is Ops.LOAD and len(u.src) == 3 and u.src[2].op is Ops.STORE, f"must be a LOAD of STORE {u}"
  return ret.src[0] if (ret:=u.src[2].src[2]).op is Ops.CONTIGUOUS and ret.src[0].op is not Ops.LOAD else ret

def _recursive_group(tr:UOp, st:ShapeTracker, r:UOp, children:DefaultDict[UOp, Dict[UOp, None]], allbufs:Dict[UOp, UOp],
                     realizes:Dict[UOp, UOp], reduce_for_op:Dict[UOp, UOp], group:Dict[UOp, None],
                     cache:Dict[Tuple[UOp, ShapeTracker], None]) -> None:
  """recursively search the UOp for groupable children, realize the UOp if a child can't group"""
  if (tr, st) in cache: return
  cache.setdefault((tr, st))
  rsize = unwrap(allbufs[r].st).size
  if tr in realizes and tr is not r:
    # can only fuse contiguous
    # max one reduceop per kernel
    if not st.contiguous or st.size != rsize or tr in reduce_for_op: group.setdefault(r)
    return group.setdefault(tr)
  for tr_next in children[tr]:
    # max one reduceop per kernel
    if (tr_next_uop:=uval(allbufs[tr_next])).op is Ops.REDUCE_AXIS: return group.setdefault(r)
    # can only fuse contiguous
    st_childs = dedup([unwrap(x.st) for x in tr_next_uop.src if x.base.op is Ops.LOAD and x.base.buf_uop is tr])
    if len(st_childs) > 1: return group.setdefault(r)
    _recursive_group(tr_next, st+st_childs[0], r, children, allbufs, realizes, reduce_for_op, group, cache)

def _get_isolated_children(r:UOp, reduce_for_op:Dict[UOp, UOp], children:DefaultDict[UOp, Dict[UOp, None]], allbufs:Dict[UOp, UOp],
                           realizes:Dict[UOp, UOp], group:Dict[UOp, None]) -> Dict[UOp, None]:
  rc_parents, cache = deque(group), set()
  while rc_parents:
    if (p:=uval(allbufs[rc_parents.pop()])) in cache: continue
    cache.add(p)
    # max one reduceop per kernel
    if p.op is Ops.REDUCE_AXIS: return {}
    rc_parents.extend(x.base.buf_uop for x in p.src if x.base.op is Ops.LOAD and x.base.buf_uop is not r)
  # search descendants of the reduceop that can cleanly group
  descendants: Dict[UOp, None] = {}
  for tr in group: _recursive_group(tr, unwrap(allbufs[tr].st), tr, children, allbufs, realizes, reduce_for_op, descendants, cache={})
  return merge_dicts([group, {} if any(tr in group for tr in descendants) else descendants])

def get_realizes(children:DefaultDict[UOp, Dict[UOp, None]], allbufs:Dict[UOp, UOp], double_reduces:Dict[UOp, None],
                 realizes:Dict[UOp, UOp], assigns:Set[UOp], buf_uops:Dict[Buffer, UOp]) -> List[List[UOp]]:
  """search the graph for all the LazyBuffers that need to realize"""
  # find all reduces, and pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce (or a contig child)
  reduce_for_op: Dict[UOp, UOp] = {}
  reduce_of_const: List[UOp] = []
  for r, r_uop in allbufs.items():
    if r in realizes or (r_uop:=uval(r_uop)).op is not Ops.REDUCE_AXIS: continue
    group: Dict[UOp, None] = {}
    _recursive_group(r, unwrap(r_uop.st), r, children, allbufs, realizes, reduce_for_op, group, cache={})
    # max one reduceop per kernel
    can_chase = all(tr not in reduce_for_op for tr in group)
    # TODO: forced_realize exists because the scheduler is incapable of checking for self-contained DAGs
    forced_realize = r in group
    if not forced_realize and len(group) > 1:
      group = _get_isolated_children(r, reduce_for_op, children, allbufs, realizes, group)
    # can only fuse assign if no other assign_target is used in the kernel
    if not forced_realize and any(x in assigns for x in group):
      parents = deque((r, *group))
      while parents and not forced_realize:
        if (p_uop:=allbufs.get(p:=parents.pop())) is None: continue
        if (p_uop:=uval(p_uop)).op is Ops.ASSIGN and p not in group: forced_realize, can_chase = True, False
        if p in realizes: continue
        parents.extend([x.base.src[0] for x in p_uop.src if x.base.op in {Ops.LOAD, Ops.PRELOAD}])
    if forced_realize or not group:
      tr = r
      if can_chase:
        # can chase this down to contiguous children
        st = unwrap(r_uop.st)
        while len(children[tr]) == 1:
          tr_next_uop = uval(allbufs[(tr_next:=next(iter(children[tr])))])
          st_childs = dedup([unwrap(x.st) for x in tr_next_uop.src if x.base.op is Ops.LOAD and x.base.buf_uop is tr])
          if len(st_childs) > 1: break
          if st.size != st_childs[0].size: break
          st = st + st_childs[0]
          if not st.contiguous or tr_next_uop.op is Ops.REDUCE_AXIS: break
          tr = tr_next
        # don't cast to higher size before store (tr cannot be realized if forced_realize)
        if (tr_uop:=uval(allbufs[tr])).op is Ops.CAST and tr_uop.dtype.base.itemsize > tr_uop.src[0].dtype.base.itemsize:
          tr = tr_uop.src[0].base.buf_uop
      group = {tr: None}
      realizes[tr] = tr
    reduce_for_op.update((tr, r) for tr in group)
    if FUSE_ARANGE and r_uop.arg[0] is Ops.ADD and r_uop.src[0].base.op is Ops.WHERE: reduce_of_const.append(r)

  # fuse double reduces with no other child
  if FUSE_CONV_BW:
    for reduceop in double_reduces:
      top_reduce = uval(allbufs[reduceop]).src[0].base.buf_uop
      if len(children[top_reduce]) == 1: del realizes[top_reduce]

  for rbuf in reduce_of_const:
    group = {tr:None for tr,rop in reduce_for_op.items() if rop is rbuf}
    if any((tr_val:=allbufs[tr].src[2].src[2]).op is Ops.CONTIGUOUS and tr_val.src[0].op is not Ops.LOAD for tr in group): continue
    kernel_children = {c for tr in group for c in children[tr] if uval(allbufs[c]).op not in {Ops.COPY, Ops.BUFFER_VIEW}}
    if len(kernel_children) == 0: continue
    for tr in group: del realizes[tr]

  output_groups: DefaultDict[UOp, List[UOp]] = defaultdict(list)
  for ubuf in realizes: output_groups[reduce_for_op.get(ubuf, ubuf)].append(ubuf)
  return list(output_groups.values())
