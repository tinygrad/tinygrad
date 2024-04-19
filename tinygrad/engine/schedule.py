import sys, pickle, atexit
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Set, DefaultDict
from tinygrad.ops import LoadOps, ScheduleItem, BufferOps, LazyOp, ReduceOps, ConstBuffer, MemBuffer, UNSAFE_PAD_OPS, UnaryOps
from tinygrad.features.graph import log_lazybuffer, realized_lazybuffer
from tinygrad.helpers import GRAPH, DEBUG, MULTIOUTPUT, GlobalCounters, prod, dedup, all_int, merge_dicts, getenv
from tinygrad.shape.symbolic import Variable
from tinygrad.dtype import ImageDType, dtypes
from tinygrad.lazy import LazyBuffer
from tinygrad.shape.shapetracker import ShapeTracker

# creation can recurse a lot
sys.setrecursionlimit(10000)

# optionally log the ops to disk
logops = open(getenv("LOGOPS", ""), "a") if getenv("LOGOPS", "") else None

# TODO: it's unfortunate this needs to exist, but because of ASSIGN, we have to retain the LazyBuffer structure until post toposort
@dataclass(frozen=True)
class _LBScheduleItem:
  ast: Tuple[LazyOp, ...]
  outputs: Tuple[LazyBuffer, ...]
  inputs: Tuple[LazyBuffer, ...]
  var_vals: Dict[Variable, int]

# recursively create a lazyop
def _recursive_lazyop(buf:LazyBuffer, inputs:List[LazyBuffer], outbufs:Tuple[LazyBuffer, ...], var_vals:Dict[Variable, int], st:ShapeTracker,
                      realizes:Dict[LazyBuffer, None], cache, assign_to:Optional[LazyBuffer]=None, assign_idx:Optional[int]=None) -> LazyOp:
  if (buf, st) in cache: return cache[(buf, st)]
  if buf != buf.base:
    st = buf.st + st
    buf = buf.base
  # all buffers here are base now
  assert buf.op is not None

  # consts are always fused and generated
  if buf.op is LoadOps.CONST:
    unbound_st, st_var_vals = st.simplify().unbind()
    var_vals.update(st_var_vals)
    return LazyOp(BufferOps.CONST, (), ConstBuffer(buf.arg, buf.dtype, unbound_st))

  # if we aren't fusing it, it's a load and we add it to the inputs
  if buf.realized or (buf in realizes and buf not in outbufs):
    unbound_st, st_var_vals = st.simplify().unbind()
    var_vals.update(st_var_vals)
    if assign_to is not None and buf is assign_to:
      assert assign_idx is not None
      if not unbound_st.contiguous:
        # we also allow masked views. if it has a single view and it's equal when you shrink a contig, it's fine
        if not (len(unbound_st.views) == 1 and unbound_st.views[0].mask is not None and
            ShapeTracker.from_shape(unbound_st.shape).shrink(unbound_st.views[0].mask) == unbound_st.shrink(unbound_st.views[0].mask)):
          raise RuntimeError(f"must be contiguous for assign {unbound_st}")
      return LazyOp(BufferOps.LOAD, (), MemBuffer(assign_idx, buf.dtype, unbound_st))
    if buf not in inputs: inputs.append(buf)
    return LazyOp(BufferOps.LOAD, (), MemBuffer(len(outbufs)+inputs.index(buf), buf.dtype, unbound_st))

  # if a CONTIGUOUS or ASSIGN made it all the way here, just skip it
  if buf.op is LoadOps.CONTIGUOUS:
    assert buf in outbufs
    return _recursive_lazyop(buf.srcs[0], inputs, outbufs, var_vals, st, realizes, cache)
  if buf.op is LoadOps.ASSIGN:
    assert buf in outbufs
    assert buf.srcs[1].base is buf.srcs[1], "assign must be to base"
    assert buf.srcs[1].realized is not None, f"assign must be already realized to schedule {buf.srcs[1]}"
    return _recursive_lazyop(buf.srcs[0], inputs, outbufs, var_vals, st, realizes, cache, assign_to=buf.srcs[1], assign_idx=outbufs.index(buf))

  # if it's a reduce, we have to change the shapetracker
  if buf.op in ReduceOps:
    assert st.contiguous, "ReduceOps late fusion must be contiguous"
    st = ShapeTracker.from_shape(buf.srcs[0].shape)

  # otherwise we fuse it like normal
  cache[(buf, st)] = ret = \
    LazyOp(buf.op, tuple(_recursive_lazyop(x, inputs, outbufs, var_vals, st, realizes, cache, assign_to, assign_idx) for x in buf.srcs), buf.arg)
  return ret

def _schedule_group(outs:Tuple[LazyBuffer, ...], realizes:Dict[LazyBuffer, None], reduce_for_op: Dict[LazyBuffer, LazyBuffer]) -> _LBScheduleItem:
  inputs: List[LazyBuffer] = []
  ast: List[LazyOp] = []
  var_vals: Dict[Variable, int] = merge_dicts([out.st.var_vals.copy() for out in outs])
  if outs[0].op in {LoadOps.CUSTOM, LoadOps.COPY, LoadOps.EMPTY}:
    ast, inputs = [LazyOp(outs[0].op, (), outs[0].arg)], list(outs[0].srcs)
  else:
    for i, out in enumerate(outs):
      output_st = ShapeTracker.from_shape(reduce_for_op[out].shape if out in reduce_for_op else out.shape)
      output_view = out.arg[0] if out.op is LoadOps.ASSIGN and out.arg else output_st
      op = _recursive_lazyop(out, inputs, outs, var_vals, output_st, realizes, cache={})
      output_view, vv = output_view.simplify().unbind()
      if vv: var_vals.update(vv)
      ast.append(LazyOp(BufferOps.STORE, (op, ), MemBuffer(i, out.dtype, output_view)))
  return _LBScheduleItem(tuple(ast), outs, tuple(inputs), var_vals)

# recursively search the entire graph for all LazyBuffers, insert realizes after expands
def _recurse_lb(buf:LazyBuffer, realizes:Dict[LazyBuffer, None], allbufs:Dict[LazyBuffer, None],
                simple_pads:Set[LazyBuffer], children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]], scheduled=False):
  if buf in allbufs or buf.base.realized: return
  if GRAPH: log_lazybuffer(buf, scheduled)
  if isinstance(buf.dtype, ImageDType) and (prod(buf.shape) != prod(buf.dtype.shape) or
                                            not any(buf.shape[x]%4 == 0 for x in buf.st.unit_stride_axes())):
    if DEBUG >= 3: print(f"forcing image {buf.dtype} with shape {buf.shape} to float32")
    buf.dtype = dtypes.float32  # NOTE: this is what makes the dtype above not match
    # hack the underlying buffer too
    if buf.base is buf:
      assert not hasattr(buf.buffer, '_buf'), "can't fixup allocated buffer"
      buf.buffer.dtype = dtypes.float32
      buf.buffer.options = None
  if buf.base != buf:
    # realize all places where the buffer is expanded
    if prod(buf.base.st.shape) < prod(buf.st.shape):
      if len(buf.st.views) == 1 and buf.st.views[-1].mask and all_int(buf.base.st.shape) and \
          prod(buf.base.st.shape) >= prod([y-x for x,y in buf.st.views[-1].mask]):
        simple_pads.add(buf.base)
      else:
        realizes[buf.base] = None
    return _recurse_lb(buf.base, realizes, allbufs, simple_pads, children)
  if buf.forced_realize: realizes[buf] = None
  allbufs[buf] = None
  if buf.op in LoadOps: realizes[buf.base] = None
  if buf.op is LoadOps.COPY:
    assert buf.srcs[0].st.contiguous and buf.srcs[0].size == buf.srcs[0].base.size, "can only copy contig"
    realizes[buf.srcs[0].base] = None
  for x in buf.srcs:
    children[x.base][buf] = None
    _recurse_lb(x, realizes, allbufs, simple_pads, children)

def _is_padding_okay(buf:LazyBuffer, realizes:Dict[LazyBuffer, None]) -> bool:
  if buf in realizes or buf.realized: return True
  # NOTE: this broke to_image_idx and coder with JIT
  if buf.op in UNSAFE_PAD_OPS: return False
  return all(_is_padding_okay(x.base, realizes) for x in buf.srcs)

def _graph_schedule(outs:List[LazyBuffer], seen:Set[LazyBuffer]) -> Tuple[DefaultDict[LazyBuffer, List[LazyBuffer]], DefaultDict[LazyBuffer, int],
                                                                    Dict[LazyBuffer, _LBScheduleItem]]:
  # start by just realizing the buffers passed in
  realizes: Dict[LazyBuffer, None] = {x.base: None for x in outs if not x.base.realized}
  allbufs: Dict[LazyBuffer, None] = {}
  simple_pads: Set[LazyBuffer] = set()
  children: DefaultDict[LazyBuffer, Dict[LazyBuffer, None]] = defaultdict(dict)
  for out in outs: _recurse_lb(out.base, realizes, allbufs, simple_pads, children, scheduled=True)

  # check if we have to realize pads
  for p in simple_pads:
    if not _is_padding_okay(p, realizes):
      realizes[p] = None

  # find all reduces, and pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce (or a contig child)
  reduce_for_op: Dict[LazyBuffer, LazyBuffer] = {}
  for r in allbufs.keys():
    if r != r.base or r.op not in ReduceOps or r in realizes: continue

    # follow the reduce down
    child_set: Dict[LazyBuffer, ShapeTracker] = {r: r.st}
    realized_children: Dict[LazyBuffer, ShapeTracker] = {}
    forced_realize = False
    can_chase = True
    while not forced_realize and len(child_set):
      next_child_set = {}
      for tr,st in child_set.items():
        if tr in realizes:
          realized_children[tr] = st
          # can only reduce contiguous
          # max one reduceop per kernel
          if not st.contiguous or st.size != r.st.size or (tr in reduce_for_op and reduce_for_op[tr] != r):
            can_chase = tr not in reduce_for_op or reduce_for_op[tr] == r
            forced_realize = True
            break
          if len(realized_children) > 1:
            for rc in realized_children:
              rc_parents = deque(x.base for x in rc.srcs)
              while rc_parents:
                if (p:=rc_parents.pop()).realized or p.op is LoadOps.CONST: continue
                if p is r: continue
                # max one reduceop per kernel
                if p.op in ReduceOps:
                  can_chase = tr not in reduce_for_op or reduce_for_op[tr] == r
                  forced_realize = True
                  break
                for x in p.srcs: rc_parents.append(x.base)
          continue
        for tr_next in children[tr].keys():
          if not tr_next.realized:
            # max one reduceop per kernel
            if tr_next.op in ReduceOps:
              forced_realize = True
              break
            st_childs = dedup([s for s in tr_next.srcs if s.base == tr])
            if len(st_childs) > 1:
              forced_realize = True
              break
            next_child_set[tr_next] = st + st_childs[0].st
      child_set = next_child_set
    if forced_realize:
      tr = r
      if can_chase:
        # can chase this down to contiguous children
        st = tr.st
        while len(children[tr]) == 1:
          tr_next = next(iter(children[tr].keys()))
          st_childs = dedup([s for s in tr_next.srcs if s.base == tr])
          if len(st_childs) > 1: break
          if st.size != st_childs[0].st.size: break
          st = st + st_childs[0].st
          if not st.contiguous or tr_next.op in ReduceOps: break
          tr = tr_next
        # don't cast to higher size before store (tr cannot be realized if forced_realize)
        if tr.op is UnaryOps.CAST and tr.arg[0].itemsize > tr.srcs[0].dtype.itemsize:
          tr = tr.srcs[0].base
        reduce_for_op[tr] = r
      realizes[tr] = None
    else: reduce_for_op.update((tr, r) for tr in realized_children)

  output_groups: DefaultDict[Tuple, List[LazyBuffer]] = defaultdict(list)
  for r in realizes:
    if r.realized is not None or r.op is LoadOps.CONST or r in seen: continue
    output_groups[(reduce_for_op[r], ) if r in reduce_for_op and MULTIOUTPUT else (r, )].append(r)

  # preschedule all buffers in realizes
  prescheduled = {group[0]:_schedule_group(tuple(group), realizes, reduce_for_op) for group in output_groups.values()}
  schedule_targets = {out:ps for ps in prescheduled.values() for out in ps.outputs}
  assign_targets = {x.srcs[1]:x for x in realizes if x.op is LoadOps.ASSIGN and x not in seen and x.realized is None}

  # breadth first ordering
  graph: DefaultDict[LazyBuffer, List[LazyBuffer]] = defaultdict(list)
  in_degree: DefaultDict[LazyBuffer, int] = defaultdict(int)
  for key, lsi in prescheduled.items():
    if key not in in_degree: in_degree[key] = 0
    # realize outputs after all parents are realized
    scheduled_parents = set(schedule_targets[x].outputs[0] for x in lsi.inputs if x in schedule_targets)
    for x in scheduled_parents:
      graph[x].append(key)
      in_degree[key] += 1
    # realize outputs before a parent is assigned to
    parents_assigns = set(schedule_targets[assign_targets[x]].outputs[0] for x in lsi.inputs if x in assign_targets)
    for assign in parents_assigns:
      graph[key].append(assign)
      in_degree[assign] += 1

  return graph, in_degree, prescheduled

SCHEDULES: List[ScheduleItem] = []
def create_schedule_with_vars(outs:List[LazyBuffer], seen:Optional[Set[LazyBuffer]]=None) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
  if seen is None: seen = set()
  graph, in_degree, prescheduled = _graph_schedule(outs, seen)
  queue = deque(si for key, si in prescheduled.items() if in_degree[key] == 0)
  schedule: List[ScheduleItem] = []
  var_vals: Dict[Variable, int] = {}
  kernel_number = GlobalCounters.kernel_count
  while queue:
    ps = queue.popleft()
    for buf in ps.outputs: seen.add(buf)
    if GRAPH:
      kernel_number += 1
      for out in ps.outputs: realized_lazybuffer(out, kernel_number)
    var_vals = merge_dicts([var_vals, ps.var_vals])
    for out in ps.outputs: del out.srcs  # can only schedule once
    schedule.append(si:=ScheduleItem(ps.ast, tuple(x.buffer for x in ps.outputs if x.size != 0), tuple(x.buffer for x in ps.inputs if x.size != 0)))
    if logops and si.ast[0].op not in LoadOps and not any(i.device.startswith("DISK:") for i in si.inputs): logops.write(str(si.ast)+"\n")
    for x in graph[ps.outputs[0]]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(prescheduled[x])

  # confirm everything was scheduled correctly
  if not all(degree == 0 for degree in in_degree.values()) or len(prescheduled) != len(schedule):
    raise RuntimeError(f"cycle detected in graph, prescheduled {len(prescheduled)} but only scheduled {len(schedule)}")
  if DEBUG >= 1 and len(schedule) >= 10: print(f"scheduled {len(schedule)} kernels")
  if getenv("SAVE_SCHEDULE"):
    def _save():
      print(f"saving {len(SCHEDULES)} schedule items to", fp:="schedule.pkl")
      pickle.dump(SCHEDULES, open(fp, "wb"))
    if len(SCHEDULES) == 0: atexit.register(_save)
    SCHEDULES.extend(schedule)
  return schedule, var_vals

def create_schedule(outs:List[LazyBuffer], seen:Optional[Set[LazyBuffer]]=None) -> List[ScheduleItem]:
  schedule, var_vals = create_schedule_with_vars(outs, seen)
  assert len(var_vals) == 0
  return schedule
