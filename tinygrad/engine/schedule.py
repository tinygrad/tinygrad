from collections import defaultdict, deque
from typing import List, Dict, Optional, Set, DefaultDict
from tinygrad.ops import LoadOps, ScheduleItem, BufferOps, LazyOp, ReduceOps, ConstBuffer, MemBuffer, BinaryOps, UnaryOps
from tinygrad.features.graph import log_lazybuffer
from tinygrad.helpers import GRAPH, DEBUG, prod, dedup, all_int
from tinygrad.shape.symbolic import Variable
from tinygrad.dtype import ImageDType, dtypes
from tinygrad.lazy import LazyBuffer
from tinygrad.shape.shapetracker import ShapeTracker

# recursively create a lazyop
def _recursive_lazyop(buf:LazyBuffer, inputs:List[LazyBuffer], var_vals:Dict[Variable, int], st:ShapeTracker,
                      realizes:Set[LazyBuffer], cache, first=True, assign_to:Optional[LazyBuffer]=None) -> LazyOp:
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
  if buf.realized or (buf in realizes and not first):
    unbound_st, st_var_vals = st.simplify().unbind()
    var_vals.update(st_var_vals)
    if assign_to is not None and buf is assign_to:
      if not unbound_st.contiguous:
        # we also allow masked views. if it has a single view and it's equal when you shrink a contig, it's fine
        if not (len(unbound_st.views) == 1 and unbound_st.views[0].mask is not None and
            ShapeTracker.from_shape(unbound_st.shape).shrink(unbound_st.views[0].mask) == unbound_st.shrink(unbound_st.views[0].mask)):
          raise RuntimeError(f"must be contiguous for assign {unbound_st}")
      return LazyOp(BufferOps.LOAD, (), MemBuffer(0, buf.dtype, unbound_st))
    if buf not in inputs: inputs.append(buf)
    return LazyOp(BufferOps.LOAD, (), MemBuffer(inputs.index(buf)+1, buf.dtype, unbound_st))

  # if a CONTIGUOUS or ASSIGN made it all the way here, just skip it
  if buf.op is LoadOps.CONTIGUOUS:
    assert first
    return _recursive_lazyop(buf.srcs[0], inputs, var_vals, st, realizes, cache, False)
  if buf.op is LoadOps.ASSIGN:
    assert first
    assert buf.srcs[1].base is buf.srcs[1], "assign must be to base"
    assert buf.srcs[1].realized is not None, f"assign must be already realized to schedule {buf.srcs[1]}"
    return _recursive_lazyop(buf.srcs[0], inputs, var_vals, st, realizes, cache, False, assign_to=buf.srcs[1])

  # if it's a reduce, we have to change the shapetracker
  if buf.op in ReduceOps:
    assert st.contiguous, "ReduceOps late fusion must be contiguous"
    st = ShapeTracker.from_shape(buf.srcs[0].shape)

  # otherwise we fuse it like normal
  cache[(buf, st)] = ret = \
    LazyOp(buf.op, tuple(_recursive_lazyop(x, inputs, var_vals, st, realizes, cache, False, assign_to) for x in buf.srcs), buf.arg)
  return ret

def _schedule_one(out:LazyBuffer, realizes:Set[LazyBuffer], reduce_for_op: Dict[LazyBuffer, LazyBuffer]) -> ScheduleItem:
  inputs: List[LazyBuffer] = []
  var_vals: Dict[Variable, int] = out.st.var_vals.copy()
  if out.op in {LoadOps.CUSTOM, LoadOps.SYNC, LoadOps.WAIT, LoadOps.COPY, LoadOps.EMPTY}:
    op, inputs = LazyOp(out.op, (), out.arg), list(out.srcs)
  else:
    output_st = ShapeTracker.from_shape(reduce_for_op[out].shape if out in reduce_for_op else out.shape)
    op = _recursive_lazyop(out, inputs, var_vals, output_st, realizes, cache={})
    op = LazyOp(BufferOps.STORE, (op, ), MemBuffer(0, out.dtype, output_st.simplify().unbind()[0]))
  return ScheduleItem((op,), (out,), tuple(inputs), var_vals)

# recursively search the entire graph for all LazyBuffers, insert realizes after expands
def _recurse_lb(buf:LazyBuffer, realizes:Set[LazyBuffer], allbufs:Dict[LazyBuffer, None],
                simple_pads:Set[LazyBuffer], children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]], scheduled=False):
  if buf in allbufs or buf.base.realized: return
  if GRAPH: log_lazybuffer(buf, scheduled)
  if isinstance(buf.dtype, ImageDType) and (prod(buf.shape) != prod(buf.dtype.shape) or
                                            not any(buf.shape[x]%4 == 0 for x in buf.st.unit_stride_axes())):
    if DEBUG >= 3: print(f"forcing image {buf.dtype} with shape {buf.shape} to float32")
    buf.dtype = dtypes.float32  # NOTE: this is what makes the dtype above not match
  if buf.base != buf:
    # realize all places where the buffer is expanded
    if prod(buf.base.st.shape) < prod(buf.st.shape):
      if len(buf.st.views) == 1 and buf.st.views[-1].mask and all_int(buf.base.st.shape) and \
          prod(buf.base.st.shape) >= prod([y-x for x,y in buf.st.views[-1].mask]):
        simple_pads.add(buf.base)
      else:
        realizes.add(buf.base)
    return _recurse_lb(buf.base, realizes, allbufs, simple_pads, children)
  if buf.forced_realize: realizes.add(buf)
  allbufs[buf] = None
  if buf.op in LoadOps: realizes.add(buf.base)
  if buf.op == LoadOps.COPY:
    assert buf.srcs[0].st.contiguous and buf.srcs[0].size == buf.srcs[0].base.size, "can only copy contig"
    realizes.add(buf.srcs[0].base)
  for x in buf.srcs:
    children[x.base][buf] = None
    _recurse_lb(x, realizes, allbufs, simple_pads, children)

UNSAFE_PAD_OPS = {BinaryOps.DIV, BinaryOps.CMPLT, BinaryOps.CMPEQ, UnaryOps.LOG2, UnaryOps.EXP2}
def _is_padding_okay(buf:LazyBuffer, realizes:Set[LazyBuffer]) -> bool:
  if buf in realizes or buf.realized: return True
  # NOTE: this broke to_image_idx and coder with JIT
  if buf.op in UNSAFE_PAD_OPS: return False
  return all(_is_padding_okay(x.base, realizes) for x in buf.srcs)

def create_schedule(outs:List[LazyBuffer], seen:Optional[Set[LazyBuffer]]=None) -> List[ScheduleItem]:
  if seen is None: seen = set()

  # start by just realizing the buffers passed in
  realizes: Set[LazyBuffer] = set([x.base for x in outs if not x.base.realized])
  allbufs: Dict[LazyBuffer, None] = {}
  simple_pads: Set[LazyBuffer] = set()
  children: DefaultDict[LazyBuffer, Dict[LazyBuffer, None]] = defaultdict(dict)
  for out in outs: _recurse_lb(out.base, realizes, allbufs, simple_pads, children, scheduled=True)

  # check if we have to realize pads
  for p in simple_pads:
    if not _is_padding_okay(p, realizes):
      realizes.add(p)

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
          # can only have one output buffer
          # can only reduce contiguous
          # max one reduceop per kernel
          if len(realized_children) > 1 or not st.contiguous or st.size != r.st.size or (tr in reduce_for_op and reduce_for_op[tr] != r):
            can_chase = tr not in reduce_for_op or reduce_for_op[tr] == r
            forced_realize = True
            break
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
        reduce_for_op[tr] = r
      realizes.add(tr)
    else:
      assert len(realized_children) == 1
      reduce_for_op[next(iter(realized_children.keys()))] = r

  # preschedule all buffers in realizes
  prescheduled = {x:_schedule_one(x, realizes, reduce_for_op) for x in realizes if x not in seen and x.realized is None and x.op is not LoadOps.CONST}
  assign_targets = {x.srcs[1]:x for x in realizes if x.op is LoadOps.ASSIGN and x not in seen and x.realized is None}

  # breadth first ordering
  graph: DefaultDict[LazyBuffer,List[LazyBuffer]] = defaultdict(list)
  in_degree: DefaultDict[LazyBuffer,int] = defaultdict(int)
  for out, si in prescheduled.items():
    for x in si.inputs:
      graph[x].append(out)
      if x in assign_targets:
        graph[out].append(assign_targets[x])
        in_degree[assign_targets[x]] += 1
      if x in prescheduled: in_degree[out] += 1

  queue = deque(out for out in prescheduled if in_degree[out] == 0)
  schedule: List[ScheduleItem] = []
  while queue:
    buf = queue.popleft()
    seen.add(buf)
    schedule.append(prescheduled[buf])
    for x in graph[buf]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)

  # confirm everything was scheduled
  assert len(prescheduled) == len(schedule), f"prescheduled {len(prescheduled)} but only scheduled {len(schedule)}"
  return schedule
