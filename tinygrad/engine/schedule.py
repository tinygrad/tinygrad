import sys, pickle, atexit, importlib, contextlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Set, DefaultDict, get_args
from tinygrad.ops import MetaOps, BufferOps, LazyOp, ReduceOps, ConstBuffer, MemBuffer, UNSAFE_PAD_OPS, UnaryOps, reduce_st
from tinygrad.engine.graph import log_lazybuffer, realized_lazybuffer
from tinygrad.helpers import GRAPH, DEBUG, MULTIOUTPUT, SAVE_SCHEDULE, FUSE_CONV_BW, FUSE_ARANGE, \
                             GlobalCounters, colored, prod, dedup, all_int, merge_dicts, getenv, Metadata
from tinygrad.shape.symbolic import Variable, sint
from tinygrad.dtype import ConstType, ImageDType, dtypes
from tinygrad.lazy import LazyBuffer
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.device import Buffer
from tinygrad.shape.view import View, strides_for_shape

# creation can recurse a lot
sys.setrecursionlimit(10000)

# optionally log the ops to disk
logops = open(getenv("LOGOPS", ""), "a") if getenv("LOGOPS", "") else None

# *** ScheduleItem return type ***

@dataclass(frozen=True)
class ScheduleItem:
  ast: LazyOp
  bufs: Tuple[Buffer, ...]
  metadata: Optional[List[Metadata]] = None
  @property
  def outputs(self) -> Tuple[Buffer, ...]:
    """Read/write or write only buffers in the schedule."""
    return self.bufs[:len(self.ast.src)] if self.ast.op is MetaOps.KERNEL else self.bufs[0:1]
  @property
  def inputs(self) -> Tuple[Buffer, ...]:
    """Read only buffers in the schedule."""
    return self.bufs[len(self.ast.src):] if self.ast.op is MetaOps.KERNEL else self.bufs[1:]

@dataclass(frozen=True)
class LBScheduleItem:
  ast: LazyOp
  outputs: List[LazyBuffer]
  inputs: List[LazyBuffer]
  var_vals: Dict[Variable, int] = field(default_factory=dict)
  metadata: List[Metadata] = field(default_factory=list)
  def __hash__(self):
    """The unique identifier of a schedule item in the toposort."""
    return hash(self.outputs[0])

# *** DAG transformation: List[LazyBuffer] -> ScheduleItem ***

def _recursive_lazyop(buf:LazyBuffer, st:ShapeTracker, outputs:Tuple[LazyBuffer, ...], var_vals:Dict[Variable, int], inputs:Dict[LazyBuffer, int],
                      realizes:Dict[LazyBuffer, None], assign_targets:Dict[LazyBuffer, LazyBuffer],
                      reduce_info:Dict[Tuple[LazyBuffer, ShapeTracker], Tuple[ShapeTracker, Tuple[int, ...]]],
                      cache:Dict[Tuple[LazyBuffer, ShapeTracker], LazyOp]) -> LazyOp:
  """recursively create a lazyop"""
  if buf is not buf.base: st, buf = buf.st+st, buf.base
  if (buf, st) in cache: return cache[(buf, st)]
  assert buf.op is not None, "base must be a base itself"

  # buffer ops define ShapeTracker
  if buf.realized is not None or (buf in realizes and buf not in outputs):
    unbound_st, st_var_vals = st.simplify().unbind()
    var_vals.update(st_var_vals)
    # if it's a const, we generate it
    if buf.op is MetaOps.CONST:
      if isinstance(val:=buf.arg, Variable):
        val, var_val = val.unbind()
        var_vals[val] = var_val
      else: assert isinstance(val, get_args(ConstType)), f"cannot create ConstBuffer with value {val}"
      return LazyOp(BufferOps.CONST, (), ConstBuffer(val, buf.dtype, unbound_st))
    # otherwise, it's a load and we add it to the inputs
    if buf in assign_targets:
      # we also allow masked views. if it has a single view and it's equal when you shrink a contig, it's fine
      if unbound_st.contiguous or (len(unbound_st.views) == 1 and unbound_st.views[0].mask is not None and\
          ShapeTracker.from_shape(unbound_st.shape).shrink(unbound_st.views[0].mask) == unbound_st.shrink(unbound_st.views[0].mask)):
        return LazyOp(BufferOps.LOAD, (), MemBuffer(outputs.index(assign_targets[buf]), buf.dtype, unbound_st))
      raise RuntimeError("self operand of augmented assign must be contiguous.\nhelp: consider using .contiguous():\n"
                         +colored("   - a += a.T\n", "red")+colored("   + a += a.T.contiguous()", "green"))
    return LazyOp(BufferOps.LOAD, (), MemBuffer(len(outputs)+inputs.setdefault(buf, len(inputs)), buf.dtype, unbound_st))

  # reduce ops change ShapeTracker
  if buf.op in ReduceOps:
    rinfo = reduce_info.get((buf, st))
    rsrc = _recursive_lazyop(buf.srcs[0], st:=(rinfo[0] if rinfo else st), outputs, var_vals, inputs, realizes, assign_targets, reduce_info, cache)
    # if we are merging the reduce, skip it
    if rinfo is None:
      assert rsrc.op is buf.op, f"can't merge reduceop {buf.op} with {rsrc.op}\n{st}"
      return rsrc
    return cache.setdefault((buf, st), LazyOp(buf.op, (rsrc,), rinfo[1]))

  # elementwise ops pass shapetracker
  in_ops = tuple(_recursive_lazyop(x, st, outputs, var_vals, inputs, realizes, assign_targets, reduce_info, cache) for x in buf.srcs)
  if buf.op in {MetaOps.CONTIGUOUS, MetaOps.ASSIGN}:
    assert buf in outputs, f"{buf.op} must be writable"
    return in_ops[0]
  return cache.setdefault((buf, st), LazyOp(buf.op, in_ops, buf.arg))

def _permute_reduce(input_st:ShapeTracker, axis:Tuple[int, ...]) -> Tuple[ShapeTracker, Tuple[sint, ...]]:
  permute_axis = tuple(i for i in range(len(input_st.shape)) if i not in axis) + axis
  tmp = input_st.permute(permute_axis)
  return tmp, tmp.shape[-len(axis):]

def _recurse_reduceops(buf:LazyBuffer, st:ShapeTracker, realizes:Dict[LazyBuffer, None], outs:List[LazyBuffer],
                       reduce_info:Dict[Tuple[LazyBuffer, ShapeTracker], Tuple[ShapeTracker, Tuple[int, ...]]],
                       cache:Dict[Tuple[LazyBuffer, ShapeTracker], Optional[Tuple[LazyBuffer, ShapeTracker]]]) -> \
                         Optional[Tuple[LazyBuffer, ShapeTracker]]:
  if (buf, st) in cache: return cache[(buf, st)]
  if buf.base.realized is not None or (buf.base in realizes and buf.base not in outs): return None
  if buf is not buf.base: st, buf = buf.st+st, buf.base
  input_st = ShapeTracker.from_shape(buf.srcs[0].shape) if buf.op in ReduceOps else st
  reduce_srcs = [r for x in buf.srcs if (r:=_recurse_reduceops(x, input_st, realizes, outs, reduce_info, cache)) is not None]
  top_reduce = reduce_srcs[-1] if len(reduce_srcs) != 0 else None
  if buf.op in ReduceOps:
    axis = buf.arg
    if not st.contiguous:
      # push the movementop to the input
      tmp, rshape = _permute_reduce(input_st, axis)
      prshape = prod(rshape)
      strides = strides_for_shape(rshape)
      nv: List[View] = []
      for v in st.views:
        nv.append(View.create(v.shape+rshape, tuple(x*prshape for x in v.strides)+strides,
                              v.offset*prshape, v.mask+tuple((0,s) for s in rshape) if v.mask is not None else None))
      input_st = tmp + ShapeTracker(tuple(nv))
      # update the axis
      _, new_rshape = _permute_reduce(input_st, axis)
      axis = tuple(range(len(input_st.shape)-len(new_rshape), len(input_st.shape)))
    elif top_reduce is not None:
      top_reduce_input_st, top_reduce_axes = reduce_info[top_reduce]
      if buf.srcs[0] is not buf.srcs[0].base and buf.srcs[0].base is top_reduce[0] and buf.op is top_reduce[0].op:
        # merge this reduce with its parent
        new_st = top_reduce[1]+st
        top_reduce = (top_reduce[0], new_st.reshape(reduce_st(top_reduce_input_st, new_axis:=axis+top_reduce_axes)))
        reduce_info[top_reduce] = (top_reduce_input_st, new_axis)
        return None
      # reshape this reduceop based on the top reduce
      input_st = input_st.reshape(tuple(1 if i in top_reduce_axes else s for i,s in enumerate(top_reduce_input_st.shape)))
    st = st.reshape(reduce_st(input_st, axis))
    reduce_info[(buf, st)] = (input_st, axis)
    return (buf, st)
  return cache.setdefault((buf, st), top_reduce)

def _lower_lazybuffer(outs:List[LazyBuffer], realizes:Dict[LazyBuffer, None]) -> LBScheduleItem:
  """describe the computation for a LazyBuffer with LazyOp + inputs + var_vals"""
  if (out:=outs[0]).op is MetaOps.COPY and getenv("USE_COPY_KERNEL") and out.device.split(":")[0] == out.srcs[0].device.split(":")[0]:
    rd = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.uint8, st:=ShapeTracker.from_shape((out.arg,))))
    wr = LazyOp(BufferOps.STORE, (rd,), MemBuffer(0, dtypes.uint8, st))
    return LBScheduleItem(LazyOp(MetaOps.KERNEL, (wr,)), outs, [x.base for x in out.srcs])
  if out.op in {MetaOps.CUSTOM, MetaOps.COPY, MetaOps.EMPTY, MetaOps.VIEW}:
    return LBScheduleItem(LazyOp(MetaOps.EXT, (), (out.op, out.arg)), outs, [x.base for x in out.srcs])
  # push through all movementops between reduceops
  reduce_info: Dict[Tuple[LazyBuffer, ShapeTracker], Tuple[ShapeTracker, Tuple[int, ...]]] = {}
  seen_ops: Dict[Tuple[LazyBuffer, ShapeTracker], Optional[Tuple[LazyBuffer, ShapeTracker]]] = {}
  for out in outs: _recurse_reduceops(out, out.st, realizes, outs, reduce_info, seen_ops)
  # pad all reduceops to the max of each dimension
  shape_dims = [sorted(dedup(dims)) for dims in zip(*[input_st.shape for input_st,_ in reduce_info.values()])]
  for i,dims in enumerate(shape_dims):
    if len(dims) == 1 or (len(dims) == 2 and dims[0] == 1): continue
    for (r,view),(input_st,axis) in reduce_info.items():
      if (dim:=input_st.shape[i]) > 1 and dim != max(dims):
        input_st = input_st.pad(((0, 0),)*i+((0, max(dims)-dim),))
        reduce_info[(r, view)] = (input_st, axis)
  # create the stores
  var_vals = merge_dicts([out.st.var_vals.copy() for out in outs])
  assign_targets = {x.srcs[1]:x for x in outs if x.op is MetaOps.ASSIGN}
  cache: Dict[Tuple[LazyBuffer, ShapeTracker], LazyOp] = {}
  ast: List[LazyOp] = []
  inputs: Dict[LazyBuffer, int] = {}
  for i, out in enumerate(outs):
    output_st = ShapeTracker.from_shape(reduce_st(*deque(reduce_info.values(), 1).pop()) if reduce_info else out.shape)
    lop = _recursive_lazyop(out, output_st, tuple(outs), var_vals, inputs, realizes, assign_targets, reduce_info, cache=cache)
    if out.op is MetaOps.ASSIGN and out.arg:
      assert out.arg[0].shape == out.shape, f"ASSIGN must not override output shape {out.arg[0].shape} != {out.shape}"
      output_st = out.arg[0].reshape(output_st.shape)
    output_st, vv = output_st.simplify().unbind()
    if vv: var_vals.update(vv)
    ast.append(LazyOp(BufferOps.STORE, (lop,), MemBuffer(i, out.dtype, output_st)))
  return LBScheduleItem(LazyOp(MetaOps.KERNEL, tuple(ast)), outs, list(inputs), var_vals,
                        dedup([x[0].metadata for x in cache if x[0].metadata and x[0] not in inputs]))

# *** DAG creation: decide which LazyBuffers should realize ***

def _recurse_lb(buf:LazyBuffer, realizes:Dict[LazyBuffer, None], allbufs:Dict[LazyBuffer, None], simple_pads:Dict[LazyBuffer, None],
                children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]], assign_targets:Dict[LazyBuffer, LazyBuffer],
                double_reduces:Dict[LazyBuffer, None], scheduled=False) -> None:
  """recursively search the entire graph for all LazyBuffers, insert realizes after expands"""
  if buf in allbufs or buf.base.realized is not None: return
  if GRAPH: log_lazybuffer(buf, scheduled)
  # check if we need to realize views
  if buf is not buf.base:
    # fuse some pads
    if len(buf.st.views) == 1 and buf.st.views[-1].mask is not None and all_int(buf.base.st.shape) and \
        prod(buf.base.st.shape) >= prod([y-x for x,y in buf.st.views[-1].mask]):
      simple_pads[buf.base] = None
    # realize all expands
    elif prod(buf.base.st.shape) < prod(buf.st.shape):
      # this was causing "test_lil_model" to fail
      if buf.base.op is UnaryOps.CAST and isinstance(buf.base.srcs[0].dtype, ImageDType) and isinstance(buf.base.arg, ImageDType):
        simple_pads[buf.base] = None # don't realize image to image casts. this is part of a larger problem
      else: realizes[buf.base] = None
    # check all other pads for safe fusion
    elif any(v.mask is not None for v in buf.st.views): simple_pads[buf.base] = None
    return _recurse_lb(buf.base, realizes, allbufs, simple_pads, children, assign_targets, double_reduces)
  if buf.op in ReduceOps and buf.srcs[0].base.op is buf.op and buf.srcs[0] is not buf.srcs[0].base: double_reduces[buf] = None
  allbufs[buf] = None
  if buf.forced_realize or buf.op in MetaOps: realizes[buf] = None
  if buf.op is MetaOps.ASSIGN:
    assert buf.srcs[1].base is buf.srcs[1], f"assign must be to base {buf.srcs[1]}"
    assert buf.srcs[1].realized is not None, f"assign must be already realized to schedule {buf.srcs[1]}"
    assign_targets[buf.srcs[1]] = buf
  if buf.op is MetaOps.COPY:
    assert buf.srcs[0].st.contiguous and buf.srcs[0].size == buf.srcs[0].base.size, "can only copy contig"
    realizes[buf.srcs[0].base] = None
  if buf.op is MetaOps.VIEW: realizes[buf.srcs[0].base] = None
  for x in buf.srcs:
    if x.base.realized is None: children[x.base][buf] = None
    _recurse_lb(x, realizes, allbufs, simple_pads, children, assign_targets, double_reduces)

def _is_padding_okay(buf:LazyBuffer, realizes:Dict[LazyBuffer, None]) -> bool:
  if buf in realizes or buf.realized is not None: return True
  # NOTE: this broke to_image_idx and coder with JIT
  if buf.op in UNSAFE_PAD_OPS: return False
  return all(_is_padding_okay(x.base, realizes) for x in buf.srcs)

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
    if tr_next.op in ReduceOps: return group.setdefault(r)
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
    if p.op in ReduceOps: return {}
    rc_parents.extend(x.base for x in p.srcs if x.base.realized is None and x.base is not r)
  # search descendants of the reduceop that can cleanly group
  descendants: Dict[LazyBuffer, None] = {}
  for tr in group: _recursive_group(tr, tr.st, tr, children, realizes, reduce_for_op, descendants, cache={})
  return merge_dicts([group, {} if any(tr in group for tr in descendants) else descendants])

SCHEDULES: List[Tuple[DefaultDict[LBScheduleItem, List[LBScheduleItem]], DefaultDict[LBScheduleItem, int]]] = []
def _graph_schedule(outs:List[LazyBuffer], seen:Set[LazyBuffer]) -> \
  Tuple[DefaultDict[LBScheduleItem, List[LBScheduleItem]],  # this is the graph
        DefaultDict[LBScheduleItem, int]]:                  # this is the in-degree of the graph
  """create a graph for realizing the outputs"""
  # start by just realizing the buffers passed in
  realizes: Dict[LazyBuffer, None] = {x.base:None for x in outs if x.base.realized is None}
  allbufs: Dict[LazyBuffer, None] = {}
  simple_pads: Dict[LazyBuffer, None] = {}
  children: DefaultDict[LazyBuffer, Dict[LazyBuffer, None]] = defaultdict(dict)
  assign_targets: Dict[LazyBuffer, LazyBuffer] = {}
  double_reduces: Dict[LazyBuffer, None] = {}
  for out in outs: _recurse_lb(out.base, realizes, allbufs, simple_pads, children, assign_targets, double_reduces, scheduled=True)

  # check if we have to realize pads
  for p in simple_pads:
    if not _is_padding_okay(p, realizes):
      realizes[p] = None

  # find all reduces, and pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce (or a contig child)
  reduce_for_op: Dict[LazyBuffer, LazyBuffer] = {}
  reduce_of_const: List[LazyBuffer] = []
  for r in allbufs:
    if r.op not in ReduceOps or r in realizes: continue

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
        if (p:=parents.pop().base).realized or p in realizes:
          if p in assign_targets and assign_targets[p] not in group: forced_realize, can_chase = True, False
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
          if not st.contiguous or tr_next.op in ReduceOps: break
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
      if len(children[top_reduce]) == 1: del realizes[top_reduce]

  for r in reduce_of_const:
    group = {tr:None for tr,rop in reduce_for_op.items() if rop is r}
    if DEBUG_ARANGE:=(getenv("DEBUG_ARANGE")): print(f"checking {r} {group=}")
    if any(tr.forced_realize for tr in group) or any(x.base in group for x in outs): continue
    kernel_children = {c for tr in group for c in children[tr] if c.op not in {MetaOps.COPY, MetaOps.VIEW}}
    if len(kernel_children) == 0: continue
    if DEBUG_ARANGE: print(colored(f"folding {r}", "green"))
    for tr in group: del realizes[tr]

  output_groups: DefaultDict[LazyBuffer, List[LazyBuffer]] = defaultdict(list)
  for buf in realizes:
    if buf.realized is not None or buf.op is MetaOps.CONST or buf in seen: continue
    output_groups[reduce_for_op[buf] if buf in reduce_for_op and MULTIOUTPUT else buf].append(buf)

    # make things that can't be images not images
    if isinstance(buf.dtype, ImageDType) and (prod(buf.shape) != prod(buf.dtype.shape) or
                                              not any(buf.shape[x]%4 == 0 for x in buf.st.unit_stride_axes())):
      if DEBUG >= 2: print(f"forcing image {buf.dtype} with shape {buf.shape} to float32")
      buf.dtype = dtypes.float32
      # hack the underlying buffer too
      if buf.base is buf:
        assert not hasattr(buf.buffer, '_buf'), "can't fixup allocated buffer"
        buf.buffer.dtype = dtypes.float32
        buf.buffer.options = None

  # preschedule all buffers in realizes
  prescheduled = [_lower_lazybuffer(group, realizes) for group in output_groups.values()]
  schedule_targets = {out:lsi for lsi in prescheduled for out in lsi.outputs}

  graph: DefaultDict[LBScheduleItem, List[LBScheduleItem]] = defaultdict(list)
  in_degree: DefaultDict[LBScheduleItem, int] = defaultdict(int)
  for lsi in prescheduled:
    if lsi not in in_degree: in_degree[lsi] = 0
    # realize outputs after all parents are realized
    scheduled_parents = dedup(schedule_targets[x] for x in lsi.inputs if x in schedule_targets)
    for x in scheduled_parents:
      graph[x].append(lsi)
      in_degree[lsi] += 1
    # realize outputs before a parent is assigned to
    parents_assigns = dedup(schedule_targets[assign_targets[x]] for x in lsi.inputs if x in assign_targets)
    for assign in parents_assigns:
      graph[lsi].append(assign)
      in_degree[assign] += 1

  if SAVE_SCHEDULE:
    def _save():
      print(f"saving {len(SCHEDULES)} schedule graphs to", fp:=getenv("SAVE_SCHEDULE_PATH", "schedule.pkl"))
      with open(fp, "wb") as f: pickle.dump(SCHEDULES, f)
    if len(SCHEDULES) == 0: atexit.register(_save)
    SCHEDULES.append((graph, in_degree))
  return graph, in_degree

# *** DAG ordering: breadth first search ***

def create_schedule_with_vars(outs:List[LazyBuffer], seen:Optional[Set[LazyBuffer]]=None) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
  if seen is None: seen = set()
  graph, in_degree = _graph_schedule(outs, seen)
  if getenv("RUN_PROCESS_REPLAY") and getenv("COMPARE_SCHEDULE", 1):
    # NOTE: process relpay needs PYTHONPATH=., remove this once it just pickles LazyBuffers
    with contextlib.suppress(Exception): importlib.import_module("test.external.process_replay.diff_schedule").process_replay(outs, graph, in_degree)

  queue = deque(lsi for lsi,deg in in_degree.items() if deg == 0)
  schedule: List[ScheduleItem] = []
  var_vals: Dict[Variable, int] = {}
  kernel_number = GlobalCounters.kernel_count
  while queue:
    lsi = queue.popleft()
    for buf in lsi.outputs: seen.add(buf)
    if GRAPH:
      kernel_number += 1
      for out in lsi.outputs: realized_lazybuffer(out, kernel_number)
    var_vals = merge_dicts([var_vals, lsi.var_vals])
    for out in lsi.outputs: del out.srcs  # can only schedule once
    schedule.append(si:=ScheduleItem(lsi.ast, tuple(x.buffer for x in lsi.outputs+lsi.inputs if x.size != 0), lsi.metadata))
    if logops and si.ast.op is MetaOps.KERNEL and not any(i.device.startswith("DISK:") for i in si.inputs): logops.write(str(si.ast)+"\n")
    for x in graph[lsi]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)

  # confirm everything was scheduled correctly
  if any(degree != 0 for degree in in_degree.values()) or len(in_degree) != len(schedule):
    raise RuntimeError(f"cycle detected in graph, prescheduled {len(in_degree)} but only scheduled {len(schedule)}")
  if DEBUG >= 1 and len(schedule) >= 10: print(f"scheduled {len(schedule)} kernels")
  return schedule, var_vals

def create_schedule(outs:List[LazyBuffer], seen:Optional[Set[LazyBuffer]]=None) -> List[ScheduleItem]:
  schedule, var_vals = create_schedule_with_vars(outs, seen)
  assert len(var_vals) == 0
  return schedule
