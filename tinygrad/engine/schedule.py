import sys, atexit, functools, itertools
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Set, Tuple, List, Dict, Optional, DefaultDict, cast
from tinygrad.ops import BUFFER_UOPS, MetaOps, ReduceOps, UnaryOps, UOp, UOps, PatternMatcher, UPat, Variable, graph_rewrite, track_rewrites, sint
from tinygrad.helpers import DEBUG, Metadata, all_same, colored, diskcache_put, prod, dedup, getenv, unwrap
from tinygrad.dtype import ImageDType
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, strides_for_shape
from tinygrad.engine.lazy import LazyBuffer
from tinygrad.engine.fuse import get_realizes
from tinygrad.device import Buffer

# creation can recurse a lot
sys.setrecursionlimit(10000)

BUF_LIMIT = {"METAL":32}
METAOPS = {MetaOps.COPY:UOps.COPY, MetaOps.EMPTY:UOps.EMPTY, MetaOps.VIEW:UOps.BUFFER_VIEW}

# **** ScheduleItem return type

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: Tuple[Buffer, ...]
  metadata: Tuple[Metadata, ...]
  assign_preloads: Tuple[UOp, ...]
  @property
  def outputs(self) -> Tuple[Buffer, ...]:
    """Read/write or write only buffers in the schedule."""
    return tuple(b for i,b in enumerate(self.bufs) if i in self.output_idxs)
  @property
  def inputs(self) -> Tuple[Buffer, ...]:
    """Read only buffers in the schedule."""
    return tuple(b for i,b in enumerate(self.bufs) if i not in self.output_idxs)
  @functools.cached_property
  def output_idxs(self) -> Tuple[int, ...]: return tuple(x.src[0].arg for x in self.ast.src) if self.ast.op is UOps.SINK else (0,)

# **** small wrapper for LazyBuffer -> UOp

def to_uop(buf:LazyBuffer, outputs:List[LazyBuffer], buf_uops:Dict[Buffer, UOp], metadata:Dict[UOp, Metadata], cache:Dict[LazyBuffer, UOp]) -> UOp:
  if (r:=cache.get(buf)) is not None: return r
  if buf is not buf.base:
    cache[buf] = ret = to_uop(buf.base, outputs, buf_uops, metadata, cache).view(buf.st)
    return ret
  if buf.op is MetaOps.CONST: return buf_uops[buf.buffer]
  dtype = buf.dtype.base if isinstance(buf.dtype, ImageDType) else buf.dtype
  if buf.is_realized(): return UOp(UOps.PRELOAD, dtype, (buf_uops[buf.buffer], buf.st.to_uop()))
  if (ubuf:=buf_uops.get(buf.buffer)) is not None and buf not in outputs: return UOp(UOps.LOAD, dtype, (ubuf, buf.st.to_uop()))
  src = tuple(to_uop(x, outputs, buf_uops, metadata, cache) for x in buf.srcs)
  if buf.op in ReduceOps: ret = src[0].r(buf.op, buf.arg)
  elif buf.op is MetaOps.CONTIGUOUS: ret = UOp(UOps.CONTIGUOUS, dtype, src)
  elif buf.op is MetaOps.ASSIGN: ret = UOp(UOps.ASSIGN, dtype, (buf_uops[buf.buffer], src[1]), buf.arg)
  elif buf.op in METAOPS: ret = UOp(METAOPS[cast(MetaOps, buf.op)], buf.dtype, (buf_uops[buf.buffer], *src), buf.arg)
  elif buf.op is UnaryOps.CAST: ret = UOp(UOps.CAST, dtype, src)
  elif buf.op is UnaryOps.BITCAST: ret = UOp(UOps.BITCAST, dtype, src)
  else: ret = UOp(UOps.ALU, dtype, src, buf.op)
  cache[buf] = ret
  if buf.metadata is not None: metadata[ret] = buf.metadata
  return ret

# **** AST graph rewrite

# ** helpers for doing movementops on uops

def st_fixup(u:UOp, apply_to_st:Callable[[ShapeTracker], ShapeTracker], cache:Dict[UOp, UOp]) -> UOp:
  if (n:=cache.get(u)) is not None: return n
  if u.op is UOps.VIEW: return u.replace(arg=apply_to_st(u.arg))
  if len(u.src) == 0 or (u.st is not None and u.st == apply_to_st(u.st)): return u
  cache[u] = ret = u.replace(src=tuple(st_fixup(x, apply_to_st, cache) for x in u.src))
  return ret

def permute_reduce(input_st:ShapeTracker, axis:Tuple[int, ...]) -> Tuple[ShapeTracker, Tuple[sint, ...]]:
  permute_axis = tuple(i for i in range(len(input_st.shape)) if i not in axis)+axis
  tmp = input_st.permute(permute_axis)
  return tmp, tmp.shape[-len(axis):]

# ** movementops rewrite rules

def view_r(view:UOp, r:UOp, rsrc:UOp) -> Optional[UOp]:
  if (st:=unwrap(view.st)).contiguous: return None
  tmp, rshape = permute_reduce(ShapeTracker.from_shape(unwrap(rsrc.st).shape), r.axis_arg)
  prshape = prod(rshape)
  strides = strides_for_shape(rshape)
  nv: List[View] = []
  for v in st.views:
    nv.append(View.create(v.shape+rshape, tuple(x*prshape for x in v.strides)+strides,
                          v.offset*prshape, v.mask+tuple((0,s) for s in rshape) if v.mask is not None else None))
  # update input_st and axis
  new_input_st = tmp + ShapeTracker(tuple(nv))
  _, new_rshape = permute_reduce(new_input_st, r.axis_arg)
  new_axis = tuple(range(len(new_input_st.shape)-len(new_rshape), len(new_input_st.shape)))
  return st_fixup(rsrc, lambda st:st+new_input_st, {}).r(r.arg[0], new_axis).view(ShapeTracker.from_shape(st.shape))

def push_swizzle_down_through_reduce(root:UOp, swizzle:UOp) -> UOp:
  swizzle_st, src_st = unwrap(swizzle.st), unwrap(swizzle.src[0].st)
  assert swizzle_st.contiguous, "can't push a non contiguous VIEW down to STORE"
  assert prod(swizzle_st.shape) == prod(src_st.shape), "can't push expands down to STORE"
  output_shape = swizzle_st.reduce(root.axis_arg)
  new_axis = tuple(i for i,(s,u) in enumerate(zip(src_st.shape, output_shape)) if s != u)
  return swizzle.src[0].r(root.arg[0], new_axis).view(ShapeTracker.from_shape(output_shape))

def push_swizzle_down_through_elementwise(root:UOp) -> Optional[UOp]:
  swizzles = [x for x in root.src if x.op is UOps.VIEW and len(x.src) != 0]
  if len(swizzles) == 0: return None
  swizzle_shapes = [(unwrap(x.st).shape, unwrap(x.src[0].st).shape) for x in swizzles]
  assert all_same([(x, prod(x), prod(y)) for x,y in swizzle_shapes]), f"swizzles must have the same size {swizzle_shapes}"
  new_shape, new_input_shape = swizzle_shapes[0]
  fixup_cache: Dict[UOp, UOp] = {}
  new_srcs = [x.src[0] if x in swizzles else st_fixup(x, lambda st:st.reshape(new_input_shape), fixup_cache) for x in root.src]
  ret = UOp(root.op, root.dtype, tuple(new_srcs), root.arg)
  return ret if ret.op is UOps.STORE else ret.view(ShapeTracker.from_shape(new_shape))

def merge_double_reduce(root:UOp, first_reduce:UOp) -> UOp:
  assert root.arg[0] == first_reduce.arg[0], "can't merge reduceops with different alu"
  assert not any(x.op is UOps.REDUCE_AXIS for x in first_reduce.parents), "can't merge more than two reduceops at a time"
  return first_reduce.src[0].r(first_reduce.arg[0], root.axis_arg+first_reduce.axis_arg)

merge_views = PatternMatcher([(UPat(UOps.VIEW, src=(UPat(UOps.VIEW, name="s0"),), name="s1"), lambda s0,s1: s0.replace(arg=s0.st+s1.st))])

# push VIEW to loads
view_left = merge_views+PatternMatcher([
  # view before ALU
  (UPat(UOps.VIEW, src=(UPat((UOps.ALU, UOps.CAST, UOps.BITCAST, UOps.ASSIGN, UOps.CONTIGUOUS, *BUFFER_UOPS), name="e"),), name="v"),
   lambda e,v: e.replace(src=tuple(s.view(v.st) if s.has_st else s for s in e.src))),
])

# push VIEW to stores
view_right = merge_views+PatternMatcher([
  # ASSIGN can override st
  (UPat(UOps.STORE, src=(UPat.var("b"), UPat.var("st"), UPat(UOps.ASSIGN, name="a"))),
   lambda a,b,st: UOp.store(b, (a.arg[0]+st.arg).to_uop(), a.replace(arg=())) if a.arg else None),
  # VIEW on a reduce creates a new VIEW
  (UPat(UOps.VIEW, src=(UPat(UOps.REDUCE_AXIS, src=UPat.var("rsrc"), name="r"),), name="view"), view_r),
  # push a VIEW down to STORE, through a reduce (ONLY reshapes)
  (UPat(UOps.REDUCE_AXIS, src=(UPat(UOps.VIEW, name="swizzle"),), name="root"), push_swizzle_down_through_reduce),
  # push VIEW(s) down to STORE, through an elementwise op (ONLY reshapes)
  (UPat((UOps.ALU, UOps.CAST, UOps.BITCAST, UOps.ASSIGN, UOps.CONTIGUOUS, UOps.STORE), name="root"), push_swizzle_down_through_elementwise),
  (UPat(UOps.REDUCE_AXIS, src=(UPat(UOps.REDUCE_AXIS, name="first_reduce"),), name="root"), merge_double_reduce),
])

# ** ScheduleItem context builder

@dataclass(frozen=True)
class ScheduleItemContext:
  var_vals: Dict[Variable, int]
  assigned: Set[UOp]
  sts: Set[ShapeTracker] = field(default_factory=set)
  bufs: List[UOp] = field(default_factory=list)
  assign_preloads: List[UOp] = field(default_factory=list)

def _append_st_vars(ctx:ScheduleItemContext, x:UOp) -> Optional[UOp]:
  if (st:=unwrap(x.st)) in ctx.sts: return None
  st, var_vals = st.simplify().unbind()
  ctx.var_vals.update(var_vals)
  ctx.sts.add(st)
  return st.to_uop() if st != x.st else None

def _append_buf(ctx:ScheduleItemContext, x:UOp) -> UOp:
  ctx.bufs.append(x)
  return UOp(UOps.DEFINE_GLOBAL, x.dtype, (), len(ctx.bufs)-1)
append_bufs = PatternMatcher([(UPat(UOps.BUFFER, name="x"), _append_buf)])

def _append_preload(ctx:ScheduleItemContext, x:UOp, b:UOp) -> UOp:
  if b in ctx.assigned: ctx.assign_preloads.append(b)
  return x.replace(op=UOps.LOAD)

to_si = PatternMatcher([
  (UPat(UOps.VIEW, name="x"), _append_st_vars),
  (UPat(UOps.PRELOAD, src=(UPat.var("b"), UPat()), name="x"), _append_preload),
  (UPat(UOps.CONTIGUOUS, src=(UPat.var("x"),)), lambda _,x: x),
  (UPat(UOps.SINK, src=(UPat.store(UPat(), UPat(), UPat(tuple(METAOPS.values()), name="x")),)), lambda _,x: x),
])

def full_ast_rewrite(pre:UOp, ctx:ScheduleItemContext) -> UOp:
  # assert cyclic dependency
  for b,reads in itertools.groupby((x for x in pre.sparents if x.op in {UOps.PRELOAD,UOps.LOAD} and x.src[0] in ctx.assigned), key=lambda x:x.src[0]):
    if not all_same([x.op for x in reads]):
      raise RuntimeError(f"cycle detected in kernel.\nhelp: use .contiguous() to break the part loading pre-assign {b} into a different kernel.")
  # do movementops
  sink = graph_rewrite(graph_rewrite(pre, view_left), view_right)
  # convert to AST
  sink = graph_rewrite(graph_rewrite(sink, to_si, ctx), append_bufs, ctx)
  # we also allow masked views. if it has a single view and it's equal when you shrink a contig, it's fine
  if len(assign_targets:=[x.src[0] for x in sink.sparents if x.op is UOps.ASSIGN]) != 0:
    if not all((s:=x.st_arg).contiguous or (len(s.views) == 1 and (m:=s.views[0].mask) is not None \
        and ShapeTracker.from_shape(s.shape).shrink(m) == s.shrink(m)) for x in sink.sparents if x.op is UOps.LOAD and x.src[0] in assign_targets):
      raise RuntimeError("self operand of augmented assign must be contiguous.\nhelp: consider using .contiguous():\n"
                         +colored("   - a += a.T\n", "red")+colored("   + a += a.T.contiguous()", "green"))
  PROCESS_REPLAY_CAPTURE.append((pre, ScheduleItemContext(ctx.var_vals, ctx.assigned), sink))
  return sink

PROCESS_REPLAY_CAPTURE: List[Tuple[UOp, ScheduleItemContext, UOp]] = []
if getenv("RUN_PROCESS_REPLAY"):
  @atexit.register
  def save_process_replay():
    for base_sink,ctx,ret in PROCESS_REPLAY_CAPTURE: diskcache_put("schedule_process_replay", str(base_sink.key), (base_sink, ctx, ret))

# **** Schedule creation and BFS toposort

@track_rewrites(named=True)
def create_schedule_with_vars(outs:List[LazyBuffer]) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
  ctx, store_groups, lazybufs_to_realize = get_realizes(outs)
  # preschedule all buffers in realizes
  prescheduled: List[ScheduleItem] = []
  for stores in store_groups:
    outs = [lazybufs_to_realize[ctx.uop_bufs[b]] for b in stores]
    cache: Dict[LazyBuffer, UOp] = {}
    metadata: Dict[UOp, Metadata] = {}
    sink = UOp(UOps.SINK, src=tuple(UOp.store(ctx.buf_uops[out.buffer], ShapeTracker.from_shape(out.shape).to_uop(),
                                              to_uop(out, outs, ctx.buf_uops, metadata, cache)) for out in outs))
    prescheduled.append(si:=ScheduleItem(full_ast_rewrite(sink, si_ctx:=ScheduleItemContext(ctx.var_vals, ctx.assigned)), \
        tuple(b for u in si_ctx.bufs if (b:=ctx.uop_bufs[u]).size != 0), tuple(dedup(metadata.values())), tuple(si_ctx.assign_preloads)))
    if (m:=BUF_LIMIT.get(device:=si.outputs[0].device)) and len(si.bufs) >= m:
      if DEBUG >= 3: print(si)
      raise RuntimeError(f"Kernel for {si.metadata} exceeded the {m} buffer count limit for {device} with {len(si.bufs)} buffers.")
  schedule_targets = {out:lsi for lsi in prescheduled for out in lsi.outputs}

  # do BFS
  graph: DefaultDict[ScheduleItem, List[ScheduleItem]] = defaultdict(list)
  in_degree: DefaultDict[ScheduleItem, int] = defaultdict(int)
  for lsi in prescheduled:
    # realize outputs before a parent is assigned to
    parents_assigns = dedup(xsi for x in lsi.assign_preloads if (xsi:=schedule_targets.get(ctx.uop_bufs[x])) and xsi is not lsi)
    for assign in parents_assigns:
      graph[lsi].append(assign)
      in_degree[assign] += 1
    # realize outputs after all parents are realized
    scheduled_parents = dedup(xsi for x in lsi.inputs if (xsi:=schedule_targets.get(x)) is not None and xsi not in parents_assigns)
    for x in scheduled_parents:
      graph[x].append(lsi)
      in_degree[lsi] += 1
  queue = deque(lsi for lsi in prescheduled if in_degree[lsi] == 0)
  schedule: List[ScheduleItem] = []
  while queue:
    schedule.append(si:=queue.popleft())
    for b in si.outputs: del lazybufs_to_realize[b].srcs  # can only schedule once
    for x in graph[si]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)
  # confirm everything was scheduled correctly
  if any(degree != 0 for degree in in_degree.values()) or len(in_degree) != len(schedule):
    raise RuntimeError(f"cycle detected in graph, prescheduled {len(in_degree)} but only scheduled {len(schedule)}")
  if DEBUG >= 1 and len(schedule) >= 10: print(f"scheduled {len(schedule)} kernels")
  return schedule, ctx.var_vals

def create_schedule(outs:List[LazyBuffer]) -> List[ScheduleItem]:
  schedule, var_vals = create_schedule_with_vars(outs)
  assert len(var_vals) == 0
  return schedule
