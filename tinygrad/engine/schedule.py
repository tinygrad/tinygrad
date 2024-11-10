import sys, atexit, functools, itertools
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Set, Tuple, List, Dict, Optional, DefaultDict, cast
from tinygrad.ops import GroupOp, UOp, Ops, PatternMatcher, UPat, Variable, can_pad, graph_rewrite, resolve, track_rewrites, sint
from tinygrad.helpers import DEBUG, Metadata, all_int, all_same, colored, diskcache_put, prod, dedup, getenv, unwrap
from tinygrad.dtype import ImageDType, dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, strides_for_shape
from tinygrad.engine.lazy import LazyBuffer
from tinygrad.engine.fuse import get_realizes
from tinygrad.device import Buffer

# creation can recurse a lot
sys.setrecursionlimit(10000)

BUF_LIMIT = {"METAL":32}

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
  def output_idxs(self) -> Tuple[int, ...]: return tuple(x.src[0].arg for x in self.ast.src) if self.ast.op is Ops.SINK else (0,)

# **** small wrapper for LazyBuffer -> UOp

@dataclass(frozen=True)
class ScheduleContext:
  buf_uops: Dict[Buffer, UOp] = field(default_factory=dict)        # this maps Buffers to BUFFER uops
  ubuf_metadata: Dict[UOp, Metadata] = field(default_factory=dict) # this maps BUFFER uops to Metadata
  var_vals: Dict[Variable, int] = field(default_factory=dict)      # this maps a BIND's DEFINE_VAR to its value
  assigns: Set[Buffer] = field(default_factory=set)                # this holds all the UOps.BUFFERs we ASSIGN to in this schedule
  lazybufs: Dict[Buffer, LazyBuffer] = field(default_factory=dict) # this is a lookup for the LazyBuffers we need to mark as realized

def to_uop(buf:LazyBuffer, ctx:ScheduleContext, children, allbufs, double_reduces, cache:Dict[LazyBuffer, UOp]) -> UOp:
  if (r:=cache.get(buf)) is not None: return r
  if buf is not buf.base:
    cache[buf] = ret = to_uop(buf.base, ctx, children, allbufs, double_reduces, cache).view(buf.st)
    return ret
  # make things that can't be images not images
  if isinstance(buf.dtype, ImageDType) and (prod(buf.shape) != prod(buf.dtype.shape) or
                                            not any(buf.shape[x]%4 == 0 for x in buf.st.unit_stride_axes())):
    if DEBUG >= 2: print(f"forcing image {buf.dtype} with shape {buf.shape} to {buf.dtype.base}")
    # hack the underlying buffer too
    buf.dtype = buf.buffer.dtype = buf.dtype.base
    assert not buf.is_realized(), "can't fixup allocated buffer"
    buf.buffer.options = None
  dtype = buf.dtype.base if isinstance(buf.dtype, ImageDType) else buf.dtype
  # consts are always fused and generated
  if buf.op is Ops.CONST:
    if isinstance(val:=buf.arg, UOp): ctx.var_vals.update([val.unbind()])
    return UOp(Ops.VALID, dtypes.bool, (buf.st.to_uop(),)).where(v:=UOp.const(dtype, buf.arg), v.const_like(0))
  # everything else has BUFFER
  ubuf = ctx.buf_uops.setdefault(b:=buf.buffer, UOp(Ops.BUFFER, b.dtype.ptr(), (), (len(ctx.buf_uops), (b.device, b.size, b.dtype))))
  # if the buffer is already realized we just load it
  if buf.is_realized(): return UOp(Ops.PRELOAD, dtype, (ubuf, buf.st.to_uop()))
  # everything else needs sources
  src = tuple(to_uop(x, ctx, children, allbufs, double_reduces, cache) for x in buf.srcs)
  if buf.op in GroupOp.Reduce: ret = src[0].r(buf.op, buf.arg)
  elif buf.op is Ops.CONTIGUOUS: ret = UOp(Ops.CONTIGUOUS, dtype, src)
  elif buf.op is Ops.ASSIGN:
    ctx.assigns.add(b)
    ret = UOp(Ops.ASSIGN, dtype, (ubuf, src[1]), buf.arg)
  elif buf.op in GroupOp.Meta: ret = UOp(buf.op, buf.dtype, (ubuf, *src), buf.arg)
  else: ret = UOp(cast(Ops, buf.op), dtype, src)
  if buf.forced_realize: ret = UOp(Ops.CONTIGUOUS, dtype, (ret,))
  cache[buf] = ret = UOp(Ops.LOAD, dtype, (ubuf, buf.st.to_uop(), UOp.store(ubuf, ShapeTracker.from_shape(buf.shape).to_uop(), ret)))
  if buf.metadata is not None: ctx.ubuf_metadata[ubuf] = buf.metadata
  ctx.lazybufs[b] = buf
  # things for fuse.py
  allbufs[buf] = None
  if buf.op in GroupOp.Reduce and buf.srcs[0].base.op is buf.op and buf.srcs[0] is not buf.srcs[0].base: double_reduces[buf] = None
  for x in buf.srcs:
    if x.base.realized is None: children[x.base][buf] = None
  return ret

# **** AST graph rewrite

# ** helpers for doing movementops on uops

def st_fixup(u:UOp, apply_to_st:Callable[[ShapeTracker], ShapeTracker], cache:Dict[UOp, UOp]) -> UOp:
  if (n:=cache.get(u)) is not None: return n
  if u.op is Ops.VIEW: return u.replace(arg=apply_to_st(u.arg))
  if len(u.src) == 0 or (u.st is not None and u.st == apply_to_st(u.st)): return u
  assert u.op is not Ops.REDUCE_AXIS, "can't push a fixup through a reduce"
  cache[u] = ret = u.replace(src=tuple(st_fixup(x, apply_to_st, cache) for x in u.src))
  return ret

def permute_reduce(input_st:ShapeTracker, axis:Tuple[int, ...]) -> Tuple[ShapeTracker, Tuple[sint, ...]]:
  permute_axis = tuple(i for i in range(len(input_st.shape)) if i not in axis)+axis
  tmp = input_st.permute(permute_axis)
  return tmp, tmp.shape[-len(axis):]

# ** movementops rewrite rules

def swizzle_r(r:UOp, src:UOp, st:ShapeTracker) -> UOp:
  tmp, rshape = permute_reduce(ShapeTracker.from_shape(unwrap(src.st).shape), r.axis_arg)
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
  return st_fixup(src, lambda st:st+new_input_st, {}).r(r.arg[0], new_axis).view(ShapeTracker.from_shape(st.shape))

def push_swizzle_down_through_reduce(root:UOp, swizzle:UOp, src:UOp) -> UOp:
  swizzle_st, src_st = unwrap(swizzle.st), unwrap(src.st)
  assert swizzle_st.contiguous, "can't push a non contiguous VIEW down to STORE"
  assert prod(swizzle_st.shape) == prod(src_st.shape), "can't push expands down to STORE"
  output_shape = swizzle_st.reduce(root.axis_arg)
  new_axis = tuple(i for i,(s,u) in enumerate(zip(src_st.shape, output_shape)) if s != u)
  return swizzle.src[0].r(root.arg[0], new_axis).view(ShapeTracker.from_shape(output_shape))

def push_swizzle_down_through_elementwise(root:UOp) -> Optional[UOp]:
  swizzles = [x for x in root.src if x.op is Ops.VIEW and len(x.src) != 0]
  if len(swizzles) == 0: return None
  swizzle_shapes = [(unwrap(x.st).shape, unwrap(x.src[0].st).shape) for x in swizzles]
  assert all_same([(x, prod(x), prod(y)) for x,y in swizzle_shapes]), f"swizzles must have the same size {swizzle_shapes}"
  new_shape, new_input_shape = swizzle_shapes[0]
  fixup_cache: Dict[UOp, UOp] = {}
  ret = root.replace(src=tuple(x.src[0] if x in swizzles else st_fixup(x, lambda st:st.reshape(new_input_shape), fixup_cache) for x in root.src))
  return ret if ret.op is Ops.STORE else ret.view(ShapeTracker.from_shape(new_shape))

def merge_double_reduce(root:UOp, first_reduce:UOp) -> UOp:
  assert root.arg[0] == first_reduce.arg[0], "can't merge reduceops with different alu"
  assert not any(x.op is Ops.REDUCE_AXIS for x in first_reduce.parents), "can't merge more than two reduceops at a time"
  return first_reduce.src[0].r(first_reduce.arg[0], root.axis_arg+first_reduce.axis_arg)

merge_views = PatternMatcher([(UPat(Ops.VIEW, src=(UPat(Ops.VIEW, name="s0"),), name="s1"), lambda s0,s1: s0.replace(arg=s0.st+s1.st))])

# push VIEW to loads
view_left = merge_views+PatternMatcher([
  # VIEW before elementwise ops
  (UPat({*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.ASSIGN}, name="e").view(name="v"),
   lambda e,v: e.replace(src=tuple(s.view(v.st) if s.has_st else s for s in e.src))),
  # early merge VIEW buffer ops
  (UPat(GroupOp.Buffer, name="b").view(name="v"), lambda b,v: b.replace(src=tuple((s.arg+v.arg).to_uop() if s.op is Ops.VIEW else s for s in b.src))),
])

# push VIEW to stores
view_right = merge_views+PatternMatcher([
  # ASSIGN can override st
  (UPat(Ops.STORE, src=(UPat.var("b"), UPat.var("st"), UPat(Ops.ASSIGN, name="a"))),
   lambda a,b,st: UOp.store(b, (a.arg[0]+st.arg).to_uop(), a.replace(arg=())) if a.arg else None),
  # non contiguous VIEW on a reduce creates a new VIEW
  (UPat(Ops.REDUCE_AXIS, src=UPat.var("src"), name="r").view(name="v"), lambda v,r,src: None if v.st.contiguous else swizzle_r(r, src, v.st)),
  # push a VIEW down to STORE, through a reduce (ONLY reshapes)
  (UPat(Ops.REDUCE_AXIS, src=(UPat.var(name="src").view(name="swizzle"),), name="root"), push_swizzle_down_through_reduce),
  # push VIEW(s) down to STORE, through an elementwise op (ONLY reshapes)
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.ASSIGN, Ops.CONTIGUOUS, Ops.STORE), name="root"), push_swizzle_down_through_elementwise),
  (UPat(Ops.REDUCE_AXIS, src=(UPat(Ops.REDUCE_AXIS, name="first_reduce"),), name="root"), merge_double_reduce),
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
  return UOp(Ops.DEFINE_GLOBAL, x.dtype, (), len(ctx.bufs)-1)
append_bufs = PatternMatcher([(UPat(Ops.BUFFER, name="x"), _append_buf)])

def _append_preload(ctx:ScheduleItemContext, x:UOp, b:UOp) -> UOp:
  if b in ctx.assigned: ctx.assign_preloads.append(b)
  return x.replace(op=Ops.LOAD)

to_si = PatternMatcher([
  (UPat(Ops.VIEW, name="x"), _append_st_vars),
  (UPat(Ops.PRELOAD, src=(UPat.var("b"), UPat()), name="x"), _append_preload),
  (UPat(Ops.SINK, src=(UPat.store(UPat(), UPat(), UPat(GroupOp.Meta, name="x")),)), lambda ctx,x: x),
])

# ** fusion

lazy = PatternMatcher([
  (UPat.load(b:=UPat.var("b"), UPat(), UPat.store(b, UPat(), UPat.var("v"))), lambda ctx,b,v: v),
  (UPat(Ops.CONTIGUOUS, src=(UPat.var("x"),)), lambda ctx,x: x),
])

multioutput = PatternMatcher([(UPat.load(UPat.var("b"), UPat()), lambda ctx,b: ctx.get(b)),])

def full_ast_rewrite(pre:UOp, var_vals:Dict[Variable, int], assigned:Set[UOp]) -> Tuple[UOp, ScheduleItemContext]:
  # fuse and fold store -> loads
  sink = graph_rewrite(pre, lazy+multioutput if len(pre.src)>1 else lazy, {x.src[0]:x.src[2] for x in pre.src})
  # assert cyclic dependency
  for b,ops in itertools.groupby((x for x in sink.sparents if x.op in {Ops.PRELOAD,Ops.LOAD} and x.src[0] in assigned), key=lambda x:x.src[0]):
    if not all_same([x.op for x in ops]):
      raise RuntimeError(f"cycle detected in kernel.\nhelp: use .contiguous() to break the part loading pre-assign {b} into a different kernel.")
  # do movementops
  sink = graph_rewrite(graph_rewrite(sink, view_left), view_right)
  # we also allow masked views. if it has a single view and it's equal when you shrink a contig, it's fine
  if len(assign_targets:=[x.src[0] for x in sink.sparents if x.op is Ops.ASSIGN]) != 0:
    if not all((s:=x.st_arg).contiguous or (len(s.views) == 1 and (m:=s.views[0].mask) is not None \
        and ShapeTracker.from_shape(s.shape).shrink(m) == s.shrink(m)) for x in sink.sparents if x.op is Ops.PRELOAD and x.src[0] in assign_targets):
      raise RuntimeError("self operand of augmented assign must be contiguous.\nhelp: consider using .contiguous():\n"
                         +colored("   - a += a.T\n", "red")+colored("   + a += a.T.contiguous()", "green"))
  # convert to AST
  sink = graph_rewrite(graph_rewrite(sink, to_si, ctx:=ScheduleItemContext(var_vals, assigned)), append_bufs, ctx)
  if getenv("RUN_PROCESS_REPLAY"): PROCESS_REPLAY_CAPTURE.append(((pre, var_vals, assigned), sink))
  return sink, ctx

PROCESS_REPLAY_CAPTURE: List[Tuple[Tuple, UOp]] = []
if getenv("RUN_PROCESS_REPLAY"):
  @atexit.register
  def save_process_replay():
    for x,ret in PROCESS_REPLAY_CAPTURE: diskcache_put("schedule_process_replay", str(x[0].key), (x, {}, ret))

# **** Schedule creation and BFS toposort

def realize(ctx:Dict[UOp, UOp], b:UOp, load:UOp, store:UOp) -> UOp:
  ctx[b] = store
  return UOp(Ops.LOAD, load.dtype, (b, load.st_arg.to_uop()))

def realize_view(ctx:Dict[UOp, UOp], base:UOp, view:UOp, **kwargs) -> Optional[UOp]:
  base_shape = unwrap(base.st).shape
  st = unwrap(view.st)
  # fold simple pads
  if len(st.views) == 1 and (m:=st.views[-1].mask) is not None and all_int(base_shape) and resolve(prod(base_shape) >= prod([y-x for x,y in m])):
    return None if can_pad(base) else realize(ctx, **kwargs).view(st)
  # early realize before expand
  if resolve(prod(base_shape) < prod(st.shape)): return realize(ctx, **kwargs).view(st)
  # otherwise safety check pads
  return None if (all(v.mask is None for v in st.views) or can_pad(base)) else realize(ctx, **kwargs).view(st)

def UPatLoadStore(to_store=UPat()): return UPat.load(b:=UPat.var("b"), UPat(), UPat.store(b, UPat(), to_store, name="store"), name="load")
do_realize = PatternMatcher([
  # always realize meta ops
  (UPatLoadStore(UPat((Ops.ASSIGN, Ops.CONTIGUOUS, *GroupOp.Meta))), realize),
  # don't realize image to image casts
  (UPatLoadStore(UPat(Ops.CAST, src=(UPat(Ops.LOAD, name="x"),), dtype=dtypes.float)).view(name="view"), lambda ctx,x,view,**kwargs: r.view(view.st)
   if (r:=ctx.get(b:=x.src[0])) is not None and r.op is Ops.STORE and isinstance(b.dtype, ImageDType) else None),
  # realize before expand or unsafe pad ops
  (UPatLoadStore(UPat.var("base")).view(name="view"), realize_view),
  # realize before COPY or BUFFER_VIEW
  (UPat((Ops.COPY, Ops.BUFFER_VIEW), src=(UPat.var("u"), UPat.any(UPatLoadStore(), UPatLoadStore().view(name="view"))), name="root"),
   lambda ctx,root,u,view=None,**kwargs: root.replace(src=(u, realize(ctx,**kwargs) if view is None else realize(ctx,**kwargs).view(view.st))),),
])
break_sched = PatternMatcher([(UPatLoadStore(), lambda ctx,b,store,load: realize(ctx, b, load, store) if b in ctx else None),])

@track_rewrites(named=True)
def create_schedule_with_vars(outs:List[LazyBuffer]) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
  if len(outs:=dedup(x.base for x in outs if x.realized is None and x.base.op is not Ops.CONST)) == 0: return [], {}
  for out in outs: out.forced_realize = True
  # create the big graph
  ctx = ScheduleContext()
  cache: Dict[LazyBuffer, UOp] = {}
  # **** TODO: delete these next 3 after big graph
  children: DefaultDict[LazyBuffer, Dict[LazyBuffer, None]] = defaultdict(dict)
  allbufs: Dict[LazyBuffer, None] = {}
  double_reduces: Dict[LazyBuffer, None] = {}
  big_graph = UOp.sink(*(to_uop(x, ctx, children, allbufs, double_reduces, cache) for x in outs))
  # get realizes
  realizes: Dict[UOp, UOp] = {}
  graph_rewrite(big_graph, do_realize, realizes)
  store_groups = get_realizes(outs, children, allbufs, double_reduces, realizes, ctx)
  # split realizes into small graphs
  graph_rewrite(big_graph, break_sched, realizes)
  sinks = [UOp.sink(*(realizes[u] for u in stores)) for stores in store_groups]
  # preschedule all realizes
  bufs = list(ctx.buf_uops)
  prescheduled: List[ScheduleItem] = []
  assigned = {ubuf for b in ctx.assigns if (ubuf:=ctx.buf_uops.get(b)) is not None}
  for sink in sinks:
    metadata = tuple({mx for x in sink.sparents if x.op in GroupOp.Buffer and len(x.src) > 2 and (mx:=ctx.ubuf_metadata.get(x.src[0]))})
    ast, ast_ctx = full_ast_rewrite(sink, ctx.var_vals, assigned)
    prescheduled.append(ScheduleItem(ast, tuple(b for u in ast_ctx.bufs if (b:=bufs[u.arg[0]]).size != 0), metadata, tuple(ast_ctx.assign_preloads)))
  # do BFS
  schedule_targets = {out:si for si in prescheduled for out in si.outputs}
  graph: DefaultDict[ScheduleItem, List[ScheduleItem]] = defaultdict(list)
  in_degree: DefaultDict[ScheduleItem, int] = defaultdict(int)
  for si in prescheduled:
    # realize outputs before a parent is assigned to
    parents_assigns = dedup(xsi for x in si.assign_preloads if (xsi:=schedule_targets.get(bufs[x.arg[0]])) and xsi is not si)
    for assign in parents_assigns:
      graph[si].append(assign)
      in_degree[assign] += 1
    # realize outputs after all parents are realized
    scheduled_parents = dedup(xsi for x in si.inputs if (xsi:=schedule_targets.get(x)) is not None and xsi not in parents_assigns)
    for x in scheduled_parents:
      graph[x].append(si)
      in_degree[si] += 1
  queue = deque(si for si in prescheduled if in_degree[si] == 0)
  schedule: List[ScheduleItem] = []
  while queue:
    schedule.append(si:=queue.popleft())
    for b in si.outputs: del ctx.lazybufs[b].srcs  # can only schedule once
    if (m:=BUF_LIMIT.get(device:=si.outputs[0].device)) and len(si.bufs) >= m:
      if DEBUG >= 3: print(si)
      raise RuntimeError(f"Kernel for {si.metadata} exceeded the {m} buffer count limit for {device} with {len(si.bufs)} buffers.")
    for x in graph[si]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)
  # confirm everything was scheduled correctly
  if len(schedule) != (groups:=len(prescheduled)): raise RuntimeError(f"cycle detected in graph, grouped {groups} but only scheduled {len(schedule)}")
  if DEBUG >= 1 and len(schedule) >= 10: print(f"scheduled {len(schedule)} kernels")
  return schedule, ctx.var_vals

def create_schedule(outs:List[LazyBuffer]) -> List[ScheduleItem]:
  schedule, var_vals = create_schedule_with_vars(outs)
  assert len(var_vals) == 0
  return schedule
