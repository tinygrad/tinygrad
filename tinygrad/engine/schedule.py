import sys, atexit, functools, itertools
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Set, Tuple, List, Dict, Optional, DefaultDict, cast
from tinygrad.ops import GroupOp, UOp, Ops, PatternMatcher, UPat, Variable, can_pad, graph_rewrite, resolve, track_rewrites, sint
from tinygrad.helpers import Context, Metadata, all_int, all_same, colored, diskcache_put, merge_dicts, prod, dedup, getenv, unwrap
from tinygrad.helpers import FUSE_CONV_BW, FUSE_ARANGE, DEBUG
from tinygrad.dtype import ImageDType, dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, strides_for_shape
from tinygrad.engine.lazy import LazyBuffer
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

def UPatSrc(*args, **kwargs): return UPat(Ops.VIEW, src=(UPat.var("b"), UPat(*args, **{**kwargs, "name":"to_store"})), name="base")
@functools.lru_cache(None)
def is_scheduled(u:UOp): return u.op is Ops.VIEW and len(u.src) == 2

@dataclass(frozen=True)
class ScheduleContext:
  buf_uops: Dict[UOp, Buffer] = field(default_factory=dict)        # this maps BUFFER uops to Buffers
  ubuf_metadata: Dict[UOp, Metadata] = field(default_factory=dict) # this maps BUFFER uops to Metadata
  var_vals: Dict[Variable, int] = field(default_factory=dict)      # this maps a BIND's DEFINE_VAR to its value
  lazybufs: Dict[Buffer, LazyBuffer] = field(default_factory=dict) # this is a lookup for the LazyBuffers we need to mark as realized

def to_uop(buf:LazyBuffer, ctx:ScheduleContext, cache:Dict[LazyBuffer, UOp]) -> UOp:
  if (r:=cache.get(buf)) is not None: return r
  if buf is not buf.base:
    cache[buf] = ret = to_uop(buf.base, ctx, cache).view(buf.st)
    return ret
  # make things that can't be images not images
  if isinstance(buf.dtype, ImageDType) and (prod(buf.shape) != prod(buf.dtype.shape) or
                                            not any(buf.shape[x]%4 == 0 for x in buf.st.unit_stride_axes())):
    if DEBUG >= 2: print(f"forcing image {buf.dtype} with shape {buf.shape} to {buf.dtype.base}")
    # hack the underlying buffer too
    buf.dtype = buf.buffer.dtype = buf.dtype.base
    assert not buf.is_realized, "can't fixup allocated buffer"
    buf.buffer.options = None
  dtype = buf.dtype if buf.op in GroupOp.Meta else buf.dtype.base
  # consts are always fused and generated
  if buf.op is Ops.CONST:
    if isinstance(val:=buf.arg, UOp): ctx.var_vals.update([val.unbind()])
    return UOp(Ops.VALID, dtypes.bool, (buf.st.to_uop(),)).where(UOp.const(dtype, val), 0)
  # everything else is a VIEW of BUFFER (with an optional op)
  if buf.is_realized:
    ctx.buf_uops[ubuf:=UOp.new_buffer((b:=buf.buffer).device, b.size, b.dtype, num=len(ctx.buf_uops))] = buf.buffer
    op = None
  elif buf.op is Ops.ASSIGN:
    target, new_val = [to_uop(x, ctx, cache) for x in buf.srcs]
    op = UOp(Ops.ASSIGN, dtype, (ubuf:=target.buf_uop, new_val), buf.arg)
  else:
    ctx.buf_uops[ubuf:=UOp.new_buffer((b:=buf.buffer).device, b.size, b.dtype, num=len(ctx.buf_uops))] = buf.buffer
    op = UOp(cast(Ops, buf.op), dtype, tuple(to_uop(x, ctx, cache) for x in buf.srcs),
             None if buf.op in {Ops.CAST, Ops.BITCAST} else buf.arg)
  cache[buf] = ret = UOp(Ops.VIEW, dtype.base, (ubuf,) if op is None else (ubuf, op.contiguous() if buf.forced_realize else op), buf.st)
  if op is not None:
    if buf.metadata is not None: ctx.ubuf_metadata[ubuf] = buf.metadata
    ctx.lazybufs[buf.buffer] = buf
  return ret

# **** AST graph rewrite

# ** helpers for doing movementops on uops

def apply_swizzle(u:UOp, arg:ShapeTracker) -> UOp:
  with Context(TRACK_MATCH_STATS=0): return graph_rewrite(u.view(arg), view_left)

def permute_reduce(input_st:ShapeTracker, axis:Tuple[int, ...]) -> Tuple[ShapeTracker, Tuple[sint, ...]]:
  permute_axis = tuple(i for i in range(len(input_st.shape)) if i not in axis)+axis
  tmp = input_st.permute(permute_axis)
  return tmp, tmp.shape[-len(axis):]

# ** movementops rewrite rules

def swizzle_r(r:UOp, src:UOp, st:ShapeTracker) -> UOp:
  tmp, rshape = permute_reduce(ShapeTracker.from_shape(unwrap(src.st).shape), r.axis_arg)
  prshape = prod(rshape)
  strides = strides_for_shape(rshape)
  nv = [View.create(v.shape+rshape, tuple(x*prshape for x in v.strides)+strides,
                    v.offset*prshape, v.mask+tuple((0,s) for s in rshape) if v.mask is not None else None) for v in st.views]
  # update input_st and axis
  new_input_st = tmp + ShapeTracker(tuple(nv))
  _, new_rshape = permute_reduce(new_input_st, r.axis_arg)
  new_axis = tuple(range(len(new_input_st.shape)-len(new_rshape), len(new_input_st.shape)))
  return apply_swizzle(src, new_input_st).r(r.arg[0], new_axis).view(ShapeTracker.from_shape(st.shape))

def push_swizzle_down_through_reduce(root:UOp, swizzle:UOp, src:UOp) -> UOp:
  swizzle_st, src_st = unwrap(swizzle.st), unwrap(src.st)
  assert swizzle_st.contiguous, "can't push a non contiguous VIEW down to STORE"
  assert prod(swizzle_st.shape) == prod(src_st.shape), "can't push expands down to STORE"
  output_shape = swizzle_st.reduce(root.axis_arg)
  new_axis = tuple(i for i,(s,u) in enumerate(zip(src_st.shape, output_shape)) if s != u)
  return swizzle.src[0].r(root.arg[0], new_axis).view(ShapeTracker.from_shape(output_shape))

def push_swizzle_down_through_elementwise(root:UOp) -> Optional[UOp]:
  swizzles = [x for x in root.src if x.base is not x]
  if len(swizzles) == 0: return None
  swizzle_shapes = [(unwrap(x.st).shape, unwrap(x.src[0].st).shape) for x in swizzles]
  assert all_same([(x, prod(x), prod(y)) for x,y in swizzle_shapes]), f"swizzles must have the same size {swizzle_shapes}"
  new_shape, new_input_shape = swizzle_shapes[0]
  ret = root.replace(src=tuple(x.src[0] if x in swizzles else apply_swizzle(x, ShapeTracker.from_shape(new_input_shape)) for x in root.src))
  return ret if ret.op is Ops.STORE else ret.view(ShapeTracker.from_shape(new_shape))

def merge_double_reduce(root:UOp, first_reduce:UOp) -> UOp:
  assert root.arg[0] == first_reduce.arg[0], "can't merge reduceops with different alu"
  assert not any(x.op is Ops.REDUCE_AXIS for x in first_reduce.parents), "can't merge more than two reduceops at a time"
  return first_reduce.src[0].r(first_reduce.arg[0], root.axis_arg+first_reduce.axis_arg)

merge_views = PatternMatcher([(UPat(Ops.VIEW, src=(UPat(Ops.VIEW, name="s0"),), name="s1"), lambda s0,s1: s0.replace(arg=s0.st+s1.st))])

# push VIEW to loads
view_left = merge_views+PatternMatcher([
  # VIEW before elementwise ops
  (UPat({*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.ASSIGN}, name="e").view(name="v"), lambda e,v: e.replace(src=tuple(s.view(v.st) for s in e.src))),
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
  (UPat(Ops.SINK, src=(UPat.store(UPat.var("b"), UPat(), UPat(GroupOp.Meta, name="x")),)), lambda ctx,b,x: x.replace(src=(b, *x.src))),
])

# ** fusion

lazy = PatternMatcher([
  (UPatSrc(), lambda ctx,to_store,**kwargs: to_store),
  (UPat(Ops.BUFFER, name="b").view(name="view"), lambda ctx,b,view: UOp(Ops.PRELOAD, view.dtype, (b, view.st.to_uop()))),
  (UPat(Ops.CONTIGUOUS, src=(UPat.var("x"),)), lambda ctx,x: x),
])

multioutput = PatternMatcher([(UPat.load(UPat.var("b"), UPat()), lambda ctx,b: ctx.get(b)),])

def full_ast_rewrite(pre:UOp, var_vals:Dict[Variable, int], assigned:Set[UOp]) -> Tuple[UOp, ScheduleItemContext]:
  # fuse and fold store -> loads
  sink = graph_rewrite(pre, lazy+multioutput if len(pre.src)>1 else lazy, {x.buf_uop:x.src[2] for x in pre.src})
  # assert cyclic dependency
  for b,ops in itertools.groupby((x for x in sink.sparents if x.op in {Ops.PRELOAD,Ops.LOAD} and x.buf_uop in assigned), key=lambda x:x.buf_uop):
    if not all_same([x.op for x in ops]):
      raise RuntimeError(f"cycle detected in kernel.\nhelp: use .contiguous() to break the part loading pre-assign {b} into a different kernel.")
  # do movementops
  sink = graph_rewrite(graph_rewrite(sink, view_left), view_right)
  # we also allow masked views. if it has a single view and it's equal when you shrink a contig, it's fine
  if len(assign_targets:=[x.buf_uop for x in sink.sparents if x.op is Ops.ASSIGN]) != 0:
    if not all((s:=x.st_arg).contiguous or (len(s.views) == 1 and (m:=s.views[0].mask) is not None \
        and ShapeTracker.from_shape(s.shape).shrink(m) == s.shrink(m)) for x in sink.sparents if x.op is Ops.PRELOAD and x.buf_uop in assign_targets):
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

# **** Schedule grouping

def uval(u:UOp) -> UOp:
  assert is_scheduled(u), f"must be a scheduled op {u}"
  return to_store.src[0] if (to_store:=u.src[1]).is_contiguous_base else to_store

def recursive_group(tr:UOp, st:ShapeTracker, r:UOp, children:DefaultDict[UOp, Dict[UOp, None]], allbufs:Dict[UOp, UOp], realizes:Dict[UOp, UOp],
                     reduce_for_op:Dict[UOp, UOp], group:Dict[UOp, None], cache:Dict[Tuple[UOp, ShapeTracker], None]) -> None:
  """recursively search the uop for groupable children, realize the UOp if a child can't group"""
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
    if (tr_next_uop:=uval(allbufs[tr_next]).base).op is Ops.REDUCE_AXIS: return group.setdefault(r)
    # can only fuse contiguous
    if len(st_childs:=dedup(unwrap(x.st) for x in tr_next_uop.src if is_scheduled(x.base) and x.base.buf_uop == tr)) > 1: return group.setdefault(r)
    recursive_group(tr_next, st+st_childs[0], r, children, allbufs, realizes, reduce_for_op, group, cache)

def get_isolated_children(r:UOp, reduce_for_op:Dict[UOp, UOp], children:DefaultDict[UOp, Dict[UOp, None]], allbufs:Dict[UOp, UOp],
                           realizes:Dict[UOp, UOp], group:Dict[UOp, None]) -> Dict[UOp, None]:
  rc_parents, cache = deque(group), set()
  while rc_parents:
    if (p:=uval(allbufs[rc_parents.pop()])) in cache: continue
    cache.add(p)
    # max one reduceop per kernel
    if p.op is Ops.REDUCE_AXIS: return {}
    rc_parents.extend(x.base.buf_uop for x in p.src if is_scheduled(x.base) and x.base.buf_uop is not r)
  # search descendants of the reduceop that can cleanly group
  descendants: Dict[UOp, None] = {}
  for tr in group: recursive_group(tr, unwrap(allbufs[tr].st), tr, children, allbufs, realizes, reduce_for_op, descendants, cache={})
  return merge_dicts([group, {} if any(tr in group for tr in descendants) else descendants])

def group_realizes(big_graph:UOp, realizes:Dict[UOp, UOp]) -> Tuple[List[List[UOp]], Set[UOp]]:
  """search the big graph for all the reduceops that need to realize, sometimes group/fuse the reduceop"""
  children: DefaultDict[UOp, Dict[UOp, None]] = defaultdict(dict)
  allbufs: Dict[UOp, UOp] = {}
  double_reduces: Dict[UOp, None] = {}
  assigns: Set[UOp] = set()
  q: List[UOp] = list(big_graph.src)
  seen: Set[UOp] = set()
  while q:
    seen.add(uop:=q.pop())
    allbufs[ubuf:=uop.buf_uop] = uop
    if (op:=uval(uop)).op is Ops.ASSIGN: assigns.add(ubuf)
    for x in op.src:
      if x.base in seen: continue
      if is_scheduled(x.base):
        children[x.base.buf_uop][ubuf] = None
        if FUSE_CONV_BW and op.op is Ops.REDUCE_AXIS and uval(x.base).op is op.op and x.base is not x: double_reduces[ubuf] = None
        q.append(x.base)
  # find all reduces, and pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce (or a contig child)
  reduce_for_op: Dict[UOp, UOp] = {}
  reduce_of_const: List[UOp] = []
  for r, r_uop in allbufs.items():
    if r in realizes or (r_uop:=uval(r_uop)).op is not Ops.REDUCE_AXIS: continue
    group: Dict[UOp, None] = {}
    recursive_group(r, unwrap(r_uop.st), r, children, allbufs, realizes, reduce_for_op, group, cache={})
    # max one reduceop per kernel
    can_chase = all(tr not in reduce_for_op for tr in group)
    # TODO: forced_realize exists because the scheduler is incapable of checking for self-contained DAGs
    forced_realize = r in group
    if not forced_realize and len(group) > 1:
      group = get_isolated_children(r, reduce_for_op, children, allbufs, realizes, group)
    # can only fuse assign if no other assign_target is used in the kernel
    if not forced_realize and any(x in assigns for x in group):
      parents = deque((r, *group))
      while parents and not forced_realize:
        if (p_uop:=allbufs.get(p:=parents.pop())) is None: continue
        if (p_uop:=uval(p_uop)).op is Ops.ASSIGN and p not in group: forced_realize, can_chase = True, False
        if p in realizes: continue
        parents.extend([x.base.src[0] for x in p_uop.src if x.base.op is Ops.VIEW and len(x.base.src) != 0])
    if forced_realize or not group:
      tr = r
      if can_chase:
        # can chase this down to contiguous children
        st = unwrap(r_uop.st)
        while len(children[tr]) == 1:
          tr_next_uop = uval(allbufs[(tr_next:=next(iter(children[tr])))])
          st_childs = dedup([unwrap(x.st) for x in tr_next_uop.src if is_scheduled(x.base) and x.base.buf_uop is tr])
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
  for reduceop in double_reduces:
    top_reduce = uval(allbufs[reduceop]).src[0].base.buf_uop
    if len(children[top_reduce]) == 1: del realizes[top_reduce]
  # maybe fuse arange with its children
  for rbuf in reduce_of_const:
    group = {tr:None for tr,rop in reduce_for_op.items() if rop is rbuf}
    if any(allbufs[tr].src[1].is_contiguous_base for tr in group): continue
    kernel_children = {c for tr in group for c in children[tr] if uval(allbufs[c]).op not in {Ops.COPY, Ops.BUFFER_VIEW}}
    if len(kernel_children) == 0: continue
    for tr in group: del realizes[tr]
  # get output groups
  output_groups: DefaultDict[UOp, List[UOp]] = defaultdict(list)
  for ubuf in realizes: output_groups[reduce_for_op.get(ubuf, ubuf)].append(ubuf)
  return list(output_groups.values()), assigns

# **** Schedule creation and BFS toposort

def realize(ctx:Dict[UOp, UOp], b:UOp, to_store:UOp, base:UOp) -> UOp:
  ctx[b] = UOp.store(b, ShapeTracker.from_shape((st:=unwrap(base.st)).shape).to_uop(), to_store)
  return UOp(Ops.LOAD, base.dtype, (b, st.to_uop()))

def realize_view(ctx:Dict[UOp, UOp], base:UOp, view:UOp, to_store:UOp, b:UOp) -> Optional[UOp]:
  base_shape = unwrap(base.st).shape
  st = unwrap(view.st)
  # fold simple pads
  if len(st.views) == 1 and (m:=st.views[-1].mask) is not None and all_int(base_shape) and resolve(prod(base_shape) >= prod([y-x for x,y in m])):
    return None if can_pad(base) else realize(ctx, b, to_store, base).view(st)
  # early realize before expand
  if resolve(prod(base_shape) < prod(st.shape)): return realize(ctx, b, to_store, base).view(st)
  # otherwise safety check pads
  return None if (all(v.mask is None for v in st.views) or can_pad(base)) else realize(ctx, b, to_store, base).view(st)

do_realize = PatternMatcher([
  # always realize meta ops
  (UPatSrc((Ops.ASSIGN, Ops.CONTIGUOUS, *GroupOp.Meta)), realize),
  # don't realize image to image casts
  (UPatSrc(Ops.CAST, src=(UPat(Ops.LOAD, name="x"),), dtype=dtypes.float).view(name="v"), lambda ctx,x,v,**kwargs: r.src[2].view(v.st)
   if (r:=ctx.get(b:=x.buf_uop)) is not None and r.op is Ops.STORE and isinstance(b.dtype, ImageDType) and r.src[2].op not in GroupOp.Meta else None),
  # realize before expand or unsafe pad ops
  (UPatSrc().view(name="view"), realize_view),
  # realize before COPY or BUFFER_VIEW
  (UPat((Ops.COPY, Ops.BUFFER_VIEW), src=(UPat.any(UPatSrc(), UPatSrc().view(name="view")),), name="root"),
   lambda ctx,root,view=None,**kwargs: root.replace(src=(realize(ctx,**kwargs) if view is None else realize(ctx,**kwargs).view(view.st),)),),
])
break_sched = PatternMatcher([(UPatSrc(), lambda ctx,b,to_store,base: realize(ctx, b, to_store, base) if b in ctx else None),])

@track_rewrites(named=True)
def create_schedule_with_vars(outs:List[LazyBuffer]) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
  if len(outs:=dedup(x.base for x in outs if not x.base.is_realized and x.base.op is not Ops.CONST)) == 0: return [], {}
  for out in outs: out.forced_realize = True
  # create the big graph
  ctx = ScheduleContext()
  cache: Dict[LazyBuffer, UOp] = {}
  big_graph = UOp.sink(*(to_uop(x, ctx, cache) for x in outs))
  # get realizes
  realizes: Dict[UOp, UOp] = {}
  graph_rewrite(big_graph, do_realize, realizes)
  store_groups, assigns = group_realizes(big_graph, realizes)
  # split realizes into small graphs
  graph_rewrite(big_graph, break_sched, realizes)
  sinks = [UOp.sink(*(realizes[u] for u in stores)) for stores in store_groups]
  # preschedule all realizes
  prescheduled: List[ScheduleItem] = []
  for sink in sinks:
    metadata = tuple({mx for x in sink.sparents if (x.op is Ops.STORE or is_scheduled(x)) and (mx:=ctx.ubuf_metadata.get(x.buf_uop))})
    ast, ast_ctx = full_ast_rewrite(sink, ctx.var_vals, assigns)
    prescheduled.append(ScheduleItem(ast, tuple(b for u in ast_ctx.bufs if (b:=ctx.buf_uops[u]).size != 0), metadata, tuple(ast_ctx.assign_preloads)))
  # do BFS
  schedule_targets = {out:si for si in prescheduled for out in si.outputs}
  graph: DefaultDict[ScheduleItem, List[ScheduleItem]] = defaultdict(list)
  in_degree: DefaultDict[ScheduleItem, int] = defaultdict(int)
  for si in prescheduled:
    # realize outputs before a parent is assigned to
    parents_assigns = dedup(xsi for x in si.assign_preloads if (xsi:=schedule_targets.get(ctx.buf_uops[x])) and xsi is not si)
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
