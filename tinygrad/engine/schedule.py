import sys, atexit, functools
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import FrozenSet, Set, Tuple, List, Dict, Optional, DefaultDict, cast
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
  assign_preloads: FrozenSet[UOp]
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

# **** Schedule context and big graph

@dataclass(frozen=True)
class ScheduleContext:
  lazybufs: Dict[UOp, LazyBuffer] = field(default_factory=dict)      # this maps BUFFER uops to Metadata
  var_vals: Dict[Variable, int] = field(default_factory=dict)        # this maps a BIND's DEFINE_VAR to its value
  assigns: Set[UOp] = field(default_factory=set)                     # this holds all the BUFFER uops we ASSIGN to in this schedule
  realizes: Dict[UOp, UOp] = field(default_factory=dict)             # this holds all the BUFFER uops we mutate in this schedule
  allbufs: Dict[UOp, UOp] = field(default_factory=dict)              # this maps BUFFER uops the actual op
  children: DefaultDict[UOp, Dict[UOp, None]] = field(default_factory=lambda: defaultdict(dict))

class UPatSrc(UPat):
  def __init__(self, *args, **kwargs): super().__init__(Ops.VIEW, src=(UPat.var("b"), UPat(*args, **{**kwargs, "name": "to_store"})), name="base")
@functools.lru_cache(None)
def is_scheduled(u:UOp) -> bool: return u.op is Ops.VIEW and len(u.src) == 2

def to_uop(buf:LazyBuffer, ctx:ScheduleContext, buffers:Dict[UOp, Buffer], cache:Dict[LazyBuffer, UOp]) -> UOp:
  if (r:=cache.get(buf)) is not None: return r
  if buf is not buf.base:
    cache[buf] = ret = to_uop(buf.base, ctx, buffers, cache).view(buf.st)
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
  if buf.is_realized:
    buffers[ubuf:=UOp.new_buffer((b:=buf.buffer).device, b.size, b.dtype, num=len(buffers))] = buf.buffer
    op = None
  elif buf.op is Ops.ASSIGN:
    target, new_val = [to_uop(x, ctx, buffers, cache) for x in buf.srcs]
    ctx.assigns.add(ubuf:=target.buf_uop)
    op = UOp(Ops.ASSIGN, dtype, (ubuf, new_val), buf.arg)
  else:
    buffers[ubuf:=UOp.new_buffer((b:=buf.buffer).device, b.size, b.dtype, num=len(buffers))] = buf.buffer
    op = UOp(cast(Ops, buf.op), dtype, tuple(to_uop(x, ctx, buffers, cache) for x in buf.srcs),
             None if buf.op in {Ops.CAST, Ops.BITCAST} else buf.arg)
  cache[buf] = ret = UOp(Ops.VIEW, dtype.base, (ubuf,) if op is None else (ubuf, op.contiguous() if buf.forced_realize else op), buf.st)
  if op is not None:
    ctx.lazybufs[ubuf] = buf
    ctx.allbufs[ubuf] = ret
    for x in op.src:
      if is_scheduled(x.base): ctx.children.setdefault(x.base.buf_uop, {})[ubuf] = None
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
  new_axis = tuple(range(len(st.shape), len(st.shape) + len(r.axis_arg)))
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
  lazybufs: Dict[UOp, LazyBuffer]
  assigns: Set[UOp]
  var_vals: Dict[Variable, int]
  sinked: Dict[UOp, UOp]
  sts: Set[ShapeTracker] = field(default_factory=set)
  bufs: List[UOp] = field(default_factory=list)
  metadata: Set[Metadata] = field(default_factory=set)
  assign_adj: Dict[UOp, List[UOp]] = field(default_factory=dict)

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
  (adj_loads:=ctx.assign_adj.setdefault(b, [])).append(x)
  if not all_same([x.op for x in adj_loads]): raise RuntimeError(f"Detected cycle when fusing {adj_loads}. Can only fuse PRELOAD or LOAD of {b}")
  return x.replace(op=Ops.LOAD)
check_preload = PatternMatcher([(UPat(Ops.PRELOAD, src=(UPat.var("b"), UPat()), name="x"), _append_preload),])

to_si = PatternMatcher([
  (UPat(Ops.VIEW, name="x"), _append_st_vars),
  (UPat(Ops.SINK, src=(UPat.store(UPat.var("b"), UPat(), UPat(GroupOp.Meta, name="x")),)), lambda ctx,b,x: x.replace(src=(b, *x.src))),
])

# ** fusion

def fuse_src(ctx:ScheduleItemContext, b:UOp, to_store:UOp, base:UOp) -> UOp:
  if (lbuf:=ctx.lazybufs.get(b)) is not None and (metadata:=lbuf.metadata) is not None: ctx.metadata.add(metadata)
  return to_store

lazy = PatternMatcher([
  (UPatSrc(), fuse_src),
  (UPat(Ops.BUFFER, name="b").view(name="v"), lambda ctx,b,v: UOp(Ops.PRELOAD if b in ctx.assigns else Ops.LOAD, v.dtype, (b, v.st.to_uop()))),
  (UPat(Ops.CONTIGUOUS, src=(UPat.var("x"),)), lambda ctx,x: x),
])

multioutput = PatternMatcher([(UPat.load(UPat.var("b"), UPat()), lambda ctx,b: ctx.sinked.get(b)),])

append_load = PatternMatcher([(UPat.load(UPat.var("b"), UPat(), name="x"), lambda ctx,b,x: ctx.assign_adj.setdefault(b, []).append(x)
                               if b in ctx.assigns else None)])

def full_ast_rewrite(pre:UOp, ctx:ScheduleContext) -> Tuple[UOp, ScheduleItemContext]:
  si_ctx = ScheduleItemContext(ctx.lazybufs, ctx.assigns, ctx.var_vals, {x.buf_uop:x.src[2] for x in pre.src},
                               metadata={l.metadata for x in pre.src if (l:=ctx.lazybufs.get(x.buf_uop)) is not None and l.metadata is not None})
  # fuse and fold store -> loads
  ops_folding = lazy if len(si_ctx.sinked) == 1 else lazy+multioutput
  sink = graph_rewrite(pre, ops_folding if len(si_ctx.assigns) == 0 else ops_folding+append_load, si_ctx)
  # do movementops
  sink = graph_rewrite(graph_rewrite(sink, view_left), view_right)
  # convert to AST
  sink = graph_rewrite(graph_rewrite(sink, to_si+check_preload if len(si_ctx.assigns) != 0 else to_si, si_ctx), append_bufs, si_ctx)
  # assert buffer count limit
  if (limit:=BUF_LIMIT.get(device:=si_ctx.bufs[0].device)) is not None and len(si_ctx.bufs) >= limit:
    if DEBUG >= 3: print(sink)
    raise RuntimeError(f"Kernel for {si_ctx.metadata} exceeded the {limit} buffer count limit for {device} with {len(si_ctx.bufs)} buffers.")
  # we also allow masked views. if it has a single view and it's equal when you shrink a contig, it's fine
  for ubuf,ops in si_ctx.assign_adj.items():
    if si_ctx.sinked.get(ubuf) is not None and not all((s:=x.st_arg).contiguous or (len(s.views) == 1 and (m:=s.views[0].mask) is not None \
        and ShapeTracker.from_shape(s.shape).shrink(m) == s.shrink(m)) for x in ops):
      raise RuntimeError("self operand of augmented assign must be contiguous.\nhelp: consider using .contiguous():\n"
                         +colored("   - a += a.T\n", "red")+colored("   + a += a.T.contiguous()", "green"))
  if getenv("RUN_PROCESS_REPLAY"): PROCESS_REPLAY_CAPTURE.append(((pre, ctx.assigns), sink))
  return sink, si_ctx

PROCESS_REPLAY_CAPTURE: List[Tuple[Tuple[UOp, Set[UOp]], UOp]] = []
if getenv("RUN_PROCESS_REPLAY"):
  @atexit.register
  def save_process_replay() -> None:
    for x,ret in PROCESS_REPLAY_CAPTURE: diskcache_put("schedule_process_replay", str(x[0].key), (*x, {}, ret))

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

def group_realizes(ctx:ScheduleContext, realizes:Dict[UOp, UOp]) -> List[List[UOp]]:
  """search the big graph for all the reduceops that need to realize, sometimes group/fuse the reduceop"""
  # find all reduces, and pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce (or a contig child)
  reduce_for_op: Dict[UOp, UOp] = {}
  reduce_of_const: List[UOp] = []
  double_reduces: List[UOp] = []
  for r, r_uop in ctx.allbufs.items():
    if (r_uop:=uval(r_uop)).op is not Ops.REDUCE_AXIS: continue
    if FUSE_CONV_BW and uval((x:=r_uop.src[0]).base).op is r_uop.op and x.base is not x: double_reduces.append(r)
    if r in realizes: continue
    group: Dict[UOp, None] = {}
    recursive_group(r, unwrap(r_uop.st), r, ctx.children, ctx.allbufs, realizes, reduce_for_op, group, cache={})
    # max one reduceop per kernel
    can_chase = all(tr not in reduce_for_op for tr in group)
    # TODO: forced_realize exists because the scheduler is incapable of checking for self-contained DAGs
    forced_realize = r in group
    if not forced_realize and len(group) > 1:
      group = get_isolated_children(r, reduce_for_op, ctx.children, ctx.allbufs, realizes, group)
    # can only fuse assign if no other assign_target is used in the kernel
    if not forced_realize and any(x in ctx.assigns for x in group):
      parents = deque((r, *group))
      while parents and not forced_realize:
        if (p_uop:=ctx.allbufs.get(p:=parents.pop())) is None: continue
        if (p_uop:=uval(p_uop)).op is Ops.ASSIGN and p not in group: forced_realize, can_chase = True, False
        if p in realizes: continue
        parents.extend([x.base.src[0] for x in p_uop.src if x.base.op is Ops.VIEW and len(x.base.src) != 0])
    if forced_realize or not group:
      tr = r
      if can_chase:
        # can chase this down to contiguous children
        st = unwrap(r_uop.st)
        while len(ctx.children[tr]) == 1:
          tr_next_uop = uval(ctx.allbufs[(tr_next:=next(iter(ctx.children[tr])))])
          st_childs = dedup([unwrap(x.st) for x in tr_next_uop.src if is_scheduled(x.base) and x.base.buf_uop is tr])
          if len(st_childs) > 1: break
          if st.size != st_childs[0].size: break
          st = st + st_childs[0]
          if not st.contiguous or tr_next_uop.op is Ops.REDUCE_AXIS: break
          tr = tr_next
        # don't cast to higher size before store (tr cannot be realized if forced_realize)
        if (tr_uop:=uval(ctx.allbufs[tr])).op is Ops.CAST and tr_uop.dtype.base.itemsize > tr_uop.src[0].dtype.base.itemsize:
          tr = tr_uop.src[0].base.buf_uop
      group = {tr: None}
      realizes[tr] = tr
    reduce_for_op.update((tr, r) for tr in group)
    if FUSE_ARANGE and r_uop.arg[0] is Ops.ADD and uval(r_uop.src[0].base).op is Ops.CONST: reduce_of_const.append(r)
  # fuse double reduces with no other child
  for reduceop in double_reduces:
    top_reduce = uval(ctx.allbufs[reduceop]).src[0].base.buf_uop
    if len(ctx.children[top_reduce]) == 1: del realizes[top_reduce]
  # maybe fuse arange with its children
  for rbuf in reduce_of_const:
    group = {tr:None for tr,rop in reduce_for_op.items() if rop is rbuf}
    if any(ctx.allbufs[tr].src[1].is_contiguous_base for tr in group): continue
    kernel_children = {c for tr in group for c in ctx.children[tr] if uval(ctx.allbufs[c]).op not in {Ops.COPY, Ops.BUFFER_VIEW}}
    if len(kernel_children) == 0: continue
    for tr in group: del realizes[tr]
  # group BUFFER uops into kernels
  output_groups: DefaultDict[UOp, List[UOp]] = defaultdict(list)
  for ubuf in realizes: output_groups[reduce_for_op.get(ubuf, ubuf)].append(ubuf)
  return list(output_groups.values())

# **** Schedule creation and BFS toposort

def realize(ctx:Dict[UOp, UOp], b:UOp, to_store:UOp, base:UOp) -> UOp:
  ctx[b] = UOp.store(b, ShapeTracker.from_shape((st:=unwrap(base.st)).shape).to_uop(), to_store)
  return UOp(Ops.LOAD, base.dtype, (b, st.to_uop()))

def realize_view(ctx:Dict[UOp, UOp], base:UOp, view:UOp, to_store:UOp, b:UOp) -> Optional[UOp]:
  if to_store.op in {Ops.CONST, Ops.BIND}: return None
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
  (UPatSrc({Ops.ASSIGN, Ops.CONTIGUOUS, *GroupOp.Meta}), realize),
  # don't realize image to image casts
  (UPatSrc(Ops.CAST, src=(UPat(Ops.LOAD, name="x"),), dtype=dtypes.float).view(name="v"), lambda ctx,x,v,**kwargs: r.src[2].view(v.st)
   if (r:=ctx.get(b:=x.buf_uop)) is not None and r.op is Ops.STORE and isinstance(b.dtype, ImageDType) and r.src[2].op not in GroupOp.Meta else None),
  # realize before expand or unsafe pad ops
  (UPatSrc().view(name="view"), realize_view),
  # realize before COPY or BUFFER_VIEW
  (UPat((Ops.COPY, Ops.BUFFER_VIEW), src=(UPat.any(UPatSrc(), UPatSrc().view(name="view")),), name="root"),
   lambda ctx,root,view=None,**kwargs: root.replace(src=(realize(ctx,**kwargs) if view is None else realize(ctx,**kwargs).view(view.st),)),),
])

def generate_valid(ctx:ScheduleContext, b:UOp, to_store:UOp, base:UOp) -> UOp:
  if isinstance((val:=to_store.arg), UOp): ctx.var_vals.update([val.unbind()])
  return UOp.const_with_shape(base.dtype, val, unwrap(base.st).shape)

break_sched = PatternMatcher([
  # consts are always fused and generated
  (UPatSrc({Ops.CONST, Ops.BIND}), generate_valid),
  # everything else is a VIEW of BUFFER that either realizes or fuses
  (UPatSrc(), lambda ctx,b,to_store,base: realize(ctx.realizes, b, to_store, base) if b in ctx.realizes else None),
])

@track_rewrites(named=True)
def create_schedule_with_vars(outs:List[LazyBuffer]) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
  if len(outs:=dedup(x.base for x in outs if x.base.realized is None and x.base.op is not Ops.CONST)) == 0: return [], {}
  for out in outs: out.forced_realize = True
  # create the big graph
  ctx = ScheduleContext()
  cache: Dict[LazyBuffer, UOp] = {}
  buffers: Dict[UOp, Buffer] = {}
  big_graph = UOp.sink(*(to_uop(x, ctx, buffers, cache) for x in outs))
  # get realizes
  graph_rewrite(big_graph, do_realize, ctx.realizes)
  store_groups = group_realizes(ctx, ctx.realizes)
  # split realizes into small graphs
  graph_rewrite(big_graph, break_sched, ctx)
  # preschedule all realizes
  prescheduled: List[ScheduleItem] = []
  for store_uops in store_groups:
    ast, ast_ctx = full_ast_rewrite(UOp.sink(*(ctx.realizes[u] for u in store_uops)), ctx)
    prescheduled.append(ScheduleItem(ast, tuple(buffers[u] for u in ast_ctx.bufs if u.size != 0), tuple(ast_ctx.metadata),
                                     frozenset(ubuf for ubuf,ops in ast_ctx.assign_adj.items() if any(x.op is Ops.PRELOAD for x in ops))))
    for u in ast_ctx.sinked: del ast_ctx.lazybufs[u].srcs  # can only schedule once
  # do BFS
  schedule_targets = {out:si for si in prescheduled for out in si.outputs}
  graph: DefaultDict[ScheduleItem, List[ScheduleItem]] = defaultdict(list)
  in_degree: DefaultDict[ScheduleItem, int] = defaultdict(int)
  for si in prescheduled:
    # realize outputs before a parent is assigned to
    parents_assigns = dedup(xsi for x in si.assign_preloads if (xsi:=schedule_targets.get(buffers[x])) and xsi is not si)
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
