import sys, functools, atexit, pickle
from collections import defaultdict, deque
from dataclasses import dataclass, field
from tinygrad.ops import UOp, Variable, Ops, GroupOp, PatternMatcher, UPat, graph_rewrite, graph_rewrite_map, track_rewrites, buffers
from tinygrad.ops import can_pad, identity_element, resolve, symbolic_simple, view_left, merge_views
from tinygrad.helpers import Context, ContextVar, Metadata, all_int, all_same, colored, diskcache_put, prod, dedup, getenv, unwrap, flatten
from tinygrad.helpers import FUSE_CONV_BW, FUSE_ARANGE, DEBUG, CAPTURE_PROCESS_REPLAY
from tinygrad.dtype import ImageDType
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, strides_for_shape
from tinygrad.device import Buffer
from tinygrad.spec import type_verify, kernel_spec

# creation can recurse a lot
sys.setrecursionlimit(10000)

# **** schedule simplifier

def simplify_reduceop(reduce:UOp, x:UOp) -> UOp|None:
  if not all_int(x.shape): return None
  # remove reduce on unmasked const
  prshape = prod(unwrap(x.st).shape[i] for i in reduce.arg[1])
  ret = x.const_arg
  match reduce.arg[0]:
    case Ops.ADD: ret *= prshape
    case Ops.MUL: ret **= prshape
    case Ops.MAX: pass # NOTE: Ops.MAX is passthrough
    case _: return None
  return reduce.const_like(ret)

def found_contiguous(ctx:dict[UOp, UOp], contig:UOp, src:UOp):
  if (sti:=unwrap(src.st).invert(src.base.shape)) is not None: ctx[src.base] = contig.view(sti)
def replace_contiguous(ctx:dict[UOp, UOp], alu:UOp):
  new_src = list(alu.src)
  for i,s in enumerate(alu.src):
    if (replace_src:=ctx.get(s, None)) is not None: new_src[i] = replace_src
  if tuple(new_src) != alu.src: return alu.replace(src=tuple(new_src))

sym = symbolic_simple+PatternMatcher([
  # UOp with size 0 is zero
  (UPat(GroupOp.All-{Ops.SINK}, name="root"), lambda root: root.const_like(0) if root.base.st is not None and root.size == 0 \
    and not (root.base.op is Ops.CONST and root.base.arg == 0) else None),
  # DETACH and CONTIGUOUS_BACKWARD are NOOPs here
  (UPat((Ops.DETACH, Ops.CONTIGUOUS_BACKWARD), name="x"), lambda x: x.src[0]),
  # reduce of size 0 is the identity element
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)),
   lambda reduce,x: reduce.const_like(identity_element(reduce.arg[0], reduce.dtype)) if x.size == 0 and reduce.size != 0 else None),
  # reduce of const is collapsed (TODO: make this a generic rule for stride0)
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.cvar("x"),)), simplify_reduceop),
  # COPY(CONST) creates a new CONST on the destination device
  (UPat(Ops.COPY, name="root", src=(UPat(), UPat.cvar("x"),)), lambda root,x: root.const_like(x.const_arg)),
  # no COPY to same device, except clone (arg is True)
  (UPat(Ops.COPY, src=(UPat(), UPat.var("copyin")), name="copy"),
   lambda copyin,copy: copyin if copyin.device == copy.device and copy.arg is not True else None),
  # remove cast to image when it's already a contiguous image
  (UPat(Ops.VIEW, name="vm1", src=(UPat(Ops.CAST, name="cast", src=(UPat(Ops.VIEW, name="vm2", src=(UPat(Ops.CONTIGUOUS, name="base"))))),)),
   lambda cast,base,vm1,vm2: base.view(vm2.st+vm1.st) if isinstance(cast.dtype, ImageDType) and isinstance(base.dtype, ImageDType) else None),
  # remove contiguous if we can just view the buffer
  (UPat(Ops.CONTIGUOUS, name="root", src=(UPat(Ops.VIEW, name="view", src=(UPat(Ops.BUFFER, name="buf"),)),)),
   lambda root,view,buf: view if view.st.contiguous and view.size == buf.size else None),
  # contiguous/buffer/copy is already contiguous
  (UPat(Ops.CONTIGUOUS, name="root", src=(UPat((Ops.CONTIGUOUS, Ops.BUFFER, Ops.COPY)),)), lambda root: root.src[0]),
  # support for using a contiguous permuted view instead of the parent view if one exists
  (UPat(Ops.CONTIGUOUS, name="contig", src=(UPat(Ops.VIEW, name="src"),)), found_contiguous),
  (UPat(GroupOp.ALU, name="alu"), replace_contiguous),
  # substitute BITCAST/CONTIGUOUS with BUFFER_VIEW on DISK
  (UPat((Ops.BITCAST, Ops.CONTIGUOUS), name="root"),
  lambda root: root.replace(op=Ops.BUFFER_VIEW) if isinstance(root.device, str) and root.device.startswith("DISK") else None),
  # remove CONST/BIND/BUFFER/VIEW from SINK
  (UPat(Ops.SINK, name="root"),
    lambda root: UOp(Ops.SINK, root.dtype, new_src, root.arg)
      if (new_src:=tuple(x.base for x in root.src if not x.is_realized and x.base.op not in {Ops.CONST, Ops.BIND})) != root.src else None),
])

remove_movement_ops = merge_views+PatternMatcher([
  # NOTE: movement ops are always applied to base
  (UPat(GroupOp.Movement, name="mov", src=(UPat.var("x"),)), lambda x,mov: x.view(unwrap(mov.st))),
  # some masked views can collapse to 0, VIEW(x) -> CONST(VIEW)
  (UPat(Ops.VIEW, name="view"),
   lambda view: view.const_like(0) if (vm:=view.st.views[-1].mask) is not None and any((x[1]-x[0]) == 0 for x in vm) else None),
])

# **** UOp realization

@dataclass(frozen=True)
class ScheduleContext:
  ops_metadata: dict[UOp, Metadata]                                  # this maps uops in the schedule to the tensor metadata
  assigns: dict[UOp, None] = field(default_factory=dict)             # this holds all the BUFFER uops we ASSIGN to in this schedule
  realizes: dict[UOp, UOp] = field(default_factory=dict)             # this holds all the BUFFER uops we mutate in this schedule
  allbufs: dict[UOp, UOp] = field(default_factory=dict)              # this maps BUFFER uops the actual op
  var_vals: dict[Variable, int] = field(default_factory=dict)
  children: defaultdict[UOp, dict[UOp, None]] = field(default_factory=lambda: defaultdict(dict))
  preloads: defaultdict[Buffer, dict[UOp, None]] = field(default_factory=lambda: defaultdict(dict))

# wrap tensor uops around a VIEW(BUFFER, <uop>)
# this BUFFER preserves a link back to the uop on the tensor after the scheduler rewrites it.
def add_buffers(buf:UOp, buffer_map:dict[UOp, UOp], cache:dict[UOp, UOp]) -> UOp:
  if (r:=cache.get(buf)) is not None: return r
  # SINK is passthrough
  if buf.op is Ops.SINK: return buf.replace(src=tuple(add_buffers(x, buffer_map, cache) for x in buf.src))
  # skip creating buffers for CONST/BIND/DEVICE/BUFFER
  if buf.base.op in {Ops.CONST, Ops.BIND, Ops.DEVICE}: return buf
  if buf.base.op is Ops.BUFFER: return buf.view(unwrap(buf.st))
  # VIEW is passthrough
  if buf is not buf.base:
    cache[buf] = ret = add_buffers(buf.base, buffer_map, cache).view(unwrap(buf.st))
    return ret
  # make things that can't be images not images
  dtype = buf.dtype
  if isinstance(dtype, ImageDType) and (prod(buf.shape)!=prod(dtype.shape) or not any(buf.shape[x]%4==0 for x in unwrap(buf.st).unit_stride_axes())):
    if DEBUG >= 2: print(f"forcing image {dtype} with shape {buf.shape} to {dtype.base}")
    dtype = buf.dtype.base
  # ASSIGN already has a target buffer, otherwise we create a new one
  assert isinstance(buf.device, str), f"buf device is str, not {buf.device}"
  buf_uop = buf.buf_uop if buf.op is Ops.ASSIGN else UOp.new_buffer(buf.device, buf.size, dtype)
  op = buf.replace(dtype=dtype, src=tuple(add_buffers(x, buffer_map, cache) for x in buf.src))
  # track the buffer uop for the simplified uop
  buffer_map[buf] = buf_uop
  if op.op is Ops.BUFFER_VIEW: buffers[buf_uop] = (x:=op.src[0]).buf_uop.buffer.view(op.size, op.dtype, unwrap(x.st).views[0].offset*x.dtype.itemsize)
  # (early) bufferize
  cache[buf] = ret = UOp(Ops.VIEW, dtype.base, (buf_uop, op), buf.st)
  return ret

class UPatScheduled(UPat):
  def __init__(self, *args, **kwargs):
    super().__init__(Ops.VIEW, name="base", src=(UPat(Ops.BUFFER, name="b"), UPat(*args, **{"name":"to_store",**kwargs})))

def realize(ctx:ScheduleContext, b:UOp, to_store:UOp, **kwargs) -> None: ctx.realizes[b] = to_store

def realize_before_view(ctx:ScheduleContext, view:UOp, src:UOp, b:UOp, **kwargs) -> None:
  st = unwrap(view.st)
  # fold simple pads
  if len(st.views) == 1 and (m:=st.views[-1].mask) is not None and all_int(src.shape) and resolve(prod(src.shape) >= prod([y-x for x,y in m])):
    return None if can_pad(src, ctx.realizes, dict()) else realize(ctx, b, src)
  # early realize before expand
  if resolve(prod(src.shape) < prod(st.shape)) and not getenv("DONT_REALIZE_EXPAND"): return realize(ctx, b, src)
  # otherwise safety check pads
  return None if (all(v.mask is None for v in st.views) or can_pad(src, ctx.realizes, dict())) else realize(ctx, b, src)

do_realize = PatternMatcher([
  # always realize SINK parents
  (UPat(Ops.SINK, name="sink"), lambda ctx,sink: ctx.realizes.update((x.buf_uop, x) for x in sink.src)),
  # always realize ASSIGN/CONTIGUOUS/COPY/BUFFER_VIEW
  (UPatScheduled({Ops.ASSIGN, Ops.CONTIGUOUS, Ops.COPY, Ops.BUFFER_VIEW}), realize),
  # realize before expand or unsafe pad ops
  (UPat(Ops.VIEW, name="view", src=(UPatScheduled(name="src"),)), realize_before_view),
  # realize before COPY
  (UPat(Ops.COPY, src=(UPat(), UPatScheduled())), realize),
])

def append_uop(ctx:ScheduleContext, view:UOp, buf_uop:UOp) -> None:
  ctx.allbufs[buf_uop] = view
  if (op:=uval(view)).op is Ops.ASSIGN: ctx.assigns[buf_uop] = None
  for x in op.base.src:
    if is_scheduled(x.base): ctx.children.setdefault(x.base.buf_uop, {})[buf_uop] = None
create_ctx = PatternMatcher([(UPat(Ops.VIEW, name="view", src=(UPat(Ops.BUFFER, name="buf_uop"), UPat())), append_uop)])

def is_scheduled(u:UOp) -> bool: return u.op is Ops.VIEW and len(u.src) == 2 and u.src[0].op is Ops.BUFFER
def uval(u:UOp) -> UOp:
  assert is_scheduled(u), f"must be a scheduled op {u}"
  return u.src[1]

def recursive_group(tr:UOp, st:ShapeTracker, r:UOp, children:defaultdict[UOp, dict[UOp, None]], allbufs:dict[UOp, UOp], realizes:dict[UOp, UOp],
                     reduce_for_op:dict[UOp, UOp], group:dict[UOp, None], cache:dict[tuple[UOp, ShapeTracker], None]) -> None:
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

def group_realizes(sink:UOp, ctx:ScheduleContext) -> dict[UOp, UOp]:
  # start by adding uops that always realize
  sink = graph_rewrite(sink, do_realize+create_ctx, ctx)
  # find all reduces, and pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce (or a contig child)
  reduce_for_op: dict[UOp, UOp] = {}
  double_reduces: list[UOp] = []
  for r, r_uop in ctx.allbufs.items():
    if (r_uop:=uval(r_uop)).op is not Ops.REDUCE_AXIS: continue
    if FUSE_CONV_BW and is_scheduled((x:=r_uop.src[0]).base) and uval(x.base).op is r_uop.op and x.base is not x: double_reduces.append(r)
    if r in ctx.realizes: continue
    group: dict[UOp, None] = {}
    recursive_group(r, unwrap(r_uop.st), r, ctx.children, ctx.allbufs, ctx.realizes, reduce_for_op, group, cache={})
    # max one reduceop per kernel
    can_chase = all(tr not in reduce_for_op for tr in group)
    # TODO: forced_realize exists because the scheduler is incapable of checking for self-contained DAGs
    forced_realize = r in group
    # can only have one output
    if not forced_realize and len(group) > 1: forced_realize = True
    # can only fuse assign if no other assign_target is used in the kernel
    if not forced_realize and any(x in ctx.assigns for x in group):
      parents = deque((r, *group))
      while parents and not forced_realize:
        if (p_uop:=ctx.allbufs.get(p:=parents.pop())) is None: continue
        if (p_uop:=uval(p_uop)).op is Ops.ASSIGN and p not in group: forced_realize, can_chase = True, False
        if p in ctx.realizes: continue
        parents.extend([x.base.buf_uop for x in p_uop.src if x.base.is_realized or (x.base.op is Ops.VIEW and len(x.base.src) != 0)])
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
      ctx.realizes[tr] = tr
    reduce_for_op.update((tr, r) for tr in group)
    if FUSE_ARANGE and r_uop.arg[0] is Ops.ADD and r_uop.src[0].base.op is Ops.CONST:
      # maybe fuse arange with its children
      if len(flatten(ctx.children[tr] for tr in group)) != 0:
        for tr in group: del ctx.realizes[tr]
  # fuse double reduces with no other child
  for reduceop in double_reduces:
    top_reduce = uval(ctx.allbufs[reduceop]).src[0].base.buf_uop
    if len(ctx.children[top_reduce]) == 1: del ctx.realizes[top_reduce]
  return ctx.realizes

# break the SINK into stores

def load_realized(ctx:ScheduleContext, b:UOp, st:UOp):
  # NOTE: if we're assigning to the BUFFER too, PRELOAD tells toposort to place this load before the ASSIGN
  return UOp(Ops.PRELOAD if b in ctx.assigns else Ops.LOAD, b.dtype.base, (b, unwrap(st.st).to_uop()))

def store_or_fuse(ctx:ScheduleContext, b:UOp, x:UOp, st:UOp):
  if (m:=ctx.ops_metadata.get(b)) is not None: ctx.ops_metadata[x] = m
  if b not in ctx.realizes: return x # collapse BUFFER
  ctx.realizes[b] = UOp.store(b, ShapeTracker.from_shape(st.shape).to_uop(), x)
  return UOp(Ops.LOAD, x.dtype, (b, unwrap(st.st).to_uop()))

break_sched = PatternMatcher([
  # VIEW of BUFFER either becomes a LOAD/STORE or we fuse it
  (UPat(Ops.VIEW, name="st", src=(UPat(Ops.BUFFER, name="b"),)), load_realized),
  (UPat(Ops.VIEW, name="st", src=(UPat(Ops.BUFFER, name="b"), UPat.var("x"))), store_or_fuse),
])

# **** convert Kernel to a ScheduleItem (for legacy reasons)

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...]
  @property
  def outputs(self) -> tuple[Buffer, ...]:
    """Read/write or write only buffers in the schedule."""
    return tuple(b for i,b in enumerate(self.bufs) if i in self.output_idxs)
  @property
  def inputs(self) -> tuple[Buffer, ...]:
    """Read only buffers in the schedule."""
    return tuple(b for i,b in enumerate(self.bufs) if i not in self.output_idxs)
  @functools.cached_property
  def output_idxs(self) -> tuple[int, ...]: return tuple(x.src[0].arg for x in self.ast.src) if self.ast.op is Ops.SINK else (0,)

def kernel_to_si(k:UOp) -> ScheduleItem:
  assert k.op is Ops.KERNEL, f"must be KERNEL {k}"
  return ScheduleItem(k.arg.ast, tuple(u.buf_uop.buffer for u in k.src), k.arg.metadata)

# **** Kernel creation

@dataclass(frozen=True)
class Kernel:
  ast: UOp
  metadata: tuple[Metadata, ...]
  def __repr__(self): return f"<Kernel {len(list(self.ast.toposort))} {self.ast.op} {self.metadata}>"

@dataclass(frozen=True)
class KernelContext:
  var_vals: dict[Variable, int]
  bufs: list[UOp] = field(default_factory=list)

def apply_swizzle(u:UOp) -> UOp:
  with Context(TRACK_MATCH_STATS=0): return graph_rewrite(u, view_left)

def swizzle_r(r:UOp, src:UOp, st:ShapeTracker) -> UOp:
  input_st = ShapeTracker.from_shape(unwrap(src.st).shape)
  tmp = input_st.permute(tuple(i for i in range(len(input_st.shape)) if i not in r.axis_arg)+r.axis_arg)
  prshape = prod(rshape:=tmp.shape[-len(r.axis_arg):])
  strides = strides_for_shape(rshape)
  nv = [View.create(v.shape+rshape, tuple(x*prshape for x in v.strides)+strides,
                    v.offset*prshape, v.mask+tuple((0,s) for s in rshape) if v.mask is not None else None) for v in st.views]
  # update input_st and axis
  new_input_st = tmp + ShapeTracker(tuple(nv))
  new_axis = tuple(range(len(st.shape), len(st.shape) + len(r.axis_arg)))
  return apply_swizzle(src.view(new_input_st)).r(r.arg[0], new_axis).view(ShapeTracker.from_shape(st.shape))

def reduceop_view_right(r:UOp, v:UOp, src:UOp) -> UOp:
  if not (swizzle_st:=unwrap(v.st)).contiguous or v.size != src.size: raise AssertionError(f"can't push {v} down through {src}")
  output_shape = swizzle_st.reduce(r.axis_arg)
  return src.r(r.arg[0], tuple(i for i,(s,u) in enumerate(zip(src.shape, output_shape)) if s != u)).view(ShapeTracker.from_shape(output_shape))

def elementwise_view_right(root:UOp) -> UOp|None:
  if len(swizzles:=[x for x in root.src if x.base is not x]) == 0: return None
  assert all(x.base.st is not None for x in swizzles), f"found shapeless VIEW src in {root}"
  assert all_same([x.base.size for x in swizzles]), f"swizzle inputs must have the same size {swizzles}"
  # push the swizzle from src to root
  output_swizzle = swizzles[0]
  new_input_st = ShapeTracker.from_shape(output_swizzle.base.shape)
  ret = root.replace(src=tuple(x if x.st is None else x.base if x in swizzles else apply_swizzle(x.view(new_input_st)) for x in root.src))
  return ret.view(ShapeTracker.from_shape(output_swizzle.shape))

def merge_double_reduce(root:UOp, first_reduce:UOp) -> UOp:
  assert root.arg[0] == first_reduce.arg[0], "can't merge reduceops with different alu"
  assert not any(x.op is Ops.REDUCE_AXIS for x in first_reduce.src[0].toposort), "can't merge more than two reduceops at a time"
  return first_reduce.replace(arg=(first_reduce.arg[0], root.axis_arg+first_reduce.axis_arg))

# push VIEW to children
view_right = merge_views+PatternMatcher([
  # STORE(.., ASSIGN(VIEW(BUFFER), new_val)) -> VIEW(STORE(.., new_val))
  (UPat(Ops.STORE, src=(UPat.var("b"), UPat.var("st"), UPat.assign(UPat.var("target"), UPat.var("val")))),
   lambda b,target,st,val: apply_swizzle(UOp.store(b, st, val).view(target.st))),
  # STORE is the last child, so we just merge the ShapeTrackers and store the base
  (UPat(Ops.STORE, src=(UPat.var("b"), UPat.var("st"), UPat(Ops.VIEW, src=(UPat.var("val"),)))), lambda b,st,val: UOp.store(b, st.view(val.st), val)),
  # REDUCE(src.view(contiguous=False)) -> REDUCE(src.view(contiguous=True)).view()
  (UPat(Ops.REDUCE_AXIS, src=(UPat.var("src"),), name="r").view(name="v"), lambda v,r,src: None if v.st.contiguous else swizzle_r(r, src, v.st)),
  # REDUCE(src.view()) -> REDUCE(src).view()
  (UPat(Ops.REDUCE_AXIS, src=(UPat.var("src").view(name="v"),), name="r"), reduceop_view_right),
  # ALU(src.view()) -> ALU(src).view()
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.ASSIGN, Ops.CONTIGUOUS, Ops.STORE), name="root"), elementwise_view_right),
  # double reduce op collapses to a single reduce op
  (UPat(Ops.REDUCE_AXIS, src=(UPat(Ops.REDUCE_AXIS, name="first_reduce"),), name="root"), merge_double_reduce),
])

def _append_st_vars(ctx:KernelContext, x:UOp) -> UOp|None:
  st = unwrap(x.st).simplify()
  if any(x.op is Ops.BIND for x in st.vars()):
    st, var_vals = st.unbind()
    ctx.var_vals.update(var_vals)
  return st.to_uop() if st != x.st else None

def _append_buf(ctx:KernelContext, x:UOp) -> UOp:
  ctx.bufs.append(x)
  return UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(size=x.size), (), len(ctx.bufs)-1)

def check_load_st(glbl:UOp, view:UOp):
  if glbl.arg != 0 or (st:=unwrap(view.st)).contiguous: return
  # if it has a single view and it becomes contiguous when you shrink expanded axes, it's fine
  if len(st.views) == 1 and st.shrink(tuple((0,1) if st == 0 else (0,s) for s,st in zip(st.shape, st.views[0].strides))).contiguous: return
  # if it has a single view and it's equal when you shrink a contig, it's fine
  if len(st.views) == 1 and (mask:=st.views[0].mask) is not None and ShapeTracker.from_shape(st.shape).shrink(mask) == st.shrink(mask): return
  # otherwise, it's not fine
  raise RuntimeError("self operand of augmented assign must be contiguous.\nhelp: consider using .contiguous():\n"
                     +colored("   - a += a.T\n", "red")+colored("   + a += a.T.contiguous()", "green"))

to_si = PatternMatcher([
  # BUFFER -> DEFINE_GLOBAL
  (UPat(Ops.BUFFER, name="x"), _append_buf),
  # simplify and unbind the final VIEWs
  (UPat(Ops.VIEW, name="x"), _append_st_vars),
  # don't need SINK on COPY or BUFFER_VIEW
  (UPat(Ops.SINK, src=(UPat.store(UPat.var("b"), UPat(), UPat((Ops.COPY, Ops.BUFFER_VIEW), name="x")),)), lambda b,x: x.replace(src=(b, *x.src))),
  # don't need contiguous or assign anymore
  (UPat(Ops.CONTIGUOUS, src=(UPat.var("x"),)), lambda x: x),
  (UPat(Ops.ASSIGN, src=(UPat(), UPat.var("x"),)), lambda x: x),
  # don't need DEVICE anymore
  (UPat(Ops.VIEW, name="view", src=(UPat(Ops.DEVICE),)), lambda view: view.replace(src=())),
  # PRELOAD becomes LOAD
  (UPat(Ops.PRELOAD, name="root"), lambda root:root.replace(op=Ops.LOAD)),
  # once images are loaded they become the base dtype
  (UPat(GroupOp.All-{Ops.DEFINE_GLOBAL}, name="x"), lambda x: x.replace(dtype=x.dtype.base) if isinstance(x.dtype, ImageDType) else None),
  # if this kernel also assigns to the loaded buffer, ensure we can index it correctly
  (UPat(Ops.LOAD, src=(UPat.var("glbl"), UPat.var("view"))), check_load_st),
])

def unbind_variable(ctx:dict[Variable, int], bind:UOp, var:UOp, val:UOp):
  ctx[var.replace(src=())] = val.arg
  return var
unbind_vars = PatternMatcher([(UPat(Ops.BIND, name="bind", src=(UPat.var("var"), UPat.cvar("val"))), unbind_variable),])

def schedule_uop(pre:UOp, ctx:ScheduleContext) -> UOp:
  # unbind_vars + push views to edges
  sink = graph_rewrite(graph_rewrite(pre, unbind_vars+view_left, ctx=ctx.var_vals), view_right)
  # remove extra uops from SINK + substitue BUFFER with DEFINE_GLOBAL
  ast = graph_rewrite(sink, to_si, si_ctx:=KernelContext(ctx.var_vals))
  # deal with ASSIGN
  if len(ctx.assigns) != 0:
    assign_preloads = ctx.preloads[si_ctx.bufs[0].buffer]
    for x in list(sink.toposort)[::-1]:
      # we only allow a kernel to depend on either the before ASSIGN or after ASSIGN version of a BUFFER
      if x.op is Ops.LOAD and x.buf_uop in assign_preloads: raise RuntimeError("cycle detected in graph")
      # PRELOAD tells the toposort this kernel should run before ASSIGN
      if x.op is Ops.PRELOAD: assign_preloads[x.buf_uop] = None
  # NOTE: we only add the metadata for fused tensors
  metadata = tuple(dedup(m for x in pre.toposort if x.op is not Ops.BUFFER and (m:=ctx.ops_metadata.get(x)) is not None))
  return UOp(Ops.KERNEL, src=tuple(si_ctx.bufs), arg=Kernel(ast, metadata))

PROCESS_REPLAY_CAPTURE:dict[str, bytes] = {}
if CAPTURE_PROCESS_REPLAY:
  @atexit.register
  def save_process_replay():
    for k,v in PROCESS_REPLAY_CAPTURE.items(): diskcache_put("schedule_process_replay", k, v, prepickled=True)

create_kernels = PatternMatcher([
  (UPat(Ops.SINK, name="x"), lambda ctx,x: x.replace(src=tuple(schedule_uop(s.sink(), ctx) for s in x.src))
    if any(s.op is not Ops.KERNEL for s in x.src) else None),
])

# **** schedule creation and toposort

@track_rewrites(named=True)
def create_schedule_with_vars(big_sink:UOp) -> tuple[list[ScheduleItem], dict[Variable, int], dict[UOp, UOp]]:
  tensor_map = graph_rewrite_map(big_sink, remove_movement_ops+sym, ctx={})
  # tensors can become an existing buffer or simplify to a const, no ScheduleItem needed
  becomes_map: dict[UOp, UOp] = {}
  for k,v in tensor_map.items():
    # NOOP
    if k.base is v.base: continue
    # NOTE: only the base tensors get a BUFFER UOp
    if v.is_realized and k is k.base: becomes_map[k] = v.view(unwrap(k.st))
    # otherwise if it simplified to a CONST the UOp just becomes that CONST
    elif v.op is Ops.CONST and all_int(v.shape): becomes_map[k] = v

  # we group the rest of UOps into ScheduleItems
  buffer_map: dict[UOp, UOp] = {}
  sink = add_buffers(tensor_map[big_sink], buffer_map, cache={})
  # get realizes
  buf_tensors: dict[UOp, list[UOp]] = {}
  ops_metadata: dict[UOp, Metadata] = {}
  for k,v in tensor_map.items():
    if (b:=buffer_map.get(v)) is not None:
      buf_tensors.setdefault(b, []).append(k)
      ops_metadata[b] = k.metadata
  realize_map = group_realizes(sink, ctx:=ScheduleContext(ops_metadata))
  if len(realize_map) == 0: return [], {}, becomes_map

  # map buffers to realized tensors
  for buf_uop in realize_map:
    for tensor_uop in buf_tensors[buf_uop]: becomes_map[tensor_uop] = buf_uop.view(unwrap(tensor_uop.st))
    buf_uop.buffer.ref(1)

  # create kernels, TODO: this should use the SINK from tensor_map
  graph_rewrite(sink, break_sched, ctx)
  sched_sink = graph_rewrite(UOp.sink(*realize_map.values()), create_kernels, ctx)
  type_verify(list(sched_sink.toposort), kernel_spec)

  # TODO: this should be the break between the "grouper" and the "linearizer"
  # here, there should just be one sink UOp with BUFFER/KERNEL/COPY/ASSIGN (assign is the parent if you want the buffer post assign)
  # call into `def linearize_schedule(sched_sink:UOp) -> list[ScheduleItem]`

  # convert kernels to ScheduleItem
  prescheduled = [kernel_to_si(k) for k in sched_sink.src]
  # add ScheduleItem children
  # TODO: this should construct the graph directly from the sched_sink
  schedule_targets = {out:si for si in prescheduled for out in si.outputs}
  graph: defaultdict[ScheduleItem, list[ScheduleItem]] = defaultdict(list)
  in_degree: defaultdict[ScheduleItem, int] = defaultdict(int)
  for si in prescheduled:
    # realize outputs before a parent is assigned to
    parents_assigns = dedup(xsi for x in ctx.preloads[si.bufs[0]] if (xsi:=schedule_targets.get(x.buffer)) and xsi is not si)
    for assign in parents_assigns:
      graph[si].append(assign)
      in_degree[assign] += 1
    # realize outputs after all parents are realized
    scheduled_parents = dedup(xsi for x in si.inputs if (xsi:=schedule_targets.get(x)) is not None and xsi not in parents_assigns)
    for x in scheduled_parents:
      graph[x].append(si)
      in_degree[si] += 1

  # do BFS
  queue = deque(si for si in prescheduled if in_degree[si] == 0)
  schedule: list[ScheduleItem] = []
  while queue:
    schedule.append(si:=queue.popleft())
    for x in graph[si]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)
  # confirm everything was scheduled correctly
  if len(schedule) != (groups:=len(prescheduled)): raise RuntimeError(f"cycle detected in graph, grouped {groups} but only scheduled {len(schedule)}")
  if DEBUG >= 1 and len(schedule) >= 10: print(f"scheduled {len(schedule)} kernels")
  # capture process replay
  if CAPTURE_PROCESS_REPLAY:
    with Context(PICKLE_BUFFERS=0): PROCESS_REPLAY_CAPTURE[str(big_sink.key)] = pickle.dumps((big_sink, ContextVar._cache, [x.ast for x in schedule]))
  return schedule, ctx.var_vals, becomes_map
