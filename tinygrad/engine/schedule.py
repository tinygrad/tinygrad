import sys, atexit, functools, pickle
from collections import defaultdict, deque
from dataclasses import dataclass, field
from tinygrad.ops import UOp, Variable, Ops, GroupOp, PatternMatcher, UPat, graph_rewrite, graph_rewrite_map, track_rewrites, type_verify, buffers
from tinygrad.ops import can_pad, identity_element, resolve, symbolic_simple, view_left, merge_views
from tinygrad.helpers import Context, ContextVar, Metadata, all_int, all_same, colored, diskcache_put, merge_dicts, prod, dedup, getenv, unwrap
from tinygrad.helpers import FUSE_CONV_BW, FUSE_ARANGE, DEBUG, CAPTURE_PROCESS_REPLAY
from tinygrad.dtype import DType, ImageDType, dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, strides_for_shape
from tinygrad.device import Buffer

# creation can recurse a lot
sys.setrecursionlimit(10000)

# **** Tensor UOp spec

tensor_uop_spec = PatternMatcher([
  (UPat(Ops.DEVICE, dtypes.void, (), name="device"), lambda device: isinstance(device.arg, str)),
  (UPat(Ops.BUFFER, src=(UPat(Ops.DEVICE),), name="buf"),
   lambda buf: isinstance(buf.arg, tuple) and len(buf.arg) == 2 and all_int(buf.arg) and isinstance(buf.dtype, (DType, ImageDType))),

  (UPat(GroupOp.Movement, name="mv", src=(UPat.var("x"),)),
   # naturally correct
   lambda mv,x: (isinstance(mv.arg, tuple) and mv.dtype == x.dtype) or
   # "make things that can't be images not images" can change the buffer dtype
   # this is fine as long as it's a realized buffer and base dtypes match.
   ((isinstance(mv.dtype, ImageDType) or isinstance(x.dtype, ImageDType)) and x.dtype.base == mv.dtype.base and x.is_realized)),

  # Tensor variable bindings
  (UPat(Ops.BIND, dtypes.int, (UPat(Ops.DEFINE_VAR), UPat.cvar(dtype=dtypes.int)), arg=None), lambda: True),

  # Tensor const has a device and an unmasked ShapeTracker of stride 0
  (UPat(Ops.CONST, src=(UPat(Ops.VIEW, name="st", src=(UPat(Ops.DEVICE),)),)),
   lambda st: st.st.views[0].mask is None and len(st.st.views) == 1 and all(s == 0 for s in st.st.views[0].strides)),

  # DETACH and CONTIGUOUS change how we interpret the source UOp
  # CONTIGUOUS ensures the source UOp realizes
  (UPat((Ops.DETACH, Ops.CONTIGUOUS, Ops.CONTIGUOUS_BACKWARD), name="root", src=(UPat.var("x"),), arg=None), lambda root,x: root.dtype == x.dtype),

  # COPY
  # NOTE: the arg here specifies clone=True, which prevents folding same device copy
  (UPat(Ops.COPY, name="copy", src=(UPat(Ops.DEVICE), UPat.var("x"))), lambda copy,x: isinstance(copy.arg, bool) and copy.dtype == x.dtype),

  # VIEW(BUFFER) applies a ShapeTracker on top of the underlying device buffer
  # NOTE: VIEW size exactly matches the underlying BUFFER, tensor doesn't apply movement ops to the VIEW
  (UPat(Ops.VIEW, name="view", src=(UPat(Ops.BUFFER, name="buf"),)),
   lambda view,buf: view.dtype == buf.dtype and view.size == buf.size and view.st.contiguous),

  # ASSIGN changes the value of a realized buffer
  (UPat(Ops.ASSIGN, name="assign", src=(UPat.var("target"), UPat.var("new_val"))),
   lambda assign,target,new_val: target.is_realized and (assign.dtype == target.dtype == new_val.dtype)),
])

# **** ScheduleItem return type

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

# **** Schedule context and big graph

@dataclass(frozen=True)
class ScheduleContext:
  tensor_uops: dict[UOp, list[UOp]] = field(default_factory=dict)    # this maps BUFFER uops of this schedule to the tensor uop
  var_vals: dict[Variable, int] = field(default_factory=dict)        # this maps a BIND's DEFINE_VAR to its value
  assigns: set[UOp] = field(default_factory=set)                     # this holds all the BUFFER uops we ASSIGN to in this schedule
  realizes: dict[UOp, UOp] = field(default_factory=dict)             # this holds all the BUFFER uops we mutate in this schedule
  allbufs: dict[UOp, UOp] = field(default_factory=dict)              # this maps BUFFER uops the actual op
  ops_metadata: dict[UOp, Metadata] = field(default_factory=dict)    # this maps fused ops to Metadata
  children: defaultdict[UOp, dict[UOp, None]] = field(default_factory=lambda: defaultdict(dict))
  preloads: defaultdict[Buffer, dict[UOp, None]] = field(default_factory=lambda: defaultdict(dict))

# wrap tensor uops around a VIEW(BUFFER, <uop>)
# this BUFFER preserves a link back to the uop on the tensor after the scheduler rewrites it.
def add_buffers(buf:UOp, tensor_map:dict[UOp, list[UOp]], ctx:ScheduleContext, cache:dict[UOp, UOp]) -> UOp:
  if (r:=cache.get(buf)) is not None: return r
  # SINK is passthrough
  if buf.op is Ops.SINK: return buf.replace(src=tuple(add_buffers(x, tensor_map, ctx, cache) for x in buf.src))
  # skip creating buffers for CONST/BIND/DEVICE/BUFFER
  if buf.base.is_realized or buf.base.op in {Ops.CONST, Ops.BIND, Ops.DEVICE}: return buf
  # VIEW is passthrough
  if buf is not buf.base:
    cache[buf] = ret = add_buffers(buf.base, tensor_map, ctx, cache).view(unwrap(buf.st))
    return ret
  # make things that can't be images not images
  dtype = buf.dtype
  if isinstance(dtype, ImageDType) and (prod(buf.shape)!=prod(dtype.shape) or not any(buf.shape[x]%4==0 for x in unwrap(buf.st).unit_stride_axes())):
    if DEBUG >= 2: print(f"forcing image {dtype} with shape {buf.shape} to {dtype.base}")
    dtype = buf.dtype.base
  # ASSIGN already has a target buffer, otherwise we create a new one
  assert isinstance(buf.device, str), f"buf device is str, not {buf.device}"
  buf_uop = buf.buf_uop if buf.op is Ops.ASSIGN else UOp.new_buffer(buf.device, buf.size, dtype)
  op = buf.replace(dtype=dtype, src=tuple(add_buffers(x, tensor_map, ctx, cache) for x in buf.src))
  # track the underlying tensor uop for this buffer
  ctx.tensor_uops[buf_uop] = tensor_map[buf]
  # (early) bufferize
  cache[buf] = ret = UOp(Ops.VIEW, dtype.base, (buf_uop, op), buf.st)
  return ret

# **** AST graph rewrite

# ** movement ops

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
  # NOTE: swizzle resolves once we hit STORE
  return ret if ret.op is Ops.STORE else ret.view(ShapeTracker.from_shape(output_swizzle.shape))

def merge_double_reduce(root:UOp, first_reduce:UOp) -> UOp:
  assert root.arg[0] == first_reduce.arg[0], "can't merge reduceops with different alu"
  assert not any(x.op is Ops.REDUCE_AXIS for x in first_reduce.src[0].toposort), "can't merge more than two reduceops at a time"
  return first_reduce.replace(arg=(first_reduce.arg[0], root.axis_arg+first_reduce.axis_arg))

# push VIEW to children
view_right = merge_views+PatternMatcher([
  # STORE(.., ASSIGN(VIEW(BUFFER), new_val)) -> VIEW(STORE(.., new_val))
  (UPat(Ops.STORE, src=(UPat.var("b"), UPat.var("st"), UPat.assign(UPat.var("target"), UPat.var("val")))),
   lambda b,target,st,val: apply_swizzle(UOp.store(b, st, val).view(target.st))),
  # REDUCE(src.view(contiguous=False)) -> REDUCE(src.view(contiguous=True)).view()
  (UPat(Ops.REDUCE_AXIS, src=(UPat.var("src"),), name="r").view(name="v"), lambda v,r,src: None if v.st.contiguous else swizzle_r(r, src, v.st)),
  # REDUCE(src.view()) -> REDUCE(src).view()
  (UPat(Ops.REDUCE_AXIS, src=(UPat.var("src").view(name="v"),), name="r"), reduceop_view_right),
  # ALU(src.view()) -> ALU(src).view()
  (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.ASSIGN, Ops.CONTIGUOUS, Ops.STORE), name="root"), elementwise_view_right),
  # double reduce op collapses to a single reduce op
  (UPat(Ops.REDUCE_AXIS, src=(UPat(Ops.REDUCE_AXIS, name="first_reduce"),), name="root"), merge_double_reduce),
])

# ** ScheduleItem context builder

@dataclass(frozen=True)
class ScheduleItemContext:
  var_vals: dict[Variable, int]
  sts: set[ShapeTracker] = field(default_factory=set)
  bufs: list[UOp] = field(default_factory=list)

def _append_st_vars(ctx:ScheduleItemContext, x:UOp) -> UOp|None:
  if (st:=unwrap(x.st)) in ctx.sts: return None
  st, var_vals = st.simplify().unbind()
  ctx.var_vals.update(var_vals)
  ctx.sts.add(st)
  return st.to_uop() if st != x.st else None

def _append_buf(ctx:ScheduleItemContext, x:UOp) -> UOp:
  ctx.bufs.append(x)
  return UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(size=x.size), (), len(ctx.bufs)-1)

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
  # PRELOAD becomes LOAD
  (UPat(Ops.PRELOAD, name="root"), lambda root:root.replace(op=Ops.LOAD)),
  # once images are loaded they become the base dtype
  (UPat(set(Ops)-{Ops.DEFINE_GLOBAL}, name="x"), lambda x: x.replace(dtype=x.dtype.base) if isinstance(x.dtype, ImageDType) else None),
  # CONST(VIEW) becomes VALID too, TODO: doesn't have to
  (UPat((Ops.CONST, Ops.DEFINE_VAR), name="x", src=(UPat(Ops.VIEW, name="st"),)), lambda x,st: x.replace(src=()).valid(st.st)),
])

# LOAD(BUFFER) -> the STORE value if it's we're doing the STORE in the same kernel
multioutput = PatternMatcher([(UPat.load(UPat.var("b"), UPat()), lambda ctx,b: ctx.get(b)),])

def schedule_uop(pre:UOp, ctx:ScheduleContext) -> ScheduleItem:
  # remove movement ops + substitute LOAD of fused STORE with just the value
  sink = graph_rewrite(graph_rewrite(pre, multioutput+view_left, store_bufs:={x.buf_uop:x.src[2] for x in pre.src}), view_right)
  # remove extra uops from SINK + substitue BUFFER with DEFINE_GLOBAL
  ast = graph_rewrite(sink, to_si, si_ctx:=ScheduleItemContext(ctx.var_vals))
  # deal with ASSIGN
  if len(ctx.assigns) != 0:
    assign_preloads = ctx.preloads[si_ctx.bufs[0].buffer]
    for x in list(sink.toposort)[::-1]:
      # we only allow a kernel to depend on either the before ASSIGN or after ASSIGN version of a BUFFER
      if x.op is Ops.LOAD and x.buf_uop in assign_preloads: raise RuntimeError("cycle detected in graph")
      # PRELOAD tells the toposort this kernel should run before ASSIGN
      if x.op is Ops.PRELOAD:
        assign_preloads[x.buf_uop] = None
        # if this kernel also assigns to the buffer, we only allow either contiguous or masked views for the LOAD
        if x.buf_uop in store_bufs and not (st:=x.st_arg).contiguous:
          # if it has a single view and it becomes contiguous when you shrink expanded axes, it's fine
          if len(st.views) == 1 and st.shrink(tuple((0,1) if st == 0 else (0,s) for s,st in zip(st.shape, st.views[0].strides))).contiguous: pass
          # if it has a single view and it's equal when you shrink a contig, it's fine
          elif len(st.views) == 1 and (mask:=st.views[0].mask) is not None and ShapeTracker.from_shape(st.shape).shrink(mask) == st.shrink(mask): pass
          # otherwise, it's not fine
          else: raise RuntimeError("self operand of augmented assign must be contiguous.\nhelp: consider using .contiguous():\n"
                                   +colored("   - a += a.T\n", "red")+colored("   + a += a.T.contiguous()", "green"))
  # capture process replay
  if CAPTURE_PROCESS_REPLAY:
    with Context(PICKLE_BUFFERS=0): PROCESS_REPLAY_CAPTURE[str(pre.key)] = pickle.dumps((pre, ContextVar._cache, ast))
  return ScheduleItem(ast, tuple(u.buffer for u in si_ctx.bufs), tuple(dedup(m for x in pre.toposort if (m:=ctx.ops_metadata.get(x)) is not None)))

PROCESS_REPLAY_CAPTURE: dict[str, bytes] = {}
if CAPTURE_PROCESS_REPLAY:
  @atexit.register
  def save_process_replay() -> None:
    for k,v in PROCESS_REPLAY_CAPTURE.items(): diskcache_put("schedule_process_replay", k, v, prepickled=True)

# **** Schedule grouping

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

def get_isolated_children(r:UOp, reduce_for_op:dict[UOp, UOp], children:defaultdict[UOp, dict[UOp, None]], allbufs:dict[UOp, UOp],
                           realizes:dict[UOp, UOp], group:dict[UOp, None]) -> dict[UOp, None]:
  rc_parents, cache = deque(group), set()
  while rc_parents:
    if (p:=uval(allbufs[rc_parents.pop()])) in cache: continue
    cache.add(p)
    # max one reduceop per kernel
    if p.op is Ops.REDUCE_AXIS: return {}
    rc_parents.extend(x.base.buf_uop for x in p.src if is_scheduled(x.base) and x.base.buf_uop is not r)
  # search descendants of the reduceop that can cleanly group
  descendants: dict[UOp, None] = {}
  for tr in group: recursive_group(tr, unwrap(allbufs[tr].st), tr, children, allbufs, realizes, reduce_for_op, descendants, cache={})
  return merge_dicts([group, {} if any(tr in group for tr in descendants) else descendants])

def group_realizes(ctx:ScheduleContext) -> list[list[UOp]]:
  """search the big graph for all the reduceops that need to realize, sometimes group/fuse the reduceop"""
  # find all reduces, and pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce (or a contig child)
  reduce_for_op: dict[UOp, UOp] = {}
  reduce_of_const: list[UOp] = []
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
    if not forced_realize and len(group) > 1:
      group = get_isolated_children(r, reduce_for_op, ctx.children, ctx.allbufs, ctx.realizes, group)
    # can only fuse assign if no other assign_target is used in the kernel
    if not forced_realize and any(x in ctx.assigns for x in group):
      parents = deque((r, *group))
      while parents and not forced_realize:
        if (p_uop:=ctx.allbufs.get(p:=parents.pop())) is None: continue
        if (p_uop:=uval(p_uop)).op is Ops.ASSIGN and p not in group: forced_realize, can_chase = True, False
        if p in ctx.realizes: continue
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
      ctx.realizes[tr] = tr
    reduce_for_op.update((tr, r) for tr in group)
    if FUSE_ARANGE and r_uop.arg[0] is Ops.ADD and r_uop.src[0].base.op is Ops.CONST: reduce_of_const.append(r)
  # fuse double reduces with no other child
  for reduceop in double_reduces:
    top_reduce = uval(ctx.allbufs[reduceop]).src[0].base.buf_uop
    if len(ctx.children[top_reduce]) == 1: del ctx.realizes[top_reduce]
  # maybe fuse arange with its children
  for rbuf in reduce_of_const:
    group = {tr:None for tr,rop in reduce_for_op.items() if rop is rbuf}
    if any(tensor_uop.op is Ops.CONTIGUOUS for tr in group for tensor_uop in ctx.tensor_uops[tr]): continue
    kernel_children = {c for tr in group for c in ctx.children[tr] if uval(ctx.allbufs[c]).op not in {Ops.COPY, Ops.BUFFER_VIEW}}
    if len(kernel_children) == 0: continue
    for tr in group: del ctx.realizes[tr]
  # group BUFFER uops into kernels
  output_groups: defaultdict[UOp, list[UOp]] = defaultdict(list)
  for ubuf in ctx.realizes: output_groups[reduce_for_op.get(ubuf, ubuf)].append(ubuf)
  return list(output_groups.values())

# **** Schedule creation and BFS toposort

# ** this is schedule level const folding

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
  (UPat(set(Ops)-{Ops.SINK}, name="root"), lambda root: root.const_like(0) if root.base.st is not None and root.size == 0 \
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
  # double contiguous is one contiguous
  (UPat(Ops.CONTIGUOUS, name="root", src=(UPat(Ops.CONTIGUOUS),)), lambda root: root.src[0]),
  # support for using a contiguous permuted view instead of the parent view if one exists
  (UPat(Ops.CONTIGUOUS, name="contig", src=(UPat(Ops.VIEW, name="src"),)), found_contiguous),
  (UPat(GroupOp.ALU, name="alu"), replace_contiguous),
  # remove CONST/BIND/BUFFER from SINK
  (UPat(Ops.SINK, name="root"),
    lambda root: UOp(Ops.SINK, root.dtype, new_src, root.arg)
      if (new_src:=tuple(x for x in root.src if not x.is_realized and x.base.op not in {Ops.CONST, Ops.BIND})) != root.src else None),
])

# ** this decides which ops get realized

class UPatScheduled(UPat):
  def __init__(self, *args, **kwargs):
    super().__init__(Ops.VIEW, name="base", src=(UPat(Ops.BUFFER, name="b"), UPat(*args, **{"name":"to_store",**kwargs})))

def realize(ctx:ScheduleContext, b:UOp, to_store:UOp, **kwargs) -> None: ctx.realizes[b] = to_store

def realize_before_view(ctx:ScheduleContext, view:UOp, src:UOp, b:UOp, **kwargs) -> None:
  st = unwrap(view.st)
  # fold simple pads
  if len(st.views) == 1 and (m:=st.views[-1].mask) is not None and all_int(src.shape) and resolve(prod(src.shape) >= prod([y-x for x,y in m])):
    return None if can_pad(src, ctx.realizes, set()) else realize(ctx, b, src)
  # early realize before expand
  if resolve(prod(src.shape) < prod(st.shape)) and not getenv("DONT_REALIZE_EXPAND"): return realize(ctx, b, src)
  # otherwise safety check pads
  return None if (all(v.mask is None for v in st.views) or can_pad(src, ctx.realizes, set())) else realize(ctx, b, src)

def fold_img_cast(ctx:ScheduleContext, xb:UOp, view:UOp, b:UOp, x:UOp, **kwargs) -> UOp|None:
  if not isinstance(xb.dtype, ImageDType) or b not in ctx.realizes or xb not in ctx.realizes or uval(x.base).op is Ops.COPY: return None
  del ctx.realizes[b]
  return x.view(unwrap(view.st))

def create_subbuffer(base:UOp, b:UOp, root:UOp, x:UOp):
  if isinstance(b.device, tuple) or not b.device.startswith("DISK"): return None
  buffers[b] = x.buf_uop.buffer.view(b.size, b.dtype, unwrap(x.st).views[0].offset*x.dtype.itemsize)
  return base.replace(src=(b, root.replace(op=Ops.BUFFER_VIEW)))

do_realize = PatternMatcher([
  # always realize SINK parents
  (UPat(Ops.SINK, name="sink"), lambda ctx,sink: ctx.realizes.update((x.buf_uop, x) for x in sink.src)),
  # always realize ASSIGN/CONTIGUOUS/COPY/BUFFER_VIEW
  (UPatScheduled({Ops.ASSIGN, Ops.CONTIGUOUS, Ops.COPY, Ops.BUFFER_VIEW}), realize),
  # realize before expand or unsafe pad ops
  (UPat(Ops.VIEW, name="view", src=(UPatScheduled(name="src"),)), realize_before_view),
  # don't realize image to image casts
  (UPat(Ops.VIEW, name="view", src=(UPatScheduled(Ops.CAST, src=(UPat(Ops.VIEW, src=(UPat.var("xb"), UPat()), name="x"),), dtype=dtypes.float),)),
   fold_img_cast),
  # realize before COPY or BUFFER_VIEW
  (UPat(Ops.COPY, src=(UPat(), UPat.any(UPatScheduled(), UPatScheduled().view()),)), realize),
  (UPat(Ops.BUFFER_VIEW, src=(UPat.any(UPatScheduled(), UPatScheduled().view()),)), realize),
  # substitute BITCAST/CONTIGUOUS with BUFFER_VIEW on DISK
  (UPatScheduled((Ops.BITCAST, Ops.CONTIGUOUS), name="root", src=(UPat.var("x"),)), create_subbuffer),
])

# **** rewrite VIEW into LOAD/STORE or fuse the underlying UOp

def unbind_variable(ctx:ScheduleContext, bind:UOp, var:UOp, val:UOp):
  assert isinstance(val.const_arg, int), f"expected BIND value to be int {val}"
  ctx.var_vals[var.replace(src=())] = val.const_arg
  return var

def load_realized(ctx:ScheduleContext, b:UOp, st:UOp):
  # NOTE: if we're assigning to the BUFFER too, PRELOAD tells toposort to place this load before the ASSIGN
  return UOp(Ops.PRELOAD if b in ctx.assigns else Ops.LOAD, b.dtype.base, (b, unwrap(st.st).to_uop()))

def store_or_fuse(ctx:ScheduleContext, b:UOp, x:UOp, st:UOp):
  if (m:=ctx.tensor_uops[b][-1].metadata) is not None: ctx.ops_metadata[x] = m
  if b not in ctx.realizes: return x # collapse BUFFER
  ctx.realizes[b] = UOp.store(b, ShapeTracker.from_shape(st.shape).to_uop(), x)
  return UOp(Ops.LOAD, x.dtype, (b, unwrap(st.st).to_uop()))

break_sched = PatternMatcher([
  (UPat(Ops.BIND, name="bind", src=(UPat.var("var"), UPat.var("val"))), unbind_variable),
  # VIEW of BUFFER either becomes a LOAD/STORE or we fuse it
  (UPat(Ops.VIEW, name="st", src=(UPat(Ops.BUFFER, name="b"),)), load_realized),
  (UPat(Ops.VIEW, name="st", src=(UPat(Ops.BUFFER, name="b"), UPat.var("x"))), store_or_fuse),
])

# **** Schedule context builder

def append_uop(ctx:ScheduleContext, view:UOp, buf_uop:UOp) -> None:
  ctx.allbufs[buf_uop] = view
  if (op:=uval(view)).op is Ops.ASSIGN: ctx.assigns.add(buf_uop)
  for x in op.base.src:
    if is_scheduled(x.base): ctx.children.setdefault(x.base.buf_uop, {})[buf_uop] = None
create_ctx = PatternMatcher([(UPat(Ops.VIEW, name="view", src=(UPat(Ops.BUFFER, name="buf_uop"), UPat())), append_uop)])

# **** movement ops

remove_movement_ops = merge_views+PatternMatcher([
  # NOTE: movement ops are always applied to base
  (UPat(GroupOp.Movement, name="mov", src=(UPat.any(UPat.var("x").view(), UPat.var("x")))), lambda x,mov: x.view(unwrap(mov.st))),
  # some masked views can collapse to 0, VIEW(x) -> CONST(VIEW)
  (UPat(Ops.VIEW, name="view"),
   lambda view: view.const_like(0) if (vm:=view.st.views[-1].mask) is not None and any((x[1]-x[0]) == 0 for x in vm) else None),
])

@track_rewrites(named=True)
def create_schedule_with_vars(big_sink:UOp, skip_check:bool=not __debug__) -> tuple[list[ScheduleItem], dict[Variable, int], dict[UOp, UOp]]:
  if not skip_check: type_verify(list(big_sink.toposort), tensor_uop_spec)
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
  rev_tensor_map: dict[UOp, list[UOp]] = {}
  for k,v in tensor_map.items(): rev_tensor_map.setdefault(v, []).append(k)
  # add BUFFER uops
  sink = add_buffers(tensor_map[big_sink], rev_tensor_map, ctx:=ScheduleContext(), cache={})
  # add realizes
  sink = graph_rewrite(sink, do_realize+create_ctx, ctx)
  # group realizes into kernels
  store_groups = group_realizes(ctx)
  graph_rewrite(sink, break_sched, ctx)
  # create schedule items + map buffers to realized tensors
  prescheduled: list[ScheduleItem] = []
  for store_uops in store_groups:
    small_sink = UOp.sink(*[ctx.realizes[u] for u in store_uops])
    if not all(x.op is Ops.STORE for x in small_sink.src): raise RuntimeError(f"expected all realized BUFFERs to get a STORE {sink}")
    prescheduled.append(schedule_uop(small_sink, ctx))
    # can only schedule once
    for buf_uop in store_uops:
      for tensor_uop in ctx.tensor_uops[buf_uop]: becomes_map[tensor_uop] = buf_uop.view(unwrap(tensor_uop.st))
      # increment refcount for this buffer
      buf_uop.buffer.ref(1)

  # add kernel children
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
  return schedule, ctx.var_vals, becomes_map
