from dataclasses import dataclass, field
from tinygrad.device import Buffer
from tinygrad.dtype import DType, ImageDType, dtypes
from tinygrad.helpers import Metadata, all_int, prod, unwrap
from tinygrad.ops import GroupOp, PatternMatcher, UOp, UPat, Variable, Ops, graph_rewrite, graph_rewrite_map, track_rewrites, type_verify, buffers
from tinygrad.ops import symbolic_simple, merge_views, view_left
from tinygrad.shape.shapetracker import ShapeTracker

BUF_LIMIT = {"METAL":32}

# ** big graph spec

tensor_uop_spec = PatternMatcher([
  # ** stable and well understood specs

  # DEVICE and BUFFER
  (UPat(Ops.DEVICE, dtypes.void, (), name="device"), lambda device: isinstance(device.arg, str)),
  (UPat(Ops.BUFFER, src=(UPat(Ops.DEVICE),), name="buf"), lambda buf:
   # arg: (number, size)
   isinstance(buf.arg, tuple) and len(buf.arg) == 2 and all_int(buf.arg) and \
   # dtype
   isinstance(buf.dtype, (DType, ImageDType))),

  # movement ops
  (UPat(GroupOp.Movement, name="mv", src=(UPat.var("x"),)), lambda mv,x:
   # naturally correct
   (isinstance(mv.arg, tuple) and mv.dtype == x.dtype) or
   # "make things that can't be images not images" can change the buffer dtype
   # this is fine as long as it's a realized buffer and base dtypes match.
   ((isinstance(mv.dtype, ImageDType) or isinstance(x.dtype, ImageDType)) and x.dtype.base == mv.dtype.base and x.is_realized)),

  # Tensor variable bindings
  (UPat(Ops.BIND, dtypes.int, (UPat(Ops.DEFINE_VAR), UPat.cvar(dtype=dtypes.int)), arg=None), lambda: True),

  # Tensor const has a ShapeTracker of shape=() and a device
  (UPat(Ops.CONST, src=(UPat(Ops.VIEW, src=(UPat(Ops.DEVICE),)),)), lambda: True),

  # DETACH and CONTIGUOUS change how we interpret the source UOp
  # CONTIGUOUS ensures the source UOp realizes
  (UPat((Ops.DETACH, Ops.CONTIGUOUS), name="root", src=(UPat.var("x"),), arg=None), lambda root,x: root.dtype == x.dtype),

  # ** specs with room for refactoring and improving

  (UPat(Ops.COPY, name="copy", src=(UPat(Ops.VIEW, src=(UPat(Ops.BUFFER),)), UPat.var("copyin"))), lambda copy,copyin:
   # arg (clone) + dtype
   isinstance(copy.arg, bool) and copy.dtype == copyin.dtype),

  # VIEW(BUFFER) applies a ShapeTracker on top of the underlying device buffer
  # NOTE: VIEW size exactly matches the underlying BUFFER, tensor doesn't apply movement ops to the VIEW
  (UPat(Ops.VIEW, name="view", src=(UPat(Ops.BUFFER, name="buf"),)),
   lambda view,buf: view.dtype == buf.dtype and view.size == buf.size and view.st.contiguous),

  # ASSIGN changes the value of an existing buffer
  (UPat(Ops.ASSIGN, name="assign", src=(UPat.var("target"), UPat.var("new_val"))), lambda assign,target,new_val:
   # target must be a realized device buffer
   (target.op is Ops.BUFFER or target.is_realized) and
   # dtype
   (assign.dtype == target.dtype == new_val.dtype)),

  # ** TODO: these UOps need new specs, the current representation relies on hacks

  # DEVICE and VIEW specify device and shape for BIND
  (UPat(Ops.VIEW, src=(UPat(Ops.DEVICE), UPat(Ops.BIND))), lambda: True),

  # NOTE: EMPTY just ensures the source BUFFER is allocated before children run
  (UPat(Ops.EMPTY, src=(UPat(Ops.VIEW, src=(UPat(Ops.BUFFER),),)), arg=None), lambda: True),

  # TODO: BUFFER_VIEW is overloaded, can we break it into multiple well defined UOps?
  # BUFFER_VIEW shares the device buffer with its source, it uses a subbuffer of the underlying source buffer

  (UPat(Ops.BUFFER_VIEW, name="root", src=(UPat(Ops.VIEW, src=(UPat(Ops.BUFFER),)), UPat.var("x"),)), lambda root,x:
   # BUFFER_VIEW can replace contiguous, keeping dtype the same
   (root.dtype == x.dtype) or
   # it can also replace bitcast, this changes the dtype, but the itemsize stays the same
   (root.dtype != x.dtype and root.dtype.itemsize == x.dtype.itemsize) or
   # it can also represent shape changing bitcast (only on DISK)
   (root.dtype != x.dtype and root.dtype.itemsize != x.dtype.itemsize and x.device.startswith("DISK"))),
])

# ** ScheduleItem return type
@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...] = ()
  @property
  def inputs(self): return self.bufs[len(self.ast.src):]
  @property
  def outputs(self): return self.bufs[:len(self.ast.src)]

# ** schedule simplification
def mv_const(view:UOp, x:UOp):
  if any(v.mask is not None for v in unwrap(view.st).views): return x.valid(unwrap(view.st))
  return x.replace(src=(x.src[0].replace(arg=unwrap(x.st)+unwrap(view.st)),))

prune_movementops = merge_views+PatternMatcher([
  (UPat(GroupOp.Movement, name="mov", src=(UPat.var("x"),)), lambda x,mov:x.view(mov.st)),
  # contiguous VIEW of the same shape is a NOOP
  (UPat(Ops.VIEW, name="view", src=(UPat.var("x"),)), lambda x,view: x if x.st is not None and view.st.contiguous and view.shape==x.shape else None),
  # some masked views can collapse to 0, VIEW(x) -> CONST(VIEW)
  (UPat(Ops.VIEW, name="view"),
   lambda view: view.const_like(0) if (vm:=view.st.views[-1].mask) is not None and any((x[1]-x[0]) == 0 for x in vm) else None),
  # VIEW(const) = masked ? VALID(st, CONST, 0) : CONST(st+st)
  (UPat(Ops.VIEW, name="view", src=(UPat.cvar("x"),)), mv_const),
])

def remove_sink_noops(root:UOp):
  if len(new_src:=[x.base for x in root.src if x.base.realized is None and x.base.op is not Ops.CONST]) == 0: return UOp(Ops.NOOP)
  return None if tuple(new_src) == root.src else UOp.sink(*new_src)

def collapse_size0_ops(root:UOp):
  if root.op in {Ops.SINK, Ops.VIEW} or root.st is None or root.st.size != 0: return None
  return None if root.op is Ops.CONST and root.const_arg == 0 else root.const_like(0)

def collapse_const_reduce(root:UOp, x:UOp):
  prshape = prod(unwrap(x.st).shape[i] for i in root.arg[1])
  ret = x.const_arg
  match root.arg[0]:
    case Ops.ADD: ret *= prshape
    case Ops.MUL: ret **= prshape
    case Ops.MAX: pass
    case _: return None
  return root.const_like(ret)

prune_ops = symbolic_simple+PatternMatcher([
  (UPat(tuple(Ops), name="root"), collapse_size0_ops),
  (UPat(Ops.REDUCE_AXIS, name="root", src=(UPat.cvar("x"),)), collapse_const_reduce),
  (UPat(Ops.DETACH, name="root"), lambda root: root.src[0]),
  (UPat(Ops.SINK, name="root"), remove_sink_noops),
])

# ** memory allocation
@dataclass(frozen=True)
class SchedulerCtx:
  realizes:dict[UOp, UOp] = field(default_factory=dict)

def realize_metaop(ctx:SchedulerCtx, root:UOp):
  ctx.realizes[dest:=root.buf_uop] = root
  # hack subbuffer here, this logic should be moved to realize
  if root.op is Ops.BUFFER_VIEW:
    buffers[dest] = (x:=root.src[1]).buf_uop.buffer.view(root.size, root.dtype, unwrap(x.st).views[0].offset*x.dtype.itemsize)
  return dest.view(unwrap(root.st))
def realize_metaop_with_src(ctx:SchedulerCtx, dest:UOp, root:UOp, src:UOp):
  new_src = realize(ctx, src)
  return realize_metaop(ctx, root.replace(src=(dest, new_src)))

def realize(ctx:SchedulerCtx, root:UOp):
  if root.base.op is Ops.BUFFER: return root
  dest = UOp.new_buffer(root.device, root.size, root.dtype)
  ctx.realizes[dest] = root
  return dest.view(unwrap(root.st))

def realize_view(ctx:SchedulerCtx, view:UOp, base:UOp):
  if base.st is None or base.op is Ops.BUFFER: return None
  if view.size <= base.size: return None
  return realize(ctx, base).view(unwrap(view.st))

allocate_bufs = PatternMatcher([
  (UPat({Ops.COPY, Ops.BUFFER_VIEW}, name="root", src=(UPat.var("dest"), UPat.var("src"))), realize_metaop_with_src),
  (UPat(Ops.EMPTY, name="root"), realize_metaop),
  (UPat(Ops.CONTIGUOUS, name="root"), realize),
  (UPat(Ops.REDUCE_AXIS, name="root"), realize),
  # sometimes realize before view
  (UPat(Ops.VIEW, name="view", src=(UPat.var("base"),)), realize_view),
  # always realize sinked ops
  (UPat(Ops.SINK, name="root"), lambda ctx,root:None if (new_src:=tuple(realize(ctx, x.base) for x in root.src)) == root.src else UOp.sink(*new_src)),
])

# ** ast creation
@dataclass(frozen=True)
class ASTCtx:
  bufs: list[UOp]

def load_buf(ctx:ASTCtx, buf:UOp):
  if buf not in ctx.bufs: ctx.bufs.append(buf)
  glbl = UOp(Ops.DEFINE_GLOBAL, buf.dtype.ptr(size=buf.size), (), ctx.bufs.index(buf))
  return UOp.load(glbl, unwrap(buf.st).to_uop(), dtype=buf.dtype.base)

def ast_sink(root:UOp):
  if any(x.op is Ops.STORE for x in root.src): return None
  new_src:list[UOp] = []
  for i,x in enumerate(root.src):
    new_src.append(UOp.store(UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(x.size), (), i), ShapeTracker.from_shape(x.shape).to_uop(), x))
  return UOp.sink(*new_src)

to_si = view_left+prune_movementops+PatternMatcher([
  # BUFFER -> LOAD(DEFINE_GLOBAL, ShapeTracker(shape=(N,)))
  (UPat(Ops.BUFFER, name="buf"), load_buf),
  # SINK(...) -> SINK(STORE(BUFFER, ...),)
  (UPat(Ops.SINK, name="root"), ast_sink),
  # STORE(.., ASSIGN(VIEW(BUFFER), new_val)) -> STORE(.., new_val).view()
  (UPat(Ops.STORE, src=(UPat.var("b"), UPat.var("st"), UPat.assign(UPat.var("target"), UPat.var("new_val")))),
   lambda b,target,st,new_val: UOp.store(b, st, new_val).view(target.st)),
])

# some dumb things the scheduler has to do to make the ast compatible with the other abstractions
fix_ast = PatternMatcher([
  # (maskless) const is shapeless once we hand it off to kernel.py, fix kernel instead.
  (UPat(Ops.CONST, name="root", src=(UPat(),)), lambda root:root.replace(src=())),
  # "meta ops" don't get sink or store, fix lower_schedule_item instead.
  (UPat(Ops.SINK, src=(UPat.store(UPat(), UPat(), UPat(GroupOp.Meta, name="m")),)), lambda m:m),
  # contiguous and assign do not exist in the final ast
  (UPat({Ops.CONTIGUOUS, Ops.ASSIGN}, name="root"), lambda root:root.src[-1]),
])

# ** one uop graph™
@track_rewrites(named=True)
def create_schedule_with_vars(outs:list[UOp]) -> tuple[list[ScheduleItem], dict[Variable, int]]:
  # create the graph and verify
  type_verify(list((sink:=UOp.sink(*outs)).toposort), tensor_uop_spec)
  # simplify pass
  tensor_map = graph_rewrite_map(sink, prune_movementops+prune_ops)
  # realize pass
  realize_map = graph_rewrite_map(tensor_map[sink], merge_views+allocate_bufs, ctx:=SchedulerCtx())

  # create schedule items
  schedule: list[ScheduleItem] = []
  for r,v in list(ctx.realizes.items()):
    ast = graph_rewrite(UOp.sink(v), to_si, sctx:=ASTCtx([r]))
    schedule.append(si:=ScheduleItem(graph_rewrite(ast, fix_ast), tuple(b.buffer for b in sctx.bufs)))
    for out in si.outputs: out.ref(1)

  # update tensor references
  for k,v in tensor_map.items():
    # it's ok for realize_map <= tensor_map
    if (r:=realize_map.get(v)) is None: continue
    # some things don't need to become
    if k is r or k is sink or k.st is None: continue
    # if the tensor is flat it becomes a BUFFER, otherwise it's a VIEW(BUFFER)
    k.become(r if k.shape == r.shape else r.view(k.st))
  # also update the sinked outputs
  rev_tensor_map = {v:k for k,v in tensor_map.items()}
  rev_realize_map = {v:k for k,v in realize_map.items()}
  for r in realize_map[tensor_map[sink]].src:
    rr = rev_realize_map.get(ctx.realizes[r])
    if rr is not None:
      k = rev_tensor_map[rr]
      k.become(r if k.shape == r.shape else r.view(unwrap(k.st)))

  return schedule, {}
