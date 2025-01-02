from dataclasses import dataclass, field
from tinygrad.device import Buffer
from tinygrad.helpers import Metadata, prod, unwrap
from tinygrad.ops import GroupOp, PatternMatcher, UOp, UPat, Variable, Ops, graph_rewrite, graph_rewrite_map, track_rewrites
from tinygrad.ops import symbolic_simple, merge_views, view_left
from tinygrad.shape.shapetracker import ShapeTracker

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
  (UPat(Ops.VIEW, name="view", src=(UPat.var("x"),)), lambda x,view: x if x.st is not None and view.st.contiguous and view.shape == x.shape else None),
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
  (UPat(Ops.SINK, name="root"), remove_sink_noops),
])

# ** memory allocation
@dataclass(frozen=True)
class SchedulerCtx:
  realizes:dict[UOp, UOp] = field(default_factory=dict)

def realize_copy(ctx:SchedulerCtx, root:UOp, dest:UOp, copyin:UOp):
  ctx.realizes[dest.buf_uop] = root
  return dest.buf_uop.view(unwrap(root.st))

def realize(ctx:SchedulerCtx, root:UOp):
  dest = UOp.new_buffer(root.device, root.size, root.dtype)
  ctx.realizes[dest] = root
  return dest.view(unwrap(root.st))

def realize_uop(ctx:SchedulerCtx, root:UOp):
  if root.op in {Ops.BUFFER, Ops.CONST, Ops.SINK, Ops.VIEW} or root.st is None: return None
  return realize(ctx, root)

allocate_bufs = PatternMatcher([
  # COPY is COPY(VIEW(BUFFER), copyin)
  (UPat(Ops.COPY, name="root", src=(UPat.var("dest"), UPat.var("copyin"))), realize_copy),
  (UPat(Ops.CONTIGUOUS, name="root"), realize),
  # otherwise check the buffer
  (UPat(tuple(Ops), name="root"), realize_uop),
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

to_ast = view_left+PatternMatcher([
  # BUFFER -> LOAD(DEFINE_GLOBAL, ShapeTracker(shape=(N,)))
  (UPat(Ops.BUFFER, name="buf"), load_buf),
  # SINK(...) -> SINK(STORE(BUFFER, ...),)
  (UPat(Ops.SINK, name="root"), ast_sink),
  (UPat(Ops.CONTIGUOUS, src=(UPat.var("x"),)), lambda x:x),
])
fix_const = PatternMatcher([
  (UPat(Ops.CONST, name="root", src=(UPat(),)), lambda root: root.replace(src=())),
])

# ** one uop graph™
@track_rewrites(named=True)
def create_schedule_with_vars(outs:list[UOp]) -> tuple[list[ScheduleItem], dict[Variable, int]]:
  sink = UOp.sink(*outs)
  # simplify pass
  tensor_map = graph_rewrite_map(sink, prune_movementops+prune_ops)
  # realize pass
  realize_map = graph_rewrite_map(tensor_map[sink], merge_views+allocate_bufs, ctx:=SchedulerCtx())

  # create schedule items
  schedule: list[ScheduleItem] = []
  for r,v in list(ctx.realizes.items()):
    ast = graph_rewrite(UOp.sink(v), to_ast, sctx:=ASTCtx([r]))
    schedule.append(si:=ScheduleItem(graph_rewrite(ast, fix_const), tuple(b.buffer for b in sctx.bufs)))
    for out in si.outputs: out.ref(1)

  for k,v in tensor_map.items():
    # it's ok for realize_map <= tensor_map
    if k.st is None or (r:=realize_map.get(v)) is None: continue
    # some things don't need to become
    if k is r or k is sink: continue
    # if the tensor is flat it becomes a BUFFER, otherwise it's a VIEW(BUFFER)
    k.become(r if k.shape == r.shape else r.view(unwrap(k.st)))
  return schedule, {}
