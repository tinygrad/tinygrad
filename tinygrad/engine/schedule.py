from dataclasses import dataclass, field
from tinygrad.device import Buffer
from tinygrad.helpers import Metadata, unwrap
from tinygrad.ops import PatternMatcher, UOp, UPat, Variable, Ops, graph_rewrite, graph_rewrite_map, track_rewrites
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
prune_movementops = merge_views+PatternMatcher([])

def remove_sink_noops(root:UOp):
  if len(new_src:=[x.base for x in root.src if x.base.realized is None and x.base.op is not Ops.CONST]) == 0: return UOp(Ops.NOOP)
  return None if tuple(new_src) == root.src else UOp.sink(*new_src)
prune_ops = symbolic_simple+PatternMatcher([
  (UPat(Ops.SINK, name="root"), remove_sink_noops),
])

# ** memory allocation
@dataclass(frozen=True)
class SchedulerCtx:
  realizes:dict[UOp, UOp] = field(default_factory=dict)

def sink_outputs(ctx:SchedulerCtx, root:UOp):
  new_src = [x.base if x.base.op is Ops.BUFFER else realize(ctx, x.base) for x in root.src]
  return None if tuple(new_src) == root.src else UOp.sink(*new_src)

def realize_copy(ctx:SchedulerCtx, root:UOp, dest:UOp, view:UOp, copyin:UOp):
  ctx.realizes[dest] = root
  return view

def realize(ctx:SchedulerCtx, root:UOp):
  dest = UOp.new_buffer(root.device, root.size, root.dtype)
  ctx.realizes[dest] = root
  return dest

allocate_bufs = PatternMatcher([
  (UPat(Ops.SINK, name="root"), sink_outputs),
  # COPY is COPY(VIEW(BUFFER), copyin)
  (UPat(Ops.COPY, name="root", src=(UPat(Ops.VIEW, name="view", src=(UPat(Ops.BUFFER, name="dest"))), UPat.var("copyin"))), realize_copy),
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
])

# ** one uop graph™
@track_rewrites(named=True)
def create_schedule_with_vars(outs:list[UOp]) -> tuple[list[ScheduleItem], dict[Variable, int]]:
  sink = UOp.sink(*outs)
  # simplify pass
  tensor_map = graph_rewrite_map(sink, prune_movementops+prune_ops)
  rev_tensor_map = {v:k for k,v in tensor_map.items()}
  # realize pass
  realize_map = graph_rewrite_map(tensor_map[sink], merge_views+allocate_bufs, ctx:=SchedulerCtx())
  # create schedule items
  schedule: list[ScheduleItem] = []
  for r,v in ctx.realizes.items():
    ast = graph_rewrite(UOp.sink(v), to_ast, sctx:=ASTCtx([r]))
    schedule.append(si:=ScheduleItem(ast, tuple(b.buffer for b in sctx.bufs)))
    for out in si.outputs: out.ref(1)
  for k, v in realize_map.items():
    if (tensor:=rev_tensor_map[k]) is v: continue
    tensor.become(v)
  return schedule, {}
