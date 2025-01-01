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
prune_ops = symbolic_simple+PatternMatcher([])

# ** memory allocation
@dataclass(frozen=True)
class SchedulerCtx:
  realizes:dict[UOp, UOp] = field(default_factory=dict)

def sink_outputs(ctx:SchedulerCtx, root:UOp):
  if len(new_src:=[x.base for x in root.src if x.base.realized is None and x.base.op is not Ops.CONST]) == 0: return UOp(Ops.NOOP)
  new_src = [x if x.op is Ops.BUFFER else realize(ctx, x) for x in new_src]
  return None if tuple(new_src) == root.src else UOp.sink(*new_src)

def realize_copy(ctx:SchedulerCtx, root:UOp, view:UOp, dest:UOp, copyin:UOp):
  ctx.realizes[dest] = root
  return view

def realize(ctx:SchedulerCtx, root:UOp):
  dest = UOp.new_buffer(root.device, root.size, root.dtype)
  ctx.realizes[dest] = root
  return dest

allocate_bufs = PatternMatcher([
  (UPat(Ops.SINK, name="root"), sink_outputs),
  (UPat(Ops.COPY, name="root", src=(UPat(Ops.VIEW, name="view", src=(UPat(Ops.BUFFER, name="dest"),)), UPat.var("copyin"))), realize_copy),
])

# ** ast creation
@dataclass(frozen=True)
class ASTCtx:
  bufs: list[UOp]

def load_buf(ctx:ASTCtx, buf:UOp):
  ctx.bufs.append(buf)
  glbl = UOp(Ops.DEFINE_GLOBAL, buf.dtype.ptr(size=buf.size), (), len(ctx.bufs)-1)
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
  tensor_uops = graph_rewrite_map(UOp.sink(*outs), prune_movementops+prune_ops+allocate_bufs, ctx:=SchedulerCtx())
  schedule: list[ScheduleItem] = []
  for r,v in ctx.realizes.items():
    ast = graph_rewrite(UOp.sink(v), to_ast, sctx:=ASTCtx([r]))
    schedule.append(si:=ScheduleItem(ast, tuple(b.buffer for b in sctx.bufs)))
    for out in si.outputs: out.ref(1)
  for k,v in tensor_uops.items():
    if k is v: continue
    if k.op is Ops.ADD: raise Exception(v)
    k.become(v)
  return schedule, {}
