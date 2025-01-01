from dataclasses import dataclass, field
from tinygrad.helpers import Metadata, dedup, unwrap
from tinygrad.ops import GroupOp, Ops, PatternMatcher, UOp, UPat, Variable, graph_rewrite, graph_rewrite_map, track_rewrites
from tinygrad.ops import merge_views, symbolic_simple, view_left
from tinygrad.device import Buffer

def todo(**kwargs): raise Exception("todo!", kwargs)

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...]
  @property
  def inputs(self): return self.bufs[self.ast.src:]
  @property
  def outputs(self): return self.bufs[:self.ast.src]

@dataclass(frozen=True)
class SchedulerCtx:
  realizes: dict[UOp, UOp] = field(default_factory=dict)

def mv_const(x:UOp, view:UOp):
  if any(v.mask is not None for v in unwrap(view.st).views): return x.valid(unwrap(view.st))
  return x.replace(src=(x.src[0].view(unwrap(view.st)),))

remove_movementops = PatternMatcher([
  (UPat(GroupOp.Movement, name="mov", src=(UPat.var("x"),)), lambda x,mov: x.view(mov.st)),
  (UPat(Ops.VIEW, name="view", src=(UPat.var("x"),)), lambda view,x: x if x.st is not None and view.st.contiguous and view.shape==x.shape else None),
  (UPat(Ops.VIEW, name="view", src=(UPat.cvar("x"),)), mv_const),
])

def realize_metaop(ctx:SchedulerCtx, root:UOp):
  ctx.realizes[root.buf_uop] = root
  return root.buf_uop.view(unwrap(root.st))

def realize_sink_src(ctx:SchedulerCtx, root:UOp):
  new_src: list[UOp] = []
  for x in root.src:
    if x.base.op is Ops.BUFFER: new_src.append(x.base)
    else: new_src.append(realize_uop(ctx, x))
  return UOp(Ops.NOOP) if len(root.src) == 0 else UOp.sink(*new_src) if tuple(new_src) != root.src else None

def realize_before_copy(ctx:SchedulerCtx, root:UOp, copyout:UOp, copyin:UOp):
  if copyin.base.op is Ops.BUFFER: return None
  new_copyin = UOp.new_buffer(copyin.device, copyin.size, copyin.dtype)
  ctx.realizes[new_copyin] = copyin
  return root.replace(src=(copyout, new_copyin.view(unwrap(copyin.st))))

def realize_uop(ctx:SchedulerCtx, root:UOp):
  buf = UOp.new_buffer(root.device, root.size, root.dtype)
  ctx.realizes[buf] = root
  return buf.view(unwrap(root.st))

realize = PatternMatcher([
  (UPat(Ops.CONTIGUOUS, name="root"), realize_uop),
  # TODO: add view_right and delete this!
  # reduce op fusion is an optimization, correctness first.
  (UPat(Ops.REDUCE_AXIS, name="root"), realize_uop),
  (UPat(Ops.COPY, name="root", src=(UPat.var("copyout"), UPat.var("copyin"),)), realize_before_copy),
  (UPat(Ops.COPY, name="root"), realize_metaop),
  (UPat(Ops.SINK, name="root"), realize_sink_src),
])

def load_buffer(ctx:list[UOp], buf:UOp):
  ctx.append(buf)
  return UOp.load(UOp(Ops.DEFINE_GLOBAL, buf.dtype.ptr(size=buf.size), (), len(ctx)-1), unwrap(buf.st).to_uop(), dtype=buf.dtype.base)

def store_outputs(ctx:list[UOp], sink:UOp):
  if all(x.op is Ops.STORE or x.op in GroupOp.Meta for x in sink.src): return None
  new_src: list[UOp] = []
  for i,x in enumerate(sink.src):
    new_src.append(UOp.store(UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(size=ctx[i].size), (), i), unwrap(x.st).to_uop(), x))
  return UOp.sink(*new_src)

astify = PatternMatcher([
  (UPat(Ops.CONTIGUOUS, src=(UPat.var("x"),)), lambda x: x),
  (UPat(Ops.BUFFER, name="buf"), load_buffer),
  (UPat(Ops.SINK, name="sink"), store_outputs),
])

# from monday meeting we conculded it's fine for const to not have a shapetracker
fix_const = PatternMatcher([(UPat(Ops.CONST, src=(UPat(),), name="x"), lambda x: x.replace(src=())),])

@track_rewrites(named=True)
def create_schedule_with_vars(outs:list[UOp]) -> tuple[list[ScheduleItem], dict[Variable, int]]:
  tensor_map = graph_rewrite_map(UOp.sink(*outs), remove_movementops+merge_views+symbolic_simple)
  schedule_map = graph_rewrite_map(UOp.sink(*[tensor_map[x] for x in outs]), remove_movementops+merge_views+realize, ctx:=SchedulerCtx())
  schedule: list[ScheduleItem] = []
  for buf,uop in ctx.realizes.items():
    ast = graph_rewrite(UOp.sink(uop), remove_movementops+merge_views+view_left+astify, bufs:=[buf])
    schedule.append(ScheduleItem(graph_rewrite(ast, fix_const), tuple(dedup(x.buffer for x in bufs)), ()))
  rev_tensor_map = {v:k for k,v in tensor_map.items()}
  for k,v in schedule_map.items():
    if k is v: continue
    if v.base.op is Ops.BUFFER:
      rev_tensor_map[k].become(v)
  return schedule, {}
