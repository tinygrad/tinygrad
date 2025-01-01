from dataclasses import dataclass, field
from tinygrad.helpers import Metadata, dedup, prod, unwrap
from tinygrad.ops import GroupOp, Ops, PatternMatcher, UOp, UPat, Variable, graph_rewrite, graph_rewrite_map, track_rewrites
from tinygrad.ops import merge_views, symbolic_simple, view_left
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.device import Buffer

def todo(**kwargs): raise Exception("todo!", kwargs)

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...]
  @property
  def inputs(self): return self.bufs[len(self.ast.src):]
  @property
  def outputs(self): return self.bufs[:len(self.ast.src)]

@dataclass(frozen=True)
class SchedulerCtx:
  realizes: dict[UOp, UOp] = field(default_factory=dict)

def mv_const(x:UOp, view:UOp):
  # masked const instantly becomes a VALID, this prevents const folding
  if any(v.mask is not None for v in unwrap(view.st).views): return x.valid(unwrap(view.st))
  # otherview merge views
  return x.replace(src=(x.src[0].view(unwrap(view.st)),))

remove_movementops = PatternMatcher([
  (UPat(GroupOp.Movement, name="mov", src=(UPat.var("x"),)), lambda x,mov: x.view(mov.st)),
  (UPat(Ops.VIEW, name="view", src=(UPat.var("x"),)), lambda view,x: x if x.st is not None and view.st.contiguous and view.shape==x.shape else None),
  (UPat(Ops.VIEW, name="view", src=(UPat.cvar("x"),)), mv_const),
])

def fold_size0_op(root:UOp):
  if root.op in {Ops.VIEW, Ops.SINK} or root.st is None or root.size != 0: return None
  if root.op is Ops.CONST and root.const_arg == 0: return None
  return root.const_like(0)

def fold_const_reduce(root:UOp, x:UOp):
  prshape = prod(unwrap(x.st).shape[i] for i in root.arg[1])
  ret = x.const_arg
  match root.arg[0]:
    case Ops.ADD: ret *= prshape
    case Ops.MUL: ret **= prshape
    case Ops.MAX: pass # NOTE: Ops.MAX is passthrough
    case _: return None
  return root.const_like(ret)

def filter_sink_noops(root:UOp):
  new_src = [x for x in root.src if x.base.op is not Ops.CONST and x.base.realized is None]
  return UOp(Ops.NOOP) if len(new_src) == 0 else UOp.sink(*new_src) if tuple(new_src) != root.src else None

ops_folding = PatternMatcher([
  (UPat(Ops.SINK, name="root"), filter_sink_noops),
  (UPat(set(Ops), name="root"), fold_size0_op),
  (UPat(Ops.REDUCE_AXIS, src=(UPat.cvar("x"),), name="root"), fold_const_reduce),
  (UPat(Ops.DETACH, name="root"), lambda root: root.src[0]),
])

# "meta ops" already have a buffer uop, we should understand why this is required
def realize_metaop(ctx:SchedulerCtx, root:UOp):
  ctx.realizes[root.buf_uop] = root
  return root.buf_uop.view(unwrap(root.st))

# otherwise we give uops buffers like normal
def realize_uop(ctx:SchedulerCtx, root:UOp):
  buf = UOp.new_buffer(root.device, root.size, root.dtype)
  ctx.realizes[buf] = root
  return buf.view(unwrap(root.st))

def realize_sink_src(ctx:SchedulerCtx, root:UOp):
  new_src: list[UOp] = []
  for x in root.src:
    if x.base.op is Ops.BUFFER: new_src.append(x.base)
    else: new_src.append(realize_uop(ctx, x))
  return UOp.sink(*new_src) if tuple(new_src) != root.src else None

def realize_before_copy(ctx:SchedulerCtx, root:UOp, copyout:UOp, copyin:UOp):
  if copyin.base.op is Ops.BUFFER: return None
  new_copyin = UOp.new_buffer(copyin.device, copyin.size, copyin.dtype)
  ctx.realizes[new_copyin] = copyin
  return root.replace(src=(copyout, new_copyin.view(unwrap(copyin.st))))

def realize_view(ctx:SchedulerCtx, root:UOp, view:UOp):
  if root.st is None or view.size <= root.size or root.base.op is Ops.BUFFER: return None
  return realize_uop(ctx, root).view(unwrap(view.st))

realize = PatternMatcher([
  (UPat(Ops.CONTIGUOUS, name="root"), realize_uop),
  # TODO: add view_right and delete this!
  # reduce op fusion is an optimization, correctness first.
  (UPat(Ops.REDUCE_AXIS, name="root"), realize_uop),
  (UPat.var("root").view(name="view"), realize_view),
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
    new_src.append(UOp.store(UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(size=ctx[i].size), (), i), ShapeTracker.from_shape(x.shape).to_uop(), x))
  return UOp.sink(*new_src)

astify = PatternMatcher([
  (UPat(Ops.CONTIGUOUS, src=(UPat.var("x"),)), lambda x: x),
  (UPat(Ops.BUFFER, name="buf"), load_buffer),
  (UPat(Ops.SINK, name="sink"), store_outputs),
])

# from monday meeting we conculded it's fine for const to not have a shapetracker
# TODO: this probably belongs to kernel's fixup_ast
fix_const = PatternMatcher([(UPat(Ops.CONST, src=(UPat(),), name="x"), lambda x: x.replace(src=())),])

@track_rewrites(named=True)
def create_schedule_with_vars(outs:list[UOp]) -> tuple[list[ScheduleItem], dict[Variable, int]]:
  tensor_map = graph_rewrite_map(UOp.sink(*outs), remove_movementops+merge_views+symbolic_simple+ops_folding)
  schedule_map = graph_rewrite_map(UOp.sink(*[tensor_map[x] for x in outs]), remove_movementops+merge_views+ops_folding+realize, ctx:=SchedulerCtx())
  schedule: list[ScheduleItem] = []
  for buf,uop in ctx.realizes.items():
    ast = graph_rewrite(UOp.sink(uop), remove_movementops+merge_views+view_left+astify, bufs:=[buf])
    schedule.append(si:=ScheduleItem(graph_rewrite(ast, fix_const), tuple(dedup(x.buffer for x in bufs)), ()))
    for out in si.outputs: out.ref(1)
  for k,v in tensor_map.items():
    if k is v: continue
    if (realized:=schedule_map.get(v)) is None: continue
    k.become(realized)
  return schedule, {}
