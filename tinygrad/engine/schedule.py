import functools
from dataclasses import dataclass, field
from tinygrad.ops import GroupOp, PatternMatcher, UOp, Ops, UPat, Variable, graph_rewrite, track_rewrites, merge_views, symbolic_simple
from tinygrad.helpers import Metadata, unwrap
from tinygrad.device import Buffer

# ***** ScheduleItem return type *****

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...]
  assign_preloads: tuple[UOp, ...]
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

remove_movement_ops = PatternMatcher([
  (UPat(GroupOp.Movement, name="mv", src=(UPat.var("x"),)), lambda x,mv: x.base.view(mv.st)),
])


@dataclass(frozen=True)
class ScheduleCtx:
  realizes:dict[UOp, UOp] = field(default_factory=dict)

def add_buffer(ctx:ScheduleCtx, root:UOp):
  if root.op is Ops.BUFFER: return None
  buffer = UOp.new_buffer(root.device, root.size, root.dtype)
  ctx.realizes[buffer] = root
  return buffer

def realize_copy(ctx:ScheduleCtx, copyin:UOp, copy:UOp):
  add_buffer(ctx, copyin)
  return add_buffer(ctx, copy)

add_realizes = PatternMatcher([
  (UPat(Ops.COPY, src=(UPat(Ops.DEVICE), UPat.var("copyin")), name="copy"), realize_copy),
])

def _add_buf(ctx:list[UOp], buf:UOp):
  glbl = UOp(Ops.DEFINE_GLOBAL, buf.dtype.ptr(size=buf.size), (), len(ctx))
  ctx.append(buf)
  return UOp.load(glbl, unwrap(buf.st).to_uop(), dtype=buf.dtype.base)

load_buffers = PatternMatcher([
  (UPat(Ops.BUFFER, name="buf"), _add_buf),
  (UPat.store(UPat.load(UPat.var("glbl"), UPat()), UPat.var("st"), UPat.var("val")), lambda glbl,st,val: UOp.store(glbl, st, val)),
])

@track_rewrites(named=True)
def create_schedule_with_vars(outs:list[UOp]) -> tuple[list[ScheduleItem], dict[Variable, int]]:
  sink = UOp.sink(*outs)
  sink = graph_rewrite(sink, remove_movement_ops+merge_views+symbolic_simple)
  sink = graph_rewrite(sink, add_realizes, ctx:=ScheduleCtx())
  schedule: list[ScheduleItem] = []
  for buffer, uop in ctx.realizes.items():
    ast = graph_rewrite(UOp.sink(UOp.store(buffer, unwrap(uop.st).to_uop(), uop)), load_buffers, bufs:=[])
    si = ScheduleItem(ast, tuple(x.buffer for x in bufs), (), ())
    schedule.append(si)
  return schedule, {}
