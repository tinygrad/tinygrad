import functools
from dataclasses import dataclass
from typing import Tuple, List, Dict
from tinygrad.dtype import dtypes
from tinygrad.ops import GroupOp, UOp, Ops, PatternMatcher, UPat, Variable, graph_rewrite, track_rewrites, merge_views, view_left
from tinygrad.helpers import Metadata, unwrap, unwrap_or
from tinygrad.device import Buffer
from tinygrad.shape.shapetracker import ShapeTracker

# **** ScheduleItem return type

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: Tuple[Buffer, ...]
  metadata: Tuple[Metadata, ...]
  assign_preloads: Tuple[UOp, ...]
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

remove_movement_ops = PatternMatcher([(UPat(GroupOp.Movement, name="x"), lambda x: x.base.view(unwrap(x.st))),])

def _bufferize(ctx:Dict[UOp, UOp], x:UOp):
  if x in ctx or x.op is Ops.VIEW and len(x.src) == 1: return None
  buf_uop = x.buf_uop if x.op is Ops.VIEW else UOp.new_buffer(x.device, x.size, x.dtype)
  ctx[buf_uop] = x
  return buf_uop.view(unwrap(x.st))
realize = PatternMatcher([
  (UPat(Ops.SINK, name="sink"),
   lambda ctx,sink: UOp(Ops.SINK, src=new_src) if (new_src:=tuple(nx for x in sink.src if (nx:=_bufferize(ctx, x)) is not None))!=sink.src else None),
  (UPat(Ops.COPY, name="cp", src=(UPat.var("x"),)), lambda ctx,cp,x: None if (bx:=unwrap_or(_bufferize(ctx, x), x)) is x else cp.replace(src=(bx,))),
  (UPat(Ops.VIEW, name="x", src=(UPat(Ops.BUFFER), UPat(GroupOp.Meta))), _bufferize),
  (UPat(Ops.CONTIGUOUS, name="x"), _bufferize),
])

def add_buf(ctx:List[UOp], b:UOp):
  ctx.append(b)
  return UOp(Ops.DEFINE_GLOBAL, b.dtype, (), len(ctx)-1)
to_ast = view_left+PatternMatcher([
  # ** buffer op rules
  # const is just const
  (UPat(Ops.VIEW, name="st", src=(UPat(), UPat.cvar("x"))), lambda st,x: UOp(Ops.VALID, dtypes.bool, (st.st.to_uop(),)).where(x, 0)),
  # buffer view is load
  (UPat(Ops.VIEW, name="st", src=(UPat(Ops.DEFINE_GLOBAL, name="ptr"),)), lambda st,ptr: UOp.load(ptr, st.st.to_uop(), dtype=ptr.dtype.base)),
  # buffer+op view is store
  (UPat(Ops.VIEW, name="st", src=(UPat(Ops.DEFINE_GLOBAL, name="ptr"), UPat.var("v"))), lambda st,ptr,v: UOp.store(ptr, st.st.to_uop(), v)),
  # ** sink rules
  (UPat(Ops.SINK, src=(UPat.store(UPat.var("dest"), UPat(), UPat(GroupOp.Meta, name="metaop")),)),
   lambda metaop,dest: metaop.replace(src=(dest,)+metaop.src)),
  (UPat(Ops.SINK, src=(UPat(Ops.LOAD),)), lambda: UOp(Ops.NOOP)),
  # ** misc rules
  # buffer becomes a ptr
  (UPat(Ops.BUFFER, name="b"), add_buf),
  # don't need contiguous anymore
  (UPat(Ops.CONTIGUOUS, src=(UPat.var("x"),)), lambda x:x),
])

cut_edges = PatternMatcher([
  (UPat(Ops.VIEW, name="st", src=(UPat(Ops.BUFFER, name="parent"), UPat(Ops.BUFFER).view())), lambda parent,st: parent.view(unwrap(st.st))),
])

def schedule_uop(out:UOp, dest:UOp) -> ScheduleItem:
  sink = out.sink() if out.op is Ops.VIEW else UOp(Ops.VIEW, out.dtype, (dest, out), unwrap(out.st)).sink()
  sink = graph_rewrite(sink, cut_edges, dest, bottom_up=True)
  bufs: List[UOp] = []
  sink = graph_rewrite(sink, to_ast, bufs)
  return ScheduleItem(sink, tuple(b.buffer for b in bufs), (), ())

@track_rewrites(named=True)
def create_schedule_with_vars(outs:List[UOp]) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
  sink = graph_rewrite(UOp.sink(*outs), remove_movement_ops+merge_views)
  realizes: Dict[UOp, UOp] = {}
  graph_rewrite(sink, realize, realizes)
  schedule = [schedule_uop(u, b) for b,u in realizes.items()]
  for tensor_uop, buf_uop in zip(outs, reversed(list(realizes))): tensor_uop.become(buf_uop.view(ShapeTracker.from_shape(tensor_uop.shape)))
  return schedule, {}

def create_schedule(outs:List[UOp]) -> List[ScheduleItem]:
  schedule, var_vals = create_schedule_with_vars(outs)
  assert len(var_vals) == 0
  return schedule
