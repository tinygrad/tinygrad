from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tinygrad.helpers import Context, Metadata
from tinygrad.dtype import dtypes
from tinygrad.ops import UOp, graph_rewrite, PatternMatcher, UPat, UOps, symbolic, track_rewrites, realized
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.device import Buffer

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: Tuple[Buffer, ...]
  metadata: Optional[Tuple[Metadata, ...]]
  @property
  def outputs(self) -> Tuple[Buffer, ...]:
    """Read/write or write only buffers in the schedule."""
    return self.bufs[:len(self.ast.src)] if self.ast.op is UOps.SINK else self.bufs[0:1]
  @property
  def inputs(self) -> Tuple[Buffer, ...]:
    """Read only buffers in the schedule."""
    return self.bufs[len(self.ast.src):] if self.ast.op is UOps.SINK else self.bufs[1:]

pm_merge_views_and_consts = PatternMatcher([
  # merge VIEW
  (UPat(UOps.VIEW, src=(UPat(UOps.VIEW, name="s0"),), name="s1"),
   lambda s0,s1: UOp(UOps.VIEW, s1.dtype, s0.src, s0.arg+s1.arg)),
  # const + copy = const
  (UPat(UOps.COPY, src=(UPat.cvar('c'),)), lambda c: c),
  # const + maskless swizzle = const
  (UPat(UOps.VIEW, src=(UPat.cvar('c'),), name="s"),
    lambda s,c: c if all(x.mask is None for x in s.st.views) else None),
])

pm_push_views = symbolic+pm_merge_views_and_consts+PatternMatcher([
  # VIEW before ALU
  (UPat(UOps.VIEW, src=(UPat(UOps.ALU, name="alu"),), name="s"),
    lambda alu,s: UOp(UOps.ALU, alu.dtype,
                      tuple(UOp(UOps.VIEW, x.dtype, (x,), s.arg) for x in alu.src), alu.arg)),
  # don't need CONTIGUOUS any more
  (UPat(UOps.CONTIGUOUS, src=(UPat.var('x'),)), lambda x: x),
  # remove unneeded (new) buffers
  (UPat(UOps.BUFFER, src=(UPat(UOps.VIEW, src=(UPat(UOps.BUFFER, name="b"),), name="v"), )), lambda b,v: b if v.st.contiguous else None)
])

# *********

def append_buffer(bufs:List[Buffer], buf:UOp, view:Optional[UOp]=None, to_store:Optional[UOp]=None):
  if buf.buffer not in bufs: bufs.append(buf.buffer)
  dg = UOp(UOps.DEFINE_GLOBAL, buf.dtype.ptr(), (), bufs.index(buf.buffer))
  if view is not None: return UOp.load(dg, view.replace(dtype=dtypes.void, src=()), dtype=buf.dtype)
  if to_store is not None: return UOp.store(dg, ShapeTracker.from_shape(to_store.shape).to_uop(), to_store)

enumerate_bufs = PatternMatcher([
  # load and store
  (UPat(UOps.VIEW, src=(UPat(UOps.BUFFER, src=(), name="buf"),), name="view"), append_buffer),
  (UPat(UOps.BUFFER, src=(UPat.var("to_store"),), name="buf"), append_buffer),
  # copy is just copy
  (UPat.sink(UPat.store(UPat(name="dest"), UPat(UOps.VIEW, name="st"),
                        UPat(UOps.COPY, src=(UPat.load(UPat(name="src"), UPat(UOps.VIEW, name="st")),), name="cpy"))),
                        lambda _, dest, src, st, cpy: UOp(UOps.COPY, dest.dtype, (dest, src), st.size) if st.st.contiguous else None),
])

# *********

def append_kernel(k:List[UOp], base:UOp):
  ret = base.replace(src=())
  # NOTE: is this slow?
  if ret not in realized.values(): k.append(base.sink())
  return ret
break_sched = PatternMatcher([
  (UPat(UOps.BUFFER, src=(UPat(),), name="base"), append_kernel),
])

# *********

pm_remove_buffer = PatternMatcher([(UPat(UOps.VIEW, src=(UPat(UOps.BUFFER, src=(UPat.var('x'),)),)), lambda x: x), ])
def add_buffer(to_realize:Dict[UOp, Optional[UOp]], x:UOp):
  # TODO: ugh, this is the worst way to do this
  with Context(TRACK_MATCH_STATS=0): x_bl = graph_rewrite(x, pm_remove_buffer)
  if to_realize.get(x_bl, True) is None:
    #print(len(to_realize), "HIT", sum((x is not None) for x in to_realize.values()))
    ret = UOp.new_buffer(x.dtype, x.device, x.size, (x,)) if x_bl not in realized else realized[x_bl].replace(src=(x,))
    to_realize[x_bl] = ret.replace(src=())
    return ret.reshape(x.shape)
  return None
pm_add_buffer = PatternMatcher([(UPat(tuple(UOps), name="x"), add_buffer), ])

@track_rewrites
def _schedule_rewrite(sink:UOp) -> List[ScheduleItem]:
  sink = graph_rewrite(sink, pm_merge_views_and_consts)
  to_realize: Dict[UOp, UOp] = {x.base:None for x in sink.src}
  # mark buffers to be realized
  for p in sink.sparents:
    if p.op is UOps.COPY:
      if (p.src[0].op is not UOps.VIEW or not p.src[0].st.contiguous) and p.op is not UOps.CONTIGUOUS: to_realize[p.src[0]] = None
      to_realize[p] = None
    if p.op is UOps.CONTIGUOUS:
      to_realize[p.src[0]] = None
    # very simple rule
    if p.op is UOps.REDUCE_AXIS:
      to_realize[p] = None
  sink = graph_rewrite(sink, pm_add_buffer, to_realize)
  sink = graph_rewrite(sink, pm_push_views)
  graph_rewrite(sink, break_sched, sched:=[])
  for k,v in to_realize.items():
    if v is None: continue
    realized[k] = v
  ret = []
  for s in sched:
    ast = graph_rewrite(s, enumerate_bufs, bufs:=[])
    # TODO: fix this, COPY order is backward
    if ast.op is UOps.COPY: bufs = bufs[::-1]
    ret.append(ScheduleItem(ast, bufs, None))
  return ret

def create_schedule_with_vars(outs:List[UOp]) -> Tuple[List[ScheduleItem], Dict[UOp, int]]:
  sink = UOp.sink(*[x.base for x in outs])
  sched = _schedule_rewrite(sink)
  return sched, {}

def create_schedule(outs:List[UOp]) -> List[ScheduleItem]:
  schedule, var_vals = create_schedule_with_vars(outs)
  assert len(var_vals) == 0
  return schedule
