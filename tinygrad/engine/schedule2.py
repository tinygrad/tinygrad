from typing import Callable, Dict, List, Tuple, Optional, cast
from dataclasses import dataclass
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes
from tinygrad.engine.lazy import LazyBuffer
from tinygrad.ops import REDUCE_ALU, MetaOps, PatternMatcher, ReduceOps, UOp, UOps, UPat, UnaryOps, graph_rewrite, track_rewrites
from tinygrad.helpers import Metadata
from tinygrad.shape.shapetracker import ShapeTracker

METAOPS = {MetaOps.CUSTOM:UOps.CUSTOM, MetaOps.COPY:UOps.COPY, MetaOps.EMPTY:UOps.EMPTY, MetaOps.VIEW:UOps.BUFFER_VIEW}

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

def _to_uop(x:LazyBuffer, buf_uops:Dict[Buffer, UOp], lbufs:Dict[Buffer, List[LazyBuffer]]) -> UOp:
  if x.buffer in buf_uops: ubuf = buf_uops[x.buffer]
  else:
    buf_uops[x.buffer] = ubuf = UOp(UOps.BUFFER, x.buffer.dtype.ptr(), arg=(len(buf_uops), (x.buffer.device, x.buffer.size, x.buffer.dtype)))
    lbufs.setdefault(x.buffer, []).append(x)
  if x.realized is not None: return ubuf
  src: List[UOp] = []
  for y in x.srcs:
    yv = _to_uop(y.base, buf_uops, lbufs)
    ld = UOp(UOps.LOAD, y.base.dtype, (buf_uops[y.base.buffer], y.base.st.to_uop(), yv))
    src.append(ld if y is y.base else ld.view(y.st))
  if x.op in METAOPS: return UOp(METAOPS[cast(MetaOps, x.op)], x.dtype, (ubuf, *src), x.arg).sink()
  if x.op is MetaOps.CONST: return UOp.const(x.dtype, x.arg)
  if x.op is MetaOps.CONTIGUOUS: val = src[0]
  elif x.op in ReduceOps: val = UOp(UOps.REDUCE_AXIS, x.dtype, tuple(src), (REDUCE_ALU[cast(ReduceOps, x.op)], x.arg))
  elif x.op is UnaryOps.CAST: val = UOp(UOps.CAST, x.dtype, tuple(src))
  elif x.op is UnaryOps.BITCAST: val = UOp(UOps.BITCAST, x.dtype, tuple(src))
  else: val = UOp(UOps.ALU, x.dtype, tuple(src), x.op)
  return UOp(UOps.STORE, dtypes.void, (ubuf, x.st.to_uop(), val)).sink()

def st_fixup(u:UOp, fix:Callable[[ShapeTracker], ShapeTracker], cache:Dict[UOp, UOp]) -> UOp:
  if (n:=cache.get(u)) is not None: return n
  if u.op is UOps.VIEW: return u.replace(arg=fix(u.arg))
  new_srcs = tuple(st_fixup(x, fix, cache) for x in u.src)
  cache[u] = ret = u if new_srcs == u.src else UOp(u.op, u.dtype, new_srcs, u.arg)
  return ret


lazy = PatternMatcher([
  (UPat(UOps.VIEW, src=(UPat.var("x"),), name="view"), lambda x,view:st_fixup(x, lambda st:st+view.arg, {})),
])

break_sched = PatternMatcher([
  (UPat(UOps.SINK, name="x"), lambda ctx,x:ctx.append(x)),
  (UPat(UOps.LOAD, src=(UPat(), UPat.var("st"), UPat.cvar("v"))), lambda _,st,v:UOp(UOps.VALID, dtypes.bool, (st,)).where(v, v.const_like(0))),
  (UPat(UOps.LOAD, src=(UPat(), UPat(), UPat((UOps.BUFFER, UOps.SINK))), name="ld"), lambda _,ld:ld.replace(src=ld.src[:2])),
])

to_ast = PatternMatcher([
  (UPat(UOps.SINK, src=(UPat(tuple(METAOPS.values()), name="x"),)), lambda _,x: x.replace(src=())),
  (UPat(UOps.BUFFER, name="x"), lambda ctx,x: UOp(UOps.DEFINE_GLOBAL, x.dtype, (), ctx.index(x.arg[0]))),
])

@track_rewrites
def _graph(k, outs:List[LazyBuffer]) -> List[ScheduleItem]:
  buf_uops: Dict[Buffer, UOp] = {}
  lbufs: Dict[Buffer, List[LazyBuffer]] = {}
  sink = UOp.sink(*(_to_uop(x.base, buf_uops, lbufs) for x in outs))
  sink = graph_rewrite(sink, lazy)
  kernels: List[UOp] = []
  graph_rewrite(sink, break_sched, kernels)
  ret: List[ScheduleItem] = []
  _bufs = tuple(buf_uops)
  for k in kernels[:-1]:
    ubufs = [x.arg[0] for x in k.parents if x.op is UOps.BUFFER]
    ast = graph_rewrite(k, to_ast, ubufs)
    ret.append(si:=ScheduleItem(ast, tuple(_bufs[x] for x in ubufs), ()))
    for out in si.outputs:
      for lb in lbufs[out]: del lb.srcs
  return ret
